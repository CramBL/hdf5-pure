//! The File Space Info message parser and writer.
//!
//! The parser reads versions 0 and 1 of the message into source-faithful fields, and
//! [`normalize_file_space_info_to_v1`] maps version 0 to the fields of version 1.
//! The writer writes version 1. HDF5 1.10.0 is the only release that writes version 0. The
//! addresses of the free-space managers follow the fixed fields only if the message persists free
//! space. The message is defined in "The File Space Info Message" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo

use alloc::vec::Vec;

use hdf5_pure_core::__private::FileSpaceInfoFields;
use hdf5_pure_core::FileSpaceInfo;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;

use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::error::FormatError;
use crate::width;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Serializes the body of a version 1 File Space Info message, without the object header message
/// prefix.
///
/// Writes the threshold and the page size at the length width of `widths` and every address at its
/// offset width, and writes the free-space manager addresses only if [`persist`] is set.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedFileSpaceInfoVersion`] if `info.eoa_pre_fsm` is `None`
/// (version 0 state that has not been normalized), [`FormatError::InvalidFreeSpaceManager`] if
/// `info.persist` is set and `info.manager_addrs.len() != NUM_FILE_FSM_MANAGERS`,
/// [`FormatError::LengthTooLarge`] if the threshold or page size does not fit the length width,
/// [`FormatError::AddressTooLarge`] if an address does not fit the offset width, or
/// [`FormatError::DefinedAddressIsUndefined`] if a defined
/// address equals the undefined sentinel value at the offset width.
///
/// [`persist`]: FileSpaceInfo::persist
pub fn serialize_file_space_info_v1(
    widths: FormatWidths,
    info: &FileSpaceInfo,
) -> Result<Vec<u8>, FormatError> {
    Ok(PreparedV1::new(widths, info)?.encode()?)
}

/// Normalizes a parsed or constructed [`FileSpaceInfo`] into version 1 format.
///
/// If `info` is already a version 1 message (`eoa_pre_fsm.is_some()`), returns `info` unchanged.
/// If `info` is a version 0 message (`eoa_pre_fsm.is_none()`):
/// - when `persist` is true: sets `eoa_pre_fsm` to `Some(DecodedStoredAddress::Defined(eoa))`
///   and pads the 6 version 0 manager addresses to 12 with [`DecodedStoredAddress::Undefined`].
/// - when `persist` is false: sets `eoa_pre_fsm` to `Some(DecodedStoredAddress::Undefined)`
///   and keeps 0 manager addresses.
pub fn normalize_file_space_info_to_v1(info: &FileSpaceInfo, eoa: StoredAddress) -> FileSpaceInfo {
    if info.eoa_pre_fsm.is_some() {
        return info.clone();
    }
    let (eoa_pre_fsm, manager_addrs) = if info.persist {
        let mut addrs = info.manager_addrs.clone();
        addrs.resize(NUM_FILE_FSM_MANAGERS, DecodedStoredAddress::Undefined);
        (Some(DecodedStoredAddress::Defined(eoa)), addrs)
    } else {
        (Some(DecodedStoredAddress::Undefined), Vec::new())
    };
    FileSpaceInfoFields {
        strategy: info.strategy,
        persist: info.persist,
        threshold: info.threshold,
        page_size: info.page_size,
        page_end_meta_threshold: info.page_end_meta_threshold,
        eoa_pre_fsm,
        manager_addrs,
    }
    .build()
}

/// Parses the body of a version 0 or 1 File Space Info message.
///
/// Reads every length at the length width of `widths` and every address at its offset width.
///
/// A version 0 message stores a strategy, a threshold, and, if it persists free space, six
/// free-space manager addresses. The parser returns it with `eoa_pre_fsm: None`.
///
/// A version 1 message stores an end-of-allocation address and, if it persists free space, twelve
/// manager addresses. The parser returns each address decoded into a [`DecodedStoredAddress`].
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the message is incomplete,
/// [`FormatError::UnsupportedFileSpaceInfoVersion`] if the version is not 0 or 1,
/// [`FormatError::InvalidFileSpaceStrategy`] if the version does not define the strategy code, and
/// [`FormatError::InvalidFileSpacePageSize`] if a version 1 page size is outside 512 bytes to 1 GiB.
/// EOF coordinates refer to the original body, except for a version 1 manager list, whose
/// coordinates refer to the start of that list.
pub fn parse_file_space_info(
    widths: FormatWidths,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    match bytes::ByteCursor::new(data).u8()? {
        VERSION_0 => parse_version_0(widths, data),
        VERSION_1 => parse_version_1(widths, data),
        version => Err(FormatError::UnsupportedFileSpaceInfoVersion(version)),
    }
}

/// Parses a version 0 message into source-faithful fields.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the prefix or manager list is incomplete, with
/// coordinates relative to the original body, or [`FormatError::InvalidFileSpaceStrategy`] if
/// the strategy code is undefined.
fn parse_version_0(
    FormatWidths { offsets, lengths }: FormatWidths,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    let mut cursor = bytes::ByteCursor::new(data);
    let head = cursor.take(VERSION_0_FIXED_BYTES + usize::from(lengths.get()))?;
    let mut prefix = bytes::ByteCursor::with_position(head, 1)?;
    let strategy_code = prefix.u8()?;
    let threshold = prefix.length(lengths)?;
    let defaults = FileSpaceInfoFields {
        strategy: FileSpaceStrategy::FsmAggr,
        persist: false,
        threshold: DEFAULT_THRESHOLD,
        page_size: FileSpacePageSize::DEFAULT,
        page_end_meta_threshold: DEFAULT_PAGE_END_META_THRESHOLD,
        eoa_pre_fsm: None,
        manager_addrs: Vec::new(),
    };
    Ok(match strategy_code {
        FILE_SPACE_ALL_PERSIST => FileSpaceInfoFields {
            persist: true,
            threshold,
            manager_addrs: read_managers(&mut cursor, offsets, VERSION_0_MANAGERS)?,
            ..defaults
        },
        FILE_SPACE_ALL => FileSpaceInfoFields {
            threshold,
            ..defaults
        },
        FILE_SPACE_AGGR_VFD => FileSpaceInfoFields {
            strategy: FileSpaceStrategy::Aggr,
            ..defaults
        },
        FILE_SPACE_VFD => FileSpaceInfoFields {
            strategy: FileSpaceStrategy::None,
            ..defaults
        },
        other => return Err(FormatError::InvalidFileSpaceStrategy(other)),
    }
    .build())
}

/// Parses the fixed prefix before reading optional manager addresses.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the prefix or manager list is incomplete,
/// [`FormatError::InvalidFileSpaceStrategy`] if the strategy code is undefined, or
/// [`FormatError::InvalidFileSpacePageSize`] if the page size is outside its range.
/// Reports prefix errors relative to the original body and manager EOF relative to the list.
fn parse_version_1(widths: FormatWidths, data: &[u8]) -> Result<FileSpaceInfo, FormatError> {
    let FormatWidths { offsets, lengths } = widths;
    let mut cursor = bytes::ByteCursor::new(data);
    let head = cursor.take(v1_prefix_len(widths))?;
    let mut prefix = bytes::ByteCursor::with_position(head, 1)?;
    let strategy = decode_strategy_v1(prefix.u8()?)?;
    let persist = prefix.u8()? != 0;
    let threshold = prefix.length(lengths)?;
    let page_size = FileSpacePageSize::try_from(prefix.length(lengths)?)?;
    let page_end_meta_threshold = prefix.u16()?;
    let eoa_pre_fsm = prefix.decoded_address(offsets)?;
    let manager_addrs = if persist {
        read_managers(
            &mut bytes::ByteCursor::new(cursor.remaining()),
            offsets,
            NUM_FILE_FSM_MANAGERS,
        )?
    } else {
        Vec::new()
    };
    Ok(FileSpaceInfoFields {
        strategy,
        persist,
        threshold,
        page_size,
        page_end_meta_threshold,
        eoa_pre_fsm: Some(eoa_pre_fsm),
        manager_addrs,
    }
    .build())
}

/// Acquires the complete manager list before decoding its addresses.
///
/// Decodes the width's all-ones address as [`DecodedStoredAddress::Undefined`].
///
/// # Errors
///
/// Returns the errors of [`bytes::ByteCursor::take_arrays`], leaving the cursor unchanged.
fn read_managers(
    cursor: &mut bytes::ByteCursor<'_>,
    offsets: OffsetWidth,
    count: usize,
) -> Result<Vec<DecodedStoredAddress>, bytes::UnexpectedEof> {
    let undefined = width::offset_width_max(offsets);
    let decoded = |bits| {
        if bits == undefined {
            DecodedStoredAddress::Undefined
        } else {
            DecodedStoredAddress::Defined(StoredAddress::new(bits))
        }
    };
    Ok(match offsets {
        OffsetWidth::Two => cursor
            .take_arrays::<2>(count)?
            .iter()
            .map(|&raw| decoded(u64::from(u16::from_le_bytes(raw))))
            .collect(),
        OffsetWidth::Four => cursor
            .take_arrays::<4>(count)?
            .iter()
            .map(|&raw| decoded(u64::from(u32::from_le_bytes(raw))))
            .collect(),
        OffsetWidth::Eight => cursor
            .take_arrays::<8>(count)?
            .iter()
            .map(|&raw| decoded(u64::from_le_bytes(raw)))
            .collect(),
    })
}

/// A version 1 message borrowed for encoding at checked field widths.
///
/// The encoder reads the checked fields through an immutable borrow. Persistent messages have
/// exactly [`NUM_FILE_FSM_MANAGERS`] manager addresses.
struct PreparedV1<'a> {
    widths: FormatWidths,
    info: &'a FileSpaceInfo,
    /// The end-of-allocation address converted at the retained offset width.
    eoa: StoredAddress,
    /// The manager addresses when free space is persisted.
    managers: Option<&'a [DecodedStoredAddress; NUM_FILE_FSM_MANAGERS]>,
}

impl<'a> PreparedV1<'a> {
    /// Borrows version 1 fields after checking their shape and encoding widths.
    ///
    /// Ignores unused manager addresses when persistence is disabled.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFileSpaceInfoVersion`] if the end-of-allocation address
    /// is absent, [`FormatError::InvalidFreeSpaceManager`] if a persistent manager list has a
    /// length other than [`NUM_FILE_FSM_MANAGERS`], [`FormatError::LengthTooLarge`] if the
    /// threshold or page size exceeds the length width, [`FormatError::AddressTooLarge`] if an
    /// address exceeds the address width, or [`FormatError::DefinedAddressIsUndefined`] if a
    /// defined address equals that width's undefined marker. Checks the version, manager count,
    /// threshold, page size, end-of-allocation address, and manager addresses in that order.
    fn new(widths: FormatWidths, info: &'a FileSpaceInfo) -> Result<Self, FormatError> {
        let FormatWidths { offsets, lengths } = widths;
        let FileSpaceInfo {
            persist,
            threshold,
            page_size,
            eoa_pre_fsm,
            manager_addrs,
            ..
        } = info;
        let Some(eoa_pre_fsm) = *eoa_pre_fsm else {
            return Err(FormatError::UnsupportedFileSpaceInfoVersion(VERSION_0));
        };
        let managers: Option<&[DecodedStoredAddress; NUM_FILE_FSM_MANAGERS]> = if *persist {
            let Ok(managers) = manager_addrs.as_slice().try_into() else {
                return Err(invalid_manager_count(manager_addrs.len()));
            };
            Some(managers)
        } else {
            None
        };
        bytes::require_length(*threshold, lengths)?;
        // `FileSpacePageSize` is at most 1 GiB, so four- and eight-byte lengths always fit.
        if lengths == LengthWidth::Two {
            bytes::require_length(page_size.get(), lengths)?;
        }
        let eoa = eoa_pre_fsm.to_stored(offsets)?;
        if let Some(managers) = managers {
            for &address in managers {
                address.to_stored(offsets)?;
            }
        }
        Ok(Self {
            widths,
            info,
            eoa,
            managers,
        })
    }

    /// Writes the checked fields into one result buffer.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if the manager-address destination cannot be acquired.
    fn encode(self) -> Result<Vec<u8>, bytes::UnexpectedEof> {
        let Self {
            widths,
            info,
            eoa,
            managers,
        } = self;
        let FormatWidths { offsets, lengths } = widths;
        let FileSpaceInfo {
            strategy,
            persist,
            threshold,
            page_size,
            page_end_meta_threshold,
            ..
        } = info;
        let manager_count = managers.map_or(0, |values| values.len());
        let capacity = v1_prefix_len(widths)
            .saturating_add(manager_count.saturating_mul(usize::from(offsets.get())));
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(&[VERSION_1, encode_strategy_v1(*strategy), u8::from(*persist)]);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the threshold and page size fit the length width retained by the prepared message"
        )]
        match lengths {
            LengthWidth::Two => {
                output.extend_from_slice(&(*threshold as u16).to_le_bytes());
                output.extend_from_slice(&(page_size.get() as u16).to_le_bytes());
            }
            LengthWidth::Four => {
                output.extend_from_slice(&(*threshold as u32).to_le_bytes());
                output.extend_from_slice(&(page_size.get() as u32).to_le_bytes());
            }
            LengthWidth::Eight => {
                output.extend_from_slice(&threshold.to_le_bytes());
                output.extend_from_slice(&page_size.get().to_le_bytes());
            }
        }
        output.extend_from_slice(&page_end_meta_threshold.to_le_bytes());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "defined addresses fit the retained offset width and truncation selects its undefined marker"
        )]
        match offsets {
            OffsetWidth::Two => {
                output.extend_from_slice(&(eoa.get() as u16).to_le_bytes());
                if let Some(managers) = managers {
                    let start = output.len();
                    output.resize(start.saturating_add(2 * NUM_FILE_FSM_MANAGERS), 0);
                    let tail =
                        bytes::array_at_mut::<{ 2 * NUM_FILE_FSM_MANAGERS }>(&mut output, start)?;
                    let (records, _) = tail.as_chunks_mut::<2>();
                    for (record, &address) in records.iter_mut().zip(managers) {
                        *record = (prepared_address_bits(address) as u16).to_le_bytes();
                    }
                }
            }
            OffsetWidth::Four => {
                output.extend_from_slice(&(eoa.get() as u32).to_le_bytes());
                if let Some(managers) = managers {
                    let start = output.len();
                    output.resize(start.saturating_add(4 * NUM_FILE_FSM_MANAGERS), 0);
                    let tail =
                        bytes::array_at_mut::<{ 4 * NUM_FILE_FSM_MANAGERS }>(&mut output, start)?;
                    let (records, _) = tail.as_chunks_mut::<4>();
                    for (record, &address) in records.iter_mut().zip(managers) {
                        *record = (prepared_address_bits(address) as u32).to_le_bytes();
                    }
                }
            }
            OffsetWidth::Eight => {
                output.extend_from_slice(&eoa.get().to_le_bytes());
                if let Some(managers) = managers {
                    let start = output.len();
                    output.resize(start.saturating_add(8 * NUM_FILE_FSM_MANAGERS), 0);
                    let tail =
                        bytes::array_at_mut::<{ 8 * NUM_FILE_FSM_MANAGERS }>(&mut output, start)?;
                    let (records, _) = tail.as_chunks_mut::<8>();
                    for (record, &address) in records.iter_mut().zip(managers) {
                        *record = prepared_address_bits(address).to_le_bytes();
                    }
                }
            }
        }
        Ok(output)
    }
}

/// Returns address bits, using all ones for an undefined address.
///
/// The encoder writes the low bytes at the offset width checked by [`PreparedV1::new`].
fn prepared_address_bits(address: DecodedStoredAddress) -> u64 {
    match address {
        DecodedStoredAddress::Undefined => u64::MAX,
        DecodedStoredAddress::Defined(address) => address.get(),
    }
}

/// Returns the version 1 body size before optional manager addresses.
fn v1_prefix_len(FormatWidths { offsets, lengths }: FormatWidths) -> usize {
    VERSION_1_FIXED_BYTES + 2 * usize::from(lengths.get()) + usize::from(offsets.get())
}

/// Parses a version 1 strategy code.
///
/// # Errors
///
/// Returns [`FormatError::InvalidFileSpaceStrategy`] if `code` is not 0, 1, 2, or 3.
fn decode_strategy_v1(code: u8) -> Result<FileSpaceStrategy, FormatError> {
    match code {
        FSM_AGGR => Ok(FileSpaceStrategy::FsmAggr),
        PAGE => Ok(FileSpaceStrategy::Page),
        AGGR => Ok(FileSpaceStrategy::Aggr),
        NONE => Ok(FileSpaceStrategy::None),
        other => Err(FormatError::InvalidFileSpaceStrategy(other)),
    }
}

/// Returns the version 1 code for a file-space strategy.
fn encode_strategy_v1(strategy: FileSpaceStrategy) -> u8 {
    match strategy {
        FileSpaceStrategy::FsmAggr => FSM_AGGR,
        FileSpaceStrategy::Page => PAGE,
        FileSpaceStrategy::Aggr => AGGR,
        FileSpaceStrategy::None => NONE,
    }
}

/// Reports an incorrect persistent manager count for version 1 encoding.
fn invalid_manager_count(count: usize) -> FormatError {
    FormatError::InvalidFreeSpaceManager(alloc::format!(
        "a persisting version 1 message requires {} manager addresses, but {} were provided",
        NUM_FILE_FSM_MANAGERS,
        count
    ))
}

// `H5O__fsinfo_decode` in `H5Ofsinfo.c`, HDF5 2.2.0, reads these fixed prefix bytes and strategy
// codes.
/// The version and strategy bytes in a version 0 prefix.
const VERSION_0_FIXED_BYTES: usize = 2;
/// The bytes in a version 1 prefix other than its address and length fields.
const VERSION_1_FIXED_BYTES: usize = 5;
const FSM_AGGR: u8 = 0;
const PAGE: u8 = 1;
const AGGR: u8 = 2;
const NONE: u8 = 3;

/// The version of the message that HDF5 1.10.0 writes.
const VERSION_0: u8 = 0;

/// The version of the message that HDF5 1.10.1 and later write, and the one the writer writes.
const VERSION_1: u8 = 1;

// The strategy codes of a version 0 message. libhdf5 names their type
// `H5F_file_space_type_t`.
const FILE_SPACE_ALL_PERSIST: u8 = 1;
const FILE_SPACE_ALL: u8 = 2;
const FILE_SPACE_AGGR_VFD: u8 = 3;
const FILE_SPACE_VFD: u8 = 4;

/// The number of free-space manager addresses a persisting version 0 message stores, one per
/// allocation type from `H5FD_MEM_SUPER` to `H5FD_MEM_OHDR`.
const VERSION_0_MANAGERS: usize = 6;

/// The page-end metadata threshold of a version 0 message mapped to version 1.
///
/// libhdf5 sets it to `H5F_FILE_SPACE_PGEND_META_THRES` (`H5Fprivate.h`, HDF5 2.2.0).
const DEFAULT_PAGE_END_META_THRESHOLD: u16 = 0;

/// The default free-space section threshold.
///
/// libhdf5 calls this value `H5F_FREE_SPACE_THRESHOLD_DEF`.
pub const DEFAULT_THRESHOLD: u64 = 1;

/// The number of free-space manager addresses a persisting version 1 message stores, the addresses
/// of a small-sized and a large-sized manager for each of six allocation types.
pub const NUM_FILE_FSM_MANAGERS: usize = 12;

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::test_runner::FileFailurePersistence;
    use rstest::rstest;
    use test_util::bytes as fixture_bytes;

    use super::*;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

    #[rstest]
    #[case::shorter_than_version_1(&[0x02, 0x00], 2)]
    #[case::only_a_version(&[0xFF], 0xFF)]
    fn an_unsupported_version_is_reported_whatever_the_length_of_its_body(
        #[case] body: &[u8],
        #[case] version: u8,
    ) {
        assert_eq!(
            parse_file_space_info(widths(8, 8), body),
            Err(FormatError::UnsupportedFileSpaceInfoVersion(version))
        );
    }

    #[test]
    fn an_empty_body_returns_unexpected_eof_at_the_version_byte() {
        assert_eq!(
            parse_file_space_info(widths(8, 8), &[]),
            Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            })
        );
    }

    #[rstest]
    #[case::all_persist(
        0x01,
        &[619, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX],
        version_0_raw(
            FileSpaceStrategy::FsmAggr,
            true,
            64,
            vec![
                DecodedStoredAddress::Defined(StoredAddress::new(619)),
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
            ],
        ),
    )]
    #[case::all(
        0x02,
        &[],
        version_0_raw(FileSpaceStrategy::FsmAggr, false, 64, Vec::new()),
    )]
    #[case::aggr_vfd(
        0x03,
        &[],
        version_0_raw(FileSpaceStrategy::Aggr, false, 1, Vec::new()),
    )]
    #[case::vfd(
        0x04,
        &[],
        version_0_raw(FileSpaceStrategy::None, false, 1, Vec::new()),
    )]
    fn a_version_0_strategy_parses_without_an_end_of_allocation_address(
        #[case] strategy: u8,
        #[case] manager_addrs: &[u64],
        #[case] expected: FileSpaceInfo,
    ) {
        let body = [
            vec![0x00, strategy],
            64u64.to_le_bytes().to_vec(),
            manager_addrs
                .iter()
                .flat_map(|addr| addr.to_le_bytes())
                .collect(),
        ]
        .concat();

        assert_eq!(parse_file_space_info(widths(8, 8), &body), Ok(expected));
    }

    #[test]
    fn normalizing_version_0_sets_end_of_allocation_and_twelve_manager_slots() {
        let raw_persistent = version_0_raw(
            FileSpaceStrategy::FsmAggr,
            true,
            64,
            vec![
                DecodedStoredAddress::Defined(StoredAddress::new(619)),
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
            ],
        );
        let normalized =
            normalize_file_space_info_to_v1(&raw_persistent, StoredAddress::new(0x4000));
        assert_eq!(
            normalized.eoa_pre_fsm,
            Some(DecodedStoredAddress::Defined(StoredAddress::new(0x4000)))
        );
        assert_eq!(normalized.manager_addrs.len(), 12);
        assert_eq!(
            normalized.manager_addrs[0],
            DecodedStoredAddress::Defined(StoredAddress::new(619))
        );
        assert!(
            normalized.manager_addrs[1..]
                .iter()
                .all(|a| *a == DecodedStoredAddress::Undefined)
        );

        let raw_non_persistent = version_0_raw(FileSpaceStrategy::FsmAggr, false, 64, Vec::new());
        let norm_non_persistent =
            normalize_file_space_info_to_v1(&raw_non_persistent, StoredAddress::new(0x4000));
        assert_eq!(
            norm_non_persistent.eoa_pre_fsm,
            Some(DecodedStoredAddress::Undefined)
        );
        assert!(norm_non_persistent.manager_addrs.is_empty());
    }

    #[test]
    fn a_version_0_message_is_read_at_the_widths_of_its_file() {
        // A 4-byte threshold of 64, then 2-byte manager addresses: 0x0841 and five undefined.
        let body = [
            0x00, 0x01, 0x40, 0x00, 0x00, 0x00, 0x41, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF,
        ];

        assert_eq!(
            parse_file_space_info(widths(2, 4), &body),
            Ok(FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: true,
                threshold: 64,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: None,
                manager_addrs: vec![
                    DecodedStoredAddress::Defined(StoredAddress::new(0x0841)),
                    DecodedStoredAddress::Undefined,
                    DecodedStoredAddress::Undefined,
                    DecodedStoredAddress::Undefined,
                    DecodedStoredAddress::Undefined,
                    DecodedStoredAddress::Undefined,
                ],
            }
            .build())
        );
    }

    #[test]
    fn the_bytes_after_a_version_0_message_s_fields_are_ignored() {
        let body = [
            0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];

        assert_eq!(
            parse_file_space_info(widths(8, 8), &body),
            Ok(version_0_raw(
                FileSpaceStrategy::FsmAggr,
                false,
                64,
                Vec::new()
            ))
        );
    }

    #[rstest]
    #[case::only_a_version(vec![0x00], 10)]
    #[case::inside_the_threshold(vec![0x00, 0x02, 0x40, 0x00, 0x00], 10)]
    #[case::inside_the_manager_addresses([vec![0x00, 0x01], vec![0x40; 8], vec![0xFF; 45]].concat(), 58)]
    fn a_version_0_message_that_ends_inside_its_fields_returns_unexpected_eof(
        #[case] body: Vec<u8>,
        #[case] expected: usize,
    ) {
        assert_eq!(
            parse_file_space_info(widths(8, 8), &body),
            Err(FormatError::UnexpectedEof {
                expected,
                available: body.len(),
            })
        );
    }

    #[rstest]
    #[case::default(0x00)]
    #[case::past_vfd(0x05)]
    fn a_version_0_message_with_an_undefined_strategy_returns_an_error(#[case] strategy: u8) {
        let body = [[0x00, strategy].as_slice(), &64u64.to_le_bytes()].concat();

        assert_eq!(
            parse_file_space_info(widths(8, 8), &body),
            Err(FormatError::InvalidFileSpaceStrategy(strategy))
        );
    }

    #[test]
    fn rejects_bad_strategy_code() {
        let mut bytes =
            serialize_file_space_info_v1(widths(8, 8), &non_persistent(FileSpaceStrategy::None))
                .unwrap();
        bytes[1] = 4;
        assert_eq!(
            parse_file_space_info(widths(8, 8), &bytes),
            Err(FormatError::InvalidFileSpaceStrategy(4))
        );
    }

    #[rstest]
    #[case::minimum(512)]
    #[case::maximum(1 << 30)]
    fn a_page_size_at_a_bound_round_trips(#[case] page_size: u64) {
        let mut info = non_persistent(FileSpaceStrategy::FsmAggr);
        info.page_size = FileSpacePageSize::try_from(page_size).unwrap();

        let bytes = serialize_file_space_info_v1(widths(8, 8), &info).unwrap();

        assert_eq!(
            (
                bytes[11..19].to_vec(),
                parse_file_space_info(widths(8, 8), &bytes)
            ),
            (page_size.to_le_bytes().to_vec(), Ok(info))
        );
    }

    #[rstest]
    #[case::zero(0)]
    #[case::below_the_minimum(511)]
    #[case::above_the_maximum((1 << 30) + 1)]
    #[case::two_gib(2 << 30)]
    fn a_page_size_outside_512_bytes_to_1_gib_fails_the_parse(#[case] page_size: u64) {
        let mut bytes =
            serialize_file_space_info_v1(widths(8, 8), &non_persistent(FileSpaceStrategy::FsmAggr))
                .unwrap();
        bytes[11..19].copy_from_slice(&page_size.to_le_bytes());

        assert_eq!(
            parse_file_space_info(widths(8, 8), &bytes),
            Err(FormatError::InvalidFileSpacePageSize(page_size))
        );
    }

    #[test]
    fn a_body_with_fewer_than_twelve_managers_returns_unexpected_eof() {
        let mut bytes = serialize_file_space_info_v1(
            widths(8, 8),
            &FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: true,
                threshold: 1,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: Some(DecodedStoredAddress::Undefined),
                manager_addrs: vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS],
            }
            .build(),
        )
        .unwrap();
        bytes.truncate(bytes.len() - 8);
        assert_eq!(
            parse_file_space_info(widths(8, 8), &bytes),
            Err(FormatError::UnexpectedEof {
                expected: 12 * 8,
                available: 11 * 8,
            })
        );
    }

    #[rstest]
    #[case::two(2)]
    #[case::four(4)]
    #[case::eight(8)]
    fn the_undefined_end_of_allocation_is_all_ones_at_the_offset_width(#[case] offset_size: u8) {
        let bytes = serialize_file_space_info_v1(
            widths(offset_size, 8),
            &non_persistent(FileSpaceStrategy::FsmAggr),
        )
        .unwrap();

        assert_eq!(bytes[3 + 8 + 8 + 2..], vec![0xFF; usize::from(offset_size)]);
    }

    #[rstest]
    #[case::end_of_allocation(
        |info: &mut FileSpaceInfo| info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(0x1_0000_0000))),
        FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 },
    )]
    #[case::manager_address(
        |info: &mut FileSpaceInfo| info.manager_addrs[1] = DecodedStoredAddress::Defined(StoredAddress::new(0x1_0000_0000)),
        FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 },
    )]
    #[case::page_size(
        |info: &mut FileSpaceInfo| info.page_size = FileSpacePageSize::try_from(0x1_0000).unwrap(),
        FormatError::LengthTooLarge { length: 0x1_0000, length_size: 2 },
    )]
    fn a_value_wider_than_its_field_returns_an_error(
        #[case] widen: fn(&mut FileSpaceInfo),
        #[case] expected: FormatError,
    ) {
        let mut info = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::FsmAggr,
            persist: true,
            threshold: 1,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: Some(DecodedStoredAddress::Defined(StoredAddress::new(0x2000))),
            manager_addrs: vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS],
        }
        .build();
        widen(&mut info);

        assert_eq!(
            serialize_file_space_info_v1(widths(4, 2), &info),
            Err(expected)
        );
    }

    #[test]
    fn serializing_unnormalized_version_0_returns_unsupported_version() {
        let info = version_0_raw(FileSpaceStrategy::FsmAggr, false, 1, Vec::new());
        assert_eq!(
            serialize_file_space_info_v1(widths(8, 8), &info),
            Err(FormatError::UnsupportedFileSpaceInfoVersion(0))
        );
    }

    #[rstest]
    #[case::two_eoa(OffsetWidth::Two, true, 0xFFFF)]
    #[case::two_manager(OffsetWidth::Two, false, 0xFFFF)]
    #[case::four_eoa(OffsetWidth::Four, true, 0xFFFF_FFFF)]
    #[case::four_manager(OffsetWidth::Four, false, 0xFFFF_FFFF)]
    #[case::eight_eoa(OffsetWidth::Eight, true, u64::MAX)]
    #[case::eight_manager(OffsetWidth::Eight, false, u64::MAX)]
    fn serializing_file_space_info_with_defined_sentinel_collision_fails(
        #[case] offset_width: OffsetWidth,
        #[case] in_eoa: bool,
        #[case] sentinel: u64,
    ) {
        let mut info = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::FsmAggr,
            persist: true,
            threshold: 1,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: Some(DecodedStoredAddress::Undefined),
            manager_addrs: vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS],
        }
        .build();

        if in_eoa {
            info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(sentinel)));
        } else {
            info.manager_addrs[0] = DecodedStoredAddress::Defined(StoredAddress::new(sentinel));
        }

        let widths = FormatWidths {
            offsets: offset_width,
            lengths: LengthWidth::Eight,
        };

        assert_eq!(
            serialize_file_space_info_v1(widths, &info),
            Err(FormatError::DefinedAddressIsUndefined {
                address: sentinel,
                offset_size: offset_width.get(),
            })
        );
    }

    #[rstest]
    fn version_1_fields_preserve_independent_widths_and_unaligned_suffixes(
        #[values(2, 4, 8)] offset_size: u8,
        #[values(2, 4, 8)] length_size: u8,
        #[values(false, true)] persist: bool,
    ) {
        let widths = widths(offset_size, length_size);
        let fixture_widths =
            test_util::widths::Widths::new(usize::from(offset_size), usize::from(length_size));
        let mut body =
            test_util::file_space_info::non_persistent_version_1(fixture_widths, 3, 0x1234, 513);
        body[2] = if persist { 0x80 } else { 0 };
        let fixed = body.len();
        let eoa = fixed - usize::from(offset_size);
        body[eoa..].fill(0);
        body[fixed - usize::from(offset_size)] = 0x42;
        body[eoa - 2..eoa].copy_from_slice(&[0xcd, 0xab]);
        let managers = if persist {
            let mut managers = vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS];
            managers[0] = DecodedStoredAddress::Defined(StoredAddress::new(0x0841));
            managers[2] = DecodedStoredAddress::Defined(StoredAddress::new(0x1806));
            for address in &managers {
                fixture_bytes::push_address(
                    &mut body,
                    address_bits(*address),
                    usize::from(offset_size),
                );
            }
            managers
        } else {
            Vec::new()
        };
        let expected = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::None,
            persist,
            threshold: 0x1234,
            page_size: FileSpacePageSize::try_from(513).unwrap(),
            page_end_meta_threshold: 0xabcd,
            eoa_pre_fsm: Some(DecodedStoredAddress::Defined(StoredAddress::new(0x42))),
            manager_addrs: managers,
        }
        .build();
        let mut unaligned = Aligned([0xbb; 136]);
        unaligned.0[1..1 + body.len()].copy_from_slice(&body);
        assert_eq!(unaligned.0[1..].as_ptr().addr() % 8, 1);
        assert_eq!(
            parse_file_space_info(widths, &unaligned.0[1..body.len() + 3]),
            Ok(expected.clone())
        );
        body[2] = u8::from(persist);
        if offset_size == 4 && length_size == 2 && persist {
            assert_eq!(
                &body[..25],
                &[
                    1, 3, 1, 0x34, 0x12, 1, 2, 0xcd, 0xab, 0x42, 0, 0, 0, 0x41, 8, 0, 0, 0xff,
                    0xff, 0xff, 0xff, 6, 0x18, 0, 0,
                ]
            );
        }
        assert_eq!(serialize_file_space_info_v1(widths, &expected), Ok(body));
    }

    #[rstest]
    fn version_1_prefix_truncation_reports_complete_extent(
        #[values(2, 4, 8)] offset_size: u8,
        #[values(2, 4, 8)] length_size: u8,
    ) {
        let body = test_util::file_space_info::non_persistent_version_1(
            test_util::widths::Widths::new(usize::from(offset_size), usize::from(length_size)),
            0xff,
            1,
            511,
        );
        for available in 1..body.len() {
            assert_eq!(
                parse_file_space_info(widths(offset_size, length_size), &body[..available]),
                Err(FormatError::UnexpectedEof {
                    expected: body.len(),
                    available
                }),
            );
        }
    }

    #[rstest]
    #[case::version_0(VERSION_0, 6)]
    #[case::version_1(VERSION_1, 12)]
    fn incomplete_manager_lists_preserve_version_specific_eof_coordinates(
        #[case] version: u8,
        #[case] count: usize,
        #[values(2, 4, 8)] offset_size: u8,
        #[values(2, 4, 8)] length_size: u8,
    ) {
        let fixture_widths =
            test_util::widths::Widths::new(usize::from(offset_size), usize::from(length_size));
        let mut body = if version == VERSION_0 {
            test_util::file_space_info::version_0(
                fixture_widths,
                test_util::file_space_info::ALL_PERSIST,
                0x1234,
                &[],
            )
        } else {
            let mut body = test_util::file_space_info::non_persistent_version_1(
                fixture_widths,
                test_util::file_space_info::PAGE,
                0x1234,
                513,
            );
            body[2] = 0x80;
            body
        };
        let prefix_len = body.len();
        for index in 0..count {
            fixture_bytes::push_uint(&mut body, index as u64, usize::from(offset_size));
        }
        let mut expected = if version == VERSION_0 {
            version_0_raw(FileSpaceStrategy::FsmAggr, true, 0x1234, Vec::new())
        } else {
            let mut info = non_persistent(FileSpaceStrategy::Page);
            info.persist = true;
            info.threshold = 0x1234;
            info.page_size = FileSpacePageSize::try_from(513).unwrap();
            info
        };
        expected.manager_addrs = (0..count)
            .map(|index| DecodedStoredAddress::Defined(StoredAddress::new(index as u64)))
            .collect();
        assert_eq!(
            parse_file_space_info(widths(offset_size, length_size), &body),
            Ok(expected)
        );
        for available in prefix_len..body.len() {
            let origin = if version == VERSION_0 { 0 } else { prefix_len };
            assert_eq!(
                parse_file_space_info(widths(offset_size, length_size), &body[..available]),
                Err(FormatError::UnexpectedEof {
                    expected: body.len() - origin,
                    available: available - origin,
                })
            );
        }
    }

    #[rstest]
    #[case::strategy(0xff, 511, FormatError::InvalidFileSpaceStrategy(0xff))]
    #[case::page_size(0, 511, FormatError::InvalidFileSpacePageSize(511))]
    #[case::tail(0, 512, FormatError::UnexpectedEof { expected: 96, available: 0 })]
    fn version_1_prefix_errors_precede_missing_manager_bytes(
        #[case] strategy: u8,
        #[case] page_size: u64,
        #[case] expected: FormatError,
    ) {
        let mut body = test_util::file_space_info::non_persistent_version_1(
            test_util::widths::Widths::EIGHT,
            strategy,
            1,
            page_size,
        );
        body[2] = 0xff;
        assert_eq!(parse_file_space_info(widths(8, 8), &body), Err(expected));
    }

    #[rstest]
    #[case::version(|info: &mut FileSpaceInfo| {
        info.eoa_pre_fsm = None;
        info.manager_addrs.clear();
        info.threshold = u64::MAX;
    }, FormatError::UnsupportedFileSpaceInfoVersion(0))]
    #[case::manager_count(|info: &mut FileSpaceInfo| {
        info.manager_addrs.clear();
        info.threshold = u64::MAX;
    }, FormatError::InvalidFreeSpaceManager(
        "a persisting version 1 message requires 12 manager addresses, but 0 were provided".into()
    )) ]
    #[case::excess_manager_count(|info: &mut FileSpaceInfo| {
        info.manager_addrs.push(DecodedStoredAddress::Undefined);
        info.threshold = u64::MAX;
    }, FormatError::InvalidFreeSpaceManager(
        "a persisting version 1 message requires 12 manager addresses, but 13 were provided".into()
    )) ]
    #[case::threshold(|info: &mut FileSpaceInfo| {
        info.threshold = u64::MAX;
        info.page_size = FileSpacePageSize::MAX;
        info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(u64::MAX)));
    }, FormatError::LengthTooLarge { length: u64::MAX, length_size: 2 })]
    #[case::page_size(|info: &mut FileSpaceInfo| {
        info.page_size = FileSpacePageSize::MAX;
        info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(u64::MAX)));
    }, FormatError::LengthTooLarge { length: 1 << 30, length_size: 2 })]
    #[case::end_of_allocation(|info: &mut FileSpaceInfo| {
        info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(0xffff)));
        info.manager_addrs[0] = DecodedStoredAddress::Defined(StoredAddress::new(u64::MAX));
    }, FormatError::DefinedAddressIsUndefined { address: 0xffff, offset_size: 2 })]
    fn version_1_preparation_preserves_error_precedence(
        #[case] change: fn(&mut FileSpaceInfo),
        #[case] expected: FormatError,
    ) {
        let mut info = non_persistent(FileSpaceStrategy::Aggr);
        info.persist = true;
        info.manager_addrs = vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS];
        change(&mut info);
        assert_eq!(
            serialize_file_space_info_v1(widths(2, 2), &info),
            Err(expected)
        );
    }

    #[test]
    fn nonpersistent_version_1_ignores_unused_manager_addresses() {
        let mut info = non_persistent(FileSpaceStrategy::None);
        info.manager_addrs = vec![DecodedStoredAddress::Defined(StoredAddress::new(u64::MAX))];
        let expected = test_util::file_space_info::non_persistent_version_1(
            test_util::widths::Widths::new(2, 2),
            3,
            1,
            4096,
        );
        assert_eq!(
            serialize_file_space_info_v1(widths(2, 2), &info),
            Ok(expected)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128,
            failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
                env!("CARGO_MANIFEST_DIR"), "/tests/proptest-regressions/file_space_info.txt"
            )))),
            ..ProptestConfig::default()
        })]

        #[test]
        fn version_1_values_match_independent_bytes(
            (widths, info) in version_1_values(),
            prefix in proptest::collection::vec(any::<u8>(), 0..16),
            suffix in proptest::collection::vec(any::<u8>(), 0..16),
        ) {
            let expected = reference_version_1(widths, &info);
            prop_assert_eq!(&serialize_file_space_info_v1(widths, &info).unwrap(), &expected);
            let input = [prefix.as_slice(), expected.as_slice(), suffix.as_slice()].concat();
            prop_assert_eq!(parse_file_space_info(widths, &input[prefix.len()..]).unwrap(), info);
        }


    }

    fn version_1_values() -> impl Strategy<Value = (FormatWidths, FileSpaceInfo)> {
        (
            proptest::sample::select(vec![
                OffsetWidth::Two,
                OffsetWidth::Four,
                OffsetWidth::Eight,
            ]),
            proptest::sample::select(vec![
                LengthWidth::Two,
                LengthWidth::Four,
                LengthWidth::Eight,
            ]),
        )
            .prop_flat_map(|(offsets, lengths)| {
                let maximum_address = match offsets {
                    OffsetWidth::Two => u64::from(u16::MAX),
                    OffsetWidth::Four => u64::from(u32::MAX),
                    OffsetWidth::Eight => u64::MAX,
                };
                let maximum_length = match lengths {
                    LengthWidth::Two => u64::from(u16::MAX),
                    LengthWidth::Four => u64::from(u32::MAX),
                    LengthWidth::Eight => u64::MAX,
                };
                let maximum_page = maximum_length.min(1 << 30);
                let address = prop_oneof![
                    Just(DecodedStoredAddress::Undefined),
                    Just(DecodedStoredAddress::Defined(StoredAddress::new(0))),
                    Just(DecodedStoredAddress::Defined(StoredAddress::new(
                        maximum_address - 1
                    ))),
                    (0..maximum_address)
                        .prop_map(|value| DecodedStoredAddress::Defined(StoredAddress::new(value))),
                ];
                (
                    proptest::sample::select(vec![
                        FileSpaceStrategy::FsmAggr,
                        FileSpaceStrategy::Page,
                        FileSpaceStrategy::Aggr,
                        FileSpaceStrategy::None,
                    ]),
                    any::<bool>(),
                    prop_oneof![Just(0), Just(maximum_length), 0..=maximum_length],
                    prop_oneof![Just(512), Just(513), Just(maximum_page), 512..=maximum_page],
                    1u16..=u16::MAX,
                    address.clone(),
                    proptest::array::uniform12(address),
                )
                    .prop_map(
                        move |(
                            strategy,
                            persist,
                            threshold,
                            page_size,
                            page_end_meta_threshold,
                            eoa,
                            managers,
                        )| {
                            (
                                FormatWidths { offsets, lengths },
                                FileSpaceInfoFields {
                                    strategy,
                                    persist,
                                    threshold,
                                    page_size: FileSpacePageSize::try_from(page_size).unwrap(),
                                    page_end_meta_threshold,
                                    eoa_pre_fsm: Some(eoa),
                                    manager_addrs: if persist {
                                        managers.to_vec()
                                    } else {
                                        Vec::new()
                                    },
                                }
                                .build(),
                            )
                        },
                    )
            })
    }

    fn reference_version_1(
        FormatWidths { offsets, lengths }: FormatWidths,
        info: &FileSpaceInfo,
    ) -> Vec<u8> {
        let FileSpaceInfo {
            strategy,
            persist,
            threshold,
            page_size,
            page_end_meta_threshold,
            eoa_pre_fsm,
            manager_addrs,
            ..
        } = info;
        let strategy = match strategy {
            FileSpaceStrategy::FsmAggr => 0,
            FileSpaceStrategy::Page => 1,
            FileSpaceStrategy::Aggr => 2,
            FileSpaceStrategy::None => 3,
        };
        let mut bytes = vec![1, strategy, u8::from(*persist)];
        fixture_bytes::push_uint(&mut bytes, *threshold, usize::from(lengths.get()));
        fixture_bytes::push_uint(&mut bytes, page_size.get(), usize::from(lengths.get()));
        bytes.extend_from_slice(&page_end_meta_threshold.to_le_bytes());
        fixture_bytes::push_address(
            &mut bytes,
            address_bits(eoa_pre_fsm.unwrap()),
            usize::from(offsets.get()),
        );
        if *persist {
            for address in manager_addrs {
                fixture_bytes::push_address(
                    &mut bytes,
                    address_bits(*address),
                    usize::from(offsets.get()),
                );
            }
        }
        bytes
    }

    fn address_bits(address: DecodedStoredAddress) -> Option<u64> {
        match address {
            DecodedStoredAddress::Defined(address) => Some(address.get()),
            DecodedStoredAddress::Undefined => None,
        }
    }

    #[repr(align(8))]
    struct Aligned([u8; 136]);

    fn widths(offset_size: u8, length_size: u8) -> FormatWidths {
        FormatWidths::from_sizes(offset_size, length_size).unwrap()
    }

    fn version_0_raw(
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
        manager_addrs: Vec<DecodedStoredAddress>,
    ) -> FileSpaceInfo {
        FileSpaceInfoFields {
            strategy,
            persist,
            threshold,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: None,
            manager_addrs,
        }
        .build()
    }

    fn non_persistent(strategy: FileSpaceStrategy) -> FileSpaceInfo {
        FileSpaceInfoFields {
            strategy,
            persist: false,
            threshold: 1,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: Some(DecodedStoredAddress::Undefined),
            manager_addrs: Vec::new(),
        }
        .build()
    }
}
