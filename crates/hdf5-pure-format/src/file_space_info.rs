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
use crate::bytes;
use crate::error::FormatError;
use crate::fixed_record;
use crate::fixed_record::{
    BoundDecodedAddress, BoundLength, BoundPageSize, DecodedAddressCodec, FileSpaceStrategyV1Codec,
    FixedRecord, FixedWire, Length2, Length4, Length8, LengthCodec, LengthEncoding, NonzeroBool,
    Offset2, Offset4, Offset8, OffsetEncoding, PageSizeCodec, U16Le,
};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

/// Serializes the body of a version 1 File Space Info message, without the object header message
/// prefix.
///
/// Writes the threshold and the page size at the length width of `widths` and every address at its
/// offset width, and writes the free-space manager addresses only if [`persist`] is set.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedFileSpaceInfoVersion`] if `info.eoa_pre_fsm` is `None`
/// (version 0 state that has not been normalized), [`FormatError::LengthTooLarge`] if the
/// threshold or page size does not fit the length width, [`FormatError::AddressTooLarge`] if an
/// address does not fit the offset width, [`FormatError::DefinedAddressIsUndefined`] if a defined
/// address equals the undefined sentinel value at the offset width, or
/// [`FormatError::InvalidFreeSpaceManager`] if `info.persist` is set and
/// `info.manager_addrs.len() != NUM_FILE_FSM_MANAGERS`.
///
/// [`persist`]: FileSpaceInfo::persist
pub fn serialize_file_space_info_v1(
    FormatWidths { offsets, lengths }: FormatWidths,
    info: &FileSpaceInfo,
) -> Result<Vec<u8>, FormatError> {
    match offsets {
        OffsetWidth::Two => serialize_with_offset::<Offset2>(lengths, info),
        OffsetWidth::Four => serialize_with_offset::<Offset4>(lengths, info),
        OffsetWidth::Eight => serialize_with_offset::<Offset8>(lengths, info),
    }
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
/// Returns [`FormatError::UnexpectedEof`] if `data` is empty, ends inside the fields of a version 0
/// message, or ends inside the fields before the manager addresses of a version 1 message,
/// [`FormatError::UnsupportedFileSpaceInfoVersion`] if the version is not 0 or 1,
/// [`FormatError::InvalidFileSpaceStrategy`] if the version does not define the strategy code, and
/// [`FormatError::InvalidFileSpacePageSize`] if a version 1 page size is outside 512 bytes to 1 GiB.
pub fn parse_file_space_info(
    widths: FormatWidths,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    match bytes::Fields::new(data, 0).u8()? {
        VERSION_0 => parse_version_0(widths, data),
        VERSION_1 => parse_version_1(widths, data),
        version => Err(FormatError::UnsupportedFileSpaceInfoVersion(version)),
    }
}

/// Parses a version 0 message into source-faithful fields.
fn parse_version_0(
    FormatWidths { offsets, lengths }: FormatWidths,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    let head_len = 2 + usize::from(lengths.get());
    bytes::ensure_len(data, 0, head_len)?;
    let mut fields = bytes::Fields::new(data, 1);
    let strategy_code = fields.u8()?;
    let threshold = fields.length(lengths)?;
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
        FILE_SPACE_ALL_PERSIST => {
            bytes::ensure_len(
                data,
                head_len,
                VERSION_0_MANAGERS * usize::from(offsets.get()),
            )?;
            let manager_addrs = (0..VERSION_0_MANAGERS)
                .map(|_| fields.decoded_address(offsets))
                .collect::<Result<Vec<DecodedStoredAddress>, FormatError>>()?;
            FileSpaceInfoFields {
                persist: true,
                threshold,
                manager_addrs,
                ..defaults
            }
        }
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

/// Selects the address encoding before parsing a version 1 message.
///
/// # Errors
///
/// Returns a [`FormatError`] if the message is incomplete or a field is invalid.
fn parse_version_1(
    FormatWidths { offsets, lengths }: FormatWidths,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    match offsets {
        OffsetWidth::Two => parse_with_offset::<Offset2>(lengths, data),
        OffsetWidth::Four => parse_with_offset::<Offset4>(lengths, data),
        OffsetWidth::Eight => parse_with_offset::<Offset8>(lengths, data),
    }
}

/// Selects the length encoding independently of the address encoding.
///
/// # Errors
///
/// Returns a [`FormatError`] if the message is incomplete or a field is invalid.
fn parse_with_offset<O: OffsetEncoding>(
    lengths: LengthWidth,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    match lengths {
        LengthWidth::Two => parse_prefix::<O, Length2>(data),
        LengthWidth::Four => parse_prefix::<O, Length4>(data),
        LengthWidth::Eight => parse_prefix::<O, Length8>(data),
    }
}

/// Parses the fixed fields before reading optional manager addresses.
///
/// Reports incomplete manager addresses relative to the start of that tail.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the prefix or manager addresses are incomplete,
/// [`FormatError::InvalidFileSpaceStrategy`] for an unknown strategy code, or
/// [`FormatError::InvalidFileSpacePageSize`] for a page size outside its range.
fn parse_prefix<O: OffsetEncoding, L: LengthEncoding>(
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    let mut fields = bytes::Fields::new(data, 1);
    let FileSpacePrefixFields {
        strategy,
        persist,
        threshold,
        page_size,
        page_end_meta_threshold,
        eoa_pre_fsm,
    } = FileSpacePrefixCodec::<O, L>::read(&mut fields)?;
    let manager_addrs = if persist {
        let managers = data
            .get(fields.pos()..)
            .ok_or_else(|| FormatError::UnexpectedEof {
                expected: fields.pos(),
                available: data.len(),
            })?;
        bytes::ensure_len(managers, 0, NUM_FILE_FSM_MANAGERS * O::Wire::SIZE)?;
        let mut fields = bytes::Fields::new(managers, 0);
        (0..NUM_FILE_FSM_MANAGERS)
            .map(|_| Ok(DecodedAddressCodec::<O>::read(&mut fields)?.get()))
            .collect::<Result<Vec<DecodedStoredAddress>, FormatError>>()?
    } else {
        Vec::new()
    };
    Ok(FileSpaceInfoFields {
        strategy,
        persist,
        threshold: threshold.get(),
        page_size: page_size.get(),
        page_end_meta_threshold,
        eoa_pre_fsm: Some(eoa_pre_fsm.get()),
        manager_addrs,
    }
    .build())
}

/// Selects the length encoding independently of the address encoding.
///
/// # Errors
///
/// Returns a [`FormatError`] if a field cannot be encoded at the selected widths.
fn serialize_with_offset<O: OffsetEncoding>(
    lengths: LengthWidth,
    info: &FileSpaceInfo,
) -> Result<Vec<u8>, FormatError> {
    match lengths {
        LengthWidth::Two => serialize_prefix::<O, Length2>(info),
        LengthWidth::Four => serialize_prefix::<O, Length4>(info),
        LengthWidth::Eight => serialize_prefix::<O, Length8>(info),
    }
}

/// Prepares the fixed fields and manager addresses before allocating the output.
///
/// # Errors
///
/// Returns the errors of [`FileSpacePrefixFields::new`] for invalid fixed fields, or
/// [`BoundDecodedAddress::new`] for manager addresses that cannot be encoded at this width.
fn serialize_prefix<O: OffsetEncoding, L: LengthEncoding>(
    info: &FileSpaceInfo,
) -> Result<Vec<u8>, FormatError> {
    let prefix = FileSpacePrefixFields::<O, L>::new(info)?;
    let mut managers =
        [BoundDecodedAddress::<O>::new(DecodedStoredAddress::Undefined)?; NUM_FILE_FSM_MANAGERS];
    let managers = if info.persist {
        for (output, &address) in managers.iter_mut().zip(&info.manager_addrs) {
            *output = BoundDecodedAddress::new(address)?;
        }
        managers.as_slice()
    } else {
        &[]
    };
    let mut output =
        Vec::with_capacity(1 + FileSpacePrefixWire::<O, L>::SIZE + managers.len() * O::Wire::SIZE);
    output.push(VERSION_1);
    FileSpacePrefixCodec::<O, L>::append(&mut output, &prefix);
    for address in managers {
        DecodedAddressCodec::<O>::append(&mut output, address);
    }
    Ok(output)
}

// "The File Space Info Message", format specification, version 4.0, defines this version 1
// prefix with independent offset and length widths. The version byte precedes these fields.
fixed_record::fixed_record! {
    /// The fixed fields of a version 1 message, with bounds for their selected widths.
    #[derive(Clone, Copy, Debug)]
    FileSpacePrefixCodec<O: OffsetEncoding, L: LengthEncoding>
        => FileSpacePrefixWire for FileSpacePrefixFields {
        strategy: FileSpaceStrategyV1Codec,
        persist: NonzeroBool,
        threshold: LengthCodec<L>,
        page_size: PageSizeCodec<L>,
        page_end_meta_threshold: U16Le,
        eoa_pre_fsm: DecodedAddressCodec<O>,
    }
}

impl<O: OffsetEncoding, L: LengthEncoding> FileSpacePrefixFields<O, L> {
    /// Prepares the version 1 fields for encoding at the selected widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFileSpaceInfoVersion`] for version 0 state,
    /// [`FormatError::InvalidFreeSpaceManager`] for an incorrect persistent manager count,
    /// [`FormatError::LengthTooLarge`] for a length that exceeds its width,
    /// [`FormatError::AddressTooLarge`] for an address that exceeds its width, or
    /// [`FormatError::DefinedAddressIsUndefined`] for a defined end-of-allocation address equal
    /// to the width's undefined marker. Checks these conditions in this order.
    fn new(info: &FileSpaceInfo) -> Result<Self, FormatError> {
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
        let Some(eoa_pre_fsm) = *eoa_pre_fsm else {
            return Err(FormatError::UnsupportedFileSpaceInfoVersion(VERSION_0));
        };
        if *persist && manager_addrs.len() != NUM_FILE_FSM_MANAGERS {
            return Err(FormatError::InvalidFreeSpaceManager(alloc::format!(
                "a persisting version 1 message requires {} manager addresses, but {} were provided",
                NUM_FILE_FSM_MANAGERS,
                manager_addrs.len()
            )));
        }
        Ok(Self {
            strategy: *strategy,
            persist: *persist,
            threshold: BoundLength::new(*threshold)?,
            page_size: BoundPageSize::new(*page_size)?,
            page_end_meta_threshold: *page_end_meta_threshold,
            eoa_pre_fsm: BoundDecodedAddress::new(eoa_pre_fsm)?,
        })
    }
}

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
    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

    #[test]
    fn parses_persistent_manager_addresses() {
        let bytes = serialize_file_space_info_v1(
            widths(8, 8),
            &FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: true,
                threshold: 1,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: Some(DecodedStoredAddress::Defined(StoredAddress::new(2072))),
                manager_addrs: {
                    let mut addrs = vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS];
                    addrs[0] = DecodedStoredAddress::Defined(StoredAddress::new(619));
                    addrs
                },
            }
            .build(),
        )
        .unwrap();
        assert_eq!(bytes.len(), 29 + NUM_FILE_FSM_MANAGERS * 8);
        let parsed = parse_file_space_info(widths(8, 8), &bytes).unwrap();
        assert_eq!(
            parsed.manager_addrs[0],
            DecodedStoredAddress::Defined(StoredAddress::new(619))
        );
        assert_eq!(parsed.manager_addrs[1], DecodedStoredAddress::Undefined);
        assert_eq!(
            parsed.eoa_pre_fsm,
            Some(DecodedStoredAddress::Defined(StoredAddress::new(2072)))
        );
        assert!(parsed.persist);
    }

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

    #[rstest]
    fn a_message_round_trips_at_the_widths_of_its_file(
        #[values(2, 4, 8)] offset_size: u8,
        #[values(2, 4, 8)] length_size: u8,
    ) {
        let info = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::Page,
            persist: true,
            threshold: 1,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: Some(DecodedStoredAddress::Defined(StoredAddress::new(0x2000))),
            manager_addrs: {
                let mut addrs = vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS];
                addrs[0] = DecodedStoredAddress::Defined(StoredAddress::new(0x0841));
                addrs[2] = DecodedStoredAddress::Defined(StoredAddress::new(0x1806));
                addrs
            },
        }
        .build();

        let bytes = serialize_file_space_info_v1(widths(offset_size, length_size), &info).unwrap();

        let (offset_len, length_len) = (usize::from(offset_size), usize::from(length_size));
        assert_eq!(
            bytes.len(),
            5 + 2 * length_len + offset_len + NUM_FILE_FSM_MANAGERS * offset_len
        );
        assert_eq!(
            parse_file_space_info(widths(offset_size, length_size), &bytes),
            Ok(info)
        );
    }

    #[test]
    fn a_mixed_width_message_holds_each_field_at_its_width() {
        let info = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::Page,
            persist: true,
            threshold: 1,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: Some(DecodedStoredAddress::Defined(StoredAddress::new(0x2000))),
            manager_addrs: {
                let mut addrs = vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS];
                addrs[0] = DecodedStoredAddress::Defined(StoredAddress::new(0x0841));
                addrs[2] = DecodedStoredAddress::Defined(StoredAddress::new(0x1806));
                addrs
            },
        }
        .build();

        let bytes = serialize_file_space_info_v1(widths(4, 2), &info).unwrap();
        assert_eq!(
            &bytes[..25],
            &[
                0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x41,
                0x08, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x06, 0x18, 0x00, 0x00,
            ]
        );
    }

    #[test]
    fn a_body_that_ends_inside_the_fixed_fields_returns_unexpected_eof() {
        let truncated = [
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00,
        ];

        assert_eq!(
            parse_file_space_info(widths(4, 2), &truncated),
            Err(FormatError::UnexpectedEof {
                expected: 13,
                available: 12,
            })
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
    fn named_prefix_preserves_independent_widths_and_unaligned_suffixes(
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
        if persist {
            body.extend_from_slice(&vec![
                0xff;
                NUM_FILE_FSM_MANAGERS * usize::from(offset_size)
            ]);
        }
        let expected = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::None,
            persist,
            threshold: 0x1234,
            page_size: FileSpacePageSize::try_from(513).unwrap(),
            page_end_meta_threshold: 0,
            eoa_pre_fsm: Some(DecodedStoredAddress::Defined(StoredAddress::new(0x42))),
            manager_addrs: if persist {
                vec![DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS]
            } else {
                Vec::new()
            },
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
        assert_eq!(serialize_file_space_info_v1(widths, &expected), Ok(body));
    }

    #[rstest]
    #[case::byte_1(1)]
    #[case::byte_2(2)]
    #[case::byte_3(3)]
    #[case::byte_4(4)]
    #[case::byte_5(5)]
    #[case::byte_6(6)]
    #[case::byte_7(7)]
    #[case::byte_8(8)]
    #[case::byte_9(9)]
    #[case::byte_10(10)]
    #[case::byte_11(11)]
    #[case::byte_12(12)]
    #[case::byte_13(13)]
    #[case::byte_14(14)]
    #[case::byte_15(15)]
    #[case::byte_16(16)]
    #[case::byte_17(17)]
    #[case::byte_18(18)]
    #[case::byte_19(19)]
    #[case::byte_20(20)]
    #[case::byte_21(21)]
    #[case::byte_22(22)]
    #[case::byte_23(23)]
    #[case::byte_24(24)]
    #[case::byte_25(25)]
    #[case::byte_26(26)]
    #[case::byte_27(27)]
    #[case::byte_28(28)]
    fn named_prefix_truncation_reports_complete_extent(#[case] available: usize) {
        let mut body = test_util::file_space_info::non_persistent_version_1(
            test_util::widths::Widths::EIGHT,
            0xff,
            1,
            511,
        );
        body.truncate(available);
        assert_eq!(
            parse_file_space_info(widths(8, 8), &body),
            Err(FormatError::UnexpectedEof {
                expected: 29,
                available,
            })
        );
    }

    #[rstest]
    #[case::strategy(0xff, 511, FormatError::InvalidFileSpaceStrategy(0xff))]
    #[case::page_size(0, 511, FormatError::InvalidFileSpacePageSize(511))]
    #[case::tail(0, 512, FormatError::UnexpectedEof { expected: 96, available: 0 })]
    fn named_prefix_errors_precede_missing_manager_bytes(
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
    fn named_prefix_preparation_preserves_error_precedence(
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
    fn nonpersistent_named_prefix_ignores_unused_manager_addresses() {
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

    #[test]
    fn prepared_named_prefix_retains_width_proofs_after_source_mutation() {
        let mut info = non_persistent(FileSpaceStrategy::Page);
        let prefix = FileSpacePrefixFields::<Offset2, Length2>::new(&info).unwrap();
        info.threshold = u64::MAX;
        info.page_size = FileSpacePageSize::MAX;
        info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(u64::MAX)));
        let mut output = [0xaa; 12];
        assert_eq!(
            FileSpacePrefixCodec::<Offset2, Length2>::write(&mut output, &prefix).unwrap(),
            10
        );
        assert_eq!(output, [1, 0, 1, 0, 0, 0x10, 0, 0, 0xff, 0xff, 0xaa, 0xaa]);
        let mut short = [0xaa; 9];
        assert_eq!(
            FileSpacePrefixCodec::<Offset2, Length2>::write(&mut short, &prefix),
            Err(FormatError::UnexpectedEof {
                expected: 10,
                available: 9
            })
        );
        assert_eq!(short, [0xaa; 9]);
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
