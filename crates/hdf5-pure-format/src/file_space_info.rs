//! The File Space Info message parser and writer.
//!
//! The parser reads versions 0 and 1 of the message and maps version 0 to the fields of version 1.
//! The writer writes version 1. HDF5 1.10.0 is the only release that writes version 0. The
//! addresses of the free-space managers follow the fixed fields only if the message persists free
//! space. The message is defined in "The File Space Info Message" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo

use alloc::vec::Vec;

use hdf5_pure_core::__private::FileSpaceInfoFields;
pub(crate) use hdf5_pure_core::FileSpaceInfo;
pub(crate) use hdf5_pure_core::FileSpacePageSize;
pub(crate) use hdf5_pure_core::FileSpaceStrategy;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::bytes;
use crate::error::FormatError;
use crate::width::FormatWidths;

/// Serializes the body of a version 1 File Space Info message, without the object header message
/// prefix.
///
/// Writes the threshold and the page size at the length width of `widths` and every address at its
/// offset width, and writes the free-space manager addresses only if [`persist`] is set.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if the threshold or the page size does not fit the
/// length width, and [`FormatError::AddressTooLarge`] if an address it writes does not fit the
/// offset width and is not `u64::MAX`, the undefined address.
///
/// [`persist`]: FileSpaceInfo::persist
pub fn serialize_file_space_info(
    widths: FormatWidths,
    info: &FileSpaceInfo,
) -> Result<Vec<u8>, FormatError> {
    let FormatWidths { offsets, lengths } = widths;
    let manager_addrs: &[u64] = if info.persist {
        &info.manager_addrs
    } else {
        &[]
    };
    let mut buf =
        Vec::with_capacity(fixed_len(widths) + manager_addrs.len() * usize::from(offsets.get()));
    buf.push(VERSION_1);
    buf.push(strategy_code(info.strategy));
    buf.push(u8::from(info.persist));
    bytes::try_write_length(&mut buf, info.threshold, lengths)?;
    bytes::try_write_length(&mut buf, info.page_size.get(), lengths)?;
    buf.extend_from_slice(&info.page_end_meta_threshold.to_le_bytes());
    bytes::try_write_offset(&mut buf, info.eoa_pre_fsm, offsets)?;
    for &addr in manager_addrs {
        bytes::try_write_offset(&mut buf, addr, offsets)?;
    }
    Ok(buf)
}

/// Parses the body of a version 0 or 1 File Space Info message into the fields of version 1.
///
/// Reads every length at the length width of `widths` and every address at its offset width. If a
/// version 1 message persists free space, the parser reads as many whole free-space manager
/// addresses as the body holds.
///
/// A version 0 message stores a strategy, a threshold, and, if it persists free space, six
/// free-space manager addresses. The parser maps the strategy code to a strategy and the persist
/// flag, keeps the threshold only for a strategy that uses free-space managers, and takes the end of
/// allocation of a persisting message from `eof_address`, the absolute end-of-file address the
/// superblock stores, less `base`. It sets every other field to its default and the six other
/// manager addresses to the undefined address.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `data` is empty, ends inside the fields of a version 0
/// message, or ends inside the fields before the manager addresses of a version 1 message,
/// [`FormatError::UnsupportedFileSpaceInfoVersion`] if the version is not 0 or 1,
/// [`FormatError::InvalidFileSpaceStrategy`] if the version does not define the strategy code, and
/// [`FormatError::AddressBelowBase`] if a persisting version 0 message is parsed with an
/// `eof_address` below `base`.
pub fn parse_file_space_info(
    widths: FormatWidths,
    base: BaseAddress,
    eof_address: u64,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    match bytes::Fields::new(data, 0).u8()? {
        VERSION_0 => parse_version_0(widths, base, eof_address, data),
        VERSION_1 => parse_version_1(widths, data),
        version => Err(FormatError::UnsupportedFileSpaceInfoVersion(version)),
    }
}

/// Parses a version 0 message into the fields of version 1.
///
/// The C library maps each field in `H5O__fsinfo_decode` and reads the end of allocation of a
/// persisting message from the open file with `H5F_get_eoa` (`H5Ofsinfo.c`, HDF5 2.2.0).
fn parse_version_0(
    widths: FormatWidths,
    base: BaseAddress,
    eof_address: u64,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    let FormatWidths { offsets, lengths } = widths;
    let head_len = 2 + usize::from(lengths.get());
    bytes::ensure_len(data, 0, head_len)?;
    let mut fields = bytes::Fields::new(data, 1);
    let strategy_code = fields.u8()?;
    let threshold = fields.length(lengths)?;
    let undefined = StoredAddress::undefined(offsets.get()).get();
    let defaults = FileSpaceInfoFields {
        strategy: FileSpaceStrategy::FsmAggr,
        persist: false,
        threshold: DEFAULT_THRESHOLD,
        page_size: FileSpacePageSize::DEFAULT,
        page_end_meta_threshold: DEFAULT_PAGE_END_META_THRESHOLD,
        eoa_pre_fsm: undefined,
        manager_addrs: Vec::new(),
    };
    Ok(match strategy_code {
        FILE_SPACE_ALL_PERSIST => {
            bytes::ensure_len(
                data,
                head_len,
                VERSION_0_MANAGERS * usize::from(offsets.get()),
            )?;
            let mut manager_addrs = (0..VERSION_0_MANAGERS)
                .map(|_| fields.address(offsets).map(StoredAddress::get))
                .collect::<Result<Vec<u64>, FormatError>>()?;
            manager_addrs.resize(NUM_FILE_FSM_MANAGERS, undefined);
            FileSpaceInfoFields {
                persist: true,
                threshold,
                eoa_pre_fsm: base.relative(eof_address)?.get(),
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

fn parse_version_1(widths: FormatWidths, data: &[u8]) -> Result<FileSpaceInfo, FormatError> {
    let FormatWidths { offsets, lengths } = widths;
    let fixed = fixed_len(widths);
    let (head, managers) = data
        .split_at_checked(fixed)
        .ok_or(FormatError::UnexpectedEof {
            expected: fixed,
            available: data.len(),
        })?;
    let mut fields = bytes::Fields::new(head, 1);
    let strategy = strategy_from_code(fields.u8()?)?;
    let persist = fields.u8()? != 0;
    let threshold = fields.length(lengths)?;
    let page_size = FileSpacePageSize::try_from(fields.length(lengths)?)?;
    let page_end_meta_threshold = fields.u16()?;
    let eoa_pre_fsm = fields.address(offsets)?.get();

    let manager_addrs = if persist {
        managers
            .chunks_exact(usize::from(offsets.get()))
            .map(|addr| bytes::read_offset_width(addr, 0, offsets))
            .collect::<Result<Vec<u64>, FormatError>>()?
    } else {
        Vec::new()
    };

    Ok(FileSpaceInfoFields {
        strategy,
        persist,
        threshold,
        page_size,
        page_end_meta_threshold,
        eoa_pre_fsm,
        manager_addrs,
    }
    .build())
}

/// Returns the length in bytes of the fields before the free-space manager addresses: the version,
/// the strategy, and the persist flag (1 byte each), the threshold and the page size (one length
/// each), the page-end metadata threshold (2), and the end of allocation (one address).
fn fixed_len(widths: FormatWidths) -> usize {
    3 + 2 * usize::from(widths.lengths.get()) + 2 + usize::from(widths.offsets.get())
}

/// The on-disk numeric code (0–3).
fn strategy_code(strategy: FileSpaceStrategy) -> u8 {
    match strategy {
        FileSpaceStrategy::FsmAggr => 0,
        FileSpaceStrategy::Page => 1,
        FileSpaceStrategy::Aggr => 2,
        FileSpaceStrategy::None => 3,
    }
}

fn strategy_from_code(code: u8) -> Result<FileSpaceStrategy, FormatError> {
    match code {
        0 => Ok(FileSpaceStrategy::FsmAggr),
        1 => Ok(FileSpaceStrategy::Page),
        2 => Ok(FileSpaceStrategy::Aggr),
        3 => Ok(FileSpaceStrategy::None),
        other => Err(FormatError::InvalidFileSpaceStrategy(other)),
    }
}

/// The version of the message that HDF5 1.10.0 writes.
const VERSION_0: u8 = 0;

/// The version of the message that HDF5 1.10.1 and later write, and the one the writer writes.
const VERSION_1: u8 = 1;

// The strategy codes of a version 0 message. The C library names their type
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
/// The C library sets it to `H5F_FILE_SPACE_PGEND_META_THRES` (`H5Fprivate.h`, HDF5 2.2.0).
const DEFAULT_PAGE_END_META_THRESHOLD: u16 = 0;

/// The default free-space section threshold.
///
/// The C library calls this value `H5F_FREE_SPACE_THRESHOLD_DEF`.
pub const DEFAULT_THRESHOLD: u64 = 1;

/// The number of free-space manager addresses a persisting version 1 message stores, the addresses
/// of a small-sized and a large-sized manager for each of six allocation types.
pub const NUM_FILE_FSM_MANAGERS: usize = 12;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddress;

    #[test]
    fn parses_persistent_manager_addresses() {
        // A persisting message: 29-byte head + three 8-byte manager addresses.
        let bytes = serialize_file_space_info(
            widths(8, 8),
            &FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: true,
                threshold: 1,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: 2072,
                manager_addrs: vec![619, u64::MAX, u64::MAX],
            }
            .build(),
        )
        .unwrap();
        assert_eq!(bytes.len(), 29 + 3 * 8);
        let parsed = parse(widths(8, 8), &bytes).unwrap();
        assert_eq!(parsed.manager_addrs, vec![619, u64::MAX, u64::MAX]);
        assert_eq!(parsed.eoa_pre_fsm, 2072);
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
            parse(widths(8, 8), body),
            Err(FormatError::UnsupportedFileSpaceInfoVersion(version))
        );
    }

    #[test]
    fn an_empty_body_returns_unexpected_eof_at_the_version_byte() {
        assert_eq!(
            parse(widths(8, 8), &[]),
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
        version_0_mapped(
            FileSpaceStrategy::FsmAggr,
            true,
            64,
            0x4000,
            [vec![619], vec![u64::MAX; 11]].concat(),
        ),
    )]
    #[case::all(
        0x02,
        &[],
        version_0_mapped(FileSpaceStrategy::FsmAggr, false, 64, u64::MAX, Vec::new()),
    )]
    #[case::aggr_vfd(
        0x03,
        &[],
        version_0_mapped(FileSpaceStrategy::Aggr, false, 1, u64::MAX, Vec::new()),
    )]
    #[case::vfd(
        0x04,
        &[],
        version_0_mapped(FileSpaceStrategy::None, false, 1, u64::MAX, Vec::new()),
    )]
    fn a_version_0_strategy_maps_to_the_fields_of_version_1(
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

        assert_eq!(parse(widths(8, 8), &body), Ok(expected));
    }

    #[test]
    fn a_version_0_message_is_read_at_the_widths_of_its_file() {
        // A 4-byte threshold of 64, then 2-byte manager addresses: 0x0841 and five undefined.
        let body = [
            0x00, 0x01, 0x40, 0x00, 0x00, 0x00, 0x41, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF,
        ];

        assert_eq!(
            parse(widths(2, 4), &body),
            Ok(FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: true,
                threshold: 64,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: 0x4000,
                manager_addrs: [vec![0x0841], vec![0xFFFF; 11]].concat(),
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
            parse(widths(8, 8), &body),
            Ok(version_0_mapped(
                FileSpaceStrategy::FsmAggr,
                false,
                64,
                u64::MAX,
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
            parse(widths(8, 8), &body),
            Err(FormatError::UnexpectedEof {
                expected,
                available: body.len(),
            })
        );
    }

    #[test]
    fn an_end_of_file_below_the_base_fails_only_a_persisting_version_0_message() {
        let all_persist = [[0x00, 0x01].as_slice(), &[0x40; 8], &[0xFF; 48]].concat();
        let version_1 =
            serialize_file_space_info(widths(8, 8), &non_persistent(FileSpaceStrategy::FsmAggr))
                .unwrap();
        let below_base = |data: &[u8]| {
            parse_file_space_info(widths(8, 8), BaseAddress::new(BASE), BASE - 1, data)
        };

        assert_eq!(
            (below_base(&all_persist), below_base(&version_1)),
            (
                Err(FormatError::AddressBelowBase {
                    address: BASE - 1,
                    base: BASE,
                }),
                Ok(non_persistent(FileSpaceStrategy::FsmAggr)),
            )
        );
    }

    #[rstest]
    #[case::default(0x00)]
    #[case::past_vfd(0x05)]
    fn a_version_0_message_with_an_undefined_strategy_returns_an_error(#[case] strategy: u8) {
        let body = [[0x00, strategy].as_slice(), &64u64.to_le_bytes()].concat();

        assert_eq!(
            parse(widths(8, 8), &body),
            Err(FormatError::InvalidFileSpaceStrategy(strategy))
        );
    }

    #[test]
    fn rejects_bad_strategy_code() {
        let mut bytes =
            serialize_file_space_info(widths(8, 8), &non_persistent(FileSpaceStrategy::None))
                .unwrap();
        bytes[1] = 4;
        assert_eq!(
            parse(widths(8, 8), &bytes),
            Err(FormatError::InvalidFileSpaceStrategy(4))
        );
    }

    #[rstest]
    #[case::minimum(512)]
    #[case::maximum(1 << 30)]
    fn a_page_size_at_a_bound_round_trips(#[case] page_size: u64) {
        let mut info = non_persistent(FileSpaceStrategy::FsmAggr);
        info.page_size = FileSpacePageSize::try_from(page_size).unwrap();

        let bytes = serialize_file_space_info(widths(8, 8), &info).unwrap();

        assert_eq!(
            (bytes[11..19].to_vec(), parse(widths(8, 8), &bytes)),
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
            serialize_file_space_info(widths(8, 8), &non_persistent(FileSpaceStrategy::FsmAggr))
                .unwrap();
        bytes[11..19].copy_from_slice(&page_size.to_le_bytes());

        assert_eq!(
            parse(widths(8, 8), &bytes),
            Err(FormatError::InvalidFileSpacePageSize(page_size))
        );
    }

    #[rstest]
    fn a_message_round_trips_at_the_widths_of_its_file(
        #[values(2, 4, 8)] offset_size: u8,
        #[values(2, 4, 8)] length_size: u8,
    ) {
        let undefined = StoredAddress::undefined(offset_size).get();
        let info = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::Page,
            persist: true,
            threshold: 1,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: 0x2000,
            manager_addrs: vec![0x0841, undefined, 0x1806],
        }
        .build();

        let bytes = serialize_file_space_info(widths(offset_size, length_size), &info).unwrap();

        // The version, strategy and persist bytes, the 2-byte page-end threshold, two lengths, and
        // four addresses.
        let (offset_len, length_len) = (usize::from(offset_size), usize::from(length_size));
        assert_eq!(
            bytes.len(),
            5 + 2 * length_len + offset_len + 3 * offset_len
        );
        assert_eq!(parse(widths(offset_size, length_size), &bytes), Ok(info));
    }

    #[test]
    fn a_mixed_width_message_holds_each_field_at_its_width() {
        let info = FileSpaceInfoFields {
            strategy: FileSpaceStrategy::Page,
            persist: true,
            threshold: 1,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: 0x2000,
            manager_addrs: vec![0x0841, 0xFFFF_FFFF, 0x1806],
        }
        .build();

        assert_eq!(
            serialize_file_space_info(widths(4, 2), &info),
            Ok(vec![
                0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x41,
                0x08, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x06, 0x18, 0x00, 0x00,
            ])
        );
    }

    #[test]
    fn a_body_that_ends_inside_the_fixed_fields_returns_unexpected_eof() {
        let truncated = [
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00,
        ];

        assert_eq!(
            parse(widths(4, 2), &truncated),
            Err(FormatError::UnexpectedEof {
                expected: 13,
                available: 12,
            })
        );
    }

    #[rstest]
    #[case::two(2)]
    #[case::four(4)]
    #[case::eight(8)]
    fn the_undefined_end_of_allocation_is_all_ones_at_the_offset_width(#[case] offset_size: u8) {
        let bytes = serialize_file_space_info(
            widths(offset_size, 8),
            &non_persistent(FileSpaceStrategy::FsmAggr),
        )
        .unwrap();

        assert_eq!(bytes[3 + 8 + 8 + 2..], vec![0xFF; usize::from(offset_size)]);
    }

    #[rstest]
    #[case::end_of_allocation(
        |info: &mut FileSpaceInfo| info.eoa_pre_fsm = 0x1_0000_0000,
        FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 },
    )]
    #[case::manager_address(
        |info: &mut FileSpaceInfo| info.manager_addrs[1] = 0x1_0000_0000,
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
            eoa_pre_fsm: 0x2000,
            manager_addrs: vec![0x0841, 0x1806],
        }
        .build();
        widen(&mut info);

        assert_eq!(
            serialize_file_space_info(widths(4, 2), &info),
            Err(expected)
        );
    }

    fn widths(offset_size: u8, length_size: u8) -> FormatWidths {
        FormatWidths::from_sizes(offset_size, length_size).unwrap()
    }

    /// Returns the fields of version 1 that a version 0 message maps to.
    fn version_0_mapped(
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
        eoa_pre_fsm: u64,
        manager_addrs: Vec<u64>,
    ) -> FileSpaceInfo {
        FileSpaceInfoFields {
            strategy,
            persist,
            threshold,
            page_size: FileSpacePageSize::DEFAULT,
            page_end_meta_threshold: 0,
            eoa_pre_fsm,
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
            eoa_pre_fsm: u64::MAX,
            manager_addrs: Vec::new(),
        }
        .build()
    }

    fn parse(widths: FormatWidths, data: &[u8]) -> Result<FileSpaceInfo, FormatError> {
        parse_file_space_info(
            widths,
            BaseAddress::new(BASE),
            BASE + END_OF_ALLOCATION,
            data,
        )
    }

    const BASE: u64 = 0x200;
    const END_OF_ALLOCATION: u64 = 0x4000;
}
