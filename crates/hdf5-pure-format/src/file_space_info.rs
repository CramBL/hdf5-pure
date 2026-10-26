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
use crate::width::FormatWidths;

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
    widths @ FormatWidths { offsets, lengths }: FormatWidths,
    info: &FileSpaceInfo,
) -> Result<Vec<u8>, FormatError> {
    let Some(eoa_pre_fsm) = info.eoa_pre_fsm else {
        return Err(FormatError::UnsupportedFileSpaceInfoVersion(VERSION_0));
    };
    if info.persist && info.manager_addrs.len() != NUM_FILE_FSM_MANAGERS {
        return Err(FormatError::InvalidFreeSpaceManager(alloc::format!(
            "a persisting version 1 message requires {} manager addresses, but {} were provided",
            NUM_FILE_FSM_MANAGERS,
            info.manager_addrs.len()
        )));
    }
    let manager_addrs: &[DecodedStoredAddress] = if info.persist {
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
    bytes::try_write_decoded_address(&mut buf, eoa_pre_fsm, offsets)?;
    for &addr in manager_addrs {
        bytes::try_write_decoded_address(&mut buf, addr, offsets)?;
    }
    Ok(buf)
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

fn parse_version_1(
    widths @ FormatWidths { offsets, lengths }: FormatWidths,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
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
    let eoa_pre_fsm = Some(fields.decoded_address(offsets)?);

    let manager_addrs = if persist {
        let expected_managers_len = NUM_FILE_FSM_MANAGERS * usize::from(offsets.get());
        bytes::ensure_len(managers, 0, expected_managers_len)?;
        let mut mgr_fields = bytes::Fields::new(managers, 0);
        (0..NUM_FILE_FSM_MANAGERS)
            .map(|_| mgr_fields.decoded_address(offsets))
            .collect::<Result<Vec<DecodedStoredAddress>, FormatError>>()?
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
