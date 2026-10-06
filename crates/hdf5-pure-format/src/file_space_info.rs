//! The File Space Info message parser and writer.
//!
//! The parser reads and the writer writes version 1 of the message, which HDF5 1.10.1 and later
//! write. The addresses of the free-space managers follow the fixed fields only if the message
//! persists free space. The message is defined in "The File Space Info Message" of the [format
//! specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo

use alloc::vec::Vec;

use hdf5_pure_core::__private::FileSpaceInfoFields;
pub(crate) use hdf5_pure_core::FileSpaceInfo;
pub(crate) use hdf5_pure_core::FileSpaceStrategy;

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
    bytes::try_write_length(&mut buf, info.page_size, lengths)?;
    buf.extend_from_slice(&info.page_end_meta_threshold.to_le_bytes());
    bytes::try_write_offset(&mut buf, info.eoa_pre_fsm, offsets)?;
    for &addr in manager_addrs {
        bytes::try_write_offset(&mut buf, addr, offsets)?;
    }
    Ok(buf)
}

/// Parses the body of a version 1 File Space Info message.
///
/// Reads the threshold and the page size at the length width of `widths` and every address at its
/// offset width. If the message persists free space, the parser reads as many whole free-space
/// manager addresses as the body holds.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `data` ends inside the fixed fields,
/// [`FormatError::UnsupportedFileSpaceInfoVersion`] if the version is not 1, and
/// [`FormatError::InvalidFileSpaceStrategy`] if the strategy code is above 3.
pub fn parse_file_space_info(
    widths: FormatWidths,
    data: &[u8],
) -> Result<FileSpaceInfo, FormatError> {
    match bytes::Fields::new(data, 0).u8()? {
        VERSION_1 => parse_version_1(widths, data),
        version => Err(FormatError::UnsupportedFileSpaceInfoVersion(version)),
    }
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
    let page_size = fields.length(lengths)?;
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

/// The version of the message that the parser reads and the writer writes.
const VERSION_1: u8 = 1;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddress;

    #[test]
    fn parses_persistent_manager_addresses() {
        // A persisting message: 29-byte head + three 8-byte manager addresses.
        let mut bytes = serialize_file_space_info(
            widths(8, 8),
            &FileSpaceInfoFields {
                strategy: FileSpaceStrategy::FsmAggr,
                persist: true,
                threshold: 1,
                page_size: 4096,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: 2072,
                manager_addrs: vec![619, u64::MAX, u64::MAX],
            }
            .build(),
        )
        .unwrap();
        assert_eq!(bytes.len(), 29 + 3 * 8);
        let parsed = parse_file_space_info(widths(8, 8), &bytes).unwrap();
        assert_eq!(parsed.manager_addrs, vec![619, u64::MAX, u64::MAX]);
        assert_eq!(parsed.eoa_pre_fsm, 2072);
        assert!(parsed.persist);

        // Set the version byte to an unsupported value.
        bytes[0] = 0;
        assert_eq!(
            parse_file_space_info(widths(8, 8), &bytes),
            Err(FormatError::UnsupportedFileSpaceInfoVersion(0))
        );
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

    #[test]
    fn rejects_bad_strategy_code() {
        let mut bytes =
            serialize_file_space_info(widths(8, 8), &non_persistent(FileSpaceStrategy::None))
                .unwrap();
        bytes[1] = 4;
        assert_eq!(
            parse_file_space_info(widths(8, 8), &bytes),
            Err(FormatError::InvalidFileSpaceStrategy(4))
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
            page_size: 4096,
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
            page_size: 4096,
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
            parse_file_space_info(widths(4, 2), &truncated),
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
        |info: &mut FileSpaceInfo| info.page_size = 0x1_0000,
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
            page_size: 4096,
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

    fn non_persistent(strategy: FileSpaceStrategy) -> FileSpaceInfo {
        FileSpaceInfoFields {
            strategy,
            persist: false,
            threshold: 1,
            page_size: 4096,
            page_end_meta_threshold: 0,
            eoa_pre_fsm: u64::MAX,
            manager_addrs: Vec::new(),
        }
        .build()
    }
}
