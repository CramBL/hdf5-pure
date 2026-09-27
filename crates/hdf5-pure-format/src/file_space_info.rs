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
pub use hdf5_pure_core::FileSpaceInfo;
pub use hdf5_pure_core::FileSpaceStrategy;

use crate::error::FormatError;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Serializes the body of a version 1 File Space Info message, without the object header message
/// prefix.
///
/// Writes every length and address 8 bytes wide, and the free-space manager addresses only if
/// [`persist`] is set.
///
/// [`persist`]: FileSpaceInfo::persist
pub fn serialize_file_space_info(info: &FileSpaceInfo) -> Vec<u8> {
    let mut buf = Vec::with_capacity(29 + info.manager_addrs.len() * 8);
    buf.push(1); // version
    buf.push(strategy_code(info.strategy));
    buf.push(info.persist as u8);
    buf.extend_from_slice(&info.threshold.to_le_bytes());
    buf.extend_from_slice(&info.page_size.to_le_bytes());
    buf.extend_from_slice(&info.page_end_meta_threshold.to_le_bytes());
    buf.extend_from_slice(&info.eoa_pre_fsm.to_le_bytes());
    if info.persist {
        for &addr in &info.manager_addrs {
            buf.extend_from_slice(&addr.to_le_bytes());
        }
    }
    buf
}

/// Parses the body of a version 1 File Space Info message.
///
/// `offset_size` and `length_size` are the widths the superblock stores. If the message persists
/// free space, the parser reads as many free-space manager addresses as the body holds.
///
/// # Errors
///
/// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is
/// not 2, 4, or 8, [`FormatError::UnexpectedEof`] if `data` ends inside the fixed fields,
/// [`FormatError::UnsupportedFileSpaceInfoVersion`] if the version is not 1, and
/// [`FormatError::InvalidFileSpaceStrategy`] if the strategy code is above 3.
pub fn parse_file_space_info(
    data: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<FileSpaceInfo, FormatError> {
    let os = usize::from(OffsetWidth::try_from(offset_size)?.get());
    let ls = usize::from(LengthWidth::try_from(length_size)?.get());
    // version(1) + strategy(1) + persist(1) + threshold(ls) + page_size(ls)
    // + page_end(2) + eoa(os)
    let fixed = 3 + ls + ls + 2 + os;
    if data.len() < fixed {
        return Err(FormatError::UnexpectedEof {
            expected: fixed,
            available: data.len(),
        });
    }
    let version = data[0];
    if version != 1 {
        return Err(FormatError::UnsupportedFileSpaceInfoVersion(version));
    }
    let strategy = strategy_from_code(data[1])?;
    let persist = data[2] != 0;
    let mut pos = 3;
    let threshold = read_uint_le(&data[pos..pos + ls]);
    pos += ls;
    let page_size = read_uint_le(&data[pos..pos + ls]);
    pos += ls;
    let page_end_meta_threshold = u16::from_le_bytes([data[pos], data[pos + 1]]);
    pos += 2;
    let eoa_pre_fsm = read_uint_le(&data[pos..pos + os]);
    pos += os;

    let mut manager_addrs = Vec::new();
    if persist {
        while pos + os <= data.len() {
            manager_addrs.push(read_uint_le(&data[pos..pos + os]));
            pos += os;
        }
    }

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

/// Read a little-endian unsigned integer of 1–8 bytes into a `u64`.
fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn parses_persistent_manager_addresses() {
        // A persisting message: 29-byte head + three 8-byte manager addresses.
        let mut bytes = serialize_file_space_info(
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
        );
        assert_eq!(bytes.len(), 29 + 3 * 8);
        let parsed = parse_file_space_info(&bytes, 8, 8).unwrap();
        assert_eq!(parsed.manager_addrs, vec![619, u64::MAX, u64::MAX]);
        assert_eq!(parsed.eoa_pre_fsm, 2072);
        assert!(parsed.persist);

        // Set the version byte to an unsupported value.
        bytes[0] = 0;
        assert!(matches!(
            parse_file_space_info(&bytes, 8, 8),
            Err(FormatError::UnsupportedFileSpaceInfoVersion(0))
        ));
    }

    #[test]
    fn rejects_bad_strategy_code() {
        let mut bytes = serialize_file_space_info(&non_persistent(FileSpaceStrategy::None));
        bytes[1] = 4;
        assert_eq!(
            parse_file_space_info(&bytes, 8, 8),
            Err(FormatError::InvalidFileSpaceStrategy(4))
        );
    }

    #[rstest]
    #[case::offset_size(0, 8, FormatError::InvalidOffsetSize(0))]
    #[case::length_size(8, 16, FormatError::InvalidLengthSize(16))]
    fn a_width_the_superblock_cannot_hold_returns_its_width_error(
        #[case] offset_size: u8,
        #[case] length_size: u8,
        #[case] expected: FormatError,
    ) {
        let bytes = serialize_file_space_info(&non_persistent(FileSpaceStrategy::FsmAggr));

        assert_eq!(
            parse_file_space_info(&bytes, offset_size, length_size),
            Err(expected)
        );
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
