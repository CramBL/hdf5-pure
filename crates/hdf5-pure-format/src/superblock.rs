//! The superblock parser and writer.
//!
//! The superblock is defined in "Format Signature and Superblock" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super

use alloc::format;
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use hdf5_pure_core::__private::SuperblockFields;
pub use hdf5_pure_core::Superblock;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;
use crate::signature::HDF5_SIGNATURE;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Serializes `superblock` in the version 2 and 3 layout.
///
/// Writes the two width bytes, the consistency flags, the four addresses at the width
/// `offset_size` gives, and the Jenkins lookup3 checksum over the bytes before it. An absent
/// superblock extension address is written as the undefined address.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedVersion`] if the version is not 2 or 3,
/// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not 2, 4,
/// or 8, and [`FormatError::Internal`] if the consistency flags do not fit their 1-byte field or an
/// address does not fit the offset width.
pub fn serialize_superblock(superblock: &Superblock) -> Result<Vec<u8>, FormatError> {
    if !matches!(superblock.version, 2 | 3) {
        return Err(FormatError::UnsupportedVersion(superblock.version));
    }
    let offset_width = OffsetWidth::try_from(superblock.offset_size)?;
    let length_width = LengthWidth::try_from(superblock.length_size)?;
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(&HDF5_SIGNATURE);
    buf.push(superblock.version);
    buf.push(offset_width.get());
    buf.push(length_width.get());
    buf.push(superblock.consistency_flags.narrow_or_else(|| {
        FormatError::Internal(format!(
            "consistency flags {:#x} do not fit the 1-byte field of a version {} superblock",
            superblock.consistency_flags, superblock.version
        ))
    })?);
    write_address(
        &mut buf,
        offset_width,
        "base",
        superblock.base_address.get(),
    )?;
    write_address(
        &mut buf,
        offset_width,
        "superblock extension",
        superblock.superblock_extension_address.unwrap_or(u64::MAX),
    )?;
    write_address(
        &mut buf,
        offset_width,
        "end-of-file",
        superblock.eof_address,
    )?;
    write_address(
        &mut buf,
        offset_width,
        "root group object header",
        superblock.root_group_address,
    )?;
    // checksum
    let checksum = crate::checksum::jenkins_lookup3(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    Ok(buf)
}

/// Parses the superblock whose signature starts at `signature_offset` in `source`.
///
/// Reads one window of at most 128 bytes from `source`, enough for the largest superblock.
///
/// # Errors
///
/// Returns the error `source` returns if the read fails, and otherwise the errors
/// [`parse_superblock`] returns.
pub fn parse_superblock_from_source<S: MetadataSource + ?Sized>(
    source: &S,
    signature_offset: u64,
) -> Result<Superblock, FormatError> {
    let available = source.len().saturating_sub(signature_offset);
    let window = available.min(MAX_SUPERBLOCK_LEN).to_usize()?;
    let buf = source.read_metadata_at(signature_offset, window)?;
    // The window begins at the signature, so within `buf` it sits at 0.
    parse_superblock(&buf, 0)
}

/// Parses the superblock whose signature starts at `signature_offset` in `data`.
///
/// Reads superblock versions 0 to 3.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `data` ends inside the superblock,
/// [`FormatError::SignatureNotFound`] if the signature is not at `signature_offset`,
/// [`FormatError::UnsupportedVersion`] if the version is above 3,
/// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not 2, 4,
/// or 8, and, under the `checksum` feature, [`FormatError::ChecksumMismatch`] if the checksum of a
/// version 2 or 3 superblock does not match its bytes.
pub fn parse_superblock(data: &[u8], signature_offset: usize) -> Result<Superblock, FormatError> {
    let d = data
        .get(signature_offset..)
        .ok_or(FormatError::UnexpectedEof {
            expected: signature_offset,
            available: data.len(),
        })?;
    bytes::ensure_len(d, 0, 9)?; // signature(8) + version(1)

    // Verify signature
    if d[..8] != HDF5_SIGNATURE {
        return Err(FormatError::SignatureNotFound);
    }

    let version = d[8];
    match version {
        0 => parse_v0(d),
        1 => parse_v1(d),
        2 | 3 => parse_v2v3(d, version),
        v => Err(FormatError::UnsupportedVersion(v)),
    }
}

fn parse_v0(d: &[u8]) -> Result<Superblock, FormatError> {
    // sig(8) + version(1) + free_space_ver(1) + root_grp_ver(1) + reserved(1)
    // + shared_hdr_ver(1) + offset_size(1) + length_size(1) + reserved(1)
    // + group_leaf_k(2) + group_internal_k(2) + consistency_flags(4)
    // = 24 bytes before variable-sized fields
    bytes::ensure_len(d, 0, 24)?;

    let offset_width = OffsetWidth::try_from(d[13])?;
    let length_width = LengthWidth::try_from(d[14])?;

    let group_leaf_node_k = LittleEndian::read_u16(&d[16..18]);
    let group_internal_node_k = LittleEndian::read_u16(&d[18..20]);
    let consistency_flags = LittleEndian::read_u32(&d[20..24]);

    let os = usize::from(offset_width.get());
    let ls = usize::from(length_width.get());
    // 4 addresses + root symbol table entry
    let var_start = 24;
    let sym_entry_size = ls + os + 4 + 4 + 16;
    let total = var_start + 4 * os + sym_entry_size;
    bytes::ensure_len(d, 0, total)?;

    let mut pos = var_start;
    let base_address = BaseAddress::new(bytes::read_offset_width(d, pos, offset_width)?);
    pos += os;
    let free_space_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;
    let eof_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;
    let driver_info_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;

    // Root symbol table entry
    let _link_name_offset = bytes::read_length_width(d, pos, length_width)?;
    pos += ls;
    let object_header_addr = bytes::read_offset_width(d, pos, offset_width)?;

    Ok(SuperblockFields {
        version: 0,
        offset_size: offset_width.get(),
        length_size: length_width.get(),
        base_address,
        eof_address,
        root_group_address: object_header_addr,
        group_leaf_node_k: Some(group_leaf_node_k),
        group_internal_node_k: Some(group_internal_node_k),
        indexed_storage_internal_node_k: None,
        free_space_address: Some(free_space_address),
        driver_info_address: Some(driver_info_address),
        consistency_flags,
        superblock_extension_address: None,
        checksum: None,
    }
    .build())
}

fn parse_v1(d: &[u8]) -> Result<Superblock, FormatError> {
    // Same as v0 but adds indexed_storage_internal_node_k(2) + reserved(2)
    // *after* the consistency flags, not before them, the order the C
    // library decodes (`H5F__sblock_deserialize`: symbol-table leaf K,
    // B-tree internal K, status flags, chunk B-tree K, reserved).
    // sig(8) + version(1) + free_space_ver(1) + root_grp_ver(1) + reserved(1)
    // + shared_hdr_ver(1) + offset_size(1) + length_size(1) + reserved(1)
    // + group_leaf_k(2) + group_internal_k(2) + consistency_flags(4)
    // + indexed_storage_k(2) + reserved(2) = 28
    bytes::ensure_len(d, 0, 28)?;

    let offset_width = OffsetWidth::try_from(d[13])?;
    let length_width = LengthWidth::try_from(d[14])?;

    let group_leaf_node_k = LittleEndian::read_u16(&d[16..18]);
    let group_internal_node_k = LittleEndian::read_u16(&d[18..20]);
    let consistency_flags = LittleEndian::read_u32(&d[20..24]);
    let indexed_storage_internal_node_k = LittleEndian::read_u16(&d[24..26]);
    // d[26..28] reserved

    let os = usize::from(offset_width.get());
    let ls = usize::from(length_width.get());
    let var_start = 28;
    let sym_entry_size = ls + os + 4 + 4 + 16;
    let total = var_start + 4 * os + sym_entry_size;
    bytes::ensure_len(d, 0, total)?;

    let mut pos = var_start;
    let base_address = BaseAddress::new(bytes::read_offset_width(d, pos, offset_width)?);
    pos += os;
    let free_space_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;
    let eof_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;
    let driver_info_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;

    // Root symbol table entry
    let _link_name_offset = bytes::read_length_width(d, pos, length_width)?;
    pos += ls;
    let object_header_addr = bytes::read_offset_width(d, pos, offset_width)?;

    Ok(SuperblockFields {
        version: 1,
        offset_size: offset_width.get(),
        length_size: length_width.get(),
        base_address,
        eof_address,
        root_group_address: object_header_addr,
        group_leaf_node_k: Some(group_leaf_node_k),
        group_internal_node_k: Some(group_internal_node_k),
        indexed_storage_internal_node_k: Some(indexed_storage_internal_node_k),
        free_space_address: Some(free_space_address),
        driver_info_address: Some(driver_info_address),
        consistency_flags,
        superblock_extension_address: None,
        checksum: None,
    }
    .build())
}

fn parse_v2v3(d: &[u8], version: u8) -> Result<Superblock, FormatError> {
    // sig(8) + version(1) + offset_size(1) + length_size(1) + consistency_flags(1) = 12
    bytes::ensure_len(d, 0, 12)?;

    let offset_width = OffsetWidth::try_from(d[9])?;
    let length_width = LengthWidth::try_from(d[10])?;
    let consistency_flags = d[11] as u32;

    let os = usize::from(offset_width.get());
    // 4 addresses + checksum(4)
    let total = 12 + 4 * os + 4;
    bytes::ensure_len(d, 0, total)?;

    let mut pos = 12;
    let base_address = BaseAddress::new(bytes::read_offset_width(d, pos, offset_width)?);
    pos += os;
    let superblock_extension_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;
    let eof_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;
    let root_group_address = bytes::read_offset_width(d, pos, offset_width)?;
    pos += os;

    let stored_checksum = LittleEndian::read_u32(&d[pos..pos + 4]);

    // Validate checksum if feature enabled
    #[cfg(feature = "checksum")]
    {
        let computed = crate::checksum::jenkins_lookup3(&d[..pos]);
        if computed != stored_checksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_checksum,
                computed,
            });
        }
    }

    Ok(SuperblockFields {
        version,
        offset_size: offset_width.get(),
        length_size: length_width.get(),
        base_address,
        eof_address,
        root_group_address,
        group_leaf_node_k: None,
        group_internal_node_k: None,
        indexed_storage_internal_node_k: None,
        free_space_address: None,
        driver_info_address: None,
        consistency_flags,
        superblock_extension_address: Some(superblock_extension_address),
        checksum: Some(stored_checksum),
    }
    .build())
}

fn write_address(
    buf: &mut Vec<u8>,
    width: OffsetWidth,
    field: &str,
    address: u64,
) -> Result<(), FormatError> {
    if !width.holds(address) {
        return Err(FormatError::Internal(format!(
            "the superblock's {field} address {address:#x} does not fit a {}-byte field",
            width.get()
        )));
    }
    bytes::write_offset(buf, address, width);
    Ok(())
}

/// Upper bound on the on-disk size of a superblock across all versions: the
/// largest is v1 with 8-byte offsets at 100 bytes (28 prefix + 4 addresses +
/// a 40-byte root symbol-table entry). 128 leaves headroom while staying a
/// tiny, fixed window to pull from a streaming source.
const MAX_SUPERBLOCK_LEN: u64 = 128;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use test_util::superblock::{v0, v2};
    use test_util::widths::Widths;

    /// A version 0 superblock whose end-of-file and root header addresses
    /// differ, so a read taken from the neighbouring field is visible.
    fn build_v0_bytes(offset_size: usize) -> Vec<u8> {
        // The free-space and driver-info addresses are left undefined, which
        // is every bit set at whatever width the superblock declares.
        v0::Superblock::new(Widths::new(offset_size, offset_size))
            .eof_address(4096)
            .root_group(0, 96)
            .build()
    }

    /// The same for version 1, whose indexed-storage K directly follows the
    /// consistency flags, so a parser that swaps their offsets reads one as the
    /// other.
    fn build_v1_bytes(offset_size: usize) -> Vec<u8> {
        v0::Superblock::new(Widths::new(offset_size, offset_size))
            .version_1()
            .consistency_flags(1)
            .indexed_storage_internal_node_k(32)
            .eof_address(8192)
            .root_group(0, 200)
            .build()
    }

    fn build_v2_bytes(
        offset_width: OffsetWidth,
        length_width: LengthWidth,
        version: u8,
    ) -> Vec<u8> {
        v2::Superblock::new(Widths::new(
            usize::from(offset_width.get()),
            usize::from(length_width.get()),
        ))
        .version(version)
        .eof_address(2048)
        .root_header_address(48)
        .build()
    }

    #[test]
    fn parse_v0_8byte_offsets() {
        let data = build_v0_bytes(8);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.base_address, BaseAddress::ZERO);
        assert_eq!(sb.eof_address, 4096);
        assert_eq!(sb.root_group_address, 96);
        assert_eq!(sb.group_leaf_node_k, Some(4));
        assert_eq!(sb.group_internal_node_k, Some(16));
        assert_eq!(sb.indexed_storage_internal_node_k, None);
        assert_eq!(sb.free_space_address, Some(0xFFFFFFFFFFFFFFFF));
        assert_eq!(sb.driver_info_address, Some(0xFFFFFFFFFFFFFFFF));
        assert_eq!(sb.checksum, None);
    }

    #[test]
    fn parse_v0_4byte_offsets() {
        let data = build_v0_bytes(4);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.offset_size, 4);
        assert_eq!(sb.eof_address, 4096);
        assert_eq!(sb.root_group_address, 96);
    }

    #[test]
    fn parse_v1_8byte_offsets() {
        let data = build_v1_bytes(8);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.eof_address, 8192);
        assert_eq!(sb.root_group_address, 200);
        assert_eq!(sb.indexed_storage_internal_node_k, Some(32));
        assert_eq!(sb.group_leaf_node_k, Some(4));
        assert_eq!(sb.consistency_flags, 1);
    }

    /// The same fields, read from a version 1 superblock **the C library
    /// wrote**, with no dev-dependency and no 64-bit requirement.
    ///
    /// [`build_v1_bytes`] and [`parse_v1`] follow the same reading of the
    /// specification, so they agree about the field order whether or not that
    /// order is right: a builder and a parser that both swap the chunk B-tree K
    /// and the status flags leave both hand-built tests passing.
    ///
    /// `crates/crosscheck/tests/main/owned_swmr.rs` already covers this against a real file,
    /// and catches the same mutation. What it costs is the `hdf5-metno`
    /// dev-dependency, which needs 64-bit pointers, so that whole file compiles
    /// out on the i686 target, where address arithmetic is most likely to be
    /// wrong. Reading committed bytes needs neither, so this runs there.
    ///
    /// HDF5 1.8.23 wrote the file with `H5Pset_sym_k(8, 16)` and
    /// `H5Pset_istore_k(64)`: all three K values differ from one another and
    /// from the library's defaults (4 leaf, 16 internal, 32 chunk), so any
    /// permutation of the three reads back wrong. See
    /// `tests/data/c/1.8/NOTICE.md`.
    #[test]
    fn parse_v1_against_a_c_written_superblock() {
        let data: &[u8] = include_bytes!("../../../tests/data/c/1.8/v1_superblock.h5");
        let sb = parse_superblock(data, 0).unwrap();

        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.length_size, 8);

        // `H5Pset_sym_k(fcpl, 8, 16)` is (internal, leaf), the C library's
        // argument order, which is the reverse of the on-disk order. Version 0
        // also defines these two. The chunk B-tree K below is the one field
        // the version 1 layout adds, and the only one whose non-default value
        // makes the C library write a version 1 superblock at all.
        assert_eq!(sb.group_leaf_node_k, Some(16));
        assert_eq!(sb.group_internal_node_k, Some(8));
        assert_eq!(sb.indexed_storage_internal_node_k, Some(64));

        // The file was closed cleanly, so no status bit is set. Asserted beside
        // the K values because the chunk B-tree K directly follows this field:
        // this field is what every `File::open` consults on a version 3
        // superblock, so a parser that confuses the two passes a wrong value to
        // the open path. It cannot reach a rejection *here*:
        // `file_lock::check_status_flags` returns early below version 3, which is
        // why this asserts what was parsed, and not that the file opens.
        assert_eq!(sb.consistency_flags, 0);

        assert_eq!(sb.base_address, BaseAddress::ZERO);
        assert_eq!(sb.eof_address, data.len() as u64);
    }

    #[test]
    fn parse_v1_4byte_offsets() {
        let data = build_v1_bytes(4);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 4);
    }

    #[test]
    fn parse_v2_8byte_offsets() {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.eof_address, 2048);
        assert_eq!(sb.root_group_address, 48);
        assert!(sb.checksum.is_some());
        assert_eq!(sb.group_leaf_node_k, None);
    }

    #[test]
    fn parse_v2_4byte_offsets() {
        let data = build_v2_bytes(OffsetWidth::Four, LengthWidth::Four, 2);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.offset_size, 4);
    }

    #[test]
    fn parse_v3() {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 3);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.version, 3);
    }

    #[test]
    fn checksum_mismatch_v2() {
        let mut data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        // Corrupt the checksum
        let len = data.len();
        data[len - 1] ^= 0xFF;
        let err = parse_superblock(&data, 0).unwrap_err();
        assert!(
            matches!(err, FormatError::ChecksumMismatch { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn unsupported_version() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 99;
        assert_eq!(
            parse_superblock(&data, 0),
            Err(FormatError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn truncated_data() {
        let data = HDF5_SIGNATURE.to_vec(); // Just the signature, no version
        // Only 8 bytes, need at least 9
        assert!(matches!(
            parse_superblock(&data, 0),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn truncated_v0() {
        let mut data = vec![0u8; 20]; // Too short for v0
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0; // version 0
        data[13] = 8; // `offset_size`
        data[14] = 8; // `length_size`
        assert!(matches!(
            parse_superblock(&data, 0),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn invalid_offset_size() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0; // version 0
        data[13] = 3; // invalid `offset_size`
        data[14] = 8;
        assert_eq!(
            parse_superblock(&data, 0),
            Err(FormatError::InvalidOffsetSize(3))
        );
    }

    #[test]
    fn invalid_length_size() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0;
        data[13] = 8;
        data[14] = 5; // invalid `length_size`
        assert_eq!(
            parse_superblock(&data, 0),
            Err(FormatError::InvalidLengthSize(5))
        );
    }

    #[test]
    fn parse_at_nonzero_offset() {
        let mut data = vec![0u8; 1024];
        let v0 = build_v0_bytes(8);
        data[512..512 + v0.len()].copy_from_slice(&v0);
        let sb = parse_superblock(&data, 512).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.root_group_address, 96);
    }

    #[test]
    fn v2_2byte_offsets() {
        let data = build_v2_bytes(OffsetWidth::Two, LengthWidth::Two, 2);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(sb.offset_size, 2);
        assert_eq!(sb.eof_address, 2048);
    }

    #[test]
    fn a_superblock_whose_two_widths_differ_round_trips_to_the_same_bytes() {
        let data = build_v2_bytes(OffsetWidth::Two, LengthWidth::Four, 2);
        let sb = parse_superblock(&data, 0).unwrap();

        assert_eq!(sb.offset_size, 2);
        assert_eq!(sb.length_size, 4);
        assert_eq!(sb.eof_address, 2048);
        assert_eq!(sb.root_group_address, 48);

        assert_eq!(serialize_superblock(&sb).unwrap(), data);
    }

    #[test]
    fn a_superblock_with_8_byte_widths_round_trips_to_the_same_bytes() {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 3);
        let sb = parse_superblock(&data, 0).unwrap();
        assert_eq!(serialize_superblock(&sb).unwrap(), data);
    }

    #[test]
    fn parse_from_a_source_matches_buffered() {
        // A superblock at offset 512 in a larger file is parsed identically from
        // a source (reading only a small window) and from the in-memory buffer.
        let mut data = vec![0u8; 4096];
        let v2 = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        data[512..512 + v2.len()].copy_from_slice(&v2);

        let buffered = parse_superblock(&data, 512).unwrap();
        let from_source = parse_superblock_from_source(data.as_slice(), 512).unwrap();

        assert_eq!(buffered, from_source);
        assert_eq!(from_source.root_group_address, 48);
    }

    #[test]
    fn parse_from_a_source_validates_checksum() {
        let mut data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        let len = data.len();
        data[len - 1] ^= 0xFF; // corrupt the stored checksum
        assert!(matches!(
            parse_superblock_from_source(data.as_slice(), 0),
            Err(FormatError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn a_signature_offset_past_the_data_returns_unexpected_eof() {
        assert_eq!(
            parse_superblock(&HDF5_SIGNATURE, 9),
            Err(FormatError::UnexpectedEof {
                expected: 9,
                available: 8,
            })
        );
    }

    #[rstest]
    #[case::offset_size(3, 8, FormatError::InvalidOffsetSize(3))]
    #[case::length_size(8, 16, FormatError::InvalidLengthSize(16))]
    fn a_superblock_with_an_invalid_width_returns_its_width_error(
        #[case] offset_size: u8,
        #[case] length_size: u8,
        #[case] expected: FormatError,
    ) {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        let mut sb = parse_superblock(&data, 0).unwrap();
        sb.offset_size = offset_size;
        sb.length_size = length_size;

        assert_eq!(serialize_superblock(&sb), Err(expected));
    }

    #[rstest]
    #[case::version_0(0)]
    #[case::version_1(1)]
    #[case::version_4(4)]
    fn a_superblock_of_a_version_other_than_2_or_3_returns_unsupported_version(
        #[case] version: u8,
    ) {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        let mut sb = parse_superblock(&data, 0).unwrap();
        sb.version = version;

        assert_eq!(
            serialize_superblock(&sb),
            Err(FormatError::UnsupportedVersion(version))
        );
    }

    #[rstest]
    #[case::consistency_flags(
        |sb: &mut Superblock| sb.consistency_flags = 0x100,
        "consistency flags 0x100 do not fit the 1-byte field of a version 2 superblock"
    )]
    #[case::end_of_file_address(
        |sb: &mut Superblock| sb.eof_address = 0x1_0000,
        "the superblock's end-of-file address 0x10000 does not fit a 2-byte field"
    )]
    fn a_value_wider_than_its_field_returns_an_internal_error(
        #[case] widen: fn(&mut Superblock),
        #[case] detail: &str,
    ) {
        let data = build_v2_bytes(OffsetWidth::Two, LengthWidth::Two, 2);
        let mut sb = parse_superblock(&data, 0).unwrap();
        widen(&mut sb);

        assert_eq!(
            serialize_superblock(&sb),
            Err(FormatError::Internal(detail.into()))
        );
    }
}
