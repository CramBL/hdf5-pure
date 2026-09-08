//! HDF5 Superblock parsing for versions 0, 1, 2, and 3.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use crate::address::BaseAddress;
use crate::bytes;
use crate::convert::TryToUsize;
use crate::error::FormatError;
use crate::signature::HDF5_SIGNATURE;
use crate::source::Source;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Upper bound on the on-disk size of a superblock across all versions: the
/// largest is v1 with 8-byte offsets at 100 bytes (28 prefix + 4 addresses +
/// a 40-byte root symbol-table entry). 128 leaves headroom while staying a
/// tiny, fixed window to pull from a streaming source.
const MAX_SUPERBLOCK_LEN: u64 = 128;

/// A file's parsed superblock, as reported by
/// [`File::superblock`](crate::File::superblock).
///
/// The superblock is the first HDF5 structure in a file and fixes the format
/// every other structure is read in: the version, the width of the offsets and
/// lengths in every message, and the [base address](Self::base_address) the
/// rest of the file's addresses are relative to. This is a read-only view of
/// what was found there; a field a given superblock version does not carry is
/// `None`, and the fields carry the values as stored rather than as validated.
///
/// `#[non_exhaustive]` because the format's later versions may add fields, and
/// growing this struct should not be a breaking change.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Superblock {
    /// Superblock version (0–3).
    pub version: u8,
    /// Size of offsets in bytes (2, 4, or 8).
    pub offset_size: u8,
    /// The width of an address in this file. [`Self::offset_size`] is this width as a `u8`.
    pub(crate) offset_width: OffsetWidth,
    /// Size of lengths in bytes (2, 4, or 8).
    pub length_size: u8,
    /// The width of a length in this file. [`Self::length_size`] is this width as a `u8`.
    pub(crate) length_width: LengthWidth,
    /// File base address.
    pub base_address: BaseAddress,
    /// End-of-file address.
    pub eof_address: u64,
    /// Root group object header address (v2/v3) or from symbol table entry (v0/v1).
    pub root_group_address: u64,
    /// Group leaf node K (v0/v1 only).
    pub group_leaf_node_k: Option<u16>,
    /// Group internal node K (v0/v1 only).
    pub group_internal_node_k: Option<u16>,
    /// Indexed storage internal node K (v1 only).
    pub indexed_storage_internal_node_k: Option<u16>,
    /// Free space address (v0/v1 only).
    pub free_space_address: Option<u64>,
    /// Driver info block address (v0/v1 only).
    pub driver_info_address: Option<u64>,
    /// File consistency flags.
    pub consistency_flags: u32,
    /// Superblock extension address (v2/v3 only).
    pub superblock_extension_address: Option<u64>,
    /// CRC32C checksum (v2/v3 only).
    pub checksum: Option<u32>,
}

impl Superblock {
    /// Serializes this superblock to bytes.
    ///
    /// Always writes the version 2 and 3 layout: the two width bytes, four addresses at
    /// [`Self::offset_width`], and the Jenkins lookup3 checksum over the bytes before it.
    pub(crate) fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(&HDF5_SIGNATURE);
        buf.push(self.version);
        buf.push(self.offset_width.get());
        buf.push(self.length_width.get());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "consistency flags occupy the 1-byte file-consistency-flags field of the superblock"
        )]
        buf.push(self.consistency_flags as u8);
        // base_address
        Self::write_offset(&mut buf, self.base_address.get(), self.offset_width);
        // superblock extension address
        let ext_addr = self.superblock_extension_address.unwrap_or(u64::MAX);
        Self::write_offset(&mut buf, ext_addr, self.offset_width);
        // eof_address
        Self::write_offset(&mut buf, self.eof_address, self.offset_width);
        // root_group_address
        Self::write_offset(&mut buf, self.root_group_address, self.offset_width);
        // checksum
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Appends `val` to `buf` as a little-endian integer of `width` bytes.
    ///
    /// Truncates `val` to its low `width` bytes, which is how the all-ones undefined address is
    /// written at each width.
    fn write_offset(buf: &mut Vec<u8>, val: u64, width: OffsetWidth) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "each arm writes the low `width` bytes of `val`"
        )]
        match width {
            OffsetWidth::Two => buf.extend_from_slice(&(val as u16).to_le_bytes()),
            OffsetWidth::Four => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            OffsetWidth::Eight => buf.extend_from_slice(&val.to_le_bytes()),
        }
    }

    /// Parse a superblock from a [`Source`], given the byte offset where its
    /// signature was found (see [`crate::signature::find_signature_in`]).
    ///
    /// The superblock is fully self-contained: every field is stored inline and
    /// none is a pointer followed during this parse, so only a small fixed
    /// window ([`MAX_SUPERBLOCK_LEN`] bytes, clamped to the bytes available) is
    /// read. This lets the entry point be parsed from a lazy streaming source
    /// without materializing the whole file.
    pub(crate) fn parse_from_source<S: Source + ?Sized>(
        source: &S,
        signature_offset: u64,
    ) -> Result<Superblock, FormatError> {
        let available = source.len().saturating_sub(signature_offset);
        let window = available.min(MAX_SUPERBLOCK_LEN).to_usize()?;
        let buf = source.read_metadata_at(signature_offset, window)?;
        // The window begins at the signature, so within `buf` it sits at 0.
        Self::parse(&buf, 0)
    }

    /// Parse a superblock from `data` starting at `signature_offset`.
    ///
    /// The signature must be present at the given offset.
    pub(crate) fn parse(data: &[u8], signature_offset: usize) -> Result<Superblock, FormatError> {
        let d = &data[signature_offset..];
        bytes::ensure_len(d, 0, 9)?; // signature(8) + version(1)

        // Verify signature
        if d[..8] != HDF5_SIGNATURE {
            return Err(FormatError::SignatureNotFound);
        }

        let version = d[8];
        match version {
            0 => Self::parse_v0(d),
            1 => Self::parse_v1(d),
            2 | 3 => Self::parse_v2v3(d, version),
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
        // 4 addresses + root symbol table entry
        let var_start = 24;
        let sym_entry_size = os + os + 4 + 4 + 16; // link_name_off, obj_hdr_addr, cache_type, reserved, scratch
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
        let _link_name_offset = bytes::read_offset_width(d, pos, offset_width)?;
        pos += os;
        let object_header_addr = bytes::read_offset_width(d, pos, offset_width)?;

        Ok(Superblock {
            version: 0,
            offset_size: offset_width.get(),
            offset_width,
            length_size: length_width.get(),
            length_width,
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
        })
    }

    fn parse_v1(d: &[u8]) -> Result<Superblock, FormatError> {
        // Same as v0 but adds indexed_storage_internal_node_k(2) + reserved(2)
        // *after* the consistency flags, not before them — the order the C
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
        let var_start = 28;
        let sym_entry_size = os + os + 4 + 4 + 16;
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
        let _link_name_offset = bytes::read_offset_width(d, pos, offset_width)?;
        pos += os;
        let object_header_addr = bytes::read_offset_width(d, pos, offset_width)?;

        Ok(Superblock {
            version: 1,
            offset_size: offset_width.get(),
            offset_width,
            length_size: length_width.get(),
            length_width,
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
        })
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

        Ok(Superblock {
            version,
            offset_size: offset_width.get(),
            offset_width,
            length_size: length_width.get(),
            length_width,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a v0 superblock byte buffer with 8-byte offsets.
    fn build_v0_bytes(offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&HDF5_SIGNATURE); // 0..8
        buf.push(0); // version = 0
        buf.push(0); // free_space_version
        buf.push(0); // root_group_version
        buf.push(0); // reserved
        buf.push(0); // shared_header_version
        buf.push(offset_size); // offset_size
        buf.push(offset_size); // length_size (same for simplicity)
        buf.push(0); // reserved
        buf.extend_from_slice(&4u16.to_le_bytes()); // group_leaf_node_k
        buf.extend_from_slice(&16u16.to_le_bytes()); // group_internal_node_k
        buf.extend_from_slice(&0u32.to_le_bytes()); // consistency_flags
        // base_address
        write_offset(&mut buf, 0, offset_size);
        // free_space_address
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size);
        // eof_address
        write_offset(&mut buf, 4096, offset_size);
        // driver_info_address
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size);
        // Root symbol table entry
        write_offset(&mut buf, 0, offset_size); // link_name_offset
        write_offset(&mut buf, 96, offset_size); // object_header_addr (root group)
        buf.extend_from_slice(&0u32.to_le_bytes()); // cache_type
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&[0u8; 16]); // scratch pad
        buf
    }

    fn write_offset(buf: &mut Vec<u8>, val: u64, size: u8) {
        match size {
            2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => panic!("bad test offset size"),
        }
    }

    fn build_v1_bytes(offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&HDF5_SIGNATURE);
        buf.push(1); // version
        buf.push(0); // free_space_version
        buf.push(0); // root_group_version
        buf.push(0); // reserved
        buf.push(0); // shared_header_version
        buf.push(offset_size);
        buf.push(offset_size);
        buf.push(0); // reserved
        buf.extend_from_slice(&4u16.to_le_bytes()); // group_leaf_node_k
        buf.extend_from_slice(&16u16.to_le_bytes()); // group_internal_node_k
        // The flags precede the chunk B-tree K in a v1 superblock. Distinct,
        // non-zero values in both, so reading them in the wrong order (which
        // this crate did until the fix for issue #245's review) is visible.
        buf.extend_from_slice(&1u32.to_le_bytes()); // consistency_flags
        buf.extend_from_slice(&32u16.to_le_bytes()); // indexed_storage_internal_node_k
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
        write_offset(&mut buf, 0, offset_size); // base
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size); // free space
        write_offset(&mut buf, 8192, offset_size); // eof
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size); // driver info
        // Root symbol table entry
        write_offset(&mut buf, 0, offset_size);
        write_offset(&mut buf, 200, offset_size); // root group addr
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        buf
    }

    fn build_v2_bytes(
        offset_width: OffsetWidth,
        length_width: LengthWidth,
        version: u8,
    ) -> Vec<u8> {
        let offset_size = offset_width.get();
        let mut buf = Vec::new();
        buf.extend_from_slice(&HDF5_SIGNATURE);
        buf.push(version);
        buf.push(offset_size);
        buf.push(length_width.get());
        buf.push(0); // consistency_flags
        write_offset(&mut buf, 0, offset_size); // base_address
        write_offset(&mut buf, 0xFFFFFFFFFFFFFFFF, offset_size); // superblock ext
        write_offset(&mut buf, 2048, offset_size); // eof
        write_offset(&mut buf, 48, offset_size); // root group obj hdr

        // Compute CRC32C of everything so far
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    #[test]
    fn parse_v0_8byte_offsets() {
        let data = build_v0_bytes(8);
        let sb = Superblock::parse(&data, 0).unwrap();
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
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.offset_size, 4);
        assert_eq!(sb.eof_address, 4096);
        assert_eq!(sb.root_group_address, 96);
    }

    #[test]
    fn parse_v1_8byte_offsets() {
        let data = build_v1_bytes(8);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.eof_address, 8192);
        assert_eq!(sb.root_group_address, 200);
        assert_eq!(sb.indexed_storage_internal_node_k, Some(32));
        assert_eq!(sb.group_leaf_node_k, Some(4));
        // The two fields either side of the v1 reserved bytes used to be read in
        // the wrong order, which reported the chunk B-tree K as the status flags
        // — the byte every open now consults. A real v1 file with the default
        // K of 32 would have read as "open for write" (bit 0 of 33, the C
        // library's other common value, likewise).
        assert_eq!(sb.consistency_flags, 1);
    }

    /// The same fields, read from a version 1 superblock **the C library
    /// wrote**, with no dev-dependency and no 64-bit requirement.
    ///
    /// [`build_v1_bytes`] and [`Superblock::parse_v1`] were written by the same
    /// hand from the same reading of the specification, so they agree about the
    /// field order whether or not that order is right — and it was not: the fix
    /// for issue #245's review corrected the chunk B-tree K and the status flags
    /// being read from each other's offsets. Reproducing that, by laying the
    /// bytes out the wrong way in *both* the builder and the parser, leaves both
    /// hand-built tests passing.
    ///
    /// `crates/crosscheck/tests/owned_swmr.rs` already covers this against a real file,
    /// and catches the same mutation. What it costs is the `hdf5-metno`
    /// dev-dependency, which needs 64-bit pointers, so that whole file compiles
    /// out on the i686 target — where address arithmetic is most likely to be
    /// wrong. Reading committed bytes needs neither, so this runs there.
    ///
    /// HDF5 1.8.23 wrote the file with `H5Pset_sym_k(8, 16)` and
    /// `H5Pset_istore_k(64)`: all three K values differ from one another and
    /// from the library's defaults (4 leaf, 16 internal, 32 chunk), so any
    /// permutation of the three reads back wrong. See
    /// `tests/data/c/1.8/NOTICE.md`.
    #[test]
    fn parse_v1_against_a_c_written_superblock() {
        let data: &[u8] = include_bytes!("../tests/data/c/1.8/v1_superblock.h5");
        let sb = Superblock::parse(data, 0).unwrap();

        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.length_size, 8);

        // `H5Pset_sym_k(fcpl, 8, 16)` is (internal, leaf) — the C library's
        // argument order, which is the reverse of the on-disk order. Version 0
        // already carries these two; the chunk B-tree K below is the one field
        // the version 1 layout adds, and the only one whose non-default value
        // makes the C library write a version 1 superblock at all.
        assert_eq!(sb.group_leaf_node_k, Some(16));
        assert_eq!(sb.group_internal_node_k, Some(8));
        assert_eq!(sb.indexed_storage_internal_node_k, Some(64));

        // The file was closed cleanly, so no status bit is set. Asserted beside
        // the K values because the two sit either side of the v1 reserved bytes
        // and were read from each other's offsets: this field is what every
        // `File::open` consults on a version 3 superblock, so a parser that
        // confuses the two carries a wrong answer into the open path. It cannot
        // reach a refusal *here* — `file_lock::check_status_flags` returns early
        // below version 3 — which is why this asserts what was parsed rather
        // than that the file opens.
        assert_eq!(sb.consistency_flags, 0);

        assert_eq!(sb.base_address, BaseAddress::ZERO);
        assert_eq!(sb.eof_address, data.len() as u64);
    }

    #[test]
    fn parse_v1_4byte_offsets() {
        let data = build_v1_bytes(4);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 1);
        assert_eq!(sb.offset_size, 4);
    }

    #[test]
    fn parse_v2_8byte_offsets() {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        let sb = Superblock::parse(&data, 0).unwrap();
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
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.offset_size, 4);
    }

    #[test]
    fn parse_v3() {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 3);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.version, 3);
    }

    #[test]
    fn checksum_mismatch_v2() {
        let mut data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        // Corrupt the checksum
        let len = data.len();
        data[len - 1] ^= 0xFF;
        let err = Superblock::parse(&data, 0).unwrap_err();
        matches!(err, FormatError::ChecksumMismatch { .. });
    }

    #[test]
    fn unsupported_version() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 99;
        assert_eq!(
            Superblock::parse(&data, 0),
            Err(FormatError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn truncated_data() {
        let data = HDF5_SIGNATURE.to_vec(); // Just the signature, no version
        // Only 8 bytes, need at least 9
        assert!(matches!(
            Superblock::parse(&data, 0),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn truncated_v0() {
        let mut data = vec![0u8; 20]; // Too short for v0
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0; // version 0
        data[13] = 8; // offset_size
        data[14] = 8; // length_size
        assert!(matches!(
            Superblock::parse(&data, 0),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn invalid_offset_size() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0; // version 0
        data[13] = 3; // invalid offset_size
        data[14] = 8;
        assert_eq!(
            Superblock::parse(&data, 0),
            Err(FormatError::InvalidOffsetSize(3))
        );
    }

    #[test]
    fn invalid_length_size() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[8] = 0;
        data[13] = 8;
        data[14] = 5; // invalid length_size
        assert_eq!(
            Superblock::parse(&data, 0),
            Err(FormatError::InvalidLengthSize(5))
        );
    }

    #[test]
    fn parse_at_nonzero_offset() {
        let mut data = vec![0u8; 1024];
        let v0 = build_v0_bytes(8);
        data[512..512 + v0.len()].copy_from_slice(&v0);
        let sb = Superblock::parse(&data, 512).unwrap();
        assert_eq!(sb.version, 0);
        assert_eq!(sb.root_group_address, 96);
    }

    #[test]
    fn v2_2byte_offsets() {
        let data = build_v2_bytes(OffsetWidth::Two, LengthWidth::Two, 2);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.offset_size, 2);
        assert_eq!(sb.offset_width, OffsetWidth::Two);
        assert_eq!(sb.eof_address, 2048);
    }

    #[test]
    fn a_superblock_whose_two_widths_differ_round_trips_to_the_same_bytes() {
        let data = build_v2_bytes(OffsetWidth::Two, LengthWidth::Four, 2);
        let sb = Superblock::parse(&data, 0).unwrap();

        assert_eq!(sb.offset_size, 2);
        assert_eq!(sb.offset_width, OffsetWidth::Two);
        assert_eq!(sb.length_size, 4);
        assert_eq!(sb.length_width, LengthWidth::Four);
        assert_eq!(sb.eof_address, 2048);
        assert_eq!(sb.root_group_address, 48);

        assert_eq!(sb.serialize(), data);
    }

    #[test]
    fn a_superblock_with_8_byte_widths_round_trips_to_the_same_bytes() {
        let data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 3);
        let sb = Superblock::parse(&data, 0).unwrap();
        assert_eq!(sb.offset_width, OffsetWidth::Eight);
        assert_eq!(sb.length_width, LengthWidth::Eight);
        assert_eq!(sb.serialize(), data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn parse_from_streaming_source_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        // A superblock at offset 512 in a larger file is parsed identically from
        // a lazy Read+Seek source (reading only a small window) and from the
        // in-memory buffer.
        let mut data = vec![0u8; 4096];
        let v2 = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        data[512..512 + v2.len()].copy_from_slice(&v2);

        let buffered = Superblock::parse(&data, 512).unwrap();
        let from_mem = Superblock::parse_from_source(&BytesSource::new(&data), 512).unwrap();
        let from_seek = Superblock::parse_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(data)).unwrap(),
            512,
        )
        .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(from_seek.root_group_address, 48);
    }

    #[cfg(feature = "std")]
    #[test]
    fn parse_from_streaming_source_validates_checksum() {
        use crate::source::ReadSeekSource;
        let mut data = build_v2_bytes(OffsetWidth::Eight, LengthWidth::Eight, 2);
        let len = data.len();
        data[len - 1] ^= 0xFF; // corrupt the stored checksum
        let src = ReadSeekSource::new(std::io::Cursor::new(data)).unwrap();
        assert!(matches!(
            Superblock::parse_from_source(&src, 0),
            Err(FormatError::ChecksumMismatch { .. })
        ));
    }
}
