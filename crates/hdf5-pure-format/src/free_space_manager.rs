//! The blocks of a persistent free-space manager: the free-space manager header (`FSHD`) and
//! its section list (`FSSE`).
//!
//! A file created with `H5Pset_file_space_strategy` and `persist` set stores the free space of
//! each of its free-space managers in these blocks, and the [File Space Info
//! message](crate::file_space_info) in its superblock extension holds the address of each header.
//! This module parses and serializes the blocks of the file client, client ID 1, with the length
//! fields of the header 8 bytes wide. The blocks are defined in "Free-space Index" of the [format
//! specification, version 4.0][spec].
//!
//! The section list stores each set of sections as a count, a size, and an offset and a class per
//! section, in the fewest bytes that hold the header's section count, its maximum section size,
//! and an address in its address space.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex

use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;

const FSHD_SIGNATURE: &[u8; 4] = b"FSHD";
const FSSE_SIGNATURE: &[u8; 4] = b"FSSE";

/// One free region of a file: the address the file stores for it and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeSection {
    /// The address of the region.
    pub addr: StoredAddress,
    /// The length of the region in bytes.
    pub size: u64,
}

/// A free-space manager header, signature `FSHD`, with the fields a reader of its section list
/// needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceManagerHeader {
    /// The number of free bytes the manager tracks, the "Total Space Tracked" field.
    pub total_space: u64,
    /// The number of sections the manager tracks, the "Total Number of Sections" field.
    pub total_sections: u64,
    /// The number of bits an address in the section list takes, the "Size of Address Space"
    /// field.
    pub addr_space_bits: u16,
    /// The largest section the manager tracks, the "Maximum Section Size" field.
    pub max_section_size: u64,
    /// The address of the section list.
    pub fsse_addr: StoredAddress,
    /// The length of the section list in bytes, the "Size of Serialized Section List Used" field.
    pub fsse_used: u64,
}

/// Returns the fewest bytes that hold `value`, and at least 1, as `H5VM_limit_enc_size` computes
/// it.
fn enc_size(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    bits.div_ceil(8).max(1)
}

/// Returns the width in bytes of a section offset in an address space of `addr_space_bits` bits.
fn offset_width(addr_space_bits: u16) -> usize {
    (addr_space_bits as usize).div_ceil(8)
}

/// Reads `bytes` as a little-endian unsigned integer.
fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

impl FreeSpaceManagerHeader {
    /// Parse an `FSHD` at the start of `data`.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<FreeSpaceManagerHeader, FormatError> {
        let os = offset_size as usize;
        // sig(4) ver(1) client(1) + 4*L + classes/shrink/expand/abits (2 each) +
        // max(L) + fsse_addr(O) + used(L) + alloc(L) + checksum(4)
        let need = 4 + 1 + 1 + 4 * 8 + 2 * 4 + 8 + os + 8 + 8 + 4;
        if data.len() < need {
            return Err(FormatError::UnexpectedEof {
                expected: need,
                available: data.len(),
            });
        }
        if &data[0..4] != FSHD_SIGNATURE {
            return Err(FormatError::InvalidFreeSpaceManager);
        }
        // sig(4) + version(1) + client(1)
        let mut pos = 6;
        let total_space = read_uint_le(&data[pos..pos + 8]);
        pos += 8;
        let total_sections = read_uint_le(&data[pos..pos + 8]);
        pos += 8;
        // skip serialized + ghost section counts
        pos += 16;
        // skip section-class count + shrink% + expand%
        pos += 6;
        let addr_space_bits = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;
        let max_section_size = read_uint_le(&data[pos..pos + 8]);
        pos += 8;
        let fsse_addr = StoredAddress::new(read_uint_le(&data[pos..pos + os]));
        pos += os;
        let fsse_used = read_uint_le(&data[pos..pos + 8]);
        Ok(FreeSpaceManagerHeader {
            total_space,
            total_sections,
            addr_space_bits,
            max_section_size,
            fsse_addr,
            fsse_used,
        })
    }
}

/// The number of section classes the header of a file free-space manager stores, the three
/// classes the C library registers for the file client (`H5MF.c`, HDF5 2.2.0).
const FILE_FSM_NUM_CLASSES: u16 = 3;
/// The shrink percent the header stores, `H5MF_FSPACE_SHRINK`.
const FILE_FSM_SHRINK_PCT: u16 = 80;
/// The expand percent the header stores, `H5MF_FSPACE_EXPAND`.
const FILE_FSM_EXPAND_PCT: u16 = 120;
/// The client ID of a file free-space manager, where 0 is a fractal heap's.
const FILE_FSM_CLIENT_ID: u8 = 1;

/// The maximum section size the header stores, `2^63 - 1`, as the C library writes it for 8-byte
/// addresses.
const FILE_FSM_MAX_SECTION_SIZE: u64 = (1u64 << 63) - 1;

/// The section class of the free space of a file that is not paged, `H5MF_FSPACE_SECT_SIMPLE`.
///
/// A section of each of the three classes stores its offset and its class, and no data.
pub const SECTION_CLASS_SIMPLE: u8 = 0;
/// The section class of free space smaller than a page in a paged file, `H5MF_FSPACE_SECT_SMALL`.
pub const SECTION_CLASS_SMALL: u8 = 1;
/// The section class of the manager of free space of a page or more in a paged file,
/// `H5MF_FSPACE_SECT_LARGE`.
///
/// The class identifies the manager, and a section of it may be smaller than a page.
pub const SECTION_CLASS_LARGE: u8 = 2;

/// Appends the low `width` bytes of `value` to `buf`, little-endian.
fn push_uint_le(buf: &mut Vec<u8>, value: u64, width: usize) {
    // `width` is always 1..=8 (offset/size/count widths). Take the low bytes of
    // the little-endian encoding without a narrowing cast.
    buf.extend_from_slice(&value.to_le_bytes()[..width]);
}

/// Serializes the header, for `fshd_addr`, and the section list, for `fsse_addr`, of a file
/// free-space manager that holds `sections`.
///
/// The header stores the address of the section list, and the section list the address of the
/// header. The section list groups the sections by size, in ascending order of size and then of
/// address, as the C library serializes them (`H5FScache.c`, HDF5 2.2.0), and tags every section
/// with `class_id`, since a manager holds sections of one class. Each block ends in its checksum,
/// and in a file with 8-byte addresses both match the blocks the C library writes for the same
/// sections.
///
/// Returns the header and the section list.
pub fn serialize_free_space_manager(
    sections: &[FreeSection],
    fshd_addr: StoredAddress,
    fsse_addr: StoredAddress,
    offset_size: u8,
    class_id: u8,
) -> (Vec<u8>, Vec<u8>) {
    let os = offset_size as usize;
    let addr_space_bits = (offset_size as u16) * 8 - 1;
    let total_sections = sections.len() as u64;
    let total_space: u64 = sections.iter().map(|s| s.size).sum();

    let off_w = offset_width(addr_space_bits);
    let size_w = enc_size(FILE_FSM_MAX_SECTION_SIZE);
    let count_w = enc_size(total_sections);

    // --- FSSE: sections grouped by size (ascending), offsets ascending within
    // a group, exactly as the C library serializes its size-ordered skip list. ---
    let mut fsse = Vec::new();
    fsse.extend_from_slice(FSSE_SIGNATURE);
    fsse.push(0); // version
    push_uint_le(&mut fsse, fshd_addr.get(), os); // back-pointer to the header

    // Group sections by size, then emit the groups in ascending size order with
    // ascending offsets within each, matching the C library's size-ordered list.
    let mut by_size: Vec<(u64, Vec<StoredAddress>)> = Vec::new();
    for s in sections {
        match by_size.iter_mut().find(|(size, _)| *size == s.size) {
            Some((_, offsets)) => offsets.push(s.addr),
            None => by_size.push((s.size, vec![s.addr])),
        }
    }
    // The loop above keeps one entry per distinct size, so there is no tie to
    // break and the sort need not be stable.
    by_size.sort_by_key(|(size, _)| *size);
    for (size, mut offsets) in by_size {
        offsets.sort_unstable();
        push_uint_le(&mut fsse, offsets.len() as u64, count_w);
        push_uint_le(&mut fsse, size, size_w);
        for addr in offsets {
            push_uint_le(&mut fsse, addr.get(), off_w);
            fsse.push(class_id); // section class id (no class-specific data)
        }
    }
    let checksum = checksum::jenkins_lookup3(&fsse);
    fsse.extend_from_slice(&checksum.to_le_bytes());
    let section_info_len = fsse.len() as u64;

    // --- FSHD: fixed-layout header referencing the section info just built. ---
    let mut fshd = Vec::with_capacity(4 + 1 + 1 + 4 * 8 + 2 * 4 + 8 + os + 8 + 8 + 4);
    fshd.extend_from_slice(FSHD_SIGNATURE);
    fshd.push(0); // version
    fshd.push(FILE_FSM_CLIENT_ID);
    push_uint_le(&mut fshd, total_space, 8);
    push_uint_le(&mut fshd, total_sections, 8);
    push_uint_le(&mut fshd, total_sections, 8); // serialized (all of them)
    push_uint_le(&mut fshd, 0, 8); // ghost (none)
    fshd.extend_from_slice(&FILE_FSM_NUM_CLASSES.to_le_bytes());
    fshd.extend_from_slice(&FILE_FSM_SHRINK_PCT.to_le_bytes());
    fshd.extend_from_slice(&FILE_FSM_EXPAND_PCT.to_le_bytes());
    fshd.extend_from_slice(&addr_space_bits.to_le_bytes());
    push_uint_le(&mut fshd, FILE_FSM_MAX_SECTION_SIZE, 8);
    push_uint_le(&mut fshd, fsse_addr.get(), os);
    push_uint_le(&mut fshd, section_info_len, 8); // section info used
    push_uint_le(&mut fshd, section_info_len, 8); // section info allocated (== used)
    let checksum = checksum::jenkins_lookup3(&fshd);
    fshd.extend_from_slice(&checksum.to_le_bytes());

    (fshd, fsse)
}

/// Returns the length in bytes of the header and the section list of a file free-space manager
/// that holds `sections`, or 0 if `sections` is empty.
///
/// The flat counterpart of `plan_paged_managers`, which computes the same length
/// for a paged file's several managers and places them as well. Both depend only on
/// the sections' count and sizes, never on their addresses, which is what lets a
/// commit size its tail before it has an address for it.
pub fn free_space_manager_len(sections: &[FreeSection], offset_size: u8) -> u64 {
    if sections.is_empty() {
        return 0;
    }
    let sizes: Vec<u64> = sections.iter().map(|s| s.size).collect();
    free_space_manager_header_len(offset_size) + section_info_len(&sizes, offset_size)
}

/// The fixed serialized byte length of an `FSHD` header with `offset_size`-byte
/// addresses (82 bytes for standard 8-byte offsets).
pub fn free_space_manager_header_len(offset_size: u8) -> u64 {
    (4 + 1 + 1 + 4 * 8 + 2 * 4 + 8 + offset_size as usize + 8 + 8 + 4) as u64
}

/// The serialized byte length an `FSSE` block will occupy for a manager holding
/// sections of `section_sizes`, which the paged writer uses to reserve
/// space for a manager's section list before the sections' offsets are placed. The
/// length depends only on the sizes (they determine the size-group count) and the
/// section count, never on the offsets or class id, so this defers to
/// [`serialize_free_space_manager`] with placeholder addresses and stays exact by
/// construction.
pub fn section_info_len(section_sizes: &[u64], offset_size: u8) -> u64 {
    let sections: Vec<FreeSection> = section_sizes
        .iter()
        .map(|&size| FreeSection {
            addr: StoredAddress::new(0),
            size,
        })
        .collect();
    let (_fshd, fsse) = serialize_free_space_manager(
        &sections,
        StoredAddress::new(0),
        StoredAddress::new(0),
        offset_size,
        SECTION_CLASS_SIMPLE,
    );
    fsse.len() as u64
}

/// Parses the section list `data` of the manager `header` describes, checksum included, into its
/// free sections.
///
/// Reads `header.total_sections` sections, and checks neither the version, the header address,
/// the class of a section, nor the checksum.
///
/// # Errors
///
/// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8,
/// [`FormatError::UnexpectedEof`] if `data` is shorter than the prefix and the checksum,
/// [`FormatError::InvalidFreeSpaceManager`] if the signature is not `FSSE`, the address space is
/// 0 bits, or the sections run past the checksum, and [`FormatError::ValueTooLargeForPlatform`]
/// if the section count does not fit a `usize`.
pub fn parse_section_info(
    data: &[u8],
    header: &FreeSpaceManagerHeader,
    offset_size: u8,
) -> Result<Vec<FreeSection>, FormatError> {
    let os = offset_size as usize;
    let header_len = 4 + 1 + os; // "FSSE" + version + back-pointer
    if data.len() < header_len + 4 {
        return Err(FormatError::UnexpectedEof {
            expected: header_len + 4,
            available: data.len(),
        });
    }
    if &data[0..4] != FSSE_SIGNATURE {
        return Err(FormatError::InvalidFreeSpaceManager);
    }
    let off_w = offset_width(header.addr_space_bits);
    let size_w = enc_size(header.max_section_size);
    let count_w = enc_size(header.total_sections);
    if off_w == 0 || size_w == 0 || count_w == 0 {
        return Err(FormatError::InvalidFreeSpaceManager);
    }

    let mut pos = header_len;
    let payload_end = data.len() - 4; // exclude checksum
    let total = header.total_sections.to_usize()?;
    let mut sections = Vec::with_capacity(total);
    while sections.len() < total {
        if pos + count_w + size_w > payload_end {
            return Err(FormatError::InvalidFreeSpaceManager);
        }
        let count = read_uint_le(&data[pos..pos + count_w]);
        pos += count_w;
        let size = read_uint_le(&data[pos..pos + size_w]);
        pos += size_w;
        for _ in 0..count {
            if pos + off_w + 1 > payload_end || sections.len() >= total {
                return Err(FormatError::InvalidFreeSpaceManager);
            }
            let addr = StoredAddress::new(read_uint_le(&data[pos..pos + off_w]));
            pos += off_w;
            // the class byte, the last field of a section of the file client
            pos += 1;
            sections.push(FreeSection { addr, size });
        }
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use test_util::free_space;

    use super::*;

    /// A free section at `addr` spanning `size` bytes.
    fn section(addr: u64, size: u64) -> FreeSection {
        FreeSection {
            addr: StoredAddress::new(addr),
            size,
        }
    }

    #[test]
    fn parses_c_library_single_section() {
        // manager @619: one 1600-byte free section at offset 2848, FSSE @701.
        let fshd = free_space::single_section_header();
        let header = FreeSpaceManagerHeader::parse(&fshd, 8).unwrap();
        assert_eq!(header.total_space, 1600);
        assert_eq!(header.total_sections, 1);
        assert_eq!(header.addr_space_bits, 63);
        assert_eq!(header.fsse_addr, StoredAddress::new(701));
        assert_eq!(header.fsse_used, 35);

        let fsse = free_space::single_section_info();
        let sections = parse_section_info(&fsse, &header, 8).unwrap();
        assert_eq!(sections, vec![section(2848, 1600)]);
    }

    #[test]
    fn parses_c_library_two_sections() {
        // manager @736: 16 bytes @871 and 893 bytes @1155, FSSE @818.
        let fshd = free_space::two_section_header();
        let header = FreeSpaceManagerHeader::parse(&fshd, 8).unwrap();
        assert_eq!(header.total_space, 909);
        assert_eq!(header.total_sections, 2);
        assert_eq!(header.fsse_addr, StoredAddress::new(818));
        assert_eq!(header.fsse_used, 53);

        let fsse = free_space::two_section_info();
        let mut sections = parse_section_info(&fsse, &header, 8).unwrap();
        sections.sort_by_key(|s| s.addr);
        assert_eq!(sections, vec![section(871, 16), section(1155, 893),]);
        // The section sizes sum to the header's tracked total.
        let total: u64 = sections.iter().map(|s| s.size).sum();
        assert_eq!(header.total_space, total);
    }

    #[test]
    fn serialize_matches_c_library_single_section() {
        // Byte-for-byte reproduction of the FSHD@619 / FSSE@701 fixtures,
        // including the Jenkins checksums the C library verifies on read.
        let fshd_fixture = free_space::single_section_header();
        let fsse_fixture = free_space::single_section_info();
        let (fshd, fsse) = serialize_free_space_manager(
            &[section(2848, 1600)],
            StoredAddress::new(619),
            StoredAddress::new(701),
            8,
            SECTION_CLASS_SIMPLE,
        );
        assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
        assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
        // The header length helper agrees with the produced bytes.
        assert_eq!(free_space_manager_header_len(8), fshd.len() as u64);
    }

    #[test]
    fn serialize_matches_c_library_two_sections() {
        // FSHD@736 / FSSE@818: two differently-sized sections (16 @871, 893 @1155)
        // emitted as two ascending size-groups.
        let fshd_fixture = free_space::two_section_header();
        let fsse_fixture = free_space::two_section_info();
        let (fshd, fsse) = serialize_free_space_manager(
            &[section(1155, 893), section(871, 16)],
            StoredAddress::new(736),
            StoredAddress::new(818),
            8,
            SECTION_CLASS_SIMPLE,
        );
        assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
        assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
    }

    #[test]
    fn fsse_len_matches_serialized_length() {
        // For any section set, the reserved FSSE length equals the serializer's
        // output length regardless of offsets or class id (fixed field widths).
        for sizes in [
            vec![],
            vec![100u64],
            vec![100, 100, 100],   // one size group
            vec![10, 20, 30],      // three size groups
            vec![16384, 512, 512], // large + repeated small
        ] {
            let sections: Vec<FreeSection> = sizes
                .iter()
                .enumerate()
                .map(|(i, &size)| section(4096 + i as u64 * 8, size))
                .collect();
            let (_fshd, fsse) = serialize_free_space_manager(
                &sections,
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                8,
                SECTION_CLASS_LARGE,
            );
            assert_eq!(
                section_info_len(&sizes, 8),
                fsse.len() as u64,
                "sizes {sizes:?}"
            );
        }
    }

    #[test]
    fn enc_size_matches_reference() {
        assert_eq!(enc_size(0), 1);
        assert_eq!(enc_size(255), 1);
        assert_eq!(enc_size(256), 2);
        assert_eq!(enc_size((1 << 63) - 1), 8);
    }

    #[test]
    fn rejects_bad_signature() {
        let mut fshd = free_space::single_section_header();
        fshd[0] = b'X';
        assert!(matches!(
            FreeSpaceManagerHeader::parse(&fshd, 8),
            Err(FormatError::InvalidFreeSpaceManager)
        ));
    }
}
