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

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::OffsetWidth;

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
    /// Parses the free-space manager header at the start of `data`.
    ///
    /// Reads each length field as 8 bytes wide, and checks neither the version, the client ID,
    /// nor the checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the header, and
    /// [`FormatError::InvalidFreeSpaceManager`] if the signature is not `FSHD`.
    pub fn parse(widths: FormatWidths, data: &[u8]) -> Result<FreeSpaceManagerHeader, FormatError> {
        let os = usize::from(widths.offsets.get());
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
    // `width` is always 1..=8 (the count, size, and length widths). Take the low bytes of
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
///
/// # Errors
///
/// Returns [`FormatError::AddressTooLarge`] if `fshd_addr`, `fsse_addr`, or the address of a
/// section does not fit the offset width of `widths`, and [`FormatError::Internal`] if the sizes
/// of `sections` sum past `u64::MAX`.
pub fn serialize_free_space_manager(
    widths: FormatWidths,
    sections: &[FreeSection],
    fshd_addr: StoredAddress,
    fsse_addr: StoredAddress,
    class_id: u8,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let sets = size_sets(sections);
    let header = FreeSpaceManagerHeader {
        total_space: sections
            .iter()
            .try_fold(0, |total: u64, section| total.checked_add(section.size))
            .ok_or_else(|| {
                FormatError::Internal(format!(
                    "the sizes of {} free sections sum past u64::MAX",
                    sections.len()
                ))
            })?,
        total_sections: sections.len().to_u64(),
        addr_space_bits: address_space_bits(widths.offsets),
        max_section_size: FILE_FSM_MAX_SECTION_SIZE,
        fsse_addr,
        fsse_used: section_list_len(widths, sets.len(), sections.len()),
    };
    let fsse = header.encode_section_list(widths, fshd_addr, &sets, class_id)?;
    let fshd = header.encode(widths)?;
    Ok((fshd, fsse))
}

/// Returns the length in bytes of the header and the section list of a file free-space manager
/// that holds `sections`, or 0 if `sections` is empty.
///
/// A file with no free space stores the undefined address in place of a manager. The length
/// depends on `widths` and on the number and the sizes of the sections, and not on their
/// addresses, so a caller sizes the blocks before it places them.
pub fn free_space_manager_len(widths: FormatWidths, sections: &[FreeSection]) -> u64 {
    if sections.is_empty() {
        return 0;
    }
    let sets = sections
        .iter()
        .map(|section| section.size)
        .collect::<BTreeSet<u64>>()
        .len();
    free_space_manager_header_len(widths) + section_list_len(widths, sets, sections.len())
}

/// Returns the length in bytes of a header with the offset width of `widths`, 82 for 8-byte
/// addresses.
pub fn free_space_manager_header_len(widths: FormatWidths) -> u64 {
    header_len(widths).to_u64()
}

/// Returns the length in bytes of the section list of a manager that holds sections of
/// `section_sizes`.
///
/// The length depends on `widths`, the number of sections, and the number of distinct sizes, and
/// not on the addresses of the sections, and is the length of the section list
/// [`serialize_free_space_manager`] serializes for sections of these sizes.
pub fn section_info_len(widths: FormatWidths, section_sizes: &[u64]) -> u64 {
    let sets = section_sizes.iter().collect::<BTreeSet<&u64>>().len();
    section_list_len(widths, sets, section_sizes.len())
}

/// Parses the section list `data` of the manager `header` describes, checksum included, into its
/// free sections.
///
/// Reads `header.total_sections` sections, and checks neither the version, the header address,
/// the class of a section, nor the checksum.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the prefix and the checksum,
/// [`FormatError::InvalidFreeSpaceManager`] if the signature is not `FSSE`, the address space is
/// 0 bits, or the sections run past the checksum, and [`FormatError::ValueTooLargeForPlatform`]
/// if the section count does not fit a `usize`.
pub fn parse_section_info(
    widths: FormatWidths,
    data: &[u8],
    header: &FreeSpaceManagerHeader,
) -> Result<Vec<FreeSection>, FormatError> {
    let header_len = section_list_prefix_len(widths.offsets);
    if data.len() < header_len + CHECKSUM_LEN {
        return Err(FormatError::UnexpectedEof {
            expected: header_len + CHECKSUM_LEN,
            available: data.len(),
        });
    }
    if &data[0..4] != FSSE_SIGNATURE {
        return Err(FormatError::InvalidFreeSpaceManager);
    }
    let SectionFields {
        count: count_w,
        size: size_w,
        offset: off_w,
    } = header.section_fields();
    if off_w == 0 {
        return Err(FormatError::InvalidFreeSpaceManager);
    }

    let mut pos = header_len;
    let payload_end = data.len() - CHECKSUM_LEN;
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

impl FreeSpaceManagerHeader {
    fn section_fields(&self) -> SectionFields {
        SectionFields::new(
            self.total_sections,
            self.addr_space_bits,
            self.max_section_size,
        )
    }

    fn encode(&self, widths: FormatWidths) -> Result<Vec<u8>, FormatError> {
        let Self {
            total_space,
            total_sections,
            addr_space_bits,
            max_section_size,
            fsse_addr,
            fsse_used,
        } = *self;
        let mut fshd = Vec::with_capacity(header_len(widths));
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
        push_uint_le(&mut fshd, max_section_size, 8);
        bytes::try_write_offset(&mut fshd, fsse_addr.get(), widths.offsets)?;
        push_uint_le(&mut fshd, fsse_used, 8); // section info used
        push_uint_le(&mut fshd, fsse_used, 8); // section info allocated (== used)
        let checksum = checksum::jenkins_lookup3(&fshd);
        fshd.extend_from_slice(&checksum.to_le_bytes());
        Ok(fshd)
    }

    fn encode_section_list(
        &self,
        widths: FormatWidths,
        fshd_addr: StoredAddress,
        sets: &BTreeMap<u64, Vec<StoredAddress>>,
        class_id: u8,
    ) -> Result<Vec<u8>, FormatError> {
        let SectionFields { count, size, .. } = self.section_fields();
        let mut fsse = Vec::new();
        fsse.extend_from_slice(FSSE_SIGNATURE);
        fsse.push(0); // version
        bytes::try_write_offset(&mut fsse, fshd_addr.get(), widths.offsets)?;
        for (&section_size, offsets) in sets {
            push_uint_le(&mut fsse, offsets.len().to_u64(), count);
            push_uint_le(&mut fsse, section_size, size);
            for addr in offsets {
                bytes::try_write_offset(&mut fsse, addr.get(), widths.offsets)?;
                fsse.push(class_id); // section class id (no class-specific data)
            }
        }
        let checksum = checksum::jenkins_lookup3(&fsse);
        fsse.extend_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(
            fsse.len().to_u64(),
            self.fsse_used,
            "the section list is as long as the header records"
        );
        Ok(fsse)
    }
}

#[derive(Clone, Copy)]
struct SectionFields {
    count: usize,
    size: usize,
    offset: usize,
}

impl SectionFields {
    fn new(total_sections: u64, addr_space_bits: u16, max_section_size: u64) -> Self {
        Self {
            count: enc_size(total_sections),
            size: enc_size(max_section_size),
            offset: offset_width(addr_space_bits),
        }
    }

    /// Returns the widths of the fields of the section list [`serialize_free_space_manager`]
    /// writes at `widths` for a manager of `sections` sections.
    fn written(widths: FormatWidths, sections: usize) -> Self {
        Self::new(
            sections.to_u64(),
            address_space_bits(widths.offsets),
            FILE_FSM_MAX_SECTION_SIZE,
        )
    }
}

fn size_sets(sections: &[FreeSection]) -> BTreeMap<u64, Vec<StoredAddress>> {
    let mut sets = BTreeMap::<u64, Vec<StoredAddress>>::new();
    for section in sections {
        sets.entry(section.size).or_default().push(section.addr);
    }
    for addresses in sets.values_mut() {
        addresses.sort_unstable();
    }
    sets
}

/// Returns the length in bytes of the section list [`serialize_free_space_manager`] writes at
/// `widths` for `sections` sections of `sets` distinct sizes.
fn section_list_len(widths: FormatWidths, sets: usize, sections: usize) -> u64 {
    let SectionFields {
        count,
        size,
        offset,
    } = SectionFields::written(widths, sections);
    (section_list_prefix_len(widths.offsets)
        + sets * (count + size)
        + sections * (offset + SECTION_CLASS_LEN)
        + CHECKSUM_LEN)
        .to_u64()
}

/// Returns the length in bytes of a header at `widths`, checksum included.
fn header_len(widths: FormatWidths) -> usize {
    4 + 1 + 1 + 4 * 8 + 2 * 4 + 8 + usize::from(widths.offsets.get()) + 8 + 8 + CHECKSUM_LEN
}

/// Returns the length in bytes of the fields of a section list before its first set: the
/// signature, the version, and the header address.
fn section_list_prefix_len(offsets: OffsetWidth) -> usize {
    FSSE_SIGNATURE.len() + 1 + usize::from(offsets.get())
}

/// Returns the "Size of Address Space" the serializer writes in a file with addresses of
/// `offsets`, one bit less than the width of an address.
fn address_space_bits(offsets: OffsetWidth) -> u16 {
    u16::from(offsets.get()) * 8 - 1
}

const CHECKSUM_LEN: usize = 4;
const SECTION_CLASS_LEN: usize = 1;

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
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_space, 1600);
        assert_eq!(header.total_sections, 1);
        assert_eq!(header.addr_space_bits, 63);
        assert_eq!(header.fsse_addr, StoredAddress::new(701));
        assert_eq!(header.fsse_used, 35);

        let fsse = free_space::single_section_info();
        let sections = parse_section_info(widths(8, 8), &fsse, &header).unwrap();
        assert_eq!(sections, vec![section(2848, 1600)]);
    }

    #[test]
    fn parses_c_library_two_sections() {
        // manager @736: 16 bytes @871 and 893 bytes @1155, FSSE @818.
        let fshd = free_space::two_section_header();
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_space, 909);
        assert_eq!(header.total_sections, 2);
        assert_eq!(header.fsse_addr, StoredAddress::new(818));
        assert_eq!(header.fsse_used, 53);

        let fsse = free_space::two_section_info();
        let mut sections = parse_section_info(widths(8, 8), &fsse, &header).unwrap();
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
            widths(8, 8),
            &[section(2848, 1600)],
            StoredAddress::new(619),
            StoredAddress::new(701),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
        assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
        // The header length helper agrees with the produced bytes.
        assert_eq!(
            free_space_manager_header_len(widths(8, 8)),
            fshd.len() as u64
        );
    }

    #[test]
    fn serialize_matches_c_library_two_sections() {
        // FSHD@736 / FSSE@818: two differently-sized sections (16 @871, 893 @1155)
        // emitted as two ascending size-groups.
        let fshd_fixture = free_space::two_section_header();
        let fsse_fixture = free_space::two_section_info();
        let (fshd, fsse) = serialize_free_space_manager(
            widths(8, 8),
            &[section(1155, 893), section(871, 16)],
            StoredAddress::new(736),
            StoredAddress::new(818),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
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
                widths(8, 8),
                &sections,
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                SECTION_CLASS_LARGE,
            )
            .unwrap();
            assert_eq!(
                section_info_len(widths(8, 8), &sizes),
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
            FreeSpaceManagerHeader::parse(widths(8, 8), &fshd),
            Err(FormatError::InvalidFreeSpaceManager)
        ));
    }

    #[rstest::rstest]
    #[case::section_list_address(
        widths(4, 4),
        vec![section(0x1000, 0x100)],
        StoredAddress::new(0x1_0000_0000),
        FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 },
    )]
    #[case::section_address(
        widths(2, 4),
        vec![section(0x1_0000, 0x100)],
        StoredAddress::new(0x100),
        FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 },
    )]
    fn an_address_wider_than_its_field_returns_an_error(
        #[case] widths: FormatWidths,
        #[case] sections: Vec<FreeSection>,
        #[case] fsse_addr: StoredAddress,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            serialize_free_space_manager(
                widths,
                &sections,
                StoredAddress::new(0x80),
                fsse_addr,
                SECTION_CLASS_SIMPLE,
            ),
            Err(expected)
        );
    }

    #[test]
    fn sections_whose_sizes_overflow_a_u64_return_an_internal_error() {
        let sections: Vec<FreeSection> = (0..5)
            .map(|i| section(0x1000 * (i + 1), (1 << 62) - 1))
            .collect();

        assert_eq!(
            serialize_free_space_manager(
                widths(8, 8),
                &sections,
                StoredAddress::new(0x80),
                StoredAddress::new(0x100),
                SECTION_CLASS_SIMPLE,
            ),
            Err(FormatError::Internal(
                "the sizes of 5 free sections sum past u64::MAX".to_owned()
            ))
        );
    }

    fn widths(offset_size: u8, length_size: u8) -> FormatWidths {
        FormatWidths::from_sizes(offset_size, length_size).unwrap()
    }
}
