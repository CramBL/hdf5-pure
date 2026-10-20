//! The superblock parser and writer.
//!
//! A caller reads the bytes of each range the parser returns and passes them back to it.
//! [`SuperblockLocator`] returns the range of each candidate address, [`SuperblockSignature`] the
//! range of the fields that fix the superblock's length, and [`SuperblockRef`] the range of the
//! whole superblock.
//!
//! The superblock is defined in "Format Signature and Superblock" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super

use alloc::format;
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_core::__private::SuperblockFields;
pub(crate) use hdf5_pure_core::Superblock;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::signature::HDF5_SIGNATURE;
use crate::width;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Serializes `superblock` in the version 2 and 3 layout.
///
/// Writes the two width bytes, the consistency flags, the four addresses at the width
/// `offset_size` gives, and the Jenkins lookup3 checksum over the bytes before it. An absent
/// superblock extension address is written as the undefined address.
/// Writes base and end-of-file addresses as absolute values, and root and extension addresses
/// as encoded base-relative values.
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
        superblock
            .superblock_extension_address
            .unwrap_or(StoredAddress::undefined(offset_width))
            .get(),
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
        superblock.root_group_address.get(),
    )?;
    // checksum
    let checksum = checksum::jenkins_lookup3(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    Ok(buf)
}

/// A candidate address in the search for the superblock signature.
///
/// The candidates are byte 0, byte 512, and then each address twice the previous one, the
/// addresses defined in "Format Signature and Superblock" of the [format specification, version
/// 4.0][spec]. The last candidate is 2^63, the largest power of two a `u64` holds. A caller reads
/// the bytes of [`parse_range`](Self::parse_range) and passes them to [`parse`](Self::parse),
/// which returns the locator of the next candidate while the signature is absent.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuperblockLocator {
    candidate: PhysicalAddress,
}

impl SuperblockLocator {
    /// Creates the locator of the first candidate, byte 0.
    pub const fn new() -> Self {
        Self {
            candidate: PhysicalAddress::new(FIRST_CANDIDATE),
        }
    }

    /// Parses the bytes a caller read for this candidate.
    ///
    /// Returns one of:
    ///
    /// - [`SuperblockCandidate::Present`] if the bytes start with the signature
    /// - [`SuperblockCandidate::Exhausted`] if the signature is absent and the source ends before
    ///   the end of this candidate's range, or this candidate is the last
    /// - [`SuperblockCandidate::Absent`] with the locator of the next candidate otherwise
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if the region of a
    /// [`SuperblockCandidateRead::Bytes`] read does not contain [`parse_range`](Self::parse_range),
    /// or the region of a [`SuperblockCandidateRead::EndOfSource`] read starts after the candidate
    /// or ends before it.
    pub fn parse(
        self,
        read: SuperblockCandidateRead<'_>,
    ) -> Result<SuperblockCandidate, FormatError> {
        let range = self.parse_range()?;
        let (bytes, next) = match read {
            SuperblockCandidateRead::Bytes(region) => {
                (region.subregion(range)?.bytes(), self.next())
            }
            SuperblockCandidateRead::EndOfSource(region) => {
                let end = region.range().end().min(range.end());
                let held =
                    FileRange::new(range.start(), end.get().saturating_sub(range.start().get()))?;
                (region.subregion(held)?.bytes(), None)
            }
            SuperblockCandidateRead::PastExtent => return Ok(SuperblockCandidate::Exhausted),
        };
        Ok(if bytes.starts_with(&HDF5_SIGNATURE) {
            SuperblockCandidate::Present(SuperblockSignature {
                address: self.candidate,
            })
        } else {
            next.map_or(SuperblockCandidate::Exhausted, SuperblockCandidate::Absent)
        })
    }

    /// Returns the range a caller reads for this candidate, the
    /// [`CANDIDATE_LEN`](Self::CANDIDATE_LEN) bytes at the candidate address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the range ends past [`u64::MAX`]. The range of
    /// every candidate ends below it.
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(self.candidate, Self::CANDIDATE_LEN.to_u64())
    }

    fn next(&self) -> Option<Self> {
        let candidate = match self.candidate.get() {
            FIRST_CANDIDATE => SECOND_CANDIDATE,
            candidate => candidate.checked_mul(2)?,
        };
        Some(Self {
            candidate: PhysicalAddress::new(candidate),
        })
    }

    /// The length in bytes of a candidate's range, which contains the signature, the version and
    /// the two width fields of every superblock version.
    ///
    /// Versions 0 and 1 store the free-space, root group entry and shared header versions and a
    /// reserved byte between the version and the widths, which fill the last two of the 15 bytes.
    /// Versions 2 and 3 store the widths right after the version. Both layouts are defined in
    /// "Format Signature and Superblock" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
    pub const CANDIDATE_LEN: usize = 15;
}

impl Default for SuperblockLocator {
    fn default() -> Self {
        Self::new()
    }
}

/// The bytes a caller read for a candidate of the superblock search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuperblockCandidateRead<'a> {
    /// The source extends through the candidate's range, and the region contains that range.
    Bytes(FileRegion<'a>),
    /// The source ends inside the candidate's range, and the region contains the bytes from the
    /// candidate to the end of the source.
    EndOfSource(FileRegion<'a>),
    /// The candidate starts at or past the end of the source.
    PastExtent,
}

/// The result of parsing one candidate of the superblock search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SuperblockCandidate {
    /// The signature is absent from the candidate, and the locator is the next candidate's.
    Absent(SuperblockLocator),
    /// The search ends without a signature.
    Exhausted,
    /// The signature starts at the candidate.
    Present(SuperblockSignature),
}

/// The address of a superblock signature, from a [`SuperblockCandidate::Present`].
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse), which returns the range of the whole superblock.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuperblockSignature {
    address: PhysicalAddress,
}

impl SuperblockSignature {
    /// Parses the version and the two widths at the signature, and returns the superblock's range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::SignatureNotFound`] if the bytes at the
    /// address do not start with the signature, [`FormatError::UnsupportedVersion`] if the version
    /// is above 3, and [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if
    /// a width is not 2, 4, or 8.
    pub fn parse(&self, region: FileRegion<'_>) -> Result<SuperblockRef, FormatError> {
        let candidate =
            SuperblockCandidateFields::parse(region.subregion(self.parse_range()?)?.bytes())?;
        Ok(SuperblockRef {
            range: FileRange::new(self.address, candidate.encoded_len().to_u64())?,
        })
    }

    /// Returns the physical address of the signature's first byte.
    pub const fn address(&self) -> PhysicalAddress {
        self.address
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the
    /// [`SuperblockLocator::CANDIDATE_LEN`] bytes at the signature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the range ends past [`u64::MAX`]. The range at
    /// every address a [`SuperblockLocator`] returns ends below it.
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(self.address, SuperblockLocator::CANDIDATE_LEN.to_u64())
    }
}

/// The range of a superblock, from its signature to its last field.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuperblockRef {
    range: FileRange,
}

impl SuperblockRef {
    /// Parses the superblock in `region` and returns its fields and its validated widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), or if the version and the widths in `region` give a
    /// longer superblock than that range. Returns the errors [`SuperblockSignature::parse`]
    /// returns for the signature, the version and the widths,
    /// [`FormatError::InvalidGroupLeafNodeK`] if a version 0 or 1 superblock stores 0 in its "Group
    /// Leaf Node K" field, and, under the `checksum` feature, [`FormatError::ChecksumMismatch`] if
    /// the checksum of a version 2 or 3 superblock does not match its bytes.
    pub fn parse(&self, region: FileRegion<'_>) -> Result<(Superblock, FormatWidths), FormatError> {
        let Self { range } = *self;
        let held = region.subregion(range)?;
        let candidate = SuperblockCandidateFields::parse(held.bytes())?;
        let required = FileRange::new(range.start(), candidate.encoded_len().to_u64())?;
        parse_superblock(held.subregion(required)?.bytes())
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the whole superblock.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
}

fn parse_superblock(bytes: &[u8]) -> Result<(Superblock, FormatWidths), FormatError> {
    let mut fields = bytes::Fields::new(bytes, 0);
    let candidate = SuperblockCandidateFields::read(&mut fields)?;
    let superblock = match candidate.version {
        SuperblockVersion::Zero | SuperblockVersion::One => candidate.read_v0_or_v1(&mut fields)?,
        SuperblockVersion::Two | SuperblockVersion::Three => {
            checksum::verify_trailing(bytes)?;
            candidate.read_v2_or_v3(&mut fields)?
        }
    };
    if fields.pos() != bytes.len() {
        return Err(FormatError::Internal(format!(
            "the version {} superblock reader consumed {} of its {} bytes",
            candidate.version.get(),
            fields.pos(),
            bytes.len()
        )));
    }
    Ok((superblock, candidate.widths))
}

/// The version and the two widths of a superblock, which a candidate's range contains and which
/// fix the superblock's length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SuperblockCandidateFields {
    version: SuperblockVersion,
    widths: FormatWidths,
}

impl SuperblockCandidateFields {
    fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        Self::read(&mut bytes::Fields::new(bytes, 0))
    }

    fn read(fields: &mut bytes::Fields<'_>) -> Result<Self, FormatError> {
        if fields.array::<{ HDF5_SIGNATURE.len() }>()? != HDF5_SIGNATURE {
            return Err(FormatError::SignatureNotFound);
        }
        let version = SuperblockVersion::parse(fields.u8()?)?;
        if matches!(version, SuperblockVersion::Zero | SuperblockVersion::One) {
            fields.array::<V0_V1_VERSION_FIELDS_LEN>()?;
        }
        let offset_size = fields.u8()?;
        let length_size = fields.u8()?;
        Ok(Self {
            version,
            widths: FormatWidths::from_sizes(offset_size, length_size)?,
        })
    }

    fn encoded_len(self) -> usize {
        let offset = usize::from(self.widths.offsets.get());
        let length = usize::from(self.widths.lengths.get());
        let addresses = ADDRESS_COUNT * offset;
        // The root group symbol table entry stores its link name offset at the length width, as
        // `H5G_ent_decode` in `H5Gent.c` reads it (HDF5 1.8.23 and 2.2.0).
        let root_entry = length + offset + ROOT_ENTRY_TAIL_LEN;
        match self.version {
            SuperblockVersion::Zero => V0_V1_FIXED_LEN + addresses + root_entry,
            SuperblockVersion::One => V0_V1_FIXED_LEN + V1_FIELDS_LEN + addresses + root_entry,
            SuperblockVersion::Two | SuperblockVersion::Three => {
                V2_V3_FIXED_LEN + addresses + CHECKSUM_LEN
            }
        }
    }

    fn read_v0_or_v1(self, fields: &mut bytes::Fields<'_>) -> Result<Superblock, FormatError> {
        let Self { version, widths } = self;
        let FormatWidths { offsets, lengths } = widths;
        fields.u8()?;
        let group_leaf_node_k = fields.u16()?;
        if group_leaf_node_k == 0 {
            return Err(FormatError::InvalidGroupLeafNodeK);
        }
        let group_internal_node_k = fields.u16()?;
        let consistency_flags = fields.u32()?;
        let indexed_storage_internal_node_k = match version {
            SuperblockVersion::One => {
                let k = fields.u16()?;
                fields.u16()?;
                Some(k)
            }
            SuperblockVersion::Zero | SuperblockVersion::Two | SuperblockVersion::Three => None,
        };
        let base_address = BaseAddress::new(fields.offset(offsets)?);
        let free_space_address = fields.address(offsets)?;
        let eof_address = fields.offset(offsets)?;
        let driver_info_address = fields.address(offsets)?;
        fields.length(lengths)?;
        let root_group_address = fields.address(offsets)?;
        fields.array::<ROOT_ENTRY_TAIL_LEN>()?;
        Ok(SuperblockFields {
            version: version.get(),
            offset_size: offsets.get(),
            length_size: lengths.get(),
            base_address,
            eof_address,
            root_group_address,
            group_leaf_node_k: Some(group_leaf_node_k),
            group_internal_node_k: Some(group_internal_node_k),
            indexed_storage_internal_node_k,
            free_space_address: Some(free_space_address),
            driver_info_address: Some(driver_info_address),
            consistency_flags,
            superblock_extension_address: None,
            checksum: None,
        }
        .build())
    }

    fn read_v2_or_v3(self, fields: &mut bytes::Fields<'_>) -> Result<Superblock, FormatError> {
        let Self { version, widths } = self;
        let FormatWidths { offsets, lengths } = widths;
        let consistency_flags = u32::from(fields.u8()?);
        let base_address = BaseAddress::new(fields.offset(offsets)?);
        let superblock_extension_address = fields.address(offsets)?;
        let eof_address = fields.offset(offsets)?;
        let root_group_address = fields.address(offsets)?;
        let checksum = fields.u32()?;
        Ok(SuperblockFields {
            version: version.get(),
            offset_size: offsets.get(),
            length_size: lengths.get(),
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
            checksum: Some(checksum),
        }
        .build())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SuperblockVersion {
    Zero,
    One,
    Two,
    Three,
}

impl SuperblockVersion {
    fn parse(version: u8) -> Result<Self, FormatError> {
        Ok(match version {
            0 => Self::Zero,
            1 => Self::One,
            2 => Self::Two,
            3 => Self::Three,
            version => return Err(FormatError::UnsupportedVersion(version)),
        })
    }

    const fn get(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

fn write_address(
    buf: &mut Vec<u8>,
    width: OffsetWidth,
    field: &str,
    address: u64,
) -> Result<(), FormatError> {
    if !width::offset_width_holds(width, address) {
        return Err(FormatError::Internal(format!(
            "the superblock's {field} address {address:#x} does not fit a {}-byte field",
            width.get()
        )));
    }
    bytes::write_offset(buf, address, width);
    Ok(())
}

const FIRST_CANDIDATE: u64 = 0;
const SECOND_CANDIDATE: u64 = 512;

const V0_V1_VERSION_FIELDS_LEN: usize = 4;

// The fields before the base address, of versions 0 and 1 and of versions 2 and 3, and the
// indexed storage K and its reserved bytes that version 1 adds: "Format Signature and
// Superblock", format specification version 4.0.
const V0_V1_FIXED_LEN: usize = 24;
const V1_FIELDS_LEN: usize = 4;
const V2_V3_FIXED_LEN: usize = 12;

// Base, free-space or superblock extension, end-of-file, and driver information or root group
// addresses: "Format Signature and Superblock", format specification version 4.0.
const ADDRESS_COUNT: usize = 4;

// The cache type, the reserved bytes and the scratch pad after the root group symbol table
// entry's object header address: "Symbol Table Entry", format specification version 4.0.
const ROOT_ENTRY_TAIL_LEN: usize = 24;

// The checksum of versions 2 and 3: "Format Signature and Superblock", format specification
// version 4.0.
const CHECKSUM_LEN: usize = 4;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::superblock::VERSION_0;
    use test_util::superblock::VERSION_1;
    use test_util::superblock::VERSION_2;
    use test_util::superblock::VERSION_3;
    use test_util::superblock::v0;
    use test_util::superblock::v2;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::StoredAddressExt;

    #[rstest]
    #[case::at_0(0, &[0])]
    #[case::at_512(512, &[0, 512])]
    #[case::at_1024(1024, &[0, 512, 1024])]
    #[case::at_2048(2048, &[0, 512, 1024, 2048])]
    #[case::at_4096(4096, &[0, 512, 1024, 2048, 4096])]
    fn the_search_reads_each_candidate_up_to_the_signature(
        #[case] signature_at: u64,
        #[case] expected: &[u64],
    ) {
        let (visited, candidate) = search(signature_at + 4096, |start| {
            candidate_bytes(start == signature_at)
        });
        assert_eq!(visited, expected);
        assert_eq!(candidate, present(signature_at));
    }

    #[rstest]
    #[case::at_4_gib(1 << 32)]
    #[case::at_8_gib(1 << 33)]
    #[case::at_the_last_candidate(1 << 63)]
    fn the_search_reaches_a_candidate_past_32_bits(#[case] signature_at: u64) {
        let (visited, candidate) = search(u64::MAX, |start| candidate_bytes(start == signature_at));
        let doublings = signature_at.trailing_zeros() - SECOND_CANDIDATE.trailing_zeros();
        let expected: Vec<u64> = core::iter::once(FIRST_CANDIDATE)
            .chain((0..=doublings).map(|doubling| SECOND_CANDIDATE << doubling))
            .collect();
        assert_eq!(visited, expected);
        assert_eq!(candidate, present(signature_at));
    }

    #[test]
    fn a_search_without_a_signature_ends_at_the_first_candidate_past_the_extent() {
        let (visited, candidate) = search(5000, |_| candidate_bytes(false));
        assert_eq!(visited, [0, 512, 1024, 2048, 4096, 8192]);
        assert_eq!(candidate, SuperblockCandidate::Exhausted);
    }

    #[test]
    fn a_search_without_a_signature_ends_after_the_last_candidate() {
        let (visited, candidate) = search(u64::MAX, |_| candidate_bytes(false));
        // Byte 0 and the 55 powers of two from 2^9 to 2^63.
        assert_eq!(visited.len(), 56);
        assert_eq!(visited.last(), Some(&(1 << 63)));
        assert_eq!(candidate, SuperblockCandidate::Exhausted);
    }

    #[rstest]
    #[case::empty_source(0, &[0], SuperblockCandidate::Exhausted)]
    #[case::short_of_the_signature(515, &[0, 512], SuperblockCandidate::Exhausted)]
    #[case::short_of_the_version(520, &[0, 512], present(512))]
    fn a_source_that_ends_inside_a_candidate_ends_the_search(
        #[case] extent: u64,
        #[case] expected: &[u64],
        #[case] outcome: SuperblockCandidate,
    ) {
        let (visited, candidate) = search(extent, |start| candidate_bytes(start == 512));
        assert_eq!(visited, expected);
        assert_eq!(candidate, outcome);
    }

    #[test]
    fn a_signature_off_the_candidates_is_not_found() {
        let mut image = vec![0; 1024];
        image[100..108].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(locate(&image), Err(FormatError::SignatureNotFound));
    }

    #[test]
    fn the_search_returns_the_earliest_signature() {
        let mut image = vec![0; 1024];
        image[..8].copy_from_slice(&HDF5_SIGNATURE);
        image[512..520].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(locate(&image).unwrap().address(), PhysicalAddress::new(0));
    }

    #[rstest]
    #[case::bytes_short_of_the_candidate(false, 0, 14)]
    #[case::bytes_after_the_candidate_start(false, 1, 15)]
    #[case::end_of_source_after_the_candidate_start(true, 1, 15)]
    fn a_candidate_outside_the_region_returns_insufficient_region(
        #[case] end_of_source: bool,
        #[case] start: u64,
        #[case] length: u64,
    ) {
        let bytes = vec![0; usize::try_from(length).unwrap()];
        let region = FileRegion::new(
            FileRange::new(PhysicalAddress::new(start), length).unwrap(),
            &bytes,
        )
        .unwrap();
        let read = if end_of_source {
            SuperblockCandidateRead::EndOfSource(region)
        } else {
            SuperblockCandidateRead::Bytes(region)
        };
        assert_eq!(
            SuperblockLocator::new().parse(read),
            Err(FormatError::InsufficientRegion {
                required_start: 0,
                required_length: SuperblockLocator::CANDIDATE_LEN.to_u64(),
                provided_start: start,
                provided_length: length,
            })
        );
    }

    #[rstest]
    fn a_version_0_or_1_superblock_of_every_width_parses_at_its_candidate(
        #[values(VERSION_0, VERSION_1)] version: u8,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
        #[values(0, 512, 1 << 33)] address: u64,
    ) {
        let lengths = LengthWidth::try_from(offsets.get()).unwrap();
        assert_parses_at(version, offsets, lengths, address);
    }

    #[rstest]
    fn a_version_2_or_3_superblock_of_every_width_parses_at_its_candidate(
        #[values(VERSION_2, VERSION_3)] version: u8,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
        #[values(0, 512, 1 << 33)] address: u64,
    ) {
        assert_parses_at(version, offsets, lengths, address);
    }

    // The width bytes are at 13 and 14 in versions 0 and 1, and at 9 and 10 in versions 2 and 3.
    #[rstest]
    #[case::version_0_offsets(VERSION_0, 13, 3, FormatError::InvalidOffsetSize(3))]
    #[case::version_0_lengths(VERSION_0, 14, 16, FormatError::InvalidLengthSize(16))]
    #[case::version_2_offsets(VERSION_2, 9, 0, FormatError::InvalidOffsetSize(0))]
    #[case::version_3_lengths(VERSION_3, 10, 5, FormatError::InvalidLengthSize(5))]
    fn a_candidate_with_an_invalid_width_returns_its_width_error(
        #[case] version: u8,
        #[case] at: usize,
        #[case] width: u8,
        #[case] expected: FormatError,
    ) {
        let mut encoded = build(version, OffsetWidth::Eight, LengthWidth::Eight);
        encoded[at] = width;
        assert_eq!(
            locate(&encoded).unwrap().parse(region(&encoded)),
            Err(expected)
        );
    }

    #[rstest]
    #[case::version_4(4)]
    #[case::version_255(255)]
    fn a_candidate_of_an_unknown_version_returns_unsupported_version(#[case] version: u8) {
        let mut encoded = build(VERSION_2, OffsetWidth::Eight, LengthWidth::Eight);
        encoded[HDF5_SIGNATURE.len()] = version;
        assert_eq!(
            locate(&encoded).unwrap().parse(region(&encoded)),
            Err(FormatError::UnsupportedVersion(version))
        );
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    #[case::version_2(VERSION_2)]
    #[case::version_3(VERSION_3)]
    fn a_superblock_whose_checksum_differs_returns_checksum_mismatch(#[case] version: u8) {
        let mut encoded = build(version, OffsetWidth::Eight, LengthWidth::Eight);
        let computed = u32::from_le_bytes(*encoded.last_chunk().unwrap());
        *encoded.last_mut().unwrap() ^= 0xFF;
        let region = region(&encoded);
        let superblock = locate(&encoded).unwrap().parse(region).unwrap();
        assert_eq!(
            superblock.parse(region),
            Err(FormatError::ChecksumMismatch {
                expected: computed ^ 0xFF00_0000,
                computed,
            })
        );
    }

    #[rstest]
    #[case::one_byte_short(0, SuperblockLocator::CANDIDATE_LEN - 1)]
    #[case::past_the_region(512, SuperblockLocator::CANDIDATE_LEN)]
    fn a_signature_region_short_of_the_candidate_returns_insufficient_region(
        #[case] address: u64,
        #[case] len: usize,
    ) {
        let encoded = build(VERSION_2, OffsetWidth::Eight, LengthWidth::Eight);
        let signature = SuperblockSignature {
            address: PhysicalAddress::new(address),
        };
        assert_eq!(
            signature.parse(region(&encoded[..len])),
            Err(FormatError::InsufficientRegion {
                required_start: address,
                required_length: SuperblockLocator::CANDIDATE_LEN.to_u64(),
                provided_start: 0,
                provided_length: len.to_u64(),
            })
        );
    }

    #[test]
    fn a_region_short_of_the_superblock_returns_insufficient_region() {
        let encoded = build(VERSION_2, OffsetWidth::Eight, LengthWidth::Eight);
        let superblock = locate(&encoded).unwrap().parse(region(&encoded)).unwrap();
        let short = &encoded[..encoded.len() - 1];
        assert_eq!(
            superblock.parse(region(short)),
            Err(FormatError::InsufficientRegion {
                required_start: 0,
                required_length: encoded.len() as u64,
                provided_start: 0,
                provided_length: short.len() as u64,
            })
        );
    }

    #[test]
    fn a_superblock_whose_version_changed_to_a_longer_one_returns_insufficient_region() {
        let encoded = build(VERSION_0, OffsetWidth::Eight, LengthWidth::Eight);
        let superblock = locate(&encoded).unwrap().parse(region(&encoded)).unwrap();
        let mut changed = encoded.clone();
        changed[HDF5_SIGNATURE.len()] = VERSION_1;
        assert_eq!(
            superblock.parse(region(&changed)),
            Err(FormatError::InsufficientRegion {
                required_start: 0,
                required_length: (encoded.len() + V1_FIELDS_LEN) as u64,
                provided_start: 0,
                provided_length: encoded.len() as u64,
            })
        );
    }

    #[rstest]
    #[case(VERSION_0, false)]
    #[case(VERSION_0, true)]
    #[case(VERSION_1, false)]
    #[case(VERSION_1, true)]
    fn legacy_address_fields_reencode_in_stored_coordinates(
        #[case] version: u8,
        #[case] defined: bool,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    ) {
        let widths = Widths::new(usize::from(width.get()), usize::from(width.get()));
        let builder = v0::Superblock::new(widths)
            .base_address(512)
            .eof_address(2048)
            .root_group(0, 96);
        let mut data = if version == VERSION_1 {
            builder.version_1()
        } else {
            builder
        }
        .build();
        let fields = v0::Fields::read(&data, 0);
        let width_bytes = usize::from(width.get());
        if defined {
            let free_space_at = fields.eof_address_at - width_bytes;
            data[free_space_at..free_space_at + width_bytes]
                .copy_from_slice(&64u64.to_le_bytes()[..width_bytes]);
            let driver_at = fields.eof_address_at + width_bytes;
            data[driver_at..driver_at + width_bytes]
                .copy_from_slice(&128u64.to_le_bytes()[..width_bytes]);
        }
        let parsed = parse(&data);
        assert_eq!(parsed.base_address.get(), 512);
        assert_eq!(parsed.eof_address, 2048);
        assert_eq!(parsed.root_group_address.get(), 96);
        assert_eq!(
            parsed.free_space_address,
            Some(if defined {
                StoredAddress::new(64)
            } else {
                StoredAddress::undefined(width)
            })
        );
        assert_eq!(
            parsed.driver_info_address,
            Some(if defined {
                StoredAddress::new(128)
            } else {
                StoredAddress::undefined(width)
            })
        );
        assert_eq!(parsed.superblock_extension_address, None);
        let root_at = fields.eof_address_at + LEGACY_ROOT_FIELDS_AFTER_EOF * width_bytes;
        let mut encoded = Vec::new();
        write_address(
            &mut encoded,
            width,
            "root group object header",
            parsed.root_group_address.get(),
        )
        .unwrap();
        assert_eq!(encoded, data[root_at..root_at + width_bytes]);
        encoded.clear();
        write_address(&mut encoded, width, "end-of-file", parsed.eof_address).unwrap();
        assert_eq!(
            encoded,
            data[fields.eof_address_at..fields.eof_address_at + width_bytes]
        );
    }

    #[rstest]
    #[case(VERSION_2, false)]
    #[case(VERSION_2, true)]
    #[case(VERSION_3, false)]
    #[case(VERSION_3, true)]
    fn direct_address_fields_roundtrip_in_stored_coordinates(
        #[case] version: u8,
        #[case] defined: bool,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    ) {
        let mut data = v2::Superblock::new(Widths::new(
            usize::from(width.get()),
            usize::from(width.get()),
        ))
        .version(version)
        .base_address(512)
        .eof_address(2048)
        .root_header_address(96)
        .build();
        if defined {
            let width_bytes = usize::from(width.get());
            let extension_at = HDF5_SIGNATURE.len() + DIRECT_PREFIX_LEN + width_bytes;
            data[extension_at..extension_at + width_bytes]
                .copy_from_slice(&64u64.to_le_bytes()[..width_bytes]);
            data.truncate(data.len() - CHECKSUM_LEN);
            test_util::checksum::append(&mut data);
        }
        let parsed = parse(&data);
        assert_eq!(parsed.base_address.get(), 512);
        assert_eq!(parsed.eof_address, 2048);
        assert_eq!(parsed.root_group_address.get(), 96);
        assert_eq!(parsed.free_space_address, None);
        assert_eq!(parsed.driver_info_address, None);
        assert_eq!(
            parsed.superblock_extension_address,
            Some(if defined {
                StoredAddress::new(64)
            } else {
                StoredAddress::undefined(width)
            })
        );
        assert_eq!(serialize_superblock(&parsed).unwrap(), data);
    }

    #[rstest]
    fn a_group_leaf_node_k_of_zero_is_rejected(#[values(VERSION_0, VERSION_1)] version: u8) {
        let builder = v0::Superblock::new(Widths::EIGHT).group_leaf_node_k(0);
        let data = match version {
            VERSION_1 => builder.version_1(),
            _ => builder,
        }
        .build();
        let image = region(&data);
        let superblock = SuperblockSignature {
            address: PhysicalAddress::new(0),
        }
        .parse(image)
        .unwrap();
        assert_eq!(
            superblock.parse(image).unwrap_err(),
            FormatError::InvalidGroupLeafNodeK
        );
    }

    // HDF5 1.8.23 wrote the file with `H5Pset_sym_k(8, 16)` and `H5Pset_istore_k(64)`.
    #[test]
    fn parse_v1_against_a_c_written_superblock() {
        let data: &[u8] = include_bytes!("../../../tests/data/c/1.8/v1_superblock.h5");
        let sb = parse(data);

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
    fn a_superblock_whose_two_widths_differ_round_trips_to_the_same_bytes() {
        let data = build(VERSION_2, OffsetWidth::Two, LengthWidth::Four);
        assert_eq!(serialize_superblock(&parse(&data)).unwrap(), data);
    }

    #[rstest]
    #[case::offset_size(3, 8, FormatError::InvalidOffsetSize(3))]
    #[case::length_size(8, 16, FormatError::InvalidLengthSize(16))]
    fn a_superblock_with_an_invalid_width_returns_its_width_error(
        #[case] offset_size: u8,
        #[case] length_size: u8,
        #[case] expected: FormatError,
    ) {
        let mut sb = parse(&build(VERSION_2, OffsetWidth::Eight, LengthWidth::Eight));
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
        let mut sb = parse(&build(VERSION_2, OffsetWidth::Eight, LengthWidth::Eight));
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
        let mut sb = parse(&build(VERSION_2, OffsetWidth::Two, LengthWidth::Two));
        widen(&mut sb);

        assert_eq!(
            serialize_superblock(&sb),
            Err(FormatError::Internal(detail.into()))
        );
    }

    fn assert_parses_at(version: u8, offsets: OffsetWidth, lengths: LengthWidth, address: u64) {
        let encoded = build(version, offsets, lengths);
        let region = FileRegion::new(
            FileRange::new(PhysicalAddress::new(address), encoded.len() as u64).unwrap(),
            &encoded,
        )
        .unwrap();
        let mut locator = SuperblockLocator::new();
        let signature = loop {
            let range = locator.parse_range().unwrap();
            let read = if range.start() == region.range().start() {
                SuperblockCandidateRead::Bytes(region)
            } else {
                SuperblockCandidateRead::Bytes(
                    FileRegion::new(range, &[0; SuperblockLocator::CANDIDATE_LEN]).unwrap(),
                )
            };
            match locator.parse(read).unwrap() {
                SuperblockCandidate::Absent(next) => locator = next,
                SuperblockCandidate::Present(signature) => break signature,
                SuperblockCandidate::Exhausted => panic!("no signature at {address}"),
            }
        };
        let superblock = signature.parse(region).unwrap();
        assert_eq!(superblock.parse_range(), region.range());
        let (parsed, widths) = superblock.parse(region).unwrap();

        assert_eq!(widths, FormatWidths { offsets, lengths });
        assert_eq!(parsed.version, version);
        assert_eq!(parsed.offset_size, offsets.get());
        assert_eq!(parsed.length_size, lengths.get());
        assert_eq!(parsed.base_address, BaseAddress::ZERO);
        assert_eq!(parsed.eof_address, 4096);
        assert_eq!(parsed.root_group_address.get(), 96);
        match version {
            VERSION_0 | VERSION_1 => {
                assert_eq!(parsed.group_leaf_node_k, Some(4));
                assert_eq!(parsed.group_internal_node_k, Some(16));
                assert_eq!(
                    parsed.indexed_storage_internal_node_k,
                    (version == VERSION_1).then_some(64)
                );
                assert_eq!(parsed.consistency_flags, 1);
                assert_eq!(parsed.checksum, None);
            }
            _ => {
                assert_eq!(parsed.group_leaf_node_k, None);
                assert_eq!(parsed.consistency_flags, 1);
                assert_eq!(
                    parsed.checksum,
                    Some(u32::from_le_bytes(*encoded.last_chunk().unwrap()))
                );
                assert_eq!(serialize_superblock(&parsed).unwrap(), encoded);
            }
        }
    }

    fn search(
        extent: u64,
        candidate: impl Fn(u64) -> [u8; SuperblockLocator::CANDIDATE_LEN],
    ) -> (Vec<u64>, SuperblockCandidate) {
        let mut locator = SuperblockLocator::new();
        let mut visited = Vec::new();
        loop {
            let range = locator.parse_range().unwrap();
            let start = range.start().get();
            visited.push(start);
            let bytes = candidate(start);
            let read = if start >= extent {
                SuperblockCandidateRead::PastExtent
            } else if range.end().get() > extent {
                let held = FileRange::new(range.start(), extent - start).unwrap();
                SuperblockCandidateRead::EndOfSource(
                    FileRegion::new(held, &bytes[..usize::try_from(extent - start).unwrap()])
                        .unwrap(),
                )
            } else {
                SuperblockCandidateRead::Bytes(FileRegion::new(range, &bytes).unwrap())
            };
            match locator.parse(read).unwrap() {
                SuperblockCandidate::Absent(next) => locator = next,
                outcome => return (visited, outcome),
            }
        }
    }

    fn candidate_bytes(signed: bool) -> [u8; SuperblockLocator::CANDIDATE_LEN] {
        let mut bytes = [0; SuperblockLocator::CANDIDATE_LEN];
        if signed {
            bytes[..HDF5_SIGNATURE.len()].copy_from_slice(&HDF5_SIGNATURE);
        }
        bytes
    }

    const fn present(address: u64) -> SuperblockCandidate {
        SuperblockCandidate::Present(SuperblockSignature {
            address: PhysicalAddress::new(address),
        })
    }

    fn locate(image: &[u8]) -> Result<SuperblockSignature, FormatError> {
        let image = region(image);
        let extent = image.range().end();
        let mut locator = SuperblockLocator::new();
        loop {
            let range = locator.parse_range()?;
            let read = if range.start() >= extent {
                SuperblockCandidateRead::PastExtent
            } else if range.end() > extent {
                SuperblockCandidateRead::EndOfSource(image)
            } else {
                SuperblockCandidateRead::Bytes(image)
            };
            match locator.parse(read)? {
                SuperblockCandidate::Absent(next) => locator = next,
                SuperblockCandidate::Exhausted => return Err(FormatError::SignatureNotFound),
                SuperblockCandidate::Present(signature) => return Ok(signature),
            }
        }
    }

    fn parse(image: &[u8]) -> Superblock {
        let image = region(image);
        let superblock = SuperblockSignature {
            address: PhysicalAddress::new(0),
        }
        .parse(image)
        .unwrap();
        superblock.parse(image).unwrap().0
    }

    fn region(bytes: &[u8]) -> FileRegion<'_> {
        FileRegion::new(
            FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
            bytes,
        )
        .unwrap()
    }

    fn build(version: u8, offsets: OffsetWidth, lengths: LengthWidth) -> Vec<u8> {
        let widths = Widths::new(usize::from(offsets.get()), usize::from(lengths.get()));
        match version {
            VERSION_0 | VERSION_1 => {
                let builder = v0::Superblock::new(widths)
                    .consistency_flags(1)
                    .indexed_storage_internal_node_k(64)
                    .eof_address(4096)
                    .root_group(0, 96);
                match version {
                    VERSION_1 => builder.version_1(),
                    _ => builder,
                }
                .build()
            }
            _ => v2::Superblock::new(widths)
                .version(version)
                .consistency_flags(1)
                .eof_address(4096)
                .root_header_address(96)
                .build(),
        }
    }

    // EOF, driver information and link name precede the root header: sections
    // `subsec_fmt4_boot_super` and `subsec_fmt4_infra_symboltableentry`, version 4.0.
    const LEGACY_ROOT_FIELDS_AFTER_EOF: usize = 3;
    // Version, widths and flags precede addresses: section `subsec_fmt4_boot_super`, version 4.0.
    const DIRECT_PREFIX_LEN: usize = 4;
}
