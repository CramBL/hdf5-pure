//! The superblock search and read over a byte slice or a [`Source`].

use alloc::format;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::SuperblockCandidate;
use hdf5_pure_format::__private::SuperblockCandidateRead;
use hdf5_pure_format::__private::SuperblockLocator;
use hdf5_pure_format::__private::SuperblockSignature;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source;
use crate::source::RangeReader;
use crate::source::Source;
use crate::width::FormatWidths;

#[cfg(all(test, feature = "std"))]
test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

pub use hdf5_pure_core::Superblock;

/// A superblock with the address of its signature and its validated widths.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct LocatedSuperblock {
    /// The physical address of the signature's first byte.
    pub(crate) address: PhysicalAddress,
    pub(crate) superblock: Superblock,
    pub(crate) widths: FormatWidths,
}

/// Locates the superblock of `file` and parses it.
///
/// Reads the [`SuperblockLocator::CANDIDATE_LEN`] bytes at each candidate address up to the
/// signature, fewer where `file` ends inside a candidate, and then the superblock in one read.
///
/// # Errors
///
/// Returns [`FormatError::SignatureNotFound`] if the signature is absent from every candidate
/// address, and [`FormatError::UnexpectedSourceEof`] if `file` ends inside the
/// [`SuperblockLocator::CANDIDATE_LEN`] bytes at the signature or inside the superblock. Its
/// `expected` field is the position in `file` the read needs to reach, and `available` the length
/// of `file`. Returns [`FormatError::UnsupportedVersion`] if the version is above 3,
/// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not 2, 4,
/// or 8, [`FormatError::InvalidGroupLeafNodeK`] if a version 0 or 1 superblock stores 0 in its
/// "Group Leaf Node K" field, and, under the `checksum` feature, [`FormatError::ChecksumMismatch`]
/// if the checksum of a version 2 or 3 superblock does not match its bytes. Returns the error of a
/// failed read of `file`.
pub(crate) fn read(
    file: &(impl SuperblockSource + ?Sized),
) -> Result<LocatedSuperblock, FormatError> {
    read_with_range(file).map(|(located, _range)| located)
}

/// Locates and parses the superblock, retaining the physical range of the full read.
///
/// Uses the candidate search and metadata read of [`read`]. The range is in physical source
/// coordinates. The caller establishes allocation ownership and source freshness separately.
///
/// # Errors
///
/// Returns the search, read, and parsing errors described by [`read`].
pub(crate) fn read_with_range(
    file: &(impl SuperblockSource + ?Sized),
) -> Result<(LocatedSuperblock, FileRange), FormatError> {
    let mut candidate = [0; SuperblockLocator::CANDIDATE_LEN];
    let (signature, held) = search(file, &mut candidate)?.ok_or(FormatError::SignatureNotFound)?;
    let candidate_range = signature.parse_range()?;
    if !held.contains(candidate_range) {
        return Err(source::unexpected_eof(candidate_range, file.extent()));
    }
    let held_bytes = candidate
        .get(..held.length().to_usize()?)
        .ok_or_else(|| candidate_overrun(held))?;
    let superblock = signature.parse(FileRegion::new(held, held_bytes)?)?;
    let range = superblock.parse_range();
    let bytes = file.read_range(range)?;
    let (superblock, widths) = superblock.parse(FileRegion::new(range, &bytes)?)?;
    Ok((
        LocatedSuperblock {
            address: signature.address(),
            superblock,
            widths,
        },
        range,
    ))
}

/// Returns the address of the first candidate that starts with the signature, or `None` if the
/// signature is absent from every candidate.
///
/// Reads the bytes at the candidate addresses alone, as [`read`] does before it reads the
/// superblock.
///
/// # Errors
///
/// Returns the error of a failed read of `file`.
pub(crate) fn find_signature(
    file: &(impl SuperblockSource + ?Sized),
) -> Result<Option<PhysicalAddress>, FormatError> {
    let mut candidate = [0; SuperblockLocator::CANDIDATE_LEN];
    Ok(search(file, &mut candidate)?.map(|(signature, _)| signature.address()))
}

/// The reads the superblock search makes of a file, a byte slice or a [`Source`].
///
/// A [`Source`] reads each candidate through [`Source::read_at`] and the superblock through
/// [`Source::read_metadata_at`], the read a metadata cache can keep.
pub(crate) trait SuperblockSource: RangeReader {
    /// Returns the length of the file in bytes.
    fn extent(&self) -> u64;

    /// Reads `buf.len()` bytes at `offset` into `buf`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the file ends before `offset + buf.len()`, and
    /// the error of a failed read of a [`Source`].
    fn read_candidate(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError>;
}

impl SuperblockSource for [u8] {
    fn extent(&self) -> u64 {
        self.len().to_u64()
    }

    fn read_candidate(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        buf.copy_from_slice(source::bytes_in(
            self,
            FileRange::new(PhysicalAddress::new(offset), buf.len().to_u64())?,
        )?);
        Ok(())
    }
}

impl<S: Source + ?Sized> SuperblockSource for S {
    fn extent(&self) -> u64 {
        self.len()
    }

    fn read_candidate(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.read_at(offset, buf)
    }
}

/// Returns the first signature and the part of its candidate's range inside `file`, or `None` if
/// the signature is absent from every candidate.
///
/// On return, `candidate` starts with the bytes of that part.
fn search(
    file: &(impl SuperblockSource + ?Sized),
    candidate: &mut [u8; SuperblockLocator::CANDIDATE_LEN],
) -> Result<Option<(SuperblockSignature, FileRange)>, FormatError> {
    let extent = file.extent();
    let mut locator = SuperblockLocator::new();
    loop {
        let range = locator.parse_range()?;
        let held = FileRange::new(
            range.start(),
            extent
                .saturating_sub(range.start().get())
                .min(range.length()),
        )?;
        let held_bytes = candidate
            .get_mut(..held.length().to_usize()?)
            .ok_or_else(|| candidate_overrun(held))?;
        let read = if held.length() == 0 {
            SuperblockCandidateRead::PastExtent
        } else {
            file.read_candidate(held.start().get(), held_bytes)?;
            let region = FileRegion::new(held, held_bytes)?;
            if held == range {
                SuperblockCandidateRead::Bytes(region)
            } else {
                SuperblockCandidateRead::EndOfSource(region)
            }
        };
        match locator.parse(read)? {
            SuperblockCandidate::Absent(next) => locator = next,
            SuperblockCandidate::Exhausted => return Ok(None),
            SuperblockCandidate::Present(signature) => return Ok(Some((signature, held))),
        }
    }
}

fn candidate_overrun(held: FileRange) -> FormatError {
    FormatError::Internal(format!(
        "a {}-byte superblock candidate exceeds its {}-byte buffer",
        held.length(),
        SuperblockLocator::CANDIDATE_LEN
    ))
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use core::cell::RefCell;

    use rstest::rstest;
    use test_util::superblock::v0;
    use test_util::superblock::v2;
    use test_util::widths::Widths;

    use super::*;
    use crate::SuperblockVersion;
    use crate::address::BaseAddress;
    use crate::address::BaseAddressExt;
    use crate::source::BytesSource;
    use crate::source::ReadSeekSource;

    #[rstest]
    fn every_reader_locates_the_superblock_at_its_candidate(
        #[values(0, 512, 1024, 4096)] signature_at: usize,
        #[values(false, true)] version_0: bool,
        #[values(Reader::Slice, Reader::Bytes, Reader::ReadSeek)] reader: Reader,
    ) {
        let image = image_at(
            signature_at,
            &superblock_bytes(version_0, signature_at as u64),
        );
        let LocatedSuperblock {
            address,
            superblock,
            widths,
        } = reader.read(&image).unwrap();
        assert_eq!(address.get(), signature_at as u64);
        assert_eq!(
            superblock.version(),
            if version_0 {
                SuperblockVersion::Zero
            } else {
                SuperblockVersion::Two
            }
        );
        assert_eq!(
            superblock.base_address,
            BaseAddress::new(signature_at as u64)
        );
        assert_eq!(superblock.root_group_address.get(), 48);
        assert_eq!(widths, FormatWidths::from_sizes(8, 8).unwrap());
        assert_eq!(
            reader.find_signature(&image).unwrap(),
            Some(PhysicalAddress::new(signature_at as u64))
        );
    }

    #[rstest]
    #[case::zero_two(SuperblockVersion::Zero, Widths::new(2, 2))]
    #[case::zero_four(SuperblockVersion::Zero, Widths::FOUR)]
    #[case::zero_eight(SuperblockVersion::Zero, Widths::EIGHT)]
    #[case::one_two(SuperblockVersion::One, Widths::new(2, 2))]
    #[case::one_four(SuperblockVersion::One, Widths::FOUR)]
    #[case::one_eight(SuperblockVersion::One, Widths::EIGHT)]
    #[case::two_two(SuperblockVersion::Two, Widths::new(2, 2))]
    #[case::two_four(SuperblockVersion::Two, Widths::FOUR)]
    #[case::two_eight(SuperblockVersion::Two, Widths::EIGHT)]
    #[case::two_long_offsets(SuperblockVersion::Two, Widths::new(8, 2))]
    #[case::two_long_lengths(SuperblockVersion::Two, Widths::new(2, 8))]
    #[case::three_two(SuperblockVersion::Three, Widths::new(2, 2))]
    #[case::three_four(SuperblockVersion::Three, Widths::FOUR)]
    #[case::three_eight(SuperblockVersion::Three, Widths::EIGHT)]
    #[case::three_long_offsets(SuperblockVersion::Three, Widths::new(8, 2))]
    #[case::three_long_lengths(SuperblockVersion::Three, Widths::new(2, 8))]
    fn parsed_ranges_preserve_versions_widths_and_candidate_reads(
        #[case] version: SuperblockVersion,
        #[case] widths: Widths,
        #[values(0, 512)] base: u64,
    ) {
        let encoded = superblock_bytes_for(version, widths, base);
        let source = SparseSource::new(base + 4_096, base, encoded.clone());
        let (located, range) = read_with_range(&source).unwrap();
        assert_eq!(
            range,
            FileRange::new(
                PhysicalAddress::new(base),
                u64::try_from(encoded.len()).unwrap()
            )
            .unwrap()
        );
        assert_eq!(located.address, range.start());
        assert_eq!(located.superblock.version(), version);
        assert_eq!(located.superblock.base_address, BaseAddress::new(base));
        assert_eq!(located.superblock.eof_address, 4_096);
        assert_eq!(located.superblock.root_group_address.get(), 48);
        assert_eq!(
            located.widths,
            FormatWidths::from_sizes(
                u8::try_from(widths.offset).unwrap(),
                u8::try_from(widths.length).unwrap()
            )
            .unwrap()
        );
        let mut expected = vec![(0, SuperblockLocator::CANDIDATE_LEN)];
        if base > 0 {
            expected.push((base, SuperblockLocator::CANDIDATE_LEN));
        }
        expected.push((base, encoded.len()));
        assert_eq!(*source.reads.borrow(), expected);
        let image = image_at(usize::try_from(base).unwrap(), &encoded);
        assert_eq!(read_with_range(image.as_slice()).unwrap(), (located, range));
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[rstest]
    #[case::legacy(SuperblockVersion::One)]
    #[case::modern(SuperblockVersion::Three)]
    fn retaining_the_parsed_range_preserves_allocation_and_read_profiles(
        #[case] version: SuperblockVersion,
    ) {
        use test_util::allocation;

        let encoded = superblock_bytes_for(version, Widths::EIGHT, 512);
        let original = SparseSource::new(4_608, 512, encoded.clone());
        let ranged = SparseSource::new(4_608, 512, encoded.clone());
        original.reads.borrow_mut().reserve_exact(3);
        ranged.reads.borrow_mut().reserve_exact(3);
        let (located, before) =
            allocation::measure(&format!("superblock_original_{version:?}"), || {
                read(&original).unwrap()
            });
        let ((retained, range), after) =
            allocation::measure(&format!("superblock_retained_{version:?}"), || {
                read_with_range(&ranged).unwrap()
            });
        assert_eq!(retained, located);
        assert_eq!(
            range,
            FileRange::new(
                PhysicalAddress::new(512),
                u64::try_from(encoded.len()).unwrap()
            )
            .unwrap()
        );
        assert_eq!(*ranged.reads.borrow(), *original.reads.borrow());
        assert_eq!(
            (
                after.blocks,
                after.bytes,
                after.peak_bytes,
                after.live_bytes
            ),
            (
                before.blocks,
                before.bytes,
                before.peak_bytes,
                before.live_bytes
            )
        );
        assert!(
            before.blocks > 0 && before.bytes >= u64::try_from(encoded.len()).unwrap(),
            "{before}"
        );
        let acquisition = SparseSource::new(4_608, 512, encoded.clone());
        acquisition.reads.borrow_mut().reserve_exact(1);
        let expected_range = FileRange::new(
            PhysicalAddress::new(512),
            u64::try_from(encoded.len()).unwrap(),
        )
        .unwrap();
        let ((), control) = allocation::measure(
            &format!("superblock_metadata_acquisition_{version:?}"),
            || {
                let bytes = acquisition.read_range(expected_range).unwrap();
                assert_eq!(bytes.as_ref(), encoded.as_slice());
            },
        );
        assert_eq!(*acquisition.reads.borrow(), [(512, encoded.len())]);
        let metadata_bytes = u64::try_from(encoded.len()).unwrap();
        let metadata_budget = metadata_bytes + 64;
        assert!(
            control.blocks > 0
                && control.blocks <= 2
                && control.bytes >= metadata_bytes
                && control.bytes <= metadata_budget
                && control.peak_bytes <= metadata_budget,
            "{control}"
        );
        assert_eq!(control.live_bytes, 0, "{control}");
        assert!(
            after.blocks <= control.blocks
                && after.bytes <= control.bytes
                && after.peak_bytes <= control.peak_bytes,
            "{after} exceeds {control}"
        );
        assert_eq!(after.live_bytes, 0, "{after}");
        let image = image_at(512, &encoded);
        let ((borrowed, borrowed_range), slice) =
            allocation::measure(&format!("superblock_borrowed_{version:?}"), || {
                read_with_range(image.as_slice()).unwrap()
            });
        assert_eq!((borrowed, borrowed_range), (retained, range));
        assert_eq!((slice.blocks, slice.bytes), (0, 0), "{slice}");
    }

    #[rstest]
    fn every_reader_reads_a_userblock_file_libhdf5_wrote(
        #[values(Reader::Slice, Reader::Bytes, Reader::ReadSeek)] reader: Reader,
    ) {
        let image = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/matlab/test_enum_v73.mat"
        ));
        let LocatedSuperblock {
            address,
            superblock,
            ..
        } = reader.read(image).unwrap();
        assert_eq!(address.get(), 512);
        assert_eq!(superblock.version(), SuperblockVersion::Zero);
        assert_eq!(superblock.base_address, BaseAddress::new(512));
        assert_eq!(superblock.eof_address, image.len() as u64);
    }

    #[rstest]
    #[case::empty(0)]
    #[case::short_of_a_signature(7)]
    #[case::past_the_last_candidate(5000)]
    fn every_reader_reports_a_missing_signature(
        #[case] len: usize,
        #[values(Reader::Slice, Reader::Bytes, Reader::ReadSeek)] reader: Reader,
    ) {
        let image = vec![0; len];
        assert!(
            matches!(reader.read(&image), Err(FormatError::SignatureNotFound)),
            "{:?}",
            reader.read(&image).err()
        );
        assert_eq!(reader.find_signature(&image).unwrap(), None);
    }

    // The candidate at the signature is 15 bytes, and a version 2 superblock of 8-byte widths 48.
    #[rstest]
    #[case::inside_the_candidate(10, 15)]
    #[case::inside_the_superblock(30, 48)]
    fn every_reader_reports_a_superblock_the_source_ends_inside(
        #[case] len: usize,
        #[case] expected: usize,
        #[values(Reader::Slice, Reader::Bytes, Reader::ReadSeek)] reader: Reader,
    ) {
        let mut image = superblock_bytes(false, 0);
        image.truncate(len);
        assert_eq!(
            reader.read(&image).err(),
            Some(FormatError::UnexpectedSourceEof {
                expected: expected.to_u64(),
                available: len.to_u64(),
            })
        );
        assert_eq!(
            reader.find_signature(&image).unwrap(),
            Some(PhysicalAddress::new(0))
        );
    }

    // The version byte is at 8, and the width bytes of version 2 are at 9 and 10.
    #[rstest]
    #[case::unknown_version(8, 9, FormatError::UnsupportedVersion(9))]
    #[case::invalid_offset_width(9, 3, FormatError::InvalidOffsetSize(3))]
    #[case::invalid_length_width(10, 5, FormatError::InvalidLengthSize(5))]
    fn a_malformed_superblock_at_the_first_signature_fails_the_read(
        #[case] at: usize,
        #[case] value: u8,
        #[case] expected: FormatError,
        #[values(Reader::Slice, Reader::Bytes, Reader::ReadSeek)] reader: Reader,
    ) {
        let mut image = image_at(512, &superblock_bytes(false, 512));
        let malformed = superblock_bytes(false, 0);
        image[..malformed.len()].copy_from_slice(&malformed);
        image[at] = value;
        assert_eq!(reader.read(&image).err(), Some(expected));
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    fn a_checksum_mismatch_at_the_first_signature_fails_the_read(
        #[values(Reader::Slice, Reader::Bytes, Reader::ReadSeek)] reader: Reader,
    ) {
        let mut image = image_at(512, &superblock_bytes(false, 512));
        let malformed = superblock_bytes(false, 0);
        let computed = u32::from_le_bytes(*malformed.last_chunk().unwrap());
        image[..malformed.len()].copy_from_slice(&malformed);
        image[malformed.len() - 1] ^= 0xFF;
        assert_eq!(
            reader.read(&image).err(),
            Some(FormatError::ChecksumMismatch {
                expected: computed ^ 0xFF00_0000,
                computed,
            })
        );
    }

    #[test]
    fn a_failed_candidate_read_is_returned_rather_than_a_missing_signature() {
        let source = SparseSource::new(4096, 2048, superblock_bytes(false, 2048)).failing_at(512);
        let failure = Some(FormatError::Source("candidate read failed".into()));
        assert_eq!(read(&source).err(), failure);
        assert_eq!(read_with_range(&source).err(), failure);
        assert_eq!(find_signature(&source).err(), failure);
    }

    #[test]
    fn a_superblock_past_4_gib_is_read_through_its_candidates_alone() {
        let at = 1 << 33;
        let encoded = superblock_bytes(false, at);
        let source = SparseSource::new(at + 4096, at, encoded.clone());
        let LocatedSuperblock {
            address,
            superblock,
            ..
        } = read(&source).unwrap();
        assert_eq!(address.get(), at);
        assert_eq!(superblock.base_address, BaseAddress::new(at));

        let candidates = core::iter::once(0).chain((9..=33).map(|shift| 1u64 << shift));
        let mut expected: Vec<(u64, usize)> = candidates
            .map(|start| (start, SuperblockLocator::CANDIDATE_LEN))
            .collect();
        expected.push((at, encoded.len()));
        assert_eq!(source.reads.into_inner(), expected);
    }

    #[test]
    fn a_source_that_overstates_its_length_is_searched_a_bounded_number_of_times() {
        let source = SparseSource::new(u64::MAX, 0, Vec::new());
        assert_eq!(find_signature(&source), Ok(None));
        // Byte 0 and the 55 powers of two from 2^9 to 2^63.
        assert_eq!(source.reads.into_inner().len(), 56);
    }

    #[derive(Clone, Copy, Debug)]
    enum Reader {
        Slice,
        Bytes,
        ReadSeek,
    }

    impl Reader {
        fn read(self, image: &[u8]) -> Result<LocatedSuperblock, FormatError> {
            match self {
                Self::Slice => read(image),
                Self::Bytes => read(&BytesSource::new(image)),
                Self::ReadSeek => read(&ReadSeekSource::new(std::io::Cursor::new(image)).unwrap()),
            }
        }

        fn find_signature(self, image: &[u8]) -> Result<Option<PhysicalAddress>, FormatError> {
            match self {
                Self::Slice => find_signature(image),
                Self::Bytes => find_signature(&BytesSource::new(image)),
                Self::ReadSeek => {
                    find_signature(&ReadSeekSource::new(std::io::Cursor::new(image)).unwrap())
                }
            }
        }
    }

    struct SparseSource {
        len: u64,
        at: u64,
        bytes: Vec<u8>,
        fail_at: Option<u64>,
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl SparseSource {
        fn new(len: u64, at: u64, bytes: Vec<u8>) -> Self {
            Self {
                len,
                at,
                bytes,
                fail_at: None,
                reads: RefCell::new(Vec::new()),
            }
        }

        fn failing_at(self, offset: u64) -> Self {
            Self {
                fail_at: Some(offset),
                ..self
            }
        }
    }

    impl Source for SparseSource {
        fn len(&self) -> u64 {
            self.len
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push((offset, buf.len()));
            if self.fail_at == Some(offset) {
                return Err(FormatError::Source("candidate read failed".into()));
            }
            for (position, byte) in (offset..).zip(buf.iter_mut()) {
                *byte = position
                    .checked_sub(self.at)
                    .and_then(|index| usize::try_from(index).ok())
                    .and_then(|index| self.bytes.get(index))
                    .copied()
                    .unwrap_or(0);
            }
            Ok(())
        }
    }

    fn image_at(signature_at: usize, superblock: &[u8]) -> Vec<u8> {
        let mut image = vec![0; signature_at + 4096];
        image[signature_at..signature_at + superblock.len()].copy_from_slice(superblock);
        image
    }

    fn superblock_bytes(version_0: bool, base_address: u64) -> Vec<u8> {
        superblock_bytes_for(
            if version_0 {
                SuperblockVersion::Zero
            } else {
                SuperblockVersion::Two
            },
            Widths::EIGHT,
            base_address,
        )
    }

    fn superblock_bytes_for(
        version: SuperblockVersion,
        widths: Widths,
        base_address: u64,
    ) -> Vec<u8> {
        match version {
            SuperblockVersion::Zero | SuperblockVersion::One => {
                let builder = v0::Superblock::new(widths)
                    .base_address(base_address)
                    .eof_address(4_096)
                    .root_group(0, 48);
                if version == SuperblockVersion::One {
                    builder.version_1().build()
                } else {
                    builder.build()
                }
            }
            SuperblockVersion::Two | SuperblockVersion::Three => v2::Superblock::new(widths)
                .version(version.get())
                .base_address(base_address)
                .eof_address(4_096)
                .root_header_address(48)
                .build(),
        }
    }
}
