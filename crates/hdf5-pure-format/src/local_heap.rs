//! The local heap parser: the heap header and the null-terminated strings of its data segment.
//!
//! A group that stores its links in a symbol table keeps their names in a local heap, and each
//! symbol table entry holds the offset of its name in the data segment. [`LocalHeapLocator`]
//! parses the header from the region of the file the caller reads, and the [`LocalHeap`] it
//! returns takes the data segment from a second region, where [`LocalHeapDataSegment::name`]
//! parses each name.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;

/// The address of a local heap header, from the local heap address of a symbol table message.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse), which returns the [`LocalHeap`] with the range of its data segment.
/// The header is defined in "Local Heaps" of the [format specification, version 4.0][spec],
/// which describes version 0, the one version [`parse`](Self::parse) accepts.
///
/// # Examples
///
/// ```
/// # use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion};
/// # use hdf5_pure_core::PhysicalAddress;
/// # use hdf5_pure_core::__private::StoredAddressExt;
/// # use hdf5_pure_core::{BaseAddress, StoredAddress};
/// # use hdf5_pure_format::__private::{FormatWidths, LocalHeapLocator};
/// # use hdf5_pure_format::FormatError;
/// # use test_util::local_heap::Segment;
/// # use test_util::widths::Widths;
/// # let segment = Segment::of_names(32, &["alpha", "beta"]);
/// # let mut bytes = segment.header(Widths::EIGHT);
/// # bytes.extend_from_slice(&segment.bytes);
/// # let range = FileRange::new(PhysicalAddress::new(512), bytes.len() as u64)?;
/// # let file = FileRegion::new(range, &bytes)?;
/// let widths = FormatWidths::from_sizes(8, 8)?;
/// let locator = LocalHeapLocator::new(widths, BaseAddress::new(512), StoredAddress::new(0));
/// let heap = locator.parse(file.subregion(locator.parse_range()?)?)?;
///
/// let names = heap.data_segment(file.subregion(heap.data_segment_range())?)?;
/// assert_eq!((names.name(0)?, names.name(6)?), ("alpha", "beta"));
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_localheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalHeapLocator {
    widths: FormatWidths,
    base: BaseAddress,
    address: StoredAddress,
}

impl LocalHeapLocator {
    /// Creates the locator of the header at the base-relative `address`.
    pub const fn new(widths: FormatWidths, base: BaseAddress, address: StoredAddress) -> Self {
        Self {
            widths,
            base,
            address,
        }
    }

    /// Parses the header in `region`.
    ///
    /// The parser ignores the reserved bytes and the offset to the head of the free list.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidLocalHeapSignature`] if the bytes
    /// do not start with `HEAP`, [`FormatError::InvalidLocalHeapVersion`] if the version is not 0,
    /// and [`FormatError::UndefinedLocalHeapDataSegment`] if the header stores the undefined address
    /// as the address of the data segment. Returns [`FormatError::OffsetOverflow`] if the base
    /// address plus the header's address or plus the data segment's address, or the end of the
    /// header or of the data segment, is past [`u64::MAX`].
    pub fn parse(&self, region: FileRegion<'_>) -> Result<LocalHeap, FormatError> {
        let Self {
            widths,
            base,
            address: _,
        } = *self;
        let mut fields = bytes::Fields::new(region.subregion(self.parse_range()?)?.bytes(), 0);
        if fields.array::<{ SIGNATURE.len() }>()? != SIGNATURE {
            return Err(FormatError::InvalidLocalHeapSignature);
        }
        let version = fields.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidLocalHeapVersion(version));
        }
        fields.array::<RESERVED_LEN>()?;
        let data_segment_size = fields.length(widths.lengths)?;
        fields.length(widths.lengths)?;
        let DecodedStoredAddress::Defined(data_segment_address) =
            fields.decoded_address(widths.offsets)?
        else {
            return Err(FormatError::UndefinedLocalHeapDataSegment);
        };
        Ok(LocalHeap {
            data_segment: FileRange::new(
                PhysicalAddress::new(base.absolute(data_segment_address)?),
                data_segment_size,
            )?,
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the whole header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the base address plus the header's address, or
    /// the end of the header, is past [`u64::MAX`].
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        let Self {
            widths,
            base,
            address,
        } = *self;
        FileRange::new(
            PhysicalAddress::new(base.absolute(address)?),
            header_len(widths),
        )
    }
}

/// A local heap, parsed from its header by [`LocalHeapLocator::parse`].
///
/// A caller reads the bytes of [`data_segment_range`](Self::data_segment_range) and passes them
/// to [`data_segment`](Self::data_segment), which returns the segment the names are parsed from.
/// The heap is defined in "Local Heaps" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_localheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalHeap {
    data_segment: FileRange,
}

impl LocalHeap {
    /// Returns the range of the data segment, from its address over the size the header stores,
    /// its free blocks included.
    pub const fn data_segment_range(&self) -> FileRange {
        self.data_segment
    }

    /// Returns the data segment in `region`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`data_segment_range`](Self::data_segment_range).
    pub fn data_segment<'a>(
        &self,
        region: FileRegion<'a>,
    ) -> Result<LocalHeapDataSegment<'a>, FormatError> {
        Ok(LocalHeapDataSegment {
            bytes: region.subregion(self.data_segment)?.bytes(),
        })
    }
}

/// The bytes of a local heap's data segment, from [`LocalHeap::data_segment`].
///
/// The segment holds the null-terminated names of a group's links, and [`name`](Self::name)
/// parses the name at the offset a symbol table entry stores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalHeapDataSegment<'a> {
    bytes: &'a [u8],
}

impl<'a> LocalHeapDataSegment<'a> {
    /// Parses the name at `offset` in the data segment, up to the first NUL byte at or after
    /// `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLocalHeapNameOffset`] if `offset` is at or past the end of
    /// the data segment, [`FormatError::UnterminatedLocalHeapName`] if the data segment ends before
    /// a NUL byte, and [`FormatError::InvalidLocalHeapName`] if the name is not UTF-8.
    pub fn name(&self, offset: u64) -> Result<&'a str, FormatError> {
        let rest = offset
            .to_usize()
            .ok()
            .and_then(|start| self.bytes.get(start..))
            .filter(|rest| !rest.is_empty())
            .ok_or(FormatError::InvalidLocalHeapNameOffset {
                offset,
                data_segment_size: self.bytes.len().to_u64(),
            })?;
        let name = rest
            .iter()
            .position(|&byte| byte == 0)
            .and_then(|end| rest.get(..end))
            .ok_or(FormatError::UnterminatedLocalHeapName { offset })?;
        core::str::from_utf8(name)
            .map_err(|source| FormatError::InvalidLocalHeapName { offset, source })
    }
}

fn header_len(FormatWidths { offsets, lengths }: FormatWidths) -> u64 {
    PREFIX_LEN + 2 * u64::from(lengths.get()) + u64::from(offsets.get())
}

const SIGNATURE: [u8; 4] = *b"HEAP";

const VERSION: u8 = 0;

const RESERVED_LEN: usize = 3;

const PREFIX_LEN: u64 = (SIGNATURE.len() + 1 + RESERVED_LEN) as u64;

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use rstest::rstest;
    use test_util::local_heap;
    use test_util::local_heap::Segment;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::StoredAddressExt;
    use crate::fixture;

    fn header(widths: Widths, data_segment_address: u64, names: &[&str]) -> Vec<u8> {
        Segment::of_names(data_segment_address, names).header(widths)
    }

    fn locator(widths: Widths, base: u64, address: u64) -> LocalHeapLocator {
        LocalHeapLocator::new(
            fixture::format_widths(widths),
            BaseAddress::new(base),
            StoredAddress::new(address),
        )
    }

    fn data_segment(names: &[&str]) -> Vec<u8> {
        Segment::of_names(0, names).bytes
    }

    #[rstest]
    fn the_header_locates_its_data_segment(
        #[values(
            Widths::new(2, 2),
            Widths::new(2, 8),
            Widths::FOUR,
            Widths::new(8, 4),
            Widths::EIGHT
        )]
        widths: Widths,
        #[values(0, 512, 5 << 32)] base: u64,
        #[values(true, false)] follows_header: bool,
    ) {
        let header_at = 0x60;
        let header_len = local_heap::header_len(widths) as u64;
        let data_segment_at = if follows_header {
            header_at + header_len
        } else {
            0x400
        };
        let locator = locator(widths, base, header_at);
        let range = locator.parse_range().unwrap();
        assert_eq!(
            (range.start().get(), range.length()),
            (base + header_at, header_len)
        );
        let bytes = header(widths, data_segment_at, &["alpha", "beta"]);
        let heap = locator
            .parse(fixture::region(base + header_at, &bytes))
            .unwrap();
        let data_segment = heap.data_segment_range();
        assert_eq!(
            (data_segment.start().get(), data_segment.length()),
            (base + data_segment_at, 11)
        );
    }

    #[rstest]
    #[case::signature(0, b'X', FormatError::InvalidLocalHeapSignature)]
    #[case::version(4, 1, FormatError::InvalidLocalHeapVersion(1))]
    fn a_malformed_header_is_rejected(
        #[case] at: usize,
        #[case] byte: u8,
        #[case] expected: FormatError,
    ) {
        let mut bytes = header(Widths::EIGHT, 0x100, &["x"]);
        bytes[at] = byte;
        assert_eq!(
            locator(Widths::EIGHT, 0, 0)
                .parse(fixture::region(0, &bytes))
                .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::two(Widths::new(2, 2), 0xFFFF)]
    #[case::four(Widths::FOUR, 0xFFFF_FFFF)]
    #[case::eight(Widths::EIGHT, u64::MAX)]
    fn an_undefined_data_segment_address_is_rejected(
        #[case] widths: Widths,
        #[case] undefined: u64,
    ) {
        let bytes = header(widths, undefined, &["x"]);
        assert_eq!(
            locator(widths, 0, 0)
                .parse(fixture::region(0, &bytes))
                .unwrap_err(),
            FormatError::UndefinedLocalHeapDataSegment
        );
    }

    #[test]
    fn a_region_short_of_the_header_is_rejected() {
        let bytes = header(Widths::EIGHT, 0x100, &["x"]);
        assert_eq!(
            locator(Widths::EIGHT, 512, 0)
                .parse(fixture::region(512, &bytes[..31]))
                .unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 512,
                required_length: 32,
                provided_start: 512,
                provided_length: 31,
            }
        );
    }

    #[rstest]
    #[case::base(512, u64::MAX - 8, FormatError::OffsetOverflow { offset: u64::MAX - 8, length: 512 })]
    #[case::end(0, u64::MAX - 8, FormatError::OffsetOverflow { offset: u64::MAX - 8, length: 32 })]
    fn a_header_past_u64_max_is_rejected(
        #[case] base: u64,
        #[case] address: u64,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            locator(Widths::EIGHT, base, address)
                .parse_range()
                .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::base(512, u64::MAX - 0x100, FormatError::OffsetOverflow { offset: u64::MAX - 0x100, length: 512 })]
    #[case::end(0, u64::MAX - 1, FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 2 })]
    fn a_data_segment_past_u64_max_is_rejected(
        #[case] base: u64,
        #[case] data_segment_address: u64,
        #[case] expected: FormatError,
    ) {
        let bytes = header(Widths::EIGHT, data_segment_address, &["x"]);
        assert_eq!(
            locator(Widths::EIGHT, base, 0)
                .parse(fixture::region(base, &bytes))
                .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::first(0, "alpha")]
    #[case::second(6, "beta")]
    #[case::inside(2, "pha")]
    #[case::empty(5, "")]
    fn a_name_ends_at_the_first_nul_at_or_after_its_offset(
        #[case] offset: u64,
        #[case] expected: &str,
    ) {
        let bytes = data_segment(&["alpha", "beta"]);
        let heap = LocalHeap {
            data_segment: FileRange::new(PhysicalAddress::new(5 << 32), 11).unwrap(),
        };
        assert_eq!(
            heap.data_segment(fixture::region(5 << 32, &bytes))
                .unwrap()
                .name(offset)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn a_data_segment_is_taken_only_from_a_region_that_contains_it() {
        let mut bytes = vec![0xAA; 8];
        bytes.extend_from_slice(&data_segment(&["alpha"]));
        bytes.extend_from_slice(&[0xBB; 8]);
        let heap = LocalHeap {
            data_segment: FileRange::new(PhysicalAddress::new(0x108), 6).unwrap(),
        };
        assert_eq!(
            heap.data_segment(fixture::region(0x100, &bytes))
                .unwrap()
                .name(0)
                .unwrap(),
            "alpha"
        );
        assert_eq!(
            heap.data_segment(fixture::region(0x109, &bytes[9..]))
                .unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 0x108,
                required_length: 6,
                provided_start: 0x109,
                provided_length: 13,
            }
        );
    }

    #[rstest]
    #[case::at_the_end(11)]
    #[case::past_the_end(12)]
    #[case::past_usize_max(u64::MAX)]
    fn a_name_offset_outside_the_data_segment_is_rejected(#[case] offset: u64) {
        let bytes = data_segment(&["alpha", "beta"]);
        let heap = LocalHeap {
            data_segment: FileRange::new(PhysicalAddress::new(0), 11).unwrap(),
        };
        assert_eq!(
            heap.data_segment(fixture::region(0, &bytes))
                .unwrap()
                .name(offset)
                .unwrap_err(),
            FormatError::InvalidLocalHeapNameOffset {
                offset,
                data_segment_size: 11,
            }
        );
    }

    #[rstest]
    #[case::whole(0)]
    #[case::inside(3)]
    fn a_name_without_a_terminator_in_the_data_segment_is_rejected(#[case] offset: u64) {
        let bytes = b"alpha\0beta";
        let heap = LocalHeap {
            data_segment: FileRange::new(PhysicalAddress::new(0), 10).unwrap(),
        };
        let segment = heap.data_segment(fixture::region(0, bytes)).unwrap();
        assert_eq!(segment.name(0).unwrap(), "alpha");
        assert_eq!(
            segment.name(6 + offset).unwrap_err(),
            FormatError::UnterminatedLocalHeapName { offset: 6 + offset }
        );
    }

    #[test]
    fn a_name_that_is_not_utf8_is_rejected() {
        let mut bytes = data_segment(&["hello"]);
        bytes[2] = 0xFF;
        let heap = LocalHeap {
            data_segment: FileRange::new(PhysicalAddress::new(0), 6).unwrap(),
        };
        let err = heap
            .data_segment(fixture::region(0, &bytes))
            .unwrap()
            .name(0)
            .unwrap_err();
        let FormatError::InvalidLocalHeapName { offset, source } = err else {
            panic!("expected InvalidLocalHeapName, got {err:?}");
        };
        assert_eq!((offset, source.valid_up_to()), (0, 2));
    }
}
