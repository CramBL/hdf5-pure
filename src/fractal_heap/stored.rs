//! Reads fractal heaps through stored sources and resident file bytes.

use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::AddressRange;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::FractalHeapHeaderLocator;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdView;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::StoredBytes;
use crate::source::StoredRead;
use crate::width::FormatWidths;

use super::Backend;
use super::FractalHeapHeader;
use super::HugeObjectReadLimits;
use super::LegacyHeapObjectReader;

/// A source for heap metadata and separately allocated huge-object bytes.
///
/// The reader preserves physical ranges, borrows resident bytes, and retains any shared ownership
/// returned by [`StoredRead::read_metadata`].
pub(crate) struct HeapSource<'a> {
    backing: Backing<'a>,
}

impl<'a> HeapSource<'a> {
    /// Borrows a stored view and preserves its metadata read operations.
    pub(crate) fn new(source: &'a dyn StoredRead) -> Self {
        Self {
            backing: Backing::Stored(source),
        }
    }

    /// Creates a heap source that borrows resident file bytes.
    pub(crate) fn resident(source: StoredBytes<'a>) -> Self {
        Self {
            backing: Backing::Resident(source),
        }
    }

    fn origin(&self) -> Result<PhysicalAddress, FormatError> {
        match self.backing {
            Backing::Resident(source) => source.metadata_origin(),
            Backing::Stored(source) => source.metadata_origin(),
        }
    }

    /// Reads heap metadata at physical file positions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if a resident region is incomplete,
    /// [`FormatError::AddressBelowBase`] if a source range starts below its metadata origin,
    /// and errors from [`StoredRead::metadata_origin`] or [`StoredRead::read_metadata`].
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if a source read exceeds [`usize::MAX`].
    fn read_range(&self, range: FileRange) -> Result<MetadataBytes<'_>, FormatError> {
        match self.backing {
            Backing::Resident(source) => {
                let frame = source.metadata_frame();
                let region = FileRegion::new(frame.range(), frame.bytes())?.subregion(range)?;
                MetadataBytes::borrowed(region.range(), region.bytes())
            }
            Backing::Stored(source) => {
                let address =
                    BaseAddress::new(source.metadata_origin()?.get()).relative(range.start())?;
                source.read_metadata(address, range.length().to_usize()?)
            }
        }
    }

    /// Reads a huge-object allocation at physical file positions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if a resident allocation is incomplete,
    /// and address and source errors from [`Self::read_range`].
    fn read_huge_range(&self, range: FileRange) -> Result<MetadataBytes<'_>, FormatError> {
        match self.backing {
            Backing::Resident(source) => {
                let address =
                    BaseAddress::new(source.metadata_origin()?.get()).relative(range.start())?;
                MetadataBytes::borrowed(
                    range,
                    source.borrow_range(AddressRange::new(address, range.length())?)?,
                )
            }
            Backing::Stored(_) => self.read_range(range),
        }
    }
}

enum Backing<'a> {
    Resident(StoredBytes<'a>),
    Stored(&'a dyn StoredRead),
}

/// Reads a heap header and returns the physical range of its encoded bytes.
///
/// Resolves `address` against the source's metadata origin. The encoded range includes any
/// filter fields stored in the header.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if an address or range end exceeds [`u64::MAX`],
/// [`FormatError::InsufficientRegion`] if a resident region is incomplete, and errors from
/// [`FractalHeapHeaderLocator::parse`],
/// [`FractalHeapHeaderRef::parse`](hdf5_pure_format::__private::FractalHeapHeaderRef::parse),
/// or a source metadata read.
pub(crate) fn read_header_and_range_from_source(
    source: &HeapSource<'_>,
    address: StoredAddress,
    widths: FormatWidths,
) -> Result<(FractalHeapHeader, FileRange), FormatError> {
    let origin = source.origin()?;
    let locator = FractalHeapHeaderLocator::new(
        BaseAddress::new(origin.get()).resolve(address)?,
        widths.offsets,
        widths.lengths,
    );
    let prefix = source.read_range(locator.parse_range()?)?;
    let reference = locator.parse(prefix.region())?;
    let bytes = source.read_range(reference.parse_range())?;
    Ok((reference.parse(bytes.region())?, reference.range()))
}

/// Reads objects from one heap through a fixed source.
///
/// The caller parses each [`FractalHeapIdView`] using the decoding parameters of the heap passed
/// to [`Self::new`]. The source must stay unchanged for the reader's lifetime. The reader caches
/// huge-object index records, successful direct-block checksum validation, and the most recently
/// decoded filtered direct block.
///
/// For unfiltered managed objects, read-only lookups require the block prefix and object bytes.
/// Read-write lookups require complete validation of checksummed direct blocks. For filtered
/// objects, the reader decodes the complete direct block or huge-object allocation before
/// returning its bytes. Tiny-object bytes are copied from the parsed ID.
pub(crate) struct HeapObjectReader<'a, 'h> {
    source: HeapSource<'a>,
    reader: LegacyHeapObjectReader<'h>,
}

impl<'a, 'h> HeapObjectReader<'a, 'h> {
    /// Binds the heap's object reader to one source.
    ///
    /// `owner` is the header's stored address. `access_mode` selects read-only lookup or
    /// read-write checksum validation. Huge-object reads have [`HugeObjectReadLimits::UNLIMITED`].
    pub(crate) fn new(
        source: HeapSource<'a>,
        header: &'h FractalHeapHeader,
        owner: StoredAddress,
        access_mode: AccessMode,
        widths: FormatWidths,
    ) -> Self {
        Self::with_huge_limits(
            source,
            header,
            owner,
            access_mode,
            widths,
            HugeObjectReadLimits::UNLIMITED,
        )
    }

    /// Creates a reader with caller-imposed huge-object payload and tracker limits.
    ///
    /// The source and access requirements are those of [`Self::new`]. The reader checks payload
    /// sizes before reading their bytes and tracker limits during index traversal.
    pub(crate) fn with_huge_limits(
        source: HeapSource<'a>,
        header: &'h FractalHeapHeader,
        owner: StoredAddress,
        access_mode: AccessMode,
        widths: FormatWidths,
        limits: HugeObjectReadLimits,
    ) -> Self {
        let origin = source.origin().map(|origin| BaseAddress::new(origin.get()));
        let reader =
            LegacyHeapObjectReader::with_origin(header, origin, owner, access_mode, widths, limits);
        Self { source, reader }
    }

    /// Returns the object's bytes, resolving managed and huge objects through the bound source.
    ///
    /// Tiny objects require only the bytes in `id`, including when the source's physical origin
    /// cannot be represented.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::MissingFractalHeapBlock`] for an unallocated managed block,
    /// [`FormatError::HugeObjectNotFound`] for an absent huge-object key, and address, parsing,
    /// filter, and resource errors from [`LegacyHeapObjectReader::read`]. Returns the source
    /// error if a metadata read fails.
    pub(crate) fn read(&mut self, id: FractalHeapIdView<'_>) -> Result<Vec<u8>, FormatError> {
        let Self { source, reader } = self;
        match id.kind() {
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } => reader.read_managed_object_with(
                |range| source.read_range(range),
                Backend::Streaming,
                heap_offset,
                object_length,
            ),
            FractalHeapIdKind::Huge(_) => reader.read_huge_with(
                |range| source.read_huge_range(range),
                Backend::Streaming,
                id,
            ),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::address::AddressContext;
    use crate::address::StoredAddressExt;
    use crate::source::BytesSource;
    #[cfg(feature = "std")]
    use crate::source::MetadataCacheConfig;
    #[cfg(feature = "std")]
    use crate::source::MetadataCachingSource;
    #[cfg(feature = "std")]
    use crate::source::PhysicalBytes;
    use crate::source::PhysicalSource;
    use crate::source::Source;
    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

    use super::*;

    #[cfg(feature = "std")]
    #[rstest]
    #[case::resident(true, false, false)]
    #[case::borrowed(false, false, false)]
    #[case::cached_borrowed(false, false, true)]
    #[case::owned(false, true, false)]
    #[case::cached_owned(false, true, true)]
    fn metadata_reads_preserve_physical_ranges_borrowing_and_cache_owners(
        #[case] resident: bool,
        #[case] owned: bool,
        #[case] cache: bool,
        #[values(0, 512)] base: u64,
    ) {
        let bytes = vec![23; base.to_usize().unwrap() + 64];
        let borrowed = BytesSource::new(&bytes);
        let allocating = AllocatingSource {
            bytes: &bytes,
            origin: 37,
        };
        let source: &dyn Source = if owned { &allocating } else { &borrowed };
        let cached = MetadataCachingSource::new(
            source,
            if cache {
                MetadataCacheConfig::new(64)
            } else {
                MetadataCacheConfig::disabled()
            },
        );
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
        let stored = PhysicalSource::for_test(&cached, addresses).stored();
        let source = if resident {
            HeapSource::resident(PhysicalBytes::for_test(&bytes, addresses).stored().unwrap())
        } else {
            HeapSource::new(&stored)
        };
        let range = FileRange::new(
            PhysicalAddress::new(base + 13 + if owned { 37 } else { 0 }),
            3,
        )
        .unwrap();
        let first = source.read_range(range).unwrap();
        let second = source.read_range(range).unwrap();
        assert_eq!(first.range(), range);
        assert_eq!(second.range(), range);
        assert_eq!(first.as_ref(), [23; 3]);
        assert_eq!(second.as_ref(), [23; 3]);
        assert_eq!(first.shared().is_some(), owned);
        if !owned {
            assert_eq!(
                first.as_ptr(),
                bytes[base.to_usize().unwrap() + 13..].as_ptr()
            );
        } else if cache {
            assert_eq!(first.as_ptr(), second.as_ptr());
        }
        if cache {
            let stats = stored.metadata_cache_stats().unwrap();
            assert_eq!(
                (stats.hits(), stats.misses(), stats.entries(), stats.bytes()),
                if owned { (1, 1, 1, 3) } else { (0, 2, 0, 0) },
            );
        } else {
            assert_eq!(stored.metadata_cache_stats(), None);
        }
        let retained = first.shared();
        drop(first);
        drop(second);
        drop(cached);
        if let Some(retained) = retained {
            assert_eq!(retained.as_ref(), [23; 3]);
            assert_eq!(retained.range(), range);
        }
    }

    #[rstest]
    #[case::effective_origin(u64::MAX, 1, 0, FormatError::OffsetOverflow { offset: u64::MAX, length: 1 })]
    #[case::header_address(37, 512, u64::MAX, FormatError::OffsetOverflow { offset: u64::MAX, length: 549 })]
    #[case::prefix_end(u64::MAX - 5, 0, 0, FormatError::OffsetOverflow { offset: u64::MAX - 5, length: 9 })]
    fn header_reads_report_physical_origin_and_locator_overflows(
        #[case] origin: u64,
        #[case] base: u64,
        #[case] address: u64,
        #[case] error: FormatError,
    ) {
        let source = AllocatingSource { bytes: &[], origin };
        let stored = PhysicalSource::for_test(
            &source,
            AddressContext::from_effective_base(BaseAddress::new(base)),
        )
        .stored();
        assert_eq!(
            read_header_and_range_from_source(
                &HeapSource::new(&stored),
                StoredAddress::new(address),
                WIDTHS,
            )
            .unwrap_err(),
            error,
        );
    }

    struct AllocatingSource<'a> {
        bytes: &'a [u8],
        origin: u64,
    }

    impl Source for AllocatingSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len().to_u64()
        }
        fn metadata_origin(&self) -> Result<u64, FormatError> {
            Ok(self.origin)
        }
        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
            BytesSource::new(self.bytes).read_at(offset, buffer)
        }
    }

    const WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
}
