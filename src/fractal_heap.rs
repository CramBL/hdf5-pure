//! Resolves managed, huge, and tiny objects stored in a fractal heap.
//!
//! [`HeapObjectReader`] reads from either an in-memory file image or a [`Source`].

use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::num::NonZeroU32;

use hdf5_pure_core::__private::{FileRange, FileRegion, PhysicalAddress};
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_filter::__private::Error as FilterError;
pub use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::{
    BTreeV2Layout, FractalHeapBlock, FractalHeapBlockFilters, FractalHeapBlockLayout,
    FractalHeapHeaderLocator, FractalHeapHugeObjectId, FractalHeapHugeObjectLayout,
    FractalHeapHugeObjectRef, FractalHeapIdKind, FractalHeapIdView, HugeIndirectKeyEncoding,
};

use crate::access_mode::AccessMode;
use crate::address::{AddressContext, StoredAddress, StoredAddressExt};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::filter_pipeline::{FILTER_SHUFFLE, FilterPipeline};
use crate::filters;
use crate::filters::ChunkContext;
use crate::source;
use crate::source::Source;
use crate::width::FormatWidths;
#[cfg(test)]
use crate::width::{LengthWidth, OffsetWidth};

/// Parses a heap header using validated file widths.
///
/// # Errors
///
/// Returns [`FormatError::InsufficientRegion`] if the image is truncated, and the address and header
/// parsing errors of [`read_header_from_source`].
pub(crate) fn parse_header(
    bytes: &[u8],
    addresses: AddressContext,
    address: StoredAddress,
    widths: FormatWidths,
) -> Result<FractalHeapHeader, FormatError> {
    let image = FileRegion::new(
        FileRange::new(
            addresses.resolve(StoredAddress::new(0))?,
            bytes.len() as u64,
        )?,
        bytes,
    )?;
    let locator =
        FractalHeapHeaderLocator::new(addresses.resolve(address)?, widths.offsets, widths.lengths);
    let prefix = image.subregion(locator.parse_range()?)?;
    let reference = locator.parse(prefix)?;
    reference.parse(image.subregion(reference.parse_range())?)
}

/// Reads a fractal heap header from a source using validated file widths.
///
/// `source` is framed at the effective base in `addresses`. The parser receives physical
/// [`FileRegion`] coordinates, and each source read uses a base-relative offset.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if an address or range end exceeds [`u64::MAX`], and
/// the parsing errors of [`FractalHeapHeaderLocator::parse`] and
/// [`FractalHeapHeaderRef::parse`](hdf5_pure_format::__private::FractalHeapHeaderRef::parse).
/// Returns the source error if a read fails.
pub(crate) fn read_header_from_source(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    address: StoredAddress,
    widths: FormatWidths,
) -> Result<FractalHeapHeader, FormatError> {
    read_header_and_range_from_source(source, addresses, address, widths).map(|(header, _)| header)
}

/// Reads a fractal heap header from a source, with the physical range of its encoded bytes.
///
/// The range includes the header's filter fields when it stores them.
///
/// # Errors
///
/// Returns the errors of [`read_header_from_source`].
pub(crate) fn read_header_and_range_from_source(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    address: StoredAddress,
    widths: FormatWidths,
) -> Result<(FractalHeapHeader, FileRange), FormatError> {
    let locator =
        FractalHeapHeaderLocator::new(addresses.resolve(address)?, widths.offsets, widths.lengths);
    let prefix_range = locator.parse_range()?;
    let prefix = read_header_range(source, addresses, prefix_range)?;
    let header_ref = locator.parse(FileRegion::new(prefix_range, &prefix)?)?;
    let range = header_ref.parse_range();
    let bytes = read_header_range(source, addresses, range)?;
    Ok((
        header_ref.parse(FileRegion::new(range, &bytes)?)?,
        header_ref.range(),
    ))
}

/// Reads a physical header range through a source framed at `addresses`' effective base.
///
/// # Errors
///
/// Returns [`FormatError::AddressBelowBase`] if the range starts below the effective base,
/// [`FormatError::ValueTooLargeForPlatform`] if the read length exceeds [`usize::MAX`], and the
/// source error if a read fails.
pub(crate) fn read_header_range(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    range: FileRange,
) -> Result<MetadataBytes<'_>, FormatError> {
    source.read_metadata(
        addresses.relative(range.start().get())?.get(),
        range.length().to_usize()?,
    )
}

/// Reverses the active filters on a direct block's encoded bytes.
///
/// The block is decoded as one chunk of `decoded_size` one-byte elements.
///
/// # Errors
///
/// Returns [`FormatError::UnsupportedFilter`] if an active filter has no decoder or an active
/// shuffle filter has an element width other than one byte,
/// [`FormatError::DataSizeMismatch`] if the decoded block is not `decoded_size` bytes long, and a
/// filter-specific [`FormatError`] if the encoded bytes are malformed.
pub(crate) fn decode_direct_block(
    encoded: &[u8],
    filters: FractalHeapBlockFilters<'_>,
    decoded_size: u64,
) -> Result<Vec<u8>, FormatError> {
    decode_heap_payload(encoded, filters.pipeline, filters.mask, decoded_size)
        .map_err(FormatError::from)
}

/// Reverses heap filters using a one-byte element context.
///
/// # Errors
///
/// Returns the errors of [`filters::decompress_chunk`], or [`FilterError::UnsupportedFilter`]
/// for an active shuffle whose element width differs from one byte.
fn decode_heap_payload(
    encoded: &[u8],
    pipeline: &FilterPipeline,
    mask: u32,
    decoded_size: u64,
) -> Result<Vec<u8>, FilterError> {
    // `H5Z__filter_shuffle` in `H5Zshuffle.c`, HDF5 2.2.0, shuffles at the width in its first
    // client value. `filters::decompress_chunk` reverses a shuffle at the element size of its
    // context, one byte here, so an active shuffle filter of another width is rejected.
    let wide_shuffle = pipeline.filters.iter().enumerate().any(|(index, filter)| {
        let skipped = u32::try_from(index)
            .ok()
            .and_then(|bit| mask.checked_shr(bit))
            .is_some_and(|mask| mask & 1 == 1);
        filter.filter_id == FILTER_SHUFFLE && !skipped && filter.client_data != [1]
    });
    if wide_shuffle {
        return Err(FilterError::UnsupportedFilter(FILTER_SHUFFLE));
    }
    filters::decompress_chunk(
        encoded,
        pipeline,
        ChunkContext {
            chunk_dims: &[decoded_size],
            element_size: NonZeroU32::MIN,
            element_type: None,
            scale_offset_type: None,
        },
        mask,
    )
}

/// Cached numeric huge-object keys and their owned payload references.
struct HugeObjectIndex {
    entries: Vec<(u64, FractalHeapHugeObjectRef)>,
}

/// Caller-imposed payload and tracker limits for one huge-object read.
#[derive(Clone, Copy)]
pub(crate) struct HugeObjectReadLimits {
    /// The maximum encoded payload length in bytes.
    pub(crate) encoded_bytes: u64,
    /// The maximum declared decoded payload length in bytes.
    pub(crate) decoded_bytes: u64,
    /// The maximum number of tracker records.
    pub(crate) index_records: u64,
    /// The maximum number of tracker nodes visited or pending traversal.
    pub(crate) index_nodes: u64,
    /// The maximum total length of tracker metadata reads in bytes.
    pub(crate) metadata_bytes: u64,
}

impl HugeObjectReadLimits {
    fn require(limit: u64, required: u64) -> Result<(), FormatError> {
        if required > limit {
            return Err(FormatError::ResourceLimit { limit, required });
        }
        Ok(())
    }
    /// The maximum representable count for each resource.
    pub(crate) const UNLIMITED: Self = Self {
        encoded_bytes: u64::MAX,
        decoded_bytes: u64::MAX,
        index_records: u64::MAX,
        index_nodes: u64::MAX,
        metadata_bytes: u64::MAX,
    };
}

#[cfg(test)]
std::thread_local! {
    /// How many huge-object indexes have been decoded on this thread.
    ///
    /// Parsing one index per object rather than one per walk returns exactly the
    /// same objects while making the walk quadratic in their number, so the
    /// count of decodes is the only thing a test can hold the invariant to. The
    /// walks build their reader internally, so the count has to be reachable
    /// from outside the reader that did the decoding; it is per-thread because
    /// the test harness runs tests concurrently, and each test's walk runs on
    /// its own thread.
    static HUGE_INDEX_DECODES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Start counting huge-object index decodes on this thread from zero.
#[cfg(test)]
pub(crate) fn reset_huge_index_decodes() {
    HUGE_INDEX_DECODES.with(|count| count.set(0));
}

/// How many huge-object indexes this thread has decoded since the last reset.
#[cfg(test)]
pub(crate) fn huge_index_decodes() -> usize {
    HUGE_INDEX_DECODES.with(|count| count.get())
}

impl HugeObjectIndex {
    fn decode<'bytes>(
        read: &impl Fn(FileRange) -> Result<MetadataBytes<'bytes>, FormatError>,
        addresses: AddressContext,
        layout: &FractalHeapHugeObjectLayout,
        widths: FormatWidths,
        limits: HugeObjectReadLimits,
    ) -> Result<Self, FormatError> {
        let range = layout.tracker_range();
        HugeObjectReadLimits::require(limits.metadata_bytes, range.length())?;
        let bytes = read(range)?;
        let tracker = layout.parse_tracker(FileRegion::new(range, &bytes)?)?;
        let header = tracker.header();
        HugeObjectReadLimits::require(limits.index_records, header.total_records)?;
        header.total_records.to_usize()?;
        let nodes = BTreeV2Layout::new(header, widths.offsets)?;
        let mut pending: Vec<_> = nodes.root().into_iter().collect();
        let mut visited = BTreeSet::new();
        let mut metadata_bytes = range.length();
        let mut entries = Vec::new();
        while let Some(node) = pending.pop() {
            if !visited.insert(node.address()) {
                return Err(FormatError::InvalidBTreeV2Signature);
            }
            HugeObjectReadLimits::require(limits.index_nodes, visited.len().to_u64())?;
            let range = FileRange::new(
                addresses.resolve(node.address())?,
                node.encoded_len()?.to_u64(),
            )?;
            metadata_bytes =
                metadata_bytes
                    .checked_add(range.length())
                    .ok_or(FormatError::ResourceLimit {
                        limit: limits.metadata_bytes,
                        required: u64::MAX,
                    })?;
            HugeObjectReadLimits::require(limits.metadata_bytes, metadata_bytes)?;
            let bytes = read(range)?;
            let view = node.parse(&bytes)?;
            let count = entries
                .len()
                .to_u64()
                .checked_add(view.record_count().to_u64())
                .ok_or(FormatError::InvalidBTreeV2Signature)?;
            HugeObjectReadLimits::require(limits.index_records, count)?;
            if count > header.total_records {
                return Err(FormatError::InvalidBTreeV2Signature);
            }
            for index in 0..view.record_count() {
                let record = tracker.parse_record(view.record(index)?)?;
                entries.push((
                    record
                        .key()
                        .ok_or(FormatError::InvalidFractalHeapGeometry)?,
                    record.reference().clone(),
                ));
            }
            for index in 0..=view.record_count() {
                if let Some(child) = view.child(index)? {
                    HugeObjectReadLimits::require(
                        limits.index_nodes,
                        visited
                            .len()
                            .to_u64()
                            .saturating_add(pending.len().to_u64())
                            .saturating_add(1),
                    )?;
                    pending.push(child);
                }
            }
        }
        if entries.len().to_u64() != header.total_records {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        entries.sort_unstable_by_key(|(key, _)| *key);
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        #[cfg(test)]
        HUGE_INDEX_DECODES.with(|count| count.set(count.get() + 1));
        Ok(Self { entries })
    }

    fn locate(&self, huge_id: u64) -> Result<FractalHeapHugeObjectRef, FormatError> {
        self.entries
            .binary_search_by_key(&huge_id, |(id, _)| *id)
            .map(|index| self.entries[index].1.clone())
            .map_err(|_index| FormatError::HugeObjectNotFound(huge_id))
    }
}

/// Reads objects from one fractal heap using parsed heap IDs.
///
/// The caller parses each [`FractalHeapIdView`] using the decoding parameters of the heap passed
/// to [`new`](Self::new). The reader caches the huge-object B-tree records after the first indexed
/// huge-object lookup. Inline huge objects and tiny objects are read directly from their parsed
/// fields.
///
/// For read-only managed-object lookups, the reader validates the direct block's prefix and
/// object bounds and reads only the object bytes. With read-write access, the reader
/// validates a checksummed direct block in full before reading an object and caches successful
/// validation for subsequent objects in that block. Validation caches require an unchanged image
/// and one backend for the lifetime of the reader.
///
/// In a filtered heap, the reader reads and decodes the whole direct block that holds a managed
/// object, then validates the block's prefix and the object bounds in the decoded bytes. It keeps
/// the decoded block for the next lookup in that block, under the same requirements. With
/// read-write access, it validates a checksummed block in full on its decoded bytes.
///
/// For a filtered huge object, the reader reads the complete encoded allocation, reverses the
/// active filters, and checks the declared decoded size. Huge-object payloads are independent of
/// managed blocks. For indirect huge objects, the reader checks tracker metadata checksums when
/// checksum support is enabled.
pub struct HeapObjectReader<'h> {
    header: &'h FractalHeapHeader,
    widths: FormatWidths,
    addresses: AddressContext,
    access_mode: AccessMode,
    /// Direct blocks whose complete checksums were validated against this reader's image.
    validated_direct: BTreeSet<StoredAddress>,
    validated_backend: Option<Backend>,
    /// The stored address and the decoded bytes of the filtered direct block read last.
    decoded_direct: Option<(StoredAddress, Vec<u8>)>,
    layout: Result<FractalHeapBlockLayout, FormatError>,
    /// Parsed on demand, since most heaps hold no huge object at all.
    huge: Option<HugeObjectIndex>,
    /// Which backend the cached index was decoded from, once one has been.
    huge_backend: Option<Backend>,
    huge_layout: Option<FractalHeapHugeObjectLayout>,
    huge_limits: HugeObjectReadLimits,
}

/// Which of the two ways of reading a file an object came through. Recorded
/// only to hold a reader to one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// An in-memory image of the whole file.
    Buffered,
    /// A [`Source`], read from on demand.
    Streaming,
}

impl<'h> HeapObjectReader<'h> {
    /// Returns a reader of the objects of the heap `header` describes.
    ///
    /// `owner` is the header's stored address. `addresses` resolves addresses against the image's
    /// effective base. The image or source must be framed at that base and stay unchanged for
    /// the reader's lifetime. `access_mode` selects bounded read-only lookup or mandatory
    /// read-write checksum validation. `widths` contains the validated address and length widths
    /// of the file.
    pub(crate) fn new(
        header: &'h FractalHeapHeader,
        addresses: AddressContext,
        owner: StoredAddress,
        access_mode: AccessMode,
        widths: FormatWidths,
    ) -> Self {
        Self::with_huge_limits(
            header,
            addresses,
            owner,
            access_mode,
            widths,
            HugeObjectReadLimits::UNLIMITED,
        )
    }

    /// Creates a reader with caller-imposed huge-object payload and tracker limits.
    ///
    /// The file context and access requirements are those of [`new`](Self::new). The reader
    /// checks payload sizes before reading their bytes and tracker limits during index traversal.
    pub(crate) fn with_huge_limits(
        header: &'h FractalHeapHeader,
        addresses: AddressContext,
        owner: StoredAddress,
        access_mode: AccessMode,
        widths: FormatWidths,
        huge_limits: HugeObjectReadLimits,
    ) -> Self {
        Self {
            header,
            widths,
            addresses,
            access_mode,
            validated_direct: BTreeSet::new(),
            validated_backend: None,
            decoded_direct: None,
            layout: FractalHeapBlockLayout::new(header, widths.offsets, widths.lengths, owner),
            huge: None,
            huge_backend: None,
            huge_layout: None,
            huge_limits,
        }
    }

    /// Reads the object identified by a parsed ID from a file image.
    ///
    /// Managed objects are read through the heap's doubling table. Indexed huge objects are
    /// looked up in the cached huge-object index. Filtered huge-object payloads are decoded
    /// using the heap's pipeline and the object's skipped-filter mask. Tiny-object bytes are
    /// copied from `id`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::HugeObjectNotFound`] if an indexed huge object is absent, and
    /// [`FormatError::MissingFractalHeapBlock`] if a managed object has no allocated block.
    /// Returns [`FormatError::InsufficientRegion`] if a managed object's required region extends
    /// past the image, and [`FormatError::UnexpectedSourceEof`] if a huge object's data extends
    /// past the image. Returns filter-specific errors for a managed block or huge payload, such as
    /// [`FormatError::UnsupportedFilter`] if an active filter in its pipeline has no decoder.
    /// Returns [`FormatError::ChunkedReadError`] if indirect-block traversal exceeds 64 levels,
    /// [`FormatError::ValueTooLargeForPlatform`] if a value exceeds [`usize::MAX`] or a filtered
    /// huge payload's encoded or decoded size exceeds [`isize::MAX`], and
    /// [`FormatError::OffsetOverflow`] if a file offset overflows.
    /// Returns [`FormatError::ResourceLimit`] if a huge-object read or index traversal exceeds
    /// its configured limits, [`FormatError::DecodedSizeMismatch`] if a huge object's decoded
    /// length differs from its declared size, or [`FormatError::DecodedSizeLimitExceeded`] if a
    /// huge-object decoder exceeds the output bound derived from that size. Returns errors from
    /// parsing heap blocks and huge-object index metadata.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if cached checksum validation or a cached huge-object index is reused
    /// through both this method and [`read_from_source`](Self::read_from_source).
    pub fn read(
        &mut self,
        file_data: &[u8],
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        match id.kind() {
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } => self.read_managed_object(file_data, heap_offset, object_length),
            FractalHeapIdKind::Huge(_) => self.read_huge(file_data, id),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }

    /// Reads the object identified by a parsed ID through a [`Source`].
    ///
    /// # Errors
    ///
    /// Returns the resolution errors of [`read`](Self::read), and the error `source` returns if
    /// a read fails.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if cached checksum validation or a cached huge-object index is reused
    /// through both this method and [`read`](Self::read).
    pub fn read_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        match id.kind() {
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } => self.read_managed_object_from_source(source, heap_offset, object_length),
            FractalHeapIdKind::Huge(_) => self.read_huge_from_source(source, id),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }

    fn read_huge(
        &mut self,
        file_data: &[u8],
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        let addresses = self.addresses;
        self.read_huge_with(
            |range| {
                source::bytes_in(
                    file_data,
                    FileRange::new(
                        PhysicalAddress::new(addresses.relative(range.start().get())?.get()),
                        range.length(),
                    )?,
                )
                .and_then(|bytes| MetadataBytes::borrowed(range, bytes))
            },
            Backend::Buffered,
            id,
        )
    }

    fn read_huge_from_source(
        &mut self,
        source: &(impl Source + ?Sized),
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        let addresses = self.addresses;
        self.read_huge_with(
            |range| read_header_range(source, addresses, range),
            Backend::Streaming,
            id,
        )
    }

    fn read_huge_with<'bytes>(
        &mut self,
        read: impl Fn(FileRange) -> Result<MetadataBytes<'bytes>, FormatError>,
        backend: Backend,
        id: FractalHeapIdView<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        if self.huge_layout.is_none() {
            self.huge_layout = Some(FractalHeapHugeObjectLayout::new(
                self.header,
                self.widths,
                self.addresses.effective_base(),
                HugeIndirectKeyEncoding::Libhdf5,
            )?);
        }
        let layout = self.huge_layout.as_ref().ok_or_else(|| {
            FormatError::Internal("huge-object layout is initialized before use".into())
        })?;
        let reference = match layout.resolve_id(id)? {
            FractalHeapHugeObjectId::Direct(reference) => reference,
            FractalHeapHugeObjectId::Indirect(lookup) => {
                if self.huge.is_none() {
                    self.huge = Some(HugeObjectIndex::decode(
                        &read,
                        self.addresses,
                        layout,
                        self.widths,
                        self.huge_limits,
                    )?);
                    self.huge_backend = Some(backend);
                }
                debug_assert_eq!(
                    self.huge_backend,
                    Some(backend),
                    "a heap's huge-object index belongs to its reading backend"
                );
                self.huge
                    .as_ref()
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "huge-object index is initialized before lookup".into(),
                        )
                    })?
                    .locate(lookup.key().value())?
            }
        };
        HugeObjectReadLimits::require(self.huge_limits.encoded_bytes, reference.range().length())?;
        HugeObjectReadLimits::require(self.huge_limits.decoded_bytes, reference.decoded_size())?;
        let filters = reference.filters();
        if filters.is_some() {
            for value in [reference.range().length(), reference.decoded_size()] {
                isize::try_from(value).map_err(|_error| FormatError::ValueTooLargeForPlatform {
                    value,
                    target: "isize",
                })?;
            }
        }
        reference.range().length().to_usize()?;
        reference.decoded_size().to_usize()?;
        let encoded = read(reference.range())?;
        let encoded_bytes = reference.parse(FileRegion::new(reference.range(), &encoded)?)?;
        let decoded = if let Some(filters) = filters {
            decode_heap_payload(
                encoded_bytes,
                filters.pipeline,
                filters.mask,
                reference.decoded_size(),
            )
            .map_err(|error| match error {
                FilterError::DataSizeMismatch { actual, .. } => FormatError::DecodedSizeMismatch {
                    expected: reference.decoded_size(),
                    actual: actual.to_u64(),
                },
                FilterError::DeflateSizeExceeded { limit, required } => {
                    FormatError::DecodedSizeLimitExceeded {
                        expected_decoded_size: reference.decoded_size(),
                        filter_id: crate::filter_pipeline::FILTER_DEFLATE,
                        output_limit: limit.to_u64(),
                        minimum_output_size: required.to_u64(),
                    }
                }
                FilterError::LzfSizeExceeded { limit, required } => {
                    FormatError::DecodedSizeLimitExceeded {
                        expected_decoded_size: reference.decoded_size(),
                        filter_id: crate::filter_pipeline::FILTER_LZF,
                        output_limit: limit.to_u64(),
                        minimum_output_size: required.to_u64(),
                    }
                }
                other => FormatError::from(other),
            })?
        } else {
            encoded.into_vec()
        };
        layout.associate_decoded(&reference, &decoded)?;
        Ok(decoded)
    }

    fn read_managed_object(
        &mut self,
        file_data: &[u8],
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let image = FileRegion::new(
            FileRange::new(
                self.addresses.resolve(StoredAddress::ZERO)?,
                file_data.len() as u64,
            )?,
            file_data,
        )?;
        self.read_managed_object_with(
            |range| {
                image
                    .subregion(range)
                    .and_then(|region| MetadataBytes::borrowed(region.range(), region.bytes()))
            },
            Backend::Buffered,
            heap_offset,
            object_length,
        )
    }

    fn read_managed_object_from_source(
        &mut self,
        source: &(impl Source + ?Sized),
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let addresses = self.addresses;
        self.read_managed_object_with(
            |range| read_header_range(source, addresses, range),
            Backend::Streaming,
            heap_offset,
            object_length,
        )
    }

    fn read_managed_object_with<'bytes>(
        &mut self,
        read: impl Fn(FileRange) -> Result<MetadataBytes<'bytes>, FormatError>,
        backend: Backend,
        heap_offset: u64,
        object_length: u64,
    ) -> Result<Vec<u8>, FormatError> {
        if self.validated_direct.is_empty() && self.decoded_direct.is_none() {
            self.validated_backend = Some(backend);
        }
        debug_assert_eq!(
            self.validated_backend,
            Some(backend),
            "validated direct blocks belong to the backend that supplied their bytes"
        );
        let blocks = self.layout.as_ref().map_err(Clone::clone)?;
        let mut block = blocks.root()?.ok_or(FormatError::MissingFractalHeapBlock {
            heap_offset,
            object_length,
        })?;
        for _depth in 0..64 {
            match block {
                FractalHeapBlock::Direct(direct) => {
                    let reference = direct.at(self.addresses.resolve(direct.address())?)?;
                    let require_checksum = self.access_mode == AccessMode::ReadWrite
                        && self.header.flags
                            & hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED
                            != 0
                        && !self.validated_direct.contains(&direct.address());
                    if let Some(filters) = direct.filters() {
                        let decoded = match self.decoded_direct.take() {
                            Some((address, decoded)) if address == direct.address() => decoded,
                            _ => decode_direct_block(
                                &read(reference.range())?,
                                filters,
                                direct.decoded_size(),
                            )?,
                        };
                        if require_checksum {
                            reference.parse_decoded_checked(&decoded)?;
                            self.validated_direct.insert(direct.address());
                        }
                        let object = reference
                            .parse_decoded_object(&decoded, heap_offset, object_length)?
                            .to_vec();
                        self.decoded_direct = Some((direct.address(), decoded));
                        return Ok(object);
                    }
                    let range = if require_checksum {
                        reference.parse_range()
                    } else {
                        reference.object_prefix_range()?
                    };
                    let bytes = read(range)?;
                    let region = FileRegion::new(range, &bytes)?;
                    let object = reference.parse_object(region, heap_offset, object_length)?;
                    if require_checksum {
                        reference.parse_checked(region)?;
                        self.validated_direct.insert(direct.address());
                        return Ok(object.parse(region)?.to_vec());
                    }
                    return Ok(read(object.range())?.into_vec());
                }
                FractalHeapBlock::Indirect(indirect) => {
                    let reference = indirect.at(self.addresses.resolve(indirect.address())?)?;
                    let range = reference.parse_range();
                    let bytes = read(range)?;
                    let view = reference.parse(FileRegion::new(range, &bytes)?)?;
                    let mut selected = None;
                    for child in view.children() {
                        let child = child?;
                        let (start, length) = match child {
                            FractalHeapBlock::Direct(direct) => {
                                (direct.heap_offset(), direct.decoded_size())
                            }
                            FractalHeapBlock::Indirect(indirect) => (
                                indirect.heap_offset(),
                                blocks
                                    .layout()
                                    .indirect_heap_size(indirect.rows())
                                    .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?,
                            ),
                        };
                        if heap_offset >= start && heap_offset - start < length {
                            selected = Some(child);
                            break;
                        }
                    }
                    block = selected.ok_or(FormatError::MissingFractalHeapBlock {
                        heap_offset,
                        object_length,
                    })?;
                }
            }
        }
        Err(FormatError::ChunkedReadError(
            "fractal heap: maximum recursion depth exceeded".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;
    use hdf5_pure_core::__private::PipelineFilterMask;
    use hdf5_pure_format::__private::FilterDescription;
    use hdf5_pure_format::__private::FilterPipeline;
    use hdf5_pure_format::__private::FractalHeapFiltering;
    use hdf5_pure_format::__private::FractalHeapHeaderFrame;
    use hdf5_pure_format::__private::FractalHeapIdLayout;
    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;
    use test_util_hdf5::reader::ReadBackend;

    use crate::address::{BaseAddress, BaseAddressExt, OpenedImageContext};
    use crate::source::{BaseOffsetSource, BytesSource};
    use crate::test_data;

    use super::*;

    // "The Data Storage - Filter Pipeline Message", format specification version 4.0, defines
    // pipeline version 2.
    const PIPELINE_VERSION_TWO: u8 = 2;
    const FORMAT_WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    #[rstest]
    #[case::plain(0, 0)]
    #[case::userblock(512, 512)]
    #[case::relocated(512, 1024)]
    #[case::relocated_back(512, 0)]
    fn source_header_reads_use_the_effective_physical_base(
        #[case] encoded_base: u64,
        #[case] effective_base: u64,
        #[values(0, 3)] root_rows: u16,
        #[values(0, 2)] huge_objects: u64,
    ) {
        let mut header = FractalHeapHeader::parse_bytes(
            &fractal_heap::Header::new(256).build(Widths::EIGHT),
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .unwrap();
        header.current_rows_in_root_indirect_block = root_rows;
        header.huge_objects_count = huge_objects;
        header.huge_objects_size = huge_objects * 1024;
        let bytes = header
            .serialize(OffsetWidth::Eight, LengthWidth::Eight)
            .unwrap();
        let header_address = 13;
        let mut image = vec![0xFF; usize::try_from(effective_base + header_address).unwrap()];
        image.extend_from_slice(&bytes);
        let source = HeaderSource {
            bytes: &image,
            reads: RefCell::new(Vec::new()),
        };
        let addresses = OpenedImageContext::reconcile(
            BaseAddress::new(encoded_base),
            encoded_base + header_address + bytes.len() as u64,
            effective_base,
        )
        .unwrap()
        .addresses();
        let framed = BaseOffsetSource {
            inner: &source,
            base: addresses.effective_base(),
        };
        assert_eq!(
            read_header_from_source(
                &framed,
                addresses,
                StoredAddress::new(header_address),
                FORMAT_WIDTHS,
            )
            .unwrap(),
            header
        );
        assert_eq!(
            *source.reads.borrow(),
            vec![
                (
                    effective_base + header_address,
                    FractalHeapHeaderFrame::PREFIX_LEN
                ),
                (effective_base + header_address, bytes.len())
            ]
        );
    }

    #[rstest]
    #[case::prefix(8, 9)]
    #[case::header(145, 146)]
    fn a_source_short_read_remains_a_source_error(
        #[case] available: usize,
        #[case] expected: usize,
    ) {
        let mut bytes = fractal_heap::Header::new(256).build(Widths::EIGHT);
        bytes.truncate(available);
        assert_eq!(
            read_header_from_source(
                &BytesSource::new(&bytes),
                AddressContext::from_effective_base(BaseAddress::ZERO),
                StoredAddress::ZERO,
                FORMAT_WIDTHS,
            )
            .unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: expected.to_u64(),
                available: available.to_u64()
            }
        );
    }

    struct HeaderSource<'a> {
        bytes: &'a [u8],
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl Source for HeaderSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push((offset, bytes.len()));
            BytesSource::new(self.bytes).read_at(offset, bytes)
        }

        fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
            self.reads.borrow_mut().push((offset, length));
            BytesSource::new(self.bytes).read_metadata_at(offset, length)
        }
    }

    #[rstest]
    #[case::read_only(AccessMode::ReadOnly)]
    #[case::read_write(AccessMode::ReadWrite)]
    fn checksummed_object_reads_follow_the_access_mode(#[case] mode: AccessMode) {
        let mut image = fractal_heap::heap_with_one_object(b"abc", Widths::EIGHT);
        let mut header = FractalHeapHeader::parse(&image, 0, 8, 8).unwrap();
        header.flags = hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
        let at = header.root_block_address.get() as usize;
        image.resize(at + 128, 0);
        image[at + 15..at + 19].fill(0);
        image[at + 19..at + 22].copy_from_slice(b"abc");
        let checksum = crate::checksum::jenkins_lookup3(&image[at..at + 128]);
        image[at + 15..at + 19].copy_from_slice(&checksum.to_le_bytes());
        let source = HeaderSource {
            bytes: &image,
            reads: RefCell::new(Vec::new()),
        };
        let id = [0, 19, 0, 3, 0, 0, 0];
        let mut reader = HeapObjectReader::new(
            &header,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            mode,
            FORMAT_WIDTHS,
        );
        assert_eq!(
            reader
                .read_from_source(&source, parse_id(&header, &id))
                .unwrap(),
            b"abc"
        );
        assert_eq!(
            reader
                .read_from_source(&source, parse_id(&header, &id))
                .unwrap(),
            b"abc"
        );
        assert_eq!(
            *source.reads.borrow(),
            if mode == AccessMode::ReadWrite {
                vec![(at as u64, 128), (at as u64, 19), (at as u64 + 19, 3)]
            } else {
                vec![
                    (at as u64, 19),
                    (at as u64 + 19, 3),
                    (at as u64, 19),
                    (at as u64 + 19, 3),
                ]
            }
        );
    }

    #[test]
    fn short_source_reads_and_short_supplied_object_prefixes_are_distinct() {
        let image = fractal_heap::heap_with_one_object(b"abc", Widths::EIGHT);
        let header = FractalHeapHeader::parse(&image, 0, 8, 8).unwrap();
        let blocks = FractalHeapBlockLayout::new(
            &header,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            StoredAddress::ZERO,
        )
        .unwrap();
        let FractalHeapBlock::Direct(block) = blocks.root().unwrap().unwrap() else {
            panic!("expected direct root")
        };
        let reference = block
            .at(AddressContext::from_effective_base(BaseAddress::ZERO)
                .resolve(block.address())
                .unwrap())
            .unwrap();
        let range = reference.object_prefix_range().unwrap();
        let supplied = FileRange::new(range.start(), range.length() - 1).unwrap();
        let bytes = &image[range.start().get() as usize..supplied.end().get() as usize];
        assert_eq!(
            reference
                .parse_object(FileRegion::new(supplied, bytes).unwrap(), 15, 3)
                .unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: range.start().get(),
                required_length: range.length(),
                provided_start: supplied.start().get(),
                provided_length: supplied.length()
            }
        );
        let source = BytesSource::new(&image[..supplied.end().get() as usize]);
        let id = [0, 15, 0, 3, 0, 0, 0];
        assert_eq!(
            HeapObjectReader::new(
                &header,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                StoredAddress::ZERO,
                AccessMode::ReadOnly,
                FORMAT_WIDTHS
            )
            .read_from_source(&source, parse_id(&header, &id))
            .unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: range.end().get(),
                available: supplied.end().get()
            }
        );
    }

    #[rstest]
    #[case::tiny(vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c', 0, 0, 0, 0], false, b"abc")]
    #[case::eighteen_byte_tiny([vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c'], vec![0; 14]].concat(), false, b"abc")]
    #[case::filtered_tiny(vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c', 0, 0, 0, 0], true, b"abc")]
    #[case::inline_huge(vec![HUGE_ID_FIRST_BYTE, 0x20, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], false, b"abc")]
    #[case::issue_485([vec![0x20, 0x03, 0xAA, 0xBB, 0xCC, 0xDD], vec![0; 12]].concat(), false, &[0x03])]
    fn a_parsed_inline_id_resolves_its_object(
        #[case] bytes: Vec<u8>,
        #[case] filtered: bool,
        #[case] expected: &[u8],
        #[values(false, true)] streaming: bool,
    ) {
        let mut header = FractalHeapHeader::parse(
            &fractal_heap::Header::new(256).build(Widths::EIGHT),
            0,
            8,
            8,
        )
        .unwrap();
        header.heap_id_length = u16::try_from(bytes.len()).unwrap();
        header.btree_huge_objects_address = StoredAddress::new(0x800);
        if filtered {
            header.filtering = FractalHeapFiltering::Filtered {
                root_direct_block_size: 91,
                root_filter_mask: 0,
                pipeline: FilterPipeline {
                    version: PIPELINE_VERSION_TWO,
                    filters: vec![FilterDescription {
                        filter_id: crate::filter_pipeline::FILTER_DEFLATE,
                        name: None,
                        flags: 0,
                        client_data: vec![6],
                    }],
                },
            };
        }
        let id = parse_id(&header, &bytes);
        let mut image = vec![0; 64];
        image[0x20..0x23].copy_from_slice(b"abc");
        let mut reader = HeapObjectReader::new(
            &header,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        );
        let source = HeaderSource {
            bytes: &image,
            reads: RefCell::new(Vec::new()),
        };
        let object = if streaming {
            reader.read_from_source(&source, id)
        } else {
            reader.read(&image, id)
        }
        .unwrap();
        assert_eq!(object, expected);
        if matches!(id.kind(), FractalHeapIdKind::Tiny { .. }) {
            assert_eq!(*source.reads.borrow(), Vec::<(u64, usize)>::new());
        }
    }

    #[test]
    fn read_managed_object_from_direct_block() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

        // Build heap ID for the test data written in build_simple_heap.
        // The test data "Hello, World!" is at the data area of the direct block.
        // The direct block header is 5 + 8 + 2 = 15 bytes (for max_heap_size=16, ceil(16/8)=2).
        // Wait, max_heap_size=16, ceil(16/8)=2. Header = sig(4)+ver(1)+addr(8)+bo(2) = 15.
        // The data was placed at data_start = block_addr + 15.
        // Since offset is from block start, the object is at offset 15 within the block.
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8); // 15
        let offset: u64 = dblock_header_size as u64;
        let length: u64 = 13;
        let payload = offset | (length << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        id[0] = 0x00;
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let obj = HeapObjectReader::new(
            &hdr,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        )
        .read(&file_data, parse_id(&hdr, &id))
        .unwrap();
        assert_eq!(&obj, b"Hello, World!");
    }

    #[rstest]
    #[case::missing_root(false)]
    #[case::missing_child(true)]
    fn a_managed_object_without_a_block_reports_its_heap_coordinates(
        #[case] indirect: bool,
        #[values(false, true)] streaming: bool,
    ) {
        let mut heap = FilteredHeap::new(
            true,
            FilterPipeline {
                version: PIPELINE_VERSION_TWO,
                filters: Vec::new(),
            },
            |_block| {},
        );
        if !indirect {
            heap.header.root_block_address = StoredAddress::undefined(OffsetWidth::Eight);
            heap.header.current_rows_in_root_indirect_block = 0;
        }
        let mut reader = heap.reader(AccessMode::ReadOnly);
        let error = if streaming {
            reader.read_managed_object_from_source(&BytesSource::new(&heap.image), 15, 3)
        } else {
            reader.read_managed_object(&heap.image, 15, 3)
        }
        .unwrap_err();
        assert_eq!(
            error,
            FormatError::MissingFractalHeapBlock {
                heap_offset: 15,
                object_length: 3
            }
        );
        assert_eq!(
            error.to_string(),
            "no allocated fractal heap block contains object at heap offset 15 with length 3"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_managed_object_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);

        // Same heap ID as `read_managed_object_from_direct_block`.
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8);
        let payload = (dblock_header_size as u64) | (13u64 << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let buffered = HeapObjectReader::new(
            &hdr,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        )
        .read(&file_data, parse_id(&hdr, &id))
        .unwrap();

        // Header parsed from a source, then the object fetched from a source.
        let mem = BytesSource::new(&file_data);
        let hdr_mem = read_header_from_source(
            &mem,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            FORMAT_WIDTHS,
        )
        .unwrap();
        let from_mem = HeapObjectReader::new(
            &hdr_mem,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        )
        .read_from_source(&mem, parse_id(&hdr_mem, &id))
        .unwrap();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
        let hdr_seek = read_header_from_source(
            &seek,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            FORMAT_WIDTHS,
        )
        .unwrap();
        let from_seek = HeapObjectReader::new(
            &hdr_seek,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        )
        .read_from_source(&seek, parse_id(&hdr_seek, &id))
        .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(&from_seek, b"Hello, World!");
    }

    #[cfg(feature = "deflate")]
    #[rstest]
    #[case::direct_root(false)]
    #[case::indirect_root(true)]
    fn filtered_managed_objects_are_read_from_their_decoded_block(
        #[case] indirect: bool,
        #[values(AccessMode::ReadOnly, AccessMode::ReadWrite)] mode: AccessMode,
        #[values(
            test_data::deflate_pipeline(),
            test_data::filter_pipeline(vec![
                test_data::shuffle(1),
                test_data::deflate_pipeline().filters[0].clone(),
            ])
        )]
        pipeline: FilterPipeline,
    ) {
        let heap = FilteredHeap::new(indirect, pipeline, |_block| {});
        let source = HeaderSource {
            bytes: &heap.image,
            reads: RefCell::new(Vec::new()),
        };
        let id = heap.object_id();
        let mut reader = heap.reader(mode);
        for _ in 0..2 {
            assert_eq!(
                reader
                    .read_from_source(&source, parse_id(&heap.header, &id))
                    .unwrap(),
                b"abc"
            );
        }
        let block = (FILTERED_BLOCK_AT, heap.encoded_len);
        let root = (FILTERED_ROOT_AT, heap.indirect_len);
        // The reader reuses the decoded block on the second read and reads only the root indirect
        // block again.
        assert_eq!(
            *source.reads.borrow(),
            if indirect {
                vec![root, block, root]
            } else {
                vec![block]
            }
        );
        assert_eq!(
            heap.reader(mode)
                .read(&heap.image, parse_id(&heap.header, &id))
                .unwrap(),
            b"abc"
        );
    }

    #[cfg(feature = "deflate")]
    #[rstest]
    #[case::read_only(AccessMode::ReadOnly, false)]
    #[case::read_write(AccessMode::ReadWrite, true)]
    fn a_decoded_block_checksum_is_validated_for_read_write_access(
        #[case] mode: AccessMode,
        #[case] validated: bool,
    ) {
        let mut stored = 0;
        let mut computed = 0;
        let heap = FilteredHeap::new(true, test_data::deflate_pipeline(), |block| {
            let checksum =
                fractal_heap::DIRECT_BLOCK_CHECKSUM_AT..fractal_heap::DIRECT_BLOCK_DATA_AT;
            stored = u32::from_le_bytes(block[checksum.clone()].try_into().unwrap());
            *block.last_mut().unwrap() ^= 1;
            let mut zeroed = block.to_vec();
            zeroed[checksum].fill(0);
            computed = crate::checksum::jenkins_lookup3(&zeroed);
        });
        let id = heap.object_id();
        let result = heap
            .reader(mode)
            .read(&heap.image, parse_id(&heap.header, &id));
        if validated {
            assert_eq!(
                result.unwrap_err(),
                FormatError::ChecksumMismatch {
                    expected: stored,
                    computed
                }
            );
        } else {
            assert_eq!(result.unwrap(), b"abc");
        }
    }

    #[rstest]
    #[case::unknown_filter(
        FilterDescription {
            filter_id: test_data::FILTER_BZIP2,
            name: None,
            flags: 0,
            client_data: vec![],
        },
        test_data::FILTER_BZIP2
    )]
    #[case::wide_shuffle(test_data::shuffle(4), FILTER_SHUFFLE)]
    fn a_filtered_block_without_a_decoder_is_rejected(
        #[case] filter: FilterDescription,
        #[case] unsupported: u16,
    ) {
        let mut heap = FilteredHeap::new(false, test_data::filter_pipeline(vec![]), |_block| {});
        let FractalHeapFiltering::Filtered { pipeline, .. } = &mut heap.header.filtering else {
            panic!("the fixture heap is filtered");
        };
        *pipeline = test_data::filter_pipeline(vec![filter]);
        let id = heap.object_id();
        assert_eq!(
            heap.reader(AccessMode::ReadOnly)
                .read(&heap.image, parse_id(&heap.header, &id))
                .unwrap_err(),
            FormatError::UnsupportedFilter(unsupported)
        );
    }

    #[test]
    fn a_block_that_skips_a_wide_shuffle_is_read() {
        let mut heap = FilteredHeap::new(false, test_data::filter_pipeline(vec![]), |_block| {});
        let FractalHeapFiltering::Filtered {
            pipeline,
            root_filter_mask,
            ..
        } = &mut heap.header.filtering
        else {
            panic!("the fixture heap is filtered");
        };
        *pipeline = test_data::filter_pipeline(vec![test_data::shuffle(4)]);
        *root_filter_mask = 1;
        let id = heap.object_id();
        assert_eq!(
            heap.reader(AccessMode::ReadOnly)
                .read(&heap.image, parse_id(&heap.header, &id))
                .unwrap(),
            b"abc"
        );
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn a_filtered_block_that_decodes_to_another_size_is_rejected() {
        let mut heap = FilteredHeap::new(false, test_data::deflate_pipeline(), |_block| {});
        let decoded = vec![0; 127];
        let encoded = test_data::encode_block(&decoded, &test_data::deflate_pipeline());
        let at = usize::try_from(FILTERED_BLOCK_AT).unwrap();
        heap.image[at..at + encoded.len()].copy_from_slice(&encoded);
        let FractalHeapFiltering::Filtered {
            root_direct_block_size,
            ..
        } = &mut heap.header.filtering
        else {
            panic!("the fixture heap is filtered");
        };
        *root_direct_block_size = u64::try_from(encoded.len()).unwrap();
        let id = heap.object_id();
        assert_eq!(
            heap.reader(AccessMode::ReadOnly)
                .read(&heap.image, parse_id(&heap.header, &id))
                .unwrap_err(),
            FormatError::DataSizeMismatch {
                expected: 128,
                actual: 127
            }
        );
    }

    /// A heap whose one checksummed direct block holds `abc` and is encoded through a pipeline:
    /// the root block, or the second child of a root indirect block.
    struct FilteredHeap {
        header: FractalHeapHeader,
        image: Vec<u8>,
        block_offset: u64,
        #[cfg(feature = "deflate")]
        encoded_len: usize,
        #[cfg(feature = "deflate")]
        indirect_len: usize,
    }

    impl FilteredHeap {
        /// Encodes the block through `pipeline`, after applying `damage` to its decoded bytes.
        fn new(indirect: bool, pipeline: FilterPipeline, damage: impl FnOnce(&mut [u8])) -> Self {
            let block_offset = if indirect { FILTERED_BLOCK_SIZE } else { 0 };
            let mut block = fractal_heap::DIRECT_BLOCK_SIGNATURE.to_vec();
            block.push(fractal_heap::BLOCK_VERSION);
            block.extend_from_slice(&0u64.to_le_bytes());
            block.extend_from_slice(&u16::try_from(block_offset).unwrap().to_le_bytes());
            block.extend_from_slice(&[0; 4]);
            block.extend_from_slice(b"abc");
            block.resize(usize::try_from(FILTERED_BLOCK_SIZE).unwrap(), 0);
            let checksum = crate::checksum::jenkins_lookup3(&block);
            block[fractal_heap::DIRECT_BLOCK_CHECKSUM_AT..fractal_heap::DIRECT_BLOCK_DATA_AT]
                .copy_from_slice(&checksum.to_le_bytes());
            damage(&mut block);
            let encoded = test_data::encode_block(&block, &pipeline);
            let encoded_len = encoded.len();
            let mut image = vec![0; 0x400];
            let at = usize::try_from(FILTERED_BLOCK_AT).unwrap();
            image[at..at + encoded_len].copy_from_slice(&encoded);
            #[cfg(feature = "deflate")]
            let mut indirect_len = 0;
            if indirect {
                let mut root = fractal_heap::INDIRECT_BLOCK_SIGNATURE.to_vec();
                root.push(fractal_heap::BLOCK_VERSION);
                root.extend_from_slice(&0u64.to_le_bytes());
                root.extend_from_slice(&0u16.to_le_bytes());
                for column in 0..4 {
                    let (address, size) = if column == 1 {
                        (FILTERED_BLOCK_AT, u64::try_from(encoded_len).unwrap())
                    } else {
                        (u64::MAX, 0)
                    };
                    root.extend_from_slice(&address.to_le_bytes());
                    root.extend_from_slice(&size.to_le_bytes());
                    root.extend_from_slice(&0u32.to_le_bytes());
                }
                let checksum = crate::checksum::jenkins_lookup3(&root);
                root.extend_from_slice(&checksum.to_le_bytes());
                let root_len = root.len();
                #[cfg(feature = "deflate")]
                {
                    indirect_len = root_len;
                }
                let at = usize::try_from(FILTERED_ROOT_AT).unwrap();
                image[at..at + root_len].copy_from_slice(&root);
            }
            let header = FractalHeapHeader {
                flags: hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
                root_block_address: StoredAddress::new(if indirect {
                    FILTERED_ROOT_AT
                } else {
                    FILTERED_BLOCK_AT
                }),
                current_rows_in_root_indirect_block: u16::from(indirect),
                filtering: FractalHeapFiltering::Filtered {
                    root_direct_block_size: if indirect {
                        0
                    } else {
                        u64::try_from(encoded_len).unwrap()
                    },
                    root_filter_mask: 0,
                    pipeline,
                },
                ..FractalHeapHeader::parse(
                    &fractal_heap::Header::new(0).build(Widths::EIGHT),
                    0,
                    8,
                    8,
                )
                .unwrap()
            };
            Self {
                header,
                image,
                block_offset,
                #[cfg(feature = "deflate")]
                encoded_len,
                #[cfg(feature = "deflate")]
                indirect_len,
            }
        }

        fn object_id(&self) -> Vec<u8> {
            let mut id = vec![0];
            let object_at =
                self.block_offset + u64::try_from(fractal_heap::DIRECT_BLOCK_DATA_AT).unwrap();
            id.extend_from_slice(&u16::try_from(object_at).unwrap().to_le_bytes());
            id.extend_from_slice(&[3, 0, 0, 0]);
            id
        }

        fn reader(&self, mode: AccessMode) -> HeapObjectReader<'_> {
            HeapObjectReader::new(
                &self.header,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                StoredAddress::ZERO,
                mode,
                FORMAT_WIDTHS,
            )
        }
    }

    /// One reader parses the heap's huge-object index once, however many huge
    /// objects are read through it, on both backends.
    ///
    /// Parsing it per object returns exactly the same bytes while making a walk
    /// over the heap quadratic in the number of huge objects, so only the parse
    /// count separates the two.
    #[test]
    fn a_reader_parses_the_huge_object_index_once() {
        const COUNT: u64 = 5;

        // Attributes too large for a managed heap object, so the heap holds
        // nothing but huge ones.
        let mut builder = crate::FileBuilder::new();
        for i in 0..COUNT {
            builder.set_attr(
                &format!("a{i}"),
                crate::AttrValue::StringArray(vec![format!("{i:0700}"); 100]),
            );
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = builder.finish().unwrap();

        let frhp = bytes
            .windows(4)
            .position(|w| w == b"FRHP")
            .expect("dense storage writes a fractal heap");
        let heap = FractalHeapHeader::parse(&bytes, frhp, 8, 8).unwrap();

        // A huge heap ID: type 1 in bits 4-5 of byte 0, then the object's id,
        // little-endian, across the rest of the ID. The writer's own encoding,
        // which lets this reach the objects without walking the name index.
        let heap_id = |id: u64| {
            let mut bytes = vec![0u8; heap.heap_id_length as usize];
            bytes[0] = 0x10;
            for (i, slot) in bytes[1..].iter_mut().take(8).enumerate() {
                *slot = ((id >> (i * 8)) & 0xFF) as u8;
            }
            bytes
        };
        let ids: Vec<Vec<u8>> = (1..=COUNT).map(heap_id).collect();

        reset_huge_index_decodes();
        let mut reader = HeapObjectReader::new(
            &heap,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(frhp as u64),
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        );
        let buffered: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| reader.read(&bytes, parse_id(&heap, id)).unwrap())
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "buffered reads re-parsed the index"
        );

        reset_huge_index_decodes();
        let source = BytesSource::new(bytes.clone());
        let mut reader = HeapObjectReader::new(
            &heap,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(frhp as u64),
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        );
        let streamed: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| {
                reader
                    .read_from_source(&source, parse_id(&heap, id))
                    .unwrap()
            })
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "streaming reads re-parsed the index"
        );

        assert_eq!(buffered, streamed);
        assert_eq!(buffered.len(), COUNT as usize);
        for (i, object) in buffered.iter().enumerate() {
            assert!(
                object.windows(2).any(|w| w == format!("a{i}").as_bytes()),
                "object {i} is not the attribute message it should be"
            );
        }
    }

    #[rstest]
    #[case::deflate(test_data::deflate_pipeline(), 0)]
    #[case::checksum_then_deflate(test_data::filter_pipeline(vec![huge_fletcher(), test_data::deflate_pipeline().filters[0].clone()]), 0)]
    #[case::deflate_then_checksum(test_data::filter_pipeline(vec![test_data::deflate_pipeline().filters[0].clone(), huge_fletcher()]), 0)]
    #[case::skipped_unknown(test_data::filter_pipeline(vec![huge_unknown(), test_data::deflate_pipeline().filters[0].clone()]), 1)]
    #[case::skipped_wide_shuffle(test_data::filter_pipeline(vec![test_data::shuffle(4), test_data::deflate_pipeline().filters[0].clone()]), 1)]
    #[case::all_skipped(test_data::filter_pipeline(vec![huge_unknown(), test_data::shuffle(4)]), 3)]
    #[case::empty(test_data::filter_pipeline(vec![]), 0)]
    #[cfg(feature = "deflate")]
    fn filtered_huge_objects_read_exact_decoded_bytes(
        #[case] pipeline: FilterPipeline,
        #[case] mask: u32,
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
        #[values(0, 512)] base: u64,
    ) {
        let heap = HugeHeap::new(direct, Some(HugeFilters::new(pipeline, mask)), base);
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED).unwrap(),
            heap.decoded
        );
        let layout = FractalHeapHugeObjectLayout::new(
            &heap.header,
            FORMAT_WIDTHS,
            BaseAddress::new(base),
            HugeIndirectKeyEncoding::Libhdf5,
        )
        .unwrap();
        let reference = match layout.parse_id(&heap.id).unwrap() {
            FractalHeapHugeObjectId::Direct(reference) => reference,
            FractalHeapHugeObjectId::Indirect(lookup) => {
                let tracker = layout.parse_tracker(heap.region()).unwrap();
                let record = tracker.parse_record(&heap.record).unwrap();
                layout.resolve(&lookup, &record).unwrap()
            }
        };
        let filters = reference.filters().unwrap();
        let bytes = decode_heap_payload(
            reference.parse(heap.region()).unwrap(),
            filters.pipeline,
            filters.mask,
            reference.decoded_size(),
        )
        .unwrap();
        assert_eq!(
            layout
                .associate_decoded(&reference, &bytes)
                .unwrap()
                .bytes(),
            heap.decoded
        );
    }

    #[rstest]
    fn unfiltered_huge_objects_keep_both_id_representations(
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
        #[values(0, 512)] base: u64,
    ) {
        let heap = HugeHeap::new(direct, None, base);
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED).unwrap(),
            heap.decoded
        );
    }

    #[rstest]
    #[case::unknown(huge_unknown(), test_data::FILTER_BZIP2)]
    #[case::wide_shuffle(test_data::shuffle(4), FILTER_SHUFFLE)]
    fn filtered_huge_objects_reject_unsupported_active_filters(
        #[case] filter: FilterDescription,
        #[case] expected: u16,
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(
            direct,
            Some(HugeFilters::new(test_data::filter_pipeline(vec![]), 0)),
            0,
        );
        let FractalHeapFiltering::Filtered { pipeline, .. } = &mut heap.header.filtering else {
            panic!("expected filtered heap")
        };
        *pipeline = test_data::filter_pipeline(vec![filter]);
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED)
                .unwrap_err(),
            FormatError::UnsupportedFilter(expected)
        );
    }

    #[rstest]
    #[case::short(-1)]
    #[case::shorter(-2)]
    #[case::much_shorter(-300)]
    #[case::long(1)]
    #[cfg(feature = "deflate")]
    fn filtered_huge_objects_reject_incorrect_decoded_sizes(
        #[case] delta: i64,
        #[values(false, true)] checksum_before_deflate: bool,
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let pipeline = if checksum_before_deflate {
            test_data::filter_pipeline(vec![
                huge_fletcher(),
                test_data::deflate_pipeline().filters[0].clone(),
            ])
        } else {
            test_data::deflate_pipeline()
        };
        let mut heap = HugeHeap::new(direct, Some(HugeFilters::new(pipeline, 0)), 0);
        let expected = heap
            .decoded
            .len()
            .to_u64()
            .checked_add_signed(delta)
            .unwrap();
        heap.set_sizes(heap.encoded_size, expected);
        let error = if delta < 0 {
            let output_limit = expected
                + if checksum_before_deflate {
                    core::mem::size_of::<u32>().to_u64()
                } else {
                    0
                };
            FormatError::DecodedSizeLimitExceeded {
                expected_decoded_size: expected,
                filter_id: crate::filter_pipeline::FILTER_DEFLATE,
                output_limit,
                minimum_output_size: output_limit + 1,
            }
        } else {
            FormatError::DecodedSizeMismatch {
                expected,
                actual: heap.decoded.len().to_u64(),
            }
        };
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED)
                .unwrap_err(),
            error
        );
    }

    #[rstest]
    #[cfg(feature = "deflate")]
    fn filtered_huge_objects_reject_malformed_compressed_input(
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(
            direct,
            Some(HugeFilters::new(test_data::deflate_pipeline(), 0)),
            0,
        );
        heap.image[HUGE_PAYLOAD_AT as usize..].fill(b'x');
        let error = heap
            .read(backend, HugeObjectReadLimits::UNLIMITED)
            .unwrap_err();
        let FormatError::FilterError(message) = error else {
            panic!("expected FilterError, got {error:?}")
        };
        assert!(
            test_data::DEFLATE_HEADER_ERRORS.contains(&message.as_str()),
            "{message}"
        );
    }

    #[rstest]
    #[cfg(feature = "deflate")]
    fn huge_source_truncation_differs_from_a_missing_parser_region(
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(
            direct,
            Some(HugeFilters::new(test_data::deflate_pipeline(), 0)),
            0,
        );
        heap.image.pop();
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED)
                .unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: HUGE_PAYLOAD_AT + heap.encoded_size,
                available: heap.image.len().to_u64()
            }
        );
        let layout = FractalHeapHugeObjectLayout::new(
            &heap.header,
            FORMAT_WIDTHS,
            BaseAddress::ZERO,
            HugeIndirectKeyEncoding::Libhdf5,
        )
        .unwrap();
        let reference = match layout.parse_id(&heap.id).unwrap() {
            FractalHeapHugeObjectId::Direct(reference) => reference,
            FractalHeapHugeObjectId::Indirect(lookup) => {
                let tracker = layout.parse_tracker(heap.region()).unwrap();
                layout
                    .resolve(&lookup, &tracker.parse_record(&heap.record).unwrap())
                    .unwrap()
            }
        };
        assert_eq!(
            reference.parse(heap.region()).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: HUGE_PAYLOAD_AT,
                required_length: heap.encoded_size,
                provided_start: 0,
                provided_length: heap.image.len().to_u64()
            }
        );
    }

    #[rstest]
    #[case::encoded_budget(Some(false), None, None)]
    #[case::decoded_budget(Some(true), None, None)]
    #[case::encoded_host_max(None, Some(isize::MAX as u64), None)]
    #[case::decoded_host_max(None, None, Some(isize::MAX as u64))]
    #[case::encoded_host_overflow(None, Some(isize::MAX as u64 + 1), None)]
    #[case::decoded_host_overflow(None, None, Some(isize::MAX as u64 + 1))]
    fn huge_object_budgets_apply_before_payload_reads(
        #[case] budget: Option<bool>,
        #[case] encoded_size: Option<u64>,
        #[case] decoded_size: Option<u64>,
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(
            direct,
            Some(HugeFilters::new(test_data::filter_pipeline(vec![]), 0)),
            0,
        );
        let FractalHeapFiltering::Filtered { pipeline, .. } = &mut heap.header.filtering else {
            panic!("expected filtered heap")
        };
        *pipeline = test_data::filter_pipeline(vec![huge_unknown()]);
        let encoded = encoded_size.unwrap_or(heap.encoded_size);
        let decoded = decoded_size.unwrap_or(heap.decoded.len().to_u64());
        heap.set_sizes(encoded, decoded);
        heap.header.huge_objects_size = decoded;
        let limits = match budget {
            Some(false) => HugeObjectReadLimits {
                encoded_bytes: 0,
                ..HugeObjectReadLimits::UNLIMITED
            },
            Some(true) => HugeObjectReadLimits {
                decoded_bytes: 0,
                ..HugeObjectReadLimits::UNLIMITED
            },
            None => HugeObjectReadLimits::UNLIMITED,
        };
        let source = HugeExtentSource {
            bytes: &heap.image,
            reads: RefCell::new(Vec::new()),
        };
        let mut reader = heap.reader(limits);
        let result = if matches!(backend, ReadBackend::Source | ReadBackend::Streaming) {
            reader.read_from_source(&source, parse_id(&heap.header, &heap.id))
        } else {
            reader.read(&heap.image, parse_id(&heap.header, &heap.id))
        };
        let expected = if let Some(decoded_budget) = budget {
            FormatError::ResourceLimit {
                limit: 0,
                required: if decoded_budget { decoded } else { encoded },
            }
        } else if encoded > isize::MAX as u64 || decoded > isize::MAX as u64 {
            FormatError::ValueTooLargeForPlatform {
                value: if encoded > isize::MAX as u64 {
                    encoded
                } else {
                    decoded
                },
                target: "isize",
            }
        } else if matches!(backend, ReadBackend::Source | ReadBackend::Streaming) {
            FormatError::Source("test payload range requested".into())
        } else if encoded_size.is_some() {
            FormatError::UnexpectedSourceEof {
                expected: HUGE_PAYLOAD_AT + encoded,
                available: heap.image.len().to_u64(),
            }
        } else {
            FormatError::UnsupportedFilter(test_data::FILTER_BZIP2)
        };
        assert_eq!(result.unwrap_err(), expected);
        let payload_read = source
            .reads
            .borrow()
            .iter()
            .any(|&(address, _)| address == HUGE_PAYLOAD_AT);
        assert_eq!(
            payload_read,
            matches!(backend, ReadBackend::Source | ReadBackend::Streaming)
                && budget.is_none()
                && encoded <= isize::MAX as u64
                && decoded <= isize::MAX as u64
        );
    }

    struct HugeExtentSource<'a> {
        bytes: &'a [u8],
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl Source for HugeExtentSource<'_> {
        fn len(&self) -> u64 {
            u64::MAX
        }

        fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push((offset, bytes.len()));
            if offset == HUGE_PAYLOAD_AT {
                return Err(FormatError::Source("test payload range requested".into()));
            }
            BytesSource::new(self.bytes).read_at(offset, bytes)
        }

        fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
            self.reads.borrow_mut().push((offset, length));
            if offset == HUGE_PAYLOAD_AT {
                return Err(FormatError::Source("test payload range requested".into()));
            }
            BytesSource::new(self.bytes).read_metadata_at(offset, length)
        }
    }

    #[rstest]
    #[case::records(HugeObjectReadLimits { index_records: 0, ..HugeObjectReadLimits::UNLIMITED }, 1)]
    #[case::nodes(HugeObjectReadLimits { index_nodes: 0, ..HugeObjectReadLimits::UNLIMITED }, 1)]
    #[case::metadata(HugeObjectReadLimits { metadata_bytes: 0, ..HugeObjectReadLimits::UNLIMITED }, 38)]
    fn huge_index_budgets_stop_traversal(
        #[case] limits: HugeObjectReadLimits,
        #[case] required: u64,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let heap = HugeHeap::new(false, None, 0);
        assert_eq!(
            heap.read(backend, limits).unwrap_err(),
            FormatError::ResourceLimit { limit: 0, required }
        );
    }

    #[rstest]
    #[cfg(feature = "deflate")]
    fn huge_index_is_cached_and_payload_reads_use_encoded_extents(#[values(0, 512)] base: u64) {
        let heap = HugeHeap::new(
            false,
            Some(HugeFilters::new(test_data::deflate_pipeline(), 0)),
            base,
        );
        let source = HeaderSource {
            bytes: &heap.image,
            reads: RefCell::new(Vec::new()),
        };
        let framed = BaseOffsetSource {
            inner: &source,
            base: BaseAddress::new(base),
        };
        let mut reader = heap.reader(HugeObjectReadLimits::UNLIMITED);
        reset_huge_index_decodes();
        assert_eq!(
            reader
                .read_from_source(&framed, parse_id(&heap.header, &heap.id))
                .unwrap(),
            heap.decoded
        );
        assert_eq!(
            reader
                .read_from_source(&framed, parse_id(&heap.header, &heap.id))
                .unwrap(),
            heap.decoded
        );
        assert_eq!(huge_index_decodes(), 1);
        let reads = source.reads.borrow();
        assert_eq!(
            reads
                .iter()
                .filter(|&&(address, _)| address == base + HUGE_PAYLOAD_AT)
                .copied()
                .collect::<Vec<_>>(),
            vec![(base + HUGE_PAYLOAD_AT, heap.encoded_size as usize); 2]
        );
        assert_eq!(
            reads
                .iter()
                .filter(|&&(address, _)| address == base + HUGE_TRACKER_AT)
                .count(),
            1
        );
    }

    #[rstest]
    #[case::missing(2, false, FormatError::HugeObjectNotFound(2))]
    #[case::duplicate(1, true, FormatError::InvalidFractalHeapGeometry)]
    fn huge_index_rejects_missing_and_duplicate_keys(
        #[case] key: u8,
        #[case] duplicate: bool,
        #[case] expected: FormatError,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(false, None, 0);
        heap.id[1] = key;
        if duplicate {
            heap.write_tracker(vec![heap.record.clone(), heap.record.clone()]);
        }
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED)
                .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::wrong_type(hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT, 36)]
    #[case::short(hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT, 16)]
    #[case::long(hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT, 25)]
    fn huge_index_checks_type_and_exact_record_size(
        #[case] tree_type: u8,
        #[case] size: u16,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(false, None, 0);
        let header =
            test_util::btree_v2::Header::new(tree_type, size, HUGE_NODE_AT, 1).build(Widths::EIGHT);
        heap.image[HUGE_TRACKER_AT as usize..HUGE_TRACKER_AT as usize + header.len()]
            .copy_from_slice(&header);
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED)
                .unwrap_err(),
            FormatError::UnexpectedHugeObjectBTree {
                tree_type,
                record_size: usize::from(size),
                required: 24
            }
        );
    }

    #[rstest]
    #[cfg(feature = "checksum")]
    fn huge_index_metadata_checksums_remain_required(
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(false, None, 0);
        heap.image[HUGE_NODE_AT as usize + HUGE_NODE_PREFIX_LEN] ^= 1;
        let checksum_at = HUGE_NODE_AT as usize + HUGE_NODE_PREFIX_LEN + heap.record.len();
        let expected = test_util::bytes::u32_at(&heap.image, checksum_at);
        let computed =
            crate::checksum::jenkins_lookup3(&heap.image[HUGE_NODE_AT as usize..checksum_at]);
        assert_eq!(
            heap.read(backend, HugeObjectReadLimits::UNLIMITED)
                .unwrap_err(),
            FormatError::ChecksumMismatch { expected, computed }
        );
    }

    #[rstest]
    #[cfg(debug_assertions)]
    fn huge_index_cache_rejects_backend_changes(#[values(false, true)] filtered: bool) {
        let heap = HugeHeap::new(
            false,
            filtered.then(|| HugeFilters::new(test_data::filter_pipeline(vec![]), 0)),
            0,
        );
        let mut reader = heap.reader(HugeObjectReadLimits::UNLIMITED);
        assert_eq!(
            reader
                .read(&heap.image, parse_id(&heap.header, &heap.id))
                .unwrap(),
            heap.decoded
        );
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            reader.read_from_source(
                &BytesSource::new(&heap.image),
                parse_id(&heap.header, &heap.id),
            )
        }));
        let payload = panic.unwrap_err();
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(
            message.contains("a heap's huge-object index belongs to its reading backend"),
            "{message}"
        );
    }

    #[rstest]
    #[case::read(HugeObjectReadLimits::UNLIMITED, false)]
    #[case::cycle(HugeObjectReadLimits::UNLIMITED, true)]
    #[case::node_budget(HugeObjectReadLimits { index_nodes: 1, ..HugeObjectReadLimits::UNLIMITED }, false)]
    fn huge_index_traversal_validates_internal_nodes(
        #[case] limits: HugeObjectReadLimits,
        #[case] cycle: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
        let mut heap = HugeHeap::new(false, None, 0);
        let record = |key: u64| {
            let mut bytes = heap.record.clone();
            bytes[16..24].copy_from_slice(&key.to_le_bytes());
            bytes
        };
        let header = test_util::btree_v2::Header::new(BTREE_V2_HUGE_OBJECT, 24, HUGE_NODE_AT, 1)
            .node_size(256)
            .depth(1)
            .total_records(3)
            .build(Widths::EIGHT);
        let children = [
            test_util::btree_v2::Child {
                address: if cycle { HUGE_NODE_AT } else { 384 },
                records: 1,
                records_width: 1,
                subtree: None,
            },
            test_util::btree_v2::Child {
                address: 640,
                records: 1,
                records_width: 1,
                subtree: None,
            },
        ];
        let root = test_util::btree_v2::internal(
            BTREE_V2_HUGE_OBJECT,
            &[record(2)],
            &children,
            Widths::EIGHT,
        );
        let first = test_util::btree_v2::leaf(BTREE_V2_HUGE_OBJECT, &[record(1)]);
        let last = test_util::btree_v2::leaf(BTREE_V2_HUGE_OBJECT, &[record(3)]);
        heap.image[HUGE_TRACKER_AT as usize..HUGE_TRACKER_AT as usize + header.len()]
            .copy_from_slice(&header);
        heap.image[HUGE_NODE_AT as usize..HUGE_NODE_AT as usize + root.len()]
            .copy_from_slice(&root);
        heap.image[384..384 + first.len()].copy_from_slice(&first);
        heap.image[640..640 + last.len()].copy_from_slice(&last);
        if cycle {
            assert_eq!(
                heap.read(backend, limits).unwrap_err(),
                FormatError::InvalidBTreeV2Signature
            );
        } else if limits.index_nodes == 1 {
            assert_eq!(
                heap.read(backend, limits).unwrap_err(),
                FormatError::ResourceLimit {
                    limit: 1,
                    required: 2
                }
            );
        } else {
            assert_eq!(heap.read(backend, limits).unwrap(), heap.decoded);
        }
    }

    #[rstest]
    #[cfg(feature = "deflate")]
    fn huge_compressed_allocation_length_is_not_the_decoded_length(
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let mut heap = HugeHeap::new(
            direct,
            Some(HugeFilters::new(test_data::deflate_pipeline(), 0)),
            0,
        );
        heap.set_sizes(heap.encoded_size - 1, heap.decoded.len().to_u64());
        let error = heap
            .read(backend, HugeObjectReadLimits::UNLIMITED)
            .unwrap_err();
        let FormatError::FilterError(message) = error else {
            panic!("expected FilterError, got {error:?}")
        };
        assert_eq!(
            message,
            "deflate: stream ended before the chunk was complete"
        );
    }

    #[rstest]
    #[case::exact(0, None)]
    #[case::short(-1, Some(12))]
    #[case::further_tokens(-7, Some(6))]
    #[case::long(1, None)]
    fn lzf_huge_objects_keep_size_errors_typed(
        #[case] delta: i64,
        #[case] minimum_output_size: Option<u64>,
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let pipeline = test_data::filter_pipeline(vec![FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_LZF,
            name: None,
            flags: 0,
            client_data: vec![],
        }]);
        let mut heap = HugeHeap::new(direct, Some(HugeFilters::new(pipeline, 0)), 0);
        heap.decoded = b"abcdefghijkl".to_vec();
        let encoded = [
            2, b'a', b'b', b'c', 2, b'd', b'e', b'f', 2, b'g', b'h', b'i', 2, b'j', b'k', b'l',
        ];
        heap.image.truncate(HUGE_PAYLOAD_AT as usize);
        heap.image.extend_from_slice(&encoded);
        heap.encoded_size = encoded.len().to_u64();
        heap.header.huge_objects_size = heap.decoded.len().to_u64();
        let expected = heap
            .decoded
            .len()
            .to_u64()
            .checked_add_signed(delta)
            .unwrap();
        heap.set_sizes(heap.encoded_size, expected);
        let result = heap.read(backend, HugeObjectReadLimits::UNLIMITED);
        if delta == 0 {
            assert_eq!(result.unwrap(), heap.decoded);
        } else {
            let error = if let Some(minimum_output_size) = minimum_output_size {
                FormatError::DecodedSizeLimitExceeded {
                    expected_decoded_size: expected,
                    filter_id: crate::filter_pipeline::FILTER_LZF,
                    output_limit: expected,
                    minimum_output_size,
                }
            } else {
                FormatError::DecodedSizeMismatch {
                    expected,
                    actual: heap.decoded.len().to_u64(),
                }
            };
            assert_eq!(result.unwrap_err(), error);
        }
    }

    #[rstest]
    #[case::forward(512, 1024)]
    #[case::backward(512, 0)]
    fn huge_object_reads_use_relocated_physical_positions(
        #[case] encoded_base: u64,
        #[case] signature: u64,
        #[values(false, true)] direct: bool,
        #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    ) {
        let heap = HugeHeap::new(
            direct,
            Some(HugeFilters::new(test_data::filter_pipeline(vec![]), 0)),
            signature,
        );
        let addresses = OpenedImageContext::reconcile(
            BaseAddress::new(encoded_base),
            encoded_base + heap.image.len().to_u64() - signature,
            signature,
        )
        .unwrap()
        .addresses();
        let mut reader = HeapObjectReader::new(
            &heap.header,
            addresses,
            StoredAddress::ZERO,
            AccessMode::ReadOnly,
            FORMAT_WIDTHS,
        );
        let result = if matches!(backend, ReadBackend::Source | ReadBackend::Streaming) {
            reader.read_from_source(
                &BaseOffsetSource {
                    inner: &BytesSource::new(&heap.image),
                    base: addresses.effective_base(),
                },
                parse_id(&heap.header, &heap.id),
            )
        } else {
            reader.read(
                &heap.image[signature as usize..],
                parse_id(&heap.header, &heap.id),
            )
        };
        assert_eq!(result.unwrap(), heap.decoded);
    }

    #[rstest]
    #[cfg(not(feature = "deflate"))]
    fn huge_deflate_objects_require_the_decoder_feature(#[values(false, true)] direct: bool) {
        let mut heap = HugeHeap::new(
            direct,
            Some(HugeFilters::new(test_data::filter_pipeline(vec![]), 0)),
            0,
        );
        let FractalHeapFiltering::Filtered { pipeline, .. } = &mut heap.header.filtering else {
            panic!("expected filtered heap")
        };
        *pipeline = test_data::deflate_pipeline();
        assert_eq!(
            heap.read(ReadBackend::Streaming, HugeObjectReadLimits::UNLIMITED)
                .unwrap_err(),
            FormatError::UnsupportedFilter(crate::filter_pipeline::FILTER_DEFLATE)
        );
    }

    /// A fixture pipeline with a mask checked against its filter count.
    struct HugeFilters {
        pipeline: FilterPipeline,
        mask: PipelineFilterMask,
    }

    impl HugeFilters {
        fn new(pipeline: FilterPipeline, bits: u32) -> Self {
            let mask = PipelineFilterMask::from_raw_bits(bits, pipeline.filters.len()).unwrap();
            assert_eq!(mask.uninterpreted_bits(), 0);
            Self { pipeline, mask }
        }
    }

    struct HugeHeap {
        header: FractalHeapHeader,
        image: Vec<u8>,
        id: Vec<u8>,
        record: Vec<u8>,
        decoded: Vec<u8>,
        encoded_size: u64,
        base: u64,
        tree_type: u8,
    }

    impl HugeHeap {
        fn new(direct: bool, filters: Option<HugeFilters>, base: u64) -> Self {
            let filtered = filters.is_some();
            let decoded = b"filtered huge object bytes".repeat(400);
            let mut header = FractalHeapHeader::parse_bytes(
                &fractal_heap::Header::new(u64::MAX).build(Widths::EIGHT),
                OffsetWidth::Eight,
                LengthWidth::Eight,
            )
            .unwrap();
            header.heap_id_length = if direct {
                if filtered { 29 } else { 17 }
            } else {
                8
            };
            header.btree_huge_objects_address = StoredAddress::new(HUGE_TRACKER_AT);
            header.huge_objects_count = 1;
            header.huge_objects_size = decoded.len().to_u64();
            let (encoded, mask) = if let Some(HugeFilters { pipeline, mask }) = filters {
                let active = test_data::filter_pipeline(
                    pipeline
                        .filters
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| !mask.is_skipped(*index).unwrap())
                        .map(|(_, filter)| filter.clone())
                        .collect(),
                );
                let encoded = test_data::encode_block(&decoded, &active);
                header.filtering = FractalHeapFiltering::Filtered {
                    root_direct_block_size: 0,
                    root_filter_mask: 0,
                    pipeline,
                };
                (encoded, mask.recorded_mask().raw_bits())
            } else {
                (decoded.clone(), 0)
            };
            let mut record = Vec::new();
            record.extend_from_slice(&HUGE_PAYLOAD_AT.to_le_bytes());
            record.extend_from_slice(&encoded.len().to_u64().to_le_bytes());
            if filtered {
                record.extend_from_slice(&mask.to_le_bytes());
                record.extend_from_slice(&decoded.len().to_u64().to_le_bytes());
            }
            let mut id = vec![HUGE_ID_FIRST_BYTE];
            if direct {
                id.extend_from_slice(&record);
            } else {
                id.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0]);
                record.extend_from_slice(&1u64.to_le_bytes());
            }
            let mut image = vec![0; (base + HUGE_PAYLOAD_AT) as usize];
            image.extend_from_slice(&encoded);
            let mut fixture = Self {
                header,
                image,
                id,
                record,
                decoded,
                encoded_size: encoded.len().to_u64(),
                base,
                tree_type: match (filtered, direct) {
                    (false, false) => hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT,
                    (true, false) => hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT,
                    (false, true) => hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT_DIRECT,
                    (true, true) => {
                        hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT
                    }
                },
            };
            fixture.write_tracker(vec![fixture.record.clone()]);
            fixture
        }

        fn write_tracker(&mut self, records: Vec<Vec<u8>>) {
            let header = test_util::btree_v2::Header::new(
                self.tree_type,
                u16::try_from(records[0].len()).unwrap(),
                HUGE_NODE_AT,
                records.len() as u16,
            )
            .node_size(256)
            .build(Widths::EIGHT);
            let node = test_util::btree_v2::leaf(self.tree_type, &records);
            let at = (self.base + HUGE_TRACKER_AT) as usize;
            self.image[at..at + header.len()].copy_from_slice(&header);
            let at = (self.base + HUGE_NODE_AT) as usize;
            self.image[at..at + node.len()].copy_from_slice(&node);
        }

        fn set_sizes(&mut self, encoded: u64, decoded: u64) {
            self.record[8..16].copy_from_slice(&encoded.to_le_bytes());
            self.record[20..28].copy_from_slice(&decoded.to_le_bytes());
            if self.header.huge_ids_direct_with_widths(FORMAT_WIDTHS) {
                self.id[1..29].copy_from_slice(&self.record);
            }
            self.write_tracker(vec![self.record.clone()]);
        }

        fn reader(&self, limits: HugeObjectReadLimits) -> HeapObjectReader<'_> {
            HeapObjectReader::with_huge_limits(
                &self.header,
                AddressContext::from_effective_base(BaseAddress::new(self.base)),
                StoredAddress::ZERO,
                AccessMode::ReadOnly,
                FORMAT_WIDTHS,
                limits,
            )
        }

        fn read(
            &self,
            backend: ReadBackend,
            limits: HugeObjectReadLimits,
        ) -> Result<Vec<u8>, FormatError> {
            if matches!(backend, ReadBackend::Source | ReadBackend::Streaming) {
                self.reader(limits).read_from_source(
                    &BaseOffsetSource {
                        inner: &BytesSource::new(&self.image),
                        base: BaseAddress::new(self.base),
                    },
                    parse_id(&self.header, &self.id),
                )
            } else {
                self.reader(limits).read(
                    &self.image[self.base as usize..],
                    parse_id(&self.header, &self.id),
                )
            }
        }

        #[cfg(feature = "deflate")]
        fn region(&self) -> FileRegion<'_> {
            FileRegion::new(
                FileRange::new(PhysicalAddress::new(0), self.image.len().to_u64()).unwrap(),
                &self.image,
            )
            .unwrap()
        }
    }

    #[cfg(feature = "deflate")]
    fn huge_fletcher() -> FilterDescription {
        FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_FLETCHER32,
            name: None,
            flags: 0,
            client_data: vec![],
        }
    }

    fn huge_unknown() -> FilterDescription {
        FilterDescription {
            filter_id: test_data::FILTER_BZIP2,
            name: None,
            flags: 0,
            client_data: vec![],
        }
    }

    /// The signature, version, and type width of a version 2 B-tree node, from "Version 2 B-trees"
    /// of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
    #[cfg(feature = "checksum")]
    const HUGE_NODE_PREFIX_LEN: usize = 6;
    const HUGE_TRACKER_AT: u64 = 64;
    const HUGE_NODE_AT: u64 = 128;
    const HUGE_PAYLOAD_AT: u64 = 1024;

    fn parse_id<'a>(header: &FractalHeapHeader, bytes: &'a [u8]) -> FractalHeapIdView<'a> {
        FractalHeapIdLayout::new(header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(bytes)
            .unwrap()
    }

    const FILTERED_ROOT_AT: u64 = 0x100;
    const FILTERED_BLOCK_AT: u64 = 0x200;
    const FILTERED_BLOCK_SIZE: u64 = 128;
    // A version 0 huge ID has type 1 in bits 4 and 5 ("Fractal Heap", specification 4.0).
    const HUGE_ID_FIRST_BYTE: u8 = 0x10;
    // A version 0 tiny ID has type 2 in bits 4 and 5 ("Fractal Heap", specification 4.0).
    const TINY_ID_FIRST_BYTE: u8 = 0x20;
}
