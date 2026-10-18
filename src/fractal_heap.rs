//! Resolves managed, huge, and tiny objects stored in a fractal heap.
//!
//! [`HeapObjectReader`] reads from either an in-memory file image or a [`Source`].

use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::num::NonZeroU32;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::FractalHeapBlock;
use hdf5_pure_format::__private::FractalHeapBlockFilters;
use hdf5_pure_format::__private::FractalHeapBlockLayout;
pub use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapHeaderLocator;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdView;
use hdf5_pure_format::__private::HugeObjectRecord;
use hdf5_pure_format::__private::HugeObjectReference;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree_v2::{
    BTreeV2Header, BTreeV2Record, collect_btree_v2_records, collect_btree_v2_records_from_source,
};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::filter_pipeline::FILTER_SHUFFLE;
use crate::filters;
use crate::filters::ChunkContext;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::FormatWidths;
#[cfg(test)]
use crate::width::{LengthWidth, OffsetWidth};

/// Parses a heap header using validated file widths.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the image is truncated, and the address and header
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
    let prefix = buffered_region(image, locator.parse_range()?)?;
    let reference = locator.parse(prefix)?;
    reference.parse(buffered_region(image, reference.parse_range())?)
}

fn buffered_region<'a>(
    image: FileRegion<'a>,
    range: FileRange,
) -> Result<FileRegion<'a>, FormatError> {
    if !image.range().contains(range) {
        return Err(FormatError::UnexpectedEof {
            expected: range
                .end()
                .get()
                .saturating_sub(image.range().start().get())
                .to_usize()
                .unwrap_or(usize::MAX),
            available: image.bytes().len(),
        });
    }
    image.subregion(range)
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
) -> Result<Vec<u8>, FormatError> {
    source.read_metadata_at(
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
    // `H5Z__filter_shuffle` in `H5Zshuffle.c`, HDF5 2.2.0, shuffles at the width in its first
    // client value. `filters::decompress_chunk` reverses a shuffle at the element size of its
    // context, one byte here, so an active shuffle filter of another width is rejected.
    let wide_shuffle = filters
        .pipeline
        .filters
        .iter()
        .enumerate()
        .any(|(index, filter)| {
            let skipped = u32::try_from(index)
                .ok()
                .and_then(|bit| filters.mask.checked_shr(bit))
                .is_some_and(|mask| mask & 1 == 1);
            filter.filter_id == FILTER_SHUFFLE && !skipped && filter.client_data != [1]
        });
    if wide_shuffle {
        return Err(FormatError::UnsupportedFilter(FILTER_SHUFFLE));
    }
    filters::decompress_chunk(
        encoded,
        filters.pipeline,
        ChunkContext {
            chunk_dims: &[decoded_size],
            element_size: NonZeroU32::MIN,
            element_type: None,
            scale_offset_type: None,
        },
        filters.mask,
    )
}

/// Copy `len` bytes at `addr` out of an in-memory file image, bounds-checked.
fn slice_object(file_data: &[u8], addr: u64, len: usize) -> Result<Vec<u8>, FormatError> {
    let start = addr.to_usize()?;
    let end = start.checked_add(len).ok_or(FormatError::OffsetOverflow {
        offset: addr,
        length: len as u64,
    })?;
    if end > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: end,
            available: file_data.len(),
        });
    }
    Ok(file_data[start..end].to_vec())
}

/// Read `len` bytes at `addr` from a streaming source.
fn read_object_at_source<S: Source + ?Sized>(
    source: &S,
    addr: u64,
    len: usize,
) -> Result<Vec<u8>, FormatError> {
    source.read_metadata_at(addr, len)
}

/// Checks the B-tree header against the indexed, unfiltered huge-object record layout.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedHugeObjectBTree`] if the record type differs or the
/// records are too short, and [`FormatError::InvalidOffsetSize`] or
/// [`FormatError::InvalidLengthSize`] if a field width is not 2, 4, or 8.
fn check_huge_object_btree(
    header: &BTreeV2Header,
    widths: FormatWidths,
) -> Result<(), FormatError> {
    let required = HugeObjectRecord::size(widths.offsets, widths.lengths);
    if header.tree_type != BTREE_V2_HUGE_OBJECT || header.record_size < required {
        return Err(FormatError::UnexpectedHugeObjectBTree {
            tree_type: header.tree_type,
            record_size: usize::from(header.record_size),
            required: usize::from(required),
        });
    }
    Ok(())
}

/// Every huge object a heap holds, decoded from its huge-objects v2 B-tree.
///
/// Resolving even one huge object costs the whole index: that B-tree is read by
/// collecting its records rather than by descending to a key. So the records are
/// decoded here once, and kept by the [`HeapObjectReader`] that asked for them.
///
/// Only the type-1 record layout appears here, which
/// [`check_huge_object_btree`] establishes before any record reaches this.
struct HugeObjectIndex {
    /// `(id, address, length)`, ordered by id so a lookup is a binary search.
    /// The B-tree already stores them in that order; sorting again costs one
    /// pass and means correctness here does not rest on that.
    entries: Vec<(u64, StoredAddress, u64)>,
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
    fn decode(records: &[BTreeV2Record], widths: FormatWidths) -> Result<Self, FormatError> {
        #[cfg(test)]
        HUGE_INDEX_DECODES.with(|count| count.set(count.get() + 1));

        let mut entries = Vec::with_capacity(records.len());
        for record in records {
            let HugeObjectRecord {
                address,
                length,
                id,
            } = record.huge_object(widths.offsets, widths.lengths)?;
            entries.push((id, address, length));
        }
        entries.sort_unstable();
        Ok(Self { entries })
    }

    /// The `(address, length)` of the object with `huge_id`.
    fn locate(&self, huge_id: u64) -> Result<(StoredAddress, u64), FormatError> {
        match self
            .entries
            .binary_search_by_key(&huge_id, |&(id, _, _)| id)
        {
            Ok(i) => {
                let (_, addr, len) = self.entries[i];
                Ok((addr, len))
            }
            Err(_) => Err(FormatError::HugeObjectNotFound(huge_id)),
        }
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
        }
    }

    /// Reads the object identified by a parsed ID from a file image.
    ///
    /// Managed objects are read through the heap's doubling table. Indexed huge objects are
    /// looked up in the cached huge-object index. Tiny-object bytes are copied from `id`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::HugeObjectNotFound`] if an indexed huge object is absent, and
    /// [`FormatError::UnexpectedEof`] if a managed object has no allocated block or a required
    /// file region is truncated. Returns the errors of decoding a filtered direct block, such as
    /// [`FormatError::UnsupportedFilter`] if its pipeline has a filter without a decoder.
    /// Returns [`FormatError::ChunkedReadError`] if indirect-block traversal exceeds 64 levels,
    /// [`FormatError::ValueTooLargeForPlatform`] if a value does not fit the platform's address
    /// width, and [`FormatError::OffsetOverflow`] if a file offset overflows. Returns errors from
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
            FractalHeapIdKind::Huge(reference) => self.read_huge(file_data, reference),
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
            FractalHeapIdKind::Huge(reference) => self.read_huge_from_source(source, reference),
            FractalHeapIdKind::Tiny { bytes } => Ok(bytes.to_vec()),
        }
    }

    /// The `(address, length)` of huge object `huge_id`, parsing the heap's
    /// huge-object index out of `records` the first time one is asked for.
    ///
    /// What the index caches are file addresses, resolved against the image
    /// `backend` names. Every caller reads a whole heap through one backend, so
    /// a reader that saw both would be a caller that had changed which file it
    /// meant mid-walk — addresses applied to the wrong image read the wrong
    /// bytes and report no error, which is worth an assertion where the two meet.
    fn locate_huge<F>(
        &mut self,
        huge_id: u64,
        backend: Backend,
        records: F,
    ) -> Result<(StoredAddress, u64), FormatError>
    where
        F: FnOnce() -> Result<Vec<BTreeV2Record>, FormatError>,
    {
        if self.huge.is_none() {
            self.huge = Some(HugeObjectIndex::decode(&records()?, self.widths)?);
            self.huge_backend = Some(backend);
        }
        debug_assert_eq!(
            self.huge_backend,
            Some(backend),
            "a heap's huge-object index holds addresses into the image it was decoded from",
        );
        self.huge
            .as_ref()
            .expect("filled immediately above")
            .locate(huge_id)
    }

    /// Reads a huge object at its inline location or through the cached index.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read`](Self::read) for a huge object.
    fn read_huge(
        &mut self,
        file_data: &[u8],
        reference: HugeObjectReference,
    ) -> Result<Vec<u8>, FormatError> {
        let (addr, len) = match reference {
            HugeObjectReference::Inline { addr, len } => (addr, len),
            HugeObjectReference::Indexed(huge_id) => {
                let widths = self.widths;
                let btree_addr = self.header.btree_huge_objects_address.get().to_usize()?;
                self.locate_huge(huge_id, Backend::Buffered, || {
                    let header = BTreeV2Header::parse_with_widths(file_data, btree_addr, widths)?;
                    check_huge_object_btree(&header, widths)?;
                    collect_btree_v2_records(file_data, &header, widths.offsets)
                })?
            }
        };
        slice_object(file_data, addr.get(), len.to_usize()?)
    }

    /// Reads a huge object through a [`Source`] at its inline location or through the cached index.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`read_from_source`](Self::read_from_source) for a huge object.
    fn read_huge_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        reference: HugeObjectReference,
    ) -> Result<Vec<u8>, FormatError> {
        let (addr, len) = match reference {
            HugeObjectReference::Inline { addr, len } => (addr, len),
            HugeObjectReference::Indexed(huge_id) => {
                let widths = self.widths;
                let btree_addr = self.header.btree_huge_objects_address.get();
                self.locate_huge(huge_id, Backend::Streaming, || {
                    let header = BTreeV2Header::parse_from_source_with_widths(
                        &SourceMetadata(source),
                        btree_addr,
                        widths,
                    )?;
                    check_huge_object_btree(&header, widths)?;
                    collect_btree_v2_records_from_source(source, &header, widths.offsets)
                })?
            }
        };
        read_object_at_source(source, addr.get(), len.to_usize()?)
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
            |range| buffered_region(image, range).map(|region| Cow::Borrowed(region.bytes())),
            file_data.len() as u64,
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
            |range| read_header_range(source, addresses, range).map(Cow::Owned),
            source.len(),
            Backend::Streaming,
            heap_offset,
            object_length,
        )
    }

    fn read_managed_object_with<'bytes>(
        &mut self,
        read: impl Fn(FileRange) -> Result<Cow<'bytes, [u8]>, FormatError>,
        source_len: u64,
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
        let mut block = blocks.root()?.ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
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
                    return Ok(read(object.range())?.into_owned());
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
                    block = selected.ok_or(FormatError::UnexpectedEof {
                        expected: heap_offset
                            .to_usize()?
                            .saturating_add(object_length.to_usize()?),
                        available: source_len.to_usize().unwrap_or(usize::MAX),
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
    use hdf5_pure_format::__private::FilterDescription;
    use hdf5_pure_format::__private::FilterPipeline;
    use hdf5_pure_format::__private::FractalHeapFiltering;
    use hdf5_pure_format::__private::FractalHeapHeaderFrame;
    use hdf5_pure_format::__private::FractalHeapIdLayout;
    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

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
            FormatError::UnexpectedEof {
                expected,
                available
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
            FormatError::UnexpectedEof {
                expected: range.end().get() as usize,
                available: supplied.end().get() as usize
            }
        );
    }

    #[rstest]
    #[case::tiny(vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c', 0, 0, 0, 0], false)]
    #[case::filtered_tiny(vec![TINY_ID_FIRST_BYTE | 2, b'a', b'b', b'c', 0, 0, 0, 0], true)]
    #[case::inline_huge(vec![HUGE_ID_FIRST_BYTE, 0x20, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0], false)]
    fn a_parsed_inline_id_resolves_its_object(
        #[case] bytes: Vec<u8>,
        #[case] filtered: bool,
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
        let object = if streaming {
            reader.read_from_source(&BytesSource::new(image), id)
        } else {
            reader.read(&image, id)
        }
        .unwrap();
        assert_eq!(object, b"abc");
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
        encoded_len: usize,
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
                indirect_len = root.len();
                let at = usize::try_from(FILTERED_ROOT_AT).unwrap();
                image[at..at + indirect_len].copy_from_slice(&root);
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
                encoded_len,
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

    /// Type-1 record: address(8) + length(8) + id(8), little-endian.
    fn huge_record(addr: u64, len: u64, id: u64) -> BTreeV2Record {
        let mut d = Vec::new();
        d.extend_from_slice(&addr.to_le_bytes());
        d.extend_from_slice(&len.to_le_bytes());
        d.extend_from_slice(&id.to_le_bytes());
        BTreeV2Record { data: d }
    }

    #[test]
    fn huge_object_index_locates_by_id() {
        let records = vec![
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
            huge_record(0x3000, 7000, 5),
        ];
        let index = HugeObjectIndex::decode(&records, FORMAT_WIDTHS).unwrap();
        assert_eq!(index.locate(2).unwrap(), (StoredAddress::new(0x2000), 6000));
        assert_eq!(index.locate(5).unwrap(), (StoredAddress::new(0x3000), 7000));
        assert_eq!(
            index.locate(9),
            Err(FormatError::HugeObjectNotFound(9)),
            "an id no record carries is not found rather than mismatched"
        );
    }

    /// The lookup is a binary search, so the index has to impose the ordering
    /// itself rather than inherit whatever order the records arrived in.
    #[test]
    fn huge_object_index_orders_records_it_receives_out_of_order() {
        let records = vec![
            huge_record(0x3000, 7000, 5),
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
        ];
        let index = HugeObjectIndex::decode(&records, FORMAT_WIDTHS).unwrap();
        for (id, want) in [
            (1, (StoredAddress::new(0x1000), 5000)),
            (2, (StoredAddress::new(0x2000), 6000)),
            (5, (StoredAddress::new(0x3000), 7000)),
        ] {
            assert_eq!(index.locate(id).unwrap(), want);
        }
    }

    /// A huge-objects B-tree that is not the type-1 layout is refused before its
    /// records are read as that layout.
    ///
    /// Type 2 is the filtered counterpart, whose records are longer and put the
    /// object ID somewhere else: decoding one as type 1 reads an ID out of the
    /// filter mask and returns some other object, or none, without complaint.
    #[test]
    fn a_huge_object_btree_of_another_type_is_refused() {
        let btree = |tree_type: u8, record_size: u16| BTreeV2Header {
            tree_type,
            node_size: 512,
            record_size,
            depth: 0,
            root_node_address: StoredAddress::new(0x100),
            num_records_in_root: 1,
            total_records: 1,
        };

        assert_eq!(
            check_huge_object_btree(&btree(2, 36), FORMAT_WIDTHS),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 2,
                record_size: 36,
                required: 24,
            }),
            "a filtered huge-object tree is long enough to pass a length check alone"
        );
        assert_eq!(
            check_huge_object_btree(&btree(1, 16), FORMAT_WIDTHS),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 1,
                record_size: 16,
                required: 24,
            }),
            "records too short for the layout the type claims"
        );
        assert_eq!(
            check_huge_object_btree(&btree(1, 24), FORMAT_WIDTHS),
            Ok(())
        );
    }

    /// A record too short for the type-1 layout is a heap this reader has
    /// misread, not one object to skip past. The backstop behind
    /// [`check_huge_object_btree`], which refuses such a tree by its declared
    /// record size before any record reaches the decode.
    #[test]
    fn huge_object_index_refuses_a_truncated_record() {
        let records = vec![BTreeV2Record {
            data: vec![0u8; 20],
        }];
        assert!(matches!(
            HugeObjectIndex::decode(&records, FORMAT_WIDTHS),
            Err(FormatError::UnexpectedEof { .. })
        ));
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
