//! Chunked dataset index traversal, chunk decoding, and dense-output assembly.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, format, vec, vec::Vec};
#[cfg(feature = "std")]
use std::borrow::Cow;

use alloc::sync::Arc;

use core::mem::MaybeUninit;
use core::num::{NonZeroU64, NonZeroUsize};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::MetadataFrame;
use hdf5_pure_core::PhysicalAddress;
#[cfg(test)]
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;

use hdf5_pure_format::__private::BTreeV1ChunkNode;
use hdf5_pure_format::__private::BTreeV1ChunkNodeBytes;
use hdf5_pure_format::__private::BTreeV2ChunkRecordContext;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::LayoutVersion;
#[cfg(feature = "std")]
use hdf5_pure_space::__private::Extent;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_cache::{CachePass, ChunkCache};
use crate::chunk_grid;
use crate::chunk_grid::ChunkGrid;
use crate::chunk_grid::ChunkGridSlot;
use crate::chunk_grid::GridOrder;
use crate::chunk_span::ChunkSpanReader;
use crate::convert::Narrow;
#[cfg(any(feature = "std", test))]
use crate::data_layout::DataLayout;
use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags};
use crate::dataspace::Dataspace;
use crate::error::FormatError;
use crate::extensible_array;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::{self, ChunkContext, FilterScratch};
use crate::fixed_array;
use crate::parsed_chunk_index::ChunkSelection;
use crate::parsed_chunk_index::ClippedAxis;
use crate::parsed_chunk_index::IndexedChunk;
use crate::parsed_chunk_index::ParsedChunkIndex;
use crate::parsed_chunk_index::SelectedChunk;
use crate::parsed_chunk_index::SelectionCoverage;
use crate::read_region::{ReadRegion, RowRegion};
use crate::read_spec::{RawReadSpec, RawReadStorage};
use crate::source;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::{FormatWidths, OffsetWidth};
#[cfg(test)]
use crate::{FixedPointType, FloatingPointType};

/// Reads the stored bytes of `chunk` from `source` and decodes them with [`decode_chunk`].
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if the stored size of `chunk` exceeds
/// [`usize::MAX`], the error `source` returns for the read, and the errors of [`decode_chunk`].
pub(crate) fn read_decoded_chunk(
    source: &(impl Source + ?Sized),
    scratch: &mut FilterScratch,
    flags: ChunkedLayoutFlags,
    chunk_dims: &[u64],
    ds_dims: &[u64],
    pipeline: Option<&FilterPipeline>,
    ctx: ChunkContext<'_>,
    chunk: &ChunkInfo,
) -> Result<Vec<u8>, FormatError> {
    let raw_chunk = source.read_exact_at(chunk.address.get(), chunk.chunk_size.to_usize()?)?;
    let filtering = ChunkFiltering::for_chunk(flags, chunk_dims, ds_dims, chunk);
    Ok(decode_chunk(scratch, Cow::Owned(raw_chunk), pipeline, ctx, filtering)?.into_owned())
}

/// Decodes one stored chunk to its full logical byte size.
///
/// The required length is the product of the chunk dimensions and element size, including the
/// portion of an edge chunk outside the dataset extent. Runs the complete inverse pipeline
/// before comparing the decoded length. For raw chunks and datasets without a pipeline,
/// preserves the ownership of `stored`.
///
/// # Errors
///
/// Returns [`FormatError::DataSizeMismatch`] if the decoded length differs from the full chunk
/// size, [`FormatError::InvalidChunkGeometry`] if that size exceeds
/// [`MAX_CHUNK_ALLOCATION_BYTES`] or its calculation overflows [`u64`], and
/// [`FormatError::ValueTooLargeForPlatform`] if the element size or chunk byte count exceeds
/// [`usize::MAX`]. Returns the errors of [`filters::decompress_chunk_with`] for active filters.
fn decode_chunk<'a>(
    scratch: &mut FilterScratch,
    stored: Cow<'a, [u8]>,
    pipeline: Option<&FilterPipeline>,
    ctx: ChunkContext<'_>,
    filtering: ChunkFiltering,
) -> Result<Cow<'a, [u8]>, FormatError> {
    let expected = bounded_chunk_len(ctx.chunk_dims, ctx.element_size.narrow()?)?;
    let decoded = match (pipeline, filtering) {
        (Some(pipeline), ChunkFiltering::Filtered { filter_mask }) => Cow::Owned(
            filters::decompress_chunk_with(scratch, &stored, pipeline, ctx, filter_mask)?,
        ),
        _ => stored,
    };
    if decoded.len() != expected {
        return Err(FormatError::DataSizeMismatch {
            expected,
            actual: decoded.len(),
        });
    }
    Ok(decoded)
}

/// Whether a read runs a chunk's bytes through the filter pipeline, with the filter mask of a
/// filtered chunk.
///
/// A writer stores the mask in the chunk's index record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkFiltering {
    /// The pipeline produced the stored bytes, and this mask holds the filters
    /// it skipped for this chunk.
    Filtered { filter_mask: u32 },
    /// The stored bytes are the chunk's data.
    Raw,
}

impl ChunkFiltering {
    /// Returns how `chunk` is stored, from the layout flags and the chunk's
    /// place in the dataset.
    ///
    /// Under [`ChunkedLayoutFlags::partial_edge_chunks_stored_raw`] a chunk
    /// that extends past the dataset's extent is [`Self::Raw`]. Every other
    /// chunk is [`Self::Filtered`], with the mask its index record stores.
    fn for_chunk(
        flags: ChunkedLayoutFlags,
        chunk_dims: &[u64],
        ds_dims: &[u64],
        chunk: &ChunkInfo,
    ) -> Self {
        match ChunkDecoding::at_extent(flags, chunk_dims, ds_dims, &chunk.offsets) {
            ChunkDecoding::Pipeline => Self::Filtered {
                filter_mask: chunk.filter_mask,
            },
            ChunkDecoding::Verbatim => Self::Raw,
        }
    }
}

/// Whether a reader at a given dataset extent runs a chunk's bytes through the filter pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkDecoding {
    /// A reader runs the chunk's bytes through the filter pipeline under the chunk's filter mask.
    Pipeline,
    /// A reader takes the chunk's bytes as its data.
    Verbatim,
}

impl ChunkDecoding {
    /// Returns how a reader at the extent `ds_dims` decodes the chunk at `offsets`.
    ///
    /// Under [`ChunkedLayoutFlags::partial_edge_chunks_stored_raw`] a chunk that extends past the
    /// extent is [`Self::Verbatim`]. Every other chunk is [`Self::Pipeline`].
    pub(crate) fn at_extent(
        flags: ChunkedLayoutFlags,
        chunk_dims: &[u64],
        ds_dims: &[u64],
        offsets: &[u64],
    ) -> Self {
        if flags.partial_edge_chunks_stored_raw()
            && extends_past_extent(chunk_dims, ds_dims, offsets)
        {
            return Self::Verbatim;
        }
        Self::Pipeline
    }
}

/// Returns `true` if the chunk at `offsets` extends past the dataset's extent.
///
/// `H5D__chunk_is_partial_edge_chunk` in `H5Dchunk.c` (HDF5 1.14.6) tests the
/// same condition on the scaled chunk coordinate, against the dataset's current
/// extent. A version 1 B-tree's offset vector holds a trailing element offset
/// beyond the dataset's dimensions, so the zip stops at the dataset's rank.
fn extends_past_extent(chunk_dims: &[u64], ds_dims: &[u64], offsets: &[u64]) -> bool {
    chunk_dims
        .iter()
        .zip(ds_dims)
        .zip(offsets)
        .any(|((&chunk_dim, &ds_dim), &offset)| offset.saturating_add(chunk_dim) > ds_dim)
}

/// Information about a single chunk in a chunked dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChunkInfo {
    /// Size of chunk data in the file (after compression).
    pub(crate) chunk_size: StoredChunkSize,
    /// Bitmask of filters that were NOT applied (0 = all applied).
    pub(crate) filter_mask: u32,
    /// N-dimensional offset of this chunk in dataset space.
    pub(crate) offsets: Vec<u64>,
    /// Address of the chunk data, as the chunk index stores it.
    ///
    /// A caller that reads the bytes from a source that is not framed at the base address adds
    /// the base first.
    pub(crate) address: StoredAddress,
}

impl ChunkInfo {
    /// Returns the chunks a leaf of a version 1 B-tree chunk index points to.
    ///
    /// # Errors
    ///
    /// Returns the error [`BTreeV1ChunkNode::entries`] yields for an entry of `node`.
    fn leaf_chunks(node: BTreeV1ChunkNode<'_>) -> Result<Vec<Self>, FormatError> {
        let mut chunks = Vec::with_capacity(usize::from(node.entries_used()));
        for entry in node.entries() {
            let (key, address) = entry?;
            chunks.push(Self {
                chunk_size: StoredChunkSize::btree_v1(key.chunk_size),
                filter_mask: key.filter_mask,
                offsets: key.offsets().collect(),
                address,
            });
        }
        Ok(chunks)
    }

    /// Returns the chunk `record` describes at `slot` of `grid`, or `None` where the slot is
    /// outside the extent of the dataset.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChunkedReadError`] if `grid` has a dimension of no chunks, or if
    /// `slot` resolves past the offsets a `u64` holds.
    fn in_extent(
        grid: &ChunkGrid,
        slot: u64,
        ChunkRecord {
            address,
            stored_size,
            filter_mask,
        }: ChunkRecord,
    ) -> Result<Option<Self>, FormatError> {
        Ok(grid.offsets_in_extent(slot)?.map(|offsets| Self {
            chunk_size: StoredChunkSize::v4(stored_size),
            filter_mask,
            offsets,
            address,
        }))
    }

    /// Reads this chunk of a rank-1 dataset of length `extent` and returns the decoded bytes of
    /// its elements below `extent`.
    ///
    /// The chunk is decoded as a reader at `extent` decodes it, so under
    /// [`ChunkedLayoutFlags::partial_edge_chunks_stored_raw`] a chunk that extends past `extent`
    /// is read as it is stored.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the chunk's offsets or the chunk dimensions of `ctx`
    /// are not of rank 1. Returns the errors of [`read_decoded_chunk`], and of
    /// [`bounded_chunk_len`] for the elements below `extent`.
    #[cfg(feature = "std")]
    pub(crate) fn read_live_elements(
        &self,
        source: &(impl Source + ?Sized),
        scratch: &mut FilterScratch,
        flags: ChunkedLayoutFlags,
        pipeline: Option<&FilterPipeline>,
        ctx: ChunkContext<'_>,
        extent: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let (&[offset], &[chunk_len]) = (self.offsets.as_slice(), ctx.chunk_dims) else {
            return Err(FormatError::Internal(format!(
                "a chunk's live elements are read at rank 1, not for {} chunk dimensions and {} \
                 offsets",
                ctx.chunk_dims.len(),
                self.offsets.len()
            )));
        };
        let mut logical = read_decoded_chunk(
            source,
            scratch,
            flags,
            ctx.chunk_dims,
            &[extent],
            pipeline,
            ctx,
            self,
        )?;
        let live = extent.saturating_sub(offset).min(chunk_len);
        logical.truncate(bounded_chunk_len(&[live], ctx.element_size.narrow()?)?);
        Ok(logical)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoredChunkSize {
    /// A version 1 B-tree stores the chunk size in a four-byte field.
    BTreeV1(u32),
    /// A version 4 chunk index stores the chunk size in a variable-width field.
    V4(u64),
}

impl StoredChunkSize {
    /// Wraps the stored chunk size from a version 1 B-tree record.
    pub(crate) const fn btree_v1(size: u32) -> Self {
        Self::BTreeV1(size)
    }

    /// Wraps the stored chunk size from a version 4 chunk index.
    pub(crate) const fn v4(size: u64) -> Self {
        Self::V4(size)
    }

    /// Returns the stored chunk size as a `u64`.
    pub(crate) const fn get(self) -> u64 {
        match self {
            Self::BTreeV1(size) => size as u64,
            Self::V4(size) => size,
        }
    }

    /// Returns `true` if the stored chunk size is zero.
    pub(crate) const fn is_zero(self) -> bool {
        self.get() == 0
    }

    /// Converts the stored chunk size to `usize`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the size is larger
    /// than this platform can address.
    pub(crate) fn to_usize(self) -> Result<usize, FormatError> {
        self.get().to_usize()
    }
}

/// Traverse B-tree v1 type 1 to collect all chunk locations.
///
/// `ndims` is the number of offset dimensions in each key, which equals
/// `chunk_dimensions.len()` from the DataLayout::Chunked message (rank+1).
pub fn collect_chunk_info(
    file_data: &[u8],
    btree_address: StoredAddress,
    ndims: usize,
    offset_width: OffsetWidth,
) -> Result<Vec<ChunkInfo>, FormatError> {
    collect_chunk_info_inner(file_data, btree_address, ndims, offset_width, 0)
}

/// Depth-tracking core of [`collect_chunk_info`]. The `depth` guard mirrors
/// [`collect_chunk_btree_node_spans_inner`]: a cyclic or pathologically deep
/// internal node in a foreign file errors out instead of recursing until the
/// stack overflows (which would abort the process uncatchably).
fn collect_chunk_info_inner(
    file_data: &[u8],
    btree_address: StoredAddress,
    ndims: usize,
    offset_width: OffsetWidth,
    depth: u32,
) -> Result<Vec<ChunkInfo>, FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let node = BTreeV1ChunkNode::parse_with_offset_width(
        file_data,
        btree_address.get().to_usize()?,
        offset_width,
        ndims,
    )?;
    if node.node_level() == 0 {
        return ChunkInfo::leaf_chunks(node);
    }
    let mut all_chunks = Vec::new();
    for child_addr in node.children() {
        all_chunks.extend(collect_chunk_info_inner(
            file_data,
            child_addr?,
            ndims,
            offset_width,
            depth + 1,
        )?);
    }
    Ok(all_chunks)
}

/// Recursion-depth cap for the chunk B-tree node-span walk, guarding against a
/// stack overflow on a cyclic or pathological index in a foreign file. A real
/// chunk B-tree is only a few levels deep, so this is far beyond any valid tree.
const MAX_CHUNK_BTREE_DEPTH: u32 = 64;

/// Enumerate the on-disk byte spans of every node in a version 1 B-tree of
/// type 1 (the raw-data chunk index), so a deleted chunked dataset's index can
/// be reclaimed. Returns one `(addr, len)` per `TREE` node (internal and leaf);
/// the chunk *data* blocks the leaves point at are enumerated separately by
/// [`collect_chunk_info`].
///
/// Each node's length is sized from `entries_used` — the region the reader
/// consumes (`header + entries_used * (key_size + offset_size) + key_size`).
/// HDF5 allocates each node at its full `2K` capacity, so this is a lower bound:
/// a partially-filled node's trailing slack is left unreclaimed rather than
/// guessed at, honoring the editor's "under-reclaim, never over-reclaim" rule.
///
/// `ndims` is the number of offset dimensions in each key (rank + 1), the same
/// value [`collect_chunk_info`] takes. Errors (a malformed node, an out-of-bounds
/// child, or a tree deeper than [`MAX_CHUNK_BTREE_DEPTH`]) propagate so the
/// caller can leave the whole index unreclaimed.
pub(crate) fn collect_chunk_btree_node_spans<S: Source + ?Sized>(
    source: &S,
    btree_address: StoredAddress,
    ndims: usize,
    offset_width: OffsetWidth,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let mut out = Vec::new();
    collect_chunk_btree_node_spans_inner(source, btree_address, ndims, offset_width, 0, &mut out)?;
    Ok(out)
}

fn collect_chunk_btree_node_spans_inner<S: Source + ?Sized>(
    source: &S,
    btree_address: StoredAddress,
    ndims: usize,
    offset_width: OffsetWidth,
    depth: u32,
    out: &mut Vec<(u64, u64)>,
) -> Result<(), FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    // Reading the node is also the bounds check: a node claiming to run past
    // end-of-file fails here rather than being recorded as reclaimable.
    let metadata = SourceMetadata(source);
    let bytes = BTreeV1ChunkNodeBytes::read_from_source_with_offset_width(
        &metadata,
        btree_address.get(),
        offset_width,
        ndims,
    )?;
    let node = bytes.node();
    out.push((btree_address.get(), node.stored_len().to_u64()));

    // Internal nodes (level > 0) hold child node addresses, not chunk records;
    // recurse so their nodes are reclaimed too.
    if node.node_level() != 0 {
        for child_addr in node.children() {
            collect_chunk_btree_node_spans_inner(
                source,
                child_addr?,
                ndims,
                offset_width,
                depth + 1,
                out,
            )?;
        }
    }
    Ok(())
}

/// Traverses a version 1 B-tree of node type 1 from a [`Source`], reading each node on demand.
///
/// The streaming counterpart of [`collect_chunk_info`]. Reads each node's 8-byte prefix and
/// remaining body through [`Source::read_metadata`]. Node views borrow the returned bytes while
/// the traversal descends into children.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the traversal depth exceeds
/// [`MAX_CHUNK_BTREE_DEPTH`], and errors from reading or parsing any node.
pub fn collect_chunk_info_from_source<S: Source + ?Sized>(
    source: &S,
    btree_address: StoredAddress,
    ndims: usize,
    offset_width: OffsetWidth,
) -> Result<Vec<ChunkInfo>, FormatError> {
    collect_chunk_info_from_source_inner(source, btree_address, ndims, offset_width, 0)
}

/// Depth-tracking core of [`collect_chunk_info_from_source`]; see
/// [`collect_chunk_info_inner`] for why the bound is required.
fn collect_chunk_info_from_source_inner<S: Source + ?Sized>(
    source: &S,
    btree_address: StoredAddress,
    ndims: usize,
    offset_width: OffsetWidth,
    depth: u32,
) -> Result<Vec<ChunkInfo>, FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let metadata = SourceMetadata(source);
    let bytes = BTreeV1ChunkNodeBytes::read_from_source_with_offset_width(
        &metadata,
        btree_address.get(),
        offset_width,
        ndims,
    )?;
    let node = bytes.node();
    if node.node_level() == 0 {
        return ChunkInfo::leaf_chunks(node);
    }
    let mut all_chunks = Vec::new();
    for child_addr in node.children() {
        all_chunks.extend(collect_chunk_info_from_source_inner(
            source,
            child_addr?,
            ndims,
            offset_width,
            depth + 1,
        )?);
    }
    Ok(all_chunks)
}

/// Returns one [`ChunkInfo`] per chunk in the current extent of `grid`, under an Implicit index
/// whose array of chunks starts at `data_address`.
///
/// An Implicit index is an array of chunks in slot order: the chunk at slot `n` of the maximum
/// chunk grid starts `n` chunk sizes past `data_address`. The index is defined in "The Implicit
/// Index" of the [format specification, version 4.0][spec], which counts the chunks along each
/// dimension from
/// the current extent. libhdf5 counts them from the maximum extent, in
/// `H5D__none_idx_get_addr` (`H5Dnone.c`, HDF5 2.2.0).
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the byte size of a chunk, the offset of a chunk in
/// the array, or the address of a chunk does not fit in a `u64`, and propagates the errors of
/// [`ChunkGrid::slots_in_extent`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_implicit
fn generate_implicit_chunks(
    data_address: StoredAddress,
    grid: &ChunkGrid,
    element_size: u64,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let chunk_byte_size = grid.chunk_byte_size(element_size)?;
    let slots = grid.slots_in_extent()?;
    let mut chunks = Vec::with_capacity(slots.len());
    for ChunkGridSlot { slot, offsets } in slots {
        let byte_offset = slot.checked_mul(chunk_byte_size).ok_or_else(|| {
            FormatError::ChunkedReadError(
                "implicit chunk slot resolves past the addressable data block".into(),
            )
        })?;
        let address = data_address.get().checked_add(byte_offset).ok_or_else(|| {
            FormatError::ChunkedReadError(
                "implicit chunk address resolves past the addressable file".into(),
            )
        })?;
        chunks.push(ChunkInfo {
            chunk_size: StoredChunkSize::v4(chunk_byte_size),
            filter_mask: 0,
            offsets,
            address: StoredAddress::new(address),
        });
    }
    Ok(chunks)
}

/// Returns the data rank and borrows the chunk and dataset dimensions.
///
/// Both dimension slices hold file-width extents. The chunk dimensions exclude the layout's
/// trailing element-size dimension.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the layout has no dimensions or its rank
/// differs from the dataspace rank.
fn chunked_dims<'a>(
    chunk_dimensions: &'a [u64],
    dataspace: &'a Dataspace,
) -> Result<(usize, &'a [u64], &'a [u64]), FormatError> {
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let chunk_dims = &chunk_dimensions[..rank];
    let ds_dims = dataspace.dimensions.as_slice();
    if ds_dims.len() != rank {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims (rank={})",
            ds_dims.len(),
            chunk_dimensions.len(),
            rank
        )));
    }
    Ok((rank, chunk_dims, ds_dims))
}

const MAX_CHUNK_ALLOCATION_BYTES: u64 = u32::MAX as u64;

/// Returns the size in bytes of a decoded chunk of `chunk_dims` elements of `elem_size` bytes
/// each.
///
/// The chunk dimensions and the element size are read from the file. The size is bounded per
/// chunk by [`MAX_CHUNK_ALLOCATION_BYTES`], so a dataset that is large through its number of
/// chunks stays readable.
///
/// # Errors
///
/// Returns [`FormatError::InvalidChunkGeometry`] if the size exceeds
/// [`MAX_CHUNK_ALLOCATION_BYTES`] or its calculation overflows [`u64`]. Returns
/// [`FormatError::ValueTooLargeForPlatform`] if the byte count exceeds [`usize::MAX`].
pub(crate) fn bounded_chunk_len(
    chunk_dims: &[u64],
    elem_size: NonZeroUsize,
) -> Result<usize, FormatError> {
    let logical = chunk_dims
        .iter()
        .try_fold(elem_size.get() as u64, |acc, &d| acc.checked_mul(d));
    match logical {
        Some(bytes) if bytes <= MAX_CHUNK_ALLOCATION_BYTES => bytes.to_usize(),
        _ => Err(FormatError::InvalidChunkGeometry(
            "chunk logical byte size exceeds the reader's single-chunk allocation limit",
        )),
    }
}

/// Reads a whole chunked dataset from a [`Source`], decoding chunks sequentially.
///
/// The output byte length must fit [`usize`] before the reader reads the chunk index.
/// Dataset dimensions are converted to host-sized indexes only for a nonempty output.
/// Returns an empty vector for a zero-element dataset without consulting the fill pattern.
/// For nonempty output, allocates without initialization if dense coverage is proven and uses
/// the fill pattern otherwise.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if the output byte length or a required
/// buffer index exceeds [`usize::MAX`], and errors from reading the datatype, fill value,
/// chunk geometry, index, or payloads. Returns [`FormatError::DataSizeMismatch`] if a present
/// chunk decodes to a byte length other than its full logical size, and
/// [`FormatError::OffsetOverflow`] if output size or placement arithmetic overflows.
pub fn read_chunked_data_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (dataspace, datatype, pipeline, fill) = (
        spec.dataspace(),
        spec.datatype(),
        spec.pipeline(),
        spec.fill(),
    );
    let total_bytes = spec.byte_len()?;
    let Some(ChunkedPlan {
        flags,
        ds_dims,
        chunk_dims_u64,
        elem_size,
        chunks,
    }) = ChunkedPlan::new(source, spec, widths)?
    else {
        return spec.unallocated_buffer();
    };
    if total_bytes == 0 {
        return Ok(Vec::new());
    }
    let index = ParsedChunkIndex::new(chunks, chunk_dims_u64, ds_dims)?;
    let mut assembly = WholeAssembly::new(&index, chunk_dims_u64, ds_dims, elem_size, fill)?;
    let ctx = ChunkContext::from_datatype(chunk_dims_u64, datatype)?;
    let mut scratch = FilterScratch::new();
    for IndexedChunk { info, .. } in index.chunks() {
        let decoded = read_decoded_chunk(
            source,
            &mut scratch,
            flags,
            chunk_dims_u64,
            &dataspace.dimensions,
            pipeline,
            ctx,
            info,
        )?;
        assembly.place(&decoded)?;
    }
    assembly.finish()
}

/// The geometry of a chunked dataset and the allocated chunks its chunk index lists.
struct ChunkedPlan<'a> {
    flags: ChunkedLayoutFlags,
    /// The complete dataset dimensions at file width.
    ds_dims: &'a [u64],
    /// The chunk dimensions at file width, excluding the trailing element-size dimension.
    chunk_dims_u64: &'a [u64],
    elem_size: NonZeroUsize,
    chunks: Vec<ChunkInfo>,
}

impl<'a> ChunkedPlan<'a> {
    /// Reads the chunk index of the chunked dataset `spec` describes, or returns `None` if the
    /// index address is undefined, as it is before the dataset's storage is allocated.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChunkedReadError`] if `spec` describes another layout or its chunk
    /// rank differs from the dataspace rank, [`FormatError::ZeroSizedDatatype`] if the size of
    /// the datatype is 0, [`FormatError::ValueTooLargeForPlatform`] if the size of the datatype or
    /// decoded chunk byte count exceeds [`usize::MAX`], [`FormatError::InvalidChunkGeometry`] if
    /// a chunk exceeds the reader's single-chunk allocation limit, and the errors of a malformed
    /// chunk index.
    fn new(
        source: &(impl Source + ?Sized),
        spec: RawReadSpec<'a>,
        widths: FormatWidths,
    ) -> Result<Option<Self>, FormatError> {
        let RawReadStorage::Chunked {
            flags,
            chunk_dimensions,
            index,
        } = spec.storage()
        else {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        };
        if index.address().is_none() {
            return Ok(None);
        }
        let elem_size = crate::datatype::element_size_usize(spec.datatype())?;
        let (_, chunk_dims, ds_dims) = chunked_dims(chunk_dimensions, spec.dataspace())?;
        bounded_chunk_len(chunk_dims, elem_size)?;
        let chunks = collect_chunks_for_layout_from_source(
            source,
            index,
            chunk_dimensions,
            spec.dataspace(),
            elem_size.get() as u64,
            widths,
        )?;
        let chunk_dims_u64 = chunk_dims;
        Ok(Some(Self {
            flags,
            ds_dims,
            chunk_dims_u64,
            elem_size,
            chunks,
        }))
    }
}

/// Sorts `chunks` in ascending order of address.
///
/// The walk of a chunk index lists its chunks in index order, and a B-tree, a Fixed Array, or an
/// Extensible Array may list a chunk at a higher address before one at a lower address.
/// [`ParsedChunkIndex::new`] puts the chunks the cached readers visit in the same order.
fn sort_chunks_by_address(chunks: &mut [ChunkInfo]) {
    chunks.sort_unstable_by_key(|c| c.address);
}

/// Plans coalesced reads over `chunks`, without the chunks whose decompressed bytes `cache` holds,
/// or returns `None` if the plan would read each chunk on its own.
///
/// The caller passes the chunks it visits: every chunk of the dataset for a whole read, and the
/// chunks a region intersects for a read of that region. The caller skips a chunk the cache
/// holds, so a span over it would read bytes the read does not use.
fn plan_chunk_spans<'a>(
    cache: &ChunkCache,
    chunks: impl Iterator<Item = IndexedChunk<'a>>,
) -> Option<ChunkSpanReader> {
    let cached = cache.decompressed_coords();
    ChunkSpanReader::new(
        chunks
            .filter(|chunk| !cached.iter().any(|held| held.as_slice() == chunk.coord))
            .map(|chunk| (chunk.info.address, chunk.info.chunk_size.get())),
    )
}

/// Reads a leading-dimension row window, decoding only overlapping chunks.
///
/// The caller clamps `row_start` and `num_rows` to the dataset. The output holds the selected
/// rows and the dataset's full inner dimensions. For an empty selection, the reader returns an
/// empty buffer before converting output dimensions to host-sized indexes. For a scalar layout,
/// the reader returns `Ok(None)` for the caller to read the whole dataset.
///
/// The reader intersects global row and chunk coordinates as [`u64`] values, then converts
/// offsets within the decoded chunk and output buffer to [`usize`].
///
/// The cache holds the parsed index and decoded chunks for reuse across windows. The caller uses
/// [`CachePass::LRU`] for a single window, or a shared pass for a sweep.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the layout is not chunked or its rank differs
/// from the dataspace rank, [`FormatError::InvalidChunkGeometry`] if a decoded chunk exceeds
/// the reader's allocation limit, and [`FormatError::OffsetOverflow`] if output byte counts
/// or strides overflow. Returns [`FormatError::ValueTooLargeForPlatform`] if a required buffer
/// size, dimension, or buffer index exceeds [`usize::MAX`], and errors from reading
/// the datatype, fill value, chunk index, or chunk payloads. Returns
/// [`FormatError::DataSizeMismatch`] if a present chunk decodes to a byte length other than its
/// full logical size.
#[cfg(test)]
pub(crate) fn read_chunked_rows_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
    pass: CachePass,
    row_start: u64,
    num_rows: u64,
) -> Result<Option<Vec<u8>>, FormatError> {
    if spec.dataspace().dimensions.is_empty() {
        return Ok(None);
    }
    let region = ReadRegion::rows(spec, row_start, num_rows)?;
    read_chunked_region_from_source(source, &region, widths, cache, pass).map(Some)
}

/// Reads intersecting chunks into the region's packed row-major result.
///
/// Uses [`ParsedChunkIndex::select`] to plan intersections at file width before conversion to
/// buffer indices. Parsed indexes and decoded chunks are retained according to the cache
/// configuration.
/// Consults fill only for a nonempty selection whose complete coverage is unproven. For proven
/// coverage, initializes output from complete decoded chunks and returns it after every selected
/// contribution has been placed.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the storage is not chunked or the chunk rank
/// differs from the dataspace rank, and [`FormatError::InvalidChunkGeometry`] if a decoded chunk
/// exceeds the allocation limit. Returns [`FormatError::ValueTooLargeForPlatform`] if a buffer
/// index exceeds [`usize::MAX`], and [`FormatError::OffsetOverflow`] if byte arithmetic overflows.
/// Returns [`FormatError::DataSizeMismatch`] if a present chunk decodes to a byte length other
/// than its full logical size, [`FormatError::UnreadableFillValue`] if fill is required and
/// unreadable, and errors from reading the chunk index, filters, or payloads.
pub(crate) fn read_chunked_region_from_source(
    source: &(impl Source + ?Sized),
    region: &ReadRegion<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
    pass: CachePass,
) -> Result<Vec<u8>, FormatError> {
    ChunkRead::new(region.spec(), widths, cache, pass).read_region(source, region)
}

/// A chunked dataset reader with one parsed index shared across its row windows.
///
/// Loads the index on the first nonempty read of allocated storage and retains it until this
/// reader is dropped. The index remains in [`ChunkCache`] after this reader is dropped only
/// when index caching is enabled. The caller must use the same source, framed at the file's
/// effective base address, for every read.
pub(crate) struct ChunkRead<'spec, 'cache> {
    spec: RawReadSpec<'spec>,
    widths: FormatWidths,
    cache: &'cache ChunkCache,
    pass: CachePass,
    parsed_index: Option<Arc<ParsedChunkIndex>>,
}

impl<'spec, 'cache> ChunkRead<'spec, 'cache> {
    /// Creates a reader without loading the chunk index.
    pub(crate) fn new(
        spec: RawReadSpec<'spec>,
        widths: FormatWidths,
        cache: &'cache ChunkCache,
        pass: CachePass,
    ) -> Self {
        Self {
            spec,
            widths,
            cache,
            pass,
            parsed_index: None,
        }
    }

    /// Reads complete rows after clamping the request to the leading dimension.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if byte strides overflow, and
    /// [`FormatError::ValueTooLargeForPlatform`] if the result exceeds the platform's buffer
    /// limit. Returns errors from selecting rows with [`ReadRegion::rows`] or reading them with
    /// [`Self::read_region`].
    pub(crate) fn read_rows(
        &mut self,
        source: &(impl Source + ?Sized),
        start: u64,
        count: u64,
    ) -> Result<Vec<u8>, FormatError> {
        let region = ReadRegion::rows(self.spec, start, count)?;
        self.read_region(source, &region)
    }

    /// Reads a region using this reader's parsed chunk index.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChunkedReadError`] if the storage is not chunked or the chunk rank
    /// differs from the dataspace rank, and [`FormatError::InvalidChunkGeometry`] if a decoded
    /// chunk exceeds the allocation limit. Returns [`FormatError::ValueTooLargeForPlatform`] if
    /// a buffer index exceeds [`usize::MAX`], and [`FormatError::OffsetOverflow`] if byte
    /// arithmetic overflows. Returns [`FormatError::DataSizeMismatch`] if a present chunk
    /// decodes to a byte length other than its full logical size,
    /// [`FormatError::UnreadableFillValue`] if fill is required and unreadable, and errors from
    /// reading the datatype, chunk index, filters, or payloads, or planning the selection.
    fn read_region(
        &mut self,
        source: &(impl Source + ?Sized),
        region: &ReadRegion<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        let Self {
            spec,
            widths,
            cache,
            pass,
            parsed_index,
        } = self;
        let spec = *spec;
        let widths = *widths;
        let cache = *cache;
        let pass = *pass;
        let (dataspace, datatype, pipeline, fill) = (
            spec.dataspace(),
            spec.datatype(),
            spec.pipeline(),
            spec.fill(),
        );
        let RawReadStorage::Chunked {
            flags,
            chunk_dimensions,
            index,
        } = spec.storage()
        else {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        };

        let elem_size = crate::datatype::element_size_usize(datatype)?;
        let (_, chunk_dims, _) = chunked_dims(chunk_dimensions, dataspace)?;
        bounded_chunk_len(chunk_dims, elem_size)?;

        if region.is_empty() {
            return Ok(Vec::new());
        }
        if index.address().is_none() {
            return fill.buffer(region.byte_len());
        }
        let chunk_dims_u64 = chunk_dims;
        let parsed_index = match parsed_index {
            Some(parsed) => parsed,
            None => parsed_index.insert(cache.parsed_index_or_load(|| {
                ParsedChunkIndex::new(
                    collect_chunks_for_layout_from_source(
                        source,
                        index,
                        chunk_dimensions,
                        dataspace,
                        elem_size.get() as u64,
                        widths,
                    )?,
                    chunk_dims_u64,
                    &dataspace.dimensions,
                )
            })?),
        };
        let selection = parsed_index.select(region)?;
        let mut assembly = SelectedAssembly::new(selection)?;

        let ctx = ChunkContext::from_datatype(chunk_dims_u64, datatype)?;

        // Coalesce over selected chunks to bound spans by the region's payloads.
        let mut spans = plan_chunk_spans(
            cache,
            assembly.selection.chunks().map(|selected| selected.chunk),
        );

        // One decoder for the whole window. See `FilterScratch`.
        let mut scratch = FilterScratch::new();

        let count = assembly.selection.chunks().len();
        for position in 0..count {
            let IndexedChunk { info: chunk, coord } = assembly
                .selection
                .chunk(position)
                .ok_or_else(|| FormatError::Internal("selected chunk is absent".into()))?;
            let hit = cache.with_decompressed(pass, coord, |bytes| assembly.place(bytes));
            if let Some(copied) = hit {
                copied?;
                continue;
            }

            let len = chunk.chunk_size.to_usize()?;
            let stored = match spans.as_mut() {
                Some(sp) => Cow::Borrowed(sp.chunk_bytes(source, chunk.address, len)?),
                None => Cow::Owned(source.read_exact_at(chunk.address.get(), len)?),
            };
            let filtering =
                ChunkFiltering::for_chunk(flags, chunk_dims_u64, &dataspace.dimensions, chunk);
            let dec = decode_chunk(&mut scratch, stored, pipeline, ctx, filtering)?;
            assembly.place(&dec)?;
            match dec {
                Cow::Owned(bytes) => cache.put_decompressed(pass, coord, bytes), // move
                Cow::Borrowed(bytes) => cache.put_decompressed_slice(pass, coord, bytes),
            }
        }

        assembly.finish()
    }
}

/// Output for a chunk selection, completed in descriptor order.
///
/// Dense storage remains at length zero until every selected contribution is initialized.
/// The region, chunk geometry, and coverage are borrowed from the selection for allocation and
/// placement.
struct SelectedAssembly<'a, 'spec> {
    selection: ChunkSelection<'a, 'spec>,
    placed: usize,
    output: AssemblyBuffer,
    geometry: RegionCopy<'a>,
    element_size: NonZeroUsize,
    chunk_bytes: usize,
}

impl<'a, 'spec> SelectedAssembly<'a, 'spec> {
    /// Allocates output and placement geometry for the bound selection.
    ///
    /// Consults fill only for [`SelectionCoverage::PotentiallyUncovered`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnreadableFillValue`] if complete coverage is unproven and fill is
    /// unreadable, [`FormatError::InvalidChunkGeometry`] if a decoded chunk exceeds the allocation
    /// limit, [`FormatError::ValueTooLargeForPlatform`] if a dimension or byte count exceeds
    /// [`usize::MAX`], and [`FormatError::OffsetOverflow`] if stride arithmetic overflows. Returns
    /// errors from reading the datatype's element size.
    fn new(selection: ChunkSelection<'a, 'spec>) -> Result<Self, FormatError> {
        let region = selection.region();
        let element_size = crate::datatype::element_size_usize(region.spec().datatype())?;
        let chunk_dims = selection.chunk_dims();
        let chunk_bytes = bounded_chunk_len(chunk_dims, element_size)?;
        let length = region.byte_len();
        let output = match selection.coverage() {
            SelectionCoverage::Empty => AssemblyBuffer::Fill(Vec::new()),
            SelectionCoverage::Complete => AssemblyBuffer::Dense {
                bytes: Vec::with_capacity(length),
                length,
            },
            SelectionCoverage::PotentiallyUncovered => {
                AssemblyBuffer::Fill(region.spec().fill().buffer(length)?)
            }
        };
        let geometry = match region.row_band() {
            Some(rows)
                if selection.trailing_offsets_are_zero()
                    && chunk_dims.iter().skip(1).eq(region
                        .spec()
                        .dataspace()
                        .dimensions
                        .iter()
                        .skip(1)) =>
            {
                RegionCopy::Band {
                    rows,
                    edge: chunk_dims.first().copied().unwrap_or(1),
                }
            }
            _ => RegionCopy::Scatter(RegionScatter::new(region, chunk_dims, element_size)?),
        };
        Ok(Self {
            selection,
            placed: 0,
            output,
            geometry,
            element_size,
            chunk_bytes,
        })
    }

    /// Writes the next selected descriptor's complete clipped contribution from `bytes`.
    ///
    /// Advances the completed descriptor count only after successful placement.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if the length of `bytes` differs from the full
    /// logical chunk size, and [`FormatError::Internal`] if every selected descriptor has been
    /// placed or a contribution exceeds its buffer. Returns [`FormatError::OffsetOverflow`] or
    /// [`FormatError::ValueTooLargeForPlatform`] if a byte offset exceeds its range.
    fn place(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        let Self {
            selection,
            placed,
            output,
            geometry,
            element_size,
            chunk_bytes,
        } = self;
        if bytes.len() != *chunk_bytes {
            return Err(FormatError::DataSizeMismatch {
                expected: *chunk_bytes,
                actual: bytes.len(),
            });
        }
        let SelectedChunk {
            chunk,
            intersection,
        } = selection.chunks().nth(*placed).ok_or_else(|| {
            FormatError::Internal("output assembly has no remaining chunk".into())
        })?;
        geometry
            .for_chunk(selection.region(), chunk.info, intersection, *element_size)?
            .copy_to(output.destination(), bytes, *element_size)?;
        *placed += 1;
        Ok(())
    }

    /// Returns initialized output after every selected descriptor has been placed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if any selected contribution remains unplaced.
    fn finish(self) -> Result<Vec<u8>, FormatError> {
        let Self {
            selection,
            placed,
            output,
            geometry: _,
            element_size: _,
            chunk_bytes: _,
        } = self;
        if placed != selection.chunks().len() {
            return Err(FormatError::Internal(
                "output assembly has chunks remaining".into(),
            ));
        }
        Ok(match output {
            AssemblyBuffer::Dense { mut bytes, length } => {
                // SAFETY: The bound selection proves its immutable contributions partition this output.
                // Every contribution advances only after all its clipped bytes have been initialized.
                unsafe {
                    bytes.set_len(length);
                }
                bytes
            }
            AssemblyBuffer::Fill(bytes) => bytes,
        })
    }
}

/// Placement geometry shared across the selected chunks.
enum RegionCopy<'a> {
    Band { rows: &'a RowRegion, edge: u64 },
    Scatter(RegionScatter),
}

impl RegionCopy<'_> {
    fn for_chunk(
        &mut self,
        region: &ReadRegion<'_>,
        chunk: &ChunkInfo,
        intersection: &[ClippedAxis],
        element_size: NonZeroUsize,
    ) -> Result<RegionChunk<'_>, FormatError> {
        match self {
            Self::Band { rows, edge } => {
                let offset = region
                    .spec()
                    .dataspace()
                    .dimensions
                    .first()
                    .and_then(|_| chunk.offsets.first())
                    .copied()
                    .unwrap_or(0);
                let (start, end) = intersection
                    .first()
                    .map_or((rows.start(), rows.end().min(*edge)), |axis| {
                        (axis.start(), axis.end())
                    });
                Ok(RegionChunk::Band {
                    destination: (start - rows.start()).to_usize()? * rows.row_bytes(),
                    source: (start - offset).to_usize()? * rows.row_bytes(),
                    length: (end - start).to_usize()? * rows.row_bytes(),
                })
            }
            Self::Scatter(geometry) => {
                geometry.for_chunk(region, chunk, intersection, element_size)
            }
        }
    }
}

/// Element strides and reusable intersection buffers for multidimensional placement.
struct RegionScatter {
    /// Chunk dimensions followed by their row-major element strides.
    chunk_geometry: Vec<usize>,
    output_dimensions: Vec<usize>,
    output_strides: Vec<usize>,
    /// Output offsets, intersection dimensions, and scatter coordinates, each of dataset rank.
    intersection: Vec<usize>,
}

impl RegionScatter {
    fn new(
        region: &ReadRegion<'_>,
        chunk_dimensions: &[u64],
        element_size: NonZeroUsize,
    ) -> Result<Self, FormatError> {
        let rank = chunk_dimensions.len();
        let mut chunk_geometry = Vec::with_capacity(multiply_offset(rank, 2)?);
        for &dimension in chunk_dimensions {
            chunk_geometry.push(dimension.to_usize()?);
        }
        let chunk_strides = row_major_strides(&chunk_geometry)?;
        chunk_geometry.extend(chunk_strides);
        Ok(Self {
            chunk_geometry,
            output_dimensions: region
                .axes()
                .map(|axis| axis.count().to_usize())
                .collect::<Result<Vec<_>, _>>()?,
            output_strides: region
                .axes()
                .map(|axis| {
                    axis.output_stride()
                        .map(|stride| stride / element_size.get())
                })
                .collect::<Result<Vec<_>, _>>()?,
            intersection: vec![0; multiply_offset(rank, 3)?],
        })
    }

    fn for_chunk(
        &mut self,
        region: &ReadRegion<'_>,
        chunk: &ChunkInfo,
        clipped: &[ClippedAxis],
        element_size: NonZeroUsize,
    ) -> Result<RegionChunk<'_>, FormatError> {
        let Self {
            chunk_geometry,
            output_dimensions,
            output_strides,
            intersection,
        } = self;
        let rank = chunk_geometry.len() / 2;
        let (chunk_dimensions, chunk_strides) = chunk_geometry.split_at(rank);
        let (offsets, scratch) = intersection.split_at_mut(rank);
        let (dimensions, _) = scratch.split_at_mut(rank);
        let mut advance = 0u64;
        for (dimension_index, ((((axis, &offset), _), &stride), clipped)) in region
            .axes()
            .zip(&chunk.offsets)
            .zip(chunk_dimensions)
            .zip(chunk_strides)
            .zip(clipped)
            .enumerate()
        {
            let start = clipped.start();
            let end = clipped.end();
            offsets[dimension_index] = (start - axis.start()).to_usize()?;
            dimensions[dimension_index] = (end - start).to_usize()?;
            let skip = (start - offset)
                .checked_mul(stride as u64)
                .and_then(|skip| skip.checked_mul(element_size.get() as u64))
                .ok_or(FormatError::OffsetOverflow {
                    offset: start - offset,
                    length: stride as u64,
                })?;
            advance = advance
                .checked_add(skip)
                .ok_or(FormatError::OffsetOverflow {
                    offset: advance,
                    length: skip,
                })?;
        }
        let source = advance.to_usize()?;
        match (offsets.first(), dimensions.first(), output_strides.first()) {
            (Some(&offset), Some(&count), Some(&stride))
                if dimensions
                    .iter()
                    .skip(1)
                    .eq(chunk_dimensions.iter().skip(1))
                    && output_dimensions
                        .iter()
                        .skip(1)
                        .eq(chunk_dimensions.iter().skip(1)) =>
            {
                Ok(RegionChunk::Band {
                    source,
                    destination: offset * stride * element_size.get(),
                    length: count * stride * element_size.get(),
                })
            }
            _ => Ok(RegionChunk::Scatter {
                source,
                geometry: self,
            }),
        }
    }
}

/// The intersection of a decoded chunk with the output region.
enum RegionChunk<'a> {
    Band {
        destination: usize,
        source: usize,
        length: usize,
    },
    Scatter {
        source: usize,
        geometry: &'a mut RegionScatter,
    },
}

impl RegionChunk<'_> {
    fn copy_to(
        &mut self,
        mut output: ScatterDestination<'_>,
        bytes: &[u8],
        element_size: NonZeroUsize,
    ) -> Result<(), FormatError> {
        match self {
            Self::Band {
                destination,
                source,
                length,
            } => {
                let end = add_offset(*source, *length)?;
                output.copy(
                    *destination,
                    bytes.get(*source..end).ok_or_else(|| {
                        FormatError::Internal("scatter exceeds decoded chunk extent".into())
                    })?,
                )
            }
            Self::Scatter { source, geometry } => {
                let RegionScatter {
                    chunk_geometry,
                    output_dimensions,
                    output_strides,
                    intersection,
                } = geometry;
                let rank = chunk_geometry.len() / 2;
                let (_, chunk_strides) = chunk_geometry.split_at(rank);
                let (offsets, scratch) = intersection.split_at_mut(rank);
                let (dimensions, coord) = scratch.split_at_mut(rank);
                place_chunk(
                    bytes.get(*source..).unwrap_or(&[]),
                    output,
                    offsets,
                    dimensions,
                    output_dimensions,
                    output_strides,
                    chunk_strides,
                    element_size,
                    coord,
                )
            }
        }
    }
}

/// Row-major strides for `dims` (innermost stride 1), checked so crafted
/// dimensions error instead of panicking (debug) or wrapping (release).
fn row_major_strides(dims: &[usize]) -> Result<Vec<usize>, FormatError> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] =
            strides[i + 1]
                .checked_mul(dims[i + 1])
                .ok_or(FormatError::OffsetOverflow {
                    offset: strides[i + 1] as u64,
                    length: dims[i + 1] as u64,
                })?;
    }
    Ok(strides)
}

/// Reads a whole chunked dataset from a [`Source`] with index and decoded-chunk caching.
///
/// Dataset dimensions are converted to host-sized indexes only for a nonempty output.
/// Returns an empty vector for a zero-element dataset without consulting the fill pattern.
/// For nonempty output, allocates without initialization if dense coverage is proven and uses
/// the fill pattern otherwise.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if the output byte length or a required
/// buffer index exceeds [`usize::MAX`], and errors from reading the datatype, fill value,
/// chunk geometry, index, or payloads. Returns [`FormatError::DataSizeMismatch`] if a present
/// chunk decodes to a byte length other than its full logical size, and
/// [`FormatError::OffsetOverflow`] if output size or placement arithmetic overflows.
pub fn read_chunked_data_cached_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    let (dataspace, datatype, pipeline, fill) = (
        spec.dataspace(),
        spec.datatype(),
        spec.pipeline(),
        spec.fill(),
    );
    let RawReadStorage::Chunked {
        flags,
        chunk_dimensions,
        index,
    } = spec.storage()
    else {
        return Err(FormatError::ChunkedReadError(
            "expected chunked layout".into(),
        ));
    };

    if index.address().is_none() {
        return spec.unallocated_buffer();
    }

    let elem_size = crate::datatype::element_size_usize(datatype)?;
    let (_, chunk_dims, ds_dims) = chunked_dims(chunk_dimensions, dataspace)?;
    bounded_chunk_len(chunk_dims, elem_size)?;
    let total_bytes = spec.byte_len()?;
    if total_bytes == 0 {
        return Ok(Vec::new());
    }
    let chunk_dims_u64 = chunk_dims;

    let parsed_index = cache.parsed_index_or_load(|| {
        ParsedChunkIndex::new(
            collect_chunks_for_layout_from_source(
                source,
                index,
                chunk_dimensions,
                dataspace,
                elem_size.get() as u64,
                widths,
            )?,
            chunk_dims_u64,
            &dataspace.dimensions,
        )
    })?;

    let mut assembly = WholeAssembly::new(&parsed_index, chunk_dims_u64, ds_dims, elem_size, fill)?;

    let ctx = ChunkContext::from_datatype(chunk_dims_u64, datatype)?;

    // Every chunk of the dataset is wanted here; what the plan leaves out is
    // whatever the chunk cache already holds from an earlier read.
    let mut spans = plan_chunk_spans(cache, parsed_index.chunks());

    let pass = cache.begin_pass();
    // One decoder for every chunk of the dataset; see `FilterScratch`.
    let mut scratch = FilterScratch::new();
    for IndexedChunk {
        info: chunk_info,
        coord,
    } in parsed_index.chunks()
    {
        // Scatter straight from the cached chunk under the lock (no copy out).
        let hit = cache.with_decompressed(pass, coord, |bytes| assembly.place(bytes));
        if let Some(placed) = hit {
            placed?;
            continue;
        }

        // Cache miss: fetch the chunk's stored bytes from the source.
        let len = chunk_info.chunk_size.to_usize()?;
        let stored = match spans.as_mut() {
            Some(sp) => Cow::Borrowed(sp.chunk_bytes(source, chunk_info.address, len)?),
            None => Cow::Owned(source.read_exact_at(chunk_info.address.get(), len)?),
        };
        // A chunk that skipped the pipeline keeps its stored bytes: read on its
        // own it moves into the cache, and borrowed from a coalesced span it is
        // copied out of the span only if the cache admits it.
        let filtering =
            ChunkFiltering::for_chunk(flags, chunk_dims_u64, &dataspace.dimensions, chunk_info);
        let dec = decode_chunk(&mut scratch, stored, pipeline, ctx, filtering)?;
        assembly.place(&dec)?;
        match dec {
            Cow::Owned(bytes) => cache.put_decompressed(pass, coord, bytes), // move
            Cow::Borrowed(bytes) => cache.put_decompressed_slice(pass, coord, bytes),
        }
    }

    assembly.finish()
}

#[allow(clippy::too_many_arguments)]
/// The chunk grid a dataset's positional index numbers its slots over.
///
/// Built from the dataspace's *maximum* extent, falling back to its current
/// one, because that is what the slot numbering is taken over; `order` says
/// whether the index rotates an unlimited dimension to the front. See
/// [`crate::chunk_grid`] for the rule itself.
fn index_grid(
    dataspace: &Dataspace,
    spatial_chunk_dims: &[u64],
    order: GridOrder,
) -> Result<ChunkGrid, FormatError> {
    ChunkGrid::new(
        spatial_chunk_dims,
        &dataspace.dimensions,
        dataspace.max_dimensions.as_deref(),
        order,
    )
}

/// Walks `index` and reports one [`ChunkInfo`] per chunk the file holds.
///
/// Returns an empty vector for an index with no address, which is a dataset
/// whose storage was never allocated.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] for invalid chunk geometry and propagates what the
/// index walkers report for malformed structures.
pub(crate) fn collect_chunks_for_layout_from_source<S: Source + ?Sized>(
    source: &S,
    index: ChunkIndexLayout,
    chunk_dimensions: &[u64],
    dataspace: &Dataspace,
    elem_size: u64,
    widths: FormatWidths,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    // An undefined index address means no storage is allocated yet, so there is
    // no index to walk and no chunk to report.
    let Some(addr) = index.address() else {
        return Ok(Vec::new());
    };
    let spatial_dims = &chunk_dimensions[..rank];
    match index {
        ChunkIndexLayout::BTreeV1 { .. } => {
            collect_chunk_info_from_source(source, addr, ndims, widths.offsets)
        }
        ChunkIndexLayout::SingleChunk { filtered, .. } => {
            let chunk_byte_size = chunk_grid::chunk_byte_size(spatial_dims, elem_size)?;
            let (chunk_size, filter_mask) = match filtered {
                Some(filtered) => (filtered.filtered_size, filtered.filter_mask),
                None => (chunk_byte_size, 0),
            };
            Ok(vec![ChunkInfo {
                chunk_size: StoredChunkSize::v4(chunk_size),
                filter_mask,
                offsets: vec![0u64; rank],
                address: addr,
            }])
        }
        ChunkIndexLayout::Implicit { .. } => Ok(generate_implicit_chunks(
            addr,
            &index_grid(dataspace, spatial_dims, GridOrder::RowMajor)?,
            elem_size,
        )?),
        ChunkIndexLayout::FixedArray { .. } => {
            let grid = index_grid(dataspace, spatial_dims, GridOrder::RowMajor)?;
            let mut chunks = Vec::new();
            fixed_array::read_chunks(
                source,
                widths,
                addr,
                grid.chunk_byte_size(elem_size)?,
                |slot, record| {
                    chunks.extend(ChunkInfo::in_extent(&grid, slot, record)?);
                    Ok(())
                },
            )?;
            Ok(chunks)
        }
        ChunkIndexLayout::ExtensibleArray { .. } => {
            let grid = index_grid(dataspace, spatial_dims, GridOrder::UnlimitedFirst)?;
            let mut chunks = Vec::new();
            extensible_array::read_chunks(
                source,
                widths,
                addr,
                grid.chunk_byte_size(elem_size)?,
                |slot, record| {
                    chunks.extend(ChunkInfo::in_extent(&grid, slot, record)?);
                    Ok(())
                },
            )?;
            Ok(chunks)
        }
        ChunkIndexLayout::BTreeV2 { .. } => collect_btree_v2_chunks_from_source(
            source,
            addr,
            spatial_dims,
            dataspace,
            elem_size,
            widths,
        ),
    }
}

/// Decodes every allocated chunk record in a version 2 B-tree.
fn collect_btree_v2_chunks_from_source<S: Source + ?Sized>(
    source: &S,
    header_address: StoredAddress,
    chunk_dims: &[u64],
    dataspace: &Dataspace,
    elem_size: u64,
    widths: FormatWidths,
) -> Result<Vec<ChunkInfo>, FormatError> {
    if dataspace.dimensions.len() != chunk_dims.len() {
        return Err(FormatError::ChunkedReadError(format!(
            "rank mismatch: dataspace has {} dims, layout has {} chunk dims",
            dataspace.dimensions.len(),
            chunk_dims.len()
        )));
    }

    let fixed_chunk_size = NonZeroU64::new(chunk_grid::chunk_byte_size(chunk_dims, elem_size)?)
        .ok_or(FormatError::InvalidChunkGeometry(
            "version 2 B-tree chunks must have a non-zero logical size",
        ))?;
    let header = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(source),
        header_address.get(),
        widths,
    )?;
    if u64::from(header.num_records_in_root) > header.total_records {
        return Err(FormatError::ChunkedReadError(
            "version 2 B-tree root record count exceeds the total record count".into(),
        ));
    }
    let context = BTreeV2ChunkRecordContext::new(
        header.tree_type,
        widths.offsets,
        chunk_dims.len(),
        fixed_chunk_size,
        LayoutVersion::Four,
    )?;
    context.require_record_size(header.record_size)?;

    let records =
        crate::btree_v2::collect_btree_v2_records_from_source(source, &header, widths.offsets)?;
    let expected_records = header.total_records.to_usize()?;
    if records.len() != expected_records {
        return Err(FormatError::ChunkedReadError(format!(
            "version 2 B-tree header declares {expected_records} records but traversal returned \
             {}",
            records.len()
        )));
    }

    let mut chunks = Vec::with_capacity(records.len());
    for raw in records {
        let record = context.decode(&raw)?;
        let offsets = chunk_grid::offsets_from_scaled(chunk_dims, &record.scaled_offsets)?;
        let end = record.address.get().checked_add(record.stored_size).ok_or(
            FormatError::OffsetOverflow {
                offset: record.address.get(),
                length: record.stored_size,
            },
        )?;
        if end > source.len() {
            return Err(FormatError::ChunkedReadError(format!(
                "version 2 B-tree chunk at {} with size {} extends past the source length {}",
                record.address.get(),
                record.stored_size,
                source.len()
            )));
        }
        if offsets
            .iter()
            .zip(&dataspace.dimensions)
            .any(|(offset, dimension)| offset >= dimension)
        {
            continue;
        }
        chunks.push(ChunkInfo {
            chunk_size: StoredChunkSize::v4(record.stored_size),
            filter_mask: record.filter_mask,
            offsets,
            address: record.address,
        });
    }
    Ok(chunks)
}

/// Enumerate every allocated chunk of a chunked dataset from an in-memory file
/// image, one [`ChunkInfo`] per chunk (file address, on-disk stored size, filter
/// mask, logical offsets). A buffered convenience wrapper over
/// [`collect_chunks_for_layout_from_source`] for callers that hold the whole
/// file as a byte slice (the in-place editor): it accepts a parsed
/// [`DataLayout::Chunked`] and the dataset's [`Dataspace`] and reads through a
/// [`BytesSource`](crate::source::BytesSource).
///
/// Returns an empty vector when the index address is undefined (a chunked dataset with no storage
/// allocated yet).
#[cfg(feature = "std")]
pub(crate) fn enumerate_chunks_from_source<S: Source + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    widths: FormatWidths,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        index,
        ..
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "enumerate_chunks_from_source called on a non-chunked layout".into(),
        ));
    };
    // `chunk_dimensions` is rank + 1 entries; the last is the element size.
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let elem_size = chunk_dimensions[rank];
    if elem_size == 0 {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has a zero element size".into(),
        ));
    }
    collect_chunks_for_layout_from_source(
        source,
        *index,
        chunk_dimensions,
        dataspace,
        elem_size,
        widths,
    )
}

/// Reads and decodes each allocated chunk of a chunked dataset in address order, and calls
/// `visit` with its bytes.
///
/// For a chunk that cannot be read or decoded, or that decodes to a length other than the size of
/// a chunk, `visit` is called with the error in place of the bytes, and the next chunk follows if
/// `visit` returns `Ok`. A dataset whose index address is undefined has no chunk to visit.
///
/// # Errors
///
/// Returns the errors of [`ChunkedPlan::new`], and the first error `visit` returns.
#[cfg(feature = "std")]
pub(crate) fn visit_decoded_chunks_from_source(
    source: &(impl Source + ?Sized),
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    mut visit: impl FnMut(Result<&[u8], &FormatError>) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let Some(ChunkedPlan {
        flags,
        chunk_dims_u64,
        mut chunks,
        ..
    }) = ChunkedPlan::new(source, spec, widths)?
    else {
        return Ok(());
    };
    sort_chunks_by_address(&mut chunks);
    let ctx = ChunkContext::from_datatype(chunk_dims_u64, spec.datatype())?;
    let mut scratch = FilterScratch::new();
    for chunk in &chunks {
        let decoded = read_decoded_chunk(
            source,
            &mut scratch,
            flags,
            chunk_dims_u64,
            &spec.dataspace().dimensions,
            spec.pipeline(),
            ctx,
            chunk,
        );
        visit(decoded.as_deref())?;
    }
    Ok(())
}

/// A proof that allocated chunks cover every logical chunk-grid slot exactly once.
///
/// Chunk coordinates are aligned and lie within the dataset extent. Their contributions,
/// clipped at dataset edges, cover every element exactly once. The chunk and dataset element
/// counts fit [`u64`], and the grid slot count fits [`usize`]. A caller validates decoded chunk
/// sizes separately.
#[derive(Debug)]
pub(crate) struct DenseCoverage {
    chunk_dims: Vec<u64>,
    ds_dims: Vec<u64>,
}

impl DenseCoverage {
    /// Proves coverage of `index` for `chunk_dims` and the dataset extent `ds_dims`.
    ///
    /// Returns `None` if the dimensions or chunk coordinates do not establish dense coverage,
    /// or if an element count or grid size overflows. Borrows the chunk descriptors during the
    /// proof and retains the dimensions for subsequent geometry comparisons.
    pub(crate) fn new(
        index: &ParsedChunkIndex,
        chunk_dims: &[u64],
        ds_dims: &[u64],
    ) -> Option<Self> {
        if chunk_dims.len() != ds_dims.len() || chunk_dims.is_empty() {
            return None;
        }
        let mut slots = 1u64;
        let mut chunk_elements = 1u64;
        let mut elements = 1u64;
        for (&chunk, &extent) in chunk_dims.iter().zip(ds_dims) {
            if chunk == 0 {
                return None;
            }
            chunk_elements = chunk_elements.checked_mul(chunk)?;
            elements = elements.checked_mul(extent)?;
            slots = slots.checked_mul(extent.div_ceil(chunk))?;
        }
        let slots = usize::try_from(slots).ok()?;
        if index.chunks().len() != slots {
            return None;
        }
        let mut seen = vec![0u64; slots.div_ceil(u64::BITS as usize)];
        let mut covered = 0u64;
        for chunk in index.chunks() {
            if chunk.coord.len() != ds_dims.len() {
                return None;
            }
            let mut slot = 0u64;
            let mut contribution = 1u64;
            for ((&offset, &chunk), &extent) in chunk.coord.iter().zip(chunk_dims).zip(ds_dims) {
                if !offset.is_multiple_of(chunk) || offset >= extent {
                    return None;
                }
                slot = slot
                    .checked_mul(extent.div_ceil(chunk))?
                    .checked_add(offset / chunk)?;
                contribution = contribution.checked_mul(chunk.min(extent - offset))?;
            }
            let slot = usize::try_from(slot).ok()?;
            let word = seen.get_mut(slot / u64::BITS as usize)?;
            let bit = 1u64 << (slot % u64::BITS as usize);
            if *word & bit != 0 {
                return None;
            }
            *word |= bit;
            covered = covered.checked_add(contribution)?;
        }
        if covered != elements {
            return None;
        }
        Some(Self {
            chunk_dims: chunk_dims.to_vec(),
            ds_dims: ds_dims.to_vec(),
        })
    }

    /// Returns `true` if both dimension slices equal those used to prove coverage.
    pub(crate) fn matches(&self, chunk_dims: &[u64], ds_dims: &[u64]) -> bool {
        self.chunk_dims == chunk_dims && self.ds_dims == ds_dims
    }
}

/// A chunked dataset's chunks mapped onto a dense logical grid: every grid slot
/// filled exactly once, ordered row-major (last dimension fastest) so the order
/// matches [`split_into_chunks`](crate::chunked_write::split_into_chunks) and the
/// verbatim layout planner.
#[cfg(feature = "std")]
pub(crate) struct DenseChunkGrid {
    /// One [`ChunkInfo`] per grid slot, in row-major grid order.
    pub(crate) grid_order: Vec<ChunkInfo>,
}

/// Map enumerated `infos` onto the logical chunk grid implied by `dims` (the
/// dataspace shape) and `chunk_dims` (the rank-only spatial chunk dimensions),
/// returning each slot's [`ChunkInfo`] in row-major order. Returns `None` when
/// the grid is *not* dense — a hole, a duplicate, a misaligned or out-of-range
/// chunk offset, a zero chunk dimension, or a chunk count other than the full
/// grid — so a caller that needs every slot filled (a verbatim copy, a
/// whole-dataset overwrite) can fall back or refuse. No chunk bytes are read.
#[cfg(feature = "std")]
pub(crate) fn plan_dense_grid(
    infos: Vec<ChunkInfo>,
    dims: &[u64],
    chunk_dims: &[u64],
) -> Option<DenseChunkGrid> {
    let rank = dims.len();
    if chunk_dims.len() < rank {
        return None;
    }
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        if chunk_dims[d] == 0 {
            return None;
        }
        num_chunks_per_dim.push(dims[d].div_ceil(chunk_dims[d]));
    }
    let total: u64 = num_chunks_per_dim.iter().product();
    if infos.len() as u64 != total {
        // A different chunk count than the full grid means holes (or duplicates).
        return None;
    }
    let total_us = usize::try_from(total).ok()?;

    // Map each chunk to its linear grid slot; detect any hole or duplicate.
    let mut slots: Vec<Option<ChunkInfo>> = (0..total_us).map(|_| None).collect();
    for info in infos {
        if info.offsets.len() < rank {
            return None;
        }
        let mut linear: u64 = 0;
        for d in 0..rank {
            if !info.offsets[d].is_multiple_of(chunk_dims[d]) {
                return None;
            }
            let grid_coord = info.offsets[d] / chunk_dims[d];
            if grid_coord >= num_chunks_per_dim[d] {
                return None;
            }
            linear = linear * num_chunks_per_dim[d] + grid_coord;
        }
        let idx = usize::try_from(linear).ok()?;
        if slots[idx].is_some() {
            return None; // duplicate offset
        }
        slots[idx] = Some(info);
    }

    // Every slot must be filled for a dense grid.
    let mut grid_order = Vec::with_capacity(total_us);
    for slot in slots {
        grid_order.push(slot?);
    }
    Some(DenseChunkGrid { grid_order })
}

/// Every on-disk byte span `(addr, len)` a chunked dataset owns: each allocated
/// chunk data block plus the chunk index's own structure blocks. This is the
/// single place that maps a chunked layout to the regions it occupies on disk.
/// The in-place editor uses it to reclaim that space on delete (issue #77).
///
/// `layout` must be a [`DataLayout::Chunked`] (anything else is an error).
/// Returns both halves empty when the index address is undefined (an empty or
/// never-written dataset owns no storage). Errors propagated from the index walkers mean the
/// layout's complete storage footprint could not be proven. For example, a structure may be
/// malformed. The caller then leaves the dataset's bytes in place because it cannot prove the
/// full region is free. The spans are
/// not pre-checked for mutual overlap or file bounds. The editor validates that before handing
/// them to the free list.
///
/// Chunk data spans come from the same index walkers the reader uses
/// ([`collect_chunks_for_layout_from_source`]); index-structure spans come from
/// [`collect_chunk_index_spans`]. `widths` are the address and length widths from the
/// superblock.
#[cfg(feature = "std")]
pub(crate) fn collect_chunked_storage_spans<S: Source + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    widths: FormatWidths,
) -> Result<ChunkedStorageSpans, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        index: chunk_index,
        ..
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "collect_chunked_storage_spans called on a non-chunked layout".into(),
        ));
    };
    // An undefined index address means no storage is allocated yet.
    if chunk_index.address().is_none() {
        return Ok(ChunkedStorageSpans {
            data: Vec::new(),
            index: Vec::new(),
        });
    }
    // `chunk_dimensions` is rank + 1 entries; the last is the element size.
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let elem_size = chunk_dimensions[rank];
    if elem_size == 0 {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has a zero element size".into(),
        ));
    }

    let mut data: Vec<(u64, u64)> = Vec::new();

    // Chunk data blocks (the walkers omit unallocated chunks). Build checked extents before the
    // caller can admit any of them to returned-space state.
    for ci in collect_chunks_for_layout_from_source(
        source,
        *chunk_index,
        chunk_dimensions,
        dataspace,
        elem_size,
        widths,
    )? {
        if ci.chunk_size.is_zero() {
            continue;
        }
        let extent = Extent::new(ci.address.get(), ci.chunk_size.get()).ok_or(
            FormatError::OffsetOverflow {
                offset: ci.address.get(),
                length: ci.chunk_size.get(),
            },
        )?;
        data.push((extent.start(), extent.len()));
    }

    // The chunk index's own structure blocks.
    let index = collect_chunk_index_spans(source, *chunk_index, chunk_dimensions.len(), widths)?;
    Ok(ChunkedStorageSpans { data, index })
}

/// A chunked dataset's on-disk footprint, split by the two kinds of structure it
/// occupies: `data` holds the chunk data blocks and `index` holds the chunk
/// index's own structure blocks. Callers that only need the whole footprint can
/// concatenate the two.
///
/// The split is structural, not a page-type classification. The editor places the array indexes
/// it writes beside raw chunks, while a version 2 B-tree remains metadata. A caller reclaiming
/// these spans assigns the page type appropriate to the index kind and its proven placement.
#[cfg(feature = "std")]
pub(crate) struct ChunkedStorageSpans {
    pub(crate) data: Vec<(u64, u64)>,
    pub(crate) index: Vec<(u64, u64)>,
}

/// On-disk byte spans `(addr, len)` of a chunked dataset's index *structure*
/// (not its chunk data): the B-tree v1 nodes, fixed-array header and data block,
/// or extensible-array header, index, super, and data blocks, by index type.
/// Single-chunk and implicit indexes have no separate structure (their footprint is the chunk
/// data alone), so they return an empty vector. Version 2 B-trees use the generic storage walker,
/// which returns the exact `BTHD`, `BTIN`, and `BTLF` allocations.
///
/// `ndims` is `chunk_dimensions.len()` (rank + 1), as [`collect_chunk_info`]
/// takes. Shared by [`collect_chunked_storage_spans`]; kept separate so the
/// per-type dispatch lives next to the chunk-data dispatch it mirrors.
#[cfg(feature = "std")]
fn collect_chunk_index_spans<S: Source + ?Sized>(
    source: &S,
    index: ChunkIndexLayout,
    ndims: usize,
    widths: FormatWidths,
) -> Result<Vec<(u64, u64)>, FormatError> {
    // An undefined index address means the index structure was never written.
    let Some(index_addr) = index.address() else {
        return Ok(Vec::new());
    };
    match index {
        ChunkIndexLayout::BTreeV1 { .. } => {
            collect_chunk_btree_node_spans(source, index_addr, ndims, widths.offsets)
        }
        // Single chunk and implicit indexes have no separate index structure.
        ChunkIndexLayout::SingleChunk { .. } | ChunkIndexLayout::Implicit { .. } => Ok(Vec::new()),
        ChunkIndexLayout::FixedArray { .. } => fixed_array::index_spans(source, widths, index_addr),
        ChunkIndexLayout::ExtensibleArray { .. } => {
            extensible_array::index_spans(source, widths, index_addr)
        }
        ChunkIndexLayout::BTreeV2 { .. } => crate::btree_v2::collect_btree_v2_storage_extents(
            source, index_addr, widths,
        )
        .map(|extents| {
            extents
                .into_iter()
                .map(|extent| (extent.start(), extent.len()))
                .collect()
        }),
    }
}

/// The on-disk byte spans of a chunked dataset's index *structure* (not its chunk
/// data), from an in-memory file image and a parsed [`DataLayout::Chunked`]. A
/// buffered convenience wrapper over [`collect_chunk_index_spans`], used by the
/// in-place editor to test whether a chunked dataset's index occupies a single
/// contiguous region it can rebuild in place. Returns an empty vector for an
/// undefined index address or a single-chunk / implicit index (no separate structure).
#[cfg(feature = "std")]
pub(crate) fn chunk_index_spans_from_source<S: Source + ?Sized>(
    source: &S,
    layout: &DataLayout,
    widths: FormatWidths,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        index,
        ..
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "chunk_index_spans_from_source called on a non-chunked layout".into(),
        ));
    };
    collect_chunk_index_spans(source, *index, chunk_dimensions.len(), widths)
}

/// Reads a whole chunked dataset from a file buffer with index and decoded-chunk caching.
///
/// Dataset dimensions are converted to host-sized indexes only for a nonempty output.
/// Returns an empty vector for a zero-element dataset without consulting the fill pattern.
/// For nonempty output, allocates without initialization if dense coverage is proven and uses
/// the fill pattern otherwise.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if the output byte length or a required
/// buffer index exceeds [`usize::MAX`], and errors from reading the datatype, fill value,
/// chunk geometry, index, or payloads. Returns [`FormatError::DataSizeMismatch`] if a present
/// chunk decodes to a byte length other than its full logical size, and
/// [`FormatError::OffsetOverflow`] if output size or placement arithmetic overflows.
pub fn read_chunked_data_cached(
    file_data: &[u8],
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    read_chunked_data_cached_framed(
        MetadataFrame::new(PhysicalAddress::new(0), file_data)?,
        spec,
        widths,
        cache,
    )
}

/// Reads a whole chunked dataset while borrowing resident payloads.
///
/// The reader resolves metadata positions from the frame's origin and borrows chunk payloads
/// from its slice at local offsets.
///
/// # Errors
///
/// Returns the errors of [`read_chunked_data_cached`].
pub(crate) fn read_chunked_data_cached_framed(
    frame: MetadataFrame<'_>,
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    let file_data = frame.bytes();
    let (dataspace, datatype, pipeline, fill) = (
        spec.dataspace(),
        spec.datatype(),
        spec.pipeline(),
        spec.fill(),
    );
    let RawReadStorage::Chunked {
        flags,
        chunk_dimensions,
        index,
    } = spec.storage()
    else {
        return Err(FormatError::ChunkedReadError(
            "expected chunked layout".into(),
        ));
    };

    if index.address().is_none() {
        return spec.unallocated_buffer();
    }

    // Taken once as a proven-non-zero width; see the buffered reader above.
    let elem_width = datatype.element_size()?;
    let elem_size = elem_width.narrow::<NonZeroUsize>()?;
    let (_, chunk_dims, ds_dims) = chunked_dims(chunk_dimensions, dataspace)?;
    bounded_chunk_len(chunk_dims, elem_size)?;
    let total_bytes = spec.byte_len()?;
    if total_bytes == 0 {
        return Ok(Vec::new());
    }
    let chunk_dims_u64 = chunk_dims;

    let parsed_index = cache.parsed_index_or_load(|| {
        ParsedChunkIndex::new(
            collect_chunks_for_layout(
                frame,
                index,
                chunk_dimensions,
                dataspace,
                elem_width.get().to_u64(),
                widths,
            )?,
            chunk_dims_u64,
            &dataspace.dimensions,
        )
    })?;

    let mut assembly = WholeAssembly::new(&parsed_index, chunk_dims_u64, ds_dims, elem_size, fill)?;

    let ctx = ChunkContext::from_datatype(chunk_dims_u64, datatype)?;

    let pass = cache.begin_pass();
    // One decoder for every chunk of the dataset; see `FilterScratch`.
    let mut scratch = FilterScratch::new();
    for IndexedChunk {
        info: chunk_info,
        coord,
    } in parsed_index.chunks()
    {
        // Scatter straight from the cached chunk under the lock (no copy out).
        let hit = cache.with_decompressed(pass, coord, |bytes| assembly.place(bytes));
        if let Some(placed) = hit {
            placed?;
            continue;
        }

        // Cache miss: read the chunk's bytes from the file.
        let raw_chunk = source::bytes_in(
            file_data,
            FileRange::new(
                PhysicalAddress::new(chunk_info.address.get()),
                chunk_info.chunk_size.get(),
            )?,
        )?;
        let filtering =
            ChunkFiltering::for_chunk(flags, chunk_dims_u64, &dataspace.dimensions, chunk_info);
        // A chunk that skipped the pipeline scatters straight from the file
        // buffer, and is copied into the cache only if it would be retained.
        let dec = decode_chunk(
            &mut scratch,
            Cow::Borrowed(raw_chunk),
            pipeline,
            ctx,
            filtering,
        )?;
        assembly.place(&dec)?;
        match dec {
            Cow::Owned(bytes) => cache.put_decompressed(pass, coord, bytes), // move
            Cow::Borrowed(bytes) => cache.put_decompressed_slice(pass, coord, bytes),
        }
    }

    assembly.finish()
}

/// Walks `index` in the file image `file_data` and returns one [`ChunkInfo`] per chunk the file
/// holds, in index order.
///
/// Returns an empty vector if the index address is undefined, as it is before the dataset's
/// storage is allocated. [`collect_chunks_for_layout_from_source`] walks an index the same way
/// through a [`Source`].
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if `chunk_dimensions` is empty or the chunk geometry
/// is invalid, and the errors of reading a malformed index.
fn collect_chunks_for_layout(
    file_data: MetadataFrame<'_>,
    index: ChunkIndexLayout,
    chunk_dimensions: &[u64],
    dataspace: &Dataspace,
    elem_size: u64,
    widths: FormatWidths,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let Some(addr) = index.address() else {
        return Ok(Vec::new());
    };
    let spatial_dims = &chunk_dimensions[..rank];
    Ok(match index {
        ChunkIndexLayout::BTreeV1 { .. } => {
            collect_chunk_info(file_data.bytes(), addr, ndims, widths.offsets)?
        }
        ChunkIndexLayout::SingleChunk { filtered, .. } => {
            let chunk_byte_size = chunk_grid::chunk_byte_size(spatial_dims, elem_size)?;
            let (chunk_size, filter_mask) = match filtered {
                Some(filtered) => (filtered.filtered_size, filtered.filter_mask),
                None => (chunk_byte_size, 0),
            };
            vec![ChunkInfo {
                chunk_size: StoredChunkSize::v4(chunk_size),
                filter_mask,
                offsets: vec![0u64; rank],
                address: addr,
            }]
        }
        ChunkIndexLayout::Implicit { .. } => generate_implicit_chunks(
            addr,
            &index_grid(dataspace, spatial_dims, GridOrder::RowMajor)?,
            elem_size,
        )?,
        ChunkIndexLayout::FixedArray { .. } => {
            let grid = index_grid(dataspace, spatial_dims, GridOrder::RowMajor)?;
            let mut chunks = Vec::new();
            fixed_array::read_chunks(
                &file_data,
                widths,
                addr,
                grid.chunk_byte_size(elem_size)?,
                |slot, record| {
                    chunks.extend(ChunkInfo::in_extent(&grid, slot, record)?);
                    Ok(())
                },
            )?;
            chunks
        }
        ChunkIndexLayout::ExtensibleArray { .. } => {
            let grid = index_grid(dataspace, spatial_dims, GridOrder::UnlimitedFirst)?;
            let mut chunks = Vec::new();
            extensible_array::read_chunks(
                &file_data,
                widths,
                addr,
                grid.chunk_byte_size(elem_size)?,
                |slot, record| {
                    chunks.extend(ChunkInfo::in_extent(&grid, slot, record)?);
                    Ok(())
                },
            )?;
            chunks
        }
        ChunkIndexLayout::BTreeV2 { .. } => collect_btree_v2_chunks_from_source(
            &file_data,
            addr,
            spatial_dims,
            dataspace,
            elem_size,
            widths,
        )?,
    })
}

fn materialized_dims(dims: &[u64]) -> Result<Vec<usize>, FormatError> {
    dims.iter().map(|&dim| dim.to_usize()).collect()
}

/// Assembles complete decoded chunks in the parsed index's descriptor order.
///
/// Dense output is allocated only with the index's coverage proof for the supplied geometry.
/// Each successful placement writes the descriptor's complete in-extent contribution before
/// advancing to the next descriptor. Finalization requires every descriptor to be placed.
/// An incomplete dense buffer has length zero and can be dropped safely.
struct WholeAssembly<'a> {
    remaining: &'a [ChunkInfo],
    output: AssemblyBuffer,
    geometry: WholeScatter,
    chunk_bytes: usize,
}

impl<'a> WholeAssembly<'a> {
    /// Selects dense or fill-initialized output for the index and geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if a decoded chunk exceeds the allocation
    /// limit, [`FormatError::ValueTooLargeForPlatform`] if a dimension or chunk size exceeds
    /// [`usize::MAX`], [`FormatError::OffsetOverflow`] if stride or output size arithmetic
    /// overflows, and [`FormatError::Internal`] if the ranks differ. Returns
    /// [`FormatError::UnreadableFillValue`] if density is unproven and the fill is unreadable.
    fn new(
        index: &'a ParsedChunkIndex,
        chunk_dims: &[u64],
        ds_dims: &[u64],
        element_size: NonZeroUsize,
        fill: FillPattern<'_>,
    ) -> Result<Self, FormatError> {
        let chunk_bytes = bounded_chunk_len(chunk_dims, element_size)?;
        let geometry = WholeScatter::new(chunk_dims, ds_dims, element_size)?;
        let length = geometry.byte_len()?;
        let output = match index.dense_coverage(chunk_dims, ds_dims) {
            Some(coverage) => AssemblyBuffer::dense(length, coverage),
            None => AssemblyBuffer::Fill(fill.buffer(length)?),
        };
        Ok(Self {
            remaining: index.descriptors(),
            output,
            geometry,
            chunk_bytes,
        })
    }

    /// Writes the next descriptor's complete clipped contribution from `bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if the length of `bytes` differs from the full
    /// logical chunk size, [`FormatError::Internal`] if every descriptor has already been placed,
    /// and the errors of [`WholeScatter::place`].
    fn place(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        let Self {
            remaining,
            output,
            geometry,
            chunk_bytes,
        } = self;
        if bytes.len() != *chunk_bytes {
            return Err(FormatError::DataSizeMismatch {
                expected: *chunk_bytes,
                actual: bytes.len(),
            });
        }
        let (chunk, rest) = remaining.split_first().ok_or_else(|| {
            FormatError::Internal("output assembly has no remaining chunk".into())
        })?;
        geometry.place(output.destination(), chunk, bytes)?;
        *remaining = rest;
        Ok(())
    }

    /// Returns the initialized output after every descriptor has been placed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if any descriptor remains unplaced.
    fn finish(self) -> Result<Vec<u8>, FormatError> {
        let Self {
            remaining,
            output,
            geometry: _,
            chunk_bytes: _,
        } = self;
        if !remaining.is_empty() {
            return Err(FormatError::Internal(
                "output assembly has chunks remaining".into(),
            ));
        }
        Ok(match output {
            AssemblyBuffer::Dense { mut bytes, length } => {
                // SAFETY: DenseCoverage proves the immutable descriptors partition the output.
                // Each descriptor advances only after its complete clipped contribution is written.
                unsafe {
                    bytes.set_len(length);
                }
                bytes
            }
            AssemblyBuffer::Fill(bytes) => bytes,
        })
    }
}

/// Output storage whose dense variant remains at length zero during assembly.
///
/// Writes to dense storage use its [`MaybeUninit`] spare capacity. The length is set after
/// complete placement by [`WholeAssembly::finish`] or [`SelectedAssembly::finish`].
enum AssemblyBuffer {
    Dense { bytes: Vec<u8>, length: usize },
    Fill(Vec<u8>),
}

impl AssemblyBuffer {
    fn dense(length: usize, _coverage: &DenseCoverage) -> Self {
        Self::Dense {
            bytes: Vec::with_capacity(length),
            length,
        }
    }

    fn destination(&mut self) -> ScatterDestination<'_> {
        match self {
            Self::Dense { bytes, length } => {
                ScatterDestination::Spare(&mut bytes.spare_capacity_mut()[..*length])
            }
            Self::Fill(bytes) => ScatterDestination::Initialized(bytes),
        }
    }
}

/// Initialized or spare output bytes for the shared scatter calculations.
///
/// The caller can copy bytes into either form without reading spare storage.
enum ScatterDestination<'a> {
    Initialized(&'a mut [u8]),
    Spare(&'a mut [MaybeUninit<u8>]),
}

impl ScatterDestination<'_> {
    fn copy(&mut self, offset: usize, source: &[u8]) -> Result<(), FormatError> {
        let end = add_offset(offset, source.len())?;
        match self {
            Self::Initialized(bytes) => u8::copy_from_bytes(
                bytes
                    .get_mut(offset..end)
                    .ok_or_else(|| FormatError::Internal("scatter exceeds output extent".into()))?,
                source,
            ),
            Self::Spare(bytes) => {
                let destination = bytes
                    .get_mut(offset..end)
                    .ok_or_else(|| FormatError::Internal("scatter exceeds output extent".into()))?;
                MaybeUninit::<u8>::copy_from_bytes(destination, source);
            }
        }
        Ok(())
    }

    fn copy_rows(&mut self, source: &[u8], rows: &ScatterRows) -> Result<(), FormatError> {
        match self {
            Self::Initialized(bytes) => rows.copy_to(source, bytes),
            Self::Spare(bytes) => rows.copy_to(source, bytes),
        }
    }
}

/// Copy geometry for equally sized rows with independent source and destination byte strides.
///
/// Both strides are nonzero, and each is at least the row byte count. The complete source and
/// destination spans are checked before any row is copied.
struct ScatterRows {
    source_stride: NonZeroUsize,
    destination_stride: NonZeroUsize,
    row_bytes: usize,
    count: usize,
    destination_offset: usize,
}

impl ScatterRows {
    /// Constructs row geometry whose rows fit within both strides.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if either stride is zero or less than `row_bytes`.
    fn new(
        source_stride: usize,
        destination_stride: usize,
        row_bytes: usize,
        count: usize,
        destination_offset: usize,
    ) -> Result<Self, FormatError> {
        if row_bytes > source_stride || row_bytes > destination_stride {
            return Err(FormatError::Internal(
                "scatter row exceeds its stride".into(),
            ));
        }
        Ok(Self {
            source_stride: NonZeroUsize::new(source_stride)
                .ok_or_else(|| FormatError::Internal("scatter source stride is zero".into()))?,
            destination_stride: NonZeroUsize::new(destination_stride).ok_or_else(|| {
                FormatError::Internal("scatter destination stride is zero".into())
            })?,
            row_bytes,
            count,
            destination_offset,
        })
    }

    /// Copies each row after checking the complete source and destination spans.
    ///
    /// Returns without writing if the row count or byte count is zero.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if span arithmetic overflows, and
    /// [`FormatError::Internal`] if a span exceeds its buffer.
    fn copy_to<Byte: ScatterByte>(
        &self,
        source: &[u8],
        destination: &mut [Byte],
    ) -> Result<(), FormatError> {
        let &Self {
            source_stride,
            destination_stride,
            row_bytes,
            count,
            destination_offset,
        } = self;
        if count == 0 || row_bytes == 0 {
            return Ok(());
        }
        let source_end = add_offset(multiply_offset(count - 1, source_stride.get())?, row_bytes)?;
        let destination_end = add_offset(
            destination_offset,
            add_offset(
                multiply_offset(count - 1, destination_stride.get())?,
                row_bytes,
            )?,
        )?;
        let source = source
            .get(..source_end)
            .ok_or_else(|| FormatError::Internal("scatter exceeds decoded chunk extent".into()))?;
        let destination = destination
            .get_mut(destination_offset..destination_end)
            .ok_or_else(|| FormatError::Internal("scatter exceeds output extent".into()))?;
        for (source, destination) in source
            .chunks(source_stride.get())
            .zip(destination.chunks_mut(destination_stride.get()))
        {
            let source = source
                .get(..row_bytes)
                .ok_or_else(|| FormatError::Internal("scatter source row is short".into()))?;
            let destination = destination
                .get_mut(..row_bytes)
                .ok_or_else(|| FormatError::Internal("scatter destination row is short".into()))?;
            Byte::copy_from_bytes(destination, source);
        }
        Ok(())
    }
}

/// A byte destination that can be initialized from a byte slice.
///
/// Implemented for [`u8`] and [`MaybeUninit<u8>`] so both output states use the same row geometry.
trait ScatterByte: Sized {
    /// Copies `source` into every element of an equal-length `destination`.
    ///
    /// # Panics
    ///
    /// Panics if the slice lengths differ.
    fn copy_from_bytes(destination: &mut [Self], source: &[u8]);
}

impl ScatterByte for u8 {
    #[inline]
    fn copy_from_bytes(destination: &mut [Self], source: &[u8]) {
        destination.copy_from_slice(source);
    }
}

impl ScatterByte for MaybeUninit<u8> {
    #[inline]
    fn copy_from_bytes(destination: &mut [Self], source: &[u8]) {
        assert_eq!(destination.len(), source.len());
        for (destination, &source) in destination.iter_mut().zip(source) {
            destination.write(source);
        }
    }
}

/// Host-sized row-major geometry and reusable coordinates for whole-dataset placement.
struct WholeScatter {
    chunk_dims: Vec<usize>,
    ds_dims: Vec<usize>,
    ds_strides: Vec<usize>,
    chunk_strides: Vec<usize>,
    offsets: Vec<usize>,
    coord: Vec<usize>,
    element_size: NonZeroUsize,
}

impl WholeScatter {
    /// Converts dimensions to buffer indexes and computes checked element strides.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the ranks differ,
    /// [`FormatError::ValueTooLargeForPlatform`] if a dimension exceeds [`usize::MAX`], and
    /// [`FormatError::OffsetOverflow`] if a stride calculation overflows.
    fn new(
        chunk_dims: &[u64],
        ds_dims: &[u64],
        element_size: NonZeroUsize,
    ) -> Result<Self, FormatError> {
        if chunk_dims.len() != ds_dims.len() {
            return Err(FormatError::Internal("scatter ranks differ".into()));
        }
        let chunk_dims = materialized_dims(chunk_dims)?;
        let ds_dims = materialized_dims(ds_dims)?;
        Ok(Self {
            ds_strides: row_major_strides(&ds_dims)?,
            chunk_strides: row_major_strides(&chunk_dims)?,
            offsets: Vec::with_capacity(chunk_dims.len()),
            coord: vec![0; chunk_dims.len()],
            chunk_dims,
            ds_dims,
            element_size,
        })
    }

    fn place(
        &mut self,
        output: ScatterDestination<'_>,
        chunk: &ChunkInfo,
        bytes: &[u8],
    ) -> Result<(), FormatError> {
        let Self {
            chunk_dims,
            ds_dims,
            ds_strides,
            chunk_strides,
            offsets,
            coord,
            element_size,
        } = self;
        offsets.clear();
        for &offset in chunk.offsets.iter().take(chunk_dims.len()) {
            offsets.push(offset.to_usize()?);
        }
        place_chunk(
            bytes,
            output,
            offsets,
            chunk_dims,
            ds_dims,
            ds_strides,
            chunk_strides,
            *element_size,
            coord,
        )
    }

    /// Returns the output byte count for the dataset extent and element size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the byte count exceeds [`usize::MAX`].
    fn byte_len(&self) -> Result<usize, FormatError> {
        self.ds_dims
            .iter()
            .try_fold(self.element_size.get(), |length, &dimension| {
                multiply_offset(length, dimension)
            })
    }
}

/// Writes every in-extent element of a chunk through shared row-major placement geometry.
///
/// Copies contiguous trailing dimensions together, uses a fixed-stride loop for one outer
/// dimension, and reuses `coord` for higher ranks. Reads complete source runs and initializes
/// complete destination runs, including when the destination is spare capacity.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the ranks differ or a copy exceeds the source or
/// destination extent, and [`FormatError::OffsetOverflow`] if placement arithmetic overflows.
#[allow(clippy::too_many_arguments)]
fn place_chunk(
    chunk_data: &[u8],
    mut output: ScatterDestination<'_>,
    chunk_offsets: &[usize],
    chunk_dims: &[usize],
    ds_dims: &[usize],
    ds_strides: &[usize],
    chunk_strides: &[usize],
    elem_size: NonZeroUsize,
    coord: &mut [usize],
) -> Result<(), FormatError> {
    let rank = ds_dims.len();
    if [
        chunk_offsets.len(),
        chunk_dims.len(),
        ds_strides.len(),
        chunk_strides.len(),
        coord.len(),
    ]
    .iter()
    .any(|&length| length != rank)
    {
        return Err(FormatError::Internal("scatter ranks differ".into()));
    }
    if rank == 0 {
        return copy_scatter_run(chunk_data, &mut output, 0, 0, elem_size.get());
    }
    if chunk_dims.contains(&0) {
        return Ok(());
    }
    if chunk_offsets
        .iter()
        .zip(ds_dims)
        .any(|(&offset, &extent)| offset >= extent)
    {
        return Ok(());
    }
    let mut inner = rank - 1;
    let count = |axis: usize| chunk_dims[axis].min(ds_dims[axis] - chunk_offsets[axis]);
    let mut contiguous = count(inner);
    while inner > 0
        && chunk_offsets[inner] == 0
        && count(inner) == chunk_dims[inner]
        && count(inner) == ds_dims[inner]
        && chunk_strides[inner - 1] == contiguous
        && ds_strides[inner - 1] == contiguous
    {
        inner -= 1;
        contiguous = multiply_offset(contiguous, count(inner))?;
    }
    let length = multiply_offset(contiguous, elem_size.get())?;
    let base = chunk_offsets
        .iter()
        .zip(ds_strides)
        .try_fold(0, |base, (&offset, &stride)| {
            add_offset(base, multiply_offset(offset, stride)?)
        })?;
    if inner == 0 {
        return copy_scatter_run(
            chunk_data,
            &mut output,
            0,
            multiply_offset(base, elem_size.get())?,
            length,
        );
    }
    if inner == 1 {
        return output.copy_rows(
            chunk_data,
            &ScatterRows::new(
                multiply_offset(chunk_strides[0], elem_size.get())?,
                multiply_offset(ds_strides[0], elem_size.get())?,
                length,
                count(0),
                multiply_offset(base, elem_size.get())?,
            )?,
        );
    }
    coord.fill(0);
    loop {
        let (source, destination) = coord[..inner]
            .iter()
            .zip(&chunk_strides[..inner])
            .zip(&ds_strides[..inner])
            .try_fold(
                (0, base),
                |(source, destination),
                 ((&position, &chunk_stride), &ds_stride)|
                 -> Result<_, FormatError> {
                    Ok((
                        add_offset(source, multiply_offset(position, chunk_stride)?)?,
                        add_offset(destination, multiply_offset(position, ds_stride)?)?,
                    ))
                },
            )?;
        copy_scatter_run(
            chunk_data,
            &mut output,
            multiply_offset(source, elem_size.get())?,
            multiply_offset(destination, elem_size.get())?,
            length,
        )?;
        let mut axis = inner;
        loop {
            if axis == 0 {
                return Ok(());
            }
            axis -= 1;
            coord[axis] += 1;
            if coord[axis] < count(axis) {
                break;
            }
            coord[axis] = 0;
        }
    }
}

#[inline(always)]
fn copy_scatter_run(
    source: &[u8],
    destination: &mut ScatterDestination<'_>,
    source_offset: usize,
    destination_offset: usize,
    length: usize,
) -> Result<(), FormatError> {
    let end = add_offset(source_offset, length)?;
    destination.copy(
        destination_offset,
        source
            .get(source_offset..end)
            .ok_or_else(|| FormatError::Internal("scatter exceeds decoded chunk extent".into()))?,
    )
}

fn add_offset(offset: usize, length: usize) -> Result<usize, FormatError> {
    match offset.checked_add(length) {
        Some(value) => Ok(value),
        None => Err(FormatError::OffsetOverflow {
            offset: offset as u64,
            length: length as u64,
        }),
    }
}

fn multiply_offset(offset: usize, length: usize) -> Result<usize, FormatError> {
    match offset.checked_mul(length) {
        Some(value) => Ok(value),
        None => Err(FormatError::OffsetOverflow {
            offset: offset as u64,
            length: length as u64,
        }),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::boxed::Box;

    use core::ops::Range;
    #[cfg(feature = "std")]
    use std::io::Cursor;

    #[cfg(feature = "std")]
    use hdf5_pure_format::__private::FilteredSingleChunk;
    use rstest::rstest;
    use test_util::btree_v1;
    use test_util::btree_v2;
    use test_util::bytes as test_bytes;
    use test_util::widths::Widths;

    use super::*;
    use crate::FillTime;
    #[cfg(feature = "std")]
    use crate::address::{BaseAddress, BaseAddressExt};
    use crate::chunked_write::ChunkArrayKind;
    use crate::convert;
    use crate::dataspace::MaxExtent;
    use crate::filter_pipeline::FilterDescription;
    #[cfg(feature = "std")]
    use crate::source::ReadSeekSource;
    #[cfg(feature = "std")]
    use crate::source::{BaseOffsetSource, MetadataCacheConfig, MetadataCachingSource};

    const FORMAT_WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: crate::width::LengthWidth::Eight,
    };

    #[rstest]
    #[case::identity(false, ChunkFiltering::Filtered { filter_mask: 0 })]
    #[case::raw_edge(true, ChunkFiltering::Raw)]
    #[case::skipped(true, ChunkFiltering::Filtered { filter_mask: FIRST_FILTER_MASK })]
    #[case::filtered(true, ChunkFiltering::Filtered { filter_mask: 0 })]
    fn short_and_oversized_chunks_return_size_mismatches(
        #[case] pipeline: bool,
        #[case] filtering: ChunkFiltering,
        #[values(4, 12)] stored_len: usize,
    ) {
        let pipeline = pipeline.then(shuffle_pipeline);
        let stored = vec![7; stored_len];
        assert_eq!(
            decode_chunk(
                &mut FilterScratch::new(),
                Cow::Borrowed(&stored),
                pipeline.as_ref(),
                ChunkContext::basic(&[2], 4),
                filtering,
            )
            .unwrap_err(),
            FormatError::DataSizeMismatch {
                expected: 8,
                actual: stored_len
            }
        );
    }

    #[rstest]
    #[case::no_pipeline(false)]
    #[case::raw_with_pipeline(true)]
    fn complete_identity_chunks_remain_borrowed(#[case] pipeline: bool) {
        let pipeline = pipeline.then(shuffle_pipeline);
        let stored = [7; 8];
        let decoded = decode_chunk(
            &mut FilterScratch::new(),
            Cow::Borrowed(&stored),
            pipeline.as_ref(),
            ChunkContext::basic(&[2], 4),
            ChunkFiltering::Raw,
        )
        .unwrap();
        let Cow::Borrowed(bytes) = decoded else {
            panic!("expected borrowed chunk bytes, got {decoded:?}");
        };
        assert_eq!(bytes, stored);
        assert_eq!(bytes.as_ptr(), stored.as_ptr());
    }

    #[cfg(feature = "std")]
    #[rstest]
    fn malformed_chunks_fail_uncached_reads_and_visitors(
        #[values(4, 12)] stored_len: usize,
        #[values(false, true)] raw_edge: bool,
    ) {
        let bytes = vec![7; stored_len];
        let flags = hdf5_pure_format::__private::SINGLE_INDEX_WITH_FILTER
            | if raw_edge {
                crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS
            } else {
                0
            };
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::new(flags),
            chunk_dimensions: vec![2, 4],
            index: ChunkIndexLayout::SingleChunk {
                filtered: Some(FilteredSingleChunk {
                    filtered_size: u64::try_from(stored_len).unwrap(),
                    filter_mask: 0,
                }),
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![if raw_edge { 1 } else { 2 }],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let pipeline = shuffle_pipeline();
        let spec = RawReadSpec::parse(
            &layout,
            &dataspace,
            &datatype,
            Some(&pipeline),
            FillPattern::ZERO,
        )
        .unwrap();
        let source = BytesSource::new(&bytes);
        let expected = FormatError::DataSizeMismatch {
            expected: 8,
            actual: stored_len,
        };
        assert_eq!(
            read_chunked_data_from_source(&source, spec, FORMAT_WIDTHS).unwrap_err(),
            expected
        );
        let cache = ChunkCache::new();
        assert_eq!(
            read_chunked_data_cached_from_source(&source, spec, FORMAT_WIDTHS, &cache).unwrap_err(),
            expected
        );
        assert_eq!(cache.stats().cached_chunks(), 0);
        let mut visits = 0;
        visit_decoded_chunks_from_source(&source, spec, FORMAT_WIDTHS, |decoded| {
            assert_eq!(decoded.unwrap_err(), &expected);
            visits += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(visits, 1);
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::cached_bytes(ReadMode::CachedBytes)]
    #[case::cached_seek_source(ReadMode::CachedSeekSource)]
    #[case::cached_source(ReadMode::CachedSource)]
    #[case::seek_source(ReadMode::SeekSource)]
    #[case::source(ReadMode::Source)]
    fn whole_reads_consult_fill_only_for_uncovered_storage(
        #[case] reader: ReadMode,
        #[values(
            StorageFill::DenseZero,
            StorageFill::DenseNonzero,
            StorageFill::DenseUnknown,
            StorageFill::SparseNonzero,
            StorageFill::SparseUnknown
        )]
        fill: StorageFill,
        #[values(false, true)] filtered: bool,
    ) {
        let (sparse, unknown, fill_value): (bool, bool, i32) = match fill {
            StorageFill::DenseNonzero => (false, false, 7),
            StorageFill::DenseUnknown => (false, true, 0),
            StorageFill::DenseZero => (false, false, 0),
            StorageFill::SparseNonzero => (true, false, 7),
            StorageFill::SparseUnknown => (true, true, 0),
        };
        let mut file_data = vec![0; 0x2000];
        let mut chunks = Vec::new();
        for (i, offsets) in [[0, 0], [0, 3], [2, 0], [2, 3]].into_iter().enumerate() {
            if sparse && offsets == [0, 3] {
                continue;
            }
            let address = 0x1000 + i * 32;
            let bytes: Vec<_> = (0..2)
                .flat_map(|row| {
                    (0..3).flat_map(move |col| {
                        ((offsets[0] + row) * 5i32 + offsets[1] + col).to_le_bytes()
                    })
                })
                .collect();
            let mask = if i % 2 == 0 { 0 } else { FIRST_FILTER_MASK };
            let bytes = if filtered && mask == 0 {
                filters::compress_chunk(
                    &bytes,
                    &shuffle_pipeline(),
                    ChunkContext::basic(&[2, 3], 4),
                )
                .unwrap()
            } else {
                bytes
            };
            file_data[address..address + bytes.len()].copy_from_slice(&bytes);
            chunks.push(ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(24),
                filter_mask: mask,
                offsets: vec![offsets[0] as u64, offsets[1] as u64, 0],
                address: StoredAddress::new(address as u64),
            });
        }
        let btree = build_chunk_btree_leaf(&chunks, 3, WIDTHS);
        file_data[0x100..0x100 + btree.len()].copy_from_slice(&btree);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 3, 4],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0x100)),
            },
        };
        let dataspace = Dataspace {
            space_type: crate::dataspace::DataspaceType::Simple,
            rank: 2,
            dimensions: vec![3, 5],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let fill_bytes = fill_value.to_le_bytes();
        let fill = if unknown {
            FillPattern::UNKNOWN
        } else {
            FillPattern::new(FillTime::IfSet, Some(&fill_bytes), convert::nz(4))
        };
        let pipeline = filtered.then(shuffle_pipeline);
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, pipeline.as_ref(), fill).unwrap();
        let cache = ChunkCache::new();
        let read = || reader.read(&file_data, spec, &cache);
        if sparse && unknown {
            assert_eq!(read().unwrap_err(), FormatError::UnreadableFillValue);
            assert_eq!(read().unwrap_err(), FormatError::UnreadableFillValue);
            return;
        }
        let expected: Vec<_> = (0..15i32)
            .flat_map(|value| {
                if sparse && value < 10 && value % 5 >= 3 {
                    fill_value.to_le_bytes()
                } else {
                    value.to_le_bytes()
                }
            })
            .collect();
        assert_eq!(read().unwrap(), expected);
        assert_eq!(read().unwrap(), expected);
        assert_eq!(
            cache.stats().hits(),
            if matches!(reader, ReadMode::Source | ReadMode::SeekSource) {
                0
            } else if sparse {
                3
            } else {
                4
            }
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::dense_subset(false, true, &[1, 1], &[2, 3], Some(&[6, 7, 8, 11, 12, 13][..]))]
    #[case::sparse_covered_subset(true, true, &[1, 1], &[1, 2], Some(&[6, 7][..]))]
    #[case::clipped_edge(true, true, &[2, 3], &[1, 2], Some(&[13, 14][..]))]
    #[case::sparse_unknown(true, true, &[1, 2], &[2, 3], None)]
    #[case::sparse_defined(true, false, &[1, 2], &[2, 3], Some(&[7, 17, 17, 12, 13, 14][..]))]
    #[case::empty(true, true, &[1, 1], &[0, 2], Some(&[][..]))]
    fn selected_reads_consult_fill_only_for_uncovered_storage(
        #[case] sparse: bool,
        #[case] unknown: bool,
        #[case] start: &[u64],
        #[case] count: &[u64],
        #[case] expected: Option<&[i32]>,
        #[values(false, true)] streaming: bool,
        #[values(false, true)] filtered: bool,
        #[values(false, true)] cached: bool,
    ) {
        let mut file_data = vec![0; 0x2000];
        let mut chunks = Vec::new();
        for (i, offsets) in [[0, 0], [0, 3], [2, 0], [2, 3]].into_iter().enumerate() {
            if sparse && offsets == [0, 3] {
                continue;
            }
            let address = 0x1000 + i * 32;
            let bytes: Vec<_> = (0..2)
                .flat_map(|row| {
                    (0..3).flat_map(move |col| {
                        ((offsets[0] + row) * 5i32 + offsets[1] + col).to_le_bytes()
                    })
                })
                .collect();
            let mask = if i % 2 == 0 { 0 } else { FIRST_FILTER_MASK };
            let bytes = if filtered && mask == 0 {
                filters::compress_chunk(
                    &bytes,
                    &shuffle_pipeline(),
                    ChunkContext::basic(&[2, 3], 4),
                )
                .unwrap()
            } else {
                bytes
            };
            file_data[address..address + bytes.len()].copy_from_slice(&bytes);
            chunks.push(ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(24),
                filter_mask: mask,
                offsets: vec![offsets[0] as u64, offsets[1] as u64, 0],
                address: StoredAddress::new(address as u64),
            });
        }
        let btree = build_chunk_btree_leaf(&chunks, 3, WIDTHS);
        file_data[0x100..0x100 + btree.len()].copy_from_slice(&btree);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 3, 4],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0x100)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![3, 5],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let fill_bytes = 17i32.to_le_bytes();
        let fill = if unknown {
            FillPattern::UNKNOWN
        } else {
            FillPattern::new(FillTime::IfSet, Some(&fill_bytes), convert::nz(4))
        };
        let pipeline = filtered.then(shuffle_pipeline);
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, pipeline.as_ref(), fill).unwrap();
        let region = ReadRegion::parse(spec, start, count).unwrap();
        let source: Box<dyn Source + '_> = if streaming {
            Box::new(ReadSeekSource::new(Cursor::new(&file_data)).unwrap())
        } else {
            Box::new(BytesSource::new(&file_data))
        };
        let cache = ChunkCache::with_config(if cached {
            crate::ChunkCacheConfig::new()
        } else {
            crate::ChunkCacheConfig::disabled()
        });
        let read = || {
            read_chunked_region_from_source(
                source.as_ref(),
                &region,
                FORMAT_WIDTHS,
                &cache,
                CachePass::LRU,
            )
        };
        let expected = expected
            .map(|values| {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .ok_or(FormatError::UnreadableFillValue);
        assert_eq!(read(), expected);
        assert_eq!(read(), expected);
        if !cached {
            assert!(!cache.stats().index_loaded());
        }
    }

    #[cfg(feature = "std")]
    #[derive(Clone, Copy)]
    enum StorageFill {
        DenseNonzero,
        DenseUnknown,
        DenseZero,
        SparseNonzero,
        SparseUnknown,
    }

    #[cfg(feature = "std")]
    #[derive(Clone, Copy)]
    enum ReadMode {
        CachedBytes,
        CachedSeekSource,
        CachedSource,
        SeekSource,
        SelectedSeekSource,
        SelectedSource,
        Source,
    }

    #[cfg(feature = "std")]
    impl ReadMode {
        fn read(
            self,
            bytes: &[u8],
            spec: RawReadSpec<'_>,
            cache: &ChunkCache,
        ) -> Result<Vec<u8>, FormatError> {
            match self {
                Self::CachedBytes => read_chunked_data_cached(bytes, spec, FORMAT_WIDTHS, cache),
                Self::CachedSeekSource => read_chunked_data_cached_from_source(
                    &ReadSeekSource::new(Cursor::new(bytes)).unwrap(),
                    spec,
                    FORMAT_WIDTHS,
                    cache,
                ),
                Self::CachedSource => read_chunked_data_cached_from_source(
                    &BytesSource::new(bytes),
                    spec,
                    FORMAT_WIDTHS,
                    cache,
                ),
                Self::SeekSource => read_chunked_data_from_source(
                    &ReadSeekSource::new(Cursor::new(bytes)).unwrap(),
                    spec,
                    FORMAT_WIDTHS,
                ),
                Self::SelectedSeekSource => {
                    let region = ReadRegion::rows(spec, 1, 2)?;
                    read_chunked_region_from_source(
                        &ReadSeekSource::new(Cursor::new(bytes)).unwrap(),
                        &region,
                        FORMAT_WIDTHS,
                        cache,
                        CachePass::LRU,
                    )
                }
                Self::SelectedSource => {
                    let region = ReadRegion::rows(spec, 1, 2)?;
                    read_chunked_region_from_source(
                        &BytesSource::new(bytes),
                        &region,
                        FORMAT_WIDTHS,
                        cache,
                        CachePass::LRU,
                    )
                }
                Self::Source => {
                    read_chunked_data_from_source(&BytesSource::new(bytes), spec, FORMAT_WIDTHS)
                }
            }
        }
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::cached_bytes(ReadMode::CachedBytes)]
    #[case::cached_seek_source(ReadMode::CachedSeekSource)]
    #[case::cached_source(ReadMode::CachedSource)]
    #[case::seek_source(ReadMode::SeekSource)]
    #[case::selected_seek_source(ReadMode::SelectedSeekSource)]
    #[case::selected_source(ReadMode::SelectedSource)]
    #[case::source(ReadMode::Source)]
    fn chunked_reads_reject_malformed_later_chunks(
        #[case] reader: ReadMode,
        #[values(4, 12)] length: usize,
        #[values(false, true)] cached: bool,
    ) {
        let chunks = [
            ChunkInfo {
                offsets: vec![0, 0],
                address: StoredAddress::new(512),
                chunk_size: StoredChunkSize::btree_v1(8),
                filter_mask: 0,
            },
            ChunkInfo {
                offsets: vec![2, 0],
                address: StoredAddress::new(520),
                chunk_size: StoredChunkSize::btree_v1(u32::try_from(length).unwrap()),
                filter_mask: 0,
            },
        ];
        let mut bytes = build_chunk_btree_leaf(&chunks, 2, WIDTHS);
        bytes.resize(520 + length, 1);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 4],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![4],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::UNKNOWN).unwrap();
        let cache = ChunkCache::new();
        if cached {
            cache.put_decompressed(CachePass::LRU, &[2], vec![1; length]);
        }
        assert_eq!(
            reader.read(&bytes, spec, &cache).unwrap_err(),
            FormatError::DataSizeMismatch {
                expected: 8,
                actual: length
            }
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::cached_bytes(ReadMode::CachedBytes)]
    #[case::cached_seek_source(ReadMode::CachedSeekSource)]
    #[case::cached_source(ReadMode::CachedSource)]
    #[case::seek_source(ReadMode::SeekSource)]
    #[case::selected_seek_source(ReadMode::SelectedSeekSource)]
    #[case::selected_source(ReadMode::SelectedSource)]
    #[case::source(ReadMode::Source)]
    fn chunked_reads_abandon_io_and_decode_errors(
        #[case] reader: ReadMode,
        #[values(ReadFailure::Io, ReadFailure::Decoding)] failure: ReadFailure,
    ) {
        let (decoding, expected) = match failure {
            ReadFailure::Decoding => (true, FormatError::UnsupportedFilter(999)),
            ReadFailure::Io => (
                false,
                FormatError::UnexpectedSourceEof {
                    expected: 528,
                    available: 524,
                },
            ),
        };
        let chunks = [
            ChunkInfo {
                offsets: vec![0, 0],
                address: StoredAddress::new(512),
                chunk_size: StoredChunkSize::btree_v1(8),
                filter_mask: FIRST_FILTER_MASK,
            },
            ChunkInfo {
                offsets: vec![2, 0],
                address: StoredAddress::new(520),
                chunk_size: StoredChunkSize::btree_v1(8),
                filter_mask: 0,
            },
        ];
        let mut bytes = build_chunk_btree_leaf(&chunks, 2, WIDTHS);
        bytes.resize(if decoding { 528 } else { 524 }, 1);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 4],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![4],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let pipeline = decoding.then(|| FilterPipeline {
            version: FILTER_PIPELINE_VERSION,
            filters: vec![FilterDescription {
                filter_id: 999,
                name: None,
                flags: 0,
                client_data: vec![],
            }],
        });
        let spec = RawReadSpec::parse(
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            FillPattern::UNKNOWN,
        )
        .unwrap();
        assert_eq!(
            reader.read(&bytes, spec, &ChunkCache::new()).unwrap_err(),
            expected
        );
    }

    #[cfg(feature = "std")]
    #[derive(Clone, Copy)]
    enum ReadFailure {
        Decoding,
        Io,
    }

    fn assembly_index(chunk_dims: &[u64], ds_dims: &[u64]) -> ParsedChunkIndex {
        let grid = ChunkGrid::new(chunk_dims, ds_dims, None, GridOrder::RowMajor).unwrap();
        let chunks = grid
            .slots_in_extent()
            .unwrap()
            .enumerate()
            .map(|(slot, chunk)| ChunkInfo {
                offsets: chunk.offsets,
                chunk_size: StoredChunkSize::v4(chunk_dims.iter().product::<u64>() * 4),
                filter_mask: 0,
                address: StoredAddress::new(0x1000 - u64::try_from(slot).unwrap() * 32),
            })
            .collect();
        ParsedChunkIndex::new(chunks, chunk_dims, ds_dims).unwrap()
    }

    fn assembly_payload(chunk: &ChunkInfo, chunk_dims: &[u64], ds_dims: &[u64]) -> Vec<u8> {
        (0..chunk_dims.iter().product::<u64>())
            .flat_map(|mut flat| {
                let mut linear = 0;
                let mut stride = 1;
                for ((&dimension, &extent), &offset) in
                    chunk_dims.iter().zip(ds_dims).zip(&chunk.offsets).rev()
                {
                    linear += (offset + flat % dimension) * stride;
                    flat /= dimension;
                    stride *= extent;
                }
                i32::try_from(linear).unwrap().to_le_bytes()
            })
            .collect()
    }

    #[rstest]
    #[case::band(&[2, 5], &[3, 5], &[1, 0], &[2, 5])]
    #[case::scatter(&[2, 3], &[3, 5], &[1, 1], &[2, 3])]
    #[case::overhang(&[2, 7], &[3, 5], &[1, 1], &[2, 4])]
    #[case::volume(&[2, 3, 2], &[3, 5, 3], &[1, 1, 1], &[2, 4, 2])]
    #[case::four_dimensions(&[2, 2, 2, 2], &[3, 3, 3, 3], &[1, 0, 1, 0], &[2, 3, 2, 3])]
    fn dense_assembly_selected_writes_every_clipped_element(
        #[case] chunk_dims: &[u64],
        #[case] ds_dims: &[u64],
        #[case] start: &[u64],
        #[case] count: &[u64],
    ) {
        let index = assembly_index(chunk_dims, ds_dims);
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: u8::try_from(ds_dims.len()).unwrap(),
            dimensions: ds_dims.to_vec(),
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::UNKNOWN).unwrap();
        let region = ReadRegion::parse(spec, start, count).unwrap();
        let selection = index.select(&region).unwrap();
        let mut assembly = SelectedAssembly::new(selection).unwrap();
        let AssemblyBuffer::Dense { bytes, length } = &assembly.output else {
            panic!("expected dense assembly");
        };
        assert_eq!(bytes.len(), 0);
        assert_eq!(*length, region.byte_len());
        let selection = index.select(&region).unwrap();
        for selected in selection.chunks() {
            assembly
                .place(&assembly_payload(selected.chunk.info, chunk_dims, ds_dims))
                .unwrap();
        }
        let whole: Vec<_> = (0..i32::try_from(ds_dims.iter().product::<u64>()).unwrap())
            .flat_map(i32::to_le_bytes)
            .collect();
        let mut expected = Vec::new();
        region
            .visit_runs(|offset, length| {
                let offset = usize::try_from(offset).unwrap();
                expected.extend_from_slice(&whole[offset..offset + length]);
                Ok(())
            })
            .unwrap();
        assert_eq!(assembly.finish().unwrap(), expected);
    }

    #[rstest]
    #[case::unplaced(0, None)]
    #[case::partly_placed(1, None)]
    #[case::short(1, Some(4))]
    #[case::oversized(1, Some(12))]
    fn dense_assembly_selected_rejects_incomplete_contributions(
        #[case] completed: usize,
        #[case] malformed: Option<usize>,
    ) {
        let index = assembly_index(&[2], &[4]);
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![4],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::UNKNOWN).unwrap();
        let region = ReadRegion::rows(spec, 1, 2).unwrap();
        let mut assembly = SelectedAssembly::new(index.select(&region).unwrap()).unwrap();
        for _ in 0..completed {
            assembly.place(&[1; 8]).unwrap();
        }
        if let Some(length) = malformed {
            assert_eq!(
                assembly.place(&vec![2; length]).unwrap_err(),
                FormatError::DataSizeMismatch {
                    expected: 8,
                    actual: length
                }
            );
        }
        assert_eq!(
            assembly.finish().unwrap_err(),
            FormatError::Internal("output assembly has chunks remaining".into())
        );
    }

    #[rstest]
    #[case::duplicate(&[0, 0], &[21, 22], &[21, 22, 0, 0])]
    #[case::misaligned(&[1], &[11, 12], &[0, 11, 12, 0])]
    fn selected_assembly_keeps_conservative_fill_for_unusual_descriptors(
        #[case] offsets: &[u64],
        #[case] expected_selected: &[i32],
        #[case] expected_whole: &[i32],
    ) {
        let chunks = offsets
            .iter()
            .enumerate()
            .map(|(i, &offset)| ChunkInfo {
                offsets: vec![offset],
                chunk_size: StoredChunkSize::v4(8),
                filter_mask: 0,
                address: StoredAddress::new(0x1000 + i as u64 * 16),
            })
            .collect();
        let index = ParsedChunkIndex::new(chunks, &[2], &[4]).unwrap();
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![4],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::UNKNOWN).unwrap();
        let region = ReadRegion::rows(spec, offsets[0], 2).unwrap();
        let Err(error) = SelectedAssembly::new(index.select(&region).unwrap()) else {
            panic!("expected unreadable fill");
        };
        assert_eq!(error, FormatError::UnreadableFillValue);
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::rows(spec, offsets[0], 2).unwrap();
        let mut selected = SelectedAssembly::new(index.select(&region).unwrap()).unwrap();
        let mut whole =
            WholeAssembly::new(&index, &[2], &[4], convert::nz(4), FillPattern::ZERO).unwrap();
        for (i, _) in index.descriptors().iter().enumerate() {
            let payload: Vec<_> = [11 + i as i32 * 10, 12 + i as i32 * 10]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect();
            selected.place(&payload).unwrap();
            whole.place(&payload).unwrap();
        }
        assert_eq!(
            whole.finish().unwrap(),
            expected_whole
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            selected.finish().unwrap(),
            expected_selected
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn selected_empty_reads_skip_fill_and_chunk_index(#[values(false, true)] allocated: bool) {
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 4],
            index: ChunkIndexLayout::BTreeV1 {
                address: allocated.then_some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![4],
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::UNKNOWN).unwrap();
        let region = ReadRegion::rows(spec, 1, 0).unwrap();
        assert_eq!(
            read_chunked_region_from_source(
                &BytesSource::new(&[]),
                &region,
                FORMAT_WIDTHS,
                &ChunkCache::new(),
                CachePass::LRU
            )
            .unwrap(),
            Vec::<u8>::new()
        );
        let index = assembly_index(&[2], &[4]);
        assert_eq!(
            SelectedAssembly::new(index.select(&region).unwrap())
                .unwrap()
                .finish()
                .unwrap(),
            Vec::<u8>::new()
        );
    }

    #[rstest]
    #[case::rank_two(&[2, 3], &[3, 3], &[0, 1], &[0, 11, 12, 0, 14, 15, 0, 0, 0])]
    #[case::rank_three(
        &[2, 2, 3], &[3, 2, 3], &[0, 0, 1],
        &[0, 11, 12, 0, 14, 15, 0, 17, 18, 0, 20, 21, 0, 0, 0, 0, 0, 0]
    )]
    #[case::rank_three_middle(
        &[2, 2, 3], &[3, 2, 3], &[0, 1, 0],
        &[0, 0, 0, 11, 12, 13, 0, 0, 0, 17, 18, 19, 0, 0, 0, 0, 0, 0]
    )]
    fn chunk_scatter_clips_nonzero_trailing_offsets(
        #[case] chunk_dims: &[u64],
        #[case] ds_dims: &[u64],
        #[case] offsets: &[u64],
        #[case] expected: &[i32],
    ) {
        let elements = chunk_dims.iter().product::<u64>();
        let payload: Vec<_> = (11..11 + i32::try_from(elements).unwrap())
            .flat_map(i32::to_le_bytes)
            .collect();
        let index = ParsedChunkIndex::new(
            vec![ChunkInfo {
                offsets: offsets.to_vec(),
                chunk_size: StoredChunkSize::v4(elements * 4),
                filter_mask: 0,
                address: StoredAddress::new(0x1000),
            }],
            chunk_dims,
            ds_dims,
        )
        .unwrap();
        let mut whole = WholeAssembly::new(
            &index,
            chunk_dims,
            ds_dims,
            convert::nz(4),
            FillPattern::ZERO,
        )
        .unwrap();
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: u8::try_from(ds_dims.len()).unwrap(),
            dimensions: ds_dims.to_vec(),
            max_dimensions: None,
        };
        let datatype = crate::type_builders::make_i32_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::rows(spec, 0, ds_dims[0]).unwrap();
        let mut selected = SelectedAssembly::new(index.select(&region).unwrap()).unwrap();
        whole.place(&payload).unwrap();
        selected.place(&payload).unwrap();
        let expected: Vec<_> = expected
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(whole.finish().unwrap(), expected);
        assert_eq!(selected.finish().unwrap(), expected);
    }

    #[rstest]
    #[case::rank_one(&[3], &[5])]
    #[case::rank_two_rows(&[2, 3], &[3, 5])]
    #[case::rank_two_band(&[2, 5], &[3, 5])]
    #[case::rank_two_inner_overhang(&[2, 7], &[3, 5])]
    #[case::rank_three(&[2, 3, 2], &[3, 5, 3])]
    #[case::rank_three_band(&[2, 5, 3], &[3, 5, 3])]
    #[case::rank_three_trailing_band(&[2, 3, 3], &[3, 5, 3])]
    #[case::rank_four(&[2, 2, 2, 2], &[3, 3, 3, 3])]
    fn dense_assembly_writes_every_clipped_element(
        #[case] chunk_dims: &[u64],
        #[case] ds_dims: &[u64],
    ) {
        let index = assembly_index(chunk_dims, ds_dims);
        let mut assembly = WholeAssembly::new(
            &index,
            chunk_dims,
            ds_dims,
            convert::nz(4),
            FillPattern::UNKNOWN,
        )
        .unwrap();
        let AssemblyBuffer::Dense { bytes, length } = &assembly.output else {
            panic!("expected dense assembly");
        };
        assert_eq!(bytes.len(), 0);
        assert_eq!(
            *length,
            usize::try_from(ds_dims.iter().product::<u64>() * 4).unwrap()
        );
        for chunk in index.descriptors() {
            assembly
                .place(&assembly_payload(chunk, chunk_dims, ds_dims))
                .unwrap();
        }
        let expected: Vec<_> = (0..i32::try_from(ds_dims.iter().product::<u64>()).unwrap())
            .flat_map(i32::to_le_bytes)
            .collect();
        assert_eq!(assembly.finish().unwrap(), expected);
    }

    #[rstest]
    fn dense_assembly_rejects_incomplete_traversal(#[values(0, 1)] completed: usize) {
        let index = assembly_index(&[2], &[4]);
        let mut assembly =
            WholeAssembly::new(&index, &[2], &[4], convert::nz(4), FillPattern::UNKNOWN).unwrap();
        for chunk in index.descriptors().iter().take(completed) {
            assembly
                .place(&assembly_payload(chunk, &[2], &[4]))
                .unwrap();
        }
        assert_eq!(
            assembly.finish().unwrap_err(),
            FormatError::Internal("output assembly has chunks remaining".into())
        );
    }

    #[rstest]
    #[case::rank_one(&[2], &[4])]
    #[case::rank_two_rows(&[2, 3], &[3, 5])]
    #[case::rank_two_band(&[2, 5], &[3, 5])]
    #[case::rank_three(&[2, 3, 2], &[3, 5, 3])]
    fn dense_assembly_rejects_wrong_payload_lengths(
        #[case] chunk_dims: &[u64],
        #[case] ds_dims: &[u64],
        #[values(false, true)] oversized: bool,
    ) {
        let index = assembly_index(chunk_dims, ds_dims);
        let mut assembly = WholeAssembly::new(
            &index,
            chunk_dims,
            ds_dims,
            convert::nz(4),
            FillPattern::UNKNOWN,
        )
        .unwrap();
        let expected = usize::try_from(chunk_dims.iter().product::<u64>()).unwrap() * 4;
        assembly.place(&vec![1; expected]).unwrap();
        let length = if oversized {
            expected + 4
        } else {
            expected - 4
        };
        assert_eq!(
            assembly.place(&vec![2; length]).unwrap_err(),
            FormatError::DataSizeMismatch {
                expected,
                actual: length
            }
        );
        assert_eq!(
            assembly.finish().unwrap_err(),
            FormatError::Internal("output assembly has chunks remaining".into())
        );
    }

    #[test]
    fn dense_assembly_abandons_failed_placement() {
        let mut bytes = Vec::<u8>::with_capacity(7);
        let mut output = ScatterDestination::Spare(bytes.spare_capacity_mut());
        copy_scatter_run(&[1; 8], &mut output, 0, 0, 4).unwrap();
        assert_eq!(
            copy_scatter_run(&[1; 8], &mut output, 4, 4, 4).unwrap_err(),
            FormatError::Internal("scatter exceeds output extent".into())
        );
        assert_eq!(bytes.len(), 0);
    }

    #[rstest]
    #[case::short_source(6, 9, FormatError::Internal("scatter exceeds decoded chunk extent".into()))]
    #[case::short_destination(7, 8, FormatError::Internal("scatter exceeds output extent".into()))]
    fn dense_assembly_rows_reject_short_spans_before_writing(
        #[case] source_len: usize,
        #[case] destination_len: usize,
        #[case] expected: FormatError,
        #[values(false, true)] spare: bool,
    ) {
        let rows = ScatterRows::new(4, 5, 3, 2, 1).unwrap();
        let source = vec![1; source_len];
        if spare {
            let mut destination = Vec::<u8>::with_capacity(destination_len);
            let mut output =
                ScatterDestination::Spare(&mut destination.spare_capacity_mut()[..destination_len]);
            assert_eq!(output.copy_rows(&source, &rows).unwrap_err(), expected);
            assert_eq!(destination.len(), 0);
        } else {
            let mut destination = vec![7; destination_len];
            let mut output = ScatterDestination::Initialized(&mut destination);
            assert_eq!(output.copy_rows(&source, &rows).unwrap_err(), expected);
            assert_eq!(destination, vec![7; destination_len]);
        }
    }

    fn shuffle_pipeline() -> FilterPipeline {
        FilterPipeline {
            version: FILTER_PIPELINE_VERSION,
            filters: vec![FilterDescription {
                filter_id: crate::filter_pipeline::FILTER_SHUFFLE,
                name: None,
                flags: 0,
                client_data: vec![4],
            }],
        }
    }

    #[test]
    fn chunk_bytes_within_limit_are_accepted() {
        // A realistic chunk: 256 KiB of f64 elements.
        assert_eq!(
            bounded_chunk_len(&[128, 256], convert::nz(8)).unwrap(),
            262_144
        );
        // A single chunk right at the 4 GiB limit is still representable.
        assert_eq!(
            bounded_chunk_len(&[u64::from(u32::MAX)], convert::nz(1)).unwrap(),
            u32::MAX as usize
        );
    }

    #[rstest]
    #[case::large_string(&[10], convert::nz(0xAAAA_AAAA))]
    #[case::overflowing_product(&[u64::MAX, u64::MAX], convert::nz(usize::MAX))]
    fn chunk_bytes_over_limit_are_rejected(
        #[case] dims: &[u64],
        #[case] element_size: NonZeroUsize,
    ) {
        assert_eq!(
            bounded_chunk_len(dims, element_size).unwrap_err(),
            FormatError::InvalidChunkGeometry(
                "chunk logical byte size exceeds the reader's single-chunk allocation limit",
            )
        );
    }

    #[test]
    fn a_hole_in_the_chunk_grid_reads_as_fill() {
        // The first and last of four chunk slots are allocated, leaving an interior hole.
        let elem = 8usize;
        let chunk_elems = 2usize;
        let mut file_data = vec![0u8; 0x10000];

        let present: [(u64, [f64; 2]); 2] = [(0, [1.0, 2.0]), (6, [7.0, 8.0])];
        let mut chunk_infos = Vec::new();
        let mut data_offset = 0x2000usize;
        for (start_elem, values) in present {
            for (i, v) in values.iter().enumerate() {
                let at = data_offset + i * elem;
                file_data[at..at + 8].copy_from_slice(&v.to_le_bytes());
            }
            chunk_infos.push(ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1((chunk_elems * elem) as u32),
                filter_mask: 0,
                offsets: vec![start_elem, 0],
                address: StoredAddress::new(data_offset as u64),
            });
            data_offset += chunk_elems * elem;
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, 2, WIDTHS);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![chunk_elems as u64, elem as u64],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(btree_addr as u64)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![8],
            max_dimensions: None,
        };

        let fill = 5.5f64;
        let fill_bytes = fill.to_le_bytes();
        let out = read_chunked_data_from_source(
            &BytesSource::new(&file_data),
            RawReadSpec::parse(
                &layout,
                &dataspace,
                &make_f64_type(),
                None,
                FillPattern::new(FillTime::IfSet, Some(&fill_bytes), convert::nz(8)),
            )
            .unwrap(),
            FORMAT_WIDTHS,
        )
        .unwrap();

        let decoded: Vec<f64> = out
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        assert_eq!(
            decoded,
            vec![1.0, 2.0, fill, fill, fill, fill, 7.0, 8.0],
            "the interior hole must read as the fill value"
        );
    }

    /// All three whole-dataset readers materialize an unallocated layout the
    /// same way, which is what [`RawReadSpec::unallocated_buffer`] promises them.
    ///
    /// Exercised directly rather than through `Dataset::read_raw`, because the
    /// buffered entry point short-circuits `num_elements == 0` in
    /// `read_raw_data_full_from_source` before it dispatches on the layout at
    /// all — so one of these three call sites is unreachable from there, and a
    /// test that went in by the front door would leave it untested while looking
    /// like coverage.
    #[test]
    fn every_whole_dataset_reader_materializes_an_unallocated_dataset() {
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 8],
            index: ChunkIndexLayout::FixedArray { address: None },
        };
        let datatype = Datatype::FixedPoint(
            FixedPointType::new(8, byte_order::DatatypeByteOrder::LittleEndian, true, 0, 64)
                .unwrap(),
        );
        let simple = |dims: Vec<u64>| Dataspace {
            space_type: DataspaceType::Simple,
            rank: dims.len() as u8,
            dimensions: dims,
            max_dimensions: None,
        };
        let file_data: Vec<u8> = Vec::new();
        let seven = 7i64.to_le_bytes();

        // Every combination of "does it own elements" and "does it have a fill
        // value", through all three readers.
        for (dims, fill, expected) in [
            (vec![0u64], FillPattern::ZERO, Vec::new()),
            (
                vec![0],
                FillPattern::new(FillTime::IfSet, Some(&seven), convert::nz(8)),
                Vec::new(),
            ),
            (vec![3], FillPattern::ZERO, vec![0u8; 24]),
            (
                vec![3],
                FillPattern::new(FillTime::IfSet, Some(&seven), convert::nz(8)),
                seven.repeat(3),
            ),
        ] {
            let ds = simple(dims.clone());
            let spec = RawReadSpec::parse(&layout, &ds, &datatype, None, fill).unwrap();
            let from_source =
                read_chunked_data_from_source(&BytesSource::new(&file_data), spec, FORMAT_WIDTHS)
                    .unwrap();
            let cached_from_source = read_chunked_data_cached_from_source(
                &BytesSource::new(&file_data),
                spec,
                FORMAT_WIDTHS,
                &ChunkCache::new(),
            )
            .unwrap();
            let cached =
                read_chunked_data_cached(&file_data, spec, FORMAT_WIDTHS, &ChunkCache::new())
                    .unwrap();
            assert_eq!(from_source, expected, "dims {dims:?}");
            assert_eq!(cached_from_source, expected, "dims {dims:?}");
            assert_eq!(cached, expected, "dims {dims:?}");
        }
    }

    /// A type 1 leaf node indexing `chunks`, whose keys carry `ndims` offsets:
    /// the dataset's rank plus the trailing within-element offset.
    fn build_chunk_btree_leaf(chunks: &[ChunkInfo], ndims: usize, widths: Widths) -> Vec<u8> {
        let mut keys: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let StoredChunkSize::BTreeV1(chunk_size) = chunk.chunk_size else {
                    panic!("B-tree fixture chunk sizes must use the B-tree v1 width");
                };
                let mut offsets = chunk.offsets.clone();
                offsets.resize(ndims, 0);
                btree_v1::chunk_key(chunk_size, chunk.filter_mask, &offsets)
            })
            .collect();
        // The key past the last entry bounds the node, so its offsets are
        // the largest a dataset can hold.
        keys.push(btree_v1::chunk_key(0, 0, &vec![u64::MAX; ndims]));

        let children: Vec<u64> = chunks.iter().map(|chunk| chunk.address.get()).collect();
        btree_v1::node(btree_v1::NodeType::CHUNK, 0, &keys, &children, widths)
    }

    // --- ChunkInfo collection tests ---

    fn btree_v2_chunk_record(
        tree_type: u8,
        address: u64,
        scaled_offsets: &[u64],
        stored: Option<(u64, u32)>,
        fixed_chunk_size: u64,
        widths: Widths,
    ) -> Vec<u8> {
        let context = BTreeV2ChunkRecordContext::new(
            tree_type,
            OffsetWidth::try_from(u8::try_from(widths.offset).unwrap()).unwrap(),
            scaled_offsets.len(),
            NonZeroU64::new(fixed_chunk_size).unwrap(),
            LayoutVersion::Four,
        )
        .unwrap();
        let mut record = Vec::new();
        test_bytes::push_address(&mut record, Some(address), widths.offset);
        if let Some((stored_size, filter_mask)) = stored {
            let size_width = context.record_size() - widths.offset - 4 - 8 * scaled_offsets.len();
            test_bytes::push_uint(&mut record, stored_size, size_width);
            record.extend_from_slice(&filter_mask.to_le_bytes());
        }
        for &scaled in scaled_offsets {
            test_bytes::push_uint(&mut record, scaled, 8);
        }
        assert_eq!(record.len(), context.record_size());
        record
    }

    fn btree_v2_dataspace(dimensions: &[u64]) -> Dataspace {
        Dataspace {
            space_type: DataspaceType::Simple,
            rank: u8::try_from(dimensions.len()).unwrap(),
            dimensions: dimensions.to_vec(),
            max_dimensions: Some(vec![MaxExtent::Unlimited; dimensions.len()]),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn btree_v2_type_10_enumerates_sparse_multi_level_tree_from_bounded_source() {
        let widths = Widths::FOUR;
        let tree_type = hdf5_pure_format::__private::BTREE_V2_CHUNK;
        let chunk_dims = [2, 2];
        let fixed_chunk_size = 16;
        // A 512-byte type-10 leaf with 4-byte offsets and rank 2 holds at most 25 records, so 27
        // records require an internal node. Keep 13 records on either side of its promoted record.
        let records = (0..27u64)
            .map(|index| {
                btree_v2_chunk_record(
                    tree_type,
                    0x1000 + index * 0x20,
                    &[index / 8, index % 8],
                    None,
                    fixed_chunk_size,
                    widths,
                )
            })
            .collect::<Vec<_>>();
        let (leaf0_at, leaf1_at, root_at) = (0x100usize, 0x300usize, 0x500usize);
        let leaf0 = btree_v2::leaf(tree_type, &records[..13]);
        let leaf1 = btree_v2::leaf(tree_type, &records[14..]);
        let root = btree_v2::internal(
            tree_type,
            &records[13..14],
            &[
                btree_v2::Child {
                    address: u64::try_from(leaf0_at).unwrap(),
                    records: 13,
                    records_width: 1,
                    subtree: None,
                },
                btree_v2::Child {
                    address: u64::try_from(leaf1_at).unwrap(),
                    records: 13,
                    records_width: 1,
                    subtree: None,
                },
            ],
            widths,
        );
        let header = btree_v2::Header::new(
            tree_type,
            u16::try_from(records[0].len()).unwrap(),
            u64::try_from(root_at).unwrap(),
            1,
        )
        .depth(1)
        .total_records(27)
        .build(widths);
        let mut file = vec![0u8; 0x1400];
        file[..header.len()].copy_from_slice(&header);
        file[leaf0_at..leaf0_at + leaf0.len()].copy_from_slice(&leaf0);
        file[leaf1_at..leaf1_at + leaf1.len()].copy_from_slice(&leaf1);
        file[root_at..root_at + root.len()].copy_from_slice(&root);

        let index = ChunkIndexLayout::BTreeV2 {
            address: Some(StoredAddress::new(0)),
        };
        let dimensions = [16, 16];
        let dataspace = btree_v2_dataspace(&dimensions);
        let chunk_dimensions = [chunk_dims[0], chunk_dims[1], 4];
        let offset_size = u8::try_from(widths.offset).unwrap();
        let length_size = u8::try_from(widths.length).unwrap();
        let memory = collect_chunks_for_layout_from_source(
            &BytesSource::new(&file),
            index,
            &chunk_dimensions,
            &dataspace,
            4,
            FormatWidths::from_sizes(offset_size, length_size).unwrap(),
        )
        .unwrap();
        let bounded = collect_chunks_for_layout_from_source(
            &ReadSeekSource::new(Cursor::new(file)).unwrap(),
            index,
            &chunk_dimensions,
            &dataspace,
            4,
            FormatWidths::from_sizes(offset_size, length_size).unwrap(),
        )
        .unwrap();

        assert_eq!(bounded, memory);
        assert_eq!(memory.len(), 27);
        assert_eq!(memory[0].offsets, vec![0, 0]);
        assert_eq!(memory[13].offsets, vec![2, 10]);
        assert_eq!(memory[26].offsets, vec![6, 4]);
        assert!(
            memory
                .iter()
                .all(|chunk| chunk.chunk_size == StoredChunkSize::v4(fixed_chunk_size))
        );
        assert!(memory.iter().all(|chunk| chunk.filter_mask == 0));
    }

    #[test]
    fn btree_v2_storage_spans_include_exact_multi_level_index_allocations() {
        let widths = Widths::FOUR;
        let tree_type = hdf5_pure_format::__private::BTREE_V2_CHUNK;
        let chunk_dims = [2, 2];
        let fixed_chunk_size = 16;
        let records = (0..27u64)
            .map(|index| {
                btree_v2_chunk_record(
                    tree_type,
                    0x1000 + index * 0x20,
                    &[index / 8, index % 8],
                    None,
                    fixed_chunk_size,
                    widths,
                )
            })
            .collect::<Vec<_>>();
        let (leaf0_at, leaf1_at, root_at) = (0x100usize, 0x300usize, 0x500usize);
        let leaf0 = btree_v2::leaf(tree_type, &records[..13]);
        let leaf1 = btree_v2::leaf(tree_type, &records[14..]);
        let root = btree_v2::internal(
            tree_type,
            &records[13..14],
            &[
                btree_v2::Child {
                    address: u64::try_from(leaf0_at).unwrap(),
                    records: 13,
                    records_width: 1,
                    subtree: None,
                },
                btree_v2::Child {
                    address: u64::try_from(leaf1_at).unwrap(),
                    records: 13,
                    records_width: 1,
                    subtree: None,
                },
            ],
            widths,
        );
        let header = btree_v2::Header::new(
            tree_type,
            u16::try_from(records[0].len()).unwrap(),
            u64::try_from(root_at).unwrap(),
            1,
        )
        .depth(1)
        .total_records(27)
        .build(widths);
        let mut file = vec![0u8; 0x1400];
        file[..header.len()].copy_from_slice(&header);
        file[leaf0_at..leaf0_at + leaf0.len()].copy_from_slice(&leaf0);
        file[leaf1_at..leaf1_at + leaf1.len()].copy_from_slice(&leaf1);
        file[root_at..root_at + root.len()].copy_from_slice(&root);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![chunk_dims[0], chunk_dims[1], 4],
            index: ChunkIndexLayout::BTreeV2 {
                address: Some(StoredAddress::new(0)),
            },
        };

        let spans = collect_chunked_storage_spans(
            &BytesSource::new(&file),
            &layout,
            &btree_v2_dataspace(&[16, 16]),
            FormatWidths::from_sizes(
                u8::try_from(widths.offset).unwrap(),
                u8::try_from(widths.length).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            spans.index,
            vec![
                (0, u64::try_from(header.len()).unwrap()),
                (u64::try_from(leaf0_at).unwrap(), 512),
                (u64::try_from(leaf1_at).unwrap(), 512),
                (u64::try_from(root_at).unwrap(), 512),
            ]
        );
        assert_eq!(spans.data.len(), 27);
        assert_eq!(spans.data[0], (0x1000, fixed_chunk_size));
        assert_eq!(spans.data[26], (0x1340, fixed_chunk_size));
        assert!(spans.data.iter().all(|&(_, len)| len == fixed_chunk_size));
    }

    #[test]
    fn btree_v2_storage_spans_reject_a_truncated_node_as_a_whole() {
        let widths = Widths::FOUR;
        let tree_type = hdf5_pure_format::__private::BTREE_V2_CHUNK;
        let fixed_chunk_size = 16;
        let record =
            btree_v2_chunk_record(tree_type, 0x80, &[0, 0], None, fixed_chunk_size, widths);
        let record_size = u16::try_from(record.len()).unwrap();
        let root_at = 0x100usize;
        let leaf = btree_v2::leaf(tree_type, &[record]);
        let header =
            btree_v2::Header::new(tree_type, record_size, u64::try_from(root_at).unwrap(), 1)
                .build(widths);
        let mut file = vec![0u8; root_at + leaf.len()];
        file[..header.len()].copy_from_slice(&header);
        file[root_at..].copy_from_slice(&leaf);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 2, 4],
            index: ChunkIndexLayout::BTreeV2 {
                address: Some(StoredAddress::new(0)),
            },
        };

        let Err(err) = collect_chunked_storage_spans(
            &BytesSource::new(&file),
            &layout,
            &btree_v2_dataspace(&[2, 2]),
            FormatWidths::from_sizes(
                u8::try_from(widths.offset).unwrap(),
                u8::try_from(widths.length).unwrap(),
            )
            .unwrap(),
        ) else {
            panic!("expected a truncated version 2 B-tree node to fail");
        };
        assert_eq!(
            err,
            FormatError::UnexpectedSourceEof {
                expected: root_at.to_u64() + 512,
                available: file.len().to_u64(),
            }
        );
    }

    #[test]
    fn btree_v2_type_11_preserves_stored_sizes_and_filter_masks() {
        let widths = Widths::EIGHT;
        let tree_type = hdf5_pure_format::__private::BTREE_V2_FILTERED_CHUNK;
        let fixed_chunk_size = 16;
        let records = [
            btree_v2_chunk_record(
                tree_type,
                0x500,
                &[0, 0],
                Some((9, 1)),
                fixed_chunk_size,
                widths,
            ),
            btree_v2_chunk_record(
                tree_type,
                0x520,
                &[2, 1],
                Some((13, 4)),
                fixed_chunk_size,
                widths,
            ),
        ];
        let root_at = 0x100usize;
        let leaf = btree_v2::leaf(tree_type, &records);
        let header = btree_v2::Header::new(
            tree_type,
            u16::try_from(records[0].len()).unwrap(),
            u64::try_from(root_at).unwrap(),
            2,
        )
        .build(widths);
        let mut file = vec![0u8; 0x600];
        file[..header.len()].copy_from_slice(&header);
        file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);

        let chunks = collect_chunks_for_layout_from_source(
            &BytesSource::new(&file),
            ChunkIndexLayout::BTreeV2 {
                address: Some(StoredAddress::new(0)),
            },
            &[2, 2, 4],
            &btree_v2_dataspace(&[6, 4]),
            4,
            FORMAT_WIDTHS,
        )
        .unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].offsets, vec![0, 0]);
        assert_eq!(chunks[0].chunk_size, StoredChunkSize::v4(9));
        assert_eq!(chunks[0].filter_mask, 1);
        assert_eq!(chunks[1].offsets, vec![4, 2]);
        assert_eq!(chunks[1].chunk_size, StoredChunkSize::v4(13));
        assert_eq!(chunks[1].filter_mask, 4);
    }

    #[test]
    fn btree_v2_filtered_storage_spans_use_encoded_chunk_sizes() {
        let widths = Widths::EIGHT;
        let tree_type = hdf5_pure_format::__private::BTREE_V2_FILTERED_CHUNK;
        let fixed_chunk_size = 16;
        let records = [
            btree_v2_chunk_record(
                tree_type,
                0x500,
                &[0, 0],
                Some((9, 1)),
                fixed_chunk_size,
                widths,
            ),
            btree_v2_chunk_record(
                tree_type,
                0x520,
                &[2, 1],
                Some((13, 4)),
                fixed_chunk_size,
                widths,
            ),
        ];
        let root_at = 0x100usize;
        let leaf = btree_v2::leaf(tree_type, &records);
        let header = btree_v2::Header::new(
            tree_type,
            u16::try_from(records[0].len()).unwrap(),
            u64::try_from(root_at).unwrap(),
            2,
        )
        .build(widths);
        let mut file = vec![0u8; 0x600];
        file[..header.len()].copy_from_slice(&header);
        file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 2, 4],
            index: ChunkIndexLayout::BTreeV2 {
                address: Some(StoredAddress::new(0)),
            },
        };

        let spans = collect_chunked_storage_spans(
            &BytesSource::new(&file),
            &layout,
            &btree_v2_dataspace(&[6, 4]),
            FORMAT_WIDTHS,
        )
        .unwrap();

        assert_eq!(spans.data, vec![(0x500, 9), (0x520, 13)]);
        assert_eq!(
            spans.index,
            vec![
                (0, u64::try_from(header.len()).unwrap()),
                (u64::try_from(root_at).unwrap(), 512),
            ]
        );
    }

    #[test]
    fn btree_v2_chunk_index_rejects_wrong_client_type_and_record_size() {
        let widths = Widths::EIGHT;
        let dataspace = btree_v2_dataspace(&[4, 4]);
        let index = ChunkIndexLayout::BTreeV2 {
            address: Some(StoredAddress::new(0)),
        };

        let wrong_type = btree_v2::Header::new(
            hdf5_pure_format::__private::BTREE_V2_ATTRIBUTE_NAME,
            24,
            0x100,
            1,
        )
        .build(widths);
        let mut file = vec![0u8; 0x400];
        file[..wrong_type.len()].copy_from_slice(&wrong_type);
        assert_eq!(
            collect_chunks_for_layout_from_source(
                &BytesSource::new(&file),
                index,
                &[2, 2, 4],
                &dataspace,
                4,
                FORMAT_WIDTHS,
            )
            .unwrap_err(),
            FormatError::InvalidBTreeNodeType(hdf5_pure_format::__private::BTREE_V2_ATTRIBUTE_NAME)
        );

        let wrong_size =
            btree_v2::Header::new(hdf5_pure_format::__private::BTREE_V2_CHUNK, 23, 0x100, 1)
                .build(widths);
        file[..wrong_size.len()].copy_from_slice(&wrong_size);
        assert_eq!(
            collect_chunks_for_layout_from_source(
                &BytesSource::new(&file),
                index,
                &[2, 2, 4],
                &dataspace,
                4,
                FORMAT_WIDTHS,
            )
            .unwrap_err(),
            FormatError::DataSizeMismatch {
                expected: 24,
                actual: 23,
            }
        );
    }

    #[test]
    fn btree_v2_chunk_coordinates_must_not_overflow() {
        let widths = Widths::EIGHT;
        let tree_type = hdf5_pure_format::__private::BTREE_V2_CHUNK;
        let record = btree_v2_chunk_record(tree_type, 0x500, &[u64::MAX, 0], None, 16, widths);
        let root_at = 0x100usize;
        let leaf = btree_v2::leaf(tree_type, core::slice::from_ref(&record));
        let header = btree_v2::Header::new(
            tree_type,
            u16::try_from(record.len()).unwrap(),
            u64::try_from(root_at).unwrap(),
            1,
        )
        .build(widths);
        let mut file = vec![0u8; 0x600];
        file[..header.len()].copy_from_slice(&header);
        file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);

        let err = collect_chunks_for_layout_from_source(
            &BytesSource::new(&file),
            ChunkIndexLayout::BTreeV2 {
                address: Some(StoredAddress::new(0)),
            },
            &[2, 2, 4],
            &btree_v2_dataspace(&[4, 4]),
            4,
            FORMAT_WIDTHS,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::ChunkedReadError(
                "chunk coordinates resolve past the addressable dataspace".into()
            )
        );
    }

    #[test]
    fn btree_v2_chunk_address_must_fit_the_source() {
        let widths = Widths::EIGHT;
        let tree_type = hdf5_pure_format::__private::BTREE_V2_CHUNK;
        let record = btree_v2_chunk_record(tree_type, 0x5f8, &[0, 0], None, 16, widths);
        let root_at = 0x100usize;
        let leaf = btree_v2::leaf(tree_type, core::slice::from_ref(&record));
        let header = btree_v2::Header::new(
            tree_type,
            u16::try_from(record.len()).unwrap(),
            u64::try_from(root_at).unwrap(),
            1,
        )
        .build(widths);
        let mut file = vec![0u8; 0x600];
        file[..header.len()].copy_from_slice(&header);
        file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);

        assert_eq!(
            collect_chunks_for_layout_from_source(
                &BytesSource::new(&file),
                ChunkIndexLayout::BTreeV2 {
                    address: Some(StoredAddress::new(0)),
                },
                &[2, 2, 4],
                &btree_v2_dataspace(&[4, 4]),
                4,
                FORMAT_WIDTHS,
            )
            .unwrap_err(),
            FormatError::ChunkedReadError(
                "version 2 B-tree chunk at 1528 with size 16 extends past the source length 1536"
                    .into()
            )
        );
    }

    #[test]
    fn collect_two_chunks_from_leaf() {
        let ndims = 2; // rank+1 for 1D dataset
        let os: u8 = 8;

        let chunks = vec![
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(80),
                filter_mask: 0,
                offsets: vec![0, 0],
                address: StoredAddress::new(0x1000),
            },
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(80),
                filter_mask: 0,
                offsets: vec![10, 0],
                address: StoredAddress::new(0x2000),
            },
        ];

        let btree = build_chunk_btree_leaf(&chunks, ndims, WIDTHS);
        let mut file_data = vec![0u8; 0x3000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(
            &file_data,
            StoredAddress::new(0),
            ndims,
            OffsetWidth::try_from(os).unwrap(),
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].address, StoredAddress::new(0x1000));
        assert_eq!(result[0].offsets, vec![0, 0]);
        assert_eq!(result[0].chunk_size, StoredChunkSize::btree_v1(80));
        assert_eq!(result[1].address, StoredAddress::new(0x2000));
        assert_eq!(result[1].offsets, vec![10, 0]);
    }

    /// A type 1 internal node above `child_addrs`, whose keys are placeholders:
    /// the node-span walk this exercises reads only the children.
    fn build_chunk_btree_internal(
        level: u8,
        child_addrs: &[u64],
        ndims: usize,
        widths: Widths,
    ) -> Vec<u8> {
        let mut keys: Vec<_> = child_addrs
            .iter()
            .map(|_| btree_v1::chunk_key(0, 0, &vec![0; ndims]))
            .collect();
        keys.push(btree_v1::chunk_key(0, 0, &vec![u64::MAX; ndims]));
        btree_v1::node(btree_v1::NodeType::CHUNK, level, &keys, child_addrs, widths)
    }

    #[test]
    fn btree_node_spans_single_leaf() {
        // A leaf's reclaim span is the whole node: its built byte length.
        let ndims = 2;
        let os: u8 = 8;
        let chunks = vec![
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(80),
                filter_mask: 0,
                offsets: vec![0, 0],
                address: StoredAddress::new(0x1000),
            },
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(80),
                filter_mask: 0,
                offsets: vec![10, 0],
                address: StoredAddress::new(0x2000),
            },
        ];
        let leaf = build_chunk_btree_leaf(&chunks, ndims, WIDTHS);
        let at = 0x40usize;
        let mut file_data = vec![0u8; 0x3000];
        file_data[at..at + leaf.len()].copy_from_slice(&leaf);

        let spans = collect_chunk_btree_node_spans(
            &BytesSource::new(&file_data),
            StoredAddress::new(at as u64),
            ndims,
            OffsetWidth::try_from(os).unwrap(),
        )
        .unwrap();
        assert_eq!(spans, vec![(at as u64, leaf.len() as u64)]);
    }

    #[test]
    fn btree_node_spans_two_level_recurses() {
        // A 2-level tree yields one span per node: the internal root and both
        // leaves, each sized to its built byte length, pairwise disjoint.
        let ndims = 2;
        let os: u8 = 8;
        let leaf_chunks = vec![ChunkInfo {
            chunk_size: StoredChunkSize::btree_v1(40),
            filter_mask: 0,
            offsets: vec![0, 0],
            address: StoredAddress::new(0x100),
        }];
        let leaf0 = build_chunk_btree_leaf(&leaf_chunks, ndims, WIDTHS);
        let leaf1 = build_chunk_btree_leaf(&leaf_chunks, ndims, WIDTHS);
        let (l0, l1, root) = (0x1000usize, 0x2000usize, 0x3000usize);
        let internal = build_chunk_btree_internal(1, &[l0 as u64, l1 as u64], ndims, WIDTHS);

        let mut file = vec![0u8; 0x4000];
        file[l0..l0 + leaf0.len()].copy_from_slice(&leaf0);
        file[l1..l1 + leaf1.len()].copy_from_slice(&leaf1);
        file[root..root + internal.len()].copy_from_slice(&internal);

        let mut spans = collect_chunk_btree_node_spans(
            &BytesSource::new(&file),
            StoredAddress::new(root as u64),
            ndims,
            OffsetWidth::try_from(os).unwrap(),
        )
        .unwrap();
        spans.sort_unstable();
        assert_eq!(
            spans,
            vec![
                (l0 as u64, leaf0.len() as u64),
                (l1 as u64, leaf1.len() as u64),
                (root as u64, internal.len() as u64),
            ]
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::borrowed(0, false)]
    #[case::borrowed_userblock(512, false)]
    #[case::cached(0, true)]
    #[case::cached_userblock(512, true)]
    fn btree_v1_traversals_preserve_multilevel_nodes(#[case] base: usize, #[case] cached: bool) {
        let chunks = vec![
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(40),
                filter_mask: 0,
                offsets: vec![0, 0],
                address: StoredAddress::new(0x1000),
            },
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(24),
                filter_mask: 2,
                offsets: vec![10, 0],
                address: StoredAddress::new(0x2000),
            },
        ];
        let nodes = [
            (
                0x100,
                build_chunk_btree_internal(2, &[0x200, 0x300], 2, WIDTHS),
            ),
            (0x200, build_chunk_btree_internal(1, &[0x400], 2, WIDTHS)),
            (0x400, build_chunk_btree_leaf(&chunks[..1], 2, WIDTHS)),
            (0x300, build_chunk_btree_internal(1, &[0x500], 2, WIDTHS)),
            (0x500, build_chunk_btree_leaf(&chunks[1..], 2, WIDTHS)),
        ];
        let mut image = test_util::image::Image::new();
        for (address, bytes) in &nodes {
            image.place(base + address, bytes);
        }
        let source: Box<dyn Source> = if cached {
            Box::new(MetadataCachingSource::new(
                ReadSeekSource::new(Cursor::new(image.build())).unwrap(),
                MetadataCacheConfig::new(4096),
            ))
        } else {
            Box::new(BytesSource::new(image.build()))
        };
        let framed = BaseOffsetSource {
            inner: source.as_ref(),
            base: BaseAddress::new(base.to_u64()),
        };
        let metadata = SourceMetadata(&framed);
        let parent = BTreeV1ChunkNodeBytes::read_from_source(&metadata, 0x100, 8, 2).unwrap();
        let node = parent.node();
        let mut children = node.children();
        assert_eq!(children.next().unwrap().unwrap(), StoredAddress::new(0x200));
        assert_eq!(
            collect_chunk_info_from_source(
                &framed,
                StoredAddress::new(0x200),
                2,
                OffsetWidth::Eight
            )
            .unwrap(),
            chunks[..1]
        );
        assert_eq!(children.next().unwrap().unwrap(), StoredAddress::new(0x300));
        assert!(children.next().is_none());
        assert_eq!(node.stored_len(), nodes[0].1.len());
        assert_eq!(
            collect_chunk_info_from_source(
                &framed,
                StoredAddress::new(0x100),
                2,
                OffsetWidth::Eight
            )
            .unwrap(),
            chunks
        );
        assert_eq!(
            collect_chunk_btree_node_spans(
                &framed,
                StoredAddress::new(0x100),
                2,
                OffsetWidth::Eight
            )
            .unwrap(),
            nodes
                .iter()
                .map(|(address, bytes)| (address.to_u64(), bytes.len().to_u64()))
                .collect::<Vec<_>>()
        );
        let stats = source.metadata_cache_stats();
        if cached {
            let stats = stats.unwrap();
            assert_eq!(
                (stats.misses(), stats.entries(), stats.evictions()),
                (10, 10, 0)
            );
            assert!(stats.hits() > 0);
            let first = framed.read_metadata(0x108, nodes[0].1.len() - 8).unwrap();
            let second = framed.read_metadata(0x108, nodes[0].1.len() - 8).unwrap();
            assert_eq!(first.offset(), base.to_u64() + 0x108);
            assert_eq!(first.as_ptr(), second.as_ptr());
        } else {
            assert_eq!(stats, None);
        }
    }

    #[rstest]
    #[case::invalid_signature(false)]
    #[case::truncated_body(true)]
    fn btree_v1_traversals_reject_a_failing_descendant(#[case] truncated: bool) {
        let root = build_chunk_btree_internal(1, &[0x100, 0x200], 2, WIDTHS);
        let leaf = build_chunk_btree_leaf(&[], 2, WIDTHS);
        let mut image = test_util::image::Image::new();
        image.place(0, &root);
        image.place(0x100, &leaf);
        let mut descendant = leaf.clone();
        let expected = if truncated {
            descendant.pop();
            FormatError::UnexpectedSourceEof {
                expected: 0x200 + leaf.len().to_u64(),
                available: 0x200 + descendant.len().to_u64(),
            }
        } else {
            descendant[0] = b'X';
            FormatError::InvalidBTreeSignature
        };
        image.place(0x200, &descendant);
        let source = BytesSource::new(image.build());
        assert_eq!(
            collect_chunk_info_from_source(&source, StoredAddress::new(0), 2, OffsetWidth::Eight)
                .unwrap_err(),
            expected
        );
        assert_eq!(
            collect_chunk_btree_node_spans(&source, StoredAddress::new(0), 2, OffsetWidth::Eight)
                .unwrap_err(),
            expected
        );
    }

    #[test]
    fn btree_v1_source_traversals_reject_excessive_depth() {
        let bytes = build_chunk_btree_internal(1, &[0], 2, WIDTHS);
        let source = BytesSource::new(bytes);
        let expected = FormatError::ChunkedReadError("chunk B-tree nested too deeply".into());
        assert_eq!(
            collect_chunk_info_from_source(&source, StoredAddress::new(0), 2, OffsetWidth::Eight)
                .unwrap_err(),
            expected
        );
        assert_eq!(
            collect_chunk_btree_node_spans(&source, StoredAddress::new(0), 2, OffsetWidth::Eight)
                .unwrap_err(),
            expected
        );
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[rstest]
    #[case::small(
        2048,
        "v1_owner_cold_small",
        "v1_owner_warm_small",
        "v1_owner_copy_small"
    )]
    #[case::large(
        8192,
        "v1_owner_cold_large",
        "v1_owner_warm_large",
        "v1_owner_copy_large"
    )]
    fn btree_v1_cached_acquisition_allocates_no_second_payload(
        #[case] entries: usize,
        #[case] cold_region: &str,
        #[case] warm_region: &str,
        #[case] control_region: &str,
    ) {
        struct NodeSource {
            prefix: Vec<u8>,
            body: core::cell::Cell<Option<Vec<u8>>>,
            len: u64,
        }

        impl Source for NodeSource {
            fn len(&self) -> u64 {
                self.len
            }

            fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FormatError> {
                Err(FormatError::Source("unexpected payload read".into()))
            }

            fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
                let bytes = match offset {
                    0 => self.prefix.clone(),
                    8 => self.body.take().unwrap(),
                    _ => return Err(FormatError::Source("unexpected metadata offset".into())),
                };
                if bytes.len() != len {
                    return Err(FormatError::DataSizeMismatch {
                        expected: len,
                        actual: bytes.len(),
                    });
                }
                Ok(bytes)
            }
        }

        let mut bytes = build_chunk_btree_internal(1, &vec![0x1000; entries], 2, WIDTHS);
        let node_len = bytes.len();
        let body = bytes.split_off(8);
        let pointer = body.as_ptr();
        let body_len = body.len();
        let (copy, control) = test_util::allocation::measure(control_region, || body.clone());
        assert_eq!(copy, body);
        assert!(control.bytes >= body_len.to_u64(), "{control}");
        assert!(control.blocks > 0, "{control}");
        drop(copy);
        let source = MetadataCachingSource::new(
            NodeSource {
                prefix: bytes,
                body: core::cell::Cell::new(Some(body)),
                len: node_len.to_u64(),
            },
            MetadataCacheConfig::new(node_len * 2).with_max_entry_bytes(body_len),
        );
        let metadata = SourceMetadata(&source);
        let (cold, admission) = test_util::allocation::measure(cold_region, || {
            BTreeV1ChunkNodeBytes::read_from_source(&metadata, 0, 8, 2).unwrap()
        });
        assert_eq!(source.read_metadata(8, body_len).unwrap().as_ptr(), pointer);
        let (warm, measured) = test_util::allocation::measure(warm_region, || {
            BTreeV1ChunkNodeBytes::read_from_source(&metadata, 0, 8, 2).unwrap()
        });
        assert_eq!(source.read_metadata(8, body_len).unwrap().as_ptr(), pointer);
        for node in [cold.node(), warm.node()] {
            assert_eq!(
                (node.node_level(), node.entries_used(), node.stored_len()),
                (1, u16::try_from(entries).unwrap(), node_len)
            );
            assert_eq!(
                node.children().map(Result::unwrap).collect::<Vec<_>>(),
                vec![StoredAddress::new(0x1000); entries]
            );
        }
        assert!(admission.bytes < 16_384, "{admission}");
        assert!(admission.peak_bytes < 16_384, "{admission}");
        assert!(measured.bytes < 16_384, "{measured}");
        assert!(measured.peak_bytes < 16_384, "{measured}");
        let stats = source.metadata_cache_stats().unwrap();
        assert_eq!(
            (stats.misses(), stats.hits(), stats.entries(), stats.bytes()),
            (2, 4, 2, node_len)
        );
    }

    #[test]
    fn collect_three_chunks() {
        let ndims = 2;
        let os: u8 = 8;

        let chunks = vec![
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(40),
                filter_mask: 0,
                offsets: vec![0, 0],
                address: StoredAddress::new(0x100),
            },
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(40),
                filter_mask: 0,
                offsets: vec![5, 0],
                address: StoredAddress::new(0x200),
            },
            ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(40),
                filter_mask: 0,
                offsets: vec![10, 0],
                address: StoredAddress::new(0x300),
            },
        ];

        let btree = build_chunk_btree_leaf(&chunks, ndims, WIDTHS);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(
            &file_data,
            StoredAddress::new(0),
            ndims,
            OffsetWidth::try_from(os).unwrap(),
        )
        .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].address, StoredAddress::new(0x100));
        assert_eq!(result[1].address, StoredAddress::new(0x200));
        assert_eq!(result[2].address, StoredAddress::new(0x300));
    }

    #[test]
    fn collect_empty_btree() {
        let ndims = 2;
        let os: u8 = 8;
        let btree = build_chunk_btree_leaf(&[], ndims, WIDTHS);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(
            &file_data,
            StoredAddress::new(0),
            ndims,
            OffsetWidth::try_from(os).unwrap(),
        )
        .unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn collect_chunk_info_rejects_cyclic_btree() {
        // An internal node whose only child points back at itself. Without a
        // depth guard this recurses forever and overflows the stack (an
        // uncatchable process abort); the guard must turn it into an error.
        let ndims = 2;
        let os: u8 = 8;
        let node = build_chunk_btree_internal(1, &[0u64], ndims, WIDTHS);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..node.len()].copy_from_slice(&node);

        let err = collect_chunk_info(
            &file_data,
            StoredAddress::new(0),
            ndims,
            OffsetWidth::try_from(os).unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::ChunkedReadError("chunk B-tree nested too deeply".into())
        );
    }

    // --- Chunked read tests (synthetic) ---

    use crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::byte_order;
    use crate::datatype::layout::FloatingPointLayout;
    use crate::datatype::{Datatype, byte_order::DatatypeByteOrder};
    use crate::source::BytesSource;

    fn make_f64_type() -> Datatype {
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64,
            )
            .unwrap(),
        )
    }

    fn make_f32_type() -> Datatype {
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                4,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY32,
            )
            .unwrap(),
        )
    }

    /// Build a synthetic file with a B-tree and chunk data for a 1D uncompressed dataset.
    fn build_1d_chunked_file(
        values: &[f64],
        chunk_size_elems: usize,
    ) -> (Vec<u8>, DataLayout, Dataspace) {
        let elem_size = 8usize;
        let ndims = 2; // rank(1) + 1
        let total = values.len();

        // Place chunk data starting at offset 0x2000
        let mut file_data = vec![0u8; 0x10000];
        let mut chunk_infos = Vec::new();
        let mut data_offset = 0x2000usize;

        let mut start = 0;
        while start < total {
            let end = (start + chunk_size_elems).min(total);
            let chunk_bytes = chunk_size_elems * elem_size; // full chunk allocation

            // Write chunk data (full chunk size, padding with zeros)
            for (k, value) in values[start..end].iter().enumerate() {
                let byte_offset = data_offset + k * elem_size;
                file_data[byte_offset..byte_offset + 8].copy_from_slice(&value.to_le_bytes());
            }

            chunk_infos.push(ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(chunk_bytes as u32),
                filter_mask: 0,
                offsets: vec![start as u64, 0],
                address: StoredAddress::new(data_offset as u64),
            });

            data_offset += chunk_bytes;
            start += chunk_size_elems;
        }

        // Build B-tree at offset 0x100
        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, WIDTHS);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![chunk_size_elems as u64, elem_size as u64],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(btree_addr as u64)),
            },
        };

        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![total as u64],
            max_dimensions: None,
        };

        (file_data, layout, dataspace)
    }

    #[test]
    fn read_1d_two_chunks_no_compression() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();

        let raw = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &ChunkCache::new(),
        )
        .unwrap();
        assert_eq!(raw.len(), 20 * 8);

        // Verify values
        for i in 0..20 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64);
        }
    }

    #[test]
    fn read_1d_three_chunks_partial_last() {
        // 25 elements, chunk size 10 => 3 chunks, last has only 5 valid
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();

        let raw = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &ChunkCache::new(),
        )
        .unwrap();
        assert_eq!(raw.len(), 25 * 8);

        for i in 0..25 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    /// Assert the streaming chunked reader matches the buffered one over both an
    /// in-memory source and a lazy `Read+Seek` source.
    #[cfg(feature = "std")]
    fn assert_chunked_streams_match(
        file_data: &[u8],
        layout: &DataLayout,
        dataspace: &Dataspace,
        datatype: &Datatype,
        pipeline: Option<&FilterPipeline>,
    ) -> Vec<u8> {
        use crate::source::ReadSeekSource;
        let spec =
            RawReadSpec::parse(layout, dataspace, datatype, pipeline, FillPattern::ZERO).unwrap();
        let buffered =
            read_chunked_data_cached(file_data, spec, FORMAT_WIDTHS, &ChunkCache::new()).unwrap();
        let from_mem =
            read_chunked_data_from_source(&BytesSource::new(file_data), spec, FORMAT_WIDTHS)
                .unwrap();
        let from_seek = read_chunked_data_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data.to_vec())).unwrap(),
            spec,
            FORMAT_WIDTHS,
        )
        .unwrap();
        assert_eq!(buffered, from_mem, "BytesSource mismatch");
        assert_eq!(buffered, from_seek, "ReadSeekSource mismatch");
        buffered
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_chunked_btree_v1_matches_buffered() {
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            assert_chunked_streams_match(&file_data, &layout, &dataspace, &make_f64_type(), None),
            raw
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::paged_fixed_array(3_000, None)]
    #[case::extensible_array_super_blocks(2_000, Some(MaxExtent::Unlimited))]
    #[case::paged_extensible_array(140_000, Some(MaxExtent::Unlimited))]
    fn streaming_chunk_arrays_match_buffered(
        #[case] elements: u64,
        #[case] max_extent: Option<MaxExtent>,
    ) {
        let values: Vec<f64> = (0..elements).map(|i| i as f64).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data_address = StoredAddress::new(0x1000);
        let options = crate::chunked_write::ChunkOptions {
            chunk_dims: Some(vec![1]),
            ..Default::default()
        };
        let maxshape = max_extent.map(|extent| vec![extent]);
        let built = crate::chunked_write::build_chunked_data_at_ext(
            &raw,
            &[elements],
            ChunkContext::basic(&[1], 8),
            &options,
            data_address,
            maxshape.as_deref(),
            FillPattern::ZERO,
        )
        .unwrap();
        let mut file_data = vec![0u8; 0x1000];
        file_data.extend_from_slice(&built.data_bytes);
        let layout = DataLayout::parse(&built.layout_message, 8, 8).unwrap();
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![elements],
            max_dimensions: maxshape,
        };
        assert_eq!(
            assert_chunked_streams_match(&file_data, &layout, &dataspace, &make_f64_type(), None),
            raw
        );
    }

    fn indexed_chunk(address: u64) -> ChunkRecord {
        ChunkRecord {
            address: StoredAddress::new(address),
            stored_size: 8,
            filter_mask: 0,
        }
    }

    fn array_index(
        kind: ChunkArrayKind,
        chunks: &[ChunkRecord],
        slot_of: &[u64],
        slots: u64,
        chunk_bytes: u64,
    ) -> (ChunkIndexLayout, Vec<u8>) {
        let slots = hdf5_pure_format::__private::IndexSlots::new(chunks, slot_of, slots).unwrap();
        let address = StoredAddress::new(0);
        match kind {
            ChunkArrayKind::FixedArray => (
                ChunkIndexLayout::FixedArray {
                    address: Some(address),
                },
                hdf5_pure_format::__private::build_fixed_array_at(
                    &slots,
                    chunk_bytes,
                    crate::width::OffsetWidth::Eight,
                    crate::width::LengthWidth::Eight,
                    false,
                    address,
                )
                .unwrap(),
            ),
            ChunkArrayKind::ExtensibleArray => (
                ChunkIndexLayout::ExtensibleArray {
                    creation: ExtensibleArrayCreationParameters::canonical(),
                    address: Some(address),
                },
                hdf5_pure_format::__private::build_extensible_array_at(
                    &slots,
                    chunk_bytes,
                    crate::width::OffsetWidth::Eight,
                    crate::width::LengthWidth::Eight,
                    false,
                    address,
                )
                .unwrap(),
            ),
        }
    }

    // The Extensible Array case uses slot 2, between the dataset's chunks at grid slots 1 and 4,
    // since its walk stops at the highest slot set.
    #[rstest]
    // Chunk [2] over shape [4] is two chunks, and a maximum of [8] numbers four
    // slots, and slot 3 lies past the dataset's four elements.
    #[case::fixed_array(
        ChunkArrayKind::FixedArray,
        &[2],
        &[4],
        &[MaxExtent::Fixed(8)],
        &[0, 1, 3],
        4,
        &[vec![0], vec![2]],
    )]
    // Shape [3, 3] with [2, 2] chunks, maximum [8, unlimited]. The rotation
    // puts the unlimited dimension first, so the multipliers are [4, 1] over
    // (column, row), and slot 2 decodes to chunk row 2, past a 3-row dataset.
    #[case::extensible_array(
        ChunkArrayKind::ExtensibleArray,
        &[2, 2],
        &[3, 3],
        &[MaxExtent::Fixed(8), MaxExtent::Unlimited],
        &[0, 1, 2],
        3,
        &[vec![0, 0], vec![2, 0]],
    )]
    fn a_chunk_at_a_slot_outside_the_dataset_is_dropped(
        #[case] kind: ChunkArrayKind,
        #[case] chunk_dims: &[u64],
        #[case] dims: &[u64],
        #[case] max_dims: &[MaxExtent],
        #[case] slot_of: &[u64],
        #[case] slots: u64,
        #[case] offsets: &[Vec<u64>],
    ) {
        let chunks = [
            indexed_chunk(0x1000),
            indexed_chunk(0x2000),
            indexed_chunk(0x3000),
        ];
        let chunk_bytes = chunk_dims.iter().product::<u64>() * 4;
        let (index, file) = array_index(kind, &chunks, slot_of, slots, chunk_bytes);
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: dims.len() as u8,
            dimensions: dims.to_vec(),
            max_dimensions: Some(max_dims.to_vec()),
        };
        let chunk_dimensions: Vec<u64> = chunk_dims.iter().copied().chain([4]).collect();

        let read = collect_chunks_for_layout_from_source(
            &BytesSource::new(&file),
            index,
            &chunk_dimensions,
            &dataspace,
            4,
            FORMAT_WIDTHS,
        )
        .unwrap();
        let expected: Vec<ChunkInfo> = [0x1000, 0x2000]
            .into_iter()
            .zip(offsets)
            .map(|(address, offsets)| ChunkInfo {
                chunk_size: StoredChunkSize::v4(chunk_bytes),
                filter_mask: 0,
                offsets: offsets.clone(),
                address: StoredAddress::new(address),
            })
            .collect();
        assert_eq!(read, expected);
    }

    // `fuzz_targets/parse_file.rs` reaches this path with a crafted dataspace.
    #[test]
    fn a_dimension_of_no_chunks_is_an_error() {
        let (index, file) = array_index(
            ChunkArrayKind::ExtensibleArray,
            &[indexed_chunk(0x1000)],
            &[0],
            1,
            16,
        );
        // Maximum extent 0 in the trailing dimension, current extent 4.
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![3, 4],
            max_dimensions: Some(vec![MaxExtent::Unlimited, MaxExtent::Fixed(0)]),
        };

        let err = collect_chunks_for_layout_from_source(
            &BytesSource::new(&file),
            index,
            &[2, 2, 4],
            &dataspace,
            4,
            FORMAT_WIDTHS,
        )
        .unwrap_err();
        let FormatError::ChunkedReadError(message) = &err else {
            panic!("expected ChunkedReadError, got {err:?}");
        };
        assert_eq!(
            message,
            "this chunked dataset has a dimension of no chunks at all, so its chunk index \
             numbers nothing; a slot in it cannot be resolved"
        );
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn read_1d_two_chunks_with_deflate() {
        use crate::filter_pipeline::{FILTER_DEFLATE, FilterDescription, FilterPipeline};
        use crate::filters;

        let elem_size = 8usize;
        let ndims = 2;
        let chunk_elems = 10usize;
        let total = 20usize;

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };

        let values: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut file_data = vec![0u8; 0x10000];
        let mut chunk_infos = Vec::new();
        let mut data_offset = 0x2000usize;

        for chunk_idx in 0..2 {
            let start = chunk_idx * chunk_elems;
            let mut chunk_bytes = Vec::new();
            for value in &values[start..start + chunk_elems] {
                chunk_bytes.extend_from_slice(&value.to_le_bytes());
            }
            let dims_u64 = [chunk_elems as u64];
            let ctx = crate::filters::ChunkContext::basic(&dims_u64, elem_size as u32);
            let compressed = filters::compress_chunk(&chunk_bytes, &pipeline, ctx).unwrap();

            file_data[data_offset..data_offset + compressed.len()].copy_from_slice(&compressed);

            chunk_infos.push(ChunkInfo {
                chunk_size: StoredChunkSize::btree_v1(compressed.len() as u32),
                filter_mask: 0,
                offsets: vec![start as u64, 0],
                address: StoredAddress::new(data_offset as u64),
            });

            data_offset += compressed.len() + 16; // some padding
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, WIDTHS);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![chunk_elems as u64, elem_size as u64],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(btree_addr as u64)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![total as u64],
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        let raw = read_chunked_data_cached(
            &file_data,
            RawReadSpec::parse(
                &layout,
                &dataspace,
                &datatype,
                Some(&pipeline),
                FillPattern::ZERO,
            )
            .unwrap(),
            FORMAT_WIDTHS,
            &ChunkCache::new(),
        )
        .unwrap();

        for i in 0..total {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }

        // The streaming reader must reproduce the same decompressed bytes.
        assert_chunked_streams_match(&file_data, &layout, &dataspace, &datatype, Some(&pipeline));
    }

    #[cfg(all(feature = "deflate", feature = "std"))]
    #[test]
    fn filtered_btree_v2_reads_match_buffered_and_bounded_sources() {
        use crate::filter_pipeline::{FILTER_DEFLATE, FilterDescription, FilterPipeline};
        use crate::filters;

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };
        let datatype = make_f64_type();
        let dataspace = btree_v2_dataspace(&[4, 4]);
        let chunk_dims = [2u64, 2u64];
        let fixed_chunk_size = 2 * 2 * 8;
        let values = (0..16).map(f64::from).collect::<Vec<_>>();
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let mut file_data = vec![0u8; 0x3000];
        let mut records = Vec::new();
        let mut data_address = 0x2000usize;
        for chunk_row in 0..2usize {
            for chunk_col in 0..2usize {
                let mut chunk = Vec::new();
                for row in 0..2usize {
                    let first = (chunk_row * 2 + row) * 4 + chunk_col * 2;
                    for value in &values[first..first + 2] {
                        chunk.extend_from_slice(&value.to_le_bytes());
                    }
                }
                let compressed =
                    filters::compress_chunk(&chunk, &pipeline, ChunkContext::basic(&chunk_dims, 8))
                        .unwrap();
                file_data[data_address..data_address + compressed.len()]
                    .copy_from_slice(&compressed);
                records.push(btree_v2_chunk_record(
                    hdf5_pure_format::__private::BTREE_V2_FILTERED_CHUNK,
                    u64::try_from(data_address).unwrap(),
                    &[
                        u64::try_from(chunk_row).unwrap(),
                        u64::try_from(chunk_col).unwrap(),
                    ],
                    Some((u64::try_from(compressed.len()).unwrap(), 0)),
                    fixed_chunk_size,
                    Widths::EIGHT,
                ));
                data_address += compressed.len() + 16;
            }
        }

        let header_address = 0x100usize;
        let root_address = 0x400usize;
        let leaf = btree_v2::leaf(
            hdf5_pure_format::__private::BTREE_V2_FILTERED_CHUNK,
            &records,
        );
        let header = btree_v2::Header::new(
            hdf5_pure_format::__private::BTREE_V2_FILTERED_CHUNK,
            u16::try_from(records[0].len()).unwrap(),
            u64::try_from(root_address).unwrap(),
            u16::try_from(records.len()).unwrap(),
        )
        .build(Widths::EIGHT);
        file_data[header_address..header_address + header.len()].copy_from_slice(&header);
        file_data[root_address..root_address + leaf.len()].copy_from_slice(&leaf);

        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, 2, 8],
            index: ChunkIndexLayout::BTreeV2 {
                address: Some(StoredAddress::new(u64::try_from(header_address).unwrap())),
            },
        };
        assert_eq!(
            assert_chunked_streams_match(
                &file_data,
                &layout,
                &dataspace,
                &datatype,
                Some(&pipeline),
            ),
            raw
        );
    }

    #[rstest]
    #[case::edge_without_skip_flag(ChunkedLayoutFlags::NONE, &[4, 4, 0], ChunkFiltering::Filtered { filter_mask: 3 })]
    #[case::interior_with_skip_flag(
        ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS),
        &[0, 0, 0],
        ChunkFiltering::Filtered { filter_mask: 3 }
    )]
    #[case::row_edge(
        ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS),
        &[4, 0, 0],
        ChunkFiltering::Raw
    )]
    #[case::column_edge(
        ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS),
        &[0, 4, 0],
        ChunkFiltering::Raw
    )]
    #[case::corner_edge(
        ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS),
        &[4, 4, 0],
        ChunkFiltering::Raw
    )]
    // 4x4 chunks over a 6x6 dataset: a chunk at offset 4 in either dimension
    // extends past the extent.
    fn the_flag_and_the_chunk_geometry_decide_the_filtering(
        #[case] flags: ChunkedLayoutFlags,
        #[case] offsets: &[u64],
        #[case] expected: ChunkFiltering,
    ) {
        let chunk = ChunkInfo {
            chunk_size: StoredChunkSize::btree_v1(128),
            filter_mask: 3,
            offsets: offsets.to_vec(),
            address: StoredAddress::new(0x1000),
        };
        assert_eq!(
            ChunkFiltering::for_chunk(flags, &[4, 4], &[6, 6], &chunk),
            expected
        );
    }

    /// Writes a shuffled one-dimensional dataset and overwrites its partial
    /// edge chunk with the unfiltered element bytes, as a library writing under
    /// `H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS` stores that chunk.
    fn file_with_a_raw_partial_edge_chunk(
        values: &[f64],
        chunk_elems: u64,
    ) -> (Vec<u8>, DataLayout, Dataspace, FilterPipeline) {
        use crate::chunked_write;
        use crate::chunked_write::{ChunkOptions, FilterKind};

        let elem_size = size_of::<f64>();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape = [values.len() as u64];
        let mut options = ChunkOptions {
            chunk_dims: Some(vec![chunk_elems]),
            ..ChunkOptions::default()
        };
        options.set_filter(FilterKind::Shuffle);
        let data_address = StoredAddress::new(0x1000);
        let written = chunked_write::build_chunked_data_at_ext(
            &raw,
            &shape,
            ChunkContext::basic(&[chunk_elems], elem_size as u32),
            &options,
            data_address,
            None,
            FillPattern::ZERO,
        )
        .unwrap();

        let base = data_address.get() as usize;
        let mut file_data = vec![0u8; base + written.data_bytes.len()];
        file_data[base..].copy_from_slice(&written.data_bytes);

        let written_layout = DataLayout::parse(&written.layout_message, 8, 8).unwrap();
        let DataLayout::Chunked {
            chunk_dimensions,
            index,
            ..
        } = written_layout
        else {
            panic!("the chunked writer emitted {written_layout:?}");
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: shape.to_vec(),
            max_dimensions: None,
        };

        for chunk in collect_chunks_for_layout_from_source(
            &BytesSource::new(&file_data),
            index,
            &chunk_dimensions,
            &dataspace,
            elem_size as u64,
            FORMAT_WIDTHS,
        )
        .unwrap()
        {
            let offset = chunk.offsets[0];
            if offset + chunk_elems <= shape[0] {
                continue;
            }
            let unfiltered: Vec<u8> = (offset..offset + chunk_elems)
                .flat_map(|i| values.get(i as usize).copied().unwrap_or(0.0).to_le_bytes())
                .collect();
            assert_eq!(
                chunk.chunk_size.get() as usize,
                unfiltered.len(),
                "shuffle keeps a chunk's stored length"
            );
            let at = chunk.address.get() as usize;
            file_data[at..at + unfiltered.len()].copy_from_slice(&unfiltered);
        }

        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS),
            chunk_dimensions,
            index,
        };
        let pipeline = FilterPipeline::parse(&written.pipeline_message.unwrap()).unwrap();
        (file_data, layout, dataspace, pipeline)
    }

    #[test]
    fn a_raw_partial_edge_chunk_decodes_through_every_reader() {
        let values: Vec<f64> = (0..10).map(f64::from).collect();
        let (file_data, layout, dataspace, pipeline) =
            file_with_a_raw_partial_edge_chunk(&values, 4);
        let datatype = make_f64_type();
        let spec = RawReadSpec::parse(
            &layout,
            &dataspace,
            &datatype,
            Some(&pipeline),
            FillPattern::ZERO,
        )
        .unwrap();
        let source = BytesSource::new(&file_data);
        let decoded = |bytes: Vec<u8>| -> Vec<f64> {
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect()
        };

        assert_eq!(
            decoded(read_chunked_data_from_source(&source, spec, FORMAT_WIDTHS).unwrap()),
            values
        );
        assert_eq!(
            decoded(
                read_chunked_data_cached(&file_data, spec, FORMAT_WIDTHS, &ChunkCache::new())
                    .unwrap()
            ),
            values
        );
        assert_eq!(
            decoded(
                read_chunked_data_cached_from_source(
                    &source,
                    spec,
                    FORMAT_WIDTHS,
                    &ChunkCache::new(),
                )
                .unwrap()
            ),
            values
        );
        let rows = read_chunked_rows_from_source(
            &source,
            spec,
            FORMAT_WIDTHS,
            &ChunkCache::new(),
            CachePass::LRU,
            8,
            2,
        )
        .unwrap()
        .unwrap();
        assert_eq!(decoded(rows), values[8..].to_vec());
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn a_raw_partial_edge_chunk_skips_the_deflate_decoder() {
        use crate::filter_pipeline::{FILTER_DEFLATE, FilterDescription};

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };
        let stored: Vec<u8> = (0..32u8).collect();
        let ctx = ChunkContext::basic(&[4], 8);
        let mut scratch = FilterScratch::new();

        let decoded = decode_chunk(
            &mut scratch,
            Cow::Borrowed(&stored),
            Some(&pipeline),
            ctx,
            ChunkFiltering::Raw,
        )
        .unwrap();
        assert_eq!(decoded.as_ref(), stored);

        let err = decode_chunk(
            &mut scratch,
            Cow::Borrowed(&stored),
            Some(&pipeline),
            ctx,
            ChunkFiltering::Filtered { filter_mask: 0 },
        )
        .unwrap_err();
        let FormatError::FilterError(reason) = &err else {
            panic!("expected FilterError, got {err:?}");
        };
        assert!(
            crate::test_data::DEFLATE_HEADER_ERRORS.contains(&reason.as_str()),
            "{reason:?}"
        );
    }

    #[test]
    fn read_2d_four_chunks() {
        // 4x6 dataset with chunk size 2x3 => 4 chunks
        let elem_size = 4usize; // f32
        let ndims = 3; // rank(2) + 1
        let ds_dims = [4usize, 6];
        let chunk_dims = [2usize, 3];

        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut file_data = vec![0u8; 0x10000];
        let mut chunk_infos = Vec::new();
        let mut data_offset = 0x2000usize;

        // Generate chunks: (0,0), (0,3), (2,0), (2,3)
        for row_start in (0..ds_dims[0]).step_by(chunk_dims[0]) {
            for col_start in (0..ds_dims[1]).step_by(chunk_dims[1]) {
                let mut chunk_bytes = Vec::new();
                for r in 0..chunk_dims[0] {
                    for c in 0..chunk_dims[1] {
                        let gr = row_start + r;
                        let gc = col_start + c;
                        let val = if gr < ds_dims[0] && gc < ds_dims[1] {
                            values[gr * ds_dims[1] + gc]
                        } else {
                            0.0
                        };
                        chunk_bytes.extend_from_slice(&val.to_le_bytes());
                    }
                }

                let chunk_size = chunk_bytes.len();
                file_data[data_offset..data_offset + chunk_size].copy_from_slice(&chunk_bytes);

                chunk_infos.push(ChunkInfo {
                    chunk_size: StoredChunkSize::btree_v1(chunk_size as u32),
                    filter_mask: 0,
                    offsets: vec![row_start as u64, col_start as u64, 0],
                    address: StoredAddress::new(data_offset as u64),
                });

                data_offset += chunk_size + 8;
            }
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, WIDTHS);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![chunk_dims[0] as u64, chunk_dims[1] as u64, elem_size as u64],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(btree_addr as u64)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![ds_dims[0] as u64, ds_dims[1] as u64],
            max_dimensions: None,
        };
        let datatype = make_f32_type();

        let raw = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &ChunkCache::new(),
        )
        .unwrap();
        assert_eq!(raw.len(), 24 * 4);

        for i in 0..24 {
            let val = f32::from_le_bytes(raw[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(val, i as f32, "mismatch at element {i}");
        }

        #[cfg(feature = "std")]
        assert_chunked_streams_match(&file_data, &layout, &dataspace, &datatype, None);
    }

    #[test]
    fn wrong_node_type_error() {
        // Build a type-0 B-tree and try to collect chunk info
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TREE");
        buf.push(0); // type 0, not 1
        buf.push(0);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&[0xFF; 16]); // siblings
        // final key
        buf.extend_from_slice(&[0u8; 24]);

        let mut file_data = vec![0u8; 512];
        file_data[..buf.len()].copy_from_slice(&buf);

        let err = collect_chunk_info(&file_data, StoredAddress::new(0), 2, OffsetWidth::Eight)
            .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeNodeType(0));
    }

    // --- Implicit chunk generation tests ---

    #[rstest]
    #[case::one_dimensional(
        vec![100],
        None,
        vec![20],
        8,
        vec![
            (0x1000, vec![0], 160),
            (0x10a0, vec![20], 160),
            (0x1140, vec![40], 160),
            (0x11e0, vec![60], 160),
            (0x1280, vec![80], 160),
        ],
    )]
    #[case::two_dimensional(
        vec![10, 6],
        None,
        vec![4, 3],
        4,
        vec![
            (0x1000, vec![0, 0], 48),
            (0x1030, vec![0, 3], 48),
            (0x1060, vec![4, 0], 48),
            (0x1090, vec![4, 3], 48),
            (0x10c0, vec![8, 0], 48),
            (0x10f0, vec![8, 3], 48),
        ],
    )]
    #[case::partial_edge(
        vec![25],
        None,
        vec![10],
        8,
        vec![
            (0x1000, vec![0], 80),
            (0x1050, vec![10], 80),
            (0x10a0, vec![20], 80),
        ],
    )]
    #[case::large_element_width(
        vec![1],
        None,
        vec![1],
        u64::from(u32::MAX) + 1,
        vec![(0x1000, vec![0], u64::from(u32::MAX) + 1)],
    )]
    #[case::maximum_extent_stride(
        vec![4, 4],
        Some(vec![MaxExtent::Fixed(4), MaxExtent::Fixed(8)]),
        vec![2, 2],
        4,
        vec![
            (0x1000, vec![0, 0], 16),
            (0x1010, vec![0, 2], 16),
            (0x1040, vec![2, 0], 16),
            (0x1050, vec![2, 2], 16),
        ],
    )]
    fn implicit_chunks_follow_the_index_grid(
        #[case] dims: Vec<u64>,
        #[case] max_dims: Option<Vec<MaxExtent>>,
        #[case] chunk_dims: Vec<u64>,
        #[case] element_size: u64,
        #[case] expected: Vec<(u64, Vec<u64>, u64)>,
    ) {
        let grid =
            ChunkGrid::new(&chunk_dims, &dims, max_dims.as_deref(), GridOrder::RowMajor).unwrap();
        let chunks =
            generate_implicit_chunks(StoredAddress::new(0x1000), &grid, element_size).unwrap();
        let actual: Vec<_> = chunks
            .into_iter()
            .map(|chunk| (chunk.address.get(), chunk.offsets, chunk.chunk_size.get()))
            .collect();
        assert_eq!(actual, expected);
    }

    // --- V4 single chunk synthetic test ---

    #[test]
    fn read_v4_single_chunk_synthetic() {
        // Build a synthetic v4 single chunk dataset (no filters)
        let values: Vec<f64> = vec![10.0, 20.0, 30.0];
        let elem_size = 8usize;
        let chunk_elems = 3usize;

        let mut file_data = vec![0u8; 0x2000];
        let data_addr = 0x1000usize;
        for (i, &v) in values.iter().enumerate() {
            file_data[data_addr + i * elem_size..data_addr + (i + 1) * elem_size]
                .copy_from_slice(&v.to_le_bytes());
        }

        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![chunk_elems as u64, elem_size as u64],
            index: ChunkIndexLayout::SingleChunk {
                filtered: None,
                address: Some(StoredAddress::new(data_addr as u64)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![3],
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        let raw = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &ChunkCache::new(),
        )
        .unwrap();
        assert_eq!(raw.len(), 24);
        for i in 0..3 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, values[i]);
        }
    }

    // --- Cached read tests ---

    use crate::chunk_cache::ChunkCache;
    use crate::chunk_cache::ChunkCacheConfig;

    #[test]
    fn cached_read_populates_index_and_returns_correct_data() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();

        assert!(!cache.stats().index_loaded());
        let raw = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &cache,
        )
        .unwrap();
        assert!(cache.stats().index_loaded());
        assert_eq!(raw.len(), 20 * 8);
        for i in 0..20 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64);
        }
    }

    #[test]
    fn cached_read_second_call_uses_cache() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();

        // First read — populates index + decompressed cache
        let raw1 = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &cache,
        )
        .unwrap();
        assert!(cache.stats().index_loaded());
        assert!(cache.stats().cached_chunks() > 0);

        // Second read — should hit the decompressed cache
        let raw2 = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &cache,
        )
        .unwrap();
        assert_eq!(raw1, raw2);
    }

    #[test]
    fn cached_read_with_partial_last_chunk() {
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        let datatype = make_f64_type();
        let cache = ChunkCache::new();

        let raw = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &cache,
        )
        .unwrap();
        assert_eq!(raw.len(), 25 * 8);
        for i in 0..25 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum GridIndex {
        BTreeV1,
        ExtensibleArray,
        FixedArray,
        Implicit,
    }

    /// Returns the little-endian bytes of the values in `rows` and `columns` of the grid dataset,
    /// in row-major order.
    fn grid_bytes(rows: Range<u64>, columns: Range<u64>) -> Vec<u8> {
        rows.flat_map(|row| columns.clone().map(move |column| grid_value(row, column)))
            .flat_map(f32::to_le_bytes)
            .collect()
    }

    /// Returns the value at `row` and `column` of the grid dataset, its row-major position.
    fn grid_value(row: u64, column: u64) -> f32 {
        let [_, columns] = GRID_DIMS;
        (row * columns + column) as f32
    }

    /// Builds a file image of a `GRID_DIMS` dataset of `f32` in `GRID_CHUNK` chunks under the
    /// `kind` index, and returns it with the dataset's layout and dataspace.
    ///
    /// The chunks are stored in the reverse of index order, except under the implicit index, whose
    /// order fixes their addresses.
    fn indexed_grid_file(kind: GridIndex) -> (Vec<u8>, DataLayout, Dataspace) {
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: GRID_DIMS.to_vec(),
            max_dimensions: match kind {
                GridIndex::ExtensibleArray => {
                    Some(vec![MaxExtent::Unlimited, MaxExtent::Fixed(GRID_DIMS[1])])
                }
                GridIndex::BTreeV1 | GridIndex::FixedArray | GridIndex::Implicit => None,
            },
        };
        let order = match kind {
            GridIndex::ExtensibleArray => GridOrder::UnlimitedFirst,
            GridIndex::BTreeV1 | GridIndex::FixedArray | GridIndex::Implicit => GridOrder::RowMajor,
        };
        let slots: Vec<_> = index_grid(&dataspace, &GRID_CHUNK, order)
            .unwrap()
            .slots_in_extent()
            .unwrap()
            .collect();
        let address = |position: usize| match kind {
            GridIndex::Implicit => GRID_DATA + position as u64 * GRID_CHUNK_BYTES,
            GridIndex::BTreeV1 | GridIndex::ExtensibleArray | GridIndex::FixedArray => {
                GRID_DATA + (slots.len() - 1 - position) as u64 * GRID_CHUNK_BYTES
            }
        };
        let mut file = vec![0u8; (GRID_DATA + slots.len() as u64 * GRID_CHUNK_BYTES) as usize];
        for (position, slot) in slots.iter().enumerate() {
            let [row, column] = slot.offsets[..] else {
                panic!("a grid chunk has two offsets, not {:?}", slot.offsets);
            };
            let payload: Vec<u8> = (row..row + GRID_CHUNK[0])
                .flat_map(|row| {
                    (column..column + GRID_CHUNK[1]).map(move |column| {
                        if row < GRID_DIMS[0] && column < GRID_DIMS[1] {
                            grid_value(row, column)
                        } else {
                            0.0
                        }
                    })
                })
                .flat_map(f32::to_le_bytes)
                .collect();
            let at = address(position) as usize;
            file[at..at + payload.len()].copy_from_slice(&payload);
        }
        let (index, index_bytes) = match kind {
            GridIndex::BTreeV1 => {
                let chunks: Vec<_> = slots
                    .iter()
                    .enumerate()
                    .map(|(position, slot)| ChunkInfo {
                        chunk_size: StoredChunkSize::btree_v1(GRID_CHUNK_BYTES as u32),
                        filter_mask: 0,
                        offsets: slot.offsets.clone(),
                        address: StoredAddress::new(address(position)),
                    })
                    .collect();
                (
                    ChunkIndexLayout::BTreeV1 {
                        address: Some(StoredAddress::new(0)),
                    },
                    build_chunk_btree_leaf(&chunks, 3, WIDTHS),
                )
            }
            GridIndex::ExtensibleArray | GridIndex::FixedArray => {
                let records: Vec<_> = (0..slots.len())
                    .map(|position| ChunkRecord {
                        address: StoredAddress::new(address(position)),
                        stored_size: GRID_CHUNK_BYTES,
                        filter_mask: 0,
                    })
                    .collect();
                let slot_of: Vec<_> = slots.iter().map(|slot| slot.slot).collect();
                let array = match kind {
                    GridIndex::ExtensibleArray => ChunkArrayKind::ExtensibleArray,
                    _ => ChunkArrayKind::FixedArray,
                };
                array_index(
                    array,
                    &records,
                    &slot_of,
                    slots.len() as u64,
                    GRID_CHUNK_BYTES,
                )
            }
            GridIndex::Implicit => (
                ChunkIndexLayout::Implicit {
                    address: Some(StoredAddress::new(GRID_DATA)),
                },
                Vec::new(),
            ),
        };
        assert!(index_bytes.len() as u64 <= GRID_DATA, "{kind:?}");
        file[..index_bytes.len()].copy_from_slice(&index_bytes);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: GRID_CHUNK.iter().copied().chain([4]).collect(),
            index,
        };
        (file, layout, dataspace)
    }

    #[rstest]
    fn reads_through_a_retained_or_a_dropped_index_return_the_same_bytes(
        #[values(
            GridIndex::BTreeV1,
            GridIndex::ExtensibleArray,
            GridIndex::FixedArray,
            GridIndex::Implicit
        )]
        kind: GridIndex,
        #[values(false, true)] index_cache: bool,
    ) {
        let (file, layout, dataspace) = indexed_grid_file(kind);
        let datatype = make_f32_type();
        let spec = RawReadSpec::plain(&layout, &dataspace, &datatype);
        let source = BytesSource::new(&file);
        let region = ReadRegion::parse(spec, &[1, 1], &[4, 3]).unwrap();
        let config = ChunkCacheConfig::disabled().with_index_cache(index_cache);
        let (buffered, streaming, windowed) = (
            ChunkCache::with_config(config),
            ChunkCache::with_config(config),
            ChunkCache::with_config(config),
        );
        let whole = grid_bytes(0..6, 0..5);
        let window = grid_bytes(1..5, 1..4);
        let read = || {
            assert_eq!(
                read_chunked_data_cached(&file, spec, FORMAT_WIDTHS, &buffered).unwrap(),
                whole
            );
            assert_eq!(
                read_chunked_data_cached_from_source(&source, spec, FORMAT_WIDTHS, &streaming)
                    .unwrap(),
                whole
            );
            assert_eq!(
                read_chunked_region_from_source(
                    &source,
                    &region,
                    FORMAT_WIDTHS,
                    &windowed,
                    CachePass::LRU
                )
                .unwrap(),
                window
            );
        };

        read();
        read();

        assert_eq!(buffered.stats().index_loaded(), index_cache);
        assert_eq!(streaming.stats().index_loaded(), index_cache);
        assert_eq!(windowed.stats().index_loaded(), index_cache);
    }

    #[rstest]
    fn both_collectors_list_the_chunks_of_the_grid_in_index_order(
        #[values(
            GridIndex::BTreeV1,
            GridIndex::ExtensibleArray,
            GridIndex::FixedArray,
            GridIndex::Implicit
        )]
        kind: GridIndex,
    ) {
        let (file, layout, dataspace) = indexed_grid_file(kind);
        let DataLayout::Chunked {
            chunk_dimensions,
            index,
            ..
        } = &layout
        else {
            panic!("the grid dataset is chunked, not {layout:?}");
        };
        // `indexed_grid_file` stores the chunks in the reverse of index order, except under the
        // Implicit index.
        let origins = [
            [0, 0],
            [0, 2],
            [0, 4],
            [2, 0],
            [2, 2],
            [2, 4],
            [4, 0],
            [4, 2],
            [4, 4],
        ];
        let expected: Vec<_> = origins
            .iter()
            .zip(0u64..)
            .map(|(origin, position)| match kind {
                GridIndex::BTreeV1 => ChunkInfo {
                    chunk_size: StoredChunkSize::btree_v1(GRID_CHUNK_BYTES as u32),
                    filter_mask: 0,
                    offsets: origin.iter().copied().chain([0]).collect(),
                    address: StoredAddress::new(GRID_DATA + (8 - position) * GRID_CHUNK_BYTES),
                },
                GridIndex::ExtensibleArray | GridIndex::FixedArray => ChunkInfo {
                    chunk_size: StoredChunkSize::v4(GRID_CHUNK_BYTES),
                    filter_mask: 0,
                    offsets: origin.to_vec(),
                    address: StoredAddress::new(GRID_DATA + (8 - position) * GRID_CHUNK_BYTES),
                },
                GridIndex::Implicit => ChunkInfo {
                    chunk_size: StoredChunkSize::v4(GRID_CHUNK_BYTES),
                    filter_mask: 0,
                    offsets: origin.to_vec(),
                    address: StoredAddress::new(GRID_DATA + position * GRID_CHUNK_BYTES),
                },
            })
            .collect();

        let buffered = collect_chunks_for_layout(
            MetadataFrame::new(PhysicalAddress::new(0), &file).unwrap(),
            *index,
            chunk_dimensions,
            &dataspace,
            4,
            FORMAT_WIDTHS,
        )
        .unwrap();
        let streamed = collect_chunks_for_layout_from_source(
            &BytesSource::new(&file),
            *index,
            chunk_dimensions,
            &dataspace,
            4,
            FORMAT_WIDTHS,
        )
        .unwrap();

        assert_eq!(buffered, expected);
        assert_eq!(streamed, expected);
    }

    #[rstest]
    #[case::interior(&[1, 1], &[4, 3], 6)]
    #[case::one_chunk(&[2, 0], &[2, 2], 1)]
    #[case::partial_edge_chunk(&[5, 4], &[1, 1], 1)]
    #[case::whole(&[0, 0], &[6, 5], 9)]
    fn a_windowed_read_looks_up_only_the_chunks_it_intersects(
        #[values(
            GridIndex::BTreeV1,
            GridIndex::ExtensibleArray,
            GridIndex::FixedArray,
            GridIndex::Implicit
        )]
        kind: GridIndex,
        #[case] start: &[u64],
        #[case] count: &[u64],
        #[case] intersected: u64,
    ) {
        let (file, layout, dataspace) = indexed_grid_file(kind);
        let datatype = make_f32_type();
        let spec = RawReadSpec::plain(&layout, &dataspace, &datatype);
        let region = ReadRegion::parse(spec, start, count).unwrap();
        let cache = ChunkCache::with_config(ChunkCacheConfig::new().with_max_slots(64));

        let read = read_chunked_region_from_source(
            &BytesSource::new(&file),
            &region,
            FORMAT_WIDTHS,
            &cache,
            CachePass::LRU,
        )
        .unwrap();

        assert_eq!(
            read,
            grid_bytes(start[0]..start[0] + count[0], start[1]..start[1] + count[1])
        );
        let stats = cache.stats();
        assert_eq!(stats.lookups(), intersected);
        assert_eq!(stats.cached_chunks() as u64, intersected);
    }

    const GRID_DIMS: [u64; 2] = [6, 5];

    const GRID_CHUNK: [u64; 2] = [2, 2];

    const GRID_CHUNK_BYTES: u64 = 16;

    /// The lowest chunk address of the grid dataset, after the bytes of its index.
    const GRID_DATA: u64 = 0x1000;

    // --- Windowed row-read guards (read_chunked_rows_from_source) ---

    /// A rank-0 (scalar) chunked layout leaves `chunk_dims` empty; the windowed
    /// reader must fall back (`Ok(None)`) rather than index `chunk_dims[0]` and
    /// panic. Only a crafted file reaches this (valid chunked layouts are rank
    /// >= 1), but the reader parses untrusted input.
    #[test]
    fn windowed_rows_rank0_chunked_falls_back() {
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![8], // dimensionality 1 => rank 0
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Scalar,
            rank: 0,
            dimensions: vec![],
            max_dimensions: None,
        };
        let cache = ChunkCache::new();
        let out = read_chunked_rows_from_source(
            &BytesSource::new(b""),
            RawReadSpec::plain(&layout, &dataspace, &make_f64_type()),
            FORMAT_WIDTHS,
            &cache,
            CachePass::LRU,
            0,
            1,
        )
        .expect("rank-0 chunked must not panic");
        assert!(
            out.is_none(),
            "rank-0 chunked should fall back to the whole reader"
        );
    }

    #[test]
    fn windowed_rows_inner_dim_product_overflow_errors() {
        let big: u64 = 1 << 22;
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![1, 2, 2, 2, 8],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 4,
            dimensions: vec![1, big, big, big],
            max_dimensions: None,
        };
        let cache = ChunkCache::new();
        let err = read_chunked_rows_from_source(
            &BytesSource::new(b""),
            RawReadSpec::plain(&layout, &dataspace, &make_f64_type()),
            FORMAT_WIDTHS,
            &cache,
            CachePass::LRU,
            0,
            1,
        )
        .expect_err("overflowing inner-dim product must error");
        assert_eq!(
            err,
            FormatError::OffsetOverflow {
                offset: 140_737_488_355_328,
                length: 4_194_304,
            },
        );
    }

    #[test]
    fn windowed_rows_huge_chunk_geometry_is_rejected() {
        // The source is empty, so the geometry error precedes a read failure.
        let big: u64 = 1 << 22;
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![2, big, big, big, 8],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 4,
            dimensions: vec![4, 2, 2, 2],
            max_dimensions: None,
        };
        let cache = ChunkCache::new();
        let err = read_chunked_rows_from_source(
            &BytesSource::new(b""),
            RawReadSpec::plain(&layout, &dataspace, &make_f64_type()),
            FORMAT_WIDTHS,
            &cache,
            CachePass::LRU,
            0,
            1,
        )
        .expect_err("an impossible chunk geometry must error");
        assert_eq!(
            err,
            FormatError::InvalidChunkGeometry(
                "chunk logical byte size exceeds the reader's single-chunk allocation limit",
            )
        );
    }

    #[test]
    fn windowed_rows_unallocated_index_matches_whole_read() {
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 8],
            index: ChunkIndexLayout::BTreeV1 { address: None },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![10],
            max_dimensions: None,
        };
        let seven = 7.0f64.to_le_bytes();
        let datatype = make_f64_type();
        for fill in [
            FillPattern::ZERO,
            FillPattern::new(FillTime::IfSet, Some(&seven), convert::nz(8)),
        ] {
            let cache = ChunkCache::new();
            let spec = RawReadSpec::parse(&layout, &dataspace, &datatype, None, fill).unwrap();
            let whole = read_chunked_data_cached_from_source(
                &BytesSource::new(b""),
                spec,
                FORMAT_WIDTHS,
                &cache,
            )
            .expect("an unallocated dataset reads as fill");
            assert_eq!(whole.len(), 80);

            let window = read_chunked_rows_from_source(
                &BytesSource::new(b""),
                spec,
                FORMAT_WIDTHS,
                &cache,
                CachePass::LRU,
                2,
                4,
            )
            .expect("a window over an unallocated index reads as fill")
            .expect("rank-1 windows are supported");
            assert_eq!(window, whole[16..48], "the window must match those rows");

            let empty = read_chunked_rows_from_source(
                &BytesSource::new(b""),
                spec,
                FORMAT_WIDTHS,
                &cache,
                CachePass::LRU,
                0,
                0,
            )
            .expect("empty window must still be Ok");
            assert_eq!(empty, Some(Vec::new()));
        }
    }

    #[rstest]
    #[case::leading_row(4_294_967_424, 1, 0, 1, 8)]
    #[case::row_above_u32(4_294_967_424, 1, 4_294_967_360, 1, 8)]
    #[case::zero_columns(4_294_967_297, 0, 1, 1, 0)]
    fn windowed_rows_materialize_only_the_selected_geometry(
        #[case] rows: u64,
        #[case] columns: u64,
        #[case] start: u64,
        #[case] count: u64,
        #[case] expected_bytes: usize,
    ) {
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 1, 8],
            index: ChunkIndexLayout::BTreeV1 { address: None },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![rows, columns],
            max_dimensions: None,
        };
        let seven = 7.0f64.to_le_bytes();
        let datatype = make_f64_type();
        let fill = FillPattern::new(FillTime::IfSet, Some(&seven), convert::nz(8));
        let spec = RawReadSpec::parse(&layout, &dataspace, &datatype, None, fill).unwrap();
        let output = read_chunked_rows_from_source(
            &BytesSource::new(b""),
            spec,
            FORMAT_WIDTHS,
            &ChunkCache::new(),
            CachePass::LRU,
            start,
            count,
        )
        .unwrap()
        .unwrap();
        assert_eq!(output, seven.repeat(expected_bytes / 8));
    }

    #[test]
    fn windowed_rows_read_an_early_chunk_in_a_large_dataspace() {
        let chunk = ChunkInfo {
            chunk_size: StoredChunkSize::BTreeV1(32),
            filter_mask: 0,
            offsets: vec![0, 0, 0],
            address: StoredAddress::new(512),
        };
        let mut bytes = build_chunk_btree_leaf(&[chunk], 3, WIDTHS);
        bytes.resize(512, 0);
        for value in [11.0f64, 12.0, 13.0, 14.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 1, 8],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![4_294_967_424, 1],
            max_dimensions: None,
        };
        let datatype = make_f64_type();
        let result = read_chunked_rows_from_source(
            &BytesSource::new(&bytes),
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            FORMAT_WIDTHS,
            &ChunkCache::new(),
            CachePass::LRU,
            2,
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result, 13.0f64.to_le_bytes());
    }

    #[rstest]
    #[case::above_u32_max(4_294_967_424, 4_294_967_360)]
    #[case::overflowing_chunk_end(u64::MAX, u64::MAX - 3)]
    fn windowed_rows_clip_high_chunk_coordinates(#[case] extent: u64, #[case] chunk_start: u64) {
        let chunk = ChunkInfo {
            chunk_size: StoredChunkSize::BTreeV1(4),
            filter_mask: 0,
            offsets: vec![chunk_start, 0, 0],
            address: StoredAddress::new(512),
        };
        let mut bytes = build_chunk_btree_leaf(&[chunk], 3, WIDTHS);
        bytes.resize(512, 0);
        bytes.extend_from_slice(&[11, 12, 13, 14]);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 1, 1],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![extent, 1],
            max_dimensions: None,
        };
        let datatype = Datatype::FixedPoint(
            FixedPointType::new(1, byte_order::DatatypeByteOrder::LittleEndian, false, 0, 8)
                .unwrap(),
        );
        let cache = ChunkCache::new();
        let output = read_chunked_rows_from_source(
            &BytesSource::new(&bytes),
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap(),
            FORMAT_WIDTHS,
            &cache,
            CachePass::LRU,
            chunk_start + 1,
            2,
        )
        .unwrap()
        .unwrap();
        assert_eq!(output, [12, 13]);
    }

    #[rstest]
    #[case::absent_index(false)]
    #[case::present_index(true)]
    fn whole_chunked_reads_return_an_empty_output_for_large_zero_element_geometry(
        #[case] allocated: bool,
        #[values(false, true)] unreadable_fill: bool,
    ) {
        let bytes = build_chunk_btree_leaf(&[], 3, WIDTHS);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 1, 8],
            index: ChunkIndexLayout::BTreeV1 {
                address: allocated.then_some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![4_294_967_297, 0],
            max_dimensions: None,
        };
        let datatype = make_f64_type();
        let fill = if unreadable_fill {
            FillPattern::UNKNOWN
        } else {
            FillPattern::ZERO
        };
        let spec = RawReadSpec::parse(&layout, &dataspace, &datatype, None, fill).unwrap();
        let source = BytesSource::new(&bytes);
        assert_eq!(
            read_chunked_data_from_source(&source, spec, FORMAT_WIDTHS).unwrap(),
            Vec::<u8>::new()
        );
        assert_eq!(
            read_chunked_data_cached_from_source(&source, spec, FORMAT_WIDTHS, &ChunkCache::new())
                .unwrap(),
            Vec::<u8>::new()
        );
        assert_eq!(
            read_chunked_data_cached(&bytes, spec, FORMAT_WIDTHS, &ChunkCache::new()).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[rstest]
    #[case::rank(vec![4, 8], FormatError::ChunkedReadError("rank mismatch: dataspace has 2 dims, layout has 2 chunk dims (rank=1)".into()))]
    #[case::no_dimensions(vec![], FormatError::ChunkedReadError("chunked layout has no dimensions".into()))]
    #[case::oversized_chunk(vec![u64::MAX, 2, 8], FormatError::InvalidChunkGeometry("chunk logical byte size exceeds the reader's single-chunk allocation limit"))]
    fn empty_chunked_reads_preserve_geometry_validation(
        #[case] chunk_dimensions: Vec<u64>,
        #[case] error: FormatError,
        #[values(false, true)] allocated: bool,
    ) {
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions,
            index: ChunkIndexLayout::BTreeV1 {
                address: allocated.then_some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![0, 0],
            max_dimensions: None,
        };
        let datatype = make_f64_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::UNKNOWN).unwrap();
        let source = BytesSource::new(b"");
        let expected = if allocated {
            Err(error)
        } else {
            Ok(Vec::new())
        };
        assert_eq!(
            read_chunked_data_from_source(&source, spec, FORMAT_WIDTHS),
            expected
        );
        assert_eq!(
            read_chunked_data_cached_from_source(&source, spec, FORMAT_WIDTHS, &ChunkCache::new()),
            expected
        );
        assert_eq!(
            read_chunked_data_cached(b"", spec, FORMAT_WIDTHS, &ChunkCache::new()),
            expected
        );
    }

    #[rstest]
    fn empty_chunked_reads_reject_zero_width_datatypes(#[values(false, true)] allocated: bool) {
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 1, 8],
            index: ChunkIndexLayout::BTreeV1 {
                address: allocated.then_some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![0, 0],
            max_dimensions: None,
        };
        let datatype = Datatype::Array {
            base_type: Box::new(make_f64_type()),
            dimensions: vec![0],
        };
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::UNKNOWN).unwrap();
        let source = BytesSource::new(b"");
        let expected = FormatError::ZeroSizedDatatype {
            class: ARRAY_DATATYPE_CLASS,
        };
        assert_eq!(
            read_chunked_data_from_source(&source, spec, FORMAT_WIDTHS).unwrap_err(),
            expected
        );
        assert_eq!(
            read_chunked_data_cached_from_source(&source, spec, FORMAT_WIDTHS, &ChunkCache::new())
                .unwrap_err(),
            expected
        );
        assert_eq!(
            read_chunked_data_cached(b"", spec, FORMAT_WIDTHS, &ChunkCache::new()).unwrap_err(),
            expected
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn whole_chunked_reads_reject_an_output_above_the_host_limit_before_reading() {
        assert_eq!(usize::BITS, 32);
        let layout = DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions: vec![4, 1, 8],
            index: ChunkIndexLayout::BTreeV1 {
                address: Some(StoredAddress::new(0)),
            },
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![4_294_967_424, 1],
            max_dimensions: None,
        };
        let datatype = make_f64_type();
        let spec = RawReadSpec::plain(&layout, &dataspace, &datatype);
        let source = BytesSource::new(b"");
        let expected = FormatError::ValueTooLargeForPlatform {
            value: 34_359_739_392,
            target: "usize",
        };
        assert_eq!(
            read_chunked_data_from_source(&source, spec, FORMAT_WIDTHS).unwrap_err(),
            expected
        );
        assert_eq!(
            read_chunked_data_cached_from_source(&source, spec, FORMAT_WIDTHS, &ChunkCache::new())
                .unwrap_err(),
            expected
        );
        assert_eq!(
            read_chunked_data_cached(b"", spec, FORMAT_WIDTHS, &ChunkCache::new()).unwrap_err(),
            expected
        );
    }

    /// The class field value for an array datatype.
    ///
    /// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    const ARRAY_DATATYPE_CLASS: u8 = 10;
    // "The Data Storage - Filter Pipeline Message" in format specification, version 4.0,
    // defines pipeline message version 2.
    const FILTER_PIPELINE_VERSION: u8 = 2;
    // "Version 1 B-trees" in format specification, version 4.0, defines the filter mask
    // of a raw data chunk key, with one bit per pipeline entry, starting at bit zero.
    const FIRST_FILTER_MASK: u32 = 1;
    const WIDTHS: Widths = Widths::EIGHT;
}
