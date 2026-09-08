//! Chunked dataset reading: B-tree v1 type 1 traversal and chunk assembly.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, format, vec, vec::Vec};
#[cfg(feature = "std")]
use std::borrow::Cow;

use core::num::NonZeroUsize;

use crate::btree_v1::btree_v1_node_header_size;
use crate::bytes::read_offset;
use crate::chunk_cache::{CachePass, ChunkCache};
use crate::chunk_grid::{ChunkGrid, GridOrder};
use crate::chunk_span::ChunkSpanReader;
use crate::convert::{TryToUsize, nonzero_usize_from, slice_range, u32_from};
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::error::FormatError;
use crate::extensible_array::{
    ExtensibleArrayHeader, read_extensible_array_chunks, read_extensible_array_chunks_from_source,
};
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::{ChunkContext, FilterScratch, decompress_chunk_with};
use crate::fixed_array::{
    FixedArrayHeader, read_fixed_array_chunks, read_fixed_array_chunks_from_source,
};
use crate::read_spec::RawReadSpec;
use crate::source::Source;

/// Decompress all chunks, reading each chunk's bytes from a [`Source`].
///
/// Decompression is sequential: a `ReadSeekSource` serializes on its mutex, so
/// there is nothing for a concurrent decoder to overlap with here.
fn decompress_all_chunks_from_source<S: Source + ?Sized>(
    source: &S,
    chunks: &[ChunkInfo],
    pipeline: Option<&FilterPipeline>,
    ctx: ChunkContext<'_>,
) -> Result<Vec<Vec<u8>>, FormatError> {
    let mut result = Vec::with_capacity(chunks.len());
    // One decoder for every chunk; see `FilterScratch`.
    let mut scratch = FilterScratch::new();
    for chunk_info in chunks {
        let raw_chunk =
            source.read_exact_at(chunk_info.address, chunk_info.chunk_size.to_usize()?)?;
        let decompressed = if let Some(pl) = pipeline {
            decompress_chunk_with(&mut scratch, &raw_chunk, pl, ctx, chunk_info.filter_mask)?
        } else {
            raw_chunk
        };
        result.push(decompressed);
    }
    Ok(result)
}

/// Information about a single chunk in a chunked dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInfo {
    /// Size of chunk data in the file (after compression).
    pub chunk_size: u32,
    /// Bitmask of filters that were NOT applied (0 = all applied).
    pub filter_mask: u32,
    /// N-dimensional offset of this chunk in dataset space.
    pub offsets: Vec<u64>,
    /// File address of the chunk data.
    pub address: u64,
}

/// Size of a single key in a version 1 B-tree of type 1 (raw data chunks):
/// `chunk_size(4) + filter_mask(4) + ndims * offset_size`. The `ndims` trailing
/// offsets are the per-dimension scaled chunk coordinates (rank + 1 values, the
/// extra one being the element-offset dimension). (HDF5 format spec, "Disk
/// Format: Level 1A1 — Version 1 B-trees", type 1 key.)
const fn chunk_record_key_size(ndims: usize, offset_size: usize) -> usize {
    4 + 4 + ndims * offset_size
}

/// Traverse B-tree v1 type 1 to collect all chunk locations.
///
/// `ndims` is the number of offset dimensions in each key, which equals
/// `chunk_dimensions.len()` from the DataLayout::Chunked message (rank+1).
pub fn collect_chunk_info(
    file_data: &[u8],
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    collect_chunk_info_inner(file_data, btree_address, ndims, offset_size, length_size, 0)
}

/// Depth-tracking core of [`collect_chunk_info`]. The `depth` guard mirrors
/// [`collect_chunk_btree_node_spans_inner`]: a cyclic or pathologically deep
/// internal node in a foreign file errors out instead of recursing until the
/// stack overflows (which would abort the process uncatchably).
fn collect_chunk_info_inner(
    file_data: &[u8],
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    _length_size: u8,
    depth: u32,
) -> Result<Vec<ChunkInfo>, FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let offset = btree_address.to_usize()?;
    let os = offset_size as usize;

    // Parse B-tree v1 header
    let header_size = btree_v1_node_header_size(offset_size);
    if header_size > file_data.len() || offset > file_data.len() - header_size {
        return Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(header_size),
            available: file_data.len(),
        });
    }

    if &file_data[offset..offset + 4] != b"TREE" {
        return Err(FormatError::InvalidBTreeSignature);
    }

    let node_type = file_data[offset + 4];
    if node_type != 1 {
        return Err(FormatError::InvalidBTreeNodeType(node_type));
    }

    let node_level = file_data[offset + 5];
    let entries_used = u16::from_le_bytes([file_data[offset + 6], file_data[offset + 7]]) as usize;

    let mut pos = offset + header_size; // first key, past signature/siblings

    let key_size = chunk_record_key_size(ndims, os);

    if node_level == 0 {
        // Leaf node: keys and children interleaved
        // key[0], child[0], key[1], child[1], ..., key[N-1], child[N-1], key[N]
        let needed = entries_used * (key_size + os) + key_size;
        if needed > file_data.len() || pos > file_data.len() - needed {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(needed),
                available: file_data.len(),
            });
        }

        let mut chunks = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            // Parse key
            let chunk_size = u32::from_le_bytes([
                file_data[pos],
                file_data[pos + 1],
                file_data[pos + 2],
                file_data[pos + 3],
            ]);
            let filter_mask = u32::from_le_bytes([
                file_data[pos + 4],
                file_data[pos + 5],
                file_data[pos + 6],
                file_data[pos + 7],
            ]);
            let mut offsets = Vec::with_capacity(ndims);
            let mut kp = pos + 8;
            for _ in 0..ndims {
                offsets.push(read_offset(file_data, kp, offset_size)?);
                kp += os;
            }
            pos += key_size;

            // Parse child address
            let address = read_offset(file_data, pos, offset_size)?;
            pos += os;

            chunks.push(ChunkInfo {
                chunk_size,
                filter_mask,
                offsets,
                address,
            });
        }
        // Skip final key
        Ok(chunks)
    } else {
        // Internal node: recurse into children
        let needed = entries_used * (key_size + os) + key_size;
        if needed > file_data.len() || pos > file_data.len() - needed {
            return Err(FormatError::UnexpectedEof {
                expected: pos.saturating_add(needed),
                available: file_data.len(),
            });
        }

        let mut child_addrs = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            pos += key_size; // skip key
            let child_addr = read_offset(file_data, pos, offset_size)?;
            child_addrs.push(child_addr);
            pos += os;
        }

        let mut all_chunks = Vec::new();
        for child_addr in child_addrs {
            let child_chunks = collect_chunk_info_inner(
                file_data,
                child_addr,
                ndims,
                offset_size,
                _length_size,
                depth + 1,
            )?;
            all_chunks.extend(child_chunks);
        }
        Ok(all_chunks)
    }
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
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let mut out = Vec::new();
    collect_chunk_btree_node_spans_inner(source, btree_address, ndims, offset_size, 0, &mut out)?;
    Ok(out)
}

fn collect_chunk_btree_node_spans_inner<S: Source + ?Sized>(
    source: &S,
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    depth: u32,
    out: &mut Vec<(u64, u64)>,
) -> Result<(), FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let os = offset_size as usize;
    let header_size = btree_v1_node_header_size(offset_size);

    // Node header: signature(4) + type(1) + level(1) + entries_used(2) + 2 siblings.
    let header = source.read_metadata_at(btree_address, header_size)?;
    if &header[0..4] != b"TREE" {
        return Err(FormatError::InvalidBTreeSignature);
    }
    let node_type = header[4];
    if node_type != 1 {
        return Err(FormatError::InvalidBTreeNodeType(node_type));
    }
    let node_level = header[5];
    let entries_used = u16::from_le_bytes([header[6], header[7]]) as usize;
    let key_size = chunk_record_key_size(ndims, os);

    // The reader-consumed body: `entries_used` (key, child) pairs and a trailing
    // key. This is the conservative node extent (see the doc comment).
    let body_len = entries_used
        .checked_mul(key_size + os)
        .and_then(|b| b.checked_add(key_size))
        .ok_or(FormatError::OffsetOverflow {
            offset: entries_used as u64,
            length: (key_size + os) as u64,
        })?;
    let node_len = header_size + body_len;
    let body_addr =
        btree_address
            .checked_add(header_size as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: btree_address,
                length: header_size as u64,
            })?;
    // Reading the body is also the bounds check: a node claiming to run past
    // end-of-file fails here rather than being recorded as reclaimable.
    let body = source.read_metadata_at(body_addr, body_len)?;
    out.push((btree_address, node_len as u64));

    // Internal nodes (level > 0) hold child node addresses, not chunk records;
    // recurse so their nodes are reclaimed too.
    if node_level != 0 {
        let mut pos = 0usize;
        for _ in 0..entries_used {
            pos += key_size; // skip the key
            let child_addr = read_offset(&body, pos, offset_size)?;
            pos += os;
            collect_chunk_btree_node_spans_inner(
                source,
                child_addr,
                ndims,
                offset_size,
                depth + 1,
                out,
            )?;
        }
    }
    Ok(())
}

/// Traverse B-tree v1 type 1 from a [`Source`], reading each node on demand.
///
/// The streaming counterpart of [`collect_chunk_info`]: it reads the node header
/// and the key/child region as two bounded windows via `read_at`, recursing into
/// child nodes by reading their regions, so the index can be walked without a
/// whole-file buffer.
pub fn collect_chunk_info_from_source<S: Source + ?Sized>(
    source: &S,
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    collect_chunk_info_from_source_inner(source, btree_address, ndims, offset_size, length_size, 0)
}

/// Depth-tracking core of [`collect_chunk_info_from_source`]; see
/// [`collect_chunk_info_inner`] for why the bound is required.
fn collect_chunk_info_from_source_inner<S: Source + ?Sized>(
    source: &S,
    btree_address: u64,
    ndims: usize,
    offset_size: u8,
    _length_size: u8,
    depth: u32,
) -> Result<Vec<ChunkInfo>, FormatError> {
    if depth > MAX_CHUNK_BTREE_DEPTH {
        return Err(FormatError::ChunkedReadError(
            "chunk B-tree nested too deeply".into(),
        ));
    }
    let os = offset_size as usize;
    let header_size = btree_v1_node_header_size(offset_size);

    // Node header: signature(4) + type(1) + level(1) + entries_used(2) + 2 siblings.
    let header = source.read_metadata_at(btree_address, header_size)?;
    if &header[0..4] != b"TREE" {
        return Err(FormatError::InvalidBTreeSignature);
    }
    let node_type = header[4];
    if node_type != 1 {
        return Err(FormatError::InvalidBTreeNodeType(node_type));
    }
    let node_level = header[5];
    let entries_used = u16::from_le_bytes([header[6], header[7]]) as usize;

    // Key/child region begins right after the header (siblings already included).
    let key_size = chunk_record_key_size(ndims, os);
    let needed = entries_used * (key_size + os) + key_size;
    let body_addr =
        btree_address
            .checked_add(header_size as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: btree_address,
                length: header_size as u64,
            })?;
    let body = source.read_metadata_at(body_addr, needed)?;
    let mut pos = 0usize;

    if node_level == 0 {
        let mut chunks = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            let chunk_size =
                u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]);
            let filter_mask =
                u32::from_le_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]]);
            let mut offsets = Vec::with_capacity(ndims);
            let mut kp = pos + 8;
            for _ in 0..ndims {
                offsets.push(read_offset(&body, kp, offset_size)?);
                kp += os;
            }
            pos += key_size;
            let address = read_offset(&body, pos, offset_size)?;
            pos += os;
            chunks.push(ChunkInfo {
                chunk_size,
                filter_mask,
                offsets,
                address,
            });
        }
        Ok(chunks)
    } else {
        let mut child_addrs = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            pos += key_size; // skip key
            child_addrs.push(read_offset(&body, pos, offset_size)?);
            pos += os;
        }
        let mut all_chunks = Vec::new();
        for child_addr in child_addrs {
            all_chunks.extend(collect_chunk_info_from_source_inner(
                source,
                child_addr,
                ndims,
                offset_size,
                _length_size,
                depth + 1,
            )?);
        }
        Ok(all_chunks)
    }
}

/// Generate ChunkInfo entries for an implicit index (v4 index type 2).
///
/// Chunks are stored contiguously starting at `data_address`. No stored index;
/// addresses are computed from the chunk position.
pub fn generate_implicit_chunks(
    data_address: u64,
    dataset_dims: &[u64],
    chunk_dimensions: &[u32],
    element_size: u32,
) -> Vec<ChunkInfo> {
    let rank = chunk_dimensions.len();
    let chunk_byte_size: u64 =
        chunk_dimensions.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;

    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        let ds = dataset_dims[d];
        let ch = chunk_dimensions[d] as u64;
        num_chunks_per_dim.push(ds.div_ceil(ch));
    }
    let total_chunks: u64 = num_chunks_per_dim.iter().product();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "with_capacity hint only; total_chunks is bounded by the dataset's chunk \
                  grid and a truncated hint merely under-reserves (the Vec still grows)"
    )]
    let mut chunks = Vec::with_capacity(total_chunks as usize);
    for linear_idx in 0..total_chunks {
        let mut offsets = vec![0u64; rank];
        let mut remaining = linear_idx;
        for d in (0..rank).rev() {
            let nchunks = num_chunks_per_dim[d];
            let chunk_idx = remaining % nchunks;
            remaining /= nchunks;
            offsets[d] = chunk_idx * chunk_dimensions[d] as u64;
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk byte size is stored in the 32-bit ChunkInfo.chunk_size field \
                      (HDF5 caps a chunk at 4 GiB)"
        )]
        chunks.push(ChunkInfo {
            chunk_size: chunk_byte_size as u32,
            filter_mask: 0,
            offsets,
            address: data_address + linear_idx * chunk_byte_size,
        });
    }

    chunks
}

/// The per-dimension geometry the chunked readers share: the data rank (the chunk
/// layout's dimensions minus the trailing element-size dim), the chunk dimensions,
/// and the dataset dimensions, all as `usize`. `chunk_dimensions` are `u32` and
/// widen; the dataspace dims are `u64` and narrow with a checked conversion. Errors
/// when the layout carries no dimensions or its rank disagrees with the dataspace.
fn chunked_dims(
    chunk_dimensions: &[u32],
    dataspace: &Dataspace,
) -> Result<(usize, Vec<usize>, Vec<usize>), FormatError> {
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let chunk_dims: Vec<usize> = chunk_dimensions[..rank]
        .iter()
        .map(|&d| d as usize)
        .collect();
    let ds_dims: Vec<usize> = dataspace
        .dimensions
        .iter()
        .map(|&d| d.to_usize())
        .collect::<Result<_, _>>()?;
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

/// The largest logical (unfiltered) byte size a single chunk may declare.
///
/// HDF5 stores a chunk's on-disk size in a 32-bit field, so a chunk cannot
/// occupy more than `u32::MAX` bytes on disk. An unfiltered chunk's logical size
/// equals its on-disk size, and the reader also narrows the computed chunk byte
/// size to `u32` on the single-chunk and implicit index paths, so a declared
/// chunk larger than this is not representable in the format regardless.
const MAX_CHUNK_LOGICAL_BYTES: u64 = u32::MAX as u64;

/// Reject a chunk geometry whose logical byte size — `product(chunk_dims) *
/// elem_size` — is impossible for the format, before it is used to size a
/// full-extent output allocation.
///
/// The chunk edge lengths come from the data-layout message and `elem_size` from
/// the datatype message, both attacker-controlled in an untrusted file. A crafted
/// datatype size (for example a fixed-length string element of billions of bytes)
/// or chunk extent otherwise multiplies into a multi-gigabyte `vec![0u8; …]` for
/// the assembled dataset — an out-of-memory abort driven by a tiny file — even
/// though the on-disk chunk data is small. Bounding the *per-chunk* logical size
/// (rather than the whole dataset) refuses that case while leaving a genuinely
/// large dataset, which is large through its element count across many
/// normally-sized chunks, entirely readable.
fn ensure_chunk_bytes_representable(
    chunk_dims: &[usize],
    elem_size: NonZeroUsize,
) -> Result<(), FormatError> {
    let logical = chunk_dims
        .iter()
        .try_fold(elem_size.get() as u64, |acc, &d| acc.checked_mul(d as u64));
    match logical {
        Some(bytes) if bytes <= MAX_CHUNK_LOGICAL_BYTES => Ok(()),
        _ => Err(FormatError::InvalidChunkGeometry(
            "chunk logical byte size exceeds the 4 GiB format limit",
        )),
    }
}

/// Resolve a chunked layout's index address, or produce the buffer an
/// unallocated dataset reads as.
///
/// A chunked dataset's index is allocated lazily — the reference C library
/// leaves the layout message's address undefined until the first chunk is
/// written — so a dataset that has been created but never written names no
/// index at all. That is not an error and not an empty answer: it reads as one
/// `fill` element per element of its dataspace, which for a zero-element
/// dataset is the empty buffer and for any other is a whole materialized
/// dataset. Both fall out of the same expression.
///
/// `Ok(Err(buffer))` is the "nothing allocated, here is what it reads as" case
/// and `Ok(Ok(addr))` the ordinary one; the outer `Result` carries a genuine
/// failure (an element count that cannot be addressed on this target).
fn chunk_index_address(
    addr: Option<u64>,
    spec: RawReadSpec<'_>,
) -> Result<Result<u64, Vec<u8>>, FormatError> {
    if let Some(a) = addr {
        return Ok(Ok(a));
    }
    let RawReadSpec {
        dataspace,
        datatype,
        fill,
        ..
    } = spec;
    let elem_size = datatype.element_size_usize()?;
    let total = dataspace
        .num_elements()
        .to_usize()?
        .checked_mul(elem_size.get())
        .ok_or(FormatError::OffsetOverflow {
            offset: dataspace.num_elements(),
            length: elem_size.get() as u64,
        })?;
    Ok(Err(fill.buffer(total)?))
}

/// Read a chunked dataset from a [`Source`], reading the chunk index and
/// each chunk's bytes on demand via `read_at`.
///
/// Streaming counterpart of [`read_chunked_data_cached`], with the same
/// chunk-index coverage: the B-tree v1 (v3) index and the v4 single-chunk, implicit,
/// Fixed-Array, and Extensible-Array indexes (index types 1-4). A v4 index
/// type 5 (version-2 B-tree) is not supported, matching the buffered reader.
/// The decompression is sequential.
pub fn read_chunked_data_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let RawReadSpec {
        layout,
        dataspace,
        datatype,
        pipeline,
        fill,
    } = spec;
    let (
        chunk_dimensions,
        version,
        chunk_index_type,
        addr_opt,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (
            chunk_dimensions,
            *version,
            *chunk_index_type,
            *btree_address,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        }
    };

    let addr = match chunk_index_address(addr_opt, spec)? {
        Ok(addr) => addr,
        Err(unallocated) => return Ok(unallocated),
    };

    let elem_size = datatype.element_size_usize()?;
    let (rank, chunk_dims, ds_dims) = chunked_dims(chunk_dimensions, dataspace)?;
    ensure_chunk_bytes_representable(&chunk_dims, elem_size)?;

    let chunks = collect_chunks_for_layout_from_source(
        source,
        version,
        chunk_index_type,
        addr,
        single_filtered_size,
        single_filter_mask,
        chunk_dimensions,
        dataspace,
        elem_size,
        offset_size,
        length_size,
    )?;

    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype)?;
    let decompressed_chunks = decompress_all_chunks_from_source(source, &chunks, pipeline, ctx)?;
    let num_elements = dataspace.num_elements();
    let total_bytes = num_elements
        .to_usize()?
        .checked_mul(elem_size.get())
        .ok_or(FormatError::OffsetOverflow {
            offset: num_elements,
            length: elem_size.get() as u64,
        })?;
    assemble_chunks(
        &chunks,
        &decompressed_chunks,
        rank,
        &chunk_dims,
        &ds_dims,
        elem_size,
        total_bytes,
        fill,
    )
}

/// Put a dataset's chunks in address order, which is the order a coalescing
/// reader has to be asked for them in (see [`crate::chunk_span`]).
///
/// Neither source of a chunk list supplies it: walking the index yields index
/// order, and a cached index yields no order at all. A reader holds one span,
/// so a walk that alternates between two of them re-reads a whole span at every
/// alternation — the read stays correct and the read *volume* multiplies. The
/// sort is free to do here because every chunk of a read scatters into a
/// disjoint part of the output, so their order was never load-bearing.
///
/// Coalescing is not the only thing resting on it. A read fills the chunk cache
/// and then stops offering (see [`CachePass`]), so the visit order also decides
/// *which* chunks a read leaves behind — and "no order at all" above is what
/// makes that an arbitrary subset the next read has no reason to reach first.
/// A loop with no spans to coalesce still owes its chunks this order.
fn sort_chunks_by_address(chunks: &mut [ChunkInfo]) {
    chunks.sort_unstable_by_key(|c| c.address);
}

/// Plan coalesced reads over the chunks a read will actually fetch, or `None`
/// when the layout has nothing to coalesce.
///
/// `wanted` picks the chunks the caller will visit — all of them for a whole
/// read, the overlapping ones for a row window. Chunks the cache already holds
/// are dropped on top of that: the caller skips them, so a span covering one
/// would fetch bytes this read has no use for, which is the guarantee
/// [`ChunkSpanReader`] is built around.
///
/// Planning over a chunk the read then skips anyway is only a missed
/// opportunity, never a wrong answer, so `wanted` may be approximate where the
/// caller's own test is fallible.
fn plan_chunk_spans(
    chunks: &[ChunkInfo],
    rank: usize,
    cache: &ChunkCache,
    wanted: impl Fn(&ChunkInfo) -> bool,
) -> Option<ChunkSpanReader> {
    let cached = cache.decompressed_coords();
    ChunkSpanReader::new(
        chunks
            .iter()
            .filter(|c| wanted(c))
            .filter(|c| {
                let coord = &c.offsets[..rank.min(c.offsets.len())];
                !cached.iter().any(|held| held.as_slice() == coord)
            })
            .map(|c| (c.address, c.chunk_size)),
    )
}

/// Read the raw element bytes of the row window `[row_start, row_start + num_rows)`
/// — a range along the leading dimension — decoding only the chunks it overlaps.
///
/// The windowed counterpart of [`read_chunked_data_from_source`]: peak memory is
/// one window plus one chunk, not the whole dataset. `cache` holds the parsed index
/// and decompressed chunks, so successive windows on the same handle walk the index
/// once and reuse boundary chunks.
///
/// A chunk that spans the dataset's full inner extent (`chunk_dims[i] == ds_dims[i]`
/// for `i >= 1` — every 1-D and frame-per-chunk layout) holds each row contiguously,
/// so its contribution is a plain row band copied in one move. A chunk of an
/// inner-split grid scatters into the window through the same N-D kernel the whole
/// read uses ([`place_chunk`]), aimed at a window-shaped output. Either way only the
/// chunks overlapping the window are decoded, and `num_rows * row_elems` elements
/// are returned. Returns `Ok(None)` only for a rank-0 (scalar) chunked layout — a
/// crafted-file corner with no leading dimension to window — so the caller falls
/// back to a whole read. The caller clamps the window to the dataset.
///
/// `pass` decides which chunks the cache keeps, and the answer differs by caller.
/// A lone window wants [`CachePass::LRU`], so that what it retains is the chunks
/// it *finished* on: its successor is the adjacent window, and the chunk they
/// share is that last one. A window that is one step of a sweep across the whole
/// dataset wants a real pass, shared by every step — the sweep never asks for a
/// chunk twice, so offering each one to a cache already full is a copy and an
/// eviction with no reader on the other side. See [`CachePass`].
pub(crate) fn read_chunked_rows_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
    pass: CachePass,
    row_start: u64,
    num_rows: u64,
) -> Result<Option<Vec<u8>>, FormatError> {
    let RawReadSpec {
        layout,
        dataspace,
        datatype,
        pipeline,
        fill,
    } = spec;
    let DataLayout::Chunked {
        chunk_dimensions,
        btree_address,
        version,
        chunk_index_type,
        single_chunk_filtered_size,
        single_chunk_filter_mask,
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "expected chunked layout".into(),
        ));
    };

    let elem_size = datatype.element_size_usize()?;
    let (rank, chunk_dims, ds_dims) = chunked_dims(chunk_dimensions, dataspace)?;
    ensure_chunk_bytes_representable(&chunk_dims, elem_size)?;

    // A rank-0 (scalar) chunked layout has no leading dimension to window; fall
    // back to the whole-dataset reader, which handles rank 0. Only a crafted
    // file reaches this (valid chunked layouts are rank >= 1), but the reader
    // parses untrusted input.
    if rank == 0 {
        return Ok(None);
    }

    // When every chunk spans the dataset's full inner extent, each row is
    // contiguous in its chunk and a window is one row band per chunk; an
    // inner-split chunk grid instead scatters each chunk through the N-D kernel.
    let full_inner = (1..rank).all(|i| chunk_dims[i] == ds_dims[i]);

    // Elements per row (product of the inner dims; 1 when 1-D). Checked so a
    // crafted dataspace whose inner dims overflow `usize` errors instead of
    // panicking (debug) or wrapping (release).
    let row_elems: usize = ds_dims.iter().skip(1).try_fold(1usize, |acc, &d| {
        acc.checked_mul(d).ok_or(FormatError::OffsetOverflow {
            offset: acc as u64,
            length: d as u64,
        })
    })?;
    let row_bytes = row_elems
        .checked_mul(elem_size.get())
        .ok_or(FormatError::OffsetOverflow {
            offset: row_elems as u64,
            length: elem_size.get() as u64,
        })?;

    let out_rows = num_rows.to_usize()?;
    let total_bytes = out_rows
        .checked_mul(row_bytes)
        .ok_or(FormatError::OffsetOverflow {
            offset: out_rows as u64,
            length: row_bytes as u64,
        })?;
    // Uncovered bytes are unallocated storage; see `assemble_chunks`.
    let mut output = fill.buffer(total_bytes)?;
    if total_bytes == 0 {
        return Ok(Some(output));
    }

    // Scatter geometry for the inner-split path: the output is a dataset whose
    // leading dimension is the window (`[out_rows, inner dims…]`), row-major.
    // Checked before any I/O: crafted chunk dimensions (u32 each) can overflow
    // the stride product where a real chunk's byte size cannot. For a full-inner
    // chunk the product equals `row_elems`, already checked above.
    let mut win_dims = ds_dims.clone();
    win_dims[0] = out_rows;
    let win_strides = row_major_strides(&win_dims)?;
    let chunk_strides = row_major_strides(&chunk_dims)?;

    // Unallocated chunk index: no storage exists, so every element of the window
    // is fill. `output` already holds exactly that (it was built from the
    // pattern), so the window is finished before any chunk is looked at — the
    // same answer the whole-dataset readers give through `chunk_index_address`,
    // which is what keeps `read_raw_rows` and `read_raw` agreeing over a dataset
    // that was created and never written.
    let Some(addr) = *btree_address else {
        return Ok(Some(output));
    };

    let row_lo = row_start.to_usize()?;
    let row_hi = row_lo.saturating_add(out_rows); // exclusive; caller clamped to the dataset
    let cd0 = chunk_dims[0];

    // A chunk contributes to the window when its rows meet `[row_lo, row_hi)`.
    // Every list this function builds and every walk over one applies this test,
    // so it is written once.
    let overlaps = |c0: usize| c0 < row_hi && c0.saturating_add(cd0) > row_lo;
    let chunk_overlaps = |chunk: &ChunkInfo| {
        match chunk.offsets.first().copied().unwrap_or(0).to_usize() {
            Ok(c0) => overlaps(c0),
            // A leading offset too large for `usize` — a 32-bit target reading a
            // crafted file. Kept rather than dropped: the walk below converts it
            // again and reports the failure, and a filter that swallowed it here
            // would turn that error into silence.
            Err(_) => true,
        }
    };

    // Walk the index once, then reuse it for later windows on this handle —
    // keeping only this window's chunks, since each `ChunkInfo` taken out of the
    // index owns a coordinate `Vec` and the rest would be allocated and dropped
    // unread. Sweeping a dataset window by window makes that a product rather
    // than a sum (issue #289).
    let mut chunks = if let Some(chunks) = cache.indexed_chunks_matching(chunk_overlaps) {
        chunks
    } else {
        let chunks = collect_chunks_for_layout_from_source(
            source,
            *version,
            *chunk_index_type,
            addr,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
            chunk_dimensions,
            dataspace,
            elem_size,
            offset_size,
            length_size,
        )?;
        cache.populate_index(&chunks, rank);
        chunks.into_iter().filter(chunk_overlaps).collect()
    };
    sort_chunks_by_address(&mut chunks);

    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype)?;

    // Coalesce over the window's own chunks. A plan built over the dataset's
    // whole chunk list would put the rows on either side of the window inside a
    // span and read them for nothing — an eight-row window measured 32 KB where
    // it needed 64 bytes.
    let mut spans = plan_chunk_spans(&chunks, rank, cache, chunk_overlaps);

    // One decoder for the whole window; see `FilterScratch`.
    let mut scratch = FilterScratch::new();

    for chunk in &chunks {
        let c0 = chunk.offsets.first().copied().unwrap_or(0).to_usize()?;
        // This chunk's exclusive leading-dim end. The saturating add/mul here and
        // below keep a crafted huge chunk offset or dimension from overflowing
        // `usize` on 32-bit (a panic in debug, a silently wrong window in
        // release): a saturated span still compares and clamps correctly, and
        // `row_band_copy` bounds the copy to the bytes actually present. `dst` and
        // `band` need no guard — both are bounded by the already-checked
        // `total_bytes` allocation.
        let c_end = c0.saturating_add(cd0);
        if !overlaps(c0) {
            continue; // no overlap with the window
        }

        // Rows of this chunk that fall in the window.
        let lo = c0.max(row_lo);
        let hi = c_end.min(row_hi);

        // Inner-split scatter parameters. `place_chunk` takes non-negative
        // offsets, so a chunk straddling the window start is instead entered
        // `skip` leading slabs in (dim 0 is outermost, so one slab is
        // `chunk_strides[0]` elements), and its leading dim becomes exactly
        // the `hi - lo` slabs the window keeps, which also clips a chunk
        // straddling the window end. A saturated `advance` (crafted dims) makes
        // `bytes.get(advance..)` come up empty and the chunk copies nothing —
        // the kernel's clamping discipline for short chunks.
        let scatter: Option<(Vec<usize>, Vec<usize>, usize)> = if full_inner {
            None
        } else {
            let mut offsets = Vec::with_capacity(rank);
            for &o in chunk.offsets.iter().take(rank) {
                offsets.push(o.to_usize()?);
            }
            let skip = lo - c0;
            offsets[0] = lo - row_lo;
            let mut dims = chunk_dims.clone();
            dims[0] = hi - lo;
            let advance = skip
                .saturating_mul(chunk_strides[0])
                .saturating_mul(elem_size.get());
            Some((offsets, dims, advance))
        };

        // Byte offsets of the contiguous row band (full-inner path).
        let src = (lo - c0).saturating_mul(row_bytes);
        let dst = (lo - row_lo) * row_bytes;
        let band = (hi - lo) * row_bytes;

        let copy = |output: &mut [u8], bytes: &[u8]| match &scatter {
            None => row_band_copy(output, dst, bytes, src, band, elem_size),
            Some((offsets, dims, advance)) => place_chunk(
                bytes.get(*advance..).unwrap_or(&[]),
                output,
                offsets,
                dims,
                &win_dims,
                &win_strides,
                &chunk_strides,
                elem_size,
                rank,
            ),
        };

        let coord = spatial_coord(chunk, rank);
        let hit = cache.with_decompressed(pass, coord, |bytes| copy(&mut output, bytes));
        if hit.is_some() {
            continue;
        }

        let len = chunk.chunk_size as usize;
        let stored = match spans.as_mut() {
            Some(sp) => Cow::Borrowed(sp.chunk_bytes(source, chunk.address, len)?),
            None => Cow::Owned(source.read_exact_at(chunk.address, len)?),
        };
        match pipeline {
            Some(pl) => {
                let dec = decompress_chunk_with(&mut scratch, &stored, pl, ctx, chunk.filter_mask)?;
                copy(&mut output, &dec);
                cache.put_decompressed(pass, coord, dec);
            }
            None => {
                copy(&mut output, &stored);
                cache.put_decompressed_slice(pass, coord, &stored);
            }
        }
    }

    Ok(Some(output))
}

/// Copy one chunk's overlapping row band into the window buffer at `dst`, clamped
/// to the bytes present on each side (element-aligned). An edge or short chunk
/// copies only its valid prefix and never reads or writes out of bounds — the same
/// discipline as [`copy_chunk_to_output`].
fn row_band_copy(
    output: &mut [u8],
    dst: usize,
    chunk: &[u8],
    src: usize,
    band: usize,
    elem_size: NonZeroUsize,
) {
    let mut len = band
        .min(chunk.len().saturating_sub(src))
        .min(output.len().saturating_sub(dst));
    len -= len % elem_size;
    if len > 0 {
        output[dst..dst + len].copy_from_slice(&chunk[src..src + len]);
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

/// Read a chunked dataset from a [`Source`] with parsed-index and
/// decompressed-chunk caching.
///
/// This is the streaming counterpart of [`read_chunked_data_cached`].
pub fn read_chunked_data_cached_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    let RawReadSpec {
        layout,
        dataspace,
        datatype,
        pipeline,
        fill,
    } = spec;
    let (
        chunk_dimensions,
        version,
        chunk_index_type,
        addr_opt,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (
            chunk_dimensions,
            *version,
            *chunk_index_type,
            *btree_address,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        }
    };

    let addr = match chunk_index_address(addr_opt, spec)? {
        Ok(addr) => addr,
        Err(unallocated) => return Ok(unallocated),
    };

    let elem_size = datatype.element_size_usize()?;
    let (rank, chunk_dims, ds_dims) = chunked_dims(chunk_dimensions, dataspace)?;
    ensure_chunk_bytes_representable(&chunk_dims, elem_size)?;

    let mut chunks = if let Some(chunks) = cache.all_indexed_chunks() {
        chunks
    } else {
        let chunks = collect_chunks_for_layout_from_source(
            source,
            version,
            chunk_index_type,
            addr,
            single_filtered_size,
            single_filter_mask,
            chunk_dimensions,
            dataspace,
            elem_size,
            offset_size,
            length_size,
        )?;
        cache.populate_index(&chunks, rank);
        chunks
    };
    sort_chunks_by_address(&mut chunks);

    let total_elements = dataspace.num_elements().to_usize()?;
    let total_bytes =
        total_elements
            .checked_mul(elem_size.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: total_elements as u64,
                length: elem_size.get() as u64,
            })?;
    // Uncovered bytes are unallocated storage; see `assemble_chunks`.
    let mut output = fill.buffer(total_bytes)?;

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }

    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype)?;

    // Every chunk of the dataset is wanted here; what the plan leaves out is
    // whatever the chunk cache already holds from an earlier read.
    let mut spans = plan_chunk_spans(&chunks, rank, cache, |_| true);

    let pass = cache.begin_pass();
    // One decoder for every chunk of the dataset; see `FilterScratch`.
    let mut scratch = FilterScratch::new();
    // Refilled per chunk rather than allocated per chunk. It holds `rank`
    // integers — 8 bytes for a 1-D dataset, invisible beside the chunk it
    // describes — but allocating it here is an allocator round trip for every
    // chunk of every dataset this crate reads (issue #228).
    let mut chunk_offsets: Vec<usize> = Vec::with_capacity(rank);

    for chunk_info in &chunks {
        let coord = spatial_coord(chunk_info, rank);
        chunk_offsets.clear();
        for &o in coord {
            chunk_offsets.push(o.to_usize()?);
        }

        // Scatter straight from the cached chunk under the lock (no copy out).
        let hit = cache.with_decompressed(pass, coord, |bytes| {
            place_chunk(
                bytes,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
        });
        if hit.is_some() {
            continue;
        }

        // Cache miss: fetch the chunk's stored bytes from the source.
        let len = chunk_info.chunk_size.to_usize()?;
        let stored = match spans.as_mut() {
            Some(sp) => Cow::Borrowed(sp.chunk_bytes(source, chunk_info.address, len)?),
            None => Cow::Owned(source.read_exact_at(chunk_info.address, len)?),
        };
        let dec = match pipeline {
            Some(pl) => Cow::Owned(decompress_chunk_with(
                &mut scratch,
                &stored,
                pl,
                ctx,
                chunk_info.filter_mask,
            )?),
            // Unfiltered: a chunk's stored bytes are its data, so they are
            // copied out of the span only if the cache admits them.
            None => stored,
        };
        place_chunk(
            &dec,
            &mut output,
            &chunk_offsets,
            &chunk_dims,
            &ds_dims,
            &ds_strides,
            &chunk_strides,
            elem_size,
            rank,
        );
        // Dropped either way if the cache does not admit it.
        match dec {
            Cow::Owned(bytes) => cache.put_decompressed(pass, coord, bytes), // move
            Cow::Borrowed(bytes) => cache.put_decompressed_slice(pass, coord, bytes),
        }
    }

    Ok(output)
}

/// A chunk's spatial coordinate: its offset vector cut to the dataset's rank.
///
/// Borrowed from the [`ChunkInfo`] rather than copied, because the read loops
/// need one per chunk and a dataset has as many chunks as it likes. The cut is
/// `get(..rank)` rather than an index so a shorter offset vector yields what it
/// has, which is what the `.take(rank)` this replaced did.
fn spatial_coord(chunk_info: &ChunkInfo, rank: usize) -> &[u64] {
    chunk_info
        .offsets
        .get(..rank)
        .unwrap_or(&chunk_info.offsets)
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
    spatial_chunk_dims: &[u32],
    order: GridOrder,
) -> Result<ChunkGrid, FormatError> {
    let chunk_dims: Vec<u64> = spatial_chunk_dims.iter().map(|&d| u64::from(d)).collect();
    ChunkGrid::new(
        &chunk_dims,
        &dataspace.dimensions,
        dataspace.max_dimensions.as_deref(),
        order,
    )
}

pub(crate) fn collect_chunks_for_layout_from_source<S: Source + ?Sized>(
    source: &S,
    version: u8,
    chunk_index_type: Option<u8>,
    addr: u64,
    single_filtered_size: Option<u64>,
    single_filter_mask: Option<u32>,
    chunk_dimensions: &[u32],
    dataspace: &Dataspace,
    elem_size: NonZeroUsize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let ndims = chunk_dimensions.len();
    let rank = ndims
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    match (version, chunk_index_type) {
        (3, _) => collect_chunk_info_from_source(source, addr, ndims, offset_size, length_size),
        (4, Some(1)) => {
            let chunk_byte_size: usize = chunk_dimensions[..rank]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                * elem_size.get();
            let (csize, fmask) = if let Some(fs) = single_filtered_size {
                (u32_from(fs)?, single_filter_mask.unwrap_or(0))
            } else {
                (u32_from(chunk_byte_size as u64)?, 0)
            };
            Ok(vec![ChunkInfo {
                chunk_size: csize,
                filter_mask: fmask,
                offsets: vec![0u64; rank],
                address: addr,
            }])
        }
        (4, Some(2)) => {
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            Ok(generate_implicit_chunks(
                addr,
                &dataspace.dimensions,
                &spatial_chunk_dims,
                u32_from(elem_size.get() as u64)?,
            ))
        }
        (4, Some(3)) => {
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let header =
                FixedArrayHeader::parse_from_source(source, addr, offset_size, length_size)?;
            read_fixed_array_chunks_from_source(
                source,
                &header,
                &index_grid(dataspace, &spatial_chunk_dims, GridOrder::RowMajor)?,
                &spatial_chunk_dims,
                u32_from(elem_size.get() as u64)?,
                offset_size,
                length_size,
            )
        }
        (4, Some(4)) => {
            let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
            let header =
                ExtensibleArrayHeader::parse_from_source(source, addr, offset_size, length_size)?;
            read_extensible_array_chunks_from_source(
                source,
                &header,
                &index_grid(dataspace, &spatial_chunk_dims, GridOrder::UnlimitedFirst)?,
                &spatial_chunk_dims,
                u32_from(elem_size.get() as u64)?,
                offset_size,
                length_size,
            )
        }
        (v, idx) => Err(FormatError::ChunkedReadError(format!(
            "unsupported chunked layout version={v}, index_type={idx:?}"
        ))),
    }
}

/// Enumerate every allocated chunk of a chunked dataset from an in-memory file
/// image, one [`ChunkInfo`] per chunk (file address, on-disk stored size, filter
/// mask, logical offsets). A buffered convenience wrapper over
/// [`collect_chunks_for_layout_from_source`] for callers that hold the whole
/// file as a byte slice (the in-place editor): it accepts a parsed
/// [`DataLayout::Chunked`] and the dataset's [`Dataspace`] and reads through a
/// [`BytesSource`](crate::source::BytesSource).
///
/// Returns an empty vector when the index address is undefined (a chunked
/// dataset with no storage allocated yet). Errors — propagated from the index
/// walkers — for a version-2 B-tree (index type 5) or any unknown index type,
/// which have no walker.
#[cfg(feature = "std")]
pub(crate) fn enumerate_chunks_from_source<S: Source + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<ChunkInfo>, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        btree_address,
        version,
        chunk_index_type,
        single_chunk_filtered_size,
        single_chunk_filter_mask,
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "enumerate_chunks_from_source called on a non-chunked layout".into(),
        ));
    };
    // An undefined index address means no storage is allocated yet.
    let Some(index_addr) = *btree_address else {
        return Ok(Vec::new());
    };
    if chunk_dimensions.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has no dimensions".into(),
        ));
    }
    // `chunk_dimensions` is rank + 1 entries; the last is the element size.
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let elem_size = NonZeroUsize::new(chunk_dimensions[rank] as usize).ok_or_else(|| {
        FormatError::ChunkedReadError("chunked layout has a zero element size".into())
    })?;
    collect_chunks_for_layout_from_source(
        source,
        *version,
        *chunk_index_type,
        index_addr,
        *single_chunk_filtered_size,
        *single_chunk_filter_mask,
        chunk_dimensions,
        dataspace,
        elem_size,
        offset_size,
        length_size,
    )
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
/// single place that maps a chunked layout to the regions it occupies on disk;
/// the in-place editor uses it to reclaim that space on delete (issue #77).
///
/// `layout` must be a [`DataLayout::Chunked`] (anything else is an error).
/// Returns both halves empty when the index address is undefined (an empty or
/// never-written dataset owns no storage). Errors — propagated from the index
/// walkers — mean the layout could not be enumerated exhaustively (a version 2
/// B-tree chunk index, or a malformed structure); the caller then leaves the
/// dataset's bytes in place rather than free a region it cannot fully account
/// for. The spans are not pre-checked for mutual disjointness or file bounds:
/// the editor validates that before handing them to the free list.
///
/// Chunk data spans come from the same index walkers the reader uses
/// ([`collect_chunks_for_layout_from_source`]); index-structure spans come from
/// [`collect_chunk_index_spans`]. `offset_size`/`length_size` are the
/// superblock's address and length widths.
#[cfg(feature = "std")]
pub(crate) fn collect_chunked_storage_spans<S: Source + ?Sized>(
    source: &S,
    layout: &DataLayout,
    dataspace: &Dataspace,
    offset_size: u8,
    length_size: u8,
) -> Result<ChunkedStorageSpans, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        btree_address,
        version,
        chunk_index_type,
        single_chunk_filtered_size,
        single_chunk_filter_mask,
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "collect_chunked_storage_spans called on a non-chunked layout".into(),
        ));
    };
    // An undefined index address means no storage is allocated yet.
    let Some(index_addr) = *btree_address else {
        return Ok(ChunkedStorageSpans {
            data: Vec::new(),
            index: Vec::new(),
        });
    };
    if chunk_dimensions.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "chunked layout has no dimensions".into(),
        ));
    }
    // `chunk_dimensions` is rank + 1 entries; the last is the element size.
    let rank = chunk_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| FormatError::ChunkedReadError("chunked layout has no dimensions".into()))?;
    let elem_size = NonZeroUsize::new(chunk_dimensions[rank] as usize).ok_or_else(|| {
        FormatError::ChunkedReadError("chunked layout has a zero element size".into())
    })?;

    let mut data: Vec<(u64, u64)> = Vec::new();

    // Chunk data blocks (the walkers omit unallocated chunks).
    for ci in collect_chunks_for_layout_from_source(
        source,
        *version,
        *chunk_index_type,
        index_addr,
        *single_chunk_filtered_size,
        *single_chunk_filter_mask,
        chunk_dimensions,
        dataspace,
        elem_size,
        offset_size,
        length_size,
    )? {
        if ci.chunk_size != 0 {
            data.push((ci.address, ci.chunk_size as u64));
        }
    }

    // The chunk index's own structure blocks.
    let index = collect_chunk_index_spans(
        source,
        *version,
        *chunk_index_type,
        index_addr,
        chunk_dimensions.len(),
        offset_size,
        length_size,
    )?;
    Ok(ChunkedStorageSpans { data, index })
}

/// A chunked dataset's on-disk footprint, split by the two kinds of structure it
/// occupies: `data` holds the chunk data blocks and `index` holds the chunk
/// index's own structure blocks. Callers that only need the whole footprint can
/// concatenate the two.
///
/// The split is *structural*, not a page-type classification. On a paged file
/// (`H5F_FSPACE_STRATEGY_PAGE`) both halves normally sit in raw pages, because
/// every writer in this crate emits the index in the same run as the chunk data;
/// a caller reclaiming this space must route by where the bytes actually are, not
/// by which half they came from. See `WriteEngine::chunked_storage_spans`.
#[cfg(feature = "std")]
pub(crate) struct ChunkedStorageSpans {
    pub(crate) data: Vec<(u64, u64)>,
    pub(crate) index: Vec<(u64, u64)>,
}

/// On-disk byte spans `(addr, len)` of a chunked dataset's index *structure*
/// (not its chunk data): the B-tree v1 nodes, fixed-array header and data block,
/// or extensible-array header, index, super, and data blocks, by index type.
/// Single-chunk and implicit indexes have no separate structure (their footprint
/// is the chunk data alone), so they return an empty vector. An index type with
/// no walker (a version 2 B-tree, index type 5) is an error.
///
/// `ndims` is `chunk_dimensions.len()` (rank + 1), as [`collect_chunk_info`]
/// takes. Shared by [`collect_chunked_storage_spans`]; kept separate so the
/// per-type dispatch lives next to the chunk-data dispatch it mirrors.
#[cfg(feature = "std")]
fn collect_chunk_index_spans<S: Source + ?Sized>(
    source: &S,
    version: u8,
    chunk_index_type: Option<u8>,
    index_addr: u64,
    ndims: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    match (version, chunk_index_type) {
        (3, _) => collect_chunk_btree_node_spans(source, index_addr, ndims, offset_size),
        // Single chunk and implicit indexes have no separate index structure.
        (4, Some(1)) | (4, Some(2)) => Ok(Vec::new()),
        (4, Some(3)) => crate::fixed_array::fixed_array_index_spans(
            source,
            index_addr,
            offset_size,
            length_size,
        ),
        (4, Some(4)) => crate::extensible_array::extensible_array_index_spans(
            source,
            index_addr,
            offset_size,
            length_size,
        ),
        (v, idx) => Err(FormatError::ChunkedReadError(format!(
            "chunk index has no reclaim walker: version={v}, index_type={idx:?}"
        ))),
    }
}

/// The on-disk byte spans of a chunked dataset's index *structure* (not its chunk
/// data), from an in-memory file image and a parsed [`DataLayout::Chunked`]. A
/// buffered convenience wrapper over [`collect_chunk_index_spans`], used by the
/// in-place editor to test whether a chunked dataset's index occupies a single
/// contiguous region it can rebuild in place. Returns an empty vector for an
/// undefined index address or a single-chunk / implicit index (no separate
/// structure); errors for an index type with no walker (a version-2 B-tree).
#[cfg(feature = "std")]
pub(crate) fn chunk_index_spans_from_source<S: Source + ?Sized>(
    source: &S,
    layout: &DataLayout,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<(u64, u64)>, FormatError> {
    let DataLayout::Chunked {
        chunk_dimensions,
        btree_address,
        version,
        chunk_index_type,
        ..
    } = layout
    else {
        return Err(FormatError::ChunkedReadError(
            "chunk_index_spans_from_source called on a non-chunked layout".into(),
        ));
    };
    let Some(index_addr) = *btree_address else {
        return Ok(Vec::new());
    };
    collect_chunk_index_spans(
        source,
        *version,
        *chunk_index_type,
        index_addr,
        chunk_dimensions.len(),
        offset_size,
        length_size,
    )
}

/// Scatter decompressed chunks into the dense output buffer. Pure (no file
/// access): shared by the buffered and streaming chunked readers.
fn assemble_chunks(
    chunks: &[ChunkInfo],
    decompressed: &[Vec<u8>],
    rank: usize,
    chunk_dims: &[usize],
    ds_dims: &[usize],
    elem_size: NonZeroUsize,
    total_bytes: usize,
    fill: FillPattern<'_>,
) -> Result<Vec<u8>, FormatError> {
    // Whatever no chunk writes over is storage that was never allocated, so the
    // buffer starts as the fill pattern rather than as zeros. A sparse chunk
    // grid — some chunks written, others never — is the case this covers that a
    // zeroed buffer got silently wrong.
    let mut output = fill.buffer(total_bytes)?;

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }
    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    for (chunk_info, decompressed) in chunks.iter().zip(decompressed.iter()) {
        // B-tree v1 (v3) offsets have rank+1 dims; v4 index offsets have rank dims
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk coordinate offsets are bounded by ds_dims, which already fit usize"
        )]
        let chunk_offsets: Vec<usize> = chunk_info
            .offsets
            .iter()
            .take(rank)
            .map(|&o| o as usize)
            .collect();

        place_chunk(
            decompressed,
            &mut output,
            &chunk_offsets,
            chunk_dims,
            ds_dims,
            &ds_strides,
            &chunk_strides,
            elem_size,
            rank,
        );
    }

    Ok(output)
}

/// Read a chunked dataset with caching support.
///
/// On the first call, scans the chunk index (B-tree / fixed array / etc.) once
/// and populates the cache's hash index.  Subsequent calls skip the index scan
/// entirely.  Decompressed chunk data is also cached with LRU eviction.
pub fn read_chunked_data_cached(
    file_data: &[u8],
    spec: RawReadSpec<'_>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    let RawReadSpec {
        layout,
        dataspace,
        datatype,
        pipeline,
        fill,
    } = spec;
    let (
        chunk_dimensions,
        version,
        chunk_index_type,
        addr_opt,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dimensions,
            btree_address,
            version,
            chunk_index_type,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
        } => (
            chunk_dimensions,
            *version,
            *chunk_index_type,
            *btree_address,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(FormatError::ChunkedReadError(
                "expected chunked layout".into(),
            ));
        }
    };

    let addr = match chunk_index_address(addr_opt, spec)? {
        Ok(addr) => addr,
        Err(unallocated) => return Ok(unallocated),
    };

    // Taken once as a proven-non-zero width; see the buffered reader above.
    let elem_width = datatype.element_size()?;
    let elem_size = nonzero_usize_from(elem_width)?;
    let (rank, chunk_dims, ds_dims) = chunked_dims(chunk_dimensions, dataspace)?;
    ensure_chunk_bytes_representable(&chunk_dims, elem_size)?;
    let ndims = rank + 1; // rank + the trailing element-size dimension

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk byte sizes are encoded into 32-bit chunk-info fields; they stay \
                  well below u32::MAX (HDF5 caps a chunk at 4 GiB)"
    )]
    let mut chunks = if let Some(chunks) = cache.all_indexed_chunks() {
        chunks
    } else {
        let chunks = match (version, chunk_index_type) {
            (3, _) => collect_chunk_info(file_data, addr, ndims, offset_size, length_size)?,
            (4, Some(1)) => {
                let chunk_byte_size: usize = chunk_dims.iter().product::<usize>() * elem_size.get();
                let (csize, fmask) = if let Some(fs) = single_filtered_size {
                    (fs as u32, single_filter_mask.unwrap_or(0))
                } else {
                    (chunk_byte_size as u32, 0)
                };
                vec![ChunkInfo {
                    chunk_size: csize,
                    filter_mask: fmask,
                    offsets: vec![0u64; rank],
                    address: addr,
                }]
            }
            (4, Some(2)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                generate_implicit_chunks(
                    addr,
                    &dataspace.dimensions,
                    &spatial_chunk_dims,
                    elem_width.get(),
                )
            }
            (4, Some(3)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header =
                    FixedArrayHeader::parse(file_data, addr.to_usize()?, offset_size, length_size)?;
                read_fixed_array_chunks(
                    file_data,
                    &header,
                    &index_grid(dataspace, &spatial_chunk_dims, GridOrder::RowMajor)?,
                    &spatial_chunk_dims,
                    elem_width.get(),
                    offset_size,
                    length_size,
                )?
            }
            (4, Some(4)) => {
                let spatial_chunk_dims: Vec<u32> = chunk_dimensions[..rank].to_vec();
                let header = ExtensibleArrayHeader::parse(
                    file_data,
                    addr.to_usize()?,
                    offset_size,
                    length_size,
                )?;
                read_extensible_array_chunks(
                    file_data,
                    &header,
                    &index_grid(dataspace, &spatial_chunk_dims, GridOrder::UnlimitedFirst)?,
                    &spatial_chunk_dims,
                    elem_width.get(),
                    offset_size,
                    length_size,
                )?
            }
            (v, idx) => {
                return Err(FormatError::ChunkedReadError(format!(
                    "unsupported chunked layout version={v}, index_type={idx:?}"
                )));
            }
        };
        cache.populate_index(&chunks, rank);
        chunks
    };
    // This loop reads from a byte slice and has no spans to coalesce, but it owes
    // its chunks an order all the same; see `sort_chunks_by_address`.
    sort_chunks_by_address(&mut chunks);

    // Assemble output
    let total_elements = dataspace.num_elements().to_usize()?;
    let total_bytes = total_elements * elem_size.get();
    // Uncovered bytes are unallocated storage; see `assemble_chunks`.
    let mut output = fill.buffer(total_bytes)?;

    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        ds_strides[i] = ds_strides[i + 1] * ds_dims[i + 1];
    }

    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1];
    }

    let chunk_dims_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let ctx = ChunkContext::from_datatype(&chunk_dims_u64, datatype)?;

    let pass = cache.begin_pass();
    // One decoder for every chunk of the dataset; see `FilterScratch`.
    let mut scratch = FilterScratch::new();
    // Refilled per chunk rather than allocated per chunk; see the counterpart in
    // `read_chunked_data_cached_from_source`.
    let mut chunk_offsets: Vec<usize> = Vec::with_capacity(rank);

    for chunk_info in &chunks {
        let coord = spatial_coord(chunk_info, rank);

        chunk_offsets.clear();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk coordinate offsets are bounded by ds_dims, which already fit usize"
        )]
        chunk_offsets.extend(coord.iter().map(|&o| o as usize));

        // Scatter straight from the cached chunk under the lock (no copy out).
        let hit = cache.with_decompressed(pass, coord, |bytes| {
            place_chunk(
                bytes,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
        });
        if hit.is_some() {
            continue;
        }

        // Cache miss: read the chunk's bytes from the file.
        let r = slice_range(chunk_info.address, u64::from(chunk_info.chunk_size))?;
        if r.end > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: r.end,
                available: file_data.len(),
            });
        }
        let raw_chunk = &file_data[r];
        if let Some(pl) = pipeline {
            let dec =
                decompress_chunk_with(&mut scratch, raw_chunk, pl, ctx, chunk_info.filter_mask)?;
            place_chunk(
                &dec,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
            cache.put_decompressed(pass, coord, dec); // move; dropped if not admitted
        } else {
            // No pipeline: scatter directly from the file buffer, and copy into
            // the cache only if it would actually be retained.
            place_chunk(
                raw_chunk,
                &mut output,
                &chunk_offsets,
                &chunk_dims,
                &ds_dims,
                &ds_strides,
                &chunk_strides,
                elem_size,
                rank,
            );
            cache.put_decompressed_slice(pass, coord, raw_chunk);
        }
    }

    Ok(output)
}

/// Place one decompressed chunk into the dense output buffer, handling the
/// scalar (`rank == 0`) case and delegating the N-D case to the row-copy kernel.
/// Shared by the buffered, cached, and streaming chunked readers so they all use
/// the same scatter logic.
#[allow(clippy::too_many_arguments)]
fn place_chunk(
    chunk_data: &[u8],
    output: &mut [u8],
    chunk_offsets: &[usize],
    chunk_dims: &[usize],
    ds_dims: &[usize],
    ds_strides: &[usize],
    chunk_strides: &[usize],
    elem_size: NonZeroUsize,
    rank: usize,
) {
    if rank == 0 {
        let copy_len = chunk_data.len().min(output.len());
        output[..copy_len].copy_from_slice(&chunk_data[..copy_len]);
    } else {
        copy_chunk_to_output(
            chunk_data,
            output,
            chunk_offsets,
            chunk_dims,
            ds_dims,
            ds_strides,
            chunk_strides,
            elem_size,
            rank,
        );
    }
}

/// Copy chunk data into the output buffer at the correct N-D position.
///
/// The innermost dimension is contiguous in both the chunk and the dataset (both
/// have stride 1 there in a row-major layout), so each in-bounds row of the
/// chunk is moved with a single `copy_from_slice` instead of one tiny copy per
/// element. Only the outer `rank - 1` dimensions are walked (with an odometer),
/// which removes the per-element flat→N-D coordinate division/modulo that
/// dominated the old kernel. Edge chunks that overhang the dataset are clamped in
/// the innermost dimension (a shortened row) and skipped per-row in the outer
/// dimensions (a row whose outer global coordinate is past the dataset bound),
/// reproducing the old per-element out-of-bounds skip exactly. The per-row length
/// is also clamped to the bytes actually available on both sides, so a malformed
/// (short) chunk copies only its valid prefix and is never an out-of-bounds
/// access — matching the old per-element guard's silent-skip behavior.
#[allow(clippy::too_many_arguments)]
fn copy_chunk_to_output(
    chunk_data: &[u8],
    output: &mut [u8],
    chunk_offsets: &[usize],
    chunk_dims: &[usize],
    ds_dims: &[usize],
    ds_strides: &[usize],
    chunk_strides: &[usize],
    elem_size: NonZeroUsize,
    rank: usize,
) {
    debug_assert!(rank >= 1, "rank == 0 is handled by the callers");
    // Row contiguity (the whole optimization) relies on a unit innermost stride
    // in both layouts, which the row-major stride construction guarantees.
    debug_assert_eq!(chunk_strides[rank - 1], 1);
    debug_assert_eq!(ds_strides[rank - 1], 1);

    let inner = rank - 1;

    // In-bounds run length along the contiguous innermost dimension: the chunk
    // is stored at full size but may hang over the dataset's edge.
    let inner_row_len = chunk_dims[inner].min(ds_dims[inner].saturating_sub(chunk_offsets[inner]));
    if inner_row_len == 0 {
        return; // the chunk lies entirely past the inner edge
    }
    let row_bytes = inner_row_len * elem_size.get();
    // Destination offset contributed by the innermost dimension (stride 1).
    let inner_dst = chunk_offsets[inner] * ds_strides[inner];

    // Walk the outer dimensions with an odometer (`coord` is the chunk-local
    // coordinate of each outer dim), copying one contiguous row per step.
    let outer_total: usize = chunk_dims[..inner].iter().product();
    let mut coord = vec![0usize; inner];

    for _ in 0..outer_total {
        let mut chunk_base = 0usize;
        let mut ds_base = inner_dst;
        let mut in_bounds = true;
        for d in 0..inner {
            chunk_base += coord[d] * chunk_strides[d];
            let global = chunk_offsets[d] + coord[d];
            if global >= ds_dims[d] {
                in_bounds = false;
                break;
            }
            ds_base += global * ds_strides[d];
        }

        if in_bounds {
            let src = chunk_base * elem_size.get();
            let dst = ds_base * elem_size.get();
            // Clamp to the bytes available on each side, on an element boundary.
            let mut avail = row_bytes
                .min(chunk_data.len().saturating_sub(src))
                .min(output.len().saturating_sub(dst));
            avail -= avail % elem_size;
            if avail > 0 {
                output[dst..dst + avail].copy_from_slice(&chunk_data[src..src + avail]);
            }
        }

        // Advance the odometer over the outer dims (last outer dim varies fastest).
        for d in (0..inner).rev() {
            coord[d] += 1;
            if coord[d] < chunk_dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::nz;

    #[test]
    fn chunk_bytes_within_limit_are_accepted() {
        // A realistic chunk: 256 KiB of f64 elements.
        assert!(ensure_chunk_bytes_representable(&[128, 256], nz(8)).is_ok());
        // A single chunk right at the 4 GiB limit is still representable.
        assert!(ensure_chunk_bytes_representable(&[u32::MAX as usize], nz(1)).is_ok());
    }

    #[test]
    fn chunk_bytes_over_limit_are_refused() {
        // The issue #185 shape: 10 elements of a ~2.86 GB fixed-length string
        // (0xAAAAAAAA bytes) is ~28.6 GB per chunk — impossible for the format.
        let err = ensure_chunk_bytes_representable(&[10], nz(0xAAAA_AAAA)).unwrap_err();
        assert!(matches!(err, FormatError::InvalidChunkGeometry(_)));

        // A geometry whose product overflows u64 is refused, not wrapped.
        assert!(
            ensure_chunk_bytes_representable(&[usize::MAX, usize::MAX], nz(usize::MAX)).is_err()
        );
    }

    /// A chunk grid with a *hole* — some chunks allocated, others never — reads
    /// the hole as the fill value.
    ///
    /// This is the case the change exists for, and the one that used to return
    /// `Ok` with zeros regardless of the fill value. It goes through
    /// `read_chunked_data_from_source` deliberately: that reader is the only one
    /// reaching [`assemble_chunks`], and the unallocated-*index* test below
    /// returns from `chunk_index_address` before ever getting there — so without
    /// a sparse grid, `assemble_chunks`'s prefill has no coverage at all.
    ///
    /// The index holds the first and last of four chunk slots, so the hole is
    /// interior rather than a tail.
    #[test]
    fn a_hole_in_the_chunk_grid_reads_as_fill() {
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
                chunk_size: (chunk_elems * elem) as u32,
                filter_mask: 0,
                offsets: vec![start_elem, 0],
                address: data_offset as u64,
            });
            data_offset += chunk_elems * elem;
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, 2, 8);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_elems as u32, elem as u32],
            btree_address: Some(btree_addr as u64),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
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
            RawReadSpec {
                layout: &layout,
                dataspace: &dataspace,
                datatype: &make_f64_type(),
                pipeline: None,
                fill: FillPattern::new(Some(&fill_bytes), nz(8)),
            },
            8,
            8,
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
    /// same way, which is what [`chunk_index_address`] promises them.
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
            chunk_dimensions: vec![4, 8],
            btree_address: None,
            version: 4,
            chunk_index_type: Some(3),
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let datatype = Datatype::FixedPoint {
            size: 8,
            signed: true,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
        };
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
            (vec![0], FillPattern::new(Some(&seven), nz(8)), Vec::new()),
            (vec![3], FillPattern::ZERO, vec![0u8; 24]),
            (
                vec![3],
                FillPattern::new(Some(&seven), nz(8)),
                seven.repeat(3),
            ),
        ] {
            let ds = simple(dims.clone());
            let spec = RawReadSpec {
                layout: &layout,
                dataspace: &ds,
                datatype: &datatype,
                pipeline: None,
                fill,
            };
            let from_source =
                read_chunked_data_from_source(&BytesSource::new(&file_data), spec, 8, 8).unwrap();
            let cached_from_source = read_chunked_data_cached_from_source(
                &BytesSource::new(&file_data),
                spec,
                8,
                8,
                &ChunkCache::new(),
            )
            .unwrap();
            let cached =
                read_chunked_data_cached(&file_data, spec, 8, 8, &ChunkCache::new()).unwrap();
            assert_eq!(from_source, expected, "dims {dims:?}");
            assert_eq!(cached_from_source, expected, "dims {dims:?}");
            assert_eq!(cached, expected, "dims {dims:?}");
        }
    }

    fn write_offset(buf: &mut Vec<u8>, val: u64, size: u8) {
        match size {
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => panic!("unsupported offset size in test"),
        }
    }

    /// Build a B-tree v1 type 1 leaf node with given chunk infos.
    fn build_chunk_btree_leaf(chunks: &[ChunkInfo], ndims: usize, offset_size: u8) -> Vec<u8> {
        let _os = offset_size as usize;
        let entries_used = chunks.len() as u16;
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(b"TREE");
        buf.push(1); // node_type = 1 (raw data chunks)
        buf.push(0); // node_level = 0 (leaf)
        buf.extend_from_slice(&entries_used.to_le_bytes());

        // Left/right sibling = undefined
        let undef: u64 = if offset_size == 4 {
            0xFFFFFFFF
        } else {
            0xFFFFFFFFFFFFFFFF
        };
        write_offset(&mut buf, undef, offset_size);
        write_offset(&mut buf, undef, offset_size);

        // Entries: key[i], child[i] pairs, then final key
        for chunk in chunks {
            // Key: chunk_size(4) + filter_mask(4) + ndims offsets
            buf.extend_from_slice(&chunk.chunk_size.to_le_bytes());
            buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
            for d in 0..ndims {
                let off = if d < chunk.offsets.len() {
                    chunk.offsets[d]
                } else {
                    0
                };
                write_offset(&mut buf, off, offset_size);
            }
            // Child: address
            write_offset(&mut buf, chunk.address, offset_size);
        }

        // Final key (dummy)
        buf.extend_from_slice(&0u32.to_le_bytes()); // chunk_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask
        for _ in 0..ndims {
            write_offset(&mut buf, u64::MAX, offset_size);
        }

        buf
    }

    // --- ChunkInfo collection tests ---

    #[test]
    fn collect_two_chunks_from_leaf() {
        let ndims = 2; // rank+1 for 1D dataset
        let os: u8 = 8;

        let chunks = vec![
            ChunkInfo {
                chunk_size: 80,
                filter_mask: 0,
                offsets: vec![0, 0],
                address: 0x1000,
            },
            ChunkInfo {
                chunk_size: 80,
                filter_mask: 0,
                offsets: vec![10, 0],
                address: 0x2000,
            },
        ];

        let btree = build_chunk_btree_leaf(&chunks, ndims, os);
        let mut file_data = vec![0u8; 0x3000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].address, 0x1000);
        assert_eq!(result[0].offsets, vec![0, 0]);
        assert_eq!(result[0].chunk_size, 80);
        assert_eq!(result[1].address, 0x2000);
        assert_eq!(result[1].offsets, vec![10, 0]);
    }

    /// Build a B-tree v1 type 1 internal node pointing at the given child node
    /// addresses (keys are dummies — the node-span walk reads only children).
    fn build_chunk_btree_internal(
        level: u8,
        child_addrs: &[u64],
        ndims: usize,
        offset_size: u8,
    ) -> Vec<u8> {
        let entries_used = child_addrs.len() as u16;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TREE");
        buf.push(1); // node_type = 1 (raw data chunks)
        buf.push(level); // node_level > 0 (internal)
        buf.extend_from_slice(&entries_used.to_le_bytes());
        let undef: u64 = if offset_size == 4 {
            0xFFFFFFFF
        } else {
            0xFFFFFFFFFFFFFFFF
        };
        write_offset(&mut buf, undef, offset_size);
        write_offset(&mut buf, undef, offset_size);
        for &addr in child_addrs {
            buf.extend_from_slice(&0u32.to_le_bytes()); // key: chunk_size
            buf.extend_from_slice(&0u32.to_le_bytes()); // key: filter_mask
            for _ in 0..ndims {
                write_offset(&mut buf, 0, offset_size);
            }
            write_offset(&mut buf, addr, offset_size); // child node address
        }
        // Final key.
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for _ in 0..ndims {
            write_offset(&mut buf, u64::MAX, offset_size);
        }
        buf
    }

    #[test]
    fn btree_node_spans_single_leaf() {
        // A leaf's reclaim span is the whole node: its built byte length.
        let ndims = 2;
        let os: u8 = 8;
        let chunks = vec![
            ChunkInfo {
                chunk_size: 80,
                filter_mask: 0,
                offsets: vec![0, 0],
                address: 0x1000,
            },
            ChunkInfo {
                chunk_size: 80,
                filter_mask: 0,
                offsets: vec![10, 0],
                address: 0x2000,
            },
        ];
        let leaf = build_chunk_btree_leaf(&chunks, ndims, os);
        let at = 0x40usize;
        let mut file_data = vec![0u8; 0x3000];
        file_data[at..at + leaf.len()].copy_from_slice(&leaf);

        let spans =
            collect_chunk_btree_node_spans(&BytesSource::new(&file_data), at as u64, ndims, os)
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
            chunk_size: 40,
            filter_mask: 0,
            offsets: vec![0, 0],
            address: 0x100,
        }];
        let leaf0 = build_chunk_btree_leaf(&leaf_chunks, ndims, os);
        let leaf1 = build_chunk_btree_leaf(&leaf_chunks, ndims, os);
        let (l0, l1, root) = (0x1000usize, 0x2000usize, 0x3000usize);
        let internal = build_chunk_btree_internal(1, &[l0 as u64, l1 as u64], ndims, os);

        let mut file = vec![0u8; 0x4000];
        file[l0..l0 + leaf0.len()].copy_from_slice(&leaf0);
        file[l1..l1 + leaf1.len()].copy_from_slice(&leaf1);
        file[root..root + internal.len()].copy_from_slice(&internal);

        let mut spans =
            collect_chunk_btree_node_spans(&BytesSource::new(&file), root as u64, ndims, os)
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

    #[test]
    fn collect_three_chunks() {
        let ndims = 2;
        let os: u8 = 8;

        let chunks = vec![
            ChunkInfo {
                chunk_size: 40,
                filter_mask: 0,
                offsets: vec![0, 0],
                address: 0x100,
            },
            ChunkInfo {
                chunk_size: 40,
                filter_mask: 0,
                offsets: vec![5, 0],
                address: 0x200,
            },
            ChunkInfo {
                chunk_size: 40,
                filter_mask: 0,
                offsets: vec![10, 0],
                address: 0x300,
            },
        ];

        let btree = build_chunk_btree_leaf(&chunks, ndims, os);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].address, 0x100);
        assert_eq!(result[1].address, 0x200);
        assert_eq!(result[2].address, 0x300);
    }

    #[test]
    fn collect_empty_btree() {
        let ndims = 2;
        let os: u8 = 8;
        let btree = build_chunk_btree_leaf(&[], ndims, os);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..btree.len()].copy_from_slice(&btree);

        let result = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn collect_chunk_info_rejects_cyclic_btree() {
        // An internal node whose only child points back at itself. Without a
        // depth guard this recurses forever and overflows the stack (an
        // uncatchable process abort); the guard must turn it into an error.
        let ndims = 2;
        let os: u8 = 8;
        let node = build_chunk_btree_internal(1, &[0u64], ndims, os);
        let mut file_data = vec![0u8; 0x1000];
        file_data[..node.len()].copy_from_slice(&node);

        let err = collect_chunk_info(&file_data, 0, ndims, os, os).unwrap_err();
        assert!(matches!(err, FormatError::ChunkedReadError(_)));
    }

    // --- Chunked read tests (synthetic) ---

    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::{Datatype, DatatypeByteOrder};
    use crate::source::BytesSource;

    fn make_f64_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        }
    }

    fn make_f32_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    }

    /// Build a synthetic file with a B-tree and chunk data for a 1D uncompressed dataset.
    fn build_1d_chunked_file(
        values: &[f64],
        chunk_size_elems: usize,
    ) -> (Vec<u8>, DataLayout, Dataspace) {
        let os: u8 = 8;
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
                chunk_size: chunk_bytes as u32,
                filter_mask: 0,
                offsets: vec![start as u64, 0],
                address: data_offset as u64,
            });

            data_offset += chunk_bytes;
            start += chunk_size_elems;
        }

        // Build B-tree at offset 0x100
        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, os);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_size_elems as u32, elem_size as u32],
            btree_address: Some(btree_addr as u64),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
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
            8,
            8,
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
            8,
            8,
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
    ) {
        use crate::source::ReadSeekSource;
        let spec = RawReadSpec {
            layout,
            dataspace,
            datatype,
            pipeline,
            fill: FillPattern::ZERO,
        };
        let buffered = read_chunked_data_cached(file_data, spec, 8, 8, &ChunkCache::new()).unwrap();
        let from_mem =
            read_chunked_data_from_source(&BytesSource::new(file_data), spec, 8, 8).unwrap();
        let from_seek = read_chunked_data_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data.to_vec())).unwrap(),
            spec,
            8,
            8,
        )
        .unwrap();
        assert_eq!(buffered, from_mem, "BytesSource mismatch");
        assert_eq!(buffered, from_seek, "ReadSeekSource mismatch");
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_chunked_btree_v1_matches_buffered() {
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (file_data, layout, dataspace) = build_1d_chunked_file(&values, 10);
        assert_chunked_streams_match(&file_data, &layout, &dataspace, &make_f64_type(), None);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn read_1d_two_chunks_with_deflate() {
        use crate::filter_pipeline::{FILTER_DEFLATE, FilterDescription, FilterPipeline};
        use crate::filters::compress_chunk;

        let os: u8 = 8;
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
            let compressed = compress_chunk(&chunk_bytes, &pipeline, ctx).unwrap();

            file_data[data_offset..data_offset + compressed.len()].copy_from_slice(&compressed);

            chunk_infos.push(ChunkInfo {
                chunk_size: compressed.len() as u32,
                filter_mask: 0,
                offsets: vec![start as u64, 0],
                address: data_offset as u64,
            });

            data_offset += compressed.len() + 16; // some padding
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, os);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_elems as u32, elem_size as u32],
            btree_address: Some(btree_addr as u64),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
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
            RawReadSpec {
                layout: &layout,
                dataspace: &dataspace,
                datatype: &datatype,
                pipeline: Some(&pipeline),
                fill: FillPattern::ZERO,
            },
            8,
            8,
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

    #[test]
    fn read_2d_four_chunks() {
        // 4x6 dataset with chunk size 2x3 => 4 chunks
        let os: u8 = 8;
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
                    chunk_size: chunk_size as u32,
                    filter_mask: 0,
                    offsets: vec![row_start as u64, col_start as u64, 0],
                    address: data_offset as u64,
                });

                data_offset += chunk_size + 8;
            }
        }

        let btree = build_chunk_btree_leaf(&chunk_infos, ndims, os);
        let btree_addr = 0x100usize;
        file_data[btree_addr..btree_addr + btree.len()].copy_from_slice(&btree);

        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![chunk_dims[0] as u32, chunk_dims[1] as u32, elem_size as u32],
            btree_address: Some(btree_addr as u64),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
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
            8,
            8,
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

        let err = collect_chunk_info(&file_data, 0, 2, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeNodeType(0));
    }

    // --- Implicit chunk generation tests ---

    #[test]
    fn implicit_chunks_1d_five_chunks() {
        let chunks = generate_implicit_chunks(
            0x1000,
            &[100],
            &[20],
            8, // f64
        );
        assert_eq!(chunks.len(), 5);
        let chunk_byte_size = 20 * 8;
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, 0x1000 + i as u64 * chunk_byte_size as u64);
            assert_eq!(c.offsets, vec![i as u64 * 20]);
            assert_eq!(c.filter_mask, 0);
            assert_eq!(c.chunk_size, chunk_byte_size as u32);
        }
    }

    #[test]
    fn implicit_chunks_2d() {
        // 10x6 dataset, 4x3 chunks => ceil(10/4)=3, ceil(6/3)=2 => 6 chunks
        let chunks = generate_implicit_chunks(
            0x2000,
            &[10, 6],
            &[4, 3],
            4, // f32
        );
        assert_eq!(chunks.len(), 6);
        let chunk_byte_size = 4 * 3 * 4;
        // Row-major: (0,0), (0,3), (4,0), (4,3), (8,0), (8,3)
        assert_eq!(chunks[0].offsets, vec![0, 0]);
        assert_eq!(chunks[1].offsets, vec![0, 3]);
        assert_eq!(chunks[2].offsets, vec![4, 0]);
        assert_eq!(chunks[3].offsets, vec![4, 3]);
        assert_eq!(chunks[4].offsets, vec![8, 0]);
        assert_eq!(chunks[5].offsets, vec![8, 3]);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.address, 0x2000 + i as u64 * chunk_byte_size as u64);
        }
    }

    #[test]
    fn implicit_chunks_partial_last() {
        // 25 elements, chunk size 10 => 3 chunks (last partial)
        let chunks = generate_implicit_chunks(0x0, &[25], &[10], 8);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].offsets, vec![0]);
        assert_eq!(chunks[1].offsets, vec![10]);
        assert_eq!(chunks[2].offsets, vec![20]);
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
            chunk_dimensions: vec![chunk_elems as u32, elem_size as u32],
            btree_address: Some(data_addr as u64),
            version: 4,
            chunk_index_type: Some(1),
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
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
            8,
            8,
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
            8,
            8,
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
            8,
            8,
            &cache,
        )
        .unwrap();
        assert!(cache.stats().index_loaded());
        assert!(cache.stats().cached_chunks() > 0);

        // Second read — should hit the decompressed cache
        let raw2 = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            8,
            8,
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
            8,
            8,
            &cache,
        )
        .unwrap();
        assert_eq!(raw.len(), 25 * 8);
        for i in 0..25 {
            let val = f64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(val, i as f64, "mismatch at index {i}");
        }
    }

    // --- Windowed row-read guards (read_chunked_rows_from_source) ---

    /// A rank-0 (scalar) chunked layout leaves `chunk_dims` empty; the windowed
    /// reader must fall back (`Ok(None)`) rather than index `chunk_dims[0]` and
    /// panic. Only a crafted file reaches this (valid chunked layouts are rank
    /// >= 1), but the reader parses untrusted input.
    #[test]
    fn windowed_rows_rank0_chunked_falls_back() {
        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![8], // dimensionality 1 => rank 0
            btree_address: Some(0),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
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
            8,
            8,
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

    /// Inner *dataspace* dimensions whose product overflows `usize` must error
    /// rather than panic (debug) or wrap (release). The chunk edges stay small so
    /// the per-chunk geometry guard passes and the `row_elems` check is what
    /// catches the overflow. `(2^22)^3` overflows 64-bit `usize`; `(2^22)^2`
    /// already overflows 32-bit, so this holds on both.
    #[test]
    fn windowed_rows_inner_dim_product_overflow_errors() {
        let big: u32 = 1 << 22;
        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![1, 2, 2, 2, 8],
            btree_address: Some(0),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 4,
            dimensions: vec![1, big.into(), big.into(), big.into()],
            max_dimensions: None,
        };
        let cache = ChunkCache::new();
        let err = read_chunked_rows_from_source(
            &BytesSource::new(b""),
            RawReadSpec::plain(&layout, &dataspace, &make_f64_type()),
            8,
            8,
            &cache,
            CachePass::LRU,
            0,
            1,
        )
        .expect_err("overflowing inner-dim product must error");
        assert!(
            matches!(err, FormatError::OffsetOverflow { .. }),
            "expected OffsetOverflow, got {err:?}"
        );
    }

    /// A chunk whose edges declare an impossible logical byte size — here a
    /// `(2^22)^3` element chunk of `f64`, far past the 4 GiB format limit — must
    /// be refused up front, even when the dataset itself is small. The per-chunk
    /// geometry guard catches it before the windowed reader sizes any allocation
    /// or builds strides, so a crafted chunk extent cannot drive an out-of-memory
    /// abort. Checked before any I/O: the empty source would otherwise EOF first.
    #[test]
    fn windowed_rows_huge_chunk_geometry_refused() {
        let big: u32 = 1 << 22;
        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![2, big, big, big, 8],
            btree_address: Some(0),
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
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
            8,
            8,
            &cache,
            CachePass::LRU,
            0,
            1,
        )
        .expect_err("an impossible chunk geometry must error");
        assert!(
            matches!(err, FormatError::InvalidChunkGeometry(_)),
            "expected InvalidChunkGeometry, got {err:?}"
        );
    }

    /// An unallocated chunk index (`btree_address == None`, e.g. a late-allocated
    /// never-written dataset) reads as fill for a non-empty window, matching the
    /// whole-dataset reader element for element, and as the empty buffer for a
    /// zero-row one.
    ///
    /// The window and the whole read reach the fill by different routes — the
    /// window returns the buffer it built up front, the whole read builds one in
    /// `chunk_index_address` — so this compares them rather than asserting a
    /// literal on either.
    #[test]
    fn windowed_rows_unallocated_index_matches_whole_read() {
        let layout = DataLayout::Chunked {
            chunk_dimensions: vec![4, 8],
            btree_address: None,
            version: 3,
            chunk_index_type: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![10],
            max_dimensions: None,
        };
        let seven = 7.0f64.to_le_bytes();
        let datatype = make_f64_type();
        for fill in [FillPattern::ZERO, FillPattern::new(Some(&seven), nz(8))] {
            let cache = ChunkCache::new();
            let spec = RawReadSpec {
                layout: &layout,
                dataspace: &dataspace,
                datatype: &datatype,
                pipeline: None,
                fill,
            };
            let whole =
                read_chunked_data_cached_from_source(&BytesSource::new(b""), spec, 8, 8, &cache)
                    .expect("an unallocated dataset reads as fill");
            assert_eq!(whole.len(), 80);

            let window = read_chunked_rows_from_source(
                &BytesSource::new(b""),
                spec,
                8,
                8,
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
                8,
                8,
                &cache,
                CachePass::LRU,
                0,
                0,
            )
            .expect("empty window must still be Ok");
            assert_eq!(empty, Some(Vec::new()));
        }
    }
}
