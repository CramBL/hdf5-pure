//! Dataset and attribute preparation for file layout and emission.

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::FillTime;
use crate::address::StoredAddress;
use crate::attribute::AttributeMessage;
use crate::attribute_request::PreparedAttributes;
use crate::chunked_write::{
    self, ByteSink, ChunkOptions, ChunkProvider, ChunkedMeasure, ChunkedPlan, CompressedChunkSet,
    VerbatimLayout, VerbatimPlan,
};
use crate::convert::Narrow;
use crate::dataset_request::{DatasetMetadata, PreparedDataset, PreparedStorage, WholeFileDataset};
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::file_emission::EmissionError;
use crate::fill_value::FillPattern;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::shared_message::DatatypeLocation;
use crate::type_builders::DatasetBuilder;

pub(super) struct DsFlat {
    /// Link name in the owning group.
    pub(super) name: LinkNameBuf,
    pub(super) dt: Datatype,
    /// Where `dt` is written: in this dataset's header, or in a committed
    /// datatype object it names.
    pub(super) dt_location: DatatypeLocation,
    /// The path of the committed datatype `dt_location` refers to. `dt_location`
    /// holds the undefined address until the writer places that datatype.
    pub(super) committed_datatype_path: Option<ObjectPathBuf>,
    pub(super) ds: Dataspace,
    pub(super) storage: PreparedStorage,
    pub(super) attrs: Vec<AttributeMessage>,
    /// The reserved content hash, completed after resolving element addresses.
    #[cfg(feature = "provenance")]
    pub(super) pending_hash: Option<PendingHash>,
    /// Attribute indices paired with committed datatype paths for address resolution.
    pub(super) attr_committed_datatype_paths: Vec<(usize, ObjectPathBuf)>,
    pub(super) chunk_options: ChunkOptions,
    pub(super) maxshape: Option<Vec<MaxExtent>>,
    /// One encoded fill element, or `None` for the library default.
    pub(super) fill: Option<Vec<u8>>,
    /// The fill time policy for the dataset.
    pub(super) fill_time: FillTime,
    /// The logical byte extent recorded in a contiguous Data Layout message,
    /// including elements whose storage remains unallocated.
    pub(super) declared_contiguous_len: u64,
}

impl DsFlat {
    /// Bytes this dataset's contiguous data region occupies. Produced
    /// data has none in hand, but its total is known from the geometry —
    /// which is the whole reason it can be laid out without being read.
    pub(super) fn contiguous_len(&self) -> u64 {
        match self.storage.produced() {
            Some(p) => p.total_bytes,
            None => self.storage.raw().len() as u64,
        }
    }

    /// Describes this dataset's contiguous region for placement and emission.
    ///
    /// The element buffer or provider stays in prepared storage.
    pub(super) fn contiguous_data(&self) -> DsData<'static> {
        match self.storage.produced() {
            Some(p) => DsData::Produced {
                total_bytes: p.total_bytes,
                block_bytes: p.block_bytes,
            },
            None => DsData::InMemory {
                len: self.storage.raw().len() as u64,
            },
        }
    }

    /// Encodes the dataset's resolved element bytes into an address-independent chunk set.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if an element has zero width,
    /// [`FormatError::UnsupportedFilter`] if a filter has no encoder, or
    /// [`FormatError::InvalidChunkGeometry`] if a chunk is larger than [`u32::MAX`] bytes.
    ///
    /// Returns [`FormatError::ChunkedReadError`] if the chunk-index geometry is
    /// unsupported or exceeds the writer's index limits, or
    /// [`FormatError::ValueTooLargeForPlatform`] if an element size or chunk count exceeds
    /// [`usize::MAX`].
    ///
    /// Returns [`FormatError::FilterError`] for invalid filter parameters or input.
    /// Filter-specific errors from [`ChunkOptions::build_pipeline`] and
    /// [`crate::filters::compress_chunk_with`] are propagated.
    pub(super) fn compress_chunks(&self) -> Result<CompressedChunkSet, FormatError> {
        let chunk_dims = self.chunk_options.resolve_chunk_dims(&self.ds.dimensions);
        let ctx = crate::filters::ChunkContext::from_datatype(&chunk_dims, &self.dt)?;
        let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
        chunked_write::compress_chunks(
            self.storage.raw(),
            &self.ds.dimensions,
            ctx,
            &self.chunk_options,
            self.maxshape.as_deref(),
            FillPattern::new(self.fill_time, self.fill.as_deref(), elem),
            self.storage.allocation(),
        )
    }
}

/// The position of a reserved hash attribute.
#[cfg(feature = "provenance")]
pub(super) struct PendingHash {
    /// The index in the dataset's flattened attribute list.
    pub(super) attribute: usize,
}

#[cfg(feature = "provenance")]
impl PendingHash {
    /// Appends hash metadata with a 64-byte digest placeholder.
    pub(super) fn reserve(
        attrs: &mut Vec<AttributeMessage>,
        provenance: crate::provenance::Provenance,
    ) -> Self {
        let attribute = attrs.len();
        attrs.extend(provenance.deferred_attrs());
        Self { attribute }
    }

    /// Writes the hash of finalized unfiltered element bytes into the reserved attribute.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the reserved attribute is absent or its
    /// data length differs from the hexadecimal digest length.
    pub(super) fn complete(
        self,
        attrs: &mut [AttributeMessage],
        bytes: &[u8],
    ) -> Result<(), FormatError> {
        let attribute = attrs.get_mut(self.attribute).ok_or_else(|| {
            FormatError::Internal("the reserved provenance digest attribute is absent".into())
        })?;
        let hash = crate::provenance::sha256_hex(bytes);
        if attribute.raw_data.len() != hash.len() {
            return Err(FormatError::Internal(
                "the provenance digest does not occupy its reserved bytes".into(),
            ));
        }
        attribute.raw_data.copy_from_slice(hash.as_bytes());
        Ok(())
    }
}

/// A dataset's planned data region.
pub(super) enum DsData<'a> {
    /// Encoded chunks borrowed from the writer's retained set, with finalized metadata.
    Encoded {
        chunks: &'a CompressedChunkSet,
        plan: ChunkedPlan,
    },
    /// A contiguous region emitted from the element buffer in prepared storage.
    InMemory { len: u64 },
    /// A contiguous region written block by block from the provider held in the
    /// dataset's prepared storage.
    Produced {
        /// Bytes the whole region occupies.
        total_bytes: u64,
        /// Bytes per block, the last one excepted.
        block_bytes: u64,
    },
    /// A verbatim chunk plan whose provider is held in the dataset's prepared storage.
    Streamed(VerbatimPlan),
}
impl DsData<'_> {
    /// Writes the data region, including a chunk index for chunked storage.
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if a streamed
    /// or produced region is paired with prepared storage of another kind. Propagates
    /// provider and serialization errors as [`EmissionError::Format`] and sink errors as
    /// [`EmissionError::Sink`].
    pub(super) fn emit<S: ByteSink>(
        &self,
        sink: &mut S,
        storage: &PreparedStorage,
    ) -> Result<(), EmissionError<S::Error>> {
        match self {
            DsData::Encoded { chunks, plan } => {
                chunked_write::emit_chunked_data(sink, chunks, plan)
            }
            DsData::InMemory { .. } => sink.put(storage.raw()).map_err(EmissionError::Sink),
            DsData::Streamed(plan) => {
                let provider = storage
                    .chunks()
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "a streamed region requires prepared chunk storage".into(),
                        )
                    })?
                    .provider
                    .0
                    .as_ref();
                chunked_write::emit_chunked_data_verbatim(sink, plan, provider)
            }
            DsData::Produced {
                total_bytes,
                block_bytes,
            } => {
                let payload = storage.produced().ok_or_else(|| {
                    FormatError::Internal("a produced region requires a prepared producer".into())
                })?;
                emit_produced_data(
                    sink,
                    payload.provider.0.as_ref(),
                    *total_bytes,
                    *block_bytes,
                )
            }
        }
    }

    /// Returns the allocated data length, including any chunk index.
    pub(super) fn len(&self) -> u64 {
        match self {
            DsData::Encoded { plan, .. } => plan.data_len,
            DsData::InMemory { len } => *len,
            DsData::Streamed(plan) => plan.total_len,
            DsData::Produced { total_bytes, .. } => *total_bytes,
        }
    }
}

/// Returns the error for a variable-length dataset whose element bytes
/// are not held in memory when its global heap references are patched.
pub(super) fn vl_element_bytes_absent(link_name: &str) -> FormatError {
    // VL staging goes only to a dataset whose bytes are in memory: a
    // chunked one is patched before its chunks are encoded, and
    // `with_produced_data` debug-asserts against the combination.
    debug_assert!(
        false,
        "a staged VL-string dataset holds its element bytes in memory"
    );
    FormatError::SerializationError(format!(
        "the element bytes of variable-length dataset {link_name:?} are not held in \
         memory, so its global heap references cannot be patched"
    ))
}

/// Emit a contiguous produced region: pull each block from `provider` and
/// write it straight through, so the region's bytes are never all
/// resident. The last block is whatever remains, and a block of any other
/// length is refused — it would shift every address after this dataset.
pub(super) fn emit_produced_data<Sk: ByteSink>(
    sink: &mut Sk,
    provider: &dyn ChunkProvider,
    total_bytes: u64,
    block_bytes: u64,
) -> Result<(), EmissionError<Sk::Error>> {
    // A zero-length block would never advance the cursor, and the loop
    // below would ask for block 0 forever. Callers construct the payload
    // from a checked block size, so this is a construction invariant —
    // but a hang is a far worse failure than a wrong file, and it costs
    // one comparison to make it an error instead.
    if block_bytes == 0 && total_bytes > 0 {
        return Err(FormatError::SerializationError(
            "a produced dataset declared a zero-length block".into(),
        )
        .into());
    }
    // One buffer for the whole region, reused across blocks.
    let mut block = Vec::new();
    let mut written = 0u64;
    let mut index = 0usize;
    while written < total_bytes {
        let expected = block_bytes.min(total_bytes - written);
        block.clear();
        provider.chunk_bytes(index, &mut block)?;
        if block.len() as u64 != expected {
            return Err(FormatError::SerializationError(
                "a produced dataset's block does not match its planned size".into(),
            )
            .into());
        }
        sink.put(&block).map_err(EmissionError::Sink)?;
        written += expected;
        index += 1;
    }
    Ok(())
}

/// A chunked dataset's object-header messages and planned data region.
pub(super) struct ChunkedBuilt<'a> {
    pub(super) layout_message: Vec<u8>,
    pub(super) pipeline_message: Option<Vec<u8>>,
    pub(super) data: DsData<'a>,
}

/// Computes a chunked dataset's data-region length and object-header messages.
///
/// # Errors
///
/// Returns the errors of planning verbatim chunks or measuring encoded chunks.
///
/// # Panics
///
/// Panics if an encoded dataset has no retained chunk set.
pub(super) fn measure_chunked(
    d: &DsFlat,
    data_address: StoredAddress,
    chunk_set: Option<&CompressedChunkSet>,
) -> Result<ChunkedMeasure, FormatError> {
    if let Some(rc) = d.storage.chunks() {
        let VerbatimLayout {
            plan,
            layout_message,
            pipeline_message,
        } = chunked_write::plan_chunked_data_verbatim(
            &rc.meta,
            &d.ds.dimensions,
            &rc.chunk_dims,
            rc.element_size,
            rc.flags,
            rc.pipeline_message.as_deref(),
            data_address,
            d.maxshape.as_deref(),
        )?;
        Ok(ChunkedMeasure {
            data_len: plan.total_len,
            layout_message,
            pipeline_message,
        })
    } else {
        let set =
            chunk_set.expect("an encode-path chunked dataset must have a precomputed chunk set");
        chunked_write::measure_chunked_at(set, data_address)
    }
}

/// Plans emission of a chunked dataset at `data_address`.
///
/// Encoded data borrows the retained chunk set. Verbatim data is read from its provider
/// during emission.
///
/// # Errors
///
/// Returns the errors of planning encoded or verbatim chunks.
///
/// # Panics
///
/// Panics if an encoded dataset has no retained chunk set.
pub(super) fn build_chunked<'a>(
    d: &DsFlat,
    data_address: StoredAddress,
    chunk_set: Option<&'a CompressedChunkSet>,
) -> Result<ChunkedBuilt<'a>, FormatError> {
    if let Some(rc) = d.storage.chunks() {
        // Verbatim chunks are always streamed: the layout is planned from
        // chunk sizes alone, and the bytes are pulled from the provider in
        // the assembly loop (buffered `finish` and streaming `finish_to`
        // share that one emitter, so their output is byte-identical).
        let VerbatimLayout {
            plan,
            layout_message,
            pipeline_message,
        } = chunked_write::plan_chunked_data_verbatim(
            &rc.meta,
            &d.ds.dimensions,
            &rc.chunk_dims,
            rc.element_size,
            rc.flags,
            rc.pipeline_message.as_deref(),
            data_address,
            d.maxshape.as_deref(),
        )?;
        Ok(ChunkedBuilt {
            layout_message,
            pipeline_message,
            data: DsData::Streamed(plan),
        })
    } else {
        // Each cached chunk set contains the encoded final element bytes.
        let set =
            chunk_set.expect("an encode-path chunked dataset must have a precomputed chunk set");
        let plan = chunked_write::plan_chunked_at(set, data_address)?;
        Ok(ChunkedBuilt {
            layout_message: plan.layout_message.clone(),
            pipeline_message: plan.pipeline_message.clone(),
            data: DsData::Encoded { chunks: set, plan },
        })
    }
}

pub(super) fn flatten_dataset(
    mut db: DatasetBuilder,
    all_ds: &mut Vec<DsFlat>,
    ds_vl: &mut Vec<Vec<VlPatch>>,
) -> Result<usize, FormatError> {
    let name = super::link_name(core::mem::take(&mut db.name))?;
    let WholeFileDataset {
        metadata,
        storage,
        committed_datatype_paths: attr_committed_datatype_paths,
    } = PreparedDataset::parse(db)?.into_whole_file()?;
    let DatasetMetadata {
        dt,
        element_size: _,
        ds: dspace,
        extent: declared_contiguous_len,
        attrs: PreparedAttributes {
            messages: attrs,
            collections,
        },
        chunk_options,
        fill,
        fill_time,
        committed_datatype_path,
        #[cfg(feature = "provenance")]
        provenance,
    } = metadata;
    #[cfg(feature = "provenance")]
    let mut attrs = attrs;
    let maxshape = dspace.max_dimensions.clone();
    let patches = collections
        .into_iter()
        .map(|(attr_index, collections)| VlPatch {
            collections,
            attr_index,
        })
        .collect();
    #[cfg(feature = "provenance")]
    let pending_hash = provenance.map(|provenance| PendingHash::reserve(&mut attrs, provenance));
    let idx = all_ds.len();
    all_ds.push(DsFlat {
        name,
        dt,
        dt_location: match committed_datatype_path {
            Some(_) => DatatypeLocation::Committed(super::UNDEF_ADDRESS),
            None => DatatypeLocation::Inline,
        },
        committed_datatype_path,
        ds: dspace,
        storage,
        attrs,
        #[cfg(feature = "provenance")]
        pending_hash,
        attr_committed_datatype_paths,
        chunk_options,
        maxshape,
        fill,
        fill_time: fill_time.unwrap_or_default(),
        declared_contiguous_len,
    });
    ds_vl.push(patches);
    Ok(idx)
}

pub(super) struct VlPatch {
    /// The attribute's collections, in the order their objects appear.
    pub(super) collections: Vec<Vec<u8>>,
    pub(super) attr_index: usize, // index into the relevant attrs Vec
}
