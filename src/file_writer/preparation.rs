//! Dataset and attribute preparation for file layout and emission.

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::FillTime;
use crate::address::StoredAddress;
use crate::attribute::AttributeMessage;
use crate::attribute_request::PreparedAttributes as AttributeRequest;
use crate::chunked_write;
use crate::chunked_write::ByteSink;
use crate::chunked_write::ChunkOptions;
use crate::chunked_write::ChunkProvider;
use crate::chunked_write::ChunkedHeader;
use crate::chunked_write::ChunkedMeasure;
use crate::chunked_write::ChunkedPlan;
use crate::chunked_write::CompressedChunkSet;
use crate::chunked_write::VerbatimChunks;
use crate::chunked_write::VerbatimLayout;
use crate::chunked_write::VerbatimPlan;
use crate::convert::Narrow;
use crate::dataset_request::{
    DatasetMetadata, PreparedDataset as DatasetRequest, PreparedFill, PreparedStorage,
    WholeFileDataset,
};
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::file_emission::EmissionError;
use crate::fill_value::FillPattern;
use crate::libver::LibVer;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectTarget;
use crate::shared_message::DatatypeLocation;
use crate::type_builders::{AttributeSpecs, DatasetBuilder};

use super::HeaderOptions;
use super::UNDEF_ADDRESS;
use super::layout::DataKind;
use super::layout::DataMeasure;

/// A dataset's metadata and storage source before assigning file addresses.
///
/// The writer resolves committed datatypes, object references, and global-heap references
/// before emission. Providers supply element bytes during emission.
pub(super) struct PreparedDataset {
    /// Link name in the owning group.
    pub(super) name: LinkNameBuf,
    pub(super) dt: Datatype,
    /// Where `dt` is written: in this dataset's header, or in a committed
    /// datatype object it refers to.
    pub(super) dt_location: DatatypeLocation,
    /// The committed datatype target for `dt_location`. `dt_location`
    /// holds the undefined address until the writer places that datatype.
    pub(super) committed_datatype_target: Option<ObjectTarget>,
    pub(super) ds: Dataspace,
    /// The storage source and deferred object and heap references.
    pub(super) storage: PreparedStorage,
    pub(super) attrs: Vec<AttributeMessage>,
    /// The reserved content hash, completed after resolving element addresses.
    #[cfg(feature = "provenance")]
    pub(super) pending_hash: Option<PendingHash>,
    /// Indices in `attrs` paired with their committed datatype targets.
    pub(super) attr_committed_datatype_targets: Vec<(usize, ObjectTarget)>,
    pub(super) chunk_options: ChunkOptions,
    pub(super) maxshape: Option<Vec<MaxExtent>>,
    /// One fill element checked against the dataset datatype, or `None` for the library default.
    pub(super) fill: Option<PreparedFill>,
    /// The fill time policy for the dataset.
    pub(super) fill_time: FillTime,
    /// The length stored in the contiguous layout message, including the logical byte extent
    /// for unallocated storage. Unused for chunked storage.
    pub(super) declared_contiguous_len: u64,
}

impl PreparedDataset {
    /// Parses a dataset request and prepares its attributes and storage source.
    ///
    /// Returns the dataset and the global-heap collections for its variable-length attributes.
    /// Provider payloads remain unread. Committed datatype addresses are undefined until the
    /// writer resolves their targets.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DatasetMissingDatatype`] or [`FormatError::DatasetMissingShape`]
    /// if either is absent. Returns [`FormatError::DatasetRankUnsupported`],
    /// [`FormatError::DatasetDimensionUnlimited`], or [`FormatError::DatasetSizeOverflow`] if
    /// the current dimensions exceed the writer's bounds.
    ///
    /// Returns [`FormatError::LogicalShapeDataMismatch`] if supplied or produced contiguous
    /// bytes differ in length from the shape's byte extent, or [`FormatError::FillValueSizeMismatch`]
    /// if a fill value differs in length from one element. Returns
    /// [`FormatError::FillValueDatatypeMismatch`] if the fill's storage encoding is incompatible
    /// with the dataset datatype. With the `provenance` feature, returns
    /// [`FormatError::DatasetProvenanceRequiresData`] for a content hash on a nonempty unallocated
    /// dataset. Propagates name, datatype, fill encoding, storage source, chunk geometry, filter
    /// option, and attribute preparation errors.
    pub(super) fn prepare(mut db: DatasetBuilder) -> Result<(Self, Vec<VlPatch>), FormatError> {
        let name = super::link_name(core::mem::take(&mut db.name))?;
        let WholeFileDataset {
            metadata,
            storage,
            committed_datatype_targets: attr_committed_datatype_targets,
        } = DatasetRequest::parse(db)?.into_whole_file()?;
        let DatasetMetadata {
            dt,
            element_size: _,
            ds: dspace,
            extent: declared_contiguous_len,
            attrs:
                AttributeRequest {
                    messages: attrs,
                    collections,
                },
            chunk_options,
            fill,
            fill_time,
            committed_datatype_target,
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
        let pending_hash =
            provenance.map(|provenance| PendingHash::reserve(&mut attrs, provenance));
        Ok((
            Self {
                name,
                dt,
                dt_location: match committed_datatype_target {
                    Some(_) => DatatypeLocation::Committed(UNDEF_ADDRESS),
                    None => DatatypeLocation::Inline,
                },
                committed_datatype_target,
                ds: dspace,
                storage,
                attrs,
                #[cfg(feature = "provenance")]
                pending_hash,
                attr_committed_datatype_targets,
                chunk_options,
                maxshape,
                fill,
                fill_time: fill_time.unwrap_or_default(),
                declared_contiguous_len,
            },
            patches,
        ))
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

    /// Returns whether storage requires a chunk index or chunk layout.
    pub(super) fn is_chunked(&self) -> bool {
        self.chunk_options.is_chunked()
            || self.maxshape.is_some()
            || self.storage.chunks().is_some()
    }

    /// Computes header lengths and storage geometry from prepared metadata.
    ///
    /// Provider payloads remain unread. For chunked object-reference data, the data length is
    /// zero until the writer resolves references and encodes the chunks.
    /// Uses `header` for the prefix properties and attribute creation metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if encoded chunks are required and `chunk_set` is
    /// absent. Propagates chunk measurement, filter, and header serialization errors.
    pub(super) fn measure(
        &self,
        data_address: StoredAddress,
        chunk_set: Option<&CompressedChunkSet>,
        dense_attr_info: Option<&[u8]>,
        libver: LibVer,
        header: HeaderOptions<'_>,
    ) -> Result<DatasetMeasure, FormatError> {
        let (data, messages) = if self.is_chunked() {
            if self.storage.reference_targets().is_some() {
                let chunk_dims = self.chunk_options.resolve_chunk_dims(&self.ds.dimensions);
                let ctx = crate::filters::ChunkContext::from_datatype(&chunk_dims, &self.dt)?;
                let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
                let header = self.chunk_options.measure_header(
                    &self.ds.dimensions,
                    ctx,
                    self.maxshape.as_deref(),
                    FillPattern::from_value(self.fill.as_ref().map(PreparedFill::as_bytes), elem),
                    self.storage.allocation(),
                )?;
                (
                    DataMeasure::new(0, DataKind::Chunked),
                    Some((header.layout_message, header.pipeline_message)),
                )
            } else {
                let (measured, data) = self.measure_chunked_storage(data_address, chunk_set)?;
                (
                    data,
                    Some((measured.layout_message, measured.pipeline_message)),
                )
            }
        } else {
            (
                DataMeasure::new(self.contiguous_len(), DataKind::Contiguous),
                None,
            )
        };
        let header_len = self
            .build_header(
                data_address,
                messages.as_ref(),
                dense_attr_info,
                libver,
                header,
            )?
            .len();
        Ok(DatasetMeasure { header_len, data })
    }

    /// Serializes the dataset's header from prepared metadata and storage messages.
    ///
    /// Uses `chunked_messages` when supplied, or writes a contiguous layout at `data_address`.
    /// With `dense_attr_info`, writes the Attribute Info message in place of inline attributes.
    /// Uses `header` for the prefix properties and attribute creation metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AttributeFieldTooLong`] if a compact attribute field exceeds its
    /// encoded width, or [`FormatError::ObjectHeaderMessageTooLarge`] if a header message
    /// exceeds the encoded size limit. Propagates fill-value and layout serialization errors.
    pub(super) fn build_header(
        &self,
        data_address: StoredAddress,
        chunked_messages: Option<&(Vec<u8>, Option<Vec<u8>>)>,
        dense_attr_info: Option<&[u8]>,
        libver: LibVer,
        header: HeaderOptions<'_>,
    ) -> Result<Vec<u8>, FormatError> {
        match chunked_messages {
            Some((layout, pipeline)) => super::build_chunked_dataset_oh(
                &self.dt,
                &self.dt_location,
                &self.ds,
                layout,
                pipeline.as_deref(),
                &self.attrs,
                dense_attr_info,
                self.fill_time,
                self.fill.as_ref().map(PreparedFill::as_bytes),
                header,
            ),
            None => super::build_dataset_oh(
                &self.dt,
                &self.dt_location,
                &self.ds,
                data_address,
                self.declared_contiguous_len,
                &self.attrs,
                dense_attr_info,
                self.fill_time,
                self.fill.as_ref().map(PreparedFill::as_bytes),
                libver,
                header,
            ),
        }
    }

    /// Returns the contiguous payload length, including a provider's declared length.
    pub(super) fn contiguous_len(&self) -> u64 {
        match self.storage.produced() {
            Some(p) => p.total_bytes,
            None => self.storage.raw().len() as u64,
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
            FillPattern::from_value(self.fill.as_ref().map(PreparedFill::as_bytes), elem),
            self.fill_time,
            self.storage.allocation(),
        )
    }

    /// Plans emission of a chunked dataset at `data_address`.
    ///
    /// Encoded chunks are borrowed from `chunk_set`. The writer reads verbatim chunks from their
    /// provider during emission.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if encoded chunks are required and `chunk_set` is
    /// absent. Propagates errors from planning encoded or verbatim chunks.
    pub(super) fn build_chunked<'a>(
        &self,
        data_address: StoredAddress,
        chunk_set: Option<&'a CompressedChunkSet>,
    ) -> Result<ChunkedBuilt<'a>, FormatError> {
        if let Some(rc) = self.storage.chunks() {
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
                &self.ds.dimensions,
                &rc.chunk_dims,
                rc.element_size,
                rc.flags,
                rc.pipeline_message.as_deref(),
                data_address,
                self.maxshape.as_deref(),
            )?;
            Ok(ChunkedBuilt {
                layout_message,
                pipeline_message,
                data: DsData::Streamed(plan),
            })
        } else {
            // Each cached chunk set contains the encoded final element bytes.
            let set = chunk_set.ok_or_else(|| {
                FormatError::Internal("an encoded dataset has no prepared chunk set".into())
            })?;
            let plan = chunked_write::plan_chunked_at(set, data_address)?;
            Ok(ChunkedBuilt {
                layout_message: plan.layout_message.clone(),
                pipeline_message: plan.pipeline_message.clone(),
                data: DsData::Encoded { chunks: set, plan },
            })
        }
    }

    /// Computes a chunked dataset's data-region length and object-header messages.
    ///
    /// Provider payloads remain unread, and encoded chunk indexes are not serialized.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if encoded chunks are required and `chunk_set` is
    /// absent. Propagates errors from planning verbatim chunks or measuring encoded chunks.
    pub(super) fn measure_chunked(
        &self,
        data_address: StoredAddress,
        chunk_set: Option<&CompressedChunkSet>,
    ) -> Result<ChunkedMeasure, FormatError> {
        if let Some(rc) = self.storage.chunks() {
            let VerbatimLayout {
                plan,
                layout_message,
                pipeline_message,
            } = chunked_write::plan_chunked_data_verbatim(
                &rc.meta,
                &self.ds.dimensions,
                &rc.chunk_dims,
                rc.element_size,
                rc.flags,
                rc.pipeline_message.as_deref(),
                data_address,
                self.maxshape.as_deref(),
            )?;
            Ok(ChunkedMeasure {
                data_len: plan.total_len,
                layout_message,
                pipeline_message,
            })
        } else {
            let set = chunk_set.ok_or_else(|| {
                FormatError::Internal("an encoded dataset has no prepared chunk set".into())
            })?;
            chunked_write::measure_chunked_at(set, data_address)
        }
    }

    /// Measures chunk payloads and retains array geometry for placement.
    ///
    /// The writer reads provider payloads during emission. For array layouts, the index is after
    /// the payload starting at `data_address`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the region extends past [`u64::MAX`], or
    /// [`FormatError::Internal`] if encoded chunks are required and `chunk_set` is absent.
    /// Propagates verbatim chunk preparation, array planning, and layout serialization errors.
    pub(super) fn measure_chunked_storage(
        &self,
        data_address: StoredAddress,
        chunk_set: Option<&CompressedChunkSet>,
    ) -> Result<(ChunkedHeader, DataMeasure), FormatError> {
        if let Some(rc) = self.storage.chunks() {
            let prepared = VerbatimChunks::new(
                &rc.meta,
                &self.ds.dimensions,
                &rc.chunk_dims,
                rc.element_size,
                rc.flags,
                rc.pipeline_message.as_deref(),
                data_address,
                self.maxshape.as_deref(),
            )?;
            let data = DataMeasure::chunked(prepared.payload_len(), prepared.array_geometry())?;
            data_address.checked_add(data.length()?)?;
            Ok((
                ChunkedHeader {
                    layout_message: prepared
                        .layout_message(data_address.checked_add(prepared.payload_len())?)?,
                    pipeline_message: rc.pipeline_message.clone(),
                },
                data,
            ))
        } else {
            let set = chunk_set.ok_or_else(|| {
                FormatError::Internal("an encoded dataset has no prepared chunk set".into())
            })?;
            let (raw_length, header) = set.measure_payload_at(data_address)?;
            let data = DataMeasure::chunked(raw_length, set.array_geometry())?;
            data_address.checked_add(data.length()?)?;
            Ok((header, data))
        }
    }
}

/// Header length and storage geometry for reserving a dataset's file regions.
pub(super) struct DatasetMeasure {
    pub(super) header_len: usize,
    /// Payload length and array geometry, or a zero-length placeholder for unresolved chunked
    /// object references.
    pub(super) data: DataMeasure,
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
///
/// Contiguous element buffers and providers remain in prepared storage.
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
///
/// # Panics
///
/// Panics if debug assertions are enabled.
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

/// Writes a contiguous region from its provider using one reusable block buffer.
///
/// # Errors
///
/// Returns [`EmissionError::Format`] with [`FormatError::SerializationError`] if a nonempty
/// region has a zero block size or a supplied block differs from its planned length.
/// Propagates provider errors as [`EmissionError::Format`] and sink errors as
/// [`EmissionError::Sink`].
fn emit_produced_data<Sk: ByteSink>(
    sink: &mut Sk,
    provider: &dyn ChunkProvider,
    total_bytes: u64,
    block_bytes: u64,
) -> Result<(), EmissionError<Sk::Error>> {
    // A nonempty region requires a nonzero block size to advance the cursor.
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

/// Attribute messages, indexed committed datatype targets, and variable-length heap collections.
///
/// Messages have input specification order. Committed datatype targets and heap collections are
/// indexed by their attribute's position in `messages`.
pub(super) struct PreparedAttributes {
    pub(super) messages: Vec<AttributeMessage>,
    pub(super) committed_datatype_targets: Vec<(usize, ObjectTarget)>,
    pub(super) heaps: Vec<VlPatch>,
}

impl PreparedAttributes {
    /// Encodes attribute messages and constructs their global-heap collections.
    ///
    /// The caller resolves committed datatype targets and patches heap addresses after allocation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeName`] or [`FormatError::AttributeFieldTooLong`]
    /// if an attribute name is invalid or exceeds its encoded width. Propagates errors from
    /// [`AttributeRequest::parse`].
    pub(super) fn new(specs: &AttributeSpecs) -> Result<Self, FormatError> {
        let AttributeRequest {
            messages,
            collections,
        } = AttributeRequest::parse(specs)?;
        Ok(Self {
            messages,
            committed_datatype_targets: specs.committed_datatype_targets(),
            heaps: collections
                .into_iter()
                .map(|(attr_index, collections)| VlPatch {
                    attr_index,
                    collections,
                })
                .collect(),
        })
    }
}

/// Global-heap collections for one variable-length attribute.
pub(super) struct VlPatch {
    /// The attribute's collections, in the order their objects appear.
    pub(super) collections: Vec<Vec<u8>>,
    /// The index of the attribute whose element references require patching.
    pub(super) attr_index: usize,
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::num::NonZeroUsize;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use rstest::rstest;

    use super::PreparedDataset;
    use crate::AttrValue;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::chunked_write::{ChunkMeta, ChunkProvider};
    use crate::data_layout::ChunkedLayoutFlags;
    use crate::error::FormatError;
    use crate::libver::LibVer;
    use crate::type_builders::DatasetBuilder;

    #[rstest]
    #[case::contiguous(false, false)]
    #[case::fixed_array(true, false)]
    #[case::extensible_array(true, true)]
    fn lazy_payload_reads_begin_at_emission(#[case] chunked: bool, #[case] unlimited: bool) {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(CountingProvider(Arc::clone(&calls)));
        let mut builder = DatasetBuilder::new("values");
        if chunked {
            builder.with_raw_chunks_lazy(
                crate::make_i32_type(),
                &[4],
                unlimited.then_some(&[crate::MaxExtent::Unlimited][..]),
                &[2],
                NonZeroUsize::new(4).unwrap(),
                ChunkedLayoutFlags::NONE,
                None,
                vec![
                    ChunkMeta {
                        compressed_size: 8,
                        filter_mask: 0,
                    },
                    ChunkMeta {
                        compressed_size: 8,
                        filter_mask: 0,
                    },
                ],
                provider,
            );
        } else {
            builder.with_produced_data(crate::make_i32_type(), &[4], 16, 8, provider);
        }
        let (dataset, heaps) = PreparedDataset::prepare(builder).unwrap();
        assert!(heaps.is_empty());
        assert_eq!(dataset.name.as_str(), "values");
        assert_eq!(dataset.ds.dimensions, [4]);
        assert_eq!(dataset.is_chunked(), chunked);
        let measured = dataset
            .measure(
                StoredAddress::new(0),
                None,
                None,
                LibVer::V110,
                super::super::HeaderOptions::PLAIN,
            )
            .unwrap();
        assert!(measured.header_len > 0);
        match measured.data.array.as_ref() {
            Some(super::super::layout::ArrayMeasure::Fixed(plan)) => {
                assert!(!unlimited);
                assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [28, 34]);
            }
            Some(super::super::layout::ArrayMeasure::Extensible(plan)) => {
                assert!(unlimited);
                assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [72, 298]);
            }
            Some(super::super::layout::ArrayMeasure::Empty(_)) => {
                panic!("expected occupied array geometry")
            }
            None => assert!(!chunked),
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let data = if chunked {
            dataset
                .build_chunked(StoredAddress::new(0), None)
                .unwrap()
                .data
        } else {
            dataset.contiguous_data()
        };
        let mut bytes = Vec::new();
        data.emit(&mut bytes, &dataset.storage).unwrap();
        assert_eq!(bytes.len() as u64, measured.data.length().unwrap());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        let expected: Vec<_> = [1_i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        assert_eq!(&bytes[..expected.len()], expected);
    }

    #[rstest]
    #[case::single(1, false, &[])]
    #[case::fixed(1025, false, &[28, 8227])]
    #[case::extensible(5, true, &[72, 298, 150])]
    fn encoded_measurements_retain_array_geometry(
        #[case] chunks: usize,
        #[case] unlimited: bool,
        #[case] expected: &[u64],
    ) {
        let values: Vec<_> = (0..i32::try_from(chunks * 2).unwrap()).collect();
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&values).with_chunks(&[2]);
        if unlimited {
            builder.with_maxshape(&[crate::MaxExtent::Unlimited]);
        }
        let (dataset, _) = PreparedDataset::prepare(builder).unwrap();
        let chunks = dataset.compress_chunks().unwrap();
        let (_, data) = dataset
            .measure_chunked_storage(StoredAddress::new(4096), Some(&chunks))
            .unwrap();
        let lengths = match data.array.as_ref() {
            Some(super::super::layout::ArrayMeasure::Fixed(plan)) => {
                plan.block_lengths().collect::<Vec<_>>()
            }
            Some(super::super::layout::ArrayMeasure::Extensible(plan)) => {
                plan.block_lengths().collect::<Vec<_>>()
            }
            Some(super::super::layout::ArrayMeasure::Empty(_)) => {
                panic!("expected occupied array geometry")
            }
            None => Vec::new(),
        };
        assert_eq!(lengths, expected);
        assert_eq!(
            data.length().unwrap(),
            values.len() as u64 * 4 + expected.iter().sum::<u64>()
        );
        let built = dataset
            .build_chunked(StoredAddress::new(4096), Some(&chunks))
            .unwrap();
        assert_eq!(built.data.len(), data.length().unwrap());
    }

    #[test]
    fn preparing_an_object_preserves_attribute_order_and_heap_indices() {
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&[1]);
        builder.set_attr("count", AttrValue::I32(7));
        builder.set_attr("label", AttrValue::VarLenString("sample".into()));
        builder.set_attr_committed("typed", AttrValue::I32(11), "types/int");
        builder.set_attr("count", AttrValue::I32(9));
        let (dataset, heaps) = PreparedDataset::prepare(builder).unwrap();
        assert_eq!(
            dataset
                .attrs
                .iter()
                .map(|attr| attr.name.as_str())
                .collect::<Vec<_>>(),
            ["count", "label", "typed"],
        );
        assert_eq!(dataset.attrs[0].raw_data, 9_i32.to_le_bytes());
        assert_eq!(dataset.attrs[2].raw_data, 11_i32.to_le_bytes());
        assert_eq!(
            dataset
                .attr_committed_datatype_targets
                .iter()
                .map(|(index, path)| (*index, path.to_string()))
                .collect::<Vec<_>>(),
            [(2, "types/int".into())],
        );
        assert_eq!(heaps.len(), 1);
        assert_eq!(heaps[0].attr_index, 1);
        assert_eq!(heaps[0].collections.len(), 1);
    }

    #[rstest]
    #[case::measurement(false)]
    #[case::emission_plan(true)]
    fn absent_encoded_chunks_report_an_internal_error(#[case] build: bool) {
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&[1, 2, 3, 4]).with_chunks(&[2]);
        let (dataset, _) = PreparedDataset::prepare(builder).unwrap();
        let result = if build {
            dataset
                .build_chunked(StoredAddress::new(0), None)
                .map(|_| ())
        } else {
            dataset
                .measure(
                    StoredAddress::new(0),
                    None,
                    None,
                    LibVer::V110,
                    super::super::HeaderOptions::PLAIN,
                )
                .map(|_| ())
        };
        let error = result.unwrap_err();
        let FormatError::Internal(reason) = &error else {
            panic!("expected Internal, got {error:?}");
        };
        assert_eq!(reason, "an encoded dataset has no prepared chunk set");
    }

    struct CountingProvider(Arc<AtomicUsize>);

    impl ChunkProvider for CountingProvider {
        fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            let first = i32::try_from(index).unwrap() * 2 + 1;
            out.extend_from_slice(&first.to_le_bytes());
            out.extend_from_slice(&(first + 1).to_le_bytes());
            Ok(())
        }
    }
}
