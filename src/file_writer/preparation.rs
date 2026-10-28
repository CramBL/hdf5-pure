//! Dataset and attribute preparation for file layout and emission.

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::FillTime;
use crate::address::StoredAddress;
use crate::attribute::AttributeMessage;
use crate::attribute_request::PreparedAttributes as AttributeRequest;
use crate::chunked_write::{
    self, ByteSink, ChunkOptions, ChunkProvider, ChunkedMeasure, ChunkedPlan, CompressedChunkSet,
    VerbatimLayout, VerbatimPlan,
};
use crate::convert::Narrow;
use crate::dataset_request::{
    DatasetMetadata, PreparedDataset as DatasetRequest, PreparedStorage, WholeFileDataset,
};
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::file_emission::EmissionError;
use crate::fill_value::FillPattern;
use crate::libver::LibVer;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::shared_message::DatatypeLocation;
use crate::type_builders::{AttributeSpecs, DatasetBuilder};

use super::UNDEF_ADDRESS;

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
    /// The path of the committed datatype `dt_location` refers to. `dt_location`
    /// holds the undefined address until the writer places that datatype.
    pub(super) committed_datatype_path: Option<ObjectPathBuf>,
    pub(super) ds: Dataspace,
    /// The storage source and deferred object and heap references.
    pub(super) storage: PreparedStorage,
    pub(super) attrs: Vec<AttributeMessage>,
    /// The reserved content hash, completed after resolving element addresses.
    #[cfg(feature = "provenance")]
    pub(super) pending_hash: Option<PendingHash>,
    /// Indices in `attrs` paired with the paths of their committed datatypes.
    pub(super) attr_committed_datatype_paths: Vec<(usize, ObjectPathBuf)>,
    pub(super) chunk_options: ChunkOptions,
    pub(super) maxshape: Option<Vec<MaxExtent>>,
    /// A user-defined fill value, encoded in the dataset's datatype, or
    /// `None` for the library default. The writer checks its length against the datatype's
    /// element size in [`PreparedDataset::prepare`].
    pub(super) fill: Option<Vec<u8>>,
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
    /// writer resolves their paths.
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
    /// if a fill value differs in length from one element. With the `provenance` feature, returns
    /// [`FormatError::DatasetProvenanceRequiresData`] for a content hash on a nonempty unallocated
    /// dataset. Propagates name, datatype, storage source, chunk geometry, filter option, and
    /// attribute preparation errors.
    pub(super) fn prepare(mut db: DatasetBuilder) -> Result<(Self, Vec<VlPatch>), FormatError> {
        let name = super::link_name(core::mem::take(&mut db.name))?;
        let WholeFileDataset {
            metadata,
            storage,
            committed_datatype_paths: attr_committed_datatype_paths,
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
        let pending_hash =
            provenance.map(|provenance| PendingHash::reserve(&mut attrs, provenance));
        Ok((
            Self {
                name,
                dt,
                dt_location: match committed_datatype_path {
                    Some(_) => DatatypeLocation::Committed(UNDEF_ADDRESS),
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

    /// Computes header and data-region lengths from prepared metadata.
    ///
    /// Provider payloads remain unread. For chunked object-reference data, the data length is
    /// zero until the writer resolves references and encodes the chunks.
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
    ) -> Result<DatasetMeasure, FormatError> {
        let (data_len, messages) = if self.is_chunked() {
            if self.storage.reference_targets().is_some() {
                let chunk_dims = self.chunk_options.resolve_chunk_dims(&self.ds.dimensions);
                let ctx = crate::filters::ChunkContext::from_datatype(&chunk_dims, &self.dt)?;
                let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
                let header = self.chunk_options.measure_header(
                    &self.ds.dimensions,
                    ctx,
                    self.maxshape.as_deref(),
                    FillPattern::new(self.fill_time, self.fill.as_deref(), elem),
                    self.storage.allocation(),
                )?;
                (0, Some((header.layout_message, header.pipeline_message)))
            } else {
                let measured = self.measure_chunked(data_address, chunk_set)?;
                (
                    measured.data_len,
                    Some((measured.layout_message, measured.pipeline_message)),
                )
            }
        } else {
            (self.contiguous_len(), None)
        };
        let header_len = self
            .build_header(data_address, messages.as_ref(), dense_attr_info, libver)?
            .len();
        Ok(DatasetMeasure {
            header_len,
            data_len,
        })
    }

    /// Serializes the dataset's header from prepared metadata and storage messages.
    ///
    /// Uses `chunked_messages` when supplied, or writes a contiguous layout at `data_address`.
    /// With `dense_attr_info`, writes the Attribute Info message in place of inline attributes.
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
                self.fill.as_deref(),
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
                self.fill.as_deref(),
                libver,
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
            FillPattern::new(self.fill_time, self.fill.as_deref(), elem),
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
}

/// The lengths needed to reserve a dataset's object header and data region.
pub(super) struct DatasetMeasure {
    pub(super) header_len: usize,
    /// The payload and chunk-index length, or zero for unresolved chunked object references.
    pub(super) data_len: u64,
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

/// Attribute messages, indexed committed datatype paths, and variable-length heap collections.
///
/// Messages have input specification order. Committed datatype paths and heap collections are
/// indexed by their attribute's position in `messages`.
pub(super) struct PreparedAttributes {
    pub(super) messages: Vec<AttributeMessage>,
    pub(super) committed_datatype_paths: Vec<(usize, ObjectPathBuf)>,
    pub(super) heaps: Vec<VlPatch>,
}

impl PreparedAttributes {
    /// Encodes attribute messages and constructs their global-heap collections.
    ///
    /// The caller resolves committed datatype paths and patches heap addresses after allocation.
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
            committed_datatype_paths: specs.committed_datatype_paths(),
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
    #[case::contiguous(false)]
    #[case::verbatim_chunks(true)]
    fn lazy_payload_reads_begin_at_emission(#[case] chunked: bool) {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(CountingProvider(Arc::clone(&calls)));
        let mut builder = DatasetBuilder::new("values");
        if chunked {
            builder.with_raw_chunks_lazy(
                crate::make_i32_type(),
                &[4],
                None,
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
            .measure(StoredAddress::new(0), None, None, LibVer::V110)
            .unwrap();
        assert!(measured.header_len > 0);
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
        assert_eq!(bytes.len() as u64, measured.data_len);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        let expected: Vec<_> = [1_i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        assert_eq!(&bytes[..expected.len()], expected);
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
                .attr_committed_datatype_paths
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
                .measure(StoredAddress::new(0), None, None, LibVer::V110)
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
