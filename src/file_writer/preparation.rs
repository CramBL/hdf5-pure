//! Dataset and attribute preparation for file layout and emission.

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_format::__private::IndexSlots;

use crate::FillTime;
use crate::address::BaseAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute::AttributeMessage;
use crate::attribute_request::PreparedAttributes as AttributeRequest;
use crate::chunked_write;
use crate::chunked_write::ArrayMeasure;
use crate::chunked_write::ByteSink;
use crate::chunked_write::ChunkArrayKind;
use crate::chunked_write::ChunkOptions;
use crate::chunked_write::ChunkProvider;
use crate::chunked_write::ChunkedHeader;
use crate::chunked_write::ChunkedMeasure;
use crate::chunked_write::CompressedChunkSet;
use crate::chunked_write::EmptyChunkIndex;
use crate::chunked_write::StorageAllocation;
use crate::chunked_write::VerbatimChunks;
use crate::chunked_write::VerbatimLayout;
use crate::chunked_write::VerbatimPayload;
use crate::convert::Narrow;
use crate::dataset_request::{
    DatasetMetadata, PreparedDataset as DatasetRequest, PreparedFill, PreparedStorage,
    WholeFileDataset,
};
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::file_emission::EmissionError;
use crate::file_emission::Reservation;
use crate::fill_value::FillPattern;
use crate::filters::ChunkContext;
use crate::libver::LibVer;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectTarget;
use crate::shared_message::DatatypeLocation;
use crate::type_builders::{AttributeSpecs, DatasetBuilder};
use crate::width::FormatWidths;
use crate::width::OffsetWidth;

use super::HeaderOptions;
use super::layout::DataKind;
use super::layout::DataMeasure;

/// A dataset's metadata and storage source before assigning file addresses.
///
/// The writer resolves committed datatypes, object references, and global-heap references
/// before emission. Providers supply element bytes during emission.
pub(super) struct PreparedDataset {
    widths: FormatWidths,
    /// Link name in the owning group.
    pub(super) name: LinkNameBuf,
    pub(super) dt: Datatype,
    /// Where `dt` is written: in this dataset's header, or in a committed
    /// datatype object it refers to.
    pub(super) dt_location: DatatypeLocation,
    /// The committed datatype target for `dt_location`, whose address is a zero placeholder
    /// until the writer places that datatype.
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
    /// Provider payloads remain unread. Committed datatype references have zero addresses until
    /// the writer resolves their targets.
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
    pub(super) fn prepare(db: DatasetBuilder) -> Result<(Self, Vec<VlPatch>), FormatError> {
        Self::prepare_with_widths(super::WIDTHS, db)
    }

    /// Prepares stored element, fill, and attribute bytes at the selected widths.
    ///
    /// Provider payloads remain unread. The writer resolves destination addresses after layout.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ReferenceWidthUnsupported`] for legacy references at narrow address
    /// widths, or [`FormatError::VariableLengthProviderUnsupported`] for variable-length providers
    /// at those widths. Returns [`FormatError::LengthTooLarge`] if a heap collection exceeds the
    /// selected length width. Propagates the request errors of [`Self::prepare`] and errors from
    /// stored-size calculation and variable-length conversion.
    pub(super) fn prepare_with_widths(
        widths: FormatWidths,
        mut db: DatasetBuilder,
    ) -> Result<(Self, Vec<VlPatch>), FormatError> {
        let name = super::link_name(core::mem::take(&mut db.name))?;
        let WholeFileDataset {
            metadata,
            mut storage,
            committed_datatype_targets: attr_committed_datatype_targets,
        } = DatasetRequest::parse(widths.lengths, db)?.into_whole_file()?;
        let DatasetMetadata {
            dt,
            element_size: canonical_element_size,
            ds: dspace,
            extent: canonical_extent,
            attrs:
                AttributeRequest {
                    messages: mut attrs,
                    collections,
                },
            chunk_options,
            mut fill,
            fill_time,
            committed_datatype_target,
            #[cfg(feature = "provenance")]
            provenance,
        } = metadata;
        require_reference_width(widths, &dt)?;
        if widths.offsets != OffsetWidth::Eight
            && (storage.produced().is_some() || storage.chunks().is_some())
            && hdf5_pure_format::__private::datatype_holds_variable_length(&dt)
        {
            return Err(FormatError::VariableLengthProviderUnsupported {
                offset_size: widths.offsets.get(),
            });
        }
        if let Some(fill) = &mut fill {
            fill.narrow_variable_length(&dt, widths.offsets)?;
        }
        let declared_contiguous_len = if widths.offsets == OffsetWidth::Eight {
            canonical_extent
        } else {
            let target_size =
                hdf5_pure_format::__private::stored_element_size(&dt, widths.offsets)?;
            (canonical_extent / canonical_element_size.get() as u64)
                .checked_mul(u64::from(target_size.get()))
                .ok_or(FormatError::DatasetSizeOverflow {
                    size: canonical_extent,
                    dimension: u64::from(target_size.get()),
                })?
        };
        if let Some(mut staging) = storage.take_vl_string_staging() {
            validate_collections(widths, &staging.collections)?;
            hdf5_pure_format::__private::narrow_variable_length_references(
                widths.offsets,
                &dt,
                storage.raw_mut()?,
                Some(&mut staging.patch_offsets),
            )?;
            storage.restore_vl_string_staging(staging)?;
        } else if storage.produced().is_none()
            && storage.chunks().is_none()
            && storage.allocation() != StorageAllocation::Unallocated
        {
            hdf5_pure_format::__private::narrow_variable_length_references(
                widths.offsets,
                &dt,
                storage.raw_mut()?,
                None,
            )?;
        }
        prepare_attribute_widths(widths, &mut attrs, &collections)?;
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
                widths,
                name,
                dt,
                dt_location: match committed_datatype_target {
                    Some(_) => DatatypeLocation::Committed(StoredAddress::ZERO),
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
                let ctx = ChunkContext::from_stored_datatype(
                    &chunk_dims,
                    &self.dt,
                    hdf5_pure_format::__private::stored_element_size(
                        &self.dt,
                        self.widths.offsets,
                    )?,
                );
                let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
                let header = self.chunk_options.measure_header(
                    self.widths,
                    &self.ds.dimensions,
                    ctx,
                    self.maxshape.as_deref(),
                    FillPattern::from_value(self.fill.as_ref().map(PreparedFill::as_bytes), elem),
                    self.storage.allocation(),
                )?;
                (
                    DataMeasure::new(ByteLength::new(0), DataKind::Chunked),
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
                DataMeasure::new(ByteLength::new(self.contiguous_len()), DataKind::Contiguous),
                None,
            )
        };
        self.measure_prepared(
            data_address,
            data,
            messages,
            dense_attr_info,
            libver,
            header,
        )
    }

    /// Measures a dataset and retains records for its occupied chunk array.
    ///
    /// The descriptor is [`None`] for contiguous storage, unresolved object references, single
    /// chunks, and arrays without stored chunks. Copies encoded chunk metadata or moves prepared
    /// provider records and logical slots into the descriptor. The writer reads provider payloads
    /// during emission.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::measure`] for contiguous storage and unresolved references.
    /// Propagates chunk preparation, array planning, header serialization, and descriptor errors.
    #[cfg(feature = "std")]
    pub(super) fn measure_source(
        &self,
        data_address: StoredAddress,
        chunk_set: Option<&CompressedChunkSet>,
        dense_attr_info: Option<&[u8]>,
        libver: LibVer,
        header: HeaderOptions<'_>,
    ) -> Result<(DatasetMeasure, Option<VerbatimPayload>), FormatError> {
        if !self.is_chunked() || self.storage.reference_targets().is_some() {
            return self
                .measure(data_address, chunk_set, dense_attr_info, libver, header)
                .map(|measure| (measure, None));
        }
        let prepared = self.prepare_chunks(data_address, chunk_set)?;
        let (messages, data) = prepared.measure()?;
        let measured = self.measure_prepared(
            data_address,
            data,
            Some((messages.layout_message, messages.pipeline_message)),
            dense_attr_info,
            libver,
            header,
        )?;
        Ok((measured, prepared.into_array_payload()?))
    }

    /// Serializes the header to measure its length and retains the supplied storage geometry.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::build_header`].
    fn measure_prepared(
        &self,
        data_address: StoredAddress,
        data: DataMeasure,
        messages: Option<(Vec<u8>, Option<Vec<u8>>)>,
        dense_attr_info: Option<&[u8]>,
        libver: LibVer,
        header: HeaderOptions<'_>,
    ) -> Result<DatasetMeasure, FormatError> {
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
        let ctx = ChunkContext::from_stored_datatype(
            &chunk_dims,
            &self.dt,
            hdf5_pure_format::__private::stored_element_size(&self.dt, self.widths.offsets)?,
        );
        let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
        chunked_write::compress_chunks(
            self.widths,
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

    /// Prepares payload and array emission with their object-header messages.
    ///
    /// Checks the payload length and array kind against `measure`. The caller supplies
    /// `array_address` exactly when the dataset has an allocated array. The returned plan borrows
    /// `measure` and any encoded `chunk_set`. The writer reads provider payloads during emission.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if required encoded chunks are absent, the payload length
    /// or array kind differs from `measure`, or array geometry and address presence disagree.
    /// Returns [`FormatError::OffsetOverflow`] if a payload or consecutive array end exceeds
    /// [`u64::MAX`].
    /// Propagates chunk preparation and layout serialization errors.
    pub(super) fn build_chunked<'a>(
        &self,
        data_address: StoredAddress,
        chunk_set: Option<&'a CompressedChunkSet>,
        measure: &'a DataMeasure,
        array_address: Option<StoredAddress>,
    ) -> Result<ChunkedBuilt<'a>, FormatError> {
        if let Some(rc) = self.storage.chunks() {
            // Verbatim chunks are always streamed: the layout is planned from
            // chunk sizes alone, and the bytes are pulled from the provider in
            // the assembly loop (buffered `finish` and streaming `finish_to`
            // share that one emitter, so their output is byte-identical).
            let prepared = VerbatimChunks::new(
                self.widths,
                &rc.meta,
                &self.ds.dimensions,
                &rc.chunk_dims,
                rc.element_size,
                rc.flags,
                rc.pipeline_message.as_deref(),
                data_address,
                self.maxshape.as_deref(),
            )?;
            let array = ArrayEmission::new(measure, prepared.array_kind(), array_address)?;
            let layout_message = prepared.layout_message(
                array_address.unwrap_or(data_address.checked_add(prepared.payload_len())?),
            )?;
            let payload = prepared.into_payload();
            if ByteLength::new(payload.len()) != measure.payload_length() {
                return Err(inconsistent_chunk_measurement());
            }
            Ok(ChunkedBuilt {
                layout_message,
                pipeline_message: rc.pipeline_message.clone(),
                data: DsData::Streamed {
                    payload,
                    array,
                    len: measure.length()?.get(),
                },
            })
        } else {
            let chunks = chunk_set.ok_or_else(|| {
                FormatError::Internal("an encoded dataset has no prepared chunk set".into())
            })?;
            let array = ArrayEmission::new(measure, chunks.array_kind(), array_address)?;
            let payload_end = chunks.payload_end_at(data_address)?;
            if payload_end
                .checked_distance_from(data_address)
                .map(ByteLength::new)
                != Some(measure.payload_length())
            {
                return Err(inconsistent_chunk_measurement());
            }
            let ChunkedHeader {
                layout_message,
                pipeline_message,
            } = chunks.header_at(data_address, array_address.unwrap_or(payload_end))?;
            Ok(ChunkedBuilt {
                layout_message,
                pipeline_message,
                data: DsData::Encoded {
                    chunks,
                    data_address,
                    array,
                    len: measure.length()?.get(),
                },
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
                self.widths,
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
    /// The writer reads provider payloads during emission. Measures consecutive array blocks
    /// after the payload at `data_address`. The caller assigns final block addresses separately.
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
        self.prepare_chunks(data_address, chunk_set)?.measure()
    }

    /// Prepares provider metadata or borrows a retained encoded chunk set.
    ///
    /// The writer reads provider payloads during emission.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if required encoded chunks are absent. Propagates
    /// provider metadata errors from [`VerbatimChunks::new`].
    fn prepare_chunks<'a>(
        &'a self,
        data_address: StoredAddress,
        chunk_set: Option<&'a CompressedChunkSet>,
    ) -> Result<PreparedChunks<'a>, FormatError> {
        let storage = if let Some(rc) = self.storage.chunks() {
            PreparedChunkStorage::Verbatim {
                chunks: VerbatimChunks::new(
                    self.widths,
                    &rc.meta,
                    &self.ds.dimensions,
                    &rc.chunk_dims,
                    rc.element_size,
                    rc.flags,
                    rc.pipeline_message.as_deref(),
                    data_address,
                    self.maxshape.as_deref(),
                )?,
                pipeline: rc.pipeline_message.as_deref(),
            }
        } else {
            PreparedChunkStorage::Encoded(chunk_set.ok_or_else(|| {
                FormatError::Internal("an encoded dataset has no prepared chunk set".into())
            })?)
        };
        Ok(PreparedChunks {
            data_address,
            storage,
        })
    }
}

/// Prepared chunk storage at a stored payload start.
///
/// Encoded chunks and filter-pipeline bytes are borrowed. Provider records are owned.
struct PreparedChunks<'a> {
    data_address: StoredAddress,
    storage: PreparedChunkStorage<'a>,
}

impl PreparedChunks<'_> {
    /// Measures storage and prepares layout and pipeline messages.
    ///
    /// Provider-backed measurement checks the complete payload and array end before constructing
    /// header buffers. Planning array geometry may allocate. Encoded measurement prepares headers
    /// through [`CompressedChunkSet::measure_payload_at`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the payload or array end exceeds [`u64::MAX`].
    /// Propagates array planning and layout serialization errors.
    fn measure(&self) -> Result<(ChunkedHeader, DataMeasure), FormatError> {
        let Self {
            data_address,
            storage,
        } = self;
        Ok(match storage {
            PreparedChunkStorage::Encoded(chunks) => {
                let (raw_length, header) = chunks.measure_payload_at(*data_address)?;
                let data =
                    DataMeasure::chunked(ByteLength::new(raw_length), chunks.array_geometry())?;
                data_address.checked_add(data.length()?.get())?;
                (header, data)
            }
            PreparedChunkStorage::Verbatim { chunks, pipeline } => {
                let data = DataMeasure::chunked(
                    ByteLength::new(chunks.payload_len()),
                    chunks.array_geometry(),
                )?;
                data_address.checked_add(data.length()?.get())?;
                (
                    ChunkedHeader {
                        layout_message: chunks
                            .layout_message(data_address.checked_add(chunks.payload_len())?)?,
                        pipeline_message: pipeline.map(<[u8]>::to_vec),
                    },
                    data,
                )
            }
        })
    }

    /// Captures records and logical slots for an occupied array.
    ///
    /// Moves prepared provider vectors or copies metadata from encoded storage. Returns [`None`]
    /// for storage without an occupied array.
    ///
    /// # Errors
    ///
    /// Returns the encoded metadata errors of [`CompressedChunkSet::array_payload_at`].
    #[cfg(feature = "std")]
    fn into_array_payload(self) -> Result<Option<VerbatimPayload>, FormatError> {
        let Self {
            data_address,
            storage,
        } = self;
        match storage {
            PreparedChunkStorage::Encoded(chunks) => chunks.array_payload_at(data_address),
            PreparedChunkStorage::Verbatim {
                chunks,
                pipeline: _,
            } => Ok(chunks.array_kind().map(|_| chunks.into_payload())),
        }
    }
}

/// Borrowed encoded storage or prepared provider records for chunk measurement.
enum PreparedChunkStorage<'a> {
    /// A retained set of encoded chunks.
    Encoded(&'a CompressedChunkSet),
    /// Owned chunk records with a borrowed filter-pipeline message.
    Verbatim {
        chunks: VerbatimChunks<'a>,
        pipeline: Option<&'a [u8]>,
    },
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

/// Plans for a dataset's payload and optional array emission.
///
/// Contiguous element buffers and providers remain in prepared storage. Array geometry is
/// borrowed from the dataset's retained measurement.
pub(super) enum DsData<'a> {
    /// Borrowed encoded chunks with their payload start and optional array placement.
    Encoded {
        chunks: &'a CompressedChunkSet,
        data_address: StoredAddress,
        array: Option<ArrayEmission<'a>>,
        len: u64,
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
    /// Provider-backed payload records with optional array placement.
    Streamed {
        payload: VerbatimPayload,
        array: Option<ArrayEmission<'a>>,
        len: u64,
    },
}
impl DsData<'_> {
    /// Writes payload bytes followed by array blocks at their stored addresses.
    ///
    /// The caller positions `sink` at the payload start and assigns consecutive array addresses
    /// after the payload. Resolves array addresses through `base`.
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if array addresses differ
    /// from the sink position, or a streamed or produced region is paired with prepared storage
    /// of another kind. Propagates address, provider, and serialization errors as
    /// [`EmissionError::Format`] and output errors as [`EmissionError::Sink`].
    pub(super) fn emit<S: ByteSink>(
        &self,
        sink: &mut S,
        base: BaseAddress,
        storage: &PreparedStorage,
    ) -> Result<(), EmissionError<S::Error>> {
        self.emit_payload(sink, storage)?;
        self.emit_array(|address, bytes| put_array_block(sink, base, address, bytes))
            .map_err(flatten_emission_error)
    }

    /// Writes stored element bytes from retained chunks, an element buffer, or a provider.
    ///
    /// The caller positions `sink` at the payload start. Provider-backed emission reuses one
    /// chunk or block buffer. Array metadata is emitted through [`Self::emit_array`].
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if a streamed or produced
    /// region is paired with prepared storage of another kind. Propagates provider and length
    /// errors as [`EmissionError::Format`] and sink errors as [`EmissionError::Sink`].
    pub(super) fn emit_payload<S: ByteSink>(
        &self,
        sink: &mut S,
        storage: &PreparedStorage,
    ) -> Result<(), EmissionError<S::Error>> {
        match self {
            DsData::Encoded { chunks, .. } => {
                chunks.emit_payload(sink).map_err(EmissionError::Sink)
            }
            DsData::InMemory { .. } => sink.put(storage.raw()).map_err(EmissionError::Sink),
            DsData::Streamed { payload, .. } => {
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
                payload.emit_payload(sink, provider)
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

    /// Serializes complete array blocks at their assigned stored addresses.
    ///
    /// Calls `emit` in serialization order, stopping at its first error. A failure can leave
    /// partial output. Constructs temporary chunk records for encoded storage and uses retained
    /// records for provider-backed storage. Payload providers are read by [`Self::emit_payload`].
    ///
    /// # Errors
    ///
    /// Returns callback errors as [`EmissionError::Sink`]. Propagates address, slot, and encoding
    /// errors as [`EmissionError::Format`].
    pub(super) fn emit_array<E>(
        &self,
        emit: impl FnMut(StoredAddress, &[u8]) -> Result<(), E>,
    ) -> Result<(), EmissionError<E>> {
        self.emit_array_with(None, emit)
    }

    /// Writes array blocks at their individually assigned stored addresses.
    ///
    /// The caller supplies `addresses` in [`ArrayMeasure::visit_blocks`] order. Resolves physical
    /// ranges through `base` and writes zero padding before each block. Payload emission is separate.
    /// A failure can leave partial output.
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if the first address
    /// differs from the prepared array root or a block overlaps earlier output. Propagates
    /// slot, placement, encoding, and physical-range errors as [`EmissionError::Format`] and
    /// output errors as [`EmissionError::Sink`].
    pub(super) fn emit_array_at<S: ByteSink>(
        &self,
        sink: &mut S,
        base: BaseAddress,
        addresses: &[StoredAddress],
    ) -> Result<(), EmissionError<S::Error>> {
        self.emit_array_with(Some(addresses), |address, bytes| {
            put_placed_array_block(sink, base, address, bytes)
        })
        .map_err(flatten_emission_error)
    }

    /// Returns the combined payload and array byte length.
    pub(super) fn len(&self) -> u64 {
        match self {
            DsData::Encoded { len, .. } | DsData::Streamed { len, .. } => *len,
            DsData::InMemory { len } => *len,
            DsData::Produced { total_bytes, .. } => *total_bytes,
        }
    }

    fn emit_array_with<E>(
        &self,
        addresses: Option<&[StoredAddress]>,
        emit: impl FnMut(StoredAddress, &[u8]) -> Result<(), E>,
    ) -> Result<(), EmissionError<E>> {
        match self {
            Self::Encoded {
                chunks,
                data_address,
                array: Some(array),
                len: _,
            } => {
                let (records, _) = chunks.records_at(*data_address)?;
                array.emit(&chunks.index_slots(&records)?, addresses, emit)
            }
            Self::Streamed {
                payload,
                array: Some(array),
                len: _,
            } => array.emit(&payload.index_slots()?, addresses, emit),
            Self::Encoded { array: None, .. }
            | Self::Streamed { array: None, .. }
            | Self::InMemory { .. }
            | Self::Produced { .. } => Ok(()),
        }
    }
}

/// A borrowed array measurement and checked stored root.
///
/// The constructor checks the array kind and the consecutive encoding end at the stored root.
/// Individual block placements are supplied during emission.
pub(super) struct ArrayEmission<'a> {
    measure: &'a ArrayMeasure,
    address: StoredAddress,
}

impl<'a> ArrayEmission<'a> {
    /// Associates matching allocated-array geometry with its checked stored root.
    ///
    /// Returns [`None`] when geometry, kind, and address are all absent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if only some of geometry, kind, and address are present,
    /// or if the measured and supplied array kinds differ. Returns [`FormatError::OffsetOverflow`]
    /// if the consecutive array end exceeds [`u64::MAX`].
    /// Propagates errors from [`ArrayMeasure::length`].
    fn new(
        data: &'a DataMeasure,
        kind: Option<ChunkArrayKind>,
        address: Option<StoredAddress>,
    ) -> Result<Option<Self>, FormatError> {
        match (&data.array, kind, address) {
            (None, None, None) => Ok(None),
            (Some(measure), Some(kind), Some(address)) if measure.kind() == kind => {
                address.checked_add(measure.length()?)?;
                Ok(Some(Self { measure, address }))
            }
            _ => Err(inconsistent_chunk_measurement()),
        }
    }

    /// Serializes the measured array through complete-block callbacks.
    ///
    /// Uses supplied block addresses or consecutive placement from the prepared root.
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if supplied addresses
    /// start at another root. Propagates the errors of [`ArrayMeasure::write_at`] or
    /// [`ArrayMeasure::write_consecutive`].
    fn emit<E>(
        &self,
        slots: &IndexSlots<'_>,
        addresses: Option<&[StoredAddress]>,
        emit: impl FnMut(StoredAddress, &[u8]) -> Result<(), E>,
    ) -> Result<(), EmissionError<E>> {
        let Self { measure, address } = self;
        match addresses {
            Some(addresses) if addresses.first() == Some(address) => {
                measure.write_at(slots, addresses, emit)
            }
            Some(_) => Err(inconsistent_chunk_measurement().into()),
            None => measure.write_consecutive(slots, *address, emit),
        }
    }
}

/// Writes an empty array through the checked sink adapter.
///
/// Builds block geometry for this emission from scalar index metadata.
///
/// # Errors
///
/// Returns sink errors as [`EmissionError::Sink`]. Propagates array planning, encoding, and
/// physical-range errors as [`EmissionError::Format`].
pub(super) fn emit_empty_index<S: ByteSink>(
    sink: &mut S,
    base: BaseAddress,
    index: EmptyChunkIndex,
    address: StoredAddress,
) -> Result<(), EmissionError<S::Error>> {
    index
        .write_at(address, |address, bytes| {
            put_array_block(sink, base, address, bytes)
        })
        .map_err(flatten_emission_error)
}

/// Serializes consecutive array blocks through the checked sink adapter.
///
/// The caller positions `sink` at the resolved physical start of `address`. A failure can leave
/// partial output. Chunk records are supplied through `slots`. Payload providers remain separate.
///
/// # Errors
///
/// Returns output errors as [`EmissionError::Sink`]. Propagates array planning, slot, encoding,
/// and physical-range errors as [`EmissionError::Format`].
pub(super) fn emit_array<S: ByteSink>(
    sink: &mut S,
    base: BaseAddress,
    measure: &ArrayMeasure,
    slots: &IndexSlots<'_>,
    address: StoredAddress,
) -> Result<(), EmissionError<S::Error>> {
    measure
        .write_consecutive(slots, address, |address, bytes| {
            put_array_block(sink, base, address, bytes)
        })
        .map_err(flatten_emission_error)
}

/// Writes individually placed array blocks, padding gaps in physical-address order.
///
/// The caller supplies `addresses` in [`ArrayMeasure::visit_blocks`] order, with the sink at or
/// before the first resolved address. A failure can leave partial output.
///
/// # Errors
///
/// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if a block overlaps earlier
/// output. Propagates planning, placement, slot, encoding, and physical-range errors as
/// [`EmissionError::Format`] and output errors as [`EmissionError::Sink`].
#[cfg(feature = "std")]
pub(super) fn emit_array_at<S: ByteSink>(
    sink: &mut S,
    base: BaseAddress,
    measure: &ArrayMeasure,
    slots: &IndexSlots<'_>,
    addresses: &[StoredAddress],
) -> Result<(), EmissionError<S::Error>> {
    measure
        .write_at(slots, addresses, |address, bytes| {
            put_placed_array_block(sink, base, address, bytes)
        })
        .map_err(flatten_emission_error)
}

/// Writes an empty array at individual block addresses using scalar index geometry.
///
/// The writer constructs block geometry for this emission and pads gaps between blocks.
/// A failure can leave partial output.
///
/// # Errors
///
/// Returns output errors as [`EmissionError::Sink`]. Propagates planning, placement, slot,
/// encoding, and physical-range errors as [`EmissionError::Format`]. Returns
/// [`EmissionError::Format`] with [`FormatError::Internal`] if a block overlaps earlier output.
#[cfg(feature = "std")]
pub(super) fn emit_empty_index_at<S: ByteSink>(
    sink: &mut S,
    base: BaseAddress,
    index: EmptyChunkIndex,
    addresses: &[StoredAddress],
) -> Result<(), EmissionError<S::Error>> {
    index
        .write_blocks_at(addresses, |address, bytes| {
            put_placed_array_block(sink, base, address, bytes)
        })
        .map_err(flatten_emission_error)
}

/// Checks a complete physical block range, then pads to its start and writes its bytes.
///
/// # Errors
///
/// Returns [`EmissionError::Format`] with [`FormatError::OffsetOverflow`] if address resolution
/// overflows, or [`FormatError::Internal`] if the block starts before the sink's position.
/// Returns padding and write errors as [`EmissionError::Sink`].
fn put_placed_array_block<S: ByteSink>(
    sink: &mut S,
    base: BaseAddress,
    address: StoredAddress,
    bytes: &[u8],
) -> Result<(), EmissionError<S::Error>> {
    let start = Reservation::new(base, address, ByteLength::new(bytes.len().to_u64()), ())?.start();
    if sink.position() > start {
        return Err(FormatError::Internal("array blocks overlap earlier output".into()).into());
    }
    while let Some(length) = sink.remaining_to(start)?.nonzero() {
        sink.put_zeros(length.length().min(ByteLength::new(65_536)).to_usize()?)
            .map_err(EmissionError::Sink)?;
    }
    put_array_block(sink, base, address, bytes)
}

/// Writes one block after checking its complete physical range and sink position.
///
/// # Errors
///
/// Returns [`EmissionError::Format`] with [`FormatError::OffsetOverflow`] if resolving the range
/// overflows, or [`FormatError::Internal`] if the sink position differs from the resolved start.
/// Returns output errors as [`EmissionError::Sink`].
fn put_array_block<S: ByteSink>(
    sink: &mut S,
    base: BaseAddress,
    address: StoredAddress,
    bytes: &[u8],
) -> Result<(), EmissionError<S::Error>> {
    let reservation = Reservation::new(base, address, ByteLength::new(bytes.len().to_u64()), ())?;
    if sink.position() != reservation.start() {
        return Err(
            FormatError::Internal("array output differs from its stored address".into()).into(),
        );
    }
    sink.put(bytes).map_err(EmissionError::Sink)
}

/// Removes one callback-error layer, preserving format and sink errors.
fn flatten_emission_error<E>(error: EmissionError<EmissionError<E>>) -> EmissionError<E> {
    match error {
        EmissionError::Format(error) => EmissionError::Format(error),
        EmissionError::Sink(error) => error,
    }
}

/// Returns the format error for incompatible chunk measurements.
fn inconsistent_chunk_measurement() -> FormatError {
    FormatError::Internal("chunk payload or array differs from its prepared measurement".into())
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
        Self::with_widths(super::WIDTHS, specs)
    }

    /// Prepares attributes with variable-length references at `widths.offsets`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new`] for invalid attribute requests, and the errors of
    /// [`prepare_attribute_widths`] for unsupported references or narrow-field failures.
    pub(super) fn with_widths(
        widths: FormatWidths,
        specs: &AttributeSpecs,
    ) -> Result<Self, FormatError> {
        let AttributeRequest {
            mut messages,
            collections,
        } = AttributeRequest::parse(specs)?;
        prepare_attribute_widths(widths, &mut messages, &collections)?;
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

/// Checks the creation policy for object and dataset-region references.
///
/// # Errors
///
/// Returns [`FormatError::ReferenceWidthUnsupported`] if an address-bearing datatype is paired
/// with two- or four-byte file offsets, including nested occurrences.
pub(crate) fn require_reference_width(
    widths: FormatWidths,
    datatype: &Datatype,
) -> Result<(), FormatError> {
    if widths.offsets != OffsetWidth::Eight
        && hdf5_pure_format::__private::datatype_holds_object_address(datatype)
    {
        return Err(FormatError::ReferenceWidthUnsupported {
            offset_size: widths.offsets.get(),
        });
    }
    Ok(())
}

/// Checks existing collection byte lengths against the selected length width.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if a collection exceeds `widths.lengths`.
pub(crate) fn validate_collections(
    widths: FormatWidths,
    collections: &[Vec<u8>],
) -> Result<(), FormatError> {
    let maximum = u64::MAX >> (64 - 8 * widths.lengths.get());
    for collection in collections {
        if collection.len() as u64 > maximum {
            return Err(FormatError::LengthTooLarge {
                length: collection.len() as u64,
                length_size: widths.lengths.get(),
            });
        }
    }
    Ok(())
}

/// Checks attribute reference policy and converts canonical variable-length raw values.
///
/// # Errors
///
/// Returns the errors of [`validate_collections`] and [`require_reference_width`]. Propagates
/// variable-length conversion errors for invalid layouts, buffer ranges, or addresses.
pub(crate) fn prepare_attribute_widths(
    widths: FormatWidths,
    messages: &mut [AttributeMessage],
    collections: &[(usize, Vec<Vec<u8>>)],
) -> Result<(), FormatError> {
    for (_, collections) in collections {
        validate_collections(widths, collections)?;
    }
    for message in messages {
        require_reference_width(widths, &message.datatype)?;
        hdf5_pure_format::__private::narrow_variable_length_references(
            widths.offsets,
            &message.datatype,
            &mut message.raw_data,
            None,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;

    use core::num::NonZeroUsize;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use hdf5_pure_core::__private::ByteLength;
    use hdf5_pure_core::PhysicalAddress;
    use hdf5_pure_format::__private as format;
    use hdf5_pure_format::__private::ChunkIndexLayout;
    use hdf5_pure_format::__private::ChunkRecord;
    use hdf5_pure_format::__private::DataLayout;
    use hdf5_pure_format::__private::IndexSlots;
    use rstest::rstest;

    use super::PreparedAttributes;
    use super::PreparedDataset;
    use crate::AttrValue;
    use crate::VariableLengthType;
    use crate::address::BaseAddress;
    use crate::address::BaseAddressExt;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::attribute::AttributeMessage;
    use crate::chunked_write;
    use crate::chunked_write::ByteSink;
    use crate::chunked_write::{ChunkMeta, ChunkProvider};
    use crate::convert::Narrow;
    use crate::data_layout::ChunkedLayoutFlags;
    use crate::dataspace::Dataspace;
    use crate::dataspace::DataspaceType;
    use crate::datatype::Datatype;
    use crate::error::FormatError;
    use crate::file_emission::EmissionError;
    use crate::libver::LibVer;
    use crate::shared_message::DatatypeLocation;
    use crate::type_builders::DatasetBuilder;
    use crate::width::FormatWidths;
    use crate::width::LengthWidth;
    use crate::width::OffsetWidth;

    #[rstest]
    #[case::contiguous_two(false, false, LengthWidth::Two)]
    #[case::contiguous_four(false, false, LengthWidth::Four)]
    #[case::contiguous_eight(false, false, LengthWidth::Eight)]
    #[case::fixed_array_two(true, false, LengthWidth::Two)]
    #[case::fixed_array_four(true, false, LengthWidth::Four)]
    #[case::fixed_array_eight(true, false, LengthWidth::Eight)]
    #[case::extensible_array_eight(true, true, LengthWidth::Eight)]
    fn lazy_payload_reads_begin_at_emission(
        #[case] chunked: bool,
        #[case] unlimited: bool,
        #[case] lengths: LengthWidth,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    ) {
        let widths = FormatWidths { offsets, lengths };
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
        let (dataset, heaps) = PreparedDataset::prepare_with_widths(widths, builder).unwrap();
        assert!(heaps.is_empty());
        assert_eq!(dataset.name.as_str(), "values");
        assert_eq!(dataset.ds.dimensions, [4]);
        assert_eq!(dataset.is_chunked(), chunked);
        let measured = dataset.measure(
            StoredAddress::new(0),
            None,
            None,
            LibVer::V110,
            super::super::HeaderOptions::PLAIN.with_widths(widths),
        );
        let measured = measured.unwrap();
        assert!(measured.header_len > 0);
        match measured.data.array.as_ref() {
            Some(super::super::layout::ArrayMeasure::Fixed(plan)) => {
                assert!(!unlimited);
                assert_eq!(
                    plan.block_lengths().collect::<Vec<_>>(),
                    [
                        12 + u64::from(offsets.get()) + u64::from(lengths.get()),
                        10 + 3 * u64::from(offsets.get())
                    ]
                );
            }
            Some(super::super::layout::ArrayMeasure::Extensible(plan)) => {
                assert!(unlimited);
                if offsets == OffsetWidth::Eight {
                    assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [72, 298]);
                }
            }
            Some(super::super::layout::ArrayMeasure::Empty(_)) => {
                panic!("expected occupied array geometry")
            }
            None => assert!(!chunked),
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let data = if chunked {
            dataset
                .build_chunked(
                    StoredAddress::ZERO,
                    None,
                    &measured.data,
                    measured.data.array_address(StoredAddress::ZERO).unwrap(),
                )
                .unwrap()
                .data
        } else {
            dataset.contiguous_data()
        };
        let mut bytes = Vec::new();
        data.emit(&mut bytes, crate::BaseAddress::ZERO, &dataset.storage)
            .unwrap();
        assert_eq!(
            ByteLength::new(bytes.len().to_u64()),
            measured.data.length().unwrap()
        );
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
            ByteLength::new(values.len() as u64 * 4 + expected.iter().sum::<u64>())
        );
        let built = dataset
            .build_chunked(
                StoredAddress::new(4096),
                Some(&chunks),
                &data,
                data.array_address(StoredAddress::new(4096)).unwrap(),
            )
            .unwrap();
        assert_eq!(ByteLength::new(built.data.len()), data.length().unwrap());
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
        let empty = super::super::layout::DataMeasure::new(
            ByteLength::new(0),
            super::super::layout::DataKind::Chunked,
        );
        let result = if build {
            dataset
                .build_chunked(StoredAddress::ZERO, None, &empty, None)
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

    #[rstest]
    #[case::encoded_fixed(false, false, false)]
    #[case::encoded_fixed_filtered(false, false, true)]
    #[case::encoded_extensible(false, true, false)]
    #[case::encoded_extensible_filtered(false, true, true)]
    #[case::verbatim_fixed(true, false, false)]
    #[case::verbatim_fixed_filtered(true, false, true)]
    #[case::verbatim_extensible(true, true, false)]
    #[case::verbatim_extensible_filtered(true, true, true)]
    fn array_emission_precedes_payload_reads_and_preserves_consecutive_bytes(
        #[case] verbatim: bool,
        #[case] unlimited: bool,
        #[case] filtered: bool,
    ) {
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&[1, 2, 3, 4]).with_chunks(&[2]);
        if unlimited {
            builder.with_maxshape(&[crate::MaxExtent::Unlimited]);
        }
        if filtered {
            builder.with_fletcher32();
        }
        let (seed, _) = PreparedDataset::prepare(builder).unwrap();
        let chunks = seed.compress_chunks().unwrap();
        let mut payload = Vec::new();
        chunks.emit_payload(&mut payload).unwrap();
        let width = if filtered { 12 } else { 8 };
        assert_eq!(payload.len(), 2 * width);
        let reads = Arc::new(AtomicUsize::new(0));
        let address = StoredAddress::new((1 << 32) + 8_192);
        let array_address = StoredAddress::new((1 << 32) + 128);
        let records = [
            ChunkRecord {
                address,
                stored_size: width as u64,
                filter_mask: 0,
            },
            ChunkRecord {
                address: address.checked_add(width as u64).unwrap(),
                stored_size: width as u64,
                filter_mask: 0,
            },
        ];
        let header = chunks.header(&records, array_address).unwrap();
        let dataset = if verbatim {
            let mut builder = DatasetBuilder::new("values");
            builder.with_raw_chunks_lazy(
                crate::make_i32_type(),
                &[4],
                unlimited.then_some(&[crate::MaxExtent::Unlimited][..]),
                &[2],
                NonZeroUsize::new(4).unwrap(),
                ChunkedLayoutFlags::NONE,
                header.pipeline_message.clone(),
                records
                    .iter()
                    .map(|record| ChunkMeta {
                        compressed_size: record.stored_size,
                        filter_mask: record.filter_mask,
                    })
                    .collect(),
                Box::new(StoredChunks {
                    chunks: payload.chunks_exact(width).map(<[u8]>::to_vec).collect(),
                    reads: Arc::clone(&reads),
                }),
            );
            PreparedDataset::prepare(builder).unwrap().0
        } else {
            seed
        };
        let encoded = (!verbatim).then_some(&chunks);
        let (_, measured) = dataset.measure_chunked_storage(address, encoded).unwrap();
        let built = dataset
            .build_chunked(address, encoded, &measured, Some(array_address))
            .unwrap();
        let parsed = DataLayout::parse(&built.layout_message, 8, 8).unwrap();
        let DataLayout::Chunked { index, .. } = parsed else {
            panic!("expected chunked storage, got {parsed:?}");
        };
        match (index, unlimited) {
            (ChunkIndexLayout::FixedArray { address, .. }, false)
            | (ChunkIndexLayout::ExtensibleArray { address, .. }, true) => {
                assert_eq!(address, Some(array_address));
            }
            other => panic!("unexpected chunk index {other:?}"),
        }
        assert_eq!(built.pipeline_message, header.pipeline_message);
        let mut callbacks = 0;
        let error = built
            .data
            .emit_array(|_, _| {
                callbacks += 1;
                Err(17_u8)
            })
            .unwrap_err();
        assert!(matches!(error, EmissionError::Sink(17)), "{error:?}");
        assert_eq!(callbacks, 1);
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        let mut array = Vec::new();
        let mut cursor = array_address;
        built
            .data
            .emit_array(|address, bytes| {
                assert_eq!(address, cursor);
                cursor = cursor.checked_add(bytes.len() as u64).unwrap();
                array.extend_from_slice(bytes);
                Ok::<(), core::convert::Infallible>(())
            })
            .unwrap();
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        let slots = IndexSlots::new(&records, &[0, 1], 2).unwrap();
        let widths = super::super::WIDTHS;
        let expected = if unlimited {
            format::build_extensible_array_at(
                &slots,
                8,
                widths.offsets,
                widths.lengths,
                filtered,
                array_address,
            )
        } else {
            format::build_fixed_array_at(
                &slots,
                8,
                widths.offsets,
                widths.lengths,
                filtered,
                array_address,
            )
        }
        .unwrap();
        assert_eq!(array, expected);
        let mut emitted = Vec::new();
        built
            .data
            .emit_payload(&mut emitted, &dataset.storage)
            .unwrap();
        assert_eq!(emitted, payload);
        assert_eq!(reads.load(Ordering::Relaxed), if verbatim { 2 } else { 0 });
        let consecutive = dataset
            .build_chunked(
                address,
                encoded,
                &measured,
                measured.array_address(address).unwrap(),
            )
            .unwrap();
        consecutive
            .data
            .emit_array(|_, bytes| {
                emitted.extend_from_slice(bytes);
                Ok::<(), core::convert::Infallible>(())
            })
            .unwrap();
        let legacy = chunked_write::assemble_chunked_at(&chunks, address).unwrap();
        assert_eq!(emitted, legacy.data_bytes);
        assert_eq!(consecutive.layout_message, legacy.layout_message);
        assert_eq!(consecutive.pipeline_message, legacy.pipeline_message);
        assert_eq!(reads.load(Ordering::Relaxed), if verbatim { 2 } else { 0 });
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[rstest]
    #[case::fixed(1_025, false, "prepared_fixed_parts", "prepared_fixed_control")]
    #[case::extensible(
        4_097,
        true,
        "prepared_extensible_parts",
        "prepared_extensible_control"
    )]
    fn array_plans_retain_headers_and_bound_emission_buffers(
        #[case] count: usize,
        #[case] unlimited: bool,
        #[case] region: &str,
        #[case] control: &str,
    ) {
        use core::mem;

        use test_util::allocation;

        let values: Vec<_> = (0..i32::try_from(count * 2).unwrap()).collect();
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&values).with_chunks(&[2]);
        if unlimited {
            builder.with_maxshape(&[crate::MaxExtent::Unlimited]);
        }
        let (dataset, _) = PreparedDataset::prepare(builder).unwrap();
        let chunks = dataset.compress_chunks().unwrap();
        let (_, measured) = dataset
            .measure_chunked_storage(StoredAddress::ZERO, Some(&chunks))
            .unwrap();
        let (control, positive) = allocation::measure(control, || vec![0_u8; 1_024]);
        assert!(positive.blocks > 0, "{positive}");
        assert_eq!(positive.peak_bytes, 1_024, "{positive}");
        drop(control);
        let index_address = measured.array_address(StoredAddress::ZERO).unwrap();
        let (built, planned) = allocation::measure(region, || {
            dataset
                .build_chunked(StoredAddress::ZERO, Some(&chunks), &measured, index_address)
                .unwrap()
        });
        let retained = built.layout_message.capacity()
            + built.pipeline_message.as_ref().map_or(0, Vec::capacity);
        let records = count * mem::size_of::<ChunkRecord>();
        assert!(planned.peak_bytes >= retained as u64, "{planned}");
        assert!(planned.peak_bytes <= retained as u64 + 1_024, "{planned}");
        assert_eq!(planned.live_bytes, retained as u64, "{planned}");
        let array = measured.array.as_ref().unwrap();
        let mut largest = 0;
        let mut blocks = 0;
        array
            .visit_blocks(|length| {
                largest = largest.max(length);
                blocks += 1;
                Ok(())
            })
            .unwrap();
        let ((), emitted) = allocation::measure(&format!("{region}_emission"), || {
            let mut cursor = index_address.unwrap();
            built
                .data
                .emit_array(|address, bytes| {
                    assert_eq!(address, cursor);
                    cursor = cursor.checked_add(bytes.len() as u64).unwrap();
                    Ok::<(), core::convert::Infallible>(())
                })
                .unwrap();
            assert_eq!(
                cursor.checked_distance_from(index_address.unwrap()),
                Some(array.length().unwrap())
            );
        });
        assert_eq!(emitted.live_bytes, 0, "{emitted}");
        assert!(emitted.peak_bytes >= records as u64 + largest, "{emitted}");
        assert!(
            emitted.peak_bytes <= records as u64 + 2 * largest + blocks * 128 + 16_384,
            "{emitted}"
        );
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[rstest]
    #[case::fixed(false, false, "fixed_provider_measurement_overflow")]
    #[case::fixed_filtered(false, true, "filtered_fixed_provider_measurement_overflow")]
    #[case::extensible(true, false, "extensible_provider_measurement_overflow")]
    #[case::extensible_filtered(true, true, "filtered_extensible_provider_measurement_overflow")]
    fn provider_measurement_rejects_index_overflow_before_allocating_headers(
        #[case] unlimited: bool,
        #[case] filtered: bool,
        #[case] region: &str,
    ) {
        use test_util::allocation;

        let reads = Arc::new(AtomicUsize::new(0));
        let pipeline = filtered.then(|| test_util::filter_pipeline::user_filter(true));
        assert!(!pipeline.as_ref().is_some_and(Vec::is_empty));
        let mut builder = DatasetBuilder::new("values");
        builder.with_raw_chunks_lazy(
            crate::make_i32_type(),
            &[4],
            unlimited.then_some(&[crate::MaxExtent::Unlimited][..]),
            &[2],
            NonZeroUsize::new(4).unwrap(),
            ChunkedLayoutFlags::NONE,
            pipeline,
            vec![
                ChunkMeta {
                    compressed_size: 8,
                    filter_mask: u32::from(filtered)
                };
                2
            ],
            Box::new(CountingProvider(Arc::clone(&reads))),
        );
        let (dataset, _) = PreparedDataset::prepare(builder).unwrap();
        let address = StoredAddress::new(u64::MAX - 16);
        let prepared = dataset.prepare_chunks(address, None).unwrap();
        let records = [
            ChunkRecord {
                address,
                stored_size: 8,
                filter_mask: u32::from(filtered),
            },
            ChunkRecord {
                address: address.checked_add(8).unwrap(),
                stored_size: 8,
                filter_mask: u32::from(filtered),
            },
        ];
        let slots = IndexSlots::new(&records, &[0, 1], 2).unwrap();
        let widths = super::super::WIDTHS;
        let index = if unlimited {
            format::build_extensible_array_at(
                &slots,
                8,
                widths.offsets,
                widths.lengths,
                filtered,
                StoredAddress::ZERO,
            )
        } else {
            format::build_fixed_array_at(
                &slots,
                8,
                widths.offsets,
                widths.lengths,
                filtered,
                StoredAddress::ZERO,
            )
        }
        .unwrap();
        let length = 16 + u64::try_from(index.len()).unwrap();
        let (control, positive) =
            allocation::measure("provider_measurement_overflow_control", || {
                vec![0_u8; 1_024]
            });
        assert!(positive.blocks > 0, "{positive}");
        assert_eq!(positive.peak_bytes, 1_024, "{positive}");
        assert_eq!(positive.live_bytes, 1_024, "{positive}");
        drop(control);
        let (error, measured) = allocation::measure(region, || prepared.measure().err().unwrap());
        assert_eq!(
            error,
            FormatError::OffsetOverflow {
                offset: address.get(),
                length
            }
        );
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert_eq!(measured.live_bytes, 0, "{measured}");
        if unlimited {
            assert!(measured.peak_bytes > 0, "{measured}");
            assert!(measured.peak_bytes <= 4_096, "{measured}");
        } else {
            assert_eq!(measured.blocks, 0, "{measured}");
            assert_eq!(measured.bytes, 0, "{measured}");
            assert_eq!(measured.peak_bytes, 0, "{measured}");
        }
    }

    #[rstest]
    #[case::fixed_missing(false, None, FormatError::Internal("chunk payload or array differs from its prepared measurement".into()))]
    #[case::extensible_missing(true, None, FormatError::Internal("chunk payload or array differs from its prepared measurement".into()))]
    #[case::fixed_overflow(false, Some(u64::MAX), FormatError::OffsetOverflow { offset: u64::MAX, length: 62 })]
    #[case::extensible_overflow(true, Some(u64::MAX), FormatError::OffsetOverflow { offset: u64::MAX, length: 370 })]
    fn array_preparation_rejects_missing_or_overflowing_addresses(
        #[case] unlimited: bool,
        #[case] array_address: Option<u64>,
        #[case] expected: FormatError,
    ) {
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&[1, 2, 3, 4]).with_chunks(&[2]);
        if unlimited {
            builder.with_maxshape(&[crate::MaxExtent::Unlimited]);
        }
        let (dataset, _) = PreparedDataset::prepare(builder).unwrap();
        let chunks = dataset.compress_chunks().unwrap();
        let (_, measured) = dataset
            .measure_chunked_storage(StoredAddress::ZERO, Some(&chunks))
            .unwrap();
        let error = dataset
            .build_chunked(
                StoredAddress::ZERO,
                Some(&chunks),
                &measured,
                array_address.map(StoredAddress::new),
            )
            .err()
            .unwrap();
        assert_eq!(error, expected);
    }

    #[rstest]
    #[case::encoded_kind(false, true)]
    #[case::encoded_payload(false, false)]
    #[case::verbatim_kind(true, true)]
    #[case::verbatim_payload(true, false)]
    fn array_preparation_requires_matching_payload_length_and_array_kind(
        #[case] verbatim: bool,
        #[case] change_kind: bool,
    ) {
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&[1, 2, 3, 4]).with_chunks(&[2]);
        let (dataset, _) = PreparedDataset::prepare(builder).unwrap();
        let chunks = dataset.compress_chunks().unwrap();
        let reads = Arc::new(AtomicUsize::new(0));
        let dataset = if verbatim {
            let mut builder = DatasetBuilder::new("values");
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
                        filter_mask: 0
                    };
                    2
                ],
                Box::new(CountingProvider(Arc::clone(&reads))),
            );
            PreparedDataset::prepare(builder).unwrap().0
        } else {
            dataset
        };
        let mut other = DatasetBuilder::new("other");
        other.with_i32_data(&[1, 2, 3, 4]).with_chunks(&[2]);
        if change_kind {
            other.with_maxshape(&[crate::MaxExtent::Unlimited]);
        } else {
            other.with_i32_data(&[1, 2, 3, 4, 5, 6]).with_shape(&[6]);
        }
        let (other, _) = PreparedDataset::prepare(other).unwrap();
        let other_chunks = other.compress_chunks().unwrap();
        let (_, measured) = other
            .measure_chunked_storage(StoredAddress::ZERO, Some(&other_chunks))
            .unwrap();
        let error = dataset
            .build_chunked(
                StoredAddress::ZERO,
                (!verbatim).then_some(&chunks),
                &measured,
                measured.array_address(StoredAddress::ZERO).unwrap(),
            )
            .err()
            .unwrap();
        let FormatError::Internal(reason) = error else {
            panic!("expected Internal, got {error:?}");
        };
        assert_eq!(
            reason,
            "chunk payload or array differs from its prepared measurement"
        );
        assert_eq!(reads.load(Ordering::Relaxed), 0);
    }

    #[rstest]
    #[case::userblock(4_096, 7, 4_103, None)]
    #[case::physical_start(u64::MAX, 1, 0, Some(FormatError::OffsetOverflow { offset: 1, length: u64::MAX }))]
    #[case::physical_end(u64::MAX - 10, 8, u64::MAX - 2, Some(FormatError::OffsetOverflow { offset: u64::MAX - 2, length: 3 }))]
    #[case::position(0, 7, 8, Some(FormatError::Internal("array output differs from its stored address".into())))]
    fn array_output_checks_complete_physical_ranges_before_writing(
        #[case] base: u64,
        #[case] address: u64,
        #[case] position: u64,
        #[case] expected: Option<FormatError>,
    ) {
        let mut sink = PositionedBytes {
            position,
            bytes: Vec::new(),
        };
        let result = super::put_array_block(
            &mut sink,
            BaseAddress::new(base),
            StoredAddress::new(address),
            b"abc",
        );
        if let Some(expected) = expected {
            assert_eq!(result.unwrap_err().into_format(), expected);
            assert!(sink.bytes.is_empty());
        } else {
            result.unwrap();
            assert_eq!(sink.bytes, b"abc");
            assert_eq!(sink.position, base + address + 3);
        }
    }

    #[rstest]
    #[case::padding(0, 7, 3, None)]
    #[case::overlap(0, 7, 8, Some(FormatError::Internal("array blocks overlap earlier output".into())))]
    #[case::physical_start(u64::MAX, 1, 0, Some(FormatError::OffsetOverflow { offset: 1, length: u64::MAX }))]
    #[case::physical_end(u64::MAX - 10, 8, 0, Some(FormatError::OffsetOverflow { offset: u64::MAX - 2, length: 3 }))]
    fn placed_array_blocks_check_complete_ranges_before_padding(
        #[case] base: u64,
        #[case] address: u64,
        #[case] position: u64,
        #[case] expected: Option<FormatError>,
    ) {
        let mut sink = PositionedBytes {
            position,
            bytes: Vec::new(),
        };
        let result = super::put_placed_array_block(
            &mut sink,
            BaseAddress::new(base),
            StoredAddress::new(address),
            b"abc",
        );
        if let Some(expected) = expected {
            assert_eq!(result.unwrap_err().into_format(), expected);
            assert!(sink.bytes.is_empty());
            assert_eq!(sink.position, position);
        } else {
            result.unwrap();
            assert_eq!(sink.position, 10);
            assert_eq!(sink.bytes, [0, 0, 0, 0, b'a', b'b', b'c']);
        }
    }

    struct PositionedBytes {
        position: u64,
        bytes: Vec<u8>,
    }

    impl ByteSink for PositionedBytes {
        type Error = core::convert::Infallible;

        fn put(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
            let Self {
                position,
                bytes: output,
            } = self;
            *position += bytes.len() as u64;
            output.extend_from_slice(bytes);
            Ok(())
        }

        fn put_zeros(&mut self, length: usize) -> Result<(), Self::Error> {
            self.put(&vec![0; length])
        }

        fn position(&self) -> PhysicalAddress {
            PhysicalAddress::new(self.position)
        }
    }

    struct StoredChunks {
        chunks: Vec<Vec<u8>>,
        reads: Arc<AtomicUsize>,
    }

    impl ChunkProvider for StoredChunks {
        fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
            let Self { chunks, reads } = self;
            reads.fetch_add(1, Ordering::Relaxed);
            out.extend_from_slice(&chunks[index]);
            Ok(())
        }
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    fn narrow_reference_attributes_are_rejected_during_preparation(
        #[case] offsets: OffsetWidth,
        #[values(false, true)] attributes_only: bool,
    ) {
        let mut builder = DatasetBuilder::new("values");
        builder.with_i32_data(&[7]);
        builder.set_attr_verbatim(
            AttributeMessage {
                name: "reference".into(),
                datatype: crate::type_builders::make_object_reference_type(),
                dataspace: Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: Vec::new(),
                    max_dimensions: None,
                },
                raw_data: vec![0; 8],
                datatype_location: DatatypeLocation::Inline,
            },
            None,
        );
        let widths = FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        };
        let result = if attributes_only {
            PreparedAttributes::with_widths(widths, &builder.attrs).map(|_| ())
        } else {
            PreparedDataset::prepare_with_widths(widths, builder).map(|_| ())
        };
        assert_eq!(
            result,
            Err(FormatError::ReferenceWidthUnsupported {
                offset_size: offsets.get()
            })
        );
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    fn narrow_variable_length_providers_are_rejected_before_reads(
        #[case] offsets: OffsetWidth,
        #[values(false, true)] chunked: bool,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(CountingProvider(Arc::clone(&calls)));
        let mut builder = DatasetBuilder::new("values");
        let datatype =
            Datatype::VariableLength(VariableLengthType::sequence(crate::make_i32_type()));
        if chunked {
            builder.with_raw_chunks_lazy(
                datatype,
                &[1],
                None,
                &[1],
                NonZeroUsize::new(16).unwrap(),
                ChunkedLayoutFlags::NONE,
                None,
                vec![ChunkMeta {
                    compressed_size: 16,
                    filter_mask: 0,
                }],
                provider,
            );
        } else {
            builder.with_produced_data(datatype, &[1], 16, 16, provider);
        }
        let error = PreparedDataset::prepare_with_widths(
            FormatWidths {
                offsets,
                lengths: LengthWidth::Eight,
            },
            builder,
        )
        .err()
        .unwrap();
        assert_eq!(
            error,
            FormatError::VariableLengthProviderUnsupported {
                offset_size: offsets.get()
            }
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
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
