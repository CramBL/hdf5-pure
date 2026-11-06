//! Prepared datasets and writes published by an editor commit.
//!
//! Overwrite preparation reads existing storage and records payload identifiers or encoded chunks.
//! Application consumes the matching replacement buffers and resolves variable-length heap
//! addresses before writing.

use core::num::{NonZeroU32, NonZeroUsize};

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::AddressRange;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::ChunkedGeometry;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::LayoutVersion;
use hdf5_pure_format::__private::SlotOccupancy;
use hdf5_pure_format::__private::{DenseAttrFixup, PreparedDenseAttrs};
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::ReservationRequest;

use crate::FillTime;
use crate::PhysicalAddress;
use crate::VariableLengthType;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute_request::PreparedAttributes;
use crate::chunked_read;
use crate::chunked_read::ChunkDecoding;
use crate::chunked_write::ChunkArrayGeometry;
use crate::chunked_write::ChunkedHeader;
use crate::chunked_write::CompressedChunkSet;
#[cfg(doc)]
use crate::chunked_write::PartialEdgeFiltering;
use crate::chunked_write::VerbatimChunks;
use crate::chunked_write::{
    self, ChunkIndexKind, ChunkMeta, ChunkOptions, ChunkProvider, StorageAllocation,
};
use crate::convert::Narrow;
use crate::data_layout::ChunkedLayoutFlags;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataset_request::{
    DatasetMetadata, InPlaceDataset, PreparedBytes, PreparedDataset, PreparedElements,
    PreparedFill, ValueOverwrite,
};
use crate::dataspace::{Dataspace, DataspaceType, MaxExtent};
use crate::datatype::Datatype;
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_emission::EmissionError;
use crate::file_writer;
use crate::fill_value::FillPattern;
use crate::fill_value::PaddingFill;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_graph::ObjectId;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;
use crate::reader::OpenedSource;
use crate::shared_message::DatatypeLocation;
use crate::source::{OriginalSource, PhysicalSource, Source, StoredRead};
use crate::type_builders::{self, DatasetBuilder, VlStringStaging};
use crate::width::FormatWidths;

use super::LossyTailReencoding;
use super::append;
use super::array::ArrayIndexPlan;
use super::attributes::{DenseAttrSet, EditedAttrs};
use super::block::{BlockRequest, EncodingDependency};
use super::commit::plan::{OverwriteId, OverwritePayloads};
use super::reservation::ReservationChoice;
use super::reservation::ReservationFailure;
use super::reservation::ReservationPlan;
use super::retirement::ChunkAllocation;
use super::staging::{StagedKind, StagedOp};
use crate::file_writer::DenseAttrCreationOrder;

use super::{AppendBuilder, LOSSY_TAIL_REFUSAL, OhRegion, UNDEF, WriteEngine};

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

impl WriteEngine {
    /// Stages a prepared dataset for creation at `path` on the next [`commit`](Self::commit).
    ///
    /// The dataset name is taken from the last component of `path`. The parent group must
    /// exist in the file or be created in the same commit. [`flatten_dataset`] prepares
    /// and checks the builder before it is stored in the staged set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root, collides with an existing
    /// object, or overlaps a live buffered appender. Returns the preparation error if
    /// [`flatten_dataset`] cannot prepare `builder`, and [`Error::Format`] with
    /// [`FormatError::PartialEdgeSkipRequiresChunking`] if `builder` sets
    /// [`PartialEdgeFiltering::Skip`] on a dataset stored contiguous.
    pub(crate) fn stage_created_dataset(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        let Some((parent, leaf)) = path.split_leaf() else {
            return Err(Error::EditUnsupported("dataset path has an empty name"));
        };
        self.refuse_creation_collision(path, StagedKind::Dataset)?;
        let fd = flatten_dataset(self.context.widths(), builder, leaf)?;
        self.staged.push_dataset(parent, fd);
        Ok(())
    }

    /// Stages replacement values for an existing dataset on the next [`commit`](Self::commit).
    ///
    /// Common request invariants and overwrite capabilities are checked before staging. Shape and
    /// datatype compatibility with the committed dataset are checked during commit planning.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root or overlaps a live buffered
    /// appender, or if `builder` specifies unallocated storage, chunking, a
    /// [`PartialEdgeFiltering`] policy, attributes, fill settings, or unresolved reference targets.
    /// Returns the preparation error from [`PreparedDataset::parse`] or
    /// [`PreparedDataset::into_overwrite`] for an invalid or unsupported request.
    pub(crate) fn stage_dataset_write(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if path.split_leaf().is_none() {
            return Err(Error::EditUnsupported("cannot overwrite the root group"));
        }
        let values = PreparedDataset::parse(self.context.widths().lengths, builder)?
            .into_overwrite()?
            .into_widths(self.context.widths())?;
        self.staged
            .push(StagedOp::WriteDataset(path.clone(), values));
        Ok(())
    }

    /// Extends a staged dataset creation with the appended elements.
    ///
    /// The caller checks that its handle addresses a pending creation. If another thread
    /// commits that creation before this call, the append is staged against the committed
    /// dataset instead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`, or if
    /// an atomic batch is open while a creation is staged there. Returns the error from
    /// [`extend_staged_dataset`](Self::extend_staged_dataset) for an incompatible pending
    /// append, or from [`refuse_lossy_partial_tail`](Self::refuse_lossy_partial_tail) for a
    /// committed dataset.
    pub(crate) fn stage_dataset_append_pending(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_none() {
            self.refuse_lossy_partial_tail(path, &builder)?;
            self.staged
                .push(StagedOp::AppendDataset(path.clone(), builder));
            return Ok(());
        }
        self.refuse_mid_batch()?;
        self.extend_staged_dataset(path, &builder)
    }

    /// Places staged attribute heap collections and patches their references.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`place_vl_collections`](Self::place_vl_collections) when placing an
    /// attribute's heap collections.
    pub(super) fn resolve_dataset_attributes(&mut self, fd: &mut FlatDataset) -> Result<(), Error> {
        for (idx, collections) in std::mem::take(&mut fd.vl_attrs) {
            let addrs = self.place_vl_collections(&collections)?;
            type_builders::patch_vl_refs_with_width(
                self.context.widths().offsets,
                &mut fd.attrs[idx].raw_data,
                &addrs,
            )?;
        }
        Ok(())
    }

    /// Writes a prepared dataset and returns its object identity.
    ///
    /// The caller resolves object references and attribute collections before this call.
    /// Variable-length
    /// element references are patched after their collections have been allocated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if chunk encoding, dataset-header encoding, or allocation geometry
    /// is invalid. Returns [`Error::Io`] if writing the dataset or allocation padding fails.
    pub(super) fn write_dataset(&mut self, fd: &mut FlatDataset) -> Result<ObjectId, Error> {
        let base = self.context.addresses();
        let oh = if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
            self.build_chunked_dataset(fd)?
        } else {
            if let Some(staging) = fd.storage.take_vl_string_staging() {
                if !staging.collections.is_empty() {
                    let addrs = self.place_vl_collections(&staging.collections)?;
                    type_builders::patch_vl_refs_masked_with_width(
                        self.context.widths().offsets,
                        fd.storage.raw_mut()?,
                        &staging.patch_offsets,
                        &addrs,
                    )?;
                }
            }
            let data_addr = if fd.storage.raw().is_empty() {
                StoredAddress::new(UNDEF)
            } else {
                base.relative(PhysicalAddress::new(
                    self.alloc_or_append_typed(fd.storage.raw(), AllocationClass::Raw)?,
                ))?
            };
            let attr_info = self.place_dense_attrs_if_needed(fd)?;
            file_writer::build_dataset_oh(
                &fd.dt,
                &DatatypeLocation::Inline,
                &fd.ds,
                data_addr,
                fd.extent,
                &fd.attrs,
                attr_info.as_deref(),
                fd.fill_time,
                fd.fill.as_ref().map(PreparedFill::as_bytes),
                self.libver(),
                file_writer::HeaderOptions::PLAIN.with_widths(self.context.widths()),
            )?
        };
        self.place_object_header(&oh)
    }

    /// Plans replacement values against the committed dataset's storage.
    ///
    /// The datatype must satisfy [`overwrite_datatypes_match`], and the shape must match. Compact,
    /// contiguous, and chunked storage are supported. Non-persistent PAGE editing relocates
    /// replacements. Values that do not need chunk encoding remain in the staged buffer
    /// identified by `source`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for incompatible values or unsupported storage. Returns
    /// metadata read, parse, and chunk-encoding errors.
    pub(super) fn prepare_write(
        &self,
        addr: ObjectId,
        fd: &ValueOverwrite,
        source: OverwriteId,
    ) -> Result<WritePlan, Error> {
        let src = &self.image();
        let base = self.context.addresses();
        let widths = self.context.widths();
        let region = Self::gather_oh_messages(src, addr.header_address(), base, widths)?;

        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        let mut fill_msg: Option<(MessageType, usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p)? {
            match msg_type {
                MessageType::DATATYPE => datatype = Some((body, body_end)),
                MessageType::DATASPACE => dataspace = Some((body, body_end)),
                // libhdf5 reads the first Data Layout message of a header
                // (`H5O_msg_read_oh` in `H5Omessage.c`, HDF5 2.2.0).
                MessageType::DATA_LAYOUT if layout.is_none() => layout = Some((body, body_end)),
                MessageType::FILTER_PIPELINE => filter = Some((body, body_end)),
                // Prefer the versioned fill message, then the first message of that type.
                MessageType::FILL_VALUE
                    if !matches!(fill_msg, Some((MessageType::FILL_VALUE, ..))) =>
                {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::FILL_VALUE_OLD if fill_msg.is_none() => {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::LINK | MessageType::LINK_INFO | MessageType::SYMBOL_TABLE => {
                    has_link = true;
                }
                _ => {}
            }
            p = body_end;
        }

        if has_link {
            return Err(Error::EditUnsupported(
                "write_dataset target is a group, not a dataset",
            ));
        }
        let (dt_b, dt_e) =
            datatype.ok_or(Error::EditUnsupported("dataset header has no datatype"))?;
        let (ds_b, ds_e) =
            dataspace.ok_or(Error::EditUnsupported("dataset header has no dataspace"))?;
        let (lb, le) = layout.ok_or(Error::EditUnsupported("dataset header has no data layout"))?;

        let (disk_dt, _) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])?;
        if !overwrite_datatypes_match(&disk_dt, &fd.dt) {
            return Err(Error::EditUnsupported(
                "write_dataset datatype does not match the on-disk dataset (overwrite, not retype)",
            ));
        }
        let disk_ds = Dataspace::parse_with_length_width(&region[ds_b..ds_e], widths.lengths)?;
        if disk_ds.space_type != fd.ds.space_type
            || disk_ds.rank != fd.ds.rank
            || disk_ds.dimensions != fd.ds.dimensions
        {
            return Err(Error::EditUnsupported(
                "write_dataset shape does not match the on-disk dataset (overwrite, not reshape)",
            ));
        }

        if le - lb < 2 {
            return Err(Error::EditUnsupported("malformed data-layout message"));
        }
        let version = region[lb];
        if version != 3 && version != 4 {
            return Err(Error::EditUnsupported(
                "an unsupported data-layout version cannot be overwritten in place yet",
            ));
        }
        match region[lb + 1] {
            0 => Ok(WritePlan::Moving(MovingWrite::Compact {
                region,
                bytes: source,
            })),
            1 => {
                let layout = match DataLayout::parse_with_widths(&region[lb..le], widths) {
                    Err(FormatError::UnexpectedEof { .. }) => {
                        return Err(Error::EditUnsupported("malformed contiguous data layout"));
                    }
                    result => result?,
                };
                let DataLayout::Contiguous {
                    address: data_addr,
                    size: data_size,
                } = layout
                else {
                    return Err(FormatError::Internal(
                        "a contiguous layout has another class".into(),
                    )
                    .into());
                };

                // Deferred variable-length values must relocate. Publishing newly allocated heap
                // references
                // inside live data would make rollback release reachable storage.
                if self.fresh_page_boundary.is_none()
                    && fd.vl_string_staging().is_none()
                    && data_addr.is_some()
                    && data_size == fd.raw().len() as u64
                {
                    if let Some(data_addr) = data_addr
                        && let Ok(start) = base.resolve(data_addr)
                    {
                        if start
                            .checked_add(data_size)
                            .is_ok_and(|end| end.get() <= src.len())
                        {
                            return Ok(WritePlan::InPlace {
                                data_addr: start,
                                source,
                            });
                        }
                    }
                }

                let old_extent = data_addr
                    .filter(|_| data_size > 0)
                    .map(|address| FileRange::new(base.resolve(address)?, data_size))
                    .transpose()?;
                Ok(WritePlan::Moving(MovingWrite::Contiguous {
                    region,
                    version: LayoutVersion::try_from(version)?,
                    bytes: source,
                    old_extent,
                }))
            }
            2 => {
                let dl = DataLayout::parse_with_widths(&region[lb..le], widths)?;
                let DataLayout::Chunked { index, flags, .. } = dl else {
                    return Err(Error::EditUnsupported("dataset is not chunked"));
                };
                if !index.enumerable() {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset with a version-2 B-tree chunk index cannot be \
                         overwritten in place yet",
                    ));
                }

                let geometry =
                    ChunkedGeometry::with_offset_width(widths.offsets, &fd.dt, &disk_ds, &dl)?;

                let padding = fill_msg
                    .map_or(crate::fill_value::PaddingFill::Zero, |(mt, b, e)| {
                        crate::fill_value::PaddingFill::from_message(mt, &region[b..e])
                    });
                let pipeline_message: Option<Vec<u8>> =
                    filter.map(|(fb, fe)| region[fb..fe].to_vec());

                let pipeline = pipeline_message
                    .as_deref()
                    .map(crate::filter_pipeline::parse_filter_pipeline)
                    .transpose()?;
                if pipeline
                    .as_ref()
                    .is_some_and(|pipeline| !super::pipeline_reencodable(pipeline))
                {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset using a filter this engine cannot re-encode \
                         cannot be overwritten in place yet",
                    ));
                }
                let overwrite = ChunkedOverwrite {
                    geometry: ChunkGeometry::new(&disk_ds, geometry),
                    flags,
                    pipeline,
                    padding,
                    dt: fd.dt.clone(),
                };

                // Heap addresses affect encoded chunk lengths, so resolve them before splitting.
                if fd.vl_string_staging().is_some() {
                    return Ok(WritePlan::Moving(MovingWrite::Chunked {
                        region,
                        overwrite,
                        pipeline_message,
                        payload: ChunkPayload::Deferred { bytes: source },
                        old_addr: addr,
                    }));
                }

                let new_chunks = overwrite.prepare_chunks(fd.raw())?;

                if self.fresh_page_boundary.is_none()
                    && let Some(writes) = try_inplace_chunk_writes(
                        widths,
                        &PhysicalSource::new(&self.opened_source()).stored(),
                        &dl,
                        &disk_ds,
                        &overwrite.geometry,
                        &new_chunks,
                    )
                {
                    let writes = writes
                        .into_iter()
                        .map(|(off, b)| Ok((base.resolve(off)?, b)))
                        .collect::<Result<Vec<_>, FormatError>>()?;
                    return Ok(WritePlan::InPlaceChunks { writes, source });
                }

                Ok(WritePlan::Moving(MovingWrite::Chunked {
                    region,
                    overwrite,
                    pipeline_message,
                    payload: ChunkPayload::Encoded {
                        chunks: new_chunks,
                        source: Some(source),
                    },
                    old_addr: addr,
                }))
            }
            _ => Err(Error::EditUnsupported(
                "an unsupported data-layout class cannot be overwritten in place yet",
            )),
        }
    }

    /// Prepares a relocating append while preserving complete chunks by their index records.
    ///
    /// A partial trailing chunk is decoded and combined with the appended elements. New chunks are
    /// encoded for the grown extent, including its partial-edge filter exemption. Reads `src`
    /// without writing it. Supports single-chunk, Fixed Array, and Extensible Array indexes,
    /// subject to the dataspace's maximum first dimension.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for incompatible values, unsupported geometry or
    /// filters, a sparse chunk grid, inconsistent filter metadata, dimension overflow, or a grown
    /// extent beyond the maximum first dimension.
    /// Returns the same error for a partial tail stored under lossy filters unless the append
    /// permits re-encoding it. Propagates metadata, chunk-decoding, and encoding errors.
    pub(super) fn prepare_append(
        src: &OpenedSource<'_>,
        addr: ObjectId,
        ab: &AppendBuilder,
        widths: FormatWidths,
    ) -> Result<MovingWrite, Error> {
        let base = src.addresses();
        if ab.dt_conflict() {
            return Err(Error::AppendUnsupported(
                "append mixes element types in one builder; use one element type per \
                 append_dataset call",
            ));
        }

        let region = Self::gather_oh_messages(src, addr.header_address(), base, widths)?;

        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        let mut fill_msg: Option<(MessageType, usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p)? {
            match msg_type {
                MessageType::DATATYPE => datatype = Some((body, body_end)),
                MessageType::DATASPACE => dataspace = Some((body, body_end)),
                MessageType::DATA_LAYOUT => layout = Some((body, body_end)),
                MessageType::FILTER_PIPELINE => filter = Some((body, body_end)),
                // Prefer the versioned fill message, then the first message of that type.
                MessageType::FILL_VALUE
                    if !matches!(fill_msg, Some((MessageType::FILL_VALUE, ..))) =>
                {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::FILL_VALUE_OLD if fill_msg.is_none() => {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::LINK | MessageType::LINK_INFO | MessageType::SYMBOL_TABLE => {
                    has_link = true;
                }
                _ => {}
            }
            p = body_end;
        }
        if has_link {
            return Err(Error::AppendUnsupported(
                "append target is a group, not a dataset",
            ));
        }
        let (dt_b, dt_e) =
            datatype.ok_or(Error::AppendUnsupported("dataset header has no datatype"))?;
        let (ds_b, ds_e) =
            dataspace.ok_or(Error::AppendUnsupported("dataset header has no dataspace"))?;
        let (lb, le) = layout.ok_or(Error::AppendUnsupported(
            "dataset header has no data layout",
        ))?;

        let (disk_dt, _) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])?;
        let disk_ds = Dataspace::parse_with_length_width(&region[ds_b..ds_e], widths.lengths)?;
        let dl = DataLayout::parse_with_widths(&region[lb..le], widths)?;

        let DataLayout::Chunked { index, flags, .. } = &dl else {
            return Err(Error::AppendUnsupported(
                "append requires a chunked dataset",
            ));
        };
        if !matches!(
            index,
            ChunkIndexLayout::ExtensibleArray { .. }
                | ChunkIndexLayout::FixedArray { .. }
                | ChunkIndexLayout::SingleChunk { .. }
        ) {
            return Err(Error::AppendUnsupported(
                "append requires a single-chunk, Fixed Array, or Extensible Array index",
            ));
        }

        if disk_ds.space_type != DataspaceType::Simple || disk_ds.dimensions.len() != 1 {
            return Err(Error::AppendUnsupported(
                "append requires a rank-1 dataset in this release",
            ));
        }

        let ChunkedGeometry {
            spatial,
            element_size,
            raw_size,
            ..
        } = ChunkedGeometry::with_offset_width(widths.offsets, &disk_dt, &disk_ds, &dl)?;
        let chunk_elems = spatial[0];
        if ab.raw().len() % element_size != 0 {
            return Err(Error::AppendUnsupported(
                "appended byte length is not a whole number of elements",
            ));
        }
        match ab.elem_dt() {
            Some(expected) if !append::datatypes_match(&disk_dt, expected) => {
                return Err(Error::AppendUnsupported(
                    "append datatype does not match the on-disk dataset (wrong element \
                     type or byte order)",
                ));
            }
            Some(_) => {}
            None => {
                if !append::datatype_is_raw_appendable(&disk_dt) {
                    return Err(Error::AppendUnsupported(
                        "append_raw onto this dataset's datatype (non-little-endian, \
                         variable-length, or reference) could misencode the bytes; use a \
                         typed append",
                    ));
                }
            }
        }

        let new_elems = (ab.raw().len() / element_size) as u64;
        let current_dim0 = disk_ds.dimensions[0];
        let new_dim0 = current_dim0
            .checked_add(new_elems)
            .ok_or(Error::AppendUnsupported(
                "append would overflow the dataset dimension",
            ))?;

        let maximum = disk_ds
            .max_dimensions
            .as_deref()
            .map_or(MaxExtent::Fixed(current_dim0), |maximum| maximum[0]);
        if let MaxExtent::Fixed(limit) = maximum
            && new_dim0 > limit
        {
            return Err(Error::AppendUnsupported(
                "append exceeds the dataset's maximum first dimension",
            ));
        }

        let pipeline_message: Option<Vec<u8>> = filter.map(|(fb, fe)| region[fb..fe].to_vec());
        let has_filters = pipeline_message.is_some();
        let pipeline = match &pipeline_message {
            Some(pm) => {
                let parsed = crate::filter_pipeline::parse_filter_pipeline(pm)?;
                if !super::pipeline_reencodable(&parsed) {
                    return Err(Error::AppendUnsupported(
                        "dataset uses a filter this engine cannot re-encode",
                    ));
                }
                Some(parsed)
            }
            None => None,
        };

        if base.effective_base().get() > src.len() {
            return Err(Error::AppendUnsupported(
                "userblock base address past end-of-file",
            ));
        }
        let indexes = PhysicalSource::new(src).stored();
        // Retained records must use the same element format as the replacement index.
        let filtered_index = match *index {
            ChunkIndexLayout::ExtensibleArray {
                address: Some(address),
                ..
            } => {
                Some(crate::extensible_array::read_header(&indexes, widths, address)?.is_filtered())
            }
            ChunkIndexLayout::FixedArray {
                address: Some(address),
            } => Some(crate::fixed_array::read_header(&indexes, widths, address)?.is_filtered()),
            ChunkIndexLayout::SingleChunk { filtered, .. } => Some(filtered.is_some()),
            _ => None,
        };
        if filtered_index.is_some_and(|filtered| filtered != has_filters) {
            return Err(Error::AppendUnsupported(
                "dataset filter metadata is inconsistent (chunk-index client id \
                 disagrees with the filter pipeline)",
            ));
        }

        let infos = chunked_read::enumerate_chunks_from_source(&indexes, &dl, &disk_ds, widths)?;
        let grid = chunked_read::plan_dense_grid(infos, &disk_ds.dimensions, spatial).ok_or(
            Error::AppendUnsupported(
                "dataset has a sparse or inconsistent chunk grid; cannot append",
            ),
        )?;
        let grid_order = grid.grid_order;

        let n_full = (current_dim0 / chunk_elems).to_usize()?;
        let has_partial = current_dim0 % chunk_elems != 0;

        if has_partial
            && !flags.partial_edge_chunks_stored_raw()
            && ab.options().lossy_tail_reencoding() == LossyTailReencoding::Reject
            && let Some(pl) = &pipeline
            && !super::pipeline_lossless(pl)
        {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }

        let mut kept_chunks: Vec<ChunkRecord> = Vec::with_capacity(n_full);
        for ci in grid_order.iter().take(n_full) {
            kept_chunks.push(ChunkRecord {
                address: ci.address,
                stored_size: ci.chunk_size.get(),
                filter_mask: ci.filter_mask,
            });
        }

        let ctx = ChunkContext::from_stored_datatype(
            spatial,
            &disk_dt,
            hdf5_pure_format::__private::stored_element_size(&disk_dt, widths.offsets)?,
        );
        let mut scratch = FilterScratch::new();
        let (mut tail_raw, old_tail_allocation) = if has_partial {
            let partial = &grid_order[n_full];
            let allocation = ChunkAllocation::capture(base, partial)?;
            let live = partial.read_live_elements(
                &indexes,
                &mut scratch,
                *flags,
                pipeline.as_ref(),
                ctx,
                current_dim0,
            )?;
            (live, Some(allocation))
        } else {
            (Vec::new(), None)
        };
        tail_raw.extend_from_slice(ab.raw());

        let tail_len_elems = new_dim0 - (n_full as u64) * chunk_elems;
        let padding = fill_msg.map_or(crate::fill_value::PaddingFill::Zero, |(mt, b, e)| {
            crate::fill_value::PaddingFill::from_message(mt, &region[b..e])
        });
        let split = chunked_write::split_into_chunks(
            &tail_raw,
            &[tail_len_elems],
            spatial,
            element_size,
            padding.pattern(element_size),
        )
        .map_err(Error::Format)?;
        let new_chunks = (n_full.to_u64()..)
            .zip(split)
            .map(|(index, logical)| {
                let offset = index * chunk_elems;
                let published = ChunkDecoding::at_extent(*flags, spatial, &[new_dim0], &[offset]);
                let encoding = ChunkEncoding::for_decoding(published);
                PreparedChunk::encode(&mut scratch, pipeline.as_ref(), ctx, encoding, logical)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut grown = disk_ds.clone();
        grown.dimensions[0] = new_dim0;
        let new_dataspace_body = grown.serialize(widths.lengths.get())?;

        let (kind, slot_of_chunk, index_slots) = chunked_write::plan_index_slots(
            widths,
            &grown.dimensions,
            spatial,
            grown.max_dimensions.as_deref(),
            raw_size,
            has_filters,
            StorageAllocation::Allocated,
        )?;
        let index = AppendIndex {
            kind,
            slot_of_chunk,
            index_slots,
        };
        let chunk_dims_u32: Vec<u32> = spatial
            .iter()
            .map(|&dim| dim.narrow::<u32>())
            .collect::<Result<_, _>>()?;

        Ok(MovingWrite::AppendedChunks {
            region,
            new_dataspace_body,
            flags: *flags,
            chunk_dims_u32,
            element_size,
            raw_size,
            has_filters,
            index,
            kept_chunks,
            new_chunks,
            old_addr: addr,
            old_tail_allocation,
        })
    }

    /// Writes a relocating edit and returns its replacement object identity.
    ///
    /// Replacement buffers are consumed through their prepared identifiers. The caller publishes
    /// the parent link and reclaims superseded storage after commit publication.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if replacement layout or attribute messages cannot be
    /// written. Returns [`Error::Format`] if heap addresses, chunk encoding, header messages, or
    /// allocation geometry are invalid, and [`Error::Io`] if writing replacement storage fails.
    pub(super) fn write_moving(
        &mut self,
        overwrites: &mut OverwritePayloads,
        mw: MovingWrite,
    ) -> Result<ObjectId, Error> {
        let base = self.context.addresses();
        match mw {
            MovingWrite::Contiguous {
                region,
                version,
                bytes,
                ..
            } => {
                let raw = self.resolve_overwrite_bytes(overwrites.take(bytes))?;
                let new_data_addr = self.alloc_or_append_typed(&raw, AllocationClass::Raw)?;
                let body = DataLayout::encode_contiguous(
                    version,
                    base.relative(PhysicalAddress::new(new_data_addr))?,
                    raw.len() as u64,
                    region.widths().offsets,
                    region.widths().lengths,
                )?;
                let splice = region.splice_first(MessageType::DATA_LAYOUT)?.ok_or(
                    Error::EditUnsupported("contiguous dataset header has no data layout"),
                )?;
                let region = region.with_bytes(splice.replace_body(&body)?);
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.place_object_header(&oh)
            }
            MovingWrite::Compact { region, bytes } => {
                let raw = self.resolve_overwrite_bytes(overwrites.take(bytes))?;
                let region = super::object_header::rebuild_compact_layout_region(&region, &raw)?;
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.place_object_header(&oh)
            }
            MovingWrite::Chunked {
                region,
                overwrite,
                pipeline_message,
                payload,
                ..
            } => {
                let chunks = match payload {
                    ChunkPayload::Encoded { chunks, source: _ } => chunks,
                    ChunkPayload::Deferred { bytes } => {
                        let raw = self.resolve_overwrite_bytes(overwrites.take(bytes))?;
                        overwrite.prepare_chunks(&raw)?
                    }
                };
                let meta: Vec<ChunkMeta> = chunks
                    .iter()
                    .map(|chunk| ChunkMeta {
                        compressed_size: chunk.stored_size(),
                        filter_mask: chunk.filter_mask(),
                    })
                    .collect();
                self.write_chunked_relocatable(
                    &region,
                    &overwrite.geometry,
                    overwrite.flags,
                    pipeline_message.as_deref(),
                    &meta,
                    &PreparedChunkProvider { chunks: &chunks },
                    DenseAttrSet::default(),
                )
            }
            MovingWrite::AppendedChunks {
                region,
                new_dataspace_body,
                flags,
                chunk_dims_u32,
                element_size,
                raw_size,
                has_filters,
                index,
                kept_chunks,
                new_chunks,
                ..
            } => self.write_appended_chunks(
                &region,
                &new_dataspace_body,
                flags,
                &chunk_dims_u32,
                element_size,
                raw_size,
                has_filters,
                &index,
                &kept_chunks,
                &new_chunks,
            ),
            MovingWrite::AttrEdit { region, attrs, .. } => {
                let mut region = region;
                self.place_edited_attrs(&mut region, attrs)?;
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.place_object_header(&oh)
            }
        }
    }

    /// Writes appended chunks, a replacement index, and the grown header of a committed dataset.
    ///
    /// Retained chunks keep their stored addresses. PAGE payload and index placement follow
    /// [`Self::write_paged_appended_chunks`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`replace_dataspace_message`] and [`replace_layout_message`] when
    /// rewriting the header. Returns [`Error::Format`] if the index or layout cannot be encoded or
    /// allocated addresses are invalid. Returns [`Error::Io`] if writing chunks, the index, the
    /// header, or allocation padding fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "the append rebuild needs the header region, grown dataspace, layout flags, \
        chunk geometry, and both chunk sets; bundling them into a struct would only move the list"
    )]
    fn write_appended_chunks(
        &mut self,
        region: &OhRegion,
        new_dataspace_body: &[u8],
        flags: ChunkedLayoutFlags,
        chunk_dims_u32: &[u32],
        element_size: NonZeroUsize,
        chunk_bytes: u64,
        has_filters: bool,
        index: &AppendIndex,
        kept_chunks: &[ChunkRecord],
        new_chunks: &[PreparedChunk],
    ) -> Result<ObjectId, Error> {
        let widths = self.context.widths();
        let base = self.context.addresses();
        let (combined, index_stored) = if self.space.is_paged() && index.kind.array_kind().is_some()
        {
            self.write_paged_appended_chunks(
                index,
                chunk_bytes,
                has_filters,
                kept_chunks,
                new_chunks,
            )?
        } else {
            let mut sizing = kept_chunks.to_vec();
            sizing.extend(new_chunks.iter().map(|chunk| ChunkRecord {
                address: StoredAddress::ZERO,
                stored_size: chunk.stored_size(),
                filter_mask: chunk.filter_mask(),
            }));
            let index_len = index.encoded_len(widths, &sizing, chunk_bytes, has_filters)?;
            let mut requests = new_chunks
                .iter()
                .map(|chunk| {
                    NonZeroByteLength::new(chunk.stored_size())
                        .map(|size| {
                            BlockRequest::complete(
                                ReservationRequest::new(size, AllocationClass::Raw),
                                EncodingDependency::Independent,
                            )
                        })
                        .ok_or(FormatError::InvalidChunkGeometry(
                            "chunk allocation is empty",
                        ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(size) = NonZeroByteLength::new(index_len) {
                requests.push(BlockRequest::complete(
                    ReservationRequest::with_page_type(
                        size,
                        AllocationClass::Metadata,
                        PageType::Raw,
                    ),
                    EncodingDependency::Reservations,
                ));
            }
            let mut plan = self
                .plan_blocks(&requests)
                .map_err(ReservationFailure::into_error)?;
            let mut combined = kept_chunks.to_vec();
            for (chunk, reservation) in new_chunks.iter().zip(plan.reservations()) {
                combined.push(ChunkRecord {
                    address: base.relative(reservation.range().start())?,
                    stored_size: chunk.stored_size(),
                    filter_mask: chunk.filter_mask(),
                });
            }
            for chunk in new_chunks {
                plan.write_next(chunk.bytes())?;
            }
            let index_stored = if index_len > 0 {
                let reservation = plan.reservations().last().ok_or_else(|| {
                    FormatError::Internal("append index reservation is absent".into())
                })?;
                debug_assert_eq!(
                    reservation.encoding_dependency(),
                    EncodingDependency::Reservations
                );
                let address = base.relative(reservation.range().start())?;
                let bytes = index.build_at(widths, &combined, chunk_bytes, has_filters, address)?;
                #[cfg(test)]
                {
                    super::observations::record_index_encoding();
                    super::observations::record_index(bytes.len(), bytes.capacity());
                }
                plan.write_next(&bytes)?;
                address
            } else {
                StoredAddress::ZERO
            };
            plan.finish()?;
            (combined, index_stored)
        };

        let layout_body = index.kind.layout_message(
            widths,
            flags,
            chunk_dims_u32,
            element_size,
            &combined,
            index_stored,
            has_filters,
        )?;
        let region = replace_dataspace_message(region, new_dataspace_body)?;
        let region = replace_layout_message(&region, &layout_body)?;
        let oh = super::object_header::build_v2_object_header(&region)?;
        self.place_object_header(&oh)
    }

    /// Writes a committed dataset's new PAGE chunks and returns its stored index address.
    ///
    /// Reserves a reusable raw allocation for the complete new payload when available, or one
    /// raw allocation per prepared chunk. Plans the chosen payload and each complete array
    /// block before emission, reserving metadata through [`ArrayIndexPlan`]. With no new
    /// chunks, reserves only index metadata.
    ///
    /// On a preferred allocation, writes the encoded chunk slices through one reserved producer.
    /// Complete reservations remain unavailable after partial emission. The caller handles
    /// allocation rollback and publication within the enclosing commit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] for chunk-size arithmetic, array geometry or encoding errors, or
    /// invalid allocated addresses. Returns [`Error::Io`] if writing payload, metadata, or PAGE
    /// padding fails, or [`Error::AllocationPublicationUncertain`] if allocation publication is
    /// uncertain.
    fn write_paged_appended_chunks(
        &mut self,
        appended_index: &AppendIndex,
        chunk_bytes: u64,
        has_filters: bool,
        kept_chunks: &[ChunkRecord],
        new_chunks: &[PreparedChunk],
    ) -> Result<(Vec<ChunkRecord>, StoredAddress), Error> {
        let base = self.context.addresses();
        let kind = appended_index.kind.array_kind().ok_or_else(|| {
            FormatError::Internal("a paged array append has no array index".into())
        })?;
        let index = ArrayIndexPlan::new(
            kind,
            SlotOccupancy::Slots(&appended_index.slot_of_chunk),
            appended_index.index_slots,
            chunk_bytes,
            self.context.widths(),
            has_filters,
        )?;
        let metadata = index.requests()?;
        let length = new_chunks.iter().try_fold(0_u64, |offset, chunk| {
            offset
                .checked_add(chunk.stored_size())
                .ok_or(FormatError::OffsetOverflow {
                    offset,
                    length: chunk.stored_size(),
                })
        })?;
        let (choice, mut reservations) = if let Some(length) = NonZeroByteLength::new(length) {
            self.plan_blocks_with_fallback(
                BlockRequest::complete(
                    ReservationRequest::new(length, AllocationClass::Raw),
                    EncodingDependency::Independent,
                ),
                &metadata,
                || {
                    new_chunks
                        .iter()
                        .map(|chunk| {
                            Ok(BlockRequest::complete(
                                ReservationRequest::new(
                                    NonZeroByteLength::new(chunk.stored_size()).ok_or(
                                        FormatError::InvalidChunkGeometry(
                                            "chunk allocation is empty",
                                        ),
                                    )?,
                                    AllocationClass::Raw,
                                ),
                                EncodingDependency::Independent,
                            ))
                        })
                        .collect()
                },
            )
        } else {
            self.plan_blocks(&metadata)
                .map(|plan| (ReservationChoice::Fallback, plan))
        }
        .map_err(ReservationFailure::into_error)?;
        let mut combined = kept_chunks.to_vec();
        match choice {
            ReservationChoice::Preferred => {
                let mut address = base.relative(reservations.reservations()[0].range().start())?;
                for chunk in new_chunks {
                    combined.push(ChunkRecord {
                        address,
                        stored_size: chunk.stored_size(),
                        filter_mask: chunk.filter_mask(),
                    });
                    address = address.checked_add(chunk.stored_size())?;
                }
                reservations.write_next_with(|writer| {
                    new_chunks
                        .iter()
                        .try_for_each(|chunk| writer.write(chunk.bytes()))
                })?;
            }
            ReservationChoice::Fallback => {
                for (chunk, reservation) in new_chunks.iter().zip(reservations.reservations()) {
                    combined.push(ChunkRecord {
                        address: base.relative(reservation.range().start())?,
                        stored_size: chunk.stored_size(),
                        filter_mask: chunk.filter_mask(),
                    });
                }
                for chunk in new_chunks {
                    reservations.write_next(chunk.bytes())?;
                }
            }
        }
        let address = index.write(&appended_index.slots(&combined)?, &mut reservations)?;
        reservations.finish()?;
        Ok((combined, address))
    }

    /// Writes a prepared dataset's chunk storage and returns its object-header bytes.
    ///
    /// Chunks are encoded once before placement. The writer calls [`Self::write_array_chunk_set`]
    /// for allocated PAGE array layouts and for nonempty flat array layouts with reusable space
    /// when a fresh PAGE boundary is not pending. For PAGE layouts with an undefined index address,
    /// the writer constructs the layout and filter-pipeline messages directly. For other layouts,
    /// the writer calls [`Self::write_contiguous_chunk_set`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the filter context or pipeline is invalid, chunks cannot be
    /// encoded, the header or attributes cannot be serialized, or allocation geometry is invalid.
    /// Returns [`Error::Io`] if writing chunk storage, attributes, or allocation padding fails.
    /// Returns [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain.
    fn build_chunked_dataset(&mut self, fd: &mut FlatDataset) -> Result<Vec<u8>, Error> {
        let widths = self.context.widths();
        let chunk_dims = fd.chunk_options.resolve_chunk_dims(&fd.ds.dimensions);
        let ctx = ChunkContext::from_stored_datatype(
            &chunk_dims,
            &fd.dt,
            hdf5_pure_format::__private::stored_element_size(&fd.dt, widths.offsets)?,
        );
        let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
        let fill_time = fd.fill_time;
        let fill = FillPattern::from_value(fd.fill.as_ref().map(PreparedFill::as_bytes), elem);
        let set = chunked_write::compress_chunks(
            widths,
            fd.storage.raw(),
            &fd.ds.dimensions,
            ctx,
            &fd.chunk_options,
            fd.maxshape.as_deref(),
            fill,
            fill_time,
            fd.storage.allocation(),
        )?;
        let page = self.space.page_size();
        let unallocated = if page.is_some() {
            set.unallocated_header()?
        } else {
            None
        };
        let geometry = if page.is_some()
            || (!set.chunks().is_empty()
                && self.fresh_page_boundary.is_none()
                && self.space.reusable_sections().next().is_some())
        {
            set.array_geometry()
        } else {
            None
        };
        let ChunkedHeader {
            layout_message,
            pipeline_message,
        } = if let Some(header) = unallocated {
            header
        } else if let Some(geometry) = geometry {
            self.write_array_chunk_set(&set, geometry)?
        } else {
            self.write_contiguous_chunk_set(&set)?
        };
        let attr_info = self.place_dense_attrs_if_needed(fd)?;
        Ok(file_writer::build_chunked_dataset_oh(
            &fd.dt,
            &DatatypeLocation::Inline,
            &fd.ds,
            &layout_message,
            pipeline_message.as_deref(),
            &fd.attrs,
            attr_info.as_deref(),
            fill_time,
            fd.fill.as_ref().map(PreparedFill::as_bytes),
            file_writer::HeaderOptions::PLAIN.with_widths(self.context.widths()),
        )?)
    }

    /// Writes retained chunks and their index through one complete raw reservation.
    ///
    /// Constructs the final index and header messages at the reserved address before writing
    /// content. Submits one borrowed slice per chunk and the final index slice through the
    /// reservation plan. A zero-length region uses a scalar zero-length reservation.
    ///
    /// Complete capacity remains reserved after partial emission. The caller handles allocation
    /// rollback and reference publication within the enclosing commit.
    ///
    /// # Working memory
    ///
    /// Index preparation allocates chunk records, slot lookup storage, and encoding buffers.
    /// The final index remains allocated until header extraction. The caller retains raw input
    /// and encoded chunks, with compression scratch allocated during their preparation.
    /// Operation buffering can copy submitted slices and retain spare capacity after emission.
    /// Backend mirroring has separate storage. Physical write grouping depends on operation
    /// buffering.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] for chunk or index geometry, address conversion, encoding, or
    /// reservation length errors. Returns [`Error::Io`] if content or padding writes fail, or
    /// [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain.
    fn write_contiguous_chunk_set(
        &mut self,
        set: &CompressedChunkSet,
    ) -> Result<ChunkedHeader, Error> {
        let length = chunked_write::chunked_data_len(set)?;
        let Some(size) = NonZeroByteLength::new(length) else {
            let placement = self.reserve(0, AllocationClass::Raw)?;
            let address = self
                .context
                .addresses()
                .relative(PhysicalAddress::new(placement.address()))?;
            let header = set.header_at(address, address)?;
            self.place(placement, &[])?;
            return Ok(header);
        };
        let request = BlockRequest::complete(
            ReservationRequest::new(size, AllocationClass::Raw),
            EncodingDependency::Allocation,
        );
        let mut reservations = self
            .plan_blocks(core::slice::from_ref(&request))
            .map_err(ReservationFailure::into_error)?;
        let address = reservations
            .addresses()
            .relative(reservations.reservations()[0].range().start())?;
        let plan = chunked_write::plan_chunked_at(set, address)?;
        if plan.data_len != length {
            return Err(FormatError::Internal(
                "chunked creation length differs from its reservation".into(),
            )
            .into());
        }
        reservations.write_next_with(|writer| plan.emit(|bytes| writer.write(bytes)))?;
        reservations.finish()?;
        Ok(plan.into_header())
    }

    /// Writes retained chunks and encodes their array index at reserved addresses.
    ///
    /// The caller obtains `geometry` from `set`. For a nonempty payload, the writer prefers one raw
    /// allocation, including the index in flat files. When attempting reuse, the writer reserves each
    /// chunk separately if the preferred allocation cannot be reserved. In PAGE files and the flat
    /// fallback, the writer reserves each complete index block through [`ArrayIndexPlan`]'s metadata
    /// requests. The writer encodes chunk records from the chosen stored addresses, sizes, filter
    /// masks, and logical slots.
    ///
    /// Complete capacity remains reserved after partial emission. The caller handles allocation
    /// rollback and reference publication within the enclosing commit.
    ///
    /// # Working memory
    ///
    /// The caller retains raw input and encoded chunks, with compression scratch allocated during
    /// their preparation. Placement allocates requests, reservations, and chunk records. Index
    /// preparation allocates slot lookup storage and encoding buffers. The writer submits retained
    /// chunk slices through the reservation plan. Operation buffering and backend mirroring have
    /// separate storage, and physical write grouping depends on operation buffering.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] for invalid geometry, addresses, slot records, or header encoding.
    /// Returns [`Error::Io`] if writing payload, metadata, or padding fails, or
    /// [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain.
    fn write_array_chunk_set(
        &mut self,
        set: &CompressedChunkSet,
        geometry: ChunkArrayGeometry<'_>,
    ) -> Result<ChunkedHeader, Error> {
        let ChunkArrayGeometry {
            widths,
            kind,
            occupied,
            slots,
            chunk_bytes,
            filtered,
        } = geometry;
        let index = ArrayIndexPlan::new(
            kind,
            SlotOccupancy::Slots(&occupied),
            slots,
            chunk_bytes,
            widths,
            filtered,
        )?;
        drop(occupied);
        let length = set.chunks().iter().try_fold(0_u64, |offset, chunk| {
            offset
                .checked_add(chunk.stored_size())
                .ok_or(FormatError::OffsetOverflow {
                    offset,
                    length: chunk.stored_size(),
                })
        })?;
        let metadata = index.requests()?;
        let paged = self.space.is_paged();
        let (choice, mut reservations) =
            self.plan_created_array_storage(set, ByteLength::new(length), &metadata)?;
        if !paged && choice == ReservationChoice::Preferred {
            let address = reservations
                .addresses()
                .relative(reservations.reservations()[0].range().start())?;
            let plan = chunked_write::plan_chunked_at(set, address)?;
            if plan.data_len != reservations.reservations()[0].range().length() {
                return Err(FormatError::Internal(
                    "chunked creation length differs from its reservation".into(),
                )
                .into());
            }
            reservations.write_next_with(|writer| plan.emit(|bytes| writer.write(bytes)))?;
            reservations.finish()?;
            return Ok(plan.into_header());
        }
        let mut records = Vec::with_capacity(set.chunks().len());
        if length > 0 {
            match choice {
                ReservationChoice::Preferred => {
                    let mut cursor = reservations
                        .addresses()
                        .relative(reservations.reservations()[0].range().start())?;
                    for chunk in set.chunks() {
                        records.push(ChunkRecord {
                            address: cursor,
                            stored_size: chunk.stored_size(),
                            filter_mask: chunk.filter_mask(),
                        });
                        cursor = cursor.checked_add(chunk.stored_size())?;
                    }
                    reservations.write_next_with(|writer| {
                        set.chunks()
                            .iter()
                            .try_for_each(|chunk| writer.write(chunk.bytes()))
                    })?;
                }
                ReservationChoice::Fallback => {
                    for (chunk, reservation) in set.chunks().iter().zip(reservations.reservations())
                    {
                        records.push(ChunkRecord {
                            address: reservations
                                .addresses()
                                .relative(reservation.range().start())?,
                            stored_size: chunk.stored_size(),
                            filter_mask: chunk.filter_mask(),
                        });
                    }
                    for chunk in set.chunks() {
                        reservations.write_next(chunk.bytes())?;
                    }
                }
            }
        }
        let slots = set.index_slots(&records)?;
        let address = index.write(&slots, &mut reservations)?;
        reservations.finish()?;
        Ok(set.header(&records, address)?)
    }

    /// Reserves a compound payload or separate chunks with complete array index capacity.
    ///
    /// The caller passes the total stored chunk length as `length` and metadata requests from the
    /// same [`ArrayIndexPlan`]. The preferred raw allocation includes metadata in flat files.
    /// In PAGE files, the writer reserves metadata blocks separately.
    ///
    /// When reuse is available and a fresh PAGE boundary is not pending, the writer tries the
    /// preferred allocation through [`Self::plan_blocks_with_fallback`]. On a miss, the
    /// writer constructs independent chunk requests once. With no reusable raw space in PAGE
    /// files, or no reusable space in flat files, the writer plans the preferred shape directly.
    /// The caller handles rollback and publication for the complete plan, including partial
    /// reservation failures.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the preferred length overflows or a fallback chunk has zero
    /// stored length. Propagates reservation errors from [`Self::plan_blocks`] and
    /// [`Self::plan_blocks_with_fallback`].
    fn plan_created_array_storage(
        &mut self,
        set: &CompressedChunkSet,
        length: ByteLength,
        metadata: &[BlockRequest],
    ) -> Result<(ReservationChoice, ReservationPlan<'_>), Error> {
        let paged = self.space.is_paged();
        let dependency = if paged {
            EncodingDependency::Independent
        } else {
            EncodingDependency::Allocation
        };
        let reuse = self.fresh_page_boundary.is_none()
            && if paged {
                #[cfg(test)]
                super::observations::record_raw_capacity_scan();
                self.space.largest(AllocationClass::Raw) > ByteLength::new(0)
            } else {
                self.space.reusable_sections().next().is_some()
            };
        let preferred = if paged {
            length
        } else {
            metadata.iter().try_fold(length, |offset, request| {
                offset
                    .checked_add(request.encoded_length())
                    .ok_or(FormatError::OffsetOverflow {
                        offset: offset.get(),
                        length: request.encoded_length().get(),
                    })
            })?
        };
        if let Some(size) = preferred.nonzero()
            && reuse
        {
            self.plan_blocks_with_fallback(
                BlockRequest::complete(
                    ReservationRequest::new(size, AllocationClass::Raw),
                    dependency,
                ),
                if paged { metadata } else { &[] },
                || {
                    let mut requests = set
                        .chunks()
                        .iter()
                        .map(|chunk| {
                            Ok(BlockRequest::complete(
                                ReservationRequest::new(
                                    NonZeroByteLength::new(chunk.stored_size()).ok_or(
                                        FormatError::InvalidChunkGeometry(
                                            "chunk allocation is empty",
                                        ),
                                    )?,
                                    AllocationClass::Raw,
                                ),
                                EncodingDependency::Independent,
                            ))
                        })
                        .collect::<Result<Vec<_>, FormatError>>()?;
                    if !paged {
                        requests.extend_from_slice(metadata);
                    }
                    Ok(requests)
                },
            )
        } else {
            let mut requests = Vec::new();
            if let Some(size) = preferred.nonzero() {
                requests.push(BlockRequest::complete(
                    ReservationRequest::new(size, AllocationClass::Raw),
                    dependency,
                ));
            }
            if paged {
                requests.extend_from_slice(metadata);
            }
            self.plan_blocks(&requests)
                .map(|plan| (ReservationChoice::Preferred, plan))
        }
        .map_err(ReservationFailure::into_error)
    }

    /// Places attributes selected for dense storage during dataset preparation.
    ///
    /// Returns an Attribute Info message for dense storage, or `None` for inline attributes.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`file_writer::finalize_dense_attrs`] and
    /// [`place_dense_attrs`](Self::place_dense_attrs) when the attributes require dense storage.
    fn place_dense_attrs_if_needed(
        &mut self,
        fd: &mut FlatDataset,
    ) -> Result<Option<Vec<u8>>, Error> {
        let Some(prepared) = fd.dense_attrs.take() else {
            return Ok(None);
        };
        self.place_dense_attrs(file_writer::finalize_dense_attrs(prepared, &fd.attrs)?)
            .map(Some)
    }

    /// Writes stored chunk bytes and a replacement dataset header.
    ///
    /// For PAGE array layouts, the writer calls [`Self::write_paged_verbatim_chunks`]. For other
    /// layouts and non-PAGE files, the writer plans a contiguous payload and index region. Chunk
    /// sizes and filter masks are preserved. The writer replaces the header's layout and any
    /// rebuilt dense attribute storage, preserving its other messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the header lacks a layout message. Returns
    /// [`Error::Format`] if chunk layout, attribute storage, or header encoding is invalid, or the
    /// built storage differs in length from its allocation. Returns [`Error::Io`] if writing the
    /// replacement storage fails, or [`Error::AllocationPublicationUncertain`] if allocation
    /// publication is uncertain.
    #[expect(
        clippy::too_many_arguments,
        reason = "the verbatim rebuild combines the header, geometry, encoding, payload, and attributes"
    )]
    pub(super) fn write_chunked_relocatable(
        &mut self,
        region: &OhRegion,
        geometry: &ChunkGeometry,
        flags: ChunkedLayoutFlags,
        pipeline_message: Option<&[u8]>,
        meta: &[ChunkMeta],
        chunks: &dyn ChunkProvider,
        dense_attrs: DenseAttrSet,
    ) -> Result<ObjectId, Error> {
        let layout_message =
            self.place_verbatim_chunks(geometry, flags, pipeline_message, meta, chunks)?;
        let mut new_region = replace_layout_message(region, &layout_message)?;
        self.append_dense_attrs(&mut new_region, dense_attrs)?;
        let oh = super::object_header::build_v2_object_header(&new_region)?;
        self.place_object_header(&oh)
    }

    /// Places stored chunk bytes and their rebuilt index and returns the replacement layout body.
    ///
    /// Uses the retained geometry for sizing and emission at the session's widths. Preserves
    /// encoded chunk payloads, sizes, and filter masks.
    ///
    /// # Errors
    ///
    /// Returns planning errors from [`ChunkGeometry::plan_verbatim`], reservation and placement
    /// errors from [`Self::place_relocatable`], or chunk-provider and emission errors.
    pub(super) fn place_verbatim_chunks(
        &mut self,
        geometry: &ChunkGeometry,
        flags: ChunkedLayoutFlags,
        pipeline_message: Option<&[u8]>,
        meta: &[ChunkMeta],
        chunks: &dyn ChunkProvider,
    ) -> Result<Vec<u8>, Error> {
        let widths = self.context.widths();
        let prepared = geometry.prepare_verbatim(
            widths,
            flags,
            pipeline_message,
            meta,
            StoredAddress::ZERO,
        )?;
        if self.space.page_size().is_some() && prepared.array_kind().is_some() {
            return self.write_paged_verbatim_chunks(prepared, chunks);
        }
        let sizing = prepared.into_layout()?;
        let (_, layout_message) =
            self.place_relocatable(sizing.plan.total_len, AllocationClass::Raw, |stored_base| {
                let layout =
                    geometry.plan_verbatim(widths, flags, pipeline_message, meta, stored_base)?;
                let mut buf =
                    Vec::with_capacity(usize::try_from(layout.plan.total_len).unwrap_or(0));
                chunked_write::emit_chunked_data_verbatim(&mut buf, &layout.plan, chunks)
                    .map_err(EmissionError::into_format)?;
                Ok((buf, layout.layout_message))
            })?;
        Ok(layout_message)
    }

    /// Writes a PAGE provider payload and complete metadata index, returning its layout message.
    ///
    /// The prepared records must describe an array and a nonempty payload. The writer reserves one
    /// contiguous raw allocation and separate complete metadata blocks through [`ArrayIndexPlan`].
    /// Payload emission through [`VerbatimChunks::emit_chunks`] precedes index emission, with
    /// stored references assigned from the reservation plan's address context.
    ///
    /// Complete reservations remain unavailable after partial emission. The caller is responsible
    /// for allocation rollback and publication within the enclosing commit.
    ///
    /// # Working memory
    ///
    /// The provider scratch buffer is separate from retained source payloads, preparation and slot
    /// metadata, index encoding buffers, and image buffering or mirroring.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if array geometry is absent, the payload is empty, provider output
    /// differs from its records, or geometry, address conversion, or encoding fails. Returns
    /// provider errors through [`Error::Format`], [`Error::Io`] for failed writes, or
    /// [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain.
    fn write_paged_verbatim_chunks(
        &mut self,
        mut prepared: VerbatimChunks<'_>,
        provider: &dyn ChunkProvider,
    ) -> Result<Vec<u8>, Error> {
        let ChunkArrayGeometry {
            widths,
            kind,
            occupied,
            slots,
            chunk_bytes,
            filtered,
        } = prepared.array_geometry().ok_or_else(|| {
            FormatError::Internal("a paged verbatim array has no index geometry".into())
        })?;
        let index = ArrayIndexPlan::new(
            kind,
            SlotOccupancy::Slots(&occupied),
            slots,
            chunk_bytes,
            widths,
            filtered,
        )?;
        drop(occupied);
        let length = NonZeroByteLength::new(prepared.payload_len()).ok_or(
            FormatError::InvalidChunkGeometry("verbatim chunk storage is empty"),
        )?;
        let mut requests = vec![BlockRequest::complete(
            ReservationRequest::new(length, AllocationClass::Raw),
            EncodingDependency::Independent,
        )];
        requests.extend(index.requests()?);
        let mut reservations = self
            .plan_blocks(&requests)
            .map_err(ReservationFailure::into_error)?;
        let address = reservations
            .addresses()
            .relative(reservations.reservations()[0].range().start())?;
        prepared.assign_data_address(address)?;
        reservations.write_next_with(|writer| {
            prepared
                .emit_chunks(provider, |bytes| writer.write(bytes))
                .map_err(|error| match error {
                    EmissionError::Format(error) => Error::Format(error),
                    EmissionError::Sink(error) => error,
                })
        })?;
        let address = index.write(&prepared.index_slots()?, &mut reservations)?;
        reservations.finish()?;
        Ok(prepared.layout_message(address)?)
    }

    /// Adds elements to the prepared bytes and leading dimension of a staged creation.
    ///
    /// Chunked storage and an unlimited dimension are not required. A nonempty append to an
    /// unallocated declaration first allocates an element buffer and fills the existing elements
    /// according to its fill settings. With the `provenance` feature, digest attributes are
    /// recomputed from the extended bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for mixed or mismatched element types, unresolved
    /// per-element references, a scalar or zero-byte row, an incomplete row, arithmetic
    /// overflow, or a leading dimension beyond its declared maximum. Returns [`Error::Format`]
    /// with [`FormatError::ValueTooLargeForPlatform`] if a nonempty append would require an
    /// element buffer larger than [`isize::MAX`] bytes.
    ///
    /// # Panics
    ///
    /// Panics if the caller has not established that a dataset is staged at `path`.
    fn extend_staged_dataset(
        &mut self,
        path: &ObjectPathBuf,
        builder: &AppendBuilder,
    ) -> Result<(), Error> {
        if builder.dt_conflict() {
            return Err(Error::AppendUnsupported(
                "this append mixes element datatypes; use one element type per append",
            ));
        }
        // The caller established that a dataset is staged at this path.
        let fd = self
            .staged
            .dataset_at_mut(path)
            .expect("caller checked a dataset is staged at this path");
        if fd.storage.vl_string_staging().is_some()
            || fd.storage.reference_targets().is_some()
            || hdf5_pure_format::__private::datatype_holds_variable_length(&fd.dt)
        {
            return Err(Error::AppendUnsupported(
                "a staged variable-length-string or object-reference dataset cannot be appended \
                 to before it is committed: its per-element side table is built with the \
                 dataset. Supply every element through the builder that creates it",
            ));
        }
        if let Some(dt) = builder.elem_dt() {
            if !append::datatypes_match(&fd.dt, dt) {
                return Err(Error::AppendUnsupported(
                    "appended element datatype does not match the staged dataset's datatype",
                ));
            }
        }
        let Some((lead, inner)) = fd.ds.dimensions.split_first() else {
            return Err(Error::AppendUnsupported(
                "a scalar dataset has no dimension to append along",
            ));
        };
        let mut row_bytes = u64::from(
            hdf5_pure_format::__private::stored_element_size(
                &fd.dt,
                self.context.widths().offsets,
            )?
            .get(),
        );
        for &d in inner {
            row_bytes = row_bytes.checked_mul(d).ok_or(Error::AppendUnsupported(
                "staged dataset row size overflows",
            ))?;
        }
        if row_bytes == 0 {
            return Err(Error::AppendUnsupported(
                "a staged dataset whose rows hold no bytes cannot be appended to",
            ));
        }
        let bytes = builder.raw();
        if bytes.len() as u64 % row_bytes != 0 {
            return Err(Error::AppendUnsupported(
                "appended byte length is not a whole number of rows of the staged dataset",
            ));
        }
        let grown = lead
            .checked_add(bytes.len() as u64 / row_bytes)
            .ok_or(Error::AppendUnsupported("staged dataset length overflows"))?;
        if let Some(max) = fd
            .ds
            .max_dimensions
            .as_ref()
            .and_then(|m| m.first().copied())
            && matches!(max, MaxExtent::Fixed(limit) if limit < grown)
        {
            return Err(Error::AppendUnsupported(
                "appending would grow the staged dataset past its maximum shape",
            ));
        }
        let extent = fd
            .extent
            .checked_add(bytes.len() as u64)
            .ok_or(Error::AppendUnsupported(
                "staged dataset byte size overflows",
            ))?;
        if !bytes.is_empty() && extent > isize::MAX as u64 {
            return Err(FormatError::ValueTooLargeForPlatform {
                value: extent,
                target: "staged dataset buffer",
            }
            .into());
        }
        if fd.storage.allocation() == StorageAllocation::Unallocated && !bytes.is_empty() {
            let elem_size = hdf5_pure_format::__private::stored_element_size_usize(
                &fd.dt,
                self.context.widths().offsets,
            )?;
            fd.storage = PreparedElements::Bytes(PreparedBytes::new(
                FillPattern::new(
                    fd.fill_time,
                    fd.fill.as_ref().map(PreparedFill::as_bytes),
                    elem_size,
                )
                .buffer(fd.extent.to_usize()?)?,
            ));
        }
        if !bytes.is_empty() {
            fd.storage.raw_mut()?.extend_from_slice(bytes);
        }
        fd.extent = extent;
        fd.ds.dimensions[0] = grown;
        #[cfg(feature = "provenance")]
        fd.rebuild_provenance();
        Ok(())
    }

    /// Returns owned replacement bytes after resolving any deferred heap references in place.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`resolve_deferred_bytes`](Self::resolve_deferred_bytes) for deferred
    /// values.
    pub(super) fn resolve_overwrite_bytes(
        &mut self,
        bytes: OverwriteBytes,
    ) -> Result<Vec<u8>, Error> {
        match bytes {
            OverwriteBytes::Prepared(raw) => Ok(raw),
            OverwriteBytes::Deferred(bytes) => self.resolve_deferred_bytes(bytes),
        }
    }

    /// Places variable-length collections and patches references in the owned element buffer.
    ///
    /// The session records the new collections and schedules superseded collections for
    /// reclamation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`place_vl_collections`](Self::place_vl_collections) when allocating
    /// replacement collections. Returns [`Error::Format`] if a placed collection's address cannot
    /// be converted to a physical position.
    fn resolve_deferred_bytes(&mut self, bytes: DeferredBytes) -> Result<Vec<u8>, Error> {
        let DeferredBytes { mut raw, vlen } = bytes;
        let staging = vlen.staging;
        let superseded = self
            .vl_overwrite_heaps
            .get(&vlen.path)
            .cloned()
            .unwrap_or_default();
        let base = self.context.addresses();
        let addrs = self.place_vl_collections(&staging.collections)?;
        type_builders::patch_vl_refs_masked_with_width(
            self.context.widths().offsets,
            &mut raw,
            &staging.patch_offsets,
            &addrs,
        )?;
        let placed = addrs
            .iter()
            .zip(&staging.collections)
            .map(|(&a, c)| FileRange::new(base.resolve(StoredAddress::new(a))?, c.len() as u64))
            .collect::<Result<Vec<_>, FormatError>>()?;
        self.vl_overwrite_heaps.insert(vlen.path, placed);
        self.superseded_heaps.extend(superseded);
        Ok(raw)
    }
}

/// Distinguishes prepared element bytes from variable-length values awaiting heap allocation.
pub(super) enum OverwriteBytes {
    /// Variable-length elements whose heap collections must be allocated.
    Deferred(DeferredBytes),
    /// Element bytes ready to write.
    Prepared(Vec<u8>),
}

impl OverwriteBytes {
    /// Prepares owned element bytes for optional heap-address resolution.
    ///
    /// The dataset path is retained only for deferred values.
    pub(super) fn new(raw: Vec<u8>, staging: Option<VlStringStaging>, path: ObjectPathBuf) -> Self {
        match staging {
            Some(staging) => Self::Deferred(DeferredBytes {
                raw,
                vlen: VlenOverwrite { staging, path },
            }),
            None => Self::Prepared(raw),
        }
    }
}

/// Pairs unresolved element bytes with the collections and path required to resolve them.
pub(super) struct DeferredBytes {
    raw: Vec<u8>,
    vlen: VlenOverwrite,
}

/// Records variable-length collections and the dataset path used to track their allocation.
struct VlenOverwrite {
    staging: VlStringStaging,
    /// The dataset path used to track current and superseded heap collections.
    path: ObjectPathBuf,
}

/// Selects in-place writes or a relocating edit after commit preparation.
// The variant-size lint fires on 64-bit targets and remains below its threshold on i686.
// `expect` would fail the 32-bit builds that deny unfulfilled lint expectations.
#[allow(
    clippy::large_enum_variant,
    reason = "boxing the large variant would cost more than it saves: this enum is returned by \
              `prepare_write` and destructured at its one call site, which moves the `MovingWrite` \
              straight into a `Vec<MovingWrite>` that stores it unboxed either way -- so a box \
              here adds an allocation and a free without removing a single copy"
)]
pub(super) enum WritePlan {
    /// A contiguous range written from the identified staged replacement.
    InPlace {
        data_addr: PhysicalAddress,
        source: OverwriteId,
    },
    /// Chunk-data writes and any index rewrite that fit their existing allocations.
    InPlaceChunks {
        writes: Vec<(PhysicalAddress, Vec<u8>)>,
        source: OverwriteId,
    },
    /// An edit published by relocating the object header and updating its parent link.
    Moving(MovingWrite),
}

/// Describes a dataset edit whose object header is published at a new address.
pub(super) enum MovingWrite {
    /// A relocating append that retains complete chunks and replaces the index.
    AppendedChunks {
        region: OhRegion,
        /// The grown dataspace message with its maximum dimensions preserved.
        new_dataspace_body: Vec<u8>,
        /// Layout flags preserved in the replacement message.
        flags: ChunkedLayoutFlags,
        /// Spatial dimensions encoded in the replacement layout.
        chunk_dims_u32: Vec<u32>,
        element_size: NonZeroUsize,
        /// Checked uncompressed bytes per complete chunk, used to size index elements.
        raw_size: u64,
        /// Whether the dataset has a filter pipeline, and so whether the rebuilt index stores the
        /// filtered chunk element format.
        has_filters: bool,
        index: AppendIndex,
        /// Existing complete chunks with their stored addresses, sizes, and filter masks.
        kept_chunks: Vec<ChunkRecord>,
        /// Rewritten trailing and appended chunks, encoded for the grown extent.
        new_chunks: Vec<PreparedChunk>,
        /// The superseded object identity used for storage reclamation.
        old_addr: ObjectId,
        /// Ownership of the complete trailing allocation, retained until root publication.
        old_tail_allocation: Option<ChunkAllocation>,
    },
    /// An attribute edit that preserves dataset storage and relocates its header.
    AttrEdit {
        region: OhRegion,
        attrs: EditedAttrs,
        /// Whether the edit supersedes dense attribute indexes.
        replaces_dense_storage: bool,
    },
    /// A chunked overwrite with replacement chunk storage and index.
    Chunked {
        region: OhRegion,
        /// Checked geometry and encoding for prepared or deferred chunks.
        overwrite: ChunkedOverwrite,
        pipeline_message: Option<Vec<u8>>,
        payload: ChunkPayload,
        old_addr: ObjectId,
    },
    /// Inline values stored in a replacement object header.
    Compact {
        region: OhRegion,
        /// The replacement payload consumed during application.
        bytes: OverwriteId,
    },
    /// Element bytes written to replacement contiguous storage.
    Contiguous {
        region: OhRegion,
        version: LayoutVersion,
        /// The replacement payload consumed during application.
        bytes: OverwriteId,
        old_extent: Option<FileRange>,
    },
}

impl MovingWrite {
    /// Borrows the owned object-header records for this relocating edit.
    pub(in crate::edit) fn region_mut(&mut self) -> &mut OhRegion {
        match self {
            Self::AppendedChunks { region, .. }
            | Self::AttrEdit { region, .. }
            | Self::Chunked { region, .. }
            | Self::Compact { region, .. }
            | Self::Contiguous { region, .. } => region,
        }
    }
}

/// A replacement chunk-index kind and slot mapping for a grown dataset.
pub(super) struct AppendIndex {
    kind: ChunkIndexKind,
    /// The index slot assigned to each chunk in logical grid order.
    slot_of_chunk: Vec<u64>,
    /// The index's total capacity, including slots outside the current extent.
    index_slots: u64,
}

impl AppendIndex {
    /// Measures the replacement array index at `widths`, or returns zero for a single chunk.
    ///
    /// # Errors
    ///
    /// Returns slot-mapping errors from [`Self::slots`] or index-sizing errors from
    /// [`chunked_write::chunk_index_len`].
    fn encoded_len(
        &self,
        widths: FormatWidths,
        chunks: &[ChunkRecord],
        chunk_bytes: u64,
        has_filters: bool,
    ) -> Result<u64, FormatError> {
        self.kind.array_kind().map_or(Ok(0), |kind| {
            chunked_write::chunk_index_len(
                kind,
                &self.slots(chunks)?,
                chunk_bytes,
                widths.offsets,
                widths.lengths,
                has_filters,
            )
        })
    }

    /// Encodes the replacement array index at its stored address and file widths.
    ///
    /// Returns an empty buffer for a single-chunk layout, whose index is stored in the message.
    ///
    /// # Errors
    ///
    /// Returns slot-mapping errors from [`Self::slots`] or array-index serialization errors.
    fn build_at(
        &self,
        widths: FormatWidths,
        chunks: &[ChunkRecord],
        chunk_bytes: u64,
        has_filters: bool,
        address: StoredAddress,
    ) -> Result<Vec<u8>, FormatError> {
        self.kind.array_kind().map_or_else(
            || Ok(Vec::new()),
            |kind| {
                kind.build_at(
                    widths,
                    &self.slots(chunks)?,
                    chunk_bytes,
                    has_filters,
                    address,
                )
            },
        )
    }

    /// Borrows chunk records under the planned logical-slot mapping.
    ///
    /// # Errors
    ///
    /// Returns mapping and capacity errors from [`IndexSlots::new`].
    fn slots<'a>(&'a self, chunks: &'a [ChunkRecord]) -> Result<IndexSlots<'a>, FormatError> {
        IndexSlots::new(chunks, &self.slot_of_chunk, self.index_slots)
    }
}

/// Distinguishes encoded chunks from element bytes awaiting heap-address resolution.
pub(super) enum ChunkPayload {
    /// Variable-length elements resolved before chunk splitting and encoding.
    Deferred { bytes: OverwriteId },
    /// Chunks encoded during preparation to determine whether existing storage can be reused.
    Encoded {
        chunks: Vec<PreparedChunk>,
        /// The original input's identifier, cleared when its buffer is released after preflight.
        source: Option<OverwriteId>,
    },
}

/// A staged dataset's values, metadata, and unresolved references.
pub(super) struct FlatDataset {
    pub(super) name: LinkNameBuf,
    pub(super) dt: crate::datatype::Datatype,
    pub(super) ds: Dataspace,
    pub(super) storage: PreparedElements,
    /// The logical byte length of all elements, including unwritten values.
    extent: u64,
    pub(super) attrs: Vec<crate::attribute::AttributeMessage>,
    /// Storage options used by the shared chunk writer.
    pub(super) chunk_options: ChunkOptions,
    /// Maximum dimensions, also recorded in the dataspace.
    pub(super) maxshape: Option<Vec<MaxExtent>>,
    /// Attribute indices paired with global-heap collections awaiting placement.
    pub(super) vl_attrs: Vec<(usize, Vec<Vec<u8>>)>,
    /// Prepared dense storage, or `None` for inline attributes or after placement.
    ///
    /// Reference patching and digest updates preserve serialized lengths.
    pub(super) dense_attrs: Option<PreparedDenseAttrs>,
    /// One fill element checked against the dataset datatype, or `None` for the default.
    pub(super) fill: Option<PreparedFill>,
    /// The fill policy after default normalization.
    pub(super) fill_time: FillTime,
    /// Inputs used to rebuild generated attributes after the prepared bytes change.
    #[cfg(feature = "provenance")]
    provenance: Option<StagedProvenance>,
}

impl FlatDataset {
    /// Recomputes generated attributes over the prepared element bytes.
    ///
    /// Attribute order and serialized lengths are preserved, including the dense-storage decision.
    #[cfg(feature = "provenance")]
    pub(super) fn rebuild_provenance(&mut self) {
        let Some(prov) = &self.provenance else {
            return;
        };
        self.attrs.truncate(prov.attrs_start);
        let rebuilt = prov.inputs.build_attrs(self.storage.raw());
        self.attrs.extend(rebuilt);
    }
}

/// Records attribute-generation inputs and the attribute range replaced when values change.
#[cfg(feature = "provenance")]
struct StagedProvenance {
    /// The inputs used to construct generated attributes.
    inputs: crate::provenance::Provenance,
    /// The first generated attribute, following caller-supplied attributes.
    attrs_start: usize,
}

/// Prepares a dataset builder for storage in the pending edit set.
///
/// Variable-length collections and object-reference targets remain unresolved until commit
/// application. Unallocated declarations retain the logical byte length derived from their shape
/// and stored element size at `widths`. Converts canonical variable-length values, fill values,
/// and attribute storage before sizing. The `provenance` feature computes digest attributes over
/// the prepared values.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] for providers, committed datatypes, unavailable filters, a
/// link name too large to encode, or chunked or extensible datasets with deferred element
/// references.
///
/// Returns [`Error::Format`] for an invalid request, as described by [`PreparedDataset::parse`],
/// or an incompatible filtering policy, as described by [`PreparedDataset::into_in_place`].
///
/// Propagates field-width, variable-length conversion, chunk-size, filter-pipeline, and
/// dense-attribute preparation errors.
pub(super) fn flatten_dataset(
    widths: FormatWidths,
    db: DatasetBuilder,
    name: LinkNameBuf,
) -> Result<FlatDataset, Error> {
    let InPlaceDataset { metadata, storage } =
        PreparedDataset::parse(widths.lengths, db)?.into_in_place()?;
    let DatasetMetadata {
        dt,
        element_size: canonical_element_size,
        ds,
        extent: canonical_extent,
        attrs:
            PreparedAttributes {
                messages: mut attrs,
                collections: vl_attrs,
            },
        chunk_options,
        mut fill,
        fill_time,
        committed_datatype_target: _,
        #[cfg(feature = "provenance")]
        provenance,
    } = metadata;
    ds.serialize(widths.lengths.get())?;
    let storage = storage.into_widths(widths, &dt)?;
    let stored_element_size =
        hdf5_pure_format::__private::stored_element_size(&dt, widths.offsets)?;
    let elem_size = stored_element_size.narrow::<NonZeroUsize>()?;
    let extent = (canonical_extent / canonical_element_size.get() as u64)
        .checked_mul(u64::from(stored_element_size.get()))
        .ok_or(FormatError::DatasetSizeOverflow {
            size: canonical_extent,
            dimension: u64::from(stored_element_size.get()),
        })?;
    if let Some(fill) = &mut fill {
        fill.narrow_variable_length(&dt, widths.offsets)?;
    }
    file_writer::prepare_attribute_widths(widths, &mut attrs, &vl_attrs)?;
    let maxshape = ds.max_dimensions.clone();
    let fill_time = fill_time.unwrap_or_default();
    if chunk_options.is_chunked() || maxshape.is_some() {
        chunk_options
            .refuse_unavailable_filters()
            .map_err(Error::EditUnsupported)?;
        let chunk_dims = chunk_options.resolve_chunk_dims(&ds.dimensions);
        let ctx = ChunkContext::from_stored_datatype(&chunk_dims, &dt, stored_element_size);
        chunked_read::bounded_chunk_len(&chunk_dims, elem_size)?;
        chunk_options.build_pipeline(
            &ctx,
            FillPattern::from_value(fill.as_ref().map(PreparedFill::as_bytes), elem_size),
        )?;
    }

    if !chunk_options.is_chunked() && maxshape.is_none() {
        DataLayout::encode_contiguous(
            LayoutVersion::Three,
            StoredAddress::new(0),
            extent,
            widths.offsets,
            widths.lengths,
        )?;
    }
    let mut sized = file_writer::make_link(name.as_str(), StoredAddress::new(0));
    sized.creation_order = Some(0);
    if sized.serialize(widths.offsets)?.len() > OBJECT_HEADER_MESSAGE_MAX {
        return Err(Error::EditUnsupported(
            "dataset name is too long to encode as a link message",
        ));
    }

    #[cfg(feature = "provenance")]
    let provenance = provenance.map(|inputs| {
        let attrs_start = attrs.len();
        attrs.extend(inputs.build_attrs(storage.raw()));
        StagedProvenance {
            inputs,
            attrs_start,
        }
    });
    let attrs_are_dense =
        crate::file_writer::needs_dense_attrs_with_widths(widths, &attrs).map_err(Error::Format)?;
    let dense_attrs = if attrs_are_dense {
        let fixups = vl_attrs
            .iter()
            .map(|(attribute, _)| DenseAttrFixup::GlobalHeap {
                attribute: *attribute,
            });
        #[cfg(feature = "provenance")]
        let fixups = fixups.chain(provenance.iter().map(|prov| DenseAttrFixup::Payload {
            attribute: prov.attrs_start,
        }));
        Some(PreparedDenseAttrs::with_widths(
            widths,
            &attrs,
            DenseAttrCreationOrder::Untracked,
            fixups,
        )?)
    } else {
        None
    };

    Ok(FlatDataset {
        name,
        dt,
        ds,
        storage,
        extent,
        attrs,
        chunk_options,
        maxshape,
        vl_attrs,
        dense_attrs,
        fill,
        fill_time,
        #[cfg(feature = "provenance")]
        provenance,
    })
}

/// Returns a copy of `region` whose first Data Layout message has `new_layout_body` as its body.
///
/// The rewritten record has no flag set and, where the region tracks creation order, a creation
/// index of 0. Every other record is copied unchanged, a later Data Layout message included.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if `region` lacks a Data Layout message or if the body of a
/// record it reads runs past the end of `region`. Returns [`Error::Format`] if `new_layout_body`
/// is longer than [`OBJECT_HEADER_MESSAGE_MAX`].
pub(super) fn replace_layout_message(
    region: &OhRegion,
    new_layout_body: &[u8],
) -> Result<OhRegion, Error> {
    let splice = region
        .splice_first(MessageType::DATA_LAYOUT)?
        .ok_or(Error::EditUnsupported(
            "chunked dataset header has no data-layout message to relocate",
        ))?;
    Ok(region.with_bytes(splice.replace_record(MessageFlags::NONE, 0, new_layout_body)?))
}

/// Returns a copy of `region` whose first Dataspace message has `new_dataspace_body` as its body.
///
/// The rewritten record has no flag set and, where the region tracks creation order, a creation
/// index of 0. Every other record is copied unchanged, a later Dataspace message included.
///
/// # Errors
///
/// Returns [`Error::AppendUnsupported`] if `region` lacks a Dataspace message,
/// [`Error::EditUnsupported`] if the body of a record it reads runs past the end of `region`, and
/// [`Error::Format`] if `new_dataspace_body` is longer than [`OBJECT_HEADER_MESSAGE_MAX`].
pub(super) fn replace_dataspace_message(
    region: &OhRegion,
    new_dataspace_body: &[u8],
) -> Result<OhRegion, Error> {
    let splice = region
        .splice_first(MessageType::DATASPACE)?
        .ok_or(Error::AppendUnsupported(
            "dataset header has no dataspace message to grow",
        ))?;
    Ok(region.with_bytes(splice.replace_record(MessageFlags::NONE, 0, new_dataspace_body)?))
}

/// Compares datatype structure while ignoring byte order for one-byte fixed-point types.
///
/// Compound members and base types are compared recursively. All other fields must match.
pub(super) fn overwrite_datatypes_match(disk: &Datatype, staged: &Datatype) -> bool {
    match (disk, staged) {
        (Datatype::FixedPoint(disk), Datatype::FixedPoint(staged))
            if disk.size().get() == 1 && staged.size().get() == 1 =>
        {
            disk.layout() == staged.layout()
        }
        (
            Datatype::Compound {
                size: disk_size,
                members: disk_members,
            },
            Datatype::Compound {
                size: staged_size,
                members: staged_members,
            },
        ) => {
            disk_size == staged_size
                && disk_members.len() == staged_members.len()
                && disk_members
                    .iter()
                    .zip(staged_members)
                    .all(|(disk, staged)| {
                        disk.name == staged.name
                            && disk.byte_offset == staged.byte_offset
                            && overwrite_datatypes_match(&disk.datatype, &staged.datatype)
                    })
        }
        (
            Datatype::Enumeration {
                size: disk_size,
                base_type: disk_base,
                members: disk_members,
            },
            Datatype::Enumeration {
                size: staged_size,
                base_type: staged_base,
                members: staged_members,
            },
        ) => {
            disk_size == staged_size
                && disk_members == staged_members
                && overwrite_datatypes_match(disk_base, staged_base)
        }
        (
            Datatype::VariableLength(VariableLengthType::String(disk)),
            Datatype::VariableLength(VariableLengthType::String(staged)),
        ) => {
            disk.padding() == staged.padding()
                && disk.charset() == staged.charset()
                && overwrite_datatypes_match(disk.base_type(), staged.base_type())
        }
        (
            Datatype::VariableLength(VariableLengthType::Sequence {
                element_type: disk_base,
            }),
            Datatype::VariableLength(VariableLengthType::Sequence {
                element_type: staged_base,
            }),
        ) => overwrite_datatypes_match(disk_base, staged_base),
        (
            Datatype::Array {
                base_type: disk_base,
                dimensions: disk_dimensions,
            },
            Datatype::Array {
                base_type: staged_base,
                dimensions: staged_dimensions,
            },
        ) => {
            disk_dimensions == staged_dimensions
                && overwrite_datatypes_match(disk_base, staged_base)
        }
        _ => disk == staged,
    }
}

/// Pairs checked chunk geometry with the encoding needed for an overwrite.
pub(super) struct ChunkedOverwrite {
    pub(super) geometry: ChunkGeometry,
    /// Layout flags used for partial-edge encoding and the replacement layout.
    pub(super) flags: ChunkedLayoutFlags,
    pub(super) pipeline: Option<FilterPipeline>,
    /// Fill bytes for elements beyond the current extent.
    pub(super) padding: PaddingFill,
    /// The datatype used to build the filter context.
    pub(super) dt: Datatype,
}

impl ChunkedOverwrite {
    /// Splits replacement values and encodes chunks for the current dataset extent.
    ///
    /// Partial edge chunks use the dataset's fill bytes. Layout flags determine whether those
    /// chunks
    /// are filtered.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] for unreadable required fill metadata, invalid filter context, or
    /// chunk-encoding errors.
    pub(super) fn prepare_chunks(&self, raw: &[u8]) -> Result<Vec<PreparedChunk>, Error> {
        let Self {
            geometry,
            flags,
            pipeline,
            padding,
            dt,
        } = self;
        let ChunkGeometry {
            shape,
            spatial: chunk_dims,
            element_size,
            ..
        } = geometry;
        Ok(PreparedChunk::encode_dataset(
            &mut FilterScratch::new(),
            pipeline.as_ref(),
            ChunkContext::from_stored_datatype(
                chunk_dims,
                dt,
                NonZeroU32::new(element_size.get().narrow::<u32>()?).ok_or_else(|| {
                    FormatError::Internal("chunk geometry has zero-sized elements".into())
                })?,
            ),
            *flags,
            shape,
            padding.pattern(*element_size),
            raw,
        )?)
    }
}

/// Owns chunk geometry parsed by [`ChunkedGeometry`] and the dataset's current shape.
///
/// The element size and spatial dimensions are nonzero, and `raw_size` is the checked uncompressed
/// byte count of a complete chunk.
pub(super) struct ChunkGeometry {
    shape: Vec<u64>,
    spatial: Vec<u64>,
    element_size: NonZeroUsize,
    raw_size: u64,
    maxshape: Option<Vec<MaxExtent>>,
}

impl ChunkGeometry {
    /// Copies parsed geometry and the dataset shape into an owned write plan.
    pub(super) fn new(dataspace: &Dataspace, geometry: ChunkedGeometry<'_>) -> Self {
        let ChunkedGeometry {
            spatial,
            element_size,
            raw_size,
            maxshape,
        } = geometry;
        Self {
            shape: dataspace.dimensions.clone(),
            spatial: spatial.to_vec(),
            element_size,
            raw_size,
            maxshape: maxshape.map(<[MaxExtent]>::to_vec),
        }
    }

    /// Plans stored chunk placement and its layout at `address` without decoding payloads.
    ///
    /// Uses this geometry's current and maximum shapes, spatial dimensions, and stored element
    /// size with the supplied file widths and filter metadata.
    ///
    /// # Errors
    ///
    /// Returns layout-planning errors from [`chunked_write::plan_chunked_data_verbatim`].
    pub(super) fn plan_verbatim(
        &self,
        widths: FormatWidths,
        flags: ChunkedLayoutFlags,
        pipeline_message: Option<&[u8]>,
        meta: &[ChunkMeta],
        address: StoredAddress,
    ) -> Result<chunked_write::VerbatimLayout, FormatError> {
        self.prepare_verbatim(widths, flags, pipeline_message, meta, address)?
            .into_layout()
    }

    fn prepare_verbatim<'a>(
        &self,
        widths: FormatWidths,
        flags: ChunkedLayoutFlags,
        pipeline_message: Option<&'a [u8]>,
        meta: &[ChunkMeta],
        address: StoredAddress,
    ) -> Result<VerbatimChunks<'a>, FormatError> {
        let Self {
            shape,
            spatial,
            element_size,
            raw_size: _,
            maxshape,
        } = self;
        VerbatimChunks::new(
            widths,
            meta,
            shape,
            spatial,
            *element_size,
            flags,
            pipeline_message,
            address,
            maxshape.as_deref(),
        )
    }
}

/// Returns writes for chunks that fit their existing dense-grid storage.
///
/// Filter masks must match. Equal stored lengths leave the index unchanged. Smaller chunks require
/// an index that [`try_rebuild_index_in_place`] can reproduce within its original allocation.
/// Returns `None` for incompatible storage, read errors, overlapping writes, or out-of-bounds
/// spans.
fn try_inplace_chunk_writes(
    widths: FormatWidths,
    src: &(impl StoredRead + ?Sized),
    layout: &DataLayout,
    ds: &Dataspace,
    geometry: &ChunkGeometry,
    new_chunks: &[PreparedChunk],
) -> Option<Vec<(StoredAddress, Vec<u8>)>> {
    let infos = chunked_read::enumerate_chunks_from_source(src, layout, ds, widths).ok()?;
    let grid = chunked_read::plan_dense_grid(infos, &geometry.shape, &geometry.spatial)?;
    if grid.grid_order.len() != new_chunks.len() {
        return None;
    }
    let mut writes = Vec::with_capacity(new_chunks.len() + 1);
    let mut spans = Vec::with_capacity(new_chunks.len() + 1);
    let mut any_shrunk = false;
    for (ci, chunk) in grid.grid_order.iter().zip(new_chunks) {
        if ci.filter_mask != chunk.filter_mask() {
            return None;
        }
        let new_len = chunk.stored_size();
        let slot = ci.chunk_size.get();
        if new_len > slot {
            return None;
        }
        if new_len < slot {
            any_shrunk = true;
        }
        let range = AddressRange::new(ci.address, new_len).ok()?;
        if range.end().get() > src.len() {
            return None;
        }
        writes.push((ci.address, chunk.bytes().to_vec()));
        spans.push(range);
    }

    if any_shrunk {
        let (index_addr, index_bytes) = try_rebuild_index_in_place(
            widths,
            src,
            layout,
            ds,
            geometry,
            &grid.grid_order,
            new_chunks,
        )?;
        spans.push(AddressRange::new(index_addr, index_bytes.len() as u64).ok()?);
        writes.push((index_addr, index_bytes));
    }

    spans.sort_unstable_by_key(|range| range.start());
    if spans
        .iter()
        .any(|range| range.is_empty() || range.end().get() > src.len())
        || spans.windows(2).any(|pair| !pair[0].precedes(pair[1]))
    {
        return None;
    }
    Some(writes)
}

/// Returns a replacement index within its original contiguous allocation.
///
/// Only fixed-array and extensible-array indexes are rebuilt. The shared index writers use the
/// checked uncompressed chunk byte count. The rebuilt length must equal the original allocation.
/// Returns `None` for incompatible storage or read and build errors.
///
/// An in-place index rewrite can tear on a failed write or crash.
fn try_rebuild_index_in_place(
    widths: FormatWidths,
    src: &(impl StoredRead + ?Sized),
    layout: &DataLayout,
    ds: &Dataspace,
    geometry: &ChunkGeometry,
    grid_order: &[crate::chunked_read::ChunkInfo],
    new_chunks: &[PreparedChunk],
) -> Option<(StoredAddress, Vec<u8>)> {
    let ChunkGeometry {
        spatial, raw_size, ..
    } = geometry;
    let raw_size = *raw_size;
    let DataLayout::Chunked { index, .. } = layout else {
        return None;
    };
    let index_addr = index.address()?;
    let written: Vec<ChunkRecord> = grid_order
        .iter()
        .zip(new_chunks)
        .map(|(ci, chunk)| ChunkRecord {
            address: ci.address,
            stored_size: chunk.stored_size(),
            filter_mask: chunk.filter_mask(),
        })
        .collect();
    // Map the current dense grid onto the index slots of the maximum shape.
    let (_, slot_of_chunk, index_slots) = crate::chunked_write::plan_index_slots(
        widths,
        &ds.dimensions,
        spatial,
        ds.max_dimensions.as_deref(),
        raw_size,
        true,
        crate::chunked_write::StorageAllocation::Allocated,
    )
    .ok()?;
    let slots = IndexSlots::new(&written, &slot_of_chunk, index_slots).ok()?;
    let new_index = match index {
        ChunkIndexLayout::FixedArray { .. } => hdf5_pure_format::__private::build_fixed_array_at(
            &slots,
            raw_size,
            widths.offsets,
            widths.lengths,
            true,
            index_addr,
        )
        .ok()?,
        ChunkIndexLayout::ExtensibleArray { .. } => {
            hdf5_pure_format::__private::build_extensible_array_at(
                &slots,
                raw_size,
                widths.offsets,
                widths.lengths,
                true,
                index_addr,
            )
            .ok()?
        }
        ChunkIndexLayout::SingleChunk { .. }
        | ChunkIndexLayout::Implicit { .. }
        | ChunkIndexLayout::BTreeV1 { .. }
        | ChunkIndexLayout::BTreeV2 { .. } => return None,
    };

    // Require the original index allocation to be contiguous and exactly the rebuilt length.
    let mut spans = crate::chunked_read::chunk_index_spans_from_source(src, layout, widths).ok()?;
    if spans.is_empty() {
        return None;
    }
    spans.sort_unstable_by_key(|range| range.start());
    if spans[0].start() != index_addr {
        return None;
    }
    let mut end = index_addr;
    for range in &spans {
        if range.start() != end {
            return None; // a gap means the index is not contiguous
        }
        end = range.end();
    }
    if Some(new_index.len() as u64) != end.checked_distance_from(index_addr) {
        return None;
    }
    index_addr
        .get()
        .checked_add(new_index.len() as u64)
        .filter(|&e| e <= src.len())?;
    Some((index_addr, new_index))
}

/// Provides encoded chunk bytes from a relocating overwrite in dense row-major order.
struct PreparedChunkProvider<'a> {
    chunks: &'a [PreparedChunk],
}

impl ChunkProvider for PreparedChunkProvider<'_> {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let chunk = self.chunks.get(index).ok_or_else(|| {
            FormatError::Internal(format!(
                "a verbatim plan asked for chunk {index} of {} prepared chunks",
                self.chunks.len()
            ))
        })?;
        out.extend_from_slice(chunk.bytes());
        Ok(())
    }
}

#[cfg(test)]
pub(in crate::edit) mod tests;
