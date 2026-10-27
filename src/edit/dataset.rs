//! Prepared datasets and writes published by an editor commit.
//!
//! Preparation reads existing storage without mutating it. The write plans distinguish in-place
//! bytes from relocating writes and defer variable-length address resolution until allocation.

use std::borrow::Cow;

use core::num::NonZeroUsize;

use hdf5_pure_format::__private::ChunkIndexInfo;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::ChunkedGeometry;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::PageType;

use crate::FillTime;
use crate::VariableLengthType;
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_read;
use crate::chunked_read::ChunkDecoding;
#[cfg(doc)]
use crate::chunked_write::PartialEdgeFiltering;
use crate::chunked_write::{self, ChunkMeta, ChunkOptions, ChunkProvider, StorageAllocation};
use crate::convert::Narrow;
use crate::data_layout::ChunkedLayoutFlags;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataspace::{Dataspace, DataspaceType, Extent, MaxExtent};
use crate::datatype::Datatype;
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_emission::EmissionError;
use crate::file_writer::{self, LENGTH_SIZE, LENGTH_WIDTH, OFFSET_WIDTH};
use crate::fill_value::FillPattern;
use crate::fill_value::PaddingFill;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;
use crate::shared_message::DatatypeLocation;
use crate::source::{BaseOffsetSource, Source};
use crate::type_builders::{self, DatasetBuilder, ObjectRefPatch, VlStringStaging};
use crate::width::FormatWidths;

use super::LossyTailReencoding;
use super::append;
use super::attributes::{DenseAttrSet, EditedAttrs};
use super::staging::{StagedKind, StagedOp};
use crate::file_writer::DenseAttrCreationOrder;

use super::{AppendBuilder, LOSSY_TAIL_REFUSAL, OhRegion, Placement, UNDEF, WriteEngine};

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
        let fd = flatten_dataset(builder, leaf)?;
        fd.chunk_options
            .validate_partial_edge_filtering(fd.maxshape.as_deref())?;
        self.staged.push_dataset(parent, fd);
        Ok(())
    }

    /// Stages replacement values for an existing dataset on the next [`commit`](Self::commit).
    ///
    /// The replacement is prepared by [`flatten_dataset`]. Shape and datatype
    /// compatibility with the committed dataset are checked during the commit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root or overlaps a live buffered
    /// appender, or if `builder` specifies unallocated storage, chunking, a
    /// [`PartialEdgeFiltering`] policy, attributes, fill settings, or unresolved reference targets.
    /// Returns the preparation error if [`flatten_dataset`] cannot prepare `builder`.
    pub(crate) fn stage_dataset_write(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        let Some((_, leaf)) = path.split_leaf() else {
            return Err(Error::EditUnsupported("cannot overwrite the root group"));
        };
        let fd = flatten_dataset(builder, leaf)?;
        Self::refuse_unsupported_overwrite(&fd)?;
        self.staged.push(StagedOp::WriteDataset(path.clone(), fd));
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
            type_builders::patch_vl_refs(&mut fd.attrs[idx].raw_data, &addrs);
        }
        Ok(())
    }

    /// Writes a prepared dataset and returns its object-header address.
    ///
    /// The caller resolves object references and attribute collections before this call.
    /// Variable-length
    /// element references are patched after their collections have been allocated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if chunk encoding, dataset-header encoding, or allocation geometry
    /// is invalid. Returns [`Error::Io`] if writing the dataset or allocation padding fails.
    pub(super) fn write_dataset(&mut self, fd: &mut FlatDataset) -> Result<u64, Error> {
        let base = self.context.addresses();
        let oh = if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
            self.build_chunked_dataset(fd)?
        } else {
            if let Some(staging) = fd.vl_string_staging.take() {
                if !staging.collections.is_empty() {
                    let addrs = self.place_vl_collections(&staging.collections)?;
                    type_builders::patch_vl_refs_masked(
                        &mut fd.raw,
                        &staging.patch_offsets,
                        &addrs,
                    );
                }
            }
            let data_addr = if fd.raw.is_empty() {
                StoredAddress::new(UNDEF)
            } else {
                base.relative(self.alloc_or_append_typed(&fd.raw, AllocationClass::Raw)?)?
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
                fd.fill_time.unwrap_or_default(),
                fd.fill.as_deref(),
                self.libver(),
            )?
        };
        self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
    }

    /// Plans replacement values against the committed dataset's storage.
    ///
    /// The datatype must satisfy [`overwrite_datatypes_match`], and the shape must match. Compact,
    /// contiguous, and chunked storage are supported. Non-persistent PAGE editing relocates
    /// replacements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for incompatible values or unsupported storage. Returns
    /// metadata read, parse, and chunk-encoding errors.
    pub(super) fn prepare_write(
        &self,
        addr: u64,
        fd: &FlatDataset,
        path: &ObjectPathBuf,
    ) -> Result<WritePlan, Error> {
        let src = &self.image();
        let base = self.context.addresses();
        let widths = self.context.widths();
        // `stage_dataset_write` checks these options before storing a prepared write.
        debug_assert!(
            Self::refuse_unsupported_overwrite(fd).is_ok(),
            "a staged write reached prepare_write without refuse_unsupported_overwrite"
        );

        let region = Self::gather_oh_messages(src, addr, base)?;

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
                bytes: staged_bytes(fd, path),
            })),
            1 => {
                if le - lb < 18 {
                    return Err(Error::EditUnsupported("malformed contiguous data layout"));
                }
                let addr_off = lb + 2;
                let data_addr = StoredAddress::new(u64::from_le_bytes(
                    region[addr_off..addr_off + 8].try_into().unwrap(),
                ));
                let data_size = u64::from_le_bytes(region[lb + 10..lb + 18].try_into().unwrap());

                // Deferred variable-length values must relocate. Publishing newly allocated heap
                // references
                // inside live data would make rollback release reachable storage.
                if self.fresh_page_boundary.is_none()
                    && fd.vl_string_staging.is_none()
                    && !data_addr.is_undefined(OFFSET_WIDTH)
                    && data_size == fd.raw.len() as u64
                {
                    if let Ok(start) = base.absolute(data_addr) {
                        if start.checked_add(data_size).is_some_and(|e| e <= src.len()) {
                            return Ok(WritePlan::InPlace {
                                data_addr: start,
                                bytes: fd.raw.clone(),
                            });
                        }
                    }
                }

                let old_extent = if !data_addr.is_undefined(OFFSET_WIDTH) && data_size > 0 {
                    Some((base.absolute(data_addr)?, data_size))
                } else {
                    None
                };
                Ok(WritePlan::Moving(MovingWrite::Contiguous {
                    region,
                    addr_off,
                    bytes: staged_bytes(fd, path),
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

                let geometry = ChunkedGeometry::new(&fd.dt, &disk_ds, &dl)?;

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
                if let Some(staging) = &fd.vl_string_staging {
                    return Ok(WritePlan::Moving(MovingWrite::Chunked {
                        region,
                        overwrite,
                        pipeline_message,
                        payload: ChunkPayload::Deferred {
                            bytes: DeferredBytes::new(&fd.raw, staging, path),
                        },
                        old_addr: addr,
                    }));
                }

                let new_chunks = overwrite.prepare_chunks(&fd.raw)?;

                if self.fresh_page_boundary.is_none()
                    && let Some(writes) = try_inplace_chunk_writes(
                        &BaseOffsetSource {
                            inner: src,
                            base: base.effective_base(),
                        },
                        &dl,
                        &disk_ds,
                        &overwrite.geometry,
                        &new_chunks,
                    )
                {
                    let writes = writes
                        .into_iter()
                        .map(|(off, b)| Ok((base.absolute(StoredAddress::new(off))?, b)))
                        .collect::<Result<Vec<_>, FormatError>>()?;
                    return Ok(WritePlan::InPlaceChunks { writes });
                }

                Ok(WritePlan::Moving(MovingWrite::Chunked {
                    region,
                    overwrite,
                    pipeline_message,
                    payload: ChunkPayload::Encoded(new_chunks),
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
    /// encoded for the grown extent, including its partial-edge filter exemption. This call reads
    /// `src`
    /// without writing it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for incompatible values, unsupported geometry or
    /// filters, a sparse chunk grid, inconsistent filter metadata, or dimension overflow.
    /// Returns the same error for a partial tail stored under lossy filters unless the append
    /// permits re-encoding it. Propagates metadata, chunk-decoding, and encoding errors.
    pub(super) fn prepare_append<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        ab: &AppendBuilder,
        base: AddressContext,
        widths: FormatWidths,
    ) -> Result<MovingWrite, Error> {
        if ab.dt_conflict() {
            return Err(Error::AppendUnsupported(
                "append mixes element types in one builder; use one element type per \
                 append_dataset call",
            ));
        }

        let region = Self::gather_oh_messages(src, addr, base)?;

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
        let ChunkIndexLayout::ExtensibleArray { address } = index else {
            return Err(Error::AppendUnsupported(
                "append requires an Extensible-Array-indexed chunked dataset (a single \
                 unlimited dimension under the latest format)",
            ));
        };

        if disk_ds.space_type != DataspaceType::Simple || disk_ds.dimensions.len() != 1 {
            return Err(Error::AppendUnsupported(
                "append requires a rank-1 dataset in this release",
            ));
        }
        if !matches!(
            disk_ds.max_dimensions.as_deref(),
            Some([MaxExtent::Unlimited, ..])
        ) {
            return Err(Error::AppendUnsupported(
                "append requires a dataset that is unlimited along its first dimension",
            ));
        }

        let ChunkedGeometry {
            spatial,
            element_size,
            raw_size,
            ..
        } = ChunkedGeometry::new(&disk_dt, &disk_ds, &dl)?;
        let chunk_elems = spatial[0];
        if ab.raw().len() % element_size != 0 {
            return Err(Error::AppendUnsupported(
                "appended byte length is not a whole number of elements",
            ));
        }
        match ab.elem_dt() {
            Some(expected) if *expected != disk_dt => {
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
        let view = BaseOffsetSource {
            inner: src,
            base: base.effective_base(),
        };

        // Retained records must use the same element format as the replacement index.
        if let Some(idx_addr) = *address {
            let array =
                crate::extensible_array::read_header(&view, crate::file_writer::WIDTHS, idx_addr)?;
            if array.is_filtered() != has_filters {
                return Err(Error::AppendUnsupported(
                    "dataset filter metadata is inconsistent (chunk-index client id \
                     disagrees with the filter pipeline)",
                ));
            }
        }

        let infos = chunked_read::enumerate_chunks_from_source(
            &view,
            &dl,
            &disk_ds,
            crate::file_writer::WIDTHS,
        )?;
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

        let ctx = ChunkContext::from_datatype(spatial, &disk_dt)?;
        let mut scratch = FilterScratch::new();
        let (mut tail_raw, old_tail_extent) = if has_partial {
            let partial = &grid_order[n_full];
            let live = partial.read_live_elements(
                &view,
                &mut scratch,
                *flags,
                pipeline.as_ref(),
                ctx,
                current_dim0,
            )?;
            let extent = (base.absolute(partial.address)?, partial.chunk_size.get());
            (live, Some(extent))
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
        let new_dataspace_body = grown.serialize(LENGTH_SIZE);

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
            kept_chunks,
            new_chunks,
            old_addr: addr,
            old_tail_extent,
        })
    }

    /// Writes a relocating edit and returns its new object-header address.
    ///
    /// The caller publishes the parent link and reclaims superseded storage after commit
    /// publication.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if replacement layout or attribute messages cannot be
    /// written. Returns [`Error::Format`] if heap addresses, chunk encoding, header messages, or
    /// allocation geometry are invalid, and [`Error::Io`] if writing replacement storage fails.
    pub(super) fn write_moving(&mut self, mw: &MovingWrite) -> Result<u64, Error> {
        let base = self.context.addresses();
        match mw {
            MovingWrite::Contiguous {
                region,
                addr_off,
                bytes,
                ..
            } => {
                let raw = self.resolve_overwrite_bytes(bytes)?;
                let new_data_addr = self.alloc_or_append_typed(&raw, AllocationClass::Raw)?;
                let mut region = region.clone();
                let relative = base.relative(new_data_addr)?.get();
                region.bytes_mut()[*addr_off..*addr_off + 8]
                    .copy_from_slice(&relative.to_le_bytes());
                let size_off = *addr_off + 8;
                let raw_len = raw.len() as u64;
                region.bytes_mut()[size_off..size_off + 8].copy_from_slice(&raw_len.to_le_bytes());
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            MovingWrite::Compact { region, bytes } => {
                let raw = self.resolve_overwrite_bytes(bytes)?;
                let region = super::object_header::rebuild_compact_layout_region(region, &raw)?;
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            MovingWrite::Chunked {
                region,
                overwrite,
                pipeline_message,
                payload,
                ..
            } => {
                let deferred;
                let chunks = match payload {
                    ChunkPayload::Encoded(chunks) => chunks,
                    ChunkPayload::Deferred { bytes } => {
                        let raw = self.resolve_deferred_bytes(bytes)?;
                        deferred = overwrite.prepare_chunks(&raw)?;
                        &deferred
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
                    region,
                    &overwrite.geometry,
                    overwrite.flags,
                    pipeline_message.as_deref(),
                    &meta,
                    &PreparedChunkProvider { chunks },
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
                kept_chunks,
                new_chunks,
                ..
            } => self.write_appended_chunks(
                region,
                new_dataspace_body,
                *flags,
                chunk_dims_u32,
                *element_size,
                *raw_size,
                *has_filters,
                kept_chunks,
                new_chunks,
            ),
            MovingWrite::AttrEdit { region, attrs, .. } => {
                let mut region = region.clone();
                self.place_edited_attrs(&mut region, attrs.clone())?;
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
        }
    }

    /// Writes appended chunks, a replacement index, and the grown dataset header.
    ///
    /// Complete chunks retain their addresses. On PAGE files, a reusable allocation is used for
    /// both
    /// new chunks and the index when it fits their combined length. Otherwise each is placed
    /// separately.
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
        kept_chunks: &[ChunkRecord],
        new_chunks: &[PreparedChunk],
    ) -> Result<u64, Error> {
        let base = self.context.addresses();
        // On PAGE files, adjacency to chunk data proves that an index occupies raw pages.
        // Use one reusable region when it fits both. Otherwise place chunks separately and retain
        // unproven index bytes as dead storage during reclamation.
        let chunk_total: u64 = new_chunks.iter().map(PreparedChunk::stored_size).sum();
        let placed_chunks = |blob_stored: StoredAddress| -> Vec<ChunkRecord> {
            let mut combined: Vec<ChunkRecord> = kept_chunks.to_vec();
            let mut next = blob_stored;
            for chunk in new_chunks {
                combined.push(ChunkRecord {
                    address: next,
                    stored_size: chunk.stored_size(),
                    filter_mask: chunk.filter_mask(),
                });
                next = next.offset(chunk.stored_size());
            }
            combined
        };
        let sizing = placed_chunks(StoredAddress::new(0));
        let ea_len = hdf5_pure_format::__private::extensible_array_len(
            &IndexSlots::dense(&sizing),
            chunk_bytes,
            OFFSET_WIDTH,
            LENGTH_WIDTH,
            has_filters,
        );
        let ea = |slots: &IndexSlots<'_>, at: StoredAddress| -> Result<Vec<u8>, Error> {
            hdf5_pure_format::__private::build_extensible_array_at(
                slots,
                chunk_bytes,
                OFFSET_WIDTH,
                LENGTH_WIDTH,
                has_filters,
                at,
            )
            .map_err(Error::Format)
        };
        let blob = if self.space.is_paged() {
            self.alloc_free(chunk_total + ea_len, AllocationClass::Raw)
        } else {
            None
        };
        let ea_stored = match blob {
            Some(blob_stored) => {
                // Return the reservation on any failure before placement completes.
                let placed = (|| -> Result<StoredAddress, Error> {
                    let combined = placed_chunks(blob_stored);
                    let mut buf =
                        Vec::with_capacity(usize::try_from(chunk_total + ea_len).unwrap_or(0));
                    for chunk in new_chunks {
                        buf.extend_from_slice(chunk.bytes());
                    }
                    buf.extend_from_slice(&ea(
                        &IndexSlots::dense(&combined),
                        blob_stored.offset(chunk_total),
                    )?);
                    self.place(
                        Placement::Reused {
                            addr: base.resolve(blob_stored)?.get(),
                            len: chunk_total + ea_len,
                        },
                        &buf,
                    )?;
                    Ok(blob_stored.offset(chunk_total))
                })();
                match placed {
                    Ok(ea_stored) => ea_stored,
                    Err(e) => {
                        self.release_raw_alloc(blob_stored, chunk_total + ea_len);
                        return Err(e);
                    }
                }
            }
            None => {
                let mut combined: Vec<ChunkRecord> = kept_chunks.to_vec();
                for chunk in new_chunks {
                    let abs = self.alloc_or_append_typed(chunk.bytes(), AllocationClass::Raw)?;
                    combined.push(ChunkRecord {
                        address: base.relative(abs)?,
                        stored_size: chunk.stored_size(),
                        filter_mask: chunk.filter_mask(),
                    });
                }
                let at =
                    self.reserve_with_page_type(ea_len, AllocationClass::Metadata, PageType::Raw)?;
                let bytes = ea(&IndexSlots::dense(&combined), base.relative(at.address())?)?;
                let ea_addr = self.place(at, &bytes)?;
                base.relative(ea_addr)?
            }
        };

        #[expect(
            clippy::cast_possible_truncation,
            reason = "element size is a datatype byte width that fits u32"
        )]
        let layout_body = DataLayout::encode_chunked(
            flags,
            chunk_dims_u32,
            element_size.get() as u32,
            ChunkIndexInfo::ExtensibleArray,
            ea_stored,
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        )?;
        let region = replace_dataspace_message(region, new_dataspace_body)?;
        let region = replace_layout_message(&region, &layout_body)?;
        let oh = super::object_header::build_v2_object_header(&region)?;
        self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
    }

    /// Writes a prepared dataset's chunk storage and returns its object-header bytes.
    ///
    /// Chunks are encoded once before their storage is sized and allocated. The shared chunk writer
    /// assembles the storage for the allocated address.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the filter context or pipeline is invalid, chunks cannot be
    /// encoded, the header or attributes cannot be serialized, or allocation geometry is invalid.
    /// Returns [`Error::Io`] if writing chunk storage, attributes, or allocation padding fails.
    fn build_chunked_dataset(&mut self, fd: &FlatDataset) -> Result<Vec<u8>, Error> {
        let chunk_dims = fd.chunk_options.resolve_chunk_dims(&fd.ds.dimensions);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &fd.dt)?;
        let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
        let fill_time = fd.fill_time.unwrap_or_default();
        let fill = FillPattern::new(fill_time, fd.fill.as_deref(), elem);
        let set = chunked_write::compress_chunks(
            &fd.raw,
            &fd.ds.dimensions,
            ctx,
            &fd.chunk_options,
            fd.maxshape.as_deref(),
            fill,
            fd.allocation,
        )?;
        let (_addr, (layout_message, pipeline_message)) = self.place_relocatable(
            chunked_write::chunked_data_len(&set)?,
            AllocationClass::Raw,
            |stored_base| {
                let result = chunked_write::assemble_chunked_at(&set, stored_base)?;
                Ok((
                    result.data_bytes,
                    (result.layout_message, result.pipeline_message),
                ))
            },
        )?;
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
            fd.fill.as_deref(),
        )?)
    }

    /// Places attributes selected for dense storage during dataset preparation.
    ///
    /// Returns an Attribute Info message for dense storage, or `None` for inline attributes.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`place_dense_attrs`](Self::place_dense_attrs) when the prepared
    /// attributes require dense storage.
    fn place_dense_attrs_if_needed(&mut self, fd: &FlatDataset) -> Result<Option<Vec<u8>>, Error> {
        if !fd.attrs_are_dense {
            return Ok(None);
        }
        self.place_dense_attrs(&fd.attrs, DenseAttrCreationOrder::Untracked)
            .map(Some)
    }

    /// Writes stored chunk bytes and a replacement dataset header.
    ///
    /// The shared chunk writer plans and emits the chunk data and index at their allocated address.
    /// Chunk sizes and filter masks are preserved. The existing header retains its messages except
    /// for
    /// the replaced layout and any rebuilt dense attribute storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the header lacks a layout message. Returns
    /// [`Error::Format`] if chunk layout, attribute storage, or header encoding is invalid, or the
    /// built storage differs in length from its allocation. Returns [`Error::Io`] if writing the
    /// replacement storage fails.
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
    ) -> Result<u64, Error> {
        let ChunkGeometry {
            shape,
            spatial: chunk_dims,
            element_size,
            maxshape,
            ..
        } = geometry;
        let element_size = *element_size;
        let maxshape = maxshape.as_deref();
        let sizing = chunked_write::plan_chunked_data_verbatim(
            meta,
            shape,
            chunk_dims,
            element_size,
            flags,
            pipeline_message,
            StoredAddress::new(0),
            maxshape,
        )?;
        let (_addr, layout_message) =
            self.place_relocatable(sizing.plan.total_len, AllocationClass::Raw, |stored_base| {
                let layout = chunked_write::plan_chunked_data_verbatim(
                    meta,
                    shape,
                    chunk_dims,
                    element_size,
                    flags,
                    pipeline_message,
                    stored_base,
                    maxshape,
                )?;
                let mut buf =
                    Vec::with_capacity(usize::try_from(layout.plan.total_len).unwrap_or(0));
                chunked_write::emit_chunked_data_verbatim(&mut buf, &layout.plan, chunks)
                    .map_err(EmissionError::into_format)?;
                Ok((buf, layout.layout_message))
            })?;
        let mut new_region = replace_layout_message(region, &layout_message)?;
        self.append_dense_attrs(&mut new_region, dense_attrs)?;
        let oh = super::object_header::build_v2_object_header(&new_region)?;
        self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
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
    /// if the existing logical byte length exceeds [`usize::MAX`] when materializing fill values.
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
        if fd.vl_string_staging.is_some() || fd.reference_targets.is_some() {
            return Err(Error::AppendUnsupported(
                "a staged variable-length-string or object-reference dataset cannot be appended \
                 to before it is committed: its per-element side table is built with the \
                 dataset. Supply every element through the builder that creates it",
            ));
        }
        if let Some(dt) = builder.elem_dt() {
            if *dt != fd.dt {
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
        let mut row_bytes = u64::from(fd.dt.type_size());
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
        if fd.allocation == StorageAllocation::Unallocated && !bytes.is_empty() {
            let elem_size = crate::datatype::element_size_usize(&fd.dt)?;
            fd.raw = FillPattern::new(
                fd.fill_time.unwrap_or_default(),
                fd.fill.as_deref(),
                elem_size,
            )
            .buffer(fd.extent.to_usize()?)?;
            fd.allocation = StorageAllocation::Allocated;
        }
        fd.raw.extend_from_slice(bytes);
        fd.extent = extent;
        fd.ds.dimensions[0] = grown;
        #[cfg(feature = "provenance")]
        fd.rebuild_provenance();
        Ok(())
    }

    /// Rejects builder options that cannot represent a value overwrite.
    ///
    /// [`stage_dataset_write`](Self::stage_dataset_write) checks these options before storing the
    /// write.
    /// Compatibility with the committed dataset is checked by
    /// [`prepare_write`](Self::prepare_write).
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for unallocated storage, chunk or maximum-shape options,
    /// a [`PartialEdgeFiltering`] policy, attributes, fill settings, or unresolved object-reference
    /// targets.
    pub(super) fn refuse_unsupported_overwrite(fd: &FlatDataset) -> Result<(), Error> {
        if fd.allocation == StorageAllocation::Unallocated {
            return Err(Error::EditUnsupported(
                "write_dataset requires element data, not unallocated storage",
            ));
        }
        if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot make a dataset \
                 chunked, filtered, or extensible",
            ));
        }

        if fd.chunk_options.partial_edge_filtering.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot set whether partial \
                 edge chunks are filtered (set it when the dataset is created)",
            ));
        }

        if !fd.attrs.is_empty() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot set attributes \
                 (set them with a separate edit)",
            ));
        }

        if fd.fill.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot change the fill \
                 value (set it when the dataset is created)",
            ));
        }
        if fd.fill_time.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot change the fill \
                 time (set it when the dataset is created)",
            ));
        }

        if fd.reference_targets.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset cannot overwrite an object-reference dataset's \
                 data in place yet",
            ));
        }

        Ok(())
    }

    /// Returns prepared bytes or resolves deferred heap references during commit application.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`resolve_deferred_bytes`](Self::resolve_deferred_bytes) for deferred
    /// values.
    fn resolve_overwrite_bytes<'b>(
        &mut self,
        bytes: &'b OverwriteBytes,
    ) -> Result<Cow<'b, [u8]>, Error> {
        match bytes {
            OverwriteBytes::Prepared(raw) => Ok(Cow::Borrowed(raw)),
            OverwriteBytes::Deferred(bytes) => Ok(Cow::Owned(self.resolve_deferred_bytes(bytes)?)),
        }
    }

    /// Places variable-length collections and patches their addresses into element bytes.
    ///
    /// The session records the new collections and schedules superseded collections for
    /// reclamation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`place_vl_collections`](Self::place_vl_collections) when allocating
    /// replacement collections. Returns [`Error::Format`] if a placed collection's address cannot
    /// be
    /// converted to a physical position.
    fn resolve_deferred_bytes(&mut self, bytes: &DeferredBytes) -> Result<Vec<u8>, Error> {
        let DeferredBytes { raw, vlen } = bytes;
        let staging = &vlen.staging;
        let superseded = self
            .vl_overwrite_heaps
            .get(&vlen.path)
            .cloned()
            .unwrap_or_default();
        let mut raw = raw.clone();
        let base = self.context.addresses();
        let addrs = self.place_vl_collections(&staging.collections)?;
        type_builders::patch_vl_refs_masked(&mut raw, &staging.patch_offsets, &addrs);
        let placed = addrs
            .iter()
            .zip(&staging.collections)
            .map(|(&a, c)| Ok((base.absolute(StoredAddress::new(a))?, c.len() as u64)))
            .collect::<Result<Vec<_>, FormatError>>()?;
        self.vl_overwrite_heaps.insert(vlen.path.clone(), placed);
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

/// Pairs unresolved element bytes with the collections and path required to resolve them.
pub(super) struct DeferredBytes {
    raw: Vec<u8>,
    vlen: VlenOverwrite,
}

impl DeferredBytes {
    fn new(raw: &[u8], staging: &VlStringStaging, path: &ObjectPathBuf) -> Self {
        Self {
            raw: raw.to_vec(),
            vlen: VlenOverwrite {
                staging: staging.clone(),
                path: path.clone(),
            },
        }
    }
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
    /// Prepared contiguous bytes written without relocating the object header.
    InPlace { data_addr: u64, bytes: Vec<u8> },
    /// Chunk-data writes and any index rewrite that fit their existing allocations.
    InPlaceChunks { writes: Vec<(u64, Vec<u8>)> },
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
        /// Existing complete chunks with their stored addresses, sizes, and filter masks.
        kept_chunks: Vec<ChunkRecord>,
        /// Rewritten trailing and appended chunks, encoded for the grown extent.
        new_chunks: Vec<PreparedChunk>,
        /// The superseded header address used for storage reclamation.
        old_addr: u64,
        /// The absolute address and length of a rewritten trailing chunk, reclaimed after
        /// publication.
        old_tail_extent: Option<(u64, u64)>,
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
        old_addr: u64,
    },
    /// Inline values stored in a replacement object header.
    Compact {
        region: OhRegion,
        bytes: OverwriteBytes,
    },
    /// Element bytes written to replacement contiguous storage.
    Contiguous {
        region: OhRegion,
        addr_off: usize,
        bytes: OverwriteBytes,
        old_extent: Option<(u64, u64)>,
    },
}

/// Distinguishes encoded chunks from element bytes awaiting heap-address resolution.
pub(super) enum ChunkPayload {
    /// Variable-length elements resolved before chunk splitting and encoding.
    Deferred { bytes: DeferredBytes },
    /// Chunks encoded during preparation to determine whether existing storage can be reused.
    Encoded(Vec<PreparedChunk>),
}

/// A staged dataset's values, metadata, and unresolved references.
pub(super) struct FlatDataset {
    pub(super) name: LinkNameBuf,
    pub(super) dt: crate::datatype::Datatype,
    pub(super) ds: Dataspace,
    /// Prepared element bytes, empty for an unallocated declaration.
    pub(super) raw: Vec<u8>,
    /// Whether element storage is supplied or remains unallocated.
    allocation: StorageAllocation,
    /// The logical byte length of all elements, including unwritten values.
    extent: u64,
    pub(super) attrs: Vec<crate::attribute::AttributeMessage>,
    /// Storage options used by the shared chunk writer.
    pub(super) chunk_options: ChunkOptions,
    /// Maximum dimensions, also recorded in the dataspace.
    pub(super) maxshape: Option<Vec<MaxExtent>>,
    /// Attribute indices paired with global-heap collections awaiting placement.
    pub(super) vl_attrs: Vec<(usize, Vec<Vec<u8>>)>,
    /// Whether the prepared attributes require a fractal heap.
    ///
    /// Reference patching and digest updates preserve serialized lengths.
    pub(super) attrs_are_dense: bool,
    /// Unresolved variable-length element references and their collections.
    pub(super) vl_string_staging: Option<VlStringStaging>,
    /// Object-reference targets resolved after the commit has assigned object addresses.
    pub(super) reference_targets: Option<Vec<ObjectRefPatch>>,
    /// One encoded fill element, or `None` for the default.
    pub(super) fill: Option<Vec<u8>>,
    /// The fill policy, or `None` for the default.
    pub(super) fill_time: Option<FillTime>,
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
        let rebuilt = prov.inputs.build_attrs(&self.raw);
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
/// application. Unallocated declarations have an empty element buffer and a logical byte length
/// derived from their shape and datatype. The `provenance` feature computes digest attributes over
/// the prepared values.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] for missing values or metadata, incompatible shape or storage
/// options, committed datatypes, unsupported filters, or a link name too large to encode.
///
/// Returns [`Error::Format`] with [`FormatError::ConflictingDatasetStorage`] if an unallocated
/// declaration also supplies element data or a provider, [`FormatError::DatasetRankUnsupported`]
/// if the rank exceeds the writer's limit, [`FormatError::DatasetDimensionUnlimited`] if a current
/// dimension is unlimited, or [`FormatError::DatasetSizeOverflow`] if the logical byte length
/// overflows. With the `provenance` feature, returns [`FormatError::DatasetProvenanceRequiresData`]
/// if a nonempty unallocated declaration requests element checksums.
///
/// Returns attribute-preparation, datatype-size, chunk-geometry, filter-pipeline, and fill-value
/// errors.
pub(super) fn flatten_dataset(db: DatasetBuilder, name: LinkNameBuf) -> Result<FlatDataset, Error> {
    let DatasetBuilder {
        name: _,
        datatype,
        shape,
        maxshape,
        data,
        attrs: attribute_specs,
        chunk_options,
        raw_chunks,
        produced,
        reference_targets,
        vl_string_staging,
        fill,
        fill_time,
        allocation,
        committed_datatype_path,
        #[cfg(feature = "provenance")]
        provenance,
    } = db;
    let dt = datatype.ok_or(Error::EditUnsupported("dataset has no datatype/data"))?;
    let shape = shape.ok_or(Error::EditUnsupported("dataset has no shape"))?;
    let is_empty = shape.contains(&0);
    let chunked = chunk_options.is_chunked() || maxshape.is_some();
    let unallocated = allocation == StorageAllocation::Unallocated;
    let data = data.transpose().map_err(|error| *error)?;
    if unallocated && (data.is_some() || raw_chunks.is_some() || produced.is_some()) {
        return Err(FormatError::ConflictingDatasetStorage.into());
    }
    if vl_string_staging.is_some() && chunked {
        return Err(Error::EditUnsupported(
            "chunked or extensible variable-length-string datasets cannot be added in place yet",
        ));
    }
    if reference_targets.is_some() && chunked {
        return Err(Error::EditUnsupported(
            "chunked or extensible object-reference datasets cannot be added in place yet",
        ));
    }
    let raw = if unallocated || is_empty {
        data.unwrap_or_default()
    } else {
        data.ok_or(Error::EditUnsupported("dataset has no data"))?
    };

    let elem_size = crate::datatype::element_size_usize(&dt)?;

    let elem = elem_size.get() as u64;
    let rank = match u8::try_from(shape.len()) {
        Ok(rank) if rank <= file_writer::MAX_DATASET_RANK => rank,
        _ => {
            return Err(FormatError::DatasetRankUnsupported {
                rank: shape.len(),
                maximum: file_writer::MAX_DATASET_RANK,
            }
            .into());
        }
    };
    if let Some(dimension) = shape
        .iter()
        .position(|&size| size == file_writer::UNLIMITED_DIMENSION)
    {
        return Err(FormatError::DatasetDimensionUnlimited { dimension }.into());
    }
    let extent = if is_empty {
        0
    } else {
        shape.iter().try_fold(elem, |size, &dimension| {
            size.checked_mul(dimension)
                .ok_or(FormatError::DatasetSizeOverflow { size, dimension })
        })?
    };
    if !unallocated && raw.len() as u64 != extent {
        return Err(Error::EditUnsupported(
            "dataset data length does not match its shape",
        ));
    }
    #[cfg(feature = "provenance")]
    if unallocated && extent != 0 && provenance.is_some() {
        return Err(FormatError::DatasetProvenanceRequiresData.into());
    }

    if chunked {
        Extent::new(&shape, maxshape.as_deref())
            .and_then(|extent| chunk_options.validate_geometry(extent))
            .map_err(Error::EditUnsupported)?;
        chunk_options
            .refuse_unavailable_filters()
            .map_err(Error::EditUnsupported)?;
        let chunk_dims = chunk_options.resolve_chunk_dims(&shape);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &dt)?;
        crate::chunked_read::bounded_chunk_len(&chunk_dims, elem_size)?;
        chunk_options.build_pipeline(
            &ctx,
            FillPattern::new(
                fill_time.unwrap_or_default(),
                fill.as_deref(),
                ctx.element_size.narrow::<NonZeroUsize>()?,
            ),
        )?;
    }

    let mut sized = file_writer::make_link(name.as_str(), StoredAddress::new(0));
    sized.creation_order = Some(0);
    if sized.serialize(crate::file_writer::OFFSET_WIDTH).len() > OBJECT_HEADER_MESSAGE_MAX {
        return Err(Error::EditUnsupported(
            "dataset name is too long to encode as a link message",
        ));
    }

    let ds = Dataspace {
        space_type: if shape.is_empty() {
            DataspaceType::Scalar
        } else {
            DataspaceType::Simple
        },
        rank,
        dimensions: shape,
        max_dimensions: maxshape.clone(),
    };
    let mut attrs: Vec<crate::attribute::AttributeMessage> =
        Vec::with_capacity(attribute_specs.len());
    for (n, v) in &attribute_specs {
        attrs.push(v.to_message(n)?);
    }
    let mut vl_attrs: Vec<(usize, Vec<Vec<u8>>)> = Vec::new();
    for (i, (_, v)) in attribute_specs.iter().enumerate() {
        if let Some(strings) = v.var_len_strings() {
            vl_attrs.push((i, type_builders::build_global_heap_collections(strings)));
        }
    }
    #[cfg(feature = "provenance")]
    let provenance = provenance.as_ref().map(|prov| {
        let inputs = crate::provenance::Provenance {
            creator: prov.creator.clone(),
            timestamp: prov.timestamp.clone(),
            source: prov.source.clone(),
        };
        let attrs_start = attrs.len();
        attrs.extend(inputs.build_attrs(&raw));
        StagedProvenance {
            inputs,
            attrs_start,
        }
    });
    let attrs_are_dense = crate::file_writer::needs_dense_attrs(&attrs).map_err(Error::Format)?;
    if attrs_are_dense {
        crate::file_writer::dense_attrs_check(&attrs).map_err(Error::Format)?;
    }

    if committed_datatype_path.is_some() || attrs.iter().any(|a| a.datatype_location.is_committed())
    {
        return Err(Error::EditUnsupported(
            "a dataset or attribute naming a committed (shared) datatype cannot be added in \
             place; write the file with FileBuilder instead",
        ));
    }

    if let Some(fill) = &fill {
        let expected = elem.to_usize()?;
        if fill.len() != expected {
            return Err(Error::Format(FormatError::FillValueSizeMismatch {
                expected,
                actual: fill.len(),
            }));
        }
    }

    Ok(FlatDataset {
        name,
        dt,
        ds,
        raw,
        allocation,
        extent,
        attrs,
        chunk_options,
        maxshape,
        vl_attrs,
        attrs_are_dense,
        vl_string_staging,
        reference_targets,
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
/// Copies staged element bytes and their deferred variable-length work.
///
/// The pending set remains intact if commit preparation rejects the write.
fn staged_bytes(fd: &FlatDataset, path: &ObjectPathBuf) -> OverwriteBytes {
    match &fd.vl_string_staging {
        Some(staging) => OverwriteBytes::Deferred(DeferredBytes::new(&fd.raw, staging, path)),
        None => OverwriteBytes::Prepared(fd.raw.clone()),
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
            ChunkContext::from_datatype(chunk_dims, dt)?,
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
}

/// Returns writes for chunks that fit their existing dense-grid storage.
///
/// Filter masks must match. Equal stored lengths leave the index unchanged. Smaller chunks require
/// an index that [`try_rebuild_index_in_place`] can reproduce within its original allocation.
/// Returns `None` for incompatible storage, read errors, overlapping writes, or out-of-bounds
/// spans.
fn try_inplace_chunk_writes<S: Source + ?Sized>(
    src: &S,
    layout: &DataLayout,
    ds: &Dataspace,
    geometry: &ChunkGeometry,
    new_chunks: &[PreparedChunk],
) -> Option<Vec<(u64, Vec<u8>)>> {
    let infos =
        chunked_read::enumerate_chunks_from_source(src, layout, ds, crate::file_writer::WIDTHS)
            .ok()?;
    let grid = chunked_read::plan_dense_grid(infos, &geometry.shape, &geometry.spatial)?;
    if grid.grid_order.len() != new_chunks.len() {
        return None;
    }
    let mut writes = Vec::with_capacity(new_chunks.len() + 1);
    let mut spans: Vec<(u64, u64)> = Vec::with_capacity(new_chunks.len() + 1);
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
        ci.address
            .get()
            .checked_add(new_len)
            .filter(|&e| e <= src.len())?;
        writes.push((ci.address.get(), chunk.bytes().to_vec()));
        spans.push((ci.address.get(), new_len));
    }

    if any_shrunk {
        let (index_addr, index_bytes) =
            try_rebuild_index_in_place(src, layout, ds, geometry, &grid.grid_order, new_chunks)?;
        spans.push((index_addr, index_bytes.len() as u64));
        writes.push((index_addr, index_bytes));
    }

    if !super::spans_disjoint_in_bounds(&mut spans, src.len()) {
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
fn try_rebuild_index_in_place<S: Source + ?Sized>(
    src: &S,
    layout: &DataLayout,
    ds: &Dataspace,
    geometry: &ChunkGeometry,
    grid_order: &[crate::chunked_read::ChunkInfo],
    new_chunks: &[PreparedChunk],
) -> Option<(u64, Vec<u8>)> {
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
            OFFSET_WIDTH,
            LENGTH_WIDTH,
            true,
            index_addr,
        )
        .ok()?,
        ChunkIndexLayout::ExtensibleArray { .. } => {
            hdf5_pure_format::__private::build_extensible_array_at(
                &slots,
                raw_size,
                OFFSET_WIDTH,
                LENGTH_WIDTH,
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
    let mut spans =
        crate::chunked_read::chunk_index_spans_from_source(src, layout, crate::file_writer::WIDTHS)
            .ok()?;
    if spans.is_empty() {
        return None;
    }
    spans.sort_unstable_by_key(|&(a, _)| a);
    if spans[0].0 != index_addr.get() {
        return None;
    }
    let mut end = index_addr.get();
    for &(a, l) in &spans {
        if a != end {
            return None; // a gap means the index is not contiguous
        }
        end = a.checked_add(l)?;
    }
    if new_index.len() as u64 != end - index_addr.get() {
        return None;
    }
    index_addr
        .get()
        .checked_add(new_index.len() as u64)
        .filter(|&e| e <= src.len())?;
    Some((index_addr.get(), new_index))
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
