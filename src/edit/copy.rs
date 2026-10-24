//! Object-tree copies, source reads, and reference screening for staged edits.

use hdf5_pure_format::__private::ChunkedGeometry;
use hdf5_pure_space::__private::AllocationClass;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_read;
use crate::chunked_write::{ChunkMeta, ChunkProvider};
use crate::convert::Narrow;
use crate::data_layout::{ChunkedLayoutFlags, DataLayout};
use crate::dataspace::Dataspace;
use crate::datatype::{self, Datatype};
use crate::error::{Error, FormatError};
use crate::file_writer::{LENGTH_WIDTH, OFFSET_WIDTH};
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_path::ObjectPathBuf;
use crate::source::{BaseOffsetSource, BytesSource, Source};
use crate::width::FormatWidths;

use super::attributes::{self, DenseAttrSet};
use super::dataset::{self, ChunkGeometry};
use super::group;
use super::object_header::OhRegion;
use super::staging::StagedOp;
use super::{InvalidatedAddresses, WriteEngine};

impl WriteEngine {
    /// Stages a deep copy of a committed object for the next [`commit`](Self::commit).
    ///
    /// The source is read during the commit. The destination must be available, and the
    /// source subtree must use storage the editor can reproduce.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps either path.
    pub fn copy(&mut self, src: &ObjectPathBuf, dst: &ObjectPathBuf) -> Result<(), Error> {
        self.refuse_if_claimed(src)?;
        self.refuse_if_claimed(dst)?;
        self.staged
            .push(StagedOp::CopyObject(src.clone(), dst.clone()));
        Ok(())
    }

    /// Reads a source subtree from another file and stages its copy at `dst`.
    ///
    /// The subtree is read before this call returns. The destination is written by the next
    /// [`commit`](Self::commit), which checks destination collisions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the source is streamed, uses a userblock or
    /// widths other than eight bytes, either path is the root, or a live buffered appender
    /// overlaps `dst`. Returns the read or preparation error if the source subtree cannot
    /// be resolved or reproduced without references to storage in the source file.
    pub fn copy_from(
        &mut self,
        source: &crate::reader::File,
        src: &ObjectPathBuf,
        dst: &ObjectPathBuf,
    ) -> Result<(), Error> {
        let src_data = source.in_memory_image().ok_or(Error::EditUnsupported(
            "cross-file copy requires a buffered source file (File::open or File::from_bytes), not a streaming one",
        ))?;
        let src_sb = source.superblock();
        if src_sb.offset_size != OFFSET_WIDTH || src_sb.length_size != LENGTH_WIDTH {
            return Err(Error::EditUnsupported(
                "cross-file copy requires the source file to use 8-byte offsets and lengths",
            ));
        }
        if !source.base_address().is_zero() {
            return Err(Error::EditUnsupported(
                "cross-file copy requires the source file to have no userblock (base address 0)",
            ));
        }

        if src.is_empty() {
            return Err(Error::EditUnsupported("cannot copy the root group"));
        }
        if dst.is_empty() {
            return Err(Error::EditUnsupported("copy destination path is empty"));
        }

        let src_addr = crate::group_v2::resolve_path_any(
            src_data,
            source.access_mode(),
            src_sb,
            source.widths(),
            source.addresses(),
            &src.as_path(),
        )?;
        // The source has base address zero, so its stored addresses are absolute.
        let tree = Self::read_copy_subtree(
            &BytesSource::new(src_data),
            source.access_mode(),
            src_addr,
            0,
            true,
            source.addresses(),
            source.widths(),
        )?;
        self.refuse_if_claimed(dst)?;
        self.staged
            .push(StagedOp::CopyObjectFrom(dst.clone(), tree));
        Ok(())
    }

    /// Parses an object header and attribute storage for copying.
    ///
    /// Dense attributes are read into an owned set for replacement storage. Hard-link children and
    /// contiguous data locations are recorded separately from the retained header messages.
    ///
    /// The layout version 3 and 4 fields are defined in "The Data Layout Message" of the
    /// [format specification, version 4.0][layout]. The Link Info version 0 fields are defined
    /// in "The Link Info Message" of the [same specification][links].
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the source uses external storage, unsupported
    /// layouts or links, or an object that is neither a dataset nor a group. Propagates metadata
    /// and attribute read, parse, and storage-planning errors.
    ///
    /// [layout]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
    /// [links]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
    pub(super) fn read_object<S: Source + ?Sized>(
        src: &S,
        access_mode: AccessMode,
        addr: u64,
        base: AddressContext,
    ) -> Result<ObjModel, Error> {
        let region = Self::gather_oh_messages(src, addr, base)?;

        reject_external_storage(&region)?;

        let mut dense = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p)? {
            if msg_type == MessageType::ATTRIBUTE_INFO {
                let ai = crate::attribute_info::AttributeInfoMessage::parse_with_offset_width(
                    &region[body..body_end],
                    crate::file_writer::OFFSET_WIDTH,
                )?;
                if ai.fractal_heap_address.is_some() {
                    dense = true;
                }
            }
            p = body_end;
        }

        // Read dense attributes into owned messages and check that they can be serialized.
        let dense_attrs = if dense {
            let stored = attributes::read_object_attrs(src, access_mode, addr, base)?;
            let set = attributes::dense_attr_set(&region, stored)?;
            crate::file_writer::dense_attrs_check(&set.attrs).map_err(Error::Format)?;
            set
        } else {
            DenseAttrSet::default()
        };

        let mut layout: Option<(usize, usize)> = None; // (body offset in kept, body size)
        let mut has_link_info = false;
        // Preserve each hard link's name, optional creation index, and source address.
        let mut children: Vec<(String, Option<u64>, StoredAddress)> = Vec::new();
        // Retain records after separating child links and dense attributes.
        let mut kept = OhRegion::empty(region.props());

        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p)? {
            let mut keep = true;
            match msg_type {
                MessageType::ATTRIBUTE_INFO => {
                    if dense {
                        keep = false;
                    }
                }
                MessageType::ATTRIBUTE => {
                    // Use the rebuilt heap as the object's only attribute storage.
                    if dense {
                        keep = false;
                    }
                }
                MessageType::LINK_INFO => {
                    has_link_info = true;
                    let mut q = body + 2;
                    if body_end - body >= 2 && region[body + 1] & 0x01 != 0 {
                        q += 8;
                    }
                    if q + 8 <= body_end {
                        let heap_addr = u64::from_le_bytes(region[q..q + 8].try_into().unwrap());
                        if heap_addr != u64::MAX {
                            return Err(Error::EditUnsupported(
                                "a group uses dense (fractal-heap) link storage (not supported in place yet)",
                            ));
                        }
                    }
                }
                MessageType::LINK => {
                    keep = false;
                    match LinkMessage::parse_with_offset_width(
                        &region[body..body_end],
                        OFFSET_WIDTH,
                    ) {
                        Ok(LinkMessage {
                            name,
                            link_target:
                                LinkTarget::Hard {
                                    object_header_address,
                                },
                            creation_order,
                            ..
                        }) => children.push((name, creation_order, object_header_address)),
                        _ => {
                            return Err(Error::EditUnsupported(
                                "a group contains a soft/external link (not copyable in place yet)",
                            ));
                        }
                    }
                }
                MessageType::DATA_LAYOUT => {
                    // Locate the data-address patch within the retained records.
                    layout = Some((kept.len() + (body - p), body_end - body));
                }
                _ => {}
            }
            if keep {
                kept.push_bytes(&region[p..body_end]);
            }
            p = body_end;
        }

        if let Some((lbody, lsize)) = layout {
            let version = kept[lbody];
            if !(version == 3 || version == 4) || lsize < 2 {
                return Err(Error::EditUnsupported(
                    "an unsupported data-layout version cannot be copied in place yet",
                ));
            }
            let class = kept[lbody + 1];
            match class {
                0 => Ok(ObjModel::DatasetVerbatim {
                    region: kept,
                    dense_attrs,
                }),
                1 => {
                    if lbody + 18 > kept.len() {
                        return Err(Error::EditUnsupported("malformed contiguous data layout"));
                    }
                    let data_addr = StoredAddress::new(u64::from_le_bytes(
                        kept[lbody + 2..lbody + 10].try_into().unwrap(),
                    ));
                    let data_size =
                        u64::from_le_bytes(kept[lbody + 10..lbody + 18].try_into().unwrap());
                    Ok(ObjModel::DatasetContiguous {
                        region: kept,
                        addr_off: lbody + 2,
                        data_addr,
                        data_size,
                        dense_attrs,
                    })
                }
                // Read stored chunks during subtree traversal and rebuild their index on write.
                2 => Ok(ObjModel::DatasetChunked {
                    region: kept,
                    dense_attrs,
                }),
                _ => Err(Error::EditUnsupported(
                    "an unsupported data-layout class cannot be copied in place yet",
                )),
            }
        } else if has_link_info {
            // Include the Group Info message required by subsequent link edits.
            group::ensure_group_info(&mut kept)?;
            Ok(ObjModel::Group {
                non_link_region: kept,
                children,
                dense_attrs,
            })
        } else {
            Err(Error::EditUnsupported(
                "an object is neither a contiguous/compact dataset nor a group",
            ))
        }
    }

    /// Captures a copyable object subtree in an owned [`CopyTree`].
    ///
    /// Preserves stored chunk bytes and filter masks. An in-file copy may share heap and object
    /// references. A cross-file copy rejects references into the source file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if nesting exceeds [`MAX_COPY_DEPTH`] or the editor
    /// cannot reproduce the source storage. Propagates metadata and payload read or parse errors.
    pub(super) fn read_copy_subtree<S: Source + ?Sized>(
        src: &S,
        access_mode: AccessMode,
        addr: u64,
        depth: u32,
        cross_file: bool,
        base: AddressContext,
        widths: FormatWidths,
    ) -> Result<CopyTree, Error> {
        if depth >= MAX_COPY_DEPTH {
            return Err(Error::EditUnsupported(
                "copy source nests too deeply (possible hard-link cycle)",
            ));
        }
        match Self::read_object(src, access_mode, addr, base)? {
            ObjModel::DatasetVerbatim {
                region,
                dense_attrs,
            } => {
                if cross_file {
                    reject_foreign_addresses(&region)?;
                    reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                Ok(CopyTree::DatasetVerbatim {
                    region,
                    dense_attrs,
                })
            }
            ObjModel::DatasetContiguous {
                region,
                addr_off,
                data_addr,
                data_size,
                dense_attrs,
            } => {
                if cross_file {
                    reject_foreign_addresses(&region)?;
                    reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                // Preserve an undefined data address without allocating a data block.
                let data = if data_addr.is_undefined(OFFSET_WIDTH) {
                    None
                } else {
                    let start = base.absolute(data_addr)?;
                    start
                        .checked_add(data_size)
                        .filter(|&e| e <= src.len())
                        .ok_or(Error::EditUnsupported("dataset data is out of bounds"))?;
                    let len = data_size.to_usize()?;
                    Some(src.read_exact_at(start, len)?)
                };
                Ok(CopyTree::DatasetContiguous {
                    region,
                    addr_off,
                    data,
                    dense_attrs,
                })
            }
            ObjModel::DatasetChunked {
                region,
                dense_attrs,
            } => {
                if cross_file {
                    reject_foreign_addresses(&region)?;
                    reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                let ChunkedHeaderParts {
                    dt,
                    ds,
                    layout,
                    pipeline_message,
                } = parse_chunked_header(&region, widths)?;
                let DataLayout::Chunked { index, flags, .. } = layout else {
                    return Err(Error::EditUnsupported("dataset is not chunked"));
                };
                if !index.enumerable() {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset with a version-2 B-tree chunk index cannot be \
                         copied in place yet",
                    ));
                }
                let geometry = ChunkedGeometry::new(&dt, &ds, &layout)?;

                let dview = BaseOffsetSource {
                    inner: src,
                    base: base.effective_base(),
                };

                let infos = chunked_read::enumerate_chunks_from_source(
                    &dview,
                    &layout,
                    &ds,
                    crate::file_writer::WIDTHS,
                )?;
                let grid = chunked_read::plan_dense_grid(infos, &ds.dimensions, geometry.spatial).ok_or(
                    Error::EditUnsupported(
                        "a chunked dataset with unallocated (sparse) chunks cannot be copied in place yet",
                    ),
                )?;
                if grid.grid_order.is_empty() {
                    return Err(Error::EditUnsupported(
                        "an empty chunked dataset cannot be copied in place yet",
                    ));
                }

                let mut meta = Vec::with_capacity(grid.grid_order.len());
                let mut chunk_bytes = Vec::with_capacity(grid.grid_order.len());
                for ci in &grid.grid_order {
                    let len = ci.chunk_size.to_usize()?;
                    ci.address
                        .get()
                        .checked_add(len as u64)
                        .filter(|&e| e <= dview.len())
                        .ok_or(Error::EditUnsupported("chunk data is out of bounds"))?;
                    chunk_bytes.push(dview.read_exact_at(ci.address.get(), len)?);
                    meta.push(ChunkMeta {
                        compressed_size: ci.chunk_size.get(),
                        filter_mask: ci.filter_mask,
                    });
                }

                let geometry = ChunkGeometry::new(&ds, geometry);
                Ok(CopyTree::DatasetChunked {
                    region,
                    geometry,
                    flags,
                    pipeline_message,
                    meta,
                    chunk_bytes,
                    dense_attrs,
                })
            }
            ObjModel::Group {
                non_link_region,
                children,
                dense_attrs,
            } => {
                if cross_file {
                    reject_foreign_addresses(&non_link_region)?;
                    reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                let mut copied_children = Vec::with_capacity(children.len());
                for (name, creation_order, child) in children {
                    let child = base.absolute(child)?;
                    copied_children.push((
                        name,
                        creation_order,
                        Self::read_copy_subtree(
                            src,
                            access_mode,
                            child,
                            depth + 1,
                            cross_file,
                            base,
                            widths,
                        )?,
                    ));
                }
                Ok(CopyTree::Group {
                    non_link_region,
                    children: copied_children,
                    dense_attrs,
                })
            }
        }
    }

    /// Writes an owned copy subtree and returns its root object-header address.
    ///
    /// Writes children before their parent links and encodes addresses relative to the
    /// destination file's base. Preserves unallocated contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a retained header cannot be rebuilt, [`Error::Format`]
    /// if header encoding, attribute encoding, or allocation geometry is invalid, and [`Error::Io`]
    /// if writing the copy fails. Propagates errors from [`Self::write_chunked_relocatable`] for
    /// chunked copies.
    pub(super) fn write_copy_subtree(&mut self, node: &CopyTree) -> Result<u64, Error> {
        let base = self.addresses;
        match node {
            CopyTree::DatasetVerbatim {
                region,
                dense_attrs,
            } => {
                let mut region = region.clone();
                self.append_dense_attrs(&mut region, dense_attrs.clone())?;
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            CopyTree::DatasetContiguous {
                region,
                addr_off,
                data,
                dense_attrs,
            } => {
                let mut region = region.clone();
                // Preserve an undefined data address without allocating a data block.
                if let Some(data) = data {
                    let new_data_addr = self.alloc_or_append_typed(data, AllocationClass::Raw)?;
                    hdf5_pure_format::__private::write_stored_address_at(
                        region.bytes_mut(),
                        *addr_off,
                        base.relative(new_data_addr)?,
                        OFFSET_WIDTH,
                    )?;
                }
                self.append_dense_attrs(&mut region, dense_attrs.clone())?;
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            CopyTree::DatasetChunked {
                region,
                geometry,
                flags,
                pipeline_message,
                meta,
                chunk_bytes,
                dense_attrs,
            } => self.write_chunked_relocatable(
                region,
                geometry,
                *flags,
                pipeline_message.as_deref(),
                meta,
                &SliceChunkProvider {
                    chunks: chunk_bytes,
                },
                dense_attrs.clone(),
            ),
            CopyTree::Group {
                non_link_region,
                children,
                dense_attrs,
            } => {
                let mut region = non_link_region.clone();
                for (name, creation_order, child) in children {
                    let new_child = self.write_copy_subtree(child)?;
                    region.push_link(name, base.relative(new_child)?, *creation_order);
                }
                self.append_dense_attrs(&mut region, dense_attrs.clone())?;
                let oh = super::object_header::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
        }
    }
}

/// A source object's retained header records, child links, and storage locations.
///
/// Dense attributes are parsed into a separate owned set. Their Attribute Info and inline
/// Attribute records are excluded from the retained region for replacement storage.
pub(super) enum ObjModel {
    /// A compact dataset with its element bytes retained in the header region.
    DatasetVerbatim {
        region: OhRegion,
        dense_attrs: DenseAttrSet,
    },
    /// A contiguous dataset with its source data address and size.
    DatasetContiguous {
        region: OhRegion,
        /// The data address field's byte offset within `region`.
        addr_off: usize,
        data_addr: StoredAddress,
        data_size: u64,
        dense_attrs: DenseAttrSet,
    },
    /// A chunked dataset whose payloads are read by [`WriteEngine::read_copy_subtree`].
    DatasetChunked {
        region: OhRegion,
        dense_attrs: DenseAttrSet,
    },
    /// A group with its hard-link records separated from the other header records.
    Group {
        non_link_region: OhRegion,
        /// Each link's name, optional creation index, and source object-header address.
        children: Vec<(String, Option<u64>, StoredAddress)>,
        dense_attrs: DenseAttrSet,
    },
}

/// An owned subtree prepared by [`WriteEngine::read_copy_subtree`].
///
/// Contains the header records, element bytes, dense attributes, and child subtrees needed by
/// [`WriteEngine::write_copy_subtree`]. The source buffer can be released after preparation.
pub(super) enum CopyTree {
    /// Stored chunks with checked geometry and their verbatim filter-pipeline message.
    ///
    /// The copy writer rebuilds the index and layout while preserving payload bytes, sizes, and
    /// filter masks. Dense attributes are written into separate replacement storage.
    DatasetChunked {
        region: OhRegion,
        geometry: dataset::ChunkGeometry,
        flags: ChunkedLayoutFlags,
        pipeline_message: Option<Vec<u8>>,
        meta: Vec<ChunkMeta>,
        chunk_bytes: Vec<Vec<u8>>,
        dense_attrs: DenseAttrSet,
    },
    /// A contiguous dataset with optional allocated element storage.
    DatasetContiguous {
        region: OhRegion,
        /// The data address field's byte offset within `region`.
        addr_off: usize,
        /// The allocated data bytes, or `None` to preserve an undefined data address.
        data: Option<Vec<u8>>,
        dense_attrs: DenseAttrSet,
    },
    /// A compact dataset header with inline values and separately rebuilt dense attributes.
    DatasetVerbatim {
        region: OhRegion,
        dense_attrs: DenseAttrSet,
    },
    /// A group with owned children to write before reconstructing its hard links.
    Group {
        non_link_region: OhRegion,
        /// Each link's name, optional creation index, and prepared child subtree.
        children: Vec<(String, Option<u64>, CopyTree)>,
        dense_attrs: DenseAttrSet,
    },
}

impl CopyTree {
    pub(super) fn has_dense_attributes(&self) -> bool {
        match self {
            Self::DatasetVerbatim { dense_attrs, .. }
            | Self::DatasetContiguous { dense_attrs, .. }
            | Self::DatasetChunked { dense_attrs, .. } => !dense_attrs.is_empty(),
            Self::Group {
                children,
                dense_attrs,
                ..
            } => {
                !dense_attrs.is_empty()
                    || children
                        .iter()
                        .any(|(_, _, child)| child.has_dense_attributes())
            }
        }
    }
}

/// Parsed dataset metadata with the filter pipeline preserved as stored bytes.
struct ChunkedHeaderParts {
    dt: crate::datatype::Datatype,
    ds: Dataspace,
    layout: DataLayout,
    pipeline_message: Option<Vec<u8>>,
}

/// Parses the metadata required to copy stored chunks.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a required message is absent or the layout is not
/// chunked. Propagates header-record, datatype, dataspace, and layout parse errors.
fn parse_chunked_header(
    region: &OhRegion,
    widths: FormatWidths,
) -> Result<ChunkedHeaderParts, Error> {
    let mut datatype: Option<(usize, usize)> = None;
    let mut dataspace: Option<(usize, usize)> = None;
    let mut layout: Option<(usize, usize)> = None;
    let mut pipeline: Option<(usize, usize)> = None;
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        match msg_type {
            MessageType::DATATYPE => datatype = Some((body, body_end)),
            MessageType::DATASPACE => dataspace = Some((body, body_end)),
            MessageType::DATA_LAYOUT => layout = Some((body, body_end)),
            MessageType::FILTER_PIPELINE => pipeline = Some((body, body_end)),
            _ => {}
        }
        p = body_end;
    }
    let (dt_b, dt_e) = datatype.ok_or(Error::EditUnsupported("dataset header has no datatype"))?;
    let (ds_b, ds_e) =
        dataspace.ok_or(Error::EditUnsupported("dataset header has no dataspace"))?;
    let (lb, le) = layout.ok_or(Error::EditUnsupported("dataset header has no data layout"))?;
    let (dt, _) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])?;
    let ds = Dataspace::parse_with_length_width(&region[ds_b..ds_e], widths.lengths)?;
    let dl = DataLayout::parse_with_widths(&region[lb..le], widths)?;
    if !matches!(dl, DataLayout::Chunked { .. }) {
        return Err(Error::EditUnsupported("dataset is not chunked"));
    }
    let pipeline_message = pipeline.map(|(b, e)| region[b..e].to_vec());
    Ok(ChunkedHeaderParts {
        dt,
        ds,
        layout: dl,
        pipeline_message,
    })
}

/// Rejects a header with an External Data Files message.
///
/// Both copy paths use this check because the editor reads dataset elements from the source
/// file only. `H5Pset_external` configures storage in other files.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if an External Data Files message is present or a record
/// extends past the region.
fn reject_external_storage(region: &OhRegion) -> Result<(), Error> {
    let mut p = 0;
    while let Some((msg_type, _, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::EXTERNAL_DATA_FILES {
            return Err(Error::EditUnsupported(
                "a dataset stores its elements in external files (H5Pset_external), \
                 which a copy cannot reproduce -- its data lives in files this crate \
                 does not read",
            ));
        }
        p = body_end;
    }
    Ok(())
}

/// Rejects source-file references in records prepared for a cross-file copy.
///
/// Checks shared records, shared attribute fields, and datatypes containing variable-length
/// values or references, including nested occurrences. In-file copies omit this check because
/// their source and destination share the same address space.
///
/// The message flags are defined in "Version 2 Data Object Header Prefix" of the [format
/// specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a source-file reference is present or a record extends
/// past the region. Returns [`Error::Format`] for datatype or attribute parse errors.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub(super) fn reject_foreign_addresses(region: &OhRegion) -> Result<(), Error> {
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        // The parsed record has a complete prefix, including the flags byte.
        if MessageFlags::new(region[p + 3]).is_shared() {
            return Err(Error::EditUnsupported(
                "a shared (committed/SOHM) object-header message cannot be copied to another file yet",
            ));
        }
        match msg_type {
            MessageType::DATATYPE => {
                let (dt, _) = hdf5_pure_format::__private::parse_datatype(&region[body..body_end])?;
                if datatype::datatype_holds_file_address(&dt) {
                    return Err(Error::EditUnsupported(
                        "variable-length or reference datasets cannot be copied to another file yet",
                    ));
                }
            }
            MessageType::ATTRIBUTE => {
                // Shared datatype and dataspace fields are separate from a shared attribute record.
                if crate::attribute::AttributeMessage::shares_a_field(&region[body..body_end]) {
                    return Err(Error::EditUnsupported(
                        "an attribute with a committed (shared) datatype cannot be copied to another file yet",
                    ));
                }
                let attr = crate::attribute::AttributeMessage::parse_with_length_width(
                    &region[body..body_end],
                    crate::file_writer::LENGTH_WIDTH,
                )?;
                if datatype::datatype_holds_file_address(&attr.datatype) {
                    return Err(Error::EditUnsupported(
                        "variable-length or reference attributes cannot be copied to another file yet",
                    ));
                }
            }
            _ => {}
        }
        p = body_end;
    }
    Ok(())
}

/// Rejects source-file references in parsed dense attribute values.
///
/// Complements [`reject_foreign_addresses`], which reads the retained header records.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if an attribute datatype contains variable-length values
/// or references, including nested occurrences.
fn reject_foreign_dense_attrs(attrs: &[crate::attribute::AttributeMessage]) -> Result<(), Error> {
    for attr in attrs {
        if datatype::datatype_holds_file_address(&attr.datatype) {
            return Err(Error::EditUnsupported(
                "variable-length or reference dense (fractal-heap) attributes cannot be copied to another file yet",
            ));
        }
    }
    Ok(())
}

/// Checks an in-file copy's references against addresses invalidated by the commit.
///
/// Checks contiguous and compact element bytes and inline and dense attribute values through
/// [`super::screen_resolved_references`]. Resolves committed datatype fields through `src`.
/// Chunk payloads are copied as stored, so a chunked dataset with object references is rejected
/// whenever `invalidated` is nonempty. Returns immediately if `invalidated` is empty.
///
/// The caller passes the session image framed at its base address as `src`. Message flags are
/// defined in "Version 2 Data Object Header Prefix" of the [format specification,
/// version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the copy contains shared attribute records, unreadable
/// compact reference elements, or chunked object references. Returns [`Error::CopyScreenUnreadable`]
/// if a shared datatype or an inline attribute cannot be read. Propagates record, datatype, and
/// reference screening errors.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub(super) fn screen_copied_references(
    tree: &CopyTree,
    addresses: AddressContext,
    widths: FormatWidths,
    invalidated: &InvalidatedAddresses,
    src: &(impl Source + ?Sized),
) -> Result<(), Error> {
    if invalidated.is_empty() {
        return Ok(());
    }
    use crate::shared_message::SharedResolver as _;
    // The resolver supports committed object headers without a shared-message table.
    let resolver = crate::shared_message::SourceResolver::new(
        src,
        addresses,
        AccessMode::ReadWrite,
        widths,
        None,
    );
    let (region, dense_attrs) = match tree {
        CopyTree::DatasetVerbatim {
            region,
            dense_attrs,
        }
        | CopyTree::DatasetContiguous {
            region,
            dense_attrs,
            ..
        }
        | CopyTree::DatasetChunked {
            region,
            dense_attrs,
            ..
        }
        | CopyTree::Group {
            non_link_region: region,
            dense_attrs,
            ..
        } => (region, dense_attrs),
    };
    for attr in &dense_attrs.attrs {
        super::screen_resolved_references(&attr.datatype, &attr.raw_data, invalidated)?;
    }

    // Read the element datatype, compact data, and inline attributes in one record traversal.
    let mut element_dt: Option<Datatype> = None;
    let mut compact: Option<Vec<u8>> = None;
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        // The parsed record has a complete prefix, including the flags byte.
        let shared = MessageFlags::new(region[p + 3]).is_shared();
        match msg_type {
            MessageType::DATATYPE => {
                // Resolve a committed datatype before parsing its encoding.
                let committed;
                let encoded = if shared {
                    committed = resolver
                        .resolve(&region[body..body_end], MessageType::DATATYPE)
                        .map_err(|source| Error::CopyScreenUnreadable {
                            message: MessageType::DATATYPE,
                            source,
                        })?;
                    &committed[..]
                } else {
                    &region[body..body_end]
                };
                let (dt, _) = hdf5_pure_format::__private::parse_datatype(encoded)?;
                element_dt = Some(dt);
            }
            MessageType::DATA_LAYOUT => {
                if let Ok(DataLayout::Compact { data }) =
                    DataLayout::parse_with_widths(&region[body..body_end], widths)
                {
                    compact = Some(data);
                }
            }
            MessageType::ATTRIBUTE => {
                // Shared attribute records require resolving the whole message before screening.
                if shared {
                    return Err(Error::EditUnsupported(
                        "a copy in this commit carries a shared (SOHM) attribute message, whose \
                         elements cannot be screened against the same commit's deletions; use \
                         separate commits",
                    ));
                }
                // Resolve the attribute's shared datatype and dataspace fields before screening.
                let attr = crate::attribute::AttributeMessage::parse_resolving_with_length_width(
                    &region[body..body_end],
                    widths.lengths,
                    &resolver,
                )
                .map_err(|source| Error::CopyScreenUnreadable {
                    message: MessageType::ATTRIBUTE,
                    source,
                })?;
                super::screen_resolved_references(&attr.datatype, &attr.raw_data, invalidated)?;
            }
            _ => {}
        }
        p = body_end;
    }

    match tree {
        // A declared reference requires readable compact element bytes.
        CopyTree::DatasetVerbatim { .. } => match (&element_dt, &compact) {
            (Some(dt), Some(data)) => super::screen_resolved_references(dt, data, invalidated)?,
            (Some(dt), None) if datatype::datatype_holds_object_address(dt) => {
                return Err(Error::EditUnsupported(
                    "a compact object-reference dataset's elements could not be read to screen \
                     them against this commit's deletions; use separate commits",
                ));
            }
            _ => {}
        },
        CopyTree::DatasetContiguous { data, .. } => {
            // Skip unallocated storage before rejecting unsupported reference encodings.
            if let (Some(dt), Some(data)) = (&element_dt, data) {
                super::screen_resolved_references(dt, data, invalidated)?;
            }
        }
        // Stored chunks are copied without decoding their element values.
        CopyTree::DatasetChunked { .. } => {
            if element_dt
                .as_ref()
                .is_some_and(datatype::datatype_holds_object_address)
            {
                return Err(Error::EditUnsupported(
                    "a chunked object-reference dataset cannot be copied in a commit that also \
                     deletes objects: its addresses live inside chunks this path does not \
                     decode; use separate commits",
                ));
            }
        }
        CopyTree::Group { children, .. } => {
            for (_, _, child) in children {
                screen_copied_references(child, addresses, widths, invalidated, src)?;
            }
        }
    }
    Ok(())
}

/// Provides stored chunk bytes from an owned copy in dense row-major order.
struct SliceChunkProvider<'a> {
    chunks: &'a [Vec<u8>],
}

impl ChunkProvider for SliceChunkProvider<'_> {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let chunk = self.chunks.get(index).ok_or_else(|| {
            FormatError::ChunkedReadError("chunk index out of range for in-memory provider".into())
        })?;
        out.extend_from_slice(chunk);
        Ok(())
    }
}

/// The subtree traversal limit, including possible hard-link cycles.
pub(super) const MAX_COPY_DEPTH: u32 = 1000;
