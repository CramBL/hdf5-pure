//! HDF5 file creation with compact or dense group links and attributes.

use core::num::NonZeroUsize;

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::format;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use hdf5_pure_core::__private::SuperblockFields;
use hdf5_pure_format::__private::ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES;
use hdf5_pure_format::__private::ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE;
use hdf5_pure_format::__private::ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;
use hdf5_pure_format::__private::ATTRIBUTE_HEAP_MAX_HEAP_SPACE;
use hdf5_pure_format::__private::ATTRIBUTE_HEAP_START_ROOT_ROWS;
use hdf5_pure_format::__private::ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE;
use hdf5_pure_format::__private::ATTRIBUTE_HEAP_TABLE_WIDTH;
use hdf5_pure_format::__private::AttributeRecord;
use hdf5_pure_format::__private::BTREE_V2_ATTRIBUTE_CREATION_ORDER;
use hdf5_pure_format::__private::BTREE_V2_ATTRIBUTE_NAME;
use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::BTREE_V2_NODE_SIZE;
use hdf5_pure_format::__private::BTreeV2Plan;
use hdf5_pure_format::__private::DenseLinkImage;
use hdf5_pure_format::__private::DenseLinkPlan;
use hdf5_pure_format::__private::DenseLinkPlanError;
use hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use hdf5_pure_format::__private::FractalHeapFiltering;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::HugeObjectRecord;
use hdf5_pure_format::__private::LayoutVersion;
use hdf5_pure_format::__private::LinkStorageError;
use hdf5_pure_format::__private::ManagedHeapPlan;
use hdf5_pure_format::__private::ManagedHeapPlanError;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::SECTION_CLASS_LARGE;
use hdf5_pure_format::__private::SECTION_CLASS_SMALL;
use hdf5_pure_space::__private::AllocationClass;

#[cfg(test)]
use crate::DeflateLevel;
use crate::FileSpaceSettings;
use crate::FillTime;
use crate::SuperblockVersion;
use crate::UserblockSize;
#[cfg(test)]
use crate::VariableLengthType;
#[cfg(test)]
use crate::address::AddressContext;
use crate::address::BaseAddressExt;
use crate::address::StoredAddressExt;
use crate::address::{BaseAddress, StoredAddress};
use crate::attribute::AttributeMessage;
use crate::attribute_info::AttributeInfoMessage;
use crate::chunked_write::{
    self, ByteSink, ChunkOptions, ChunkProvider, ChunkedMeasure, ChunkedPlan, CompressedChunkSet,
    StorageAllocation, VerbatimLayout, VerbatimPlan,
};
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, DataspaceType, Extent, MaxExtent};
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_allocation::FileAllocation;
use crate::file_create_properties::FileCreateProperties;
use crate::file_emission::Emission;
use crate::file_emission::EmissionError;
use crate::file_emission::Reservation;
use crate::file_space_info;
use crate::file_space_info::DEFAULT_THRESHOLD;
use crate::file_space_info::FileSpaceInfo;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::fill_value::FillPattern;
use crate::free_space_manager;
use crate::libver::LibVer;
use crate::link_info::LinkInfoMessage;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header_writer::ObjectHeaderWriter;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::shared_message::DatatypeLocation;
use crate::type_builders::ProducedPayload;
use crate::type_builders::RawChunkPayload;
use crate::type_builders::{
    self, AttrSpec, AttributeSpecs, CommittedDatatype, DatasetBuilder, FinishedGroup, GroupBuilder,
    VlStringStaging,
};
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;
use crate::{SuperblockStatus, SuperblockStatusFlags};

// `AttrValue` lives in `type_builders`; `types` and `mat` reference it through
// this module's path, so keep it re-exported here.
pub use crate::type_builders::AttrValue;

use crate::datatype::{CharacterSet, Datatype};

/// The width of every address in a file this crate writes.
pub(crate) const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
/// The width of every length in a file this crate writes.
pub(crate) const LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;
/// [`OFFSET_WIDTH`] and [`LENGTH_WIDTH`] as one [`FormatWidths`].
pub(crate) const WIDTHS: FormatWidths = FormatWidths {
    offsets: OFFSET_WIDTH,
    lengths: LENGTH_WIDTH,
};
/// [`OFFSET_WIDTH`] as the `u8` width the writers take.
pub(crate) const OFFSET_SIZE: u8 = OFFSET_WIDTH.get();
/// [`LENGTH_WIDTH`] as the `u8` width the writers take.
pub(crate) const LENGTH_SIZE: u8 = LENGTH_WIDTH.get();
const SUPERBLOCK_SIZE: usize = 48;

/// Threshold for switching from compact (inline) to dense attribute storage.
pub(crate) const DENSE_ATTR_THRESHOLD: usize = 8;

/// The Link Info message of a group that keeps its links in its object header and
/// does not track their creation order.
pub(crate) const COMPACT_LINK_INFO: LinkInfoMessage = LinkInfoMessage {
    max_creation_order: None,
    fractal_heap_address: None,
    btree_name_index_address: None,
    btree_creation_order_address: None,
};

// ---- OH builders ----

pub(crate) fn build_chunked_dataset_oh(
    dt: &Datatype,
    dt_location: &DatatypeLocation,
    ds: &Dataspace,
    layout_message: &[u8],
    pipeline_message: Option<&[u8]>,
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
    fill_time: FillTime,
    fill: Option<&[u8]>,
) -> Result<Vec<u8>, FormatError> {
    let mut w = ObjectHeaderWriter::new();
    add_datatype(&mut w, dt, dt_location);
    w.add_message(MessageType::DATASPACE, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(
        MessageType::FILL_VALUE,
        crate::fill_value::serialize_fill_value_message_v3(fill_time, fill)?,
        MessageFlags::CONSTANT,
    );
    w.add_message(MessageType::DATA_LAYOUT, layout_message.to_vec());
    if let Some(pm) = pipeline_message {
        w.add_message(MessageType::FILTER_PIPELINE, pm.to_vec());
    }
    add_attributes(&mut w, attrs, attr_info)?;
    w.serialize()
}

/// The superblock version the given on-disk format writes.
///
/// Version 3 arrived with HDF5 1.10. Its only difference from version 2 is the
/// meaning of the file-consistency-flags byte, which 1.10 uses to mark a file
/// held by a SWMR writer; the two encodings are otherwise identical, so the 1.8
/// format writes the same fields under the version number a 1.8 library
/// understands. That byte is why a SWMR writer needs the newer superblock, and
/// why [`File::open_swmr_writer`](crate::File::open_swmr_writer) requires one.
pub(crate) fn superblock_version(libver: LibVer) -> SuperblockVersion {
    if libver >= LibVer::V110 {
        SuperblockVersion::Three
    } else {
        SuperblockVersion::Two
    }
}

/// The data-layout message version the given on-disk format emits for a
/// *contiguous* dataset.
///
/// Version 4 arrived with HDF5 1.10, alongside the chunk indices that are its
/// only substantive addition; for a contiguous dataset the version 3 and version
/// 4 bodies are byte-identical (address then size), so the 1.8 format writes the
/// same bytes under the version number a 1.8 library understands.
pub(crate) fn contiguous_layout_version(libver: LibVer) -> LayoutVersion {
    if libver >= LibVer::V110 {
        LayoutVersion::Four
    } else {
        LayoutVersion::Three
    }
}

/// Builds a contiguous dataset's object header, whose data layout message stores
/// `data_addr` and the `data_size` bytes there.
pub(crate) fn build_dataset_oh(
    dt: &Datatype,
    dt_location: &DatatypeLocation,
    ds: &Dataspace,
    data_addr: StoredAddress,
    data_size: u64,
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
    fill_time: FillTime,
    fill: Option<&[u8]>,
    libver: LibVer,
) -> Result<Vec<u8>, FormatError> {
    let mut w = ObjectHeaderWriter::new();
    add_datatype(&mut w, dt, dt_location);
    w.add_message(MessageType::DATASPACE, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(
        MessageType::FILL_VALUE,
        crate::fill_value::serialize_fill_value_message_v3(fill_time, fill)?,
        MessageFlags::CONSTANT,
    );
    w.add_message(
        MessageType::DATA_LAYOUT,
        DataLayout::encode_contiguous(
            contiguous_layout_version(libver),
            data_addr,
            data_size,
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        ),
    );
    add_attributes(&mut w, attrs, attr_info)?;
    w.serialize()
}

/// Add a dataset's Datatype message: the encoding itself, or — when the type is
/// committed — the reference standing in for it, under a record whose shared flag
/// says the body is one. Without that flag the ten reference bytes decode as a
/// zero-width time datatype, which is the defect issue #254 was filed for.
fn add_datatype(w: &mut ObjectHeaderWriter, dt: &Datatype, location: &DatatypeLocation) {
    match location.reference_bytes(OFFSET_WIDTH) {
        Some(reference) => {
            w.add_message_with_flags(
                MessageType::DATATYPE,
                reference,
                MessageFlags::CONSTANT | MessageFlags::SHARED,
            );
        }
        None => w.add_message_with_flags(
            MessageType::DATATYPE,
            hdf5_pure_format::__private::serialize_datatype(dt),
            MessageFlags::CONSTANT,
        ),
    }
}

/// Build the object header of a committed (`H5Tcommit`) datatype object.
///
/// What makes the object a named datatype is the message set: the C library's
/// `H5O__dtype_isa` calls any header holding a Datatype message and no dataspace
/// or layout one a datatype, which is also how this crate's reader tells them
/// apart.
///
/// `references` is the object's link count *plus* every message that names it
/// through a shared reference, matching `H5O_link`, which the C library calls
/// once per hard link and once more each time a dataset or attribute is created
/// against the committed type. It goes in an Object Reference Count message,
/// which the format only carries above one — a header without one reads as
/// singly referenced.
///
/// Nothing *reading* the file notices a wrong count. It decides what a later
/// unlink does: the C library decrements it and deletes the object at zero, so a
/// count of 1 means unlinking the name takes the type away from every object
/// still naming it. (Measured here, the datasets kept reading afterwards — the
/// freed header had not been reused yet — which is what makes an undercount a
/// latent fault rather than an immediate one.)
///
/// The message flags match what libhdf5 writes: constant, since a committed
/// type's datatype never changes, and do-not-share, since this copy *is* the
/// shared one and must not be moved into shared storage again.
pub(crate) fn build_committed_datatype_oh(
    dt: &Datatype,
    references: u32,
) -> Result<Vec<u8>, FormatError> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message_with_flags(
        MessageType::DATATYPE,
        hdf5_pure_format::__private::serialize_datatype(dt),
        MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
    );
    if references > 1 {
        let mut refcount = Vec::with_capacity(5);
        refcount.push(0); // version
        refcount.extend_from_slice(&references.to_le_bytes());
        w.add_message_with_flags(
            MessageType::OBJECT_REFERENCE_COUNT,
            refcount,
            MessageFlags::FORBID_SHARING,
        );
    }
    w.serialize()
}

/// Builds a version 2 group header with compact or dense link storage.
///
/// Writes `links` into the header when `link_info` is [`None`]. Otherwise, writes
/// `link_info` and omits the inline links. Writes `attr_info` in place of compact attributes
/// when supplied.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if a compact attribute field exceeds its
/// encoding, or [`FormatError::ObjectHeaderMessageTooLarge`] if a message exceeds the
/// object-header message size limit.
pub(crate) fn build_group_oh(
    links: &[LinkMessage],
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
    link_info: Option<&LinkInfoMessage>,
) -> Result<Vec<u8>, FormatError> {
    let mut w = ObjectHeaderWriter::new();
    w.add_message(
        MessageType::LINK_INFO,
        link_info
            .unwrap_or(&COMPACT_LINK_INFO)
            .serialize(OFFSET_WIDTH),
    );
    // A new-style group (one with a Link Info message) must also carry a Group
    // Info message, or the HDF5 C library refuses to insert links into it:
    // `H5G_obj_insert` reads the Group Info message unconditionally and fails
    // with "message type not found", so the file is readable but not writable by
    // the C library. The minimal body (version 0, no optional fields) leaves the
    // C library to use its defaults (max compact = 8, min dense = 6).
    w.add_message(MessageType::GROUP_INFO, vec![0, 0]);
    if link_info.is_none() {
        for link in links {
            w.add_message(MessageType::LINK, link.serialize(OFFSET_WIDTH));
        }
    }
    add_attributes(&mut w, attrs, attr_info)?;
    w.serialize()
}

/// Serializes dense links at the reserved address and checks the reserved length.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the links require compact storage or the serialized
/// length differs from the reservation. Propagates errors from [`dense_links_plan`] and
/// [`DenseLinkPlan::serialize`].
fn dense_links_image(
    links: &[LinkMessage],
    (address, reserved): (StoredAddress, usize),
) -> Result<DenseLinkImage, FormatError> {
    let plan = dense_links_plan(links)?
        .ok_or_else(|| FormatError::Internal("a reserved dense group has compact links".into()))?;
    let image = plan.serialize(address).map_err(dense_link_error)?;
    if image.bytes.len() != reserved {
        return Err(FormatError::Internal(
            "dense group link emission differs from its reserved length".into(),
        ));
    }
    Ok(image)
}

/// Plans a group's dense link storage when its link count exceeds [`MAX_COMPACT_LINKS`].
///
/// Returns [`None`] for compact storage. Plans a managed fractal heap and a name index
/// without creation-order tracking.
///
/// # Errors
///
/// Returns [`FormatError::DenseLinkObjectTooLarge`] if an encoded link exceeds the
/// managed-object size limit, [`FormatError::DenseLinkStorageTooLarge`] if the heap or
/// indexes exceed their supported size, or [`FormatError::DuplicateLinkName`] if link names
/// repeat. Propagates errors from link parsing and heap planning.
fn dense_links_plan(links: &[LinkMessage]) -> Result<Option<DenseLinkPlan>, FormatError> {
    if links.len() <= MAX_COMPACT_LINKS {
        return Ok(None);
    }
    let links = links
        .iter()
        .map(|link| OwnedLink::parse(&link.serialize(OFFSET_WIDTH), OFFSET_WIDTH))
        .collect::<Result<Vec<_>, _>>()
        .map_err(link_storage_error)?;
    let storage = GroupLinkStorage::new(&COMPACT_LINK_INFO, links, OFFSET_WIDTH)
        .map_err(link_storage_error)?;
    DenseLinkPlan::new(storage, LENGTH_WIDTH)
        .map(Some)
        .map_err(dense_link_error)
}

fn dense_link_error(error: DenseLinkPlanError) -> FormatError {
    match error {
        DenseLinkPlanError::Format(error) => error,
        DenseLinkPlanError::Link(error) => link_storage_error(error),
        DenseLinkPlanError::Heap(ManagedHeapPlanError::ObjectSize { size, maximum }) => {
            FormatError::DenseLinkObjectTooLarge {
                size,
                limit: maximum,
            }
        }
        DenseLinkPlanError::Heap(ManagedHeapPlanError::Host { bytes }) => {
            FormatError::ValueTooLargeForPlatform {
                value: bytes,
                target: "usize",
            }
        }
        DenseLinkPlanError::Heap(ManagedHeapPlanError::HeapSpace)
        | DenseLinkPlanError::IndexGeometry
        | DenseLinkPlanError::LengthOverflow => FormatError::DenseLinkStorageTooLarge,
        DenseLinkPlanError::Heap(ManagedHeapPlanError::Geometry) => {
            FormatError::Internal("the group link heap geometry is invalid".into())
        }
    }
}

fn link_storage_error(error: LinkStorageError) -> FormatError {
    match error {
        LinkStorageError::Format(error) => error,
        LinkStorageError::DuplicateName(name) => {
            FormatError::DuplicateLinkName(String::from_utf8_lossy(&name).into_owned())
        }
        LinkStorageError::InvalidName(name) => {
            FormatError::InvalidLinkName(String::from_utf8_lossy(&name).into_owned())
        }
        error => FormatError::Internal(format!(
            "generated group link metadata is invalid: {error:?}"
        )),
    }
}

// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, is the default compact link limit.
const MAX_COMPACT_LINKS: usize = 8;

/// Add an object's attribute storage to its header: either the Attribute Info
/// message naming a dense (fractal-heap) attribute store, or the inline
/// Attribute messages preceded by the compact-storage Attribute Info message
/// that lets the reference library count them (see
/// [`compact_attribute_info_message`]).
///
/// The dense case takes the Attribute Info message rather than the heap itself
/// because that message is all a header needs. A sizing pass can therefore get
/// its header size from a [`DenseAttrPlan`] without the heap's bytes existing.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold.
fn add_attributes(
    w: &mut ObjectHeaderWriter,
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
) -> Result<(), FormatError> {
    if let Some(message) = attr_info {
        w.add_message(MessageType::ATTRIBUTE_INFO, message.to_vec());
    } else {
        if !attrs.is_empty() {
            w.add_message(
                MessageType::ATTRIBUTE_INFO,
                compact_attribute_info_message(),
            );
        }
        for attr in attrs {
            w.add_message(
                MessageType::ATTRIBUTE,
                attr.serialize(OFFSET_WIDTH, LENGTH_SIZE)?,
            );
        }
    }
    Ok(())
}

/// Returns the link name `name` spells.
///
/// A builder takes a name and returns a builder, so the write is where a name that is not a
/// component of an object path is rejected.
///
/// # Errors
///
/// Returns [`FormatError::InvalidLinkName`] if `name` is the empty name, `.`, or a name
/// holding `/`.
fn link_name(name: String) -> Result<LinkNameBuf, FormatError> {
    match LinkNameBuf::new(&name) {
        Some(link_name) => Ok(link_name),
        None => Err(FormatError::InvalidLinkName(name)),
    }
}

/// Returns a hard Link message that binds `name` to the object header at `addr`, with no creation
/// index.
pub(crate) fn make_link(name: &str, addr: StoredAddress) -> LinkMessage {
    LinkMessage {
        name: name.to_string(),
        link_target: LinkTarget::Hard {
            object_header_address: addr,
        },
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
}

// ---- Dense attribute blob ----

/// Pre-built dense attribute storage (fractal heap + B-tree v2 + attribute info message).
pub(crate) struct DenseAttrBlob {
    /// Serialized AttributeInfo message data (to embed in the object header).
    pub(crate) attr_info_message: Vec<u8>,
    /// The combined fractal heap header + direct block + B-tree v2 bytes.
    pub(crate) blob: Vec<u8>,
}

/// Bits of heap offset the dense attribute heap declares (its "Maximum Heap
/// Size"), and the byte width that implies for a block offset.
const DENSE_ATTR_MAX_HEAP_SIZE_BITS: u16 = ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;
const DENSE_ATTR_BLOCK_OFFSET_BYTES: usize = ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES;

/// Returns `true` if `attrs` require dense storage.
///
/// Selects dense storage when the attribute count exceeds [`DENSE_ATTR_THRESHOLD`] or a compact
/// message exceeds [`OBJECT_HEADER_MESSAGE_MAX`] bytes. Measures compact messages through
/// [`AttributeMessage::serialize`], including the character-set byte for non-ASCII names.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold.
pub(crate) fn needs_dense_attrs(attrs: &[AttributeMessage]) -> Result<bool, FormatError> {
    if attrs.len() > DENSE_ATTR_THRESHOLD {
        return Ok(true);
    }
    for a in attrs {
        if a.serialize(OFFSET_WIDTH, LENGTH_SIZE)?.len() > OBJECT_HEADER_MESSAGE_MAX {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The largest attribute [`build_dense_attrs`] stores as a managed object: what
/// is left of the heap's largest direct block once its header is accounted for.
///
/// The externally imposed limit is the heap ID: the 8-byte managed IDs this
/// emitter writes spend one byte on flags and [`DENSE_ATTR_BLOCK_OFFSET_BYTES`]
/// on the offset, leaving 2 bytes for the object's length, so no length above
/// 65,535 is representable at all. This constant is the slightly tighter value
/// the heap header also declares as its maximum managed object size, which the
/// reference C library then enforces on read: it rejects the oversized object as
/// one that should have been standalone, and an assertion-enabled build goes on
/// to abort while releasing the half-built attribute table.
///
/// An object past this belongs in fractal-heap *huge* storage, which
/// [`build_dense_attrs`] writes: the bytes go outside the managed blocks and a
/// huge-objects v2 B-tree maps a generated ID to their address and length. So
/// this is where the emitter changes representation, not where it gives up.
///
/// libhdf5 splits far earlier - it declares 4,096 for an
/// attribute heap — but the threshold is the heap's own declaration, read back
/// out of the header, so a higher one is equally readable.
pub(crate) const DENSE_ATTR_MAX_MANAGED_OBJECT: usize =
    hdf5_pure_format::__private::attribute_heap_max_managed_object(OFFSET_WIDTH);

/// The length of the heap IDs of the attribute heaps [`build_dense_attrs`] writes.
const DENSE_ATTR_HEAP_ID_LENGTH: u16 = 8;

/// One name-index B-tree v2 record as [`build_dense_attrs`] writes it.
const DENSE_ATTR_BTREE_RECORD: u16 = AttributeRecord::name_record_size(DENSE_ATTR_HEAP_ID_LENGTH);

/// One huge-objects B-tree v2 record as [`build_dense_attrs`] writes it.
const DENSE_ATTR_HUGE_BTREE_RECORD: u16 = HugeObjectRecord::size(OFFSET_WIDTH, LENGTH_WIDTH);

/// One creation-order B-tree v2 record as [`build_dense_attrs`] writes it.
const DENSE_ATTR_CORDER_BTREE_RECORD: u16 =
    AttributeRecord::creation_order_record_size(DENSE_ATTR_HEAP_ID_LENGTH);

/// What a dense attribute set records about attribute creation order.
///
/// An object tracks the order when its object header says so
/// (`H5Pset_attr_creation_order`, h5py's `track_order=True`, every object
/// netCDF-4 writes). The heap and the name index are the same either way; what
/// changes is that the Attribute Info message records the next index to hand
/// out, the name index records each attribute's real creation index rather than
/// its position, and an object that *indexes* the order gets a second B-tree
/// (type 9) over the same heap IDs, ordered by that index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum DenseAttrCreationOrder {
    /// Not tracked: what this crate's whole-file writer emits, and what an
    /// object edited in place keeps unless its header already tracked the order.
    #[default]
    Untracked,
    /// Tracked, with one creation index per attribute in the caller's order.
    Tracked {
        /// Creation index of each attribute, parallel to the attribute slice.
        indices: Vec<u16>,
        /// The next index the object would hand out. libhdf5
        /// does not lower this when an attribute is deleted, so it is carried
        /// rather than derived from `indices`.
        max: u16,
        /// Emit the creation-order B-tree beside the name index.
        indexed: bool,
    },
}

impl DenseAttrCreationOrder {
    /// The creation index recorded for attribute `i`, or `None` where the
    /// object does not track the order — or, for a tracked set, where `i` is
    /// past the attributes this describes.
    pub(crate) fn index_at(&self, i: usize) -> Option<u16> {
        match self {
            Self::Untracked => None,
            Self::Tracked { indices, .. } => indices.get(i).copied(),
        }
    }
}

/// Checks that [`build_dense_attrs`] can represent the attribute fields and managed heap.
///
/// Serializes the fields with [`AttributeMessage::serialize_v3`] and plans the managed blocks
/// with [`ManagedHeapPlan::attributes`]. Attributes larger than [`DENSE_ATTR_MAX_MANAGED_OBJECT`]
/// use huge-object storage.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold, [`FormatError::DenseAttributeHeapTooLarge`] if
/// the managed attributes need more space than the heap can address, and
/// [`FormatError::ValueTooLargeForPlatform`] if the size of the blocks that hold them does not fit
/// a `usize`. Returns [`FormatError::Internal`] if the fixed attribute-heap geometry or the
/// selected managed-object sizes violate the planner's invariants.
pub(crate) fn dense_attrs_check(attrs: &[AttributeMessage]) -> Result<(), FormatError> {
    let mut managed = Vec::new();
    for a in attrs {
        let size = a.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE)?.len();
        if size <= DENSE_ATTR_MAX_MANAGED_OBJECT {
            // A managed attribute's bytes go in the heap's direct blocks; a huge
            // one's sit outside them and are bounded only by the file.
            managed.push(size as u64);
        }
    }
    // The bound is checked by planning the very layout `build_dense_attrs` goes
    // on to emit, so the validated shape and the emitted shape cannot drift
    // apart. Reaching either refusal takes about a terabyte of attributes on one
    // object, which is why nothing exercises them end to end; the boundary each
    // draws is tested where it is computed.
    match ManagedHeapPlan::attributes(&managed, OFFSET_WIDTH) {
        Ok(_) => Ok(()),
        Err(ManagedHeapPlanError::HeapSpace) => Err(FormatError::DenseAttributeHeapTooLarge {
            limit: ATTRIBUTE_HEAP_MAX_HEAP_SPACE,
        }),
        Err(ManagedHeapPlanError::Host { bytes }) => Err(FormatError::ValueTooLargeForPlatform {
            value: bytes,
            target: "usize",
        }),
        Err(error @ (ManagedHeapPlanError::Geometry | ManagedHeapPlanError::ObjectSize { .. })) => {
            Err(FormatError::Internal(format!(
                "dense attribute managed heap planning failed: {error:?}"
            )))
        }
    }
}

/// Everything about a set of attributes' dense storage that does not depend on
/// where the heap is placed: the serialized attribute messages, the fractal
/// heap's doubling-table layout, both B-tree v2 layouts, and the blob's length.
///
/// [`DenseAttrPlan::build`] emits the blob at a chosen heap address, but no term
/// of this plan is that address — the length is the same for every base. That is
/// what lets a caller reserve the span before the bytes exist
/// ([`DenseAttrPlan::blob_len`]), and take that length from the very computation
/// the emission then works from rather than from a whole throwaway build.
pub(crate) struct DenseAttrPlan {
    /// Each attribute's version 3 message bytes and creation index, in the caller's order.
    attributes: Vec<PreparedDenseAttribute>,
    /// Huge-object ID for each attribute whose bytes go outside the managed
    /// blocks; `None` for a managed one.
    huge_id_of: Vec<Option<u64>>,
    huge_count: usize,
    huge_total: u64,
    managed_plan: ManagedHeapPlan,
    name_plan: BTreeV2Plan,
    corder_plan: Option<BTreeV2Plan>,
    huge_plan: Option<BTreeV2Plan>,
    /// `(name hash, attribute index)` in the order the name index is searched.
    order: Vec<(u32, u32)>,
    /// Attribute indices in the order the creation-order index is searched.
    /// Empty unless that index is emitted.
    corder_order: Vec<u32>,
    /// The next creation index, or `None` when creation order is untracked.
    next_index: Option<u16>,
    /// Whether the creation-order B-tree is emitted.
    indexed: bool,
    /// Where each part of the blob starts, relative to the heap's own address.
    managed_off: u64,
    btree_off: u64,
    name_nodes_off: u64,
    corder_bthd_off: u64,
    corder_nodes_off: u64,
    huge_bthd_off: u64,
    huge_nodes_off: u64,
    huge_data_off: u64,
    total_len: u64,
}

/// Lay out dense attribute storage for a set of attributes, without building it.
///
/// Attributes that fit [`DENSE_ATTR_MAX_MANAGED_OBJECT`] are stored as managed
/// objects in the heap's direct blocks; larger ones are stored as *huge*
/// objects, whose bytes sit outside the managed blocks and whose address and
/// length are indexed by a huge-objects v2 B-tree.
///
/// The caller must have checked [`dense_attrs_check`] first: an attribute set
/// past the heap's own address space cannot be laid out, and this emitter has
/// nowhere left to put it.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold.
/// Returns [`FormatError::DuplicateAttributeName`] if names repeat,
/// [`FormatError::DuplicateAttributeCreationIndex`] if tracked indexes repeat, or
/// [`FormatError::InvalidAttributeCreationCounter`] if a tracked index is at or above the
/// next creation index. Returns [`FormatError::Internal`] if the counts of attributes and
/// tracked indexes differ. Propagates errors from conversion of attribute positions.
pub(crate) fn dense_attrs_plan(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
) -> Result<DenseAttrPlan, FormatError> {
    let PreparedDenseAttrs {
        attributes,
        order,
        corder_order,
        next_index,
        indexed,
    } = PreparedDenseAttrs::new(attrs, creation)?;

    // Huge object IDs are assigned in attribute order, starting at 1 — the
    // reference C library's `H5HF__huge_insert` pre-increments, so 0 is never a
    // valid ID and the header's `next_huge_object_id` ends up holding the last
    // one assigned rather than the next one free.
    // A count of attributes, so a `usize` — it is bounded by `attrs.len()` and
    // widened only where it goes into one of the header's 8-byte fields.
    let mut huge_id_of: Vec<Option<u64>> = vec![None; attrs.len()];
    let mut huge_count: usize = 0;
    for (slot, attribute) in huge_id_of.iter_mut().zip(&attributes) {
        if attribute.bytes.len() > DENSE_ATTR_MAX_MANAGED_OBJECT {
            huge_count += 1;
            *slot = Some(huge_count as u64);
        }
    }
    let huge_total: u64 = attributes
        .iter()
        .zip(&huge_id_of)
        .filter(|(_, id)| id.is_some())
        .map(|(s, _)| s.bytes.len() as u64)
        .sum();

    // Only managed objects occupy the doubling table, so only they are planned
    // into it. `managed_sizes` holds their sizes in attribute order, and the
    // plan's object indices are indices into that order.
    let managed_sizes: Vec<u64> = attributes
        .iter()
        .zip(&huge_id_of)
        .filter(|(_, id)| id.is_none())
        .map(|(s, _)| s.bytes.len() as u64)
        .collect();
    let managed_plan = ManagedHeapPlan::attributes(&managed_sizes, OFFSET_WIDTH)
        .expect("dense_attrs_check, which every caller must run first, plans the same layout");

    // Every v2 B-tree header this emitter writes has the same fixed layout, so
    // one size covers both the name index and the huge-objects index.
    let bthd_size = hdf5_pure_format::__private::btree_v2_header_size(OFFSET_WIDTH, LENGTH_WIDTH);
    debug_assert_eq!(
        bthd_size,
        4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + OFFSET_SIZE as usize + 2 + LENGTH_SIZE as usize + 4
    );

    // Both indexes are planned before anything is placed, because their node
    // counts decide where everything after them goes. The node and record sizes
    // are this emitter's own constants, and every count is plannable at them —
    // a tree grows a level long before it could need an empty node — so a plan
    // here cannot fail on anything a caller controls.
    let name_plan = BTreeV2Plan::new(
        BTREE_V2_ATTRIBUTE_NAME,
        attrs.len(),
        DENSE_ATTR_BTREE_RECORD,
        BTREE_V2_NODE_SIZE,
        OFFSET_WIDTH,
    )
    .expect("a 512-byte node holds 29 name records, enough to plan any count");
    let corder_plan = indexed.then(|| {
        BTreeV2Plan::new(
            BTREE_V2_ATTRIBUTE_CREATION_ORDER,
            attrs.len(),
            DENSE_ATTR_CORDER_BTREE_RECORD,
            BTREE_V2_NODE_SIZE,
            OFFSET_WIDTH,
        )
        .expect("a 512-byte node holds 38 creation-order records, enough to plan any count")
    });
    let huge_plan = (huge_count > 0).then(|| {
        BTreeV2Plan::new(
            BTREE_V2_HUGE_OBJECT,
            huge_count,
            DENSE_ATTR_HUGE_BTREE_RECORD,
            BTREE_V2_NODE_SIZE,
            OFFSET_WIDTH,
        )
        .expect("a 512-byte node holds 20 huge records, enough to plan any count")
    });

    // Blob layout, all relative to the heap address the blob is built for, so
    // its length is the same wherever it lands: heap header, the managed blocks,
    // name index (header + nodes), then — only when there are huge objects — the
    // huge index (header + nodes) and the huge object bytes themselves.
    let managed_off = FractalHeapHeader::serialized_size(OFFSET_WIDTH, LENGTH_WIDTH) as u64;
    let btree_off = managed_off + managed_plan.region_size();
    let name_nodes_off = btree_off + bthd_size as u64;
    let corder_bthd_off = name_nodes_off + name_plan.nodes_size();
    let corder_nodes_off = corder_bthd_off
        + if corder_plan.is_some() {
            bthd_size as u64
        } else {
            0
        };
    let huge_bthd_off = corder_nodes_off + corder_plan.as_ref().map_or(0, BTreeV2Plan::nodes_size);
    let huge_nodes_off = huge_bthd_off + bthd_size as u64;
    let huge_data_off = huge_nodes_off + huge_plan.as_ref().map_or(0, BTreeV2Plan::nodes_size);
    // With no huge objects the blob ends where the huge index would have begun;
    // with them it ends past their bytes.
    let total_len = if huge_count > 0 {
        huge_data_off + huge_total
    } else {
        huge_bthd_off
    };

    Ok(DenseAttrPlan {
        attributes,
        huge_id_of,
        huge_count,
        huge_total,
        managed_plan,
        name_plan,
        corder_plan,
        huge_plan,
        order,
        corder_order,
        next_index,
        indexed,
        managed_off,
        btree_off,
        name_nodes_off,
        corder_bthd_off,
        corder_nodes_off,
        huge_bthd_off,
        huge_nodes_off,
        huge_data_off,
        total_len,
    })
}

impl DenseAttrPlan {
    /// Byte length of the blob [`DenseAttrPlan::build`] produces, at any heap
    /// address.
    pub(crate) fn blob_len(&self) -> u64 {
        self.total_len
    }

    /// The Attribute Info (0x0015) message naming this heap once it is placed at
    /// `heap_address`. Its length does not depend on the address, so an
    /// object-header sizing pass can take it from a provisional one.
    pub(crate) fn attr_info_message(&self, heap_address: StoredAddress) -> Vec<u8> {
        self.attribute_info(heap_address).serialize(OFFSET_WIDTH)
    }

    /// The Attribute Info message this storage is named by, once placed at
    /// `heap_address`.
    fn attribute_info(&self, heap_address: StoredAddress) -> AttributeInfoMessage {
        AttributeInfoMessage {
            max_creation_index: self.next_index,
            indexes_creation_order: self.indexed,
            fractal_heap_address: Some(heap_address),
            btree_name_index_address: Some(heap_address.offset(self.btree_off)),
            btree_creation_order_address: self
                .indexed
                .then(|| heap_address.offset(self.corder_bthd_off)),
        }
    }

    /// Emits the blob for a heap placed at `heap_address`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the heap ID of an attribute does not
    /// fit the 8-byte heap IDs of the heap. Returns [`FormatError::OffsetOverflow`] if a managed
    /// block's address or region end overflows `u64`, [`FormatError::AddressTooLarge`] if it
    /// exceeds the encoded address width, and [`FormatError::DefinedAddressIsUndefined`] if a
    /// required managed-heap address equals the width's undefined value.
    pub(crate) fn build(&self, heap_address: StoredAddress) -> Result<DenseAttrBlob, FormatError> {
        let max_heap_size: u16 = DENSE_ATTR_MAX_HEAP_SIZE_BITS;
        let block_offset_bytes = DENSE_ATTR_BLOCK_OFFSET_BYTES; // 5
        let heap_id_length = DENSE_ATTR_HEAP_ID_LENGTH;

        let Self {
            attributes,
            huge_id_of,
            huge_count,
            huge_total,
            managed_plan,
            name_plan,
            corder_plan,
            huge_plan,
            order,
            corder_order,
            ..
        } = self;
        let huge_count = *huge_count;

        // Every address the blob embeds is `heap_address + <plan offset>`, which
        // is what makes the bytes relocatable: the same plan emits the same
        // length wherever the heap is placed.
        let address_of = |offset: u64| heap_address.offset(offset);
        let frhp_addr = heap_address;
        let managed_addr = address_of(self.managed_off);
        let name_nodes_addr = address_of(self.name_nodes_off);
        let corder_nodes_addr = address_of(self.corder_nodes_off);
        let huge_bthd_addr = address_of(self.huge_bthd_off);
        let huge_nodes_addr = address_of(self.huge_nodes_off);
        let huge_data_addr = address_of(self.huge_data_off);

        // libhdf5 does not read a heap ID's length field at a fixed
        // width: it derives that width from the heap's declared maximum managed
        // object size, then decodes `1 + offset_bytes + length_bytes` from the ID. If
        // that total ever exceeded `heap_id_length` it would read past the ID stored
        // in each B-tree record. Keeping the declared maximum pinned to
        // DENSE_ATTR_MAX_MANAGED_OBJECT is what holds the two in agreement, so assert
        // it here rather than leaving it to the constant's doc comment.
        debug_assert_eq!(
            1 + block_offset_bytes + encoded_size_width(DENSE_ATTR_MAX_MANAGED_OBJECT as u64),
            heap_id_length as usize,
            "managed heap ID width must match what the declared maximum managed object size implies"
        );

        // Deliberately a constant rather than a function of `max_direct_block_size`:
        // this is the per-object cap the 2-byte length field of an 8-byte managed
        // heap ID can encode, so it must not grow with the block. `dense_attrs_check`
        // bounds every attribute by the same constant.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "DENSE_ATTR_MAX_MANAGED_OBJECT is 65,514, well inside the 4-byte \
                      max-managed-object-size field"
        )]
        let max_managed = DENSE_ATTR_MAX_MANAGED_OBJECT as u32;
        let header = FractalHeapHeader {
            heap_id_length,
            filtering: FractalHeapFiltering::Unfiltered,
            flags: FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
            max_managed_object_size: max_managed,
            next_huge_object_id: huge_count as u64,
            btree_huge_objects_address: if huge_count == 0 {
                UNDEF_ADDRESS
            } else {
                huge_bthd_addr
            },
            free_space_in_managed_blocks: managed_plan.free_space(),
            managed_block_free_space_manager_address: UNDEF_ADDRESS,
            managed_space: managed_plan.managed_space(),
            allocated_managed_space: managed_plan.allocated_space(),
            direct_block_allocation_iterator_offset: managed_plan.allocation_iterator(),
            // Managed and huge objects are counted separately, and an attribute is
            // in exactly one of the two.
            managed_objects_count: (attributes.len() - huge_count) as u64,
            huge_objects_size: *huge_total,
            huge_objects_count: huge_count as u64,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: ATTRIBUTE_HEAP_TABLE_WIDTH,
            starting_block_size: ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
            max_direct_block_size: ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE,
            max_heap_size,
            start_root_rows: ATTRIBUTE_HEAP_START_ROOT_ROWS,
            root_block_address: managed_plan.root_address(managed_addr)?,
            // Zero when the root is a direct block, which is how a reader decides
            // which of the two the address above points at.
            current_rows_in_root_indirect_block: managed_plan.root_rows(),
        };
        let frhp = header.serialize(OFFSET_WIDTH, LENGTH_WIDTH)?;

        // Heap IDs. A managed object's carries the offset the plan gave it; a huge
        // object's carries a B-tree key instead, since its bytes sit outside the
        // managed blocks entirely.
        let mut heap_ids: Vec<Vec<u8>> = Vec::with_capacity(attributes.len());
        // (huge object ID, address, length) for the huge-objects B-tree, in ID order.
        let mut huge_records: Vec<(u64, StoredAddress, u64)> = Vec::with_capacity(huge_count);
        // The managed objects' bytes in attribute order, which is the order the
        // plan assigned their heap offsets in.
        let mut managed: Vec<&[u8]> = Vec::with_capacity(attributes.len() - huge_count);
        let mut next_huge_addr = huge_data_addr;
        for (s, huge_id) in attributes.iter().zip(huge_id_of) {
            match huge_id {
                Some(id) => {
                    huge_records.push((*id, next_huge_addr, s.bytes.len() as u64));
                    next_huge_addr = next_huge_addr.offset(s.bytes.len() as u64);
                    heap_ids.push(header.encode_huge_id(*id, OFFSET_WIDTH, LENGTH_WIDTH)?);
                }
                None => {
                    heap_ids.push(header.encode_managed_id(
                        managed_plan.heap_offset(managed.len()),
                        s.bytes.len() as u64,
                    )?);
                    managed.push(s.bytes.as_slice());
                }
            }
        }

        let managed_blocks = managed_plan.serialize(&managed, managed_addr, frhp_addr)?;
        debug_assert_eq!(managed_blocks.len() as u64, managed_plan.region_size());

        // Build B-tree v2 type 8 records (17 bytes each), in the order the plan
        // sorted them into — the order the index is searched in.
        let mut name_records =
            Vec::with_capacity(attributes.len() * usize::from(DENSE_ATTR_BTREE_RECORD));
        for &(hash, i) in order {
            AttributeRecord {
                heap_id: &heap_ids[i as usize],
                flags: MessageFlags::NONE,
                creation_order: attributes[i as usize].creation_index,
            }
            .encode_name_record(&mut name_records, hash);
        }

        let name_tree =
            name_plan.serialize(&name_records, name_nodes_addr, OFFSET_WIDTH, LENGTH_WIDTH);

        let mut blob = Vec::with_capacity(self.total_len.to_usize().unwrap_or(0));
        blob.extend_from_slice(&frhp);
        blob.extend_from_slice(&managed_blocks);
        debug_assert_eq!(blob.len() as u64, self.btree_off);
        blob.extend_from_slice(&name_tree.header);
        blob.extend_from_slice(&name_tree.nodes);

        if let Some(corder_plan) = corder_plan {
            // The same heap IDs the name index carries, ordered by creation
            // index and without the name hash: this is the index `H5Aiterate2`
            // walks for `H5_INDEX_CRT_ORDER`.
            let mut corder_records =
                Vec::with_capacity(corder_order.len() * DENSE_ATTR_CORDER_BTREE_RECORD as usize);
            for &i in corder_order {
                AttributeRecord {
                    heap_id: &heap_ids[i as usize],
                    flags: MessageFlags::NONE,
                    creation_order: attributes[i as usize].creation_index,
                }
                .encode_creation_order_record(&mut corder_records);
            }
            let corder_tree = corder_plan.serialize(
                &corder_records,
                corder_nodes_addr,
                OFFSET_WIDTH,
                LENGTH_WIDTH,
            );
            debug_assert_eq!(blob.len() as u64, self.corder_bthd_off);
            blob.extend_from_slice(&corder_tree.header);
            blob.extend_from_slice(&corder_tree.nodes);
        }

        if let Some(huge_plan) = huge_plan {
            // Records are already in ascending ID order, which is the order the
            // B-tree is searched in.
            let mut huge_bytes =
                Vec::with_capacity(huge_records.len() * DENSE_ATTR_HUGE_BTREE_RECORD as usize);
            for (id, addr, len) in &huge_records {
                HugeObjectRecord {
                    address: *addr,
                    length: *len,
                    id: *id,
                }
                .encode(&mut huge_bytes, OFFSET_WIDTH, LENGTH_WIDTH);
            }
            let huge_tree =
                huge_plan.serialize(&huge_bytes, huge_nodes_addr, OFFSET_WIDTH, LENGTH_WIDTH);

            debug_assert_eq!(blob.len() as u64, self.huge_bthd_off);
            blob.extend_from_slice(&huge_tree.header);
            blob.extend_from_slice(&huge_tree.nodes);
            debug_assert_eq!(blob.len() as u64, self.huge_data_off);
            for (s, huge_id) in attributes.iter().zip(huge_id_of) {
                if huge_id.is_some() {
                    blob.extend_from_slice(&s.bytes);
                }
            }
        }

        // The length `blob_len` promises a caller reserving a span for this heap,
        // checked against the bytes actually produced. A reservation that
        // disagreed with the emission would put the next object on top of this
        // one, so pin it where it is emitted as well as in a test.
        debug_assert_eq!(
            blob.len() as u64,
            self.total_len,
            "a dense attribute heap must fill the length its plan promised"
        );

        Ok(DenseAttrBlob {
            attr_info_message: self.attr_info_message(frhp_addr),
            blob,
        })
    }
}

/// The heap address pass 1 of a file build asks a [`DenseAttrPlan`] for its
/// Attribute Info message at, before the heap's real address is known. The
/// message's *length* is what that pass needs, and that does not depend on the
/// address; pass 2 emits the real message at the address it reserves.
const DUMMY_DENSE_BASE: StoredAddress = StoredAddress::ZERO;

/// Builds dense attribute storage for a set of attributes, at a known address.
///
/// A caller that has to reserve the heap's span before its bytes exist wants
/// [`dense_attrs_plan`] and [`DenseAttrPlan::blob_len`] instead; this is the
/// one-shot form for callers that already know where the heap goes.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold. Propagates errors from [`dense_attrs_plan`]
/// and [`DenseAttrPlan::build`].
pub(crate) fn build_dense_attrs(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
    heap_address: StoredAddress,
) -> Result<DenseAttrBlob, FormatError> {
    dense_attrs_plan(attrs, creation)?.build(heap_address)
}

/// Encoded attributes with unique names and creation indexes.
///
/// Tracked indexes are below the stored counter. Gaps and the counter are preserved
/// independently of the live attributes.
#[derive(Debug)]
struct PreparedDenseAttrs {
    attributes: Vec<PreparedDenseAttribute>,
    /// `(name hash, attribute index)` pairs sorted by hash and then by name.
    order: Vec<(u32, u32)>,
    /// Attribute indexes sorted by creation index for the optional creation-order B-tree.
    corder_order: Vec<u32>,
    next_index: Option<u16>,
    indexed: bool,
}

impl PreparedDenseAttrs {
    /// Encodes attributes and pairs each message with its creation index.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the counts of attributes and tracked indexes differ.
    /// Returns the metadata errors documented by [`dense_attrs_plan`] for repeated names or
    /// indexes and inconsistent counters. Propagates attribute encoding and position conversion
    /// errors.
    fn new(
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
    ) -> Result<Self, FormatError> {
        let (indices, next_index, indexed) = match creation {
            DenseAttrCreationOrder::Untracked => (None, None, false),
            DenseAttrCreationOrder::Tracked {
                indices,
                max,
                indexed,
            } => {
                if indices.len() != attrs.len() {
                    return Err(FormatError::Internal(
                        "a tracked attribute set needs one creation index per attribute".into(),
                    ));
                }
                (Some(indices), Some(max), indexed)
            }
        };
        let mut attributes = Vec::with_capacity(attrs.len());
        let mut order = Vec::with_capacity(attrs.len());
        for (i, attribute) in attrs.iter().enumerate() {
            let position = i.narrow::<u32>()?;
            let creation_index = match &indices {
                Some(indices) => {
                    let index = indices[i];
                    if let Some(next) = next_index {
                        if index >= next {
                            return Err(FormatError::InvalidAttributeCreationCounter {
                                index,
                                next,
                            });
                        }
                    }
                    u32::from(index)
                }
                None => position,
            };
            attributes.push(PreparedDenseAttribute {
                bytes: attribute.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE)?,
                creation_index,
            });
            order.push((
                crate::checksum::jenkins_lookup3(attribute.name.as_bytes()),
                position,
            ));
        }
        order.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| attrs[a.1 as usize].name.cmp(&attrs[b.1 as usize].name))
        });
        for pair in order.windows(2) {
            let left = &attrs[pair[0].1 as usize].name;
            let right = &attrs[pair[1].1 as usize].name;
            if left == right {
                return Err(FormatError::DuplicateAttributeName(left.clone()));
            }
        }
        let mut corder_order = if next_index.is_some() {
            (0..attrs.len().narrow::<u32>()?).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        corder_order.sort_unstable_by_key(|&i| attributes[i as usize].creation_index);
        for pair in corder_order.windows(2) {
            let left = attributes[pair[0] as usize].creation_index;
            let right = attributes[pair[1] as usize].creation_index;
            if left == right {
                return Err(FormatError::DuplicateAttributeCreationIndex(left));
            }
        }
        Ok(Self {
            attributes,
            order,
            corder_order: if indexed { corder_order } else { Vec::new() },
            next_index,
            indexed,
        })
    }
}

/// A version 3 attribute message paired with its creation index.
///
/// For untracked attributes, the creation index is the position in the caller's set.
#[derive(Debug)]
struct PreparedDenseAttribute {
    bytes: Vec<u8>,
    creation_index: u32,
}

/// Bytes libhdf5 uses to encode a limit of `value`
/// (`H5VM_limit_enc_size`): the width of the smallest field that can hold it.
fn encoded_size_width(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).div_ceil(8).max(1)
}

/// The Attribute Info (0x0015) message a version 2 object header needs when its
/// attributes are stored *compactly*, as inline Attribute messages.
///
/// On a version 2 header libhdf5 does not count attribute
/// messages to answer `H5Oget_info().num_attrs`; it reads the Attribute Info
/// message and, finding an undefined fractal-heap address there, falls back to
/// the header's own attribute-message tally (`H5A__get_ainfo`, `H5Aint.c`). With
/// no Attribute Info message at all, `H5O__attr_count_real` reports zero — while
/// `H5Aiterate` and `H5Aopen_by_name` still find every attribute. Tools that size
/// their work by the count then skip the attributes silently: `h5repack` copies
/// such an object with none of them.
///
/// So the message is emitted for a compactly-stored attribute set too, with both
/// addresses undefined, which is what the C library and h5py write in the same
/// position. An object with no attributes gets no message, again matching them.
pub(crate) fn compact_attribute_info_message() -> Vec<u8> {
    AttributeInfoMessage {
        max_creation_index: None,
        indexes_creation_order: false,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    }
    .serialize(OFFSET_WIDTH)
}

/// The undefined address at the [`OFFSET_SIZE`] address width this writer emits.
///
/// The writer stores it in a data layout message for a dataset with no storage
/// allocated, and in an object reference whose path it places nowhere. The
/// undefined address is defined in "Appendix A: Definitions" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_appendixa
pub(crate) const UNDEF_ADDRESS: StoredAddress = StoredAddress::undefined(OffsetWidth::Eight);

// ---- FileWriter ----

/// An in-memory builder for an HDF5 file.
///
/// Stores up to eight links per group in the object header. For larger groups, including
/// the root group, writes a fractal heap and a name index. Limits encoded dense link messages
/// to 4096 bytes and omits creation-order tracking.
pub struct FileWriter {
    root_datasets: Vec<DatasetBuilder>,
    root_attrs: AttributeSpecs,
    root_committed: Vec<CommittedDatatype>,
    groups: Vec<FinishedGroup>,
    userblock_size: UserblockSize,
    /// Bytes written at the start of the prefix, from
    /// [`with_userblock_content`](FileWriter::with_userblock_content). The writer fills the
    /// remaining prefix with zeros.
    userblock_content: Vec<u8>,
    /// Requested library-version bounds (low, high), validated in `finish`.
    /// `None` means no constraint (any output the writer produces is accepted).
    libver_bounds: Option<(LibVer, LibVer)>,
    file_space_settings: Option<FileSpaceSettings>,
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriter {
    pub fn new() -> Self {
        Self {
            root_datasets: Vec::new(),
            root_attrs: AttributeSpecs::default(),
            root_committed: Vec::new(),
            groups: Vec::new(),
            userblock_size: UserblockSize::NONE,
            userblock_content: Vec::new(),
            libver_bounds: None,
            file_space_settings: None,
        }
    }

    /// Constrain the on-disk format version of the file, mirroring HDF5's
    /// `H5Pset_libver_bounds`. The file is written in the newest format the
    /// bounds allow, between [`LibVer::WRITER_OLDEST`] and
    /// [`LibVer::WRITER_DEFAULT`]; bounds that leave no such format fail with
    /// [`FormatError::LibverBoundsUnsatisfiable`].
    ///
    /// `high` is what selects the format, so `Earliest..=V18` writes the 1.8
    /// format and anything reaching 1.10 or beyond writes the 1.10 one. This
    /// differs from the C library, which picks the *oldest* format the content
    /// needs and treats `low` as the floor; the difference shows on
    /// `Earliest..=Latest`, where `H5Fcreate` writes a version 0 superblock and
    /// this writes a version 3 one. Leaving the bounds unset is the same as
    /// leaving `high` at `Latest`. `low` only rules formats out, licensing newer
    /// encodings without requiring them, so a lower bound of `V112`, `V114` or
    /// `LATEST` is satisfied by the 1.10 format rather than refused — provided
    /// `high` reaches it. An inverted range such as `V114..=V110` is refused with
    /// [`FormatError::LibverBoundsUnsatisfiable`], as `H5Pset_libver_bounds`
    /// refuses one.
    ///
    /// Some content cannot be written in the 1.8 format at all, and asking for
    /// both is an error rather than a silent upgrade: a chunked (or resizable,
    /// or filtered) dataset needs the version 4 data-layout message and the
    /// chunk indices that came with 1.10, and a file-space setting — a strategy
    /// or a page size — needs the File Space Info message. Both report
    /// [`FormatError::LibverTooOldForContent`].
    pub fn with_libver_bounds(&mut self, low: LibVer, high: LibVer) -> &mut Self {
        self.libver_bounds = Some((low, high));
        self
    }

    /// Replace *every* file-creation property with what `properties` carries,
    /// including the ones it leaves unset — handing over a property list is
    /// asking for it to define the creation properties in full, which is what
    /// [`FileBuilder::with_create_properties`] documents.
    ///
    /// Written as one assignment per field rather than as a run of `if let
    /// Some(..)` calls to the individual setters, because those skip the fields
    /// the list does not carry: a bound set before the call would survive it and
    /// go on selecting the on-disk format for a list that names no version at
    /// all.
    ///
    /// [`FileBuilder::with_create_properties`]: crate::FileBuilder::with_create_properties
    pub(crate) fn apply_create_properties(&mut self, properties: &FileCreateProperties) {
        self.userblock_size = properties.userblock();
        self.libver_bounds = properties.libver_bounds();
        self.file_space_settings = properties.file_space_settings();
    }

    /// The format to write, being the newest this crate produces that the
    /// requested [`libver_bounds`](Self::libver_bounds) admit.
    ///
    /// Resolved once, before any layout work, so every version field in the file
    /// comes from one decision rather than from each emitter's own reading of the
    /// bounds.
    fn resolve_libver(&self) -> Result<LibVer, FormatError> {
        LibVer::resolve_writable(self.libver_bounds)
    }

    /// Sets the size of the reserved prefix before the HDF5 signature.
    ///
    /// The writer fills the prefix with zeros unless content is supplied through
    /// [`with_userblock_content`](Self::with_userblock_content).
    pub fn with_userblock(&mut self, size: UserblockSize) -> &mut Self {
        self.userblock_size = size;
        self
    }

    /// Sets the bytes at the start of the reserved userblock.
    ///
    /// The writer fills the remaining prefix with zeros. [`finish`](Self::finish) returns
    /// [`FormatError::UserblockContentTooLarge`] if the content exceeds the configured size.
    pub fn with_userblock_content(&mut self, content: &[u8]) -> &mut Self {
        self.userblock_content = content.to_vec();
        self
    }

    /// Emit the userblock region: the caller's content, then zeros out to `ub`.
    /// Callers have already checked that the content fits.
    fn put_userblock<S: ByteSink>(
        sink: &mut S,
        ub: usize,
        content: &[u8],
    ) -> Result<(), EmissionError<S::Error>> {
        if ub == 0 {
            return Ok(());
        }
        sink.put(content).map_err(EmissionError::Sink)?;
        sink.put_zeros(ub - content.len())
            .map_err(EmissionError::Sink)
    }

    /// Sets the complete file-space configuration for the emitted File Space Info message.
    pub fn with_file_space_settings(&mut self, settings: FileSpaceSettings) -> &mut Self {
        self.file_space_settings = Some(settings);
        self
    }

    /// Whether anything staged needs the 1.10 format, i.e. whether an
    /// `Earliest..=V18` bound would be refused with
    /// [`FormatError::LibverTooOldForContent`].
    ///
    /// This is the same predicate `finish_to_sink` refuses on — chunked storage,
    /// which a filter and an unlimited dimension both imply — asked *before* the
    /// bound is chosen. Repack is the caller: it carries the source file's format
    /// forward, and needs to know whether the content it staged permits that
    /// before committing to an answer, since a great many files the C library
    /// wrote hold chunked datasets under a version 0 or 2 superblock and there is
    /// no older chunk index this crate can write them back into.
    pub(crate) fn needs_latest_format(&self) -> bool {
        fn any_chunked(datasets: &[DatasetBuilder]) -> bool {
            datasets.iter().any(|d| {
                d.chunk_options.is_chunked() || d.maxshape.is_some() || d.raw_chunks.is_some()
            })
        }
        fn group_needs(group: &FinishedGroup) -> bool {
            any_chunked(&group.datasets) || group.sub_groups.iter().any(group_needs)
        }
        self.file_space_settings.is_some()
            || any_chunked(&self.root_datasets)
            || self.groups.iter().any(group_needs)
    }

    pub fn create_group(&mut self, name: &str) -> GroupBuilder {
        GroupBuilder::new(name)
    }

    /// Configures and attaches a group in the root group after the callback succeeds.
    ///
    /// See [`FileBuilder::create_group_with`](crate::FileBuilder::create_group_with) for the
    /// callback and attachment behavior.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLinkName`] if `name` is empty, is `.`, or contains `/`,
    /// before calling `configure`. Returns the callback's error if `configure` fails.
    pub fn create_group_with(
        &mut self,
        name: &str,
        configure: impl FnOnce(&mut GroupBuilder) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        self.groups.push(GroupBuilder::build_with(name, configure)?);
        Ok(())
    }

    pub fn add_group(&mut self, group: FinishedGroup) {
        self.groups.push(group);
    }

    pub fn create_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.root_datasets.push(DatasetBuilder::new(name));
        self.root_datasets.last_mut().unwrap()
    }

    /// Commit `datatype` in the root group under `name`.
    ///
    /// See [`FileBuilder::commit_datatype`](crate::FileBuilder::commit_datatype).
    pub fn commit_datatype(&mut self, name: &str, datatype: Datatype) {
        self.root_committed.push(CommittedDatatype {
            name: name.to_string(),
            datatype,
        });
    }

    /// Stages a root-group attribute using the committed datatype at `path`.
    ///
    /// The datatype implied by `value` must match the datatype committed at `path`.
    /// See [`Self::set_root_attr`] for replacement and ordering.
    pub fn set_root_attr_committed(&mut self, name: &str, value: AttrValue, path: &str) {
        self.root_attrs.insert(
            name.to_string(),
            crate::type_builders::committed_attr_spec(value, path),
        );
    }

    /// Stages an attribute for the root group.
    ///
    /// The writer replaces a staged attribute with the same name and preserves its first insertion
    /// position. The replacement includes its value and datatype information.
    pub fn set_root_attr(&mut self, name: &str, value: AttrValue) {
        self.root_attrs
            .insert(name.to_string(), AttrSpec::Value(value));
    }

    /// Attaches an already-encoded attribute message to the root group, written as given except
    /// for its datatype location.
    ///
    /// See [`Self::set_root_attr`] for replacement and ordering.
    ///
    /// Where `committed_datatype_path` is set, the written message refers to the committed
    /// datatype at that path. See [`AttrSpec::Verbatim`] for what this preserves that
    /// `set_root_attr` cannot, and for the datatypes it must not be used with.
    pub(crate) fn set_root_attr_verbatim(
        &mut self,
        message: crate::attribute::AttributeMessage,
        committed_datatype_path: Option<ObjectPathBuf>,
    ) {
        self.root_attrs.insert(
            message.name.clone(),
            AttrSpec::Verbatim {
                message,
                committed_datatype_path,
            },
        );
    }

    /// Attaches a variable-length string attribute to the root group with the datatype and
    /// dataspace of `message`, staging `strings` into a heap of this file's own.
    ///
    /// See [`Self::set_root_attr`] for replacement and ordering.
    ///
    /// Where `committed_datatype_path` is set, the written message refers to the committed
    /// datatype at that path. See [`AttrSpec::VerbatimVarLen`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if a string exceeds
    /// [`u32::MAX`] bytes.
    pub(crate) fn set_root_attr_var_len_verbatim(
        &mut self,
        mut message: crate::attribute::AttributeMessage,
        strings: Vec<String>,
        committed_datatype_path: Option<ObjectPathBuf>,
    ) -> Result<(), FormatError> {
        message.raw_data = crate::type_builders::vl_string_reference_bytes(&strings)?;
        self.root_attrs.insert(
            message.name.clone(),
            AttrSpec::VerbatimVarLen {
                message,
                strings,
                committed_datatype_path,
            },
        );
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>, FormatError> {
        let mut buf = Vec::new();
        self.finish_to_sink(&mut buf)
            .map_err(EmissionError::into_format)?;
        Ok(buf)
    }

    /// Assembles the file and writes it to `sink` in ascending-address order.
    ///
    /// Buffered and streaming sinks receive byte-identical files. The writer reads a streamed
    /// dataset's chunk bytes from its provider one chunk at a time.
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Format`] if planning or serialization fails, including capacity
    /// errors from [`ByteSink::reserve`]. Returns [`EmissionError::Sink`] with the original output
    /// error if a write fails. Either kind of failure can leave partially written output.
    pub(crate) fn finish_to_sink<S: ByteSink>(
        self,
        sink: &mut S,
    ) -> Result<(), EmissionError<S::Error>> {
        let libver = self.resolve_libver()?;
        let file_space = self.file_space_settings;
        let reuse_threshold = match file_space {
            None => Some(DEFAULT_THRESHOLD),
            Some(settings) if settings.strategy() == FileSpaceStrategy::FsmAggr => {
                Some(settings.threshold())
            }
            Some(_) => None,
        };
        let aggregated = reuse_threshold.is_some()
            || file_space.is_some_and(|settings| settings.strategy() == FileSpaceStrategy::Aggr);
        let mut allocation =
            FileAllocation::new(SUPERBLOCK_SIZE as u64, aggregated, reuse_threshold);

        // File-space settings are recorded in a File Space Info message, which
        // arrived with HDF5 1.10. Refused beside the bounds themselves rather
        // than where the message is emitted, so the answer does not depend on
        // how far into the layout the writer got.
        if file_space.is_some() && libver < LibVer::V110 {
            return Err(FormatError::LibverTooOldForContent {
                content: "a file-space setting",
                needs: LibVer::V110.name(),
                writing: libver.name(),
            }
            .into());
        }

        // Checked before any layout work: content that overruns its region would
        // otherwise displace the superblock, and the failure would surface as an
        // unopenable file rather than as the caller's mistake.
        if self.userblock_content.len() as u64 > self.userblock_size.get() {
            return Err(FormatError::UserblockContentTooLarge {
                content: self.userblock_content.len() as u64,
                userblock: self.userblock_size.get(),
            }
            .into());
        }

        // Genuine paged allocation: page-align every allocation and, when
        // persisting, emit per-page-type free-space managers. Gated entirely on
        // the Page strategy so every other strategy keeps its exact byte layout.
        let paged = match file_space {
            Some(settings) if settings.strategy() == FileSpaceStrategy::Page => {
                let page_size = settings.page_size();
                // File-space pages are measured from the file base; the layout
                // below is base-relative, so base-relative boundaries coincide
                // with absolute ones only when the userblock is a whole number of
                // pages (zero trivially qualifies).
                if self.userblock_size.get() % page_size.get() != 0 {
                    return Err(FormatError::UserblockNotPageAligned(
                        self.userblock_size.get(),
                        page_size.get(),
                    )
                    .into());
                }
                Some((page_size, settings.persist(), settings.threshold()))
            }
            _ => None,
        };

        // The superblock-extension header (carrying a File Space Info message)
        // is independent of the file layout, so build it up front and place it
        // after all other content below.
        let ext_oh = file_space
            .map(|settings| file_space_extension_oh(&file_space_placeholder(settings)))
            .transpose()?;
        // Persistent File Space Info requires final manager addresses and the EOA before
        // manager allocation. Retain the settings for the rewrite after placement.
        let nonpaged_persist = file_space.filter(|settings| {
            settings.strategy() != FileSpaceStrategy::Page && settings.persist()
        });
        struct DsFlat {
            /// Link name in the owning group.
            name: LinkNameBuf,
            dt: Datatype,
            /// Where `dt` is written: in this dataset's header, or in a committed
            /// datatype object it names.
            dt_location: DatatypeLocation,
            /// The path of the committed datatype `dt_location` refers to. `dt_location`
            /// holds the undefined address until the writer places that datatype.
            committed_datatype_path: Option<ObjectPathBuf>,
            ds: Dataspace,
            storage: PreparedStorage,
            attrs: Vec<AttributeMessage>,
            /// The reserved content hash, completed after resolving element addresses.
            #[cfg(feature = "provenance")]
            pending_hash: Option<PendingHash>,
            /// The committed datatype path of each attribute, parallel to `attrs`.
            attr_committed_datatype_paths: Vec<Option<ObjectPathBuf>>,
            chunk_options: ChunkOptions,
            maxshape: Option<Vec<MaxExtent>>,
            reference_targets: Option<Vec<crate::type_builders::ObjectRefPatch>>,
            /// Global heap collections and patch offsets for variable-length strings whose
            /// element bytes are held in `storage`. The writer patches their heap addresses
            /// after placing the collections.
            vl_string_staging: Option<VlStringStaging>,
            /// A user-defined fill value, encoded in the dataset's datatype, or
            /// `None` for the library default. Validated against the datatype
            /// element size in `flatten_dataset`.
            fill: Option<Vec<u8>>,
            /// The fill time policy for the dataset.
            fill_time: FillTime,
            /// The byte length this dataset's contiguous data-layout message
            /// records — the region it holds, or the extent an unallocated one
            /// would occupy (issue #293). Derived in `flatten_dataset` so the
            /// sizing pass and the emit pass, which reach the region by different
            /// routes, cannot disagree about it. A chunked dataset describes its
            /// storage in its index instead and never reads this.
            declared_contiguous_len: u64,
        }

        impl DsFlat {
            /// Bytes this dataset's contiguous data region occupies. Produced
            /// data has none in hand, but its total is known from the geometry —
            /// which is the whole reason it can be laid out without being read.
            fn contiguous_len(&self) -> u64 {
                match self.storage.produced() {
                    Some(p) => p.total_bytes,
                    None => self.storage.raw().len() as u64,
                }
            }

            /// Removes this dataset's contiguous buffer or describes its produced region.
            ///
            /// A produced region's provider remains in the dataset's prepared storage.
            fn take_contiguous(&mut self) -> DsData<'static> {
                match self.storage.produced() {
                    Some(p) => DsData::Produced {
                        total_bytes: p.total_bytes,
                        block_bytes: p.block_bytes,
                    },
                    None => DsData::InMemory(self.storage.take_raw()),
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
            fn compress_chunks(&self) -> Result<CompressedChunkSet, FormatError> {
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
        struct PendingHash {
            /// The index in the dataset's flattened attribute list.
            attribute: usize,
        }

        #[cfg(feature = "provenance")]
        impl PendingHash {
            /// Appends hash metadata with a 64-byte digest placeholder.
            fn reserve(
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
            fn complete(
                self,
                attrs: &mut [AttributeMessage],
                bytes: &[u8],
            ) -> Result<(), FormatError> {
                let attribute = attrs.get_mut(self.attribute).ok_or_else(|| {
                    FormatError::Internal(
                        "the reserved provenance digest attribute is absent".into(),
                    )
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

        /// A dataset's data region, retained as bytes or emitted from a plan.
        enum DsData<'a> {
            /// Encoded chunks borrowed from the writer's retained set, with finalized metadata.
            Encoded {
                chunks: &'a CompressedChunkSet,
                plan: ChunkedPlan,
            },
            InMemory(Vec<u8>),
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
            /// Returns the contiguous element buffer used to patch variable-length references.
            fn in_memory_mut(&mut self) -> Option<&mut Vec<u8>> {
                match self {
                    DsData::InMemory(bytes) => Some(bytes),
                    DsData::Encoded { .. } | DsData::Streamed(_) | DsData::Produced { .. } => None,
                }
            }

            /// Writes the data region, including a chunk index for chunked storage.
            ///
            /// # Errors
            ///
            /// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if a streamed
            /// or produced region is paired with prepared storage of another kind. Propagates
            /// provider and serialization errors as [`EmissionError::Format`] and sink errors as
            /// [`EmissionError::Sink`].
            fn emit<S: ByteSink>(
                &self,
                sink: &mut S,
                storage: &PreparedStorage,
            ) -> Result<(), EmissionError<S::Error>> {
                match self {
                    DsData::Encoded { chunks, plan } => {
                        chunked_write::emit_chunked_data(sink, chunks, plan)
                    }
                    DsData::InMemory(bytes) => sink.put(bytes).map_err(EmissionError::Sink),
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
                            FormatError::Internal(
                                "a produced region requires a prepared producer".into(),
                            )
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

            fn len(&self) -> u64 {
                match self {
                    DsData::Encoded { plan, .. } => plan.data_len,
                    DsData::InMemory(v) => v.len() as u64,
                    DsData::Streamed(plan) => plan.total_len,
                    DsData::Produced { total_bytes, .. } => *total_bytes,
                }
            }
        }

        /// Returns the error for a variable-length dataset whose element bytes
        /// are not held in memory when its global heap references are patched.
        fn vl_element_bytes_absent(link_name: &str) -> FormatError {
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
        fn emit_produced_data<Sk: ByteSink>(
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
        struct ChunkedBuilt<'a> {
            layout_message: Vec<u8>,
            pipeline_message: Option<Vec<u8>>,
            data: DsData<'a>,
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
        fn measure_chunked(
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
                let set = chunk_set
                    .expect("an encode-path chunked dataset must have a precomputed chunk set");
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
        fn build_chunked<'a>(
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
                let set = chunk_set
                    .expect("an encode-path chunked dataset must have a precomputed chunk set");
                let plan = chunked_write::plan_chunked_at(set, data_address)?;
                Ok(ChunkedBuilt {
                    layout_message: plan.layout_message.clone(),
                    pipeline_message: plan.pipeline_message.clone(),
                    data: DsData::Encoded { chunks: set, plan },
                })
            }
        }
        struct GrpFlat {
            /// Link name in the owning group.
            name: LinkNameBuf,
            attrs: Vec<AttributeMessage>,
            /// The committed datatype path of each attribute, parallel to `attrs`.
            attr_committed_datatype_paths: Vec<Option<ObjectPathBuf>>,
            ds_indices: Vec<usize>,
            sub_group_indices: Vec<usize>,
            committed_indices: Vec<usize>,
        }

        /// One committed datatype object, flattened out of the group tree.
        struct CtFlat {
            /// Link name in the owning group.
            name: LinkNameBuf,
            dt: Datatype,
            /// Hard links plus shared references, completed once every dataset
            /// and attribute in the file has been counted (see
            /// `register_committed_use`).
            references: u32,
        }

        let mut all_ds: Vec<DsFlat> = Vec::new();
        let mut groups: Vec<GrpFlat> = Vec::new();
        let mut committed: Vec<CtFlat> = Vec::new();
        let mut root_ds_indices: Vec<usize> = Vec::new();
        let mut root_group_indices: Vec<usize> = Vec::new();

        fn flatten_dataset(
            db: DatasetBuilder,
            all_ds: &mut Vec<DsFlat>,
            ds_vl: &mut Vec<Vec<VlPatch>>,
        ) -> Result<usize, FormatError> {
            let DatasetBuilder {
                name,
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
            let name = link_name(name)?;
            let dt = datatype.ok_or(FormatError::DatasetMissingDatatype)?;
            let shape = shape.ok_or(FormatError::DatasetMissingShape)?;
            let storage =
                PreparedStorage::parse(data, raw_chunks, produced, allocation, shape.contains(&0))?;
            let unallocated = storage.allocation() == StorageAllocation::Unallocated;
            // A zero-byte element type is refused before it reaches a layout
            // decision: the chunk splitter clamps a row copy to an element
            // boundary with `% element_size` and would divide by zero, and a
            // contiguous write of one produces a file this crate's own reader
            // refuses. A caller-built `Datatype` reaches the writer without
            // passing through `Datatype::parse`, which refuses the file-sourced
            // ones, so this is where the same invariant is held on the way out.
            //
            // Taking the size as a `NonZeroU32` here is what holds that for the
            // rest of the write: every later stage receives the proof rather
            // than the bare number, so none of them re-checks it.
            let element_size = dt.element_size()?;
            let elem_size = element_size.narrow::<NonZeroUsize>()?;
            let elem_bytes = u64::from(element_size.get());
            let rank = match u8::try_from(shape.len()) {
                Ok(rank) if rank <= MAX_DATASET_RANK => rank,
                _ => {
                    return Err(FormatError::DatasetRankUnsupported {
                        rank: shape.len(),
                        maximum: MAX_DATASET_RANK,
                    });
                }
            };
            if let Some(dimension) = shape.iter().position(|&size| size == UNLIMITED_DIMENSION) {
                return Err(FormatError::DatasetDimensionUnlimited { dimension });
            }
            let extent = if shape.contains(&0) {
                0
            } else {
                shape.iter().try_fold(elem_bytes, |size, &dimension| {
                    size.checked_mul(dimension)
                        .ok_or(FormatError::DatasetSizeOverflow { size, dimension })
                })?
            };
            #[cfg(feature = "provenance")]
            if unallocated && extent != 0 && provenance.is_some() {
                return Err(FormatError::DatasetProvenanceRequiresData);
            }
            // The bytes this dataset's data region occupies. A produced region is
            // measured by the same rule as a materialized one — its size is
            // declared rather than counted, and a declaration that disagrees with
            // the shape would produce a file the reader refuses.
            let region_len = storage
                .produced()
                .map_or(storage.raw().len() as u64, |p| p.total_bytes);
            // What a contiguous data-layout message records for this dataset: the
            // region it actually holds, or — for a dataset that allocates no
            // storage at all — the extent it *would* occupy beside the undefined
            // address, which is what the reference library records there and what
            // makes `Layout::Contiguous`'s promise that `size` is "the extent that
            // would be written" hold for a file this crate produced (issue #293).
            // Derived once, so the sizing pass and the emit pass — which reach the
            // region by different routes — cannot disagree about it. Read only by
            // the contiguous layout message; a chunked dataset describes its
            // storage in its index instead.
            let declared_contiguous_len = if unallocated { extent } else { region_len };
            // Compare the shape's byte extent with the supplied or produced bytes, including
            // zero-element shapes. Verbatim chunks have no flat region to compare, and
            // unallocated storage has no element payload.
            if !unallocated && storage.chunks().is_none() && region_len != extent {
                return Err(FormatError::LogicalShapeDataMismatch {
                    expected: extent,
                    actual: region_len,
                    element_size: element_size.into(),
                });
            }
            // Validate the chunk geometry up front for a chunked / filtered /
            // extensible dataset, so a malformed request (chunk dimensions of the
            // wrong rank, a zero chunk dimension, a bad maximum shape, or
            // chunking a scalar) is refused here instead of panicking in the
            // chunk splitter or producing an unreadable dataset.
            if chunk_options.is_chunked() || maxshape.is_some() {
                Extent::new(&shape, maxshape.as_deref())
                    .and_then(|geometry| chunk_options.validate_geometry(geometry))
                    .map_err(FormatError::InvalidChunkGeometry)?;
            }
            chunk_options.validate_partial_edge_filtering(maxshape.as_deref())?;
            // Variable-length string element references live in the global heap.
            // For chunked/filtered/resizable storage the references sit inside
            // chunks that are split (and possibly compressed) before the rest of
            // the file is laid out, so their heap addresses cannot be patched in
            // afterwards. Such a dataset instead has its collections placed
            // *ahead* of everything else, at a fixed address known before any
            // chunk is encoded — see the early-placement block in
            // `finish_to_sink` that fills `early_gcol`. Nothing is refused here.
            let max_dimensions = maxshape.clone();
            let dspace = Dataspace {
                space_type: if shape.is_empty() {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                },
                rank,
                dimensions: shape,
                max_dimensions,
            };
            let mut attrs = Vec::new();
            let mut attr_committed_datatype_paths = Vec::new();
            for (n, v) in &attribute_specs {
                attrs.push(AttributeName::try_from(n.as_str())?.to_message(v)?);
                attr_committed_datatype_paths.push(v.committed_datatype_path().cloned());
            }
            let patches = collect_vl_patches(&attribute_specs);
            #[cfg(feature = "provenance")]
            let pending_hash = provenance.map(|prov| {
                PendingHash::reserve(
                    &mut attrs,
                    crate::provenance::Provenance {
                        creator: prov.creator,
                        timestamp: prov.timestamp,
                        source: prov.source,
                    },
                )
            });
            // A user-defined fill value is one element wide, so its byte length
            // must equal the datatype's element size.
            if let Some(fill) = &fill {
                if fill.len() != elem_size.get() {
                    return Err(FormatError::FillValueSizeMismatch {
                        expected: elem_size.get(),
                        actual: fill.len(),
                    });
                }
            }
            let idx = all_ds.len();
            all_ds.push(DsFlat {
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
                reference_targets,
                vl_string_staging,
                fill,
                fill_time: fill_time.unwrap_or_default(),
                declared_contiguous_len,
            });
            ds_vl.push(patches);
            Ok(idx)
        }

        /// Flatten one group's committed datatypes into the file-wide list,
        /// returning their indices.
        ///
        /// A committed type is written as an object of its own and named by
        /// datasets and attributes, so a zero-byte element type is refused here
        /// for the same reason `flatten_dataset` refuses one: it reaches the
        /// writer without passing through `Datatype::parse`.
        fn flatten_committed(
            types: Vec<CommittedDatatype>,
            committed: &mut Vec<CtFlat>,
        ) -> Result<Vec<usize>, FormatError> {
            types
                .into_iter()
                .map(|ct| {
                    ct.datatype.element_size()?;
                    committed.push(CtFlat {
                        name: link_name(ct.name)?,
                        dt: ct.datatype,
                        // One hard link. Every shared reference to it adds one
                        // more, counted once the whole file is flattened.
                        references: 1,
                    });
                    Ok(committed.len() - 1)
                })
                .collect()
        }

        fn flatten_group(
            g: FinishedGroup,
            all_ds: &mut Vec<DsFlat>,
            groups: &mut Vec<GrpFlat>,
            committed: &mut Vec<CtFlat>,
            grp_vl: &mut Vec<Vec<VlPatch>>,
            ds_vl: &mut Vec<Vec<VlPatch>>,
        ) -> Result<usize, FormatError> {
            let name = g.name?;
            let mut gattrs = Vec::new();
            let mut attr_committed_datatype_paths = Vec::new();
            for (n, v) in &g.attrs {
                gattrs.push(AttributeName::try_from(n.as_str())?.to_message(v)?);
                attr_committed_datatype_paths.push(v.committed_datatype_path().cloned());
            }
            let patches = collect_vl_patches(&g.attrs);
            let committed_idx = flatten_committed(g.committed, committed)?;
            let mut ds_idx = Vec::new();
            for db in g.datasets {
                ds_idx.push(flatten_dataset(db, all_ds, ds_vl)?);
            }
            let mut sub_grp_idx = Vec::new();
            for sg in g.sub_groups {
                sub_grp_idx.push(flatten_group(sg, all_ds, groups, committed, grp_vl, ds_vl)?);
            }
            let gi = groups.len();
            groups.push(GrpFlat {
                name,
                attrs: gattrs,
                attr_committed_datatype_paths,
                ds_indices: ds_idx,
                sub_group_indices: sub_grp_idx,
                committed_indices: committed_idx,
            });
            grp_vl.push(patches);
            Ok(gi)
        }

        let mut grp_vl: Vec<Vec<VlPatch>> = Vec::new();
        let mut ds_vl: Vec<Vec<VlPatch>> = Vec::new();

        let root_committed_indices = flatten_committed(self.root_committed, &mut committed)?;

        for db in self.root_datasets {
            root_ds_indices.push(flatten_dataset(db, &mut all_ds, &mut ds_vl)?);
        }

        for g in self.groups.into_iter() {
            root_group_indices.push(flatten_group(
                g,
                &mut all_ds,
                &mut groups,
                &mut committed,
                &mut grp_vl,
                &mut ds_vl,
            )?);
        }

        // Build global heap collections for variable-length string attributes.
        // Track which attribute messages need VL patching, across root, groups, and datasets.
        struct VlPatch {
            /// The attribute's collections, in the order their objects appear.
            collections: Vec<Vec<u8>>,
            attr_index: usize, // index into the relevant attrs Vec
        }

        /// Reserves raw collection allocations and returns addresses for reference patching.
        ///
        /// If `retain` is set, appends the addresses to `retained` for address-ordered emission.
        /// Under direct placement, the writer derives emission addresses from collection lengths.
        ///
        /// # Errors
        ///
        /// Returns [`FormatError::OffsetOverflow`] if a reservation exceeds [`u64::MAX`], or
        /// [`FormatError::Internal`] if an aggregate extends beyond the writer's EOA.
        fn place_collections(
            collections: &[Vec<u8>],
            allocation: &mut FileAllocation,
            retained: &mut Vec<StoredAddress>,
            retain: bool,
        ) -> Result<Vec<u64>, FormatError> {
            let mut addresses = Vec::with_capacity(collections.len());
            for collection in collections {
                let address = allocation.reserve(collection.len() as u64, AllocationClass::Raw)?;
                if retain {
                    retained.push(address);
                }
                addresses.push(address.get());
            }
            Ok(addresses)
        }

        fn collect_vl_patches(attrs_raw: &AttributeSpecs) -> Vec<VlPatch> {
            let mut patches = Vec::new();
            for (i, (_n, v)) in attrs_raw.iter().enumerate() {
                if let Some(strings) = v.var_len_strings() {
                    patches.push(VlPatch {
                        collections: type_builders::build_global_heap_collections(strings),
                        attr_index: i,
                    });
                }
            }
            patches
        }

        let mut root_attrs: Vec<AttributeMessage> = Vec::new();
        let mut root_attr_committed_datatype_paths = Vec::new();
        for (n, v) in &self.root_attrs {
            root_attrs.push(AttributeName::try_from(n.as_str())?.to_message(v)?);
            root_attr_committed_datatype_paths.push(v.committed_datatype_path().cloned());
        }

        let vl_root = collect_vl_patches(&self.root_attrs);

        // ---- Committed datatypes: paths, reference counts, and agreement ----
        //
        // Done before any sizing, because a committed type's own header carries
        // its reference count and so changes length with it, and because a
        // dataset that names a type the file does not commit must be refused
        // before the layout is half built.
        let committed_by_path: HashMap<ObjectPathBuf, usize> = {
            fn walk(
                prefix: &ObjectPathBuf,
                gi: usize,
                groups: &[GrpFlat],
                committed: &[CtFlat],
                out: &mut HashMap<ObjectPathBuf, usize>,
            ) {
                for &ci in &groups[gi].committed_indices {
                    out.insert(prefix.join(&committed[ci].name), ci);
                }
                for &sgi in &groups[gi].sub_group_indices {
                    walk(&prefix.join(&groups[sgi].name), sgi, groups, committed, out);
                }
            }
            let root = ObjectPathBuf::root();
            let mut out: HashMap<ObjectPathBuf, usize> = root_committed_indices
                .iter()
                .map(|&ci| (root.join(&committed[ci].name), ci))
                .collect();
            for &gi in &root_group_indices {
                walk(
                    &root.join(&groups[gi].name),
                    gi,
                    &groups,
                    &committed,
                    &mut out,
                );
            }
            out
        };

        /// Register one use of a committed datatype: resolve the path it names,
        /// check that the user and the committed object describe the same type,
        /// and add the use to the object's reference count.
        ///
        /// The agreement check is what keeps the two encodings from disagreeing.
        /// A reader takes the committed one — it is the only one in the file — so
        /// a dataset written with a mismatched inline declaration would have its
        /// element bytes read under a type it was not encoded in, silently.
        ///
        /// `user` names the object for that error and is built only on that
        /// path: every dataset and every attribute in every file passes through
        /// here, and almost none of them name a committed type at all.
        fn register_committed_use(
            committed: &mut [CtFlat],
            by_path: &HashMap<ObjectPathBuf, usize>,
            path: &ObjectPathBuf,
            dt: &Datatype,
            user: impl FnOnce() -> String,
        ) -> Result<(), FormatError> {
            let Some(&ci) = by_path.get(path) else {
                return Err(FormatError::UnknownCommittedDatatype(path.to_string()));
            };
            if hdf5_pure_format::__private::serialize_datatype(&committed[ci].dt)
                != hdf5_pure_format::__private::serialize_datatype(dt)
            {
                return Err(FormatError::CommittedDatatypeMismatch {
                    path: path.to_string(),
                    user: user(),
                });
            }
            committed[ci].references = committed[ci].references.saturating_add(1);
            Ok(())
        }

        for (attr, path) in root_attrs.iter().zip(&root_attr_committed_datatype_paths) {
            let Some(path) = path else {
                continue;
            };
            register_committed_use(
                &mut committed,
                &committed_by_path,
                path,
                &attr.datatype,
                || format!("root attribute {:?}", attr.name),
            )?;
        }
        for g in &groups {
            for (attr, path) in g.attrs.iter().zip(&g.attr_committed_datatype_paths) {
                let Some(path) = path else {
                    continue;
                };
                register_committed_use(
                    &mut committed,
                    &committed_by_path,
                    path,
                    &attr.datatype,
                    || format!("attribute {:?} of group {:?}", attr.name, g.name.as_str()),
                )?;
            }
        }
        for d in &all_ds {
            if let Some(path) = &d.committed_datatype_path {
                register_committed_use(&mut committed, &committed_by_path, path, &d.dt, || {
                    format!("dataset {:?}", d.name.as_str())
                })?;
            }
            for (attr, path) in d.attrs.iter().zip(&d.attr_committed_datatype_paths) {
                let Some(path) = path else {
                    continue;
                };
                register_committed_use(
                    &mut committed,
                    &committed_by_path,
                    path,
                    &attr.datatype,
                    || format!("attribute {:?} of dataset {:?}", attr.name, d.name.as_str()),
                )?;
            }
        }

        // Every committed type's header is now final: its content is a datatype
        // and a reference count, neither of which depends on where anything in
        // the file lands. So it is built once here and only placed below, unlike
        // every other header, which is sized against dummy addresses and rebuilt.
        let committed_oh: Vec<Vec<u8>> = committed
            .iter()
            .map(|ct| build_committed_datatype_oh(&ct.dt, ct.references))
            .collect::<Result<_, _>>()?;

        /// Everything a link can name, paired with the address assigned to it.
        ///
        /// The sizing pass and the two emission paths differ only in which
        /// addresses they hold — dummy zeros against real ones — so they build
        /// this and ask it the same question rather than assembling links
        /// apiece.
        struct LinkTables<'a> {
            all_ds: &'a [DsFlat],
            committed: &'a [CtFlat],
            groups: &'a [GrpFlat],
            ds_addrs: &'a [StoredAddress],
            committed_addrs: &'a [StoredAddress],
            group_addrs: &'a [StoredAddress],
        }

        impl LinkTables<'_> {
            /// The links one group carries, in the order they are written: its
            /// datasets, its committed datatypes, then its subgroups.
            ///
            /// Assembled in one place because the sizing pass and the two
            /// emission paths have to agree exactly on what a group contains. A
            /// link the sizing pass does not know about is a header longer than
            /// the span reserved for it, which silently moves every object after
            /// it.
            fn links(
                &self,
                ds_indices: &[usize],
                committed_indices: &[usize],
                sub_group_indices: &[usize],
            ) -> Vec<LinkMessage> {
                let mut links = Vec::with_capacity(
                    ds_indices.len() + committed_indices.len() + sub_group_indices.len(),
                );
                for &i in ds_indices {
                    links.push(make_link(self.all_ds[i].name.as_str(), self.ds_addrs[i]));
                }
                for &ci in committed_indices {
                    links.push(make_link(
                        self.committed[ci].name.as_str(),
                        self.committed_addrs[ci],
                    ));
                }
                for &gi in sub_group_indices {
                    links.push(make_link(
                        self.groups[gi].name.as_str(),
                        self.group_addrs[gi],
                    ));
                }
                links
            }
        }

        // Addresses the sizing pass uses. A link's byte length does not depend on
        // the address it carries, so zeros give the real header size.
        let dummy_ds_addrs = vec![StoredAddress::new(0); all_ds.len()];
        let dummy_ct_addrs = vec![StoredAddress::new(0); committed.len()];
        let dummy_grp_addrs = vec![StoredAddress::new(0); groups.len()];

        let root_dense = needs_dense_attrs(&root_attrs)?;
        let group_dense: Vec<bool> = groups
            .iter()
            .map(|g| needs_dense_attrs(&g.attrs))
            .collect::<Result<_, _>>()?;
        let ds_dense: Vec<bool> = all_ds
            .iter()
            .map(|d| needs_dense_attrs(&d.attrs))
            .collect::<Result<_, _>>()?;

        // A compact attribute is stored as an object-header message, whose size
        // field is 2 bytes wide. An oversized one would be written with a
        // truncated length, which silently loses the attribute or leaves the
        // reader parsing the next message body as a message header, so refuse it
        // here — while the attribute's name is still at hand — before any header
        // is built. `ObjectHeaderWriter::serialize` is the unnamed backstop for
        // every other message this writer emits.
        //
        // This runs before the compression pass below so a file that will be
        // refused does not first pay to compress every chunked dataset in it.
        //
        // Dense attributes are stored in a fractal heap rather than the header, so
        // this limit does not apply to them and the check skips them; the dense
        // check just below bounds those instead. An attribute that would exceed it
        // is *why* `needs_dense_attrs` picked the dense path, so what reaches here
        // is only what a header can hold — this is a backstop, not the decision.
        fn check_compact_attrs(attrs: &[AttributeMessage]) -> Result<(), FormatError> {
            for a in attrs {
                let size = a.serialize(OFFSET_WIDTH, LENGTH_SIZE)?.len();
                if size > OBJECT_HEADER_MESSAGE_MAX {
                    return Err(FormatError::AttributeMessageTooLarge {
                        name: a.name.clone(),
                        size,
                    });
                }
            }
            Ok(())
        }
        // The dense emitter has bounds of its own — the attribute message's own
        // 2-byte header fields and the size of the one direct block it builds —
        // which it documents its callers must check. An attribute set past them
        // was previously written anyway, producing a heap that reads back empty
        // here and aborts an assertion-enabled reference C library (issue #191).
        // Between this and the compact check above, every attribute is bounded on
        // whichever path it takes. No bound here is on an attribute's size or on
        // how many there are: the first is what selects dense storage rather than
        // what it refuses, and the second is unbounded now that both indexes are
        // multi-level B-trees.
        if root_dense {
            dense_attrs_check(&root_attrs)?;
        } else {
            check_compact_attrs(&root_attrs)?;
        }
        for (gi, g) in groups.iter().enumerate() {
            if group_dense[gi] {
                dense_attrs_check(&g.attrs)?;
            } else {
                check_compact_attrs(&g.attrs)?;
            }
        }
        for (i, d) in all_ds.iter().enumerate() {
            if ds_dense[i] {
                dense_attrs_check(&d.attrs)?;
            } else {
                check_compact_attrs(&d.attrs)?;
            }
        }

        let is_chunked: Vec<bool> = all_ds
            .iter()
            .map(|d| {
                d.chunk_options.is_chunked() || d.maxshape.is_some() || d.storage.chunks().is_some()
            })
            .collect();

        // Chunked storage here means a version 4 data-layout message and one of
        // the chunk indices HDF5 1.10 introduced (extensible array, fixed array,
        // single chunk, implicit). The 1.8 format indexes chunks with a version 1
        // B-tree instead, which this crate reads but does not write, so there is
        // no older encoding to fall back to. Refused as soon as the flattened
        // datasets are known, which is still before any byte reaches the sink.
        //
        // A filter or an unlimited dimension arrives here too: both require
        // chunked storage, so `is_chunked` already covers them.
        if libver < LibVer::V110 && is_chunked.iter().any(|&c| c) {
            return Err(FormatError::LibverTooOldForContent {
                content: "a chunked, filtered, or resizable dataset",
                needs: LibVer::V110.name(),
                writing: libver.name(),
            }
            .into());
        }

        // Reserve chunked datasets' global heaps before encoding their references into chunks.
        // Under PAGE, reserve every collection in raw pages before the object headers. Under
        // other strategies, reserve the remaining collections after reserving dataset data.
        let mut early_gcol: Vec<usize> = Vec::new();
        let mut paged_collection_addresses = Vec::new();
        let mut aggregate_collection_addresses = Vec::new();
        let mut collection_meta_sections = Vec::new();
        let mut collection_raw_sections = Vec::new();
        let mut collection_large_sections = Vec::new();
        let early_gcol_size = if let Some((page_size, _, _)) = paged {
            let has_collections = vl_root.iter().any(|patch| !patch.collections.is_empty())
                || grp_vl
                    .iter()
                    .flatten()
                    .any(|patch| !patch.collections.is_empty())
                || ds_vl
                    .iter()
                    .flatten()
                    .any(|patch| !patch.collections.is_empty())
                || all_ds.iter().any(|dataset| {
                    dataset.storage.chunks().is_none()
                        && dataset
                            .vl_string_staging
                            .as_ref()
                            .is_some_and(|staging| !staging.collections.is_empty())
                });
            let mut cursor = SUPERBLOCK_SIZE as u64;
            if has_collections {
                let start = free_space_manager::align_up(cursor, page_size);
                if start > cursor {
                    collection_meta_sections.push(FreeSection {
                        addr: StoredAddress::new(cursor),
                        size: start - cursor,
                    });
                }
                cursor = start;
            }
            let mut place = |collections: &[Vec<u8>]| -> Result<Vec<u64>, FormatError> {
                collections.iter().try_fold(
                    Vec::with_capacity(collections.len()),
                    |mut addrs, collection| {
                        let addr = cursor;
                        let len = collection.len() as u64;
                        let end = cursor.checked_add(len).ok_or(FormatError::OffsetOverflow {
                            offset: cursor,
                            length: len,
                        })?;
                        let padding = (page_size.get() - end % page_size.get()) % page_size.get();
                        cursor = end
                            .checked_add(padding)
                            .ok_or(FormatError::OffsetOverflow {
                                offset: end,
                                length: padding,
                            })?;
                        paged_collection_addresses.push(addr);
                        if cursor > end {
                            let section = FreeSection {
                                addr: StoredAddress::new(end),
                                size: cursor - end,
                            };
                            if len >= page_size.get() {
                                collection_large_sections.push(section);
                            } else {
                                collection_raw_sections.push(section);
                            }
                        }
                        addrs.push(addr);
                        Ok(addrs)
                    },
                )
            };
            for patch in &vl_root {
                let addrs = place(&patch.collections)?;
                type_builders::patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, &addrs);
            }
            for (gi, patches) in grp_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place(&patch.collections)?;
                    type_builders::patch_vl_refs(
                        &mut groups[gi].attrs[patch.attr_index].raw_data,
                        &addrs,
                    );
                }
            }
            for (di, patches) in ds_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place(&patch.collections)?;
                    type_builders::patch_vl_refs(
                        &mut all_ds[di].attrs[patch.attr_index].raw_data,
                        &addrs,
                    );
                }
            }
            for (i, dataset) in all_ds.iter_mut().enumerate() {
                if dataset.storage.chunks().is_some() {
                    continue;
                }
                if let Some(staging) = &dataset.vl_string_staging {
                    let addrs = place(&staging.collections)?;
                    type_builders::patch_vl_refs_masked(
                        dataset.storage.raw_mut()?,
                        &staging.patch_offsets,
                        &addrs,
                    );
                    early_gcol.push(i);
                }
            }
            cursor - SUPERBLOCK_SIZE as u64
        } else {
            for i in 0..all_ds.len() {
                if !is_chunked[i] || all_ds[i].storage.chunks().is_some() {
                    continue;
                }
                let Some(staging) = all_ds[i].vl_string_staging.take() else {
                    continue;
                };
                let addrs = place_collections(
                    &staging.collections,
                    &mut allocation,
                    &mut aggregate_collection_addresses,
                    aggregated,
                )?;
                type_builders::patch_vl_refs_masked(
                    all_ds[i].storage.raw_mut()?,
                    &staging.patch_offsets,
                    &addrs,
                );
                all_ds[i].vl_string_staging = Some(staging);
                early_gcol.push(i);
            }
            allocation.end() - SUPERBLOCK_SIZE as u64
        };

        #[cfg(feature = "provenance")]
        for (i, dataset) in all_ds.iter_mut().enumerate() {
            if dataset.reference_targets.is_none()
                && (dataset.vl_string_staging.is_none() || early_gcol.contains(&i))
            {
                if let Some(hash) = dataset.pending_hash.take() {
                    hash.complete(&mut dataset.attrs, dataset.storage.raw())?;
                }
            }
        }

        let mut chunk_sets: Vec<Option<CompressedChunkSet>> = all_ds
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if is_chunked[i] && d.storage.chunks().is_none() && d.reference_targets.is_none() {
                    d.compress_chunks().map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<_, FormatError>>()?;

        /// Where each object's dense-attribute heap will be written and how many
        /// bytes it occupies, for the objects that have one.
        struct DenseSpans {
            root: Option<(StoredAddress, usize)>,
            groups: Vec<Option<(StoredAddress, usize)>>,
            datasets: Vec<Option<(StoredAddress, usize)>>,
        }

        /// The heaps themselves, in the same order as the spans reserved for them.
        struct DenseBlobs {
            root: Option<DenseAttrBlob>,
            groups: Vec<Option<DenseAttrBlob>>,
            datasets: Vec<Option<DenseAttrBlob>>,
        }

        impl DenseSpans {
            /// Build every reserved heap, at the address reserved for it.
            ///
            /// Call once the attributes are final: after the global-heap
            /// collections have addresses and the variable-length attributes'
            /// references have been patched with them.
            ///
            /// Each blob fills its reserved span exactly, and nothing here has to
            /// arrange that. A heap's length is a function of its attributes'
            /// serialized sizes, the fixed offset and length widths, and how many
            /// of those attributes are huge — no term of it is the address it is
            /// built at. Patching cannot move any of those terms either: it takes
            /// the attribute bytes as `&mut [u8]` and overwrites references in
            /// place, so the slice it returns is the length it was given.
            ///
            /// The span was reserved from [`DenseAttrPlan::blob_len`], which
            /// derives the length rather than measuring a build of it, and the
            /// cursor that reservation advanced is what every later object's
            /// address is computed from. So a disagreement between the two would
            /// not stop at this heap: it would move every address after it, in a
            /// file that still looks well formed. That is a refusal rather than a
            /// `debug_assert`, because a release build has to catch it too.
            fn build(
                &self,
                root_attrs: &[AttributeMessage],
                groups: &[GrpFlat],
                datasets: &[DsFlat],
            ) -> Result<DenseBlobs, FormatError> {
                #[cfg(feature = "provenance")]
                if datasets
                    .iter()
                    .any(|dataset| dataset.pending_hash.is_some())
                {
                    return Err(FormatError::Internal(
                        "a provenance digest is incomplete before attribute encoding".into(),
                    ));
                }
                fn one(
                    attrs: &[AttributeMessage],
                    span: Option<(StoredAddress, usize)>,
                ) -> Result<Option<DenseAttrBlob>, FormatError> {
                    let Some((address, reserved)) = span else {
                        return Ok(None);
                    };
                    let blob =
                        build_dense_attrs(attrs, DenseAttrCreationOrder::Untracked, address)?;
                    if blob.blob.len() != reserved {
                        return Err(FormatError::SerializationError(format!(
                            "a dense attribute heap built {} bytes into a span of {reserved} \
                             reserved for it; every address after it would be wrong",
                            blob.blob.len(),
                        )));
                    }
                    Ok(Some(blob))
                }
                Ok(DenseBlobs {
                    root: one(root_attrs, self.root)?,
                    groups: groups
                        .iter()
                        .zip(&self.groups)
                        .map(|(g, &span)| one(&g.attrs, span))
                        .collect::<Result<_, _>>()?,
                    datasets: datasets
                        .iter()
                        .zip(&self.datasets)
                        .map(|(d, &span)| one(&d.attrs, span))
                        .collect::<Result<_, _>>()?,
                })
            }
        }

        // Pass 1: compute OH sizes with dummy addresses. Each object needing
        // dense attributes also records its heap's byte length here, so pass 2
        // can reserve the span without yet building the bytes that go in it —
        // see `DenseSpans`.
        //
        // Each plan is used for its length and its Attribute Info message and is
        // then dropped; `DenseSpans::build` plans again from scratch. Do not
        // "optimize" that by carrying these plans forward: `patch_vl_refs` below
        // overwrites variable-length references inside `attrs` between the two
        // passes, so a plan made here holds pre-patch bytes. Only its *length*
        // survives the patch, which is why that is all this pass takes from it.
        // Built here rather than beside the dummy addresses above because the
        // passes between still mutate `all_ds` (VL staging, compression), and
        // these tables borrow it.
        let dummy_tables = LinkTables {
            all_ds: &all_ds,
            committed: &committed,
            groups: &groups,
            ds_addrs: &dummy_ds_addrs,
            committed_addrs: &dummy_ct_addrs,
            group_addrs: &dummy_grp_addrs,
        };
        let mut group_oh_sizes: Vec<usize> = Vec::with_capacity(groups.len());
        let mut group_dense_lens: Vec<Option<usize>> = Vec::with_capacity(groups.len());
        let mut group_link_lens = Vec::with_capacity(groups.len());
        for (gi, g) in groups.iter().enumerate() {
            let dummy_links =
                dummy_tables.links(&g.ds_indices, &g.committed_indices, &g.sub_group_indices);
            let link_plan = dense_links_plan(&dummy_links)?;
            let link_info = link_plan
                .as_ref()
                .map(|plan| plan.link_info(DUMMY_DENSE_BASE))
                .transpose()
                .map_err(dense_link_error)?;
            group_link_lens.push(
                link_plan
                    .as_ref()
                    .map(|plan| plan.serialized_len().to_usize())
                    .transpose()?,
            );
            let (oh, dense_len) = if group_dense[gi] {
                // The plan answers both questions this pass asks — how long the
                // heap will be, and what its Attribute Info message costs the
                // header — without emitting the heap, which pass 2 does once at
                // the address reserved here.
                let plan = dense_attrs_plan(&g.attrs, DenseAttrCreationOrder::Untracked)?;
                (
                    build_group_oh(
                        &dummy_links,
                        &g.attrs,
                        Some(&plan.attr_info_message(DUMMY_DENSE_BASE)),
                        link_info.as_ref(),
                    )?,
                    Some(plan.blob_len().to_usize()?),
                )
            } else {
                (
                    build_group_oh(&dummy_links, &g.attrs, None, link_info.as_ref())?,
                    None,
                )
            };
            group_oh_sizes.push(oh.len());
            group_dense_lens.push(dense_len);
        }

        let root_dummy_links = dummy_tables.links(
            &root_ds_indices,
            &root_committed_indices,
            &root_group_indices,
        );
        let root_link_plan = dense_links_plan(&root_dummy_links)?;
        let root_link_info = root_link_plan
            .as_ref()
            .map(|plan| plan.link_info(DUMMY_DENSE_BASE))
            .transpose()
            .map_err(dense_link_error)?;
        let root_link_len = root_link_plan
            .as_ref()
            .map(|plan| plan.serialized_len().to_usize())
            .transpose()?;
        let (root_oh_size, root_dense_len) = if root_dense {
            let plan = dense_attrs_plan(&root_attrs, DenseAttrCreationOrder::Untracked)?;
            (
                build_group_oh(
                    &root_dummy_links,
                    &root_attrs,
                    Some(&plan.attr_info_message(DUMMY_DENSE_BASE)),
                    root_link_info.as_ref(),
                )?
                .len(),
                Some(plan.blob_len().to_usize()?),
            )
        } else {
            (
                build_group_oh(
                    &root_dummy_links,
                    &root_attrs,
                    None,
                    root_link_info.as_ref(),
                )?
                .len(),
                None,
            )
        };

        drop(root_link_plan);

        // Pass 1: compute dataset object-header sizes from a dummy layout. No
        // data bytes are materialized here — the object-header size depends only
        // on the layout/pipeline messages, and a chunk index's byte size is a
        // function of chunk count/size, not of the (dummy) data address. For a
        // streamed (lazy) dataset this touches no chunk bytes at all.
        let mut actual_ds_oh_sizes: Vec<usize> = Vec::with_capacity(all_ds.len());
        // Data-region lengths for allocation and PAGE free-space planning. Reference-bearing
        // chunks have a zero placeholder until their target addresses are resolved and their
        // encoded sizes are known.
        let mut ds_data_lens: Vec<u64> = Vec::with_capacity(all_ds.len());
        let mut ds_dense_lens: Vec<Option<usize>> = Vec::with_capacity(all_ds.len());
        let mut dummy_cursor = 0u64;
        for (i, d) in all_ds.iter().enumerate() {
            let dense_plan = if ds_dense[i] {
                Some(dense_attrs_plan(
                    &d.attrs,
                    DenseAttrCreationOrder::Untracked,
                )?)
            } else {
                None
            };
            let dense_attr_info = dense_plan
                .as_ref()
                .map(|plan| plan.attr_info_message(DUMMY_DENSE_BASE));
            ds_dense_lens.push(
                dense_plan
                    .as_ref()
                    .map(|plan| plan.blob_len().to_usize())
                    .transpose()?,
            );
            let oh = if is_chunked[i] {
                let (layout_message, pipeline_message) = if d.reference_targets.is_some() {
                    let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                    let ctx = crate::filters::ChunkContext::from_datatype(&chunk_dims, &d.dt)?;
                    let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
                    let header = d.chunk_options.measure_header(
                        &d.ds.dimensions,
                        ctx,
                        d.maxshape.as_deref(),
                        FillPattern::new(d.fill_time, d.fill.as_deref(), elem),
                        d.storage.allocation(),
                    )?;
                    ds_data_lens.push(0);
                    (header.layout_message, header.pipeline_message)
                } else {
                    let measured = measure_chunked(
                        d,
                        StoredAddress::new(dummy_cursor),
                        chunk_sets[i].as_ref(),
                    )?;
                    dummy_cursor += measured.data_len;
                    ds_data_lens.push(measured.data_len);
                    (measured.layout_message, measured.pipeline_message)
                };
                build_chunked_dataset_oh(
                    &d.dt,
                    &d.dt_location,
                    &d.ds,
                    &layout_message,
                    pipeline_message.as_deref(),
                    &d.attrs,
                    dense_attr_info.as_deref(),
                    d.fill_time,
                    d.fill.as_deref(),
                )?
            } else {
                ds_data_lens.push(d.contiguous_len());
                build_dataset_oh(
                    &d.dt,
                    &d.dt_location,
                    &d.ds,
                    StoredAddress::new(0),
                    d.declared_contiguous_len,
                    &d.attrs,
                    dense_attr_info.as_deref(),
                    d.fill_time,
                    d.fill.as_deref(),
                    libver,
                )?
            };
            actual_ds_oh_sizes.push(oh.len());
        }

        // Pass 2: compute real addresses.
        // The allocator returns stored addresses relative to the userblock base.
        let ub = self.userblock_size.get().to_usize()?;
        // Under direct placement, reserve the root after the early collections and PAGE padding.
        // Under aggregation, reserve the root header through the metadata aggregate.
        let advance = |cursor: &mut u64, len: u64| -> Result<StoredAddress, FormatError> {
            let addr = *cursor;
            *cursor = cursor.checked_add(len).ok_or(FormatError::OffsetOverflow {
                offset: addr,
                length: len,
            })?;
            Ok(StoredAddress::new(addr))
        };
        if paged.is_some() {
            allocation = FileAllocation::new(SUPERBLOCK_SIZE as u64 + early_gcol_size, false, None);
        }
        let root_group_addr = allocation.reserve(root_oh_size as u64, AllocationClass::Metadata)?;

        // Reserve dense-attribute spans before building their bytes. The heap contains copies of
        // the attributes, so every global-heap reference must be resolved before serialization.
        let reserve = |allocation: &mut FileAllocation, len: Option<usize>| {
            len.map(|len| {
                allocation
                    .reserve(len as u64, AllocationClass::Metadata)
                    .map(|addr| (addr, len))
            })
            .transpose()
        };

        let root_dense_span = reserve(&mut allocation, root_dense_len)?;
        let root_link_span = reserve(&mut allocation, root_link_len)?;
        let mut group_link_spans = Vec::with_capacity(groups.len());

        let mut group_dense_spans: Vec<Option<(StoredAddress, usize)>> =
            Vec::with_capacity(groups.len());
        let group_addrs2: Vec<StoredAddress> = group_oh_sizes.iter().enumerate().try_fold(
            Vec::with_capacity(group_oh_sizes.len()),
            |mut addrs, (gi, &sz)| {
                let addr = allocation.reserve(sz as u64, AllocationClass::Metadata)?;
                group_dense_spans.push(reserve(&mut allocation, group_dense_lens[gi])?);
                group_link_spans.push(reserve(&mut allocation, group_link_lens[gi])?);
                addrs.push(addr);
                Ok::<_, FormatError>(addrs)
            },
        )?;

        // Committed datatype headers. Placed with the other metadata so a paged
        // file keeps them in its metadata region, and before the datasets that
        // reference them only because the cursor has to run in some order —
        // nothing here depends on the relative placement.
        let committed_addrs: Vec<StoredAddress> = committed_oh.iter().try_fold(
            Vec::with_capacity(committed_oh.len()),
            |mut addrs, oh| {
                addrs.push(allocation.reserve(oh.len() as u64, AllocationClass::Metadata)?);
                Ok::<_, FormatError>(addrs)
            },
        )?;

        let mut ds_dense_spans: Vec<Option<(StoredAddress, usize)>> =
            Vec::with_capacity(all_ds.len());
        let ds_oh_addrs2: Vec<StoredAddress> = actual_ds_oh_sizes.iter().enumerate().try_fold(
            Vec::with_capacity(actual_ds_oh_sizes.len()),
            |mut addrs, (i, &sz)| {
                let addr = allocation.reserve(sz as u64, AllocationClass::Metadata)?;
                ds_dense_spans.push(reserve(&mut allocation, ds_dense_lens[i])?);
                addrs.push(addr);
                Ok::<_, FormatError>(addrs)
            },
        )?;

        let dense_spans = DenseSpans {
            root: root_dense_span,
            groups: group_dense_spans,
            datasets: ds_dense_spans,
        };

        // Resolve path-based references now that all addresses are known.
        // Build a map of (group_name, child_name) -> address for resolution.
        {
            // Build a path->address map for all datasets and groups.
            // Root-level datasets: path = dataset_name
            // Group-level datasets: path = group_name/dataset_name (recursive)
            // Groups: path = group_name (recursive)
            let mut path_map = HashMap::<ObjectPathBuf, StoredAddress>::new();
            let root = ObjectPathBuf::root();
            // An object reference can point at the root group, which the map holds
            // under its own path, the one with no components (repack resolves a
            // reference to the source root group to that path).
            path_map.insert(root.clone(), root_group_addr);
            for &i in &root_ds_indices {
                path_map.insert(root.join(&all_ds[i].name), ds_oh_addrs2[i]);
            }
            // A committed datatype is an object like any other, so an object
            // reference can point at one. Its path was already computed, above.
            for (path, &ci) in &committed_by_path {
                path_map.insert(path.clone(), committed_addrs[ci]);
            }
            for &gi in &root_group_indices {
                fn register_group(
                    prefix: &ObjectPathBuf,
                    gi: usize,
                    groups: &[GrpFlat],
                    ds_addrs: &[StoredAddress],
                    grp_addrs: &[StoredAddress],
                    all_ds: &[DsFlat],
                    map: &mut HashMap<ObjectPathBuf, StoredAddress>,
                ) {
                    map.insert(prefix.clone(), grp_addrs[gi]);
                    for &di in &groups[gi].ds_indices {
                        map.insert(prefix.join(&all_ds[di].name), ds_addrs[di]);
                    }
                    for &sgi in &groups[gi].sub_group_indices {
                        register_group(
                            &prefix.join(&groups[sgi].name),
                            sgi,
                            groups,
                            ds_addrs,
                            grp_addrs,
                            all_ds,
                            map,
                        );
                    }
                }
                register_group(
                    &root.join(&groups[gi].name),
                    gi,
                    &groups,
                    &ds_oh_addrs2,
                    &group_addrs2,
                    &all_ds,
                    &mut path_map,
                );
            }

            // Patch reference datasets: a path target resolves to its object's
            // destination address (an unknown path falls back to the undefined
            // address); a raw target is written verbatim (null / undefined).
            for (i, d) in all_ds.iter_mut().enumerate() {
                let Some(patches) = d.reference_targets.take() else {
                    continue;
                };
                for patch in patches {
                    let addr = match &patch.target {
                        crate::type_builders::ObjectRefTarget::Path(path) => {
                            path_map.get(path).copied().unwrap_or(UNDEF_ADDRESS)
                        }
                        crate::type_builders::ObjectRefTarget::Raw(addr) => *addr,
                    };
                    type_builders::write_reference_address(
                        d.storage.raw_mut()?,
                        patch.byte_offset,
                        addr,
                    );
                }
                #[cfg(feature = "provenance")]
                if let Some(hash) = d.pending_hash.take() {
                    hash.complete(&mut d.attrs, d.storage.raw())?;
                }
                if is_chunked[i] {
                    let set = d.compress_chunks()?;
                    ds_data_lens[i] =
                        chunked_write::measure_chunked_at(&set, StoredAddress::new(0))?.data_len;
                    chunk_sets[i] = Some(set);
                }
            }
        }

        // Resolve every committed-datatype reference, now that the objects have
        // addresses. From here on no staged path is left unresolved: the loops
        // below cover every dataset and every attribute in the file, which is
        // exactly the set `register_committed_use` walked to validate the paths.
        {
            let committed_location = |path: &ObjectPathBuf| {
                let ci = *committed_by_path.get(path).expect(
                    "every committed-datatype path was resolved against this same map \
                     before any header was sized",
                );
                DatatypeLocation::Committed(committed_addrs[ci])
            };
            let resolve_attrs = |attrs: &mut [AttributeMessage],
                                 paths: &[Option<ObjectPathBuf>]| {
                for (attr, path) in attrs.iter_mut().zip(paths) {
                    if let Some(path) = path {
                        attr.datatype_location = committed_location(path);
                    }
                }
            };
            resolve_attrs(&mut root_attrs, &root_attr_committed_datatype_paths);
            for g in &mut groups {
                resolve_attrs(&mut g.attrs, &g.attr_committed_datatype_paths);
            }
            for d in &mut all_ds {
                if let Some(path) = &d.committed_datatype_path {
                    d.dt_location = committed_location(path);
                }
                resolve_attrs(&mut d.attrs, &d.attr_committed_datatype_paths);
            }
        }

        struct DsLayout<'a> {
            data: DsData<'a>,
            data_addr: StoredAddress,
            chunked_msgs: Option<(Vec<u8>, Option<Vec<u8>>)>,
        }

        /// Build every dataset's object header from its final layout.
        ///
        /// The paged and non-paged paths reach this point with different
        /// addresses and the same datasets, so they share the build: a header
        /// field added on one path and missed on the other would be a file that
        /// changes shape with a property that is supposed to change only where
        /// its bytes land.
        fn build_ds_ohs(
            all_ds: &[DsFlat],
            ds_layouts: &[DsLayout<'_>],
            ds_dense_blobs: &[Option<DenseAttrBlob>],
            libver: LibVer,
        ) -> Result<Vec<Vec<u8>>, FormatError> {
            let mut oh_bytes: Vec<Vec<u8>> = Vec::with_capacity(all_ds.len());
            for (i, d) in all_ds.iter().enumerate() {
                let layout = &ds_layouts[i];
                let oh = if let Some((ref lm, ref pm)) = layout.chunked_msgs {
                    build_chunked_dataset_oh(
                        &d.dt,
                        &d.dt_location,
                        &d.ds,
                        lm,
                        pm.as_deref(),
                        &d.attrs,
                        ds_dense_blobs[i]
                            .as_ref()
                            .map(|b| b.attr_info_message.as_slice()),
                        d.fill_time,
                        d.fill.as_deref(),
                    )?
                } else {
                    build_dataset_oh(
                        &d.dt,
                        &d.dt_location,
                        &d.ds,
                        layout.data_addr,
                        d.declared_contiguous_len,
                        &d.attrs,
                        ds_dense_blobs[i]
                            .as_ref()
                            .map(|b| b.attr_info_message.as_slice()),
                        d.fill_time,
                        d.fill.as_deref(),
                        libver,
                    )?
                };
                oh_bytes.push(oh);
            }
            Ok(oh_bytes)
        }

        let tables = LinkTables {
            all_ds: &all_ds,
            committed: &committed,
            groups: &groups,
            ds_addrs: &ds_oh_addrs2,
            committed_addrs: &committed_addrs,
            group_addrs: &group_addrs2,
        };
        let root_link_image = root_link_span
            .map(|span| {
                dense_links_image(
                    &tables.links(
                        &root_ds_indices,
                        &root_committed_indices,
                        &root_group_indices,
                    ),
                    span,
                )
            })
            .transpose()?;
        let group_link_images = groups
            .iter()
            .zip(&group_link_spans)
            .map(|(group, &span)| {
                span.map(|span| {
                    dense_links_image(
                        &tables.links(
                            &group.ds_indices,
                            &group.committed_indices,
                            &group.sub_group_indices,
                        ),
                        span,
                    )
                })
                .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ---- Paged file-space layout + emission ----
        // Lay the metadata (object headers, dense blobs, global heaps, the
        // superblock extension, and the free-space-manager blocks) into a
        // page-0+ metadata region, then start the raw data on a fresh page
        // boundary. Small (< page) raw data packs into its own page run; each
        // large (>= page) block gets its own page-aligned run. Every region's
        // page tail is tracked in a per-page-type free-space manager (SUPER for
        // metadata, DRAW for small raw, generic-large for large fragments) when
        // persisting. Emission is address-driven: gaps are zero-filled so the
        // physical file reaches the page-aligned end-of-allocation.
        if let Some((page_size, persist_paged, fs_threshold)) = paged {
            let base = BaseAddress::new(ub as u64);
            let mut meta = allocation.end(); // metadata cursor, base-relative

            // The attributes are final now, so the dense heaps that copy them can
            // be built into the spans reserved above.
            let DenseBlobs {
                root: root_dense_blob,
                groups: group_dense_blobs,
                datasets: ds_dense_blobs,
            } = dense_spans.build(&root_attrs, &groups, &all_ds)?;

            // (b) Superblock extension (File Space Info) in the metadata region.
            let ext_addr = meta;
            let ext_len = ext_oh.as_ref().map_or(0, |b| b.len()) as u64;
            meta += ext_len;
            let meta_content_end = meta;

            // (c) Classify each dataset's data region by length. Contiguous empty
            // data is unallocated (undefined address); a chunked region always
            // has index bytes, so it is never "empty".
            let mut empty_indices: Vec<usize> = Vec::new();
            let mut small_indices: Vec<usize> = Vec::new();
            let mut large_indices: Vec<usize> = Vec::new();
            for i in 0..all_ds.len() {
                let len = ds_data_lens[i];
                if !is_chunked[i] && len == 0 {
                    empty_indices.push(i);
                } else if len < page_size.get() {
                    small_indices.push(i);
                } else {
                    large_indices.push(i);
                }
            }
            let small_raw_total: u64 = small_indices.iter().map(|&i| ds_data_lens[i]).sum();
            let mut large_frag_sizes: Vec<u64> = large_indices
                .iter()
                .map(|&i| {
                    free_space_manager::align_up(ds_data_lens[i], page_size) - ds_data_lens[i]
                })
                .filter(|&f| f > 0)
                .collect();

            large_frag_sizes.extend(collection_large_sections.iter().map(|section| section.size));

            // (d) Which per-page-type managers are active, and place their
            // FSHD/FSSE blocks in the metadata region (persisting only). Block
            // lengths depend only on section counts (fixed field widths), so the
            // page tails are computed in a single forward pass with no iteration.
            let draw_active = !collection_raw_sections.is_empty()
                || (small_raw_total > 0
                    && free_space_manager::align_up(small_raw_total, page_size) != small_raw_total);
            let large_active = !large_frag_sizes.is_empty();
            let mut slots = [u64::MAX; NUM_FILE_FSM_MANAGERS];
            let mut super_fsm: Option<(StoredAddress, StoredAddress, u64)> = None;
            let mut draw_fsm: Option<(StoredAddress, StoredAddress)> = None;
            let mut large_fsm: Option<(StoredAddress, StoredAddress)> = None;
            let mut draw_sizes: Vec<_> = collection_raw_sections
                .iter()
                .map(|section| section.size)
                .collect();
            draw_sizes.push(0);
            let draw_block_len = if draw_active {
                hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS)
                    + hdf5_pure_format::__private::section_info_len(WIDTHS, &draw_sizes)
            } else {
                0
            };
            let large_block_len = if large_active {
                hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS)
                    + hdf5_pure_format::__private::section_info_len(WIDTHS, &large_frag_sizes)
            } else {
                0
            };
            let super_allocation = if persist_paged {
                paged_metadata_section_allocation(
                    page_size,
                    meta_content_end,
                    [draw_block_len, large_block_len],
                    &collection_meta_sections,
                )?
            } else {
                None
            };
            if persist_paged {
                if let Some(allocated) = super_allocation {
                    let fshd_addr = StoredAddress::new(meta);
                    meta += hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS);
                    let fsse_addr = StoredAddress::new(meta);
                    meta += allocated;
                    slots[0] = fshd_addr.get();
                    super_fsm = Some((fshd_addr, fsse_addr, allocated));
                }
                if draw_active {
                    let fshd_addr = StoredAddress::new(meta);
                    meta += hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS);
                    let fsse_addr = StoredAddress::new(meta);
                    meta += hdf5_pure_format::__private::section_info_len(WIDTHS, &draw_sizes);
                    slots[2] = fshd_addr.get();
                    draw_fsm = Some((fshd_addr, fsse_addr));
                }
                if large_active {
                    let fshd_addr = StoredAddress::new(meta);
                    meta += hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS);
                    let fsse_addr = StoredAddress::new(meta);
                    meta +=
                        hdf5_pure_format::__private::section_info_len(WIDTHS, &large_frag_sizes);
                    slots[6] = fshd_addr.get();
                    large_fsm = Some((fshd_addr, fsse_addr));
                }
            }
            let meta_end = meta;

            // (e) The raw-data region starts on a fresh page boundary. The
            // metadata page tail is the SUPER section (when active).
            let raw_start = free_space_manager::align_up(meta_end, page_size);
            if raw_start > meta_end {
                collection_meta_sections.push(FreeSection {
                    addr: StoredAddress::new(meta_end),
                    size: raw_start - meta_end,
                });
            }

            // (f) Build the raw data. Small blocks pack; the region is padded to
            // a page boundary (DRAW tail). Each large block starts a fresh page
            // run and its sub-page remainder becomes a generic-large section.
            let mut layouts: Vec<Option<DsLayout<'_>>> = (0..all_ds.len()).map(|_| None).collect();
            for &i in &empty_indices {
                let raw = all_ds[i].storage.take_raw();
                layouts[i] = Some(DsLayout {
                    data: DsData::InMemory(raw),
                    data_addr: UNDEF_ADDRESS,
                    chunked_msgs: None,
                });
            }
            let mut c = raw_start;
            for &i in &small_indices {
                let base_addr = StoredAddress::new(c);
                let layout = if is_chunked[i] {
                    let built = build_chunked(&all_ds[i], base_addr, chunk_sets[i].as_ref())?;
                    // The small/large classification and the free-space-manager
                    // sizing used the sizing-pass length (`ds_data_lens[i]`); the
                    // real build must match it, or the reserved manager space and
                    // the emitted layout would diverge (chunk-index byte length is
                    // base-address independent, so this always holds).
                    debug_assert_eq!(built.data.len(), ds_data_lens[i]);
                    c += built.data.len();
                    DsLayout {
                        data: built.data,
                        data_addr: base_addr,
                        chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                    }
                } else {
                    let data = all_ds[i].take_contiguous();
                    c += data.len();
                    DsLayout {
                        data,
                        data_addr: base_addr,
                        chunked_msgs: None,
                    }
                };
                layouts[i] = Some(layout);
            }
            let small_raw_end = c;
            let draw_section = if small_raw_total > 0
                && free_space_manager::align_up(small_raw_total, page_size) != small_raw_total
            {
                let padded = free_space_manager::align_up(small_raw_end, page_size);
                c = padded;
                Some(FreeSection {
                    addr: StoredAddress::new(small_raw_end),
                    size: padded - small_raw_end,
                })
            } else {
                if small_raw_total > 0 {
                    c = free_space_manager::align_up(small_raw_end, page_size);
                }
                None
            };
            let mut large_sections = collection_large_sections;
            for &i in &large_indices {
                c = free_space_manager::align_up(c, page_size);
                let data_addr = StoredAddress::new(c);
                let built_len;
                let layout = if is_chunked[i] {
                    let built = build_chunked(&all_ds[i], data_addr, chunk_sets[i].as_ref())?;
                    // See the small-run note: the real build length must equal the
                    // sizing-pass length the large classification/fragment used.
                    debug_assert_eq!(built.data.len(), ds_data_lens[i]);
                    built_len = built.data.len();
                    DsLayout {
                        data: built.data,
                        data_addr,
                        chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                    }
                } else {
                    let data = all_ds[i].take_contiguous();
                    built_len = data.len();
                    DsLayout {
                        data,
                        data_addr,
                        chunked_msgs: None,
                    }
                };
                layouts[i] = Some(layout);
                let data_end = data_addr.get() + built_len;
                let frag = free_space_manager::align_up(data_end, page_size) - data_end;
                if frag > 0 {
                    large_sections.push(FreeSection {
                        addr: StoredAddress::new(data_end),
                        size: frag,
                    });
                }
                c = free_space_manager::align_up(data_end, page_size);
            }
            let eoa_rel = c; // already page-aligned
            let eof_addr2 = base.absolute(StoredAddress::new(eoa_rel))?;
            let eoa_pre_fsm = eoa_rel;

            let ds_layouts: Vec<DsLayout<'_>> = layouts
                .into_iter()
                .map(|layout| layout.expect("every dataset placed"))
                .collect();

            // (h) Build dataset OHs from the final data addresses.
            let ds_oh_bytes = build_ds_ohs(&all_ds, &ds_layouts, &ds_dense_blobs, libver)?;
            debug_assert_eq!(
                ds_oh_bytes.iter().map(|b| b.len()).collect::<Vec<_>>(),
                actual_ds_oh_sizes
            );

            // (i) Rebuild the real superblock-extension header. For persisting
            // files this replaces the placeholder (empty-manager) message with
            // the per-page-type manager addresses; its length is unchanged.
            let real_ext_oh = if persist_paged {
                let info = file_space_info::persistent_managers(
                    FileSpaceStrategy::Page,
                    fs_threshold,
                    page_size,
                    slots,
                    eoa_pre_fsm,
                );
                let mut oh = ObjectHeaderWriter::new();
                oh.add_message_with_flags(
                    MessageType::FILE_SPACE_INFO,
                    hdf5_pure_format::__private::serialize_file_space_info_v1(WIDTHS, &info)?,
                    MessageFlags::FORBID_SHARING | MessageFlags::MARK_IF_UNKNOWN,
                );
                oh.serialize()?
            } else {
                ext_oh
                    .clone()
                    .expect("a paged file always emits a File Space Info message")
            };
            debug_assert_eq!(real_ext_oh.len() as u64, ext_len);

            collection_raw_sections.extend(draw_section);

            // (j) Serialize the free-space-manager blocks.
            let super_blocks = super_fsm
                .map(|(fshd_addr, fsse_addr, allocated)| {
                    hdf5_pure_format::__private::serialize_free_space_manager_with_allocation(
                        WIDTHS,
                        &collection_meta_sections,
                        fshd_addr,
                        fsse_addr,
                        SECTION_CLASS_SMALL,
                        Some(allocated),
                    )
                })
                .transpose()?;
            let draw_blocks = draw_fsm
                .map(|(fshd_addr, fsse_addr)| {
                    hdf5_pure_format::__private::serialize_free_space_manager_with_allocation(
                        WIDTHS,
                        &collection_raw_sections,
                        fshd_addr,
                        fsse_addr,
                        SECTION_CLASS_SMALL,
                        Some(hdf5_pure_format::__private::section_info_len(
                            WIDTHS,
                            &draw_sizes,
                        )),
                    )
                })
                .transpose()?;
            let large_blocks = large_fsm
                .map(|(fshd_addr, fsse_addr)| {
                    hdf5_pure_format::__private::serialize_free_space_manager(
                        WIDTHS,
                        &large_sections,
                        fshd_addr,
                        fsse_addr,
                        SECTION_CLASS_LARGE,
                    )
                })
                .transpose()?;

            // (k) Emit, address-ascending, zero-filling every alignment gap.
            sink.reserve(eof_addr2)?;
            Self::put_userblock(sink, ub, &self.userblock_content)?;
            let sb = SuperblockFields {
                status: SuperblockStatus::new(
                    superblock_version(libver),
                    SuperblockStatusFlags::new(),
                )?,
                offset_size: OFFSET_WIDTH,
                length_size: LENGTH_WIDTH,
                base_address: base,
                eof_address: eof_addr2,
                root_group_address: root_group_addr,
                group_leaf_node_k: None,
                group_internal_node_k: None,
                indexed_storage_internal_node_k: None,
                free_space_address: None,
                driver_info_address: None,
                superblock_extension_address: Some(StoredAddress::new(ext_addr)),
                checksum: None,
            }
            .build();
            sink.put(&hdf5_pure_format::__private::serialize_superblock(&sb)?)
                .map_err(EmissionError::Sink)?;

            let collections = vl_root
                .iter()
                .map(|patch| &patch.collections)
                .chain(grp_vl.iter().flatten().map(|patch| &patch.collections))
                .chain(ds_vl.iter().flatten().map(|patch| &patch.collections))
                .chain(early_gcol.iter().filter_map(|&i| {
                    all_ds[i]
                        .vl_string_staging
                        .as_ref()
                        .map(|staging| &staging.collections)
                }))
                .flatten();
            for (address, collection) in paged_collection_addresses.iter().zip(collections) {
                let physical = base.absolute(StoredAddress::new(*address))?;
                sink.put_zeros((physical - sink.position()).to_usize()?)
                    .map_err(EmissionError::Sink)?;
                sink.put(collection).map_err(EmissionError::Sink)?;
            }
            let root_physical = base.absolute(root_group_addr)?;
            sink.put_zeros((root_physical - sink.position()).to_usize()?)
                .map_err(EmissionError::Sink)?;

            // Root group OH + dense blob.
            let tables = LinkTables {
                all_ds: &all_ds,
                committed: &committed,
                groups: &groups,
                ds_addrs: &ds_oh_addrs2,
                committed_addrs: &committed_addrs,
                group_addrs: &group_addrs2,
            };
            let root_links = tables.links(
                &root_ds_indices,
                &root_committed_indices,
                &root_group_indices,
            );
            let root_oh = build_group_oh(
                &root_links,
                &root_attrs,
                root_dense_blob
                    .as_ref()
                    .map(|b| b.attr_info_message.as_slice()),
                root_link_image.as_ref().map(|image| &image.info),
            )?;
            if root_oh.len() != root_oh_size {
                return Err(FormatError::Internal(
                    "the root group header differs from its reserved length".into(),
                )
                .into());
            }
            sink.put(&root_oh).map_err(EmissionError::Sink)?;
            if let Some(ref blob) = root_dense_blob {
                sink.put(&blob.blob).map_err(EmissionError::Sink)?;
            }
            if let Some(image) = &root_link_image {
                sink.put(&image.bytes).map_err(EmissionError::Sink)?;
            }
            // Group OHs + dense blobs.
            for (gi, g) in groups.iter().enumerate() {
                let links = tables.links(&g.ds_indices, &g.committed_indices, &g.sub_group_indices);
                let oh = build_group_oh(
                    &links,
                    &g.attrs,
                    group_dense_blobs[gi]
                        .as_ref()
                        .map(|b| b.attr_info_message.as_slice()),
                    group_link_images[gi].as_ref().map(|image| &image.info),
                )?;
                if oh.len() != group_oh_sizes[gi] {
                    return Err(FormatError::Internal(
                        "a group header differs from its reserved length".into(),
                    )
                    .into());
                }
                sink.put(&oh).map_err(EmissionError::Sink)?;
                if let Some(ref blob) = group_dense_blobs[gi] {
                    sink.put(&blob.blob).map_err(EmissionError::Sink)?;
                }
                if let Some(image) = &group_link_images[gi] {
                    sink.put(&image.bytes).map_err(EmissionError::Sink)?;
                }
            }
            // Committed datatype OHs, at the addresses their references name.
            for oh in &committed_oh {
                sink.put(oh).map_err(EmissionError::Sink)?;
            }
            // Dataset OHs + dense blobs.
            for (i, oh) in ds_oh_bytes.iter().enumerate() {
                sink.put(oh).map_err(EmissionError::Sink)?;
                if let Some(ref dense) = ds_dense_blobs[i] {
                    sink.put(&dense.blob).map_err(EmissionError::Sink)?;
                }
            }
            debug_assert_eq!(sink.position(), base.get() + ext_addr);
            sink.put(&real_ext_oh).map_err(EmissionError::Sink)?;
            // Free-space-manager blocks (SUPER, DRAW, generic-large), ascending.
            for blocks in [&super_blocks, &draw_blocks, &large_blocks]
                .into_iter()
                .flatten()
            {
                sink.put(&blocks.0).map_err(EmissionError::Sink)?;
                sink.put(&blocks.1).map_err(EmissionError::Sink)?;
            }
            debug_assert_eq!(sink.position(), base.get() + meta_end);

            // Raw data region: metadata page tail, then small data, DRAW tail,
            // large runs (with their fragments), padded to the page-aligned EOA.
            sink.put_zeros((raw_start - meta_end).to_usize()?)
                .map_err(EmissionError::Sink)?;
            for &i in &small_indices {
                let at = base.absolute(ds_layouts[i].data_addr)?;
                debug_assert_eq!(sink.position(), at);
                ds_layouts[i].data.emit(sink, &all_ds[i].storage)?;
            }
            if small_raw_total > 0 {
                sink.put_zeros(
                    (free_space_manager::align_up(small_raw_end, page_size) - small_raw_end)
                        .to_usize()?,
                )
                .map_err(EmissionError::Sink)?;
            }
            for &i in &large_indices {
                let gap = base.absolute(ds_layouts[i].data_addr)? - sink.position();
                sink.put_zeros(gap.to_usize()?)
                    .map_err(EmissionError::Sink)?;
                ds_layouts[i].data.emit(sink, &all_ds[i].storage)?;
                let end_rel = base.relative(sink.position())?.get();
                sink.put_zeros(
                    (free_space_manager::align_up(end_rel, page_size) - end_rel).to_usize()?,
                )
                .map_err(EmissionError::Sink)?;
            }
            let final_pad = eof_addr2 - sink.position();
            sink.put_zeros(final_pad.to_usize()?)
                .map_err(EmissionError::Sink)?;
            debug_assert_eq!(sink.position(), eof_addr2);
            return Ok(());
        }

        let mut ds_layouts: Vec<DsLayout<'_>> = Vec::new();
        for (i, d) in all_ds.iter_mut().enumerate() {
            if is_chunked[i] {
                let data_address = allocation.reserve(ds_data_lens[i], AllocationClass::Raw)?;
                let built = build_chunked(d, data_address, chunk_sets[i].as_ref())?;
                if built.data.len() != ds_data_lens[i] {
                    return Err(FormatError::Internal(
                        "chunk data differs from its allocation".into(),
                    )
                    .into());
                }
                ds_layouts.push(DsLayout {
                    data: built.data,
                    data_addr: data_address,
                    chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                });
            } else {
                // Move in-memory element bytes into the layout. Retain a produced region's
                // provider in prepared storage for emission.
                let data = d.take_contiguous();
                let addr = if data.len() == 0 {
                    UNDEF_ADDRESS
                } else {
                    allocation.reserve(data.len(), AllocationClass::Raw)?
                };
                ds_layouts.push(DsLayout {
                    data,
                    data_addr: addr,
                    chunked_msgs: None,
                });
            }
        }

        // Reserve the remaining global heaps as raw allocations and patch their references.
        // Reserve attribute collections in root, group, and dataset order, then dataset-element
        // collections. With aggregation enabled, retain every address for emission.
        let has_vl = !vl_root.is_empty()
            || grp_vl.iter().any(|v| !v.is_empty())
            || ds_vl.iter().any(|v| !v.is_empty())
            || all_ds.iter().any(|d| d.vl_string_staging.is_some());

        let gcol_start = allocation.end();
        if has_vl {
            for patch in &vl_root {
                let addrs = place_collections(
                    &patch.collections,
                    &mut allocation,
                    &mut aggregate_collection_addresses,
                    aggregated,
                )?;
                type_builders::patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, &addrs);
            }
            for (gi, patches) in grp_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place_collections(
                        &patch.collections,
                        &mut allocation,
                        &mut aggregate_collection_addresses,
                        aggregated,
                    )?;
                    type_builders::patch_vl_refs(
                        &mut groups[gi].attrs[patch.attr_index].raw_data,
                        &addrs,
                    );
                }
            }
            for (di, patches) in ds_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place_collections(
                        &patch.collections,
                        &mut allocation,
                        &mut aggregate_collection_addresses,
                        aggregated,
                    )?;
                    type_builders::patch_vl_refs(
                        &mut all_ds[di].attrs[patch.attr_index].raw_data,
                        &addrs,
                    );
                }
            }
            // Patch dataset-element references in the contiguous element bytes moved into
            // `ds_layouts[i].data`. The writer reserves chunked datasets' collections and patches
            // their references before chunk encoding.
            for (i, d) in all_ds.iter_mut().enumerate() {
                // A chunked VL dataset was placed and patched before its chunks
                // were encoded; its references are already final.
                if early_gcol.contains(&i) {
                    continue;
                }
                if let Some(staging) = &d.vl_string_staging {
                    // A dataset that carries VL staging always has its element
                    // bytes in memory. Report it: heap addresses patched into
                    // another buffer would corrupt the file in silence.
                    let Some(bytes) = ds_layouts[i].data.in_memory_mut() else {
                        return Err(vl_element_bytes_absent(d.name.as_str()).into());
                    };
                    let addrs = place_collections(
                        &staging.collections,
                        &mut allocation,
                        &mut aggregate_collection_addresses,
                        aggregated,
                    )?;
                    type_builders::patch_vl_refs_masked(bytes, &staging.patch_offsets, &addrs);
                    #[cfg(feature = "provenance")]
                    if let Some(hash) = d.pending_hash.take() {
                        hash.complete(&mut d.attrs, bytes)?;
                    }
                }
            }
        }

        // The attributes are final now, so the dense heaps that copy them can be
        // built into the spans reserved for them.
        let DenseBlobs {
            root: root_dense_blob,
            groups: group_dense_blobs,
            datasets: ds_dense_blobs,
        } = dense_spans.build(&root_attrs, &groups, &all_ds)?;

        // Build dataset OHs now that attrs are patched. Only the header bytes
        // are kept here; each dataset's data is emitted directly from
        // `ds_layouts` in the assembly loop (a streamed dataset has no data
        // bytes to keep at all).
        let ds_oh_bytes2 = build_ds_ohs(&all_ds, &ds_layouts, &ds_dense_blobs, libver)?;

        let actual_ds_oh_sizes2: Vec<usize> = ds_oh_bytes2.iter().map(|b| b.len()).collect();
        debug_assert_eq!(actual_ds_oh_sizes, actual_ds_oh_sizes2);

        // Reserve the superblock extension as metadata after reserving the global heaps.
        // Its stored address can lie in an earlier aggregate window.
        let ext_len = ext_oh.as_ref().map_or(0, |bytes| bytes.len());
        let ext_addr = ext_oh
            .as_ref()
            .map(|bytes| allocation.reserve(bytes.len() as u64, AllocationClass::Metadata))
            .transpose()?;
        let ext_addr = ext_addr.map(StoredAddress::get);

        let (cursor2, manager) = if nonpaged_persist.is_some() {
            allocation.finish_persisted(WIDTHS)?
        } else {
            (allocation.finish()?, None)
        };

        // eof_address is absolute file size (includes userblock + GCOLs + ext)
        let eof_addr2 = (ub as u64)
            .checked_add(cursor2)
            .ok_or(FormatError::OffsetOverflow {
                offset: ub as u64,
                length: cursor2,
            })?;

        // Buffered sinks reserve the complete output capacity. Streaming sinks ignore the hint.
        sink.reserve(eof_addr2)?;

        // Userblock: the caller's header bytes, then zeros.
        Self::put_userblock(sink, ub, &self.userblock_content)?;

        let sb = SuperblockFields {
            status: SuperblockStatus::new(
                superblock_version(libver),
                SuperblockStatusFlags::new(),
            )?,
            offset_size: OFFSET_WIDTH,
            length_size: LENGTH_WIDTH,
            base_address: BaseAddress::new(ub as u64),
            eof_address: eof_addr2,
            root_group_address: root_group_addr,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            superblock_extension_address: Some(
                ext_addr
                    .map(StoredAddress::new)
                    .unwrap_or(StoredAddress::undefined(OffsetWidth::Eight)),
            ),
            checksum: None,
        }
        .build();
        sink.put(&hdf5_pure_format::__private::serialize_superblock(&sb)?)
            .map_err(EmissionError::Sink)?;

        /// Identifies borrowed bytes or a deferred dataset or group serialization.
        enum FileData<'a> {
            Bytes(&'a [u8]),
            Dataset(usize),
            Group(usize),
        }

        // Use retained aggregate addresses or derive consecutive addresses in reservation order.
        let base = BaseAddress::new(ub as u64);
        // Each early index is recorded after the dataset's VL staging is restored.
        let early_count = early_gcol
            .iter()
            .map(|&i| {
                all_ds[i]
                    .vl_string_staging
                    .as_ref()
                    .expect("early_gcol only holds datasets with VL staging")
                    .collections
                    .len()
            })
            .sum::<usize>();
        let (early_addresses, late_addresses) =
            aggregate_collection_addresses.split_at(if aggregated { early_count } else { 0 });
        let mut early_addresses = early_addresses.iter();
        let mut late_addresses = late_addresses.iter();
        let mut early_collections = early_gcol
            .iter()
            .flat_map(|&i| {
                all_ds[i]
                    .vl_string_staging
                    .as_ref()
                    .expect("early_gcol only holds datasets with VL staging")
                    .collections
                    .iter()
            })
            .scan(SUPERBLOCK_SIZE as u64, |cursor, collection| {
                Some(
                    (if aggregated {
                        early_addresses.next().copied().ok_or_else(|| {
                            FormatError::Internal("global heap placement is absent".into())
                        })
                    } else {
                        advance(cursor, collection.len() as u64)
                    })
                    .and_then(|address| {
                        Reservation::new(
                            base,
                            address,
                            collection.len() as u64,
                            FileData::Bytes(collection),
                        )
                    }),
                )
            });
        let attribute_collections = vl_root
            .iter()
            .chain(grp_vl.iter().flatten())
            .chain(ds_vl.iter().flatten())
            .flat_map(|patch| &patch.collections);
        let late_collections = all_ds
            .iter()
            .enumerate()
            .filter(|(i, _)| !early_gcol.contains(i))
            .filter_map(|(_, d)| d.vl_string_staging.as_ref())
            .flat_map(|staging| &staging.collections);
        let mut late_collections =
            attribute_collections
                .chain(late_collections)
                .scan(gcol_start, |cursor, collection| {
                    Some(
                        (if aggregated {
                            late_addresses.next().copied().ok_or_else(|| {
                                FormatError::Internal("global heap placement is absent".into())
                            })
                        } else {
                            advance(cursor, collection.len() as u64)
                        })
                        .and_then(|address| {
                            Reservation::new(
                                base,
                                address,
                                collection.len() as u64,
                                FileData::Bytes(collection),
                            )
                        }),
                    )
                });

        let tables = LinkTables {
            all_ds: &all_ds,
            committed: &committed,
            groups: &groups,
            ds_addrs: &ds_oh_addrs2,
            committed_addrs: &committed_addrs,
            group_addrs: &group_addrs2,
        };
        let root_links = tables.links(
            &root_ds_indices,
            &root_committed_indices,
            &root_group_indices,
        );
        let root_oh = build_group_oh(
            &root_links,
            &root_attrs,
            root_dense_blob
                .as_ref()
                .map(|b| b.attr_info_message.as_slice()),
            root_link_image.as_ref().map(|image| &image.info),
        )?;
        if root_oh.len() != root_oh_size {
            return Err(FormatError::Internal(
                "the root group header differs from its reserved length".into(),
            )
            .into());
        }
        let mut root_header = core::iter::once(Reservation::new(
            base,
            root_group_addr,
            root_oh.len() as u64,
            FileData::Bytes(&root_oh),
        ));
        let mut group_headers = group_addrs2.iter().enumerate().map(|(gi, &address)| {
            Reservation::new(
                base,
                address,
                group_oh_sizes[gi] as u64,
                FileData::Group(gi),
            )
        });
        let mut committed_headers =
            committed_addrs
                .iter()
                .zip(&committed_oh)
                .map(|(&address, oh)| {
                    Reservation::new(base, address, oh.len() as u64, FileData::Bytes(oh))
                });
        let mut dataset_headers = ds_oh_addrs2
            .iter()
            .zip(&ds_oh_bytes2)
            .map(|(&address, oh)| {
                Reservation::new(base, address, oh.len() as u64, FileData::Bytes(oh))
            });
        let mut dense_blobs = dense_spans
            .root
            .iter()
            .chain(dense_spans.groups.iter().flatten())
            .chain(dense_spans.datasets.iter().flatten())
            .zip(
                root_dense_blob
                    .iter()
                    .chain(group_dense_blobs.iter().flatten())
                    .chain(ds_dense_blobs.iter().flatten()),
            )
            .map(|(span, blob)| {
                Reservation::new(
                    base,
                    span.0,
                    blob.blob.len() as u64,
                    FileData::Bytes(&blob.blob),
                )
            });
        let mut link_blobs = root_link_span
            .iter()
            .chain(group_link_spans.iter().flatten())
            .zip(
                root_link_image
                    .iter()
                    .chain(group_link_images.iter().flatten()),
            )
            .map(|(span, image)| {
                Reservation::new(
                    base,
                    span.0,
                    image.bytes.len().to_u64(),
                    FileData::Bytes(&image.bytes),
                )
            });
        let mut dataset_regions = ds_layouts
            .iter()
            .enumerate()
            .filter(|(_, layout)| layout.data.len() != 0)
            .map(|(i, layout)| {
                Reservation::new(
                    base,
                    layout.data_addr,
                    layout.data.len(),
                    FileData::Dataset(i),
                )
            });

        // Replace the placeholder with the manager address and pre-manager EOA, or the final
        // EOA if no manager is needed. The message has the same length in either case.
        let real_ext_oh = match (&ext_oh, nonpaged_persist) {
            (Some(_), Some(settings)) => {
                let info = match &manager {
                    Some(manager) => file_space_info::persistent_single_manager(
                        OFFSET_WIDTH,
                        settings.strategy(),
                        settings.threshold(),
                        settings.page_size(),
                        manager.header_address.get(),
                        manager.header_address.get(),
                    ),
                    None => file_space_info::persistent_empty(
                        OFFSET_WIDTH,
                        settings.strategy(),
                        settings.threshold(),
                        settings.page_size(),
                        cursor2,
                    ),
                };
                let mut oh = ObjectHeaderWriter::new();
                oh.add_message_with_flags(
                    MessageType::FILE_SPACE_INFO,
                    hdf5_pure_format::__private::serialize_file_space_info_v1(WIDTHS, &info)?,
                    MessageFlags::FORBID_SHARING | MessageFlags::MARK_IF_UNKNOWN,
                );
                Some(oh.serialize()?)
            }
            (other, _) => other.clone(),
        };
        debug_assert_eq!(
            real_ext_oh.as_ref().map_or(0, |b| b.len()),
            ext_len,
            "rebuilt extension header length must match the reserved length"
        );
        let mut extension = match (real_ext_oh.as_ref(), ext_addr) {
            (Some(bytes), Some(address)) => Some(Reservation::new(
                base,
                StoredAddress::new(address),
                bytes.len() as u64,
                FileData::Bytes(bytes),
            )),
            (None, None) => None,
            _ => {
                return Err(FormatError::Internal(
                    "a superblock extension has no output reservation".into(),
                )
                .into());
            }
        }
        .into_iter();
        // Under direct placement, supply each source in ascending physical order. With aggregation
        // enabled, a small reservation can precede a large one already placed outside its window,
        // so sort the reservations before emission.
        let mut manager_blocks = manager
            .iter()
            .flat_map(|manager| {
                [
                    (manager.header_address, manager.header.as_slice()),
                    (
                        manager.section_list_address,
                        manager.section_list.as_slice(),
                    ),
                ]
            })
            .map(|(address, bytes)| {
                Reservation::new(base, address, bytes.len() as u64, FileData::Bytes(bytes))
            });
        let sources: [&mut dyn Iterator<Item = Result<Reservation<FileData<'_>>, FormatError>>;
            11] = [
            &mut early_collections,
            &mut root_header,
            &mut group_headers,
            &mut committed_headers,
            &mut dataset_headers,
            &mut dense_blobs,
            &mut link_blobs,
            &mut dataset_regions,
            &mut late_collections,
            &mut extension,
            &mut manager_blocks,
        ];
        let emit = |sink: &mut S, emission| match emission {
            FileData::Bytes(bytes) => sink.put(bytes).map_err(EmissionError::Sink),
            FileData::Group(gi) => {
                let g = &groups[gi];
                let links = tables.links(&g.ds_indices, &g.committed_indices, &g.sub_group_indices);
                let oh = build_group_oh(
                    &links,
                    &g.attrs,
                    group_dense_blobs[gi]
                        .as_ref()
                        .map(|b| b.attr_info_message.as_slice()),
                    group_link_images[gi].as_ref().map(|image| &image.info),
                )?;
                if oh.len() != group_oh_sizes[gi] {
                    return Err(FormatError::Internal(
                        "a group header differs from its reserved length".into(),
                    )
                    .into());
                }
                sink.put(&oh).map_err(EmissionError::Sink)
            }
            FileData::Dataset(i) => ds_layouts[i].data.emit(sink, &all_ds[i].storage),
        };
        if aggregated {
            let count = aggregate_collection_addresses.len()
                + 1
                + group_addrs2.len()
                + committed_addrs.len()
                + ds_oh_addrs2.len()
                + usize::from(dense_spans.root.is_some())
                + dense_spans.groups.iter().flatten().count()
                + dense_spans.datasets.iter().flatten().count()
                + usize::from(root_link_span.is_some())
                + group_link_spans.iter().flatten().count()
                + ds_layouts
                    .iter()
                    .filter(|layout| layout.data.len() != 0)
                    .count()
                + usize::from(real_ext_oh.is_some())
                + 2 * usize::from(manager.is_some());
            let mut reservations = Vec::with_capacity(count);
            for reservation in sources.into_iter().flatten() {
                reservations.push(reservation?);
            }
            reservations.sort_unstable_by_key(Reservation::start);
            Emission::new([&mut reservations.into_iter().map(Ok)]).emit(sink, eof_addr2, emit)?;
        } else {
            Emission::new(sources).emit(sink, eof_addr2, emit)?;
        }

        Ok(())
    }
}

/// A dataset's single storage source after parsing its builder request.
///
/// An unallocated schema and a payload source are mutually exclusive. An empty dataset
/// without an explicit storage request is represented by an empty in-memory buffer.
enum PreparedStorage {
    /// Encoded element bytes held in memory.
    Bytes(Vec<u8>),
    /// Encoded chunks read verbatim from a provider during emission.
    Chunks(RawChunkPayload),
    /// Contiguous element bytes read from a provider during emission.
    Produced(ProducedPayload),
    /// A schema without element storage.
    Unallocated,
}

impl PreparedStorage {
    /// Parses a single storage source or an unallocated schema.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ConflictingDatasetStorage`] if more than one source is requested
    /// or unallocated storage is paired with a payload. Returns
    /// [`FormatError::DatasetMissingData`] if a nonempty allocated dataset has no source.
    /// Propagates an error staged in `data`.
    fn parse(
        data: Option<Result<Vec<u8>, Box<FormatError>>>,
        chunks: Option<RawChunkPayload>,
        produced: Option<ProducedPayload>,
        allocation: StorageAllocation,
        empty: bool,
    ) -> Result<Self, FormatError> {
        let data = data.transpose().map_err(|error| *error)?;
        match (allocation, data, chunks, produced) {
            (StorageAllocation::Unallocated, None, None, None) => Ok(Self::Unallocated),
            (StorageAllocation::Allocated, Some(bytes), None, None) => Ok(Self::Bytes(bytes)),
            (StorageAllocation::Allocated, None, Some(chunks), None) => Ok(Self::Chunks(chunks)),
            (StorageAllocation::Allocated, None, None, Some(produced)) => {
                Ok(Self::Produced(produced))
            }
            (StorageAllocation::Allocated, None, None, None) if empty => {
                Ok(Self::Bytes(Vec::new()))
            }
            (StorageAllocation::Allocated, None, None, None) => {
                Err(FormatError::DatasetMissingData)
            }
            _ => Err(FormatError::ConflictingDatasetStorage),
        }
    }

    /// Returns the in-memory element bytes for reference patching.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the dataset's storage is not an in-memory buffer.
    fn raw_mut(&mut self) -> Result<&mut Vec<u8>, FormatError> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Chunks(_) | Self::Produced(_) | Self::Unallocated => Err(FormatError::Internal(
                "element patching requires an in-memory dataset".into(),
            )),
        }
    }

    /// Removes the in-memory buffer, or returns an empty buffer for another storage kind.
    fn take_raw(&mut self) -> Vec<u8> {
        match self {
            Self::Bytes(bytes) => core::mem::take(bytes),
            Self::Chunks(_) | Self::Produced(_) | Self::Unallocated => Vec::new(),
        }
    }

    /// Returns the in-memory element bytes, or an empty slice for another storage kind.
    fn raw(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Chunks(_) | Self::Produced(_) | Self::Unallocated => &[],
        }
    }

    fn chunks(&self) -> Option<&RawChunkPayload> {
        match self {
            Self::Chunks(chunks) => Some(chunks),
            Self::Bytes(_) | Self::Produced(_) | Self::Unallocated => None,
        }
    }

    fn produced(&self) -> Option<&ProducedPayload> {
        match self {
            Self::Produced(produced) => Some(produced),
            Self::Bytes(_) | Self::Chunks(_) | Self::Unallocated => None,
        }
    }

    fn allocation(&self) -> StorageAllocation {
        match self {
            Self::Unallocated => StorageAllocation::Unallocated,
            Self::Bytes(_) | Self::Chunks(_) | Self::Produced(_) => StorageAllocation::Allocated,
        }
    }
}

/// A nonempty attribute name without null bytes.
///
/// The name and its null terminator fit in the two-byte "Name Size" field of attribute message
/// versions 1, 2, and 3, defined in "The Attribute Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attribute
struct AttributeName<'a>(&'a str);

impl AttributeName<'_> {
    /// Encodes or copies an attribute through [`AttrSpec::to_message`].
    ///
    /// Uses this name for decoded values and preserves the name of a copied message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if a variable-length string
    /// exceeds [`u32::MAX`] bytes.
    fn to_message(&self, spec: &AttrSpec) -> Result<AttributeMessage, FormatError> {
        spec.to_message(self.0)
    }
}

impl<'a> TryFrom<&'a str> for AttributeName<'a> {
    type Error = FormatError;

    /// Parses an attribute name for a new file.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeName`] if `name` is empty or contains a null byte.
    /// Returns [`FormatError::AttributeFieldTooLong`] if `name` and its null terminator exceed
    /// [`u16::MAX`] bytes.
    fn try_from(name: &'a str) -> Result<Self, FormatError> {
        if name.is_empty() || name.as_bytes().contains(&0) {
            return Err(FormatError::InvalidAttributeName { name: name.into() });
        }
        if name.len() >= usize::from(u16::MAX) {
            return Err(FormatError::AttributeFieldTooLong {
                name: name.into(),
                field: "name",
                size: name.len().saturating_add(1),
                limit: usize::from(u16::MAX),
            });
        }
        Ok(Self(name))
    }
}

/// Returns the File Space Info message used to size the superblock extension.
///
/// Persistent manager addresses and the pre-manager end of allocation are undefined until
/// placement is complete.
fn file_space_placeholder(settings: FileSpaceSettings) -> FileSpaceInfo {
    if settings.persist() {
        file_space_info::persistent_placeholder(
            settings.strategy(),
            settings.threshold(),
            settings.page_size(),
        )
    } else {
        file_space_info::non_persistent(
            OFFSET_WIDTH,
            settings.strategy(),
            settings.threshold(),
            settings.page_size(),
        )
    }
}

/// Serializes the object header of the superblock extension, with `info` as its one message.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] or [`FormatError::AddressTooLarge`] if a value of
/// `info` does not fit its field at the writer's widths.
fn file_space_extension_oh(info: &FileSpaceInfo) -> Result<Vec<u8>, FormatError> {
    let mut oh = ObjectHeaderWriter::new();
    // The flags libhdf5 writes for this message (`H5Fsuper.c`,
    // HDF5 2.2.0). Neither must-understand bit is among them, so an older reader
    // still opens the file.
    oh.add_message_with_flags(
        MessageType::FILE_SPACE_INFO,
        hdf5_pure_format::__private::serialize_file_space_info_v1(WIDTHS, info)?,
        MessageFlags::FORBID_SHARING | MessageFlags::MARK_IF_UNKNOWN,
    );
    oh.serialize()
}

/// Reserves section-list capacity for the final metadata page tail.
///
/// `metadata_end` is a stored address before manager placement. Includes the lengths in
/// `other_managers` when testing page alignment. Returns `None` when the metadata ends on a
/// page boundary without this manager and `sections` is empty. When adding the manager would
/// fill a page exactly with no other metadata sections, reserves one extra padding byte so
/// the manager has a page tail to record.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if a manager extent or its padding exceeds [`u64::MAX`].
fn paged_metadata_section_allocation(
    page: FileSpacePageSize,
    metadata_end: u64,
    other_managers: impl IntoIterator<Item = u64>,
    sections: &[FreeSection],
) -> Result<Option<u64>, FormatError> {
    let end = other_managers
        .into_iter()
        .try_fold(metadata_end, |offset, length| {
            offset
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow { offset, length })
        })?;
    if sections.is_empty() && end % page.get() == 0 {
        return Ok(None);
    }
    let sizes: Vec<_> = sections
        .iter()
        .map(|section| section.size)
        .chain(core::iter::once(0))
        .collect();
    let allocation = hdf5_pure_format::__private::section_info_len(WIDTHS, &sizes);
    let end = [
        hdf5_pure_format::__private::free_space_manager_header_len(WIDTHS),
        allocation,
    ]
    .into_iter()
    .try_fold(end, |offset, length| {
        offset
            .checked_add(length)
            .ok_or(FormatError::OffsetOverflow { offset, length })
    })?;
    Ok(Some(if sections.is_empty() && end % page.get() == 0 {
        allocation
            .checked_add(1)
            .ok_or(FormatError::OffsetOverflow {
                offset: allocation,
                length: 1,
            })?
    } else {
        allocation
    }))
}

/// The value reserved for an unlimited maximum dimension.
///
/// `H5Screate_simple` rejects this value in current dimensions (`H5S.c`, HDF5 1.14.6).
pub(crate) const UNLIMITED_DIMENSION: u64 = u64::MAX;

/// The largest dataset rank accepted by the writer.
///
/// `H5Screate_simple` uses `H5S_MAX_RANK` for this bound (`H5S.c`, HDF5 1.14.6).
pub(crate) const MAX_DATASET_RANK: u8 = 32;

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::BTreeV2Header;
    use rstest::rstest;

    use super::*;
    use crate::File;
    use crate::access_mode::AccessMode;
    use crate::group_v2;
    use crate::link_info::LinkInfoMessage;
    use crate::object_header::ObjectHeader;
    use crate::object_path::ObjectPath;
    use crate::superblock::Superblock;
    use crate::type_builders;

    #[rstest]
    #[case::empty("", None)]
    #[case::nul("prefix\0suffix", None)]
    #[case::too_long(&"x".repeat(usize::from(u16::MAX)), Some(usize::from(u16::MAX) + 1))]
    fn rejects_attribute_names_before_sink_output(
        #[case] name: &str,
        #[case] encoded_size: Option<usize>,
        #[values(
            AttributeForm::Committed,
            AttributeForm::CommittedVarLen,
            AttributeForm::CommittedVarLenVerbatim,
            AttributeForm::CommittedVerbatim,
            AttributeForm::Value,
            AttributeForm::VarLenValue,
            AttributeForm::VarLenVerbatim,
            AttributeForm::Verbatim
        )]
        form: AttributeForm,
        #[values(
            AttributeTarget::Dataset,
            AttributeTarget::Group,
            AttributeTarget::Root
        )]
        target: AttributeTarget,
        #[values(false, true)] dense: bool,
    ) {
        let mut writer = attribute_writer(false);
        let mut group = writer.create_group("group");
        {
            let mut owner = match target {
                AttributeTarget::Dataset => {
                    AttributeOwner::Dataset(writer.create_dataset("data").with_i32_data(&[1]))
                }
                AttributeTarget::Group => AttributeOwner::Group(&mut group),
                AttributeTarget::Root => AttributeOwner::Root(&mut writer),
            };
            owner.insert("preserved", AttributeForm::Value, 1);
            if dense {
                for index in 0..10 {
                    owner.insert(&format!("filler{index}"), AttributeForm::Value, index);
                }
            }
            owner.insert(name, form, 42);
        }
        writer.add_group(group.finish());
        let mut bytes = b"previous bytes".to_vec();
        let error = writer.finish_to_sink(&mut bytes).unwrap_err().into_format();
        let expected = match encoded_size {
            Some(size) => FormatError::AttributeFieldTooLong {
                name: name.into(),
                field: "name",
                size,
                limit: usize::from(u16::MAX),
            },
            None => FormatError::InvalidAttributeName { name: name.into() },
        };
        assert_eq!(error, expected);
        assert_eq!(bytes, b"previous bytes");
    }

    #[rstest]
    #[case("scale")]
    #[case("échelle")]
    fn attribute_names_retain_encoding_across_staging_forms(
        #[case] name: &str,
        #[values(
            AttributeForm::Committed,
            AttributeForm::CommittedVarLen,
            AttributeForm::CommittedVarLenVerbatim,
            AttributeForm::CommittedVerbatim,
            AttributeForm::Value,
            AttributeForm::VarLenValue,
            AttributeForm::VarLenVerbatim,
            AttributeForm::Verbatim
        )]
        form: AttributeForm,
        #[values(false, true)] dense: bool,
        #[values(false, true)] paged: bool,
    ) {
        let mut writer = attribute_writer(paged);
        stage_encoded_name_attributes(AttributeOwner::Root(&mut writer), name, form, dense);
        stage_encoded_name_attributes(
            AttributeOwner::Dataset(writer.create_dataset("data").with_i32_data(&[1])),
            name,
            form,
            dense,
        );
        let mut group = writer.create_group("group");
        stage_encoded_name_attributes(AttributeOwner::Group(&mut group), name, form, dense);
        writer.add_group(group.finish());
        let bytes = writer.finish().unwrap();
        assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
        let file = File::from_bytes(bytes.clone()).unwrap();
        for messages in [
            file.root().attr_messages().unwrap(),
            file.group("group").unwrap().attr_messages().unwrap(),
            file.dataset("data").unwrap().attr_messages().unwrap(),
        ] {
            assert_eq!(messages.len(), if dense { 11 } else { 1 });
            let message = messages
                .iter()
                .find(|message| message.name == name)
                .unwrap();
            let encoded = if dense {
                message.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap()
            } else {
                message.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap()
            };
            assert_eq!(
                encoded[0],
                if dense || !name.is_ascii() {
                    ATTRIBUTE_MESSAGE_VERSION_THREE
                } else {
                    ATTRIBUTE_MESSAGE_VERSION_TWO
                }
            );
            if encoded[0] == ATTRIBUTE_MESSAGE_VERSION_THREE {
                assert_eq!(
                    encoded[ATTRIBUTE_NAME_CHARACTER_SET_OFFSET],
                    if name.is_ascii() {
                        ASCII_NAME_CHARACTER_SET
                    } else {
                        UTF8_NAME_CHARACTER_SET
                    }
                );
            }
            assert!(bytes.windows(encoded.len()).any(|window| window == encoded));
        }
        let expected = match form {
            AttributeForm::CommittedVarLen
            | AttributeForm::CommittedVarLenVerbatim
            | AttributeForm::VarLenValue
            | AttributeForm::VarLenVerbatim => AttrValue::VarLenString("value-42".into()),
            _ => AttrValue::I32(42),
        };
        for attributes in [
            file.root().attrs().unwrap(),
            file.group("group").unwrap().attrs().unwrap(),
            file.dataset("data").unwrap().attrs().unwrap(),
        ] {
            assert_eq!(attributes.get(name), Some(&expected));
            if dense {
                assert_eq!(attributes.get("filler0"), Some(&AttrValue::I32(0)));
                assert_eq!(attributes.get("filler9"), Some(&AttrValue::I32(9)));
            }
        }
    }

    fn stage_encoded_name_attributes(
        mut owner: AttributeOwner<'_>,
        name: &str,
        form: AttributeForm,
        dense: bool,
    ) {
        owner.insert(name, AttributeForm::Value, 7);
        if dense {
            for index in 0..10 {
                owner.insert(&format!("filler{index}"), AttributeForm::Value, index);
            }
        }
        owner.insert(name, form, 42);
    }

    #[rstest]
    #[case("xx", 0, false)]
    #[case("é", 1, false)]
    #[case("é", 0, true)]
    fn attribute_name_encoding_counts_toward_compact_storage_limit(
        #[case] name: &str,
        #[case] trim: usize,
        #[case] dense: bool,
        #[values(false, true)] paged: bool,
    ) {
        let probe =
            type_builders::build_attr_message("xx", &AttrValue::AsciiString("y".into())).unwrap();
        let overhead = probe.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len() - 1;
        let value = AttrValue::AsciiString("y".repeat(OBJECT_HEADER_MESSAGE_MAX - overhead - trim));
        let message = type_builders::build_attr_message(name, &value).unwrap();
        assert_eq!(
            message.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len(),
            OBJECT_HEADER_MESSAGE_MAX + usize::from(!name.is_ascii()) - trim
        );
        assert_eq!(needs_dense_attrs(&[message]).unwrap(), dense);
        let mut writer = attribute_writer(paged);
        writer.set_root_attr(name, value.clone());
        let bytes = writer.finish().unwrap();
        assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
        let file = File::from_bytes(bytes).unwrap();
        assert_eq!(file.root().attrs().unwrap().get(name), Some(&value));
    }

    #[derive(Clone, Copy)]
    enum AttributeTarget {
        Dataset,
        Group,
        Root,
    }

    #[test]
    fn repeated_root_attribute_has_one_serialized_message() {
        let mut writer = FileWriter::new();
        writer.set_root_attr("answer", AttrValue::I32(7));
        writer.set_root_attr("answer", AttrValue::I32(42));
        let file = File::from_bytes(writer.finish().unwrap()).unwrap();
        let messages = file.root().attr_messages().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].name, "answer");
        assert_eq!(messages[0].raw_data, 42_i32.to_le_bytes());
    }

    #[rstest]
    fn repeated_attributes_replace_complete_specs(
        #[values(
            AttributeForm::Committed,
            AttributeForm::CommittedVarLen,
            AttributeForm::CommittedVarLenVerbatim,
            AttributeForm::CommittedVerbatim,
            AttributeForm::Value,
            AttributeForm::VarLenValue,
            AttributeForm::VarLenVerbatim,
            AttributeForm::Verbatim
        )]
        before: AttributeForm,
        #[values(
            AttributeForm::Committed,
            AttributeForm::CommittedVarLen,
            AttributeForm::CommittedVarLenVerbatim,
            AttributeForm::CommittedVerbatim,
            AttributeForm::Value,
            AttributeForm::VarLenValue,
            AttributeForm::VarLenVerbatim,
            AttributeForm::Verbatim
        )]
        after: AttributeForm,
        #[values(false, true)] dense: bool,
        #[values(false, true)] paged: bool,
    ) {
        let bytes = replacement_writer(Some(before), after, dense, paged)
            .finish()
            .unwrap();
        assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
        assert_eq!(
            bytes,
            replacement_writer(None, after, dense, paged)
                .finish()
                .unwrap()
        );
        let file = File::from_bytes(bytes).unwrap();
        let expected = if dense { 13 } else { 3 };
        assert_eq!(
            [
                file.root().attr_messages().unwrap().len(),
                file.group("group").unwrap().attr_messages().unwrap().len(),
                file.dataset("data").unwrap().attr_messages().unwrap().len(),
            ],
            [expected; 3]
        );
    }

    fn replacement_writer(
        before: Option<AttributeForm>,
        after: AttributeForm,
        dense: bool,
        paged: bool,
    ) -> FileWriter {
        let mut writer = attribute_writer(paged);
        replacement_attrs(AttributeOwner::Root(&mut writer), before, after, dense);
        replacement_attrs(
            AttributeOwner::Dataset(writer.create_dataset("data").with_i32_data(&[1])),
            before,
            after,
            dense,
        );
        let mut group = writer.create_group("group");
        replacement_attrs(AttributeOwner::Group(&mut group), before, after, dense);
        writer.add_group(group.finish());
        writer
    }

    fn attribute_writer(paged: bool) -> FileWriter {
        let mut writer = FileWriter::new();
        if paged {
            writer.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
        }
        writer.commit_datatype("integer_type", type_builders::make_i32_type());
        writer.commit_datatype(
            "string_type",
            type_builders::build_attr_message("type", &AttrValue::VarLenString("type".into()))
                .unwrap()
                .datatype,
        );
        writer
    }

    fn replacement_attrs(
        mut owner: AttributeOwner<'_>,
        before: Option<AttributeForm>,
        after: AttributeForm,
        dense: bool,
    ) {
        owner.insert("first", AttributeForm::Value, 1);
        owner.insert(
            "replace",
            before.unwrap_or(after),
            if before.is_some() { 7 } else { 42 },
        );
        owner.insert("last", AttributeForm::Value, 3);
        if dense {
            for index in 0..10 {
                owner.insert(&format!("filler{index}"), AttributeForm::Value, index);
            }
        }
        if before.is_some() {
            owner.insert("replace", after, 42);
        }
    }

    enum AttributeOwner<'a> {
        Dataset(&'a mut DatasetBuilder),
        Group(&'a mut GroupBuilder),
        Root(&'a mut FileWriter),
    }

    impl AttributeOwner<'_> {
        fn insert(&mut self, name: &str, form: AttributeForm, value: i32) {
            let value = match form {
                AttributeForm::CommittedVarLen
                | AttributeForm::CommittedVarLenVerbatim
                | AttributeForm::VarLenValue
                | AttributeForm::VarLenVerbatim => {
                    AttrValue::VarLenString(format!("value-{value}"))
                }
                _ => AttrValue::I32(value),
            };
            match form {
                AttributeForm::Value | AttributeForm::VarLenValue => match self {
                    Self::Dataset(owner) => {
                        owner.set_attr(name, value);
                    }
                    Self::Group(owner) => owner.set_attr(name, value),
                    Self::Root(owner) => owner.set_root_attr(name, value),
                },
                AttributeForm::Committed | AttributeForm::CommittedVarLen => {
                    let path = if matches!(form, AttributeForm::CommittedVarLen) {
                        "string_type"
                    } else {
                        "integer_type"
                    };
                    match self {
                        Self::Dataset(owner) => {
                            owner.set_attr_committed(name, value, path);
                        }
                        Self::Group(owner) => owner.set_attr_committed(name, value, path),
                        Self::Root(owner) => owner.set_root_attr_committed(name, value, path),
                    }
                }
                AttributeForm::CommittedVerbatim | AttributeForm::Verbatim => {
                    let message = type_builders::build_attr_message(name, &value).unwrap();
                    let path = matches!(form, AttributeForm::CommittedVerbatim)
                        .then(|| ObjectPathBuf::parse("integer_type"));
                    match self {
                        Self::Dataset(owner) => {
                            owner.set_attr_verbatim(message, path);
                        }
                        Self::Group(owner) => owner.set_attr_verbatim(message, path),
                        Self::Root(owner) => owner.set_root_attr_verbatim(message, path),
                    }
                }
                AttributeForm::CommittedVarLenVerbatim | AttributeForm::VarLenVerbatim => {
                    let message = type_builders::build_attr_message(name, &value).unwrap();
                    let strings = value.var_len_strings().unwrap().to_vec();
                    let path = matches!(form, AttributeForm::CommittedVarLenVerbatim)
                        .then(|| ObjectPathBuf::parse("string_type"));
                    match self {
                        Self::Dataset(owner) => {
                            owner
                                .set_attr_var_len_verbatim(message, strings, path)
                                .unwrap();
                        }
                        Self::Group(owner) => owner
                            .set_attr_var_len_verbatim(message, strings, path)
                            .unwrap(),
                        Self::Root(owner) => owner
                            .set_root_attr_var_len_verbatim(message, strings, path)
                            .unwrap(),
                    }
                }
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum AttributeForm {
        Committed,
        CommittedVarLen,
        CommittedVarLenVerbatim,
        CommittedVerbatim,
        Value,
        VarLenValue,
        VarLenVerbatim,
        Verbatim,
    }

    #[rstest::rstest]
    #[case::before_exact_fill(394, 0, false, Some(35))]
    #[case::exact_fill(395, 0, false, Some(36))]
    #[case::after_exact_fill(396, 0, false, Some(35))]
    #[case::no_tail(512, 0, false, None)]
    #[case::other_manager_fills_page(395, 117, false, None)]
    #[case::existing_section(377, 0, true, Some(53))]
    fn paged_metadata_capacity_keeps_known_sections(
        #[case] end: u64,
        #[case] other: u64,
        #[case] existing: bool,
        #[case] expected: Option<u64>,
    ) {
        let sections = if existing {
            vec![FreeSection {
                addr: StoredAddress::new(48),
                size: 17,
            }]
        } else {
            Vec::new()
        };
        assert_eq!(
            paged_metadata_section_allocation(
                FileSpacePageSize::try_from(512).unwrap(),
                end,
                [other],
                &sections
            )
            .unwrap(),
            expected
        );
    }

    #[rstest::rstest]
    #[case::other_manager(u64::MAX, 1, u64::MAX, 1)]
    #[case::header(u64::MAX - 10, 0, u64::MAX - 10, 82)]
    #[case::section_list(u64::MAX - 90, 0, u64::MAX - 8, 35)]
    fn paged_metadata_capacity_rejects_overflow(
        #[case] end: u64,
        #[case] other: u64,
        #[case] offset: u64,
        #[case] length: u64,
    ) {
        assert_eq!(
            paged_metadata_section_allocation(
                FileSpacePageSize::try_from(512).unwrap(),
                end,
                [other],
                &[]
            )
            .unwrap_err(),
            FormatError::OffsetOverflow { offset, length }
        );
    }

    #[test]
    fn small_paged_collections_can_place_metadata_above_four_gibibytes() {
        let mut writer = FileWriter::new();
        writer.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::MAX),
        );
        writer.set_root_attr("first", AttrValue::VarLenString("first-value".into()));
        writer.set_root_attr("second", AttrValue::VarLenString("second-value".into()));
        writer.set_root_attr("third", AttrValue::VarLenString("third-value".into()));
        let mut sink = SparseSink {
            position: 0,
            chunks: Vec::new(),
        };
        writer.finish_to_sink(&mut sink).unwrap();
        let superblock = crate::superblock::read(sink.chunks[0].1.as_slice())
            .unwrap()
            .superblock;
        assert_eq!(superblock.root_group_address.get(), 1u64 << 32);
        assert_eq!(superblock.eof_address, sink.position);
        let collections: Vec<u64> = sink
            .chunks
            .iter()
            .filter(|(_, bytes)| {
                bytes.get(..GLOBAL_HEAP_COLLECTION_SIGNATURE.len())
                    == Some(GLOBAL_HEAP_COLLECTION_SIGNATURE)
            })
            .map(|(position, _)| *position)
            .collect();
        assert_eq!(collections, [1u64 << 30, 2u64 << 30, 3u64 << 30]);
        let root = sink
            .chunks
            .iter()
            .find(|(position, _)| *position == 1u64 << 32)
            .unwrap();
        assert_eq!(
            root.1.get(..OBJECT_HEADER_V2_SIGNATURE.len()),
            Some(OBJECT_HEADER_V2_SIGNATURE)
        );
        assert!(
            sink.chunks
                .iter()
                .map(|(_, bytes)| bytes.len())
                .sum::<usize>()
                < 20_000
        );
    }

    #[rstest::rstest]
    #[case::large(1_024, 0)]
    #[case::whole(4_096, 4_096)]
    #[case::small(8_192, 8_192)]
    fn paged_global_heaps_use_pages_disjoint_from_object_headers(
        #[case] page_size: u64,
        #[case] userblock: u64,
    ) {
        let mut writer = FileWriter::new();
        writer.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::try_from(page_size).unwrap()),
        );
        writer.with_userblock(UserblockSize::try_from(userblock).unwrap());
        writer.set_root_attr("root_label", AttrValue::VarLenString("root-value".into()));
        writer
            .create_dataset("contiguous")
            .with_vlen_strings(&["first", "second"]);
        writer
            .create_dataset("chunked")
            .with_vlen_strings(&["third", "fourth"])
            .with_chunks(&[1])
            .with_deflate(DeflateLevel::Six);
        let bytes = writer.finish().unwrap();
        let heap_addresses: Vec<u64> = bytes
            .windows(GLOBAL_HEAP_COLLECTION_SIGNATURE.len())
            .enumerate()
            .filter(|(_, signature)| *signature == GLOBAL_HEAP_COLLECTION_SIGNATURE)
            .map(|(offset, _)| offset as u64 - userblock)
            .collect();
        assert_eq!(heap_addresses.len(), 3);
        let metadata_addresses: Vec<u64> = bytes
            .windows(OBJECT_HEADER_V2_SIGNATURE.len())
            .enumerate()
            .filter(|(_, signature)| *signature == OBJECT_HEADER_V2_SIGNATURE)
            .map(|(offset, _)| offset as u64 - userblock)
            .collect();
        assert_eq!(metadata_addresses.len(), 4);
        for address in heap_addresses {
            assert_eq!(address % page_size, 0);
            let index = (userblock + address + 8).to_usize().unwrap();
            let size = u64::from_le_bytes(bytes[index..index + 8].try_into().unwrap());
            let end = address + size;
            for metadata in &metadata_addresses {
                assert!(
                    *metadata / page_size < address / page_size
                        || *metadata / page_size > (end - 1) / page_size
                );
            }
        }
    }

    struct SparseSink {
        position: u64,
        chunks: Vec<(u64, Vec<u8>)>,
    }

    impl ByteSink for SparseSink {
        type Error = FormatError;

        fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
            self.chunks.push((self.position, bytes.to_vec()));
            self.position += bytes.len() as u64;
            Ok(())
        }

        fn put_zeros(&mut self, len: usize) -> Result<(), FormatError> {
            self.position += len as u64;
            Ok(())
        }

        fn position(&self) -> u64 {
            self.position
        }
    }

    /// A committed datatype object is a header holding the type and nothing else,
    /// which is what makes the C library call it a named datatype rather than a
    /// malformed dataset (`H5O__dtype_isa`: a datatype message, no dataspace, no
    /// layout).
    #[test]
    fn a_committed_datatype_header_holds_only_its_type() {
        let bytes = build_committed_datatype_oh(&type_builders::make_i32_type(), 1).unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();

        let types: Vec<MessageType> = hdr.messages.iter().map(|m| m.msg_type).collect();
        assert_eq!(
            types,
            vec![MessageType::DATATYPE],
            "a singly referenced committed type carries its datatype and nothing else"
        );
        assert_eq!(
            hdf5_pure_format::__private::parse_datatype(&hdr.messages[0].data)
                .unwrap()
                .0,
            type_builders::make_i32_type()
        );
        assert_eq!(
            hdr.messages[0].flags,
            MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING,
            "the type of a committed object never changes and must not be shared onward"
        );
    }

    /// Above one reference the count is stored, in the message the format defines
    /// for it. A header without one reads as singly referenced, so the count is
    /// the difference between unlinking the name and destroying the type.
    #[test]
    fn a_committed_datatype_header_records_a_count_above_one() {
        let bytes = build_committed_datatype_oh(&type_builders::make_i32_type(), 4).unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();

        let refcount = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::OBJECT_REFERENCE_COUNT)
            .expect("a count above one is stored");
        assert_eq!(
            refcount.data,
            vec![0, 4, 0, 0, 0],
            "version 0 followed by the 4-byte count"
        );
    }

    /// The count is the object's hard links plus every message naming it, which is
    /// what the C library maintains: one `H5O_link` per link, and one more each
    /// time a dataset or attribute is created against the type.
    #[test]
    fn the_reference_count_is_the_link_plus_every_user() {
        let mut w = FileWriter::new();
        w.commit_datatype("mytype", type_builders::make_i32_type());
        w.set_root_attr_committed("root_attr", AttrValue::I32(1), "mytype");
        let ds = w.create_dataset("typed");
        ds.with_i32_data(&[1, 2]);
        ds.with_committed_datatype("mytype");
        ds.set_attr_committed("shared_attr", AttrValue::I32(2), "mytype");
        let bytes = w.finish().unwrap();

        // 1 hard link + the dataset + two attributes.
        assert_eq!(committed_reference_count(&bytes, "mytype"), Some(4));
    }

    /// A committed type nothing references stores no count at all: its single
    /// hard link is what a header without the message already says.
    #[test]
    fn an_unreferenced_committed_type_stores_no_count() {
        let mut w = FileWriter::new();
        w.commit_datatype("mytype", type_builders::make_i32_type());
        let bytes = w.finish().unwrap();

        assert_eq!(committed_reference_count(&bytes, "mytype"), None);
    }

    /// The reference count recorded in the committed datatype object at `path`, or
    /// `None` when the header carries no Object Reference Count message.
    fn committed_reference_count(bytes: &[u8], path: &str) -> Option<u32> {
        let sb = crate::superblock::read(bytes).unwrap().superblock;
        let addr = group_v2::resolve_path_any(
            bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse(path),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::OBJECT_REFERENCE_COUNT)?;
        Some(u32::from_le_bytes(msg.data[1..5].try_into().unwrap()))
    }

    /// A dataset naming a committed type stores a *reference* in place of the
    /// encoding, under a record whose shared flag says so — and the reference
    /// names the object the link points at.
    ///
    /// Without the flag those same ten bytes decode as a zero-width time datatype,
    /// with no error anywhere, which is the defect issue #254 was filed for.
    #[test]
    fn a_dataset_naming_a_committed_type_stores_a_reference_to_it() {
        let mut w = FileWriter::new();
        w.commit_datatype("mytype", type_builders::make_i32_type());
        let ds = w.create_dataset("typed");
        ds.with_i32_data(&[1, 2]);
        ds.with_committed_datatype("mytype");
        let bytes = w.finish().unwrap();

        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        let type_addr = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("mytype"),
        )
        .unwrap();
        let ds_addr = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("typed"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            ds_addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATATYPE)
            .expect("a dataset always has a datatype message");

        assert!(
            msg.flags.is_shared(),
            "the record must say its body is a reference, or the reference decodes as a type"
        );
        assert_eq!(
            msg.data,
            crate::shared_message::encode_committed_ref(
                StoredAddress::new(type_addr),
                OFFSET_WIDTH
            ),
            "the reference must name the object the link resolves to"
        );
    }

    fn parse_file(bytes: &[u8]) -> (Superblock, ObjectHeader) {
        let sb = crate::superblock::read(bytes).unwrap().superblock;
        let oh = ObjectHeader::parse_with_widths(
            bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(sb.base_address)
                .absolute(sb.root_group_address)
                .unwrap()
                .to_usize()
                .unwrap(),
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        (sb, oh)
    }

    fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
        let sb = crate::superblock::read(bytes).unwrap().superblock;
        let addr = group_v2::resolve_path_any(
            bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse(path),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let dt_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATATYPE)
            .unwrap()
            .data;
        let ds_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATASPACE)
            .unwrap()
            .data;
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
            .unwrap()
            .data;
        let (dt, _) = hdf5_pure_format::__private::parse_datatype(dt_data).unwrap();
        let ds = Dataspace::parse_with_length_width(ds_data, sb.length_size).unwrap();
        let dl = crate::data_layout::DataLayout::parse_with_widths(
            dl_data,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let raw = crate::data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
        crate::data_read::read_as_f64(&raw, &dt).unwrap()
    }

    #[test]
    fn empty_file_root_group_only() {
        let fw = FileWriter::new();
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        assert_eq!(sb.version(), SuperblockVersion::Three);
        assert_eq!(oh.version, 2);
    }

    #[test]
    fn file_with_dataset_attrs() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data")
            .with_f64_data(&[1.0, 2.0])
            .set_attr("scale", AttrValue::F64(0.5));
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0]);
        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        let addr = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("data"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let attrs = crate::attribute::extract_attributes(&hdr, LENGTH_WIDTH).unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "scale");
    }

    #[test]
    fn file_with_group_and_dataset() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        gb.create_dataset("vals").with_f64_data(&[10.0, 20.0]);
        fw.add_group(gb.finish());
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "grp/vals"), vec![10.0, 20.0]);
    }

    // hdf5-pure has no group creation property list: every object header it
    // writes is fixed to one shape, equivalent to the C library's
    // `obj_track_times = false` (see issue #131) — never toggleable, so these
    // lock in the "no timestamps" half of that fixed shape for both the root
    // group and an ordinary sub-group.
    #[test]
    fn root_group_carries_no_timestamps() {
        let fw = FileWriter::new();
        let bytes = fw.finish().unwrap();
        let (_, oh) = parse_file(&bytes);
        assert_eq!(oh.flags & 0x20, 0, "times-stored flag must be clear");
        assert!(oh.modification_time.is_none());
        assert!(oh.access_time.is_none());
        assert!(oh.change_time.is_none());
        assert!(oh.birth_time.is_none());
    }

    #[test]
    fn sub_group_carries_no_timestamps() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        gb.create_dataset("vals").with_f64_data(&[1.0]);
        fw.add_group(gb.finish());
        let bytes = fw.finish().unwrap();
        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        let addr = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("grp"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        assert_eq!(hdr.flags & 0x20, 0, "times-stored flag must be clear");
        assert!(hdr.modification_time.is_none());
    }

    #[rstest]
    #[case::empty(0)]
    #[case::below_threshold(MAX_COMPACT_LINKS - 1)]
    #[case::at_threshold(MAX_COMPACT_LINKS)]
    #[case::above_threshold(MAX_COMPACT_LINKS + 1)]
    #[case::multiple_blocks(100)]
    #[case::multiple_levels(2_000)]
    fn group_link_representation_matches_child_count(
        #[case] count: usize,
        #[values(false, true)] nested: bool,
        #[values(
            None,
            Some(FileSpaceStrategy::None),
            Some(FileSpaceStrategy::Aggr),
            Some(FileSpaceStrategy::FsmAggr),
            Some(FileSpaceStrategy::Page)
        )]
        strategy: Option<FileSpaceStrategy>,
        #[values(0, 4_096)] userblock: u64,
    ) {
        let bytes = link_writer(count, nested, strategy, userblock)
            .finish()
            .unwrap();
        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        assert_eq!(sb.eof_address, bytes.len().to_u64());
        let path = if nested { "parent/group" } else { "/" };
        let address = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse(path),
        )
        .unwrap();
        let header = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            address.to_usize().unwrap(),
            WIDTHS,
        )
        .unwrap();
        let info = LinkInfoMessage::parse_with_offset_width(
            &header
                .messages
                .iter()
                .find(|message| message.msg_type == MessageType::LINK_INFO)
                .unwrap()
                .data,
            OFFSET_WIDTH,
        )
        .unwrap();
        let dense = count > MAX_COMPACT_LINKS;
        assert_eq!(info.fractal_heap_address.is_some(), dense);
        assert_eq!(info.btree_name_index_address.is_some(), dense);
        assert_eq!(info.max_creation_order, None);
        assert_eq!(info.btree_creation_order_address, None);
        assert_eq!(
            header
                .messages
                .iter()
                .filter(|message| message.msg_type == MessageType::LINK)
                .count(),
            if dense { 0 } else { count }
        );
        assert_eq!(
            header
                .messages
                .iter()
                .find(|message| message.msg_type == MessageType::GROUP_INFO)
                .unwrap()
                .data,
            DEFAULT_GROUP_INFO
        );
        if count >= 100 {
            let heap = FractalHeapHeader::parse(
                &bytes,
                (info.fractal_heap_address.unwrap().get() + userblock)
                    .to_usize()
                    .unwrap(),
                OFFSET_SIZE,
                LENGTH_SIZE,
            )
            .unwrap();
            assert!(heap.current_rows_in_root_indirect_block > 0);
            assert_eq!(heap.managed_objects_count, count.to_u64());
        }
        if count == 2_000 {
            let index = BTreeV2Header::parse_with_widths(
                &bytes,
                (info.btree_name_index_address.unwrap().get() + userblock)
                    .to_usize()
                    .unwrap(),
                WIDTHS,
            )
            .unwrap();
            assert!(index.depth >= 2, "{index:?}");
        }
        let file = File::from_bytes(bytes).unwrap();
        for index in 0..count {
            let child = format!("{path}/data_{index:04}_λ");
            assert_eq!(
                file.dataset(&child).unwrap().read_i32().unwrap(),
                [i32::try_from(index).unwrap(), 17]
            );
        }
    }

    #[rstest]
    #[case::root(false)]
    #[case::nested(true)]
    fn dense_link_addresses_exceed_four_gibibytes(#[case] nested: bool) {
        let mut writer = link_writer(9, nested, Some(FileSpaceStrategy::Page), 0);
        writer.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page).with_page_size(FileSpacePageSize::MAX),
        );
        for name in ["first", "second", "third"] {
            writer.set_root_attr(name, AttrValue::VarLenString(name.into()));
        }
        let mut sink = SparseSink {
            position: 0,
            chunks: Vec::new(),
        };
        writer.finish_to_sink(&mut sink).unwrap();
        let superblock = crate::superblock::read(sink.chunks[0].1.as_slice())
            .unwrap()
            .superblock;
        assert_eq!(superblock.root_group_address.get(), 1u64 << 32);
        assert_eq!(superblock.eof_address, sink.position);
        let (_, bytes) = sink
            .chunks
            .iter()
            .find(|(_, bytes)| bytes.starts_with(FRACTAL_HEAP_SIGNATURE))
            .unwrap();
        let heap = FractalHeapHeader::parse_bytes(bytes, OFFSET_WIDTH, LENGTH_WIDTH).unwrap();
        assert!(heap.root_block_address.get() > u64::from(u32::MAX));
        assert_eq!(heap.managed_objects_count, 9);
        assert!(
            sink.chunks
                .iter()
                .filter(|(position, _)| *position >= 1u64 << 32)
                .all(|(position, bytes)| position + bytes.len().to_u64() <= sink.position)
        );
    }

    fn link_writer(
        count: usize,
        nested: bool,
        strategy: Option<FileSpaceStrategy>,
        userblock: u64,
    ) -> FileWriter {
        let mut writer = FileWriter::new();
        if let Some(strategy) = strategy {
            writer.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(
                matches!(
                    strategy,
                    FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
                ),
            ));
        }
        if userblock != 0 {
            writer.with_userblock(UserblockSize::try_from(userblock).unwrap());
        }
        let mut group = writer.create_group("group");
        for index in 0..count {
            let link_name = format!("data_{index:04}_λ");
            let data = [i32::try_from(index).unwrap(), 17];
            if nested {
                group.create_dataset(&link_name).with_i32_data(&data);
            } else {
                writer.create_dataset(&link_name).with_i32_data(&data);
            }
        }
        if nested {
            let mut parent = writer.create_group("parent");
            parent.add_group(group.finish());
            writer.add_group(parent.finish());
        }
        writer
    }

    #[test]
    fn file_with_root_attr() {
        let mut fw = FileWriter::new();
        fw.set_root_attr("version", AttrValue::I64(42));
        let bytes = fw.finish().unwrap();
        let (_sb, oh) = parse_file(&bytes);
        let attrs = crate::attribute::extract_attributes(&oh, LENGTH_WIDTH).unwrap();
        assert_eq!(attrs[0].name, "version");
    }

    #[test]
    fn dense_attrs_self_roundtrip() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0, 2.0, 3.0]);
        for i in 0..20 {
            ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64 * 1.5));
        }
        let bytes = fw.finish().unwrap();
        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        let addr = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("data"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let attrs = crate::attribute::extract_attributes_full(
            &bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(BaseAddress::new(0)),
            &hdr,
            WIDTHS,
            None,
        )
        .unwrap();
        assert_eq!(attrs.len(), 20);
        for i in 0..20 {
            let attr = attrs
                .iter()
                .find(|a| a.name == format!("attr_{i:03}"))
                .unwrap();
            let v = crate::data_read::read_as_f64(&attr.raw_data, &attr.datatype).unwrap();
            assert!((v[0] - i as f64 * 1.5).abs() < 1e-10);
        }
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dense_attrs_root_group_self_roundtrip() {
        let mut fw = FileWriter::new();
        fw.create_dataset("dummy").with_f64_data(&[0.0]);
        for i in 0..15 {
            fw.set_root_attr(&format!("root_{i:02}"), AttrValue::F64(i as f64 * 2.0));
        }
        let bytes = fw.finish().unwrap();
        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        let oh = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(sb.base_address)
                .absolute(sb.root_group_address)
                .unwrap()
                .to_usize()
                .unwrap(),
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let attrs = crate::attribute::extract_attributes_full(
            &bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(BaseAddress::new(0)),
            &oh,
            WIDTHS,
            None,
        )
        .unwrap();
        assert_eq!(attrs.len(), 15);
    }

    #[test]
    fn inline_attrs_below_threshold() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0]);
        for i in 0..5 {
            ds.set_attr(&format!("a{i}"), AttrValue::F64(i as f64));
        }
        let bytes = fw.finish().unwrap();
        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        let addr = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("data"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        // The Attribute Info message is present but names no fractal heap: that
        // is what tells the reference library the attributes are the header's own
        // inline messages, and how many there are.
        let info = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::ATTRIBUTE_INFO)
            .map(|m| {
                crate::attribute_info::AttributeInfoMessage::parse_with_offset_width(
                    &m.data,
                    sb.offset_size,
                )
                .unwrap()
            })
            .expect("a compact attribute set still carries an Attribute Info message");
        assert_eq!(info.fractal_heap_address, None);
        assert_eq!(info.btree_name_index_address, None);
        let attrs = crate::attribute::extract_attributes(&hdr, LENGTH_WIDTH).unwrap();
        assert_eq!(attrs.len(), 5);
    }

    /// An `AsciiString` attribute named `name` whose serialized (v3) size is
    /// exactly `size` bytes, so a bound can be tested on the value it bounds.
    fn dense_attr_of_size(name: &str, size: usize) -> AttributeMessage {
        // Probe with one character and subtract it, rather than probing with an
        // empty string: a fixed-width string datatype is at least one byte wide,
        // so an empty value still carries a padding byte and would be measured
        // as overhead.
        let probe =
            type_builders::build_attr_message(name, &AttrValue::AsciiString("y".to_string()))
                .unwrap();
        let overhead = probe.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len() - 1;
        let attr = type_builders::build_attr_message(
            name,
            &AttrValue::AsciiString("y".repeat(size - overhead)),
        )
        .unwrap();
        assert_eq!(
            attr.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len(),
            size
        );
        attr
    }

    #[test]
    fn dense_attrs_check_bounds_each_attribute_not_the_total() {
        // A multi-megabyte set of individually small attributes is exactly what
        // the emitter handles: it sizes its root direct block to the content, and
        // both this crate and libhdf5 read such a heap back. The
        // old bound rejected these.
        let many: Vec<AttributeMessage> = (0..40)
            .map(|i| dense_attr_of_size(&format!("a{i}"), 60_000))
            .collect();
        let total: usize = many
            .iter()
            .map(|a| a.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len())
            .sum();
        assert!(total > 2_000_000, "expected a multi-megabyte set");
        assert_eq!(dense_attrs_check(&many), Ok(()));

        // One attribute at the managed-object limit is still fine.
        let at_limit = vec![dense_attr_of_size("edge", DENSE_ATTR_MAX_MANAGED_OBJECT)];
        assert_eq!(dense_attrs_check(&at_limit), Ok(()));

        // One byte past it needs fractal-heap huge storage, which the emitter now
        // writes — so it is accepted, and the attribute is no longer in the
        // direct block.
        let past = vec![dense_attr_of_size(
            "edge",
            DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
        )];
        assert_eq!(dense_attrs_check(&past), Ok(()));
        assert_eq!(
            huge_object_count(
                &build_dense_attrs(
                    &past,
                    DenseAttrCreationOrder::Untracked,
                    StoredAddress::new(0)
                )
                .unwrap()
                .blob
            ),
            1
        );
        assert_eq!(
            huge_object_count(
                &build_dense_attrs(
                    &at_limit,
                    DenseAttrCreationOrder::Untracked,
                    StoredAddress::new(0)
                )
                .unwrap()
                .blob
            ),
            0
        );
    }

    /// `DenseAttrPlan::blob_len` is the span a caller reserves for a heap before
    /// a byte of it exists, so it has to equal the length `build` goes on to
    /// emit — at whatever heap address the reservation lands on. A reservation
    /// that came out short would place the next object on top of the heap.
    ///
    /// The attribute counts are swept *contiguously* rather than at hand-picked
    /// boundaries, so the sweep crosses every direct-block fill and every new row
    /// of the doubling table in its range, and every split of the name index's
    /// leaf, without anyone having to work out where those fall. The named shapes
    /// after it reach what a contiguous sweep of small attributes cannot: blocks
    /// large enough to be reached through nested indirect blocks, and the
    /// managed/huge boundary from both sides and mixed.
    #[test]
    fn dense_attr_plan_length_matches_what_it_builds() {
        let mut shapes: Vec<(String, Vec<AttributeMessage>)> = Vec::new();

        for n in 0..=64usize {
            shapes.push((
                format!("{n} x 512B"),
                (0..n)
                    .map(|i| dense_attr_of_size(&format!("a{i}"), 512))
                    .collect(),
            ));
        }
        for n in [100usize, 200] {
            shapes.push((
                format!("{n} x 512B"),
                (0..n)
                    .map(|i| dense_attr_of_size(&format!("a{i}"), 512))
                    .collect(),
            ));
        }

        // Objects this size need a block from a row past the last direct one, so
        // the heap reaches them through nested indirect blocks.
        shapes.push((
            "40 x 60,000B".to_string(),
            (0..40)
                .map(|i| dense_attr_of_size(&format!("big{i}"), 60_000))
                .collect(),
        ));

        // The managed/huge boundary, from both sides and mixed. "One past" also
        // leaves the doubling table with no managed objects at all.
        shapes.push((
            "at the managed limit".to_string(),
            vec![dense_attr_of_size("edge", DENSE_ATTR_MAX_MANAGED_OBJECT)],
        ));
        shapes.push((
            "one past the managed limit".to_string(),
            vec![dense_attr_of_size(
                "edge",
                DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
            )],
        ));
        // Managed attributes on both sides of the huge ones, so a managed object
        // that *follows* a huge one still takes the heap offset its own position
        // in the managed order implies.
        let mut mixed: Vec<AttributeMessage> = (0..10)
            .map(|i| dense_attr_of_size(&format!("small{i}"), 300))
            .collect();
        mixed.push(dense_attr_of_size(
            "huge0",
            DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
        ));
        mixed.push(dense_attr_of_size(
            "huge1",
            DENSE_ATTR_MAX_MANAGED_OBJECT * 4,
        ));
        mixed.extend((10..20).map(|i| dense_attr_of_size(&format!("small{i}"), 300)));
        shapes.push(("managed and huge mixed".to_string(), mixed));

        for (label, attrs) in shapes {
            assert_eq!(dense_attrs_check(&attrs), Ok(()), "{label}");
            let plan = dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked).unwrap();
            for base in [0u64, 0x1000, 0x8000_0000].map(StoredAddress::new) {
                let built = plan.build(base).unwrap();
                assert_eq!(
                    plan.blob_len(),
                    built.blob.len() as u64,
                    "planned length must match the emitted heap for {label} at base {:#x}",
                    base.get()
                );
            }
        }
    }

    /// The attribute names of a built heap's name-index records, in the order the
    /// index stores them.
    ///
    /// A record carries the attribute's creation order — its index in `attrs` —
    /// which names it without decoding the heap. Only valid for a set small enough
    /// that the index is a single leaf.
    fn name_index_order(attrs: &[AttributeMessage]) -> Vec<String> {
        const RECORD: usize = 8 + 1 + 4 + 4;
        let blob = build_dense_attrs(
            attrs,
            DenseAttrCreationOrder::Untracked,
            StoredAddress::new(0),
        )
        .unwrap()
        .blob;
        let header = blob
            .windows(4)
            .position(|w| w == b"BTHD")
            .expect("a name index has a header");
        // The single-leaf assumption above, enforced rather than assumed: the
        // header's depth field sits past signature(4) + version(1) + type(1) +
        // node size(4) + record size(2).
        let depth = u16::from_le_bytes(blob[header + 12..header + 14].try_into().expect("2 bytes"));
        assert_eq!(depth, 0, "the fixture outgrew a single leaf");
        let leaf = blob
            .windows(4)
            .position(|w| w == b"BTLF")
            .expect("a name index has a leaf node");
        // signature(4) + version(1) + type(1), then the records.
        (0..attrs.len())
            .map(|i| {
                let at = leaf + 6 + i * RECORD + 9;
                let order = u32::from_le_bytes(blob[at..at + 4].try_into().expect("4 bytes"));
                attrs[order as usize].name.clone()
            })
            .collect()
    }

    /// The index order is a function of the names alone, never of the order the
    /// attributes arrived in.
    ///
    /// This is the property that makes a file written by a version that ordered a
    /// name-hash collision wrong (issue #225) repairable rather than lost: any
    /// path that reads such a file's attributes and writes them out again —
    /// `repack`, the editor's object copy, a plain read-and-rebuild — emits the
    /// same correct index whatever order they came back in. Breaking the tie on
    /// the hash alone would not give that, because the sort would carry through
    /// whatever relative order the colliding pair was handed in.
    #[test]
    fn the_name_index_order_does_not_depend_on_insertion_order() {
        // "k155448" sorts before "k69209" and hashes the same, so an order that is
        // not fully determined by the names shows up as a difference here.
        let names = [
            "k69209", "k155448", "zeta", "alpha", "m", "beta", "gamma", "delta", "epsilon", "eta",
        ];
        let forward: Vec<AttributeMessage> = names
            .iter()
            .map(|n| type_builders::build_attr_message(n, &AttrValue::I64(1)).unwrap())
            .collect();
        let mut reversed = forward.clone();
        reversed.reverse();

        let order = name_index_order(&forward);
        assert_eq!(
            order,
            name_index_order(&reversed),
            "the index order changed with the insertion order"
        );

        // Non-vacuity, asserted on the hashes rather than on where the two
        // records landed: names that stopped colliding could still come out
        // adjacent and in this order by chance, leaving nothing under test.
        assert_eq!(
            crate::checksum::jenkins_lookup3(b"k69209"),
            crate::checksum::jenkins_lookup3(b"k155448"),
            "the fixture names no longer hash alike; pick a new colliding pair"
        );
        let at = |name: &str| {
            order
                .iter()
                .position(|n| n == name)
                .expect("every attribute is indexed")
        };
        assert_eq!(
            at("k155448") + 1,
            at("k69209"),
            "the colliding pair is not indexed in name order"
        );
    }

    /// Creation indexes for [`scrambled_creation_order`]: a permutation that is
    /// not its own inverse, so an index taken from an attribute's *position*
    /// disagrees with it in both directions. Attribute `i` was created `SCRAMBLE[i]`th.
    const SCRAMBLE: [u16; 6] = [3, 1, 4, 0, 2, 5];

    #[rstest]
    #[case::missing_index(vec![1], 9, false,
        FormatError::Internal("a tracked attribute set needs one creation index per attribute".into()))]
    #[case::extra_index(vec![1, 2, 3], 9, true,
        FormatError::Internal("a tracked attribute set needs one creation index per attribute".into()))]
    #[case::duplicate_index(vec![3, 3], 9, true,
        FormatError::DuplicateAttributeCreationIndex(3))]
    #[case::duplicate_unindexed(vec![3, 3], 9, false,
        FormatError::DuplicateAttributeCreationIndex(3))]
    #[case::counter_equals_index(vec![1, 3], 3, true,
        FormatError::InvalidAttributeCreationCounter { index: 3, next: 3 })]
    #[case::counter_below_index(vec![1, 3], 2, false,
        FormatError::InvalidAttributeCreationCounter { index: 3, next: 2 })]
    fn dense_attribute_preparation_rejects_incoherent_creation_metadata(
        #[case] indices: Vec<u16>,
        #[case] max: u16,
        #[case] indexed: bool,
        #[case] expected: FormatError,
    ) {
        let attrs = [
            type_builders::build_attr_message("a", &AttrValue::I64(1)).unwrap(),
            type_builders::build_attr_message("b", &AttrValue::I64(2)).unwrap(),
        ];
        assert_eq!(
            PreparedDenseAttrs::new(
                &attrs,
                DenseAttrCreationOrder::Tracked {
                    indices,
                    max,
                    indexed
                }
            )
            .unwrap_err(),
            expected,
        );
    }

    #[rstest]
    #[case::tracked_and_indexed(true)]
    #[case::tracked_without_index(false)]
    fn dense_attribute_preparation_preserves_index_gaps_and_counter(#[case] indexed: bool) {
        let attrs = [
            type_builders::build_attr_message("a", &AttrValue::I64(1)).unwrap(),
            type_builders::build_attr_message("b", &AttrValue::I64(2)).unwrap(),
        ];
        let prepared = PreparedDenseAttrs::new(
            &attrs,
            DenseAttrCreationOrder::Tracked {
                indices: vec![7, 2],
                max: u16::MAX,
                indexed,
            },
        )
        .unwrap();
        assert_eq!(prepared.next_index, Some(u16::MAX));
        assert_eq!(prepared.indexed, indexed);
        assert_eq!(
            prepared
                .attributes
                .iter()
                .map(|a| a.creation_index)
                .collect::<Vec<_>>(),
            vec![7, 2]
        );
        assert_eq!(
            prepared
                .attributes
                .iter()
                .map(|a| a.bytes.clone())
                .collect::<Vec<_>>(),
            attrs
                .iter()
                .map(|a| a.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            prepared.corder_order,
            if indexed { vec![1, 0] } else { vec![] }
        );
    }

    #[test]
    fn dense_attribute_preparation_preserves_untracked_positions() {
        let attrs = [
            type_builders::build_attr_message("b", &AttrValue::I64(1)).unwrap(),
            type_builders::build_attr_message("a", &AttrValue::I64(2)).unwrap(),
        ];
        let prepared = PreparedDenseAttrs::new(&attrs, DenseAttrCreationOrder::Untracked).unwrap();
        assert_eq!(prepared.next_index, None);
        assert!(!prepared.indexed);
        assert!(prepared.corder_order.is_empty());
        assert_eq!(
            prepared
                .attributes
                .iter()
                .map(|a| a.creation_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn dense_attribute_preparation_rejects_duplicate_names() {
        let attrs = [
            type_builders::build_attr_message("duplicate", &AttrValue::I64(1)).unwrap(),
            type_builders::build_attr_message("duplicate", &AttrValue::I64(2)).unwrap(),
        ];
        assert_eq!(
            PreparedDenseAttrs::new(&attrs, DenseAttrCreationOrder::Untracked).unwrap_err(),
            FormatError::DuplicateAttributeName("duplicate".into())
        );
    }

    #[rstest]
    #[case::untracked(DenseAttrCreationOrder::Untracked, None, false)]
    #[case::tracked(DenseAttrCreationOrder::Tracked { indices: vec![], max: 17, indexed: true }, Some(17), true)]
    fn dense_attribute_preparation_preserves_empty_set_metadata(
        #[case] creation: DenseAttrCreationOrder,
        #[case] next: Option<u16>,
        #[case] indexed: bool,
    ) {
        let prepared = PreparedDenseAttrs::new(&[], creation).unwrap();
        assert!(prepared.attributes.is_empty());
        assert!(prepared.order.is_empty());
        assert!(prepared.corder_order.is_empty());
        assert_eq!(prepared.next_index, next);
        assert_eq!(prepared.indexed, indexed);
    }

    /// A dense set whose creation order is [`SCRAMBLE`] rather than the order
    /// the attributes are handed over in, so nothing that reads the index
    /// records can pass by reproducing a position.
    fn scrambled_creation_order() -> (Vec<AttributeMessage>, DenseAttrCreationOrder) {
        let attrs: Vec<AttributeMessage> = (0..SCRAMBLE.len())
            .map(|i| {
                type_builders::build_attr_message(&format!("a{i:02}"), &AttrValue::I64(i as i64))
                    .unwrap()
            })
            .collect();
        (
            attrs,
            DenseAttrCreationOrder::Tracked {
                indices: SCRAMBLE.to_vec(),
                max: 40,
                indexed: true,
            },
        )
    }

    /// The B-tree v2 header of `tree_type` in `blob`, as `(offset, record size,
    /// root address, record count)`.
    fn btree_header(blob: &[u8], tree_type: u8) -> (usize, u16, u64, u16) {
        let at = (0..blob.len().saturating_sub(6))
            .find(|&i| &blob[i..i + 4] == b"BTHD" && blob[i + 4] == 0 && blob[i + 5] == tree_type)
            .unwrap_or_else(|| panic!("no v2 B-tree of type {tree_type} in the blob"));
        let record_size = u16::from_le_bytes(blob[at + 10..at + 12].try_into().unwrap());
        let depth = u16::from_le_bytes(blob[at + 12..at + 14].try_into().unwrap());
        assert_eq!(depth, 0, "the fixture outgrew a single leaf");
        let root = u64::from_le_bytes(blob[at + 16..at + 24].try_into().unwrap());
        let nrec = u16::from_le_bytes(blob[at + 24..at + 26].try_into().unwrap());
        (at, record_size, root, nrec)
    }

    /// An object that indexes attribute creation order gets a second v2 B-tree
    /// (type 9) beside the name index, over the same heap IDs and ordered by
    /// creation index — and the Attribute Info message names it.
    ///
    /// libhdf5 reaches this index from `H5Aopen_by_idx`. Its
    /// `H5Aiterate2` builds a table off the name index instead, so a file
    /// missing the tree it declares still iterates correctly. That is why this
    /// reads the bytes rather than leaving the claim to a crosscheck.
    #[test]
    fn an_indexed_dense_set_emits_a_creation_order_btree() {
        const COUNT: usize = SCRAMBLE.len();
        let (attrs, creation) = scrambled_creation_order();
        let heap_address = StoredAddress::new(0x4000);
        let built = build_dense_attrs(&attrs, creation, heap_address).unwrap();

        let info = crate::attribute_info::AttributeInfoMessage::parse(
            &built.attr_info_message,
            OFFSET_SIZE,
        )
        .expect("the emitted message parses");
        assert_eq!(info.max_creation_index, Some(40), "the recorded maximum");
        assert!(info.indexes_creation_order);
        let corder_addr = info
            .btree_creation_order_address
            .expect("an indexed set names its creation-order B-tree");

        let (at, record_size, root, nrec) = btree_header(&built.blob, 9);
        assert_eq!(
            heap_address.offset(at as u64),
            corder_addr,
            "the Attribute Info message names a different address than the tree sits at"
        );
        assert_eq!(
            record_size,
            8 + 1 + 4,
            "heap ID, message flags, creation order"
        );
        assert_eq!(nrec as usize, COUNT);

        // Records are searched by creation index, so they are stored ascending
        // in it — the reverse of the order the attributes were handed over.
        let leaf = (root - heap_address.get()) as usize + 6; // signature(4) + version(1) + type(1)
        let orders: Vec<u32> = (0..COUNT)
            .map(|i| {
                let at = leaf + i * record_size as usize + 9;
                u32::from_le_bytes(built.blob[at..at + 4].try_into().unwrap())
            })
            .collect();
        assert_eq!(orders, (0..COUNT as u32).collect::<Vec<_>>());

        // Which attribute each creation-order record names, established
        // through the heap ID it shares with a name-index record — whose hash
        // identifies the attribute by name.
        let (name_at, name_record, name_root, _) = btree_header(&built.blob, 8);
        assert_ne!(name_at, at, "the two indexes must be separate trees");
        let name_leaf = (name_root - heap_address.get()) as usize + 6;
        for (position, index) in SCRAMBLE.iter().enumerate() {
            let corder_at = leaf + *index as usize * record_size as usize;
            let id = &built.blob[corder_at..corder_at + 8];
            let hash = (0..COUNT)
                .find_map(|j| {
                    let at = name_leaf + j * name_record as usize;
                    (&built.blob[at..at + 8] == id).then(|| {
                        u32::from_le_bytes(built.blob[at + 13..at + 17].try_into().unwrap())
                    })
                })
                .expect("every creation-order record's heap ID is in the name index");
            assert_eq!(
                hash,
                crate::checksum::jenkins_lookup3(format!("a{position:02}").as_bytes()),
                "creation index {index} names the wrong attribute",
            );
        }
    }

    /// A set the object does not track keeps the storage this crate has always
    /// written: a name index alone, and an Attribute Info message that says
    /// nothing about creation order.
    #[test]
    fn an_untracked_dense_set_emits_no_creation_order_btree() {
        let (attrs, _) = scrambled_creation_order();
        let built = build_dense_attrs(
            &attrs,
            DenseAttrCreationOrder::Untracked,
            StoredAddress::new(0x4000),
        )
        .unwrap();
        let info = crate::attribute_info::AttributeInfoMessage::parse(
            &built.attr_info_message,
            OFFSET_SIZE,
        )
        .expect("the emitted message parses");
        assert_eq!(info.max_creation_index, None);
        assert!(!info.indexes_creation_order);
        assert_eq!(info.btree_creation_order_address, None);
        assert!(
            !built
                .blob
                .windows(6)
                .any(|w| &w[..4] == b"BTHD" && w[5] == 9),
            "an untracked set must carry no creation-order index"
        );
    }

    /// The `huge_objects_count` a built heap declares in its fractal-heap header.
    fn huge_object_count(blob: &[u8]) -> u64 {
        let ls = LENGTH_SIZE as usize;
        let os = OFFSET_SIZE as usize;
        assert_eq!(&blob[..4], b"FRHP");
        // version(1) + heap ID length(2) + I/O filter length(2) + flags(1) +
        // max managed object size(4) + next huge object ID + huge B-tree address +
        // free space + free-space manager address + managed space + allocated
        // managed space + allocation iterator + managed object count + huge
        // objects size.
        let at = 4 + 1 + 2 + 2 + 1 + 4 + ls + os + ls + os + ls + ls + ls + ls + ls;
        u64::from_le_bytes(blob[at..at + 8].try_into().unwrap())
    }

    /// Both indexes used to carry a count limit, and it came from the reference C
    /// library deriving a record-count width from a leaf's *capacity*: a node
    /// grown large enough to hold every record in one leaf pushed that width past
    /// the 2 bytes `H5B2__hdr_init` asserts on. A fixed node size takes the
    /// derivation out of the record count's hands entirely — this is the property
    /// that replaced the limits, so it is the one worth pinning.
    #[test]
    fn a_fixed_node_size_keeps_the_derived_count_width_at_one_byte() {
        for record_size in [DENSE_ATTR_BTREE_RECORD, DENSE_ATTR_HUGE_BTREE_RECORD] {
            let (info, depth) = hdf5_pure_format::__private::BTreeV2NodeInfo::for_record_count(
                BTREE_V2_NODE_SIZE,
                record_size,
                hdf5_pure_format::__private::OffsetWidth::Eight,
                0,
            )
            .expect("the emitted geometry is plannable");
            assert_eq!(depth, 0);
            let capacity = info.max_nrec(0);
            assert!(
                capacity >= 20,
                "a {record_size}-byte record should leave room for 20 per node, got {capacity}"
            );
            assert_eq!(
                encoded_size_width(capacity),
                1,
                "a {record_size}-byte record's leaf capacity must stay inside one byte"
            );
        }
    }

    /// Both classes are counted, and the split follows the declared managed-object
    /// limit exactly.
    #[test]
    fn the_managed_object_limit_is_where_storage_changes_class() {
        // Sized under their final names: the name is part of the serialized
        // message, so renaming afterwards would move the attribute back across
        // the very threshold this is testing.
        let mixed = vec![
            dense_attr_of_size("at", DENSE_ATTR_MAX_MANAGED_OBJECT),
            dense_attr_of_size("past", DENSE_ATTR_MAX_MANAGED_OBJECT + 1),
        ];
        assert_eq!(dense_attrs_check(&mixed), Ok(()));
        let blob = build_dense_attrs(
            &mixed,
            DenseAttrCreationOrder::Untracked,
            StoredAddress::new(0),
        )
        .unwrap()
        .blob;
        assert_eq!(huge_object_count(&blob), 1);
        assert_eq!(managed_object_count(&blob), 1);
    }

    /// The `managed_objects_count` a built heap declares in its fractal-heap
    /// header.
    fn managed_object_count(blob: &[u8]) -> u64 {
        let ls = LENGTH_SIZE as usize;
        let os = OFFSET_SIZE as usize;
        assert_eq!(&blob[..4], b"FRHP");
        let at = 4 + 1 + 2 + 2 + 1 + 4 + ls + os + ls + os + ls + ls + ls;
        u64::from_le_bytes(blob[at..at + 8].try_into().unwrap())
    }

    /// The heap header's root-block address and its "current # of rows in root
    /// indirect block", the field that says which kind of block that address
    /// points at.
    fn root_block(blob: &[u8]) -> (usize, u16) {
        let ls = LENGTH_SIZE as usize;
        let os = OFFSET_SIZE as usize;
        assert_eq!(&blob[..4], b"FRHP");
        // The fixed prefix, then next huge object ID, huge B-tree address, free
        // space, free-space manager address, the eight 8-byte statistics, table
        // width, starting block size, maximum direct block size, maximum heap
        // size and starting root rows.
        let at = 4 + 1 + 2 + 2 + 1 + 4 + ls + os + ls + os + 8 * ls + 2 + ls + ls + 2 + 2;
        let address = u64::from_le_bytes(blob[at..at + os].try_into().unwrap());
        let rows = u16::from_le_bytes(blob[at + os..at + os + 2].try_into().unwrap());
        (address as usize, rows)
    }

    /// The root stays a single direct block only while the attributes fit one
    /// starting-size block, which is where libhdf5 also grows a
    /// root indirect block instead of a bigger block.
    #[test]
    fn the_root_grows_into_an_indirect_block_rather_than_a_bigger_direct_one() {
        let fits = vec![dense_attr_of_size("a", 400)];
        let blob = build_dense_attrs(
            &fits,
            DenseAttrCreationOrder::Untracked,
            StoredAddress::new(0),
        )
        .unwrap()
        .blob;
        let (address, rows) = root_block(&blob);
        assert_eq!(rows, 0, "one starting-size block still holds this heap");
        assert_eq!(&blob[address..address + 4], b"FHDB");

        let spills: Vec<AttributeMessage> = (0..8)
            .map(|i| dense_attr_of_size(&format!("a{i}"), 400))
            .collect();
        let blob = build_dense_attrs(
            &spills,
            DenseAttrCreationOrder::Untracked,
            StoredAddress::new(0),
        )
        .unwrap()
        .blob;
        let (address, rows) = root_block(&blob);
        assert!(rows >= 1, "content past one block needs an indirect root");
        assert_eq!(&blob[address..address + 4], b"FHIB");
    }

    /// An attribute close to the managed-object limit needs the table's largest
    /// direct block, and those only recur inside *nested* indirect blocks once
    /// the root's own row of them is used up. Reading every attribute back proves
    /// the emitter and the reader agree about where the deeper blocks sit.
    #[test]
    fn attributes_survive_a_heap_with_nested_indirect_blocks() {
        let mut fw = FileWriter::new();
        let ds = fw.create_dataset("data");
        ds.with_f64_data(&[1.0]);
        // Past DENSE_ATTR_THRESHOLD so the set goes to the heap at all, and each
        // one big enough that only the table's largest direct block will take it.
        let count = DENSE_ATTR_THRESHOLD + 2;
        let value = |i: usize| char::from(b'a' + i as u8).to_string().repeat(60_000);
        for i in 0..count {
            ds.set_attr(&format!("big{i}"), AttrValue::AsciiString(value(i)));
        }
        let bytes = fw.finish().unwrap();

        let nested = bytes.windows(4).filter(|w| *w == b"FHIB").count();
        assert!(
            nested >= 2,
            "expected a nested indirect block, got {nested}"
        );

        let sb = crate::superblock::read(bytes.as_slice())
            .unwrap()
            .superblock;
        let addr = group_v2::resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("data"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse_with_widths(
            &bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let attrs = crate::attribute::extract_attributes_full(
            &bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(BaseAddress::new(0)),
            &hdr,
            WIDTHS,
            None,
        )
        .unwrap();
        assert_eq!(attrs.len(), count);
        for i in 0..count {
            let attr = attrs.iter().find(|a| a.name == format!("big{i}")).unwrap();
            assert_eq!(
                crate::data_read::read_as_strings(&attr.raw_data, &attr.datatype).unwrap(),
                vec![value(i)]
            );
        }
    }

    /// The number of `i64` elements in the largest attribute whose serialized
    /// message still fits the object header's 2-byte message-size field, derived
    /// from a measured probe rather than hard-coded so it tracks the encoder.
    fn largest_fitting_i64_attr_elements() -> usize {
        let one =
            type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; 1]))
                .unwrap();
        let overhead = one.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len() - 8;
        (OBJECT_HEADER_MESSAGE_MAX - overhead) / 8
    }

    #[test]
    fn compact_attr_at_the_message_size_limit_is_written() {
        let n = largest_fitting_i64_attr_elements();
        let attr =
            type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![7i64; n]))
                .unwrap();
        // Pin the probe to the boundary itself: one more element must not fit,
        // or this test would still pass while exercising a tiny attribute.
        let size = attr.serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len();
        assert!(size <= OBJECT_HEADER_MESSAGE_MAX);
        assert!(
            size + 8 > OBJECT_HEADER_MESSAGE_MAX,
            "probe is not at the limit (got {size})"
        );

        let mut fw = FileWriter::new();
        fw.set_root_attr("boundary", AttrValue::I64Array(vec![7i64; n]));
        fw.create_dataset("d").with_f64_data(&[1.0]);
        let bytes = fw.finish().unwrap();

        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(attrs.len(), 1);
    }

    /// One byte past the compact limit the attribute is not refused — it moves to
    /// the fractal heap, which has no such field. This is the only way a lone
    /// large attribute can be written, since an object with one attribute is far
    /// below the count at which dense storage would otherwise be chosen.
    #[test]
    fn an_attr_past_the_message_size_limit_moves_to_dense_storage() {
        let n = largest_fitting_i64_attr_elements() + 1;
        let attrs = vec![
            type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; n]))
                .unwrap(),
        ];
        assert!(
            attrs[0].serialize(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len()
                > OBJECT_HEADER_MESSAGE_MAX
        );
        assert!(
            needs_dense_attrs(&attrs).unwrap(),
            "one oversized attribute must select dense storage by itself"
        );

        let mut fw = FileWriter::new();
        fw.set_root_attr("boundary", AttrValue::I64Array(vec![7i64; n]));
        fw.create_dataset("d").with_f64_data(&[1.0]);
        let bytes = fw.finish().expect("written, not refused");

        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(attrs.len(), 1);
        match attrs.get("boundary") {
            Some(AttrValue::I64Array(v)) => assert_eq!(v.len(), n),
            other => panic!("expected the attribute back, got {other:?}"),
        }
    }

    /// And the compact path is still chosen for everything that fits, so the size
    /// rule has not promoted every attribute to a heap.
    #[test]
    fn an_attr_at_the_message_size_limit_stays_compact() {
        let n = largest_fitting_i64_attr_elements();
        let attrs = vec![
            type_builders::build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; n]))
                .unwrap(),
        ];
        assert!(!needs_dense_attrs(&attrs).unwrap());
    }

    /// Read a dataset's VL-string byte objects from a freshly-written file.
    fn read_vl_bytes(bytes: Vec<u8>, path: &str) -> Vec<crate::vl_data::VlByteObject> {
        let file = crate::reader::File::from_bytes(bytes).unwrap();
        file.dataset(path)
            .unwrap()
            .read_vlen_string_bytes(crate::vl_data::VlenStringReadOptions::default())
            .unwrap()
    }

    #[test]
    fn vlen_string_dataset_roundtrips_values() {
        let mut fw = FileWriter::new();
        fw.create_dataset("labels")
            .with_vlen_strings(&["alpha", "beta", "gamma"]);
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "labels");
        let got: Vec<_> = objs
            .iter()
            .map(|o| match o {
                crate::vl_data::VlByteObject::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
                crate::vl_data::VlByteObject::Null => "<null>".to_string(),
            })
            .collect();
        assert_eq!(got, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn vlen_string_dataset_preserves_null_vs_empty() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let elements = vec![
            VlStringElement::Bytes(b"hi".to_vec()),
            VlStringElement::Null,
            VlStringElement::Bytes(Vec::new()), // empty string, not null
            VlStringElement::Bytes(b"end".to_vec()),
        ];
        let mut fw = FileWriter::new();
        fw.create_dataset("mixed")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "mixed");
        assert_eq!(
            objs,
            vec![
                VlByteObject::Bytes(b"hi".to_vec()),
                VlByteObject::Null,
                VlByteObject::Bytes(Vec::new()),
                VlByteObject::Bytes(b"end".to_vec()),
            ]
        );
    }

    #[test]
    fn vlen_string_dataset_preserves_embedded_nul() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Ascii);
        let payload = b"a\0b\0c".to_vec();
        let elements = vec![VlStringElement::Bytes(payload.clone())];
        let mut fw = FileWriter::new();
        fw.create_dataset("nul")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "nul");
        assert_eq!(objs, vec![VlByteObject::Bytes(payload)]);
    }

    #[test]
    fn vlen_string_dataset_preserves_non_utf8_bytes() {
        // The byte-exact write/read path must round-trip a payload that is not
        // valid UTF-8 (the headline faithfulness claim for issue #83). A
        // String-based path would corrupt this via lossy decoding; the
        // VlStringElement::Bytes / read_vlen_string_bytes path must not.
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Ascii);
        let payload = vec![0xffu8, 0xfe, 0x80, 0x00, 0x41];
        let elements = vec![VlStringElement::Bytes(payload.clone())];
        let mut fw = FileWriter::new();
        fw.create_dataset("raw")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "raw");
        assert_eq!(objs, vec![VlByteObject::Bytes(payload)]);
    }

    #[test]
    fn vlen_string_dataset_2d_shape_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("grid")
            .with_vlen_strings(&["a", "bb", "ccc", "dddd"])
            .with_shape(&[2, 2]);
        let bytes = fw.finish().unwrap();
        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = file.dataset("grid").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![2, 2]);
        assert_eq!(
            ds.read_vlen_strings(crate::vl_data::VlenStringReadOptions::default())
                .unwrap(),
            vec!["a", "bb", "ccc", "dddd"]
        );
    }

    #[test]
    fn vlen_string_dataset_all_null_no_heap() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let elements = vec![VlStringElement::Null, VlStringElement::Null];
        let mut fw = FileWriter::new();
        fw.create_dataset("nulls")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "nulls");
        assert_eq!(objs, vec![VlByteObject::Null, VlByteObject::Null]);
    }

    #[test]
    fn vlen_string_dataset_with_nulls_spans_multiple_heap_collections() {
        // A null element takes no heap object, so the collection an element's
        // reference resolves to follows its *object* position, not its element
        // position: interleaving nulls shifts the split point away from element
        // 65,535. Elements around both are checked, plus the nulls themselves.
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let count = 100_000;
        let elements: Vec<VlStringElement> = (0..count)
            .map(|i| {
                if i % 3 == 0 {
                    VlStringElement::Null
                } else {
                    VlStringElement::Bytes(format!("s{i}").into_bytes())
                }
            })
            .collect();
        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let mut fw = FileWriter::new();
        fw.create_dataset("mixed")
            .with_vlen_string_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();
        let objs = read_vl_bytes(bytes, "mixed");

        assert_eq!(objs.len(), count);
        // Two thirds of the elements carry objects, so the 65,536th object —
        // the first of the second collection — is element 98,303, not 65,535.
        for i in [0, 1, 65_535, 98_302, 98_303, 98_304, count - 1] {
            let expected = if i % 3 == 0 {
                VlByteObject::Null
            } else {
                VlByteObject::Bytes(format!("s{i}").into_bytes())
            };
            assert_eq!(objs[i], expected, "element {i} did not round-trip");
        }
    }

    /// A chunked VL-string dataset has its heap collections placed ahead of the
    /// object headers, so the references inside its chunks carry real addresses
    /// (issue #109). Reading the elements back is the check that the addresses
    /// patched before chunk encoding are the ones the collections landed at.
    #[test]
    fn chunked_vlen_string_dataset_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("chunked")
            .with_vlen_strings(&["a", "bb", "ccc", "dddd"])
            .with_chunks(&[2]);
        let bytes = fw.finish().unwrap();
        let f = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = f.dataset("chunked").unwrap();
        assert_eq!(ds.read_string().unwrap(), ["a", "bb", "ccc", "dddd"]);
    }

    /// The compressed case: the filter runs over element bytes whose heap
    /// addresses are already final, which is the whole reason the collections are
    /// placed first. A stale placeholder address would survive compression and
    /// read back as a dangling reference.
    #[test]
    #[cfg(feature = "deflate")]
    fn filtered_chunked_vlen_string_dataset_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("filtered")
            .with_vlen_strings(&["alpha", "beta", "gamma", "delta"])
            .with_chunks(&[2])
            .with_deflate(DeflateLevel::Six);
        let bytes = fw.finish().unwrap();
        let f = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = f.dataset("filtered").unwrap();
        assert_eq!(
            ds.read_string().unwrap(),
            ["alpha", "beta", "gamma", "delta"]
        );
    }

    /// A resizable (unlimited) VL-string dataset takes the same path — `maxshape`
    /// alone makes a dataset chunked.
    #[test]
    fn resizable_vlen_string_dataset_roundtrips() {
        let mut fw = FileWriter::new();
        fw.create_dataset("growable")
            .with_vlen_strings(&["one", "two", "three"])
            .with_shape(&[3])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[2]);
        let bytes = fw.finish().unwrap();
        let f = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = f.dataset("growable").unwrap();
        assert_eq!(ds.read_string().unwrap(), ["one", "two", "three"]);
    }

    /// Null elements keep an undefined address through the early patch, exactly as
    /// they do on the contiguous path: the mask selects which references move.
    #[test]
    fn chunked_vlen_string_dataset_preserves_nulls() {
        use crate::type_builders::VlStringElement;
        use crate::vl_data::VlByteObject;

        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let elements = vec![
            VlStringElement::Bytes(b"set".to_vec()),
            VlStringElement::Null,
            VlStringElement::Bytes(Vec::new()), // empty string, not null
            VlStringElement::Bytes(b"tail".to_vec()),
        ];
        let mut fw = FileWriter::new();
        fw.create_dataset("mixed")
            .with_vlen_string_elements(dt, &elements)
            .unwrap()
            .with_chunks(&[2]);
        let bytes = fw.finish().unwrap();
        assert_eq!(
            read_vl_bytes(bytes, "mixed"),
            vec![
                VlByteObject::Bytes(b"set".to_vec()),
                VlByteObject::Null,
                VlByteObject::Bytes(Vec::new()),
                VlByteObject::Bytes(b"tail".to_vec()),
            ]
        );
    }

    /// A file with only *contiguous* VL datasets must keep the established layout
    /// (collections after the data, nothing placed early), so adding the chunked
    /// capability cannot perturb the bytes of files that do not use it.
    #[test]
    fn contiguous_vlen_layout_is_unchanged_by_the_early_path() {
        let build = || {
            let mut fw = FileWriter::new();
            fw.create_dataset("plain")
                .with_vlen_strings(&["a", "bb", "ccc"]);
            fw.finish().unwrap()
        };
        // The root group object header sits immediately after the superblock when
        // nothing is placed early.
        let bytes = build();
        assert_eq!(
            &bytes[SUPERBLOCK_SIZE..SUPERBLOCK_SIZE + 4],
            b"OHDR",
            "a contiguous-only VL file must still open with the root OH at SUPERBLOCK_SIZE"
        );
    }

    #[test]
    fn vlen_sequence_dataset_roundtrips_i32() {
        // Non-string VL (`H5T_VLEN { i32 }`): the per-element reference stores an
        // element *count*, while the heap object holds count*4 bytes. The
        // writer/reader pair must agree on that, including an empty sequence.
        use crate::type_builders::VlStringElement;
        use crate::vl_data::{VlByteObject, VlenStringReadOptions};

        let dt = Datatype::VariableLength(VariableLengthType::sequence(
            crate::type_builders::make_i32_type(),
        ));
        let seqs: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![], vec![-7, 42]];
        let elements: Vec<VlStringElement> = seqs
            .iter()
            .map(|s| VlStringElement::Bytes(s.iter().flat_map(|v| v.to_le_bytes()).collect()))
            .collect();
        let mut fw = FileWriter::new();
        fw.create_dataset("seq")
            .with_vlen_sequence_elements(dt, &elements)
            .unwrap();
        let bytes = fw.finish().unwrap();

        let file = crate::reader::File::from_bytes(bytes).unwrap();
        let ds = file.dataset("seq").unwrap();
        assert!(
            matches!(
                ds.datatype().unwrap(),
                Datatype::VariableLength(VariableLengthType::Sequence { .. })
            ),
            "datatype must stay a non-string variable-length sequence"
        );
        let (objs, elem_size) = ds
            .read_vlen_sequence_bytes(VlenStringReadOptions::default())
            .unwrap();
        assert_eq!(elem_size, 4);
        let got: Vec<Vec<i32>> = objs
            .iter()
            .map(|o| match o {
                VlByteObject::Null => Vec::new(),
                VlByteObject::Bytes(b) => b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect(),
            })
            .collect();
        assert_eq!(got, seqs);
    }

    #[test]
    fn vlen_sequence_rejects_string_datatype() {
        // The sequence builder must refuse a string-shaped VL datatype, which
        // belongs to the VL-string path.
        use crate::type_builders::VlStringElement;
        let dt = crate::type_builders::make_vlen_string_type(CharacterSet::Utf8);
        let mut fw = FileWriter::new();
        let res = fw
            .create_dataset("x")
            .with_vlen_sequence_elements(dt, &[VlStringElement::Bytes(b"hi".to_vec())]);
        assert!(matches!(res, Err(FormatError::TypeMismatch { .. })));
    }

    // ---- library-version bounds select the on-disk format ----

    /// A one-dataset file written under `bounds`.
    fn write_bounded(bounds: Option<(LibVer, LibVer)>) -> Result<Vec<u8>, FormatError> {
        let mut fw = FileWriter::new();
        if let Some((low, high)) = bounds {
            fw.with_libver_bounds(low, high);
        }
        fw.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
        fw.finish()
    }

    /// The version byte of `path`'s data-layout message.
    ///
    /// Read out of the object header rather than inferred from the superblock:
    /// the two are set from one resolved format, and a test that derived one from
    /// the other could not tell whether they had come apart.
    fn layout_message_version(bytes: &[u8], path: &str) -> u8 {
        let sb = crate::superblock::read(bytes).unwrap().superblock;
        let addr = group_v2::resolve_path_any(
            bytes,
            AccessMode::ReadOnly,
            &sb,
            WIDTHS,
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse(path),
        )
        .unwrap();
        let oh = ObjectHeader::parse_with_widths(
            bytes,
            AccessMode::ReadOnly,
            addr as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        oh.messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
            .expect("a dataset carries a data-layout message")
            .data[0]
    }

    /// `high` selects the format, and every version field in the file follows
    /// that one decision.
    #[test]
    fn libver_bounds_select_the_superblock_and_layout_versions() {
        for (bounds, superblock, layout) in [
            (None, 3, 4),
            (Some((LibVer::Earliest, LibVer::LATEST)), 3, 4),
            (Some((LibVer::Earliest, LibVer::V110)), 3, 4),
            (Some((LibVer::V110, LibVer::LATEST)), 3, 4),
            // A lower bound above the newest format this crate writes licenses
            // newer encodings without requiring any, so it writes the 1.10 one.
            (Some((LibVer::LATEST, LibVer::LATEST)), 3, 4),
            (Some((LibVer::V112, LibVer::V112)), 3, 4),
            (Some((LibVer::Earliest, LibVer::V18)), 2, 3),
            (Some((LibVer::V18, LibVer::V18)), 2, 3),
        ] {
            let bytes = write_bounded(bounds).expect("these bounds are satisfiable");
            let sig = usize::try_from(
                crate::superblock::read(bytes.as_slice())
                    .unwrap()
                    .address
                    .get(),
            )
            .unwrap();
            assert_eq!(
                bytes[sig + 8],
                superblock,
                "superblock version under bounds {bounds:?}"
            );
            assert_eq!(
                layout_message_version(&bytes, "values"),
                layout,
                "data-layout message version under bounds {bounds:?}"
            );
        }
    }

    /// A version 2 superblock and a version 3 contiguous layout message are the
    /// only differences, so the 1.8 file is otherwise the 1.10 file byte for
    /// byte. Stated as a test because it is what makes the older format cheap:
    /// were the bodies to diverge, this would fail rather than quietly emit a
    /// second layout to keep in sync.
    #[test]
    fn the_two_formats_differ_only_in_their_version_bytes() {
        let newer = write_bounded(Some((LibVer::Earliest, LibVer::V110))).unwrap();
        let older = write_bounded(Some((LibVer::Earliest, LibVer::V18))).unwrap();
        assert_eq!(newer.len(), older.len(), "the two formats differ in size");

        let differing: Vec<usize> = (0..newer.len()).filter(|&i| newer[i] != older[i]).collect();
        let sig = usize::try_from(
            crate::superblock::read(newer.as_slice())
                .unwrap()
                .address
                .get(),
        )
        .unwrap();

        // Ten bytes, and every one of them accounted for: the superblock's
        // version byte and the four-byte checksum covering it, the layout
        // message's version byte, and the four-byte checksum of the object header
        // holding it. Any eleventh byte means a body diverged, which is the thing
        // worth failing over.
        assert_eq!(
            differing.len(),
            10,
            "expected two version bytes and the two checksums over them, got {differing:?}"
        );
        assert_eq!(
            &differing[..5],
            &[sig + 8, sig + 44, sig + 45, sig + 46, sig + 47],
            "the superblock version byte and its checksum must be the first to differ"
        );
        assert_eq!(
            newer[differing[5]], 4,
            "the sixth differing byte is the data-layout message version"
        );
        assert_eq!(older[differing[5]], 3);
        let trailing = &differing[6..];
        assert!(
            trailing[0] > differing[5] && trailing.windows(2).all(|w| w[1] == w[0] + 1),
            "the last four differing bytes must be the contiguous checksum trailing \
             the object header that holds the layout message, got {differing:?}"
        );
    }

    /// An upper bound older than the 1.8 format, and one below the lower bound,
    /// are what is left unsatisfiable — and the inverted pairs keep a `high` of
    /// 1.10 or newer, so nothing but the inversion can be what refuses them.
    #[test]
    fn bounds_admitting_no_format_this_crate_writes_are_refused() {
        for (low, high) in [
            (LibVer::Earliest, LibVer::Earliest),
            (LibVer::LATEST, LibVer::V18),
            (LibVer::V114, LibVer::V110),
            (LibVer::LATEST, LibVer::V112),
            (LibVer::V112, LibVer::V110),
        ] {
            let err = write_bounded(Some((low, high))).unwrap_err();
            assert!(
                matches!(err, FormatError::LibverBoundsUnsatisfiable { .. }),
                "bounds [{}, {}] gave {err:?}",
                low.name(),
                high.name()
            );
        }
    }

    #[test]
    fn content_the_older_format_cannot_carry_is_refused_not_upgraded() {
        let mut fw = FileWriter::new();
        fw.with_libver_bounds(LibVer::Earliest, LibVer::V18);
        fw.create_dataset("d")
            .with_i32_data(&(0..100).collect::<Vec<_>>())
            .with_chunks(&[10]);
        assert!(matches!(
            fw.finish().unwrap_err(),
            FormatError::LibverTooOldForContent { .. }
        ));

        let mut fw = FileWriter::new();
        fw.with_libver_bounds(LibVer::Earliest, LibVer::V18);
        fw.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
        );
        fw.create_dataset("d").with_i32_data(&[1]);
        assert!(matches!(
            fw.finish().unwrap_err(),
            FormatError::LibverTooOldForContent { .. }
        ));

        let mut fw = FileWriter::new();
        fw.with_libver_bounds(LibVer::Earliest, LibVer::V18);
        fw.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
        fw.create_dataset("d").with_i32_data(&[1]);
        assert!(matches!(
            fw.finish().unwrap_err(),
            FormatError::LibverTooOldForContent { .. }
        ));

        // An unlimited dimension implies chunked storage, so it takes the same
        // refusal rather than reaching the chunk writer.
        let mut fw = FileWriter::new();
        fw.with_libver_bounds(LibVer::Earliest, LibVer::V18);
        fw.create_dataset("d")
            .with_i32_data(&[1, 2, 3])
            .with_shape(&[3])
            .with_maxshape(&[MaxExtent::Unlimited]);
        assert!(matches!(
            fw.finish().unwrap_err(),
            FormatError::LibverTooOldForContent { .. }
        ));
    }
    // "The Attribute Message" in format specification, version 4.0, defines message
    // versions 2 and 3 and the version 3 Name Character Set Encoding field.
    const ATTRIBUTE_MESSAGE_VERSION_TWO: u8 = 2;
    const ATTRIBUTE_MESSAGE_VERSION_THREE: u8 = 3;
    const ATTRIBUTE_NAME_CHARACTER_SET_OFFSET: usize = 8;
    const ASCII_NAME_CHARACTER_SET: u8 = 0;
    const UTF8_NAME_CHARACTER_SET: u8 = 1;
    // Section `subsec_fmt4_infra_fractalheap`, version 4.0, defines the heap header signature.
    const FRACTAL_HEAP_SIGNATURE: &[u8] = b"FRHP";
    // Section `subsubsec_fmt4_dataobject_hdr_msg_groupinfo`, version 4.0, defines version zero
    // and flags zero for Group Info messages with default phase-change and entry settings.
    const DEFAULT_GROUP_INFO: [u8; 2] = [0, 0];
    const GLOBAL_HEAP_COLLECTION_SIGNATURE: &[u8] = b"GCOL";
    const OBJECT_HEADER_V2_SIGNATURE: &[u8] = b"OHDR";
}
