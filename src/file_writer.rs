//! HDF5 file creation (write pipeline).
//!
//! Produces valid HDF5 files with v3 superblock, v2 object headers,
//! link messages, contiguous datasets, inline and dense attributes.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::format;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::address::BaseAddress;
use crate::attribute::AttributeMessage;
use crate::attribute_info::AttributeInfoMessage;
use crate::btree_v2_write::{self, BTreeV2Plan};
use crate::chunked_write::{
    ByteSink, ChunkOptions, ChunkProvider, ChunkedMeasure, CompressedChunkSet, StorageAllocation,
    VerbatimLayout, VerbatimPlan, assemble_chunked_at, compress_chunks, emit_chunked_data_verbatim,
    measure_chunked_at, plan_chunked_data_verbatim,
};
use crate::convert::TryToUsize;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_create_properties::FileCreateProperties;
use crate::file_space_info::{
    DEFAULT_PAGE_SIZE, DEFAULT_THRESHOLD, FileSpaceInfo, FileSpaceStrategy, NUM_FILE_FSM_MANAGERS,
};
use crate::fractal_heap_write::{self, ManagedPlan, PlanRefusal};
use crate::free_space_manager::{
    FreeSection, SECT_CLASS_LARGE, SECT_CLASS_SMALL, fshd_len, fsse_len, serialize_file_fsm,
};
use crate::libver::LibVer;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header_writer::ObjectHeaderWriter;
use crate::shared_message::DatatypeLocation;
use crate::superblock::Superblock;
use crate::type_builders::{
    AttrSpec, CommittedDatatype, DatasetBuilder, FinishedGroup, GroupBuilder, VlStringStaging,
    build_global_heap_collections, patch_vl_refs, patch_vl_refs_masked, write_reference_address,
};
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

// `AttrValue` lives in `type_builders`; `types` and `mat` reference it through
// this module's path, so keep it re-exported here.
pub use crate::type_builders::AttrValue;

use crate::datatype::{CharacterSet, Datatype};

/// The width of every address in a file this crate writes.
const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
/// The width of every length in a file this crate writes.
const LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;
/// [`OFFSET_WIDTH`] as the `u8` width the writers take.
pub(crate) const OFFSET_SIZE: u8 = OFFSET_WIDTH.get();
/// [`LENGTH_WIDTH`] as the `u8` width the writers take.
pub(crate) const LENGTH_SIZE: u8 = LENGTH_WIDTH.get();
const SUPERBLOCK_SIZE: usize = 48;

/// Object-header message record flags (`H5O_MSG_FLAG_*`).
///
/// The message's content cannot change once written (`H5O_MSG_FLAG_CONSTANT`).
const MSG_CONSTANT: u8 = 0x01;
/// The message body is a reference to the message rather than the message
/// itself (`H5O_MSG_FLAG_SHARED`).
const MSG_SHARED: u8 = 0x02;
/// The message must not be moved into shared storage (`H5O_MSG_FLAG_DONTSHARE`).
const MSG_DONTSHARE: u8 = 0x04;

/// Threshold for switching from compact (inline) to dense attribute storage.
pub(crate) const DENSE_ATTR_THRESHOLD: usize = 8;

/// Round `value` up to the next multiple of `page` (a power of two). Used by the
/// paged file-space writer to page-align region starts and the end-of-allocation.
fn align_up(value: u64, page: u64) -> u64 {
    value.div_ceil(page) * page
}

// ---- OH builders ----

pub(crate) fn build_chunked_dataset_oh(
    dt: &Datatype,
    dt_location: &DatatypeLocation,
    ds: &Dataspace,
    layout_message: &[u8],
    pipeline_message: Option<&[u8]>,
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
    fill: Option<&[u8]>,
) -> Result<Vec<u8>, FormatError> {
    let mut w = ObjectHeaderWriter::new();
    add_datatype(&mut w, dt, dt_location);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(
        MessageType::FillValue,
        crate::fill_value::fill_value_message_v3(fill),
        0x01,
    );
    w.add_message(MessageType::DataLayout, layout_message.to_vec());
    if let Some(pm) = pipeline_message {
        w.add_message(MessageType::FilterPipeline, pm.to_vec());
    }
    add_attributes(&mut w, attrs, attr_info);
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
pub(crate) fn superblock_version(libver: LibVer) -> u8 {
    if libver >= LibVer::V110 { 3 } else { 2 }
}

/// The data-layout message version the given on-disk format emits for a
/// *contiguous* dataset.
///
/// Version 4 arrived with HDF5 1.10, alongside the chunk indices that are its
/// only substantive addition; for a contiguous dataset the version 3 and version
/// 4 bodies are byte-identical (address then size), so the 1.8 format writes the
/// same bytes under the version number a 1.8 library understands.
pub(crate) fn contiguous_layout_version(libver: LibVer) -> u8 {
    if libver >= LibVer::V110 { 4 } else { 3 }
}

pub(crate) fn build_dataset_oh(
    dt: &Datatype,
    dt_location: &DatatypeLocation,
    ds: &Dataspace,
    data_addr: u64,
    data_size: u64,
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
    fill: Option<&[u8]>,
    libver: LibVer,
) -> Result<Vec<u8>, FormatError> {
    let mut w = ObjectHeaderWriter::new();
    add_datatype(&mut w, dt, dt_location);
    w.add_message(MessageType::Dataspace, ds.serialize(LENGTH_SIZE));
    w.add_message_with_flags(
        MessageType::FillValue,
        crate::fill_value::fill_value_message_v3(fill),
        0x01,
    );
    let mut dl = Vec::new();
    dl.push(contiguous_layout_version(libver));
    dl.push(1); // class = contiguous
    dl.extend_from_slice(&data_addr.to_le_bytes());
    dl.extend_from_slice(&data_size.to_le_bytes());
    w.add_message(MessageType::DataLayout, dl);
    add_attributes(&mut w, attrs, attr_info);
    w.serialize()
}

/// Add a dataset's Datatype message: the encoding itself, or — when the type is
/// committed — the reference standing in for it, under a record whose shared flag
/// says the body is one. Without that flag the ten reference bytes decode as a
/// zero-width time datatype, which is the defect issue #254 was filed for.
fn add_datatype(w: &mut ObjectHeaderWriter, dt: &Datatype, location: &DatatypeLocation) {
    match location.reference_bytes(OFFSET_SIZE) {
        Some(reference) => {
            w.add_message_with_flags(MessageType::Datatype, reference, MSG_CONSTANT | MSG_SHARED);
        }
        None => w.add_message_with_flags(MessageType::Datatype, dt.serialize(), MSG_CONSTANT),
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
        MessageType::Datatype,
        dt.serialize(),
        MSG_CONSTANT | MSG_DONTSHARE,
    );
    if references > 1 {
        let mut refcount = Vec::with_capacity(5);
        refcount.push(0); // version
        refcount.extend_from_slice(&references.to_le_bytes());
        w.add_message_with_flags(MessageType::ObjectReferenceCount, refcount, MSG_DONTSHARE);
    }
    w.serialize()
}

pub(crate) fn build_group_oh(
    links: &[LinkMessage],
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
) -> Result<Vec<u8>, FormatError> {
    let mut w = ObjectHeaderWriter::new();
    let mut li = Vec::new();
    li.push(0); // version
    li.push(0); // flags
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap addr = UNDEF
    li.extend_from_slice(&u64::MAX.to_le_bytes()); // btree name index addr = UNDEF
    w.add_message(MessageType::LinkInfo, li);
    // A new-style group (one with a Link Info message) must also carry a Group
    // Info message, or the HDF5 C library refuses to insert links into it:
    // `H5G_obj_insert` reads the Group Info message unconditionally and fails
    // with "message type not found", so the file is readable but not writable by
    // the C library. The minimal body (version 0, no optional fields) leaves the
    // C library to use its defaults (max compact = 8, min dense = 6).
    w.add_message(MessageType::GroupInfo, vec![0, 0]);
    for link in links {
        w.add_message(MessageType::Link, link.serialize(OFFSET_SIZE));
    }
    add_attributes(&mut w, attrs, attr_info);
    w.serialize()
}

/// Add an object's attribute storage to its header: either the Attribute Info
/// message naming a dense (fractal-heap) attribute store, or the inline
/// Attribute messages preceded by the compact-storage Attribute Info message
/// that lets the reference library count them (see
/// [`compact_attribute_info_message`]).
///
/// The dense case takes the Attribute Info message rather than the heap itself
/// because that message is all a header needs. A sizing pass can therefore get
/// its header size from a [`DenseAttrPlan`] without the heap's bytes existing.
fn add_attributes(
    w: &mut ObjectHeaderWriter,
    attrs: &[AttributeMessage],
    attr_info: Option<&[u8]>,
) {
    if let Some(message) = attr_info {
        w.add_message(MessageType::AttributeInfo, message.to_vec());
    } else {
        if !attrs.is_empty() {
            w.add_message(MessageType::AttributeInfo, compact_attribute_info_message());
        }
        for attr in attrs {
            w.add_message(MessageType::Attribute, attr.serialize(LENGTH_SIZE));
        }
    }
}

pub(crate) fn make_link(name: &str, addr: u64) -> LinkMessage {
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
const DENSE_ATTR_MAX_HEAP_SIZE_BITS: u16 = fractal_heap_write::MAX_HEAP_SIZE_BITS;
const DENSE_ATTR_BLOCK_OFFSET_BYTES: usize = fractal_heap_write::BLOCK_OFFSET_BYTES;

/// Whether `attrs` must go in a fractal heap rather than the object header.
///
/// Two independent reasons, matching the reference C library's own disjunction in
/// `H5Oattribute.c` (`nattrs == max_compact || raw_size >= H5O_MESG_MAX_SIZE`):
/// too many attributes to keep compact, *or* one attribute too large for an
/// object-header message, whose size field is 2 bytes wide.
///
/// The second is what lets a single large attribute be written at all. Selecting
/// on count alone would send it to the compact path, where the only available
/// answer is [`FormatError::AttributeMessageTooLarge`] — even though dense
/// storage can hold it, as a huge object if need be.
///
/// The comparison reads `>` against [`OBJECT_HEADER_MESSAGE_MAX`] (65,535) where
/// the C library reads `>=` against `H5O_MESG_MAX_SIZE` (65,536): the same
/// predicate, written from the widest value that fits rather than the first that
/// does not. What each side measures does differ by a byte, since this writer's
/// compact attribute messages are version 2 and the C library's latest-format
/// ones are version 3, one character-set byte longer. An attribute landing in
/// that single-byte window is therefore stored compactly here and densely there.
/// Both are readable, and both stay within the header's field width.
///
/// Variable-length attributes are selected on the same terms as any other. They
/// were briefly excluded from the size half of the rule, because a heap built
/// before their global-heap references were patched embedded the placeholders and
/// this crate's reader then dropped the attribute; the writer now builds each
/// heap after that patching, so there is nothing to exclude.
pub(crate) fn needs_dense_attrs(attrs: &[AttributeMessage]) -> bool {
    attrs.len() > DENSE_ATTR_THRESHOLD
        || attrs
            .iter()
            .any(|a| a.serialize(LENGTH_SIZE).len() > OBJECT_HEADER_MESSAGE_MAX)
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
/// The reference C library splits far earlier — it declares 4,096 for an
/// attribute heap — but the threshold is the heap's own declaration, read back
/// out of the header, so a higher one is equally readable.
pub(crate) const DENSE_ATTR_MAX_MANAGED_OBJECT: usize =
    fractal_heap_write::max_managed_object(OFFSET_SIZE);

/// One name-index B-tree v2 record as [`build_dense_attrs`] writes it: heap
/// ID(8) + message flags(1) + creation order(4) + name hash(4).
const DENSE_ATTR_BTREE_RECORD: u16 = 8 + 1 + 4 + 4;

/// One huge-objects B-tree v2 record (type 1, indirectly accessed and
/// non-filtered): address + length + huge object ID. Matches what
/// `fractal_heap::HugeObjectIndex::decode` reads on the way back in.
const DENSE_ATTR_HUGE_BTREE_RECORD: u16 =
    OFFSET_SIZE as u16 + LENGTH_SIZE as u16 + LENGTH_SIZE as u16;

/// B-tree v2 type for an attribute name index.
const DENSE_ATTR_NAME_BTREE_TYPE: u8 = 8;

/// B-tree v2 type for an attribute creation-order index, and the record it
/// stores: heap ID(8) + message flags(1) + creation order(4). The name index's
/// record is this plus the name hash, which is why the two indexes can be built
/// over the same heap IDs.
const DENSE_ATTR_CORDER_BTREE_TYPE: u8 = 9;
const DENSE_ATTR_CORDER_BTREE_RECORD: u16 = 8 + 1 + 4;

/// B-tree v2 type for a fractal heap's huge objects, indirectly accessed and
/// not filtered.
const DENSE_ATTR_HUGE_BTREE_TYPE: u8 = 1;

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
        /// The next index the object would hand out. The reference C library
        /// does not lower this when an attribute is deleted, so it is carried
        /// rather than derived from `indices`.
        max: u16,
        /// Emit the creation-order B-tree beside the name index.
        indexed: bool,
    },
}

impl DenseAttrCreationOrder {
    /// The creation index to record for attribute `i` in the name index. An
    /// untracked object has none to record; the position it has always written
    /// there is kept, since a reader that has no creation order to compare
    /// against cannot tell the two apart and existing files carry it.
    fn index_of(&self, i: u32) -> u32 {
        match self {
            Self::Untracked => i,
            Self::Tracked { indices, .. } => u32::from(indices[i as usize]),
        }
    }

    /// The Attribute Info message's recorded maximum, where there is one.
    fn max(&self) -> Option<u16> {
        match self {
            Self::Untracked => None,
            Self::Tracked { max, .. } => Some(*max),
        }
    }

    /// Whether a creation-order B-tree is part of this storage.
    fn indexed(&self) -> bool {
        matches!(self, Self::Tracked { indexed: true, .. })
    }

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

/// Whether [`build_dense_attrs`] can faithfully represent `attrs`.
///
/// The one bound the emitter still has to honour is the heap's own address
/// space: its offsets are [`DENSE_ATTR_MAX_HEAP_SIZE_BITS`] wide, so the managed
/// blocks cannot span more than that between them. Nothing else is bounded any
/// more. The attribute and huge-object counts are not, since both indexes are
/// multi-node v2 B-trees that grow a level rather than overflow a leaf; an
/// individual attribute is not, since past [`DENSE_ATTR_MAX_MANAGED_OBJECT`] the
/// emitter changes representation rather than refusing; and the managed total is
/// not, since the blocks holding it are a doubling table rather than one block
/// sized to the whole set.
///
/// What remains refused is what the attribute message itself cannot encode: its
/// name, datatype and dataspace lengths live in 2-byte fields, and huge storage
/// lifts the limit on an attribute's *data*, not on those. Without this check
/// they would truncate silently rather than fail.
///
/// Callers that cannot fall back to a larger layout must refuse rather than
/// mis-encode (see [`build_dense_attrs`]).
pub(crate) fn dense_attrs_check(attrs: &[AttributeMessage]) -> Result<(), FormatError> {
    let mut managed = Vec::new();
    for a in attrs {
        if let Some((field, size)) = a.v3_header_field_overflow(LENGTH_SIZE) {
            return Err(FormatError::AttributeFieldTooLong {
                name: a.name.clone(),
                field,
                size,
                limit: u16::MAX as usize,
            });
        }
        let size = a.serialize_v3(LENGTH_SIZE).len();
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
    match ManagedPlan::new(&managed, OFFSET_SIZE) {
        Ok(_) => Ok(()),
        Err(PlanRefusal::HeapSpace) => Err(FormatError::DenseAttributeHeapTooLarge {
            limit: fractal_heap_write::MAX_HEAP_SPACE,
        }),
        Err(PlanRefusal::Host { bytes }) => Err(FormatError::ValueTooLargeForPlatform {
            value: bytes,
            target: "usize",
        }),
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
    /// Each attribute's version 3 message bytes, in the caller's order.
    serialized: Vec<Vec<u8>>,
    /// Huge-object ID for each attribute whose bytes go outside the managed
    /// blocks; `None` for a managed one.
    huge_id_of: Vec<Option<u64>>,
    huge_count: usize,
    huge_total: u64,
    managed_plan: ManagedPlan,
    name_plan: BTreeV2Plan,
    corder_plan: Option<BTreeV2Plan>,
    huge_plan: Option<BTreeV2Plan>,
    /// `(name hash, attribute index)` in the order the name index is searched.
    order: Vec<(u32, u32)>,
    /// Attribute indices in the order the creation-order index is searched.
    /// Empty unless that index is emitted.
    corder_order: Vec<u32>,
    /// What this set records about attribute creation order.
    creation: DenseAttrCreationOrder,
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
pub(crate) fn dense_attrs_plan(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
) -> DenseAttrPlan {
    debug_assert!(
        match &creation {
            DenseAttrCreationOrder::Untracked => true,
            DenseAttrCreationOrder::Tracked { indices, .. } => indices.len() == attrs.len(),
        },
        "a tracked set needs one creation index per attribute"
    );
    // Dense attrs use v3 attribute messages (adds character set encoding byte).
    let serialized: Vec<Vec<u8>> = attrs.iter().map(|a| a.serialize_v3(LENGTH_SIZE)).collect();

    let name_hashes: Vec<u32> = attrs
        .iter()
        .map(|a| crate::checksum::jenkins_lookup3(a.name.as_bytes()))
        .collect();

    // Huge object IDs are assigned in attribute order, starting at 1 — the
    // reference C library's `H5HF__huge_insert` pre-increments, so 0 is never a
    // valid ID and the header's `next_huge_object_id` ends up holding the last
    // one assigned rather than the next one free.
    // A count of attributes, so a `usize` — it is bounded by `attrs.len()` and
    // widened only where it goes into one of the header's 8-byte fields.
    let mut huge_id_of: Vec<Option<u64>> = vec![None; attrs.len()];
    let mut huge_count: usize = 0;
    for (slot, bytes) in huge_id_of.iter_mut().zip(&serialized) {
        if bytes.len() > DENSE_ATTR_MAX_MANAGED_OBJECT {
            huge_count += 1;
            *slot = Some(huge_count as u64);
        }
    }
    let huge_total: u64 = serialized
        .iter()
        .zip(&huge_id_of)
        .filter(|(_, id)| id.is_some())
        .map(|(s, _)| s.len() as u64)
        .sum();

    // Only managed objects occupy the doubling table, so only they are planned
    // into it. `managed_sizes` holds their sizes in attribute order, and the
    // plan's object indices are indices into that order.
    let managed_sizes: Vec<u64> = serialized
        .iter()
        .zip(&huge_id_of)
        .filter(|(_, id)| id.is_none())
        .map(|(s, _)| s.len() as u64)
        .collect();
    let managed_plan = ManagedPlan::new(&managed_sizes, OFFSET_SIZE)
        .expect("dense_attrs_check, which every caller must run first, plans the same layout");

    // Every v2 B-tree header this emitter writes has the same fixed layout, so
    // one size covers both the name index and the huge-objects index.
    let bthd_size = btree_v2_write::header_size(OFFSET_SIZE, LENGTH_SIZE);
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
        DENSE_ATTR_NAME_BTREE_TYPE,
        attrs.len(),
        DENSE_ATTR_BTREE_RECORD,
        btree_v2_write::NODE_SIZE,
        OFFSET_SIZE,
    )
    .expect("a 512-byte node holds 29 name records, enough to plan any count");
    let corder_plan = creation.indexed().then(|| {
        BTreeV2Plan::new(
            DENSE_ATTR_CORDER_BTREE_TYPE,
            attrs.len(),
            DENSE_ATTR_CORDER_BTREE_RECORD,
            btree_v2_write::NODE_SIZE,
            OFFSET_SIZE,
        )
        .expect("a 512-byte node holds 38 creation-order records, enough to plan any count")
    });
    let huge_plan = (huge_count > 0).then(|| {
        BTreeV2Plan::new(
            DENSE_ATTR_HUGE_BTREE_TYPE,
            huge_count,
            DENSE_ATTR_HUGE_BTREE_RECORD,
            btree_v2_write::NODE_SIZE,
            OFFSET_SIZE,
        )
        .expect("a 512-byte node holds 20 huge records, enough to plan any count")
    });

    // Blob layout, all relative to the heap address the blob is built for, so
    // its length is the same wherever it lands: heap header, the managed blocks,
    // name index (header + nodes), then — only when there are huge objects — the
    // huge index (header + nodes) and the huge object bytes themselves.
    let managed_off = DENSE_ATTR_FRHP_SIZE as u64;
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

    // Records are ordered the way the index is searched: by name hash, and by
    // the name itself where two names hash alike. The tie-break has to be the
    // name because that is what the reference C library compares on a hash
    // collision — `H5A__dense_fh_name_cmp` does `strcmp` against the name pulled
    // back out of the heap — and a binary search ordered any other way walks
    // away from a colliding record. Rust's `str` ordering is byte-wise over
    // unsigned bytes, which is what `strcmp` specifies, so the two agree for
    // non-ASCII names as well.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "i is an attribute index bounded by the attribute count, far below u32::MAX"
    )]
    let mut order: Vec<(u32, u32)> = (0..attrs.len())
        .map(|i| (name_hashes[i], i as u32))
        .collect();
    order.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| attrs[a.1 as usize].name.cmp(&attrs[b.1 as usize].name))
    });

    // The creation-order index is searched on the creation index alone, which
    // is unique per attribute, so nothing needs a tie-break here.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "i is an attribute index bounded by the attribute count, far below u32::MAX"
    )]
    let mut corder_order: Vec<u32> = if corder_plan.is_some() {
        (0..attrs.len() as u32).collect()
    } else {
        Vec::new()
    };
    corder_order.sort_unstable_by_key(|&i| creation.index_of(i));

    DenseAttrPlan {
        serialized,
        huge_id_of,
        huge_count,
        huge_total,
        managed_plan,
        name_plan,
        corder_plan,
        huge_plan,
        order,
        corder_order,
        creation,
        managed_off,
        btree_off,
        name_nodes_off,
        corder_bthd_off,
        corder_nodes_off,
        huge_bthd_off,
        huge_nodes_off,
        huge_data_off,
        total_len,
    }
}

/// On-disk byte size of the fractal heap header this emitter writes.
const DENSE_ATTR_FRHP_SIZE: usize = {
    let os = OFFSET_SIZE as usize;
    let ls = LENGTH_SIZE as usize;
    4 + 1
        + 2
        + 2
        + 1
        + 4
        + ls
        + os
        + ls
        + os
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + ls
        + 2
        + ls
        + ls
        + 2
        + 2
        + os
        + 2
        + 4
};

impl DenseAttrPlan {
    /// Byte length of the blob [`DenseAttrPlan::build`] produces, at any heap
    /// address.
    pub(crate) fn blob_len(&self) -> u64 {
        self.total_len
    }

    /// The Attribute Info (0x0015) message naming this heap once it is placed at
    /// `heap_address`. Its length does not depend on the address, so an
    /// object-header sizing pass can take it from a provisional one.
    pub(crate) fn attr_info_message(&self, heap_address: u64) -> Vec<u8> {
        self.attribute_info(heap_address).serialize(OFFSET_SIZE)
    }

    /// The Attribute Info message this storage is named by, once placed at
    /// `heap_address`.
    fn attribute_info(&self, heap_address: u64) -> AttributeInfoMessage {
        AttributeInfoMessage {
            max_creation_index: self.creation.max(),
            indexes_creation_order: self.creation.indexed(),
            fractal_heap_address: Some(heap_address),
            btree_name_index_address: Some(heap_address + self.btree_off),
            btree_creation_order_address: self
                .creation
                .indexed()
                .then(|| heap_address + self.corder_bthd_off),
        }
    }

    /// Emit the blob for a heap placed at `heap_address`.
    pub(crate) fn build(&self, heap_address: u64) -> DenseAttrBlob {
        let max_heap_size: u16 = DENSE_ATTR_MAX_HEAP_SIZE_BITS;
        let block_offset_bytes = DENSE_ATTR_BLOCK_OFFSET_BYTES; // 5
        let heap_id_length: u16 = 8;

        let Self {
            serialized,
            huge_id_of,
            huge_count,
            huge_total,
            managed_plan,
            name_plan,
            corder_plan,
            huge_plan,
            order,
            corder_order,
            creation,
            ..
        } = self;
        let huge_count = *huge_count;

        // Every address the blob embeds is `heap_address + <plan offset>`, which
        // is what makes the bytes relocatable: the same plan emits the same
        // length wherever the heap is placed.
        let frhp_addr = heap_address;
        let managed_addr = heap_address + self.managed_off;
        let btree_addr = heap_address + self.btree_off;
        let name_nodes_addr = heap_address + self.name_nodes_off;
        let corder_bthd_addr = heap_address + self.corder_bthd_off;
        let corder_nodes_addr = heap_address + self.corder_nodes_off;
        let huge_bthd_addr = heap_address + self.huge_bthd_off;
        let huge_nodes_addr = heap_address + self.huge_nodes_off;
        let huge_data_addr = heap_address + self.huge_data_off;

        // The reference C library does not read a heap ID's length field at a fixed
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

        // Build fractal heap header
        let mut frhp = Vec::with_capacity(DENSE_ATTR_FRHP_SIZE);
        frhp.extend_from_slice(b"FRHP");
        frhp.push(0); // version
        frhp.extend_from_slice(&heap_id_length.to_le_bytes());
        frhp.extend_from_slice(&0u16.to_le_bytes()); // io_filter_encoded_length
        frhp.push(0x02); // flags: bit 1 = checksum direct blocks
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
        frhp.extend_from_slice(&max_managed.to_le_bytes());
        write_length(&mut frhp, huge_count as u64, LENGTH_SIZE); // next_huge_object_id
        if huge_count == 0 {
            write_undef_offset(&mut frhp, OFFSET_SIZE); // btree_huge_objects_address
        } else {
            write_offset(&mut frhp, huge_bthd_addr, OFFSET_SIZE);
        }
        write_length(&mut frhp, managed_plan.free_space(), LENGTH_SIZE); // free_space_managed_blocks
        write_undef_offset(&mut frhp, OFFSET_SIZE); // free_space_mgr_addr
        write_length(&mut frhp, managed_plan.managed_space(), LENGTH_SIZE); // managed_space_in_heap
        write_length(&mut frhp, managed_plan.allocated_space(), LENGTH_SIZE); // allocated_managed_space
        write_length(&mut frhp, managed_plan.allocation_iterator(), LENGTH_SIZE); // dblock_alloc_iter
        // Managed and huge objects are counted separately; an attribute is in exactly
        // one of the two.
        let managed_count = (serialized.len() - huge_count) as u64;
        write_length(&mut frhp, managed_count, LENGTH_SIZE); // managed_objects_count
        write_length(&mut frhp, *huge_total, LENGTH_SIZE); // huge_objects_size
        write_length(&mut frhp, huge_count as u64, LENGTH_SIZE); // huge_objects_count
        write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_size
        write_length(&mut frhp, 0, LENGTH_SIZE); // tiny_objects_count
        frhp.extend_from_slice(&fractal_heap_write::TABLE_WIDTH.to_le_bytes()); // table_width
        write_length(
            &mut frhp,
            fractal_heap_write::STARTING_BLOCK_SIZE,
            LENGTH_SIZE,
        );
        write_length(
            &mut frhp,
            fractal_heap_write::MAX_DIRECT_BLOCK_SIZE,
            LENGTH_SIZE,
        ); // max_direct_block_size
        frhp.extend_from_slice(&max_heap_size.to_le_bytes());
        frhp.extend_from_slice(&fractal_heap_write::START_ROOT_ROWS.to_le_bytes()); // start_root_rows
        write_offset(
            &mut frhp,
            managed_plan.root_address(managed_addr),
            OFFSET_SIZE,
        );
        // Zero when the root is a direct block, which is how a reader tells which of
        // the two the address above points at.
        frhp.extend_from_slice(&managed_plan.root_rows().to_le_bytes());
        let frhp_checksum = crate::checksum::jenkins_lookup3(&frhp);
        frhp.extend_from_slice(&frhp_checksum.to_le_bytes());
        debug_assert_eq!(frhp.len(), DENSE_ATTR_FRHP_SIZE);

        // Heap IDs. A managed object's carries the offset the plan gave it; a huge
        // object's carries a B-tree key instead, since its bytes sit outside the
        // managed blocks entirely.
        let mut heap_ids: Vec<Vec<u8>> = Vec::with_capacity(serialized.len());
        // (huge object ID, address, length) for the huge-objects B-tree, in ID order.
        let mut huge_records: Vec<(u64, u64, u64)> = Vec::with_capacity(huge_count);
        // The managed objects' bytes in attribute order, which is the order the
        // plan assigned their heap offsets in.
        let mut managed: Vec<&[u8]> = Vec::with_capacity(serialized.len() - huge_count);
        let mut next_huge_addr = huge_data_addr;
        for (s, huge_id) in serialized.iter().zip(huge_id_of) {
            match huge_id {
                Some(id) => {
                    huge_records.push((*id, next_huge_addr, s.len() as u64));
                    next_huge_addr += s.len() as u64;
                    heap_ids.push(encode_huge_id(*id, heap_id_length));
                }
                None => {
                    heap_ids.push(encode_managed_id(
                        managed_plan.heap_offset(managed.len()),
                        s.len() as u64,
                        max_heap_size,
                        heap_id_length,
                    ));
                    managed.push(s.as_slice());
                }
            }
        }

        let managed_blocks = managed_plan.serialize(&managed, managed_addr, frhp_addr);
        debug_assert_eq!(managed_blocks.len() as u64, managed_plan.region_size());

        // Build B-tree v2 type 8 records (17 bytes each), in the order the plan
        // sorted them into — the order the index is searched in.
        let record_size: u16 = heap_id_length + 1 + 4 + 4;
        debug_assert_eq!(record_size, DENSE_ATTR_BTREE_RECORD);
        let mut name_records = Vec::with_capacity(serialized.len() * record_size as usize);
        for &(hash, i) in order {
            name_records.extend_from_slice(&heap_ids[i as usize]);
            name_records.push(0); // msg_flags
            name_records.extend_from_slice(&creation.index_of(i).to_le_bytes()); // creation_order
            name_records.extend_from_slice(&hash.to_le_bytes()); // hash
        }

        let bthd_addr = btree_addr;
        let name_tree =
            name_plan.serialize(&name_records, name_nodes_addr, OFFSET_SIZE, LENGTH_SIZE);

        let mut blob = Vec::with_capacity(self.total_len.to_usize().unwrap_or(0));
        blob.extend_from_slice(&frhp);
        blob.extend_from_slice(&managed_blocks);
        debug_assert_eq!(blob.len() as u64, bthd_addr - heap_address);
        blob.extend_from_slice(&name_tree.header);
        blob.extend_from_slice(&name_tree.nodes);

        if let Some(corder_plan) = corder_plan {
            // The same heap IDs the name index carries, ordered by creation
            // index and without the name hash: this is the index `H5Aiterate2`
            // walks for `H5_INDEX_CRT_ORDER`.
            let mut corder_records =
                Vec::with_capacity(corder_order.len() * DENSE_ATTR_CORDER_BTREE_RECORD as usize);
            for &i in corder_order {
                corder_records.extend_from_slice(&heap_ids[i as usize]);
                corder_records.push(0); // msg_flags
                corder_records.extend_from_slice(&creation.index_of(i).to_le_bytes());
            }
            let corder_tree =
                corder_plan.serialize(&corder_records, corder_nodes_addr, OFFSET_SIZE, LENGTH_SIZE);
            debug_assert_eq!(blob.len() as u64, corder_bthd_addr - heap_address);
            blob.extend_from_slice(&corder_tree.header);
            blob.extend_from_slice(&corder_tree.nodes);
        }

        if let Some(huge_plan) = huge_plan {
            // Records are already in ascending ID order, which is the order the
            // B-tree is searched in.
            let mut huge_bytes =
                Vec::with_capacity(huge_records.len() * DENSE_ATTR_HUGE_BTREE_RECORD as usize);
            for (id, addr, len) in &huge_records {
                write_offset(&mut huge_bytes, *addr, OFFSET_SIZE);
                write_length(&mut huge_bytes, *len, LENGTH_SIZE);
                write_length(&mut huge_bytes, *id, LENGTH_SIZE);
            }
            let huge_tree =
                huge_plan.serialize(&huge_bytes, huge_nodes_addr, OFFSET_SIZE, LENGTH_SIZE);

            debug_assert_eq!(blob.len() as u64, huge_bthd_addr - heap_address);
            blob.extend_from_slice(&huge_tree.header);
            blob.extend_from_slice(&huge_tree.nodes);
            debug_assert_eq!(blob.len() as u64, huge_data_addr - heap_address);
            for (s, huge_id) in serialized.iter().zip(huge_id_of) {
                if huge_id.is_some() {
                    blob.extend_from_slice(s);
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

        DenseAttrBlob {
            attr_info_message: self.attr_info_message(frhp_addr),
            blob,
        }
    }
}

/// The heap address pass 1 of a file build asks a [`DenseAttrPlan`] for its
/// Attribute Info message at, before the heap's real address is known. The
/// message's *length* is what that pass needs, and that does not depend on the
/// address; pass 2 emits the real message at the address it reserves.
const DUMMY_DENSE_BASE: u64 = 0;

/// Build dense attribute storage for a set of attributes, at a known address.
///
/// A caller that has to reserve the heap's span before its bytes exist wants
/// [`dense_attrs_plan`] and [`DenseAttrPlan::blob_len`] instead; this is the
/// one-shot form for callers that already know where the heap goes.
pub(crate) fn build_dense_attrs(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
    heap_address: u64,
) -> DenseAttrBlob {
    dense_attrs_plan(attrs, creation).build(heap_address)
}

/// Bytes the reference C library uses to encode a limit of `value`
/// (`H5VM_limit_enc_size`): the width of the smallest field that can hold it.
fn encoded_size_width(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).div_ceil(8).max(1)
}

/// A heap ID for a "huge" object: type 1 in bits 4-5 of the first byte, then the
/// huge object ID little-endian across the rest.
///
/// The ID is a B-tree key rather than an address because this heap's IDs are too
/// narrow to hold an address and a length inline — `huge_ids_direct` in
/// `fractal_heap` recomputes that same choice on the way back in, so the two must
/// agree on the ID width.
fn encode_huge_id(huge_id: u64, id_length: u16) -> Vec<u8> {
    let payload_len = (id_length as usize) - 1;
    debug_assert!(
        payload_len >= 8 || huge_id < (1u64 << (payload_len * 8)),
        "huge object ID overflows the heap ID payload"
    );
    let mut id = vec![0u8; id_length as usize];
    id[0] = 0x10; // type = 1 (huge)
    for i in 0..payload_len.min(8) {
        id[1 + i] = ((huge_id >> (i * 8)) & 0xFF) as u8;
    }
    id
}

fn encode_managed_id(offset: u64, length: u64, max_heap_size: u16, id_length: u16) -> Vec<u8> {
    // `length << max_heap_size` must not overflow, and the offset must not run
    // into the length's bits. Both hold for every set `dense_attrs_check` admits;
    // asserted so a change to either constant cannot silently break the packing.
    debug_assert!(length <= DENSE_ATTR_MAX_MANAGED_OBJECT as u64);
    debug_assert_eq!(
        offset >> max_heap_size,
        0,
        "heap offset overflows its field"
    );
    let mut id = vec![0u8; id_length as usize];
    id[0] = 0x00; // type = 0 (managed)
    let combined = offset | (length << max_heap_size);
    let payload_len = (id_length as usize) - 1;
    for i in 0..payload_len.min(8) {
        id[1 + i] = ((combined >> (i * 8)) & 0xFF) as u8;
    }
    id
}

/// The Attribute Info (0x0015) message a version 2 object header needs when its
/// attributes are stored *compactly*, as inline Attribute messages.
///
/// On a version 2 header the reference C library does not count attribute
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
    .serialize(OFFSET_SIZE)
}

pub(crate) fn write_offset(buf: &mut Vec<u8>, val: u64, offset_size: u8) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "each arm narrows to offset_size, the on-disk address width chosen for this file"
    )]
    match offset_size {
        2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&val.to_le_bytes()),
        _ => {}
    }
}

fn write_length(buf: &mut Vec<u8>, val: u64, length_size: u8) {
    write_offset(buf, val, length_size);
}

pub(crate) fn write_undef_offset(buf: &mut Vec<u8>, offset_size: u8) {
    for _ in 0..offset_size {
        buf.push(0xFF);
    }
}

// ---- FileWriter ----

/// The main file creation API.
pub struct FileWriter {
    root_datasets: Vec<DatasetBuilder>,
    root_attrs: Vec<(String, AttrSpec)>,
    root_committed: Vec<CommittedDatatype>,
    groups: Vec<FinishedGroup>,
    userblock_size: u64,
    /// Bytes to emit at the head of the userblock region, from
    /// [`with_userblock_content`](FileWriter::with_userblock_content). The rest
    /// of the region is zero-filled. Empty by default, which reproduces the
    /// all-zero userblock this writer emitted before.
    userblock_content: Vec<u8>,
    /// Requested library-version bounds (low, high), validated in `finish`.
    /// `None` means no constraint (any output the writer produces is accepted).
    libver_bounds: Option<(LibVer, LibVer)>,
    /// File-space strategy `(strategy, persist, threshold)` from
    /// `with_file_space_strategy`. `None` leaves the file-space defaults.
    file_space_strategy: Option<(FileSpaceStrategy, bool, u64)>,
    /// File-space page size from `with_file_space_page_size`.
    file_space_page_size: Option<u64>,
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
            root_attrs: Vec::new(),
            root_committed: Vec::new(),
            groups: Vec::new(),
            userblock_size: 0,
            userblock_content: Vec::new(),
            libver_bounds: None,
            file_space_strategy: None,
            file_space_page_size: None,
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
        self.file_space_strategy = properties.file_space_strategy();
        self.file_space_page_size = properties.file_space_page_size();
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

    /// Set the userblock size in bytes: zero (no userblock), or a power of two of
    /// at least 512. Any other size is refused by [`finish`](Self::finish) with
    /// [`FormatError::InvalidUserblockSize`].
    ///
    /// The region will be filled with zeros; the caller can write into the
    /// returned bytes at `[0..userblock_size]`, or supply them up front with
    /// [`with_userblock_content`](Self::with_userblock_content).
    pub fn with_userblock(&mut self, size: u64) -> &mut Self {
        self.userblock_size = size;
        self
    }

    /// Set the bytes that occupy the head of the userblock region, so the writer
    /// *emits* them rather than the caller patching them in afterwards. The rest
    /// of the region is zero-filled, and `content` longer than the userblock is
    /// refused by [`finish`](Self::finish).
    ///
    /// The userblock is the first thing written in address order, so this is what
    /// lets a format that wraps HDF5 in a header — MATLAB v7.3, say — be produced
    /// by the non-seekable streaming path without a second pass over the file.
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
    ) -> Result<(), FormatError> {
        if ub == 0 {
            return Ok(());
        }
        sink.put(content)?;
        sink.put_zeros(ub - content.len())
    }

    /// Set the file-space management strategy, mirroring
    /// `H5Pset_file_space_strategy`. The choice is recorded in the file's
    /// superblock extension so other tools (and a later reopen) see it.
    ///
    /// `persist` requests that freed space be tracked on disk across closes. A
    /// freshly built file has no free space to track, so this records the persist
    /// intent (matching what the C library writes for a brand-new persisted
    /// file); a later [`File::open_rw`](crate::File::open_rw) that frees space writes
    /// the on-disk free-space-manager blocks. `threshold` is the smallest
    /// free-space section size the managers track.
    pub fn with_file_space_strategy(
        &mut self,
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
    ) -> &mut Self {
        self.file_space_strategy = Some((strategy, persist, threshold));
        self
    }

    /// Set the file-space page size, mirroring `H5Pset_file_space_page_size`.
    /// Recorded in the superblock extension; meaningful for the paged strategy.
    pub fn with_file_space_page_size(&mut self, page_size: u64) -> &mut Self {
        self.file_space_page_size = Some(page_size);
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
        self.file_space_info().is_some()
            || any_chunked(&self.root_datasets)
            || self.groups.iter().any(group_needs)
    }

    /// Reject file-space settings this writer cannot reproduce yet.
    /// The File Space Info message to write, if any file-space option was set.
    ///
    /// A freshly built file has no free space, so `persist = true` emits the
    /// persisting-but-empty form (persist flag set, all managers undefined, no
    /// FSM blocks); a later [`File::open_rw`](crate::File::open_rw) that frees space
    /// fills in the on-disk managers. `persist = false` emits the non-persistent
    /// form.
    fn file_space_info(&self) -> Option<FileSpaceInfo> {
        if self.file_space_strategy.is_none() && self.file_space_page_size.is_none() {
            return None;
        }
        let (strategy, persist, threshold) = self.file_space_strategy.unwrap_or((
            FileSpaceStrategy::FsmAggr,
            false,
            DEFAULT_THRESHOLD,
        ));
        let page_size = self.file_space_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
        Some(if persist {
            FileSpaceInfo::persistent_empty(strategy, threshold, page_size)
        } else {
            FileSpaceInfo::non_persistent(strategy, threshold, page_size)
        })
    }

    /// The superblock-extension object header bytes carrying the File Space Info
    /// message, if file-space was configured.
    fn file_space_extension_oh(&self) -> Result<Option<Vec<u8>>, FormatError> {
        self.file_space_info()
            .map(|info| {
                let mut oh = ObjectHeaderWriter::new();
                // Message flags 0x14 match what the reference C library writes for
                // this message (do-not-share + mark-if-unknown); no must-understand
                // bit, so older readers still open the file.
                oh.add_message_with_flags(MessageType::FileSpaceInfo, info.serialize(), 0x14);
                oh.serialize()
            })
            .transpose()
    }

    pub fn create_group(&mut self, name: &str) -> GroupBuilder {
        GroupBuilder::new(name)
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

    /// Attach a root-group attribute whose datatype is the committed one at
    /// `path`. See [`DatasetBuilder::set_attr_committed`].
    pub fn set_root_attr_committed(&mut self, name: &str, value: AttrValue, path: &str) {
        self.root_attrs.push((
            name.to_string(),
            crate::type_builders::committed_attr_spec(name, &value, path),
        ));
    }

    pub fn set_root_attr(&mut self, name: &str, value: AttrValue) {
        self.root_attrs
            .push((name.to_string(), AttrSpec::Value(value)));
    }

    /// Attach an already-encoded attribute message to the root group, written
    /// exactly as given.
    ///
    /// See [`AttrSpec::Verbatim`] for what this preserves that `set_root_attr`
    /// cannot, and for the datatypes it must not be used with.
    pub(crate) fn set_root_attr_verbatim(&mut self, message: crate::attribute::AttributeMessage) {
        self.root_attrs
            .push((message.name.clone(), AttrSpec::Verbatim(message)));
    }

    /// Attach a variable-length string attribute to the root group with the given
    /// datatype and dataspace, staging `strings` into a heap of this file's own.
    /// See [`AttrSpec::VerbatimVarLen`].
    pub(crate) fn set_root_attr_var_len_verbatim(
        &mut self,
        mut message: crate::attribute::AttributeMessage,
        strings: Vec<String>,
    ) {
        message.raw_data = crate::type_builders::vl_string_reference_bytes(&strings);
        self.root_attrs.push((
            message.name.clone(),
            AttrSpec::VerbatimVarLen { message, strings },
        ));
    }

    pub fn finish(self) -> Result<Vec<u8>, FormatError> {
        let mut buf = Vec::new();
        self.finish_to_sink(&mut buf)?;
        Ok(buf)
    }

    /// Assemble the file and write it to `sink` in ascending-address order.
    /// Backs both the buffered [`finish`](Self::finish) (a `Vec<u8>` sink) and
    /// the streaming `FileBuilder::finish_to` (an `io::Write` sink), so the two
    /// produce byte-identical files. A streamed dataset's chunk bytes are pulled
    /// from its provider one chunk at a time here, never all held at once.
    pub(crate) fn finish_to_sink<S: ByteSink>(self, sink: &mut S) -> Result<(), FormatError> {
        let libver = self.resolve_libver()?;

        // File-space settings are recorded in a File Space Info message, which
        // arrived with HDF5 1.10. Refused beside the bounds themselves rather
        // than where the message is emitted, so the answer does not depend on
        // how far into the layout the writer got.
        //
        // Asked of `file_space_info` rather than of the strategy field, because
        // that is the one function that decides whether the message is written:
        // a page size with no strategy emits one too, and testing the strategy
        // alone let exactly that case through.
        if self.file_space_info().is_some() && libver < LibVer::V110 {
            return Err(FormatError::LibverTooOldForContent {
                content: "a file-space setting",
                needs: LibVer::V110.name(),
                writing: libver.name(),
            });
        }

        // The userblock's size is what the superblock's base address is, and the
        // format defines it as zero or a power of two of at least 512 bytes. A
        // reader scans for the signature at 0, 512, 1024, … doubling, so any
        // other size hides the superblock somewhere it will never look: the
        // writer would emit a file that nothing, including this crate, can open.
        // Refused here rather than at `with_userblock` so a caller who overwrites
        // an earlier size is judged on the one that actually applies.
        if self.userblock_size != 0
            && (self.userblock_size < 512 || !self.userblock_size.is_power_of_two())
        {
            return Err(FormatError::InvalidUserblockSize(self.userblock_size));
        }

        // Checked before any layout work: content that overruns its region would
        // otherwise displace the superblock, and the failure would surface as an
        // unopenable file rather than as the caller's mistake.
        if self.userblock_content.len() as u64 > self.userblock_size {
            return Err(FormatError::UserblockContentTooLarge {
                content: self.userblock_content.len() as u64,
                userblock: self.userblock_size,
            });
        }

        // Genuine paged allocation: page-align every allocation and, when
        // persisting, emit per-page-type free-space managers. Gated entirely on
        // the Page strategy so every other strategy keeps its exact byte layout.
        let (paged, persist_paged, page_size, fs_threshold) = match self.file_space_strategy {
            Some((FileSpaceStrategy::Page, persist, threshold)) => {
                let ps = self.file_space_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
                if ps < 512 || !ps.is_power_of_two() {
                    return Err(FormatError::InvalidFileSpacePageSize(ps));
                }
                // File-space pages are measured from the file base; the layout
                // below is base-relative, so base-relative boundaries coincide
                // with absolute ones only when the userblock is a whole number of
                // pages (zero trivially qualifies).
                if self.userblock_size % ps != 0 {
                    return Err(FormatError::UserblockNotPageAligned(
                        self.userblock_size,
                        ps,
                    ));
                }
                (true, persist, ps, threshold)
            }
            _ => (false, false, 0, DEFAULT_THRESHOLD),
        };

        // The superblock-extension header (carrying a File Space Info message)
        // is independent of the file layout, so build it up front and place it
        // after all other content below.
        let ext_oh = self.file_space_extension_oh()?;
        // A persisting *non-paged* file's placeholder File Space Info message (built
        // just above) records `eoa_pre_fsm` = UNDEF, because a fresh file has no
        // free-space-manager blocks. libhdf5 requires `fs_persist => eoa_fsm_fsalloc
        // != UNDEF`, and an assertion-enabled build aborts on the sentinel
        // (H5Fsuper.c), so the non-paged tail below rewrites the message with a real
        // end-of-allocation once the layout is known (issue #178). Capture the
        // parameters here, while `self` is intact. (The paged path has its own
        // manager-aware rewrite; a `Page` file never reaches the non-paged tail.)
        let nonpaged_persist: Option<(FileSpaceStrategy, u64, u64)> = match self.file_space_strategy
        {
            Some((strategy, true, threshold)) if strategy != FileSpaceStrategy::Page => Some((
                strategy,
                threshold,
                self.file_space_page_size.unwrap_or(DEFAULT_PAGE_SIZE),
            )),
            _ => None,
        };
        struct DsFlat {
            name: String,
            dt: Datatype,
            /// Where `dt` is written: in this dataset's header, or in a committed
            /// datatype object it names.
            dt_location: DatatypeLocation,
            ds: Dataspace,
            raw: Vec<u8>,
            attrs: Vec<AttributeMessage>,
            chunk_options: ChunkOptions,
            maxshape: Option<Vec<u64>>,
            /// Repack's verbatim chunk payload, when this dataset's chunks are
            /// copied compressed-as-is rather than encoded from `raw`.
            raw_chunks: Option<crate::type_builders::RawChunkPayload>,
            reference_targets: Option<Vec<crate::type_builders::ObjectRefPatch>>,
            /// Staged global heap collections + patch mask for a VL-string
            /// dataset, whose element references in `raw` need their heap
            /// addresses patched once the post-data cursor is known.
            vl_string_staging: Option<VlStringStaging>,
            /// A user-defined fill value, encoded in the dataset's datatype, or
            /// `None` for the library default. Validated against the datatype
            /// element size in `flatten_dataset`.
            fill: Option<Vec<u8>>,
            /// When set, this contiguous dataset's element bytes are produced at
            /// write time rather than held in `raw`, which stays empty.
            produced: Option<crate::type_builders::ProducedPayload>,
            /// Whether this dataset allocates storage at all. An unallocated one
            /// declares its shape and stores nothing. `raw_chunks` and `produced`
            /// also leave `raw` empty over a non-empty dataspace, but each still
            /// has a data region and fills it during emission; this one has no
            /// region to fill.
            allocation: StorageAllocation,
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
                match &self.produced {
                    Some(p) => p.total_bytes,
                    None => self.raw.len() as u64,
                }
            }

            /// Move this dataset's contiguous data region out for the assembly
            /// pass. A produced region leaves its provider behind in `self`, the
            /// way a verbatim chunked one does: the emitter reaches for it there,
            /// and the region is described twice (sizing, then placement).
            fn take_contiguous(&mut self) -> DsData {
                match &self.produced {
                    Some(p) => DsData::Produced {
                        total_bytes: p.total_bytes,
                        block_bytes: p.block_bytes,
                    },
                    None => DsData::InMemory(core::mem::take(&mut self.raw)),
                }
            }
        }

        /// One dataset's data region for the assembly pass: either materialized
        /// in memory, or a plan whose chunk bytes are streamed from a provider.
        enum DsData {
            InMemory(Vec<u8>),
            /// A verbatim chunked dataset streamed one chunk at a time; the
            /// provider lives in the matching `DsFlat.raw_chunks` (`Lazy`).
            Streamed(VerbatimPlan),
            /// A contiguous dataset whose element bytes are produced at write
            /// time, block by block; the producer lives in the matching
            /// `DsFlat.produced`. The region is a plain run of `total_bytes`, so
            /// it is laid out exactly as if the bytes had been handed over.
            Produced {
                /// Bytes the whole region occupies.
                total_bytes: u64,
                /// Bytes per block, the last one excepted.
                block_bytes: u64,
            },
        }
        impl DsData {
            fn len(&self) -> u64 {
                match self {
                    DsData::InMemory(v) => v.len() as u64,
                    DsData::Streamed(plan) => plan.total_len,
                    DsData::Produced { total_bytes, .. } => *total_bytes,
                }
            }
        }

        /// Emit one dataset's data region: in-memory bytes directly, or a
        /// streamed verbatim plan pulled from its raw-chunk provider one chunk at
        /// a time (so a streamed dataset's bytes never all reside in memory).
        fn emit_ds_data<Sk: ByteSink>(
            sink: &mut Sk,
            data: &DsData,
            raw_chunks: Option<&crate::type_builders::RawChunkPayload>,
            produced: Option<&crate::type_builders::ProducedPayload>,
        ) -> Result<(), FormatError> {
            match data {
                DsData::InMemory(bytes) => sink.put(bytes),
                DsData::Streamed(plan) => {
                    let provider = raw_chunks
                        .expect("a streamed data region implies a raw-chunk payload")
                        .provider
                        .0
                        .as_ref();
                    emit_chunked_data_verbatim(sink, plan, provider)
                }
                DsData::Produced {
                    total_bytes,
                    block_bytes,
                } => {
                    let payload =
                        produced.expect("a produced data region implies a produced payload");
                    emit_produced_data(
                        sink,
                        payload.provider.0.as_ref(),
                        *total_bytes,
                        *block_bytes,
                    )
                }
            }
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
        ) -> Result<(), FormatError> {
            // A zero-length block would never advance the cursor, and the loop
            // below would ask for block 0 forever. Callers construct the payload
            // from a checked block size, so this is a construction invariant —
            // but a hang is a far worse failure than a wrong file, and it costs
            // one comparison to make it an error instead.
            if block_bytes == 0 && total_bytes > 0 {
                return Err(FormatError::SerializationError(
                    "a produced dataset declared a zero-length block".into(),
                ));
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
                    ));
                }
                sink.put(&block)?;
                written += expected;
                index += 1;
            }
            Ok(())
        }

        /// Emit one variable-length dataset's or attribute's global heap
        /// collections back to back, in the order `place_collections` assigned
        /// their addresses.
        fn emit_collections<Sk: ByteSink>(
            sink: &mut Sk,
            collections: &[Vec<u8>],
        ) -> Result<(), FormatError> {
            for collection in collections {
                sink.put(collection)?;
            }
            Ok(())
        }

        /// A built chunked dataset's layout/pipeline messages plus its data
        /// region (materialized for the encode and eager-verbatim paths, planned
        /// for the streamed verbatim path).
        struct ChunkedBuilt {
            layout_message: Vec<u8>,
            pipeline_message: Option<Vec<u8>>,
            data: DsData,
        }

        /// What the object-header sizing pass needs from a chunked dataset: the
        /// two messages that go in the header, and how long the data region will
        /// be — but not the region.
        ///
        /// [`build_chunked`] materializes it, which for the sizing pass meant
        /// allocating and dropping a whole second copy of every chunked dataset
        /// in the file (issue #228). Both paths take their length from the same
        /// place the emitter does, so the header sized here cannot disagree with
        /// the bytes written later.
        fn measure_chunked(
            d: &DsFlat,
            data_address: u64,
            chunk_set: Option<&CompressedChunkSet>,
        ) -> Result<ChunkedMeasure, FormatError> {
            if let Some(rc) = &d.raw_chunks {
                let VerbatimLayout {
                    plan,
                    layout_message,
                    pipeline_message,
                } = plan_chunked_data_verbatim(
                    &rc.meta,
                    &d.ds.dimensions,
                    &rc.chunk_dims,
                    rc.element_size,
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
                measure_chunked_at(set, data_address)
            }
        }

        /// Build the chunked data + layout/pipeline messages for one chunked
        /// dataset at `data_address`, dispatching to the verbatim path when the
        /// dataset carries a raw-chunk payload, else the normal encode path. The
        /// layout is computed from chunk *sizes* alone, so it is identical
        /// whether the chunks are in memory or streamed.
        ///
        /// [`measure_chunked`] dispatches the same two ways and must reach the
        /// same length and the same layout message, since the sizing pass uses
        /// it and this produces what that pass sized. They were one function
        /// until sizing stopped building a region to measure it (issue #228);
        /// what holds them together now is
        /// `measuring_a_chunked_region_agrees_with_assembling_it` on the encode
        /// side and `plan_chunked_data_verbatim` being shared on the other.
        fn build_chunked(
            d: &DsFlat,
            data_address: u64,
            chunk_set: Option<&CompressedChunkSet>,
        ) -> Result<ChunkedBuilt, FormatError> {
            if let Some(rc) = &d.raw_chunks {
                // Verbatim chunks are always streamed: the layout is planned from
                // chunk sizes alone, and the bytes are pulled from the provider in
                // the assembly loop (buffered `finish` and streaming `finish_to`
                // share that one emitter, so their output is byte-identical).
                let VerbatimLayout {
                    plan,
                    layout_message,
                    pipeline_message,
                } = plan_chunked_data_verbatim(
                    &rc.meta,
                    &d.ds.dimensions,
                    &rc.chunk_dims,
                    rc.element_size,
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
                // Encode path: the chunks were compressed once up front; just lay
                // the cached set out at this address (no recompression).
                let set = chunk_set
                    .expect("an encode-path chunked dataset must have a precomputed chunk set");
                let result = assemble_chunked_at(set, data_address)?;
                Ok(ChunkedBuilt {
                    layout_message: result.layout_message,
                    pipeline_message: result.pipeline_message,
                    data: DsData::InMemory(result.data_bytes),
                })
            }
        }
        struct GrpFlat {
            name: String,
            attrs: Vec<AttributeMessage>,
            ds_indices: Vec<usize>,
            sub_group_indices: Vec<usize>,
            committed_indices: Vec<usize>,
        }

        /// One committed datatype object, flattened out of the group tree.
        struct CtFlat {
            /// Link name in the owning group.
            name: String,
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
            let dt = db.datatype.ok_or(FormatError::DatasetMissingData)?;
            let shape = db.shape.ok_or(FormatError::DatasetMissingShape)?;
            // A verbatim-chunk dataset (repack) owns no flat `raw` element bytes;
            // its storage is the pre-compressed chunks in `raw_chunks`. Skip the
            // flat-data requirement and the shape/data-length check for it.
            let raw_chunks = db.raw_chunks;
            // A produced dataset owns no flat bytes either: its region is a run
            // of `total_bytes` filled by its provider during emission.
            let produced = db.produced;
            // And an unallocated one owns none because it stores nothing at all.
            // The two above leave `raw` empty as well; the difference is that
            // each of them still has a data region, filled during emission,
            // where this one has none (issue #293).
            let allocation = db.allocation;
            let unallocated = allocation == StorageAllocation::Unallocated;
            // Staging bytes for a dataset that stores nothing is a contradiction,
            // and it is the builder's to prevent rather than this function's to
            // resolve: whichever half were honored, the other would be dropped
            // without a word. Asserted rather than refused because
            // `with_unallocated_storage` is crate-internal and its one caller
            // stages nothing — a caller could not construct this, only a change
            // here could, and every test run is a debug build.
            debug_assert!(
                !(unallocated && (db.data.is_some() || produced.is_some() || raw_chunks.is_some())),
                "dataset {:?} declares unallocated storage and stages data for it",
                db.name,
            );
            // Allow empty data for zero-element datasets (e.g. shape [0, 0]).
            let is_empty = shape.contains(&0);
            let raw = if is_empty || unallocated || raw_chunks.is_some() || produced.is_some() {
                db.data.unwrap_or_default()
            } else {
                db.data.ok_or(FormatError::DatasetMissingData)?
            };
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
            let elem_size = dt.element_size_usize()?;
            let elem_bytes = elem_size.get() as u64;
            // The bytes this dataset's elements occupy end to end — the same
            // `num_elements * element_size` invariant the reader enforces (see
            // `data_read::read_raw_data_full`), which is what both readings of it
            // below are for. Multiply with checked arithmetic, saturating on
            // overflow: an absurd shape whose element count exceeds `u64` must
            // not panic a debug build in `Iterator::product` (nor silently wrap a
            // release build into a false match below). A saturated `u64::MAX` can
            // never equal a real `data.len()`, so it is correctly reported as a
            // mismatch.
            let extent = shape
                .iter()
                .copied()
                .try_fold(1u64, |acc, d| acc.checked_mul(d))
                .unwrap_or(u64::MAX)
                .saturating_mul(elem_bytes);
            // The bytes this dataset's data region occupies. A produced region is
            // measured by the same rule as a materialized one — its size is
            // declared rather than counted, and a declaration that disagrees with
            // the shape would produce a file the reader refuses.
            let region_len = produced
                .as_ref()
                .map_or(raw.len() as u64, |p| p.total_bytes);
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
            // Guard against a shape that disagrees with the supplied data:
            // without this, a mismatch (data for 3 elements with shape `[2, 2]`)
            // would produce a file that fails to read back. A zero-element shape
            // is held to that rule rather than exempted from it (issue #332): its
            // extent is 0, so bytes staged beside it disagree with it like any
            // other mismatch, and exempting them wrote a file the reference
            // library refuses to open — twelve bytes of elements at a defined
            // address under a dataspace declaring none, which our own reader
            // answers `Ok([])` for without ever looking at the region. A chunked
            // dataset of that shape had the milder version of the same fault: an
            // empty chunk grid, so the staged bytes went nowhere and the caller
            // was told `Ok`. The in-place writer refuses both already
            // (`edit::flatten_dataset`, which shares this function's name and
            // holds the same invariant with its own error type), and this is the
            // same refusal on the whole-file path.
            //
            // Staging *no* data for a zero-element shape stays legal, which is
            // what the `is_empty` bypass above is for: `region_len` and `extent`
            // are both 0 there, so this comparison passes. A verbatim-chunk
            // dataset has no flat region to compare, and an unallocated one
            // stages no bytes to disagree with its shape — the whole point is
            // that the two do not have to match.
            if !unallocated && raw_chunks.is_none() && region_len != extent {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "byte counts reported in a shape-mismatch error; display-only"
                )]
                return Err(FormatError::ShapeDataMismatch {
                    expected: extent as usize,
                    actual: region_len as usize,
                    element_size: elem_size,
                });
            }
            // Validate the chunk geometry up front for a chunked / filtered /
            // extensible dataset, so a malformed request (chunk dimensions of the
            // wrong rank, a zero chunk dimension, a bad maximum shape, or
            // chunking a scalar) is refused here instead of panicking in the
            // chunk splitter or producing an unreadable dataset.
            if db.chunk_options.is_chunked() || db.maxshape.is_some() {
                db.chunk_options
                    .validate_geometry(&shape, db.maxshape.as_deref())
                    .map_err(FormatError::InvalidChunkGeometry)?;
            }
            // Variable-length string element references live in the global heap.
            // For chunked/filtered/resizable storage the references sit inside
            // chunks that are split (and possibly compressed) before the rest of
            // the file is laid out, so their heap addresses cannot be patched in
            // afterwards. Such a dataset instead has its collections placed
            // *ahead* of everything else, at a fixed address known before any
            // chunk is encoded — see the early-placement block in
            // `finish_to_sink` that fills `early_gcol`. Nothing is refused here.
            let max_dimensions = db.maxshape.clone();
            let dspace = Dataspace {
                space_type: if shape.is_empty() {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                },
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "dataspace rank fits the 1-byte dimensionality field (HDF5 caps \
                              rank at 32)"
                )]
                rank: shape.len() as u8,
                dimensions: shape,
                max_dimensions,
            };
            let patches = collect_vl_patches(&db.attrs);
            let mut attrs = Vec::new();
            for (n, v) in &db.attrs {
                attrs.push(v.to_message(n));
            }
            #[cfg(feature = "provenance")]
            if let Some(ref prov) = db.provenance {
                let p = crate::provenance::Provenance {
                    creator: prov.creator.clone(),
                    timestamp: prov.timestamp.clone(),
                    source: prov.source.clone(),
                };
                attrs.extend(p.build_attrs(&raw));
            }
            // A user-defined fill value is one element wide, so its byte length
            // must equal the datatype's element size.
            if let Some(fill) = &db.fill {
                if fill.len() != elem_size.get() {
                    return Err(FormatError::FillValueSizeMismatch {
                        expected: elem_size.get(),
                        actual: fill.len(),
                    });
                }
            }
            let idx = all_ds.len();
            all_ds.push(DsFlat {
                name: db.name,
                dt,
                dt_location: db.datatype_location,
                ds: dspace,
                raw,
                attrs,
                chunk_options: db.chunk_options,
                maxshape: db.maxshape,
                raw_chunks,
                produced,
                reference_targets: db.reference_targets,
                vl_string_staging: db.vl_string_staging,
                fill: db.fill,
                allocation,
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
                        name: ct.name,
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
            let patches = collect_vl_patches(&g.attrs);
            let mut gattrs = Vec::new();
            for (n, v) in &g.attrs {
                gattrs.push(v.to_message(n));
            }
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
                name: g.name,
                attrs: gattrs,
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

        /// Assign consecutive addresses to `collections` starting at `*cursor`,
        /// advancing it past them, and return those addresses in order. The
        /// GCOL emission loops below walk the collections in this same order,
        /// so the addresses patched into the references are the ones the
        /// collections land at.
        fn place_collections(collections: &[Vec<u8>], cursor: &mut u64) -> Vec<u64> {
            collections
                .iter()
                .map(|c| {
                    let addr = *cursor;
                    *cursor += c.len() as u64;
                    addr
                })
                .collect()
        }

        fn collect_vl_patches(attrs_raw: &[(String, AttrSpec)]) -> Vec<VlPatch> {
            let mut patches = Vec::new();
            for (i, (_n, v)) in attrs_raw.iter().enumerate() {
                if let Some(strings) = v.var_len_strings() {
                    patches.push(VlPatch {
                        collections: build_global_heap_collections(strings),
                        attr_index: i,
                    });
                }
            }
            patches
        }

        let vl_root = collect_vl_patches(&self.root_attrs);

        let mut root_attrs: Vec<AttributeMessage> = Vec::new();
        for (n, v) in &self.root_attrs {
            root_attrs.push(v.to_message(n));
        }

        // ---- Committed datatypes: paths, reference counts, and agreement ----
        //
        // Done before any sizing, because a committed type's own header carries
        // its reference count and so changes length with it, and because a
        // dataset that names a type the file does not commit must be refused
        // before the layout is half built.
        let committed_by_path: HashMap<String, usize> = {
            fn walk(
                prefix: &str,
                gi: usize,
                groups: &[GrpFlat],
                committed: &[CtFlat],
                out: &mut HashMap<String, usize>,
            ) {
                for &ci in &groups[gi].committed_indices {
                    out.insert(format!("{prefix}/{}", committed[ci].name), ci);
                }
                for &sgi in &groups[gi].sub_group_indices {
                    walk(
                        &format!("{prefix}/{}", groups[sgi].name),
                        sgi,
                        groups,
                        committed,
                        out,
                    );
                }
            }
            let mut out: HashMap<String, usize> = root_committed_indices
                .iter()
                .map(|&ci| (committed[ci].name.clone(), ci))
                .collect();
            for &gi in &root_group_indices {
                walk(&groups[gi].name, gi, &groups, &committed, &mut out);
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
            by_path: &HashMap<String, usize>,
            location: &DatatypeLocation,
            dt: &Datatype,
            user: impl FnOnce() -> String,
        ) -> Result<(), FormatError> {
            let Some(path) = location.unresolved_path() else {
                return Ok(());
            };
            let Some(&ci) = by_path.get(path) else {
                return Err(FormatError::UnknownCommittedDatatype(path.to_string()));
            };
            if committed[ci].dt.serialize() != dt.serialize() {
                return Err(FormatError::CommittedDatatypeMismatch {
                    path: path.to_string(),
                    user: user(),
                });
            }
            committed[ci].references = committed[ci].references.saturating_add(1);
            Ok(())
        }

        for attr in &root_attrs {
            register_committed_use(
                &mut committed,
                &committed_by_path,
                &attr.datatype_location,
                &attr.datatype,
                || format!("root attribute {:?}", attr.name),
            )?;
        }
        for g in &groups {
            for attr in &g.attrs {
                register_committed_use(
                    &mut committed,
                    &committed_by_path,
                    &attr.datatype_location,
                    &attr.datatype,
                    || format!("attribute {:?} of group {:?}", attr.name, g.name),
                )?;
            }
        }
        for d in &all_ds {
            register_committed_use(
                &mut committed,
                &committed_by_path,
                &d.dt_location,
                &d.dt,
                || format!("dataset {:?}", d.name),
            )?;
            for attr in &d.attrs {
                register_committed_use(
                    &mut committed,
                    &committed_by_path,
                    &attr.datatype_location,
                    &attr.datatype,
                    || format!("attribute {:?} of dataset {:?}", attr.name, d.name),
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
            ds_addrs: &'a [u64],
            committed_addrs: &'a [u64],
            group_addrs: &'a [u64],
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
                    links.push(make_link(&self.all_ds[i].name, self.ds_addrs[i]));
                }
                for &ci in committed_indices {
                    links.push(make_link(
                        &self.committed[ci].name,
                        self.committed_addrs[ci],
                    ));
                }
                for &gi in sub_group_indices {
                    links.push(make_link(&self.groups[gi].name, self.group_addrs[gi]));
                }
                links
            }
        }

        // Addresses the sizing pass uses. A link's byte length does not depend on
        // the address it carries, so zeros give the real header size.
        let dummy_ds_addrs = vec![0u64; all_ds.len()];
        let dummy_ct_addrs = vec![0u64; committed.len()];
        let dummy_grp_addrs = vec![0u64; groups.len()];

        let root_dense = needs_dense_attrs(&root_attrs);
        let group_dense: Vec<bool> = groups.iter().map(|g| needs_dense_attrs(&g.attrs)).collect();
        let ds_dense: Vec<bool> = all_ds.iter().map(|d| needs_dense_attrs(&d.attrs)).collect();

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
                let size = a.serialize(LENGTH_SIZE).len();
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
            .map(|d| d.chunk_options.is_chunked() || d.maxshape.is_some() || d.raw_chunks.is_some())
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
            });
        }

        // A chunked/filtered/resizable variable-length dataset's element
        // references are split into chunks (and compressed) below, before the
        // file layout that would normally fix its global-heap addresses. Placing
        // those collections *first* — immediately after the superblock, at an
        // address that depends on nothing but the userblock — makes the addresses
        // known up front, so the references can be patched into `raw` before a
        // single chunk is encoded. Everything else shifts down by the collections'
        // total size, which is known here because staging serialized them
        // already.
        //
        // Only a chunked dataset takes this path. A contiguous one keeps the
        // established late placement (its element bytes stay patchable in place
        // until emission), so a file without a chunked VL dataset is laid out
        // byte-for-byte as before.
        let mut early_gcol: Vec<usize> = Vec::new();
        let early_gcol_size = {
            let mut cursor = SUPERBLOCK_SIZE as u64;
            for i in 0..all_ds.len() {
                // `raw_chunks` is excluded for the same reason the `chunk_sets`
                // loop below excludes it: a verbatim chunk payload is emitted
                // as-is and its `raw` is empty, so there is nothing to patch.
                if !is_chunked[i] || all_ds[i].raw_chunks.is_some() {
                    continue;
                }
                let Some(staging) = all_ds[i].vl_string_staging.take() else {
                    continue;
                };
                let addrs = place_collections(&staging.collections, &mut cursor);
                patch_vl_refs_masked(&mut all_ds[i].raw, &staging.patch_offsets, &addrs);
                all_ds[i].vl_string_staging = Some(staging);
                early_gcol.push(i);
            }
            (cursor - SUPERBLOCK_SIZE as u64).to_usize()?
        };

        // Compress each encode-path chunked dataset exactly once, up front. The
        // object-header sizing pass and the data-emit pass both need the chunk
        // layout, but only the embedded addresses differ between them — the
        // (expensive) compression does not. Caching the `CompressedChunkSet` here
        // lets both passes call the cheap `assemble_chunked_at` instead of
        // recompressing the whole dataset twice. Verbatim datasets carry their
        // bytes pre-compressed and are planned (not recompressed), so they get no
        // entry.
        let chunk_sets: Vec<Option<CompressedChunkSet>> = all_ds
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if is_chunked[i] && d.raw_chunks.is_none() {
                    let chunk_dims = d.chunk_options.resolve_chunk_dims(&d.ds.dimensions);
                    let ctx = crate::filters::ChunkContext::from_datatype(&chunk_dims, &d.dt)?;
                    // An allocated chunk holds the dataset's fill value
                    // wherever nothing was written, so the edge overhang of a
                    // partial chunk is filled rather than zeroed (issue #296).
                    let elem = crate::convert::nonzero_usize_from(ctx.element_size)?;
                    let fill = crate::fill_value::FillPattern::new(d.fill.as_deref(), elem);
                    Ok(Some(compress_chunks(
                        &d.raw,
                        &d.ds.dimensions,
                        ctx,
                        &d.chunk_options,
                        d.maxshape.as_deref(),
                        fill,
                        d.allocation,
                    )?))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<_, FormatError>>()?;

        /// Where each object's dense-attribute heap will be written and how many
        /// bytes it occupies, for the objects that have one.
        struct DenseSpans {
            root: Option<(u64, usize)>,
            groups: Vec<Option<(u64, usize)>>,
            datasets: Vec<Option<(u64, usize)>>,
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
                fn one(
                    attrs: &[AttributeMessage],
                    span: Option<(u64, usize)>,
                ) -> Result<Option<DenseAttrBlob>, FormatError> {
                    let Some((address, reserved)) = span else {
                        return Ok(None);
                    };
                    let blob = build_dense_attrs(attrs, DenseAttrCreationOrder::Untracked, address);
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
        for (gi, g) in groups.iter().enumerate() {
            let dummy_links =
                dummy_tables.links(&g.ds_indices, &g.committed_indices, &g.sub_group_indices);
            let (oh, dense_len) = if group_dense[gi] {
                // The plan answers both questions this pass asks — how long the
                // heap will be, and what its Attribute Info message costs the
                // header — without emitting the heap, which pass 2 does once at
                // the address reserved here.
                let plan = dense_attrs_plan(&g.attrs, DenseAttrCreationOrder::Untracked);
                (
                    build_group_oh(
                        &dummy_links,
                        &g.attrs,
                        Some(&plan.attr_info_message(DUMMY_DENSE_BASE)),
                    )?,
                    Some(plan.blob_len().to_usize()?),
                )
            } else {
                (build_group_oh(&dummy_links, &g.attrs, None)?, None)
            };
            group_oh_sizes.push(oh.len());
            group_dense_lens.push(dense_len);
        }

        let root_dummy_links = dummy_tables.links(
            &root_ds_indices,
            &root_committed_indices,
            &root_group_indices,
        );
        let (root_oh_size, root_dense_len) = if root_dense {
            let plan = dense_attrs_plan(&root_attrs, DenseAttrCreationOrder::Untracked);
            (
                build_group_oh(
                    &root_dummy_links,
                    &root_attrs,
                    Some(&plan.attr_info_message(DUMMY_DENSE_BASE)),
                )?
                .len(),
                Some(plan.blob_len().to_usize()?),
            )
        } else {
            (
                build_group_oh(&root_dummy_links, &root_attrs, None)?.len(),
                None,
            )
        };

        // Pass 1: compute dataset object-header sizes from a dummy layout. No
        // data bytes are materialized here — the object-header size depends only
        // on the layout/pipeline messages, and a chunk index's byte size is a
        // function of chunk count/size, not of the (dummy) data address. For a
        // streamed (lazy) dataset this touches no chunk bytes at all.
        let mut actual_ds_oh_sizes: Vec<usize> = Vec::with_capacity(all_ds.len());
        // Each dataset's data-region byte length, captured here (where chunked
        // data is already built once for OH sizing) so the paged layout can
        // classify small vs large allocations and size its free-space managers
        // without a second build. A chunked data length is base-address
        // independent, so the dummy-base build gives the true length.
        let mut ds_data_lens: Vec<u64> = Vec::with_capacity(all_ds.len());
        let mut ds_dense_lens: Vec<Option<usize>> = Vec::with_capacity(all_ds.len());
        let mut dummy_cursor = 0u64;
        for (i, d) in all_ds.iter().enumerate() {
            let dense_plan = if ds_dense[i] {
                Some(dense_attrs_plan(
                    &d.attrs,
                    DenseAttrCreationOrder::Untracked,
                ))
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
                let measured = measure_chunked(d, dummy_cursor, chunk_sets[i].as_ref())?;
                dummy_cursor += measured.data_len;
                ds_data_lens.push(measured.data_len);
                build_chunked_dataset_oh(
                    &d.dt,
                    &d.dt_location,
                    &d.ds,
                    &measured.layout_message,
                    measured.pipeline_message.as_deref(),
                    &d.attrs,
                    dense_attr_info.as_deref(),
                    d.fill.as_deref(),
                )?
            } else {
                ds_data_lens.push(d.contiguous_len());
                build_dataset_oh(
                    &d.dt,
                    &d.dt_location,
                    &d.ds,
                    0,
                    d.declared_contiguous_len,
                    &d.attrs,
                    dense_attr_info.as_deref(),
                    d.fill.as_deref(),
                    libver,
                )?
            };
            actual_ds_oh_sizes.push(oh.len());
        }

        // Pass 2: compute real addresses.
        // All addresses stored in the file are relative to base_address.
        // base_address = userblock_size. cursor2 tracks relative positions.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "userblock_size is a small power-of-two header size used as an in-memory \
                      buffer offset; it fits usize on every supported target"
        )]
        let ub = self.userblock_size as usize;
        // Early-placed VL collections (if any) occupy the span immediately after
        // the superblock, so the root object header — and everything after it —
        // starts past them. `early_gcol_size` is 0 for a file without a chunked
        // VL dataset, leaving every other file's addresses unchanged.
        let root_group_addr = (SUPERBLOCK_SIZE + early_gcol_size) as u64;
        let mut cursor2 = SUPERBLOCK_SIZE + early_gcol_size + root_oh_size;

        // Space set aside for a dense-attribute heap, and the heaps themselves
        // once the attribute bytes they copy are final. The two are separate
        // steps on purpose: a heap embeds a *copy* of each attribute's serialized
        // bytes, and a variable-length attribute's bytes hold global-heap
        // references that only get real addresses after every heap's span is
        // fixed — the collections sit past the heaps in the layout. Building the
        // bytes here would freeze the placeholder references into the heap, and a
        // reader that cannot resolve them drops the attribute entirely.
        let reserve = |cursor2: &mut usize, len: Option<usize>| {
            len.map(|len| {
                let addr = *cursor2 as u64;
                *cursor2 += len;
                (addr, len)
            })
        };

        let root_dense_span = reserve(&mut cursor2, root_dense_len);

        let mut group_dense_spans: Vec<Option<(u64, usize)>> = Vec::with_capacity(groups.len());
        let group_addrs2: Vec<u64> = group_oh_sizes
            .iter()
            .enumerate()
            .map(|(gi, &sz)| {
                let addr = cursor2 as u64;
                cursor2 += sz;
                group_dense_spans.push(reserve(&mut cursor2, group_dense_lens[gi]));
                addr
            })
            .collect();

        // Committed datatype headers. Placed with the other metadata so a paged
        // file keeps them in its metadata region, and before the datasets that
        // reference them only because the cursor has to run in some order —
        // nothing here depends on the relative placement.
        let committed_addrs: Vec<u64> = committed_oh
            .iter()
            .map(|oh| {
                let addr = cursor2 as u64;
                cursor2 += oh.len();
                addr
            })
            .collect();

        let mut ds_dense_spans: Vec<Option<(u64, usize)>> = Vec::with_capacity(all_ds.len());
        let ds_oh_addrs2: Vec<u64> = actual_ds_oh_sizes
            .iter()
            .enumerate()
            .map(|(i, &sz)| {
                let addr = cursor2 as u64;
                cursor2 += sz;
                ds_dense_spans.push(reserve(&mut cursor2, ds_dense_lens[i]));
                addr
            })
            .collect();

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
            let mut path_map = HashMap::<String, u64>::new();
            // The root group is referenceable under the empty path (repack maps a
            // reference to the source root group to "").
            path_map.insert(String::new(), root_group_addr);
            for &i in &root_ds_indices {
                path_map.insert(all_ds[i].name.clone(), ds_oh_addrs2[i]);
            }
            // A committed datatype is an object like any other, so an object
            // reference can point at one. Its path was already computed, above.
            for (path, &ci) in &committed_by_path {
                path_map.insert(path.clone(), committed_addrs[ci]);
            }
            for &gi in &root_group_indices {
                fn register_group(
                    prefix: &str,
                    gi: usize,
                    groups: &[GrpFlat],
                    ds_addrs: &[u64],
                    grp_addrs: &[u64],
                    all_ds: &[DsFlat],
                    map: &mut HashMap<String, u64>,
                ) {
                    map.insert(prefix.to_string(), grp_addrs[gi]);
                    for &di in &groups[gi].ds_indices {
                        map.insert(format!("{}/{}", prefix, all_ds[di].name), ds_addrs[di]);
                    }
                    for &sgi in &groups[gi].sub_group_indices {
                        register_group(
                            &format!("{}/{}", prefix, groups[sgi].name),
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
                    &groups[gi].name,
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
            for d in all_ds.iter_mut() {
                let Some(ref patches) = d.reference_targets else {
                    continue;
                };
                for patch in patches {
                    let addr = match &patch.target {
                        crate::type_builders::ObjectRefTarget::Path(path) => {
                            path_map.get(path).copied().unwrap_or(u64::MAX)
                        }
                        crate::type_builders::ObjectRefTarget::Raw(addr) => *addr,
                    };
                    write_reference_address(&mut d.raw, patch.byte_offset, addr);
                }
            }
        }

        // Resolve every committed-datatype reference, now that the objects have
        // addresses. From here on a `CommittedPath` cannot survive: the loops
        // below cover every dataset and every attribute in the file, which is
        // exactly the set `register_committed_use` walked to validate the paths.
        {
            let resolve = |location: &mut DatatypeLocation| {
                let ci = match location.unresolved_path() {
                    Some(path) => *committed_by_path.get(path).expect(
                        "every committed-datatype path was resolved against this same map \
                         before any header was sized",
                    ),
                    None => return,
                };
                *location = DatatypeLocation::Committed(committed_addrs[ci]);
            };
            for attr in &mut root_attrs {
                resolve(&mut attr.datatype_location);
            }
            for g in &mut groups {
                for attr in &mut g.attrs {
                    resolve(&mut attr.datatype_location);
                }
            }
            for d in &mut all_ds {
                resolve(&mut d.dt_location);
                for attr in &mut d.attrs {
                    resolve(&mut attr.datatype_location);
                }
            }
        }

        // Compute data layout (addresses + chunked data blobs) separately from OHs
        // so we can patch VL attrs before building OHs.
        struct DsLayout {
            data: DsData,
            data_addr: u64,
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
            ds_layouts: &[DsLayout],
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
                        d.fill.as_deref(),
                        libver,
                    )?
                };
                oh_bytes.push(oh);
            }
            Ok(oh_bytes)
        }

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
        if paged {
            let os = OFFSET_SIZE;
            let base = BaseAddress::new(ub as u64);
            let mut meta = cursor2 as u64; // metadata cursor, base-relative

            // (a) Global-heap collections live in the metadata region. Assign
            // their addresses and patch attribute VL references now; dataset
            // element references are patched after their data is built below.
            let gcol_start = meta;
            let mut gcol_cursor = meta;
            let mut elem_gcol: Vec<(usize, Vec<u64>)> = Vec::new();
            {
                for patch in &vl_root {
                    let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                    patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, &addrs);
                }
                for (gi, patches) in grp_vl.iter().enumerate() {
                    for patch in patches {
                        let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                        patch_vl_refs(&mut groups[gi].attrs[patch.attr_index].raw_data, &addrs);
                    }
                }
                for (di, patches) in ds_vl.iter().enumerate() {
                    for patch in patches {
                        let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                        patch_vl_refs(&mut all_ds[di].attrs[patch.attr_index].raw_data, &addrs);
                    }
                }
                for (i, d) in all_ds.iter().enumerate() {
                    // A chunked VL dataset's collections were placed and patched
                    // before its chunks were encoded; only the contiguous ones
                    // are still awaiting an address here.
                    if early_gcol.contains(&i) {
                        continue;
                    }
                    if let Some(staging) = &d.vl_string_staging {
                        elem_gcol
                            .push((i, place_collections(&staging.collections, &mut gcol_cursor)));
                    }
                }
            }
            let gcol_total_size = gcol_cursor - gcol_start;
            meta += gcol_total_size;

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
                } else if len < page_size {
                    small_indices.push(i);
                } else {
                    large_indices.push(i);
                }
            }
            let small_raw_total: u64 = small_indices.iter().map(|&i| ds_data_lens[i]).sum();
            let large_frag_sizes: Vec<u64> = large_indices
                .iter()
                .map(|&i| align_up(ds_data_lens[i], page_size) - ds_data_lens[i])
                .filter(|&f| f > 0)
                .collect();

            // (d) Which per-page-type managers are active, and place their
            // FSHD/FSSE blocks in the metadata region (persisting only). Block
            // lengths depend only on section counts (fixed field widths), so the
            // page tails are computed in a single forward pass with no iteration.
            let draw_active =
                small_raw_total > 0 && align_up(small_raw_total, page_size) != small_raw_total;
            let large_active = !large_frag_sizes.is_empty();
            let mut slots = [u64::MAX; NUM_FILE_FSM_MANAGERS];
            let mut super_fsm: Option<(u64, u64)> = None;
            let mut draw_fsm: Option<(u64, u64)> = None;
            let mut large_fsm: Option<(u64, u64)> = None;
            let super_block_len = fshd_len(os) + fsse_len(&[0], os);
            let draw_block_len = if draw_active {
                fshd_len(os) + fsse_len(&[0], os)
            } else {
                0
            };
            let large_block_len = if large_active {
                fshd_len(os) + fsse_len(&large_frag_sizes, os)
            } else {
                0
            };
            // SUPER tracks the metadata page tail. Placing its own block shifts
            // that tail, so only keep SUPER when a tail actually remains; in the
            // (astronomically rare) exact-fill case, drop it and leave the tiny
            // tail untracked. This decision is O(1), not a fixpoint.
            let super_active = if persist_paged {
                let with = meta_content_end + super_block_len + draw_block_len + large_block_len;
                align_up(with, page_size) > with
            } else {
                false
            };
            if persist_paged {
                if super_active {
                    let fshd_addr = meta;
                    meta += fshd_len(os);
                    let fsse_addr = meta;
                    meta += fsse_len(&[0], os);
                    slots[0] = fshd_addr;
                    super_fsm = Some((fshd_addr, fsse_addr));
                }
                if draw_active {
                    let fshd_addr = meta;
                    meta += fshd_len(os);
                    let fsse_addr = meta;
                    meta += fsse_len(&[0], os);
                    slots[2] = fshd_addr;
                    draw_fsm = Some((fshd_addr, fsse_addr));
                }
                if large_active {
                    let fshd_addr = meta;
                    meta += fshd_len(os);
                    let fsse_addr = meta;
                    meta += fsse_len(&large_frag_sizes, os);
                    slots[6] = fshd_addr;
                    large_fsm = Some((fshd_addr, fsse_addr));
                }
            }
            let meta_end = meta;

            // (e) The raw-data region starts on a fresh page boundary. The
            // metadata page tail is the SUPER section (when active).
            let raw_start = align_up(meta_end, page_size);
            let super_section = super_fsm.map(|_| FreeSection {
                addr: meta_end,
                size: raw_start - meta_end,
            });

            // (f) Build the raw data. Small blocks pack; the region is padded to
            // a page boundary (DRAW tail). Each large block starts a fresh page
            // run and its sub-page remainder becomes a generic-large section.
            let mut layouts: Vec<Option<DsLayout>> = (0..all_ds.len()).map(|_| None).collect();
            for &i in &empty_indices {
                let raw = core::mem::take(&mut all_ds[i].raw);
                layouts[i] = Some(DsLayout {
                    data: DsData::InMemory(raw),
                    data_addr: u64::MAX,
                    chunked_msgs: None,
                });
            }
            let mut c = raw_start;
            for &i in &small_indices {
                let base_addr = c;
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
            let draw_section = if draw_active {
                let padded = align_up(small_raw_end, page_size);
                c = padded;
                Some(FreeSection {
                    addr: small_raw_end,
                    size: padded - small_raw_end,
                })
            } else {
                if small_raw_total > 0 {
                    c = align_up(small_raw_end, page_size);
                }
                None
            };
            let mut large_sections: Vec<FreeSection> = Vec::new();
            for &i in &large_indices {
                c = align_up(c, page_size);
                let data_addr = c;
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
                let data_end = data_addr + built_len;
                let frag = align_up(data_end, page_size) - data_end;
                if frag > 0 {
                    large_sections.push(FreeSection {
                        addr: data_end,
                        size: frag,
                    });
                }
                c = align_up(data_end, page_size);
            }
            let eoa_rel = c; // already page-aligned
            let eof_addr2 = base.absolute(eoa_rel)?;
            let eoa_pre_fsm = eoa_rel;

            // (g) Now that the element bytes exist, patch dataset-element VL refs.
            for (i, gaddrs) in &elem_gcol {
                let staging = all_ds[*i]
                    .vl_string_staging
                    .as_ref()
                    .expect("elem_gcol only holds datasets with VL staging");
                let Some(DsLayout {
                    data: DsData::InMemory(bytes),
                    ..
                }) = layouts[*i].as_mut()
                else {
                    unreachable!(
                        "a staged VL-string dataset holds its element bytes in memory: the \
                         chunked path patches before encoding, and a produced region refuses \
                         to carry VL staging at all"
                    )
                };
                patch_vl_refs_masked(bytes, &staging.patch_offsets, gaddrs);
            }

            let ds_layouts: Vec<DsLayout> = layouts
                .into_iter()
                .map(|o| o.expect("every dataset placed"))
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
                let info = FileSpaceInfo::persistent_managers(
                    FileSpaceStrategy::Page,
                    fs_threshold,
                    page_size,
                    slots,
                    eoa_pre_fsm,
                );
                let mut oh = ObjectHeaderWriter::new();
                oh.add_message_with_flags(MessageType::FileSpaceInfo, info.serialize(), 0x14);
                oh.serialize()?
            } else {
                ext_oh
                    .clone()
                    .expect("a paged file always emits a File Space Info message")
            };
            debug_assert_eq!(real_ext_oh.len() as u64, ext_len);

            // (j) Serialize the free-space-manager blocks.
            let super_blocks = super_fsm.map(|(fshd_addr, fsse_addr)| {
                serialize_file_fsm(
                    &[super_section.expect("SUPER active implies a section")],
                    fshd_addr,
                    fsse_addr,
                    os,
                    SECT_CLASS_SMALL,
                )
            });
            let draw_blocks = draw_fsm.map(|(fshd_addr, fsse_addr)| {
                serialize_file_fsm(
                    &[draw_section.expect("DRAW active implies a section")],
                    fshd_addr,
                    fsse_addr,
                    os,
                    SECT_CLASS_SMALL,
                )
            });
            let large_blocks = large_fsm.map(|(fshd_addr, fsse_addr)| {
                serialize_file_fsm(&large_sections, fshd_addr, fsse_addr, os, SECT_CLASS_LARGE)
            });

            // (k) Emit, address-ascending, zero-filling every alignment gap.
            sink.reserve(eof_addr2.to_usize()?);
            Self::put_userblock(sink, ub, &self.userblock_content)?;
            let sb = Superblock {
                version: superblock_version(libver),
                offset_size: OFFSET_SIZE,
                offset_width: OFFSET_WIDTH,
                length_size: LENGTH_SIZE,
                length_width: LENGTH_WIDTH,
                base_address: base,
                eof_address: eof_addr2,
                root_group_address: root_group_addr,
                group_leaf_node_k: None,
                group_internal_node_k: None,
                indexed_storage_internal_node_k: None,
                free_space_address: None,
                driver_info_address: None,
                consistency_flags: 0,
                superblock_extension_address: Some(ext_addr),
                checksum: None,
            };
            sink.put(&sb.serialize())?;

            // Early-placed VL collections, at the addresses patched into the
            // chunked datasets' references before their chunks were encoded.
            for &i in &early_gcol {
                let staging = all_ds[i]
                    .vl_string_staging
                    .as_ref()
                    .expect("early_gcol only holds datasets with VL staging");
                emit_collections(sink, &staging.collections)?;
            }

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
            )?;
            debug_assert_eq!(root_oh.len(), root_oh_size);
            sink.put(&root_oh)?;
            if let Some(ref blob) = root_dense_blob {
                sink.put(&blob.blob)?;
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
                )?;
                debug_assert_eq!(oh.len(), group_oh_sizes[gi]);
                sink.put(&oh)?;
                if let Some(ref blob) = group_dense_blobs[gi] {
                    sink.put(&blob.blob)?;
                }
            }
            // Committed datatype OHs, at the addresses their references name.
            for oh in &committed_oh {
                sink.put(oh)?;
            }
            // Dataset OHs + dense blobs.
            for (i, oh) in ds_oh_bytes.iter().enumerate() {
                sink.put(oh)?;
                if let Some(ref dense) = ds_dense_blobs[i] {
                    sink.put(&dense.blob)?;
                }
            }
            // Global heap collections.
            for patch in &vl_root {
                emit_collections(sink, &patch.collections)?;
            }
            for patches in &grp_vl {
                for patch in patches {
                    emit_collections(sink, &patch.collections)?;
                }
            }
            for patches in &ds_vl {
                for patch in patches {
                    emit_collections(sink, &patch.collections)?;
                }
            }
            for (i, d) in all_ds.iter().enumerate() {
                if early_gcol.contains(&i) {
                    continue; // emitted right after the superblock
                }
                if let Some(staging) = &d.vl_string_staging {
                    emit_collections(sink, &staging.collections)?;
                }
            }
            debug_assert_eq!(sink.position(), base.get() + ext_addr);
            sink.put(&real_ext_oh)?;
            // Free-space-manager blocks (SUPER, DRAW, generic-large), ascending.
            for blocks in [&super_blocks, &draw_blocks, &large_blocks]
                .into_iter()
                .flatten()
            {
                sink.put(&blocks.0)?;
                sink.put(&blocks.1)?;
            }
            debug_assert_eq!(sink.position(), base.get() + meta_end);

            // Raw data region: metadata page tail, then small data, DRAW tail,
            // large runs (with their fragments), padded to the page-aligned EOA.
            sink.put_zeros((raw_start - meta_end).to_usize()?)?;
            for &i in &small_indices {
                debug_assert_eq!(sink.position(), base.get() + ds_layouts[i].data_addr);
                emit_ds_data(
                    sink,
                    &ds_layouts[i].data,
                    all_ds[i].raw_chunks.as_ref(),
                    all_ds[i].produced.as_ref(),
                )?;
            }
            if small_raw_total > 0 {
                sink.put_zeros((align_up(small_raw_end, page_size) - small_raw_end).to_usize()?)?;
            }
            for &i in &large_indices {
                let data_addr = ds_layouts[i].data_addr;
                let gap = base.absolute(data_addr)? - sink.position();
                sink.put_zeros(gap.to_usize()?)?;
                emit_ds_data(
                    sink,
                    &ds_layouts[i].data,
                    all_ds[i].raw_chunks.as_ref(),
                    all_ds[i].produced.as_ref(),
                )?;
                let end_rel = base.relative(sink.position())?;
                sink.put_zeros((align_up(end_rel, page_size) - end_rel).to_usize()?)?;
            }
            let final_pad = eof_addr2 - sink.position();
            sink.put_zeros(final_pad.to_usize()?)?;
            debug_assert_eq!(sink.position(), eof_addr2);
            return Ok(());
        }

        let mut ds_layouts: Vec<DsLayout> = Vec::new();
        for (i, d) in all_ds.iter_mut().enumerate() {
            if is_chunked[i] {
                let data_address = cursor2 as u64;
                let built = build_chunked(d, data_address, chunk_sets[i].as_ref())?;
                cursor2 += built.data.len().to_usize()?;
                ds_layouts.push(DsLayout {
                    data: built.data,
                    data_addr: data_address,
                    chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                });
            } else {
                // `d.raw` is not read again for a contiguous/compact dataset, so
                // move its element buffer into the layout rather than cloning it.
                // A produced region leaves its provider behind for the emitter.
                let data = d.take_contiguous();
                let addr = if data.len() == 0 {
                    u64::MAX
                } else {
                    let a = cursor2 as u64;
                    cursor2 += data.len().to_usize()?;
                    a
                };
                ds_layouts.push(DsLayout {
                    data,
                    data_addr: addr,
                    chunked_msgs: None,
                });
            }
        }

        // Patch VL references (attribute and dataset-element) with the GCOL
        // addresses, which sit after all dataset data. Attribute collections are
        // emitted first (root, groups, datasets), then dataset-element
        // collections, and the cursor walk below assigns addresses in that same
        // order so it matches the emission order at the end of the buffer.
        let has_vl = !vl_root.is_empty()
            || grp_vl.iter().any(|v| !v.is_empty())
            || ds_vl.iter().any(|v| !v.is_empty())
            || all_ds.iter().any(|d| d.vl_string_staging.is_some());

        let mut gcol_total_size = 0usize;
        if has_vl {
            let mut gcol_cursor = cursor2 as u64;
            for patch in &vl_root {
                let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                patch_vl_refs(&mut root_attrs[patch.attr_index].raw_data, &addrs);
            }
            for (gi, patches) in grp_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                    patch_vl_refs(&mut groups[gi].attrs[patch.attr_index].raw_data, &addrs);
                }
            }
            for (di, patches) in ds_vl.iter().enumerate() {
                for patch in patches {
                    let addrs = place_collections(&patch.collections, &mut gcol_cursor);
                    patch_vl_refs(&mut all_ds[di].attrs[patch.attr_index].raw_data, &addrs);
                }
            }
            // Dataset-element VL references. The references live in the
            // contiguous/compact element bytes (`ds_layouts[i].data`, cloned
            // from `d.raw`). A chunked dataset's references are *not* here: they
            // sit inside chunks that were encoded earlier, so the loop below must
            // skip it. Dropping that skip would patch heap addresses into the
            // encoded (possibly compressed) chunk blob and silently corrupt it.
            for (i, d) in all_ds.iter().enumerate() {
                // A chunked VL dataset was placed and patched before its chunks
                // were encoded; its references are already final.
                if early_gcol.contains(&i) {
                    continue;
                }
                if let Some(staging) = &d.vl_string_staging {
                    // A dataset that carries VL staging always has its element
                    // bytes in memory. Neither of the two data regions that do
                    // not — a streamed (lazy) chunked one, or a produced
                    // contiguous one — can carry VL staging: the chunked path
                    // patches before encoding, and `with_produced_data` refuses
                    // the combination. Assert that rather than risk silently
                    // patching heap addresses into the wrong buffer.
                    let DsData::InMemory(ref mut bytes) = ds_layouts[i].data else {
                        unreachable!(
                            "a chunked VL-string dataset is patched before encoding, so a \
                             dataset patched here always has its data in memory"
                        );
                    };
                    let addrs = place_collections(&staging.collections, &mut gcol_cursor);
                    patch_vl_refs_masked(bytes, &staging.patch_offsets, &addrs);
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "global-heap total size is an in-memory output span bounded by \
                          addressable memory on the target"
            )]
            {
                gcol_total_size = (gcol_cursor - cursor2 as u64) as usize;
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

        // The superblock extension, if any, is appended after the GCOLs. Its
        // address is base-relative (like every other stored address); the reader
        // adds the base address. eof grows by the extension's size.
        let ext_addr = ext_oh.as_ref().map(|_| (cursor2 + gcol_total_size) as u64);
        let ext_len = ext_oh.as_ref().map_or(0, |b| b.len());

        // eof_address is absolute file size (includes userblock + GCOLs + ext)
        let eof_addr2 = (ub + cursor2 + gcol_total_size + ext_len) as u64;

        // Let a buffered (Vec) sink preallocate the whole file up front, as the
        // writer did before streaming; a streaming sink ignores this.
        sink.reserve(eof_addr2.to_usize()?);

        // Userblock: the caller's header bytes, then zeros.
        Self::put_userblock(sink, ub, &self.userblock_content)?;

        let sb = Superblock {
            version: superblock_version(libver),
            offset_size: OFFSET_SIZE,
            offset_width: OFFSET_WIDTH,
            length_size: LENGTH_SIZE,
            length_width: LENGTH_WIDTH,
            base_address: BaseAddress::new(ub as u64),
            eof_address: eof_addr2,
            root_group_address: root_group_addr,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            consistency_flags: 0,
            superblock_extension_address: Some(ext_addr.unwrap_or(u64::MAX)),
            checksum: None,
        };
        sink.put(&sb.serialize())?;

        // Early-placed VL collections, at the addresses patched into the chunked
        // datasets' references before their chunks were encoded. This must walk
        // `early_gcol` in the order the placement loop built it, or the patched
        // addresses name the wrong collections.
        for &i in &early_gcol {
            let staging = all_ds[i]
                .vl_string_staging
                .as_ref()
                .expect("early_gcol only holds datasets with VL staging");
            emit_collections(sink, &staging.collections)?;
        }
        debug_assert_eq!(
            sink.position(),
            ub as u64 + root_group_addr,
            "early VL collections must occupy exactly the space reserved for them"
        );

        // Root group OH
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
        )?;
        debug_assert_eq!(root_oh.len(), root_oh_size);
        sink.put(&root_oh)?;
        if let Some(ref blob) = root_dense_blob {
            sink.put(&blob.blob)?;
        }

        // Group OHs + dense blobs
        for (gi, g) in groups.iter().enumerate() {
            let links = tables.links(&g.ds_indices, &g.committed_indices, &g.sub_group_indices);
            let oh = build_group_oh(
                &links,
                &g.attrs,
                group_dense_blobs[gi]
                    .as_ref()
                    .map(|b| b.attr_info_message.as_slice()),
            )?;
            debug_assert_eq!(oh.len(), group_oh_sizes[gi]);
            sink.put(&oh)?;
            if let Some(ref blob) = group_dense_blobs[gi] {
                sink.put(&blob.blob)?;
            }
        }

        // Committed datatype OHs, at the addresses their references name.
        for oh in &committed_oh {
            sink.put(oh)?;
        }

        // Dataset OHs + dense blobs
        for (i, oh) in ds_oh_bytes2.iter().enumerate() {
            sink.put(oh)?;
            if let Some(ref dense) = ds_dense_blobs[i] {
                sink.put(&dense.blob)?;
            }
        }

        // Data. Contiguous/compact and eager chunked datasets emit their
        // in-memory bytes; a streamed (lazy) chunked dataset pulls each chunk
        // from its provider one at a time, so its bytes never all reside here.
        for (i, layout) in ds_layouts.iter().enumerate() {
            emit_ds_data(
                sink,
                &layout.data,
                all_ds[i].raw_chunks.as_ref(),
                all_ds[i].produced.as_ref(),
            )?;
        }

        // Global heap collections
        for patch in &vl_root {
            emit_collections(sink, &patch.collections)?;
        }
        for patches in &grp_vl {
            for patch in patches {
                emit_collections(sink, &patch.collections)?;
            }
        }
        for patches in &ds_vl {
            for patch in patches {
                emit_collections(sink, &patch.collections)?;
            }
        }
        // Dataset-element VL string collections, in the same order their
        // addresses were assigned above.
        for (i, d) in all_ds.iter().enumerate() {
            if early_gcol.contains(&i) {
                continue; // emitted right after the superblock
            }
            if let Some(staging) = &d.vl_string_staging {
                emit_collections(sink, &staging.collections)?;
            }
        }

        // Superblock extension (File Space Info), at the address recorded above.
        // For a persisting non-paged file, rebuild the message with a real
        // end-of-allocation — the end of all content, since no FSM blocks follow —
        // in place of the UNDEF placeholder (see the capture above; issue #178). The
        // message length is unchanged (only the `eoa_pre_fsm` field differs), so the
        // reserved layout still holds.
        let real_ext_oh = match (&ext_oh, nonpaged_persist) {
            (Some(_), Some((strategy, threshold, np_page_size))) => {
                let mut info = FileSpaceInfo::persistent_empty(strategy, threshold, np_page_size);
                info.eoa_pre_fsm = eof_addr2 - ub as u64;
                let mut oh = ObjectHeaderWriter::new();
                oh.add_message_with_flags(MessageType::FileSpaceInfo, info.serialize(), 0x14);
                Some(oh.serialize()?)
            }
            (other, _) => other.clone(),
        };
        debug_assert_eq!(
            real_ext_oh.as_ref().map_or(0, |b| b.len()),
            ext_len,
            "rebuilt extension header length must match the reserved length"
        );
        if let Some(bytes) = &real_ext_oh {
            debug_assert_eq!(
                sink.position(),
                ub as u64 + ext_addr.unwrap(),
                "extension header must land at its recorded base-relative address"
            );
            sink.put(bytes)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group_v2::resolve_path_any;
    use crate::link_info::LinkInfoMessage;
    use crate::object_header::ObjectHeader;
    use crate::signature;
    use crate::type_builders::{build_attr_message, make_i32_type};

    /// A committed datatype object is a header holding the type and nothing else,
    /// which is what makes the C library call it a named datatype rather than a
    /// malformed dataset (`H5O__dtype_isa`: a datatype message, no dataspace, no
    /// layout).
    #[test]
    fn a_committed_datatype_header_holds_only_its_type() {
        let bytes = build_committed_datatype_oh(&make_i32_type(), 1).unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();

        let types: Vec<MessageType> = hdr.messages.iter().map(|m| m.msg_type).collect();
        assert_eq!(
            types,
            vec![MessageType::Datatype],
            "a singly referenced committed type carries its datatype and nothing else"
        );
        assert_eq!(
            Datatype::parse(&hdr.messages[0].data).unwrap().0,
            make_i32_type()
        );
        assert_eq!(
            hdr.messages[0].flags,
            MSG_CONSTANT | MSG_DONTSHARE,
            "the type of a committed object never changes and must not be shared onward"
        );
    }

    /// Above one reference the count is stored, in the message the format defines
    /// for it. A header without one reads as singly referenced, so the count is
    /// the difference between unlinking the name and destroying the type.
    #[test]
    fn a_committed_datatype_header_records_a_count_above_one() {
        let bytes = build_committed_datatype_oh(&make_i32_type(), 4).unwrap();
        let hdr = ObjectHeader::parse(&bytes, 0, OFFSET_SIZE, LENGTH_SIZE).unwrap();

        let refcount = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::ObjectReferenceCount)
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
        w.commit_datatype("mytype", make_i32_type());
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
        w.commit_datatype("mytype", make_i32_type());
        let bytes = w.finish().unwrap();

        assert_eq!(committed_reference_count(&bytes, "mytype"), None);
    }

    /// The reference count recorded in the committed datatype object at `path`, or
    /// `None` when the header carries no Object Reference Count message.
    fn committed_reference_count(bytes: &[u8], path: &str) -> Option<u32> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr =
            ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::ObjectReferenceCount)?;
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
        w.commit_datatype("mytype", make_i32_type());
        let ds = w.create_dataset("typed");
        ds.with_i32_data(&[1, 2]);
        ds.with_committed_datatype("mytype");
        let bytes = w.finish().unwrap();

        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let type_addr = resolve_path_any(&bytes, &sb, "mytype").unwrap();
        let ds_addr = resolve_path_any(&bytes, &sb, "typed").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, ds_addr as usize, sb.offset_size, sb.length_size).unwrap();
        let msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .expect("a dataset always has a datatype message");

        assert!(
            crate::shared_message::is_shared(msg.flags),
            "the record must say its body is a reference, or the reference decodes as a type"
        );
        assert_eq!(
            msg.data,
            crate::shared_message::encode_committed_ref(type_addr, OFFSET_SIZE),
            "the reference must name the object the link resolves to"
        );
    }

    fn parse_file(bytes: &[u8]) -> (Superblock, ObjectHeader) {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let oh = ObjectHeader::parse(
            bytes,
            sb.root_group_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        (sb, oh)
    }

    fn read_dataset_f64(bytes: &[u8], path: &str) -> Vec<f64> {
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let hdr =
            ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let dt_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .unwrap()
            .data;
        let ds_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .unwrap()
            .data;
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .unwrap()
            .data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, sb.length_size).unwrap();
        let dl =
            crate::data_layout::DataLayout::parse(dl_data, sb.offset_size, sb.length_size).unwrap();
        let raw = crate::data_read::read_raw_data(bytes, &dl, &ds, &dt).unwrap();
        crate::data_read::read_as_f64(&raw, &dt).unwrap()
    }

    #[test]
    fn empty_file_root_group_only() {
        let fw = FileWriter::new();
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        assert_eq!(sb.version, 3);
        assert_eq!(oh.version, 2);
    }

    #[test]
    fn file_with_f64_dataset() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn file_with_dataset_attrs() {
        let mut fw = FileWriter::new();
        fw.create_dataset("data")
            .with_f64_data(&[1.0, 2.0])
            .set_attr("scale", AttrValue::F64(0.5));
        let bytes = fw.finish().unwrap();
        assert_eq!(read_dataset_f64(&bytes, "data"), vec![1.0, 2.0]);
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
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
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "grp").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        assert_eq!(hdr.flags & 0x20, 0, "times-stored flag must be clear");
        assert!(hdr.modification_time.is_none());
    }

    // The other half of the fixed shape: every group is "new style" (a Link
    // Info + Group Info message pair) with links stored inline, regardless of
    // child count — hdf5-pure never converts a group to dense (fractal-heap)
    // link storage on write (see issue #131 and the tracked gap in #102).
    #[test]
    fn group_links_stay_compact_regardless_of_child_count() {
        let mut fw = FileWriter::new();
        let mut gb = fw.create_group("grp");
        for i in 0..20 {
            gb.create_dataset(&format!("d{i}"))
                .with_f64_data(&[i as f64]);
        }
        fw.add_group(gb.finish());
        let bytes = fw.finish().unwrap();
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "grp").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();

        let link_info_msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::LinkInfo)
            .unwrap();
        let link_info = LinkInfoMessage::parse(&link_info_msg.data, sb.offset_size).unwrap();
        assert!(
            link_info.fractal_heap_address.is_none(),
            "no dense link storage is ever used"
        );

        let group_info_msg = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::GroupInfo)
            .unwrap();
        assert_eq!(group_info_msg.data, vec![0, 0]);

        let link_count = hdr
            .messages
            .iter()
            .filter(|m| m.msg_type == MessageType::Link)
            .count();
        assert_eq!(link_count, 20);
    }

    #[test]
    fn file_with_root_attr() {
        let mut fw = FileWriter::new();
        fw.set_root_attr("version", AttrValue::I64(42));
        let bytes = fw.finish().unwrap();
        let (sb, oh) = parse_file(&bytes);
        let attrs = crate::attribute::extract_attributes(&oh, sb.length_size).unwrap();
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
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes_full(
            &bytes,
            &hdr,
            sb.offset_size,
            sb.length_size,
            None,
        )
        .unwrap();
        assert_eq!(attrs.len(), 20);
        for i in 0..20 {
            let attr = attrs
                .iter()
                .find(|a| a.name == format!("attr_{i:03}"))
                .unwrap();
            let v = attr.read_as_f64().unwrap();
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
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let oh = ObjectHeader::parse(
            &bytes,
            sb.root_group_address as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let attrs = crate::attribute::extract_attributes_full(
            &bytes,
            &oh,
            sb.offset_size,
            sb.length_size,
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
        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        // The Attribute Info message is present but names no fractal heap: that
        // is what tells the reference library the attributes are the header's own
        // inline messages, and how many there are.
        let info = hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::AttributeInfo)
            .map(|m| {
                crate::attribute_info::AttributeInfoMessage::parse(&m.data, sb.offset_size).unwrap()
            })
            .expect("a compact attribute set still carries an Attribute Info message");
        assert_eq!(info.fractal_heap_address, None);
        assert_eq!(info.btree_name_index_address, None);
        let attrs = crate::attribute::extract_attributes(&hdr, sb.length_size).unwrap();
        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn encode_decode_managed_id_roundtrip() {
        let id = encode_managed_id(100, 42, 40, 8);
        let fh = crate::fractal_heap::FractalHeapHeader {
            heap_id_length: 8,
            io_filter_encoded_length: 0,
            max_managed_object_size: 1024,
            btree_huge_objects_address: u64::MAX,
            table_width: 4,
            starting_block_size: 4096,
            max_direct_block_size: 65536,
            max_heap_size: 40,
            start_root_rows: 1,
            root_block_address: 0,
            current_rows_in_root_indirect_block: 0,
            managed_objects_count: 0,
        };
        let (off, len) = fh.decode_managed_id(&id).unwrap();
        assert_eq!(off, 100);
        assert_eq!(len, 42);
    }

    /// An `AsciiString` attribute named `name` whose serialized (v3) size is
    /// exactly `size` bytes, so a bound can be tested on the value it bounds.
    fn dense_attr_of_size(name: &str, size: usize) -> AttributeMessage {
        // Probe with one character and subtract it, rather than probing with an
        // empty string: a fixed-width string datatype is at least one byte wide,
        // so an empty value still carries a padding byte and would be measured
        // as overhead.
        let probe = build_attr_message(name, &AttrValue::AsciiString("y".to_string()));
        let overhead = probe.serialize_v3(LENGTH_SIZE).len() - 1;
        let attr = build_attr_message(name, &AttrValue::AsciiString("y".repeat(size - overhead)));
        assert_eq!(attr.serialize_v3(LENGTH_SIZE).len(), size);
        attr
    }

    #[test]
    fn dense_attrs_check_bounds_each_attribute_not_the_total() {
        // A multi-megabyte set of individually small attributes is exactly what
        // the emitter handles: it sizes its root direct block to the content, and
        // both this crate and the reference C library read such a heap back. The
        // old bound rejected these.
        let many: Vec<AttributeMessage> = (0..40)
            .map(|i| dense_attr_of_size(&format!("a{i}"), 60_000))
            .collect();
        let total: usize = many.iter().map(|a| a.serialize_v3(LENGTH_SIZE).len()).sum();
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
            huge_object_count(&build_dense_attrs(&past, DenseAttrCreationOrder::Untracked, 0).blob),
            1
        );
        assert_eq!(
            huge_object_count(
                &build_dense_attrs(&at_limit, DenseAttrCreationOrder::Untracked, 0).blob
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
            let plan = dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked);
            for base in [0u64, 0x1000, 0x8000_0000] {
                let built = plan.build(base);
                assert_eq!(
                    plan.blob_len(),
                    built.blob.len() as u64,
                    "planned length must match the emitted heap for {label} at base {base:#x}"
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
        let blob = build_dense_attrs(attrs, DenseAttrCreationOrder::Untracked, 0).blob;
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
            .map(|n| build_attr_message(n, &AttrValue::I64(1)))
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

    /// A dense set whose creation order is [`SCRAMBLE`] rather than the order
    /// the attributes are handed over in, so nothing that reads the index
    /// records can pass by reproducing a position.
    fn scrambled_creation_order() -> (Vec<AttributeMessage>, DenseAttrCreationOrder) {
        let attrs: Vec<AttributeMessage> = (0..SCRAMBLE.len())
            .map(|i| build_attr_message(&format!("a{i:02}"), &AttrValue::I64(i as i64)))
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
    /// The reference C library reaches this index from `H5Aopen_by_idx`; its
    /// `H5Aiterate2` builds a table off the name index instead, so a file
    /// missing the tree it declares still iterates correctly. That is why this
    /// reads the bytes rather than leaving the claim to a crosscheck.
    #[test]
    fn an_indexed_dense_set_emits_a_creation_order_btree() {
        const COUNT: usize = SCRAMBLE.len();
        let (attrs, creation) = scrambled_creation_order();
        let heap_address = 0x4000u64;
        let built = build_dense_attrs(&attrs, creation, heap_address);

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
            heap_address + at as u64,
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
        let leaf = (root - heap_address) as usize + 6; // signature(4) + version(1) + type(1)
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
        let name_leaf = (name_root - heap_address) as usize + 6;
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
        let built = build_dense_attrs(&attrs, DenseAttrCreationOrder::Untracked, 0x4000);
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
            let (info, depth) = crate::btree_v2::NodeInfo::for_record_count(
                btree_v2_write::NODE_SIZE,
                record_size,
                OFFSET_SIZE,
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
        let blob = build_dense_attrs(&mixed, DenseAttrCreationOrder::Untracked, 0).blob;
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
    /// starting-size block, which is where the reference C library also grows a
    /// root indirect block instead of a bigger block.
    #[test]
    fn the_root_grows_into_an_indirect_block_rather_than_a_bigger_direct_one() {
        let fits = vec![dense_attr_of_size("a", 400)];
        let blob = build_dense_attrs(&fits, DenseAttrCreationOrder::Untracked, 0).blob;
        let (address, rows) = root_block(&blob);
        assert_eq!(rows, 0, "one starting-size block still holds this heap");
        assert_eq!(&blob[address..address + 4], b"FHDB");

        let spills: Vec<AttributeMessage> = (0..8)
            .map(|i| dense_attr_of_size(&format!("a{i}"), 400))
            .collect();
        let blob = build_dense_attrs(&spills, DenseAttrCreationOrder::Untracked, 0).blob;
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

        let sig = signature::find_signature(&bytes).unwrap();
        let sb = Superblock::parse(&bytes, sig).unwrap();
        let addr = resolve_path_any(&bytes, &sb, "data").unwrap();
        let hdr =
            ObjectHeader::parse(&bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let attrs = crate::attribute::extract_attributes_full(
            &bytes,
            &hdr,
            sb.offset_size,
            sb.length_size,
            None,
        )
        .unwrap();
        assert_eq!(attrs.len(), count);
        for i in 0..count {
            let attr = attrs.iter().find(|a| a.name == format!("big{i}")).unwrap();
            assert_eq!(attr.read_as_string().unwrap(), value(i));
        }
    }

    /// The number of `i64` elements in the largest attribute whose serialized
    /// message still fits the object header's 2-byte message-size field, derived
    /// from a measured probe rather than hard-coded so it tracks the encoder.
    fn largest_fitting_i64_attr_elements() -> usize {
        let one = build_attr_message("boundary", &AttrValue::I64Array(vec![0i64; 1]));
        let overhead = one.serialize(LENGTH_SIZE).len() - 8;
        (OBJECT_HEADER_MESSAGE_MAX - overhead) / 8
    }

    #[test]
    fn compact_attr_at_the_message_size_limit_is_written() {
        let n = largest_fitting_i64_attr_elements();
        let attr = build_attr_message("boundary", &AttrValue::I64Array(vec![7i64; n]));
        // Pin the probe to the boundary itself: one more element must not fit,
        // or this test would still pass while exercising a tiny attribute.
        let size = attr.serialize(LENGTH_SIZE).len();
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
        let attrs = vec![build_attr_message(
            "boundary",
            &AttrValue::I64Array(vec![0i64; n]),
        )];
        assert!(attrs[0].serialize(LENGTH_SIZE).len() > OBJECT_HEADER_MESSAGE_MAX);
        assert!(
            needs_dense_attrs(&attrs),
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
        let attrs = vec![build_attr_message(
            "boundary",
            &AttrValue::I64Array(vec![0i64; n]),
        )];
        assert!(!needs_dense_attrs(&attrs));
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
            .with_deflate(6);
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
            .with_maxshape(&[u64::MAX])
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

        let dt = Datatype::VariableLength {
            is_string: false,
            padding: None,
            charset: None,
            base_type: Box::new(crate::type_builders::make_i32_type()),
        };
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
                Datatype::VariableLength {
                    is_string: false,
                    ..
                }
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
        let sig = signature::find_signature(bytes).unwrap();
        let sb = Superblock::parse(bytes, sig).unwrap();
        let addr = resolve_path_any(bytes, &sb, path).unwrap();
        let oh = ObjectHeader::parse(bytes, addr as usize, sb.offset_size, sb.length_size).unwrap();
        oh.messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
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
            let sig = signature::find_signature(&bytes).unwrap();
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
        let sig = signature::find_signature(&newer).unwrap();

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
        fw.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
        fw.create_dataset("d").with_i32_data(&[1]);
        assert!(matches!(
            fw.finish().unwrap_err(),
            FormatError::LibverTooOldForContent { .. }
        ));

        // A page size on its own emits the same 1.10-only File Space Info
        // message — `file_space_info` fires on either field — so it takes the
        // same refusal. The guard used to test the strategy alone and let this
        // through, writing a version 2 superblock carrying a message that did
        // not exist before 1.10.
        let mut fw = FileWriter::new();
        fw.with_libver_bounds(LibVer::Earliest, LibVer::V18);
        fw.with_file_space_page_size(4096);
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
            .with_maxshape(&[u64::MAX]);
        assert!(matches!(
            fw.finish().unwrap_err(),
            FormatError::LibverTooOldForContent { .. }
        ));
    }
}
