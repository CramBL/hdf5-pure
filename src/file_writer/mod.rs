//! HDF5 file creation with compact or dense group links and attributes.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::format;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use hdf5_pure_core::__private::SuperblockFields;
pub(crate) use hdf5_pure_format::__private as dense_attribute_write;
#[cfg(test)]
use hdf5_pure_format::__private::DENSE_ATTR_MAX_MANAGED_OBJECT;
use hdf5_pure_format::__private::DenseAttrBlob;
pub(crate) use hdf5_pure_format::__private::DenseAttrCreationOrder;
use hdf5_pure_format::__private::DenseAttrFixup;
use hdf5_pure_format::__private::DenseAttrPlan;
use hdf5_pure_format::__private::DenseAttrReference;
use hdf5_pure_format::__private::DenseLinkImage;
use hdf5_pure_format::__private::DenseLinkPlan;
use hdf5_pure_format::__private::DenseLinkPlanError;
#[cfg(test)]
use hdf5_pure_format::__private::FractalHeapHeader;
#[cfg(test)]
use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::LayoutVersion;
use hdf5_pure_format::__private::LinkStorageError;
use hdf5_pure_format::__private::ManagedHeapPlanError;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::PreparedDenseAttrs;
use hdf5_pure_space::__private::AllocationClass;

use self::preparation::{DsData, PreparedAttributes, PreparedDataset, VlPatch};
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
use crate::bytes;
use crate::chunked_write::{self, ByteSink, CompressedChunkSet};
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
#[cfg(test)]
use crate::dataspace::MaxExtent;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_allocation::FileAllocation;
use crate::file_create_properties::FileCreateProperties;
use crate::file_emission::Emission;
use crate::file_emission::EmissionError;
use crate::file_emission::Reservation;
use crate::file_space_info;
use crate::file_space_info::DEFAULT_THRESHOLD;
use crate::file_space_info::FileSpaceInfo;
#[cfg(test)]
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::free_space_manager;
use crate::libver::LibVer;
use crate::link_info::LinkInfoMessage;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header_writer::ObjectHeaderWriter;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::shared_message::DatatypeLocation;
use crate::type_builders::{
    self, AttrSpec, AttributeSpecs, CommittedDatatype, DatasetBuilder, FinishedGroup, GroupBuilder,
};
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;
use crate::{SuperblockStatus, SuperblockStatusFlags};

// `AttrValue` lives in `type_builders`; `types` and `mat` reference it through
// this module's path, so keep it re-exported here.
pub use crate::type_builders::AttrValue;

use crate::datatype::{CharacterSet, Datatype};

mod layout;
mod preparation;
#[cfg(feature = "std")]
pub(crate) mod source;

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

/// Resolves prepared dense storage from the attributes' completed references and payloads.
///
/// The caller passes the same attribute set in its prepared order, after placing global-heap
/// collections, resolving committed datatypes, and completing deferred payloads. Only declared
/// updates are copied into the encoded messages. Heap and index geometry are preserved.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if a prepared attribute or collection reference is absent,
/// a committed datatype is inline, a collection offset overflows, or a payload changes length.
/// Propagates address errors from [`PreparedDenseAttrs::finalize`] and reference-read errors.
pub(crate) fn finalize_dense_attrs(
    prepared: PreparedDenseAttrs,
    attrs: &[AttributeMessage],
) -> Result<DenseAttrPlan, FormatError> {
    let attribute = |index: usize| {
        attrs.get(index).ok_or_else(|| {
            FormatError::Internal("a prepared dense attribute is absent during finalization".into())
        })
    };
    prepared.finalize(
        |reference| match reference {
            DenseAttrReference::CommittedDatatype { attribute: index } => {
                match attribute(index)?.datatype_location {
                    DatatypeLocation::Committed(address) => Ok(address),
                    DatatypeLocation::Inline => Err(FormatError::Internal(
                        "a prepared committed attribute has an inline datatype".into(),
                    )),
                }
            }
            DenseAttrReference::GlobalHeap {
                attribute: index,
                collection,
            } => {
                let offset = collection
                    .checked_mul(dense_attribute_write::GLOBAL_HEAP_MAX_OBJECTS)
                    .and_then(|element| element.checked_mul(type_builders::VL_REF_SIZE))
                    .and_then(|offset| offset.checked_add(core::mem::size_of::<u32>()))
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "a prepared attribute collection offset overflows".into(),
                        )
                    })?;
                // `H5T__vlen_disk_write` writes the sequence length before the heap address
                // (`H5Tvlen.c`, HDF5 2.2.0). This writer uses eight-byte addresses.
                bytes::read_offset_width(&attribute(index)?.raw_data, offset, OFFSET_WIDTH)
                    .map(StoredAddress::new)
                    .map_err(|error| match error {
                        FormatError::UnexpectedEof { .. } => FormatError::Internal(
                            "a prepared attribute collection reference is absent".into(),
                        ),
                        error => error,
                    })
            }
        },
        |index| Ok(attribute(index)?.raw_data.as_slice()),
    )
}

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

/// The provisional heap address used to size object headers.
///
/// The writer sizes headers with [`PreparedDenseAttrs::attr_info_message`] before reserving
/// their space. The message length is independent of the chosen heap address.
const DUMMY_DENSE_BASE: StoredAddress = StoredAddress::ZERO;

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
        let mut allocation = FileAllocation::new(
            StoredAddress::new(SUPERBLOCK_SIZE as u64),
            aggregated,
            reuse_threshold,
        );

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
        // The writer needs final manager addresses and EOA for persistent File Space Info.
        // Retain the settings for the rewrite after placement.
        let nonpaged_persist = file_space.filter(|settings| {
            settings.strategy() != FileSpaceStrategy::Page && settings.persist()
        });
        struct GrpFlat {
            /// Link name in the owning group.
            name: LinkNameBuf,
            attrs: Vec<AttributeMessage>,
            /// Attribute indices paired with committed datatype paths for address resolution.
            attr_committed_datatype_paths: Vec<(usize, ObjectPathBuf)>,
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

        let mut all_ds: Vec<PreparedDataset> = Vec::new();
        let mut groups: Vec<GrpFlat> = Vec::new();
        let mut committed: Vec<CtFlat> = Vec::new();
        let mut root_ds_indices: Vec<usize> = Vec::new();
        let mut root_group_indices: Vec<usize> = Vec::new();

        /// Prepares a dataset and appends its metadata and attribute heap patches.
        ///
        /// Returns its index in `datasets`.
        ///
        /// # Errors
        ///
        /// Propagates errors from [`PreparedDataset::prepare`].
        fn flatten_dataset(
            builder: DatasetBuilder,
            datasets: &mut Vec<PreparedDataset>,
            heaps: &mut Vec<Vec<VlPatch>>,
        ) -> Result<usize, FormatError> {
            let (dataset, patches) = PreparedDataset::prepare(builder)?;
            let index = datasets.len();
            datasets.push(dataset);
            heaps.push(patches);
            Ok(index)
        }

        /// Prepares committed datatype objects and returns their indices in `committed`.
        ///
        /// # Errors
        ///
        /// Returns [`FormatError::ZeroSizedDatatype`] if a datatype has zero element width.
        /// Propagates datatype-size and link-name errors.
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
            all_ds: &mut Vec<PreparedDataset>,
            groups: &mut Vec<GrpFlat>,
            committed: &mut Vec<CtFlat>,
            grp_vl: &mut Vec<Vec<VlPatch>>,
            ds_vl: &mut Vec<Vec<VlPatch>>,
        ) -> Result<usize, FormatError> {
            let name = g.name?;
            let PreparedAttributes {
                messages: gattrs,
                committed_datatype_paths: attr_committed_datatype_paths,
                heaps: patches,
            } = PreparedAttributes::new(&g.attrs)?;
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

        let PreparedAttributes {
            messages: mut root_attrs,
            committed_datatype_paths: root_attr_committed_datatype_paths,
            heaps: vl_root,
        } = PreparedAttributes::new(&self.root_attrs)?;

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

        for (index, path) in &root_attr_committed_datatype_paths {
            let attr = &root_attrs[*index];
            register_committed_use(
                &mut committed,
                &committed_by_path,
                path,
                &attr.datatype,
                || format!("root attribute {:?}", attr.name),
            )?;
        }
        for g in &groups {
            for (index, path) in &g.attr_committed_datatype_paths {
                let attr = &g.attrs[*index];
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
            for (index, path) in &d.attr_committed_datatype_paths {
                let attr = &d.attrs[*index];
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
            all_ds: &'a [PreparedDataset],
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

        /// Checks compact message lengths before chunk compression.
        ///
        /// # Errors
        ///
        /// Returns [`FormatError::AttributeMessageTooLarge`] if a compact message exceeds the
        /// object-header size limit. Propagates attribute encoding errors.
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
        /// Prepares dense geometry and deferred updates, or checks compact message lengths.
        ///
        /// # Errors
        ///
        /// Propagates compact attribute encoding and size errors, or errors from
        /// [`PreparedDenseAttrs::new`] for dense storage.
        fn prepare_dense(
            attrs: &[AttributeMessage],
            dense: bool,
            patches: &[VlPatch],
            payload: Option<usize>,
        ) -> Result<Option<PreparedDenseAttrs>, FormatError> {
            if !dense {
                check_compact_attrs(attrs)?;
                return Ok(None);
            }
            let fixups = patches
                .iter()
                .map(|patch| DenseAttrFixup::GlobalHeap {
                    attribute: patch.attr_index,
                })
                .chain(attrs.iter().enumerate().filter_map(|(attribute, attr)| {
                    attr.datatype_location
                        .is_committed()
                        .then_some(DenseAttrFixup::CommittedDatatype { attribute })
                }))
                .chain(payload.map(|attribute| DenseAttrFixup::Payload { attribute }));
            PreparedDenseAttrs::new(attrs, DenseAttrCreationOrder::Untracked, fixups).map(Some)
        }
        let root_dense_plan = prepare_dense(&root_attrs, root_dense, &vl_root, None)?;
        let group_dense_plans = groups
            .iter()
            .enumerate()
            .map(|(i, group)| prepare_dense(&group.attrs, group_dense[i], &grp_vl[i], None))
            .collect::<Result<Vec<_>, _>>()?;
        let ds_dense_plans = all_ds
            .iter()
            .enumerate()
            .map(|(i, dataset)| {
                #[cfg(feature = "provenance")]
                let payload = dataset.pending_hash.as_ref().map(|hash| hash.attribute);
                #[cfg(not(feature = "provenance"))]
                let payload = None;
                prepare_dense(&dataset.attrs, ds_dense[i], &ds_vl[i], payload)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let is_chunked: Vec<bool> = all_ds.iter().map(PreparedDataset::is_chunked).collect();

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
        let mut collection_sections = layout::CollectionSections::default();
        let early_gcol_size = if let Some((page_size, _, _)) = paged {
            let mut placements = layout::PagedCollections::new(SUPERBLOCK_SIZE as u64, page_size);
            let mut place = |collections: &[Vec<u8>]| -> Result<Vec<u64>, FormatError> {
                collections.iter().try_fold(
                    Vec::with_capacity(collections.len()),
                    |mut addrs, collection| {
                        let placed = placements.reserve(collection.len() as u64)?;
                        paged_collection_addresses.push(placed.address.get());
                        addrs.push(placed.address.get());
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
                if let Some(staging) = dataset.storage.take_vl_string_staging() {
                    let addrs = place(&staging.collections)?;
                    type_builders::patch_vl_refs_masked(
                        dataset.storage.raw_mut()?,
                        &staging.patch_offsets,
                        &addrs,
                    );
                    dataset.storage.restore_vl_string_staging(staging)?;
                    early_gcol.push(i);
                }
            }
            collection_sections = placements.sections;
            placements.end - SUPERBLOCK_SIZE as u64
        } else {
            for i in 0..all_ds.len() {
                if !is_chunked[i] || all_ds[i].storage.chunks().is_some() {
                    continue;
                }
                let Some(staging) = all_ds[i].storage.take_vl_string_staging() else {
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
                all_ds[i].storage.restore_vl_string_staging(staging)?;
                early_gcol.push(i);
            }
            allocation.end().get() - SUPERBLOCK_SIZE as u64
        };

        #[cfg(feature = "provenance")]
        for (i, dataset) in all_ds.iter_mut().enumerate() {
            if dataset.storage.reference_targets().is_none()
                && (dataset.storage.vl_string_staging().is_none() || early_gcol.contains(&i))
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
                if is_chunked[i]
                    && d.storage.chunks().is_none()
                    && d.storage.reference_targets().is_none()
                {
                    d.compress_chunks().map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<_, FormatError>>()?;

        /// Prepared dense storage paired with its reserved file address for each object.
        struct DenseReservations {
            root: Option<(StoredAddress, PreparedDenseAttrs)>,
            groups: Vec<Option<(StoredAddress, PreparedDenseAttrs)>>,
            datasets: Vec<Option<(StoredAddress, PreparedDenseAttrs)>>,
        }

        /// File addresses of the emitted dense heaps, in object order.
        struct DenseSpans {
            root: Option<StoredAddress>,
            groups: Vec<Option<StoredAddress>>,
            datasets: Vec<Option<StoredAddress>>,
        }

        /// Serialized dense heaps and Attribute Info messages, in object order.
        struct DenseBlobs {
            root: Option<DenseAttrBlob>,
            groups: Vec<Option<DenseAttrBlob>>,
            datasets: Vec<Option<DenseAttrBlob>>,
        }

        impl DenseReservations {
            /// Finalizes and serializes each heap at its reserved address.
            ///
            /// The caller completes attribute references and element digests before emission.
            /// Each prepared value retains the geometry used to reserve its file space.
            ///
            /// # Errors
            ///
            /// Returns [`FormatError::Internal`] if an element digest is pending. Propagates
            /// errors from [`finalize_dense_attrs`] and [`DenseAttrPlan::build`], including an
            /// emitted length that differs from its reservation.
            fn build(
                self,
                root_attrs: &[AttributeMessage],
                groups: &[GrpFlat],
                datasets: &[PreparedDataset],
            ) -> Result<(DenseSpans, DenseBlobs), FormatError> {
                let Self {
                    root,
                    groups: group_reservations,
                    datasets: dataset_reservations,
                } = self;
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
                    span: Option<(StoredAddress, PreparedDenseAttrs)>,
                ) -> Result<Option<DenseEmission>, FormatError> {
                    match span {
                        Some((address, prepared)) => {
                            let blob = finalize_dense_attrs(prepared, attrs)?.build(address)?;
                            Ok(Some(DenseEmission { address, blob }))
                        }
                        None => Ok(None),
                    }
                }
                let split = |emission: Option<DenseEmission>| match emission {
                    Some(DenseEmission { address, blob }) => (Some(address), Some(blob)),
                    None => (None, None),
                };
                let (root_span, root_blob) = split(one(root_attrs, root)?);
                let (group_spans, group_blobs) = groups
                    .iter()
                    .zip(group_reservations)
                    .map(|(group, span)| one(&group.attrs, span).map(split))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .unzip();
                let (dataset_spans, dataset_blobs) = datasets
                    .iter()
                    .zip(dataset_reservations)
                    .map(|(dataset, span)| one(&dataset.attrs, span).map(split))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .unzip();
                Ok((
                    DenseSpans {
                        root: root_span,
                        groups: group_spans,
                        datasets: dataset_spans,
                    },
                    DenseBlobs {
                        root: root_blob,
                        groups: group_blobs,
                        datasets: dataset_blobs,
                    },
                ))
            }
        }

        struct DenseEmission {
            address: StoredAddress,
            blob: DenseAttrBlob,
        }

        // Size object headers at provisional addresses using the prepared dense metadata.
        // Borrow the object tables after variable-length staging and chunk compression.
        let dummy_tables = LinkTables {
            all_ds: &all_ds,
            committed: &committed,
            groups: &groups,
            ds_addrs: &dummy_ds_addrs,
            committed_addrs: &dummy_ct_addrs,
            group_addrs: &dummy_grp_addrs,
        };
        let mut group_oh_sizes: Vec<usize> = Vec::with_capacity(groups.len());
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
            let attr_info = group_dense_plans[gi]
                .as_ref()
                .map(|plan| plan.attr_info_message(DUMMY_DENSE_BASE))
                .transpose()?;
            let oh = build_group_oh(
                &dummy_links,
                &g.attrs,
                attr_info.as_deref(),
                link_info.as_ref(),
            )?;
            group_oh_sizes.push(oh.len());
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
        let root_attr_info = root_dense_plan
            .as_ref()
            .map(|plan| plan.attr_info_message(DUMMY_DENSE_BASE))
            .transpose()?;
        let root_oh_size = build_group_oh(
            &root_dummy_links,
            &root_attrs,
            root_attr_info.as_deref(),
            root_link_info.as_ref(),
        )?
        .len();

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
        let mut dummy_cursor = 0u64;
        for (i, d) in all_ds.iter().enumerate() {
            let dense_plan = ds_dense_plans[i].as_ref();
            let dense_attr_info = dense_plan
                .as_ref()
                .map(|plan| plan.attr_info_message(DUMMY_DENSE_BASE))
                .transpose()?;
            let measured = d.measure(
                StoredAddress::new(dummy_cursor),
                chunk_sets[i].as_ref(),
                dense_attr_info.as_deref(),
                libver,
            )?;
            if is_chunked[i] {
                dummy_cursor += measured.data_len;
            }
            ds_data_lens.push(measured.data_len);
            actual_ds_oh_sizes.push(measured.header_len);
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
            allocation = FileAllocation::new(
                StoredAddress::new(SUPERBLOCK_SIZE as u64 + early_gcol_size),
                false,
                None,
            );
        }
        let reserve = |allocation: &mut FileAllocation,
                       header: u64,
                       dense: Option<PreparedDenseAttrs>,
                       links: Option<usize>|
         -> Result<_, FormatError> {
            let layout::ObjectAllocation {
                header,
                attributes,
                links: placed_links,
            } = layout::ObjectMeasure {
                header,
                attributes: dense.as_ref().map(PreparedDenseAttrs::blob_len),
                links: links.map(|len| len as u64),
            }
            .reserve(allocation)?;
            let attributes = match (attributes, dense) {
                (Some(span), Some(plan)) => Some((span.address, plan)),
                (None, None) => None,
                _ => {
                    return Err(FormatError::Internal(
                        "dense attribute placement is absent".into(),
                    ));
                }
            };
            let links = placed_links
                .map(|layout::Allocation { address, length }| {
                    length.to_usize().map(|length| (address, length))
                })
                .transpose()?;
            Ok((header.address, attributes, links))
        };
        let (root_group_addr, root_dense_span, root_link_span) = reserve(
            &mut allocation,
            root_oh_size as u64,
            root_dense_plan,
            root_link_len,
        )?;
        let mut group_link_spans = Vec::with_capacity(groups.len());
        let mut group_dense_spans = Vec::with_capacity(groups.len());
        let group_addrs2: Vec<StoredAddress> = group_oh_sizes
            .iter()
            .zip(group_dense_plans)
            .enumerate()
            .try_fold(
                Vec::with_capacity(group_oh_sizes.len()),
                |mut addrs, (gi, (&size, dense))| {
                    let (address, attributes, links) =
                        reserve(&mut allocation, size as u64, dense, group_link_lens[gi])?;
                    group_dense_spans.push(attributes);
                    group_link_spans.push(links);
                    addrs.push(address);
                    Ok::<_, FormatError>(addrs)
                },
            )?;
        let committed_addrs: Vec<StoredAddress> = committed_oh.iter().try_fold(
            Vec::with_capacity(committed_oh.len()),
            |mut addrs, header| {
                let placed = layout::ObjectMeasure {
                    header: header.len() as u64,
                    attributes: None,
                    links: None,
                }
                .reserve(&mut allocation)?;
                addrs.push(placed.header.address);
                Ok::<_, FormatError>(addrs)
            },
        )?;
        let mut ds_dense_spans = Vec::with_capacity(all_ds.len());
        let ds_oh_addrs2: Vec<StoredAddress> =
            actual_ds_oh_sizes.iter().zip(ds_dense_plans).try_fold(
                Vec::with_capacity(actual_ds_oh_sizes.len()),
                |mut addrs, (&size, dense)| {
                    let (address, attributes, _) =
                        reserve(&mut allocation, size as u64, dense, None)?;
                    ds_dense_spans.push(attributes);
                    addrs.push(address);
                    Ok::<_, FormatError>(addrs)
                },
            )?;

        let dense_reservations = DenseReservations {
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
                    all_ds: &[PreparedDataset],
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
                let Some(patches) = d.storage.take_reference_targets() else {
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
                                 paths: &[(usize, ObjectPathBuf)]| {
                for (index, path) in paths {
                    attrs[*index].datatype_location = committed_location(path);
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
            all_ds: &[PreparedDataset],
            ds_layouts: &[DsLayout<'_>],
            ds_dense_blobs: &[Option<DenseAttrBlob>],
            libver: LibVer,
        ) -> Result<Vec<Vec<u8>>, FormatError> {
            let mut oh_bytes: Vec<Vec<u8>> = Vec::with_capacity(all_ds.len());
            for (i, d) in all_ds.iter().enumerate() {
                let layout = &ds_layouts[i];
                let oh = d.build_header(
                    layout.data_addr,
                    layout.chunked_msgs.as_ref(),
                    ds_dense_blobs[i]
                        .as_ref()
                        .map(|blob| blob.attr_info_message.as_slice()),
                    libver,
                )?;
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
            let (
                _,
                DenseBlobs {
                    root: root_dense_blob,
                    groups: group_dense_blobs,
                    datasets: ds_dense_blobs,
                },
            ) = dense_reservations.build(&root_attrs, &groups, &all_ds)?;
            let measurements = ds_data_lens
                .iter()
                .zip(&is_chunked)
                .map(|(&length, &chunked)| {
                    let kind = if chunked {
                        layout::DataKind::Chunked
                    } else {
                        layout::DataKind::Contiguous
                    };
                    layout::DataMeasure::new(length, kind)
                });
            let layout::PagedLayout {
                extension:
                    layout::Allocation {
                        address: extension_address,
                        length: ext_len,
                    },
                data: ds_layouts,
                small: small_indices,
                large: large_indices,
                slots,
                managers,
                metadata_end: meta_end,
                raw_start,
                small_end: small_raw_end,
                end: eoa_pre_fsm,
            } = layout::PagedLayout::new(
                page_size,
                persist_paged,
                allocation.end().get(),
                ext_oh.as_ref().map_or(0, |bytes| bytes.len() as u64),
                measurements,
                collection_sections,
                |i, placed: Option<layout::Allocation>| {
                    let dataset = &mut all_ds[i];
                    let address = placed.map_or(
                        UNDEF_ADDRESS,
                        |layout::Allocation { address, length: _ }| address,
                    );
                    if is_chunked[i] {
                        let built = dataset.build_chunked(address, chunk_sets[i].as_ref())?;
                        if built.data.len() != ds_data_lens[i] {
                            return Err(FormatError::Internal(
                                "chunk data differs from its allocation".into(),
                            ));
                        }
                        Ok(DsLayout {
                            data: built.data,
                            data_addr: address,
                            chunked_msgs: Some((built.layout_message, built.pipeline_message)),
                        })
                    } else {
                        Ok(DsLayout {
                            data: dataset.contiguous_data(),
                            data_addr: address,
                            chunked_msgs: None,
                        })
                    }
                },
            )?;
            let ext_addr = extension_address.get();
            let small_raw_total = small_raw_end - raw_start;
            let eof_addr2 = base.absolute(StoredAddress::new(eoa_pre_fsm))?;

            let ds_oh_bytes = build_ds_ohs(&all_ds, &ds_layouts, &ds_dense_blobs, libver)?;
            debug_assert_eq!(
                ds_oh_bytes.iter().map(|b| b.len()).collect::<Vec<_>>(),
                actual_ds_oh_sizes
            );

            // Rebuild the superblock-extension header. For persisting
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
                        .storage
                        .vl_string_staging()
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
            for manager in managers.iter().flatten() {
                sink.put(&manager.header).map_err(EmissionError::Sink)?;
                sink.put(&manager.section_list)
                    .map_err(EmissionError::Sink)?;
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
                let span = layout::DataMeasure::new(ds_data_lens[i], layout::DataKind::Chunked)
                    .reserve(&mut allocation)?
                    .ok_or_else(|| {
                        FormatError::Internal("chunk data allocation is absent".into())
                    })?;
                let data_address = span.address;
                let built = d.build_chunked(data_address, chunk_sets[i].as_ref())?;
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
                let data = d.contiguous_data();
                let addr = layout::DataMeasure::new(data.len(), layout::DataKind::Contiguous)
                    .reserve(&mut allocation)?
                    .map_or(UNDEF_ADDRESS, |span| span.address);
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
            || all_ds
                .iter()
                .any(|d| d.storage.vl_string_staging().is_some());

        let gcol_start = allocation.end().get();
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
            // Patch dataset-element references in the contiguous bytes held in prepared storage.
            // The writer reserves chunked datasets' collections and patches their references before
            // chunk encoding.
            for (i, d) in all_ds.iter_mut().enumerate() {
                // A chunked VL dataset was placed and patched before its chunks
                // were encoded; its references are already final.
                if early_gcol.contains(&i) {
                    continue;
                }
                if let Some(staging) = d.storage.take_vl_string_staging() {
                    // A dataset that carries VL staging always has its element
                    // bytes in memory. Report it: heap addresses patched into
                    // another buffer would corrupt the file in silence.
                    if !matches!(ds_layouts[i].data, DsData::InMemory { .. }) {
                        return Err(preparation::vl_element_bytes_absent(d.name.as_str()).into());
                    }
                    let bytes = d.storage.raw_mut()?;
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
                    d.storage.restore_vl_string_staging(staging)?;
                }
            }
        }

        let (
            dense_spans,
            DenseBlobs {
                root: root_dense_blob,
                groups: group_dense_blobs,
                datasets: ds_dense_blobs,
            },
        ) = dense_reservations.build(&root_attrs, &groups, &all_ds)?;

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
        let layout::ClassicLayout {
            extension,
            end: cursor2,
            manager,
        } = layout::ClassicLayout::finish(
            allocation,
            ext_oh.as_ref().map(|bytes| bytes.len() as u64),
            nonpaged_persist.is_some(),
        )?;
        let ext_addr = extension.map(|layout::Allocation { address, length: _ }| address.get());

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
                    .storage
                    .vl_string_staging()
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
                    .storage
                    .vl_string_staging()
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
            .filter_map(|(_, d)| d.storage.vl_string_staging())
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
                    *span,
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

        // Replace the placeholder with the manager address and final EOA. The message has
        // the same length with or without a manager.
        let real_ext_oh = match (&ext_oh, nonpaged_persist) {
            (Some(_), Some(settings)) => {
                let info = match &manager {
                    Some(manager) => file_space_info::persistent_single_manager(
                        OFFSET_WIDTH,
                        settings.strategy(),
                        settings.threshold(),
                        settings.page_size(),
                        manager.header_address.get(),
                        cursor2,
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

/// Returns the File Space Info message used to size the superblock extension.
///
/// Persistent manager addresses and the end of allocation are undefined until
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

/// The value reserved for an unlimited maximum dimension.
///
/// `H5Screate_simple` rejects this value in current dimensions (`H5S.c`, HDF5 1.14.6).
pub(crate) const UNLIMITED_DIMENSION: u64 = u64::MAX;

/// The largest dataset rank accepted by the writer.
///
/// `H5Screate_simple` uses `H5S_MAX_RANK` for this bound (`H5S.c`, HDF5 1.14.6).
pub(crate) const MAX_DATASET_RANK: u8 = 32;

#[cfg(test)]
mod tests;
