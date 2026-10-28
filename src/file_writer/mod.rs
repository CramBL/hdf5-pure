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

#[cfg(all(
    test,
    feature = "std",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod allocation_tests;
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
            allocation.end() - SUPERBLOCK_SIZE as u64
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
            allocation = FileAllocation::new(SUPERBLOCK_SIZE as u64 + early_gcol_size, false, None);
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
                allocation.end(),
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
            layout::paged_metadata_section_allocation(
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
            layout::paged_metadata_section_allocation(
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
    fn dense_attribute_preparation_accepts_managed_and_huge_messages() {
        let many: Vec<AttributeMessage> = (0..40)
            .map(|i| dense_attr_of_size(&format!("a{i}"), 60_000))
            .collect();
        let total: usize = many
            .iter()
            .map(|a| a.serialize_v3(OFFSET_WIDTH, LENGTH_SIZE).unwrap().len())
            .sum();
        assert!(total > 2_000_000, "expected a multi-megabyte set");
        PreparedDenseAttrs::new(&many, DenseAttrCreationOrder::Untracked, []).unwrap();

        let at_limit = vec![dense_attr_of_size("edge", DENSE_ATTR_MAX_MANAGED_OBJECT)];
        PreparedDenseAttrs::new(&at_limit, DenseAttrCreationOrder::Untracked, []).unwrap();

        let past = vec![dense_attr_of_size(
            "edge",
            DENSE_ATTR_MAX_MANAGED_OBJECT + 1,
        )];
        PreparedDenseAttrs::new(&past, DenseAttrCreationOrder::Untracked, []).unwrap();
        assert_eq!(
            huge_object_count(
                &dense_attribute_write::build_dense_attrs(
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
                &dense_attribute_write::build_dense_attrs(
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
            let plan =
                dense_attribute_write::dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked)
                    .unwrap();
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
        let blob = dense_attribute_write::build_dense_attrs(
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
        let built =
            dense_attribute_write::build_dense_attrs(&attrs, creation, heap_address).unwrap();

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
        let built = dense_attribute_write::build_dense_attrs(
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
        let blob = dense_attribute_write::build_dense_attrs(
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
        let blob = dense_attribute_write::build_dense_attrs(
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
        let blob = dense_attribute_write::build_dense_attrs(
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
