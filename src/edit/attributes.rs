//! Attribute edit preparation and compact and dense storage placement.

use hdf5_pure_format::__private::{
    DenseAttrFixup, DenseAttrPlan, PreparedDenseAttrs, RecordSplice,
};
use hdf5_pure_space::__private::AllocationClass;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::attribute::{AttributeMessage, StoredAttribute};
use crate::attribute_info::AttributeInfoMessage;
use crate::convert::Narrow;
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_writer::{DenseAttrCreationOrder, LENGTH_SIZE};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::{BaseOffsetSource, Source, SourceMetadata};
use crate::type_builders::{self, AttrValue};

use super::WriteEngine;
use super::object_header::OhRegion;
use super::staging::StagedOp;

impl WriteEngine {
    /// Stages an attribute addition or replacement for the next [`commit`](Self::commit).
    ///
    /// The group may be committed or staged. [`plan_attr_ops`] plans attribute storage
    /// when the commit rebuilds its header.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn set_group_attr(
        &mut self,
        path: &ObjectPathBuf,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditGroupAttribute(
            path.clone(),
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stages attribute removal from a group for the next [`commit`](Self::commit).
    ///
    /// The attribute must exist after earlier staged attribute operations are applied.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn remove_group_attr(&mut self, path: &ObjectPathBuf, name: &str) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditGroupAttribute(
            path.clone(),
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
    }

    /// Stages an attribute addition or replacement on a committed dataset.
    ///
    /// The next [`commit`](Self::commit) relocates the dataset header and requires a single
    /// hard link. To set attributes on a staged creation, use
    /// [`DatasetBuilder::set_attr`](crate::DatasetBuilder::set_attr) before staging it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn set_dataset_attr(
        &mut self,
        path: &ObjectPathBuf,
        name: &str,
        value: AttrValue,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditDatasetAttribute(
            path.clone(),
            AttrOp::Set {
                name: name.to_string(),
                value,
            },
        ));
        Ok(())
    }

    /// Stages attribute removal from a committed dataset.
    ///
    /// The attribute must exist after earlier staged attribute operations are applied.
    /// The next [`commit`](Self::commit) relocates the dataset header and requires a single
    /// hard link.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`.
    pub fn remove_dataset_attr(&mut self, path: &ObjectPathBuf, name: &str) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        self.staged.push(StagedOp::EditDatasetAttribute(
            path.clone(),
            AttrOp::Remove {
                name: name.to_string(),
            },
        ));
        Ok(())
    }

    /// Places a nonempty dense attribute set and adds its Attribute Info message to `region`.
    ///
    /// Resolves deferred updates within the geometry prepared before commit application.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if a prepared update cannot be resolved, the emitted length
    /// differs from its reservation, or allocation geometry is invalid.
    /// Returns [`Error::Io`] if writing the heap or allocation padding fails.
    pub(super) fn append_dense_attrs(
        &mut self,
        region: &mut OhRegion,
        set: DenseAttrSet,
    ) -> Result<(), Error> {
        if set.is_empty() {
            return Ok(());
        }
        let attr_info_message =
            self.place_dense_attrs(crate::file_writer::finalize_dense_attrs(
                set.prepared.ok_or_else(|| {
                    FormatError::Internal("a dense set has no prepared geometry".into())
                })?,
                &set.attrs,
            )?)?;
        region.push(MessageType::ATTRIBUTE_INFO, &attr_info_message);
        Ok(())
    }

    /// Writes dense attribute storage and returns its Attribute Info message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if heap addresses or allocation geometry are invalid, or the
    /// emitted length differs from its reservation. Returns [`Error::Io`] if writing the heap
    /// or allocation padding fails.
    pub(super) fn place_dense_attrs(&mut self, plan: DenseAttrPlan) -> Result<Vec<u8>, Error> {
        let (_addr, attr_info_message) =
            self.place_relocatable(plan.blob_len(), AllocationClass::Metadata, |stored_base| {
                let blob = plan.build(stored_base)?;
                Ok((blob.blob, blob.attr_info_message))
            })?;
        Ok(attr_info_message)
    }

    /// Places prepared attribute storage and appends its header records.
    ///
    /// Resolves new variable-length values' global heap addresses before serializing compact
    /// records or building dense storage. An empty compact result requires no placement.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a compact record cannot be rewritten. Propagates
    /// attribute encoding, allocation, and write errors from the selected storage path.
    pub(super) fn place_edited_attrs(
        &mut self,
        region: &mut OhRegion,
        attrs: EditedAttrs,
    ) -> Result<(), Error> {
        match attrs {
            EditedAttrs::Compact(pending) => {
                for pending in pending {
                    let PendingVlAttr {
                        mut msg,
                        collections,
                        creation_index,
                    } = pending;
                    let addrs = self.place_vl_collections(&collections)?;
                    type_builders::patch_vl_refs(&mut msg.raw_data, &addrs);
                    *region = put_attr_message(
                        region,
                        &msg.name,
                        &msg.serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)?,
                        creation_index,
                    )?;
                }
                Ok(())
            }
            EditedAttrs::Dense(mut dense) => {
                for (idx, collections) in std::mem::take(&mut dense.vl) {
                    let addrs = self.place_vl_collections(&collections)?;
                    type_builders::patch_vl_refs(&mut dense.set.attrs[idx].raw_data, &addrs);
                }
                self.append_dense_attrs(region, dense.set)
            }
        }
    }
}

/// An attribute addition, replacement, or removal staged for a group or dataset.
pub(super) enum AttrOp {
    Set { name: String, value: AttrValue },
    Remove { name: String },
}

/// A compact attribute whose global heap collections are placed during commit application.
#[derive(Clone)]
pub(super) struct PendingVlAttr {
    /// The attribute message with placeholder global heap references.
    msg: crate::attribute::AttributeMessage,
    /// The global heap collections to allocate before patching the message.
    collections: Vec<Vec<u8>>,
    /// The existing creation index, when it must be preserved.
    creation_index: Option<u16>,
}

/// Compact variable-length attributes prepared by [`apply_compact_attr_ops`].
type PendingVlAttrs = Vec<PendingVlAttr>;

/// Appends compact Attribute Info metadata if the region has inline attributes and no such message.
///
/// Preserves an existing Attribute Info message, including one naming dense storage.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or the next creation
/// index cannot be represented.
pub(super) fn ensure_attribute_info(region: &mut OhRegion) -> Result<(), Error> {
    let mut has_attrs = false;
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = region.next_message(p)? {
        match msg_type {
            MessageType::ATTRIBUTE_INFO => return Ok(()),
            MessageType::ATTRIBUTE => has_attrs = true,
            _ => {}
        }
        p = body_end;
    }
    if has_attrs {
        let body = compact_attribute_info_body(region)?;
        region.push(MessageType::ATTRIBUTE_INFO, &body);
    }
    Ok(())
}

/// Applies compact attribute operations and prepares variable-length values for placement.
///
/// Applies fixed-size values directly to a copy of `region`. Collects variable-length values
/// for [`WriteEngine::place_edited_attrs`] to resolve their heap addresses and append their
/// records. Clears any earlier pending value with the same name before applying each operation.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a required attribute is absent, a record cannot be
/// rewritten, or an encoded attribute exceeds [`OBJECT_HEADER_MESSAGE_MAX`]. Propagates
/// attribute encoding errors.
fn apply_compact_attr_ops(
    region: &OhRegion,
    ops: &[&AttrOp],
) -> Result<(OhRegion, PendingVlAttrs), Error> {
    let mut out = region.clone();
    let mut pending_vl: PendingVlAttrs = Vec::new();
    for op in ops {
        match op {
            AttrOp::Set { name, value } => {
                pending_vl.retain(|a| &a.msg.name != name);
                if let Some(strings) = value.var_len_strings() {
                    // Read the existing creation index before removing the record.
                    let creation_index = attr_creation_index(&out, name)?;
                    // A new attribute may have no inline record to remove.
                    out = remove_attr_from_region(&out, name, false)?;
                    let msg = type_builders::build_attr_message(name, value)?;
                    if msg
                        .serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)?
                        .len()
                        > OBJECT_HEADER_MESSAGE_MAX
                    {
                        return Err(Error::EditUnsupported(
                            "attribute is too large to encode in place",
                        ));
                    }
                    pending_vl.push(PendingVlAttr {
                        msg,
                        collections: type_builders::build_global_heap_collections(strings),
                        creation_index,
                    });
                } else {
                    out = set_attr_in_region(&out, name, value)?;
                }
            }
            AttrOp::Remove { name } => {
                let before = pending_vl.len();
                pending_vl.retain(|a| &a.msg.name != name);
                if pending_vl.len() == before {
                    out = remove_attr_from_region(&out, name, true)?;
                }
            }
        }
    }
    Ok((out, pending_vl))
}

/// A rewritten header region and the attribute storage prepared for commit application.
pub(super) struct AttrEdits {
    /// The header region after compact edits or removal of the previous dense metadata.
    pub(super) region: OhRegion,
    /// Attribute storage awaiting placement.
    pub(super) attrs: EditedAttrs,
    /// Whether the edit replaces dense storage named by the pre-commit header.
    pub(super) replaces_dense_storage: bool,
}

/// Attribute storage that requires placement during commit application.
///
/// Compact results contain pending variable-length attributes. Dense results contain the
/// complete replacement set.
#[derive(Clone)]
pub(super) enum EditedAttrs {
    /// Variable-length attributes to place and append as inline records.
    Compact(PendingVlAttrs),
    /// A set to rebuild into a fresh fractal heap.
    Dense(DenseAttrEdit),
}

impl Default for EditedAttrs {
    /// Creates a compact result with no pending placements.
    fn default() -> Self {
        Self::Compact(Vec::new())
    }
}

/// Attribute messages and prepared geometry for replacement dense storage.
///
/// The default is an empty set without a heap. Cloning copies the messages and deferred
/// updates while sharing immutable heap and index geometry.
#[derive(Clone, Default)]
pub(super) struct DenseAttrSet {
    attrs: Vec<AttributeMessage>,
    /// Prepared storage, absent for the default empty set.
    prepared: Option<PreparedDenseAttrs>,
}

impl DenseAttrSet {
    /// Encodes the set with its creation metadata and deferred updates before placement.
    ///
    /// # Errors
    ///
    /// Returns the encoding, metadata, and update errors of [`PreparedDenseAttrs::new`].
    pub(super) fn new(
        attrs: Vec<AttributeMessage>,
        creation: DenseAttrCreationOrder,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            prepared: Some(PreparedDenseAttrs::new(&attrs, creation, fixups)?),
            attrs,
        })
    }

    /// Returns the messages used to resolve the prepared updates during placement.
    pub(super) fn attrs(&self) -> &[AttributeMessage] {
        &self.attrs
    }

    /// Returns whether the set requires no attribute heap.
    pub(super) fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }
}

/// A dense replacement set with new global heap collections awaiting placement.
#[derive(Clone)]
pub(super) struct DenseAttrEdit {
    /// The complete attribute set and its creation-order metadata.
    set: DenseAttrSet,
    /// Attribute positions paired with new global heap collections to place before serialization.
    ///
    /// Existing variable-length values retain their stored collection addresses.
    vl: Vec<(usize, Vec<Vec<u8>>)>,
}

/// Prepares attribute edits and selects compact or dense storage for the result.
///
/// `addr` is the existing object's absolute header address, or `None` for a new group. Compact
/// edits preserve records of attributes whose names are absent from `ops`. New variable-length
/// values are placed during commit application.
///
/// Selects dense storage if the object is already dense, a staged value exceeds the record
/// size limit, or a compact result with a staged addition or replacement exceeds
/// [`MAX_COMPACT_ATTRS`]. A dense rebuild preserves attribute values and creation indexes,
/// serializing the messages in
/// version 3 format. An empty dense result requires no replacement heap.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if an edit cannot be reproduced, a required attribute is
/// absent, creation indexes are exhausted, or conversion to dense storage would prevent
/// reference patching. Propagates source-read, parsing, and storage-planning errors.
pub(super) fn plan_attr_ops<S: Source + ?Sized>(
    src: &S,
    base: AddressContext,
    addr: Option<u64>,
    region: &OhRegion,
    ops: &[&AttrOp],
) -> Result<AttrEdits, Error> {
    let dense_now = region_uses_dense_attrs(region)?;
    if !dense_now {
        // Detect oversized values before attempting compact editing.
        let mut oversized = false;
        for op in ops {
            if let AttrOp::Set { name, value } = op {
                oversized |= type_builders::build_attr_message(name, value)?
                    .serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)?
                    .len()
                    > OBJECT_HEADER_MESSAGE_MAX;
            }
        }
        if !oversized {
            let (out, pending_vl) = apply_compact_attr_ops(region, ops)?;
            // Removals preserve compact storage even above this writer's count threshold.
            let only_removes = !ops.iter().any(|op| matches!(op, AttrOp::Set { .. }));
            if only_removes || compact_attr_count(&out)? + pending_vl.len() <= MAX_COMPACT_ATTRS {
                return Ok(AttrEdits {
                    region: out,
                    attrs: EditedAttrs::Compact(pending_vl),
                    replaces_dense_storage: false,
                });
            }
        }
    }

    // Reject flagged records before resolving attributes for a dense rebuild.
    if region_has_shared_attr(region)? {
        return Err(Error::EditUnsupported(SHARED_ATTRIBUTE_MESSAGE));
    }

    // Read the full stored set before applying dense edits. New groups have no stored set.
    let existing = match addr {
        Some(addr) => read_object_attrs(src, AccessMode::ReadWrite, addr, base)?,
        None => Vec::new(),
    };
    // Keep each attribute with its creation index and pending collections during editing.
    let existing = dense_attr_set(region, existing)?;
    let tracked = region.layout().tracks_creation_order();
    // Assign new attributes indexes after the stored set's highest index.
    let mut next_index = match &existing.creation {
        DenseAttrCreationOrder::Tracked { max, .. } => *max,
        DenseAttrCreationOrder::Untracked => 0,
    };
    let mut set: Vec<DenseAttrSlot> = existing
        .attrs
        .into_iter()
        .enumerate()
        .map(|(i, msg)| DenseAttrSlot {
            msg,
            collections: None,
            creation_index: existing.creation.index_at(i),
        })
        .collect();
    for op in ops {
        match op {
            AttrOp::Set { name, value } => {
                let msg = type_builders::build_attr_message(name, value)?;
                let collections = value
                    .var_len_strings()
                    .map(type_builders::build_global_heap_collections);
                match set.iter_mut().find(|slot| &slot.msg.name == name) {
                    // Preserve the existing position and creation index on replacement.
                    Some(slot) => {
                        slot.msg = msg;
                        slot.collections = collections;
                    }
                    None => {
                        let creation_index = tracked.then_some(next_index);
                        if tracked {
                            next_index = bump_creation_index(next_index)?;
                        }
                        set.push(DenseAttrSlot {
                            msg,
                            collections,
                            creation_index,
                        });
                    }
                }
            }
            AttrOp::Remove { name } => {
                let before = set.len();
                // Preserve the counter and the remaining creation indexes on removal.
                set.retain(|slot| &slot.msg.name != name);
                if set.len() == before {
                    return Err(Error::EditUnsupported("attribute to remove was not found"));
                }
            }
        }
    }

    // Remove the old attribute records and storage metadata before appending replacements.
    let region = strip_attr_messages(region)?;
    // An empty result requires no attribute storage.
    if set.is_empty() {
        return Ok(AttrEdits {
            region,
            attrs: EditedAttrs::default(),
            replaces_dense_storage: dense_now,
        });
    }
    let mut attrs = Vec::with_capacity(set.len());
    let mut indices = Vec::with_capacity(set.len());
    let mut vl = Vec::new();
    for (i, slot) in set.into_iter().enumerate() {
        if let Some(collections) = slot.collections {
            vl.push((i, collections));
        }
        // Every tracked slot has a creation index read from storage or assigned during editing.
        // The untracked result omits `indices`.
        debug_assert!(
            !tracked || slot.creation_index.is_some(),
            "a tracked object's attribute must carry a creation index"
        );
        indices.push(slot.creation_index.unwrap_or_default());
        attrs.push(slot.msg);
    }
    let creation = if tracked {
        DenseAttrCreationOrder::Tracked {
            indices,
            max: next_index,
            indexed: region.layout().indexes_creation_order(),
        }
    } else {
        DenseAttrCreationOrder::Untracked
    };
    // Reference patching supports inline attributes, so reject conversions of repointable values.
    if !dense_now {
        for attr in &attrs {
            if crate::reference_patch::attribute_references_are_repointable(&attr.datatype) {
                return Err(Error::EditUnsupported(
                    REFERENCE_ATTRIBUTE_WOULD_LEAVE_THE_HEADER,
                ));
            }
        }
    }
    Ok(AttrEdits {
        region,
        attrs: EditedAttrs::Dense(DenseAttrEdit {
            set: DenseAttrSet::new(
                attrs,
                creation,
                vl.iter().map(|(attribute, _)| DenseAttrFixup::GlobalHeap {
                    attribute: *attribute,
                }),
            )?,
            vl,
        }),
        replaces_dense_storage: dense_now,
    })
}

/// One attribute, its creation index, and any global heap collections awaiting placement.
struct DenseAttrSlot {
    msg: crate::attribute::AttributeMessage,
    /// Global heap collections to place for a new variable-length value.
    collections: Option<Vec<Vec<u8>>>,
    /// The attribute's creation index, or `None` for an untracked object.
    creation_index: Option<u16>,
}

/// Returns whether an Attribute Info message identifies dense storage.
///
/// Treats an unparseable Attribute Info message as dense.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region.
fn region_uses_dense_attrs(region: &OhRegion) -> Result<bool, Error> {
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::ATTRIBUTE_INFO
            && attribute_info_is_dense(&region[body..body_end])
        {
            return Ok(true);
        }
        p = body_end;
    }
    Ok(false)
}

/// Returns whether any Attribute record has nonzero message flags.
///
/// Both compact and dense editing reject such records because the editor cannot preserve
/// their flag semantics when rebuilding attributes.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region.
pub(super) fn region_has_shared_attr(region: &OhRegion) -> Result<bool, Error> {
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::ATTRIBUTE && region[p + 3] != 0 {
            return Ok(true);
        }
        p = body_end;
    }
    Ok(false)
}

/// Copies the region without Attribute or Attribute Info records.
///
/// Preserves unrelated records and any trailing gap.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region.
fn strip_attr_messages(region: &OhRegion) -> Result<OhRegion, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = region.next_message(p)? {
        if !matches!(
            msg_type,
            MessageType::ATTRIBUTE | MessageType::ATTRIBUTE_INFO
        ) {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if p < region.len() {
        out.extend_from_slice(&region[p..]);
    }
    Ok(region.with_bytes(out))
}

/// Reads an object's compact or dense attributes into owned messages.
///
/// Frames heap reads relative to the file's effective base address. The caller checks whether
/// the resulting set can be rebuilt after applying any edits.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the effective base exceeds the source length.
/// Propagates header parsing, attribute parsing, and source-read errors.
pub(super) fn read_object_attrs<S: Source + ?Sized>(
    src: &S,
    access_mode: AccessMode,
    addr: u64,
    base: AddressContext,
) -> Result<Vec<crate::attribute::StoredAttribute>, Error> {
    let header = ObjectHeader::parse_from_source_with_widths(
        &SourceMetadata(src),
        access_mode,
        addr,
        crate::file_writer::WIDTHS,
        base.effective_base(),
    )?;
    if base.effective_base().get() > src.len() {
        return Err(Error::EditUnsupported(
            "this file's userblock is larger than the file itself",
        ));
    }
    // Read heap addresses through the source framed at the effective base.
    let framed = BaseOffsetSource {
        inner: src,
        base: base.effective_base(),
    };
    crate::attribute::extract_stored_attributes_from_source(
        &framed,
        base,
        access_mode,
        &header,
        crate::file_writer::WIDTHS,
        // Read attributes without a shared-message table.
        None,
    )
    .map_err(Error::Format)
}

/// Combines stored attributes with the header's creation-order metadata.
///
/// Preserves stored creation indexes and advances the counter past them. Assigns missing
/// indexes from the next unused value. An untracked header requires no creation-order metadata.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or the next creation
/// index cannot be represented.
pub(super) fn dense_attr_set(
    region: &OhRegion,
    stored: Vec<crate::attribute::StoredAttribute>,
) -> Result<StoredAttrSet, Error> {
    let layout = region.layout();
    if !layout.tracks_creation_order() {
        return Ok(StoredAttrSet {
            attrs: stored.into_iter().map(|a| a.message).collect(),
            creation: DenseAttrCreationOrder::Untracked,
        });
    }
    let mut next = next_attr_creation_index(region)?;
    let mut attrs = Vec::with_capacity(stored.len());
    let mut indices = Vec::with_capacity(stored.len());
    for StoredAttribute {
        message,
        creation_index,
    } in stored
    {
        let index = match creation_index {
            Some(index) => index,
            None => {
                let index = next;
                next = bump_creation_index(next)?;
                index
            }
        };
        next = next.max(bump_creation_index(index)?);
        attrs.push(message);
        indices.push(index);
    }
    Ok(StoredAttrSet {
        attrs,
        creation: DenseAttrCreationOrder::Tracked {
            indices,
            max: next,
            indexed: layout.indexes_creation_order(),
        },
    })
}

/// Stored attribute messages combined with their header's creation-order metadata.
pub(super) struct StoredAttrSet {
    pub(super) attrs: Vec<AttributeMessage>,
    pub(super) creation: DenseAttrCreationOrder,
}

/// Returns the index after `index`.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if `index` is [`u16::MAX`].
fn bump_creation_index(index: u16) -> Result<u16, Error> {
    index.checked_add(1).ok_or(Error::EditUnsupported(
        "an object has assigned every attribute creation index its header can record",
    ))
}

/// Returns whether the Attribute Info body refers to a fractal heap or cannot be parsed.
///
/// An undefined heap address denotes compact attributes. The address fields of a version 0
/// message are defined in "The Attribute Info Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
pub(crate) fn attribute_info_is_dense(body: &[u8]) -> bool {
    match crate::attribute_info::AttributeInfoMessage::parse_with_offset_width(
        body,
        crate::file_writer::OFFSET_WIDTH,
    ) {
        Ok(ai) => ai.fractal_heap_address.is_some(),
        Err(_) => true,
    }
}

/// Replaces attributes named `name` with a compact record for `value`.
///
/// Preserves the existing creation index or assigns the next one for a new attribute.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record cannot be rewritten, creation indexes are
/// exhausted, or the encoded body exceeds [`OBJECT_HEADER_MESSAGE_MAX`]. Propagates attribute
/// encoding errors.
///
/// # Panics
///
/// Panics in debug builds if `value` requires global heap storage.
pub(super) fn set_attr_in_region(
    region: &OhRegion,
    name: &str,
    value: &AttrValue,
) -> Result<OhRegion, Error> {
    let body = encode_attr_body(name, value)?;
    let keep = attr_creation_index(region, name)?;
    put_attr_message(region, name, &body, keep)
}

/// Returns the first matching attribute's creation index.
///
/// Returns `None` if the header is untracked or no matching record exists.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or an Attribute
/// record has nonzero flags. Returns [`Error::Format`] if a name cannot be parsed.
pub(super) fn attr_creation_index(region: &OhRegion, name: &str) -> Result<Option<u16>, Error> {
    if !region.layout().tracks_creation_order() {
        return Ok(None);
    }
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::ATTRIBUTE
            && parse_compact_attr_name(region, p, body, body_end)? == name
        {
            return Ok(region.creation_index(p));
        }
        p = body_end;
    }
    Ok(None)
}

/// Returns the highest creation index among the region's Attribute records.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region.
fn highest_attr_creation_index(region: &OhRegion) -> Result<Option<u16>, Error> {
    let mut highest = None;
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::ATTRIBUTE {
            highest = highest.max(region.creation_index(p));
        }
        p = body_end;
    }
    Ok(highest)
}

/// Returns the larger of the recorded counter and one past the highest stored index.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or the next index
/// cannot be represented.
pub(super) fn next_attr_creation_index(region: &OhRegion) -> Result<u16, Error> {
    let recorded = find_attribute_info(region)?
        .and_then(|(_, _, info)| info.max_creation_index)
        .unwrap_or(0);
    let derived = highest_attr_creation_index(region)?.map_or(0, |i| u32::from(i) + 1);
    let next = u32::from(recorded).max(derived);
    next.narrow_or_else(|| {
        Error::EditUnsupported(
            "an object has assigned every attribute creation index its header can record",
        )
    })
}

/// Returns the first parsed Attribute Info message, with its record start and body range.
///
/// Returns `None` if every Attribute Info message is absent or unparseable.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region.
pub(super) fn find_attribute_info(
    region: &OhRegion,
) -> Result<Option<(usize, core::ops::Range<usize>, AttributeInfoMessage)>, Error> {
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::ATTRIBUTE_INFO
            && let Ok(info) = AttributeInfoMessage::parse_with_offset_width(
                &region[body..body_end],
                crate::file_writer::OFFSET_WIDTH,
            )
        {
            return Ok(Some((p, body..body_end, info)));
        }
        p = body_end;
    }
    Ok(None)
}

/// Serializes compact Attribute Info metadata using the region's record layout.
///
/// Uses [`next_attr_creation_index`] for tracked headers and undefined addresses for dense
/// storage fields.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or the next creation
/// index cannot be represented.
fn compact_attribute_info_body(region: &OhRegion) -> Result<Vec<u8>, Error> {
    let layout = region.layout();
    let info = AttributeInfoMessage {
        max_creation_index: layout
            .tracks_creation_order()
            .then(|| next_attr_creation_index(region))
            .transpose()?,
        indexes_creation_order: layout.indexes_creation_order(),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    Ok(info.serialize(crate::file_writer::OFFSET_WIDTH))
}

/// Replaces attributes named `name` with one compact record and updates creation metadata.
///
/// `keep` preserves an existing creation index. `None` assigns the next unused index and
/// updates each parsed Attribute Info message's counter. Untracked headers use index zero.
/// Preserves unrelated bytes and the metadata of rewritten Attribute Info records.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region, an Attribute record
/// has nonzero flags, or creation indexes are exhausted. Propagates attribute-name parsing
/// and record-splicing errors.
///
/// # Panics
///
/// Panics if `body` is longer than [`OBJECT_HEADER_MESSAGE_MAX`].
pub(super) fn put_attr_message(
    region: &OhRegion,
    name: &str,
    body: &[u8],
    keep: Option<u16>,
) -> Result<OhRegion, Error> {
    let layout = region.layout();
    let (creation_index, new_max) = if layout.tracks_creation_order() {
        match keep {
            Some(index) => (index, None),
            None => {
                let index = next_attr_creation_index(region)?;
                let next = index.checked_add(1).ok_or(Error::EditUnsupported(
                    "an object has assigned every attribute creation index its header can record",
                ))?;
                (index, Some(next))
            }
        }
    } else {
        (0, None)
    };
    let new_msg = layout.record_with_creation_index(
        MessageType::ATTRIBUTE,
        MessageFlags::NONE,
        creation_index,
        body,
    );

    let mut out = Vec::with_capacity(region.len() + new_msg.len());
    let mut p = 0;
    while let Some((msg_type, msg_body, body_end)) = region.next_message(p)? {
        match msg_type {
            MessageType::ATTRIBUTE
                if parse_compact_attr_name(region, p, msg_body, body_end)? == name =>
            {
                p = body_end;
                continue;
            }
            // Re-encode the body to update or add the creation counter.
            MessageType::ATTRIBUTE_INFO if new_max.is_some() => {
                if let Ok(mut info) = AttributeInfoMessage::parse_with_offset_width(
                    &region[msg_body..body_end],
                    crate::file_writer::OFFSET_WIDTH,
                ) {
                    info.max_creation_index = new_max;
                    info.indexes_creation_order |= layout.indexes_creation_order();
                    let encoded = info.serialize(crate::file_writer::OFFSET_WIDTH);
                    let splice = RecordSplice::first(
                        layout,
                        &region[p..body_end],
                        MessageType::ATTRIBUTE_INFO,
                    )?
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "a parsed Attribute Info record is absent from its source range"
                                .to_owned(),
                        )
                    })?;
                    splice.replace_body_into(&mut out, &encoded)?;
                    p = body_end;
                    continue;
                }
                out.extend_from_slice(&region[p..body_end]);
            }
            _ => out.extend_from_slice(&region[p..body_end]),
        }
        p = body_end;
    }
    out.extend_from_slice(&new_msg);
    if p < region.len() {
        out.extend_from_slice(&region[p..]);
    }
    let mut out = region.with_bytes(out);
    // Add Attribute Info metadata if the counter has no existing record.
    if new_max.is_some() && find_attribute_info(&out)?.is_none() {
        let body = compact_attribute_info_body(&out)?;
        out.push(MessageType::ATTRIBUTE_INFO, &body);
    }
    Ok(out)
}

/// Copies the region without Attribute records named `name`.
///
/// Preserves unrelated records, creation metadata, and any trailing gap.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region, an Attribute record
/// has nonzero flags, or `required` is true and the attribute is absent. Returns
/// [`Error::Format`] if an attribute name cannot be parsed.
pub(super) fn remove_attr_from_region(
    region: &OhRegion,
    name: &str,
    required: bool,
) -> Result<OhRegion, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut removed = false;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        let mut skip = false;
        if msg_type == MessageType::ATTRIBUTE {
            let attr_name = parse_compact_attr_name(region, p, body, body_end)?;
            if attr_name == name {
                skip = true;
                removed = true;
            }
        }
        if !skip {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if p < region.len() {
        out.extend_from_slice(&region[p..]);
    }
    if !removed && required {
        return Err(Error::EditUnsupported("attribute to remove was not found"));
    }
    Ok(region.with_bytes(out))
}

/// Counts inline Attribute records in a compact region.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region.
fn compact_attr_count(region: &OhRegion) -> Result<usize, Error> {
    let mut count = 0usize;
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::ATTRIBUTE {
            count += 1;
        }
        p = body_end;
    }
    Ok(count)
}

/// Parses the name of an Attribute record with zero message flags.
///
/// The caller passes record and body bounds from the region's message parser. Version 2
/// record fields are defined in "Version 2 Data Object Header Prefix" of the [format
/// specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the record has nonzero flags. Returns
/// [`Error::Format`] if the attribute name cannot be parsed.
///
/// # Panics
///
/// Panics if the flags byte is outside `region` or `body..body_end` is not a valid range in it.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub(super) fn parse_compact_attr_name(
    region: &[u8],
    msg_start: usize,
    body: usize,
    body_end: usize,
) -> Result<String, Error> {
    if region[msg_start + 3] != 0 {
        return Err(Error::EditUnsupported(SHARED_ATTRIBUTE_MESSAGE));
    }
    // Parse the name without resolving committed datatype or dataspace fields.
    crate::attribute::AttributeMessage::parse_name(&region[body..body_end]).map_err(Error::Format)
}

/// Serializes an inline value within the compact message-size limit.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the encoded message exceeds
/// [`OBJECT_HEADER_MESSAGE_MAX`]. Propagates attribute encoding errors.
///
/// # Panics
///
/// Panics in debug builds if `value` requires global heap storage.
fn encode_attr_body(name: &str, value: &AttrValue) -> Result<Vec<u8>, Error> {
    // Compact planning stages variable-length values separately before reaching this encoder.
    debug_assert!(
        value.var_len_strings().is_none(),
        "a variable-length attribute must be intercepted by apply_compact_attr_ops before reaching encode_attr_message"
    );
    let body = type_builders::build_attr_message(name, value)?
        .serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)?;
    if body.len() > OBJECT_HEADER_MESSAGE_MAX {
        return Err(Error::EditUnsupported(
            "group attribute is too large to encode in place",
        ));
    }
    Ok(body)
}

/// The error text for an Attribute record whose message flags prevent editing.
pub(super) const SHARED_ATTRIBUTE_MESSAGE: &str =
    "a target object has a shared attribute message (not editable in place yet)";

/// The error text for a compact-to-dense conversion that would prevent reference patching.
const REFERENCE_ATTRIBUTE_WOULD_LEAVE_THE_HEADER: &str = "an attribute holding an object reference cannot be moved to dense \
     (fractal-heap) storage, where a later commit could no longer repoint it when \
     its target moves; keep this object within compact attribute storage";

/// The compact attribute-count limit shared with the whole-file writer.
pub(super) const MAX_COMPACT_ATTRS: usize = crate::file_writer::DENSE_ATTR_THRESHOLD;
