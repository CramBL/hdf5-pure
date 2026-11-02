//! Attribute edit preparation and compact and dense storage placement.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::ops::Range;

use hdf5_pure_format::__private::{
    DenseAttrBlockKind, DenseAttrFixup, DenseAttrPlan, PreparedDenseAttrs, RecordSplice,
};
use hdf5_pure_space::__private::{AllocationClass, ReservationRequest};

use crate::access_mode::AccessMode;
use crate::attribute::{AttributeMessage, StoredAttribute};
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_policy::AttributePolicy;
use crate::attribute_policy::AttributeStorage;
use crate::attribute_request::AttributeName;
use crate::convert::Narrow;
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_writer::{DenseAttrCreationOrder, LENGTH_SIZE};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::shared_message::SourceResolver;
use crate::source::OriginalSource;
use crate::source::PhysicalSource;
use crate::source::SourceMetadata;
use crate::type_builders::{self, AttrValue};

use super::WriteEngine;
use super::block::{BlockRequest, EncodingDependency};
use super::object_header;
use super::object_header::OhRegion;
use super::reservation::ReservationFailure;
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
    /// In PAGE files, reserves and writes each heap and index block separately, using raw-data
    /// space for huge-object payloads and metadata space for the remaining blocks. Complete
    /// blocks smaller than a page stay within one page. Blocks at least one page long start at
    /// a page boundary.
    /// Reserves one contiguous allocation for other strategies.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if heap addresses or allocation geometry are invalid, or the
    /// emitted length differs from its reservation. Returns
    /// [`Error::AllocationPublicationUncertain`] if allocation publication is uncertain.
    /// Propagates allocation and address-conversion
    /// errors. Returns [`Error::Io`] if writing the heap, indexes, or allocation padding fails.
    pub(super) fn place_dense_attrs(&mut self, plan: DenseAttrPlan) -> Result<Vec<u8>, Error> {
        if !self.space.is_paged() {
            let (_, attr_info_message) = self.place_relocatable(
                plan.blob_len(),
                AllocationClass::Metadata,
                |stored_base| {
                    let blob = plan.build(stored_base)?;
                    Ok((blob.blob, blob.attr_info_message))
                },
            )?;
            return Ok(attr_info_message);
        }
        let requests = plan
            .block_lengths()
            .zip(plan.block_kinds())
            .map(|(length, kind)| {
                NonZeroU64::new(length)
                    .map(|length| {
                        BlockRequest::complete(
                            ReservationRequest::new(
                                length,
                                match kind {
                                    DenseAttrBlockKind::HugeObject => AllocationClass::Raw,
                                    DenseAttrBlockKind::Metadata => AllocationClass::Metadata,
                                },
                            )
                            .for_block(),
                            EncodingDependency::Reservations,
                        )
                    })
                    .ok_or_else(|| FormatError::Internal("a dense attribute block is empty".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let base = self.context.addresses();
        let mut reservations = self
            .plan_blocks(&requests)
            .map_err(ReservationFailure::into_error)?;
        let addresses = reservations
            .reservations()
            .iter()
            .map(|reservation| base.relative(reservation.range().start()))
            .collect::<Result<Vec<_>, _>>()?;
        let placement = plan.place(&addresses)?;
        for block in placement.serialize_blocks()? {
            reservations.write_next(&block?.bytes)?;
        }
        reservations.finish()?;
        Ok(placement
            .attribute_info()
            .serialize(crate::file_writer::OFFSET_WIDTH))
    }

    /// Places prepared attribute storage and appends its header records.
    ///
    /// Places new global heap collections and patches their references before appending
    /// compact records or building dense storage. An empty compact result requires no placement.
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
            EditedAttrs::RebuildCompact(mut compact) => {
                for CompactAttrFixup { range, collections } in compact.fixups {
                    let addrs = self.place_vl_collections(&collections)?;
                    let data = compact.records.get_mut(range).ok_or_else(|| {
                        FormatError::Internal(
                            "a compact attribute fixup extends past its records".into(),
                        )
                    })?;
                    type_builders::patch_vl_refs(data, &addrs);
                }
                region.push_bytes(&compact.records);
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
                    let msg = AttributeName::try_from(name.as_str())?.build_value(value)?;
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
    /// The header region before appending prepared attribute records.
    pub(super) region: OhRegion,
    /// Attribute storage awaiting placement.
    pub(super) attrs: EditedAttrs,
    /// Whether the edit replaces dense storage named by the pre-commit header.
    pub(super) replaces_dense_storage: bool,
}

/// Attribute storage prepared for commit application.
///
/// Compact edits retain records in the header region or provide a complete encoded set.
/// Dense edits provide a complete replacement set.
#[derive(Clone)]
pub(super) enum EditedAttrs {
    /// Variable-length attributes to place and append as inline records.
    Compact(PendingVlAttrs),
    /// A set to rebuild into a fresh fractal heap.
    Dense(DenseAttrEdit),
    /// A complete inline set to append after patching new global heap references.
    RebuildCompact(CompactAttrEdit),
}

/// Encoded inline records with deferred variable-length references.
#[derive(Clone)]
pub(super) struct CompactAttrEdit {
    /// Complete Attribute records in the destination header's layout.
    records: Vec<u8>,
    /// Payload ranges with new global heap collections awaiting placement.
    fixups: Vec<CompactAttrFixup>,
}

/// New global heap collections and their payload range in the encoded records.
#[derive(Clone)]
struct CompactAttrFixup {
    /// The raw-data range in [`CompactAttrEdit::records`].
    range: Range<usize>,
    /// Global heap collections to place before patching the range.
    collections: Vec<Vec<u8>>,
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
/// Compact edits preserve records of attributes whose names are absent from `ops`.
/// New variable-length values are placed during commit application.
///
/// Uses [`AttributePolicy`] for count transitions and defaults. For a compact set, selects
/// dense storage when its count increases above the maximum, the maximum is zero, or a surviving
/// message exceeds the record size limit. A nonempty dense set becomes compact after its count
/// decreases below the minimum, provided every surviving message fits
/// [`OBJECT_HEADER_MESSAGE_MAX`]. Retains dense storage by count when the minimum is zero.
///
/// Reuses stored bodies for surviving attributes absent from `ops` when rebuilding compact
/// storage. Serializes dense replacements in version 3 format. Both rebuilds preserve creation
/// indexes and the next-index counter. An empty result requires no attribute storage.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if an edit cannot be reproduced, a required attribute is
/// absent, creation indexes are exhausted, or conversion to dense storage would prevent
/// reference patching. Returns [`FormatError::InvalidAttributePhaseChange`] if the stored
/// minimum dense count exceeds the maximum compact count. Propagates source-read, parsing,
/// and storage-planning errors.
pub(super) fn plan_attr_ops(
    src: &(impl OriginalSource + ?Sized),
    region: &OhRegion,
    ops: &[&AttrOp],
) -> Result<AttrEdits, Error> {
    let policy = AttributePolicy::from_prefix(region.props().attribute_phase_change)?;
    let dense_now = region_uses_dense_attrs(region)?;
    let current = if dense_now {
        AttributeStorage::Dense
    } else {
        AttributeStorage::Compact
    };
    if current == AttributeStorage::Compact {
        // Detect oversized values before attempting compact editing.
        let mut oversized = false;
        for op in ops {
            if let AttrOp::Set { name, value } = op {
                oversized |= AttributeName::try_from(name.as_str())?
                    .build_value(value)?
                    .serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)?
                    .len()
                    > OBJECT_HEADER_MESSAGE_MAX;
            }
        }
        if !oversized {
            let previous_count = compact_attr_count(region)?;
            let (out, pending_vl) = apply_compact_attr_ops(region, ops)?;
            if policy.select(
                current,
                previous_count,
                compact_attr_count(&out)? + pending_vl.len(),
                true,
            ) == AttributeStorage::Compact
            {
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

    let (mut set, mut next_index) = read_attr_slots(src, region)?;
    let previous_count = set.len();
    let tracked = region.layout().tracks_creation_order();
    for op in ops {
        match op {
            AttrOp::Set { name, value } => {
                let msg = AttributeName::try_from(name.as_str())?.build_value(value)?;
                let collections = value
                    .var_len_strings()
                    .map(type_builders::build_global_heap_collections);
                match set.get_mut(name) {
                    // Preserve the existing creation index on replacement.
                    Some(slot) => {
                        slot.msg = msg;
                        slot.collections = collections;
                        slot.body = None;
                    }
                    None => {
                        let creation_index = tracked.then_some(next_index);
                        if tracked {
                            next_index = bump_creation_index(next_index)?;
                        }
                        set.insert(
                            name.clone(),
                            AttributeSlot {
                                msg,
                                collections,
                                creation_index,
                                body: None,
                            },
                        );
                    }
                }
            }
            AttrOp::Remove { name } => {
                if set.remove(name).is_none() {
                    return Err(Error::EditUnsupported("attribute to remove was not found"));
                }
            }
        }
    }

    if set.is_empty() {
        return Ok(AttrEdits {
            region: strip_attr_messages(region)?,
            attrs: EditedAttrs::default(),
            replaces_dense_storage: dense_now,
        });
    }
    if policy.select(current, previous_count, set.len(), true) == AttributeStorage::Compact {
        for slot in set.values_mut() {
            if slot.body.is_none() {
                slot.body = Some(
                    slot.msg
                        .serialize(crate::file_writer::OFFSET_WIDTH, LENGTH_SIZE)?,
                );
            }
        }
        if set.values().all(|slot| {
            slot.body
                .as_ref()
                .is_some_and(|body| body.len() <= OBJECT_HEADER_MESSAGE_MAX)
        }) {
            return prepare_compact_attrs(region, set, next_index, dense_now);
        }
    }
    let region = strip_attr_messages(region)?;
    let mut attrs = Vec::with_capacity(set.len());
    let mut indices = Vec::with_capacity(set.len());
    let mut vl = Vec::new();
    for (i, slot) in set.into_values().enumerate() {
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
struct AttributeSlot {
    msg: crate::attribute::AttributeMessage,
    /// Global heap collections to place for a new variable-length value.
    collections: Option<Vec<Vec<u8>>>,
    /// The attribute's creation index, or `None` for an untracked object.
    creation_index: Option<u16>,
    /// The encoded message body, absent after an addition or replacement.
    body: Option<Vec<u8>>,
}

/// Reads compact and dense attributes while retaining their encoded bodies.
///
/// Returns the attributes by name and the next creation index. Resolves shared datatype and
/// dataspace fields through `src` while retaining their stored encodings for compact emission.
///
/// # Errors
///
/// Returns [`FormatError::DuplicateAttributeName`] if names repeat, or
/// [`FormatError::UnsupportedMessage`] if an unknown message must be understood for writing.
/// Propagates source-read, parsing, and creation-metadata errors.
fn read_attr_slots(
    src: &(impl OriginalSource + ?Sized),
    region: &OhRegion,
) -> Result<(BTreeMap<String, AttributeSlot>, u16), Error> {
    let resolver =
        SourceResolver::new(src, AccessMode::ReadWrite, crate::file_writer::WIDTHS, None);
    let mut stored = Vec::new();
    let mut bodies = Vec::new();
    let mut visit = |body: &[u8], creation_index| {
        stored.push(StoredAttribute {
            message: AttributeMessage::parse_resolving_with_length_width(
                body,
                crate::file_writer::LENGTH_WIDTH,
                &resolver,
            )?,
            creation_index,
        });
        bodies.push(body.to_vec());
        Ok(())
    };
    let mut position = 0;
    let mut info = None;
    while let Some(record) = region
        .layout()
        .next_message(region, position)
        .map_err(object_header::record_error)?
    {
        if let Some(id) = record.msg_type.unknown_id()
            && record.flags.must_be_understood(AccessMode::ReadWrite)
        {
            return Err(FormatError::UnsupportedMessage(id).into());
        }
        match record.msg_type {
            MessageType::ATTRIBUTE => visit(record.body, record.creation_index)?,
            MessageType::ATTRIBUTE_INFO if info.is_none() => {
                info = Some(AttributeInfoMessage::parse_with_offset_width(
                    record.body,
                    crate::file_writer::OFFSET_WIDTH,
                )?);
            }
            _ => {}
        }
        position = record.body_range.end;
    }
    if let Some(info) = info
        && let Some(heap) = info.fractal_heap_address
    {
        crate::attribute::visit_dense_attribute_messages_from_source(
            &PhysicalSource::new(src).stored(),
            AccessMode::ReadWrite,
            &info,
            heap,
            crate::file_writer::WIDTHS,
            None,
            |_| true,
            &mut visit,
        )?;
    }
    let existing = dense_attr_set(region, stored)?;
    let next = match &existing.creation {
        DenseAttrCreationOrder::Tracked { max, .. } => *max,
        DenseAttrCreationOrder::Untracked => 0,
    };
    let mut slots = BTreeMap::new();
    for (index, (msg, body)) in existing.attrs.into_iter().zip(bodies).enumerate() {
        let slot = AttributeSlot {
            msg,
            body: Some(body),
            collections: None,
            creation_index: existing.creation.index_at(index),
        };
        if let Some(previous) = slots.insert(slot.msg.name.clone(), slot) {
            return Err(FormatError::DuplicateAttributeName(previous.msg.name).into());
        }
    }
    Ok((slots, next))
}

/// Prepares inline records and Attribute Info metadata for the complete set.
///
/// Uses the prepared bodies and creation indexes, preserving the header's prefix properties
/// and unrelated records. Stores undefined heap and index addresses in Attribute Info and
/// preserves its record metadata. Defers new global heap references until placement.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a body exceeds [`OBJECT_HEADER_MESSAGE_MAX`]. Returns
/// [`FormatError::InvalidAttributeCreationCounter`] or
/// [`FormatError::DuplicateAttributeCreationIndex`] for invalid creation metadata. Propagates
/// record-splicing and allocation errors.
fn prepare_compact_attrs(
    original: &OhRegion,
    slots: BTreeMap<String, AttributeSlot>,
    next: u16,
    replaces_dense_storage: bool,
) -> Result<AttrEdits, Error> {
    let layout = original.layout();
    let mut records = Vec::new();
    let mut fixups = Vec::new();
    let mut indices = BTreeSet::new();
    for slot in slots.into_values() {
        let body = slot.body.ok_or_else(|| {
            FormatError::Internal("a compact attribute has no prepared body".into())
        })?;
        if body.len() > OBJECT_HEADER_MESSAGE_MAX {
            return Err(Error::EditUnsupported(
                "attribute is too large to encode in place",
            ));
        }
        let creation_index = if layout.tracks_creation_order() {
            let index = slot.creation_index.ok_or_else(|| {
                FormatError::Internal("a tracked compact attribute has no creation index".into())
            })?;
            if index >= next {
                return Err(FormatError::InvalidAttributeCreationCounter { index, next }.into());
            }
            if !indices.insert(index) {
                return Err(FormatError::DuplicateAttributeCreationIndex(u32::from(index)).into());
            }
            index
        } else {
            0
        };
        let record_len = layout.prefix_len() + body.len();
        records
            .try_reserve(record_len)
            .map_err(|source| FormatError::AllocationFailed {
                values: record_len,
                source,
            })?;
        layout.write_record(
            &mut records,
            MessageType::ATTRIBUTE,
            MessageFlags::NONE,
            creation_index,
            &body,
        );
        if let Some(collections) = slot.collections {
            let end = records.len();
            let start = end.checked_sub(slot.msg.raw_data.len()).ok_or_else(|| {
                FormatError::Internal("a compact attribute payload exceeds its record".into())
            })?;
            fixups.push(CompactAttrFixup {
                range: start..end,
                collections,
            });
        }
    }
    let info = AttributeInfoMessage {
        max_creation_index: layout.tracks_creation_order().then_some(next),
        indexes_creation_order: layout.indexes_creation_order(),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    }
    .serialize(crate::file_writer::OFFSET_WIDTH);
    let mut region = strip_attr_messages(original)?;
    if let Some((start, body, _)) = find_attribute_info(original)? {
        let splice = RecordSplice::first(
            layout,
            &original[start..body.end],
            MessageType::ATTRIBUTE_INFO,
        )?
        .ok_or_else(|| {
            FormatError::Internal(
                "a parsed Attribute Info record is absent from its source range".into(),
            )
        })?;
        let mut encoded = Vec::new();
        splice.replace_body_into(&mut encoded, &info)?;
        region.push_bytes(&encoded);
    } else {
        region.push(MessageType::ATTRIBUTE_INFO, &info);
    }
    Ok(AttrEdits {
        region,
        attrs: EditedAttrs::RebuildCompact(CompactAttrEdit { records, fixups }),
        replaces_dense_storage,
    })
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
/// Resolves heap addresses using the opened source's effective address context. The caller
/// checks whether the resulting set can be rebuilt after applying any edits.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the effective base exceeds the source length.
/// Propagates header parsing, attribute parsing, and source-read errors.
pub(super) fn read_object_attrs(
    src: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    addr: u64,
) -> Result<Vec<crate::attribute::StoredAttribute>, Error> {
    let base = src.addresses();
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
    crate::attribute::extract_stored_attributes_from_source(
        src,
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
    let body = AttributeName::try_from(name)?
        .build_value(value)?
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

#[cfg(test)]
mod tests;
