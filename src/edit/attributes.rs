//! Attribute edit preparation and compact and dense storage placement.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_format::__private::{
    DenseAttrBlockKind, DenseAttrFixup, DenseAttrPlan, PreparedAttributeMessage,
    PreparedDenseAttrs, RecordSplice,
};
use hdf5_pure_space::__private::{AllocationClass, ReservationRequest};

use crate::access_mode::AccessMode;
use crate::attribute::{AttributeMessage, StoredAttribute};
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_policy::AttributePolicy;
use crate::attribute_policy::AttributeStorage;
use crate::attribute_request::AttributeName;
use crate::convert::Narrow;
use crate::datatype::Datatype;
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_writer::DenseAttrCreationOrder;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::shared_message::SourceResolver;
use crate::source::OriginalSource;
use crate::source::PhysicalSource;
use crate::source::SourceMetadata;
use crate::type_builders::{self, AttrValue};
use crate::width::{FormatWidths, OffsetWidth};

use super::WriteEngine;
use super::block::{BlockRequest, EncodingDependency};
use super::object_header;
use super::object_header::OhRegion;
use super::reservation::ReservationFailure;
use super::staging::StagedOp;

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

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
        let Some(plan) = set.into_plan()? else {
            return Ok(());
        };
        let attr_info_message = self.place_dense_attrs(plan)?;
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
                NonZeroByteLength::new(length)
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
            .serialize(self.context.widths().offsets)?)
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
            EditedAttrs::Compact => Ok(()),
            EditedAttrs::RebuildCompact(mut compact) => {
                for CompactAttrFixup { range, collections } in compact.fixups {
                    let addrs = self.place_vl_collections(&collections)?;
                    let data = compact.records.get_mut(range).ok_or_else(|| {
                        FormatError::Internal(
                            "a compact attribute fixup extends past its records".into(),
                        )
                    })?;
                    type_builders::patch_vl_refs_with_width(region.widths().offsets, data, &addrs)?;
                }
                region.push_bytes(&compact.records);
                Ok(())
            }
            EditedAttrs::Dense(mut dense) => {
                for (idx, collections) in std::mem::take(&mut dense.vl) {
                    let addrs = self.place_vl_collections(&collections)?;
                    type_builders::patch_vl_refs_with_width(
                        region.widths().offsets,
                        &mut dense.set.attrs[idx].raw_data,
                        &addrs,
                    )?;
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

/// A rewritten header region and the attribute storage prepared for commit application.
pub(super) struct AttrEdits {
    /// The header region before appending prepared attribute records.
    pub(super) region: OhRegion,
    /// Attribute storage awaiting placement.
    pub(super) attrs: EditedAttrs,
    /// Parsed original reference-bearing attributes replaced or removed by the edit.
    pub(super) predecessors: AttributePredecessors,
    /// Whether the edit replaces dense storage named by the pre-commit header.
    pub(super) replaces_dense_storage: bool,
}

/// Parsed original attributes replaced or removed by an edit.
///
/// Only attributes whose datatypes can hold file addresses are retained. Each original value
/// is retained once across repeated operations on its name. Allocation ownership and reference
/// absence require separate proofs.
#[derive(Default)]
pub(super) struct AttributePredecessors {
    attributes: Vec<AttributeMessage>,
}

impl AttributePredecessors {
    /// Collects original compact attributes for the names in `ops`.
    ///
    /// Skips shared records and attributes that cannot be parsed. Retains the parsed prefix
    /// if record traversal fails.
    fn compact(src: &(impl OriginalSource + ?Sized), region: &OhRegion, ops: &[&AttrOp]) -> Self {
        let resolver = SourceResolver::new(src, AccessMode::ReadWrite, region.widths(), None);
        let names: BTreeSet<&str> = ops
            .iter()
            .map(|op| match op {
                AttrOp::Set { name, .. } | AttrOp::Remove { name } => name.as_str(),
            })
            .collect();
        let mut predecessors = Self::default();
        let mut position = 0;
        while let Ok(Some(record)) = region.layout().next_message(region, position) {
            if record.msg_type == MessageType::ATTRIBUTE
                && !record.flags.is_shared()
                && let Ok(name) = AttributeMessage::parse_name(record.body)
                && names.contains(name.as_str())
                && let Ok(attribute) = AttributeMessage::parse_resolving_with_widths(
                    record.body,
                    region.widths(),
                    &resolver,
                )
            {
                predecessors.push(attribute);
            }
            position = record.body_range.end;
        }
        predecessors
    }

    /// Retains `attribute` if its datatype can hold a file address.
    fn push(&mut self, attribute: AttributeMessage) {
        if crate::datatype::datatype_holds_file_address(&attribute.datatype) {
            self.attributes.push(attribute);
        }
    }
}

impl IntoIterator for AttributePredecessors {
    type Item = AttributeMessage;
    type IntoIter = std::vec::IntoIter<AttributeMessage>;

    fn into_iter(self) -> Self::IntoIter {
        self.attributes.into_iter()
    }
}

/// Attribute storage prepared for commit application.
///
/// Compact edits combine retained inline records with prepared records to append.
/// Dense edits provide a complete replacement set.
#[derive(Clone)]
pub(super) enum EditedAttrs {
    /// Inline records already present in the header region, with no pending placement.
    Compact,
    /// A set to rebuild into a fresh fractal heap.
    Dense(DenseAttrEdit),
    /// Inline records to append after patching new global heap references.
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
        Self::Compact
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
    /// Returns the encoding, metadata, and update errors of [`PreparedDenseAttrs::with_widths`].
    pub(super) fn new(
        widths: FormatWidths,
        attrs: Vec<AttributeMessage>,
        creation: DenseAttrCreationOrder,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            prepared: Some(PreparedDenseAttrs::with_widths(
                widths, &attrs, creation, fixups,
            )?),
            attrs,
        })
    }

    /// Resolves deferred body updates into a dense-storage plan.
    ///
    /// Returns `None` for an empty set.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a nonempty set lacks prepared geometry. Propagates
    /// finalization errors from [`crate::file_writer::finalize_dense_attrs`].
    pub(super) fn into_plan(self) -> Result<Option<DenseAttrPlan>, FormatError> {
        let Self { attrs, prepared } = self;
        if attrs.is_empty() {
            return Ok(None);
        }
        crate::file_writer::finalize_dense_attrs(
            prepared.ok_or_else(|| {
                FormatError::Internal("a dense set has no prepared geometry".into())
            })?,
            &attrs,
        )
        .map(Some)
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
/// Applies operations in staged order and reports the first operation error. Compact edits
/// preserve records of attributes whose names are absent from `ops`. New variable-length
/// values are placed during commit application.
/// Retains parsed original reference-bearing values in [`AttrEdits::predecessors`] for
/// collection analysis.
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
        return plan_compact_batch(src, region, ops, policy);
    }

    // Reject flagged records before resolving attributes for a dense rebuild.
    if region_has_shared_attr(region)? {
        return Err(Error::EditUnsupported(SHARED_ATTRIBUTE_MESSAGE));
    }

    let (mut set, mut next_index) = read_attr_slots(src, region)?;
    let previous_count = set.len();
    let tracked = region.layout().tracks_creation_order();
    let mut predecessors = AttributePredecessors::default();
    let mut touched = BTreeSet::new();
    for op in ops {
        let name = match op {
            AttrOp::Set { name, .. } | AttrOp::Remove { name } => name,
        };
        let first_edit = touched.insert(name.as_str());
        match op {
            AttrOp::Set { name, value } => {
                let PreparedAttributeValue {
                    attribute,
                    collections,
                } = prepare_attribute_value(region.widths(), name, value)?;
                let attribute = SlotAttribute::Prepared(attribute);
                match set.get_mut(name) {
                    // Preserve the existing creation index on replacement.
                    Some(slot) => {
                        let original = std::mem::replace(&mut slot.attribute, attribute);
                        if first_edit {
                            predecessors.push(original.into_message());
                        }
                        slot.collections = collections;
                    }
                    None => {
                        let creation_index = tracked.then_some(next_index);
                        if tracked {
                            next_index = bump_creation_index(next_index)?;
                        }
                        set.insert(
                            name.clone(),
                            AttributeSlot {
                                attribute,
                                collections,
                                creation_index,
                            },
                        );
                    }
                }
            }
            AttrOp::Remove { name } => {
                if let Some(original) = set.remove(name) {
                    if first_edit {
                        predecessors.push(original.attribute.into_message());
                    }
                } else {
                    return Err(Error::EditUnsupported("attribute to remove was not found"));
                }
            }
        }
    }

    if set.is_empty() {
        return Ok(AttrEdits {
            region: strip_attr_messages(region)?,
            attrs: EditedAttrs::default(),
            predecessors,
            replaces_dense_storage: dense_now,
        });
    }
    if policy.select(current, previous_count, set.len(), true) == AttributeStorage::Compact
        && set
            .values()
            .all(|slot| slot.attribute.body().len() <= OBJECT_HEADER_MESSAGE_MAX)
    {
        return prepare_compact_attrs(region, set, next_index, dense_now, predecessors);
    }
    prepare_dense_attrs(region, set, next_index, dense_now, predecessors)
}

/// An attribute body encoded at the destination widths with unresolved heap collections.
struct PreparedAttributeValue {
    attribute: PreparedAttributeMessage,
    collections: Option<Vec<Vec<u8>>>,
}

/// Encodes an attribute value and prepares its variable-length storage at `widths`.
///
/// Retains heap collections separately for address resolution during commit application.
///
/// # Errors
///
/// Returns attribute-name and value-encoding errors, collection-size errors from
/// [`crate::file_writer::validate_collections`], width-conversion errors from
/// [`crate::file_writer::prepare_attribute_widths`], or errors from
/// [`PreparedAttributeMessage::with_widths`].
fn prepare_attribute_value(
    widths: FormatWidths,
    name: &str,
    value: &AttrValue,
) -> Result<PreparedAttributeValue, Error> {
    let mut message = AttributeName::try_from(name)?.build_value(value)?;
    let collections = value
        .var_len_strings()
        .map(type_builders::build_global_heap_collections)
        .transpose()?;
    if let Some(collections) = &collections {
        crate::file_writer::validate_collections(widths, collections)?;
    }
    crate::file_writer::prepare_attribute_widths(widths, core::slice::from_mut(&mut message), &[])?;
    Ok(PreparedAttributeValue {
        attribute: PreparedAttributeMessage::with_widths(widths, message)?,
        collections,
    })
}

/// Prepares the complete dense set from stored or encoded attribute values.
///
/// # Errors
///
/// Returns the metadata, geometry, and update errors of [`PreparedDenseAttrs::from_prepared`].
/// Propagates record parsing and attribute encoding errors.
fn prepare_dense_attrs(
    original: &OhRegion,
    set: BTreeMap<String, AttributeSlot>,
    next_index: u16,
    replaces_dense_storage: bool,
    predecessors: AttributePredecessors,
) -> Result<AttrEdits, Error> {
    let tracked = original.layout().tracks_creation_order();
    let region = strip_attr_messages(original)?;
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
        attrs.push(slot.attribute.into_prepared(original.widths())?);
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
    let (attrs, prepared) = PreparedDenseAttrs::from_prepared(
        attrs,
        creation,
        vl.iter().map(|(attribute, _)| DenseAttrFixup::GlobalHeap {
            attribute: *attribute,
        }),
    )?;
    Ok(AttrEdits {
        region,
        attrs: EditedAttrs::Dense(DenseAttrEdit {
            set: DenseAttrSet {
                attrs,
                prepared: Some(prepared),
            },
            vl,
        }),
        predecessors,
        replaces_dense_storage,
    })
}

/// Applies ordered edits before selecting storage and reconstructing compact records.
///
/// # Errors
///
/// Returns the operation and storage-planning errors documented by [`plan_attr_ops`].
fn plan_compact_batch(
    src: &(impl OriginalSource + ?Sized),
    original: &OhRegion,
    ops: &[&AttrOp],
    policy: AttributePolicy,
) -> Result<AttrEdits, Error> {
    if ops.is_empty() {
        let count = compact_attr_count(original)?;
        if policy.select(AttributeStorage::Compact, count, count, true) == AttributeStorage::Compact
        {
            return Ok(AttrEdits {
                region: original.clone(),
                attrs: EditedAttrs::default(),
                predecessors: AttributePredecessors::default(),
                replaces_dense_storage: false,
            });
        }
    }
    let mut batch = CompactBatch::parse(original)?;
    let previous_count = batch.count;
    for (order, op) in ops.iter().enumerate() {
        batch.apply(op, order)?;
    }
    let predecessors = AttributePredecessors::compact(src, original, ops);
    let compact = batch.slots.iter().all(|slot| match &slot.value {
        CompactValue::Prepared { attribute, .. } => {
            attribute.body().len() <= OBJECT_HEADER_MESSAGE_MAX
        }
        _ => true,
    });
    if !batch.oversized
        && policy.select(
            AttributeStorage::Compact,
            previous_count,
            batch.count,
            compact,
        ) == AttributeStorage::Compact
    {
        return batch.into_compact(predecessors);
    }
    batch.into_replacement(src, policy, previous_count, predecessors)
}

/// Original record ranges and the latest staged value of each attribute name.
///
/// Creation indexes are updated as operations are applied. Unchanged original records are copied
/// during reconstruction, and new global heap references are patched during placement.
struct CompactBatch<'a> {
    original: &'a OhRegion,
    records: Vec<CompactRecord>,
    names: BTreeMap<String, usize>,
    slots: Vec<CompactSlot>,
    /// The number of live records using each creation index.
    indices: BTreeMap<u16, usize>,
    counter: Option<u16>,
    /// Whether an operation assigned a new creation index.
    assigned: bool,
    /// Whether any staged value requires the full attribute-decoding path.
    oversized: bool,
    count: usize,
    /// The start of trailing bytes after the final complete record.
    tail: usize,
}

impl<'a> CompactBatch<'a> {
    /// Parses record boundaries and indexes attribute names and creation indexes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for malformed records or flagged attributes.
    /// Propagates attribute-name parsing errors.
    fn parse(original: &'a OhRegion) -> Result<Self, Error> {
        let mut batch = Self {
            original,
            records: Vec::new(),
            names: BTreeMap::new(),
            slots: Vec::new(),
            indices: BTreeMap::new(),
            counter: find_attribute_info(original)?
                .and_then(|(_, _, info)| info.max_creation_index),
            assigned: false,
            oversized: false,
            count: 0,
            tail: 0,
        };
        let mut position = 0;
        while let Some((kind, body, end)) = original.next_message(position)? {
            let slot = if kind == MessageType::ATTRIBUTE {
                let name = parse_compact_attr_name(original, position, body, end)?;
                let index = match batch.names.get(&name).copied() {
                    Some(index) => index,
                    None => {
                        let index = batch.slots.len();
                        batch.names.insert(name, index);
                        batch.slots.push(CompactSlot {
                            value: CompactValue::Original,
                            indices: Vec::new(),
                            count: 0,
                            touched: false,
                            order: 0,
                        });
                        index
                    }
                };
                let slot = &mut batch.slots[index];
                if let Some(creation) = original.creation_index(position) {
                    slot.indices.push(creation);
                    *batch.indices.entry(creation).or_default() += 1;
                }
                slot.count += 1;
                batch.count += 1;
                Some(index)
            } else {
                None
            };
            batch.records.push(CompactRecord {
                range: position..end,
                kind,
                body,
                slot,
            });
            position = end;
        }
        batch.tail = position;
        Ok(batch)
    }

    /// Applies an operation and retains its position for final record ordering.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a removed name is absent or creation indexes are
    /// exhausted. Propagates attribute-name and encoding errors.
    fn apply(&mut self, operation: &AttrOp, order: usize) -> Result<(), Error> {
        let name = match operation {
            AttrOp::Set { name, .. } | AttrOp::Remove { name } => name,
        };
        let prepared = match operation {
            AttrOp::Set { name, value } => Some(prepare_attribute_value(
                self.original.widths(),
                name,
                value,
            )?),
            AttrOp::Remove { .. } => None,
        };
        let index = match self.names.get(name).copied() {
            Some(index) => index,
            None => {
                if prepared.is_none() {
                    return Err(Error::EditUnsupported("attribute to remove was not found"));
                }
                let index = self.slots.len();
                self.names.insert(name.clone(), index);
                self.slots.push(CompactSlot {
                    value: CompactValue::Removed,
                    indices: Vec::new(),
                    count: 0,
                    touched: false,
                    order,
                });
                index
            }
        };
        let slot = &mut self.slots[index];
        match prepared {
            Some(PreparedAttributeValue {
                attribute,
                collections,
            }) => {
                self.oversized |= attribute.body().len() > OBJECT_HEADER_MESSAGE_MAX;
                if slot.count == 0 && self.original.layout().tracks_creation_order() {
                    let derived = self
                        .indices
                        .last_key_value()
                        .map_or(0, |(&index, _)| u32::from(index) + 1);
                    let index = u32::from(self.counter.unwrap_or(0)).max(derived).narrow_or_else(|| {
                        Error::EditUnsupported("an object has assigned every attribute creation index its header can record")
                    })?;
                    self.counter = Some(bump_creation_index(index)?);
                    self.assigned = true;
                    slot.indices.push(index);
                    *self.indices.entry(index).or_default() += 1;
                } else if slot.indices.len() > 1 {
                    for index in slot.indices.drain(1..) {
                        if let Some(count) = self.indices.get_mut(&index) {
                            *count -= 1;
                            if *count == 0 {
                                self.indices.remove(&index);
                            }
                        }
                    }
                }
                self.count = self.count - slot.count + 1;
                slot.count = 1;
                slot.value = CompactValue::Prepared {
                    attribute: Box::new(attribute),
                    collections,
                };
            }
            None => {
                if slot.count == 0 {
                    return Err(Error::EditUnsupported("attribute to remove was not found"));
                }
                self.count -= slot.count;
                slot.count = 0;
                slot.value = CompactValue::Removed;
                for index in slot.indices.drain(..) {
                    if let Some(count) = self.indices.get_mut(&index) {
                        *count -= 1;
                        if *count == 0 {
                            self.indices.remove(&index);
                        }
                    }
                }
            }
        }
        slot.touched = true;
        slot.order = order;
        Ok(())
    }

    /// Copies unchanged records and prepares the surviving replacements for inline storage.
    ///
    /// # Errors
    ///
    /// Propagates Attribute Info record-splicing errors.
    fn into_compact(self, predecessors: AttributePredecessors) -> Result<AttrEdits, Error> {
        let Self {
            original,
            records,
            names: _,
            slots,
            indices: _,
            counter,
            assigned,
            oversized: _,
            count: _,
            tail,
        } = self;
        let layout = original.layout();
        let mut bytes = Vec::with_capacity(original.len());
        let mut has_info = false;
        for CompactRecord {
            range,
            kind,
            body,
            slot,
        } in records
        {
            if slot.is_some_and(|index| slots[index].touched) {
                continue;
            }
            if kind == MessageType::ATTRIBUTE_INFO
                && assigned
                && let Ok(mut info) = AttributeInfoMessage::parse_with_offset_width(
                    &original[body..range.end],
                    original.widths().offsets,
                )
            {
                info.max_creation_index = counter;
                info.indexes_creation_order |= layout.indexes_creation_order();
                let splice =
                    RecordSplice::first(layout, &original[range], MessageType::ATTRIBUTE_INFO)?
                        .ok_or_else(|| {
                            FormatError::Internal(
                                "a parsed Attribute Info record is absent from its source range"
                                    .into(),
                            )
                        })?;
                splice
                    .replace_body_into(&mut bytes, &info.serialize(original.widths().offsets)?)?;
                has_info = true;
            } else {
                bytes.extend_from_slice(&original[range]);
            }
        }
        let mut slots: Vec<_> = slots
            .into_iter()
            .filter(|slot| slot.touched && slot.count != 0)
            .collect();
        slots.sort_unstable_by_key(|slot| slot.order);
        let mut records = Vec::new();
        let mut fixups = Vec::new();
        for slot in slots {
            let CompactValue::Prepared {
                attribute,
                collections,
            } = slot.value
            else {
                return Err(FormatError::Internal(
                    "a changed compact attribute has no prepared value".into(),
                )
                .into());
            };
            layout.write_record(
                &mut records,
                MessageType::ATTRIBUTE,
                MessageFlags::NONE,
                slot.indices.first().copied().unwrap_or(0),
                attribute.body(),
            );
            if let Some(collections) = collections {
                let end = records.len();
                let start = end - attribute.message().raw_data.len();
                fixups.push(CompactAttrFixup {
                    range: start..end,
                    collections,
                });
            }
        }
        records.extend_from_slice(&original[tail..]);
        let mut region = original.with_bytes(bytes);
        if assigned && !has_info {
            let info = AttributeInfoMessage {
                max_creation_index: counter,
                indexes_creation_order: layout.indexes_creation_order(),
                fractal_heap_address: None,
                btree_name_index_address: None,
                btree_creation_order_address: None,
            };
            region.push(
                MessageType::ATTRIBUTE_INFO,
                &info.serialize(original.widths().offsets)?,
            );
        }
        Ok(AttrEdits {
            region,
            attrs: EditedAttrs::RebuildCompact(CompactAttrEdit { records, fixups }),
            predecessors,
            replaces_dense_storage: false,
        })
    }

    /// Decodes original attributes and prepares the final compact or dense set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if dense storage would prevent reference patching.
    /// Propagates source-read, attribute parsing, and storage-planning errors.
    fn into_replacement(
        self,
        src: &(impl OriginalSource + ?Sized),
        policy: AttributePolicy,
        previous_count: usize,
        predecessors: AttributePredecessors,
    ) -> Result<AttrEdits, Error> {
        let (mut stored, original_next) = read_attr_slots(src, self.original)?;
        let mut slots: Vec<_> = self.slots.into_iter().map(Some).collect();
        let mut set = BTreeMap::new();
        for (name, index) in self.names {
            let slot = slots[index].take().ok_or_else(|| {
                FormatError::Internal("a compact attribute slot was consumed twice".into())
            })?;
            let slot = match slot.value {
                CompactValue::Original => stored.remove(&name).ok_or_else(|| {
                    FormatError::Internal("an original compact attribute is absent".into())
                })?,
                CompactValue::Prepared {
                    attribute,
                    collections,
                } => AttributeSlot {
                    attribute: SlotAttribute::Prepared(*attribute),
                    collections,
                    creation_index: slot.indices.first().copied(),
                },
                CompactValue::Removed => continue,
            };
            set.insert(name, slot);
        }
        if set.is_empty() {
            return Ok(AttrEdits {
                region: strip_attr_messages(self.original)?,
                attrs: EditedAttrs::default(),
                predecessors,
                replaces_dense_storage: false,
            });
        }
        let next = self.counter.unwrap_or(original_next).max(original_next);
        if policy.select(AttributeStorage::Compact, previous_count, set.len(), true)
            == AttributeStorage::Compact
            && set
                .values()
                .all(|slot| slot.attribute.body().len() <= OBJECT_HEADER_MESSAGE_MAX)
        {
            return prepare_compact_attrs(self.original, set, next, false, predecessors);
        }
        for slot in set.values() {
            if crate::reference_patch::attribute_references_are_repointable(
                &slot.attribute.message().datatype,
            ) {
                return Err(Error::EditUnsupported(
                    REFERENCE_ATTRIBUTE_WOULD_LEAVE_THE_HEADER,
                ));
            }
        }
        prepare_dense_attrs(self.original, set, next, false, predecessors)
    }
}

/// A complete original record and its optional attribute-name slot.
struct CompactRecord {
    range: Range<usize>,
    kind: MessageType,
    body: usize,
    slot: Option<usize>,
}

/// The latest value and live creation indexes for one attribute name.
struct CompactSlot {
    value: CompactValue,
    indices: Vec<u16>,
    /// The number of live records for the name, including duplicate original records.
    count: usize,
    touched: bool,
    /// The position of the latest operation on this name.
    order: usize,
}

/// An unchanged original value, a prepared replacement, or a removed attribute.
enum CompactValue {
    Original,
    Prepared {
        attribute: Box<PreparedAttributeMessage>,
        collections: Option<Vec<Vec<u8>>>,
    },
    Removed,
}

/// Applies compact edits for tests whose values require no global heap placement.
///
/// # Errors
///
/// Returns the parsing and operation errors of [`CompactBatch::parse`] and
/// [`CompactBatch::apply`]. Returns [`FormatError::Internal`] if the prepared result requires
/// heap placement. Propagates record-preparation errors from [`CompactBatch::into_compact`].
#[cfg(test)]
pub(super) fn edit_compact_region(region: &OhRegion, ops: &[AttrOp]) -> Result<OhRegion, Error> {
    let mut batch = CompactBatch::parse(region)?;
    for (order, operation) in ops.iter().enumerate() {
        batch.apply(operation, order)?;
    }
    let mut edit = batch.into_compact(AttributePredecessors::default())?;
    let EditedAttrs::RebuildCompact(compact) = edit.attrs else {
        return Err(FormatError::Internal("a compact batch returned dense storage".into()).into());
    };
    if !compact.fixups.is_empty() {
        return Err(
            FormatError::Internal("a compact test batch requires heap placement".into()).into(),
        );
    }
    edit.region.push_bytes(&compact.records);
    Ok(edit.region)
}

/// One attribute, its creation index, and any global heap collections awaiting placement.
struct AttributeSlot {
    attribute: SlotAttribute,
    /// Global heap collections to place for a new variable-length value.
    collections: Option<Vec<Vec<u8>>>,
    /// The attribute's creation index, or `None` for an untracked object.
    creation_index: Option<u16>,
}

/// A decoded attribute paired with its stored or prepared body.
enum SlotAttribute {
    Prepared(PreparedAttributeMessage),
    Stored {
        message: AttributeMessage,
        body: Vec<u8>,
    },
}

impl SlotAttribute {
    fn message(&self) -> &AttributeMessage {
        match self {
            Self::Prepared(attribute) => attribute.message(),
            Self::Stored { message, .. } => message,
        }
    }

    fn body(&self) -> &[u8] {
        match self {
            Self::Prepared(attribute) => attribute.body(),
            Self::Stored { body, .. } => body,
        }
    }

    fn into_message(self) -> AttributeMessage {
        match self {
            Self::Prepared(attribute) => attribute.into_parts().0,
            Self::Stored { message, .. } => message,
        }
    }

    /// Returns the prepared encoding, serializing a stored message when needed.
    ///
    /// # Errors
    ///
    /// Returns the encoding errors documented by [`PreparedAttributeMessage::with_widths`].
    fn into_prepared(self, widths: FormatWidths) -> Result<PreparedAttributeMessage, FormatError> {
        match self {
            Self::Prepared(attribute) => Ok(attribute),
            Self::Stored { message, .. } => PreparedAttributeMessage::with_widths(widths, message),
        }
    }
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
    let resolver = SourceResolver::new(src, AccessMode::ReadWrite, region.widths(), None);
    let mut stored = Vec::new();
    let mut bodies = Vec::new();
    let mut visit = |body: &[u8], creation_index| {
        stored.push(StoredAttribute {
            message: AttributeMessage::parse_resolving_with_widths(
                body,
                region.widths(),
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
                    region.widths().offsets,
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
            region.widths(),
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
            attribute: SlotAttribute::Stored { message: msg, body },
            collections: None,
            creation_index: existing.creation.index_at(index),
        };
        if let Some(previous) = slots.insert(slot.attribute.message().name.clone(), slot) {
            return Err(FormatError::DuplicateAttributeName(
                previous.attribute.into_message().name,
            )
            .into());
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
    predecessors: AttributePredecessors,
) -> Result<AttrEdits, Error> {
    let layout = original.layout();
    let mut records = Vec::new();
    let mut fixups = Vec::new();
    let mut indices = BTreeSet::new();
    for slot in slots.into_values() {
        let body = slot.attribute.body();
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
            body,
        );
        if let Some(collections) = slot.collections {
            let end = records.len();
            let start = end
                .checked_sub(slot.attribute.message().raw_data.len())
                .ok_or_else(|| {
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
    .serialize(original.widths().offsets)?;
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
        predecessors,
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
            && attribute_info_is_dense(region.widths().offsets, &region[body..body_end])
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
/// checks whether the resulting set can be rebuilt after applying any edits. Calls `inspect`
/// after resolving each datatype and before parsing its dataspace or copying its values.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the effective base exceeds the source length.
/// Propagates header parsing, attribute parsing, source-read, and inspection errors.
pub(super) fn read_object_attrs(
    src: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    addr: u64,
    widths: FormatWidths,
    inspect: impl FnMut(&Datatype) -> Result<(), FormatError>,
) -> Result<Vec<crate::attribute::StoredAttribute>, Error> {
    let base = src.addresses();
    let header = ObjectHeader::parse_from_source_with_widths(
        &SourceMetadata(src),
        access_mode,
        addr,
        widths,
        base.effective_base(),
    )?;
    if base.effective_base().get() > src.len() {
        return Err(Error::EditUnsupported(
            "this file's userblock is larger than the file itself",
        ));
    }
    crate::attribute::extract_stored_attributes_with(
        src,
        access_mode,
        &header,
        widths,
        // Read attributes without a shared-message table.
        None,
        inspect,
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
pub(crate) fn attribute_info_is_dense(offsets: OffsetWidth, body: &[u8]) -> bool {
    match crate::attribute_info::AttributeInfoMessage::parse_with_offset_width(body, offsets) {
        Ok(ai) => ai.fractal_heap_address.is_some(),
        Err(_) => true,
    }
}

/// Returns the first matching attribute's creation index.
///
/// Returns `None` if the header is untracked or no matching record exists.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or an Attribute
/// record has nonzero flags. Returns [`Error::Format`] if a name cannot be parsed.
#[cfg(test)]
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
                region.widths().offsets,
            )
        {
            return Ok(Some((p, body..body_end, info)));
        }
        p = body_end;
    }
    Ok(None)
}

/// Serializes compact Attribute Info metadata at the region's offset width.
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
    Ok(info.serialize(region.widths().offsets)?)
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
#[cfg(test)]
fn encode_attr_body(name: &str, value: &AttrValue) -> Result<Vec<u8>, Error> {
    debug_assert!(
        value.var_len_strings().is_none(),
        "a compact test value must not require global heap placement"
    );
    let body = AttributeName::try_from(name)?
        .build_value(value)?
        .serialize(
            crate::file_writer::OFFSET_WIDTH,
            crate::file_writer::LENGTH_SIZE,
        )?;
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
