//! Dense attribute messages, fractal heaps, and version 2 B-tree indexes.
//!
//! Plans retain independent file address and length widths. Attribute messages are paired
//! with creation indexes before heap and index planning.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::__private as format_boundary;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_message::{AttributeMessage, PreparedAttributeMessage};
use crate::btree::v2::AttributeRecord;
use crate::btree::v2::BTREE_V2_ATTRIBUTE_CREATION_ORDER;
use crate::btree::v2::BTREE_V2_ATTRIBUTE_NAME;
use crate::btree::v2::BTREE_V2_HUGE_OBJECT;
use crate::btree::v2::BTREE_V2_HUGE_OBJECT_DIRECT;
use crate::btree::v2::HugeObjectDirectRecord;
use crate::btree::v2::HugeObjectRecord;
use crate::btree::v2::write::BTREE_V2_NODE_SIZE;
use crate::btree::v2::write::BTreeV2Plan;
use crate::btree::v2::write::PreparedHeapId;
use crate::bytes;
use crate::convert::Narrow;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::fractal_heap::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::FractalHeapIdLayout;
use crate::fractal_heap_write::ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES;
use crate::fractal_heap_write::ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE;
use crate::fractal_heap_write::ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;
use crate::fractal_heap_write::ATTRIBUTE_HEAP_MAX_HEAP_SPACE;
use crate::fractal_heap_write::ATTRIBUTE_HEAP_START_ROOT_ROWS;
use crate::fractal_heap_write::ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE;
use crate::fractal_heap_write::ATTRIBUTE_HEAP_TABLE_WIDTH;
use crate::fractal_heap_write::ManagedHeapPlan;
use crate::fractal_heap_write::ManagedHeapPlanError;
use crate::message_flags::MessageFlags;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

mod placement;

pub use placement::{DenseAttrBlock, DenseAttrPlacement};

// ---- Dense attribute blob ----

/// Serialized dense attribute storage and its object-header message.
pub struct DenseAttrBlob {
    /// The serialized [`AttributeInfoMessage`] for the object header.
    pub attr_info_message: Vec<u8>,
    /// The heap header, managed blocks, indexes, and huge-object payloads.
    pub blob: Vec<u8>,
}

/// Bits of heap offset the dense attribute heap declares (its "Maximum Heap
/// Size"), and the byte width that implies for a block offset.
const DENSE_ATTR_MAX_HEAP_SIZE_BITS: u16 = ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;
const DENSE_ATTR_BLOCK_OFFSET_BYTES: usize = ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES;

/// The managed-object size limit of the default eight-byte-width attribute heap.
///
/// The writer uses an eight-byte version 0 managed heap ID with a flag byte, a five-byte
/// offset, and a two-byte length. Larger messages are stored and indexed as huge objects,
/// as defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
pub const DENSE_ATTR_MAX_MANAGED_OBJECT: usize =
    format_boundary::attribute_heap_max_managed_object(DENSE_ATTR_OFFSET_WIDTH);

/// The length of the heap IDs of the attribute heaps [`build_dense_attrs`] writes.
const DENSE_ATTR_HEAP_ID_LENGTH: u16 = 8;

/// One name-index B-tree v2 record as [`build_dense_attrs`] writes it.
const DENSE_ATTR_BTREE_RECORD: u16 = AttributeRecord::name_record_size(DENSE_ATTR_HEAP_ID_LENGTH);

/// One creation-order B-tree v2 record as [`build_dense_attrs`] writes it.
const DENSE_ATTR_CORDER_BTREE_RECORD: u16 =
    AttributeRecord::creation_order_record_size(DENSE_ATTR_HEAP_ID_LENGTH);

/// Creation indexes and the counter recorded for a dense attribute set.
///
/// A tracked set includes one index per attribute in caller order. Gaps and the next-index
/// counter are retained independently of the number of live attributes. An attribute's heap
/// ID is the same in the name index and the optional creation-order index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum DenseAttrCreationOrder {
    /// Untracked creation order, with positions recorded in name-index records.
    #[default]
    Untracked,
    /// Tracked, with one creation index per attribute in the caller's order.
    Tracked {
        /// Creation index of each attribute, parallel to the attribute slice.
        indices: Vec<u16>,
        /// The counter for the next creation index, independent of `indices`.
        max: u16,
        /// Whether to emit the creation-order B-tree beside the name index.
        indexed: bool,
    },
}

impl DenseAttrCreationOrder {
    /// Returns the tracked creation index at `i`.
    ///
    /// Returns `None` for untracked order or an index outside the set.
    pub fn index_at(&self, i: usize) -> Option<u16> {
        match self {
            Self::Untracked => None,
            Self::Tracked { indices, .. } => indices.get(i).copied(),
        }
    }
}

/// Dense attribute geometry and deferred payload updates.
///
/// Each encoded attribute has a creation index. Names and tracked creation indexes are
/// unique, and tracked indexes are below the retained next-index counter. The managed heap,
/// huge-object placement, index ordering, and total length are retained during
/// [`finalize`](Self::finalize).
///
/// Cloning copies encoded messages and deferred updates while sharing immutable geometry.
#[derive(Clone)]
pub struct PreparedDenseAttrs {
    plan: DenseAttrPlan,
    fixups: Vec<PreparedAttrFixup>,
}

impl PreparedDenseAttrs {
    /// Encodes the attribute set at eight-byte address and length widths.
    ///
    /// Attribute numbers in `fixups` refer to positions in `attrs`. A committed-datatype
    /// fixup requires a committed datatype reference. A global-heap fixup requires a
    /// variable-length datatype and consecutive object indexes starting at one per collection.
    /// A payload fixup requires a datatype without embedded file addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if declared updates overlap, refer to an absent
    /// attribute, or do not match its representation. Propagates errors from [`dense_attrs_plan`].
    pub fn new(
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        Self::with_widths(
            FormatWidths {
                offsets: DENSE_ATTR_OFFSET_WIDTH,
                lengths: DENSE_ATTR_LENGTH_WIDTH,
            },
            attrs,
            creation,
            fixups,
        )
    }

    /// Encodes attributes and prepares deferred updates at independent file widths.
    ///
    /// Variable-length raw values must already use `widths.offsets`-byte heap addresses. Fixup
    /// positions and requirements are the same as for [`Self::new`]. The managed-object limit
    /// is bounded by the direct-block capacity at these widths.
    ///
    /// # Errors
    ///
    /// Returns the metadata, geometry, and deferred-update errors of [`Self::new`], or
    /// [`FormatError::LengthTooLarge`] if a huge-object total or count exceeds `widths.lengths`.
    /// Propagates attribute serialization errors at `widths`.
    pub fn with_widths(
        widths: FormatWidths,
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        Self::with_plan(
            attrs,
            EncodedDenseAttrs::new(widths, attrs, creation)?.into_plan()?,
            fixups,
        )
    }

    /// Reuses encoded messages when preparing dense storage.
    ///
    /// Returns the original messages in input order alongside the prepared storage.
    /// Converts version 2 bodies to version 3 by inserting the name character set. Other
    /// encoded fields are reused. Attribute numbers in `fixups` refer to positions in
    /// `prepared`, with the same requirements as [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns the creation-metadata and geometry errors documented by [`dense_attrs_plan`]
    /// and the update errors documented by [`Self::new`].
    pub fn from_prepared(
        prepared: Vec<PreparedAttributeMessage>,
        creation: DenseAttrCreationOrder,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<(Vec<AttributeMessage>, Self), FormatError> {
        let (attrs, bodies): (Vec<_>, Vec<_>) = prepared
            .into_iter()
            .map(PreparedAttributeMessage::into_v3_parts)
            .unzip();
        let encoded = EncodedDenseAttrs::with_bodies(
            FormatWidths {
                offsets: DENSE_ATTR_OFFSET_WIDTH,
                lengths: DENSE_ATTR_LENGTH_WIDTH,
            },
            &attrs,
            creation,
            bodies,
        )?;
        let plan = encoded.into_plan()?;
        let prepared = Self::with_plan(&attrs, plan, fixups)?;
        Ok((attrs, prepared))
    }

    /// Prepares deferred updates within the encoded plan.
    ///
    /// The caller supplies the same messages in `attrs` that were encoded in `plan`, in the
    /// same order. Attribute numbers in `fixups` refer to that order.
    ///
    /// # Errors
    ///
    /// Returns the deferred-update errors documented by [`Self::new`].
    fn with_plan(
        attrs: &[AttributeMessage],
        plan: DenseAttrPlan,
        fixups: impl IntoIterator<Item = DenseAttrFixup>,
    ) -> Result<Self, FormatError> {
        let widths = plan.geometry.widths;
        let reference_size = DENSE_ATTR_VL_ADDRESS_OFFSET
            + usize::from(widths.offsets.get())
            + core::mem::size_of::<u32>();
        let object_offset = DENSE_ATTR_VL_ADDRESS_OFFSET + usize::from(widths.offsets.get());
        let mut prepared = Vec::new();
        let mut seen = BTreeSet::new();
        for fixup in fixups {
            let (attribute, kind) = match fixup {
                DenseAttrFixup::CommittedDatatype { attribute } => {
                    (attribute, FixupTarget::Datatype)
                }
                DenseAttrFixup::GlobalHeap { attribute }
                | DenseAttrFixup::Payload { attribute } => (attribute, FixupTarget::Data),
            };
            let attr = attrs.get(attribute).ok_or_else(|| {
                FormatError::Internal(
                    "a dense attribute fixup refers to an absent attribute".into(),
                )
            })?;
            if !seen.insert((attribute, kind)) {
                return Err(FormatError::Internal(
                    "dense attribute fixups overlap".into(),
                ));
            }
            let bytes = &plan.attributes[attribute].bytes;
            let data_offset = bytes.len() - attr.raw_data.len();
            prepared.push(match fixup {
                DenseAttrFixup::CommittedDatatype { .. } => {
                    if !attr.datatype_location.is_committed() {
                        return Err(FormatError::Internal(
                            "a committed datatype fixup requires a committed attribute datatype".into(),
                        ));
                    }
                    let dataspace_size = usize::from(u16::from_le_bytes([
                        bytes[DENSE_ATTR_DATASPACE_SIZE_OFFSET],
                        bytes[DENSE_ATTR_DATASPACE_SIZE_OFFSET + 1],
                    ]));
                    PreparedAttrFixup::CommittedDatatype {
                        attribute,
                        offset: data_offset - dataspace_size - usize::from(widths.offsets.get()),
                    }
                }
                DenseAttrFixup::GlobalHeap { .. } => {
                    if !matches!(attr.datatype, Datatype::VariableLength(_))
                        || attr.raw_data.len() % reference_size != 0 {
                        return Err(FormatError::Internal(
                            "a global heap fixup requires whole variable-length attribute references".into(),
                        ));
                    }
                    for (index, reference) in attr.raw_data.chunks_exact(reference_size).enumerate() {
                        let object = u32::from_le_bytes([
                            reference[object_offset], reference[object_offset + 1],
                            reference[object_offset + 2], reference[object_offset + 3],
                        ]);
                        if u64::from(object) != (index % format_boundary::GLOBAL_HEAP_MAX_OBJECTS + 1) as u64 {
                            return Err(FormatError::Internal(
                                "a global heap fixup requires consecutive collection object indexes".into(),
                            ));
                        }
                    }
                    PreparedAttrFixup::GlobalHeap { attribute, offset: data_offset }
                }
                DenseAttrFixup::Payload { .. } => {
                    if format_boundary::datatype_holds_file_address(&attr.datatype) {
                        return Err(FormatError::Internal(
                            "a dense attribute payload fixup cannot replace file references".into(),
                        ));
                    }
                    PreparedAttrFixup::Payload { attribute, offset: data_offset }
                },
            });
        }
        Ok(Self {
            plan,
            fixups: prepared,
        })
    }

    /// Resolves the deferred updates within the retained geometry.
    ///
    /// Calls `address` for each committed datatype and each global-heap collection used by
    /// an attribute. Calls `payload` with the attribute position for each whole-payload fixup.
    /// Replacement payloads must have the prepared length. The returned [`DenseAttrPlan`]
    /// contains the resolved bytes and the original heap and index layouts.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if a resolved address is undefined,
    /// [`FormatError::AddressTooLarge`] if it exceeds the retained address width, or
    /// [`FormatError::Internal`] if a replacement payload has a different length.
    /// Propagates errors from either resolver.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::{CharacterSet, Datatype, FormatError, StringPadding};
    /// use hdf5_pure_format::__private::{
    ///     AttributeMessage, Dataspace, DataspaceType, DatatypeLocation,
    ///     DenseAttrCreationOrder, DenseAttrFixup, PreparedAttributeMessage, PreparedDenseAttrs,
    /// };
    ///
    /// let attribute = AttributeMessage {
    ///     name: "label".into(),
    ///     datatype: Datatype::String {
    ///         size: 1,
    ///         padding: StringPadding::NullPad,
    ///         charset: CharacterSet::Ascii,
    ///     },
    ///     dataspace: Dataspace {
    ///         space_type: DataspaceType::Scalar,
    ///         rank: 0,
    ///         dimensions: vec![],
    ///         max_dimensions: None,
    ///     },
    ///     raw_data: b"a".to_vec(),
    ///     datatype_location: DatatypeLocation::Inline,
    /// };
    /// let (attributes, prepared) = PreparedDenseAttrs::from_prepared(
    ///     vec![PreparedAttributeMessage::new(attribute)?],
    ///     DenseAttrCreationOrder::Untracked,
    ///     [DenseAttrFixup::Payload { attribute: 0 }],
    /// )?;
    /// assert_eq!(attributes[0].name, "label");
    /// assert_eq!(attributes[0].raw_data, b"a");
    /// let reserved = prepared.blob_len();
    /// let ready = prepared.finalize(
    ///     |reference| panic!("unexpected reference {reference:?}"),
    ///     |attribute| {
    ///         assert_eq!(attribute, 0);
    ///         Ok(b"b")
    ///     },
    /// )?;
    /// let emitted = ready.build(StoredAddress::new(0x8000))?;
    /// assert_eq!(emitted.blob.len() as u64, reserved);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn finalize<'a>(
        self,
        mut address: impl FnMut(DenseAttrReference) -> Result<StoredAddress, FormatError>,
        mut payload: impl FnMut(usize) -> Result<&'a [u8], FormatError>,
    ) -> Result<DenseAttrPlan, FormatError> {
        let Self { mut plan, fixups } = self;
        let widths = plan.geometry.widths;
        let reference_size = DENSE_ATTR_VL_ADDRESS_OFFSET
            + usize::from(widths.offsets.get())
            + core::mem::size_of::<u32>();
        let object_offset = DENSE_ATTR_VL_ADDRESS_OFFSET + usize::from(widths.offsets.get());
        for fixup in fixups {
            match fixup {
                PreparedAttrFixup::CommittedDatatype { attribute, offset } => {
                    let resolved = address(DenseAttrReference::CommittedDatatype { attribute })?;
                    defined_dense_attr_reference(widths, resolved)?;
                    plan.attributes[attribute].bytes
                        [offset..offset + usize::from(widths.offsets.get())]
                        .copy_from_slice(
                            &resolved.get().to_le_bytes()[..usize::from(widths.offsets.get())],
                        );
                }
                PreparedAttrFixup::GlobalHeap { attribute, offset } => {
                    for (collection, references) in plan.attributes[attribute].bytes[offset..]
                        .chunks_mut(format_boundary::GLOBAL_HEAP_MAX_OBJECTS * reference_size)
                        .enumerate()
                    {
                        let resolved = address(DenseAttrReference::GlobalHeap {
                            attribute,
                            collection,
                        })?;
                        defined_dense_attr_reference(widths, resolved)?;
                        for reference in references.chunks_exact_mut(reference_size) {
                            reference[DENSE_ATTR_VL_ADDRESS_OFFSET..object_offset].copy_from_slice(
                                &resolved.get().to_le_bytes()[..usize::from(widths.offsets.get())],
                            );
                        }
                    }
                }
                PreparedAttrFixup::Payload { attribute, offset } => {
                    let resolved = payload(attribute)?;
                    let target = &mut plan.attributes[attribute].bytes[offset..];
                    if target.len() != resolved.len() {
                        return Err(FormatError::Internal(
                            "a dense attribute payload fixup changes its prepared length".into(),
                        ));
                    }
                    target.copy_from_slice(resolved);
                }
            }
        }
        Ok(plan)
    }

    /// Returns the allocation kind of each block in placement order.
    ///
    /// The order matches [`Self::block_lengths`] and is preserved by [`Self::finalize`].
    pub fn block_kinds(&self) -> impl ExactSizeIterator<Item = DenseAttrBlockKind> + '_ {
        self.plan.block_kinds()
    }

    /// Returns the complete block lengths before resolving deferred updates.
    ///
    /// The lengths and their order match [`DenseAttrPlan::block_lengths`] after
    /// [`Self::finalize`].
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.plan.block_lengths()
    }

    /// Returns the object-header metadata for independently placed blocks.
    ///
    /// Pass one address per entry of [`Self::block_lengths`], in the same order. The returned
    /// message contains the placement addresses and creation metadata, independently of the
    /// deferred payload updates. Resolve those updates with [`Self::finalize`] before
    /// serializing the blocks through [`DenseAttrPlacement::serialize_blocks`].
    ///
    /// # Errors
    ///
    /// Returns the placement errors documented by [`DenseAttrPlan::place`].
    pub fn attribute_info(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<AttributeInfoMessage, FormatError> {
        self.plan.placement_info(addresses)
    }

    /// Returns the widths retained for reference resolution, placement, and serialization.
    pub fn widths(&self) -> FormatWidths {
        self.plan.geometry.widths
    }

    /// Returns the reserved length, including heap blocks, indexes, and huge objects.
    pub fn blob_len(&self) -> u64 {
        self.plan.blob_len()
    }

    /// Serializes the object-header message for the retained heap placement.
    ///
    /// # Errors
    ///
    /// Returns the address errors documented by [`DenseAttrPlan::attr_info_message`].
    pub fn attr_info_message(&self, heap_address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        self.plan.attr_info_message(heap_address)
    }
}

/// A deferred change to one attribute's encoded bytes.
///
/// Attribute positions refer to the input order of [`PreparedDenseAttrs::new`] or
/// [`PreparedDenseAttrs::from_prepared`].
#[derive(Clone, Copy, Debug)]
pub enum DenseAttrFixup {
    /// Replaces the committed datatype's file address.
    CommittedDatatype {
        /// The attribute position in the prepared set.
        attribute: usize,
    },
    /// Replaces the collection addresses in variable-length references.
    GlobalHeap {
        /// The attribute position in the prepared set.
        attribute: usize,
    },
    /// Replaces raw data with a payload of the same length.
    Payload {
        /// The attribute position in the prepared set.
        attribute: usize,
    },
}

/// A file address requested during [`PreparedDenseAttrs::finalize`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseAttrReference {
    /// The address of an attribute's committed datatype.
    CommittedDatatype {
        /// The attribute position in the prepared set.
        attribute: usize,
    },
    /// The address of a collection used by an attribute's variable-length values.
    GlobalHeap {
        /// The attribute position in the prepared set.
        attribute: usize,
        /// The zero-based collection number within the attribute's values.
        collection: usize,
    },
}

/// Encoded dense attribute payloads and the geometry used to emit them.
///
/// The heap and index offsets, creation metadata, and total length are independent of the
/// heap's file address. [`build`](Self::build) writes the retained layout at a chosen address.
/// Cloning copies encoded messages while sharing immutable geometry.
#[derive(Clone)]
pub struct DenseAttrPlan {
    /// Each attribute's version 3 message bytes and creation index, in the caller's order.
    attributes: Vec<EncodedDenseAttribute>,
    geometry: Arc<DenseAttrGeometry>,
}

/// Encodes attributes and plans their heap and indexes before file placement.
///
/// Attributes that fit [`DENSE_ATTR_MAX_MANAGED_OBJECT`] are stored as managed
/// objects in the heap's direct blocks. Larger ones are stored as *huge*
/// objects, whose bytes sit outside the managed blocks and whose address and
/// length are indexed by a huge-objects v2 B-tree.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold.
/// Returns [`FormatError::DuplicateAttributeName`] if names repeat,
/// [`FormatError::DuplicateAttributeCreationIndex`] if tracked indexes repeat, or
/// [`FormatError::InvalidAttributeCreationCounter`] if a tracked index is at or above the
/// next creation index. Returns [`FormatError::Internal`] if the counts of attributes and
/// tracked indexes differ. Returns [`FormatError::DenseAttributeHeapTooLarge`] if managed
/// storage exceeds the heap address space, [`FormatError::ValueTooLargeForPlatform`] if
/// storage does not fit a [`usize`], [`FormatError::ValueTooLargeForRepresentation`] if an
/// attribute count or position does not fit a [`u32`], or [`FormatError::OffsetOverflow`] if
/// the total length exceeds [`u64`].
/// Propagates other managed-heap planning errors.
pub fn dense_attrs_plan(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
) -> Result<DenseAttrPlan, FormatError> {
    EncodedDenseAttrs::new(
        FormatWidths {
            offsets: DENSE_ATTR_OFFSET_WIDTH,
            lengths: DENSE_ATTR_LENGTH_WIDTH,
        },
        attrs,
        creation,
    )?
    .into_plan()
}

impl DenseAttrPlan {
    /// Returns the allocation kind of each block in placement order.
    ///
    /// The order matches [`Self::block_lengths`]. Huge-object payloads follow the metadata.
    pub fn block_kinds(&self) -> impl ExactSizeIterator<Item = DenseAttrBlockKind> + '_ {
        let metadata = self.geometry.block_lengths.len() - self.geometry.huge_count;
        (0..self.geometry.block_lengths.len()).map(move |index| {
            if index < metadata {
                DenseAttrBlockKind::Metadata
            } else {
                DenseAttrBlockKind::HugeObject
            }
        })
    }

    /// Returns the complete lengths of independently placed blocks.
    ///
    /// The order is the heap header, managed heap blocks, name index, optional creation-order
    /// index, optional huge-object index, and huge-object payloads. Each index includes its
    /// header followed by its nodes. Huge objects retain the input attribute order.
    /// [`Self::place`] takes addresses in this order.
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.geometry.block_lengths.iter().copied()
    }

    /// Places each heap block, index block, and huge object at its supplied address.
    ///
    /// Pass one address per entry of [`Self::block_lengths`], in the same order. Blocks may be
    /// separated by gaps or placed in a different address order. Every complete block must
    /// have a defined start address, a representable exclusive end, and no overlap with
    /// another block. The exclusive end may equal the width's undefined-address value.
    /// The caller chooses alignment and page boundaries.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DenseAttributeBlockCount`] if the number of addresses differs
    /// from the planned count, or [`FormatError::DenseAttributeBlockOverlap`] if blocks
    /// overlap. Returns [`FormatError::DefinedAddressIsUndefined`] if a start address is
    /// undefined, [`FormatError::AddressTooLarge`] if a start or exclusive end exceeds the
    /// retained address width, or [`FormatError::OffsetOverflow`] if an exclusive end exceeds
    /// [`u64::MAX`]. Propagates heap-ID encoding, heap-ID parsing, and index position conversion
    /// errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::FormatError;
    /// use hdf5_pure_format::__private::{dense_attrs_plan, DenseAttrCreationOrder};
    ///
    /// let plan = dense_attrs_plan(&[], DenseAttrCreationOrder::Untracked)?;
    /// let mut next = 0x8000;
    /// let addresses: Vec<_> = plan.block_lengths().map(|length| {
    ///     let address = StoredAddress::new(next);
    ///     next += length + 256;
    ///     address
    /// }).collect();
    /// let placement = plan.place(&addresses)?;
    /// assert_eq!(placement.attribute_info().fractal_heap_address, Some(addresses[0]));
    /// for ((address, length), block) in placement.blocks().zip(placement.serialize_blocks()?) {
    ///     let block = block?;
    ///     assert_eq!(block.address, address);
    ///     assert_eq!(block.bytes.len() as u64, length);
    /// }
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn place(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<DenseAttrPlacement<'_>, FormatError> {
        DenseAttrPlacement::new(self, addresses)
    }

    /// Returns the reserved length, including heap blocks, indexes, and huge objects.
    pub fn blob_len(&self) -> u64 {
        self.geometry.total_len
    }

    /// Serializes the object-header message for a heap placed at `heap_address`.
    ///
    /// The message length is independent of the chosen address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `heap_address` is undefined,
    /// [`FormatError::AddressTooLarge`] if the start or reserved exclusive end exceeds the
    /// retained address width, or [`FormatError::OffsetOverflow`] if the extent exceeds
    /// [`u64::MAX`].
    pub fn attr_info_message(&self, heap_address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        self.extent(heap_address)?;
        self.attribute_info(heap_address)?
            .serialize(self.geometry.widths.offsets)
    }

    /// Returns the creation metadata and placement addresses for the object header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if an index address exceeds [`u64::MAX`].
    fn attribute_info(
        &self,
        heap_address: StoredAddress,
    ) -> Result<AttributeInfoMessage, FormatError> {
        Ok(AttributeInfoMessage {
            max_creation_index: self.geometry.next_index,
            indexes_creation_order: self.geometry.indexed,
            fractal_heap_address: Some(heap_address),
            btree_name_index_address: Some(heap_address.checked_add(self.geometry.btree_off)?),
            btree_creation_order_address: self
                .geometry
                .indexed
                .then(|| heap_address.checked_add(self.geometry.corder_bthd_off))
                .transpose()?,
        })
    }

    /// Emits the blob for a heap placed at `heap_address`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a managed block's address or region end
    /// overflows `u64`, [`FormatError::AddressTooLarge`] if it exceeds the encoded address width,
    /// and [`FormatError::DefinedAddressIsUndefined`] if a required address equals the width's
    /// undefined value. Returns [`FormatError::Internal`] if the serialized length differs from
    /// the reservation. Returns [`FormatError::LengthTooLarge`] if a heap or index field exceeds
    /// the retained length width. Propagates heap-header, index-header, heap-ID, and huge-object
    /// record encoding and parsing errors.
    pub fn build(&self, heap_address: StoredAddress) -> Result<DenseAttrBlob, FormatError> {
        let widths = self.geometry.widths;
        self.extent(heap_address)?;
        let block_offset_bytes = DENSE_ATTR_BLOCK_OFFSET_BYTES; // 5
        let heap_id_length = DENSE_ATTR_HEAP_ID_LENGTH;

        let Self {
            attributes,
            geometry,
        } = self;
        let DenseAttrGeometry {
            huge_id_of,
            huge_count,
            huge_total: _,
            managed_plan,
            name_plan,
            corder_plan,
            huge_plan,
            order,
            corder_order,
            ..
        } = geometry.as_ref();
        let huge_count = *huge_count;

        // The writer adds the heap base to offsets for its blocks, indexes, and huge objects.
        // Attribute references in payload bytes are resolved independently of this base.
        // The serialized length is independent of the chosen heap address.
        let address_of = |offset: u64| heap_address.checked_add(offset);
        let frhp_addr = heap_address;
        let managed_addr = address_of(geometry.managed_off)?;
        let name_nodes_addr = address_of(geometry.name_nodes_off)?;
        let corder_nodes_addr = address_of(geometry.corder_nodes_off)?;
        let (huge_bthd_addr, huge_nodes_addr, huge_data_addr) = if huge_count > 0 {
            (
                address_of(geometry.huge_bthd_off)?,
                address_of(geometry.huge_nodes_off)?,
                address_of(geometry.huge_data_off)?,
            )
        } else {
            (
                StoredAddress::undefined(widths.offsets),
                StoredAddress::undefined(widths.offsets),
                StoredAddress::undefined(widths.offsets),
            )
        };

        // libhdf5 uses the smaller of the maximum direct-block offset width and the encoded
        // width of the maximum managed-object size for a managed ID's length field
        // (`H5HF__hdr_finish_init_phase1`, `H5HFhdr.c`, HDF5 2.2.0). For this geometry, the flag
        // byte, offset, and length occupy the eight-byte heap ID.
        debug_assert_eq!(
            1 + block_offset_bytes + encoded_size_width(u64::from(self.geometry.max_managed)),
            heap_id_length as usize,
            "managed heap ID width must match what the declared maximum managed object size implies"
        );

        let header = self.heap_header(managed_plan.root_address(managed_addr)?, huge_bthd_addr);
        let ids = FractalHeapIdLayout::new(&header, widths.offsets, widths.lengths);
        let frhp = header.serialize(widths.offsets, widths.lengths)?;

        // Encode managed heap offsets and huge-object references.
        let mut heap_ids: Vec<Vec<u8>> = Vec::with_capacity(attributes.len());
        // (huge object ID, address, length) for the huge-objects B-tree, in ID order.
        let mut huge_records: Vec<(u64, StoredAddress, u64)> = Vec::with_capacity(huge_count);
        // Managed-object payloads in attribute order, matching the prepared heap offsets.
        let mut managed: Vec<&[u8]> = Vec::with_capacity(attributes.len() - huge_count);
        let mut next_huge_addr = huge_data_addr;
        for (s, huge_id) in attributes.iter().zip(huge_id_of) {
            match huge_id {
                Some(id) => {
                    let address = next_huge_addr;
                    huge_records.push((*id, address, s.bytes.len() as u64));
                    next_huge_addr = next_huge_addr.checked_add(s.bytes.len() as u64)?;
                    heap_ids.push(if header.huge_ids_direct_with_widths(widths) {
                        header.encode_huge_direct_id(address, s.bytes.len().to_u64(), widths)?
                    } else {
                        header.encode_huge_id(*id, widths.offsets, widths.lengths)?
                    });
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

        // Encode name-index records in the retained order by hash and name.
        let mut name_records =
            Vec::with_capacity(attributes.len() * usize::from(DENSE_ATTR_BTREE_RECORD));
        for &(hash, i) in order {
            AttributeRecord {
                heap_id: PreparedHeapId::new(&ids, &heap_ids[i as usize])?.bytes(),
                flags: MessageFlags::NONE,
                creation_order: attributes[i as usize].creation_index,
            }
            .encode_name_record(&mut name_records, hash);
        }

        let name_tree = name_plan.serialize(
            &name_records,
            name_nodes_addr,
            widths.offsets,
            widths.lengths,
        )?;

        let mut blob = Vec::with_capacity(geometry.total_len.to_usize().unwrap_or(0));
        blob.extend_from_slice(&frhp);
        blob.extend_from_slice(&managed_blocks);
        debug_assert_eq!(blob.len() as u64, geometry.btree_off);
        blob.extend_from_slice(&name_tree.header);
        blob.extend_from_slice(&name_tree.nodes);

        if let Some(corder_plan) = corder_plan {
            // Encode creation-order records with the name-index heap IDs, sorted by creation index.
            let mut corder_records =
                Vec::with_capacity(corder_order.len() * DENSE_ATTR_CORDER_BTREE_RECORD as usize);
            for &i in corder_order {
                AttributeRecord {
                    heap_id: PreparedHeapId::new(&ids, &heap_ids[i as usize])?.bytes(),
                    flags: MessageFlags::NONE,
                    creation_order: attributes[i as usize].creation_index,
                }
                .encode_creation_order_record(&mut corder_records);
            }
            let corder_tree = corder_plan.serialize(
                &corder_records,
                corder_nodes_addr,
                widths.offsets,
                widths.lengths,
            )?;
            debug_assert_eq!(blob.len() as u64, geometry.corder_bthd_off);
            blob.extend_from_slice(&corder_tree.header);
            blob.extend_from_slice(&corder_tree.nodes);
        }

        if let Some(huge_plan) = huge_plan {
            // Contiguous huge payloads have increasing addresses and assigned IDs, matching
            // either tree ordering.
            let mut huge_bytes = Vec::with_capacity(
                huge_records.len()
                    * usize::from(HugeObjectRecord::size(widths.offsets, widths.lengths)),
            );
            for (id, addr, len) in &huge_records {
                if header.huge_ids_direct_with_widths(widths) {
                    HugeObjectDirectRecord {
                        address: *addr,
                        length: *len,
                    }
                    .encode(&mut huge_bytes, widths.offsets, widths.lengths)?;
                } else {
                    HugeObjectRecord {
                        address: *addr,
                        length: *len,
                        id: *id,
                    }
                    .encode(&mut huge_bytes, widths.offsets, widths.lengths)?;
                }
            }
            let huge_tree = huge_plan.serialize(
                &huge_bytes,
                huge_nodes_addr,
                widths.offsets,
                widths.lengths,
            )?;

            debug_assert_eq!(blob.len() as u64, geometry.huge_bthd_off);
            blob.extend_from_slice(&huge_tree.header);
            blob.extend_from_slice(&huge_tree.nodes);
            debug_assert_eq!(blob.len() as u64, geometry.huge_data_off);
            for (s, huge_id) in attributes.iter().zip(huge_id_of) {
                if huge_id.is_some() {
                    blob.extend_from_slice(&s.bytes);
                }
            }
        }

        // Compare the emitted length with the reserved length returned by `blob_len`.
        if blob.len() as u64 != geometry.total_len {
            return Err(FormatError::Internal(
                "a dense attribute heap does not fill its planned reservation".into(),
            ));
        }

        Ok(DenseAttrBlob {
            attr_info_message: self.attr_info_message(frhp_addr)?,
            blob,
        })
    }
    fn heap_header(&self, root: StoredAddress, huge_index: StoredAddress) -> FractalHeapHeader {
        let widths = self.geometry.widths;
        let geometry = &self.geometry;
        FractalHeapHeader {
            heap_id_length: DENSE_ATTR_HEAP_ID_LENGTH,
            filtering: FractalHeapFiltering::Unfiltered,
            flags: FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
            max_managed_object_size: self.geometry.max_managed,
            next_huge_object_id: geometry.huge_count as u64,
            btree_huge_objects_address: if geometry.huge_count == 0 {
                StoredAddress::undefined(widths.offsets)
            } else {
                huge_index
            },
            free_space_in_managed_blocks: geometry.managed_plan.free_space(),
            managed_block_free_space_manager_address: StoredAddress::undefined(widths.offsets),
            managed_space: geometry.managed_plan.managed_space(),
            allocated_managed_space: geometry.managed_plan.allocated_space(),
            direct_block_allocation_iterator_offset: geometry.managed_plan.allocation_iterator(),
            // Managed and huge objects are counted separately, and an attribute is
            // in exactly one of the two.
            managed_objects_count: (self.attributes.len() - geometry.huge_count) as u64,
            huge_objects_size: geometry.huge_total,
            huge_objects_count: geometry.huge_count as u64,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: ATTRIBUTE_HEAP_TABLE_WIDTH,
            starting_block_size: ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
            max_direct_block_size: self.geometry.max_direct,
            max_heap_size: DENSE_ATTR_MAX_HEAP_SIZE_BITS,
            start_root_rows: ATTRIBUTE_HEAP_START_ROOT_ROWS,
            root_block_address: root,
            // Zero when the root is a direct block, which is how a reader decides
            // which of the two the address above points at.
            current_rows_in_root_indirect_block: geometry.managed_plan.root_rows(),
        }
    }

    /// Checks the heap's base address and reserved end.
    ///
    /// # Errors
    ///
    /// Returns the start and exclusive-end errors of [`Self::block_end`].
    fn extent(&self, base: StoredAddress) -> Result<(), FormatError> {
        self.block_end(base, self.geometry.total_len).map(|_| ())
    }

    /// Checks a defined block start and returns its representable exclusive end.
    ///
    /// The exclusive end may equal the address width's undefined marker.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `address` is undefined,
    /// [`FormatError::AddressTooLarge`] if the start or end exceeds the retained width, or
    /// [`FormatError::OffsetOverflow`] if the end exceeds [`u64::MAX`].
    fn block_end(&self, address: StoredAddress, length: u64) -> Result<u64, FormatError> {
        let offsets = self.geometry.widths.offsets;
        let maximum = StoredAddress::undefined(offsets).get();
        if address.get() == maximum {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: offsets.get(),
            });
        }
        if address.get() > maximum {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: offsets.get(),
            });
        }
        let end = address.checked_add(length)?.get();
        if end > maximum {
            return Err(FormatError::AddressTooLarge {
                address: end,
                offset_size: offsets.get(),
            });
        }
        Ok(end)
    }
}

/// Builds dense attribute storage for a set of attributes, at a known address.
///
/// A caller that has to reserve the heap's span before its bytes exist wants
/// [`dense_attrs_plan`] and [`DenseAttrPlan::blob_len`]. This is the
/// one-shot form for callers that already know where the heap goes.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold. Propagates errors from [`dense_attrs_plan`]
/// and [`DenseAttrPlan::build`].
pub fn build_dense_attrs(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
    heap_address: StoredAddress,
) -> Result<DenseAttrBlob, FormatError> {
    dense_attrs_plan(attrs, creation)?.build(heap_address)
}

/// The allocation kind of an independently placed dense-attribute block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseAttrBlockKind {
    /// An encoded attribute stored outside the managed heap blocks.
    HugeObject,
    /// A heap header, managed heap block, B-tree header, or B-tree node.
    Metadata,
}

/// Heap classification, index ordering, and relative offsets retained during finalization.
struct DenseAttrGeometry {
    widths: FormatWidths,
    max_managed: u32,
    max_direct: u64,
    block_lengths: Vec<u64>,
    /// Huge-object ID for each attribute whose bytes go outside the managed
    /// blocks. `None` for a managed one.
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

/// Encoded attributes with unique names and creation indexes.
///
/// Tracked indexes are below the stored counter. Gaps and the counter are preserved
/// independently of the live attributes.
#[derive(Debug)]
struct EncodedDenseAttrs {
    widths: FormatWidths,
    attributes: Vec<EncodedDenseAttribute>,
    /// `(name hash, attribute index)` pairs sorted by hash and then by name.
    order: Vec<(u32, u32)>,
    /// Attribute indexes sorted by creation index for the optional creation-order B-tree.
    corder_order: Vec<u32>,
    next_index: Option<u16>,
    indexed: bool,
}

impl EncodedDenseAttrs {
    /// Encodes attributes and pairs each message with its creation index.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the counts of attributes and tracked indexes differ.
    /// Returns the metadata errors documented by [`dense_attrs_plan`] for repeated names or
    /// indexes and inconsistent counters. Propagates attribute encoding and position conversion
    /// errors.
    fn new(
        widths: FormatWidths,
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
    ) -> Result<Self, FormatError> {
        Self::encode(widths, attrs, creation, |attribute| {
            attribute.serialize_v3(widths.offsets, widths.lengths.get())
        })
    }

    /// Pairs prepared version 3 bodies with their names and creation indexes.
    ///
    /// The caller supplies a matching body for each message in `attrs`, in the same order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if an attribute has no corresponding body.
    /// Returns the metadata and position-conversion errors of [`Self::encode`].
    fn with_bodies(
        widths: FormatWidths,
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
        bodies: Vec<Vec<u8>>,
    ) -> Result<Self, FormatError> {
        let mut bodies = bodies.into_iter();
        Self::encode(widths, attrs, creation, |_| {
            bodies.next().ok_or_else(|| {
                FormatError::Internal("a prepared dense attribute has no encoded body".into())
            })
        })
    }

    /// Assigns creation indexes and index order while obtaining each body's bytes through `body`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the counts of attributes and tracked indexes differ.
    /// Returns the metadata errors documented by [`dense_attrs_plan`] for repeated names or
    /// indexes and inconsistent counters. Returns [`FormatError::ValueTooLargeForRepresentation`]
    /// if an attribute count or position does not fit a [`u32`]. Propagates errors from `body`.
    fn encode(
        widths: FormatWidths,
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
        mut body: impl FnMut(&AttributeMessage) -> Result<Vec<u8>, FormatError>,
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
            attributes.push(EncodedDenseAttribute {
                bytes: body(attribute)?,
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
            widths,
            attributes,
            order,
            corder_order: if indexed { corder_order } else { Vec::new() },
            next_index,
            indexed,
        })
    }

    /// Plans heap blocks and indexes for the encoded attributes.
    ///
    /// # Errors
    ///
    /// Returns the heap-space, platform-size, and length errors documented by [`dense_attrs_plan`].
    fn into_plan(self) -> Result<DenseAttrPlan, FormatError> {
        let EncodedDenseAttrs {
            widths,
            attributes,
            order,
            corder_order,
            next_index,
            indexed,
        } = self;
        let count = attributes.len();
        let max_direct = crate::fractal_heap_write::maximum_direct_block_size(
            ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE,
            widths.lengths,
        );
        let max_managed: u32 = (max_direct
            - crate::fractal_heap_write::attribute_heap_direct_block_header(widths.offsets)
                .to_u64())
        .narrow()?;

        // Huge object IDs start at one in attribute order. The header stores the last assigned ID.
        // `H5HF__huge_new_id` pre-increments the counter (`H5HFhuge.c`, HDF5 2.2.0).
        let mut huge_id_of: Vec<Option<u64>> = vec![None; count];
        let mut huge_count: usize = 0;
        for (slot, attribute) in huge_id_of.iter_mut().zip(&attributes) {
            if attribute.bytes.len().to_u64() > u64::from(max_managed) {
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
        bytes::require_length(huge_total, widths.lengths)?;
        bytes::require_length(huge_count.to_u64(), widths.lengths)?;

        // Only managed objects occupy the doubling table, so only they are planned
        // into it. `managed_sizes` holds their sizes in attribute order, and the
        // plan's object indices are indices into that order.
        let managed_sizes: Vec<u64> = attributes
            .iter()
            .zip(&huge_id_of)
            .filter(|(_, id)| id.is_none())
            .map(|(s, _)| s.bytes.len() as u64)
            .collect();
        let managed_plan = ManagedHeapPlan::attributes_with_widths(&managed_sizes, widths)
            .map_err(dense_attr_heap_error)?;

        // The writer uses the same widths for every version 2 B-tree header.
        let bthd_size = format_boundary::btree_v2_header_size(widths.offsets, widths.lengths);
        debug_assert_eq!(
            bthd_size,
            4 + 1
                + 1
                + 4
                + 2
                + 2
                + 1
                + 1
                + widths.offsets.get() as usize
                + 2
                + widths.lengths.get() as usize
                + 4
        );

        // The writer plans every index before computing subsequent offsets from the node counts.
        // Each fixed-size leaf holds multiple records, so additional levels suffice for larger
        // sets.
        let name_plan = BTreeV2Plan::new(
            BTREE_V2_ATTRIBUTE_NAME,
            count,
            DENSE_ATTR_BTREE_RECORD,
            BTREE_V2_NODE_SIZE,
            widths.offsets,
        )
        .expect("a 512-byte node holds 29 name records, enough to plan any count");
        let corder_plan = indexed.then(|| {
            BTreeV2Plan::new(
                BTREE_V2_ATTRIBUTE_CREATION_ORDER,
                count,
                DENSE_ATTR_CORDER_BTREE_RECORD,
                BTREE_V2_NODE_SIZE,
                widths.offsets,
            )
            .expect("a 512-byte node holds 38 creation-order records, enough to plan any count")
        });
        let direct =
            FractalHeapHeader::unfiltered_huge_ids_direct(DENSE_ATTR_HEAP_ID_LENGTH, widths);
        let huge_plan = (huge_count > 0).then(|| {
            BTreeV2Plan::new(
                if direct {
                    BTREE_V2_HUGE_OBJECT_DIRECT
                } else {
                    BTREE_V2_HUGE_OBJECT
                },
                huge_count,
                if direct {
                    HugeObjectDirectRecord::size(widths.offsets, widths.lengths)
                } else {
                    HugeObjectRecord::size(widths.offsets, widths.lengths)
                },
                BTREE_V2_NODE_SIZE,
                widths.offsets,
            )
            .expect("a 512-byte node holds 20 huge records, enough to plan any count")
        });

        // The blob contains the heap header, managed blocks, name index, optional creation-order
        // index, and optional huge-object index and payloads, in that order. The offsets are
        // relative to the heap address, and the total length is independent of that address.
        let managed_off = FractalHeapHeader::serialized_size(widths.offsets, widths.lengths) as u64;
        let btree_off = managed_off + managed_plan.region_size();
        let name_nodes_off = btree_off + bthd_size as u64;
        let corder_bthd_off = name_nodes_off + name_plan.nodes_size();
        let corder_nodes_off = corder_bthd_off
            + if corder_plan.is_some() {
                bthd_size as u64
            } else {
                0
            };
        let huge_bthd_off =
            corder_nodes_off + corder_plan.as_ref().map_or(0, BTreeV2Plan::nodes_size);
        let huge_nodes_off = huge_bthd_off + bthd_size as u64;
        let huge_data_off = huge_nodes_off + huge_plan.as_ref().map_or(0, BTreeV2Plan::nodes_size);
        // With no huge objects the blob ends where the huge index would have begun.
        // With them it ends past their bytes.
        let total_len = if huge_count > 0 {
            huge_data_off
                .checked_add(huge_total)
                .ok_or(FormatError::OffsetOverflow {
                    offset: huge_data_off,
                    length: huge_total,
                })?
        } else {
            huge_bthd_off
        };

        total_len.to_usize()?;

        let block_lengths = core::iter::once(managed_off)
            .chain(managed_plan.block_lengths())
            .chain(
                core::iter::once(&name_plan)
                    .chain(corder_plan.iter())
                    .chain(huge_plan.iter())
                    .flat_map(|tree| {
                        core::iter::once(bthd_size.to_u64()).chain(tree.block_lengths())
                    }),
            )
            .chain(
                attributes
                    .iter()
                    .zip(&huge_id_of)
                    .filter(|(_, id)| id.is_some())
                    .map(|(attribute, _)| attribute.bytes.len().to_u64()),
            )
            .collect();

        Ok(DenseAttrPlan {
            attributes,
            geometry: Arc::new(DenseAttrGeometry {
                widths,
                max_managed,
                max_direct,
                block_lengths,
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
            }),
        })
    }
}

/// A version 3 attribute message paired with its creation index.
///
/// For untracked attributes, the creation index is the position in the caller's set.
#[derive(Clone, Debug)]
struct EncodedDenseAttribute {
    bytes: Vec<u8>,
    creation_index: u32,
}

/// Returns the minimum nonzero byte width needed to encode `value`.
fn encoded_size_width(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).div_ceil(8).max(1)
}

const DENSE_ATTR_OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
const DENSE_ATTR_LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;

fn dense_attr_heap_error(error: ManagedHeapPlanError) -> FormatError {
    match error {
        ManagedHeapPlanError::HeapSpace => FormatError::DenseAttributeHeapTooLarge {
            limit: ATTRIBUTE_HEAP_MAX_HEAP_SPACE,
        },
        ManagedHeapPlanError::Host { bytes } => FormatError::ValueTooLargeForPlatform {
            value: bytes,
            target: "usize",
        },
        ManagedHeapPlanError::Geometry | ManagedHeapPlanError::ObjectSize { .. } => {
            FormatError::Internal(format!(
                "dense attribute managed heap planning failed: {error:?}"
            ))
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum FixupTarget {
    Data,
    Datatype,
}

/// A declared fixup with its offset in the encoded attribute message.
#[derive(Clone, Debug)]
enum PreparedAttrFixup {
    CommittedDatatype { attribute: usize, offset: usize },
    GlobalHeap { attribute: usize, offset: usize },
    Payload { attribute: usize, offset: usize },
}

/// Checks a resolved reference against the retained address width.
///
/// # Errors
///
/// Returns [`FormatError::DefinedAddressIsUndefined`] if `address` is undefined, or
/// [`FormatError::AddressTooLarge`] if it exceeds `widths.offsets`.
fn defined_dense_attr_reference(
    widths: FormatWidths,
    address: StoredAddress,
) -> Result<(), FormatError> {
    if address.get() > StoredAddress::undefined(widths.offsets).get() {
        return Err(FormatError::AddressTooLarge {
            address: address.get(),
            offset_size: widths.offsets.get(),
        });
    }
    if address.is_undefined(widths.offsets) {
        return Err(FormatError::DefinedAddressIsUndefined {
            address: address.get(),
            offset_size: widths.offsets.get(),
        });
    }
    Ok(())
}

const DENSE_ATTR_VL_ADDRESS_OFFSET: usize = 4;

// The dataspace-size field begins at byte six of a version 3 Attribute message
// ("The Attribute Message", format specification, version 4.0).
const DENSE_ATTR_DATASPACE_SIZE_OFFSET: usize = 6;

#[cfg(test)]
mod tests;
