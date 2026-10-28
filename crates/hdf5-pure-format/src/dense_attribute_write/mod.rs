//! Dense attribute messages, fractal heaps, and version 2 B-tree indexes.
//!
//! The writer uses eight-byte file addresses and lengths. Attribute messages are paired with
//! creation indexes before heap and index planning.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::__private as format_boundary;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_message::AttributeMessage;
use crate::btree_v2::AttributeRecord;
use crate::btree_v2::BTREE_V2_ATTRIBUTE_CREATION_ORDER;
use crate::btree_v2::BTREE_V2_ATTRIBUTE_NAME;
use crate::btree_v2::BTREE_V2_HUGE_OBJECT;
use crate::btree_v2::HugeObjectRecord;
use crate::btree_v2_write::BTREE_V2_NODE_SIZE;
use crate::btree_v2_write::BTreeV2Plan;
use crate::convert::Narrow;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::fractal_heap::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::FractalHeapHeader;
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
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

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

/// The largest encoded attribute stored in the managed heap blocks.
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

/// One huge-objects B-tree v2 record as [`build_dense_attrs`] writes it.
const DENSE_ATTR_HUGE_BTREE_RECORD: u16 =
    HugeObjectRecord::size(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH);

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
    /// Encodes the attribute set and prepares its declared updates.
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
        let plan = dense_attrs_plan(attrs, creation)?;
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
                        offset: data_offset - dataspace_size - usize::from(DENSE_ATTR_OFFSET_SIZE),
                    }
                }
                DenseAttrFixup::GlobalHeap { .. } => {
                    if !matches!(attr.datatype, Datatype::VariableLength(_))
                        || attr.raw_data.len() % DENSE_ATTR_VL_REF_SIZE != 0 {
                        return Err(FormatError::Internal(
                            "a global heap fixup requires whole variable-length attribute references".into(),
                        ));
                    }
                    for (index, reference) in attr.raw_data.as_chunks::<DENSE_ATTR_VL_REF_SIZE>().0.iter().enumerate() {
                        let object = u32::from_le_bytes([
                            reference[DENSE_ATTR_VL_OBJECT_OFFSET], reference[DENSE_ATTR_VL_OBJECT_OFFSET + 1],
                            reference[DENSE_ATTR_VL_OBJECT_OFFSET + 2], reference[DENSE_ATTR_VL_OBJECT_OFFSET + 3],
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
    /// or [`FormatError::Internal`] if a replacement payload has a different length.
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
    ///     DenseAttrCreationOrder, DenseAttrFixup, PreparedDenseAttrs,
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
    /// let prepared = PreparedDenseAttrs::new(
    ///     &[attribute],
    ///     DenseAttrCreationOrder::Untracked,
    ///     [DenseAttrFixup::Payload { attribute: 0 }],
    /// )?;
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
        for fixup in fixups {
            match fixup {
                PreparedAttrFixup::CommittedDatatype { attribute, offset } => {
                    let resolved = address(DenseAttrReference::CommittedDatatype { attribute })?;
                    defined_dense_attr_reference(resolved)?;
                    plan.attributes[attribute].bytes
                        [offset..offset + usize::from(DENSE_ATTR_OFFSET_SIZE)]
                        .copy_from_slice(&resolved.get().to_le_bytes());
                }
                PreparedAttrFixup::GlobalHeap { attribute, offset } => {
                    for (collection, references) in plan.attributes[attribute].bytes[offset..]
                        .chunks_mut(
                            format_boundary::GLOBAL_HEAP_MAX_OBJECTS * DENSE_ATTR_VL_REF_SIZE,
                        )
                        .enumerate()
                    {
                        let resolved = address(DenseAttrReference::GlobalHeap {
                            attribute,
                            collection,
                        })?;
                        defined_dense_attr_reference(resolved)?;
                        for reference in references.as_chunks_mut::<DENSE_ATTR_VL_REF_SIZE>().0 {
                            reference[DENSE_ATTR_VL_ADDRESS_OFFSET..DENSE_ATTR_VL_OBJECT_OFFSET]
                                .copy_from_slice(&resolved.get().to_le_bytes());
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
/// Attribute positions refer to the input of [`PreparedDenseAttrs::new`].
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
/// storage does not fit a `usize` or an attribute count or position does not fit a `u32`, or
/// [`FormatError::OffsetOverflow`] if the total length exceeds `u64`.
/// Propagates other managed-heap planning errors.
pub fn dense_attrs_plan(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
) -> Result<DenseAttrPlan, FormatError> {
    let EncodedDenseAttrs {
        attributes,
        order,
        corder_order,
        next_index,
        indexed,
    } = EncodedDenseAttrs::new(attrs, creation)?;

    // Huge object IDs start at one in attribute order. The header stores the last assigned ID.
    // `H5HF__huge_new_id` pre-increments the counter (`H5HFhuge.c`, HDF5 2.2.0).
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
    let managed_plan = ManagedHeapPlan::attributes(&managed_sizes, DENSE_ATTR_OFFSET_WIDTH)
        .map_err(dense_attr_heap_error)?;

    // The writer uses the same widths for every version 2 B-tree header.
    let bthd_size =
        format_boundary::btree_v2_header_size(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH);
    debug_assert_eq!(
        bthd_size,
        4 + 1
            + 1
            + 4
            + 2
            + 2
            + 1
            + 1
            + DENSE_ATTR_OFFSET_SIZE as usize
            + 2
            + DENSE_ATTR_LENGTH_SIZE as usize
            + 4
    );

    // The writer plans every index before computing subsequent offsets from the node counts.
    // Each fixed-size leaf holds multiple records, so additional levels suffice for larger sets.
    let name_plan = BTreeV2Plan::new(
        BTREE_V2_ATTRIBUTE_NAME,
        attrs.len(),
        DENSE_ATTR_BTREE_RECORD,
        BTREE_V2_NODE_SIZE,
        DENSE_ATTR_OFFSET_WIDTH,
    )
    .expect("a 512-byte node holds 29 name records, enough to plan any count");
    let corder_plan = indexed.then(|| {
        BTreeV2Plan::new(
            BTREE_V2_ATTRIBUTE_CREATION_ORDER,
            attrs.len(),
            DENSE_ATTR_CORDER_BTREE_RECORD,
            BTREE_V2_NODE_SIZE,
            DENSE_ATTR_OFFSET_WIDTH,
        )
        .expect("a 512-byte node holds 38 creation-order records, enough to plan any count")
    });
    let huge_plan = (huge_count > 0).then(|| {
        BTreeV2Plan::new(
            BTREE_V2_HUGE_OBJECT,
            huge_count,
            DENSE_ATTR_HUGE_BTREE_RECORD,
            BTREE_V2_NODE_SIZE,
            DENSE_ATTR_OFFSET_WIDTH,
        )
        .expect("a 512-byte node holds 20 huge records, enough to plan any count")
    });

    // The blob contains the heap header, managed blocks, name index, optional creation-order
    // index, and optional huge-object index and payloads, in that order. The offsets are
    // relative to the heap address, and the total length is independent of that address.
    let managed_off =
        FractalHeapHeader::serialized_size(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH) as u64;
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

    Ok(DenseAttrPlan {
        attributes,
        geometry: Arc::new(DenseAttrGeometry {
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

impl DenseAttrPlan {
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
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `heap_address` is undefined, or
    /// [`FormatError::OffsetOverflow`] if the reserved extent exceeds `u64`.
    pub fn attr_info_message(&self, heap_address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        self.extent(heap_address)?;
        Ok(self
            .attribute_info(heap_address)
            .serialize(DENSE_ATTR_OFFSET_WIDTH))
    }

    /// Returns the creation metadata and placement addresses for the object header.
    fn attribute_info(&self, heap_address: StoredAddress) -> AttributeInfoMessage {
        AttributeInfoMessage {
            max_creation_index: self.geometry.next_index,
            indexes_creation_order: self.geometry.indexed,
            fractal_heap_address: Some(heap_address),
            btree_name_index_address: Some(heap_address.offset(self.geometry.btree_off)),
            btree_creation_order_address: self
                .geometry
                .indexed
                .then(|| heap_address.offset(self.geometry.corder_bthd_off)),
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
    /// required address equals the width's undefined value. Returns [`FormatError::Internal`]
    /// if the serialized length differs from the reservation.
    pub fn build(&self, heap_address: StoredAddress) -> Result<DenseAttrBlob, FormatError> {
        self.extent(heap_address)?;
        let max_heap_size: u16 = DENSE_ATTR_MAX_HEAP_SIZE_BITS;
        let block_offset_bytes = DENSE_ATTR_BLOCK_OFFSET_BYTES; // 5
        let heap_id_length = DENSE_ATTR_HEAP_ID_LENGTH;

        let Self {
            attributes,
            geometry,
        } = self;
        let DenseAttrGeometry {
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
        } = geometry.as_ref();
        let huge_count = *huge_count;

        // The writer adds the heap base to offsets for its blocks, indexes, and huge objects.
        // Attribute references in payload bytes are resolved independently of this base.
        // The serialized length is independent of the chosen heap address.
        let address_of = |offset: u64| heap_address.offset(offset);
        let frhp_addr = heap_address;
        let managed_addr = address_of(geometry.managed_off);
        let name_nodes_addr = address_of(geometry.name_nodes_off);
        let corder_nodes_addr = address_of(geometry.corder_nodes_off);
        let (huge_bthd_addr, huge_nodes_addr, huge_data_addr) = if huge_count > 0 {
            (
                address_of(geometry.huge_bthd_off),
                address_of(geometry.huge_nodes_off),
                address_of(geometry.huge_data_off),
            )
        } else {
            (
                DENSE_ATTR_UNDEFINED,
                DENSE_ATTR_UNDEFINED,
                DENSE_ATTR_UNDEFINED,
            )
        };

        // libhdf5 uses the smaller of the maximum direct-block offset width and the encoded
        // width of the maximum managed-object size for a managed ID's length field
        // (`H5HF__hdr_finish_init_phase1`, `H5HFhdr.c`, HDF5 2.2.0). For this geometry, the flag
        // byte, offset, and length occupy the eight-byte heap ID.
        debug_assert_eq!(
            1 + block_offset_bytes + encoded_size_width(DENSE_ATTR_MAX_MANAGED_OBJECT as u64),
            heap_id_length as usize,
            "managed heap ID width must match what the declared maximum managed object size implies"
        );

        // A constant independent of `max_direct_block_size`:
        // this is the per-object cap the 2-byte length field of an 8-byte managed
        // heap ID can encode. Larger messages use huge-object storage.
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
                DENSE_ATTR_UNDEFINED
            } else {
                huge_bthd_addr
            },
            free_space_in_managed_blocks: managed_plan.free_space(),
            managed_block_free_space_manager_address: DENSE_ATTR_UNDEFINED,
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
        let frhp = header.serialize(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH)?;

        // Encode a managed ID with its heap offset, or a huge ID with its B-tree key.
        let mut heap_ids: Vec<Vec<u8>> = Vec::with_capacity(attributes.len());
        // (huge object ID, address, length) for the huge-objects B-tree, in ID order.
        let mut huge_records: Vec<(u64, StoredAddress, u64)> = Vec::with_capacity(huge_count);
        // Managed-object payloads in attribute order, matching the prepared heap offsets.
        let mut managed: Vec<&[u8]> = Vec::with_capacity(attributes.len() - huge_count);
        let mut next_huge_addr = huge_data_addr;
        for (s, huge_id) in attributes.iter().zip(huge_id_of) {
            match huge_id {
                Some(id) => {
                    huge_records.push((*id, next_huge_addr, s.bytes.len() as u64));
                    next_huge_addr = next_huge_addr.offset(s.bytes.len() as u64);
                    heap_ids.push(header.encode_huge_id(
                        *id,
                        DENSE_ATTR_OFFSET_WIDTH,
                        DENSE_ATTR_LENGTH_WIDTH,
                    )?);
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
                heap_id: &heap_ids[i as usize],
                flags: MessageFlags::NONE,
                creation_order: attributes[i as usize].creation_index,
            }
            .encode_name_record(&mut name_records, hash);
        }

        let name_tree = name_plan.serialize(
            &name_records,
            name_nodes_addr,
            DENSE_ATTR_OFFSET_WIDTH,
            DENSE_ATTR_LENGTH_WIDTH,
        );

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
                    heap_id: &heap_ids[i as usize],
                    flags: MessageFlags::NONE,
                    creation_order: attributes[i as usize].creation_index,
                }
                .encode_creation_order_record(&mut corder_records);
            }
            let corder_tree = corder_plan.serialize(
                &corder_records,
                corder_nodes_addr,
                DENSE_ATTR_OFFSET_WIDTH,
                DENSE_ATTR_LENGTH_WIDTH,
            );
            debug_assert_eq!(blob.len() as u64, geometry.corder_bthd_off);
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
                .encode(
                    &mut huge_bytes,
                    DENSE_ATTR_OFFSET_WIDTH,
                    DENSE_ATTR_LENGTH_WIDTH,
                );
            }
            let huge_tree = huge_plan.serialize(
                &huge_bytes,
                huge_nodes_addr,
                DENSE_ATTR_OFFSET_WIDTH,
                DENSE_ATTR_LENGTH_WIDTH,
            );

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
    /// Checks the heap's base address and reserved end.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `base` is undefined, or
    /// [`FormatError::OffsetOverflow`] if the reserved end exceeds `u64`.
    fn extent(&self, base: StoredAddress) -> Result<(), FormatError> {
        if base.is_undefined(DENSE_ATTR_OFFSET_WIDTH) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: base.get(),
                offset_size: DENSE_ATTR_OFFSET_SIZE,
            });
        }
        base.get()
            .checked_add(self.geometry.total_len)
            .ok_or(FormatError::OffsetOverflow {
                offset: base.get(),
                length: self.geometry.total_len,
            })?;
        Ok(())
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

/// Heap classification, index ordering, and relative offsets retained during finalization.
struct DenseAttrGeometry {
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
            attributes.push(EncodedDenseAttribute {
                bytes: attribute.serialize_v3(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_SIZE)?,
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
const DENSE_ATTR_OFFSET_SIZE: u8 = DENSE_ATTR_OFFSET_WIDTH.get();
const DENSE_ATTR_LENGTH_SIZE: u8 = DENSE_ATTR_LENGTH_WIDTH.get();
const DENSE_ATTR_UNDEFINED: StoredAddress = StoredAddress::undefined(DENSE_ATTR_OFFSET_WIDTH);

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

/// Rejects an undefined address returned by the reference resolver.
///
/// # Errors
///
/// Returns [`FormatError::DefinedAddressIsUndefined`] if `address` is undefined.
fn defined_dense_attr_reference(address: StoredAddress) -> Result<(), FormatError> {
    if address.is_undefined(DENSE_ATTR_OFFSET_WIDTH) {
        return Err(FormatError::DefinedAddressIsUndefined {
            address: address.get(),
            offset_size: DENSE_ATTR_OFFSET_SIZE,
        });
    }
    Ok(())
}

// `H5T__vlen_disk_write` writes a four-byte sequence length (`H5Tvlen.c`, HDF5 2.2.0).
// "Global Heap" of the format specification, version 4.0, defines the following collection
// address and four-byte object index. The writer uses eight-byte addresses.
const DENSE_ATTR_VL_REF_SIZE: usize = 16;
const DENSE_ATTR_VL_ADDRESS_OFFSET: usize = 4;
const DENSE_ATTR_VL_OBJECT_OFFSET: usize = 12;

// The dataspace-size field begins at byte six of a version 3 Attribute message
// ("The Attribute Message", format specification, version 4.0).
const DENSE_ATTR_DATASPACE_SIZE_OFFSET: usize = 6;

#[cfg(test)]
mod tests;
