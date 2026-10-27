use alloc::format;
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

#[cfg(test)]
mod tests;

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
pub enum DenseAttrCreationOrder {
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
    pub fn index_at(&self, i: usize) -> Option<u16> {
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
pub fn dense_attrs_check(attrs: &[AttributeMessage]) -> Result<(), FormatError> {
    let mut managed = Vec::new();
    for a in attrs {
        let size = a
            .serialize_v3(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_SIZE)?
            .len();
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
    ManagedHeapPlan::attributes(&managed, DENSE_ATTR_OFFSET_WIDTH)
        .map(|_| ())
        .map_err(dense_attr_heap_error)
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
pub struct DenseAttrPlan {
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
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if an attribute's name, datatype, or dataspace
/// is longer than its 2-byte size field can hold.
/// Returns [`FormatError::DuplicateAttributeName`] if names repeat,
/// [`FormatError::DuplicateAttributeCreationIndex`] if tracked indexes repeat, or
/// [`FormatError::InvalidAttributeCreationCounter`] if a tracked index is at or above the
/// next creation index. Returns [`FormatError::Internal`] if the counts of attributes and
/// tracked indexes differ. Returns [`FormatError::DenseAttributeHeapTooLarge`] if managed
/// storage exceeds the heap address space. Propagates platform-size and managed-heap errors.
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
    let managed_plan = ManagedHeapPlan::attributes(&managed_sizes, DENSE_ATTR_OFFSET_WIDTH)
        .map_err(dense_attr_heap_error)?;

    // Every v2 B-tree header this emitter writes has the same fixed layout, so
    // one size covers both the name index and the huge-objects index.
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

    // Blob layout, all relative to the heap address the blob is built for, so
    // its length is the same wherever it lands: heap header, the managed blocks,
    // name index (header + nodes), then — only when there are huge objects — the
    // huge index (header + nodes) and the huge object bytes themselves.
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
    pub fn blob_len(&self) -> u64 {
        self.total_len
    }

    /// The Attribute Info (0x0015) message naming this heap once it is placed at
    /// `heap_address`. Its length does not depend on the address, so an
    /// object-header sizing pass can take it from a provisional one.
    pub fn attr_info_message(&self, heap_address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        self.extent(heap_address)?;
        Ok(self
            .attribute_info(heap_address)
            .serialize(DENSE_ATTR_OFFSET_WIDTH))
    }

    fn extent(&self, base: StoredAddress) -> Result<(), FormatError> {
        if base.is_undefined(DENSE_ATTR_OFFSET_WIDTH) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: base.get(),
                offset_size: DENSE_ATTR_OFFSET_SIZE,
            });
        }
        base.get()
            .checked_add(self.total_len)
            .ok_or(FormatError::OffsetOverflow {
                offset: base.get(),
                length: self.total_len,
            })?;
        Ok(())
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
    pub fn build(&self, heap_address: StoredAddress) -> Result<DenseAttrBlob, FormatError> {
        self.extent(heap_address)?;
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

        let name_tree = name_plan.serialize(
            &name_records,
            name_nodes_addr,
            DENSE_ATTR_OFFSET_WIDTH,
            DENSE_ATTR_LENGTH_WIDTH,
        );

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
                DENSE_ATTR_OFFSET_WIDTH,
                DENSE_ATTR_LENGTH_WIDTH,
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
        if blob.len() as u64 != self.total_len {
            return Err(FormatError::Internal(
                "a dense attribute heap does not fill its planned reservation".into(),
            ));
        }

        Ok(DenseAttrBlob {
            attr_info_message: self.attr_info_message(frhp_addr)?,
            blob,
        })
    }
}

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
pub fn build_dense_attrs(
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
struct EncodedDenseAttrs {
    attributes: Vec<PreparedDenseAttribute>,
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
            attributes.push(PreparedDenseAttribute {
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
