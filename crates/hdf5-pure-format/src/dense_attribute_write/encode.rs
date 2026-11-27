use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_message::AttributeMessage;
use crate::btree::v2::AttributeRecord;
use crate::btree::v2::HugeObjectDirectRecord;
use crate::btree::v2::HugeObjectRecord;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::FractalHeapIdLayout;
use crate::fractal_heap::PreparedHeapId;
use crate::fractal_heap::write::ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES;
use crate::fractal_heap::write::ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;
use crate::fractal_heap::write::ATTRIBUTE_HEAP_START_ROOT_ROWS;
use crate::fractal_heap::write::ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE;
use crate::fractal_heap::write::ATTRIBUTE_HEAP_TABLE_WIDTH;
use crate::message_flags::MessageFlags;

use super::attributes::DenseAttrCreationOrder;
use super::geometry::{
    DENSE_ATTR_BTREE_RECORD, DENSE_ATTR_CORDER_BTREE_RECORD, DENSE_ATTR_HEAP_ID_LENGTH,
};
use super::plan::{self, DenseAttrPlan};

/// Serialized dense attribute storage and its object-header message.
pub struct DenseAttrBlob {
    /// The checked object-header message for the placed storage.
    pub attribute_info: AttributeInfoMessage,
    /// The heap header, managed blocks, indexes, and huge-object payloads.
    pub blob: Vec<u8>,
}

/// Bits of heap offset the dense attribute heap declares (its "Maximum Heap
/// Size"), and the byte width that implies for a block offset.
const DENSE_ATTR_MAX_HEAP_SIZE_BITS: u16 = ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;
const DENSE_ATTR_BLOCK_OFFSET_BYTES: usize = ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES;

impl DenseAttrPlan {
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
        let geometry = self.geometry();
        let widths = geometry.widths();
        self.extent(heap_address)?;
        let block_offset_bytes = DENSE_ATTR_BLOCK_OFFSET_BYTES; // 5
        let heap_id_length = DENSE_ATTR_HEAP_ID_LENGTH;

        let attributes = self.attributes();
        let huge_id_of = geometry.huge_id_of();
        let huge_count = geometry.huge_count();
        let managed_plan = geometry.managed_plan();
        let name_plan = geometry.name_plan();
        let corder_plan = geometry.corder_plan();
        let huge_plan = geometry.huge_plan();

        // The writer adds the heap base to offsets for its blocks, indexes, and huge objects.
        // Attribute references in payload bytes are resolved independently of this base.
        // The serialized length is independent of the chosen heap address.
        let address_of = |offset: u64| heap_address.checked_add(offset);
        let frhp_addr = heap_address;
        let managed_addr = address_of(geometry.offsets().managed_off)?;
        let name_nodes_addr = address_of(geometry.offsets().name_nodes_off)?;
        let corder_nodes_addr = address_of(geometry.offsets().corder_nodes_off)?;
        let (huge_bthd_addr, huge_nodes_addr, huge_data_addr) = if huge_count > 0 {
            (
                address_of(geometry.offsets().huge_bthd_off)?,
                address_of(geometry.offsets().huge_nodes_off)?,
                address_of(geometry.offsets().huge_data_off)?,
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
            1 + block_offset_bytes + plan::encoded_size_width(u64::from(geometry.max_managed())),
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
                    huge_records.push((*id, address, s.bytes().len() as u64));
                    next_huge_addr = next_huge_addr.checked_add(s.bytes().len() as u64)?;
                    heap_ids.push(if header.huge_ids_direct_with_widths(widths) {
                        header.encode_huge_direct_id(address, s.bytes().len().to_u64(), widths)?
                    } else {
                        header.encode_huge_id(*id, widths.offsets, widths.lengths)?
                    });
                }
                None => {
                    heap_ids.push(header.encode_managed_id(
                        managed_plan.heap_offset(managed.len()),
                        s.bytes().len() as u64,
                    )?);
                    managed.push(s.bytes());
                }
            }
        }

        let managed_blocks = managed_plan.serialize(&managed, managed_addr, frhp_addr)?;
        debug_assert_eq!(managed_blocks.len() as u64, managed_plan.region_size());

        let records = AttributeIndexRecords::new(self, &ids, &heap_ids)?;
        let name_records = records.name_records()?;

        let name_tree = name_plan.serialize(
            &name_records,
            name_nodes_addr,
            widths.offsets,
            widths.lengths,
        )?;

        let mut blob = Vec::with_capacity(geometry.offsets().total_len.to_usize().unwrap_or(0));
        blob.extend_from_slice(&frhp);
        blob.extend_from_slice(&managed_blocks);
        debug_assert_eq!(blob.len() as u64, geometry.offsets().btree_off);
        blob.extend_from_slice(&name_tree.header);
        blob.extend_from_slice(&name_tree.nodes);

        if let Some(corder_plan) = corder_plan {
            let corder_records = records.creation_records()?;
            let corder_tree = corder_plan.serialize(
                &corder_records,
                corder_nodes_addr,
                widths.offsets,
                widths.lengths,
            )?;
            debug_assert_eq!(blob.len() as u64, geometry.offsets().corder_bthd_off);
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

            debug_assert_eq!(blob.len() as u64, geometry.offsets().huge_bthd_off);
            blob.extend_from_slice(&huge_tree.header);
            blob.extend_from_slice(&huge_tree.nodes);
            debug_assert_eq!(blob.len() as u64, geometry.offsets().huge_data_off);
            for (s, huge_id) in attributes.iter().zip(huge_id_of) {
                if huge_id.is_some() {
                    blob.extend_from_slice(s.bytes());
                }
            }
        }

        // Compare the emitted length with the reserved length returned by `blob_len`.
        if blob.len() as u64 != geometry.offsets().total_len {
            return Err(FormatError::Internal(
                "a dense attribute heap does not fill its planned reservation".into(),
            ));
        }

        Ok(DenseAttrBlob {
            attribute_info: self.attribute_info_for_heap(frhp_addr)?,
            blob,
        })
    }

    pub(super) fn heap_header(
        &self,
        root: StoredAddress,
        huge_index: StoredAddress,
    ) -> FractalHeapHeader {
        let widths = self.geometry().widths();
        let geometry = self.geometry();
        FractalHeapHeader {
            heap_id_length: DENSE_ATTR_HEAP_ID_LENGTH,
            filtering: FractalHeapFiltering::Unfiltered,
            flags: FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
            max_managed_object_size: self.geometry().max_managed(),
            next_huge_object_id: geometry.huge_count() as u64,
            btree_huge_objects_address: if geometry.huge_count() == 0 {
                StoredAddress::undefined(widths.offsets)
            } else {
                huge_index
            },
            free_space_in_managed_blocks: geometry.managed_plan().free_space(),
            managed_block_free_space_manager_address: StoredAddress::undefined(widths.offsets),
            managed_space: geometry.managed_plan().managed_space(),
            allocated_managed_space: geometry.managed_plan().allocated_space(),
            direct_block_allocation_iterator_offset: geometry.managed_plan().allocation_iterator(),
            // Managed and huge objects are counted separately, and an attribute is
            // in exactly one of the two.
            managed_objects_count: (self.attributes().len() - geometry.huge_count()) as u64,
            huge_objects_size: geometry.huge_total(),
            huge_objects_count: geometry.huge_count() as u64,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: ATTRIBUTE_HEAP_TABLE_WIDTH,
            starting_block_size: ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
            max_direct_block_size: self.geometry().max_direct(),
            max_heap_size: DENSE_ATTR_MAX_HEAP_SIZE_BITS,
            start_root_rows: ATTRIBUTE_HEAP_START_ROOT_ROWS,
            root_block_address: root,
            // Zero when the root is a direct block, which is how a reader decides
            // which of the two the address above points at.
            current_rows_in_root_indirect_block: geometry.managed_plan().root_rows(),
        }
    }
}

/// Builds dense attribute storage for a set of attributes, at a known address.
///
/// To reserve the contiguous span before choosing its address, use
/// [`dense_attrs_plan`](plan::dense_attrs_plan) and [`DenseAttrPlan::blob_len`].
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if the name with its NUL terminator, the encoded
/// datatype, or the encoded dataspace exceeds 65,535 bytes.
/// Propagates errors from [`plan::dense_attrs_plan`] and [`DenseAttrPlan::build`].
pub fn build_dense_attrs(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
    heap_address: StoredAddress,
) -> Result<DenseAttrBlob, FormatError> {
    plan::dense_attrs_plan(attrs, creation)?.build(heap_address)
}

/// The plan's attributes paired by input position with the heap IDs for their placement.
pub(super) struct AttributeIndexRecords<'a> {
    plan: &'a DenseAttrPlan,
    layout: &'a FractalHeapIdLayout,
    heap_ids: &'a [Vec<u8>],
}

impl<'a> AttributeIndexRecords<'a> {
    /// Pairs each planned attribute with its encoded heap ID.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the number of IDs differs from the attribute count.
    pub(super) fn new(
        plan: &'a DenseAttrPlan,
        layout: &'a FractalHeapIdLayout,
        heap_ids: &'a [Vec<u8>],
    ) -> Result<Self, FormatError> {
        if heap_ids.len() != plan.attributes().len() {
            return Err(FormatError::Internal(
                "dense attribute count differs from heap ID count".into(),
            ));
        }
        Ok(Self {
            plan,
            layout,
            heap_ids,
        })
    }

    /// Encodes name index records in name hash order, breaking ties by name.
    ///
    /// # Errors
    ///
    /// Propagates errors from decoding a heap ID for this layout or converting an attribute
    /// position to a host index.
    pub(super) fn name_records(&self) -> Result<Vec<u8>, FormatError> {
        let order = self.plan.geometry().indexes().order();
        let mut records = Vec::with_capacity(order.len() * usize::from(DENSE_ATTR_BTREE_RECORD));
        for &(hash, position) in order {
            self.record(position)?
                .encode_name_record(&mut records, hash);
        }
        Ok(records)
    }

    /// Encodes creation-order index records in ascending creation order.
    ///
    /// # Errors
    ///
    /// Returns the same heap ID and position errors as [`Self::name_records`].
    pub(super) fn creation_records(&self) -> Result<Vec<u8>, FormatError> {
        let order = self.plan.geometry().indexes().corder_order();
        let mut records =
            Vec::with_capacity(order.len() * usize::from(DENSE_ATTR_CORDER_BTREE_RECORD));
        for &position in order {
            self.record(position)?
                .encode_creation_order_record(&mut records);
        }
        Ok(records)
    }

    fn record(&self, position: u32) -> Result<AttributeRecord<'_>, FormatError> {
        let index = position.to_usize()?;
        Ok(AttributeRecord {
            heap_id: PreparedHeapId::new(self.layout, &self.heap_ids[index])?.bytes(),
            flags: MessageFlags::NONE,
            creation_order: self.plan.attributes()[index].creation_index(),
        })
    }
}

#[cfg(test)]
#[path = "tests/encode.rs"]
mod tests;
