use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::__private as format_boundary;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::attribute_info::AttributeCreationOrderIndex;
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
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

use super::attributes::{DenseAttrCreationOrder, EncodedDenseAttribute, EncodedDenseAttrs};
use super::fixup::FixupRange;
use super::geometry::{
    DENSE_ATTR_BTREE_RECORD, DENSE_ATTR_CORDER_BTREE_RECORD, DENSE_ATTR_HEAP_ID_LENGTH,
    DenseAttrGeometry,
};
use super::placement::DenseAttrPlacement;

// ---- Dense attribute blob ----

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

/// The managed-object size limit of the default eight-byte-width attribute heap.
///
/// The writer uses an eight-byte version 0 managed heap ID with a flag byte, a five-byte
/// offset, and a two-byte length. Larger messages are stored and indexed as huge objects,
/// as defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
pub const DENSE_ATTR_MAX_MANAGED_OBJECT: usize =
    format_boundary::attribute_heap_max_managed_object(DENSE_ATTR_OFFSET_WIDTH);

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
/// tracked indexes differ or if the writer cannot plan one of the indexes. Returns
/// [`FormatError::DenseAttributeHeapTooLarge`] if managed storage exceeds the heap address
/// space, [`FormatError::ValueTooLargeForPlatform`] if
/// storage does not fit a [`usize`], [`FormatError::ValueTooLargeForRepresentation`] if an
/// attribute count or position does not fit a [`u32`], or [`FormatError::OffsetOverflow`] if
/// the total length exceeds [`u64`].
/// Propagates other managed-heap planning errors.
pub fn dense_attrs_plan(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
) -> Result<DenseAttrPlan, FormatError> {
    DenseAttrPlan::new(EncodedDenseAttrs::new(
        FormatWidths {
            offsets: DENSE_ATTR_OFFSET_WIDTH,
            lengths: DENSE_ATTR_LENGTH_WIDTH,
        },
        attrs,
        creation,
    )?)
}

impl DenseAttrPlan {
    pub(super) fn new(encoded: EncodedDenseAttrs) -> Result<Self, FormatError> {
        let (geometry, attributes) = DenseAttrGeometry::prepare(encoded)?;
        Ok(Self {
            attributes,
            geometry: Arc::new(geometry),
        })
    }

    pub(super) fn geometry(&self) -> &DenseAttrGeometry {
        &self.geometry
    }

    pub(super) fn attributes(&self) -> &[EncodedDenseAttribute] {
        &self.attributes
    }

    pub(super) fn fixup_bytes_mut(&mut self, field: &FixupRange) -> Result<&mut [u8], FormatError> {
        EncodedDenseAttribute::fixup_bytes_mut(&mut self.attributes, field)
    }

    /// Returns the allocation kind of each block in placement order.
    ///
    /// The order matches [`Self::block_lengths`]. Huge-object payloads follow the metadata.
    pub fn block_kinds(&self) -> impl ExactSizeIterator<Item = DenseAttrBlockKind> + '_ {
        let metadata = self.geometry.block_lengths().len() - self.geometry.huge_count();
        (0..self.geometry.block_lengths().len()).map(move |index| {
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
        self.geometry.block_lengths().iter().copied()
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
    /// assert_eq!(placement.attribute_info().fractal_heap_address(), Some(addresses[0]));
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
        self.geometry.offsets().total_len
    }

    /// Returns the object-header metadata for a heap placed at `heap_address`.
    ///
    /// The message length is independent of the chosen address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `heap_address` is undefined,
    /// [`FormatError::AddressTooLarge`] if the start or reserved exclusive end exceeds the
    /// retained address width, or [`FormatError::OffsetOverflow`] if the extent exceeds
    /// [`u64::MAX`].
    pub fn attribute_info_for_heap(
        &self,
        heap_address: StoredAddress,
    ) -> Result<AttributeInfoMessage, FormatError> {
        self.extent(heap_address)?;
        AttributeInfoMessage::new(
            self.geometry.widths().offsets,
            self.geometry.indexes().next_index(),
            DecodedStoredAddress::Defined(heap_address),
            DecodedStoredAddress::Defined(
                heap_address.checked_add(self.geometry.offsets().btree_off)?,
            ),
            if self.geometry.indexes().indexed() {
                AttributeCreationOrderIndex::Defined(
                    heap_address.checked_add(self.geometry.offsets().corder_bthd_off)?,
                )
            } else {
                AttributeCreationOrderIndex::Absent
            },
        )
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
        let widths = self.geometry.widths();
        self.extent(heap_address)?;
        let block_offset_bytes = DENSE_ATTR_BLOCK_OFFSET_BYTES; // 5
        let heap_id_length = DENSE_ATTR_HEAP_ID_LENGTH;

        let Self {
            attributes,
            geometry,
        } = self;
        let huge_id_of = geometry.huge_id_of();
        let huge_count = geometry.huge_count();
        let managed_plan = geometry.managed_plan();
        let name_plan = geometry.name_plan();
        let corder_plan = geometry.corder_plan();
        let huge_plan = geometry.huge_plan();
        let order = geometry.indexes().order();
        let corder_order = geometry.indexes().corder_order();

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
            1 + block_offset_bytes + encoded_size_width(u64::from(self.geometry.max_managed())),
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

        // Encode name-index records in the retained order by hash and name.
        let mut name_records =
            Vec::with_capacity(attributes.len() * usize::from(DENSE_ATTR_BTREE_RECORD));
        for &(hash, i) in order {
            AttributeRecord {
                heap_id: PreparedHeapId::new(&ids, &heap_ids[i as usize])?.bytes(),
                flags: MessageFlags::NONE,
                creation_order: attributes[i as usize].creation_index(),
            }
            .encode_name_record(&mut name_records, hash);
        }

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
            // Encode creation-order records with the name-index heap IDs, sorted by creation index.
            let mut corder_records =
                Vec::with_capacity(corder_order.len() * DENSE_ATTR_CORDER_BTREE_RECORD as usize);
            for &i in corder_order {
                AttributeRecord {
                    heap_id: PreparedHeapId::new(&ids, &heap_ids[i as usize])?.bytes(),
                    flags: MessageFlags::NONE,
                    creation_order: attributes[i as usize].creation_index(),
                }
                .encode_creation_order_record(&mut corder_records);
            }
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
        let widths = self.geometry.widths();
        let geometry = self.geometry();
        FractalHeapHeader {
            heap_id_length: DENSE_ATTR_HEAP_ID_LENGTH,
            filtering: FractalHeapFiltering::Unfiltered,
            flags: FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
            max_managed_object_size: self.geometry.max_managed(),
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
            managed_objects_count: (self.attributes.len() - geometry.huge_count()) as u64,
            huge_objects_size: geometry.huge_total(),
            huge_objects_count: geometry.huge_count() as u64,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: ATTRIBUTE_HEAP_TABLE_WIDTH,
            starting_block_size: ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
            max_direct_block_size: self.geometry.max_direct(),
            max_heap_size: DENSE_ATTR_MAX_HEAP_SIZE_BITS,
            start_root_rows: ATTRIBUTE_HEAP_START_ROOT_ROWS,
            root_block_address: root,
            // Zero when the root is a direct block, which is how a reader decides
            // which of the two the address above points at.
            current_rows_in_root_indirect_block: geometry.managed_plan().root_rows(),
        }
    }

    /// Checks the heap's base address and reserved end.
    ///
    /// # Errors
    ///
    /// Returns the start and exclusive-end errors of [`Self::block_end`].
    fn extent(&self, base: StoredAddress) -> Result<(), FormatError> {
        self.block_end(base, self.geometry.offsets().total_len)
            .map(|_| ())
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
    pub(super) fn block_end(
        &self,
        address: StoredAddress,
        length: u64,
    ) -> Result<u64, FormatError> {
        let offsets = self.geometry.widths().offsets;
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

/// Returns the minimum nonzero byte width needed to encode `value`.
pub(super) fn encoded_size_width(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).div_ceil(8).max(1)
}

pub(super) const DENSE_ATTR_OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
pub(super) const DENSE_ATTR_LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;
