use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::attribute_info::AttributeInfoMessage;
use crate::btree_v2::{AttributeRecord, HugeObjectRecord};
use crate::btree_v2_write::BTreeV2Placement;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap_write::ManagedHeapPlacement;
use crate::message_flags::MessageFlags;

use super::{
    DENSE_ATTR_LENGTH_WIDTH, DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_UNDEFINED, DenseAttrPlan,
};

/// Dense attribute storage with checked, independent block addresses.
///
/// The complete block ranges are disjoint and have defined start addresses and representable
/// exclusive ends. [`DenseAttrPlan::place`] constructs the placement in
/// [`DenseAttrPlan::block_lengths`] order. Alignment and page boundaries are chosen by the caller.
pub struct DenseAttrPlacement<'a> {
    plan: &'a DenseAttrPlan,
    addresses: Vec<StoredAddress>,
    managed: Vec<&'a [u8]>,
    heap: ManagedHeapPlacement<'a>,
    indexes: Vec<PlacedIndex<'a>>,
    huge: Vec<&'a [u8]>,
    info: AttributeInfoMessage,
}

impl<'a> DenseAttrPlacement<'a> {
    pub(super) fn new(
        plan: &'a DenseAttrPlan,
        addresses: &[StoredAddress],
    ) -> Result<Self, FormatError> {
        let DenseAttrPlan {
            attributes,
            geometry,
        } = plan;
        let info = plan.placement_info(addresses)?;
        let heap_end = 1 + geometry.managed_plan.block_lengths().len();
        let heap = geometry.managed_plan.place(&addresses[1..heap_end])?;
        let mut cursor = heap_end;
        let mut indexes = Vec::with_capacity(3);
        for tree in core::iter::once(&geometry.name_plan)
            .chain(geometry.corder_plan.iter())
            .chain(geometry.huge_plan.iter())
        {
            let end = cursor + 1 + tree.block_lengths().len();
            indexes.push(PlacedIndex {
                address: addresses[cursor],
                placement: tree.place(&addresses[cursor + 1..end])?,
                records: Vec::new(),
            });
            cursor = end;
        }
        let huge_addresses = &addresses[cursor..];
        let header = plan.heap_header(
            heap.root_address(),
            if geometry.huge_plan.is_some() {
                indexes
                    .last()
                    .map_or(DENSE_ATTR_UNDEFINED, |index| index.address)
            } else {
                DENSE_ATTR_UNDEFINED
            },
        );
        let mut managed = Vec::with_capacity(attributes.len() - geometry.huge_count);
        let mut huge = Vec::with_capacity(geometry.huge_count);
        let mut heap_ids = Vec::with_capacity(attributes.len());
        let mut huge_records = Vec::new();
        for (attribute, huge_id) in attributes.iter().zip(&geometry.huge_id_of) {
            let id = if let Some(id) = huge_id {
                HugeObjectRecord {
                    address: huge_addresses[huge.len()],
                    length: attribute.bytes.len().to_u64(),
                    id: *id,
                }
                .encode(
                    &mut huge_records,
                    DENSE_ATTR_OFFSET_WIDTH,
                    DENSE_ATTR_LENGTH_WIDTH,
                );
                huge.push(attribute.bytes.as_slice());
                header.encode_huge_id(*id, DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH)?
            } else {
                let id = header.encode_managed_id(
                    geometry.managed_plan.heap_offset(managed.len()),
                    attribute.bytes.len().to_u64(),
                )?;
                managed.push(attribute.bytes.as_slice());
                id
            };
            heap_ids.push(id);
        }
        for &(hash, position) in &geometry.order {
            let position = position.to_usize()?;
            AttributeRecord {
                heap_id: &heap_ids[position],
                flags: MessageFlags::NONE,
                creation_order: attributes[position].creation_index,
            }
            .encode_name_record(&mut indexes[0].records, hash);
        }
        if geometry.indexed {
            for &position in &geometry.corder_order {
                let position = position.to_usize()?;
                AttributeRecord {
                    heap_id: &heap_ids[position],
                    flags: MessageFlags::NONE,
                    creation_order: attributes[position].creation_index,
                }
                .encode_creation_order_record(&mut indexes[1].records);
            }
        }
        if geometry.huge_plan.is_some() {
            indexes[1 + usize::from(geometry.indexed)].records = huge_records;
        }
        Ok(Self {
            plan,
            addresses: addresses.to_vec(),
            managed,
            heap,
            indexes,
            huge,
            info,
        })
    }

    /// Returns complete block addresses and lengths in serialization order.
    ///
    /// The order matches [`DenseAttrPlan::block_lengths`].
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.addresses
            .iter()
            .copied()
            .zip(self.plan.block_lengths())
    }

    /// Returns the object-header metadata with the placed heap and index addresses.
    pub fn attribute_info(&self) -> &AttributeInfoMessage {
        &self.info
    }

    /// Serializes complete blocks in placement order.
    ///
    /// The order matches [`Self::blocks`]. Metadata bytes are owned, and huge-object bytes are
    /// borrowed from the plan. Managed heap blocks and B-tree nodes are serialized as the
    /// iterator advances.
    /// See [`DenseAttrPlan::place`] for an example.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if a managed payload length differs from its
    /// planned length, or [`FormatError::Internal`] if the retained object count or index record
    /// lengths differ from the plan. Propagates heap-header and index-header serialization errors.
    /// Iterator items propagate managed-heap block and B-tree node serialization errors.
    pub fn serialize_blocks(
        &self,
    ) -> Result<impl Iterator<Item = Result<DenseAttrBlock<'_>, FormatError>> + '_, FormatError>
    {
        let Self {
            plan,
            addresses,
            managed,
            heap,
            indexes,
            huge,
            info: _,
        } = self;
        let heap_address = addresses[0];
        let huge_index = if plan.geometry.huge_plan.is_some() {
            indexes[1 + usize::from(plan.geometry.indexed)].address
        } else {
            DENSE_ATTR_UNDEFINED
        };
        let header = plan
            .heap_header(heap.root_address(), huge_index)
            .serialize(DENSE_ATTR_OFFSET_WIDTH, DENSE_ATTR_LENGTH_WIDTH)?;
        let heap = heap.serialize_blocks(managed, heap_address)?;
        let indexes = indexes
            .iter()
            .map(|index| {
                let header = index.placement.serialize_header(DENSE_ATTR_LENGTH_WIDTH)?;
                let nodes = index.placement.serialize_nodes(&index.records)?;
                Ok::<_, FormatError>(core::iter::once(Ok((index.address, header))).chain(nodes))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let huge_start = addresses.len() - huge.len();
        let huge = addresses[huge_start..]
            .iter()
            .copied()
            .zip(huge.iter().copied())
            .map(|(address, bytes)| {
                Ok(DenseAttrBlock {
                    address,
                    bytes: Cow::Borrowed(bytes),
                })
            });
        Ok(core::iter::once(Ok((heap_address, header)))
            .chain(heap)
            .chain(indexes.into_iter().flatten())
            .map(|block| {
                block.map(|(address, bytes)| DenseAttrBlock {
                    address,
                    bytes: Cow::Owned(bytes),
                })
            })
            .chain(huge))
    }
}

impl DenseAttrPlan {
    pub(super) fn placement_info(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<AttributeInfoMessage, FormatError> {
        let lengths = self.block_lengths();
        let expected = lengths.len();
        if addresses.len() != expected {
            return Err(FormatError::DenseAttributeBlockCount {
                expected,
                actual: addresses.len(),
            });
        }
        let mut ranges = Vec::with_capacity(expected);
        for (index, (&address, length)) in addresses.iter().zip(lengths).enumerate() {
            if address.is_undefined(DENSE_ATTR_OFFSET_WIDTH) {
                return Err(FormatError::DefinedAddressIsUndefined {
                    address: address.get(),
                    offset_size: DENSE_ATTR_OFFSET_WIDTH.get(),
                });
            }
            let end = address
                .get()
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow {
                    offset: address.get(),
                    length,
                })?;
            ranges.push((address.get(), end, index));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(FormatError::DenseAttributeBlockOverlap {
                    first: pair[0].2,
                    second: pair[1].2,
                });
            }
        }
        let geometry = &self.geometry;
        let name = 1 + geometry.managed_plan.block_lengths().len();
        let creation = name + 1 + geometry.name_plan.block_lengths().len();
        Ok(AttributeInfoMessage {
            max_creation_index: geometry.next_index,
            indexes_creation_order: geometry.indexed,
            fractal_heap_address: Some(addresses[0]),
            btree_name_index_address: Some(addresses[name]),
            btree_creation_order_address: geometry.indexed.then(|| addresses[creation]),
        })
    }
}

/// One complete serialized dense-attribute block at its stored address.
///
/// Metadata bytes are owned. Huge-object bytes are borrowed from the [`DenseAttrPlan`].
pub struct DenseAttrBlock<'a> {
    /// The block's stored file address.
    pub address: StoredAddress,
    /// The complete encoded block, including framing where present.
    pub bytes: Cow<'a, [u8]>,
}

struct PlacedIndex<'a> {
    address: StoredAddress,
    placement: BTreeV2Placement<'a>,
    records: Vec<u8>,
}
