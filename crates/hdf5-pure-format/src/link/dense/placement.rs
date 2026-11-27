use alloc::vec::Vec;

use super::plan::DenseLinkPlan;
use crate::address::StoredAddress;
use crate::btree::v2::write::BTreeV2Placement;
use crate::error::FormatError;
use crate::fractal_heap::write::ManagedHeapPlacement;
use crate::link::info::{LinkInfoAddressField, LinkInfoMessage};

/// A dense-link plan with checked, disjoint stored ranges for its heap and indexes.
///
/// Created by [`DenseLinkPlan::place`]. Includes the heap and index headers and all their blocks.
/// The caller allocates storage and writes each serialized block at its stored address.
pub struct DenseLinkPlacement<'a> {
    plan: &'a DenseLinkPlan,
    addresses: Vec<StoredAddress>,
    objects: Vec<&'a [u8]>,
    heap: ManagedHeapPlacement<'a>,
    name: BTreeV2Placement<'a>,
    creation_index: Option<BTreeV2Placement<'a>>,
    info: LinkInfoMessage,
}

impl<'a> DenseLinkPlacement<'a> {
    pub(super) fn new(
        plan: &'a DenseLinkPlan,
        addresses: &[StoredAddress],
    ) -> Result<Self, FormatError> {
        let expected = plan.block_count();
        if addresses.len() != expected {
            return Err(FormatError::DenseLinkBlockCount {
                expected,
                actual: addresses.len(),
            });
        }
        let heap_end = 1 + plan.heap().block_lengths().len();
        let name_end = heap_end + 1 + plan.name_index().tree().block_lengths().len();
        let mut header_fields = [
            (0, LinkInfoAddressField::Heap),
            (heap_end, LinkInfoAddressField::NameIndex),
            (name_end, LinkInfoAddressField::CreationIndex),
        ]
        .into_iter()
        .take(2 + usize::from(plan.creation_index().is_some()))
        .peekable();
        let mut info = plan.compact_info();
        let mut ranges = Vec::with_capacity(expected);
        for (index, (&address, length)) in addresses.iter().zip(plan.block_lengths()).enumerate() {
            if let Some((_, field)) = header_fields.next_if(|(position, _)| *position == index) {
                info.set_defined_address(field, address)?;
            } else {
                plan.address(address, 0)?;
            }
            let end = address
                .get()
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow {
                    offset: address.get(),
                    length,
                })?;
            if end > StoredAddress::undefined(plan.widths().offsets).get() {
                return Err(FormatError::AddressTooLarge {
                    address: end,
                    offset_size: plan.widths().offsets.get(),
                });
            }
            ranges.push((address.get(), end, index));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(FormatError::DenseLinkBlockOverlap {
                    first: pair[0].2,
                    second: pair[1].2,
                });
            }
        }
        let heap = plan.heap().place(&addresses[1..heap_end])?;
        let name = plan
            .name_index()
            .tree()
            .place(&addresses[heap_end + 1..name_end])?;
        let creation_placement = plan
            .creation_index()
            .map(|index| index.tree().place(&addresses[name_end + 1..]))
            .transpose()?;
        Ok(DenseLinkPlacement {
            plan,
            addresses: addresses.to_vec(),
            objects: plan.objects(),
            heap,
            name,
            creation_index: creation_placement,
            info,
        })
    }

    /// Returns each block's stored address and complete serialized length in placement order.
    ///
    /// The order is the same as [`DenseLinkPlan::block_lengths`].
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.addresses
            .iter()
            .copied()
            .zip(self.plan.block_lengths())
    }

    /// Returns Link Info metadata with the placed heap and index header addresses.
    ///
    /// Preserves the plan's creation-order policy and next-index counter.
    pub fn link_info(&self) -> &LinkInfoMessage {
        &self.info
    }

    /// Serializes the placed heap and indexes into separate block buffers.
    ///
    /// Serializes headers before returning the iterator, and heap blocks and index nodes as
    /// they are requested. Each item contains a stored address and its encoded bytes, including
    /// checksums and unused block space. The order is the same as [`Self::blocks`].
    /// See [`DenseLinkPlan`] for an example.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the object count or index record buffer length differs
    /// from the plan, [`FormatError::DataSizeMismatch`] if an object's length differs from its
    /// planned size, and [`FormatError::LengthTooLarge`] if an index record count exceeds the
    /// encoded length width. Iterator items report [`FormatError::ValueTooLargeForPlatform`] if a
    /// block's serialized length exceeds [`usize::MAX`].
    pub fn serialize_blocks(
        &self,
    ) -> Result<impl Iterator<Item = Result<(StoredAddress, Vec<u8>), FormatError>> + '_, FormatError>
    {
        let Self {
            plan,
            addresses,
            objects,
            heap,
            name,
            creation_index,
            info: _,
        } = self;
        let widths = plan.widths();
        let mut header = plan.header().clone();
        header.root_block_address = heap.root_address();
        let heap_address = addresses[0];
        let heap_header = header.serialize(widths.offsets, widths.lengths)?;
        let heap = heap.serialize_blocks(objects, heap_address)?;
        let name_header = name.serialize_header(widths.lengths)?;
        let name_address = addresses[1 + plan.heap().block_lengths().len()];
        let name = name.serialize_nodes(plan.name_index().records())?;
        let creation = creation_index
            .as_ref()
            .zip(plan.creation_index())
            .map(|(placement, index)| {
                let header = placement.serialize_header(widths.lengths)?;
                let address = addresses[2
                    + plan.heap().block_lengths().len()
                    + plan.name_index().tree().block_lengths().len()];
                let nodes = placement.serialize_nodes(index.records())?;
                Ok::<_, FormatError>(core::iter::once(Ok((address, header))).chain(nodes))
            })
            .transpose()?;
        Ok(core::iter::once(Ok((heap_address, heap_header)))
            .chain(heap)
            .chain(core::iter::once(Ok((name_address, name_header))))
            .chain(name)
            .chain(creation.into_iter().flatten()))
    }
}
