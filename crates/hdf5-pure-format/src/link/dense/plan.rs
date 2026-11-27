use alloc::vec::Vec;

use super::error::DenseLinkPlanError;
use super::index::{
    CREATION_RECORD_LEN, HEAP_ID_LEN, LinkIndexEntries, NAME_RECORD_LEN, PlannedIndex,
};
use super::placement::DenseLinkPlacement;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2 as btree_v2;
use crate::btree::v2::BTREE_V2_LINK_CREATION_ORDER;
use crate::btree::v2::BTREE_V2_LINK_NAME;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::FractalHeapIdLayout;
use crate::fractal_heap::write::GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE;
use crate::fractal_heap::write::GROUP_HEAP_MAX_HEAP_SIZE_BITS;
use crate::fractal_heap::write::GROUP_HEAP_MAX_MANAGED_OBJECT;
use crate::fractal_heap::write::GROUP_HEAP_START_ROOT_ROWS;
use crate::fractal_heap::write::GROUP_HEAP_STARTING_BLOCK_SIZE;
use crate::fractal_heap::write::GROUP_HEAP_TABLE_WIDTH;
use crate::fractal_heap::write::ManagedHeapPlan;
use crate::link::creation_order::LinkCreationPolicy;
use crate::link::info::{LinkInfoAddressField, LinkInfoMessage};
use crate::link::storage::GroupLinkStorage;
use crate::width;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The layout and encoded links of a dense group before its storage address is chosen.
///
/// Uses an unfiltered managed fractal heap, a type-5 name index, and a type-6 creation-order
/// index when enabled by [`LinkCreationPolicy`]. Link names, character-set markers, targets,
/// creation indexes, and the next-index counter are preserved. Existing storage addresses are
/// replaced during serialization.
///
/// The index records are defined in "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_format::CharacterSet;
/// use hdf5_pure_format::LengthWidth;
/// use hdf5_pure_format::OffsetWidth;
/// use hdf5_pure_format::__private::DenseLinkPlan;
/// use hdf5_pure_format::__private::GroupLinkStorage;
/// use hdf5_pure_format::__private::LinkInfoMessage;
/// use hdf5_pure_format::__private::OwnedLink;
/// use hdf5_pure_format::__private::OwnedLinkTarget;
///
/// let links = vec![OwnedLink::new(
///     b"alias".to_vec(),
///     CharacterSet::Ascii,
///     OwnedLinkTarget::soft(b"/data".to_vec()),
///     None,
/// )?];
/// let info = LinkInfoMessage::compact(OffsetWidth::Eight, None, false);
/// let storage = GroupLinkStorage::new(&info, links)?;
/// let plan = DenseLinkPlan::new(storage, LengthWidth::Eight)?;
/// let allocation_len = plan.serialized_len();
/// let heap_address = StoredAddress::new(4096);
/// let image = plan.serialize(heap_address)?;
/// assert_eq!(image.bytes.len() as u64, allocation_len);
/// assert_eq!(image.info, plan.link_info(heap_address)?);
///
/// let mut cursor = 8192;
/// let addresses: Vec<_> = plan.block_lengths().map(|length| {
///     let address = StoredAddress::new(cursor);
///     cursor += length + 128;
///     address
/// }).collect();
/// let placement = plan.place(&addresses)?;
/// assert_eq!(placement.link_info().fractal_heap_address(), Some(addresses[0]));
/// for ((address, length), encoded) in placement.blocks().zip(placement.serialize_blocks()?) {
///     let (stored_address, bytes) = encoded?;
///     assert_eq!(stored_address, address);
///     assert_eq!(bytes.len() as u64, length);
/// }
/// # Ok::<(), hdf5_pure_format::__private::DenseLinkPlanError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub struct DenseLinkPlan {
    offset_width: OffsetWidth,
    length_width: LengthWidth,
    creation: LinkCreationPolicy,
    serialized: Vec<Vec<u8>>,
    header: FractalHeapHeader,
    heap: ManagedHeapPlan,
    name: PlannedIndex,
    creation_index: Option<PlannedIndex>,
    heap_offset: u64,
    total_len: u64,
}

impl DenseLinkPlan {
    /// Encodes the links and plans their heap and indexes without allocating file space.
    ///
    /// The address width comes from `storage`. Supports four-byte and eight-byte lengths and
    /// encoded link messages of at most 4096 bytes. The heap uses managed objects exclusively.
    /// Name-index records are sorted by hash, then by raw name bytes for equal hashes.
    ///
    /// # Errors
    ///
    /// Returns [`DenseLinkPlanError::Heap`] if the managed-heap limits are exceeded,
    /// [`DenseLinkPlanError::Format`] if a field exceeds its encoding or platform width or an
    /// encoded heap ID is invalid, and
    /// [`DenseLinkPlanError::LengthOverflow`] if layout arithmetic overflows. Returns
    /// [`DenseLinkPlanError::Link`] if link encoding fails, or
    /// [`DenseLinkPlanError::IndexGeometry`] if the index geometry cannot represent the links.
    pub fn new(
        storage: GroupLinkStorage,
        length_width: LengthWidth,
    ) -> Result<Self, DenseLinkPlanError> {
        let offset_width = storage.offset_width();
        let creation = storage.creation_order();
        let serialized = storage
            .links()
            .iter()
            .map(|link| link.serialize(offset_width))
            .collect::<Result<Vec<_>, _>>()?;
        let sizes = serialized
            .iter()
            .map(|bytes| bytes.len().to_u64())
            .collect::<Vec<_>>();
        let heap = ManagedHeapPlan::group_links_with_widths(
            &sizes,
            FormatWidths {
                offsets: offset_width,
                lengths: length_width,
            },
        )?;
        let header = Self::heap_header(offset_width, length_width, &heap, serialized.len())?;
        let id_layout = FractalHeapIdLayout::new(&header, offset_width, length_width);
        let ids = sizes
            .iter()
            .enumerate()
            .map(|(index, &size)| header.encode_managed_id(heap.heap_offset(index), size))
            .collect::<Result<Vec<_>, _>>()?;
        let entries = LinkIndexEntries::new(storage.links(), &id_layout, &ids)?;
        let name_records = entries.name_records()?;
        let heap_offset = FractalHeapHeader::serialized_size(offset_width, length_width).to_u64();
        let mut cursor = heap_offset
            .checked_add(heap.region_size())
            .ok_or(DenseLinkPlanError::LengthOverflow)?;
        let widths = FormatWidths {
            offsets: offset_width,
            lengths: length_width,
        };
        let name = PlannedIndex::new(
            widths,
            BTREE_V2_LINK_NAME,
            serialized.len(),
            NAME_RECORD_LEN,
            name_records,
            &mut cursor,
        )?;
        let creation_index = if creation.is_indexed() {
            let records = entries.creation_records()?;
            Some(PlannedIndex::new(
                widths,
                BTREE_V2_LINK_CREATION_ORDER,
                serialized.len(),
                CREATION_RECORD_LEN,
                records,
                &mut cursor,
            )?)
        } else {
            None
        };
        cursor.to_usize()?;
        Ok(Self {
            offset_width,
            length_width,
            creation,
            serialized,
            header,
            heap,
            name,
            creation_index,
            heap_offset,
            total_len: cursor,
        })
    }

    /// Returns each complete metadata block's serialized length in placement order.
    ///
    /// The heap header precedes the heap blocks, followed by the name index and the optional
    /// creation-order index, each with its header before its nodes. Heap blocks follow
    /// [`ManagedHeapPlan::block_lengths`], and index nodes follow
    /// [`BTreeV2Plan::block_lengths`](crate::btree::v2::write::BTreeV2Plan::block_lengths).
    /// Pass addresses in this order to [`Self::place`].
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> + DoubleEndedIterator + '_ {
        (0..self.block_count()).map(|index| self.block_length(index))
    }

    /// Assigns independently selected stored addresses to the heap and index headers and blocks.
    ///
    /// Pass one address per length returned by [`Self::block_lengths`], in the same order.
    /// Complete blocks must fit the encoded address width and occupy disjoint ranges. A block's
    /// exclusive end may equal the undefined address value. The caller allocates file space and
    /// chooses any alignment or gaps between blocks. See [`DenseLinkPlan`] for an example.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DenseLinkBlockCount`] if the address count differs from the block
    /// count, and [`FormatError::DenseLinkBlockOverlap`] if complete block ranges overlap.
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a block's exclusive end exceeds `u64`,
    /// [`FormatError::AddressTooLarge`] if an address or exclusive end exceeds the encoded width,
    /// and [`FormatError::DefinedAddressIsUndefined`] if a block starts at the undefined address.
    pub fn place(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<DenseLinkPlacement<'_>, FormatError> {
        DenseLinkPlacement::new(self, addresses)
    }

    /// Returns the bytes required for the heap header, heap blocks, and index headers and nodes.
    pub fn serialized_len(&self) -> u64 {
        self.total_len
    }

    /// Returns Link Info metadata for storage beginning at the stored file address `base`.
    ///
    /// Preserves the creation-order policy and next-index counter. Checks the complete allocation
    /// extent before deriving the index addresses.
    ///
    /// # Errors
    ///
    /// Returns [`DenseLinkPlanError::Format`] if `base` is undefined, an address exceeds the
    /// stored address width, or address arithmetic overflows.
    pub fn link_info(&self, base: StoredAddress) -> Result<LinkInfoMessage, DenseLinkPlanError> {
        let mut info = self.compact_info();
        info.set_defined_address(LinkInfoAddressField::Heap, base)?;
        let end = base
            .get()
            .checked_add(self.total_len)
            .ok_or(FormatError::OffsetOverflow {
                offset: base.get(),
                length: self.total_len,
            })?;
        if end > StoredAddress::undefined(self.offset_width).get() {
            return Err(FormatError::AddressTooLarge {
                address: end,
                offset_size: self.offset_width.get(),
            }
            .into());
        }
        info.set_defined_address(
            LinkInfoAddressField::NameIndex,
            base.checked_add(self.name.offset())?,
        )?;
        if let Some(index) = &self.creation_index {
            info.set_defined_address(
                LinkInfoAddressField::CreationIndex,
                base.checked_add(index.offset())?,
            )?;
        }
        Ok(info)
    }

    /// Serializes the heap and indexes at the stored file address `base` into one buffer.
    ///
    /// The first byte is the fractal-heap header. Heap blocks follow, then the name index and
    /// the optional creation-order index, each with its header before its nodes. Empty indexes
    /// have an undefined root address and no nodes.
    ///
    /// # Errors
    ///
    /// Returns [`DenseLinkPlanError::Format`] if the output extent fails the checks of
    /// [`Self::link_info`], a field exceeds its encoding or platform width, or serialization
    /// differs from the planned layout.
    pub fn serialize(&self, base: StoredAddress) -> Result<DenseLinkImage, DenseLinkPlanError> {
        let info = self.link_info(base)?;
        let Self {
            offset_width,
            length_width,
            creation: _,
            serialized: _,
            header,
            heap,
            name,
            creation_index,
            heap_offset,
            total_len,
        } = self;
        let mut header = header.clone();
        let heap_address = self.address(base, *heap_offset)?;
        header.root_block_address = heap.root_address(heap_address)?;
        let mut bytes = Vec::with_capacity(total_len.to_usize()?);
        bytes.extend(header.serialize(*offset_width, *length_width)?);
        let objects = self.objects();
        Self::append(
            &mut bytes,
            *heap_offset,
            &heap.serialize(&objects, heap_address, base)?,
        )?;
        for index in core::iter::once(name).chain(creation_index.iter()) {
            let nodes_address = if index.records().is_empty() {
                StoredAddress::undefined(*offset_width)
            } else {
                self.address(base, index.nodes_offset())?
            };
            let image = index.tree().serialize(
                index.records(),
                nodes_address,
                *offset_width,
                *length_width,
            )?;
            Self::append(&mut bytes, index.offset(), &image.header)?;
            Self::append(&mut bytes, index.nodes_offset(), &image.nodes)?;
        }
        if bytes.len().to_u64() != *total_len {
            return Err(FormatError::Internal(
                "dense link emission length differs from its plan".into(),
            )
            .into());
        }
        Ok(DenseLinkImage { bytes, info })
    }

    pub(super) fn heap(&self) -> &ManagedHeapPlan {
        &self.heap
    }

    pub(super) fn header(&self) -> &FractalHeapHeader {
        &self.header
    }

    pub(super) fn widths(&self) -> FormatWidths {
        FormatWidths {
            offsets: self.offset_width,
            lengths: self.length_width,
        }
    }

    pub(super) fn name_index(&self) -> &PlannedIndex {
        &self.name
    }

    pub(super) fn creation_index(&self) -> Option<&PlannedIndex> {
        self.creation_index.as_ref()
    }

    pub(super) fn objects(&self) -> Vec<&[u8]> {
        self.serialized.iter().map(Vec::as_slice).collect()
    }

    pub(super) fn compact_info(&self) -> LinkInfoMessage {
        LinkInfoMessage::compact(self.offset_width, self.creation.next_index(), false)
    }

    pub(super) fn block_count(&self) -> usize {
        2 + self.heap.block_lengths().len()
            + self.name.tree().block_lengths().len()
            + self
                .creation_index
                .as_ref()
                .map_or(0, |index| 1 + index.tree().block_lengths().len())
    }

    #[expect(
        clippy::unreachable,
        reason = "the iterator bounds each index by the count of these same components"
    )]
    fn block_length(&self, mut index: usize) -> u64 {
        let Self {
            offset_width,
            length_width,
            creation: _,
            serialized: _,
            header: _,
            heap,
            name,
            creation_index,
            heap_offset,
            total_len: _,
        } = self;
        if index == 0 {
            return *heap_offset;
        }
        index -= 1;
        if let Some(length) = heap.block_lengths().nth(index) {
            return length;
        }
        index -= heap.block_lengths().len();
        for tree in core::iter::once(name).chain(creation_index.iter()) {
            if index == 0 {
                return btree_v2::btree_v2_header_size(*offset_width, *length_width).to_u64();
            }
            index -= 1;
            if let Some(length) = tree.tree().block_lengths().nth(index) {
                return length;
            }
            index -= tree.tree().block_lengths().len();
        }
        // The iterator supplies indexes below `block_count`, which counts these same components.
        unreachable!("block index is bounded by the complete block count")
    }

    fn heap_header(
        offset_width: OffsetWidth,
        length_width: LengthWidth,
        heap: &ManagedHeapPlan,
        count: usize,
    ) -> Result<FractalHeapHeader, DenseLinkPlanError> {
        let header = FractalHeapHeader {
            heap_id_length: HEAP_ID_LEN,
            flags: FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED,
            max_managed_object_size: u32::try_from(GROUP_HEAP_MAX_MANAGED_OBJECT)
                .map_err(|_error| DenseLinkPlanError::LengthOverflow)?,
            next_huge_object_id: 0,
            btree_huge_objects_address: StoredAddress::undefined(offset_width),
            free_space_in_managed_blocks: heap.free_space(),
            managed_block_free_space_manager_address: StoredAddress::undefined(offset_width),
            managed_space: heap.managed_space(),
            allocated_managed_space: heap.allocated_space(),
            direct_block_allocation_iterator_offset: heap.allocation_iterator(),
            managed_objects_count: count.to_u64(),
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: GROUP_HEAP_TABLE_WIDTH,
            starting_block_size: GROUP_HEAP_STARTING_BLOCK_SIZE,
            max_direct_block_size: crate::fractal_heap::write::maximum_direct_block_size(
                GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE,
                length_width,
            ),
            max_heap_size: GROUP_HEAP_MAX_HEAP_SIZE_BITS,
            start_root_rows: GROUP_HEAP_START_ROOT_ROWS,
            root_block_address: StoredAddress::new(0),
            current_rows_in_root_indirect_block: heap.root_rows(),
            filtering: FractalHeapFiltering::Unfiltered,
        };
        for value in [
            header.free_space_in_managed_blocks,
            header.managed_space,
            header.allocated_managed_space,
            header.direct_block_allocation_iterator_offset,
            header.managed_objects_count,
            header.starting_block_size,
            header.max_direct_block_size,
        ] {
            if !width::length_width_holds(length_width, value) {
                return Err(FormatError::LengthTooLarge {
                    length: value,
                    length_size: length_width.get(),
                }
                .into());
            }
        }
        Ok(header)
    }

    pub(super) fn address(
        &self,
        base: StoredAddress,
        offset: u64,
    ) -> Result<StoredAddress, FormatError> {
        let address = StoredAddress::new(base.get().checked_add(offset).ok_or(
            FormatError::OffsetOverflow {
                offset: base.get(),
                length: offset,
            },
        )?);
        if address.get() > width::offset_width_max(self.offset_width) {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: self.offset_width.get(),
            });
        }
        if address.is_undefined(self.offset_width) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: self.offset_width.get(),
            });
        }
        Ok(address)
    }

    fn append(bytes: &mut Vec<u8>, expected: u64, part: &[u8]) -> Result<(), DenseLinkPlanError> {
        if bytes.len().to_u64() != expected {
            return Err(FormatError::Internal(
                "dense link metadata offset differs from its plan".into(),
            )
            .into());
        }
        bytes.extend_from_slice(part);
        Ok(())
    }
}

/// A serialized dense group heap and its indexes, with matching Link Info metadata.
pub struct DenseLinkImage {
    /// The heap header, heap blocks, and indexes in one contiguous allocation.
    pub bytes: Vec<u8>,
    /// The storage addresses and creation-order policy for the group's object header.
    pub info: LinkInfoMessage,
}
