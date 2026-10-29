//! Relocatable dense group links in a managed fractal heap and version 2 B-tree indexes.
//!
//! [`DenseLinkPlan`] retains encoded links and lays out their heap and indexes before a caller
//! allocates file space. Serialization uses stored file addresses relative to the superblock's
//! base address.

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree_v2;
use crate::btree_v2::BTREE_V2_LINK_CREATION_ORDER;
use crate::btree_v2::BTREE_V2_LINK_NAME;
use crate::btree_v2_write::BTreeV2Plan;
use crate::btree_v2_write::BTreeV2WriteDialect;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap_write::GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE;
use crate::fractal_heap_write::GROUP_HEAP_MAX_HEAP_SIZE_BITS;
use crate::fractal_heap_write::GROUP_HEAP_MAX_MANAGED_OBJECT;
use crate::fractal_heap_write::GROUP_HEAP_START_ROOT_ROWS;
use crate::fractal_heap_write::GROUP_HEAP_STARTING_BLOCK_SIZE;
use crate::fractal_heap_write::GROUP_HEAP_TABLE_WIDTH;
use crate::fractal_heap_write::ManagedHeapPlan;
use crate::fractal_heap_write::ManagedHeapPlanError;
use crate::link_info::LinkInfoMessage;
use crate::link_storage::GroupLinkStorage;
use crate::link_storage::LinkCreationPolicy;
use crate::link_storage::LinkStorageError;
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
/// let info = LinkInfoMessage {
///     max_creation_order: None,
///     fractal_heap_address: None,
///     btree_name_index_address: None,
///     btree_creation_order_address: None,
/// };
/// let storage = GroupLinkStorage::new(&info, links, OffsetWidth::Eight)?;
/// let plan = DenseLinkPlan::new(storage, LengthWidth::Eight)?;
/// let allocation_len = plan.serialized_len();
/// let heap_address = StoredAddress::new(4096);
/// let image = plan.serialize(heap_address)?;
/// assert_eq!(image.bytes.len() as u64, allocation_len);
/// assert_eq!(image.info, plan.link_info(heap_address)?);
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
    /// [`DenseLinkPlanError::Format`] if a field exceeds its encoding or platform width, and
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
        let heap = ManagedHeapPlan::group_links(&sizes, offset_width)?;
        let header = Self::heap_header(offset_width, length_width, &heap, serialized.len())?;
        let ids = sizes
            .iter()
            .enumerate()
            .map(|(index, &size)| header.encode_managed_id(heap.heap_offset(index), size))
            .collect::<Result<Vec<_>, _>>()?;
        let mut order = storage
            .links()
            .iter()
            .enumerate()
            .map(|(index, link)| (checksum::jenkins_lookup3(link.name()), index))
            .collect::<Vec<_>>();
        order.sort_unstable_by(|&(hash_a, index_a), &(hash_b, index_b)| {
            hash_a.cmp(&hash_b).then_with(|| {
                storage.links()[index_a]
                    .name()
                    .cmp(storage.links()[index_b].name())
            })
        });
        let mut name_records = Vec::new();
        for (hash, index) in order {
            name_records.extend_from_slice(&hash.to_le_bytes());
            name_records.extend_from_slice(&ids[index]);
        }
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
            let mut order = storage
                .links()
                .iter()
                .enumerate()
                .map(|(index, link)| {
                    link.creation_order()
                        .map(|creation| (creation, index))
                        .ok_or(LinkStorageError::CreationOrderPresence)
                })
                .collect::<Result<Vec<_>, _>>()?;
            order.sort_unstable_by_key(|&(creation, _)| creation);
            let mut records = Vec::new();
            for (creation, index) in order {
                records.extend_from_slice(&creation.to_le_bytes());
                records.extend_from_slice(&ids[index]);
            }
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

    /// Returns the bytes required for the heap header, heap blocks, and index headers and nodes.
    pub fn serialized_len(&self) -> u64 {
        self.total_len
    }

    /// Returns Link Info metadata for storage beginning at the stored file address `base`.
    ///
    /// Preserves the creation-order policy and next-index counter. Checks the complete output
    /// extent before computing the heap and index addresses.
    ///
    /// # Errors
    ///
    /// Returns [`DenseLinkPlanError::Format`] if `base` is undefined, an address exceeds the
    /// stored address width, or address arithmetic overflows.
    pub fn link_info(&self, base: StoredAddress) -> Result<LinkInfoMessage, DenseLinkPlanError> {
        self.address(base, 0)?;
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
        Ok(LinkInfoMessage {
            max_creation_order: self.creation.next_index(),
            fractal_heap_address: Some(base),
            btree_name_index_address: Some(self.address(base, self.name.offset)?),
            btree_creation_order_address: self
                .creation_index
                .as_ref()
                .map(|index| self.address(base, index.offset).map(Some))
                .transpose()?,
        })
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
            serialized,
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
        let objects = serialized.iter().map(Vec::as_slice).collect::<Vec<_>>();
        Self::append(
            &mut bytes,
            *heap_offset,
            &heap.serialize(&objects, heap_address, base)?,
        )?;
        for index in core::iter::once(name).chain(creation_index.iter()) {
            let nodes_address = if index.records.is_empty() {
                StoredAddress::undefined(*offset_width)
            } else {
                self.address(base, index.nodes_offset)?
            };
            let image =
                index
                    .plan
                    .serialize(&index.records, nodes_address, *offset_width, *length_width);
            Self::append(&mut bytes, index.offset, &image.header)?;
            Self::append(&mut bytes, index.nodes_offset, &image.nodes)?;
        }
        if bytes.len().to_u64() != *total_len {
            return Err(FormatError::Internal(
                "dense link emission length differs from its plan".into(),
            )
            .into());
        }
        Ok(DenseLinkImage { bytes, info })
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
            max_direct_block_size: GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE,
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

    fn address(
        &self,
        base: StoredAddress,
        offset: u64,
    ) -> Result<StoredAddress, DenseLinkPlanError> {
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
            }
            .into());
        }
        if address.is_undefined(self.offset_width) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: self.offset_width.get(),
            }
            .into());
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

/// An error planning or serializing dense group links.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseLinkPlanError {
    /// A field or address cannot be encoded, or serialization differs from the plan.
    Format(FormatError),
    /// Managed-heap planning failed.
    Heap(ManagedHeapPlanError),
    /// The index geometry cannot represent the records.
    IndexGeometry,
    /// Layout arithmetic exceeded the range of its integer type.
    LengthOverflow,
    /// Link encoding or creation-order validation failed.
    Link(LinkStorageError),
}

impl From<FormatError> for DenseLinkPlanError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}
impl From<ManagedHeapPlanError> for DenseLinkPlanError {
    fn from(error: ManagedHeapPlanError) -> Self {
        Self::Heap(error)
    }
}
impl From<LinkStorageError> for DenseLinkPlanError {
    fn from(error: LinkStorageError) -> Self {
        Self::Link(error)
    }
}

struct PlannedIndex {
    plan: BTreeV2Plan,
    records: Vec<u8>,
    /// The header offset from the start of the dense-link allocation.
    offset: u64,
    /// The first node offset from the start of the dense-link allocation.
    nodes_offset: u64,
}

impl PlannedIndex {
    fn new(
        widths: FormatWidths,
        tree_type: u8,
        count: usize,
        record_len: u16,
        records: Vec<u8>,
        cursor: &mut u64,
    ) -> Result<Self, DenseLinkPlanError> {
        let FormatWidths {
            offsets: offset_width,
            lengths: length_width,
        } = widths;
        let expected = count
            .to_u64()
            .checked_mul(u64::from(record_len))
            .ok_or(DenseLinkPlanError::LengthOverflow)?;
        expected.to_usize()?;
        if records.len().to_u64() != expected {
            return Err(FormatError::Internal(
                "dense link index record length differs from its plan".into(),
            )
            .into());
        }
        let plan = BTreeV2Plan::new_with_undefined_empty_root(
            BTreeV2WriteDialect::Libhdf5Global,
            tree_type,
            count,
            record_len,
            GROUP_INDEX_NODE_SIZE,
            offset_width,
        )
        .ok_or(DenseLinkPlanError::IndexGeometry)?;
        let offset = *cursor;
        let nodes_offset = offset
            .checked_add(btree_v2::btree_v2_header_size(offset_width, length_width).to_u64())
            .ok_or(DenseLinkPlanError::LengthOverflow)?;
        *cursor = nodes_offset
            .checked_add(
                plan.checked_nodes_size()
                    .ok_or(DenseLinkPlanError::LengthOverflow)?,
            )
            .ok_or(DenseLinkPlanError::LengthOverflow)?;
        Ok(Self {
            plan,
            records,
            offset,
            nodes_offset,
        })
    }
}

// H5G__dense_create in H5Gdense.c, HDF5 2.2.0, uses seven-byte IDs and 512-byte index nodes.
const HEAP_ID_LEN: u16 = 7;
const GROUP_INDEX_NODE_SIZE: u32 = 512;
/// The four-byte name hash and seven-byte heap ID of a type-5 record.
///
/// Defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub(crate) const NAME_RECORD_LEN: u16 = 4 + HEAP_ID_LEN;
// "Version 2 B-trees", format specification version 4.0, defines an eight-byte creation index
// and seven-byte heap ID for type-6 records.
const CREATION_RECORD_LEN: u16 = 8 + HEAP_ID_LEN;

#[cfg(test)]
mod tests;
