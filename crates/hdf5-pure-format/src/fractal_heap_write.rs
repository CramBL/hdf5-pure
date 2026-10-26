//! Relocatable layouts and serialization for unfiltered managed fractal-heap blocks.
//!
//! [`ManagedHeapPlan`] places objects using [`FractalHeapGeometry`] and serializes the direct
//! and indirect blocks at caller-supplied stored addresses. The caller writes the heap header
//! and encodes the object heap IDs.

use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddressExt;
use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::bytes;
use crate::convert::{self, Narrow};
use crate::error::FormatError;
use crate::fractal_heap_block::{CHECKSUM_LEN, DIRECT_SIGNATURE, INDIRECT_SIGNATURE, VERSION};
use crate::fractal_heap_geometry::FractalHeapGeometry;
use crate::width::OffsetWidth;

/// The attribute-heap row width, `H5O_FHEAP_MAN_WIDTH` in `H5Oprivate.h` (HDF5 2.2.0).
pub const ATTRIBUTE_HEAP_TABLE_WIDTH: u16 = 4;

/// The attribute-heap starting block size in bytes.
///
/// libhdf5 defines `H5O_FHEAP_MAN_START_BLOCK_SIZE` in `H5Oprivate.h` (HDF5 2.2.0).
pub const ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE: u64 = 1024;

/// The attribute-heap maximum direct-block size in bytes.
///
/// libhdf5 defines `H5O_FHEAP_MAN_MAX_DIRECT_SIZE` in `H5Oprivate.h` (HDF5 2.2.0).
pub const ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE: u64 = 65_536;

/// The attribute-heap offset bit count, stored in the header's "Maximum Heap Size" field.
///
/// libhdf5 defines `H5O_FHEAP_MAN_MAX_INDEX` in `H5Oprivate.h` (HDF5 2.2.0).
pub const ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS: u16 = 40;

/// The width in bytes of the "Block Offset" field of a direct or indirect block, the
/// [`ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS`] of a heap offset rounded up to whole bytes.
pub const ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES: usize =
    (ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS as usize).div_ceil(8);

/// The attribute-heap header's starting root-indirect-block row count.
///
/// libhdf5 defines `H5O_FHEAP_MAN_START_ROOT_ROWS` in `H5Oprivate.h` (HDF5 2.2.0).
pub const ATTRIBUTE_HEAP_START_ROOT_ROWS: u16 = 1;

/// The attribute-heap address-space size, derived from [`ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS`].
pub const ATTRIBUTE_HEAP_MAX_HEAP_SPACE: u64 = 1u64 << ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;

/// The group-heap row width, `H5G_FHEAP_MAN_WIDTH` in `H5Gdense.c` (HDF5 2.2.0).
pub const GROUP_HEAP_TABLE_WIDTH: u16 = 4;
/// The group-heap starting block size in bytes.
///
/// libhdf5 defines `H5G_FHEAP_MAN_START_BLOCK_SIZE` in `H5Gdense.c` (HDF5 2.2.0).
pub const GROUP_HEAP_STARTING_BLOCK_SIZE: u64 = 512;
/// The group-heap maximum direct-block size in bytes.
///
/// libhdf5 defines `H5G_FHEAP_MAN_MAX_DIRECT_SIZE` in `H5Gdense.c` (HDF5 2.2.0).
pub const GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE: u64 = 65_536;
/// The group-heap offset bit count, stored in the header's "Maximum Heap Size" field.
///
/// libhdf5 defines `H5G_FHEAP_MAN_MAX_INDEX` in `H5Gdense.c` (HDF5 2.2.0).
pub const GROUP_HEAP_MAX_HEAP_SIZE_BITS: u16 = 32;
/// The group-heap header's starting root-indirect-block row count.
///
/// libhdf5 defines `H5G_FHEAP_MAN_START_ROOT_ROWS` in `H5Gdense.c` (HDF5 2.2.0).
pub const GROUP_HEAP_START_ROOT_ROWS: u16 = 1;
/// The group-heap maximum managed-object size in bytes.
///
/// libhdf5 defines `H5G_FHEAP_MAX_MAN_SIZE` in `H5Gdense.c` (HDF5 2.2.0).
pub const GROUP_HEAP_MAX_MANAGED_OBJECT: u64 = 4096;

/// Returns the checksummed attribute-heap direct-block prefix length at `offset_width`.
///
/// The fields are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
pub(crate) const fn attribute_heap_direct_block_header(offset_width: OffsetWidth) -> usize {
    DIRECT_SIGNATURE.len()
        + 1
        + offset_width.get() as usize
        + ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES
        + CHECKSUM_LEN
}

/// Returns the largest object that fits an attribute-heap direct block at `offset_width`.
pub const fn attribute_heap_max_managed_object(offset_width: OffsetWidth) -> usize {
    (1usize << ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE.trailing_zeros())
        - attribute_heap_direct_block_header(offset_width)
}

/// Reports a geometry, object-size, or address-space limit during managed-heap planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedHeapPlanError {
    /// The block geometry cannot represent the required blocks and object data.
    Geometry,
    /// The required blocks or their statistics exceed the heap's managed address space or `u64`.
    HeapSpace,
    /// The serialized region length exceeds [`usize::MAX`].
    Host {
        /// The required region length in bytes.
        bytes: u64,
    },
    /// An object is empty or exceeds the selected managed-object size limit.
    ObjectSize {
        /// The rejected object size in bytes.
        size: u64,
        /// The largest permitted object size in bytes.
        maximum: u64,
    },
}

/// An index into one of a plan's block lists.
#[derive(Clone, Copy)]
enum Child {
    Direct(usize),
    Indirect(usize),
}

/// A direct block's managed offset, serialized offset, and object indices.
struct PlannedDirect {
    heap_offset: u64,
    size: u64,
    region_offset: u64,
    objects: Vec<usize>,
}

/// An indirect block's managed offset, serialized offset, and child-block indices.
struct PlannedIndirect {
    heap_offset: u64,
    nrows: u16,
    region_offset: u64,
    entries: Vec<Option<Child>>,
}

/// A relocatable layout of unfiltered version 0 managed fractal-heap blocks.
///
/// Objects are placed in doubling-table order. The layout includes direct and indirect blocks,
/// their serialized offsets, each object's managed heap offset, and the heap statistics.
/// [`serialize`](Self::serialize) writes the blocks with checksums at stored addresses supplied
/// by the caller. The heap header and heap IDs are encoded separately.
///
/// These blocks are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
pub struct ManagedHeapPlan {
    offset_width: OffsetWidth,
    geometry: FractalHeapGeometry,
    block_offset_bytes: usize,
    direct_header_size: u64,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    directs: Vec<PlannedDirect>,
    indirects: Vec<PlannedIndirect>,
    region_size: u64,
    managed_space: u64,
    allocated_space: u64,
    allocation_iterator: u64,
    free_space: u64,
}

impl ManagedHeapPlan {
    /// Lays out objects of `sizes` bytes using `geometry` and the encoded file-address width.
    ///
    /// Objects occupy consecutive bytes within a direct block. A direct-block slot whose data
    /// area is too small for an object is left unallocated. An empty object list produces one
    /// empty starting-size direct block.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::Geometry`] if a starting block has no data space after its
    /// metadata prefix, and [`ManagedHeapPlanError::ObjectSize`] if an object is empty or cannot
    /// fit a reachable direct block.
    ///
    /// Returns [`ManagedHeapPlanError::HeapSpace`] if placement or a statistic exceeds the heap's
    /// address space or `u64`, and [`ManagedHeapPlanError::Host`] if the serialized region length
    /// exceeds [`usize::MAX`].
    pub fn new(
        sizes: &[u64],
        offset_width: OffsetWidth,
        geometry: FractalHeapGeometry,
    ) -> Result<Self, ManagedHeapPlanError> {
        let block_offset_bytes = usize::from(geometry.max_heap_size()).div_ceil(8);
        let direct_header_size = (DIRECT_SIGNATURE.len()
            + 1
            + usize::from(offset_width.get())
            + block_offset_bytes
            + CHECKSUM_LEN) as u64;
        if geometry.starting_block_size() <= direct_header_size {
            return Err(ManagedHeapPlanError::Geometry);
        }
        let last_direct = geometry
            .direct_rows()
            .min(usize::from(geometry.max_root_rows()))
            - 1;
        let maximum = geometry
            .block_size_for_row(last_direct)
            .map_err(|_error| ManagedHeapPlanError::Geometry)?
            .checked_sub(direct_header_size)
            .ok_or(ManagedHeapPlanError::Geometry)?;
        for &size in sizes {
            if size == 0 || size > maximum {
                return Err(ManagedHeapPlanError::ObjectSize { size, maximum });
            }
        }
        let mut plan = Self {
            offset_width,
            geometry,
            block_offset_bytes,
            direct_header_size,
            offsets: Vec::with_capacity(sizes.len()),
            sizes: sizes.to_vec(),
            directs: Vec::new(),
            indirects: Vec::new(),
            region_size: 0,
            managed_space: 0,
            allocated_space: 0,
            allocation_iterator: 0,
            free_space: 0,
        };
        plan.place_objects()?;
        plan.place_blocks()?;
        Ok(plan)
    }

    /// Lays out objects with the attribute-heap geometry.
    ///
    /// libhdf5 sets this geometry in `H5A__dense_create` in `H5Adense.c` (HDF5 2.2.0).
    /// The object-size limit is [`attribute_heap_max_managed_object`].
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::ObjectSize`] if an object is empty or exceeds
    /// [`attribute_heap_max_managed_object`], [`ManagedHeapPlanError::HeapSpace`] if placement or
    /// a statistic exceeds the heap's address space or `u64`, and [`ManagedHeapPlanError::Host`]
    /// if the serialized region length exceeds [`usize::MAX`].
    pub fn attributes(
        sizes: &[u64],
        offset_width: OffsetWidth,
    ) -> Result<Self, ManagedHeapPlanError> {
        let geometry = FractalHeapGeometry::new(
            ATTRIBUTE_HEAP_TABLE_WIDTH,
            ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
            ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE,
            ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS,
        )
        .map_err(|_error| ManagedHeapPlanError::Geometry)?;
        Self::new(sizes, offset_width, geometry)
    }

    /// Lays out objects with the group-link heap geometry and managed-object limit.
    ///
    /// libhdf5 sets these parameters in `H5G__dense_create` in `H5Gdense.c` (HDF5 2.2.0).
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::ObjectSize`] if an object is empty or exceeds
    /// [`GROUP_HEAP_MAX_MANAGED_OBJECT`], [`ManagedHeapPlanError::HeapSpace`] if placement or a
    /// statistic exceeds the heap's address space or `u64`, and [`ManagedHeapPlanError::Host`]
    /// if the serialized region length exceeds [`usize::MAX`].
    pub fn group_links(
        sizes: &[u64],
        offset_width: OffsetWidth,
    ) -> Result<Self, ManagedHeapPlanError> {
        if let Some(&size) = sizes
            .iter()
            .find(|&&size| size == 0 || size > GROUP_HEAP_MAX_MANAGED_OBJECT)
        {
            return Err(ManagedHeapPlanError::ObjectSize {
                size,
                maximum: GROUP_HEAP_MAX_MANAGED_OBJECT,
            });
        }
        let geometry = FractalHeapGeometry::new(
            GROUP_HEAP_TABLE_WIDTH,
            GROUP_HEAP_STARTING_BLOCK_SIZE,
            GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE,
            GROUP_HEAP_MAX_HEAP_SIZE_BITS,
        )
        .map_err(|_error| ManagedHeapPlanError::Geometry)?;
        Self::new(sizes, offset_width, geometry)
    }

    /// Returns the serialized length of all blocks, excluding the heap header.
    pub fn region_size(&self) -> u64 {
        self.region_size
    }

    /// Returns the root block's stored address when the block region starts at `region_address`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if adding the root offset overflows `u64`,
    /// [`FormatError::AddressTooLarge`] if the address exceeds the encoded file-address width,
    /// and [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined value.
    pub fn root_address(
        &self,
        region_address: StoredAddress,
    ) -> Result<StoredAddress, FormatError> {
        let root_offset = self
            .indirects
            .last()
            .map_or(self.directs[0].region_offset, |root| root.region_offset);
        self.address_at(region_address, root_offset)
    }

    /// Returns the root indirect block's row count, or zero for a direct root.
    pub fn root_rows(&self) -> u16 {
        self.indirects.last().map_or(0, |root| root.nrows)
    }

    /// Returns the managed address-space size, including unallocated direct-block slots.
    pub fn managed_space(&self) -> u64 {
        self.managed_space
    }

    /// Returns the total length of allocated direct blocks, including their metadata prefixes.
    pub fn allocated_space(&self) -> u64 {
        self.allocated_space
    }

    /// Returns the next direct-block allocation offset, or zero for a direct root.
    pub fn allocation_iterator(&self) -> u64 {
        self.allocation_iterator
    }

    /// Returns the managed free space, including capacity in unallocated direct-block slots.
    pub fn free_space(&self) -> u64 {
        self.free_space
    }

    /// Returns the managed heap offset of the object at `index` in the planned input list.
    ///
    /// # Panics
    ///
    /// Panics if `index` is outside the input list.
    pub fn heap_offset(&self, index: usize) -> u64 {
        self.offsets[index]
    }

    /// Serializes the blocks at `region_address` with their owner set to `heap_header_address`.
    ///
    /// Both addresses are stored file addresses. The returned buffer has
    /// [`region_size`](Self::region_size) bytes. Unused direct-block data bytes are zeroed, and
    /// each block includes its checksum. The caller sets the heap header's direct-block checksum
    /// flag.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the object count differs from the plan and
    /// [`FormatError::DataSizeMismatch`] if an object's length differs from its planned size.
    /// Returns [`FormatError::OffsetOverflow`] if a block address or the region end overflows
    /// `u64`, [`FormatError::AddressTooLarge`] if an address or the region extent exceeds the
    /// encoded file-address width, and [`FormatError::DefinedAddressIsUndefined`] if a required
    /// address equals that width's undefined value.
    pub fn serialize(
        &self,
        objects: &[&[u8]],
        region_address: StoredAddress,
        heap_header_address: StoredAddress,
    ) -> Result<Vec<u8>, FormatError> {
        if objects.len() != self.sizes.len() {
            return Err(FormatError::Internal(
                "managed heap emission received another object count".into(),
            ));
        }
        for (object, &size) in objects.iter().zip(&self.sizes) {
            if object.len() as u64 != size {
                return Err(FormatError::DataSizeMismatch {
                    expected: size.to_usize()?,
                    actual: object.len(),
                });
            }
        }
        self.address_at(region_address, 0)?;
        self.address_at(heap_header_address, 0)?;
        let end = region_address.get().checked_add(self.region_size).ok_or(
            FormatError::OffsetOverflow {
                offset: region_address.get(),
                length: self.region_size,
            },
        )?;
        if end > StoredAddress::undefined(self.offset_width).get() {
            return Err(FormatError::AddressTooLarge {
                address: end,
                offset_size: self.offset_width.get(),
            });
        }
        let mut region = vec![0; self.region_size.to_usize()?];
        for block in &self.indirects {
            let mut data = Vec::new();
            data.extend_from_slice(&INDIRECT_SIGNATURE);
            data.push(VERSION);
            self.write_address(&mut data, heap_header_address)?;
            self.write_heap_offset(&mut data, block.heap_offset)?;
            for entry in &block.entries {
                let address = match entry {
                    Some(Child::Direct(index)) => {
                        Some(self.address_at(region_address, self.directs[*index].region_offset)?)
                    }
                    Some(Child::Indirect(index)) => {
                        Some(self.address_at(region_address, self.indirects[*index].region_offset)?)
                    }
                    None => None,
                };
                bytes::try_write_decoded_address(
                    &mut data,
                    address.map_or(
                        DecodedStoredAddress::Undefined,
                        DecodedStoredAddress::Defined,
                    ),
                    self.offset_width,
                )?;
            }
            let checksum = crate::checksum::jenkins_lookup3(&data);
            data.extend_from_slice(&checksum.to_le_bytes());
            debug_assert_eq!(Ok(data.len() as u64), self.indirect_block_size(block.nrows));
            place(&mut region, block.region_offset, &data)?;
        }
        for block in &self.directs {
            let size = block.size.to_usize()?;
            let mut data = Vec::with_capacity(size);
            data.extend_from_slice(&DIRECT_SIGNATURE);
            data.push(VERSION);
            self.write_address(&mut data, heap_header_address)?;
            self.write_heap_offset(&mut data, block.heap_offset)?;
            let checksum_at = data.len();
            data.extend_from_slice(&[0; CHECKSUM_LEN]);
            debug_assert_eq!(data.len() as u64, self.direct_header_size);
            for &index in &block.objects {
                debug_assert_eq!(
                    block.heap_offset.checked_add(data.len() as u64),
                    Some(self.offsets[index])
                );
                data.extend_from_slice(objects[index]);
            }
            data.resize(size, 0);
            let checksum = crate::checksum::jenkins_lookup3(&data);
            data[checksum_at..checksum_at + CHECKSUM_LEN].copy_from_slice(&checksum.to_le_bytes());
            place(&mut region, block.region_offset, &data)?;
        }
        Ok(region)
    }

    /// Places objects in managed heap order and computes the heap statistics.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::HeapSpace`] if placement or statistics overflow the heap's
    /// address space or `u64`, and [`ManagedHeapPlanError::Geometry`] if a slot cannot contain its
    /// metadata prefix or its child rows cannot be derived.
    fn place_objects(&mut self) -> Result<(), ManagedHeapPlanError> {
        let mut cursor = 0;
        let mut fill = 0;
        for (index, &size) in self.sizes.iter().enumerate() {
            loop {
                if let Some(block) = self.directs.last_mut() {
                    if block
                        .size
                        .checked_sub(fill)
                        .ok_or(ManagedHeapPlanError::HeapSpace)?
                        >= size
                    {
                        block.objects.push(index);
                        self.offsets.push(
                            block
                                .heap_offset
                                .checked_add(fill)
                                .ok_or(ManagedHeapPlanError::HeapSpace)?,
                        );
                        fill = fill
                            .checked_add(size)
                            .ok_or(ManagedHeapPlanError::HeapSpace)?;
                        break;
                    }
                }
                let (heap_offset, block_size) = self.locate(cursor)?;
                if self.geometry.max_heap_size() < 64
                    && heap_offset >= 1u64 << self.geometry.max_heap_size()
                {
                    return Err(ManagedHeapPlanError::HeapSpace);
                }
                cursor = heap_offset
                    .checked_add(block_size)
                    .ok_or(ManagedHeapPlanError::HeapSpace)?;
                if block_size
                    .checked_sub(self.direct_header_size)
                    .ok_or(ManagedHeapPlanError::Geometry)?
                    >= size
                {
                    self.directs.push(PlannedDirect {
                        heap_offset,
                        size: block_size,
                        region_offset: 0,
                        objects: Vec::new(),
                    });
                    fill = self.direct_header_size;
                }
            }
        }
        if self.directs.is_empty() {
            self.directs.push(PlannedDirect {
                heap_offset: 0,
                size: self.geometry.starting_block_size(),
                region_offset: 0,
                objects: Vec::new(),
            });
            cursor = self.geometry.starting_block_size();
        }
        let root_is_direct = self.directs.len() == 1
            && self.directs[0].heap_offset == 0
            && self.directs[0].size == self.geometry.starting_block_size();
        let capacity = if root_is_direct {
            self.managed_space = self.geometry.starting_block_size();
            self.managed_space
                .checked_sub(self.direct_header_size)
                .ok_or(ManagedHeapPlanError::Geometry)?
        } else {
            let nrows = (1..=self.geometry.max_root_rows())
                .find(|&rows| {
                    self.geometry
                        .indirect_heap_size(rows)
                        .is_ok_and(|span| span >= cursor)
                })
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
            let mut placed = 0;
            self.build_indirect(0, nrows, &mut placed)?;
            debug_assert_eq!(placed, self.directs.len(), "every block belongs to a slot");
            self.managed_space = self
                .geometry
                .indirect_heap_size(nrows)
                .map_err(|_error| ManagedHeapPlanError::HeapSpace)?;
            self.allocation_iterator = cursor;
            self.rows_capacity(nrows)?
        };
        let used = self
            .sizes
            .iter()
            .try_fold(0u64, |total, &size| total.checked_add(size))
            .ok_or(ManagedHeapPlanError::HeapSpace)?;
        self.free_space = capacity
            .checked_sub(used)
            .ok_or(ManagedHeapPlanError::HeapSpace)?;
        Ok(())
    }

    /// Assigns serialized offsets to the indirect and direct blocks.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::HeapSpace`] if a serialized length exceeds `u64`, and
    /// [`ManagedHeapPlanError::Host`] if the complete region length exceeds [`usize::MAX`].
    fn place_blocks(&mut self) -> Result<(), ManagedHeapPlanError> {
        for index in 0..self.indirects.len() {
            let size = self.indirect_block_size(self.indirects[index].nrows)?;
            let block = &mut self.indirects[index];
            block.region_offset = self.region_size;
            self.region_size = self
                .region_size
                .checked_add(size)
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
        }
        for block in &mut self.directs {
            block.region_offset = self.region_size;
            self.region_size = self
                .region_size
                .checked_add(block.size)
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
            self.allocated_space = self
                .allocated_space
                .checked_add(block.size)
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
        }
        if usize::try_from(self.region_size).is_err() {
            return Err(ManagedHeapPlanError::Host {
                bytes: self.region_size,
            });
        }
        Ok(())
    }

    /// Builds an indirect subtree and returns its index, advancing `placed` past its direct blocks.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::HeapSpace`] if child-slot arithmetic overflows, and
    /// [`ManagedHeapPlanError::Geometry`] if the child geometry cannot be derived.
    fn build_indirect(
        &mut self,
        base: u64,
        nrows: u16,
        placed: &mut usize,
    ) -> Result<usize, ManagedHeapPlanError> {
        let width = self.geometry.table_width();
        let entry_count = usize::from(nrows)
            .checked_mul(width)
            .ok_or(ManagedHeapPlanError::HeapSpace)?;
        let mut entries = vec![None; entry_count];
        'rows: for row in 0..usize::from(nrows) {
            let block_size = self
                .geometry
                .block_size_for_row(row)
                .map_err(|_error| ManagedHeapPlanError::Geometry)?;
            for column in 0..width {
                let Some(next) = self.directs.get(*placed) else {
                    break 'rows;
                };
                let slot = self
                    .geometry
                    .slot_heap_offset(base, row, column)
                    .map_err(|_error| ManagedHeapPlanError::HeapSpace)?;
                let end = slot
                    .checked_add(block_size)
                    .ok_or(ManagedHeapPlanError::HeapSpace)?;
                if next.heap_offset >= end {
                    continue;
                }
                let child = if row < self.geometry.direct_rows() {
                    debug_assert_eq!(next.heap_offset, slot, "a block fills its whole slot");
                    let index = *placed;
                    *placed = placed
                        .checked_add(1)
                        .ok_or(ManagedHeapPlanError::HeapSpace)?;
                    Child::Direct(index)
                } else {
                    let child_rows = self
                        .geometry
                        .child_indirect_rows(row)
                        .map_err(|_error| ManagedHeapPlanError::Geometry)?;
                    Child::Indirect(self.build_indirect(slot, child_rows, placed)?)
                };
                let index = row
                    .checked_mul(width)
                    .and_then(|index| index.checked_add(column))
                    .ok_or(ManagedHeapPlanError::HeapSpace)?;
                entries[index] = Some(child);
            }
        }
        self.indirects.push(PlannedIndirect {
            heap_offset: base,
            nrows,
            region_offset: 0,
            entries,
        });
        Ok(self.indirects.len() - 1)
    }

    /// Returns the managed offset and size of the direct-block slot containing `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::HeapSpace`] if `offset` is outside the doubling table or
    /// slot arithmetic overflows, and [`ManagedHeapPlanError::Geometry`] if a row cannot be read.
    fn locate(&self, offset: u64) -> Result<(u64, u64), ManagedHeapPlanError> {
        let mut base = 0u64;
        let mut local = offset;
        loop {
            let row = (0..usize::from(self.geometry.max_root_rows()))
                .rev()
                .find(|&row| {
                    self.geometry
                        .row_offset(row)
                        .is_ok_and(|start| start <= local)
                })
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
            let size = self
                .geometry
                .block_size_for_row(row)
                .map_err(|_error| ManagedHeapPlanError::Geometry)?;
            let start = self
                .geometry
                .row_offset(row)
                .map_err(|_error| ManagedHeapPlanError::Geometry)?;
            let column = usize::try_from(
                local
                    .checked_sub(start)
                    .ok_or(ManagedHeapPlanError::HeapSpace)?
                    / size,
            )
            .map_err(|_error| ManagedHeapPlanError::HeapSpace)?;
            let within = self
                .geometry
                .slot_offset(row, column)
                .map_err(|_error| ManagedHeapPlanError::HeapSpace)?;
            base = base
                .checked_add(within)
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
            if row < self.geometry.direct_rows() {
                return Ok((base, size));
            }
            local = local
                .checked_sub(within)
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
        }
    }

    /// Returns the object capacity of `nrows` rows, including unallocated direct-block slots.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::HeapSpace`] if capacity arithmetic overflows, and
    /// [`ManagedHeapPlanError::Geometry`] if a row's direct-block capacity cannot be derived.
    fn rows_capacity(&self, nrows: u16) -> Result<u64, ManagedHeapPlanError> {
        let width = u64::try_from(self.geometry.table_width())
            .map_err(|_error| ManagedHeapPlanError::Geometry)?;
        let mut cumulative = Vec::with_capacity(usize::from(nrows));
        let mut total = 0u64;
        for row in 0..usize::from(nrows) {
            let free = if row < self.geometry.direct_rows() {
                self.geometry
                    .block_size_for_row(row)
                    .map_err(|_error| ManagedHeapPlanError::Geometry)?
                    .checked_sub(self.direct_header_size)
                    .ok_or(ManagedHeapPlanError::Geometry)?
            } else {
                let child_rows = self
                    .geometry
                    .child_indirect_rows(row)
                    .map_err(|_error| ManagedHeapPlanError::Geometry)?;
                *cumulative
                    .get(usize::from(child_rows - 1))
                    .ok_or(ManagedHeapPlanError::Geometry)?
            };
            total = total
                .checked_add(
                    width
                        .checked_mul(free)
                        .ok_or(ManagedHeapPlanError::HeapSpace)?,
                )
                .ok_or(ManagedHeapPlanError::HeapSpace)?;
            cumulative.push(total);
        }
        Ok(total)
    }

    /// Returns the encoded indirect-block length for `nrows` rows at this plan's field widths.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::HeapSpace`] if the encoded length exceeds `u64`.
    fn indirect_block_size(&self, nrows: u16) -> Result<u64, ManagedHeapPlanError> {
        let width = u64::try_from(self.geometry.table_width())
            .map_err(|_error| ManagedHeapPlanError::Geometry)?;
        let block_offset_bytes = u64::try_from(self.block_offset_bytes)
            .map_err(|_error| ManagedHeapPlanError::Geometry)?;
        let entries = u64::from(nrows)
            .checked_mul(width)
            .and_then(|count| count.checked_mul(u64::from(self.offset_width.get())))
            .ok_or(ManagedHeapPlanError::HeapSpace)?;
        (INDIRECT_SIGNATURE.len() as u64)
            .checked_add(1)
            .and_then(|size| size.checked_add(u64::from(self.offset_width.get())))
            .and_then(|size| size.checked_add(block_offset_bytes))
            .and_then(|size| size.checked_add(entries))
            .and_then(|size| size.checked_add(CHECKSUM_LEN as u64))
            .ok_or(ManagedHeapPlanError::HeapSpace)
    }

    /// Adds a serialized block offset to a stored address and checks its encoded width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if addition overflows `u64`,
    /// [`FormatError::AddressTooLarge`] if the address exceeds the encoded width, and
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined value.
    fn address_at(&self, base: StoredAddress, offset: u64) -> Result<StoredAddress, FormatError> {
        let address = base
            .get()
            .checked_add(offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: base.get(),
                length: offset,
            })?;
        let undefined = StoredAddress::undefined(self.offset_width).get();
        if address > undefined {
            return Err(FormatError::AddressTooLarge {
                address,
                offset_size: self.offset_width.get(),
            });
        }
        if address == undefined {
            return Err(FormatError::DefinedAddressIsUndefined {
                address,
                offset_size: self.offset_width.get(),
            });
        }
        Ok(StoredAddress::new(address))
    }

    fn write_address(&self, data: &mut Vec<u8>, address: StoredAddress) -> Result<(), FormatError> {
        bytes::try_write_decoded_address(
            data,
            DecodedStoredAddress::Defined(address),
            self.offset_width,
        )
    }

    fn write_heap_offset(&self, data: &mut Vec<u8>, offset: u64) -> Result<(), FormatError> {
        let encoded = offset.to_le_bytes();
        data.extend_from_slice(
            encoded
                .get(..self.block_offset_bytes)
                .ok_or(FormatError::InvalidFractalHeapGeometry)?,
        );
        Ok(())
    }
}

fn place(region: &mut [u8], offset: u64, data: &[u8]) -> Result<(), FormatError> {
    let range = convert::slice_range(offset, data.len() as u64)?;
    let available = region.len();
    region
        .get_mut(range)
        .ok_or(FormatError::UnexpectedEof {
            expected: offset.to_usize()?,
            available,
        })?
        .copy_from_slice(data);
    Ok(())
}

#[cfg(test)]
mod tests {
    use hdf5_pure_core::__private::{FileRange, FileRegion, PhysicalAddress};
    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use crate::fractal_heap::{
        FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED, FractalHeapHeader, FractalHeapIdKind,
        FractalHeapIdLayout,
    };
    use crate::fractal_heap_block::{FractalHeapBlock, FractalHeapBlockLayout};
    use crate::width::LengthWidth;

    use super::*;

    fn header(plan: &ManagedHeapPlan, region_address: StoredAddress) -> FractalHeapHeader {
        let mut header =
            FractalHeapHeader::parse(&fractal_heap::Header::new(0).build(Widths::EIGHT), 0, 8, 8)
                .unwrap();
        header.flags = FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
        header.btree_huge_objects_address = StoredAddress::undefined(plan.offset_width);
        header.managed_block_free_space_manager_address =
            StoredAddress::undefined(plan.offset_width);
        header.heap_id_length = 12;
        header.table_width = u16::try_from(plan.geometry.table_width()).unwrap();
        header.starting_block_size = plan.geometry.starting_block_size();
        header.max_direct_block_size = plan.geometry.max_direct_block_size();
        header.max_heap_size = plan.geometry.max_heap_size();
        header.max_managed_object_size =
            u32::try_from(header.max_direct_block_size - plan.direct_header_size).unwrap();
        header.root_block_address = plan.root_address(region_address).unwrap();
        header.current_rows_in_root_indirect_block = plan.root_rows();
        header.managed_space = plan.managed_space();
        header.allocated_managed_space = plan.allocated_space();
        header.free_space_in_managed_blocks = plan.free_space();
        header.direct_block_allocation_iterator_offset = plan.allocation_iterator();
        header.managed_objects_count = plan.sizes.len() as u64;
        header
    }

    #[rstest]
    #[case::empty_attributes(1024, 65_536, 40, vec![], OffsetWidth::Eight)]
    #[case::attributes_direct(1024, 65_536, 40, vec![1002], OffsetWidth::Eight)]
    #[case::attributes_indirect(1024, 65_536, 40, vec![1002, 1], OffsetWidth::Eight)]
    #[case::group_direct(512, 65_536, 32, vec![491], OffsetWidth::Eight)]
    #[case::group_indirect(512, 65_536, 32, vec![491, 1], OffsetWidth::Eight)]
    #[case::skip_small_blocks(1024, 65_536, 40, vec![65_514; 3], OffsetWidth::Eight)]
    #[case::nested_attributes(1024, 65_536, 40, vec![65_514; 40], OffsetWidth::Eight)]
    #[case::deep_small_geometry(64, 128, 16, vec![100; 100], OffsetWidth::Eight)]
    #[case::non_byte_aligned_heap_offsets(512, 65_536, 33, vec![491; 8], OffsetWidth::Eight)]
    #[case::two_byte_addresses(512, 65_536, 32, vec![100; 50], OffsetWidth::Two)]
    #[case::four_byte_addresses(1024, 65_536, 40, vec![65_518; 40], OffsetWidth::Four)]
    fn planned_blocks_and_objects_conform_to_the_format_parser(
        #[case] start: u64,
        #[case] max_direct: u64,
        #[case] bits: u16,
        #[case] sizes: Vec<u64>,
        #[values(512, 0x2000)] base: u64,
        #[case] width: OffsetWidth,
    ) {
        let geometry = FractalHeapGeometry::new(4, start, max_direct, bits).unwrap();
        let plan = ManagedHeapPlan::new(&sizes, width, geometry).unwrap();
        let objects: Vec<Vec<u8>> = sizes
            .iter()
            .enumerate()
            .map(|(index, &size)| {
                vec![u8::try_from(index).unwrap(); usize::try_from(size).unwrap()]
            })
            .collect();
        let borrowed: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
        let region_address = StoredAddress::new(base);
        let owner = StoredAddress::new(128);
        let emitted = plan.serialize(&borrowed, region_address, owner).unwrap();
        assert_eq!(emitted.len() as u64, plan.region_size());
        let header = header(&plan, region_address);
        let header_bytes = header.serialize(width, LengthWidth::Eight).unwrap();
        let parsed_header = FractalHeapHeader::parse(&header_bytes, 0, width.get(), 8).unwrap();
        assert_eq!(parsed_header, header);
        let layout =
            FractalHeapBlockLayout::new(&parsed_header, width, LengthWidth::Eight, owner).unwrap();
        let image = FileRegion::new(
            FileRange::new(PhysicalAddress::new(base), emitted.len() as u64).unwrap(),
            &emitted,
        )
        .unwrap();
        let ids = FractalHeapIdLayout::new(&parsed_header, width, LengthWidth::Eight);
        let mut blocks = vec![layout.root().unwrap().unwrap()];
        let mut object_indices = Vec::new();
        let mut direct_count = 0;
        let mut indirect_count = 0;
        let mut allocated = 0;
        while let Some(block) = blocks.pop() {
            match block {
                FractalHeapBlock::Direct(block) => {
                    direct_count += 1;
                    allocated += block.decoded_size();
                    let reference = block
                        .at(PhysicalAddress::new(block.address().get()))
                        .unwrap();
                    reference.parse_checked(image).unwrap();
                    for (index, object) in objects.iter().enumerate() {
                        let offset = plan.heap_offset(index);
                        if offset < block.heap_offset()
                            || offset >= block.heap_offset() + block.decoded_size()
                        {
                            continue;
                        }
                        let id = parsed_header
                            .encode_managed_id(offset, object.len() as u64)
                            .unwrap();
                        assert_eq!(
                            ids.parse(&id).unwrap().kind(),
                            FractalHeapIdKind::Managed {
                                heap_offset: offset,
                                object_length: object.len() as u64
                            }
                        );
                        let object_ref = reference
                            .parse_object(image, offset, object.len() as u64)
                            .unwrap();
                        assert_eq!(object_ref.parse(image).unwrap(), object);
                        object_indices.push(index);
                    }
                }
                FractalHeapBlock::Indirect(block) => {
                    indirect_count += 1;
                    let view = block
                        .at(PhysicalAddress::new(block.address().get()))
                        .unwrap()
                        .parse(image)
                        .unwrap();
                    let children: Vec<_> = view.children().map(Result::unwrap).collect();
                    assert!(!children.is_empty());
                    blocks.extend(children);
                }
            }
        }
        object_indices.sort_unstable();
        assert_eq!(object_indices, (0..sizes.len()).collect::<Vec<_>>());
        assert_eq!(direct_count, plan.directs.len());
        assert_eq!(indirect_count, plan.indirects.len());
        assert_eq!(allocated, plan.allocated_space());
        assert_eq!(
            plan.allocated_space()
                + plan
                    .indirects
                    .iter()
                    .map(|block| plan.indirect_block_size(block.nrows).unwrap())
                    .sum::<u64>(),
            plan.region_size()
        );
        let used: u64 = sizes.iter().sum();
        let mut cursor = 0;
        let mut capacity = 0;
        while cursor < plan.managed_space() {
            let (offset, size) = plan.locate(cursor).unwrap();
            assert_eq!(offset, cursor);
            capacity += size - plan.direct_header_size;
            cursor += size;
        }
        assert_eq!(cursor, plan.managed_space());
        assert_eq!(plan.free_space(), capacity - used);
        if plan.root_rows() == 0 {
            assert_eq!(plan.free_space(), start - plan.direct_header_size - used);
            assert_eq!(plan.allocation_iterator(), 0);
        } else {
            let last = plan.directs.last().unwrap();
            assert_eq!(plan.allocation_iterator(), last.heap_offset + last.size);
        }
    }

    #[rstest]
    #[case::attribute_direct(false, vec![1002], 1024, 0, 0, 0)]
    #[case::attribute_exact_fill(false, vec![1001, 1], 1024, 0, 0, 0)]
    #[case::attribute_skips_starting_blocks(false, vec![1003], 16_384, 3, 10_240, 15_117)]
    #[case::attribute_indirect(false, vec![1002, 1], 4096, 1, 2048, 3005)]
    #[case::group_direct(true, vec![491], 512, 0, 0, 0)]
    #[case::group_indirect(true, vec![491, 1], 2048, 1, 1024, 1472)]
    fn header_statistics_match_known_root_shapes(
        #[case] group: bool,
        #[case] sizes: Vec<u64>,
        #[case] managed: u64,
        #[case] rows: u16,
        #[case] iterator: u64,
        #[case] free: u64,
    ) {
        let plan = if group {
            ManagedHeapPlan::group_links(&sizes, OffsetWidth::Eight)
        } else {
            ManagedHeapPlan::attributes(&sizes, OffsetWidth::Eight)
        }
        .unwrap();
        assert_eq!(plan.managed_space(), managed);
        assert_eq!(plan.root_rows(), rows);
        assert_eq!(plan.allocation_iterator(), iterator);
        assert_eq!(plan.free_space(), free);
    }

    #[rstest]
    #[case::zero(0, 65_514)]
    #[case::past_largest_block(65_515, 65_514)]
    #[case::unrepresentable_size(u64::MAX, 65_514)]
    fn invalid_object_sizes_are_rejected(#[case] size: u64, #[case] maximum: u64) {
        let error = ManagedHeapPlan::attributes(&[size], OffsetWidth::Eight)
            .err()
            .unwrap();
        assert_eq!(error, ManagedHeapPlanError::ObjectSize { size, maximum });
    }

    #[rstest]
    #[case::zero(0)]
    #[case::above_limit(GROUP_HEAP_MAX_MANAGED_OBJECT + 1)]
    fn invalid_group_object_sizes_report_the_canonical_managed_limit(#[case] size: u64) {
        let error = ManagedHeapPlan::group_links(&[size], OffsetWidth::Eight)
            .err()
            .unwrap();
        assert_eq!(
            error,
            ManagedHeapPlanError::ObjectSize {
                size,
                maximum: GROUP_HEAP_MAX_MANAGED_OBJECT,
            }
        );
    }

    #[test]
    fn blocks_too_small_for_the_metadata_prefix_are_rejected() {
        let geometry = FractalHeapGeometry::new(4, 16, 32, 16).unwrap();
        assert_eq!(
            ManagedHeapPlan::new(&[1], OffsetWidth::Eight, geometry)
                .err()
                .unwrap(),
            ManagedHeapPlanError::Geometry
        );
    }

    #[rstest]
    #[case::at_boundary(2, true)]
    #[case::past_boundary(3, false)]
    #[case::further_past_boundary(5, false)]
    fn heap_allocation_accepts_the_last_block_and_rejects_further_objects(
        #[case] count: usize,
        #[case] succeeds: bool,
    ) {
        let geometry = FractalHeapGeometry::new(2, 64, 128, 9).unwrap();
        let result = ManagedHeapPlan::new(&vec![100; count], OffsetWidth::Eight, geometry);
        if succeeds {
            let plan = result.unwrap();
            assert_eq!(plan.managed_space(), 512);
            assert_eq!(plan.allocation_iterator(), 512);
        } else {
            assert_eq!(result.err().unwrap(), ManagedHeapPlanError::HeapSpace);
        }
    }

    #[rstest]
    #[case::too_wide_owner(OffsetWidth::Two, 0, 0x1_0000)]
    #[case::undefined_owner(OffsetWidth::Four, 0, 0xffff_ffff)]
    #[case::region_extent(OffsetWidth::Two, 0xff00, 0)]
    #[case::overflowing_extent(OffsetWidth::Eight, u64::MAX - 10, 0)]
    fn invalid_relocation_addresses_are_rejected(
        #[case] width: OffsetWidth,
        #[case] base: u64,
        #[case] owner: u64,
    ) {
        let plan = ManagedHeapPlan::attributes(&[1], width).unwrap();
        let error = plan
            .serialize(&[&[0]], StoredAddress::new(base), StoredAddress::new(owner))
            .unwrap_err();
        match (width, base, owner) {
            (OffsetWidth::Two, 0, _) => assert!(
                matches!(
                    error,
                    FormatError::AddressTooLarge {
                        address: 0x1_0000,
                        offset_size: 2
                    }
                ),
                "{error:?}"
            ),
            (OffsetWidth::Four, _, _) => assert!(
                matches!(
                    error,
                    FormatError::DefinedAddressIsUndefined {
                        address: 0xffff_ffff,
                        offset_size: 4
                    }
                ),
                "{error:?}"
            ),
            (OffsetWidth::Two, _, _) => assert!(
                matches!(
                    error,
                    FormatError::AddressTooLarge {
                        address: 66_304,
                        offset_size: 2
                    }
                ),
                "{error:?}"
            ),
            _ => assert!(
                matches!(error, FormatError::OffsetOverflow { offset, length: 1024 } if offset == u64::MAX - 10),
                "{error:?}"
            ),
        }
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    #[case::eight(OffsetWidth::Eight)]
    fn undefined_roots_are_rejected_at_the_encoding_width(#[case] width: OffsetWidth) {
        let plan = ManagedHeapPlan::attributes(&[1], width).unwrap();
        let address = StoredAddress::undefined(width);
        let error = plan.root_address(address).unwrap_err();
        assert!(
            matches!(error, FormatError::DefinedAddressIsUndefined { address: value, offset_size } if value == address.get() && offset_size == width.get()),
            "{error:?}"
        );
    }

    #[test]
    fn region_lengths_are_checked_before_host_materialization() {
        let geometry = FractalHeapGeometry::new(2, 1u64 << 32, 1u64 << 32, 35).unwrap();
        let result = ManagedHeapPlan::new(&[], OffsetWidth::Eight, geometry);
        if usize::BITS == 32 {
            assert_eq!(
                result.err().unwrap(),
                ManagedHeapPlanError::Host { bytes: 1u64 << 32 }
            );
        } else {
            assert_eq!(result.unwrap().region_size(), 1u64 << 32);
        }
    }

    #[test]
    fn serializing_objects_of_other_sizes_is_rejected() {
        let plan = ManagedHeapPlan::attributes(&[5], OffsetWidth::Eight).unwrap();
        let error = plan
            .serialize(&[b"hello!"], StoredAddress::new(0), StoredAddress::new(0))
            .unwrap_err();
        assert!(
            matches!(
                error,
                FormatError::DataSizeMismatch {
                    expected: 5,
                    actual: 6
                }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn serializing_another_object_count_reports_an_internal_error() {
        let plan = ManagedHeapPlan::attributes(&[5], OffsetWidth::Eight).unwrap();
        let error = plan
            .serialize(&[], StoredAddress::new(0), StoredAddress::new(0))
            .unwrap_err();
        assert!(
            matches!(error, FormatError::Internal(ref detail) if detail == "managed heap emission received another object count"),
            "{error:?}"
        );
    }

    #[test]
    fn block_addresses_above_32_bits_use_local_buffers() {
        let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Eight).unwrap();
        let base = StoredAddress::new(1u64 << 40);
        let owner = StoredAddress::new(1u64 << 39);
        let data = plan.serialize(&[&[37]], base, owner).unwrap();
        let header = header(&plan, base);
        let layout =
            FractalHeapBlockLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight, owner)
                .unwrap();
        let FractalHeapBlock::Direct(block) = layout.root().unwrap().unwrap() else {
            panic!("expected a direct root")
        };
        let reference = block.at(PhysicalAddress::new(base.get())).unwrap();
        let image = FileRegion::new(reference.range(), &data).unwrap();
        reference.parse_checked(image).unwrap();
        let object = reference
            .parse_object(image, plan.heap_offset(0), 1)
            .unwrap();
        assert_eq!(object.parse(image).unwrap(), &[37]);
    }

    #[test]
    fn direct_blocks_include_a_checksum_over_the_zeroed_checksum_field() {
        let plan = ManagedHeapPlan::group_links(&[1], OffsetWidth::Eight).unwrap();
        let base = StoredAddress::new(512);
        let owner = StoredAddress::new(128);
        let mut data = plan.serialize(&[&[37]], base, owner).unwrap();
        let header = header(&plan, base);
        let layout =
            FractalHeapBlockLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight, owner)
                .unwrap();
        let FractalHeapBlock::Direct(block) = layout.root().unwrap().unwrap() else {
            panic!("expected a direct root")
        };
        let reference = block.at(PhysicalAddress::new(base.get())).unwrap();
        let checksum_at = usize::try_from(plan.direct_header_size).unwrap() - CHECKSUM_LEN;
        let stored = u32::from_le_bytes(
            data[checksum_at..checksum_at + CHECKSUM_LEN]
                .try_into()
                .unwrap(),
        );
        let last = data.last_mut().unwrap();
        *last ^= 1;
        let image = FileRegion::new(reference.range(), &data).unwrap();
        let error = reference.parse_checked(image).unwrap_err();
        data[checksum_at..checksum_at + CHECKSUM_LEN].fill(0);
        let computed = crate::checksum::jenkins_lookup3(&data);
        assert!(
            matches!(error, FormatError::ChecksumMismatch { expected, computed: actual } if expected == stored && actual == computed),
            "{error:?}"
        );
        assert_ne!(stored, computed);
    }

    #[test]
    fn root_address_addition_is_checked_for_nested_indirect_blocks() {
        let geometry = FractalHeapGeometry::new(4, 64, 128, 16).unwrap();
        let plan = ManagedHeapPlan::new(&[100; 100], OffsetWidth::Eight, geometry).unwrap();
        let root_offset = plan.indirects.last().unwrap().region_offset;
        assert!(root_offset > 1);
        let error = plan
            .root_address(StoredAddress::new(u64::MAX - 1))
            .unwrap_err();
        assert!(
            matches!(error, FormatError::OffsetOverflow { offset, length } if offset == u64::MAX - 1 && length == root_offset),
            "{error:?}"
        );
    }
}
