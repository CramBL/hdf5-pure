use alloc::vec;
use alloc::vec::Vec;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapGeometry;
use crate::fractal_heap::block::{CHECKSUM_LEN, DIRECT_SIGNATURE, INDIRECT_SIGNATURE};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

use super::encode::BlockPrefix;
use super::parameters::{
    self, ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE, ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS,
    ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE, ATTRIBUTE_HEAP_TABLE_WIDTH,
    GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE, GROUP_HEAP_MAX_HEAP_SIZE_BITS, GROUP_HEAP_MAX_MANAGED_OBJECT,
    GROUP_HEAP_STARTING_BLOCK_SIZE, GROUP_HEAP_TABLE_WIDTH,
};

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
    size: u64,
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

    /// Lays out objects with the attribute-heap geometry and eight-byte length fields.
    ///
    /// libhdf5 sets this geometry in `H5A__dense_create` in `H5Adense.c` (HDF5 2.2.0).
    /// The object-size limit depends on the address width.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::ObjectSize`] if an object is empty or exceeds
    /// [`attribute_heap_max_managed_object`](parameters::attribute_heap_max_managed_object),
    /// [`ManagedHeapPlanError::HeapSpace`] if placement or a statistic exceeds the heap's
    /// address space or `u64`, and [`ManagedHeapPlanError::Host`]
    /// if the serialized region length exceeds [`usize::MAX`].
    pub fn attributes(
        sizes: &[u64],
        offset_width: OffsetWidth,
    ) -> Result<Self, ManagedHeapPlanError> {
        Self::attributes_with_widths(
            sizes,
            FormatWidths {
                offsets: offset_width,
                lengths: LengthWidth::Eight,
            },
        )
    }

    /// Lays out attribute objects with direct blocks bounded by the selected length width.
    ///
    /// Uses the attribute-heap geometry of [`Self::attributes`], with a smaller direct-block
    /// maximum when necessary to fit `widths.lengths`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagedHeapPlanError::ObjectSize`] if an object is empty or exceeds the managed
    /// payload capacity at `widths`. Returns the heap-space and host-size errors of
    /// [`Self::attributes`].
    pub fn attributes_with_widths(
        sizes: &[u64],
        widths: FormatWidths,
    ) -> Result<Self, ManagedHeapPlanError> {
        let geometry = FractalHeapGeometry::new(
            ATTRIBUTE_HEAP_TABLE_WIDTH,
            ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
            parameters::maximum_direct_block_size(
                ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE,
                widths.lengths,
            ),
            ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS,
        )
        .map_err(|_error| ManagedHeapPlanError::Geometry)?;
        Self::new(sizes, widths.offsets, geometry)
    }

    /// Lays out group-link objects with eight-byte length fields and the managed-object limit.
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
        Self::group_links_with_widths(
            sizes,
            FormatWidths {
                offsets: offset_width,
                lengths: LengthWidth::Eight,
            },
        )
    }

    /// Lays out group links with direct blocks bounded by the selected length width.
    ///
    /// Uses the group-link geometry and object-size limit of [`Self::group_links`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::group_links`] for invalid object sizes, heap-space bounds,
    /// or host-size bounds.
    pub fn group_links_with_widths(
        sizes: &[u64],
        widths: FormatWidths,
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
            parameters::maximum_direct_block_size(GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE, widths.lengths),
            GROUP_HEAP_MAX_HEAP_SIZE_BITS,
        )
        .map_err(|_error| ManagedHeapPlanError::Geometry)?;
        Self::new(sizes, widths.offsets, geometry)
    }

    /// Returns each complete block's serialized length in placement order.
    ///
    /// Indirect blocks precede direct blocks. Pass addresses in this order to [`Self::place`].
    /// Each length includes the metadata prefix, checksum, and any unused direct-block data.
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> + DoubleEndedIterator + '_ {
        (0..self.indirects.len() + self.directs.len()).map(|index| self.block_length(index))
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
        self.address_at(region_address, self.block_offset(self.root_index()))
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

    pub(super) fn check_objects(&self, objects: &[&[u8]]) -> Result<(), FormatError> {
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
        Ok(())
    }

    /// Checks an exclusive block end, which may equal the undefined address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if the end exceeds the address width.
    pub(super) fn require_end(&self, end: u64) -> Result<(), FormatError> {
        if end > StoredAddress::undefined(self.offset_width).get() {
            return Err(FormatError::AddressTooLarge {
                address: end,
                offset_size: self.offset_width.get(),
            });
        }
        Ok(())
    }

    /// Returns the placement index of the last indirect block, or zero for a direct root.
    pub(super) fn root_index(&self) -> usize {
        self.indirects.len().saturating_sub(1)
    }

    pub(super) fn block_length(&self, index: usize) -> u64 {
        if let Some(block) = self.indirects.get(index) {
            block.size
        } else {
            self.directs[index - self.indirects.len()].size
        }
    }

    pub(super) fn block_offset(&self, index: usize) -> u64 {
        if let Some(block) = self.indirects.get(index) {
            block.region_offset
        } else {
            self.directs[index - self.indirects.len()].region_offset
        }
    }

    pub(super) fn encode_block(
        &self,
        index: usize,
        objects: &[&[u8]],
        owner: StoredAddress,
        address: impl Fn(usize) -> Result<StoredAddress, FormatError>,
    ) -> Result<Vec<u8>, FormatError> {
        if let Some(block) = self.indirects.get(index) {
            let prefix = self.block_prefix(owner, block.heap_offset);
            prefix.encode_indirect(
                block.size,
                block.entries.iter().map(|entry| match entry {
                    Some(Child::Direct(index)) => address(self.indirects.len() + index).map(Some),
                    Some(Child::Indirect(index)) => address(*index).map(Some),
                    None => Ok(None),
                }),
            )
        } else {
            let block = &self.directs[index - self.indirects.len()];
            self.block_prefix(owner, block.heap_offset).encode_direct(
                block.size,
                self.direct_header_size,
                |data| {
                    for &index in &block.objects {
                        debug_assert_eq!(
                            block.heap_offset.checked_add(data.len() as u64),
                            Some(self.offsets[index])
                        );
                        data.extend_from_slice(objects[index]);
                    }
                },
            )
        }
    }

    fn block_prefix(&self, owner: StoredAddress, heap_offset: u64) -> BlockPrefix {
        BlockPrefix {
            owner,
            offset_width: self.offset_width,
            heap_offset,
            block_offset_bytes: self.block_offset_bytes,
        }
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
            let size = self.indirects[index].size;
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
            size: self.indirect_block_size(nrows)?,
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
    pub(super) fn address_at(
        &self,
        base: StoredAddress,
        offset: u64,
    ) -> Result<StoredAddress, FormatError> {
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
}

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
