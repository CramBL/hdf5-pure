use crate::fractal_heap::block::{CHECKSUM_LEN, DIRECT_SIGNATURE};
use crate::width::{LengthWidth, OffsetWidth};

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

/// Caps the direct-block maximum at the largest power of two representable by `lengths`.
pub(crate) fn maximum_direct_block_size(maximum: u64, lengths: LengthWidth) -> u64 {
    maximum.min(1u64 << (u32::from(lengths.get()) * u8::BITS - 1))
}
