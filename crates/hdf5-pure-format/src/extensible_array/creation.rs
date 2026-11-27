use core::num::NonZeroU8;

use super::geometry::Layout;
use crate::bytes;
use crate::chunk_record::ElementEncoding;
use crate::error::FormatError;

/// Checked block and page geometry from Extensible Array indexing information.
///
/// The five creation fields are stored in a version 4 data layout message before the header
/// address, including when that address is undefined. The fields are defined in "The Data Layout
/// Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayCreationParameters {
    bits: ExtensibleArrayBitGeometry,
    inline_elements: u8,
    min_data_blocks: NonZeroU8,
    min_elements: NonZeroU8,
}

impl ExtensibleArrayCreationParameters {
    /// Parses the first five bytes in data layout message order.
    ///
    /// The fields are the maximum element bit count, inline element count, minimum data block
    /// pointer count, minimum data block element count, and page exponent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` has fewer than five bytes, or
    /// [`FormatError::InvalidExtensibleArrayGeometry`] if the fields are outside the supported
    /// block and page geometry.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let fields = bytes::ByteCursor::new(data).array()?;
        Self::from_fields(fields).ok_or(FormatError::InvalidExtensibleArrayGeometry)
    }

    /// Returns the writer's default creation parameters.
    ///
    /// The maximum element bit count is 32, the index block holds four elements, the minimum
    /// data block pointer count is four, the minimum data block size is 16 elements, and a page
    /// holds 1024 elements.
    pub fn canonical() -> Self {
        const {
            // Constant evaluation checks these creation fields before the writer uses them.
            Self::from_fields([
                EA_MAX_NELMTS_BITS,
                EA_IDX_BLK_ELMTS,
                EA_SUPER_BLK_MIN_DATA_PTRS,
                EA_MIN_DBLK_NELMTS,
                EA_MAX_DBLK_NELMTS_BITS,
            ])
            .expect("the writer's creation fields have valid geometry")
        }
    }

    /// Returns the creation fields in data layout message order.
    pub fn encode(self) -> [u8; 5] {
        [
            self.bits.max_index_bits().get(),
            self.inline_elements,
            self.min_data_blocks.get(),
            self.min_elements.get(),
            self.bits.page_nelmts().exponent(),
        ]
    }

    /// Returns the maximum element bit count and data block page geometry.
    pub const fn bit_geometry(self) -> ExtensibleArrayBitGeometry {
        self.bits
    }

    /// Returns the number of elements stored directly in the index block.
    pub fn inline_elements(self) -> u8 {
        self.inline_elements
    }

    /// Returns the minimum number of data block pointers in a super block.
    pub fn min_data_blocks(self) -> NonZeroU8 {
        self.min_data_blocks
    }

    /// Returns the minimum number of elements in a data block.
    pub fn min_elements(self) -> NonZeroU8 {
        self.min_elements
    }

    /// Constructs creation parameters from the five fields in data layout message order.
    ///
    /// Returns `None` if the fields describe an unsupported block or page geometry.
    pub(super) const fn from_fields(fields: [u8; 5]) -> Option<Self> {
        let [
            max_index_bits,
            inline_elements,
            min_data_blocks,
            min_elements,
            page_bits,
        ] = fields;
        let Ok(bits) = ExtensibleArrayBitGeometry::from_fields(max_index_bits, page_bits) else {
            return None;
        };
        Self::from_bit_geometry(bits, inline_elements, min_data_blocks, min_elements)
    }

    /// Combines checked bit geometry with the counts for index and data blocks.
    ///
    /// Returns `None` if the minimum block element or pointer count is unsupported, or if the
    /// block geometry exceeds the array or page bounds.
    const fn from_bit_geometry(
        bits: ExtensibleArrayBitGeometry,
        inline_elements: u8,
        min_data_blocks: u8,
        min_elements: u8,
    ) -> Option<Self> {
        let (Some(min_data_blocks), Some(min_elements)) = (
            NonZeroU8::new(min_data_blocks),
            NonZeroU8::new(min_elements),
        ) else {
            return None;
        };
        if !min_elements.get().is_power_of_two()
            || !min_data_blocks.get().is_power_of_two()
            || min_data_blocks.get() < MIN_DATA_BLOCKS
        {
            return None;
        }
        let min_elements_bits = min_elements.get().trailing_zeros();
        let first_super_block = 2 * min_data_blocks.get().trailing_zeros();
        let Some(super_blocks) =
            (bits.max_index_bits().get() as u32 + 1).checked_sub(min_elements_bits)
        else {
            return None;
        };
        if first_super_block > super_blocks
            || min_elements_bits + first_super_block / 2 > bits.page_nelmts().exponent() as u32
        {
            return None;
        }
        Some(Self {
            bits,
            inline_elements,
            min_data_blocks,
            min_elements,
        })
    }

    pub(super) fn layout(self, encoding: ElementEncoding) -> Layout {
        Layout::new(self, encoding)
    }
}

/// The parsed maximum element bit count and data block page exponent.
///
/// The maximum element bit count is in `1..=64`, and the page exponent is in `0..=63` and at
/// most that count. The page element count is a nonzero power of two representable by a [`u64`].
/// The block-offset width is the maximum element bit count rounded up to whole bytes.
/// libhdf5 computes the block-offset width with `H5EA_SIZEOF_OFFSET_BITS` (`H5EApkg.h`,
/// HDF5 2.2.0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayBitGeometry {
    max_index_bits: MaxIndexBits,
    page_nelmts: PageElementCount,
}

impl ExtensibleArrayBitGeometry {
    /// Returns the "Max Nelmts Bits" field, in `1..=64`.
    pub const fn max_index_bits(self) -> MaxIndexBits {
        self.max_index_bits
    }

    /// Returns the number of elements in a data block page, a nonzero power of two.
    pub const fn page_nelmts(self) -> PageElementCount {
        self.page_nelmts
    }

    /// Returns the maximum element bit count rounded up to a block-offset byte width.
    pub fn block_offset_width(self) -> BlockOffsetWidth {
        match self.max_index_bits.get() {
            1..=8 => BlockOffsetWidth::One,
            9..=16 => BlockOffsetWidth::Two,
            17..=24 => BlockOffsetWidth::Three,
            25..=32 => BlockOffsetWidth::Four,
            33..=40 => BlockOffsetWidth::Five,
            41..=48 => BlockOffsetWidth::Six,
            49..=56 => BlockOffsetWidth::Seven,
            _ => BlockOffsetWidth::Eight,
        }
    }

    /// Constructs checked bit geometry from the maximum element bit count and page exponent.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidBitGeometry::MaximumIndexBits`] if the maximum bit count is outside
    /// `1..=64`, [`InvalidBitGeometry::PageElementCount`] if the page element count does not fit in
    /// [`u64`], or [`InvalidBitGeometry::PageExponentExceedsIndexBits`] if the page exponent
    /// exceeds the maximum bit count.
    const fn from_fields(max_index_bits: u8, page_bits: u8) -> Result<Self, InvalidBitGeometry> {
        let max_index_bits = match NonZeroU8::new(max_index_bits) {
            Some(bits) if bits.get() <= MAX_INDEX_BITS => MaxIndexBits(bits),
            _ => return Err(InvalidBitGeometry::MaximumIndexBits),
        };
        if page_bits > MAX_PAGE_BITS {
            return Err(InvalidBitGeometry::PageElementCount);
        }
        if page_bits > max_index_bits.get() {
            return Err(InvalidBitGeometry::PageExponentExceedsIndexBits);
        }
        Ok(Self {
            max_index_bits,
            page_nelmts: PageElementCount(page_bits),
        })
    }
}

#[derive(Debug, Eq, PartialEq)]
enum InvalidBitGeometry {
    MaximumIndexBits,
    PageElementCount,
    PageExponentExceedsIndexBits,
}

/// An Extensible Array header's "Max Nelmts Bits" field, in `1..=64`.
///
/// The field is defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct MaxIndexBits(NonZeroU8);

impl MaxIndexBits {
    /// Returns the bit count as a [`u8`], in `1..=64`.
    pub const fn get(self) -> u8 {
        self.0.get()
    }
}

/// A data block page's element count, a nonzero power of two.
///
/// Stores the base 2 exponent in `0..=63`. The version 0 Extensible Array header's
/// `Max Dblk Page Nelmts Bits` field is defined in "The Extensible Array Index" of the
/// [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
///
/// let geometry = ExtensibleArrayCreationParameters::canonical().bit_geometry();
/// let count = geometry.page_nelmts();
/// assert_eq!(count.get(), 1024);
/// assert_eq!(count.exponent(), 10);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct PageElementCount(u8);

impl PageElementCount {
    /// Returns the number of elements in a page, a nonzero power of two.
    pub const fn get(self) -> u64 {
        1 << self.0
    }

    /// Returns the base 2 exponent stored in the page size field.
    pub const fn exponent(self) -> u8 {
        self.0
    }
}

/// The width in bytes of an Extensible Array block offset, in `1..=8`.
///
/// The block offset is defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum BlockOffsetWidth {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
}

impl BlockOffsetWidth {
    /// Returns the number of bytes in the field, in `1..=8`.
    pub fn bytes(self) -> usize {
        self as usize
    }

    /// Reads a little-endian block offset from the start of `data`.
    ///
    /// Reads [`bytes`](Self::bytes) bytes and ignores any remaining bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the field width.
    pub fn read(self, data: &[u8]) -> Result<u64, FormatError> {
        let size = self.bytes();
        let field = data.get(..size).ok_or(FormatError::UnexpectedEof {
            expected: size,
            available: data.len(),
        })?;
        let mut bytes = [0u8; 8];
        bytes[..size].copy_from_slice(field);
        Ok(u64::from_le_bytes(bytes))
    }
}

// The creation parameters the writer stores and `extensible_array_capacity` derives the capacity
// from, the C library's defaults (`H5Dpkg.h`, HDF5 2.2.0).

/// The number of bits that hold the maximum number of elements, `H5D_EARRAY_MAX_NELMTS_BITS`.
pub(crate) const EA_MAX_NELMTS_BITS: u8 = 32;
/// The number of elements in the index block, `H5D_EARRAY_IDX_BLK_ELMTS`.
pub(crate) const EA_IDX_BLK_ELMTS: u8 = 4;
/// The number of elements in the smallest data block, `H5D_EARRAY_DATA_BLK_MIN_ELMTS`.
pub(crate) const EA_MIN_DBLK_NELMTS: u8 = 16;
/// The fewest data block addresses a super block holds, `H5D_EARRAY_SUP_BLK_MIN_DATA_PTRS`.
pub(crate) const EA_SUPER_BLK_MIN_DATA_PTRS: u8 = 4;
/// The base 2 logarithm of the number of elements in a data block page,
/// `H5D_EARRAY_MAX_DBLOCK_PAGE_NELMTS_BITS`.
pub(crate) const EA_MAX_DBLK_NELMTS_BITS: u8 = 10;

// The bounds `H5EA__hdr_create` checks the creation parameters against (`H5EAhdr.c`, HDF5 2.2.0).
const MAX_INDEX_BITS: u8 = 64;
const MIN_DATA_BLOCKS: u8 = 2;
// The largest page exponent whose page element count fits a `u64`.
const MAX_PAGE_BITS: u8 = 63;

const _: () = {
    assert!(MAX_INDEX_BITS as u32 == u64::BITS);
    assert!(MAX_PAGE_BITS as u32 + 1 == u64::BITS);
};

#[cfg(test)]
#[path = "tests/creation.rs"]
mod tests;
