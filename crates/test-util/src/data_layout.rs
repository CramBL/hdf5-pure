//! Data layout message bodies for independent byte fixtures.
//!
//! Version 3 builders and [`implicit`] use eight-byte address and length fields.
//! [`single_chunk`] and [`array_chunk`] use the caller's [`Widths`].
//!
//! The message is defined in "The Data Layout Message" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout

use crate::bytes;
use crate::widths::Widths;

/// The array's index-specific bytes in a version 4 layout message.
///
/// These values are written verbatim, including parameters for malformed-input fixtures.
#[expect(
    variant_size_differences,
    reason = "the variants retain the one-byte Fixed Array or five-byte Extensible Array information"
)]
#[derive(Clone, Copy, Debug)]
pub enum ArrayIndex {
    /// A Fixed Array with the supplied Page Bits byte.
    Fixed {
        /// The base 2 logarithm of the maximum page entry count.
        page_bits: u8,
    },
    /// An Extensible Array with the five creation parameter bytes in wire order.
    Extensible {
        /// Max Bits, Index Elements, Min Pointers, Min Elements, and Page Bits.
        parameters: [u8; 5],
    },
}

/// The size and filter mask written before a Single Chunk address.
#[derive(Clone, Copy, Debug)]
pub struct FilteredChunk {
    /// The filtered size, encoded at the file's length width.
    pub size: u64,
    /// The skipped filters, encoded as a four-byte mask.
    pub mask: u32,
}

/// Builds a version 4 Single Chunk layout for four elements of four bytes each.
///
/// Writes filtered information only for `Some(filtered)`. An absent address is the undefined
/// marker at `widths.offset` bytes. Parameters are written without semantic checks.
///
/// # Panics
///
/// Panics if an encoded width is outside 1 through 8 bytes or a value exceeds its field width.
pub fn single_chunk(
    widths: Widths,
    filtered: Option<FilteredChunk>,
    address: Option<u64>,
) -> Vec<u8> {
    single_chunk_with_geometry(widths, filtered, address, 4, 4)
}

/// Builds a version 4 Single Chunk layout with explicit one-dimensional chunk geometry.
///
/// Writes `chunk_elements` and `element_size` in four-byte fields without semantic checks.
/// Filter information and the address follow [`single_chunk`]'s encoding.
///
/// # Panics
///
/// Panics if an encoded width is outside 1 through 8 bytes or a value exceeds its field width.
pub fn single_chunk_with_geometry(
    widths: Widths,
    filtered: Option<FilteredChunk>,
    address: Option<u64>,
    chunk_elements: u32,
    element_size: u32,
) -> Vec<u8> {
    let mut layout = chunked_prefix_with_geometry(
        if filtered.is_some() {
            SINGLE_INDEX_WITH_FILTER
        } else {
            NO_CHUNKED_FLAGS
        },
        SINGLE_CHUNK_INDEX,
        chunk_elements,
        element_size,
    );
    if let Some(FilteredChunk { size, mask }) = filtered {
        bytes::push_uint(&mut layout, size, widths.length);
        layout.extend_from_slice(&mask.to_le_bytes());
    }
    bytes::push_address(&mut layout, address, widths.offset);
    layout
}

/// Builds a version 4 array-index layout for four elements of four bytes each.
///
/// Writes `index` parameters verbatim and `None` as the undefined header address.
///
/// # Panics
///
/// Panics if `widths.offset` is outside 1 through 8 bytes or `address` exceeds its field width.
pub fn array_chunk(widths: Widths, index: ArrayIndex, address: Option<u64>) -> Vec<u8> {
    array_chunk_with_geometry(widths, index, address, 4, 4)
}

/// Builds a version 4 array-index layout with explicit one-dimensional chunk geometry.
///
/// Writes four-byte dimension fields and raw index parameters as in [`array_chunk`].
///
/// # Panics
///
/// Panics if an offset width is outside 1 through 8 bytes or an address exceeds its field width.
pub fn array_chunk_with_geometry(
    widths: Widths,
    index: ArrayIndex,
    address: Option<u64>,
    chunk_elements: u32,
    element_size: u32,
) -> Vec<u8> {
    let mut layout = match index {
        ArrayIndex::Fixed { page_bits } => {
            let mut layout = chunked_prefix_with_geometry(
                NO_CHUNKED_FLAGS,
                FIXED_ARRAY_INDEX,
                chunk_elements,
                element_size,
            );
            layout.push(page_bits);
            layout
        }
        ArrayIndex::Extensible { parameters } => {
            let mut layout = chunked_prefix_with_geometry(
                NO_CHUNKED_FLAGS,
                EXTENSIBLE_ARRAY_INDEX,
                chunk_elements,
                element_size,
            );
            layout.extend_from_slice(&parameters);
            layout
        }
    };
    bytes::push_address(&mut layout, address, widths.offset);
    layout
}

fn chunked_prefix_with_geometry(
    flags: u8,
    index: u8,
    chunk_elements: u32,
    element_size: u32,
) -> Vec<u8> {
    let mut layout = vec![
        VERSION_FOUR,
        CHUNKED_CLASS,
        flags,
        ONE_DIMENSION_FIELD_COUNT,
        DIMENSION_SIZE_LEN,
    ];
    layout.extend_from_slice(&chunk_elements.to_le_bytes());
    layout.extend_from_slice(&element_size.to_le_bytes());
    layout.push(index);
    layout
}

/// The index-specific field offset in the one-dimensional version 4 fixtures.
///
/// The prefix, dimension array, and index byte are defined in "The Data Layout Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub const ONE_DIMENSION_INDEX_INFO_OFFSET: usize = 14;
const ONE_DIMENSION_FIELD_COUNT: u8 = 2;
// "The Data Layout Message", format specification version 4.0, defines the version 4 Single
// Chunk filter flag and array index type bytes.
const SINGLE_INDEX_WITH_FILTER: u8 = 0x02;
const FIXED_ARRAY_INDEX: u8 = 3;
const EXTENSIBLE_ARRAY_INDEX: u8 = 4;

/// Returns a version 3 compact layout message that holds `data`.
pub fn compact(data: &[u8]) -> Vec<u8> {
    let mut layout = vec![VERSION_THREE, COMPACT_CLASS];
    layout.extend_from_slice(&(data.len() as u16).to_le_bytes());
    layout.extend_from_slice(data);
    layout
}

/// Returns a version 3 contiguous layout message for `size` bytes of raw data at `address`.
///
/// An `address` of [`u64::MAX`] is the undefined address, for storage not yet allocated.
pub fn contiguous(address: u64, size: u64) -> Vec<u8> {
    let mut layout = vec![VERSION_THREE, CONTIGUOUS_CLASS];
    layout.extend_from_slice(&address.to_le_bytes());
    layout.extend_from_slice(&size.to_le_bytes());
    layout
}

/// Returns a version 3 chunked layout message for chunks of `chunk_dims` elements of
/// `element_size` bytes, indexed by the version 1 B-tree at `index_address`.
///
/// An `index_address` of [`u64::MAX`] is the undefined address, for storage not yet allocated.
pub fn chunked(index_address: u64, chunk_dims: &[u32], element_size: u32) -> Vec<u8> {
    let mut layout = vec![VERSION_THREE, CHUNKED_CLASS, (chunk_dims.len() + 1) as u8];
    layout.extend_from_slice(&index_address.to_le_bytes());
    for dim in chunk_dims.iter().chain([&element_size]) {
        layout.extend_from_slice(&dim.to_le_bytes());
    }
    layout
}

/// Returns a version 4 chunked layout message for chunks of `chunk_dims` elements of
/// `element_size` bytes, indexed by an Implicit index whose array of chunks starts at
/// `data_address`.
///
/// The index is defined in "The Implicit Index" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_implicit
pub fn implicit(data_address: u64, chunk_dims: &[u32], element_size: u32) -> Vec<u8> {
    let mut layout = vec![
        VERSION_FOUR,
        CHUNKED_CLASS,
        NO_CHUNKED_FLAGS,
        (chunk_dims.len() + 1) as u8,
        DIMENSION_SIZE_LEN,
    ];
    for dim in chunk_dims.iter().chain([&element_size]) {
        layout.extend_from_slice(&dim.to_le_bytes());
    }
    layout.push(IMPLICIT_INDEX);
    layout.extend_from_slice(&data_address.to_le_bytes());
    layout
}

/// Returns the version and layout class of a version 4 virtual layout message, without the
/// global heap address and index that follow them.
pub fn virtual_storage() -> Vec<u8> {
    vec![VERSION_FOUR, VIRTUAL_CLASS]
}

/// The undefined address at the 8-byte width, with every bit set.
///
/// The undefined address is defined in "Appendix A: Definitions" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_appendixa
pub const UNDEFINED_ADDRESS: u64 = u64::MAX;

const VERSION_THREE: u8 = 3;

/// Version 4 of a data layout message, defined in "The Data Layout Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub const VERSION_FOUR: u8 = 4;

const COMPACT_CLASS: u8 = 0;
/// Contiguous storage in the "Layout Class" field, defined in "The Data Layout Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub const CONTIGUOUS_CLASS: u8 = 1;

/// Chunked storage in the "Layout Class" field, defined in "The Data Layout Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub const CHUNKED_CLASS: u8 = 2;
const VIRTUAL_CLASS: u8 = 3;

/// The "Flags" byte of a chunked layout with neither `DONT_FILTER_PARTIAL_BOUND_CHUNKS` nor
/// `SINGLE_INDEX_WITH_FILTER` set.
///
/// Defined in "The Data Layout Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub const NO_CHUNKED_FLAGS: u8 = 0;

/// The four-byte width this writer selects for each chunk dimension, encoded in the
/// "Dimension Size Encoded Length" byte.
///
/// Defined in "The Data Layout Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
const DIMENSION_SIZE_LEN: u8 = 4;

/// The "Chunk Indexing Type" byte of a Single Chunk index.
///
/// Defined in "The Data Layout Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub const SINGLE_CHUNK_INDEX: u8 = 1;

/// The "Chunk Indexing Type" byte of an Implicit index.
///
/// Defined in "The Data Layout Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
const IMPLICIT_INDEX: u8 = 2;
