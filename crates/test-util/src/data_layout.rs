//! Data layout message bodies, with 8-byte addresses and lengths.
//!
//! The message is defined in "The Data Layout Message" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout

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

/// Returns the version and layout class of a version 4 virtual layout message, without the
/// global heap address and index that follow them.
pub fn virtual_storage() -> Vec<u8> {
    vec![VERSION_FOUR, VIRTUAL_CLASS]
}

const VERSION_THREE: u8 = 3;
const VERSION_FOUR: u8 = 4;

const COMPACT_CLASS: u8 = 0;
const CONTIGUOUS_CLASS: u8 = 1;
const CHUNKED_CLASS: u8 = 2;
const VIRTUAL_CLASS: u8 = 3;
