//! Version 0 Fixed Array headers and empty data blocks for independent byte fixtures.
//!
//! The structures are defined in "The Fixed Array Index" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr

use core::mem;

use crate::bytes;
use crate::checksum;
use crate::widths::Widths;

/// The entry encoding written into a Fixed Array fixture.
#[derive(Clone, Copy, Debug)]
pub enum Client {
    /// An offset-width chunk address with no size or mask.
    Plain,
    /// A chunk address, a variable-width size, and a four-byte filter mask.
    Filtered {
        /// The size field width in bytes, independently of the file's length width.
        size_bytes: u8,
    },
}

impl Client {
    /// Returns the entry length in bytes for the supplied offset width.
    pub fn stride(self, widths: Widths) -> usize {
        match self {
            Self::Plain => widths.offset,
            Self::Filtered { size_bytes } => {
                widths.offset + usize::from(size_bytes) + mem::size_of::<u32>()
            }
        }
    }

    fn id(self) -> u8 {
        match self {
            Self::Plain => PLAIN_CLIENT,
            Self::Filtered { .. } => FILTERED_CLIENT,
        }
    }
}

/// Builds a checksummed header for `entries` slots at the supplied widths.
///
/// An absent `data_block` is the undefined address. Writes the supplied page bits and entry
/// encoding without semantic checks, including malformed geometry.
///
/// # Panics
///
/// Panics if an offset or length width is outside 1 through 8 bytes, a value exceeds its field
/// width, or the client stride exceeds one byte.
pub fn header(
    widths: Widths,
    client: Client,
    page_bits: u8,
    entries: u64,
    data_block: Option<u64>,
) -> Vec<u8> {
    let mut output = HEADER_SIGNATURE.to_vec();
    output.extend_from_slice(&[
        VERSION,
        client.id(),
        u8::try_from(client.stride(widths)).unwrap(),
        page_bits,
    ]);
    bytes::push_uint(&mut output, entries, widths.length);
    bytes::push_address(&mut output, data_block, widths.offset);
    checksum::append(&mut output);
    output
}

/// Builds an empty data block whose header is at the stored address `owner`.
///
/// Unpaged entries store undefined addresses and zero filtered fields. Paged blocks have a zero
/// initialization bitmap and zero bytes reserved for their pages. Computes the checksum of the
/// whole unpaged block, or of the paged prefix and bitmap.
///
/// # Panics
///
/// Panics if an offset width is outside 1 through 8 bytes, `owner` exceeds that width, `page_bits`
/// exceeds the platform's shift range, or the complete output exceeds the platform's capacity.
pub fn empty_data_block(
    widths: Widths,
    client: Client,
    page_bits: u8,
    entries: usize,
    owner: u64,
) -> Vec<u8> {
    let mut output = DATA_BLOCK_SIGNATURE.to_vec();
    output.extend_from_slice(&[VERSION, client.id()]);
    bytes::push_uint(&mut output, owner, widths.offset);
    let page_entries = 1_usize.checked_shl(u32::from(page_bits)).unwrap();
    if entries <= page_entries {
        for _ in 0..entries {
            bytes::push_undefined_address(&mut output, widths.offset);
            if let Client::Filtered { size_bytes } = client {
                output.resize(
                    output.len() + usize::from(size_bytes) + mem::size_of::<u32>(),
                    0,
                );
            }
        }
        checksum::append(&mut output);
    } else {
        let pages = entries.div_ceil(page_entries);
        output.resize(
            output.len() + pages.div_ceil(usize::try_from(u8::BITS).unwrap()),
            0,
        );
        checksum::append(&mut output);
        output.resize(
            output.len() + entries * client.stride(widths) + pages * mem::size_of::<u32>(),
            0,
        );
    }
    output
}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub address: u64,
    pub stored_size: u64,
    pub filter_mask: u32,
}

pub fn page(widths: Widths, client: Client, records: &[Option<Entry>]) -> Vec<u8> {
    let mut output = Vec::new();
    for record in records {
        bytes::push_address(
            &mut output,
            record.map(|record| record.address),
            widths.offset,
        );
        if let Client::Filtered { size_bytes } = client {
            bytes::push_uint(
                &mut output,
                record.map_or(0, |record| record.stored_size),
                usize::from(size_bytes),
            );
            output.extend_from_slice(&record.map_or(0, |record| record.filter_mask).to_le_bytes());
        }
    }
    checksum::append(&mut output);
    output
}

/// The signature of a version 0 Fixed Array header.
pub const HEADER_SIGNATURE: &[u8; 4] = b"FAHD";
/// The signature of a version 0 Fixed Array data block.
pub const DATA_BLOCK_SIGNATURE: &[u8; 4] = b"FADB";
/// The Entry Size byte's offset in a version 0 header.
pub const HEADER_ENTRY_SIZE_OFFSET: usize = 6;
/// The data-block prefix length before its offset-width header address.
pub const DATA_BLOCK_FIXED_PREFIX_LEN: u64 = 6;
/// The second page's initialization bit in the first bitmap byte.
pub const SECOND_PAGE_MASK: u8 = 0x40;
/// The third page's initialization bit in the first bitmap byte.
pub const THIRD_PAGE_MASK: u8 = 0x20;
const VERSION: u8 = 0;
const PLAIN_CLIENT: u8 = 0;
const FILTERED_CLIENT: u8 = 1;
