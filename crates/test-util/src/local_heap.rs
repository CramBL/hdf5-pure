//! Local heaps, which hold the link names of a version 1 group: section
//! `subsec_fmt4_infra_localheap`, version 4.0.

use crate::bytes;
use crate::widths::Widths;

/// The bytes a heap's header occupies, which is where a data segment laid out
/// directly behind it begins.
pub fn header_len(widths: Widths) -> usize {
    PREFIX + widths.length * 2 + widths.offset
}

/// A heap's data segment: the strings it holds, laid out back to back, and
/// where in the file it sits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    pub at: u64,
    pub bytes: Vec<u8>,
    /// The offset of the first free block within the segment.
    pub free_list_head: u64,
    offsets: Vec<u64>,
}

impl Segment {
    /// The segment at `at` holding each of `names`, null-terminated, in order.
    pub fn of_names(at: u64, names: &[&str]) -> Self {
        let mut bytes = Vec::new();
        let offsets = names
            .iter()
            .map(|name| {
                let offset = bytes.len() as u64;
                bytes.extend_from_slice(name.as_bytes());
                bytes.push(0);
                offset
            })
            .collect();
        Self {
            at,
            bytes,
            free_list_head: NO_FREE_LIST,
            offsets,
        }
    }

    /// The bytes of the local heap header that states this segment's address,
    /// its size and its first free block.
    pub fn header(&self, widths: Widths) -> Vec<u8> {
        let mut header = SIGNATURE.to_vec();
        header.push(VERSION);
        header.extend_from_slice(&[0; 3]);
        bytes::push_uint(&mut header, self.bytes.len() as u64, widths.length);
        bytes::push_uint(&mut header, self.free_list_head, widths.length);
        bytes::push_uint(&mut header, self.at, widths.offset);
        header
    }

    /// The offset within the segment of the `index`th name given to
    /// [`Self::of_names`], which is what a symbol-table entry stores.
    #[track_caller]
    pub fn offset_of(&self, index: usize) -> u64 {
        *self.offsets.get(index).unwrap_or_else(|| {
            panic!(
                "the segment holds {} names, not {index}",
                self.offsets.len()
            )
        })
    }
}

pub const SIGNATURE: &[u8; 4] = b"HEAP";

/// signature(4) + version(1) + reserved(3), which the header begins with.
const PREFIX: usize = 8;

/// The free-list head offset of a segment with no free block, which
/// `H5HL__cache_datablock_deserialize` in `H5HLcache.c`, release 1.12.3,
/// rejects as anything but this or an offset inside the segment:
/// `H5HL_FREE_NULL` in `H5HLpkg.h`, same release.
const NO_FREE_LIST: u64 = 1;

const VERSION: u8 = 0;
