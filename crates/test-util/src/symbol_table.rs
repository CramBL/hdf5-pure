//! Symbol table nodes and entries: sections `subsec_fmt4_infra_symboltable`
//! and `subsec_fmt4_infra_symboltableentry`, version 4.0.

use crate::bytes;
use crate::widths::Widths;

/// The bytes of a symbol table node holding `entries`.
pub fn node(entries: &[Entry], widths: Widths) -> Vec<u8> {
    let mut node = SIGNATURE.to_vec();
    node.push(VERSION);
    node.push(0);
    node.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    node.extend(entries.iter().flat_map(|entry| entry.build(widths)));
    node
}

/// One entry: a link name's offset in the group's local heap, the address of
/// the object it refers to, and how the scratch pad that follows is read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Entry {
    pub link_name_offset: u64,
    pub header_address: u64,
    pub cache_type: u32,
    /// The sixteen bytes whose meaning `cache_type` decides: the addresses of
    /// a subgroup's B-tree and local heap for cache type 1, and nothing at all
    /// for cache type 0.
    pub scratch_pad: [u8; SCRATCH_PAD],
}

impl Entry {
    /// An entry that caches nothing, which is what cache type 0 means.
    pub fn new(link_name_offset: u64, header_address: u64) -> Self {
        Self {
            link_name_offset,
            header_address,
            cache_type: 0,
            scratch_pad: [0; SCRATCH_PAD],
        }
    }

    /// Returns the offset of the cache type within an entry with `widths`.
    pub fn cache_type_at(widths: Widths) -> usize {
        widths.length + widths.offset
    }

    /// Returns the offset of the scratch pad within an entry with `widths`.
    pub fn scratch_pad_at(widths: Widths) -> usize {
        Self::cache_type_at(widths) + CACHE_TYPE_LEN + RESERVED_LEN
    }

    /// Returns the length of an entry with `widths`, scratch pad included.
    pub fn encoded_len(widths: Widths) -> usize {
        Self::scratch_pad_at(widths) + SCRATCH_PAD
    }

    pub fn build(&self, widths: Widths) -> Vec<u8> {
        let mut entry = Vec::new();
        bytes::push_uint(&mut entry, self.link_name_offset, widths.length);
        bytes::push_uint(&mut entry, self.header_address, widths.offset);
        entry.extend_from_slice(&self.cache_type.to_le_bytes());
        entry.extend_from_slice(&0u32.to_le_bytes());
        entry.extend_from_slice(&self.scratch_pad);
        entry
    }
}

pub const SIGNATURE: &[u8; 4] = b"SNOD";

/// signature(4) + version(1) + reserved(1) + symbol count(2), which a node's entries follow.
pub const NODE_PREFIX: usize = 8;

/// The offset of the symbol count within a node.
pub const SYMBOL_COUNT_AT: usize = 6;

/// The cache type of a soft link's entry.
pub const SYMBOLIC_LINK: u32 = 2;

/// The scratch pad an entry ends with.
pub const SCRATCH_PAD: usize = 16;

const VERSION: u8 = 1;

const CACHE_TYPE_LEN: usize = 4;

const RESERVED_LEN: usize = 4;
