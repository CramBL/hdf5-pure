//! Shared object header messages: section `subsec_fmt4_infra_sohm`,
//! version 4.0.

use crate::bytes;
use crate::checksum;
use crate::widths::Widths;

/// The bytes of the master table, one header per index, checksum included.
pub fn table(indexes: &[Index], widths: Widths) -> Vec<u8> {
    let mut table = TABLE_SIGNATURE.to_vec();
    for index in indexes {
        table.push(VERSION);
        table.push(index.kind.0);
        table.extend_from_slice(&index.message_type_flags.to_le_bytes());
        table.extend_from_slice(&index.min_message_size.to_le_bytes());
        table.extend_from_slice(&index.list_max.to_le_bytes());
        table.extend_from_slice(&index.btree_min.to_le_bytes());
        table.extend_from_slice(&index.message_count.to_le_bytes());
        for address in [index.index_address, index.heap_address] {
            bytes::push_address(&mut table, address, widths.offset);
        }
    }
    checksum::append(&mut table);
    table
}

/// One index in the master table: which message types it shares, when a
/// message is large enough to share, and where its records live.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Index {
    pub kind: Kind,
    /// One bit per message type the index shares.
    pub message_type_flags: u16,
    pub min_message_size: u32,
    /// How many records the index holds before it becomes a B-tree.
    pub list_max: u16,
    /// How few it holds before a B-tree becomes a list again.
    pub btree_min: u16,
    pub message_count: u16,
    pub index_address: Option<u64>,
    pub heap_address: Option<u64>,
}

/// The bytes of a list index holding `records`, checksum included.
pub fn list(records: &[Vec<u8>]) -> Vec<u8> {
    let mut list = LIST_SIGNATURE.to_vec();
    list.extend(records.iter().flatten());
    checksum::append(&mut list);
    list
}

/// One record whose message is stored in the shared-message fractal heap,
/// padded out to the width every record in an index shares.
pub fn heap_record(hash: u32, reference_count: u32, heap_id: [u8; 8], widths: Widths) -> Vec<u8> {
    let mut record = vec![HEAP_LOCATION];
    record.extend_from_slice(&hash.to_le_bytes());
    record.extend_from_slice(&reference_count.to_le_bytes());
    record.extend_from_slice(&heap_id);
    record.resize(record_len(widths), 0);
    record
}

/// The width every record in an index shares: the wider of the two locations a
/// record can hold, behind the location byte and the hash.
fn record_len(widths: Widths) -> usize {
    let object_header = 1 + 1 + 2 + widths.offset;
    RECORD_PREFIX + HEAP_LOCATION_LEN.max(object_header)
}

/// Whether an index holds its records in a list or in a version 2 B-tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Kind(pub u8);

impl Kind {
    pub const LIST: Self = Self(0);
    pub const BTREE: Self = Self(1);
}

pub const TABLE_SIGNATURE: &[u8; 4] = b"SMTB";

pub const LIST_SIGNATURE: &[u8; 4] = b"SMLI";

/// The location byte of a record whose message lives in the fractal heap.
const HEAP_LOCATION: u8 = 0;

/// A heap location is a reference count and an eight-byte heap identifier.
const HEAP_LOCATION_LEN: usize = 4 + 8;

/// The location byte and the message's hash, which every record begins with.
const RECORD_PREFIX: usize = 1 + 4;

const VERSION: u8 = 0;
