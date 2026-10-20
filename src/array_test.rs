//! The test helpers the Fixed Array and Extensible Array reads share.

use alloc::vec::Vec;

use hdf5_pure_format::__private::ChunkRecord;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;

/// Returns `n` records 8 bytes apart from address 0x100, of 8 bytes each or, where `has_filters` is
/// set, of stored sizes that vary.
pub(crate) fn chunks(n: u64, has_filters: bool) -> Vec<ChunkRecord> {
    (0..n)
        .map(|i| ChunkRecord {
            address: StoredAddress::new(0x100 + i * 8),
            stored_size: if has_filters { 8 + (i % 7) } else { 8 },
            filter_mask: 0,
        })
        .collect()
}

/// Pairs each record with its index in `chunks` as its slot.
pub(crate) fn slotted(chunks: Vec<ChunkRecord>) -> Vec<(u64, ChunkRecord)> {
    (0..).zip(chunks).collect()
}
