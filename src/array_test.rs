//! The test helpers the Fixed Array and Extensible Array reads share.

use alloc::vec::Vec;
use core::cell::RefCell;

use hdf5_pure_format::__private::ChunkRecord;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::BaseOffsetSource;
use crate::source::BytesSource;
use crate::source::Source;

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

/// A source whose bytes start at address `start` and that records each metadata read.
pub(crate) struct PlacedSource {
    start: u64,
    bytes: Vec<u8>,
    /// The offset and the length of each metadata read, in the order of the reads.
    pub(crate) reads: RefCell<Vec<(u64, usize)>>,
}

impl PlacedSource {
    pub(crate) fn new(start: u64, bytes: Vec<u8>) -> Self {
        Self {
            start,
            bytes,
            reads: RefCell::new(Vec::new()),
        }
    }

    /// Returns the source with `start` as its base address.
    pub(crate) fn framed(&self) -> BaseOffsetSource<'_, Self> {
        BaseOffsetSource {
            inner: self,
            base: BaseAddress::new(self.start),
        }
    }

    /// Returns the length of each metadata read, in the order of the reads.
    pub(crate) fn read_lengths(&self) -> Vec<u64> {
        self.reads
            .borrow()
            .iter()
            .map(|&(_, len)| len.to_u64())
            .collect()
    }
}

impl Source for PlacedSource {
    fn len(&self) -> u64 {
        self.start + self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        BytesSource::new(&self.bytes).read_at(offset - self.start, buf)
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        self.reads.borrow_mut().push((offset, len));
        self.read_exact_at(offset, len)
    }
}
