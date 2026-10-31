//! The [`PlacedSource`] the tests of the metadata readers share.

use alloc::vec::Vec;
use core::cell::RefCell;

use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::BytesSource;
use crate::source::PhysicalSource;
use crate::source::Source;
use crate::source::StoredSource;

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

    /// Borrows a stored-coordinate view with `start` as its effective base.
    pub(crate) fn framed(&self) -> StoredSource<'_, Self> {
        PhysicalSource::for_test(
            self,
            AddressContext::from_effective_base(BaseAddress::new(self.start)),
        )
        .stored()
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
