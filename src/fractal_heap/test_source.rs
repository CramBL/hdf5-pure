use core::cell::RefCell;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::FormatError;
use crate::source::{BytesSource, Source};

/// A fixture source that records each requested byte range.
pub(crate) struct ObservedSource<'a> {
    bytes: &'a [u8],
    pub(crate) reads: RefCell<Vec<(u64, usize)>>,
}

impl<'a> ObservedSource<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            reads: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn reads_at(&self, offset: u64) -> usize {
        self.reads
            .borrow()
            .iter()
            .filter(|&&(at, _)| at == offset)
            .count()
    }
}

impl Source for ObservedSource<'_> {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.reads.borrow_mut().push((offset, bytes.len()));
        BytesSource::new(self.bytes).read_at(offset, bytes)
    }

    fn read_metadata_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, FormatError> {
        self.reads.borrow_mut().push((offset, length));
        BytesSource::new(self.bytes).read_metadata_at(offset, length)
    }
}
