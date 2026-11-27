use alloc::boxed::Box;
use core::num::NonZeroU16;

/// A set of global heap object indices, one bit for each `u16` value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct IndexSet(Box<[u64; INDEX_WORDS]>);

impl IndexSet {
    pub(super) fn new() -> Self {
        Self(Box::new([0; INDEX_WORDS]))
    }

    /// Adds `index` to the set, and returns whether it was newly inserted.
    pub(super) fn insert(&mut self, index: NonZeroU16) -> bool {
        let index = index.get();
        // `index / 64` is at most `u16::MAX / 64`, the last of the `INDEX_WORDS` words.
        let word = &mut self.0[usize::from(index / 64)];
        let bit = 1 << (index % 64);
        let inserted = *word & bit == 0;
        *word |= bit;
        inserted
    }
}

/// The most objects one global heap collection holds, 65,535.
///
/// The "Heap Object Index" field is 2 bytes wide, and index 0 is the free space. A writer with more
/// objects writes them to several collections, each with its objects at indices from 1. The field
/// is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub const GLOBAL_HEAP_MAX_OBJECTS: usize = u16::MAX as usize;

/// The number of 64-bit words in an [`IndexSet`].
const INDEX_WORDS: usize = (u16::MAX as usize + 1) / 64;
