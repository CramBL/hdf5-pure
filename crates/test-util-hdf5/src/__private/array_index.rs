//! The datasets and the chunk comparison the Fixed Array and Extensible Array crosschecks share.

use std::iter;
use std::ops::Range;
use std::path::Path;

use hdf5_pure::File;

/// A chunk as either library reports it, with its absolute address.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Chunk {
    offset: Vec<u64>,
    address: u64,
    size: u64,
    filter_mask: u32,
}

impl Chunk {
    /// Returns the chunk coordinates in dataset elements.
    pub fn offset(&self) -> &[u64] {
        &self.offset
    }

    /// Returns the absolute file address of the stored chunk.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// Returns the stored chunk size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the bit mask of filters skipped for this chunk.
    pub fn filter_mask(&self) -> u32 {
        self.filter_mask
    }
}

/// The elements a crosscheck writes into a rank 1 dataset of `len` elements: all of them, or those
/// of the ranges in `sparse`.
#[derive(Clone, Copy, Debug)]
pub struct Writes {
    /// The dataset length in elements.
    pub len: usize,
    /// The ranges to write, or `None` to write the whole dataset.
    pub sparse: Option<&'static [Range<usize>]>,
}

impl Writes {
    /// Writes the elements of each range into `dataset` through libhdf5.
    ///
    /// # Panics
    ///
    /// Panics if a selected range is reversed or extends past `len`, a generated element index
    /// does not fit in [`i32`], or libhdf5 cannot write a range.
    pub fn write(self, dataset: &hdf5::Dataset) {
        for range in self.ranges() {
            dataset
                .write_slice(&self.values()[range.clone()], range.clone())
                .unwrap();
        }
    }

    /// Returns the elements expected after [`write`](Self::write), with zeros in unwritten ranges.
    ///
    /// # Panics
    ///
    /// Panics if a selected range is reversed or extends past `len`, or a generated element index
    /// does not fit in [`i32`].
    ///
    /// # Examples
    ///
    /// ```
    /// use test_util_hdf5::__private::array_index::Writes;
    ///
    /// let writes = Writes { len: 4, sparse: Some(&[1..3]) };
    /// assert_eq!(writes.expected(), [0, 2, 3, 0]);
    /// ```
    pub fn expected(self) -> Vec<i32> {
        let mut expected = vec![0; self.len];
        for range in self.ranges() {
            expected[range.clone()].copy_from_slice(&self.values()[range.clone()]);
        }
        expected
    }

    fn ranges(self) -> Vec<Range<usize>> {
        self.sparse.map_or_else(
            || iter::once(0..self.len).collect(),
            <[Range<usize>]>::to_vec,
        )
    }

    fn values(self) -> Vec<i32> {
        (0..self.len)
            .map(|i| i32::try_from(i).unwrap() % 7 + 1)
            .collect()
    }
}

/// Returns the chunks libhdf5 reports for dataset `d` of the file at `path`, sorted.
///
/// Pass the file's userblock size as `userblock` to obtain absolute chunk addresses.
///
/// # Panics
///
/// Panics if the file or dataset cannot be opened, or chunk information cannot be read.
pub fn libhdf5_chunks(path: &Path, userblock: u64) -> Vec<Chunk> {
    #[cfg(feature = "__hdf5-1.14")]
    let mut chunks = {
        let file = hdf5::File::open(path).unwrap();
        let dataset = file.dataset("d").unwrap();
        let mut chunks = Vec::new();
        dataset
            .chunks_visit(|info| {
                chunks.push(Chunk {
                    offset: info.offset.to_vec(),
                    address: absolute(info.addr, userblock),
                    size: info.size,
                    filter_mask: info.filter_mask,
                });
                0
            })
            .unwrap();
        chunks
    };
    #[cfg(not(feature = "__hdf5-1.14"))]
    let mut chunks = {
        let file = hdf5::File::open(path).unwrap();
        let count = file.dataset("d").unwrap().num_chunks().unwrap();
        let indices: Vec<usize> = (0..count).collect();
        libhdf5_chunks_at(path, userblock, &indices)
    };
    chunks.sort();
    chunks
}

/// Returns the chunks libhdf5 reports at `indices` for dataset `d`, in the order of `indices`.
///
/// Pass the file's userblock size as `userblock` to obtain absolute chunk addresses.
///
/// # Panics
///
/// Panics if the file or dataset cannot be opened, or an index has no chunk information.
pub fn libhdf5_chunks_at(path: &Path, userblock: u64, indices: &[usize]) -> Vec<Chunk> {
    let file = hdf5::File::open(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    indices
        .iter()
        .map(|&index| {
            let info = dataset.chunk_info(index).unwrap();
            Chunk {
                offset: info.offset,
                address: absolute(info.addr, userblock),
                size: info.size,
                filter_mask: info.filter_mask,
            }
        })
        .collect()
}

// `H5D__get_chunk_info` in `H5Dchunk.c` reports a base-relative chunk address in HDF5 1.10.11 and
// 1.12.3, and adds the base address in 1.14.6 and 2.2.0.
fn absolute(address: u64, userblock: u64) -> u64 {
    if cfg!(feature = "__hdf5-1.14") {
        address
    } else {
        address + userblock
    }
}

/// Returns the chunks hdf5-pure reports for dataset `d` of `file`, sorted.
///
/// # Panics
///
/// Panics if dataset `d` cannot be opened or its chunk information cannot be read.
pub fn pure_chunks(file: &File) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = file
        .dataset("d")
        .unwrap()
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| {
            let filter_mask = chunk.filter_mask().raw_bits();
            Chunk {
                offset: chunk.offset,
                address: chunk.address,
                size: chunk.storage_size,
                filter_mask,
            }
        })
        .collect();
    chunks.sort();
    chunks
}
