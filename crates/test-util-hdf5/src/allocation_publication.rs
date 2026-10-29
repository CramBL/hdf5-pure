//! Native fixtures and observations for allocation publication crosschecks.

use std::fs;
use std::path::Path;

use hdf5::Extent;
use hdf5::File;
use hdf5::plist::file_create::FileSpaceStrategy;

use crate::file_space;
use crate::numeric::ByteOrder;

/// Chunk records, persisted free sections, and values read through libhdf5.
#[derive(Debug)]
pub struct Observation {
    /// Chunk offsets, physical addresses, stored byte lengths, and filter masks, sorted by offset.
    pub chunks: Vec<(Vec<u64>, u64, u64, u32)>,
    /// Free-section addresses and byte lengths reported by `H5Fget_free_sections`.
    pub free: Vec<(u64, u64)>,
    /// Decoded elements of the dataset named `d`.
    pub values: Vec<i32>,
}

impl Observation {
    /// Reads the dataset and free-space metadata without changing the fixture bytes.
    ///
    /// Checks the values of the dataset named `ceiling` and closes the native file before
    /// checking that its bytes are unchanged.
    ///
    /// # Panics
    ///
    /// Panics if reading, inspecting, or closing the fixture fails, the ceiling values differ
    /// from the fixture values, or inspection changes the file bytes.
    pub fn read(path: &Path) -> Self {
        let before = fs::read(path).unwrap();
        let file = File::open(path).unwrap();
        let dataset = file.dataset("d").unwrap();
        let values = dataset.read_raw::<i32>().unwrap();
        let mut chunks = (0..dataset.num_chunks().unwrap())
            .map(|index| {
                let chunk = dataset.chunk_info(index).unwrap();
                (chunk.offset, chunk.addr, chunk.size, chunk.filter_mask)
            })
            .collect::<Vec<_>>();
        chunks.sort_unstable();
        let free = file_space::libhdf5_free_sections(&file).unwrap();
        assert_eq!(
            file.dataset("ceiling").unwrap().read_raw::<u8>().unwrap(),
            [9; CEILING]
        );
        drop(dataset);
        file.close().unwrap();
        assert_eq!(
            fs::read(path).unwrap(),
            before,
            "native inspection changes bytes"
        );
        Self {
            chunks,
            free,
            values,
        }
    }
}

/// Writes a persistent fixture with an interior free hole and returns its address and byte length.
///
/// Selects PAGE when `paged` is `true`, and FSM_AGGR otherwise, with a free-section threshold of one.
/// The unlimited dataset named `d` has 1024-element chunks and a little-endian [`i32`] datatype.
/// It is empty when `deferred` is `true`, and contains one complete chunk of consecutive integers
/// from zero otherwise. Enables Fletcher32 when `filtered` is `true`.
///
/// Deletes the contiguous dataset named `scratch` to create the hole. Live contiguous storage
/// for `ceiling` remains above the hole. Closes the file before returning.
///
/// # Panics
///
/// Panics if creating, writing, inspecting, deleting, or closing the fixture fails, or the hole
/// length or ceiling placement differs from the fixture requirements.
///
/// # Examples
///
/// ```
/// use test_util_hdf5::allocation_publication::{self, Observation};
///
/// let directory = tempfile::tempdir()?;
/// let path = directory.path().join("publication.h5");
/// let (_, length) = allocation_publication::write_fixture(&path, false, true, false);
/// assert_eq!(length, 131_072);
/// assert!(Observation::read(&path).values.is_empty());
///
/// allocation_publication::append(&path);
/// assert_eq!(Observation::read(&path).values, (0..1024).collect::<Vec<i32>>());
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn write_fixture(path: &Path, paged: bool, deferred: bool, filtered: bool) -> (u64, u64) {
    let file = File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties.file_space_strategy(FileSpaceStrategy::FreeSpaceManager {
                paged,
                persist: true,
                threshold: 1,
            })
        })
        .create(path)
        .unwrap();
    let length = if deferred { 0 } else { CHUNK };
    let builder = file
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .shape([Extent::resizable(length)])
        .chunk([CHUNK]);
    let builder = if filtered {
        builder.fletcher32()
    } else {
        builder
    };
    let dataset = builder.create("d").unwrap();
    drop(builder);
    if !deferred {
        dataset
            .write_raw(&(0..i32::try_from(CHUNK).unwrap()).collect::<Vec<_>>())
            .unwrap();
    }
    assert_eq!(dataset.storage_size() == 0, deferred);
    drop(dataset);
    let scratch = file
        .new_dataset::<u8>()
        .shape(HOLE)
        .create("scratch")
        .unwrap();
    scratch.write_raw(&vec![7_u8; HOLE]).unwrap();
    let hole = (scratch.offset().unwrap(), scratch.storage_size());
    drop(scratch);
    let ceiling = file
        .new_dataset::<u8>()
        .shape(CEILING)
        .create("ceiling")
        .unwrap();
    ceiling.write_raw(&[9_u8; CEILING]).unwrap();
    assert!(ceiling.offset().unwrap() >= hole.0 + hole.1);
    drop(ceiling);
    file.unlink("scratch").unwrap();
    file.close().unwrap();
    assert_eq!(hole.1, u64::try_from(HOLE).unwrap());
    hole
}

/// Appends one complete chunk through libhdf5 and returns its physical address.
///
/// Extends the dataset named `d` by 1024 consecutive integers, starting at its previous length.
/// Checks all decoded values and closes the file before returning.
///
/// # Panics
///
/// Panics if the fixture cannot be opened, extended, written, inspected, or closed, its element
/// indices exceed [`i32::MAX`], its length arithmetic overflows, or its decoded values differ
/// from the consecutive sequence.
pub fn append(path: &Path) -> u64 {
    let file = File::open_rw(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let start = dataset.shape()[0];
    let end = start + CHUNK;
    dataset.resize([end]).unwrap();
    dataset
        .write_slice(
            &(start..end)
                .map(|value| i32::try_from(value).unwrap())
                .collect::<Vec<_>>(),
            start..end,
        )
        .unwrap();
    assert_eq!(
        dataset.read_raw::<i32>().unwrap(),
        (0..end)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>()
    );
    let chunk = dataset
        .chunk_info(dataset.num_chunks().unwrap() - 1)
        .unwrap();
    let address = chunk.addr;
    drop(dataset);
    file.close().unwrap();
    address
}

/// Writes `length` bytes of value 19 to `native_reuse` and returns its physical address.
///
/// Creates a contiguous dataset through libhdf5 and closes the file before returning.
///
/// # Panics
///
/// Panics if opening, creating, writing, inspecting, or closing the fixture fails, the dataset
/// has no physical address, or its stored byte length differs from `length`.
pub fn allocate_contiguous(path: &Path, length: usize) -> u64 {
    let file = File::open_rw(path).unwrap();
    let dataset = file
        .new_dataset::<u8>()
        .shape(length)
        .create("native_reuse")
        .unwrap();
    dataset.write_raw(&vec![19_u8; length]).unwrap();
    let address = dataset.offset().unwrap();
    assert_eq!(dataset.storage_size(), u64::try_from(length).unwrap());
    drop(dataset);
    file.close().unwrap();
    address
}

const CHUNK: usize = 1024;
const HOLE: usize = 131_072;
const CEILING: usize = 8192;
