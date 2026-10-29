//! Provides a threshold-one persistent free-space fixture for full-chunk append measurements.
//!
//! # Examples
//!
//! ```
//! use test_util_hdf5::file_space::publication;
//!
//! # let directory = tempfile::tempdir()?;
//! let path = directory.path().join("append.h5");
//! let file = publication::session(&path);
//! let mut dataset = file.dataset("d")?;
//! for chunk in publication::values(2).chunks(publication::CHUNK) {
//!     dataset.append(chunk)?;
//! }
//! drop(dataset);
//! file.close()?;
//! publication::verify(&path, 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::path::Path;

use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::SyncPolicy;

use crate::file_space;

/// Creates a writable fixture with a 4 MiB reusable hole below a live dataset.
///
/// Uses persistent [`FileSpaceStrategy::FsmAggr`] with a threshold of one byte,
/// [`MemoryStrategy::Bounded`], and [`SyncPolicy::OnClose`]. Dataset `d` contains one
/// [`CHUNK`] of consecutive integers starting at zero and has an unlimited extent.
/// Dataset `ceiling` contains 8192 bytes equal to nine above the reusable hole.
///
/// # Panics
///
/// Panics if creating, opening, editing, or inspecting fails, or the fixture geometry differs
/// from the expected layout.
pub fn session(path: &Path) -> File {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(1),
    );
    builder
        .create_dataset("d")
        .with_i32_data(&(0..CHUNK as i32).collect::<Vec<_>>())
        .with_chunks(&[CHUNK as u64])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder
        .create_dataset("scratch")
        .with_u8_data(&vec![7; 4 * 1024 * 1024]);
    builder
        .create_dataset("ceiling")
        .with_u8_data(&[9; CEILING]);
    builder.write(path).unwrap();
    let file = open(path);
    let hole = file_space::contiguous_extent(&file, "scratch");
    let ceiling = file_space::contiguous_extent(&file, "ceiling");
    assert_eq!(hole.end - hole.start, 4 * 1024 * 1024);
    assert!(hole.end <= ceiling.start);
    file.root().delete("scratch").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let file = open(path);
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .any(|section| {
                section.address() <= hole.start
                    && section.address() + section.length().get() >= hole.end
            })
    );
    file
}

/// Returns consecutive integers for `chunks` full chunks after the fixture's initial chunk.
///
/// # Panics
///
/// Panics if an integer exceeds [`i32::MAX`]. With overflow checks enabled, also panics if
/// computing the element count overflows.
pub fn values(chunks: usize) -> Vec<i32> {
    (CHUNK..(chunks + 1) * CHUNK)
        .map(|value| i32::try_from(value).unwrap())
        .collect()
}

/// Checks the complete dataset shape and values after appending `chunks` full chunks.
///
/// Also checks the unchanged contents of dataset `ceiling`.
///
/// # Panics
///
/// Panics if reading fails, either dataset differs from the expected contents, or an expected
/// integer exceeds [`i32::MAX`]. With overflow checks enabled, also panics if computing the
/// element count overflows.
pub fn verify(path: &Path, chunks: usize) {
    let file = File::open(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(
        dataset.shape().unwrap(),
        [(chunks + 1) as u64 * CHUNK as u64]
    );
    assert_eq!(
        dataset.read_i32().unwrap(),
        (0..(chunks + 1) * CHUNK)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().read_u8().unwrap(),
        [9; CEILING]
    );
}

fn open(path: &Path) -> File {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap()
}

/// The number of elements per chunk in dataset `d`.
pub const CHUNK: usize = 64;
const CEILING: usize = 8192;
