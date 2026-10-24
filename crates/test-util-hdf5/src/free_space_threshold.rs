//! Exercises interior deletions across free-space persistence thresholds.

use std::ops::Range;
use std::path::Path;

use test_util::range;

use super::api::{File, FileBuilder, FileSpaceStrategy};
use super::file_space;

/// The fixture contents and deleted extents from a threshold workload.
pub struct Deletions {
    /// The original values of each dataset, indexed by creation order.
    pub data: Vec<Vec<i32>>,
    /// Whether each dataset was deleted, indexed by creation order.
    pub deleted: Vec<bool>,
    /// The deleted extents below the threshold whose preceding dataset survives.
    pub dropped: Vec<Range<u64>>,
}

/// Writes contiguous datasets and commits the selected interior deletions.
///
/// The caller passes each dataset's element count and deletion flag in `datasets`. The first and
/// last datasets are kept. The callback receives the editing session, the deleted extent, and
/// whether the preceding dataset survives after each deletion commit.
///
/// # Panics
///
/// Panics if a file operation fails, a workload assertion fails, or the callback panics.
pub fn run(
    path: &Path,
    persist: bool,
    threshold: u64,
    datasets: &[(usize, bool)],
    mut checkpoint: impl FnMut(&File, &Range<u64>, bool),
) -> Deletions {
    let data = datasets
        .iter()
        .enumerate()
        .map(|(i, &(len, _))| vec![i32::try_from(i).unwrap(); len])
        .collect::<Vec<_>>();
    let deleted = datasets
        .iter()
        .enumerate()
        .map(|(i, &(_, delete))| delete && i > 0 && i + 1 < datasets.len())
        .collect::<Vec<_>>();
    let mut builder = FileBuilder::new();
    for (i, values) in data.iter().enumerate() {
        builder
            .create_dataset(&format!("d{i}"))
            .with_i32_data(values);
    }
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, persist, threshold);
    builder.write(path).unwrap();
    let file = File::open_rw(path).unwrap();
    let mut dropped = Vec::new();
    for i in (0..data.len()).filter(|&i| deleted[i]) {
        let extent = file_space::contiguous_extent(&file, &format!("d{i}"));
        file.root().delete(&format!("d{i}")).unwrap();
        file.commit().unwrap();
        let isolated = !deleted[i - 1];
        checkpoint(&file, &extent, isolated);
        if isolated && extent.end - extent.start < threshold {
            dropped.push(extent);
        }
    }
    for extent in &dropped {
        assert_eq!(
            range::covered_len(
                &file.space_accounting().unwrap().reusable_free_space,
                extent
            ),
            0
        );
    }
    drop(file);
    Deletions {
        data,
        deleted,
        dropped,
    }
}
