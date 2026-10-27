//! Editing fixtures and storage-ownership assertions for private tests.

use std::path::Path;

use crate::FileLocking;
use crate::PersistedFreeSpacePolicy;
use crate::edit::MemoryStrategy;
use crate::edit::SpaceOwnership;
use crate::edit::WriteEngine;
use crate::source::MetadataCacheConfig;

impl SpaceOwnership {
    /// Asserts complete classification of every byte in the file.
    ///
    /// # Panics
    ///
    /// Panics if the analysis is incomplete, any byte is unclassified, trailing slack is present,
    /// the physical and logical file lengths differ, or the extents do not partition the file
    /// into nonempty ranges.
    pub(crate) fn assert_exact(&self) {
        self.assert_classified(&[]);
    }

    /// Asserts that the ownership categories partition the file with the expected unclassified
    /// extents.
    ///
    /// # Panics
    ///
    /// Panics if the analysis is incomplete, the unclassified extents differ, trailing slack is
    /// present, the physical and logical file lengths differ, or the extents do not partition the
    /// file into nonempty ranges.
    pub(crate) fn assert_classified(&self, expected_unclassified: &[(u64, u64)]) {
        let Self {
            physical_size,
            logical_size,
            userblock_space,
            live_raw_space,
            live_metadata_space,
            free_space_manager_space,
            reusable_free_space,
            unclassified_space,
            analysis_complete,
            trailing_slack_space,
        } = self;
        assert!(*analysis_complete, "ownership analysis must be complete");
        assert_eq!(unclassified_space, expected_unclassified);
        assert_eq!(trailing_slack_space, &[]);
        assert_eq!(physical_size, logical_size);
        let mut extents = userblock_space
            .iter()
            .chain(live_raw_space)
            .chain(live_metadata_space)
            .chain(free_space_manager_space)
            .chain(reusable_free_space)
            .chain(unclassified_space)
            .chain(trailing_slack_space)
            .copied()
            .collect::<Vec<_>>();
        extents.sort_unstable();
        let mut end = 0;
        for (start, len) in extents {
            assert!(len > 0, "empty ownership extent at {start}");
            assert_eq!(start, end, "ownership partition has a gap or overlap");
            end = start.checked_add(len).unwrap();
            assert!(
                end <= *physical_size,
                "ownership extent exceeds physical EOF"
            );
        }
        assert_eq!(
            end, *physical_size,
            "ownership partition must cover the file"
        );
    }
}

/// Opens a writable fixture with metadata caching disabled.
///
/// # Panics
///
/// Panics if the fixture cannot be opened for editing.
pub(super) fn open_engine(path: &Path) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Auto,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap()
}
