//! Native persistent PAGE fixtures with alternating live and deleted page-sized datasets.

use std::path::Path;

use hdf5::File as NativeFile;
use hdf5::plist::file_create::FileSpaceStrategy;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::PersistedFreeSections;
use hdf5_pure::PersistedFreeSpacePolicy;
use hdf5_pure::SyncPolicy;

/// The number of deleted datasets in an alternating PAGE fixture.
#[derive(Clone, Copy, Debug)]
pub enum Fragmentation {
    /// Eight thousand deleted datasets and eight thousand survivors.
    EightThousand,
    /// Four thousand deleted datasets and four thousand survivors.
    FourThousand,
    /// One thousand deleted datasets and one thousand survivors.
    Thousand,
    /// Two thousand deleted datasets and two thousand survivors.
    TwoThousand,
}

impl Fragmentation {
    /// Returns the number of deleted datasets, equal to the number of survivors.
    pub fn holes(self) -> usize {
        match self {
            Self::Thousand => 1_000,
            Self::TwoThousand => 2_000,
            Self::FourThousand => 4_000,
            Self::EightThousand => 8_000,
        }
    }

    /// Writes a persistent PAGE file with libhdf5, then deletes every other dataset.
    ///
    /// Creates alternating `hole` and `keep` datasets with page-aligned, page-sized storage.
    /// Each survivor contains its index repeated across the page. Uses a one-byte persistence
    /// threshold and checks the persisted section count and free-byte floor after closing.
    ///
    /// # Panics
    ///
    /// Panics if creating, writing, deleting, inspecting, or closing fails, dataset storage is
    /// misaligned or has the wrong size, or persisted free space falls below the fixture's floor.
    pub fn write_native(self, path: &Path) {
        let file = NativeFile::with_options()
            .with_fapl(|properties| properties.libver_latest())
            .with_fcpl(|properties| {
                properties
                    .file_space_strategy(FileSpaceStrategy::FreeSpaceManager {
                        paged: true,
                        persist: true,
                        threshold: 1,
                    })
                    .file_space_page_size(PAGE_BYTES)
            })
            .create(path)
            .unwrap();
        for index in 0..self.holes() {
            let value = u32::try_from(index).unwrap();
            for (prefix, value) in [("hole", !value), ("keep", value)] {
                let name = format!("{prefix}{index:05}");
                let dataset = file
                    .new_dataset_builder()
                    .with_data(&vec![value; PAGE_ELEMENTS])
                    .create(name.as_str())
                    .unwrap();
                assert_eq!(dataset.storage_size(), PAGE_BYTES);
                assert_eq!(dataset.offset().unwrap() % PAGE_BYTES, 0);
            }
        }
        for index in 0..self.holes() {
            file.unlink(&format!("hole{index:05}")).unwrap();
        }
        file.close().unwrap();
        let report = persisted(path);
        assert!(report.free_sections().section_count() >= self.holes());
        assert!(report.free_sections().free_bytes() >= self.raw_free_bytes());
    }

    /// Checks every survivor's contents and the absence of deleted datasets through libhdf5.
    ///
    /// # Panics
    ///
    /// Panics if opening, reading, or closing fails, a survivor has different contents, or a
    /// deleted dataset's link remains present.
    pub fn assert_native_survivors(self, path: &Path) {
        let file = NativeFile::open(path).unwrap();
        for index in 0..self.holes() {
            let dataset = file.dataset(&format!("keep{index:05}")).unwrap();
            assert_eq!(
                dataset.read_raw::<u32>().unwrap(),
                vec![u32::try_from(index).unwrap(); PAGE_ELEMENTS]
            );
            assert!(!file.link_exists(&format!("hole{index:05}")));
        }
        file.close().unwrap();
    }

    /// Returns the raw-data byte count freed by deleting one dataset from each pair.
    pub fn raw_free_bytes(self) -> u64 {
        u64::try_from(self.holes())
            .unwrap()
            .checked_mul(PAGE_BYTES)
            .unwrap()
    }
}

/// Selects bounded writable opening with `policy` and [`SyncPolicy::OnClose`].
pub fn bounded_options(policy: PersistedFreeSpacePolicy) -> FileAccessProperties {
    FileAccessProperties::new()
        .with_memory_strategy(MemoryStrategy::Bounded)
        .with_sync_policy(SyncPolicy::OnClose)
        .with_persisted_free_space_policy(policy)
}

/// Returns the complete persisted-manager report after closing the streaming reader.
///
/// # Panics
///
/// Panics if opening, reading, or closing fails, or the file has no persisted-manager report.
pub fn persisted(path: &Path) -> PersistedFreeSections {
    let file = File::open_streaming(path).unwrap();
    let report = file.persisted_free_sections().unwrap().unwrap();
    file.close().unwrap();
    report
}

/// The page size and each dataset's raw storage length, in bytes.
pub const PAGE_BYTES: u64 = 4_096;
/// The number of `u32` elements in one fixture page.
pub const PAGE_ELEMENTS: usize = 1_024;
