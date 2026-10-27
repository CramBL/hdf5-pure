//! Reference fixtures and readback for PAGE publication recovery.

use std::path::Path;

use hdf5::File;
use hdf5::FileBuilder;
use hdf5::plist::file_create::FileSpaceStrategy;

/// The file-space strategies rejected by the PAGE publication failure hook.
#[derive(Clone, Copy, Debug)]
pub enum PublicationStrategy {
    /// Aggregates allocations without free-space managers.
    Aggr,
    /// Uses non-paged, non-persistent free-space managers and aggregation.
    FsmAggr,
    /// Allocates without free-space managers or aggregation.
    None,
    /// Uses paged allocation with persistent free-space managers.
    PersistentPage,
}

/// Writes a non-persistent PAGE fixture with datasets and a group attribute through libhdf5.
///
/// Sets `page_size` and `userblock` in bytes. The `keep`, `replace`, and `remove` datasets
/// contain 64 copies of 7, 3, and 5, respectively. The `group` group's scalar `attribute` is 9.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
pub fn write_origin(path: &Path, page_size: u64, userblock: u64) {
    let file = FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| {
            fcpl.userblock(userblock)
                .file_space_strategy(FileSpaceStrategy::FreeSpaceManager {
                    paged: true,
                    persist: false,
                    threshold: 1,
                })
                .file_space_page_size(page_size)
        })
        .create(path)
        .unwrap();
    for (name, value) in [("keep", 7), ("replace", 3), ("remove", 5)] {
        file.new_dataset::<u8>()
            .shape(64)
            .create(name)
            .unwrap()
            .write_raw(&[value; 64])
            .unwrap();
    }
    let group = file.create_group("group").unwrap();
    group
        .new_attr::<i32>()
        .create("attribute")
        .unwrap()
        .write_scalar(&9)
        .unwrap();
    drop(group);
    file.close().unwrap();
}

/// Writes a fixture with `strategy` and a `keep` dataset through libhdf5.
///
/// The dataset contains 64 copies of 7.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
pub fn write_unsupported_origin(path: &Path, strategy: PublicationStrategy) {
    let strategy = match strategy {
        PublicationStrategy::Aggr => FileSpaceStrategy::PageAggregation,
        PublicationStrategy::FsmAggr => FileSpaceStrategy::FreeSpaceManager {
            paged: false,
            persist: false,
            threshold: 1,
        },
        PublicationStrategy::None => FileSpaceStrategy::None,
        PublicationStrategy::PersistentPage => FileSpaceStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 1,
        },
    };
    let file = FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| fcpl.file_space_strategy(strategy))
        .create(path)
        .unwrap();
    file.new_dataset::<u8>()
        .shape(64)
        .create("keep")
        .unwrap()
        .write_raw(&[7; 64])
        .unwrap();
    file.close().unwrap();
}

/// Checks recovered datasets and PAGE settings through libhdf5, then overwrites `second`.
///
/// Checks the root member names, dataset values, non-persistent PAGE strategy, and page size of
/// 4,096 bytes. Writes 64 copies of 29 to `second` and closes the file.
///
/// # Panics
///
/// Panics if libhdf5 cannot read, write, or close the fixture, or an assertion fails.
pub fn assert_recovered_and_mutate(path: &Path) {
    let file = File::open_rw(path).unwrap();
    let mut paths = file.member_names().unwrap();
    paths.sort_unstable();
    assert_eq!(paths, ["added", "group", "keep", "replace", "second"]);
    for (path, value) in [("keep", 7), ("replace", 17), ("added", 9), ("second", 11)] {
        assert_eq!(
            file.dataset(path).unwrap().read_raw::<u8>().unwrap(),
            [value; 64]
        );
    }
    assert_eq!(
        file.fcpl().unwrap().get_file_space_strategy().unwrap(),
        FileSpaceStrategy::FreeSpaceManager {
            paged: true,
            persist: false,
            threshold: 1
        }
    );
    assert_eq!(
        file.fcpl().unwrap().get_file_space_page_size().unwrap(),
        4096
    );
    file.dataset("second")
        .unwrap()
        .write_raw(&[29_u8; 64])
        .unwrap();
    file.close().unwrap();
}
