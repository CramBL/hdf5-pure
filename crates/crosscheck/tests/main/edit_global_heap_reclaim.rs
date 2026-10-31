#![cfg(feature = "hdf5")]

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ops::Range;
use std::path::Path;

use hdf5::file::LibraryVersion;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::types::VarLenUnicode;
use hdf5_pure::File;
use hdf5_pure::FreeSections;
use hdf5_pure::PhysicalAddress;
use tempfile::TempDir;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::aggregation;
use test_util_hdf5::file_space;
#[cfg(feature = "__hdf5-1.10")]
use test_util_hdf5::paged;

#[test]
fn libhdf5_reads_every_remaining_value_after_unreferenced_collections_are_reclaimed() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("reclaim.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .create(&path)
        .unwrap();
    file.close().unwrap();

    write_delete_and_read_back(&path, LibraryVersion::V18);
}

#[cfg(feature = "__hdf5-1.10")]
#[test]
fn libhdf5_reads_every_remaining_value_after_unreferenced_collections_are_reclaimed_in_pages() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("reclaim.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties
                .file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 0,
                })
                .file_space_page_size(PAGE_SIZE)
        })
        .create(&path)
        .unwrap();
    file.close().unwrap();

    write_delete_and_read_back(&path, LibraryVersion::V110);
    paged::assert_consistent(&path, PAGE_SIZE);
}

fn write_delete_and_read_back(path: &Path, low: LibraryVersion) {
    for names in SESSIONS {
        let file = reopen_libhdf5_file(path, low);
        for &name in names {
            file.new_dataset::<VarLenUnicode>()
                .shape([STRINGS.len()])
                .create(name)
                .unwrap()
                .write(&strings())
                .unwrap();
        }
        file.close().unwrap();
    }
    let file = reopen_libhdf5_file(path, low);
    file.new_dataset::<i32>()
        .shape([3])
        .create("tail")
        .unwrap()
        .write(&[1, 2, 3])
        .unwrap();
    file.close().unwrap();
    let collections = collection_of_each_dataset(path);
    assert_eq!(collections["a"], collections["b"]);
    assert_eq!(collections["c"], collections["d"]);
    let kept = collections["b"];
    let reclaimed = [collections["d"], collections["e"]];
    assert_eq!(
        BTreeSet::from([kept, reclaimed[0], reclaimed[1]]).len(),
        3,
        "{collections:?}"
    );
    let ranges: BTreeMap<u64, Range<u64>> =
        global_heap::collection_ranges(&std::fs::read(path).unwrap(), Widths::EIGHT)
            .into_iter()
            .map(|range| (range.start, range))
            .collect();
    let file = File::open_rw(path).unwrap();

    for name in ["a", "c", "d", "e"] {
        file.root().delete(name).unwrap();
    }
    file.commit().unwrap();

    let free = file.reusable_free_sections().unwrap();
    let _: &FreeSections<PhysicalAddress> = &free;
    for address in reclaimed {
        let range = &ranges[&address];
        assert!(
            free.sections()
                .iter()
                .any(|section| section.address().get() <= range.start
                    && range.end <= section.end_address().get())
                || file.file_size() <= range.start,
            "the collection at {range:?} is not reclaimed: {free:?}"
        );
    }
    let range = &ranges[&kept];
    assert!(
        free.sections()
            .iter()
            .all(|section| section.end_address().get() <= range.start
                || range.end <= section.address().get()),
        "the collection at {range:?} that b refers to is reclaimed: {free:?}"
    );
    file.close().unwrap();

    let file = hdf5::File::open(path).unwrap();
    let mut names = file.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["b", "tail"]);
    assert_eq!(
        file.dataset("b")
            .unwrap()
            .read_raw::<VarLenUnicode>()
            .unwrap(),
        strings()
    );
    assert_eq!(
        file.dataset("tail").unwrap().read_raw::<i32>().unwrap(),
        [1, 2, 3]
    );
}

fn reopen_libhdf5_file(path: &Path, low: LibraryVersion) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(low, LibraryVersion::latest()))
        .open_rw(path)
        .unwrap()
}

fn strings() -> Vec<VarLenUnicode> {
    STRINGS.map(|s| s.parse().unwrap()).to_vec()
}

fn collection_of_each_dataset(path: &Path) -> BTreeMap<&'static str, u64> {
    let bytes = std::fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    SESSIONS
        .iter()
        .flat_map(|names| names.iter())
        .map(|&name| {
            let data = file_space::contiguous_extent(&file, name);
            (name, aggregation::collection_address(&bytes, data.start))
        })
        .collect()
}

const SESSIONS: [&[&str]; 3] = [&["a", "b"], &["c", "d"], &["e"]];
const STRINGS: [&str; 2] = ["alpha", "beta"];
#[cfg(feature = "__hdf5-1.10")]
const PAGE_SIZE: u64 = 4096;
