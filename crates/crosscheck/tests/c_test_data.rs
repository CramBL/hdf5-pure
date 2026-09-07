//! The files under `tests/data/c/` that the crate's unit tests read.
//!
//! The reference C library is a dependency of this package alone, so a unit
//! test that needs what only it writes reads a committed file, written here.
//! `just test-data::c` rewrites them, and without `HDF5_PURE_UPDATE_TEST_DATA`
//! set each test checks the committed copy instead.
#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::plist::file_create::FileSpaceStrategy;

const UPDATE: &str = "HDF5_PURE_UPDATE_TEST_DATA";

/// The committed copy of `name`, rewritten from `write` first when `UPDATE` is set.
fn c_written(name: &str, write: impl FnOnce(&Path)) -> hdf5::File {
    let committed = test_util::data("c").join(name);
    if std::env::var_os(UPDATE).is_some() {
        let tmp = tempfile::tempdir().unwrap();
        let produced = tmp.path().join(name);
        write(&produced);
        std::fs::copy(&produced, &committed).unwrap();
    }
    assert!(
        committed.is_file(),
        "{} is missing; `just test-data::c` writes it",
        committed.display()
    );
    hdf5::File::open(&committed).unwrap()
}

fn persisting(paged: bool) -> FileSpaceStrategy {
    FileSpaceStrategy::FreeSpaceManager {
        paged,
        persist: true,
        threshold: 1,
    }
}

/// One group of 12 links, above the C library's `max_compact` of 8 so they are
/// stored densely, each name long enough that its link message is a huge heap
/// object. For `src/group_v2.rs`.
#[test]
fn huge_links_for_the_dense_link_walk() {
    const COUNT: usize = 12;
    let file = c_written(&format!("huge_links_{COUNT}.h5"), |path| {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_latest())
            .create(path)
            .unwrap();
        let group = file.create_group("g").unwrap();
        for i in 0..COUNT {
            let name = format!("d{i}_{}", "x".repeat(5000));
            group
                .new_dataset::<i32>()
                .shape((1,))
                .create(name.as_str())
                .unwrap()
                .write(&[i as i32])
                .unwrap();
        }
        file.close().unwrap();
    });
    assert_eq!(
        file.group("g").unwrap().member_names().unwrap().len(),
        COUNT
    );
}

/// A paged, persisting file whose chunked dataset's index the C library placed
/// as metadata. For `src/edit.rs`.
#[test]
fn paged_file_whose_chunk_index_the_c_library_placed() {
    let file = c_written("paged_index.h5", |path| {
        let f = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(persisting(true))
                    .file_space_page_size(4096)
            })
            .create(path)
            .unwrap();
        let ds = f
            .new_dataset::<i32>()
            .shape(hdf5::SimpleExtents::resizable(vec![8192]))
            .chunk((512,))
            .create("victim")
            .unwrap();
        ds.write_raw(&(0..8192i32).collect::<Vec<i32>>()).unwrap();
        f.new_dataset::<i32>()
            .shape((4,))
            .create("keep")
            .unwrap()
            .write_raw(&[1i32, 2, 3, 4])
            .unwrap();
        f.close().unwrap();
    });
    assert_eq!(file.dataset("victim").unwrap().shape(), vec![8192]);
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3, 4]
    );
}

/// A paged, persisting file with one attribute far larger than its page, so
/// the object-header chunk holding it is a large metadata allocation and the
/// generic-large manager holds a sub-page fragment. For `src/edit.rs`.
#[test]
fn paged_file_with_a_sub_page_fragment_in_the_large_manager() {
    let file = c_written("generic_large.h5", |path| {
        let f = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(persisting(true))
                    .file_space_page_size(512)
            })
            .create(path)
            .unwrap();
        let ds = f.new_dataset::<f64>().shape((64,)).create("d").unwrap();
        ds.write_raw(&vec![1.0f64; 64]).unwrap();
        let a = ds
            .new_attr::<i64>()
            .shape((512,))
            .create("big_attr")
            .unwrap();
        a.write_raw(&vec![7i64; 512]).unwrap();
        f.close().unwrap();
    });
    let attr = file.dataset("d").unwrap().attr("big_attr").unwrap();
    assert_eq!(attr.read_raw::<i64>().unwrap().len(), 512);
}

/// Two hard links to one dataset, which this crate has no API to create, in a
/// persisting file. For `src/edit.rs`.
#[test]
fn two_hard_links_to_one_dataset() {
    let file = c_written("hard_link_undo.h5", |path| {
        let f = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| fcpl.file_space_strategy(persisting(false)))
            .create(path)
            .unwrap();
        f.new_dataset::<i32>()
            .shape((3,))
            .create("aa")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        f.link_hard("aa", "bb").unwrap();
        f.close().unwrap();
    });
    for name in ["aa", "bb"] {
        assert_eq!(
            file.dataset(name).unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3]
        );
    }
}
