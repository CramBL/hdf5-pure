use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy;

/// Writes twelve datasets under `g`, with link names longer than 5,000 bytes.
///
/// Each dataset contains its index as one [`i32`] value. The links occupy huge
/// fractal-heap objects in dense storage. Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn huge_links_12(path: &Path) {
    const COUNT: usize = 12;
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
    drop(group);
    file.close().unwrap();
}

/// Writes an unlimited `victim` dataset with 512-element chunks and a separate `keep` dataset.
///
/// `victim` contains [`i32`] values 0 through 8,191, and `keep` contains `[1, 2, 3, 4]`.
/// The file persists free-space managers with 4,096-byte pages. Closes the file
/// before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn paged_index(path: &Path) {
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
    drop(ds);
    f.close().unwrap();
}

/// Writes a 512-element attribute on `d` in a file with persisted 512-byte pages.
///
/// The dataset contains 64 [`f64`] values equal to 1.0, and its attribute `big_attr`
/// contains 512 [`i64`] values equal to 7. Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn generic_large(path: &Path) {
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
    drop(a);
    drop(ds);
    f.close().unwrap();
}

/// Writes `mytype` in a version 1 object header, shared by `first` and `second`.
///
/// The committed datatype is [`i32`], and both datasets contain `[1, 2]`.
/// Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn committed_datatype_v1(path: &Path) {
    let f = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18))
        .create(path)
        .unwrap();
    let dtype = hdf5::Datatype::from_type::<i32>().unwrap();
    f.commit_datatype("mytype", &dtype).unwrap();
    for name in ["first", "second"] {
        f.new_dataset_builder()
            .empty_as(&dtype)
            .shape([2])
            .create(name)
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
    }
    drop(dtype);
    f.close().unwrap();
}

/// Writes `aa` and `bb` as two hard links to one [`i32`] dataset containing `[1, 2, 3]`.
///
/// The file persists free-space managers without paging. Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn hard_link_undo(path: &Path) {
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
}

fn persisting(paged: bool) -> FileSpaceStrategy {
    FileSpaceStrategy::FreeSpaceManager {
        paged,
        persist: true,
        threshold: 1,
    }
}
