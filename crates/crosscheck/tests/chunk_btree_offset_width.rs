use std::path::Path;

use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5_pure::File;
use rstest::rstest;
use tempfile::tempdir;

fn write_c_chunked_file(path: &Path, sizeof_addr: Sizeof, sizeof_size: Sizeof) {
    let sizes = SizeofInfo {
        sizeof_addr,
        sizeof_size,
    };
    let f = hdf5::File::with_options()
        .with_create_plist(|p| p.sizes(sizes))
        .create(path)
        .unwrap();

    // 1D dataset of 10 elements, chunk size 5 -> 2 chunks
    let ds = f
        .new_dataset::<i32>()
        .chunk((5,))
        .shape((10,))
        .create("dset1d")
        .unwrap();
    let data: Vec<i32> = (0..10).collect();
    ds.write(&data).unwrap();

    // 2D dataset of 4x4 elements, chunk size 2x2 -> 4 chunks
    let ds2d = f
        .new_dataset::<i32>()
        .chunk((2, 2))
        .shape((4, 4))
        .create("dset2d")
        .unwrap();
    let data2d: Vec<i32> = (0..16).collect();
    ds2d.write_raw(&data2d).unwrap();

    f.close().unwrap();
}

#[rstest]
#[case(Sizeof::Bytes2, Sizeof::Bytes2)]
#[case(Sizeof::Bytes4, Sizeof::Bytes4)]
#[case(Sizeof::Bytes2, Sizeof::Bytes8)]
#[case(Sizeof::Bytes4, Sizeof::Bytes8)]
#[case(Sizeof::Bytes8, Sizeof::Bytes2)]
#[case(Sizeof::Bytes8, Sizeof::Bytes4)]
#[case(Sizeof::Bytes4, Sizeof::Bytes2)]
#[case(Sizeof::Bytes2, Sizeof::Bytes4)]
fn read_chunked_dataset_with_various_offset_and_length_sizes(
    #[case] sizeof_addr: Sizeof,
    #[case] sizeof_size: Sizeof,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_sizes.h5");
    write_c_chunked_file(&path, sizeof_addr, sizeof_size);

    let pure_file = File::open(&path).unwrap();
    let pure_ds = pure_file.dataset("dset1d").unwrap();
    let read_back: Vec<i32> = pure_ds.read_i32().unwrap();
    let expected: Vec<i32> = (0..10).collect();
    assert_eq!(read_back, expected);

    let pure_ds2d = pure_file.dataset("dset2d").unwrap();
    let read_back2d: Vec<i32> = pure_ds2d.read_i32().unwrap();
    let expected2d: Vec<i32> = (0..16).collect();
    assert_eq!(read_back2d, expected2d);
}
