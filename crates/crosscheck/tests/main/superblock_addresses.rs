#![cfg(feature = "hdf5")]

use hdf5::file::LibraryVersion;
use hdf5_pure::File;
use hdf5_pure::ReadSeekSource;
use rstest::rstest;
use test_util::superblock;

#[rstest]
#[case::version_zero(LibraryVersion::Earliest, 32, superblock::VERSION_0)]
#[case::version_one(LibraryVersion::Earliest, 33, superblock::VERSION_1)]
#[case::version_two(LibraryVersion::V18, 32, superblock::VERSION_2)]
fn libhdf5_superblock_retains_stored_addresses_across_open_and_edit(
    #[case] low: LibraryVersion,
    #[case] indexed_storage_k: u32,
    #[case] version: u8,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("addresses.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|properties| properties.libver_bounds(low, LibraryVersion::V18))
            .with_fcpl(|properties| properties.userblock(512).istore_k(indexed_storage_k))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([3])
            .create("child")
            .unwrap()
            .write(&[1, 2, 3])
            .unwrap();
        file.create_group("outer")
            .unwrap()
            .create_group("inner")
            .unwrap()
            .new_dataset::<i32>()
            .shape([3])
            .create("data")
            .unwrap()
            .write(&[4, 5, 6])
            .unwrap();
        file.close().unwrap();
    }
    let original = std::fs::read(&path).unwrap();
    assert_eq!(superblock::version_at(&original, 512), version);
    let root_before = superblock::root_header_address(&original, 512);
    assert_reader(&File::open(&path).unwrap(), &original);
    assert_reader(&File::open_streaming(&path).unwrap(), &original);
    assert_reader(&File::from_bytes(original.clone()).unwrap(), &original);
    assert_reader(
        &File::from_source(ReadSeekSource::new(std::fs::File::open(&path).unwrap()).unwrap())
            .unwrap(),
        &original,
    );
    {
        let file = File::open_rw(&path).unwrap();
        let root = file.root();
        root.create_dataset("added", |builder| {
            builder.with_i32_data(&[7, 8, 9]);
        })
        .unwrap();
        file.commit().unwrap();
        assert_eq!(
            root.dataset("added").unwrap().read_i32().unwrap(),
            [7, 8, 9]
        );
    }
    let edited = std::fs::read(&path).unwrap();
    assert_ne!(superblock::root_header_address(&edited, 512), root_before);
    assert_eq!(&edited[..512], &original[..512]);
    assert_reader(&File::open_streaming(&path).unwrap(), &edited);
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        [7, 8, 9]
    );
    assert_eq!(
        file.dataset("outer/inner/data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [4, 5, 6]
    );
    file.close().unwrap();
}

fn assert_reader(file: &File, encoded: &[u8]) {
    assert_eq!(file.superblock().base_address.get(), 512);
    assert_eq!(file.superblock().eof_address, encoded.len() as u64);
    assert_eq!(
        file.superblock().root_group_address.get(),
        superblock::root_header_address(encoded, 512)
    );
    assert_eq!(file.root().groups().unwrap(), ["outer"]);
    assert_eq!(
        file.root().dataset("child").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
    assert_eq!(
        file.group("outer")
            .unwrap()
            .dataset("inner/data")
            .unwrap()
            .read_i32()
            .unwrap(),
        [4, 5, 6]
    );
    assert_eq!(
        file.root()
            .group("/outer/inner")
            .unwrap()
            .dataset("data")
            .unwrap()
            .read_i32()
            .unwrap(),
        [4, 5, 6]
    );
}
