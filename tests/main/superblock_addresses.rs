use std::path::Path;

use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::LibVer;
use rstest::rstest;
use test_util::superblock;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case(ReadBackend::Buffered)]
#[case(ReadBackend::Streaming)]
#[case(ReadBackend::Bytes)]
#[case(ReadBackend::Source)]
fn opened_superblock_retains_encoded_addresses(
    #[case] backend: ReadBackend,
    #[values(LibVer::V18, LibVer::V110)] libver: LibVer,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stored-addresses.h5");
    write_fixture(&path, libver);
    let encoded = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    let metadata = file.superblock();
    assert_eq!(metadata.base_address.get(), 512);
    assert_eq!(metadata.eof_address, encoded.len() as u64);
    assert_eq!(
        metadata.root_group_address.get(),
        superblock::root_header_address(&encoded, 512)
    );
}

#[rstest]
#[case(ReadBackend::Buffered)]
#[case(ReadBackend::Streaming)]
#[case(ReadBackend::Bytes)]
#[case(ReadBackend::Source)]
fn stored_root_and_children_resolve_in_userblock_image(
    #[case] backend: ReadBackend,
    #[values(LibVer::V18, LibVer::V110)] libver: LibVer,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("navigation.h5");
    write_fixture(&path, libver);
    let file = backend.open(&path).unwrap();
    assert_eq!(file.root().groups().unwrap(), ["outer"]);
    assert_eq!(
        file.root().dataset("child").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
    assert_eq!(file.group("outer").unwrap().groups().unwrap(), ["inner"]);
    assert_eq!(
        file.dataset("outer/inner/data")
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

#[rstest]
#[case(LibVer::V18)]
#[case(LibVer::V110)]
fn root_relocation_stores_relative_address_in_userblock_image(#[case] libver: LibVer) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relocation.h5");
    write_fixture(&path, libver);
    let before = superblock::root_header_address(&std::fs::read(&path).unwrap(), 512);
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
    let encoded = std::fs::read(&path).unwrap();
    assert_ne!(superblock::root_header_address(&encoded, 512), before);
    assert_eq!(&encoded[..512], vec![0; 512]);
    let file = File::open_streaming(&path).unwrap();
    assert_eq!(
        file.superblock().root_group_address.get(),
        superblock::root_header_address(&encoded, 512)
    );
    assert_eq!(
        file.dataset("outer/inner/data")
            .unwrap()
            .read_i32()
            .unwrap(),
        [4, 5, 6]
    );
    assert_eq!(
        file.root().dataset("added").unwrap().read_i32().unwrap(),
        [7, 8, 9]
    );
}

fn write_fixture(path: &Path, libver: LibVer) {
    let mut builder = FileBuilder::new();
    builder
        .with_userblock(512)
        .with_libver_bounds(libver, libver);
    builder.create_dataset("child").with_i32_data(&[1, 2, 3]);
    let mut outer = builder.create_group("outer");
    let mut inner = outer.create_group("inner");
    inner.create_dataset("data").with_i32_data(&[4, 5, 6]);
    outer.add_group(inner.finish());
    builder.add_group(outer.finish());
    builder.write(path).unwrap();
}
