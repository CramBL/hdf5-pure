#![cfg(feature = "__hdf5-1.10")]

use hdf5::file::LibraryVersion;
use hdf5_pure::LengthWidth;
use hdf5_pure::OffsetWidth;
use hdf5_pure::SuperblockVersion;
use rstest::rstest;

use test_util::superblock;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn every_reader_finds_the_superblock_after_a_userblock_libhdf5_wrote(
    #[values(512, 1024, 2048, 4096, 65_536)] userblock: u64,
    #[values(
        (LibraryVersion::Earliest, LibraryVersion::V18, 32, superblock::VERSION_0),
        (LibraryVersion::Earliest, LibraryVersion::V18, 64, superblock::VERSION_1),
        (LibraryVersion::V18, LibraryVersion::V18, 32, superblock::VERSION_2),
        (LibraryVersion::V110, LibraryVersion::V110, 32, superblock::VERSION_3)
    )]
    format: (LibraryVersion, LibraryVersion, u32, u8),
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let (low, high, indexed_storage_k, version) = format;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock.h5");
    {
        let file = hdf5::File::with_options()
            .with_fcpl(|properties| properties.userblock(userblock).istore_k(indexed_storage_k))
            .with_fapl(|properties| properties.libver_bounds(low, high))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([3])
            .create("values")
            .unwrap()
            .write(&[1, 2, 3])
            .unwrap();
        file.close().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        superblock::version_at(&bytes, usize::try_from(userblock).unwrap()),
        version
    );
    assert!(hdf5_pure::is_hdf5(&path).unwrap());
    assert!(hdf5_pure::is_hdf5_bytes(&bytes));
    let file = backend.open(&path).unwrap();
    assert_eq!(
        file.superblock().version(),
        SuperblockVersion::try_from(version).unwrap()
    );
    assert_eq!(file.superblock().offset_size, OffsetWidth::Eight);
    assert_eq!(file.superblock().length_size, LengthWidth::Eight);
    assert_eq!(file.superblock().base_address.get(), userblock);
    assert_eq!(file.superblock().eof_address, bytes.len() as u64);
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
}
