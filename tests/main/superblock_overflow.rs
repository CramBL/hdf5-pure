use hdf5_pure::Error;
use hdf5_pure::FormatError;
use rstest::rstest;

use super::superblock_addresses::ReadBackend;

fn overflowing_base_address_file() -> Vec<u8> {
    vec![
        137, 72, 68, 70, 13, 10, 26, 10, 0, 0, 0, 15, 3, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255,
        255, 255, 255, 255, 255, 255, 51, 65, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 223, 12,
        42, 186, 79, 72, 68, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 64, 182, 217, 83, 0, 0, 0, 0, 0, 0,
        0, 64, 182, 15, 188, 148, 5, 20, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 0,
    ]
}

fn assert_root_address_overflow(err: Error) {
    match err {
        Error::Format(FormatError::OffsetOverflow { offset, length }) => {
            assert_eq!(offset, 0x53d9_b640_0000);
            assert_eq!(length, u64::MAX);
        }
        other => panic!("expected root-group address overflow, got {other:?}"),
    }
}

#[rstest]
#[case(ReadBackend::Buffered)]
#[case(ReadBackend::Streaming)]
#[case(ReadBackend::Bytes)]
#[case(ReadBackend::Source)]
fn root_address_overflow_is_reported_when_navigation_resolves_it(#[case] backend: ReadBackend) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overflowing-base-address.h5");
    std::fs::write(&path, overflowing_base_address_file()).unwrap();
    let file = backend.open(&path).unwrap();
    assert_eq!(file.superblock().base_address.get(), u64::MAX);
    assert_eq!(file.superblock().root_group_address.get(), 0x53d9_b640_0000);
    assert_root_address_overflow(file.group("/").unwrap_err());
    assert_root_address_overflow(file.root().groups().unwrap_err());
    assert_root_address_overflow(file.dataset("child").unwrap_err());
}
