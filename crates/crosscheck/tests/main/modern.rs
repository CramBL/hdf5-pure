//! Native conformance for compact and dense group fixtures.

#![cfg(feature = "hdf5")]
use hdf5::datatype::ByteOrder;
use hdf5_pure::DatatypeByteOrder;
use rstest::rstest;
use test_util::modern::Fixture;
use test_util::temp;
use test_util_hdf5::modern;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::groups(Fixture::Groups)]
#[case::many_links(Fixture::ManyLinks)]
fn libhdf5_link_message_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("native_modern.h5");
    modern::write_libhdf5(&path, fixture);
    let (pure_order, native_order) = if cfg!(target_endian = "big") {
        (DatatypeByteOrder::BigEndian, ByteOrder::BigEndian)
    } else {
        (DatatypeByteOrder::LittleEndian, ByteOrder::LittleEndian)
    };
    modern::assert_libhdf5_contents(&path, fixture, native_order);
    let bytes = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    modern::assert_contents(&file, fixture, pure_order);
    modern::assert_representation(&file, &bytes, fixture);
    drop(file);
}

test_util_hdf5::hdf5_1_10_tests! {
#[rstest]
#[case::groups(Fixture::Groups)]
#[case::many_links(Fixture::ManyLinks)]
fn libhdf5_reads_independent_link_message_files(#[case] fixture: Fixture) {
    let path = temp::temp_path("independent_modern.h5");
    std::fs::write(&path, fixture.build()).unwrap();
    modern::assert_libhdf5_contents(&path, fixture, ByteOrder::LittleEndian);
}
}
