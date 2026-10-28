//! Native reads of independent committed-datatype and hard-link layouts.

#![cfg(feature = "__hdf5-1.10")]

use rstest::rstest;
use test_util::identity::Fixture;
use test_util::temp;
use test_util_hdf5::native_fixtures::Fixture as NativeFixture;
use test_util_hdf5::numeric::ByteOrder;

#[rstest]
#[case::committed_datatype_v1(Fixture::CommittedDatatypeV1)]
#[case::hard_link_undo(Fixture::HardLinkUndo)]
fn libhdf5_reads_independent_identity_files(#[case] fixture: Fixture) {
    let path = temp::temp_path("identity.h5");
    std::fs::write(&path, fixture.build()).unwrap();
    NativeFixture::from(fixture).assert_native(&path, ByteOrder::LittleEndian);
}
