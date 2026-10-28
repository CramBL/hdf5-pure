//! Fresh native fixtures and retained files under `tests/data/c/`.
//!
//! Fresh tests check current constructor output across reader backends. Tests of retained
//! files check the same contents and storage expectations.
//! Constructors are supplied by [`test_util_hdf5::native_fixtures`].
//!
//! `just test-data::c` sets `HDF5_PURE_UPDATE_TEST_DATA` to rewrite the committed copies
//! through the bundled reference library. Without that variable, tests read the existing copies.
#![cfg(feature = "__hdf5-1.10")]

use std::path::PathBuf;

use hdf5_pure::File;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::committed_datatype_v1(Fixture::CommittedDatatypeV1)]
#[case::generic_large(Fixture::GenericLarge)]
#[case::hard_link_undo(Fixture::HardLinkUndo)]
#[case::huge_links_12(Fixture::HugeLinks12)]
#[case::paged_index(Fixture::PagedIndex)]
fn fresh_native_fixture_has_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path(fixture.file_name());
    fixture.write(&path);
    let order = native_order();
    fixture.assert_native(&path, order);
    let bytes = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    fixture.assert_contents(&file, order);
    fixture.assert_representation(&file, &bytes);
    drop(file);
}

#[rstest]
#[case::committed_datatype_v1(Fixture::CommittedDatatypeV1)]
#[case::generic_large(Fixture::GenericLarge)]
#[case::hard_link_undo(Fixture::HardLinkUndo)]
#[case::huge_links_12(Fixture::HugeLinks12)]
#[case::paged_index(Fixture::PagedIndex)]
fn committed_native_fixture_has_complete_contents(#[case] fixture: Fixture) {
    let (path, order) = committed_fixture(fixture);
    fixture.assert_native(&path, order);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    fixture.assert_contents(&file, order);
    fixture.assert_representation(&file, &bytes);
    drop(file);
}

fn committed_fixture(fixture: Fixture) -> (PathBuf, ByteOrder) {
    let committed = test_util::data("c").join(fixture.file_name());
    let order = if std::env::var_os(UPDATE).is_some() {
        let produced = temp::temp_path(fixture.file_name());
        fixture.write(&produced);
        std::fs::copy(&produced, &committed).unwrap();
        native_order()
    } else {
        ByteOrder::LittleEndian
    };
    assert!(
        committed.is_file(),
        "{committed:?} is missing; `just test-data::c` writes it"
    );
    (committed, order)
}

fn native_order() -> ByteOrder {
    if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    }
}

const UPDATE: &str = "HDF5_PURE_UPDATE_TEST_DATA";
