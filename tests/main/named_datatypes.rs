//! Committed datatype inspection and in-place creation restrictions.

use hdf5_pure::AttrValue;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use rstest::rstest;
use test_util::identity::Fixture;
use test_util_hdf5::identity;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::dataset(|builder: &mut DatasetBuilder| { builder.with_committed_datatype("integer_type"); })]
#[case::attribute(|builder: &mut DatasetBuilder| { builder.set_attr_committed("scale", AttrValue::I32(7), "integer_type"); })]
fn in_place_creation_rejects_committed_datatypes_without_changing_the_file(
    #[case] configure: fn(&mut DatasetBuilder),
) {
    let path = test_util::temp::temp_path("committed.h5");
    let mut builder = FileBuilder::new();
    builder.commit_datatype("integer_type", hdf5_pure::make_i32_type());
    builder.create_dataset("preserved").with_i32_data(&[11, 13]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let error = file
            .root()
            .create_dataset("created", |builder| {
                builder.with_i32_data(&[1]);
                configure(builder);
            })
            .unwrap_err();
        let Error::EditUnsupported(reason) = error else {
            panic!("expected an unsupported committed datatype, got {error:?}");
        };
        assert_eq!(
            reason,
            "a dataset or attribute naming a committed (shared) datatype cannot be added in place; write the file with FileBuilder instead"
        );
        assert!(!file.has_staged_edits());
        file.commit().unwrap();
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("preserved")
            .unwrap()
            .read_i32()
            .unwrap(),
        [11, 13]
    );
}

#[rstest]
#[case::committed_datatype_v1(Fixture::CommittedDatatypeV1)]
#[case::hard_link_undo(Fixture::HardLinkUndo)]
fn independent_identity_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = test_util::temp::temp_path("identity.h5");
    let bytes = fixture.build();
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    identity::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
    identity::assert_storage(&file, &bytes, fixture);
    drop(file);
}
