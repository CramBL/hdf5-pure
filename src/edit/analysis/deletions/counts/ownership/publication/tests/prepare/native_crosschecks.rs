use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use crate::convert;
use crate::edit::test_util as edit_test_util;

#[rstest]
#[case::version_1(Fixture::CommittedDatatypeV1, "mytype", 3)]
#[case::version_2(Fixture::HardLinkUndo, "aa", 2)]
fn preparation_reads_counts_from_native_headers(
    #[case] fixture: Fixture,
    #[case] name: &str,
    #[case] count: u32,
) {
    let path = test_util::temp::temp_path("native-current-count.h5");
    fixture.write(&path);
    fixture.assert_native(
        &path,
        if cfg!(target_endian = "big") {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        },
    );
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.loc_info_by_name(name).unwrap().num_links,
        usize::try_from(count).unwrap()
    );
    native.close().unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    let mut bounds = super::publication_bounds(&mut engine, &[], &[name]);
    let prepared = bounds.prepare_counts().unwrap();
    let (original, update) = prepared.updates().next().unwrap();
    assert_eq!(original.field().value(), count);
    assert_eq!(original.count().get(), count - 1);
    let range = original.allocation().range();
    let mut patched =
        before[convert::slice_range(range.start().get(), range.length()).unwrap()].to_vec();
    for (field, bytes) in update.writes() {
        patched[convert::slice_range(field.start().get() - range.start().get(), field.length())
            .unwrap()]
        .copy_from_slice(bytes);
    }
    let field = original.field().range();
    assert_eq!(
        &patched[convert::slice_range(field.start().get() - range.start().get(), field.length())
            .unwrap()],
        &(count - 1).to_le_bytes()
    );
    if matches!(fixture, Fixture::HardLinkUndo) {
        let footer = patched.len() - test_util::checksum::CHECKSUM;
        assert_eq!(
            &patched[footer..],
            &test_util::checksum::lookup3(&patched[..footer]).to_le_bytes()
        );
    }
    assert_eq!(prepared.engine.disk_bytes().unwrap(), before);
}
