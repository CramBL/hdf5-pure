use std::collections::BTreeMap;

use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use super::completion_tests;
use crate::edit::test_util as edit_test_util;

#[rstest]
#[case::version_1(Fixture::CommittedDatatypeV1, "mytype", "first", 2)]
#[case::version_2(Fixture::HardLinkUndo, "aa", "bb", 1)]
fn native_readers_observe_the_published_count_and_surviving_values(
    #[case] fixture: Fixture,
    #[case] removed: &str,
    #[case] surviving: &str,
    #[case] count: usize,
    #[values(Publication::Ordered, Publication::Completion)] publication: Publication,
) {
    let path = test_util::temp::temp_path("native-published-count.h5");
    fixture.write(&path);
    fixture.assert_native(
        &path,
        if cfg!(target_endian = "big") {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        },
    );
    let mut engine = edit_test_util::open_engine(&path);
    let bounds = completion_tests::bounds(&mut engine, vec![], &[removed]);
    match publication {
        Publication::Ordered => bounds.publish_counts(BTreeMap::new()).unwrap(),
        Publication::Completion => bounds
            .with_publication(BTreeMap::new(), |scope| {
                scope.unwrap().write_counts().unwrap();
                Ok(())
            })
            .unwrap(),
    }
    drop(engine);
    let native = file::libhdf5_open(&path);
    {
        let dataset = native.dataset(surviving).unwrap();
        match fixture {
            Fixture::CommittedDatatypeV1 => {
                assert_eq!(dataset.read_raw::<i32>().unwrap(), [1, 2]);
                let datatype = dataset.dtype().unwrap();
                assert!(datatype.is_committed());
                assert_eq!(
                    datatype
                        .as_location()
                        .unwrap()
                        .loc_info()
                        .unwrap()
                        .num_links,
                    count
                );
                assert_eq!(native.member_names().unwrap(), ["first", "second"]);
            }
            Fixture::HardLinkUndo => {
                assert_eq!(dataset.read_raw::<i32>().unwrap(), [1, 2, 3]);
                assert_eq!(dataset.loc_info().unwrap().num_links, count);
                assert_eq!(native.member_names().unwrap(), ["bb"]);
            }
            other => panic!("unsupported native fixture: {other:?}"),
        }
    }
    native.close().unwrap();
}

#[derive(Clone, Copy)]
enum Publication {
    Completion,
    Ordered,
}
