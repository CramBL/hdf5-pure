use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

use crate::File;

#[rstest]
#[case::increase_public(2, false, false, 3)]
#[case::decrease_public(1, true, false, 2)]
#[case::balanced_public(1, false, false, 2)]
#[case::removed_dataset_user_public(1, false, true, 2)]
fn native_committed_type_counts_follow_copy_and_delete_commits(
    #[case] copies: usize,
    #[case] extra_alias: bool,
    #[case] remove_source: bool,
    #[case] expected: usize,
) {
    let path = test_util::temp::temp_path("native-public-counts.h5");
    let native = file::libhdf5_create_v18(&path);
    let datatype = ByteOrder::LittleEndian.i32_datatype();
    native.commit_datatype("type", &datatype).unwrap();
    let dataset = native
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape([2])
        .create("source")
        .unwrap();
    dataset.write(&[7_i32, 11]).unwrap();
    if extra_alias {
        native.link_hard("type", "alias").unwrap();
    }
    assert_eq!(
        native.loc_info_by_name("type").unwrap().num_links,
        if extra_alias { 3 } else { 2 }
    );
    drop(dataset);
    drop(datatype);
    native.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    for index in 0..copies {
        file.copy("source", &format!("copy{index}")).unwrap();
    }
    file.root().delete("type").unwrap();
    if extra_alias {
        file.root().delete("alias").unwrap();
    }
    if remove_source {
        file.root().delete("source").unwrap();
    }
    file.commit().unwrap();
    file.close().unwrap();
    let native = file::libhdf5_open(&path);
    let dataset = native.dataset("copy0").unwrap();
    let datatype = dataset.dtype().unwrap();
    assert!(datatype.is_committed());
    assert_eq!(
        datatype
            .as_location()
            .unwrap()
            .loc_info()
            .unwrap()
            .num_links,
        expected
    );
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [7, 11]);
    drop(datatype);
    drop(dataset);
    native.close().unwrap();
}
