use std::collections::BTreeMap;

use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util_hdf5::creation_order;
use test_util_hdf5::file;

use crate::AttrValue;
use crate::File;
use crate::edit::group_inspection;

#[rstest]
#[case::compact(4)]
#[case::dense(64)]
#[case::wide_dense(128)]
fn batched_link_relocations_preserve_native_values_and_creation_metadata(
    #[case] count: usize,
    #[values("/", "/parent/g")] group_path: &str,
    #[values(false, true)] source_only: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("batch-links.h5");
    creation_order::write_link_relocation_fixture(&path, group_path, count);
    let before = group_inspection::inspect_group_links(&path, group_path, source_only).unwrap();
    assert_eq!(
        before.links.info().fractal_heap_address.is_some(),
        count > 4
    );
    let session = group_inspection::open_group_inspection(&path, source_only).unwrap();
    let group = session.group(group_path).unwrap();
    for index in 0..count - 1 {
        group
            .dataset(&format!("link_{index:04}"))
            .unwrap()
            .set_attr("changed", AttrValue::I64(i64::try_from(index).unwrap()))
            .unwrap();
    }
    group.delete(&format!("link_{:04}", count - 1)).unwrap();
    group
        .create_dataset("added", |dataset| {
            dataset.with_i32_data(&[71, 73]);
        })
        .unwrap();
    group
        .group("child")
        .unwrap()
        .create_group("nested")
        .unwrap();
    session.commit().unwrap();
    drop(group);
    drop(session);
    let after = group_inspection::inspect_group_links(&path, group_path, source_only).unwrap();
    assert_eq!(after.links.info().fractal_heap_address.is_some(), count > 4);
    assert_eq!(after.group_info, before.group_info);
    let actual: BTreeMap<_, _> = after
        .links
        .links()
        .iter()
        .map(|link| (link.name(), link))
        .collect();
    let deleted = format!("link_{:04}", count - 1);
    assert!(!actual.contains_key(deleted.as_bytes()));
    for old in before
        .links
        .links()
        .iter()
        .filter(|link| link.name() != deleted.as_bytes())
    {
        let new = actual[old.name()];
        assert_eq!(new.creation_order(), old.creation_order());
        assert_eq!(new.charset(), old.charset());
        match old.target() {
            OwnedLinkTarget::Hard { .. } => assert_ne!(new.target(), old.target()),
            OwnedLinkTarget::Indirect(_) => assert_eq!(new, old),
        }
    }
    assert_eq!(
        actual[b"added".as_slice()].creation_order(),
        Some(u64::try_from(count + 3).unwrap())
    );
    assert_eq!(actual.len(), count + 3);
    let reopened = File::open(&path).unwrap();
    for index in 0..count - 1 {
        let dataset = reopened
            .group(group_path)
            .unwrap()
            .dataset(&format!("link_{index:04}"))
            .unwrap();
        assert_eq!(
            dataset.read_i32().unwrap(),
            [i32::try_from(index).unwrap(), 42]
        );
        assert_eq!(
            dataset.attrs().unwrap()["changed"],
            AttrValue::I64(i64::try_from(index).unwrap())
        );
    }
    assert_eq!(
        reopened
            .group(group_path)
            .unwrap()
            .dataset("added")
            .unwrap()
            .read_i32()
            .unwrap(),
        [71, 73]
    );
    reopened
        .group(group_path)
        .unwrap()
        .group("child/nested")
        .unwrap();
    drop(reopened);
    let native = file::libhdf5_open(&path);
    let group = native.group(group_path).unwrap();
    for index in 0..count - 1 {
        let dataset = group.dataset(&format!("link_{index:04}")).unwrap();
        assert_eq!(
            dataset.read_raw::<i32>().unwrap(),
            [i32::try_from(index).unwrap(), 42]
        );
        assert_eq!(
            dataset
                .attr("changed")
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            i64::try_from(index).unwrap()
        );
    }
    assert_eq!(
        group.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        [71, 73]
    );
    group.group("child/nested").unwrap();
    drop(group);
    native.close().unwrap();
    directory.close().unwrap();
}
