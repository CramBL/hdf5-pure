#![cfg(feature = "hdf5")]

use hdf5_pure::AttrValue;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure::LibVer;

#[test]
fn c_reads_attached_and_detached_group_hierarchies() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("groups.h5");
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder
        .create_group_with("attached", |group| {
            group.set_attr("version", AttrValue::I32(7));
            group.create_dataset("data").with_i32_data(&[1, 2, 3]);
            group.create_group_with("nested", |nested| {
                nested.set_attr("scale", AttrValue::F64(0.5));
                nested.create_dataset("values").with_f64_data(&[4.0, 5.0]);
                Ok(())
            })?;
            let mut detached = group.create_group("detached");
            detached.create_dataset("data").with_i32_data(&[42]);
            detached.set_attr("version", AttrValue::I32(1));
            group.add_group(detached.finish());
            let error = group
                .create_group_with("failed", |failed| {
                    failed.create_dataset("data").with_i32_data(&[0]);
                    Err(FormatError::InvalidLinkName("callback error".into()))
                })
                .unwrap_err();
            assert_eq!(error, FormatError::InvalidLinkName("callback error".into()));
            Ok(())
        })
        .unwrap();
    let mut detached = builder.create_group("detached");
    detached.create_dataset("data").with_i32_data(&[8, 9]);
    builder.add_group(detached.finish());
    drop(builder.create_group("discarded").finish());
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let mut names = file.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["attached", "detached"]);
    let attached = file.group("attached").unwrap();
    let mut names = attached.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["data", "detached", "nested"]);
    assert_eq!(attached.attr_names().unwrap(), ["version"]);
    assert_eq!(
        attached
            .attr("version")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        7
    );
    assert_eq!(
        attached.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [1, 2, 3]
    );
    let nested = attached.group("nested").unwrap();
    assert_eq!(nested.member_names().unwrap(), ["values"]);
    assert_eq!(nested.attr_names().unwrap(), ["scale"]);
    assert_eq!(
        nested.attr("scale").unwrap().read_scalar::<f64>().unwrap(),
        0.5
    );
    assert_eq!(
        nested.dataset("values").unwrap().read_raw::<f64>().unwrap(),
        [4.0, 5.0]
    );
    assert_eq!(
        attached
            .dataset("detached/data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [42]
    );
    assert_eq!(
        attached
            .group("detached")
            .unwrap()
            .attr("version")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        1
    );
    assert_eq!(
        file.dataset("detached/data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [8, 9]
    );
}
