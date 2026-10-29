use std::panic;
use std::panic::AssertUnwindSafe;

use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure::GroupBuilder;
use rstest::rstest;

#[test]
fn attaching_groups_retain_their_contents_once() {
    let mut builder = FileBuilder::new();
    let data = vec![1, 2, 3];
    builder
        .create_group_with("outer", |outer| {
            outer.set_attr("version", AttrValue::I32(7));
            outer.create_dataset("data").with_i32_data(&data);
            drop(data);
            outer.create_group_with("inner", |inner| {
                inner.set_attr("scale", AttrValue::F64(0.5));
                inner.create_dataset("values").with_f64_data(&[4.0, 5.0]);
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(file.root().groups().unwrap(), vec!["outer"]);
    let outer = file.group("outer").unwrap();
    assert_eq!(outer.groups().unwrap(), vec!["inner"]);
    assert_eq!(outer.datasets().unwrap(), vec!["data"]);
    assert_eq!(outer.attrs().unwrap()["version"], AttrValue::I32(7));
    assert_eq!(
        outer.dataset("data").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
    let inner = outer.group("inner").unwrap();
    assert_eq!(inner.attrs().unwrap()["scale"], AttrValue::F64(0.5));
    assert_eq!(inner.datasets().unwrap(), vec!["values"]);
    assert_eq!(
        inner.dataset("values").unwrap().read_f64().unwrap(),
        [4.0, 5.0]
    );
}

#[rstest]
#[case::root(false)]
#[case::nested(true)]
fn callback_failure_leaves_the_parent_unchanged(#[case] nested: bool) {
    let mut builder = FileBuilder::new();
    builder.create_group_with("preserved", |_| Ok(())).unwrap();
    let configure = |group: &mut GroupBuilder| {
        group.set_attr("discarded", AttrValue::I32(1));
        group.create_dataset("discarded").with_i32_data(&[2]);
        group.create_group_with("discarded", |_| Ok(()))?;
        Err(FormatError::InvalidLinkName("callback error".into()))
    };
    let error = if nested {
        builder
            .create_group_with("parent", |parent| {
                let error = parent.create_group_with("failed", configure).unwrap_err();
                assert_eq!(error, FormatError::InvalidLinkName("callback error".into()));
                parent.create_group_with("retained", |_| Ok(()))
            })
            .unwrap();
        None
    } else {
        Some(builder.create_group_with("failed", configure).unwrap_err())
    };
    if let Some(error) = error {
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(error, FormatError::InvalidLinkName("callback error".into()));
    }
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    if nested {
        assert_eq!(file.root().groups().unwrap(), ["preserved", "parent"]);
        assert_eq!(
            file.group("parent").unwrap().groups().unwrap(),
            ["retained"]
        );
    } else {
        assert_eq!(file.root().groups().unwrap(), ["preserved"]);
    }
}

#[rstest]
#[case::empty("")]
#[case::dot(".")]
#[case::path("a/b")]
fn invalid_names_are_rejected_before_the_callback(
    #[case] name: &str,
    #[values(false, true)] nested: bool,
) {
    let mut builder = FileBuilder::new();
    let configure = |_: &mut GroupBuilder| {
        panic!("a rejected name reaches the callback");
    };
    if nested {
        builder
            .create_group_with("parent", |parent| {
                assert_eq!(
                    parent.create_group_with(name, configure).unwrap_err(),
                    FormatError::InvalidLinkName(name.into())
                );
                Ok(())
            })
            .unwrap();
    } else {
        let error = builder.create_group_with(name, configure).unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(error, FormatError::InvalidLinkName(name.into()));
    }
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    if nested {
        assert_eq!(
            file.group("parent").unwrap().groups().unwrap(),
            Vec::<String>::new()
        );
    } else {
        assert_eq!(file.root().groups().unwrap(), Vec::<String>::new());
    }
}

#[rstest]
#[case::literal_parent("..")]
#[case::utf8("測定")]
fn attaching_groups_accept_link_names(#[case] name: &str) {
    let mut builder = FileBuilder::new();
    builder
        .create_group_with(name, |group| group.create_group_with(name, |_| Ok(())))
        .unwrap();
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(file.root().groups().unwrap(), [name]);
    assert_eq!(file.group(name).unwrap().groups().unwrap(), [name]);
}

#[rstest]
#[case::different_link("different")]
#[case::path_shaped_replacement("invalid/name")]
fn replacement_builders_attach_under_the_requested_name(#[case] replacement_name: &str) {
    let mut builder = FileBuilder::new();
    let mut detached = builder.create_group(replacement_name);
    detached.create_dataset("data").with_i32_data(&[42]);
    builder
        .create_group_with("requested", |group| {
            *group = detached;
            let mut source = FileBuilder::new();
            group.create_group_with("nested", |nested| {
                *nested = source.create_group(replacement_name);
                nested.create_dataset("data").with_i32_data(&[7]);
                Ok(())
            })
        })
        .unwrap();
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(file.root().groups().unwrap(), ["requested"]);
    assert_eq!(
        file.group("requested").unwrap().groups().unwrap(),
        ["nested"]
    );
    assert_eq!(
        file.dataset("requested/data").unwrap().read_i32().unwrap(),
        [42]
    );
    assert_eq!(
        file.dataset("requested/nested/data")
            .unwrap()
            .read_i32()
            .unwrap(),
        [7]
    );
}

#[test]
fn detached_groups_attach_only_when_added() {
    let mut builder = FileBuilder::new();
    drop(builder.create_group("discarded_builder"));
    drop(builder.create_group("discarded_group").finish());
    let mut outer = builder.create_group("outer");
    let mut inner = outer.create_group("inner");
    inner.create_dataset("data").with_i32_data(&[42]);
    inner.set_attr("version", AttrValue::I32(1));
    outer.add_group(inner.finish());
    builder.add_group(outer.finish());
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(file.root().groups().unwrap(), ["outer"]);
    assert_eq!(file.group("outer").unwrap().groups().unwrap(), ["inner"]);
    assert_eq!(
        file.dataset("outer/inner/data")
            .unwrap()
            .read_i32()
            .unwrap(),
        [42]
    );
    assert_eq!(
        file.group("outer/inner").unwrap().attrs().unwrap()["version"],
        AttrValue::I32(1)
    );
}

#[rstest]
#[case::root(false)]
#[case::nested(true)]
fn callback_panic_leaves_parent_empty(#[case] nested: bool) {
    let mut builder = FileBuilder::new();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        builder
            .create_group_with("failed", |group| {
                group.create_dataset("data").with_i32_data(&[1]);
                if nested {
                    group.create_group_with("child", |_| panic!("callback panic"))?;
                }
                panic!("callback panic");
            })
            .unwrap();
    }));
    let error = result.unwrap_err();
    assert_eq!(error.downcast_ref::<&str>(), Some(&"callback panic"));
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(file.root().groups().unwrap(), Vec::<String>::new());
}
