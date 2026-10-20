use std::collections::HashMap;

use hdf5_pure::{AttrValue, File, FileBuilder, FileSpaceStrategy};
use rstest::rstest;

#[rstest]
#[case::integer(AttrValue::I64(42))]
#[case::variable_length(AttrValue::VarLenStringArray(vec!["new".into(), "values".into()]))]
fn replaced_attributes_discard_unresolved_committed_paths(
    #[case] value: AttrValue,
    #[values(false, true)] dense: bool,
    #[values(false, true)] paged: bool,
    #[values(false, true)] write: bool,
) {
    let mut builder = FileBuilder::new();
    if paged {
        builder.with_file_space_strategy(FileSpaceStrategy::Page, false, 1);
    }
    builder.set_attr("first", AttrValue::VarLenString("preserved".into()));
    builder.set_attr_committed("replace", AttrValue::I32(7), "missing_type");
    builder.set_attr("last", AttrValue::I32(3));
    builder.set_attr("replace", value.clone());
    let dataset = builder.create_dataset("data").with_i32_data(&[1]);
    dataset.set_attr("first", AttrValue::VarLenString("preserved".into()));
    dataset.set_attr_committed("replace", AttrValue::I32(7), "missing_type");
    dataset.set_attr("last", AttrValue::I32(3));
    dataset.set_attr("replace", value.clone());
    if dense {
        for index in 0..10 {
            dataset.set_attr(&format!("filler{index}"), AttrValue::I32(index));
        }
    }
    let mut group = builder.create_group("group");
    group.set_attr("first", AttrValue::VarLenString("preserved".into()));
    group.set_attr_committed("replace", AttrValue::I32(7), "missing_type");
    group.set_attr("last", AttrValue::I32(3));
    group.set_attr("replace", value.clone());
    if dense {
        for index in 0..10 {
            builder.set_attr(&format!("filler{index}"), AttrValue::I32(index));
            group.set_attr(&format!("filler{index}"), AttrValue::I32(index));
        }
    }
    builder.add_group(group.finish());
    let file = if write {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("attributes.h5");
        builder.write(&path).unwrap();
        File::from_bytes(std::fs::read(path).unwrap()).unwrap()
    } else {
        File::from_bytes(builder.finish().unwrap()).unwrap()
    };
    let mut expected = HashMap::from([
        (
            "first".to_string(),
            AttrValue::VarLenString("preserved".into()),
        ),
        ("replace".to_string(), value),
        ("last".to_string(), AttrValue::I32(3)),
    ]);
    if dense {
        expected.extend((0..10).map(|index| (format!("filler{index}"), AttrValue::I32(index))));
    }
    assert_eq!(
        [
            file.root().attrs().unwrap(),
            file.group("group").unwrap().attrs().unwrap(),
            file.dataset("data").unwrap().attrs().unwrap(),
        ],
        [expected.clone(), expected.clone(), expected]
    );
}
