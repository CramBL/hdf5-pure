#![cfg(feature = "__hdf5-1.10")]

use hdf5::types::VarLenUnicode;
use hdf5_pure::{AttrValue, File, FileBuilder};
use rstest::rstest;

#[rstest]
fn c_reads_one_attribute_with_the_replacement_value(
    #[values(
        AttributeForm::Committed,
        AttributeForm::Inline,
        AttributeForm::VariableLength
    )]
    before: AttributeForm,
    #[values(
        AttributeForm::Committed,
        AttributeForm::Inline,
        AttributeForm::VariableLength
    )]
    after: AttributeForm,
    #[values(false, true)] dense: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replacement.h5");
    let mut builder = FileBuilder::new();
    builder.commit_datatype("integer_type", hdf5_pure::make_i32_type());
    stage_attributes(
        |name, value, committed| {
            if committed {
                builder.set_attr_committed(name, value, "integer_type");
            } else {
                builder.set_attr(name, value);
            }
        },
        before,
        after,
        dense,
    );
    let dataset = builder.create_dataset("data").with_i32_data(&[1]);
    stage_attributes(
        |name, value, committed| {
            if committed {
                dataset.set_attr_committed(name, value, "integer_type");
            } else {
                dataset.set_attr(name, value);
            }
        },
        before,
        after,
        dense,
    );
    let mut group = builder.create_group("group");
    stage_attributes(
        |name, value, committed| {
            if committed {
                group.set_attr_committed(name, value, "integer_type");
            } else {
                group.set_attr(name, value);
            }
        },
        before,
        after,
        dense,
    );
    builder.add_group(group.finish());
    builder.write(&path).unwrap();
    let file = hdf5::File::open(&path).unwrap();
    assert_attributes(&file, after, dense);
    assert_attributes(&file.group("group").unwrap(), after, dense);
    assert_attributes(&file.dataset("data").unwrap(), after, dense);
}

fn stage_attributes(
    mut set: impl FnMut(&str, AttrValue, bool),
    before: AttributeForm,
    after: AttributeForm,
    dense: bool,
) {
    set("first", AttrValue::VarLenString("preserved".into()), false);
    set(
        "replace",
        before.value(7),
        before == AttributeForm::Committed,
    );
    set("last", AttrValue::I32(3), false);
    if dense {
        for index in 0..10 {
            set(&format!("filler{index}"), AttrValue::I32(index), false);
        }
    }
    set(
        "replace",
        after.value(42),
        after == AttributeForm::Committed,
    );
}

fn assert_attributes(owner: &hdf5::Location, after: AttributeForm, dense: bool) {
    let mut expected = vec![
        "first".to_string(),
        "last".to_string(),
        "replace".to_string(),
    ];
    if dense {
        expected.extend((0..10).map(|index| format!("filler{index}")));
        for index in 0..10 {
            assert_eq!(
                owner
                    .attr(&format!("filler{index}"))
                    .unwrap()
                    .read_scalar::<i32>()
                    .unwrap(),
                index
            );
        }
    }
    expected.sort();
    let mut names = owner.attr_names().unwrap();
    names.sort();
    assert_eq!(names, expected);
    assert_eq!(owner.loc_info().unwrap().num_attrs, expected.len());
    assert_eq!(
        owner
            .attr("first")
            .unwrap()
            .read_scalar::<VarLenUnicode>()
            .unwrap()
            .as_str(),
        "preserved"
    );
    assert_eq!(owner.attr("last").unwrap().read_scalar::<i32>().unwrap(), 3);
    let replacement = owner.attr("replace").unwrap();
    assert_eq!(
        replacement.dtype().unwrap().is_committed(),
        after == AttributeForm::Committed
    );
    if after == AttributeForm::VariableLength {
        assert_eq!(
            replacement.read_scalar::<VarLenUnicode>().unwrap().as_str(),
            "value-42"
        );
    } else {
        assert_eq!(replacement.read_scalar::<i32>().unwrap(), 42);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttributeForm {
    Committed,
    Inline,
    VariableLength,
}

impl AttributeForm {
    fn value(self, value: i32) -> AttrValue {
        match self {
            Self::Committed | Self::Inline => AttrValue::I32(value),
            Self::VariableLength => AttrValue::VarLenString(format!("value-{value}")),
        }
    }
}

#[rstest]
#[case::no_fillers(0)]
#[case::ten_fillers(10)]
fn rust_reads_the_attribute_updated_by_c(#[case] fillers: i32) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c-replacement.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        let attribute = file.new_attr::<i32>().create("replace").unwrap();
        attribute.write_scalar(&7).unwrap();
        for index in 0..fillers {
            file.new_attr::<i32>()
                .create(format!("filler{index}").as_str())
                .unwrap()
                .write_scalar(&index)
                .unwrap();
        }
        attribute.write_scalar(&42).unwrap();
        assert_eq!(
            file.loc_info().unwrap().num_attrs,
            usize::try_from(fillers + 1).unwrap()
        );
    }
    let attrs = File::open(&path).unwrap().root().attrs().unwrap();
    assert_eq!(attrs.len(), usize::try_from(fillers + 1).unwrap());
    assert_eq!(attrs["replace"], AttrValue::I32(42));
    for index in 0..fillers {
        assert_eq!(attrs[&format!("filler{index}")], AttrValue::I32(index));
    }
}
