#![cfg(feature = "__hdf5-1.10")]

use std::collections::HashMap;
use std::path::Path;

use hdf5::Location;
use hdf5::plist::attribute_create::CharEncoding;
use hdf5::types::VarLenUnicode;
use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure_format::__private::AttributeMessage;
use hdf5_pure_format::__private::Dataspace;
use hdf5_pure_format::__private::DataspaceType;
use hdf5_pure_format::__private::DatatypeLocation;
use hdf5_pure_format::OffsetWidth;
use rstest::rstest;
use test_util_hdf5::file;

#[rstest]
#[case("échelle")]
#[case("温度")]
#[case(".")]
#[case("/")]
#[case("a/b")]
#[case("./a//b/")]
#[case("../name")]
#[case(" space:[]\n")]
#[case(&"x".repeat(usize::from(u16::MAX) - 1))]
fn attribute_name_punctuation_whitespace_and_maximum_length_round_trip(
    #[case] name: &str,
    #[values(false, true)] dense: bool,
    #[values(false, true)] c_writer: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("names.h5");
    let encoding = if name.is_ascii() {
        CharEncoding::Ascii
    } else {
        CharEncoding::Utf8
    };
    if c_writer {
        let file = file::libhdf5_create_v18(&path);
        let group = file.create_group("group").unwrap();
        let dataset = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
        let owners: [&Location; 3] = [&file, &group, &dataset];
        for owner in owners {
            owner
                .new_attr::<i32>()
                .char_encoding(encoding)
                .create(name)
                .unwrap()
                .write_scalar(&7)
                .unwrap();
            if dense {
                for index in 0..10 {
                    owner
                        .new_attr::<i32>()
                        .create(format!("filler{index}").as_str())
                        .unwrap()
                        .write_scalar(&index)
                        .unwrap();
                }
            }
            owner.attr(name).unwrap().write_scalar(&42).unwrap();
        }
    } else {
        write_pure(&path, name, dense);
    }
    let mut names = vec![name.to_string()];
    if dense {
        names.extend((0..10).map(|index| format!("filler{index}")));
    }
    names.sort();
    {
        let c = hdf5::File::open(&path).unwrap();
        let group = c.group("group").unwrap();
        let dataset = c.dataset("data").unwrap();
        let owners: [&Location; 3] = [&c, &group, &dataset];
        for owner in owners {
            let mut actual = owner.attr_names().unwrap();
            actual.sort();
            assert_eq!(actual, names);
            assert_eq!(owner.loc_info().unwrap().num_attrs, names.len());
            let attribute = owner.attr(name).unwrap();
            assert_eq!(attribute.read_scalar::<i32>().unwrap(), 42);
            assert_eq!(attribute.acpl().unwrap().char_encoding(), encoding);
        }
    }
    let file = File::open(&path).unwrap();
    let expected: HashMap<_, _> = names
        .iter()
        .map(|attr_name| {
            let value = if attr_name == name {
                42
            } else {
                attr_name
                    .strip_prefix("filler")
                    .unwrap()
                    .parse::<i32>()
                    .unwrap()
            };
            (attr_name.clone(), AttrValue::I32(value))
        })
        .collect();
    assert_eq!(
        [
            file.root().attrs().unwrap(),
            file.group("group").unwrap().attrs().unwrap(),
            file.dataset("data").unwrap().attrs().unwrap()
        ],
        [expected.clone(), expected.clone(), expected]
    );
}

#[rstest]
#[case::inline_ascii("scale", AttrValue::I32(42), false)]
#[case::inline_utf8("échelle", AttrValue::I32(42), false)]
#[case::committed_ascii("scale", AttrValue::I32(42), true)]
#[case::committed_utf8("échelle", AttrValue::I32(42), true)]
#[case::variable_length_ascii("scale", AttrValue::VarLenString("value-42".into()), false)]
#[case::variable_length_utf8("échelle", AttrValue::VarLenString("value-42".into()), false)]
fn staged_attribute_names_declare_encoding_independently_of_value_type(
    #[case] name: &str,
    #[case] value: AttrValue,
    #[case] committed: bool,
    #[values(false, true)] dense: bool,
    #[values(false, true)] write: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encoding.h5");
    let mut builder = FileBuilder::new();
    builder.commit_datatype("integer_type", hdf5_pure::make_i32_type());
    stage_encoded_attributes(
        |name, value, committed| {
            if committed {
                builder.set_attr_committed(name, value, "integer_type");
            } else {
                builder.set_attr(name, value);
            }
        },
        name,
        &value,
        committed,
        dense,
    );
    let dataset = builder.create_dataset("data").with_i32_data(&[1]);
    stage_encoded_attributes(
        |name, value, committed| {
            if committed {
                dataset.set_attr_committed(name, value, "integer_type");
            } else {
                dataset.set_attr(name, value);
            }
        },
        name,
        &value,
        committed,
        dense,
    );
    let mut group = builder.create_group("group");
    stage_encoded_attributes(
        |name, value, committed| {
            if committed {
                group.set_attr_committed(name, value, "integer_type");
            } else {
                group.set_attr(name, value);
            }
        },
        name,
        &value,
        committed,
        dense,
    );
    builder.add_group(group.finish());
    if write {
        builder.write(&path).unwrap();
    } else {
        std::fs::write(&path, builder.finish().unwrap()).unwrap();
    }
    {
        let file = hdf5::File::open(&path).unwrap();
        let group = file.group("group").unwrap();
        let dataset = file.dataset("data").unwrap();
        let owners: [&Location; 3] = [&file, &group, &dataset];
        let mut expected = vec![name.to_string()];
        if dense {
            expected.extend((0..10).map(|index| format!("filler{index}")));
        }
        expected.sort();
        for owner in owners {
            let mut names = owner.attr_names().unwrap();
            names.sort();
            assert_eq!(names, expected);
            assert_eq!(owner.loc_info().unwrap().num_attrs, expected.len());
            let attribute = owner.attr(name).unwrap();
            assert_eq!(
                attribute.acpl().unwrap().char_encoding(),
                if name.is_ascii() {
                    CharEncoding::Ascii
                } else {
                    CharEncoding::Utf8
                }
            );
            match &value {
                AttrValue::I32(value) => {
                    assert_eq!(attribute.read_scalar::<i32>().unwrap(), *value);
                    assert_eq!(attribute.dtype().unwrap().is_committed(), committed);
                }
                AttrValue::VarLenString(value) => assert_eq!(
                    attribute.read_scalar::<VarLenUnicode>().unwrap().as_str(),
                    value
                ),
                _ => panic!("unexpected attribute value {value:?}"),
            }
            if dense {
                assert_eq!(
                    owner.attr("filler0").unwrap().read_scalar::<i32>().unwrap(),
                    0
                );
                assert_eq!(
                    owner.attr("filler9").unwrap().read_scalar::<i32>().unwrap(),
                    9
                );
            }
        }
    }
    let file = File::open(&path).unwrap();
    for attributes in [
        file.root().attrs().unwrap(),
        file.group("group").unwrap().attrs().unwrap(),
        file.dataset("data").unwrap().attrs().unwrap(),
    ] {
        assert_eq!(attributes.get(name), Some(&value));
    }
}

#[rstest]
#[case("xx", 0, false)]
#[case("é", 1, false)]
#[case("é", 0, true)]
fn c_reads_attributes_at_the_utf8_compact_storage_boundary(
    #[case] name: &str,
    #[case] trim: usize,
    #[case] dense: bool,
) {
    let probe = AttributeMessage {
        name: "xx".into(),
        datatype: hdf5_pure::make_i8_type(),
        dataspace: Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![1],
            max_dimensions: None,
        },
        raw_data: vec![7],
        datatype_location: DatatypeLocation::Inline,
    };
    let overhead = probe.serialize(OffsetWidth::Eight, 8).unwrap().len() - 1;
    let values = vec![7; usize::from(u16::MAX) - overhead - trim];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("boundary.h5");
    let mut builder = FileBuilder::new();
    builder.set_attr(name, AttrValue::I8Array(values.clone()));
    builder.write(&path).unwrap();
    assert_eq!(
        test_util::fractal_heap::has_fractal_heap(&std::fs::read(&path).unwrap()),
        dense
    );
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(file.attr_names().unwrap(), [name]);
    assert_eq!(file.loc_info().unwrap().num_attrs, 1);
    let attribute = file.attr(name).unwrap();
    assert_eq!(attribute.read_raw::<i8>().unwrap(), values);
    assert_eq!(
        attribute.acpl().unwrap().char_encoding(),
        if name.is_ascii() {
            CharEncoding::Ascii
        } else {
            CharEncoding::Utf8
        }
    );
}

#[rstest]
#[case("scale", AttrValue::I32(99))]
#[case("échelle", AttrValue::I32(99))]
#[case("scale", AttrValue::VarLenString("new".into()))]
#[case("échelle", AttrValue::VarLenString("new".into()))]
fn c_reads_attribute_name_encoding_after_compact_edits(
    #[case] name: &str,
    #[case] value: AttrValue,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edited.h5");
    write_pure(&path, name, false);
    {
        let file = File::open_rw(&path).unwrap();
        file.root().set_attr(name, value.clone()).unwrap();
        file.group("group")
            .unwrap()
            .set_attr(name, value.clone())
            .unwrap();
        file.dataset("data")
            .unwrap()
            .set_attr(name, value.clone())
            .unwrap();
        file.commit().unwrap();
    }
    let file = hdf5::File::open(&path).unwrap();
    let group = file.group("group").unwrap();
    let dataset = file.dataset("data").unwrap();
    let owners: [&Location; 3] = [&file, &group, &dataset];
    for owner in owners {
        assert_eq!(owner.attr_names().unwrap(), [name]);
        assert_eq!(owner.loc_info().unwrap().num_attrs, 1);
        let attribute = owner.attr(name).unwrap();
        assert_eq!(
            attribute.acpl().unwrap().char_encoding(),
            if name.is_ascii() {
                CharEncoding::Ascii
            } else {
                CharEncoding::Utf8
            }
        );
        match &value {
            AttrValue::I32(value) => assert_eq!(attribute.read_scalar::<i32>().unwrap(), *value),
            AttrValue::VarLenString(value) => assert_eq!(
                attribute.read_scalar::<VarLenUnicode>().unwrap().as_str(),
                value
            ),
            _ => panic!("unexpected attribute value {value:?}"),
        }
    }
}

fn stage_encoded_attributes(
    mut set: impl FnMut(&str, AttrValue, bool),
    name: &str,
    value: &AttrValue,
    committed: bool,
    dense: bool,
) {
    set(name, AttrValue::I32(7), false);
    if dense {
        for index in 0..10 {
            set(&format!("filler{index}"), AttrValue::I32(index), false);
        }
    }
    set(name, value.clone(), committed);
}

fn write_pure(path: &Path, name: &str, dense: bool) {
    let mut builder = FileBuilder::new();
    stage_attributes(|name, value| builder.set_attr(name, value), name, dense);
    let dataset = builder.create_dataset("data").with_i32_data(&[1]);
    stage_attributes(
        |name, value| {
            dataset.set_attr(name, value);
        },
        name,
        dense,
    );
    let mut group = builder.create_group("group");
    stage_attributes(|name, value| group.set_attr(name, value), name, dense);
    builder.add_group(group.finish());
    builder.write(path).unwrap();
}

fn stage_attributes(mut set: impl FnMut(&str, AttrValue), name: &str, dense: bool) {
    set(name, AttrValue::I32(7));
    if dense {
        for index in 0..10 {
            set(&format!("filler{index}"), AttrValue::I32(index));
        }
    }
    set(name, AttrValue::I32(42));
}
