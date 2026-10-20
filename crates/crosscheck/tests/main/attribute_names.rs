#![cfg(feature = "__hdf5-1.10")]

use std::collections::HashMap;
use std::path::Path;

use hdf5::Location;
use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use rstest::rstest;
use test_util_hdf5::file;

#[rstest]
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
    if c_writer {
        let file = file::libhdf5_create_v18(&path);
        let group = file.create_group("group").unwrap();
        let dataset = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
        let owners: [&Location; 3] = [&file, &group, &dataset];
        for owner in owners {
            owner
                .new_attr::<i32>()
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
            assert_eq!(owner.attr(name).unwrap().read_scalar::<i32>().unwrap(), 42);
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
