use std::collections::HashMap;

use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use rstest::rstest;
use test_util::fractal_heap;

#[rstest]
#[case("", None)]
#[case("prefix\0suffix", None)]
#[case(&"x".repeat(usize::from(u16::MAX)), Some(usize::from(u16::MAX) + 1))]
fn finish_rejects_invalid_attribute_names(
    #[case] name: &str,
    #[case] encoded_size: Option<usize>,
    #[values(Owner::Dataset, Owner::Group, Owner::Root)] owner: Owner,
    #[values(false, true)] dense: bool,
    #[values(Output::ExistingFile, Output::Finish, Output::NewFile, Output::Sink)] output: Output,
) {
    let builder = named_attribute_builder(owner, name, dense);
    let error = match output {
        Output::Finish => builder.finish().unwrap_err(),
        Output::Sink => {
            let mut bytes = b"previous bytes".to_vec();
            let error = builder.finish_to(&mut bytes).unwrap_err();
            assert_eq!(bytes, b"previous bytes");
            error
        }
        Output::ExistingFile | Output::NewFile => {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("attributes.h5");
            if matches!(output, Output::ExistingFile) {
                std::fs::write(&path, b"previous file").unwrap();
            }
            let error = builder.write(&path).unwrap_err();
            if matches!(output, Output::ExistingFile) {
                assert_eq!(std::fs::read(&path).unwrap(), b"previous file");
            } else {
                assert!(!path.exists());
            }
            error
        }
    };
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, invalid_attribute_name_error(name, encoded_size));
}

#[rstest]
#[case::empty("", None)]
#[case::embedded_null("prefix\0suffix", None)]
#[case::oversized(&"x".repeat(usize::from(u16::MAX)), Some(usize::from(u16::MAX) + 1))]
fn committed_attribute_edits_reject_invalid_names_without_changing_the_file(
    #[case] name: &str,
    #[case] encoded_size: Option<usize>,
    #[values(Owner::Dataset, Owner::Group, Owner::Root)] owner: Owner,
    #[values(false, true)] dense: bool,
    #[values(AttrValue::I32(42), AttrValue::VarLenString("value".into()))] value: AttrValue,
) {
    let path = test_util::temp::temp_path("attributes.h5");
    named_attribute_builder(owner, "valid", dense)
        .write(&path)
        .unwrap();
    let before = std::fs::read(&path).unwrap();
    assert_eq!(fractal_heap::has_fractal_heap(&before), dense);
    let error = {
        let file = File::open_rw(&path).unwrap();
        match owner {
            Owner::Dataset => file.dataset("data").unwrap().set_attr(name, value).unwrap(),
            Owner::Group => file.group("group").unwrap().set_attr(name, value).unwrap(),
            Owner::Root => file.root().set_attr(name, value).unwrap(),
        }
        file.commit().unwrap_err()
    };
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, invalid_attribute_name_error(name, encoded_size));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    let attrs = match owner {
        Owner::Dataset => file.dataset("data").unwrap().attrs().unwrap(),
        Owner::Group => file.group("group").unwrap().attrs().unwrap(),
        Owner::Root => file.root().attrs().unwrap(),
    };
    let mut expected = HashMap::from([
        ("preserved".into(), AttrValue::I32(1)),
        ("valid".into(), AttrValue::I32(42)),
    ]);
    if dense {
        for index in 0..10 {
            expected.insert(format!("filler{index}"), AttrValue::I32(index));
        }
    }
    assert_eq!(attrs, expected);
}

#[test]
fn maximum_length_attribute_name_round_trips() {
    let name = "x".repeat(usize::from(u16::MAX) - 1);
    let file = File::from_bytes(
        named_attribute_builder(Owner::Root, &name, false)
            .finish()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(file.root().attrs().unwrap()[&name], AttrValue::I32(42));
}

fn named_attribute_builder(owner: Owner, name: &str, dense: bool) -> FileBuilder {
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("group");
    match owner {
        Owner::Dataset => {
            let dataset = builder.create_dataset("data").with_i32_data(&[1]);
            stage_attributes(
                |name, value| {
                    dataset.set_attr(name, value);
                },
                name,
                dense,
            );
        }
        Owner::Group => stage_attributes(|name, value| group.set_attr(name, value), name, dense),
        Owner::Root => stage_attributes(|name, value| builder.set_attr(name, value), name, dense),
    }
    builder.add_group(group.finish());
    builder
}

fn stage_attributes(mut set: impl FnMut(&str, AttrValue), name: &str, dense: bool) {
    set("preserved", AttrValue::I32(1));
    if dense {
        for index in 0..10 {
            set(&format!("filler{index}"), AttrValue::I32(index));
        }
    }
    set(name, AttrValue::I32(7));
    set(name, AttrValue::I32(42));
}

fn invalid_attribute_name_error(name: &str, encoded_size: Option<usize>) -> FormatError {
    match encoded_size {
        Some(size) => FormatError::AttributeFieldTooLong {
            name: name.into(),
            field: "name",
            size,
            limit: usize::from(u16::MAX),
        },
        None => FormatError::InvalidAttributeName { name: name.into() },
    }
}

#[derive(Clone, Copy)]
enum Owner {
    Dataset,
    Group,
    Root,
}

#[derive(Clone, Copy)]
enum Output {
    ExistingFile,
    Finish,
    NewFile,
    Sink,
}
