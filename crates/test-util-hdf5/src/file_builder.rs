//! File-builder fixtures reopened through `hdf5-pure`'s reader.

use std::path::Path;

use hdf5_pure::{AttrValue, File, FileBuilder};

/// Builds a file with `build`, serializes it, and opens the serialized bytes through the reader.
#[track_caller]
pub fn build_and_open(build: impl FnOnce(&mut FileBuilder)) -> File {
    let mut builder = FileBuilder::new();
    build(&mut builder);
    let bytes = builder
        .finish()
        .unwrap_or_else(|e| panic!("finish test file: {e}"));
    File::from_bytes(bytes).unwrap_or_else(|e| panic!("open serialized test file: {e}"))
}

/// Sets `per_object` variable-length string attributes `a0`, `a1`, and so on, of `strings_each`
/// strings on the root, on group `g` and on dataset `g/d`, and returns every value set, keyed by
/// object path and attribute name.
///
/// The strings are `s0`, `s1`, and so on, distinct across the file.
pub fn set_string_attrs(
    builder: &mut FileBuilder,
    per_object: usize,
    strings_each: usize,
) -> Vec<(&'static str, String, Vec<String>)> {
    let mut expected = Vec::new();
    let mut next = 0;
    let mut values = || {
        let values: Vec<String> = (next..next + strings_each)
            .map(|i| format!("s{i}"))
            .collect();
        next += strings_each;
        values
    };
    for i in 0..per_object {
        let value = values();
        builder.set_attr(
            &format!("a{i}"),
            AttrValue::VarLenStringArray(value.clone()),
        );
        expected.push(("/", format!("a{i}"), value));
    }
    let mut group = builder.create_group("g");
    for i in 0..per_object {
        let value = values();
        group.set_attr(
            &format!("a{i}"),
            AttrValue::VarLenStringArray(value.clone()),
        );
        expected.push(("g", format!("a{i}"), value));
    }
    let dataset = group.create_dataset("d");
    dataset.with_f64_data(&[1.0]);
    for i in 0..per_object {
        let value = values();
        dataset.set_attr(
            &format!("a{i}"),
            AttrValue::VarLenStringArray(value.clone()),
        );
        expected.push(("g/d", format!("a{i}"), value));
    }
    builder.add_group(group.finish());
    expected
}

/// Writes a one-dimensional `i32` dataset to `path`.
#[track_caller]
pub fn write_i32_dataset(path: &Path, dataset_name: &str, data: &[i32]) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset(dataset_name)
        .with_i32_data(data)
        .with_shape(&[data.len() as u64]);
    builder
        .write(path)
        .unwrap_or_else(|e| panic!("write test file {path:?}: {e}"));
}
