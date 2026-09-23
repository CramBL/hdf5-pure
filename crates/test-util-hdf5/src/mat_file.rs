//! MAT v7.3 files, read through `hdf5-pure`'s reader and assembled through its writer: the
//! attributes MATLAB reads a variable's class from, and the object tree a file holds.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use hdf5_pure::{AttrValue, DatasetBuilder, File, FileBuilder};

/// Opens the MAT file at `path`, first checking that it carries the MATLAB v7.3 userblock.
///
/// `File::open` skips the userblock without reading it, so a file that lost its MATLAB
/// signature still opens here, and MATLAB rejects it.
#[track_caller]
pub fn open(path: &Path) -> File {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    hdf5_pure::mat::userblock::verify_header(&bytes)
        .unwrap_or_else(|e| panic!("{path:?} carries no MATLAB v7.3 userblock: {e}"));
    File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"))
}

/// The `MATLAB_class` attribute of the dataset at `path`, or `None` where it carries none.
#[track_caller]
pub fn class(file: &File, path: &str) -> Option<String> {
    let attrs = file
        .dataset(path)
        .and_then(|dataset| dataset.attrs())
        .unwrap_or_else(|e| panic!("read the attributes of {path:?}: {e}"));
    class_in(&attrs)
}

/// The `MATLAB_class` attribute of the group at `path`, or `None` where it carries none.
#[track_caller]
pub fn group_class(file: &File, path: &str) -> Option<String> {
    let attrs = file
        .group(path)
        .and_then(|group| group.attrs())
        .unwrap_or_else(|e| panic!("read the attributes of {path:?}: {e}"));
    class_in(&attrs)
}

/// Every group and dataset below the root of `file`, each group before its members.
#[track_caller]
pub fn objects(file: &File) -> Vec<Object> {
    let mut found = Vec::new();
    walk(file, "", &mut found);
    found
}

/// One group or dataset in a file, with its attributes.
#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    pub path: String,
    pub kind: ObjectKind,
    pub attrs: BTreeMap<String, AttrValue>,
}

impl Object {
    pub fn string_attr(&self, attr_name: &str) -> Option<&str> {
        self.attrs.get(attr_name).and_then(AttrValue::as_str)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectKind {
    Dataset,
    Group,
}

/// A file holding one MATLAB object: the variable `var`, of class `class_name`, whose metadata
/// is `metadata`, and the MCOS subsystem that stores the object's properties.
///
/// `heap` is every MCOS cell from cell 0, the `FileWrapper__` blob. `aux` are the further
/// `#refs#` datasets a cell reaches by reference but the MCOS array does not list.
#[track_caller]
pub fn object_file(
    var: &str,
    class_name: &str,
    metadata: &[u32],
    heap: &[(&str, Cell)],
    aux: &[(&str, Cell)],
) -> Vec<u8> {
    let mut builder = FileBuilder::new();

    let mut refs = builder.create_group(REFS);
    for (name, cell) in heap.iter().chain(aux) {
        cell.write_to(refs.create_dataset(name));
    }
    builder.add_group(refs.finish());

    let mut subsystem = builder.create_group("#subsystem#");
    let cell_paths: Vec<String> = heap
        .iter()
        .map(|(name, _)| format!("{REFS}/{name}"))
        .collect();
    let cell_paths: Vec<&str> = cell_paths.iter().map(String::as_str).collect();
    let mcos_array = subsystem.create_dataset("MCOS");
    mcos_array
        .with_path_references(&cell_paths)
        .with_shape(&[1, cell_paths.len() as u64]);
    mcos_array.set_attr(CLASS, AttrValue::AsciiString("FileWrapper__".into()));
    mcos_array.set_attr(OBJECT_DECODE, AttrValue::I32(OBJECT_DECODE_OPAQUE));
    builder.add_group(subsystem.finish());

    let variable = builder.create_dataset(var);
    variable
        .with_u32_data(metadata)
        .with_shape(&[1, metadata.len() as u64]);
    variable.set_attr(CLASS, AttrValue::AsciiString(class_name.into()));
    variable.set_attr(OBJECT_DECODE, AttrValue::I32(OBJECT_DECODE_OPAQUE));

    builder
        .finish()
        .unwrap_or_else(|e| panic!("assemble a file holding {var:?}: {e}"))
}

/// The contents of one MCOS cell, or of a further `#refs#` dataset a cell reaches.
#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    /// The `FileWrapper__` metadata blob, which [`test_util::mcos::FileWrapper`] builds.
    Blob(Vec<u8>),
    Char(String),
    ComplexF64(Vec<(f64, f64)>),
    /// A placeholder cell no property reads, as the canonical empty is.
    Empty,
    F64(Vec<f64>),
    /// A cell array of the objects at these `#refs#` paths.
    Refs(Vec<String>),
    U8(Vec<u8>),
}

impl Cell {
    /// Gives `dataset` this cell's values, as a row vector of the class MATLAB stores them in.
    pub fn write_to(&self, dataset: &mut DatasetBuilder) {
        let class = match self {
            Self::Blob(bytes) => {
                dataset
                    .with_u8_data(bytes)
                    .with_shape(&[1, bytes.len() as u64]);
                "uint8"
            }
            Self::Char(text) => {
                let units: Vec<u16> = text.encode_utf16().collect();
                dataset
                    .with_u16_data(&units)
                    .with_shape(&[1, units.len() as u64]);
                "char"
            }
            Self::ComplexF64(values) => {
                dataset
                    .with_complex64_data(values)
                    .with_shape(&[1, values.len() as u64]);
                "double"
            }
            Self::Empty => {
                dataset.with_u8_data(&[0]).with_shape(&[1, 1]);
                "uint8"
            }
            Self::F64(values) => {
                dataset
                    .with_f64_data(values)
                    .with_shape(&[1, values.len() as u64]);
                "double"
            }
            Self::Refs(paths) => {
                let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
                dataset
                    .with_path_references(&paths)
                    .with_shape(&[1, paths.len() as u64]);
                "cell"
            }
            Self::U8(values) => {
                dataset
                    .with_u8_data(values)
                    .with_shape(&[1, values.len() as u64]);
                "uint8"
            }
        };
        dataset.set_attr(CLASS, AttrValue::AsciiString(class.into()));
    }
}

fn class_in(attrs: &HashMap<String, AttrValue>) -> Option<String> {
    attrs
        .get(CLASS)
        .and_then(AttrValue::as_str)
        .map(str::to_owned)
}

#[track_caller]
fn walk(file: &File, path: &str, found: &mut Vec<Object>) {
    let group = if path.is_empty() {
        file.root()
    } else {
        file.group(path)
            .unwrap_or_else(|e| panic!("open the group {path:?}: {e}"))
    };
    let child = |name: &str| {
        if path.is_empty() {
            name.to_owned()
        } else {
            format!("{path}/{name}")
        }
    };
    for name in group
        .datasets()
        .unwrap_or_else(|e| panic!("list the datasets of {path:?}: {e}"))
    {
        let dataset_path = child(&name);
        let attrs = file
            .dataset(&dataset_path)
            .and_then(|dataset| dataset.attrs())
            .unwrap_or_else(|e| panic!("read the attributes of {dataset_path:?}: {e}"));
        found.push(Object {
            path: dataset_path,
            kind: ObjectKind::Dataset,
            attrs: attrs.into_iter().collect(),
        });
    }
    for name in group
        .groups()
        .unwrap_or_else(|e| panic!("list the groups of {path:?}: {e}"))
    {
        let group_path = child(&name);
        let attrs = file
            .group(&group_path)
            .and_then(|group| group.attrs())
            .unwrap_or_else(|e| panic!("read the attributes of {group_path:?}: {e}"));
        found.push(Object {
            path: group_path.clone(),
            kind: ObjectKind::Group,
            attrs: attrs.into_iter().collect(),
        });
        walk(file, &group_path, found);
    }
}

/// The attribute MATLAB reads a variable's class from.
pub const CLASS: &str = "MATLAB_class";

/// The attribute that says how MATLAB decodes a variable that is an object.
const OBJECT_DECODE: &str = "MATLAB_object_decode";

/// The `MATLAB_object_decode` of an object the MCOS subsystem stores.
const OBJECT_DECODE_OPAQUE: i32 = 3;

/// The group MATLAB interns the values a cell array or an object reaches.
const REFS: &str = "#refs#";

#[cfg(test)]
mod tests {
    use hdf5_pure::File;
    use hdf5_pure::mat::{MatBuilder, Options};

    use crate::mat_file::{self, ObjectKind};

    #[test]
    fn the_walk_reaches_a_struct_field_and_reads_its_class() {
        let mut builder = MatBuilder::new(Options::default());
        builder
            .struct_("s", |fields| {
                fields.write_scalar_f64("x", 1.0)?;
                Ok(())
            })
            .unwrap();
        let file = File::from_bytes(builder.finish().unwrap()).unwrap();

        let objects: Vec<(String, ObjectKind)> = mat_file::objects(&file)
            .into_iter()
            .map(|object| (object.path, object.kind))
            .collect();
        assert_eq!(
            objects,
            vec![
                ("s".to_owned(), ObjectKind::Group),
                ("s/x".to_owned(), ObjectKind::Dataset),
            ]
        );
        assert_eq!(mat_file::group_class(&file, "s").as_deref(), Some("struct"));
        assert_eq!(mat_file::class(&file, "s/x").as_deref(), Some("double"));
    }
}
