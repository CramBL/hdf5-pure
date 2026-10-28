//! Native observations of synthetic MATLAB struct-array graphs.
//!
//! libhdf5 reads the object references, symbol-table groups, MATLAB attributes,
//! and contiguous values. The malformed MATLAB case has different field shapes
//! in an otherwise valid HDF5 graph.
//!
//! # Examples
//!
//! ```
//! use test_util::{mat_struct_array, temp};
//! use test_util_hdf5::mat_struct_array::native;
//!
//! let path = temp::temp_path("struct_array.mat");
//! std::fs::write(&path, mat_struct_array::struct_array())?;
//! native::assert_contents(&path);
//! # Ok::<(), std::io::Error>(())
//! ```

use std::path::Path;

use hdf5::Dataset;
use hdf5::Datatype;
use hdf5::File;
use hdf5::GroupStorageType;
use hdf5::H5Type;
use hdf5::Location;
use hdf5::LocationType;
use hdf5::ObjectReference;
use hdf5::ObjectReference1;
use hdf5::ReferencedObject;
use hdf5::dataset::Layout;
use hdf5::datatype::ByteOrder;
use hdf5::types::FixedAscii;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenArray;

/// Checks the complete valid graph through libhdf5.
///
/// `path` must contain the unmodified image from
/// [`test_util::mat_struct_array::struct_array`]. The assertions check exact
/// group and dataset inventories, reference target identities and order,
/// nested aliases and link counts, attributes, datatypes, shapes, and values.
/// Groups use symbol tables, and datasets use contiguous unfiltered storage.
/// The native file is closed before returning.
///
/// # Panics
///
/// Panics if opening, reading, or closing fails, or an observation differs from
/// the expected image.
pub fn assert_contents(path: &Path) {
    let file = crate::file::libhdf5_open(path);
    assert_eq!(file.userblock(), USERBLOCK_SIZE);
    let image = Observation { file: &file };
    let no_datasets: &[&str] = &[];
    image.group(
        "/",
        &["#refs#", "col", "grid", "nested", "row", "scalar"],
        no_datasets,
        &[],
    );
    image.group(
        "#refs#",
        &["r51", "r53"],
        &(0..54)
            .filter(|i| *i != 51 && *i != 53)
            .map(|i| format!("r{i}"))
            .collect::<Vec<_>>(),
        &[],
    );
    for (path, shape, start) in [("row", [6, 1], 0), ("col", [1, 6], 18)] {
        image.group(
            path,
            &[],
            &["fieldA", "fieldB", "fieldC"],
            &["fieldA", "fieldB", "fieldC"],
        );
        for (field, offset) in [("fieldA", 0), ("fieldB", 6), ("fieldC", 12)] {
            image.references(
                &format!("{path}/{field}"),
                shape,
                &(start + offset..start + offset + 6).collect::<Vec<_>>(),
            );
        }
        for i in 0..6 {
            image.numeric(
                &format!("#refs#/r{}", start + i),
                &[1, 1],
                &[f64::from(i + 1)],
            );
            image.character(
                &format!("#refs#/r{}", start + 6 + i),
                u16::try_from(97 + i).unwrap(),
            );
            image.numeric(
                &format!("#refs#/r{}", start + 12 + i),
                &[6, 1],
                &[-6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
            );
        }
    }
    image.group("grid", &[], &["id", "tag"], &["id", "tag"]);
    image.references("grid/id", [3, 2], &[36, 39, 37, 40, 38, 41]);
    image.references("grid/tag", [3, 2], &[42, 45, 43, 46, 44, 47]);
    for i in 0..6 {
        image.numeric(
            &format!("#refs#/r{}", 36 + i),
            &[1, 1],
            &[f64::from(i / 3 * 10 + i % 3)],
        );
        image.character(
            &format!("#refs#/r{}", 42 + i),
            u16::try_from(97 + i).unwrap(),
        );
    }
    image.group("nested", &[], &["fieldA", "inner"], &["fieldA", "inner"]);
    image.references("nested/fieldA", [2, 1], &[48, 49]);
    image.references("nested/inner", [2, 1], &[51, 53]);
    image.numeric("#refs#/r48", &[1, 1], &[1.0]);
    image.numeric("#refs#/r49", &[1, 1], &[2.0]);
    for (group, target, value) in [(51, 50, 0.0), (53, 52, 100.0)] {
        image.group(&format!("#refs#/r{group}"), &[], &["p"], &["p"]);
        let target = format!("#refs#/r{target}");
        let alias = format!("#refs#/r{group}/p");
        assert_eq!(
            file.loc_info_by_name(&target).unwrap().token,
            file.loc_info_by_name(&alias).unwrap().token
        );
        assert_eq!(file.loc_info_by_name(&target).unwrap().num_links, 2);
        image.numeric(&target, &[1, 1], &[value]);
        image.numeric(&alias, &[1, 1], &[value]);
    }
    image.group(
        "scalar",
        &[],
        &["fieldA", "fieldB", "fieldC"],
        &["fieldA", "fieldB", "fieldC"],
    );
    image.numeric("scalar/fieldA", &[1, 1], &[1.0]);
    image.character("scalar/fieldB", 97);
    image.numeric(
        "scalar/fieldC",
        &[6, 1],
        &[-6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
    );
    file.close().unwrap();
}

/// Checks both equal-count fields and their different shapes through libhdf5.
///
/// `path` must contain the unmodified image from
/// [`test_util::mat_struct_array::mismatched_struct_array`]. The assertions check
/// group contents, MATLAB attributes, reference order and identities, numeric
/// values, datatypes, and contiguous storage. Both fields contain six references,
/// with HDF5 dimensions `[3, 2]` and `[2, 3]`. The native file is closed before
/// returning. This checks HDF5 readability without MATLAB deserialization.
///
/// # Panics
///
/// Panics if opening, reading, or closing fails, or an observation differs from
/// the expected image.
pub fn assert_mismatched_contents(path: &Path) {
    let file = crate::file::libhdf5_open(path);
    assert_eq!(file.userblock(), USERBLOCK_SIZE);
    let image = Observation { file: &file };
    let no_datasets: &[&str] = &[];
    image.group("/", &["#refs#", "mismatched"], no_datasets, &[]);
    image.group(
        "#refs#",
        &[],
        &(0..12).map(|i| format!("r{i}")).collect::<Vec<_>>(),
        &[],
    );
    image.group("mismatched", &[], &["a", "b"], &["a", "b"]);
    image.references("mismatched/a", [3, 2], &[0, 3, 1, 4, 2, 5]);
    image.references("mismatched/b", [2, 3], &[6, 8, 10, 7, 9, 11]);
    for i in 0..12 {
        image.numeric(&format!("#refs#/r{i}"), &[1, 1], &[f64::from(i % 6)]);
    }
    file.close().unwrap();
}

/// A libhdf5 file for group, reference, and value observations.
struct Observation<'a> {
    file: &'a File,
}

impl Observation<'_> {
    /// Checks direct members and the MATLAB structure attributes named by `fields`.
    fn group(&self, path: &str, groups: &[&str], datasets: &[impl AsRef<str>], fields: &[&str]) {
        let group = self.file.group(path).unwrap();
        assert_eq!(
            group.info().unwrap().storage_type,
            GroupStorageType::SymbolTable
        );
        let mut names = group.member_names().unwrap();
        names.sort();
        let mut expected = groups
            .iter()
            .copied()
            .chain(datasets.iter().map(AsRef::as_ref))
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(names, expected);
        let mut tokens = Vec::new();
        for name in names {
            let info = group.loc_info_by_name(&name).unwrap();
            assert!(!tokens.contains(&info.token));
            tokens.push(info.token);
            assert_eq!(
                info.loc_type,
                if groups.contains(&name.as_str()) {
                    LocationType::Group
                } else {
                    LocationType::Dataset
                }
            );
        }
        if fields.is_empty() {
            assert!(group.attr_names().unwrap().is_empty());
        } else {
            assert_names(&group, &["MATLAB_class", "MATLAB_fields"]);
            assert_class(&group, "struct");
            let attr = group.attr("MATLAB_fields").unwrap();
            assert_eq!(attr.shape(), [fields.len()]);
            assert_eq!(
                attr.dtype().unwrap(),
                Datatype::from_type::<VarLenArray<FixedAscii<1>>>().unwrap()
            );
            let values = attr.read_raw::<VarLenArray<FixedAscii<1>>>().unwrap();
            assert_eq!(
                values
                    .iter()
                    .map(|chars| chars.iter().map(FixedAscii::as_str).collect::<String>())
                    .collect::<Vec<_>>(),
                fields
            );
        }
    }

    /// Dereferences each stored reference and compares its target's native identity and type.
    fn references(&self, path: &str, shape: [usize; 2], targets: &[u32]) {
        let dataset = self.file.dataset(path).unwrap();
        assert_eq!(dataset.shape(), shape);
        assert_eq!(
            dataset.dtype().unwrap(),
            Datatype::from_type::<ObjectReference1>().unwrap()
        );
        assert!(dataset.attr_names().unwrap().is_empty());
        assert_storage(&dataset);
        let refs = dataset.read_raw::<ObjectReference1>().unwrap();
        assert_eq!(refs.len(), targets.len());
        for (reference, target) in refs.into_iter().zip(targets) {
            let object = reference.dereference(self.file).unwrap();
            let info = match object {
                ReferencedObject::Group(group) => group.loc_info().unwrap(),
                ReferencedObject::Dataset(dataset) => dataset.loc_info().unwrap(),
                other => panic!("unexpected reference target {other:?}"),
            };
            let expected = self
                .file
                .loc_info_by_name(&format!("#refs#/r{target}"))
                .unwrap();
            assert_eq!(info.token, expected.token);
            assert_eq!(info.loc_type, expected.loc_type);
        }
    }

    fn numeric(&self, path: &str, shape: &[usize], values: &[f64]) {
        let dataset = self.file.dataset(path).unwrap();
        assert_eq!(dataset.shape(), shape);
        assert_numeric::<f64>(&dataset.dtype().unwrap());
        assert_eq!(dataset.read_raw::<f64>().unwrap(), values);
        assert_storage(&dataset);
        assert_names(&dataset, &["MATLAB_class"]);
        assert_class(&dataset, "double");
    }

    fn character(&self, path: &str, value: u16) {
        let dataset = self.file.dataset(path).unwrap();
        assert_eq!(dataset.shape(), [1, 1]);
        assert_numeric::<u16>(&dataset.dtype().unwrap());
        assert_eq!(dataset.read_raw::<u16>().unwrap(), [value]);
        assert_storage(&dataset);
        assert_names(&dataset, &["MATLAB_class", "MATLAB_int_decode"]);
        assert_class(&dataset, "char");
        let hint = dataset.attr("MATLAB_int_decode").unwrap();
        assert!(hint.space().unwrap().is_scalar());
        assert_numeric::<i32>(&hint.dtype().unwrap());
        assert_eq!(hint.read_scalar::<i32>().unwrap(), 4);
    }
}

fn assert_class(owner: &Location, class: &str) {
    let attr = owner.attr("MATLAB_class").unwrap();
    assert!(attr.space().unwrap().is_scalar());
    assert_eq!(
        attr.dtype().unwrap(),
        Datatype::from_descriptor(&TypeDescriptor::FixedAscii(class.len())).unwrap()
    );
    assert_eq!(
        attr.read_scalar::<FixedAscii<16>>().unwrap().as_str(),
        class
    );
}

fn assert_names(owner: &Location, expected: &[&str]) {
    let mut names = owner.attr_names().unwrap();
    names.sort();
    assert_eq!(names, expected);
}

fn assert_storage(dataset: &Dataset) {
    assert_eq!(dataset.layout(), Layout::Contiguous);
    assert_eq!(dataset.chunk(), None);
    assert!(dataset.filters().is_empty());
}

fn assert_numeric<T: H5Type>(datatype: &Datatype) {
    assert_eq!(datatype.to_descriptor().unwrap(), T::type_descriptor());
    assert_eq!(datatype.byte_order(), ByteOrder::LittleEndian);
}

// Section `subsec_fmt4_boot_super`, version 4.0.
const USERBLOCK_SIZE: u64 = 512;
