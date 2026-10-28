//! Complete value and storage observations for independent MATLAB struct-array images.
//!
//! The assertions check MATLAB attributes, column-major object references,
//! nested aliases, symbol-table links, and contiguous value datasets. Ordinary
//! assertions inspect headers and storage in the supplied image bytes alongside
//! the reader API.
//! With the `hdf5` feature, native assertions read the graphs through libhdf5.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::File;
//! use test_util::mat_struct_array::struct_array;
//! use test_util_hdf5::mat_struct_array;
//!
//! let bytes = struct_array();
//! let file = File::from_bytes(bytes.clone())?;
//! mat_struct_array::assert_contents(&file, &bytes);
//! file.close()?;
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use hdf5_pure::AttrValue;
use hdf5_pure::CharacterSet;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure::ReferenceType;
use hdf5_pure::StoredAddress;
use hdf5_pure::StringPadding;
use hdf5_pure::SuperblockVersion;
use hdf5_pure::VariableLengthType;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::AttributeMessage;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::ObjectHeader;

use crate::symbol_table;

#[cfg(feature = "hdf5")]
pub mod native;

/// Checks every structure, reference target, attribute, and value in the valid image.
///
/// `file` and `bytes` must describe the same unmodified image from
/// [`test_util::mat_struct_array::struct_array`]. The assertions include the
/// zero-filled user block, version 0 superblock, symbol-table groups,
/// column-major reference order, and nested hard-link identities.
///
/// # Panics
///
/// Panics if an observation differs from the expected image or reading fails.
pub fn assert_contents(file: &File, bytes: &[u8]) {
    let image = Observation::new(file, bytes);
    let no_datasets: &[&str] = &[];
    image.group(
        "",
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
        assert_eq!(image.address(&target), image.address(&alias));
        assert_eq!(image.header(&target).reference_count, Some(2));
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
}

/// Checks both equal-count fields and their distinct shapes in the malformed image.
///
/// `file` and `bytes` must describe the same unmodified image from
/// [`test_util::mat_struct_array::mismatched_struct_array`]. The assertions check
/// HDF5 storage, MATLAB attributes, reference order, and all numeric targets.
/// They do not deserialize the image as a MATLAB struct array.
///
/// # Panics
///
/// Panics if an observation differs from the expected image or reading fails.
pub fn assert_mismatched_contents(file: &File, bytes: &[u8]) {
    let image = Observation::new(file, bytes);
    let no_datasets: &[&str] = &[];
    image.group("", &["#refs#", "mismatched"], no_datasets, &[]);
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
}

/// A reader and its complete image for link and header inspection.
struct Observation<'a> {
    file: &'a File,
    bytes: &'a [u8],
    symbols: symbol_table::Fixture<'a>,
}

impl<'a> Observation<'a> {
    fn new(file: &'a File, bytes: &'a [u8]) -> Self {
        assert_eq!(file.superblock().version(), SuperblockVersion::Zero);
        assert_eq!(file.superblock().base_address.get(), 512);
        assert_eq!(&bytes[..512], &[0; 512]);
        Self {
            file,
            bytes,
            symbols: symbol_table::Fixture::new(file, bytes),
        }
    }

    /// Checks direct links and MATLAB structure attributes named by `fields`.
    fn group(&self, path: &str, groups: &[&str], datasets: &[impl AsRef<str>], fields: &[&str]) {
        let group = self.file.group(path).unwrap();
        let mut actual = group.groups().unwrap();
        actual.sort();
        assert_eq!(actual, groups, "{path:?}");
        let mut expected = datasets.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        expected.sort();
        let mut actual = group.datasets().unwrap();
        actual.sort();
        assert_eq!(actual, expected, "{path:?}");
        assert!(group.named_datatypes().unwrap().is_empty());
        let attrs = if fields.is_empty() {
            HashMap::new()
        } else {
            HashMap::from([
                (
                    "MATLAB_class".into(),
                    AttrValue::AsciiString("struct".into()),
                ),
                (
                    "MATLAB_fields".into(),
                    AttrValue::VarLenAsciiCharArray(
                        fields.iter().map(|field| (*field).into()).collect(),
                    ),
                ),
            ])
        };
        assert_eq!(group.attrs().unwrap(), attrs);
        self.attributes(path, &attrs);
        let mut links = groups.iter().copied().chain(expected).collect::<Vec<_>>();
        links.sort();
        let entries = self.symbols.links(self.address(path));
        assert_eq!(
            entries
                .iter()
                .map(|(_, address)| *address)
                .collect::<BTreeSet<_>>()
                .len(),
            entries.len()
        );
        assert_eq!(
            entries
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            links
        );
    }

    fn references(&self, path: &str, shape: [u64; 2], targets: &[u32]) {
        let dataset = self.file.dataset(path).unwrap();
        assert_eq!(dataset.shape().unwrap(), shape);
        assert_eq!(
            dataset.datatype().unwrap(),
            Datatype::Reference {
                size: 8,
                ref_type: ReferenceType::Object
            }
        );
        assert!(dataset.attrs().unwrap().is_empty());
        self.attributes(path, &HashMap::new());
        let expected = targets
            .iter()
            .map(|i| self.address(&format!("#refs#/r{i}")).get())
            .collect::<Vec<_>>();
        assert_eq!(
            dataset.read_raw().unwrap(),
            expected
                .iter()
                .flat_map(|address| address.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(dataset.dereference().unwrap().len(), targets.len());
        self.contiguous(path, targets.len() as u64 * 8);
    }

    fn numeric(&self, path: &str, shape: &[u64], values: &[f64]) {
        let dataset = self.file.dataset(path).unwrap();
        assert_eq!(dataset.shape().unwrap(), shape);
        assert_eq!(
            dataset.datatype().unwrap(),
            Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
                DatatypeByteOrder::LittleEndian
            ))
        );
        assert_eq!(dataset.read_f64().unwrap(), values);
        let attrs = HashMap::from([(
            "MATLAB_class".into(),
            AttrValue::AsciiString("double".into()),
        )]);
        assert_eq!(dataset.attrs().unwrap(), attrs);
        self.attributes(path, &attrs);
        self.contiguous(path, values.len() as u64 * 8);
    }

    fn character(&self, path: &str, value: u16) {
        let dataset = self.file.dataset(path).unwrap();
        assert_eq!(dataset.shape().unwrap(), [1, 1]);
        assert_eq!(
            dataset.datatype().unwrap(),
            Datatype::FixedPoint(
                FixedPointType::new(2, DatatypeByteOrder::LittleEndian, false, 0, 16).unwrap()
            )
        );
        assert_eq!(dataset.read_u16().unwrap(), [value]);
        let attrs = HashMap::from([
            ("MATLAB_class".into(), AttrValue::AsciiString("char".into())),
            ("MATLAB_int_decode".into(), AttrValue::I32(4)),
        ]);
        assert_eq!(dataset.attrs().unwrap(), attrs);
        self.attributes(path, &attrs);
        self.contiguous(path, 2);
    }

    fn contiguous(&self, path: &str, expected: u64) {
        let dataset = self.file.dataset(path).unwrap();
        assert!(
            matches!(dataset.layout().unwrap(), Layout::Contiguous {address: Some(_), size} if size == expected)
        );
        assert!(dataset.filter_pipeline().unwrap().is_empty());
        assert_eq!(dataset.chunk_shape().unwrap(), None);
    }

    /// Checks attribute datatypes, dimensions, and fixed-size value bytes.
    fn attributes(&self, path: &str, expected: &HashMap<String, AttrValue>) {
        let header = self.header(path);
        let attrs = header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::ATTRIBUTE)
            .map(|message| {
                AttributeMessage::parse_with_length_width(
                    &message.data,
                    self.file.superblock().length_size,
                )
                .unwrap()
            })
            .map(|attr| (attr.name.clone(), attr))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(attrs.len(), expected.len());
        for (name, value) in expected {
            let attr = &attrs[name];
            match value {
                AttrValue::AsciiString(value) => {
                    assert_eq!(attr.datatype, ascii(value.len() as u32));
                    assert!(attr.dataspace.dimensions.is_empty());
                    assert_eq!(attr.raw_data, value.as_bytes());
                }
                AttrValue::VarLenAsciiCharArray(fields) => {
                    assert_eq!(
                        attr.datatype,
                        Datatype::VariableLength(VariableLengthType::sequence(ascii(1)))
                    );
                    assert_eq!(attr.dataspace.dimensions, [fields.len() as u64]);
                }
                AttrValue::I32(value) => {
                    assert_eq!(
                        attr.datatype,
                        Datatype::FixedPoint(
                            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32)
                                .unwrap()
                        )
                    );
                    assert!(attr.dataspace.dimensions.is_empty());
                    assert_eq!(attr.raw_data, value.to_le_bytes());
                }
                other => panic!("unsupported expected attribute {other:?}"),
            }
        }
    }

    fn address(&self, path: &str) -> StoredAddress {
        path.split('/').filter(|part| !part.is_empty()).fold(
            self.file.superblock().root_group_address,
            |group, part| {
                self.symbols
                    .links(group)
                    .into_iter()
                    .find(|(name, _)| name == part)
                    .unwrap()
                    .1
            },
        )
    }

    fn header(&self, path: &str) -> ObjectHeader {
        let superblock = self.file.superblock();
        ObjectHeader::parse_with_base_and_widths(
            self.bytes,
            AccessMode::ReadOnly,
            usize::try_from(superblock.base_address.get() + self.address(path).get()).unwrap(),
            FormatWidths {
                offsets: superblock.offset_size,
                lengths: superblock.length_size,
            },
            superblock.base_address,
        )
        .unwrap()
    }
}

fn ascii(size: u32) -> Datatype {
    Datatype::String {
        size,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Ascii,
    }
}
