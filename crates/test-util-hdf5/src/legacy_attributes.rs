//! Legacy attribute and string observations through pure and native readers.
//!
//! Independent construction lives in [`test_util::legacy_attributes`]. The assertion
//! helpers check complete contents separately from compact attribute and symbol-table
//! storage, including global-heap string payloads.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util::legacy_attributes::Fixture;
//! use test_util_hdf5::legacy_attributes;
//!
//! let fixture = Fixture::VlenStrings;
//! let bytes = fixture.build();
//! let file = File::from_bytes(bytes.clone())?;
//! legacy_attributes::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
//! legacy_attributes::assert_storage(&file, &bytes, fixture);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

use std::collections::HashMap;
use std::num::NonZeroU16;
#[cfg(feature = "hdf5")]
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::GroupStorageType;
#[cfg(feature = "hdf5")]
use hdf5::H5Type;
#[cfg(feature = "hdf5")]
use hdf5::Location;
#[cfg(feature = "hdf5")]
use hdf5::dataset::Layout as NativeLayout;
#[cfg(feature = "hdf5")]
use hdf5::datatype::ByteOrder;
#[cfg(feature = "hdf5")]
use hdf5::types::FixedAscii;
#[cfg(feature = "hdf5")]
use hdf5::types::TypeDescriptor;
#[cfg(feature = "hdf5")]
use hdf5::types::VarLenUnicode;

use hdf5_pure::AttrValue;
use hdf5_pure::CharacterSet;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::Layout;
use hdf5_pure::LengthWidth;
use hdf5_pure::StringPadding;
use hdf5_pure::SuperblockVersion;
use hdf5_pure::VariableLengthStringType;
use hdf5_pure::VariableLengthType;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_format::__private::AttributeMessage;
use hdf5_pure_format::__private::DataspaceType;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use test_util::bytes;
use test_util::legacy_attributes::Fixture;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::widths::Widths;

use crate::symbol_table;

/// Checks the hierarchy, attribute metadata and values, and dataset contents.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, fixture: Fixture, byte_order: DatatypeByteOrder) {
    let (groups, datasets): (&[&str], &[&str]) = match fixture {
        Fixture::CompactAttributes => (&[], &["data"]),
        Fixture::MixedAttributes => (&["experiment"], &[]),
        Fixture::VlenStrings => (&[], &["names"]),
    };
    assert_eq!(file.root().groups().unwrap(), groups);
    assert_eq!(file.root().datasets().unwrap(), datasets);
    assert!(file.root().named_datatypes().unwrap().is_empty());
    assert_attributes(
        file.root().attrs().unwrap(),
        file.root().attr_datatypes().unwrap(),
        fixture,
        "",
        byte_order,
    );
    match fixture {
        Fixture::CompactAttributes => {
            let data = file.dataset("data").unwrap();
            assert_eq!(data.shape().unwrap(), [3]);
            assert_eq!(
                data.datatype().unwrap(),
                Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(byte_order))
            );
            assert_eq!(data.read_f64().unwrap(), [1.0, 2.0, 3.0]);
            assert_contiguous(&data.layout().unwrap(), 24);
            assert_attributes(
                data.attrs().unwrap(),
                data.attr_datatypes().unwrap(),
                fixture,
                "data",
                byte_order,
            );
        }
        Fixture::MixedAttributes => {
            let group = file.group("experiment").unwrap();
            assert!(group.groups().unwrap().is_empty());
            assert!(group.datasets().unwrap().is_empty());
            assert!(group.named_datatypes().unwrap().is_empty());
            assert_attributes(
                group.attrs().unwrap(),
                group.attr_datatypes().unwrap(),
                fixture,
                "experiment",
                byte_order,
            );
        }
        Fixture::VlenStrings => {
            let names = file.dataset("names").unwrap();
            assert_eq!(names.shape().unwrap(), [3]);
            assert_eq!(names.datatype().unwrap(), vlen_datatype(byte_order));
            assert_eq!(names.read_string().unwrap(), ["Alice", "Bob", "Charlie"]);
            assert_eq!(names.vlen_string_payload_size().unwrap(), 15);
            assert_contiguous(&names.layout().unwrap(), 48);
            assert_attributes(
                names.attrs().unwrap(),
                names.attr_datatypes().unwrap(),
                fixture,
                "names",
                byte_order,
            );
        }
    }
}

/// Checks the version 0 superblock, symbol-table groups, and compact attribute storage.
///
/// For [`Fixture::VlenStrings`], also checks the global-heap string references and payloads.
///
/// Pass the bytes from which `file` was opened.
///
/// # Panics
///
/// Panics if metadata cannot be read, or the superblock, group links, or attribute
/// storage differ from the fixture.
pub fn assert_storage(file: &File, image: &[u8], fixture: Fixture) {
    assert_eq!(file.superblock().version(), SuperblockVersion::Zero);
    let tables = symbol_table::Fixture::new(file, image);
    let root = file.superblock().root_group_address;
    assert_header(image, root.get(), fixture, "");
    let links = tables.links(root);
    assert_eq!(links.len(), 1);
    let (name, address) = &links[0];
    let expected = match fixture {
        Fixture::CompactAttributes => "data",
        Fixture::MixedAttributes => "experiment",
        Fixture::VlenStrings => "names",
    };
    assert_eq!(name, expected);
    assert_header(image, address.get(), fixture, name);
    if matches!(fixture, Fixture::MixedAttributes) {
        assert!(tables.links(*address).is_empty());
    }
    if matches!(fixture, Fixture::VlenStrings) {
        let raw = file.dataset("names").unwrap().read_raw().unwrap();
        assert_eq!(raw.len(), 48);
        for (reference, expected) in raw
            .as_chunks::<VLEN_REFERENCE_BYTES>()
            .0
            .iter()
            .zip(["Alice", "Bob", "Charlie"])
        {
            assert_heap_string(image, reference, expected);
        }
    }
}

#[cfg(feature = "hdf5")]
/// Writes a native attribute fixture and closes the file.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn write_libhdf5(path: &Path, fixture: Fixture) {
    let file = crate::file::libhdf5_create_earliest(path);
    let root = file.group("/").unwrap();
    write_attributes(&root, fixture, "");
    match fixture {
        Fixture::CompactAttributes => {
            let dataset = file
                .new_dataset_builder()
                .with_data(&[1.0f64, 2.0, 3.0])
                .create("data")
                .unwrap();
            write_attributes(&dataset, fixture, "data");
        }
        Fixture::MixedAttributes => {
            let group = file.create_group("experiment").unwrap();
            write_attributes(&group, fixture, "experiment");
        }
        Fixture::VlenStrings => {
            let names: Vec<_> = ["Alice", "Bob", "Charlie"]
                .into_iter()
                .map(|name| name.parse::<VarLenUnicode>().unwrap())
                .collect();
            file.new_dataset_builder()
                .with_data(&names)
                .create("names")
                .unwrap();
        }
    }
    drop(root);
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
/// Checks native group storage, attribute metadata and values, and dataset contents.
///
/// # Panics
///
/// Panics if the file cannot be opened, metadata or values cannot be read,
/// any observation differs from the fixture, or close fails.
pub fn assert_libhdf5_contents(path: &Path, fixture: Fixture, order: ByteOrder) {
    let file = crate::file::libhdf5_open(path);
    let root = file.group("/").unwrap();
    assert_eq!(
        root.info().unwrap().storage_type,
        GroupStorageType::SymbolTable
    );
    let expected = match fixture {
        Fixture::CompactAttributes => "data",
        Fixture::MixedAttributes => "experiment",
        Fixture::VlenStrings => "names",
    };
    assert_eq!(root.member_names().unwrap(), [expected]);
    assert_native_attributes(&root, fixture, "", order);
    match fixture {
        Fixture::CompactAttributes => {
            let dataset = file.dataset("data").unwrap();
            assert_eq!(dataset.shape(), [3]);
            assert_eq!(dataset.layout(), NativeLayout::Contiguous);
            assert_eq!(
                dataset.dtype().unwrap().to_descriptor().unwrap(),
                f64::type_descriptor()
            );
            assert_eq!(dataset.dtype().unwrap().byte_order(), order);
            assert_eq!(dataset.read_raw::<f64>().unwrap(), [1.0, 2.0, 3.0]);
            assert_native_attributes(&dataset, fixture, "data", order);
        }
        Fixture::MixedAttributes => {
            let group = file.group("experiment").unwrap();
            assert_eq!(
                group.info().unwrap().storage_type,
                GroupStorageType::SymbolTable
            );
            assert!(group.member_names().unwrap().is_empty());
            assert_native_attributes(&group, fixture, "experiment", order);
        }
        Fixture::VlenStrings => {
            let dataset = file.dataset("names").unwrap();
            assert_eq!(dataset.shape(), [3]);
            assert_eq!(dataset.layout(), NativeLayout::Contiguous);
            assert_eq!(
                dataset.dtype().unwrap().to_descriptor().unwrap(),
                TypeDescriptor::VarLenUnicode
            );
            assert_eq!(
                dataset
                    .read_raw::<VarLenUnicode>()
                    .unwrap()
                    .iter()
                    .map(VarLenUnicode::as_str)
                    .collect::<Vec<_>>(),
                ["Alice", "Bob", "Charlie"]
            );
            assert_native_attributes(&dataset, fixture, "names", order);
        }
    }
    drop(root);
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
fn write_attributes(location: &Location, fixture: Fixture, path: &str) {
    for (name, value) in attributes(fixture, path) {
        match value {
            Value::Ascii(value) => location
                .new_attr_builder()
                .empty_as(TypeDescriptor::FixedAscii(value.len()))
                .shape(())
                .create(*name)
                .unwrap()
                .write_scalar(&FixedAscii::<32>::from_ascii(value).unwrap())
                .unwrap(),
            Value::F64(value) => location
                .new_attr::<f64>()
                .shape(())
                .create(*name)
                .unwrap()
                .write_scalar(value)
                .unwrap(),
            Value::F64Array(value) => location
                .new_attr::<f64>()
                .shape(value.len())
                .create(*name)
                .unwrap()
                .write_raw(*value)
                .unwrap(),
            Value::I64(value) => location
                .new_attr::<i64>()
                .shape(())
                .create(*name)
                .unwrap()
                .write_scalar(value)
                .unwrap(),
            Value::Unicode(value) => location
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create(*name)
                .unwrap()
                .write_scalar(&value.parse::<VarLenUnicode>().unwrap())
                .unwrap(),
        }
    }
}

#[cfg(feature = "hdf5")]
fn assert_native_attributes(location: &Location, fixture: Fixture, path: &str, order: ByteOrder) {
    let mut actual = location.attr_names().unwrap();
    actual.sort();
    assert_eq!(
        actual,
        attributes(fixture, path)
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
    );
    for (name, value) in attributes(fixture, path) {
        let attribute = location.attr(name).unwrap();
        assert_eq!(
            attribute.shape(),
            value
                .shape()
                .iter()
                .map(|dim| usize::try_from(*dim).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(attribute.is_scalar(), value.shape().is_empty());
        let datatype = attribute.dtype().unwrap();
        let descriptor = match value {
            Value::Ascii(value) => TypeDescriptor::FixedAscii(value.len()),
            Value::F64(_) | Value::F64Array(_) => f64::type_descriptor(),
            Value::I64(_) => i64::type_descriptor(),
            Value::Unicode(_) => TypeDescriptor::VarLenUnicode,
        };
        assert_eq!(datatype.to_descriptor().unwrap(), descriptor);
        match value {
            Value::Ascii(expected) => assert_eq!(
                attribute.read_scalar::<FixedAscii<32>>().unwrap().as_str(),
                *expected
            ),
            Value::F64(expected) => {
                assert_eq!(datatype.byte_order(), order);
                assert_eq!(attribute.read_scalar::<f64>().unwrap(), *expected);
            }
            Value::F64Array(expected) => {
                assert_eq!(datatype.byte_order(), order);
                assert_eq!(attribute.read_raw::<f64>().unwrap(), *expected);
            }
            Value::I64(expected) => {
                assert_eq!(datatype.byte_order(), order);
                assert_eq!(attribute.read_scalar::<i64>().unwrap(), *expected);
            }
            Value::Unicode(expected) => assert_eq!(
                attribute.read_scalar::<VarLenUnicode>().unwrap().as_str(),
                *expected
            ),
        }
    }
}

fn assert_header(image: &[u8], address: u64, fixture: Fixture, path: &str) {
    let chunks = v1::chunks(image, usize::try_from(address).unwrap(), Widths::EIGHT);
    let records: Vec<_> = chunks.iter().flat_map(|chunk| &chunk.records).collect();
    assert!(
        records
            .iter()
            .all(|record| record.msg_type != MessageType::ATTRIBUTE_INFO)
    );
    let mut actual = Vec::new();
    for record in records
        .iter()
        .filter(|record| record.msg_type == MessageType::ATTRIBUTE)
    {
        let attribute = AttributeMessage::parse(&image[record.body.clone()], 8).unwrap();
        let (_, value) = attributes(fixture, path)
            .iter()
            .find(|(name, _)| *name == attribute.name)
            .unwrap();
        assert_eq!(attribute.dataspace.dimensions, value.shape());
        assert_eq!(
            attribute.dataspace.space_type,
            if value.shape().is_empty() {
                DataspaceType::Scalar
            } else {
                DataspaceType::Simple
            }
        );
        actual.push(attribute.name);
        if let Value::Unicode(expected) = value {
            assert_heap_string(image, &attribute.raw_data, expected);
        }
    }
    actual.sort();
    let expected: Vec<_> = attributes(fixture, path)
        .iter()
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(actual, expected);
    let is_group = path.is_empty() || matches!(fixture, Fixture::MixedAttributes);
    assert_eq!(
        records
            .iter()
            .filter(|record| record.msg_type == MessageType::SYMBOL_TABLE)
            .count(),
        usize::from(is_group)
    );
}

fn assert_heap_string(image: &[u8], reference: &[u8], expected: &str) {
    assert_eq!(reference.len(), VLEN_REFERENCE_BYTES);
    assert_eq!(bytes::u32_at(reference, 0) as usize, expected.len());
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), image.len() as u64).unwrap(),
        image,
    )
    .unwrap();
    let locator = GlobalHeapCollectionLocator::new(
        PhysicalAddress::new(bytes::u64_at(reference, SEQUENCE_LENGTH_BYTES)),
        LengthWidth::Eight,
    );
    let collection = locator
        .parse(region.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap();
    let index = NonZeroU16::new(
        u16::try_from(bytes::u32_at(
            reference,
            SEQUENCE_LENGTH_BYTES + Widths::EIGHT.offset,
        ))
        .unwrap(),
    )
    .unwrap();
    let mut lookup = collection.lookup(index);
    while let Some(range) = lookup.parse_range() {
        if let Some(object) = lookup.parse(region.subregion(range).unwrap()).unwrap() {
            assert_eq!(object.payload().parse(region).unwrap(), expected.as_bytes());
        }
    }
}

fn assert_attributes(
    actual: HashMap<String, AttrValue>,
    datatypes: HashMap<String, Datatype>,
    fixture: Fixture,
    path: &str,
    byte_order: DatatypeByteOrder,
) {
    let expected = attributes(fixture, path);
    assert_eq!(
        actual,
        expected
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.value()))
            .collect()
    );
    assert_eq!(
        datatypes,
        expected
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.datatype(byte_order)))
            .collect()
    );
}

fn assert_contiguous(layout: &Layout, expected: u64) {
    let Layout::Contiguous {
        address: Some(_),
        size,
    } = layout
    else {
        panic!("expected allocated contiguous storage, got {layout:?}");
    };
    assert_eq!(*size, expected);
}

fn attributes(fixture: Fixture, path: &str) -> &'static [(&'static str, Value)] {
    match (fixture, path) {
        (Fixture::CompactAttributes, "") => &[("file_type", Value::Ascii("experiment"))],
        (Fixture::CompactAttributes, "data") => &[
            ("description", Value::Ascii("test dataset")),
            ("scale", Value::F64(314.0 / 100.0)),
            ("version", Value::I64(42)),
        ],
        (Fixture::MixedAttributes, "experiment") => &[
            ("iterations", Value::I64(1_000)),
            ("name", Value::Ascii("run_001")),
            ("temperatures", Value::F64Array(&[22.5, 23.1, 21.8])),
        ],
        (Fixture::VlenStrings, "") => &[("vl_attr", Value::Unicode("hello variable length"))],
        _ => &[],
    }
}

enum Value {
    Ascii(&'static str),
    F64(f64),
    F64Array(&'static [f64]),
    I64(i64),
    Unicode(&'static str),
}

impl Value {
    fn value(&self) -> AttrValue {
        match self {
            Self::Ascii(value) => AttrValue::AsciiString((*value).to_owned()),
            Self::F64(value) => AttrValue::F64(*value),
            Self::F64Array(value) => AttrValue::F64Array(value.to_vec()),
            Self::I64(value) => AttrValue::I64(*value),
            Self::Unicode(value) => AttrValue::VarLenString((*value).to_owned()),
        }
    }

    fn datatype(&self, order: DatatypeByteOrder) -> Datatype {
        match self {
            Self::Ascii(value) => Datatype::String {
                size: value.len() as u32,
                padding: StringPadding::NullPad,
                charset: CharacterSet::Ascii,
            },
            Self::F64(_) | Self::F64Array(_) => {
                Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(order))
            }
            Self::I64(_) => {
                Datatype::FixedPoint(FixedPointType::new(8, order, true, 0, 64).unwrap())
            }
            Self::Unicode(_) => vlen_datatype(order),
        }
    }

    fn shape(&self) -> Vec<u64> {
        match self {
            Self::F64Array(values) => vec![values.len() as u64],
            _ => Vec::new(),
        }
    }
}

fn vlen_datatype(order: DatatypeByteOrder) -> Datatype {
    Datatype::VariableLength(VariableLengthType::String(
        VariableLengthStringType::with_base_type(
            StringPadding::NullTerminate,
            CharacterSet::Utf8,
            Datatype::FixedPoint(FixedPointType::new(1, order, false, 0, 8).unwrap()),
        )
        .unwrap(),
    ))
}

// H5T__vlen_disk_write in H5Tvlen.c, HDF5 2.2.0, writes a four-byte sequence count.
const SEQUENCE_LENGTH_BYTES: usize = 4;

// Section `subsec_fmt4_infra_globalheap`, version 4.0, stores a four-byte object index.
const HEAP_INDEX_BYTES: usize = 4;

// H5T__vlen_disk_write in H5Tvlen.c, HDF5 2.2.0, prefixes the global heap ID
// from section `subsec_fmt4_infra_globalheap`, version 4.0, with a sequence count.
const VLEN_REFERENCE_BYTES: usize = SEQUENCE_LENGTH_BYTES + Widths::EIGHT.offset + HEAP_INDEX_BYTES;
