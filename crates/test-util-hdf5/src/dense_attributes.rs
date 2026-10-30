//! Dense attribute observations through pure and native readers.
//!
//! Independent construction lives in [`test_util::dense_attributes`]. The assertion
//! helpers check complete contents separately from heap and attribute-index storage.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util::dense_attributes::Fixture;
//! use test_util_hdf5::dense_attributes;
//!
//! let fixture = Fixture::Dataset;
//! let bytes = fixture.build();
//! let file = File::from_bytes(bytes.clone())?;
//! dense_attributes::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
//! dense_attributes::assert_storage(&file, &bytes, fixture, DatatypeByteOrder::LittleEndian);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

use std::collections::BTreeMap;
use std::collections::HashMap;
#[cfg(feature = "hdf5")]
use std::fmt::Debug;
#[cfg(feature = "hdf5")]
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::Dataset;
#[cfg(feature = "hdf5")]
use hdf5::H5Type;
#[cfg(feature = "hdf5")]
use hdf5::Location;
#[cfg(feature = "hdf5")]
use hdf5::dataset::Layout as NativeLayout;
#[cfg(feature = "hdf5")]
use hdf5::datatype::ByteOrder;

use hdf5_pure::AttrValue;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::AttributeMessage;
use hdf5_pure_format::__private::BTREE_V2_ATTRIBUTE_NAME;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::DataspaceType;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapBlock;
use hdf5_pure_format::__private::FractalHeapBlockLayout;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::LinkMessage;
use hdf5_pure_format::__private::LinkTarget;
use hdf5_pure_format::__private::ObjectHeader;
use test_util::bytes;
use test_util::checksum;
use test_util::dense_attributes::Fixture;

/// Checks the hierarchy, dataset metadata and values, and complete attribute maps.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, fixture: Fixture, byte_order: DatatypeByteOrder) {
    let root = file.root();
    assert!(root.groups().unwrap().is_empty());
    assert!(root.named_datatypes().unwrap().is_empty());
    let path = match fixture {
        Fixture::Dataset => "data",
        Fixture::Root => "values",
    };
    assert_eq!(root.datasets().unwrap(), [path]);
    let dataset = file.dataset(path).unwrap();
    assert_eq!(dataset.shape().unwrap(), [3]);
    assert!(
        matches!(dataset.layout().unwrap(), Layout::Contiguous { address: Some(_), size } if size == match fixture { Fixture::Dataset => 24, Fixture::Root => 12 })
    );
    let (attributes, datatypes) = match fixture {
        Fixture::Dataset => {
            assert!(root.attrs().unwrap().is_empty());
            assert!(root.attr_datatypes().unwrap().is_empty());
            assert_eq!(dataset.datatype().unwrap(), f64_datatype(byte_order));
            assert_eq!(dataset.read_f64().unwrap(), [1.0, 2.0, 3.0]);
            (dataset.attrs().unwrap(), dataset.attr_datatypes().unwrap())
        }
        Fixture::Root => {
            assert!(dataset.attrs().unwrap().is_empty());
            assert!(dataset.attr_datatypes().unwrap().is_empty());
            assert_eq!(
                dataset.datatype().unwrap(),
                Datatype::FixedPoint(FixedPointType::new(4, byte_order, true, 0, 32).unwrap())
            );
            assert_eq!(dataset.read_i32().unwrap(), [10, 20, 30]);
            (root.attrs().unwrap(), root.attr_datatypes().unwrap())
        }
    };
    assert_eq!(
        attributes,
        fixture
            .attributes()
            .map(|(name, value)| (name, AttrValue::F64(value)))
            .collect::<HashMap<_, _>>()
    );
    assert_eq!(
        datatypes,
        fixture
            .attributes()
            .map(|(name, _)| (name, f64_datatype(byte_order)))
            .collect::<HashMap<_, _>>()
    );
}

/// Checks version 2 headers, fractal heaps, and attribute-name indexes.
///
/// Pass the bytes from which `file` was opened. The checks cover managed object
/// counts, index records, scalar dataspaces, datatypes, and attribute-name hashes.
///
/// # Panics
///
/// Panics if metadata cannot be read, or storage or indexed attribute metadata
/// differs from the fixture.
pub fn assert_storage(file: &File, image: &[u8], fixture: Fixture, byte_order: DatatypeByteOrder) {
    let superblock = file.superblock();
    assert_eq!(superblock.base_address.get(), 0);
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let root = ObjectHeader::parse_with_widths(
        image,
        AccessMode::ReadOnly,
        usize::try_from(superblock.root_group_address.get()).unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(root.version, OBJECT_HEADER_VERSION);
    let links: Vec<_> = root
        .messages
        .iter()
        .filter(|message| message.msg_type == MessageType::LINK)
        .collect();
    assert_eq!(links.len(), 1);
    let link = LinkMessage::parse_with_offset_width(&links[0].data, widths.offsets).unwrap();
    assert_eq!(
        link.name,
        match fixture {
            Fixture::Dataset => "data",
            Fixture::Root => "values",
        }
    );
    let LinkTarget::Hard {
        object_header_address,
    } = link.link_target
    else {
        panic!("expected a hard link")
    };
    let dataset = ObjectHeader::parse_with_widths(
        image,
        AccessMode::ReadOnly,
        usize::try_from(object_header_address.get()).unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(dataset.version, OBJECT_HEADER_VERSION);
    let owner = match fixture {
        Fixture::Dataset => &dataset,
        Fixture::Root => &root,
    };
    assert!(
        !owner
            .messages
            .iter()
            .any(|message| message.msg_type == MessageType::ATTRIBUTE)
    );
    let infos: Vec<_> = owner
        .messages
        .iter()
        .filter(|message| message.msg_type == MessageType::ATTRIBUTE_INFO)
        .collect();
    assert_eq!(infos.len(), 1);
    let info =
        AttributeInfoMessage::parse_with_offset_width(&infos[0].data, widths.offsets).unwrap();
    let heap_address = info.fractal_heap_address.unwrap();
    let heap = FractalHeapHeader::parse(
        image,
        usize::try_from(heap_address.get()).unwrap(),
        widths.offsets.get(),
        widths.lengths.get(),
    )
    .unwrap();
    assert_eq!(
        heap.managed_objects_count,
        u64::from(fixture.attribute_count())
    );
    assert_eq!(heap.huge_objects_count, 0);
    assert_eq!(heap.tiny_objects_count, 0);
    let header = BTreeV2Header::parse_with_widths(
        image,
        usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(header.tree_type, BTREE_V2_ATTRIBUTE_NAME);
    assert_eq!(header.record_size, ATTRIBUTE_RECORD_SIZE);
    assert_eq!(header.total_records, u64::from(fixture.attribute_count()));
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), image.len() as u64).unwrap(),
        image,
    )
    .unwrap();
    let blocks =
        FractalHeapBlockLayout::new(&heap, widths.offsets, widths.lengths, heap_address).unwrap();
    let mut pending = vec![blocks.root().unwrap().unwrap()];
    let mut direct = Vec::new();
    while let Some(block) = pending.pop() {
        match block {
            FractalHeapBlock::Direct(block) => {
                let reference = block
                    .at(PhysicalAddress::new(block.address().get()))
                    .unwrap();
                let checked = reference.parse_checked(region).unwrap();
                direct.push((block.heap_offset(), checked.bytes()));
            }
            FractalHeapBlock::Indirect(block) => {
                let reference = block
                    .at(PhysicalAddress::new(block.address().get()))
                    .unwrap();
                pending.extend(
                    reference
                        .parse(region)
                        .unwrap()
                        .children()
                        .map(Result::unwrap),
                );
            }
        }
    }
    let ids = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths);
    let tree = BTreeV2Layout::new(&header, widths.offsets).unwrap();
    let mut pending = vec![tree.root().unwrap()];
    let mut attributes = BTreeMap::new();
    while let Some(node) = pending.pop() {
        let node = node
            .parse(&image[usize::try_from(node.address().get()).unwrap()..])
            .unwrap();
        for index in 0..node.record_count() {
            let record = node.record(index).unwrap();
            let FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } = ids
                .parse(&record[..usize::from(heap.heap_id_length)])
                .unwrap()
                .kind()
            else {
                panic!("expected a managed attribute")
            };
            let (offset, block) = direct
                .iter()
                .find(|(offset, block)| {
                    heap_offset >= *offset && heap_offset < *offset + block.len() as u64
                })
                .unwrap();
            let start = usize::try_from(heap_offset - offset).unwrap();
            let end = start + usize::try_from(object_length).unwrap();
            let attribute =
                AttributeMessage::parse(&block[start..end], widths.lengths.get()).unwrap();
            assert_eq!(attribute.dataspace.space_type, DataspaceType::Scalar);
            assert_eq!(attribute.dataspace.rank, 0);
            assert!(attribute.dataspace.dimensions.is_empty());
            assert_eq!(attribute.datatype, f64_datatype(byte_order));
            assert_eq!(
                bytes::u32_at(record, record.len() - size_of::<u32>()),
                checksum::lookup3(attribute.name.as_bytes())
            );
            assert!(
                attributes
                    .insert(attribute.name, attribute.datatype)
                    .is_none()
            );
        }
        for index in 0..=node.record_count() {
            if let Some(child) = node.child(index).unwrap() {
                pending.push(child);
            }
        }
    }
    assert_eq!(
        attributes,
        fixture
            .attributes()
            .map(|(name, _)| (name, f64_datatype(byte_order)))
            .collect()
    );
}

#[cfg(feature = "hdf5")]
/// Writes a native dense attribute fixture and closes the file.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn write_libhdf5(path: &Path, fixture: Fixture) {
    let file = crate::file::libhdf5_create_v18(path);
    match fixture {
        Fixture::Dataset => {
            let dataset = file
                .new_dataset_builder()
                .with_data(&[1.0f64, 2.0, 3.0])
                .create("data")
                .unwrap();
            write_attributes(&dataset, fixture);
        }
        Fixture::Root => {
            file.new_dataset_builder()
                .with_data(&[10i32, 20, 30])
                .create("values")
                .unwrap();
            write_attributes(&file, fixture);
        }
    }
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
/// Checks native dataset and attribute metadata, byte order, and values.
///
/// # Panics
///
/// Panics if the file cannot be opened, metadata or values cannot be read,
/// any observation differs from the fixture, or close fails.
pub fn assert_libhdf5_contents(path: &Path, fixture: Fixture, byte_order: ByteOrder) {
    let file = crate::file::libhdf5_open(path);
    let mut members = file.member_names().unwrap();
    members.sort();
    match fixture {
        Fixture::Dataset => {
            assert_eq!(members, ["data"]);
            assert!(file.attr_names().unwrap().is_empty());
            let dataset = file.dataset("data").unwrap();
            assert_native_dataset(&dataset, &[1.0f64, 2.0, 3.0], byte_order);
            assert_native_attributes(&dataset, fixture, byte_order);
        }
        Fixture::Root => {
            assert_eq!(members, ["values"]);
            let dataset = file.dataset("values").unwrap();
            assert_native_dataset(&dataset, &[10i32, 20, 30], byte_order);
            assert!(dataset.attr_names().unwrap().is_empty());
            assert_native_attributes(&file, fixture, byte_order);
        }
    }
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
fn write_attributes(owner: &Location, fixture: Fixture) {
    for (name, value) in fixture.attributes() {
        owner
            .new_attr::<f64>()
            .create(name.as_str())
            .unwrap()
            .write_scalar(&value)
            .unwrap();
    }
}

#[cfg(feature = "hdf5")]
fn assert_native_attributes(owner: &Location, fixture: Fixture, byte_order: ByteOrder) {
    let mut names = owner.attr_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        fixture
            .attributes()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    );
    for (name, value) in fixture.attributes() {
        let attribute = owner.attr(&name).unwrap();
        assert!(attribute.shape().is_empty());
        assert_eq!(
            attribute.dtype().unwrap().to_descriptor().unwrap(),
            f64::type_descriptor()
        );
        assert_eq!(attribute.dtype().unwrap().byte_order(), byte_order);
        assert_eq!(attribute.read_scalar::<f64>().unwrap(), value);
    }
}

#[cfg(feature = "hdf5")]
fn assert_native_dataset<T: H5Type + Debug + PartialEq>(
    dataset: &Dataset,
    expected: &[T],
    byte_order: ByteOrder,
) {
    assert_eq!(dataset.shape(), [expected.len()]);
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        T::type_descriptor()
    );
    assert_eq!(dataset.dtype().unwrap().byte_order(), byte_order);
    assert_eq!(dataset.layout(), NativeLayout::Contiguous);
    assert_eq!(dataset.read_raw::<T>().unwrap(), expected);
}

fn f64_datatype(byte_order: DatatypeByteOrder) -> Datatype {
    Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(byte_order))
}

// Section `subsubsec_fmt4_dataobject_hdr_prefix_two`, version 4.0.
const OBJECT_HEADER_VERSION: u8 = 2;

// Section `subsubsec_fmt4_infra_btrees_v2`, version 4.0, defines the type 8
// record's eight-byte heap ID, flag byte, creation order and name hash.
const ATTRIBUTE_RECORD_SIZE: u16 = 17;
