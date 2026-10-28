//! Compact and dense link observations through pure and native readers.
//!
//! Independent construction lives in [`test_util::modern`]. The assertion helpers
//! check complete hierarchy and dataset contents separately from link storage.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util::modern::Fixture;
//! use test_util_hdf5::modern;
//!
//! let fixture = Fixture::ManyLinks;
//! let bytes = fixture.build();
//! let file = File::from_bytes(bytes.clone())?;
//! modern::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
//! modern::assert_representation(&file, &bytes, fixture);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

#[cfg(feature = "hdf5")]
use std::path::Path;

#[cfg(feature = "hdf5")]
use std::fmt::Debug;

#[cfg(feature = "hdf5")]
use hdf5::Dataset;
#[cfg(feature = "hdf5")]
use hdf5::GroupStorageType;
#[cfg(feature = "hdf5")]
use hdf5::H5Type;
#[cfg(feature = "hdf5")]
use hdf5::dataset::Layout as NativeLayout;
#[cfg(feature = "hdf5")]
use hdf5::datatype::ByteOrder;

use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::LinkMessage;
use hdf5_pure_format::__private::LinkTarget;
use hdf5_pure_format::__private::ObjectHeader;
use test_util::modern::Fixture;

/// Checks the hierarchy, dataset metadata, byte order, and values of a group fixture.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, fixture: Fixture, byte_order: DatatypeByteOrder) {
    let root = file.group("").unwrap();
    assert!(root.attrs().unwrap().is_empty());
    assert!(root.named_datatypes().unwrap().is_empty());
    match fixture {
        Fixture::Groups => {
            assert_eq!(root.groups().unwrap(), ["sensors"]);
            assert!(root.datasets().unwrap().is_empty());
            let sensors = file.group("sensors").unwrap();
            assert!(sensors.groups().unwrap().is_empty());
            assert!(sensors.named_datatypes().unwrap().is_empty());
            assert!(sensors.attrs().unwrap().is_empty());
            let mut names = sensors.datasets().unwrap();
            names.sort();
            assert_eq!(names, ["humidity", "temperature"]);
            assert_f64(file, "sensors/temperature", &[22.5, 23.1, 21.8], byte_order);
            let humidity = file.dataset("sensors/humidity").unwrap();
            assert_eq!(humidity.shape().unwrap(), [3]);
            assert_eq!(
                humidity.datatype().unwrap(),
                Datatype::FixedPoint(FixedPointType::new(4, byte_order, true, 0, 32).unwrap())
            );
            assert_eq!(humidity.read_i32().unwrap(), [45, 50, 55]);
            assert!(humidity.attrs().unwrap().is_empty());
            assert_contiguous(&humidity.layout().unwrap(), 12);
        }
        Fixture::ManyLinks => {
            assert!(root.groups().unwrap().is_empty());
            let mut names = root.datasets().unwrap();
            names.sort();
            assert_eq!(names, dataset_names());
            for index in 0..20 {
                assert_f64(
                    file,
                    &format!("dataset_{index:03}"),
                    &[f64::from(index)],
                    byte_order,
                );
            }
        }
    }
}

/// Checks version 2 group headers and compact or dense link storage.
///
/// Pass the bytes from which `file` was opened. For [`Fixture::ManyLinks`], also
/// checks the fractal heap's object count and the link-name index's type, record size,
/// and record count.
///
/// # Panics
///
/// Panics if metadata cannot be read, or group storage differs from the fixture.
pub fn assert_representation(file: &File, bytes: &[u8], fixture: Fixture) {
    let superblock = file.superblock();
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let mut pending = vec![superblock.root_group_address.get()];
    let mut observed = 0;
    while let Some(address) = pending.pop() {
        let header = ObjectHeader::parse_with_widths(
            bytes,
            AccessMode::ReadOnly,
            usize::try_from(address).unwrap(),
            widths,
        )
        .unwrap();
        assert_eq!(header.version, OBJECT_HEADER_VERSION);
        assert!(
            !header
                .messages
                .iter()
                .any(|message| message.msg_type == MessageType::SYMBOL_TABLE)
        );
        let infos: Vec<_> = header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::LINK_INFO)
            .collect();
        assert_eq!(infos.len(), 1);
        let info =
            LinkInfoMessage::parse_with_offset_width(&infos[0].data, widths.offsets).unwrap();
        let links: Vec<_> = header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::LINK)
            .collect();
        match fixture {
            Fixture::Groups => {
                assert_eq!(info.fractal_heap_address, None);
                assert_eq!(info.btree_name_index_address, None);
                assert_eq!(links.len(), if observed == 0 { 1 } else { 2 });
                if observed == 0 {
                    let link = LinkMessage::parse_with_offset_width(&links[0].data, widths.offsets)
                        .unwrap();
                    assert_eq!(link.name, "sensors");
                    let LinkTarget::Hard {
                        object_header_address,
                    } = link.link_target
                    else {
                        panic!("expected a hard link")
                    };
                    pending.push(object_header_address.get());
                }
            }
            Fixture::ManyLinks => {
                assert!(links.is_empty());
                let heap = FractalHeapHeader::parse(
                    bytes,
                    usize::try_from(info.fractal_heap_address.unwrap().get()).unwrap(),
                    widths.offsets.get(),
                    widths.lengths.get(),
                )
                .unwrap();
                assert_eq!(heap.managed_objects_count, 20);
                let index = BTreeV2Header::parse_with_widths(
                    bytes,
                    usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
                    widths,
                )
                .unwrap();
                assert_eq!(index.tree_type, LINK_NAME_INDEX);
                assert_eq!(index.record_size, LINK_RECORD_SIZE);
                assert_eq!(index.total_records, 20);
            }
        }
        observed += 1;
    }
    assert_eq!(
        observed,
        match fixture {
            Fixture::Groups => 2,
            Fixture::ManyLinks => 1,
        }
    );
}

#[cfg(feature = "hdf5")]
/// Writes a native group fixture and closes the file.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn write_libhdf5(path: &Path, fixture: Fixture) {
    let file = crate::file::libhdf5_create_v18(path);
    match fixture {
        Fixture::Groups => {
            let sensors = file.create_group("sensors").unwrap();
            sensors
                .new_dataset_builder()
                .with_data(&[22.5f64, 23.1, 21.8])
                .create("temperature")
                .unwrap();
            sensors
                .new_dataset_builder()
                .with_data(&[45i32, 50, 55])
                .create("humidity")
                .unwrap();
        }
        Fixture::ManyLinks => {
            for index in 0..20 {
                file.new_dataset_builder()
                    .with_data(&[f64::from(index)])
                    .create(format!("dataset_{index:03}").as_str())
                    .unwrap();
            }
        }
    }
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
/// Checks native group storage, dataset metadata, byte order, and values.
///
/// # Panics
///
/// Panics if the file cannot be opened, metadata or values cannot be read,
/// any observation differs from the fixture, or close fails.
pub fn assert_libhdf5_contents(path: &Path, fixture: Fixture, byte_order: ByteOrder) {
    let file = crate::file::libhdf5_open(path);
    assert!(file.attr_names().unwrap().is_empty());
    let mut names = file.member_names().unwrap();
    names.sort();
    match fixture {
        Fixture::Groups => {
            assert_eq!(file.info().unwrap().storage_type, GroupStorageType::Compact);
            assert_eq!(names, ["sensors"]);
            let sensors = file.group("sensors").unwrap();
            assert_eq!(
                sensors.info().unwrap().storage_type,
                GroupStorageType::Compact
            );
            let mut names = sensors.member_names().unwrap();
            names.sort();
            assert_eq!(names, ["humidity", "temperature"]);
            assert!(sensors.attr_names().unwrap().is_empty());
            assert_native_dataset(
                &sensors.dataset("temperature").unwrap(),
                &[22.5f64, 23.1, 21.8],
                byte_order,
            );
            assert_native_dataset(
                &sensors.dataset("humidity").unwrap(),
                &[45i32, 50, 55],
                byte_order,
            );
        }
        Fixture::ManyLinks => {
            assert_eq!(file.info().unwrap().storage_type, GroupStorageType::Dense);
            assert_eq!(names, dataset_names());
            for index in 0..20 {
                assert_native_dataset(
                    &file.dataset(&format!("dataset_{index:03}")).unwrap(),
                    &[f64::from(index)],
                    byte_order,
                );
            }
        }
    }
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
fn assert_native_dataset<T: H5Type + Debug + PartialEq>(
    dataset: &Dataset,
    expected: &[T],
    byte_order: ByteOrder,
) {
    assert_eq!(dataset.shape(), [expected.len()]);
    let datatype = dataset.dtype().unwrap();
    assert_eq!(datatype.to_descriptor().unwrap(), T::type_descriptor());
    assert_eq!(datatype.byte_order(), byte_order);
    assert_eq!(dataset.layout(), NativeLayout::Contiguous);
    assert_eq!(dataset.read_raw::<T>().unwrap(), expected);
    assert!(dataset.attr_names().unwrap().is_empty());
}

fn assert_f64(file: &File, path: &str, expected: &[f64], byte_order: DatatypeByteOrder) {
    let dataset = file.dataset(path).unwrap();
    assert_eq!(dataset.shape().unwrap(), [expected.len() as u64]);
    assert_eq!(
        dataset.datatype().unwrap(),
        Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(byte_order))
    );
    assert_eq!(dataset.read_f64().unwrap(), expected);
    assert!(dataset.attrs().unwrap().is_empty());
    assert_contiguous(&dataset.layout().unwrap(), expected.len() as u64 * 8);
}

fn assert_contiguous(layout: &Layout, expected: u64) {
    let Layout::Contiguous {
        address: Some(_),
        size,
    } = layout
    else {
        panic!("expected allocated contiguous storage, got {layout:?}")
    };
    assert_eq!(*size, expected);
}

fn dataset_names() -> Vec<String> {
    (0..20).map(|index| format!("dataset_{index:03}")).collect()
}

// Section `subsubsec_fmt4_infra_btrees_v2`, version 4.0, defines the link-name
// index type and its four-byte hash followed by a seven-byte heap ID.
const LINK_NAME_INDEX: u8 = 5;
const LINK_RECORD_SIZE: u16 = 11;

// Section `subsubsec_fmt4_dataobject_hdr_prefix_two`, version 4.0.
const OBJECT_HEADER_VERSION: u8 = 2;
