//! Content and storage observations for shared datatypes and hard-link aliases.
//!
//! Independent construction lives in [`test_util::identity`]. The helpers also
//! check native files for either [`Fixture`] variant.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util::identity::Fixture;
//! use test_util_hdf5::identity;
//!
//! let fixture = Fixture::CommittedDatatypeV1;
//! let bytes = fixture.build();
//! let file = File::from_bytes(bytes.clone())?;
//! identity::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
//! identity::assert_storage(&file, &bytes, fixture);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FixedPointType;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::SharedLocation;
use test_util::identity::Fixture;

use crate::storage;
use crate::symbol_table;

/// Checks complete hierarchy, datatypes, values, contiguous storage, and persistence properties.
///
/// All numeric types must use `order`.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, fixture: Fixture, order: DatatypeByteOrder) {
    let root = file.root();
    assert!(root.groups().unwrap().is_empty());
    assert!(root.attrs().unwrap().is_empty());
    let mut datasets = root.datasets().unwrap();
    datasets.sort();
    assert_eq!(
        datasets,
        match fixture {
            Fixture::CommittedDatatypeV1 => ["first", "second"],
            Fixture::HardLinkUndo => ["aa", "bb"],
        }
    );
    assert_eq!(
        root.named_datatypes().unwrap(),
        match fixture {
            Fixture::CommittedDatatypeV1 => vec!["mytype"],
            Fixture::HardLinkUndo => vec![],
        }
    );
    assert_data(file, fixture, order);
    for name in datasets {
        let dataset = file.dataset(&name).unwrap();
        assert!(dataset.filter_pipeline().unwrap().is_empty());
        assert_eq!(dataset.chunk_shape().unwrap(), None);
        let expected_size = match fixture {
            Fixture::CommittedDatatypeV1 => 8,
            Fixture::HardLinkUndo => 12,
        };
        assert!(
            matches!(dataset.layout().unwrap(), Layout::Contiguous { address: Some(_), size } if size == expected_size)
        );
    }
    if matches!(fixture, Fixture::HardLinkUndo) {
        let info = file.file_space_info().unwrap().unwrap();
        assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
        assert!(info.persist);
        assert_eq!(info.threshold, 1);
    }
}

/// Checks shared datatype references or hard-link identity in the encoded metadata.
///
/// Pass the bytes from which `file` was opened. The committed datatype must have
/// three references in its version 1 prefix, with both dataset messages pointing
/// to that header. The hard-link fixture must use one header for both names and
/// an Object Reference Count message equal to two.
///
/// # Panics
///
/// Panics if metadata cannot be read, or the storage premise differs from the fixture.
pub fn assert_storage(file: &File, bytes: &[u8], fixture: Fixture) {
    match fixture {
        Fixture::CommittedDatatypeV1 => {
            let links =
                symbol_table::Fixture::new(file, bytes).links(file.superblock().root_group_address);
            assert_eq!(
                links
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>(),
                ["first", "mytype", "second"]
            );
            let address = links[1].1.get();
            let committed = header(file, bytes, address);
            assert_eq!(committed.version, HEADER_VERSION_ONE);
            assert_eq!(committed.reference_count, Some(3));
            assert_eq!(
                committed
                    .messages
                    .iter()
                    .filter(|record| record.msg_type == MessageType::DATATYPE)
                    .count(),
                1
            );
            assert_ne!(links[0].1, links[2].1);
            for (_, dataset) in [&links[0], &links[2]] {
                let dataset = header(file, bytes, dataset.get());
                let message = dataset
                    .messages
                    .iter()
                    .find(|record| record.msg_type == MessageType::DATATYPE)
                    .unwrap();
                assert!(message.flags.is_shared());
                let reference =
                    format::parse_shared_ref_with_widths(&message.data, widths(file)).unwrap();
                assert_eq!(reference.location, SharedLocation::ObjectHeader(links[1].1));
            }
        }
        Fixture::HardLinkUndo => {
            let aa = storage::root_object_address(file, bytes, "aa");
            let bb = storage::root_object_address(file, bytes, "bb");
            assert_eq!(aa, bb);
            let header = header(file, bytes, aa);
            let references = header
                .messages
                .iter()
                .find(|record| record.msg_type == MessageType::OBJECT_REFERENCE_COUNT)
                .unwrap();
            assert_eq!(references.data, [REFERENCE_COUNT_VERSION, 2, 0, 0, 0]);
        }
    }
}

/// Checks numeric types, shapes, values, and the committed datatype's reference count.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub(crate) fn assert_data(file: &File, fixture: Fixture, order: DatatypeByteOrder) {
    let datatype = Datatype::FixedPoint(FixedPointType::new(4, order, true, 0, 32).unwrap());
    let (names, values): (_, &[i32]) = match fixture {
        Fixture::CommittedDatatypeV1 => {
            assert_eq!(file.root().named_datatype("mytype").unwrap(), datatype);
            assert_eq!(file.root().named_datatype_references("mytype").unwrap(), 3);
            (["first", "second"], &[1, 2])
        }
        Fixture::HardLinkUndo => (["aa", "bb"], &[1, 2, 3]),
    };
    for name in names {
        let dataset = file.dataset(name).unwrap();
        assert_eq!(dataset.shape().unwrap(), [values.len() as u64]);
        assert_eq!(dataset.datatype().unwrap(), datatype);
        assert_eq!(dataset.read_i32().unwrap(), values);
        assert!(dataset.attrs().unwrap().is_empty());
    }
}

fn header(file: &File, bytes: &[u8], address: u64) -> ObjectHeader {
    ObjectHeader::parse_with_base_and_widths(
        bytes,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        widths(file),
        file.superblock().base_address,
    )
    .unwrap()
}

fn widths(file: &File) -> FormatWidths {
    FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    }
}

// Sections `subsubsec_fmt4_dataobject_hdr_prefix_one` and
// `subsubsec_fmt4_dataobject_hdr_msg_refcount`, version 4.0.
const HEADER_VERSION_ONE: u8 = 1;
const REFERENCE_COUNT_VERSION: u8 = 0;
