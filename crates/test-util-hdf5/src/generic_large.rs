//! Content, storage, and header-preservation observations for generic-large fixtures.
//!
//! Independent construction lives in [`test_util::generic_large::generic_large`].
//! Header captures retain complete encoded chunks for exact comparisons across edits.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{AttrValue, DatatypeByteOrder, File};
//! use test_util::temp;
//! use test_util_hdf5::generic_large::{self, HeaderImage, State};
//!
//! let path = temp::temp_path("generic_large.h5");
//! let image = test_util::generic_large::generic_large();
//! std::fs::write(&path, &image)?;
//! let file = File::open(&path)?;
//! generic_large::assert_contents(&file, DatatypeByteOrder::LittleEndian, State::Original);
//! generic_large::assert_storage(&file, &image);
//! generic_large::assert_independent_spans(&file, &image);
//! file.close()?;
//!
//! let header = HeaderImage::read(&path);
//! let file = File::open_rw(&path)?;
//! file.root().set_attr("probe", AttrValue::I32(9))?;
//! file.commit()?;
//! file.close()?;
//! assert_eq!(HeaderImage::read(&path), header);
//! let file = File::open(&path)?;
//! generic_large::assert_contents(&file, DatatypeByteOrder::LittleEndian, State::RootAttribute);
//! file.close()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "__hdf5-1.10")]
use hdf5::H5Type;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::dataset::Layout as NativeLayout;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::datatype::ByteOrder as NativeByteOrder;
use hdf5_pure::AddressRange;
use hdf5_pure::AttrValue;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure::PhysicalAddress;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::AttributeMessage;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::ObjectHeader;

#[cfg(feature = "__hdf5-1.10")]
use crate::numeric::ByteOrder as NumericOrder;
use crate::storage;

/// The expected hierarchy before or after root-attribute and dataset-deletion edits.
#[derive(Clone, Copy)]
pub enum State {
    /// The root attribute `probe` equals 9 and `d` is absent.
    Deleted,
    /// The root has no attributes and `d` retains its original contents.
    Original,
    /// The root attribute `probe` equals 9 and `d` retains its original contents.
    RootAttribute,
}

/// The addresses and complete bytes of dataset `d`'s object-header chunks.
///
/// Equality compares every encoded byte, including attribute records and padding,
/// as well as each chunk's address.
#[derive(Debug, PartialEq)]
pub struct HeaderImage(Vec<(PhysicalAddress, Vec<u8>)>);

impl HeaderImage {
    /// Reads every header chunk of `d` for an exact preservation comparison.
    ///
    /// # Panics
    ///
    /// Panics if the file or dataset header cannot be read.
    pub fn read(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let image = std::fs::read(path).unwrap();
        Self(
            spans(&file, &image)
                .into_iter()
                .map(|range| {
                    let start = usize::try_from(range.start().get()).unwrap();
                    let end = usize::try_from(range.end().get()).unwrap();
                    (range.start(), image[start..end].to_vec())
                })
                .collect(),
        )
    }
}

/// Checks hierarchy, attributes, values, storage, and persisted page policy for `state`.
///
/// While `d` is present, its numeric datatype and the datatype of `big_attr` must
/// use `order`.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, order: DatatypeByteOrder, state: State) {
    let root = file.root();
    assert!(root.groups().unwrap().is_empty());
    assert!(root.named_datatypes().unwrap().is_empty());
    let expected = match state {
        State::Original => HashMap::new(),
        State::RootAttribute | State::Deleted => {
            HashMap::from([("probe".to_owned(), AttrValue::I32(9))])
        }
    };
    assert_eq!(root.attrs().unwrap(), expected);
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert!(info.persist);
    assert_eq!(info.threshold, 1);
    assert_eq!(info.page_size.get(), PAGE);
    if matches!(state, State::Deleted) {
        assert!(root.datasets().unwrap().is_empty());
        return;
    }
    assert_eq!(root.datasets().unwrap(), ["d"]);
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.shape().unwrap(), [64]);
    assert_eq!(
        dataset.datatype().unwrap(),
        Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(order))
    );
    assert_eq!(dataset.read_f64().unwrap(), [1.0; 64]);
    assert_eq!(dataset.chunk_shape().unwrap(), None);
    assert!(dataset.filter_pipeline().unwrap().is_empty());
    assert!(matches!(
        dataset.layout().unwrap(),
        Layout::Contiguous {
            address: Some(_),
            size: 512
        }
    ));
    assert_eq!(
        dataset.attrs().unwrap(),
        HashMap::from([("big_attr".to_owned(), AttrValue::I64Array(vec![7; 512]))])
    );
    assert_eq!(
        dataset.attr_datatypes().unwrap(),
        HashMap::from([(
            "big_attr".to_owned(),
            Datatype::FixedPoint(FixedPointType::new(8, order, true, 0, 64).unwrap())
        )])
    );
}

/// Checks a continued dataset header and a fragment in the generic-large manager.
///
/// Pass the bytes from which `file` was opened. A header chunk must start on a
/// 512-byte page boundary, exceed one page, and end within a page. The parsed
/// header must contain one `big_attr` message with dimensions `[512]`.
/// Addresses and free-space fragments are read from the file.
///
/// # Panics
///
/// Panics if metadata cannot be read, or storage differs from the fixture.
pub fn assert_storage(file: &File, image: &[u8]) {
    let spans = spans(file, image);
    assert!(spans.len() > 1);
    assert!(spans.iter().any(|range| {
        range.start().get() % PAGE == 0 && range.length() > PAGE && range.length() % PAGE != 0
    }));
    let superblock = file.superblock();
    let header = ObjectHeader::parse_with_base_and_widths(
        image,
        AccessMode::ReadOnly,
        usize::try_from(storage::root_object_address(file, image, "d").get()).unwrap(),
        FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        },
        BaseAddress::new(file.resolve_address(StoredAddress::ZERO).unwrap().get()),
    )
    .unwrap();
    let attributes = header
        .messages
        .iter()
        .filter(|record| record.msg_type == MessageType::ATTRIBUTE)
        .map(|record| {
            AttributeMessage::parse_with_length_width(&record.data, superblock.length_size).unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(attributes.len(), 1);
    assert_eq!(attributes[0].name, "big_attr");
    assert_eq!(attributes[0].dataspace.dimensions, [512]);
    let report = file.persisted_free_sections().unwrap().unwrap();
    let manager = report
        .managers()
        .iter()
        .find(|manager| manager.role() == FreeSpaceManagerRole::PageLargeGeneric)
        .unwrap();
    assert!(manager.free_sections().sections().iter().any(|section| {
        section.address().get() % PAGE != 0 || section.length().get() % PAGE != 0
    }));
}

/// Checks the fixed header-chunk addresses and lengths of the independent image.
///
/// Pass matching inputs from [`test_util::generic_large::generic_large`]. The
/// chunks must occupy `(512, 268)` and `(1536, 4158)` as `(address, length)` pairs.
///
/// # Panics
///
/// Panics if the header cannot be read, or its extents differ.
pub fn assert_independent_spans(file: &File, image: &[u8]) {
    assert_eq!(
        spans(file, image),
        [
            AddressRange::new(PhysicalAddress::new(512), 268).unwrap(),
            AddressRange::new(PhysicalAddress::new(1_536), 4_158).unwrap(),
        ]
    );
}

/// Checks the root probe and unchanged dataset contents through the reference library.
///
/// The probe must be a scalar little-endian [`i32`] equal to 9. The dataset and
/// its attribute must use `order`. Closes the file before returning.
///
/// # Panics
///
/// Panics if opening, reading, or close fails, or any observation differs from the edited fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_native_after_probe(path: &Path, order: NumericOrder) {
    let file = crate::file::libhdf5_open(path);
    assert_eq!(file.member_names().unwrap(), ["d"]);
    assert_eq!(file.attr_names().unwrap(), ["probe"]);
    {
        let probe = file.attr("probe").unwrap();
        assert!(probe.shape().is_empty());
        assert_eq!(
            probe.dtype().unwrap().to_descriptor().unwrap(),
            i32::type_descriptor()
        );
        assert_eq!(
            probe.dtype().unwrap().byte_order(),
            NativeByteOrder::LittleEndian
        );
        assert_eq!(probe.read_scalar::<i32>().unwrap(), 9);
        let dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.shape(), [64]);
        assert_eq!(
            dataset.dtype().unwrap().to_descriptor().unwrap(),
            f64::type_descriptor()
        );
        assert_eq!(dataset.dtype().unwrap().byte_order(), order.native_order());
        assert_eq!(dataset.read_raw::<f64>().unwrap(), [1.0; 64]);
        assert_eq!(dataset.layout(), NativeLayout::Contiguous);
        assert_eq!(dataset.chunk(), None);
        assert!(dataset.filters().is_empty());
        assert_eq!(dataset.attr_names().unwrap(), ["big_attr"]);
        let attribute = dataset.attr("big_attr").unwrap();
        assert_eq!(attribute.shape(), [512]);
        assert_eq!(
            attribute.dtype().unwrap().to_descriptor().unwrap(),
            i64::type_descriptor()
        );
        assert_eq!(
            attribute.dtype().unwrap().byte_order(),
            order.native_order()
        );
        assert_eq!(attribute.read_raw::<i64>().unwrap(), [7; 512]);
    }
    file.close().unwrap();
}

fn spans(file: &File, image: &[u8]) -> Vec<AddressRange<PhysicalAddress>> {
    storage::header_extents(file, image, storage::root_object_address(file, image, "d"))
}

const PAGE: u64 = 512;
