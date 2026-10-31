//! Content and storage observations for paged `keep` and `victim` datasets.
//!
//! Independent construction lives in [`test_util::paged_index::paged_index`].
//! Storage checks follow encoded addresses in independent or native files.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util_hdf5::paged_index;
//!
//! let image = test_util::paged_index::paged_index();
//! let file = File::from_bytes(image.clone())?;
//! paged_index::assert_contents(&file, DatatypeByteOrder::LittleEndian);
//! paged_index::assert_storage(&file, &image);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

#[cfg(feature = "__hdf5-1.10")]
use std::path::Path;

#[cfg(feature = "__hdf5-1.10")]
use hdf5::H5Type;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::dataset::Layout as NativeLayout;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::datatype::ByteOrder as NativeByteOrder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FixedPointType;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_core::MaxExtent;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::Dataspace;
use hdf5_pure_format::__private::ExtensibleArrayDataBlock;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::ObjectHeader;

#[cfg(feature = "__hdf5-1.10")]
use crate::numeric::ByteOrder as NumericOrder;
use crate::storage;

/// Checks the complete hierarchy, numeric contents, chunking, and persistence properties.
///
/// Both datasets must use [`i32`] in `order`. `victim` contains values 0 through
/// 8,191 in 512-element chunks, and `keep` contains `[1, 2, 3, 4]` contiguously.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, order: DatatypeByteOrder) {
    assert_root(file, &["keep", "victim"]);
    assert_i32(file, "keep", &[1, 2, 3, 4], order);
    assert_i32(file, "victim", &(0..8192).collect::<Vec<_>>(), order);
    let victim = file.dataset("victim").unwrap();
    assert_eq!(victim.chunk_shape().unwrap(), Some(vec![512]));
    assert_eq!(
        victim.chunk_index().unwrap(),
        Some(ChunkIndex::ExtensibleArray)
    );
    let keep = file.dataset("keep").unwrap();
    assert_eq!(keep.chunk_shape().unwrap(), None);
    assert!(matches!(
        keep.layout().unwrap(),
        Layout::Contiguous {
            address: Some(_),
            size: 16
        }
    ));
    assert_persistence(file);
}

/// Checks the extensible-array index and separation of metadata and raw-data pages.
///
/// Pass the bytes from which `file` was opened. Checks the unlimited dimension,
/// sixteen unfiltered chunks, and nonempty persisted metadata and raw-data managers.
/// The layout's five creation parameters must match the array header. Allocated
/// index and data blocks must occupy separate pages from the raw chunks.
///
/// # Panics
///
/// Panics if metadata cannot be read, or storage differs from the fixture.
pub fn assert_storage(file: &File, image: &[u8]) {
    let superblock = file.superblock();
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let address = storage::root_object_address(file, image, "victim");
    let header = ObjectHeader::parse_with_base_and_widths(
        image,
        AccessMode::ReadOnly,
        usize::try_from(address.get()).unwrap(),
        widths,
        BaseAddress::new(file.resolve_address(StoredAddress::ZERO).unwrap().get()),
    )
    .unwrap();
    let space = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATASPACE)
        .unwrap();
    let space = Dataspace::parse_with_length_width(&space.data, widths.lengths).unwrap();
    assert_eq!(space.dimensions, [8192]);
    assert_eq!(space.max_dimensions, Some(vec![MaxExtent::Unlimited]));
    let layout = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let DataLayout::Chunked {
        chunk_dimensions,
        index:
            ChunkIndexLayout::ExtensibleArray {
                address: Some(address),
                creation: _,
            },
        ..
    } = DataLayout::parse_with_widths(&layout.data, widths).unwrap()
    else {
        panic!("expected an allocated extensible array")
    };
    assert_eq!(chunk_dimensions, [512, 4]);
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), image.len() as u64).unwrap(),
        image,
    )
    .unwrap();
    let locator = ExtensibleArrayHeaderLocator::new(
        widths,
        BaseAddress::new(file.resolve_address(StoredAddress::ZERO).unwrap().get()),
        address,
    );
    let array = locator.parse(region).unwrap();
    assert_eq!(array.header().client_id, CLIENT_UNFILTERED);
    assert_eq!(array.header().max_idx_set, 16);
    let parameters_at = LAYOUT_PREFIX_LEN
        + usize::from(layout.data[DIMENSIONALITY_AT])
            * usize::from(layout.data[DIMENSION_WIDTH_AT])
        + INDEX_TYPE_LEN;
    assert_eq!(
        &layout.data[parameters_at..parameters_at + EA_PARAMETER_COUNT],
        &[
            array.header().max_index_bits().get(),
            array.header().idx_blk_elmts,
            array.header().super_blk_min_data_ptrs,
            array.header().min_dblk_nelmts,
            u8::try_from(array.header().page_nelmts().get().trailing_zeros()).unwrap(),
        ],
    );
    let index = array.index_block().unwrap();
    let mut ranges = vec![locator.parse_range().unwrap(), index.parse_range()];
    let index = index.parse(region).unwrap();
    assert!(index.super_blocks().next().is_none());
    let mut records = index
        .elements()
        .records(2048)
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    for block in index.data_blocks() {
        let block = block.unwrap();
        ranges.push(block.range());
        let ExtensibleArrayDataBlock::Elements(elements) = block.parse(region).unwrap() else {
            panic!("expected an unpaged data block")
        };
        records.extend(elements.records(2048).map(Result::unwrap));
    }
    assert!(ranges.len() > 2, "fixture includes an allocated data block");
    records.sort_by_key(|(slot, _)| *slot);
    assert_eq!(
        records.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(),
        (0..16).collect::<Vec<_>>()
    );
    let mut chunks = file.dataset("victim").unwrap().chunks().unwrap();
    chunks.sort_by(|left, right| left.offset.cmp(&right.offset));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.offset.clone())
            .collect::<Vec<_>>(),
        (0..16).map(|index| vec![index * 512]).collect::<Vec<_>>()
    );
    for ((_, record), chunk) in records.iter().zip(&chunks) {
        assert_eq!(
            file.resolve_address(record.address).unwrap().get(),
            chunk.address
        );
        assert_eq!(record.stored_size, 2048);
        assert_eq!(record.filter_mask, 0);
        assert_eq!(chunk.storage_size, 2048);
        assert_eq!(chunk.filter_mask().raw_bits(), 0);
        for metadata in &ranges {
            let start = metadata.start().get() / 4096;
            let end = (metadata.end().get() - 1) / 4096;
            let first_raw = chunk.address / 4096;
            let last_raw = (chunk.address + chunk.storage_size - 1) / 4096;
            assert!(end < first_raw || last_raw < start);
        }
    }
    let report = file.persisted_free_sections().unwrap().unwrap();
    for role in [
        FreeSpaceManagerRole::PageSmallMetadata,
        FreeSpaceManagerRole::PageSmallRawData,
    ] {
        let manager = report
            .managers()
            .iter()
            .find(|manager| manager.role() == role)
            .unwrap();
        assert!(!manager.free_sections().sections().is_empty());
    }
}

/// Checks `added`, `keep`, and the persisted page policy through the reference library.
///
/// `keep` must contain `[1, 2, 3, 4]` as [`i32`] values in `order`. `added` must
/// contain eight little-endian [`f64`] values equal to 2.5. Closes the file
/// before returning.
///
/// # Panics
///
/// Panics if opening, reading, or close fails, or any observation differs from the edited fixture.
#[cfg(feature = "__hdf5-1.10")]
pub fn assert_native_after_edit(path: &Path, order: NumericOrder) {
    let file = crate::file::libhdf5_open(path);
    let mut names = file.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["added", "keep"]);
    assert!(file.attr_names().unwrap().is_empty());
    {
        let keep = file.dataset("keep").unwrap();
        assert_eq!(keep.shape(), [4]);
        assert_eq!(
            keep.dtype().unwrap().to_descriptor().unwrap(),
            i32::type_descriptor()
        );
        assert_eq!(keep.dtype().unwrap().byte_order(), order.native_order());
        assert_eq!(keep.read_raw::<i32>().unwrap(), [1, 2, 3, 4]);
        let added = file.dataset("added").unwrap();
        assert_eq!(added.shape(), [8]);
        assert_eq!(
            added.dtype().unwrap().to_descriptor().unwrap(),
            f64::type_descriptor()
        );
        assert_eq!(
            added.dtype().unwrap().byte_order(),
            NativeByteOrder::LittleEndian
        );
        assert_eq!(added.read_raw::<f64>().unwrap(), [2.5; 8]);
        for dataset in [&keep, &added] {
            assert!(dataset.attr_names().unwrap().is_empty());
            assert!(dataset.filters().is_empty());
            assert_eq!(dataset.chunk(), None);
            assert_eq!(dataset.layout(), NativeLayout::Contiguous);
        }
    }
    let properties = file.fcpl().unwrap();
    assert_eq!(
        properties.file_space_strategy(),
        NativeStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 1
        }
    );
    assert_eq!(properties.file_space_page_size(), 4096);
    drop(properties);
    file.close().unwrap();
}

fn assert_persistence(file: &File) {
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert!(info.persist);
    assert_eq!(info.threshold, 1);
    assert_eq!(info.page_size.get(), 4096);
}

fn assert_root(file: &File, names: &[&str]) {
    let root = file.root();
    let mut datasets = root.datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, names);
    assert!(root.groups().unwrap().is_empty());
    assert!(root.named_datatypes().unwrap().is_empty());
    assert!(root.attrs().unwrap().is_empty());
}

fn assert_i32(file: &File, name: &str, values: &[i32], order: DatatypeByteOrder) {
    let dataset = file.dataset(name).unwrap();
    assert_eq!(dataset.shape().unwrap(), [values.len() as u64]);
    assert_eq!(
        dataset.datatype().unwrap(),
        Datatype::FixedPoint(FixedPointType::new(4, order, true, 0, 32).unwrap())
    );
    assert_eq!(dataset.read_i32().unwrap(), values);
    assert!(dataset.attrs().unwrap().is_empty());
    assert!(dataset.filter_pipeline().unwrap().is_empty());
}

// Section `subsec_fmt4_appendixc_extarr`, version 4.0.
const CLIENT_UNFILTERED: u8 = 0;

// Section `subsubsec_fmt4_dataobject_hdr_msg_layout`, version 4.0, defines the
// version 4 prefix, dimension fields and five extensible-array parameters.
const LAYOUT_PREFIX_LEN: usize = 5;
const DIMENSIONALITY_AT: usize = 3;
const DIMENSION_WIDTH_AT: usize = 4;
const INDEX_TYPE_LEN: usize = 1;
const EA_PARAMETER_COUNT: usize = 5;
