//! Compressed legacy chunk observations through pure and native readers.
//!
//! Independent construction lives in [`test_util::legacy_chunked`]. Contents
//! assertions apply to both the source and its repacked image. Storage assertions
//! distinguish their chunk indexes and superblocks.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util_hdf5::legacy_chunked;
//!
//! let bytes = test_util::legacy_chunked::chunked_deflate();
//! let file = File::from_bytes(bytes.clone())?;
//! legacy_chunked::assert_contents(&file, DatatypeByteOrder::LittleEndian);
//! legacy_chunked::assert_source_storage(&file, &bytes);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

#[cfg(feature = "hdf5")]
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::H5Type;
#[cfg(feature = "hdf5")]
use hdf5::dataset::Layout as NativeLayout;
#[cfg(feature = "hdf5")]
use hdf5::datatype::ByteOrder;
#[cfg(feature = "hdf5")]
use hdf5::filters::Filter as NativeFilter;

use hdf5_pure::ChunkIndex;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FilterId;
use hdf5_pure::FilterParameters;
use hdf5_pure::FloatingPointType;
use hdf5_pure::MessageType;
use hdf5_pure::SuperblockVersion;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::BTreeV1ChunkNode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::LayoutVersion;
use hdf5_pure_format::__private::ObjectHeader;

use crate::symbol_table;

/// Checks the hierarchy, numeric values, chunk geometry, and deflate pipeline.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, byte_order: DatatypeByteOrder) {
    let root = file.root();
    assert!(root.groups().unwrap().is_empty());
    assert!(root.named_datatypes().unwrap().is_empty());
    assert!(root.attrs().unwrap().is_empty());
    assert_eq!(root.datasets().unwrap(), ["data"]);
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.shape().unwrap(), [100]);
    assert_eq!(
        dataset.datatype().unwrap(),
        Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(byte_order))
    );
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![20]));
    assert!(dataset.attrs().unwrap().is_empty());
    assert_eq!(
        dataset.read_f64().unwrap(),
        (0..100).map(f64::from).collect::<Vec<_>>()
    );
    let pipeline = dataset.filter_pipeline().unwrap();
    assert_eq!(pipeline.len(), 1);
    assert_eq!(pipeline[0].id(), FilterId::Deflate);
    assert_eq!(
        pipeline[0].parameters(),
        &FilterParameters::Deflate(DeflateLevel::Six)
    );
    assert!(pipeline[0].is_optional());
    assert_eq!(pipeline[0].name(), Some("deflate"));
    assert_eq!(pipeline[0].client_data(), [6]);
    let mut chunks = dataset.chunks().unwrap();
    chunks.sort_by(|a, b| a.offset.cmp(&b.offset));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.offset.clone())
            .collect::<Vec<_>>(),
        (0..5).map(|index| vec![index * 20]).collect::<Vec<_>>()
    );
    for chunk in chunks {
        assert_eq!(chunk.filter_mask().raw_bits(), 0);
        assert!(chunk.storage_size > 0 && chunk.storage_size < 160);
    }
}

/// Checks the version 0 superblock, symbol-table links, and version 1 chunk index.
///
/// Pass the bytes from which `file` was opened.
///
/// # Panics
///
/// Panics if metadata cannot be read, or the source storage differs from the fixture.
pub fn assert_source_storage(file: &File, image: &[u8]) {
    assert_eq!(file.superblock().version(), SuperblockVersion::Zero);
    assert_eq!(
        file.dataset("data").unwrap().chunk_index().unwrap(),
        Some(ChunkIndex::BTreeV1)
    );
    let tables = symbol_table::Fixture::new(file, image);
    let links = tables.links(file.superblock().root_group_address);
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].0, "data");
    let widths = FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    };
    let header = ObjectHeader::parse_with_widths(
        image,
        AccessMode::ReadOnly,
        usize::try_from(links[0].1.get()).unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(header.version, OBJECT_HEADER_VERSION);
    let layout = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    assert_eq!(layout.data[0], LayoutVersion::Three.get());
    let DataLayout::Chunked {
        chunk_dimensions,
        index: ChunkIndexLayout::BTreeV1 {
            address: Some(address),
        },
        ..
    } = DataLayout::parse_with_widths(&layout.data, widths).unwrap()
    else {
        panic!("expected an allocated version 1 chunk index")
    };
    assert_eq!(chunk_dimensions, [20, 8]);
    let node = BTreeV1ChunkNode::parse_with_offset_width(
        image,
        usize::try_from(address.get()).unwrap(),
        widths.offsets,
        2,
    )
    .unwrap();
    assert_eq!(node.node_level(), 0);
    assert_eq!(node.entries_used(), 5);
    assert_eq!(
        node.entries()
            .map(|entry| entry.unwrap().0.offsets().collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        (0..5).map(|index| vec![index * 20, 0]).collect::<Vec<_>>()
    );
}

/// Checks the version 3 superblock, fixed-array chunk index, and preserved contents.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or the repacked fixture differs from its
/// expected storage or contents.
pub fn assert_repacked(file: &File, byte_order: DatatypeByteOrder) {
    assert_eq!(file.superblock().version(), SuperblockVersion::Three);
    assert_eq!(
        file.dataset("data").unwrap().chunk_index().unwrap(),
        Some(ChunkIndex::FixedArray)
    );
    assert_contents(file, byte_order);
}

#[cfg(feature = "hdf5")]
/// Writes a native legacy chunked fixture with deflate and closes the file.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn write_libhdf5(path: &Path) {
    let file = crate::file::libhdf5_create_earliest(path);
    file.new_dataset_builder()
        .with_data(&(0..100).map(f64::from).collect::<Vec<_>>())
        .chunk([20])
        .deflate(6)
        .create("data")
        .unwrap();
    file.close().unwrap();
}

#[cfg(feature = "hdf5")]
/// Checks native dataset metadata, values, and the deflate pipeline.
///
/// # Panics
///
/// Panics if the file cannot be opened, metadata or values cannot be read,
/// any observation differs from the fixture, or close fails.
pub fn assert_libhdf5_contents(path: &Path, byte_order: ByteOrder) {
    let file = crate::file::libhdf5_open(path);
    assert_eq!(file.member_names().unwrap(), ["data"]);
    assert!(file.attr_names().unwrap().is_empty());
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.shape(), [100]);
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        f64::type_descriptor()
    );
    assert_eq!(dataset.dtype().unwrap().byte_order(), byte_order);
    assert_eq!(dataset.chunk(), Some(vec![20]));
    assert_eq!(dataset.layout(), NativeLayout::Chunked);
    assert_eq!(dataset.filters(), [NativeFilter::Deflate(6)]);
    assert!(dataset.attr_names().unwrap().is_empty());
    assert_eq!(
        dataset.read_raw::<f64>().unwrap(),
        (0..100).map(f64::from).collect::<Vec<_>>()
    );
    drop(dataset);
    file.close().unwrap();
}

// Section `subsubsec_fmt4_dataobject_hdr_prefix_one`, version 4.0.
const OBJECT_HEADER_VERSION: u8 = 1;
