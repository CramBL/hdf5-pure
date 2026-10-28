use hdf5_pure::ChunkIndex;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_core::MaxExtent;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::Dataspace;
use hdf5_pure_format::__private::ExtensibleArrayDataBlock;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::ObjectHeader;

use crate::storage;

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
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert!(info.persist);
    assert_eq!(info.threshold, 1);
    assert_eq!(info.page_size.get(), 4096);
}

pub fn assert_after_edit(file: &File, order: DatatypeByteOrder) {
    assert_root(file, &["added", "keep"]);
    assert_i32(file, "keep", &[1, 2, 3, 4], order);
    let added = file.dataset("added").unwrap();
    assert_eq!(added.shape().unwrap(), [8]);
    assert_eq!(
        added.datatype().unwrap(),
        Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
            DatatypeByteOrder::LittleEndian
        ))
    );
    assert_eq!(added.read_f64().unwrap(), [2.5; 8]);
    assert!(added.attrs().unwrap().is_empty());
    assert!(added.filter_pipeline().unwrap().is_empty());
    assert_eq!(added.chunk_shape().unwrap(), None);
    assert!(matches!(
        added.layout().unwrap(),
        Layout::Contiguous {
            address: Some(_),
            size: 64
        }
    ));
}

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
        usize::try_from(address).unwrap(),
        widths,
        superblock.base_address,
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
        index: ChunkIndexLayout::ExtensibleArray {
            address: Some(address),
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
    let locator = ExtensibleArrayHeaderLocator::new(widths, superblock.base_address, address);
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
            record.address.get() + superblock.base_address.get(),
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
