//! Content and storage observations for twelve indexed huge links under `g`.
//!
//! The helpers accept independent images from [`test_util::modern::huge_links_12`]
//! and native files with the same hierarchy.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util_hdf5::huge_links;
//!
//! let bytes = test_util::modern::huge_links_12();
//! let file = File::from_bytes(bytes.clone())?;
//! huge_links::assert_contents(&file, DatatypeByteOrder::LittleEndian);
//! huge_links::assert_storage(&file, &bytes);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

use std::collections::BTreeMap;

use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Record;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::HugeObjectReference;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::LinkMessage;
use hdf5_pure_format::__private::LinkTarget;
use hdf5_pure_format::__private::ObjectHeader;
use test_util::bytes;
use test_util::checksum;

use crate::storage;

/// Checks the complete hierarchy, numeric datatypes, values, and contiguous dataset storage.
///
/// Each dataset under `g` must contain its index as one [`i32`] value in `order`.
///
/// # Panics
///
/// Panics if metadata or values cannot be read, or any observation differs from the fixture.
pub fn assert_contents(file: &File, order: DatatypeByteOrder) {
    let root = file.root();
    assert_eq!(root.groups().unwrap(), ["g"]);
    assert!(root.datasets().unwrap().is_empty());
    assert!(root.named_datatypes().unwrap().is_empty());
    assert!(root.attrs().unwrap().is_empty());
    let group = file.group("g").unwrap();
    assert!(group.groups().unwrap().is_empty());
    assert!(group.named_datatypes().unwrap().is_empty());
    assert!(group.attrs().unwrap().is_empty());
    let mut names = group.datasets().unwrap();
    names.sort();
    let mut expected = link_names();
    expected.sort();
    assert_eq!(names, expected);
    for (index, name) in link_names().iter().enumerate() {
        let dataset = file.dataset(&format!("g/{name}")).unwrap();
        assert_eq!(dataset.shape().unwrap(), [1]);
        assert_eq!(
            dataset.datatype().unwrap(),
            Datatype::FixedPoint(FixedPointType::new(4, order, true, 0, 32).unwrap())
        );
        assert_eq!(dataset.read_i32().unwrap(), [i32::try_from(index).unwrap()]);
        assert!(dataset.attrs().unwrap().is_empty());
        assert!(dataset.filter_pipeline().unwrap().is_empty());
        assert_eq!(dataset.chunk_shape().unwrap(), None);
        let Layout::Contiguous {
            address: Some(_),
            size: 4,
        } = dataset.layout().unwrap()
        else {
            panic!("expected four contiguous bytes")
        };
    }
}

/// Checks indirect huge-object storage and the complete link-name index.
///
/// Pass the bytes from which `file` was opened. Checks twelve unfiltered huge objects,
/// seven-byte heap IDs, eleven-byte link-name records, and the correspondence between
/// each name hash, heap ID, and link message. The heap must have no managed blocks.
/// Addresses are read from the file.
///
/// # Panics
///
/// Panics if metadata cannot be read, or storage differs from the fixture.
pub fn assert_storage(file: &File, image: &[u8]) {
    let superblock = file.superblock();
    assert_eq!(superblock.base_address.get(), 0);
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let address = storage::root_object_address(file, image, "g");
    let header = ObjectHeader::parse_with_widths(
        image,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        widths,
    )
    .unwrap();
    assert!(
        !header
            .messages
            .iter()
            .any(|message| message.msg_type == MessageType::LINK)
    );
    let message = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    let info = LinkInfoMessage::parse_with_offset_width(&message.data, widths.offsets).unwrap();
    let heap = FractalHeapHeader::parse(
        image,
        usize::try_from(info.fractal_heap_address.unwrap().get()).unwrap(),
        widths.offsets.get(),
        widths.lengths.get(),
    )
    .unwrap();
    assert_eq!(heap.heap_id_length, GROUP_HEAP_ID_LENGTH);
    assert_eq!(heap.huge_objects_count, 12);
    assert_eq!(heap.managed_objects_count, 0);
    assert_eq!(heap.tiny_objects_count, 0);
    assert_eq!(heap.allocated_managed_space, 0);
    assert!(heap.root_block_address.is_undefined(widths.offsets));
    assert!(!heap.huge_ids_direct(widths.offsets.get(), widths.lengths.get()));
    let huge_index = BTreeV2Header::parse_with_widths(
        image,
        usize::try_from(heap.btree_huge_objects_address.get()).unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(huge_index.tree_type, format::BTREE_V2_HUGE_OBJECT);
    assert_eq!(
        huge_index.record_size,
        u16::from(widths.offsets.get()) + 2 * u16::from(widths.lengths.get())
    );
    let mut names = BTreeMap::new();
    let mut huge_bytes = 0;
    for record in leaf_records(image, &huge_index) {
        let offsets = usize::from(widths.offsets.get());
        let lengths = usize::from(widths.lengths.get());
        let address = bytes::uint_at(&record.data, 0, offsets);
        let size = bytes::uint_at(&record.data, offsets, lengths);
        let id = bytes::uint_at(&record.data, offsets + lengths, lengths);
        assert!(size > u64::from(heap.max_managed_object_size));
        huge_bytes += size;
        let link = LinkMessage::parse_with_offset_width(
            &image[usize::try_from(address).unwrap()..usize::try_from(address + size).unwrap()],
            widths.offsets,
        )
        .unwrap();
        assert!(matches!(link.link_target, LinkTarget::Hard { .. }));
        assert!(names.insert(id, link.name).is_none());
    }
    assert_eq!(heap.huge_objects_size, huge_bytes);
    let name_index = BTreeV2Header::parse_with_widths(
        image,
        usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(name_index.tree_type, LINK_NAME_INDEX);
    assert_eq!(name_index.record_size, NAME_RECORD_SIZE);
    let mut observed = Vec::new();
    for record in leaf_records(image, &name_index) {
        let hash = bytes::u32_at(&record.data, 0);
        let id = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths)
            .parse(&record.data[usize::from(NAME_HASH_SIZE)..])
            .unwrap();
        let FractalHeapIdKind::Huge(HugeObjectReference::Indexed(key)) = id.kind() else {
            panic!("expected an indirect unfiltered huge ID")
        };
        let id = key.value();
        let name = names.remove(&id).unwrap();
        assert_eq!(hash, checksum::lookup3(name.as_bytes()));
        observed.push((hash, name));
    }
    assert!(names.is_empty());
    assert!(observed.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    let mut observed: Vec<_> = observed.into_iter().map(|(_, name)| name).collect();
    observed.sort();
    let mut expected = link_names();
    expected.sort();
    assert_eq!(observed, expected);
}

/// Returns names `d0_` through `d11_`, each followed by 5,000 `x` characters.
pub fn link_names() -> Vec<String> {
    (0..12)
        .map(|index| format!("d{index}_{}", "x".repeat(5000)))
        .collect()
}

fn leaf_records(image: &[u8], header: &BTreeV2Header) -> Vec<BTreeV2Record> {
    assert_eq!(header.depth, 0);
    assert_eq!(header.total_records, 12);
    assert_eq!(header.num_records_in_root, 12);
    format::parse_btree_v2_leaf_records(
        image,
        usize::try_from(header.root_node_address.get()).unwrap(),
        header.num_records_in_root,
        header.record_size,
    )
    .unwrap()
}

// Section `subsubsec_fmt4_infra_btrees_v2`, version 4.0, defines type 5
// records with a four-byte hash and seven-byte heap ID.
const LINK_NAME_INDEX: u8 = 5;
const NAME_HASH_SIZE: u16 = 4;

const GROUP_HEAP_ID_LENGTH: u16 = 7;
const NAME_RECORD_SIZE: u16 = 11;
