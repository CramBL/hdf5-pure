#![cfg(feature = "__hdf5-1.10")]

use std::fs;

use hdf5::File as CFile;
use hdf5::plist::GroupCreate;
use hdf5::plist::file_access::LibraryVersion;
use hdf5_pure::File;
use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, PhysicalAddress};
use hdf5_pure_core::BaseAddress;
use hdf5_pure_format::__private::FilterDescription;
use hdf5_pure_format::__private::FilterPipeline;
use hdf5_pure_format::__private::FractalHeapFiltering;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::{
    BTREE_V2_LINK_NAME, BTreeV2Header, BTreeV2Layout, BTreeV2LinkNameIndex, FormatWidths,
    FractalHeapHugeObjectId, FractalHeapHugeObjectLayout, FractalHeapIdKind, FractalHeapIdLayout,
    HugeIndirectKeyEncoding, HugeObjectRecordContext, HugeObjectReference, HugeObjectTrackerRecord,
};
use hdf5_pure_format::__private::{FractalHeapHeaderLocator, LengthWidth, OffsetWidth};
use rstest::rstest;
use tempfile::TempDir;
use test_util_hdf5::group_filter;

#[rstest]
#[case::earliest(LibraryVersion::V18)]
#[case::latest(LibraryVersion::latest())]
fn a_c_written_filtered_link_heap_is_parsed_and_its_links_are_listed(
    #[case] bounds: LibraryVersion,
    #[values(0, 512)] userblock: u64,
) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("filtered.h5");
    let gcpl = GroupCreate::build()
        .obj_track_times(false)
        .finish()
        .unwrap();
    assert_eq!(group_filter::set_deflate(&gcpl, 6), 0);
    let file = CFile::with_options()
        .with_fapl(|p| p.libver_bounds(bounds, bounds))
        .with_fcpl(|p| p.userblock(userblock))
        .create(&path)
        .unwrap();
    let group = file
        .create_group_builder()
        .set_gcpl(&gcpl)
        .create("g")
        .unwrap();
    for i in 0..12 {
        group.create_group(&format!("child_{i:02}")).unwrap();
    }
    drop(group);
    file.close().unwrap();
    let expected_names: Vec<_> = (0..12).map(|i| format!("child_{i:02}")).collect();
    let bytes = fs::read(&path).unwrap();
    let header_offset = bytes
        .windows(FRACTAL_HEAP_SIGNATURE.len())
        .position(|window| window == FRACTAL_HEAP_SIGNATURE)
        .unwrap();
    let header = FractalHeapHeader::parse(&bytes, header_offset, 8, 8).unwrap();
    let FractalHeapFiltering::Filtered {
        root_direct_block_size,
        root_filter_mask,
        pipeline,
    } = &header.filtering
    else {
        panic!("expected filtered heap, got {:?}", header.filtering);
    };
    assert!(*root_direct_block_size > 0);
    assert!(*root_direct_block_size < header.starting_block_size);
    assert_eq!(*root_filter_mask, 0);
    assert_eq!(
        *pipeline,
        FilterPipeline {
            version: PIPELINE_VERSION_TWO,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: FILTER_OPTIONAL,
                client_data: vec![6],
            }],
        }
    );
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        &bytes,
    )
    .unwrap();
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(header_offset as u64),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    assert_eq!(locator.parse(region).unwrap().parse(region), Ok(header));
    let c_file = CFile::open(&path).unwrap();
    let mut names = c_file.group("g").unwrap().member_names().unwrap();
    names.sort();
    assert_eq!(names, expected_names);
    c_file.close().unwrap();
    let mut buffered = File::open(&path)
        .unwrap()
        .group("g")
        .unwrap()
        .groups()
        .unwrap();
    buffered.sort();
    assert_eq!(buffered, expected_names);
    let mut streaming = File::open_streaming(&path)
        .unwrap()
        .group("g")
        .unwrap()
        .groups()
        .unwrap();
    streaming.sort();
    assert_eq!(streaming, expected_names);
}

#[test]
fn a_c_written_filtered_huge_object_has_a_type_two_record() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("filtered_huge.h5");
    let gcpl = GroupCreate::build()
        .obj_track_times(false)
        .finish()
        .unwrap();
    assert_eq!(group_filter::set_deflate(&gcpl, 6), 0);
    let file = CFile::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()))
        .create(&path)
        .unwrap();
    let group = file
        .create_group_builder()
        .set_gcpl(&gcpl)
        .create("g")
        .unwrap();
    for index in 0..12 {
        group
            .link_soft("/unresolved", &format!("small_{index:02}"))
            .unwrap();
    }
    group
        .link_soft(&format!("/{}", "x".repeat(12_000)), "huge")
        .unwrap();
    drop(group);
    file.close().unwrap();
    let image = fs::read(&path).unwrap();
    let header_offset = test_util::fractal_heap::header_offsets(&image)[0];
    let heap = FractalHeapHeader::parse(&image, header_offset, 8, 8).unwrap();
    assert_eq!(heap.huge_objects_count, 1);
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    let tree = BTreeV2Header::parse_with_widths(
        &image,
        usize::try_from(heap.btree_huge_objects_address.get()).unwrap(),
        widths,
    )
    .unwrap();
    let records = HugeObjectRecordContext::new(&heap, widths).unwrap();
    assert_eq!(tree.tree_type, records.tree_type());
    assert_eq!(tree.record_size, records.record_size());
    let decoder = records.resolve(&tree).unwrap();
    let layout = BTreeV2Layout::new(&tree, widths.offsets).unwrap();
    let root = layout.root().unwrap();
    let view = layout
        .parse(
            &image[usize::try_from(root.address().get()).unwrap()..],
            root,
        )
        .unwrap();
    let HugeObjectTrackerRecord::FilteredIndirect(record) =
        decoder.parse(view.record(0).unwrap()).unwrap()
    else {
        panic!("expected filtered indirect huge record")
    };
    assert_eq!(record.id(), 1);
    assert_eq!(record.location().decoded_size(), heap.huge_objects_size);
    assert!(record.location().encoded_size() < record.location().decoded_size());
    assert_eq!(record.location().filter_mask(), 0);
    let objects = FractalHeapHugeObjectLayout::new(
        &heap,
        widths,
        BaseAddress::ZERO,
        HugeIndirectKeyEncoding::Libhdf5,
    )
    .unwrap();
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), image.len() as u64).unwrap(),
        &image,
    )
    .unwrap();
    let tracker = objects.parse_tracker(region).unwrap();
    let payload_record = tracker.parse_record(view.record(0).unwrap()).unwrap();
    let name_index = image
        .windows(BTREE_SIGNATURE.len())
        .enumerate()
        .find_map(|(offset, signature)| {
            if signature != BTREE_SIGNATURE {
                return None;
            }
            BTreeV2Header::parse_with_widths(&image, offset, widths)
                .ok()
                .filter(|header| header.tree_type == BTREE_V2_LINK_NAME)
        })
        .unwrap();
    let names = BTreeV2Layout::new(&name_index, widths.offsets).unwrap();
    let root = names.root().unwrap();
    let view = names
        .parse(
            &image[usize::try_from(root.address().get()).unwrap()..],
            root,
        )
        .unwrap();
    let link_records =
        BTreeV2LinkNameIndex::new(&name_index, widths.offsets, heap.heap_id_length).unwrap();
    let huge_id = (0..view.record_count())
        .map(|index| link_records.record(view.record(index).unwrap()).unwrap().1)
        .find(|id| id.first() == Some(&HUGE_ID_FIRST_BYTE))
        .unwrap();
    let parsed = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths)
        .parse(huge_id)
        .unwrap()
        .kind();
    let FractalHeapIdKind::Huge(HugeObjectReference::FilteredIndexed(key)) = parsed else {
        panic!("expected filtered indirect huge ID, got {parsed:?}")
    };
    assert_eq!(key.value(), record.id());
    let FractalHeapHugeObjectId::Indirect(lookup) = objects.parse_id(huge_id).unwrap() else {
        panic!("expected filtered indirect huge lookup")
    };
    assert_eq!(lookup.tracker_range(), objects.tracker_range());
    let reference = objects.resolve(&lookup, &payload_record).unwrap();
    assert_eq!(
        reference.range().start().get(),
        record.location().address().get()
    );
    assert_eq!(reference.range().length(), record.location().encoded_size());
    assert_eq!(reference.decoded_size(), heap.huge_objects_size);
    assert_eq!(reference.filters().unwrap().mask, 0);
    assert_eq!(
        reference.parse(region).unwrap().len() as u64,
        record.location().encoded_size()
    );
}

// "Fractal Heap", format specification version 4.0, defines the header signature.
const FRACTAL_HEAP_SIGNATURE: &[u8; 4] = b"FRHP";
// "The Data Storage - Filter Pipeline Message", format specification version 4.0, defines the
// versions and filter fields.
const PIPELINE_VERSION_TWO: u8 = 2;
const FILTER_DEFLATE: u16 = 1;
const FILTER_OPTIONAL: u16 = 0x0001;

/// The version 2 B-tree header signature from "Version 2 B-trees" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
const BTREE_SIGNATURE: &[u8; 4] = b"BTHD";
/// The version-0 huge-ID type byte from "Fractal Heap" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const HUGE_ID_FIRST_BYTE: u8 = 0x10;
