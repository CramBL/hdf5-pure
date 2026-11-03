#![cfg(feature = "__hdf5-1.10")]

use std::fs;
use std::ops::Range;
use std::path::Path;
use std::slice;

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5_pure::File;
use hdf5_pure::MessageType;
use hdf5_pure::PhysicalAddress;
use hdf5_pure::StoredAddress;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::ExtensibleArray;
use hdf5_pure_format::__private::ExtensibleArrayBlockKind;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use hdf5_pure_format::__private::ExtensibleArrayInitialization;
use hdf5_pure_format::__private::ExtensibleArrayPlacement;
use hdf5_pure_format::__private::ExtensibleArrayPlan;
use hdf5_pure_format::__private::ExtensibleArrayStats;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::LengthWidth;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::OffsetWidth;
use hdf5_pure_format::__private::SlotOccupancy;
use rstest::rstest;
use test_util::bytes;
use test_util::extensible_array;
use test_util::extensible_array::HeaderStats as NativeHeaderStats;
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;
use test_util_hdf5::chunk_filter;
use test_util_hdf5::numeric::ByteOrder;

use crate::test_util::array_index;
use crate::test_util::array_index::Writes;

#[rstest]
#[case::empty(Fixture::new([32, 4, 4, 16, 10], 0, None, 4, false))]
#[case::empty_nondefault(Fixture::new([24, 8, 2, 8, 8], 0, None, 8, false))]
#[case::inline(Fixture::new([32, 4, 4, 16, 10], 4, None, 1, false))]
#[case::direct(Fixture::new([32, 4, 4, 16, 10], 52, Some(&[0..1, 20..21, 51..52]), 4, false))]
#[case::secondary(Fixture::new([24, 8, 2, 8, 8], 128, Some(&[0..1, 32..33, 63..64, 100..101]), 48, false))]
#[case::wide_offsets(Fixture::new([64, 4, 4, 16, 10], 300, Some(&[0..1, 244..245, 299..300]), 245, false))]
#[case::paged(Fixture::new([16, 2, 2, 8, 5], 700, Some(&[506..507, 546..547, 666..667]), 634, true))]
#[case::canonical_paged(Fixture::new([32, 4, 4, 16, 10], 131_061, Some(slice::from_ref(&CANONICAL_PAGED_RANGE)), 132_084, true))]
fn native_reads_and_mutates_independently_placed_extensible_array_allocations(
    #[case] fixture: Fixture,
    #[values((Sizeof::Bytes2, OffsetWidth::Two), (Sizeof::Bytes4, OffsetWidth::Four), (Sizeof::Bytes8, OffsetWidth::Eight))]
    widths: (Sizeof, OffsetWidth),
    #[values(0, 512)] userblock: u64,
    #[values(false, true)] filtered: bool,
) {
    let Fixture {
        creation,
        writes,
        mutation,
        paged,
    } = fixture;
    let len = writes.len;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("placed-extensible-array.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties
                .sizes(SizeofInfo {
                    sizeof_addr: widths.0,
                    sizeof_size: Sizeof::Bytes8,
                })
                .userblock(userblock)
        })
        .create(&path)
        .unwrap();
    let builder = file
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .shape([Extent::resizable(len)])
        .chunk([1]);
    let builder = if filtered {
        builder.shuffle().deflate(6)
    } else {
        builder
    };
    let dataset = builder.create("d").unwrap();
    assert_eq!(dataset.num_chunks().unwrap(), 0);
    drop(dataset);
    drop(builder);
    file.close().unwrap();
    let widths = FormatWidths {
        offsets: widths.1,
        lengths: LengthWidth::Eight,
    };
    let mut image = fs::read(&path).unwrap();
    let mut layout = LayoutRecord::find(&image, widths);
    layout.set_creation(&mut image, creation);
    fs::write(&path, &image).unwrap();
    if len > 0 {
        let file = hdf5::File::open_rw(&path).unwrap();
        let dataset = file.dataset("d").unwrap();
        writes.write(&dataset);
        assert_eq!(dataset.read_raw::<i32>().unwrap(), writes.expected());
        drop(dataset);
        file.close().unwrap();
    }
    replace_index(&path, userblock, 1, creation.encode(), filtered, paged);
    let mut expected = writes.expected();
    let final_len = (len + 2).max(mutation + 1);
    expected.resize(final_len, 0);
    expected[len..len + 2].copy_from_slice(&[83, 89]);
    expected[mutation] = 97;
    let native = hdf5::File::open_rw(&path).unwrap();
    let dataset = native.dataset("d").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), writes.expected());
    dataset.resize([final_len]).unwrap();
    dataset.write_slice(&[83i32, 89], len..len + 2).unwrap();
    dataset
        .write_slice(&[97i32], mutation..mutation + 1)
        .unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), expected);
    let dense = dataset.num_chunks().unwrap() == final_len;
    drop(dataset);
    native.close().unwrap();
    let pure = File::open(&path).unwrap();
    assert_eq!(pure.dataset("d").unwrap().read_i32().unwrap(), expected);
    pure.close().unwrap();
    if widths.offsets == OffsetWidth::Eight && (userblock == 0 || dense) {
        let pure = File::open_rw(&path).unwrap();
        let mut dataset = pure.dataset("d").unwrap();
        if userblock == 0 {
            dataset.append(&[101i32]).unwrap();
        } else {
            dataset
                .append_staged(|append| {
                    append.append(&[101i32]);
                })
                .unwrap();
        }
        expected.push(101);
        drop(dataset);
        if userblock != 0 {
            pure.commit().unwrap();
        }
        pure.close().unwrap();
    }
    let native = hdf5::File::open(&path).unwrap();
    let dataset = native.dataset("d").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), expected);
    drop(dataset);
    native.close().unwrap();
}

#[test]
fn independent_placement_preserves_native_skipped_filter_masks_and_stored_lengths() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("masked-extensible-array.h5");
    chunk_filter::write_masked_chunks(&path, 0b10, &[[1, 2, 3, 4], [5, 6, 7, 8]]);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| (chunk.filter_mask(), chunk.size()))
            .collect::<Vec<_>>(),
        [(2, 16), (2, 16)]
    );
    replace_index(&path, 0, 4, [32, 4, 4, 16, 10], true, false);
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
    drop(dataset);
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
struct Fixture {
    creation: ExtensibleArrayCreationParameters,
    writes: Writes,
    mutation: usize,
    paged: bool,
}

impl Fixture {
    fn new(
        creation: [u8; 5],
        len: usize,
        sparse: Option<&'static [Range<usize>]>,
        mutation: usize,
        paged: bool,
    ) -> Self {
        Self {
            creation: ExtensibleArrayCreationParameters::parse(&creation).unwrap(),
            writes: Writes { len, sparse },
            mutation,
            paged,
        }
    }
}

fn replace_index(
    path: &Path,
    userblock: u64,
    chunk: u64,
    creation: [u8; 5],
    filtered: bool,
    paged: bool,
) {
    let original = array_index::libhdf5_chunks(path, userblock);
    let mut image = fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(array_index::pure_chunks(&file), original);
    let widths = FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    };
    let base = BaseAddress::new(file.resolve_address(StoredAddress::ZERO).unwrap().get());
    let mut superblock = file.superblock().clone();
    let mut layout = LayoutRecord::find(&image, widths);
    let FormatWidths { offsets, lengths } = widths;
    let DataLayout::Chunked {
        index:
            ChunkIndexLayout::ExtensibleArray {
                address,
                creation: parsed,
            },
        ..
    } = DataLayout::parse(&layout.body, offsets.get(), lengths.get()).unwrap()
    else {
        panic!("expected Extensible Array layout")
    };
    assert_eq!(parsed.encode(), creation);
    let records: Vec<_> = original
        .iter()
        .map(|chunk| ChunkRecord {
            address: file
                .relative_address(PhysicalAddress::new(chunk.address()))
                .unwrap(),
            stored_size: chunk.size(),
            filter_mask: chunk.filter_mask(),
        })
        .collect();
    let slots: Vec<_> = original
        .iter()
        .map(|record| record.offset()[0] / chunk)
        .collect();
    let source = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(image.len()).unwrap()).unwrap(),
        &image,
    )
    .unwrap();
    let populated;
    let empty;
    let plan = if let Some(address) = address {
        let array = ExtensibleArrayHeaderLocator::new(widths, base, address)
            .parse(source)
            .unwrap();
        populated = ExtensibleArrayPlan::new(
            parsed,
            SlotOccupancy::Slots(&slots),
            array.header().max_idx_set,
            4 * chunk,
            widths,
            filtered,
        )
        .unwrap();
        assert_eq!(array.is_filtered(), filtered);
        let observed = geometry(source, array, paged);
        assert_plan(&observed, &populated);
        let NativeHeaderStats {
            secondary_block_count,
            secondary_block_size,
            data_block_count,
            data_block_size,
            max_index_set,
            element_count,
        } = extensible_array::header_stats(path);
        let ExtensibleArrayStats {
            nsuper_blks,
            super_blk_size,
            ndata_blks,
            data_blk_size,
            max_idx_set,
            nelmts,
        } = populated.stats();
        assert_eq!(
            [
                nsuper_blks,
                super_blk_size,
                ndata_blks,
                data_blk_size,
                max_idx_set,
                nelmts
            ],
            [
                secondary_block_count,
                secondary_block_size,
                data_block_count,
                data_block_size,
                max_index_set,
                element_count
            ]
        );
        &populated
    } else {
        assert!(original.is_empty());
        empty = ExtensibleArrayInitialization::new(parsed, 4 * chunk, widths, filtered).unwrap();
        empty.plan()
    };
    let slots = IndexSlots::new(&records, &slots, plan.stats().max_idx_set).unwrap();
    let mut cursor = file
        .relative_address(PhysicalAddress::new(file.file_size()))
        .unwrap();
    let lengths: Vec<_> = plan.block_lengths().collect();
    let mut addresses = vec![StoredAddress::ZERO; lengths.len()];
    for index in (0..lengths.len()).rev() {
        addresses[index] = cursor;
        cursor = cursor.checked_add(lengths[index]).unwrap();
        if index != 0 {
            cursor = cursor.checked_add(17).unwrap();
        }
    }
    let placement = plan.place(&addresses).unwrap();
    assert!(addresses.windows(2).all(|pair| pair[0] > pair[1]));
    let encoding = placement.encoding(&slots).unwrap();
    let end = file.resolve_address(cursor).unwrap().get();
    let ranges: Vec<_> = placement
        .blocks()
        .map(|(address, length)| {
            FileRange::new(file.resolve_address(address).unwrap(), length).unwrap()
        })
        .collect();
    superblock.eof_address = end;
    file.close().unwrap();
    image.resize(usize::try_from(end).unwrap(), 0);
    let mut output = Vec::new();
    for (emission, range) in encoding.blocks().zip(&ranges) {
        emission.encode_into(&mut output).unwrap();
        assert_eq!(u64::try_from(output.len()).unwrap(), range.length());
        image[usize::try_from(range.start().get()).unwrap()
            ..usize::try_from(range.end().get()).unwrap()]
            .copy_from_slice(&output);
    }
    layout.set_index(&mut image, &placement);
    let encoded = format::serialize_superblock(&superblock).unwrap();
    let at = usize::try_from(userblock).unwrap();
    image[at..at + encoded.len()].copy_from_slice(&encoded);
    let source = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(image.len()).unwrap()).unwrap(),
        &image,
    )
    .unwrap();
    let array = ExtensibleArrayHeaderLocator::new(widths, base, addresses[0])
        .parse(source)
        .unwrap();
    let observed = geometry(source, array, paged);
    assert_plan(&observed, plan);
    let mut actual: Vec<_> = observed.iter().map(|block| block.allocation).collect();
    actual.sort_by_key(|range| range.start());
    let mut expected = ranges;
    expected.sort_by_key(|range| range.start());
    assert_eq!(actual, expected);
    fs::write(path, &image).unwrap();
    let native = hdf5::File::open(path).unwrap();
    assert_eq!(
        native.dataset("d").unwrap().num_chunks().unwrap(),
        original.len()
    );
    native.close().unwrap();
    assert_eq!(array_index::libhdf5_chunks(path, userblock), original);
    let pure = File::open(path).unwrap();
    assert_eq!(array_index::pure_chunks(&pure), original);
    pure.close().unwrap();
}

fn geometry(source: FileRegion<'_>, array: ExtensibleArray, paged: bool) -> Vec<ObservedBlock> {
    let index = array.index_block().unwrap();
    let mut result = vec![
        ObservedBlock {
            kind: ExtensibleArrayBlockKind::Header,
            slots: None,
            allocation: array.range(),
        },
        ObservedBlock {
            kind: ExtensibleArrayBlockKind::Index,
            slots: Some(0..u64::from(array.header().idx_blk_elmts)),
            allocation: index.range(),
        },
    ];
    let index = index.parse(source).unwrap();
    let mut data = index.data_blocks().collect::<Result<Vec<_>, _>>().unwrap();
    for reference in index.super_blocks() {
        let reference = reference.unwrap();
        result.push(ObservedBlock {
            kind: ExtensibleArrayBlockKind::Secondary,
            slots: Some(reference.slots().unwrap()),
            allocation: reference.parse_range(),
        });
        data.extend(
            reference
                .parse(source)
                .unwrap()
                .data_blocks()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
        );
    }
    let mut has_pages = false;
    for reference in data {
        reference.parse(source).unwrap();
        if let Some(pages) = reference.pages().unwrap() {
            has_pages = true;
            assert!(reference.parse_range().length() < reference.range().length());
            for page in pages.pages() {
                let page = page.unwrap();
                assert!(reference.range().contains(page.parse_range()));
                page.parse(source).unwrap();
            }
        }
        result.push(ObservedBlock {
            kind: ExtensibleArrayBlockKind::Data,
            slots: Some(reference.slots().unwrap()),
            allocation: reference.range(),
        });
    }
    assert_eq!(has_pages, paged);
    result
}

fn assert_plan(observed: &[ObservedBlock], plan: &ExtensibleArrayPlan) {
    let mut actual: Vec<_> = observed
        .iter()
        .map(
            |ObservedBlock {
                 kind,
                 slots,
                 allocation,
             }| (key(*kind, slots), allocation.length()),
        )
        .collect();
    let mut expected: Vec<_> = plan
        .blocks()
        .map(|block| (key(block.kind(), &block.slots()), block.length()))
        .collect();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

fn key(kind: ExtensibleArrayBlockKind, slots: &Option<Range<u64>>) -> (u8, u64, u64) {
    let kind = match kind {
        ExtensibleArrayBlockKind::Header => 0,
        ExtensibleArrayBlockKind::Index => 1,
        ExtensibleArrayBlockKind::Secondary => 2,
        ExtensibleArrayBlockKind::Data => 3,
    };
    (
        kind,
        slots.as_ref().map_or(0, |slots| slots.start),
        slots.as_ref().map_or(0, |slots| slots.end),
    )
}

struct ObservedBlock {
    kind: ExtensibleArrayBlockKind,
    slots: Option<Range<u64>>,
    allocation: FileRange,
}

struct LayoutRecord {
    header: usize,
    body: Vec<u8>,
    widths: FormatWidths,
}

impl LayoutRecord {
    fn find(image: &[u8], widths: FormatWidths) -> Self {
        let FormatWidths { offsets, lengths } = widths;
        let mut candidates: Vec<_> = bytes::signature_offsets(image, v2::SIGNATURE)
            .into_iter()
            .filter_map(|header| {
                ObjectHeader::parse(
                    image,
                    AccessMode::ReadOnly,
                    header,
                    offsets.get(),
                    lengths.get(),
                )
                .unwrap()
                .messages
                .into_iter()
                .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
                .map(|message| (header, message.data))
            })
            .collect();
        assert_eq!(candidates.len(), 1);
        let (header, body) = candidates.pop().unwrap();
        let parsed = DataLayout::parse(&body, offsets.get(), lengths.get()).unwrap();
        assert!(
            matches!(
                parsed,
                DataLayout::Chunked {
                    index: ChunkIndexLayout::ExtensibleArray { .. },
                    ..
                }
            ),
            "{parsed:?}"
        );
        Self {
            header,
            body,
            widths,
        }
    }

    // The five EA creation fields precede the index address in "The Data Layout Message" of
    // the format specification, version 4.0.
    fn set_creation(&mut self, image: &mut [u8], creation: ExtensibleArrayCreationParameters) {
        let Self {
            header,
            body,
            widths,
        } = self;
        let creation = creation.encode();
        let address = body.len() - usize::from(widths.offsets.get());
        body[address - creation.len()..address].copy_from_slice(&creation);
        v2::replace_message(image, *header, FixtureMessage::DATA_LAYOUT, body);
    }

    // For Extensible Array indexing, the index address is the final field in "The Data Layout
    // Message" of the format specification, version 4.0.
    fn set_index(&mut self, image: &mut [u8], placement: &ExtensibleArrayPlacement<'_>) {
        let Self {
            header,
            body,
            widths,
        } = self;
        assert_eq!(placement.plan().widths(), *widths);
        let address = placement.blocks().next().unwrap().0;
        let width = usize::from(widths.offsets.get());
        let start = body.len() - width;
        body[start..].copy_from_slice(&address.get().to_le_bytes()[..width]);
        v2::replace_message(image, *header, FixtureMessage::DATA_LAYOUT, body);
    }
}

const CANONICAL_PAGED_RANGE: Range<usize> = 131_060..131_061;
