use std::io::ErrorKind;

use hdf5_pure_format::__private::FixedArrayHeader;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use hdf5_pure_format::__private::JenkinsLookup3;
use rstest::rstest;
use test_util::checksum;
use test_util::filtered_chunks::PipelinePattern;
use test_util::superblock::v2 as superblock_bytes;

use crate::File;
use crate::Layout;
use crate::PhysicalAddress;
use crate::edit::LocatedState;

use super::*;

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn fixed_array_appends_preserve_finite_values_and_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] empty: bool,
) {
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(offsets)
        .with_length_width(lengths);
    let initial = if empty { &[][..] } else { &[0][..] };
    let dataset = builder
        .create_dataset("d")
        .with_i32_data(initial)
        .with_shape(&[initial.len().to_u64()])
        .with_maxshape(&[MaxExtent::Fixed(16)])
        .with_chunks(&[4]);
    if filtered {
        dataset.with_deflate(DeflateLevel::One);
    }
    let mut store = WindowProbeStore::open(builder.finish().unwrap());
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let values = (i32::try_from(initial.len()).unwrap()..6).collect::<Vec<_>>();
    append(&mut store, &mut state, &values, 4);
    append(&mut store, &mut state, &[6, 7, 8], 4);
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..9).collect::<Vec<_>>()
    );
    assert_eq!(file.superblock().offset_size, offsets);
    assert_eq!(file.superblock().length_size, lengths);
    file.close().unwrap();
}

#[rstest]
#[case::eight_pages(8)]
#[case::many_pages(1_024)]
fn fixed_array_initializes_page_bits_in_one_bounded_batch(#[case] pages: usize) {
    let entries = pages * 4;
    let mut store = fixture_with_geometry(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
        FixedSource::SparsePages,
        entries,
        2,
    );
    let address = dataset_address(&store);
    let mut state = edit::locate_dataset_state(&store, address).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let array = header(&store);
    let block = array.data_block().unwrap();
    let bitmap_body = block.checksummed_range().unwrap();
    let bitmap_checksum = block.checksum_range().unwrap();
    let values = (3..i32::try_from(entries * 4).unwrap()).collect::<Vec<_>>();
    store.reset_counters();
    store.writes.clear();
    let plan = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            None,
            LossyTailReencoding::Reject,
            &i32_bytes(&values),
            NonZeroU64::new(values.len().to_u64()).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    index.apply(&mut store, plan, 4).unwrap();
    let bitmap_reads = store
        .reads
        .borrow()
        .iter()
        .filter(|(offset, length)| {
            *offset < bitmap_body.end().get()
                && *offset + length.to_u64() > bitmap_body.start().get()
        })
        .map(|(_, length)| *length)
        .sum::<usize>();
    assert!(bitmap_reads <= pages * 3 + 128, "{pages}: {bitmap_reads}");
    assert!(store.max_read.get() <= 4_096);
    let bitmap_writes = store
        .writes
        .iter()
        .copied()
        .filter(|(offset, _)| {
            *offset >= bitmap_body.start().get() && *offset < bitmap_checksum.end().get()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        bitmap_writes,
        [
            (
                bitmap_body.end().get() - pages.div_ceil(8).to_u64(),
                pages.div_ceil(8)
            ),
            (bitmap_checksum.start().get(), checksum::CHECKSUM)
        ]
    );
    store.reset_counters();
    let mut scratch = [0; 4_096];
    for _ in 0..pages {
        let mut position = bitmap_body.start();
        let mut hash = JenkinsLookup3::new(bitmap_body.length());
        while position < bitmap_body.end() {
            let length = (bitmap_body.end().get() - position.get())
                .min(scratch.len().to_u64())
                .to_usize()
                .unwrap();
            store
                .read_at(position.get(), &mut scratch[..length])
                .unwrap();
            hash.update(&scratch[..length]).unwrap();
            position = position.checked_add(length.to_u64()).unwrap();
        }
        assert_eq!(hash.finish().unwrap(), {
            let bytes = &store.data[bitmap_checksum.start().get().to_usize().unwrap()
                ..bitmap_checksum.end().get().to_usize().unwrap()];
            u32::from_le_bytes(bytes.try_into().unwrap())
        });
    }
    assert_eq!(
        store.total_read.get(),
        pages * bitmap_body.length().to_usize().unwrap()
    );
    assert!(store.total_read.get() > bitmap_reads);
    eprintln!(
        "{pages} pages: {bitmap_reads} bitmap bytes read, {} control bytes read, {} bitmap writes",
        store.total_read.get(),
        bitmap_writes.len()
    );
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        [0; 3].into_iter().chain(values).collect::<Vec<_>>()
    );
    file.close().unwrap();
}

#[test]
fn public_fixed_array_appends_use_the_session_store() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixed.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(16)])
        .with_chunks(&[4]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    dataset.append(&[1_i32, 2, 3, 4, 5]).unwrap();
    dataset.append(&[6_i32, 7, 8]).unwrap();
    assert_eq!(dataset.read_i32().unwrap(), (0..9).collect::<Vec<_>>());
    drop(dataset);
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..9).collect::<Vec<_>>()
    );
    file.close().unwrap();
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn fixed_array_interruptions_preserve_visible_values_and_retirement(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(
        FixedSource::Allocated,
        FixedSource::Unallocated,
        FixedSource::HeaderOnly,
        FixedSource::SparsePages
    )]
    source: FixedSource,
    #[values(false, true)] filtered: bool,
    #[values(1, 2, 3, 4)] phase: u8,
) {
    let mut store = fixture(FormatWidths { offsets, lengths }, filtered, source);
    let initial = match source {
        FixedSource::Allocated => vec![0],
        FixedSource::SparsePages => vec![0; 3],
        _ => Vec::new(),
    };
    let end = if matches!(source, FixedSource::HeaderOnly | FixedSource::SparsePages) {
        21
    } else {
        9
    };
    let values = (i32::try_from(initial.len()).unwrap()..end).collect::<Vec<_>>();
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    append(&mut store, &mut state, &values, phase);
    let expected = if phase == 4 {
        initial
            .iter()
            .copied()
            .chain(values.iter().copied())
            .collect::<Vec<_>>()
    } else {
        initial
    };
    let file = File::from_bytes(store.data.clone()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
    file.close().unwrap();
    assert_eq!(
        store.retired.len(),
        usize::from(phase >= 2 && matches!(source, FixedSource::Allocated))
    );
    let mut store = WindowProbeStore::open(store.data);
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    append(&mut store, &mut state, &[99], 4);
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        expected.into_iter().chain([99]).collect::<Vec<_>>()
    );
    file.close().unwrap();
}

#[rstest]
#[case::content(1)]
#[case::eof(2)]
#[case::reference(3)]
#[case::dimension(4)]
fn fixed_array_barrier_failures_preserve_live_allocations(#[case] barrier: usize) {
    let mut store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
        FixedSource::Allocated,
    );
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let plan = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            None,
            LossyTailReencoding::Reject,
            &i32_bytes(&[1, 2, 3, 4]),
            NonZeroU64::new(4).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    store.reset_counters();
    store.fail_sync = Some(barrier);
    let error = index.apply(&mut store, plan, 4).unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected an I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected synchronization failure");
    assert_eq!(store.retired.len(), usize::from(barrier == 4));
    let error = match store.allocate(&[0], AllocationClass::Raw) {
        Ok(_) => panic!("uncertain storage accepts an allocation"),
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        if barrier == 4 {
            vec![0, 1, 2, 3, 4]
        } else {
            vec![0]
        }
    );
    file.close().unwrap();
}

#[test]
fn fixed_array_plans_reject_a_different_session_before_io() {
    let store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
        FixedSource::Allocated,
    );
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let plan = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            None,
            LossyTailReencoding::Reject,
            &i32_bytes(&[1]),
            NonZeroU64::new(1).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    let mut other = WindowProbeStore::open(store.data.clone());
    let mut other_state = edit::locate_dataset_state(&other, dataset_address(&other)).unwrap();
    let AppendIndex::Fixed(other_index) = &mut other_state.index else {
        panic!("expected a Fixed Array index")
    };
    let before = other.data.clone();
    other.reset_counters();
    let error = other_index.apply(&mut other, plan, 4).unwrap_err();
    assert!(
        matches!(
            error,
            Error::AppendUnsupported("append plan refers to a different editing session")
        ),
        "{error:?}"
    );
    assert_eq!(other.total_read.get(), 0);
    assert_eq!(other.data, before);
    assert_eq!(*other.allocations.borrow(), []);
    assert_eq!(other.retired, []);
}

#[rstest]
#[case::root(HiddenOverlap::Root)]
#[case::shared_chunk(HiddenOverlap::SharedChunk)]
#[case::other_payload(HiddenOverlap::OtherPayload)]
fn hidden_fixed_array_records_require_exclusive_ownership(#[case] overlap: HiddenOverlap) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(12)])
        .with_chunks(&[4]);
    builder
        .create_dataset("other")
        .with_i32_data(&[7, 8, 9, 10]);
    let mut store = WindowProbeStore::open(builder.finish().unwrap());
    let header = header(&store);
    let block = header.data_block().unwrap();
    let prefix = block.prefix_range().unwrap();
    let bytes = store
        .read_exact_at(prefix.start().get(), prefix.length().to_usize().unwrap())
        .unwrap();
    let window = block
        .parse_unpaged_prefix(FileRegion::new(prefix, &bytes).unwrap())
        .unwrap()
        .window(0, NonZeroUsize::new(3).unwrap())
        .unwrap()
        .unwrap();
    let range = window.parse_range();
    let bytes = store
        .read_exact_at(range.start().get(), range.length().to_usize().unwrap())
        .unwrap();
    let (_, original) = window
        .parse(FileRegion::new(range, &bytes).unwrap())
        .unwrap()
        .records(16)
        .next()
        .unwrap()
        .unwrap();
    let root = store.superblock.root_group_address.get();
    let (address, previous_end) = match overlap {
        HiddenOverlap::Root => {
            let walk =
                walk_v2_object_header(&store, PhysicalAddress::new(root), store.widths()).unwrap();
            (root + 1, walk.messages[0].chunk_msg_end + EA_CHECKSUM_SIZE)
        }
        HiddenOverlap::SharedChunk => (original.address.get(), original.address.get() + 16),
        HiddenOverlap::OtherPayload => {
            let file = File::from_bytes(store.data.clone()).unwrap();
            let dataset = file.dataset("other").unwrap();
            let Layout::Contiguous {
                address: Some(address),
                size,
            } = dataset.layout().unwrap()
            else {
                panic!("expected contiguous storage")
            };
            assert_eq!(size, 16);
            drop(dataset);
            file.close().unwrap();
            (address, address + 16)
        }
    };
    let records = [
        Some(test_util::fixed_array::Entry {
            address: original.address.get(),
            stored_size: 16,
            filter_mask: 0,
        }),
        Some(test_util::fixed_array::Entry {
            address,
            stored_size: 16,
            filter_mask: 0,
        }),
        None,
    ];
    let replacement = test_util::fixed_array::data_block(
        Widths::EIGHT,
        test_util::fixed_array::Client::Plain,
        header.page_bits(),
        &records,
        header.range().start().get(),
    );
    assert_eq!(replacement.len().to_u64(), block.range().length());
    let start = block.range().start().get().to_usize().unwrap();
    store.data[start..start + replacement.len()].copy_from_slice(&replacement);
    let file = File::from_bytes(store.data.clone()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0]);
    file.close().unwrap();
    let before = store.data;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("hidden.h5");
    std::fs::write(&path, &before).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let error = dataset.append(&[1_i32]).unwrap_err();
    assert!(
        matches!(error, Error::InvalidSpaceOwnershipExtent {
        address: actual, length: 16, previous_end: actual_end, logical_size,
    } if actual == address && actual_end == previous_end && logical_size == before.len().to_u64()),
        "{error:?}"
    );
    assert_eq!(file.disk_bytes().unwrap(), before);
    drop(dataset);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0]);
    assert_eq!(
        file.dataset("other").unwrap().read_i32().unwrap(),
        [7, 8, 9, 10]
    );
    file.close().unwrap();
}

fn header(store: &WindowProbeStore) -> FixedArrayHeader {
    let walk = walk_v2_object_header(
        store,
        dataset_address(store).header_address(),
        store.widths(),
    )
    .unwrap();
    let message = walk
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let bytes = store.read_exact_at(message.data_off, message.size).unwrap();
    let DataLayout::Chunked {
        index: ChunkIndexLayout::FixedArray {
            address: Some(address),
        },
        ..
    } = DataLayout::parse_with_widths(&bytes, store.widths()).unwrap()
    else {
        panic!("expected an allocated Fixed Array header")
    };
    let locator =
        FixedArrayHeaderLocator::new(store.widths(), store.addresses().effective_base(), address);
    let range = locator.parse_range().unwrap();
    let bytes = store
        .read_exact_at(range.start().get(), range.length().to_usize().unwrap())
        .unwrap();
    locator
        .parse(FileRegion::new(range, &bytes).unwrap())
        .unwrap()
}

#[derive(Clone, Copy)]
enum HiddenOverlap {
    OtherPayload,
    Root,
    SharedChunk,
}

#[rstest]
#[case::dimension(Stale::Dimension)]
#[case::header(Stale::Header)]
#[case::entries(Stale::Entries)]
#[case::bitmap(Stale::Bitmap)]
#[case::width(Stale::Width)]
fn fixed_array_plans_reject_changed_source_fields(#[case] stale: Stale) {
    let source = if matches!(stale, Stale::Bitmap) {
        FixedSource::SparsePages
    } else {
        FixedSource::Allocated
    };
    let mut store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
        source,
    );
    let address = dataset_address(&store);
    let array = header(&store);
    let mut state = edit::locate_dataset_state(&store, address).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let plan = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            None,
            LossyTailReencoding::Reject,
            &i32_bytes(&[1]),
            NonZeroU64::new(1).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    let expected = match stale {
        Stale::Width => {
            store.widths.lengths = LengthWidth::Four;
            "append field widths changed after planning"
        }
        Stale::Dimension => {
            let walk =
                walk_v2_object_header(&store, address.header_address(), store.widths()).unwrap();
            let message = walk
                .messages
                .iter()
                .find(|message| message.msg_type == MessageType::DATASPACE)
                .unwrap();
            let start = message.data_off.to_usize().unwrap() + test_util::dataspace::V2_HEADER_LEN;
            store.data[start..start + 8].copy_from_slice(&2_u64.to_le_bytes());
            test_util::checksum::restamp(
                &mut store.data,
                message.chunk_start.to_usize().unwrap(),
                (message.chunk_msg_end - message.chunk_start + EA_CHECKSUM_SIZE)
                    .to_usize()
                    .unwrap(),
            );
            "dataspace extent changed after append planning"
        }
        Stale::Header => {
            let block = array.data_block().unwrap();
            let bytes = test_util::fixed_array::header(
                Widths::EIGHT,
                test_util::fixed_array::Client::Plain,
                array.page_bits(),
                array.entries() + 1,
                Some(block.range().start().get()),
            );
            let start = array.range().start().get().to_usize().unwrap();
            store.data[start..start + bytes.len()].copy_from_slice(&bytes);
            "fixed-array header changed after append planning"
        }
        Stale::Entries => {
            let block = array.data_block().unwrap();
            let start = block
                .prefix_range()
                .unwrap()
                .end()
                .get()
                .to_usize()
                .unwrap();
            store.data[start..start + 8].copy_from_slice(&32_u64.to_le_bytes());
            test_util::checksum::restamp(
                &mut store.data,
                block.range().start().get().to_usize().unwrap(),
                block.range().length().to_usize().unwrap(),
            );
            "fixed-array checksum changed after append planning"
        }
        Stale::Bitmap => {
            let block = array.data_block().unwrap();
            let page = block.page_for_slot(0).unwrap().unwrap();
            let bytes = test_util::fixed_array::page(
                Widths::EIGHT,
                test_util::fixed_array::Client::Plain,
                &[None; 4],
            );
            let start = page.parse_range().start().get().to_usize().unwrap();
            store.data[start..start + bytes.len()].copy_from_slice(&bytes);
            let (range, mask) = block.page_bitmap_byte(0).unwrap().unwrap();
            store.data[range.start().get().to_usize().unwrap()] |= mask.get();
            let prefix = block.parse_range().unwrap();
            test_util::checksum::restamp(
                &mut store.data,
                prefix.start().get().to_usize().unwrap(),
                prefix.length().to_usize().unwrap(),
            );
            "fixed-array checksum changed after append planning"
        }
    };
    let before = store.data.clone();
    store.reset_counters();
    let error = index.apply(&mut store, plan, 4).unwrap_err();
    assert!(
        matches!(error, Error::AppendUnsupported(actual) if actual == expected),
        "{error:?}"
    );
    assert_eq!(store.data, before);
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.retired, []);
}

#[test]
fn fixed_array_filtered_lengths_fail_before_allocating_payloads() {
    let bytes = PipelinePattern::Incompressible.data(20_000);
    let mut builder = FileBuilder::new();
    builder.with_length_width(LengthWidth::Two);
    builder
        .create_dataset("d")
        .with_raw_data(
            Datatype::Opaque {
                size: 20_000,
                tag: Vec::new(),
            },
            bytes[..20_000].to_vec(),
            1,
        )
        .with_maxshape(&[MaxExtent::Fixed(12)])
        .with_chunks(&[4])
        .with_deflate(DeflateLevel::One);
    let mut store = WindowProbeStore::open(builder.finish().unwrap());
    let array = header(&store);
    let block = array.data_block().unwrap();
    let prefix = block.prefix_range().unwrap();
    let input = store
        .read_exact_at(prefix.start().get(), prefix.length().to_usize().unwrap())
        .unwrap();
    let window = block
        .parse_unpaged_prefix(FileRegion::new(prefix, &input).unwrap())
        .unwrap()
        .window(0, NonZeroUsize::new(1).unwrap())
        .unwrap()
        .unwrap();
    let range = window.parse_range();
    let input = store
        .read_exact_at(range.start().get(), range.length().to_usize().unwrap())
        .unwrap();
    let (_, record) = window
        .parse(FileRegion::new(range, &input).unwrap())
        .unwrap()
        .records(80_000)
        .next()
        .unwrap()
        .unwrap();
    let widths = Widths::new(8, 2);
    let client = test_util::fixed_array::Client::Filtered { size_bytes: 2 };
    let new_header = test_util::fixed_array::header(
        widths,
        client,
        array.page_bits(),
        3,
        Some(block.range().start().get()),
    );
    let start = array.range().start().get().to_usize().unwrap();
    store.data[start..start + new_header.len()].copy_from_slice(&new_header);
    let new_block = test_util::fixed_array::data_block(
        widths,
        client,
        array.page_bits(),
        &[
            Some(test_util::fixed_array::Entry {
                address: record.address.get(),
                stored_size: record.stored_size,
                filter_mask: record.filter_mask,
            }),
            None,
            None,
        ],
        array.range().start().get(),
    );
    let start = block.range().start().get().to_usize().unwrap();
    store.data[start..start + new_block.len()].copy_from_slice(&new_block);
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let before = store.data.clone();
    store.reset_counters();
    let error = match index.plan(
        &store,
        &state.datatype,
        &state.spatial,
        state.pipeline.as_ref(),
        LossyTailReencoding::Reject,
        &bytes[20_000..],
        NonZeroU64::new(3).unwrap(),
        state.fill.pattern(state.element_size),
    ) {
        Ok(_) => panic!("an encoded chunk exceeds its field"),
        Err(error) => error,
    };
    let Error::Format(FormatError::LengthTooLarge {
        length,
        length_size,
    }) = error
    else {
        panic!("expected LengthTooLarge, got {error:?}")
    };
    assert_eq!(length_size, 2);
    assert!(length > u64::from(u16::MAX), "{length}");
    assert_eq!(store.data, before);
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.retired, []);
}

#[test]
fn fixed_array_address_overflow_preserves_published_storage() {
    let mut store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Two,
            lengths: LengthWidth::Eight,
        },
        false,
        FixedSource::Unallocated,
    );
    store.data.resize(65_534, 0);
    store.patch_superblock_eof().unwrap();
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let plan = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            None,
            LossyTailReencoding::Reject,
            &i32_bytes(&[0]),
            NonZeroU64::new(1).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    let before = store.data.clone();
    store.reset_counters();
    let error = index.apply(&mut store, plan, 4).unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::AddressTooLarge {
                address: 65_550,
                offset_size: 2
            })
        ),
        "{error:?}"
    );
    assert_eq!(store.data, before);
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.retired, []);
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0_i32; 0]);
    file.close().unwrap();
}

#[derive(Clone, Copy)]
enum Stale {
    Bitmap,
    Dimension,
    Entries,
    Header,
    Width,
}

fn append(store: &mut WindowProbeStore, state: &mut LocatedState, values: &[i32], phase: u8) {
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let plan = index
        .plan(
            store,
            &state.datatype,
            &state.spatial,
            state.pipeline.as_ref(),
            LossyTailReencoding::Reject,
            &i32_bytes(values),
            NonZeroU64::new(values.len().to_u64()).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    index.apply(store, plan, phase).unwrap();
}

pub(super) fn fixture(
    widths: FormatWidths,
    filtered: bool,
    source: FixedSource,
) -> WindowProbeStore {
    fixture_with_geometry(widths, filtered, source, 17, 2)
}

fn fixture_with_geometry(
    widths: FormatWidths,
    filtered: bool,
    source: FixedSource,
    entries: usize,
    page_bits: u8,
) -> WindowProbeStore {
    let maximum = match source {
        FixedSource::Allocated | FixedSource::Unallocated => 12,
        _ => entries.to_u64() * 4,
    };
    let values = if matches!(source, FixedSource::Allocated) {
        &[0][..]
    } else {
        &[][..]
    };
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(widths.offsets)
        .with_length_width(widths.lengths);
    let dataset = builder
        .create_dataset("d")
        .with_i32_data(values)
        .with_shape(&[values.len().to_u64()])
        .with_maxshape(&[MaxExtent::Fixed(maximum)])
        .with_chunks(&[4]);
    if filtered {
        dataset.with_deflate(DeflateLevel::One);
    }
    let store = WindowProbeStore::open(builder.finish().unwrap());
    if matches!(source, FixedSource::Allocated | FixedSource::Unallocated) {
        return store;
    }
    let fixture_widths = Widths::new(
        usize::from(widths.offsets.get()),
        usize::from(widths.lengths.get()),
    );
    let client = if filtered {
        test_util::fixed_array::Client::Filtered { size_bytes: 2 }
    } else {
        test_util::fixed_array::Client::Plain
    };
    let mut image = Image::starting_with(&store.data);
    let header_address = image.as_bytes().len().to_u64();
    let header =
        test_util::fixed_array::header(fixture_widths, client, page_bits, entries.to_u64(), None);
    let block_address = header_address + header.len().to_u64();
    let sparse = matches!(source, FixedSource::SparsePages);
    image.append(&test_util::fixed_array::header(
        fixture_widths,
        client,
        page_bits,
        entries.to_u64(),
        sparse.then_some(block_address),
    ));
    if sparse {
        image.append(&test_util::fixed_array::empty_data_block(
            fixture_widths,
            client,
            page_bits,
            entries,
            header_address,
        ));
    }
    let layout = test_util::data_layout::array_chunk(
        fixture_widths,
        test_util::data_layout::ArrayIndex::Fixed { page_bits },
        Some(header_address),
    );
    let space = test_util::dataspace::v2_with_widths(
        1,
        test_util::dataspace::Flags::MAX_DIMS,
        test_util::dataspace::Kind::SIMPLE,
        &[if sparse { 3 } else { 0 }],
        Some(&[maximum]),
        fixture_widths,
    );
    let walk =
        walk_v2_object_header(&store, dataset_address(&store).header_address(), widths).unwrap();
    let messages = walk.messages.into_iter().map(|message| {
        let body = match message.msg_type {
            MessageType::DATA_LAYOUT => layout.clone(),
            MessageType::DATASPACE => space.clone(),
            _ => store.read_exact_at(message.data_off, message.size).unwrap(),
        };
        Message::new(RecordType(message.msg_type.to_u16()), &body)
    });
    let dataset = image.append(&v2_bytes::Header::new().messages(messages).build());
    let root_walk = walk_v2_object_header(
        &store,
        PhysicalAddress::new(store.superblock.root_group_address.get()),
        widths,
    )
    .unwrap();
    let root_messages = root_walk
        .messages
        .into_iter()
        .filter(|message| message.msg_type != MessageType::LINK)
        .map(|message| {
            let body = store.read_exact_at(message.data_off, message.size).unwrap();
            Message::new(RecordType(message.msg_type.to_u16()), &body)
        });
    let root = image.append(
        &v2_bytes::Header::new()
            .messages(root_messages)
            .message(Message::new(
                RecordType::LINK,
                &test_util::link_message::HardLink::new("d", dataset.to_u64())
                    .build(fixture_widths),
            ))
            .build(),
    );
    let superblock = superblock_bytes::Superblock::new(fixture_widths)
        .root_header_address(root.to_u64())
        .eof_address(image.as_bytes().len().to_u64())
        .build();
    image.place(0, &superblock);
    WindowProbeStore::open(image.build())
}

#[derive(Clone, Copy, Debug)]
pub(super) enum FixedSource {
    Allocated,
    HeaderOnly,
    SparsePages,
    Unallocated,
}
