use hdf5_pure_format::__private::ChunkIndexFields;
use hdf5_pure_format::__private::ChunkRecord;
use rstest::rstest;
use test_util::data_layout::ArrayIndex;
use test_util::filtered_chunks::PipelinePattern;
use test_util::superblock::v2 as superblock_bytes;
use test_util_hdf5::dataset::FiniteChunkIndex;

use super::*;
use crate::File;
use crate::FileLocking;
use crate::Layout;
use crate::MemoryStrategy;
use crate::MetadataCacheConfig;
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::exact_limit(u32::MAX, 1, true)]
#[case::one_beyond(1 << 31, 2, false)]
#[case::host_exact(i32::MAX as u32, 1, true)]
#[case::host_one_beyond(1 << 31, 1, true)]
#[cfg_attr(
    target_pointer_width = "64",
    case::host_capacity(u32::MAX, u32::MAX, false)
)]
fn unallocated_finite_chunks_are_bounded_before_materialization(
    #[case] elements: u32,
    #[case] element_size: u32,
    #[case] supported: bool,
    #[values(FiniteChunkIndex::Fixed, FiniteChunkIndex::Single)] index: FiniteChunkIndex,
) {
    let layout = match index {
        FiniteChunkIndex::Single => test_util::data_layout::single_chunk_with_geometry(
            Widths::EIGHT,
            None,
            None,
            elements,
            element_size,
        ),
        FiniteChunkIndex::Fixed => test_util::data_layout::array_chunk_with_geometry(
            Widths::EIGHT,
            ArrayIndex::Fixed { page_bits: 4 },
            None,
            elements,
            element_size,
        ),
    };
    let datatype = test_util::datatype::fixed_string(element_size);
    let dataspace = test_util::dataspace::v2(
        1,
        test_util::dataspace::Flags::MAX_DIMS,
        test_util::dataspace::Kind::SIMPLE,
        &[0],
        Some(&[u64::from(elements)]),
    );
    let reserved = superblock_bytes::Superblock::new(Widths::EIGHT)
        .build()
        .len();
    let store =
        WindowProbeStore::open(Image::starting_with(&vec![0; reserved]).build_with_dataset(
            "d",
            [
                Message::new(RecordType::DATATYPE, &datatype),
                Message::new(RecordType::DATASPACE, &dataspace),
                Message::new(RecordType::DATA_LAYOUT, &layout),
            ],
        ));
    let before = store.data.clone();
    let result = edit::locate_dataset_state(&store, dataset_address(&store));
    if supported && u64::from(elements) <= isize::MAX as u64 {
        let state = result.unwrap();
        assert_eq!(
            state.index.chunk_bytes().to_u64(),
            u64::from(elements) * u64::from(element_size)
        );
        assert_eq!(state.index.current_dim(), 0);
    } else {
        let error = match result {
            Ok(_) => panic!("an oversized finite chunk is accepted"),
            Err(error) => error,
        };
        let expected = if supported {
            FormatError::ValueTooLargeForPlatform {
                value: u64::from(elements) * u64::from(element_size),
                target: "single-chunk allocation",
            }
        } else {
            FormatError::InvalidChunkGeometry(
                "chunk logical byte size exceeds the reader's single-chunk allocation limit",
            )
        };
        assert!(
            matches!(error, Error::Format(ref actual) if actual == &expected),
            "{error:?}"
        );
    }
    assert_eq!(store.data, before);
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.retired, []);
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn single_chunk_interruptions_preserve_the_published_prefix(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(1, 2, 3, 4)] phase: u8,
    #[values(false, true)] filtered: bool,
) {
    let mut store = fixture(FormatWidths { offsets, lengths }, filtered);
    let address = dataset_address(&store);
    let mut state = edit::locate_dataset_state(&store, address).unwrap();
    let AppendIndex::Single(index) = &mut state.index else {
        panic!("expected a Single Chunk index")
    };
    let plan = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            state.pipeline.as_ref(),
            LossyTailReencoding::Reject,
            &i32_bytes(&[1]),
            NonZeroU64::new(1).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    index.apply(&mut store, plan, phase).unwrap();
    let expected = if phase == 4 { vec![0, 1] } else { vec![0] };
    let file = File::from_bytes(store.data.clone()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
    file.close().unwrap();
    assert_eq!(store.retired.len(), usize::from(phase >= 2));
    let mut store = WindowProbeStore::open(store.data);
    let mut state = edit::locate_dataset_state(&store, address).unwrap();
    assert_eq!(state.index.current_dim(), expected.len().to_u64());
    let AppendIndex::Single(index) = &mut state.index else {
        panic!("expected a Single Chunk index")
    };
    let value = i32::try_from(expected.len()).unwrap();
    let plan = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            state.pipeline.as_ref(),
            LossyTailReencoding::Reject,
            &i32_bytes(&[value]),
            NonZeroU64::new(1).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    index.apply(&mut store, plan, 4).unwrap();
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..=value).collect::<Vec<_>>()
    );
    file.close().unwrap();
}

#[rstest]
#[case::reference(Stale::Reference)]
#[case::dimension(Stale::Dimension)]
#[case::width(Stale::Width)]
fn single_chunk_plans_reject_changed_fields_before_allocating(#[case] stale: Stale) {
    let mut store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
    );
    let address = dataset_address(&store);
    let mut state = edit::locate_dataset_state(&store, address).unwrap();
    let AppendIndex::Single(index) = &mut state.index else {
        panic!("expected a Single Chunk index")
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
    match stale {
        Stale::Width => store.widths.lengths = LengthWidth::Four,
        Stale::Dimension => {
            store.data[usize::try_from(index.geometry.dimension.field.start().get()).unwrap()] = 2
        }
        Stale::Reference => {
            let walk =
                walk_v2_object_header(&store, address.header_address(), store.widths()).unwrap();
            let message = walk
                .messages
                .iter()
                .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
                .unwrap();
            let body = store.read_exact_at(message.data_off, message.size).unwrap();
            let (_, Some(ChunkIndexFields::SingleChunk(field))) =
                DataLayout::parse_with_index_fields(&body, store.widths()).unwrap()
            else {
                panic!("expected Single Chunk fields")
            };
            store.data[usize::try_from(message.data_off).unwrap() + field.range().start] ^= 1;
        }
    }
    let before = store.data.clone();
    store.reset_counters();
    let error = index.apply(&mut store, plan, 4).unwrap_err();
    let expected = match stale {
        Stale::Reference => "chunk reference changed after append planning",
        Stale::Dimension => "dataspace extent changed after append planning",
        Stale::Width => "append field widths changed after planning",
    };
    assert!(
        matches!(&error,Error::AppendUnsupported(reason) if *reason==expected),
        "{error:?}"
    );
    assert_eq!(store.data, before);
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.retired, []);
}

#[test]
fn single_chunk_filtered_lengths_fail_before_payload_allocation() {
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
        .with_maxshape(&[MaxExtent::Fixed(4)])
        .with_chunks(&[4])
        .with_deflate(DeflateLevel::One);
    let store = WindowProbeStore::open(builder.finish().unwrap());
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Single(index) = &mut state.index else {
        panic!("expected a Single Chunk index")
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
        Ok(_) => panic!("encoded length exceeds its field"),
        Err(error) => error,
    };
    let Error::Format(FormatError::LengthTooLarge {
        length,
        length_size,
    }) = error
    else {
        panic!("expected compressed length overflow, got {error:?}")
    };
    assert_eq!(length_size, 2);
    assert!(length > u64::from(u16::MAX), "compressed size {length}");
    assert_eq!(store.data, before);
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.retired, []);
}

#[rstest]
#[case::content(1, 0)]
#[case::eof(2, 0)]
#[case::reference(3, 0)]
#[case::dimension(4, 1)]
fn single_chunk_barrier_failures_keep_unconfirmed_extents_unavailable(
    #[case] synchronization: usize,
    #[case] retired: usize,
) {
    let mut store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
    );
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Single(index) = &mut state.index else {
        panic!("expected a Single Chunk index")
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
    store.reset_counters();
    store.fail_sync = Some(synchronization);
    let error = index.apply(&mut store, plan, 4).unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected synchronization error, got {error:?}")
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected synchronization failure");
    assert_eq!(store.retired.len(), retired);
    let error = match store.allocate(&[0; 16], AllocationClass::Raw) {
        Ok(_) => panic!("uncertain allocation succeeds"),
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    let error = store.write_at(0, &[0]).unwrap_err();
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        if synchronization == 4 {
            vec![0, 1]
        } else {
            vec![0]
        }
    );
    file.close().unwrap();
}

#[test]
fn single_chunk_storage_overlapping_its_header_is_rejected() {
    let mut store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
    );
    let address = dataset_address(&store);
    let walk = walk_v2_object_header(&store, address.header_address(), store.widths()).unwrap();
    let message = walk
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let body = store.read_exact_at(message.data_off, message.size).unwrap();
    let (_, Some(ChunkIndexFields::SingleChunk(field))) =
        DataLayout::parse_with_index_fields(&body, store.widths()).unwrap()
    else {
        panic!("expected Single Chunk fields")
    };
    let encoded = field
        .encode(Some(ChunkRecord {
            address: StoredAddress::new(address.header_address().get()),
            stored_size: 16,
            filter_mask: 0,
        }))
        .unwrap();
    let start = usize::try_from(message.data_off).unwrap() + field.range().start;
    store.data[start..start + encoded.as_bytes().len()].copy_from_slice(encoded.as_bytes());
    test_util::checksum::restamp(
        &mut store.data,
        usize::try_from(message.chunk_start).unwrap(),
        usize::try_from(message.chunk_msg_end - message.chunk_start + EA_CHECKSUM_SIZE).unwrap(),
    );
    let error = match edit::locate_dataset_state(&store, address) {
        Ok(_) => panic!("header storage is accepted as a chunk"),
        Err(error) => error,
    };
    assert!(
        matches!(&error,Error::AppendUnsupported(reason) if *reason=="chunk storage overlaps its dataset's object header"),
        "{error:?}"
    );
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.retired, []);
}

#[rstest]
#[case::root_metadata(Overlap::Root)]
#[case::shared_payload(Overlap::Shared)]
#[case::partial_shared_payload(Overlap::PartialShared)]
fn single_chunk_overlapping_live_storage_is_rejected_before_publication(#[case] overlap: Overlap) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(4)])
        .with_chunks(&[4]);
    builder
        .create_dataset("other")
        .with_i32_data(&[7])
        .with_maxshape(&[MaxExtent::Fixed(4)])
        .with_chunks(&[4]);
    let mut store = WindowProbeStore::open(builder.finish().unwrap());
    let root = store.superblock.root_group_address.get();
    let (target, previous_end) = match overlap {
        Overlap::Root => {
            let root_header =
                walk_v2_object_header(&store, PhysicalAddress::new(root), store.widths()).unwrap();
            let message = &root_header.messages[0];
            (root + 1, message.chunk_msg_end + EA_CHECKSUM_SIZE)
        }
        Overlap::Shared | Overlap::PartialShared => {
            let file = File::from_bytes(store.data.clone()).unwrap();
            let dataset = file.dataset("other").unwrap();
            let Layout::Chunked { .. } = dataset.layout().unwrap() else {
                panic!("expected chunked storage")
            };
            let address = dataset.header_address().unwrap();
            drop(dataset);
            file.close().unwrap();
            let header =
                walk_v2_object_header(&store, PhysicalAddress::new(address), store.widths())
                    .unwrap();
            let message = header
                .messages
                .iter()
                .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
                .unwrap();
            let body = store.read_exact_at(message.data_off, message.size).unwrap();
            let DataLayout::Chunked {
                index:
                    ChunkIndexLayout::SingleChunk {
                        address: Some(address),
                        ..
                    },
                ..
            } = DataLayout::parse_with_widths(&body, store.widths()).unwrap()
            else {
                panic!("expected allocated Single Chunk storage")
            };
            (
                address.get() + u64::from(matches!(overlap, Overlap::PartialShared)),
                address.get() + 16,
            )
        }
    };
    let address = dataset_address(&store);
    let header = walk_v2_object_header(&store, address.header_address(), store.widths()).unwrap();
    let message = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let body = store.read_exact_at(message.data_off, message.size).unwrap();
    let (_, Some(ChunkIndexFields::SingleChunk(field))) =
        DataLayout::parse_with_index_fields(&body, store.widths()).unwrap()
    else {
        panic!("expected Single Chunk fields")
    };
    let encoded = field
        .encode(Some(ChunkRecord {
            address: StoredAddress::new(target),
            stored_size: 16,
            filter_mask: 0,
        }))
        .unwrap();
    let start = usize::try_from(message.data_off).unwrap() + field.range().start;
    store.data[start..start + encoded.as_bytes().len()].copy_from_slice(encoded.as_bytes());
    test_util::checksum::restamp(
        &mut store.data,
        usize::try_from(message.chunk_start).unwrap(),
        usize::try_from(message.chunk_msg_end - message.chunk_start + EA_CHECKSUM_SIZE).unwrap(),
    );
    store.data.resize(store.data.len() + 16, 0);
    store.patch_superblock_eof().unwrap();
    let before = store.data;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("overlap.h5");
    std::fs::write(&path, &before).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let error = dataset.append(&[1_i32]).unwrap_err();
    assert!(
        matches!(error, Error::InvalidSpaceOwnershipExtent {
            address, length: 16, previous_end: actual_end, logical_size,
        } if address == target && actual_end == previous_end && logical_size == before.len().to_u64()),
        "{error:?}"
    );
    assert_eq!(file.disk_bytes().unwrap(), before);
    drop(dataset);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("other").unwrap().read_i32().unwrap(), [7]);
    assert_eq!(file.root().datasets().unwrap(), ["d", "other"]);
    file.close().unwrap();
}

#[rstest]
#[case::detached_header(Ownership::Detached)]
#[case::incomplete_metadata(Ownership::Incomplete)]
fn single_chunk_storage_requires_complete_reachable_ownership(#[case] ownership: Ownership) {
    let store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        false,
    );
    let old_address = dataset_address(&store);
    let header =
        walk_v2_object_header(&store, old_address.header_address(), store.widths()).unwrap();
    let mut messages = header
        .messages
        .into_iter()
        .map(|message| {
            let body = store.read_exact_at(message.data_off, message.size).unwrap();
            Message::new(RecordType(message.msg_type.to_u16()), &body)
        })
        .collect::<Vec<_>>();
    if matches!(ownership, Ownership::Incomplete) {
        messages.push(Message::new(RecordType::OBJECT_COMMENT, b"note\0"));
    }
    let mut image = store.data.clone();
    let reserved = superblock_bytes::Superblock::new(Widths::EIGHT)
        .build()
        .len();
    image[..reserved].fill(0);
    let before = Image::starting_with(&image).build_with_dataset("d", messages);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ownership.h5");
    std::fs::write(&path, &before).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    let path_name = ObjectPathBuf::parse("d");
    let target = match ownership {
        Ownership::Detached => AppendTarget::Header(old_address),
        Ownership::Incomplete => AppendTarget::Path(&path_name),
    };
    let error = match engine.append_geometry(target) {
        Ok(_) => panic!("unproven Single Chunk ownership is accepted"),
        Err(error) => error,
    };
    let reason = match ownership {
        Ownership::Detached => "chunk storage has no reachable ownership proof",
        Ownership::Incomplete => "chunk storage ownership analysis is incomplete",
    };
    assert!(
        matches!(error, Error::AppendUnsupported(actual) if actual == reason),
        "{error:?}"
    );
    assert_eq!(
        engine.image().read_exact_at(0, before.len()).unwrap(),
        before
    );
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0]);
    file.close().unwrap();
}

#[test]
fn single_chunk_address_overflow_preserves_the_previous_reference() {
    let mut store = fixture(
        FormatWidths {
            offsets: OffsetWidth::Two,
            lengths: LengthWidth::Eight,
        },
        false,
    );
    store.data.resize(65_534, 0);
    store.patch_superblock_eof().unwrap();
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Single(index) = &mut state.index else {
        panic!("expected a Single Chunk index")
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
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0]);
    file.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(128, "single_plan_small", "single_plan_small_control")]
#[case::large(4_096, "single_plan_large", "single_plan_large_control")]
fn single_chunk_planning_retains_one_logical_chunk(
    #[case] chunk: u64,
    #[case] region: &'static str,
    #[case] control_region: &'static str,
) {
    use test_util::allocation;

    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    let mut store = fixture_chunk(widths, false, chunk);
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Single(index) = &mut state.index else {
        panic!("expected a Single Chunk index")
    };
    let raw = i32_bytes(&[1]);
    let warm = index
        .plan(
            &store,
            &state.datatype,
            &state.spatial,
            None,
            LossyTailReencoding::Reject,
            &raw,
            NonZeroU64::new(1).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    drop(warm);
    let chunk_bytes = usize::try_from(chunk).unwrap() * 4;
    let (control, positive) = allocation::measure(control_region, || {
        std::hint::black_box(vec![0_u8; chunk_bytes])
    });
    assert!(positive.blocks > 0, "{positive}");
    assert!(
        positive.peak_bytes >= u64::try_from(chunk_bytes).unwrap(),
        "{positive}"
    );
    drop(control);
    store.reset_counters();
    let (plan, costs) = allocation::measure(region, || {
        index
            .plan(
                &store,
                &state.datatype,
                &state.spatial,
                None,
                LossyTailReencoding::Reject,
                &raw,
                NonZeroU64::new(1).unwrap(),
                state.fill.pattern(state.element_size),
            )
            .unwrap()
    });
    assert!(
        costs.bytes <= u64::try_from(chunk_bytes + 512).unwrap(),
        "{costs}"
    );
    assert!(
        costs.peak_bytes <= u64::try_from(chunk_bytes + 512).unwrap(),
        "{costs}"
    );
    assert_eq!(costs.live_bytes, u64::try_from(chunk_bytes).unwrap());
    assert_eq!(
        store.total_read.get(),
        chunk_bytes + usize::from(widths.offsets.get()) + usize::from(widths.lengths.get())
    );
    assert_eq!(store.max_read.get(), chunk_bytes);
    assert_eq!(*store.allocations.borrow(), []);
    index.apply(&mut store, plan, 4).unwrap();
    let file = File::from_bytes(store.data).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0, 1]);
    file.close().unwrap();
}

#[test]
fn single_chunk_plans_reject_a_different_dataset_before_allocating() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(4)])
        .with_chunks(&[4]);
    builder
        .create_dataset("other")
        .with_i32_data(&[7])
        .with_maxshape(&[MaxExtent::Fixed(4)])
        .with_chunks(&[4]);
    let mut store = WindowProbeStore::open(builder.finish().unwrap());
    let mut first = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let other = group_v2::resolve_path_any(
        PhysicalBytes::for_test(&store.data, store.addresses()),
        AccessMode::ReadOnly,
        &store.superblock,
        store.widths(),
        &ObjectPath::parse("other"),
    )
    .unwrap();
    let mut second =
        edit::locate_dataset_state(&store, ObjectId::new(PhysicalAddress::new(other))).unwrap();
    let AppendIndex::Single(first) = &mut first.index else {
        panic!("expected a Single Chunk index")
    };
    let plan = first
        .plan(
            &store,
            &second.datatype,
            &second.spatial,
            None,
            LossyTailReencoding::Reject,
            &i32_bytes(&[1]),
            NonZeroU64::new(1).unwrap(),
            second.fill.pattern(second.element_size),
        )
        .unwrap();
    let AppendIndex::Single(second) = &mut second.index else {
        panic!("expected a Single Chunk index")
    };
    let before = store.data.clone();
    store.reset_counters();
    let error = second.apply(&mut store, plan, 4).unwrap_err();
    assert!(
        matches!(&error,Error::AppendUnsupported(reason) if *reason=="append plan refers to a different chunk index"),
        "{error:?}"
    );
    assert_eq!(store.data, before);
    assert_eq!(*store.allocations.borrow(), []);
    assert_eq!(store.total_read.get(), 0);
}

fn fixture(widths: FormatWidths, filtered: bool) -> WindowProbeStore {
    fixture_chunk(widths, filtered, 4)
}

fn fixture_chunk(widths: FormatWidths, filtered: bool, chunk: u64) -> WindowProbeStore {
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(widths.offsets)
        .with_length_width(widths.lengths);
    let target = builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(chunk)])
        .with_chunks(&[chunk]);
    if filtered {
        target.with_deflate(DeflateLevel::One);
    }
    WindowProbeStore::open(builder.finish().unwrap())
}

#[derive(Clone, Copy)]
enum Stale {
    Dimension,
    Reference,
    Width,
}

#[derive(Clone, Copy)]
enum Overlap {
    PartialShared,
    Root,
    Shared,
}

#[derive(Clone, Copy)]
enum Ownership {
    Detached,
    Incomplete,
}
