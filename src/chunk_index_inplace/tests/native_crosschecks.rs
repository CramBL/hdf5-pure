use std::fs;
use std::path::Path;

use hdf5_pure_format::__private::ChunkIndexFields;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FixedArrayDataBlock;
use hdf5_pure_format::__private::FixedArrayHeader;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use hdf5_pure_format::__private::FixedArrayPlan;
use rstest::rstest;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::FiniteChunkIndex;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

use super::fixed::FixedSource;
use super::*;
use crate::File as PureFile;
use crate::address::BaseAddressExt;

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn ea_publication_preserves_native_values_and_recorded_widths(
    #[case] offsets: OffsetWidth,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
    #[values(0, 3)] initial: i32,
    #[values(1, 4)] chunk: u64,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.h5");
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Eight,
    };
    producer.write(&input, widths, initial, chunk, filtered);
    let mut store = WindowProbeStore::open(fs::read(&input).unwrap());
    let mut expected = initial;
    for end in [5, 21, 331] {
        let address = dataset_address(&store);
        let mut state = edit::locate_dataset_state(&store, address).unwrap();
        assert_eq!(state.index.current_dim(), u64::try_from(expected).unwrap());
        assert_eq!(state.index.chunk_elems(), chunk);
        assert_eq!(state.pipeline.is_some(), filtered);
        if expected == 0 && matches!(producer, Producer::Native) {
            assert!(matches!(state.index, AppendIndex::Unallocated(_)));
        }
        let values: Vec<i32> = (expected..end).collect();
        let raw = i32_bytes(&values);
        let plan = match &state.index {
            AppendIndex::Allocated(index) => plan_ea_append(
                &store,
                index,
                &state.datatype,
                &state.spatial,
                state.element_size,
                state.pipeline.as_ref(),
                true,
                LossyTailReencoding::Reject,
                &raw,
                values.len().to_u64(),
                state.fill.pattern(state.element_size),
            )
            .unwrap(),
            AppendIndex::Fixed(_) | AppendIndex::Single(_) => {
                panic!("expected an Extensible Array index")
            }
            AppendIndex::Unallocated(index) => {
                let (plan, initialization) = plan_initial_ea_append(
                    index,
                    widths,
                    &state.datatype,
                    &state.spatial,
                    state.element_size,
                    state.pipeline.as_ref(),
                    &raw,
                    values.len().to_u64(),
                    state.fill.pattern(state.element_size),
                )
                .unwrap();
                state.index = AppendIndex::Allocated(
                    initialize_ea_index(&mut store, address, index, &initialization).unwrap(),
                );
                plan
            }
        };
        apply_ea_append(&mut store, state.index.allocated_mut().unwrap(), plan, 4).unwrap();
        let index = state.index.allocated().unwrap();
        assert_eq!(index.current_dim, u64::try_from(end).unwrap());
        assert_eq!(
            index.num_chunks,
            u64::try_from(end).unwrap().div_ceil(chunk)
        );
        assert_eq!(index.offset_width, offsets);
        assert_eq!(index.length_width, LengthWidth::Eight);
        assert_eq!(
            index.allocation_stats.unwrap().max_idx_set,
            index.num_chunks
        );
        if end == 331 && chunk == 1 {
            let array = metadata_io::read_header(&store, index.ea_addr).unwrap();
            let bytes = IndexBlockBytes::read(&store, &array).unwrap();
            let block = bytes.parse().unwrap();
            let secondary = block
                .super_blocks()
                .next()
                .expect("the emitted index must address a secondary block")
                .unwrap();
            let range = secondary.parse_range();
            metadata_io::require_range(&store, range).unwrap();
            let bytes = store
                .read_exact_at(range.start().get(), range.length().to_usize().unwrap())
                .unwrap();
            secondary
                .parse(FileRegion::new(range, &bytes).unwrap())
                .unwrap();
        }
        assert_eq!(store.superblock.eof_address, store.data.len().to_u64());
        let issued = directory.path().join("issued.h5");
        fs::write(&issued, &store.data).unwrap();
        let native = file::libhdf5_open(&issued);
        let properties = native.create_plist().unwrap().get_sizes().unwrap();
        assert_eq!(
            usize::from(properties.sizeof_addr),
            usize::from(offsets.get())
        );
        assert_eq!(
            usize::from(properties.sizeof_size),
            usize::from(LengthWidth::Eight.get())
        );
        let dataset = native.dataset("d").unwrap();
        assert_eq!(dataset.shape(), [usize::try_from(end).unwrap()]);
        assert_eq!(dataset.chunk(), Some(vec![usize::try_from(chunk).unwrap()]));
        assert_eq!(
            dataset.read_raw::<i32>().unwrap(),
            (0..end).collect::<Vec<_>>()
        );
        drop(dataset);
        native.close().unwrap();
        let reopened = crate::File::from_bytes(store.data.clone()).unwrap();
        assert_eq!(
            reopened.dataset("d").unwrap().read_i32().unwrap(),
            (0..end).collect::<Vec<_>>()
        );
        reopened.close().unwrap();
        store = WindowProbeStore::open(store.data);
        expected = end;
    }
    directory.close().unwrap();
}

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
fn finite_layout_field_encoders_match_native_storage(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
    #[values(4, 12)] maximum: usize,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] allocated: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("finite-layout.h5");
    let widths = FormatWidths { offsets, lengths };
    let values: Vec<i32> = if allocated { vec![0, 1, 2] } else { Vec::new() };
    producer.write_finite(&path, widths, maximum, filtered, &values);
    let native = file::libhdf5_open(&path);
    let sizes = native.create_plist().unwrap().get_sizes().unwrap();
    assert_eq!(usize::from(sizes.sizeof_addr), usize::from(offsets.get()));
    assert_eq!(usize::from(sizes.sizeof_size), usize::from(lengths.get()));
    let dataset = native.dataset("d").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), values);
    let stored_size = dataset.storage_size();
    drop(dataset);
    native.close().unwrap();
    let store = WindowProbeStore::open(fs::read(&path).unwrap());
    let walk = walk_v2_object_header(&store, dataset_address(&store), widths).unwrap();
    let message = walk
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let body = store.read_exact_at(message.data_off, message.size).unwrap();
    let (layout, fields) = DataLayout::parse_with_index_fields(&body, widths).unwrap();
    let (range, encoded) = match (layout, fields.unwrap()) {
        (
            DataLayout::Chunked {
                index:
                    ChunkIndexLayout::SingleChunk {
                        address,
                        filtered: info,
                    },
                ..
            },
            ChunkIndexFields::SingleChunk(field),
        ) => {
            assert_eq!(maximum, 4);
            assert!(allocated || matches!(producer, Producer::Native));
            assert_eq!(info.is_some(), filtered);
            if let Some(info) = info {
                assert_eq!(info.filtered_size, stored_size);
            }
            (
                field.range(),
                field
                    .encode(address.map(|address| ChunkRecord {
                        address,
                        stored_size,
                        filter_mask: 0,
                    }))
                    .unwrap(),
            )
        }
        (
            DataLayout::Chunked {
                index: ChunkIndexLayout::FixedArray { address },
                ..
            },
            ChunkIndexFields::FixedArray {
                address: field,
                page_bits,
            },
        ) => {
            assert!(maximum == 12 || (!allocated && matches!(producer, Producer::Rust)));
            assert!(u32::from(page_bits) < u64::BITS);
            (field.range(), field.encode(address).unwrap())
        }
        (layout, fields) => panic!("unexpected finite layout {layout:?} with fields {fields:?}"),
    };
    assert_eq!(&body[range], encoded.as_bytes());
    let reopened = crate::File::from_bytes(store.data).unwrap();
    assert_eq!(reopened.dataset("d").unwrap().read_i32().unwrap(), values);
    reopened.close().unwrap();
    directory.close().unwrap();
}

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
fn native_writes_consume_streamed_empty_fixed_arrays(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(3, 1_025)] entries: usize,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty-fixed.h5");
    let widths = FormatWidths { offsets, lengths };
    Producer::Native.write_finite(&path, widths, entries * 4, filtered, &[]);
    let mut store = WindowProbeStore::open(fs::read(&path).unwrap());
    let walk = walk_v2_object_header(&store, dataset_address(&store), widths).unwrap();
    let message = walk
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let body = store.read_exact_at(message.data_off, message.size).unwrap();
    let (
        layout,
        Some(ChunkIndexFields::FixedArray {
            page_bits,
            address: field,
        }),
    ) = DataLayout::parse_with_index_fields(&body, widths).unwrap()
    else {
        panic!("expected Fixed Array fields");
    };
    assert!(matches!(
        layout,
        DataLayout::Chunked {
            index: ChunkIndexLayout::FixedArray { address: None },
            ..
        }
    ));
    let plan = FixedArrayPlan::new(entries.to_u64(), 16, widths, filtered).unwrap();
    assert_eq!(page_bits, plan.page_bits());
    let header_length = plan.block_lengths().next().unwrap();
    let addresses = store.addresses();
    let allocation = store
        .allocate_encoded(plan.total_len(), AllocationClass::Metadata, |at| {
            let header = addresses.relative(at)?;
            let placed = plan.place(header, header.checked_add(header_length)?)?;
            let mut output = placed.serialize_header()?;
            placed.emit_empty_data_block::<FormatError>(|bytes| {
                output.extend_from_slice(bytes);
                Ok(())
            })?;
            Ok(output)
        })
        .unwrap();
    let confirmed = store
        .publish_allocations()
        .unwrap()
        .confirm(allocation)
        .unwrap();
    let range = store.allocation_publication().require(&confirmed).unwrap();
    let header_address = addresses.relative(range.start()).unwrap();
    let encoded = field.encode(Some(header_address)).unwrap();
    ObjectHeaderField::new(message, field.range())
        .unwrap()
        .publish(&mut store, encoded.as_bytes())
        .unwrap();
    store.sync().unwrap();
    let locator = FixedArrayHeaderLocator::new(widths, BaseAddress::ZERO, header_address);
    let header_range = locator.parse_range().unwrap();
    let header_bytes = store
        .read_exact_at(
            header_range.start().get(),
            header_range.length().to_usize().unwrap(),
        )
        .unwrap();
    let header = locator
        .parse(FileRegion::new(header_range, &header_bytes).unwrap())
        .unwrap();
    assert_eq!(header.entries(), entries.to_u64());
    assert_eq!(header.is_filtered(), filtered);
    let block = header.data_block().unwrap();
    let prefix = block.parse_range().unwrap();
    let prefix_bytes = store
        .read_exact_at(prefix.start().get(), prefix.length().to_usize().unwrap())
        .unwrap();
    match block
        .parse(FileRegion::new(prefix, &prefix_bytes).unwrap())
        .unwrap()
    {
        FixedArrayDataBlock::Elements(elements) => {
            assert_eq!(entries, 3);
            assert!(elements.records(16).next().is_none());
        }
        FixedArrayDataBlock::PageBitmap(bitmap) => {
            assert_eq!(entries, 1_025);
            assert!(bitmap.pages().next().is_none());
            assert!(!bitmap.page_for_slot(0).unwrap().unwrap().1);
            assert!(
                !bitmap
                    .page_for_slot(entries.to_u64() - 1)
                    .unwrap()
                    .unwrap()
                    .1
            );
        }
    }
    fs::write(&path, &store.data).unwrap();
    let native = file::libhdf5_open_rw(&path);
    let dataset = native.dataset("d").unwrap();
    dataset.resize((1,)).unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [0]);
    dataset.write_raw(&[37_i32]).unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [37]);
    let sizes = native.create_plist().unwrap().get_sizes().unwrap();
    assert_eq!(
        (
            usize::from(sizes.sizeof_addr),
            usize::from(sizes.sizeof_size)
        ),
        (usize::from(offsets.get()), usize::from(lengths.get()))
    );
    drop(dataset);
    native.close().unwrap();
    let reopened = crate::File::from_bytes(fs::read(&path).unwrap()).unwrap();
    assert_eq!(reopened.dataset("d").unwrap().read_i32().unwrap(), [37]);
    reopened.close().unwrap();
    let rewritten = WindowProbeStore::open(fs::read(&path).unwrap());
    let bytes = rewritten
        .read_exact_at(
            header_range.start().get(),
            header_range.length().to_usize().unwrap(),
        )
        .unwrap();
    let native_header = locator
        .parse(FileRegion::new(header_range, &bytes).unwrap())
        .unwrap();
    assert_eq!(native_header.data_block().unwrap().range(), block.range());
    directory.close().unwrap();
}

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
fn fixed_array_appends_preserve_native_values_pages_and_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
    #[values(FixedInitial::Empty, FixedInitial::Fill, FixedInitial::Partial)] initial: FixedInitial,
    #[values(false, true)] paged: bool,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixed.h5");
    let widths = FormatWidths { offsets, lengths };
    let maximum = if paged { 4_100 } else { 12 };
    let length = match initial {
        FixedInitial::Empty | FixedInitial::Fill => 0,
        FixedInitial::Partial => {
            if paged {
                4_091
            } else {
                3
            }
        }
    };
    let mut expected = (0..length).collect::<Vec<i32>>();
    producer.write_finite(&path, widths, maximum, filtered, &expected);
    let native = file::libhdf5_open_rw(&path);
    native.link_hard("d", "alias").unwrap();
    if matches!(initial, FixedInitial::Fill) {
        let dataset = native.dataset("d").unwrap();
        dataset.resize((3,)).unwrap();
        expected = vec![0; 3];
        assert_eq!(dataset.read_raw::<i32>().unwrap(), expected);
    }
    native.close().unwrap();
    let mut store = WindowProbeStore::open(fs::read(&path).unwrap());
    let header = fixed_header(&store);
    assert_eq!(header.is_some(), matches!(initial, FixedInitial::Partial));
    if let Some(header) = header {
        assert_eq!(header.entries(), maximum.to_u64().div_ceil(4));
        assert_eq!(header.is_filtered(), filtered);
        let block = header.data_block().unwrap();
        let range = block.parse_range().unwrap();
        let bytes = store
            .read_exact_at(range.start().get(), range.length().to_usize().unwrap())
            .unwrap();
        match block
            .parse(FileRegion::new(range, &bytes).unwrap())
            .unwrap()
        {
            FixedArrayDataBlock::Elements(_) => assert!(!paged),
            FixedArrayDataBlock::PageBitmap(bitmap) => {
                assert!(paged);
                let (_, initialized) = bitmap.page_for_slot(1_024).unwrap().unwrap();
                assert_eq!(initialized, matches!(producer, Producer::Rust));
            }
        }
    }
    append_fixed_until(&mut store, &mut expected, if paged { 4_096 } else { 8 });
    assert_fixed_native(
        &directory.path().join("first.h5"),
        &store,
        &expected,
        widths,
        maximum,
    );
    let mut store = WindowProbeStore::open(store.data);
    append_fixed_until(&mut store, &mut expected, maximum);
    assert_fixed_native(
        &directory.path().join("second.h5"),
        &store,
        &expected,
        widths,
        maximum,
    );
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let before = store.data.clone();
    let error = match index.plan(
        &store,
        &state.datatype,
        &state.spatial,
        state.pipeline.as_ref(),
        LossyTailReencoding::Reject,
        &i32_bytes(&[i32::try_from(maximum).unwrap()]),
        NonZeroU64::new(1).unwrap(),
        state.fill.pattern(state.element_size),
    ) {
        Ok(_) => panic!("a Fixed Array grows beyond its maximum"),
        Err(error) => error,
    };
    assert!(
        matches!(
            error,
            Error::AppendUnsupported("append exceeds the dataset's maximum first dimension")
        ),
        "{error:?}"
    );
    assert_eq!(store.data, before);
    directory.close().unwrap();
}

fn append_fixed_until(store: &mut WindowProbeStore, expected: &mut Vec<i32>, end: usize) {
    let mut state = edit::locate_dataset_state(store, dataset_address(store)).unwrap();
    let AppendIndex::Fixed(index) = &mut state.index else {
        panic!("expected a Fixed Array index")
    };
    let values =
        (i32::try_from(expected.len()).unwrap()..i32::try_from(end).unwrap()).collect::<Vec<_>>();
    let plan = index
        .plan(
            store,
            &state.datatype,
            &state.spatial,
            state.pipeline.as_ref(),
            LossyTailReencoding::Reject,
            &i32_bytes(&values),
            NonZeroU64::new(values.len().to_u64()).unwrap(),
            state.fill.pattern(state.element_size),
        )
        .unwrap();
    index.apply(store, plan, 4).unwrap();
    expected.extend_from_slice(&values);
}

fn fixed_header(store: &WindowProbeStore) -> Option<FixedArrayHeader> {
    let walk = walk_v2_object_header(store, dataset_address(store), store.widths()).unwrap();
    let message = walk
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let bytes = store.read_exact_at(message.data_off, message.size).unwrap();
    let DataLayout::Chunked {
        index: ChunkIndexLayout::FixedArray { address },
        ..
    } = DataLayout::parse_with_widths(&bytes, store.widths()).unwrap()
    else {
        panic!("expected a Fixed Array layout")
    };
    address.map(|address| {
        let locator = FixedArrayHeaderLocator::new(
            store.widths(),
            store.addresses().effective_base(),
            address,
        );
        let range = locator.parse_range().unwrap();
        let bytes = store
            .read_exact_at(range.start().get(), range.length().to_usize().unwrap())
            .unwrap();
        locator
            .parse(FileRegion::new(range, &bytes).unwrap())
            .unwrap()
    })
}

fn assert_fixed_native(
    path: &Path,
    store: &WindowProbeStore,
    expected: &[i32],
    widths: FormatWidths,
    maximum: usize,
) {
    let header = fixed_header(store).unwrap();
    assert_eq!(header.entries(), maximum.to_u64().div_ceil(4));
    assert!(header.data_block().is_some());
    fs::write(path, &store.data).unwrap();
    let native = file::libhdf5_open(path);
    let target = native.dataset("d").unwrap();
    assert_eq!(target.read_raw::<i32>().unwrap(), expected);
    assert_eq!(target.shape(), [expected.len()]);
    drop(target);
    assert_eq!(
        native.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    let sizes = native.create_plist().unwrap().get_sizes().unwrap();
    assert_eq!(
        (
            usize::from(sizes.sizeof_addr),
            usize::from(sizes.sizeof_size)
        ),
        (
            usize::from(widths.offsets.get()),
            usize::from(widths.lengths.get())
        )
    );
    native.close().unwrap();
    let pure = PureFile::open(path).unwrap();
    assert_eq!(pure.dataset("d").unwrap().read_i32().unwrap(), expected);
    assert_eq!(pure.dataset("alias").unwrap().read_i32().unwrap(), expected);
    pure.close().unwrap();
}

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
fn fixed_array_appends_preserve_source_page_parameters(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(FixedSource::HeaderOnly, FixedSource::SparsePages)] source: FixedSource,
    #[values(false, true)] filtered: bool,
) {
    let widths = FormatWidths { offsets, lengths };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source-pages.h5");
    let store = super::fixed::fixture(widths, filtered, source);
    let header = fixed_header(&store).unwrap();
    assert_eq!(header.page_bits(), 2);
    assert_eq!(header.entries(), 17);
    assert_eq!(
        header.data_block().is_some(),
        matches!(source, FixedSource::SparsePages)
    );
    fs::write(&path, &store.data).unwrap();
    let native = file::libhdf5_open_rw(&path);
    native.link_hard("d", "alias").unwrap();
    let mut expected = if matches!(source, FixedSource::SparsePages) {
        vec![0; 3]
    } else {
        Vec::new()
    };
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    native.close().unwrap();
    let mut store = WindowProbeStore::open(fs::read(&path).unwrap());
    append_fixed_until(&mut store, &mut expected, 21);
    assert_eq!(fixed_header(&store).unwrap().page_bits(), 2);
    assert_fixed_native(
        &directory.path().join("first.h5"),
        &store,
        &expected,
        widths,
        68,
    );
    let mut store = WindowProbeStore::open(store.data);
    append_fixed_until(&mut store, &mut expected, 68);
    assert_eq!(fixed_header(&store).unwrap().page_bits(), 2);
    assert_fixed_native(
        &directory.path().join("second.h5"),
        &store,
        &expected,
        widths,
        68,
    );
    directory.close().unwrap();
}

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
fn single_chunk_appends_preserve_native_values_and_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(
        SingleSource::Empty,
        SingleSource::Unallocated,
        SingleSource::Native,
        SingleSource::Rust
    )]
    source: SingleSource,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("single.h5");
    let widths = FormatWidths { offsets, lengths };
    let (producer, values) = match source {
        SingleSource::Empty | SingleSource::Unallocated => (Producer::Native, &[][..]),
        SingleSource::Native => (Producer::Native, &[0][..]),
        SingleSource::Rust => (Producer::Rust, &[0][..]),
    };
    producer.write_finite(&path, widths, 4, filtered, values);
    let native = file::libhdf5_open_rw(&path);
    native.link_hard("d", "alias").unwrap();
    native.close().unwrap();
    let mut expected = values.to_vec();
    if matches!(source, SingleSource::Unallocated) {
        let native = file::libhdf5_open_rw(&path);
        let dataset = native.dataset("d").unwrap();
        dataset.resize((3,)).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), [0, 0, 0]);
        drop(dataset);
        native.close().unwrap();
        expected = vec![0; 3];
    }
    let mut store = WindowProbeStore::open(fs::read(&path).unwrap());
    let address = dataset_address(&store);
    let walk = walk_v2_object_header(&store, address, widths).unwrap();
    let layout = walk
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let body = store.read_exact_at(layout.data_off, layout.size).unwrap();
    let DataLayout::Chunked {
        index:
            ChunkIndexLayout::SingleChunk {
                address: chunk,
                filtered: filter_fields,
            },
        ..
    } = DataLayout::parse_with_widths(&body, widths).unwrap()
    else {
        panic!("expected a Single Chunk layout")
    };
    assert_eq!(
        chunk.is_none(),
        matches!(source, SingleSource::Empty | SingleSource::Unallocated)
    );
    assert_eq!(filter_fields.is_some(), filtered);
    let mut state = edit::locate_dataset_state(&store, address).unwrap();
    assert_eq!(state.pipeline.is_some(), filtered);
    assert_eq!(state.index.maximum(), MaxExtent::Fixed(4));
    while expected.len() < 4 {
        let value = i32::try_from(expected.len()).unwrap();
        let raw = i32_bytes(&[value]);
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
                &raw,
                NonZeroU64::new(1).unwrap(),
                state.fill.pattern(state.element_size),
            )
            .unwrap();
        index.apply(&mut store, plan, 4).unwrap();
        expected.push(value);
        assert_eq!(state.index.current_dim(), expected.len().to_u64());
        assert_eq!(store.superblock.eof_address, store.data.len().to_u64());
        let snapshot = directory.path().join("issued-single.h5");
        fs::write(&snapshot, &store.data).unwrap();
        let native = file::libhdf5_open(&snapshot);
        let dataset = native.dataset("d").unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), expected);
        assert_eq!(
            native.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
            expected
        );
        let sizes = native.create_plist().unwrap().get_sizes().unwrap();
        assert_eq!(
            (
                usize::from(sizes.sizeof_addr),
                usize::from(sizes.sizeof_size)
            ),
            (usize::from(offsets.get()), usize::from(lengths.get()))
        );
        drop(dataset);
        native.close().unwrap();
        let reopened = crate::File::from_bytes(store.data.clone()).unwrap();
        assert_eq!(reopened.dataset("d").unwrap().read_i32().unwrap(), expected);
        reopened.close().unwrap();
        let located = edit::locate_dataset_state(&store, address).unwrap();
        assert!(matches!(located.index, AppendIndex::Single(_)));
    }
    let before = store.data.clone();
    let retired = store.retired.clone();
    let AppendIndex::Single(index) = &state.index else {
        panic!("expected a Single Chunk index")
    };
    let error = match index.plan(
        &store,
        &state.datatype,
        &state.spatial,
        state.pipeline.as_ref(),
        LossyTailReencoding::Reject,
        &i32_bytes(&[4]),
        NonZeroU64::new(1).unwrap(),
        state.fill.pattern(state.element_size),
    ) {
        Ok(_) => panic!("append exceeds its finite maximum"),
        Err(error) => error,
    };
    assert!(
        matches!(&error,Error::AppendUnsupported(reason) if *reason=="append exceeds the dataset's maximum first dimension"),
        "{error:?}"
    );
    assert_eq!(store.data, before);
    assert_eq!(store.retired, retired);
    directory.close().unwrap();
}

#[rstest]
#[case::before_extent(2)]
#[case::after_extent(4)]
fn finite_chunk_lossy_completion_preserves_raw_prefix_until_extent_publication(
    #[case] phase: u8,
    #[values(FiniteChunkIndex::Fixed, FiniteChunkIndex::Single)] kind: FiniteChunkIndex,
) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_f32_data(&[1.234])
        .with_maxshape(&[MaxExtent::Fixed(kind.maximum())])
        .with_create_properties(
            crate::DatasetCreateProperties::new()
                .with_chunks(&[4])
                .with_scale_offset(crate::ScaleOffset::FloatDScale(2))
                .with_partial_edge_filtering(crate::PartialEdgeFiltering::Skip),
        );
    let mut store = WindowProbeStore::open(builder.finish().unwrap());
    let mut state = edit::locate_dataset_state(&store, dataset_address(&store)).unwrap();
    let raw = [2.345_f32, 3.456, 4.567]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    match &mut state.index {
        AppendIndex::Single(index) => {
            let plan = index
                .plan(
                    &store,
                    &state.datatype,
                    &state.spatial,
                    state.pipeline.as_ref(),
                    LossyTailReencoding::Reject,
                    &raw,
                    NonZeroU64::new(3).unwrap(),
                    state.fill.pattern(state.element_size),
                )
                .unwrap();
            index.apply(&mut store, plan, phase).unwrap();
        }
        AppendIndex::Fixed(index) => {
            let plan = index
                .plan(
                    &store,
                    &state.datatype,
                    &state.spatial,
                    state.pipeline.as_ref(),
                    LossyTailReencoding::Reject,
                    &raw,
                    NonZeroU64::new(3).unwrap(),
                    state.fill.pattern(state.element_size),
                )
                .unwrap();
            index.apply(&mut store, plan, phase).unwrap();
        }
        _ => panic!("expected a finite chunk index"),
    }
    assert_eq!(store.retired.len(), if phase == 4 { 2 } else { 1 });
    assert_eq!(
        *store.allocations.borrow(),
        [AllocationClass::Raw, AllocationClass::Raw]
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lossy-single.h5");
    fs::write(&path, &store.data).unwrap();
    let native = file::libhdf5_open(&path);
    let native_values = native.dataset("d").unwrap().read_raw::<f32>().unwrap();
    native.close().unwrap();
    let file = crate::File::from_bytes(store.data).unwrap();
    let actual = file.dataset("d").unwrap().read_f32().unwrap();
    if phase == 4 {
        assert_eq!(native_values.len(), 4);
        assert!(
            native_values
                .iter()
                .zip([1.234_f32, 2.345, 3.456, 4.567])
                .all(|(actual, expected)| (actual - expected).abs() < 0.011),
            "{native_values:?}"
        );
        assert_eq!(actual.len(), 4);
        assert!(
            actual
                .iter()
                .zip([1.234_f32, 2.345, 3.456, 4.567])
                .all(|(actual, expected)| (actual - expected).abs() < 0.011),
            "{actual:?}"
        );
    } else {
        assert_eq!(actual, [1.234]);
        assert_eq!(native_values, [1.234]);
    }
    file.close().unwrap();
    directory.close().unwrap();
}

#[derive(Clone, Copy)]
enum FixedInitial {
    Empty,
    Fill,
    Partial,
}

#[derive(Clone, Copy)]
enum SingleSource {
    Empty,
    Native,
    Rust,
    Unallocated,
}

#[derive(Clone, Copy)]
enum Producer {
    Native,
    Rust,
}

impl Producer {
    fn write_finite(
        self,
        path: &Path,
        widths: FormatWidths,
        maximum: usize,
        filtered: bool,
        values: &[i32],
    ) {
        match self {
            Self::Native => {
                if maximum == 4 && values.is_empty() {
                    dataset::write_native_empty_single_chunk_i32(path, widths, filtered);
                    return;
                }
                let native =
                    file::libhdf5_create_v110_with_widths(path, widths.offsets, widths.lengths, 0);
                let mut builder = native
                    .new_dataset_builder()
                    .empty_as(ByteOrder::LittleEndian.i32_datatype())
                    .shape((4..=maximum,))
                    .chunk((4,));
                if filtered {
                    builder = builder.deflate(1);
                }
                let dataset = builder.create("d").unwrap();
                dataset.resize((values.len(),)).unwrap();
                if !values.is_empty() {
                    dataset.write_raw(values).unwrap();
                }
                drop(dataset);
                native.close().unwrap();
            }
            Self::Rust => {
                let mut builder = FileBuilder::new();
                builder
                    .with_offset_width(widths.offsets)
                    .with_length_width(widths.lengths);
                let dataset = builder
                    .create_dataset("d")
                    .with_i32_data(values)
                    .with_shape(&[values.len().to_u64()])
                    .with_maxshape(&[MaxExtent::Fixed(maximum.to_u64())])
                    .with_chunks(&[4]);
                if filtered {
                    dataset.with_deflate(DeflateLevel::One);
                }
                fs::write(path, builder.finish().unwrap()).unwrap();
            }
        }
    }

    fn write(self, path: &Path, widths: FormatWidths, initial: i32, chunk: u64, filtered: bool) {
        let values: Vec<i32> = (0..initial).collect();
        match self {
            Self::Native => {
                let native =
                    file::libhdf5_create_v110_with_widths(path, widths.offsets, widths.lengths, 0);
                let mut builder = native
                    .new_dataset_builder()
                    .empty_as(ByteOrder::LittleEndian.i32_datatype())
                    .shape((usize::try_from(initial).unwrap()..,))
                    .chunk((usize::try_from(chunk).unwrap(),));
                if filtered {
                    builder = builder.deflate(1);
                }
                let dataset = builder.create("d").unwrap();
                if initial != 0 {
                    dataset.write_raw(&values).unwrap();
                }
                drop(dataset);
                native.close().unwrap();
            }
            Self::Rust => {
                let mut builder = FileBuilder::new();
                builder
                    .with_offset_width(widths.offsets)
                    .with_length_width(widths.lengths);
                let dataset = builder
                    .create_dataset("d")
                    .with_i32_data(&values)
                    .with_shape(&[u64::try_from(initial).unwrap()])
                    .with_maxshape(&[MaxExtent::Unlimited])
                    .with_chunks(&[chunk]);
                if filtered {
                    dataset.with_deflate(DeflateLevel::One);
                }
                fs::write(path, builder.finish().unwrap()).unwrap();
            }
        }
    }
}
