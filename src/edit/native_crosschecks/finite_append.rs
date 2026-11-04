use std::fs;
use std::path::Path;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use rstest::rstest;
use test_util_hdf5::attr;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::AppendMode;
use test_util_hdf5::dataset::FiniteChunkIndex;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::numeric::ByteOrder;

use crate::DeflateLevel;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::address::AddressContext;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::message_type::MessageType;
use crate::source::BytesSource;
use crate::type_builders;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
#[case::none(FileSpaceStrategy::None)]
#[case::aggregate(FileSpaceStrategy::Aggr)]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn public_finite_chunk_publication_matches_native_values_and_managers(
    #[case] strategy: FileSpaceStrategy,
    #[values(FiniteChunkIndex::Fixed, FiniteChunkIndex::Single)] index: FiniteChunkIndex,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
    #[values(false, true)] filtered: bool,
    #[values(AppendMode::Immediate, AppendMode::Buffered, AppendMode::OwnedBuffered)]
    operation: AppendMode,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("public-single.h5");
    dataset::write_finite_chunked_i32(
        &path,
        FileSpaceSettings::new(strategy).with_persistence(matches!(
            strategy,
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
        )),
        filtered,
        index,
    );
    let native = file::libhdf5_open_rw(&path);
    native.link_hard("d", "alias").unwrap();
    native.close().unwrap();
    assert_index(&path, index, true);
    for value in 1_i32..i32::try_from(index.maximum()).unwrap() {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(backing),
        )
        .unwrap();
        let mut target = file.dataset("d").unwrap();
        match operation {
            AppendMode::Immediate => target.append(&[value]).unwrap(),
            AppendMode::Buffered => {
                let mut appender = target.buffered_appender().unwrap();
                appender.append(&[value]).unwrap();
                appender.finish().unwrap();
            }
            AppendMode::OwnedBuffered => {
                let mut appender = target.clone().into_buffered_appender().unwrap();
                appender.append(&[value]).unwrap();
                appender.finish().unwrap();
            }
            AppendMode::Staged => panic!("unexpected staged append mode"),
        }
        let expected = (0..=value).collect::<Vec<_>>();
        assert_eq!(target.read_i32().unwrap(), expected);
        assert_eq!(file.dataset("alias").unwrap().read_i32().unwrap(), expected);
        let snapshot = directory.path().join("issued.h5");
        fs::write(&snapshot, file.disk_bytes().unwrap()).unwrap();
        assert_index(&snapshot, index, true);
        let native = file::libhdf5_open(&snapshot);
        assert_eq!(
            native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            expected
        );
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
            (8, 8)
        );
        native.close().unwrap();
        if matches!(
            strategy,
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
        ) {
            file_space_oracle::assert_persisted_sections(&snapshot);
            file_space_oracle::assert_manager_metadata_bytes(&snapshot);
        }
        drop(target);
        file.close().unwrap();
    }
    directory.close().unwrap();
}

#[rstest]
#[case::borrowed(false)]
#[case::owned(true)]
fn empty_finite_chunk_buffers_publish_automatic_writes(
    #[case] owned: bool,
    #[values(FiniteChunkIndex::Fixed, FiniteChunkIndex::Single)] index: FiniteChunkIndex,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("automatic.h5");
    dataset::write_native_empty_finite_chunked_i32(
        &path,
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        true,
        index,
    );
    assert_index(&path, index, false);
    let file = File::open_rw(&path).unwrap();
    let mut target = file.dataset("d").unwrap();
    if owned {
        let mut appender = target.into_buffered_appender().unwrap();
        appender.append(&[0_i32, 1, 2, 3]).unwrap();
        assert_eq!(appender.unwritten(), b"");
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0, 1, 2, 3]);
        let before = file.disk_bytes().unwrap();
        appender.finish().unwrap();
        assert_eq!(file.disk_bytes().unwrap(), before);
        target = file.dataset("d").unwrap();
    } else {
        let mut appender = target.buffered_appender().unwrap();
        appender.append(&[0_i32, 1, 2, 3]).unwrap();
        assert_eq!(appender.unwritten(), b"");
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0, 1, 2, 3]);
        let before = file.disk_bytes().unwrap();
        appender.finish().unwrap();
        assert_eq!(file.disk_bytes().unwrap(), before);
    }
    drop(target);
    file.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        [0, 1, 2, 3]
    );
    native.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::small_pages(512)]
#[case::large_pages(4_096)]
fn deferred_fixed_arrays_preserve_complete_page_metadata_allocations(
    #[case] page: u64,
    #[values(false, true)] native_source: bool,
    #[values(false, true)] filtered: bool,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("deferred-fixed.h5");
    let page = FileSpacePageSize::try_from(page).unwrap();
    if native_source {
        let native = file::libhdf5_create_page(&path, page, UserblockSize::NONE, true);
        let mut builder = native
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype())
            .shape((4..=4_100,))
            .chunk((4,));
        if filtered {
            builder = builder.deflate(1);
        }
        let dataset = builder.create("d").unwrap();
        dataset.resize((0,)).unwrap();
        drop(dataset);
        native.close().unwrap();
    } else {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_page_size(page)
                .with_persistence(true),
        );
        let dataset = builder
            .create_dataset("d")
            .with_unallocated_storage(type_builders::make_i32_type(), &[0])
            .with_maxshape(&[MaxExtent::Fixed(4_100)])
            .with_chunks(&[4]);
        if filtered {
            dataset.with_deflate(DeflateLevel::One);
        }
        builder.write(&path).unwrap();
    }
    assert_index(&path, FiniteChunkIndex::Fixed, false);
    let mut previous = 0;
    for end in [3, 4_097, 4_100] {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(backing)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        let mut dataset = file.dataset("d").unwrap();
        dataset
            .append(&(previous..end).collect::<Vec<i32>>())
            .unwrap();
        assert_eq!(dataset.read_i32().unwrap(), (0..end).collect::<Vec<_>>());
        let address = dataset.header_address().unwrap();
        drop(dataset);
        file.close().unwrap();
        let engine = edit_test_util::open_engine(&path);
        engine.assert_array_metadata_pages(address);
        assert_eq!(engine.chunked_index_spans(address).unwrap().len(), 2);
        drop(engine);
        let native = file::libhdf5_open(&path);
        assert_eq!(
            native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            (0..end).collect::<Vec<_>>()
        );
        let widths = native.create_plist().unwrap().get_sizes().unwrap();
        assert_eq!(
            (
                usize::from(widths.sizeof_addr),
                usize::from(widths.sizeof_size)
            ),
            (8, 8)
        );
        native.close().unwrap();
        file_space_oracle::assert_persisted_sections(&path);
        file_space_oracle::assert_manager_metadata_bytes(&path);
        previous = end;
    }
}

fn assert_index(path: &Path, index: FiniteChunkIndex, allocated: bool) {
    let file = File::open(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let address = dataset.header_address().unwrap();
    let bytes = fs::read(path).unwrap();
    let source = BytesSource::new(&bytes);
    let addresses = AddressContext::from_effective_base(file.superblock().base_address);
    let widths = FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    };
    let region = WriteEngine::gather_oh_messages(&source, address, addresses, widths).unwrap();
    let mut position = 0;
    let message = loop {
        let record = region
            .layout()
            .next_message(&region, position)
            .unwrap()
            .expect("finite fixture has a layout");
        if record.msg_type == MessageType::DATA_LAYOUT {
            break record.body;
        }
        position = record.body_range.end;
    };
    let DataLayout::Chunked { index: actual, .. } =
        DataLayout::parse_with_widths(message, widths).unwrap()
    else {
        panic!("expected a chunked layout")
    };
    let address = match (index, actual) {
        (FiniteChunkIndex::Single, ChunkIndexLayout::SingleChunk { address, .. }) => address,
        (FiniteChunkIndex::Fixed, ChunkIndexLayout::FixedArray { address }) => {
            if let Some(address) = address {
                let locator =
                    FixedArrayHeaderLocator::new(widths, addresses.effective_base(), address);
                let range = locator.parse_range().unwrap();
                let from = usize::try_from(range.start().get()).unwrap();
                let end = usize::try_from(range.end().get()).unwrap();
                let header = locator
                    .parse(FileRegion::new(range, &bytes[from..end]).unwrap())
                    .unwrap();
                assert_eq!(header.entries(), index.maximum().div_ceil(4));
                assert!(header.data_block().is_some());
            }
            address
        }
        (index, actual) => panic!("expected {index:?}, got {actual:?}"),
    };
    assert_eq!(address.is_some(), allocated);
    drop(dataset);
    file.close().unwrap();
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
fn native_reads_preserved_variable_length_append_siblings(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] memory: MemoryStrategy,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("siblings.h5");
    edit_test_util::append_with_variable_length_siblings(
        &path,
        FormatWidths { offsets, lengths },
        memory,
        strategy,
    );
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("series").unwrap().read_raw::<i32>().unwrap(),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(
        native.dataset("single").unwrap().read_raw::<i32>().unwrap(),
        [1, 2, 3, 4]
    );
    assert_eq!(attr::unicode_array(&native, "labels"), ["alpha", "", "ζ"]);
    assert_eq!(
        attr::unicode_array(&native.dataset("series").unwrap(), "labels"),
        ["alpha", "", "ζ"]
    );
    let sizes = native.create_plist().unwrap().get_sizes().unwrap();
    assert_eq!(
        (
            usize::from(sizes.sizeof_addr),
            usize::from(sizes.sizeof_size)
        ),
        (usize::from(offsets.get()), usize::from(lengths.get()))
    );
    native.close().unwrap();
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "strings"),
        ["alpha", "", "ζ"]
    );
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
    directory.close().unwrap();
}
