use std::fs;
use std::path::Path;

use rstest::rstest;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::AppendMode;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;

use crate::File;
use crate::FileAccessProperties;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MemoryStrategy;
use crate::address::AddressContext;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::WriteEngine;
use crate::message_type::MessageType;
use crate::source::BytesSource;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
#[case::none(FileSpaceStrategy::None)]
#[case::aggregate(FileSpaceStrategy::Aggr)]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn public_single_chunk_publication_matches_native_values_and_managers(
    #[case] strategy: FileSpaceStrategy,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
    #[values(false, true)] filtered: bool,
    #[values(AppendMode::Immediate, AppendMode::Buffered, AppendMode::OwnedBuffered)]
    operation: AppendMode,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("public-single.h5");
    dataset::write_single_chunk_i32(
        &path,
        FileSpaceSettings::new(strategy).with_persistence(matches!(
            strategy,
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
        )),
        filtered,
    );
    let native = file::libhdf5_open_rw(&path);
    native.link_hard("d", "alias").unwrap();
    native.close().unwrap();
    assert_single(&path, true);
    for value in 1_i32..4 {
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
        assert_single(&snapshot, true);
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
fn empty_single_chunk_buffers_publish_automatic_writes(#[case] owned: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("automatic.h5");
    dataset::write_native_empty_single_chunk_i32(
        &path,
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        true,
    );
    assert_single(&path, false);
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

fn assert_single(path: &Path, allocated: bool) {
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
            .expect("single-chunk fixture has a layout");
        if record.msg_type == MessageType::DATA_LAYOUT {
            break record.body;
        }
        position = record.body_range.end;
    };
    let DataLayout::Chunked {
        index: ChunkIndexLayout::SingleChunk { address, .. },
        ..
    } = DataLayout::parse_with_widths(message, widths).unwrap()
    else {
        panic!("expected a Single Chunk layout")
    };
    assert_eq!(address.is_some(), allocated);
    drop(dataset);
    file.close().unwrap();
}
