use std::path::Path;

use rstest::rstest;
use test_util::disk_log::{self, DiskOp};
use test_util_hdf5::{file_space_oracle, large_page};

use crate::edit::WriteEngine;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::type_builders::DatasetBuilder;
use crate::{
    File, FileAccessProperties, FileBuilder, FileLocking, FileSpacePageSize, FileSpaceSettings,
    FileSpaceStrategy, MemoryStrategy, PersistedFreeSpacePolicy, SyncPolicy,
};

#[rstest]
#[case::rust(false)]
#[case::native(true)]
fn large_page_small_commits_preserve_native_values_and_free_sections(
    #[case] native: bool,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reopen.h5");
    build(&path, native);
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    match buffering {
        Buffering::Unbuffered => engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap(),
        Buffering::Operation => engine
            .set_write_buffering(WriteBuffering::Operation {
                page_size: PAGE,
                max_bytes: 131_072,
            })
            .unwrap(),
        Buffering::Session => engine
            .set_page_buffer_size(usize::try_from(PAGE).unwrap())
            .unwrap(),
    }
    let initial_len = usize::try_from(engine.image.len()).unwrap();
    disk_log::start();
    let mut first = DatasetBuilder::new("first");
    first.with_i32_data(&[11, 12, 13, 14]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("first"), first)
        .unwrap();
    engine.commit().unwrap();
    let mut second = DatasetBuilder::new("second");
    second.with_i32_data(&[21, 22, 23]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("second"), second)
        .unwrap();
    engine.commit().unwrap();
    engine.force_sync().unwrap();
    engine.release_status_flags().unwrap();
    let written = disk_log::take()
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { bytes, .. } => Some(bytes.len() as u64),
            DiskOp::SetLen(_) => None,
        })
        .sum::<u64>();
    assert!(written > PAGE / 2);
    let length = engine.image.len();
    drop(engine);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(u64::try_from(bytes.len()).unwrap(), length);
    assert!(
        bytes[initial_len..]
            .split(|&byte| byte != 0)
            .any(|run| run.len() > usize::try_from(PAGE / 2).unwrap())
    );
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.superblock().eof_address, length);
    assert_eq!(
        reader.file_space_info().unwrap().unwrap().page_size.get(),
        PAGE
    );
    assert_eq!(
        reader.dataset("first").unwrap().read_i32().unwrap(),
        [11, 12, 13, 14]
    );
    assert_eq!(
        reader.dataset("second").unwrap().read_i32().unwrap(),
        [21, 22, 23]
    );
    let sections = reader.persisted_free_sections().unwrap().unwrap();
    assert!(sections.free_sections().free_bytes() > PAGE / 2);
    reader.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
    let metadata = file_space_oracle::assert_manager_metadata_bytes(&path);
    assert!(metadata > 0);
    assert!(metadata < PAGE);
    let reopened = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    reopened
        .root()
        .create_dataset("third", |builder| {
            builder.with_i32_data(&[31, 32]);
        })
        .unwrap();
    reopened.commit().unwrap();
    reopened.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
    large_page::assert_native(&path, PAGE);
}

#[derive(Clone, Copy)]
enum Buffering {
    Operation,
    Session,
    Unbuffered,
}

fn build(path: &Path, native: bool) {
    if native {
        large_page::write_native(path, PAGE);
    } else {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
        );
        builder.write(path).unwrap();
    }
}

const PAGE: u64 = 1_048_576;
