//! Shared fixtures for bounded sessions and gathered writes.

use std::path::Path;

use crate::FileSpaceSettings;
use crate::access_mode::AccessMode;
use crate::dataspace::MaxExtent;
use crate::edit::MemoryStrategy;
use crate::edit::WriteEngine;
use crate::file_lock::FileLocking;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::object_path::ObjectPath;

/// Writes an unlimited, chunked `i32` dataset named `d`, initialized with `0..n`.
pub(super) fn build_appendable(path: &Path, n: i32, chunk: u64) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = crate::writer::FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[chunk]);
    b.write(path).unwrap();
}

/// Opens a bounded editor with locking enabled and the metadata cache disabled.
pub(super) fn open_bounded_session(path: &Path) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        crate::source::MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
    )
    .unwrap()
}

/// Resolves dataset `d` to its absolute position in the session image.
pub(super) fn dataset_addr(engine: &WriteEngine) -> u64 {
    crate::group_v2::resolve_path_any_from_source(
        &engine.image(),
        AccessMode::ReadWrite,
        engine.superblock(),
        engine.widths(),
        engine.addresses(),
        &ObjectPath::parse("d"),
    )
    .unwrap()
}

/// Writes `tables` unlimited datasets with optional persistent PAGE storage.
///
/// Paged fixtures use 16 KiB pages to distinguish the file's page size from the gather default.
pub(super) fn gather_fixture(path: &std::path::Path, tables: usize, paged: bool) {
    use crate::writer::FileBuilder;
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::try_from(16 * 1024).unwrap()),
        );
    }
    for t in 0..tables {
        b.create_dataset(&std::format!("t{t}"))
            .with_i32_data(&(0..256).collect::<Vec<_>>())
            .with_shape(&[256])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[64]);
    }
    b.write(path).unwrap();
}
