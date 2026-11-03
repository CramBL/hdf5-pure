use std::io::ErrorKind;

use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;

use crate::DeflateLevel;
use crate::File;
use crate::FileAccessProperties;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FillTime;
use crate::MemoryStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;
use crate::UserblockSize;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::address::StoredAddress;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::address::StoredAddressExt;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::chunked_write;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::chunked_write::ChunkOptions;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::chunked_write::StorageAllocation;
use crate::dataspace::MaxExtent;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::fill_value::FillPattern;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::filters::ChunkContext;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::MetadataCacheConfig;
use crate::type_builders;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

#[rstest]
#[case::empty_fixed(Creation::Fixed(0))]
#[case::single(Creation::Fixed(1))]
#[case::fixed(Creation::Fixed(40))]
#[case::fixed_sparse(Creation::FixedSparse)]
#[case::fixed_paged(Creation::Fixed(1_025))]
#[case::empty_extensible(Creation::Extensible(0))]
#[case::extensible(Creation::Extensible(40))]
#[case::rotated(Creation::Rotated(8))]
#[case::paged_parent(Creation::Rotated(20_000))]
#[case::pending_append(Creation::Pending)]
#[case::unallocated(Creation::Unallocated)]
fn staged_page_creation_preserves_values_and_complete_index_allocations(
    #[case] creation: Creation,
    #[values(512, 4_096)] page: u64,
    #[values(
        CreationAccess::Mirrored,
        CreationAccess::Bounded,
        CreationAccess::Userblock
    )]
    access: CreationAccess,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("created.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(access.base()).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1)
            .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
    );
    builder.create_dataset("kept").with_i32_data(&[31; 32]);
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(access.strategy())
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let mut dataset = file
        .root()
        .create_dataset("d", |builder| creation.configure(builder, false))
        .unwrap();
    if matches!(creation, Creation::Pending) {
        dataset
            .append_staged(|builder| {
                builder.append_i32(&[5, 6, 7, 8]);
            })
            .unwrap();
    }
    file.commit().unwrap();
    let header = dataset.header_address().unwrap();
    assert_eq!(dataset.read_i32().unwrap(), creation.expected());
    let mut coordinates: Vec<_> = dataset
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| chunk.offset)
        .collect();
    coordinates.sort();
    assert_eq!(coordinates, creation.coordinates());
    drop(dataset);
    file.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let index = engine.chunked_index_spans(header).unwrap();
    match creation {
        Creation::Fixed(0 | 1) => assert!(index.is_empty()),
        Creation::Fixed(_) | Creation::FixedSparse => assert_eq!(index.len(), 2),
        _ => assert!(index.len() >= 2),
    }
    if matches!(creation, Creation::Fixed(1_025) | Creation::Rotated(20_000)) {
        assert!(index.iter().any(|range| range.length() > page));
    }
    assert!(engine.space_ownership().unwrap().analysis_complete);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        creation.expected()
    );
    assert_eq!(file.dataset("kept").unwrap().read_i32().unwrap(), [31; 32]);
    file.close().unwrap();
}

#[rstest]
#[case::first_payload(0, 0, 16)]
#[case::partial_payload(1, 3, 16)]
#[case::array_header(40, 0, 28)]
#[case::array_data(41, 3, 338)]
fn failed_page_creation_preserves_reusable_state_and_existing_values(
    #[case] writes: usize,
    #[case] prefix: usize,
    #[case] requested: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("creation-failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1),
    );
    builder
        .create_dataset("hole")
        .with_u8_data(&vec![13; 8_192]);
    builder.create_dataset("kept").with_i32_data(&[31; 1_024]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.sync_policy = SyncPolicy::OnClose;
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    let before: Vec<_> = engine.space.admitted_snapshot().sections().collect();
    assert!(
        engine
            .space
            .reusable_sections()
            .any(|extent| extent.len() >= 640)
    );
    let mut builder = DatasetBuilder::new("d");
    Creation::Fixed(40).configure(&mut builder, false);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("d"), builder)
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(writes, prefix);
    let error = engine.commit().unwrap_err();
    let recording = disk_log::finish();
    let failed: Vec<_> = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                requested,
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            } => Some(*requested),
            _ => None,
        })
        .collect();
    assert_eq!(failed, [requested]);
    let Error::Io(error) = error else {
        panic!("expected an I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        before
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["kept"]);
    assert_eq!(
        file.dataset("kept").unwrap().read_i32().unwrap(),
        [31; 1_024]
    );
    file.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(256)]
#[case::large(2_048)]
fn page_creation_emits_retained_chunks_without_an_aggregate_payload_buffer(#[case] count: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("creation-memory.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let raw = vec![17; count * 1_024];
    let options = ChunkOptions {
        chunk_dims: Some(vec![256]),
        ..ChunkOptions::default()
    };
    let set = chunked_write::compress_chunks(
        &raw,
        &[u64::try_from(count * 256).unwrap()],
        ChunkContext::basic(&[256], 4),
        &options,
        None,
        FillPattern::ZERO,
        FillTime::IfSet,
        StorageAllocation::Allocated,
    )
    .unwrap();
    let geometry = set.array_geometry().unwrap();
    let writes = engine.image.issued_write_order().len();
    let (header, measured) =
        allocation::measure(&format!("page_creation_encoding_{count}"), || {
            engine.write_paged_chunk_set(&set, geometry).unwrap()
        });
    assert!(
        measured.peak_bytes < u64::try_from(raw.len() / 2).unwrap(),
        "{measured:?}, payload={}",
        raw.len()
    );
    assert!(!header.layout_message.is_empty());
    let issued = engine.image.issued_write_order();
    let payload = &issued[writes..writes + count];
    let start = payload[0].0;
    assert_eq!(
        payload,
        (0..count)
            .map(|index| (start + u64::try_from(index * 1_024).unwrap(), 1_024))
            .collect::<Vec<_>>()
    );
    assert_eq!(engine.image().read_exact_at(start, raw.len()).unwrap(), raw);
    let (control, copied) = allocation::measure(&format!("page_creation_assembly_{count}"), || {
        chunked_write::assemble_chunked_at(&set, StoredAddress::new(0)).unwrap()
    });
    assert!(control.data_bytes.starts_with(&raw));
    assert!(
        copied.live_bytes >= u64::try_from(raw.len()).unwrap(),
        "{copied:?}"
    );
}

#[derive(Clone, Copy)]
pub(in crate::edit) enum Creation {
    Extensible(usize),
    Fixed(usize),
    FixedSparse,
    Pending,
    Rotated(u64),
    Unallocated,
}

impl Creation {
    pub(in crate::edit) fn configure(self, builder: &mut DatasetBuilder, filtered: bool) {
        if matches!(self, Self::Unallocated) {
            builder
                .with_unallocated_storage(type_builders::make_i32_type(), &[8])
                .with_fill_value(-7_i32);
        } else {
            builder.with_i32_data(&self.initial());
        }
        match self {
            Self::Extensible(_) | Self::Pending | Self::Unallocated => {
                builder
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            }
            Self::Fixed(_) => {
                builder.with_chunks(&[4]);
            }
            Self::FixedSparse => {
                builder
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Fixed(320)]);
            }
            Self::Rotated(maximum) => {
                builder
                    .with_shape(&[2, 2])
                    .with_chunks(&[1, 1])
                    .with_maxshape(&[MaxExtent::Fixed(maximum), MaxExtent::Unlimited]);
            }
        }
        if filtered {
            builder.with_deflate(DeflateLevel::Six);
        }
    }

    pub(in crate::edit) fn expected(self) -> Vec<i32> {
        if matches!(self, Self::Pending) {
            return (1..=8).collect();
        }
        self.initial()
    }

    pub(in crate::edit) fn coordinates(self) -> Vec<Vec<u64>> {
        match self {
            Self::Extensible(count) | Self::Fixed(count) => (0..count)
                .map(|index| vec![u64::try_from(index * 4).unwrap()])
                .collect(),
            Self::FixedSparse => (0..40).map(|index| vec![index * 4]).collect(),
            Self::Pending => vec![vec![0], vec![4]],
            Self::Unallocated => Vec::new(),
            Self::Rotated(_) => vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
        }
    }

    fn initial(self) -> Vec<i32> {
        match self {
            Self::Extensible(count) | Self::Fixed(count) => (0..count * 4)
                .map(|index| i32::try_from(index / 16 % 7).unwrap())
                .collect(),
            Self::FixedSparse => Self::Fixed(40).initial(),
            Self::Pending => (1..=4).collect(),
            Self::Unallocated => vec![-7; 8],
            Self::Rotated(_) => vec![11, 12, 13, 14],
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::edit) enum CreationAccess {
    Bounded,
    Mirrored,
    Userblock,
}

impl CreationAccess {
    pub(in crate::edit) fn base(self) -> u64 {
        match self {
            Self::Bounded | Self::Mirrored => 0,
            Self::Userblock => 8_192,
        }
    }

    pub(in crate::edit) fn strategy(self) -> MemoryStrategy {
        match self {
            Self::Bounded => MemoryStrategy::Bounded,
            Self::Mirrored | Self::Userblock => MemoryStrategy::Mirrored,
        }
    }
}
