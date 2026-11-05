use core::num::NonZeroUsize;
use std::io::ErrorKind;
use std::sync::Mutex;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;

use crate::File;
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::SyncPolicy;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_write::ChunkMeta;
use crate::chunked_write::ChunkProvider;
use crate::chunked_write::VerbatimChunks;
use crate::convert;
use crate::data_layout::ChunkedLayoutFlags;
use crate::edit::WriteEngine;
use crate::edit::commit::PublicationState;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::writer::FileBuilder;

#[rstest]
#[case::short(ProviderFailure::Short)]
#[case::excess(ProviderFailure::Excess)]
#[case::provider(ProviderFailure::Error)]
fn failed_provider_output_keeps_all_reserved_capacity_unavailable(
    #[case] failure: ProviderFailure,
) {
    let (_directory, mut engine, original) = engine();
    let rollback = engine.snapshot_free();
    let provider = Provider::new(40, 16, failure);
    let error = engine
        .write_paged_verbatim_chunks(prepared(40, 16), &provider)
        .unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a provider format error, got {error:?}");
    };
    match failure {
        ProviderFailure::Error => assert_eq!(
            error,
            FormatError::InvalidChunkGeometry("injected provider failure")
        ),
        ProviderFailure::Excess | ProviderFailure::Short => assert_eq!(
            error,
            FormatError::ChunkedReadError(
                "verbatim chunk provider returned a chunk whose size differs from the planned size"
                    .into()
            )
        ),
        ProviderFailure::None => panic!("expected a failing provider"),
    }
    assert_eq!(*provider.calls.lock().unwrap(), [0, 1, 2]);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    engine.restore_free(rollback);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        original
    );
    let address = engine
        .context
        .addresses()
        .resolve(original[0].start())
        .unwrap();
    assert_eq!(
        engine.image().read_exact_at(address.get(), 16).unwrap(),
        [17; 16]
    );
}

#[rstest]
#[case::payload(2, 16, 3)]
#[case::header(40, 28, 40)]
#[case::data_block(41, 338, 40)]
fn partial_provider_writes_retain_payload_and_complete_index_reservations(
    #[case] completed: usize,
    #[case] requested: usize,
    #[case] calls: usize,
) {
    let (_directory, mut engine, _) = engine();
    let provider = Provider::new(40, 16, ProviderFailure::None);
    disk_log::start();
    disk_log::fail_write_after(completed, 3);
    let error = engine
        .write_paged_verbatim_chunks(prepared(40, 16), &provider)
        .unwrap_err();
    let recording = disk_log::finish();
    let Error::Io(error) = error else {
        panic!("expected an I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    let failed: Vec<_> = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                requested,
                written,
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            } => Some((*requested, *written)),
            _ => None,
        })
        .collect();
    assert_eq!(failed, [(requested, 3)]);
    assert_eq!(
        *provider.calls.lock().unwrap(),
        (0..calls).collect::<Vec<_>>()
    );
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
}

#[test]
fn empty_provider_storage_is_rejected_before_reservation() {
    let (_directory, mut engine, _) = engine();
    let before: Vec<_> = engine.space.admitted_snapshot().sections().collect();
    let meta = vec![
        ChunkMeta {
            compressed_size: 0,
            filter_mask: 0
        };
        2
    ];
    let prepared = VerbatimChunks::new(
        crate::file_writer::WIDTHS,
        &meta,
        &[2],
        &[1],
        convert::nz(1),
        ChunkedLayoutFlags::NONE,
        None,
        StoredAddress::ZERO,
        None,
    )
    .unwrap();
    let provider = Provider::new(2, 1, ProviderFailure::None);
    let error = engine
        .write_paged_verbatim_chunks(prepared, &provider)
        .unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected invalid chunk storage, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::InvalidChunkGeometry("verbatim chunk storage is empty")
    );
    assert!(provider.calls.lock().unwrap().is_empty());
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        before
    );
}

#[test]
fn uncertain_publication_rejects_provider_placement_without_reading_chunks() {
    let (_directory, mut engine, _) = engine();
    let before: Vec<_> = engine.space.admitted_snapshot().sections().collect();
    engine.publication = PublicationState::Uncertain;
    let provider = Provider::new(40, 16, ProviderFailure::None);
    let error = engine
        .write_paged_verbatim_chunks(prepared(40, 16), &provider)
        .unwrap_err();
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    assert!(provider.calls.lock().unwrap().is_empty());
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        before
    );
}

#[test]
fn failed_staged_chunk_copy_restores_original_free_space_before_publication() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("copy-failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1),
    );
    builder
        .create_dataset("d")
        .with_i32_data(&[0; 160])
        .with_chunks(&[4]);
    builder.create_dataset("hole").with_u8_data(&[7; 8_192]);
    builder.create_dataset("kept").with_u8_data(&[31; 4_096]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.sync_policy = SyncPolicy::OnClose;
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    let before: Vec<_> = engine.space.admitted_snapshot().sections().collect();
    engine
        .copy(&ObjectPathBuf::parse("d"), &ObjectPathBuf::parse("copied"))
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(2, 3);
    let error = engine.commit().unwrap_err();
    let recording = disk_log::finish();
    let Error::Io(error) = error else {
        panic!("expected an I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
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
    assert_eq!(failed, [16]);
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
    assert_eq!(file.root().datasets().unwrap(), ["d", "kept"]);
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [0; 160]);
    assert_eq!(
        file.dataset("kept").unwrap().read_u8().unwrap(),
        [31; 4_096]
    );
    file.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(512)]
#[case::large(2_048)]
fn provider_emission_reuses_one_buffer_without_collecting_payloads(#[case] count: usize) {
    let (_directory, mut engine, _) = engine();
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::MIN,
        [],
    )
    .unwrap();
    assert!(engine.image.as_slice().is_none());
    let provider = Provider::new(count, 1_024, ProviderFailure::None);
    let prepared = prepared(count, 1_024);
    let (message, measured) =
        test_util::allocation::measure(&format!("page_provider_emission_{count}"), || {
            engine
                .write_paged_verbatim_chunks(prepared, &provider)
                .unwrap()
        });
    assert!(!message.is_empty());
    assert!(
        measured.peak_bytes < u64::try_from(count * 512).unwrap(),
        "{measured:?}"
    );
    assert_eq!(
        *provider.calls.lock().unwrap(),
        (0..count).collect::<Vec<_>>()
    );
    let capacities = provider.capacities.lock().unwrap();
    assert_eq!(capacities[0], 0);
    assert!(capacities[1..].iter().all(|&capacity| capacity == 1_024));
    let meta = vec![
        ChunkMeta {
            compressed_size: 1_024,
            filter_mask: 0
        };
        count
    ];
    let layout = crate::chunked_write::plan_chunked_data_verbatim(
        crate::file_writer::WIDTHS,
        &meta,
        &[u64::try_from(count * 1_024).unwrap()],
        &[1_024],
        convert::nz(1),
        ChunkedLayoutFlags::NONE,
        None,
        StoredAddress::ZERO,
        None,
    )
    .unwrap();
    let control = Provider::new(count, 1_024, ProviderFailure::None);
    let (bytes, copied) =
        test_util::allocation::measure(&format!("provider_aggregate_{count}"), || {
            let mut bytes = Vec::new();
            crate::chunked_write::emit_chunked_data_verbatim(&mut bytes, &layout.plan, &control)
                .unwrap();
            bytes
        });
    assert_eq!(bytes.len() as u64, layout.plan.total_len);
    assert!(
        copied.live_bytes >= u64::try_from(count * 1_024).unwrap(),
        "{copied:?}"
    );
}

fn engine() -> (tempfile::TempDir, WriteEngine, Vec<Extent>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("provider.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::MIN),
    );
    builder.create_dataset("kept").with_i32_data(&[31; 16]);
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine.sync_policy = SyncPolicy::OnClose;
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.begin_page(PageType::Raw).unwrap();
    let raw = engine.append(&[0; 1_024]).unwrap();
    engine.begin_page(PageType::Meta).unwrap();
    let metadata = engine.append(&[0; 512]).unwrap();
    let addresses = engine.context.addresses();
    let original = vec![
        Extent::new(
            addresses.relative(PhysicalAddress::new(raw)).unwrap(),
            ByteLength::new(640),
        )
        .unwrap(),
        Extent::new(
            addresses.relative(PhysicalAddress::new(metadata)).unwrap(),
            ByteLength::new(28),
        )
        .unwrap(),
        Extent::new(
            addresses
                .relative(PhysicalAddress::new(metadata + 64))
                .unwrap(),
            ByteLength::new(338),
        )
        .unwrap(),
    ];
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::MIN,
        [
            (
                Some(ManagerKind::PagedLargeGeneric),
                Extent::new(original[0].start(), ByteLength::new(512)).unwrap(),
            ),
            (
                Some(ManagerKind::PagedSmallRaw),
                Extent::new(
                    original[0].start().checked_add(512).unwrap(),
                    ByteLength::new(128),
                )
                .unwrap(),
            ),
            (Some(ManagerKind::PagedSmallMetadata), original[1]),
            (Some(ManagerKind::PagedSmallMetadata), original[2]),
        ],
    )
    .unwrap();
    (directory, engine, original)
}

fn prepared(count: usize, length: usize) -> VerbatimChunks<'static> {
    let meta = vec![
        ChunkMeta {
            compressed_size: u64::try_from(length).unwrap(),
            filter_mask: 0
        };
        count
    ];
    VerbatimChunks::new(
        crate::file_writer::WIDTHS,
        &meta,
        &[u64::try_from(count * length).unwrap()],
        &[u64::try_from(length).unwrap()],
        convert::nz(1),
        ChunkedLayoutFlags::NONE,
        None,
        StoredAddress::ZERO,
        None,
    )
    .unwrap()
}

#[derive(Clone, Copy)]
enum ProviderFailure {
    Error,
    Excess,
    None,
    Short,
}

struct Provider {
    count: usize,
    length: NonZeroUsize,
    failure: ProviderFailure,
    calls: Mutex<Vec<usize>>,
    capacities: Mutex<Vec<usize>>,
}

impl Provider {
    fn new(count: usize, length: usize, failure: ProviderFailure) -> Self {
        Self {
            count,
            length: convert::nz(length),
            failure,
            calls: Mutex::new(Vec::with_capacity(count)),
            capacities: Mutex::new(Vec::with_capacity(count)),
        }
    }
}

impl ChunkProvider for Provider {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let Self {
            count,
            length,
            failure,
            calls,
            capacities,
        } = self;
        if index >= *count {
            return Err(FormatError::Internal(
                "test provider index exceeds available chunks".into(),
            ));
        }
        calls.lock().unwrap().push(index);
        capacities.lock().unwrap().push(out.capacity());
        out.resize(length.get(), 17);
        if index == 2 {
            match failure {
                ProviderFailure::Error => {
                    return Err(FormatError::InvalidChunkGeometry(
                        "injected provider failure",
                    ));
                }
                ProviderFailure::Excess => out.push(19),
                ProviderFailure::Short => out.truncate(length.get() - 1),
                ProviderFailure::None => {}
            }
        }
        Ok(())
    }
}
