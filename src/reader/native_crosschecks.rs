//! Dataset, free-space, and PAGE publication recovery crosschecks against libhdf5.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use rstest::rstest;
use test_util::publication as publication_bytes;
use test_util::temp;
use test_util_hdf5::file_space;
use test_util_hdf5::file_space::persisted::PersistedStrategy;
use test_util_hdf5::page_publication;
use test_util_hdf5::page_publication::PublicationStrategy;
use test_util_hdf5::paged;
use test_util_hdf5::reader::ReadBackend;

use crate::AttrValue;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::PhysicalAddress;
use crate::StoredAddress;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::address::StoredAddressExt;
use crate::chunk_cache::ChunkCacheConfig;
use crate::message_type::MessageType;
use crate::page_publication_failure::PagePublicationFailure;
use crate::parsed_chunk_index::test_util as chunk_index_test_util;
use crate::reader::test_util as reader_test_util;

#[rstest]
#[case::plain(0, 0)]
#[case::userblock(4096, 4096)]
#[case::inserted_userblock(0, 4096)]
#[case::forward_relocation(4096, 8192)]
#[case::removed_userblock(4096, 0)]
fn native_persisted_sections_preserve_stored_and_physical_coordinates(
    #[case] encoded_base: usize,
    #[case] signature: usize,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let original_path = directory.path().join("native-free-sections.h5");
    let relocated_path = directory.path().join("relocated-native-free-sections.h5");
    let expected = file_space::write_native_persisted_sections(
        &original_path,
        strategy,
        UserblockSize::try_from(u64::try_from(encoded_base).unwrap()).unwrap(),
    );
    let bytes = std::fs::read(&original_path).unwrap();
    let original = reader_test_util::open(&original_path, backend).unwrap();
    assert_eq!(
        original.file_space_settings().unwrap().unwrap(),
        FileSpaceSettings::new(match strategy {
            PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
            PersistedStrategy::Page => FileSpaceStrategy::Page,
        })
        .with_persistence(true)
    );
    let persisted = original.persisted_free_sections().unwrap().unwrap();
    assert_eq!(
        persisted
            .free_sections()
            .sections()
            .iter()
            .map(|section| (section.address().get(), section.length().get()))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        persisted.free_sections().free_bytes(),
        expected.iter().map(|(_, length)| length).sum::<u64>()
    );
    original.close().unwrap();
    let session = File::open_rw(&original_path).unwrap();
    let reusable = session.reusable_free_sections().unwrap();
    assert_eq!(reusable.section_count(), expected.len());
    assert_eq!(
        reusable.free_bytes(),
        persisted.free_sections().free_bytes()
    );
    assert_eq!(
        reusable.free_bytes(),
        session
            .reusable_space_stats()
            .unwrap()
            .reusable_free_bytes()
    );
    for (section, &(address, length)) in reusable.sections().iter().zip(&expected) {
        let stored = StoredAddress::new(address);
        let physical: PhysicalAddress = section.address();
        let end: PhysicalAddress = section.end_address();
        assert_eq!(physical, session.resolve_address(stored).unwrap());
        assert_eq!(
            physical.get(),
            u64::try_from(encoded_base)
                .unwrap()
                .checked_add(address)
                .unwrap()
        );
        assert_eq!(
            end,
            session
                .inner
                .addresses()
                .resolve(stored.checked_add(length).unwrap())
                .unwrap()
        );
        assert_eq!(section.length().get(), length);
    }
    session.close().unwrap();
    let mut relocated = vec![0; signature];
    relocated.extend_from_slice(&bytes[encoded_base..]);
    std::fs::write(&relocated_path, &relocated).unwrap();
    let reader = reader_test_util::open(&relocated_path, backend).unwrap();
    let report = reader.persisted_free_sections().unwrap().unwrap();
    assert_eq!(
        reader.superblock().base_address.get(),
        u64::try_from(encoded_base).unwrap()
    );
    assert_eq!(
        reader.inner.addresses().effective_base().get(),
        u64::try_from(signature).unwrap()
    );
    assert_eq!(report, persisted);
    assert!(report.manager_metadata_bytes() > 0);
    assert_eq!(report.free_sections().section_count(), expected.len());
    assert_eq!(
        report.free_sections().free_bytes(),
        report
            .managers()
            .iter()
            .map(|manager| manager.free_sections().free_bytes())
            .sum::<u64>()
    );
    assert_eq!(
        report.free_sections().section_count(),
        report
            .managers()
            .iter()
            .map(|manager| manager.free_sections().section_count())
            .sum::<usize>()
    );
    for (section, &(address, length)) in report.free_sections().sections().iter().zip(&expected) {
        let stored: StoredAddress = section.address();
        let end: StoredAddress = section.end_address();
        assert_eq!(stored.get(), address);
        assert_eq!(end.get(), address.checked_add(length).unwrap());
        let physical = reader.resolve_address(stored).unwrap();
        let physical_end = reader.resolve_address(end).unwrap();
        assert_eq!(
            physical.get(),
            u64::try_from(signature)
                .unwrap()
                .checked_add(address)
                .unwrap()
        );
        assert_eq!(physical.checked_add(length).unwrap(), physical_end);
        assert_eq!(reader.relative_address(physical).unwrap(), stored);
        assert_eq!(reader.relative_address(physical_end).unwrap(), end);
        assert_eq!(section.length().get(), length);
        assert!(physical_end.get() <= u64::try_from(relocated.len()).unwrap());
    }
    assert_eq!(reader.dataset("a").unwrap().read_i32().unwrap(), [1; 100]);
    assert_eq!(reader.dataset("c").unwrap().read_i32().unwrap(), [3; 100]);
    reader.close().unwrap();
    assert_eq!(std::fs::read(&relocated_path).unwrap(), relocated);
}

#[rstest]
#[case::dense(false, false)]
#[case::dense_unknown(false, true)]
#[case::sparse(true, false)]
#[case::sparse_unknown(true, true)]
fn typed_sweeps_read_reference_chunks_without_retaining_indexes(
    #[case] sparse: bool,
    #[case] unknown: bool,
    #[values(false, true)] streaming: bool,
) {
    let path = temp::temp_path("sweep.h5");
    let rows = 180_005;
    let expected = test_util_hdf5::numeric::write_chunked_sweep(&path, rows, sparse);
    let options = FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled());
    let file = if streaming {
        File::open_streaming_with_options(&path, options).unwrap()
    } else {
        File::open_with_options(&path, options).unwrap()
    };
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape().unwrap(), [rows as u64, 3]);
    let descriptors = dataset.chunks().unwrap().len();
    assert!(descriptors > 0);
    if unknown {
        let mut state = dataset.state.write().unwrap();
        let state = Arc::get_mut(state.as_mut().unwrap()).unwrap();
        let fill = state
            .header
            .messages
            .iter_mut()
            .find(|message| message.msg_type == MessageType::FILL_VALUE)
            .unwrap();
        fill.data = vec![u8::MAX];
    }
    chunk_index_test_util::reset();
    if sparse && unknown {
        let error = dataset.read_i32().unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::UnreadableFillValue)),
            "{error:?}"
        );
    } else {
        assert_eq!(dataset.read_i32().unwrap(), expected);
    }
    assert_eq!(chunk_index_test_util::counts(), (1, descriptors));
    assert!(!dataset.chunk_cache_stats().index_loaded());
    assert_eq!(dataset.chunk_cache_stats().cached_chunks(), 0);
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
#[case::published(PagePublicationFailure::AfterPublish)]
#[case::publication_sync(PagePublicationFailure::PublishedSync)]
#[case::completed_publication(PagePublicationFailure::AfterDrain)]
fn nonpersistent_page_publication_errors_preserve_c_created_objects(
    #[case] failure: PagePublicationFailure,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(false, true)] gathered: bool,
    #[values(false, true)] reopen: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recovery.h5");
    page_publication::write_origin(&path, 4096, 0);
    let original = std::fs::read(&path).unwrap();
    let mut file = File::open_page_publication_failure(&path, failure, sync, gathered).unwrap();
    stage_page_recovery_edits(&file);
    let error = file.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected publication failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected PAGE publication failure");
    let published = file.page_publication_snapshot().unwrap();
    assert_page_recovery_state(&File::from_bytes(published.clone()).unwrap(), &[]);
    assert_page_recovery_state(&file, &[]);
    assert_preserved_page_bytes(&original, &published);
    if reopen {
        file = reopen_page_recovery(file, &path, &published, sync);
    }
    file.root()
        .create_dataset("second", |dataset| {
            dataset.with_u8_data(&[11; 64]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_page_recovery_state(&file, &[("second", 11)]);
    file.close().unwrap();
    assert_preserved_page_bytes(&published, &std::fs::read(&path).unwrap());
    paged::assert_pages_homogeneous(&path, 4096, &["added", "replace", "second"]);
    assert_recovered_page_with_c(&path);
}

#[rstest]
#[case::persistent_page(PublicationStrategy::PersistentPage)]
#[case::fsm_aggr(PublicationStrategy::FsmAggr)]
#[case::aggr(PublicationStrategy::Aggr)]
#[case::none(PublicationStrategy::None)]
fn publication_failure_hook_rejects_other_strategies(#[case] strategy: PublicationStrategy) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsupported.h5");
    page_publication::write_unsupported_origin(&path, strategy);
    let before = std::fs::read(&path).unwrap();
    let Err(error) = File::open_page_publication_failure(
        &path,
        PagePublicationFailure::AfterPublish,
        SyncPolicy::Always,
        true,
    ) else {
        panic!("expected rejection of unsupported publication failure input");
    };
    assert!(
        matches!(error, Error::EditUnsupported(reason)
            if reason == "publication failure crosschecks require non-persistent PAGE allocation"),
        "{error:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

fn stage_page_recovery_edits(file: &File) {
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&[9; 64]);
        })
        .unwrap();
    file.dataset("replace")
        .unwrap()
        .write_staged(|dataset| {
            dataset.with_u8_data(&[17; 64]);
        })
        .unwrap();
    file.root().delete("remove").unwrap();
}

/// Edits an interrupted snapshot before reopening the original session's file.
fn reopen_page_recovery(file: File, path: &Path, published: &[u8], sync: SyncPolicy) -> File {
    let interrupted = path.with_file_name("interrupted.h5");
    std::fs::write(&interrupted, published).unwrap();
    let snapshot = File::open_rw_with_options(
        &interrupted,
        FileAccessProperties::new().with_sync_policy(sync),
    )
    .unwrap();
    snapshot
        .root()
        .create_dataset("reopened", |dataset| {
            dataset.with_u8_data(&[23; 64]);
        })
        .unwrap();
    snapshot.commit().unwrap();
    assert_page_recovery_state(&snapshot, &[("reopened", 23)]);
    snapshot.close().unwrap();
    assert_preserved_page_bytes(published, &std::fs::read(&interrupted).unwrap());
    drop(file);
    assert_eq!(std::fs::read(path).unwrap(), published);
    File::open_rw_with_options(path, FileAccessProperties::new().with_sync_policy(sync)).unwrap()
}

/// Checks published datasets, the surviving group attribute, and non-persistence.
fn assert_page_recovery_state(file: &File, additional: &[(&str, u8)]) {
    let mut expected = vec![("keep", 7), ("replace", 17), ("added", 9)];
    expected.extend_from_slice(additional);
    let mut paths = expected.iter().map(|(path, _)| *path).collect::<Vec<_>>();
    paths.sort_unstable();
    let mut actual = file.root().datasets().unwrap();
    actual.sort_unstable();
    assert_eq!(actual, paths);
    for (path, value) in expected {
        assert_eq!(file.dataset(path).unwrap().read_u8().unwrap(), [value; 64]);
    }
    assert_eq!(file.root().groups().unwrap(), ["group"]);
    assert_eq!(
        file.group("group").unwrap().attrs().unwrap(),
        HashMap::from([("attribute".into(), AttrValue::I32(9))])
    );
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert_eq!(info.page_size.get(), 4096);
    assert!(!info.persist);
    assert!(info.manager_addrs.is_empty());
    if file.edit_backing().is_some() {
        let error = file.persisted_free_sections().unwrap_err();
        assert!(
            matches!(error, Error::FreeSectionInspectionUnsupported),
            "{error:?}"
        );
        assert_eq!(file.reusable_free_sections().unwrap().section_count(), 0);
    } else {
        assert_eq!(file.persisted_free_sections().unwrap(), None);
    }
}

/// Checks that the original bytes following the superblock are preserved.
fn assert_preserved_page_bytes(before: &[u8], after: &[u8]) {
    let file = File::from_bytes(before.to_vec()).unwrap();
    let superblock_len = hdf5_pure_format::__private::serialize_superblock(file.superblock())
        .unwrap()
        .len();
    publication_bytes::assert_preserved_bytes(before, after, 0, superblock_len);
}

/// Checks and mutates the recovered file through libhdf5, then reads it through the facade.
fn assert_recovered_page_with_c(path: &Path) {
    page_publication::assert_recovered_and_mutate(path);
    assert_page_recovery_state(&File::open(path).unwrap(), &[("second", 29)]);
}
