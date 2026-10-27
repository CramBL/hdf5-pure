//! Fractal-heap and PAGE publication recovery crosschecks against libhdf5.

use std::path::Path;

use rstest::rstest;
use test_util::publication as publication_bytes;
use test_util_hdf5::page_publication;
use test_util_hdf5::page_publication::PublicationStrategy;
use test_util_hdf5::paged;

use crate::AttrValue;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileSpaceStrategy;
use crate::SyncPolicy;
use crate::page_publication_failure::PagePublicationFailure;

#[cfg(feature = "deflate")]
test_util_hdf5::hdf5_bundled_tests! {
    use hdf5_pure_format::__private::{FormatWidths, FractalHeapHeader, FractalHeapIdLayout};
    use hdf5_pure_space::__private::Extent;
    use test_util_hdf5::heap_allocations::{AllocationKind, Heap};

    use crate::access_mode::AccessMode;
    use crate::address::{
        AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
    };
    use crate::convert::Narrow;
    use crate::fractal_heap::HeapObjectReader;
    use crate::fractal_heap_storage::{self, FractalHeapAllocation};
    use crate::source::BytesSource;
    use crate::width::{LengthWidth, OffsetWidth};

    #[cfg(feature = "deflate")]
    #[rstest::rstest]
    #[case::indirect(false)]
    #[case::direct(true)]
    fn c_written_filtered_huge_objects_have_exact_storage_and_contents(#[case] direct: bool) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("filtered_huge.h5");
        test_util_hdf5::fractal_heap::write_filtered_huge(&path, direct);
        let bytes = std::fs::read(&path).unwrap();
        let file = crate::reader::File::from_bytes(bytes.clone()).unwrap();
        let address = file.dataset("heap_address").unwrap().read_u64().unwrap()[0];
        let id = file.dataset("heap_id").unwrap().read_u8().unwrap();
        let header = FractalHeapHeader::parse(&bytes, address.to_usize().unwrap(), 8, 8).unwrap();
        let widths = FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        };
        let inspected = Heap::new(&bytes, address, &header);
        let storage = fractal_heap_storage::collect_fractal_heap_storage_from_source(
            &BytesSource::new(&bytes),
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(address),
            widths,
        )
        .unwrap();
        let expected: Vec<_> = inspected
            .allocations
            .iter()
            .map(|item| {
                let (at, length) = item.extent();
                let kind = match item.kind() {
                    AllocationKind::Header => FractalHeapAllocation::Header,
                    AllocationKind::HugeIndex => FractalHeapAllocation::HugeObjectIndex,
                    AllocationKind::HugeObject => FractalHeapAllocation::HugeObject,
                    kind => panic!("unexpected heap allocation: {kind:?}"),
                };
                (Extent::new(at, length).unwrap(), kind)
            })
            .collect();
        assert_eq!(storage.allocations(), expected);
        let expected: Vec<_> = (0..12_000)
            .map(|index| u8::try_from(index % 251).unwrap())
            .collect();
        let mut reader = HeapObjectReader::new(
            &header,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(address),
            AccessMode::ReadOnly,
            widths,
        );
        assert_eq!(
            reader
                .read(
                    &bytes,
                    FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
                        .parse(&id)
                        .unwrap()
                )
                .unwrap(),
            expected
        );
        let mut reader = HeapObjectReader::new(
            &header,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(address),
            AccessMode::ReadOnly,
            widths,
        );
        assert_eq!(
            reader
                .read_from_source(
                    &BytesSource::new(&bytes),
                    FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
                        .parse(&id)
                        .unwrap()
                )
                .unwrap(),
            expected
        );
    }
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
        file.group("group")
            .unwrap()
            .attrs()
            .unwrap()
            .get("attribute"),
        Some(&AttrValue::I32(9))
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
