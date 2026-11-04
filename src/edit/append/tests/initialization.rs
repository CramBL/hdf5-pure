use std::io::ErrorKind;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use hdf5_pure_format::__private::ExtensibleArrayInitialization;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::PageTransition;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::disk_log;

use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::image::WriteBuffering;
use crate::source::Source;
use crate::width::FormatWidths;

#[rstest]
#[case::small_append(Placement::Append, [32, 4, 4, 16, 10])]
#[case::small_reuse(Placement::Reusable, [24, 8, 2, 8, 8])]
#[case::large_append(Placement::Append, [32, 255, 4, 16, 10])]
#[case::large_reuse(Placement::Reusable, [32, 255, 4, 16, 10])]
fn empty_index_blocks_use_complete_metadata_allocations(
    #[case] placement: Placement,
    #[case] parameters: [u8; 5],
    #[values(512, 4_096)] page: u64,
    #[values(false, true)] filtered: bool,
) {
    let initialization = initialization(parameters, filtered);
    let (_directory, mut engine, raw) = engine(page, placement, &initialization);
    let initial_end = engine.image.len();
    let [header, index] = engine
        .store()
        .allocate_paged_initialization(&initialization)
        .unwrap();
    let ranges = [header.range(), index.range()];
    assert!(!ranges[0].overlaps(ranges[1]));
    for (range, length) in ranges
        .into_iter()
        .zip(initialization.plan().block_lengths())
    {
        assert_eq!(range.length(), length);
        assert!(range.end().get() <= engine.image.len());
        if length < page {
            assert_eq!(range.start().get() / page, (range.end().get() - 1) / page);
        } else {
            assert_eq!(range.start().get() % page, 0);
        }
        match placement {
            Placement::Append => assert!(range.start().get() >= initial_end),
            Placement::Reusable => assert!(range.end().get() <= initial_end),
        }
    }
    if let Some(raw) = raw {
        assert!(engine.space.reusable_sections().any(|extent| extent == raw));
    }
    let addresses = engine.context.addresses();
    let locator = ExtensibleArrayHeaderLocator::new(
        engine.context.widths(),
        addresses.effective_base(),
        addresses.relative(ranges[0].start()).unwrap(),
    );
    let bytes = engine
        .image
        .read_exact_at(
            ranges[0].start().get(),
            usize::try_from(ranges[0].length()).unwrap(),
        )
        .unwrap();
    let view = locator
        .parse(FileRegion::new(ranges[0], &bytes).unwrap())
        .unwrap();
    assert_eq!(view.index_block().unwrap().range(), ranges[1]);
    assert_eq!(view.header().idx_blk_elmts, parameters[1]);
    let error = engine.context.allocations.before_reference().unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reference publication precedes allocation ownership publication"),
        "{error:?}"
    );
}

#[rstest]
#[case::header(Placement::Reusable, 0, 0)]
#[case::index(Placement::Reusable, 1, 0)]
#[case::partial_index(Placement::Reusable, 1, 7)]
#[case::partial_gap(Placement::Append, 1, 3)]
#[case::appended_index(Placement::Append, 2, 7)]
fn failed_initialization_restores_unpublished_allocation_state(
    #[case] placement: Placement,
    #[case] writes: usize,
    #[case] prefix: usize,
) {
    let initialization = initialization([32, 4, 4, 16, 10], false);
    let (_directory, mut engine, _) = engine(512, placement, &initialization);
    let before = engine.space.reusable_sections().collect::<Vec<_>>();
    let length = engine.image.len();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    disk_log::fail_write_after(writes, prefix);
    let error = match engine
        .store()
        .allocate_paged_initialization(&initialization)
    {
        Ok(_) => panic!("expected initialization write error"),
        Err(error) => error,
    };
    let Error::Io(error) = error else {
        panic!("expected I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(engine.image.len(), length);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), before);
    engine.context.allocations.before_reference().unwrap();
}

#[test]
fn uncertain_publication_rejects_initialization_before_allocating() {
    let initialization = initialization([32, 4, 4, 16, 10], false);
    let (_directory, mut engine, _) = engine(512, Placement::Reusable, &initialization);
    let sections = engine.space.reusable_sections().collect::<Vec<_>>();
    let length = engine.image.len();
    engine.make_allocation_uncertain();
    disk_log::start();
    let error = match engine
        .store()
        .allocate_paged_initialization(&initialization)
    {
        Ok(_) => panic!("expected uncertain publication"),
        Err(error) => error,
    };
    let operations = disk_log::finish().operations;
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    assert!(operations.is_empty(), "{operations:?}");
    assert_eq!(engine.image.len(), length);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        sections
    );
}

fn initialization(parameters: [u8; 5], filtered: bool) -> ExtensibleArrayInitialization {
    ExtensibleArrayInitialization::new(
        ExtensibleArrayCreationParameters::parse(&parameters).unwrap(),
        16,
        FormatWidths::from_sizes(8, 8).unwrap(),
        filtered,
    )
    .unwrap()
}

fn engine(
    page: u64,
    placement: Placement,
    initialization: &ExtensibleArrayInitialization,
) -> (tempfile::TempDir, WriteEngine, Option<Extent>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("initialization.h5");
    let size = FileSpacePageSize::try_from(page).unwrap();
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1)
            .with_page_size(size),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.sync_policy = SyncPolicy::OnClose;
    let start = engine.image.len();
    assert_eq!(start % page, 0);
    let (sections, raw) = match placement {
        Placement::Append => {
            engine
                .append(&vec![17; usize::try_from(page - 80).unwrap()])
                .unwrap();
            (Vec::new(), None)
        }
        Placement::Reusable => {
            let mut lengths = initialization.plan().block_lengths();
            let header = lengths.next().unwrap();
            let index = lengths.next().unwrap();
            let index_capacity = if index < page {
                index
            } else {
                index.div_ceil(page) * page
            };
            let capacity = 3 * page + index_capacity;
            engine
                .append(&vec![17; usize::try_from(capacity).unwrap()])
                .unwrap();
            let raw = Extent::new(StoredAddress::new(start + page + 16), page - 32).unwrap();
            (
                vec![
                    (
                        Some(ManagerKind::PagedSmallMetadata),
                        Extent::new(StoredAddress::new(start + 16), header).unwrap(),
                    ),
                    (Some(ManagerKind::PagedSmallRaw), raw),
                    (
                        Some(ManagerKind::PagedSmallMetadata),
                        Extent::new(StoredAddress::new(start + 3 * page), index_capacity).unwrap(),
                    ),
                ],
                Some(raw),
            )
        }
    };
    engine.space =
        SessionSpace::from_persisted(Some(FileSpaceStrategy::Page), 1, size, sections).unwrap();
    engine
        .space
        .record_transition(PageType::Meta, PageTransition::Unchanged);
    (directory, engine, raw)
}

#[derive(Clone, Copy)]
enum Placement {
    Append,
    Reusable,
}
