use std::io::ErrorKind;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_space::__private::AllocationClass;
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
use crate::chunk_index_inplace::Store;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::source::Source;

#[rstest]
#[case::small(PageType::Meta, Block::Small)]
#[case::crossing(PageType::Meta, Block::Crossing)]
#[case::large(PageType::Meta, Block::Large)]
#[case::raw_tail(PageType::Raw, Block::Small)]
fn metadata_allocations_keep_complete_blocks_in_metadata_pages(
    #[case] tail: PageType,
    #[case] block: Block,
    #[values(512, 4_096)] page: u64,
) {
    let (_directory, mut engine) = engine(page, tail);
    let start = engine.image.len();
    let bytes = vec![35; block.length(page)];
    let allocation = engine
        .store()
        .allocate(&bytes, AllocationClass::Metadata)
        .unwrap();
    let range = allocation.range();
    assert_eq!(range.length(), u64::try_from(bytes.len()).unwrap());
    let expected = if tail == PageType::Meta && bytes.len() <= 80 {
        start
    } else {
        start.next_multiple_of(page)
    };
    assert_eq!(range.start().get(), expected);
    assert_eq!(range.end().get(), expected + range.length());
    assert_eq!(engine.image.len(), range.end().get());
    assert_eq!(
        engine.image.read_exact_at(expected, bytes.len()).unwrap(),
        bytes
    );
    if range.length() < page {
        assert_eq!(expected / page, (range.end().get() - 1) / page);
    } else {
        assert_eq!(expected % page, 0);
    }
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Meta)));
}

#[rstest]
#[case::metadata(AllocationClass::Metadata, 2)]
#[case::raw(AllocationClass::Raw, 1)]
fn immediate_allocations_use_only_matching_partial_page_holes(
    #[case] class: AllocationClass,
    #[case] selected: u64,
) {
    let page = 512;
    let (_directory, mut engine) = engine(page, PageType::Meta);
    engine.pad_to_page().unwrap();
    let start = engine.image.len();
    engine.append(&[17; 1_536]).unwrap();
    let raw = Extent::new(StoredAddress::new(start + page + 16), ByteLength::new(64)).unwrap();
    let meta = Extent::new(
        StoredAddress::new(start + 2 * page + 16),
        ByteLength::new(64),
    )
    .unwrap();
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::MIN,
        [
            (Some(ManagerKind::PagedSmallRaw), raw),
            (Some(ManagerKind::PagedSmallMetadata), meta),
        ],
    )
    .unwrap();
    let length = engine.image.len();
    let bytes = [31; 64];
    let allocation = engine.store().allocate(&bytes, class).unwrap();
    assert_eq!(
        allocation.range().start().get(),
        start + selected * page + 16
    );
    assert_eq!(allocation.range().length(), 64);
    assert_eq!(engine.image.len(), length);
    assert_eq!(
        engine
            .image
            .read_exact_at(allocation.range().start().get(), 64)
            .unwrap(),
        bytes
    );
    let remaining = engine.space.reusable_sections().collect::<Vec<_>>();
    assert_eq!(remaining, [if selected == 1 { meta } else { raw }]);
}

#[rstest]
#[case::before_padding(0, 0)]
#[case::partial_padding(0, 3)]
#[case::partial_metadata(1, 7)]
fn failed_metadata_growth_restores_the_prior_tail_and_bytes(
    #[case] writes: usize,
    #[case] prefix: usize,
) {
    let (_directory, mut engine) = engine(512, PageType::Raw);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let length = engine.image.len();
    let before = engine
        .image
        .read_exact_at(0, usize::try_from(length).unwrap())
        .unwrap();
    disk_log::fail_write_after(writes, prefix);
    let error = match engine
        .store()
        .allocate(&[31; 64], AllocationClass::Metadata)
    {
        Ok(_) => panic!("expected metadata write error"),
        Err(error) => error,
    };
    let Error::Io(error) = error else {
        panic!("expected I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(engine.image.len(), length);
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Raw)));
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    assert_eq!(
        engine
            .image
            .read_exact_at(0, usize::try_from(length).unwrap())
            .unwrap(),
        before
    );
    engine.context.allocations.before_reference().unwrap();
}

#[test]
fn uncertain_publication_rejects_metadata_growth_before_io() {
    let (_directory, mut engine) = engine(512, PageType::Raw);
    let length = engine.image.len();
    engine.make_allocation_uncertain();
    disk_log::start();
    let error = match engine
        .store()
        .allocate(&[31; 64], AllocationClass::Metadata)
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
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
}

fn engine(page: u64, tail: PageType) -> (tempfile::TempDir, WriteEngine) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("growth.h5");
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
    assert_eq!(engine.image.len() % page, 0);
    engine.space = SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, size);
    engine
        .space
        .record_transition(tail, PageTransition::Unchanged);
    engine
        .append(&vec![17; usize::try_from(page - 80).unwrap()])
        .unwrap();
    (directory, engine)
}

#[derive(Clone, Copy)]
enum Block {
    Crossing,
    Large,
    Small,
}

impl Block {
    fn length(self, page: u64) -> usize {
        usize::try_from(match self {
            Self::Crossing => 128,
            Self::Large => page + 37,
            Self::Small => 64,
        })
        .unwrap()
    }
}
