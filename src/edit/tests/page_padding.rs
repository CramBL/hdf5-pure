use std::io::ErrorKind;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_space::__private::{Extent, PageTransition, PageType, SessionSpace};
use rstest::rstest;
use test_util::disk_log;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::edit::WriteEngine;
use crate::edit::observations::{self, ZeroWork};
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::source::Source;
use crate::{FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy, UserblockSize};

#[derive(Clone, Copy, Debug)]
enum Padding {
    Finish,
    Switch,
}

#[rstest]
#[case::finish(Padding::Finish)]
#[case::switch(Padding::Switch)]
fn page_padding_preserves_addresses_and_records_only_complete_gaps(
    #[case] padding: Padding,
    #[values(0, 262_144)] base: u64,
) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::Page, base);
    engine.begin_page(PageType::Meta).unwrap();
    let start = engine.append(&[17; 19]).unwrap();
    let gap = PAGE - 19;
    observations::reset_zeros();
    pad(&mut engine, padding);
    assert_eq!(engine.image.len(), start + PAGE);
    assert_eq!(engine.image().read_exact_at(start, 19).unwrap(), [17; 19]);
    assert_eq!(
        engine
            .image()
            .read_exact_at(start + 19, gap as usize)
            .unwrap(),
        vec![0; gap as usize]
    );
    assert_eq!(
        engine.space.pending_padding(PageType::Meta).unwrap(),
        [Extent::new(StoredAddress::new(start + 19 - base), ByteLength::new(gap)).unwrap()]
    );
    assert_eq!(engine.space.pending_padding(PageType::Raw).unwrap(), []);
    assert_eq!(
        observations::zero_work(),
        ZeroWork {
            calls: 1,
            bytes: gap,
            scratch_peak: 65_536
        }
    );
    assert_eq!(
        engine.space.paged_tail_type(),
        Some(match padding {
            Padding::Finish => Some(PageType::Meta),
            Padding::Switch => Some(PageType::Raw),
        })
    );
}

#[rstest]
#[case::flat_finish(FileSpaceStrategy::None, Padding::Finish, 19)]
#[case::flat_switch(FileSpaceStrategy::None, Padding::Switch, 19)]
#[case::aligned_finish(FileSpaceStrategy::Page, Padding::Finish, 0)]
#[case::aligned_switch(FileSpaceStrategy::Page, Padding::Switch, 0)]
fn unnecessary_page_padding_does_not_emit_zeros(
    #[case] strategy: FileSpaceStrategy,
    #[case] padding: Padding,
    #[case] extra: usize,
) {
    let (_directory, mut engine) = engine(strategy, 0);
    engine.begin_page(PageType::Meta).unwrap();
    engine.append(&vec![17; extra]).unwrap();
    let length = engine.image.len();
    observations::reset_zeros();
    pad(&mut engine, padding);
    assert_eq!(engine.image.len(), length);
    assert_eq!(observations::zero_work(), ZeroWork::default());
}

#[rstest]
#[case::finish(Padding::Finish)]
#[case::switch(Padding::Switch)]
fn partial_page_padding_retains_the_tail_class_and_successful_prefix(#[case] padding: Padding) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::Page, PAGE);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.begin_page(PageType::Meta).unwrap();
    engine.append(&[17; 19]).unwrap();
    let start = engine.image.len();
    disk_log::fail_write_after(1, 7);
    let error = match padding {
        Padding::Finish => engine.pad_to_page(),
        Padding::Switch => engine.begin_page(PageType::Raw),
    }
    .unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(engine.image.len(), start + 65_536);
    assert_eq!(
        engine.image().read_exact_at(start, 65_536).unwrap(),
        vec![0; 65_536]
    );
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Meta)));
    assert_eq!(engine.space.pending_padding(PageType::Meta).unwrap(), []);
    assert_eq!(engine.space.pending_padding(PageType::Raw).unwrap(), []);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
}

fn pad(engine: &mut WriteEngine, padding: Padding) {
    match padding {
        Padding::Finish => engine.pad_to_page(),
        Padding::Switch => engine.begin_page(PageType::Raw),
    }
    .unwrap();
}

fn engine(strategy: FileSpaceStrategy, base: u64) -> (tempfile::TempDir, WriteEngine) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("padding.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(strategy == FileSpaceStrategy::Page)
            .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    if strategy == FileSpaceStrategy::Page {
        engine.space = SessionSpace::from_strategy(
            Some(strategy),
            ByteLength::new(1),
            FileSpacePageSize::try_from(PAGE).unwrap(),
        );
        engine
            .space
            .record_transition(PageType::Meta, PageTransition::Unchanged);
    }
    (directory, engine)
}

const PAGE: u64 = 262_144;
