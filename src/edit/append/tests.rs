use std::io::ErrorKind;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{Extent, PageTransition, PageType, SessionSpace};
use rstest::rstest;
use test_util::disk_log;
use test_util_hdf5::dataset::FiniteChunkIndex;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::edit::WriteEngine;
use crate::edit::observations::{self, ZeroWork};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::source::Source;
use crate::{
    File, FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy, MaxExtent,
    UserblockSize,
};

#[rstest]
#[case::small(4)]
#[case::large(64)]
fn finite_chunk_ownership_analysis_is_reused_until_a_commit(
    #[case] groups: usize,
    #[values(FiniteChunkIndex::Fixed, FiniteChunkIndex::Single)] index: FiniteChunkIndex,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("owned.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(index.maximum() * 4)])
        .with_chunks(&[16]);
    for index in 0..groups {
        let group = builder.create_group(&format!("g{index}")).finish();
        builder.add_group(group);
    }
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    observations::reset_chunk_ownership_analyses();
    engine.append_inplace_i32_phased("d", &[1], 4).unwrap();
    assert_eq!(observations::chunk_ownership_analyses(), 1);
    assert_eq!(engine.chunk_ownership.get().unwrap().len(), groups + 2);
    engine.located.clear();
    engine.append_inplace_i32_phased("d", &[2], 4).unwrap();
    assert_eq!(observations::chunk_ownership_analyses(), 1);
    engine.chunk_ownership.take();
    engine.located.clear();
    engine.append_inplace_i32_phased("d", &[3], 4).unwrap();
    assert_eq!(observations::chunk_ownership_analyses(), 2);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    engine.commit().unwrap();
    assert!(engine.chunk_ownership.get().is_none());
    assert!(engine.located.is_empty());
    engine.append_inplace_i32_phased("d", &[4], 4).unwrap();
    assert_eq!(observations::chunk_ownership_analyses(), 3);
    assert_eq!(engine.chunk_ownership.get().unwrap().len(), groups + 3);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        [0, 1, 2, 3, 4]
    );
    file.close().unwrap();
}

mod growth;
mod initialization;

#[rstest]
fn raw_fallback_prepares_the_chosen_address_before_bounded_padding(
    #[values(0, 262_144)] base: u64,
) {
    let (_directory, mut engine) = engine(base);
    let start = engine.image.len();
    let address = PhysicalAddress::new(start + PAGE - 19);
    observations::reset_zeros();
    assert_eq!(
        engine
            .store()
            .append_into_raw_page(0, |at| {
                assert_eq!(at, address);
                assert_eq!(observations::zero_work(), ZeroWork::default());
                Ok(Vec::<u8>::new())
            })
            .unwrap(),
        address.get()
    );
    assert_eq!(engine.image.len(), address.get());
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Raw)));
    assert_eq!(
        engine.space.pending_padding(PageType::Meta).unwrap(),
        [Extent::new(StoredAddress::new(start - base), ByteLength::new(PAGE - 19)).unwrap()]
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(start, (PAGE - 19) as usize)
            .unwrap(),
        vec![0; (PAGE - 19) as usize]
    );
    assert_eq!(
        observations::zero_work(),
        ZeroWork {
            calls: 1,
            bytes: PAGE - 19,
            scratch_peak: 65_536
        }
    );
}

#[rstest]
#[case::same_type(true)]
#[case::aligned(false)]
fn raw_fallback_without_a_gap_emits_no_zeros(#[case] same_type: bool) {
    let (_directory, mut engine) = engine(0);
    if same_type {
        engine.begin_page(PageType::Raw).unwrap();
    } else {
        engine.pad_to_page().unwrap();
    }
    let start = engine.image.len();
    observations::reset_zeros();
    assert_eq!(
        engine
            .store()
            .append_into_raw_page(0, |at| {
                assert_eq!(at.get(), start);
                Ok(Vec::<u8>::new())
            })
            .unwrap(),
        start
    );
    assert_eq!(engine.image.len(), start);
    assert_eq!(observations::zero_work(), ZeroWork::default());
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Raw)));
}

#[rstest]
#[case::preparation(false)]
#[case::length(true)]
fn raw_fallback_preparation_errors_precede_padding(#[case] length: bool) {
    let (_directory, mut engine) = engine(PAGE);
    let start = engine.image.len();
    observations::reset_zeros();
    let error = engine
        .store()
        .append_into_raw_page(0, |at| {
            assert_eq!(at.get(), start + PAGE - 19);
            if length {
                Ok(vec![7])
            } else {
                Err(FormatError::Internal("injected preparation error".into()).into())
            }
        })
        .unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == if length { "prepared allocation length differs from its reservation" } else { "injected preparation error" }),
        "{error:?}"
    );
    assert_eq!(engine.image.len(), start);
    assert_eq!(engine.space.paged_tail_type(), Some(Some(PageType::Meta)));
    assert_eq!(engine.space.pending_padding(PageType::Meta).unwrap(), []);
    assert_eq!(observations::zero_work(), ZeroWork::default());
}

#[test]
fn raw_fallback_partial_padding_keeps_the_outgoing_page_class() {
    let (_directory, mut engine) = engine(PAGE);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let start = engine.image.len();
    disk_log::fail_write_after(1, 7);
    let error = engine
        .store()
        .append_into_raw_page(0, |at| {
            assert_eq!(at.get(), start + PAGE - 19);
            Ok(Vec::<u8>::new())
        })
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

fn engine(base: u64) -> (tempfile::TempDir, WriteEngine) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fallback.h5");
    let page = FileSpacePageSize::try_from(PAGE).unwrap();
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(page),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.space = SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, page);
    engine
        .space
        .record_transition(PageType::Meta, PageTransition::Unchanged);
    engine.append(&[17; 19]).unwrap();
    (directory, engine)
}

const PAGE: u64 = 262_144;
