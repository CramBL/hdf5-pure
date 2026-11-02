use std::io::ErrorKind;

use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{DenseAttrCreationOrder, ObjectHeaderPrefix};
use hdf5_pure_space::__private::{AllocationClass, Extent, SessionSpace};
use rstest::rstest;
use test_util::disk_log;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::attribute_info::AttributeInfoMessage;
use crate::edit::WriteEngine;
use crate::edit::attributes::DenseAttrSet;
use crate::edit::object_header::OhRegion;
use crate::edit::test_util as edit_test_util;
use crate::fractal_heap_storage::{self, FractalHeapAllocation};
use crate::image::WriteBuffering;
use crate::source::{PhysicalSource, Source};
use crate::type_builders;
use crate::{
    AttrValue, Error, FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy,
};

#[rstest]
#[case::managed(&[48])]
#[case::huge(&[70_000])]
#[case::mixed(&[48, 70_000, 48])]
fn dense_attribute_blocks_reuse_only_their_page_class(#[case] lengths: &[usize]) {
    let (_directory, mut engine, start) = engine();
    let attrs = lengths
        .iter()
        .enumerate()
        .map(|(index, &length)| {
            type_builders::build_attr_message(
                &format!("a{index}"),
                &AttrValue::U8Array(vec![u8::try_from(index).unwrap(); length]),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = format::dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked).unwrap();
    let expected_bytes = plan.block_lengths().sum::<u64>();
    let before = engine.space.reusable_bytes().unwrap();
    let end = engine.image.len();
    let bytes = engine.place_dense_attrs(plan).unwrap();
    assert_eq!(engine.image.len(), end);
    assert_eq!(
        engine.space.reusable_bytes().unwrap(),
        before - expected_bytes
    );
    let info =
        AttributeInfoMessage::parse_with_offset_width(&bytes, crate::file_writer::OFFSET_WIDTH)
            .unwrap();
    let opened = engine.opened_source();
    let source = PhysicalSource::new(&opened).stored();
    let storage = fractal_heap_storage::collect_fractal_heap_storage_from_source(
        &source,
        info.fractal_heap_address.unwrap(),
        engine.widths(),
    )
    .unwrap();
    let indexes = crate::attribute::collect_dense_attribute_index_storage_extents(
        &source,
        &info,
        engine.widths(),
    )
    .unwrap();
    for &(extent, kind) in storage.allocations() {
        let origin = if kind == FractalHeapAllocation::HugeObject {
            start + 1_024
        } else {
            start + PAGE + 1_024
        };
        assert!(extent.start().get() >= origin, "{extent:?} {kind:?}");
        assert!(
            extent.end().get()
                <= origin
                    + if kind == FractalHeapAllocation::HugeObject {
                        131_072
                    } else {
                        8_192
                    },
            "{extent:?} {kind:?}"
        );
    }
    assert_eq!(
        storage
            .allocations()
            .iter()
            .filter(|(_, kind)| *kind == FractalHeapAllocation::HugeObject)
            .count(),
        lengths.iter().filter(|&&length| length == 70_000).count()
    );
    for extent in indexes {
        assert!(extent.start().get() >= start + PAGE + 1_024);
        assert!(extent.end().get() <= start + PAGE + 9_216);
    }
    assert_eq!(
        engine.image().read_exact_at(start, 1_024).unwrap(),
        vec![13; 1_024]
    );
    assert_eq!(
        engine.image().read_exact_at(start + PAGE, 1_024).unwrap(),
        vec![13; 1_024]
    );
}

#[test]
fn empty_dense_attribute_sets_do_not_reserve_or_emit() {
    let (_directory, mut engine, _) = engine();
    let end = engine.image.len();
    let free = engine.space.reusable_sections().collect::<Vec<_>>();
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    engine
        .append_dense_attrs(&mut region, DenseAttrSet::default())
        .unwrap();
    assert!(region.is_empty());
    assert_eq!(engine.image.len(), end);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), free);
}

#[test]
fn failed_dense_block_emission_keeps_all_reservations_unavailable() {
    let (_directory, mut engine, _) = engine();
    let attrs = [
        type_builders::build_attr_message("huge", &AttrValue::U8Array(vec![17; 70_000])).unwrap(),
    ];
    let plan = format::dense_attrs_plan(&attrs, DenseAttrCreationOrder::Untracked).unwrap();
    let size = plan.block_lengths().sum::<u64>();
    let snapshot = engine.snapshot_free();
    let before = engine.space.reusable_bytes().unwrap();
    let length = engine.image.len();
    disk_log::fail_write_after(1, 7);
    let error = engine.place_dense_attrs(plan).unwrap_err();
    disk_log::clear_failures();
    let Error::Io(error) = error else {
        panic!("expected I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(engine.image.len(), length);
    assert_eq!(engine.space.reusable_bytes().unwrap(), before - size);
    engine.restore_free(snapshot);
    assert_eq!(engine.space.reusable_bytes().unwrap(), before);
}

fn engine() -> (tempfile::TempDir, WriteEngine, u64) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense.h5");
    let page = FileSpacePageSize::try_from(PAGE).unwrap();
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(page),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let start = engine.image.len();
    assert_eq!(start % PAGE, 0);
    engine
        .append(&vec![13; usize::try_from(3 * PAGE).unwrap()])
        .unwrap();
    engine.space = SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, page);
    engine.space.return_allocation(
        Extent::new(StoredAddress::new(start + 1_024), 131_072).unwrap(),
        AllocationClass::Raw,
    );
    engine.space.return_allocation(
        Extent::new(StoredAddress::new(start + PAGE + 1_024), 8_192).unwrap(),
        AllocationClass::Metadata,
    );
    (directory, engine, start)
}

const PAGE: u64 = 262_144;
