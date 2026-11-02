use std::path::Path;

use hdf5_pure_space::__private::Extent;
use rstest::rstest;
use test_util_hdf5::creation_order::Indexing;
use test_util_hdf5::page_dense::{self, Contents};

use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::edit::attributes;
use crate::edit::group_links;
use crate::edit::test_util;
use crate::fractal_heap_storage;
use crate::fractal_heap_storage::FractalHeapAllocation;
use crate::object_path::ObjectPath;
use crate::source::PhysicalSource;
use crate::{AttrValue, File};

#[rstest]
#[case::managed(Contents::Managed)]
#[case::huge(Contents::Huge)]
#[case::mixed(Contents::Mixed)]
#[case::empty(Contents::Empty)]
#[case::revert(Contents::Revert)]
#[case::promote(Contents::Promote)]
fn dense_editor_blocks_preserve_page_geometry_and_reopen(
    #[case] contents: Contents,
    #[values(512, 4_096)] page: u64,
    #[values(0, 8_192)] base: u64,
    #[values(Indexing::No, Indexing::Yes)] indexing: Indexing,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense.h5");
    page_dense::write_native(&path, page, base, contents, indexing);
    let initial_eof = std::fs::metadata(&path).unwrap().len();
    let file = File::open_rw(&path).unwrap();
    let group = file.group("g").unwrap();
    for index in 0..contents.removed() {
        group.delete(&format!("c{index:02}")).unwrap();
        group.remove_attr(&format!("a{index:02}")).unwrap();
    }
    if contents.end() > contents.initial() {
        group
            .create_group(&format!("c{:02}", contents.initial()))
            .unwrap();
        group
            .set_attr("a00", AttrValue::U8Array(vec![99; contents.length(0)]))
            .unwrap();
        group
            .set_attr(
                &format!("a{:02}", contents.initial()),
                AttrValue::U8Array(vec![
                    u8::try_from(contents.initial()).unwrap();
                    contents.length(contents.initial())
                ]),
            )
            .unwrap();
    }
    file.commit().unwrap();
    drop(group);
    file.close().unwrap();
    let first = allocations(&path, page, contents, indexing, initial_eof);
    page_dense::assert_edited(&path, page, contents, indexing);
    let file = File::open_rw(&path).unwrap();
    file.root()
        .set_attr("checkpoint", AttrValue::I64(17))
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    assert_eq!(
        allocations(&path, page, contents, indexing, initial_eof),
        first
    );
    page_dense::assert_edited(&path, page, contents, indexing);
    directory.close().unwrap();
}

fn allocations(
    path: &Path,
    page: u64,
    contents: Contents,
    indexing: Indexing,
    initial_eof: u64,
) -> Vec<Extent> {
    let engine = test_util::open_engine(path);
    let address = crate::group::v2::resolve_path_any_from_source(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        engine.superblock(),
        engine.widths(),
        &ObjectPath::parse("g"),
    )
    .unwrap();
    let region =
        WriteEngine::gather_oh_messages(engine.image(), address, engine.addresses()).unwrap();
    let mut allocations = Vec::new();
    let opened = engine.opened_source();
    let source = PhysicalSource::new(&opened).stored();
    let attribute_info = attributes::find_attribute_info(&region).unwrap();
    assert_eq!(
        attribute_info.is_none(),
        matches!(contents, Contents::Empty)
    );
    if let Some((_, _, info)) = attribute_info {
        let dense = !matches!(contents, Contents::Empty | Contents::Revert);
        assert_eq!(info.fractal_heap_address.is_some(), dense);
        assert_eq!(info.btree_name_index_address.is_some(), dense);
        assert_eq!(
            info.btree_creation_order_address.is_some(),
            dense && matches!(indexing, Indexing::Yes)
        );
        assert_eq!(
            info.max_creation_index,
            Some(u16::try_from(contents.end()).unwrap())
        );
        assert_eq!(
            info.indexes_creation_order,
            matches!(indexing, Indexing::Yes)
        );
        if let Some(heap) = info.fractal_heap_address {
            allocations.extend(
                crate::attribute::collect_dense_attribute_index_storage_extents(
                    &source,
                    &info,
                    engine.widths(),
                )
                .unwrap(),
            );
            let heap = fractal_heap_storage::collect_fractal_heap_storage_from_source(
                &source,
                heap,
                engine.widths(),
            )
            .unwrap();
            assert_eq!(
                heap.allocations()
                    .iter()
                    .filter(|(_, kind)| *kind == FractalHeapAllocation::HugeObject)
                    .count(),
                match contents {
                    Contents::Huge => 10,
                    Contents::Mixed => 5,
                    _ => 0,
                }
            );
            allocations.extend(heap.allocations().iter().map(|&(extent, _)| extent));
        }
    }
    let links = group_links::inspect(&opened, engine.widths(), &region).unwrap();
    assert_eq!(
        links.creation_order().next_index(),
        Some(u64::try_from(contents.end()).unwrap())
    );
    assert_eq!(
        links.creation_order().is_indexed(),
        matches!(indexing, Indexing::Yes)
    );
    if let Some(heap) = links.info().fractal_heap_address {
        allocations.extend(
            crate::group::v2::collect_dense_group_index_storage_extents(
                &source,
                &links.info(),
                engine.widths(),
            )
            .unwrap(),
        );
        allocations.extend(
            fractal_heap_storage::collect_fractal_heap_storage_from_source(
                &source,
                heap,
                engine.widths(),
            )
            .unwrap()
            .allocations()
            .iter()
            .map(|&(extent, _)| extent),
        );
    }
    allocations.sort_unstable();
    assert!(
        allocations
            .windows(2)
            .all(|pair| pair[0].end() <= pair[1].start())
    );
    for extent in &allocations {
        let offset = extent.start().get() % page;
        assert!(
            if extent.len() < page {
                offset + extent.len() <= page
            } else {
                offset == 0
            },
            "{extent:?}"
        );
        assert!(
            engine
                .space
                .reusable_sections()
                .all(|free| !free.range().overlaps(extent.range()))
        );
    }
    let ownership = engine.space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    assert!(
        ownership
            .unclassified_space
            .iter()
            .all(|&(start, length)| start + length <= initial_eof),
        "{:?}",
        ownership.unclassified_space
    );
    allocations
}
