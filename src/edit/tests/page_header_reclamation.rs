use std::fs;

use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util_hdf5::userblock;

use crate::FileSpacePageSize;
#[cfg(test)]
use crate::FileSpaceSettings;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::edit::WriteEngine;
use crate::error::FormatError;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;
use crate::writer::FileBuilder;
use crate::{AttrValue, FileLocking, FileSpaceStrategy};

#[test]
fn deleting_a_generic_large_continuation_keeps_its_partial_tail_out_of_small_free_lists() {
    let path = test_util::temp::temp_path("generic_large.h5");
    std::fs::write(&path, test_util::generic_large::generic_large()).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_eq!(
        engine.oh_chunk_spans(512).unwrap(),
        [(512, 268), (1_536, 4_158)]
    );
    assert_eq!(
        engine
            .object_header_free_spans(&[(512, 268), (1_536, 4_158)])
            .unwrap(),
        [
            (512, 268, FreeClass::Page(PageType::Meta)),
            (1_536, 4_158, FreeClass::Dead)
        ],
    );
    engine.delete(&ObjectPathBuf::parse("/d")).unwrap();
    engine.commit().unwrap();
    assert_large_header_tail(&engine);
    drop(engine);
    let engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_large_header_tail(&engine);
}

#[rstest]
#[case::stored_aligned(4_096, 4_096, FreeClass::Dead)]
#[case::physical_aligned(3_584, 4_096, FreeClass::Page(PageType::Meta))]
#[case::small(4_096, 4_095, FreeClass::Page(PageType::Meta))]
fn header_classification_uses_stored_alignment_and_preserves_physical_extents(
    #[case] stored: u64,
    #[case] length: u64,
    #[case] expected: FreeClass,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.create_dataset("keep").with_i32_data(&[7; 64]);
    builder.write(&path).unwrap();
    userblock::prepend(&path, &[0xA5; 512]);

    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_eq!(engine.context.addresses().effective_base().get(), 512);
    let physical = stored + 512;
    let spans = engine
        .object_header_free_spans(&[(physical, length)])
        .unwrap();
    assert_eq!(spans, [(physical, length, expected)]);
    assert_eq!(engine.stored_spans(spans).unwrap()[0].0.get(), stored);
    let error = engine.object_header_free_spans(&[(511, 10)]).unwrap_err();
    assert!(
        matches!(
            error,
            FormatError::AddressBelowBase {
                address: 511,
                base: 512
            }
        ),
        "{error:?}"
    );
    engine
        .set_group_attr(&ObjectPathBuf::parse("/"), "changed", AttrValue::I32(9))
        .unwrap();
    engine.commit().unwrap();
    drop(engine);
    assert_eq!(&fs::read(&path).unwrap()[..512], &[0xA5; 512]);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [7; 64]);
    assert_eq!(file.root().attrs().unwrap()["changed"], AttrValue::I32(9));
}

#[rstest]
fn deleting_a_small_header_preserves_metadata_reuse(#[values(512, 4_096, 8_192)] page: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small_header.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
    );
    builder.create_dataset("victim").with_i32_data(&[3; 64]);
    builder.create_dataset("keep").with_i32_data(&[7; 64]);
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let address = file.dataset("victim").unwrap().header_address().unwrap();
    drop(file);
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let spans = engine.oh_chunk_spans(address).unwrap();
    assert_eq!(spans.len(), 1);
    assert!(spans[0].1 < page);
    engine.delete(&ObjectPathBuf::parse("/victim")).unwrap();
    engine.commit().unwrap();
    let metadata: Vec<_> = engine
        .space
        .paged_sections()
        .unwrap()
        .metadata()
        .iter()
        .map(|extent| {
            (
                engine.addresses().resolve(extent.start()).unwrap().get(),
                extent.len(),
            )
        })
        .collect();
    assert_eq!(
        test_util::range::covered_len(&metadata, &(address..address + spans[0].1)),
        spans[0].1
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [7; 64]);
}

fn assert_large_header_tail(engine: &WriteEngine) {
    let owned = engine.space_ownership().unwrap();
    assert!(owned.analysis_complete);
    assert_eq!(
        test_util::range::covered_len(&owned.unclassified_space, &(5_632..5_694)),
        62
    );
    assert_eq!(
        test_util::range::covered_len(&owned.unclassified_space, &(5_694..6_144)),
        450
    );

    let sections = engine.space.paged_sections().unwrap();
    assert!(
        sections
            .unclassified()
            .contains(&Extent::new(StoredAddress::new(5_694), 450).unwrap())
    );
    assert!(
        sections
            .metadata()
            .iter()
            .chain(sections.raw())
            .all(|extent| {
                extent.end() <= StoredAddress::new(5_632)
                    || extent.start() >= StoredAddress::new(5_694)
            })
    );
    assert!(
        sections
            .raw()
            .iter()
            .any(|extent| extent.start() <= StoredAddress::new(1_536)
                && extent.end() >= StoredAddress::new(5_632))
    );
}
