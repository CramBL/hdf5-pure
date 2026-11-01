//! Native reuse of compact PAGE free runs after metadata edits.

#![cfg(feature = "__hdf5-1.10")]

use core::slice;
use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::PersistedFreeSections;
use hdf5_pure::SyncPolicy;
use rstest::rstest;
use test_util::snapshot::text::{Report, Value};
use test_util_hdf5::file_space_oracle;

use crate::test_util::file_space;

#[rstest]
#[case::bounded(MemoryStrategy::Bounded)]
#[case::mirrored(MemoryStrategy::Mirrored)]
fn a_metadata_edit_preserves_native_multi_page_hole_reuse(#[case] memory: MemoryStrategy) {
    let directory = tempfile::tempdir().unwrap();
    let template = directory.path().join("template.h5");
    let control = directory.path().join("control.h5");
    let edited = directory.path().join("edited.h5");
    let hole = native_fixture(&template);
    fs::copy(&template, &control).unwrap();
    fs::copy(&template, &edited).unwrap();
    assert_eq!(fs::read(&control).unwrap(), fs::read(&edited).unwrap());
    let before = persisted(&template);
    assert_eq!(large_runs(&before), slice::from_ref(&hole));
    let initial_len = fs::metadata(&template).unwrap().len();

    let file = File::open_rw_with_options(
        &edited,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    file.root()
        .set_attr("audit_counter", AttrValue::I32(1))
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    assert_eq!(fs::read(&template).unwrap(), fs::read(&control).unwrap());
    let after = persisted(&edited);
    assert_eq!(large_runs(&after), slice::from_ref(&hole));
    assert!(after.free_sections().section_count() <= before.free_sections().section_count() + 2);
    assert!(after.manager_metadata_bytes() < PAGE);
    assert_eq!(fs::metadata(&edited).unwrap().len(), initial_len);
    let control_allocation = allocate_native(&control, 0);
    let edited_allocation = allocate_native(&edited, 1);
    assert_eq!(control_allocation.address, hole.start);
    assert_eq!(edited_allocation.address, hole.start);
    assert_eq!(control_allocation.final_len, control_allocation.initial_len);
    assert_eq!(edited_allocation.final_len, edited_allocation.initial_len);
    assert!(edited_allocation.address + REQUEST as u64 <= hole.end);
    assert_pure_values(&control, 0);
    assert_pure_values(&edited, 1);
    let control_final = persisted(&control);
    let edited_final = persisted(&edited);
    let remaining = hole.start + REQUEST as u64..hole.end;
    assert_eq!(large_runs(&control_final), slice::from_ref(&remaining));
    assert_eq!(large_runs(&edited_final), [remaining]);

    let mut report = Report::new();
    report.table(
        "persisted managers",
        [
            "phase",
            "records",
            "metadata bytes",
            "free bytes",
            "large run bytes",
        ],
        [
            manager_row("native fixture", &before),
            manager_row("pure metadata edit", &after),
            manager_row("native control allocation", &control_final),
            manager_row("native edited allocation", &edited_final),
        ],
    );
    report.table(
        "native allocation",
        [
            "file",
            "initial bytes",
            "data address",
            "final bytes",
            "growth bytes",
        ],
        [
            allocation_row("control", &control_allocation),
            allocation_row("edited", &edited_allocation),
        ],
    );
    file_space::snapshot("compact-page-run-reuse", &report.finish());
}

fn native_fixture(path: &Path) -> Range<u64> {
    let file = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_latest())
        .with_fcpl(|properties| {
            properties
                .file_space_strategy(NativeStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(PAGE)
        })
        .create(path)
        .unwrap();
    let dataset = file.new_dataset::<u8>().shape(HOLE).create("hole").unwrap();
    dataset.write_raw(&vec![0; HOLE]).unwrap();
    let start = dataset.offset().unwrap();
    assert_eq!(start % PAGE, 0);
    drop(dataset);
    let survivor = file
        .new_dataset_builder()
        .with_data(&keep_values())
        .create("keep")
        .unwrap();
    assert!(survivor.offset().unwrap() >= start + HOLE as u64);
    drop(survivor);
    file.new_attr::<i32>()
        .shape(())
        .create("audit_counter")
        .unwrap()
        .write_scalar(&0)
        .unwrap();
    file.unlink("hole").unwrap();
    file.close().unwrap();
    start..start + HOLE as u64
}

fn persisted(path: &Path) -> PersistedFreeSections {
    file_space_oracle::assert_persisted_sections(path);
    let metadata = file_space_oracle::assert_manager_metadata_bytes(path);
    let file = File::open_streaming(path).unwrap();
    let report = file.persisted_free_sections().unwrap().unwrap();
    assert_eq!(report.manager_metadata_bytes(), metadata);
    file.close().unwrap();
    report
}

fn large_runs(report: &PersistedFreeSections) -> Vec<Range<u64>> {
    report
        .managers()
        .iter()
        .filter(|manager| manager.role() == FreeSpaceManagerRole::PageLargeGeneric)
        .flat_map(|manager| manager.free_sections().sections())
        .map(|section| {
            assert_eq!(section.address().get() % PAGE, 0);
            assert_eq!(section.length().get() % PAGE, 0);
            section.address().get()..section.end_address().get()
        })
        .collect()
}

struct NativeAllocation {
    initial_len: u64,
    address: u64,
    final_len: u64,
}

fn allocate_native(path: &Path, counter: i32) -> NativeAllocation {
    let initial_len = fs::metadata(path).unwrap().len();
    let file = hdf5::File::open_rw(path).unwrap();
    assert!(file.free_space() >= REQUEST as u64);
    let dataset = file
        .new_dataset::<u8>()
        .shape(REQUEST)
        .create("native_large")
        .unwrap();
    dataset.write_raw(&vec![7; REQUEST]).unwrap();
    let address = dataset.offset().unwrap();
    assert_eq!(dataset.read_raw::<u8>().unwrap(), vec![7; REQUEST]);
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        keep_values()
    );
    assert_eq!(
        file.attr("audit_counter")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        counter
    );
    drop(dataset);
    file.close().unwrap();
    NativeAllocation {
        initial_len,
        address,
        final_len: fs::metadata(path).unwrap().len(),
    }
}

fn assert_pure_values(path: &Path, counter: i32) {
    let file = File::open_streaming(path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        keep_values()
    );
    assert_eq!(
        file.dataset("native_large").unwrap().read_u8().unwrap(),
        vec![7; REQUEST]
    );
    assert_eq!(
        file.root().attrs().unwrap().get("audit_counter"),
        Some(&AttrValue::I32(counter))
    );
    file.close().unwrap();
}

fn manager_row(label: &str, report: &PersistedFreeSections) -> [Value; 5] {
    [
        Value::text(label),
        Value::Number(report.free_sections().section_count() as u64),
        Value::Number(report.manager_metadata_bytes()),
        Value::Number(report.free_sections().free_bytes()),
        Value::Number(
            large_runs(report)
                .iter()
                .map(|run| run.end - run.start)
                .sum(),
        ),
    ]
}

fn allocation_row(label: &str, allocation: &NativeAllocation) -> [Value; 5] {
    [
        Value::text(label),
        Value::Number(allocation.initial_len),
        Value::Hexadecimal {
            value: allocation.address,
            digits: 8,
        },
        Value::Number(allocation.final_len),
        Value::Number(allocation.final_len - allocation.initial_len),
    ]
}

fn keep_values() -> Vec<i32> {
    (0..2_048).collect()
}

const PAGE: u64 = 4_096;
const HOLE: usize = 1 << 20;
const REQUEST: usize = 64 << 10;
