//! Native observations of bytes issued through an active editing session.
//!
//! The tests read through the locked handle and write separate snapshots for inspection through
//! libhdf5. Observations precede file closure and include buffered input before its first commit.

use std::fs;

use rstest::rstest;
use test_util_hdf5::__private::ownership_append;
use test_util_hdf5::__private::unallocated_append;
use test_util_hdf5::file_space_oracle;

use crate::{Dataset, Error, File, FileAccessProperties, MemoryStrategy, SyncPolicy};

#[rstest]
#[case::flat_plain(false, false, false)]
#[case::flat_filtered(false, true, false)]
#[case::page_plain(true, false, false)]
#[case::page_filtered(true, true, false)]
#[case::page_plain_incompressible(true, false, true)]
#[case::page_filtered_incompressible(true, true, true)]
fn immediate_ownership_publication_matches_native_issued_free_sections(
    #[case] paged: bool,
    #[case] filtered: bool,
    #[case] incompressible: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-ownership.h5");
    let fixture = ownership_append::Fixture::write(&path, paged, filtered, incompressible);
    for cycle in 0..3 {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(MemoryStrategy::Bounded)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        let mut target = file.dataset("d").unwrap();
        target
            .append(&ownership_append::payload(0x3cb6_7710 + cycle))
            .unwrap();
        let snapshot = directory.path().join(format!("issued-{cycle}.h5"));
        let issued = file.disk_bytes().unwrap();
        fs::write(&snapshot, &issued).unwrap();
        fixture.assert_contents(&snapshot, cycle);
        assert_eq!(file.disk_bytes().unwrap(), issued);
        drop(target);
        file.close().unwrap();
    }
}

#[rstest]
#[case::immediate(FirstWrite::Immediate)]
#[case::borrowed_manual(FirstWrite::BorrowedManual)]
#[case::borrowed_automatic(FirstWrite::BorrowedAutomatic)]
#[case::borrowed_terminal(FirstWrite::BorrowedTerminal)]
#[case::owned_manual(FirstWrite::OwnedManual)]
#[case::owned_automatic(FirstWrite::OwnedAutomatic)]
#[case::owned_terminal(FirstWrite::OwnedTerminal)]
fn deferred_native_indexes_preserve_issued_bytes_until_first_commit(
    #[case] operation: FirstWrite,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values([32, 4, 4, 16, 10], [24, 8, 2, 8, 8])] parameters: [u8; 5],
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("initialize-issued.h5");
    let header = unallocated_append::fixture(&path, filtered, paged, parameters);
    let before = fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let count = operation.run(&file, file.dataset("d").unwrap(), &before);
    let mut target = file.dataset("d").unwrap();
    assert_eq!(target.read_i32().unwrap(), (0..count).collect::<Vec<_>>());
    target.append(&(count..40).collect::<Vec<i32>>()).unwrap();
    assert_eq!(target.read_i32().unwrap(), (0..40).collect::<Vec<_>>());
    let issued = file.disk_bytes().unwrap();
    let snapshot = directory.path().join("initialized-issued.h5");
    fs::write(&snapshot, &issued).unwrap();
    unallocated_append::inspect_initialized_index(&snapshot, header, parameters);
    file_space_oracle::assert_persisted_sections(&snapshot);
    file_space_oracle::assert_manager_metadata_bytes(&snapshot);
    assert_eq!(file.disk_bytes().unwrap(), issued);
    drop(target);
    file.close().unwrap();
}

#[derive(Clone, Copy)]
enum FirstWrite {
    BorrowedAutomatic,
    BorrowedManual,
    BorrowedTerminal,
    Immediate,
    OwnedAutomatic,
    OwnedManual,
    OwnedTerminal,
}

impl FirstWrite {
    fn run(self, file: &File, mut target: Dataset, before: &[u8]) -> i32 {
        let count = match self {
            Self::BorrowedAutomatic | Self::OwnedAutomatic => 4,
            _ => 3,
        };
        let values: Vec<i32> = (0..count).collect();
        match self {
            Self::Immediate => target.append(&values).unwrap(),
            Self::BorrowedAutomatic | Self::BorrowedManual | Self::BorrowedTerminal => {
                let mut appender = target.buffered_appender().unwrap();
                appender.append(&values).unwrap();
                if !matches!(self, Self::BorrowedAutomatic) {
                    assert_eq!(file.disk_bytes().unwrap(), before);
                }
                if matches!(self, Self::BorrowedManual) {
                    appender.flush().unwrap();
                }
                appender.finish().unwrap();
            }
            Self::OwnedAutomatic | Self::OwnedManual | Self::OwnedTerminal => {
                let mut appender = target.into_buffered_appender().unwrap();
                appender.append(&values).unwrap();
                if !matches!(self, Self::OwnedAutomatic) {
                    assert_eq!(file.disk_bytes().unwrap(), before);
                }
                if matches!(self, Self::OwnedManual) {
                    appender.flush().unwrap();
                }
                appender.finish().unwrap();
            }
        }
        count
    }
}

#[rstest]
#[case::immediate(EmptyOperation::Immediate)]
#[case::borrowed(EmptyOperation::Borrowed)]
#[case::owned(EmptyOperation::Owned)]
#[case::borrowed_pending(EmptyOperation::BorrowedPending)]
#[case::owned_pending(EmptyOperation::OwnedPending)]
#[case::incompatible_input(EmptyOperation::IncompatibleInput)]
fn unallocated_native_indexes_preserve_issued_storage_during_read_only_operations(
    #[case] operation: EmptyOperation,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values([32, 4, 4, 16, 10], [24, 8, 2, 8, 8])] parameters: [u8; 5],
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.h5");
    let header = unallocated_append::fixture(&path, filtered, paged, parameters);
    let original = fs::read(&path).unwrap();
    unallocated_append::assert_unallocated_layout(&original, header, parameters);
    file_space_oracle::assert_persisted_sections(&path);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let sections = File::from_bytes(original.clone())
        .unwrap()
        .persisted_free_sections()
        .unwrap();
    assert!(sections.as_ref().unwrap().free_sections().free_bytes() > 0);
    let reusable = file.reusable_space_stats().unwrap();
    operation.run(file.dataset("d").unwrap());
    assert_eq!(
        File::from_bytes(file.disk_bytes().unwrap())
            .unwrap()
            .persisted_free_sections()
            .unwrap(),
        sections
    );
    assert_eq!(file.reusable_space_stats().unwrap(), reusable);
    let target = file.dataset("d").unwrap();
    assert_eq!(target.shape().unwrap(), [0]);
    assert_eq!(target.chunk_shape().unwrap(), Some(vec![4]));
    assert!(target.read_i32().unwrap().is_empty());
    assert!(target.chunks().unwrap().is_empty());
    drop(target);
    let observed = file.disk_bytes().unwrap();
    assert_eq!(observed, original);
    unallocated_append::assert_unallocated_layout(&observed, header, parameters);
    file.close().unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    file_space_oracle::assert_persisted_sections(&path);
}

#[derive(Clone, Copy)]
enum EmptyOperation {
    Borrowed,
    BorrowedPending,
    Immediate,
    IncompatibleInput,
    Owned,
    OwnedPending,
}

impl EmptyOperation {
    fn run(self, mut dataset: Dataset) {
        match self {
            Self::Immediate => {
                dataset.append::<i32>(&[]).unwrap();
                dataset.append_raw(&[]).unwrap();
            }
            Self::IncompatibleInput => {
                let error = dataset.append(&[1_u32]).unwrap_err();
                let Error::AppendInPlaceUnsupported(reason) = error else {
                    panic!("expected incompatible input, got {error:?}");
                };
                assert_eq!(
                    reason,
                    "append datatype does not match the on-disk dataset (wrong element type or byte order)"
                );
                let error = dataset.append_raw(&[1]).unwrap_err();
                let Error::AppendInPlaceUnsupported(reason) = error else {
                    panic!("expected an incomplete element, got {error:?}");
                };
                assert_eq!(
                    reason,
                    "appended byte length is not a whole number of elements"
                );
            }
            Self::Borrowed => {
                let mut appender = dataset.buffered_appender().unwrap();
                appender.append::<i32>(&[]).unwrap();
                appender.append_raw(&[]).unwrap();
                appender.flush().unwrap();
                appender.finish().unwrap();
                drop(dataset.buffered_appender().unwrap());
            }
            Self::Owned => {
                let mut appender = dataset.into_buffered_appender().unwrap();
                appender.append::<i32>(&[]).unwrap();
                appender.append_raw(&[]).unwrap();
                appender.flush().unwrap();
                appender.finish().unwrap();
            }
            Self::BorrowedPending => {
                let mut appender = dataset.buffered_appender().unwrap();
                appender.append(&[7_i32, 9]).unwrap();
                assert_eq!(appender.buffered_elements(), 2);
                assert_eq!(
                    appender.discard(),
                    [7_i32.to_le_bytes(), 9_i32.to_le_bytes()].concat()
                );
            }
            Self::OwnedPending => {
                let mut appender = dataset.into_buffered_appender().unwrap();
                appender.append(&[7_i32, 9]).unwrap();
                assert_eq!(appender.buffered_elements(), 2);
                assert_eq!(
                    appender.discard(),
                    [7_i32.to_le_bytes(), 9_i32.to_le_bytes()].concat()
                );
            }
        }
    }
}
