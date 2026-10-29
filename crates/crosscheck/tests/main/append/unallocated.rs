use std::fs;
use std::path::Path;

use hdf5_pure::{Dataset, Error, File, FileAccessProperties, MemoryStrategy, SyncPolicy};
use rstest::rstest;
use test_util_hdf5::__private::unallocated_append;
use test_util_hdf5::file_space_oracle;

use crate::test_util::array_index;

#[rstest]
#[case::immediate(FirstWrite::Immediate)]
#[case::borrowed_manual(FirstWrite::BorrowedManual)]
#[case::borrowed_automatic(FirstWrite::BorrowedAutomatic)]
#[case::borrowed_terminal(FirstWrite::BorrowedTerminal)]
#[case::owned_manual(FirstWrite::OwnedManual)]
#[case::owned_automatic(FirstWrite::OwnedAutomatic)]
#[case::owned_terminal(FirstWrite::OwnedTerminal)]
fn deferred_native_indexes_initialize_and_resume_through_both_libraries(
    #[case] operation: FirstWrite,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values([32, 4, 4, 16, 10], [24, 8, 2, 8, 8])] parameters: [u8; 5],
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("initialize.h5");
    let header = unallocated_append::fixture(&path, filtered, paged, parameters);
    let before = fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let count = operation.run(file.dataset("d").unwrap(), &path, &before);
    let mut target = file.dataset("d").unwrap();
    assert_eq!(target.read_i32().unwrap(), (0..count).collect::<Vec<_>>());
    target.append(&(count..40).collect::<Vec<i32>>()).unwrap();
    assert_eq!(target.read_i32().unwrap(), (0..40).collect::<Vec<_>>());
    drop(target);
    file.close().unwrap();
    unallocated_append::inspect_initialized_index(&path, header, parameters);
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
    let pure = File::open(&path).unwrap();
    assert_eq!(
        array_index::pure_chunks(&pure),
        array_index::libhdf5_chunks(&path, 0)
    );
    assert_eq!(
        pure.dataset("d").unwrap().read_i32().unwrap(),
        (0..40).collect::<Vec<_>>()
    );
    pure.close().unwrap();
    let native = hdf5::File::open_rw(&path).unwrap();
    let target = native.dataset("d").unwrap();
    assert_eq!(
        target.read_raw::<i32>().unwrap(),
        (0..40).collect::<Vec<_>>()
    );
    target.resize([44]).unwrap();
    target
        .write_slice(&(40..44).collect::<Vec<i32>>(), 40..44)
        .unwrap();
    drop(target);
    native.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut target = file.dataset("d").unwrap();
    assert_eq!(target.read_i32().unwrap(), (0..44).collect::<Vec<_>>());
    target.append(&(44..47).collect::<Vec<i32>>()).unwrap();
    drop(target);
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let target = native.dataset("d").unwrap();
    assert_eq!(
        target.read_raw::<i32>().unwrap(),
        (0..47).collect::<Vec<_>>()
    );
    drop(target);
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
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
    fn run(self, mut target: Dataset, path: &Path, before: &[u8]) -> i32 {
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
                    assert_eq!(fs::read(path).unwrap(), before);
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
                    assert_eq!(fs::read(path).unwrap(), before);
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
fn unallocated_native_indexes_preserve_storage_during_read_only_append_operations(
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
        File::from_bytes(fs::read(&path).unwrap())
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
    let observed = fs::read(&path).unwrap();
    assert_eq!(observed, original);
    unallocated_append::assert_unallocated_layout(&observed, header, parameters);
    file.close().unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    file_space_oracle::assert_persisted_sections(&path);
    let native = hdf5::File::open(&path).unwrap();
    let target = native.dataset("d").unwrap();
    assert_eq!(target.shape(), [0]);
    assert_eq!(target.num_chunks(), Some(0));
    assert_eq!(target.storage_size(), 0);
    assert!(target.read_raw::<i32>().unwrap().is_empty());
    drop(target);
    native.close().unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
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
