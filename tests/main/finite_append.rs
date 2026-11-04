use std::fs;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileLocking;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::Layout;
use hdf5_pure::MemoryStrategy;
use rstest::rstest;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::FiniteChunkIndex;

#[rstest]
#[case::none(FileSpaceStrategy::None)]
#[case::aggregate(FileSpaceStrategy::Aggr)]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn finite_chunk_appends_preserve_extents_after_reopening(
    #[case] strategy: FileSpaceStrategy,
    #[values(FiniteChunkIndex::Fixed, FiniteChunkIndex::Single)] index: FiniteChunkIndex,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
    #[values(false, true)] filtered: bool,
    #[values(
        Operation::Immediate,
        Operation::Raw,
        Operation::Borrowed,
        Operation::Owned
    )]
    operation: Operation,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("finite.h5");
    dataset::write_finite_chunked_i32(
        &path,
        FileSpaceSettings::new(strategy).with_persistence(matches!(
            strategy,
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
        )),
        filtered,
        index,
    );
    let maximum = i32::try_from(index.maximum()).unwrap();
    let expected = (0..maximum).collect::<Vec<_>>();
    let access = byte_observation_access(backing);
    let file = File::open_rw_with_options(&path, access).unwrap();
    let mut target = file.dataset("d").unwrap();
    match operation {
        Operation::Immediate | Operation::Raw => {
            for value in 1_i32..maximum {
                if matches!(operation, Operation::Raw) {
                    target.append_raw(&value.to_le_bytes()).unwrap();
                } else {
                    target.append(&[value]).unwrap();
                }
                assert_eq!(target.read_i32().unwrap(), (0..=value).collect::<Vec<_>>());
            }
        }
        Operation::Borrowed => {
            let mut appender = target.buffered_appender().unwrap();
            for value in 1_i32..maximum {
                appender.append(&[value]).unwrap();
                appender.flush().unwrap();
            }
            appender.finish().unwrap();
        }
        Operation::Owned => {
            let mut appender = target.into_buffered_appender().unwrap();
            appender.append(&expected[1..]).unwrap();
            appender.finish().unwrap();
            target = file.dataset("d").unwrap();
        }
    }
    assert_eq!(target.read_i32().unwrap(), expected);
    assert_eq!(target.shape().unwrap(), [index.maximum()]);
    assert!(matches!(target.layout().unwrap(), Layout::Chunked { .. }));
    let before = fs::read(&path).unwrap();
    let error = target.append(&[maximum]).unwrap_err();
    assert!(
        matches!(&error,Error::AppendInPlaceUnsupported(reason) if *reason=="append exceeds the dataset's maximum first dimension"),
        "{error:?}"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    drop(target);
    file.close().unwrap();
    let file = File::open_rw_with_options(&path, access).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
    assert_eq!(file.superblock().offset_size.get(), 8);
    assert_eq!(file.superblock().length_size.get(), 8);
    file.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::immediate(Operation::Immediate)]
#[case::raw(Operation::Raw)]
#[case::borrowed(Operation::Borrowed)]
#[case::owned(Operation::Owned)]
fn finite_maximum_rejection_preserves_pending_input_and_published_values(
    #[case] operation: Operation,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("maximum.h5");
    dataset::write_single_chunk_i32(&path, FileSpaceSettings::default(), false);
    let file =
        File::open_rw_with_options(&path, byte_observation_access(MemoryStrategy::Auto)).unwrap();
    let mut target = file.dataset("d").unwrap();
    let before = fs::read(&path).unwrap();
    let expected_pending = 1_i32.to_le_bytes();
    match operation {
        Operation::Immediate | Operation::Raw => {
            let error = if matches!(operation, Operation::Raw) {
                target
                    .append_raw(
                        &[1_i32, 2, 3, 4]
                            .into_iter()
                            .flat_map(i32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap_err()
            } else {
                target.append(&[1_i32, 2, 3, 4]).unwrap_err()
            };
            assert!(
                matches!(&error,Error::AppendInPlaceUnsupported(reason) if *reason=="append exceeds the dataset's maximum first dimension"),
                "{error:?}"
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            target.append(&[1_i32, 2, 3]).unwrap();
        }
        Operation::Borrowed => {
            let mut appender = target.buffered_appender().unwrap();
            appender.append(&[1_i32]).unwrap();
            let error = appender.append(&[2_i32, 3, 4]).unwrap_err();
            assert!(
                matches!(&error,Error::AppendInPlaceUnsupported(reason) if *reason=="append exceeds the dataset's maximum first dimension"),
                "{error:?}"
            );
            assert_eq!(appender.unwritten(), expected_pending);
            assert_eq!(fs::read(&path).unwrap(), before);
            appender.append(&[2_i32, 3]).unwrap();
            appender.finish().unwrap();
        }
        Operation::Owned => {
            let mut appender = target.into_buffered_appender().unwrap();
            appender.append(&[1_i32]).unwrap();
            let error = appender.append(&[2_i32, 3, 4]).unwrap_err();
            assert!(
                matches!(&error,Error::AppendInPlaceUnsupported(reason) if *reason=="append exceeds the dataset's maximum first dimension"),
                "{error:?}"
            );
            assert_eq!(appender.unwritten(), expected_pending);
            assert_eq!(fs::read(&path).unwrap(), before);
            appender.append(&[2_i32, 3]).unwrap();
            appender.finish().unwrap();
            target = file.dataset("d").unwrap();
        }
    }
    assert_eq!(target.read_i32().unwrap(), [0, 1, 2, 3]);
    drop(target);
    file.close().unwrap();
    directory.close().unwrap();
}

#[test]
fn buffered_finite_capacity_accounts_for_other_handle_appends() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("other-handle.h5");
    dataset::write_single_chunk_i32(&path, FileSpaceSettings::default(), false);
    let file =
        File::open_rw_with_options(&path, byte_observation_access(MemoryStrategy::Auto)).unwrap();
    let mut target = file.dataset("d").unwrap();
    let mut other = file.dataset("d").unwrap();
    let mut appender = target.buffered_appender().unwrap();
    other.append(&[1_i32]).unwrap();
    appender.append(&[2_i32]).unwrap();
    let before = fs::read(&path).unwrap();
    let error = appender.append(&[3_i32, 4]).unwrap_err();
    assert!(
        matches!(&error,Error::AppendInPlaceUnsupported(reason) if *reason=="append exceeds the dataset's maximum first dimension"),
        "{error:?}"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(appender.unwritten(), 2_i32.to_le_bytes());
    appender.append(&[3_i32]).unwrap();
    appender.finish().unwrap();
    assert_eq!(target.read_i32().unwrap(), [0, 1, 2, 3]);
    assert_eq!(other.read_i32().unwrap(), [0, 1, 2, 3]);
    drop(other);
    drop(target);
    file.close().unwrap();
    directory.close().unwrap();
}

#[derive(Clone, Copy)]
enum Operation {
    Borrowed,
    Immediate,
    Owned,
    Raw,
}

/// Returns fixture access properties with locking disabled for live byte snapshots.
fn byte_observation_access(memory: MemoryStrategy) -> FileAccessProperties {
    FileAccessProperties::new()
        .with_memory_strategy(memory)
        .with_locking(FileLocking::Disabled)
}
