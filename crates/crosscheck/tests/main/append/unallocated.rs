use std::fs;
use std::path::Path;

use hdf5_pure::{
    Dataset, DatasetBuilder, DeflateLevel, Error, File, FileAccessProperties, FileBuilder,
    FileSpaceSettings, FileSpaceStrategy, MaxExtent, MemoryStrategy, SyncPolicy,
};
use rstest::rstest;
use test_util::extensible_array;
use test_util_hdf5::__private::{c_object, unallocated_append};
use test_util_hdf5::file_space_oracle;

use crate::test_util::array_index;

#[rstest]
#[case::immediate(AppendMethod::Immediate)]
#[case::borrowed_manual(AppendMethod::BorrowedManual)]
#[case::borrowed_automatic(AppendMethod::BorrowedAutomatic)]
#[case::borrowed_terminal(AppendMethod::BorrowedTerminal)]
#[case::owned_manual(AppendMethod::OwnedManual)]
#[case::owned_automatic(AppendMethod::OwnedAutomatic)]
#[case::owned_terminal(AppendMethod::OwnedTerminal)]
fn deferred_indexes_initialize_and_resume_through_both_libraries(
    #[case] operation: AppendMethod,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(
        Fixture::NativeDefault,
        Fixture::NativeCustom,
        Fixture::Pure(Creation::Buffered),
        Fixture::Pure(Creation::Streaming),
        Fixture::Pure(Creation::Edit)
    )]
    fixture: Fixture,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("initialize.h5");
    let (header, parameters) = fixture.write(&path, filtered, paged);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let count = operation.run(file.dataset("d").unwrap());
    let target = file.dataset("d").unwrap();
    assert_eq!(target.read_i32().unwrap(), (0..count).collect::<Vec<_>>());
    operation.write(file.dataset("d").unwrap(), &(count..40).collect::<Vec<_>>());
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
enum AppendMethod {
    BorrowedAutomatic,
    BorrowedManual,
    BorrowedTerminal,
    Immediate,
    OwnedAutomatic,
    OwnedManual,
    OwnedTerminal,
}

impl AppendMethod {
    fn run(self, target: Dataset) -> i32 {
        let count = match self {
            Self::BorrowedAutomatic | Self::OwnedAutomatic => 4,
            _ => 3,
        };
        self.write(target, &(0..count).collect::<Vec<_>>());
        count
    }

    fn write(self, mut target: Dataset, values: &[i32]) {
        match self {
            Self::Immediate => target.append(values).unwrap(),
            Self::BorrowedAutomatic | Self::BorrowedManual | Self::BorrowedTerminal => {
                let mut appender = target.buffered_appender().unwrap();
                appender.append(values).unwrap();
                if matches!(self, Self::BorrowedManual) {
                    appender.flush().unwrap();
                }
                appender.finish().unwrap();
            }
            Self::OwnedAutomatic | Self::OwnedManual | Self::OwnedTerminal => {
                let mut appender = target.into_buffered_appender().unwrap();
                appender.append(values).unwrap();
                if matches!(self, Self::OwnedManual) {
                    appender.flush().unwrap();
                }
                appender.finish().unwrap();
            }
        }
    }
}

#[rstest]
#[case::immediate(EmptyOperation::Immediate)]
#[case::borrowed(EmptyOperation::Borrowed)]
#[case::owned(EmptyOperation::Owned)]
#[case::borrowed_pending(EmptyOperation::BorrowedPending)]
#[case::owned_pending(EmptyOperation::OwnedPending)]
#[case::incompatible_input(EmptyOperation::IncompatibleInput)]
fn deferred_indexes_preserve_storage_during_read_only_append_operations(
    #[case] operation: EmptyOperation,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(
        Fixture::NativeDefault,
        Fixture::NativeCustom,
        Fixture::Pure(Creation::Buffered),
        Fixture::Pure(Creation::Streaming),
        Fixture::Pure(Creation::Edit)
    )]
    fixture: Fixture,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.h5");
    let (header, parameters) = fixture.write(&path, filtered, paged);
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
    if matches!(fixture, Fixture::NativeDefault | Fixture::NativeCustom) {
        assert!(sections.as_ref().unwrap().free_sections().free_bytes() > 0);
    }
    let reusable = file.reusable_space_stats().unwrap();
    operation.run(file.dataset("d").unwrap());
    assert_eq!(file.reusable_space_stats().unwrap(), reusable);
    let target = file.dataset("d").unwrap();
    assert_eq!(target.shape().unwrap(), [0]);
    assert_eq!(target.chunk_shape().unwrap(), Some(vec![4]));
    assert!(target.read_i32().unwrap().is_empty());
    assert!(target.chunks().unwrap().is_empty());
    drop(target);
    file.close().unwrap();
    let observed = fs::read(&path).unwrap();
    assert_eq!(observed, original);
    unallocated_append::assert_unallocated_layout(&observed, header, parameters);
    assert_eq!(
        File::from_bytes(observed)
            .unwrap()
            .persisted_free_sections()
            .unwrap(),
        sections
    );
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

#[derive(Clone, Copy)]
enum Fixture {
    NativeCustom,
    NativeDefault,
    Pure(Creation),
}

impl Fixture {
    fn write(self, path: &Path, filtered: bool, paged: bool) -> (usize, [u8; 5]) {
        let parameters = match self {
            Self::NativeCustom => [24, 8, 2, 8, 8],
            _ => [32, 4, 4, 16, 10],
        };
        match self {
            Self::NativeDefault | Self::NativeCustom => (
                unallocated_append::fixture(path, filtered, paged, parameters),
                parameters,
            ),
            Self::Pure(creation) => {
                creation.write(path, filtered, paged, 1, true);
                let native = hdf5::File::open(path).unwrap();
                let header = usize::try_from(c_object::header_address(&native, "d")).unwrap();
                native.close().unwrap();
                let bytes = fs::read(path).unwrap();
                unallocated_append::assert_unallocated_layout(&bytes, header, parameters);
                assert!(
                    test_util::bytes::signature_offsets(&bytes, extensible_array::SIGNATURE)
                        .is_empty()
                );
                assert!(
                    test_util::bytes::signature_offsets(&bytes, extensible_array::INDEX_SIGNATURE)
                        .is_empty()
                );
                (header, parameters)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Creation {
    Buffered,
    Edit,
    Streaming,
}

impl Creation {
    fn write(self, path: &Path, filtered: bool, paged: bool, count: usize, unallocated: bool) {
        let configure = |dataset: &mut DatasetBuilder| {
            if unallocated {
                dataset.with_unallocated_storage(hdf5_pure::make_i32_type(), &[0]);
            } else {
                dataset.with_i32_data(&[]).with_shape(&[0]);
            }
            dataset
                .with_chunks(&[4])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_fill_value(-7_i32);
            if filtered {
                dataset.with_shuffle().with_deflate(DeflateLevel::Six);
            }
        };
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(if paged {
                FileSpaceStrategy::Page
            } else {
                FileSpaceStrategy::FsmAggr
            })
            .with_persistence(true),
        );
        match self {
            Self::Buffered | Self::Streaming => {
                for column in 0..count {
                    configure(builder.create_dataset(&column_name(column)));
                }
                if matches!(self, Self::Buffered) {
                    fs::write(path, builder.finish().unwrap()).unwrap();
                } else {
                    builder.write(path).unwrap();
                }
            }
            Self::Edit => {
                builder.write(path).unwrap();
                let file = File::open_rw_with_options(
                    path,
                    FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
                )
                .unwrap();
                for column in 0..count {
                    file.root()
                        .create_dataset(&column_name(column), configure)
                        .unwrap();
                }
                file.commit().unwrap();
                file.close().unwrap();
            }
        }
    }
}

fn column_name(column: usize) -> String {
    if column == 0 {
        "d".into()
    } else {
        format!("column_{column:04}")
    }
}

#[rstest]
fn native_first_writes_initialize_pure_deferred_indexes(
    #[values(Creation::Buffered, Creation::Streaming, Creation::Edit)] creation: Creation,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-first.h5");
    Fixture::Pure(creation).write(&path, filtered, paged);
    let native = hdf5::File::open_rw(&path).unwrap();
    let target = native.dataset("d").unwrap();
    assert_eq!(target.shape(), [0]);
    assert_eq!(target.num_chunks(), Some(0));
    target.resize([3]).unwrap();
    target.write_raw(&[0_i32, 1, 2]).unwrap();
    assert_eq!(target.num_chunks(), Some(1));
    assert!(target.storage_size() > 0);
    drop(target);
    native.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut target = file.dataset("d").unwrap();
    assert_eq!(target.read_i32().unwrap(), [0, 1, 2]);
    target.append(&[3_i32, 4, 5]).unwrap();
    drop(target);
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let target = native.dataset("d").unwrap();
    assert_eq!(target.read_raw::<i32>().unwrap(), [0, 1, 2, 3, 4, 5]);
    drop(target);
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
}

#[rstest]
fn unused_columns_have_no_index_allocations(
    #[values(Creation::Buffered, Creation::Streaming, Creation::Edit)] creation: Creation,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("columns.h5");
    let eager_path = directory.path().join("eager.h5");
    let count = 128;
    creation.write(&path, filtered, paged, count, true);
    creation.write(&eager_path, filtered, paged, count, false);
    let original = fs::read(&path).unwrap();
    let eager = fs::read(&eager_path).unwrap();
    assert!(test_util::bytes::signature_offsets(&original, extensible_array::SIGNATURE).is_empty());
    assert!(
        test_util::bytes::signature_offsets(&original, extensible_array::INDEX_SIGNATURE)
            .is_empty()
    );
    assert_eq!(
        test_util::bytes::signature_offsets(&eager, extensible_array::SIGNATURE).len(),
        count
    );
    assert_eq!(
        test_util::bytes::signature_offsets(&eager, extensible_array::INDEX_SIGNATURE).len(),
        count
    );
    assert!(
        eager.len() > original.len() + count * 128,
        "{} eager bytes, {} deferred bytes",
        eager.len(),
        original.len()
    );
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(native.member_names().unwrap().len(), count);
    for column in 0..count {
        let name = column_name(column);
        let header = usize::try_from(c_object::header_address(&native, &name)).unwrap();
        unallocated_append::assert_unallocated_layout(&original, header, [32, 4, 4, 16, 10]);
        let target = native.dataset(&name).unwrap();
        assert_eq!(target.shape(), [0]);
        assert_eq!(target.num_chunks(), Some(0));
        assert_eq!(target.storage_size(), 0);
        assert!(target.read_raw::<i32>().unwrap().is_empty());
    }
    native.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    file.dataset("d").unwrap().append(&[17_i32, 19]).unwrap();
    file.close().unwrap();
    let initialized = fs::read(&path).unwrap();
    assert_eq!(
        test_util::bytes::signature_offsets(&initialized, extensible_array::SIGNATURE).len(),
        1
    );
    assert_eq!(
        test_util::bytes::signature_offsets(&initialized, extensible_array::INDEX_SIGNATURE).len(),
        1
    );
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        [17, 19]
    );
    for column in 1..count {
        let name = column_name(column);
        let header = usize::try_from(c_object::header_address(&native, &name)).unwrap();
        unallocated_append::assert_unallocated_layout(&initialized, header, [32, 4, 4, 16, 10]);
        let target = native.dataset(&name).unwrap();
        assert_eq!(target.shape(), [0]);
        assert_eq!(target.num_chunks(), Some(0));
    }
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
}
