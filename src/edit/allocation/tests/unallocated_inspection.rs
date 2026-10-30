//! Physical-byte checks for empty appends and rejected input during a locked editing session.

use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::ObjectHeader;
use rstest::rstest;
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;

use crate::Dataset;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::MessageType;
use crate::ScaleOffset;
use crate::type_builders;

#[rstest]
#[case::plain(false)]
#[case::filtered(true)]
fn unreadable_padding_rejection_preserves_issued_bytes(#[case] filtered: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fill.h5");
    let mut builder = FileBuilder::new();
    let target = builder.create_dataset("schema");
    target
        .with_unallocated_storage(type_builders::make_i32_type(), &[0])
        .with_chunks(&[4])
        .with_maxshape(&[MaxExtent::Unlimited]);
    if filtered {
        target.with_fletcher32();
    }
    builder.write(&path).unwrap();
    let mut image = std::fs::read(&path).unwrap();
    assert_index_allocation(&image, false);
    let (header, fill) = test_util::bytes::signature_offsets(&image, v2::SIGNATURE)
        .into_iter()
        .find_map(|at| {
            ObjectHeader::parse(&image, AccessMode::ReadOnly, at, 8, 8)
                .unwrap()
                .messages
                .into_iter()
                .find(|message| message.msg_type == MessageType::FILL_VALUE)
                .map(|message| (at, message.data))
        })
        .unwrap();
    let mut fill = fill;
    fill[0] = u8::MAX;
    v2::replace_message(&mut image, header, FixtureMessage::FILL_VALUE, &fill);
    std::fs::write(&path, &image).unwrap();
    let file = File::open_rw(&path).unwrap();
    let error = file
        .dataset("schema")
        .unwrap()
        .append(&[3_i32])
        .unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::UnreadableFillValue)),
        "{error:?}"
    );
    assert_eq!(file.disk_bytes().unwrap(), image);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), image);
}

#[rstest]
#[case::immediate(EmptyAppendOperation::Immediate)]
#[case::borrowed_drop(EmptyAppendOperation::BorrowedDrop)]
#[case::borrowed_flush(EmptyAppendOperation::BorrowedFlush)]
#[case::borrowed_finish(EmptyAppendOperation::BorrowedFinish)]
#[case::owned_drop(EmptyAppendOperation::OwnedDrop)]
#[case::owned_flush(EmptyAppendOperation::OwnedFlush)]
#[case::owned_finish(EmptyAppendOperation::OwnedFinish)]
#[case::borrowed_pending(EmptyAppendOperation::BorrowedPending)]
#[case::owned_pending(EmptyAppendOperation::OwnedPending)]
fn empty_append_operations_preserve_issued_bytes(
    #[case] operation: EmptyAppendOperation,
    #[values(false, true)] unallocated: bool,
    #[values(false, true)] filtered: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("schema.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    let dataset = builder.create_dataset("schema");
    if unallocated {
        dataset.with_unallocated_storage(type_builders::make_i32_type(), &[0]);
    } else {
        dataset.with_i32_data(&[]).with_shape(&[0]);
    }
    dataset
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    if filtered {
        dataset.with_fletcher32();
    }
    builder.write(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    assert_index_allocation(&original, !unallocated);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    let free = File::from_bytes(original.clone())
        .unwrap()
        .persisted_free_sections()
        .unwrap();
    let reusable = file.reusable_space_stats().unwrap();
    operation.run(file.dataset("schema").unwrap());
    assert_eq!(
        File::from_bytes(file.disk_bytes().unwrap())
            .unwrap()
            .persisted_free_sections()
            .unwrap(),
        free
    );
    assert_eq!(file.reusable_space_stats().unwrap(), reusable);
    assert_eq!(file.dataset("schema").unwrap().shape().unwrap(), [0]);
    assert!(file.dataset("schema").unwrap().chunks().unwrap().is_empty());
    assert_eq!(file.disk_bytes().unwrap(), original);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[derive(Clone, Copy)]
enum EmptyAppendOperation {
    BorrowedDrop,
    BorrowedFinish,
    BorrowedFlush,
    BorrowedPending,
    Immediate,
    OwnedDrop,
    OwnedFinish,
    OwnedFlush,
    OwnedPending,
}

impl EmptyAppendOperation {
    fn run(self, mut dataset: Dataset) {
        match self {
            Self::Immediate => {
                dataset.append::<i32>(&[]).unwrap();
                dataset.append_raw(&[]).unwrap();
            }
            Self::BorrowedDrop
            | Self::BorrowedFinish
            | Self::BorrowedFlush
            | Self::BorrowedPending => {
                let mut appender = dataset.buffered_appender().unwrap();
                appender.append::<i32>(&[]).unwrap();
                appender.append_raw(&[]).unwrap();
                assert_eq!(appender.buffered_elements(), 0);
                match self {
                    Self::BorrowedFinish => appender.finish().unwrap(),
                    Self::BorrowedFlush => appender.flush().unwrap(),
                    Self::BorrowedPending => {
                        appender.append(&[7_i32, 9]).unwrap();
                        assert_eq!(appender.buffered_elements(), 2);
                        assert_eq!(
                            appender.discard(),
                            [7_i32.to_le_bytes(), 9_i32.to_le_bytes()].concat()
                        );
                    }
                    _ => {}
                }
            }
            Self::OwnedDrop | Self::OwnedFinish | Self::OwnedFlush | Self::OwnedPending => {
                let mut appender = dataset.into_buffered_appender().unwrap();
                appender.append::<i32>(&[]).unwrap();
                appender.append_raw(&[]).unwrap();
                assert_eq!(appender.buffered_elements(), 0);
                match self {
                    Self::OwnedFinish => appender.finish().unwrap(),
                    Self::OwnedFlush => appender.flush().unwrap(),
                    Self::OwnedPending => {
                        appender.append(&[7_i32, 9]).unwrap();
                        assert_eq!(appender.buffered_elements(), 2);
                        assert_eq!(
                            appender.discard(),
                            [7_i32.to_le_bytes(), 9_i32.to_le_bytes()].concat()
                        );
                    }
                    _ => {}
                }
            }
        }
    }
}

#[rstest]
#[case::typed(|dataset: &mut Dataset| dataset.append(&[1_u32]), "append datatype does not match the on-disk dataset (wrong element type or byte order)")]
#[case::typed_empty(|dataset: &mut Dataset| dataset.append::<u16>(&[]), "append datatype does not match the on-disk dataset (wrong element type or byte order)")]
#[case::raw_fragment(|dataset: &mut Dataset| dataset.append_raw(&[1]), "appended byte length is not a whole number of elements")]
fn incompatible_append_rejections_preserve_issued_bytes(
    #[case] operation: fn(&mut Dataset) -> Result<(), Error>,
    #[case] expected: &str,
    #[values(false, true)] unallocated: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("incompatible.h5");
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("d");
    if unallocated {
        dataset.with_unallocated_storage(type_builders::make_i32_type(), &[0]);
    } else {
        dataset.with_i32_data(&[]).with_shape(&[0]);
    }
    dataset
        .with_chunks(&[4])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    assert_index_allocation(&original, !unallocated);
    let file = File::open_rw(&path).unwrap();
    let mut target = file.dataset("d").unwrap();
    let error = operation(&mut target).unwrap_err();
    let Error::AppendInPlaceUnsupported(reason) = error else {
        panic!("expected incompatible append input, got {error:?}");
    };
    assert_eq!(reason, expected);
    drop(target);
    assert_eq!(file.disk_bytes().unwrap(), original);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[rstest]
fn empty_lossy_appenders_preserve_issued_bytes(
    #[values(false, true)] unallocated: bool,
    #[values(false, true)] owned: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lossy.h5");
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("d");
    if unallocated {
        dataset.with_unallocated_storage(type_builders::make_f64_type(), &[0]);
    } else {
        dataset.with_f64_data(&[]).with_shape(&[0]);
    }
    dataset
        .with_chunks(&[4])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_scale_offset(ScaleOffset::FloatDScale(1));
    builder.write(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    assert_index_allocation(&original, !unallocated);
    let file = File::open_rw(&path).unwrap();
    let mut target = file.dataset("d").unwrap();
    if owned {
        let mut appender = target.into_buffered_appender().unwrap();
        appender.append::<f64>(&[]).unwrap();
        appender.flush().unwrap();
        appender.finish().unwrap();
    } else {
        let mut appender = target.buffered_appender().unwrap();
        appender.append::<f64>(&[]).unwrap();
        appender.flush().unwrap();
        appender.finish().unwrap();
        drop(target);
    }
    assert_eq!(file.disk_bytes().unwrap(), original);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

fn assert_index_allocation(image: &[u8], allocated: bool) {
    let layouts: Vec<_> = test_util::bytes::signature_offsets(image, v2::SIGNATURE)
        .into_iter()
        .flat_map(|at| {
            ObjectHeader::parse(image, AccessMode::ReadOnly, at, 8, 8)
                .unwrap()
                .messages
        })
        .filter(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .collect();
    assert_eq!(layouts.len(), 1);
    let layout = &layouts[0].data;
    let DataLayout::Chunked {
        index: ChunkIndexLayout::ExtensibleArray { address, .. },
        ..
    } = DataLayout::parse(layout, 8, 8).unwrap()
    else {
        panic!("expected an Extensible Array layout");
    };
    assert_eq!(address.is_some(), allocated);
    assert_eq!(layout[layout.len() - 8..] == [u8::MAX; 8], !allocated);
}
