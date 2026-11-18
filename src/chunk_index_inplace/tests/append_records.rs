use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_core::LengthWidth;
use hdf5_pure_format::__private::{ExtensibleArrayHeader, ExtensibleArrayStats};
use rstest::rstest;
use test_util::disk_log::{self, DiskOp, FailurePoint, IoEvent, IoOutcome};

use crate::{
    AddressRange, AppendOptions, DatasetCreateProperties, DeflateLevel, Error, File,
    FileAccessProperties, FileBuilder, FileLocking, FileSpaceSettings, FileSpaceStrategy,
    FreeSections, LossyTailReencoding, MaxExtent, PartialEdgeFiltering, PhysicalAddress,
    ScaleOffset, SyncPolicy,
};

pub(super) fn open(path: &Path) -> File {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_sync_policy(SyncPolicy::Always),
    )
    .unwrap()
}

#[rstest]
#[case::direct(4, 54, [0, 0, 1, 16], [0, 0, 3, 80])]
#[case::secondary(244, 444, [1, 54, 7, 304], [1, 54, 10, 496])]
fn retry_records_all_referenced_ea_blocks(
    #[case] start: i32,
    #[case] staged_end: i32,
    #[case] before_references: [u64; 4],
    #[case] after_references: [u64; 4],
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ea.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
    let dataset = builder
        .create_dataset("d")
        .with_i32_data(&(0..start).collect::<Vec<_>>())
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[1]);
    if filtered {
        dataset.with_deflate(DeflateLevel::One);
    }
    builder.write(&path).unwrap();
    let file = open(&path);
    let base = fs::read(&path).unwrap();
    let [header] = test_util::bytes::signature_offsets(&base, b"EAHD")[..] else {
        panic!("expected one Extensible Array header");
    };
    disk_log::start();
    file.dataset("d")
        .unwrap()
        .append(&(start + 1000..staged_end + 1000).collect::<Vec<_>>())
        .unwrap();
    let trace = disk_log::finish();
    file.close().unwrap();

    let allocation = trace
        .operations
        .iter()
        .position(|operation| matches!(operation, DiskOp::Write { offset: 0, .. }))
        .unwrap();
    let mut payloads = base.clone();
    for operation in &trace.operations[..=allocation] {
        operation.apply(&mut payloads);
    }
    let recorded_eof = |image: &[u8]| {
        crate::superblock::read(image)
            .unwrap()
            .superblock
            .eof_address
    };
    assert!(recorded_eof(&base) < recorded_eof(&payloads));
    assert!(recorded_eof(&payloads) <= payloads.len() as u64);
    let fields_start =
        header + ExtensibleArrayStats::fields_range(LengthWidth::Eight).start as usize;
    let statistics = trace
        .operations
        .iter()
        .position(|op| match op {
            DiskOp::Write { offset, bytes } => {
                let start = usize::try_from(*offset).unwrap();
                start <= fields_start && fields_start < start + bytes.len()
            }
            DiskOp::SetLen(_) => false,
        })
        .unwrap();
    assert!(allocation < statistics);
    let mut references = base;
    for op in &trace.operations[..statistics] {
        op.apply(&mut references);
    }
    assert_ne!(payloads, references);
    let mut published_statistics = references.clone();
    trace.operations[statistics].apply(&mut published_statistics);
    assert_eq!(ea_stats(&references, header)[4], start as u64);
    assert_eq!(
        ea_stats(&published_statistics, header)[4],
        staged_end as u64
    );

    for (image, expected) in [
        (payloads, before_references),
        (references, after_references),
        (published_statistics, after_references),
    ] {
        fs::write(&path, image).unwrap();
        let file = open(&path);
        let mut dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.read_i32().unwrap(), (0..start).collect::<Vec<_>>());
        dataset.append(&[start]).unwrap();
        dataset.append(&[start + 1]).unwrap();
        drop(dataset);
        file.close().unwrap();

        let [secondary, secondary_bytes, data, capacity] = expected;
        let bytes = fs::read(&path).unwrap();
        let element_size = ExtensibleArrayHeader::parse(&bytes, header, 8, 8)
            .unwrap()
            .element_size;
        assert_eq!(
            ea_stats(&bytes, header),
            [
                secondary,
                secondary_bytes,
                data,
                capacity * u64::from(element_size) + data * 22,
                u64::try_from(start + 2).unwrap(),
                4 + capacity,
            ]
        );
        let reopened = File::open(&path).unwrap();
        assert_eq!(
            reopened.dataset("d").unwrap().read_i32().unwrap(),
            (0..start + 2).collect::<Vec<_>>()
        );
        reopened.close().unwrap();
    }
}

fn ea_stats(bytes: &[u8], at: usize) -> [u64; 6] {
    ExtensibleArrayHeader::parse(bytes, at, 8, 8).unwrap();
    let fields = ExtensibleArrayStats::fields_range(LengthWidth::Eight);
    let start = at + usize::try_from(fields.start).unwrap();
    std::array::from_fn(|field| {
        let offset = start + field * 8;
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    })
}

#[test]
fn lossy_ea_raw_completion_keeps_readable_records_at_each_sync() {
    let states = replay_lossy_ea_append(
        PartialEdgeFiltering::Skip,
        LossyTailReencoding::Reject,
        &[1.0, 2.0],
        &[1.5, 1.6, 1.0, 2.0],
        |observed, state, free| {
            if state.mask == 1 {
                let start = usize::try_from(state.storage.start().get()).unwrap();
                let end = usize::try_from(state.storage.end().get()).unwrap();
                let expected: Vec<_> = [1.46_f64, 1.56, 1.0, 2.0]
                    .into_iter()
                    .flat_map(f64::to_le_bytes)
                    .collect();
                assert_eq!(observed[start..end], expected);
                assert!(
                    free.sections()
                        .iter()
                        .any(|section| section.range().contains(state.storage))
                );
            }
        },
    );
    assert_eq!(
        states,
        [
            (vec![1.46, 1.56], 0),
            (vec![1.46, 1.56], 1),
            (vec![1.46, 1.56, 1.0, 2.0], 1),
            (vec![1.5, 1.6, 1.0, 2.0], 0),
        ]
    );
}

#[rstest]
#[case::partial(&[-0.26], &[1.44, 1.54, -0.26])]
#[case::completion(&[-0.26, 2.0], &[1.44, 1.54, -0.26, 2.04])]
fn lossy_ea_filtered_reencoding_keeps_readable_records_at_each_sync(
    #[case] appended: &[f64],
    #[case] final_values: &[f64],
) {
    let states = replay_lossy_ea_append(
        PartialEdgeFiltering::Apply,
        LossyTailReencoding::Allow,
        appended,
        final_values,
        |_, _, _| {},
    );
    assert_eq!(
        states,
        [
            (vec![1.46, 1.56], 0),
            (vec![1.44, 1.54], 0),
            (final_values.to_vec(), 0),
        ]
    );
}

fn replay_lossy_ea_append(
    filtering: PartialEdgeFiltering,
    reencoding: LossyTailReencoding,
    appended: &[f64],
    final_values: &[f64],
    mut inspect_record: impl FnMut(&[u8], &LossyRecord, &FreeSections<PhysicalAddress>),
) -> Vec<(Vec<f64>, u32)> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lossy.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(1),
    );
    builder
        .create_dataset("d")
        .with_f64_data(&[1.46, 1.56])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_create_properties(
            DatasetCreateProperties::new()
                .with_chunks(&[4])
                .with_scale_offset(ScaleOffset::FloatDScale(1))
                .with_partial_edge_filtering(filtering),
        );
    builder.write(&path).unwrap();
    let fixture = fs::read(&path).unwrap();
    let options = AppendOptions::new().with_lossy_tail_reencoding(reencoding);
    let file = open(&path);
    let base = fs::read(&path).unwrap();
    let initial = lossy_record(&base);
    assert_eq!(initial.values, [1.46, 1.56]);
    assert_eq!(initial.mask, 0);
    disk_log::start();
    file.dataset("d")
        .unwrap()
        .append_with_options(options, appended)
        .unwrap();
    let trace = disk_log::finish();
    let final_image = fs::read(&path).unwrap();
    let complete = lossy_record(&final_image);
    assert_eq!(complete.values, final_values);
    assert_eq!(complete.mask, 0);
    assert_ne!(complete.storage, initial.storage);
    let free = file.reusable_free_sections().unwrap();
    assert!(
        free.sections()
            .iter()
            .any(|section| section.range().contains(initial.storage))
    );
    file.close().unwrap();

    let mut image = base.clone();
    let mut barriers = Vec::new();
    for event in trace.events {
        if let Some(index) = event.operation() {
            trace.operations[index].apply(&mut image);
        }
        if matches!(event, IoEvent::Synchronize { .. }) {
            assert_eq!(event.outcome(), IoOutcome::Succeeded);
            barriers.push(image.clone());
        }
    }
    assert_eq!(image, final_image);
    let mut states = Vec::new();
    let mut confirmed = initial.storage;
    for (syncs, observed) in barriers.into_iter().enumerate() {
        let state = lossy_record(&observed);
        inspect_record(&observed, &state, &free);
        if !states
            .last()
            .is_some_and(|(values, mask)| *values == state.values && *mask == state.mask)
        {
            states.push((state.values.clone(), state.mask));
        }
        fs::write(&path, &fixture).unwrap();
        let file = open(&path);
        assert_eq!(fs::read(&path).unwrap(), base);
        disk_log::start();
        disk_log::fail_sync_after(syncs, FailurePoint::Before);
        let error = file
            .dataset("d")
            .unwrap()
            .append_with_options(options, appended)
            .unwrap_err();
        let interrupted = disk_log::finish();
        disk_log::clear_failures();
        let Error::Io(error) = error else {
            panic!("expected a synchronization error, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk synchronization failure");
        assert_eq!(
            interrupted.events.last().unwrap().outcome(),
            IoOutcome::Failed(ErrorKind::Other)
        );
        assert_eq!(
            fs::read(&path).unwrap(),
            observed,
            "synchronization call {syncs}"
        );
        let reusable = file.reusable_free_sections().unwrap();
        assert!(
            reusable.sections().iter().all(|section| {
                let range = section.range();
                !range.overlaps(state.storage) && !range.overlaps(confirmed)
            }),
            "synchronization call {syncs} exposed live or unconfirmed storage"
        );
        assert_eq!(
            reusable
                .sections()
                .iter()
                .any(|section| section.range().contains(initial.storage)),
            confirmed != initial.storage,
            "synchronization call {syncs}"
        );
        confirmed = state.storage;
        drop(file);
        let reopened = File::open(&path).unwrap();
        assert_eq!(
            reopened.dataset("d").unwrap().read_f64().unwrap(),
            state.values
        );
        reopened.close().unwrap();
        let disk = fs::read(&path).unwrap();
        assert_eq!(lossy_record(&disk), state);
    }

    states
}

#[derive(Debug, PartialEq)]
struct LossyRecord {
    values: Vec<f64>,
    mask: u32,
    storage: AddressRange<PhysicalAddress>,
}

fn lossy_record(bytes: &[u8]) -> LossyRecord {
    let file = File::from_bytes(bytes.to_vec()).unwrap();
    let dataset = file.dataset("d").unwrap();
    let chunks = dataset.chunks().unwrap();
    let [chunk] = chunks.as_slice() else {
        panic!("expected one chunk")
    };
    LossyRecord {
        values: dataset.read_f64().unwrap(),
        mask: chunk.filter_mask().raw_bits(),
        storage: AddressRange::new(PhysicalAddress::new(chunk.address), chunk.storage_size)
            .unwrap(),
    }
}
