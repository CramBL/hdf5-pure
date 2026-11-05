use std::io::ErrorKind;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_space::__private::{Extent, ManagerKind, SessionSpace};
use rstest::rstest;
use test_util::disk_log;
use test_util_hdf5::file_space::chunk_holes;
use test_util_hdf5::file_space::persisted::PersistedStrategy;

use crate::Datatype;
use crate::DeflateLevel;
use crate::File;
use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::PhysicalAddress;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::dataspace::MaxExtent;
use crate::edit::SpaceOwnership;
use crate::edit::block::EncodingDependency;
use crate::edit::reservation::ReservationChoice;
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::fixed(None, 1_024)]
#[case::extensible(Some(MaxExtent::Unlimited), 1_024)]
#[case::sparse(Some(MaxExtent::Fixed(2_048)), 1_024)]
#[case::partial(None, 1_007)]
fn fragmented_creation_uses_original_holes_and_accounts_for_metadata_growth(
    #[case] maximum: Option<MaxExtent>,
    #[case] count: usize,
    #[values(false, true)] filtered: bool,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fragmented.h5");
    let (file, holes) = chunk_holes::prepare(&path, strategy);
    let before = file.file_size();
    let mut expected = values();
    expected.truncate(count);
    let available = file
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| section.address().get()..section.end_address().get())
        .collect::<Vec<_>>();
    for hole in &holes {
        assert!(
            available
                .iter()
                .any(|section| section.start <= hole.start && hole.end <= section.end)
        );
    }
    let payload = expected.len() * 4;
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| section.length().get() < u64::try_from(payload).unwrap())
    );
    let dataset = file
        .root()
        .create_dataset("replacement", |builder| {
            builder.with_i32_data(&expected).with_chunks(&[64]);
            if let Some(maximum) = maximum {
                builder.with_maxshape(&[maximum]);
            }
            if filtered {
                builder.with_deflate(DeflateLevel::Six);
            }
        })
        .unwrap();
    file.commit().unwrap();
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), chunk_holes::APPENDED_CHUNKS);
    let encoded_payload = chunks.iter().map(|chunk| chunk.storage_size).sum::<u64>();
    assert!(
        available
            .iter()
            .all(|section| section.end - section.start < encoded_payload)
    );
    for chunk in &chunks {
        assert!(
            available.iter().any(|hole| hole.start <= chunk.address
                && chunk.address + chunk.storage_size <= hole.end),
            "{chunk:?}, {holes:?}, {available:?}"
        );
    }
    assert_eq!(dataset.read_i32().unwrap(), expected);
    drop(dataset);
    file.close().unwrap();
    let reopened = File::open(&path).unwrap();
    let header = reopened
        .dataset("replacement")
        .unwrap()
        .header_address()
        .unwrap();
    reopened.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let index = engine
        .chunked_index_spans(ObjectId::new(PhysicalAddress::new(header)))
        .unwrap();
    assert!(index.len() >= 2);
    for span in &index {
        assert!(chunks.iter().all(|chunk| chunk.address + chunk.storage_size
            <= span.start().get()
            || span.end().get() <= chunk.address));
    }
    if matches!(strategy, PersistedStrategy::Page) {
        for span in &index {
            let start = span.start().get();
            assert!(if span.length() <= 4_096 {
                start % 4_096 + span.length() <= 4_096
            } else {
                start % 4_096 == 0
            });
            assert!(chunks.iter().all(|chunk| chunk.address / 4_096
                > (span.end().get() - 1) / 4_096
                || (chunk.address + chunk.storage_size - 1) / 4_096 < start / 4_096));
        }
    }
    let SpaceOwnership {
        physical_size,
        logical_size: _,
        userblock_space: _,
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        reusable_free_space,
        unclassified_space,
        analysis_complete,
        trailing_slack_space: _,
    } = engine.space_ownership().unwrap();
    assert!(analysis_complete);
    let growth = physical_size.saturating_sub(before);
    let appended = |regions: &[(u64, u64)]| {
        regions
            .iter()
            .map(|&(start, length)| (start + length).saturating_sub(start.max(before)))
            .sum::<u64>()
    };
    assert_eq!(appended(&live_raw_space), 0);
    assert_eq!(
        growth,
        appended(&live_metadata_space)
            + appended(&free_space_manager_space)
            + appended(&reusable_free_space)
            + appended(&unclassified_space)
    );
    assert!(
        growth
            < u64::try_from(payload).unwrap()
                + if matches!(strategy, PersistedStrategy::Page) {
                    8_192
                } else {
                    0
                },
        "growth={growth}, metadata={live_metadata_space:?}, managers={free_space_manager_space:?}, gaps={unclassified_space:?}"
    );
    eprintln!(
        "{strategy:?} filtered={filtered} maximum={maximum:?}: logical={payload} encoded={encoded_payload} growth={growth} index={} metadata={} managers={} reusable={} gaps={}",
        index.iter().map(|span| span.length()).sum::<u64>(),
        appended(&live_metadata_space),
        appended(&free_space_manager_space),
        appended(&reusable_free_space),
        appended(&unclassified_space)
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("replacement").unwrap().read_i32().unwrap(),
        expected
    );
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<i32>>()
    );
    for index in 0..chunk_holes::APPENDED_CHUNKS {
        assert_eq!(
            file.dataset(&format!("spacer{index}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..72).collect::<Vec<i32>>()
        );
    }
    file.close().unwrap();
}

#[rstest]
#[case::geometry(Failure::Geometry)]
#[case::header(Failure::Header)]
#[case::written_prefix(Failure::Prefix)]
fn fragmented_creation_failures_preserve_free_state_and_published_datasets(
    #[case] failure: Failure,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed-fragmented.h5");
    let (file, holes) = chunk_holes::prepare(&path, strategy);
    let names = file.root().datasets().unwrap();
    file.close().unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let before = engine
        .space
        .admitted_snapshot()
        .sections()
        .collect::<Vec<_>>();
    let expected = values();
    let mut builder = DatasetBuilder::new("replacement");
    match failure {
        Failure::Header => {
            builder
                .with_raw_data(
                    Datatype::Opaque {
                        size: 4,
                        tag: vec![b'x'; 248],
                    },
                    vec![17; 4_096],
                    1_024,
                )
                .with_chunks(&[64]);
        }
        Failure::Geometry | Failure::Prefix => {
            builder.with_i32_data(&expected).with_chunks(&[
                if matches!(failure, Failure::Geometry) {
                    1 << 30
                } else {
                    64
                },
            ]);
        }
    }
    disk_log::start();
    let staged = engine.stage_created_dataset(&ObjectPathBuf::parse("replacement"), builder);
    let error = if matches!(failure, Failure::Geometry) {
        staged.unwrap_err()
    } else {
        staged.unwrap();
        if matches!(failure, Failure::Prefix) {
            disk_log::fail_write_after(1, 3);
        }
        engine.commit().unwrap_err()
    };
    let operations = disk_log::take();
    match failure {
        Failure::Geometry => {
            assert!(
                matches!(
                    error,
                    Error::Format(FormatError::InvalidChunkGeometry(
                        "chunk logical byte size exceeds the reader's single-chunk allocation limit"
                    ))
                ),
                "{error:?}"
            );
            assert!(operations.is_empty(), "{operations:?}");
        }
        Failure::Header => {
            let mut payload = operations
                .iter()
                .filter_map(|operation| match operation {
                    disk_log::DiskOp::Write { offset, bytes } if bytes.as_slice() == [17; 256] => {
                        Some(*offset)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            payload.sort_unstable();
            assert_eq!(payload.len(), chunk_holes::APPENDED_CHUNKS);
            assert!(payload.windows(2).all(|pair| pair[0] + 256 <= pair[1]));
            assert!(
                matches!(
                    error,
                    Error::Format(FormatError::OpaqueTagTooLong { length: 248 })
                ),
                "{error:?}"
            );
        }
        Failure::Prefix => {
            let Error::Io(error) = error else {
                panic!("expected an injected write error, got {error:?}")
            };
            assert_eq!(error.kind(), ErrorKind::Other);
            assert_eq!(error.to_string(), "injected disk write failure");
            let writes = operations
                .iter()
                .filter_map(|operation| match operation {
                    disk_log::DiskOp::Write { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let raw = expected
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(writes, [&raw[..256], &raw[256..259]]);
            assert!(operations.iter().all(|operation| match operation {
                disk_log::DiskOp::Write { offset, bytes } =>
                    holes.iter().any(|hole| hole.start <= *offset
                        && *offset + u64::try_from(bytes.len()).unwrap() <= hole.end),
                _ => false,
            }));
        }
    }
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        before
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), names);
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<i32>>()
    );
    for index in 0..chunk_holes::APPENDED_CHUNKS {
        assert_eq!(
            file.dataset(&format!("spacer{index}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..72).collect::<Vec<i32>>()
        );
    }
    file.close().unwrap();
}

#[test]
fn metadata_only_page_space_keeps_one_compound_payload_request() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metadata-only.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let start = engine.append(&[13; 64]).unwrap();
    let extent = Extent::new(StoredAddress::new(start), ByteLength::new(64)).unwrap();
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::DEFAULT,
        [(Some(ManagerKind::PagedSmallMetadata), extent)],
    )
    .unwrap();
    let set = super::contiguous::prepared(super::Creation::Fixed(40), true);
    let length = set.chunks().iter().map(|chunk| chunk.stored_size()).sum();
    let (choice, plan) = engine
        .plan_created_array_storage(&set, length, &[])
        .unwrap();
    assert_eq!(choice, ReservationChoice::Preferred);
    assert_eq!(plan.reservations().len(), 1);
    assert_eq!(plan.reservations()[0].range().length(), length);
    assert_eq!(
        plan.reservations()[0].encoding_dependency(),
        EncodingDependency::Independent
    );
    assert_eq!(plan.work().indexed_extents, 0);
    drop(plan);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [extent]
    );
    assert_eq!(engine.image.len(), start + 64);
}

#[derive(Clone, Copy)]
enum Failure {
    Geometry,
    Header,
    Prefix,
}

fn values() -> Vec<i32> {
    (0..chunk_holes::APPENDED_CHUNKS * chunk_holes::CHUNK_ELEMENTS)
        .scan(0x34a9_12b7_u32, |state, _| {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            Some(i32::from_le_bytes(state.to_le_bytes()))
        })
        .collect()
}
