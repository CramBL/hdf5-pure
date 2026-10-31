//! Checks section thresholds, trailing truncation, and PAGE allocation classes and buffering.

use std::fs;

use hdf5_pure_space::__private::AllocationClass;
use rstest::rstest;
use test_util::allocation_oracle;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::AllocationState;
use test_util::allocation_oracle::Context;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::disk_log;
use test_util::disk_log::DiskOp;

use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceInfo;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::SyncPolicy;
use crate::chunk_index_inplace::Store;
use crate::data_layout::DataLayout;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::free_sections::FreeSpaceManagerRole;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;

use super::super::CHUNK;
use super::super::Fact;
use super::super::Observation;
use super::super::Workload;
use super::super::WritePath;

#[rstest]
#[case::every_section(1)]
#[case::large_sections(65_535)]
fn append_publication_preserves_configured_section_threshold(
    #[case] threshold: u64,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("threshold.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(settings(strategy, threshold));
    builder
        .create_dataset("d")
        .with_unallocated_storage(crate::make_i32_type(), &[0])
        .with_chunks(&[u64::try_from(CHUNK).unwrap()])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder
        .create_dataset("scratch")
        .with_u8_data(&vec![7; SCRATCH]);
    builder.create_dataset("ceiling").with_u8_data(&[9; 8192]);
    builder.write(&path).unwrap();
    let reader = File::open(&path).unwrap();
    let DataLayout::Contiguous {
        address: Some(address),
        size,
    } = reader.dataset("scratch").unwrap().data_layout().unwrap()
    else {
        panic!("expected contiguous scratch storage")
    };
    let hole = Coverage::new([Interval::new(address.get(), size).unwrap()]);
    drop(reader);
    let before_free = Space::read(&fs::read(&path).unwrap()).free;
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    drop(engine);
    let initial = Space::read(&fs::read(&path).unwrap());
    assert_eq!(initial.info.threshold, threshold);
    assert_eq!(initial.info.strategy, strategy);
    let data = (0..CHUNK)
        .map(|value| i32::try_from(value).unwrap())
        .collect::<Vec<_>>();
    WritePath::Immediate.append(&path, &data);
    let bytes = fs::read(&path).unwrap();
    let after = Space::read(&bytes);
    assert_eq!(after.info.threshold, threshold);
    if threshold == 1 {
        assert!(
            !after.free.intersection(&hole).is_empty(),
            "{strategy:?}: expected retained unused scratch storage"
        );
    } else {
        assert!(after.eof < threshold);
        match strategy {
            FileSpaceStrategy::Page => {
                assert!(
                    before_free
                        .intervals()
                        .iter()
                        .any(|section| section.start() == address.get() + size
                            || section.end() == address.get())
                );
                assert!(!initial.free.intersection(&hole).is_empty());
                assert!(!after.free.intersection(&hole).is_empty());
                assert!(!after.free.is_empty());
                assert!(
                    after
                        .sections
                        .iter()
                        .all(|section| section.length() < threshold)
                );
            }
            FileSpaceStrategy::FsmAggr => {
                assert!(after.free.intersection(&hole).is_empty());
                assert!(after.free.is_empty());
            }
            _ => panic!("unexpected strategy {strategy:?}"),
        }
    }
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.dataset("d").unwrap().read_i32().unwrap(), data);
    assert_eq!(
        reader.dataset("ceiling").unwrap().read_u8().unwrap(),
        [9; 8192]
    );
    drop(reader);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn confirmed_deletion_reclaims_a_trailing_allocation(#[case] strategy: FileSpaceStrategy) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trailing.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(settings(strategy, 1));
    builder.create_dataset("d").with_i32_data(&[0, 1, 2, 3]);
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    let mut tail = DatasetBuilder::new("tail");
    tail.with_u8_data(&vec![17; 131_072]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("tail"), tail)
        .unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    drop(engine);
    let base = fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    disk_log::start();
    engine.delete(&ObjectPathBuf::parse("tail")).unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    drop(engine);
    let recording = disk_log::finish();
    assert!(!recording.operations.is_empty());
    assert!(!recording.events.is_empty());
    let after = fs::read(&path).unwrap();
    assert!(
        after.len() < base.len(),
        "{strategy:?}: original {}, final {}",
        base.len(),
        after.len()
    );
    let mut image = base;
    let mut shrinks = 0;
    for (point, operation) in recording.operations.iter().enumerate() {
        if let DiskOp::SetLen(length) = operation {
            shrinks += usize::from(*length < u64::try_from(image.len()).unwrap());
        }
        operation.apply(&mut image);
        let reader = File::from_bytes(image.clone()).unwrap();
        assert_eq!(
            reader.dataset("d").unwrap().read_i32().unwrap(),
            [0, 1, 2, 3],
            "{strategy:?}, prefix {point}"
        );
        let datasets = reader.root().datasets().unwrap();
        if datasets.iter().any(|dataset| dataset == "tail") {
            assert_eq!(datasets, ["d", "tail"]);
            assert_eq!(
                reader.dataset("tail").unwrap().read_u8().unwrap(),
                vec![17; 131_072]
            );
        } else {
            assert_eq!(datasets, ["d"]);
        }
    }
    assert!(shrinks > 0);
    assert_eq!(image, after);
    let space = Space::read(&after);
    assert_eq!(space.eof, u64::try_from(after.len()).unwrap());
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.root().datasets().unwrap(), ["d"]);
    drop(reader);
    assert_eq!(fs::read(&path).unwrap(), after);
}

#[rstest]
#[case::ordered(false)]
#[case::page_buffered(true)]
fn page_classes_retain_unused_tails_across_ownership_publications(#[case] buffered: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("classes.h5");
    let hole = super::super::pure_fixture(&path, true, false, false);
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    if buffered {
        engine
            .set_write_buffering(WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 65_536,
            })
            .unwrap();
    }
    let mut actual: Vec<Payload> = Vec::new();
    for (step, length) in [137, 3000, 5000].into_iter().enumerate() {
        let base = engine.disk_bytes().unwrap();
        let payload = vec![u8::try_from(step + 31).unwrap(); length];
        disk_log::start();
        let allocation = engine
            .store()
            .allocate(&payload, AllocationClass::Raw)
            .unwrap();
        engine.image.flush().unwrap();
        let prepared = engine.disk_bytes().unwrap();
        let stored = Payload::find(&prepared, payload);
        let publication = engine.store().publish_allocations().unwrap();
        publication.confirm(allocation).unwrap();
        let recording = disk_log::finish();
        assert!(!recording.operations.is_empty());
        assert!(!recording.events.is_empty());
        let published = engine.disk_bytes().unwrap();
        let mut image = base;
        let mut started = false;
        let mut complete = false;
        let mut owned = false;
        let mut saw_provisional = false;
        for point in 0..=recording.operations.len() {
            if point > 0 {
                let operation = &recording.operations[point - 1];
                started |= stored.written_by(operation);
                operation.apply(&mut image);
            }
            let snapshot = Observation::read(&image);
            let mut allocations = snapshot
                .live
                .iter()
                .map(|fact| fact.allocation(AllocationState::Linked))
                .collect::<Vec<_>>();
            for payload in &actual {
                payload.assert_owned(&image, snapshot.eof);
                allocations.push(payload.fact.allocation(AllocationState::Orphaned));
            }
            if started {
                complete |= stored.bytes_in(&image) == Some(stored.bytes.as_slice());
                if complete {
                    stored.assert_bytes(&image);
                }
                owned |= stored.fact.range.end() <= snapshot.eof
                    && Coverage::new([stored.fact.range])
                        .intersection(&snapshot.free)
                        .is_empty();
                let state = if owned {
                    stored.assert_owned(&image, snapshot.eof);
                    AllocationState::Orphaned
                } else {
                    saw_provisional = true;
                    AllocationState::Provisional
                };
                allocations.push(stored.fact.allocation(state));
            }
            allocation_oracle::inspect(
                Context {
                    case: "PAGE class allocation prefix",
                    interruption: point,
                    publication: snapshot.state(),
                },
                &hole,
                &snapshot.free,
                &snapshot.reachable(),
                &allocations,
            )
            .unwrap();
        }
        assert!(started);
        assert!(complete);
        assert!(owned);
        assert!(saw_provisional);
        assert_eq!(image, published);
        actual.push(stored);
        let space = Space::read(&published);
        assert!(
            space
                .roles
                .contains(&FreeSpaceManagerRole::PageSmallRawData)
        );
        assert!(
            space
                .roles
                .contains(&FreeSpaceManagerRole::PageLargeGeneric)
        );
        assert!(
            space
                .roles
                .contains(&FreeSpaceManagerRole::PageSmallMetadata)
        );
    }
    let before = engine.disk_bytes().unwrap();
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), before);
    let eof = Space::read(&before).eof;
    for payload in actual {
        payload.assert_owned(&before, eof);
    }
}

#[rstest]
#[case::flat(false)]
#[case::page(true)]
fn page_buffered_index_creation_preserves_complete_extents(#[case] paged: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("buffered-index.h5");
    let hole = super::super::pure_fixture(&path, paged, true, true);
    super::compact_geometry(&path);
    let workload = Workload::record_with(&path, super::ROUNDS, |data| {
        let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        engine.set_sync_policy(SyncPolicy::OnClose);
        engine
            .set_write_buffering(WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 65_536,
            })
            .unwrap();
        let mut append = AppendBuilder::new();
        append.append_i32(data);
        engine
            .append_inplace_gathered(
                AppendTarget::Path(&ObjectPathBuf::parse("d")),
                &append,
                u8::MAX,
            )
            .unwrap();
        let before = engine.disk_bytes().unwrap();
        drop(engine);
        assert_eq!(fs::read(&path).unwrap(), before);
    });
    let mut saw_uninitialized = false;
    workload.replay(&directory.path().join("prefix.h5"), &hole, |point, _, _| {
        let storage = super::IndexStorage::read(&workload.prefix(point));
        saw_uninitialized |= storage.paged.iter().any(|block| !block.spare().is_empty());
    });
    assert!(saw_uninitialized);
    super::IndexStorage::read(&workload.final_bytes).assert_paths();
}

/// Stored bytes and their complete allocation for an unlinked raw payload.
struct Payload {
    fact: Fact,
    bytes: Vec<u8>,
}

impl Payload {
    fn find(image: &[u8], bytes: Vec<u8>) -> Self {
        let positions = image
            .windows(bytes.len())
            .enumerate()
            .filter_map(|(at, stored)| (stored == bytes).then_some(at))
            .collect::<Vec<_>>();
        assert_eq!(positions.len(), 1);
        Self {
            fact: Fact::new(
                AllocationKind::Payload,
                u64::try_from(positions[0]).unwrap(),
                u64::try_from(bytes.len()).unwrap(),
            ),
            bytes,
        }
    }

    fn assert_owned(&self, image: &[u8], eof: u64) {
        self.assert_bytes(image);
        assert!(self.fact.range.end() <= eof);
    }

    fn assert_bytes(&self, image: &[u8]) {
        let Self { fact, bytes } = self;
        assert_eq!(
            self.bytes_in(image),
            Some(bytes.as_slice()),
            "payload {:?}",
            fact
        );
    }

    fn bytes_in<'a>(&self, image: &'a [u8]) -> Option<&'a [u8]> {
        image.get(
            usize::try_from(self.fact.range.start()).unwrap()
                ..usize::try_from(self.fact.range.end()).unwrap(),
        )
    }

    /// Returns whether the write contains the payload's first 16 bytes at its stored offset.
    fn written_by(&self, operation: &DiskOp) -> bool {
        let Self {
            fact,
            bytes: payload,
        } = self;
        let DiskOp::Write { offset, bytes } = operation else {
            return false;
        };
        fact.range
            .start()
            .checked_sub(*offset)
            .and_then(|local| usize::try_from(local).ok())
            .is_some_and(|local| bytes.get(local..local + 16) == Some(&payload[..16]))
    }
}

/// Persisted sections, manager roles, policy, and published EOF for a fixture.
struct Space {
    info: FileSpaceInfo,
    free: Coverage,
    sections: Vec<Interval>,
    roles: Vec<FreeSpaceManagerRole>,
    eof: u64,
}

impl Space {
    fn read(bytes: &[u8]) -> Self {
        let reader = File::from_bytes(bytes.to_vec()).unwrap();
        let info = reader.file_space_info().unwrap().unwrap();
        let persisted = reader.persisted_free_sections().unwrap().unwrap();
        let sections = persisted
            .free_sections()
            .sections()
            .iter()
            .map(|section| Interval::new(section.address().get(), section.length().get()).unwrap())
            .collect::<Vec<_>>();
        let roles = persisted
            .managers()
            .iter()
            .map(|manager| manager.role())
            .collect();
        Self {
            info,
            free: Coverage::new(sections.iter().copied()),
            sections,
            roles,
            eof: reader.superblock().eof_address,
        }
    }
}

fn settings(strategy: FileSpaceStrategy, threshold: u64) -> FileSpaceSettings {
    FileSpaceSettings::new(strategy)
        .with_persistence(true)
        .with_threshold(threshold)
}

/// The scratch payload size for threshold cases, including a partial PAGE allocation.
const SCRATCH: usize = 8192 + 113;
