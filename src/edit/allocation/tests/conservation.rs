//! Checks persisted ownership across completed full-chunk append prefixes.
//!
//! Fixtures use persistent FSM_AGGR or PAGE storage with threshold one, a zero base address, and
//! eight-byte addresses and lengths. Tests derive complete live allocations from parsed standard
//! metadata and inspect stored EA pointers before final dataspace growth. Native crosschecks compare
//! normalized free ranges and chunk records with libhdf5.

use std::fs;
use std::path::Path;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::allocation_oracle;
use test_util::allocation_oracle::Allocation;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::AllocationState;
use test_util::allocation_oracle::Context;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::allocation_oracle::PublicationState;
use test_util::allocation_oracle::Violation;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;

use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::access_mode::AccessMode;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_index_inplace::Store;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::WriteEngine;
use crate::edit::context::FileContext;
use crate::edit::object_header;
use crate::extensible_array;
use crate::file_lock::FileLocking;
use crate::free_sections::PersistedManagerSet;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::BytesSource;
use crate::source::MetadataCacheConfig;
use crate::source::RangeReader;

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::borrowed(WritePath::Borrowed)]
#[case::owned(WritePath::Owned)]
fn every_append_write_preserves_complete_ownership_and_unused_holes(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("append.h5");
    let hole = pure_fixture(&path, paged, deferred, filtered);
    let workload = Workload::record(&path, writer, 5);
    let replay = directory.path().join("interrupted.h5");
    let mut saw_provisional = false;
    let mut saw_orphan = false;
    let mut saw_references = false;
    workload.replay(&replay, &hole, |point, observation, allocations| {
        saw_provisional |= allocations.iter().any(|fact| {
            fact.state() == AllocationState::Provisional
                && !Coverage::new([fact.complete()])
                    .intersection(&hole)
                    .is_empty()
                && Coverage::new([fact.complete()])
                    .difference(&observation.free)
                    .is_empty()
        });
        saw_orphan |= allocations
            .iter()
            .any(|fact| fact.state() == AllocationState::Orphaned);
        saw_references |= observation.values.len() > workload.initial_values;
        assert_eq!(fs::read(&replay).unwrap(), workload.prefix(point));
    });
    assert!(
        saw_provisional,
        "{writer:?}/{paged}/{deferred}/{filtered}: no provisional hole allocation"
    );
    assert!(
        saw_orphan,
        "{writer:?}/{paged}/{deferred}/{filtered}: no owned unlinked allocation"
    );
    assert!(saw_references);
}

#[rstest]
#[case::flat_existing(false, false)]
#[case::flat_deferred(false, true)]
#[case::page_existing(true, false)]
#[case::page_deferred(true, true)]
fn repeated_interrupted_appends_reuse_the_surviving_unique_interval(
    #[case] paged: bool,
    #[case] deferred: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cycles.h5");
    let hole = pure_fixture(&path, paged, deferred, false);
    let mut histories = Vec::new();
    for cycle in 0..3 {
        let workload = Workload::record(&path, WritePath::Immediate, 2);
        let point = workload
            .snapshots
            .iter()
            .enumerate()
            .find_map(|(point, snapshot)| {
                (point > 0
                    && snapshot.values.len() == workload.initial_values
                    && workload.facts(point, snapshot).iter().any(|allocation| {
                        allocation.kind() == AllocationKind::Payload
                            && allocation.state() == AllocationState::Orphaned
                    }))
                .then_some(point)
            })
            .unwrap();
        let interrupted = workload.prefix(point);
        fs::write(&path, &interrupted).unwrap();
        let observation = Observation::read(&interrupted);
        histories.extend(workload.published_history(point));
        observation.inspect("repeated interruption", cycle, &hole, &histories);
        let before = fs::read(&path).unwrap();
        let reader = File::open(&path).unwrap();
        assert_eq!(
            reader.dataset("d").unwrap().read_i32().unwrap(),
            observation.values
        );
        drop(reader);
        assert_eq!(fs::read(&path).unwrap(), before);
    }
    force_unique_reuse(&path, &hole, &histories, paged);
}

#[rstest]
#[case::missing_unused(Control::MissingUnused)]
#[case::live_payload(Control::Payload)]
#[case::live_index(Control::Index)]
#[case::live_manager(Control::Manager)]
fn readable_file_controls_reject_lost_unused_or_free_live_storage(#[case] control: Control) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("control.h5");
    let hole = pure_fixture(&path, false, false, false);
    let workload = Workload::record(&path, WritePath::Immediate, 2);
    let before = Observation::read(&workload.final_bytes);
    let mut sections = before.free.clone();
    let changed = match control {
        Control::MissingUnused => {
            let unused = hole.intersection(&before.free).intervals()[0];
            let removed = Interval::new(unused.start() + 32, 32).unwrap();
            sections = sections.difference(&Coverage::new([removed]));
            removed
        }
        Control::Payload | Control::Index | Control::Manager => {
            let kind = match control {
                Control::Payload => AllocationKind::Payload,
                Control::Index => AllocationKind::Index,
                Control::Manager => AllocationKind::Manager,
                Control::MissingUnused => panic!("missing-unused control has no live kind"),
            };
            let address = if matches!(control, Control::Manager) {
                u64::try_from(workload.final_bytes.len()).unwrap()
            } else {
                before
                    .live
                    .iter()
                    .find(|fact| {
                        fact.kind == kind
                            && (kind != AllocationKind::Index
                                || workload.final_bytes
                                    [usize::try_from(fact.range.start()).unwrap()..]
                                    .starts_with(test_util::extensible_array::SIGNATURE))
                    })
                    .unwrap()
                    .range
                    .start()
            };
            let range = Interval::new(address, 8).unwrap();
            sections = Coverage::new(sections.intervals().iter().copied().chain([range]));
            range
        }
    };
    let damaged = replace_free_sections(&workload.final_bytes, &sections);
    fs::write(&path, &damaged).unwrap();
    let observation = Observation::read(&damaged);
    assert_eq!(observation.values, before.values);
    let allocations = workload.facts(workload.operations.len(), &observation);
    let error = allocation_oracle::inspect(
        Context {
            case: "readable file control",
            interruption: workload.operations.len(),
            publication: PublicationState::ReferencesPublished,
        },
        &hole,
        &observation.free,
        &observation.reachable(),
        &allocations,
    )
    .unwrap_err();
    let conflicting = match (control, error.violation) {
        (Control::MissingUnused, Violation::UnusedSpaceMissing { missing }) => missing,
        (
            Control::Payload | Control::Index | Control::Manager,
            Violation::LiveSpaceAdvertisedFree { conflicting },
        ) => conflicting,
        (_, violation) => panic!("{control:?}: unexpected {violation:?}"),
    };
    assert_eq!(conflicting, Coverage::new([changed]));
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        before.values
    );
    drop(file);
    assert_eq!(fs::read(&path).unwrap(), damaged);
}

/// The appender form exercised by a recorded workload.
#[derive(Clone, Copy, Debug)]
pub(super) enum WritePath {
    Borrowed,
    Immediate,
    Owned,
}

impl WritePath {
    fn append(self, path: &Path, values: &[i32]) {
        let file = File::open_rw_with_options(
            path,
            FileAccessProperties::new()
                .with_sync_policy(SyncPolicy::OnClose)
                .with_memory_strategy(MemoryStrategy::Bounded),
        )
        .unwrap();
        let mut target = file.dataset("d").unwrap();
        match self {
            Self::Borrowed => {
                let mut writer = target.buffered_appender().unwrap();
                writer.append(values).unwrap();
                writer.finish().unwrap();
            }
            Self::Immediate => target.append(values).unwrap(),
            Self::Owned => {
                let mut writer = target.into_buffered_appender().unwrap();
                writer.append(values).unwrap();
                writer.finish().unwrap();
            }
        }
        file.close().unwrap();
    }
}

/// Recorded byte operations, per-prefix observations, and published allocation histories.
pub(super) struct Workload {
    base: Vec<u8>,
    pub(super) final_bytes: Vec<u8>,
    operations: Vec<DiskOp>,
    snapshots: Vec<Observation>,
    initial_values: usize,
    candidates: Vec<Fact>,
    history: Vec<(usize, Fact)>,
}

impl Workload {
    /// Appends `rounds` complete chunks and records each completed byte operation.
    ///
    /// Parses each prefix, locates stored references, and checks that complete replay reproduces
    /// the finished file and its expected values.
    pub(super) fn record(path: &Path, writer: WritePath, rounds: usize) -> Self {
        let base = fs::read(path).unwrap();
        let initial = Observation::read(&base);
        let initial_values = initial.values.len();
        let values: Vec<_> = (initial_values..initial_values + rounds * CHUNK)
            .map(|value| i32::try_from(value).unwrap())
            .collect();
        disk_log::start();
        writer.append(path, &values);
        let recording = disk_log::finish();
        let operations = recording.operations;
        assert!(!operations.is_empty());
        assert!(!recording.events.is_empty());
        let final_bytes = fs::read(path).unwrap();
        let mut image = base.clone();
        let mut snapshots = vec![initial];
        let mut history = Vec::new();
        for (point, operation) in operations.iter().enumerate() {
            operation.apply(&mut image);
            let snapshot = Observation::read(&image);
            for fact in snapshot
                .live
                .iter()
                .filter(|fact| fact.kind == AllocationKind::Manager)
            {
                if !history.iter().any(|(_, prior)| prior == fact) {
                    history.push((point + 1, *fact));
                }
            }
            snapshots.push(snapshot);
        }
        let final_observation = snapshots.last().unwrap();
        let references = final_observation
            .chunks
            .iter()
            .map(|chunk| {
                let encoded = chunk.1.to_le_bytes();
                let fields = final_observation
                    .live
                    .iter()
                    .filter(|fact| final_observation.element_blocks.contains(&fact.range))
                    .flat_map(|parent| {
                        let start = usize::try_from(parent.range.start()).unwrap();
                        let end = usize::try_from(parent.range.end()).unwrap();
                        test_util::bytes::signature_offsets(&final_bytes[start..end], &encoded)
                            .into_iter()
                            .map(move |local| (parent.range, start + local))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    fields.len(),
                    1,
                    "chunk {chunk:?} must have one stored index pointer: {fields:?}"
                );
                (chunk.clone(), fields[0])
            })
            .collect::<Vec<_>>();
        let mut prefix = base.clone();
        for (point, snapshot) in snapshots.iter_mut().enumerate() {
            if point > 0 {
                operations[point - 1].apply(&mut prefix);
            }
            for (chunk, (parent, field)) in &references {
                if snapshot.live.iter().any(|fact| fact.range == *parent)
                    && prefix.get(*field..*field + 8) == Some(chunk.1.to_le_bytes().as_slice())
                    && !snapshot.chunks.contains(chunk)
                {
                    let fact = Fact::new(AllocationKind::Payload, chunk.1, chunk.2);
                    assert!(fact.range.end() <= u64::try_from(prefix.len()).unwrap());
                    assert!(fact.range.end() <= snapshot.eof);
                    snapshot.live.push(fact);
                    snapshot.chunks.push(chunk.clone());
                }
            }
            snapshot.chunks.sort_unstable();
            let mut raw_live = snapshot
                .live
                .iter()
                .map(|fact| fact.range)
                .collect::<Vec<_>>();
            assert_disjoint(&mut raw_live, "complete reachable pointer allocations");
        }
        assert_eq!(image, final_bytes, "full-log reconstruction differs");
        assert_eq!(
            snapshots.last().unwrap().values,
            (0..initial_values + rounds * CHUNK)
                .map(|value| i32::try_from(value).unwrap())
                .collect::<Vec<_>>()
        );
        let candidates = snapshots
            .last()
            .unwrap()
            .live
            .iter()
            .filter(|fact| {
                matches!(fact.kind, AllocationKind::Index | AllocationKind::Payload)
                    && !snapshots[0].live.contains(fact)
            })
            .copied()
            .collect::<Vec<_>>();
        assert!(
            candidates
                .iter()
                .any(|fact| fact.kind == AllocationKind::Payload)
        );
        Self {
            base,
            final_bytes,
            operations,
            snapshots,
            initial_values,
            candidates,
            history,
        }
    }

    /// Checks every byte-operation prefix and calls `observe` with its allocation facts.
    ///
    /// Writes reconstructed bytes directly, without opening an editing session on the replay.
    pub(super) fn replay(
        &self,
        path: &Path,
        hole: &Coverage,
        mut observe: impl FnMut(usize, &Observation, &[Allocation]),
    ) {
        let mut image = self.base.clone();
        let mut previous_values = self.initial_values;
        for point in 0..=self.operations.len() {
            if point > 0 {
                self.operations[point - 1].apply(&mut image);
            }
            fs::write(path, &image).unwrap();
            let snapshot = &self.snapshots[point];
            assert!(snapshot.values.len() >= previous_values);
            previous_values = snapshot.values.len();
            let allocations = self.facts(point, snapshot);
            snapshot.inspect("append prefix", point, hole, &self.published_history(point));
            allocation_oracle::inspect(
                Context {
                    case: "append allocation states",
                    interruption: point,
                    publication: snapshot.state(),
                },
                hole,
                &snapshot.free,
                &snapshot.reachable(),
                &allocations,
            )
            .unwrap();
            observe(point, snapshot, &allocations);
            assert_eq!(
                fs::read(path).unwrap(),
                image,
                "inspection mutates prefix {point}"
            );
        }
        assert_eq!(image, self.final_bytes);
    }

    /// Reconstructs the file after `point` completed byte operations.
    ///
    /// # Panics
    ///
    /// Panics if `point` exceeds the recorded operation count.
    pub(super) fn prefix(&self, point: usize) -> Vec<u8> {
        let mut image = self.base.clone();
        for operation in &self.operations[..point] {
            operation.apply(&mut image);
        }
        image
    }

    fn facts(&self, point: usize, observation: &Observation) -> Vec<Allocation> {
        let mut facts = observation
            .live
            .iter()
            .map(|fact| fact.allocation(AllocationState::Linked))
            .collect::<Vec<_>>();
        for candidate in &self.candidates {
            if observation.live.contains(candidate) {
                continue;
            }
            let coverage = Coverage::new([candidate.range]);
            let prepared = self.operations[..point]
                .iter()
                .any(|operation| match operation {
                    DiskOp::Write { offset, bytes } => !Coverage::new([Interval::new(
                        *offset,
                        u64::try_from(bytes.len()).unwrap(),
                    )
                    .unwrap()])
                    .intersection(&coverage)
                    .is_empty(),
                    DiskOp::SetLen(_) => false,
                });
            if !prepared {
                continue;
            }
            let state = if coverage.intersection(&observation.free).is_empty()
                && candidate.range.end() <= observation.eof
            {
                AllocationState::Orphaned
            } else {
                AllocationState::Provisional
            };
            facts.push(candidate.allocation(state));
        }
        for (_, fact) in self
            .history
            .iter()
            .filter(|(published, _)| *published <= point)
        {
            if observation.live.contains(fact) {
                continue;
            }
            let coverage = Coverage::new([fact.range]);
            if coverage.intersection(&observation.free).is_empty()
                && coverage.intersection(&observation.reachable()).is_empty()
                && !facts.iter().any(|other| {
                    !coverage
                        .intersection(&Coverage::new([other.complete()]))
                        .is_empty()
                })
            {
                facts.push(fact.allocation(AllocationState::Orphaned));
            }
        }
        facts
    }

    fn published_history(&self, point: usize) -> Vec<Fact> {
        self.history
            .iter()
            .filter(|(published, _)| *published <= point)
            .map(|(_, fact)| *fact)
            .chain(
                self.candidates
                    .iter()
                    .filter(|candidate| {
                        let coverage = Coverage::new([candidate.range]);
                        (0..=point).any(|index| {
                            coverage
                                .intersection(&self.snapshots[index].free)
                                .is_empty()
                                && candidate.range.end() <= self.snapshots[index].eof
                        })
                    })
                    .copied(),
            )
            .collect()
    }
}

/// A complete allocation range and its metadata parse range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Fact {
    kind: AllocationKind,
    range: Interval,
    parsed: Interval,
}

impl Fact {
    fn new(kind: AllocationKind, address: u64, length: u64) -> Self {
        Self {
            kind,
            range: Interval::new(address, length).unwrap(),
            parsed: Interval::new(address, length).unwrap(),
        }
    }

    fn allocation(self, state: AllocationState) -> Allocation {
        match self.kind {
            AllocationKind::Index => Allocation::index(self.range, self.parsed, state).unwrap(),
            AllocationKind::Manager => Allocation::manager(self.range, self.parsed, state).unwrap(),
            AllocationKind::Payload => {
                Allocation::payload(self.range.start(), self.range.length(), state).unwrap()
            }
        }
    }
}

/// Reachable allocations, persisted free ranges, and visible fixture values.
pub(super) struct Observation {
    pub(super) free: Coverage,
    large_free: Coverage,
    page_size: Option<u64>,
    pub(super) live: Vec<Fact>,
    pub(super) values: Vec<i32>,
    eof: u64,
    index_defined: bool,
    element_blocks: Vec<Interval>,
    pub(super) chunks: Vec<(Vec<u64>, u64, u64, u32)>,
}

impl Observation {
    /// Parses persisted managers, reachable metadata, and visible fixture values.
    ///
    /// # Panics
    ///
    /// Panics if parsing or fixture-value checks fail, free sections overlap, or a complete
    /// allocation exceeds published EOF or physical file length. Also panics for a nonzero base
    /// address, widths other than eight bytes, disabled persistence, or a threshold other than one.
    pub(super) fn read(bytes: &[u8]) -> Self {
        let file = File::from_bytes(bytes.to_vec()).unwrap();
        let located = crate::superblock::read(bytes).unwrap();
        let context = FileContext::new(located).unwrap();
        assert!(context.superblock().eof_address <= u64::try_from(bytes.len()).unwrap());
        assert_eq!(context.superblock().base_address.get(), 0);
        let source = BytesSource::new(bytes);
        let addresses = context.addresses();
        let widths = context.widths();
        assert_eq!((widths.offsets.get(), widths.lengths.get()), (8, 8));
        let (extension, info) = WriteEngine::extension_fsinfo(&source, &context)
            .unwrap()
            .unwrap();
        assert!(info.persist);
        assert_eq!(info.threshold, 1);
        let (report, metadata) = PersistedManagerSet::read(
            &source,
            widths,
            addresses.effective_base(),
            context.superblock().eof_address,
            &info,
        )
        .unwrap()
        .into_parts();
        let mut raw_free = report
            .free_sections()
            .sections()
            .iter()
            .map(|section| Interval::new(section.address(), section.length().get()).unwrap())
            .collect::<Vec<_>>();
        assert_disjoint(&mut raw_free, "raw free sections");
        let free = Coverage::new(raw_free);
        let large_free = Coverage::new(
            report
                .managers()
                .iter()
                .filter(|manager| {
                    matches!(
                        manager.role(),
                        crate::free_sections::FreeSpaceManagerRole::FsmAggr
                            | crate::free_sections::FreeSpaceManagerRole::PageLargeGeneric
                    )
                })
                .flat_map(|manager| {
                    manager.free_sections().sections().iter().map(|section| {
                        Interval::new(section.address(), section.length().get()).unwrap()
                    })
                }),
        );
        let page_size = (info.strategy == FileSpaceStrategy::Page).then_some(info.page_size.get());

        let mut live = metadata
            .iter()
            .map(|range| Fact::new(AllocationKind::Manager, range.start().get(), range.length()))
            .collect::<Vec<_>>();
        for address in &info.manager_addrs {
            let DecodedStoredAddress::Defined(address) = address else {
                continue;
            };
            let at = usize::try_from(addresses.absolute(*address).unwrap()).unwrap();
            let length = usize::try_from(format::free_space_manager_header_len(widths)).unwrap();
            let header =
                format::FreeSpaceManagerHeader::parse(widths, &bytes[at..at + length]).unwrap();
            if !header.section_list_addr().is_undefined(widths.offsets) {
                let list_at = addresses.absolute(header.section_list_addr()).unwrap();
                let allocation = live
                    .iter_mut()
                    .find(|fact| fact.range.start() == list_at)
                    .unwrap();
                allocation.parsed = Interval::new(list_at, header.section_list_used()).unwrap();
                assert!(allocation.parsed.end() <= allocation.range.end());
            }
        }
        live.extend(
            object_header::read_oh_chunks(&source, extension, addresses)
                .unwrap()
                .into_iter()
                .map(|chunk| Fact::new(AllocationKind::Manager, chunk.span.0, chunk.span.1)),
        );
        let root = addresses
            .absolute(context.superblock().root_group_address)
            .unwrap();
        let root_header = ObjectHeader::parse(
            bytes,
            AccessMode::ReadOnly,
            usize::try_from(root).unwrap(),
            widths.offsets.get(),
            widths.lengths.get(),
        )
        .unwrap();
        let mut headers = vec![root];
        headers.extend(
            root_header
                .messages
                .iter()
                .filter(|message| message.msg_type == MessageType::LINK)
                .filter_map(|message| {
                    let link =
                        format::LinkMessage::parse_with_offset_width(&message.data, widths.offsets)
                            .unwrap();
                    match link.link_target {
                        format::LinkTarget::Hard {
                            object_header_address,
                        } => Some(addresses.absolute(object_header_address).unwrap()),
                        _ => None,
                    }
                }),
        );
        live.push(Fact::new(
            AllocationKind::Index,
            0,
            u64::try_from(
                format::serialize_superblock(context.superblock())
                    .unwrap()
                    .len(),
            )
            .unwrap(),
        ));
        let mut index_defined = false;
        let mut element_blocks = Vec::new();
        for header in headers {
            live.extend(
                object_header::read_oh_chunks(&source, header, addresses)
                    .unwrap()
                    .into_iter()
                    .map(|chunk| Fact::new(AllocationKind::Index, chunk.span.0, chunk.span.1)),
            );
            let object = ObjectHeader::parse(
                bytes,
                AccessMode::ReadOnly,
                usize::try_from(header).unwrap(),
                widths.offsets.get(),
                widths.lengths.get(),
            )
            .unwrap();
            for message in object
                .messages
                .iter()
                .filter(|message| message.msg_type == MessageType::DATA_LAYOUT)
            {
                match DataLayout::parse(&message.data, widths.offsets.get(), widths.lengths.get())
                    .unwrap()
                {
                    DataLayout::Contiguous {
                        address: Some(address),
                        size,
                    } => live.push(Fact::new(
                        AllocationKind::Payload,
                        addresses.absolute(address).unwrap(),
                        size,
                    )),
                    DataLayout::Chunked {
                        index:
                            ChunkIndexLayout::ExtensibleArray {
                                address: Some(address),
                                ..
                            },
                        ..
                    } => {
                        index_defined = true;
                        element_blocks.extend(index_element_blocks(&source, widths, address));
                        live.extend(
                            extensible_array::index_spans(&source, widths, address)
                                .unwrap()
                                .into_iter()
                                .map(|(address, length)| {
                                    Fact::new(AllocationKind::Index, address, length)
                                }),
                        );
                    }
                    _ => {}
                }
            }
        }
        let dataset = file.dataset("d").unwrap();
        let values = dataset.read_i32().unwrap();
        assert_eq!(
            values,
            (0..values.len())
                .map(|value| i32::try_from(value).unwrap())
                .collect::<Vec<_>>()
        );
        let mut chunks = dataset
            .chunks()
            .unwrap()
            .into_iter()
            .map(|chunk| {
                let filter_mask = chunk.filter_mask().raw_bits();
                (chunk.offset, chunk.address, chunk.storage_size, filter_mask)
            })
            .collect::<Vec<_>>();
        chunks.sort_unstable();
        live.extend(
            chunks
                .iter()
                .map(|(_, address, size, _)| Fact::new(AllocationKind::Payload, *address, *size)),
        );
        assert_eq!(
            file.dataset("ceiling").unwrap().read_u8().unwrap(),
            [9; CEILING]
        );
        let mut raw_live = live.iter().map(|fact| fact.range).collect::<Vec<_>>();
        assert_disjoint(&mut raw_live, "complete live allocations");
        for range in raw_live {
            assert!(
                range.end() <= context.superblock().eof_address,
                "complete allocation {range:?} exceeds published EOF {}",
                context.superblock().eof_address
            );
            assert!(
                range.end() <= u64::try_from(bytes.len()).unwrap(),
                "complete allocation {range:?} exceeds physical length {}",
                bytes.len()
            );
        }
        Self {
            free,
            large_free,
            page_size,
            live,
            values,
            eof: context.superblock().eof_address,
            index_defined,
            element_blocks,
            chunks,
        }
    }

    fn inspect(&self, case: &str, point: usize, hole: &Coverage, history: &[Fact]) {
        let mut facts = self
            .live
            .iter()
            .map(|fact| fact.allocation(AllocationState::Linked))
            .collect::<Vec<_>>();
        for fact in history {
            let coverage = Coverage::new([fact.range]);
            if coverage.intersection(&self.free).is_empty()
                && coverage.intersection(&self.reachable()).is_empty()
                && !facts.iter().any(|other| {
                    !coverage
                        .intersection(&Coverage::new([other.complete()]))
                        .is_empty()
                })
            {
                facts.push(fact.allocation(AllocationState::Orphaned));
            }
        }
        allocation_oracle::inspect(
            Context {
                case,
                interruption: point,
                publication: self.state(),
            },
            hole,
            &self.free,
            &self.reachable(),
            &facts,
        )
        .unwrap();
    }

    fn reachable(&self) -> Coverage {
        Coverage::new(self.live.iter().map(|fact| fact.range))
    }

    fn unique_large_interval(&self, hole: &Coverage) -> (Interval, u64) {
        let candidates = self.large_free.intersection(hole);
        let available = candidates
            .intervals()
            .iter()
            .map(|interval| {
                if let Some(page) = self.page_size {
                    let start = interval.start().div_ceil(page) * page;
                    let end = interval.end() / page * page;
                    Interval::new(start, end.saturating_sub(start)).unwrap()
                } else {
                    *interval
                }
            })
            .collect::<Vec<_>>();
        let candidate = *available
            .iter()
            .max_by_key(|interval| interval.length())
            .unwrap();
        let length = candidate.length() - self.page_size.unwrap_or(128);
        assert!(length > 8192);
        assert!(
            self.large_free
                .intervals()
                .iter()
                .all(|other| !Coverage::new([candidate])
                    .intersection(&Coverage::new([*other]))
                    .is_empty()
                    || other.length() < length),
            "another eligible interval fits {length}: {:?}",
            self.large_free
        );
        (candidate, length)
    }

    fn state(&self) -> PublicationState {
        if !self.index_defined {
            PublicationState::BeforeOwnership
        } else if self.values.is_empty() {
            PublicationState::OwnershipPublished
        } else {
            PublicationState::ReferencesPublished
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Control {
    Index,
    Manager,
    MissingUnused,
    Payload,
}

/// Writes a fixture with an interior free hole and live contiguous storage above it.
///
/// Selects PAGE when `paged` is `true`, and FSM_AGGR otherwise. Sets the layout's EA header address
/// to undefined when `deferred` is `true`, and enables Fletcher32 when `filtered` is `true`.
pub(super) fn pure_fixture(path: &Path, paged: bool, deferred: bool, filtered: bool) -> Coverage {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(if paged {
            FileSpaceStrategy::Page
        } else {
            FileSpaceStrategy::FsmAggr
        })
        .with_persistence(true)
        .with_threshold(1),
    );
    let target = if deferred {
        builder
            .create_dataset("d")
            .with_unallocated_storage(crate::make_i32_type(), &[0])
    } else {
        builder
            .create_dataset("d")
            .with_i32_data(&(0..i32::try_from(CHUNK).unwrap()).collect::<Vec<_>>())
    };
    target
        .with_chunks(&[u64::try_from(CHUNK).unwrap()])
        .with_maxshape(&[MaxExtent::Unlimited]);
    if filtered {
        target.with_fletcher32();
    }
    builder
        .create_dataset("scratch")
        .with_u8_data(&vec![7; HOLE]);
    builder
        .create_dataset("ceiling")
        .with_u8_data(&[9; CEILING]);
    builder.write(path).unwrap();
    let file = File::open(path).unwrap();
    let DataLayout::Contiguous {
        address: Some(address),
        size,
    } = file.dataset("scratch").unwrap().data_layout().unwrap()
    else {
        panic!("scratch must have contiguous storage")
    };
    let hole = Coverage::new([Interval::new(address.get(), size).unwrap()]);
    let DataLayout::Contiguous {
        address: Some(ceiling),
        ..
    } = file.dataset("ceiling").unwrap().data_layout().unwrap()
    else {
        panic!("ceiling must have contiguous storage")
    };
    assert!(ceiling.get() >= address.get() + size);
    drop(file);
    let mut editor = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    editor.set_sync_policy(SyncPolicy::OnClose);
    editor.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    editor.commit().unwrap();
    editor.finalize_persist().unwrap();
    drop(editor);
    if deferred {
        let mut bytes = fs::read(path).unwrap();
        let (header, mut layout) = test_util::bytes::signature_offsets(&bytes, v2::SIGNATURE)
            .into_iter()
            .find_map(|at| {
                ObjectHeader::parse(&bytes, AccessMode::ReadOnly, at, 8, 8)
                    .unwrap()
                    .messages
                    .into_iter()
                    .find(|message| {
                        message.msg_type == MessageType::DATA_LAYOUT
                            && matches!(
                                DataLayout::parse(&message.data, 8, 8).unwrap(),
                                DataLayout::Chunked { .. }
                            )
                    })
                    .map(|message| (at, message.data))
            })
            .unwrap();
        let (_, pointer) = DataLayout::parse_with_ea_address(
            &layout,
            crate::width::FormatWidths::from_sizes(8, 8).unwrap(),
        )
        .unwrap();
        layout[pointer.unwrap()].fill(u8::MAX);
        v2::replace_message(&mut bytes, header, FixtureMessage::DATA_LAYOUT, &layout);
        fs::write(path, bytes).unwrap();
    }
    let snapshot = Observation::read(&fs::read(path).unwrap());
    assert_eq!(snapshot.index_defined, !deferred);
    assert!(hole.difference(&snapshot.free).is_empty());
    hole
}

fn force_unique_reuse(path: &Path, hole: &Coverage, history: &[Fact], paged: bool) {
    let snapshot = Observation::read(&fs::read(path).unwrap());
    let (candidate, length) = snapshot.unique_large_interval(hole);
    assert_eq!(snapshot.page_size.is_some(), paged);
    let mut editor = WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        crate::PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    editor.set_sync_policy(SyncPolicy::OnClose);
    let payload = vec![43; usize::try_from(length).unwrap()];
    let mut store = editor.store();
    let issued = store
        .allocate(&payload, hdf5_pure_space::__private::AllocationClass::Raw)
        .unwrap();
    let actual = issued.range();
    store
        .publish_allocations()
        .unwrap()
        .confirm(issued)
        .unwrap();
    let published = fs::read(path).unwrap();
    drop(editor);
    assert_eq!(fs::read(path).unwrap(), published);
    let starts = published
        .windows(payload.len())
        .enumerate()
        .filter_map(|(at, bytes)| (bytes == payload).then_some(at))
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 1);
    assert_eq!(u64::try_from(starts[0]).unwrap(), actual.start().get());
    assert_eq!(actual.start().get(), candidate.start());
    let mut history = history.to_vec();
    history.push(Fact::new(
        AllocationKind::Payload,
        actual.start().get(),
        length,
    ));
    Observation::read(&published).inspect("unique interval reused", 0, hole, &history);
}

fn replace_free_sections(original: &[u8], coverage: &Coverage) -> Vec<u8> {
    let mut bytes = original.to_vec();
    let located = crate::superblock::read(original).unwrap();
    let mut superblock = located.superblock.clone();
    let context = FileContext::new(located).unwrap();
    let (extension, mut info) =
        WriteEngine::extension_fsinfo(&BytesSource::new(original), &context)
            .unwrap()
            .unwrap();
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    let header_at = u64::try_from(bytes.len()).unwrap();
    let list_at = header_at + format::free_space_manager_header_len(context.widths());
    let sections = coverage
        .intervals()
        .iter()
        .map(|interval| format::FreeSection {
            addr: StoredAddress::new(interval.start()),
            size: interval.length(),
        })
        .collect::<Vec<_>>();
    let (header, list) = format::serialize_free_space_manager(
        context.widths(),
        &sections,
        StoredAddress::new(header_at),
        StoredAddress::new(list_at),
        format::SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    bytes.extend(header);
    bytes.extend(list);
    info.manager_addrs.fill(DecodedStoredAddress::Undefined);
    info.manager_addrs[0] = DecodedStoredAddress::Defined(StoredAddress::new(header_at));
    let encoded = format::serialize_file_space_info_v1(context.widths(), &info).unwrap();
    v2::replace_message(
        &mut bytes,
        usize::try_from(extension).unwrap(),
        FixtureMessage::FILE_SPACE_INFO,
        &encoded,
    );
    superblock.eof_address = u64::try_from(bytes.len()).unwrap();
    let encoded = format::serialize_superblock(&superblock).unwrap();
    bytes[..encoded.len()].copy_from_slice(&encoded);
    bytes
}

fn index_element_blocks(
    source: &impl RangeReader,
    widths: crate::width::FormatWidths,
    address: StoredAddress,
) -> Vec<Interval> {
    let header = extensible_array::read_header(source, widths, address).unwrap();
    let reference = header.index_block().unwrap();
    let range = reference.parse_range();
    let bytes = source.read_range(range).unwrap();
    let index = reference
        .parse(FileRegion::new(range, &bytes).unwrap())
        .unwrap();
    let mut blocks = vec![Interval::new(range.start().get(), range.length()).unwrap()];
    blocks.extend(
        index
            .data_blocks()
            .map(|block| parse_data_block(source, block.unwrap())),
    );
    for reference in index.super_blocks() {
        let reference = reference.unwrap();
        let range = reference.parse_range();
        let bytes = source.read_range(range).unwrap();
        let block = reference
            .parse(FileRegion::new(range, &bytes).unwrap())
            .unwrap();
        blocks.extend(
            block
                .data_blocks()
                .map(|block| parse_data_block(source, block.unwrap())),
        );
    }
    blocks
}

fn parse_data_block(
    source: &impl RangeReader,
    reference: format::ExtensibleArrayDataBlockRef<'_>,
) -> Interval {
    let range = reference.parse_range();
    let bytes = source.read_range(range).unwrap();
    let parsed = reference
        .parse(FileRegion::new(range, &bytes).unwrap())
        .unwrap();
    if let format::ExtensibleArrayDataBlock::Pages(pages) = parsed {
        for page in pages.pages() {
            let page = page.unwrap();
            let range = page.parse_range();
            let bytes = source.read_range(range).unwrap();
            page.parse(FileRegion::new(range, &bytes).unwrap()).unwrap();
        }
    }
    let range = reference.range();
    Interval::new(range.start().get(), range.length()).unwrap()
}

fn assert_disjoint(intervals: &mut [Interval], label: &str) {
    intervals.sort_unstable();
    for adjacent in intervals.windows(2) {
        assert!(
            adjacent[0].end() <= adjacent[1].start(),
            "{label}: overlapping {adjacent:?}"
        );
    }
}

/// The number of [`i32`] elements per fixture chunk.
pub(super) const CHUNK: usize = 1024;
const HOLE: usize = 131_072;
const CEILING: usize = 8192;
