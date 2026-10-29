//! Checks allocation ownership while replacing partial chunks and flushing buffered appends.

use core::cmp::Reverse;
#[cfg(feature = "deflate")]
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use rstest::rstest;
use test_util::allocation_oracle;
use test_util::allocation_oracle::Allocation;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::AllocationState;
use test_util::allocation_oracle::Context;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::allocation_oracle::Violation;
use test_util::disk_log;
use test_util::disk_log::DiskOp;

use crate::BufferedAppender;
use crate::DatasetBuilder;
#[cfg(feature = "deflate")]
use crate::DatasetCreateProperties;
#[cfg(feature = "deflate")]
use crate::DeflateLevel;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::MemoryStrategy;
use crate::OwnedBufferedAppender;
#[cfg(feature = "deflate")]
use crate::PartialEdgeFiltering;
use crate::SyncPolicy;

use super::CHUNK;
use super::Fact;
use super::Observation;

#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::borrowed_manual(WritePath::Borrowed(Flush::Manual))]
#[case::borrowed_terminal(WritePath::Borrowed(Flush::Terminal))]
#[case::owned_manual(WritePath::Owned(Flush::Manual))]
#[case::owned_terminal(WritePath::Owned(Flush::Terminal))]
#[case::borrowed_drop(WritePath::Borrowed(Flush::Drop))]
#[case::owned_drop(WritePath::Owned(Flush::Drop))]
fn transient_tails_conserve_owned_and_unused_intervals(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(Encoding::Plain, Encoding::Fletcher32)] encoding: Encoding,
) {
    run_pure(writer, paged, deferred, encoding);
}

#[cfg(feature = "deflate")]
#[rstest]
#[case::compressed(Encoding::Deflate)]
#[case::raw_edges(Encoding::RawEdges)]
fn compressed_tail_sizes_and_masks_preserve_ownership(
    #[case] encoding: Encoding,
    #[values(
        WritePath::Immediate,
        WritePath::Borrowed(Flush::Manual),
        WritePath::Owned(Flush::Terminal)
    )]
    writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
) {
    run_pure(writer, paged, deferred, encoding);
}

#[rstest]
#[case::borrowed(false)]
#[case::owned(true)]
fn empty_buffers_and_incomplete_element_flush_preserve_unallocated_storage(
    #[case] owned: bool,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.h5");
    let hole = super::pure_fixture(&path, paged, true, false);
    assert_empty_operations(&path, owned, &hole);
}

#[rstest]
#[case::existing(false)]
#[case::deferred(true)]
fn released_tail_storage_requires_persisted_free_coverage(
    #[case] deferred: bool,
    #[values(false, true)] inherited: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("retired-control.h5");
    let hole = super::pure_fixture(&path, false, deferred, true);
    let workload = Workload::record(&path, WritePath::Immediate);
    workload.replay(&path, &hole, |_, _| {});
    let final_snapshot = workload.snapshots.last().unwrap();
    let retired = workload
        .retired()
        .intersection(&final_snapshot.free)
        .intersection(&hole);
    let range = retired.intervals()[0];
    let removed = Interval::new(range.start(), range.length().min(32)).unwrap();
    let damaged = super::replace_free_sections(
        &fs::read(&path).unwrap(),
        &final_snapshot.free.difference(&Coverage::new([removed])),
    );
    fs::write(&path, &damaged).unwrap();
    let observation = Observation::read(&damaged);
    assert_eq!(observation.values, final_snapshot.values);
    let point = workload.operations.len();
    let history = if inherited {
        vec![
            workload
                .chunks
                .iter()
                .find(|chunk| {
                    !Coverage::new([chunk.fact.range])
                        .intersection(&Coverage::new([removed]))
                        .is_empty()
                })
                .unwrap()
                .fact,
        ]
    } else {
        Vec::new()
    };
    let allocations = workload.allocations(point, &observation, &history);
    let failure = allocation_oracle::inspect(
        Context {
            case: "released tail control",
            interruption: point,
            publication: observation.state(),
        },
        &hole,
        &observation.free,
        &observation.reachable(),
        &allocations,
    )
    .unwrap_err();
    let Violation::UnusedSpaceMissing { missing } = failure.violation else {
        panic!("expected missing released storage, got {failure:?}")
    };
    assert_eq!(missing, Coverage::new([removed]));
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        final_snapshot.values
    );
    drop(file);
    assert_eq!(fs::read(&path).unwrap(), damaged);
}

fn run_pure(writer: WritePath, paged: bool, deferred: bool, encoding: Encoding) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tails.h5");
    let hole =
        super::pure_fixture_with(&path, paged, deferred, |target| encoding.configure(target));
    let mut history = Vec::new();
    for _cycle in 0..2 {
        let workload = Workload::record(&path, writer);
        workload.replay_with_history(&path, &hole, &history, |_, _| {});
        workload.assert_paths(encoding);
        workload.assert_reclaimed();
        let point = workload.interruption();
        fs::write(&path, workload.prefix(point)).unwrap();
        let mut current = workload.owned_history(point);
        current.extend(history);
        history = current;
    }
    let resumed = Workload::record(&path, WritePath::Immediate);
    resumed.replay_with_history(&path, &hole, &history, |_, _| {});
    history.extend(resumed.owned_history(resumed.operations.len()));
    super::force_unique_reuse(&path, &hole, &history, paged);
}

/// Chunk filtering for pure and native transient fixtures.
#[derive(Clone, Copy, Debug)]
pub(super) enum Encoding {
    #[cfg(feature = "deflate")]
    /// Deflate at level six.
    Deflate,
    Fletcher32,
    Plain,
    #[cfg(feature = "deflate")]
    /// Deflate at level six, with partial chunks stored without filters.
    RawEdges,
}

impl Encoding {
    fn configure(self, target: &mut DatasetBuilder) {
        match self {
            #[cfg(feature = "deflate")]
            Self::Deflate => {
                target.with_deflate(DeflateLevel::Six);
            }
            Self::Fletcher32 => {
                target.with_fletcher32();
            }
            Self::Plain => {}
            #[cfg(feature = "deflate")]
            Self::RawEdges => {
                target
                    .with_create_properties(
                        DatasetCreateProperties::new()
                            .with_partial_edge_filtering(PartialEdgeFiltering::Skip),
                    )
                    .with_deflate(DeflateLevel::Six);
            }
        }
    }
}

/// The operation used to write a buffered workload's partial tail.
#[derive(Clone, Copy, Debug)]
pub(super) enum Flush {
    /// Writes the tail when the buffer is dropped.
    Drop,
    /// Flushes after each input append, then finishes the buffer.
    Manual,
    /// Finishes the buffer after all input appends.
    Terminal,
}

/// The appender and flush behavior exercised by a transient workload.
#[derive(Clone, Copy, Debug)]
pub(super) enum WritePath {
    Borrowed(Flush),
    Immediate,
    Owned(Flush),
}

impl WritePath {
    fn run(self, path: &Path, mut checkpoint: impl FnMut()) {
        let file = File::open_rw_with_options(
            path,
            FileAccessProperties::new()
                .with_sync_policy(SyncPolicy::OnClose)
                .with_memory_strategy(MemoryStrategy::Bounded),
        )
        .unwrap();
        let initial = Observation::read(&file.disk_bytes().unwrap()).values.len();
        let terminal = match self {
            Self::Immediate => {
                let mut dataset = file.dataset("d").unwrap();
                let mut start = initial;
                for length in APPENDS {
                    dataset.append(&values(start, length)).unwrap();
                    start += length;
                    checkpoint();
                }
                None
            }
            Self::Borrowed(flush) => {
                let mut dataset = file.dataset("d").unwrap();
                Buffer::Borrowed(dataset.buffered_appender().unwrap()).run(
                    initial,
                    flush,
                    &mut checkpoint,
                );
                Some((BufferKind::Borrowed, flush))
            }
            Self::Owned(flush) => {
                let dataset = file.dataset("d").unwrap();
                Buffer::Owned(dataset.into_buffered_appender().unwrap()).run(
                    initial,
                    flush,
                    &mut checkpoint,
                );
                Some((BufferKind::Owned, flush))
            }
        };
        file.close().unwrap();
        checkpoint();
        if let Some((kind, terminal @ (Flush::Terminal | Flush::Drop))) = terminal {
            let file = File::open_rw_with_options(
                path,
                FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
            )
            .unwrap();
            let mut dataset = file.dataset("d").unwrap();
            let start = Observation::read(&file.disk_bytes().unwrap()).values.len();
            let mut buffer = match kind {
                BufferKind::Borrowed => Buffer::Borrowed(dataset.buffered_appender().unwrap()),
                BufferKind::Owned => Buffer::Owned(dataset.into_buffered_appender().unwrap()),
            };
            buffer.append(&values(start, CHUNK + 17));
            checkpoint();
            if matches!(terminal, Flush::Drop) {
                drop(buffer);
            } else {
                buffer.finish();
            }
            checkpoint();
            file.close().unwrap();
            checkpoint();
        }
    }

    fn final_elements(self, initial: usize) -> usize {
        initial
            + APPENDS.iter().sum::<usize>()
            + if matches!(
                self,
                Self::Borrowed(Flush::Terminal | Flush::Drop)
                    | Self::Owned(Flush::Terminal | Flush::Drop)
            ) {
                CHUNK + 17
            } else {
                0
            }
    }
}

#[derive(Clone, Copy)]
enum BufferKind {
    Borrowed,
    Owned,
}

enum Buffer<'a> {
    Borrowed(BufferedAppender<'a>),
    Owned(OwnedBufferedAppender),
}

impl Buffer<'_> {
    fn run(mut self, initial: usize, flush: Flush, mut checkpoint: impl FnMut()) {
        let mut start = initial;
        for length in APPENDS {
            self.append(&values(start, length));
            start += length;
            checkpoint();
            if matches!(flush, Flush::Manual) {
                self.flush().unwrap();
                checkpoint();
            }
        }
        if matches!(flush, Flush::Drop) {
            drop(self);
        } else {
            self.finish();
        }
        checkpoint();
    }

    fn append(&mut self, values: &[i32]) {
        match self {
            Self::Borrowed(buffer) => buffer.append(values).unwrap(),
            Self::Owned(buffer) => buffer.append(values).unwrap(),
        }
    }

    fn append_raw(&mut self, bytes: &[u8]) {
        match self {
            Self::Borrowed(buffer) => buffer.append_raw(bytes).unwrap(),
            Self::Owned(buffer) => buffer.append_raw(bytes).unwrap(),
        }
    }

    fn flush(&mut self) -> Result<(), Error> {
        match self {
            Self::Borrowed(buffer) => buffer.flush(),
            Self::Owned(buffer) => buffer.flush(),
        }
    }

    fn finish(self) {
        match self {
            Self::Borrowed(buffer) => buffer.finish().unwrap(),
            Self::Owned(buffer) => buffer.finish().unwrap(),
        }
    }

    fn discard(self) -> Vec<u8> {
        match self {
            Self::Borrowed(buffer) => buffer.discard(),
            Self::Owned(buffer) => buffer.discard(),
        }
    }
}

/// Recorded writes, per-prefix observations, and transient allocation histories.
pub(super) struct Workload {
    base: Vec<u8>,
    operations: Vec<DiskOp>,
    pub(super) snapshots: Vec<Observation>,
    chunks: Vec<Chunk>,
    indexes: Vec<PendingIndex>,
    checkpoints: Vec<usize>,
}

impl Workload {
    /// Records the appends and flushes selected by `writer`.
    ///
    /// Parses each completed write and resize prefix and checks the final bytes and element count.
    /// Records actual chunk entries, payload bytes, and metadata allocations to locate preparation,
    /// ownership publication, and references.
    ///
    /// # Panics
    ///
    /// Panics if fixture I/O or parsing fails, the recording is empty, complete replay differs from
    /// the file, or a new allocation's preparation and ownership cannot be located before its
    /// reference.
    pub(super) fn record(path: &Path, writer: WritePath) -> Self {
        let base = fs::read(path).unwrap();
        let mut operations = Vec::new();
        let mut checkpoints = Vec::new();
        let mut events = 0;
        disk_log::start();
        writer.run(path, || {
            let recording = disk_log::finish();
            events += recording.events.len();
            operations.extend(recording.operations);
            checkpoints.push(operations.len());
            disk_log::start();
        });
        let recording = disk_log::finish();
        events += recording.events.len();
        operations.extend(recording.operations);
        assert!(!operations.is_empty());
        assert!(events > 0);
        let mut image = base.clone();
        let mut snapshots = vec![Observation::read(&image)];
        let mut images = vec![image.clone()];
        for operation in &operations {
            operation.apply(&mut image);
            snapshots.push(Observation::read(&image));
            images.push(image.clone());
        }
        assert_eq!(
            image,
            fs::read(path).unwrap(),
            "full transient reconstruction"
        );
        assert_eq!(
            snapshots.last().unwrap().values.len(),
            writer.final_elements(snapshots[0].values.len())
        );
        if !matches!(writer, WritePath::Immediate) {
            assert_eq!(checkpoints[0], 0, "buffered input allocated before a write");
        }
        let mut chunks = Vec::new();
        let mut indexes = Vec::new();
        for (point, (snapshot, image)) in snapshots.iter().zip(&images).enumerate() {
            for fact in snapshot.live.iter().filter(|fact| {
                fact.kind == AllocationKind::Index && !snapshots[0].live.contains(fact)
            }) {
                if !indexes
                    .iter()
                    .any(|pending: &PendingIndex| pending.fact == *fact)
                {
                    indexes.push(PendingIndex::capture(
                        *fact,
                        image,
                        point,
                        &operations,
                        &snapshots,
                    ));
                }
            }
            for record in &snapshot.chunks {
                let captured = Chunk::capture(record, snapshot, image, point);
                if !chunks.iter().any(|chunk: &Chunk| chunk.same(&captured)) {
                    chunks.push(captured);
                }
            }
        }
        for chunk in &mut chunks {
            chunk.locate_writes(&operations, &images, &mut snapshots);
        }
        for snapshot in &mut snapshots {
            snapshot.chunks.sort_unstable();
            let mut live = snapshot
                .live
                .iter()
                .map(|fact| fact.range)
                .collect::<Vec<_>>();
            super::assert_disjoint(&mut live, "transient reachable allocations");
        }
        Self {
            base,
            operations,
            snapshots,
            chunks,
            indexes,
            checkpoints,
        }
    }

    /// Checks each recorded prefix and passes its observation to `observe`.
    ///
    /// Writes each reconstructed prefix directly to `path` and checks ownership before calling
    /// `observe`. Checks the file bytes after `observe` returns.
    ///
    /// # Panics
    ///
    /// Panics if writing a prefix fails, ownership or visible-prefix checks fail, or `observe`
    /// changes the file bytes.
    pub(super) fn replay(
        &self,
        path: &Path,
        hole: &Coverage,
        observe: impl FnMut(usize, &Observation),
    ) {
        self.replay_with_history(path, hole, &[], observe);
    }

    /// Reconstructs the file after `point` completed byte operations.
    ///
    /// # Panics
    ///
    /// Panics if `point` exceeds the recorded operation count.
    pub(super) fn prefix(&self, point: usize) -> Vec<u8> {
        let mut bytes = self.base.clone();
        for operation in &self.operations[..point] {
            operation.apply(&mut bytes);
        }
        bytes
    }

    /// Returns complete allocations owned at `point` and retained through that prefix.
    ///
    /// Orders payload histories by decreasing ownership-publication point and omits a payload
    /// once a later persisted free section overlaps it.
    ///
    /// # Panics
    ///
    /// Panics if `point` exceeds the recorded operation count.
    pub(super) fn owned_history(&self, point: usize) -> Vec<Fact> {
        let mut history = Vec::new();
        for index in self.indexes.iter().filter(|index| index.owned <= point) {
            history.push(index.fact);
        }
        let mut issued = self
            .chunks
            .iter()
            .filter(|chunk| {
                chunk.owned <= point
                    && !(chunk.last_reference < point
                        && self.snapshots[chunk.last_reference + 1..=point].iter().any(
                            |snapshot| {
                                !Coverage::new([chunk.fact.range])
                                    .intersection(&snapshot.free)
                                    .is_empty()
                            },
                        ))
            })
            .collect::<Vec<_>>();
        issued.sort_by_key(|chunk| Reverse(chunk.owned));
        for chunk in issued {
            if !history.contains(&chunk.fact) {
                history.push(chunk.fact);
            }
        }
        for (last, snapshot) in self.snapshots[..=point].iter().enumerate().rev() {
            for fact in &snapshot.live {
                if !history.contains(fact)
                    && !self.snapshots[last + 1..=point].iter().any(|later| {
                        !Coverage::new([fact.range])
                            .intersection(&later.free)
                            .is_empty()
                    })
                {
                    history.push(*fact);
                }
            }
        }
        history
    }

    /// Checks replacement and batching premises, including a replacement with unchanged EOF.
    ///
    /// Also checks stored sizes and filter masks for `encoding`.
    ///
    /// # Panics
    ///
    /// Panics if a required path is absent or stored sizes or masks differ from the encoding's
    /// requirements.
    pub(super) fn assert_paths(&self, encoding: Encoding) {
        let first = &self.snapshots[0];
        let final_snapshot = self.snapshots.last().unwrap();
        assert!(final_snapshot.values.len() > first.values.len() + 3 * CHUNK);
        assert!(self.chunks.iter().any(|chunk| chunk.first_reference > 0));
        let mut saw_replacement = false;
        let mut saw_unchanged_eof = false;
        for (point, pair) in self.snapshots.windows(2).enumerate() {
            for old in &pair[0].chunks {
                if pair[1]
                    .chunks
                    .iter()
                    .any(|new| new.0 == old.0 && new.1 != old.1)
                {
                    saw_replacement = true;
                    saw_unchanged_eof |= self.chunks.iter().any(|chunk| {
                        chunk.record.0 == old.0
                            && chunk.first_reference == point + 1
                            && chunk.prepared > 0
                            && self.snapshots[chunk.prepared - 1].eof
                                == self.snapshots[chunk.first_reference].eof
                    });
                    assert!(
                        Coverage::new([Interval::new(old.1, old.2).unwrap()])
                            .intersection(&pair[0].free)
                            .is_empty()
                    );
                }
            }
        }
        assert!(saw_replacement, "no transient replacement");
        assert!(
            saw_unchanged_eof,
            "no replacement with unchanged published EOF"
        );
        assert!(self.checkpoints.len() >= 5);
        #[cfg(feature = "deflate")]
        if matches!(encoding, Encoding::Deflate | Encoding::RawEdges) {
            let sizes = self
                .chunks
                .iter()
                .map(|chunk| chunk.record.2)
                .collect::<BTreeSet<_>>();
            assert!(
                sizes.len() > 1,
                "no differing compressed stored sizes: {sizes:?}"
            );
        }
        #[cfg(feature = "deflate")]
        if matches!(encoding, Encoding::RawEdges) {
            assert!(self.chunks.iter().any(|chunk| chunk.record.3 != 0));
            assert!(self.chunks.iter().any(|chunk| chunk.record.3 == 0));
        }
        match encoding {
            Encoding::Plain => {
                assert!(
                    self.chunks
                        .iter()
                        .all(|chunk| chunk.record.2
                            == u64::try_from(CHUNK * size_of::<i32>()).unwrap())
                )
            }
            Encoding::Fletcher32 => assert!(self.chunks.iter().all(|chunk| chunk.record.2
                == u64::try_from(CHUNK * size_of::<i32>() + size_of::<u32>()).unwrap())),
            #[cfg(feature = "deflate")]
            Encoding::Deflate | Encoding::RawEdges => {}
        }
        assert!(
            self.chunks.iter().any(|chunk| chunk.owned > 0
                && self
                    .chunks
                    .iter()
                    .filter(|other| other.owned == chunk.owned)
                    .count()
                    > 1),
            "no multiple-payload allocation batch"
        );
    }

    /// Checks that superseded payload storage is free or reused at completion.
    ///
    /// # Panics
    ///
    /// Panics if no payload was replaced or superseded storage is missing from both final reachable
    /// and free coverage.
    pub(super) fn assert_reclaimed(&self) {
        let final_snapshot = self.snapshots.last().unwrap();
        let retired = self.retired();
        assert!(!retired.is_empty(), "no superseded payload allocation");
        let reused = final_snapshot.reachable();
        let missing = retired.difference(&reused).difference(&final_snapshot.free);
        assert!(
            missing.is_empty(),
            "superseded tails are not reclaimed: {missing:?}"
        );
    }

    /// Returns the first prefix with visible growth and an owned, unlinked payload allocation.
    ///
    /// # Panics
    ///
    /// Panics if no prefix satisfies both conditions.
    pub(super) fn interruption(&self) -> usize {
        self.snapshots
            .iter()
            .enumerate()
            .find_map(|(point, snapshot)| {
                (point > 0
                    && snapshot.values.len() > self.snapshots[0].values.len()
                    && self.chunks.iter().any(|chunk| {
                        chunk.owned <= point
                            && chunk.first_reference > point
                            && Coverage::new([chunk.fact.range])
                                .intersection(&snapshot.free)
                                .is_empty()
                            && chunk.fact.range.end() <= snapshot.eof
                    }))
                .then_some(point)
            })
            .unwrap()
    }
    /// Checks each prefix with allocations retained from earlier interrupted sessions.
    fn replay_with_history(
        &self,
        path: &Path,
        hole: &Coverage,
        history: &[Fact],
        mut observe: impl FnMut(usize, &Observation),
    ) {
        let mut image = self.base.clone();
        let mut previous = self.snapshots[0].values.len();
        for (point, snapshot) in self.snapshots.iter().enumerate() {
            if point > 0 {
                self.operations[point - 1].apply(&mut image);
            }
            fs::write(path, &image).unwrap();
            assert!(
                snapshot.values.len() >= previous,
                "committed prefix shrinks at {point}"
            );
            previous = snapshot.values.len();
            let facts = self.allocations(point, snapshot, history);
            allocation_oracle::inspect(
                Context {
                    case: "transient tail",
                    interruption: point,
                    publication: snapshot.state(),
                },
                hole,
                &snapshot.free,
                &snapshot.reachable(),
                &facts,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{error:?}; facts {facts:?}; records {:?}; free {:?}",
                    self.chunks
                        .iter()
                        .map(|chunk| (
                            &chunk.record,
                            chunk.prepared,
                            chunk.first_reference,
                            chunk.last_reference
                        ))
                        .collect::<Vec<_>>(),
                    snapshot.free
                )
            });
            observe(point, snapshot);
            assert_eq!(
                fs::read(path).unwrap(),
                image,
                "inspection changes transient prefix {point}"
            );
        }
        assert_eq!(image, self.prefix(self.operations.len()));
    }

    /// Combines reachable, retained, and prepared allocation facts for one prefix.
    fn allocations(
        &self,
        point: usize,
        snapshot: &Observation,
        history: &[Fact],
    ) -> Vec<Allocation> {
        let mut facts = snapshot
            .live
            .iter()
            .map(|fact| fact.allocation(AllocationState::Linked))
            .collect::<Vec<_>>();
        let mut retained = self.owned_history(point);
        retained.extend(
            history
                .iter()
                .filter(|fact| {
                    !self.snapshots[..=point].iter().any(|observed| {
                        !Coverage::new([fact.range])
                            .intersection(&observed.free)
                            .is_empty()
                    })
                })
                .copied(),
        );
        for fact in retained {
            snapshot.retain_owned(&mut facts, fact);
        }
        for chunk in &self.chunks {
            if chunk.prepared > point || chunk.first_reference <= point {
                continue;
            }
            let coverage = Coverage::new([chunk.fact.range]);
            let state = if chunk.owned <= point {
                AllocationState::Orphaned
            } else {
                AllocationState::Provisional
            };
            if !facts.iter().any(|other| {
                !coverage
                    .intersection(&Coverage::new([other.complete()]))
                    .is_empty()
            }) {
                facts.push(chunk.fact.allocation(state));
            }
        }
        for index in self
            .indexes
            .iter()
            .filter(|index| index.prepared <= point && point < index.owned)
        {
            let coverage = Coverage::new([index.fact.range]);
            if !facts.iter().any(|other| {
                !coverage
                    .intersection(&Coverage::new([other.complete()]))
                    .is_empty()
            }) {
                facts.push(index.fact.allocation(AllocationState::Provisional));
            }
        }
        facts
    }

    /// Returns payload ranges whose final reference precedes the completed log.
    fn retired(&self) -> Coverage {
        Coverage::new(
            self.chunks
                .iter()
                .filter(|chunk| chunk.last_reference < self.operations.len())
                .map(|chunk| chunk.fact.range),
        )
    }
}

/// A new index allocation and its preparation and ownership-publication points.
///
/// Points are counts of completed byte operations.
struct PendingIndex {
    fact: Fact,
    prepared: usize,
    owned: usize,
}

impl PendingIndex {
    /// Locates complete preparation and distinct ownership publication before an index reference.
    fn capture(
        fact: Fact,
        image: &[u8],
        reference: usize,
        operations: &[DiskOp],
        snapshots: &[Observation],
    ) -> Self {
        let start = usize::try_from(fact.range.start()).unwrap();
        let signature = &image[start..start + test_util::extensible_array::SIGNATURE.len()];
        let mut written = Coverage::default();
        let mut bytes = Vec::new();
        let prepared = (1..=reference)
            .find(|&point| {
                if let DiskOp::Write {
                    offset,
                    bytes: payload,
                } = &operations[point - 1]
                {
                    let changed = Coverage::new([Interval::new(
                        *offset,
                        u64::try_from(payload.len()).unwrap(),
                    )
                    .unwrap()])
                    .intersection(&Coverage::new([fact.range]));
                    written = Coverage::new(
                        written
                            .intervals()
                            .iter()
                            .copied()
                            .chain(changed.intervals().iter().copied()),
                    );
                }
                operations[point - 1].apply(&mut bytes);
                Coverage::new([fact.range]).difference(&written).is_empty()
                    && bytes.get(start..start + signature.len()) == Some(signature)
            })
            .unwrap_or_else(|| {
                panic!("index allocation is not prepared: {fact:?}, prefix {reference}")
            });
        let owned = (prepared..=reference)
            .find(|&point| {
                matches!(operations[point - 1], DiskOp::Write { offset: 0, .. })
                    && fact.range.end() <= snapshots[point].eof
                    && Coverage::new([fact.range])
                        .intersection(&snapshots[point].free)
                        .is_empty()
            })
            .unwrap_or_else(|| {
                panic!("index allocation is not published: {fact:?}, prefix {reference}")
            });
        assert!(
            owned < reference,
            "index reference shares ownership publication: {fact:?}, prefix {reference}"
        );
        Self {
            fact,
            prepared,
            owned,
        }
    }
}

/// A stored chunk record, its encoded index entry, and its payload bytes.
///
/// Preparation, ownership, and reference points are counts of completed byte operations.
struct Chunk {
    record: (Vec<u64>, u64, u64, u32),
    fact: Fact,
    parent: Interval,
    field: usize,
    encoded: Vec<u8>,
    payload: Vec<u8>,
    prepared: usize,
    owned: usize,
    first_reference: usize,
    last_reference: usize,
}

impl Chunk {
    /// Reads a chunk's complete stored bytes and encoded entry from a supplied prefix.
    fn capture(
        record: &(Vec<u64>, u64, u64, u32),
        observation: &Observation,
        bytes: &[u8],
        point: usize,
    ) -> Self {
        let address = record.1.to_le_bytes();
        let fields = observation
            .element_blocks
            .iter()
            .flat_map(|parent| {
                let start = usize::try_from(parent.start()).unwrap();
                let end = usize::try_from(parent.end()).unwrap();
                test_util::bytes::signature_offsets(&bytes[start..end], &address)
                    .into_iter()
                    .map(move |local| (*parent, start + local))
            })
            .collect::<Vec<_>>();
        assert_eq!(fields.len(), 1, "stored pointer for {record:?}: {fields:?}");
        let stride = observation.element_size.unwrap();
        let (parent, field) = fields[0];
        let fact = Fact::new(AllocationKind::Payload, record.1, record.2);
        Self {
            record: record.clone(),
            fact,
            parent,
            field,
            encoded: bytes[field..field + stride].to_vec(),
            payload: bytes[usize::try_from(fact.range.start()).unwrap()
                ..usize::try_from(fact.range.end()).unwrap()]
                .to_vec(),
            prepared: 0,
            owned: 0,
            first_reference: point,
            last_reference: point,
        }
    }

    /// Locates complete payload preparation, ownership publication, and stored references.
    ///
    /// Adds a payload when its parent is reachable and its encoded entry and stored bytes match.
    fn locate_writes(
        &mut self,
        operations: &[DiskOp],
        images: &[Vec<u8>],
        snapshots: &mut [Observation],
    ) {
        self.prepared = if self.first_reference == 0 {
            0
        } else {
            let mut written = Coverage::default();
            (1..=self.first_reference)
                .find(|&point| {
                    if let DiskOp::Write { offset, bytes } = &operations[point - 1] {
                        let changed = Coverage::new([Interval::new(
                            *offset,
                            u64::try_from(bytes.len()).unwrap(),
                        )
                        .unwrap()])
                        .intersection(&Coverage::new([self.fact.range]));
                        written = Coverage::new(
                            written
                                .intervals()
                                .iter()
                                .copied()
                                .chain(changed.intervals().iter().copied()),
                        );
                    }
                    Coverage::new([self.fact.range])
                        .difference(&written)
                        .is_empty()
                        && self.payload_matches(&images[point])
                })
                .unwrap_or_else(|| panic!("no complete preparation for {:?}", self.record))
        };
        self.owned = if self.first_reference == 0 {
            0
        } else {
            (self.prepared..=self.first_reference)
                .find(|&point| {
                    point > 0
                        && matches!(operations[point - 1], DiskOp::Write { offset: 0, .. })
                        && self.fact.range.end() <= snapshots[point].eof
                        && Coverage::new([self.fact.range])
                            .intersection(&snapshots[point].free)
                            .is_empty()
                })
                .unwrap_or_else(|| panic!("no ownership publication for {:?}", self.record))
        };
        for (point, (snapshot, image)) in snapshots.iter_mut().zip(images.iter()).enumerate() {
            if self.referenced(snapshot, image) {
                assert!(
                    self.owned < point || self.first_reference == 0,
                    "reference before ownership: {:?}, prefix {point}, ownership {}",
                    self.record,
                    self.owned
                );
                self.first_reference = self.first_reference.min(point);
                self.last_reference = point;
                if !snapshot.chunks.contains(&self.record) {
                    assert!(self.fact.range.end() <= snapshot.eof);
                    assert!(self.fact.range.end() <= u64::try_from(image.len()).unwrap());
                    snapshot.live.push(self.fact);
                    snapshot.chunks.push(self.record.clone());
                }
            }
        }
    }

    fn same(&self, other: &Self) -> bool {
        self.record == other.record && self.payload == other.payload
    }

    fn payload_matches(&self, bytes: &[u8]) -> bool {
        bytes.get(
            usize::try_from(self.fact.range.start()).unwrap()
                ..usize::try_from(self.fact.range.end()).unwrap(),
        ) == Some(self.payload.as_slice())
    }

    /// Returns whether the parent is reachable and the encoded entry and payload bytes match.
    fn referenced(&self, observation: &Observation, bytes: &[u8]) -> bool {
        observation.element_blocks.contains(&self.parent)
            && bytes.get(self.field..self.field + self.encoded.len())
                == Some(self.encoded.as_slice())
            && self.payload_matches(bytes)
    }
}

impl Observation {
    /// Adds an orphan allocation whose extent overlaps neither free sections nor existing facts.
    fn retain_owned(&self, facts: &mut Vec<Allocation>, fact: Fact) {
        let coverage = Coverage::new([fact.range]);
        if coverage.intersection(&self.free).is_empty()
            && !facts.iter().any(|other| {
                !coverage
                    .intersection(&Coverage::new([other.complete()]))
                    .is_empty()
            })
        {
            facts.push(fact.allocation(AllocationState::Orphaned));
        }
    }
}

/// Checks that empty buffered operations and an incomplete-element flush preserve allocation state.
///
/// The fixture has an empty, unallocated dataset. Uses an owned appender when `owned` is `true`,
/// and a borrowed appender otherwise. Checks the fixture bytes and undefined index address.
///
/// # Panics
///
/// Panics if fixture I/O fails, the fixture has an allocated index, the incomplete-element error
/// differs from its expected variant and message, or the file bytes change.
pub(super) fn assert_empty_operations(path: &Path, owned: bool, hole: &Coverage) {
    let before = fs::read(path).unwrap();
    assert!(!Observation::read(&before).index_defined);
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let mut buffer = if owned {
        Buffer::Owned(dataset.into_buffered_appender().unwrap())
    } else {
        Buffer::Borrowed(dataset.buffered_appender().unwrap())
    };
    assert_eq!(file.disk_bytes().unwrap(), before);
    buffer.append(&[]);
    buffer.flush().unwrap();
    assert_eq!(file.disk_bytes().unwrap(), before);
    buffer.append_raw(&[0]);
    let error = buffer.flush().unwrap_err();
    let Error::AppendInPlaceUnsupported(reason) = error else {
        panic!("expected incomplete buffered element, got {error:?}")
    };
    assert_eq!(
        reason,
        "the buffered byte length is not a whole number of elements; append the rest of the trailing element before flushing"
    );
    assert_eq!(file.disk_bytes().unwrap(), before);
    assert_eq!(buffer.discard(), [0]);
    if owned {
        drop(file.dataset("d").unwrap().into_buffered_appender().unwrap());
    } else {
        let mut empty = file.dataset("d").unwrap();
        drop(empty.buffered_appender().unwrap());
    }
    assert_eq!(file.disk_bytes().unwrap(), before);
    file.close().unwrap();
    assert_eq!(fs::read(path).unwrap(), before);
    let after = Observation::read(&before);
    assert!(!after.index_defined);
    after.inspect("empty buffered operations", 0, hole, &[]);
}

fn values(start: usize, length: usize) -> Vec<i32> {
    (start..start + length)
        .map(|value| i32::try_from(value).unwrap())
        .collect()
}

/// Input batch lengths that cross complete-chunk and partial-tail boundaries.
const APPENDS: [usize; 4] = [CHUNK / 3, CHUNK / 2, 3 * CHUNK + CHUNK / 7, CHUNK / 5];
