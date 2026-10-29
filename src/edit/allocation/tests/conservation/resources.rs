//! Checks append I/O costs and publication-state lifetimes.
//!
//! Tests compare read costs for fixed batches after growing the dataset and extending the session.
//! The read measurements use a persistence threshold of 4096 bytes and bound the free-section
//! count. Tests repeat an append after clearing the located-dataset cache to measure metadata
//! traversal as a control.
//!
//! Write measurements use a threshold of one byte and separate or batched full-chunk appends.
//! Tests bound recorded metadata bytes and free-section counts and check allocation conservation.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util_hdf5::file_space::publication;

use crate::File;
use crate::FileAccessProperties;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::allocation::State;
use crate::file_lock::FileLocking;
use crate::image::CountingImage;
use crate::image::FileImage;
use crate::image::HandleImage;
use crate::image::WriteBuffering;
use crate::source::MetadataCacheConfig;

use super::CHUNK;
use super::Observation;

#[rstest]
#[case::bounded_flat(MemoryStrategy::Bounded, false)]
#[case::bounded_page(MemoryStrategy::Bounded, true)]
#[case::mirrored_flat(MemoryStrategy::Mirrored, false)]
#[case::mirrored_page(MemoryStrategy::Mirrored, true)]
fn append_reads_and_publication_state_remain_bounded_as_the_session_grows(
    #[case] backing: MemoryStrategy,
    #[case] paged: bool,
    #[values(false, true)] deferred: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let short = run(
        &directory.path().join("short.h5"),
        backing,
        paged,
        deferred,
        32,
        16,
    );
    let long = run(
        &directory.path().join("long.h5"),
        backing,
        paged,
        deferred,
        2048,
        512,
    );
    assert!(long.maximum <= short.maximum + 4096, "{short:?}; {long:?}");
    assert!(
        long.bytes / 512 <= short.bytes / 16 + 4096,
        "{short:?}; {long:?}"
    );
    assert!(
        long.maximum_calls <= short.maximum_calls + 32,
        "{short:?}; {long:?}"
    );
    assert!(
        long.calls / 512 <= short.calls / 16 + 32,
        "{short:?}; {long:?}"
    );
    assert!(long.cold_calls > long.maximum_calls, "{long:?}");
    assert!(
        long.cold_calls > short.maximum_calls + 32,
        "{short:?}; {long:?}"
    );
    assert!(long.cold > long.maximum, "{long:?}");
    assert!(long.cold > short.maximum + 4096, "{short:?}; {long:?}");
    assert!(long.cold < 2048 * CHUNK as u64, "{long:?}");
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded)]
#[case::mirrored(MemoryStrategy::Mirrored)]
fn interrupted_reopens_release_each_sessions_publication_identity(
    #[case] backing: MemoryStrategy,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reopen.h5");
    super::pure_fixture(&path, paged, deferred, false);
    let initial = if deferred { 0 } else { CHUNK };
    let values = std::array::from_fn::<_, CHUNK, _>(|i| i32::try_from(initial + i).unwrap());
    for cycle in 0..32 {
        let reads = Arc::new(AtomicU64::new(0));
        let calls = Arc::new(AtomicU64::new(0));
        let mut engine = open(&path, backing, Arc::clone(&reads), Arc::clone(&calls));
        let identity = Arc::downgrade(&engine.context.allocations.session);
        let writes = engine.image.issued_writes();
        engine.append_inplace_i32_phased("d", &values, 2).unwrap();
        assert!(engine.image.issued_writes() > writes);
        assert!(engine.context.allocations.sequence > 0);
        assert!(engine.append_allocator.pending.is_none());
        assert!(matches!(
            engine.context.allocations.state,
            State::Confirmed(_)
        ));
        assert_eq!(Arc::strong_count(&engine.context.allocations.session), 1);
        assert!(engine.located.is_empty());
        assert!(engine.resolved.len() <= 1);
        assert!(
            (256..64 * 1024).contains(&reads.load(Ordering::Relaxed)),
            "cycle {cycle}"
        );
        assert!(
            (8..256).contains(&calls.load(Ordering::Relaxed)),
            "cycle {cycle}"
        );
        drop(engine);
        assert!(
            identity.upgrade().is_none(),
            "cycle {cycle} retained its session"
        );
        let observation = Observation::read(&fs::read(&path).unwrap());
        assert_eq!(
            observation.values,
            (0..initial)
                .map(|i| i32::try_from(i).unwrap())
                .collect::<Vec<_>>()
        );
        assert!(
            observation
                .free
                .intersection(&observation.reachable())
                .is_empty()
        );
    }
    let mut engine = open(&path, backing, Arc::default(), Arc::default());
    engine.append_inplace_i32_phased("d", &values, 4).unwrap();
    assert_session_state(&engine);
    engine.finalize_persist().unwrap();
    drop(engine);
    let observation = Observation::read(&fs::read(&path).unwrap());
    assert_eq!(
        observation.values,
        (0..initial + CHUNK)
            .map(|i| i32::try_from(i).unwrap())
            .collect::<Vec<_>>()
    );
}

/// Measures fixed-batch reads after index initialization and dataset growth.
///
/// Excludes fixture construction, initial appends, free-section queries, and final ownership
/// inspection from the read totals. Checks publication state after each measured append.
fn run(
    path: &Path,
    backing: MemoryStrategy,
    paged: bool,
    deferred: bool,
    retained_chunks: usize,
    appends: usize,
) -> Reads {
    super::pure_fixture_with_threshold(path, paged, deferred, 4096, |_| {});
    let initial = if deferred { 0 } else { CHUNK };
    let mut next = initial;
    let read_bytes = Arc::new(AtomicU64::new(0));
    let read_calls = Arc::new(AtomicU64::new(0));
    let mut engine = open(
        path,
        backing,
        Arc::clone(&read_bytes),
        Arc::clone(&read_calls),
    );
    let first = read_bytes.load(Ordering::Relaxed);
    engine
        .append_inplace_i32_phased(
            "d",
            &std::array::from_fn::<_, CHUNK, _>(|i| i32::try_from(next + i).unwrap()),
            4,
        )
        .unwrap();
    let first = read_bytes.load(Ordering::Relaxed) - first;
    assert!(
        (256..64 * 1024).contains(&first),
        "initialization reads: {first}"
    );
    assert_session_state(&engine);
    next += CHUNK;
    let retained = (next..next + retained_chunks * CHUNK)
        .map(|i| i32::try_from(i).unwrap())
        .collect::<Vec<_>>();
    next += retained.len();
    engine.append_inplace_i32_phased("d", &retained, 4).unwrap();
    drop(retained);
    assert_session_state(&engine);
    let sections = engine.reusable_free_sections().unwrap().section_count();
    assert!(sections <= 16, "sections before appends: {sections}");
    let sequence = engine.context.allocations.sequence;
    let writes = engine.image.issued_writes();
    let mut reads = Reads {
        bytes: 0,
        maximum: 0,
        cold: 0,
        calls: 0,
        maximum_calls: 0,
        cold_calls: 0,
    };
    for _ in 0..appends {
        let before = read_bytes.load(Ordering::Relaxed);
        let calls = read_calls.load(Ordering::Relaxed);
        engine
            .append_inplace_i32_phased(
                "d",
                &std::array::from_fn::<_, CHUNK, _>(|i| i32::try_from(next + i).unwrap()),
                4,
            )
            .unwrap();
        next += CHUNK;
        let bytes = read_bytes.load(Ordering::Relaxed) - before;
        assert!((128..32 * 1024).contains(&bytes), "append reads: {bytes}");
        let calls = read_calls.load(Ordering::Relaxed) - calls;
        assert!((8..128).contains(&calls), "append read calls: {calls}");
        reads.calls += calls;
        reads.maximum_calls = reads.maximum_calls.max(calls);
        reads.bytes += bytes;
        reads.maximum = reads.maximum.max(bytes);
        assert_session_state(&engine);
    }
    assert!(engine.context.allocations.sequence >= sequence + appends as u64);
    assert!(engine.image.issued_writes() >= writes + appends as u64);
    assert!(
        engine.reusable_free_sections().unwrap().section_count() <= 16,
        "sections: {sections} -> {}",
        engine.reusable_free_sections().unwrap().section_count()
    );
    engine.located.clear();
    let before = read_bytes.load(Ordering::Relaxed);
    let calls = read_calls.load(Ordering::Relaxed);
    engine
        .append_inplace_i32_phased(
            "d",
            &std::array::from_fn::<_, CHUNK, _>(|i| i32::try_from(next + i).unwrap()),
            4,
        )
        .unwrap();
    reads.cold = read_bytes.load(Ordering::Relaxed) - before;
    reads.cold_calls = read_calls.load(Ordering::Relaxed) - calls;
    assert_session_state(&engine);
    engine.finalize_persist().unwrap();
    drop(engine);
    let expected = initial + (retained_chunks + appends + 2) * CHUNK;
    let file = File::open(path).unwrap();
    let values = (0..expected)
        .map(|i| i32::try_from(i).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), values);
    drop(file);
    let bytes = fs::read(path).unwrap();
    let observation = Observation::read_with_threshold(&bytes, 4096);
    assert_eq!(observation.values, values);
    assert!(
        observation
            .free
            .intersection(&observation.reachable())
            .is_empty()
    );
    assert!(observation.eof <= bytes.len() as u64);
    assert!(
        observation
            .live
            .iter()
            .all(|fact| fact.range.end() <= observation.eof)
    );
    eprintln!("{backing:?}/{paged}/{deferred}/{retained_chunks}/{appends}: {reads:?}");
    reads
}

/// Read totals and per-append maxima for a prepared session.
#[derive(Debug)]
struct Reads {
    /// Total requested bytes across the measured append loop.
    bytes: u64,
    /// Maximum requested bytes for one append in the loop.
    maximum: u64,
    /// Requested bytes for the control append after clearing the located-dataset cache.
    cold: u64,
    /// Total read calls across the measured append loop.
    calls: u64,
    /// Maximum read calls for one append in the loop.
    maximum_calls: u64,
    /// Read calls for the control append after clearing the located-dataset cache.
    cold_calls: u64,
}

/// Opens an unbuffered engine with read counters and the image metadata cache disabled.
fn open(
    path: &Path,
    backing: MemoryStrategy,
    reads: Arc<AtomicU64>,
    calls: Arc<AtomicU64>,
) -> WriteEngine {
    let mut engine = WriteEngine::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
        let image: Box<dyn FileImage> = match backing {
            MemoryStrategy::Bounded => Box::new(HandleImage::new(
                handle,
                len,
                MetadataCacheConfig::disabled(),
            )),
            MemoryStrategy::Mirrored => Box::new(WriteEngine::read_mirror(handle)?),
            MemoryStrategy::Auto => panic!("resource test requires an explicit backing"),
        };
        Ok(Box::new(
            CountingImage::new(image, reads, Arc::default()).with_read_calls(calls),
        ))
    })
    .unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.batched_appends = true;
    engine
}

/// Checks completed publication, bounded lookup caches, and sole ownership of the session identity.
fn assert_session_state(engine: &WriteEngine) {
    assert!(engine.append_allocator.pending.is_none());
    assert!(matches!(
        engine.context.allocations.state,
        State::Confirmed(_)
    ));
    assert_eq!(Arc::strong_count(&engine.context.allocations.session), 1);
    assert_eq!(Arc::weak_count(&engine.context.allocations.session), 0);
    assert_eq!(engine.located.len(), 1);
    assert!(engine.resolved.len() <= 1);
    engine.context.allocations.before_reference().unwrap();
}

#[rstest]
#[case::separate_2000(2000, 1)]
#[case::batched_2000(2000, 250)]
#[case::separate_8000(8000, 1)]
#[case::batched_8000(8000, 1000)]
fn threshold_one_manager_writes_and_sections_remain_bounded(
    #[case] chunks: usize,
    #[case] batch: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    threshold_one_workload(&directory.path().join("threshold-one.h5"), chunks, batch);
}

/// Checks write and free-section bounds for `chunks` new chunks in batches of `batch` chunks.
///
/// Records only the append loop and subtracts one copy of the appended payload from the written
/// bytes. Checks complete values, live/free separation, and conservation of the initial free space.
pub(super) fn threshold_one_workload(path: &Path, chunks: usize, batch: usize) {
    publication::session(path).close().unwrap();
    let initial = Observation::read(&fs::read(path).unwrap());
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap();
    let initial_sections = file.reusable_free_sections().unwrap().section_count();
    let values = publication::values(chunks);
    let mut dataset = file.dataset("d").unwrap();
    disk_log::start();
    for values in values.chunks(batch * publication::CHUNK) {
        dataset.append(values).unwrap();
    }
    let recording = disk_log::finish();
    drop(dataset);
    let written: usize = recording
        .operations
        .iter()
        .map(|operation| match operation {
            DiskOp::Write { bytes, .. } => bytes.len(),
            DiskOp::SetLen(_) => 0,
        })
        .sum();
    let metadata = written.checked_sub(values.len() * 4).unwrap();
    let sections = file.reusable_free_sections().unwrap().section_count();
    file.close().unwrap();
    publication::verify(path, chunks);
    let final_state = Observation::read(&fs::read(path).unwrap());
    let live = Coverage::new(final_state.live.iter().map(|fact| fact.range));
    assert_eq!(
        initial.free.difference(&live),
        final_state.free.intersection(&initial.free)
    );
    assert!(
        final_state
            .free
            .intersection(&final_state.reachable())
            .is_empty()
    );
    eprintln!(
        "chunks={chunks}, batch={batch}, metadata={metadata}, sections={initial_sections}->{sections}"
    );
    assert!(metadata >= chunks * 8);
    assert!(metadata <= (chunks / batch) * 2048 + chunks * 32);
    assert!(sections <= initial_sections + 8);
}
