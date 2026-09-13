//! Public chunk-cache observability: `Dataset::chunk_cache_stats()` lets a
//! downstream caller confirm their chunk-cache tuning is taking effect, without
//! reaching into crate internals.

use hdf5_pure::{ChunkCacheConfig, File, FileAccessProperties, FileBuilder, MaxExtent};

fn chunked_file_bytes() -> Vec<u8> {
    let data: Vec<i32> = (0..256).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("chunked")
        .with_i32_data(&data)
        .with_shape(&[256])
        .with_chunks(&[32]);
    b.finish().unwrap()
}

#[test]
fn fresh_handle_reports_empty_stats_before_any_read() {
    let file = File::from_bytes(chunked_file_bytes()).unwrap();
    let ds = file.dataset("chunked").unwrap();
    let stats = ds.chunk_cache_stats();
    assert!(!stats.index_loaded());
    assert_eq!(stats.cached_chunks(), 0);
    assert_eq!(stats.cached_bytes(), 0);
    assert_eq!(stats.lookups(), 0);
    // `None`, not `Some(0.0)` — a cache nobody asked and a cache that missed
    // everything are opposite signals.
    assert_eq!(stats.hit_rate(), None);
}

#[test]
fn enabled_cache_reports_retained_index_and_chunks_after_read() {
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
    )
    .unwrap();
    let ds = file.dataset("chunked").unwrap();
    assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

    let stats = ds.chunk_cache_stats();
    assert!(stats.index_loaded());
    assert!(stats.cached_chunks() > 0);
    assert_eq!(stats.cached_bytes(), stats.cached_chunks() * 32 * 4);
}

#[test]
fn disabled_cache_reports_nothing_retained_after_read() {
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
    )
    .unwrap();
    let ds = file.dataset("chunked").unwrap();
    // Reads still return correct data; the cache simply retains nothing.
    assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

    let stats = ds.chunk_cache_stats();
    assert!(!stats.index_loaded());
    assert_eq!(stats.cached_chunks(), 0);
    assert_eq!(stats.cached_bytes(), 0);
}

// ---------------------------------------------------------------------------
// Cumulative counters (issue #356)
// ---------------------------------------------------------------------------

const CHUNK_ROWS: u64 = 16;
const COLS: u64 = 64;
const CHUNKS: u64 = 128;
const ROWS: u64 = CHUNK_ROWS * CHUNKS;

/// A dataset of 128 chunks of 8 KiB: eight times the default 16 slots, and
/// exactly the default 1 MiB, so the slot count is what binds.
fn oversized_file_bytes() -> Vec<u8> {
    let data: Vec<f64> = (0..(ROWS * COLS)).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("grid")
        .with_f64_data(&data)
        .with_shape(&[ROWS, COLS])
        .with_chunks(&[CHUNK_ROWS, COLS]);
    b.finish().unwrap()
}

fn oversized_file() -> File {
    File::from_bytes_with_options(
        oversized_file_bytes(),
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
    )
    .unwrap()
}

/// The figure the eviction count cannot give.
///
/// A read visits each of its chunks once and stops offering once it has filled
/// the cache, so a dataset eight times the budget evicts *nothing* while
/// retaining only an eighth of itself. A caller reading `evictions()` alone
/// would call this cache comfortably sized.
#[test]
fn a_read_larger_than_the_cache_reports_rejections_and_no_evictions() {
    let file = oversized_file();
    let ds = file.dataset("grid").unwrap();
    let _ = ds.read_f64().unwrap();

    let stats = ds.chunk_cache_stats();
    let slots = ds.chunk_cache_config().max_slots() as u64;
    assert_eq!(stats.hits(), 0);
    assert_eq!(stats.misses(), CHUNKS);
    assert_eq!(stats.lookups(), CHUNKS);
    assert_eq!(stats.hit_rate(), Some(0.0));

    assert_eq!(stats.evictions(), 0);
    assert_eq!(stats.rejections(), CHUNKS - slots);
    assert_eq!(stats.oversize_chunks(), 0);
    assert_eq!(stats.cached_chunks() as u64, slots);
}

/// The same dataset read in row windows reports the mirror image, and a caller
/// reading only one of the two counters would call one of these cases healthy.
///
/// A row window asks for the plain LRU rule, because the chunk its successor
/// needs is the one it finished on. So it evicts freely — including chunks it
/// placed itself — and never reaches the refusal that a whole read hits.
#[test]
fn a_windowed_read_reports_the_reverse_of_a_whole_one() {
    let file = oversized_file();
    let ds = file.dataset("grid").unwrap();
    let slots = ds.chunk_cache_config().max_slots() as u64;

    let mut row = 0;
    while row < ROWS {
        let n = 40.min(ROWS - row);
        let _ = ds.read_f64_rows(row, n).unwrap();
        row += n;
    }

    let stats = ds.chunk_cache_stats();
    // Every chunk is placed once; all but the last `max_slots` are then evicted.
    assert_eq!(stats.evictions(), CHUNKS - slots);
    assert_eq!(stats.rejections(), 0);
    assert_eq!(stats.cached_chunks() as u64, slots);
}

/// The address sort: a second read must reach the chunks the first one left.
///
/// Both reads visit in ascending file-address order, so the first `max_slots`
/// chunks the second read asks for are exactly the ones the first read retained.
/// Taking the chunk list from the index map without sorting it makes the second
/// read's order the map's instead, and the retained set an arbitrary subset it
/// walks past at random.
#[test]
fn a_repeat_read_hits_every_chunk_the_first_read_retained() {
    let file = oversized_file();
    let ds = file.dataset("grid").unwrap();
    let _ = ds.read_f64().unwrap();
    let slots = ds.chunk_cache_config().max_slots() as u64;
    assert_eq!(ds.chunk_cache_stats().cached_chunks() as u64, slots);

    ds.reset_chunk_cache_stats();
    let _ = ds.read_f64().unwrap();

    let stats = ds.chunk_cache_stats();
    assert_eq!(stats.hits(), slots);
    assert_eq!(stats.misses(), CHUNKS - slots);
    // And it keeps them. Being served a chunk claims the slot for this read, so
    // the misses that follow have nothing they are allowed to take and are
    // refused instead of giving back what the read was just served.
    assert_eq!(stats.evictions(), 0);
    assert_eq!(stats.rejections(), CHUNKS - slots);
}

/// The retained set is stable, not a two-cycle.
///
/// Before a hit claimed its slot, each read gave back the `max_slots` chunks it
/// had just been served — one per miss — so the read after it hit nothing and
/// the hit count alternated `max_slots`, 0, `max_slots`, 0. Reading twice was
/// enough to look correct, which is why this reads five times.
#[test]
fn a_dataset_read_over_and_over_keeps_hitting_what_it_retained() {
    let file = oversized_file();
    let ds = file.dataset("grid").unwrap();
    let slots = ds.chunk_cache_config().max_slots() as u64;
    let _ = ds.read_f64().unwrap();

    for pass in 2..=5 {
        ds.reset_chunk_cache_stats();
        let _ = ds.read_f64().unwrap();
        let stats = ds.chunk_cache_stats();
        assert_eq!(stats.hits(), slots, "read {pass} hit a different count");
        // Steady state costs nothing: no chunk is placed and none given up.
        assert_eq!(stats.evictions(), 0, "read {pass} evicted");
        assert_eq!(stats.cached_chunks() as u64, slots);
    }
}

/// Claiming on a hit must not freeze the cache onto the first thing it saw.
///
/// A read only claims what it is actually served, so a read whose chunks are not
/// in the cache claims nothing and takes the slots it needs. This is the
/// over-correction guard: a rule that simply refused to evict an earlier read's
/// chunks would pass the test above and strand the cache here.
#[test]
fn a_read_that_wants_different_chunks_still_takes_the_slots() {
    let file = oversized_file();
    let ds = file.dataset("grid").unwrap();
    let slots = ds.chunk_cache_config().max_slots() as u64;

    // Fill from the head of the dataset, repeatedly, so the set is settled.
    let _ = ds.read_f64().unwrap();
    let _ = ds.read_f64().unwrap();
    assert_eq!(ds.chunk_cache_stats().cached_chunks() as u64, slots);

    // Now ask for the far end, which none of that retained.
    ds.reset_chunk_cache_stats();
    let _ = ds.read_f64_rows(ROWS - 200, 200).unwrap();

    let stats = ds.chunk_cache_stats();
    assert_eq!(
        stats.hits(),
        0,
        "the tail should share no chunk with the head"
    );
    assert!(
        stats.evictions() > 0,
        "a read wanting chunks the cache does not hold must be able to take slots"
    );
    assert_eq!(stats.cached_chunks() as u64, slots);
}

#[test]
fn a_chunk_larger_than_the_byte_budget_is_counted_apart_from_rejections() {
    // Every chunk is 32 * 4 = 128 bytes, and the budget is 100.
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(100)),
    )
    .unwrap();
    let ds = file.dataset("chunked").unwrap();
    let _ = ds.read_i32().unwrap();

    let stats = ds.chunk_cache_stats();
    assert_eq!(stats.cached_chunks(), 0);
    assert_eq!(stats.oversize_chunks(), 8);
    // Not folded into `rejections`: more slots would not admit any of them.
    assert_eq!(stats.rejections(), 0);
    assert_eq!(stats.evictions(), 0);
}

#[test]
fn a_disabled_cache_counts_no_rejections_for_chunks_it_never_wanted() {
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
    )
    .unwrap();
    let ds = file.dataset("chunked").unwrap();
    let _ = ds.read_i32().unwrap();

    let stats = ds.chunk_cache_stats();
    assert_eq!(stats.misses(), 8);
    assert_eq!(stats.rejections(), 0);
    assert_eq!(stats.oversize_chunks(), 0);
    assert_eq!(stats.evictions(), 0);
}

#[test]
fn resetting_the_counters_leaves_the_retained_chunks_in_place() {
    let file = oversized_file();
    let ds = file.dataset("grid").unwrap();
    let _ = ds.read_f64().unwrap();

    let before = ds.chunk_cache_stats();
    assert!(before.misses() > 0);
    assert!(before.rejections() > 0);

    ds.reset_chunk_cache_stats();

    let after = ds.chunk_cache_stats();
    assert_eq!(after.misses(), 0);
    assert_eq!(after.hits(), 0);
    assert_eq!(after.rejections(), 0);
    assert_eq!(after.lookups(), 0);
    assert_eq!(after.hit_rate(), None);
    // Occupancy is what the cache *holds*, not what it has done.
    assert_eq!(after.cached_chunks(), before.cached_chunks());
    assert_eq!(after.cached_bytes(), before.cached_bytes());
    assert_eq!(after.index_loaded(), before.index_loaded());
}

/// A file with two appendable chunked datasets, for the invalidation tests.
fn two_dataset_session(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    for name in ["log", "other"] {
        b.create_dataset(name)
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[8]);
    }
    b.write(path).unwrap();
}

/// A write drops the chunks it may have made stale, and says how many. Nothing
/// else distinguishes "the budget is too small" from "this session keeps
/// rewriting what it caches".
///
/// The scope is the **session**, not the handle: the write here goes through a
/// different dataset entirely, and the reading handle still loses everything,
/// because a commit advances the file's content revision and every handle drops
/// its chunks the next time it resolves.
#[test]
fn a_write_anywhere_in_the_session_invalidates_a_reading_handle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.h5");
    two_dataset_session(&path);

    let session = File::open_rw(&path).unwrap();
    let reader = session.dataset("log").unwrap();
    assert_eq!(reader.read_i32().unwrap().len(), 64);
    let retained = reader.chunk_cache_stats().cached_chunks();
    assert!(retained > 0, "the read should have retained chunks to lose");
    reader.reset_chunk_cache_stats();

    // A different dataset, through a handle of its own.
    session.dataset("other").unwrap().append(&[64i32]).unwrap();

    let stats = reader.chunk_cache_stats();
    assert_eq!(stats.invalidations(), retained as u64);
    assert_eq!(stats.cached_chunks(), 0);
    // An invalidation is not an eviction: no budget was exceeded.
    assert_eq!(stats.evictions(), 0);

    drop(reader);
    session.close().unwrap();
}

/// Resetting notices the pending invalidation first, so the chunks a write threw
/// away are charged to the window being closed rather than to the next one.
///
/// Without that, a caller who resets to measure one read would open the window
/// already owing an invalidation count from before it.
#[test]
fn resetting_charges_a_pending_invalidation_to_the_window_it_closes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.h5");
    two_dataset_session(&path);

    let session = File::open_rw(&path).unwrap();
    let reader = session.dataset("log").unwrap();
    assert_eq!(reader.read_i32().unwrap().len(), 64);
    assert!(reader.chunk_cache_stats().cached_chunks() > 0);

    session.dataset("other").unwrap().append(&[64i32]).unwrap();

    // The reset resolves before it zeroes, so the invalidation lands *inside*
    // the window being discarded and the fresh window starts empty.
    reader.reset_chunk_cache_stats();
    assert_eq!(reader.chunk_cache_stats().invalidations(), 0);

    drop(reader);
    session.close().unwrap();
}
