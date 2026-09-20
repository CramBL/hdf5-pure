//! Bounded-memory read-write backend (issue #147): reads, immediate appends,
//! internal batching, cache coherence, and the typed refusals for everything
//! mirror-only.

use hdf5_pure::{
    AttrValue, Error, File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MaxExtent,
    MemoryStrategy, MetadataCacheConfig, SyncPolicy,
};
use tempfile::tempdir;

/// Open with the bounded engine demanded rather than merely preferred.
///
/// `File::open_rw` would pick it for most of the files below anyway, but these
/// tests are *about* the bounded engine: asking for it explicitly means a file
/// that stops being bounded-editable fails here instead of quietly retargeting
/// the whole file at the mirror.
///
/// `SyncPolicy::OnClose` for the same reason the memory strategy is explicit:
/// what these tests assert is content and batching behaviour, not durability,
/// and the default `Always` costs one `fsync` per append. The policies write
/// byte-identical files (`tests/main/sync_policy.rs`), and close still barriers.
fn open_bounded(path: &std::path::Path) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose),
    )
}

/// Build a rank-1 unlimited chunked i32 dataset `d` seeded with `0..n`, with
/// optional deflate.
fn build(path: &std::path::Path, n: i32, chunk: u64, deflate: bool) {
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[chunk]);
    if deflate {
        ds.with_deflate(6);
    }
    b.write(path).unwrap();
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap()
}

#[test]
fn append_and_read_through_one_handle() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("a.h5");
    build(&p, 6, 4, false);
    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[6i32, 7, 8]).unwrap();
        // The appending handle observes the new length immediately.
        assert_eq!(ds.shape().unwrap(), vec![9]);
        assert_eq!(ds.read_i32().unwrap(), (0..9).collect::<Vec<_>>());
        ds.append(&[9i32]).unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..10).collect::<Vec<_>>());
    }
    // Scope the writer before re-opening: Windows file locks are mandatory.
    assert_eq!(read_i32(&p), (0..10).collect::<Vec<_>>());
}

#[test]
fn many_appends_across_calls_stay_o1() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("many.h5");
    build(&p, 0, 8, false);
    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let mut next = 0i32;
        for _ in 0..200 {
            let batch: Vec<i32> = (next..next + 5).collect();
            ds.append(&batch).unwrap();
            next += 5;
        }
        assert_eq!(ds.shape().unwrap(), vec![1000]);
    }
    assert_eq!(read_i32(&p), (0..1000).collect::<Vec<_>>());
}

#[test]
fn refetched_handle_observes_appends() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("refetch.h5");
    build(&p, 4, 4, false);
    let file = open_bounded(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[4i32, 5, 6, 7]).unwrap();
    let fresh = file.dataset("d").unwrap();
    assert_eq!(fresh.shape().unwrap(), vec![8]);
    assert_eq!(fresh.read_i32().unwrap(), (0..8).collect::<Vec<_>>());
}

#[test]
fn filtered_appends_start_and_end_at_any_length() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("filtered.h5");
    build(&p, 8, 4, true);
    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        // Chunk-aligned.
        ds.append(&[8i32, 9, 10, 11]).unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..12).collect::<Vec<_>>());
        // A length that is not a chunk multiple: adds a new partial chunk.
        ds.append(&[12i32]).unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..13).collect::<Vec<_>>());
        // Growing that partial chunk: the bounded engine reads and re-encodes
        // exactly the one trailing chunk, the same rule as open_rw.
        ds.append(&[13i32]).unwrap();
        ds.append(&[14i32, 15, 16]).unwrap();
        assert_eq!(ds.read_i32().unwrap(), (0..17).collect::<Vec<_>>());
    }
    assert_eq!(read_i32(&p), (0..17).collect::<Vec<_>>());
}

#[test]
fn large_append_batches_internally() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("large.h5");
    build(&p, 3, 256, false);
    // ~2.5 MiB of i32 in ONE call: far past the 1 MiB batch budget, unaligned
    // start (3), so the run exercises partial-tail fill + several whole-chunk
    // batches + a trailing remainder.
    let total = 655_360i32 + 7;
    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let batch: Vec<i32> = (3..total).collect();
        ds.append(&batch).unwrap();
        assert_eq!(ds.shape().unwrap(), vec![total as u64]);
    }
    let got = read_i32(&p);
    assert_eq!(got.len(), total as usize);
    assert!(got.iter().enumerate().all(|(i, &v)| v == i as i32));
}

/// The whole staged edit surface now runs on a bounded file: it is the same
/// engine as `open_rw`, differing only in how it holds the file's bytes (issue
/// #198). Each operation is exercised and its effect read back, so the test
/// fails if any of them silently no-ops rather than merely returning `Ok`.
#[test]
fn staged_surface_works_on_a_bounded_file() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("staged.h5");
    build(&p, 4, 4, false);
    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let root = file.root();

        // Transaction 1: a value overwrite and a new group.
        ds.write(&[9i32, 9, 9, 9]).unwrap();
        root.create_group("g").unwrap();
        let before = file.space_accounting().unwrap();
        assert_eq!(
            before.logical_size,
            std::fs::metadata(&p).unwrap().len(),
            "a bounded session must report the file's real size"
        );
        file.commit().unwrap();

        // Transaction 2: copy the (now overwritten) dataset. On its own, so the
        // copy reads the committed values rather than racing the overwrite.
        file.copy("d", "d2").unwrap();
        file.commit().unwrap();

        // Transaction 3: an attribute edit, a staged append that rebuilds the
        // chunk index rather than growing it in place, and a delete. The
        // attribute edit relocates its dataset's header, which the engine will
        // not combine with another edit to the same dataset, so it goes with
        // edits to *other* objects.
        ds.set_attr("units", AttrValue::String("m".into())).unwrap();
        let mut d2 = file.dataset("d2").unwrap();
        d2.append_staged(|b| {
            b.append_i32(&[7]);
        })
        .unwrap();
        file.root().delete("g").unwrap();
        file.commit().unwrap();

        // The delete freed the group's blocks into the session's own free list,
        // which a bounded session keeps like any other. Reading it back is what
        // distinguishes a working accounting from one that reports zeroes.
        let acct = file.space_accounting().unwrap();
        assert!(
            acct.reusable_free_bytes > 0,
            "a committed delete left nothing reusable: {acct:?}"
        );
        assert_eq!(
            acct.reusable_free_bytes,
            acct.reusable_free_space
                .iter()
                .map(|(_, len)| len)
                .sum::<u64>(),
            "the total must be the sum of the regions it describes"
        );
        file.close().unwrap();
    }

    let file = File::open(&p).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![9, 9, 9, 9]);
    assert_eq!(
        ds.attrs().unwrap().get("units").map(|v| format!("{v:?}")),
        Some(format!("{:?}", AttrValue::String("m".into())))
    );
    assert_eq!(
        file.dataset("d2").unwrap().read_i32().unwrap(),
        vec![9, 9, 9, 9, 7]
    );
    assert!(file.group("g").is_err(), "the deleted group must be gone");
}

#[test]
fn close_seals_writes_but_not_reads() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("close.h5");
    build(&p, 4, 4, false);
    let file = open_bounded(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[4i32]).unwrap();
    file.clone().close().unwrap();
    assert!(matches!(ds.append(&[5i32]), Err(Error::FileClosed)));
    // Reads through the surviving handle still work.
    assert_eq!(ds.read_i32().unwrap(), (0..5).collect::<Vec<_>>());
}

#[test]
fn bounded_open_takes_the_exclusive_lock() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("lock.h5");
    build(&p, 4, 4, false);
    let bounded = open_bounded(&p).unwrap();
    let err = File::open_rw(&p).unwrap_err();
    assert!(matches!(err, Error::FileLocked(_)), "got: {err:?}");
    drop(bounded);
    File::open_rw(&p).unwrap();
}

#[test]
fn userblock_file_is_refused_at_open() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("userblock.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[2]);
    b.write(&p).unwrap();
    let err = open_bounded(&p).unwrap_err();
    assert!(matches!(err, Error::EditUnsupported(_)), "got: {err:?}");
}

/// A file that persists its free space (non-paged) is now supported by the
/// bounded backend (issue #173): append in place, then `close` rewrites the
/// on-disk free-space managers so a reopen recovers the data and the strategy.
#[test]
fn persisted_free_space_file_appends_and_finalizes() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("persist.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&(0..10).collect::<Vec<i32>>())
        .with_shape(&[10])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.write(&p).unwrap();

    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(10..25).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    // The reopened file reads the full sequence and still records the persisting
    // strategy, and hdf5-pure recovers free space the finalize wrote on disk.
    let f = File::open(&p).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..25).collect::<Vec<_>>()
    );
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::FsmAggr));
    assert!(
        f.file_space_info().unwrap().persist,
        "the finalize keeps the persist flag set"
    );
    // The finalize freed the superseded extension + manager blocks, so the
    // reopened file recovers at least one persisted free section.
    assert!(
        !f.persisted_free_space().is_empty(),
        "expected persisted free sections after finalize"
    );
}

/// Several bounded appends over one held-open persisting file, then one close:
/// the finalize runs once and the reopened file reads the whole sequence.
#[test]
fn persisted_free_space_many_appends_one_finalize() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("persist_many.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 3);
    b.create_dataset("d")
        .with_i32_data(&[0])
        .with_shape(&[1])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[16]);
    b.write(&p).unwrap();

    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        for start in (1..200).step_by(20) {
            let end = (start + 20).min(200);
            ds.append(&(start..end).collect::<Vec<i32>>()).unwrap();
        }
        file.close().unwrap();
    }
    assert_eq!(read_i32(&p), (0..200).collect::<Vec<_>>());
}

/// A dropped bounded persisting session (no `close`, i.e. an unclean exit) still
/// leaves a file whose appended data is durable and readable — the finalize is
/// on a persist file finalizes it (rewrites the managers into canonical shape),
/// just like an explicit `close`.
#[test]
fn persisted_free_space_drop_finalizes() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("persist_drop.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&(0..8).collect::<Vec<i32>>())
        .with_shape(&[8])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.write(&p).unwrap();

    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(8..16).collect::<Vec<i32>>()).unwrap();
        // Drop without close: the Drop guard runs the finalize best-effort.
    }
    assert_eq!(read_i32(&p), (0..16).collect::<Vec<_>>());
    // The dropped-without-close file is finalized: managers are canonical, so a
    // reopen recovers persisted free space, exactly as after an explicit close.
    let f = File::open(&p).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::FsmAggr));
    assert!(
        !f.persisted_free_space().is_empty(),
        "a dropped handle finalizes like close"
    );
}

/// Opening a persist file and closing it without appending anything must not
/// grow the file — the finalize is skipped when nothing was appended.
#[test]
fn persisted_free_space_noop_close_does_not_grow() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("persist_noop.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&(0..8).collect::<Vec<i32>>())
        .with_shape(&[8])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.write(&p).unwrap();
    let before = std::fs::metadata(&p).unwrap().len();

    open_bounded(&p).unwrap().close().unwrap();
    assert_eq!(
        std::fs::metadata(&p).unwrap().len(),
        before,
        "a no-append close must not grow the file"
    );
    // Repeated open/close cycles also leave the size fixed.
    for _ in 0..3 {
        open_bounded(&p).unwrap().close().unwrap();
    }
    assert_eq!(std::fs::metadata(&p).unwrap().len(), before);
}

/// A paged file that does NOT persist its free space is refused by the bounded
/// backend (issue #173 Phase 2): without on-disk managers there is no record of
/// which pages are metadata vs raw, so bounded appends cannot keep the paging
/// segregated. A paged file that *does* persist is supported (see
/// `tests/main/paged_mutation.rs`).
#[test]
fn paged_non_persist_is_refused_at_open() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("paged_plain.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0)
        .with_file_space_page_size(4096);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[2]);
    b.write(&p).unwrap();
    let err = open_bounded(&p).unwrap_err();
    let Error::EditUnsupported(msg) = err else {
        panic!("paged non-persist should be refused with EditUnsupported, got: {err:?}");
    };
    // The guidance must be actionable: a paged file is grown in place only when it
    // persists its free space, so the message points at recreating with persist=true
    // rather than at another entry point. There is none to point at — the mirror
    // cannot commit such a file either, which is why `MemoryStrategy::Auto` does not
    // fall back for it.
    assert!(
        msg.contains("persist"),
        "refusal should guide toward persisted free space, got: {msg:?}"
    );
}

#[test]
fn metadata_cache_stays_coherent_across_appends() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("cache.h5");
    build(&p, 4, 4, false);
    let properties = FileAccessProperties::new()
        .with_memory_strategy(MemoryStrategy::Bounded)
        .with_metadata_cache(MetadataCacheConfig::new(256 * 1024));
    let file = File::open_rw_with_options(&p, properties).unwrap();
    let mut ds = file.dataset("d").unwrap();
    // Prime the metadata cache with the object-header windows.
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(ds.read_i32().unwrap(), vec![0, 1, 2, 3]);
    // The append patches the dataspace dimension in place; overlapping cached
    // windows must be invalidated so re-reads observe the new length.
    ds.append(&[4i32, 5]).unwrap();
    assert_eq!(ds.shape().unwrap(), vec![6]);
    assert_eq!(ds.read_i32().unwrap(), (0..6).collect::<Vec<_>>());
    let fresh = file.dataset("d").unwrap();
    assert_eq!(fresh.shape().unwrap(), vec![6]);
}

/// The coherence above is kept by dropping cached windows an append overwrites.
/// That is the one thing a read-only budget never does, and the figure that
/// tells a caller a larger budget will not help this workload (issue #353).
#[test]
fn appends_report_the_cached_windows_they_invalidate() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("mdc_invalidations.h5");
    build(&p, 4, 4, false);
    let properties = FileAccessProperties::new()
        .with_memory_strategy(MemoryStrategy::Bounded)
        .with_sync_policy(SyncPolicy::OnClose)
        .with_metadata_cache(MetadataCacheConfig::new(256 * 1024));
    let file = File::open_rw_with_options(&p, properties).unwrap();
    let mut ds = file.dataset("d").unwrap();

    assert_eq!(ds.read_i32().unwrap(), vec![0, 1, 2, 3]);
    let primed = file
        .metadata_cache_stats()
        .expect("a bounded session with a budget has a cache");
    assert!(primed.entries() > 0, "the read filled it: {primed:?}");
    assert_eq!(primed.invalidations(), 0, "nothing written yet");

    ds.append(&[4i32, 5]).unwrap();

    let after = file.metadata_cache_stats().unwrap();
    assert!(
        after.invalidations() > 0,
        "the append patched the dataspace under a cached window: {after:?}"
    );
    assert_eq!(ds.read_i32().unwrap(), (0..6).collect::<Vec<_>>());
}

/// A mirrored session holds the whole file, so there is no metadata cache in
/// front of it whatever the budget says.
#[test]
fn a_mirrored_session_reports_no_metadata_cache() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("mdc_mirrored.h5");
    build(&p, 4, 4, false);
    let file = File::open_rw_with_options(
        &p,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Mirrored)
            .with_sync_policy(SyncPolicy::OnClose)
            .with_metadata_cache(MetadataCacheConfig::new(256 * 1024)),
    )
    .unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(file.metadata_cache_stats(), None);
}

#[test]
fn reads_match_streaming_capabilities() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("reads.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[2]);
    let mut grp = b.create_group("grp");
    grp.create_dataset("nested")
        .with_f64_data(&[1.5, 2.5])
        .with_shape(&[2]);
    b.add_group(grp.finish());
    b.write(&p).unwrap();

    let file = open_bounded(&p).unwrap();
    // Groups, nested paths, and non-append datasets all read.
    assert_eq!(
        file.dataset("grp/nested").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5]
    );
    let grp = file.group("grp").unwrap();
    assert_eq!(grp.datasets().unwrap(), vec!["nested"]);
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn unaligned_filtered_multi_batch_append_lands_every_element() {
    // A filtered append onto a partial trailing chunk, larger than the internal
    // batch budget. Only the first batch may re-encode the trailing chunk: it
    // fills to the boundary, and every batch after it starts aligned. Each
    // batch is its own crash-atomic apply, so this is also the shape where a
    // batching mistake shows up as lost or duplicated elements.
    let dir = tempdir().unwrap();
    let p = dir.path().join("atomic.h5");
    build(&p, 257, 256, true); // one element past a chunk boundary
    const ADDED: i32 = 524_288; // ~2 MiB of i32, several batches
    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(257..257 + ADDED).collect::<Vec<i32>>())
            .unwrap();
        assert_eq!(ds.shape().unwrap(), vec![(257 + ADDED) as u64]);
    }
    assert_eq!(read_i32(&p), (0..257 + ADDED).collect::<Vec<_>>());

    // A raw append whose byte length is not a whole number of elements is
    // refused before any batch applies, so the file is untouched.
    let before = std::fs::read(&p).unwrap();
    {
        let file = open_bounded(&p).unwrap();
        let mut ds = file.dataset("d").unwrap();
        let mut bytes = vec![0u8; 2 * 1024 * 1024];
        bytes.push(0); // not a whole i32
        let err = ds.append_raw(&bytes).unwrap_err();
        assert!(
            matches!(&err, Error::AppendInPlaceUnsupported(m) if m.contains("whole number of elements")),
            "got: {err:?}"
        );
    }
    assert_eq!(std::fs::read(&p).unwrap(), before);
}

#[test]
fn chunk_introspection_works_on_bounded_files() {
    // chunks() walks the chunk index through the file source; on a bounded
    // (and mirror) file that must go through the engine's store, not the
    // empty borrowed view.
    let dir = tempdir().unwrap();
    let p = dir.path().join("chunks.h5");
    build(&p, 8, 4, false);
    let file = open_bounded(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    let chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    ds.append(&[8i32, 9]).unwrap();
    let chunks = file.dataset("d").unwrap().chunks().unwrap();
    assert_eq!(chunks.len(), 3);
}

/// A bounded commit that frees a run reaching end-of-file truncates the file,
/// which is the one `FileImage` primitive the bounded backing never exercised
/// before it shared the edit engine: the old bounded store had no `truncate` at
/// all. The mirror side has a dozen tests for this; this is the bounded one.
///
/// Both commits run in one session, because that is what makes the freed run
/// reach the end: reuse draws only on regions this session freed.
#[test]
fn a_bounded_commit_truncates_when_the_freed_run_reaches_the_end() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("truncate.h5");
    build(&p, 4, 4, false);

    let (grown, shrunk) = {
        let file = open_bounded(&p).unwrap();
        file.root()
            .create_dataset("big", |b| {
                b.with_i32_data(&(0..2000).collect::<Vec<i32>>())
                    .with_shape(&[2000]);
            })
            .unwrap();
        file.commit().unwrap();
        let grown = std::fs::metadata(&p).unwrap().len();

        file.root().delete("big").unwrap();
        file.commit().unwrap();
        let shrunk = std::fs::metadata(&p).unwrap().len();
        file.close().unwrap();
        (grown, shrunk)
    };

    assert!(
        shrunk < grown,
        "deleting the trailing dataset did not shrink the file: {grown} -> {shrunk}"
    );
    // The surviving dataset is intact, so the truncate cut slack rather than data.
    assert_eq!(read_i32(&p), (0..4).collect::<Vec<_>>());
    assert!(File::open(&p).unwrap().dataset("big").is_err());
}
