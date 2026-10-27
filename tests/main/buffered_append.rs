//! `Dataset::buffered_appender` (issue #262): appended elements are held in
//! memory and written a whole chunk at a time, so a caller appending less than
//! a chunk per call writes once per chunk rather than once per call. Since
//! issue #393 every write it makes is the immediate in-place one, filtered or
//! not, so the type buys write *frequency* and nothing else.

use rstest::rstest;

use hdf5_pure::DeflateLevel;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileLocking;
use hdf5_pure::{
    AppendOptions, AttrValue, Error, File, FileBuilder, Object, OwnedBufferedAppender, ScaleOffset,
};

use test_util_hdf5::dataset::{self, Filter, Unlimited};

const SHUFFLE_DEFLATE: &[Filter] = &[Filter::Shuffle, Filter::Deflate(DeflateLevel::Four)];

// ---- what the buffer buys ----------------------------------------------------

/// Compares complete file bytes to detect writes within retained aggregate reservations.
#[test]
fn a_call_that_does_not_complete_a_chunk_does_not_touch_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    let chunk = 16u64;
    Unlimited::<i32>::new("d", &[], chunk)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);

    let session = File::open_rw_with_options(
        &p,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    assert_eq!(app.chunk_elements(), chunk);

    // Four elements per call complete a sixteen-element chunk on calls 4, 8, and 12.
    let mut images = vec![std::fs::read(&p).unwrap()];
    for i in 0..12i32 {
        app.append(&[i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
            .unwrap();
        images.push(std::fs::read(&p).unwrap());
    }
    let wrote_on: Vec<usize> = (1..images.len())
        .filter(|&i| images[i] != images[i - 1])
        .collect();
    assert_eq!(
        wrote_on,
        vec![4, 8, 12],
        "expected file changes only where a chunk completed"
    );
    assert_eq!(app.buffered_elements(), 0);

    app.finish().unwrap();
    drop(ds);
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..48).collect::<Vec<_>>()
    );
}

/// The invariant the whole type rests on: after every call the dataset's
/// on-disk length is a whole number of chunks, and written + buffered accounts
/// for every element appended. Chunk alignment is what keeps each later write on
/// the cheap in-place path; nothing else observes it, so a write prefix that
/// failed to round down to a chunk boundary would go unnoticed while every
/// round-trip test still passed.
///
/// The batch/chunk pair matters. 5 into 16 puts the completed-chunk boundary
/// *inside* a call (call 4 carries elements 15..20), so a prefix length that
/// took the whole buffer would leave 20 on disk against a chunk of 16. A pair
/// that divides evenly — 4 into 16 — lands on the boundary either way and cannot
/// tell the two apart.
#[test]
fn the_on_disk_length_stays_chunk_aligned_after_every_call() {
    let dir = tempfile::tempdir().unwrap();
    for &filtered in &[true, false] {
        let p = dir.path().join(format!("aligned_{filtered}.h5"));
        let chunk = 16u64;
        Unlimited::<i32>::new("d", &[], chunk)
            .filters(if filtered { SHUFFLE_DEFLATE } else { &[] })
            .pure_create(&p);

        let session = File::open_rw(&p).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        for i in 0..12i32 {
            let lo = i * 5;
            app.append(&(lo..lo + 5).collect::<Vec<i32>>()).unwrap();
            // Read the length through a second handle: the appender holds the
            // first one, and this is what any other reader of the file sees.
            let on_disk = session.dataset("d").unwrap().shape().unwrap()[0];
            assert_eq!(
                on_disk % chunk,
                0,
                "call {i} (filtered={filtered}): on-disk length {on_disk} is not chunk-aligned"
            );
            assert_eq!(
                on_disk + app.buffered_elements(),
                (lo + 5) as u64,
                "call {i} (filtered={filtered}): written + buffered lost elements"
            );
        }
        app.finish().unwrap();
        drop(ds);
        drop(session);
        assert_eq!(
            dataset::read_pure::<i32>(&p, "d"),
            (0..60).collect::<Vec<_>>()
        );
    }
}

/// No write the appender makes commits, on a filtered dataset resumed mid-chunk
/// least of all (issue #393). The observable is an unrelated edit staged before
/// the writes: if any of them committed, that edit would have been published as
/// a side effect and `has_staged_edits` would have gone false.
#[test]
fn an_unaligned_filtered_flush_never_commits() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p); // filtered, and sitting on a partial trailing chunk

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();

    // Staged beside a live appender on an unaligned filtered dataset: accepted,
    // because the appender has no commit of its own left to protect.
    session
        .root()
        .create_dataset("other", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();

    // The first write grows the partial trailing chunk; the flush leaves a new
    // partial tail for the next write to grow again.
    app.append(&(10..30i32).collect::<Vec<_>>()).unwrap();
    assert_eq!(session.dataset("d").unwrap().shape().unwrap(), vec![24]);
    app.flush().unwrap();
    assert_eq!(session.dataset("d").unwrap().shape().unwrap(), vec![30]);
    for i in 0..6i32 {
        app.append(&(30 + i * 8..38 + i * 8).collect::<Vec<_>>())
            .unwrap();
    }
    app.flush().unwrap();
    assert!(
        session.has_staged_edits(),
        "a buffered write published the caller's staged edit"
    );

    drop(app);
    drop(ds);
    session.commit().unwrap();
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..78).collect::<Vec<_>>()
    );
}

/// Every element still shows up, across filtered and unfiltered datasets,
/// aligned and unaligned starting lengths, and batch sizes above, below, and
/// equal to the chunk length.
#[test]
fn every_shape_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    for &filtered in &[true, false] {
        for &(start, chunk, batch, calls) in &[
            (0i32, 4u64, 1usize, 13usize), // one element at a time
            (8, 4, 3, 7),                  // aligned start, sub-chunk batches
            (6, 4, 5, 9),                  // unaligned start
            (5, 8, 8, 4),                  // unaligned start, batch == chunk
            (0, 3, 7, 5),                  // batch larger than the chunk
            (0, 64, 100, 9),               // several chunks per call
        ] {
            let p = dir
                .path()
                .join(format!("r_{filtered}_{start}_{chunk}_{batch}_{calls}.h5"));
            Unlimited::new("d", &(0..start).collect::<Vec<i32>>(), chunk)
                .filters(if filtered { SHUFFLE_DEFLATE } else { &[] })
                .pure_create(&p);
            let total = start + (batch * calls) as i32;
            {
                let session = File::open_rw(&p).unwrap();
                let mut ds = session.dataset("d").unwrap();
                let mut app = ds.buffered_appender().unwrap();
                for i in 0..calls {
                    let lo = start + (i * batch) as i32;
                    app.append(&(lo..lo + batch as i32).collect::<Vec<i32>>())
                        .unwrap();
                }
                app.finish().unwrap();
            }
            assert_eq!(
                dataset::read_pure::<i32>(&p, "d"),
                (0..total).collect::<Vec<_>>(),
                "filtered={filtered} start={start} chunk={chunk} batch={batch} calls={calls}"
            );
        }
    }
}

/// A filtered dataset appended by a length that is not a chunk multiple, over
/// and over: the buffer keeps the on-disk length chunk-aligned so no write has
/// to re-encode a trailing chunk.
#[test]
fn a_filtered_dataset_takes_any_append_length() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 100)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);
    {
        let session = File::open_rw(&p).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        for i in 0..7i32 {
            app.append(&(i * 37..(i + 1) * 37).collect::<Vec<i32>>())
                .unwrap();
        }
        app.finish().unwrap();
    }
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..259).collect::<Vec<_>>()
    );
}

// ---- durability boundaries ---------------------------------------------------

/// Buffered elements are not in the file, and the ones already written are —
/// which is the whole contract, and the thing a caller has to be able to rely
/// on when a crash takes the buffer.
#[test]
fn the_file_holds_exactly_what_was_written_not_what_was_buffered() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append(&(0..8i32).collect::<Vec<_>>()).unwrap(); // one whole chunk: written
    app.append(&[8i32, 9, 10]).unwrap(); // buffered
    assert_eq!(app.buffered_elements(), 3);
    drop(app);
    drop(ds);
    drop(session);
    // The dataset is the written prefix, not the buffered whole. (`drop` on the
    // appender flushes, so the drops above are ordered: appender, handle,
    // session — this asserts the state *between* those flushes is coherent.)
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..11).collect::<Vec<_>>()
    );
}

/// `flush` publishes the buffered tail immediately, without consuming the
/// appender, and appending afterwards still works.
#[test]
fn flush_publishes_the_tail_and_the_appender_stays_usable() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append(&[0i32, 1, 2]).unwrap();
    app.flush().unwrap();
    assert_eq!(app.buffered_elements(), 0);
    // Readable through a second handle on the same session, so it really is in
    // the file and not merely in this handle's view.
    assert_eq!(
        session.dataset("d").unwrap().read_i32().unwrap(),
        vec![0, 1, 2]
    );
    // The dataset now sits on a partial chunk; the appender recovers from that
    // itself rather than refusing.
    app.append(&(3..20i32).collect::<Vec<_>>()).unwrap();
    app.finish().unwrap();
    drop(ds);
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..20).collect::<Vec<_>>()
    );
}

/// Dropping without `finish` still flushes: forgetting the call must not lose
/// data silently, even though it cannot report an error.
#[test]
fn dropping_the_appender_flushes_its_buffer() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 16)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);
    {
        let session = File::open_rw(&p).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        app.append(&(0..5i32).collect::<Vec<_>>()).unwrap();
        // no finish()
    }
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..5).collect::<Vec<_>>()
    );
}

// ---- resuming an unaligned log ----------------------------------------------

/// Resuming a filtered log left on a partial trailing chunk: the appender lands
/// it back on a chunk boundary with its first write, and an immediate
/// `Dataset::append` afterwards picks it up from there.
#[test]
fn resuming_an_unaligned_filtered_log_lands_back_on_a_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p); // 10 of 8 = a partial trailing chunk

    {
        let session = File::open_rw(&p).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        for i in 0..6i32 {
            app.append(&(10 + i * 5..15 + i * 5).collect::<Vec<_>>())
                .unwrap();
        }
        // 10 + 30 = 40 = five whole chunks, so the buffer is empty and the file
        // is chunk-aligned again.
        assert_eq!(app.buffered_elements(), 0);
        app.finish().unwrap();
    }
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..40).collect::<Vec<_>>()
    );

    // Chunk-aligned now, which is exactly what the immediate path requires.
    {
        let session = File::open_rw(&p).unwrap();
        session
            .dataset("d")
            .unwrap()
            .append(&(40..48i32).collect::<Vec<_>>())
            .unwrap();
    }
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..48).collect::<Vec<_>>()
    );
}

/// The borrowed handle keeps working across the writes, and reads the length
/// and values they published.
#[test]
fn the_handle_reads_what_the_appender_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p); // filtered, sitting on a partial trailing chunk

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    {
        let mut app = ds.buffered_appender().unwrap();
        app.append(&(10..30i32).collect::<Vec<_>>()).unwrap();
        app.finish().unwrap();
    }
    assert_eq!(ds.shape().unwrap(), vec![30]);
    assert_eq!(ds.read_i32().unwrap(), (0..30).collect::<Vec<_>>());
    assert_eq!(ds.chunk_shape().unwrap(), Some(vec![8]));
}

/// An appender over a filtered dataset with a partial trailing chunk is made
/// while unrelated edits are staged, and both go through: it has no commit of
/// its own, so there is nothing for the staged set to collide with (issue #393).
#[test]
fn an_appender_on_an_unaligned_filtered_dataset_is_made_beside_staged_edits() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    session
        .root()
        .create_dataset("other", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();

    let mut ds = session.dataset("d").unwrap();
    {
        let mut app = ds.buffered_appender().unwrap();
        app.append(&(10..30i32).collect::<Vec<_>>()).unwrap();
        app.finish().unwrap();
    }
    // The staged dataset is still staged: no write published it.
    assert!(session.has_staged_edits());
    drop(ds);
    session.commit().unwrap();
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..30).collect::<Vec<_>>()
    );
    assert_eq!(
        File::open(&p)
            .unwrap()
            .dataset("other")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![1, 2, 3]
    );
}

// ---- refusals ----------------------------------------------------------------

/// Eligibility is reported when the appender is made, not on the first write,
/// so a caller learns immediately rather than after buffering a run of data.
#[test]
fn an_ineligible_dataset_is_refused_at_construction() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]); // contiguous
    b.write(&p).unwrap();

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let err = ds
        .buffered_appender()
        .expect_err("contiguous is ineligible");
    assert!(
        matches!(err, Error::AppendInPlaceUnsupported(_)),
        "got: {err:?}"
    );
}

/// A read-only file has nothing to append to.
#[test]
fn a_read_only_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 4).pure_create(&p);
    let file = File::open(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    assert!(matches!(
        ds.buffered_appender().unwrap_err(),
        Error::ReadOnly
    ));
}

#[test]
fn a_refused_append_buffers_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append(&[0i32, 1, 2]).unwrap();
    assert_eq!(app.buffered_elements(), 3);

    session.close().unwrap();

    app.append(&(3..9i32).collect::<Vec<_>>())
        .expect_err("a closed session cannot be appended to");
    assert_eq!(
        app.buffered_elements(),
        3,
        "a refused append left its elements in the buffer"
    );
}

/// The guarantee that replaced the old drop-time data-loss trap: a staged edit
/// that would stop a live appender from flushing is refused at the call that
/// makes it. Before this, `set_attr` succeeded, the appender's drop-time flush
/// was then refused by that very edit, and the accepted elements vanished with
/// no error reported anywhere.
#[test]
fn a_staged_edit_on_a_dataset_with_a_live_appender_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append(&[0i32, 1, 2]).unwrap(); // accepted, buffered, only `app` can write it

    let err = session
        .dataset("d")
        .unwrap()
        .set_attr("units", AttrValue::I32(1))
        .expect_err("staging onto a dataset with a live appender must be refused");
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("live buffered appender")),
        "got: {err:?}"
    );
    // Deleting it is the same conflict.
    assert!(session.root().delete("d").is_err());

    // So the drop-time flush still works and the elements are not lost.
    drop(app);
    drop(ds);
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..3).collect::<Vec<_>>()
    );
}

/// An edit naming a *different* object is not blocked: the claim refuses what
/// would break the appender, not everything.
#[test]
fn an_unrelated_staged_edit_is_allowed_beside_a_live_appender() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append(&[0i32, 1, 2]).unwrap();

    session
        .root()
        .create_dataset("other", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();

    app.finish().unwrap();
    drop(ds);
    session.commit().unwrap();
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..3).collect::<Vec<_>>()
    );
    let f = File::open(&p).unwrap();
    assert_eq!(
        f.dataset("other").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

/// An unrelated staged edit is accepted beside a live appender on a filtered,
/// unaligned dataset — the shape that used to block *every* staged edit while a
/// realignment commit was owed (issue #393).
#[test]
fn an_unrelated_staged_edit_is_allowed_beside_an_unaligned_filtered_appender() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..10).collect::<Vec<i32>>(), 8)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p); // filtered and unaligned

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();

    session
        .root()
        .create_dataset("unrelated", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();

    app.append(&(10..30i32).collect::<Vec<_>>()).unwrap();
    app.finish().unwrap();
    drop(ds);
    session.commit().unwrap();
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..30).collect::<Vec<_>>()
    );
}

/// Two appenders on one dataset would interleave their buffers a chunk at a
/// time, so the second is refused — and the claim is released when the first
/// goes, so a later one works.
#[test]
fn a_second_appender_on_the_same_dataset_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut first = session.dataset("d").unwrap();
    let app = first.buffered_appender().unwrap();
    let mut second = session.dataset("d").unwrap();
    let err = second
        .buffered_appender()
        .expect_err("a second appender on one dataset must be refused");
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("already has a live buffered appender")),
        "got: {err:?}"
    );

    drop(app);
    drop(first);
    let mut app = second.buffered_appender().unwrap();
    app.append(&[0i32, 1, 2]).unwrap();
    app.finish().unwrap();
    drop(second);
    drop(session);
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..3).collect::<Vec<_>>()
    );
}

/// A SWMR writer requires the *appended* length to be chunk-aligned too, so the
/// one write this type exists to make — the partial trailing chunk — can never
/// succeed there. Refused when the appender is constructed, rather than by a
/// `finish` that always fails and a drop that discards the buffer silently.
#[test]
fn a_swmr_writer_is_refused_at_construction() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 8).pure_create(&p); // unfiltered and aligned: SWMR-eligible otherwise

    let file = File::open_swmr_writer(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    let err = ds.buffered_appender().unwrap_err();
    assert!(
        matches!(&err, Error::SwmrAppendUnsupported(m) if m.contains("partial trailing chunk")),
        "got: {err:?}"
    );
    // The SWMR writer itself still appends whole chunks.
    ds.append(&(8..16i32).collect::<Vec<_>>()).unwrap();
    drop(ds);
    file.close().unwrap();
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..16).collect::<Vec<_>>()
    );
}

/// A failed write leaves the file holding exactly the elements that landed and
/// the appender holding the ones that did not, rather than losing them.
#[test]
fn a_failed_write_keeps_the_unwritten_elements() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 4).pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    // f64 into an i32 dataset: refused by the write, not by the buffer.
    app.append(&[1.5f64, 2.5, 3.5, 4.5]).unwrap_err();
    assert_eq!(app.buffered_elements(), 8); // 32 bytes / 4-byte elements
    assert_eq!(app.unwritten().len(), 32);
    // Poisoned: further use is refused rather than half-working.
    assert!(app.append(&[1i32]).is_err());
    drop(app);
    drop(ds);
    drop(session);
    assert!(
        dataset::read_pure::<i32>(&p, "d").is_empty(),
        "a refused append wrote data"
    );
}

/// `append_raw` admits a trailing partial element so a byte-oriented caller can
/// split wherever it likes; the halves join up and the pair writes as one.
#[test]
fn append_raw_completes_an_element_split_across_calls() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 4).pure_create(&p);
    {
        let session = File::open_rw(&p).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        let bytes: Vec<u8> = (0..6i32).flat_map(i32::to_le_bytes).collect();
        app.append_raw(&bytes[..10]).unwrap(); // 2.5 elements
        assert_eq!(app.buffered_elements(), 2);
        app.append_raw(&bytes[10..]).unwrap();
        app.finish().unwrap();
    }
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..6).collect::<Vec<_>>()
    );
}

/// Flushing a buffer that ends mid-element is refused rather than writing a
/// truncated value or padding one.
#[test]
fn a_partial_trailing_element_is_refused_at_the_flush() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 4).pure_create(&p);
    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append_raw(&[1u8, 2, 3]).unwrap();
    let err = app.flush().unwrap_err();
    assert!(
        matches!(&err, Error::AppendInPlaceUnsupported(m) if m.contains("whole number of elements")),
        "got: {err:?}"
    );
}

/// `discard` is the only way to abandon appended elements, since dropping
/// writes them. Elements an earlier call already wrote stay in the file.
#[test]
fn discard_abandons_the_buffer_but_not_what_was_written() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 4)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);
    {
        let session = File::open_rw(&p).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        app.append(&(0..6i32).collect::<Vec<_>>()).unwrap(); // writes 0..4, buffers 4,5
        assert_eq!(app.buffered_elements(), 2);
        let abandoned = app.discard();
        assert_eq!(abandoned.len(), 8); // two i32
    }
    assert_eq!(
        dataset::read_pure::<i32>(&p, "d"),
        (0..4).collect::<Vec<_>>()
    );
}

/// A `BufferedAppender`'s writes draw on freed space like any other immediate
/// append (issue #349). It reaches the engine through `Dataset::append`, so this
/// is true by construction — but the fix's changelog entry names this type, and
/// a named behaviour is worth an assertion.
#[test]
fn a_buffered_appender_writes_into_freed_space() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 1024).pure_create(&p);
    let payload: Vec<i32> = (0..16384).collect();

    let session = File::open_rw(&p).unwrap();
    // A dataset to vacate, and a live one above it so the deletion leaves a hole
    // rather than a trailing run the commit truncates away.
    session
        .root()
        .create_dataset("scratch", |b| {
            b.with_i32_data(&payload);
        })
        .unwrap();
    session
        .root()
        .create_dataset("ceiling", |b| {
            b.with_i32_data(&[9, 9, 9]);
        })
        .unwrap();
    session.commit().unwrap();
    session.root().delete("scratch").unwrap();
    session.commit().unwrap();

    let before = session
        .reusable_space_stats()
        .unwrap()
        .reusable_free_bytes();
    assert!(
        before >= payload.len() as u64 * 4,
        "expected a sizable hole"
    );
    {
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        for batch in payload.chunks(100) {
            app.append(batch).unwrap();
        }
        app.finish().unwrap();
    }
    let after = session
        .reusable_space_stats()
        .unwrap()
        .reusable_free_bytes();
    assert!(
        after < before / 2,
        "the buffered appender should have spent the hole ({before} -> {after})"
    );
    session.close().unwrap();

    assert_eq!(dataset::read_pure::<i32>(&p, "d"), payload);
    assert_eq!(
        File::open(&p)
            .unwrap()
            .dataset("ceiling")
            .unwrap()
            .read_i32()
            .unwrap(),
        [9, 9, 9]
    );
}

// ---- a lossy pipeline -------------------------------------------------------

/// A chunk-aligned, lossy (float D-scale scale-offset) dataset of `n` elements,
/// chunked four at a time. The values are clear of the default fill value's
/// tolerance window, so each is stored as itself.
fn build_lossy(path: &std::path::Path, n: usize) {
    let data: Vec<f64> = (0..n).map(|i| 1.05 + i as f64 * 0.1).collect();
    Unlimited::new("d", &data, 4)
        .filters(&[Filter::ScaleOffset(ScaleOffset::FloatDScale(1))])
        .pure_create(path);
}
/// A mid-stream `flush` on a lossy dataset is refused rather than left to strand
/// the appender one call later.
///
/// Writing a partial tail is legal — it costs no re-encoding, since the length
/// was chunk-aligned — but the *next* write has to grow that tail, and a lossy
/// pipeline cannot: it fails, poisons the appender, and the caller loses elements
/// at a call that did nothing wrong. So the refusal is moved to the flush, which
/// is where the caller can still act on it, and the buffer is left whole.
#[test]
fn a_partial_flush_on_a_lossy_dataset_is_refused_with_the_buffer_intact() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_lossy(&p, 4);
    let committed = dataset::read_pure::<f64>(&p, "d");

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append(&[9.05f64, 9.15]).unwrap();
    let err = app
        .flush()
        .expect_err("a lossy dataset must not be left off a chunk boundary mid-stream");
    assert!(
        matches!(&err, Error::AppendInPlaceUnsupported(m) if m.contains("lossy")),
        "got: {err:?}"
    );
    // Refused before any write, so nothing landed and nothing was lost.
    assert_eq!(app.buffered_elements(), 2);
    assert_eq!(app.unwritten().len(), 16);
    assert_eq!(
        session.dataset("d").unwrap().read_f64().unwrap().len(),
        4,
        "the refused flush wrote elements"
    );

    // The appender is usable: completing the chunk writes it, and a flush that
    // lands on a boundary is accepted.
    app.append(&[9.25f64, 9.35]).unwrap();
    app.flush().unwrap();
    assert_eq!(app.buffered_elements(), 0);

    // `finish` is the one flush that may leave a partial tail: no write follows
    // it, so there is nothing left to refuse.
    app.append(&[9.45f64]).unwrap();
    app.finish().unwrap();
    drop(ds);
    session.close().unwrap();

    let back = dataset::read_pure::<f64>(&p, "d");
    assert_eq!(back.len(), 9);
    assert_eq!(
        back[..4],
        committed[..],
        "an already-committed chunk was re-encoded"
    );
    for (got, want) in back[4..]
        .iter()
        .zip([9.05f64, 9.15, 9.25, 9.35, 9.45].iter())
    {
        assert!((got - want).abs() < 0.06, "{got} != {want}");
    }
}

/// Dropping without `finish` writes the partial tail, as it always has: a drop is
/// terminal, so the refusal a live `flush` raises does not apply to it — and
/// applying it would discard the elements instead of landing them.
#[test]
fn dropping_a_lossy_appender_still_writes_its_partial_tail() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_lossy(&p, 4);
    {
        let session = File::open_rw(&p).unwrap();
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        app.append(&[9.05f64, 9.15]).unwrap();
        // no finish()
        drop(app);
        drop(ds);
        session.close().unwrap();
    }
    assert_eq!(dataset::read_pure::<f64>(&p, "d").len(), 6);
}

#[rstest]
#[case::empty(&[])]
#[case::whole_element(&[1, 0, 0, 0])]
#[case::split_element(&[1, 0])]
fn closed_session_rejects_buffered_operations_without_accepting_bytes(#[case] prior: &[u8]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let mut appender = dataset.buffered_appender().unwrap();
    appender.append_raw(prior).unwrap();
    file.close().unwrap();
    let error = appender.append::<i32>(&[]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.append(&[2i32]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.append_raw(&[]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.append_raw(&[0]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.flush().unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    assert_eq!(appender.unwritten(), prior);
    assert_eq!(appender.discard(), prior);
    drop(dataset);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), Vec::<i32>::new());
}

#[rstest]
#[case::empty(&[])]
#[case::sub_chunk(&[1, 0, 0, 0])]
fn closed_session_rejects_buffered_finish(#[case] prior: &[u8]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed_finish.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let mut appender = dataset.buffered_appender().unwrap();
    appender.append_raw(prior).unwrap();
    file.close().unwrap();
    let error = appender.finish().unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    drop(dataset);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), Vec::<i32>::new());
}

#[test]
fn an_owned_appender_retains_the_session_without_a_lifetime_parameter() {
    struct Log {
        appender: OwnedBufferedAppender,
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retained.h5");
    Unlimited::<i32>::new("d", &[], 4)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut log = Log {
        appender: file.dataset("d").unwrap().into_buffered_appender().unwrap(),
    };
    drop(file);
    log.appender.append(&[0i32, 1, 2, 3, 4, 5]).unwrap();
    assert_eq!(log.appender.chunk_elements(), 4);
    assert_eq!(log.appender.buffered_elements(), 2);
    assert_eq!(log.appender.unwritten(), i32_bytes(&[4, 5]));
    assert_eq!(
        format!("{:?}", log.appender),
        "OwnedBufferedAppender { chunk_elements: 4, buffered_elements: 2, lossy_filters: false, poisoned: false }"
    );
    log.appender.finish().unwrap();
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), [0, 1, 2, 3, 4, 5]);
}

#[test]
fn owned_automatic_writes_end_at_chunk_boundaries_and_preserve_a_split_element() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("split_owned.h5");
    Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut appender = file
        .dataset("d")
        .unwrap()
        .into_buffered_appender_with_options(AppendOptions::new())
        .unwrap();
    let bytes = i32_bytes(&[0, 1, 2, 3, 4, 5]);
    appender.append_raw(&bytes[..19]).unwrap();
    let observer = file.dataset("d").unwrap();
    assert_eq!(observer.shape().unwrap(), [4]);
    assert_eq!(observer.read_i32().unwrap(), [0, 1, 2, 3]);
    assert_eq!(appender.unwritten(), &bytes[16..19]);
    assert_eq!(appender.buffered_elements(), 0);
    assert_append_rejection(
        appender.flush().unwrap_err(),
        "the buffered byte length is not a whole number of elements; append the rest of \
         the trailing element before flushing",
    );
    assert_eq!(appender.unwritten(), &bytes[16..19]);
    appender.append_raw(&bytes[19..]).unwrap();
    assert_eq!(appender.unwritten(), &bytes[16..]);
    appender.flush().unwrap();
    assert_eq!(appender.unwritten(), Vec::<u8>::new());
    assert_eq!(observer.read_i32().unwrap(), [0, 1, 2, 3, 4, 5]);
    appender.finish().unwrap();
    drop(observer);
    drop(file);
}

#[rstest]
#[case::owned_claim(true)]
#[case::borrowed_claim(false)]
fn mixed_ownership_conflicts_preserve_the_first_claim(#[case] owned_first: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_claims.h5");
    Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut borrowed = file.dataset("d").unwrap();
    if owned_first {
        let mut first = file.dataset("d").unwrap().into_buffered_appender().unwrap();
        assert_appender_conflict(borrowed.buffered_appender().unwrap_err());
        assert_appender_conflict(
            file.dataset("d")
                .unwrap()
                .into_buffered_appender()
                .unwrap_err(),
        );
        first.append(&[0i32, 1]).unwrap();
        first.finish().unwrap();
    } else {
        let mut first = borrowed.buffered_appender().unwrap();
        assert_appender_conflict(
            file.dataset("d")
                .unwrap()
                .into_buffered_appender()
                .unwrap_err(),
        );
        assert_appender_conflict(
            file.dataset("d")
                .unwrap()
                .into_buffered_appender()
                .unwrap_err(),
        );
        first.append(&[0i32, 1]).unwrap();
        first.finish().unwrap();
    }
    let mut replacement = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    replacement.append(&[2i32]).unwrap();
    replacement.finish().unwrap();
    assert_eq!(borrowed.read_i32().unwrap(), [0, 1, 2]);
    drop(borrowed);
    drop(file);
}

#[rstest]
#[case::finish(OwnedTermination::Finish, &[0, 1, 2, 3, 4, 5])]
#[case::discard(OwnedTermination::Discard, &[0, 1, 2, 3])]
#[case::drop(OwnedTermination::Drop, &[0, 1, 2, 3, 4, 5])]
#[case::failed_finish(OwnedTermination::IncompleteFinish, &[0, 1, 2, 3])]
fn owned_terminal_operations_release_the_claim(
    #[case] termination: OwnedTermination,
    #[case] expected: &[i32],
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("owned_terminal.h5");
    Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    appender.append(&[0i32, 1, 2, 3, 4, 5]).unwrap();
    assert_eq!(appender.unwritten(), i32_bytes(&[4, 5]));
    match termination {
        OwnedTermination::Discard => assert_eq!(appender.discard(), i32_bytes(&[4, 5])),
        OwnedTermination::Drop => drop(appender),
        OwnedTermination::Finish => appender.finish().unwrap(),
        OwnedTermination::IncompleteFinish => {
            appender.append_raw(&[6]).unwrap();
            assert_append_rejection(
                appender.finish().unwrap_err(),
                "the buffered byte length is not a whole number of elements; append the rest of \
                 the trailing element before flushing",
            );
        }
    }
    let mut observer = file.dataset("d").unwrap();
    observer.buffered_appender().unwrap().finish().unwrap();
    assert_eq!(observer.read_i32().unwrap(), expected);
    drop(observer);
    drop(file);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);
}

#[test]
fn an_owned_appender_retains_its_buffer_and_claim_across_an_unrelated_commit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retained_commit.h5");
    Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    appender.append(&[0i32, 1]).unwrap();
    file.root()
        .create_dataset("other", |builder| {
            builder.with_i32_data(&[7, 8]);
        })
        .unwrap();
    assert!(file.has_staged_edits());
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    assert_eq!(appender.unwritten(), i32_bytes(&[0, 1]));
    assert_eq!(file.dataset("other").unwrap().read_i32().unwrap(), [7, 8]);
    assert_appender_conflict(
        file.dataset("d")
            .unwrap()
            .into_buffered_appender()
            .unwrap_err(),
    );
    assert_edit_claim(file.root().set_attr("tag", AttrValue::I32(1)).unwrap_err());
    appender.append(&[2i32, 3, 4]).unwrap();
    appender.finish().unwrap();
    drop(file);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), [0, 1, 2, 3, 4]);
    assert_eq!(dataset::read_pure::<i32>(&path, "other"), [7, 8]);
}

#[test]
fn owned_appender_writes_leave_unrelated_edits_staged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("owned_staged.h5");
    Unlimited::new("d", &[0i32, 1], 4)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    file.root()
        .create_dataset("other", |builder| {
            builder.with_i32_data(&[9]);
        })
        .unwrap();
    appender.append(&[2i32, 3, 4, 5, 6]).unwrap();
    appender.flush().unwrap();
    appender.finish().unwrap();
    assert!(file.has_staged_edits());
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        [0, 1, 2, 3, 4, 5, 6]
    );
    file.commit().unwrap();
    drop(file);
    assert_eq!(dataset::read_pure::<i32>(&path, "other"), [9]);
}

#[test]
fn an_owned_address_only_handle_preserves_session_restrictions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("owned_reference.h5");
    let mut builder = FileBuilder::new();
    Unlimited::<i32>::new("d", &[], 4).add_to(&mut builder);
    Unlimited::<i32>::new("other", &[], 4).add_to(&mut builder);
    builder.create_dataset("refs").with_path_references(&["d"]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let object = file
        .dataset("refs")
        .unwrap()
        .dereference()
        .unwrap()
        .remove(0);
    let Object::Dataset(dataset) = object else {
        panic!("expected dataset reference, got {object:?}");
    };
    let mut appender = dataset.into_buffered_appender().unwrap();
    appender.append(&[1i32, 2]).unwrap();
    assert_edit_claim(
        file.dataset("other")
            .unwrap()
            .set_attr("tag", AttrValue::I32(1))
            .unwrap_err(),
    );
    assert_appender_conflict(
        file.dataset("other")
            .unwrap()
            .into_buffered_appender()
            .unwrap_err(),
    );
    let error = appender.flush().unwrap_err();
    assert!(matches!(error, Error::StaleHandle), "{error:?}");
    assert_eq!(appender.unwritten(), i32_bytes(&[1, 2]));
    assert_eq!(appender.discard(), i32_bytes(&[1, 2]));
    let mut replacement = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    replacement.append(&[1i32, 2]).unwrap();
    replacement.finish().unwrap();
    file.dataset("other")
        .unwrap()
        .set_attr("tag", AttrValue::I32(1))
        .unwrap();
    file.commit().unwrap();
    drop(file);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), [1, 2]);
}

#[rstest]
#[case::empty(&[])]
#[case::whole_element(&[1, 0, 0, 0])]
#[case::split_element(&[1, 0])]
fn closed_session_rejects_owned_operations_without_accepting_bytes(#[case] prior: &[u8]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("owned_closed.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    appender.append_raw(prior).unwrap();
    file.close().unwrap();
    let error = appender.append::<i32>(&[]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.append(&[2i32]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.append_raw(&[]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.append_raw(&[0]).unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    let error = appender.flush().unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    assert_eq!(appender.unwritten(), prior);
    assert_eq!(appender.discard(), prior);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), Vec::<i32>::new());
}

#[rstest]
#[case::empty(&[])]
#[case::sub_chunk(&[1, 0, 0, 0])]
fn closed_session_rejects_owned_finish(#[case] prior: &[u8]) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("owned_closed_finish.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&path);
    let file = File::open_rw(&path).unwrap();
    let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
    appender.append_raw(prior).unwrap();
    file.close().unwrap();
    let error = appender.finish().unwrap_err();
    assert!(matches!(error, Error::FileClosed), "{error:?}");
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), Vec::<i32>::new());
}

#[derive(Clone, Copy, Debug)]
enum OwnedTermination {
    Discard,
    Drop,
    Finish,
    IncompleteFinish,
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn assert_appender_conflict(error: Error) {
    let Error::EditUnsupported(reason) = error else {
        panic!("expected appender conflict, got {error:?}");
    };
    assert_eq!(
        reason,
        "this dataset already has a live buffered appender; two of them would interleave \
                 their buffers a chunk at a time"
    );
}

fn assert_edit_claim(error: Error) {
    let Error::EditUnsupported(reason) = error else {
        panic!("expected live-appender edit rejection, got {error:?}");
    };
    assert_eq!(
        reason,
        "this dataset has a live buffered appender holding elements only its own flush \
                 can write, and this edit would stop that flush; finish or discard the appender \
                 first"
    );
}

fn assert_append_rejection(error: Error, expected: &str) {
    let Error::AppendInPlaceUnsupported(reason) = error else {
        panic!("expected append rejection, got {error:?}");
    };
    assert_eq!(reason, expected);
}
