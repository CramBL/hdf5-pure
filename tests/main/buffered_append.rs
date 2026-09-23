//! `Dataset::buffered_appender` (issue #262): appended elements are held in
//! memory and written a whole chunk at a time, so a caller appending less than
//! a chunk per call writes once per chunk rather than once per call. Since
//! issue #393 every write it makes is the immediate in-place one, filtered or
//! not, so the type buys write *frequency* and nothing else.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, ScaleOffset};
use tempfile::tempdir;
use test_util_hdf5::dataset::{self, Filter, Unlimited};

const SHUFFLE_DEFLATE: &[Filter] = &[Filter::Shuffle, Filter::Deflate(4)];

// ---- what the buffer buys ----------------------------------------------------

/// The point of the type: a call that does not complete a chunk does not touch
/// the file. Asserted against the file's own size, which changes only when
/// something is written, so this measures the property rather than restating
/// the implementation.
#[test]
fn a_call_that_does_not_complete_a_chunk_does_not_touch_the_file() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    let chunk = 16u64;
    Unlimited::<i32>::new("d", &[], chunk)
        .filters(SHUFFLE_DEFLATE)
        .pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    assert_eq!(app.chunk_elements(), chunk);

    // Four elements per call into a chunk of sixteen: twelve calls complete
    // three chunks, so the file must change on calls 4, 8, and 12 and on no
    // other. Recording the size before the first call makes every call
    // comparable to its predecessor.
    let mut sizes = vec![session.file_size()];
    for i in 0..12i32 {
        app.append(&[i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
            .unwrap();
        sizes.push(session.file_size());
    }
    let wrote_on: Vec<usize> = (1..sizes.len())
        .filter(|&i| sizes[i] != sizes[i - 1])
        .collect();
    assert_eq!(
        wrote_on,
        vec![4, 8, 12],
        "expected a write only where a chunk completed; sizes {sizes:?}"
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::new("d", &(0..8).collect::<Vec<i32>>(), 4).pure_create(&p);
    let file = File::open(&p).unwrap();
    let mut ds = file.dataset("d").unwrap();
    assert!(matches!(
        ds.buffered_appender().unwrap_err(),
        Error::ReadOnly
    ));
}

/// A refused `append` buffers nothing. The refusal here comes from closing the
/// session out from under the appender, which fails before the write touches
/// the file. A caller who reads `Err` as "not appended" — the contract
/// `Dataset::append` has — would otherwise append the same elements twice on a
/// retry, so the call rolls its own bytes back out of the buffer.
#[test]
fn a_refused_append_buffers_nothing() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    Unlimited::<i32>::new("d", &[], 8).pure_create(&p);

    let session = File::open_rw(&p).unwrap();
    let mut ds = session.dataset("d").unwrap();
    let mut app = ds.buffered_appender().unwrap();
    app.append(&[0i32, 1, 2]).unwrap();
    assert_eq!(app.buffered_elements(), 3);

    session.close().unwrap();

    // Six more crosses the chunk boundary, so this call attempts a write.
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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

    let before = session.space_accounting().unwrap().reusable_free_bytes;
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
    let after = session.space_accounting().unwrap().reusable_free_bytes;
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
    let dir = tempdir().unwrap();
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
    let dir = tempdir().unwrap();
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
