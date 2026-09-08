//! The superblock's status-flags byte gates every open (issue #245).
//!
//! The byte is durable: a SWMR writer raises it and clears it on a clean close,
//! so a writer that exits without one leaves it set. These tests pin which opens
//! that byte turns away — a plain read, a streaming read, an editor, a second
//! SWMR writer — which follow it — a SWMR reader — and that
//! `File::clear_swmr_flag` restores all of them.

#![expect(clippy::mem_forget, reason = "the tests model a writer that crashed")]

use hdf5_pure::{
    Error, File, FileAccessProperties, FileBuilder, FileLocking, FileSpaceStrategy, MemoryStrategy,
    SyncPolicy, WriteMarkPolicy,
};
use tempfile::tempdir;

/// An appendable file: rank-1, unlimited, Extensible-Array indexed, unfiltered —
/// what the SWMR writer accepts.
fn build_swmr(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&[0i32, 1, 2, 3])
        .with_shape(&[4])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(path).unwrap();
}

/// The status-flags byte as it stands on disk, read as bytes because the opens
/// under test refuse a flagged file.
fn flags(path: &std::path::Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    let sig = b"\x89HDF\r\n\x1a\n";
    let off = bytes.windows(sig.len()).position(|w| w == sig).unwrap();
    bytes[off + 11]
}

/// Leave the file flagged exactly as a crashed writer would: leak the writer so
/// neither `close` nor `Drop` clears the byte.
fn flag_as_if_crashed(path: &std::path::Path) {
    std::mem::forget(File::open_swmr_writer(path).unwrap());
    assert_eq!(flags(path), 0x05, "the leaked writer left the file flagged");
}

#[track_caller]
fn assert_marked_in_use(err: Error, what: &str) {
    let msg = err.to_string();
    assert!(
        matches!(err, Error::FileMarkedInUse(_)),
        "{what} must report the status flag, got {err:?}"
    );
    assert!(
        msg.contains("clear_swmr_flag"),
        "{what} must name the recovery, got: {msg}"
    );
}

/// The issue's repro: a file left flagged by a crashed SWMR writer is refused by
/// every open that would read a snapshot of it or write to it, where all of them
/// used to succeed — `open_rw` going on to edit the file in place under the
/// writer that (as far as the file records) still holds it.
#[test]
fn a_flagged_file_is_refused_by_the_reading_and_writing_opens() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    assert_marked_in_use(File::open(&path).unwrap_err(), "File::open");
    assert_marked_in_use(
        File::open_streaming(&path).unwrap_err(),
        "File::open_streaming",
    );
    assert_marked_in_use(File::open_rw(&path).unwrap_err(), "File::open_rw");
    assert_marked_in_use(
        File::open_swmr_writer(&path).unwrap_err(),
        "a second File::open_swmr_writer",
    );

    // A path is what an open by path reports, not what it decides by: the same
    // superblock byte refuses a source, which names itself instead.
    let handle = std::fs::File::open(&path).unwrap();
    let err = File::from_source(hdf5_pure::ReadSeekSource::new(handle).unwrap()).unwrap_err();
    assert!(
        matches!(&err, Error::FileMarkedInUse(said) if said.contains("byte source")),
        "File::from_source reported {err:?} rather than naming the source it refused"
    );
    assert_marked_in_use(err, "File::from_source");
}

/// The bounded editor reaches the same refusal as the mirrored one: both open the
/// same engine, and a strategy that picks a backing must not pick a rule.
#[test]
fn both_edit_backings_refuse_a_flagged_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    for strategy in [MemoryStrategy::Bounded, MemoryStrategy::Mirrored] {
        let props = hdf5_pure::FileAccessProperties::new().with_memory_strategy(strategy);
        assert_marked_in_use(
            File::open_rw_with_options(&path, props).unwrap_err(),
            &format!("open_rw under {strategy:?}"),
        );
    }
}

/// A SWMR reader is the one open that follows the flag rather than being turned
/// away by it — that pairing is the whole point of the flag — and it reads the
/// data a live writer has already appended.
#[test]
fn a_swmr_reader_follows_a_flagged_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("live.h5");
    build_swmr(&path);

    let writer = File::open_swmr_writer(&path).unwrap();
    writer
        .dataset("d")
        .unwrap()
        .append(&[4i32, 5, 6, 7])
        .unwrap();

    let reader = File::open_swmr(&path).expect("a SWMR reader attaches to a flagged file");
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        (0..8).collect::<Vec<_>>()
    );
    drop(reader);
    writer.close().unwrap();
}

/// `clear_swmr_flag` is the documented recovery, so every open it was blocking
/// has to work afterwards — including the SWMR writer, whose own flag it cleared.
#[test]
fn clearing_the_flag_restores_every_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("recovered.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    File::clear_swmr_flag(&path).unwrap();
    assert_eq!(flags(&path), 0x00);

    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..4).collect::<Vec<_>>()
    );
    File::open_streaming(&path).unwrap();
    {
        let handle = std::fs::File::open(&path).unwrap();
        File::from_source(hdf5_pure::ReadSeekSource::new(handle).unwrap()).unwrap();
    }
    File::open_swmr(&path).unwrap();
    drop(File::open_rw(&path).unwrap());
    File::open_swmr_writer(&path).unwrap().close().unwrap();
}

/// `File::from_bytes` does not check the flag: the caller already holds a
/// snapshot, there is no live file to coordinate over, and it is the way out for
/// a flagged file on a read-only mount, where the recovery `clear_swmr_flag`
/// needs write access it cannot get. The C library *does* check under its
/// in-memory driver, so this is a deliberate divergence, not parity.
#[test]
fn from_bytes_reads_a_flagged_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flagged.h5");
    build_swmr(&path);
    flag_as_if_crashed(&path);

    let file = File::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(file.superblock().consistency_flags, 0x05);
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..4).collect::<Vec<_>>()
    );
}

// --- The page buffer's crash mark (issue #308) ---------------------------------
//
// A SWMR writer is not the only session that raises this byte. A session given
// `FileAccessProperties::with_page_buffer_size` holds dirty pages across the
// write engine's ordering barriers, which issues every publish point ahead of
// the content it names — so a session that dies mid-flush can leave a dataset
// that reads *clean* and returns fill values or a deleted object's bytes, with
// every checksum verifying. The mark is what makes that file refuse to open
// instead, here and in the C library alike.
//
// It raises bit 0 alone, where a SWMR writer raises bits 0 and 2, so `flags`
// below distinguishes the two rather than only reporting "marked".

/// A paged file with an appendable dataset: what a page buffer requires.
fn build_paged(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(4096);
    b.create_dataset("d")
        .with_i32_data(&[0i32; 64])
        .with_shape(&[64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[64]);
    b.write(path).unwrap();
}

/// Access properties for a page-buffered session.
///
/// Locking is disabled deliberately, and it is not incidental to the tests: OS
/// locks are **mandatory** on Windows, so a held `open_rw` lock blocks the very
/// `std::fs::read` these tests use to observe the byte mid-session. What is under
/// test is the superblock flag, not the lock.
fn page_buffered() -> FileAccessProperties {
    FileAccessProperties::new()
        .with_sync_policy(SyncPolicy::OnClose)
        .with_locking(FileLocking::Disabled)
        .with_page_buffer_size(1 << 20)
}

/// The mark stands from the moment the buffer is installed until the session
/// closes, and a clean close takes it back down.
///
/// The `while open` half is the one that matters: a mark raised at close would
/// be useless, since a crashed session never reaches close. The `after close`
/// half is what keeps the property usable at all — a file left flagged is
/// refused by every later open.
#[test]
fn a_page_buffered_session_marks_the_file_for_its_lifetime() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("buffered.h5");
    build_paged(&path);
    assert_eq!(flags(&path), 0x00, "a fresh file carries no mark");

    let file = File::open_rw_with_options(&path, page_buffered()).unwrap();
    assert_eq!(
        flags(&path),
        0x01,
        "the mark must be on the disk before the first buffered write, not at close"
    );
    assert_marked_in_use(
        File::open(&path).unwrap_err(),
        "a read of a page-buffered file",
    );

    let mut ds = file.dataset("d").unwrap();
    ds.append(&[1i32; 64]).unwrap();
    drop(ds);
    file.close().unwrap();

    assert_eq!(flags(&path), 0x00, "a clean close must take the mark down");
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap()
            .len(),
        128,
        "and leave a file that reads"
    );
}

/// A commit inside the epoch rewrites the superblock, and must publish the mark
/// rather than the zero every other clean commit publishes.
///
/// This is the sharp edge of the feature: the commit's zero is deliberate — it
/// scrubs a flag the file arrived carrying — so a page-buffered session's very
/// first commit would take its own mark down and leave the rest of the session
/// unguarded, with nothing failing to say so.
///
/// The `sync` is load-bearing, not tidying. The commit's superblock write lands
/// in the page buffer like any other, so reading the disk straight after it sees
/// the byte the *open* wrote and passes whatever the commit put there. Forcing
/// the flush is what puts the commit's own superblock on the disk to be read —
/// and it is not a contrivance either: `File::sync` is how an application on
/// `SyncPolicy::OnClose` names its own cadence, and a commit exceeding the byte
/// budget flushes on its own.
#[test]
fn a_commit_in_a_page_buffered_session_leaves_the_mark_standing() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    build_paged(&path);

    let file = File::open_rw_with_options(&path, page_buffered()).unwrap();
    file.root()
        .create_dataset("added", |b| {
            b.with_f64_data(&[2.5f64; 32]).with_shape(&[32]);
        })
        .unwrap();
    file.commit().unwrap();
    file.sync().unwrap();
    assert_eq!(
        flags(&path),
        0x01,
        "a commit must publish the session's mark, not scrub it"
    );

    // A second one. Not because it takes a different tail — instrumenting the
    // three superblock publish sites shows both commits go through the same one,
    // the paged persisting tail — but because a mark republished once and then
    // dropped would pass a single-commit test.
    file.root()
        .create_dataset("again", |b| {
            b.with_f64_data(&[3.5f64; 32]).with_shape(&[32]);
        })
        .unwrap();
    file.commit().unwrap();
    file.sync().unwrap();
    assert_eq!(flags(&path), 0x01, "and so must every later commit");

    file.close().unwrap();
    assert_eq!(flags(&path), 0x00);
    assert!(File::open(&path).is_ok(), "and the file must open again");
}

/// The case the mark exists for: a page-buffered session that never closes.
///
/// Leaked rather than crashed, which is the same thing as far as the file is
/// concerned — neither `close` nor `Drop` runs, so nothing clears the byte. Every
/// open that would hand back the file's contents is then refused, which is the
/// whole trade: without the mark this file opens and reads clean.
#[test]
fn a_crashed_page_buffered_session_leaves_a_file_every_open_refuses() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("crashed.h5");
    build_paged(&path);

    let file = File::open_rw_with_options(&path, page_buffered()).unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[7i32; 64]).unwrap();
    std::mem::forget(ds);
    std::mem::forget(file);
    assert_eq!(
        flags(&path),
        0x01,
        "the leaked session left the file marked"
    );

    assert_marked_in_use(File::open(&path).unwrap_err(), "a buffered read");
    assert_marked_in_use(File::open_streaming(&path).unwrap_err(), "a streaming read");
    assert_marked_in_use(File::open_rw(&path).unwrap_err(), "an editor");
    assert_marked_in_use(
        File::open_swmr(&path).unwrap_err(),
        "a SWMR reader, whose pair is half-set",
    );

    // And the documented recovery reaches this mark too, since it clears the
    // byte whole. What comes back is access, not a promise about the contents.
    File::clear_swmr_flag(&path).unwrap();
    assert_eq!(flags(&path), 0x00);
    File::open(&path).unwrap();
}

// --- The write mark and the snapshot opt-in (issue #419) ----------------------
//
// The mark a page-buffered session leaves is bit 0 alone, and no SWMR reader can
// follow that: a SWMR reader needs the pair. So every path-based read was refused
// while such a writer ran, and the refusal named `File::open_swmr`, which refused
// the same file in its turn. `FileAccessProperties::with_write_mark_policy` is
// the reader that does apply, on the caller's assertion that the writer has
// flushed, and the refusal names it instead.

/// Access properties that take a snapshot of a write-marked file.
fn snapshot() -> FileAccessProperties {
    FileAccessProperties::new().with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)
}

/// Leave a live page-buffered session holding `path`, its append flushed.
///
/// Leaked deliberately: the mark has to still be standing when the reads under
/// test run, which is the whole scenario. The `sync` is what makes the snapshot
/// a consistent one, and is the assertion the opt-in's contract asks for.
fn live_page_buffered_writer(path: &std::path::Path) {
    build_paged(path);
    let file = File::open_rw_with_options(path, page_buffered()).unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[7i32; 64]).unwrap();
    file.sync().unwrap();
    std::mem::forget(ds);
    std::mem::forget(file);
    assert_eq!(flags(path), 0x01, "the live session marks the file");
}

/// The issue's repro: every path-based read is refused, and the refusal points
/// at a recovery that exists rather than at `File::open_swmr`, which refuses the
/// same mark itself.
#[test]
fn a_write_marked_file_names_a_recovery_that_works() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("live.h5");
    live_page_buffered_writer(&path);

    for (what, err) in [
        ("File::open", File::open(&path).unwrap_err()),
        (
            "File::open_streaming",
            File::open_streaming(&path).unwrap_err(),
        ),
    ] {
        let msg = err.to_string();
        assert_marked_in_use(err, what);
        assert!(
            !msg.contains("open_swmr"),
            "{what} names a reader that refuses this mark too: {msg}"
        );
        assert!(
            msg.contains("with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)"),
            "{what} does not name the opt-in that reads it: {msg}"
        );
    }

    // The mark is half a pair, so the SWMR reader refuses it — the behaviour the
    // error text used to send callers into, and which stays as it is.
    assert_marked_in_use(
        File::open_swmr(&path).unwrap_err(),
        "a SWMR reader following half a pair",
    );
}

/// The opt-in reads what the writer flushed, through both path-based readers and
/// through a source, none of which copies the file as `File::from_bytes` does.
#[test]
fn the_opt_in_reads_a_flushed_write_marked_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("live.h5");
    live_page_buffered_writer(&path);

    let expected: Vec<i32> = [0i32; 64].into_iter().chain([7i32; 64]).collect();
    for (what, file) in [
        (
            "File::open_with_options",
            File::open_with_options(&path, snapshot()).unwrap(),
        ),
        (
            "File::open_streaming_with_options",
            File::open_streaming_with_options(&path, snapshot()).unwrap(),
        ),
        ("File::from_source_with_options", {
            let handle = std::fs::File::open(&path).unwrap();
            let source = hdf5_pure::ReadSeekSource::new(handle).unwrap();
            File::from_source_with_options(source, snapshot()).unwrap()
        }),
    ] {
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            expected,
            "{what} read something other than what the writer committed"
        );
        assert_eq!(file.superblock().consistency_flags, 0x01);
    }
}

/// The opt-in is a *reader's*. It does not let a second writer join a file a
/// writer holds, whichever backing that writer would take.
#[test]
fn the_opt_in_does_not_unlock_a_read_write_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("live.h5");
    live_page_buffered_writer(&path);

    // Locking off so the superblock byte is what refuses this, not the OS lock
    // the leaked session still holds.
    let props = snapshot().with_locking(FileLocking::Disabled);
    assert_marked_in_use(
        File::open_rw_with_options(&path, props).unwrap_err(),
        "File::open_rw under the opt-in",
    );
    assert_marked_in_use(
        File::open_rw_with_options(&path, props.with_memory_strategy(MemoryStrategy::Mirrored))
            .unwrap_err(),
        "a mirrored File::open_rw under the opt-in",
    );
}

/// A live SWMR writer's pair is not what the opt-in admits: that file has a
/// reader of its own, which follows the writer's later appends as a snapshot
/// cannot.
#[test]
fn the_opt_in_does_not_unlock_a_swmr_pair() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("swmr.h5");
    build_swmr(&path);
    let writer = File::open_swmr_writer(&path).unwrap();
    writer
        .dataset("d")
        .unwrap()
        .append(&[4i32, 5, 6, 7])
        .unwrap();
    assert_eq!(flags(&path), 0x05);

    assert_marked_in_use(
        File::open_with_options(&path, snapshot()).unwrap_err(),
        "a buffered read of a SWMR pair under the opt-in",
    );
    assert_marked_in_use(
        File::open_streaming_with_options(&path, snapshot()).unwrap_err(),
        "a streaming read of a SWMR pair under the opt-in",
    );
    File::open_swmr(&path).expect("the reader that does follow a SWMR writer");
    writer.close().unwrap();
}

/// An ordinary editor raises nothing. The crate's divergence from the C library
/// — which marks the file for *any* writer — is deliberate, and it is what makes
/// the mark above mean "a page buffer held this file" rather than "someone
/// opened it".
///
/// Locking is off for the same reason as everywhere else here, and it changes
/// nothing about what is under test: the lock and the flag are separate guards,
/// and this one is about the flag. The default `open_rw` takes an exclusive lock,
/// which on Windows is *mandatory* and would block the read below.
#[test]
fn an_unbuffered_editor_raises_no_mark() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("plain.h5");
    build_paged(&path);

    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    let mut ds = file.dataset("d").unwrap();
    ds.append(&[1i32; 64]).unwrap();
    drop(ds);
    assert_eq!(
        flags(&path),
        0x00,
        "an open_rw session without a page buffer marks nothing"
    );
    // And so the plain readers need no opt-in to snapshot it while it runs:
    // nothing on disk says a writer holds the file.
    File::open(&path).unwrap();
    File::open_streaming(&path).unwrap();
    file.close().unwrap();
    assert_eq!(flags(&path), 0x00);
}

/// Dropping the handle without `close` takes the mark down too.
///
/// `close` is the documented ending, but `drop` is the one a `?` on an unrelated
/// error takes, and a session that left the mark standing there would leave a
/// file nothing can open until `clear_swmr_flag` — an availability failure with
/// no crash behind it. Both teardowns write, and both must finish the job.
#[test]
fn dropping_a_page_buffered_session_takes_the_mark_down() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dropped.h5");
    build_paged(&path);

    {
        let file = File::open_rw_with_options(&path, page_buffered()).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&[4i32; 64]).unwrap();
        assert_eq!(flags(&path), 0x01, "the mark stands while the session does");
    }

    assert_eq!(flags(&path), 0x00, "drop must take the mark down");
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap()
            .len(),
        128,
        "and the append must be in the file it leaves"
    );
}
