//! Replay every prefix of what reached the disk, and read what it left.
//!
//! # The defect class this exists for
//!
//! Since the write gathering of issue #288, the writes an operation makes
//! between two ordering barriers are issued in **address order** rather than in
//! the order the engine made them. In this format that is systematically the
//! wrong way round: every publish point sits at a *lower* address than the
//! content it names — the superblock at address 0, an object header's dataspace
//! dimension and Extensible-Array element count near the front of the file, the
//! chunk bytes and index blocks they name at end-of-file. So an
//! append-then-publish pair with no barrier between them now publishes *first*.
//!
//! The resulting failure is the bad kind. A pointer that survives while the
//! block it names does not is a checksummed, valid-looking address into bytes
//! past end-of-file, and the *next* append believes the block exists and writes
//! into nothing.
//!
//! What holds this together is [`FileImage::ordering_barrier`], and a missing one
//! is invisible to every ordinary test: the operation completes, the file is
//! correct, and only a machine that stops at the wrong instant can tell. This
//! module is what stops the machine at every instant.
//!
//! # How it works
//!
//! [`disk_log`] records each write and each `set_len` that a session actually
//! issued. A [`Recording`] pairs that log with the file's bytes as they were
//! before the session opened, which makes every prefix of the log a file that
//! could really have existed: the operations are positioned, so applying the
//! first *k* of them to the starting bytes is exactly the disk state after *k* of
//! them completed. Each prefix is written out and read back, and judged as
//! [`Verdict::Clean`], [`Verdict::Loud`] or [`Verdict::Silent`].
//!
//! # This one can fail
//!
//! A crash harness that cannot fail is decoration, and the first draft of this
//! one was: it swept five hundred prefixes, called every one of them clean, and
//! went on doing so with **both** of the ordering barriers it was written for
//! deleted. Nothing about that was visible from the outside — the sweep was wide,
//! the assertions were real, and the numbers looked healthy.
//!
//! Three separate things were wrong, and the fix for each is a *demand the code
//! makes* rather than a value that happens to be right:
//!
//! 1. The window did not cross the allocations that reach the barriers — see
//!    [`WARMUP_ROUNDS`] and [`Recording::assert_positioned`].
//! 2. The file was small enough that a publish point and the content it names
//!    merged into a single write, which is atomic — see [`CHUNK`] and the same
//!    method's second condition.
//! 3. A prefix that refused to read was counted as benign — see
//!    [`Tally::assert_sound`].
//!
//! Each of those is now something a future change trips over rather than
//! silently loses. [`Recording::replay_every_prefix`] adds two more of the same
//! kind: replaying the *whole* log must reproduce the file the session really
//! left, so a write escaping [`disk_log`] cannot pass unnoticed, and the last
//! prefix must satisfy the workload's finished state, so a session that quietly
//! stopped doing work cannot either.
//!
//! # What it covers, and what it does not
//!
//! Measured by deleting each barrier in the Extensible-Array append engine in
//! turn and running the library:
//!
//! | barrier | caught |
//! | --- | --- |
//! | before a fresh data-block pointer | yes, and by the pre-existing structural test |
//! | before a fresh super-block pointer | yes, only by [`a_crashed_append_can_be_reopened_and_appended_to`] |
//! | `apply_ea_append` phase 1, chunk bytes → superblock end-of-file | yes, by [`recorded_eof_covers_the_file`] |
//! | `apply_ea_append` phase 1, end-of-file → index writes | **no** |
//! | `apply_ea_append` phase 2 → 3 | **no** |
//! | `apply_ea_append` phase 3, header count → dimension | yes |
//!
//! A barrier is not the only way this class of defect arrives. A *publish* is
//! the other: a value written apart from the checksum covering it is two writes
//! with a crash point between them, no matter how the barriers fall. That was
//! issue #307, and it is why the sweeps here are run at more than one header
//! width — see [`warmed_base_padded`] and
//! [`a_publish_costs_the_same_whether_or_not_it_spans_a_page`].
//!
//! The two uncovered ones are not gaps in the sweep, and it is worth knowing why
//! before adding a workload to chase them. Deleting the first changes **nothing
//! that reaches the disk** under gathering — the recorded write order is
//! byte-identical, because the order it enforces is the order address-sorting
//! already produces. Deleting the second does reorder two metadata writes, but
//! the state between them is a published element count with a stale block, which
//! is indistinguishable from a normal in-progress append to a reader bounded by
//! the dataspace dimension. Both remain load-bearing for the two things this
//! module does not model: `fsync` durability under
//! [`SyncPolicy::Always`](crate::SyncPolicy), and the visibility order a *SWMR*
//! reader follows, which is defined in terms of the header counts rather than the
//! dimension.
//!
//! An append that allocates out of *freed* space rather than at end-of-file
//! (issue #349) is swept by [`appending_into_a_reserved_hole_survives_a_crash_at_every_write`],
//! which reaches the branch through a file that **persists** its free space: such
//! a session seeds its free list from the on-disk managers at open, so the hole
//! is there before the recorded window rather than made by a commit inside it.
//! That sweep is also the only one that crosses a *reservation* — the manager
//! rewrite an append on such a file makes before it may spend a hole (issue
//! #387) — which is a superblock repoint in the middle of an append sequence.
//!
//! On a *non*-persisting file the same branch stays out of reach, and is closed
//! by argument instead: a session's free list is populated only by its own
//! commits, so reaching it would mean bringing a whole `commit` into the recorded
//! window and undoing the positioning [`WARMUP_ROUNDS`] and
//! [`Recording::assert_positioned`] exist to guarantee. The barriers are there
//! because an *appended* region sits above the pointer naming it, and a reused
//! region either sits above it too — the same case — or below it, where
//! address-ordered gathering already issues the content first. See
//! [`Store::alloc_raw`](crate::chunk_index_inplace::Store::alloc_raw).
//!
//! Filtered chunk writes are covered only through this engine, by
//! [`regrowing_a_filtered_trailing_chunk_survives_a_crash_at_every_write`],
//! which sweeps the one write in the crate that moves an *already visible*
//! chunk: the trailing-chunk rewrite of issue #393. Outside this engine, nothing
//! is covered: dense attribute heaps, the Fixed-Array and v2 B-tree chunk
//! indexes, the chunked writer's own filtered output, and group link-table
//! growth all publish low-address pointers to high-address content and have no
//! sweep here. That is the remaining half of issue #309.

use std::path::Path;

use crate::chunk_index_inplace::alloc_probe;
use crate::dataspace::MaxExtent;
use crate::edit::{AppendBuilder, AppendTarget, MemoryStrategy, SyncPolicy, WriteEngine};
use crate::error::{Error, FormatError};
use crate::file_lock::FileLocking;
use crate::file_space_info::FileSpaceStrategy;
use crate::image::disk_log::{self, DiskOp};
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

/// What reading one replayed prefix made of it.
enum Verdict {
    /// Read back, and the state is one a crash at this point may leave.
    Clean,
    /// Refused to read. Benign in general — the caller is told rather than
    /// misled — but see [`Tally::assert_sound`]: the operations swept here
    /// promise that the previous value stays readable, so this is a failure too.
    Loud(String),
    /// Read back without complaint, and returned something else.
    Silent(String),
}

impl Verdict {
    /// Classify a read. An `Error` is [`Loud`](Verdict::Loud) whatever it says —
    /// the file declined to hand over data, which is the outcome that cannot
    /// mislead. Only a *successful* read is judged on its contents.
    fn of<T>(read: Result<T, Error>, judge: impl FnOnce(T) -> Result<(), String>) -> Verdict {
        match read {
            Err(e) => Verdict::Loud(std::format!("{e:?}")),
            Ok(v) => match judge(v) {
                Ok(()) => Verdict::Clean,
                Err(why) => Verdict::Silent(why),
            },
        }
    }
}

/// A session's disk operations, and the file they started from.
struct Recording {
    label: String,
    /// The file's bytes before the recorded session opened it. Read before the
    /// session exists rather than alongside it, so no platform's file locking
    /// has an opinion about it.
    base: Vec<u8>,
    /// The file the recorded session actually wrote, kept so the last prefix can
    /// be compared against it.
    real: std::path::PathBuf,
    ops: Vec<DiskOp>,
    /// Fresh Extensible-Array blocks the recorded window allocated, by kind.
    /// See [`assert_allocated`](Self::assert_allocated).
    data_blocks: usize,
    super_blocks: usize,
}

impl Recording {
    /// Record everything `work` puts on the disk. `work` is handed `path`, and
    /// must open and close its own session inside the call: a session's teardown
    /// writes are part of what a crash can interrupt, so they belong in the log.
    fn of(label: &str, path: &Path, work: impl FnOnce(&Path)) -> Self {
        let base = std::fs::read(path).expect("the starting file");
        // Discard whatever the warm-up allocated, so the counts below describe
        // the recorded window rather than the whole test.
        alloc_probe::take();
        disk_log::start();
        work(path);
        let ops = disk_log::take();
        let (data_blocks, super_blocks) = alloc_probe::take();
        // The recorded order itself, on request. This is what shows the
        // address-order inversion directly: with a barrier the block at
        // end-of-file is issued before the pointer naming it, and without one the
        // same two writes come out the other way round.
        if std::env::var_os("CRASH_REPLAY_OPS").is_some() {
            for (i, op) in ops.iter().enumerate() {
                std::eprintln!("OP {i} {}", op.describe());
            }
        }
        assert!(
            !ops.is_empty(),
            "{label}: recorded nothing, so there is no crash point to replay"
        );
        Self {
            label: label.to_string(),
            base,
            real: path.to_path_buf(),
            ops,
            data_blocks,
            super_blocks,
        }
    }

    /// Require that this fixture is positioned where the defect is visible. Two
    /// separate conditions, because the first draft of this module satisfied
    /// neither while sweeping five hundred prefixes and reporting every one
    /// clean:
    ///
    /// 1. The window **allocated** fresh index blocks of each kind, since the two
    ///    ordering barriers in [`crate::chunk_index_inplace`] are only reached by
    ///    an allocation. A window over the wrong rounds crosses neither.
    /// 2. End-of-file is **far enough from the index** that the two sides of a
    ///    publish cannot share a gathered write. This is the condition that is
    ///    easy to lose by retuning a constant: [`CHUNK`] used to be one element,
    ///    which kept the whole file inside two pages, and two writes that merge
    ///    into one are atomic — the inversion still happens and no prefix can
    ///    observe it. Deleting a barrier and shrinking `CHUNK` back to 1 makes
    ///    the sweeps pass again with condition 1 still satisfied, so nothing but
    ///    this catches it.
    fn assert_positioned(&self, data_blocks: usize, super_blocks: usize) {
        assert!(
            self.data_blocks >= data_blocks && self.super_blocks >= super_blocks,
            "{}: the recorded window allocated {} data block(s) and {} super block(s), \
             short of the {data_blocks} and {super_blocks} it is positioned for. \
             Nothing below can see a barrier that is not crossed.",
            self.label,
            self.data_blocks,
            self.super_blocks
        );
        let pages = std::fs::metadata(&self.real).unwrap().len() / GATHER_PAGE;
        assert!(
            pages >= MIN_PAGES,
            "{}: the file spans {pages} gather pages of {GATHER_PAGE} bytes, under the \
             {MIN_PAGES} this needs. Below that, a publish point near the front of the \
             file and the block it names at end-of-file merge into one gathered write, \
             which is atomic, and no replayed prefix can fall between them.",
            self.label
        );
    }

    /// Demand that a *publish* write in this recording crosses a gather-page
    /// boundary — that the structure being published really is wider than the
    /// page whose merging hid issue #307.
    ///
    /// A publish is a patch near the front of the file, so this looks for a write
    /// that begins inside the first gather page and reaches past it. Any write is
    /// not enough: an append at end-of-file crosses page boundaries all day and
    /// says nothing about the header's width.
    ///
    /// Without this, a padded fixture is one object-header layout change away
    /// from sitting wholly inside a page, where every publish is atomic for free
    /// and the sweep passes while proving nothing. The padding was measured, and
    /// a measured constant is exactly the kind that drifts.
    fn assert_publishes_across_a_gather_page(&self) {
        let crossing = self.ops.iter().any(|op| match *op {
            DiskOp::Write { offset, ref bytes } => {
                offset < GATHER_PAGE && offset + bytes.len() as u64 > GATHER_PAGE
            }
            DiskOp::SetLen(_) => false,
        });
        assert!(
            crossing,
            "{}: no write starts inside the first {GATHER_PAGE}-byte gather page and \
             reaches past it, so the header this sweep publishes into fits in one page \
             and every publish is atomic whatever the engine does. The fixture needs a \
             wider object header.",
            self.label
        );
    }

    /// Apply one operation to an in-memory image of the file, exactly as the
    /// filesystem would: a positioned write past end-of-file extends it, leaving
    /// the gap reading as zeros, and `set_len` both truncates and extends.
    fn apply(image: &mut Vec<u8>, op: &DiskOp) {
        match op {
            DiskOp::Write { offset, bytes } => {
                let start = *offset as usize;
                let end = start + bytes.len();
                if image.len() < end {
                    image.resize(end, 0);
                }
                image[start..end].copy_from_slice(bytes);
            }
            DiskOp::SetLen(len) => image.resize(*len as usize, 0),
        }
    }

    /// Materialize every prefix in turn and hand each to `check`, then hold it to
    /// `expect`.
    ///
    /// `finished` is what the *last* prefix must satisfy — the state the workload
    /// was supposed to reach. Without it every sweep here is satisfiable by doing
    /// nothing: prefix 0 is the untouched starting file, every checker accepts it
    /// (an append that has not happened and a commit that has not landed are both
    /// legitimate crash states), and so a regression that silently stopped
    /// publishing would leave all five sweeps reporting 100% clean.
    fn replay_every_prefix(
        &self,
        dir: &Path,
        check: impl Fn(&Path) -> Verdict,
        finished: impl Fn(&Path) -> Result<(), String>,
    ) -> Tally {
        let path = dir.join(std::format!("{}.replay.h5", self.label));
        let mut image = self.base.clone();
        let mut tally = Tally {
            total: self.ops.len() + 1,
            clean: 0,
            loud: Vec::new(),
            silent: Vec::new(),
        };

        for k in 0..=self.ops.len() {
            if k > 0 {
                Self::apply(&mut image, &self.ops[k - 1]);
            }
            std::fs::write(&path, &image).unwrap();
            if k == self.ops.len() {
                // Replaying the whole log must reproduce the file the session
                // actually left. Anything reaching the disk outside
                // `BufferedWrites` — a raw positioned write on the handle, a
                // second thread — is invisible to the log, and every prefix
                // above would then be a state that never existed.
                assert_eq!(
                    image,
                    std::fs::read(&self.real).unwrap(),
                    "{}: replaying every recorded operation does not reproduce the \
                     file the session left, so the log is not everything that \
                     reached the disk",
                    self.label
                );
                if let Err(why) = finished(&path) {
                    panic!(
                        "{}: the completed workload did not reach the state this \
                         sweep is judging interruptions of: {why}",
                        self.label
                    );
                }
            }
            let after = if k == 0 {
                "the starting file".to_string()
            } else {
                self.ops[k - 1].describe()
            };
            match check(&path) {
                Verdict::Clean => tally.clean += 1,
                Verdict::Loud(why) => tally
                    .loud
                    .push(std::format!("after op {k} ({after}): {why}")),
                Verdict::Silent(why) => tally
                    .silent
                    .push(std::format!("after op {k} ({after}): {why}")),
            }
        }
        std::fs::remove_file(&path).ok();
        // How wide the sweep actually was, for whoever is changing it. The
        // assertions below are rules rather than counts — an exact figure here is
        // a property of one platform's allocator and page size — so the numbers
        // are printed on request instead of pinned.
        if std::env::var_os("CRASH_REPLAY_STATS").is_some() {
            std::eprintln!(
                "{}: {} prefixes, {} clean, {} loud, {} silent",
                self.label,
                tally.total,
                tally.clean,
                tally.loud.len(),
                tally.silent.len()
            );
        }
        tally.assert_sound(&self.label);
        tally
    }
}

/// What one sweep found, kept so a caller can compare two of them.
struct Tally {
    total: usize,
    clean: usize,
    /// Why each prefix refused to read. The benign outcome, but the reasons are
    /// what say *which* benign outcome, and they are the first thing to read when
    /// the floor below fails.
    loud: Vec<String>,
    silent: Vec<String>,
}

impl Tally {
    /// The two rules every sweep is held to.
    ///
    /// **Nothing silent.** A prefix that reads cleanly and hands back the wrong
    /// data is the outcome with no signal at all, and no crash point may produce
    /// one.
    ///
    /// **Nothing unreadable.** Stronger, and the real promise of an operation the
    /// crate calls crash-atomic: the dataset was readable before the operation
    /// began, so a crash part-way through must leave it readable, as the old
    /// value. A prefix that refuses to read is a window in which the file went
    /// from readable to not — a regression a caller experiences even though
    /// nothing is silently wrong.
    ///
    /// The second is what the ordering barriers buy, though not all of them
    /// through this rule. Removing the barrier before a fresh *data* block breaks
    /// it directly: the pointer is issued before the block, and the prefix between
    /// them reads `UnexpectedEof`, naming an address past end-of-file. The *super*
    /// block's barrier produces no unreadable prefix at all and is caught by the
    /// first rule instead, in [`a_crashed_append_can_be_reopened_and_appended_to`].
    fn assert_sound(&self, label: &str) {
        assert!(
            self.silent.is_empty(),
            "{label}: {} of {} replayed prefixes read cleanly and returned the wrong data:\n  {}",
            self.silent.len(),
            self.total,
            self.silent.join("\n  ")
        );
        assert!(
            self.loud.is_empty(),
            "{label}: {} of {} replayed prefixes refused to read, though every one \
             of them is a crash during an operation that leaves the previous value \
             in place and readable:\n  {}",
            self.loud.len(),
            self.total,
            self.loud.join("\n  ")
        );
    }
}

// ---------------------------------------------------------------------------
// The append workload
// ---------------------------------------------------------------------------

/// The page a locked session merges its writes within, and the least number of
/// them a fixture must span. See [`Recording::assert_positioned`].
const GATHER_PAGE: u64 = crate::file_space_info::DEFAULT_PAGE_SIZE;
const MIN_PAGES: u64 = 8;

/// Elements per chunk, and per recorded round: one whole chunk per append.
///
/// The Extensible Array allocates blocks by *chunk count*, so the chunk width is
/// free, and it is not free in the one way that matters here. With a
/// one-element chunk the whole file is a few kilobytes, the index block and
/// end-of-file share a gathered write, and a missing barrier between them is
/// **invisible** — the two writes coalesce into one, which is atomic. Widening
/// the chunk moves end-of-file away from the index without changing which round
/// allocates what, which is what puts a prefix between the pointer and the block
/// it names.
const CHUNK: u64 = 64;
/// Elements each recorded round appends: exactly one chunk.
const ROUND: i32 = CHUNK as i32;

/// Rounds the base file already holds when recording starts, and rounds recorded
/// after it.
///
/// The window is positioned, not picked. With the C library's Extensible-Array
/// defaults a fresh *data* block is allocated at chunk 4, 20, 52, 84, 116, 180,
/// 244, 308, ..., and a fresh *super* block at 244 and then not again until 500.
/// Recording chunks 230..=320 therefore crosses two data-block allocations and
/// the super-block one between them, which are the crate's two barrier sites.
///
/// A window at 270 — the obvious round number — crosses no super-block
/// allocation at all, and a sweep positioned there passes with both barriers
/// deleted while still replaying five hundred prefixes and calling every one of
/// them clean. [`crate::chunk_index_inplace::alloc_probe`] is what turns that
/// into a failure rather than a silently narrower sweep.
const WARMUP_ROUNDS: i32 = 230;
/// Recorded rounds.
const ROUNDS: i32 = 90;
/// Elements the base file holds when recording starts.
const WARMUP: i32 = WARMUP_ROUNDS * ROUND;

/// Build a chunked, unlimited `i32` dataset holding `0..n`, and append to it
/// until it holds `WARMUP`. The appends are what matter: a file *written* with
/// 270 elements has one index shape, and a file *appended* to 270 has the shape
/// this harness is about, with the blocks allocated in the order an append
/// allocates them.
fn warmed_base(path: &Path, paged: bool) {
    warmed_base_padded(path, paged, 0);
}

/// [`warmed_base`], with `pad` bytes of attribute in the dataset's object header.
///
/// Padding is a positioning tool, not decoration. `patch_dimension` writes the
/// dataspace dimension near the front of chunk 0 and the chunk checksum at its
/// end; the two are joined into one write only when they share a gather page. A
/// header under [`GATHER_PAGE`] therefore publishes atomically for free and no
/// prefix can fall between them, which is why the unpadded sweeps below cannot
/// see issue #307. Above it they split, and the crash state between them is a
/// new value under its old checksum.
fn warmed_base_padded(path: &Path, paged: bool, pad: usize) {
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(4096);
    }
    {
        let d = b.create_dataset("d");
        d.with_i32_data(&[0i32])
            .with_shape(&[1])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[CHUNK]);
        if pad > 0 {
            d.set_attr(
                "pad",
                crate::type_builders::AttrValue::AsciiString("x".repeat(pad)),
            );
        }
    }
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let mut n = 1i32;
    while n < WARMUP {
        let take = ROUND.min(WARMUP - n);
        append(&mut s, n, take);
        n += take;
    }
    drop(s);
}

/// Append `count` elements continuing the sequence at `from`.
fn append(s: &mut WriteEngine, from: i32, count: i32) {
    let mut b = AppendBuilder::new();
    b.append_i32(&(from..from + count).collect::<Vec<_>>());
    s.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &b, 4)
        .unwrap();
}

/// Read `d` and require that whatever length the file claims, every element up to
/// it is the one that belongs there, and that the length is between the one the
/// recording started at and the one it ended at.
///
/// Element *i* holds the value *i*, so a stale byte, a hole read as a fill value,
/// or a block resurrected from a previous shape all show up as a wrong value
/// rather than having to be looked for. A file that simply has fewer elements is
/// a crash before that append published, which is the whole point of the
/// barriers.
fn appended_prefix_is_intact(path: &Path, lo: i32, hi: i32) -> Verdict {
    if let Err(why) = recorded_eof_covers_the_file(path) {
        return Verdict::Silent(why);
    }
    let read = crate::reader::File::open(path).and_then(|f| f.dataset("d")?.read_i32());
    Verdict::of(read, |data| {
        let n = data.len() as i32;
        if !(lo..=hi).contains(&n) {
            return Err(std::format!(
                "length {n} is outside the {lo}..={hi} this crash point can produce"
            ));
        }
        if let Some(i) = (0..data.len()).find(|&i| data[i] != i as i32) {
            return Err(std::format!(
                "element {i} is {} rather than {i} (length {n})",
                data[i]
            ));
        }
        Ok(())
    })
}

/// The superblock's recorded end-of-file must not name bytes the file does not
/// have.
///
/// Reading one dataset cannot see this. The recorded end-of-file is the
/// allocator's cursor, not something a read follows, so a superblock claiming
/// more file than exists decodes perfectly and returns every correct value — and
/// then the next session allocates from a cursor past the real end, placing a
/// block in a hole.
///
/// It is the check the three barriers in the first half of
/// [`apply_ea_append`](crate::chunk_index_inplace::apply_ea_append) exist for,
/// and the reason they were invisible to the first version of this module: they
/// separate content at end-of-file from the *superblock field that covers it*,
/// which is at address 8-ish and therefore first in address order. Every one of
/// them is caught here and nowhere else in the crate.
fn recorded_eof_covers_the_file(path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| std::format!("reading the file back: {e}"))?;
    // A prefix whose superblock does not parse is a *loud* state, not this
    // function's business; the read below will classify it.
    let Ok(sb) = crate::superblock::Superblock::parse(&bytes, 0) else {
        return Ok(());
    };
    if sb.eof_address > bytes.len() as u64 {
        return Err(std::format!(
            "the superblock records end-of-file at {} but the file is {} bytes: \
             every byte between is one a later session would allocate from and \
             never find",
            sb.eof_address,
            bytes.len()
        ));
    }
    Ok(())
}

/// The completed workload must have reached the length it was aiming at.
///
/// This is what the last replayed prefix is held to. Every *other* prefix
/// legitimately accepts a shorter dataset — that is what a crash mid-append
/// leaves — so without this the whole sweep is satisfied by a session that
/// appended nothing at all.
fn appended_all_the_way(path: &Path, hi: i32) -> Result<(), String> {
    let data = crate::reader::File::open(path)
        .and_then(|f| f.dataset("d")?.read_i32())
        .map_err(|e| std::format!("the finished file does not read: {e:?}"))?;
    if data.len() as i32 != hi {
        return Err(std::format!(
            "it holds {} elements rather than the {hi} the workload appended",
            data.len()
        ));
    }
    Ok(())
}

/// [`warmed_base`], with a **shuffle** filter on the dataset.
///
/// Shuffle rather than a compressor because this fixture has to stay wide: the
/// sweeps demand a file spanning [`MIN_PAGES`] gather pages, and the values here
/// are `0..n`, which deflate takes to a small fraction of a page. Shuffle stores
/// a chunk at its full size, so the file is as big as the unfiltered one — and
/// it makes the element a filtered record all the same (client id 1: address +
/// stored size + filter mask), which is what these sweeps are about.
///
/// It is also the most adversarial shape for the trailing-chunk rewrite of issue
/// #393. A re-encoded shuffle chunk is exactly as long as the one it replaces,
/// so an implementation that overwrote the old slot rather than allocating a
/// fresh one would always "fit" — and the prefix in the middle of that overwrite
/// is what the sweep below reads.
fn warmed_shuffled_base(path: &Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..CHUNK as i32).collect::<Vec<_>>())
        .with_shape(&[CHUNK])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK])
        .with_shuffle();
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let mut n = ROUND;
    while n < WARMUP {
        append(&mut s, n, ROUND);
        n += ROUND;
    }
    drop(s);
}

/// A *filtered* dataset grown by appends that leave a partial trailing chunk,
/// which the next append then re-encodes (issue #393).
///
/// Each round appends 30 elements and then 34: the first opens a partial chunk,
/// the second decodes it, extends it, re-encodes it into a fresh allocation and
/// repoints the index element that a reader at the old dimension is still
/// following. That repoint is the only write in this crate that changes where an
/// *already visible* chunk lives, so every prefix of it has to read back as the
/// old length with the old values — which is what the module's `Silent` rule
/// says, applied to a rewrite rather than an insert.
///
/// Positioned like the unfiltered sweeps, so the Extensible-Array block barriers
/// are crossed here too; a filtered array reaches them at the same element
/// indices, since the block geometry counts elements rather than bytes. Deleting
/// the barrier before a fresh data-block pointer reddens this sweep, which is
/// what says it is a sweep rather than decoration.
///
/// One thing it cannot see, and the reason the rewrite allocates rather than
/// overwriting: the re-encoded chunk is a single contiguous write, and this
/// module replays whole writes atomically, so an implementation that overwrote
/// the old slot when the new bytes happened to fit would pass here. The argument
/// against that one is the disk's rather than the log's — a torn sector inside
/// the old slot leaves a reader decoding a mix of two encodings, with no crash
/// point this harness can position.
#[test]
fn regrowing_a_filtered_trailing_chunk_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_filtered.h5");
    warmed_shuffled_base(&path);

    let rec = Recording::of("append-filtered-tail", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            let base = WARMUP + r * ROUND;
            append(&mut s, base, 30);
            append(&mut s, base + 30, ROUND - 30);
        }
        drop(s);
    });

    rec.assert_positioned(2, 1);
    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// The same sweep on a paged file, where the free-space managers are written and
/// re-homed at close and the allocator rounds to a page. That teardown is a
/// second set of low-address publish points, at the end of a session rather than
/// of an operation.
#[test]
fn appending_to_a_paged_file_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_paged.h5");
    warmed_base(&path, true);

    let rec = Recording::of("append-paged", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        s.finalize_persist().unwrap();
        drop(s);
    });

    rec.assert_positioned(2, 1);
    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// A file left by a crash must not merely *read* — it must be usable. Every
/// prefix that reads cleanly is reopened, appended to, and read again, which is
/// what catches a state that decodes but whose index no longer describes where
/// the next element goes.
///
/// This is the only sweep that catches the barrier before a fresh **super**
/// block. That one publishes a pointer into the index block while the block it
/// names is still in the buffer, and the dimension covering it has not been
/// written, so no read follows the pointer and every prefix looks perfect. The
/// next append follows it, reads an address out of bytes that were never
/// written, and tries to write eight bytes at 4397202087197605906.
#[test]
fn a_crashed_append_can_be_reopened_and_appended_to() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recover.h5");
    warmed_base(&path, false);

    const RECOVER_ROUNDS: i32 = ROUNDS;
    // Chunks, not elements. The recovery has to reach past the region the
    // crashed session was allocating rather than filling in behind it, and a
    // data block is `data_blk_min_elmts` = 16 *chunks*, so this crosses two of
    // them. Written as a count of `ROUND` because it was once a bare 40 back
    // when a chunk was one element, and widening `CHUNK` silently turned it into
    // two thirds of a single chunk — an append that advanced the dataset barely
    // at all while its comment still claimed to cross a block.
    const RECOVER_CHUNKS: i32 = 40;
    const RECOVER_APPEND: i32 = RECOVER_CHUNKS * ROUND;
    let rec = Recording::of("recover", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..RECOVER_ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        drop(s);
    });

    rec.assert_positioned(2, 1);
    let hi = WARMUP + RECOVER_ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| {
        // Read it first: a file that will not open is loud, and there is nothing
        // to recover.
        match appended_prefix_is_intact(p, WARMUP, hi) {
            Verdict::Clean => {}
            other => return other,
        }
        let before = crate::reader::File::open(p)
            .and_then(|f| f.dataset("d")?.read_i32())
            .expect("just read it")
            .len() as i32;
        // Caught rather than allowed to unwind, because one shape of this defect
        // panics rather than returning: a super-block pointer published without
        // the block gives the next append a garbage address read out of bytes
        // that were never written, and the image asserts on the write rather
        // than performing it. Letting that abort the sweep would report the
        // first occurrence and name no prefix; caught, every one is listed with
        // the write that produced it.
        let reopened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled)?;
            s.set_sync_policy(SyncPolicy::OnClose);
            let mut b = AppendBuilder::new();
            b.append_i32(&(before..before + RECOVER_APPEND).collect::<Vec<_>>());
            s.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &b, 4)?;
            drop(s);
            Ok::<(), Error>(())
        }));
        // A file that read cleanly and then cannot be written to is *not* the
        // benign outcome, however loudly the append fails. The read said the file
        // was whole; the failure says an index pointer names a block that is not
        // there, which no read follows because the dimension covering it was
        // never published. That is the shape the ordering barriers exist to
        // prevent, and it is only ever visible from here.
        match reopened {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Verdict::Silent(std::format!(
                    "reads cleanly at {before} elements, but reopening and appending fails: {e:?}"
                ));
            }
            Err(panic) => {
                let why = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("(non-string panic)")
                    .to_string();
                return Verdict::Silent(std::format!(
                    "reads cleanly at {before} elements, but reopening and appending panics: {why}"
                ));
            }
        }
        appended_prefix_is_intact(p, before + RECOVER_APPEND, before + RECOVER_APPEND)
        },
        |p| appended_all_the_way(p, hi),
    );
}

/// Bytes of hole the reserved-append sweep leaves on the disk for the recorded
/// session to draw on.
///
/// It has to exceed one reserve batch, or the session declines to draw at all
/// and the sweep is back to covering an ordinary end-of-file append. Twice the
/// batch, so the draw leaves a remainder and the close-time rewrite that hands it
/// back is inside the window too.
const RESERVE_HOLE: usize = 2 << 20;

/// Recorded rounds for that sweep. Fewer than [`ROUNDS`], because every prefix
/// writes out a file that carries the hole above.
const RESERVE_ROUNDS: i32 = 20;

/// [`warmed_base`] on a file that persists its free space, with a hole of
/// [`RESERVE_HOLE`] bytes already recorded in the on-disk managers when the
/// recorded window opens.
///
/// The delete that makes the hole runs in *this* session, which closes before the
/// recording starts. That is what lets the sweep reach the reuse branch without a
/// commit inside the window: a persisting session seeds its free list from the
/// managers at open, so the recorded session finds the hole rather than making
/// it.
fn warmed_base_with_a_persisted_hole(path: &Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.create_dataset("d")
        .with_i32_data(&[0i32])
        .with_shape(&[1])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK]);
    b.create_dataset("victim")
        .with_i32_data(&vec![7i32; RESERVE_HOLE / 4]);
    // Above the hole, so deleting the victim leaves an interior region rather
    // than a run reaching end-of-file that the commit would truncate away.
    b.create_dataset("ceiling").with_i32_data(&[9i32, 9, 9]);
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let mut n = 1i32;
    while n < WARMUP {
        let take = ROUND.min(WARMUP - n);
        append(&mut s, n, take);
        n += take;
    }
    s.delete(&ObjectPathBuf::parse("victim")).unwrap();
    s.commit().unwrap();
    drop(s);
}

/// No region the on-disk free-space managers advertise may overlap a chunk of
/// `d`, or the survivor beside it.
///
/// This is what a reservation buys, stated as the thing a crash must not be able
/// to break: the managers are a durable record that the *next* session allocates
/// out of, so one offering space a live chunk occupies is not a stale number but
/// a file the next writer corrupts. A prefix caught between the reservation's
/// repoint and the bytes it lets the append write must still satisfy it, and so
/// must one caught before the reservation at all.
fn persisted_free_space_holds_nothing_live(path: &Path) -> Result<(), String> {
    let Ok(f) = crate::reader::File::open(path) else {
        // A prefix whose file does not open at all is the read's business, not
        // this one's; it is classified there.
        return Ok(());
    };
    let free = f.persisted_free_space();
    let Ok(chunks) = f.dataset("d").and_then(|d| d.chunks()) else {
        return Ok(());
    };
    for c in chunks {
        let (lo, hi) = (c.address, c.address + c.storage_size);
        for &(addr, len) in &free {
            if addr < hi && lo < addr + len {
                return Err(std::format!(
                    "the managers advertise [{addr}, {}) as free, which overlaps a live \
                     chunk of `d` at [{lo}, {hi})",
                    addr + len
                ));
            }
        }
    }
    Ok(())
}

/// One prefix of the reserved-append sweep: the dataset reads as a prefix of
/// itself, and the managers describe nothing live.
fn reserved_append_prefix_is_sound(path: &Path, lo: i32, hi: i32) -> Verdict {
    match appended_prefix_is_intact(path, lo, hi) {
        Verdict::Clean => {}
        other => return other,
    }
    // The survivor above the hole, which a reservation that handed out the wrong
    // bytes would be written over.
    let ceiling = crate::reader::File::open(path).and_then(|f| f.dataset("ceiling")?.read_i32());
    match ceiling {
        Ok(v) if v == vec![9, 9, 9] => {}
        Ok(v) => return Verdict::Silent(std::format!("`ceiling` reads as {v:?}")),
        Err(e) => return Verdict::Loud(std::format!("{e:?}")),
    }
    match persisted_free_space_holds_nothing_live(path) {
        Ok(()) => Verdict::Clean,
        Err(why) => Verdict::Silent(why),
    }
}

/// An in-place append on a file that persists its free space takes the space it
/// spends *out* of the on-disk managers first, with a superblock repoint of its
/// own, and only then writes into it (issue #387). A crash at any instant of that
/// sequence must leave a file whose managers describe nothing live.
///
/// The reservation is a commit in the middle of an append run, so it is the one
/// publish point in this module that is neither an append's own phase nor a
/// session's teardown. The states it can be caught between are the interesting
/// ones: after the draw and before the repoint, where the managers still list the
/// batch and nothing has been written into it, and after the repoint and before
/// the append, where they do not and it has not.
#[test]
fn appending_into_a_reserved_hole_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reserved.h5");
    warmed_base_with_a_persisted_hole(&path);
    let before = std::fs::metadata(&path).unwrap().len();

    let rec = Recording::of("reserved", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..RESERVE_ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        drop(s);
    });

    // The window has to have *reused* the hole, or this sweeps the ordinary
    // end-of-file append under a longer file. Every appended byte landing inside
    // the file is what says the reservation ran.
    let after = std::fs::metadata(&path).unwrap().len();
    let appended = (RESERVE_ROUNDS * ROUND) as u64 * 4;
    assert!(
        after < before + appended / 2,
        "the recorded session grew the file from {before} to {after} while appending \
         {appended} bytes, so it appended at end-of-file and no reservation was swept"
    );
    rec.assert_positioned(1, 0);

    let hi = WARMUP + RESERVE_ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| reserved_append_prefix_is_sound(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

// ---------------------------------------------------------------------------
// The commit workload
// ---------------------------------------------------------------------------

/// Churn rounds in one recorded commit session.
const CHURN: i32 = 6;
/// Elements `d` starts the commit workload with, and gains per round.
///
/// Sized so the file spans well over [`MIN_PAGES`] gather pages, for the reason
/// [`Recording::assert_positioned`] gives: at the 64 and 16 this used to be, the
/// whole file was under 4 KB, every write merged with every other, and the sweep
/// could not observe an inversion it nonetheless produced.
const COMMIT_BASE: i32 = 4096;
const COMMIT_STEP: i32 = 1024;

/// Elements of `added{r}`, and of the `doomed{r}` it replaces. Different lengths
/// so a round that resurrected the wrong object reads as the wrong length rather
/// than as plausible data.
fn added_values(r: i32) -> Vec<i32> {
    (0..2048 + r).map(|i| i * 7 + r).collect()
}
fn doomed_values(r: i32) -> Vec<i32> {
    (0..1024 + r).map(|i| i * 3 + r).collect()
}

/// Elements of `slot` after `g` rounds have committed. `slot` is deleted and
/// recreated at its own path in one commit (issue #305), so unlike `added`/
/// `doomed` it is never absent — the generation is what says which commit a
/// prefix caught. Lengths differ per generation, so a prefix that resurrected
/// the previous object reads as the wrong length rather than as plausible data.
fn slot_values(g: i32) -> Vec<i32> {
    (0..512 + 37 * g).map(|i| i * 11 + g).collect()
}

/// Build the file the commit workload churns: one growable dataset, and one
/// `doomed{r}` per round for the delete half to consume.
fn commit_base(path: &Path, paged: bool) {
    let mut b = FileBuilder::new();
    if paged {
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(4096);
    }
    b.create_dataset("d")
        .with_i32_data(&(0..COMMIT_BASE).collect::<Vec<i32>>())
        .with_shape(&[COMMIT_BASE as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[COMMIT_STEP as u64]);
    for r in 0..CHURN {
        let v = doomed_values(r);
        b.create_dataset(&std::format!("doomed{r}"))
            .with_i32_data(&v)
            .with_shape(&[v.len() as u64]);
    }
    let v = slot_values(0);
    b.create_dataset("slot")
        .with_i32_data(&v)
        .with_shape(&[v.len() as u64]);
    b.write(path).unwrap();
}

/// Read a dataset that may legitimately not be in the file, distinguishing "no
/// such path" from "the path is there and will not decode".
///
/// The difference matters and is easy to lose: mapping every error to `None`
/// turns a commit that published a link to a half-written object header — a
/// genuinely torn commit — into "this round has not committed yet", which every
/// rule below accepts.
fn read_optional(f: &crate::reader::File, path: &str) -> Result<Option<Vec<i32>>, Error> {
    match f.dataset(path) {
        Ok(ds) => ds.read_i32().map(Some),
        Err(Error::Format(FormatError::PathNotFound { .. })) => Ok(None),
        Err(e) => Err(e),
    }
}

/// `d`'s elements, then each round's `added{r}` and `doomed{r}` — `None` where
/// the path is not in the file at all — and finally `slot`, which every prefix
/// must have.
type CommitState = (
    Vec<i32>,
    Vec<Option<Vec<i32>>>,
    Vec<Option<Vec<i32>>>,
    Vec<i32>,
);

/// A commit publishes at its superblock write, so every object must read as
/// either its pre-commit or its post-commit state — never a mixture, and never
/// something that is neither.
///
/// Each round creates one object and deletes another, so a later round reuses
/// space an earlier one freed (issue #261). That is what makes "reads as
/// something that is neither" reachable rather than theoretical: a header
/// published over a region a previous commit released is a *readable* file whose
/// dataset returns the deleted object's bytes.
fn commit_state_is_one_or_the_other(path: &Path) -> Verdict {
    let read = (|| -> Result<CommitState, Error> {
        let f = crate::reader::File::open(path)?;
        let d = f.dataset("d")?.read_i32()?;
        let mut added = Vec::new();
        let mut doomed = Vec::new();
        for r in 0..CHURN {
            added.push(read_optional(&f, &std::format!("added{r}"))?);
            doomed.push(read_optional(&f, &std::format!("doomed{r}"))?);
        }
        // Not `read_optional`: a replacement leaves the path occupied at every
        // instant, so "no such path" is a failure here rather than a state, and
        // it has to reach the judge as one. `dataset` returning `PathNotFound`
        // is an `Error`, which `Verdict::of` would file as `Loud` — which
        // `Tally::assert_sound` also fails on, but under a reason about the file
        // refusing to read rather than about the path being gone.
        let slot = match f.dataset("slot") {
            Ok(ds) => ds.read_i32()?,
            Err(Error::Format(FormatError::PathNotFound { .. })) => Vec::new(),
            Err(e) => return Err(e),
        };
        Ok((d, added, doomed, slot))
    })();

    Verdict::of(read, |(d, added, doomed, slot)| {
        // `d` is judged on its own. An in-place append publishes at its own
        // phase 4, before the staged commit in the same round runs, so its length
        // says nothing about whether that commit landed — the two are separate
        // publish points and this must not conflate them.
        let n = d.len() as i32;
        if !(COMMIT_BASE..=COMMIT_BASE + CHURN * COMMIT_STEP).contains(&n) {
            return Err(std::format!(
                "`d` has {n} elements, outside the {COMMIT_BASE}..={} this session can produce",
                COMMIT_BASE + CHURN * COMMIT_STEP
            ));
        }
        if let Some(i) = (0..d.len()).find(|&i| d[i] != i as i32) {
            return Err(std::format!("`d`[{i}] is {} rather than {i}", d[i]));
        }

        for r in 0..CHURN {
            if let Some(a) = &added[r as usize] {
                if *a != added_values(r) {
                    return Err(std::format!(
                        "`added{r}` is present with the wrong bytes (len {}, wanted {})",
                        a.len(),
                        added_values(r).len()
                    ));
                }
            }
            if let Some(x) = &doomed[r as usize] {
                if *x != doomed_values(r) {
                    return Err(std::format!(
                        "`doomed{r}` is still present but its bytes have changed"
                    ));
                }
            }
            // The create and the delete are staged together and published by one
            // superblock write, so they land together or not at all. This is the
            // commit's atomicity, and the one thing a torn commit cannot satisfy.
            if added[r as usize].is_some() != doomed[r as usize].is_none() {
                return Err(std::format!(
                    "round {r} is half-committed: added{r} {}, doomed{r} {}",
                    if added[r as usize].is_some() {
                        "present"
                    } else {
                        "absent"
                    },
                    if doomed[r as usize].is_some() {
                        "present"
                    } else {
                        "absent"
                    }
                ));
            }
        }
        // Rounds commit in order, so the committed ones are a prefix. A later
        // round present while an earlier one is missing means a commit was
        // published over a state it did not build on.
        let committed: Vec<bool> = (0..CHURN).map(|r| added[r as usize].is_some()).collect();
        if let Some(r) = (1..CHURN as usize).find(|&r| committed[r] && !committed[r - 1]) {
            return Err(std::format!(
                "round {r} committed but round {} did not, though {r} came second",
                r - 1
            ));
        }

        // `slot` is deleted and recreated at its own path in the same commit, so
        // it is occupied at every instant: a prefix where the path is gone is a
        // window in which a replacement was two operations rather than one.
        if slot.is_empty() {
            return Err("`slot` is not in the file, though every commit that \
                        removes it puts its replacement back in the same commit"
                .to_string());
        }
        // And it is occupied by the generation this prefix's *other* evidence
        // says: rounds commit in order and each round replaces `slot` once, so
        // after `c` committed rounds `slot` holds generation `c`. This is the
        // rule that makes a replacement one linearization point rather than
        // merely an atomic-looking pair — a `slot` a generation ahead of or
        // behind the round count would mean the delete and the create were
        // published at different instants.
        let c = committed.iter().filter(|&&b| b).count() as i32;
        if slot != slot_values(c) {
            let found = (0..=CHURN).find(|&g| slot == slot_values(g));
            return Err(match found {
                Some(g) => {
                    std::format!("`slot` holds generation {g} while {c} round(s) have committed")
                }
                None => std::format!(
                    "`slot` holds {} elements, which is no generation this session writes \
                     ({c} round(s) have committed, so generation {c} was expected)",
                    slot.len()
                ),
            });
        }
        Ok(())
    })
}

/// Every round of the commit workload must actually have committed, or the sweep
/// above is judging interruptions of a session that did nothing. `d` must also
/// have grown by every round's append.
fn every_round_committed(path: &Path) -> Result<(), String> {
    let f = crate::reader::File::open(path)
        .map_err(|e| std::format!("the finished file does not open: {e:?}"))?;
    let d = f
        .dataset("d")
        .and_then(|d| d.read_i32())
        .map_err(|e| std::format!("the finished `d` does not read: {e:?}"))?;
    let want = COMMIT_BASE + CHURN * COMMIT_STEP;
    if d.len() as i32 != want {
        return Err(std::format!(
            "`d` holds {} elements rather than {want}",
            d.len()
        ));
    }
    for r in 0..CHURN {
        if f.dataset(&std::format!("added{r}")).is_err() {
            return Err(std::format!("`added{r}` was never created"));
        }
        if f.dataset(&std::format!("doomed{r}")).is_ok() {
            return Err(std::format!("`doomed{r}` was never deleted"));
        }
    }
    let slot = f
        .dataset("slot")
        .and_then(|d| d.read_i32())
        .map_err(|e| std::format!("the finished `slot` does not read: {e:?}"))?;
    if slot != slot_values(CHURN) {
        return Err(std::format!(
            "`slot` holds {} elements rather than generation {CHURN}'s {}",
            slot.len(),
            slot_values(CHURN).len()
        ));
    }
    Ok(())
}

/// One recorded churn round: grow a dataset in place, create another, delete a
/// third, and commit. The three edits take different routes to the disk — the
/// append patches the header in place, the create and the delete both go through
/// the staged commit — so one recording covers all of them.
fn churn(s: &mut WriteEngine, r: i32) {
    let from = COMMIT_BASE + r * COMMIT_STEP;
    let mut b = AppendBuilder::new();
    b.append_i32(&(from..from + COMMIT_STEP).collect::<Vec<_>>());
    s.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &b, 4)
        .unwrap();

    let v = added_values(r);
    let mut db = DatasetBuilder::new(&std::format!("added{r}"));
    db.with_i32_data(&v).with_shape(&[v.len() as u64]);
    s.stage_created_dataset(&ObjectPathBuf::parse(&std::format!("/added{r}")), db)
        .unwrap();
    s.delete(&ObjectPathBuf::parse(&std::format!("doomed{r}")))
        .unwrap();

    // Delete and recreate one path in the *same* commit (issue #305), beside the
    // unrelated create and delete above. This is the edit that has no two-commit
    // intermediate state to fall into, so a prefix where `slot` is missing is a
    // real failure rather than a legitimate half-done rotation.
    s.delete(&ObjectPathBuf::parse("slot")).unwrap();
    let v = slot_values(r + 1);
    let mut sb = DatasetBuilder::new("slot");
    sb.with_i32_data(&v).with_shape(&[v.len() as u64]);
    s.stage_created_dataset(&ObjectPathBuf::parse("/slot"), sb)
        .unwrap();

    s.commit().unwrap();

    // The workload has to have done all three, or the sweep is judging a session
    // that quietly did nothing.
    assert!(!s.has_staged_edits(), "the commit left edits staged");
}

/// The staged-commit path, swept over both backings and both file-space
/// strategies. A commit's publish point is the superblock at address zero, which
/// is the lowest address in the file and therefore the first thing gathering
/// wants to issue.
#[test]
fn committing_survives_a_crash_at_every_write() {
    for paged in [false, true] {
        for bounded in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("commit.h5");
            commit_base(&path, paged);
            let label = std::format!(
                "commit-{}-{}",
                if paged { "paged" } else { "plain" },
                if bounded { "bounded" } else { "mirrored" }
            );

            let rec = Recording::of(&label, &path, |p| {
                let mut s = if bounded {
                    WriteEngine::open_rw_with_strategy(
                        p,
                        MetadataCacheConfig::new(64 * 1024),
                        FileLocking::Enabled,
                        MemoryStrategy::Bounded,
                    )
                    .unwrap()
                } else {
                    WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap()
                };
                s.set_sync_policy(SyncPolicy::OnClose);
                for r in 0..CHURN {
                    churn(&mut s, r);
                }
                if paged {
                    s.finalize_persist().unwrap();
                }
                drop(s);
            });

            rec.assert_positioned(1, 0);
            rec.replay_every_prefix(
                dir.path(),
                commit_state_is_one_or_the_other,
                every_round_committed,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Gathering against not gathering
// ---------------------------------------------------------------------------

/// The guarantee does not depend on the write gathering.
///
/// It once did. A publish was a value write followed by a checksum write, joined
/// into one only where the gatherer found the two in the same page, so an
/// unbuffered session left this sweep 20 unreadable states in 168 — every one a
/// checksum torn from the value it covered. Since issue #307 the engine publishes
/// a checksummed structure as one write itself, and both configurations sweep
/// clean.
///
/// Running both is what keeps it that way. A publish point that regressed to two
/// writes would pass the gathered sweep for as long as the two shared a page, and
/// fail here at once — which is exactly how #307 hid: it was reachable only on a
/// dataset whose object header outgrew one page.
#[test]
fn crash_states_read_back_with_and_without_write_gathering() {
    use crate::image::WriteBuffering;

    const ROUNDS_EACH: i32 = 20;
    let hi = WARMUP + ROUNDS_EACH * ROUND;

    let sweep = |label: &str, mode: Option<WriteBuffering>| -> Tally {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compare.h5");
        warmed_base(&path, false);
        let rec = Recording::of(label, &path, |p| {
            let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
            s.set_sync_policy(SyncPolicy::OnClose);
            if let Some(mode) = mode {
                s.set_write_buffering(mode).unwrap();
            }
            for r in 0..ROUNDS_EACH {
                append(&mut s, WARMUP + r * ROUND, ROUND);
            }
            drop(s);
        });
        rec.assert_positioned(1, 0);
        rec.replay_every_prefix(
            dir.path(),
            |p| appended_prefix_is_intact(p, WARMUP, hi),
            |p| appended_all_the_way(p, hi),
        )
    };

    // `None` keeps the session default, which is the gathering a locked
    // read-write session takes.
    let gathered = sweep("compare-gathered", None);
    let straight = sweep("compare-unbuffered", Some(WriteBuffering::Unbuffered));

    // Both sweeps assert the rule inside `replay_every_prefix`, so what is left
    // to pin here is that the second one is the *finer* of the two. Unbuffered
    // writing issues each patch separately, so it stops the machine at instants
    // the gathered sweep never reaches; if the two ever swept the same number of
    // prefixes, the mode would be being ignored and this whole comparison would
    // be one sweep run twice.
    assert!(
        straight.total > gathered.total,
        "the unbuffered sweep should stop at more instants than the gathered one, \
         but swept {} prefixes against {}",
        straight.total,
        gathered.total
    );
}

/// A super block is published one data-block pointer at a time, and the *second*
/// such publish lands in a super block a reader can already see — so a checksum
/// torn from the pointer it covers is a state a reader reaches, and since #312 a
/// state it refuses.
///
/// The first pointer into a *fresh* super block is not that state. The block
/// becomes reachable only once the header's element count is published, and by
/// then its checksum is whole — the ordering hides the window rather than the
/// reader missing it. So this sweep starts *past* the super block's own
/// allocation instead of across it, which is what the sweeps positioned at the
/// warm-up boundary do.
///
/// Unbuffered for the reason [`crash_states_read_back_with_and_without_write_gathering`]
/// gives: a super block is small, so gathering merges the two writes of a split
/// publish into one and hides exactly what this exists to catch.
#[test]
fn publishing_into_a_live_super_block_survives_a_crash_at_every_write() {
    use crate::image::WriteBuffering;

    /// The round to start recording at. With the C library's defaults a fresh
    /// data block is allocated at chunk 244 — which is also where the super
    /// block holding it is created — and the next at 308. Recording 300..=320
    /// therefore crosses a data-block allocation inside a super block that has
    /// held live elements since chunk 244.
    const START_ROUNDS: i32 = 300;
    const ROUNDS_EACH: i32 = 20;
    let lo = START_ROUNDS * ROUND;
    let hi = lo + ROUNDS_EACH * ROUND;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live_super.h5");
    warmed_base(&path, false);
    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        let mut n = WARMUP;
        while n < lo {
            append(&mut s, n, ROUND);
            n += ROUND;
        }
        drop(s);
    }

    let rec = Recording::of("live-super", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        s.set_write_buffering(WriteBuffering::Unbuffered).unwrap();
        for r in 0..ROUNDS_EACH {
            append(&mut s, lo + r * ROUND, ROUND);
        }
        drop(s);
    });
    // A floor of one data-block allocation. `assert_positioned` states only
    // floors, so the super-block *ceiling* is its own assertion: a window that
    // allocated one would be back at the case the ordering already hides, and
    // would sweep clean however the publish were written.
    rec.assert_positioned(1, 0);
    assert_eq!(
        rec.super_blocks, 0,
        "the window must publish *into* a live super block, not allocate one"
    );
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, lo, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// Bytes of attribute padding that push the dataspace dimension and its
/// object-header checksum into different gather pages. Measured threshold on the
/// default 4096-byte page: 3800 still joins, 3900 splits.
const HEADER_PAD: usize = 4096;

/// An in-place append publishes its new dimension and the checksum covering it.
/// A header chunk wider than one gather page splits those into two writes, and a
/// crash between them leaves the file **unreadable** — the new value under the
/// old checksum (issue #307).
#[test]
fn publishing_a_dimension_in_a_wide_header_survives_a_crash_at_every_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wide_header.h5");
    warmed_base_padded(&path, true, HEADER_PAD);

    let rec = Recording::of("wide-header", &path, |p| {
        let mut s = WriteEngine::open_with_locking(p, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        for r in 0..ROUNDS {
            append(&mut s, WARMUP + r * ROUND, ROUND);
        }
        s.finalize_persist().unwrap();
        drop(s);
    });

    rec.assert_publishes_across_a_gather_page();
    let hi = WARMUP + ROUNDS * ROUND;
    rec.replay_every_prefix(
        dir.path(),
        |p| appended_prefix_is_intact(p, WARMUP, hi),
        |p| appended_all_the_way(p, hi),
    );
}

/// Publishing costs the same number of writes whether or not the structure being
/// published spans a gather page — which is the defect of issue #307 stated as a
/// rule rather than as one of its consequences.
///
/// The sweep above catches the consequence, and only for the object header: it
/// reads the file back, so it sees a torn publish only where a reader follows the
/// torn field. This sees the split itself, so it covers every publish an append
/// makes — the array header's statistics, an index block's element and its data-
/// block pointers, a data block's element — without needing a workload that
/// reads each one.
///
/// Measured before the fix: 5 writes at 3800 bytes of padding against 6 at 3900,
/// the extra one being the checksum stranded in the next page. Both buffering
/// modes are checked, because the unbuffered one splits every publish regardless
/// of where the pages fall.
#[test]
fn a_publish_costs_the_same_whether_or_not_it_spans_a_page() {
    use crate::image::WriteBuffering;

    for mode in [None, Some(WriteBuffering::Unbuffered)] {
        let mut counts = Vec::new();
        // A sweep rather than the two sides of the measured threshold: picking
        // the boundary by hand is how a fixture ends up just short of the case it
        // was written for, and stepping across it costs milliseconds.
        for pad in [0usize, 2000, 3800, 3900, 4200, 8300] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("pad.h5");
            warmed_base_padded(&path, true, pad);

            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            s.set_sync_policy(SyncPolicy::OnClose);
            if let Some(mode) = mode {
                s.set_write_buffering(mode).unwrap();
            }
            let before = s.issued_write_order().len();
            append(&mut s, WARMUP, ROUND);
            let made = s.issued_write_order()[before..].to_vec();
            drop(s);
            // Whether this padding put the publish across a page: a write that
            // starts inside the first gather page and reaches past it.
            let wide = made
                .iter()
                .any(|&(off, len)| off < GATHER_PAGE && off + len > GATHER_PAGE);
            counts.push((pad, made.len(), wide));
        }
        // The comparison is only meaningful across the boundary. All-narrow or
        // all-wide passes with the defect present — measured — so the sweep has to
        // say it landed on both sides rather than be trusted to.
        assert!(
            counts.iter().any(|&(_, _, w)| w) && counts.iter().any(|&(_, _, w)| !w),
            "{mode:?}: every padding fell on the same side of the {GATHER_PAGE}-byte \
             page, so this comparison holds nothing: {counts:?}"
        );
        let (_, first, _) = counts[0];
        assert!(
            counts.iter().all(|&(_, n, _)| n == first),
            "{mode:?}: one append must cost the same writes at every header width, \
             but cost {counts:?}"
        );
    }
}

/// A publish being one write leaves the gathering almost nothing to merge inside
/// an append, so the two buffering modes must cost writes that differ by a
/// **constant** rather than by a per-append one.
///
/// This is the half [`a_publish_costs_the_same_whether_or_not_it_spans_a_page`]
/// cannot see. That one compares header widths, so a publish that splits at
/// *every* width — the array header's six statistics and its checksum are fifty
/// bytes apart and share a page whatever the dataset looks like — changes both
/// sides of its comparison and passes. Splitting shows up here instead, as a gap
/// that grows with the number of appends.
///
/// Measured: 5 writes against 5 for one append and 40 against 40 for eight,
/// unpaged; one more on the unbuffered side throughout when the file is paged,
/// from padding a page at the raw/metadata boundary. Before issue #307 the same
/// eight appends were 40 against 64, and the excess was three writes per append.
#[test]
fn buffering_saves_a_constant_number_of_writes_not_one_per_append() {
    use crate::image::WriteBuffering;

    /// Room for the handful of writes gathering merges that are *not* per-append:
    /// a paged file's boundary padding, and the joins around an allocation round.
    const SLACK: usize = 4;

    for paged in [false, true] {
        let cost = |rounds: i32, unbuffered: bool| -> usize {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("gap.h5");
            warmed_base(&path, paged);
            let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
            s.set_sync_policy(SyncPolicy::OnClose);
            if unbuffered {
                s.set_write_buffering(WriteBuffering::Unbuffered).unwrap();
            }
            for i in 0..rounds {
                append(&mut s, WARMUP + i * ROUND, ROUND);
            }
            let n = s.issued_write_order().len();
            drop(s);
            n
        };

        // Two append counts an order of magnitude apart: a per-append excess
        // grows between them and a constant one does not, which is the whole
        // distinction. Comparing one count against an absolute would pin a number
        // that every unrelated change to the append path moves.
        let (few, many) = (2i32, 24i32);
        let gap = |rounds: i32| cost(rounds, true).saturating_sub(cost(rounds, false));
        let (gap_few, gap_many) = (gap(few), gap(many));
        // A floor as well as a ceiling. If `set_write_buffering` silently did
        // nothing, both sides would be the same session and every gap would be
        // zero, which satisfies the rule below without measuring anything.
        assert!(
            gap_many > 0,
            "paged={paged}: unbuffered writing must cost more writes than gathered \
             over {many} appends, but the two were equal — the buffering mode is \
             not taking effect and nothing below is being measured"
        );
        assert!(
            gap_many <= gap_few + SLACK,
            "paged={paged}: buffering must save a constant number of writes, but saved \
             {gap_few} over {few} appends and {gap_many} over {many} — a gap that grows \
             with the appends is a publish that is still two writes"
        );
    }
}

/// A publish writes back from the byte it changed, not from the front of the
/// structure that byte sits in.
///
/// Two appends into the same Extensible-Array data block touch a slot one
/// element further along each time, so the second writes strictly fewer bytes
/// than the first. Rewriting the whole block would make the two equal — which is
/// what this test exists to fail on, since the write *count* is the same either
/// way and every other check here counts writes.
///
/// Measured on this fixture: 531 bytes then 523, against 992 and 992 when the
/// publish starts at the structure instead. The block, not the object header, is
/// where the saving is: 534 bytes of index block against the 116 the first of
/// these appends actually changes.
#[test]
fn a_publish_writes_from_the_byte_it_changed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("from.h5");
    warmed_base(&path, false);

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    let written = |s: &WriteEngine| -> u64 { s.issued_write_order().iter().map(|&(_, n)| n).sum() };
    let mut cost = Vec::new();
    for i in 0..2 {
        let before = written(&s);
        append(&mut s, WARMUP + i * ROUND, ROUND);
        cost.push(written(&s) - before);
    }
    drop(s);
    assert!(
        cost[1] < cost[0],
        "an append that touches a later slot of the same block must write fewer \
         bytes than one that touches an earlier slot, but the two cost {cost:?}"
    );
}
