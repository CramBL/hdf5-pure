//! The byte image a mutating session edits: the [`FileImage`] trait and its
//! whole-file mirror backend.
//!
//! # Why this exists
//!
//! [`Source`] describes how the *reader* gets bytes out of a file. This is its
//! write-side counterpart: what a [`WriteEngine`](crate::edit::WriteEngine)
//! needs of the bytes it is editing — the same random-access reads, plus the
//! four primitives a commit performs (append at end-of-file, overwrite in
//! place, truncate, and force durability).
//!
//! Separating it from the engine is what lets one engine drive two very
//! different backings (issue #198). Historically `File::open_rw` held the whole
//! file in a `Vec<u8>` mirror and carried the full staged-edit vocabulary,
//! while the bounded read-write open held no mirror and could therefore only read
//! and append. *That* limitation is a property of where the bytes live, not of
//! what edits are expressible, so it belongs behind this trait rather than in
//! two engines.
//!
//! The trait abstracts residency and nothing else. Which files each backing can
//! open at all still differs, the bounded one refusing a v0/v1 superblock, 4-byte
//! offsets, and any userblock, but that is no longer a difference between two
//! entry points: `File::open_rw` picks a backing per file and mirrors what the
//! bounded one turns down (issue #198, step 4).
//!
//! # Reads
//!
//! [`FileImage`] extends [`Source`], so every parser the edit engine drives
//! reads through the same interface it already uses. [`as_slice`](FileImage::as_slice)
//! is the one concession to backing: a mirror can hand out its whole buffer,
//! letting a slice-walking parser borrow rather than copy, and a file-backed
//! image cannot. Callers that can exploit a slice should, and must have a
//! `Source` path for when there is none.
//!
//! # The end-of-file cursor
//!
//! [`Source::len`] is the image's end-of-file: [`append`](FileImage::append)
//! places bytes there and extends it by exactly their length. For the mirror
//! that is the buffer length; for a file-backed image it is a counter seeded
//! from the file's real length at open.
//!
//! Callers depend on the exact-extension part, not just on monotonicity: the
//! engine reads `len()`, builds a blob whose interior addresses assume it will
//! land at that address, and only then appends. An implementation that inserted
//! padding inside `append` would silently relocate every such blob, so padding
//! belongs in a layer above this one.
//!
//! # Buffered writes
//!
//! Both images reach the disk through one [`BufferedWrites`], which gathers the
//! many small writes a commit or an in-place append issues and emits one write
//! per dirty page. Where the *bytes* live still differs between the two — that is
//! what the images are — but how those bytes reach the operating system no longer
//! does (issue #288).
//!
//! [`WriteBuffering`] says how long a write may sit in memory, and every image is
//! built [`Unbuffered`](WriteBuffering::Unbuffered) until an entry point says
//! otherwise, so a path that forgets to configure it is slow rather than wrong.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::convert::Narrow;
use crate::error::{Error, FormatError};
use crate::source::{
    BytesSource, MetadataCacheConfig, MetadataCacheStats, MetadataReadCache, Source,
};

/// How long a write may sit in memory before it must reach the operating system.
///
/// An `fsync` always flushes first, so nothing here weakens durability against
/// *power* loss. What each mode trades is which intermediate states another
/// process can observe, and — for [`Session`](Self::Session) alone — the order
/// they become observable in, which is what decides whether a failed write
/// leaves the previous file or a broken one. Each variant states its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteBuffering {
    /// Every write reaches the operating system as it is made, in the order the
    /// engine issued it.
    ///
    /// This is what a lock-free session takes. A SWMR writer's ordered phases are
    /// read *concurrently*, so the order in which its writes become visible is
    /// part of the format's contract with the reader, not an implementation
    /// detail free to be coalesced away.
    Unbuffered,
    /// Dirty bytes live until the next ordering barrier, or until `max_bytes` of
    /// them accumulate, whichever comes first. Every commit and every in-place
    /// append ends with a barrier, so this also means: until the operation that
    /// wrote them finishes.
    ///
    /// This is the default for a locked session, and it leaves the ordering the
    /// engine already had: the barriers still separate content from the publish
    /// points that reach it, so a failed write leaves what it left before this
    /// gathering existed. What it stops making visible are the intermediate states
    /// *between* two barriers of one operation — which no reader outside SWMR has
    /// a contract to see, and a read-write session is normally alone with the file
    /// anyway.
    ///
    /// "Normally" because the exclusive lock is not guaranteed: `FileLocking::Disabled`,
    /// `HDF5_USE_FILE_LOCKING=FALSE`, and `BestEffort` on a filesystem that cannot
    /// lock all reach this mode without one. That costs nothing here — hiding
    /// *more* intermediate states cannot break a reader that was never promised
    /// them — but it is why the argument above rests on the contract rather than
    /// on the lock.
    ///
    /// It is not what makes a publish point atomic, and was never able to be. A
    /// value and the checksum covering it, written separately, would join here
    /// only when both landed in one page; a structure wider than that published
    /// the new value under the old checksum whichever mode was in force. That was
    /// issue #307, and the fix was to write such a structure once from the engine
    /// rather than to widen what this merges — so the guarantee now holds under
    /// [`Unbuffered`](WriteBuffering::Unbuffered) too.
    Operation { page_size: u64, max_bytes: usize },
    /// Dirty bytes live until `max_bytes` of them accumulate, an `fsync` is
    /// issued, or the session closes — spanning both operations and the ordering
    /// barriers inside them.
    ///
    /// This is the `H5Pset_page_buffer_size` analogue, and the one mode that does
    /// nothing at [`ordering_barrier`](FileImage::ordering_barrier). That is the
    /// whole of what this layer decides. What it costs is a file-format question
    /// rather than a byte-image one — held across a barrier, an operation's
    /// publish points are issued ahead of the content they name, and the file a
    /// crash then leaves can read *clean* and return the wrong bytes — so it is
    /// stated where the format lives, on
    /// [`WriteEngine::set_page_buffer_size`](crate::edit::WriteEngine::set_page_buffer_size)
    /// and on the public
    /// [`with_page_buffer_size`](crate::FileAccessProperties::with_page_buffer_size).
    ///
    /// Installing this mode is not the whole feature: the engine raises the
    /// superblock's write-access flag for the life of such a session, so those
    /// files are refused rather than read. The C library ships its page buffer
    /// without one; this mode is not offered without it (issue #308).
    ///
    /// A completed commit's bytes may still be in this process's memory when it
    /// returns. The engine refuses to pair this with
    /// [`SyncPolicy::Always`](crate::SyncPolicy::Always), where every barrier is
    /// an `fsync` that would flush it: the caller would pay the mark and hold
    /// nothing.
    Session { page_size: u64, max_bytes: usize },
}

impl WriteBuffering {
    /// The page a write is rounded to when deciding what to merge, and the byte
    /// budget; `None` when nothing is buffered at all.
    const fn budget(self) -> Option<(u64, usize)> {
        match self {
            WriteBuffering::Unbuffered => None,
            WriteBuffering::Operation {
                page_size,
                max_bytes,
            }
            | WriteBuffering::Session {
                page_size,
                max_bytes,
            } => Some((page_size, max_bytes)),
        }
    }
}

/// A recording of everything that reached the operating system, so a test can
/// replay a prefix of it and see what a crash at that instant would have left.
///
/// The log is per thread and off by default, which is what lets it live under
/// every write in the crate: an inactive thread pays one thread-local read per
/// issued write, and the lib tests that are not recording are unaffected by the
/// ones that are, however the harness schedules them.
///
/// It records only operations that *succeeded*. A write that returned an error
/// may have put any prefix of its bytes on the disk, and this cannot know which,
/// so a failed write is not a point this can replay. That is the boundary
/// between this and a fault injector: this models the machine stopping between
/// two completed operations, which is the case the crate's ordering barriers are
/// written against.
#[cfg(test)]
pub(crate) mod disk_log {
    use std::cell::RefCell;

    /// One completed operation against the file, in the order it was issued.
    pub(crate) enum DiskOp {
        /// `bytes` landed at `offset`. Positioned, so replaying it is exact
        /// wherever the file's length happens to be.
        Write { offset: u64, bytes: Vec<u8> },
        /// The file was resized to this length, discarding anything past it.
        SetLen(u64),
    }

    impl DiskOp {
        /// A short label for a failure message: what it touched, not its bytes.
        pub(crate) fn describe(&self) -> String {
            match self {
                DiskOp::Write { offset, bytes } => {
                    std::format!("write {offset}..{}", offset + bytes.len() as u64)
                }
                DiskOp::SetLen(len) => std::format!("set_len {len}"),
            }
        }
    }

    thread_local! {
        /// `None` when this thread is not recording, which is every thread that
        /// has not asked to be.
        static LOG: RefCell<Option<Vec<DiskOp>>> = const { RefCell::new(None) };
    }

    /// Begin recording on this thread, discarding any previous log.
    pub(crate) fn start() {
        LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
    }

    /// Stop recording and return what was recorded.
    pub(crate) fn take() -> Vec<DiskOp> {
        LOG.with(|l| l.borrow_mut().take()).unwrap_or_default()
    }

    /// Note one completed write, if this thread is recording.
    ///
    /// Takes the bytes by reference and copies them *inside* the recording
    /// check, which is the difference between the claim above and a lie: built
    /// as a `DiskOp` at the call site, every write in the whole `cfg(test)`
    /// build would pay a heap allocation and a copy to hand it to a thread that
    /// is not recording and will drop it.
    pub(crate) fn record_write(offset: u64, bytes: &[u8]) {
        LOG.with(|l| {
            if let Some(log) = l.borrow_mut().as_mut() {
                log.push(DiskOp::Write {
                    offset,
                    bytes: bytes.to_vec(),
                });
            }
        });
    }

    /// Note one completed resize, if this thread is recording.
    pub(crate) fn record_set_len(len: u64) {
        LOG.with(|l| {
            if let Some(log) = l.borrow_mut().as_mut() {
                log.push(DiskOp::SetLen(len));
            }
        });
    }
}

/// The file bytes a mutating session works on.
///
/// Implementors keep the image and the file on disk consistent, and are free to
/// choose the order in which they update them; see [`MirrorImage`] for why the
/// mirror writes to disk first.
pub(crate) trait FileImage: Source + Send + Sync {
    /// Append `bytes` at end-of-file, returning the absolute address they were
    /// written at — which is the pre-call [`Source::len`]. Extends `len` by
    /// exactly `bytes.len()`; see the module docs for why callers rely on that.
    ///
    /// An implementation that caches reads must invalidate any entry covering
    /// the appended range: a preceding [`truncate`](Self::truncate) can make an
    /// address readable, then cached, then appended over. An implementation that
    /// also refuses reads past end-of-file satisfies this through its `truncate`
    /// alone, since nothing above the original end could have been cached; the
    /// obligation is stated here because it belongs to the contract rather than
    /// to that policy.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error>;

    /// Overwrite `[offset, offset + bytes.len())` in place. The range must
    /// already exist; the engine computes it from its own allocation, so a
    /// range past end-of-file is a bug rather than a bad file, and
    /// implementations may assert it.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error>;

    /// Shrink the image to `len` bytes, physically shortening the file. `len`
    /// must not exceed the current [`Source::len`]: this cannot grow an image,
    /// and implementations may assert that rather than define a growth
    /// semantics no caller wants.
    ///
    /// The same cache-invalidation obligation as [`append`](Self::append)
    /// applies to the discarded range.
    fn truncate(&mut self, len: u64) -> Result<(), Error>;

    /// Flush buffered writes and force the file's *data* to durable storage.
    ///
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy) skips every in-session call to
    /// this and to [`sync_all`](Self::sync_all), so an implementation that buffers
    /// must not treat this as its only drain: a buffer emptied nowhere else would
    /// hold a *committed* edit in this process's memory under that policy.
    /// [`ordering_barrier`](Self::ordering_barrier) is the other drain, and the
    /// engine calls it at every point where the order of two writes matters —
    /// which every operation ends with. Which of the two a given
    /// [`WriteBuffering`] answers to is that enum's whole subject.
    fn sync_data(&mut self) -> Result<(), Error>;

    /// Flush buffered writes and force the file's data **and metadata** to
    /// durable storage. Distinct from [`sync_data`](Self::sync_data) because a
    /// commit changes the file's length, which lives in that metadata.
    ///
    /// The distinction is not observable in this crate's test suite, which
    /// simulates a crash by copying the file rather than by losing the page
    /// cache. Weakening a call site from `sync_all` to `sync_data` would pass
    /// every test and lose a committed file's length on power loss, so the
    /// choice at each call site has to be preserved by inspection.
    fn sync_all(&mut self) -> Result<(), Error>;

    /// An ordering point has been reached: every write made before it must reach
    /// the operating system before any write made after it.
    ///
    /// Named for the event rather than the effect, because the effect is the
    /// mode's to choose. [`WriteBuffering::Operation`] issues what it holds, which
    /// is what makes gathering free — the engine's barriers keep their ordering
    /// meaning under every [`SyncPolicy`](crate::SyncPolicy), and every operation
    /// ends with one, so a finished commit or append has reached the operating
    /// system either way. [`WriteBuffering::Session`] deliberately does nothing
    /// here; that is exactly the guarantee an explicit page buffer trades.
    ///
    /// It forces nothing to durable storage. That is [`sync_all`](Self::sync_all)'s
    /// job, and whether it happens is the policy's decision.
    fn ordering_barrier(&mut self) -> Result<(), Error>;

    /// Every issued write as `(offset, length)`, in the order it went out. What
    /// a count cannot say: that a publish point followed the bytes it names.
    #[cfg(test)]
    fn issued_write_order(&self) -> Vec<(u64, u64)>;

    /// How many writes this image has issued against the file since it was
    /// opened, for the tests that assert what an operation costs. Distinct from
    /// what the engine *called*: turning many of those into one is the point.
    #[cfg(test)]
    fn issued_writes(&self) -> u64 {
        self.issued_write_order().len() as u64
    }

    /// How many bytes those writes carried. Distinct from the count because the
    /// two answer different questions: joining runs lowers the count, and
    /// declining to write bytes that are about to be truncated away lowers only
    /// this.
    #[cfg(test)]
    fn issued_write_bytes(&self) -> u64 {
        self.issued_write_order().iter().map(|&(_, n)| n).sum()
    }

    /// Adopt `mode` for every write from here on, flushing anything already
    /// buffered that the new mode would not have held.
    ///
    /// Deliberately not defaulted: an image that silently ignored this would be a
    /// SWMR writer coalescing the ordered writes its readers depend on, and the
    /// only sound default — do nothing — is exactly that bug.
    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error>;

    /// The whole image as one slice, for parsers that walk bytes directly.
    ///
    /// `Some` only for a backing that already holds the file in memory. A
    /// caller must always have a [`Source`] path for the `None` case; this is a
    /// fast path, not a capability check.
    ///
    /// Buffering does not withdraw it: an image that lends its buffer out keeps
    /// that buffer current as it writes, and defers only the *disk* write.
    fn as_slice(&self) -> Option<&[u8]> {
        None
    }
}

/// An open read/write handle plus the writes not yet issued against it: the one
/// place either image touches the disk.
///
/// # What it gathers
///
/// A commit or an in-place append issues many small writes into a few pages. One
/// measured in-place append costs five write calls — the chunk (appended here,
/// though it may instead be written into freed space, which drops the next
/// call), the superblock's recorded end-of-file, and a patch each to the chunk index,
/// the array header and the object header's dataspace dimension — so four small
/// patches around one payload write, landing in pages the *next* append dirties
/// again. An append that has to allocate a new extensible-array data block costs
/// eight, which the first append into a dataset always does.
///
/// A barrier is where held bytes are released, and an append puts one between
/// each pair of phases that must reach the disk in order, so on the appends there
/// is little left to merge: measured on a paged fixture, four of those five
/// writes fall in two pages, two apiece, and all five still go out separately.
/// What the gathering collects is the commit tail, which puts many writes into a
/// handful of pages as it rebuilds a group, repoints a root and re-homes the
/// free-space managers: a commit staging eight dataset creations makes 24 write
/// calls and issues 4 (issue #288).
///
/// Pending writes are held as disjoint byte runs, merged on insert when they
/// touch or overlap, and emitted at flush as **one write per dirty page**: runs
/// sharing a page are joined, and the clean bytes between them are read back so
/// the join is a single write rather than a lie. That read is the deliberate
/// trade — it is a page-cache hit against a write this crate is trying not to
/// issue, and the flash it is issued to charges for writes.
///
/// # Why both images share it
///
/// The mirror already holds every byte of the file, so for that backing the runs
/// are a second copy of what it is about to write — bounded by the byte budget,
/// and it could instead have gathered dirty *page indices* and sliced its own
/// buffer at flush. That was weighed and declined: the saving is ~78 KB per
/// sixteen appends with no change in peak, and the cost would be two
/// implementations of write ordering, of which only one would be exercised by any
/// given test. Every rule in this module's tests is asserted `for backing in
/// BACKINGS`, from one body — including the crash-ordering rules — and this crate
/// has already paid for the alternative once, where two emit paths each needed
/// their own tests and a test through one was blind to the other.
///
/// # What it does not do
///
/// It does not evict. Exceeding the byte budget flushes everything rather than
/// choosing a victim page. Under [`WriteBuffering::Operation`] the budget is only
/// ever reached by one large operation, whose runs are long and contiguous and
/// gain nothing from being kept; under [`WriteBuffering::Session`] it is reached
/// by accumulation, and flushing whole then costs one extra pass over pages that
/// were about to be written anyway. Choosing a victim would buy the difference
/// between those two, which no measurement here has asked for.
pub(crate) struct BufferedWrites {
    handle: fs::File,
    mode: WriteBuffering,
    /// Pending writes keyed by start offset. Disjoint and non-touching: two runs
    /// that met would have been merged when the second was inserted, which is
    /// what lets a flush walk them in order and lets [`overlay`](Self::overlay)
    /// stop at the first run past its range.
    runs: BTreeMap<u64, Vec<u8>>,
    /// The total length of every pending run, maintained by hand at each site
    /// that inserts, extends, merges or drops one.
    ///
    /// Under-counting already fails loudly: `flush` subtracts a run's length
    /// from this, so too small a value underflows a `usize` and panics. Over-
    /// counting is silent, and what it costs is the budget in
    /// [`set_mode`](Self::set_mode) firing early — this gatherer quietly
    /// reverting toward one write per patch, which no byte comparison and no
    /// write-count ratio in the suite is tight enough to see. That asymmetry is
    /// why [`invariants_hold`](Self::invariants_hold) exists.
    pending_bytes: usize,
    /// The file's real length on disk, moved by every write this issues and by
    /// [`set_len`](Self::set_len), and by nothing else.
    ///
    /// A flush reads the clean bytes between two runs sharing a page, and that
    /// read must not fall past the end of the actual file — which trails the
    /// image's logical end-of-file whenever an append is still pending.
    on_disk_len: u64,
    /// Every write actually issued against the handle, as `(offset, length)` in
    /// the order it went out — the figure this whole type exists to lower, and
    /// the *order*, which is what says a publish point followed the bytes it
    /// names. Recording cannot change what is gathered, so it is carried only
    /// where it is read: the unit tests. `cfg(test)` is the lib's own test build,
    /// so the integration tests that measure allocation never compile this.
    #[cfg(test)]
    issued_order: Vec<(u64, u64)>,
}

impl BufferedWrites {
    /// Wrap `handle`, whose length on disk is `on_disk_len`. Buffers nothing
    /// until [`set_mode`](Self::set_mode) says otherwise.
    pub(crate) fn new(handle: fs::File, on_disk_len: u64) -> Self {
        Self {
            handle,
            mode: WriteBuffering::Unbuffered,
            runs: BTreeMap::new(),
            pending_bytes: 0,
            on_disk_len,
            #[cfg(test)]
            issued_order: Vec::new(),
        }
    }

    /// Every issued write as `(offset, length)`, in the order it went out.
    #[cfg(test)]
    pub(crate) fn issued_order(&self) -> &[(u64, u64)] {
        &self.issued_order
    }

    /// The handle, for the positioned reads an image serves from it.
    pub(crate) fn handle(&self) -> &fs::File {
        &self.handle
    }

    /// The file's current length on disk, which is short of the image's logical
    /// end-of-file by exactly the appends still pending.
    pub(crate) fn on_disk_len(&self) -> u64 {
        self.on_disk_len
    }

    /// Adopt `mode`, flushing first so nothing gathered under the old rules
    /// outlives them.
    pub(crate) fn set_mode(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.flush()?;
        self.mode = mode;
        Ok(())
    }

    /// Record (or issue) a write of `bytes` at `offset`.
    pub(crate) fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        // Above the mode check: an empty write is nothing to do under any of
        // them, and issuing one costs a `seek` and a `write_all` to change no
        // byte.
        if bytes.is_empty() {
            return Ok(());
        }
        let Some((_, max_bytes)) = self.mode.budget() else {
            return self.issue(offset, bytes);
        };
        // A write the budget cannot hold gains nothing from being held: it is
        // already at least a page, so it merges with nothing, and absorbing it
        // first would copy it into a run only to flush that run on the next line.
        // Measured on a 16 MiB staged commit, that copy was the whole dataset a
        // second time. Flush before issuing so this cannot overtake a pending
        // byte at a lower address.
        //
        // The threshold is the **budget**, not the page. `H5PB_write` bypasses at
        // `size >= page_size`, and the difference is what the two are for: the C
        // page buffer is a cache, and its rule keeps data that will not be re-read
        // from evicting data that will. This is a coalescer, and a write between
        // one page and the budget is precisely the kind that still merges with its
        // neighbours — bypassing it would forfeit the merge that earns the
        // reduction. A chunked dataset with 8 KiB chunks would go from adjacent
        // chunks joined into one issue to one issue per chunk.
        //
        // The rule would also buy nothing to offset that, because this engine's
        // writes are far smaller than a page. Measured on a page-buffered session,
        // incoming sizes: 32 chunk appends into eight datasets are 184 writes with
        // a median of 60 bytes and a maximum of 256, none of them a page; one
        // 4 MiB append is 32,912 writes with a median of 256 bytes, of which 34 —
        // 0.1% of the writes and 0.5% of the bytes — reach 4 KiB. What looks like
        // one long run at the dataset level arrives here as tens of thousands of
        // small writes, and merging them into that run is this type's whole job
        // (issue #357).
        if bytes.len() >= max_bytes {
            self.flush()?;
            return self.issue(offset, bytes);
        }
        self.absorb(offset, bytes);
        if self.pending_bytes > max_bytes {
            self.flush()?;
        }
        Ok(())
    }

    /// Every run is non-empty, the runs are disjoint *and* non-touching in
    /// address order, and [`pending_bytes`](Self::pending_bytes) is exactly their
    /// total length.
    ///
    /// Not gated on `cfg(debug_assertions)`, because `debug_assert!` type-checks
    /// its argument in every profile: a gated helper called from one compiles in
    /// debug and fails every release build, which is the trap the `Check
    /// (release)` CI job exists to catch.
    fn invariants_hold(&self) -> bool {
        let mut total = 0usize;
        let mut prev_end: Option<u64> = None;
        for (&k, v) in &self.runs {
            if v.is_empty() {
                return false;
            }
            // Touching is a violation, not just overlapping: two runs that meet
            // exactly should have been merged into one by `absorb`.
            if prev_end.is_some_and(|end| k <= end) {
                return false;
            }
            total += v.len();
            prev_end = Some(k + v.len() as u64);
        }
        total == self.pending_bytes
    }

    /// Merge `bytes` at `offset` into the pending runs, joining every run it
    /// touches or overlaps into one.
    ///
    /// The two fast paths below are not tuning. The general merge allocates a
    /// buffer the size of the whole joined span and copies the old runs into it,
    /// so without them the gathering is **quadratic in the writes it holds**,
    /// against a run grown to the byte budget.
    ///
    /// The extension path is the one an operation reaches by itself, and the
    /// expensive one to lose: measured on a four-megabyte append, whose batches
    /// each grow a run at end-of-file to the budget, disabling it costs 553 MB of
    /// copying against 22 MB with it.
    ///
    /// The containment path costs almost nothing in that same append, because the
    /// patches land in the header and index runs rather than in the long one at
    /// end-of-file. It is reached instead by a buffer held *across* operations —
    /// [`WriteBuffering::Session`] — where a patch lands inside a run already
    /// grown to the budget: measured over 256 appends into a one-megabyte page
    /// buffer, 407 MB of copying against 7.7 MB, to write one megabyte.
    ///
    /// One test per path, each through the one configuration that reaches it:
    /// `gathering_writes_does_not_recopy_what_it_holds` bounds the extension
    /// path, and `a_page_buffer_does_not_recopy_what_it_holds_across_operations`
    /// the containment path — the only test in the suite that fails when that
    /// one is disabled.
    fn absorb(&mut self, offset: u64, bytes: &[u8]) {
        let mut lo = offset;
        let mut hi = offset + bytes.len() as u64;
        // A write wholly inside a run already held — an index element patched
        // into a block this same operation appended, a superblock rewritten a
        // second time — is that run's own bytes changing.
        if let Some(k) = self.run_containing(offset, hi) {
            let run = self.runs.get_mut(&k).expect("just enumerated");
            #[expect(
                clippy::cast_possible_truncation,
                reason = "run_containing proved k <= offset and that the run reaches                           offset + bytes.len(), so this is an index into a buffer already                           resident, and a resident buffer's length is a usize"
            )]
            let at = (offset - k) as usize;
            run[at..at + bytes.len()].copy_from_slice(bytes);
            return;
        }
        // A write that continues the run before it and reaches no run after it is
        // what a sequence of appends at end-of-file is.
        if let Some(k) = self.run_ending_at(offset, hi) {
            let run = self.runs.get_mut(&k).expect("just enumerated");
            run.extend_from_slice(bytes);
            self.pending_bytes += bytes.len();
            debug_assert!(self.invariants_hold(), "absorb: extension path");
            return;
        }
        // The run that starts before this write and may reach it. Taking its end
        // into `hi` matters for a write that lands wholly inside a longer run.
        if let Some((&k, v)) = self.runs.range(..lo).next_back() {
            let end = k + v.len() as u64;
            if end >= lo {
                lo = k;
                hi = hi.max(end);
            }
        }
        // Runs starting at or after `lo`, in order, while each still touches what
        // has been gathered. They are disjoint, so one pass settles `hi`.
        let mut absorbed: Vec<u64> = Vec::new();
        for (&k, v) in self.runs.range(lo..) {
            if k > hi {
                break;
            }
            hi = hi.max(k + v.len() as u64);
            absorbed.push(k);
        }
        debug_assert!(
            lo <= offset && hi >= offset + bytes.len() as u64,
            "the merged span must contain the write that caused it"
        );
        // `[lo, hi)` is the union of the new write and the runs it touches, all of
        // which are already resident, so its length is the sum of some `usize`
        // lengths and cannot fail to be one.
        let span = (hi - lo)
            .to_usize()
            .expect("a merged run is the union of buffers already in memory");
        let mut merged = vec![0u8; span];
        for k in absorbed {
            let old = self.runs.remove(&k).expect("just enumerated");
            self.pending_bytes -= old.len();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "every absorbed run starts within [lo, hi), which `merged` spans,                           so this indexes `merged`"
            )]
            let at = (k - lo) as usize;
            merged[at..at + old.len()].copy_from_slice(&old);
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "lo <= offset, asserted above, and `merged` spans [lo, hi) which                       contains the write"
        )]
        let at = (offset - lo) as usize;
        merged[at..at + bytes.len()].copy_from_slice(bytes);
        self.pending_bytes += merged.len();
        self.runs.insert(lo, merged);
        debug_assert!(self.invariants_hold(), "absorb: general merge");
    }

    /// The start offset of the pending run that wholly contains `[offset, end)`,
    /// so the write can be patched into it in place.
    fn run_containing(&self, offset: u64, end: u64) -> Option<u64> {
        let (&k, v) = self.runs.range(..=offset).next_back()?;
        (k + v.len() as u64 >= end).then_some(k)
    }

    /// The start offset of the pending run that ends exactly at `offset` when
    /// `[offset, end)` touches nothing after it, so the write can be appended to
    /// that run in place rather than merged into a fresh one.
    fn run_ending_at(&self, offset: u64, end: u64) -> Option<u64> {
        let (&k, v) = self.runs.range(..offset).next_back()?;
        (k + v.len() as u64 == offset && self.runs.range(offset..=end).next().is_none())
            .then_some(k)
    }

    /// Where a walk that must see every run reaching `offset` has to start: the
    /// run before it can still reach into the window, and a walk from `offset`
    /// would step over it.
    fn walk_from(&self, offset: u64) -> u64 {
        self.runs
            .range(..=offset)
            .next_back()
            .map_or(offset, |(&k, _)| k)
    }

    /// Whether the pending runs wholly cover `[offset, end)`.
    ///
    /// Only a debug assertion asks this. It is the one way a buffered read can
    /// go wrong *quietly*: an address past the file's real length reads as zeros
    /// and is then patched by the overlay, so a range the overlay does not reach
    /// returns zeros rather than an error, and zeros parse.
    ///
    /// Not `cfg(debug_assertions)`: `debug_assert!` type-checks its argument in
    /// every profile and only skips *running* it, so gating this would compile in
    /// a debug build and fail every release one.
    fn covers(&self, offset: u64, end: u64) -> bool {
        let mut at = offset;
        for (&k, v) in self.runs.range(self.walk_from(offset)..) {
            if at >= end {
                return true;
            }
            if k > at {
                return false;
            }
            at = at.max(k + v.len() as u64);
        }
        at >= end
    }

    /// Patch pending bytes over `buf`, which the caller filled from `offset` on
    /// disk. An image whose reads go to the disk must call this or read stale
    /// bytes; one that reads from its own current mirror must not need to.
    pub(crate) fn overlay(&self, offset: u64, buf: &mut [u8]) {
        if self.runs.is_empty() || buf.is_empty() {
            return;
        }
        let end = offset + buf.len() as u64;
        for (&k, v) in self.runs.range(self.walk_from(offset)..) {
            if k >= end {
                break;
            }
            let run_end = k + v.len() as u64;
            if run_end <= offset {
                continue;
            }
            let from = k.max(offset);
            let to = run_end.min(end);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "from and to are clamped to both the read window and the run, so                           each delta is at most buf.len() or v.len() — lengths of buffers                           already in memory"
            )]
            let (buf_from, buf_to, run_from, run_to) = (
                (from - offset) as usize,
                (to - offset) as usize,
                (from - k) as usize,
                (to - k) as usize,
            );
            buf[buf_from..buf_to].copy_from_slice(&v[run_from..run_to]);
        }
    }

    /// Issue what is gathered, so everything written before this reaches the
    /// operating system before anything written after it.
    ///
    /// Dispatched rather than flushed unconditionally — which would be
    /// equivalent today, since an unbuffered image never holds runs — so that a
    /// mode which does *not* release here has to say so instead of inheriting it.
    pub(crate) fn ordering_barrier(&mut self) -> Result<(), Error> {
        match self.mode {
            WriteBuffering::Unbuffered | WriteBuffering::Session { .. } => Ok(()),
            WriteBuffering::Operation { .. } => self.flush(),
        }
    }

    /// Issue every pending run, joining those that share a page into one write.
    ///
    /// A run is removed from the map only once it has been issued, and a run that
    /// cannot be is put back. Taking the whole map up front and returning on the
    /// first error would drop everything not yet written *and* leave
    /// `pending_bytes` at zero, so the next flush — including the one
    /// [`File::close`](crate::File::close) makes — would find an empty buffer and
    /// report success over a batch it had silently lost. An error here means the
    /// writes are still pending and the caller may retry or report; it never
    /// means they are gone.
    pub(crate) fn flush(&mut self) -> Result<(), Error> {
        // A flush is where the page size is spent; an unbuffered image never
        // reaches here with runs, and a mode change flushes under the old size.
        let page_size = self.mode.budget().map_or(1, |(p, _)| p).max(1);
        while let Some((start, mut bytes)) = self.runs.pop_first() {
            self.pending_bytes -= bytes.len();
            // Join every following run sharing a page with this one's last byte,
            // reading back the clean bytes between them so the join is one write
            // rather than a lie.
            while let Some((&next, _)) = self.runs.first_key_value() {
                if next > self.on_disk_len
                    || !same_page(start + bytes.len() as u64 - 1, next, page_size)
                {
                    break;
                }
                let gap_at = start + bytes.len() as u64;
                if gap_at < next {
                    let filled = bytes.len();
                    // The one length here not bounded by a buffer already in
                    // memory: it is the hole between two runs sharing a page, so
                    // it is bounded by the page size, which is the file's to
                    // choose. Checked rather than asserted for that reason.
                    let gap = match (next - gap_at).to_usize() {
                        Ok(gap) => gap,
                        Err(e) => {
                            self.restore(start, bytes);
                            return Err(Error::Format(e));
                        }
                    };
                    bytes.resize(filled + gap, 0);
                    if let Err(e) =
                        read_at_handle(&self.handle, self.on_disk_len, gap_at, &mut bytes[filled..])
                    {
                        // Give back the run as it stood before the gap read, so it
                        // still ends short of `next` and the runs stay disjoint.
                        bytes.truncate(filled);
                        self.restore(start, bytes);
                        return Err(Error::Format(e));
                    }
                }
                let (_, tail) = self.runs.pop_first().expect("just peeked");
                self.pending_bytes -= tail.len();
                bytes.extend_from_slice(&tail);
            }
            if let Err(e) = self.issue(start, &bytes) {
                self.restore(start, bytes);
                return Err(e);
            }
        }
        Ok(())
    }

    /// Put a run that could not be issued back in the map, pending again.
    fn restore(&mut self, start: u64, bytes: Vec<u8>) {
        self.pending_bytes += bytes.len();
        self.runs.insert(start, bytes);
    }

    /// Write `bytes` at `offset` through to the operating system now.
    fn issue(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.handle
            .seek(SeekFrom::Start(offset))
            .map_err(Error::Io)?;
        self.handle.write_all(bytes).map_err(Error::Io)?;
        #[cfg(test)]
        self.issued_order.push((offset, bytes.len() as u64));
        #[cfg(test)]
        disk_log::record_write(offset, bytes);
        self.on_disk_len = self.on_disk_len.max(offset + bytes.len() as u64);
        Ok(())
    }

    /// Physically resize the file to `len`, dropping the pending writes the new
    /// length puts past end-of-file and issuing the rest first, so a run below the
    /// cut still lands.
    pub(crate) fn set_len(&mut self, len: u64) -> Result<(), Error> {
        // Two rules meet here, and the order is the only one that keeps both.
        // Nothing may write bytes the truncate is about to remove, so the doomed
        // runs must go before the flush. And `discard_from` permanently forgets
        // them, so it must not run ahead of a step that can fail — discarding
        // first leaves a failed truncate reporting a length whose bytes nothing
        // will ever write, against `flush`'s own contract that an error leaves
        // the writes pending rather than gone. The truncate therefore goes first:
        // it is the step that dooms them, and until it succeeds they are alive.
        //
        // What that leaves is a failed *flush* after a successful truncate, which
        // keeps every surviving run pending but has already moved the file. That
        // is an error path in both orders; this one is the half that cannot lose
        // a write.
        self.handle.set_len(len).map_err(Error::Io)?;
        #[cfg(test)]
        disk_log::record_set_len(len);
        self.discard_from(len);
        self.on_disk_len = len;
        self.flush()?;
        Ok(())
    }

    /// Forget every pending byte at or past `len`, trimming a run that straddles
    /// it. Those bytes are about to stop existing.
    fn discard_from(&mut self, len: u64) {
        let doomed: Vec<u64> = self.runs.range(len..).map(|(&k, _)| k).collect();
        for k in doomed {
            let v = self.runs.remove(&k).expect("just enumerated");
            self.pending_bytes -= v.len();
        }
        if let Some((&k, _)) = self.runs.range(..len).next_back() {
            let v = self.runs.get_mut(&k).expect("just enumerated");
            // Saturating rather than `as`: a gap wider than this platform's
            // `usize` cannot be shorter than the run, so saturation keeps the run
            // whole, which is the answer. Truncating would trim — or at an exact
            // multiple of the word size empty — a run that must be kept.
            let keep = (len - k).to_usize().unwrap_or(usize::MAX);
            if keep < v.len() {
                self.pending_bytes -= v.len() - keep;
                v.truncate(keep);
            }
        }
        debug_assert!(self.invariants_hold(), "discard_from");
    }

    /// Flush, then force the file's data to durable storage.
    pub(crate) fn sync_data(&mut self) -> Result<(), Error> {
        self.flush()?;
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_data().map_err(Error::Io)?;
        Ok(())
    }

    /// Flush, then force the file's data and metadata to durable storage.
    pub(crate) fn sync_all(&mut self) -> Result<(), Error> {
        self.flush()?;
        self.handle.flush().map_err(Error::Io)?;
        self.handle.sync_all().map_err(Error::Io)?;
        Ok(())
    }
}

impl Drop for BufferedWrites {
    /// Last resort for a session dropped without a teardown — a bare engine in a
    /// test, or an unwind. The engine's own `close`/`drop` path syncs, which
    /// flushes; this is what keeps a path that does neither from silently
    /// discarding a write it reported as done.
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Whether two file offsets fall in the same `page_size`-aligned page.
const fn same_page(a: u64, b: u64, page_size: u64) -> bool {
    a / page_size == b / page_size
}

/// A whole-file in-memory mirror plus the read/write handle it mirrors, kept
/// byte-for-byte in sync. This is the backing
/// [`File::open_rw`](crate::File::open_rw) falls back to for a file the bounded
/// engine cannot edit, and the one
/// [`MemoryStrategy::Mirrored`](crate::MemoryStrategy) always takes: reads are
/// slice accesses and never touch the disk, at the cost of holding the entire
/// file resident.
///
/// Every mutation reaches the disk *before* updating the mirror, so a failed
/// write can leave the mirror behind the file but never ahead of it. That
/// direction is the safe one: the session re-reads its own mirror to plan
/// later edits, and planning against bytes that are not yet on disk would
/// commit a structure pointing at content that does not exist.
///
/// "Reaches the disk" is [`BufferedWrites`]'s business, not this type's, so
/// under a buffering mode the mirror does run ahead of the *file* between
/// flushes. The ordering above is preserved where it matters — the mirror is
/// updated after the write is accepted, so a rejected write never enters it —
/// and the reads that plan the next edit come from the mirror, which is current
/// by construction.
pub(crate) struct MirrorImage {
    writes: BufferedWrites,
    data: Vec<u8>,
}

impl MirrorImage {
    /// Wrap an open read/write `handle` and the `data` already read from it.
    /// The caller is responsible for the two agreeing.
    pub(crate) fn new(handle: fs::File, data: Vec<u8>) -> Self {
        let on_disk_len = data.len() as u64;
        Self {
            writes: BufferedWrites::new(handle, on_disk_len),
            data,
        }
    }
}

impl Source for MirrorImage {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        BytesSource::new(&self.data[..]).read_at(offset, buf)
    }
}

impl FileImage for MirrorImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.data.len() as u64;
        self.writes.write_at(addr, bytes)?;
        self.data.extend_from_slice(bytes);
        Ok(addr)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        debug_assert!(
            offset.saturating_add(bytes.len() as u64) <= self.len(),
            "write_at past end-of-file: {offset}+{} > {}",
            bytes.len(),
            self.len()
        );
        // Convert before touching the file so a failure leaves both sides
        // untouched rather than the mirror behind the disk.
        let offset_usize = offset.to_usize()?;
        self.writes.write_at(offset, bytes)?;
        self.data[offset_usize..offset_usize + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        debug_assert!(
            len <= self.len(),
            "truncate would grow the image: {len} > {}",
            self.len()
        );
        // `set_len` grows a file where `Vec::truncate` no-ops, and a failed
        // conversion after `set_len` would leave the mirror longer than the
        // file. Convert first so neither side moves unless both can.
        let len_usize = len.to_usize()?;
        self.writes.set_len(len)?;
        self.data.truncate(len_usize);
        Ok(())
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.writes.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.writes.sync_all()
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.writes.ordering_barrier()
    }

    #[cfg(test)]
    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.writes.issued_order().to_vec()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.writes.set_mode(mode)
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(&self.data)
    }
}

/// Read exactly `buf.len()` bytes at `offset` from a shared file handle,
/// bounds-checked against `len` (mirroring `ReadSeekSource`). Uses the
/// `Read`/`Seek` impls on `&fs::File`, so it can serve a `&self` read: callers
/// serialize access through the session's engine lock, and the shared cursor is
/// never raced.
pub(crate) fn read_at_handle(
    handle: &fs::File,
    len: u64,
    offset: u64,
    buf: &mut [u8],
) -> Result<(), FormatError> {
    let end = offset
        .checked_add(buf.len() as u64)
        .ok_or(FormatError::OffsetOverflow {
            offset,
            length: buf.len() as u64,
        })?;
    if end > len {
        return Err(FormatError::UnexpectedEof {
            expected: end.to_usize().unwrap_or(usize::MAX),
            available: len.to_usize().unwrap_or(usize::MAX),
        });
    }
    let mut h = handle;
    h.seek(SeekFrom::Start(offset))
        .map_err(|e| FormatError::Source(std::format!("{e}")))?;
    h.read_exact(buf)
        .map_err(|e| FormatError::Source(std::format!("{e}")))?;
    Ok(())
}

/// A [`Source`] over a *borrowed* open handle, for the reads an open has to make
/// before it decides which image will own that handle.
///
/// It exists so a read-write open can locate and validate the superblock — a few
/// bounded windows — before building an image that might read the whole file.
/// Refusing after that build costs `O(file size)` on a file that is then
/// rejected. It moves the handle's shared cursor, as everything using
/// [`read_at_handle`] does, so a caller that later reads sequentially from the
/// same handle must position it itself ([`MirrorImage`] does).
pub(crate) struct BorrowedHandle<'a> {
    handle: &'a fs::File,
    len: u64,
}

impl<'a> BorrowedHandle<'a> {
    pub(crate) fn new(handle: &'a fs::File, len: u64) -> Self {
        Self { handle, len }
    }
}

impl Source for BorrowedHandle<'_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        read_at_handle(self.handle, self.len, offset, buf)
    }
}

/// A file-backed image that holds no whole-file mirror: reads are positioned I/O
/// against the handle, served through a bounded metadata cache when one is
/// configured. This is the backing a bounded read-write open uses,
/// and the reason [`FileImage`] exists — resident memory is the cache budget
/// plus whatever the caller is parsing, independent of the file's size.
///
/// The end-of-file cursor is explicit ([`len`](Source::len)), seeded from the
/// file's real length at open and advanced by [`append`](FileImage::append). It
/// is not re-read from the filesystem, so it stays the authority even if the
/// handle's own cursor moves.
///
/// Every mutation invalidates the cache entries it overlaps *before* the write
/// reaches the disk, so a concurrent-looking read can miss a fresh byte but
/// never return a stale one.
///
/// Holding no mirror is also what makes this the image that has to *overlay* its
/// pending writes onto every read: it has no second copy of the bytes to keep
/// current, so a buffered write is visible only through [`BufferedWrites`] until
/// it lands.
pub(crate) struct HandleImage {
    writes: BufferedWrites,
    /// Logical end-of-file: the real file length at open, moved by `append` and
    /// `truncate` and by nothing else.
    len: u64,
    /// Bounded read cache for metadata-sized reads; `None` when disabled.
    metadata_cache: Option<(MetadataCacheConfig, std::sync::Mutex<MetadataReadCache>)>,
}

impl HandleImage {
    /// Wrap an open read/write `handle` whose current length is `len`, caching
    /// metadata reads under `cache` (see [`MetadataCacheConfig::disabled`] to
    /// opt out).
    pub(crate) fn new(handle: fs::File, len: u64, cache: MetadataCacheConfig) -> Self {
        Self {
            writes: BufferedWrites::new(handle, len),
            len,
            metadata_cache: cache
                .is_enabled()
                .then(|| (cache, std::sync::Mutex::new(MetadataReadCache::new()))),
        }
    }

    /// Drop every cached read overlapping `[offset, offset + len)`. Called before
    /// each mutation; a no-op when caching is off or the range is empty.
    fn invalidate(&self, offset: u64, len: u64) {
        let Some((_, cache)) = &self.metadata_cache else {
            return;
        };
        MetadataReadCache::locked(cache)
            .invalidate_overlapping(offset, len.to_usize().unwrap_or(usize::MAX));
    }
}

impl Source for HandleImage {
    fn len(&self) -> u64 {
        self.len
    }

    /// Bytes from the file, with every pending write patched over them.
    ///
    /// The disk read is clamped to the file's *real* length rather than the
    /// image's: a pending append leaves addresses that are readable by this
    /// image's contract but do not exist yet on disk, and `read_exact` past the
    /// end of a file is an error, not a short read. Those addresses are covered
    /// by the pending writes that created them, which the overlay then supplies.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > self.len {
            return Err(FormatError::UnexpectedEof {
                expected: end.to_usize().unwrap_or(usize::MAX),
                available: self.len.to_usize().unwrap_or(usize::MAX),
            });
        }
        let on_disk = self.writes.on_disk_len();
        // Where the disk runs out inside this window: everything below comes off
        // the file, everything above it exists only as a pending write.
        let disk_end = on_disk.clamp(offset, end);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "disk_end is clamped to [offset, offset + buf.len()), so this is at \
                      most buf.len()"
        )]
        let take = (disk_end - offset) as usize;
        if take > 0 {
            read_at_handle(self.writes.handle(), on_disk, offset, &mut buf[..take])?;
        }
        buf[take..].fill(0);
        debug_assert!(
            disk_end >= end || self.writes.covers(disk_end, end),
            "read of [{offset}, {end}) reaches past the file's {on_disk} bytes into a \
             range no pending write covers, so it would return zeros"
        );
        self.writes.overlay(offset, buf);
        Ok(())
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        let Some((config, cache)) = &self.metadata_cache else {
            return self.read_exact_at(offset, len);
        };
        MetadataReadCache::read_through(cache, *config, offset, len, || {
            self.read_exact_at(offset, len)
        })
    }

    fn metadata_cache_stats(&self) -> Option<MetadataCacheStats> {
        let (_, cache) = self.metadata_cache.as_ref()?;
        Some(MetadataReadCache::locked(cache).stats())
    }

    fn reset_metadata_cache_stats(&self) {
        if let Some((_, cache)) = &self.metadata_cache {
            MetadataReadCache::locked(cache).reset_stats();
        }
    }
}

impl FileImage for HandleImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let addr = self.len;
        // The trait requires this. It is belt-and-braces for *this* image, whose
        // reads past end-of-file are refused: the only way a cached entry can
        // cover an address an append reuses is a preceding `truncate`, and that
        // already invalidated everything it discarded. It is kept because the
        // obligation belongs to the contract, not to this implementation's
        // read-bounds policy.
        self.invalidate(addr, bytes.len() as u64);
        self.writes.write_at(addr, bytes)?;
        self.len += bytes.len() as u64;
        Ok(addr)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        // The mirror image fails loudly on this in every build — it indexes its
        // buffer — so this one must too rather than extend the file behind the
        // `len` cursor and leave a later `append` overwriting what was written.
        // A caller bug that is a panic on one backing and silent corruption on the
        // other is exactly what the two being interchangeable has to rule out.
        let end = offset
            .checked_add(bytes.len() as u64)
            .filter(|&e| e <= self.len)
            .ok_or(Error::Format(FormatError::UnexpectedEof {
                expected: offset.to_usize().unwrap_or(usize::MAX),
                available: self.len.to_usize().unwrap_or(usize::MAX),
            }))?;
        debug_assert!(end <= self.len);
        self.invalidate(offset, bytes.len() as u64);
        self.writes.write_at(offset, bytes)
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        debug_assert!(
            len <= self.len,
            "truncate would grow the image: {len} > {}",
            self.len
        );
        self.invalidate(len, self.len.saturating_sub(len));
        self.writes.set_len(len)?;
        self.len = len;
        Ok(())
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.writes.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.writes.sync_all()
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.writes.ordering_barrier()
    }

    #[cfg(test)]
    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.writes.issued_order().to_vec()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.writes.set_mode(mode)
    }

    // No `as_slice`: withholding the whole-file slice is what this image is for.
}

/// A [`FileImage`] that counts what passes through it — bytes read, and
/// durability barriers issued — so a test can measure how much of a file an
/// operation touches and how many `fsync`s it costs. A caller that cares about
/// only one counter passes a throwaway `Arc` for the other.
///
/// It deliberately does *not* forward `read_metadata_at`, taking [`Source`]'s
/// default instead, so every read funnels through this one `read_at` and is
/// counted. Callers build the inner image with the metadata cache disabled, so
/// nothing is bypassed and the count is exact rather than an upper bound.
///
/// Both barriers feed one counter. What a [`SyncPolicy`](crate::SyncPolicy)
/// governs is whether a barrier is issued at all, and a test that also pinned
/// *which* of the two each site chose would fail on any later re-weighing of
/// that choice — which the trait above documents as an inspection-time decision,
/// not a tested one.
#[cfg(test)]
pub(crate) struct CountingImage {
    inner: Box<dyn FileImage>,
    read_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
    syncs: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

#[cfg(test)]
impl CountingImage {
    pub(crate) fn new(
        inner: Box<dyn FileImage>,
        read_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
        syncs: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Self {
        Self {
            inner,
            read_bytes,
            syncs,
        }
    }

    fn count_sync(&self) {
        self.syncs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
impl Source for CountingImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.read_bytes
            .fetch_add(buf.len() as u64, std::sync::atomic::Ordering::Relaxed);
        self.inner.read_at(offset, buf)
    }
}

#[cfg(test)]
impl FileImage for CountingImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.inner.write_at(offset, bytes)
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        self.inner.truncate(len)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.count_sync();
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.count_sync();
        self.inner.sync_all()
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.inner.ordering_barrier()
    }

    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.inner.issued_write_order()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(mode)
    }

    fn as_slice(&self) -> Option<&[u8]> {
        self.inner.as_slice()
    }
}

/// A [`MirrorImage`] whose writes into one byte range fail the way a dying
/// device does: the first such write **alters the file and then reports an
/// error**, and every one after it is refused outright, having changed nothing.
///
/// Both halves are needed to reach the single state that a refused commit
/// cannot repair (issue #344). The first makes the commit's in-place overwrite
/// land while failing the commit; the second makes the undo that would put the
/// prior bytes back fail in turn, so the file keeps a value from a batch that
/// was refused. A fake that merely refused the write would leave the file intact
/// and prove nothing about that state, and one that applied the write without
/// reporting an error would not fail the commit at all.
///
/// Writes outside the range are ordinary writes, so a commit can be given one
/// overwrite that survives its rollback and one that does not.
#[cfg(test)]
pub(crate) struct TornWriteImage {
    inner: MirrorImage,
    fails: core::ops::Range<u64>,
    struck: bool,
}

#[cfg(test)]
impl TornWriteImage {
    pub(crate) fn new(inner: MirrorImage, fails: core::ops::Range<u64>) -> Self {
        Self {
            inner,
            fails,
            struck: false,
        }
    }

    /// Whether `[offset, offset + len)` meets the failing range.
    fn hits(&self, offset: u64, len: usize) -> bool {
        offset < self.fails.end && self.fails.start < offset.saturating_add(len as u64)
    }

    fn failure() -> Error {
        Error::Io(std::io::Error::other("simulated device write failure"))
    }
}

#[cfg(test)]
impl Source for TornWriteImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, buf)
    }
}

#[cfg(test)]
impl FileImage for TornWriteImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        if !self.hits(offset, bytes.len()) {
            return self.inner.write_at(offset, bytes);
        }
        if self.struck {
            return Err(Self::failure());
        }
        self.struck = true;
        self.inner.write_at(offset, bytes)?;
        Err(Self::failure())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        self.inner.truncate(len)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.inner.sync_all()
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.inner.ordering_barrier()
    }

    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.inner.issued_write_order()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(mode)
    }

    fn as_slice(&self) -> Option<&[u8]> {
        self.inner.as_slice()
    }
}

/// A [`MirrorImage`] that withholds its slice, so every read routed through it
/// takes the [`Source`] path instead of the whole-file fast path.
///
/// This exists to make the mirrorless read paths reachable before a mirrorless
/// backing does. Each read the engine serves has two forms — one walking a
/// borrowed slice, one going through `Source` — and until the bounded read-write open
/// runs on this engine (issue #198), only the first would ever execute. Opening
/// a file through this image runs the same tests down the other form and lets
/// them be compared, which is the only thing that keeps the two from drifting.
#[cfg(test)]
pub(crate) struct SourceOnlyImage(MirrorImage);

#[cfg(test)]
impl SourceOnlyImage {
    pub(crate) fn new(inner: MirrorImage) -> Self {
        Self(inner)
    }
}

#[cfg(test)]
impl Source for SourceOnlyImage {
    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.0.read_at(offset, buf)
    }
}

#[cfg(test)]
impl FileImage for SourceOnlyImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.0.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.0.write_at(offset, bytes)
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        self.0.truncate(len)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.0.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.0.sync_all()
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.0.ordering_barrier()
    }

    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.0.issued_write_order()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.0.set_write_buffering(mode)
    }

    // Deliberately inherits the `None` default: that is the whole point.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two backings [`FileImage`] exists to make interchangeable. Every case
    /// below runs against both, because a primitive that holds for one and not
    /// the other is exactly the divergence this seam would otherwise hide.
    #[derive(Clone, Copy, Debug)]
    enum Backing {
        Mirror,
        Handle,
    }

    const BACKINGS: [Backing; 2] = [Backing::Mirror, Backing::Handle];

    /// Open a fresh file holding `initial`, wrapped in the given backing. The
    /// handle image caches metadata reads, so the cache is exercised rather
    /// than configured away.
    fn image(
        dir: &std::path::Path,
        initial: &[u8],
        backing: Backing,
    ) -> (std::path::PathBuf, Box<dyn FileImage>) {
        let path = dir.join(std::format!("{backing:?}.bin"));
        std::fs::write(&path, initial).unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let img: Box<dyn FileImage> = match backing {
            Backing::Mirror => Box::new(MirrorImage::new(handle, initial.to_vec())),
            Backing::Handle => Box::new(HandleImage::new(
                handle,
                initial.len() as u64,
                MetadataCacheConfig::new(64 * 1024),
            )),
        };
        (path, img)
    }

    /// The whole image read back through [`Source`], which is the interface
    /// every parser above this layer uses.
    fn bytes(img: &dyn FileImage) -> Vec<u8> {
        let mut buf = vec![0u8; img.len().to_usize().unwrap()];
        img.read_at(0, &mut buf).unwrap();
        buf
    }

    /// The invariant the layer exists to hold: after any primitive, what the
    /// image reports and what is on disk agree. A primitive that updates only
    /// one side is the defect that would otherwise surface as a corrupt file
    /// much later.
    fn assert_in_sync(path: &std::path::Path, img: &dyn FileImage, backing: Backing) {
        let on_disk = std::fs::read(path).unwrap();
        assert_eq!(
            img.len(),
            on_disk.len() as u64,
            "{backing:?}: end-of-file disagrees with the file"
        );
        assert_eq!(
            bytes(img),
            on_disk,
            "{backing:?}: reads disagree with the file"
        );
        if let Some(slice) = img.as_slice() {
            assert_eq!(
                slice,
                &on_disk[..],
                "{backing:?}: the slice disagrees with the file"
            );
        }
    }

    #[test]
    fn append_returns_the_pre_append_end_and_extends_by_exactly_the_length() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcd", backing);

            let addr = img.append(b"XYZ").unwrap();

            assert_eq!(addr, 4, "{backing:?}: append must report where it wrote");
            assert_eq!(
                img.len(),
                7,
                "{backing:?}: append must extend len by exactly bytes.len()"
            );
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    #[test]
    fn write_at_overwrites_both_sides_in_place() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcdef", backing);

            img.write_at(2, b"ZZ").unwrap();

            assert_eq!(bytes(img.as_ref()), b"abZZef");
            assert_eq!(
                img.len(),
                6,
                "{backing:?}: an in-place write must not move end-of-file"
            );
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    #[test]
    fn truncate_shrinks_both_sides() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcdef", backing);

            img.truncate(2).unwrap();

            assert_eq!(bytes(img.as_ref()), b"ab");
            assert_eq!(img.len(), 2, "{backing:?}");
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// A write following a truncate must land at the shortened end-of-file, not
    /// wherever the previous operation left the handle's cursor.
    #[test]
    fn append_after_truncate_lands_at_the_new_end() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = image(dir.path(), b"abcdef", backing);

            img.truncate(3).unwrap();
            let addr = img.append(b"Z").unwrap();

            assert_eq!(addr, 3, "{backing:?}");
            assert_eq!(bytes(img.as_ref()), b"abcZ");
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// Reads go through `Source`, so they must observe writes immediately —
    /// the engine plans its next edit against bytes it just wrote.
    #[test]
    fn reads_observe_writes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, mut img) = image(dir.path(), b"abcdef", backing);
            img.write_at(0, b"ZY").unwrap();
            img.append(b"!").unwrap();

            let mut buf = [0u8; 3];
            img.read_at(0, &mut buf).unwrap();
            assert_eq!(&buf, b"ZYc", "{backing:?}");
            img.read_at(6, &mut buf[..1]).unwrap();
            assert_eq!(buf[0], b'!', "{backing:?}");
        }
    }

    /// The same, through the *cached* read path: a metadata read taken before a
    /// write must not survive it. This is the handle image's own hazard — the
    /// mirror has no cache to go stale — but it runs on both so the case is
    /// stated once.
    #[test]
    fn cached_reads_observe_writes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, mut img) = image(dir.path(), b"abcdef", backing);

            assert_eq!(img.read_metadata_at(0, 4).unwrap(), b"abcd", "{backing:?}");
            img.write_at(1, b"ZZ").unwrap();

            assert_eq!(
                img.read_metadata_at(0, 4).unwrap(),
                b"aZZd",
                "{backing:?}: a cached read outlived the write that overwrote it"
            );
        }
    }

    /// Truncate makes a range unreadable, a later append reuses those addresses,
    /// and a read cached before the truncate must not resurface.
    ///
    /// What this pins is `truncate`'s invalidation, not `append`'s: for an image
    /// that refuses reads past end-of-file the two overlap, and deleting the call
    /// in `append` leaves this green. See the trait's `append` contract for why
    /// the call stays anyway.
    #[test]
    fn a_cached_read_does_not_survive_being_truncated_and_appended_over() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, mut img) = image(dir.path(), b"abcdef", backing);

            assert_eq!(img.read_metadata_at(4, 2).unwrap(), b"ef", "{backing:?}");
            img.truncate(4).unwrap();
            img.append(b"ZZ").unwrap();

            assert_eq!(
                img.read_metadata_at(4, 2).unwrap(),
                b"ZZ",
                "{backing:?}: a read cached before the truncate survived the append"
            );
        }
    }

    #[test]
    fn reads_past_end_of_file_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, img) = image(dir.path(), b"abcd", backing);

            let mut buf = [0u8; 2];
            assert!(img.read_at(3, &mut buf).is_err(), "{backing:?}");
        }
    }

    /// A 64-byte page, so a test can put two writes in one page or in two
    /// without writing kilobytes to say so.
    const PAGE: u64 = 64;

    /// Gather writes for the duration of one operation, at [`PAGE`].
    const GATHERED: WriteBuffering = WriteBuffering::Operation {
        page_size: PAGE,
        max_bytes: 4096,
    };

    /// An image over `initial`, gathering its writes under `mode`.
    fn gathering(
        dir: &std::path::Path,
        initial: &[u8],
        backing: Backing,
        mode: WriteBuffering,
    ) -> (std::path::PathBuf, Box<dyn FileImage>) {
        gathering_named(dir, "g", initial, backing, mode)
    }

    /// The same, under a caller-chosen name. A test that holds two images at once
    /// needs it: [`image`] names its file after the backing alone, so two of them
    /// would share a path — and the first, still alive and still flushing on drop,
    /// would be writing into a file the second had truncated.
    fn gathering_named(
        dir: &std::path::Path,
        name: &str,
        initial: &[u8],
        backing: Backing,
        mode: WriteBuffering,
    ) -> (std::path::PathBuf, Box<dyn FileImage>) {
        let path = dir.join(std::format!("{name}_{backing:?}.bin"));
        std::fs::write(&path, initial).unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut img: Box<dyn FileImage> = match backing {
            Backing::Mirror => Box::new(MirrorImage::new(handle, initial.to_vec())),
            Backing::Handle => Box::new(HandleImage::new(
                handle,
                initial.len() as u64,
                MetadataCacheConfig::new(64 * 1024),
            )),
        };
        img.set_write_buffering(mode).unwrap();
        (path, img)
    }

    /// The engine plans its next edit against bytes it just wrote, so a gathered
    /// write has to read back through the image even while the file still holds
    /// the old ones. That divide is the whole hazard the gathering introduces,
    /// and it runs on both backings because they answer it differently: the
    /// mirror is already current, the handle image overlays.
    #[test]
    fn a_gathered_write_reads_back_before_it_reaches_the_file() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = gathering(dir.path(), b"abcdef", backing, GATHERED);

            img.write_at(1, b"ZZ").unwrap();
            img.append(b"gh").unwrap();

            assert_eq!(bytes(img.as_ref()), b"aZZdefgh", "{backing:?}");
            assert_eq!(
                img.read_metadata_at(0, 4).unwrap(),
                b"aZZd",
                "{backing:?}: the cached read path must see it too"
            );
            assert_eq!(
                std::fs::read(&path).unwrap(),
                b"abcdef",
                "{backing:?}: nothing was to be issued yet"
            );

            // A read that *starts inside* a pending run, rather than at or before
            // it. The overlay has to walk back to the run covering the window's
            // first byte; starting the walk at the window would miss it, and the
            // reader would get the stale byte off the disk.
            let mut inner = [0u8; 1];
            img.read_at(2, &mut inner).unwrap();
            assert_eq!(
                &inner, b"Z",
                "{backing:?}: a read starting inside a pending run missed it"
            );

            img.ordering_barrier().unwrap();
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// Two writes into one page cost one write; the same two, one page apart,
    /// cost two. The join reads the clean bytes between them back, so it must
    /// leave them alone — the case where "one write per page" would otherwise be
    /// implemented by writing zeros over a neighbor.
    #[test]
    fn writes_sharing_a_page_are_issued_as_one() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let initial: Vec<u8> = (0..PAGE as u8 * 3).collect();

            let (path, mut img) =
                gathering_named(dir.path(), "one_page", &initial, backing, GATHERED);
            let before = img.issued_writes();
            img.write_at(2, b"XX").unwrap();
            img.write_at(40, b"YY").unwrap();
            img.ordering_barrier().unwrap();
            assert_eq!(
                img.issued_writes() - before,
                1,
                "{backing:?}: two writes in one page are one write"
            );
            let mut want = initial.clone();
            want[2..4].copy_from_slice(b"XX");
            want[40..42].copy_from_slice(b"YY");
            assert_eq!(
                std::fs::read(&path).unwrap(),
                want,
                "{backing:?}: joining two runs must not disturb the bytes between them"
            );

            let (path2, mut img2) =
                gathering_named(dir.path(), "two_pages", &initial, backing, GATHERED);
            let before = img2.issued_writes();
            img2.write_at(2, b"XX").unwrap();
            img2.write_at(2 + PAGE, b"YY").unwrap();
            img2.ordering_barrier().unwrap();
            assert_eq!(
                img2.issued_writes() - before,
                2,
                "{backing:?}: two writes a page apart stay two"
            );
            let mut want2 = initial.clone();
            want2[2..4].copy_from_slice(b"XX");
            want2[PAGE as usize + 2..PAGE as usize + 4].copy_from_slice(b"YY");
            assert_eq!(std::fs::read(&path2).unwrap(), want2, "{backing:?}");
        }
    }

    /// A write over a gathered one replaces it rather than racing it to the
    /// file: the superblock is rewritten twice within one in-place append, and
    /// the second is the one that must land.
    #[test]
    fn a_later_write_wins_over_the_gathered_one_it_covers() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = gathering(dir.path(), b"abcdefgh", backing, GATHERED);

            img.write_at(2, b"1111").unwrap();
            img.write_at(3, b"22").unwrap();
            img.ordering_barrier().unwrap();

            assert_eq!(std::fs::read(&path).unwrap(), b"ab1221gh", "{backing:?}");
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// Every durability barrier drains first, so what an `fsync` forces is
    /// everything written up to it. A barrier that synced around the gathered
    /// bytes would report a commit durable while it sat in this process.
    #[test]
    fn a_barrier_issues_what_was_gathered() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = gathering(dir.path(), b"abcdef", backing, GATHERED);
            img.write_at(0, b"Z").unwrap();
            img.sync_data().unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), b"Zbcdef", "{backing:?}");

            img.write_at(1, b"Y").unwrap();
            img.sync_all().unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), b"ZYcdef", "{backing:?}");
        }
    }

    /// Truncation drops the gathered bytes it puts past end-of-file *without
    /// issuing them*, and keeps the ones below the cut.
    ///
    /// The result alone cannot say this: writing a doomed run and then cutting it
    /// off leaves the same file, which is why the count is asserted. What it buys
    /// is real — a commit that appends and then trims can otherwise write out
    /// every byte it is about to discard, which is the write amplification this
    /// whole layer exists to remove.
    #[test]
    fn truncate_discards_the_gathered_bytes_past_the_cut_rather_than_writing_them() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = gathering(dir.path(), b"abcdefgh", backing, GATHERED);

            img.append(&[9u8; 200]).unwrap();
            let before = img.issued_writes();
            img.truncate(8).unwrap();

            assert_eq!(
                img.issued_writes(),
                before,
                "{backing:?}: bytes about to stop existing were written out first"
            );
            assert_eq!(img.len(), 8, "{backing:?}");
            assert_eq!(std::fs::read(&path).unwrap(), b"abcdefgh", "{backing:?}");

            // A run straddling the cut keeps its half below it, and carries only
            // that half. Here the write happens either way, so it is the *bytes*
            // that say whether the doomed half rode along.
            let before = img.issued_write_bytes();
            img.write_at(1, b"ZZZZZZ").unwrap();
            img.truncate(4).unwrap();
            assert_eq!(
                img.issued_write_bytes() - before,
                3,
                "{backing:?}: the part of the run past the cut was written anyway"
            );
            assert_eq!(std::fs::read(&path).unwrap(), b"aZZZ", "{backing:?}");
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// After a truncate, the image's idea of the file's real length is the
    /// truncated one — so a later append reads back as what was appended, not as
    /// the end of a file that is no longer there.
    ///
    /// The bounded image serves a read by taking what exists on disk and patching
    /// the pending writes over it, and the boundary between those two is exactly
    /// this length. Leave it stale after a truncate and the read tries to take
    /// bytes off a file that has since shrunk.
    #[test]
    fn a_truncate_leaves_the_real_length_where_a_later_append_can_read_back() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (_path, mut img) = gathering(dir.path(), b"abcdefgh", backing, GATHERED);

            img.truncate(4).unwrap();
            img.append(b"WXYZ").unwrap();

            let mut buf = [0u8; 4];
            img.read_at(4, &mut buf)
                .unwrap_or_else(|e| panic!("{backing:?}: reading the appended range failed: {e}"));
            assert_eq!(&buf, b"WXYZ", "{backing:?}");
            assert_eq!(bytes(img.as_ref()), b"abcdWXYZ", "{backing:?}");
        }
    }

    /// In the general merge — the path taken when a write joins runs on both
    /// sides of it rather than landing inside one — the new bytes still win over
    /// the older run they overlap.
    ///
    /// `a_later_write_wins_over_the_gathered_one_it_covers` pins the same rule on
    /// the wholly-contained fast path. This is the other branch, and it needs a
    /// write that *partly* overlaps a held run and reaches a second one, which no
    /// engine path happens to produce — so nothing else distinguishes copying the
    /// absorbed runs before the new bytes from copying them after.
    #[test]
    fn the_general_merge_also_lets_the_later_write_win() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = gathering(dir.path(), b"abcdefghij", backing, GATHERED);

            // Two runs with a hole between them, then one write that overlaps the
            // tail of the first, spans the hole, and touches the second.
            img.write_at(0, b"111").unwrap();
            img.write_at(6, b"222").unwrap();
            img.write_at(2, b"XXXXX").unwrap();
            img.ordering_barrier().unwrap();

            assert_eq!(
                std::fs::read(&path).unwrap(),
                b"11XXXXX22j",
                "{backing:?}: the general merge must let the later write win"
            );
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// Operation retention drains at every ordering barrier; session retention
    /// does not, which is the whole difference between the default and an
    /// explicit page buffer.
    #[test]
    fn session_retention_survives_an_ordering_barrier_and_operation_retention_does_not() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let held = WriteBuffering::Session {
                page_size: PAGE,
                max_bytes: 4096,
            };
            let (path, mut img) = gathering_named(dir.path(), "held", b"abcdef", backing, held);
            img.write_at(0, b"Z").unwrap();
            img.ordering_barrier().unwrap();
            assert_eq!(
                std::fs::read(&path).unwrap(),
                b"abcdef",
                "{backing:?}: a page buffer must survive an ordering barrier"
            );
            img.sync_all().unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), b"Zbcdef", "{backing:?}");

            // The other half of the name, which is the default and the one the
            // engine's barriers depend on.
            let (released, mut img) =
                gathering_named(dir.path(), "released", b"abcdef", backing, GATHERED);
            img.write_at(0, b"Z").unwrap();
            img.ordering_barrier().unwrap();
            assert_eq!(
                std::fs::read(&released).unwrap(),
                b"Zbcdef",
                "{backing:?}: operation retention must release at an ordering barrier"
            );
        }
    }

    /// The budget is a ceiling on what is held, not advice: a session-retained
    /// buffer that never drained until close would spend memory without bound on
    /// a long-running writer.
    #[test]
    fn the_budget_drains_a_buffer_that_would_outgrow_it() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = gathering(
                dir.path(),
                &vec![0u8; 4096],
                backing,
                WriteBuffering::Session {
                    page_size: PAGE,
                    max_bytes: 100,
                },
            );
            // Three separate pages, 64 bytes each: the third crosses 100 bytes.
            for page in 0..3u64 {
                img.write_at(page * PAGE, &[7u8; 40]).unwrap();
            }
            assert_ne!(
                std::fs::read(&path).unwrap(),
                vec![0u8; 4096],
                "{backing:?}: the budget must have forced a drain"
            );
            img.sync_all().unwrap();
            assert_in_sync(&path, img.as_ref(), backing);
        }
    }

    /// A truncate that **fails** keeps every write it would have discarded.
    ///
    /// `discard_from` is irreversible, so the order it sits in is the whole
    /// subject: the doomed runs have to go before the flush, or bytes about to
    /// stop existing get written out (the test above), and they must not go
    /// before the truncate, or a truncate that fails leaves the image reporting
    /// a length whose bytes nothing will ever write. Putting the truncate first
    /// satisfies both, since until it succeeds nothing is doomed.
    ///
    /// Asserted through a read-only handle, which is the cheapest `set_len`
    /// failure there is. No engine path reaches this today — the one
    /// `FileImage::truncate` caller barriers first, so the buffer is always
    /// empty by the time it arrives — which is exactly why it needs a test: the
    /// suite cannot otherwise tell this order from the one that loses the write.
    #[test]
    fn a_failed_truncate_keeps_the_writes_it_would_have_discarded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("read_only");
        std::fs::write(&path, b"abcdefgh").unwrap();

        let mut writes = BufferedWrites::new(fs::File::open(&path).unwrap(), 8);
        writes.set_mode(GATHERED).unwrap();
        writes.write_at(8, &[9u8; 200]).unwrap();
        assert_eq!(
            writes.pending_bytes, 200,
            "the write must be held, not issued"
        );

        assert!(
            writes.set_len(8).is_err(),
            "a read-only handle must refuse set_len, or this proves nothing"
        );
        assert_eq!(
            writes.pending_bytes, 200,
            "a truncate that failed discarded the writes it never doomed"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"abcdefgh",
            "a failed truncate must not have written anything either"
        );
    }

    /// A flush that cannot write leaves its writes **pending**, so the next one
    /// retries them instead of reporting success over a batch it lost.
    ///
    /// This is the failure mode buffering introduces and straight-through writing
    /// cannot have: a write is accepted, reported as fine, and only fails later at
    /// the drain. If that drain then emptied the buffer, the `force_sync` on the
    /// close path would find nothing to do and return `Ok` over a file missing up
    /// to a whole budget of writes — turning "the commit errored" into "the commit
    /// errored and then close said fine".
    ///
    /// A read-only handle is the cheapest write failure to arrange.
    #[test]
    fn a_failed_flush_keeps_its_writes_pending() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("readonly.bin");
        std::fs::write(&path, b"abcdefgh").unwrap();
        let handle = fs::OpenOptions::new().read(true).open(&path).unwrap();
        let mut img = MirrorImage::new(handle, b"abcdefgh".to_vec());
        img.set_write_buffering(GATHERED).unwrap();

        img.write_at(0, b"ZZ").unwrap();
        assert!(
            img.sync_all().is_err(),
            "a write to a read-only handle must fail"
        );
        assert!(
            img.sync_all().is_err(),
            "the retry must report the failure too, not an empty buffer's success"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"abcdefgh",
            "nothing should have reached the file"
        );

        // The other failure inside a flush is the read that fills the gap between
        // two runs sharing a page. A write-only handle reaches exactly it: the
        // mirror serves ordinary reads from memory, so this is the only read in
        // play, and the writes must survive it just the same.
        let write_only = fs::OpenOptions::new().write(true).open(&path).unwrap();
        let mut gapped = MirrorImage::new(write_only, b"abcdefgh".to_vec());
        gapped.set_write_buffering(GATHERED).unwrap();
        gapped.write_at(0, b"X").unwrap();
        gapped.write_at(4, b"Y").unwrap();
        assert!(
            gapped.sync_all().is_err(),
            "the gap read must fail on a handle that cannot read"
        );
        assert!(
            gapped.sync_all().is_err(),
            "and the retry must still report it rather than an empty buffer"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"abcdefgh",
            "a failed gap read must not have written a partial join"
        );

        // A failed gap read must also leave the run at the length it had, not at
        // the length the aborted read resized it to. Keeping the zero padding
        // would leave the run *touching* its neighbour, so a retry would find no
        // gap to read and would write those zeros over the clean bytes between
        // them. Asserted against the buffer directly: the retry cannot be run
        // here (the handle still cannot read), and the state is the invariant.
        let write_only = fs::OpenOptions::new().write(true).open(&path).unwrap();
        let mut w = BufferedWrites::new(write_only, 8);
        w.set_mode(GATHERED).unwrap();
        w.write_at(0, b"X").unwrap();
        w.write_at(4, b"Y").unwrap();
        assert!(w.flush().is_err(), "the gap read must fail");
        assert_eq!(
            w.pending_bytes, 2,
            "the run kept the padding the failed gap read added"
        );
        assert_eq!(
            w.runs.get(&0).map(Vec::len),
            Some(1),
            "the restored run must end where it did, clear of its neighbour"
        );

        // And once the obstacle is gone, the pending writes are still there to
        // land — the point of keeping them rather than merely reporting.
        let writable = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut recovered = MirrorImage::new(writable, b"abcdefgh".to_vec());
        recovered.set_write_buffering(GATHERED).unwrap();
        recovered.write_at(0, b"ZZ").unwrap();
        recovered.sync_all().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"ZZcdefgh");
    }

    /// A session dropped without a teardown still lands its writes. The engine's
    /// own close syncs, so this covers the paths that do not — an unwind, and the
    /// bare engines the tests build.
    #[test]
    fn dropping_the_image_issues_what_it_still_holds() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let (path, mut img) = gathering(dir.path(), b"abcdef", backing, GATHERED);
            img.write_at(0, b"Z").unwrap();
            drop(img);

            assert_eq!(std::fs::read(&path).unwrap(), b"Zbcdef", "{backing:?}");
        }
    }

    /// Only the mirror lends its buffer out; the handle image withholds it, and
    /// that difference is the one the read paths branch on.
    #[test]
    fn only_the_mirror_offers_a_whole_file_slice() {
        let dir = tempfile::tempdir().unwrap();
        let (_p1, mirror) = image(dir.path(), b"abcdef", Backing::Mirror);
        let (_p2, handle) = image(dir.path(), b"abcdef", Backing::Handle);

        assert_eq!(mirror.as_slice(), Some(&b"abcdef"[..]));
        assert!(handle.as_slice().is_none());
    }

    /// `SourceOnlyImage` must differ from the mirror in exactly one respect.
    #[test]
    fn source_only_withholds_the_slice_but_reads_the_same() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source_only.bin");
        std::fs::write(&path, b"abcdef").unwrap();
        let handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut only = SourceOnlyImage::new(MirrorImage::new(handle, b"abcdef".to_vec()));

        assert!(only.as_slice().is_none(), "the slice must be withheld");

        only.write_at(1, b"Z").unwrap();
        only.append(b"gh").unwrap();
        assert_eq!(only.len(), 8);

        let mut buf = [0u8; 8];
        only.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"aZcdefgh");
    }

    // -----------------------------------------------------------------------
    // Differential tests against a byte model
    // -----------------------------------------------------------------------

    /// A deterministic xorshift, so a failure is reproducible from its seed.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A value in `0..n`. Every caller passes a positive `n`.
        fn upto(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// The gathering configurations the sweep below runs under.
    ///
    /// The narrow ones are the point. `max_bytes` of 96 against writes of up to
    /// 80 bytes means a single write frequently exceeds the budget on its own,
    /// which is the bypass in [`BufferedWrites::write_at`] that flushes and
    /// issues rather than absorbing; a configuration generous enough never to
    /// reach it leaves that path untested. The 1-byte page is the degenerate end,
    /// where two writes merge only by touching.
    const MIXES: [WriteBuffering; 5] = [
        WriteBuffering::Unbuffered,
        WriteBuffering::Operation {
            page_size: 1,
            max_bytes: 4096,
        },
        WriteBuffering::Operation {
            page_size: 16,
            max_bytes: 96,
        },
        WriteBuffering::Operation {
            page_size: 64,
            max_bytes: 4096,
        },
        WriteBuffering::Operation {
            page_size: 4096,
            max_bytes: 1 << 20,
        },
    ];

    /// Random sequences of every primitive, against a `Vec<u8>` that models what
    /// the file should hold, on both backings and under every gathering
    /// configuration.
    ///
    /// Two claims, checked after *every* operation rather than at the end, so a
    /// failure names the operation that caused it rather than the one that
    /// happened to notice:
    ///
    /// - reads through the image equal the model, which for the handle backing
    ///   means the overlay reassembles pending runs, clean bytes and the gaps
    ///   between them correctly;
    /// - at every ordering barrier the *file* equals the model too. The
    ///   gathering may delay a write, but never lose or reorder one across the
    ///   point that exists to bound it.
    ///
    /// Sequences rather than cases, because what this shape catches is about
    /// *history*: a run left touching its neighbour, a merge that copies the old
    /// bytes over the new, a trailing-run scan off by one at a page boundary.
    /// None of those is reachable by a single write.
    ///
    /// The in-loop barrier is [`FileImage::ordering_barrier`] and not
    /// `sync_data`, which would prove exactly the same thing about the model and
    /// cost twenty times the wall clock: measured here at **12.4s against 0.6s**,
    /// the difference being 2,400 `fsync`s that flush what the barrier had
    /// already flushed. One `sync_data` per configuration covers the flush-then-
    /// fsync ordering, and `assert_in_sync` closes each one.
    #[test]
    fn a_random_operation_sequence_matches_a_byte_model() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            for (mi, mode) in MIXES.into_iter().enumerate() {
                for seed in 0..8u64 {
                    let initial: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
                    let sub = dir.path().join(std::format!("m{mi}s{seed}"));
                    std::fs::create_dir_all(&sub).unwrap();
                    let (path, mut img) = gathering(&sub, &initial, backing, mode);
                    let mut model = initial.clone();
                    let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15 ^ (seed << 32) ^ (mi as u64));
                    let at =
                        |step: u32| std::format!("{backing:?} mode {mi} seed {seed} step {step}");
                    // One step per configuration takes the durable barrier
                    // instead of the ordering one.
                    let fsync_at = rng.upto(300) as u32;

                    for step in 0..300u32 {
                        match rng.upto(10) {
                            0..=3 if !model.is_empty() => {
                                let len = 1 + rng.upto(48).min(model.len() as u64 - 1);
                                let offset = rng.upto(model.len() as u64 - len + 1);
                                let bytes = vec![(rng.next() & 0xff) as u8; len as usize];
                                img.write_at(offset, &bytes).unwrap();
                                model[offset as usize..(offset + len) as usize]
                                    .copy_from_slice(&bytes);
                            }
                            4..=6 => {
                                let bytes =
                                    vec![(rng.next() & 0xff) as u8; 1 + rng.upto(80) as usize];
                                let placed = img.append(&bytes).unwrap();
                                assert_eq!(placed, model.len() as u64, "{}: append", at(step));
                                model.extend_from_slice(&bytes);
                            }
                            7 => {
                                let keep = rng.upto(model.len() as u64 + 1);
                                img.truncate(keep).unwrap();
                                model.truncate(keep as usize);
                            }
                            _ => {
                                if step == fsync_at {
                                    img.sync_data().unwrap();
                                } else {
                                    img.ordering_barrier().unwrap();
                                }
                                assert_eq!(
                                    std::fs::read(&path).unwrap(),
                                    model,
                                    "{}: a barrier left the file disagreeing with the model",
                                    at(step)
                                );
                            }
                        }
                        assert_eq!(img.len(), model.len() as u64, "{}: length", at(step));
                        assert_eq!(bytes(img.as_ref()), model, "{}: reads", at(step));
                        if let Some(slice) = img.as_slice() {
                            assert_eq!(slice, &model[..], "{}: slice", at(step));
                        }
                    }

                    img.sync_all().unwrap();
                    assert_in_sync(&path, img.as_ref(), backing);
                }
            }
        }
    }

    /// Every window of a buffer holding several pending runs reads back as the
    /// model says, including the ones that make the overlay's bookkeeping
    /// awkward: a window entirely inside one run, one spanning a gap, one
    /// covering two runs that meet on the same byte, a single byte, an empty
    /// window, and one reaching past the file's real end into bytes that exist
    /// only as a pending append.
    ///
    /// Exhaustive rather than sampled. There are only a few thousand windows, and
    /// choosing a handful by hand is how an off-by-one at exactly one boundary
    /// survives.
    #[test]
    fn every_window_over_a_buffered_image_matches_the_model() {
        let dir = tempfile::tempdir().unwrap();
        for backing in BACKINGS {
            let initial: Vec<u8> = (0..200u32).map(|i| (i % 251) as u8).collect();
            let (_, mut img) = gathering(dir.path(), &initial, backing, GATHERED);
            let mut model = initial.clone();

            // Runs chosen for their relationships rather than their contents: two
            // that meet exactly at byte 40, one alone in the middle of a page, a
            // single byte, and one that exists only as a pending append past the
            // file's real end.
            let put = |img: &mut Box<dyn FileImage>, model: &mut Vec<u8>, at: u64, b: &[u8]| {
                img.write_at(at, b).unwrap();
                model[at as usize..at as usize + b.len()].copy_from_slice(b);
            };
            put(&mut img, &mut model, 30, &[0xA1; 10]);
            put(&mut img, &mut model, 40, &[0xA2; 10]);
            put(&mut img, &mut model, 90, &[0xB0; 1]);
            put(&mut img, &mut model, 130, &[0xC0; 20]);
            let tail = [0xD0u8; 30];
            img.append(&tail).unwrap();
            model.extend_from_slice(&tail);

            let len = model.len() as u64;
            assert_eq!(img.len(), len);
            for start in 0..=len {
                for end in start..=len {
                    let mut buf = vec![0u8; (end - start) as usize];
                    img.read_at(start, &mut buf).unwrap();
                    assert_eq!(
                        buf,
                        &model[start as usize..end as usize],
                        "{backing:?}: window {start}..{end}"
                    );
                }
            }
        }
    }
}
