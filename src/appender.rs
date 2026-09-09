//! A write-side chunk buffer for appends ([`BufferedAppender`]).
//!
//! [`Dataset::append`](crate::Dataset::append) writes on every call: it encodes
//! the appended elements, places the chunks they land in, extends the chunk
//! index, and fsyncs five times so a power loss leaves either the old length or
//! the new one. That is the right trade for a caller appending a chunk at a
//! time, and the wrong one for a caller appending a hundred elements at a time
//! into a chunk that holds a thousand — which pays the whole sequence ten times
//! over to write one chunk.
//!
//! A [`BufferedAppender`] holds appended elements in memory and writes them
//! only when they complete a chunk, so the file sees one append per chunk
//! rather than one per call. It is this crate's equivalent of the reference C
//! library's raw-data chunk cache, and it carries the same bargain: buffered
//! elements are not in the file until [`flush`](BufferedAppender::flush) or
//! [`finish`](BufferedAppender::finish) puts them there.

use core::num::NonZeroUsize;

use crate::convert::TryToUsize;
use crate::edit::AppendBuilder;
use crate::element::H5Element;
use crate::error::Error;
use crate::reader::Dataset;

/// Buffers appended elements and writes them a whole chunk at a time.
///
/// Obtained from [`Dataset::buffered_appender`]. The dataset must be one
/// [`Dataset::append`] accepts — chunked, rank-1, unlimited along axis 0, and
/// Extensible-Array indexed — and that eligibility, along with the refusal of a
/// SWMR session (which cannot write a partial trailing chunk at all), is
/// reported when the appender is constructed rather than on the first write.
///
/// The appender borrows the [`Dataset`] for its lifetime, so that handle cannot
/// be read while it is alive; open a second handle off the same [`File`](crate::File) to read
/// the written prefix as it grows. That borrow is what lets the appender re-read
/// the dataset's length from the session before every write instead of trusting
/// a cached one.
///
/// # What it costs
///
/// Every [`append`](Self::append) that does not complete a chunk is a memory
/// copy and nothing else. When one or more chunks are complete, exactly those
/// chunks are written through the immediate, crash-atomic in-place path, and
/// the remainder stays buffered. So a caller appending `k` elements at a time
/// into a chunk of `n` performs one file write per `n/k` calls instead of one
/// per call. Every write it makes is the immediate in-place one, on a filtered
/// dataset as much as an unfiltered one.
///
/// # Durability
///
/// Buffered elements are not in the file. Each write the appender does make is
/// itself crash-atomic, so a crash loses the buffered tail and never the file:
/// the dataset reads back as the prefix that was written. That holds against
/// power loss under the default [`SyncPolicy`](crate::SyncPolicy) and against a
/// process crash under either — see [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose). [`flush`](Self::flush)
/// writes the buffered tail immediately, and [`finish`](Self::finish) flushes
/// and consumes the appender.
///
/// Dropping without either still flushes, but a `Drop` has nowhere to return an
/// error, so prefer [`finish`](Self::finish) where the error matters — or
/// [`flush`](Self::flush), which keeps the appender and therefore keeps
/// [`unwritten`](Self::unwritten) reachable.
///
/// # Why staged edits are refused around it
///
/// An appender holds elements the caller has already been told were accepted,
/// and only its own flush can write them. The immediate append path refuses a
/// dataset that has a staged edit on it or an ancestor, so staging one while an
/// appender is live would turn accepted data into data lost silently at drop
/// time. The session therefore refuses that edit at the call that makes it,
/// with [`Error::EditUnsupported`], for as long as the appender is alive:
///
/// - an edit naming this dataset or an ancestor —
///   [`Group::create_dataset`](crate::Group::create_dataset),
///   [`Dataset::set_attr`](crate::Dataset::set_attr),
///   [`Group::delete`](crate::Group::delete), [`File::copy`](crate::File::copy)
///   and the rest;
/// - a second appender on the same dataset, which would interleave the two
///   buffers a chunk at a time.
///
/// An edit naming something *else* is accepted beside a live appender: every
/// write the appender makes is immediate and stages nothing, so it has no commit
/// of its own to protect.
///
/// Each of those is reported to the caller who created the conflict, which is
/// the only place it can be acted on. What remains outside that guarantee is
/// closing the file — [`File::close`](crate::File::close) does not flush live
/// appenders — and a buffer left ending mid-element, which only
/// [`append_raw`](Self::append_raw) can produce.
///
/// # A filtered dataset resumed mid-chunk
///
/// A *filtered* dataset whose on-disk length is not a whole multiple of its
/// chunk length — a log resumed across sessions, typically — costs nothing
/// extra: the in-place path re-encodes that trailing chunk into a fresh
/// allocation and repoints its one index element, and the bytes it vacates are
/// reclaimed by [`repack`](crate::repack()). The appender still chooses its write
/// prefix so the on-disk length lands back on a chunk boundary, which keeps the
/// re-encoding to the first write rather than one per call.
///
/// A **lossy** pipeline (ZFP, or float D-scale scale-offset) cannot have that
/// chunk re-encoded at all — the values it holds would change — so an appender
/// over a lossy dataset that already sits on a partial trailing chunk is refused
/// when it is constructed. One over a chunk-aligned lossy dataset is fine, and
/// stays fine as long as its writes land on boundaries. A mid-stream
/// [`flush`](Self::flush) is what would break that, so on a lossy dataset a
/// flush that would leave the length unaligned is refused at the call, with the
/// buffer untouched: [`finish`](Self::finish) (and the flush a drop makes) writes
/// the partial tail, because it is the last write the appender will make and
/// there is no later one for the unaligned length to refuse. Buffer until a whole
/// chunk, or finish.
///
/// ```no_run
/// # use hdf5_pure::File;
/// # fn main() -> Result<(), hdf5_pure::Error> {
/// let file = File::open_rw("telemetry.h5")?;
/// let mut ds = file.dataset("samples")?;
/// let mut app = ds.buffered_appender()?;
/// for batch in 0..1000 {
///     app.append(&[batch as f64; 100])?;   // buffered; writes once a chunk fills
/// }
/// app.finish()?;                            // the partial tail reaches the file here
/// # Ok(())
/// # }
/// ```
pub struct BufferedAppender<'a> {
    dataset: &'a mut Dataset,
    /// Chunk length, element width, and filter state, read once at construction.
    /// The dataset's length is not cached here: it is re-read from the session
    /// whenever a write is actually due, so an append through another handle
    /// cannot leave this appender writing against a stale length.
    chunk_elems: u64,
    element_size: NonZeroUsize,
    /// Whether the pipeline is lossy, and so cannot have a partial trailing
    /// chunk re-encoded. Checked at construction, again by every non-terminal
    /// [`flush`](BufferedAppender::flush) — which is the only way this type could
    /// leave a tail for a later write to regrow — and reported by [`Debug`].
    lossy_filters: bool,
    /// Elements appended but not yet written, as little-endian bytes.
    pending: AppendBuilder,
    /// Set by a failed write. Blocks further use and stops `Drop` from retrying
    /// a write that already failed once.
    poisoned: bool,
    /// Set by `finish`, so its `Drop` does not flush a second time.
    finished: bool,
    /// This appender's registration with the session. While it stands, a staged
    /// edit that would stop this appender from flushing is refused at the call
    /// that makes it, instead of failing later in `Drop` where the buffer would
    /// simply be lost.
    claim: u64,
}

impl std::fmt::Debug for BufferedAppender<'_> {
    /// Reports the buffer state, not the dataset: the borrowed `Dataset`'s own
    /// `Debug` would re-print a header this type does not touch.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferedAppender")
            .field("chunk_elements", &self.chunk_elems)
            .field("buffered_elements", &self.buffered_elements())
            .field("lossy_filters", &self.lossy_filters)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

/// Whether a flush is the last write an appender will make, which is what
/// decides if it may leave a lossy dataset's length off a chunk boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Terminal {
    /// [`BufferedAppender::flush`]: the appender lives on, and a later write
    /// would have to regrow whatever tail this one leaves.
    No,
    /// [`BufferedAppender::finish`] and the flush a drop makes: nothing follows.
    Yes,
}

impl<'a> BufferedAppender<'a> {
    /// Construct an appender over `dataset`, applying every eligibility refusal
    /// [`Dataset::append`] applies.
    pub(crate) fn new(dataset: &'a mut Dataset) -> Result<Self, Error> {
        let g = dataset.append_geometry()?;
        // The SWMR writer's append rules are a strict subset: it requires the
        // *appended* length to be chunk-aligned too, so the one write this type
        // exists to make — the partial tail — can never succeed there. Refusing
        // now keeps the promise that eligibility is settled at construction; the
        // alternative is a `finish` that always fails and a drop that silently
        // discards.
        if dataset.session_is_swmr() {
            return Err(Error::SwmrAppendUnsupported(
                "a SWMR writer cannot write a partial trailing chunk, which is what a buffered \
                 appender exists to do; append whole chunks with Dataset::append instead",
            ));
        }
        let chunk_elems = g.chunk_elems.max(1);
        // A *lossy* pipeline (ZFP, float D-scale scale-offset) cannot have a
        // partial trailing chunk grown at all — that would re-encode committed
        // values. An appender over one that is already sitting on such a chunk
        // could not make even its first write, so it is refused here for the same
        // reason the SWMR case above is, rather than accepting elements it would
        // then have to hand back.
        if g.lossy_filters && g.current_dim % chunk_elems != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "this dataset's filter pipeline is lossy and its length is not a whole multiple \
                 of the chunk length, so a buffered appender's first write would have to \
                 re-encode already-committed values; append whole chunks from a chunk-aligned \
                 length instead",
            ));
        }
        let claim = dataset.claim_for_appender()?;
        Ok(Self {
            dataset,
            chunk_elems,
            element_size: g.element_size,
            lossy_filters: g.lossy_filters,
            pending: AppendBuilder::new(),
            poisoned: false,
            finished: false,
            claim,
        })
    }

    /// Append a slice of any supported scalar type, buffering it and writing
    /// out whatever whole chunks it completes.
    ///
    /// The element type must match the dataset's on-disk datatype exactly; a
    /// mismatch is reported by the write this call or a later one triggers.
    pub fn append<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        self.check_usable()?;
        let mark = self.pending.raw().len();
        self.pending.append(data);
        self.settle(mark)
    }

    /// Append already-little-endian element bytes verbatim, buffering them and
    /// writing out whatever whole chunks they complete.
    ///
    /// Unlike [`Dataset::append_raw`], `bytes` need not be a whole number of
    /// elements: a trailing partial element stays buffered for the next call to
    /// complete. It must be whole by the time the appender writes it, which
    /// [`flush`](Self::flush) and [`finish`](Self::finish) check.
    pub fn append_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.check_usable()?;
        let mark = self.pending.raw().len();
        self.pending.append_raw(bytes);
        self.settle(mark)
    }

    /// Whole elements buffered in memory and not yet written to the file.
    ///
    /// A trailing *partial* element — which only [`append_raw`](Self::append_raw)
    /// can leave — is not counted here but is present in
    /// [`unwritten`](Self::unwritten), so the two disagree by up to one element's
    /// worth of bytes until the next call completes it.
    #[must_use]
    pub fn buffered_elements(&self) -> u64 {
        self.buffered()
    }

    /// Buffered whole elements, discarding any trailing partial one. The count
    /// comes from an in-memory byte length, so it is a widening on every
    /// supported target.
    fn buffered(&self) -> u64 {
        (self.pending.raw().len() / self.element_size) as u64
    }

    /// The buffered element bytes not yet written, little-endian and in append
    /// order. Non-empty after a failed write, which is the case it exists for:
    /// the file holds exactly the elements the length reports, and these are
    /// the ones that did not reach it.
    #[must_use]
    pub fn unwritten(&self) -> &[u8] {
        self.pending.raw()
    }

    /// Elements per chunk along the appended axis — the granularity at which
    /// [`append`](Self::append) reaches the file on its own.
    #[must_use]
    pub const fn chunk_elements(&self) -> u64 {
        self.chunk_elems
    }

    /// Write every buffered element to the file now, including a partial
    /// trailing chunk.
    ///
    /// This is the durability point, and under a lossless pipeline it costs one
    /// in-place write whatever the dataset's alignment. Flushing a partial chunk
    /// does leave the on-disk length unaligned, so the next write re-encodes that
    /// trailing chunk; let the appender batch where the last few elements can
    /// wait.
    ///
    /// Under a **lossy** pipeline (ZFP, or float D-scale scale-offset) that
    /// trailing chunk cannot be re-encoded at all, so a flush that would leave the
    /// on-disk length unaligned is refused here, before anything is written and
    /// with every buffered element still buffered. Left to happen, it would strand
    /// the appender one call later: the *next* write is the one that has to regrow
    /// the tail, and it fails with the lossy refusal and poisons the appender —
    /// reporting a loss of the caller's elements at a call that did nothing wrong.
    /// Keep buffering until the elements complete a chunk, or end the appender
    /// with [`finish`](Self::finish), which writes the partial tail because no
    /// write follows it.
    pub fn flush(&mut self) -> Result<(), Error> {
        self.flush_prefix(Terminal::No)
    }

    /// [`flush`](Self::flush)'s body, parameterized by whether a write may follow.
    ///
    /// The partial tail a lossy dataset cannot regrow is refused only while the
    /// appender lives on: `finish` and the flush a drop makes are the last write
    /// there will be, and refusing them would discard the elements instead of
    /// landing them — the opposite of what the refusal is for.
    fn flush_prefix(&mut self, terminal: Terminal) -> Result<(), Error> {
        self.check_usable()?;
        let avail = self.whole_elements()?;
        if avail == 0 {
            return Ok(());
        }
        if self.lossy_filters && terminal == Terminal::No {
            let current = self.dataset.append_geometry()?.current_dim;
            if current.saturating_add(avail) % self.chunk_elems != 0 {
                return Err(Error::AppendInPlaceUnsupported(
                    "this dataset's filter pipeline is lossy, and flushing these elements would \
                     leave its length short of a chunk boundary — a later write would then have \
                     to re-encode already-committed values; buffer until the elements complete \
                     a chunk, or end the appender with BufferedAppender::finish, which writes \
                     the partial tail",
                ));
            }
        }
        self.write_prefix(avail)
    }

    /// Discard every buffered element and consume the appender without writing
    /// them, returning the bytes that were dropped.
    ///
    /// The counterpart to [`finish`](Self::finish): because dropping flushes,
    /// this is the only way to abandon appended elements that have not reached
    /// the file. Elements already written by an earlier call are in the file and
    /// are not affected.
    pub fn discard(mut self) -> Vec<u8> {
        self.finished = true;
        std::mem::replace(&mut self.pending, AppendBuilder::new()).into_raw()
    }

    /// Flush every buffered element and consume the appender.
    ///
    /// Prefer this to dropping, which performs the same flush but cannot report
    /// its error. Note that consuming the appender also consumes the buffer: if
    /// this returns `Err`, the elements that did not land are gone with it. Use
    /// [`flush`](Self::flush) instead when you mean to recover them through
    /// [`unwritten`](Self::unwritten), and `finish` once it has succeeded.
    ///
    /// This is also the one flush that writes a partial trailing chunk onto a
    /// **lossy** dataset. [`flush`](Self::flush) refuses that, because the write
    /// after it could not regrow the tail; here there is no write after it.
    pub fn finish(mut self) -> Result<(), Error> {
        self.finished = true;
        self.flush_prefix(Terminal::Yes)
    }

    /// Write out whatever chunks the just-appended bytes completed, rolling the
    /// call back to `mark` if that was refused before anything reached the file.
    ///
    /// Without the rollback an `Err` would leave the appended bytes buffered,
    /// and a caller who reads `Err` as "not appended" — the contract
    /// [`Dataset::append`] has — would append them a second time. A refusal that
    /// happens *after* a partial write instead poisons the appender and leaves
    /// the un-landed elements buffered, which is the documented recovery.
    fn settle(&mut self, mark: usize) -> Result<(), Error> {
        let result = self.write_complete_chunks();
        if result.is_err() && !self.poisoned {
            self.pending.truncate(mark);
        }
        result
    }

    /// Refuse use after a failed write. The file is intact and its length
    /// reports exactly which elements landed; [`unwritten`](Self::unwritten)
    /// holds the rest.
    fn check_usable(&self) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::AppendInPlaceUnsupported(
                "this buffered appender failed a write and holds unwritten elements; read them \
                 back with BufferedAppender::unwritten and start a new appender",
            ));
        }
        Ok(())
    }

    /// Buffered whole elements, refusing a trailing partial element (only
    /// reachable through [`append_raw`](Self::append_raw), which admits one).
    fn whole_elements(&self) -> Result<u64, Error> {
        if self.pending.raw().len() % self.element_size != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "the buffered byte length is not a whole number of elements; append the rest of \
                 the trailing element before flushing",
            ));
        }
        Ok(self.buffered())
    }

    /// Write out whatever whole chunks the buffer now completes, leaving the
    /// remainder buffered. Chooses the prefix length so the file lands on a
    /// chunk boundary, which is what keeps every later write on the cheap
    /// in-place path.
    fn write_complete_chunks(&mut self) -> Result<(), Error> {
        let avail = self.buffered();
        let c = self.chunk_elems;
        // Cheap pre-check against the length as of the last write: a call that
        // cannot possibly complete a chunk takes no lock and reads no geometry.
        if avail < c {
            return Ok(());
        }
        let partial = self.dataset.append_geometry()?.current_dim % c;
        let to_boundary = (c - partial) % c;
        // `to_boundary < c <= avail` after the early return above, so the prefix
        // is always at least one element and never more than the buffer holds.
        debug_assert!(to_boundary < c && to_boundary <= avail);
        let take = to_boundary + ((avail - to_boundary) / c) * c;
        debug_assert!((1..=avail).contains(&take));
        self.write_prefix(take)
    }

    /// Write the first `take_elems` buffered elements, draining them on success.
    ///
    /// Always the immediate in-place path: since issue #393 that path re-encodes
    /// a filtered partial trailing chunk into a fresh allocation, so there is no
    /// shape of dataset left for which this type has to commit.
    fn write_prefix(&mut self, take_elems: u64) -> Result<(), Error> {
        let before = self.dataset.append_geometry()?.current_dim;
        // Copy the prefix rather than split it out: a failed write must leave
        // the buffer holding exactly the elements that did not land, and how
        // many those are is only known afterwards.
        let head = self.pending.head(
            take_elems
                .to_usize()?
                .saturating_mul(self.element_size.get()),
        );

        let result = self.dataset.append_prebuilt(&head);
        // A multi-batch in-place append can fail with earlier batches durable, so
        // on failure ask the file how far it actually got rather than assuming
        // all or none. A success advanced the dimension by exactly the prefix.
        let landed = if result.is_ok() {
            take_elems
        } else {
            self.dataset
                .append_geometry()
                .map_or(0, |g| g.current_dim.saturating_sub(before))
        };
        self.pending.drop_front(
            landed
                .to_usize()
                .unwrap_or(usize::MAX)
                .saturating_mul(self.element_size.get()),
        );
        if let Err(e) = result {
            self.poisoned = true;
            return Err(e);
        }
        Ok(())
    }
}

impl Drop for BufferedAppender<'_> {
    fn drop(&mut self) {
        if !(self.finished || self.poisoned || self.pending.raw().is_empty()) {
            // Best effort: an error here has nowhere to go. `finish` is the way
            // to see it, which is what that method's documentation says. The
            // claim held until now is what keeps the reachable causes of a
            // refusal here from arising in the first place.
            let _ = self.flush_prefix(Terminal::Yes);
        }
        self.dataset.release_appender_claim(self.claim);
    }
}
