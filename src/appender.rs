//! Chunk buffers for immediate dataset appends.
//!
//! [`BufferedAppender`] borrows a dataset, and [`OwnedBufferedAppender`] owns one. Both buffer
//! elements until an automatic write or explicit flush. Writes use the immediate append path
//! and its [`SyncPolicy`](crate::SyncPolicy) barriers.

use core::num::{NonZeroU64, NonZeroUsize};

use crate::convert::Narrow;
use crate::edit::{AppendBuilder, AppendGeometry, AppendOptions, LossyTailReencoding};
use crate::element::H5Element;
use crate::error::{Error, FormatError};
use crate::reader::Dataset;

/// Buffers appended elements and writes complete chunks automatically.
///
/// Created by [`Dataset::buffered_appender`] or
/// [`Dataset::buffered_appender_with_options`]. The dataset must be chunked, rank one,
/// unlimited along axis 0, and Extensible-Array indexed. Construction rejects SWMR sessions.
///
/// The appender borrows the [`Dataset`] for its lifetime. A second handle from the same
/// [`File`](crate::File) can read the written prefix. Before each write, the appender reads
/// the dataset's current length from the session. For an appender that owns its dataset, see
/// [`OwnedBufferedAppender`].
///
/// An eligible empty dataset may have an unallocated index. Construction and buffering leave
/// index and element storage unallocated until the first write, as described by [`Dataset::append`].
/// Empty appends, empty flushes, and dropping an empty appender leave that storage unallocated.
///
/// # Buffering and recovery
///
/// Automatic writes end at a chunk boundary and leave the remaining bytes buffered.
/// [`flush`](Self::flush) writes a partial trailing chunk and retains the appender.
/// [`finish`](Self::finish) flushes and consumes it, including on error.
/// [`discard`](Self::discard) consumes it and returns the pending bytes without writing them.
///
/// If appending fails before the write path, only the bytes submitted by that call are rejected.
/// If a write fails, the appender is poisoned, including when no elements were written.
/// Further writes are rejected. [`unwritten`](Self::unwritten) and [`discard`](Self::discard)
/// provide the bytes after the written prefix for recovery.
///
/// Dropping attempts a terminal flush unless a write has failed or the appender was consumed
/// by a terminal operation. Errors during drop are ignored. Use [`finish`](Self::finish) to
/// receive a flush error, or [`flush`](Self::flush) to retain access to unwritten bytes.
/// Buffered bytes are lost on a crash. Writes use the immediate append path and its
/// [`SyncPolicy`](crate::SyncPolicy) barriers.
///
/// # Storage reuse
///
/// During automatic writes and partial or terminal flushes, the editor retires a replaced tail's
/// complete allocation after confirming the replacement index record. Reuse follows the file-space
/// strategy, threshold, and page-class rules described by [`Dataset::append`]. The editor retains
/// confirmed retirements if a later write fails. Possibly referenced allocations remain unavailable
/// after uncertain publication.
///
/// For persistent files, the editor publishes remaining free-space manager changes during
/// [`File::commit`](crate::File::commit) or clean session closure, even when EOF is unchanged.
/// After uncertain publication, the editor rejects manager cleanup in that session. Call
/// [`File::commit`](crate::File::commit) before [`File::sync`](crate::File::sync) to include those
/// pending manager updates in an explicit synchronization.
///
/// # Session state
///
/// The session rejects a second appender on the same dataset and staged edits to the
/// dataset or an ancestor with [`Error::EditUnsupported`]. For datasets opened by path,
/// unrelated staged edits are allowed and remain pending after appender writes. An appender
/// over a dataset handle reached by object reference blocks all staged edits and other
/// appenders in the session.
///
/// [`File::close`](crate::File::close) does not flush live appender buffers. After closure,
/// appending, flushing, and finishing return [`Error::FileClosed`], including with empty input
/// or an empty buffer. Rejected appends preserve previously buffered bytes, which remain
/// accessible through [`unwritten`](Self::unwritten) and [`discard`](Self::discard).
/// After an uncertain editing operation in the shared session, the editor rejects construction,
/// appending, flushing, and finishing with [`Error::AllocationPublicationUncertain`], including
/// empty operations. Rejected append and flush calls preserve previously buffered bytes.
///
/// # Filter policies
///
/// The captured [`AppendOptions`] apply to automatic writes, [`flush`](Self::flush),
/// [`finish`](Self::finish), and drop. They are not stored in the file or applied to other
/// append calls.
///
/// By default, construction rejects an existing filtered lossy partial tail, and
/// [`flush`](Self::flush) rejects a write that would leave one. A rejected partial flush
/// preserves the buffer and permits another call. [`finish`](Self::finish) can write a partial
/// tail because it consumes the appender.
///
/// [`LossyTailReencoding::Allow`] permits growing filtered lossy partial tails and repeated
/// partial flushes. Each rewrite can introduce additional quantization error in previously
/// published tail values, with the interruption behavior described by
/// [`Dataset::append_with_options`]. Under
/// [`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), partial tails stay
/// unfiltered until complete, including with a lossy pipeline.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{File, FileBuilder, MaxExtent};
/// # fn main() -> Result<(), hdf5_pure::Error> {
/// # let path = test_util::temp::temp_path("samples.h5");
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("samples")
///     .with_i32_data(&[])
///     .with_maxshape(&[MaxExtent::Unlimited])
///     .with_chunks(&[4]);
/// builder.write(&path)?;
///
/// let file = File::open_rw(&path)?;
/// let mut dataset = file.dataset("samples")?;
/// let mut appender = dataset.buffered_appender()?;
/// appender.append(&[0i32, 1, 2, 3, 4, 5])?;
/// assert_eq!(appender.buffered_elements(), 2);
/// appender.finish()?;
/// assert_eq!(dataset.read_i32()?, [0, 1, 2, 3, 4, 5]);
/// drop(dataset);
/// drop(file);
/// # Ok(())
/// # }
/// ```
pub struct BufferedAppender<'a> {
    dataset: &'a mut Dataset,
    state: BufferState,
}

impl<'a> BufferedAppender<'a> {
    /// Creates an appender whose writes use `options` for its lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReadOnly`] if the file is read-only. Other errors are described by
    /// [`Dataset::buffered_appender_with_options`].
    pub(crate) fn new(dataset: &'a mut Dataset, options: AppendOptions) -> Result<Self, Error> {
        Ok(Self {
            state: BufferState::new(dataset, options)?,
            dataset,
        })
    }

    /// Buffers scalar elements and writes complete chunks automatically.
    ///
    /// The element type must match the dataset's on-disk datatype exactly. A mismatch is
    /// reported when this call or a later operation writes the buffered elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for empty `data`, or
    /// [`Error::AppendInPlaceUnsupported`] if a write has already failed. Propagates errors
    /// from the immediate append. See [`BufferedAppender`] for recovery after an error.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn append<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        self.state.append(self.dataset, data)
    }

    /// Buffers little-endian element bytes and writes complete chunks automatically.
    ///
    /// Raw input can end within an element. Automatic writes use only the complete prefix,
    /// and a later call can complete the trailing element. [`flush`](Self::flush) and
    /// [`finish`](Self::finish) require the entire buffer to contain complete elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for empty `bytes`, or
    /// [`Error::AppendInPlaceUnsupported`] if a write has already failed. Propagates errors
    /// from the immediate append. See [`BufferedAppender`] for recovery after an error.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn append_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.state.append_raw(self.dataset, bytes)
    }

    /// Returns the number of complete elements buffered in memory.
    ///
    /// A partial trailing element from [`append_raw`](Self::append_raw) is excluded from the
    /// count. Its bytes are included in [`unwritten`](Self::unwritten).
    #[must_use]
    pub fn buffered_elements(&self) -> u64 {
        self.state.buffered_elements()
    }

    /// Returns the pending little-endian bytes in append order.
    ///
    /// After a write error, these are the bytes after the prefix written to the file. This
    /// method remains usable after an error or file closure.
    #[must_use]
    pub fn unwritten(&self) -> &[u8] {
        self.state.pending.raw()
    }

    /// Returns the chunk length along the appended axis.
    #[must_use]
    pub const fn chunk_elements(&self) -> u64 {
        self.state.chunk_elems.get()
    }

    /// Writes every buffered element to the file, including a partial trailing chunk.
    ///
    /// A partial tail is rewritten by the next write. Under
    /// [`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), it stays unfiltered
    /// until complete, even with a lossy pipeline.
    /// [`LossyTailReencoding::Allow`] permits repeated partial flushes of a filtered lossy
    /// tail, with the quantization and interruption behavior described by [`BufferedAppender`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for an empty buffer.
    /// Returns [`Error::AppendInPlaceUnsupported`] if a write has already failed, the buffer
    /// ends within an element, or a write would leave a filtered lossy partial tail and
    /// re-encoding is forbidden. Propagates errors from the immediate append.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn flush(&mut self) -> Result<(), Error> {
        self.state.flush(self.dataset, FlushMode::Continue)
    }

    /// Returns the pending bytes and consumes the appender without writing them.
    ///
    /// Elements written by earlier calls remain in the file. This method remains usable after
    /// an error or file closure.
    pub fn discard(mut self) -> Vec<u8> {
        self.state.discard()
    }

    /// Flushes every buffered element and consumes the appender.
    ///
    /// Prefer this to dropping, which performs the same flush but cannot report its error.
    /// The appender is consumed even on error. Use [`flush`](Self::flush) to retain unwritten
    /// elements for recovery through [`unwritten`](Self::unwritten).
    ///
    /// Writes a partial tail even under a lossy pipeline, since the appender is consumed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for an empty buffer.
    /// Returns [`Error::AppendInPlaceUnsupported`] if a write has already failed or the buffer
    /// ends within an element. Propagates errors from the immediate append.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn finish(mut self) -> Result<(), Error> {
        self.state.finish(self.dataset)
    }
}

impl Drop for BufferedAppender<'_> {
    fn drop(&mut self) {
        self.state.teardown(self.dataset);
    }
}

impl std::fmt::Debug for BufferedAppender<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.state.debug(f, "BufferedAppender")
    }
}

/// Owns a dataset and buffers appended elements until complete chunks can be written.
///
/// Created by [`Dataset::into_buffered_appender`] or
/// [`Dataset::into_buffered_appender_with_options`]. The dataset must be chunked, rank one,
/// unlimited along axis 0, and Extensible-Array indexed. Construction rejects SWMR sessions.
///
/// The owned dataset retains the editing session after the original [`File`](crate::File)
/// is dropped. The appender can be stored in a struct without a lifetime parameter. For an
/// appender that borrows a dataset, see [`BufferedAppender`]. Before each write, the appender
/// reads the dataset's current length from the session.
///
/// An eligible empty dataset may have an unallocated index. Construction and buffering leave
/// index and element storage unallocated until the first write, as described by [`Dataset::append`].
/// Empty appends, empty flushes, and dropping an empty appender leave that storage unallocated.
///
/// # Buffering and recovery
///
/// Automatic writes end at a chunk boundary and leave the remaining bytes buffered.
/// [`append_raw`](Self::append_raw) can buffer a partial element across calls. An explicit
/// [`flush`](Self::flush) requires complete elements and writes partial trailing chunks.
/// [`finish`](Self::finish) flushes and consumes the appender, including on error.
/// [`discard`](Self::discard) consumes it and returns the pending bytes without writing them.
/// Both terminal methods release the session claim and drop the owned dataset.
///
/// If appending fails before the write path, only the bytes submitted by that call are rejected.
/// If a write fails, the appender is poisoned, including when no elements were written.
/// Further writes are rejected. [`unwritten`](Self::unwritten) and [`discard`](Self::discard)
/// provide the bytes after the written prefix for recovery. If [`flush`](Self::flush) rejects
/// an incomplete element or a forbidden partial lossy tail, it preserves the buffer and
/// permits another call.
///
/// Dropping attempts a terminal flush unless a write has failed or a terminal operation has
/// consumed the appender. Errors during drop are ignored. Use [`finish`](Self::finish) to
/// receive a flush error, or [`flush`](Self::flush)
/// to retain access to unwritten bytes. Buffered bytes are lost on a crash. Writes use the
/// immediate append path and its [`SyncPolicy`](crate::SyncPolicy) barriers.
///
/// # Storage reuse
///
/// During automatic writes and partial or terminal flushes, the editor uses the retirement, reuse,
/// and persistence timing described by [`BufferedAppender`].
///
/// # Session state
///
/// The session rejects a second appender on the same dataset and staged edits to the
/// dataset or an ancestor with [`Error::EditUnsupported`], for either ownership form.
/// For datasets opened by path, unrelated staged edits and commits are allowed. Appender
/// writes leave those edits pending, and unrelated commits preserve the buffer and claim.
/// An appender over a dataset handle reached by object reference blocks all staged edits
/// and other appenders in the session.
///
/// [`File::close`](crate::File::close) closes the shared session without flushing live
/// appender buffers. After closure, appending, flushing, and finishing return
/// [`Error::FileClosed`], including with empty input or an empty buffer. Rejected appends
/// preserve previously buffered bytes, which remain accessible through
/// [`unwritten`](Self::unwritten) and [`discard`](Self::discard).
/// After an uncertain editing operation in the shared session, the editor rejects construction,
/// appending, flushing, and finishing with [`Error::AllocationPublicationUncertain`], including
/// empty operations. Rejected append and flush calls preserve previously buffered bytes.
///
/// # Filter policies
///
/// The captured [`AppendOptions`] apply to automatic writes, [`flush`](Self::flush),
/// [`finish`](Self::finish), and drop. They are not stored in the file or applied to other
/// append calls.
///
/// By default, construction rejects an existing filtered lossy partial tail, and
/// [`flush`](Self::flush) rejects a write that would leave one. [`finish`](Self::finish) can
/// write a partial tail because it consumes the appender. [`LossyTailReencoding::Allow`]
/// permits growing filtered lossy partial tails and repeated partial flushes. Each rewrite
/// can introduce additional quantization error in previously published tail values, with
/// the interruption behavior described by [`Dataset::append_with_options`]. Under
/// [`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), partial tails stay
/// unfiltered until complete, including with a lossy pipeline.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{File, FileBuilder, MaxExtent, OwnedBufferedAppender};
/// # fn main() -> Result<(), hdf5_pure::Error> {
/// # let path = test_util::temp::temp_path("samples.h5");
/// struct Samples {
///     appender: OwnedBufferedAppender,
/// }
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("samples")
///     .with_i32_data(&[])
///     .with_maxshape(&[MaxExtent::Unlimited])
///     .with_chunks(&[4]);
/// builder.write(&path)?;
///
/// let file = File::open_rw(&path)?;
/// let mut samples = Samples {
///     appender: file.dataset("samples")?.into_buffered_appender()?,
/// };
/// drop(file);
/// samples.appender.append(&[0i32, 1, 2, 3, 4, 5])?;
/// assert_eq!(samples.appender.buffered_elements(), 2);
/// samples.appender.finish()?;
///
/// let file = File::open(&path)?;
/// assert_eq!(file.dataset("samples")?.read_i32()?, [0, 1, 2, 3, 4, 5]);
/// drop(file);
/// # Ok(())
/// # }
/// ```
pub struct OwnedBufferedAppender {
    dataset: Dataset,
    state: BufferState,
}

impl OwnedBufferedAppender {
    /// Creates an appender that owns the dataset and captures `options` for every write.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Dataset::into_buffered_appender_with_options`].
    pub(crate) fn new(dataset: Dataset, options: AppendOptions) -> Result<Self, Error> {
        Ok(Self {
            state: BufferState::new(&dataset, options)?,
            dataset,
        })
    }

    /// Buffers scalar elements and writes complete chunks automatically.
    ///
    /// The element type must match the dataset's on-disk datatype exactly. A mismatch is
    /// reported when this call or a later operation writes the buffered elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for empty `data`, or
    /// [`Error::AppendInPlaceUnsupported`] if a write has already failed. Propagates errors
    /// from the immediate append. See [`OwnedBufferedAppender`] for recovery after an error.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn append<T: H5Element>(&mut self, data: &[T]) -> Result<(), Error> {
        self.state.append(&mut self.dataset, data)
    }

    /// Buffers little-endian element bytes and writes complete chunks automatically.
    ///
    /// Raw input can end within an element. Automatic writes use only the complete prefix,
    /// and a later call can complete the trailing element. [`flush`](Self::flush) and
    /// [`finish`](Self::finish) require the entire buffer to contain complete elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for empty `bytes`, or
    /// [`Error::AppendInPlaceUnsupported`] if a write has already failed. Propagates errors
    /// from the immediate append. See [`OwnedBufferedAppender`] for recovery after an error.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn append_raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.state.append_raw(&mut self.dataset, bytes)
    }

    /// Returns the number of complete elements buffered in memory.
    ///
    /// A partial trailing element from [`append_raw`](Self::append_raw) is excluded from the
    /// count. Its bytes are included in [`unwritten`](Self::unwritten).
    #[must_use]
    pub fn buffered_elements(&self) -> u64 {
        self.state.buffered_elements()
    }

    /// Returns the pending little-endian bytes in append order.
    ///
    /// After a write error, these are the bytes after the prefix written to the file. This
    /// method remains usable after an error or file closure.
    #[must_use]
    pub fn unwritten(&self) -> &[u8] {
        self.state.pending.raw()
    }

    /// Returns the chunk length along the appended axis.
    #[must_use]
    pub const fn chunk_elements(&self) -> u64 {
        self.state.chunk_elems.get()
    }

    /// Writes every buffered element to the file, including a partial trailing chunk.
    ///
    /// A partial tail is rewritten by the next write. Under
    /// [`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), it stays unfiltered
    /// until complete, even with a lossy pipeline.
    /// [`LossyTailReencoding::Allow`] permits repeated partial flushes of a filtered lossy
    /// tail, with the quantization and interruption behavior described by
    /// [`OwnedBufferedAppender`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for an empty buffer.
    /// Returns [`Error::AppendInPlaceUnsupported`] if a write has already failed, the buffer
    /// ends within an element, or a write would leave a filtered lossy partial tail and
    /// re-encoding is forbidden. Propagates errors from the immediate append.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn flush(&mut self) -> Result<(), Error> {
        self.state.flush(&mut self.dataset, FlushMode::Continue)
    }

    /// Returns the pending bytes and consumes the appender without writing them.
    ///
    /// Elements written by earlier calls remain in the file. This method remains usable after
    /// an error or file closure.
    pub fn discard(mut self) -> Vec<u8> {
        self.state.discard()
    }

    /// Flushes every buffered element and consumes the appender.
    ///
    /// Prefer this to dropping, which performs the same flush but cannot report its error.
    /// The appender is consumed even on error. Use [`flush`](Self::flush) to retain unwritten
    /// elements for recovery through [`unwritten`](Self::unwritten).
    ///
    /// Writes a partial tail even under a lossy pipeline, since the appender is consumed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FileClosed`] if the file is closed, including for an empty buffer.
    /// Returns [`Error::AppendInPlaceUnsupported`] if a write has already failed or the buffer
    /// ends within an element. Propagates errors from the immediate append.
    /// Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome
    /// is uncertain.
    pub fn finish(mut self) -> Result<(), Error> {
        self.state.finish(&mut self.dataset)
    }
}

impl Drop for OwnedBufferedAppender {
    fn drop(&mut self) {
        self.state.teardown(&mut self.dataset);
    }
}

impl std::fmt::Debug for OwnedBufferedAppender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.state.debug(f, "OwnedBufferedAppender")
    }
}

/// Buffers bytes under captured append geometry and policy, and owns the session claim.
struct BufferState {
    chunk_elems: NonZeroU64,
    element_size: NonZeroUsize,
    lossy_filters: bool,
    pending: AppendBuilder,
    lifecycle: Lifecycle,
    claim: u64,
    #[cfg(test)]
    failure: Option<tests::WriteFailure>,
}

impl BufferState {
    fn new(dataset: &Dataset, options: AppendOptions) -> Result<Self, Error> {
        let AppendGeometry {
            chunk_elems,
            element_size,
            current_dim,
            lossy_filters,
            full_batch_elems: _,
        } = dataset.append_geometry()?;
        if dataset.session_is_swmr() {
            return Err(Error::SwmrAppendUnsupported(
                "a SWMR writer cannot write a partial trailing chunk, which is what a buffered \
                 appender exists to do; append whole chunks with Dataset::append instead",
            ));
        }
        let chunk_elems = NonZeroU64::new(chunk_elems).ok_or(FormatError::InvalidChunkGeometry(
            "the append chunk length is zero",
        ))?;
        if lossy_filters
            && options.lossy_tail_reencoding() == LossyTailReencoding::Reject
            && current_dim % chunk_elems.get() != 0
        {
            return Err(Error::AppendInPlaceUnsupported(
                "this dataset's filter pipeline is lossy and its length is not a whole multiple \
                 of the chunk length, so a buffered appender's first write would have to \
                 re-encode already-committed values; append whole chunks from a chunk-aligned \
                 length instead",
            ));
        }
        let pending = AppendBuilder::with_options(options);
        let claim = dataset.claim_for_appender()?;
        Ok(Self {
            chunk_elems,
            element_size,
            lossy_filters,
            pending,
            lifecycle: Lifecycle::Active,
            claim,
            #[cfg(test)]
            failure: None,
        })
    }

    fn append<T: H5Element>(&mut self, dataset: &mut Dataset, data: &[T]) -> Result<(), Error> {
        self.check_usable(dataset)?;
        let mark = self.pending.raw().len();
        self.pending.append(data);
        if let Some(error) = self.pending.take_error() {
            return Err(error.into());
        }
        self.settle(dataset, mark)
    }

    fn append_raw(&mut self, dataset: &mut Dataset, bytes: &[u8]) -> Result<(), Error> {
        self.check_usable(dataset)?;
        let mark = self.pending.raw().len();
        self.pending.append_raw(bytes);
        self.settle(dataset, mark)
    }

    fn flush(&mut self, dataset: &mut Dataset, mode: FlushMode) -> Result<(), Error> {
        self.check_usable(dataset)?;
        let avail = self.whole_elements()?;
        if avail == 0 {
            return Ok(());
        }
        if self.lossy_filters
            && self.pending.options().lossy_tail_reencoding() == LossyTailReencoding::Reject
            && mode == FlushMode::Continue
        {
            let current = dataset.append_geometry()?.current_dim;
            let end = current
                .checked_add(avail)
                .ok_or(FormatError::OffsetOverflow {
                    offset: current,
                    length: avail,
                })?;
            if end % self.chunk_elems.get() != 0 {
                return Err(Error::AppendInPlaceUnsupported(
                    "this dataset's filter pipeline is lossy, and flushing these elements would \
                     leave its length short of a chunk boundary — a later write would then have \
                     to re-encode already-committed values; buffer until the elements complete \
                     a chunk, or end the appender with BufferedAppender::finish, which writes \
                     the partial tail",
                ));
            }
        }
        self.write_prefix(dataset, avail)
    }

    fn discard(&mut self) -> Vec<u8> {
        self.lifecycle = Lifecycle::Stopped;
        std::mem::replace(&mut self.pending, AppendBuilder::new()).into_raw()
    }

    fn finish(&mut self, dataset: &mut Dataset) -> Result<(), Error> {
        let result = self.flush(dataset, FlushMode::Finish);
        self.lifecycle = Lifecycle::Stopped;
        result
    }

    /// Attempts the terminal flush of an active buffer and releases its session claim.
    fn teardown(&mut self, dataset: &mut Dataset) {
        if self.lifecycle == Lifecycle::Active && !self.pending.raw().is_empty() {
            self.flush(dataset, FlushMode::Finish).ok();
        }
        self.lifecycle = Lifecycle::Stopped;
        dataset.release_appender_claim(self.claim);
    }

    fn buffered_elements(&self) -> u64 {
        (self.pending.raw().len() / self.element_size) as u64
    }

    /// Writes complete chunks, rolling back new input on errors before entering the write path.
    ///
    /// # Errors
    ///
    /// Propagates errors from reading the current append geometry, converting the write prefix
    /// length, and the immediate append.
    fn settle(&mut self, dataset: &mut Dataset, mark: usize) -> Result<(), Error> {
        let result = self.write_complete_chunks(dataset);
        if result.is_err() && self.lifecycle == Lifecycle::Active {
            self.pending.truncate(mark);
        }
        result
    }

    fn check_usable(&self, dataset: &Dataset) -> Result<(), Error> {
        dataset.check_appender_mutable()?;
        if self.lifecycle == Lifecycle::Poisoned {
            return Err(Error::AppendInPlaceUnsupported(
                "this buffered appender failed a write and holds unwritten elements; read them \
                 back with BufferedAppender::unwritten and start a new appender",
            ));
        }
        Ok(())
    }

    fn whole_elements(&self) -> Result<u64, Error> {
        if self.pending.raw().len() % self.element_size != 0 {
            return Err(Error::AppendInPlaceUnsupported(
                "the buffered byte length is not a whole number of elements; append the rest of \
                 the trailing element before flushing",
            ));
        }
        Ok(self.buffered_elements())
    }

    fn write_complete_chunks(&mut self, dataset: &mut Dataset) -> Result<(), Error> {
        let avail = self.buffered_elements();
        let chunk_elems = self.chunk_elems.get();
        if avail < chunk_elems {
            return Ok(());
        }
        let partial = dataset.append_geometry()?.current_dim % chunk_elems;
        let to_boundary = (chunk_elems - partial) % chunk_elems;
        debug_assert!(to_boundary < chunk_elems && to_boundary <= avail);
        let take = to_boundary + ((avail - to_boundary) / chunk_elems) * chunk_elems;
        debug_assert!((1..=avail).contains(&take));
        self.write_prefix(dataset, take)
    }

    fn write_prefix(&mut self, dataset: &mut Dataset, take_elems: u64) -> Result<(), Error> {
        let before = dataset.append_geometry()?.current_dim;
        let head = self.pending.head(self.byte_len(take_elems)?);
        #[cfg(not(test))]
        let result = dataset.append_prebuilt(&head);
        #[cfg(test)]
        let result = match &self.failure {
            Some(failure) => failure.append(dataset, &head, self.element_size),
            None => dataset.append_prebuilt(&head),
        };
        let landed = if result.is_ok() {
            take_elems
        } else {
            self.lifecycle = Lifecycle::Poisoned;
            dataset.append_geometry().map_or(0, |geometry| {
                geometry.current_dim.saturating_sub(before).min(take_elems)
            })
        };
        self.pending.drop_front(self.byte_len(landed)?);
        result
    }

    fn byte_len(&self, elements: u64) -> Result<usize, Error> {
        Ok(elements
            .checked_mul(self.element_size.get() as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: elements,
                length: self.element_size.get() as u64,
            })?
            .to_usize()?)
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>, name: &str) -> std::fmt::Result {
        f.debug_struct(name)
            .field("chunk_elements", &self.chunk_elems)
            .field("buffered_elements", &self.buffered_elements())
            .field("lossy_filters", &self.lossy_filters)
            .field("poisoned", &(self.lifecycle == Lifecycle::Poisoned))
            .finish()
    }
}

/// Whether the buffer can attempt another write.
#[derive(Eq, PartialEq)]
enum Lifecycle {
    Active,
    /// A write failed. The pending bytes remain available for recovery.
    Poisoned,
    /// A terminal operation consumed the buffer, suppressing a drop-time flush.
    Stopped,
}

/// Whether another write can follow a flush under the captured lossy-tail policy.
#[derive(Eq, PartialEq)]
enum FlushMode {
    Continue,
    /// Permits a partial tail because the appender is consumed after the flush.
    Finish,
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroUsize;
    use std::cell::Cell;
    use std::io::{Error as IoError, ErrorKind};
    use std::rc::Rc;

    use rstest::rstest;
    use test_util::disk_log;
    use test_util_hdf5::dataset::{self, Unlimited};

    use crate::edit::AppendBuilder;
    use crate::{Dataset, Error, File, FileAccessProperties, FileBuilder, MaxExtent, SyncPolicy};

    #[rstest]
    #[case::borrowed_empty(false, false)]
    #[case::borrowed_pending(false, true)]
    #[case::owned_empty(true, false)]
    #[case::owned_pending(true, true)]
    fn session_uncertainty_rejects_buffered_input_and_empty_flushes(
        #[case] owned: bool,
        #[case] pending: bool,
        #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("uncertain-appender.h5");
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("d")
            .with_i32_data(&[0, 1, 2, 3])
            .with_chunks(&[4])
            .with_maxshape(&[MaxExtent::Unlimited]);
        builder
            .create_dataset("other")
            .with_i32_data(&[0, 1, 2, 3])
            .with_chunks(&[4])
            .with_maxshape(&[MaxExtent::Unlimited]);
        builder.write(&path).unwrap();
        let before = std::fs::read(&path).unwrap();
        let file =
            File::open_rw_with_options(&path, FileAccessProperties::new().with_sync_policy(policy))
                .unwrap();
        let expected = if pending {
            7i32.to_le_bytes().to_vec()
        } else {
            Vec::new()
        };
        if owned {
            let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
            if pending {
                appender.append(&[7i32]).unwrap();
            }
            fail_other_append(&file);
            assert_uncertain(appender.append::<i32>(&[]).unwrap_err());
            assert_uncertain(appender.append_raw(&[]).unwrap_err());
            assert_uncertain(appender.append(&[8i32]).unwrap_err());
            assert_uncertain(appender.flush().unwrap_err());
            assert_eq!(appender.unwritten(), expected);
            assert_uncertain(appender.finish().unwrap_err());
            assert!(matches!(
                file.dataset("d").unwrap().into_buffered_appender(),
                Err(Error::AllocationPublicationUncertain)
            ));
        } else {
            let mut dataset = file.dataset("d").unwrap();
            let mut appender = dataset.buffered_appender().unwrap();
            if pending {
                appender.append(&[7i32]).unwrap();
            }
            fail_other_append(&file);
            assert_uncertain(appender.append::<i32>(&[]).unwrap_err());
            assert_uncertain(appender.append_raw(&[]).unwrap_err());
            assert_uncertain(appender.append(&[8i32]).unwrap_err());
            assert_uncertain(appender.flush().unwrap_err());
            assert_eq!(appender.unwritten(), expected);
            assert_uncertain(appender.finish().unwrap_err());
            assert!(matches!(
                dataset.buffered_appender(),
                Err(Error::AllocationPublicationUncertain)
            ));
            drop(dataset);
        }
        disk_log::start();
        assert_uncertain(file.sync().unwrap_err());
        assert_uncertain(file.clone().close().unwrap_err());
        drop(file);
        assert!(disk_log::take().is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    fn fail_other_append(file: &File) {
        let mut other = file.dataset("other").unwrap();
        disk_log::fail_write_after(0, 0);
        let error = other.append(&[4i32, 5, 6, 7]).unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected injected append failure, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
    }

    fn assert_uncertain(error: Error) {
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::before_any_elements(0)]
    #[case::after_a_complete_chunk(4)]
    fn write_failure_retains_exact_remaining_bytes_and_prevents_retries(#[case] landed: usize) {
        let path = test_util::temp::temp_path("failure.h5");
        Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let mut appender = dataset.buffered_appender().unwrap();
        let attempts = Rc::new(Cell::new(0));
        appender.state.failure = Some(WriteFailure {
            landed,
            attempts: attempts.clone(),
        });
        let bytes: Vec<_> = (0..10i32).flat_map(i32::to_le_bytes).collect();
        appender.append_raw(&bytes[..5]).unwrap();
        let error = appender.append_raw(&bytes[5..]).unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected injected write error, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected appender write failure");
        assert_eq!(attempts.get(), 1);
        assert_eq!(appender.unwritten(), &bytes[landed * 4..]);
        assert_eq!(appender.buffered_elements(), (10 - landed) as u64);
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            (0..landed as i32).collect::<Vec<_>>()
        );
        assert_poisoned(appender.append(&[10i32]).unwrap_err());
        assert_poisoned(appender.append_raw(&[0]).unwrap_err());
        assert_poisoned(appender.flush().unwrap_err());
        assert_eq!(appender.unwritten(), &bytes[landed * 4..]);
        assert_poisoned(appender.finish().unwrap_err());
        assert_eq!(attempts.get(), 1);
        let mut replacement = file.dataset("d").unwrap();
        replacement.buffered_appender().unwrap().finish().unwrap();
        drop(replacement);
        drop(dataset);
        drop(file);
        assert_eq!(
            dataset::read_pure::<i32>(&path, "d"),
            (0..landed as i32).collect::<Vec<_>>()
        );
    }

    #[rstest]
    #[case::discard(true)]
    #[case::drop(false)]
    fn failed_flush_never_retries_during_teardown(#[case] discard: bool) {
        let path = test_util::temp::temp_path("flush.h5");
        Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let mut appender = dataset.buffered_appender().unwrap();
        let attempts = Rc::new(Cell::new(0));
        appender.state.failure = Some(WriteFailure {
            landed: 0,
            attempts: attempts.clone(),
        });
        let bytes = 7i32.to_le_bytes();
        appender.append_raw(&bytes).unwrap();
        let error = appender.flush().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected injected write error, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected appender write failure");
        assert_eq!(appender.unwritten(), bytes);
        if discard {
            assert_eq!(appender.discard(), bytes);
        } else {
            drop(appender);
        }
        assert_eq!(attempts.get(), 1);
        dataset.buffered_appender().unwrap().finish().unwrap();
        drop(dataset);
        drop(file);
        assert_eq!(dataset::read_pure::<i32>(&path, "d"), Vec::<i32>::new());
    }

    #[rstest]
    #[case::finish(true)]
    #[case::drop(false)]
    fn terminal_write_failure_releases_the_claim_without_retrying(#[case] finish: bool) {
        let path = test_util::temp::temp_path("terminal.h5");
        Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let mut appender = dataset.buffered_appender().unwrap();
        let attempts = Rc::new(Cell::new(0));
        appender.state.failure = Some(WriteFailure {
            landed: 1,
            attempts: attempts.clone(),
        });
        appender.append(&[7i32, 8]).unwrap();
        if finish {
            let error = appender.finish().unwrap_err();
            let Error::Io(error) = error else {
                panic!("expected injected write error, got {error:?}");
            };
            assert_eq!(error.kind(), ErrorKind::Other);
            assert_eq!(error.to_string(), "injected appender write failure");
        } else {
            drop(appender);
        }
        assert_eq!(attempts.get(), 1);
        assert_eq!(dataset.read_i32().unwrap(), [7]);
        dataset.buffered_appender().unwrap().finish().unwrap();
        drop(dataset);
        drop(file);
        assert_eq!(dataset::read_pure::<i32>(&path, "d"), [7]);
    }

    #[rstest]
    #[case::before_any_elements(0)]
    #[case::after_a_complete_chunk(4)]
    fn owned_write_failure_preserves_the_unwritten_tail_without_retrying(#[case] landed: usize) {
        let path = test_util::temp::temp_path("owned_failure.h5");
        Unlimited::<i32>::new("d", &[], 4).pure_create(&path);
        let file = File::open_rw(&path).unwrap();
        let mut appender = file.dataset("d").unwrap().into_buffered_appender().unwrap();
        let attempts = Rc::new(Cell::new(0));
        appender.state.failure = Some(WriteFailure {
            landed,
            attempts: attempts.clone(),
        });
        let bytes: Vec<_> = (0..10i32).flat_map(i32::to_le_bytes).collect();
        appender.append_raw(&bytes[..5]).unwrap();
        let error = appender.append_raw(&bytes[5..]).unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected injected write error, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected appender write failure");
        assert_eq!(appender.unwritten(), &bytes[landed * 4..]);
        assert_poisoned(appender.flush().unwrap_err());
        assert_eq!(appender.discard(), &bytes[landed * 4..]);
        assert_eq!(attempts.get(), 1);
        file.dataset("d")
            .unwrap()
            .into_buffered_appender()
            .unwrap()
            .finish()
            .unwrap();
        drop(file);
        assert_eq!(
            dataset::read_pure::<i32>(&path, "d"),
            (0..landed as i32).collect::<Vec<_>>()
        );
    }

    fn assert_poisoned(error: Error) {
        let Error::AppendInPlaceUnsupported(reason) = error else {
            panic!("expected poisoned appender error, got {error:?}");
        };
        assert_eq!(
            reason,
            "this buffered appender failed a write and holds unwritten elements; read them \
                     back with BufferedAppender::unwritten and start a new appender"
        );
    }

    pub(super) struct WriteFailure {
        landed: usize,
        attempts: Rc<Cell<usize>>,
    }

    impl WriteFailure {
        pub(super) fn append(
            &self,
            dataset: &mut Dataset,
            head: &AppendBuilder,
            element_size: NonZeroUsize,
        ) -> Result<(), Error> {
            self.attempts.set(self.attempts.get() + 1);
            if self.landed != 0 {
                dataset.append_prebuilt(&head.head(self.landed * element_size.get()))?;
            }
            Err(IoError::other("injected appender write failure").into())
        }
    }
}
