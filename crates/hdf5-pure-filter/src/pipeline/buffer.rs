//! Borrowed and owned storage for filter execution.

use alloc::vec::Vec;
use core::mem;
use core::ops::Range;

use hdf5_pure_core::FormatError;

use crate::error::Error;

/// A borrowed input or an owned allocation with a checked live byte range.
///
/// Views into owned storage retain the allocation and initialized extent. The constructors and
/// consuming transitions keep the live range within the initialized bytes.
pub(crate) struct FilterBuffer<'a>(Storage<'a>);

impl<'a> FilterBuffer<'a> {
    /// Borrows the input without acquiring storage.
    #[inline]
    pub(crate) fn borrowed(bytes: &'a [u8]) -> Self {
        Self(Storage::Borrowed(bytes))
    }

    /// Replaces the live bytes and recycles the displaced owned allocation.
    #[inline]
    pub(crate) fn replace(self, spare: &mut InitializedBuffer, bytes: Vec<u8>) -> Self {
        self.recycle(spare);
        Self::owned(bytes)
    }

    /// Writes the live input into the spare allocation and recycles displaced owned storage.
    ///
    /// The writer starts with empty logical output over retained initialized storage. On failure,
    /// that storage is retained in `spare` and the writer's error is returned.
    ///
    /// # Errors
    ///
    /// Returns the writer's error if writing fails.
    #[inline]
    pub(super) fn materialize(
        self,
        spare: &mut InitializedBuffer,
        write: impl FnOnce(&[u8], &mut AppendBuffer) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let mut output = AppendBuffer::new(mem::take(spare));
        if let Err(error) = write(self.as_slice(), &mut output) {
            *spare = output.bytes;
            return Err(error);
        }
        self.recycle(spare);
        Ok(output.finish())
    }

    /// Writes into a bounded initialized window and recycles displaced owned storage.
    ///
    /// The writer sets its writable length and returns the produced prefix length. On failure,
    /// initialized storage is retained in `spare`.
    ///
    /// # Errors
    ///
    /// Returns the writer's error if writing fails. Returns [`Error::Pipeline`] if the produced
    /// prefix exceeds the writable window.
    #[inline]
    pub(super) fn overwrite(
        self,
        spare: &mut InitializedBuffer,
        write: impl FnOnce(&[u8], &mut OutputWindow) -> Result<usize, Error>,
    ) -> Result<Self, Error> {
        let mut output = OutputWindow {
            bytes: mem::take(spare),
            end: 0,
        };
        let produced = match write(self.as_slice(), &mut output) {
            Ok(produced) => produced,
            Err(error) => {
                *spare = output.bytes;
                return Err(error);
            }
        };
        match output.finish(produced) {
            Ok(output) => {
                self.recycle(spare);
                Ok(output)
            }
            Err(output) => {
                *spare = output.bytes;
                Err(Error::Pipeline(FormatError::Internal(
                    "filter output count exceeds its writable window".into(),
                )))
            }
        }
    }

    /// Restricts the view to a range relative to the live bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pipeline`] if `range` extends beyond the live bytes or is reversed.
    #[inline]
    pub(crate) fn narrow(self, range: Range<usize>) -> Result<Self, Error> {
        if self.as_slice().get(range.clone()).is_none() {
            return Err(Error::Pipeline(FormatError::Internal(
                "filter view extends beyond its input".into(),
            )));
        }
        Ok(Self(match self.0 {
            Storage::Borrowed(bytes) => Storage::Borrowed(&bytes[range]),
            Storage::Owned { bytes, live } => Storage::Owned {
                bytes,
                live: live.start + range.start..live.start + range.end,
            },
        }))
    }

    /// Appends to owned live bytes, copying borrowed input into the spare allocation first.
    #[inline]
    pub(super) fn append(self, spare: &mut InitializedBuffer, suffix: &[u8]) -> Self {
        let mut output = match self.0 {
            Storage::Borrowed(input) => {
                let mut output = AppendBuffer::new(mem::take(spare));
                output.reserve_exact(input.len().saturating_add(suffix.len()));
                output.extend_from_slice(input);
                output
            }
            Storage::Owned { bytes, live } => AppendBuffer::from_live(bytes, live),
        };
        output.extend_from_slice(suffix);
        output.finish()
    }

    /// Replaces the spare with the owned initialized allocation, if present.
    #[inline]
    pub(super) fn recycle(self, spare: &mut InitializedBuffer) {
        if let Storage::Owned { bytes, .. } = self.0 {
            *spare = bytes;
        }
    }

    /// Returns only the live bytes, copying borrowed input and compacting owned views in place.
    #[inline]
    pub(crate) fn into_owned(self) -> Vec<u8> {
        match self.0 {
            Storage::Borrowed(bytes) => bytes.to_vec(),
            Storage::Owned { bytes, live } => AppendBuffer::from_live(bytes, live).into_owned(),
        }
    }

    /// Returns only the live bytes.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match &self.0 {
            Storage::Borrowed(bytes) => bytes,
            Storage::Owned { bytes, live } => &bytes.0[live.clone()],
        }
    }

    #[inline]
    fn owned(bytes: Vec<u8>) -> Self {
        let live = 0..bytes.len();
        Self(Storage::Owned {
            bytes: InitializedBuffer::from_vec(bytes),
            live,
        })
    }
}

enum Storage<'a> {
    Borrowed(&'a [u8]),
    Owned {
        bytes: InitializedBuffer,
        live: Range<usize>,
    },
}

/// Initialized byte storage retained independently of live output.
///
/// The vector length records the initialized extent. Filter views and writers track their live
/// bytes separately.
#[derive(Default)]
pub(crate) struct InitializedBuffer(Vec<u8>);

impl InitializedBuffer {
    /// Preserves the vector's initialized extent and allocation.
    #[inline]
    pub(crate) fn from_vec(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.0.capacity()
    }

    #[cfg(test)]
    pub(crate) fn initialized_len(&self) -> usize {
        self.0.len()
    }

    #[cfg(test)]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

/// A writable prefix of initialized storage for one stage.
///
/// The writable length is independent of retained storage. Shrinking the window preserves the
/// initialized extent. The stage returns the produced prefix length.
pub(super) struct OutputWindow {
    bytes: InitializedBuffer,
    end: usize,
}

impl OutputWindow {
    /// Sets the writable prefix length, zeroing only newly initialized bytes.
    #[inline]
    pub(super) fn resize(&mut self, end: usize) {
        let Self {
            bytes,
            end: window_end,
        } = self;
        if end > bytes.0.len() {
            bytes.0.resize(end, 0);
        }
        *window_end = end;
    }

    /// Returns the initialized writable prefix.
    #[inline]
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        let Self { bytes, end } = self;
        &mut bytes.0[..*end]
    }

    /// Returns the writable bound for this operation.
    #[inline]
    #[cfg(any(feature = "deflate", test))]
    pub(super) fn len(&self) -> usize {
        self.end
    }

    /// Returns a live view of the produced prefix.
    ///
    /// # Errors
    ///
    /// Returns this window if `produced` exceeds its writable length.
    #[inline]
    fn finish<'a>(self, produced: usize) -> Result<FilterBuffer<'a>, Self> {
        let Self { bytes, end } = self;
        if produced > end {
            return Err(Self { bytes, end });
        }
        Ok(FilterBuffer(Storage::Owned {
            bytes,
            live: 0..produced,
        }))
    }
}

/// An append writer with a live prefix separate from its initialized storage.
///
/// Writes reuse initialized bytes before extending the vector. The initialized extent is retained
/// when logical output is cleared.
#[derive(Default)]
pub(crate) struct AppendBuffer {
    bytes: InitializedBuffer,
    written: usize,
}

impl AppendBuffer {
    #[inline]
    fn new(bytes: InitializedBuffer) -> Self {
        Self { bytes, written: 0 }
    }

    /// Moves the live bytes to the start of initialized storage.
    ///
    /// The caller supplies a range within the initialized bytes. Bytes beyond the written prefix
    /// remain initialized.
    #[inline]
    fn from_live(mut bytes: InitializedBuffer, live: Range<usize>) -> Self {
        if live.start != 0 {
            bytes.0.copy_within(live.clone(), 0);
        }
        Self {
            bytes,
            written: live.len(),
        }
    }

    /// Clears live output while retaining initialized bytes and capacity.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.written = 0;
    }

    /// Reserves room for `additional` bytes after the written prefix.
    ///
    /// The live and initialized extents are unchanged.
    #[inline]
    pub(crate) fn reserve_exact(&mut self, additional: usize) {
        let Self { bytes, written } = self;
        let required = *written + additional;
        if required > bytes.0.len() {
            bytes.0.reserve_exact(required - bytes.0.len());
        }
    }

    /// Appends one byte, overwriting retained initialized storage when available.
    #[inline]
    pub(crate) fn push(&mut self, byte: u8) {
        let Self { bytes, written } = self;
        if let Some(slot) = bytes.0.get_mut(*written) {
            *slot = byte;
        } else {
            bytes.0.push(byte);
        }
        *written += 1;
    }

    /// Appends `bytes`, overwriting retained initialized storage when available.
    #[inline]
    pub(crate) fn extend_from_slice(&mut self, bytes: &[u8]) {
        let Self {
            bytes: storage,
            written,
        } = self;
        let retained = (storage.0.len() - *written).min(bytes.len());
        storage.0[*written..*written + retained].copy_from_slice(&bytes[..retained]);
        storage.0.extend_from_slice(&bytes[retained..]);
        *written += bytes.len();
    }

    /// Appends a copy of `range` from the written prefix.
    ///
    /// The caller supplies a range within the bytes already written.
    #[inline]
    pub(crate) fn extend_from_within(&mut self, range: Range<usize>) {
        let Self { bytes, written } = self;
        let retained = (bytes.0.len() - *written).min(range.len());
        bytes
            .0
            .copy_within(range.start..range.start + retained, *written);
        bytes
            .0
            .extend_from_within(range.start + retained..range.end);
        *written += range.len();
    }

    /// Returns the written prefix as an owned vector.
    #[inline]
    pub(crate) fn into_owned(self) -> Vec<u8> {
        let Self { bytes, written } = self;
        let InitializedBuffer(mut bytes) = bytes;
        bytes.truncate(written);
        bytes
    }

    /// Returns only the written prefix.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        let Self { bytes, written } = self;
        &bytes.0[..*written]
    }

    /// Returns the length of the written prefix.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.written
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    #[cfg(test)]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    #[inline]
    fn finish<'a>(self) -> FilterBuffer<'a> {
        let Self { bytes, written } = self;
        FilterBuffer(Storage::Owned {
            bytes,
            live: 0..written,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::borrowed(false)]
    #[case::owned(true)]
    fn nested_views_expose_only_live_bytes(#[case] owned: bool) {
        let bytes = [0, 1, 2, 3, 4, 5, 6, 7];
        let buffer = if owned {
            FilterBuffer::owned(bytes.to_vec())
        } else {
            FilterBuffer::borrowed(&bytes)
        };
        let pointer = buffer.as_slice().as_ptr();
        let buffer = buffer.narrow(1..7).unwrap().narrow(2..4).unwrap();
        assert_eq!(buffer.as_slice(), [3, 4]);
        assert_eq!(buffer.as_slice().as_ptr(), pointer.wrapping_add(3));
        let output = buffer.into_owned();
        assert_eq!(output, [3, 4]);
        if owned {
            assert_eq!(output.as_ptr(), pointer);
        }
        assert_eq!(bytes, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[rstest]
    #[case::reversed(Range { start: 3, end: 2 })]
    #[case::past_end(0..5)]
    #[case::overflowing_start(usize::MAX..usize::MAX)]
    fn invalid_views_are_rejected(#[case] range: Range<usize>) {
        let error = FilterBuffer::owned(vec![1, 2, 3, 4])
            .narrow(range)
            .err()
            .unwrap();
        assert_eq!(
            error,
            Error::Pipeline(FormatError::Internal(
                "filter view extends beyond its input".into()
            ))
        );
    }

    #[rstest]
    #[case::borrowed(false)]
    #[case::owned(true)]
    fn append_uses_the_live_view_and_retains_storage(#[case] owned: bool) {
        let bytes = [0, 1, 2, 3, 4, 5];
        let mut spare = InitializedBuffer::from_vec(Vec::with_capacity(32));
        let buffer = if owned {
            let mut input = Vec::with_capacity(32);
            input.extend_from_slice(&bytes);
            FilterBuffer::owned(input)
        } else {
            FilterBuffer::borrowed(&bytes)
        };
        let pointer = if owned {
            buffer.as_slice().as_ptr()
        } else {
            spare.as_ptr()
        };
        let buffer = buffer.narrow(2..4).unwrap().append(&mut spare, &[9, 10]);
        assert_eq!(buffer.as_slice(), [2, 3, 9, 10]);
        let output = buffer.into_owned();
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output, [2, 3, 9, 10]);
    }

    #[rstest]
    #[case::borrowed(false)]
    #[case::owned(true)]
    fn materialization_reuses_the_spare_and_recycles_displaced_storage(#[case] owned: bool) {
        let bytes = [1, 2, 3, 4];
        let mut spare = InitializedBuffer::from_vec(Vec::with_capacity(32));
        let spare_pointer = spare.as_ptr();
        let buffer = if owned {
            FilterBuffer::owned(bytes.to_vec())
        } else {
            FilterBuffer::borrowed(&bytes)
        };
        let input_pointer = buffer.as_slice().as_ptr();
        let buffer = buffer
            .materialize(&mut spare, |input, output| {
                assert_eq!(output.len(), 0);
                input.iter().rev().for_each(|&byte| output.push(byte));
                Ok(())
            })
            .unwrap();
        assert_eq!(buffer.as_slice(), [4, 3, 2, 1]);
        assert_eq!(buffer.as_slice().as_ptr(), spare_pointer);
        assert_eq!(spare.initialized_len(), if owned { bytes.len() } else { 0 });
        if owned {
            assert_eq!(spare.as_ptr(), input_pointer);
            assert!(spare.capacity() >= bytes.len());
        } else {
            assert_eq!(spare.capacity(), 0);
        }
        assert_eq!(bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn overwrite_preserves_initialized_bytes_through_smaller_windows() {
        let mut spare = InitializedBuffer::from_vec(vec![0xa5; 16]);
        let buffer = FilterBuffer::borrowed(&[])
            .overwrite(&mut spare, |_, output| {
                assert_eq!(output.len(), 0);
                output.resize(6);
                assert_eq!(output.as_mut_slice(), [0xa5; 6]);
                output.as_mut_slice().fill(1);
                output.resize(3);
                assert_eq!(output.as_mut_slice(), [1; 3]);
                output.resize(16);
                assert_eq!(&output.as_mut_slice()[..6], [1; 6]);
                assert_eq!(&output.as_mut_slice()[6..], [0xa5; 10]);
                output.resize(20);
                assert_eq!(&output.as_mut_slice()[16..], [0; 4]);
                output.as_mut_slice()[..2].copy_from_slice(&[7, 8]);
                Ok(2)
            })
            .unwrap();
        assert_eq!(buffer.as_slice(), [7, 8]);
        let pointer = buffer.as_slice().as_ptr();
        buffer.recycle(&mut spare);
        assert_eq!(spare.initialized_len(), 20);
        assert_eq!(spare.as_ptr(), pointer);
        let output = FilterBuffer::borrowed(&[])
            .overwrite(&mut spare, |_, output| {
                output.resize(1);
                assert_eq!(output.as_mut_slice(), [7]);
                output.as_mut_slice()[0] = 9;
                Ok(1)
            })
            .unwrap()
            .into_owned();
        assert_eq!(output, [9]);
        assert_eq!(output.as_ptr(), pointer);
    }

    #[rstest]
    #[case::writer_error(None)]
    #[case::past_window(Some(5))]
    #[case::within_retained_storage(Some(12))]
    fn failed_overwrites_retain_initialized_storage_without_live_output(
        #[case] produced: Option<usize>,
    ) {
        let mut spare = InitializedBuffer::from_vec(vec![0xa5; 16]);
        let pointer = spare.as_ptr();
        let error = FilterBuffer::borrowed(&[])
            .overwrite(&mut spare, |_, output| {
                output.resize(4);
                output.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
                produced.ok_or(Error::UnsupportedFilter(40000))
            })
            .err()
            .unwrap();
        assert_eq!(
            error,
            if produced.is_some() {
                Error::Pipeline(FormatError::Internal(
                    "filter output count exceeds its writable window".into(),
                ))
            } else {
                Error::UnsupportedFilter(40000)
            }
        );
        assert_eq!(spare.initialized_len(), 16);
        assert_eq!(spare.as_ptr(), pointer);
        let output = FilterBuffer::borrowed(&[])
            .overwrite(&mut spare, |_, output| {
                output.resize(2);
                assert_eq!(output.as_mut_slice(), [1, 2]);
                output.as_mut_slice().copy_from_slice(&[5, 6]);
                Ok(2)
            })
            .unwrap()
            .into_owned();
        assert_eq!(output, [5, 6]);
        assert_eq!(output.as_ptr(), pointer);
    }

    #[rstest]
    #[case::new_storage(0)]
    #[case::partly_initialized(6)]
    #[case::fully_initialized(64)]
    fn append_writers_begin_empty_and_preserve_the_initialized_extent(#[case] initialized: usize) {
        let mut spare = InitializedBuffer::from_vec(vec![0xa5; initialized]);
        let buffer = FilterBuffer::borrowed(b"abcd")
            .materialize(&mut spare, |input, output| {
                assert_eq!(output.as_slice(), []);
                output.extend_from_slice(input);
                output.extend_from_within(0..4);
                output.push(b'e');
                Ok(())
            })
            .unwrap();
        assert_eq!(buffer.as_slice(), b"abcdabcde");
        buffer.recycle(&mut spare);
        assert_eq!(spare.initialized_len(), initialized.max(9));
        let output = FilterBuffer::borrowed(b"z")
            .materialize(&mut spare, |input, output| {
                assert_eq!(output.len(), 0);
                output.extend_from_slice(input);
                Ok(())
            })
            .unwrap();
        assert_eq!(output.as_slice(), b"z");
        output.recycle(&mut spare);
        assert_eq!(spare.initialized_len(), initialized.max(9));
    }

    #[test]
    fn appending_to_a_live_view_preserves_initialized_storage_and_normalizes_the_result() {
        let mut spare = InitializedBuffer::from_vec(vec![0xa5; 32]);
        let pointer = spare.as_ptr();
        let buffer = FilterBuffer::borrowed(&[])
            .overwrite(&mut spare, |_, output| {
                output.resize(8);
                output
                    .as_mut_slice()
                    .copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
                Ok(8)
            })
            .unwrap()
            .narrow(2..4)
            .unwrap()
            .append(&mut spare, &[9, 10]);
        assert_eq!(buffer.as_slice(), [2, 3, 9, 10]);
        buffer.recycle(&mut spare);
        assert_eq!(spare.initialized_len(), 32);
        let output = FilterBuffer::borrowed(&[11])
            .append(&mut spare, &[12])
            .into_owned();
        assert_eq!(output, [11, 12]);
        assert_eq!(output.as_ptr(), pointer);
    }

    #[test]
    fn failed_materialization_retains_capacity_without_partial_output() {
        let mut spare = InitializedBuffer::from_vec(Vec::with_capacity(32));
        let pointer = spare.as_ptr();
        let error = FilterBuffer::borrowed(&[1, 2, 3, 4])
            .materialize(&mut spare, |_, output| {
                output.extend_from_slice(&[7, 8]);
                Err(Error::UnsupportedFilter(40000))
            })
            .err()
            .unwrap();
        assert_eq!(error, Error::UnsupportedFilter(40000));
        assert_eq!(spare.initialized_len(), 2);
        assert_eq!(spare.as_ptr(), pointer);
        let output = FilterBuffer::borrowed(&[5, 6])
            .materialize(&mut spare, |input, output| {
                assert_eq!(output.len(), 0);
                output.extend_from_slice(input);
                Ok(())
            })
            .unwrap()
            .into_owned();
        assert_eq!(output, [5, 6]);
        assert_eq!(output.as_ptr(), pointer);
    }
}
