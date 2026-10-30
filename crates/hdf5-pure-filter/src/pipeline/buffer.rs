use alloc::vec::Vec;
use core::mem;
use core::ops::Range;

use hdf5_pure_core::FormatError;

use crate::error::Error;

/// A borrowed input or an owned allocation with a checked live byte range.
///
/// Views into owned storage retain the allocation. The constructors and consuming transitions
/// keep the live range within the initialized bytes.
pub(super) struct FilterBuffer<'a>(Storage<'a>);

impl<'a> FilterBuffer<'a> {
    /// Borrows the input without acquiring storage.
    pub(super) fn borrowed(bytes: &'a [u8]) -> Self {
        Self(Storage::Borrowed(bytes))
    }

    /// Replaces the live bytes and recycles the displaced owned allocation.
    pub(super) fn replace(self, spare: &mut Vec<u8>, bytes: Vec<u8>) -> Self {
        self.recycle(spare);
        Self::owned(bytes)
    }

    /// Writes the live input into the spare allocation and recycles displaced owned storage.
    ///
    /// The writer receives an empty output vector. On failure, the output is cleared and returned
    /// to `spare`.
    ///
    /// # Errors
    ///
    /// Returns the writer's error if writing fails.
    pub(super) fn materialize(
        self,
        spare: &mut Vec<u8>,
        write: impl FnOnce(&[u8], &mut Vec<u8>) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let mut output = mem::take(spare);
        output.clear();
        if let Err(error) = write(self.as_slice(), &mut output) {
            output.clear();
            *spare = output;
            return Err(error);
        }
        Ok(self.replace(spare, output))
    }

    /// Restricts the view to a range relative to the live bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pipeline`] if `range` extends beyond the live bytes or is reversed.
    pub(super) fn narrow(self, range: Range<usize>) -> Result<Self, Error> {
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
    pub(super) fn append(self, spare: &mut Vec<u8>, suffix: &[u8]) -> Self {
        let mut bytes = match self.0 {
            Storage::Borrowed(input) => {
                let mut bytes = mem::take(spare);
                bytes.clear();
                bytes.reserve(input.len().saturating_add(suffix.len()));
                bytes.extend_from_slice(input);
                bytes
            }
            Storage::Owned { bytes, live } => Self::normalize(bytes, live),
        };
        bytes.extend_from_slice(suffix);
        Self::owned(bytes)
    }

    /// Replaces the spare with the cleared owned allocation, if present.
    pub(super) fn recycle(self, spare: &mut Vec<u8>) {
        if let Storage::Owned { mut bytes, .. } = self.0 {
            bytes.clear();
            *spare = bytes;
        }
    }

    /// Returns only the live bytes, copying borrowed input and compacting owned views in place.
    pub(super) fn into_owned(self) -> Vec<u8> {
        match self.0 {
            Storage::Borrowed(bytes) => bytes.to_vec(),
            Storage::Owned { bytes, live } => Self::normalize(bytes, live),
        }
    }

    /// Returns only the live bytes.
    pub(super) fn as_slice(&self) -> &[u8] {
        match &self.0 {
            Storage::Borrowed(bytes) => bytes,
            Storage::Owned { bytes, live } => &bytes[live.clone()],
        }
    }

    fn owned(bytes: Vec<u8>) -> Self {
        let live = 0..bytes.len();
        Self(Storage::Owned { bytes, live })
    }

    fn normalize(mut bytes: Vec<u8>, live: Range<usize>) -> Vec<u8> {
        if live.start != 0 {
            bytes.copy_within(live.clone(), 0);
        }
        bytes.truncate(live.len());
        bytes
    }
}

enum Storage<'a> {
    Borrowed(&'a [u8]),
    Owned { bytes: Vec<u8>, live: Range<usize> },
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
        let mut spare = Vec::with_capacity(32);
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
        let mut spare = Vec::with_capacity(32);
        let spare_pointer = spare.as_ptr();
        let buffer = if owned {
            FilterBuffer::owned(bytes.to_vec())
        } else {
            FilterBuffer::borrowed(&bytes)
        };
        let input_pointer = buffer.as_slice().as_ptr();
        let buffer = buffer
            .materialize(&mut spare, |input, output| {
                assert!(output.is_empty());
                output.extend(input.iter().rev());
                Ok(())
            })
            .unwrap();
        assert_eq!(buffer.as_slice(), [4, 3, 2, 1]);
        assert_eq!(buffer.as_slice().as_ptr(), spare_pointer);
        assert!(spare.is_empty());
        if owned {
            assert_eq!(spare.as_ptr(), input_pointer);
            assert!(spare.capacity() >= bytes.len());
        } else {
            assert_eq!(spare.capacity(), 0);
        }
        assert_eq!(bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn failed_materialization_retains_capacity_without_partial_output() {
        let mut spare = Vec::with_capacity(32);
        let pointer = spare.as_ptr();
        let error = FilterBuffer::borrowed(&[1, 2, 3, 4])
            .materialize(&mut spare, |_, output| {
                output.extend_from_slice(&[7, 8]);
                Err(Error::UnsupportedFilter(40000))
            })
            .err()
            .unwrap();
        assert_eq!(error, Error::UnsupportedFilter(40000));
        assert!(spare.is_empty());
        assert_eq!(spare.as_ptr(), pointer);
        let output = FilterBuffer::borrowed(&[5, 6])
            .materialize(&mut spare, |input, output| {
                assert!(output.is_empty());
                output.extend_from_slice(input);
                Ok(())
            })
            .unwrap()
            .into_owned();
        assert_eq!(output, [5, 6]);
        assert_eq!(output.as_ptr(), pointer);
    }
}
