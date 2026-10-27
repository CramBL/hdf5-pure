//! Address-ordered output from sorted reservation iterators.

#[cfg(not(feature = "std"))]
use alloc::format;

use core::convert::Infallible;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::PhysicalAddress;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::chunked_write::ByteSink;
use crate::convert::Narrow;
use crate::error::FormatError;

/// Merges reservation sources while retaining one pending reservation from each.
///
/// The caller supplies each source in ascending physical-address order. The writer fills gaps
/// with zeros and delegates each reservation's contents to a callback.
pub(crate) struct Emission<'a, T, const N: usize> {
    sources: [&'a mut dyn Iterator<Item = Result<Reservation<T>, FormatError>>; N],
}

impl<'a, T, const N: usize> Emission<'a, T, N> {
    /// Borrows the sorted sources for one emission.
    pub(crate) fn new(
        sources: [&'a mut dyn Iterator<Item = Result<Reservation<T>, FormatError>>; N],
    ) -> Self {
        Self { sources }
    }

    /// Writes reservations in physical-address order and pads the output to `eof`.
    ///
    /// Calls `emit` once per reservation, with the sink positioned at its start. The callback
    /// must write exactly the reserved length. Errors can leave partially written output.
    ///
    /// # Errors
    ///
    /// Returns [`EmissionError::Format`] with [`FormatError::Internal`] if a reservation precedes
    /// the sink's position, extends past `eof`, or differs in length from the callback's output,
    /// or if output begins past `eof`. Propagates source errors as [`EmissionError::Format`],
    /// sink errors as [`EmissionError::Sink`], and callback errors unchanged.
    pub(crate) fn emit<S: ByteSink>(
        self,
        sink: &mut S,
        eof: u64,
        mut emit: impl FnMut(&mut S, T) -> Result<(), EmissionError<S::Error>>,
    ) -> Result<(), EmissionError<S::Error>> {
        if sink.position() > eof {
            return Err(
                FormatError::Internal("output begins beyond the end of the file".into()).into(),
            );
        }
        let Self { mut sources } = self;
        let mut heads: [Option<Reservation<T>>; N] = core::array::from_fn(|_| None);
        for (head, source) in heads.iter_mut().zip(sources.iter_mut()) {
            *head = source.next().transpose()?;
        }
        while let Some(index) = heads
            .iter()
            .enumerate()
            .filter_map(|(i, head)| head.as_ref().map(|head| (i, head.range.start())))
            .min_by_key(|&(_, address)| address)
            .map(|(i, _)| i)
        {
            let Some(Reservation { range, value }) = heads[index].take() else {
                return Err(
                    FormatError::Internal("the selected output source is empty".into()).into(),
                );
            };
            if range.start().get() < sink.position() || range.end().get() > eof {
                return Err(FormatError::Internal(
                    "output reservations overlap or extend beyond the file".into(),
                )
                .into());
            }
            Self::pad_to(sink, range.start().get())?;
            emit(sink, value)?;
            if sink.position() != range.end().get() {
                return Err(FormatError::Internal(format!(
                    "output reservation ends at {}, but emission ends at {}",
                    range.end().get(),
                    sink.position(),
                ))
                .into());
            }
            heads[index] = sources[index].next().transpose()?;
        }
        Self::pad_to(sink, eof)
    }

    fn pad_to<S: ByteSink>(sink: &mut S, end: u64) -> Result<(), EmissionError<S::Error>> {
        while sink.position() < end {
            let length = (end - sink.position()).min(65_536).to_usize()?;
            sink.put_zeros(length).map_err(EmissionError::Sink)?;
        }
        Ok(())
    }
}

/// A failure during output planning, serialization, or sink operations.
///
/// Format failures include capacity errors from [`ByteSink::reserve`]. Sink failures hold the
/// original error from [`ByteSink::put`] or [`ByteSink::put_zeros`].
#[derive(Debug, PartialEq)]
pub(crate) enum EmissionError<E> {
    /// A planning or serialization failure.
    Format(FormatError),
    /// A failure to write output bytes or padding.
    Sink(E),
}

impl EmissionError<Infallible> {
    pub(crate) fn into_format(self) -> FormatError {
        match self {
            Self::Format(error) => error,
            Self::Sink(error) => match error {},
        }
    }
}

impl<E> From<FormatError> for EmissionError<E> {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Associates an output value with its checked physical output range.
pub(crate) struct Reservation<T> {
    range: FileRange,
    value: T,
}

impl<T> Reservation<T> {
    /// Resolves `address` through `base` to describe the physical output range for `value`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the physical start or end exceeds [`u64::MAX`].
    pub(crate) fn new(
        base: BaseAddress,
        address: StoredAddress,
        length: u64,
        value: T,
    ) -> Result<Self, FormatError> {
        let start = PhysicalAddress::new(base.absolute(address)?);
        Ok(Self {
            range: FileRange::new(start, length)?,
            value,
        })
    }

    pub(crate) fn start(&self) -> PhysicalAddress {
        self.range.start()
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    use rstest::rstest;

    use super::Emission;
    use super::EmissionError;
    use super::Reservation;
    use crate::address::BaseAddress;
    use crate::address::BaseAddressExt;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::chunked_write::ByteSink;
    use crate::error::FormatError;

    #[rstest]
    #[case::base_zero(0)]
    #[case::userblock(512)]
    fn reservation_sources_merge_in_address_order(#[case] base: u64) {
        let base = BaseAddress::new(base);
        let mut first = [
            Reservation::new(base, StoredAddress::new(7), 2, b"cd".as_slice()),
            Reservation::new(base, StoredAddress::new(10), 1, b"e".as_slice()),
        ]
        .into_iter();
        let mut second = core::iter::once(Reservation::new(
            base,
            StoredAddress::new(2),
            3,
            b"abc".as_slice(),
        ));
        let mut sink = vec![0xAB; base.get() as usize];
        Emission::new([&mut first, &mut second])
            .emit(&mut sink, base.get() + 12, |sink, bytes| {
                sink.put(bytes).map_err(EmissionError::Sink)
            })
            .unwrap();
        let mut expected = vec![0xAB; base.get() as usize];
        expected.extend_from_slice(b"\0\0abc\0\0cd\0e\0");
        assert_eq!(sink, expected);
    }

    #[rstest]
    #[case::overlap(0, 3, 5, 7, 1, 8, 8)]
    #[case::before_cursor(4, 3, 1, 7, 1, 8, 4)]
    #[case::beyond_eof(0, 3, 1, 7, 2, 8, 4)]
    #[case::unordered_source(0, 5, 1, 3, 1, 8, 6)]
    fn inconsistent_reservations_are_rejected(
        #[case] position: u64,
        #[case] first: u64,
        #[case] length: u64,
        #[case] second: u64,
        #[case] second_length: u64,
        #[case] eof: u64,
        #[case] final_position: u64,
    ) {
        let base = BaseAddress::new(0);
        let mut source = [
            Reservation::new(base, StoredAddress::new(first), length, length),
            Reservation::new(
                base,
                StoredAddress::new(second),
                second_length,
                second_length,
            ),
        ]
        .into_iter();
        let mut sink = CountingSink {
            position,
            ..CountingSink::default()
        };
        let error = Emission::new([&mut source])
            .emit(&mut sink, eof, |sink, length| {
                sink.put_zeros(usize::try_from(length).unwrap())
                    .map_err(EmissionError::Sink)
            })
            .unwrap_err();
        assert_eq!(
            error,
            EmissionError::Format(FormatError::Internal(
                "output reservations overlap or extend beyond the file".into()
            ))
        );
        assert_eq!(sink.position, final_position);
    }

    #[rstest]
    #[case::base_overflow(u64::MAX, 1, 0, 1, u64::MAX)]
    #[case::end_overflow(0, u64::MAX, 1, u64::MAX, 1)]
    fn overflowing_reservations_are_rejected(
        #[case] base: u64,
        #[case] start: u64,
        #[case] length: u64,
        #[case] offset: u64,
        #[case] overflow_length: u64,
    ) {
        let mut source = core::iter::once(Reservation::new(
            BaseAddress::new(base),
            StoredAddress::new(start),
            length,
            (),
        ));
        let error = Emission::new([&mut source])
            .emit(&mut Vec::new(), u64::MAX, |_, ()| {
                Err(FormatError::Internal("unexpected output reservation".into()).into())
            })
            .unwrap_err();
        assert_eq!(
            error,
            FormatError::OffsetOverflow {
                offset,
                length: overflow_length
            }
            .into()
        );
    }

    #[rstest]
    #[case::short(b"x".as_slice(), 4)]
    #[case::long(b"xyz".as_slice(), 6)]
    fn emission_must_fill_its_reservation(#[case] bytes: &[u8], #[case] actual_end: u64) {
        let mut source = core::iter::once(Reservation::new(
            BaseAddress::new(0),
            StoredAddress::new(3),
            2,
            bytes,
        ));
        let error = Emission::new([&mut source])
            .emit(&mut Vec::new(), 5, |sink, bytes| {
                sink.put(bytes).map_err(EmissionError::Sink)
            })
            .unwrap_err();
        assert_eq!(
            error,
            FormatError::Internal(format!(
                "output reservation ends at 5, but emission ends at {actual_end}"
            ))
            .into()
        );
    }

    #[test]
    fn positions_above_four_gibibytes_use_bounded_gaps() {
        let start = (1u64 << 32) + 3;
        let mut source = core::iter::once(Reservation::new(
            BaseAddress::new(512),
            StoredAddress::new(start),
            1,
            b"x".as_slice(),
        ));
        let mut sink = CountingSink::default();
        Emission::new([&mut source])
            .emit(&mut sink, start + 515, |sink, bytes| {
                sink.put(bytes).map_err(EmissionError::Sink)
            })
            .unwrap();
        assert_eq!(sink.position, start + 515);
        assert_eq!(sink.writes, [(start + 512, 1)]);
        assert_eq!(sink.largest_gap, 65_536);
    }

    #[rstest]
    #[case::empty(0, 0)]
    #[case::padding(0, 3)]
    #[case::at_eof(3, 3)]
    fn empty_sources_preserve_the_end(#[case] position: u64, #[case] eof: u64) {
        let mut source = core::iter::empty();
        let mut sink = CountingSink {
            position,
            ..CountingSink::default()
        };
        Emission::new([&mut source])
            .emit(&mut sink, eof, |_, ()| {
                Err(FormatError::Internal("unexpected output reservation".into()).into())
            })
            .unwrap();
        assert_eq!(sink.position, eof);
        assert_eq!(sink.writes, []);
    }

    #[rstest]
    #[case::no_sources(0)]
    #[case::before_cursor(3)]
    fn an_end_before_the_cursor_is_rejected(#[case] position: u64) {
        let mut sink = CountingSink {
            position: position + 1,
            ..CountingSink::default()
        };
        let error = Emission::new([])
            .emit(&mut sink, position, |_, ()| Ok(()))
            .unwrap_err();
        assert_eq!(
            error,
            EmissionError::Format(FormatError::Internal(
                "output begins beyond the end of the file".into()
            ))
        );
    }

    #[test]
    fn callback_errors_stop_before_later_reservations() {
        let mut source = [
            Reservation::new(BaseAddress::new(0), StoredAddress::new(3), 1, 1),
            Reservation::new(BaseAddress::new(0), StoredAddress::new(5), 1, 2),
        ]
        .into_iter();
        let mut calls = Vec::new();
        let mut sink = CountingSink::default();
        let expected = FormatError::SerializationError("source failed".into());
        let error = Emission::new([&mut source])
            .emit(&mut sink, 6, |_, index| {
                calls.push(index);
                Err(expected.clone().into())
            })
            .unwrap_err();
        assert_eq!(error, EmissionError::Format(expected));
        assert_eq!(calls, [1]);
        assert_eq!(sink.position, 3);
    }

    #[rstest]
    #[case::padding(1, 0, 0, 0)]
    #[case::payload(0, 0, 0, 1)]
    #[case::trailing_padding(0, 1, 1, 1)]
    fn sink_errors_stop_emission(
        #[case] start: u64,
        #[case] accepted: usize,
        #[case] position: u64,
        #[case] expected_calls: usize,
    ) {
        let mut source = core::iter::once(Reservation::new(
            BaseAddress::new(0),
            StoredAddress::new(start),
            1,
            b"x".as_slice(),
        ));
        let mut calls = 0;
        let error = Emission::new([&mut source])
            .emit(
                &mut FailingSink {
                    accepted,
                    position: 0,
                },
                start + 2,
                |sink, bytes| {
                    calls += 1;
                    sink.put(bytes).map_err(EmissionError::Sink)
                },
            )
            .unwrap_err();
        assert_eq!(error, EmissionError::Sink(SinkFailure { position }));
        assert_eq!(calls, expected_calls);
    }

    #[derive(Debug, PartialEq)]
    struct SinkFailure {
        position: u64,
    }

    struct FailingSink {
        accepted: usize,
        position: u64,
    }

    impl FailingSink {
        fn accept(&mut self, length: usize) -> Result<(), SinkFailure> {
            if self.accepted == 0 {
                return Err(SinkFailure {
                    position: self.position,
                });
            }
            self.accepted -= 1;
            self.position += length as u64;
            Ok(())
        }
    }

    impl ByteSink for FailingSink {
        type Error = SinkFailure;

        fn put(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
            self.accept(bytes.len())
        }

        fn put_zeros(&mut self, length: usize) -> Result<(), Self::Error> {
            self.accept(length)
        }

        fn position(&self) -> u64 {
            self.position
        }
    }

    #[derive(Default)]
    struct CountingSink {
        position: u64,
        largest_gap: usize,
        writes: Vec<(u64, usize)>,
    }

    impl ByteSink for CountingSink {
        type Error = FormatError;
        fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
            self.writes.push((self.position, bytes.len()));
            self.position += bytes.len() as u64;
            Ok(())
        }

        fn put_zeros(&mut self, length: usize) -> Result<(), FormatError> {
            self.largest_gap = self.largest_gap.max(length);
            self.position += length as u64;
            Ok(())
        }

        fn position(&self) -> u64 {
            self.position
        }
    }
}
