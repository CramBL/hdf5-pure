//! Checked integer conversions with host-width and fixed-width overflow diagnostics.
//!
//! [`Narrow::narrow`] reports host-width limits for [`usize`] and [`NonZeroUsize`], and
//! representation limits for [`u32`]. Callers pass field-specific errors to
//! [`Narrow::narrow_or_else`]. [`slice_range`] checks the addition and conversion of a byte range's
//! bounds before indexing a buffer.

use core::num::NonZeroU32;
use core::num::NonZeroUsize;
use core::num::TryFromIntError;
use core::ops::Range;

use crate::error::FormatError;

/// Checked narrowing of a file-derived integer to another integer type.
///
/// Where the source type fits the target on this host, the conversion collapses to a widening and
/// the error arm is cold. Where a value exceeds the target, it reaches the caller in an error.
pub trait Narrow: Copy {
    /// Narrows `self` to the width that indexes an in-memory buffer.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if `self` is above `usize::MAX` on this
    /// host. A 64-bit host never reaches it.
    #[inline]
    fn to_usize(self) -> Result<usize, FormatError>
    where
        usize: TryFrom<Self, Error = TryFromIntError>,
    {
        self.narrow()
    }

    /// Narrows `self` to `T`, one of the targets [`NarrowTarget`] lists.
    ///
    /// The target is inferred where the binding or the argument fixes it, and written out as
    /// `narrow::<u32>()` where it does not.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if `self` exceeds the range of [`usize`]
    /// or [`NonZeroUsize`], or [`FormatError::ValueTooLargeForRepresentation`] if `self` exceeds
    /// the range of [`u32`]. Both errors include the original value and the target's
    /// [`name`](NarrowTarget::NAME).
    #[inline]
    fn narrow<T>(self) -> Result<T, FormatError>
    where
        T: NarrowTarget + TryFrom<Self, Error = TryFromIntError>,
    {
        self.narrow_or_else(|| T::overflow(self.to_u64()))
    }

    /// Narrows `self` to `T`, reporting the error `error` builds.
    ///
    /// The counterpart to [`narrow`](Narrow::narrow) where the caller has an error of its own:
    /// for a target [`NarrowTarget`] does not list, such as the `u16` size of an object header
    /// message, and for one it does list whose limit the caller's error reports better, such as
    /// the byte offset of a compound field. The bound admits only [`TryFromIntError`], so the
    /// caller's error loses nothing by replacing it.
    ///
    /// # Errors
    ///
    /// Returns `error()` if `self` does not fit `T`.
    #[expect(
        clippy::map_err_ignore,
        reason = "the bound admits only `TryFromIntError`, which carries nothing beyond \
                  the conversion having failed, and that is what the error `error` builds reports"
    )]
    #[inline]
    fn narrow_or_else<T, E>(self, error: impl FnOnce() -> E) -> Result<T, E>
    where
        T: TryFrom<Self, Error = TryFromIntError>,
    {
        T::try_from(self).map_err(|_| error())
    }

    /// Converts `self` to [`u64`] for overflow diagnostics.
    fn to_u64(self) -> u64;
}

impl Narrow for u64 {
    #[inline]
    fn to_u64(self) -> u64 {
        self
    }
}

impl Narrow for u32 {
    #[inline]
    fn to_u64(self) -> u64 {
        u64::from(self)
    }
}

impl Narrow for usize {
    #[inline]
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl Narrow for NonZeroU32 {
    #[inline]
    fn to_u64(self) -> u64 {
        u64::from(self.get())
    }
}

/// A target of [`Narrow::narrow`], which reports it by name.
///
/// Implemented for `usize`, [`NonZeroUsize`] and `u32`, the targets whose overflow needs no
/// diagnostic beyond the value and the target's name. Any other target, such as the `u16` size of
/// an object header message, goes through [`Narrow::narrow_or_else`] with the caller's error.
pub trait NarrowTarget {
    /// The target's name in overflow diagnostics, `"usize"` for both [`usize`] and
    /// [`NonZeroUsize`].
    const NAME: &'static str;

    /// Returns the error for a value above the target's range.
    ///
    /// Uses [`FormatError::ValueTooLargeForPlatform`] for [`usize`] and [`NonZeroUsize`], and
    /// [`FormatError::ValueTooLargeForRepresentation`] for [`u32`].
    fn overflow(value: u64) -> FormatError;
}

impl NarrowTarget for usize {
    const NAME: &'static str = "usize";

    fn overflow(value: u64) -> FormatError {
        FormatError::ValueTooLargeForPlatform {
            value,
            target: Self::NAME,
        }
    }
}

impl NarrowTarget for u32 {
    const NAME: &'static str = "u32";

    fn overflow(value: u64) -> FormatError {
        FormatError::ValueTooLargeForRepresentation {
            value,
            target: Self::NAME,
        }
    }
}

impl NarrowTarget for NonZeroUsize {
    const NAME: &'static str = "usize";

    fn overflow(value: u64) -> FormatError {
        FormatError::ValueTooLargeForPlatform {
            value,
            target: Self::NAME,
        }
    }
}

/// Compute `offset .. offset + len` as a `usize` range, checking both the
/// 64-bit addition and the narrowing of each bound to `usize`.
///
/// Use this anywhere a file-derived `(offset, length)` pair becomes a slice
/// index. It guards two distinct hazards a bare `offset as usize + len as usize`
/// misses: the `u64` addition wrapping ([`FormatError::OffsetOverflow`]) and
/// either operand exceeding `usize` on a 32-bit target
/// ([`FormatError::ValueTooLargeForPlatform`]). The returned `range.end` is the
/// already-checked `usize` end bound, so a subsequent bounds check against the
/// buffer length stays truthful.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if `offset + len` exceeds `u64`, or
/// [`FormatError::ValueTooLargeForPlatform`] if either bound exceeds `usize`.
#[inline]
pub fn slice_range(offset: u64, len: u64) -> Result<Range<usize>, FormatError> {
    let end = offset.checked_add(len).ok_or(FormatError::OffsetOverflow {
        offset,
        length: len,
    })?;
    Ok(offset.to_usize()?..end.to_usize()?)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::zero(0)]
    #[case::interior(1000)]
    #[case::u32_limit(u64::from(u32::MAX))]
    fn a_value_within_the_target_width_converts(#[case] value: u64) {
        assert_eq!(u64::from(value.narrow::<u32>().unwrap()), value);
    }

    #[rstest]
    #[case::above_u32_limit(u64::from(u32::MAX) + 1)]
    #[case::u64_limit(u64::MAX)]
    fn a_value_past_the_target_width_is_reported_with_the_value(#[case] original: u64) {
        let err = original.narrow::<u32>().unwrap_err();
        let FormatError::ValueTooLargeForRepresentation { value, target } = err else {
            panic!("expected ValueTooLargeForRepresentation, got {err:?}");
        };
        assert_eq!(value, original);
        assert_eq!(target, "u32");
    }

    #[test]
    fn the_callers_error_replaces_the_platform_one() {
        assert_eq!(
            65_535u32
                .narrow_or_else::<u16, _>(|| "index does not fit u16")
                .unwrap(),
            65_535
        );
        assert_eq!(
            65_536u32
                .narrow_or_else::<u16, _>(|| "index does not fit u16")
                .unwrap_err(),
            "index does not fit u16"
        );
        assert_eq!(
            (u64::from(u32::MAX) + 1)
                .narrow_or_else::<u32, _>(|| "count does not fit u32")
                .unwrap_err(),
            "count does not fit u32"
        );
    }

    /// The whole point of the conversion is that the proof survives it: what
    /// goes in non-zero comes out non-zero, at the platform's width.
    #[test]
    fn a_nonzero_width_survives_the_narrowing() {
        let width = NonZeroU32::new(8).unwrap();
        assert_eq!(width.narrow::<NonZeroUsize>().unwrap().get(), 8);

        let widest = NonZeroU32::new(u32::MAX).unwrap();
        assert_eq!(
            widest.narrow::<NonZeroUsize>().unwrap().get(),
            u32::MAX as usize
        );
    }

    #[test]
    fn slice_range_basic() {
        let r = slice_range(10, 20).unwrap();
        assert_eq!(r, 10..30);
    }

    #[test]
    fn slice_range_addition_overflow_is_caught() {
        let err = slice_range(u64::MAX, 1).unwrap_err();
        assert_eq!(
            err,
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        );
    }

    // On 64-bit hosts `u64::MAX` fits the `u64` add but not `usize`... actually
    // it does fit usize on 64-bit, so this only errors on the addition. Verify
    // the platform-narrowing guard directly with a value that overflows usize
    // only where usize < 64 bits. On 64-bit it succeeds, which is correct.
    #[test]
    fn large_u64_behaviour_matches_platform_width() {
        let big: u64 = u64::from(u32::MAX) + 1; // 2^32
        if usize::BITS >= 64 {
            assert_eq!(big.to_usize().unwrap() as u64, big);
        } else {
            assert_eq!(
                big.to_usize().unwrap_err(),
                FormatError::ValueTooLargeForPlatform {
                    value: big,
                    target: "usize",
                }
            );
        }
    }
}
