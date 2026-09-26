//! Checked narrowing of file-derived integers to in-memory widths.
//!
//! HDF5 stores an offset, a length, a size or an element count as a 64-bit value, and the reader
//! indexes an in-memory buffer with `usize`. On a 64-bit host the conversion between the two is
//! infallible. On a 32-bit host `usize` is 32 bits, where `value as usize` truncates anything
//! above 4 GiB, so the read lands on the wrong bytes or the allocation comes out short.
//!
//! [`Narrow`] replaces such a cast with a conversion that reports [`FormatError`], which composes
//! with `?` in the format layer and, through `From<FormatError>`, in the high-level layer.
//! [`Narrow::to_usize`] and [`Narrow::narrow`] report [`FormatError::ValueTooLargeForPlatform`]
//! for a value that overruns one of the targets [`NarrowTarget`] lists. A caller with an error of
//! its own passes it to [`Narrow::narrow_or_else`], which reports that error in place of the
//! platform one.
//!
//! Use them for any value that is file-derived and not structurally bounded: the result of
//! `read_offset` or `read_length`, a data layout address or size, a chunk offset or size, a heap
//! or collection size, an element count, and any arithmetic on those that becomes a slice index or
//! an allocation size. [`slice_range`] takes both bounds of an `(offset, length)` pair at once.
//!
//! Write a narrowing `as` cast only where the source is bounded on every supported target: a
//! "size of offsets" byte that is 2, 4, or 8, a version or flags byte, a `u16` message size, a
//! match arm keyed on the on-disk field width it casts to, or a small loop counter. A widening,
//! such as a `u8` to `usize`, needs no guard. Annotate a cast you keep with
//! `#[expect(clippy::cast_possible_truncation /* or cast_possible_wrap */, reason = "…")]`, whose
//! reason states the bound. The 32-bit cast gate (`just portability::cast-gate`) denies both lints
//! and `unfulfilled_lint_expectations`, so a stale `#[expect]` fails it as an unguarded cast does.

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
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if `self` does not fit `T`, carrying
    /// `self` as a `u64` and naming `T` through [`NarrowTarget::NAME`].
    #[inline]
    fn narrow<T>(self) -> Result<T, FormatError>
    where
        T: NarrowTarget + TryFrom<Self, Error = TryFromIntError>,
    {
        self.narrow_or_else(|| FormatError::ValueTooLargeForPlatform {
            value: self.to_u64(),
            target: T::NAME,
        })
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

    /// Converts `self` to `u64`, the width the platform error reports a value at.
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
    /// The name of the target in [`FormatError::ValueTooLargeForPlatform`], `"usize"` for both
    /// `usize` and [`NonZeroUsize`].
    const NAME: &'static str;
}

impl NarrowTarget for usize {
    const NAME: &'static str = "usize";
}

impl NarrowTarget for u32 {
    const NAME: &'static str = "u32";
}

impl NarrowTarget for NonZeroUsize {
    const NAME: &'static str = "usize";
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

/// True when `addr` is the all-`0xFF` "undefined address" marker for a file
/// whose size-of-offsets is `offset_size` (2, 4, or 8 bytes). HDF5 stores an
/// unallocated block or chunk pointer this way. Readers and the free-space
/// reclaim walk skip such entries. Any other `offset_size` returns `false`.
#[inline]
pub fn is_undefined_addr(addr: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => addr == 0xFFFF,
        4 => addr == 0xFFFF_FFFF,
        8 => addr == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_values_round_trip() {
        assert_eq!(0u64.to_usize().unwrap(), 0);
        assert_eq!(1234u64.to_usize().unwrap(), 1234);
        assert_eq!(42u32.to_usize().unwrap(), 42);
        assert_eq!(1000u64.narrow::<u32>().unwrap(), 1000);
    }

    #[test]
    fn a_value_past_the_target_width_is_reported_with_the_value() {
        let err = 0x1_0000_0000u64.narrow::<u32>().unwrap_err();
        let FormatError::ValueTooLargeForPlatform { value, target } = err else {
            panic!("expected ValueTooLargeForPlatform, got {err:?}");
        };
        assert_eq!(value, 0x1_0000_0000);
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
        assert!(matches!(err, FormatError::OffsetOverflow { .. }));
    }

    // On 64-bit hosts `u64::MAX` fits the `u64` add but not `usize`... actually
    // it does fit usize on 64-bit, so this only errors on the addition. Verify
    // the platform-narrowing guard directly with a value that overflows usize
    // only where usize < 64 bits. On 64-bit it succeeds, which is correct.
    #[test]
    fn large_u64_behaviour_matches_platform_width() {
        let big: u64 = u64::from(u32::MAX) + 1; // 2^32
        match big.to_usize() {
            // 64-bit (and wider) hosts: fits.
            Ok(v) => assert_eq!(v as u64, big),
            // 32-bit hosts: must be reported, never truncated.
            Err(e) => assert!(matches!(e, FormatError::ValueTooLargeForPlatform { .. })),
        }
    }
}
