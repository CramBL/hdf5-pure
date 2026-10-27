//! Byte copies for native numeric representations.

use alloc::vec::Vec;
use core::{mem, ptr};

use crate::FormatError;

/// A fixed-width numeric primitive with a padding-free representation.
///
/// This trait is sealed to [`i8`], [`i16`], [`i32`], [`i64`], [`u8`], [`u16`], [`u32`], [`u64`],
/// [`f32`], and [`f64`]. Every bit pattern is valid for each of these types.
pub(crate) trait PrimitiveBytes: Copy + sealed::Sealed {
    /// Appends `src` as native-endian values, preserving the existing elements of `dst`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if `src` contains an incomplete element.
    /// On error, `dst` is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes.
    fn append_native(src: &[u8], dst: &mut Vec<Self>) -> Result<(), FormatError> {
        let remainder = src.len() % mem::size_of::<Self>();
        if remainder != 0 {
            return Err(FormatError::DataSizeMismatch {
                expected: src.len() - remainder,
                actual: src.len(),
            });
        }

        let count = src.len() / mem::size_of::<Self>();
        let len = dst.len();
        dst.reserve(count);
        // SAFETY: Sealing restricts `Self` to padding-free numeric primitives for which every bit
        // pattern is valid. The whole-element check and `reserve` provide `count` aligned slots after
        // the existing elements, and the disjoint source initializes those slots before `set_len`.
        unsafe {
            ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.spare_capacity_mut().as_mut_ptr().cast::<u8>(),
                src.len(),
            );
            dst.set_len(len + count);
        }
        Ok(())
    }
}

macro_rules! impl_primitive_bytes {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            impl PrimitiveBytes for $ty {}
        )+
    };
}

impl_primitive_bytes!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

mod sealed {
    pub trait Sealed {}
}
