use core::num::NonZeroU32;
use core::num::NonZeroUsize;

use crate::FormatError;

pub use hdf5_pure_core::FixedPointLayout;
pub use hdf5_pure_core::FloatingPointLayout;
pub use hdf5_pure_format::StandardNumericLayout;
pub use hdf5_pure_format::StandardWidth;

/// A numeric element size supported by the integer and floating-point readers.
///
/// The size is between one and eight bytes, inclusive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericElementSize(NonZeroUsize);

impl NumericElementSize {
    const MAX: usize = 8;

    /// Converts a datatype's element size to a supported numeric width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NumericElementTooWide`] for a size above eight bytes.
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the size does not fit in `usize`.
    pub fn new(size: NonZeroU32) -> Result<Self, FormatError> {
        #[expect(
            clippy::map_err_ignore,
            reason = "The semantics are fully communicated through the format error"
        )]
        let size =
            usize::try_from(size.get()).map_err(|_| FormatError::ValueTooLargeForPlatform {
                value: size.get().into(),
                target: "usize",
            })?;

        // A checked conversion of a nonzero integer remains nonzero.
        let size = NonZeroUsize::new(size).expect("nonzero size remains nonzero");

        if size.get() > Self::MAX {
            return Err(FormatError::NumericElementTooWide { size: size.get() });
        }

        Ok(Self(size))
    }

    /// Returns the element size in bytes.
    pub const fn get(self) -> usize {
        self.0.get()
    }
}
