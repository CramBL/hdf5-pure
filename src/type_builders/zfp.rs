//! Fixed-rate ZFP settings for dataset creation.

#![cfg(feature = "zfp")]

use super::DatasetBuilder;
use crate::ZfpRate;
use crate::chunked_write::FilterKind;

impl DatasetBuilder {
    /// Enables fixed-rate ZFP compression and chunked storage.
    ///
    /// `rate` is measured in compressed bits per scalar. The writer resolves the rate against
    /// the dataset's datatype and chunk dimensions. The supported scalar types are [`f32`],
    /// [`f64`], [`i32`], and [`i64`], with one through four chunk dimensions.
    ///
    /// Writing returns [`crate::FormatError::InvalidZfpRateForScalar`] if the rate exceeds
    /// the scalar's width, or [`crate::FormatError::InvalidZfpBitBudget`] if the encoded block budget
    /// is zero. Writing returns [`crate::FormatError::UnsupportedZfp`] for unsupported scalar types or
    /// chunk geometry.
    /// Floating-point block headers can require more bits depending on the data, producing
    /// [`crate::FormatError::FilterError`] during compression.
    ///
    /// Combining ZFP with shuffle, scale-offset, deflate, or LZF returns a filter error when
    /// writing. The encoded data is compatible with the H5Z-ZFP plugin, filter ID 32013.
    pub fn with_zfp(&mut self, rate: ZfpRate) -> &mut Self {
        self.chunk_options.set_filter(FilterKind::Zfp(rate));
        self
    }
}
