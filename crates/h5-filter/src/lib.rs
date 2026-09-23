//! Encodes and decodes HDF5 chunk filter pipelines.
//!
//! Shuffle, Fletcher32, LZF, Scale-Offset, and the optional ZFP codec work with `alloc`.
//! Deflate requires the `deflate` feature and `std`.
//! Pipeline ordering, conflict detection, and re-encoding classification use filter identifiers
//! and parameters supplied by the caller.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(rustdoc::missing_crate_level_docs)]

extern crate alloc;

use core::fmt;

mod lzf;
mod pipeline;
mod scaleoffset;
#[cfg(feature = "zfp")]
mod zfp;

pub use scaleoffset::{
    FillAvailability, HEADER_LEN as SCALE_OFFSET_HEADER_LEN, ScaleOffset, ScaleOffsetByteOrder,
    ScaleOffsetFill, ScaleOffsetType, build_cd_values as build_scale_offset_cd_values,
    compress as compress_scale_offset, decompress as decompress_scale_offset, scale_offset_mode,
};

#[cfg(feature = "zfp")]
pub use zfp::{
    ZfpElementType, compress as compress_zfp, compress_filter as compress_zfp_filter,
    decompress as decompress_zfp, decompress_filter as decompress_zfp_filter, zfp_cd_values_rate,
    zfp_rate_from_cd_values,
};

#[cfg(feature = "zfp")]
pub use pipeline::FILTER_ZFP;
pub use pipeline::{
    ChunkContext, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET,
    FILTER_SHUFFLE, FilterScratch, FilterStep, H5Z_FLAG_OPTIONAL, canonical_filter_position,
    compress_chunk_with, decompress_chunk, decompress_chunk_with, filters_lossless,
    filters_reencodable, first_filter_conflict,
};

pub use lzf::{
    MAX_EXPANSION as LZF_MAX_EXPANSION, compress as compress_lzf, decompress as decompress_lzf,
    h5py_cd_values as lzf_h5py_cd_values,
};

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
/// Reports an invalid filter stream or codec configuration.
pub enum Error {
    /// Reports a decoded chunk whose length differs from its full chunk size.
    DataSizeMismatch {
        /// Number of bytes required by the chunk dimensions and element size.
        expected: usize,
        /// Number of bytes returned by the filter pipeline.
        actual: usize,
    },
    /// Reports malformed Deflate or Shuffle input, or a filter operation that failed.
    FilterError(alloc::string::String),
    /// Reports a Fletcher32 checksum that differs from the stored checksum.
    Fletcher32Mismatch {
        /// Checksum stored after the payload.
        expected: u32,
        /// Checksum calculated from the payload.
        computed: u32,
    },
    /// Reports malformed LZF input or output beyond the caller's size limit.
    InvalidLzfStream(&'static str),
    /// Reports invalid scale-offset parameters or input bytes.
    ScaleOffset(alloc::string::String),
    /// Reports a scale-offset element count that exceeds the platform's index width.
    ScaleOffsetValueTooLargeForPlatform {
        /// Element count read from the filter parameters.
        value: u64,
        /// Platform index type that cannot hold the value.
        target: &'static str,
    },
    /// Reports fewer encoded bytes than the ZFP chunk requires.
    #[cfg(feature = "zfp")]
    TruncatedZfpStream {
        /// Number of encoded bytes required for the chunk.
        expected: usize,
        /// Number of encoded bytes present.
        actual: usize,
    },
    /// Reports a filter identifier without an available encoder or decoder.
    UnsupportedFilter(u16),
    /// Reports a ZFP configuration the codec cannot encode.
    #[cfg(feature = "zfp")]
    UnsupportedZfp(alloc::string::String),
    /// Reports a dimension that does not fit the platform's index width.
    #[cfg(feature = "zfp")]
    ValueTooLargeForPlatform {
        /// Dimension read from the chunk shape.
        value: u64,
        /// Platform index type that cannot hold the value.
        target: &'static str,
    },
    /// Reports invalid ZFP parameters or input bytes.
    #[cfg(feature = "zfp")]
    ZfpFilter(alloc::string::String),
    /// Reports a float block whose header exceeds the fixed-rate bit budget.
    #[cfg(feature = "zfp")]
    ZfpHeaderTooLarge {
        /// Number of bits available at the configured rate.
        budget: usize,
        /// Number of bits required by the block header.
        required: usize,
    },
    /// Reports an overflow while calculating a ZFP chunk size.
    #[cfg(feature = "zfp")]
    ZfpSizeOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataSizeMismatch { expected, actual } => {
                write!(f, "data size mismatch: expected {expected}, got {actual}")
            }
            Self::FilterError(reason) => write!(f, "filter error: {reason}"),
            Self::Fletcher32Mismatch { expected, computed } => write!(
                f,
                "fletcher32 checksum mismatch: expected {expected}, computed {computed}"
            ),
            Self::InvalidLzfStream(reason) => write!(f, "lzf: {reason}"),
            Self::ScaleOffset(reason) => write!(f, "filter error: {reason}"),
            Self::ScaleOffsetValueTooLargeForPlatform { value, target } => write!(
                f,
                "scaleoffset: value {value} does not fit in {target} on this platform"
            ),
            #[cfg(feature = "zfp")]
            Self::TruncatedZfpStream { expected, actual } => {
                write!(f, "ZFP: encoded chunk needs {expected} bytes, got {actual}")
            }
            Self::UnsupportedFilter(id) => write!(f, "unsupported filter {id}"),
            #[cfg(feature = "zfp")]
            Self::UnsupportedZfp(reason) => write!(f, "unsupported ZFP configuration: {reason}"),
            #[cfg(feature = "zfp")]
            Self::ValueTooLargeForPlatform { value, target } => write!(
                f,
                "ZFP chunk dimension {value} does not fit in {target} on this platform"
            ),
            #[cfg(feature = "zfp")]
            Self::ZfpFilter(reason) => write!(f, "filter error: {reason}"),
            #[cfg(feature = "zfp")]
            Self::ZfpHeaderTooLarge { budget, required } => write!(
                f,
                "ZFP: nonzero float block needs {required} header bits, rate allows {budget} bits"
            ),
            #[cfg(feature = "zfp")]
            Self::ZfpSizeOverflow => {
                write!(f, "ZFP: chunk dimensions or encoded size overflow usize")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Returns an initial allocation size bounded by the output cap and the input
/// size times the expansion factor.
///
/// A decoder may grow its output beyond this reservation. Passing `None` for
/// `cap` returns zero because the caller has no output size to reserve against.
/// Multiplication saturates if the input size and expansion factor exceed `usize`.
///
/// # Examples
///
/// ```
/// use h5_filter::decode_reservation;
///
/// assert_eq!(decode_reservation(Some(4096), 10, 88), 880);
/// assert_eq!(decode_reservation(Some(4096), 100, 88), 4096);
/// assert_eq!(decode_reservation(None, 100, 88), 0);
/// ```
pub fn decode_reservation(cap: Option<usize>, in_size: usize, max_expansion: usize) -> usize {
    cap.map_or(0, |cap| cap.min(in_size.saturating_mul(max_expansion)))
}
