//! Encoding and decoding for HDF5 chunk filters.
//!
//! LZF and the optional ZFP codec work with `alloc` and do not require `std`.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(rustdoc::missing_crate_level_docs)]

extern crate alloc;

use core::fmt;

mod lzf;
#[cfg(feature = "zfp")]
mod zfp;

#[cfg(feature = "zfp")]
pub use zfp::{
    ZfpElementType, compress as compress_zfp, compress_filter as compress_zfp_filter,
    decompress as decompress_zfp, decompress_filter as decompress_zfp_filter, zfp_cd_values_rate,
    zfp_rate_from_cd_values,
};

pub use lzf::{
    MAX_EXPANSION as LZF_MAX_EXPANSION, compress as compress_lzf, decompress as decompress_lzf,
    h5py_cd_values as lzf_h5py_cd_values,
};

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
/// Reports an invalid filter stream or codec configuration.
pub enum Error {
    /// Reports malformed LZF input or output beyond the caller's size limit.
    InvalidLzfStream(&'static str),
    #[cfg(feature = "zfp")]
    /// Reports fewer encoded bytes than the ZFP chunk requires.
    TruncatedZfpStream { expected: usize, actual: usize },
    #[cfg(feature = "zfp")]
    /// Reports a ZFP configuration the codec cannot encode.
    UnsupportedZfp(alloc::string::String),
    #[cfg(feature = "zfp")]
    /// Reports a dimension that does not fit the platform's index width.
    ValueTooLargeForPlatform { value: u64, target: &'static str },
    #[cfg(feature = "zfp")]
    /// Reports invalid ZFP parameters or input bytes.
    ZfpFilter(alloc::string::String),
    #[cfg(feature = "zfp")]
    /// Reports a float block whose header exceeds the fixed-rate bit budget.
    ZfpHeaderTooLarge { budget: usize, required: usize },
    #[cfg(feature = "zfp")]
    /// Reports an overflow while calculating a ZFP chunk size.
    ZfpSizeOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLzfStream(reason) => write!(f, "lzf: {reason}"),
            #[cfg(feature = "zfp")]
            Self::TruncatedZfpStream { expected, actual } => {
                write!(f, "ZFP: encoded chunk needs {expected} bytes, got {actual}")
            }
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
