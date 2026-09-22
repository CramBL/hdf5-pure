//! HDF5 filter encoding, decoding, and shared allocation bounds.
//!
//! The LZF implementation works on chunk bytes and uses `alloc` without requiring `std`.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(rustdoc::missing_crate_level_docs)]

extern crate alloc;

use core::fmt;

mod lzf;

pub use lzf::{
    MAX_EXPANSION as LZF_MAX_EXPANSION, compress as compress_lzf, decompress as decompress_lzf,
    h5py_cd_values as lzf_h5py_cd_values,
};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Reports a filter stream that cannot be decoded.
pub enum Error {
    /// Reports malformed LZF input or output beyond the caller's size limit.
    InvalidLzfStream(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLzfStream(reason) => write!(f, "lzf: {reason}"),
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
