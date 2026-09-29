//! Encodes and decodes HDF5 chunk filter pipelines.
//!
//! Shuffle, Fletcher32, LZF, Scale-Offset, and the optional ZFP codec work with `alloc`.
//! Deflate requires the `deflate` feature and `std`. The crate exports its filters and its
//! pipeline to the workspace crates alone, through a hidden module that may change in any release.
//!
//! The project supports `hdf5-pure` as its API entry point.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(unreachable_pub)]

extern crate alloc;

#[doc(hidden)]
pub mod __private;

mod error;
mod lzf;
mod pipeline;
mod scaleoffset;
#[cfg(feature = "zfp")]
mod zfp;

/// Returns an initial allocation size bounded by the output cap and the input
/// size times the expansion factor.
///
/// A decoder may grow its output beyond this reservation. Passing `None` for
/// `cap` returns zero because the caller has no output size to reserve against.
/// Multiplication saturates if the input size and expansion factor exceed `usize`.
pub(crate) fn decode_reservation(
    cap: Option<usize>,
    in_size: usize,
    max_expansion: usize,
) -> usize {
    cap.map_or(0, |cap| cap.min(in_size.saturating_mul(max_expansion)))
}
