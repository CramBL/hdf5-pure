//! Encodes and decodes HDF5 chunk filter pipelines.
//!
//! Shuffle, Fletcher32, LZF, Scale-Offset, and the optional ZFP codec work with `alloc`.
//! Deflate requires the `deflate` feature and `std`.
//! Pipeline ordering, conflict detection, and re-encoding classification use filter identifiers
//! and parameters supplied by the caller.
//!
//! The project supports `hdf5-pure` as its API entry point. Direct use of this published
//! support crate has no independent API compatibility guarantee.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[doc(hidden)]
pub mod __private;

mod error;
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
    FILTER_SHUFFLE, FilterScratch, FilterStep, FilterSteps, H5Z_FLAG_OPTIONAL,
    canonical_filter_position, compress_chunk_with, decompress_chunk, decompress_chunk_with,
    filters_lossless, filters_reencodable, first_filter_conflict,
};

pub use error::Error;

pub use lzf::{
    MAX_EXPANSION as LZF_MAX_EXPANSION, compress as compress_lzf, decompress as decompress_lzf,
    h5py_cd_values as lzf_h5py_cd_values,
};

/// Returns an initial allocation size bounded by the output cap and the input
/// size times the expansion factor.
///
/// A decoder may grow its output beyond this reservation. Passing `None` for
/// `cap` returns zero because the caller has no output size to reserve against.
/// Multiplication saturates if the input size and expansion factor exceed `usize`.
pub fn decode_reservation(cap: Option<usize>, in_size: usize, max_expansion: usize) -> usize {
    cap.map_or(0, |cap| cap.min(in_size.saturating_mul(max_expansion)))
}
