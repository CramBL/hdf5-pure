//! The chunk filters and the filter pipeline, for the workspace crates.
//!
//! This module may change in any release.

pub use crate::error::Error;
pub use crate::lzf::compress as compress_lzf;
pub use crate::lzf::decompress as decompress_lzf;
pub use crate::lzf::h5py_cd_values as lzf_h5py_cd_values;
pub use crate::pipeline::ChunkContext;
pub use crate::pipeline::FILTER_DEFLATE;
pub use crate::pipeline::FILTER_FLETCHER32;
pub use crate::pipeline::FILTER_LZF;
pub use crate::pipeline::FILTER_SCALEOFFSET;
pub use crate::pipeline::FILTER_SHUFFLE;
#[cfg(feature = "zfp")]
pub use crate::pipeline::FILTER_ZFP;
pub use crate::pipeline::FilterScratch;
pub use crate::pipeline::FilterStep;
pub use crate::pipeline::FilterSteps;
pub use crate::pipeline::H5Z_FLAG_OPTIONAL;
pub use crate::pipeline::canonical_filter_position;
pub use crate::pipeline::compress_chunk_with;
pub use crate::pipeline::decompress_chunk;
pub use crate::pipeline::decompress_chunk_with;
pub use crate::pipeline::filters_lossless;
pub use crate::pipeline::filters_reencodable;
pub use crate::pipeline::first_filter_conflict;
pub use crate::scaleoffset::FillAvailability;
pub use crate::scaleoffset::ScaleOffset;
pub use crate::scaleoffset::ScaleOffsetByteOrder;
pub use crate::scaleoffset::ScaleOffsetFill;
pub use crate::scaleoffset::ScaleOffsetType;
pub use crate::scaleoffset::build_cd_values as build_scale_offset_cd_values;
pub use crate::scaleoffset::scale_offset_mode;
#[cfg(feature = "zfp")]
pub use crate::zfp::ZfpElementType;
#[cfg(feature = "zfp")]
pub use crate::zfp::compress as compress_zfp;
#[cfg(feature = "zfp")]
pub use crate::zfp::decompress as decompress_zfp;
#[cfg(feature = "zfp")]
pub use crate::zfp::zfp_cd_values_rate;
