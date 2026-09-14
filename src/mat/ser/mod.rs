//! Serializer implementation for MATLAB v7.3 `.mat` files.

mod emit;
mod emit_with_builder;
mod root;
// `pub(crate)` so `mat::complex`'s Miri test can drive the real serializer
// over the byte view its array helpers build. Nothing is re-exported.
pub(crate) mod value;
pub(crate) mod value_ser;

pub use root::{
    to_bytes, to_bytes_with_options, to_path, to_path_with_options, to_writer,
    to_writer_with_options,
};
