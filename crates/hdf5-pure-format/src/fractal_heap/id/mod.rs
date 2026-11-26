//! Fractal heap ID prefixes, payloads, and encoding.

mod frame;
mod huge;
mod managed;
mod prepared;
mod tiny;

pub use frame::{FractalHeapIdKind, FractalHeapIdLayout, FractalHeapIdView};
pub use huge::{HugeIndirectKey, HugeIndirectKeyEncoding, HugeObjectReference};
pub(crate) use prepared::PreparedHeapId;
pub use tiny::TinyHeapIdEncoding;
