//! Byte lengths, element counts, and positions in a chunk index.
//!
//! Arithmetic uses [`u64`]. Host indexing uses explicit checked conversions to [`usize`]. Callers
//! establish encoded-field limits, allocation ownership, and resource bounds separately.

mod byte_length;
mod chunk_slot;
mod element;
mod host;

pub use byte_length::{ByteLength, NonZeroByteLength};
pub use chunk_slot::{ChunkSlot, SlotCount};
pub use element::{ElementCount, ElementSize, NonZeroElementCount};

#[cfg(test)]
mod tests;
