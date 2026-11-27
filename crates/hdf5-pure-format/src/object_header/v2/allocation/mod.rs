//! Chunk planning and encoding for version 2 object-header allocations.

mod chunk;
mod geometry;
mod placement;

pub use geometry::ObjectHeaderAllocationGeometry;
pub use placement::ObjectHeaderAllocation;

#[cfg(test)]
mod tests;
