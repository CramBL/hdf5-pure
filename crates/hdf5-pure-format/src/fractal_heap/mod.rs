//! Fractal heap headers, object IDs, managed-block layouts, and free-space records.

mod free_space;
mod geometry;
mod header;
mod id;
mod layout;
pub(crate) mod location;
#[cfg(test)]
mod test_support;

pub use free_space::FractalHeapFreeSection;
pub use free_space::FractalHeapFreeSpace;
pub use free_space::FractalHeapFreeSpaceManager;
pub use free_space::FractalHeapFreeSpaceSectionListRef;
pub use free_space::FractalHeapIndirectSection;
pub use free_space::FractalHeapRowSection;
pub use free_space::FractalHeapSectionClass;
pub use geometry::FractalHeapGeometry;
pub use geometry::FractalHeapLayoutError;
pub use header::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
pub use header::FractalHeapFiltering;
pub use header::FractalHeapHeader;
pub use header::FractalHeapHeaderFrame;
pub use header::FractalHeapHeaderLocator;
pub use header::FractalHeapHeaderRef;
pub use id::FilteredHugeObjectLocation;
pub use id::FractalHeapIdKind;
pub use id::FractalHeapIdLayout;
pub use id::FractalHeapIdView;
pub use id::HugeIndirectKey;
pub use id::HugeIndirectKeyEncoding;
pub use id::HugeObjectReference;
pub use id::TinyHeapIdEncoding;
pub use layout::FractalHeapChild;
pub use layout::FractalHeapLayout;
