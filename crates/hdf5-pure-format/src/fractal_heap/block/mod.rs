//! Caller-supplied regions for version 0 fractal-heap direct and indirect blocks.
//!
//! The caller resolves each stored address to a physical position before supplying a region.
//! Block prefixes and doubling-table geometry are defined in "Fractal Heap" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap

mod direct;
mod indirect;
mod layout;

pub use direct::{
    FractalHeapBlockFilters, FractalHeapDirectBlock, FractalHeapDirectBlockRef,
    FractalHeapDirectBlockView, FractalHeapObjectRef,
};
pub use indirect::{
    FractalHeapIndirectBlock, FractalHeapIndirectBlockRef, FractalHeapIndirectBlockView,
};

pub(crate) use layout::{CHECKSUM_LEN, DIRECT_SIGNATURE, INDIRECT_SIGNATURE, VERSION};
pub use layout::{FractalHeapBlock, FractalHeapBlockLayout};
