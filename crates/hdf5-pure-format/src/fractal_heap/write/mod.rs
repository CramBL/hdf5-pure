//! Relocatable layouts and serialization for unfiltered managed fractal-heap blocks.
//!
//! Use [`ManagedHeapPlan`] to lay out objects and serialize their blocks at caller-supplied
//! addresses. The caller writes the heap header and encodes the object IDs.

mod encode;
mod parameters;
mod placement;
mod plan;

pub use parameters::{
    ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES, ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE,
    ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS, ATTRIBUTE_HEAP_MAX_HEAP_SPACE,
    ATTRIBUTE_HEAP_START_ROOT_ROWS, ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE, ATTRIBUTE_HEAP_TABLE_WIDTH,
    GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE, GROUP_HEAP_MAX_HEAP_SIZE_BITS, GROUP_HEAP_MAX_MANAGED_OBJECT,
    GROUP_HEAP_START_ROOT_ROWS, GROUP_HEAP_STARTING_BLOCK_SIZE, GROUP_HEAP_TABLE_WIDTH,
    attribute_heap_max_managed_object,
};
pub(crate) use parameters::{attribute_heap_direct_block_header, maximum_direct_block_size};
pub use placement::ManagedHeapPlacement;
pub use plan::{ManagedHeapPlan, ManagedHeapPlanError};
