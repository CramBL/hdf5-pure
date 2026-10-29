//! Free-space implementation types shared with the workspace.
//!
//! This module is outside the crate's compatibility guarantee and may change in any release.

pub use crate::admission::Release;
pub use crate::admission::TrackedSpace;
pub use crate::aggregator::AGGREGATE_BLOCK_SIZE;
pub use crate::aggregator::AggregationAllocation;
pub use crate::aggregator::AggregationError;
pub use crate::aggregator::AggregationRelease;
pub use crate::aggregator::ClassicAggregators;
pub use crate::allocation::AllocationClass;
pub use crate::extent::Extent;
pub use crate::list::FreeList;
pub use crate::list::trailing_run_start;
pub use crate::paged::FreeClass;
pub use crate::paged::PageTransition;
pub use crate::paged::PageType;
pub use crate::paged::PagedAllocationSnapshot;
pub use crate::paged::PagedAppendSnapshot;
pub use crate::paged::PagedPostFree;
pub use crate::paged::PagedSections;
pub use crate::paged::PagedTail;
pub use crate::persistence::ManagerKind;
pub use crate::persistence::ManagerSections;
pub use crate::persistence::TRAILING_RESERVE_TAILS;
pub use crate::session::FlatSpace;
pub use crate::session::PagedSpace;
pub use crate::session::SessionAppendSnapshot;
pub use crate::session::SessionSpace;
pub use crate::session::SessionSpaceSnapshot;
