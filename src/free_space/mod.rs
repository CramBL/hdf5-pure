//! The free list of a read-write session, which records the space its commits vacate.
//!
//! A commit vacates the object headers it supersedes and the blocks of the objects it deletes.
//! Under a strategy with free-space managers, `H5F_FSPACE_STRATEGY_FSM_AGGR` or
//! `H5F_FSPACE_STRATEGY_PAGE`, the session records the vacated extents in a [`FreeList`] and
//! writes a later object into a free region that fits it. It records an extent at least as long as
//! the file's threshold, the smallest free-space section the managers track, or one that merges
//! into recorded space ([`TrackedSpace`]), which for a [`FreeList`] is one that adjoins a region.
//!
//! A session on a file without persistence starts with an empty list. On a file created with
//! `persist = true`, [`File::open_rw`](crate::File::open_rw) seeds the list from the file's
//! free-space managers, and each commit writes the list back to them.

mod admission;
mod extent;
mod list;

#[cfg(test)]
pub(crate) use admission::Release;
pub(crate) use admission::TrackedSpace;
pub(crate) use extent::Extent;
pub(crate) use list::{FreeList, trailing_run_start};
