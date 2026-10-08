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
//!
//! [`SessionSpace`] owns the active runtime allocation strategy. Its PAGE state keeps typed free
//! lists, page transitions, threshold admission, whole-page promotion, and post-release EOA policy.
//! [`ManagerSections`] describes the manager contents persistence should write without assigning
//! wire slots, section classes, block addresses, or encoded lengths.

mod admission;
mod extent;
mod list;
mod paged;
mod persistence;
mod session;

#[cfg(test)]
pub(crate) use admission::Release;
pub(crate) use admission::TrackedSpace;
pub(crate) use extent::Extent;
pub(crate) use list::{FreeList, trailing_run_start};
#[cfg(test)]
pub(crate) use paged::PagedEdit;
pub(crate) use paged::{FreeClass, PageTransition, PageType, PagedPostFree};
pub(crate) use persistence::{ManagerKind, ManagerSections};
pub(crate) use session::{SessionSpace, SessionSpaceSnapshot};

/// Specifies the manager-tail lengths retained when releasing trailing free space.
pub(crate) const TRAILING_RESERVE_TAILS: u64 = 4;
