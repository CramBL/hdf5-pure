//! In-place compaction of a global heap collection.
//!
//! After collecting the member metadata with [`GlobalHeapMemberCollector`], call
//! [`GlobalHeapMembers::compact`] to plan the record moves.

mod collector;
mod plan;

pub use collector::{GlobalHeapMemberCollector, GlobalHeapMembers};
pub use plan::{GlobalHeapCompaction, GlobalHeapFreeSpace, GlobalHeapMove};

#[cfg(test)]
mod tests;
