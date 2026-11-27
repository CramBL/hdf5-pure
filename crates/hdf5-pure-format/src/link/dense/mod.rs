//! Relocatable dense group links in a managed fractal heap and version 2 B-tree indexes.
//!
//! [`DenseLinkPlan`] retains encoded links and lays out their heap and indexes before a caller
//! allocates file space. Serialization uses stored file addresses relative to the superblock's
//! base address.

mod error;
mod index;
mod placement;
mod plan;
#[cfg(test)]
mod tests;

pub use error::DenseLinkPlanError;
#[cfg(test)]
pub(crate) use index::NAME_RECORD_LEN;
pub use placement::DenseLinkPlacement;
pub use plan::{DenseLinkImage, DenseLinkPlan};
