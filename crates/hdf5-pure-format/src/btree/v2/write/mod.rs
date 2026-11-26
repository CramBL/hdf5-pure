//! Layout and serialization of version 2 B-trees over sorted records.
//!
//! [`BTreeV2Plan`] places leaves at one depth and bounds each node's record count by its capacity.
//! A caller chooses the representation of an empty tree through the plan's constructor.

mod encode;
mod placement;
mod plan;
mod prepared;

pub use encode::BTreeV2Image;
pub use placement::BTreeV2Placement;
pub use plan::{BTREE_V2_NODE_SIZE, BTreeV2Plan, BTreeV2WriteDialect};
