//! The version 2 B-tree, fractal heap, and shared message structures, for `hdf5-pure`.
//!
//! `hdf5-pure` walks version 2 B-trees, reads fractal heap objects and shared message indexes, and
//! writes dense attribute storage with the parsers and the writers this module exports.
//!
//! This module is outside the crate's compatibility guarantee and may change in any release.

pub use crate::btree_v2::BTreeV2Header;
pub use crate::btree_v2::BTreeV2NodeInfo;
pub use crate::btree_v2::BTreeV2Record;
pub use crate::btree_v2::parse_btree_v2_internal_child_pointers;
pub use crate::btree_v2::parse_btree_v2_leaf_records;
pub use crate::btree_v2_write::BTREE_V2_NODE_SIZE;
pub use crate::btree_v2_write::BTreeV2Image;
pub use crate::btree_v2_write::BTreeV2Plan;
pub use crate::btree_v2_write::btree_v2_header_size;
pub use crate::fractal_heap::FractalHeapChild;
pub use crate::fractal_heap::FractalHeapHeader;
pub use crate::fractal_heap::FractalHeapIdType;
pub use crate::fractal_heap::HugeObjectReference;
pub use crate::fractal_heap_write::ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES;
pub use crate::fractal_heap_write::ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE;
pub use crate::fractal_heap_write::ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;
pub use crate::fractal_heap_write::ATTRIBUTE_HEAP_MAX_HEAP_SPACE;
pub use crate::fractal_heap_write::ATTRIBUTE_HEAP_START_ROOT_ROWS;
pub use crate::fractal_heap_write::ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE;
pub use crate::fractal_heap_write::ATTRIBUTE_HEAP_TABLE_WIDTH;
pub use crate::fractal_heap_write::AttributeHeapPlan;
pub use crate::fractal_heap_write::AttributeHeapPlanError;
pub use crate::fractal_heap_write::attribute_heap_max_managed_object;
pub use crate::sohm::SharedMessageTableMessage;
pub use crate::sohm::SohmIndexHeader;
pub use crate::sohm::SohmIndexKind;
pub use crate::sohm::SohmLocation;
pub use crate::sohm::SohmRecord;
pub use crate::sohm::SohmTable;
pub use crate::sohm::parse_sohm_list;
pub use crate::sohm::sohm_list_len;
