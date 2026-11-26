//! Parses the version 2 B-tree header and nodes and computes internal-node child-pointer widths.
//!
//! "Version 2 B-trees" of the [format specification, version 4.0][spec] defines the `BTHD` header,
//! `BTIN` internal nodes, and `BTLF` leaf nodes. The header's "Node Size" field is the size in
//! bytes of every B-tree node. The header itself has a separate variable-length layout whose
//! address and length fields use the superblock widths. [`BTreeV2NodeInfo`] computes the
//! record-count widths that the internal-node child-pointer layout leaves variable.
//!
//! A reader selects a huge-object tracker encoding with [`HugeObjectRecordContext`] and decodes
//! chunk-index client types 10 and 11 with [`BTreeV2ChunkRecordContext`]. A writer encodes
//! attribute-index records with [`AttributeRecord`].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2

mod geometry;
mod header;
mod integer;
mod node;
mod records;
pub(crate) mod write;

pub use geometry::BTreeV2NodeInfo;
pub use header::BTreeV2Header;
pub use header::btree_v2_header_size;
pub use node::BTreeV2CollectionRoot;
pub use node::BTreeV2Layout;
pub use node::BTreeV2Node;
pub use node::BTreeV2NodeView;
pub use records::AttributeRecord;
pub use records::BTREE_V2_ATTRIBUTE_CREATION_ORDER;
pub use records::BTREE_V2_ATTRIBUTE_NAME;
pub use records::BTREE_V2_CHUNK;
pub use records::BTREE_V2_FILTERED_CHUNK;
pub use records::BTREE_V2_FILTERED_HUGE_OBJECT;
pub use records::BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT;
pub use records::BTREE_V2_HUGE_OBJECT;
pub use records::BTREE_V2_HUGE_OBJECT_DIRECT;
pub use records::BTREE_V2_LINK_CREATION_ORDER;
pub use records::BTREE_V2_LINK_NAME;
pub use records::BTreeV2ChunkRecord;
pub use records::BTreeV2ChunkRecordContext;
pub use records::BTreeV2LinkNameIndex;
pub use records::BTreeV2LinkRecord;
pub use records::BTreeV2LinkRecordContext;
pub use records::BTreeV2Record;
pub use records::DenseAttributeRecord;
pub use records::DenseAttributeTarget;
pub use records::FilteredHugeObjectDirectRecord;
pub use records::FilteredHugeObjectRecord;
pub use records::HugeObjectDirectRecord;
pub use records::HugeObjectRecord;
pub use records::HugeObjectRecordContext;
pub use records::HugeObjectRecordDecoder;
pub use records::HugeObjectTrackerRecord;
pub use records::PrivateAttributeId;
pub use records::SharedAttributeId;
