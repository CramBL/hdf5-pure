//! Dense attribute messages, fractal heaps, and version 2 B-tree indexes.
//!
//! Plans retain independent file address and length widths. Attribute messages are paired
//! with creation indexes before heap and index planning.

mod attributes;
mod encode;
mod fixup;
mod geometry;
mod placement;
mod plan;
#[cfg(test)]
#[path = "tests/support.rs"]
mod test_support;

pub use attributes::DenseAttrCreationOrder;
pub use encode::{DenseAttrBlob, build_dense_attrs};
pub use fixup::{DenseAttrFixup, PreparedDenseAttrs};
pub use placement::{DenseAttrBlock, DenseAttrPlacement};
pub use plan::{
    DENSE_ATTR_MAX_MANAGED_OBJECT, DenseAttrBlockKind, DenseAttrPlan, dense_attrs_plan,
};
