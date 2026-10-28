//! Native fixtures and observations shared by the crosscheck suites.

pub(crate) mod array_index;
pub(crate) mod c_object;
#[cfg(feature = "__hdf5-1.10")]
#[path = "../../fixtures/dense_attributes/protocol.rs"]
pub(crate) mod dense_attributes;
pub(crate) mod file_space;
pub(crate) mod hard_link_graphs;
