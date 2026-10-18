//! Test helpers that read or write a file through `hdf5-pure`'s own API, and through the
//! reference C libraries behind the `hdf5` and `__matio` features.

#[cfg(feature = "hdf5")]
pub mod absence;
pub mod aggregation;
pub mod attr;
#[cfg(feature = "__hdf5-1.10")]
pub mod creation_order;
pub mod dataset;
pub mod dense_attr;
#[cfg(feature = "hdf5")]
pub mod file;
pub mod file_builder;
pub mod file_space;
pub mod fixed_string;
pub mod fixtures;
pub mod global_heap_fixture;
#[cfg(feature = "__hdf5-1.10")]
pub mod group_filter;
pub mod lock;
pub mod mat_file;
pub mod paged;
pub mod reader;
pub mod session;
#[cfg(feature = "__hdf5-1.10")]
pub mod space_ownership;
pub mod userblock;
