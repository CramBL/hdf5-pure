//! Test helpers that read or write a file through `hdf5-pure`'s own API, and through the
//! reference C libraries behind the `hdf5` and `__matio` features.

#[cfg(feature = "hdf5")]
pub mod absence;
#[cfg(feature = "__hdf5-1.10")]
pub mod creation_order;
pub mod dataset;
pub mod dense_attr;
#[cfg(feature = "hdf5")]
pub mod file;
pub mod lock;
pub mod mat_file;
pub mod paged;
pub mod session;
pub mod userblock;
