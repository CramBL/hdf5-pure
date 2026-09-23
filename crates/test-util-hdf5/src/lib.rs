//! Test helpers that read or write a file through `hdf5-pure`'s own API, and through the
//! reference C libraries behind the `hdf5` and `__matio` features.

#[cfg(feature = "hdf5")]
pub mod absence;
pub mod dataset;
#[cfg(feature = "hdf5")]
pub mod file;
pub mod lock;
pub mod mat_file;
pub mod paged;
pub mod session;
