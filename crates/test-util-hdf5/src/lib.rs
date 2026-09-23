//! Test helpers that read or write a file through `hdf5-pure`'s own API, and through the
//! reference C libraries behind the `hdf5` and `__matio` features.

#[cfg(feature = "hdf5")]
pub mod absence;
#[cfg(feature = "hdf5")]
pub mod file;
pub mod lock;
pub mod paged;
