//! Test helpers that read or write a file through `hdf5-pure`'s own API, and through the
//! reference C library behind the `hdf5` feature.

#[cfg(feature = "hdf5")]
pub mod absence;
#[cfg(feature = "hdf5")]
pub mod file;
pub mod paged;
