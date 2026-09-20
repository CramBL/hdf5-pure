#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! The object header message flags that make a decoder reject an object whose message type it
//! cannot name, against the reference C library: this crate reads the files `H5Fopen` reads and
//! rejects the objects it rejects.

mod always;
mod fixture;
mod for_write;
