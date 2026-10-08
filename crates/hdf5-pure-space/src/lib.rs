//! Pure HDF5 file-space allocation and free-space policy.
//!
//! This implementation crate owns extents, reusable-space sets, threshold admission, PAGE typing
//! and merge rules, session strategy state, and semantic persistence planning. It performs no file
//! I/O, owns no commit or superblock publication protocol, does not reclaim objects, and does not
//! encode or decode HDF5 free-space manager bytes. It always builds with `no_std` and `alloc`. It
//! does not model file access provided by the operating system.
//!
//! The project supports `hdf5-pure` as its public API entry point. Workspace crates use the hidden
//! [`__private`] module, whose contents may change in any release.

#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

#[doc(hidden)]
pub mod __private;

mod admission;
mod extent;
mod list;
mod paged;
mod persistence;
mod session;
