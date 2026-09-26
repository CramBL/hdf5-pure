//! The portable shared types of the stable `hdf5-pure` API.
//!
//! This crate owns the types whose identity the workspace crates share, and `hdf5-pure`
//! re-exports them. Its documented public API is compatibility-checked and follows the same
//! versioning as `hdf5-pure`. Most users depend on `hdf5-pure`, which is the recommended entry
//! point. The hidden `__private` module is not part of that API.

#![cfg_attr(not(test), no_std)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod dataspace;
mod datatype;
mod display;
mod error;
mod message_type;

#[doc(hidden)]
pub mod __private;

pub use dataspace::MaxExtent;
pub use datatype::CharacterSet;
pub use datatype::CompoundMember;
pub use datatype::Datatype;
pub use datatype::EnumMember;
pub use datatype::ReferenceType;
pub use datatype::StringPadding;
pub use datatype::byte_order::DatatypeByteOrder;
pub use datatype::layout::FixedPointLayout;
pub use datatype::layout::FloatingPointLayout;
pub use error::FormatError;
pub use error::OBJECT_HEADER_MESSAGE_MAX;
pub use message_type::MessageType;
