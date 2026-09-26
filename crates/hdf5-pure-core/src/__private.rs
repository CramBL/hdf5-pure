//! Constructs shared member types for the format parser, and shares the display helpers.
//!
//! The member structs are non-exhaustive, so only this crate can construct
//! them with struct literals. The format parser calls these functions to build
//! members across the crate boundary.
//!
//! This module serves the workspace crates only. It is outside the crate's compatibility
//! guarantee and may change in any release.

use alloc::string::String;
use alloc::vec::Vec;

use crate::CompoundMember;
use crate::Datatype;
use crate::EnumMember;

pub use crate::display::DISPLAY_MAX_MEMBERS;
pub use crate::display::Dims;
pub use crate::display::EscapedName;
pub use crate::display::QuotedBytes;
pub use crate::display::write_elided;

/// Constructs a compound member from fields read from a datatype message.
pub fn compound_member(name: String, byte_offset: u64, datatype: Datatype) -> CompoundMember {
    CompoundMember {
        name,
        byte_offset,
        datatype,
    }
}

/// Constructs an enumeration member from fields read from a datatype message.
pub fn enum_member(name: String, value: Vec<u8>) -> EnumMember {
    EnumMember { name, value }
}
