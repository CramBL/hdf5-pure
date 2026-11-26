use alloc::string::String;

use super::Datatype;

/// A member of a compound datatype.
///
/// The struct is non-exhaustive.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset within the compound.
    pub byte_offset: u64,
    /// Member datatype.
    pub datatype: Datatype,
}
