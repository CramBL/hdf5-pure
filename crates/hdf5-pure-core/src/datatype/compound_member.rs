use super::definition::Datatype;
use super::member_name::DatatypeMemberName;

/// A member of a compound datatype.
///
/// The struct is non-exhaustive.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CompoundMember {
    /// The field's name as bytes. Use [`DatatypeMemberName::as_str`] to interpret it as UTF-8.
    pub name: DatatypeMemberName,
    /// Byte offset within the compound.
    pub byte_offset: u64,
    /// Member datatype.
    pub datatype: Datatype,
}

impl CompoundMember {
    /// Creates a member at a byte offset from the start of the compound element.
    pub const fn new(name: DatatypeMemberName, byte_offset: u64, datatype: Datatype) -> Self {
        Self {
            name,
            byte_offset,
            datatype,
        }
    }
}
