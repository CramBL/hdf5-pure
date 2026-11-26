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

/// Borrows a compound member's name and datatype and copies its byte offset.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{__private, CompoundMember, Datatype, DatatypeMemberName, FixedPointType};
///
/// let member = CompoundMember::new(
///     DatatypeMemberName::from(c"field"), 4, Datatype::FixedPoint(FixedPointType::u16()),
/// );
/// let (name, offset, datatype) = __private::compound_member_parts(&member);
/// assert_eq!(name.as_bytes(), b"field");
/// assert_eq!(offset, 4);
/// assert_eq!(datatype, &Datatype::FixedPoint(FixedPointType::u16()));
/// ```
pub fn compound_member_parts(member: &CompoundMember) -> (&DatatypeMemberName, u64, &Datatype) {
    let CompoundMember {
        name,
        byte_offset,
        datatype,
    } = member;
    (name, *byte_offset, datatype)
}
