//! Modern compound datatype message fixtures.
//!
//! The layout is defined in "The Datatype Message" of the [format specification,
//! version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use super::Class;

/// Builds a compound message using the property layout of versions 3 through 5.
///
/// Accepts unchecked header fields, names, offsets, and child encodings for malformed
/// fixtures. The count in `flags` is independent of `members`. Appends a NUL terminator
/// to each name and truncates each offset to the low bytes selected by `width`.
///
/// # Examples
///
/// ```
/// use test_util::datatype::{self, containers::{compound, CompoundField, MemberOffset}};
///
/// let child = datatype::fixed_string(1);
/// let bytes = compound(3, [1, 0, 0], 1, MemberOffset::One, &[CompoundField {
///     name: b"x",
///     offset: 0,
///     datatype: &child,
/// }]);
/// assert_eq!(&bytes[..11], &[0x36, 1, 0, 0, 1, 0, 0, 0, b'x', 0, 0]);
/// assert_eq!(&bytes[11..], child);
/// ```
pub fn compound(
    version: u8,
    flags: [u8; 3],
    size: u32,
    width: MemberOffset,
    members: &[CompoundField<'_>],
) -> Vec<u8> {
    let mut bytes = super::header(Class::COMPOUND, version, flags, size);
    for CompoundField {
        name,
        offset,
        datatype,
    } in members
    {
        bytes.extend_from_slice(name);
        bytes.push(0);
        bytes.extend_from_slice(&offset.to_le_bytes()[..width.get()]);
        bytes.extend_from_slice(datatype);
    }
    bytes
}

/// A compound member's raw property fields.
pub struct CompoundField<'a> {
    /// The name bytes without the appended NUL terminator.
    pub name: &'a [u8],
    /// The byte offset from the start of the compound element.
    pub offset: u64,
    /// The complete encoded child description.
    pub datatype: &'a [u8],
}
/// The width of a compound member's byte-offset field.
#[derive(Clone, Copy, Debug)]
pub enum MemberOffset {
    /// A four-byte offset.
    Four,
    /// A one-byte offset.
    One,
    /// A three-byte offset.
    Three,
    /// A two-byte offset.
    Two,
}

impl MemberOffset {
    /// Returns the number of bytes in the offset field.
    pub const fn get(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
        }
    }
}
