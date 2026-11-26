use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroU32;

use super::definition::Datatype;
use super::layout::FixedPointType;
use super::member_name::DatatypeMemberName;
use super::stored_size::DatatypeOffsets;
use crate::FormatError;

/// A name and its encoded value in an enumeration datatype.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EnumMember {
    /// The stored name bytes.
    pub name: DatatypeMemberName,
    /// The encoded value bytes. [`EnumerationType::new`] checks their length against the base
    /// datatype's width.
    pub value: Vec<u8>,
}

impl EnumMember {
    /// Constructs a named member and stores its value bytes as supplied.
    pub fn new(name: DatatypeMemberName, value: Vec<u8>) -> Self {
        Self { name, value }
    }
}

/// An enumeration whose named values have the stored width of its base datatype.
///
/// [Construction](Self::new) requires a fixed-point base. Decoding retains the stored base,
/// including supported floating-point datatypes. In libhdf5, `H5Tenum_create` requires an integer
/// base (`H5Tenum.c`, HDF5 2.2.0). The libhdf5 message decoder accepts other parent classes
/// when the parent and enumeration sizes agree (`H5O__dtype_decode_helper` in `H5Odtype.c`,
/// HDF5 2.2.0).
///
/// The writer preserves the member names and value bytes. The offset width of each variable-length
/// descriptor in a decoded base must match the destination's offset width.
///
/// Class 8 of the datatype message is defined in "The Datatype Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub struct EnumerationType {
    base_type: Box<Datatype>,
    size: NonZeroU32,
    members: Vec<EnumMember>,
    offsets: DatatypeOffsets,
}

impl EnumerationType {
    /// Constructs an enumeration with a fixed-point base.
    ///
    /// Pass each value in the base type's byte order. Preserves the supplied bytes and
    /// member order, and accepts an empty member list.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::EnumMemberValueSize`] if a value has a different byte length than
    /// the base type. The error reports the first such member.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{Datatype, DatatypeMemberName, EnumMember, EnumerationType, FixedPointType};
    ///
    /// let base = FixedPointType::u16();
    /// let name = DatatypeMemberName::new("active")?;
    /// let member = EnumMember::new(name, 1_u16.to_le_bytes().to_vec());
    /// let enumeration = EnumerationType::new(base, vec![member])?;
    /// assert_eq!(enumeration.base_type(), &Datatype::FixedPoint(base));
    /// assert_eq!(enumeration.size().get(), 2);
    /// assert_eq!(enumeration.members()[0].name.as_bytes(), b"active");
    /// assert_eq!(enumeration.members()[0].value, [1, 0]);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn new(base_type: FixedPointType, members: Vec<EnumMember>) -> Result<Self, FormatError> {
        Self::from_parts(
            Datatype::FixedPoint(base_type),
            base_type.size(),
            members,
            DatatypeOffsets::default(),
        )
    }

    /// Pairs a base with its established size and checks each value against that size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::EnumMemberValueSize`] for the first value whose width differs
    /// from `size`.
    fn from_parts(
        base_type: Datatype,
        size: NonZeroU32,
        mut members: Vec<EnumMember>,
        offsets: DatatypeOffsets,
    ) -> Result<Self, FormatError> {
        if let Some(index) = members
            .iter()
            .position(|member| u32::try_from(member.value.len()) != Ok(size.get()))
        {
            let EnumMember { name, value } = members.swap_remove(index);
            return Err(FormatError::EnumMemberValueSize(
                name,
                size.get(),
                value.len(),
            ));
        }
        Ok(Self {
            base_type: Box::new(base_type),
            size,
            members,
            offsets,
        })
    }

    /// Returns the datatype that describes each member's encoded value.
    pub const fn base_type(&self) -> &Datatype {
        &self.base_type
    }

    /// Returns the stored size of one value in bytes.
    pub const fn size(&self) -> NonZeroU32 {
        self.size
    }

    /// Borrows the members in the order supplied at construction.
    pub fn members(&self) -> &[EnumMember] {
        &self.members
    }
}

/// Borrows an enumeration member's name and encoded value bytes.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{__private, DatatypeMemberName, EnumMember};
///
/// let member = EnumMember::new(DatatypeMemberName::from(c"active"), vec![1, 0]);
/// let (name, value) = __private::enum_member_parts(&member);
/// assert_eq!(name.as_bytes(), b"active");
/// assert_eq!(value, &[1, 0]);
/// ```
pub fn enum_member_parts(member: &EnumMember) -> (&DatatypeMemberName, &[u8]) {
    let EnumMember { name, value } = member;
    (name, value)
}

/// Constructs an enumeration whose value width matches the base's stored size.
///
/// Pass the expected width of each value in bytes and the offset widths of variable-length
/// descriptors in the base.
///
/// # Errors
///
/// Returns [`FormatError::EnumBaseSizeMismatch`] if `size` differs from the base's stored
/// size, or [`FormatError::EnumMemberValueSize`] for the first value whose width differs from `size`.
///
/// Returns [`FormatError::Internal`] if a descriptor width needed to compute the base size is
/// missing, or [`FormatError::LengthTooLarge`] if an array size exceeds the datatype size field.
pub fn enumeration_from_stored_parts(
    base_type: Datatype,
    size: NonZeroU32,
    offsets: DatatypeOffsets,
    members: Vec<EnumMember>,
) -> Result<EnumerationType, FormatError> {
    let base_size = offsets.stored_type_size(&base_type)?;
    if base_size != size.get() {
        return Err(FormatError::EnumBaseSizeMismatch {
            size: size.get(),
            base_size,
        });
    }
    EnumerationType::from_parts(base_type, size, members, offsets)
}

/// Returns the descriptor offset widths recorded when reading the enumeration's base.
///
/// The result is empty for an enumeration created with [`EnumerationType::new`].
pub const fn enumeration_base_offsets(datatype: &EnumerationType) -> DatatypeOffsets {
    datatype.offsets
}
