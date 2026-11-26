//! Sizing and writing properties that contain nested datatype descriptions.

use core::num::NonZeroU32;

use hdf5_pure_core::__private;
use hdf5_pure_core::__private::DatatypeOffsets;
use hdf5_pure_core::{CharacterSet, EnumMember, FormatError, StringPadding};

use super::output::{NodeHeader, Output};
use super::prepared::Preparation;
use super::source::{CompoundMemberSource, CompoundMembers, Description};
use crate::convert::Narrow;
use crate::datatype::header::{
    ARRAY_CLASS, COMPLEX_CLASS, COMPOUND_CLASS, ENUMERATION_CLASS, VARIABLE_LENGTH_CLASS,
};
use crate::datatype::legacy_member::DIMENSION_BLOCK_SIZE;
use crate::datatype::message::DatatypeVersion;
use crate::datatype::message::compound_layout::{CompoundFields, CompoundLayout};
use crate::datatype::message::decode::COMPLEX_HOMOGENEOUS_BIT;
use crate::datatype::message::name_layout::{NAME_ALIGNMENT, NameLayout};
use crate::datatype::message::representation;
use crate::datatype::nesting::MessageDepth;
use crate::datatype::variable_length::{
    VARIABLE_LENGTH_PADDING_SHIFT, VARIABLE_LENGTH_SEQUENCE, VARIABLE_LENGTH_STRING,
};
use crate::datatype::{stored_size, string};

/// Borrowed container fields shared by canonical and retained encoding.
pub(super) enum ContainerProperties<'a> {
    Compound {
        size: u32,
        members: CompoundMembers<'a>,
        fields: CompoundFields,
    },
    Enumeration {
        size: u32,
        base_type: Description<'a>,
        members: &'a [EnumMember],
        name_layout: NameLayout,
        offsets: DatatypeOffsets,
    },
    VariableLength {
        base_type: Description<'a>,
        /// Padding and character set for a string, or none for a sequence.
        string: Option<(&'a StringPadding, &'a CharacterSet)>,
    },
    Array {
        base_type: Description<'a>,
        dimensions: &'a [u32],
    },
    Complex {
        component: Description<'a>,
        /// The checked semantic size, or none when preparation must compute the size.
        size: Option<u32>,
    },
}

impl ContainerProperties<'_> {
    /// Checks container sizes at the destination offset width and returns the stored element size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundFieldOutOfBounds`] if a member extends past its element,
    /// or [`FormatError::CompoundFieldOverlap`] if two nonempty member ranges overlap.
    ///
    /// Returns [`FormatError::EnumBaseSizeMismatch`] if an enumeration's value width differs
    /// from its base's stored size at the destination offset width. If the sizes agree, returns
    /// [`FormatError::EnumBaseOffsetWidthMismatch`] if a descriptor in the decoded base uses
    /// another offset width.
    ///
    /// Propagates errors from preparing nested descriptions with [`Description::prepare`],
    /// checking counts and element sizes, and recording lengths and sizes with
    /// [`Preparation::add`] and [`Preparation::record_size`].
    pub(super) fn prepare(
        &self,
        preparation: &mut Preparation,
        child_depth: Option<MessageDepth>,
    ) -> Result<u32, FormatError> {
        match self {
            Self::Compound {
                size,
                members,
                fields,
            } => {
                check_member_count(members.len(), child_depth.is_some())?;
                members.prepare(preparation, *size, *fields, child_depth)?;
                Ok(*size)
            }
            Self::Enumeration {
                size,
                base_type,
                members,
                name_layout,
                offsets,
            } => {
                check_member_count(members.len(), child_depth.is_some())?;
                let parent_size = base_type.prepare(preparation, child_depth)?;
                representation::check_enum_base_layout(
                    *size,
                    parent_size,
                    *offsets,
                    preparation.offset_width(),
                )?;
                for member in *members {
                    preparation.add(name_layout.field_len(member.name.as_bytes().len()))?;
                    preparation.add(member.value.len())?;
                }
                Ok(parent_size)
            }
            Self::VariableLength { base_type, .. } => {
                base_type.prepare(preparation, child_depth)?;
                Ok(preparation.descriptor_size())
            }
            Self::Array {
                base_type,
                dimensions,
            } => {
                let rank: u8 = dimensions.len().narrow_or_else(|| {
                    if child_depth.is_some() {
                        FormatError::Internal("a checked array rank exceeds u8".into())
                    } else {
                        FormatError::ValueTooLargeForRepresentation {
                            value: dimensions.len().to_u64(),
                            target: "u8",
                        }
                    }
                })?;
                preparation.add(1 + usize::from(rank) * size_of::<u32>())?;
                let parent_size = base_type.prepare(preparation, child_depth)?;
                let size = __private::array_type_size(parent_size, dimensions)?;
                preparation.record_size(size)?;
                Ok(size)
            }
            Self::Complex { component, size } => {
                let component_size = component.prepare(preparation, child_depth)?;
                match size {
                    Some(size) => Ok(*size),
                    None => {
                        let size = component_size
                            .checked_mul(2)
                            .ok_or(FormatError::ComplexSizeOverflow { component_size })?;
                        preparation.record_size(size)?;
                        Ok(size)
                    }
                }
            }
        }
    }

    /// Writes the prepared properties and returns the header for the container.
    pub(super) fn write(
        &self,
        output: &mut Output<'_, impl FnMut(usize, &[u8])>,
        preserved: Option<DatatypeVersion>,
    ) -> NodeHeader {
        let (class, canonical_minimum, child_version, flags, size) = match self {
            Self::Compound {
                size,
                members,
                fields,
            } => {
                let width = fields.offset_width();
                let name_layout = fields.name_layout();
                let mut version = DatatypeVersion::Three;
                for index in 0..members.len() {
                    let CompoundMemberSource {
                        name,
                        offset,
                        child,
                        shape,
                    } = members.member(index);
                    let name = name.as_bytes();
                    output.write(name);
                    output.write(&[0; NAME_ALIGNMENT][..name_layout.suffix_len(name.len())]);
                    output.write(&offset.to_le_bytes()[..width]);
                    if let Some(shape) = shape {
                        output.write(&shape.to_bytes());
                    }
                    version = version.max(child.write(output).version);
                }
                (
                    COMPOUND_CLASS,
                    DatatypeVersion::Three,
                    version,
                    member_flags(members.len()),
                    *size,
                )
            }
            Self::Enumeration {
                size,
                base_type,
                members,
                name_layout,
                offsets: _,
            } => {
                let child = base_type.write(output);
                debug_assert_eq!(*size, child.size);
                for member in *members {
                    let name = member.name.as_bytes();
                    output.write(name);
                    output.write(&[0; NAME_ALIGNMENT][..name_layout.suffix_len(name.len())]);
                }
                for member in *members {
                    output.write(&member.value);
                }
                (
                    ENUMERATION_CLASS,
                    DatatypeVersion::Three,
                    child.version,
                    member_flags(members.len()),
                    child.size,
                )
            }
            Self::VariableLength { base_type, string } => {
                let child = base_type.write(output);
                let flags = match string {
                    Some((padding, charset)) => [
                        VARIABLE_LENGTH_STRING
                            | (string::padding_code(padding) << VARIABLE_LENGTH_PADDING_SHIFT),
                        string::charset_code(charset),
                        0,
                    ],
                    None => [VARIABLE_LENGTH_SEQUENCE, 0, 0],
                };
                (
                    VARIABLE_LENGTH_CLASS,
                    DatatypeVersion::One,
                    child.version,
                    flags,
                    output.descriptor_size(),
                )
            }
            Self::Array {
                base_type,
                dimensions,
            } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the prepared dimension count fits the one-byte rank field"
                )]
                let rank = dimensions.len() as u8;
                output.write(&[rank]);
                for &dimension in *dimensions {
                    output.write(&dimension.to_le_bytes());
                }
                let child = base_type.write(output);
                (
                    ARRAY_CLASS,
                    DatatypeVersion::Three,
                    child.version,
                    [0; 3],
                    output.next_size(),
                )
            }
            Self::Complex { component, size } => {
                let child = component.write(output);
                (
                    COMPLEX_CLASS,
                    DatatypeVersion::Five,
                    child.version,
                    [COMPLEX_HOMOGENEOUS_BIT, 0, 0],
                    size.unwrap_or_else(|| output.next_size()),
                )
            }
        };
        // `H5O__dtype_encode_helper` requires the enclosing version to be at least each
        // child's version (`H5Odtype.c`, HDF5 2.2.0).
        NodeHeader {
            class,
            version: preserved.unwrap_or(canonical_minimum.max(child_version)),
            flags,
            size,
        }
    }
}

fn check_member_count(count: usize, preserved: bool) -> Result<(), FormatError> {
    representation::member_count(count)
        .map(|_| ())
        .map_err(|error| {
            if preserved {
                FormatError::Internal("a checked datatype member count exceeds u16".into())
            } else {
                error
            }
        })
}

impl CompoundMembers<'_> {
    fn prepare(
        &self,
        preparation: &mut Preparation,
        size: u32,
        fields: CompoundFields,
        child_depth: Option<MessageDepth>,
    ) -> Result<(), FormatError> {
        let width = fields.offset_width();
        let name_layout = fields.name_layout();
        let mut layout = CompoundLayout::new(size);
        for index in 0..self.len() {
            let CompoundMemberSource {
                name,
                offset,
                child,
                shape,
            } = self.member(index);
            preparation.add(name_layout.field_len(name.as_bytes().len()))?;
            preparation.add(width)?;
            if shape.is_some() {
                preparation.add(DIMENSION_BLOCK_SIZE)?;
            }
            let field_size = match child.prepare(preparation, child_depth) {
                Ok(size) => size,
                Err(error) => {
                    // Check the canonical member's bounds before returning its preparation error.
                    if let Description::Canonical(datatype) = child {
                        let size =
                            stored_size::stored_type_size(datatype, preparation.offset_width())?;
                        layout.member_end(offset, size, || Ok(name.clone()))?;
                    }
                    return Err(error);
                }
            };
            if let Some(field_size) = NonZeroU32::new(field_size) {
                let field_size = match shape {
                    Some(shape) => shape.stored_size(field_size)?,
                    None => field_size,
                };
                layout.insert(offset, field_size, index, |index| {
                    self.member_name(index).cloned()
                })?;
            } else {
                layout.member_end(offset, field_size, || Ok(name.clone()))?;
            }
        }
        Ok(())
    }
}

/// Encodes a checked compound or enumeration member count in the low two class bit-field bytes.
///
/// "The Datatype Message" of the [format specification, version 4.0][spec] defines the count field.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
fn member_flags(count: usize) -> [u8; 3] {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the prepared member count fits its two-byte field"
    )]
    let [low, high] = (count as u16).to_le_bytes();
    [low, high, 0]
}
