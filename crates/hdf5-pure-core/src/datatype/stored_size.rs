use super::definition::Datatype;
use crate::{FormatError, OffsetWidth};

/// The offset widths of variable-length descriptors in a datatype description.
///
/// The element size depends on the outermost descriptor's width for a variable-length datatype
/// or an array with a variable-length element. The writer also checks widths deeper in a
/// decoded enumeration's base to preserve the layout of its stored values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DatatypeOffsets {
    /// The descriptor width used to compute the stored element size, if one applies.
    element: Option<OffsetWidth>,
    /// The set of descriptor widths used anywhere in the description.
    used: u8,
}

impl DatatypeOffsets {
    /// Adds a descriptor width to the widths from its base and uses it for element sizing.
    pub const fn descriptor(width: OffsetWidth, base: Self) -> Self {
        Self {
            element: Some(width),
            used: base.used | width.get(),
        }
    }

    /// Includes the widths from a compound member. The compound has a declared element size.
    pub fn include_field(&mut self, field: Self) {
        self.used |= field.used;
    }

    /// Keeps the widths from the base. The enumeration has a declared element size.
    pub const fn in_enumeration(self) -> Self {
        Self {
            element: None,
            used: self.used,
        }
    }

    /// Checks that the stored descriptors use the destination's offset width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::EnumBaseOffsetWidthMismatch`] with the smallest stored width
    /// that differs from `destination`, if any.
    pub fn check_destination(self, destination: OffsetWidth) -> Result<(), FormatError> {
        for source in [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight] {
            if source != destination && self.used & source.get() != 0 {
                return Err(FormatError::EnumBaseOffsetWidthMismatch {
                    source_offset_size: source.get(),
                    destination_offset_size: destination.get(),
                });
            }
        }
        Ok(())
    }

    /// Computes the stored element size with the descriptor width recorded for this datatype.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the required descriptor width is absent, or
    /// [`FormatError::LengthTooLarge`] if an array size exceeds the datatype size field.
    pub(super) fn stored_type_size(self, datatype: &Datatype) -> Result<u32, FormatError> {
        stored_type_size_with_offsets(datatype, self.element)
    }
}

/// Returns the size in bytes of one stored element, including zero-sized types.
///
/// Uses `offset_width` for variable-length descriptors and computes array sizes recursively.
/// Uses [`Datatype::type_size`] for other types, including the retained width of an enumeration.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte datatype
/// size field.
pub fn stored_type_size(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<u32, FormatError> {
    stored_type_size_with_offsets(datatype, Some(offset_width))
}

/// Returns the stored element size in bytes.
///
/// Supply `offset_width` for a variable-length datatype or an array whose innermost element
/// is variable-length. Uses the declared element size for compounds and enumerations.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if a required offset width is absent, or
/// [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte datatype size field.
fn stored_type_size_with_offsets(
    datatype: &Datatype,
    offset_width: Option<OffsetWidth>,
) -> Result<u32, FormatError> {
    match datatype {
        Datatype::VariableLength(_) => {
            let offset_width = offset_width.ok_or_else(|| {
                FormatError::Internal("missing the stored variable-length descriptor width".into())
            })?;
            Ok(variable_length_descriptor_size(offset_width))
        }
        Datatype::Array {
            base_type,
            dimensions,
        } => array_type_size(
            stored_type_size_with_offsets(base_type, offset_width)?,
            dimensions,
        ),
        _ => Ok(datatype.type_size()),
    }
}

/// Computes the stored array size in bytes, returning zero if any dimension is zero.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if the size exceeds the four-byte datatype size field.
pub fn array_type_size(base_size: u32, dimensions: &[u32]) -> Result<u32, FormatError> {
    if dimensions.contains(&0) {
        return Ok(0);
    }
    dimensions.iter().try_fold(base_size, |size, &dimension| {
        size.checked_mul(dimension)
            .ok_or(FormatError::LengthTooLarge {
                length: u64::from(size) * u64::from(dimension),
                length_size: DATATYPE_SIZE_BYTES,
            })
    })
}

/// Returns the size in bytes of a sequence count and a global heap ID at `offset_width`.
pub const fn variable_length_descriptor_size(offset_width: OffsetWidth) -> u32 {
    SEQUENCE_LENGTH_SIZE + offset_width.get() as u32 + HEAP_INDEX_SIZE
}

// `H5T__vlen_disk_write` writes this count before the heap ID (`H5Tvlen.c`, HDF5 2.2.0).
const SEQUENCE_LENGTH_SIZE: u32 = 4;
// "Global Heap" of the format specification, version 4.0, defines the object index in the heap ID.
const HEAP_INDEX_SIZE: u32 = 4;
// "The Datatype Message" of the format specification, version 4.0, defines the four-byte size field.
const DATATYPE_SIZE_BYTES: u8 = 4;
