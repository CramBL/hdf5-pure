use hdf5_pure_core::{Datatype, ReferenceType};

use super::header::{
    ARRAY_CLASS, COMPOUND_CLASS, DATATYPE_CLASS_MASK, ENUMERATION_CLASS, Header, REFERENCE_CLASS,
    VARIABLE_LENGTH_CLASS,
};
use super::message;
use crate::error::FormatError;

/// Returns `true` if a datatype of this encoded class can hold an object
/// address, based only on the first byte of a datatype message.
///
/// A `false` result lets the caller skip parsing the datatype.
/// For `true`, use [`datatype_holds_object_address`] after parsing to determine whether
/// the datatype contains a reference. For example, a compound of two integers returns `true`
/// here and `false` after parsing.
pub fn class_may_hold_object_address(class_and_version: u8) -> bool {
    matches!(
        class_and_version & DATATYPE_CLASS_MASK,
        COMPOUND_CLASS | REFERENCE_CLASS | ENUMERATION_CLASS | VARIABLE_LENGTH_CLASS | ARRAY_CLASS
    )
}

/// Returns `true` if `dt` contains an object address anywhere in its structure.
///
/// An object address is an object reference or a dataset-region reference. This
/// function searches compound members, array entries, enumeration bases, and
/// variable-length contents.
///
/// [`embedded_reference_slots`](super::slots::embedded_reference_slots) describes the positions
/// of eight-byte object references stored directly in an element. The search here includes
/// references in a variable-length sequence's heap payload.
pub fn datatype_holds_object_address(dt: &Datatype) -> bool {
    match dt {
        Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_object_address(&m.datatype)),
        Datatype::Array { base_type, .. } => datatype_holds_object_address(base_type),
        Datatype::Enumeration(datatype) => datatype_holds_object_address(datatype.base_type()),
        Datatype::VariableLength(kind) => datatype_holds_object_address(kind.base_type()),
        Datatype::Complex(_) => false,
        _ => false,
    }
}

/// Returns `true` if an element of `dt` holds a file address at any depth.
///
/// A variable-length element and a dataset region reference each hold the address of a global
/// heap collection and an index into it, and an object reference holds the address of an object
/// header. The search descends into compound members, array elements, and enumeration bases.
///
/// Use [`datatype_holds_object_address`] to check for object and dataset-region references
/// specifically.
pub fn datatype_holds_file_address(dt: &Datatype) -> bool {
    match dt {
        Datatype::VariableLength(_) | Datatype::Reference { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|m| datatype_holds_file_address(&m.datatype)),
        Datatype::Array { base_type, .. } => datatype_holds_file_address(base_type),
        Datatype::Enumeration(datatype) => datatype_holds_file_address(datatype.base_type()),
        Datatype::Complex(_) => false,
        _ => false,
    }
}

/// Decodes a legacy object or dataset-region datatype from its common header.
///
/// Retains the declared element size.
///
/// # Errors
///
/// Returns [`FormatError::InvalidReferenceType`] for a kind other than legacy object or dataset
/// region, or [`FormatError::ZeroSizedDatatype`] for a zero element size.
pub(super) fn decode_reference(header: &Header) -> Result<Datatype, FormatError> {
    // `H5O__dtype_decode_helper` reads legacy kinds at every supported datatype message version
    // (`H5Odtype.c`, HDF5 2.2.0).
    let ref_type = match header.flags[0] & REFERENCE_KIND_MASK {
        OBJECT_REFERENCE_KIND => ReferenceType::Object,
        REGION_REFERENCE_KIND => ReferenceType::DatasetRegion,
        kind => return Err(FormatError::InvalidReferenceType(kind)),
    };
    Ok(Datatype::Reference {
        size: message::positive_size(header.class.code(), header.size)?.get(),
        ref_type,
    })
}

// "The Datatype Message" in format specification version 4.0 assigns bits 0 through 3
// of the reference class bit field to the reference kind.
const REFERENCE_KIND_MASK: u8 = 0x0f;

/// The object-reference kind for datatype message versions 1 through 3.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const OBJECT_REFERENCE_KIND: u8 = 0;
/// The dataset-region reference kind for datatype message versions 1 through 3.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const REGION_REFERENCE_KIND: u8 = 1;
