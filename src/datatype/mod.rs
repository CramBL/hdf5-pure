pub use hdf5_pure_core::CharacterSet;
pub use hdf5_pure_core::CompoundMember;
pub use hdf5_pure_core::Datatype;
pub use hdf5_pure_core::ReferenceType;
pub use hdf5_pure_core::StringPadding;

pub(crate) use hdf5_pure_core::__private;
pub(crate) use hdf5_pure_format::element_size_usize;

pub(crate) fn datatype_class_name(datatype: &Datatype) -> &'static str {
    match datatype {
        Datatype::FixedPoint { .. } => "FixedPoint",
        Datatype::FloatingPoint { .. } => "FloatingPoint",
        Datatype::String { .. } => "String",
        Datatype::Time { .. } => "Time",
        Datatype::BitField { .. } => "BitField",
        Datatype::Opaque { .. } => "Opaque",
        Datatype::Compound { .. } => "Compound",
        Datatype::Reference { .. } => "Reference",
        Datatype::Enumeration { .. } => "Enumeration",
        Datatype::VariableLength { .. } => "VariableLength",
        Datatype::Array { .. } => "Array",
        _ => "Unknown",
    }
}

#[cfg(feature = "std")]
pub use hdf5_pure_format::class_may_hold_object_address;
#[cfg(feature = "std")]
pub use hdf5_pure_format::datatype_holds_file_address;
#[cfg(feature = "std")]
pub use hdf5_pure_format::datatype_holds_object_address;
#[cfg(feature = "std")]
pub use hdf5_pure_format::embedded_reference_slots;
#[cfg(feature = "std")]
pub use hdf5_pure_format::stored_object_references;

pub(crate) mod byte_order;
pub(crate) mod layout;
pub(crate) mod numeric;
