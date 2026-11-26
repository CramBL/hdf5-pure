//! Parsing and serialization of HDF5 Datatype messages.

pub(crate) use hdf5_pure_core::{CharacterSet, Datatype};

pub use self::legacy_member::CompoundV1Shape;
pub use self::references::{
    class_may_hold_object_address, datatype_holds_file_address, datatype_holds_object_address,
};
pub use self::semantic::{
    parse_datatype, serialize_datatype, serialize_datatype_with_offset_width,
};
pub use self::slots::{EmbeddedReferenceSlots, embedded_reference_slots};
pub use self::stored_size::{
    element_size_usize, stored_element_size, stored_element_size_usize, stored_type_size,
};
pub use self::variable_length::datatype_holds_variable_length;

pub(crate) mod byte_order;
mod header;
pub(crate) mod layout;
mod legacy_member;
pub(crate) mod message;
mod opaque;
mod references;
mod semantic;
pub(crate) mod slots;
mod stored_size;
mod string;
mod variable_length;

#[cfg(test)]
mod tests;
