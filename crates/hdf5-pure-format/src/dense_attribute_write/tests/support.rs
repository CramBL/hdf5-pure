use alloc::vec;

use hdf5_pure_core::{CharacterSet, DatatypeByteOrder, FixedPointType, StringPadding};

use crate::attribute_message::AttributeMessage;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::shared_message::DatatypeLocation;

pub(super) fn canonical_attribute(name: &str, len: u32) -> AttributeMessage {
    AttributeMessage {
        name: name.into(),
        datatype: Datatype::String {
            size: len,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        },
        dataspace: Dataspace::scalar(),
        raw_data: vec![b'y'; len as usize],
        datatype_location: DatatypeLocation::Inline,
    }
}

pub(super) fn metadata_attribute(name: &str, value: i64) -> AttributeMessage {
    AttributeMessage {
        name: name.into(),
        datatype: Datatype::FixedPoint(
            FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap(),
        ),
        dataspace: Dataspace::scalar(),
        raw_data: value.to_le_bytes().to_vec(),
        datatype_location: DatatypeLocation::Inline,
    }
}
