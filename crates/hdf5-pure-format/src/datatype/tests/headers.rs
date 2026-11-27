use hdf5_pure_core::FormatError;

use super::super::header;
use super::super::message::DatatypeMessage;
use super::super::message::version::DATATYPE_VERSION_ONE;
use super::super::semantic;
use crate::codec::{DecodeFrame, DecodePrefix};

#[test]
fn rejects_zero_element_size_in_the_header() {
    let buf = header::build_datatype_header(3, DATATYPE_VERSION_ONE, [0x01, 0, 0], 0);
    assert_eq!(
        semantic::parse_datatype(&buf).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 3 }
    );
}

#[test]
fn rejects_unknown_class_in_the_header() {
    let buf = header::build_datatype_header(13, DATATYPE_VERSION_ONE, [0, 0, 0], 4);
    let err = semantic::parse_datatype(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidDatatypeClass(13));
}

#[test]
fn rejects_a_truncated_header() {
    let buf = [0u8; 4];
    let err = semantic::parse_datatype(&buf).unwrap_err();
    assert_eq!(
        err,
        FormatError::UnexpectedEof {
            expected: 8,
            available: 4
        }
    );
}

#[test]
fn retained_projection_preserves_the_description_extent() {
    let bytes = [0x10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0, 0xff];
    let prefix = DatatypeMessage::decode_prefix(&bytes, ()).unwrap();
    let projected = prefix.try_map(|message| message.to_datatype()).unwrap();
    let frame = DatatypeMessage::decode_frame(&bytes[..12], ()).unwrap();
    assert_eq!(projected.into_parts(), (frame.to_datatype().unwrap(), 12));
}
