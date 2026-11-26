use hdf5_pure_core::FormatError;

use super::super::header;
use super::super::message::version::DATATYPE_VERSION_ONE;
use super::super::semantic;

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
