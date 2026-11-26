use alloc::boxed::Box;
use alloc::vec;

use hdf5_pure_core::{Datatype, DatatypeByteOrder, FixedPointType, FormatError};

use super::super::stored_size;

#[test]
fn element_size_matches_fixed_point_size() {
    let dt = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    assert_eq!(dt.element_size().unwrap().get(), dt.type_size());
}

#[test]
fn element_size_rejects_an_array_with_a_zero_dimension() {
    let dt = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )),
        dimensions: vec![0, 4],
    };
    assert_eq!(dt.type_size(), 0);
    assert_eq!(
        dt.element_size().unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 10 }
    );
    assert_eq!(
        stored_size::element_size_usize(&dt).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 10 }
    );
}

#[test]
fn element_size_reports_the_class_of_a_zero_sized_type() {
    let dt = Datatype::Compound {
        size: 0,
        members: vec![],
    };
    assert_eq!(
        dt.element_size().unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 6 }
    );
}

#[test]
fn type_size_includes_array_dimensions() {
    let dt = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    assert_eq!(dt.type_size(), 4);

    let dt = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )),
        dimensions: vec![3, 4],
    };
    assert_eq!(dt.type_size(), 48);
}
