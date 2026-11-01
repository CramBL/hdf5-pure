use hdf5_pure_core::{ComplexType, Datatype, DatatypeByteOrder, FloatingPointType, FormatError};
use rstest::rstest;

#[rstest]
#[case::binary32(
    FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian),
    8,
    "complex<f32>"
)]
#[case::binary64_be(
    FloatingPointType::ieee754_binary64(DatatypeByteOrder::BigEndian),
    16,
    "complex<f64 be>"
)]
fn native_complex_preserves_component_and_element_size(
    #[case] component: FloatingPointType,
    #[case] size: u32,
    #[case] display: &str,
) {
    let complex = ComplexType::new(component).unwrap();
    assert_eq!(complex.component(), component);
    assert_eq!(complex.size().get(), size);
    let datatype = Datatype::Complex(complex);
    assert_eq!(datatype.type_size(), size);
    assert_eq!(datatype.element_size().unwrap().get(), size);
    assert_eq!(datatype.to_string(), display);
}

#[test]
fn native_complex_rejects_element_size_overflow() {
    let component = FloatingPointType::from_layout(
        0x8000_0000,
        DatatypeByteOrder::LittleEndian,
        FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian).layout(),
    )
    .unwrap();
    assert_eq!(
        ComplexType::new(component).unwrap_err(),
        FormatError::ComplexSizeOverflow {
            component_size: 0x8000_0000
        }
    );
}

#[test]
fn native_complex_accepts_largest_component_size() {
    let component = FloatingPointType::from_layout(
        u32::MAX / 2,
        DatatypeByteOrder::LittleEndian,
        FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian).layout(),
    )
    .unwrap();
    let complex = ComplexType::new(component).unwrap();
    assert_eq!(complex.component(), component);
    assert_eq!(complex.size().get(), u32::MAX - 1);
}
