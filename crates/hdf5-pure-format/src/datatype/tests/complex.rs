use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use hdf5_pure_core::{
    ComplexType, CompoundMember, Datatype, DatatypeByteOrder, DatatypeMemberName,
    FloatingPointType, FormatError, VariableLengthType,
};
use rstest::rstest;

use super::super::header::{
    self, ARRAY_CLASS, COMPLEX_CLASS, COMPOUND_CLASS, DATATYPE_VERSION_FIVE,
    DATATYPE_VERSION_THREE, FIXED_POINT_CLASS, FLOATING_POINT_CLASS, VARIABLE_LENGTH_CLASS,
};
use super::super::message::atomic::{FLOAT_NORMALIZATION_RESERVED, FLOAT_NORMALIZATION_SHIFT};
use super::super::message::{FLOAT_BYTE_ORDER_HIGH_BIT, FLOAT_BYTE_ORDER_LOW_BIT};
use super::super::references;
use super::super::semantic;
use super::super::slots;

#[rstest]
#[case::binary32_le(4, false)]
#[case::binary32_be(4, true)]
#[case::binary64_le(8, false)]
#[case::binary64_be(8, true)]
fn native_complex_preserves_component_layout_and_message_extent(
    #[case] component_size: u32,
    #[case] big_endian: bool,
    #[values(1, 2, 3, 4, 5)] parent_version: u8,
) {
    let mut parent = ieee_parent(component_size, big_endian);
    let canonical_parent = parent.clone();
    parent[0] = FLOATING_POINT_CLASS | (parent_version << 4);
    let mut encoded = header::build_datatype_header(
        COMPLEX_CLASS,
        DATATYPE_VERSION_FIVE,
        [1, 0, 0],
        component_size * 2,
    );
    encoded.extend_from_slice(&parent);
    let consumed = encoded.len();
    let following = super::build_fixed_point(1, false, false, 0, 8);
    encoded.extend_from_slice(&following);

    let (datatype, actual_consumed) = semantic::parse_datatype(&encoded).unwrap();
    let Datatype::Complex(complex) = &datatype else {
        panic!("expected native complex, got {datatype:?}");
    };
    let order = if big_endian {
        DatatypeByteOrder::BigEndian
    } else {
        DatatypeByteOrder::LittleEndian
    };
    let expected_component = match component_size {
        4 => FloatingPointType::ieee754_binary32(order),
        8 => FloatingPointType::ieee754_binary64(order),
        _ => panic!("unsupported test component size {component_size}"),
    };
    assert_eq!(complex.component(), expected_component);
    assert_eq!(complex.component().layout(), expected_component.layout());
    assert_eq!(complex.size().get(), component_size * 2);
    assert_eq!(datatype.type_size(), component_size * 2);
    assert_eq!(datatype.element_size().unwrap().get(), component_size * 2);
    assert_eq!(actual_consumed, consumed);
    assert_eq!(actual_consumed, 28);
    assert_eq!(
        semantic::parse_datatype(&encoded[actual_consumed..])
            .unwrap()
            .1,
        following.len()
    );

    let mut expected = header::build_datatype_header(
        COMPLEX_CLASS,
        DATATYPE_VERSION_FIVE,
        [1, 0, 0],
        component_size * 2,
    );
    expected.extend_from_slice(&canonical_parent);
    assert_eq!(semantic::serialize_datatype(&datatype).unwrap(), expected);
    assert_eq!(
        semantic::parse_datatype(&expected).unwrap(),
        (datatype, expected.len())
    );
}

#[rstest]
#[case::heterogeneous([0, 0, 0], FormatError::UnsupportedComplexForm { homogeneous: false, form: 0 })]
#[case::polar([3, 0, 0], FormatError::UnsupportedComplexForm { homogeneous: true, form: 1 })]
#[case::exponential([5, 0, 0], FormatError::UnsupportedComplexForm { homogeneous: true, form: 2 })]
#[case::reserved_form([7, 0, 0], FormatError::UnsupportedComplexForm { homogeneous: true, form: 3 })]
#[case::reserved_low([9, 0, 0], FormatError::InvalidComplexFlags(9))]
#[case::reserved_middle([1, 1, 0], FormatError::InvalidComplexFlags(0x0101))]
#[case::reserved_high([1, 0, 1], FormatError::InvalidComplexFlags(0x01_0001))]
fn native_complex_rejects_unsupported_flags(#[case] flags: [u8; 3], #[case] error: FormatError) {
    let encoded = header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, flags, 8);
    assert_eq!(semantic::parse_datatype(&encoded).unwrap_err(), error);
}

#[rstest]
#[case::one(1)]
#[case::two(2)]
#[case::three(3)]
#[case::four(4)]
fn native_complex_rejects_other_message_versions(#[case] version: u8) {
    let encoded = header::build_datatype_header(COMPLEX_CLASS, version, [1, 0, 0], 8);
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: COMPLEX_CLASS,
            version
        },
    );
}

#[rstest]
#[case::no_parent(0, 16, 8)]
#[case::partial_parent_header(7, 16, 15)]
#[case::no_parent_properties(8, 28, 16)]
#[case::partial_parent_properties(19, 28, 27)]
fn native_complex_rejects_truncated_parent(
    #[case] parent_len: usize,
    #[case] expected: usize,
    #[case] available: usize,
) {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    encoded.extend_from_slice(&ieee_parent(4, false)[..parent_len]);
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[rstest]
#[case::integer(FIXED_POINT_CLASS)]
#[case::compound(COMPOUND_CLASS)]
#[case::complex(COMPLEX_CLASS)]
fn native_complex_rejects_non_floating_parent(#[case] class: u8) {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    encoded.extend_from_slice(&header::build_datatype_header(
        class,
        DATATYPE_VERSION_FIVE,
        [0, 0, 0],
        4,
    ));
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::InvalidComplexParent { class }
    );
}

#[rstest]
#[case::zero(0)]
#[case::too_small(4)]
#[case::too_large(16)]
fn native_complex_rejects_inconsistent_element_size(#[case] size: u32) {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], size);
    encoded.extend_from_slice(&ieee_parent(4, false));
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::InvalidComplexSize {
            size,
            component_size: 4
        }
    );
}

#[rstest]
#[case::zero(0)]
#[case::six(6)]
#[case::fifteen(15)]
fn native_complex_rejects_invalid_parent_version(#[case] version: u8) {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    let mut parent = ieee_parent(4, false);
    parent[0] = (version << 4) | FLOATING_POINT_CLASS;
    encoded.extend_from_slice(&parent);
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::InvalidDatatypeVersion {
            class: FLOATING_POINT_CLASS,
            version
        }
    );
}

#[rstest]
#[case::reserved_bit(5, 0x80)]
#[case::reserved_high_byte(5, 0x01_0000)]
#[case::reserved_order(5, 0x40)]
#[case::old_vax_order(1, 0x41)]
fn native_complex_rejects_reserved_parent_flags(#[case] version: u8, #[case] extra: u32) {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    let mut parent = ieee_parent(4, false);
    parent[0] = (version << 4) | FLOATING_POINT_CLASS;
    let flags = 0x1F20 | extra;
    parent[1..4].copy_from_slice(&flags.to_le_bytes()[..3]);
    encoded.extend_from_slice(&parent);
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::InvalidComplexParentFlags { version, flags }
    );
}

#[test]
fn native_complex_reports_reserved_parent_normalization() {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    let mut parent = ieee_parent(4, false);
    parent[1] |= FLOAT_NORMALIZATION_RESERVED << FLOAT_NORMALIZATION_SHIFT;
    encoded.extend_from_slice(&parent);
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::InvalidMantissaNormalization(FLOAT_NORMALIZATION_RESERVED)
    );
}

#[test]
fn native_complex_rejects_element_size_overflow() {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    let mut parent = ieee_parent(4, false);
    parent[4..8].copy_from_slice(&0x8000_0000u32.to_le_bytes());
    encoded.extend_from_slice(&parent);
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap_err(),
        FormatError::ComplexSizeOverflow {
            component_size: 0x8000_0000
        }
    );
}

#[test]
fn native_complex_contains_no_file_addresses() {
    let datatype = Datatype::Complex(
        ComplexType::new(FloatingPointType::ieee754_binary32(
            DatatypeByteOrder::LittleEndian,
        ))
        .unwrap(),
    );
    assert!(!references::class_may_hold_object_address(
        semantic::serialize_datatype(&datatype).unwrap()[0]
    ));
    assert!(!references::datatype_holds_object_address(&datatype));
    assert!(!references::datatype_holds_file_address(&datatype));
    assert!(
        slots::embedded_reference_slots(&datatype)
            .unwrap()
            .is_empty()
    );
}

fn ieee_parent(size: u32, big_endian: bool) -> Vec<u8> {
    let mut parent = match size {
        4 => super::build_float(4, 23, 8, 0, 23, 127),
        8 => super::build_float(8, 52, 11, 0, 52, 1023),
        _ => panic!("unsupported test component size {size}"),
    };
    if big_endian {
        parent[1] |= FLOAT_BYTE_ORDER_LOW_BIT;
    }
    parent
}

#[test]
fn native_complex_preserves_vax_component_metadata() {
    let mut encoded =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    let mut parent = ieee_parent(4, false);
    parent[0] = (DATATYPE_VERSION_THREE << 4) | FLOATING_POINT_CLASS;
    parent[1] |= FLOAT_BYTE_ORDER_LOW_BIT | FLOAT_BYTE_ORDER_HIGH_BIT;
    encoded.extend_from_slice(&parent);
    let (datatype, consumed) = semantic::parse_datatype(&encoded).unwrap();
    assert_eq!(
        datatype,
        Datatype::Complex(
            ComplexType::new(FloatingPointType::ieee754_binary32(DatatypeByteOrder::Vax)).unwrap()
        ),
    );
    assert_eq!(consumed, encoded.len());
    assert_eq!(semantic::serialize_datatype(&datatype).unwrap(), encoded);
}

#[rstest]
#[case::array(vec![ComplexContainer::Array])]
#[case::compound(vec![ComplexContainer::Compound])]
#[case::sequence(vec![ComplexContainer::Sequence])]
#[case::array_compound_sequence(vec![ComplexContainer::Array, ComplexContainer::Compound, ComplexContainer::Sequence])]
#[case::compound_sequence_array(vec![ComplexContainer::Compound, ComplexContainer::Sequence, ComplexContainer::Array])]
#[case::sequence_array_compound(vec![ComplexContainer::Sequence, ComplexContainer::Array, ComplexContainer::Compound])]
#[case::nested_compounds(vec![ComplexContainer::Compound, ComplexContainer::Compound])]
fn native_complex_propagates_versions_through_containers(
    #[case] containers: Vec<ComplexContainer>,
) {
    let mut expected =
        header::build_datatype_header(COMPLEX_CLASS, DATATYPE_VERSION_FIVE, [1, 0, 0], 8);
    expected.extend_from_slice(&ieee_parent(4, false));
    let datatype = Datatype::Complex(
        ComplexType::new(FloatingPointType::ieee754_binary32(
            DatatypeByteOrder::LittleEndian,
        ))
        .unwrap(),
    );
    let (datatype, expected) = containers
        .into_iter()
        .fold((datatype, expected), |(datatype, encoded), container| {
            container.wrap(datatype, encoded)
        });
    let encoded = semantic::serialize_datatype(&datatype).unwrap();
    assert_eq!(encoded, expected);
    assert_eq!(
        semantic::parse_datatype(&encoded).unwrap(),
        (datatype, encoded.len())
    );
}

#[derive(Clone, Copy)]
enum ComplexContainer {
    Array,
    Compound,
    Sequence,
}

impl ComplexContainer {
    fn wrap(self, child: Datatype, encoded: Vec<u8>) -> (Datatype, Vec<u8>) {
        let child_size = LittleEndian::read_u32(&encoded[4..8]);
        let (datatype, mut expected) = match self {
            Self::Array => {
                let mut expected = header::build_datatype_header(
                    ARRAY_CLASS,
                    DATATYPE_VERSION_FIVE,
                    [0, 0, 0],
                    child_size * 2,
                );
                expected.push(1);
                expected.extend_from_slice(&2u32.to_le_bytes());
                (
                    Datatype::Array {
                        base_type: Box::new(child),
                        dimensions: vec![2],
                    },
                    expected,
                )
            }
            Self::Compound => {
                let mut expected = header::build_datatype_header(
                    COMPOUND_CLASS,
                    DATATYPE_VERSION_FIVE,
                    [2, 0, 0],
                    child_size + 1,
                );
                expected.extend_from_slice(b"value\0\0");
                (
                    Datatype::Compound {
                        size: child_size + 1,
                        members: vec![
                            CompoundMember::new(DatatypeMemberName::from(c"value"), 0, child),
                            CompoundMember::new(
                                DatatypeMemberName::from(c"tag"),
                                u64::from(child_size),
                                super::byte_member_type(),
                            ),
                        ],
                    },
                    expected,
                )
            }
            Self::Sequence => (
                Datatype::VariableLength(VariableLengthType::sequence(child)),
                header::build_datatype_header(
                    VARIABLE_LENGTH_CLASS,
                    DATATYPE_VERSION_FIVE,
                    [0, 0, 0],
                    16,
                ),
            ),
        };
        expected.extend_from_slice(&encoded);
        if matches!(self, Self::Compound) {
            expected.extend_from_slice(b"tag\0");
            expected.push(u8::try_from(child_size).unwrap());
            expected.extend_from_slice(&super::build_fixed_point(1, false, false, 0, 8));
        }
        (datatype, expected)
    }
}
