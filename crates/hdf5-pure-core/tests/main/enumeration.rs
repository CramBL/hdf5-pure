use core::num::NonZeroU32;

use hdf5_pure_core::__private;
use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, EnumMember, EnumerationType, FixedPointType, FormatError,
    OffsetWidth, VariableLengthType,
};
use rstest::rstest;

#[rstest]
#[case::first(&[1], &[2, 0, 0], "first", 1)]
#[case::later(&[1, 0], &[2, 0, 0], "second", 3)]
fn enumeration_rejects_the_first_value_with_the_wrong_width(
    #[case] first: &[u8],
    #[case] second: &[u8],
    #[case] name: &str,
    #[case] actual: usize,
) {
    let members = vec![
        EnumMember::new("first".into(), first.to_vec()),
        EnumMember::new("second".into(), second.to_vec()),
    ];
    assert_eq!(
        EnumerationType::new(FixedPointType::u16(), members).unwrap_err(),
        FormatError::EnumMemberValueSize(name.into(), 2, actual)
    );
}

#[rstest]
#[case::little_endian(DatatypeByteOrder::LittleEndian)]
#[case::big_endian(DatatypeByteOrder::BigEndian)]
fn enumeration_preserves_wide_base_and_member_bytes(#[case] byte_order: DatatypeByteOrder) {
    let base_type = FixedPointType::new(16, byte_order, true, 1, 125).unwrap();
    let members = vec![
        EnumMember::new(
            "first".into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 240, 241, 242, 243, 244, 245, 246, 247,
            ],
        ),
        EnumMember::new(
            "second".into(),
            vec![
                255, 254, 253, 252, 251, 250, 249, 248, 9, 10, 11, 12, 13, 14, 15, 16,
            ],
        ),
    ];
    let expected = members.clone();
    let member_storage = members.as_ptr();
    let enumeration = EnumerationType::new(base_type, members).unwrap();
    assert_eq!(enumeration.base_type(), &Datatype::FixedPoint(base_type));
    assert_eq!(enumeration.size().get(), 16);
    assert_eq!(enumeration.members(), expected);
    assert_eq!(enumeration.members().as_ptr(), member_storage);
}

#[test]
fn empty_enumeration_keeps_its_base_and_width() {
    let base_type = FixedPointType::u32();
    let enumeration = EnumerationType::new(base_type, Vec::new()).unwrap();
    assert_eq!(enumeration.base_type(), &Datatype::FixedPoint(base_type));
    assert_eq!(enumeration.size().get(), 4);
    assert_eq!(enumeration.members(), &[]);
}

#[test]
fn decoded_enumeration_uses_the_supplied_base_offset_width() {
    let base = Datatype::VariableLength(VariableLengthType::sequence(Datatype::FixedPoint(
        FixedPointType::u8(),
    )));
    assert_eq!(
        __private::enumeration_from_stored_parts(
            base,
            NonZeroU32::new(12).unwrap(),
            __private::DatatypeOffsets::descriptor(
                OffsetWidth::Two,
                __private::DatatypeOffsets::default(),
            ),
            vec![EnumMember::new("value".into(), vec![0; 12])],
        )
        .unwrap_err(),
        FormatError::EnumBaseSizeMismatch {
            size: 12,
            base_size: 10
        },
    );
}
