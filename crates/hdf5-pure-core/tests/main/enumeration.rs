use core::num::NonZeroU32;

use hdf5_pure_core::__private;
use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, DatatypeMemberName, EnumMember, EnumerationType, FixedPointType,
    FormatError, OffsetWidth, VariableLengthType,
};
use rstest::rstest;

#[rstest]
#[case::text(b"bad\0name", 3)]
#[case::initial(b"\0abc", 0)]
#[case::non_utf8(b"\xff\0\xfe\0", 1)]
fn datatype_member_names_reject_nul_bytes(#[case] bytes: &[u8], #[case] index: usize) {
    let expected = FormatError::DatatypeMemberNameContainsNul { index };
    assert_eq!(DatatypeMemberName::new(bytes).unwrap_err(), expected);
    assert_eq!(
        DatatypeMemberName::try_from(bytes.to_vec()).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::invalid_byte(b"a\xff", Some(1))]
#[case::incomplete_sequence(b"a\xe2\x82", None)]
fn datatype_member_name_text_access_reports_the_utf8_error(
    #[case] bytes: &[u8],
    #[case] error_len: Option<usize>,
) {
    let name = DatatypeMemberName::new(bytes).unwrap();
    let error = name.as_str().unwrap_err();
    assert_eq!(name.as_bytes(), bytes);
    assert_eq!(error.valid_up_to(), 1);
    assert_eq!(error.error_len(), error_len);
}

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
        EnumMember::new(DatatypeMemberName::new("first").unwrap(), first.to_vec()),
        EnumMember::new(DatatypeMemberName::new("second").unwrap(), second.to_vec()),
    ];
    assert_eq!(
        EnumerationType::new(FixedPointType::u16(), members).unwrap_err(),
        FormatError::EnumMemberValueSize(DatatypeMemberName::new(name).unwrap(), 2, actual)
    );
}

#[rstest]
#[case::little_endian(DatatypeByteOrder::LittleEndian)]
#[case::big_endian(DatatypeByteOrder::BigEndian)]
fn enumeration_preserves_wide_base_and_member_bytes(#[case] byte_order: DatatypeByteOrder) {
    let base_type = FixedPointType::new(16, byte_order, true, 1, 125).unwrap();
    let members = vec![
        EnumMember::new(
            DatatypeMemberName::new("first").unwrap(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 240, 241, 242, 243, 244, 245, 246, 247,
            ],
        ),
        EnumMember::new(
            DatatypeMemberName::new("second").unwrap(),
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
            vec![EnumMember::new(
                DatatypeMemberName::new("value").unwrap(),
                vec![0; 12]
            )],
        )
        .unwrap_err(),
        FormatError::EnumBaseSizeMismatch {
            size: 12,
            base_size: 10
        },
    );
}
