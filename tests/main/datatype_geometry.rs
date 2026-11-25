use hdf5_pure::{
    CompoundTypeBuilder, Datatype, EnumTypeBuilder, Error, FileBuilder, FormatError, OffsetWidth,
};
use hdf5_pure_format::__private::PreparedDatatype;
use rstest::rstest;

#[rstest]
#[case::overlap(overlapping_compound(), FormatError::CompoundFieldOverlap {
    first: "first".into(),
    second: "second".into(),
})]
#[case::enum_size(mismatched_enumeration(), FormatError::EnumBaseSizeMismatch {
    size: 2,
    base_size: 1,
})]
fn writer_rejects_invalid_datatype_geometry_before_output(
    #[case] datatype: Datatype,
    #[case] expected: FormatError,
    #[values(true, false)] committed: bool,
) {
    assert_eq!(
        PreparedDatatype::new(&datatype, OffsetWidth::Eight).unwrap_err(),
        expected
    );
    let mut builder = FileBuilder::new();
    if committed {
        builder.commit_datatype("value", datatype);
    } else {
        builder
            .create_dataset("value")
            .with_unallocated_storage(datatype, &[1]);
    }
    let mut destination = b"previous bytes".to_vec();
    let error = builder.finish_to(&mut destination).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
    assert_eq!(destination, b"previous bytes");
}

#[rstest]
#[case::inline_dataset(false)]
#[case::committed_datatype(true)]
fn writer_rejects_mismatched_enumeration_value_widths_before_output(#[case] committed: bool) {
    let build = || {
        let mut datatype = EnumTypeBuilder::u8_based()
            .u8_value("first", 1)
            .u8_value("last", 2)
            .build()
            .unwrap();
        let Datatype::Enumeration { members, .. } = &mut datatype else {
            panic!("expected an enumeration datatype");
        };
        members[0].value.push(2);
        members[1].value.clear();

        let mut builder = FileBuilder::new();
        if committed {
            builder.commit_datatype("value", datatype);
        } else {
            builder
                .create_dataset("value")
                .with_enum_u8_data(datatype, &[1, 2]);
        }
        builder
    };
    let expected = FormatError::EnumMemberValueSize("first".into(), 1, 2);
    let error = build().finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);

    let mut destination = b"previous bytes".to_vec();
    let error = build().finish_to(&mut destination).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
    assert_eq!(destination, b"previous bytes");
}

fn overlapping_compound() -> Datatype {
    let mut datatype = CompoundTypeBuilder::with_size(8)
        .field("first", 0, hdf5_pure::make_i32_type())
        .field("second", 4, hdf5_pure::make_i32_type())
        .build()
        .unwrap();
    let Datatype::Compound { members, .. } = &mut datatype else {
        panic!("expected a compound datatype");
    };
    members[1].byte_offset = 3;
    datatype
}

fn mismatched_enumeration() -> Datatype {
    let mut datatype = EnumTypeBuilder::u8_based()
        .u8_value("first", 1)
        .build()
        .unwrap();
    let Datatype::Enumeration { size, .. } = &mut datatype else {
        panic!("expected an enumeration datatype");
    };
    *size = 2;
    datatype
}
