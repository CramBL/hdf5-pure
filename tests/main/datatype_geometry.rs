use hdf5_pure::{
    CompoundTypeBuilder, Datatype, Error, File, FileBuilder, FormatError, OffsetWidth,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::PreparedDatatype;
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::VariableLengthSequence;
use test_util::datatype::containers::{self, EnumValue};
use test_util::datatype::leaf::{Leaf, Version};

#[rstest]
#[case::overlap(overlapping_compound(), FormatError::CompoundFieldOverlap {
    first: "first".into(),
    second: "second".into(),
})]
#[case::enumeration(decoded_narrow_enumeration(), FormatError::EnumBaseSizeMismatch {
    size: 12,
    base_size: 16,
})]
#[case::padded_enumeration(decoded_enumeration_with_padded_compound_base(), FormatError::EnumBaseOffsetWidthMismatch {
    source_offset_size: 4,
    destination_offset_size: 8,
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
    builder.with_offset_width(OffsetWidth::Eight);
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

#[test]
fn writer_preserves_padded_compound_enumeration_at_the_source_width() {
    let datatype = decoded_enumeration_with_padded_compound_base();
    let mut builder = FileBuilder::new();
    builder.with_offset_width(OffsetWidth::Four);
    builder
        .create_dataset("value")
        .with_unallocated_storage(datatype.clone(), &[1]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("value").unwrap();
    assert_eq!(dataset.datatype().unwrap(), datatype);
    assert_eq!(dataset.shape().unwrap(), [1]);
}

fn decoded_enumeration_with_padded_compound_base() -> Datatype {
    format::parse_datatype(&datatype::enum_with_padded_sequence_parent(Version::Three))
        .unwrap()
        .0
}

fn decoded_narrow_enumeration() -> Datatype {
    let leaf = Leaf::LittleInteger.bytes(Version::One);
    let sequence = VariableLengthSequence::new(Version::One, 4, &leaf);
    let size = sequence.descriptor_size();
    let bytes = containers::enumeration(
        Version::Three.get(),
        [1, 0, 0],
        size,
        &sequence.build(),
        &[EnumValue {
            name: b"empty",
            value: &[0; 12],
        }],
    );
    let (datatype, consumed) = format::parse_datatype(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    datatype
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
