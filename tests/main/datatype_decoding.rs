use hdf5_pure::{Datatype, Error, File, FixedPointType, FormatError, VariableLengthType};
use rstest::rstest;
use test_util::datatype::leaf::{self, Leaf, Version};
use test_util::datatype::{self, Class, VariableLengthSequence};
use test_util::image::Image;
use test_util::legacy_compound::{self, SequenceMemberShape};
use test_util::object_header::{Message, MessageType};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util::{data_layout, dataspace};

#[test]
fn dataset_datatype_rejects_message_version_fifteen() {
    let mut datatype = Leaf::Integer.bytes(Version::One);
    datatype[0] = 0xf0;
    let file = dataset_file(
        &datatype,
        &dataspace::VersionTwo::scalar().build(),
        &data_layout::compact(&[0; 4]),
    );
    let dataset = file.dataset("value").unwrap();
    let error = dataset.datatype().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected an unsupported datatype version, got {error:?}");
    };
    assert_eq!(error, FormatError::InvalidDatatypeMessageVersion(15));
}

#[test]
fn dataset_datatype_reports_child_eof_from_the_message_start() {
    let datatype = VariableLengthSequence::new(Version::One, 8, &[]).build();
    let file = dataset_file(
        &datatype,
        &dataspace::VersionTwo::scalar().build(),
        &data_layout::compact(&[0; 16]),
    );
    let error = file.dataset("value").unwrap().datatype().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a datatype EOF error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::UnexpectedEof {
            expected: 16,
            available: 8,
        }
    );
}

#[test]
fn dataset_datatype_rejects_an_array_size_that_disagrees_with_its_dimensions() {
    let datatype = datatype::array(
        Version::Three.get(),
        [0; 3],
        1,
        1,
        &[2],
        &leaf::unsigned_byte(Version::One),
    );
    let file = dataset_file(
        &datatype,
        &dataspace::VersionTwo::scalar().build(),
        &data_layout::compact(&[0; 2]),
    );
    let dataset = file.dataset("value").unwrap();
    let error = dataset.datatype().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected an array size error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::InvalidDatatypeSize {
            class: Class::ARRAY.0,
            size: 1,
            expected: 2,
        }
    );
}

#[rstest]
#[case::scalar(SequenceMemberShape::Scalar, 10)]
#[case::implicit_array(SequenceMemberShape::ImplicitArray, 20)]
#[case::explicit_array(SequenceMemberShape::ExplicitArray, 20)]
fn dataset_datatype_reads_a_narrow_legacy_sequence_member(
    #[case] shape: SequenceMemberShape,
    #[case] size: u32,
) {
    let file = File::from_bytes(legacy_compound::narrow_sequence_file(shape)).unwrap();
    let datatype = file.dataset("value").unwrap().datatype().unwrap();
    let Datatype::Compound {
        size: compound_size,
        members,
    } = datatype
    else {
        panic!("expected a compound datatype, got {datatype:?}");
    };
    let [member] = members.as_slice() else {
        panic!("expected one member, got {members:?}");
    };
    let sequence = Datatype::VariableLength(VariableLengthType::sequence(Datatype::FixedPoint(
        FixedPointType::u8(),
    )));
    let expected = match shape {
        SequenceMemberShape::Scalar => sequence,
        SequenceMemberShape::ExplicitArray | SequenceMemberShape::ImplicitArray => {
            Datatype::Array {
                base_type: Box::new(sequence),
                dimensions: vec![2],
            }
        }
    };
    assert_eq!(
        (
            compound_size,
            member.name.as_str().unwrap(),
            member.byte_offset,
            &member.datatype
        ),
        (size, "arr", 0, &expected),
    );
}

#[test]
fn chunked_dataset_rejects_zero_width_string_elements() {
    let datatype = datatype::header(Class::STRING, Version::One.get(), [0; 3], 0);
    let file = dataset_file(
        &datatype,
        &dataspace::VersionTwo::simple(Widths::EIGHT, &[4], None).build(),
        &data_layout::chunked(u64::MAX, &[2], 1),
    );
    let dataset = file.dataset("value").unwrap();
    assert_eq!(dataset.shape().unwrap(), [4]);
    for (name, result) in [
        ("element_size", dataset.element_size().map(|_| ())),
        ("dtype", dataset.dtype().map(|_| ())),
        ("read_raw", dataset.read_raw().map(|_| ())),
        ("read_string", dataset.read_string().map(|_| ())),
    ] {
        let error = result.unwrap_err();
        let Error::Format(error) = error else {
            panic!("{name}: expected a datatype size error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::ZeroSizedDatatype {
                class: Class::STRING.0
            },
            "{name}"
        );
    }
}

fn dataset_file(datatype: &[u8], dataspace: &[u8], layout: &[u8]) -> File {
    let image = Image::starting_with(&vec![0; Superblock::new(Widths::EIGHT).build().len()]);
    let bytes = image.build_with_dataset(
        "value",
        [
            Message::new(MessageType::DATASPACE, dataspace),
            Message::new(MessageType::DATATYPE, datatype),
            Message::new(MessageType::DATA_LAYOUT, layout),
        ],
    );
    File::from_bytes(bytes).unwrap()
}
