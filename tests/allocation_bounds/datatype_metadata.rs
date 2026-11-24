use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::VariableLengthType;
use hdf5_pure_core::FormatError;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::allocation;
use test_util::datatype::Class;
use test_util::datatype::containers::{self, EnumValue};
use test_util::datatype::leaf::{Leaf, Version};

#[rstest]
#[case::atomic(Shape::Sequence(0))]
#[case::vl_depth_16(Shape::Sequence(16))]
#[case::vl_depth_64(Shape::Sequence(64))]
#[case::vl_depth_256(Shape::Sequence(256))]
#[case::array_depth_16(Shape::Array(16))]
#[case::array_depth_64(Shape::Array(64))]
#[case::array_depth_256(Shape::Array(256))]
#[case::wide_compound(Shape::Compound(256))]
fn canonical_datatype_encoding_allocates_only_for_owned_output(#[case] shape: Shape) {
    let datatype = shape.datatype();
    let names = match &datatype {
        Datatype::Compound { members, .. } => members
            .iter()
            .map(|member| member.name.as_str())
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    let region = format!("datatype_metadata_{shape:?}");
    let (control, control_metrics) =
        allocation::measure(&format!("{region}_child_buffers"), || {
            encode_with_child_buffers(shape, &names)
        });
    let (expected, copied) = control;
    let control_capacity = expected.capacity();
    assert!(control_metrics.blocks > 0, "{control_metrics}");
    assert!(
        control_metrics.bytes >= expected.len() as u64,
        "{control_metrics}"
    );
    assert_eq!(
        control_metrics.live_bytes, control_capacity as u64,
        "{control_metrics}"
    );
    assert!(
        control_metrics.peak_bytes >= control_metrics.live_bytes,
        "{control_metrics}"
    );
    let (encoded, measured) =
        allocation::measure(&region, || format::serialize_datatype(&datatype).unwrap());
    eprintln!("{region}: {measured}");
    assert_eq!(encoded, expected);
    assert_eq!(encoded.len(), shape.encoded_len());
    let length = u64::try_from(expected.len()).unwrap();
    assert_eq!(measured.blocks, 1, "{measured}");
    assert_eq!(measured.bytes, length, "{measured}");
    assert_eq!(measured.live_bytes, length, "{measured}");
    assert_eq!(measured.peak_bytes, length, "{measured}");
    assert!(
        measured.blocks < control_metrics.blocks,
        "{measured}; {control_metrics}"
    );
    assert!(
        measured.bytes < control_metrics.bytes,
        "{measured}; {control_metrics}"
    );
    if let Shape::Sequence(depth) | Shape::Array(depth) = shape {
        assert!(copied >= length * depth as u64 / 3);
    }

    let mut destination = vec![0xa5; expected.len()];
    let (written, measured) = allocation::measure(&format!("{region}_bounded"), || {
        format::PreparedDatatype::new(&datatype, format::OffsetWidth::Eight)
            .unwrap()
            .write_into(&mut destination)
            .unwrap()
    });
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(written, expected.len());
    assert_eq!(destination, expected);
}

#[test]
fn retained_compound_encoding_matches_fixture_and_retains_only_output() {
    let shape = Shape::Compound(256);
    let Datatype::Compound { members, .. } = shape.datatype() else {
        panic!("the compound fixture must have members");
    };
    let names = members
        .iter()
        .map(|member| member.name.as_str())
        .collect::<Vec<_>>();
    let (expected, _) = encode_with_child_buffers(shape, &names);
    let (datatype, consumed) = format::DatatypeMessage::parse(&expected).unwrap();
    assert_eq!(consumed, expected.len());

    let region = "datatype_metadata_retained_compound";
    let (bytes, measured) = allocation::measure(region, || {
        datatype.serialize(format::OffsetWidth::Eight).unwrap()
    });
    eprintln!("{region}: {measured}");
    assert_eq!(measured.live_bytes, u64::try_from(expected.len()).unwrap());
    assert_eq!(bytes, expected);
}

#[test]
fn unterminated_first_enum_name_does_not_allocate_members() {
    let version = Version::Three.get();
    let parent = Leaf::Integer.bytes(Version::One);
    let valid = containers::enumeration(
        version,
        [1, 0, 0],
        4,
        &parent,
        &[EnumValue {
            name: b"field",
            value: &0xf000_000fu32.to_be_bytes(),
        }],
    );
    let ((message, consumed), control) =
        allocation::measure("valid_enum_allocates_members", || {
            format::DatatypeMessage::parse(&valid).unwrap()
        });
    assert!(control.blocks > 0, "{control}");
    assert_eq!(consumed, valid.len());
    assert_eq!(
        message.serialize(format::OffsetWidth::Eight).unwrap(),
        valid
    );

    let count = u16::MAX;
    let mut bytes = valid;
    bytes[1..3].copy_from_slice(&count.to_le_bytes());
    let first_name = bytes.len() - b"field\0".len() - size_of::<u32>();
    bytes.truncate(first_name);
    bytes.resize(first_name + usize::from(count), b'a');

    let (result, measured) = allocation::measure("unterminated_enum_name", || {
        format::DatatypeMessage::parse(&bytes)
    });
    assert_eq!(
        result.unwrap_err(),
        FormatError::UnexpectedEof {
            expected: bytes.len() + 1,
            available: bytes.len(),
        },
    );
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.peak_bytes, 0, "{measured}");
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Array(usize),
    Compound(usize),
    Sequence(usize),
}

impl Shape {
    fn datatype(self) -> Datatype {
        match self {
            Self::Sequence(depth) => (0..depth).fold(hdf5_pure::make_i32_type(), |base, _| {
                Datatype::VariableLength(VariableLengthType::sequence(base))
            }),
            Self::Array(depth) => {
                (0..depth).fold(hdf5_pure::make_i32_type(), |base, _| Datatype::Array {
                    base_type: Box::new(base),
                    dimensions: vec![1],
                })
            }
            Self::Compound(width) => {
                let mut builder = CompoundTypeBuilder::with_size(u32::try_from(width * 4).unwrap());
                for index in 0..width {
                    builder = builder.field(
                        &format!("field{index:04}"),
                        u64::try_from(index * 4).unwrap(),
                        hdf5_pure::make_i32_type(),
                    );
                }
                builder.build().unwrap()
            }
        }
    }

    fn encoded_len(self) -> usize {
        match self {
            Self::Sequence(depth) => 8 * depth + 12,
            Self::Array(depth) => 13 * depth + 12,
            Self::Compound(width) => 8 + width * (10 + 2 + 12),
        }
    }
}

/// Returns independent datatype bytes and the total child bytes copied into parent buffers.
fn encode_with_child_buffers(shape: Shape, names: &[&str]) -> (Vec<u8>, u64) {
    let mut copied = 0;
    let encoded = match shape {
        Shape::Sequence(0) | Shape::Array(0) => {
            let mut encoded = control_header(Class::FIXED_POINT, 1, [8, 0, 0], 4);
            encoded.extend_from_slice(&[0, 0, 32, 0]);
            encoded
        }
        Shape::Sequence(depth) => {
            let child = encode_with_child_buffers(Shape::Sequence(depth - 1), &[]);
            let mut encoded = control_header(Class::VARIABLE_LENGTH, 1, [0; 3], 16);
            copied += child.1 + child.0.len() as u64;
            encoded.extend_from_slice(&child.0);
            encoded
        }
        Shape::Array(depth) => {
            let child = encode_with_child_buffers(Shape::Array(depth - 1), &[]);
            let mut encoded = control_header(Class::ARRAY, 3, [0; 3], 4);
            encoded.extend_from_slice(&[1, 1, 0, 0, 0]);
            copied += child.1 + child.0.len() as u64;
            encoded.extend_from_slice(&child.0);
            encoded
        }
        Shape::Compound(width) => {
            let count = u16::try_from(width).unwrap().to_le_bytes();
            let mut encoded = control_header(
                Class::COMPOUND,
                3,
                [count[0], count[1], 0],
                u32::try_from(width * 4).unwrap(),
            );
            assert_eq!(names.len(), width);
            for (index, name) in names.iter().enumerate() {
                encoded.extend_from_slice(name.as_bytes());
                encoded.push(0);
                encoded.extend_from_slice(&u16::try_from(index * 4).unwrap().to_le_bytes());
                let child = encode_with_child_buffers(Shape::Array(0), &[]);
                copied += child.1 + child.0.len() as u64;
                encoded.extend_from_slice(&child.0);
            }
            encoded
        }
    };
    (std::hint::black_box(encoded), copied)
}

fn control_header(class: Class, version: u8, flags: [u8; 3], size: u32) -> Vec<u8> {
    let mut header = Vec::with_capacity(8);
    header.push(class.0 | (version << 4));
    header.extend_from_slice(&flags);
    header.extend_from_slice(&size.to_le_bytes());
    header
}
