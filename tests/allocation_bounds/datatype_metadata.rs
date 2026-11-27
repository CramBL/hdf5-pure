use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeMemberName;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::Layout;
use hdf5_pure::RepackOptions;
use hdf5_pure::VariableLengthType;
use hdf5_pure_core::FormatError;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::DecodedDatatype;
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
};
use rstest::rstest;
use test_util::allocation;
use test_util::data_layout;
use test_util::dataspace::VersionTwo;
use test_util::datatype::containers::{self, CompoundField, EnumValue, MemberOffset};
use test_util::datatype::leaf::{Leaf, Version};
use test_util::datatype::{self, Class, VariableLengthSequence};
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::{Message, MessageType, v2};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;

#[rstest]
#[case::atomic(Shape::Sequence(0))]
#[case::vl_depth_16(Shape::Sequence(16))]
#[case::vl_depth_64(Shape::Sequence(64))]
#[case::vl_depth_256(Shape::Sequence(256))]
#[case::array_depth_16(Shape::Array(16))]
#[case::array_depth_64(Shape::Array(64))]
#[case::array_depth_256(Shape::Array(256))]
#[case::wide_compound(Shape::Compound(256))]
fn canonical_datatype_encoding_has_bounded_allocations(#[case] shape: Shape) {
    let datatype = shape.datatype();
    let names = match &datatype {
        Datatype::Compound { members, .. } => members
            .iter()
            .map(|member| member.name.as_str().unwrap())
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
    assert_eq!(measured.live_bytes, length, "{measured}");
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

    let (prepared, preparation) = allocation::measure(&format!("{region}_prepare"), || {
        format::PreparedDatatype::new(&datatype, format::OffsetWidth::Eight).unwrap()
    });
    match shape {
        Shape::Sequence(_) => {
            assert_eq!(preparation.blocks, 0, "{preparation}");
            assert_eq!(measured.blocks, 1, "{measured}");
            assert_eq!(measured.peak_bytes, length, "{measured}");
        }
        Shape::Array(depth) => {
            let size_table = u64::try_from(depth * size_of::<u32>()).unwrap();
            assert!(
                preparation.blocks <= u64::from(depth.ilog2() + 1),
                "{preparation}"
            );
            assert!(preparation.bytes <= 2 * size_table, "{preparation}");
            assert!(preparation.live_bytes <= size_table, "{preparation}");
            assert!(measured.peak_bytes <= length + size_table, "{measured}");
        }
        Shape::Compound(width) => {
            let width = u64::try_from(width).unwrap();
            assert!(preparation.blocks < width, "{preparation}");
            assert!(preparation.bytes <= 64 * width, "{preparation}");
            assert_eq!(preparation.live_bytes, 0, "{preparation}");
            assert!(measured.peak_bytes <= length.max(64 * width), "{measured}");
        }
    }

    let mut destination = vec![0xa5; expected.len()];
    let (written, measured) = allocation::measure(&format!("{region}_bounded"), || {
        prepared.write_into(&mut destination).unwrap()
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
        .map(|member| member.name.as_str().unwrap())
        .collect::<Vec<_>>();
    let (expected, _) = encode_with_child_buffers(shape, &names);
    let (datatype, consumed) = format::DatatypeMessage::parse_prefix(&expected).unwrap();
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
            format::DatatypeMessage::parse_prefix(&valid).unwrap()
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

#[rstest]
#[case::header_only(compound_header_without_members(), 8, FormatError::UnexpectedEof {
    expected: 9,
    available: 8,
})]
#[case::nested_at_the_depth_limit(nested_compound_prefix(), 704, FormatError::NestingDepthExceeded)]
fn parsers_reject_compound_headers_and_nested_prefixes_without_allocating(
    #[case] bytes: Vec<u8>,
    #[case] length: usize,
    #[case] expected: FormatError,
) {
    assert_eq!(bytes.len(), length);
    let child = Leaf::Integer.bytes(Version::One);
    let valid = containers::compound(
        Version::Three.get(),
        [1, 0, 0],
        4,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &child,
        }],
    );
    let semantic: fn(&[u8]) -> Result<usize, FormatError> =
        |bytes| DecodedDatatype::parse_prefix(bytes).map(|(_, consumed)| consumed);
    let retained =
        |bytes: &[u8]| DatatypeMessage::parse_prefix(bytes).map(|(_, consumed)| consumed);
    for (projection, parse) in [("semantic", semantic), ("retained", retained)] {
        let region = format!("compound_member_{projection}_{length}_bytes");
        let (result, control) = allocation::measure(&format!("{region}_valid"), || parse(&valid));
        assert_eq!(result.unwrap(), valid.len());
        assert!(control.blocks > 0, "{control}");
        assert!(control.peak_bytes >= 1, "{control}");

        let (result, measured) = allocation::measure(&region, || parse(&bytes));
        assert_eq!(result.unwrap_err(), expected);
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(measured.peak_bytes, 0, "{measured}");
    }
}

#[rstest]
#[case::semantic(|bytes: &[u8]| DecodedDatatype::parse_prefix(bytes).map(|_| ()))]
#[case::retained(|bytes: &[u8]| DatatypeMessage::parse(bytes).map(|_| ()))]
fn compound_bounds_errors_copy_the_rejected_name_once(
    #[case] parse: fn(&[u8]) -> Result<(), FormatError>,
) {
    let name = [b'x'; 4096];
    let child = Leaf::Integer.bytes(Version::One);
    let bytes = containers::compound(
        Version::Three.get(),
        [1, 0, 0],
        4,
        MemberOffset::One,
        &[CompoundField {
            name: &name,
            offset: 1,
            datatype: &child,
        }],
    );
    let (result, measured) = allocation::measure("compound_bounds_error", || parse(&bytes));
    assert_eq!(
        result.unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: DatatypeMemberName::new(name).unwrap(),
            offset: 1,
            field_size: 4,
            compound_size: 4,
        }
    );
    assert_eq!(measured.blocks, 1, "{measured}");
    assert_eq!(
        measured.peak_bytes,
        u64::try_from(name.len()).unwrap(),
        "{measured}"
    );
}

#[test]
fn explicit_compound_build_reuses_the_member_storage() {
    let count = 256;
    let mut builder = CompoundTypeBuilder::with_size(count);
    for offset in 0..count {
        builder = builder.u8_field(format!("m{offset}"), u64::from(offset));
    }
    let (datatype, measured) = allocation::measure("explicit_compound_build", || builder.build());
    let Datatype::Compound { size, members } = datatype.unwrap() else {
        panic!("expected a compound datatype");
    };
    assert_eq!(size, count);
    assert_eq!(members.len(), usize::try_from(count).unwrap());
    assert_eq!(measured.live_bytes, 0, "{measured}");
}

fn compound_header_without_members() -> Vec<u8> {
    let [low, high] = u16::MAX.to_le_bytes();
    containers::compound(
        Version::Three.get(),
        [low, high, 0],
        1,
        MemberOffset::One,
        &[],
    )
}

fn nested_compound_prefix() -> Vec<u8> {
    let [low, high] = u16::MAX.to_le_bytes();
    containers::compound(
        Version::Three.get(),
        [low, high, 0],
        1,
        MemberOffset::One,
        &[CompoundField {
            name: b"x",
            offset: 0,
            datatype: &[],
        }],
    )
    .repeat(64)
}

#[rstest]
#[case::members_256(256)]
#[case::members_1024(1024)]
fn wide_compound_decoding_has_bounded_allocations(#[case] count: usize) {
    let shape = Shape::Compound(count);
    let expected = shape.datatype();
    let Datatype::Compound { size, members } = &expected else {
        panic!("expected a compound datatype");
    };
    let names = members
        .iter()
        .map(|member| member.name.as_str().unwrap())
        .collect::<Vec<_>>();
    let (bytes, _) = encode_with_child_buffers(shape, &names);
    let child = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::FixedPoint(FixedPointType::i32()),
    )
    .unwrap();
    let expected_retained = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Compound {
            size: *size,
            members: members
                .iter()
                .map(|member| {
                    DatatypeMessageMember::new(
                        member.name.clone(),
                        member.byte_offset,
                        child.clone(),
                    )
                })
                .collect(),
        },
    )
    .unwrap();

    let ((semantic, consumed), semantic_metrics) =
        allocation::measure(&format!("wide_compound_semantic_{count}"), || {
            DecodedDatatype::parse_prefix(&bytes).unwrap()
        });
    assert_eq!(
        (semantic.into_datatype(), consumed),
        (expected, bytes.len())
    );
    let (retained, retained_metrics) =
        allocation::measure(&format!("wide_compound_retained_{count}"), || {
            DatatypeMessage::parse_prefix(&bytes).unwrap()
        });
    assert_eq!(retained, (expected_retained, bytes.len()));
    assert_eq!(
        retained.0.serialize(format::OffsetWidth::Eight).unwrap(),
        bytes
    );
    let count = u64::try_from(count).unwrap();
    for (metrics, member_size) in [
        (semantic_metrics, size_of::<hdf5_pure::CompoundMember>()),
        (retained_metrics, size_of::<DatatypeMessageMember>()),
    ] {
        let member_size = u64::try_from(member_size).unwrap();
        assert_eq!(metrics.live_bytes, count * (member_size + 9), "{metrics}");
        assert!(metrics.blocks <= 2 * count, "{metrics}");
        assert!(metrics.bytes <= count * (2 * member_size + 96), "{metrics}");
        assert!(
            metrics.peak_bytes <= count * (member_size + 96),
            "{metrics}"
        );
    }
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
                        format!("field{index:04}"),
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

#[rstest]
#[case::moderate(1 << 17)]
#[case::largest(u32::MAX / 8)]
fn object_reference_planning_has_bounded_allocations(#[case] count: u32) {
    let body = datatype::array(
        3,
        [0; 3],
        count * 8,
        1,
        &[count],
        &datatype::object_reference(8),
    );
    let (decoded, length) = DecodedDatatype::parse_prefix(&body).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(length, 21);
    assert_eq!(length, body.len());
    let (slots, measured) = allocation::measure("object_reference_slot_geometry", || {
        format::embedded_reference_slots(&datatype).unwrap()
    });
    eprintln!("object reference slot geometry ({count} slots): {measured}");
    assert!(measured.bytes <= 1024, "{measured}");
    assert_eq!(slots.len(), usize::try_from(count).unwrap());
    assert_eq!(slots.element_size(), usize::try_from(count * 8).unwrap());
    assert_eq!(slots.stored_references(&[0; 8]).count(), 0);
}

#[derive(Clone, Copy, Debug)]
enum Storage {
    Unallocated,
    EmptyCompact,
}

#[rstest]
#[case::object_131072(131_072, 8, datatype::object_reference(8))]
#[case::object_maximum(u32::MAX / 8, 8, datatype::object_reference(8))]
#[case::vlen_maximum(
    u32::MAX / 16,
    16,
    VariableLengthSequence::new(
        Version::One,
        8,
        &Leaf::LittleInteger.bytes(Version::One),
    ).build()
)]
fn repack_reference_planning_is_bounded_without_payload(
    #[case] count: u32,
    #[case] stride: u32,
    #[case] parent: Vec<u8>,
    #[values(Storage::Unallocated, Storage::EmptyCompact)] storage: Storage,
) {
    let body = datatype::array(3, [0; 3], count * stride, 1, &[count], &parent);
    let (decoded, length) = DecodedDatatype::parse_prefix(&body).unwrap();
    let datatype = decoded.into_datatype();
    assert_eq!(length, body.len());
    let (layout, shape, layout_body) = match storage {
        Storage::Unallocated => (
            Layout::Contiguous {
                address: None,
                size: u64::from(count * stride),
            },
            [1],
            data_layout::contiguous(u64::MAX, u64::from(count * stride)),
        ),
        Storage::EmptyCompact => (Layout::Compact { size: 0 }, [0], data_layout::compact(&[])),
    };
    let mut image = Image::starting_with(&[0; 48]);
    let address = image.append(
        &v2::Header::new()
            .messages([
                Message::new(MessageType::DATATYPE, &body),
                Message::new(
                    MessageType::DATASPACE,
                    &VersionTwo::simple(Widths::EIGHT, &shape, None).build(),
                ),
                Message::new(MessageType::DATA_LAYOUT, &layout_body),
            ])
            .build(),
    ) as u64;
    let root = image.append(
        &v2::Header::new()
            .message(Message::new(
                MessageType::LINK,
                &HardLink::new("references", address).build(Widths::EIGHT),
            ))
            .build(),
    ) as u64;
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .version(3)
            .root_header_address(root)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );

    let source = File::from_bytes(image.build()).unwrap();
    let dataset = source.dataset("references").unwrap();
    assert_eq!(dataset.layout().unwrap(), layout);
    assert_eq!(dataset.shape().unwrap(), shape);
    assert_eq!(dataset.datatype().unwrap(), datatype);
    let mut output = Vec::new();
    let ((), measured) = allocation::measure("repack_reference_geometry", || {
        hdf5_pure::repack_into(&source, &mut output, &RepackOptions::new()).unwrap();
    });
    eprintln!("repack reference geometry ({count} slots, {storage:?}): {measured}");
    assert!(measured.bytes <= 64 * 1024, "{measured}");

    let destination = File::from_bytes(output).unwrap();
    let dataset = destination.dataset("references").unwrap();
    assert_eq!(dataset.shape().unwrap(), shape);
    assert_eq!(dataset.datatype().unwrap(), datatype);
    match storage {
        Storage::Unallocated => assert_eq!(dataset.layout().unwrap(), layout),
        Storage::EmptyCompact => assert_eq!(dataset.read_raw().unwrap(), Vec::<u8>::new()),
    }
}
