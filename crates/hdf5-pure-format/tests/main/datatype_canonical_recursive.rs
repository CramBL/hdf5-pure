use hdf5_pure_core::__private;
use hdf5_pure_core::{
    Datatype, EnumMember, EnumerationType, FormatError, ReferenceType, VariableLengthType,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::Class;
use test_util::datatype::containers::{self, CompoundField, EnumValue, MemberOffset};
use test_util::datatype::leaf::{self, Leaf, OpaqueTag, Order, VaxVersion, Version};

#[rstest]
#[case::integer(Parent::Atom(AtomicParent::Integer))]
#[case::float(Parent::Atom(AtomicParent::Float))]
#[case::time(Parent::Atom(AtomicParent::Time))]
#[case::string(Parent::Atom(AtomicParent::String))]
#[case::bitfield(Parent::Atom(AtomicParent::Bitfield))]
#[case::opaque(Parent::Atom(AtomicParent::Opaque))]
#[case::vax(Parent::Vax)]
#[case::compound(Parent::Compound)]
#[case::enumeration(Parent::Enumeration)]
#[case::sequence(Parent::Sequence)]
#[case::variable_string(Parent::String)]
#[case::complex(Parent::Complex(false))]
#[case::vax_complex(Parent::Complex(true))]
fn datatype_canonical_recursive_preserves_properties_and_minimal_versions(
    #[case] parent: Parent,
    #[values(Wrapper::Sequence, Wrapper::Array, Wrapper::Compound)] wrapper: Wrapper,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let source = parent.bytes(OffsetWidth::Eight, true);
    let semantic = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    let expected = wrapper.bytes(
        width,
        &parent.bytes(width, false),
        parent.size(width),
        parent.version(false),
    );
    assert_eq!(
        format::serialize_datatype_with_offset_width(&wrapper.wrap(semantic), width).unwrap(),
        expected
    );
}

#[rstest]
fn prepared_atomic_datatype_checks_capacity_before_writing(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let leaf = Leaf::Integer;
    let source = leaf.bytes(Version::Five);
    let datatype = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    assert_prepared_capacity(&datatype, width, &leaf.bytes(Version::One));
}

#[rstest]
fn prepared_nested_datatype_checks_capacity_before_writing(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = Parent::Sequence;
    let wrapper = Wrapper::Compound;
    let source = parent.bytes(OffsetWidth::Eight, true);
    let semantic = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    let expected = wrapper.bytes(
        width,
        &parent.bytes(width, false),
        parent.size(width),
        parent.version(false),
    );
    assert_prepared_capacity(&wrapper.wrap(semantic), width, &expected);
}

#[rstest]
fn prepared_nested_array_datatype_checks_capacity_before_writing(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = Parent::Sequence;
    let source = parent.bytes(OffsetWidth::Eight, true);
    let semantic = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    let array = Wrapper::Array.bytes(
        width,
        &parent.bytes(width, false),
        parent.size(width),
        parent.version(false),
    );
    let expected = Wrapper::Array.bytes(width, &array, parent.size(width) * 6, Version::Three);
    let datatype = Wrapper::Array.wrap(Wrapper::Array.wrap(semantic));
    assert_prepared_capacity(&datatype, width, &expected);
}

fn assert_prepared_capacity(datatype: &Datatype, width: OffsetWidth, expected: &[u8]) {
    let prepared = format::PreparedDatatype::new(datatype, width).unwrap();
    assert_eq!(prepared.encoded_size(), expected.len());
    assert_eq!(prepared.to_bytes(), expected);

    #[repr(C, align(8))]
    struct Aligned([u8; 128]);

    let guard = 0xa5;
    let mut storage = Aligned([guard; 128]);
    let output = &mut storage.0[..expected.len() + 3];
    for available in 0..expected.len() {
        assert_eq!(
            prepared
                .write_into(&mut output[1..1 + available])
                .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: expected.len(),
                available,
            },
        );
        assert_eq!(output, &[guard; 128][..expected.len() + 3]);
    }
    assert_eq!(
        prepared.write_into(&mut output[1..]).unwrap(),
        expected.len()
    );
    assert_eq!(output[0], guard);
    assert_eq!(&output[1..1 + expected.len()], expected);
    assert_eq!(output[1 + expected.len()..], [guard; 2]);
}

#[rstest]
#[case::invalid_child(0, FormatError::InvalidOpaqueTagByte { offset: 0, byte: 0 })]
#[case::invalid_range_and_child(4, FormatError::CompoundFieldOutOfBounds {
    name: "tagged".into(),
    offset: 4,
    field_size: 1,
    compound_size: 4,
})]
fn compound_member_bounds_error_precedes_an_invalid_opaque_tag(
    #[case] offset: u64,
    #[case] expected: FormatError,
) {
    let datatype = Datatype::Compound {
        size: 4,
        members: vec![__private::compound_member(
            "tagged".into(),
            offset,
            Datatype::Opaque {
                size: 1,
                tag: vec![0],
            },
        )],
    };
    assert_eq!(
        format::PreparedDatatype::new(&datatype, OffsetWidth::Eight).unwrap_err(),
        expected,
    );
}

#[test]
fn sequence_member_bounds_error_precedes_an_invalid_base() {
    let base = Datatype::Opaque {
        size: 1,
        tag: vec![0],
    };
    let datatype = Datatype::Compound {
        size: 4,
        members: vec![__private::compound_member(
            "sequence".into(),
            0,
            Datatype::VariableLength(VariableLengthType::sequence(base)),
        )],
    };
    assert_eq!(
        format::PreparedDatatype::new(&datatype, OffsetWidth::Two).unwrap_err(),
        FormatError::CompoundFieldOutOfBounds {
            name: "sequence".into(),
            offset: 0,
            field_size: 10,
            compound_size: 4,
        },
    );
}

#[rstest]
#[case::out_of_bounds_member(3, FormatError::CompoundFieldOutOfBounds {
    name: "array".into(),
    offset: 0,
    field_size: 6,
    compound_size: 4,
})]
#[case::unrepresentable_array_size(u32::MAX, FormatError::LengthTooLarge {
    length: 8_589_934_590,
    length_size: 4,
})]
fn array_member_size_and_bounds_errors_precede_an_invalid_base(
    #[case] dimension: u32,
    #[case] expected: FormatError,
) {
    let datatype = Datatype::Compound {
        size: 4,
        members: vec![__private::compound_member(
            "array".into(),
            0,
            Datatype::Array {
                base_type: Box::new(Datatype::Opaque {
                    size: 2,
                    tag: vec![0],
                }),
                dimensions: vec![dimension],
            },
        )],
    };
    assert_eq!(
        format::PreparedDatatype::new(&datatype, OffsetWidth::Eight).unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::object(ReferenceType::Object, [0; 3], 8)]
#[case::region(ReferenceType::DatasetRegion, REGION_REFERENCE, 12)]
fn datatype_canonical_recursive_preserves_legacy_reference_parents(
    #[case] ref_type: ReferenceType,
    #[case] flags: [u8; 3],
    #[case] size: u32,
    #[values(Wrapper::Sequence, Wrapper::Array, Wrapper::Compound)] wrapper: Wrapper,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = Datatype::Reference { size, ref_type };
    let bytes = datatype::header(Class::REFERENCE, Version::One.get(), flags, size);
    assert_eq!(
        format::serialize_datatype_with_offset_width(&wrapper.wrap(parent), width).unwrap(),
        wrapper.bytes(width, &bytes, size, Version::One)
    );
}

#[rstest]
#[case::compound(EmptyModel::Compound)]
#[case::enumeration(EmptyModel::Enumeration)]
#[case::zero_array(EmptyModel::Array)]
fn datatype_canonical_recursive_preserves_constructed_models_outside_checked_messages(
    #[case] model: EmptyModel,
    #[values(Wrapper::Sequence, Wrapper::Array, Wrapper::Compound)] wrapper: Wrapper,
) {
    let parent = model.datatype();
    let expected = wrapper.bytes(
        OffsetWidth::Eight,
        &model.bytes(),
        model.size(),
        Version::Three,
    );
    assert_eq!(
        format::serialize_datatype(&wrapper.wrap(parent)).unwrap(),
        expected
    );
    assert_eq!(
        DatatypeMessage::parse(&expected).unwrap_err(),
        model.error()
    );
}

#[rstest]
#[case::big(Leaf::Integer)]
#[case::little(Leaf::LittleInteger)]
fn datatype_canonical_enumeration_preserves_integer_properties_and_raw_values(
    #[case] leaf: Leaf,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let base = DatatypeMessage::parse(&leaf.bytes(Version::Five))
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    let Datatype::FixedPoint(base) = base else {
        panic!("expected fixed-point base, got {base:?}");
    };
    let datatype = Datatype::Enumeration(
        EnumerationType::new(
            base,
            vec![
                EnumMember::new("first".into(), vec![0x11; 4]),
                EnumMember::new("第二".into(), vec![0x22; 4]),
            ],
        )
        .unwrap(),
    );
    let parent = leaf.bytes(Version::One);
    let expected = containers::enumeration(
        Version::Three.get(),
        [2, 0, 0],
        4,
        &parent,
        &[
            EnumValue {
                name: b"first",
                value: &[0x11; 4],
            },
            EnumValue {
                name: "第二".as_bytes(),
                value: &[0x22; 4],
            },
        ],
    );
    assert_prepared_capacity(&datatype, width, &expected);
    assert_eq!(
        DatatypeMessage::parse(&expected)
            .unwrap()
            .0
            .to_datatype()
            .unwrap(),
        datatype
    );
}

#[rstest]
fn enumeration_values_keep_the_width_of_the_decoded_parent(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] source: OffsetWidth,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] destination: OffsetWidth,
    #[values(false, true)] array_parent: bool,
) {
    let sequence = Parent::Sequence;
    let mut size = sequence.size(source);
    let mut base = sequence.bytes(source, false);
    if array_parent {
        size *= 2;
        base = datatype::array(Version::Three.get(), [0; 3], size, 1, &[2], &base);
    }
    let value = vec![0x5a; size as usize];
    let bytes = containers::enumeration(
        Version::Three.get(),
        [1, 0, 0],
        size,
        &base,
        &[EnumValue {
            name: b"value",
            value: &value,
        }],
    );

    let (datatype, consumed) = format::parse_datatype(&bytes).unwrap();
    let (retained, retained_consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(retained_consumed, bytes.len());
    assert_eq!(retained.to_datatype().unwrap(), datatype);
    let Datatype::Enumeration(enumeration) = &datatype else {
        panic!("expected an enumeration, got {datatype:?}");
    };
    assert_eq!(
        enumeration.base_type(),
        &format::parse_datatype(&base).unwrap().0
    );
    assert_eq!(enumeration.size().get(), size);
    assert_eq!(
        enumeration.members(),
        [EnumMember::new("value".into(), value)]
    );

    if source == destination {
        assert_eq!(retained.stored_size(destination).unwrap(), size);
        assert_prepared_capacity(&datatype, source, &bytes);
        assert_eq!(retained.serialize(source).unwrap(), bytes);
    } else {
        let base_size = sequence.size(destination) * if array_parent { 2 } else { 1 };
        let expected = FormatError::EnumBaseSizeMismatch { size, base_size };
        assert_eq!(
            format::PreparedDatatype::new(&datatype, destination).unwrap_err(),
            expected
        );
        assert_eq!(retained.stored_size(destination).unwrap_err(), expected);
        assert_eq!(retained.serialize(destination).unwrap_err(), expected);
    }
}

#[rstest]
fn constructed_containers_keep_a_retained_enumeration_width(
    #[values(OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let sequence = Parent::Sequence;
    let size = sequence.size(width);
    let enumeration = containers::enumeration(
        Version::Three.get(),
        [1, 0, 0],
        size,
        &sequence.bytes(width, false),
        &[EnumValue {
            name: b"value",
            value: &vec![0; size as usize],
        }],
    );
    let (retained, consumed) = DatatypeMessage::parse(&enumeration).unwrap();
    assert_eq!(consumed, enumeration.len());
    let array = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Array {
            base_type: Box::new(retained.clone()),
            dimensions: vec![2],
        },
    )
    .unwrap();
    assert_eq!(array.stored_size(width).unwrap(), 2 * size);
    assert_eq!(
        array.serialize(width).unwrap(),
        datatype::array(
            Version::Three.get(),
            [0; 3],
            2 * size,
            1,
            &[2],
            &enumeration
        )
    );
    let compound = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Compound {
            size,
            members: vec![DatatypeMessageMember::new("member".into(), 0, retained).unwrap()],
        },
    )
    .unwrap();
    assert_eq!(compound.stored_size(width).unwrap(), size);
    assert_eq!(
        compound.serialize(width).unwrap(),
        containers::compound(
            Version::Three.get(),
            [1, 0, 0],
            size,
            MemberOffset::One,
            &[CompoundField {
                name: b"member",
                offset: 0,
                datatype: &enumeration,
            }],
        )
    );
}

#[test]
fn enumeration_preserves_padded_compound_values_at_the_source_width() {
    let source = datatype::enum_with_padded_sequence_parent(Version::Three);
    let canonical = datatype::enum_with_padded_sequence_parent(Version::One);
    assert_eq!(source.len(), 65);
    let (datatype, retained) = parse_enumeration(&source);
    let Datatype::Enumeration(enumeration) = &datatype else {
        panic!("expected an enumeration, got {datatype:?}");
    };
    assert_eq!(enumeration.size().get(), 16);
    assert_eq!(
        enumeration.members(),
        [EnumMember::new("value".into(), source[49..].to_vec())]
    );
    let (descriptor, consumed) =
        format::VariableLengthDescriptor::parse(OffsetWidth::Four, &enumeration.members()[0].value)
            .unwrap();
    assert_eq!(consumed, 12);
    assert_eq!(descriptor.element_count(), 1);
    assert_eq!(descriptor.collection_address().get(), 0x1000);
    assert_eq!(descriptor.object_index(), 1);
    assert_eq!(retained.stored_size(OffsetWidth::Four).unwrap(), 16);
    assert_eq!(retained.serialize(OffsetWidth::Four).unwrap(), source);
    assert_prepared_capacity(&datatype, OffsetWidth::Four, &canonical);
}

#[rstest]
#[case::smaller(OffsetWidth::Two)]
#[case::larger(OffsetWidth::Eight)]
fn enumeration_rejects_descriptor_width_mismatch_with_equal_parent_size(
    #[case] destination: OffsetWidth,
) {
    let source = datatype::enum_with_padded_sequence_parent(Version::Three);
    let (datatype, retained) = parse_enumeration(&source);
    assert_enum_destination_error(
        &datatype,
        &retained,
        destination,
        FormatError::EnumBaseOffsetWidthMismatch {
            source_offset_size: 4,
            destination_offset_size: destination.get(),
        },
    );
}

#[rstest]
#[case::two(OffsetWidth::Two, 4)]
#[case::four(OffsetWidth::Four, 8)]
#[case::eight(OffsetWidth::Eight, 4)]
fn nested_enumeration_base_rejects_every_destination_for_mixed_offsets(
    #[case] destination: OffsetWidth,
    #[case] source_offset_size: u8,
) {
    let parent = array_of_sequence_compounds(4, 8);
    let bytes = containers::enumeration(
        Version::Three.get(),
        [1, 0, 0],
        64,
        &parent,
        &[EnumValue {
            name: b"value",
            value: &[0; 64],
        }],
    );
    let (datatype, retained) = parse_enumeration(&bytes);
    assert_enum_destination_error(
        &datatype,
        &retained,
        destination,
        FormatError::EnumBaseOffsetWidthMismatch {
            source_offset_size,
            destination_offset_size: destination.get(),
        },
    );
}

#[test]
fn enumeration_rejects_inner_sequence_descriptor_width_mismatch() {
    let leaf = leaf::unsigned_byte(Version::One);
    let inner = datatype::VariableLengthSequence::new(Version::One, 4, &leaf).build();
    let outer = datatype::VariableLengthSequence::new(Version::One, 8, &inner);
    let bytes = containers::enumeration(
        Version::Three.get(),
        [1, 0, 0],
        outer.descriptor_size(),
        &outer.build(),
        &[EnumValue {
            name: b"value",
            value: &[0; 16],
        }],
    );
    let (datatype, retained) = parse_enumeration(&bytes);
    assert_enum_destination_error(
        &datatype,
        &retained,
        OffsetWidth::Eight,
        FormatError::EnumBaseOffsetWidthMismatch {
            source_offset_size: 4,
            destination_offset_size: 8,
        },
    );
}

#[rstest]
fn a_compound_enumeration_without_descriptors_accepts_every_offset_width(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let leaf = leaf::unsigned_byte(Version::One);
    let parent = containers::compound(
        Version::Three.get(),
        [1, 0, 0],
        4,
        MemberOffset::One,
        &[CompoundField {
            name: b"field",
            offset: 2,
            datatype: &leaf,
        }],
    );
    let bytes = containers::enumeration(
        Version::Three.get(),
        [1, 0, 0],
        4,
        &parent,
        &[EnumValue {
            name: b"value",
            value: &[0xa1, 0x22, 0x45, 0xee],
        }],
    );
    let (datatype, retained) = parse_enumeration(&bytes);
    assert_eq!(retained.stored_size(width).unwrap(), 4);
    assert_eq!(retained.serialize(width).unwrap(), bytes);
    assert_prepared_capacity(&datatype, width, &bytes);
}

#[test]
fn an_array_of_compounds_can_change_descriptor_width_outside_an_enumeration() {
    let source = array_of_sequence_compounds(4, 8);
    let expected = array_of_sequence_compounds(8, 8);
    let (datatype, consumed) = format::parse_datatype(&source).unwrap();
    assert_eq!(consumed, source.len());
    let (retained, consumed) = DatatypeMessage::parse(&source).unwrap();
    assert_eq!(consumed, source.len());
    assert_eq!(retained.stored_size(OffsetWidth::Eight).unwrap(), 64);
    assert_eq!(retained.serialize(OffsetWidth::Eight).unwrap(), expected);
    assert_prepared_capacity(&datatype, OffsetWidth::Eight, &expected);
}

fn array_of_sequence_compounds(first_offsets: usize, second_offsets: usize) -> Vec<u8> {
    let version = Version::Three.get();
    let leaf = leaf::unsigned_byte(Version::One);
    let first = datatype::VariableLengthSequence::new(Version::One, first_offsets, &leaf).build();
    let second = datatype::VariableLengthSequence::new(Version::One, second_offsets, &leaf).build();
    let compound = containers::compound(
        version,
        [2, 0, 0],
        32,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"first",
                offset: 0,
                datatype: &first,
            },
            CompoundField {
                name: b"second",
                offset: 16,
                datatype: &second,
            },
        ],
    );
    datatype::array(version, [0; 3], 64, 1, &[2], &compound)
}

fn parse_enumeration(bytes: &[u8]) -> (Datatype, DatatypeMessage) {
    let (datatype, consumed) = format::parse_datatype(bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let (retained, consumed) = DatatypeMessage::parse(bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(retained.to_datatype().unwrap(), datatype);
    (datatype, retained)
}

fn assert_enum_destination_error(
    datatype: &Datatype,
    retained: &DatatypeMessage,
    destination: OffsetWidth,
    expected: FormatError,
) {
    assert_eq!(
        format::PreparedDatatype::new(datatype, destination).unwrap_err(),
        expected
    );
    assert_eq!(retained.stored_size(destination).unwrap_err(), expected);
    assert_eq!(retained.serialize(destination).unwrap_err(), expected);
}

#[derive(Clone, Copy, Debug)]
enum Wrapper {
    Array,
    Compound,
    Sequence,
}

impl Wrapper {
    fn wrap(self, parent: Datatype) -> Datatype {
        match self {
            Self::Array => Datatype::Array {
                base_type: Box::new(parent),
                dimensions: vec![2, 3],
            },
            Self::Compound => Datatype::Compound {
                size: 32,
                members: vec![__private::compound_member("field".into(), 8, parent)],
            },
            Self::Sequence => Datatype::VariableLength(VariableLengthType::sequence(parent)),
        }
    }

    fn bytes(self, width: OffsetWidth, parent: &[u8], size: u32, version: Version) -> Vec<u8> {
        match self {
            Self::Array => datatype::array(
                version.get().max(Version::Three.get()),
                [0; 3],
                size * 6,
                2,
                &[2, 3],
                parent,
            ),
            Self::Compound => containers::compound(
                version.get().max(Version::Three.get()),
                [1, 0, 0],
                32,
                MemberOffset::One,
                &[CompoundField {
                    name: b"field",
                    offset: 8,
                    datatype: parent,
                }],
            ),
            Self::Sequence => {
                datatype::variable_length(version.get(), [0; 3], descriptor_size(width), parent)
            }
        }
    }
}

#[derive(Debug)]
enum Parent {
    Atom(AtomicParent),
    Complex(bool),
    Compound,
    Enumeration,
    Sequence,
    String,
    Vax,
}

impl Parent {
    fn bytes(&self, width: OffsetWidth, source: bool) -> Vec<u8> {
        let version = self.version(source);
        let child_version = if source { Version::Five } else { Version::One };
        let integer = Leaf::Integer.bytes(child_version);
        match self {
            Self::Atom(leaf) => leaf.leaf().bytes(version),
            Self::Complex(vax) => {
                let parent = if *vax {
                    leaf::vax_bytes(if source {
                        VaxVersion::Five
                    } else {
                        VaxVersion::Three
                    })
                } else {
                    Leaf::Float(Order::Big).bytes(child_version)
                };
                datatype::complex(version.get(), HOMOGENEOUS_COMPLEX, 16, &parent)
            }
            Self::Compound => containers::compound(
                version.get(),
                [1, 0, 0],
                8,
                MemberOffset::One,
                &[CompoundField {
                    name: b"field",
                    offset: 0,
                    datatype: &integer,
                }],
            ),
            Self::Enumeration => containers::enumeration(
                version.get(),
                [2, 0, 0],
                4,
                &integer,
                &[
                    EnumValue {
                        name: b"first",
                        value: &[0; 4],
                    },
                    EnumValue {
                        name: b"second",
                        value: &[1; 4],
                    },
                ],
            ),
            Self::Sequence => {
                datatype::variable_length(version.get(), [0; 3], descriptor_size(width), &integer)
            }
            Self::String => datatype::variable_length(
                version.get(),
                UTF8_SPACE_PADDED,
                descriptor_size(width),
                &integer,
            ),
            Self::Vax => leaf::vax_bytes(if source {
                VaxVersion::Five
            } else {
                VaxVersion::Three
            }),
        }
    }

    fn size(&self, width: OffsetWidth) -> u32 {
        match self {
            Self::Atom(leaf) => leaf.size(),
            Self::Complex(_) => 16,
            Self::Compound | Self::Vax => 8,
            Self::Enumeration => 4,
            Self::Sequence | Self::String => descriptor_size(width),
        }
    }

    fn version(&self, source: bool) -> Version {
        if source {
            return Version::Five;
        }
        match self {
            Self::Complex(_) => Version::Five,
            Self::Compound | Self::Enumeration | Self::Vax => Version::Three,
            Self::Atom(_) | Self::Sequence | Self::String => Version::One,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AtomicParent {
    Bitfield,
    Float,
    Integer,
    Opaque,
    String,
    Time,
}

impl AtomicParent {
    fn leaf(self) -> Leaf {
        match self {
            Self::Bitfield => Leaf::Bitfield,
            Self::Float => Leaf::Float(Order::Big),
            Self::Integer => Leaf::Integer,
            Self::Opaque => Leaf::Opaque(OpaqueTag::Aligned),
            Self::String => Leaf::string(),
            Self::Time => Leaf::Time,
        }
    }

    fn size(self) -> u32 {
        match self {
            Self::Bitfield | Self::Integer | Self::Time => 4,
            Self::Float => 8,
            Self::Opaque => 3,
            Self::String => 7,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EmptyModel {
    Array,
    Compound,
    Enumeration,
}

impl EmptyModel {
    fn datatype(self) -> Datatype {
        let parent = DatatypeMessage::parse(&Leaf::Integer.bytes(Version::One))
            .unwrap()
            .0
            .to_datatype()
            .unwrap();
        match self {
            Self::Array => Datatype::Array {
                base_type: Box::new(parent),
                dimensions: vec![0],
            },
            Self::Compound => Datatype::Compound {
                size: self.size(),
                members: vec![],
            },
            Self::Enumeration => {
                let Datatype::FixedPoint(base) = parent else {
                    panic!("expected fixed-point base, got {parent:?}");
                };
                Datatype::Enumeration(EnumerationType::new(base, Vec::new()).unwrap())
            }
        }
    }

    fn bytes(self) -> Vec<u8> {
        let integer = Leaf::Integer.bytes(Version::One);
        match self {
            Self::Array => datatype::array(Version::Three.get(), [0; 3], 0, 1, &[0], &integer),
            Self::Compound => {
                datatype::header(Class::COMPOUND, Version::Three.get(), [0; 3], self.size())
            }
            Self::Enumeration => {
                containers::enumeration(Version::Three.get(), [0; 3], self.size(), &integer, &[])
            }
        }
    }

    fn size(self) -> u32 {
        match self {
            Self::Array => 0,
            Self::Compound | Self::Enumeration => 4,
        }
    }
    fn error(self) -> FormatError {
        match self {
            Self::Array => FormatError::ZeroSizedDatatype {
                class: Class::ARRAY.0,
            },
            Self::Compound => FormatError::InvalidCompoundMemberCount { count: 0 },
            Self::Enumeration => FormatError::InvalidEnumerationMemberCount { count: 0 },
        }
    }
}

fn descriptor_size(width: OffsetWidth) -> u32 {
    HEAP_REFERENCE_OVERHEAD + u32::from(width.get())
}

/// The eight descriptor bytes outside the heap address.
///
/// `H5T__vlen_disk_write` in `H5Tvlen.c`, HDF5 2.2.0, writes the four-byte sequence length.
/// The four-byte index is defined in "Global Heap" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
const HEAP_REFERENCE_OVERHEAD: u32 = 8;
/// The flags for a version 5 homogeneous rectangular complex datatype message.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const HOMOGENEOUS_COMPLEX: [u8; 3] = [1, 0, 0];
/// The dataset-region reference flags for datatype message versions 1 through 3.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const REGION_REFERENCE: [u8; 3] = [1, 0, 0];
/// The space-padded UTF-8 variable-length string flags for datatype message versions 1 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const UTF8_SPACE_PADDED: [u8; 3] = [0x21, 1, 0];
