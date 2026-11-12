use hdf5_pure_core::FormatError;
use hdf5_pure_format::__private::{
    DatatypeMessage, DatatypeProperties, DatatypeVersion, OffsetWidth,
};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::leaf::{self, Leaf, OpaqueTag, Order, VaxVersion, Version};

#[rstest]
#[case::integer(Leaf::Integer, 4)]
#[case::float_le(Leaf::Float(Order::Little), 8)]
#[case::float_be(Leaf::Float(Order::Big), 8)]
#[case::time(Leaf::Time, 4)]
#[case::string(Leaf::string(), 7)]
#[case::bitfield(Leaf::Bitfield, 4)]
#[case::opaque(Leaf::Opaque(OpaqueTag::Aligned), 3)]
fn datatype_array_complex_codecs_preserve_array_properties(
    #[case] leaf: Leaf,
    #[case] size: u32,
    #[values(DatatypeVersion::Three, DatatypeVersion::Four, DatatypeVersion::Five)]
    version: DatatypeVersion,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = leaf.bytes(Version::One);
    let bytes = datatype::array(version.get(), [0; 3], size * 6, 2, &[2, 3], &parent);
    let expected = DatatypeMessage::new(
        version,
        DatatypeProperties::Array {
            base_type: Box::new(DatatypeMessage::parse(&parent).unwrap().0),
            dimensions: vec![2, 3],
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, width);
}

#[rstest]
#[case::v3_v1(3, Version::One)]
#[case::v3_v3(3, Version::Three)]
#[case::v4_v2(4, Version::Two)]
#[case::v4_v4(4, Version::Four)]
#[case::v5_v1(5, Version::One)]
#[case::v5_v5(5, Version::Five)]
fn datatype_array_complex_codecs_preserve_nested_versions(
    #[case] version: u8,
    #[case] child: Version,
) {
    let leaf = Leaf::Integer.bytes(child);
    let array = datatype::array(version, [0; 3], 8, 1, &[2], &leaf);
    let bytes = datatype::variable_length(5, [0; 3], 16, &array);
    let expected_array = DatatypeMessage::new(
        DatatypeVersion::try_from(version).unwrap(),
        DatatypeProperties::Array {
            base_type: Box::new(DatatypeMessage::parse(&leaf).unwrap().0),
            dimensions: vec![2],
        },
    )
    .unwrap();
    let expected = DatatypeMessage::new(
        DatatypeVersion::Five,
        DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(expected_array),
        },
    )
    .unwrap();
    assert_encoding(&bytes, expected, OffsetWidth::Eight);
}

#[rstest]
fn datatype_array_complex_codecs_derive_nested_array_sizes_from_selected_width(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let leaf = Leaf::Integer.bytes(Version::One);
    let sequence = datatype::variable_length(3, [0; 3], 10, &leaf);
    let inner = datatype::array(4, [0; 3], 20, 1, &[2], &sequence);
    let bytes = datatype::array(5, [0; 3], 60, 1, &[3], &inner);
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let size = 8 + u32::from(width.get());
    let sequence = datatype::variable_length(3, [0; 3], size, &leaf);
    let inner = datatype::array(4, [0; 3], size * 2, 1, &[2], &sequence);
    assert_eq!(
        message.serialize(width).unwrap(),
        datatype::array(5, [0; 3], size * 6, 1, &[3], &inner)
    );
}

#[rstest]
#[case::little(Leaf::Float(Order::Little))]
#[case::big(Leaf::Float(Order::Big))]
fn datatype_array_complex_codecs_preserve_complex_parent_versions(
    #[case] leaf: Leaf,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
) {
    assert_complex(&leaf.bytes(version));
}

#[rstest]
fn datatype_array_complex_codecs_preserve_vax_parent_versions(
    #[values(VaxVersion::Three, VaxVersion::Four, VaxVersion::Five)] version: VaxVersion,
) {
    let parent = leaf::vax_bytes(version);
    assert_complex(&parent);
    let bytes = datatype::array(5, [0; 3], 16, 1, &[2], &parent);
    assert_eq!(
        DatatypeMessage::parse(&bytes)
            .unwrap()
            .0
            .serialize(OffsetWidth::Eight)
            .unwrap(),
        bytes
    );
}

#[rstest]
#[case::array_v1(10, 1)]
#[case::array_v2(10, 2)]
#[case::complex_v1(11, 1)]
#[case::complex_v2(11, 2)]
#[case::complex_v3(11, 3)]
#[case::complex_v4(11, 4)]
fn datatype_array_complex_parser_rejects_incompatible_layout_versions(
    #[case] class: u8,
    #[case] version: u8,
) {
    let bytes = datatype::header(datatype::Class(class), version, [0; 3], 16);
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeVersion { class, version }
    );
}

#[rstest]
#[case::low([1, 0, 0], 1)]
#[case::middle([0, 1, 0], 0x100)]
#[case::high([0, 0, 0x80], 0x80_0000)]
fn datatype_array_complex_parser_rejects_array_flags(#[case] flags: [u8; 3], #[case] bits: u32) {
    let bytes = datatype::array(3, flags, 8, 1, &[2], &Leaf::Integer.bytes(Version::One));
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class: 10,
            version: 3,
            flags: bits
        }
    );
}

#[rstest]
#[case::size(7, &[2], FormatError::InvalidDatatypeSize { class: 10, size: 7, expected: 8 })]
#[case::zero_dimension(0, &[0], FormatError::ZeroSizedDatatype { class: 10 })]
#[case::overflow(4, &[u32::MAX], FormatError::LengthTooLarge { length: u64::from(u32::MAX) * 4, length_size: 4 })]
fn datatype_array_complex_parser_rejects_array_geometry(
    #[case] size: u32,
    #[case] dimensions: &[u32],
    #[case] expected: FormatError,
) {
    let bytes = datatype::array(
        3,
        [0; 3],
        size,
        1,
        dimensions,
        &Leaf::Integer.bytes(Version::One),
    );
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::array(10, 3)]
#[case::sequence(9, 4)]
fn datatype_array_complex_parser_rejects_newer_array_children(
    #[case] class: u8,
    #[case] version: u8,
) {
    let array = datatype::array(5, [0; 3], 8, 1, &[2], &Leaf::Integer.bytes(Version::One));
    let bytes = if class == 10 {
        datatype::array(version, [0; 3], 16, 1, &[2], &array)
    } else {
        datatype::variable_length(version, [0; 3], 16, &array)
    };
    assert_eq!(
        DatatypeMessage::parse(&bytes).unwrap_err(),
        FormatError::DatatypeChildVersion {
            class,
            version,
            child_version: 5
        }
    );
}

#[rstest]
#[case::reserved_low([9, 0, 0], FormatError::InvalidComplexFlags(9))]
#[case::reserved_high([1, 0, 1], FormatError::InvalidComplexFlags(0x01_0001))]
#[case::heterogeneous([0; 3], FormatError::UnsupportedComplexForm { homogeneous: false, form: 0 })]
#[case::polar([3, 0, 0], FormatError::UnsupportedComplexForm { homogeneous: true, form: 1 })]
#[case::exponential([5, 0, 0], FormatError::UnsupportedComplexForm { homogeneous: true, form: 2 })]
#[case::reserved_form([7, 0, 0], FormatError::UnsupportedComplexForm { homogeneous: true, form: 3 })]
fn datatype_array_complex_parser_rejects_complex_flags(
    #[case] flags: [u8; 3],
    #[case] expected: FormatError,
) {
    let bytes = datatype::complex(
        5,
        flags,
        16,
        &Leaf::Float(Order::Little).bytes(Version::One),
    );
    assert_eq!(DatatypeMessage::parse(&bytes).unwrap_err(), expected);
}

#[rstest]
#[case::zero(0)]
#[case::six(6)]
fn datatype_array_complex_parser_rejects_complex_parent_versions(#[case] version: u8) {
    let mut parent = Leaf::Float(Order::Little).bytes(Version::One);
    parent[0] = (version << 4) | 1;
    assert_eq!(
        DatatypeMessage::parse(&datatype::complex(5, [1, 0, 0], 16, &parent)).unwrap_err(),
        FormatError::InvalidDatatypeVersion { class: 1, version }
    );
}

#[rstest]
#[case::reserved_order(0x40)]
#[case::vax_order(0x41)]
fn datatype_array_complex_parser_applies_float_order_in_each_context(
    #[case] order: u8,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
) {
    let mut parent = Leaf::Float(Order::Little).bytes(version);
    parent[1] |= order;
    let flags = u32::from_le_bytes([parent[1], parent[2], parent[3], 0]);
    let complex = datatype::complex(5, [1, 0, 0], 16, &parent);
    let array = datatype::array(5, [0; 3], 16, 1, &[2], &parent);
    if order == 0x41 && version.get() >= 3 {
        assert_eq!(
            DatatypeMessage::parse(&complex)
                .unwrap()
                .0
                .serialize(OffsetWidth::Eight)
                .unwrap(),
            complex
        );
        assert_eq!(
            DatatypeMessage::parse(&array)
                .unwrap()
                .0
                .serialize(OffsetWidth::Eight)
                .unwrap(),
            array
        );
    } else {
        assert_eq!(
            DatatypeMessage::parse(&complex).unwrap_err(),
            FormatError::InvalidComplexParentFlags {
                version: version.get(),
                flags
            }
        );
        let expected = if order == 0x40 {
            FormatError::InvalidByteOrder(order)
        } else {
            FormatError::InvalidDatatypeVersion {
                class: 1,
                version: version.get(),
            }
        };
        assert_eq!(DatatypeMessage::parse(&array).unwrap_err(), expected);
    }
}

#[rstest]
#[case::reserved_bit([0xa0, 63, 0], FormatError::InvalidComplexParentFlags { version: 1, flags: 0x3fa0 })]
#[case::reserved_high([0x20, 63, 1], FormatError::InvalidComplexParentFlags { version: 1, flags: 0x01_3f20 })]
#[case::normalization([0x30, 63, 0], FormatError::InvalidMantissaNormalization(3))]
fn datatype_array_complex_parser_preserves_complex_parent_errors(
    #[case] flags: [u8; 3],
    #[case] expected: FormatError,
) {
    let mut parent = datatype::f64_le();
    parent[1..4].copy_from_slice(&flags);
    assert_eq!(
        DatatypeMessage::parse(&datatype::complex(5, [1, 0, 0], 16, &parent)).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::parent_class(Leaf::Integer, 8, FormatError::InvalidComplexParent { class: 0 })]
#[case::size(Leaf::Float(Order::Little), 15, FormatError::InvalidComplexSize { size: 15, component_size: 8 })]
fn datatype_array_complex_parser_rejects_complex_geometry(
    #[case] leaf: Leaf,
    #[case] size: u32,
    #[case] expected: FormatError,
) {
    let parent = leaf.bytes(Version::One);
    assert_eq!(
        DatatypeMessage::parse(&datatype::complex(5, [1, 0, 0], size, &parent)).unwrap_err(),
        expected
    );
}

#[test]
fn datatype_array_complex_parser_rejects_complex_size_overflow() {
    let mut parent = datatype::f64_le();
    parent[4..8].copy_from_slice(&0x8000_0000u32.to_le_bytes());
    assert_eq!(
        DatatypeMessage::parse(&datatype::complex(5, [1, 0, 0], 16, &parent)).unwrap_err(),
        FormatError::ComplexSizeOverflow {
            component_size: 0x8000_0000
        }
    );
}

#[rstest]
#[case::missing_parent(8, 9)]
#[case::parent_header(15, 16)]
#[case::parent_properties(27, 28)]
fn datatype_array_complex_parser_reports_incomplete_complex_fields(
    #[case] available: usize,
    #[case] expected: usize,
) {
    let bytes = datatype::complex(5, [1, 0, 0], 16, &datatype::f64_le());
    assert_eq!(
        DatatypeMessage::parse(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[test]
fn datatype_array_complex_codecs_bound_array_recursion() {
    let bytes = (0..63).fold(Leaf::Integer.bytes(Version::One), |parent, _| {
        datatype::array(3, [0; 3], 4, 1, &[1], &parent)
    });
    let (message, consumed) = DatatypeMessage::parse(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message.serialize(OffsetWidth::Eight).unwrap(), bytes);
    let too_deep = datatype::array(3, [0; 3], 4, 1, &[1], &bytes);
    assert_eq!(
        DatatypeMessage::parse(&too_deep).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
    let too_deep = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Array {
            base_type: Box::new(message),
            dimensions: vec![1],
        },
    )
    .unwrap();
    assert_eq!(
        too_deep.serialize(OffsetWidth::Eight).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
}

#[rstest]
#[case::missing_rank(8, 9)]
#[case::dimension(12, 13)]
#[case::parent_header(13, 14)]
#[case::parent_properties(24, 25)]
fn datatype_array_complex_parser_reports_incomplete_array_fields(
    #[case] available: usize,
    #[case] expected: usize,
) {
    let bytes = datatype::array(3, [0; 3], 8, 1, &[2], &Leaf::Integer.bytes(Version::One));
    assert_eq!(
        DatatypeMessage::parse(&bytes[..available]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[test]
fn datatype_array_complex_encoder_rejects_sizes_that_overflow_at_selected_width() {
    let sequence = DatatypeMessage::new(
        DatatypeVersion::One,
        DatatypeProperties::VariableLengthSequence {
            base_type: Box::new(
                DatatypeMessage::parse(&Leaf::Integer.bytes(Version::One))
                    .unwrap()
                    .0,
            ),
        },
    )
    .unwrap();
    let array = DatatypeMessage::new(
        DatatypeVersion::Three,
        DatatypeProperties::Array {
            base_type: Box::new(sequence),
            dimensions: vec![u32::MAX / 10],
        },
    )
    .unwrap();
    assert_eq!(
        array.serialize(OffsetWidth::Eight).unwrap_err(),
        FormatError::LengthTooLarge {
            length: u64::from(u32::MAX / 10) * 16,
            length_size: 4
        }
    );
}

fn assert_complex(parent: &[u8]) {
    let bytes = datatype::complex(5, [1, 0, 0], 16, parent);
    let complex = DatatypeMessage::new(
        DatatypeVersion::Five,
        DatatypeProperties::Complex {
            component: Box::new(DatatypeMessage::parse(parent).unwrap().0),
        },
    )
    .unwrap();
    assert_encoding(&bytes, complex.clone(), OffsetWidth::Eight);
    let array = datatype::array(5, [0; 3], 32, 1, &[2], &bytes);
    assert_encoding(
        &array,
        DatatypeMessage::new(
            DatatypeVersion::Five,
            DatatypeProperties::Array {
                base_type: Box::new(complex),
                dimensions: vec![2],
            },
        )
        .unwrap(),
        OffsetWidth::Eight,
    );
}

fn assert_encoding(bytes: &[u8], expected: DatatypeMessage, width: OffsetWidth) {
    let mut input = bytes.to_vec();
    input.push(0xff);
    let (message, consumed) = DatatypeMessage::parse(&input).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(message, expected);
    assert_eq!(message.serialize(width).unwrap(), bytes);
}
