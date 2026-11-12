use hdf5_pure_core::__private;
use hdf5_pure_core::ComplexType;
use hdf5_pure_core::Datatype;
use hdf5_pure_core::DatatypeByteOrder;
use hdf5_pure_core::FloatingPointType;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::MantissaNormalization;
use hdf5_pure_core::VariableLengthType;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::Class;
use test_util::datatype::float_context::Context;
use test_util::datatype::leaf::Leaf;
use test_util::datatype::leaf::Order;
use test_util::datatype::leaf::Version;

#[rstest]
#[case::little(LITTLE_ENDIAN_ORDER, Ok(DatatypeByteOrder::LittleEndian))]
#[case::big(BIG_ENDIAN_ORDER, Ok(DatatypeByteOrder::BigEndian))]
#[case::vax(VAX_ORDER, Ok(DatatypeByteOrder::Vax))]
#[case::reserved(RESERVED_ORDER, Err(FormatError::InvalidByteOrder(RESERVED_ORDER)))]
fn ordinary_float_parser_applies_order_rules_in_every_context(
    #[case] bits: u8,
    #[case] order: Result<DatatypeByteOrder, FormatError>,
    #[values(
        Version::One,
        Version::Two,
        Version::Three,
        Version::Four,
        Version::Five
    )]
    version: Version,
    #[values(
        Context::Direct,
        Context::CompoundOne,
        Context::CompoundTwo,
        Context::CompoundThree,
        Context::EnumerationOne,
        Context::EnumerationTwo,
        Context::EnumerationThree,
        Context::ArrayTwo,
        Context::ArrayThree,
        Context::Sequence,
        Context::Complex
    )]
    context: Context,
) {
    let mut parent = Leaf::Float(Order::Little).bytes(version);
    parent[1] |= bits;
    let bytes = context.bytes(&parent);
    let order = match order {
        Ok(DatatypeByteOrder::Vax) if version.get() < Version::Three.get() => {
            Err(FormatError::InvalidDatatypeVersion {
                class: Class::FLOATING_POINT.0,
                version: version.get(),
            })
        }
        order => order,
    };
    let order = match order {
        Ok(order) => order,
        Err(error) => {
            let error = if matches!(context, Context::Complex) {
                FormatError::InvalidComplexParentFlags {
                    version: version.get(),
                    flags: u32::from(parent[1]) | (u32::from(parent[2]) << 8),
                }
            } else {
                error
            };
            assert_eq!(format::parse_datatype(&bytes).unwrap_err(), error);
            return;
        }
    };
    let component = FloatingPointType::new(
        8,
        order,
        MantissaNormalization::LeadingOne,
        63,
        3,
        61,
        52,
        11,
        3,
        49,
        1023,
    )
    .unwrap()
    .with_padding(true, true, true);
    let datatype = Datatype::FloatingPoint(component);
    let expected = match context {
        Context::Direct => datatype,
        Context::CompoundOne | Context::CompoundTwo | Context::CompoundThree => {
            Datatype::Compound {
                size: 8,
                members: vec![__private::compound_member(
                    if matches!(context, Context::CompoundOne) {
                        "arr"
                    } else {
                        "x"
                    }
                    .into(),
                    0,
                    datatype,
                )],
            }
        }
        Context::EnumerationOne | Context::EnumerationTwo | Context::EnumerationThree => {
            Datatype::Enumeration {
                size: 8,
                base_type: Box::new(datatype),
                members: vec![__private::enum_member("x".into(), vec![0; 8])],
            }
        }
        Context::ArrayTwo | Context::ArrayThree => Datatype::Array {
            base_type: Box::new(datatype),
            dimensions: vec![1],
        },
        Context::Sequence => Datatype::VariableLength(VariableLengthType::sequence(datatype)),
        Context::Complex => Datatype::Complex(ComplexType::new(component).unwrap()),
    };
    assert_eq!(
        format::parse_datatype(&bytes).unwrap(),
        (expected, bytes.len())
    );
}

#[rstest]
#[case::reserved_low(FLOAT_RESERVED_BIT_SEVEN, 0x0000_3f9e)]
#[case::reserved_high(FLOAT_RESERVED_BIT_SIXTEEN, 0x0001_3f1e)]
fn ordinary_float_parser_rejects_reserved_flags(#[case] extra: [u8; 3], #[case] flags: u32) {
    let mut bytes = Leaf::Float(Order::Little).bytes(Version::Five);
    bytes[1] |= extra[0];
    bytes[2] |= extra[1];
    bytes[3] |= extra[2];
    assert_eq!(
        format::parse_datatype(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeFlags {
            class: Class::FLOATING_POINT.0,
            version: Version::Five.get(),
            flags,
        }
    );
}

#[rstest]
#[case::zero(0)]
#[case::six(6)]
#[case::fifteen(15)]
fn ordinary_float_parser_rejects_unsupported_versions(#[case] version: u8) {
    let mut bytes = Leaf::Float(Order::Little).bytes(Version::One);
    bytes[0] = datatype::header(Class::FLOATING_POINT, version, [0; 3], 0)[0];
    assert_eq!(
        format::parse_datatype(&bytes).unwrap_err(),
        FormatError::InvalidDatatypeMessageVersion(version)
    );
}

/// The little-endian floating-point order bits for datatype message versions 1 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const LITTLE_ENDIAN_ORDER: u8 = 0x00;

/// The big-endian floating-point order bits for datatype message versions 1 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const BIG_ENDIAN_ORDER: u8 = 0x01;

/// The VAX floating-point order bits for datatype message versions 3 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VAX_ORDER: u8 = 0x41;

/// The reserved floating-point order pattern tested at datatype message versions 1 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const RESERVED_ORDER: u8 = 0x40;

/// Reserved floating-point class bit 7 for datatype message versions 1 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_RESERVED_BIT_SEVEN: [u8; 3] = [0x80, 0, 0];

/// Reserved floating-point class bit 16 for datatype message versions 1 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const FLOAT_RESERVED_BIT_SIXTEEN: [u8; 3] = [0, 0, 1];
