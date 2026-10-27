use core::num::NonZeroU32;
use core::num::NonZeroU64;
use core::num::NonZeroUsize;

use hdf5_pure::CompoundMember;
use hdf5_pure::Datatype;
use hdf5_pure::FillTime;
use hdf5_pure::FixedPointType;
use hdf5_pure::FormatError;
use hdf5_pure::MaxExtent;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;

#[test]
fn file_region_failures_use_the_public_error() {
    let range = FileRange::new(PhysicalAddress::new(u64::MAX - 4), 4).unwrap();
    let mismatch: FormatError = FileRegion::new(range, &[1, 2]).unwrap_err();
    assert_eq!(
        mismatch,
        FormatError::FileRegionLengthMismatch {
            range_length: 4,
            bytes_length: 2,
        }
    );
    assert_eq!(
        mismatch.to_string(),
        "file region length is 4 bytes, supplied buffer has 2 bytes"
    );
    let region = FileRegion::new(range, &[1, 2, 3, 4]).unwrap();
    let required = FileRange::new(PhysicalAddress::new(u64::MAX - 5), 2).unwrap();
    let insufficient: FormatError = region.subregion(required).unwrap_err();
    assert_eq!(
        insufficient,
        FormatError::InsufficientRegion {
            required_start: u64::MAX - 5,
            required_length: 2,
            provided_start: u64::MAX - 4,
            provided_length: 4,
        }
    );
    assert_eq!(
        insufficient.to_string(),
        "required file range at 18446744073709551610 with length 2 is outside supplied region \
         at 18446744073709551611 with length 4"
    );
}

#[test]
fn core_types_keep_their_original_paths() {
    let _: Option<Datatype> = None::<hdf5_pure_core::Datatype>;
    let _: Option<Datatype> = None::<hdf5_pure_format::Datatype>;
    let _: Option<FormatError> = None::<hdf5_pure_core::FormatError>;
    let _: Option<FormatError> = None::<hdf5_pure_format::FormatError>;
    let _: Option<CompoundMember> = None::<hdf5_pure_core::CompoundMember>;
    let _: Option<FillTime> = None::<hdf5_pure_core::FillTime>;
    let _: Option<FillTime> = None::<hdf5_pure_format::FillTime>;
    let _: Option<hdf5_pure::EnumMember> = None::<hdf5_pure_core::EnumMember>;
    let _: Option<hdf5_pure::FixedPointType> = None::<hdf5_pure_core::FixedPointType>;
    let _: Option<hdf5_pure::FixedPointType> = None::<hdf5_pure_format::FixedPointType>;
    let _: Option<hdf5_pure::FloatingPointType> = None::<hdf5_pure_core::FloatingPointType>;
    let _: Option<hdf5_pure::FloatingPointType> = None::<hdf5_pure_format::FloatingPointType>;
    let _: Option<hdf5_pure::BitFieldType> = None::<hdf5_pure_core::BitFieldType>;
    let _: Option<hdf5_pure::BitFieldType> = None::<hdf5_pure_format::BitFieldType>;
    let _: Option<hdf5_pure::TimeType> = None::<hdf5_pure_core::TimeType>;
    let _: Option<hdf5_pure::TimeType> = None::<hdf5_pure_format::TimeType>;
    let _: Option<hdf5_pure::FloatingPointField> = None::<hdf5_pure_core::FloatingPointField>;
    let _: Option<hdf5_pure::FloatingPointField> = None::<hdf5_pure_format::FloatingPointField>;
    let _: Option<hdf5_pure::FixedPointLayout> = None::<hdf5_pure_core::FixedPointLayout>;
    let _: Option<hdf5_pure::FloatingPointLayout> = None::<hdf5_pure_core::FloatingPointLayout>;
    let _: Option<hdf5_pure::DatatypeByteOrder> = None::<hdf5_pure_core::DatatypeByteOrder>;
    let _: Option<hdf5_pure::ReferenceType> = None::<hdf5_pure_core::ReferenceType>;
    let _: Option<hdf5_pure::CharacterSet> = None::<hdf5_pure_core::CharacterSet>;
    let _: Option<hdf5_pure::StringPadding> = None::<hdf5_pure_core::StringPadding>;
    let _: Option<MaxExtent> = None::<hdf5_pure_core::MaxExtent>;
    let _: Option<hdf5_pure::FileSpaceInfo> = None::<hdf5_pure_core::FileSpaceInfo>;
    let _: Option<hdf5_pure::FileSpaceInfo> = None::<hdf5_pure_format::FileSpaceInfo>;
    let _: Option<hdf5_pure::FileSpaceStrategy> = None::<hdf5_pure_core::FileSpaceStrategy>;
    let _: Option<hdf5_pure::OffsetWidth> = None::<hdf5_pure_core::OffsetWidth>;
    let _: Option<hdf5_pure::OffsetWidth> = None::<hdf5_pure_format::OffsetWidth>;
    let _: Option<hdf5_pure::LengthWidth> = None::<hdf5_pure_core::LengthWidth>;
    let _: Option<hdf5_pure::LengthWidth> = None::<hdf5_pure_format::LengthWidth>;
    assert_eq!(
        hdf5_pure::OBJECT_HEADER_MESSAGE_MAX,
        hdf5_pure_core::OBJECT_HEADER_MESSAGE_MAX
    );
}

#[test]
fn fill_time_api_is_public() {
    assert_eq!(FillTime::default(), FillTime::IfSet);
    assert_eq!(FillTime::Alloc.to_u8(), 0);
    assert_eq!(FillTime::Never.to_u8(), 1);
    assert_eq!(FillTime::IfSet.to_u8(), 2);
    assert!(FillTime::Alloc.is_alloc());
    assert!(FillTime::Never.is_never());
    assert!(FillTime::IfSet.is_if_set());
    assert_eq!(FillTime::try_from(0), Ok(FillTime::Alloc));
}

#[cfg(feature = "std")]
#[test]
fn std_message_type_reexport_and_base_address_get_are_public() {
    let _: Option<hdf5_pure::MessageType> = None::<hdf5_pure_core::MessageType>;
    let _: Option<hdf5_pure::BaseAddress> = None::<hdf5_pure_core::BaseAddress>;
    let _: Option<hdf5_pure::BaseAddress> = None::<hdf5_pure_format::BaseAddress>;
    let _: Option<hdf5_pure::Superblock> = None::<hdf5_pure_core::Superblock>;
    let _: Option<hdf5_pure::Superblock> = None::<hdf5_pure_format::Superblock>;
    fn get(base: hdf5_pure::BaseAddress) -> u64 {
        base.get()
    }
    let _: fn(hdf5_pure::BaseAddress) -> u64 = get;
    assert_eq!(hdf5_pure::MessageType::DATATYPE.to_u16(), 3);
    assert_eq!(
        hdf5_pure::MessageType::from_u16(3),
        hdf5_pure::MessageType::DATATYPE
    );
}

#[cfg(feature = "std")]
#[test]
#[allow(deprecated)]
fn deprecated_message_type_names_remain_usable() {
    let named: hdf5_pure::MessageType = hdf5_pure::MessageType::Datatype;
    let unknown: hdf5_pure::MessageType = hdf5_pure::MessageType::Unknown(0x00FF);
    assert_eq!(named, hdf5_pure::MessageType::DATATYPE);
    assert_eq!(unknown.to_u16(), 0x00FF);
}

#[test]
fn existing_error_signatures_and_shapes_compile() {
    fn element_size(datatype: &Datatype) -> Result<NonZeroU32, FormatError> {
        datatype.element_size()
    }
    fn display_and_clone<T: core::fmt::Display + Clone>(value: &T) -> String {
        value.clone().to_string()
    }
    let datatype = Datatype::String {
        size: 4,
        padding: hdf5_pure::StringPadding::NullPad,
        charset: hdf5_pure::CharacterSet::Utf8,
    };
    assert_eq!(element_size(&datatype).unwrap().get(), datatype.type_size());
    assert_eq!(display_and_clone(&datatype), "string[4] utf8 null-pad");
    let fixed = FixedPointType::i16().layout();
    assert_eq!(fixed.bit_precision(), 16);
    let float = hdf5_pure::FloatingPointLayout::IEEE754_BINARY32;
    assert_eq!(float.exponent_location(), 23);
    assert_eq!(float.exponent_size(), 8);
    assert_eq!(float.mantissa_location(), 0);
    assert_eq!(float.mantissa_size(), 23);
    assert_eq!(float.exponent_bias(), 127);
    let integer = Datatype::FixedPoint(
        FixedPointType::from_layout(2, hdf5_pure::DatatypeByteOrder::LittleEndian, fixed).unwrap(),
    );
    assert_eq!(integer.element_size().unwrap().get(), 2);
    assert_eq!(
        hdf5_pure::DatatypeByteOrder::BigEndian,
        hdf5_pure_core::DatatypeByteOrder::BigEndian
    );
    assert_eq!(
        hdf5_pure::ReferenceType::Object,
        hdf5_pure_core::ReferenceType::Object
    );
    assert_eq!(
        hdf5_pure::StringPadding::SpacePad,
        hdf5_pure_core::StringPadding::SpacePad
    );
    assert_eq!(MaxExtent::Unlimited.size(), None);
    assert_eq!(MaxExtent::Fixed(3).size(), Some(3));
    assert_eq!(hdf5_pure::OBJECT_HEADER_MESSAGE_MAX, usize::from(u16::MAX));
    assert_eq!(
        FormatError::SignatureNotFound,
        hdf5_pure_core::FormatError::SignatureNotFound
    );
    let error = FormatError::UnexpectedEof {
        expected: 8,
        available: 4,
    };
    assert!(matches!(
        error,
        FormatError::UnexpectedEof {
            expected: 8,
            available: 4
        }
    ));
}

#[rstest::rstest]
#[case::source_eof(
    FormatError::UnexpectedSourceEof { expected: 4_294_967_600, available: 4_294_967_552 },
    "unexpected source EOF: need end position 4294967600, source extent is 4294967552"
)]
#[case::missing_heap_block(
    FormatError::MissingFractalHeapBlock { heap_offset: 4_294_967_536, object_length: 64 },
    "no allocated fractal heap block contains object at heap offset 4294967536 with length 64"
)]
#[case::decoded_size_limit(
    FormatError::DecodedSizeLimitExceeded {
        expected_decoded_size: 4_294_967_600,
        filter_id: hdf5_pure_filter::__private::FILTER_DEFLATE,
        output_limit: 4_294_967_604,
        minimum_output_size: 4_294_967_605,
    },
    "filter 1 output needs at least 4294967605 bytes, above limit 4294967604 for a payload expected to decode to 4294967600 bytes"
)]
#[case::resource_limit(
    FormatError::ResourceLimit { limit: 4_294_967_600, required: 4_294_967_616 },
    "resource requires 4294967616, above limit 4294967600"
)]
fn source_and_heap_diagnostics_preserve_full_width_positions(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

#[test]
fn public_member_fields_remain_nameable() {
    fn compound(member: &CompoundMember) -> (&str, u64, &Datatype) {
        (&member.name, member.byte_offset, &member.datatype)
    }
    fn enumeration(member: &hdf5_pure::EnumMember) -> (&str, &[u8]) {
        (&member.name, &member.value)
    }
    let _: fn(&CompoundMember) -> (&str, u64, &Datatype) = compound;
    let _: fn(&hdf5_pure::EnumMember) -> (&str, &[u8]) = enumeration;
}

#[test]
fn filter_errors_convert_to_the_public_error() {
    let filter_error: FormatError = hdf5_pure_filter::__private::Error::UnsupportedFilter(1).into();
    assert_eq!(filter_error, FormatError::UnsupportedFilter(1));
}

#[test]
fn width_and_undefined_address_apis_are_public() {
    assert_eq!(
        hdf5_pure::OffsetWidth::try_from(2),
        Ok(hdf5_pure::OffsetWidth::Two)
    );
    assert_eq!(
        hdf5_pure::OffsetWidth::try_from(4),
        Ok(hdf5_pure::OffsetWidth::Four)
    );
    assert_eq!(
        hdf5_pure::OffsetWidth::try_from(8),
        Ok(hdf5_pure::OffsetWidth::Eight)
    );
    assert_eq!(
        hdf5_pure::OffsetWidth::try_from(1),
        Err(FormatError::InvalidOffsetSize(1))
    );
    assert_eq!(hdf5_pure::OffsetWidth::Two.get(), 2);
    assert_eq!(hdf5_pure::OffsetWidth::Four.get(), 4);
    assert_eq!(hdf5_pure::OffsetWidth::Eight.get(), 8);

    assert_eq!(
        hdf5_pure::LengthWidth::try_from(2),
        Ok(hdf5_pure::LengthWidth::Two)
    );
    assert_eq!(
        hdf5_pure::LengthWidth::try_from(4),
        Ok(hdf5_pure::LengthWidth::Four)
    );
    assert_eq!(
        hdf5_pure::LengthWidth::try_from(8),
        Ok(hdf5_pure::LengthWidth::Eight)
    );
    assert_eq!(
        hdf5_pure::LengthWidth::try_from(1),
        Err(FormatError::InvalidLengthSize(1))
    );
    assert_eq!(hdf5_pure::LengthWidth::Two.get(), 2);
    assert_eq!(hdf5_pure::LengthWidth::Four.get(), 4);
    assert_eq!(hdf5_pure::LengthWidth::Eight.get(), 8);

    assert_eq!(
        hdf5_pure::StoredAddress::undefined(hdf5_pure::OffsetWidth::Two).get(),
        0xFFFF
    );
    assert_eq!(
        hdf5_pure::StoredAddress::undefined(hdf5_pure::OffsetWidth::Four).get(),
        0xFFFF_FFFF
    );
    assert_eq!(
        hdf5_pure::StoredAddress::undefined(hdf5_pure::OffsetWidth::Eight).get(),
        u64::MAX
    );
    assert!(
        hdf5_pure::StoredAddress::undefined(hdf5_pure::OffsetWidth::Two)
            .is_undefined(hdf5_pure::OffsetWidth::Two)
    );
    assert!(
        !hdf5_pure::StoredAddress::undefined(hdf5_pure::OffsetWidth::Two)
            .is_undefined(hdf5_pure::OffsetWidth::Four)
    );
}

#[rstest::rstest]
#[case::logical_data(
    FormatError::LogicalDataSizeMismatch { expected: 4_294_967_312, actual: 4_294_967_328 },
    "data size mismatch: expected 4294967312 bytes, got 4294967328 bytes"
)]
#[case::logical_shape(
    FormatError::LogicalShapeDataMismatch {
        expected: 8_589_934_592,
        actual: 4_294_967_296,
        element_size: NonZeroU64::new(4_294_967_296).unwrap(),
    },
    "shape/data mismatch: shape requires 2 elements (8589934592 bytes), but 1 elements \
     (4294967296 bytes) were supplied"
)]
#[case::buffer_data(
    FormatError::DataSizeMismatch { expected: 16, actual: 8 },
    "data size mismatch: expected 16 bytes, got 8 bytes"
)]
#[case::buffer_shape(
    FormatError::ShapeDataMismatch {
        expected: 32,
        actual: 24,
        element_size: NonZeroUsize::new(8).unwrap(),
    },
    "shape/data mismatch: shape requires 4 elements (32 bytes), but 3 elements (24 bytes) \
     were supplied"
)]
fn size_mismatch_diagnostics_preserve_counts_and_element_widths(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}
