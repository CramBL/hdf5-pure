use hdf5_pure_core::__private;
use hdf5_pure_core::Datatype;
use hdf5_pure_core::FixedPointType;
use hdf5_pure_core::FormatError;

#[rstest::rstest]
#[case::decoded_size_limit(
    FormatError::DecodedSizeLimitExceeded {
        expected_decoded_size: 4_294_967_600,
        filter_id: 32_001,
        output_limit: 4_294_967_604,
        minimum_output_size: 4_294_967_605,
    },
    "filter 32001 output needs at least 4294967605 bytes, above limit 4294967604 for a payload expected to decode to 4294967600 bytes"
)]
#[case::resource_limit(
    FormatError::ResourceLimit { limit: 4_294_967_600, required: 4_294_967_616 },
    "resource requires 4294967616, above limit 4294967600"
)]
#[case::decoded_size(
    FormatError::DecodedSizeMismatch { expected: 4_294_967_600, actual: 4_294_967_616 },
    "decoded payload has 4294967616 bytes, expected 4294967600"
)]
#[case::wrong_layout(FormatError::WrongLayout, "reference belongs to another layout")]
#[case::reserved_heap_bits(
    FormatError::InvalidFractalHeapIdFlags(0x01),
    "reserved bits in fractal heap ID: 0x01"
)]
#[case::heap_id_version(
    FormatError::UnsupportedFractalHeapIdVersion(1),
    "unsupported fractal heap ID version: 1"
)]
fn huge_object_diagnostics_preserve_their_fields(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

#[rstest::rstest]
#[case::duplicate_creation_index(
    FormatError::DuplicateAttributeCreationIndex(17),
    "duplicate attribute creation index 17"
)]
#[case::duplicate_name(FormatError::DuplicateAttributeName("a\nb".into()), "duplicate attribute name \"a\\nb\"")]
#[case::creation_counter(FormatError::InvalidAttributeCreationCounter { index: 7, next: 3 }, "attribute creation index 7 is not below the next index 3")]
#[case::phase_change(FormatError::InvalidAttributePhaseChange { max_compact: 7, min_dense: 8 }, "attribute minimum dense count 8 exceeds maximum compact count 7")]
fn attribute_creation_diagnostics_preserve_their_fields(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

#[rstest::rstest]
#[case::duplicate(FormatError::DuplicateGroupInfo, "duplicate Group Info message")]
#[case::flags(
    FormatError::InvalidGroupInfoFlags(0x80),
    "reserved bits in Group Info flags: 0x80"
)]
#[case::length(FormatError::InvalidGroupInfoLength { expected: 6, actual: 4 }, "Group Info message has 4 bytes, expected 6")]
#[case::thresholds(FormatError::InvalidGroupInfoThresholds { max_compact: 3, min_dense: 4 }, "Group Info minimum dense count 4 exceeds maximum compact count 3")]
#[case::version(
    FormatError::InvalidGroupInfoVersion(1),
    "unsupported Group Info version: 1"
)]
fn group_info_diagnostics_preserve_their_fields(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

#[rstest::rstest]
#[case::chunk_length(
    FormatError::InvalidObjectHeaderChunkLength { index: 17, length: 4_294_967_600 },
    "object-header chunk 17 cannot represent its records in 4294967600 bytes"
)]
#[case::allocation_length(
    FormatError::ObjectHeaderAllocationLengthMismatch { expected: 4_294_967_600, actual: 4_294_967_616 },
    "object-header chunks cover 4294967616 bytes, expected 4294967600"
)]
fn object_header_allocation_diagnostics_preserve_their_fields(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

#[test]
fn enum_member_parts_borrow_the_complete_member() {
    let member = hdf5_pure_core::__private::enum_member("value".into(), vec![1, 2, 3, 4]);
    let (name, value) = hdf5_pure_core::__private::enum_member_parts(&member);
    assert_eq!((name, value), ("value", &[1, 2, 3, 4][..]));
    assert_eq!(name.as_ptr(), member.name.as_ptr());
    assert_eq!(value.as_ptr(), member.value.as_ptr());
}

#[rstest::rstest]
#[case::empty(FormatError::InvalidEnumerationMemberCount { count: 0 }, "invalid enumeration member count 0")]
#[case::empty_name(FormatError::EmptyEnumMemberName { index: 2 }, "enumeration member 2 has an empty name")]
#[case::duplicate_name(FormatError::DuplicateEnumMemberName("x\ny".into()), "duplicate enumeration member name \"x\\ny\"")]
#[case::duplicate_value(FormatError::DuplicateEnumMemberValue { first: "x\ny".into(), second: "z".into() }, "enumeration members \"x\\ny\" and \"z\" have the same stored value")]
fn enumeration_diagnostics_preserve_their_fields(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

#[test]
fn compound_member_parts_borrow_the_complete_member() {
    let datatype = Datatype::FixedPoint(FixedPointType::u32());
    let member = __private::compound_member("field".into(), 3, datatype.clone());
    let (name, offset, borrowed) = __private::compound_member_parts(&member);
    assert_eq!((name, offset, borrowed), ("field", 3, &datatype));
    assert_eq!(name.as_ptr(), member.name.as_ptr());
    assert!(core::ptr::eq(borrowed, &member.datatype));
}

#[rstest::rstest]
#[case::duplicate_member(FormatError::DuplicateGlobalHeapMemberIndex { index: 400 }, "global heap members hold index 400 more than once")]
#[case::insufficient_capacity(FormatError::GlobalHeapCapacityExceeded { capacity: HEAP_MINIMUM, required: HEAP_MINIMUM + HEAP_ALIGNMENT }, "global heap members require 4104 bytes, beyond capacity 4096")]
#[case::minimum_capacity(FormatError::GlobalHeapCapacityTooSmall { capacity: HEAP_MINIMUM - HEAP_ALIGNMENT, minimum: HEAP_MINIMUM }, "global heap capacity 4088 is below the minimum 4096")]
#[case::alignment(FormatError::UnalignedGlobalHeapCapacity { capacity: HEAP_MINIMUM + 1, alignment: HEAP_ALIGNMENT }, "global heap capacity 4097 is not aligned to 8 bytes")]
fn global_heap_preparation_errors_report_unplaced_inputs(
    #[case] error: FormatError,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

// Version 1 collections use this geometry ("Global Heap", format specification, version 4.0).
const HEAP_MINIMUM: u64 = 4096;

// Version 1 collections use this geometry ("Global Heap", format specification, version 4.0).
const HEAP_ALIGNMENT: u64 = 8;
