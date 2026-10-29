use rstest::rstest;

use super::*;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;

#[cfg(feature = "std")]
mod display;

#[rstest]
#[case::deflate(1, FilterId::Deflate)]
#[case::shuffle(2, FilterId::Shuffle)]
#[case::fletcher32(3, FilterId::Fletcher32)]
#[case::szip(4, FilterId::Szip)]
#[case::nbit(5, FilterId::Nbit)]
#[case::scale_offset(6, FilterId::ScaleOffset)]
#[case::lzf(32000, FilterId::Lzf)]
#[case::zfp(32013, FilterId::Zfp)]
fn parses_named_ids(#[case] raw: u16, #[case] id: FilterId) {
    assert_eq!(FilterId::try_from(raw).unwrap(), id);
    assert_eq!(id.get(), raw);
    assert_eq!(
        CustomFilterId::try_from(raw).unwrap_err(),
        FormatError::InvalidCustomFilterId { id: raw }
    );
}

#[rstest]
#[case::id_seven(7)]
#[case::byte_limit(255)]
#[case::above_byte(256)]
#[case::high_bit(32768)]
#[case::field_limit(u16::MAX)]
fn preserves_custom_ids(#[case] raw: u16) {
    let custom = CustomFilterId::try_from(raw).unwrap();
    assert_eq!(custom.get(), raw);
    let id = FilterId::try_from(raw).unwrap();
    assert_eq!(id, FilterId::Custom(custom));
    assert_eq!(id.get(), raw);
}

#[test]
fn rejects_reserved_id() {
    assert_eq!(
        FilterId::try_from(0).unwrap_err(),
        FormatError::InvalidFilterId { id: 0 }
    );
    assert_eq!(
        CustomFilterId::try_from(0).unwrap_err(),
        FormatError::InvalidFilterId { id: 0 }
    );
}

#[rstest]
#[case::deflate_zero(FilterId::Deflate, vec![0], FilterParameters::Deflate(DeflateLevel::Zero))]
#[case::deflate_six(FilterId::Deflate, vec![6], FilterParameters::Deflate(DeflateLevel::Six))]
#[case::deflate_nine(FilterId::Deflate, vec![9], FilterParameters::Deflate(DeflateLevel::Nine))]
#[case::shuffle_width(FilterId::Shuffle, vec![4], FilterParameters::Shuffle(NonZeroU32::new(4).unwrap()))]
#[case::fletcher32(FilterId::Fletcher32, vec![], FilterParameters::Fletcher32)]
#[case::scale_offset(FilterId::ScaleOffset, scale_offset_values(), FilterParameters::ScaleOffset(ScaleOffsetParameters { mode: ScaleOffset::Integer(0) }))]
#[case::lzf_opaque(FilterId::Lzf, vec![4, 0x0105, 256], FilterParameters::Opaque)]
#[case::zfp_opaque(FilterId::Zfp, vec![1, 2, 3, 4, 5, 6], FilterParameters::Opaque)]
#[case::szip_opaque(FilterId::Szip, vec![1, 2, 3, 4], FilterParameters::Opaque)]
#[case::nbit_opaque(FilterId::Nbit, vec![1, 2], FilterParameters::Opaque)]
#[case::custom_opaque(FilterId::try_from(40000).unwrap(), vec![u32::MAX, 0, 7], FilterParameters::Opaque)]
fn interprets_parameters_without_losing_wire_metadata(
    #[case] id: FilterId,
    #[case] values: Vec<u32>,
    #[case] parameters: FilterParameters,
    #[values(0x8000, 0x8001)] flags: u16,
) {
    let filter = Filter::from_description(FilterDescription {
        filter_id: id.get(),
        name: Some("recorded filter name".into()),
        flags,
        client_data: values.clone(),
    })
    .unwrap();
    assert_eq!(filter.id, id);
    assert_eq!(filter.parameters, parameters);
    assert_eq!(filter.flags(), flags);
    assert_eq!(
        filter.is_optional,
        flags & crate::filter_pipeline::H5Z_FLAG_OPTIONAL != 0
    );
    assert_eq!(filter.name.as_deref(), Some("recorded filter name"));
    assert_eq!(filter.client_data(), values);
}

#[rstest]
#[case::missing_deflate(FilterId::Deflate, vec![], FormatError::InvalidFilterParameterCount { filter_id: FILTER_DEFLATE, expected: 1, actual: 0 })]
#[case::excess_deflate(FilterId::Deflate, vec![6, 7], FormatError::InvalidFilterParameterCount { filter_id: FILTER_DEFLATE, expected: 1, actual: 2 })]
#[case::missing_shuffle(FilterId::Shuffle, vec![], FormatError::InvalidFilterParameterCount { filter_id: FILTER_SHUFFLE, expected: 1, actual: 0 })]
#[case::excess_shuffle(FilterId::Shuffle, vec![4, 8], FormatError::InvalidFilterParameterCount { filter_id: FILTER_SHUFFLE, expected: 1, actual: 2 })]
#[case::zero_shuffle_width(FilterId::Shuffle, vec![0], FormatError::InvalidShuffleElementSize { size: 0 })]
#[case::excess_fletcher32(FilterId::Fletcher32, vec![1], FormatError::InvalidFilterParameterCount { filter_id: FILTER_FLETCHER32, expected: 0, actual: 1 })]
#[case::short_scale_offset(FilterId::ScaleOffset, vec![2, 0], FormatError::InvalidFilterParameterCount { filter_id: FILTER_SCALEOFFSET, expected: 20, actual: 2 })]
fn rejects_malformed_known_parameters(
    #[case] id: FilterId,
    #[case] values: Vec<u32>,
    #[case] expected: FormatError,
) {
    assert_eq!(FilterParameters::parse(id, &values).unwrap_err(), expected);
}

#[rstest]
#[case::above_level_limit(10)]
#[case::above_byte_limit(256)]
#[case::field_limit(u32::MAX)]
fn rejects_invalid_deflate_metadata(#[case] level: u32) {
    assert_eq!(
        FilterParameters::parse(FilterId::Deflate, &[level]).unwrap_err(),
        FormatError::InvalidDeflateLevel { level }
    );
}

#[rstest]
#[case::excess_integer_precision(1, 33, FormatError::InvalidScaleOffsetIntegerPrecision { bits: 33, element_size: 4 })]
#[case::zero_element_count(2, 0, FormatError::InvalidScaleOffsetElementCount { count: 0 })]
#[case::invalid_mode(3, 2, FormatError::InvalidScaleOffsetMode { mode: 2, class: 2 })]
#[case::zero_element_size(4, 0, FormatError::InvalidScaleOffsetElementSize { size: 0 })]
#[case::oversized_element(4, 9, FormatError::InvalidScaleOffsetElementSize { size: 9 })]
#[case::invalid_sign(5, 2, FormatError::InvalidScaleOffsetIntegerSign { sign: 2 })]
#[case::invalid_byte_order(6, 2, FormatError::InvalidScaleOffsetByteOrder { order: 2 })]
#[case::invalid_fill_availability(7, 2, FormatError::InvalidScaleOffsetFillAvailability { value: 2 })]
fn rejects_invalid_scale_offset_fields(
    #[case] index: usize,
    #[case] value: u32,
    #[case] expected: FormatError,
) {
    let mut values = scale_offset_values();
    values[index] = value;
    assert_eq!(
        FilterParameters::parse(FilterId::ScaleOffset, &values).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::one_byte(1)]
#[case::three_bytes(3)]
#[case::five_bytes(5)]
fn rejects_invalid_scale_offset_float_size(#[case] size: u32) {
    let mut values = scale_offset_values();
    values[0] = 0;
    values[3] = 1;
    values[4] = size;
    assert_eq!(
        FilterParameters::parse(FilterId::ScaleOffset, &values).unwrap_err(),
        FormatError::InvalidScaleOffsetFloatSize { size }
    );
}

#[rstest]
#[case::decimal_scale(0, Some(ScaleOffset::FloatDScale(-2)))]
#[case::opaque_mode_one(1, None)]
fn distinguishes_supported_scale_offset_modes(
    #[case] mode: u32,
    #[case] expected: Option<ScaleOffset>,
    #[values(0, u32::MAX)] sign: u32,
) {
    let mut values = scale_offset_values();
    values[0] = mode;
    values[1] = u32::from_ne_bytes((-2_i32).to_ne_bytes());
    values[3] = 1;
    values[5] = sign;
    let parsed = FilterParameters::parse(FilterId::ScaleOffset, &values).unwrap();
    match (parsed, expected) {
        (FilterParameters::ScaleOffset(parameters), Some(mode)) => {
            assert_eq!(parameters.mode(), mode)
        }
        (FilterParameters::Opaque, None) => {}
        (parsed, expected) => panic!("expected {expected:?}, got {parsed:?}"),
    }
}

fn scale_offset_values() -> Vec<u32> {
    let mut values = vec![0; 20];
    values[..8].copy_from_slice(&[2, 0, 4, 0, 4, 1, 0, 1]);
    values
}

#[rstest]
#[case::btree_v1(ChunkIndexLayout::BTreeV1 { address: None }, ChunkIndex::BTreeV1)]
#[case::single_chunk(
    ChunkIndexLayout::SingleChunk { filtered: None, address: Some(StoredAddress::new(0x100)) },
    ChunkIndex::SingleChunk
)]
#[case::implicit(ChunkIndexLayout::Implicit { address: Some(StoredAddress::new(0x100)) }, ChunkIndex::Implicit)]
#[case::fixed_array(ChunkIndexLayout::FixedArray { address: Some(StoredAddress::new(0x100)) }, ChunkIndex::FixedArray)]
#[case::extensible_array(
    ChunkIndexLayout::ExtensibleArray { creation: ExtensibleArrayCreationParameters::canonical(), address: Some(StoredAddress::new(0x100)) },
    ChunkIndex::ExtensibleArray
)]
#[case::btree_v2(ChunkIndexLayout::BTreeV2 { address: Some(StoredAddress::new(0x100)) }, ChunkIndex::BTreeV2)]
fn every_parsed_chunk_index_maps_to_a_kind(
    #[case] parsed: ChunkIndexLayout,
    #[case] kind: ChunkIndex,
) {
    assert_eq!(ChunkIndex::from_layout(parsed), kind);
}

#[test]
fn only_extensible_array_supports_inplace_append() {
    assert!(ChunkIndex::ExtensibleArray.supports_inplace_append());
    for idx in [
        ChunkIndex::BTreeV1,
        ChunkIndex::SingleChunk,
        ChunkIndex::Implicit,
        ChunkIndex::FixedArray,
        ChunkIndex::BTreeV2,
    ] {
        assert!(!idx.supports_inplace_append());
    }
}
