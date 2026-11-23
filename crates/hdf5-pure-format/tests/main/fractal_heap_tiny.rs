use core::ptr;

use hdf5_pure_core::FormatError;
use hdf5_pure_format::__private::{
    FractalHeapHeader, FractalHeapIdKind, FractalHeapIdLayout, LengthWidth, OffsetWidth,
    TinyHeapIdEncoding,
};
use rstest::rstest;
use test_util::fractal_heap::{Header, TinyId};
use test_util::widths::Widths;

#[rstest]
#[case::minimum_width(2, 1, &[0x20])]
#[case::width_17(17, 16, &[0x2f])]
#[case::width_18(18, 16, &[0x2f])]
#[case::normal_one_byte_payload(18, 1, &[0x20])]
#[case::extended_one_byte_payload(19, 1, &[0x20, 0x00])]
#[case::width_19(19, 17, &[0x20, 0x10])]
#[case::extended_256_byte_payload(258, 256, &[0x20, 0xff])]
#[case::extended_257_byte_payload(260, 257, &[0x21, 0x00])]
#[case::maximum_extended_payload(4098, 4096, &[0x2f, 0xff])]
fn tiny_id_encoding_matches_independent_bytes(
    #[case] width: u16,
    #[case] length: usize,
    #[case] prefix: &[u8],
) {
    let payload = payload(length);
    let expected = TinyId::new(width, &payload).unwrap().build();
    assert_eq!(&expected[..prefix.len()], prefix);
    let encoding = TinyHeapIdEncoding::new(width, &payload).unwrap();
    assert_eq!(encoding.encoded_size(), usize::from(width));

    let mut output = vec![GUARD; 1 + usize::from(width) + 2];
    let end = 1 + usize::from(width);
    assert_eq!(
        encoding.write_into(&mut output[1..]).unwrap(),
        usize::from(width)
    );
    let mut guarded = vec![GUARD; output.len()];
    guarded[1..end].copy_from_slice(&expected);
    assert_eq!(output, guarded);
}

#[rstest]
#[case::normal(18, 16)]
#[case::extended(19, 1)]
#[case::maximum_extended_payload(4098, 4096)]
fn tiny_id_encoding_preserves_short_destinations_and_can_be_reused(
    #[case] width: u16,
    #[case] length: usize,
) {
    let payload = payload(length);
    let encoding = TinyHeapIdEncoding::new(width, &payload).unwrap();
    let required = usize::from(width);
    let mut output = vec![GUARD; required + 2];
    for available in [0, required - 1] {
        assert_eq!(
            encoding
                .write_into(&mut output[1..1 + available])
                .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: required,
                available
            },
        );
        assert_eq!(output, vec![GUARD; required + 2]);
    }

    let end = 1 + required;
    assert_eq!(encoding.write_into(&mut output[1..end]).unwrap(), required);
    assert_eq!(output[0], GUARD);
    assert_eq!(output[end], GUARD);
    let view = layout(width).parse(&output[1..end]).unwrap();
    assert_eq!(view.kind(), FractalHeapIdKind::Tiny { bytes: &payload });
}

#[rstest]
#[case::zero_width(0, 1, FormatError::TinyHeapIdWidthTooSmall(0))]
#[case::prefix_only_width(1, 1, FormatError::TinyHeapIdWidthTooSmall(1))]
#[case::width_precedes_empty_payload(1, 0, FormatError::TinyHeapIdWidthTooSmall(1))]
#[case::normal_empty(17, 0, FormatError::EmptyTinyHeapIdPayload)]
#[case::extended_empty(19, 0, FormatError::EmptyTinyHeapIdPayload)]
#[case::normal_length_field_limit(18, 17, FormatError::TinyHeapIdPayloadLengthTooLarge { length: 17, maximum: 16 })]
#[case::extended_length_field_limit(4099, 4097, FormatError::TinyHeapIdPayloadLengthTooLarge { length: 4097, maximum: 4096 })]
#[case::length_field_limit_precedes_capacity(17, 17, FormatError::TinyHeapIdPayloadLengthTooLarge { length: 17, maximum: 16 })]
#[case::normal_capacity(16, 16, FormatError::TinyHeapIdPayloadExceedsCapacity { length: 16, capacity: 15 })]
#[case::extended_capacity(19, 18, FormatError::TinyHeapIdPayloadExceedsCapacity { length: 18, capacity: 17 })]
fn tiny_id_preparation_rejects_invalid_inputs(
    #[case] width: u16,
    #[case] length: usize,
    #[case] expected: FormatError,
) {
    assert_eq!(
        TinyHeapIdEncoding::new(width, &payload(length)).unwrap_err(),
        expected,
    );
}

#[rstest]
#[case::short_normal(7, 4, 1)]
#[case::normal_full(17, 16, 1)]
#[case::last_normal_width(18, 1, 1)]
#[case::first_extended_width(19, 1, 2)]
#[case::longer_extended_id(20, 5, 2)]
#[case::extended_256_byte_payload(258, 256, 2)]
#[case::extended_257_byte_payload(260, 257, 2)]
#[case::maximum_extended_payload(4098, 4096, 2)]
fn tiny_id_decoding_borrows_only_its_payload(
    #[case] width: u16,
    #[case] length: usize,
    #[case] start: usize,
) {
    let expected = payload(length);
    let mut id = TinyId::new(width, &expected).unwrap().build();
    id[start + length..].fill(0xb5);
    let kind = layout(width).parse(&id).unwrap().kind();
    let FractalHeapIdKind::Tiny { bytes } = kind else {
        panic!("expected a tiny ID, got {kind:?}");
    };
    assert_eq!(bytes, expected);
    assert!(ptr::eq(bytes, &id[start..start + length]));
}

#[rstest]
#[case::normal(2, &[0x2f], 17)]
#[case::extended(19, &[0x20, 17], 20)]
#[case::extended_high_bits(19, &[0x21, 0], 259)]
fn tiny_id_decoding_rejects_a_payload_that_extends_past_the_id(
    #[case] width: u16,
    #[case] prefix: &[u8],
    #[case] expected: usize,
) {
    let mut id = TinyId::new(width, b"x").unwrap().build();
    id[..prefix.len()].copy_from_slice(prefix);
    assert_eq!(
        layout(width).parse(&id).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available: usize::from(width)
        },
    );
}

#[rstest]
#[case::missing_byte_with_invalid_type(&[0x30], 18)]
#[case::extra_byte_with_truncated_payload(&[0x21, 0], 20)]
fn rejects_wrong_id_length_before_invalid_contents(#[case] prefix: &[u8], #[case] actual: usize) {
    let width = 19;
    let mut id = TinyId::new(width, b"x").unwrap().build();
    id[..prefix.len()].copy_from_slice(prefix);
    id.resize(actual, 0);
    assert_eq!(
        layout(width).parse(&id).unwrap_err(),
        FormatError::InvalidFractalHeapIdLength {
            expected: usize::from(width),
            actual
        },
    );
}

fn payload(length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| {
            let [low, high, ..] = index.to_le_bytes();
            low.wrapping_mul(17).wrapping_add(high).wrapping_add(0x35)
        })
        .collect()
}

fn layout(width: u16) -> FractalHeapIdLayout {
    let bytes = Header::new(512).heap_id_length(width).build(Widths::EIGHT);
    let header =
        FractalHeapHeader::parse_bytes(&bytes, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
}

const GUARD: u8 = 0xa5;
