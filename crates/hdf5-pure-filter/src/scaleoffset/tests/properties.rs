//! Scale-offset roundtrip properties and encoding boundary cases.
//!
//! Proptest records regression seeds in `proptest-regressions/scaleoffset.txt` under the
//! filter crate.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use rstest::rstest;

use super as tests;
use super::super::{self as scaleoffset, ScaleOffsetByteOrder};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/proptest-regressions/scaleoffset.txt"
        )))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn packed_offsets_preserve_lengths_and_values(
        (minbits, offsets) in (1u32..=64).prop_flat_map(|minbits| {
            let max = u64::MAX >> (64 - minbits);
            (Just(minbits), prop::collection::vec(0..=max, 1..=257))
        }),
    ) {
        let packed = scaleoffset::pack_offsets(&offsets, minbits, offsets.len()).unwrap();
        prop_assert_eq!(packed.len(), offsets.len() * usize::try_from(minbits).unwrap() / 8 + 1);
        prop_assert_eq!(
            scaleoffset::unpack_bits(&packed, offsets.len(), minbits).unwrap(),
            offsets
        );
    }

    #[test]
    fn u8_values_roundtrip(values in prop::collection::vec(any::<u8>(), 1..=199)) {
        IntegerValues::U8(values).assert_roundtrip();
    }

    #[test]
    fn u16_values_roundtrip_in_both_byte_orders(
        values in prop::collection::vec(any::<u16>(), 1..=199),
        order in byte_order(),
    ) {
        IntegerValues::U16 { values, order }.assert_roundtrip();
    }

    #[test]
    fn i32_narrow_and_full_range_values_roundtrip(
        values in prop_oneof![
            prop::collection::vec(-512i32..512, 1..=199),
            prop::collection::vec(any::<i32>(), 1..=199),
        ],
    ) {
        IntegerValues::I32(values).assert_roundtrip();
    }

    #[test]
    fn i64_offsets_narrow_spans_and_full_range_values_roundtrip(
        values in prop_oneof![
            (i64::MIN / 2..=i64::MAX / 2, 1u32..=48).prop_flat_map(|(offset, bits)| {
                prop::collection::vec(0i64..(1i64 << bits), 1..=199)
                    .prop_map(move |values| values.into_iter().map(|value| value + offset).collect())
            }),
            prop::collection::vec(any::<i64>(), 1..=199),
        ],
    ) {
        IntegerValues::I64(values).assert_roundtrip();
    }

    #[test]
    fn f64_decimal_scaling_preserves_length_and_tolerance(
        values in prop::collection::vec(any::<i32>().prop_map(|value| f64::from(value) * 1e-9), 1..=99),
        decimals in 1i32..=5,
    ) {
        let raw: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let filter = tests::float_filter(8, decimals, scaleoffset::ORDER_LE, u32::try_from(values.len()).unwrap());
        let encoded = tests::compress(&raw, &filter).unwrap();
        let decoded = tests::decompress(&encoded, &filter, None).unwrap();
        prop_assert_eq!(decoded.len(), raw.len());
        let (chunks, remainder) = decoded.as_chunks::<8>();
        prop_assert!(remainder.is_empty());
        let actual: Vec<f64> = chunks.iter().map(|chunk| f64::from_le_bytes(*chunk)).collect();
        prop_assert_eq!(actual.len(), values.len());
        // Half a decimal step, with a margin for floating-point arithmetic.
        let tolerance = 0.501 * 10f64.powi(-decimals);
        for (actual, expected) in actual.iter().zip(&values) {
            prop_assert!((actual - expected).abs() <= tolerance, "got {actual}, want {expected}, tolerance {tolerance}");
        }
    }
}

#[rstest]
#[case::one_bit(1, vec![0, 1], vec![0x40])]
#[case::partial_byte(3, vec![0, 1, 7, 2], vec![0x07, 0xA0])]
#[case::byte_boundary(8, vec![0x12, 0xAB], vec![0x12, 0xAB, 0])]
#[case::full_width(64, vec![u64::MAX], vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0])]
#[case::maximum_length(1, vec![1; 257], [vec![0xFF; 32], vec![0x80]].concat())]
fn packed_offset_boundaries(
    #[case] minbits: u32,
    #[case] offsets: Vec<u64>,
    #[case] expected: Vec<u8>,
) {
    let packed = scaleoffset::pack_offsets(&offsets, minbits, offsets.len()).unwrap();
    assert_eq!(packed, expected);
    assert_eq!(
        scaleoffset::unpack_bits(&packed, offsets.len(), minbits).unwrap(),
        offsets
    );
}

#[rstest]
#[case::u8_single(IntegerValues::U8(vec![u8::MAX]), 0)]
#[case::u8_maximum_length(IntegerValues::U8(vec![u8::MAX; 199]), 0)]
#[case::u8_full_range(IntegerValues::U8(vec![u8::MIN, u8::MAX]), 8)]
#[case::u16_little_endian(IntegerValues::U16 { values: vec![u16::MIN, u16::MAX], order: ScaleOffsetByteOrder::LittleEndian }, 16)]
#[case::u16_big_endian(IntegerValues::U16 { values: vec![u16::MIN, u16::MAX], order: ScaleOffsetByteOrder::BigEndian }, 16)]
#[case::i32_narrow(IntegerValues::I32(vec![-512, 511]), 10)]
#[case::i32_full_range(IntegerValues::I32(vec![i32::MIN, i32::MAX]), 32)]
#[case::i64_offset_narrow(IntegerValues::I64(vec![i64::MIN / 2, i64::MIN / 2 + 1]), 1)]
#[case::i64_offset_wide(IntegerValues::I64(vec![i64::MAX / 2, i64::MAX / 2 + (1i64 << 48) - 1]), 48)]
#[case::i64_full_range(IntegerValues::I64(vec![i64::MIN, i64::MAX]), 64)]
fn integer_roundtrip_boundaries(#[case] values: IntegerValues, #[case] minbits: u32) {
    let (raw, filter) = values.encode();
    let encoded = tests::compress(&raw, &filter).unwrap();
    assert_eq!(
        u32::from_le_bytes(encoded[..4].try_into().unwrap()),
        minbits
    );
    if minbits == filter[scaleoffset::PARM_SIZE] * 8 {
        assert_eq!(&encoded[scaleoffset::HEADER_LEN..], raw);
    }
    assert_eq!(
        tests::decompress(&encoded, &filter, None).unwrap(),
        raw
    );
}

/// Integer payloads with matching signedness, width, and byte order.
#[derive(Debug)]
enum IntegerValues {
    I32(Vec<i32>),
    I64(Vec<i64>),
    U16 {
        values: Vec<u16>,
        order: ScaleOffsetByteOrder,
    },
    U8(Vec<u8>),
}

impl IntegerValues {
    fn assert_roundtrip(&self) {
        let (raw, filter) = self.encode();
        let encoded = tests::compress(&raw, &filter).unwrap();
        assert_eq!(
            tests::decompress(&encoded, &filter, None).unwrap(),
            raw
        );
    }

    /// Returns element bytes and lossless filter parameters for the selected scalar type.
    fn encode(&self) -> (Vec<u8>, Vec<u32>) {
        let (raw, size, signed, order, len) = match self {
            Self::I32(values) => (
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                4,
                true,
                scaleoffset::ORDER_LE,
                values.len(),
            ),
            Self::I64(values) => (
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                8,
                true,
                scaleoffset::ORDER_LE,
                values.len(),
            ),
            Self::U8(values) => (
                values.clone(),
                1,
                false,
                scaleoffset::ORDER_LE,
                values.len(),
            ),
            Self::U16 { values, order } => (
                values
                    .iter()
                    .flat_map(|value| match order {
                        ScaleOffsetByteOrder::BigEndian => value.to_be_bytes(),
                        ScaleOffsetByteOrder::LittleEndian => value.to_le_bytes(),
                    })
                    .collect(),
                2,
                false,
                order.code(),
                values.len(),
            ),
        };
        (
            raw,
            tests::int_filter(size, signed, order, u32::try_from(len).unwrap()),
        )
    }
}

fn byte_order() -> impl Strategy<Value = ScaleOffsetByteOrder> {
    prop_oneof![
        Just(ScaleOffsetByteOrder::BigEndian),
        Just(ScaleOffsetByteOrder::LittleEndian),
    ]
}
