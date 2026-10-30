//! Numeric roundtrip properties for generated shapes and values.
//!
//! Tests compare each supported numeric type in contiguous storage, using float bit patterns.
//! Chunked tests cover deflate-compressed `i32` values. Proptest records regression seeds in
//! `tests/proptest-regressions/roundtrip_properties.txt`.

use hdf5_pure::DeflateLevel;
use hdf5_pure::{DType, File, FileBuilder};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

// ---------------------------------------------------------------------------
// Roundtrip identity
// ---------------------------------------------------------------------------

/// Strategy: a shape of 1..=3 dimensions, each extent 1..=6, paired with
/// exactly `product(shape)` elements drawn from `$elem`.
macro_rules! shaped {
    ($elem:expr) => {
        prop::collection::vec(1u64..=6, 1..=3).prop_flat_map(|shape| {
            let n = shape.iter().product::<u64>() as usize;
            (Just(shape), prop::collection::vec($elem, n..=n))
        })
    };
}

/// Generate a roundtrip property test for one numeric type. `$eq` compares the
/// readback against the input (plain `==` for integers, bit-exact for floats).
macro_rules! roundtrip_test {
    ($name:ident, $ty:ty, $with:ident, $read:ident, $dtype:expr, $eq:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig {
                failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/proptest-regressions/roundtrip_properties.txt"
                )))),
                ..ProptestConfig::default()
            })]

            #[test]
            fn $name((shape, data) in shaped!(any::<$ty>())) {
                let mut b = FileBuilder::new();
                b.create_dataset("d").$with(&data).with_shape(&shape);
                let bytes = b.finish().expect("write");
                let file = File::from_bytes(bytes).expect("parse");
                let ds = file.dataset("d").expect("dataset");

                prop_assert_eq!(ds.shape().expect("shape"), shape);
                prop_assert_eq!(ds.dtype().expect("dtype"), $dtype);
                let got = ds.$read().expect("read");
                prop_assert!($eq(&got, &data), "roundtrip mismatch");
            }
        }
    };
}

fn int_eq<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    a == b
}
fn f32_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn f64_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

roundtrip_test!(
    roundtrip_f64,
    f64,
    with_f64_data,
    read_f64,
    DType::F64,
    f64_eq
);
roundtrip_test!(
    roundtrip_f32,
    f32,
    with_f32_data,
    read_f32,
    DType::F32,
    f32_eq
);
roundtrip_test!(roundtrip_i8, i8, with_i8_data, read_i8, DType::I8, int_eq);
roundtrip_test!(
    roundtrip_i16,
    i16,
    with_i16_data,
    read_i16,
    DType::I16,
    int_eq
);
roundtrip_test!(
    roundtrip_i32,
    i32,
    with_i32_data,
    read_i32,
    DType::I32,
    int_eq
);
roundtrip_test!(
    roundtrip_i64,
    i64,
    with_i64_data,
    read_i64,
    DType::I64,
    int_eq
);
roundtrip_test!(roundtrip_u8, u8, with_u8_data, read_u8, DType::U8, int_eq);
roundtrip_test!(
    roundtrip_u16,
    u16,
    with_u16_data,
    read_u16,
    DType::U16,
    int_eq
);
roundtrip_test!(
    roundtrip_u32,
    u32,
    with_u32_data,
    read_u32,
    DType::U32,
    int_eq
);
roundtrip_test!(
    roundtrip_u64,
    u64,
    with_u64_data,
    read_u64,
    DType::U64,
    int_eq
);

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/proptest-regressions/roundtrip_properties.txt"
        )))),
        ..ProptestConfig::default()
    })]

    /// Chunked + deflate layout roundtrip (lossless integer path). Exercises
    /// the B-tree v1 chunk index and the filter pipeline, not just contiguous
    /// storage. Chunk dims are clamped to the shape so the request is valid.
    #[test]
    fn roundtrip_i32_chunked_deflate(
        (shape, chunk, data) in prop::collection::vec(1u64..=6, 1..=3).prop_flat_map(|shape| {
            let k = shape.len();
            let n = shape.iter().product::<u64>() as usize;
            (Just(shape), prop::collection::vec(1u64..=6, k..=k), prop::collection::vec(any::<i32>(), n..=n))
        }).prop_map(|(shape, seeds, data)| {
            let chunk = shape.iter().zip(&seeds).map(|(&d, &s)| s.min(d)).collect::<Vec<u64>>();
            (shape, chunk, data)
        })
    ) {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&shape)
            .with_chunks(&chunk)
            .with_deflate(DeflateLevel::Six);
        let bytes = b.finish().expect("write");
        let file = File::from_bytes(bytes).expect("parse");
        let ds = file.dataset("d").expect("dataset");

        prop_assert_eq!(ds.shape().expect("shape"), shape);
        prop_assert_eq!(ds.read_i32().expect("read"), data);
    }
}
