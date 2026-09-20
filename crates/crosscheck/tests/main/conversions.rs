#![cfg(feature = "__hdf5-1.10")]

use hdf5_pure::FileBuilder;
use rstest::rstest;

macro_rules! assert_hdf5_conversion {
    (
        from: $input:expr => $in_type:ty [$write_pure:ident],
        to:   $expected:expr => $out_type:ty [$read_pure:ident]
    ) => {{
        let dir = tempfile::tempdir().unwrap();
        let path_ref = dir.path().join("ref.h5");
        let path_pure = dir.path().join("pure.h5");

        {
            let ref_file = hdf5::File::create(&path_ref).unwrap();
            let ref_ds = ref_file
                .new_dataset::<$in_type>()
                .shape([1])
                .create("data")
                .unwrap();
            ref_ds.write_raw(&[$input]).unwrap();
        }

        let mut builder = FileBuilder::new();
        builder
            .create_dataset("data")
            .$write_pure(&[$input])
            .with_shape(&[1]);
        builder.write(&path_pure).unwrap();

        {
            let baseline_file = hdf5::File::open(&path_ref).unwrap();
            let baseline_val = baseline_file
                .dataset("data")
                .unwrap()
                .read_raw::<$out_type>()
                .unwrap()[0];
            assert_eq!(baseline_val, $expected, "libhdf5 baseline mismatch");
        }

        {
            let pure_write_file = hdf5::File::open(&path_pure).unwrap();
            let pure_write_val = pure_write_file
                .dataset("data")
                .unwrap()
                .read_raw::<$out_type>()
                .unwrap()[0];
            assert_eq!(pure_write_val, $expected, "libhdf5 misread pure file");
        }

        let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
        let pure_read_val = pure_read_file
            .dataset("data")
            .unwrap()
            .$read_pure()
            .unwrap()[0];
        assert_eq!(pure_read_val, $expected, "pure misread libhdf5 file");

        let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
        let pure_roundtrip_val = pure_roundtrip_file
            .dataset("data")
            .unwrap()
            .$read_pure()
            .unwrap()[0];
        assert_eq!(pure_roundtrip_val, $expected, "pure roundtrip mismatch");
    }};
}

#[rstest]
#[case::zero(0, 0.0)]
#[case::exact_max_mantissa(16_777_216, 16_777_216.0)]
#[case::loses_precision_rounds_down(16_777_217, 16_777_216.0)]
fn i32_to_f32_precision_loss(#[case] input: i32, #[case] expected: f32) {
    assert_hdf5_conversion! {
        from: input    => i32 [with_i32_data],
        to:   expected => f32 [read_f32]
    }
}

#[rstest]
#[case::zero(0, 0)]
#[case::positive(127, 127)]
#[case::clamps_to_max(128, 127)]
#[case::clamps_to_max_far(255, 127)]
#[case::clamps_to_min(-129, -128)]
fn i32_to_i8_truncation(#[case] input: i32, #[case] expected: i8) {
    assert_hdf5_conversion! {
        from: input    => i32 [with_i32_data],
        to:   expected => i8  [read_i8]
    }
}

#[rstest]
#[case::zero(0, 0)]
#[case::in_bounds_positive(100, 100)]
#[case::clamps_negative_one_to_zero(-1, 0)]
#[case::clamps_large_negative_to_zero(-100, 0)]
fn i32_to_u32_sign_loss(#[case] input: i32, #[case] expected: u32) {
    assert_hdf5_conversion! {
        from: input    => i32 [with_i32_data],
        to:   expected => u32 [read_u32]
    }
}

#[rstest]
#[case::min(i8::MIN, i8::MIN as i64)]
#[case::negative_one(-1, -1)]
#[case::zero(0, 0)]
#[case::max(i8::MAX, i8::MAX as i64)]
fn i8_to_i64_widening(#[case] input: i8, #[case] expected: i64) {
    assert_hdf5_conversion! {
        from: input    => i8  [with_i8_data],
        to:   expected => i64 [read_i64]
    }
}

#[rstest]
#[case::zero(0.0, 0)]
#[case::negative_zero(-0.0, 0)]
#[case::in_bounds_positive(42.0, 42)]
#[case::in_bounds_negative(-42.0, -42)]
#[case::truncates_positive(42.9, 42)]
#[case::truncates_negative(-42.9, -42)]
#[case::clamps_max(5_000_000_000.0, i32::MAX)]
#[case::clamps_min(-5_000_000_000.0, i32::MIN)]
#[case::infinity_clamps_to_max(f64::INFINITY, i32::MAX)]
#[case::neg_infinity_clamps_to_min(f64::NEG_INFINITY, i32::MIN)]
fn f64_to_i32_clamping(#[case] input: f64, #[case] expected: i32) {
    assert_hdf5_conversion! {
        from: input    => f64 [with_f64_data],
        to:   expected => i32 [read_i32]
    }
}

#[rstest]
#[case::zero(0, 0)]
#[case::in_bounds(255, 255)]
#[case::clamps_max(256, 255)]
#[case::clamps_max_far(1_000_000, 255)]
#[case::clamps_negative(-1, 0)]
#[case::clamps_negative_far(-1_000_000, 0)]
fn i64_to_u8_extreme_truncation(#[case] input: i64, #[case] expected: u8) {
    assert_hdf5_conversion! {
        from: input    => i64 [with_i64_data],
        to:   expected => u8  [read_u8]
    }
}

#[rstest]
#[case::precision_loss_pi(3.141592653589793, 3.1415927)]
#[case::overflow_to_infinity(f64::MAX, f32::INFINITY)]
#[case::overflow_to_neg_infinity(f64::MIN, f32::NEG_INFINITY)]
#[case::min_positive_underflow_to_zero(f64::MIN_POSITIVE, 0.0f32)]
#[case::subnormal_underflow_to_zero(f64::from_bits(1), 0.0f32)]
fn f64_to_f32_narrowing(#[case] input: f64, #[case] expected: f32) {
    assert_hdf5_conversion! {
        from: input    => f64 [with_f64_data],
        to:   expected => f32 [read_f32]
    }
}

#[rstest]
#[case::exact_max_mantissa(9_007_199_254_740_992, 9_007_199_254_740_992.0)]
#[case::loses_precision_rounds_down(9_007_199_254_740_993, 9_007_199_254_740_992.0)]
#[case::loses_precision_rounds_up(9_007_199_254_740_995, 9_007_199_254_740_996.0)]
#[case::u64_max(u64::MAX, 1.8446744073709552e19)]
fn u64_to_f64_precision_loss(#[case] input: u64, #[case] expected: f64) {
    assert_hdf5_conversion! {
        from: input    => u64 [with_u64_data],
        to:   expected => f64 [read_f64]
    }
}
