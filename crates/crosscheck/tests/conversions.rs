#![cfg(feature = "__hdf5-1.10")]

use hdf5_pure::FileBuilder;
use rstest::rstest;
use tempfile::tempdir;

#[rstest]
#[case::zero(0, 0.0)]
#[case::exact_max_mantissa(16_777_216, 16_777_216.0)]
#[case::loses_precision_rounds_down(16_777_217, 16_777_216.0)]
fn i32_to_f32_precision_loss(#[case] input: i32, #[case] expected: f32) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<i32>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<f32>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<f32>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_f32().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_f32()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::zero(0, 0)]
#[case::positive(127, 127)]
#[case::clamps_to_max(128, 127)]
#[case::clamps_to_max_far(255, 127)]
#[case::clamps_to_min(-129, -128)]
fn i32_to_i8_truncation(#[case] input: i32, #[case] expected: i8) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<i32>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<i8>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<i8>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_i8().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_i8()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::zero(0, 0)]
#[case::in_bounds_positive(100, 100)]
#[case::clamps_negative_one_to_zero(-1, 0)]
#[case::clamps_large_negative_to_zero(-100, 0)]
fn i32_to_u32_sign_loss(#[case] input: i32, #[case] expected: u32) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<i32>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<u32>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<u32>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_u32().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_u32()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::min(i8::MIN, i8::MIN as i64)]
#[case::negative_one(-1, -1)]
#[case::zero(0, 0)]
#[case::max(i8::MAX, i8::MAX as i64)]
fn i8_to_i64_widening(#[case] input: i8, #[case] expected: i64) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<i8>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i8_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<i64>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<i64>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_i64().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_i64()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::zero(0, 0)]
#[case::in_bounds(32767, 32767)]
#[case::clamps_max(32768, 32767)]
#[case::clamps_max_far(65535, 32767)]
fn u16_to_i16_wrap_prevention(#[case] input: u16, #[case] expected: i16) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<u16>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u16_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<i16>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<i16>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_i16().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_i16()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::zero(0.0, 0)]
#[case::in_bounds_positive(42.0, 42)]
#[case::in_bounds_negative(-42.0, -42)]
#[case::clamps_max(5_000_000_000.0, i32::MAX)]
#[case::clamps_min(-5_000_000_000.0, i32::MIN)]
fn f64_to_i32_clamping(#[case] input: f64, #[case] expected: i32) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<f64>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_i32().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_i32()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::zero(0, 0)]
#[case::in_bounds(255, 255)]
#[case::clamps_max(256, 255)]
#[case::clamps_max_far(1_000_000, 255)]
#[case::clamps_negative(-1, 0)]
#[case::clamps_negative_far(-1_000_000, 0)]
fn i64_to_u8_extreme_truncation(#[case] input: i64, #[case] expected: u8) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<i64>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i64_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<u8>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<u8>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_u8().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_u8()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::zero(0.0, 0.0)]
#[case::one(1.0, 1.0)]
#[case::fraction(1.5, 1.5)]
#[case::negative_fraction(-1.5, -1.5)]
fn f32_to_f64_widening(#[case] input: f32, #[case] expected: f64) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<f32>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f32_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<f64>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<f64>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_f64().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_f64()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
#[case::precision_loss_pi(3.141592653589793, 3.1415927)]
#[case::overflow_to_infinity(f64::MAX, f32::INFINITY)]
#[case::overflow_to_neg_infinity(f64::MIN, f32::NEG_INFINITY)]
#[case::subnormal_underflow_to_zero(f64::MIN_POSITIVE, 0.0f32)]
fn f64_to_f32_narrowing(#[case] input: f64, #[case] expected: f32) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<f64>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<f32>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<f32>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_f32().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_f32()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
// 2^53 is the maximum exact integer in f64
#[case::exact_max_mantissa(9_007_199_254_740_992, 9_007_199_254_740_992.0)]
// 2^53 + 1 cannot be represented, rounds down to 2^53
#[case::loses_precision_rounds_down(9_007_199_254_740_993, 9_007_199_254_740_992.0)]
// 2^53 + 3 cannot be represented, rounds up to 2^53 + 4
#[case::loses_precision_rounds_up(9_007_199_254_740_995, 9_007_199_254_740_996.0)]
// u64::MAX loses significant precision
#[case::u64_max(u64::MAX, 1.8446744073709552e19)]
fn u64_to_f64_precision_loss(#[case] input: u64, #[case] expected: f64) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<u64>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u64_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<f64>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<f64>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_f64().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_f64()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}

#[rstest]
// Float representation of 2^31 pushes slightly past i32::MAX
#[case::just_above_max(2_147_483_904.0, i32::MAX)]
#[case::just_below_min(-2_147_483_904.0, i32::MIN)]
#[case::infinity_clamps_to_max(f32::INFINITY, i32::MAX)]
#[case::neg_infinity_clamps_to_min(f32::NEG_INFINITY, i32::MIN)]
fn f32_to_i32_edge_cases(#[case] input: f32, #[case] expected: i32) {
    let dir = tempdir().unwrap();
    let path_ref = dir.path().join("ref.h5");
    let path_pure = dir.path().join("pure.h5");

    {
        let ref_file = hdf5::File::create(&path_ref).unwrap();
        let ref_ds = ref_file
            .new_dataset::<f32>()
            .shape([1])
            .create("data")
            .unwrap();
        ref_ds.write_raw(&[input]).unwrap();
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f32_data(&[input])
        .with_shape(&[1]);
    builder.write(&path_pure).unwrap();

    {
        let baseline_file = hdf5::File::open(&path_ref).unwrap();
        let baseline_val = baseline_file
            .dataset("data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap()[0];
        assert_eq!(baseline_val, expected, "libhdf5 baseline mismatch");
    }

    {
        let pure_write_file = hdf5::File::open(&path_pure).unwrap();
        let pure_write_val = pure_write_file
            .dataset("data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap()[0];
        assert_eq!(pure_write_val, expected, "libhdf5 misread pure file");
    }

    let pure_read_file = hdf5_pure::File::open(&path_ref).unwrap();
    let pure_read_val = pure_read_file.dataset("data").unwrap().read_i32().unwrap()[0];
    assert_eq!(pure_read_val, expected, "pure misread libhdf5 file");

    let pure_roundtrip_file = hdf5_pure::File::open(&path_pure).unwrap();
    let pure_roundtrip_val = pure_roundtrip_file
        .dataset("data")
        .unwrap()
        .read_i32()
        .unwrap()[0];
    assert_eq!(pure_roundtrip_val, expected, "pure roundtrip mismatch");
}
