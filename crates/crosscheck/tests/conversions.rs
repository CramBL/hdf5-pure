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
#[ignore = "TODO: Implement HDF5 soft conversion clamping logic for narrowing integers"]
#[case::clamps_to_max(128, 127)]
#[ignore = "TODO: Implement HDF5 soft conversion clamping logic for narrowing integers"]
#[case::clamps_to_max_far(255, 127)]
#[ignore = "TODO: Implement HDF5 soft conversion clamping logic for narrowing integers"]
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
#[ignore = "TODO: Implement HDF5 soft conversion clamping logic"]
#[case::clamps_negative_one_to_zero(-1, 0)]
#[ignore = "TODO: Implement HDF5 soft conversion clamping logic"]
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
#[ignore = "TODO: Implement HDF5 soft conversion clamping logic"]
#[case::clamps_max(32768, 32767)]
#[ignore = "TODO: Implement HDF5 soft conversion clamping logic"]
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
