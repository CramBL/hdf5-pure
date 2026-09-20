//! End-to-end write/read roundtrip through the full HDF5 stack with ZFP
//! compression enabled. This is the Step 4 integration test — the codec
//! itself is covered by `src/zfp_crosscheck.rs`; this harness checks that the
//! chunked writer, filter pipeline, and chunked reader all thread the new
//! ChunkContext through correctly.

#![cfg(feature = "zfp")]

use hdf5_pure::{Error, File, FileBuilder, FormatError};

fn max_abs_err(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0f64, f64::max)
}

#[test]
fn zfp_f32_roundtrip_rate16() {
    let vals: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_chunks(&[32])
        .with_zfp(16.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    // Rate 16 on f32 should give very small error for this simple ramp.
    assert!(
        max_abs_err(&vals, &back) < 0.01,
        "max_err {} > 0.01",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_f64_roundtrip_rate32() {
    let vals: Vec<f64> = (0..32).map(|i| i as f64 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f64_data(&vals)
        .with_chunks(&[32])
        .with_zfp(32.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f64().unwrap();
    assert_eq!(back.len(), vals.len());
    let max_err = vals
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(max_err < 1e-6, "max_err {max_err} > 1e-6");
}

#[test]
fn zfp_i32_roundtrip_rate32() {
    let vals: Vec<i32> = (0..16).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_i32_data(&vals)
        .with_chunks(&[16])
        .with_zfp(32.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_i32().unwrap();
    assert_eq!(back.len(), vals.len());
    // ZFP integer mode at max rate is not bit-identical but should be close.
    let max_err = vals
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap();
    assert!(max_err <= 2, "max_err {max_err} > 2");
}

#[test]
fn zfp_i64_roundtrip_rate64() {
    let vals: Vec<i64> = (0..16).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_i64_data(&vals)
        .with_chunks(&[16])
        .with_zfp(64.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_i64().unwrap();
    assert_eq!(back.len(), vals.len());
    let max_err = vals
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap();
    assert!(max_err <= 2, "max_err {max_err} > 2");
}

#[test]
fn zfp_f32_partial_chunk_1d() {
    // 13 values exercises the partial last block: 3 full ZFP blocks of 4 +
    // 1 partial block of 1 valid element + 3 pad. Before the pad-handling
    // fix this path produced off-by-one output lengths and broken decodes.
    let vals: Vec<f32> = (0..13).map(|i| i as f32 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_chunks(&[13])
        .with_zfp(16.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    assert!(
        max_abs_err(&vals, &back) < 0.05,
        "max_err {}",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_f64_partial_chunk_2d() {
    // 5×7 grid — both axes non-multiple of 4. Covers 4 ZFP blocks: one
    // 4×4 full, one 4×3 partial-x, one 1×4 partial-y, one 1×3 partial-xy.
    let dims = [5usize, 7];
    let n = dims.iter().product::<usize>();
    let vals: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f64_data(&vals)
        .with_shape(&[dims[0] as u64, dims[1] as u64])
        .with_chunks(&[dims[0] as u64, dims[1] as u64])
        .with_zfp(32.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f64().unwrap();
    assert_eq!(back.len(), vals.len());
    let max_err = vals
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(max_err < 0.01, "max_err {max_err} > 0.01");
}

#[test]
fn zfp_f32_roundtrip_2d() {
    // Smooth 8×12 ramp (values ~0 to ~95/4).
    let vals: Vec<f32> = (0..96).map(|i| i as f32 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_shape(&[8, 12])
        .with_chunks(&[8, 12])
        .with_zfp(16.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    assert!(
        max_abs_err(&vals, &back) < 0.1,
        "max_err {} > 0.1",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_f64_roundtrip_3d() {
    let dims = [4usize, 4, 4];
    let n = dims.iter().product();
    let vals: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f64_data(&vals)
        .with_shape(&[dims[0] as u64, dims[1] as u64, dims[2] as u64])
        .with_chunks(&[dims[0] as u64, dims[1] as u64, dims[2] as u64])
        .with_zfp(32.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f64().unwrap();
    assert_eq!(back.len(), vals.len());
    let max_err = vals
        .iter()
        .zip(back.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(max_err < 0.01, "max_err {max_err} > 0.01");
}

#[test]
fn zfp_f32_roundtrip_4d() {
    let dims = [4usize, 4, 4, 4];
    let n = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.125).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_shape(&dims.map(|d| d as u64))
        .with_chunks(&dims.map(|d| d as u64))
        .with_zfp(16.0);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    assert!(
        max_abs_err(&vals, &back) < 0.1,
        "max_err {} > 0.1",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_on_unsupported_scalar_errors_not_panics() {
    // u8 is not one of ZFP's supported scalar types. Finalize must surface
    // a FormatError::UnsupportedZfp, not panic.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_u8_data(&[1u8, 2, 3, 4])
        .with_chunks(&[4])
        .with_zfp(16.0);
    let err = builder
        .finish()
        .expect_err("ZFP on u8 should error, not succeed");
    assert!(
        matches!(err, Error::Format(FormatError::UnsupportedZfp(_))),
        "unexpected error variant: {err:?}",
    );
}

#[test]
fn zfp_dtype_is_sole_source_of_truth() {
    // Regression: the data setter must no longer carry a parallel
    // `zfp_element_type`. The codec reads the scalar type from the dataset's
    // datatype, so writing raw bytes through `with_u8_data` and then
    // overriding the dtype with `with_dtype(make_f32_type())` must produce a
    // valid ZFP-compressed f32 dataset. Before the refactor this errored
    // because `with_u8_data` never populated `chunk_options.zfp_element_type`.
    use hdf5_pure::make_f32_type;
    let vals: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_u8_data(&raw)
        .with_dtype(make_f32_type())
        .with_shape(&[vals.len() as u64])
        .with_chunks(&[vals.len() as u64])
        .with_zfp(16.0);
    let bytes = builder.finish().expect("dtype-driven ZFP should succeed");

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("v").unwrap();
    let back = ds.read_f32().unwrap();
    assert_eq!(back.len(), vals.len());
    assert!(
        max_abs_err(&vals, &back) < 0.1,
        "max_err {}",
        max_abs_err(&vals, &back)
    );
}

#[test]
fn zfp_with_5d_chunks_errors_not_panics() {
    // 5D chunks are beyond the ZFP rank limit. Finalize must surface a
    // FormatError::UnsupportedZfp, not panic inside the cd_values builder.
    let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("v")
        .with_f32_data(&vals)
        .with_shape(&[2, 2, 2, 2, 2])
        .with_chunks(&[2, 2, 2, 2, 2])
        .with_zfp(16.0);
    let err = builder
        .finish()
        .expect_err("ZFP with 5D chunks should error, not succeed");
    assert!(
        matches!(err, Error::Format(FormatError::UnsupportedZfp(_))),
        "unexpected error variant: {err:?}",
    );
}
