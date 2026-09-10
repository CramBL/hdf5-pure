// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Cross-validation tests: write with hdf5-pure, read with the official C HDF5 library.
//!
//! These tests verify that files produced by hdf5-pure are valid HDF5 files
//! readable by the reference C implementation.

use hdf5_pure::{
    AttrValue, CompoundTypeBuilder, Datatype, File, FileBuilder, VlenStringReadOptions,
    make_f64_type,
};
use tempfile::tempdir;

/// Read a MATLAB-style variable-length ASCII attribute from an `hdf5::Attribute`.
/// Our serializer emits the same `H5T_VLEN { H5T_STRING { STRSIZE 1 } }` shape
/// that real MATLAB and matio emit, so reading via `VarLenArray<FixedAscii<1>>`
/// matches. Returns the decoded strings in order.
fn read_vl_ascii_attr(attr: &hdf5::Attribute) -> Vec<String> {
    let raw: Vec<hdf5::types::VarLenArray<hdf5::types::FixedAscii<1>>> =
        attr.read_raw().expect("VLEN{string,1} attr");
    raw.iter()
        .map(|vl| vl.iter().flat_map(|c| c.as_str().chars()).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Numeric dataset round-trips
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_f64_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f64.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .with_shape(&[5]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn crosscheck_f32_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f32.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f32_data(&[1.5f32, 2.5, 3.5]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f32>().unwrap();
    assert_eq!(values, vec![1.5f32, 2.5, 3.5]);
}

#[test]
fn crosscheck_i8_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("i8.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i8_data(&[-128, -1, 0, 1, 127]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<i8>().unwrap();
    assert_eq!(values, vec![-128, -1, 0, 1, 127]);
}

#[test]
fn crosscheck_i16_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("i16.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i16_data(&[-32768, 0, 32767]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<i16>().unwrap();
    assert_eq!(values, vec![-32768, 0, 32767]);
}

#[test]
fn crosscheck_i32_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("i32.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&[-100, 0, 100, i32::MAX]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<i32>().unwrap();
    assert_eq!(values, vec![-100, 0, 100, i32::MAX]);
}

#[test]
fn crosscheck_i64_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("i64.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i64_data(&[i64::MIN, 0, i64::MAX]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<i64>().unwrap();
    assert_eq!(values, vec![i64::MIN, 0, i64::MAX]);
}

#[test]
fn crosscheck_u8_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("u8.h5");

    let mut builder = FileBuilder::new();
    builder.create_dataset("data").with_u8_data(&[0, 128, 255]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<u8>().unwrap();
    assert_eq!(values, vec![0, 128, 255]);
}

#[test]
fn crosscheck_u16_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("u16.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u16_data(&[0, 1000, 65535]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<u16>().unwrap();
    assert_eq!(values, vec![0, 1000, 65535]);
}

#[test]
fn crosscheck_u32_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("u32.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u32_data(&[0, 1_000_000, u32::MAX]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<u32>().unwrap();
    assert_eq!(values, vec![0, 1_000_000, u32::MAX]);
}

#[test]
fn crosscheck_u64_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("u64.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u64_data(&[0, 1, u64::MAX]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<u64>().unwrap();
    assert_eq!(values, vec![0, 1, u64::MAX]);
}

// ---------------------------------------------------------------------------
// Multi-dimensional datasets
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_2d_f64_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("2d.h5");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("matrix")
        .with_f64_data(&data)
        .with_shape(&[2, 3]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("matrix").unwrap();
    let shape = ds.shape();
    assert_eq!(shape, vec![2, 3]);
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values, data);
}

// ---------------------------------------------------------------------------
// Multiple datasets
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_multiple_datasets() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi.h5");

    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_f64_data(&[1.0, 2.0, 3.0]);
    builder.create_dataset("y").with_i32_data(&[10, 20, 30]);
    builder.create_dataset("z").with_u8_data(&[0xFF, 0x00]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    let x = file.dataset("x").unwrap().read_raw::<f64>().unwrap();
    assert_eq!(x, vec![1.0, 2.0, 3.0]);

    let y = file.dataset("y").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(y, vec![10, 20, 30]);

    let z = file.dataset("z").unwrap().read_raw::<u8>().unwrap();
    assert_eq!(z, vec![0xFF, 0x00]);
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_group_with_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("group.h5");

    let mut builder = FileBuilder::new();
    let mut grp = builder.create_group("sensors");
    grp.create_dataset("temperature")
        .with_f64_data(&[20.5, 21.0, 22.3]);
    builder.add_group(grp.finish());
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let grp = file.group("sensors").unwrap();
    let member_names = grp.member_names().unwrap();
    assert_eq!(member_names, vec!["temperature"]);

    let ds = file.dataset("sensors/temperature").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values, vec![20.5, 21.0, 22.3]);
}

#[test]
fn crosscheck_nested_groups() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.h5");

    let mut builder = FileBuilder::new();
    let mut outer = builder.create_group("outer");
    outer.create_dataset("outer_data").with_f64_data(&[1.0]);

    let mut inner = outer.create_group("inner");
    inner
        .create_dataset("inner_data")
        .with_f64_data(&[2.0, 3.0]);
    outer.add_group(inner.finish());
    builder.add_group(outer.finish());
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    let outer_data = file
        .dataset("outer/outer_data")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    assert_eq!(outer_data, vec![1.0]);

    let inner_data = file
        .dataset("outer/inner/inner_data")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    assert_eq!(inner_data, vec![2.0, 3.0]);

    // Verify group hierarchy
    let outer_grp = file.group("outer").unwrap();
    let mut names = outer_grp.member_names().unwrap();
    names.sort();
    assert_eq!(names, vec!["inner", "outer_data"]);

    let inner_grp = file.group("outer/inner").unwrap();
    let inner_names = inner_grp.member_names().unwrap();
    assert_eq!(inner_names, vec!["inner_data"]);
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_f64_attribute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("attr_f64.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("scale", AttrValue::F64(0.5));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let attr = file.attr("scale").unwrap();
    let val = attr.read_scalar::<f64>().unwrap();
    assert!((val - 0.5).abs() < 1e-15);
}

#[test]
fn crosscheck_i64_attribute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("attr_i64.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("version", AttrValue::I64(42));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let attr = file.attr("version").unwrap();
    let val = attr.read_scalar::<i64>().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn crosscheck_i32_attribute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("attr_i32.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_int_decode", AttrValue::I32(1));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let attr = file.attr("MATLAB_int_decode").unwrap();
    let val = attr.read_scalar::<i32>().unwrap();
    assert_eq!(val, 1);
}

#[test]
fn crosscheck_u32_attribute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("attr_u32.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_empty", AttrValue::U32(1));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let attr = file.attr("MATLAB_empty").unwrap();
    let val = attr.read_scalar::<u32>().unwrap();
    assert_eq!(val, 1);
}

#[test]
fn crosscheck_string_attribute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("attr_str.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("description", AttrValue::String("test data".into()));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let attr_names = file.attr_names().unwrap();
    assert!(attr_names.contains(&"description".to_string()));
}

#[test]
fn crosscheck_ascii_string_attribute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("attr_ascii.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let attr_names = file.attr_names().unwrap();
    assert!(attr_names.contains(&"MATLAB_class".to_string()));
}

#[test]
fn crosscheck_dataset_attributes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ds_attr.h5");

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0])
        .set_attr("unit", AttrValue::String("m/s".into()))
        .set_attr("scale", AttrValue::F64(0.001));
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let attr_names = ds.attr_names().unwrap();
    assert!(attr_names.contains(&"unit".to_string()));
    assert!(attr_names.contains(&"scale".to_string()));

    let scale = ds.attr("scale").unwrap().read_scalar::<f64>().unwrap();
    assert!((scale - 0.001).abs() < 1e-15);
}

#[test]
fn crosscheck_group_attributes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("grp_attr.h5");

    let mut builder = FileBuilder::new();
    let mut grp = builder.create_group("sensors");
    grp.set_attr("location", AttrValue::String("lab".into()));
    grp.create_dataset("temp").with_f64_data(&[20.0]);
    builder.add_group(grp.finish());
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let grp = file.group("sensors").unwrap();
    let attr_names = grp.attr_names().unwrap();
    assert!(attr_names.contains(&"location".to_string()));
}

// ---------------------------------------------------------------------------
// Userblock
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_userblock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("userblock.h5");

    let mut builder = FileBuilder::new();
    builder.with_userblock(512);
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    let mut bytes = builder.finish().unwrap();

    // Write MATLAB-style header into userblock
    bytes[..6].copy_from_slice(b"MATLAB");
    std::fs::write(&path, &bytes).unwrap();

    // C HDF5 should still read the file correctly
    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// Chunked + compressed datasets
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_chunked_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("chunked.h5");

    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_chunks(&[25]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values.len(), 100);
    for (i, &v) in values.iter().enumerate() {
        assert!((v - i as f64 * 0.1).abs() < 1e-10);
    }
}

#[test]
fn crosscheck_paged_fixed_array() {
    // > 1024 chunks pushes the Fixed Array index into its paged data block
    // layout. A non-compliant (flat) data block makes the C library reject the
    // file or misread it; this asserts the reference reader accepts ours.
    let dir = tempdir().unwrap();
    let path = dir.path().join("paged_fa.h5");

    let n = 2500usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[1]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values.len(), n);
    for (i, &v) in values.iter().enumerate() {
        assert!((v - i as f64 * 0.25).abs() < 1e-10);
    }
}

#[test]
fn crosscheck_paged_fixed_array_deflate() {
    // Paged data block on the filtered element path, read back by the reference
    // C library (which validates each page checksum).
    let dir = tempdir().unwrap();
    let path = dir.path().join("paged_fa_deflate.h5");

    let n = 1500usize;
    let data: Vec<f64> = (0..n).map(|i| (i % 11) as f64).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[1])
        .with_deflate(4);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values.len(), n);
    for (i, &v) in values.iter().enumerate() {
        assert!((v - (i % 11) as f64).abs() < 1e-10);
    }
}

#[test]
fn crosscheck_read_paged_fixed_array_from_c_lib() {
    // Reverse direction: the reference C library writes a paged Fixed Array
    // (the layout that previously triggered "paged Fixed Array data blocks not
    // yet supported"), and hdf5-pure must read it back. `libver_v110` forces the
    // v110+ Fixed Array index; > 1024 fixed-shape chunks make it paged.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_paged_fa.h5");

    let n = 2500usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .chunk([1])
            .shape([n])
            .create("data")
            .unwrap();
        ds.write_raw(data.as_slice()).unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let file = hdf5_pure::File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), n);
    for (i, &v) in values.iter().enumerate() {
        assert!((v - i as f64 * 0.25).abs() < 1e-10);
    }
}

#[test]
fn crosscheck_read_paged_fixed_array_with_uninitialized_page() {
    // The reference C library writes a paged Fixed Array but, with incremental
    // allocation, only the first page's chunks are written. The second page is
    // never allocated, so its page-init bitmap bit stays cleared. hdf5-pure must
    // skip that page (no chunk records) and fill the unwritten tail with the
    // dataset's fill value, exercising the bitmap-skip path against real bytes.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_paged_fa_sparse.h5");

    let n = 2000usize; // 2 pages of 1024; page 1 (chunks 1024..) left untouched
    let written = 1000usize; // entirely within page 0
    let head: Vec<f64> = (0..written).map(|i| (i + 1) as f64).collect();
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .chunk([1])
            .shape([n])
            .create("data")
            .unwrap();
        // Write only the first `written` chunks; the rest stay unallocated.
        ds.write_slice(head.as_slice(), 0..written).unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let file = hdf5_pure::File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_f64().unwrap();
    assert_eq!(values.len(), n);
    for (i, &v) in values.iter().enumerate() {
        let expected = if i < written { (i + 1) as f64 } else { 0.0 };
        assert!(
            (v - expected).abs() < 1e-10,
            "index {i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn crosscheck_deflate_compressed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("deflate.h5");

    let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_chunks(&[50])
        .with_deflate(6);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values.len(), 200);
    for (i, &v) in values.iter().enumerate() {
        assert!((v - i as f64).abs() < 1e-10);
    }
}

#[test]
fn crosscheck_shuffle_deflate() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shuffle_deflate.h5");

    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_chunks(&[25])
        .with_shuffle()
        .with_deflate(4);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values.len(), 100);
    for (i, &v) in values.iter().enumerate() {
        assert!((v - i as f64 * 0.01).abs() < 1e-10);
    }
}

#[test]
fn crosscheck_fletcher32_all_ones_written_by_pure_read_by_c() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fletcher32_ones.h5");

    let data = vec![-1i32; 128];
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&data)
        .with_chunks(&[64])
        .with_fletcher32();
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<i32>().unwrap();
    assert_eq!(values, data);
}

#[test]
fn crosscheck_fletcher32_all_ones_written_by_c_read_by_pure() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_fletcher32_ones.h5");

    let data = vec![-1i32; 128];
    {
        let file = hdf5::File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk([64])
            .shape([128])
            .fletcher32()
            .create("data")
            .unwrap();
        ds.write_raw(data.as_slice()).unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), data);
}

// ---------------------------------------------------------------------------
// Compound types (complex numbers)
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_compound_f64_pairs() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("compound.h5");

    let ct = CompoundTypeBuilder::new()
        .f64_field("real")
        .f64_field("imag")
        .build()
        .unwrap();
    let pairs = [(1.0f64, 2.0f64), (3.0, 4.0), (5.0, 6.0)];
    let mut raw = Vec::new();
    for (r, i) in &pairs {
        raw.extend_from_slice(&r.to_le_bytes());
        raw.extend_from_slice(&i.to_le_bytes());
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("complex")
        .with_compound_data(ct, raw, 3);
    builder.write(&path).unwrap();

    // The C library should be able to open the file and see the dataset
    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("complex").unwrap();
    assert_eq!(ds.shape(), vec![3]);
}

#[repr(C)]
#[derive(hdf5::H5Type, Clone, Copy, Debug, PartialEq)]
struct PaddedTuple(
    #[hdf5(rename = "0")] i8,
    #[hdf5(rename = "1")] u64,
    #[hdf5(rename = "2")] f32,
);

#[test]
fn c_library_reads_explicit_offset_compound() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("padded_compound.h5");

    let datatype = CompoundTypeBuilder::with_size(24)
        .i8_field("0", 0)
        .u64_field("1", 8)
        .f32_field("2", 16)
        .build()
        .unwrap();
    let expected = [PaddedTuple(-4, 17, 2.5), PaddedTuple(8, u64::MAX, -1.25)];
    let mut raw = vec![0u8; expected.len() * 24];
    for (index, value) in expected.iter().enumerate() {
        let record = &mut raw[index * 24..(index + 1) * 24];
        record[0] = value.0 as u8;
        record[8..16].copy_from_slice(&value.1.to_le_bytes());
        record[16..20].copy_from_slice(&value.2.to_le_bytes());
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("padded")
        .with_compound_data(datatype, raw, expected.len() as u64);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let actual: Vec<PaddedTuple> = file.dataset("padded").unwrap().read_raw().unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn pure_reader_decodes_c_written_padded_compound() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_padded_compound.h5");
    let expected = [PaddedTuple(-9, 23, 4.5), PaddedTuple(3, 77, -6.25)];

    let file = hdf5::File::create(&path).unwrap();
    file.new_dataset_builder()
        .with_data(&expected)
        .create("padded")
        .unwrap();
    drop(file);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("padded").unwrap();
    assert_eq!(
        ds.read_compound::<(i8, u64, f32)>().unwrap(),
        expected.map(|value| (value.0, value.1, value.2))
    );
    match ds.datatype().unwrap() {
        Datatype::Compound { size, members } => {
            assert_eq!(size as usize, core::mem::size_of::<PaddedTuple>());
            assert_eq!(
                members
                    .iter()
                    .map(|member| (member.name.as_str(), member.byte_offset))
                    .collect::<Vec<_>>(),
                vec![("0", 0), ("1", 8), ("2", 16)]
            );
        }
        other => panic!("expected compound, got {other:?}"),
    }
}

#[repr(C)]
#[derive(hdf5::H5Type, Clone, Copy, Debug, PartialEq)]
struct NestedInner(#[hdf5(rename = "0")] i8, #[hdf5(rename = "1")] u64);

#[repr(C)]
#[derive(hdf5::H5Type, Clone, Copy, Debug, PartialEq)]
struct NestedOuter(#[hdf5(rename = "0")] NestedInner, #[hdf5(rename = "1")] f32);

#[test]
fn c_library_reads_nested_explicit_compound() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested_compound.h5");

    let inner = CompoundTypeBuilder::with_size(16)
        .i8_field("0", 0)
        .u64_field("1", 8)
        .build()
        .unwrap();
    let outer = CompoundTypeBuilder::with_size(24)
        .field("0", 0, inner)
        .f32_field("1", 16)
        .build()
        .unwrap();
    let expected = [
        NestedOuter(NestedInner(-1, 100), 2.0),
        NestedOuter(NestedInner(6, 200), -3.0),
    ];
    let mut raw = vec![0u8; expected.len() * 24];
    for (index, value) in expected.iter().enumerate() {
        let record = &mut raw[index * 24..(index + 1) * 24];
        record[0] = value.0.0 as u8;
        record[8..16].copy_from_slice(&value.0.1.to_le_bytes());
        record[16..20].copy_from_slice(&value.1.to_le_bytes());
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("nested")
        .with_compound_data(outer, raw, expected.len() as u64);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let actual: Vec<NestedOuter> = file.dataset("nested").unwrap().read_raw().unwrap();
    assert_eq!(actual, expected);
}

// ---------------------------------------------------------------------------
// Large dataset
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_large_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.h5");

    let n = 10_000;
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let mut builder = FileBuilder::new();
    builder.create_dataset("big").with_f64_data(&data);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("big").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values.len(), n);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[n - 1], (n - 1) as f64);
}

// ---------------------------------------------------------------------------
// Empty file (root group only)
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_empty_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.h5");

    let builder = FileBuilder::new();
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let members = file.member_names().unwrap();
    assert!(members.is_empty());
}

// ---------------------------------------------------------------------------
// Dense attributes (> 8 attributes triggers fractal heap storage)
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_varlen_ascii_array_attr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vl_ascii.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["x".into(), "y".into(), "velocity".into()]),
    );
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    // C library should be able to open and read the VL attribute
    let file = hdf5::File::open(&path).unwrap();
    let attr = file.attr("MATLAB_fields").unwrap();
    let vals = read_vl_ascii_attr(&attr);
    assert_eq!(vals, vec!["x", "y", "velocity"]);

    // Read dataset to verify file integrity
    let ds = file.dataset("x").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values, vec![1.0]);
}

/// A variable-length attribute in *dense* storage, which the compact case above
/// does not reach. The heap embeds a copy of the attribute's element bytes, and
/// those bytes are global-heap references: a heap built before they were patched
/// left the reference C library resolving addresses that were never assigned.
#[test]
fn crosscheck_varlen_ascii_array_attr_in_dense_storage() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vl_ascii_dense.h5");

    let expected = ["x", "y", "velocity"];
    let mut builder = FileBuilder::new();
    builder.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(expected.iter().map(|s| (*s).to_string()).collect()),
    );
    // Past the eight-attribute threshold, so the set is stored in a fractal heap.
    for i in 0..12 {
        builder.set_attr(&format!("filler_{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(file.attr_names().unwrap().len(), 13);
    let attr = file.attr("MATLAB_fields").unwrap();
    assert_eq!(read_vl_ascii_attr(&attr), expected);
}

/// Dense attributes in a file with a userblock. This crate's reader rejected these
/// files for a while, which made the C library the arbiter of whether the *writer*
/// was also wrong: it was not, and this pins that. `hdf5-pure`'s own side of it is
/// `tests/userblock_dense_attrs.rs`.
#[test]
fn crosscheck_dense_attributes_past_a_userblock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_dense_attrs.h5");

    let mut builder = FileBuilder::new();
    builder.with_userblock(512);
    let ds = builder.create_dataset("data");
    ds.with_f64_data(&[1.0, 2.0]);
    for i in 0..12 {
        ds.set_attr(&format!("attr_{i:02}"), AttrValue::I64(100 + i));
    }
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.attr_names().unwrap().len(), 12);
    for i in 0..12i64 {
        let got: i64 = ds
            .attr(&format!("attr_{i:02}"))
            .unwrap()
            .read_scalar()
            .unwrap();
        assert_eq!(got, 100 + i);
    }
}

#[test]
fn crosscheck_dense_attributes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense_attrs.h5");

    let mut builder = FileBuilder::new();
    let ds = builder.create_dataset("data");
    ds.with_f64_data(&[1.0, 2.0, 3.0]);
    for i in 0..20 {
        ds.set_attr(&format!("attr_{i:03}"), AttrValue::F64(i as f64 * 1.5));
    }
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);

    let attr_names = ds.attr_names().unwrap();
    assert_eq!(attr_names.len(), 20);

    // Verify a few attribute values
    let a0 = ds.attr("attr_000").unwrap().read_scalar::<f64>().unwrap();
    assert!((a0 - 0.0).abs() < 1e-10);

    let a5 = ds.attr("attr_005").unwrap().read_scalar::<f64>().unwrap();
    assert!((a5 - 7.5).abs() < 1e-10);

    let a19 = ds.attr("attr_019").unwrap().read_scalar::<f64>().unwrap();
    assert!((a19 - 28.5).abs() < 1e-10);
}

#[test]
fn crosscheck_empty_dataset_zero_dims() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty_ds.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_empty", AttrValue::U32(1));
    builder
        .create_dataset("empty")
        .with_dtype(make_f64_type())
        .with_shape(&[0, 0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("empty").unwrap();
    let shape = ds.shape();
    assert_eq!(shape, vec![0, 0]);

    let attr = file.attr("MATLAB_empty").unwrap();
    let val = attr.read_scalar::<u32>().unwrap();
    assert_eq!(val, 1);
}

#[test]
fn crosscheck_matlab_empty_shape_marker() {
    // MATLAB pattern: store original shape as u64 data with MATLAB_empty=1
    let dir = tempdir().unwrap();
    let path = dir.path().join("matlab_empty.h5");

    let matlab_dims: &[usize] = &[3, 4];
    let shape_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u64_data(&shape_data)
        .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()))
        .set_attr("MATLAB_empty", AttrValue::U32(1));
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape(), vec![2]);
    let values: Vec<u64> = ds.read_1d().unwrap().to_vec();
    assert_eq!(values, vec![3u64, 4]);

    // Verify dataset-level attributes
    let class_attr = ds.attr("MATLAB_class").unwrap();
    let class_val = class_attr
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(class_val.as_str(), "double");

    let empty_attr = ds.attr("MATLAB_empty").unwrap();
    let empty_val = empty_attr.read_scalar::<u32>().unwrap();
    assert_eq!(empty_val, 1);
}

#[test]
fn crosscheck_path_references_2d_shape() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("refs_2d.h5");

    let mut builder = FileBuilder::new();

    let mut refs_grp = builder.create_group("#refs#");
    refs_grp.create_dataset("a").with_f64_data(&[1.0]);
    refs_grp.create_dataset("b").with_f64_data(&[2.0]);
    refs_grp.create_dataset("c").with_f64_data(&[3.0]);
    refs_grp.create_dataset("d").with_f64_data(&[4.0]);
    builder.add_group(refs_grp.finish());

    // Shape [1, 4] — row vector of references
    builder
        .create_dataset("row_refs")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c", "#refs#/d"])
        .with_shape(&[1, 4]);

    // Shape [2, 1] — column vector of references
    builder
        .create_dataset("col_refs")
        .with_path_references(&["#refs#/a", "#refs#/b"])
        .with_shape(&[2, 1]);

    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    let row = file.dataset("row_refs").unwrap();
    assert_eq!(row.shape(), vec![1, 4]);

    let col = file.dataset("col_refs").unwrap();
    assert_eq!(col.shape(), vec![2, 1]);
}

#[test]
fn crosscheck_matlab_refs_subsystem_pattern() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("refs_subsystem.h5");

    let mut builder = FileBuilder::new();

    // --- #refs# group ---
    let mut refs_grp = builder.create_group("#refs#");
    refs_grp
        .create_dataset("a")
        .with_u16_data(&[72, 101, 108, 108, 111]);
    refs_grp
        .create_dataset("b")
        .with_u16_data(&[87, 111, 114, 108, 100]);
    refs_grp.create_dataset("c").with_u16_data(&[70, 111, 111]);

    // Cross-references within #refs#
    refs_grp
        .create_dataset("cell_data")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c"])
        .with_shape(&[1, 3]);

    refs_grp
        .create_dataset("type_info")
        .with_u8_data(&[0, 0, 0, 0, 0, 0, 1, 0])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));

    builder.add_group(refs_grp.finish());

    // --- #subsystem# group ---
    let mut subsys_grp = builder.create_group("#subsystem#");
    subsys_grp
        .create_dataset("MCOS")
        .with_path_references(&["#refs#/cell_data", "#refs#/type_info"])
        .with_shape(&[2, 1]);
    builder.add_group(subsys_grp.finish());

    // --- Root dataset with reference ---
    builder
        .create_dataset("data")
        .with_path_references(&["#refs#/a"])
        .with_shape(&[1, 1])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));

    builder.write(&path).unwrap();

    // --- Verify with C HDF5 ---
    let file = hdf5::File::open(&path).unwrap();

    // Group structure
    let root_members = file.member_names().unwrap();
    assert!(root_members.contains(&"#refs#".to_string()));
    assert!(root_members.contains(&"#subsystem#".to_string()));
    assert!(root_members.contains(&"data".to_string()));

    // #refs# children
    let refs = file.group("#refs#").unwrap();
    let refs_members = refs.member_names().unwrap();
    assert!(refs_members.contains(&"a".to_string()));
    assert!(refs_members.contains(&"b".to_string()));
    assert!(refs_members.contains(&"c".to_string()));
    assert!(refs_members.contains(&"cell_data".to_string()));
    assert!(refs_members.contains(&"type_info".to_string()));

    // Data shapes
    let a = file.dataset("#refs#/a").unwrap();
    assert_eq!(a.shape(), vec![5]);
    let a_data: Vec<u16> = a.read_1d().unwrap().to_vec();
    assert_eq!(a_data, vec![72, 101, 108, 108, 111]);

    let cell_data = file.dataset("#refs#/cell_data").unwrap();
    assert_eq!(cell_data.shape(), vec![1, 3]);

    let mcos = file.dataset("#subsystem#/MCOS").unwrap();
    assert_eq!(mcos.shape(), vec![2, 1]);

    let data = file.dataset("data").unwrap();
    assert_eq!(data.shape(), vec![1, 1]);

    // Dataset-level attribute
    let class_attr = data.attr("MATLAB_class").unwrap();
    let class_val = class_attr
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(class_val.as_str(), "string");
}

/// Verify forge VarLenAsciiCharArray GCOL encoding: minimum size, structure, and
/// that the C library can read back the exact string values.
#[test]
fn crosscheck_varlen_ascii_array_encoding_vs_metno() {
    let dir = tempdir().unwrap();
    let forge_path = dir.path().join("forge_vl.h5");

    let fields = vec!["x".to_string(), "y".to_string(), "z".to_string()];

    // --- Write with hdf5-pure ---
    let mut builder = FileBuilder::new();
    builder.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(fields.clone()),
    );
    builder.create_dataset("dummy").with_f64_data(&[1.0]);
    builder.write(&forge_path).unwrap();

    // --- Verify raw bytes: GCOL present and well-formed ---
    let forge_bytes = std::fs::read(&forge_path).unwrap();
    let fg = forge_bytes
        .windows(4)
        .position(|w| w == b"GCOL")
        .expect("forge file missing GCOL signature");
    assert_eq!(forge_bytes[fg + 4], 1, "GCOL version should be 1");

    // Collection size must be >= 4096 (H5HG_MINSIZE)
    let collection_size = u64::from_le_bytes(forge_bytes[fg + 8..fg + 16].try_into().unwrap());
    assert!(
        collection_size >= 4096,
        "GCOL size {collection_size} < H5HG_MINSIZE"
    );

    // First object at GCOL+16: index=1, data="x"
    let obj0 = fg + 16;
    let idx = u16::from_le_bytes([forge_bytes[obj0], forge_bytes[obj0 + 1]]);
    assert_eq!(idx, 1);
    let obj_size = u64::from_le_bytes(forge_bytes[obj0 + 8..obj0 + 16].try_into().unwrap());
    assert_eq!(obj_size, 1); // "x" is 1 byte
    assert_eq!(forge_bytes[obj0 + 16], b'x');

    // --- Read back with C library ---
    let file = hdf5::File::open(&forge_path).unwrap();
    let attr = file.attr("MATLAB_fields").unwrap();
    assert_eq!(attr.shape(), vec![3]);

    let vals = read_vl_ascii_attr(&attr);
    assert_eq!(vals, vec!["x", "y", "z"]);

    // Verify the encoded datatype matches the MATLAB-expected shape
    // (`H5T_VLEN { H5T_STRING { STRSIZE 1 } }`).
    let descr = format!("{:?}", attr.dtype().unwrap().to_descriptor().unwrap());
    assert!(
        descr.contains("VarLenArray") && descr.contains("FixedAscii(1)"),
        "unexpected dtype: {descr}"
    );
}

#[test]
fn crosscheck_varlen_ascii_on_nested_group() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("grp_vl.h5");

    let mut builder = FileBuilder::new();
    let mut grp = builder.create_group("my_struct");
    grp.create_dataset("x").with_f64_data(&[1.0, 2.0]);
    grp.create_dataset("y").with_f64_data(&[3.0, 4.0]);
    grp.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
    grp.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["x".into(), "y".into()]),
    );
    builder.add_group(grp.finish());
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let grp = file.group("my_struct").unwrap();

    // Read VL attribute on the nested group
    let attr = grp.attr("MATLAB_fields").unwrap();
    let vals = read_vl_ascii_attr(&attr);
    assert_eq!(vals.len(), 2);
    assert_eq!(vals, vec!["x", "y"]);

    // Verify ASCII attr
    let class_attr = grp.attr("MATLAB_class").unwrap();
    let class_val = class_attr
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(class_val.as_str(), "struct");

    // Verify datasets
    let x: Vec<f64> = file
        .dataset("my_struct/x")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(x, vec![1.0, 2.0]);
    let y: Vec<f64> = file
        .dataset("my_struct/y")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(y, vec![3.0, 4.0]);
}

/// Comprehensive userblock address test: exercises every address type
/// (contiguous data, group links, path references, VarLenAsciiCharArray GCOL)
/// with a 512-byte userblock. A single off-by-userblock address would
/// cause the C library to fail.
#[test]
fn crosscheck_userblock_all_address_types() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_all.h5");

    let mut builder = FileBuilder::new();
    builder.with_userblock(512);

    // Root VL attr (GCOL address must be relative to base)
    builder.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["alpha".into(), "beta".into()]),
    );
    builder.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));

    // Nested group with its own VL attr (second GCOL)
    let mut grp = builder.create_group("inner");
    grp.create_dataset("vals")
        .with_f64_data(&[10.0, 20.0, 30.0]);
    grp.create_dataset("ids").with_i32_data(&[1, 2, 3]);
    grp.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["vals".into(), "ids".into()]),
    );
    builder.add_group(grp.finish());

    // Root contiguous dataset (data address must be relative to base)
    builder.create_dataset("alpha").with_f64_data(&[1.0, 2.0]);
    builder.create_dataset("beta").with_i32_data(&[42]);

    // Reference dataset (resolved addresses must be relative to base)
    builder
        .create_dataset("refs")
        .with_path_references(&["alpha", "beta", "inner"])
        .with_shape(&[1, 3]);

    builder.write(&path).unwrap();

    // --- Verify with C library ---
    let file = hdf5::File::open(&path).unwrap();

    // Root VL attr
    let root_fields = file.attr("MATLAB_fields").unwrap();
    let vals = read_vl_ascii_attr(&root_fields);
    assert_eq!(vals, vec!["alpha", "beta"]);

    // Root ASCII attr
    let class = file.attr("MATLAB_class").unwrap();
    let cv = class.read_scalar::<hdf5::types::FixedAscii<32>>().unwrap();
    assert_eq!(cv.as_str(), "struct");

    // Contiguous datasets
    let alpha: Vec<f64> = file.dataset("alpha").unwrap().read_1d().unwrap().to_vec();
    assert_eq!(alpha, vec![1.0, 2.0]);
    let beta: Vec<i32> = file.dataset("beta").unwrap().read_1d().unwrap().to_vec();
    assert_eq!(beta, vec![42]);

    // Nested group with VL attr
    let inner = file.group("inner").unwrap();
    let inner_fields = inner.attr("MATLAB_fields").unwrap();
    let iv = read_vl_ascii_attr(&inner_fields);
    assert_eq!(iv, vec!["vals", "ids"]);

    let inner_vals: Vec<f64> = file
        .dataset("inner/vals")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(inner_vals, vec![10.0, 20.0, 30.0]);
    let inner_ids: Vec<i32> = file
        .dataset("inner/ids")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(inner_ids, vec![1, 2, 3]);

    // Reference dataset shape
    let refs = file.dataset("refs").unwrap();
    assert_eq!(refs.shape(), vec![1, 3]);
}

#[test]
fn crosscheck_group_only_no_datasets() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("grp_only.h5");

    let mut builder = FileBuilder::new();

    let mut outer = builder.create_group("outer");
    outer.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));

    let mut child_a = outer.create_group("a");
    child_a.create_dataset("val").with_f64_data(&[1.0]);
    child_a.set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    outer.add_group(child_a.finish());

    let mut child_b = outer.create_group("b");
    child_b.create_dataset("val").with_i32_data(&[42]);
    child_b.set_attr("MATLAB_class", AttrValue::AsciiString("int32".into()));
    outer.add_group(child_b.finish());

    builder.add_group(outer.finish());
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    // outer group has only sub-groups, no datasets
    let outer = file.group("outer").unwrap();
    let members = outer.member_names().unwrap();
    assert!(members.contains(&"a".to_string()));
    assert!(members.contains(&"b".to_string()));

    let class = outer.attr("MATLAB_class").unwrap();
    let cv = class.read_scalar::<hdf5::types::FixedAscii<32>>().unwrap();
    assert_eq!(cv.as_str(), "struct");

    // Leaf data
    let a_val: Vec<f64> = file
        .dataset("outer/a/val")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(a_val, vec![1.0]);
    let b_val: Vec<i32> = file
        .dataset("outer/b/val")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(b_val, vec![42]);

    // Sub-group attrs
    let a_class = file.group("outer/a").unwrap().attr("MATLAB_class").unwrap();
    let acv = a_class
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(acv.as_str(), "double");
}

/// Verify AsciiString is written as fixed-length (not variable-length) and
/// works for long class names like "canonical empty" (15 chars).
#[test]
fn crosscheck_long_ascii_class_attr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("long_class.h5");

    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_f64_data(&[1.0]).set_attr(
        "MATLAB_class",
        AttrValue::AsciiString("canonical empty".into()),
    );
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("x").unwrap();
    let attr = ds.attr("MATLAB_class").unwrap();

    // Must be fixed-length string, not variable-length
    let dtype = attr.dtype().unwrap();
    assert!(
        format!("{dtype:?}").contains("string") || format!("{dtype:?}").contains("String"),
        "dtype = {dtype:?}"
    );

    let val = attr.read_scalar::<hdf5::types::FixedAscii<32>>().unwrap();
    assert_eq!(val.as_str(), "canonical empty");
}

/// Full MATLAB struct pattern: group with MATLAB_class, MATLAB_fields,
/// and child datasets with their own MATLAB_class attributes.
#[test]
fn crosscheck_matlab_struct_pattern() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("matlab_struct.h5");

    let mut builder = FileBuilder::new();
    let mut g = builder.create_group("mystruct");
    g.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
    g.create_dataset("field1")
        .with_f64_data(&[1.0])
        .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    g.create_dataset("field2")
        .with_i32_data(&[10, 20])
        .set_attr("MATLAB_class", AttrValue::AsciiString("int32".into()));
    g.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["field1".into(), "field2".into()]),
    );
    builder.add_group(g.finish());
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let grp = file.group("mystruct").unwrap();

    // Struct class
    let class = grp.attr("MATLAB_class").unwrap();
    let cv = class.read_scalar::<hdf5::types::FixedAscii<32>>().unwrap();
    assert_eq!(cv.as_str(), "struct");

    // Field list
    let fields = grp.attr("MATLAB_fields").unwrap();
    let fv = read_vl_ascii_attr(&fields);
    assert_eq!(fv, vec!["field1", "field2"]);

    // Child datasets and their attrs
    let f1: Vec<f64> = file
        .dataset("mystruct/field1")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(f1, vec![1.0]);
    let f1_class = file
        .dataset("mystruct/field1")
        .unwrap()
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(f1_class.as_str(), "double");

    let f2: Vec<i32> = file
        .dataset("mystruct/field2")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(f2, vec![10, 20]);
}

/// MATLAB cell array: #refs# group + reference dataset with cell class.
#[test]
fn crosscheck_matlab_cell_array() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("matlab_cell.h5");

    let mut builder = FileBuilder::new();
    let mut refs_group = builder.create_group("#refs#");
    refs_group.create_dataset("ref_0").with_f64_data(&[1.0]);
    refs_group.create_dataset("ref_1").with_i32_data(&[42]);
    builder.add_group(refs_group.finish());

    builder
        .create_dataset("mycell")
        .with_path_references(&["#refs#/ref_0", "#refs#/ref_1"])
        .with_shape(&[1, 2])
        .set_attr("MATLAB_class", AttrValue::AsciiString("cell".into()));

    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    // Cell reference dataset
    let mycell = file.dataset("mycell").unwrap();
    assert_eq!(mycell.shape(), vec![1, 2]);
    let cell_class = mycell
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cell_class.as_str(), "cell");

    // Referenced data
    let r0: Vec<f64> = file
        .dataset("#refs#/ref_0")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(r0, vec![1.0]);
    let r1: Vec<i32> = file
        .dataset("#refs#/ref_1")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(r1, vec![42]);
}

/// Userblock with MATLAB header bytes at offsets 126-127.
#[test]
fn crosscheck_matlab_userblock_header() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("matlab_ub.h5");

    let mut builder = FileBuilder::new();
    builder.with_userblock(512);
    builder.create_dataset("data").with_f64_data(&[3.14]);
    let mut bytes = builder.finish().unwrap();

    // Write MATLAB "IM" marker at offsets 126-127
    bytes[126] = b'I';
    bytes[127] = b'M';
    std::fs::write(&path, &bytes).unwrap();

    // C library should still read the file
    let file = hdf5::File::open(&path).unwrap();
    let vals: Vec<f64> = file.dataset("data").unwrap().read_1d().unwrap().to_vec();
    assert_eq!(vals, vec![3.14]);

    // Verify userblock is intact
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(raw[126], b'I');
    assert_eq!(raw[127], b'M');
    // HDF5 signature should be at offset 512
    assert_eq!(&raw[512..520], b"\x89HDF\r\n\x1a\n");
}

// ---------------------------------------------------------------------------
// Scale-offset filter (id 6) interop, both directions.
// ---------------------------------------------------------------------------

#[test]
fn crosscheck_read_scale_offset_int_from_c_lib() {
    // Reference C library writes an integer scale-offset dataset (multi-chunk,
    // partial last chunk); hdf5-pure must decode it losslessly.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_scaleoffset_int.h5");

    let n = 250usize;
    let data: Vec<i32> = (0..n).map(|i| 1000 + (i as i32 % 37)).collect();
    {
        let file = hdf5::File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .scale_offset(hdf5::filters::ScaleOffset::Integer(0))
            .chunk([40])
            .shape([n])
            .create("data")
            .unwrap();
        ds.write_raw(data.as_slice()).unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let file = hdf5_pure::File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), data);
}

#[test]
fn crosscheck_read_scale_offset_float_from_c_lib() {
    // Reference C library writes a float D-scale dataset; hdf5-pure decodes it
    // within the decimal tolerance.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_scaleoffset_float.h5");

    let decimals = 3u8;
    let n = 200usize;
    let data: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 0.01).collect();
    {
        let file = hdf5::File::create(&path).unwrap();
        let ds = file
            .new_dataset::<f64>()
            .scale_offset(hdf5::filters::ScaleOffset::FloatDScale(decimals))
            .chunk([64])
            .shape([n])
            .create("data")
            .unwrap();
        ds.write_raw(data.as_slice()).unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let file = hdf5_pure::File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    let got = ds.read_f64().unwrap();
    let tol = 0.5 * 10f64.powi(-(decimals as i32));
    assert_eq!(got.len(), n);
    for (g, w) in got.iter().zip(data.iter()) {
        assert!((g - w).abs() <= tol, "got {g}, want {w}");
    }
}

#[test]
fn crosscheck_scale_offset_int_compressed() {
    // hdf5-pure writes integer scale-offset; the reference C library reads it.
    let dir = tempdir().unwrap();
    let path = dir.path().join("scaleoffset_int.h5");

    let n = 250usize;
    let data: Vec<i32> = (0..n).map(|i| -50 + (i as i32 % 91)).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[40])
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<i32>().unwrap();
    assert_eq!(values, data);
}

#[test]
fn crosscheck_scale_offset_float_dscale() {
    // hdf5-pure writes float D-scale; the reference C library reads it within
    // the decimal tolerance.
    let dir = tempdir().unwrap();
    let path = dir.path().join("scaleoffset_float.h5");

    let decimals = 2i32;
    let n = 180usize;
    let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.05).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[50])
        .with_scale_offset(hdf5_pure::ScaleOffset::FloatDScale(decimals));
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<f64>().unwrap();
    let tol = 0.5 * 10f64.powi(-decimals);
    assert_eq!(values.len(), n);
    for (g, w) in values.iter().zip(data.iter()) {
        assert!((g - w).abs() <= tol, "got {g}, want {w}");
    }
}

#[test]
fn crosscheck_scale_offset_then_deflate() {
    // hdf5-pure writes [scaleoffset, deflate]; the reference C library reads it.
    let dir = tempdir().unwrap();
    let path = dir.path().join("scaleoffset_deflate.h5");

    let n = 300usize;
    let data: Vec<i32> = (0..n).map(|i| 7000 + (i as i32 % 23)).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[64])
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0))
        .with_deflate(5);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let values = ds.read_raw::<i32>().unwrap();
    assert_eq!(values, data);
}

#[test]
fn crosscheck_vlen_string_dataset() {
    // hdf5-pure writes a variable-length UTF-8 string dataset; the reference C
    // library reads back the values and confirms the datatype is VL Unicode.
    use hdf5::types::VarLenUnicode;

    let dir = tempdir().unwrap();
    let path = dir.path().join("vlen_strings.h5");

    let words = ["alpha", "", "γάμμα", "delta"];
    let mut builder = FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&words);
    builder.write(&path).unwrap();

    // hdf5-pure round-trips the values.
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("labels")
            .unwrap()
            .read_vlen_strings(VlenStringReadOptions::default())
            .unwrap(),
        words
    );

    // The reference C library agrees on values and that the type is VL Unicode.
    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    let vals = ds.read_raw::<VarLenUnicode>().unwrap();
    let got: Vec<String> = vals.iter().map(|v| v.as_str().to_string()).collect();
    assert_eq!(got, words);
    assert!(
        ds.dtype().unwrap().is::<VarLenUnicode>(),
        "datatype must be variable-length Unicode"
    );
}

#[test]
fn crosscheck_vlen_strings_span_multiple_heap_collections() {
    // One global heap collection indexes at most 65,535 objects (the object
    // index is a u16, 0 reserved for the free-space marker), so a larger
    // variable-length dataset is split across collections whose indices restart
    // at 1. Pin that against the reference library, not just our own reader: it
    // is what proves the per-collection indices and addresses agree with the
    // format rather than only with this crate's reader.
    use hdf5::types::VarLenUnicode;

    let dir = tempdir().unwrap();
    let path = dir.path().join("vlen_multi_collection.h5");

    // Just past two collections, so the split happens twice and the last
    // collection is only partly filled.
    let count = 2 * (u16::MAX as usize) + 7;
    let values: Vec<String> = (0..count).map(|i| format!("s{i}")).collect();
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let mut builder = FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    let vals = ds.read_raw::<VarLenUnicode>().unwrap();
    assert_eq!(vals.len(), count);
    // Both sides of each collection boundary, and the ends.
    for i in [0, 65_534, 65_535, 131_069, 131_070, count - 1] {
        assert_eq!(vals[i].as_str(), format!("s{i}"), "element {i} differs");
    }
}

/// An empty-string attribute must not make the object's *other* attributes
/// unreadable to the C library.
///
/// A fixed-width string datatype has to be at least one byte wide. Writing
/// `STRSIZE 0` for an empty string made libhdf5 fail with "invalid datatype
/// size" — and it fails inside `H5Aiterate2`, so every attribute on that object
/// became unreachable, not only the empty one. The healthy attribute here is
/// what makes that blast radius visible: a test carrying only the empty string
/// would pass the moment the empty one alone was fixed.
#[test]
fn an_empty_string_attribute_does_not_hide_its_neighbours() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty_string_attr.h5");

    let mut builder = FileBuilder::new();
    builder.set_attr("empty_utf8", AttrValue::String(String::new()));
    builder.set_attr("empty_ascii", AttrValue::AsciiString(String::new()));
    builder.set_attr("all_empty", AttrValue::StringArray(vec![String::new(); 2]));
    builder.set_attr("healthy", AttrValue::AsciiString("v".into()));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    // Iteration is where the zero-size datatype used to fail.
    let mut names = file.attr_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        vec![
            "all_empty".to_string(),
            "empty_ascii".to_string(),
            "empty_utf8".to_string(),
            "healthy".to_string(),
        ]
    );

    // The neighbour opens and reads.
    let healthy = file.attr("healthy").unwrap();
    assert_eq!(
        healthy
            .read_scalar::<hdf5::types::FixedAscii<1>>()
            .unwrap()
            .as_str(),
        "v"
    );

    // And each empty one is a one-byte-wide string holding nothing. The C
    // library will not convert between charsets, so each is read as its own.
    let utf8 = file.attr("empty_utf8").unwrap();
    assert_eq!(utf8.size(), 1);
    assert!(
        utf8.read_scalar::<hdf5::types::FixedUnicode<1>>()
            .unwrap()
            .as_str()
            .is_empty()
    );
    let ascii = file.attr("empty_ascii").unwrap();
    assert_eq!(ascii.size(), 1);
    assert!(
        ascii
            .read_scalar::<hdf5::types::FixedAscii<1>>()
            .unwrap()
            .as_str()
            .is_empty()
    );
    let all_empty = file.attr("all_empty").unwrap();
    assert_eq!(all_empty.size(), 2);

    // hdf5-pure reads its own file back with the values intact.
    let ours = File::open(&path).unwrap();
    let attrs = ours.root().attrs().unwrap();
    assert_eq!(
        attrs.get("empty_utf8"),
        Some(&AttrValue::String(String::new()))
    );
    assert_eq!(
        attrs.get("empty_ascii"),
        Some(&AttrValue::AsciiString(String::new()))
    );
    assert_eq!(
        attrs.get("all_empty"),
        Some(&AttrValue::StringArray(vec![String::new(); 2]))
    );
}
