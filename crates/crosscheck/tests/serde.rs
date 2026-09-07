// Crosschecks against the reference HDF5 C library (`hdf5-metno`), gated to
// 64-bit little-endian targets; skip elsewhere so `cross test` on `i686` and
// `s390x` stays pure-Rust.
#![cfg(all(
    feature = "serde",
    not(target_pointer_width = "32"),
    target_endian = "little"
))]
//! Crosscheck: files produced by our serde layer are readable by the C HDF5
//! library (via `hdf5-metno`). This gives us confidence that the `.mat` v7.3
//! files we produce are valid HDF5 and follow MATLAB conventions that other
//! tools (scipy, MATLAB itself) should also accept.

use hdf5_pure::mat::{self, Complex64, ComplexI16, Matrix};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Experiment {
    name: String,
    trial: u32,
    temperature: f64,
    samples: Vec<f64>,
    config: Config,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Config {
    threshold: f64,
    tag: String,
}

#[test]
fn c_library_reads_nested_struct_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("experiment.mat");

    let e = Experiment {
        name: "alpha".into(),
        trial: 3,
        temperature: 300.0,
        samples: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        config: Config {
            threshold: 0.5,
            tag: "run".into(),
        },
    };

    mat::to_file(&e, &path).unwrap();

    // Now open with hdf5-metno (C library).
    let file = hdf5::File::open(&path).unwrap();

    // Scalars in MATLAB are 1×1 datasets (not true HDF5 scalars).
    let trial = file.dataset("trial").unwrap();
    assert_eq!(trial.shape(), vec![1, 1]);
    let tv: Vec<u32> = trial.read_raw().unwrap();
    assert_eq!(tv, vec![3]);

    let temp = file.dataset("temperature").unwrap();
    assert_eq!(temp.shape(), vec![1, 1]);
    let tv: Vec<f64> = temp.read_raw().unwrap();
    assert_eq!(tv, vec![300.0]);

    let samples = file.dataset("samples").unwrap();
    assert_eq!(samples.shape(), vec![1, 5]);
    let sv: Vec<f64> = samples.read_raw().unwrap();
    assert_eq!(sv, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Nested group with MATLAB struct class + MATLAB_fields VL attribute.
    let cfg = file.group("config").unwrap();
    let class_attr = cfg.attr("MATLAB_class").unwrap();
    let class_val = class_attr
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(class_val.as_str(), "struct");
    let fields_attr = cfg.attr("MATLAB_fields").unwrap();
    // MATLAB-compatible encoding is `H5T_VLEN { H5T_STRING { STRSIZE 1 } }`,
    // so we read as a VLEN of single-byte FixedAscii.
    let raw: Vec<hdf5::types::VarLenArray<hdf5::types::FixedAscii<1>>> =
        fields_attr.read_raw().unwrap();
    let field_names: Vec<String> = raw
        .iter()
        .map(|vl| vl.iter().flat_map(|c| c.as_str().chars()).collect())
        .collect();
    assert_eq!(field_names, vec!["threshold", "tag"]);

    // char dataset: MATLAB char = UTF-16 in uint16 storage. Strings are
    // MATLAB row vectors, which means HDF5 shape [N, 1] (column-major).
    let name = file.dataset("name").unwrap();
    assert_eq!(name.shape(), vec![5, 1]); // "alpha" = 5 UTF-16 code units
    let units: Vec<u16> = name.read_raw().unwrap();
    assert_eq!(String::from_utf16(&units).unwrap(), "alpha");
    let name_class = name
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(name_class.as_str(), "char");
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct MatStruct {
    data: Matrix<f64>,
}

#[test]
fn c_library_reads_2d_matrix_with_matlab_orientation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("matrix.mat");

    // Rust 2×3 row-major matrix:
    //   [[1 2 3]
    //    [4 5 6]]
    let s = MatStruct {
        data: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    };
    mat::to_file(&s, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();

    // HDF5 shape is [cols, rows] = [3, 2]; MATLAB will interpret as 2×3.
    assert_eq!(ds.shape(), vec![3, 2]);
    // Column-major bytes: col0=(1,4), col1=(2,5), col2=(3,6).
    let raw: Vec<f64> = ds.read_raw().unwrap();
    assert_eq!(raw, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Waves {
    signal: Vec<Complex64>,
}

#[test]
fn c_library_reads_complex_as_compound() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("waves.mat");

    let w = Waves {
        signal: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
        ],
    };
    mat::to_file(&w, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("signal").unwrap();
    assert_eq!(ds.shape(), vec![1, 3]);

    // Read as a compound {real: f64, imag: f64} type.
    #[repr(C)]
    #[derive(hdf5::H5Type, Clone, Copy, Debug)]
    struct Pair {
        real: f64,
        imag: f64,
    }
    let pairs: Vec<Pair> = ds.read_raw().unwrap();
    assert_eq!(pairs.len(), 3);
    assert_eq!(pairs[0].real, 1.0);
    assert_eq!(pairs[0].imag, 0.0);
    assert_eq!(pairs[1].real, 0.0);
    assert_eq!(pairs[1].imag, 1.0);
    assert_eq!(pairs[2].real, -1.0);

    // MATLAB_class is the real-number class for complex.
    let class_attr = ds.attr("MATLAB_class").unwrap();
    let cls = class_attr
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "double");
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Capture {
    samples: Vec<ComplexI16>,
}

/// The C library resolves the compound and its component class independently
/// of anything this crate believes, so it is the check that the `int16`
/// complex layout is the one MATLAB will see: two 2-byte signed members at
/// offsets 0 and 2, and four bytes per sample rather than eight.
#[test]
fn c_library_reads_complex_int16_as_a_four_byte_compound() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("capture.mat");

    let c = Capture {
        samples: vec![
            ComplexI16::new(i16::MIN, i16::MAX),
            ComplexI16::new(0, -1),
            ComplexI16::new(1234, -4321),
        ],
    };
    mat::to_file(&c, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("samples").unwrap();
    assert_eq!(ds.shape(), vec![1, 3]);

    #[repr(C)]
    #[derive(hdf5::H5Type, Clone, Copy, Debug)]
    struct Pair {
        real: i16,
        imag: i16,
    }
    assert_eq!(ds.dtype().unwrap().size(), 4);
    assert_eq!(ds.storage_size(), 4 * 3);

    let pairs: Vec<Pair> = ds.read_raw().unwrap();
    assert_eq!(pairs.len(), 3);
    assert_eq!((pairs[0].real, pairs[0].imag), (i16::MIN, i16::MAX));
    assert_eq!((pairs[1].real, pairs[1].imag), (0, -1));
    assert_eq!((pairs[2].real, pairs[2].imag), (1234, -4321));

    // The class names the component, not anything complex-specific.
    let class_attr = ds.attr("MATLAB_class").unwrap();
    let cls = class_attr
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "int16");
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Flags {
    on: bool,
    bits: Vec<bool>,
}

#[test]
fn c_library_reads_logical_as_uint8() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("flags.mat");

    let f = Flags {
        on: true,
        bits: vec![true, false, true, true, false],
    };
    mat::to_file(&f, &path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    let on = file.dataset("on").unwrap();
    assert_eq!(on.shape(), vec![1, 1]);
    let on_raw: Vec<u8> = on.read_raw().unwrap();
    assert_eq!(on_raw, vec![1]);
    let cls = on
        .attr("MATLAB_class")
        .unwrap()
        .read_scalar::<hdf5::types::FixedAscii<32>>()
        .unwrap();
    assert_eq!(cls.as_str(), "logical");

    let bits = file.dataset("bits").unwrap();
    assert_eq!(bits.shape(), vec![1, 5]);
    let raw: Vec<u8> = bits.read_raw().unwrap();
    assert_eq!(raw, vec![1, 0, 1, 1, 0]);
}

/// End-to-end: write with hdf5-pure, read back with hdf5-pure via the serde
/// deserializer, using the same `to_file` / `from_file` convenience methods.
#[test]
fn to_file_and_from_file_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("roundtrip.mat");

    let e = Experiment {
        name: "omega".into(),
        trial: 9,
        temperature: 77.3,
        samples: vec![0.1, 0.2, 0.3],
        config: Config {
            threshold: 0.8,
            tag: "production".into(),
        },
    };

    mat::to_file(&e, &path).unwrap();
    let back: Experiment = mat::from_file(&path).unwrap();
    assert_eq!(back, e);
}
