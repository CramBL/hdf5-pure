#![cfg(feature = "serde")]
//! Complex arrays with integer components.
//!
//! Hardware that samples in the complex plane — ADCs, digitizer front ends —
//! produces pairs of signed 16-bit integers, and widening them to `single` on
//! the way into a `.mat` file doubles the payload while adding nothing: the
//! mantissa bits are zero by construction. These tests pin the layout that
//! avoids that (a `{real, imag}` compound of the component class, with
//! `MATLAB_class` naming the component), the exact-width guarantee that makes
//! the file self-describing, and the shapes and edge values the float complex
//! path already handles.

use hdf5_pure::mat::options::{Compression, OneDimensionalMode};
use hdf5_pure::mat::{
    self, Complex32, Complex64, ComplexI8, ComplexI16, ComplexI32, ComplexI64, ComplexU8,
    ComplexU16, ComplexU32, ComplexU64, Matrix, Options,
};
use hdf5_pure::{
    AttrValue, CompoundTypeBuilder, Datatype, DatatypeByteOrder, File, FileBuilder, LibVer,
    make_i64_type,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Shapes {
    scalar: ComplexI16,
    row: Vec<ComplexI16>,
    column: Matrix<ComplexI16>,
    matrix: Matrix<ComplexI16>,
    empty: Matrix<ComplexI16>,
}

fn shapes() -> Shapes {
    Shapes {
        scalar: ComplexI16::new(7, -9),
        row: vec![
            ComplexI16::new(1, 2),
            ComplexI16::new(3, 4),
            ComplexI16::new(5, 6),
        ],
        column: Matrix::from_row_major(
            3,
            1,
            vec![
                ComplexI16::new(10, -10),
                ComplexI16::new(20, -20),
                ComplexI16::new(30, -30),
            ],
        ),
        matrix: Matrix::from_row_major(
            2,
            3,
            vec![
                ComplexI16::new(1, -1),
                ComplexI16::new(2, -2),
                ComplexI16::new(3, -3),
                ComplexI16::new(4, -4),
                ComplexI16::new(5, -5),
                ComplexI16::new(6, -6),
            ],
        ),
        empty: Matrix::from_row_major(0, 0, Vec::new()),
    }
}

#[test]
fn every_shape_round_trips() {
    let v = shapes();
    let bytes = mat::to_bytes(&v).unwrap();
    let back: Shapes = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

/// A 1×N and an N×1 array of the same values are different MATLAB arrays, and
/// the shape has to survive the column-major storage round trip in both
/// orientations.
#[test]
fn row_and_column_orientation_survive() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Oriented {
        row: Matrix<ComplexI16>,
        column: Matrix<ComplexI16>,
    }

    let pairs = || {
        vec![
            ComplexI16::new(10, -10),
            ComplexI16::new(20, -20),
            ComplexI16::new(30, -30),
        ]
    };
    let v = Oriented {
        row: Matrix::from_row_major(1, 3, pairs()),
        column: Matrix::from_row_major(3, 1, pairs()),
    };

    let back: Oriented = mat::from_bytes(&mat::to_bytes(&v).unwrap()).unwrap();
    assert_eq!((back.row.rows(), back.row.cols()), (1, 3));
    assert_eq!((back.column.rows(), back.column.cols()), (3, 1));
    assert_eq!(back, v);
}

/// A one-element complex array reads back as a complex *scalar*, which still
/// has to satisfy a `Vec` target — the same allowance the real numeric path
/// makes for a one-element numeric array.
#[test]
fn a_one_element_array_still_deserializes_as_a_vec() {
    let one = Capture {
        samples: vec![ComplexI16::new(5, -5)],
    };
    let back: Capture = mat::from_bytes(&mat::to_bytes(&one).unwrap()).unwrap();
    assert_eq!(back, one);
}

/// An empty complex array carries no pairs, so only the class attribute and
/// the compound datatype can say what it was — which is exactly what a reader
/// needs to hand it back as an empty `int16` complex array rather than an
/// untyped empty.
#[test]
fn an_empty_array_keeps_its_component_class() {
    let v = shapes();
    let bytes = mat::to_bytes(&v).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("empty").unwrap();

    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("int16")
    );
    assert_compound_of(&ds.datatype().unwrap(), 2, true);
    assert!(ds.read_u8().unwrap().is_empty());
    // A zero-element compound of the component class is what this writer emits
    // for an empty complex array, and it round-trips. It is *not* what MATLAB
    // itself emits: `Mat_VarWriteEmpty` writes the dims as data under
    // `MATLAB_empty=1`, keeping the plain class name, and libmatio reads that
    // form back as complex `int16` too. Pinning the shape we write here so the
    // divergence is visible rather than assumed — see the empty-array note in
    // the `mat` module documentation, under "How an empty value is stored".
    assert!(!ds.attrs().unwrap().contains_key("MATLAB_empty"));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// Assert a `{real, imag}` compound whose components are `size` bytes wide,
/// `real` at offset 0 and `imag` at `size`.
fn assert_compound_of(dt: &Datatype, size: u32, signed: bool) {
    let Datatype::Compound {
        size: total,
        members,
    } = dt
    else {
        panic!("expected a compound datatype, got {dt:?}");
    };
    assert_eq!(*total, size * 2, "compound is two components wide");
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].name, "real");
    assert_eq!(members[0].byte_offset, 0);
    assert_eq!(members[1].name, "imag");
    assert_eq!(members[1].byte_offset, u64::from(size));
    for m in members {
        match &m.datatype {
            Datatype::FixedPoint {
                size: component,
                byte_order,
                signed: is_signed,
                bit_offset,
                bit_precision,
            } => {
                assert_eq!(*component, size);
                assert_eq!(*byte_order, DatatypeByteOrder::LittleEndian);
                assert_eq!(*is_signed, signed);
                assert_eq!(*bit_offset, 0);
                assert_eq!(u32::from(*bit_precision), size * 8);
            }
            other => panic!("expected a fixed-point component, got {other:?}"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Capture {
    samples: Vec<ComplexI16>,
}

/// The datatype is what MATLAB reads the class off, so asserting the round
/// trip alone would accept a file this crate is happy with and MATLAB refuses.
#[test]
fn the_datatype_is_a_four_byte_int16_compound() {
    let c = Capture {
        samples: vec![ComplexI16::new(1, -1), ComplexI16::new(2, -2)],
    };
    let bytes = mat::to_bytes(&c).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("samples").unwrap();

    assert_compound_of(&ds.datatype().unwrap(), 2, true);
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("int16")
    );
}

/// Halving the payload is the entire point of the feature, so it is pinned by
/// a test rather than assumed from the datatype.
#[test]
fn the_payload_is_four_bytes_per_element_half_of_the_single_equivalent() {
    #[derive(Serialize)]
    struct AsSingle {
        samples: Vec<Complex32>,
    }

    let n = 64;
    let ints = Capture {
        samples: (0..n)
            .map(|i| ComplexI16::new(i as i16, -(i as i16)))
            .collect(),
    };
    let floats = AsSingle {
        samples: (0..n)
            .map(|i| Complex32::new(i as f32, -(i as f32)))
            .collect(),
    };

    let int_payload = payload_len(mat::to_bytes(&ints).unwrap(), "samples");
    let float_payload = payload_len(mat::to_bytes(&floats).unwrap(), "samples");

    assert_eq!(int_payload, 4 * n as usize);
    assert_eq!(float_payload, 8 * n as usize);
}

fn payload_len(bytes: Vec<u8>, name: &str) -> usize {
    File::from_bytes(bytes)
        .unwrap()
        .dataset(name)
        .unwrap()
        .read_u8()
        .unwrap()
        .len()
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Edges {
    i8s: Vec<ComplexI8>,
    i16s: Vec<ComplexI16>,
    i32s: Vec<ComplexI32>,
    i64s: Vec<ComplexI64>,
    u8s: Vec<ComplexU8>,
    u16s: Vec<ComplexU16>,
    u32s: Vec<ComplexU32>,
    u64s: Vec<ComplexU64>,
}

/// Every width's extremes, including the minima that a naive negation or a
/// signed/unsigned mix-up would mangle.
#[test]
fn extreme_component_values_survive() {
    let v = Edges {
        i8s: vec![
            ComplexI8::new(i8::MIN, i8::MAX),
            ComplexI8::new(0, -1),
            ComplexI8::new(-1, 0),
        ],
        i16s: vec![
            ComplexI16::new(i16::MIN, i16::MAX),
            ComplexI16::new(0, 0),
            ComplexI16::new(-1, 1),
            ComplexI16::new(i16::MAX, i16::MIN),
        ],
        i32s: vec![ComplexI32::new(i32::MIN, i32::MAX)],
        i64s: vec![ComplexI64::new(i64::MIN, i64::MAX)],
        u8s: vec![ComplexU8::new(0, u8::MAX)],
        u16s: vec![ComplexU16::new(u16::MAX, 0)],
        u32s: vec![ComplexU32::new(u32::MAX, 1)],
        u64s: vec![ComplexU64::new(u64::MAX, 0)],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let back: Edges = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

/// Each width reports its own MATLAB class and its own component width; a
/// shared code path that got the class from the wrong place would still round
/// trip through this crate and still be wrong in MATLAB.
#[test]
fn each_width_reports_its_own_class_and_size() {
    let v = Edges {
        i8s: vec![ComplexI8::new(1, 2)],
        i16s: vec![ComplexI16::new(1, 2)],
        i32s: vec![ComplexI32::new(1, 2)],
        i64s: vec![ComplexI64::new(1, 2)],
        u8s: vec![ComplexU8::new(1, 2)],
        u16s: vec![ComplexU16::new(1, 2)],
        u32s: vec![ComplexU32::new(1, 2)],
        u64s: vec![ComplexU64::new(1, 2)],
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    for (name, class, size, signed) in [
        ("i8s", "int8", 1u32, true),
        ("i16s", "int16", 2, true),
        ("i32s", "int32", 4, true),
        ("i64s", "int64", 8, true),
        ("u8s", "uint8", 1, false),
        ("u16s", "uint16", 2, false),
        ("u32s", "uint32", 4, false),
        ("u64s", "uint64", 8, false),
    ] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(
            ds.attrs()
                .unwrap()
                .get("MATLAB_class")
                .and_then(AttrValue::as_str),
            Some(class),
            "class of {name}"
        );
        assert_compound_of(&ds.datatype().unwrap(), size, signed);
        assert_eq!(
            ds.read_u8().unwrap().len(),
            2 * size as usize,
            "payload of {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// No silent width changes, in either direction
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct AsI16 {
    samples: Vec<ComplexI16>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct AsF64 {
    samples: Vec<Complex64>,
}

/// Reading an `int16` capture as a float complex would be lossless and is
/// still refused: the component width is part of what the file says it holds,
/// so a Rust type that quietly disagrees with it is a worse witness than an
/// error.
#[test]
fn an_integer_capture_does_not_deserialize_as_float_complex() {
    let bytes = mat::to_bytes(&AsI16 {
        samples: vec![ComplexI16::new(3, -4)],
    })
    .unwrap();
    let err = mat::from_bytes::<AsF64>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("complex double"),
        "error should name the class it wanted: {msg}"
    );
}

/// The lossy direction is refused for the same reason, and would truncate.
#[test]
fn a_float_capture_does_not_deserialize_as_integer_complex() {
    let bytes = mat::to_bytes(&AsF64 {
        samples: vec![Complex64::new(3.5, -4.5)],
    })
    .unwrap();
    let err = mat::from_bytes::<AsI16>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("complex int16"),
        "error should name the class it wanted: {msg}"
    );
}

// ---------------------------------------------------------------------------
// A file that disagrees with itself
// ---------------------------------------------------------------------------

/// Assemble a complex `double` array carrying a `MATLAB_class` attribute that
/// names some other class, the way a malformed or third-party writer can.
fn complex_f64_labelled(class: Option<&str>) -> Vec<u8> {
    let mut fb = FileBuilder::new();
    {
        let d = fb.create_dataset("samples");
        d.with_complex64_data(&[(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)])
            .with_shape(&[1, 3]);
        if let Some(class) = class {
            d.set_attr("MATLAB_class", AttrValue::AsciiString(class.to_owned()));
        }
    }
    fb.finish().unwrap()
}

/// The component width is read from `MATLAB_class` while the bytes come from
/// the compound, and nothing in the format forces the two to agree. A narrower
/// claim over wider members passes every length check and decodes to the low
/// halves of the real components, so it has to be refused on the datatype.
#[test]
fn a_class_attribute_that_contradicts_the_compound_is_refused() {
    let bytes = complex_f64_labelled(Some("int16"));
    let err = mat::from_bytes::<AsI16>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("int16") && msg.contains("f64"),
        "error should name both the claimed class and the stored one: {msg}"
    );
}

/// The same disagreement reached without a malicious attribute at all: with no
/// `MATLAB_class`, the class is guessed from the datatype, and the guess for a
/// compound is `double`. A complex `int64` array has the identical 16-byte
/// element size, so the length check cannot tell the two apart.
#[test]
fn an_unlabelled_complex_integer_array_is_not_decoded_as_double() {
    let ct = CompoundTypeBuilder::new()
        .field("real", make_i64_type())
        .field("imag", make_i64_type())
        .build()
        .unwrap();
    let mut raw = Vec::new();
    for (re, im) in [(1i64, -1i64), (2, -2)] {
        raw.extend_from_slice(&re.to_le_bytes());
        raw.extend_from_slice(&im.to_le_bytes());
    }

    let mut fb = FileBuilder::new();
    {
        let d = fb.create_dataset("samples");
        d.with_compound_data(ct, raw, 2).with_shape(&[1, 2]);
    }
    let bytes = fb.finish().unwrap();

    // The bytes are i64 pairs; the absent attribute makes the reader guess
    // `double`, which is the same element size and would decode as garbage.
    let err = mat::from_bytes::<AsF64>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("double") && msg.contains("i64"),
        "error should name the guessed class and the stored one: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/// The deflate path is shape-sensitive, and an empty array must stay
/// unfiltered (a filtered zero-element dataset is what MATLAB chokes on).
#[test]
fn compression_round_trips_and_leaves_the_empty_array_alone() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Both {
        samples: Vec<ComplexI16>,
        empty: Matrix<ComplexI16>,
    }

    let v = Both {
        // Highly compressible: a repeating ramp.
        samples: (0..4096)
            .map(|i| ComplexI16::new((i % 8) as i16, -((i % 8) as i16)))
            .collect(),
        empty: Matrix::from_row_major(0, 0, Vec::new()),
    };

    // Compression needs chunked storage, whose chunk indices need the HDF5 1.10
    // format, so both sides of the comparison ask for it. The MAT default is the
    // 1.8 format (MATLAB linked HDF5 1.8.12 before R2021b), which refuses compression by
    // name — comparing a 1.10 file against a 1.8 one would also be measuring the
    // superblock.
    let mut plain_opts = Options::default();
    plain_opts.libver = LibVer::V110;
    let plain = mat::to_bytes_with_options(&v, &plain_opts).unwrap();
    let mut opts = plain_opts.clone();
    opts.compression = Compression::Deflate {
        level: 6,
        shuffle: false,
    };
    let deflated = mat::to_bytes_with_options(&v, &opts).unwrap();

    assert!(
        deflated.len() < plain.len(),
        "deflate should shrink a repeating ramp: {} vs {}",
        deflated.len(),
        plain.len()
    );
    let back: Both = mat::from_bytes(&deflated).unwrap();
    assert_eq!(back, v);
}

/// A complex vector is a vector, so it takes the configured orientation like
/// every other one. It used to be written as a MATLAB row regardless, which
/// made it the only kind of 1-D array whose shape contradicted both the option
/// and the default write path.
#[test]
fn a_complex_vector_takes_the_configured_orientation_like_a_real_one() {
    #[derive(Serialize)]
    struct Both {
        reals: Vec<i16>,
        cplx: Vec<ComplexI16>,
    }

    let v = Both {
        reals: vec![1, 2, 3],
        cplx: vec![
            ComplexI16::new(1, -1),
            ComplexI16::new(2, -2),
            ComplexI16::new(3, -3),
        ],
    };
    let shapes = |opts: &Options| {
        let file = File::from_bytes(mat::to_bytes_with_options(&v, opts).unwrap()).unwrap();
        (
            file.dataset("reals").unwrap().shape().unwrap(),
            file.dataset("cplx").unwrap().shape().unwrap(),
        )
    };

    let (reals, cplx) = shapes(&Options::default());
    assert_eq!(reals, cplx, "column-vector default");

    let mut row = Options::default();
    row.one_dimensional_mode = OneDimensionalMode::RowVector;
    let (reals_row, cplx_row) = shapes(&row);
    assert_eq!(reals_row, cplx_row, "row-vector mode");
    assert_ne!(reals, reals_row, "the option has to move the shape at all");

    // And the default options path agrees with `to_bytes`, which has no
    // options to consult.
    let plain = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    assert_eq!(plain.dataset("cplx").unwrap().shape().unwrap(), cplx);
}

// ---------------------------------------------------------------------------
// The mid-level builder
// ---------------------------------------------------------------------------

#[test]
fn the_builder_writes_complex_integers_in_every_scope() {
    use hdf5_pure::mat::MatBuilder;

    let mut mb = MatBuilder::new(Options::default());
    mb.write_complex_i16("top", &[1, 2], &[(1, -1), (2, -2)])
        .unwrap();
    mb.struct_("nested", |sw| {
        sw.write_complex_i32("field", &[1, 1], &[(7, -7)])?;
        Ok(())
    })
    .unwrap();
    mb.cell("cell", &[1, 1], |cw| {
        cw.push_complex_u8(&[1, 2], &[(1, 2), (3, 4)])?;
        Ok(())
    })
    .unwrap();
    let bytes = mb.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    assert_compound_of(&file.dataset("top").unwrap().datatype().unwrap(), 2, true);
    assert_compound_of(
        &file.dataset("nested/field").unwrap().datatype().unwrap(),
        4,
        true,
    );
    assert_eq!(
        file.dataset("nested/field")
            .unwrap()
            .attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("int32")
    );
    // The cell element lives under `#refs#`; the first (and only) one here.
    let refs = file.group("#refs#").unwrap();
    let name = refs.datasets().unwrap().into_iter().next().unwrap();
    let ds = refs.dataset(&name).unwrap();
    assert_compound_of(&ds.datatype().unwrap(), 1, false);
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("uint8")
    );
}

// ---------------------------------------------------------------------------
// Every arm shape, every integer class
// ---------------------------------------------------------------------------

/// A complex value reaches the writer through one of five distinct arms —
/// scalar, 1-D vector, 2-D matrix, a matrix built from equal-length rows, and a
/// cell element — and each one is a separate dispatch site. The per-class risk
/// is small because the dispatch is macro-generated from one list, but the
/// per-*arm* risk is not, so every arm is exercised for every integer class
/// rather than for `int16` alone. The float classes take the same arms and are
/// covered by the pre-existing `Complex64`/`Complex32` round trips.
macro_rules! arm_shapes_for {
    ($($name:ident: $ty:ident, $class:literal, $size:literal, $signed:literal;)*) => {
        $(
            #[test]
            fn $name() {
                #[derive(Serialize, Deserialize, Debug, PartialEq)]
                struct Arms {
                    scalar: $ty,
                    vector: Vec<$ty>,
                    matrix: Matrix<$ty>,
                    rows: Vec<Vec<$ty>>,
                    cells: Vec<Matrix<$ty>>,
                    empty: Matrix<$ty>,
                }

                // Small non-negative values, so one table serves the signed and
                // unsigned classes alike; ranges are pinned separately by
                // `extreme_component_values_survive`.
                let e = |re: i64, im: i64| $ty::new(re as _, im as _);
                let v = Arms {
                    scalar: e(7, 9),
                    vector: vec![e(1, 2), e(3, 4), e(5, 6)],
                    matrix: Matrix::from_row_major(
                        2,
                        3,
                        vec![e(1, 1), e(2, 2), e(3, 3), e(4, 4), e(5, 5), e(6, 6)],
                    ),
                    rows: vec![vec![e(1, 2), e(3, 4)], vec![e(5, 6), e(7, 8)]],
                    cells: vec![
                        Matrix::from_row_major(2, 2, vec![e(1, 1), e(2, 2), e(3, 3), e(4, 4)]),
                        Matrix::from_row_major(1, 2, vec![e(5, 5), e(6, 6)]),
                    ],
                    empty: Matrix::from_row_major(0, 0, Vec::new()),
                };

                // `to_bytes` and `to_bytes_with_options` are two separate emit
                // paths with their own copy of every arm — the orientation
                // defect this PR fixes lived in one and not the other — so
                // each arm is exercised through both.
                let written = [
                    ("to_bytes", mat::to_bytes(&v).unwrap()),
                    (
                        "to_bytes_with_options",
                        mat::to_bytes_with_options(&v, &Options::default()).unwrap(),
                    ),
                ];

                for (path, bytes) in written {
                    let back: Arms = mat::from_bytes(&bytes).unwrap();
                    assert_eq!(back, v, "{path}");

                    // The round trip alone would accept a compound this crate
                    // is happy with and MATLAB refuses, so pin the stored
                    // layout of every arm that writes one directly.
                    let file = File::from_bytes(bytes).unwrap();
                    for name in ["scalar", "vector", "matrix", "rows", "empty"] {
                        let ds = file.dataset(name).unwrap();
                        assert_compound_of(&ds.datatype().unwrap(), $size, $signed);
                        assert_eq!(
                            ds.attrs()
                                .unwrap()
                                .get("MATLAB_class")
                                .and_then(AttrValue::as_str),
                            Some($class),
                            "{path}: {name} should carry the component class"
                        );
                    }

                    // The cell's elements live under `#refs#`, and each is a
                    // complex matrix — the one arm `push_complex` reaches that
                    // no other field here exercises.
                    let refs = file.group("#refs#").unwrap();
                    let names = refs.datasets().unwrap();
                    assert_eq!(names.len(), 2, "{path}: one dataset per cell element");
                    for name in names {
                        let ds = refs.dataset(&name).unwrap();
                        assert_compound_of(&ds.datatype().unwrap(), $size, $signed);
                        assert_eq!(
                            ds.attrs()
                                .unwrap()
                                .get("MATLAB_class")
                                .and_then(AttrValue::as_str),
                            Some($class),
                            "{path}: cell element"
                        );
                    }
                }
            }
        )*
    };
}

arm_shapes_for! {
    every_arm_shape_survives_for_int8: ComplexI8, "int8", 1, true;
    every_arm_shape_survives_for_int16: ComplexI16, "int16", 2, true;
    every_arm_shape_survives_for_int32: ComplexI32, "int32", 4, true;
    every_arm_shape_survives_for_int64: ComplexI64, "int64", 8, true;
    every_arm_shape_survives_for_uint8: ComplexU8, "uint8", 1, false;
    every_arm_shape_survives_for_uint16: ComplexU16, "uint16", 2, false;
    every_arm_shape_survives_for_uint32: ComplexU32, "uint32", 4, false;
    every_arm_shape_survives_for_uint64: ComplexU64, "uint64", 8, false;
}

// ---------------------------------------------------------------------------
// Promoting a real value to complex
// ---------------------------------------------------------------------------

/// A real array of the component class reads as complex with a zero imaginary
/// part — a file written by something that dropped an all-real capture's
/// `imag` field still loads into a complex target. This generalized from the
/// two float classes to all ten, so it is checked at a class that did not
/// exist on the old path.
#[test]
fn a_real_scalar_of_the_same_class_reads_as_complex_with_zero_imag() {
    #[derive(Serialize)]
    struct Real {
        a: i16,
        b: u64,
    }
    #[derive(Deserialize, Debug, PartialEq)]
    struct Cplx {
        a: ComplexI16,
        b: ComplexU64,
    }

    let bytes = mat::to_bytes(&Real { a: -42, b: 42 }).unwrap();
    let back: Cplx = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back.a, ComplexI16::new(-42, 0));
    assert_eq!(back.b, ComplexU64::new(42, 0));
}

/// The promotion is class-exact like every other complex read: a real `double`
/// does not become a complex `int16` just because the imaginary part it is
/// missing would have been zero.
#[test]
fn a_real_scalar_of_another_class_is_not_promoted() {
    #[derive(Serialize)]
    struct Real {
        x: f64,
    }
    #[derive(Deserialize, Debug)]
    struct Cplx {
        // Never read: the point is that deserializing into it fails.
        #[allow(dead_code)]
        x: ComplexI16,
    }

    let bytes = mat::to_bytes(&Real { x: 42.0 }).unwrap();
    let err = mat::from_bytes::<Cplx>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("complex int16"),
        "error should name the class it wanted: {msg}"
    );
}
