#![cfg(feature = "serde")]
//! Roundtrip tests for `.mat` v7.3 serde support.
//!
//! These tests exercise the full serializer path and verify the produced
//! bytes can be read back via the hdf5-pure reader (and will eventually be
//! deserialized back into the original struct).

use hdf5_pure::mat::{self, Complex32, Complex64, Matrix};
use hdf5_pure::{AttrValue, File};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sanity checks on the raw output — userblock shape, HDF5 structure, MATLAB
// class attributes. These run before the deserializer exists.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SimpleRoot {
    trial: u32,
    temperature: f64,
    name: String,
}

#[test]
fn userblock_signature_present() {
    let v = SimpleRoot {
        trial: 1,
        temperature: 23.5,
        name: "alpha".into(),
    };
    let bytes = mat::to_bytes(&v).unwrap();
    assert!(bytes.len() > 512, "file must be at least userblock + HDF5");
    assert_eq!(&bytes[..6], b"MATLAB");
    assert_eq!(&bytes[124..128], &[0x00, 0x02, b'I', b'M']);
}

#[test]
fn serialize_top_level_struct_produces_three_variables() {
    let v = SimpleRoot {
        trial: 7,
        temperature: 300.0,
        name: "beta".into(),
    };
    let bytes = mat::to_bytes(&v).unwrap();

    let file = File::from_bytes(bytes).unwrap();

    // trial: uint32 scalar
    let trial = file.dataset("trial").unwrap();
    assert_eq!(trial.shape().unwrap(), vec![1, 1]);
    assert_eq!(trial.read_u32().unwrap(), vec![7]);
    let trial_attrs = trial.attrs().unwrap();
    assert_eq!(
        trial_attrs.get("MATLAB_class").and_then(AttrValue::as_str),
        Some("uint32")
    );

    // temperature: double scalar
    let temp = file.dataset("temperature").unwrap();
    assert_eq!(temp.shape().unwrap(), vec![1, 1]);
    assert_eq!(temp.read_f64().unwrap(), vec![300.0]);
    assert_eq!(
        temp.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("double")
    );

    // name: char [4, 1] HDF5 → MATLAB row [1, 4] (UTF-16 code units of "beta")
    let name = file.dataset("name").unwrap();
    assert_eq!(name.shape().unwrap(), vec![4, 1]);
    assert_eq!(
        name.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("char")
    );
    let units = name.read_u16().unwrap();
    let s = String::from_utf16(&units).unwrap();
    assert_eq!(s, "beta");
}

#[test]
fn root_must_be_struct_errors_for_primitives() {
    let r = mat::to_bytes(&42u32);
    assert!(matches!(r, Err(mat::MatError::RootMustBeStruct)));
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Vecs {
    xs: Vec<f64>,
    ns: Vec<i32>,
    bytes: Vec<u8>,
}

#[test]
fn vec_fields_produce_row_vectors() {
    let v = Vecs {
        xs: vec![1.0, 2.0, 3.0, 4.0],
        ns: vec![-1, 0, 1],
        bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    let xs = file.dataset("xs").unwrap();
    assert_eq!(xs.shape().unwrap(), vec![1, 4]);
    assert_eq!(xs.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let ns = file.dataset("ns").unwrap();
    assert_eq!(ns.shape().unwrap(), vec![1, 3]);
    assert_eq!(ns.read_i32().unwrap(), vec![-1, 0, 1]);

    let bytes = file.dataset("bytes").unwrap();
    assert_eq!(bytes.shape().unwrap(), vec![1, 4]);
    assert_eq!(bytes.read_u8().unwrap(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithMatrix {
    m: Matrix<f64>,
}

#[test]
fn matrix_field_has_transposed_shape() {
    // 2-row, 3-col Rust matrix
    //   [[1 2 3]
    //    [4 5 6]]
    let v = WithMatrix {
        m: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    let m = file.dataset("m").unwrap();
    // HDF5 shape is [cols, rows] so MATLAB sees [rows, cols].
    assert_eq!(m.shape().unwrap(), vec![3, 2]);
    // Stored column-major: col0=[1,4], col1=[2,5], col2=[3,6]
    assert_eq!(m.read_f64().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithNested {
    x: f64,
    inner: Inner,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Inner {
    y: i32,
    z: Vec<f64>,
}

#[test]
fn nested_struct_becomes_group() {
    let v = WithNested {
        x: 1.5,
        inner: Inner {
            y: 42,
            z: vec![10.0, 20.0],
        },
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    // Outer fields
    assert_eq!(file.dataset("x").unwrap().read_f64().unwrap(), vec![1.5]);

    // Nested group + MATLAB_class + MATLAB_fields
    let inner = file.group("inner").unwrap();
    let attrs = inner.attrs().unwrap();
    assert_eq!(
        attrs.get("MATLAB_class").and_then(AttrValue::as_str),
        Some("struct")
    );
    // `MATLAB_fields` is written as a variable-length ASCII array and reads
    // back as one; `as_strings` spans that and the fixed-width shapes.
    let fields = attrs
        .get("MATLAB_fields")
        .and_then(AttrValue::as_strings)
        .expect("MATLAB_fields must hold strings");
    assert_eq!(fields, vec!["y", "z"]);

    // Nested datasets exist
    let y = file.dataset("inner/y").unwrap();
    assert_eq!(y.read_i32().unwrap(), vec![42]);
    let z = file.dataset("inner/z").unwrap();
    assert_eq!(z.shape().unwrap(), vec![1, 2]);
    assert_eq!(z.read_f64().unwrap(), vec![10.0, 20.0]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithOption {
    required: f64,
    #[serde(default)]
    maybe: Option<String>,
}

/// Assert `name` is present and carries MATLAB's empty-struct-array marker:
/// struct-classed, flagged empty, and holding the dimension vector `[0, 0]` as
/// its *payload*. Checking the class alone would pass for any struct at all,
/// empty or not.
///
/// The payload is where the dims live under the default
/// `EmptyMarkerEncoding::DataAsDims`, which is what MATLAB itself writes: every
/// empty in `tests/data/matlab` is a two-element `uint64` dataset holding
/// its own dims, including the `[1 1]` and `[1 0]` ones a zero-element dataset
/// could not express at all.
fn assert_empty_struct_array(file: &File, name: &str) {
    let ds = file
        .dataset(name)
        .unwrap_or_else(|e| panic!("`{name}` must be present, got: {e}"));
    let attrs = ds.attrs().unwrap();
    assert_eq!(
        attrs.get("MATLAB_class").and_then(AttrValue::as_str),
        Some("struct"),
        "`{name}` must be a struct",
    );
    assert!(
        matches!(attrs.get("MATLAB_empty"), Some(AttrValue::U32(1))),
        "`{name}` must be flagged empty, got {:?}",
        attrs.get("MATLAB_empty"),
    );
    assert_eq!(
        ds.shape().unwrap(),
        vec![2],
        "`{name}` holds a two-element dimension vector",
    );
    assert_eq!(
        ds.read_u64().unwrap(),
        vec![0, 0],
        "`{name}`'s payload must be the dims [0, 0]",
    );
}

#[test]
fn option_none_becomes_an_empty_struct_array() {
    // The default `NullPolicy::EmptyStructArray`: the field stays present, so
    // MATLAB code can reference it unconditionally.
    let v = WithOption {
        required: 1.0,
        maybe: None,
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    assert!(file.dataset("required").is_ok());
    assert_empty_struct_array(&file, "maybe");
}

/// The same value through the other emitter. `to_bytes` writes through
/// `ser::emit` and `to_bytes_with_options` through `MatBuilder`, so a default
/// `Options` has to reach the same file; the two carried different empty
/// markers until 0.30, which nothing caught while `None` still dropped the
/// field and never reached one.
#[test]
fn option_none_agrees_across_both_emitters() {
    let v = WithOption {
        required: 1.0,
        maybe: None,
    };
    let no_options = mat::to_bytes(&v).unwrap();
    let with_options = mat::to_bytes_with_options(&v, &mat::Options::default()).unwrap();
    assert_eq!(no_options, with_options);

    let file = File::from_bytes(with_options).unwrap();
    assert!(file.dataset("required").is_ok());
    assert_empty_struct_array(&file, "maybe");
}

#[test]
fn option_none_is_omitted_under_the_omit_policy() {
    // `NullPolicy::Omit` is the pre-0.30 serde behavior, now opt-in.
    let v = WithOption {
        required: 1.0,
        maybe: None,
    };
    let mut options = mat::Options::default();
    options.null_policy = mat::NullPolicy::Omit;
    let file = File::from_bytes(mat::to_bytes_with_options(&v, &options).unwrap()).unwrap();
    assert!(file.dataset("required").is_ok());
    assert!(file.dataset("maybe").is_err());
}

#[test]
fn option_none_is_rejected_under_the_error_policy() {
    let v = WithOption {
        required: 1.0,
        maybe: None,
    };
    let mut options = mat::Options::default();
    options.null_policy = mat::NullPolicy::Error;
    assert!(mat::to_bytes_with_options(&v, &options).is_err());
}

/// The root names no slot, so a null there is an empty variable namespace, not a
/// null value in one. Every policy that can express that writes the same
/// zero-variable file, and it is the same file three unambiguously legitimate
/// requests already produce, so refusing it would fork the crate against itself.
///
/// Locks both halves: that the silence is deliberate, and that `Error` still
/// fires here. `Error` reaching every other slot but this one made the policy
/// fail at its only purpose.
#[test]
fn a_root_null_writes_the_same_empty_workspace_every_legitimate_route_does() {
    use std::collections::BTreeMap;

    #[derive(Serialize)]
    struct Fieldless {}

    let none: Option<f64> = None;
    let mut omit = mat::Options::default();
    omit.null_policy = mat::NullPolicy::Omit;

    // The two policies that can express an empty namespace, over the three ways
    // to spell a root null.
    let baseline = mat::to_bytes_with_options(&none, &omit).unwrap();
    for (label, bytes) in [
        ("root () under Omit", mat::to_bytes_with_options(&(), &omit)),
        (
            "root Null under Omit",
            mat::to_bytes_with_options(&serde_json::Value::Null, &omit),
        ),
        (
            "root None under the default",
            mat::to_bytes_with_options(&none, &mat::Options::default()),
        ),
        (
            "root () under the default",
            mat::to_bytes_with_options(&(), &mat::Options::default()),
        ),
        // The requests that make refusing indefensible: by the time the emitter
        // sees an empty variable list it cannot tell these from a root null.
        (
            "empty root map",
            mat::to_bytes_with_options(&BTreeMap::<String, f64>::new(), &mat::Options::default()),
        ),
        (
            "fieldless struct",
            mat::to_bytes_with_options(&Fieldless {}, &mat::Options::default()),
        ),
    ] {
        assert_eq!(
            bytes.unwrap(),
            baseline,
            "{label} must write the same empty workspace"
        );
    }

    // It is a real file, not a degenerate blob: it opens and has no variables.
    let file = File::from_bytes(baseline).unwrap();
    assert!(
        file.root().datasets().unwrap().is_empty(),
        "the empty workspace must contain no variables"
    );

    // `Error` is the one policy expressible at the root, and it must fire.
    let mut error = mat::Options::default();
    error.null_policy = mat::NullPolicy::Error;
    assert!(mat::to_bytes_with_options(&none, &error).is_err());
    assert!(mat::to_bytes_with_options(&(), &error).is_err());
    assert!(mat::to_bytes_with_options(&serde_json::Value::Null, &error).is_err());
    // ...without catching the legitimate empty-workspace requests.
    assert!(mat::to_bytes_with_options(&BTreeMap::<String, f64>::new(), &error).is_ok());
    assert!(mat::to_bytes_with_options(&Fieldless {}, &error).is_ok());
}

/// `Matrix::data` is the sentinel's own payload, not a sequence the caller
/// wrote, so `empty_sequence_policy` must not reach it. It did, and an empty
/// matrix then lowered to a cell array that the sentinel handler could not
/// recover the element class from, making every empty `Matrix<T>` unserializable
/// under a policy that has nothing to do with matrices.
#[test]
fn an_empty_matrix_serializes_under_every_empty_sequence_policy() {
    #[derive(Serialize)]
    struct WithMatrix {
        m: mat::Matrix<f64>,
    }
    #[derive(Serialize)]
    struct WithComplexMatrix {
        m: mat::Matrix<mat::Complex64>,
    }
    #[derive(Serialize)]
    struct WithVec {
        v: Vec<f64>,
    }

    let mut cell = mat::Options::default();
    cell.empty_sequence_policy = mat::EmptySequencePolicy::Cell;

    for (rows, cols) in [(0usize, 0usize), (0, 5), (5, 0)] {
        let under_cell = mat::to_bytes_with_options(
            &WithMatrix {
                m: mat::Matrix::from_row_major(rows, cols, vec![]),
            },
            &cell,
        )
        .unwrap_or_else(|e| panic!("empty {rows}x{cols} matrix under Cell: {e}"));
        let under_default = mat::to_bytes_with_options(
            &WithMatrix {
                m: mat::Matrix::from_row_major(rows, cols, vec![]),
            },
            &mat::Options::default(),
        )
        .unwrap();
        // The policy is pinned for this field, so it cannot change the output.
        assert_eq!(
            under_cell, under_default,
            "empty {rows}x{cols} matrix must not vary with empty_sequence_policy"
        );
    }

    // The complex sentinel recovers its class from the same field.
    mat::to_bytes_with_options(
        &WithComplexMatrix {
            m: mat::Matrix::from_row_major(0, 0, vec![]),
        },
        &cell,
    )
    .expect("empty complex matrix under Cell");

    // A sequence the caller actually wrote still honors the policy, which is the
    // distinction the fix has to preserve.
    let seq_cell = mat::to_bytes_with_options(&WithVec { v: vec![] }, &cell).unwrap();
    let seq_default =
        mat::to_bytes_with_options(&WithVec { v: vec![] }, &mat::Options::default()).unwrap();
    assert_ne!(
        seq_cell, seq_default,
        "a user's empty Vec must still follow empty_sequence_policy"
    );
}

/// `Options` gains fields, so one persisted by an older version has to keep
/// loading. `#[serde(default)]` is on the struct rather than on the fields added
/// in 0.30, so this holds for future additions too without anyone remembering.
#[test]
fn an_options_persisted_by_an_older_version_still_deserializes() {
    // The exact field set 0.29 wrote: the two fields added in 0.30 are absent.
    let json_0_29 = r#"{
        "string_class": "Char",
        "compression": "None",
        "invalid_name_policy": "Error",
        "null_policy": "EmptyStructArray",
        "unsupported_policy": "Error",
        "one_dimensional_mode": "ColumnVector",
        "row_major_policy": "ReorderToColumnMajor",
        "empty_marker_encoding": "ZeroElement"
    }"#;
    let opts: mat::Options = serde_json::from_str(json_0_29).expect("0.29 Options must still load");

    // What it recorded is what it gets back, including a value that has since
    // stopped being the default — that is the whole point of persisting options.
    assert_eq!(
        opts.empty_marker_encoding,
        mat::EmptyMarkerEncoding::ZeroElement
    );
    assert_ne!(
        opts.empty_marker_encoding,
        mat::Options::default().empty_marker_encoding,
        "this assertion is only meaningful while the recorded value differs from \
         the current default; pick another moved field if that changes"
    );
    // Fields it never heard of take the current default rather than failing.
    assert_eq!(opts.libver, mat::Options::default().libver);
    assert_eq!(
        opts.unit_variant_encoding,
        mat::Options::default().unit_variant_encoding
    );

    // An empty document is the degenerate case of the same property, and there
    // every field defaults.
    let all_defaulted: mat::Options = serde_json::from_str("{}").unwrap();
    assert_eq!(all_defaulted, mat::Options::default());
}

/// What a `struct([])` field reads back into. This is the migration hazard in
/// 0.30: the field is now *present*, so `#[serde(default)]` is never consulted
/// and a non-`Option` target fails where it used to succeed. Pinned here so the
/// documented leniency set and the real one cannot drift.
#[test]
fn an_empty_struct_array_field_reads_back_only_into_these_targets() {
    #[derive(Serialize)]
    struct Writer {
        v: Option<f64>,
    }
    let bytes = mat::to_bytes(&Writer { v: None }).unwrap();

    macro_rules! reads_as {
        ($ty:ty, $expected:expr) => {{
            #[derive(Deserialize, Debug, PartialEq)]
            struct R {
                v: $ty,
            }
            let r: R = mat::from_bytes(&bytes).expect("must deserialize");
            assert_eq!(r.v, $expected);
        }};
    }
    reads_as!(Option<f64>, None);
    reads_as!(Vec<f64>, Vec::<f64>::new());
    reads_as!(serde_json::Value, serde_json::Value::Null);
    reads_as!((), ());

    macro_rules! rejects {
        ($ty:ty) => {{
            #[derive(Deserialize, Debug)]
            struct R {
                #[allow(dead_code)]
                v: $ty,
            }
            assert!(
                mat::from_bytes::<R>(&bytes).is_err(),
                "{} must not silently accept struct([])",
                stringify!($ty)
            );
        }};
    }
    rejects!(f64);
    rejects!(u32);
    rejects!(bool);
    rejects!(String);

    // `#[serde(default)]` does not rescue a scalar: the field is present, not
    // missing, so serde never reaches for the default. This is the hazard.
    #[derive(Deserialize, Debug)]
    struct WithDefault {
        #[serde(default)]
        #[allow(dead_code)]
        v: u32,
    }
    assert!(
        mat::from_bytes::<WithDefault>(&bytes).is_err(),
        "#[serde(default)] must not appear to rescue a present null field"
    );
}

#[test]
fn option_some_serializes_underlying() {
    let v = WithOption {
        required: 1.0,
        maybe: Some("hello".into()),
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    let ds = file.dataset("maybe").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![5, 1]);
}

// A Rust unit `()` — the shape `serde_json::Value::Null` serializes as — goes
// through `null_policy` exactly like `Option::None`, instead of aborting the
// whole encode. See the `serialize_unit` handler in `mat/ser/value_ser.rs`.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithNullMeta {
    id: u32,
    #[serde(default)]
    meta: serde_json::Value,
}

#[test]
fn null_field_lowers_like_none() {
    let v = WithNullMeta {
        id: 7,
        meta: serde_json::Value::Null,
    };
    // Must not abort the encode the way it used to with
    // `UnsupportedType("() / unit")`.
    let bytes = mat::to_bytes(&v).expect("Null field must not abort the encode");
    let file = File::from_bytes(bytes.clone()).unwrap();
    assert!(file.dataset("id").is_ok());
    assert_empty_struct_array(&file, "meta");
    // `struct([])` reads back as a unit, so `Value::Null` survives the trip.
    let back: WithNullMeta = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

#[test]
fn omitted_unit_field_without_default_fails_to_deserialize() {
    // Documents the read-back contract for `NullPolicy::Omit`: a non-`Option`
    // field that serialized away must carry `#[serde(default)]` to
    // reconstruct. The encode still succeeds; only the deserialize of the
    // absent field fails. Under the default policy the field is present, so
    // this hazard is specific to opting into `Omit`.
    #[derive(Serialize)]
    struct Writer {
        id: u32,
        meta: serde_json::Value, // Null -> dropped
    }
    #[derive(Deserialize, Debug)]
    struct ReaderNoDefault {
        #[allow(dead_code)]
        id: u32,
        #[allow(dead_code)]
        meta: serde_json::Value, // no #[serde(default)]
    }
    let mut options = mat::Options::default();
    options.null_policy = mat::NullPolicy::Omit;
    let bytes = mat::to_bytes_with_options(
        &Writer {
            id: 1,
            meta: serde_json::Value::Null,
        },
        &options,
    )
    .unwrap();
    let err = mat::from_bytes::<ReaderNoDefault>(&bytes).unwrap_err();
    assert!(
        err.to_string().contains("meta"),
        "expected a missing-field error naming `meta`, got: {err}"
    );
}

#[test]
fn some_null_lowers_like_none() {
    // `Some(Value::Null)` forwards through `serialize_some` to the inner
    // `serialize_unit`, so it takes the same path as a bare `None`; wrapping in
    // `Option` is not a way to sidestep the policy.
    #[derive(Serialize)]
    struct S {
        id: u32,
        meta: Option<serde_json::Value>,
    }
    let v = S {
        id: 3,
        meta: Some(serde_json::Value::Null),
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    assert!(file.dataset("id").is_ok());
    assert_empty_struct_array(&file, "meta");
}

#[test]
fn unit_struct_field_lowers_like_none() {
    #[derive(Serialize)]
    struct Marker;
    #[derive(Serialize)]
    struct S {
        id: u32,
        marker: Marker,
    }
    let file = File::from_bytes(
        mat::to_bytes(&S {
            id: 5,
            marker: Marker,
        })
        .unwrap(),
    )
    .unwrap();
    assert!(file.dataset("id").is_ok());
    assert_empty_struct_array(&file, "marker");
}

// A flat numeric or complex sequence is collected packed, one element wide,
// rather than one 56-byte `MatValue` per element. These pin the point where it
// stops being packed: an element that breaks the pattern must spill the run
// collected so far back into individual values without losing or reordering
// any of it.

#[test]
fn sequence_that_changes_scalar_type_midway_keeps_every_element() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S {
        items: Vec<serde_json::Value>,
    }
    let v = S {
        items: vec![
            serde_json::json!(1.0),
            serde_json::json!(2.0),
            serde_json::json!(3.0),
            serde_json::json!("four"),
            serde_json::json!(5.0),
        ],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let ds = file.dataset("items").unwrap();
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("cell"),
        "a run broken by a string lowers to a cell"
    );
    let back: S = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v, "the spilled run must survive element for element");
}

#[test]
fn sequence_that_changes_numeric_width_midway_keeps_every_element() {
    // Same spill, triggered by a tag change rather than a kind change: an
    // integer after a run of floats cannot extend the packed vector.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S {
        items: Vec<serde_json::Value>,
    }
    let v = S {
        items: vec![
            serde_json::json!(1.5),
            serde_json::json!(2.5),
            serde_json::json!(7u64),
        ],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let back: S = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

#[test]
fn long_flat_numeric_sequence_still_lowers_to_one_array() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S {
        items: Vec<f64>,
    }
    let v = S {
        items: (0..10_000).map(|i| i as f64).collect(),
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let ds = file.dataset("items").unwrap();
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("double"),
        "a flat numeric run stays one array, not a cell of scalars"
    );
    assert_eq!(ds.shape().unwrap(), vec![1, 10_000]);
    let back: S = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Width {
    #[serde(rename = "i16")]
    I16,
    #[serde(rename = "complex_f32")]
    ComplexF32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithWidth {
    width: Width,
}

#[test]
fn unit_variant_defaults_to_its_name() {
    let file = File::from_bytes(
        mat::to_bytes(&WithWidth {
            width: Width::ComplexF32,
        })
        .unwrap(),
    )
    .unwrap();
    let ds = file.dataset("width").unwrap();
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("char")
    );
    let s = String::from_utf16(&ds.read_u16().unwrap()).unwrap();
    assert_eq!(s, "complex_f32");
}

fn index_options() -> mat::Options {
    let mut options = mat::Options::default();
    options.unit_variant_encoding = mat::UnitVariantEncoding::Index;
    options
}

#[test]
fn unit_variant_can_be_written_as_its_index() {
    let bytes = mat::to_bytes_with_options(
        &WithWidth {
            width: Width::ComplexF32,
        },
        &index_options(),
    )
    .unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("width").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![1, 1]);
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("uint32")
    );
    assert_eq!(ds.read_u32().unwrap(), vec![1]);
}

#[test]
fn unit_variant_written_as_an_index_reads_back() {
    let v = WithWidth {
        width: Width::ComplexF32,
    };
    let bytes = mat::to_bytes_with_options(&v, &index_options()).unwrap();
    let back: WithWidth = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

#[test]
fn unit_variant_written_as_a_name_reads_back() {
    let v = WithWidth { width: Width::I16 };
    let back: WithWidth = mat::from_bytes(&mat::to_bytes(&v).unwrap()).unwrap();
    assert_eq!(back, v);
}

/// The reader takes an index from whatever numeric class the file happens to
/// carry, since a file edited in MATLAB brings `double` rather than `uint32`.
#[test]
fn unit_variant_index_reads_from_a_double() {
    let mut mb = mat::MatBuilder::new(mat::Options::default());
    mb.write_scalar_f64("width", 1.0).unwrap();
    let back: WithWidth = mat::from_bytes(&mb.finish().unwrap()).unwrap();
    assert_eq!(back.width, Width::ComplexF32);
}

/// An index the enum has no variant for is serde's error to report, not a
/// silently wrong variant.
#[test]
fn unit_variant_index_out_of_range_is_an_error() {
    let mut mb = mat::MatBuilder::new(mat::Options::default());
    mb.write_scalar_u32("width", 99).unwrap();
    let err = mat::from_bytes::<WithWidth>(&mb.finish().unwrap()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("99"), "unexpected error: {msg}");
}

#[derive(Serialize)]
struct WithEmptySeq {
    items: Vec<Nested>,
}

#[derive(Serialize)]
struct Nested {
    #[allow(dead_code)]
    a: u32,
}

#[test]
fn empty_sequence_defaults_to_an_empty_double_array() {
    let file =
        File::from_bytes(mat::to_bytes(&WithEmptySeq { items: Vec::new() }).unwrap()).unwrap();
    let ds = file.dataset("items").unwrap();
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("double")
    );
}

#[test]
fn empty_sequence_can_be_written_as_an_empty_cell() {
    // An empty `serialize_seq` carries no element type; `Cell` is the right
    // answer when the sequence would have held structs.
    let mut options = mat::Options::default();
    options.empty_sequence_policy = mat::EmptySequencePolicy::Cell;
    let file = File::from_bytes(
        mat::to_bytes_with_options(&WithEmptySeq { items: Vec::new() }, &options).unwrap(),
    )
    .unwrap();
    let ds = file.dataset("items").unwrap();
    assert_eq!(
        ds.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("cell")
    );
}

#[test]
fn vec_of_unit_roundtrips() {
    // A unit inside a sequence lowers to a `struct([])` cell slot on write; the
    // reader accepts `struct([])` back as `()` so the degenerate `Vec<()>` still
    // round-trips rather than failing on read.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S {
        units: Vec<()>,
    }
    let v = S {
        units: vec![(), (), ()],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let back: S = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

#[test]
fn vec_of_values_with_null_roundtrips() {
    // The realistic sequence case: a `serde_json::Value` array carrying a `Null`
    // among numbers round-trips element-for-element.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct S {
        items: Vec<serde_json::Value>,
    }
    let v = S {
        items: vec![
            serde_json::Value::from(1),
            serde_json::Value::Null,
            serde_json::Value::from(3),
        ],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let back: S = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, v);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithComplex {
    z: Complex64,
    zs: Vec<Complex64>,
}

#[test]
fn complex_scalar_and_vector() {
    let v = WithComplex {
        z: Complex64::new(1.0, -2.0),
        zs: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    let z = file.dataset("z").unwrap();
    assert_eq!(z.shape().unwrap(), vec![1, 1]);
    // The compound dataset is double class per MATLAB convention.
    assert_eq!(
        z.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("double")
    );

    let zs = file.dataset("zs").unwrap();
    assert_eq!(zs.shape().unwrap(), vec![1, 2]);
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Flag {
    On,
    Off,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithEnum {
    state: Flag,
}

#[test]
fn unit_enum_variant_becomes_char_string() {
    let v = WithEnum { state: Flag::On };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();
    let ds = file.dataset("state").unwrap();
    // MATLAB row-vector string: HDF5 shape [N, 1].
    assert_eq!(ds.shape().unwrap(), vec![2, 1]);
    let units = ds.read_u16().unwrap();
    assert_eq!(String::from_utf16(&units).unwrap(), "On");
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithBool {
    flag: bool,
    flags: Vec<bool>,
}

#[test]
fn bool_fields_are_logical_uint8() {
    let v = WithBool {
        flag: true,
        flags: vec![true, false, true],
    };
    let file = File::from_bytes(mat::to_bytes(&v).unwrap()).unwrap();

    let f = file.dataset("flag").unwrap();
    assert_eq!(f.shape().unwrap(), vec![1, 1]);
    assert_eq!(
        f.attrs()
            .unwrap()
            .get("MATLAB_class")
            .and_then(AttrValue::as_str),
        Some("logical")
    );
    assert_eq!(f.read_u8().unwrap(), vec![1]);

    let fs = file.dataset("flags").unwrap();
    assert_eq!(fs.shape().unwrap(), vec![1, 3]);
    assert_eq!(fs.read_u8().unwrap(), vec![1, 0, 1]);
}

// ---------------------------------------------------------------------------
// Full roundtrip: serialize then deserialize back, assert equality.
// ---------------------------------------------------------------------------

fn rt<T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(value: T) {
    let bytes = mat::to_bytes(&value).expect("serialize");
    let back: T = mat::from_bytes(&bytes).expect("deserialize");
    assert_eq!(back, value);
}

#[test]
fn roundtrip_simple_primitives() {
    rt(SimpleRoot {
        trial: 9,
        temperature: 273.15,
        name: "hello".into(),
    });
}

#[test]
fn roundtrip_vectors() {
    rt(Vecs {
        xs: vec![1.0, 2.0, 3.0, 4.0],
        ns: vec![-1, 0, 1],
        bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
    });
}

#[test]
fn roundtrip_matrix_f64() {
    rt(WithMatrix {
        m: Matrix::from_row_major(
            3,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ),
    });
}

#[test]
fn roundtrip_matrix_f64_column_vector() {
    // N×1 column vector: shape must round-trip distinctly from 1×N.
    rt(WithMatrix {
        m: Matrix::from_row_major(4, 1, vec![1.0, 2.0, 3.0, 4.0]),
    });
}

#[test]
fn roundtrip_matrix_f64_row_vector() {
    rt(WithMatrix {
        m: Matrix::from_row_major(1, 4, vec![1.0, 2.0, 3.0, 4.0]),
    });
}

#[test]
fn roundtrip_nested() {
    rt(WithNested {
        x: 1.5,
        inner: Inner {
            y: 42,
            z: vec![10.0, 20.0, 30.0],
        },
    });
}

#[test]
fn roundtrip_option_some_and_none() {
    rt(WithOption {
        required: 1.0,
        maybe: Some("world".into()),
    });
    rt(WithOption {
        required: 2.5,
        maybe: None,
    });
}

#[test]
fn roundtrip_complex_scalar_and_vec() {
    rt(WithComplex {
        z: Complex64::new(2.0, -1.0),
        zs: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, -1.0),
        ],
    });
}

#[test]
fn roundtrip_unit_enum() {
    rt(WithEnum { state: Flag::On });
    rt(WithEnum { state: Flag::Off });
}

#[test]
fn roundtrip_bool() {
    rt(WithBool {
        flag: true,
        flags: vec![false, true, false, true],
    });
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithStringUnicode {
    tag: String,
}

#[test]
fn roundtrip_non_ascii_string() {
    rt(WithStringUnicode {
        tag: "résumé 💡 αβγ".into(),
    });
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct VecOfVec {
    grid: Vec<Vec<f64>>,
}

#[test]
fn roundtrip_vec_of_vec_as_matrix() {
    rt(VecOfVec {
        grid: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    });
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct ComplexMat {
    m: Vec<Vec<Complex64>>,
}

#[test]
fn roundtrip_complex_matrix() {
    rt(ComplexMat {
        m: vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            vec![Complex64::new(-1.0, 0.0), Complex64::new(0.0, -1.0)],
        ],
    });
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithMatrixComplex64 {
    m: Matrix<Complex64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct WithMatrixComplex32 {
    m: Matrix<Complex32>,
}

#[test]
fn roundtrip_matrix_complex64_2d() {
    rt(WithMatrixComplex64 {
        m: Matrix::from_row_major(
            2,
            3,
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, 1.5),
                Complex64::new(3.0, -0.5),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 2.5),
                Complex64::new(7.0, -7.0),
            ],
        ),
    });
}

#[test]
fn roundtrip_matrix_complex64_column_vector() {
    // N×1 column vector: the orientation a downstream caller might choose
    // when they need the legacy `[N, 1]` MATLAB shape for 1-D complex data.
    rt(WithMatrixComplex64 {
        m: Matrix::from_row_major(
            4,
            1,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, -1.0),
            ],
        ),
    });
}

#[test]
fn roundtrip_matrix_complex64_row_vector() {
    rt(WithMatrixComplex64 {
        m: Matrix::from_row_major(
            1,
            3,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        ),
    });
}

#[test]
fn roundtrip_matrix_complex64_single_element() {
    rt(WithMatrixComplex64 {
        m: Matrix::from_row_major(1, 1, vec![Complex64::new(1.5, -2.5)]),
    });
}

#[test]
fn roundtrip_matrix_complex32_2d() {
    // Asymmetric shape (3×2) so a row/col swap or transpose regression fails
    // distinctly from the Complex64 2×3 pin above.
    rt(WithMatrixComplex32 {
        m: Matrix::from_row_major(
            3,
            2,
            vec![
                Complex32::new(1.0, 0.5),
                Complex32::new(2.0, 1.5),
                Complex32::new(3.0, -0.5),
                Complex32::new(-1.0, 0.0),
                Complex32::new(0.0, 2.5),
                Complex32::new(7.0, -7.0),
            ],
        ),
    });
}

#[test]
fn roundtrip_matrix_complex32_column_vector() {
    rt(WithMatrixComplex32 {
        m: Matrix::from_row_major(
            4,
            1,
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 1.0),
                Complex32::new(-1.0, 0.0),
                Complex32::new(0.0, -1.0),
            ],
        ),
    });
}

#[test]
fn roundtrip_matrix_complex32_row_vector() {
    rt(WithMatrixComplex32 {
        m: Matrix::from_row_major(
            1,
            3,
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
            ],
        ),
    });
}

#[test]
fn roundtrip_matrix_complex32_single_element() {
    rt(WithMatrixComplex32 {
        m: Matrix::from_row_major(1, 1, vec![Complex32::new(1.5, -2.5)]),
    });
}

/// Serialize once, assert the on-disk dtype is compound (real/imag), then
/// deserialize from those same bytes and assert value equality. Folds the
/// dtype check and the roundtrip into a single serialization.
fn assert_empty_complex_compound_roundtrip<T>(value: T)
where
    T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = mat::to_bytes(&value).expect("serialize");
    let file = File::from_bytes(bytes.clone()).unwrap();
    let dt = file.dataset("m").unwrap().dtype().unwrap();
    assert!(
        matches!(dt, hdf5_pure::DType::Compound(_)),
        "expected compound (real/imag) datatype for empty complex Matrix, got {dt:?}"
    );
    let back: T = mat::from_bytes(&bytes).expect("deserialize");
    assert_eq!(back, value);
}

#[test]
fn empty_matrix_complex64_writes_compound_dataset() {
    // A 0×0 Matrix<Complex64> has nothing in `data` for the seq path to
    // observe, so without the dedicated sentinel the Vec<Complex64> would
    // collapse to an f64-empty default and write as a numeric (non-complex)
    // dataset. The sentinel name preserves the element class.
    assert_empty_complex_compound_roundtrip(WithMatrixComplex64 {
        m: Matrix::from_row_major(0, 0, Vec::<Complex64>::new()),
    });
}

#[test]
fn empty_matrix_complex32_writes_compound_dataset() {
    assert_empty_complex_compound_roundtrip(WithMatrixComplex32 {
        m: Matrix::from_row_major(0, 0, Vec::<Complex32>::new()),
    });
}

// ---------------------------------------------------------------------------
// Edge-case roundtrips
// ---------------------------------------------------------------------------

/// Every integer Matrix<T> must survive roundtrip preserving dtype and values.
#[test]
fn roundtrip_int_matrices() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Ints {
        m_i8: Matrix<i8>,
        m_i16: Matrix<i16>,
        m_i32: Matrix<i32>,
        m_i64: Matrix<i64>,
        m_u8: Matrix<u8>,
        m_u16: Matrix<u16>,
        m_u32: Matrix<u32>,
        m_u64: Matrix<u64>,
    }
    rt(Ints {
        m_i8: Matrix::from_row_major(2, 2, vec![-128_i8, 127, 0, -1]),
        m_i16: Matrix::from_row_major(2, 2, vec![-32768_i16, 32767, 0, -1]),
        m_i32: Matrix::from_row_major(2, 3, vec![-1_i32, 2, 3, 4, 5, 6]),
        m_i64: Matrix::from_row_major(2, 2, vec![i64::MIN, -1_i64, 0, i64::MAX]),
        m_u8: Matrix::from_row_major(2, 2, vec![0_u8, 1, 254, 255]),
        m_u16: Matrix::from_row_major(2, 2, vec![0_u16, 1, 65534, 65535]),
        m_u32: Matrix::from_row_major(2, 2, vec![0_u32, 1, u32::MAX - 1, u32::MAX]),
        m_u64: Matrix::from_row_major(2, 2, vec![0_u64, 1, 2, u64::MAX]),
    });
}

#[test]
fn roundtrip_bool_matrix() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct BoolMat {
        m: Matrix<bool>,
    }
    rt(BoolMat {
        m: Matrix::from_row_major(
            3,
            3,
            vec![true, false, true, false, true, false, true, false, true],
        ),
    });
}

/// 1×1, 1×N, and N×1 matrix orientations for multiple element types.
#[test]
fn roundtrip_matrix_shape_edges() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Shapes {
        m_1x1: Matrix<f64>,
        m_1x5: Matrix<f64>,
        m_5x1: Matrix<f64>,
        i_5x1: Matrix<i32>,
        i_1x5: Matrix<i32>,
        b_3x1: Matrix<bool>,
        b_1x3: Matrix<bool>,
    }
    rt(Shapes {
        m_1x1: Matrix::from_row_major(1, 1, vec![42.0]),
        m_1x5: Matrix::from_row_major(1, 5, vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        m_5x1: Matrix::from_row_major(5, 1, vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        i_5x1: Matrix::from_row_major(5, 1, vec![-1, 0, 1, 2, 3]),
        i_1x5: Matrix::from_row_major(1, 5, vec![-1, 0, 1, 2, 3]),
        b_3x1: Matrix::from_row_major(3, 1, vec![true, false, true]),
        b_1x3: Matrix::from_row_major(1, 3, vec![true, false, true]),
    });
}

/// Bit-exact preservation of IEEE 754 specials: NaN, ±Inf, subnormal, -0.
#[test]
fn roundtrip_f64_bit_exact_specials() {
    #[derive(Serialize, Deserialize)]
    struct Specials {
        nan_bits: f64,
        pos_inf: f64,
        neg_inf: f64,
        neg_zero: f64,
        subnormal: f64,
        vals: Vec<f64>,
    }
    // Specific NaN bit pattern we can check after roundtrip.
    let quiet_nan = f64::from_bits(0x7FF8_0000_DEAD_BEEF);
    let v = Specials {
        nan_bits: quiet_nan,
        pos_inf: f64::INFINITY,
        neg_inf: f64::NEG_INFINITY,
        neg_zero: -0.0,
        subnormal: f64::from_bits(1),
        vals: vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 0.0],
    };
    let bytes = mat::to_bytes(&v).unwrap();
    let back: Specials = mat::from_bytes(&bytes).unwrap();
    // NaN payload must survive (quiet_nan bits preserved exactly).
    assert_eq!(back.nan_bits.to_bits(), quiet_nan.to_bits(), "nan payload");
    assert!(back.pos_inf.is_infinite() && back.pos_inf > 0.0);
    assert!(back.neg_inf.is_infinite() && back.neg_inf < 0.0);
    // -0 preserved bit-exact.
    assert_eq!(back.neg_zero.to_bits(), 0x8000_0000_0000_0000);
    // Subnormal preserved bit-exact.
    assert_eq!(back.subnormal.to_bits(), 1);
    // Vec contents.
    assert!(back.vals[0].is_nan());
    assert!(back.vals[1].is_infinite() && back.vals[1] > 0.0);
    assert!(back.vals[2].is_infinite() && back.vals[2] < 0.0);
    assert_eq!(back.vals[3].to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(back.vals[4].to_bits(), 0);
}

/// Integer extremes: i64::MIN/MAX, u64::MAX.
#[test]
fn roundtrip_integer_extremes() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Ex {
        i64_min: i64,
        i64_max: i64,
        u64_max: u64,
        i8_min: i8,
        i8_max: i8,
        vals: Vec<i64>,
    }
    rt(Ex {
        i64_min: i64::MIN,
        i64_max: i64::MAX,
        u64_max: u64::MAX,
        i8_min: i8::MIN,
        i8_max: i8::MAX,
        vals: vec![i64::MIN, -1, 0, 1, i64::MAX],
    });
}

/// Deeply nested struct (4 levels) roundtrip.
#[test]
fn roundtrip_deep_nested() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L1 {
        label: String,
        l2: L2,
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L2 {
        depth: u32,
        l3: L3,
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L3 {
        tag: String,
        l4: L4,
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct L4 {
        id: u64,
        values: Vec<f64>,
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Root {
        root: L1,
    }
    rt(Root {
        root: L1 {
            label: "top".into(),
            l2: L2 {
                depth: 2,
                l3: L3 {
                    tag: "middle".into(),
                    l4: L4 {
                        id: 999,
                        values: vec![1.5, 2.5, 3.5],
                    },
                },
            },
        },
    });
}

/// Non-BMP Unicode (surrogate pair) roundtrip.
#[test]
fn roundtrip_surrogate_pair() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Emoji {
        party: String,
        mixed: String,
    }
    rt(Emoji {
        party: "🎉".into(),
        mixed: "é日🎉A".into(),
    });
}

/// Empty containers of every primitive type survive roundtrip.
#[test]
fn roundtrip_empty_variants() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct E {
        e_f64: Vec<f64>,
        e_f32: Vec<f32>,
        e_i32: Vec<i32>,
        e_u8: Vec<u8>,
        e_bool: Vec<bool>,
        e_str: String,
    }
    rt(E {
        e_f64: vec![],
        e_f32: vec![],
        e_i32: vec![],
        e_u8: vec![],
        e_bool: vec![],
        e_str: String::new(),
    });
}

/// Large matrix roundtrip — catches any off-by-one in transposition at scale.
#[test]
fn roundtrip_large_matrix() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Big {
        m: Matrix<f64>,
    }
    let rows = 100;
    let cols = 50;
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push((r * 1000 + c) as f64);
        }
    }
    rt(Big {
        m: Matrix::from_row_major(rows, cols, data),
    });
}

/// Complex32 scalar and vec survive roundtrip preserving precision.
#[test]
fn roundtrip_complex32() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct C32 {
        z: Complex32,
        v: Vec<Complex32>,
    }
    rt(C32 {
        z: Complex32::new(1.25, -0.5),
        v: vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 1.0),
            Complex32::new(-1.0, -1.0),
        ],
    });
}

/// Bit-exact preservation of NaN/Inf/-0 inside a Matrix (not just scalars).
#[test]
fn roundtrip_matrix_with_specials() {
    #[derive(Serialize, Deserialize)]
    struct MatS {
        m: Matrix<f64>,
    }
    let m = Matrix::from_row_major(
        2,
        3,
        vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 0.0],
    );
    let bytes = mat::to_bytes(&MatS { m }).unwrap();
    let back: MatS = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back.m.rows(), 2);
    assert_eq!(back.m.cols(), 3);
    let d = back.m.data();
    assert_eq!(d[0], 1.0);
    assert!(d[1].is_nan());
    assert!(d[2].is_infinite() && d[2] > 0.0);
    assert!(d[3].is_infinite() && d[3] < 0.0);
    assert_eq!(d[4].to_bits(), 0x8000_0000_0000_0000); // -0
    assert_eq!(d[5].to_bits(), 0); // +0
}

/// Option<Struct> and Option<Vec<T>> work in both Some and None.
#[test]
fn roundtrip_option_complex_inner() {
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Inner {
        k: u32,
        v: f64,
    }
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Outer {
        some_struct: Option<Inner>,
        none_struct: Option<Inner>,
        some_vec: Option<Vec<f64>>,
        none_vec: Option<Vec<f64>>,
    }
    rt(Outer {
        some_struct: Some(Inner { k: 7, v: 3.14 }),
        none_struct: None,
        some_vec: Some(vec![1.0, 2.0, 3.0]),
        none_vec: None,
    });
}

// ---------------------------------------------------------------------------
// Enum edge cases (issue #129). The `.mat` serde path encodes a Rust enum only
// as a unit variant, by variant name, as a MATLAB char string. Non-unit
// variants and unknown names must be rejected cleanly, renamed variants must
// honor the serde name, and enums must round-trip inside sequences.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
enum NewtypeVariant {
    N(i32),
}
#[derive(Serialize)]
enum TupleVariant {
    T(i32, i32),
}
#[derive(Serialize)]
enum StructVariant {
    S { a: i32 },
}
#[derive(Serialize)]
struct HoldsNewtype {
    v: NewtypeVariant,
}
#[derive(Serialize)]
struct HoldsTuple {
    v: TupleVariant,
}
#[derive(Serialize)]
struct HoldsStruct {
    v: StructVariant,
}

#[test]
fn non_unit_enum_variants_are_unsupported() {
    assert!(matches!(
        mat::to_bytes(&HoldsNewtype {
            v: NewtypeVariant::N(1)
        }),
        Err(mat::MatError::UnsupportedType(_))
    ));
    assert!(matches!(
        mat::to_bytes(&HoldsTuple {
            v: TupleVariant::T(1, 2)
        }),
        Err(mat::MatError::UnsupportedType(_))
    ));
    assert!(matches!(
        mat::to_bytes(&HoldsStruct {
            v: StructVariant::S { a: 1 }
        }),
        Err(mat::MatError::UnsupportedType(_))
    ));
}

#[derive(Serialize)]
struct StateStr {
    state: String,
}
#[derive(Deserialize, Debug)]
struct StateFlag {
    #[allow(dead_code)]
    state: Flag,
}

#[test]
fn deserialize_unknown_enum_variant_errors() {
    // A char dataset whose value isn't a known variant name deserializes to a
    // clean serde `unknown variant` error, not a silent wrong variant.
    let bytes = mat::to_bytes(&StateStr {
        state: "Maybe".into(),
    })
    .unwrap();
    let err = mat::from_bytes::<StateFlag>(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unknown variant") && msg.contains("Maybe"),
        "expected an unknown-variant error mentioning \"Maybe\", got: {msg}"
    );
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Color {
    #[serde(rename = "rouge")]
    Red,
    #[serde(rename = "vert")]
    Green,
    Blue,
}
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Palette {
    c: Color,
}

#[test]
fn renamed_enum_variant_uses_serde_name() {
    let bytes = mat::to_bytes(&Palette { c: Color::Red }).unwrap();
    // The on-disk char string is the serde-renamed identifier, not "Red".
    let file = File::from_bytes(bytes.clone()).unwrap();
    let ds = file.dataset("c").unwrap();
    let units = ds.read_u16().unwrap();
    assert_eq!(String::from_utf16(&units).unwrap(), "rouge");
    // And it round-trips back to the original variant by that name.
    let back: Palette = mat::from_bytes(&bytes).unwrap();
    assert_eq!(back, Palette { c: Color::Red });
}

#[test]
fn all_enum_variants_roundtrip() {
    for c in [Color::Red, Color::Green, Color::Blue] {
        rt(Palette { c });
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct FlagList {
    states: Vec<Flag>,
}

#[test]
fn vec_of_unit_enums_roundtrips() {
    rt(FlagList {
        states: vec![Flag::On, Flag::Off, Flag::On],
    });
}

// ---------------------------------------------------------------------------
// The two emit paths must not drift apart
// ---------------------------------------------------------------------------

/// `to_bytes` and `to_bytes_with_options(&Options::default())` run through
/// *different emitters* — `ser::emit` and `ser::emit_with_builder` — and each
/// carries its own copy of every file-level setting. So a new default applied to
/// one silently leaves the other behind, which is what happened when the HDF5
/// 1.8 format became the MAT default: `to_file` kept writing a version 3
/// superblock, and `to_file` is the entry point a `.mat` file is usually
/// produced by.
///
/// The `matio` crosscheck holds the same invariant, but only for one value and
/// only when libmatio is installed. This runs on the default feature set, over
/// shapes that reach both emitters' interesting arms.
///
/// Emptiness is where the two have come apart most: they recorded different
/// MATLAB dims for an empty `Vec`, `String`, and `Vec<bool>` (`[0, 0]` against
/// `[0, 1]`), and `apply_matrix` was left out of the `DataAsDims` default
/// entirely, so an empty `Matrix` stayed a zero-element array on one side and
/// became a dimension vector on the other. Every shape of emptiness the writers
/// treat separately therefore belongs in `Every` below — a shape that is missing
/// is a divergence this test cannot see.
#[test]
fn both_emit_paths_produce_the_same_bytes() {
    #[derive(Serialize)]
    struct Every {
        scalar: f64,
        vector: Vec<i32>,
        text: String,
        matrix: Matrix<f64>,
        nested: Inner,
        cells: Vec<Vec<i32>>,
        complex: Vec<Complex64>,
        // The empty cases carry their own history: the two emitters described an
        // empty `Vec` with different MATLAB dimensions (`0x0` against `0x1`) and
        // nothing caught it, because this comparison held no empty value.
        empty_vector: Vec<f64>,
        empty_ints: Vec<i32>,
        empty_flags: Vec<bool>,
        empty_text: String,
        empty_cells: Vec<Vec<i32>>,
        // A `Matrix` reaches an emit arm of its own, and only one of the two was
        // converted to the empty marker. The non-square case is the one that
        // pins the dims: a `0x0` marker would still match if both sides
        // collapsed every empty matrix to `0x0`.
        empty_matrix: Matrix<f64>,
        empty_matrix_cols: Matrix<f64>,
        // An empty complex array reaches its own emit arm, and only the
        // options emitter routed it through `vector_dims`; the other described
        // it `0x1`. Unreachable until `mat::complex`'s array helpers made an
        // empty `ComplexVec1D` constructible, which is exactly the shape this
        // test warns about above.
        #[serde(serialize_with = "mat::complex::f64_array")]
        empty_complex: Vec<Complex64>,
        absent: Option<f64>,
        nothing: (),
    }
    #[derive(Serialize)]
    struct Inner {
        flag: bool,
        count: u32,
    }

    let value = Every {
        scalar: 1.5,
        vector: vec![1, 2, 3],
        text: "hello".to_string(),
        matrix: Matrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
        nested: Inner {
            flag: true,
            count: 7,
        },
        cells: vec![vec![1], vec![2, 3]],
        complex: vec![Complex64::new(1.0, -1.0)],
        empty_vector: Vec::new(),
        empty_ints: Vec::new(),
        empty_flags: Vec::new(),
        empty_text: String::new(),
        empty_cells: Vec::new(),
        empty_matrix: Matrix::from_row_major(0, 0, Vec::new()),
        empty_matrix_cols: Matrix::from_row_major(0, 3, Vec::new()),
        empty_complex: Vec::new(),
        absent: None,
        nothing: (),
    };

    let plain = mat::to_bytes(&value).unwrap();
    let explicit = mat::to_bytes_with_options(&value, &mat::Options::default()).unwrap();
    assert_eq!(
        plain, explicit,
        "to_bytes and to_bytes_with_options(&Options::default()) must agree byte for byte"
    );

    // And the shared result is the format MATLAB's `load` can read.
    let sig = plain
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("the file carries an HDF5 signature");
    assert_eq!(plain[sig + 8], 2, "version 2 superblock");
}

/// The streaming entry points are a third and fourth copy of the same decision.
#[test]
fn the_streaming_emit_paths_match_the_buffered_ones() {
    #[derive(Serialize)]
    struct Small {
        values: Vec<f64>,
        label: String,
    }
    let value = Small {
        values: vec![1.0, 2.0, 3.0],
        label: "demo".to_string(),
    };

    let buffered = mat::to_bytes(&value).unwrap();
    let mut streamed = Vec::new();
    mat::to_writer(&value, &mut streamed).unwrap();
    assert_eq!(buffered, streamed, "to_bytes and to_writer must agree");

    let mut streamed_opts = Vec::new();
    mat::to_writer_with_options(&value, &mat::Options::default(), &mut streamed_opts).unwrap();
    assert_eq!(
        buffered, streamed_opts,
        "to_writer_with_options(&Options::default()) must agree too"
    );
}
