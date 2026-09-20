#![cfg(feature = "serde")]
//! What an empty MATLAB value looks like on disk, measured against MATLAB.
//!
//! The `.mat` files under `tests/data/matlab` are genuine real-MATLAB v7.3
//! output (their userblock records `Platform: PCWIN64`; see `NOTICE.md` there),
//! so they are ground truth for questions the format specification does not
//! answer — MAT v7.3 is not publicly documented, and the community write-ups
//! describe the encoding without pinning these particulars.
//!
//! Two such questions cost this crate a defect each, both of which showed up as
//! the two MAT emitters disagreeing with each other:
//!
//! 1. **Which dimensions does an empty value carry?** `0x0` and `0x1` are both
//!    legal — MATLAB stores whatever `size(x)` is, so `[]` gives one and
//!    `zeros(0,1)` the other. The emitters had drifted to different answers for
//!    the same empty `Vec`.
//! 2. **Does an empty value carry `MATLAB_int_decode`?** Both emitters wrote it,
//!    in different attribute orders, and MATLAB writes it on no empty at all.
//!
//! Rather than assert the answers as constants, [`matlab_writes_no_int_decode_on_an_empty`]
//! re-derives them from the fixtures on every run. If a future fixture disagrees,
//! the measurement fails rather than silently disagreeing with the rule the
//! writer follows.

use hdf5_pure::mat::{self, EmptySequencePolicy, MatBuilder, MatClass, Options};
use hdf5_pure::{AttrValue, File};
use serde::Serialize;

/// Every dataset in `file` that carries a `MATLAB_empty` attribute, as
/// `(class, sorted attribute names, dims payload)`.
fn empty_markers(file: &File) -> Vec<(String, Vec<String>, Vec<u64>)> {
    fn walk(file: &File, path: &str, out: &mut Vec<(String, Vec<String>, Vec<u64>)>) {
        let group = match if path.is_empty() {
            Ok(file.root())
        } else {
            file.group(path)
        } {
            Ok(g) => g,
            Err(_) => return,
        };
        for name in group.datasets().unwrap_or_default() {
            let full = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}/{name}")
            };
            let Ok(ds) = file.dataset(&full) else {
                continue;
            };
            let Ok(attrs) = ds.attrs() else { continue };
            if !attrs.contains_key("MATLAB_empty") {
                continue;
            }
            let class = match attrs.get("MATLAB_class") {
                Some(AttrValue::AsciiString(s)) | Some(AttrValue::String(s)) => s.clone(),
                _ => continue,
            };
            let mut names: Vec<String> = attrs.keys().cloned().collect();
            names.sort();
            let dims = ds.read_u64().unwrap_or_default();
            out.push((class, names, dims));
        }
        for sub in group.groups().unwrap_or_default() {
            let full = if path.is_empty() {
                sub.clone()
            } else {
                format!("{path}/{sub}")
            };
            walk(file, &full, out);
        }
    }
    let mut out = Vec::new();
    walk(file, "", &mut out);
    out
}

fn all_matlab_empty_markers() -> Vec<(String, Vec<String>, Vec<u64>)> {
    let mut all = Vec::new();
    let dir = std::path::Path::new("tests/data/matlab");
    for entry in std::fs::read_dir(dir).expect("the real-MATLAB fixture directory") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("mat") {
            continue;
        }
        // Two fixtures are deliberately corrupted read-failure cases.
        if let Ok(f) = File::open(&path) {
            all.extend(empty_markers(&f));
        }
    }
    all
}

/// The measurement this crate's empty-marker encoding is built on, re-derived
/// from MATLAB's own files on every run.
///
/// The count assertion is what keeps this from passing vacuously: a walk that
/// silently found nothing would otherwise satisfy every `all(...)` below.
#[test]
fn matlab_writes_no_int_decode_on_an_empty() {
    let markers = all_matlab_empty_markers();
    assert!(
        markers.len() > 300,
        "expected the fixtures to hold hundreds of empty markers, found {} — \
         the walk is not reaching them",
        markers.len()
    );

    let with_decode: Vec<_> = markers
        .iter()
        .filter(|(_, attrs, _)| attrs.iter().any(|a| a == "MATLAB_int_decode"))
        .collect();
    assert!(
        with_decode.is_empty(),
        "MATLAB wrote MATLAB_int_decode on an empty value: {with_decode:?}"
    );

    // Every marker is a two-element dimension vector, including the `[1 1]` and
    // `[1 0]` shapes a zero-element encoding could not express.
    assert!(
        markers.iter().all(|(_, _, dims)| dims.len() == 2),
        "every empty marker's payload is a 2-element dimension vector"
    );

    // And `0x0` is the dominant empty by a wide margin, which is why an empty
    // Rust sequence maps to it rather than to an oriented `0x1`/`1x0`.
    let zero_by_zero = markers
        .iter()
        .filter(|(_, _, dims)| dims == &vec![0, 0])
        .count();
    assert!(
        zero_by_zero * 2 > markers.len(),
        "0x0 should be the majority empty shape, got {zero_by_zero} of {}",
        markers.len()
    );
}

/// Our writer follows the rule the fixtures show: `MATLAB_class` and
/// `MATLAB_empty`, and nothing else.
#[test]
fn our_empty_markers_carry_the_same_attributes_matlab_writes() {
    // Char and logical are the classes that carry `MATLAB_int_decode` when they
    // hold data, so they are the ones that could wrongly carry it when empty.
    for class in [
        MatClass::Double,
        MatClass::Char,
        MatClass::Logical,
        MatClass::UInt8,
        MatClass::Struct,
    ] {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_empty("e", class, &[0, 0]).unwrap();
        let f = File::from_bytes(mb.finish().unwrap()).unwrap();
        let mut got: Vec<String> = f
            .dataset("e")
            .unwrap()
            .attrs()
            .unwrap()
            .into_keys()
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec!["MATLAB_class".to_string(), "MATLAB_empty".to_string()],
            "{class:?} empty marker carries the wrong attribute set"
        );
    }
}

/// The serde path, through both emitters, for every kind of empty a Rust value
/// can produce. `MATLAB_int_decode` reached this path through `String` and
/// `Vec<bool>`, which are the char and logical cases.
///
/// `EmptySequencePolicy::Cell` is a case of its own because it is the only way
/// an empty *cell* is reachable, and an empty cell took its shape from a rule
/// of its own rather than from `dims`. That is how it kept writing `0x1` after
/// every other empty moved to `0x0`, leaving the serde writer disagreeing with
/// the BEVE walker for the same value.
#[test]
fn the_serde_emitters_write_matlab_conformant_empties() {
    #[derive(Serialize)]
    struct Doc {
        empty_f64: Vec<f64>,
        empty_bool: Vec<bool>,
        empty_text: String,
        empty_cells: Vec<Vec<i32>>,
        absent: Option<f64>,
    }
    let doc = Doc {
        empty_f64: Vec::new(),
        empty_bool: Vec::new(),
        empty_text: String::new(),
        empty_cells: Vec::new(),
        absent: None,
    };

    let mut as_cells = Options::default();
    as_cells.empty_sequence_policy = EmptySequencePolicy::Cell;

    for (label, bytes) in [
        ("to_bytes", mat::to_bytes(&doc).unwrap()),
        (
            "to_bytes_with_options",
            mat::to_bytes_with_options(&doc, &Options::default()).unwrap(),
        ),
        (
            "to_bytes_with_options(Cell)",
            mat::to_bytes_with_options(&doc, &as_cells).unwrap(),
        ),
    ] {
        let f = File::from_bytes(bytes).unwrap();
        let markers = empty_markers(&f);
        assert_eq!(markers.len(), 5, "{label}: every field is an empty marker");
        for (class, attrs, dims) in markers {
            assert_eq!(
                attrs,
                vec!["MATLAB_class".to_string(), "MATLAB_empty".to_string()],
                "{label}: {class} empty carries a non-MATLAB attribute set"
            );
            assert_eq!(dims, vec![0, 0], "{label}: {class} empty is not 0x0");
        }
    }
}
