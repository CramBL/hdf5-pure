//! Crosscheck the pure-Rust ZFP codec against reference fixtures produced by
//! h5py + the real H5Z-ZFP plugin.
//!
//! Fixtures and generator script live in `tests/data/h5py/zfp/`. Regenerate:
//!
//!     tests/data/h5py/zfp/.venv/bin/python tests/data/h5py/zfp/regen.py
//!
//! The test iterates every fixture in the manifest. For each fixture that
//! falls inside the currently-implemented codec slice (see `is_supported`),
//! it runs both paths:
//!
//!   * decompress: our codec reads the reference compressed bytes and the
//!     decoded values are compared against the raw values within a
//!     rate-dependent tolerance.
//!   * compress: our codec encodes the raw values and the output is compared
//!     byte-equal against the reference compressed bytes.
//!
//! Fixtures outside the supported slice are recorded as skipped; they become
//! active as the codec is parameterized (Step 3) and extended to N-D
//! (Step 5). Any supported-fixture failure fails the test.

use std::fs;
use std::path::PathBuf;

use crate::zfp::{self, ZfpElementType};
use serde::Deserialize;

fn dtype_to_elem_type(dtype: &str) -> Result<ZfpElementType, String> {
    match dtype {
        "f32" => Ok(ZfpElementType::F32),
        "f64" => Ok(ZfpElementType::F64),
        "i32" => Ok(ZfpElementType::I32),
        "i64" => Ok(ZfpElementType::I64),
        other => Err(format!("unknown dtype {other}")),
    }
}

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    rate: f64,
    mode: String,
    #[allow(dead_code)]
    filter_name: String,
    #[allow(dead_code)]
    cd_values_u32: Vec<u32>,
    raw_bytes_len: usize,
    compressed_bytes_len: usize,
    #[allow(dead_code)]
    notes: String,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/h5py/zfp")
}

fn load_manifest() -> Manifest {
    let path = fixture_dir().join("manifest.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\n(run regen.py to produce fixtures)",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("parse manifest.json")
}

fn read_bin(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// True if the fixture falls inside the slice of the codec implemented so
/// far. All four types and 1D–4D are in scope after Step 5.
fn is_supported(fix: &Fixture) -> bool {
    let dtype_ok = matches!(fix.dtype.as_str(), "f32" | "f64" | "i32" | "i64");
    let rank_ok = matches!(fix.shape.len(), 1..=4);
    dtype_ok && rank_ok && fix.mode == "rate"
}

/// Expected element size in bytes for a fixture's dtype.
fn element_size(dtype: &str) -> usize {
    match dtype {
        "f32" | "i32" => 4,
        "f64" | "i64" => 8,
        other => panic!("unknown dtype {other}"),
    }
}

/// Max absolute reconstruction error we accept at a given rate.
///
/// Floats: the block-floating-point cast gives roughly `2^-(rate/2)` relative
/// error, so a rate-16 f32 reconstruction is good to ~2^-8.
///
/// Integers: at rates below `INTPREC`, ZFP's lifting+truncation introduces a
/// sizeable but well-defined quantization error. The encode-byte-match check
/// is the real crosscheck gate (any deviation fails there); the decode bound
/// is just a sanity floor, so we use a loose `max(|raw|) / 2` here.
fn decode_tolerance(fix: &Fixture, raw: &[u8]) -> f64 {
    match fix.dtype.as_str() {
        "f32" | "f64" => 2f64.powi(-(fix.rate as i32 / 2)),
        "i32" => {
            let max_abs = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c).unsigned_abs() as f64)
                .fold(0f64, f64::max);
            (max_abs / 2.0).max(1.0)
        }
        "i64" => {
            let max_abs = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c).unsigned_abs() as f64)
                .fold(0f64, f64::max);
            (max_abs / 2.0).max(1.0)
        }
        _ => 1e30,
    }
}

/// Per-dtype decode + max-error computation.
fn decode_and_max_err(
    dtype: &str,
    reference: &[u8],
    dims: &[usize],
    rate: f64,
    raw: &[u8],
) -> Result<f64, String> {
    fn max_abs<T: Copy + Into<f64>>(a: &[T], b: &[T]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x.into() - y.into()).abs())
            .fold(0f64, f64::max)
    }
    let elem_ty = dtype_to_elem_type(dtype)?;
    let decoded = zfp::decompress(reference, dims, rate, elem_ty).map_err(|e| format!("{e:?}"))?;
    match dtype {
        "f32" => {
            let expected: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let got: Vec<f32> = decoded
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            if got.len() != expected.len() {
                return Err(format!(
                    "decoded {} != expected {}",
                    got.len(),
                    expected.len()
                ));
            }
            Ok(max_abs(&expected, &got))
        }
        "f64" => {
            let expected: Vec<f64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect();
            let got: Vec<f64> = decoded
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect();
            if got.len() != expected.len() {
                return Err(format!(
                    "decoded {} != expected {}",
                    got.len(),
                    expected.len()
                ));
            }
            Ok(max_abs(&expected, &got))
        }
        "i32" => {
            let expected: Vec<i32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect();
            let got: Vec<i32> = decoded
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect();
            if got.len() != expected.len() {
                return Err(format!(
                    "decoded {} != expected {}",
                    got.len(),
                    expected.len()
                ));
            }
            let max = expected
                .iter()
                .zip(got.iter())
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0f64, f64::max);
            Ok(max)
        }
        "i64" => {
            let expected: Vec<i64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect();
            let got: Vec<i64> = decoded
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect();
            if got.len() != expected.len() {
                return Err(format!(
                    "decoded {} != expected {}",
                    got.len(),
                    expected.len()
                ));
            }
            let max = expected
                .iter()
                .zip(got.iter())
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0f64, f64::max);
            Ok(max)
        }
        other => Err(format!("unknown dtype {other}")),
    }
}

fn encode_per_dtype(dtype: &str, raw: &[u8], dims: &[usize], rate: f64) -> Result<Vec<u8>, String> {
    let elem_ty = dtype_to_elem_type(dtype)?;
    zfp::compress(raw, dims, rate, elem_ty).map_err(|e| format!("{e:?}"))
}

#[derive(Debug)]
struct CaseOutcome {
    name: String,
    decode_max_err: Option<f64>,
    encode_byte_match: Option<bool>,
    decode_err_msg: Option<String>,
    encode_err_msg: Option<String>,
}

fn run_case(fix: &Fixture) -> CaseOutcome {
    let raw = read_bin(&format!("{}.raw.bin", fix.name));
    let reference = read_bin(&format!("{}.compressed.bin", fix.name));
    assert_eq!(raw.len(), fix.raw_bytes_len);
    assert_eq!(reference.len(), fix.compressed_bytes_len);

    let elem_size = element_size(&fix.dtype);
    let num_values: usize = fix.shape.iter().product();
    assert_eq!(raw.len(), num_values * elem_size);

    let (decode_max_err, decode_err_msg) =
        match decode_and_max_err(&fix.dtype, &reference, &fix.shape, fix.rate, &raw) {
            Ok(e) => (Some(e), None),
            Err(m) => (None, Some(m)),
        };

    let (encode_byte_match, encode_err_msg) =
        match encode_per_dtype(&fix.dtype, &raw, &fix.shape, fix.rate) {
            Ok(encoded) => (Some(encoded == reference), None),
            Err(m) => (None, Some(m)),
        };

    CaseOutcome {
        name: fix.name.clone(),
        decode_max_err,
        encode_byte_match,
        decode_err_msg,
        encode_err_msg,
    }
}

#[test]
fn zfp_crosscheck() {
    let manifest = load_manifest();
    let mut skipped: Vec<&str> = Vec::new();
    let mut passed: Vec<String> = Vec::new();
    let mut failed: Vec<(String, String)> = Vec::new();

    for fix in &manifest.fixtures {
        if !is_supported(fix) {
            skipped.push(&fix.name);
            continue;
        }
        let outcome = run_case(fix);
        let raw = read_bin(&format!("{}.raw.bin", fix.name));
        let tol = decode_tolerance(fix, &raw);
        let decode_ok = match (&outcome.decode_err_msg, outcome.decode_max_err) {
            (None, Some(e)) => e <= tol,
            _ => false,
        };
        let encode_ok = outcome.encode_byte_match.unwrap_or(false);

        if decode_ok && encode_ok {
            passed.push(outcome.name.clone());
        } else {
            let mut why = Vec::new();
            if let Some(msg) = &outcome.decode_err_msg {
                why.push(format!("decode error: {msg}"));
            } else if let Some(e) = outcome.decode_max_err {
                if !decode_ok {
                    why.push(format!("decode max_err={e:.6} > tol={tol:.6}"));
                }
            }
            if let Some(msg) = &outcome.encode_err_msg {
                why.push(format!("encode error: {msg}"));
            } else if !encode_ok {
                why.push("encode: bytes differ from reference".to_string());
            }
            failed.push((outcome.name.clone(), why.join("; ")));
        }
    }

    eprintln!("\n=== ZFP crosscheck ===");
    eprintln!("passed:  {}", passed.len());
    eprintln!("failed:  {}", failed.len());
    eprintln!("skipped: {} (outside current codec slice)", skipped.len());

    if !failed.is_empty() {
        eprintln!("\nfailures:");
        for (name, why) in &failed {
            eprintln!("  {name}: {why}");
        }
        panic!("{} supported fixtures failed crosscheck", failed.len());
    }

    // A green result with zero positively-verified fixtures is not a real pass:
    // it means the manifest listed no in-slice fixtures (e.g. fixtures were
    // never generated). We do not fail the build for legitimately-absent
    // fixtures, but make the "nothing was checked" outcome explicit/observable
    // rather than a silent green.
    if passed.is_empty() {
        eprintln!(
            "WARNING: ZFP crosscheck verified 0 fixtures \
             ({} total in manifest, {} skipped). Nothing was cross-checked \
             against the reference plugin; run regen.py to produce fixtures.",
            manifest.fixtures.len(),
            skipped.len()
        );
    }
}
