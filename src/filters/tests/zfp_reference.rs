//! Checks ZFP encoding and decoding against independent H5Z-ZFP vectors.
//!
//! Supported vectors must match the encoded bytes exactly and decode within an absolute
//! tolerance for their scalar type and rate. Non-finite reconstruction differences fail.
//! The suite requires at least one verified vector. Unsupported modes, ranks, and scalar
//! types do not count as verified evidence, and malformed descriptions fail the suite.
//!
//! Regenerate the vectors on a little-endian host with `just test-data::zfp`. The generator
//! pins its dependencies in `tests/data/h5py/zfp/regen.py` and uses temporary HDF5 containers
//! for extraction.

#![cfg(all(test, feature = "zfp", feature = "std"))]

use core::num::NonZeroUsize;

use std::fs;

use hdf5_pure_core::FormatError;
use hdf5_pure_core::ZfpRate;
use hdf5_pure_filter::__private::Error as FilterError;
use hdf5_pure_filter::__private::ResolvedZfpSettings;
use hdf5_pure_filter::__private::ZfpElementType;
use rstest::rstest;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Clone, Debug, Deserialize)]
struct Fixture {
    name: String,
    dtype: Scalar,
    shape: Vec<usize>,
    rate: f64,
    mode: Mode,
    #[allow(dead_code)]
    filter_name: String,
    #[allow(dead_code)]
    cd_values_u32: Vec<u32>,
    raw_bytes_len: usize,
    compressed_bytes_len: usize,
    #[allow(dead_code)]
    notes: String,
}

fn load_manifest() -> Manifest {
    let path = test_util::data("h5py/zfp/manifest.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\n(run regen.py to produce fixtures)",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("parse manifest.json")
}

fn read_bin(name: &str) -> Vec<u8> {
    let path = test_util::data("h5py/zfp").join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Scalar {
    F32,
    F64,
    I16,
    I32,
    I64,
    I8,
    U16,
    U32,
    U64,
    U8,
}

impl Scalar {
    fn width(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    fn element_type(self) -> Option<ZfpElementType> {
        match self {
            Self::F32 => Some(ZfpElementType::F32),
            Self::F64 => Some(ZfpElementType::F64),
            Self::I32 => Some(ZfpElementType::I32),
            Self::I64 => Some(ZfpElementType::I64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Accuracy,
    Precision,
    Rate,
    Reversible,
}

#[derive(Debug)]
struct SupportedFixture {
    name: String,
    element_type: ZfpElementType,
    shape: Vec<usize>,
    rate: ZfpRate,
    raw_bytes_len: usize,
    compressed_bytes_len: usize,
}

impl SupportedFixture {
    fn verify(&self) -> Result<(), String> {
        let raw = read_bin(&format!("{}.raw.bin", self.name));
        let reference = read_bin(&format!("{}.compressed.bin", self.name));
        if raw.len() != self.raw_bytes_len {
            return Err(format!(
                "raw bytes {} != recorded {}",
                raw.len(),
                self.raw_bytes_len
            ));
        }
        if reference.len() != self.compressed_bytes_len {
            return Err(format!(
                "reference bytes {} != recorded {}",
                reference.len(),
                self.compressed_bytes_len
            ));
        }
        let tolerance = self.decode_tolerance(&raw);
        let error = decode_and_max_err(
            self.element_type,
            &reference,
            &self.shape,
            self.rate.get(),
            &raw,
        )?;
        if error > tolerance || !error.is_finite() {
            return Err(format!("decode max_err={error:.6} > tol={tolerance:.6}"));
        }
        let encoded = encode_per_dtype(self.element_type, &raw, &self.shape, self.rate.get())?;
        if encoded != reference {
            return Err("encode: bytes differ from reference".into());
        }
        Ok(())
    }

    /// Returns the fixture's absolute reconstruction-error bound.
    ///
    /// Floating-point fixtures use a power of two with exponent `-(self.rate.get() as i32 / 2)`.
    /// Integer fixtures use half the largest absolute input value, with a minimum of one.
    fn decode_tolerance(&self, raw: &[u8]) -> f64 {
        match self.element_type {
            ZfpElementType::F32 | ZfpElementType::F64 => 2f64.powi(-(self.rate.get() as i32 / 2)),
            ZfpElementType::I32 => {
                let max_abs = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c).unsigned_abs() as f64)
                    .fold(0f64, f64::max);
                (max_abs / 2.0).max(1.0)
            }
            ZfpElementType::I64 => {
                let max_abs = raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c).unsigned_abs() as f64)
                    .fold(0f64, f64::max);
                (max_abs / 2.0).max(1.0)
            }
            other => panic!("unsupported parsed scalar {other:?}"),
        }
    }
}

#[derive(Debug)]
enum FixtureCase {
    Supported(SupportedFixture),
    Unsupported {
        name: String,
        reason: UnsupportedReason,
    },
}

#[derive(Debug, PartialEq)]
enum UnsupportedReason {
    Mode(Mode),
    RankAboveFour,
    Scalar(Scalar),
}

#[derive(Debug, PartialEq)]
enum MalformedDescription {
    EmptyReference,
    InvalidName(String),
    InvalidRate(FormatError),
    InvalidSettings(FilterError),
    InvalidShape(Vec<usize>),
    RawLength { expected: usize, recorded: usize },
    SizeOverflow,
}

impl TryFrom<Fixture> for FixtureCase {
    type Error = MalformedDescription;

    fn try_from(fixture: Fixture) -> Result<Self, Self::Error> {
        let Fixture {
            name,
            dtype,
            shape,
            rate,
            mode,
            filter_name: _,
            cd_values_u32: _,
            raw_bytes_len,
            compressed_bytes_len,
            notes: _,
        } = fixture;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(MalformedDescription::InvalidName(name));
        }
        let rate = ZfpRate::try_from(rate).map_err(MalformedDescription::InvalidRate)?;
        if shape.is_empty() || shape.contains(&0) {
            return Err(MalformedDescription::InvalidShape(shape));
        }
        let expected = shape
            .iter()
            .try_fold(dtype.width(), |length, size| length.checked_mul(*size))
            .ok_or(MalformedDescription::SizeOverflow)?;
        if raw_bytes_len != expected {
            return Err(MalformedDescription::RawLength {
                expected,
                recorded: raw_bytes_len,
            });
        }
        if compressed_bytes_len == 0 {
            return Err(MalformedDescription::EmptyReference);
        }
        if mode != Mode::Rate {
            return Ok(Self::Unsupported {
                name,
                reason: UnsupportedReason::Mode(mode),
            });
        }
        if shape.len() > 4 {
            return Ok(Self::Unsupported {
                name,
                reason: UnsupportedReason::RankAboveFour,
            });
        }
        let Some(element_type) = dtype.element_type() else {
            return Ok(Self::Unsupported {
                name,
                reason: UnsupportedReason::Scalar(dtype),
            });
        };
        let dimensions: Vec<_> = shape
            .iter()
            .map(|&size| u64::try_from(size).unwrap())
            .collect();
        ResolvedZfpSettings::new(rate, element_type, &dimensions)
            .map_err(MalformedDescription::InvalidSettings)?;
        Ok(Self::Supported(SupportedFixture {
            name,
            element_type,
            shape,
            rate,
            raw_bytes_len,
            compressed_bytes_len,
        }))
    }
}

/// Decodes a reference stream and returns the largest absolute difference from its input.
///
/// # Errors
///
/// Returns an error if decoding fails, the decoded byte length differs from `raw`, or
/// `dtype` is outside the supported scalar set.
fn decode_and_max_err(
    dtype: ZfpElementType,
    reference: &[u8],
    dims: &[usize],
    rate: f64,
    raw: &[u8],
) -> Result<f64, String> {
    let decoded = hdf5_pure_filter::__private::decompress_zfp(reference, dims, rate, dtype)
        .map_err(|e| format!("{e:?}"))?;
    if decoded.len() != raw.len() {
        return Err(format!(
            "decoded {} bytes != expected {}",
            decoded.len(),
            raw.len()
        ));
    }
    match dtype {
        ZfpElementType::F32 => {
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
            Ok(maximum_error(&expected, &got))
        }
        ZfpElementType::F64 => {
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
            Ok(maximum_error(&expected, &got))
        }
        ZfpElementType::I32 => {
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
        ZfpElementType::I64 => {
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
        other => Err(format!("unsupported parsed scalar {other:?}")),
    }
}

/// Returns the maximum absolute difference, or infinity if any paired difference is non-finite.
fn maximum_error<T: Copy + Into<f64>>(reference: &[T], reconstructed: &[T]) -> f64 {
    reference
        .iter()
        .zip(reconstructed)
        .map(|(&expected, &actual)| {
            let error = (expected.into() - actual.into()).abs();
            if error.is_finite() {
                error
            } else {
                f64::INFINITY
            }
        })
        .fold(0f64, f64::max)
}

fn encode_per_dtype(
    dtype: ZfpElementType,
    raw: &[u8],
    dims: &[usize],
    rate: f64,
) -> Result<Vec<u8>, String> {
    hdf5_pure_filter::__private::compress_zfp(raw, dims, rate, dtype).map_err(|e| format!("{e:?}"))
}

#[derive(Debug, PartialEq)]
enum CrosscheckFailure {
    Malformed {
        name: String,
        reason: MalformedDescription,
    },
    NoVerifiedFixtures {
        skipped: usize,
    },
    VerificationFailed(Vec<(String, String)>),
}

fn verify_manifest(
    manifest: Manifest,
    mut verify: impl FnMut(&SupportedFixture) -> Result<(), String>,
) -> Result<NonZeroUsize, CrosscheckFailure> {
    let cases = manifest
        .fixtures
        .into_iter()
        .map(|fixture| {
            let name = fixture.name.clone();
            FixtureCase::try_from(fixture)
                .map_err(|reason| CrosscheckFailure::Malformed { name, reason })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut verified = 0;
    let mut skipped = 0;
    let mut failed = Vec::new();
    for case in cases {
        match case {
            FixtureCase::Supported(fixture) => match verify(&fixture) {
                Ok(()) => verified += 1,
                Err(reason) => failed.push((fixture.name, reason)),
            },
            FixtureCase::Unsupported { name, reason } => {
                eprintln!("skipped {name}: {reason:?}");
                skipped += 1;
            }
        }
    }
    eprintln!(
        "ZFP reference checks: {verified} verified, {} failed, {skipped} unsupported",
        failed.len()
    );
    if !failed.is_empty() {
        return Err(CrosscheckFailure::VerificationFailed(failed));
    }
    NonZeroUsize::new(verified).ok_or(CrosscheckFailure::NoVerifiedFixtures { skipped })
}

#[test]
fn supported_vectors_match_reference_bytes_and_values() {
    verify_manifest(load_manifest(), SupportedFixture::verify).unwrap();
}

#[test]
fn an_empty_manifest_cannot_verify_reference_evidence() {
    assert_eq!(
        verify_manifest(
            Manifest {
                fixtures: Vec::new()
            },
            |_| panic!("no supported case")
        ),
        Err(CrosscheckFailure::NoVerifiedFixtures { skipped: 0 }),
    );
}

#[rstest]
#[case::accuracy(serde_json::json!({"mode": "accuracy"}), UnsupportedReason::Mode(Mode::Accuracy))]
#[case::precision(serde_json::json!({"mode": "precision"}), UnsupportedReason::Mode(Mode::Precision))]
#[case::reversible(serde_json::json!({"mode": "reversible"}), UnsupportedReason::Mode(Mode::Reversible))]
#[case::unsigned(serde_json::json!({"dtype": "u32"}), UnsupportedReason::Scalar(Scalar::U32))]
#[case::small_unsigned(serde_json::json!({"dtype": "u8", "raw_bytes_len": 16}), UnsupportedReason::Scalar(Scalar::U8))]
#[case::small_signed(serde_json::json!({"dtype": "i16", "raw_bytes_len": 32}), UnsupportedReason::Scalar(Scalar::I16))]
#[case::rank_five(serde_json::json!({"shape": [1,1,1,1,16]}), UnsupportedReason::RankAboveFour)]
fn unsupported_descriptions_have_explicit_reasons_and_cannot_verify_evidence(
    #[case] changes: serde_json::Value,
    #[case] expected: UnsupportedReason,
) {
    let fixture = fixture_description(changes);
    let FixtureCase::Unsupported { name, reason } = FixtureCase::try_from(fixture.clone()).unwrap()
    else {
        panic!("expected unsupported description")
    };
    assert_eq!(name, "f32_1d_16_ramp_rate16");
    assert_eq!(reason, expected);
    assert_eq!(
        verify_manifest(
            Manifest {
                fixtures: vec![fixture]
            },
            |_| panic!("unsupported case reached verifier")
        ),
        Err(CrosscheckFailure::NoVerifiedFixtures { skipped: 1 })
    );
}

#[rstest]
#[case::empty_name(serde_json::json!({"name": ""}), MalformedDescription::InvalidName(String::new()))]
#[case::path_name(serde_json::json!({"name": "../outside"}), MalformedDescription::InvalidName("../outside".into()))]
#[case::empty_shape(serde_json::json!({"shape": []}), MalformedDescription::InvalidShape(vec![]))]
#[case::zero_dimension(serde_json::json!({"shape": [0]}), MalformedDescription::InvalidShape(vec![0]))]
#[case::unsupported_mode_with_invalid_shape(serde_json::json!({"shape": [0], "mode": "precision"}), MalformedDescription::InvalidShape(vec![0]))]
#[case::size_overflow(serde_json::json!({"shape": [usize::MAX, 2]}), MalformedDescription::SizeOverflow)]
#[case::zero_rate(serde_json::json!({"rate": 0.0}), MalformedDescription::InvalidRate(FormatError::InvalidZfpRate { rate_bits: 0.0f64.to_bits() }))]
#[case::raw_length(serde_json::json!({"raw_bytes_len": 63}), MalformedDescription::RawLength { expected: 64, recorded: 63 })]
#[case::empty_reference(serde_json::json!({"compressed_bytes_len": 0}), MalformedDescription::EmptyReference)]
#[case::wide_scalar_rate(serde_json::json!({"rate": 64.0}), MalformedDescription::InvalidSettings(FilterError::ZfpConfiguration(FormatError::InvalidZfpRateForScalar { rate_bits: 64.0f64.to_bits(), scalar_bits: 32 })))]
fn malformed_descriptions_fail_before_verification(
    #[case] changes: serde_json::Value,
    #[case] expected: MalformedDescription,
) {
    let fixture = fixture_description(changes);
    let name = fixture.name.clone();
    assert_eq!(
        verify_manifest(
            Manifest {
                fixtures: vec![fixture]
            },
            |_| panic!("malformed case reached verifier")
        ),
        Err(CrosscheckFailure::Malformed {
            name,
            reason: expected
        })
    );
}

#[rstest]
#[case::unknown_mode(
    "mode",
    "rtae",
    "unknown variant `rtae`, expected one of `accuracy`, `precision`, `rate`, `reversible`"
)]
#[case::unknown_scalar(
    "dtype",
    "float32",
    "unknown variant `float32`, expected one of `f32`, `f64`, `i16`, `i32`, `i64`, `i8`, `u16`, `u32`, `u64`, `u8`"
)]
fn unknown_description_values_are_malformed(
    #[case] field: &str,
    #[case] value: &str,
    #[case] expected: &str,
) {
    let mut description = fixture_json();
    description[field] = serde_json::json!(value);
    let error = serde_json::from_value::<Fixture>(description).unwrap_err();
    assert_eq!(error.classify(), serde_json::error::Category::Data);
    assert_eq!(error.to_string(), expected);
}

#[rstest]
#[case::nothing_verified(0)]
#[case::one_verified(1)]
fn verifier_failures_cannot_be_counted_as_verified_evidence(#[case] successful: usize) {
    let fixture = fixture_description(serde_json::json!({}));
    let mut calls = 0;
    let result = verify_manifest(
        Manifest {
            fixtures: vec![fixture.clone(), fixture],
        },
        |fixture| {
            calls += 1;
            if calls <= successful {
                fixture.verify()
            } else {
                Err("injected verifier failure".into())
            }
        },
    );
    assert_eq!(calls, 2);
    assert_eq!(
        result,
        Err(CrosscheckFailure::VerificationFailed(vec![
            (
                "f32_1d_16_ramp_rate16".into(),
                "injected verifier failure".into()
            );
            2 - successful
        ]))
    );
}

#[rstest]
#[case::finite_f32([0.0f32, 1.0], [0.25f32, 0.5], 0.5)]
#[case::finite_f64([0.0f64, 1.0], [0.25f64, 0.5], 0.5)]
#[case::reference_nan_f32([f32::NAN, 1.0], [0.0f32, 1.0], f64::INFINITY)]
#[case::reference_nan_f64([f64::NAN, 1.0], [0.0f64, 1.0], f64::INFINITY)]
#[case::reconstructed_nan_f32([0.0f32, 1.0], [f32::NAN, 1.0], f64::INFINITY)]
#[case::reconstructed_nan_f64([0.0f64, 1.0], [f64::NAN, 1.0], f64::INFINITY)]
#[case::equal_infinities_f32([f32::INFINITY, 1.0], [f32::INFINITY, 1.0], f64::INFINITY)]
#[case::equal_infinities_f64([f64::INFINITY, 1.0], [f64::INFINITY, 1.0], f64::INFINITY)]
#[case::difference_overflow([f64::MAX, 1.0], [-f64::MAX, 1.0], f64::INFINITY)]
fn reconstruction_error_retains_non_finite_differences<T: Copy + Into<f64>>(
    #[case] reference: [T; 2],
    #[case] reconstructed: [T; 2],
    #[case] expected: f64,
) {
    assert_eq!(maximum_error(&reference, &reconstructed), expected);
}

fn fixture_description(changes: serde_json::Value) -> Fixture {
    let mut description = fixture_json();
    for (key, value) in changes.as_object().unwrap() {
        description[key] = value.clone();
    }
    serde_json::from_value(description).unwrap()
}

fn fixture_json() -> serde_json::Value {
    serde_json::json!({
        "name": "f32_1d_16_ramp_rate16", "dtype": "f32", "shape": [16], "rate": 16.0,
        "mode": "rate", "filter_name": "H5Z-ZFP-1.1.1 (ZFP-1.0.1)", "cd_values_u32": [],
        "raw_bytes_len": 64, "compressed_bytes_len": 32, "notes": "",
    })
}
