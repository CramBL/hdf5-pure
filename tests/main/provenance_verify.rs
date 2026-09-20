//! Public-API tests for provenance verification (`Dataset::verify_provenance`).
#![cfg(feature = "provenance")]

use hdf5_pure::{AttrValue, File, FileBuilder, VerifyResult};

// The attribute name is an internal constant; mirror it here so the test
// exercises the crate strictly through its public surface.
const ATTR_SHA256: &str = "_provenance_sha256";

#[test]
fn verify_ok_on_roundtrip() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("sensor")
        .with_f64_data(&(0..24).map(|v| v as f64).collect::<Vec<_>>())
        .with_provenance("test-suite", "2026-02-19T12:00:00Z", None);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("sensor").unwrap();
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::Ok);
}

#[test]
fn verify_no_hash_without_provenance() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("plain").with_f64_data(&[1.0, 2.0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("plain").unwrap();
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::NoHash);
}

#[test]
fn verify_mismatch_on_wrong_stored_hash() {
    // Store a deliberately bogus hash directly (rather than via
    // `with_provenance`, which would compute the correct one) so the recomputed
    // digest is guaranteed to differ.
    let bogus = "0".repeat(64);
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .set_attr(ATTR_SHA256, AttrValue::String(bogus.clone()));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    match ds.verify_provenance().unwrap() {
        VerifyResult::Mismatch { stored, computed } => {
            assert_eq!(stored, bogus);
            assert_ne!(computed, bogus);
            assert_eq!(computed.len(), 64);
        }
        other => panic!("expected Mismatch, got {other:?}"),
    }
}
