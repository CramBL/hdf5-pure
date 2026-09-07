//! Crosscheck the pure-Rust LZF codec against fixtures produced by h5py's
//! built-in LZF filter.
//!
//! Fixtures and generator script live in `tests/data/h5py/lzf/`. Regenerate:
//!
//!     tests/data/h5py/lzf/.venv/bin/python tests/data/h5py/lzf/regen.py
//!
//! Two directions are checked, and they are not equally strong.
//!
//! **Read.** For every single-filter LZF fixture with an extracted chunk
//! stream: an unmasked chunk decodes through our codec to exactly the raw
//! bytes (LZF is lossless), and a masked chunk (h5py registers LZF as optional
//! and stores an incompressible chunk raw) is byte-equal to the raw data.
//!
//! **Write.** For *every* fixture, the filter pipeline our writer builds for
//! the same dataset geometry equals the one h5py recorded: same filter ids in
//! the same order, same name, same optional flag, same cd_values. The codec
//! stream is deliberately not byte-compared — any conforming stream is valid
//! (see the module doc in `lzf.rs`) — so what pins our encoder here is only
//! that its output round-trips through our own decoder. That h5py can decode
//! it is verified at fixture-regeneration time instead, by the read-back phase
//! of `regen.py`, which needs a live h5py this test does not have.
//!
//! Chained (shuffle+lzf) and multi-chunk fixtures are covered end-to-end in
//! `tests/lzf_roundtrip.rs`, which opens their `.h5` files through the full
//! reader.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

use crate::chunked_write::{ChunkOptions, FilterKind};
use crate::filter_pipeline::{FILTER_LZF, FILTER_SHUFFLE};
use crate::lzf;

#[derive(Debug, Deserialize)]
struct Manifest {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    /// Element type, as one of `regen.py`'s `NP_DTYPE` keys.
    dtype: String,
    chunk_shape: Vec<u64>,
    filters: Vec<u16>,
    /// h5py's flags word on the LZF filter: 1 (`H5Z_FLAG_OPTIONAL`).
    lzf_flags: u16,
    cd_values_u32: Vec<u32>,
    /// Measured per-chunk mask; absent for multi-chunk fixtures, whose
    /// manifest records only whole-dataset facts.
    filter_mask: Option<u32>,
    raw_bytes_len: usize,
    compressed_bytes_len: Option<usize>,
}

impl Fixture {
    fn element_size(&self) -> u32 {
        match self.dtype.as_str() {
            "u8" | "i8" => 1,
            "i16" | "u16" => 2,
            "i32" | "u32" | "f32" => 4,
            "i64" | "u64" | "f64" => 8,
            other => panic!("{}: unhandled manifest dtype {other}", self.name),
        }
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/h5py/lzf")
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

/// Our writer's filter pipeline for a fixture's geometry is the one h5py
/// recorded for it, field by field.
///
/// This is the whole of the write-direction crosscheck that can run without
/// h5py present, and it covers every fixture rather than only the
/// codec-comparable ones — including `i64_multichunk`, the only geometry whose
/// `cd_values[2]` (8 x 128) distinguishes the element size from the chunk
/// dimensions. A single-chunk-of-bytes fixture cannot tell those apart, since
/// both orderings multiply out to the same product.
#[test]
fn writer_pipeline_matches_h5py() {
    let manifest = load_manifest();
    assert!(
        !manifest.fixtures.is_empty(),
        "manifest.json lists no fixtures; run regen.py"
    );

    for fix in &manifest.fixtures {
        assert!(
            fix.filters.contains(&FILTER_LZF),
            "{}: fixture is not an LZF fixture",
            fix.name
        );

        let mut options = ChunkOptions {
            chunk_dims: Some(fix.chunk_shape.clone()),
            ..ChunkOptions::default()
        };
        if fix.filters.contains(&FILTER_SHUFFLE) {
            options.set_filter(FilterKind::Shuffle);
        }
        options.set_filter(FilterKind::Lzf);
        let pipeline = options
            .build_pipeline(
                &crate::filters::ChunkContext::basic(&fix.chunk_shape, fix.element_size()),
                crate::fill_value::FillPattern::ZERO,
            )
            .unwrap_or_else(|e| panic!("{}: build_pipeline: {e:?}", fix.name))
            .unwrap_or_else(|| panic!("{}: writer produced no pipeline", fix.name));

        let ids: Vec<u16> = pipeline.filters.iter().map(|f| f.filter_id).collect();
        assert_eq!(ids, fix.filters, "{}: filter ids or order", fix.name);

        let ours = pipeline
            .filters
            .iter()
            .find(|f| f.filter_id == FILTER_LZF)
            .expect("lzf requested");
        assert_eq!(ours.name.as_deref(), Some("lzf"), "{}: name", fix.name);
        assert_eq!(ours.flags, fix.lzf_flags, "{}: flags", fix.name);
        assert_eq!(
            ours.client_data, fix.cd_values_u32,
            "{}: cd_values",
            fix.name
        );
    }
}

#[test]
fn lzf_crosscheck() {
    let manifest = load_manifest();
    let mut verified = 0usize;

    for fix in &manifest.fixtures {
        // Only single-filter LZF fixtures with an extracted single-chunk
        // stream are codec-comparable here; the rest are exercised through
        // the full reader in tests/lzf_roundtrip.rs.
        if fix.filters != [FILTER_LZF] || fix.compressed_bytes_len.is_none() {
            continue;
        }
        let raw = read_bin(&format!("{}.raw.bin", fix.name));
        let stored = read_bin(&format!("{}.compressed.bin", fix.name));
        assert_eq!(raw.len(), fix.raw_bytes_len, "{}: manifest drift", fix.name);

        if fix.filter_mask.unwrap_or(0) & 1 == 1 {
            // h5py skipped its optional LZF: the chunk is stored raw.
            assert_eq!(stored, raw, "{}: masked chunk must be raw", fix.name);
        } else {
            let decoded = lzf::decompress(&stored, Some(raw.len()))
                .unwrap_or_else(|e| panic!("{}: reference stream decode: {e:?}", fix.name));
            assert_eq!(decoded, raw, "{}: decode mismatch", fix.name);
        }

        // Our stream need not equal liblzf's, but it must round-trip.
        let ours = lzf::compress(&raw);
        assert_eq!(
            lzf::decompress(&ours, Some(raw.len())).unwrap(),
            raw,
            "{}: our stream fails to round-trip",
            fix.name
        );
        verified += 1;
    }

    // A green result with zero positively-verified fixtures is not a pass at
    // all: every assertion above sits behind the `continue`, so a manifest
    // that stopped recording extracted streams would leave this test green
    // while checking nothing.
    assert!(
        verified > 0,
        "LZF crosscheck verified 0 of {} manifest fixtures against h5py; \
         no codec-comparable stream was found. Run regen.py.",
        manifest.fixtures.len()
    );
}
