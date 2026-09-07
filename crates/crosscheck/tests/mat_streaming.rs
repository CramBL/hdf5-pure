// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! The reference C library must read a `.mat` written by the streaming path.
//!
//! This exercises the combination nothing else does: a userblock-offset file
//! assembled front-to-back onto a non-seekable sink, whose dataset's contiguous
//! data region was filled block by block from a producer during that same pass.
//! The pure reader agreeing proves the bytes are self-consistent; the C library
//! agreeing proves they are HDF5.
//!
//! Every call here goes through the safe `hdf5-metno` API, which serializes its
//! own C calls through an internal lock, so these tests need no extra guard.

use hdf5_pure::mat::{Block, Blocking, DataProducer, MatBuilder, MatError, Options};
use tempfile::tempdir;

/// Generates `f64` elements from their linear index, so a multi-block dataset
/// costs nothing to produce and its expected values are computable.
struct Ramp;

impl Ramp {
    fn value(i: u64) -> f64 {
        i as f64 * 0.25
    }
}

impl DataProducer for Ramp {
    fn block_bytes(&self, block: Block, out: &mut Vec<u8>) -> Result<(), MatError> {
        for i in 0..block.elements {
            out.extend_from_slice(&Self::value(block.first_element + i).to_le_bytes());
        }
        Ok(())
    }
}

#[test]
fn c_reads_a_produced_dataset_streamed_to_disk() {
    // Several blocks with a short last one, so the C library sees a region the
    // producer filled in pieces rather than a single-block special case.
    const DIMS: [usize; 2] = [4, 100_001];
    let n = (DIMS[0] * DIMS[1]) as u64;

    let dir = tempdir().unwrap();
    let path = dir.path().join("produced.mat");

    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    assert!(blocking.block_count > 2, "the fixture must span blocks");
    assert!(blocking.last_block_elements < blocking.block_elements);

    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("samples", &DIMS, Box::new(Ramp))
        .unwrap();
    mb.write_f64("meta", &[1, 2], &[1.0, 2.0]).unwrap();
    mb.write(&path).unwrap();

    let file = hdf5::File::open(&path).expect("the C library should open the streamed file");
    let ds = file.dataset("samples").expect("dataset should exist");
    // HDF5 storage shape is the MATLAB shape reversed.
    assert_eq!(ds.shape(), vec![DIMS[1], DIMS[0]]);

    let values = ds.read_raw::<f64>().expect("the C library should read it");
    assert_eq!(values.len() as u64, n);
    assert_eq!(values[0], Ramp::value(0));
    // Straddle the first block boundary, where a mis-sized block would show up
    // as a discontinuity rather than as a missing dataset.
    let boundary = blocking.block_elements;
    assert_eq!(values[boundary as usize - 1], Ramp::value(boundary - 1));
    assert_eq!(values[boundary as usize], Ramp::value(boundary));
    assert_eq!(values[(n - 1) as usize], Ramp::value(n - 1));

    // The MATLAB class attribute survives the streaming path. It is a
    // fixed-length ASCII string, which is what MATLAB itself writes.
    let class: hdf5::types::FixedAscii<6> = ds.attr("MATLAB_class").unwrap().read_scalar().unwrap();
    assert_eq!(class.as_str(), "double");
}
