//! Compressing dataset storage with deflate, shuffle, scale-offset, and LZF.
//!
//! Deflate, shuffle, and scale-offset are built-in HDF5 filters, so those
//! files stay readable by libhdf5, h5py, and MATLAB. LZF is
//! h5py's registered filter (id 32000), which h5py reads with no plugin.
//! Compression is transparent on read: the same `read_*` call returns the
//! decoded data regardless of how it was stored.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example compression
//! ```

use hdf5_pure::DeflateLevel;
use hdf5_pure::{File, FileBuilder, ScaleOffset};

fn main() {
    // A compressible signal: a smooth ramp with mild structure. Real data
    // compresses to varying degrees; this is just enough to be illustrative.
    let n = 10_000usize;
    let signal: Vec<f64> = (0..n).map(|i| (i as f64 / 50.0).sin() * 1000.0).collect();
    let counts: Vec<i32> = (0..n as i32).map(|i| 1000 + (i % 7)).collect();

    // Filters apply per chunk, so a chunked layout is required to compress.
    let chunk = &[1000u64];

    let uncompressed = build(|b| {
        b.create_dataset("signal").with_f64_data(&signal);
    });

    let deflated = build(|b| {
        b.create_dataset("signal")
            .with_f64_data(&signal)
            .with_chunks(chunk)
            .with_deflate(DeflateLevel::Six);
    });

    // Shuffle reorders bytes so like-significance bytes sit together, which
    // usually helps the following deflate pass.
    let shuffled = build(|b| {
        b.create_dataset("signal")
            .with_f64_data(&signal)
            .with_chunks(chunk)
            .with_shuffle()
            .with_deflate(DeflateLevel::Six);
    });

    // LZF trades compression ratio for speed; like deflate it benefits from a
    // preceding shuffle pass.
    let lzf = build(|b| {
        b.create_dataset("signal")
            .with_f64_data(&signal)
            .with_chunks(chunk)
            .with_shuffle()
            .with_lzf();
    });

    // Scale-offset in integer mode is lossless: each chunk's values are stored
    // as offsets from its minimum, packed into the fewest bits the range needs.
    let scale_offset_int = build(|b| {
        b.create_dataset("counts")
            .with_i32_data(&counts)
            .with_chunks(chunk)
            .with_scale_offset(ScaleOffset::Integer(0)); // 0 = pick bit width per chunk
    });

    println!("dataset storage (whole-file bytes, lower is better):");
    report("f64 uncompressed", uncompressed.len());
    report("f64 deflate(6)", deflated.len());
    report("f64 shuffle+deflate(6)", shuffled.len());
    report("f64 shuffle+lzf (lossless)", lzf.len());
    report("i32 scale-offset (lossless)", scale_offset_int.len());

    // Reads are identical regardless of filtering, and the lossless paths
    // reproduce the input exactly.
    let back = File::from_bytes(shuffled)
        .unwrap()
        .dataset("signal")
        .unwrap()
        .read_f64()
        .unwrap();
    assert_eq!(back, signal);

    let back_lzf = File::from_bytes(lzf)
        .unwrap()
        .dataset("signal")
        .unwrap()
        .read_f64()
        .unwrap();
    assert_eq!(back_lzf, signal);

    let back_counts = File::from_bytes(scale_offset_int)
        .unwrap()
        .dataset("counts")
        .unwrap()
        .read_i32()
        .unwrap();
    assert_eq!(back_counts, counts);

    // Float D-scale is the lossy mode: values are rounded to N decimal digits
    // before packing, so the read-back is close but not exact.
    let dscale = build(|b| {
        b.create_dataset("signal")
            .with_f64_data(&signal)
            .with_chunks(chunk)
            .with_scale_offset(ScaleOffset::FloatDScale(2)) // keep 2 decimal digits
            .with_deflate(DeflateLevel::Six);
    });
    report("\nf64 D-scale(2)+deflate (lossy)", dscale.len());
    let lossy = File::from_bytes(dscale)
        .unwrap()
        .dataset("signal")
        .unwrap()
        .read_f64()
        .unwrap();
    let max_err = signal
        .iter()
        .zip(&lossy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("D-scale(2) max abs error: {max_err:.4} (bounded by the kept digits)");

    println!("\nlossless filters verified to reproduce the input exactly");
}

/// Build a file from a configuring closure and return its bytes.
fn build(configure: impl FnOnce(&mut FileBuilder)) -> Vec<u8> {
    let mut b = FileBuilder::new();
    configure(&mut b);
    b.finish().unwrap()
}

fn report(label: &str, bytes: usize) {
    println!("  {label:<32} {bytes:>8} bytes");
}
