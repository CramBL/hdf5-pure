// Links the reference HDF5 C library (`hdf5-metno`), gated to 64-bit little-endian
// targets; skip on 32-bit so `cross test --target i686-...` stays pure-Rust.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Cross-validation for Extensible-Array-indexed chunked datasets (one unlimited
//! dimension), in both directions against the reference C HDF5 library.
//!
//! These guard the EA block-size geometry, super blocks, and paged data blocks.
//! Sizes are chosen to span every structural range:
//!   - 20      inline (4) + first direct data block (16)
//!   - 300     all 6 direct data blocks + the first super block
//!   - 2000    several super blocks
//!   - 50000   deeper super-block nesting
//!   - 140000  past 131060 chunks: paged data blocks
//!
//! Before the geometry fix, anything past 20 chunks silently corrupted on read.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

const SIZES: &[usize] = &[20, 300, 2000, 50000, 140000];

/// Create a 1-D unlimited, chunked i32 dataset with the reference C library,
/// using the latest format so the chunk index is an Extensible Array.
fn write_with_c(path: &std::path::Path, n: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((1,))
        .shape((Extent::resizable(n),))
        .create("d")
        .unwrap();
    let data: Vec<i32> = (0..n as i32).collect();
    ds.write(&data).unwrap();
    file.close().unwrap();
}

/// Create the same dataset with hdf5-pure.
fn write_with_pure(path: &std::path::Path, n: usize) {
    let data: Vec<i32> = (0..n as i32).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[1]);
    b.write(path).unwrap();
}

/// Direction 1: hdf5-pure writes, the reference C library reads.
#[test]
fn pure_writes_c_reads() {
    for &n in SIZES {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ea.h5");
        write_with_pure(&path, n);

        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        let values = ds.read_raw::<i32>().unwrap();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(values.len(), n, "C read wrong length for n={n}");
        assert_eq!(values, expected, "C read wrong data for n={n}");
    }
}

/// Direction 2: the reference C library writes, hdf5-pure reads.
#[test]
fn c_writes_pure_reads() {
    for &n in SIZES {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ea.h5");
        write_with_c(&path, n);

        let bytes = std::fs::read(&path).unwrap();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("d").unwrap();
        let values = ds.read_i32().unwrap();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(values.len(), n, "hdf5-pure read wrong length for n={n}");
        assert_eq!(values, expected, "hdf5-pure read wrong data for n={n}");
    }
}

/// The EA header's six statistics fields must match the C library byte-for-byte
/// (the C library recomputes and would reject inconsistent computed stats; the
/// stored max_idx_set / nelmts must also agree). Compares the headers of the two
/// files for identical datasets.
#[test]
fn eahd_stats_match_c() {
    fn stats_of(path: &std::path::Path) -> Vec<u64> {
        let b = std::fs::read(path).unwrap();
        let h = (0..b.len() - 4).find(|&i| &b[i..i + 4] == b"EAHD").unwrap();
        (0..6)
            .map(|k| {
                let p = h + 12 + k * 8;
                u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
            })
            .collect()
    }

    for &n in SIZES {
        let dir = tempdir().unwrap();
        let c_path = dir.path().join("c.h5");
        let pure_path = dir.path().join("pure.h5");
        write_with_c(&c_path, n);
        write_with_pure(&pure_path, n);
        assert_eq!(
            stats_of(&pure_path),
            stats_of(&c_path),
            "EAHD stats differ from the C library at n={n}"
        );
    }
}

/// The same six statistics, published by the *in-place* append path rather than
/// by the whole-file writer.
///
/// `Located::update_ea_header` writes them on every append, and until this test
/// nothing read them back: zeroing `nelmts` or transposing two fields passed the
/// entire suite, C-library crosschecks included, because the reference library
/// recomputes the geometry from the block sizes rather than trusting these
/// fields on read. They are still what `h5stat` and `H5EA__hdr_debug` report and
/// what a future reader may believe, so they are checked against the one oracle
/// that cannot be wrong about them: the same dataset written whole, by the C
/// library, at the size the append grew it to.
///
/// Sizes are small on purpose — this grows the file one append at a time — but
/// span the inline elements, the first direct data blocks, and the first super
/// block, which is where the six values stop being constants.
#[test]
fn eahd_stats_match_c_after_an_in_place_append() {
    fn stats_of(path: &std::path::Path) -> Vec<u64> {
        let b = std::fs::read(path).unwrap();
        let h = (0..b.len() - 4).find(|&i| &b[i..i + 4] == b"EAHD").unwrap();
        (0..6)
            .map(|k| {
                let p = h + 12 + k * 8;
                u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
            })
            .collect()
    }

    for &(start, grown) in &[(4usize, 20usize), (16, 300), (300, 700)] {
        let dir = tempdir().unwrap();
        let c_path = dir.path().join("c.h5");
        let pure_path = dir.path().join("pure.h5");
        write_with_c(&c_path, grown);
        write_with_pure(&pure_path, start);

        {
            let file = File::open_rw(&pure_path).unwrap();
            let mut ds = file.dataset("d").unwrap();
            let more: Vec<i32> = (start as i32..grown as i32).collect();
            ds.append(&more).unwrap();
            file.close().unwrap();
        }

        assert_eq!(
            stats_of(&pure_path),
            stats_of(&c_path),
            "EAHD stats after growing {start} -> {grown} in place differ from the C library's \
             for a dataset written whole at {grown}"
        );
        // The append itself must still be right, or the stats above would be
        // describing a file nobody can read.
        let file = hdf5::File::open(&pure_path).unwrap();
        let values = file.dataset("d").unwrap().read_raw::<i32>().unwrap();
        assert_eq!(values, (0..grown as i32).collect::<Vec<_>>());
    }
}
