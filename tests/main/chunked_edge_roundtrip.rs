//! Roundtrip tests for chunked datasets whose chunk dimensions do *not* evenly
//! divide the dataset dimensions, so chunks overhang the dataset edge in both
//! the innermost (contiguous) and outer dimensions. This is the case the bulk
//! row-copy assembly kernel (`copy_chunk_to_output`) must get exactly right: it
//! clamps the inner row and skips out-of-bounds outer rows. Even/odd ranks,
//! several element types, and an uncompressed vs shuffle+deflate path are all
//! covered, comparing the read-back against the known row-major input.

use hdf5_pure::{File, FileBuilder};

/// Write a flat row-major `f64` dataset with the given shape and chunk shape,
/// read it back, and assert it equals the input.
fn roundtrip_f64(shape: &[u64], chunks: &[u64], shuffle_deflate: bool) {
    let total: u64 = shape.iter().product();
    let data: Vec<f64> = (0..total).map(|i| i as f64 * 1.5 - 7.0).collect();

    let mut b = FileBuilder::new();
    {
        let ds = b
            .create_dataset("d")
            .with_f64_data(&data)
            .with_shape(shape)
            .with_chunks(chunks);
        if shuffle_deflate {
            ds.with_shuffle().with_deflate(6);
        }
    }
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(
        ds.shape().unwrap(),
        shape,
        "shape {shape:?} chunks {chunks:?}"
    );
    assert_eq!(
        ds.read_f64().unwrap(),
        data,
        "f64 roundtrip mismatch for shape {shape:?} chunks {chunks:?} sd={shuffle_deflate}"
    );
}

#[test]
fn edge_chunks_1d() {
    // 25 elements, chunk 10 => last chunk overhangs (inner-dim overhang).
    roundtrip_f64(&[25], &[10], false);
    roundtrip_f64(&[25], &[10], true);
    // Chunk larger than the dataset (single overhanging chunk).
    roundtrip_f64(&[7], &[16], false);
}

#[test]
fn edge_chunks_2d() {
    // 5x7 with 2x3 chunks: rows {0,2,4} (row-4 chunk overhangs row 5 -> outer
    // overhang) and cols {0,3,6} (col-6 chunk overhangs cols 7,8 -> inner
    // overhang). Both edge cases in one fixture.
    roundtrip_f64(&[5, 7], &[2, 3], false);
    roundtrip_f64(&[5, 7], &[2, 3], true);
    // Tall-and-thin and wide-and-short variants.
    roundtrip_f64(&[13, 4], &[4, 4], false);
    roundtrip_f64(&[4, 13], &[4, 4], false);
}

#[test]
fn edge_chunks_3d() {
    // 3x3x5 with 2x2x2 chunks: overhang in all three dimensions.
    roundtrip_f64(&[3, 3, 5], &[2, 2, 2], false);
    roundtrip_f64(&[3, 3, 5], &[2, 2, 2], true);
}

#[test]
fn edge_chunks_integer_types() {
    // Exercise the assembly kernel together with the integer decode fast paths
    // for both a narrowing read (i32 stored, read as i32) and a wide read.
    let shape = [5u64, 7];
    let total: u64 = shape.iter().product();

    let i32_data: Vec<i32> = (0..total as i32).map(|i| i.wrapping_mul(99) - 5).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&i32_data)
        .with_shape(&shape)
        .with_chunks(&[2, 3])
        .with_shuffle()
        .with_deflate(6);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), i32_data);

    let u16_data: Vec<u16> = (0..total).map(|i| ((i * 137) & 0xFFFF) as u16).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u16_data(&u16_data)
        .with_shape(&shape)
        .with_chunks(&[3, 2]);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_u16().unwrap(), u16_data);
}
