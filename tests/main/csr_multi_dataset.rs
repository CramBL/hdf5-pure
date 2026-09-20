//! Regression test for issue #4: reading multiple chunked datasets in the
//! same file (e.g. an h5ad CSR triple `X/data`, `X/indices`, `X/indptr`)
//! returned wrong values for every dataset after the first because the
//! file-level chunk-cache index was keyed by chunk coordinate only.

use hdf5_pure::{File, FileBuilder};

#[test]
fn csr_triple_chunked_read() {
    let mut builder = FileBuilder::new();
    let mut grp = builder.create_group("X");

    let data: Vec<f32> = (0..200).map(|i| i as f32 * 0.5).collect();
    let indices: Vec<i32> = (0..200).map(|i| i + 1000).collect();
    let indptr: Vec<i32> = (0..50).map(|i| i * 4).collect();

    grp.create_dataset("data")
        .with_f32_data(&data)
        .with_chunks(&[64]);
    grp.create_dataset("indices")
        .with_i32_data(&indices)
        .with_chunks(&[64]);
    grp.create_dataset("indptr")
        .with_i32_data(&indptr)
        .with_chunks(&[16]);
    builder.add_group(grp.finish());

    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    let read_data = file.dataset("X/data").unwrap().read_f32().unwrap();
    let read_indices = file.dataset("X/indices").unwrap().read_i32().unwrap();
    let read_indptr = file.dataset("X/indptr").unwrap().read_i32().unwrap();

    assert_eq!(read_data, data, "X/data mismatch");
    assert_eq!(
        read_indices, indices,
        "X/indices mismatch (first regression)"
    );
    assert_eq!(read_indptr, indptr, "X/indptr mismatch (second regression)");
}

#[test]
fn csr_triple_compressed_chunked_read() {
    let mut builder = FileBuilder::new();
    let mut grp = builder.create_group("X");

    let data: Vec<f32> = (0..200).map(|i| i as f32 * 0.25).collect();
    let indices: Vec<i32> = (0..200).map(|i| i + 5000).collect();
    let indptr: Vec<i32> = (0..50).map(|i| i * 4).collect();

    grp.create_dataset("data")
        .with_f32_data(&data)
        .with_chunks(&[64])
        .with_deflate(4);
    grp.create_dataset("indices")
        .with_i32_data(&indices)
        .with_chunks(&[64])
        .with_deflate(4);
    grp.create_dataset("indptr")
        .with_i32_data(&indptr)
        .with_chunks(&[16])
        .with_deflate(4);
    builder.add_group(grp.finish());

    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    let read_data = file.dataset("X/data").unwrap().read_f32().unwrap();
    let read_indices = file.dataset("X/indices").unwrap().read_i32().unwrap();
    let read_indptr = file.dataset("X/indptr").unwrap().read_i32().unwrap();

    assert_eq!(read_data, data);
    assert_eq!(read_indices, indices);
    assert_eq!(read_indptr, indptr);
}
