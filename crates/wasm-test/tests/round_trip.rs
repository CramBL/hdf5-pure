#![cfg(target_arch = "wasm32")]

use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::Layout;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn reads_back_a_contiguous_dataset() {
    let values = [1.0, 2.5, -3.0, 4.25];
    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_f64_data(&values);

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("x").unwrap();

    let layout = dataset.layout().unwrap();
    let Layout::Contiguous {
        address: Some(_),
        size,
    } = &layout
    else {
        panic!("expected an allocated contiguous layout, got {layout:?}");
    };
    assert_eq!(*size, 32);
    assert_eq!(dataset.shape().unwrap(), vec![4]);
    assert_eq!(dataset.read_f64().unwrap(), values);
}

#[wasm_bindgen_test]
fn reads_back_a_deflate_compressed_dataset() {
    let values: Vec<i32> = (0..4096).map(|i| i % 16).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("x")
        .with_i32_data(&values)
        .with_chunks(&[1024])
        .with_deflate(DeflateLevel::Six);

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("x").unwrap();

    let layout = dataset.layout().unwrap();
    let Layout::Chunked { chunk_shape, .. } = &layout else {
        panic!("expected a chunked layout, got {layout:?}");
    };
    assert_eq!(chunk_shape, &vec![1024]);
    let filters: Vec<String> = dataset
        .filter_pipeline()
        .unwrap()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(filters, vec!["deflate(6)"]);
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 4);
    for chunk in &chunks {
        assert!(chunk.storage_size < 1024 * 4, "{chunk:?}");
    }
    assert_eq!(dataset.read_i32().unwrap(), values);
}
