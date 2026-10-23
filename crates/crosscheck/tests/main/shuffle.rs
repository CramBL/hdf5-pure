#![cfg(feature = "hdf5")]

use hdf5::filters::Filter;
use hdf5_pure::File;

#[test]
fn fletcher32_then_shuffle_reads_c_created_chunk_with_remainder() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fletcher32_shuffle.h5");
    let data: Vec<f64> = (0..64).map(|i| f64::from(i) + 0.25).collect();
    {
        let file = hdf5::File::create(&path).unwrap();
        let dataset = file
            .new_dataset::<f64>()
            .shape([64])
            .chunk([64])
            .set_filters(&[Filter::Fletcher32, Filter::Shuffle])
            .create("data")
            .unwrap();
        dataset.write_raw(&data).unwrap();
        assert_eq!(dataset.filters(), vec![Filter::Fletcher32, Filter::Shuffle]);
        assert_eq!(dataset.storage_size(), 516);
        assert_eq!(dataset.read_raw::<f64>().unwrap(), data);
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let dataset = file.dataset("data").unwrap();
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].storage_size, 516);
    assert_eq!(chunks[0].filter_mask().raw_bits(), 0);
    assert_eq!(dataset.read_f64().unwrap(), data);
}
