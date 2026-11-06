use hdf5_pure::ChunkCacheConfig;
use hdf5_pure::FileAccessProperties;
use rstest::rstest;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
fn an_opened_image_reads_back_with_its_access_properties(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let properties = FileAccessProperties::new()
        .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(0).with_max_slots(0));
    let file = backend
        .open_image_with_options(test_util::legacy::simple_dataset(), properties)
        .unwrap();
    assert_eq!(file.access_properties(), properties);
    assert_eq!(
        file.dataset("data").unwrap().read_f64().unwrap(),
        [1.0, 2.0, 3.0]
    );
    file.close().unwrap();
}
