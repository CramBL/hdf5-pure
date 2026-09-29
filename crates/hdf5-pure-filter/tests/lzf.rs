use hdf5_pure_filter::Error;

#[test]
fn the_codec_round_trips_builds_h5py_cd_values_and_rejects_a_truncated_literal_run() {
    let stream = [4, b'a', b'b', b'c', b'd', b'e', 3 << 5, 4];
    assert_eq!(
        hdf5_pure_filter::decompress_lzf(&stream, Some(10)).unwrap(),
        b"abcdeabcde"
    );

    let raw = b"abcdeabcde";
    let encoded = hdf5_pure_filter::compress_lzf(raw);
    assert_eq!(
        hdf5_pure_filter::decompress_lzf(&encoded, Some(raw.len())).unwrap(),
        raw
    );
    assert_eq!(
        hdf5_pure_filter::lzf_h5py_cd_values(2, &[3, 4]),
        [4, 0x0105, 24]
    );
    assert_eq!(
        hdf5_pure_filter::decompress_lzf(&[10, b'x'], None).unwrap_err(),
        Error::InvalidLzfStream("truncated literal run")
    );
}
