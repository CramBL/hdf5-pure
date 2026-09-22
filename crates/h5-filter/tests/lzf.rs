use h5_filter::Error;

#[test]
fn exported_lzf_codec_and_parameters() {
    let stream = [4, b'a', b'b', b'c', b'd', b'e', 3 << 5, 4];
    assert_eq!(
        h5_filter::decompress_lzf(&stream, Some(10)).unwrap(),
        b"abcdeabcde"
    );

    let raw = b"abcdeabcde";
    let encoded = h5_filter::compress_lzf(raw);
    assert_eq!(
        h5_filter::decompress_lzf(&encoded, Some(raw.len())).unwrap(),
        raw
    );
    assert_eq!(h5_filter::lzf_h5py_cd_values(2, &[3, 4]), [4, 0x0105, 24]);
    assert_eq!(
        h5_filter::decompress_lzf(&[10, b'x'], None).unwrap_err(),
        Error::InvalidLzfStream("truncated literal run")
    );
}
