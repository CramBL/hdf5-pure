#![cfg(feature = "zfp")]

use h5_filter::Error;
use h5_filter::ZfpElementType;

#[test]
fn exported_codec_and_filter_entry_points_encode_the_same_chunk() {
    let raw = [0u8; 16];
    let cd_values = h5_filter::zfp_cd_values_rate(16.0, ZfpElementType::F32, &[4]).unwrap();
    let encoded = h5_filter::compress_zfp(&raw, &[4], 16.0, ZfpElementType::F32).unwrap();

    assert_eq!(encoded, vec![0u8; 8]);
    assert_eq!(
        h5_filter::compress_zfp_filter(&raw, &cd_values, &[4], Some(ZfpElementType::F32)).unwrap(),
        encoded
    );
    assert_eq!(
        h5_filter::decompress_zfp_filter(&encoded, &cd_values, &[4], Some(ZfpElementType::F32))
            .unwrap(),
        raw
    );
}

#[test]
fn exported_filter_rejects_truncated_input_and_invalid_parameters() {
    let cd_values = h5_filter::zfp_cd_values_rate(16.0, ZfpElementType::F32, &[4]).unwrap();
    assert_eq!(
        h5_filter::decompress_zfp_filter(&[], &cd_values, &[4], Some(ZfpElementType::F32)),
        Err(Error::TruncatedZfpStream {
            expected: 8,
            actual: 0,
        })
    );
    assert_eq!(
        h5_filter::compress_zfp_filter(&[0u8; 16], &[], &[4], Some(ZfpElementType::F32)),
        Err(Error::ZfpFilter(
            "ZFP: invalid or non-rate cd_values".into()
        ))
    );
    assert_eq!(
        h5_filter::compress_zfp_filter(&[0u8; 16], &cd_values, &[4], None),
        Err(Error::ZfpFilter(
            "ZFP: element_type missing from ChunkContext (caller must set it)".into()
        ))
    );
}

#[test]
fn exported_codec_rejects_dimensions_that_cannot_be_encoded_or_multiplied() {
    assert_eq!(
        h5_filter::zfp_cd_values_rate(16.0, ZfpElementType::F32, &[0]),
        Err(Error::UnsupportedZfp(
            "chunk dimension 0 does not fit the 1D ZFP metadata field (1..=281474976710656)".into()
        ))
    );
    assert_eq!(
        h5_filter::zfp_cd_values_rate(16.0, ZfpElementType::F32, &[(1u64 << 48) + 1]),
        Err(Error::UnsupportedZfp(
            "chunk dimension 281474976710657 does not fit the 1D ZFP metadata field (1..=281474976710656)"
                .into()
        ))
    );
    assert_eq!(
        h5_filter::zfp_cd_values_rate(0.0, ZfpElementType::F32, &[4]),
        Err(Error::ZfpFilter(
            "ZFP: rate must be in (0, 32]; got 0".into()
        ))
    );
    assert_eq!(
        h5_filter::decompress_zfp(&[], &[usize::MAX, 2], 16.0, ZfpElementType::F32),
        Err(Error::ZfpSizeOverflow)
    );
}

#[test]
fn a_nonzero_f32_block_needs_room_for_its_header() {
    let raw: Vec<u8> = [1.0f32; 4]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(
        h5_filter::compress_zfp(&raw, &[4], 0.25, ZfpElementType::F32),
        Err(Error::ZfpHeaderTooLarge {
            budget: 1,
            required: 9,
        })
    );

    let encoded = h5_filter::compress_zfp(&raw, &[4], 2.25, ZfpElementType::F32).unwrap();
    assert_eq!(encoded.len(), 2);
    assert_eq!(
        h5_filter::decompress_zfp(&encoded, &[4], 2.25, ZfpElementType::F32).unwrap(),
        [0u8; 16]
    );
}

#[test]
fn a_nonzero_f64_block_needs_room_for_its_header() {
    let raw: Vec<u8> = [1.0f64; 4]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(
        h5_filter::compress_zfp(&raw, &[4], 2.75, ZfpElementType::F64),
        Err(Error::ZfpHeaderTooLarge {
            budget: 11,
            required: 12,
        })
    );

    let encoded = h5_filter::compress_zfp(&raw, &[4], 3.0, ZfpElementType::F64).unwrap();
    assert_eq!(encoded.len(), 2);
    assert_eq!(
        h5_filter::decompress_zfp(&encoded, &[4], 3.0, ZfpElementType::F64).unwrap(),
        [0u8; 32]
    );
}
