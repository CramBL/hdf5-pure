#![cfg(feature = "zfp")]

use hdf5_pure_core::ZfpRate;

use super::*;
use crate::zfp;
use crate::zfp::ResolvedZfpSettings;

#[rstest::rstest]
#[cfg(feature = "deflate")]
#[case::rank1(&[1], 16)]
#[case::rank2(&[1, 1], 64)]
#[case::rank3(&[1, 1, 1], 256)]
#[case::rank4(&[1, 1, 1, 1], 1024)]
fn zfp_partial_block_inner_deflate_outer_roundtrips(
    #[case] dims: &[u64],
    #[case] encoded_len: usize,
) {
    let mut scratch = FilterScratch::new();
    let settings =
        ResolvedZfpSettings::new(ZfpRate::try_from(32.0).unwrap(), ZfpElementType::F32, dims)
            .unwrap();
    let cd_values = scratch.zfp_client_data(settings, 0);
    let filters = [
        FilterDescription {
            filter_id: FILTER_ZFP,
            flags: 0,
            client_data: cd_values,
        },
        FilterDescription {
            filter_id: FILTER_DEFLATE,
            flags: 0,
            client_data: vec![6],
        },
    ];
    let ctx = ChunkContext {
        chunk_dims: dims,
        element_size: NonZeroU32::new(4).unwrap(),
        element_type: Some(ZfpElementType::F32),
        scale_offset_type: None,
    };
    let data = 1.0f32.to_le_bytes();
    let zfp_bytes = zfp::compress_filter(
        &data,
        filters[0].client_data(),
        ctx.chunk_dims,
        ctx.element_type,
    )
    .unwrap();
    assert_eq!(zfp_bytes.len(), encoded_len);
    let stored = compress_chunk_with(&mut scratch, &data, &filters, ctx).unwrap();
    let repeated = compress_chunk_with(&mut scratch, &data, &filters, ctx).unwrap();
    assert_eq!(repeated, stored);
    assert_eq!(
        decompress_chunk_with(&mut scratch, &stored, &filters, ctx, 0).unwrap(),
        data
    );
    assert_eq!(scratch.zfp_preparations, 1);

    let mut reader = FilterScratch::new();
    assert_eq!(
        decompress_chunk_with(&mut reader, &stored, &filters, ctx, 0).unwrap(),
        data
    );
    assert_eq!(
        decompress_chunk_with(&mut reader, &stored, &filters, ctx, 0).unwrap(),
        data
    );
    assert_eq!(reader.zfp_preparations, 1);
}

#[rstest::rstest]
#[case::geometry(&[8], ZfpElementType::F32, 4, 16.0, 32, 16)]
#[case::scalar(&[4], ZfpElementType::F64, 8, 16.0, 32, 8)]
#[case::rate(&[4], ZfpElementType::F32, 4, 32.0, 16, 16)]
fn prepared_zfp_settings_are_replaced_when_metadata_or_context_changes(
    #[case] dimensions: &[u64],
    #[case] scalar: ZfpElementType,
    #[case] element_size: u32,
    #[case] requested: f64,
    #[case] decoded_bytes: usize,
    #[case] encoded_bytes: usize,
) {
    let original =
        ResolvedZfpSettings::new(ZfpRate::try_from(16.0).unwrap(), ZfpElementType::F32, &[4])
            .unwrap();
    let mut scratch = FilterScratch::new();
    let client_data = scratch.zfp_client_data(original, 0);
    let mut filters = [FilterDescription {
        filter_id: FILTER_ZFP,
        flags: 0,
        client_data,
    }];
    let ctx = ChunkContext {
        chunk_dims: &[4],
        element_size: NonZeroU32::new(4).unwrap(),
        element_type: Some(ZfpElementType::F32),
        scale_offset_type: None,
    };
    assert_eq!(
        compress_chunk_with(&mut scratch, &[0; 16], &filters, ctx).unwrap(),
        [0; 8]
    );
    assert_eq!(scratch.zfp_preparations, 1);

    if requested != 16.0 {
        filters[0].client_data = zfp::zfp_cd_values_rate(
            ZfpRate::try_from(requested).unwrap(),
            ZfpElementType::F32,
            &[4],
        )
        .unwrap();
    }
    let ctx = ChunkContext {
        chunk_dims: dimensions,
        element_type: Some(scalar),
        element_size: NonZeroU32::new(element_size).unwrap(),
        ..ctx
    };
    let raw = vec![0; decoded_bytes];
    let encoded = compress_chunk_with(&mut scratch, &raw, &filters, ctx).unwrap();
    assert_eq!(encoded, vec![0; encoded_bytes]);
    assert_eq!(
        decompress_chunk_with(&mut scratch, &encoded, &filters, ctx, 0).unwrap(),
        raw
    );
    assert_eq!(scratch.zfp_preparations, 2);
    let repeated = compress_chunk_with(&mut scratch, &raw, &filters, ctx).unwrap();
    assert_eq!(repeated, encoded);
    assert_eq!(scratch.zfp_preparations, 2);
}

#[test]
fn prepared_zfp_settings_check_float_headers_for_each_chunk() {
    let settings =
        ResolvedZfpSettings::new(ZfpRate::try_from(0.25).unwrap(), ZfpElementType::F32, &[4])
            .unwrap();
    let mut scratch = FilterScratch::new();
    let client_data = scratch.zfp_client_data(settings, 0);
    let filters = [FilterDescription {
        filter_id: FILTER_ZFP,
        flags: 0,
        client_data,
    }];
    let ctx = ChunkContext {
        chunk_dims: &[4],
        element_size: NonZeroU32::new(4).unwrap(),
        element_type: Some(ZfpElementType::F32),
        scale_offset_type: None,
    };
    assert_eq!(
        compress_chunk_with(&mut scratch, &[0; 16], &filters, ctx).unwrap(),
        [0]
    );
    let nonzero = [1.0f32; 4].map(f32::to_le_bytes).concat();
    assert_eq!(
        compress_chunk_with(&mut scratch, &nonzero, &filters, ctx),
        Err(Error::ZfpHeaderTooLarge {
            budget: 1,
            required: 9
        })
    );
    assert_eq!(
        compress_chunk_with(&mut scratch, &[0; 16], &filters, ctx).unwrap(),
        [0]
    );
    assert_eq!(scratch.zfp_preparations, 1);
}

#[test]
fn malformed_zfp_metadata_does_not_use_or_replace_prepared_settings() {
    let settings =
        ResolvedZfpSettings::new(ZfpRate::try_from(16.0).unwrap(), ZfpElementType::F32, &[4])
            .unwrap();
    let mut scratch = FilterScratch::new();
    let client_data = scratch.zfp_client_data(settings, 0);
    let mut filters = [FilterDescription {
        filter_id: FILTER_ZFP,
        flags: 0,
        client_data,
    }];
    let ctx = ChunkContext {
        chunk_dims: &[4],
        element_size: NonZeroU32::new(4).unwrap(),
        element_type: Some(ZfpElementType::F32),
        scale_offset_type: None,
    };
    filters[0].client_data[1] ^= 1;
    assert_eq!(
        compress_chunk_with(&mut scratch, &[0; 16], &filters, ctx),
        Err(Error::ZfpFilter(
            "ZFP: invalid or non-rate cd_values".into()
        ))
    );
    assert_eq!(scratch.zfp_preparations, 1);
    filters[0].client_data[1] ^= 1;
    assert_eq!(
        compress_chunk_with(&mut scratch, &[0; 16], &filters, ctx).unwrap(),
        [0; 8]
    );
    assert_eq!(scratch.zfp_preparations, 1);
}

#[test]
fn zfp_conflict_precedes_other_conflicts() {
    assert_eq!(
        first_filter_conflict([FILTER_LZF, FILTER_DEFLATE, FILTER_ZFP, FILTER_SHUFFLE].into_iter(),),
        Some(("shuffle", "ZFP"))
    );
}
