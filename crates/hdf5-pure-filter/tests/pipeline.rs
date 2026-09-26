use core::num::NonZeroU32;

use hdf5_pure_filter::{
    ChunkContext, FilterScratch, FilterStep, H5Z_FLAG_OPTIONAL, ScaleOffset, ScaleOffsetByteOrder,
    ScaleOffsetFill, ScaleOffsetType,
};
use rstest::rstest;

struct Step {
    id: u16,
    flags: u16,
    data: Vec<u32>,
}

impl FilterStep for Step {
    fn id(&self) -> u16 {
        self.id
    }

    fn flags(&self) -> u16 {
        self.flags
    }

    fn client_data(&self) -> &[u32] {
        &self.data
    }
}

#[test]
fn exported_pipeline_encodes_known_bytes_and_decodes_a_partial_mask() {
    let steps = [
        Step {
            id: hdf5_pure_filter::FILTER_SHUFFLE,
            flags: 0,
            data: Vec::new(),
        },
        Step {
            id: hdf5_pure_filter::FILTER_FLETCHER32,
            flags: H5Z_FLAG_OPTIONAL,
            data: Vec::new(),
        },
    ];
    assert!(steps[1].optional());
    let ctx = ChunkContext {
        chunk_dims: &[2],
        element_size: NonZeroU32::new(2).unwrap(),
        element_type: None,
        scale_offset_type: None,
    };
    let data = [1, 2, 3, 4];
    let encoded =
        hdf5_pure_filter::compress_chunk_with(&mut FilterScratch::new(), &data, &steps, ctx)
            .unwrap();
    assert_eq!(encoded, [1, 3, 2, 4, 7, 3, 10, 4]);
    let sliced =
        hdf5_pure_filter::compress_chunk_with(&mut FilterScratch::new(), &data, &steps[..1], ctx)
            .unwrap();
    assert_eq!(sliced, [1, 3, 2, 4]);
    assert_eq!(
        hdf5_pure_filter::decompress_chunk(&encoded, &steps, ctx, 0).unwrap(),
        data
    );
    assert_eq!(
        hdf5_pure_filter::decompress_chunk(&encoded[..4], &steps, ctx, 1 << 1).unwrap(),
        data
    );
}

#[rstest]
#[case(&[hdf5_pure_filter::FILTER_DEFLATE, hdf5_pure_filter::FILTER_FLETCHER32], hdf5_pure_filter::FILTER_SHUFFLE, 0)]
#[case(&[hdf5_pure_filter::FILTER_SHUFFLE, hdf5_pure_filter::FILTER_FLETCHER32], hdf5_pure_filter::FILTER_DEFLATE, 1)]
#[case(&[hdf5_pure_filter::FILTER_SHUFFLE, hdf5_pure_filter::FILTER_DEFLATE], hdf5_pure_filter::FILTER_FLETCHER32, 2)]
#[case(&[hdf5_pure_filter::FILTER_FLETCHER32, hdf5_pure_filter::FILTER_DEFLATE], hdf5_pure_filter::FILTER_SHUFFLE, 0)]
fn canonical_position_keeps_existing_pipeline_order(
    #[case] existing: &[u16],
    #[case] added: u16,
    #[case] position: usize,
) {
    assert_eq!(
        hdf5_pure_filter::canonical_filter_position(existing.iter().copied(), added),
        position
    );
}

#[rstest]
#[case(&[hdf5_pure_filter::FILTER_LZF, hdf5_pure_filter::FILTER_DEFLATE], Some(("lzf", "deflate")))]
#[case(&[hdf5_pure_filter::FILTER_DEFLATE, hdf5_pure_filter::FILTER_LZF], Some(("lzf", "deflate")))]
#[case(&[hdf5_pure_filter::FILTER_SCALEOFFSET, hdf5_pure_filter::FILTER_SHUFFLE], Some(("shuffle", "scale-offset")))]
#[case(&[hdf5_pure_filter::FILTER_SHUFFLE, hdf5_pure_filter::FILTER_SCALEOFFSET], Some(("shuffle", "scale-offset")))]
#[case(&[hdf5_pure_filter::FILTER_SHUFFLE, hdf5_pure_filter::FILTER_FLETCHER32], None)]
fn conflicts_are_independent_of_filter_order(
    #[case] filters: &[u16],
    #[case] expected: Option<(&str, &str)>,
) {
    assert_eq!(
        hdf5_pure_filter::first_filter_conflict(filters.iter().copied()),
        expected
    );
}

#[cfg(feature = "zfp")]
#[test]
fn zfp_conflict_precedes_other_conflicts() {
    assert_eq!(
        hdf5_pure_filter::first_filter_conflict(
            [
                hdf5_pure_filter::FILTER_LZF,
                hdf5_pure_filter::FILTER_DEFLATE,
                hdf5_pure_filter::FILTER_ZFP,
                hdf5_pure_filter::FILTER_SHUFFLE,
            ]
            .into_iter(),
        ),
        Some(("shuffle", "ZFP"))
    );
}

#[test]
fn edit_classification_distinguishes_lossless_scale_offset() {
    let scalar = ScaleOffsetType::integer(false, ScaleOffsetByteOrder::LittleEndian);
    let integer = hdf5_pure_filter::build_scale_offset_cd_values(
        ScaleOffset::Integer(0),
        scalar,
        1,
        4,
        ScaleOffsetFill::Undefined,
    )
    .unwrap();
    let integer_pipeline = [Step {
        id: hdf5_pure_filter::FILTER_SCALEOFFSET,
        flags: 0,
        data: integer,
    }];
    assert!(hdf5_pure_filter::filters_reencodable(&integer_pipeline));
    assert!(hdf5_pure_filter::filters_lossless(&integer_pipeline));

    let scalar = ScaleOffsetType::floating(ScaleOffsetByteOrder::LittleEndian);
    let float = hdf5_pure_filter::build_scale_offset_cd_values(
        ScaleOffset::FloatDScale(2),
        scalar,
        4,
        4,
        ScaleOffsetFill::Undefined,
    )
    .unwrap();
    let float_pipeline = [Step {
        id: hdf5_pure_filter::FILTER_SCALEOFFSET,
        flags: 0,
        data: float,
    }];
    assert!(hdf5_pure_filter::filters_reencodable(&float_pipeline));
    assert!(!hdf5_pure_filter::filters_lossless(&float_pipeline));

    let unknown_pipeline = [Step {
        id: u16::MAX,
        flags: 0,
        data: Vec::new(),
    }];
    assert!(!hdf5_pure_filter::filters_reencodable(&unknown_pipeline));
    assert!(!hdf5_pure_filter::filters_lossless(&unknown_pipeline));
}
