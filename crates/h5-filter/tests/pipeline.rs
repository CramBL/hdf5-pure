use core::num::NonZeroU32;

use h5_filter::{ChunkContext, FilterScratch, FilterStep, H5Z_FLAG_OPTIONAL};

struct Step {
    id: u16,
    flags: u16,
}

impl FilterStep for Step {
    fn id(&self) -> u16 {
        self.id
    }

    fn flags(&self) -> u16 {
        self.flags
    }

    fn client_data(&self) -> &[u32] {
        &[]
    }
}

#[test]
fn exported_pipeline_encodes_known_bytes_and_decodes_a_partial_mask() {
    let steps = [
        Step {
            id: h5_filter::FILTER_SHUFFLE,
            flags: 0,
        },
        Step {
            id: h5_filter::FILTER_FLETCHER32,
            flags: H5Z_FLAG_OPTIONAL,
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
        h5_filter::compress_chunk_with(&mut FilterScratch::new(), &data, &steps, ctx).unwrap();
    assert_eq!(encoded, [1, 3, 2, 4, 7, 3, 10, 4]);
    assert_eq!(
        h5_filter::decompress_chunk(&encoded, &steps, ctx, 0).unwrap(),
        data
    );
    assert_eq!(
        h5_filter::decompress_chunk(&encoded[..4], &steps, ctx, 1 << 1).unwrap(),
        data
    );
}
