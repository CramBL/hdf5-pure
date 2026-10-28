//! Chunk placement at dataset edges and at nonzero trailing offsets.
//!
//! Compares clipped edge chunks with row-major input and checks malformed offsets using
//! independently constructed version 1 B-tree files.

use hdf5_pure::DeflateLevel;
use hdf5_pure::{File, FileBuilder};
use rstest::rstest;
use test_util::btree_v1;
use test_util::data_layout;
use test_util::dataspace;
use test_util::datatype;
use test_util::image::Image;
use test_util::object_header::{Message, MessageType};
use test_util::widths::Widths;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::rank_two(MalformedChunk::RankTwo, &[0, 11, 12, 0, 14, 15, 0, 0, 0])]
#[case::rank_three_inner(
    MalformedChunk::RankThreeInner,
    &[0, 11, 12, 0, 14, 15, 0, 17, 18, 0, 20, 21, 0, 0, 0, 0, 0, 0]
)]
#[case::rank_three_middle(
    MalformedChunk::RankThreeMiddle,
    &[0, 0, 0, 11, 12, 13, 0, 0, 0, 17, 18, 19, 0, 0, 0, 0, 0, 0]
)]
fn row_bands_clip_chunks_with_nonzero_trailing_offsets(
    #[case] case: MalformedChunk,
    #[case] expected: &[i32],
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let chunks = case.chunk_shape();
    let shape = case.shape();
    let offsets = case.offsets();
    let path = test_util::temp::temp_path("trailing_offsets.h5");
    std::fs::write(&path, case.build()).unwrap();
    let file = backend.open(&path).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.shape().unwrap(), shape);
    assert_eq!(dataset.chunk_shape().unwrap(), Some(chunks.to_vec()));
    let descriptors = dataset.chunks().unwrap();
    assert_eq!(descriptors.len(), 1);
    assert_eq!(&descriptors[0].offset[..shape.len()], offsets);
    assert_eq!(dataset.read_i32().unwrap(), expected);
    assert_eq!(dataset.read_i32_rows(0, shape[0]).unwrap(), expected);
    let row_elements = usize::try_from(shape[1..].iter().product::<u64>()).unwrap();
    assert_eq!(
        dataset.read_i32_rows(1, 1).unwrap(),
        expected[row_elements..2 * row_elements]
    );
    drop(dataset);
    file.close().unwrap();
}

/// Write a flat row-major `f64` dataset with the given shape and chunk shape,
/// read it back, and assert it equals the input.
fn roundtrip_f64(shape: &[u64], chunks: &[u64], shuffle_deflate: bool) {
    let total: u64 = shape.iter().product();
    let data: Vec<f64> = (0..total).map(|i| i as f64 * 1.5 - 7.0).collect();

    let mut b = FileBuilder::new();
    {
        let ds = b
            .create_dataset("d")
            .with_f64_data(&data)
            .with_shape(shape)
            .with_chunks(chunks);
        if shuffle_deflate {
            ds.with_shuffle().with_deflate(DeflateLevel::Six);
        }
    }
    let bytes = b.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(
        ds.shape().unwrap(),
        shape,
        "shape {shape:?} chunks {chunks:?}"
    );
    assert_eq!(
        ds.read_f64().unwrap(),
        data,
        "f64 roundtrip mismatch for shape {shape:?} chunks {chunks:?} sd={shuffle_deflate}"
    );
}

#[test]
fn edge_chunks_1d() {
    // 25 elements, chunk 10 => last chunk overhangs (inner-dim overhang).
    roundtrip_f64(&[25], &[10], false);
    roundtrip_f64(&[25], &[10], true);
    // Chunk larger than the dataset (single overhanging chunk).
    roundtrip_f64(&[7], &[16], false);
}

#[test]
fn edge_chunks_2d() {
    // 5x7 with 2x3 chunks: rows {0,2,4} (row-4 chunk overhangs row 5 -> outer
    // overhang) and cols {0,3,6} (col-6 chunk overhangs cols 7,8 -> inner
    // overhang). Both edge cases in one fixture.
    roundtrip_f64(&[5, 7], &[2, 3], false);
    roundtrip_f64(&[5, 7], &[2, 3], true);
    // Tall-and-thin and wide-and-short variants.
    roundtrip_f64(&[13, 4], &[4, 4], false);
    roundtrip_f64(&[4, 13], &[4, 4], false);
}

#[test]
fn edge_chunks_3d() {
    // 3x3x5 with 2x2x2 chunks: overhang in all three dimensions.
    roundtrip_f64(&[3, 3, 5], &[2, 2, 2], false);
    roundtrip_f64(&[3, 3, 5], &[2, 2, 2], true);
}

#[test]
fn edge_chunks_integer_types() {
    // Exercise the assembly kernel together with the integer decode fast paths
    // for both a narrowing read (i32 stored, read as i32) and a wide read.
    let shape = [5u64, 7];
    let total: u64 = shape.iter().product();

    let i32_data: Vec<i32> = (0..total as i32).map(|i| i.wrapping_mul(99) - 5).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&i32_data)
        .with_shape(&shape)
        .with_chunks(&[2, 3])
        .with_shuffle()
        .with_deflate(DeflateLevel::Six);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), i32_data);

    let u16_data: Vec<u16> = (0..total).map(|i| ((i * 137) & 0xFFFF) as u16).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u16_data(&u16_data)
        .with_shape(&shape)
        .with_chunks(&[3, 2]);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_u16().unwrap(), u16_data);
}

#[derive(Clone, Copy, Debug)]
pub enum MalformedChunk {
    RankThreeInner,
    RankThreeMiddle,
    RankTwo,
}

impl MalformedChunk {
    pub fn build(self) -> Vec<u8> {
        let chunks = self.chunk_shape();
        let shape = self.shape();
        let offsets = self.offsets();
        // Each fixed case contains at most twelve elements.
        let payload: Vec<_> = (11..11 + u32::try_from(chunks.iter().product::<u64>()).unwrap())
            .flat_map(|value| f64::from(value).to_le_bytes())
            .collect();
        let mut image = Image::starting_with(&[0; 64]);
        let payload_address = image.append(&payload);
        let mut key_offsets = offsets.to_vec();
        key_offsets.push(0);
        let index = image.append(&btree_v1::node(
            btree_v1::NodeType::CHUNK,
            0,
            &[
                // The payload contains at most twelve f64 values.
                btree_v1::chunk_key(u32::try_from(payload.len()).unwrap(), 0, &key_offsets),
                btree_v1::chunk_key(0, 0, &vec![u64::MAX; key_offsets.len()]),
            ],
            &[payload_address as u64],
            Widths::EIGHT,
        ));
        // Each fixed case has chunk dimensions no greater than three.
        let chunk_dimensions: Vec<_> = chunks
            .iter()
            .map(|&edge| u32::try_from(edge).unwrap())
            .collect();
        image.build_with_dataset(
            "data",
            [
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::v2(
                        // Each fixed case has rank two or three.
                        u8::try_from(shape.len()).unwrap(),
                        dataspace::Flags::NONE,
                        dataspace::Kind::SIMPLE,
                        shape,
                        None,
                    ),
                ),
                Message::new(MessageType::DATATYPE, &datatype::f64_le()),
                Message::new(
                    MessageType::DATA_LAYOUT,
                    &data_layout::chunked(index as u64, &chunk_dimensions, 8),
                ),
            ],
        )
    }

    pub fn chunk_shape(self) -> &'static [u64] {
        match self {
            Self::RankThreeInner | Self::RankThreeMiddle => &[2, 2, 3],
            Self::RankTwo => &[2, 3],
        }
    }

    pub fn shape(self) -> &'static [u64] {
        match self {
            Self::RankThreeInner | Self::RankThreeMiddle => &[3, 2, 3],
            Self::RankTwo => &[3, 3],
        }
    }

    pub fn offsets(self) -> &'static [u64] {
        match self {
            Self::RankThreeInner => &[0, 0, 1],
            Self::RankThreeMiddle => &[0, 1, 0],
            Self::RankTwo => &[0, 1],
        }
    }
}
