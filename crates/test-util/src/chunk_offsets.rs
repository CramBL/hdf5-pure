//! Independent malformed files with unaligned chunk-index offsets.

use crate::btree_v1;
use crate::data_layout;
use crate::dataspace;
use crate::datatype;
use crate::image::Image;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::widths::Widths;

/// A fixed chunk-index entry with a deliberately unaligned trailing offset.
#[derive(Clone, Copy, Debug)]
pub enum MalformedChunk {
    /// A chunk with shape `[2, 2, 3]` at offset `[0, 0, 1]`.
    RankThreeInner,
    /// A chunk with shape `[2, 2, 3]` at offset `[0, 1, 0]`.
    RankThreeMiddle,
    /// A chunk with shape `[2, 3]` at offset `[0, 1]`.
    RankTwo,
}

impl MalformedChunk {
    /// Encodes the dataset `data` with one chunk at the malformed offset.
    ///
    /// The chunk contains sequential [`f64`] values starting at 11.
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

    /// Returns the declared chunk dimensions.
    pub fn chunk_shape(self) -> &'static [u64] {
        match self {
            Self::RankThreeInner | Self::RankThreeMiddle => &[2, 2, 3],
            Self::RankTwo => &[2, 3],
        }
    }

    /// Returns the logical dataset dimensions.
    pub fn shape(self) -> &'static [u64] {
        match self {
            Self::RankThreeInner | Self::RankThreeMiddle => &[3, 2, 3],
            Self::RankTwo => &[3, 3],
        }
    }

    /// Returns the chunk's starting element coordinates in the dataset.
    pub fn offsets(self) -> &'static [u64] {
        match self {
            Self::RankThreeInner => &[0, 0, 1],
            Self::RankThreeMiddle => &[0, 1, 0],
            Self::RankTwo => &[0, 1],
        }
    }
}
