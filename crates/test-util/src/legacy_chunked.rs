//! Independent legacy file images with deflate-compressed chunks.
//!
//! The image has a version 0 superblock, version 1 object headers, eight-byte
//! addresses and lengths, and a version 1 B-tree chunk index.

use crate::btree_v1;
use crate::btree_v1::NodeType;
use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::datatype;
use crate::image::Image;
use crate::legacy;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v1;
use crate::widths::Widths;

/// Builds `data` with [`f64`] values 0 through 99 in five compressed chunks.
///
/// Each chunk contains twenty little-endian values. The filter pipeline declares
/// deflate level 6.
pub fn chunked_deflate() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let chunks: Vec<_> = (0..5)
        .map(|index| {
            let raw: Vec<_> = (index * 20..(index + 1) * 20)
                .flat_map(|value| f64::from(value).to_le_bytes())
                .collect();
            let compressed = zlib(&raw);
            assert!(compressed.len() < raw.len());
            let address = image.append(&compressed) as u64;
            // Each chunk encodes twenty f64 values, so its compressed size fits u32.
            (
                btree_v1::chunk_key(
                    u32::try_from(compressed.len()).unwrap(),
                    0,
                    &[index as u64 * 20, 0],
                ),
                address,
            )
        })
        .collect();
    let mut keys: Vec<_> = chunks.iter().map(|(key, _)| key.clone()).collect();
    keys.push(btree_v1::chunk_key(0, 0, &[100, 0]));
    let addresses: Vec<_> = chunks.iter().map(|(_, address)| *address).collect();
    let mut index = btree_v1::node(NodeType::CHUNK, 0, &keys, &addresses, Widths::EIGHT);
    index.resize(CHUNK_NODE_SIZE, 0);
    let index = image.append_aligned(&index, legacy::MESSAGE_ALIGNMENT) as u64;
    let dataset = image.append_aligned(
        &v1::Header::new()
            .messages([
                padded(
                    MessageType::DATASPACE,
                    dataspace::v1(1, Flags::NONE, &[100], None),
                ),
                padded(MessageType::DATATYPE, datatype::f64_le()),
                padded(
                    MessageType::DATA_LAYOUT,
                    data_layout::chunked(index, &[20], 8),
                ),
                padded(MessageType::FILTER_PIPELINE, deflate_pipeline()),
            ])
            .build(),
        legacy::MESSAGE_ALIGNMENT,
    ) as u64;
    let root = legacy::group_with_messages(&mut image, &[("data", dataset)], &[]);
    legacy::finish(image, root)
}

fn padded(kind: MessageType, mut body: Vec<u8>) -> Message {
    body.resize(body.len().next_multiple_of(legacy::MESSAGE_ALIGNMENT), 0);
    Message::new(kind, &body)
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_filter`, version 4.0, defines
// version 1 filter descriptions and their eight-byte padding.
fn deflate_pipeline() -> Vec<u8> {
    let mut pipeline = vec![PIPELINE_VERSION, 1, 0, 0, 0, 0, 0, 0];
    pipeline.extend_from_slice(&FILTER_DEFLATE.to_le_bytes());
    pipeline.extend_from_slice(&8u16.to_le_bytes());
    pipeline.extend_from_slice(&FILTER_OPTIONAL.to_le_bytes());
    pipeline.extend_from_slice(&1u16.to_le_bytes());
    pipeline.extend_from_slice(b"deflate\0");
    pipeline.extend_from_slice(&6u32.to_le_bytes());
    pipeline.extend_from_slice(&0u32.to_le_bytes());
    pipeline
}

// RFC 1950 section 2.2 defines the zlib header and Adler-32 trailer. RFC 1951
// sections 3.2.5 and 3.2.6 define fixed Huffman codes and distance-one matches.
fn zlib(raw: &[u8]) -> Vec<u8> {
    let mut stream = Bits::new();
    stream.push(FINAL_FIXED_BLOCK, BLOCK_HEADER_BITS);
    let mut at = 0;
    while at < raw.len() {
        let repeated = if at > 0 {
            raw[at..]
                .iter()
                .take(MAX_MATCH)
                .take_while(|byte| **byte == raw[at - 1])
                .count()
        } else {
            0
        };
        if repeated >= MIN_MATCH {
            stream.push(
                ((repeated - MIN_MATCH) as u16 + FIRST_LENGTH_CODE).reverse_bits()
                    >> (u16::BITS - u32::from(SHORT_CODE_BITS)),
                SHORT_CODE_BITS,
            );
            stream.push(DISTANCE_ONE, DISTANCE_CODE_BITS);
            at += repeated;
            continue;
        }
        let (code, width) = match raw[at] {
            byte @ 0..=LOW_LITERAL_MAX => (LITERAL_FIRST + u16::from(byte), LOW_LITERAL_BITS),
            byte => (
                LITERAL_LAST + u16::from(byte) - u16::from(LOW_LITERAL_MAX) - 1,
                HIGH_LITERAL_BITS,
            ),
        };
        stream.push(code.reverse_bits() >> (u16::BITS - u32::from(width)), width);
        at += 1;
    }
    stream.push(END_OF_BLOCK, SHORT_CODE_BITS);
    let (first, second) = raw.iter().fold((1u32, 0u32), |(first, second), byte| {
        let first = (first + u32::from(*byte)) % ADLER_MODULUS;
        (first, (second + first) % ADLER_MODULUS)
    });
    let mut compressed = ZLIB_HEADER.to_vec();
    compressed.extend(stream.finish());
    compressed.extend_from_slice(&((second << 16) | first).to_be_bytes());
    compressed
}

struct Bits {
    bytes: Vec<u8>,
    pending: u32,
    count: u8,
}

impl Bits {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            pending: 0,
            count: 0,
        }
    }

    fn push(&mut self, value: u16, count: u8) {
        self.pending |= u32::from(value) << self.count;
        self.count += count;
        while self.count >= u8::BITS as u8 {
            self.bytes.push(self.pending as u8);
            self.pending >>= u8::BITS;
            self.count -= u8::BITS as u8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.bytes.push(self.pending as u8);
        }
        self.bytes
    }
}

// HDF5_BTREE_CHUNK_IK_DEF in H5Fprivate.h, HDF5 2.2.0, defines K=32 for
// superblock version 0. Section `subsubsec_fmt4_infra_btrees_v1`, version 4.0,
// reserves 2K children and 2K+1 keys after the prefix and sibling addresses.
const CHUNK_NODE_K: usize = 32;
const CHUNK_NODE_SIZE: usize = 24 + 2 * CHUNK_NODE_K * 8 + (2 * CHUNK_NODE_K + 1) * 24;

// Section `subsubsec_fmt4_dataobject_hdr_msg_filter`, version 4.0.
const PIPELINE_VERSION: u8 = 1;
const FILTER_DEFLATE: u16 = 1;
const FILTER_OPTIONAL: u16 = 0x01;

// RFC 1950 section 2.2 defines the header and Adler-32 modulus. RFC 1951
// sections 3.2.3, 3.2.5 and 3.2.6 define the block and Huffman codes.
const ZLIB_HEADER: [u8; 2] = [0x78, 0x9C];
const ADLER_MODULUS: u32 = 65_521;
const FINAL_FIXED_BLOCK: u16 = 0x03;
const LITERAL_FIRST: u16 = 0x30;
const LITERAL_LAST: u16 = 0x190;
const END_OF_BLOCK: u16 = 0;
const DISTANCE_ONE: u16 = 0;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 10;

const BLOCK_HEADER_BITS: u8 = 3;
const SHORT_CODE_BITS: u8 = 7;
const DISTANCE_CODE_BITS: u8 = 5;
const LOW_LITERAL_MAX: u8 = 143;
const LOW_LITERAL_BITS: u8 = 8;
const HIGH_LITERAL_BITS: u8 = 9;
const FIRST_LENGTH_CODE: u16 = 1;
