//! Independent version 1 compound datatype messages and a contiguous dataset image.

use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::datatype;
use crate::datatype::Class;
use crate::image::Image;
use crate::legacy;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v1;

/// Builds a version 1 compound message with one member named `arr` from raw metadata.
///
/// Encodes the supplied metadata unchanged, including deliberately malformed ranks, extents,
/// and permutations. `base` contains the member's encoded base datatype.
pub fn raw_member(
    size: u32,
    offset: u32,
    rank: u8,
    permutation: u32,
    dimensions: [u32; 4],
    base: &[u8],
) -> Vec<u8> {
    let mut message = datatype::header(Class::COMPOUND, DATATYPE_VERSION, [1, 0, 0], size);
    message.extend_from_slice(b"arr\0\0\0\0\0");
    message.extend_from_slice(&offset.to_le_bytes());
    message.extend_from_slice(&[rank, 0, 0, 0]);
    message.extend_from_slice(&permutation.to_le_bytes());
    message.extend_from_slice(&[0; 4]);
    for dimension in dimensions {
        message.extend_from_slice(&dimension.to_le_bytes());
    }
    message.extend_from_slice(base);
    message
}

/// Builds a 16-byte compound with one four-element little-endian signed [`i32`] array member.
pub fn array_member() -> Vec<u8> {
    let mut base = datatype::header(
        Class::FIXED_POINT,
        DATATYPE_VERSION,
        [SIGNED_FIXED_POINT, 0, 0],
        4,
    );
    base.extend_from_slice(&0u16.to_le_bytes());
    base.extend_from_slice(&32u16.to_le_bytes());
    raw_member(16, 0, 1, 0, [4, 0, 0, 0], &base)
}

/// Builds `data` with one compound element whose `arr` member contains `[11, 22, 33, 44]`.
pub fn file() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let payload: Vec<_> = [11i32, 22, 33, 44]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    let address = image.append(&payload) as u64;
    let dataset = image.append_aligned(
        &v1::Header::new()
            .messages([
                padded(
                    MessageType::DATASPACE,
                    dataspace::v1(1, Flags::NONE, &[1], None),
                ),
                padded(MessageType::DATATYPE, array_member()),
                padded(
                    MessageType::DATA_LAYOUT,
                    data_layout::contiguous(address, 16),
                ),
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

/// The version 1 encoding in "The Datatype Message" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DATATYPE_VERSION: u8 = 1;

/// The signed fixed-point flag in "The Datatype Message" of the [format specification,
/// version 4.0][spec]. Bit 3 selects the two's complement representation.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const SIGNED_FIXED_POINT: u8 = 0x08;
