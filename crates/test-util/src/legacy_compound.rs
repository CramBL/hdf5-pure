//! Independent version 1 compound datatype messages and a contiguous dataset image.

use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::datatype;
use crate::datatype::leaf::{self, Version};
use crate::datatype::{Class, VariableLengthSequence};
use crate::image::Image;
use crate::legacy;
use crate::link_info::{self, CreationIndex, Width};
use crate::link_message::HardLink;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::{v1, v2};
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

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

/// The shape of a sequence member in a version 1 compound fixture.
#[derive(Clone, Copy)]
pub enum SequenceMemberShape {
    /// A version 3 array of two sequences as the member datatype.
    ExplicitArray,
    /// Two sequences with their extent in the member's dimension block.
    ImplicitArray,
    /// One sequence with zero member dimensionality.
    Scalar,
}

/// Builds a version 1 compound with a variable-length byte sequence member.
///
/// The compound ends at the member's last byte, using the descriptor size for `offset_width`.
///
/// # Panics
///
/// Panics if `offset_width` is other than 2, 4, or 8, or if the member end exceeds [`u32::MAX`].
pub fn sequence_member(offset_width: usize, offset: u32, shape: SequenceMemberShape) -> Vec<u8> {
    let element = leaf::unsigned_byte(Version::One);
    let sequence = VariableLengthSequence::new(Version::One, offset_width, &element);
    let descriptor_size = sequence.descriptor_size();
    let sequence = sequence.build();
    let element_count = match shape {
        SequenceMemberShape::Scalar => 1,
        SequenceMemberShape::ExplicitArray | SequenceMemberShape::ImplicitArray => 2,
    };
    let member_size = descriptor_size
        .checked_mul(element_count)
        .expect("two descriptors fit in a member");
    let compound_size = offset
        .checked_add(member_size)
        .expect("the member end fits in the compound size field");
    match shape {
        SequenceMemberShape::ExplicitArray => {
            let array = datatype::array(
                Version::Three.get(),
                [0; 3],
                member_size,
                1,
                &[element_count],
                &sequence,
            );
            raw_member(compound_size, offset, 0, 0, [0; 4], &array)
        }
        SequenceMemberShape::ImplicitArray => raw_member(
            compound_size,
            offset,
            1,
            0,
            [element_count, 0, 0, 0],
            &sequence,
        ),
        SequenceMemberShape::Scalar => raw_member(compound_size, offset, 0, 0, [0; 4], &sequence),
    }
}

/// Builds an empty `value` dataset with a sequence member and two-byte offsets and lengths.
pub fn narrow_sequence_file(shape: SequenceMemberShape) -> Vec<u8> {
    let widths = Widths::new(2, 2);
    let datatype = sequence_member(widths.offset, 0, shape);
    let superblock = Superblock::new(widths);
    let mut image = Image::starting_with(&vec![0; superblock.build().len()]);
    let dataset = image.append(
        &v2::Header::new()
            .messages([
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::VersionTwo::simple(widths, &[0], None).build(),
                ),
                Message::new(MessageType::DATATYPE, &datatype),
                Message::new(MessageType::DATA_LAYOUT, &data_layout::compact(&[])),
            ])
            .build(),
    );
    let link_info = link_info::Body::new(Width::Two, None, None, None, CreationIndex::Absent)
        .unwrap()
        .build();
    let link = HardLink::new("value", u64::try_from(dataset).unwrap()).build(widths);
    let root = image.append(
        &v2::Header::new()
            .messages([
                Message::new(MessageType::LINK_INFO, &link_info),
                Message::new(MessageType::LINK, &link),
            ])
            .build(),
    );
    let superblock = superblock
        .root_header_address(u64::try_from(root).unwrap())
        .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
        .build();
    image.place(0, &superblock);
    image.build()
}

/// Builds `data` with one compound element whose `arr` member contains `[11, 22, 33, 44]`.
pub fn file() -> Vec<u8> {
    file_with_datatype(array_member())
}

/// Builds a file with the supplied datatype message and one 16-byte element in `data`.
///
/// Writes four consecutive little-endian `i32` values, `[11, 22, 33, 44]`, as the element.
///
/// # Panics
///
/// Panics if the padded datatype message exceeds [`u16::MAX`] bytes.
pub fn file_with_datatype(datatype: Vec<u8>) -> Vec<u8> {
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
                padded(MessageType::DATATYPE, datatype),
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
