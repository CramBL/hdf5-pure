//! Independent images with shared committed datatypes or hard-link aliases.

use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::dataspace::Kind;
use crate::file_space_info;
use crate::image::Image;
use crate::legacy;
use crate::link_message::HardLink;
use crate::object_header::Message;
use crate::object_header::MessageFlags;
use crate::object_header::MessageType;
use crate::object_header::v1;
use crate::object_header::v2;
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// A fixed hierarchy with a shared datatype or two links to one dataset.
#[derive(Clone, Copy, Debug)]
pub enum Fixture {
    /// A version 1 committed datatype header with one link and two dataset uses.
    CommittedDatatypeV1,
    /// Two hard links to a contiguous dataset in a persisted free-space file.
    HardLinkUndo,
}

impl Fixture {
    /// Encodes the selected hierarchy as an independent file image.
    pub fn build(self) -> Vec<u8> {
        match self {
            Self::CommittedDatatypeV1 => committed_datatype_v1(),
            Self::HardLinkUndo => hard_link_undo(),
        }
    }
}

/// Builds `mytype` with a reference count of three in its version 1 header prefix.
///
/// The root uses a legacy symbol table. Distinct datasets `first` and `second`
/// each contain `[1, 2]` as little-endian [`i32`] values and refer to the committed
/// datatype through shared messages. The third reference is its root link.
pub fn committed_datatype_v1() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let mut datatype_bytes = legacy::i32_datatype();
    datatype_bytes.resize(datatype_bytes.len().next_multiple_of(MESSAGE_ALIGNMENT), 0);
    let datatype = image.append_aligned(
        &v1::Header::new()
            .reference_count(3)
            .message(Message::new(MessageType::DATATYPE, &datatype_bytes))
            .build(),
        MESSAGE_ALIGNMENT,
    ) as u64;
    let mut reference = vec![SHARED_VERSION, COMMITTED_LOCATION];
    reference.extend_from_slice(&datatype.to_le_bytes());
    reference.resize(reference.len().next_multiple_of(MESSAGE_ALIGNMENT), 0);
    let datatype_message =
        Message::new(MessageType::DATATYPE, &reference).with_flags(MessageFlags::SHARED);
    let first = legacy_dataset(&mut image, &datatype_message);
    let second = legacy_dataset(&mut image, &datatype_message);
    let root = legacy::group_with_messages(
        &mut image,
        &[("first", first), ("mytype", datatype), ("second", second)],
        &[],
    );
    legacy::finish(image, root)
}

/// Builds hard links `aa` and `bb` to one contiguous dataset containing `[1, 2, 3]`.
///
/// The dataset uses little-endian [`i32`] storage and a version 2 header with an
/// Object Reference Count message equal to two. The file has a version 3
/// superblock and persists free-space managers without paging.
pub fn hard_link_undo() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 48]);
    let raw = [1i32, 2, 3]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let data = image.append(&raw) as u64;
    let mut references = vec![REFERENCE_COUNT_VERSION];
    references.extend_from_slice(&2u32.to_le_bytes());
    let dataset = image.append(
        &v2::Header::new()
            .messages([
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::v2(1, Flags::NONE, Kind::SIMPLE, &[3], None),
                ),
                Message::new(MessageType::DATATYPE, &legacy::i32_datatype()),
                Message::new(
                    MessageType::DATA_LAYOUT,
                    &data_layout::contiguous(data, raw.len() as u64),
                ),
                Message::new(MessageType::OBJECT_REFERENCE_COUNT, &references),
            ])
            .build(),
    ) as u64;
    let mut info = vec![LINK_INFO_VERSION, NO_FLAGS];
    info.extend_from_slice(&u64::MAX.to_le_bytes());
    info.extend_from_slice(&u64::MAX.to_le_bytes());
    let root = image.append(
        &v2::Header::new()
            .messages([
                Message::new(MessageType::LINK_INFO, &info),
                Message::new(MessageType::GROUP_INFO, &[GROUP_INFO_VERSION, NO_FLAGS]),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("aa", dataset).build(Widths::EIGHT),
                ),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("bb", dataset).build(Widths::EIGHT),
                ),
            ])
            .build(),
    ) as u64;
    let extension = image.append(
        &v2::Header::new()
            .message(Message::new(
                MessageType::FILE_SPACE_INFO,
                &file_space_info::version_0(
                    Widths::EIGHT,
                    file_space_info::ALL_PERSIST,
                    1,
                    &[u64::MAX; file_space_info::VERSION_0_MANAGERS],
                ),
            ))
            .build(),
    ) as u64;
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .version(SUPERBLOCK_VERSION)
            .extension_address(extension)
            .root_header_address(root)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    image.build()
}

fn legacy_dataset(image: &mut Image, datatype: &Message) -> u64 {
    let raw = [1i32, 2]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let data = image.append_aligned(&raw, MESSAGE_ALIGNMENT) as u64;
    let mut layout = data_layout::contiguous(data, raw.len() as u64);
    layout.resize(layout.len().next_multiple_of(MESSAGE_ALIGNMENT), 0);
    image.append_aligned(
        &v1::Header::new()
            .messages([
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::v1(1, Flags::NONE, &[2], None),
                ),
                datatype.clone(),
                Message::new(MessageType::DATA_LAYOUT, &layout),
            ])
            .build(),
        MESSAGE_ALIGNMENT,
    ) as u64
}

// Sections `subsubsec_fmt4_dataobject_hdr_prefix_one` and
// `subsec_fmt4_dataobject_hdr_msg`, version 4.0.
const MESSAGE_ALIGNMENT: usize = 8;
const SHARED_VERSION: u8 = 2;
const COMMITTED_LOCATION: u8 = 0;

// Sections `subsec_fmt4_boot_super`, `subsubsec_fmt4_dataobject_hdr_msg_refcount`,
// `subsubsec_fmt4_dataobject_hdr_msg_linkinfo` and
// `subsubsec_fmt4_dataobject_hdr_msg_groupinfo`, version 4.0.
const SUPERBLOCK_VERSION: u8 = 3;
const REFERENCE_COUNT_VERSION: u8 = 0;
const LINK_INFO_VERSION: u8 = 0;
const GROUP_INFO_VERSION: u8 = 0;
const NO_FLAGS: u8 = 0;
