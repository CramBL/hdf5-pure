//! Independent HDF5 fixtures for attribute storage and creation metadata.

use crate::attribute;
use crate::bytes;
use crate::dataspace;
use crate::datatype;
use crate::object_header::v2::{Header, HeaderFlags};
use crate::object_header::{Message, MessageType};
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// A selected inconsistency in otherwise valid attribute metadata.
#[derive(Clone, Copy, Debug)]
pub enum Damage {
    /// Creation-order indexing differs between the prefix and Attribute Info.
    Coherence,
    /// An attribute creation index equals the counter for the next index.
    Counter,
    /// Two attributes have the same creation index.
    DuplicateIndex,
    /// Two attributes have the same name.
    DuplicateName,
    /// The prefix indexes creation order without tracking it.
    IndexWithoutTracking,
    /// The minimum dense count exceeds the maximum compact count.
    Policy,
}

/// Builds a file with two compact scalar attributes and indexed attribute creation order.
///
/// Uses attribute thresholds 32/24 and creation indexes 2 and 5 with counter 17.
pub fn file() -> Vec<u8> {
    build_file(None)
}

/// Builds the fixture from [`file()`] with the selected metadata inconsistency.
///
/// Preserves valid checksums so a reader can reach the inconsistent metadata.
pub fn damaged_file(damage: Damage) -> Vec<u8> {
    build_file(Some(damage))
}

fn build_file(damage: Option<Damage>) -> Vec<u8> {
    let flags = if matches!(damage, Some(Damage::IndexWithoutTracking)) {
        HeaderFlags::INDEXES_CREATION_ORDER
    } else {
        HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER
    };
    let (maximum, minimum) = if matches!(damage, Some(Damage::Policy)) {
        (3, 4)
    } else {
        (32, 24)
    };
    let next = if matches!(damage, Some(Damage::Counter)) {
        5
    } else {
        17
    };
    let info = attribute::tracked_info(
        next,
        !matches!(damage, Some(Damage::Coherence)),
        Widths::EIGHT,
    );
    let datatype = datatype::f64_le();
    let dataspace = dataspace::scalar();
    let first =
        attribute::Attribute::new("a", &datatype, &dataspace, &71_f64.to_le_bytes()).build();
    let second_name = if matches!(damage, Some(Damage::DuplicateName)) {
        "a"
    } else {
        "b"
    };
    let second_index = if matches!(damage, Some(Damage::DuplicateIndex)) {
        2
    } else {
        5
    };
    let second =
        attribute::Attribute::new(second_name, &datatype, &dataspace, &73_f64.to_le_bytes())
            .build();
    let mut link_info = vec![LINK_INFO_VERSION, LINK_INFO_DEFAULT_FLAGS];
    bytes::push_undefined_address(&mut link_info, Widths::EIGHT.offset);
    bytes::push_undefined_address(&mut link_info, Widths::EIGHT.offset);
    let header = Header::new()
        .flags(flags)
        .phase_change(maximum, minimum)
        .message(Message::new(MessageType::LINK_INFO, &link_info))
        .message(Message::new(
            MessageType::GROUP_INFO,
            &[GROUP_INFO_VERSION, GROUP_INFO_DEFAULT_FLAGS],
        ))
        .message(Message::new(MessageType::ATTRIBUTE_INFO, &info))
        .message(Message::new(MessageType::ATTRIBUTE, &first).with_creation_order(2))
        .message(Message::new(MessageType::ATTRIBUTE, &second).with_creation_order(second_index))
        .build();
    let root = Superblock::new(Widths::EIGHT).build().len();
    let mut file = Superblock::new(Widths::EIGHT)
        .root_header_address(u64::try_from(root).unwrap())
        .eof_address(u64::try_from(root + header.len()).unwrap())
        .build();
    file.extend_from_slice(&header);
    file
}

// "The Group Info Message" of the format specification, version 4.0, defines version 0.
const GROUP_INFO_VERSION: u8 = 0;
// "The Group Info Message" of the format specification, version 4.0, defines the flags for
// optional fields.
const GROUP_INFO_DEFAULT_FLAGS: u8 = 0;

// "The Link Info Message" of the format specification, version 4.0, defines version 0.
const LINK_INFO_VERSION: u8 = 0;
// "The Link Info Message" of the format specification, version 4.0, defines the flags for
// creation-order fields.
const LINK_INFO_DEFAULT_FLAGS: u8 = 0;
