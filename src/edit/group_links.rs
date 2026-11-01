//! Reads compact and dense group links with their encoded names, targets, and creation order.

use hdf5_pure_format::__private::GroupInfoError;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::GroupPhaseChange;
use hdf5_pure_format::__private::OwnedLink;

use crate::error::Error;
use crate::group_v2::storage;
pub(super) use crate::group_v2::storage::storage_error;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::OriginalSource;
use crate::source::PhysicalSource;
use crate::width::FormatWidths;

use super::object_header::OhRegion;

/// Reads the group's links and checks their storage metadata for editing.
///
/// Derives dense-storage coordinates from the opened source's captured address context.
/// Reads metadata through the original source's metadata operation, preserving ownership,
/// physical ranges, and cache behavior.
///
/// Dense storage is limited to unfiltered managed objects with seven-byte heap IDs and
/// encoded links of at most 4096 bytes. Checks the name index and any creation-order index
/// against the link bodies and heap object count.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the region describes a dataset, link metadata is
/// inconsistent or unsupported, or the Link Info message is absent. Returns parsing and
/// source-read errors from the object-header, heap, and index readers.
pub(super) fn inspect(
    source: &(impl OriginalSource + ?Sized),
    widths: FormatWidths,
    region: &OhRegion,
) -> Result<GroupLinkStorage, Error> {
    let mut info = None;
    let mut group_info = false;
    let mut links = Vec::new();
    let mut pos = 0;
    while let Some(record) = region.layout().next_message(region, pos)? {
        match record.msg_type {
            MessageType::LINK_INFO | MessageType::GROUP_INFO | MessageType::LINK
                if !record.flags.difference(MessageFlags::CONSTANT).is_empty() =>
            {
                return Err(Error::EditUnsupported(
                    "a group's link metadata has object-header flags this editor cannot reproduce",
                ));
            }
            MessageType::LINK_INFO => {
                if info.replace(record.body).is_some() {
                    return Err(Error::EditUnsupported(
                        "a group has repeated link-info messages",
                    ));
                }
            }
            MessageType::GROUP_INFO => {
                if group_info {
                    return Err(Error::EditUnsupported(
                        "a group has repeated group-info messages",
                    ));
                }
                phase_change(record.body)?;
                group_info = true;
            }
            MessageType::LINK => {
                links.push(
                    OwnedLink::parse(record.body, widths.offsets)
                        .map_err(storage::storage_error)?,
                );
            }
            MessageType::SYMBOL_TABLE => {
                return Err(Error::EditUnsupported(
                    "a group mixes symbol-table storage with modern link storage",
                ));
            }
            MessageType::DATA_LAYOUT => {
                return Err(Error::EditUnsupported(
                    "a target path names a dataset, not a group",
                ));
            }
            _ => {}
        }
        pos = record.body_range.end;
    }
    let info_bytes = info.ok_or(Error::EditUnsupported(
        "a target group's object header has no link-info message",
    ))?;
    let info = GroupLinkStorage::parse(info_bytes, Vec::new(), widths.offsets)
        .map_err(storage::storage_error)?
        .info();
    if let Some(heap) = info.fractal_heap_address {
        if !links.is_empty() {
            return Err(Error::EditUnsupported(
                "a group mixes compact and dense links",
            ));
        }
        links = storage::read_dense(&PhysicalSource::new(source).stored(), widths, &info, heap)?;
    }
    GroupLinkStorage::new(&info, links, widths.offsets).map_err(storage::storage_error)
}

/// Reads the Group Info thresholds and converts parse errors for the editor.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the message has an unsupported encoding or its
/// minimum dense threshold exceeds its maximum compact threshold.
pub(super) fn phase_change(bytes: &[u8]) -> Result<GroupPhaseChange, Error> {
    GroupPhaseChange::parse(bytes).map_err(group_info_error)
}

pub(super) fn group_info_error(error: GroupInfoError) -> Error {
    match error {
        GroupInfoError::Length { expected, actual } if actual < u16::from(expected) => {
            Error::EditUnsupported("a group's group-info message is truncated")
        }
        GroupInfoError::Thresholds { .. } => Error::EditUnsupported(
            "a group's minimum dense threshold exceeds its maximum compact threshold",
        ),
        GroupInfoError::Flags(_)
        | GroupInfoError::Version(_)
        | GroupInfoError::Length { .. }
        | GroupInfoError::TrailingBytes => {
            Error::EditUnsupported("a group's group-info message uses an unsupported encoding")
        }
    }
}
