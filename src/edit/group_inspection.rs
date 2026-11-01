//! Group inspection hooks for libhdf5 crosschecks.

use std::path::Path;

use hdf5_pure_format::__private::GroupLinkStorage;

use crate::File;
use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_lock::FileLocking;
use crate::group_v2;
use crate::message_type::MessageType;
use crate::object_path::ObjectPath;

/// A group's stored address, link metadata, and Group Info message body.
pub(super) struct GroupLinkInspection {
    /// The address used by hard links to this group.
    pub(super) object_header_address: StoredAddress,
    /// Links with their encoded names, targets, and creation indexes.
    pub(super) links: GroupLinkStorage,
    /// The Group Info message body, including any stored phase-change thresholds.
    pub(super) group_info: Vec<u8>,
}

/// Opens a file and reads a group's links and storage metadata for crosschecks.
///
/// The editor retains a complete file mirror in both modes. With `source_only` set, the reader
/// uses [`crate::Source`] for each read. With it cleared, the reader can borrow the mirror's
/// whole-file slice. Resolves `group_path` through hard links.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the target is a dataset or its link metadata cannot
/// be reproduced by the editor. Returns opening, path-resolution, parsing, and source-read
/// errors from the editor. Returns [`FormatError::Internal`] if the inspected group has no
/// Group Info message.
pub(super) fn inspect_group_links(
    path: &Path,
    group_path: &str,
    source_only: bool,
) -> Result<GroupLinkInspection, Error> {
    let engine = if source_only {
        WriteEngine::open_source_only(path)?
    } else {
        WriteEngine::open_with_locking(path, FileLocking::Enabled)?
    };
    let address = group_v2::resolve_path_any_from_source(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        engine.superblock(),
        engine.widths(),
        &ObjectPath::parse(group_path),
    )?;
    let info = engine.inspect_group(address)?;
    let mut pos = 0;
    while let Some(record) = info.region.layout().next_message(&info.region, pos)? {
        if record.msg_type == MessageType::GROUP_INFO {
            return Ok(GroupLinkInspection {
                object_header_address: engine
                    .addresses()
                    .relative(PhysicalAddress::new(address))?,
                links: info.links,
                group_info: record.body.to_vec(),
            });
        }
        pos = record.body_range.end;
    }
    Err(FormatError::Internal("inspected group has no Group Info message".into()).into())
}

/// Opens an editing session with the read access used by the group-link crosschecks.
///
/// When `source_only` is true, the editor retains a complete mirror and the reader uses
/// [`crate::Source`] for each read. When it is false, opens through [`File::open_rw`].
///
/// # Errors
///
/// Returns [`Error::Io`] if `path` cannot be opened for reading and writing. Returns locking,
/// format, and configuration errors from the editor.
pub(super) fn open_group_inspection(path: &Path, source_only: bool) -> Result<File, Error> {
    if source_only {
        File::open_rw_source_only(path)
    } else {
        File::open_rw(path)
    }
}
