//! Compact-link group inspection and rewriting for staged edits.

use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;

use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::error::{Error, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_writer::{self, OFFSET_WIDTH};
use crate::group_v1;
use crate::group_v2;
use crate::link_info::LinkInfoMessage;
use crate::link_message::LinkMessage;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source::SourceMetadata;
use crate::symbol_table::SymbolTableMessage;

use super::WriteEngine;
use super::object_header::{self, OhRegion};

impl WriteEngine {
    /// Converts a symbol-table group to a compact-link version 2 message region.
    ///
    /// Preserves the existing links and attribute bodies, re-encoding shared-message references.
    /// Other messages, including the symbol table, are omitted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the object is a dataset, the group contains a soft
    /// link, or an attribute is too large or has flags the conversion cannot preserve. Returns
    /// parsing and source-read errors as reported by the group and object-header readers.
    pub(super) fn reconstruct_v1_group(&self, addr: u64) -> Result<GroupInfo, Error> {
        let base = self.context.addresses();
        let oh = ObjectHeader::parse_from_source_with_widths(
            &SourceMetadata(&self.image()),
            AccessMode::ReadWrite,
            addr,
            self.context.widths(),
            base.effective_base(),
        )?;
        if oh
            .messages
            .iter()
            .any(|m| m.msg_type == MessageType::DATA_LAYOUT)
        {
            return Err(Error::EditUnsupported(
                "a target path names a dataset, not a group",
            ));
        }
        if let Some(message) = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
        {
            let symbol_table = SymbolTableMessage::parse_with_offset_width(
                &message.data,
                self.context.widths().offsets,
            )?;
            if group_v1::v1_group_holds_symbolic_link_from_source(
                &self.image(),
                &symbol_table,
                self.context.widths(),
                self.context.addresses(),
                group_v1::group_leaf_node_k(self.context.superblock()),
            )? {
                return Err(Error::EditUnsupported(
                    "a version 1 group holding a soft link cannot be converted in place",
                ));
            }
        }
        let entries = group_v2::resolve_group_entries_from_source(
            &self.image(),
            AccessMode::ReadWrite,
            &oh,
            self.context.widths(),
            self.context.addresses(),
            group_v1::group_leaf_node_k(self.context.superblock()),
        )?;

        let mut region = fresh_group_region();
        let mut link_names = Vec::with_capacity(entries.len());
        for e in &entries {
            // The new compact region does not track link creation order.
            region.push_link(&e.name, e.object_header_address, None);
            link_names.push(e.name.clone());
        }
        for m in &oh.messages {
            if m.msg_type == MessageType::ATTRIBUTE {
                if m.data.len() > OBJECT_HEADER_MESSAGE_MAX {
                    return Err(Error::EditUnsupported(
                        "a v0/v1 group attribute is too large to convert in place",
                    ));
                }
                if !m.flags.difference(MessageFlags::SHARED).is_empty() {
                    // The conversion supports only the shared flag.
                    return Err(Error::EditUnsupported(
                        "a v0/v1 group attribute message carries an object-header flag this \
                         conversion cannot reproduce",
                    ));
                }
                if m.flags.is_shared() {
                    // Re-encode the reference for the version 2 header before writing its record.
                    region.push_shared(
                        MessageType::ATTRIBUTE,
                        &object_header::modernize_shared_reference(&m.data, self.context.widths())?,
                    );
                    continue;
                }
                // Preserve the attribute body in the new record layout.
                region.push(MessageType::ATTRIBUTE, &m.data);
            }
        }
        Ok(GroupInfo { region, link_names })
    }
    /// Reads a compact-link group's message region and existing link names.
    ///
    /// Converts a symbol-table group through [`Self::reconstruct_v1_group`]. For a version 2
    /// header, collects continuation chunks and appends a Group Info message if one is absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the object is a dataset, the Link Info message
    /// names dense storage, or a version 2 header has no Link Info message. Returns errors from
    /// [`Self::gather_oh_messages`], [`Self::reconstruct_v1_group`], and [`ensure_group_info`].
    pub(super) fn inspect_group(&self, addr: u64) -> Result<GroupInfo, Error> {
        let sig = self.image().read_metadata_at(addr, 4);
        if sig.as_deref() != Ok(&b"OHDR"[..]) {
            return self.reconstruct_v1_group(addr);
        }
        let mut region = Self::gather_oh_messages(&self.image(), addr, self.context.addresses())?;
        let mut p = 0;
        let mut has_link_info = false;
        let mut link_names = Vec::new();
        while let Some((msg_type, body, body_end)) = region.next_message(p)? {
            match msg_type {
                MessageType::LINK_INFO => {
                    has_link_info = true;
                    // Keep the heap address read within this message body.
                    let mut q = body + 2;
                    if body_end - body >= 2 && region[body + 1] & 0x01 != 0 {
                        q += 8;
                    }
                    if q + 8 <= body_end {
                        let heap_addr = u64::from_le_bytes(region[q..q + 8].try_into().unwrap());
                        if heap_addr != u64::MAX {
                            return Err(Error::EditUnsupported(
                                "a target group uses dense (fractal-heap) link storage (not supported in place yet)",
                            ));
                        }
                    }
                }
                MessageType::LINK => {
                    if let Ok(link) =
                        LinkMessage::parse_with_offset_width(&region[body..body_end], OFFSET_WIDTH)
                    {
                        link_names.push(link.name);
                    }
                }
                MessageType::DATA_LAYOUT => {
                    return Err(Error::EditUnsupported(
                        "a target path names a dataset, not a group",
                    ));
                }
                _ => {}
            }
            p = body_end;
        }
        if !has_link_info {
            return Err(Error::EditUnsupported(
                "a target group's object header has no link-info message",
            ));
        }
        ensure_group_info(&mut region)?;
        Ok(GroupInfo { region, link_names })
    }
}

/// A group's message region and existing link names.
pub(super) struct GroupInfo {
    pub(super) region: OhRegion,
    pub(super) link_names: Vec<String>,
}

/// The version 0 Group Info body with both sets of optional fields omitted.
///
/// The fields are defined in "The Group Info Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
pub(super) const GROUP_INFO_BODY: [u8; 2] = [0, 0];

/// Creates an empty compact-link region with Link Info and Group Info messages.
pub(super) fn fresh_group_region() -> OhRegion {
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    region.push(
        MessageType::LINK_INFO,
        &crate::file_writer::COMPACT_LINK_INFO.serialize(OFFSET_WIDTH),
    );
    region.push(MessageType::GROUP_INFO, &GROUP_INFO_BODY);
    region
}

/// Appends a minimal Group Info message if one is absent.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a message body extends past the region.
pub(super) fn ensure_group_info(region: &mut OhRegion) -> Result<(), Error> {
    let mut p = 0;
    while let Some((msg_type, _body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::GROUP_INFO {
            return Ok(());
        }
        p = body_end;
    }
    region.push(MessageType::GROUP_INFO, &GROUP_INFO_BODY);
    Ok(())
}

/// Serializes a hard link using the caller's message-record layout.
///
/// `creation_order` is the link's creation index, separate from the header record's attribute
/// creation index.
///
/// # Panics
///
/// Panics if the serialized Link body is longer than [`OBJECT_HEADER_MESSAGE_MAX`].
pub(super) fn encode_link_message(
    layout: MessageRecordLayout,
    name: &str,
    addr: StoredAddress,
    creation_order: Option<u64>,
) -> Vec<u8> {
    let mut link = file_writer::make_link(name, addr);
    link.creation_order = creation_order;
    let body = link.serialize(crate::file_writer::OFFSET_WIDTH);
    layout.record(MessageType::LINK, &body)
}

/// Replaces the address of the hard link named `name`, preserving its other fields.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a message body extends past the region, the named
/// link is absent or symbolic, or its body has bytes after the address field. Returns an
/// address-encoding error if `new_addr` does not fit the editor's address width.
pub(super) fn patch_link_target(
    region: &mut OhRegion,
    name: &str,
    new_addr: StoredAddress,
) -> Result<(), Error> {
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::LINK {
            if let Ok(link) =
                LinkMessage::parse_with_offset_width(&region[body..body_end], OFFSET_WIDTH)
                && link.name == name
            {
                let Some(address) =
                    LinkMessage::hard_link_address_range(&region[body..body_end], OFFSET_WIDTH)?
                else {
                    return Err(Error::EditUnsupported(
                        "a group on the edited path is reached by a soft/external link",
                    ));
                };
                // The editor accepts hard links only when the address is the final body field.
                if body + address.end != body_end {
                    return Err(Error::EditUnsupported(
                        "a link message holds bytes past its hard link address",
                    ));
                }
                hdf5_pure_format::__private::write_stored_address_at(
                    region.bytes_mut(),
                    body + address.start,
                    new_addr,
                    OFFSET_WIDTH,
                )?;
                return Ok(());
            }
        }
        p = body_end;
    }
    Err(Error::EditUnsupported(
        "expected child link not found in parent group",
    ))
}

/// The creation counter for links added to one group in a commit.
///
/// The counter is read from the Link Info message, separate from the attribute creation order
/// in the object-header prefix. An untracked group has no counter. `H5G_obj_insert` assigns
/// the recorded maximum and then increments it (`H5Gobj.c`, HDF5 2.2.0).
pub(super) struct LinkCreationOrder {
    /// The next index to assign, or `None` for an untracked group.
    next: Option<u64>,
    /// Whether an index has been assigned and the counter needs writing.
    assigned: bool,
}

impl LinkCreationOrder {
    /// Reads the group's next link creation index from its Link Info message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a message body extends past the region.
    pub(super) fn for_region(region: &OhRegion) -> Result<Self, Error> {
        let next = find_link_info(region)?.and_then(|(_, _, info)| info.max_creation_order);
        Ok(Self {
            next,
            assigned: false,
        })
    }

    /// Returns the next creation index, or `None` for an untracked group.
    ///
    /// Assigns indexes in insertion order and increments the counter after each assignment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the counter cannot be incremented.
    pub(super) fn take(&mut self) -> Result<Option<u64>, Error> {
        let Some(index) = self.next else {
            return Ok(None);
        };
        self.next = Some(index.checked_add(1).ok_or(Error::EditUnsupported(
            "a group has assigned every link creation index its link-info message can record",
        ))?);
        self.assigned = true;
        Ok(Some(index))
    }

    /// Writes the incremented counter into the group's Link Info message.
    ///
    /// Preserves the record if no index has been assigned. The caller passes the region from
    /// which [`Self::for_region`] read the counter, with its Link Info message retained.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a message body extends past the region, or the Link
    /// Info message is absent or too short to store the counter.
    pub(super) fn record(&self, region: &mut OhRegion) -> Result<(), Error> {
        if !self.assigned {
            return Ok(());
        }
        let Some(next) = self.next else {
            return Ok(());
        };
        let (_, body, _) = find_link_info(region)?.ok_or(Error::EditUnsupported(
            "a group's link-info message went missing while its links were being added",
        ))?;
        let max = body.start + 2..body.start + 10;
        if max.end > body.end {
            return Err(Error::EditUnsupported(
                "a group's link-info message is too short to record its link creation order",
            ));
        }
        region.bytes_mut()[max].copy_from_slice(&next.to_le_bytes());
        Ok(())
    }
}

/// Returns the first parsed Link Info message, with its record start and body range.
///
/// Returns `None` if every Link Info message is absent or unparseable.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a message body extends past the region.
pub(super) fn find_link_info(
    region: &OhRegion,
) -> Result<Option<(usize, core::ops::Range<usize>, LinkInfoMessage)>, Error> {
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::LINK_INFO
            && let Ok(info) =
                LinkInfoMessage::parse_with_offset_width(&region[body..body_end], OFFSET_WIDTH)
        {
            return Ok(Some((p, body..body_end, info)));
        }
        p = body_end;
    }
    Ok(None)
}

/// Returns the maximum compact link count from the Group Info message.
///
/// Uses 8 if the message is absent, omits the threshold, or is too short to store it. The
/// threshold in a version 0 Group Info message is defined in "The Group Info Message" of the
/// [format specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a message body extends past the region.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
pub(super) fn max_compact_links(region: &OhRegion) -> Result<u16, Error> {
    /// The `H5G_CRT_GINFO_MAX_COMPACT` default (`H5Gprivate.h`, HDF5 2.2.0).
    const DEFAULT_MAX_COMPACT: u16 = 8;
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::GROUP_INFO {
            let stores_phase_change = body_end - body >= 2 && region[body + 1] & 0x01 != 0;
            if stores_phase_change && body + 4 <= body_end {
                return Ok(u16::from_le_bytes([region[body + 2], region[body + 3]]));
            }
            return Ok(DEFAULT_MAX_COMPACT);
        }
        p = body_end;
    }
    Ok(DEFAULT_MAX_COMPACT)
}

/// Rejects additions that exceed a tracked group's compact link threshold.
///
/// `links_after_commit` includes this commit's additions and deletions. The editor supports
/// compact link creation order and cannot write dense link creation-order indexes.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a message body extends past the region, or the group
/// tracks creation order and `links_after_commit` exceeds [`max_compact_links`].
pub(super) fn reject_dense_link_creation_order(
    region: &OhRegion,
    links_after_commit: usize,
) -> Result<(), Error> {
    let tracked =
        find_link_info(region)?.is_some_and(|(_, _, info)| info.max_creation_order.is_some());
    if tracked && links_after_commit > usize::from(max_compact_links(region)?) {
        return Err(Error::EditUnsupported(
            "a group on the edited path tracks link creation order, and the links this commit \
             adds to it would take it past the compact storage its group-info message allows; \
             dense (fractal-heap) link storage cannot be written in place yet",
        ));
    }
    Ok(())
}

/// Copies the region without Link messages named `name`, preserving every other byte.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a message body extends past the region or the named
/// link is absent.
pub(super) fn remove_link_from_region(region: &OhRegion, name: &str) -> Result<OhRegion, Error> {
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut removed = false;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        let mut skip = false;
        if msg_type == MessageType::LINK {
            if let Ok(link) =
                LinkMessage::parse_with_offset_width(&region[body..body_end], OFFSET_WIDTH)
            {
                if link.name == name {
                    skip = true;
                    removed = true;
                }
            }
        }
        if !skip {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if p < region.len() {
        out.extend_from_slice(&region[p..]);
    }
    if !removed {
        return Err(Error::EditUnsupported(
            "link to delete not found in its parent group",
        ));
    }
    Ok(region.with_bytes(out))
}

impl OhRegion {
    /// Appends a hard Link record with an optional link creation index.
    ///
    /// The caller obtains `creation_order` from [`LinkCreationOrder`].
    ///
    /// # Panics
    ///
    /// Panics if the serialized Link body is longer than [`OBJECT_HEADER_MESSAGE_MAX`].
    pub(super) fn push_link(
        &mut self,
        name: &str,
        addr: StoredAddress,
        creation_order: Option<u64>,
    ) {
        let record = encode_link_message(self.layout(), name, addr, creation_order);
        self.push_bytes(&record);
    }
}
