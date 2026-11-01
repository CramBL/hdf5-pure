//! Group-link inspection, storage selection, and placement for staged edits.

use hdf5_pure_format::__private::DenseLinkPlan;
use hdf5_pure_format::__private::DenseLinkPlanError;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::GroupPhaseChange;
use hdf5_pure_format::__private::LinkCreationPolicy;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::OwnedLink;

use hdf5_pure_space::__private::AllocationClass;

use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::error::Error;
use crate::error::FormatError;
use crate::error::OBJECT_HEADER_MESSAGE_MAX;
use crate::file_writer::{self, OFFSET_WIDTH};
use crate::group_v1;
use crate::group_v2;
use crate::link_info::LinkInfoMessage;
use crate::link_message::LinkMessage;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::HeaderMessage;
use crate::object_header::ObjectHeader;
use crate::source::SourceMetadata;
use crate::symbol_table::SymbolTableMessage;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

use super::WriteEngine;
use super::group_links;
use super::object_header::{self, OhRegion};

impl WriteEngine {
    /// Converts the group whose version 1 object header is at `addr` to a version 2 message
    /// region.
    ///
    /// For a symbol-table group, the region is compact and holds one Link message per symbol
    /// table entry, hard or soft. For a group that stores its links in Link Info, Group Info, and
    /// Link messages, the region holds those messages, see [`link_message_region`]. Attribute
    /// bodies are copied in both cases, with shared-message references re-encoded. Other
    /// messages, including the symbol table, are omitted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the object is a dataset, the header stores a symbol
    /// table beside link storage messages, the links or their storage messages are inconsistent
    /// or cannot be reproduced, or an attribute is too large or has flags the conversion cannot
    /// preserve. Returns [`FormatError::PathNotFound`] if the object is not a group. Returns
    /// parsing and source-read errors as reported by the group, object-header, heap, and index
    /// readers.
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
        if !group_v2::is_group(&oh) {
            return Err(
                FormatError::PathNotFound(String::from("object header is not a group")).into(),
            );
        }
        let mut region = match oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
        {
            Some(message) => {
                if oh
                    .messages
                    .iter()
                    .any(|m| is_link_storage_message(m.msg_type))
                {
                    return Err(Error::EditUnsupported(
                        "a group mixes symbol-table storage with modern link storage",
                    ));
                }
                self.symbol_table_region(&SymbolTableMessage::parse_with_offset_width(
                    &message.data,
                    self.context.widths().offsets,
                )?)?
            }
            None => link_message_region(&oh, self.context.widths().offsets)?,
        };
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
        let links = group_links::inspect(&self.opened_source(), self.context.widths(), &region)?;
        Ok(GroupInfo { region, links })
    }
    /// Reads a group's message region and compact or dense links.
    ///
    /// Converts a symbol-table group through [`Self::reconstruct_v1_group`]. For a version 2
    /// header, retains encoded names and targets, collects continuation chunks, and appends a
    /// Group Info message if one is absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the object is a dataset, link metadata is
    /// inconsistent or unsupported, or a version 2 header has no Link Info message. Dense
    /// storage is limited to unfiltered managed objects with seven-byte heap IDs and encoded
    /// links of at most 4096 bytes. Returns parsing and source-read errors from the object
    /// header, heap, and index readers.
    pub(super) fn inspect_group(&self, addr: u64) -> Result<GroupInfo, Error> {
        let sig = self.image().read_metadata_at(addr, 4);
        if sig.as_deref() != Ok(&b"OHDR"[..]) {
            return self.reconstruct_v1_group(addr);
        }
        let mut region = Self::gather_oh_messages(&self.image(), addr, self.context.addresses())?;
        let links = group_links::inspect(&self.opened_source(), self.context.widths(), &region)?;
        ensure_group_info(&mut region)?;
        Ok(GroupInfo { region, links })
    }

    /// Returns a compact-link region with one Link message per entry of the version 1 group
    /// `symbol_table` describes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a link cannot be reproduced as a Link message, and
    /// the parsing and source-read errors of the group readers.
    fn symbol_table_region(&self, symbol_table: &SymbolTableMessage) -> Result<OhRegion, Error> {
        let widths = self.context.widths();
        let addresses = self.context.addresses();
        let k = group_v1::group_leaf_node_k(self.context.superblock());
        let links = match self.image_slice() {
            Some(bytes) => {
                group_v1::resolve_v1_group_links(bytes, symbol_table, widths, addresses, k)
            }
            None => group_v1::resolve_v1_group_links_from_source(
                self.image(),
                symbol_table,
                widths,
                addresses,
                k,
            ),
        }
        .map_err(group_links::storage_error)?;
        let mut region = fresh_group_region();
        for link in links {
            let body = link
                .serialize(OFFSET_WIDTH)
                .map_err(group_links::storage_error)?;
            if body.len() > OBJECT_HEADER_MESSAGE_MAX {
                return Err(Error::EditUnsupported(
                    "a version 1 group's link is too large to convert in place",
                ));
            }
            region.push(MessageType::LINK, &body);
        }
        Ok(region)
    }
}

/// A group's message region and links with their encoded names, targets, and creation order.
pub(super) struct GroupInfo {
    pub(super) region: OhRegion,
    pub(super) links: GroupLinkStorage,
}

impl GroupInfo {
    /// Returns inline Link records and UTF-8 names for staged editing.
    ///
    /// Expands dense links into the region and clears their storage addresses while preserving
    /// the creation-order policy, counter, and per-link indexes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a link name is not valid UTF-8. Returns a
    /// serialization error if a link cannot be encoded.
    pub(super) fn into_compact(self) -> Result<(OhRegion, Vec<String>), Error> {
        let Self {
            mut region,
            mut links,
        } = self;
        let dense = links.info().fractal_heap_address.is_some();
        let names = links
            .links()
            .iter()
            .map(|link| {
                String::from_utf8(link.name().to_vec()).map_err(|_error| {
                    Error::EditUnsupported(
                        "a group's link name cannot be represented by the editor's UTF-8 paths",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dense {
            links.use_compact_storage();
            region.replace_link_info(&links.info())?;
            for link in links.links() {
                region.push(
                    MessageType::LINK,
                    &link
                        .serialize(OFFSET_WIDTH)
                        .map_err(group_links::storage_error)?,
                );
            }
        }
        Ok((region, names))
    }
}

impl WriteEngine {
    /// Selects link storage from the final count and checks dense encoding before placement.
    ///
    /// Reads phase-change thresholds from Group Info, using the defaults when omitted.
    /// Uses the minimum dense count if `was_dense` is `true` and the maximum compact count
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if link metadata, thresholds, or the selected dense
    /// representation cannot be reproduced. Returns parsing and source-read errors from inspection.
    pub(super) fn plan_group_links(
        &self,
        region: &OhRegion,
        was_dense: bool,
    ) -> Result<bool, Error> {
        let links = group_links::inspect(&self.opened_source(), self.context.widths(), region)?;
        let mut pos = 0;
        let mut thresholds = group_links::phase_change(&GROUP_INFO_BODY)?;
        while let Some((kind, body, end)) = region.next_message(pos)? {
            if kind == MessageType::GROUP_INFO {
                thresholds = group_links::phase_change(&region[body..end])?;
                break;
            }
            pos = end;
        }
        let count = links.links().len();
        let dense = if was_dense {
            count >= usize::from(thresholds.min_dense())
        } else {
            count > usize::from(thresholds.max_compact())
        };
        if dense {
            dense_link_plan(links, self.context.widths().lengths)?;
        }
        Ok(dense)
    }

    /// Places dense link metadata and replaces inline Link records with its addresses.
    ///
    /// Preserves the region when `dense` is `false`. The caller passes the storage decision
    /// from [`Self::plan_group_links`] after resolving hard-link targets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if link storage cannot be reproduced or Link Info is
    /// absent. Returns inspection, serialization, allocation, and write errors during placement.
    pub(super) fn place_group_links(
        &mut self,
        region: &mut OhRegion,
        dense: bool,
    ) -> Result<(), Error> {
        if !dense {
            return Ok(());
        }
        let storage = group_links::inspect(&self.opened_source(), self.context.widths(), region)?;
        let plan = dense_link_plan(storage, self.context.widths().lengths)?;
        let (_, info) =
            self.place_relocatable(plan.serialized_len(), AllocationClass::Metadata, |base| {
                let image = plan.serialize(base).map_err(dense_link_error)?;
                Ok((image.bytes, image.info))
            })?;
        let mut bytes = Vec::new();
        let mut pos = 0;
        while let Some((kind, _, end)) = region.next_message(pos)? {
            if kind != MessageType::LINK {
                bytes.extend_from_slice(&region[pos..end]);
            }
            pos = end;
        }
        *region = region.with_bytes(bytes);
        region.replace_link_info(&info)
    }
}

fn dense_link_plan(
    storage: GroupLinkStorage,
    length_width: LengthWidth,
) -> Result<DenseLinkPlan, Error> {
    DenseLinkPlan::new(storage, length_width).map_err(dense_link_error)
}

fn dense_link_error(error: DenseLinkPlanError) -> Error {
    match error {
        DenseLinkPlanError::Format(error) => error.into(),
        _ => Error::EditUnsupported("a group's resulting links cannot be encoded in dense storage"),
    }
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

/// Returns a region with the link storage messages of the version 1 header `object_header`, and a
/// Group Info message if the header has none.
///
/// Each message is copied with its type and flags, and with its body stripped of the padding of
/// the version 1 header, see [`unpadded_body`].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a link storage message has a flag other than the
/// constant flag, and the errors of [`unpadded_body`] and of [`ensure_group_info`].
pub(super) fn link_message_region(
    object_header: &ObjectHeader,
    offset_width: OffsetWidth,
) -> Result<OhRegion, Error> {
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    for message in object_header
        .messages
        .iter()
        .filter(|m| is_link_storage_message(m.msg_type))
    {
        if !message.flags.difference(MessageFlags::CONSTANT).is_empty() {
            return Err(Error::EditUnsupported(
                "a group's link metadata has object-header flags this editor cannot reproduce",
            ));
        }
        // A version 1 header stores each message size in two bytes, so no body exceeds
        // `OBJECT_HEADER_MESSAGE_MAX` and the record encoder cannot panic.
        let record = region.layout().record_with_creation_index(
            message.msg_type,
            message.flags,
            0,
            unpadded_body(message, offset_width)?,
        );
        region.push_bytes(&record);
    }
    ensure_group_info(&mut region)?;
    Ok(region)
}

/// Returns the body of the link storage message `message` without its trailing padding.
///
/// In a version 1 object header, each message body is padded to a multiple of eight bytes and the
/// padding is included in the message size, as defined in "Version 1 Data Object Header Prefix"
/// of the [format specification, version 4.0][spec]. The length of the body is read from the
/// message's own fields.
///
/// # Errors
///
/// Returns the parse errors of the Link Info, Group Info, or Link message, and
/// [`FormatError::Internal`] if `message` is not a link storage message.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
fn unpadded_body(message: &HeaderMessage, offset_width: OffsetWidth) -> Result<&[u8], Error> {
    let data = message.data.as_slice();
    let len = match message.msg_type {
        MessageType::LINK_INFO => LinkInfoMessage::parse_with_offset_width(data, offset_width)?
            .serialize(offset_width)
            .len(),
        MessageType::GROUP_INFO => {
            GroupPhaseChange::encoded_len(data).map_err(group_links::group_info_error)?
        }
        MessageType::LINK => {
            OwnedLink::parse_padded(data, offset_width)
                .map_err(group_links::storage_error)?
                .1
        }
        _ => {
            return Err(FormatError::Internal(
                "only link storage messages have their padding removed".into(),
            )
            .into());
        }
    };
    Ok(data.get(..len).ok_or_else(|| {
        FormatError::Internal("a parsed link storage message is longer than its body".into())
    })?)
}

fn is_link_storage_message(msg_type: MessageType) -> bool {
    matches!(
        msg_type,
        MessageType::LINK_INFO | MessageType::GROUP_INFO | MessageType::LINK
    )
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
    /// The tracking and indexing policy with a checked next-index counter.
    policy: LinkCreationPolicy,
    /// Whether an index has been assigned and the counter needs writing.
    assigned: bool,
}

impl LinkCreationOrder {
    /// Reads the group's next link creation index from its Link Info message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a message body extends past the region, indexing is
    /// enabled without tracking, or the counter exceeds [`i64::MAX`].
    pub(super) fn for_region(region: &OhRegion) -> Result<Self, Error> {
        let policy = find_link_info(region)?
            .map(|(_, _, info)| LinkCreationPolicy::from_info(&info))
            .transpose()
            .map_err(|_error| {
                Error::EditUnsupported("a group's link creation-order policy cannot be reproduced")
            })?
            .unwrap_or(LinkCreationPolicy::Untracked);
        Ok(Self {
            policy,
            assigned: false,
        })
    }

    /// Returns the next creation index, or `None` for an untracked group.
    ///
    /// Assigns indexes in insertion order and increments the counter after each assignment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the counter is [`i64::MAX`]. An error leaves the
    /// assignment state unchanged.
    pub(super) fn take(&mut self) -> Result<Option<u64>, Error> {
        let Self { policy, assigned } = self;
        let index = policy.take().map_err(|_error| {
            Error::EditUnsupported(
                "a group has assigned every link creation index its link-info message can record",
            )
        })?;
        *assigned |= index.is_some();
        Ok(index)
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
        let Some(next) = self.policy.next_index() else {
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

    fn replace_link_info(&mut self, info: &LinkInfoMessage) -> Result<(), Error> {
        let splice = self
            .splice_first(MessageType::LINK_INFO)?
            .ok_or(Error::EditUnsupported(
                "a target group's object header has no link-info message",
            ))?;
        *self = self.with_bytes(splice.replace_body(&info.serialize(OFFSET_WIDTH))?);
        Ok(())
    }
}
