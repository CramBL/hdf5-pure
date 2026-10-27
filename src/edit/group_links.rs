//! Reads compact and dense group links with their encoded names, targets, and creation order.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::BTREE_V2_LINK_CREATION_ORDER;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::BTreeV2LinkNameIndex;
use hdf5_pure_format::__private::BTreeV2Node;
use hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::FractalHeapLayout;
use hdf5_pure_format::__private::GROUP_HEAP_MAX_MANAGED_OBJECT;
use hdf5_pure_format::__private::GroupInfoError;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::GroupPhaseChange;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::LinkStorageError;
use hdf5_pure_format::__private::OwnedLink;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::Error;
use crate::error::FormatError;
use crate::fractal_heap;
use crate::fractal_heap::HeapObjectReader;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::BaseOffsetSource;
use crate::source::RangeReader;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::FormatWidths;

use super::object_header::OhRegion;

/// Reads the group's links and checks their storage metadata for editing.
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
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
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
                links.push(OwnedLink::parse(record.body, widths.offsets).map_err(storage_error)?);
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
        .map_err(storage_error)?
        .info();
    if let Some(heap) = info.fractal_heap_address {
        if !links.is_empty() {
            return Err(Error::EditUnsupported(
                "a group mixes compact and dense links",
            ));
        }
        links = read_dense(source, addresses, widths, &info, heap)?;
    }
    GroupLinkStorage::new(&info, links, widths.offsets).map_err(storage_error)
}

fn read_dense(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    widths: FormatWidths,
    info: &LinkInfoMessage,
    heap_address: StoredAddress,
) -> Result<Vec<OwnedLink>, Error> {
    let framed = BaseOffsetSource {
        inner: source,
        base: addresses.effective_base(),
    };
    let header = fractal_heap::read_header_from_source(&framed, addresses, heap_address, widths)?;
    if header.filtering.is_filtered()
        || header.huge_objects_count != 0
        || header.tiny_objects_count != 0
        || header.huge_objects_size != 0
        || header.tiny_objects_size != 0
        || !header
            .btree_huge_objects_address
            .is_undefined(widths.offsets)
    {
        return Err(Error::EditUnsupported(
            "a group's fractal heap uses filters, huge objects, or tiny objects",
        ));
    }
    if header.flags & !FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED != 0 {
        return Err(Error::EditUnsupported(
            "a group's fractal heap uses unsupported flags",
        ));
    }
    FractalHeapLayout::new(&header, widths.offsets, widths.lengths)
        .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
    let name_address = info
        .btree_name_index_address
        .ok_or(Error::EditUnsupported("a dense group has no name index"))?;
    let name_header = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(&framed),
        name_address.get(),
        widths,
    )?;
    let index = BTreeV2LinkNameIndex::new(&name_header, widths.offsets, header.heap_id_length)?;
    if name_header.record_size != NAME_RECORD_SIZE || header.heap_id_length != LINK_HEAP_ID_LENGTH {
        return Err(Error::EditUnsupported(
            "a group's name index uses an unsupported record encoding",
        ));
    }
    let ids = FractalHeapIdLayout::new(&header, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(
        &header,
        addresses,
        heap_address,
        AccessMode::ReadWrite,
        widths,
    );
    let mut links = Vec::new();
    let mut creation = BTreeMap::new();
    let mut previous_hash = None;
    walk_index(source, addresses, index.nodes(), |record| {
        let (hash, id) = index.record(record)?;
        let view = ids.parse(id)?;
        if id.first() != Some(&MANAGED_HEAP_ID_FLAGS)
            || !matches!(view.kind(), FractalHeapIdKind::Managed { .. })
        {
            return Err(Error::EditUnsupported(
                "a group's link uses a non-managed heap ID",
            ));
        }
        let bytes = heap.read_from_source(&framed, view)?;
        let link = OwnedLink::parse(&bytes, widths.offsets).map_err(storage_error)?;
        if link.encoded_len(widths.offsets).map_err(storage_error)?
            > GROUP_HEAP_MAX_MANAGED_OBJECT.to_usize()?
        {
            return Err(Error::EditUnsupported(
                "a group's link exceeds the supported managed object size",
            ));
        }
        if checksum::jenkins_lookup3(link.name()) != hash
            || previous_hash.zip(links.last()).is_some_and(
                |(previous_hash, previous): (u32, &OwnedLink)| {
                    (previous_hash, previous.name()) >= (hash, link.name())
                },
            )
        {
            return Err(Error::EditUnsupported(
                "a group's name index has inconsistent keys",
            ));
        }
        previous_hash = Some(hash);
        if let Some(order) = link.creation_order() {
            if creation.insert(order, id.to_vec()).is_some() {
                return Err(Error::EditUnsupported(
                    "a group's links have repeated creation indexes",
                ));
            }
        }
        links.push(link);
        Ok(())
    })?;
    if header.managed_objects_count != links.len().to_u64() {
        return Err(Error::EditUnsupported(
            "a group's heap and name index have different link counts",
        ));
    }
    if let Some(Some(address)) = info.btree_creation_order_address {
        read_creation_index(source, addresses, widths, address, creation)?;
    }
    Ok(links)
}

fn read_creation_index(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    widths: FormatWidths,
    address: StoredAddress,
    mut creation: BTreeMap<u64, Vec<u8>>,
) -> Result<(), Error> {
    let framed = BaseOffsetSource {
        inner: source,
        base: addresses.effective_base(),
    };
    let header = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(&framed),
        address.get(),
        widths,
    )?;
    if header.tree_type != BTREE_V2_LINK_CREATION_ORDER
        || header.record_size != CREATION_RECORD_SIZE
    {
        return Err(Error::EditUnsupported(
            "a group's creation-order index uses an unsupported record encoding",
        ));
    }
    let mut previous = None;
    let layout = BTreeV2Layout::new(&header, widths.offsets)?;
    walk_index(source, addresses, &layout, |record| {
        let (index, id) = record
            .split_first_chunk::<LINK_CREATION_ORDER_LENGTH>()
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        let index = u64::from_le_bytes(*index);
        if previous.is_some_and(|previous| previous >= index)
            || creation.remove(&index).as_deref() != Some(id)
        {
            return Err(Error::EditUnsupported(
                "a group's creation-order index disagrees with its links",
            ));
        }
        previous = Some(index);
        Ok(())
    })?;
    if !creation.is_empty() {
        return Err(Error::EditUnsupported(
            "a group's creation-order index omits links",
        ));
    }
    Ok(())
}

fn walk_index(
    source: &(impl Source + ?Sized),
    addresses: AddressContext,
    layout: &BTreeV2Layout,
    mut read_record: impl FnMut(&[u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    if let Some(root) = layout.root() {
        IndexReader {
            source,
            addresses,
            layout,
            visited: BTreeSet::new(),
        }
        .walk(root, &mut read_record)?;
    }
    Ok(())
}

/// Reads one index while retaining the addresses of visited nodes.
struct IndexReader<'a, S: Source + ?Sized> {
    source: &'a S,
    addresses: AddressContext,
    layout: &'a BTreeV2Layout,
    visited: BTreeSet<StoredAddress>,
}

impl<S: Source + ?Sized> IndexReader<'_, S> {
    fn walk(
        &mut self,
        node: BTreeV2Node,
        read_record: &mut impl FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<u64, Error> {
        let Self {
            source,
            addresses,
            layout,
            visited,
        } = self;
        if !visited.insert(node.address()) {
            return Err(FormatError::InvalidBTreeV2Signature.into());
        }
        let range = FileRange::new(
            addresses.resolve(node.address())?,
            layout.encoded_len(node)?.to_u64(),
        )?;
        let bytes = source.read_range(range)?;
        let view = layout.parse(&bytes, node)?;
        let mut count = 0u64;
        for record in 0..view.record_count() {
            if let Some(child) = view.child(record)? {
                count = count
                    .checked_add(self.walk(child, read_record)?)
                    .ok_or(FormatError::InvalidBTreeV2Signature)?;
            }
            read_record(view.record(record)?)?;
            count = count
                .checked_add(1)
                .ok_or(FormatError::InvalidBTreeV2Signature)?;
        }
        if let Some(child) = view.child(view.record_count())? {
            count = count
                .checked_add(self.walk(child, read_record)?)
                .ok_or(FormatError::InvalidBTreeV2Signature)?;
        }
        if count != node.total_records() {
            return Err(FormatError::InvalidBTreeV2Signature.into());
        }
        Ok(count)
    }
}

/// Reads the Group Info thresholds and converts parse errors for the editor.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the message has an unsupported encoding or its
/// minimum dense threshold exceeds its maximum compact threshold.
pub(super) fn phase_change(bytes: &[u8]) -> Result<GroupPhaseChange, Error> {
    GroupPhaseChange::parse(bytes).map_err(|error| match error {
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
    })
}

pub(super) fn storage_error(error: LinkStorageError) -> Error {
    match error {
        LinkStorageError::Format(error) => error.into(),
        LinkStorageError::CreationOrderCounterRange { .. }
        | LinkStorageError::IndexingWithoutTracking => {
            Error::EditUnsupported("a group's link creation-order policy cannot be reproduced")
        }
        _ => Error::EditUnsupported("a group's links or link-info metadata cannot be reproduced"),
    }
}

// "Version 2 B-trees" in the format specification, version 4.0, defines seven-byte heap IDs
// in type 5 and type 6 records.
const LINK_HEAP_ID_LENGTH: u16 = 7;
// "Version 2 B-trees" in the format specification, version 4.0, defines a four-byte name hash
// before the heap ID in type 5 records.
const NAME_RECORD_SIZE: u16 = 4 + LINK_HEAP_ID_LENGTH;
// "Version 2 B-trees" in the format specification, version 4.0, defines an eight-byte creation
// index before the heap ID in type 6 records.
const CREATION_RECORD_SIZE: u16 = 8 + LINK_HEAP_ID_LENGTH;
const LINK_CREATION_ORDER_LENGTH: usize = 8;
// "Fractal Heap" in the format specification, version 4.0, defines zero for the version and
// type bits of a version 0 managed heap ID.
const MANAGED_HEAP_ID_FLAGS: u8 = 0;
