//! Group child lookup, hard link listings, and path resolution through hard and soft links.
//!
//! Mixed-format readers parse the original image with its captured address context. Dense-storage
//! readers read at stored coordinates relative to the opened image's effective base.

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::NonZeroU16;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::BTreeV2LinkNameIndex;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::{BTREE_V2_LINK_CREATION_ORDER, BTREE_V2_LINK_NAME};
use hdf5_pure_space::__private::Extent;

#[cfg(test)]
use crate::UserblockSize;
use crate::access_mode::AccessMode;
#[cfg(test)]
use crate::address::BaseAddress;
#[cfg(test)]
use crate::address::BaseAddressExt;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, StoredAddress};
use crate::btree_v2::BTreeV2Source;
use crate::btree_v2::{self, BTreeV2Header};
use crate::checksum;
use crate::convert::Narrow;
use crate::dense_storage::{self, DenseIndex};
use crate::error::{FormatError, ResolveError};
use crate::fractal_heap;
use crate::fractal_heap::HeapObjectReader;
use crate::fractal_heap::HeapSource;
use crate::group_v1::{self, GroupEntry};
use crate::link_info::LinkInfoMessage;
use crate::link_message::{self, LinkMessage, LinkTarget, SymbolicLinkTarget};
use crate::message_type::MessageType;
use crate::object_header::{MessageFilter, ObjectHeader};
use crate::object_path::{LinkName, ObjectPath, ObjectPathBuf, SoftLinkPath};
use crate::source::OriginalSource;
use crate::source::PhysicalBytes;
use crate::source::PhysicalSource;
use crate::source::SourceMetadata;
use crate::source::StoredBytes;
use crate::source::StoredRead;
use crate::superblock::Superblock;
use crate::symbol_table::SymbolTableMessage;
use crate::width::{FormatWidths, OffsetWidth};

#[cfg(feature = "std")]
pub(crate) mod storage;

#[cfg(test)]
mod test_util;

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    #[cfg(test)]
    mod native_crosschecks;
}

/// Returns the physical object-header position reached through hard and soft links.
///
/// Resolves `path` from the root using the address context captured in `physical`.
///
/// # Errors
///
/// Returns errors from address resolution, [`resolve_link_path`], and group metadata parsing.
pub(crate) fn resolve_read_path_any(
    physical: PhysicalBytes<'_>,
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let addresses = physical.addresses();
    let root = addresses.resolve(superblock.root_group_address)?.get();
    resolve_link_path(root, root, &ObjectPathBuf::root(), path, |address, name| {
        find_child_target(
            physical,
            access_mode,
            address,
            widths,
            group_v1::group_leaf_node_k(superblock),
            name,
        )
    })
}

/// Returns the physical object-header position reached through hard and soft links in `source`.
///
/// Resolves `path` from the root using the opened backend's address context. Reads metadata
/// through the original source's metadata operation.
///
/// # Errors
///
/// Returns errors from address resolution, [`resolve_link_path`], group metadata parsing, and
/// source metadata reads.
pub(crate) fn resolve_read_path_any_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let addresses = source.addresses();
    let root = addresses.resolve(superblock.root_group_address)?.get();
    resolve_link_path(root, root, &ObjectPathBuf::root(), path, |address, name| {
        find_child_target_from_source(
            source,
            access_mode,
            address,
            widths,
            group_v1::group_leaf_node_k(superblock),
            name,
        )
    })
}

/// Resolves `path` from `start`, or from `root` for an absolute path.
///
/// `H5G__traverse_slink` resolves relative targets from their containing group, and
/// `H5G__traverse_real` resolves absolute targets from the root group (`H5Gtraverse.c`,
/// HDF5 1.14.6).
///
/// `base` is the root-relative path of `start`, used to report errors. At most 64 soft links
/// are followed.
///
/// # Errors
///
/// Returns [`ResolveError::NotFound`] for a missing component and [`ResolveError::NotAGroup`]
/// for an intermediate non-group object. Returns [`FormatError::DanglingSoftLink`] for a
/// missing target component, [`FormatError::SoftLinkTraversalLimit`] on exceeding the
/// traversal limit, and [`FormatError::ExternalLinkUnsupported`] for an external link.
/// Returns errors from `lookup` for malformed group metadata or link targets.
pub(crate) fn resolve_link_path(
    root: u64,
    start: u64,
    base: &ObjectPathBuf,
    path: &ObjectPath<'_>,
    lookup: impl FnMut(u64, LinkName<'_>) -> Result<ChildLookup, FormatError>,
) -> Result<u64, ResolveError> {
    LinkResolver {
        root,
        traversed: 0,
        lookup,
    }
    .resolve(start, base, path)
}

/// Resolves nested soft targets with a shared traversal count.
struct LinkResolver<F> {
    root: u64,
    traversed: u32,
    lookup: F,
}

impl<F: FnMut(u64, LinkName<'_>) -> Result<ChildLookup, FormatError>> LinkResolver<F> {
    fn resolve(
        &mut self,
        start: u64,
        base: &ObjectPathBuf,
        path: &ObjectPath<'_>,
    ) -> Result<u64, ResolveError> {
        let root_path = ObjectPathBuf::root();
        let (mut address, base) = if path.is_absolute() {
            (self.root, &root_path)
        } else {
            (start, base)
        };
        for (walked, &name) in path.components().iter().enumerate() {
            address = match (self.lookup)(address, name)? {
                ChildLookup::Found(ChildTarget::Hard(address)) => address.get(),
                ChildLookup::Found(ChildTarget::Soft(target)) => {
                    if self.traversed == SOFT_LINK_TRAVERSAL_LIMIT {
                        return Err(FormatError::SoftLinkTraversalLimit {
                            path: base.join_path(&path.prefix(walked + 1)).to_string(),
                            limit: SOFT_LINK_TRAVERSAL_LIMIT,
                        }
                        .into());
                    }
                    self.traversed += 1;
                    match self.resolve(
                        address,
                        &base.join_path(&path.prefix(walked)),
                        &target.as_path(),
                    ) {
                        Err(ResolveError::NotFound(_)) => {
                            return Err(FormatError::DanglingSoftLink {
                                path: base.join_path(&path.prefix(walked + 1)).to_string(),
                                target: target.to_string(),
                            }
                            .into());
                        }
                        result => result?,
                    }
                }
                ChildLookup::Found(ChildTarget::External {
                    filename,
                    object_path,
                }) => {
                    return Err(FormatError::ExternalLinkUnsupported {
                        path: base.join_path(&path.prefix(walked + 1)).to_string(),
                        filename,
                        object_path,
                    }
                    .into());
                }
                ChildLookup::Absent => {
                    return Err(ResolveError::NotFound(
                        base.join_path(&path.prefix(walked + 1)).to_string(),
                    ));
                }
                ChildLookup::NotAGroup => {
                    return Err(ResolveError::NotAGroup(
                        base.join_path(&path.prefix(walked)).to_string(),
                    ));
                }
            };
        }
        Ok(address)
    }
}

const SOFT_LINK_TRAVERSAL_LIMIT: u32 = 64;

/// Returns a dense group's version 2 B-tree allocations in stored-address order.
///
/// Reads each index through [`StoredRead::read_metadata`] using the captured effective base.
/// Returns an empty vector if the group has no fractal heap or name index.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if either index has the wrong client type. Returns
/// the structural error from the version 2 B-tree storage walker if an index cannot be enumerated
/// completely. Returns [`FormatError::InvalidBTreeV2Signature`] if the two indexes overlap or an
/// index allocation contains the fractal-heap header address. Returns errors from address
/// resolution and source metadata reads.
pub(crate) fn collect_dense_group_index_storage_extents(
    source: &(impl StoredRead + ?Sized),
    link_info: &LinkInfoMessage,
    widths: FormatWidths,
) -> Result<Vec<Extent>, FormatError> {
    let Some(fractal_heap_address) = link_info.fractal_heap_address else {
        return Ok(Vec::new());
    };
    let Some(name_address) = link_info.btree_name_index_address else {
        return Ok(Vec::new());
    };

    let mut indexes = Vec::with_capacity(2);
    indexes.push(DenseIndex::new(name_address, BTREE_V2_LINK_NAME));
    if let Some(Some(address)) = link_info.btree_creation_order_address {
        indexes.push(DenseIndex::new(address, BTREE_V2_LINK_CREATION_ORDER));
    }
    dense_storage::collect_dense_index_storage_extents_from_source(
        source,
        fractal_heap_address,
        indexes,
        widths,
    )
}

/// Returns one entry per hard link of a version 2 group.
///
/// Borrows dense metadata from `physical` using its captured effective base. Each entry holds
/// the child's stored object-header address.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if a dense group has no B-tree name index address, and
/// the [`FormatError`] of the first structure that does not parse. Returns errors from
/// [`PhysicalBytes::stored`] when constructing the stored view.
pub fn resolve_v2_group_entries(
    physical: PhysicalBytes<'_>,
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
) -> Result<Vec<GroupEntry>, FormatError> {
    let link_info = find_link_info(object_header, widths.offsets)?;

    if let Some(fh_addr) = link_info.fractal_heap_address {
        resolve_dense_entries(physical.stored()?, access_mode, &link_info, fh_addr, widths)
    } else {
        resolve_compact_entries(object_header, widths.offsets)
    }
}

/// Returns one entry per hard link stored directly in the object header.
///
/// # Errors
///
/// Returns errors from [`LinkMessage::parse_with_offset_width`].
fn resolve_compact_entries(
    object_header: &ObjectHeader,
    offset_width: OffsetWidth,
) -> Result<Vec<GroupEntry>, FormatError> {
    let mut entries = Vec::new();
    for msg in &object_header.messages {
        if msg.msg_type == MessageType::LINK {
            let link = LinkMessage::parse_with_offset_width(&msg.data, offset_width)?;
            if let LinkTarget::Hard {
                object_header_address,
            } = link.link_target
            {
                entries.push(GroupEntry {
                    name: link.name,
                    object_header_address,
                });
            }
        }
    }
    Ok(entries)
}

/// The result of looking up one link name in an object.
///
/// The caller supplies the path used to report an absent child or a non-group object.
#[derive(Debug)]
pub(crate) enum ChildLookup {
    /// The group holds no child of that name.
    Absent,
    /// The named link's parsed target.
    Found(ChildTarget),
    /// The object searched is not a group at all, in either form.
    NotAGroup,
}

impl ChildLookup {
    /// Parses a soft target or converts a stored hard link address to a physical position.
    ///
    /// Returns [`ChildLookup::Absent`] if `target` is [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum of the stored address and effective base
    /// exceeds [`u64::MAX`], and [`FormatError::InvalidSoftLinkTarget`] for an invalid soft target.
    fn of(addresses: AddressContext, target: Option<LinkTarget>) -> Result<Self, FormatError> {
        Ok(match target {
            Some(LinkTarget::Hard {
                object_header_address,
            }) => ChildLookup::Found(ChildTarget::Hard(addresses.resolve(object_header_address)?)),
            Some(LinkTarget::Indirect(target)) => match *target {
                SymbolicLinkTarget::Soft { target_path } => {
                    ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse(&target_path)?))
                }
                SymbolicLinkTarget::External {
                    filename,
                    object_path,
                } => ChildLookup::Found(ChildTarget::External {
                    filename,
                    object_path,
                }),
            },
            None => ChildLookup::Absent,
        })
    }
}

/// A link target with a resolved hard address or a parsed soft path.
#[derive(Debug)]
pub(crate) enum ChildTarget {
    /// A file name and object path in another file.
    External {
        filename: String,
        object_path: String,
    },
    /// The physical position of an object header.
    Hard(PhysicalAddress),
    /// A path resolved from its containing group or the root group.
    Soft(SoftLinkPath),
}

/// Returns the parsed target of the child named `name` in the resident image.
///
/// `group_address` is relative to the original image's start. Resolves hard targets using the
/// address context captured in `physical` and borrows dense metadata from its stored view.
///
/// # Errors
///
/// Returns errors from address resolution, group metadata parsing, or link target parsing.
pub(crate) fn find_child_target(
    physical: PhysicalBytes<'_>,
    access_mode: AccessMode,
    group_address: u64,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<ChildLookup, FormatError> {
    let file_data = physical.bytes();
    let addresses = physical.addresses();
    let mut saw_link = false;
    let header = {
        let mut wanted = wanted_link_only(name, &mut saw_link);
        ObjectHeader::parse_filtered_with_widths(
            file_data,
            access_mode,
            group_address.to_usize()?,
            widths,
            addresses.effective_base(),
            MessageFilter::Only(&mut wanted),
        )?
    };
    if !filtered_is_group(&header, saw_link) {
        return Ok(ChildLookup::NotAGroup);
    }
    if holds_compact_links(&header, widths.offsets)? {
        return ChildLookup::of(
            addresses,
            scan_compact_links(&header, widths.offsets, name)?,
        );
    }
    ChildLookup::of(
        addresses,
        find_noncompact_link(
            physical,
            access_mode,
            &header,
            widths,
            group_leaf_node_k,
            name,
        )?,
    )
}

/// Returns the parsed target of the child named `name` through the opened source.
///
/// `group_address` is relative to the original source's start. Resolves hard targets using the
/// source's captured address context and reads metadata through its metadata operation.
///
/// # Errors
///
/// Returns errors from address resolution, source metadata reads, group metadata parsing, or
/// link target parsing.
pub(crate) fn find_child_target_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    group_address: u64,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<ChildLookup, FormatError> {
    let addresses = source.addresses();
    let mut saw_link = false;
    let header = {
        let mut wanted = wanted_link_only(name, &mut saw_link);
        ObjectHeader::parse_from_source_filtered_with_widths(
            &SourceMetadata(source),
            access_mode,
            group_address,
            widths,
            addresses.effective_base(),
            MessageFilter::Only(&mut wanted),
        )?
    };
    if !filtered_is_group(&header, saw_link) {
        return Ok(ChildLookup::NotAGroup);
    }
    if holds_compact_links(&header, widths.offsets)? {
        return ChildLookup::of(
            addresses,
            scan_compact_links(&header, widths.offsets, name)?,
        );
    }
    ChildLookup::of(
        addresses,
        find_noncompact_link_from_source(
            source,
            access_mode,
            &header,
            widths,
            group_leaf_node_k,
            name,
        )?,
    )
}

/// A message filter that keeps everything except the links this lookup did not
/// ask for, recording in `saw_link` that the header held links at all.
///
/// That record is what keeps the filter honest: dropping the other links must
/// not change what the header *is*, and a compact group is recognized partly by
/// having Link messages ([`is_v2_group`]). So the ones dropped here are still
/// counted, and [`filtered_is_group`] classifies the header the way an
/// unfiltered parse would have.
fn wanted_link_only<'a>(
    name: LinkName<'a>,
    saw_link: &'a mut bool,
) -> impl FnMut(MessageType, &[u8]) -> bool + 'a {
    move |ty, body| {
        if ty != MessageType::LINK {
            return true;
        }
        *saw_link = true;
        link_message::link_is_named(body, name.as_str())
    }
}

/// Returns whether the group stores its links as messages in the object header.
///
/// `header` must describe a group accepted by [`filtered_is_group`].
///
/// Returns `false` for a header containing a Symbol Table message, even if a Link Info message
/// is also present, matching [`resolve_group_entries`].
///
/// # Errors
///
/// Returns the [`FormatError`] from parsing the Link Info message.
fn holds_compact_links(
    header: &ObjectHeader,
    offset_width: OffsetWidth,
) -> Result<bool, FormatError> {
    Ok(!is_v1_group(header)
        && find_link_info(header, offset_width)?
            .fractal_heap_address
            .is_none())
}

/// Whether the filtered header a child lookup parsed describes a group at all.
///
/// [`is_group`] alone is not the question here. [`wanted_link_only`] drops the
/// Link messages the lookup did not ask for, so a compact group that keeps its
/// links as Link messages and carries no Link Info message comes back with
/// nothing left to recognize it by whenever the wanted link is not one of them.
/// `saw_link` is the record that it had links, and consulting it is what makes
/// this answer what an unfiltered parse would have: filtering drops only Link
/// messages, so `saw_link || is_group(filtered)` is exactly `is_group` of the
/// header before the filter ran.
fn filtered_is_group(header: &ObjectHeader, saw_link: bool) -> bool {
    saw_link || is_group(header)
}

/// Parses the target of the compact link named `name`.
///
/// Every retained Link message is inspected so that an unreadable prefix is reported
/// regardless of its position. Only matching targets are parsed.
///
/// # Errors
///
/// Returns the errors of [`LinkMessage::target_if_named`].
fn scan_compact_links(
    object_header: &ObjectHeader,
    offset_width: OffsetWidth,
    name: LinkName<'_>,
) -> Result<Option<LinkTarget>, FormatError> {
    let mut found = None;
    for msg in &object_header.messages {
        if msg.msg_type != MessageType::LINK {
            continue;
        }
        let target = LinkMessage::target_if_named(&msg.data, offset_width, name.as_str())?;
        found = found.or(target);
    }
    Ok(found)
}

/// Looks up a link in a version 1 symbol table or resident dense storage.
///
/// Searches a version 1 symbol table when a Symbol Table message is present. For dense storage,
/// searches the name index or scans its records if `name` contains a Unicode replacement
/// character. Resolves symbol-table addresses with the captured context and reads dense metadata
/// through the stored view.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if a dense group has no name index. Returns errors
/// from address resolution, group metadata parsing, heap access, or link target parsing.
fn find_noncompact_link(
    physical: PhysicalBytes<'_>,
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<Option<LinkTarget>, FormatError> {
    if let Some(message) = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
    {
        let table = SymbolTableMessage::parse_with_offset_width(&message.data, widths.offsets)?;
        return group_v1::find_link(physical, &table, widths, group_leaf_node_k, name);
    }
    let info = find_link_info(header, widths.offsets)?;
    let Some(heap_address) = info.fractal_heap_address else {
        return Ok(None);
    };
    let stored = physical.stored()?;
    let file_data = stored.metadata_frame().bytes();
    let heap_source = HeapSource::resident(stored);
    let (heap_header, _) =
        fractal_heap::read_header_and_range_from_source(&heap_source, heap_address, widths)?;
    let index = info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree = BTreeV2Header::parse_with_widths(file_data, index.get().to_usize()?, widths)?;
    let index = BTreeV2LinkNameIndex::new(&btree, widths.offsets, heap_header.heap_id_length)?;
    let layout = FractalHeapIdLayout::new(&heap_header, widths.offsets, widths.lengths);
    let mut heap =
        HeapObjectReader::new(heap_source, &heap_header, heap_address, access_mode, widths);
    if !name.as_str().contains(char::REPLACEMENT_CHARACTER) {
        return find_indexed_dense_link(&stored, &index, widths.offsets, name, |id| {
            heap.read(layout.parse(id)?)
        });
    }
    let records = btree_v2::collect_btree_v2_records(stored, &btree, widths.offsets)?;
    let mut found = None;
    for record in &records {
        let Some(id) = record.link_heap_id(btree.tree_type, heap_header.heap_id_length) else {
            continue;
        };
        let data = heap.read(layout.parse(id)?)?;
        found = found.or(LinkMessage::target_if_named(
            &data,
            widths.offsets,
            name.as_str(),
        )?);
    }
    Ok(found)
}

/// Looks up a link in a version 1 symbol table or dense storage through the opened source.
///
/// Searches the same storage as [`find_noncompact_link`]. Reads metadata through the
/// original source's metadata operation, preserving its ownership, physical ranges, and cache.
///
/// # Errors
///
/// Returns the errors of [`find_noncompact_link`] and source metadata reads.
fn find_noncompact_link_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<Option<LinkTarget>, FormatError> {
    if let Some(message) = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
    {
        let table = SymbolTableMessage::parse_with_offset_width(&message.data, widths.offsets)?;
        return group_v1::find_link_from_source(
            PhysicalSource::new(source),
            &table,
            widths,
            group_leaf_node_k,
            name,
        );
    }
    let info = find_link_info(header, widths.offsets)?;
    let Some(heap_address) = info.fractal_heap_address else {
        return Ok(None);
    };
    let stored = PhysicalSource::new(source).stored();
    let heap_source = HeapSource::new(&stored);
    let (heap_header, _) =
        fractal_heap::read_header_and_range_from_source(&heap_source, heap_address, widths)?;
    let index = info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree = BTreeV2Header::parse_from_source_with_widths(
        &BTreeV2Source::new(&stored),
        index.get(),
        widths,
    )?;
    let index = BTreeV2LinkNameIndex::new(&btree, widths.offsets, heap_header.heap_id_length)?;
    let layout = FractalHeapIdLayout::new(&heap_header, widths.offsets, widths.lengths);
    let mut heap =
        HeapObjectReader::new(heap_source, &heap_header, heap_address, access_mode, widths);
    if !name.as_str().contains(char::REPLACEMENT_CHARACTER) {
        return find_indexed_dense_link(&stored, &index, widths.offsets, name, |id| {
            heap.read(layout.parse(id)?)
        });
    }
    let records = btree_v2::collect_btree_v2_records_from_source(
        &BTreeV2Source::new(&stored),
        &btree,
        widths.offsets,
    )?;
    let mut found = None;
    for record in &records {
        let Some(id) = record.link_heap_id(btree.tree_type, heap_header.heap_id_length) else {
            continue;
        };
        let data = heap.read(layout.parse(id)?)?;
        found = found.or(LinkMessage::target_if_named(
            &data,
            widths.offsets,
            name.as_str(),
        )?);
    }
    Ok(found)
}

/// Parses a link target after matching the name hash and raw stored name bytes.
///
/// Reads index nodes in stored coordinates through [`StoredRead::read_metadata`]. `read_heap`
/// reads link-message bytes for an encoded heap ID. On hash equality, compares the stored name
/// bytes with `name` before decoding the matching target.
///
/// # Errors
///
/// Returns the [`FormatError`] from index traversal, heap access, name parsing, or target parsing.
fn find_indexed_dense_link(
    reader: &(impl StoredRead + ?Sized),
    index: &BTreeV2LinkNameIndex,
    offset_width: OffsetWidth,
    name: LinkName<'_>,
    mut read_heap: impl FnMut(&[u8]) -> Result<Vec<u8>, FormatError>,
) -> Result<Option<LinkTarget>, FormatError> {
    let wanted_hash = checksum::jenkins_lookup3(name.as_str().as_bytes());
    let mut matched = None;
    let record = btree_v2::search_btree_v2(reader, index.nodes(), |record| {
        let (hash, id) = index.record(record)?;
        if hash != wanted_hash {
            return Ok(hash.cmp(&wanted_hash));
        }
        let data = read_heap(id)?;
        let order = LinkMessage::name_bytes(&data)?.cmp(name.as_str().as_bytes());
        if order == Ordering::Equal {
            matched = Some(data);
        }
        Ok(order)
    })?;
    if record.is_none() {
        return Ok(None);
    }
    let data = matched.ok_or_else(|| {
        FormatError::Internal(String::from("a matching B-tree record has no link message"))
    })?;
    LinkMessage::target_if_named(&data, offset_width, name.as_str())
}

/// Returns one entry per hard link in resident dense storage.
///
/// Borrows heap and index metadata from the stored view, retaining its physical origin.
/// Each entry holds the child's stored object-header address.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if the link info has no B-tree name index address, and
/// the [`FormatError`] of the first dense-storage structure that does not parse.
fn resolve_dense_entries(
    source: StoredBytes<'_>,
    access_mode: AccessMode,
    link_info: &LinkInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<GroupEntry>, FormatError> {
    let file_data = source.metadata_frame().bytes();
    let heap_source = HeapSource::resident(source);
    let (fh, _) = fractal_heap::read_header_and_range_from_source(&heap_source, fh_addr, widths)?;

    let btree_addr = link_info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree_hdr =
        BTreeV2Header::parse_with_widths(file_data, btree_addr.get().to_usize()?, widths)?;
    let records = btree_v2::collect_btree_v2_records(source, &btree_hdr, widths.offsets)?;

    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(heap_source, &fh, fh_addr, access_mode, widths);
    let mut entries = Vec::new();
    for record in &records {
        let Some(id_bytes) = record.link_heap_id(btree_hdr.tree_type, fh.heap_id_length) else {
            continue;
        };

        let link_data = heap.read(layout.parse(id_bytes)?)?;

        let link = LinkMessage::parse_with_offset_width(&link_data, widths.offsets)?;
        if let LinkTarget::Hard {
            object_header_address,
        } = link.link_target
        {
            entries.push(GroupEntry {
                name: link.name,
                object_header_address,
            });
        }
    }

    Ok(entries)
}

/// Parses the Link Info message, or returns compact-storage defaults when it is absent.
///
/// # Errors
///
/// Returns errors from [`LinkInfoMessage::parse_with_offset_width`].
fn find_link_info(
    object_header: &ObjectHeader,
    offset_width: OffsetWidth,
) -> Result<LinkInfoMessage, FormatError> {
    for msg in &object_header.messages {
        if msg.msg_type == MessageType::LINK_INFO {
            return LinkInfoMessage::parse_with_offset_width(&msg.data, offset_width);
        }
    }
    Ok(LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    })
}

/// Returns whether the header contains Link Info or Link messages.
fn is_v2_group(object_header: &ObjectHeader) -> bool {
    object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::LINK_INFO || m.msg_type == MessageType::LINK)
}

fn is_v1_group(object_header: &ObjectHeader) -> bool {
    object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::SYMBOL_TABLE)
}

/// Returns whether an object header uses either supported group representation.
///
/// [`resolve_group_entries`] enumerates every header accepted here. By-name group
/// lookups ([`crate::File::group`], [`crate::Group::group`]) use the same predicate before opening
/// a group. Path traversal uses it through [`filtered_is_group`] before descending.
pub(crate) fn is_group(object_header: &ObjectHeader) -> bool {
    is_v1_group(object_header) || is_v2_group(object_header)
}

/// Returns the physical object-header position reached through hard links.
///
/// Resolves `path` from the root using the address context captured in `physical`. Reports
/// soft and external links as missing components.
///
/// # Errors
///
/// Returns the structural and address errors of the group lookup, and
/// [`ResolveError::NotAGroup`] if an intermediate path component is not a group.
pub fn resolve_path_any(
    physical: PhysicalBytes<'_>,
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let addresses = physical.addresses();
    let mut current_addr = addresses.resolve(superblock.root_group_address)?.get();

    for (walked, &name) in path.components().iter().enumerate() {
        match find_child_target(
            physical,
            access_mode,
            current_addr,
            widths,
            group_v1::group_leaf_node_k(superblock),
            name,
        )? {
            ChildLookup::Found(ChildTarget::Hard(abs_addr)) => current_addr = abs_addr.get(),
            ChildLookup::Absent | ChildLookup::Found(_) => {
                return Err(FormatError::PathNotFound(path.prefix(walked + 1).to_string()).into());
            }
            ChildLookup::NotAGroup => {
                return Err(ResolveError::NotAGroup(path.prefix(walked).to_string()));
            }
        }
    }

    Ok(current_addr)
}

/// Returns one entry per hard link in a version 1 or version 2 group.
///
/// Resolves stored addresses with the context captured in `physical`. Reads version 1 storage
/// if the header contains a Symbol Table message, and compact or dense storage otherwise.
/// Each entry holds the child's stored object-header address.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if the header describes no group, and the
/// [`FormatError`] of the first structure of the group that does not parse.
pub fn resolve_group_entries(
    physical: PhysicalBytes<'_>,
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<GroupEntry>, FormatError> {
    if is_v1_group(object_header) {
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
            .ok_or_else(|| FormatError::PathNotFound(String::from("no symbol table message")))?;
        let stm = SymbolTableMessage::parse_with_offset_width(&sym_msg.data, widths.offsets)?;
        group_v1::resolve_v1_group_entries(physical, &stm, widths, group_leaf_node_k)
    } else if is_v2_group(object_header) {
        resolve_v2_group_entries(physical, access_mode, object_header, widths)
    } else {
        Err(FormatError::PathNotFound(String::from(
            "object header is not a group",
        )))
    }
}

/// Returns the physical object-header position reached through hard links in `source`.
///
/// Resolves `path` from the root using the opened backend's address context and reads metadata
/// through its metadata operation. Reports soft and external links as missing components.
///
/// # Errors
///
/// Returns the errors of [`resolve_path_any`], and the error `source` reports if a structure
/// cannot be read.
pub fn resolve_path_any_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let addresses = source.addresses();
    let mut current_addr = addresses.resolve(superblock.root_group_address)?.get();

    for (walked, &name) in path.components().iter().enumerate() {
        match find_child_target_from_source(
            source,
            access_mode,
            current_addr,
            widths,
            group_v1::group_leaf_node_k(superblock),
            name,
        )? {
            ChildLookup::Found(ChildTarget::Hard(abs_addr)) => current_addr = abs_addr.get(),
            ChildLookup::Absent | ChildLookup::Found(_) => {
                return Err(FormatError::PathNotFound(path.prefix(walked + 1).to_string()).into());
            }
            ChildLookup::NotAGroup => {
                return Err(ResolveError::NotAGroup(path.prefix(walked).to_string()));
            }
        }
    }

    Ok(current_addr)
}

/// Returns one entry per hard link through the opened source.
///
/// Selects storage as [`resolve_group_entries`] does. Resolves stored addresses with the source's
/// captured context and reads metadata through its metadata operation. Each entry holds the child's
/// stored object-header address.
///
/// # Errors
///
/// Returns the errors of [`resolve_group_entries`], and the error `source` reports for
/// a structure it cannot read.
pub fn resolve_group_entries_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<GroupEntry>, FormatError> {
    if is_v1_group(object_header) {
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
            .ok_or_else(|| FormatError::PathNotFound(String::from("no symbol table message")))?;
        let stm = SymbolTableMessage::parse_with_offset_width(&sym_msg.data, widths.offsets)?;
        group_v1::resolve_v1_group_entries_from_source(
            PhysicalSource::new(source),
            &stm,
            widths,
            group_leaf_node_k,
        )
    } else if is_v2_group(object_header) {
        resolve_v2_group_entries_from_source(source, access_mode, object_header, widths)
    } else {
        Err(FormatError::PathNotFound(String::from(
            "object header is not a group",
        )))
    }
}

/// Returns one entry per hard link of the version 2 group in `object_header`.
///
/// Reads dense metadata through the original source's metadata operation using its captured
/// effective base. Parses compact links directly from `object_header`.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if a dense group has no B-tree name index address, and
/// the [`FormatError`] of the first structure that does not parse or cannot be read from `source`.
fn resolve_v2_group_entries_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
) -> Result<Vec<GroupEntry>, FormatError> {
    let link_info = find_link_info(object_header, widths.offsets)?;
    if let Some(fh_addr) = link_info.fractal_heap_address {
        let stored = PhysicalSource::new(source).stored();
        resolve_dense_entries_from_source(&stored, access_mode, &link_info, fh_addr, widths)
    } else {
        resolve_compact_entries(object_header, widths.offsets)
    }
}

/// Returns one entry per hard link in dense storage through stored-coordinate metadata reads.
///
/// Reads through [`StoredRead::read_metadata`], preserving the source's ownership, physical ranges,
/// and cache behavior. Each entry holds the child's stored object-header address.
///
/// # Errors
///
/// Returns the errors of [`resolve_dense_entries`], and the error `source` reports for
/// a dense-storage structure it cannot read.
pub(crate) fn resolve_dense_entries_from_source(
    source: &dyn StoredRead,
    access_mode: AccessMode,
    link_info: &LinkInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<GroupEntry>, FormatError> {
    let heap_source = HeapSource::new(source);
    let (fh, _) = fractal_heap::read_header_and_range_from_source(&heap_source, fh_addr, widths)?;

    let btree_addr = link_info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree_hdr = BTreeV2Header::parse_from_source_with_widths(
        &BTreeV2Source::new(source),
        btree_addr.get(),
        widths,
    )?;
    let records = btree_v2::collect_btree_v2_records_from_source(
        &BTreeV2Source::new(source),
        &btree_hdr,
        widths.offsets,
    )?;

    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(heap_source, &fh, fh_addr, access_mode, widths);
    let mut entries = Vec::new();
    for record in &records {
        let Some(id_bytes) = record.link_heap_id(btree_hdr.tree_type, fh.heap_id_length) else {
            continue;
        };
        let link_data = heap.read(layout.parse(id_bytes)?)?;
        let link = LinkMessage::parse_with_offset_width(&link_data, widths.offsets)?;
        if let LinkTarget::Hard {
            object_header_address,
        } = link.link_target
        {
            entries.push(GroupEntry {
                name: link.name,
                object_header_address,
            });
        }
    }

    Ok(entries)
}

#[cfg(test)]
mod tests;
