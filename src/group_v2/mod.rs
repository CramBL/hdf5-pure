//! Group child lookup, hard link listings, and path resolution through hard and soft links.

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
use crate::btree_v2::{self, BTreeV2Header};
use crate::checksum;
use crate::convert::Narrow;
use crate::dense_storage::{self, DenseIndex};
use crate::error::{FormatError, ResolveError};
use crate::fractal_heap::HeapObjectReader;
use crate::group_v1::{self, GroupEntry};
use crate::link_info::LinkInfoMessage;
use crate::link_message::{self, LinkMessage, LinkTarget, SymbolicLinkTarget};
use crate::message_type::MessageType;
use crate::object_header::{MessageFilter, ObjectHeader};
use crate::object_path::{LinkName, ObjectPath, ObjectPathBuf, SoftLinkPath};
use crate::source::{self, BaseOffsetSource, Source};
use crate::source::{RangeReader, SourceMetadata};
use crate::superblock::Superblock;
use crate::symbol_table::SymbolTableMessage;
use crate::width::{FormatWidths, OffsetWidth};

#[cfg(test)]
mod test_util;

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    #[cfg(test)]
    mod native_crosschecks;
}

/// Resolves `path` from the root group, following hard and soft links.
///
/// # Errors
///
/// Returns the errors of [`resolve_link_path`] and of the group metadata parsers.
pub(crate) fn resolve_read_path_any(
    file_data: &[u8],
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    addresses: AddressContext,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let root = addresses.absolute(superblock.root_group_address)?;
    resolve_link_path(root, root, &ObjectPathBuf::root(), path, |address, name| {
        find_child_target(
            file_data,
            access_mode,
            address,
            widths,
            addresses,
            group_v1::group_leaf_node_k(superblock),
            name,
        )
    })
}

/// Resolves `path` from the root group through `source`, following hard and soft links.
///
/// # Errors
///
/// Returns the errors of [`resolve_link_path`], the group metadata parsers, and `source` reads.
pub(crate) fn resolve_read_path_any_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    addresses: AddressContext,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let root = addresses.absolute(superblock.root_group_address)?;
    resolve_link_path(root, root, &ObjectPathBuf::root(), path, |address, name| {
        find_child_target_from_source(
            source,
            access_mode,
            address,
            widths,
            addresses,
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

/// Returns the version 2 B-tree allocations owned by a dense group's link set using validated
/// file widths.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if either index has the wrong client type. Returns
/// the structural error from the version 2 B-tree storage walker if an index cannot be enumerated
/// completely. Returns [`FormatError::InvalidBTreeV2Signature`] if the two indexes overlap or an
/// index allocation contains the fractal-heap header address.
pub(crate) fn collect_dense_group_index_storage_extents<S: Source + ?Sized>(
    source: &S,
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
    dense_storage::collect_dense_index_storage_extents(
        source,
        fractal_heap_address,
        indexes,
        widths,
    )
}

/// Returns one entry per hard link of a version 2 group using validated file widths.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if a dense group has no B-tree name index address, and
/// the [`FormatError`] of the first structure that does not parse.
pub fn resolve_v2_group_entries(
    file_data: &[u8],
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
    addresses: AddressContext,
) -> Result<Vec<GroupEntry>, FormatError> {
    // Look for Link Info message to determine storage type
    let link_info = find_link_info(object_header, widths.offsets)?;

    if let Some(fh_addr) = link_info.fractal_heap_address {
        // Dense storage
        let framed = source::frame(file_data, addresses.effective_base())?;
        resolve_dense_entries(framed, access_mode, addresses, &link_info, fh_addr, widths)
    } else {
        // Compact storage: links are stored directly as Link messages
        resolve_compact_entries(object_header, widths.offsets)
    }
}

/// Extract link entries from Link messages directly in the object header (compact storage).
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
            // Listings include only hard links.
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

/// Returns the child named `name` using validated file widths.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first structure needed for the lookup that does not parse.
pub(crate) fn find_child_target(
    file_data: &[u8],
    access_mode: AccessMode,
    group_address: u64,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<ChildLookup, FormatError> {
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
            file_data,
            access_mode,
            &header,
            widths,
            addresses,
            group_leaf_node_k,
            name,
        )?,
    )
}

/// Returns the child named `name` from `source` using validated file widths.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first structure needed for the lookup that does not parse,
/// or the error `source` reports for a structure it cannot read.
pub(crate) fn find_child_target_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    group_address: u64,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<ChildLookup, FormatError> {
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
            addresses,
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

fn find_noncompact_link(
    file_data: &[u8],
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<Option<LinkTarget>, FormatError> {
    if let Some(message) = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
    {
        let table = SymbolTableMessage::parse_with_offset_width(&message.data, widths.offsets)?;
        return group_v1::find_link(
            file_data,
            &table,
            widths,
            addresses,
            group_leaf_node_k,
            name,
            || {
                crate::btree_v1::collect_symbol_table_nodes(
                    file_data,
                    table.btree_address,
                    widths,
                    addresses,
                )
            },
        );
    }
    let info = find_link_info(header, widths.offsets)?;
    let Some(heap_address) = info.fractal_heap_address else {
        return Ok(None);
    };
    let reader = file_data;
    let file_data = source::frame(file_data, addresses.effective_base())?;
    let heap_header =
        crate::fractal_heap::parse_header(file_data, addresses, heap_address, widths)?;
    let index = info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree = BTreeV2Header::parse_with_widths(file_data, index.get().to_usize()?, widths)?;
    let index = BTreeV2LinkNameIndex::new(&btree, widths.offsets, heap_header.heap_id_length)?;
    let layout = FractalHeapIdLayout::new(&heap_header, widths.offsets, widths.lengths);
    let mut heap =
        HeapObjectReader::new(&heap_header, addresses, heap_address, access_mode, widths);
    if !name.as_str().contains(char::REPLACEMENT_CHARACTER) {
        return find_indexed_dense_link(reader, &index, addresses, widths.offsets, name, |id| {
            heap.read(file_data, layout.parse(id)?)
        });
    }
    let records = btree_v2::collect_btree_v2_records(file_data, &btree, widths.offsets)?;
    let mut found = None;
    for record in &records {
        let Some(id) = record.link_heap_id(btree.tree_type, heap_header.heap_id_length) else {
            continue;
        };
        let data = heap.read(file_data, layout.parse(id)?)?;
        found = found.or(LinkMessage::target_if_named(
            &data,
            widths.offsets,
            name.as_str(),
        )?);
    }
    Ok(found)
}

fn find_noncompact_link_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<Option<LinkTarget>, FormatError> {
    if let Some(message) = header
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
    {
        let table = SymbolTableMessage::parse_with_offset_width(&message.data, widths.offsets)?;
        return group_v1::find_link(
            source,
            &table,
            widths,
            addresses,
            group_leaf_node_k,
            name,
            || {
                crate::btree_v1::collect_symbol_table_nodes_from_source(
                    source,
                    table.btree_address,
                    widths,
                    addresses,
                )
            },
        );
    }
    let info = find_link_info(header, widths.offsets)?;
    let Some(heap_address) = info.fractal_heap_address else {
        return Ok(None);
    };
    let reader = source;
    let source = BaseOffsetSource {
        inner: source,
        base: addresses.effective_base(),
    };
    let heap_header =
        crate::fractal_heap::read_header_from_source(&source, addresses, heap_address, widths)?;
    let index = info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(&source),
        index.get(),
        widths,
    )?;
    let index = BTreeV2LinkNameIndex::new(&btree, widths.offsets, heap_header.heap_id_length)?;
    let layout = FractalHeapIdLayout::new(&heap_header, widths.offsets, widths.lengths);
    let mut heap =
        HeapObjectReader::new(&heap_header, addresses, heap_address, access_mode, widths);
    if !name.as_str().contains(char::REPLACEMENT_CHARACTER) {
        return find_indexed_dense_link(reader, &index, addresses, widths.offsets, name, |id| {
            heap.read_from_source(&source, layout.parse(id)?)
        });
    }
    let records = btree_v2::collect_btree_v2_records_from_source(&source, &btree, widths.offsets)?;
    let mut found = None;
    for record in &records {
        let Some(id) = record.link_heap_id(btree.tree_type, heap_header.heap_id_length) else {
            continue;
        };
        let data = heap.read_from_source(&source, layout.parse(id)?)?;
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
/// `reader` uses physical file coordinates. `read_heap` receives an encoded heap ID and returns
/// its link-message bytes. On hash equality, compares the stored name bytes with `name` before
/// decoding the matching target.
///
/// # Errors
///
/// Returns the [`FormatError`] from index traversal, heap access, name parsing, or target parsing.
fn find_indexed_dense_link(
    reader: &(impl RangeReader + ?Sized),
    index: &BTreeV2LinkNameIndex,
    addresses: AddressContext,
    offset_width: OffsetWidth,
    name: LinkName<'_>,
    mut read_heap: impl FnMut(&[u8]) -> Result<Vec<u8>, FormatError>,
) -> Result<Option<LinkTarget>, FormatError> {
    let wanted_hash = checksum::jenkins_lookup3(name.as_str().as_bytes());
    let mut matched = None;
    let record = btree_v2::search_btree_v2(reader, index.nodes(), addresses, |record| {
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

/// Returns dense group entries using validated file widths.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if the link info has no B-tree name index address, and
/// the [`FormatError`] of the first dense-storage structure that does not parse.
fn resolve_dense_entries(
    file_data: &[u8],
    access_mode: AccessMode,
    addresses: AddressContext,
    link_info: &LinkInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<GroupEntry>, FormatError> {
    // Parse fractal heap
    let fh = crate::fractal_heap::parse_header(file_data, addresses, fh_addr, widths)?;

    // Parse B-tree v2 for name index
    let btree_addr = link_info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree_hdr =
        BTreeV2Header::parse_with_widths(file_data, btree_addr.get().to_usize()?, widths)?;
    let records = btree_v2::collect_btree_v2_records(file_data, &btree_hdr, widths.offsets)?;

    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(&fh, addresses, fh_addr, access_mode, widths);
    let mut entries = Vec::new();
    for record in &records {
        let Some(id_bytes) = record.link_heap_id(btree_hdr.tree_type, fh.heap_id_length) else {
            continue;
        };

        // Read the link message from the fractal heap (managed or huge object).
        let link_data = heap.read(file_data, layout.parse(id_bytes)?)?;

        // Parse as Link message
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

/// Find and parse the Link Info message from an object header.
fn find_link_info(
    object_header: &ObjectHeader,
    offset_width: OffsetWidth,
) -> Result<LinkInfoMessage, FormatError> {
    for msg in &object_header.messages {
        if msg.msg_type == MessageType::LINK_INFO {
            return LinkInfoMessage::parse_with_offset_width(&msg.data, offset_width);
        }
    }
    // No Link Info message — might have direct Link messages
    // Return a "compact" link info with no fractal heap
    Ok(LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    })
}

/// Detect whether an object header represents a v1 group, v2 group, or neither.
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

/// Returns the address of the object header `path` identifies through hard links.
///
/// Soft and external links are reported as missing components.
///
/// # Errors
///
/// Returns the structural and address errors of the group lookup, and
/// [`ResolveError::NotAGroup`] if an intermediate path component is not a group.
pub fn resolve_path_any(
    file_data: &[u8],
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    addresses: AddressContext,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let mut current_addr = addresses.absolute(superblock.root_group_address)?;

    for (walked, &name) in path.components().iter().enumerate() {
        match find_child_target(
            file_data,
            access_mode,
            current_addr,
            widths,
            addresses,
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

/// Returns the entries of a group using validated file widths.
///
/// A version 2 group's soft and external links, and a version 1 group's symbolic links, are passed
/// over: they identify no object header in this file.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if the header describes no group, and the
/// [`FormatError`] of the first structure of the group that does not parse.
pub fn resolve_group_entries(
    file_data: &[u8],
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<GroupEntry>, FormatError> {
    if is_v1_group(object_header) {
        // v1: find SymbolTableMessage and use existing v1 code
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
            .ok_or_else(|| FormatError::PathNotFound(String::from("no symbol table message")))?;
        let stm = SymbolTableMessage::parse_with_offset_width(&sym_msg.data, widths.offsets)?;
        group_v1::resolve_v1_group_entries(file_data, &stm, widths, addresses, group_leaf_node_k)
    } else if is_v2_group(object_header) {
        resolve_v2_group_entries(file_data, access_mode, object_header, widths, addresses)
    } else {
        // Names no object, and cannot: an enumeration is handed a header and
        // nothing else. Where there is a path to give, the walks give it: they
        // classify through `find_child_target` before they descend and reject a
        // non-group by name (issue #365). What is left reaches this holding no
        // path either — `reference_patch` and the incoming-hard-link tally read
        // it as "not a group, skip", and `FileInner::group_children` and
        // `reconstruct_v1_group` surface it — so there is nothing better for it
        // to say.
        Err(FormatError::PathNotFound(String::from(
            "object header is not a group",
        )))
    }
}

// ---------------------------------------------------------------------------
// Streaming path resolution (latest-format / v2 groups), reading each metadata
// structure from a `Source` on demand.
// ---------------------------------------------------------------------------

/// Returns the address of the object header `path` identifies from `source` using validated widths.
///
/// # Errors
///
/// Returns the errors of [`resolve_path_any`], and the error `source` reports if a structure
/// cannot be read.
pub fn resolve_path_any_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    superblock: &Superblock,
    widths: FormatWidths,
    addresses: AddressContext,
    path: &ObjectPath<'_>,
) -> Result<u64, ResolveError> {
    let mut current_addr = addresses.absolute(superblock.root_group_address)?;

    for (walked, &name) in path.components().iter().enumerate() {
        match find_child_target_from_source(
            source,
            access_mode,
            current_addr,
            widths,
            addresses,
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

/// Returns the entries of a group from `source` using validated file widths.
///
/// # Errors
///
/// Returns the errors of [`resolve_group_entries`], and the error `source` reports for
/// a structure it cannot read.
pub fn resolve_group_entries_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
    addresses: AddressContext,
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
            source,
            &stm,
            widths,
            addresses,
            group_leaf_node_k,
        )
    } else if is_v2_group(object_header) {
        resolve_v2_group_entries_from_source(source, access_mode, object_header, widths, addresses)
    } else {
        Err(FormatError::PathNotFound(String::from(
            "object header is not a group",
        )))
    }
}

/// Returns one entry per hard link of the version 2 group in `object_header`.
///
/// Reads dense-group metadata from `source` on demand. `addresses` supplies the effective file
/// base used to interpret stored addresses.
///
/// # Errors
///
/// Returns [`FormatError::PathNotFound`] if a dense group has no B-tree name index address, and
/// the [`FormatError`] of the first structure that does not parse or cannot be read from `source`.
fn resolve_v2_group_entries_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    object_header: &ObjectHeader,
    widths: FormatWidths,
    addresses: AddressContext,
) -> Result<Vec<GroupEntry>, FormatError> {
    let link_info = find_link_info(object_header, widths.offsets)?;
    if let Some(fh_addr) = link_info.fractal_heap_address {
        let framed = BaseOffsetSource {
            inner: source,
            base: addresses.effective_base(),
        };
        resolve_dense_entries_from_source(
            &framed,
            access_mode,
            addresses,
            &link_info,
            fh_addr,
            widths,
        )
    } else {
        // Compact storage: links live in the (already-parsed) object header.
        resolve_compact_entries(object_header, widths.offsets)
    }
}

/// Returns dense group entries from `source` using validated file widths.
///
/// # Errors
///
/// Returns the errors of [`resolve_dense_entries`], and the error `source` reports for
/// a dense-storage structure it cannot read.
pub(crate) fn resolve_dense_entries_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    addresses: AddressContext,
    link_info: &LinkInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<GroupEntry>, FormatError> {
    let fh = crate::fractal_heap::read_header_from_source(source, addresses, fh_addr, widths)?;

    let btree_addr = link_info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree_hdr = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(source),
        btree_addr.get(),
        widths,
    )?;
    let records =
        btree_v2::collect_btree_v2_records_from_source(source, &btree_hdr, widths.offsets)?;

    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(&fh, addresses, fh_addr, access_mode, widths);
    let mut entries = Vec::new();
    for record in &records {
        let Some(id_bytes) = record.link_heap_id(btree_hdr.tree_type, fh.heap_id_length) else {
            continue;
        };
        let link_data = heap.read_from_source(source, layout.parse(id_bytes)?)?;
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
