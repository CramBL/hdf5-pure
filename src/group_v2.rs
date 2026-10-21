//! Group child lookup, hard link listings, and path resolution through hard and soft links.

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::num::NonZeroU16;

use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::{BTREE_V2_LINK_CREATION_ORDER, BTREE_V2_LINK_NAME};
use hdf5_pure_space::__private::Extent;

use crate::access_mode::AccessMode;
#[cfg(test)]
use crate::address::BaseAddress;
#[cfg(test)]
use crate::address::BaseAddressExt;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, StoredAddress};
use crate::btree_v2::{self, BTreeV2Header};
use crate::convert::Narrow;
use crate::dense_storage::{self, DenseIndex};
use crate::error::{FormatError, ResolveError};
use crate::fractal_heap::HeapObjectReader;
use crate::group_v1::{self, GroupEntry};
use crate::link_info::LinkInfoMessage;
use crate::link_message::{self, LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::object_header::{MessageFilter, ObjectHeader};
use crate::object_path::{LinkName, ObjectPath, ObjectPathBuf, SoftLinkPath};
use crate::source::SourceMetadata;
use crate::source::{self, BaseOffsetSource, Source};
use crate::superblock::Superblock;
use crate::symbol_table::SymbolTableMessage;
use crate::width::{FormatWidths, OffsetWidth};

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
                    cache_type: 0,
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
            Some(LinkTarget::Soft { target_path }) => {
                ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse(&target_path)?))
            }
            Some(LinkTarget::External {
                filename,
                object_path,
            }) => ChildLookup::Found(ChildTarget::External {
                filename,
                object_path,
            }),
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

/// Whether `header` is a v2 group holding its links compactly, as Link messages
/// — the case [`scan_compact_links`] can answer from the header alone, now that
/// [`wanted_link_only`] has narrowed it to the link asked for.
///
/// **Only ask this of a header [`filtered_is_group`] has accepted.** A header that
/// is no group at all has no fractal heap either, so it would answer `true` here
/// and be scanned for links it cannot have; both callers classify first, which is
/// what lets this ask the two questions that remain. A v1 (symbol-table) group
/// and a dense v2 group both answer `false` and fall back to the full walk —
/// neither stores links as Link messages, so that walk reads a header the filter
/// took nothing from.
///
/// The `!is_v1_group` term is what gives the symbol table precedence, matching
/// [`resolve_group_entries`], which checks it first: a header carrying both a
/// Symbol Table message and a Link Info message is not something any writer here
/// produces, but the two paths must agree on what it is, or a child that the
/// fallback finds through the symbol table would be reported as absent by the
/// scan.
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
    let file_data = source::frame(file_data, addresses.effective_base())?;
    let heap_header =
        crate::fractal_heap::parse_header(file_data, addresses, heap_address, widths)?;
    let index = info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree = BTreeV2Header::parse_with_widths(file_data, index.get().to_usize()?, widths)?;
    let records = btree_v2::collect_btree_v2_records(file_data, &btree, widths.offsets)?;
    let layout = FractalHeapIdLayout::new(&heap_header, widths.offsets, widths.lengths);
    let mut heap =
        HeapObjectReader::new(&heap_header, addresses, heap_address, access_mode, widths);
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
    let records = btree_v2::collect_btree_v2_records_from_source(&source, &btree, widths.offsets)?;
    let layout = FractalHeapIdLayout::new(&heap_header, widths.offsets, widths.lengths);
    let mut heap =
        HeapObjectReader::new(&heap_header, addresses, heap_address, access_mode, widths);
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
                cache_type: 0,
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
                cache_type: 0,
            });
        }
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use test_util::btree_v2;
    use test_util::widths::Widths;

    use super::*;
    use crate::data_layout::DataLayout;
    use crate::data_read;
    use crate::dataspace::Dataspace;
    use crate::datatype::Datatype;
    use crate::message_flags::MessageFlags;
    #[cfg(feature = "std")]
    use crate::reader::File;
    use crate::source::BytesSource;
    #[cfg(feature = "std")]
    use crate::writer::FileBuilder;

    #[rstest::rstest]
    #[case::relative("alias/value", 3)]
    #[case::absolute("absolute/value", 4)]
    #[case::chain("chain/value", 3)]
    #[case::group("alias", 2)]
    #[case::current_group_target("alias/again/value", 3)]
    fn soft_link_targets_use_relative_or_absolute_starting_groups(
        #[case] path: &str,
        #[case] expected: u64,
    ) {
        let result = resolve_link_path(
            0,
            1,
            &ObjectPathBuf::parse("g"),
            &ObjectPath::parse(path),
            |address, name| {
                Ok(match (address, name.as_str()) {
                    (1, "alias") => ChildLookup::Found(ChildTarget::Soft(
                        SoftLinkPath::parse("target").unwrap(),
                    )),
                    (1, "absolute") => ChildLookup::Found(ChildTarget::Soft(
                        SoftLinkPath::parse("/target").unwrap(),
                    )),
                    (1, "chain") => {
                        ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse("alias").unwrap()))
                    }
                    (1, "target") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(2))),
                    (0, "target") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(5))),
                    (2, "value") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(3))),
                    (5, "value") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(4))),
                    (2, "again") => {
                        ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse(".").unwrap()))
                    }
                    _ => ChildLookup::Absent,
                })
            },
        )
        .unwrap();
        assert_eq!(result, expected);
    }

    #[rstest::rstest]
    #[case::self_link("loop")]
    #[case::expanding_target("loop/suffix")]
    fn soft_link_cycles_stop_after_a_bounded_number_of_lookups(#[case] target: &str) {
        let mut reads = 0;
        let err = resolve_link_path(
            0,
            0,
            &ObjectPathBuf::root(),
            &ObjectPath::parse("loop"),
            |_, _| {
                reads += 1;
                Ok(ChildLookup::Found(ChildTarget::Soft(
                    SoftLinkPath::parse(target).unwrap(),
                )))
            },
        )
        .unwrap_err();
        let ResolveError::Format(error) = err else {
            panic!("expected a format error, got {err:?}");
        };
        assert_eq!(
            error,
            FormatError::SoftLinkTraversalLimit {
                path: String::from("loop"),
                limit: 64
            }
        );
        assert_eq!(reads, 65);
    }

    #[test]
    fn structure_errors_preserve_their_format_variant() {
        let expected = FormatError::PathNotFound(String::from("no B-tree v2 name index"));
        let error = resolve_link_path(
            0,
            0,
            &ObjectPathBuf::root(),
            &ObjectPath::parse("alias"),
            |_, name| {
                if name.as_str() == "alias" {
                    Ok(ChildLookup::Found(ChildTarget::Soft(
                        SoftLinkPath::parse("target").unwrap(),
                    )))
                } else {
                    Err(expected.clone())
                }
            },
        )
        .unwrap_err();
        let ResolveError::Format(error) = error else {
            panic!("expected format error, got {error:?}");
        };
        assert_eq!(error, expected);
    }

    fn test_widths(superblock: &Superblock) -> FormatWidths {
        FormatWidths::from_sizes(superblock.offset_size, superblock.length_size).unwrap()
    }

    #[cfg(feature = "std")]
    #[rstest::rstest]
    #[case::buffered(false)]
    #[case::source(true)]
    fn path_navigation_uses_the_opened_image_context(#[case] source_backed: bool) {
        let mut builder = FileBuilder::new();
        builder.with_userblock(512);
        let mut group = builder.create_group("group");
        group.create_dataset("data").with_i32_data(&[1, 2, 3]);
        builder.add_group(group.finish());
        let bytes = builder.finish().unwrap();
        let file = File::from_bytes(bytes.clone()).unwrap();
        let mut superblock = file.superblock().clone();
        let addresses = AddressContext::from_effective_base(superblock.base_address);
        superblock.base_address = BaseAddress::new(2048);
        let path = ObjectPath::parse("group/data");

        let object_at = if source_backed {
            resolve_path_any_from_source(
                &BytesSource::new(&bytes[..]),
                AccessMode::ReadOnly,
                &superblock,
                test_widths(&superblock),
                addresses,
                &path,
            )
        } else {
            resolve_path_any(
                &bytes,
                AccessMode::ReadOnly,
                &superblock,
                test_widths(&superblock),
                addresses,
                &path,
            )
        }
        .unwrap();

        let header = ObjectHeader::parse_with_base(
            &bytes,
            AccessMode::ReadOnly,
            object_at.to_usize().unwrap(),
            superblock.offset_size,
            superblock.length_size,
            addresses.effective_base(),
        )
        .unwrap();
        let (datatype, dataspace, layout) = extract_dataset(
            &bytes,
            &header,
            superblock.offset_size,
            superblock.length_size,
        );
        let raw = data_read::read_raw_data(
            source::frame(&bytes, addresses.effective_base()).unwrap(),
            &layout,
            &dataspace,
            &datatype,
        )
        .unwrap();
        assert_eq!(data_read::read_as_i32(&raw, &datatype).unwrap(), [1, 2, 3]);
    }

    const WIDTHS: Widths = Widths::EIGHT;
    const NODE_SIZE: u64 = 512;
    const RECORD_SIZE: u16 = 16;

    fn place_link_name_index(file: &mut [u8], header_at: usize, root_at: usize) -> Vec<Extent> {
        let record = vec![0; usize::from(RECORD_SIZE)];
        let leaf = btree_v2::leaf(BTREE_V2_LINK_NAME, &[record]);
        let header = btree_v2::Header::new(
            BTREE_V2_LINK_NAME,
            RECORD_SIZE,
            u64::try_from(root_at).unwrap(),
            1,
        )
        .build(WIDTHS);
        file[header_at..header_at + header.len()].copy_from_slice(&header);
        file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);
        vec![
            Extent::new(
                u64::try_from(header_at).unwrap(),
                u64::try_from(header.len()).unwrap(),
            )
            .unwrap(),
            Extent::new(u64::try_from(root_at).unwrap(), NODE_SIZE).unwrap(),
        ]
    }

    #[test]
    fn dense_group_storage_walk_does_not_infer_an_index_from_tracking() {
        let mut file = vec![0; 0x800];
        let expected = place_link_name_index(&mut file, 0x100, 0x200);
        let info = LinkInfoMessage {
            max_creation_order: Some(2),
            fractal_heap_address: Some(StoredAddress::new(0x80)),
            btree_name_index_address: Some(StoredAddress::new(0x100)),
            btree_creation_order_address: None,
        };

        let extents = collect_dense_group_index_storage_extents(
            &BytesSource::new(file),
            &info,
            crate::file_writer::WIDTHS,
        )
        .unwrap();

        assert_eq!(extents, expected);
    }

    #[test]
    fn dense_group_storage_walk_omits_an_undefined_indexed_creation_order_tree() {
        let mut file = vec![0; 0x800];
        let expected = place_link_name_index(&mut file, 0x100, 0x200);
        let info = LinkInfoMessage {
            max_creation_order: Some(2),
            fractal_heap_address: Some(StoredAddress::new(0x80)),
            btree_name_index_address: Some(StoredAddress::new(0x100)),
            btree_creation_order_address: Some(None),
        };

        let extents = collect_dense_group_index_storage_extents(
            &BytesSource::new(file),
            &info,
            crate::file_writer::WIDTHS,
        )
        .unwrap();

        assert_eq!(extents, expected);
    }

    fn extract_dataset(
        _file_data: &[u8],
        hdr: &ObjectHeader,
        offset_size: u8,
        length_size: u8,
    ) -> (Datatype, Dataspace, DataLayout) {
        let dt_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATATYPE)
            .unwrap()
            .data;
        let ds_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATASPACE)
            .unwrap()
            .data;
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
            .unwrap()
            .data;
        let (dt, _) = hdf5_pure_format::__private::parse_datatype(dt_data).unwrap();
        let ds = Dataspace::parse(ds_data, length_size).unwrap();
        let dl = DataLayout::parse(dl_data, offset_size, length_size).unwrap();
        (dt, ds, dl)
    }

    #[test]
    fn compact_storage_link_messages() {
        // Build a v2 object header with Link messages (compact storage)
        // We'll test with the actual v2_groups.h5 file since building synthetic v2 headers
        // with proper checksums is complex.

        // Instead, test the resolve_compact_entries path with a simple object header
        let link_data = {
            // Build a Link message: hard link, name="test", addr=0x1000
            let mut d = Vec::new();
            d.push(1); // version
            d.push(0x00); // flags: no creation order, no link type (=hard), no charset, name_size=1byte
            d.push(4); // name length = 4
            d.extend_from_slice(b"test");
            d.extend_from_slice(&0x1000u64.to_le_bytes()); // address
            d
        };

        let oh = ObjectHeader {
            version: 2,
            messages: vec![
                crate::object_header::HeaderMessage {
                    msg_type: MessageType::LINK_INFO,
                    size: 18,
                    flags: MessageFlags::NONE,
                    creation_order: None,
                    data: {
                        let mut d = Vec::new();
                        d.push(0); // version
                        d.push(0); // flags
                        d.extend_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes()); // fh undef
                        d.extend_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes()); // btree undef
                        d
                    },
                },
                crate::object_header::HeaderMessage {
                    msg_type: MessageType::LINK,
                    size: link_data.len(),
                    flags: MessageFlags::NONE,
                    creation_order: None,
                    data: link_data,
                },
            ],
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        };

        let entries = resolve_v2_group_entries(
            &[],
            AccessMode::ReadOnly,
            &oh,
            FormatWidths::from_sizes(8, 8).unwrap(),
            AddressContext::from_effective_base(BaseAddress::new(0)),
        )
        .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test");
        assert_eq!(entries[0].object_header_address, StoredAddress::new(0x1000));
    }

    /// Build an object header carrying the given messages, as a lookup's filtered
    /// parse would leave it.
    fn header_of(messages: Vec<(MessageType, Vec<u8>)>) -> ObjectHeader {
        ObjectHeader {
            version: 2,
            messages: messages
                .into_iter()
                .map(|(msg_type, data)| crate::object_header::HeaderMessage {
                    msg_type,
                    size: data.len(),
                    flags: MessageFlags::NONE,
                    creation_order: None,
                    data,
                })
                .collect(),
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        }
    }

    /// A Link Info message naming no fractal heap: compact storage.
    fn compact_link_info() -> (MessageType, Vec<u8>) {
        let mut d = vec![0, 0]; // version, flags
        d.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap: undefined
        d.extend_from_slice(&u64::MAX.to_le_bytes()); // name index: undefined
        (MessageType::LINK_INFO, d)
    }

    /// A Link Info message naming a fractal heap: dense storage.
    fn dense_link_info_message() -> (MessageType, Vec<u8>) {
        let mut d = vec![0, 0];
        d.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap address
        d.extend_from_slice(&0x2000u64.to_le_bytes()); // name index address
        (MessageType::LINK_INFO, d)
    }

    /// A lookup asks the parse for one link, so by the time the header is
    /// classified its other links are gone. The classification therefore takes the
    /// parse's word for having seen them — otherwise a group that stores links as
    /// messages *without* a Link Info message (which [`find_link_info`] accepts on
    /// purpose) would stop looking like a group the moment its links were filtered
    /// out, and a missing child would be reported as "not a group" instead. Since
    /// issue #365 that report is the refusal a lookup gives, so this is what keeps
    /// such a group from being refused outright rather than merely searched twice.
    #[test]
    fn a_filtered_out_link_still_makes_the_header_a_compact_group() {
        let no_link_info = header_of(vec![]);
        assert!(
            filtered_is_group(&no_link_info, true),
            "a header whose links the filter dropped is still a group"
        );
        assert!(
            holds_compact_links(&no_link_info, OffsetWidth::Eight).unwrap(),
            "and one holding its links compactly, so the scan answers it"
        );
        assert!(
            !filtered_is_group(&no_link_info, false),
            "a header that held no link and no link info is no group of ours"
        );
    }

    /// Dense storage keeps its links in a fractal heap, so its header holds none
    /// to scan whatever the filter saw.
    #[test]
    fn a_dense_group_is_never_scanned_for_compact_links() {
        let dense = header_of(vec![dense_link_info_message()]);
        assert!(
            filtered_is_group(&dense, true),
            "a dense group is a group, so a lookup goes on to ask the question below"
        );
        assert!(
            !holds_compact_links(&dense, OffsetWidth::Eight).unwrap(),
            "a fractal heap decides, not the presence of a link message"
        );
    }

    #[test]
    fn scanning_compact_links_distinguishes_hard_soft_and_absent_names() {
        let mut hard = vec![1u8, 0x00, 4];
        hard.extend_from_slice(b"data");
        hard.extend_from_slice(&0x4000u64.to_le_bytes());

        let mut soft = vec![1u8, 0x08, 1, 4];
        soft.extend_from_slice(b"soft");
        soft.extend_from_slice(&4u16.to_le_bytes());
        soft.extend_from_slice(b"/abc");

        let header = header_of(vec![
            compact_link_info(),
            (MessageType::LINK, soft),
            (MessageType::LINK, hard),
        ]);

        assert_eq!(
            scan_compact_links(&header, OffsetWidth::Eight, LinkName::new("data").unwrap(),)
                .unwrap(),
            Some(LinkTarget::Hard {
                object_header_address: StoredAddress::new(0x4000)
            })
        );
        assert_eq!(
            scan_compact_links(&header, OffsetWidth::Eight, LinkName::new("soft").unwrap(),)
                .unwrap(),
            Some(LinkTarget::Soft {
                target_path: String::from("/abc")
            })
        );
        assert_eq!(
            scan_compact_links(
                &header,
                OffsetWidth::Eight,
                LinkName::new("absent").unwrap(),
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn integration_v2_groups_temperature() {
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/v2_groups.h5");
        let sb = crate::superblock::read(file_data).unwrap().superblock;
        assert!(sb.version >= 2); // v2/v3 superblock

        let addr = resolve_path_any(
            file_data,
            AccessMode::ReadOnly,
            &sb,
            test_widths(&sb),
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("sensors/temperature"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse(
            file_data,
            AccessMode::ReadOnly,
            addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn integration_v2_groups_humidity() {
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/v2_groups.h5");
        let sb = crate::superblock::read(file_data).unwrap().superblock;

        let addr = resolve_path_any(
            file_data,
            AccessMode::ReadOnly,
            &sb,
            test_widths(&sb),
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("sensors/humidity"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse(
            file_data,
            AccessMode::ReadOnly,
            addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_i32(&raw, &dt).unwrap();
        assert_eq!(values, vec![45, 50, 55]);
    }

    #[test]
    fn integration_v2_many_links() {
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/v2_many_links.h5");
        let sb = crate::superblock::read(file_data).unwrap().superblock;

        let addr = resolve_path_any(
            file_data,
            AccessMode::ReadOnly,
            &sb,
            test_widths(&sb),
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("dataset_015"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse(
            file_data,
            AccessMode::ReadOnly,
            addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![15.0]);
    }

    #[test]
    fn integration_resolve_path_any_v1() {
        // Test that resolve_path_any also works for v1 files
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/two_groups.h5");
        let sb = crate::superblock::read(file_data).unwrap().superblock;

        let addr = resolve_path_any(
            file_data,
            AccessMode::ReadOnly,
            &sb,
            test_widths(&sb),
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("group1/values"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse(
            file_data,
            AccessMode::ReadOnly,
            addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_i32(&raw, &dt).unwrap();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn integration_resolve_path_any_v2() {
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/v2_groups.h5");
        let sb = crate::superblock::read(file_data).unwrap().superblock;

        let addr = resolve_path_any(
            file_data,
            AccessMode::ReadOnly,
            &sb,
            test_widths(&sb),
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("sensors/temperature"),
        )
        .unwrap();
        let hdr = ObjectHeader::parse(
            file_data,
            AccessMode::ReadOnly,
            addr as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn path_not_found_v2() {
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/v2_groups.h5");
        let sb = crate::superblock::read(file_data).unwrap().superblock;

        let err = resolve_path_any(
            file_data,
            AccessMode::ReadOnly,
            &sb,
            test_widths(&sb),
            AddressContext::from_effective_base(sb.base_address),
            &ObjectPath::parse("nonexistent"),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ResolveError::Format(FormatError::PathNotFound(_))
        ));
    }

    /// A link target that overflows when the superblock base address is added to
    /// it is reported, not wrapped.
    ///
    /// `resolve_path_any` and its streaming twin each did this addition with a
    /// bare `+`, so a crafted link naming `HADDR_UNDEF` in a file that has a
    /// userblock panicked in a debug build and wrapped to a near-zero address in
    /// a release one. `FileInner::group_child` — a third consumer of the same
    /// stored address — got it right only because someone wrote the check there
    /// by hand. All three now share [`ChildLookup::of`], so this covers the one
    /// place the addition happens.
    ///
    /// The file needs a userblock for this to be reachable at all: with a base of
    /// zero the addition is the identity and `HADDR_UNDEF` survives it to be
    /// refused further down by `ensure_len` as an out-of-range read, which is why
    /// every base-0 fixture in the suite passed either way.
    #[cfg(feature = "std")]
    #[test]
    fn a_link_target_that_overflows_the_base_address_is_refused() {
        const UB: u64 = 512;

        let mut b = crate::writer::FileBuilder::new();
        b.with_userblock(UB);
        b.create_dataset("d")
            .with_i32_data(&[1, 2, 3])
            .with_shape(&[3]);
        let mut bytes = b.finish().unwrap();

        // Find the root group's object header and, inside it, the Link message
        // naming "d" — by pattern rather than by a recorded offset, so a layout
        // change moves the test rather than quietly emptying it.
        let root_oh = bytes
            .windows(4)
            .position(|w| w == b"OHDR")
            .expect("a latest-format file has a version 2 root object header");
        let link = bytes
            .windows(2)
            .position(|w| w == [1, b'd'])
            .expect("the root header holds a one-character link name");
        let target = link + 2;
        let stored = u64::from_le_bytes(bytes[target..target + 8].try_into().unwrap());
        assert_eq!(
            stored + UB,
            bytes
                .windows(4)
                .enumerate()
                .filter(|(_, w)| *w == b"OHDR")
                .nth(1)
                .expect("the dataset has an object header too")
                .0 as u64,
            "the bytes after the link name must be the dataset's stored address"
        );

        // The fixture resolves before it is corrupted, through both backends:
        // what each assertion below observes is the corruption, not a file that
        // never worked.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ub.h5");
        std::fs::write(&path, &bytes).unwrap();
        for read in [
            crate::reader::File::from_bytes(bytes.clone()).unwrap(),
            crate::reader::File::open_streaming(&path).unwrap(),
        ] {
            assert_eq!(read.dataset("d").unwrap().read::<i32>().unwrap(), [1, 2, 3]);
        }

        // Point the link at HADDR_UNDEF and repair the header checksum, which
        // would otherwise refuse the file before the addition is ever reached.
        bytes[target..target + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let (chunk_at, chunk_len) = crate::edit::read_oh_chunks(
            &crate::source::BytesSource::new(&bytes[..]),
            root_oh as u64,
            AddressContext::from_effective_base(BaseAddress::new(0)),
        )
        .expect("the root header is readable")[0]
            .span;
        let end = (chunk_at + chunk_len).to_usize().unwrap() - 4;
        let repaired = crate::checksum::jenkins_lookup3(&bytes[chunk_at.to_usize().unwrap()..end]);
        bytes[end..end + 4].copy_from_slice(&repaired.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        // Each backend reaches `ChildLookup::of` through its own lookup
        // (`find_child_target` and `find_child_target_from_source`), so both
        // are asserted.
        for (backend, file) in [
            ("buffered", crate::reader::File::from_bytes(bytes).unwrap()),
            (
                "streaming",
                crate::reader::File::open_streaming(&path).unwrap(),
            ),
        ] {
            let err = file.dataset("d").unwrap_err();
            assert!(
                matches!(
                    err,
                    crate::Error::Format(FormatError::OffsetOverflow {
                        offset: u64::MAX,
                        length: UB,
                    })
                ),
                "{backend}: expected the base-address addition to be refused, got {err:?}"
            );
        }
    }
}

/// The dense-link walks hold to the same one-parse-per-walk invariant as the
/// dense-attribute ones. The file comes from libhdf5, the only
/// writer that produces a huge *link*: this crate's writer stores even a
/// 60,000-byte link name as a managed heap object, so the huge path these tests
/// cover is unreachable from a file it wrote.
#[cfg(test)]
mod huge_link_tests {
    use super::*;
    use crate::fractal_heap;
    use crate::source::BytesSource;

    fn test_widths(superblock: &Superblock) -> FormatWidths {
        FormatWidths::from_sizes(superblock.offset_size, superblock.length_size).unwrap()
    }

    /// A file with one group of `count` links, each name long enough that its
    /// link message exceeds the heap's managed-object limit and is stored as a
    /// huge object. Written by `crates/crosscheck/tests/main/c_test_data.rs`.
    fn file_with_huge_links(count: usize) -> Vec<u8> {
        std::fs::read(test_util::data(&format!("c/huge_links_{count}.h5"))).unwrap()
    }

    /// Group `g`'s dense-link storage: its info message, its heap address, and
    /// the file's offset and length sizes.
    fn dense_link_info(bytes: &[u8]) -> (LinkInfoMessage, StoredAddress, u8, u8) {
        let superblock = crate::superblock::read(bytes).unwrap().superblock;
        let (offset_size, length_size) = (superblock.offset_size, superblock.length_size);
        let group_addr = resolve_path_any(
            bytes,
            AccessMode::ReadOnly,
            &superblock,
            test_widths(&superblock),
            AddressContext::from_effective_base(superblock.base_address),
            &ObjectPath::parse("g"),
        )
        .unwrap();
        let header = ObjectHeader::parse(
            bytes,
            AccessMode::ReadOnly,
            group_addr.to_usize().unwrap(),
            offset_size,
            length_size,
        )
        .unwrap();
        let link_info =
            find_link_info(&header, OffsetWidth::try_from(offset_size).unwrap()).unwrap();
        let fh_addr = link_info
            .fractal_heap_address
            .expect("this many long-named links are stored densely");
        (link_info, fh_addr, offset_size, length_size)
    }

    /// Both dense-link walks resolve every huge object against one parse of the
    /// heap's huge-object index, not one parse per link.
    ///
    /// Costs, not answers, are what regress here: parsing the index per object
    /// returns exactly the same links while making the walk quadratic in their
    /// number, so the count of parses is the only thing that catches it.
    #[test]
    fn a_dense_link_walk_parses_its_huge_object_index_once() {
        // Above the C library's max_compact of 8, so the links are stored densely.
        const COUNT: usize = 12;
        let bytes = file_with_huge_links(COUNT);
        let (link_info, fh_addr, offset_size, length_size) = dense_link_info(&bytes);

        fractal_heap::reset_huge_index_decodes();
        let buffered = resolve_dense_entries(
            &bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(BaseAddress::new(0)),
            &link_info,
            fh_addr,
            FormatWidths::from_sizes(offset_size, length_size).unwrap(),
        )
        .unwrap();
        assert_eq!(buffered.len(), COUNT, "the walk must still read every link");
        assert_eq!(
            fractal_heap::huge_index_decodes(),
            1,
            "the buffered walk parsed the huge-object index per link rather than per walk"
        );

        fractal_heap::reset_huge_index_decodes();
        let source = BytesSource::new(bytes);
        let streamed = resolve_dense_entries_from_source(
            &source,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(BaseAddress::new(0)),
            &link_info,
            fh_addr,
            FormatWidths::from_sizes(offset_size, length_size).unwrap(),
        )
        .unwrap();
        assert_eq!(streamed.len(), COUNT);
        assert_eq!(
            fractal_heap::huge_index_decodes(),
            1,
            "the streaming walk parsed the huge-object index per link rather than per walk"
        );

        let names: Vec<&str> = buffered.iter().map(|e| e.name.as_str()).collect();
        for (i, entry) in streamed.iter().enumerate() {
            assert!(
                names.contains(&entry.name.as_str()),
                "link {i} differs between the two backends"
            );
        }
    }
}
