//! V2 group traversal: resolve group children and navigate paths.
//!
//! Handles both compact storage (Link messages in object header) and
//! dense storage (fractal heap + B-tree v2).

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::address::BaseAddress;
use crate::btree_v2::{
    BTreeV2Header, collect_btree_v2_records, collect_btree_v2_records_from_source,
};
use crate::convert::TryToUsize;
use crate::error::{FormatError, ResolveError};
use crate::fractal_heap::FractalHeapHeader;
use crate::group_v1::{self, GroupEntry};
use crate::link_info::LinkInfoMessage;
use crate::link_message::{LinkMessage, LinkTarget, link_is_named};
use crate::message_type::MessageType;
use crate::object_header::{MessageFilter, ObjectHeader};
use crate::source::{BaseOffsetSource, Source, frame};
use crate::superblock::Superblock;
use crate::symbol_table::SymbolTableMessage;

/// Resolve v2 group entries from an object header.
///
/// Handles both compact (Link messages) and dense (fractal heap + B-tree v2) storage.
///
/// `base_address` is the superblock base address. The fractal heap and B-tree
/// addresses in a Link Info message are file addresses, so on a file with a
/// userblock they are short of their real positions by the base; dense storage is
/// read through a base-framed view of `file_data` so they index it directly. The
/// entry addresses this returns stay relative, as the compact path's do.
pub fn resolve_v2_group_entries(
    file_data: &[u8],
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<GroupEntry>, FormatError> {
    // Look for Link Info message to determine storage type
    let link_info = find_link_info(object_header, offset_size)?;

    if let Some(fh_addr) = link_info.fractal_heap_address {
        // Dense storage
        let framed = frame(file_data, base_address)?;
        resolve_dense_entries(framed, &link_info, fh_addr, offset_size, length_size)
    } else {
        // Compact storage: links are stored directly as Link messages
        resolve_compact_entries(object_header, offset_size)
    }
}

/// Extract link entries from Link messages directly in the object header (compact storage).
fn resolve_compact_entries(
    object_header: &ObjectHeader,
    offset_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    let mut entries = Vec::new();
    for msg in &object_header.messages {
        if msg.msg_type == MessageType::Link {
            let link = LinkMessage::parse(&msg.data, offset_size)?;
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
            // Skip soft and external links for path resolution
        }
    }
    Ok(entries)
}

/// What a by-name child lookup found in the object it was pointed at.
///
/// Three answers rather than an `Option`, because "this is not a group, so it
/// has no children to search" is a different fact from "this group has no child
/// of that name". Both used to leave the lookup as a `FormatError::PathNotFound`
/// — absence naming the component, a non-group naming nothing at all — so a
/// caller could not tell them apart, and a path *through* a dataset read as a
/// path that does not exist (issue #365).
///
/// Which object the answer is about is left to the caller. Naming it takes the
/// path walked to reach it, and only the caller has that; a lookup one component
/// deep knows an address and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChildLookup {
    /// The group holds a child of that name, at this *absolute* file address.
    ///
    /// Absolute rather than stored: every consumer wants it that way, and
    /// [`of`](Self::of) — the one thing in the crate that builds this variant —
    /// already takes the base address the search itself needed. Back when each
    /// consumer converted for itself, one of the three did so with an unchecked
    /// `+`.
    Found(u64),
    /// The group holds no child of that name.
    Absent,
    /// The object searched is not a group at all, in either form.
    NotAGroup,
}

impl ChildLookup {
    /// The answer reached by searching an object that *is* a group, given the
    /// stored (base-relative) address the search found.
    ///
    /// [`FormatError::OffsetOverflow`] if the base cannot be added to it, which
    /// a crafted link naming `HADDR_UNDEF` in a file with a userblock arranges.
    fn of(base: BaseAddress, address: Option<u64>) -> Result<Self, FormatError> {
        Ok(match address {
            Some(address) => ChildLookup::Found(base.absolute(address)?),
            None => ChildLookup::Absent,
        })
    }
}

/// The child named `name` in the group whose header is at `group_address`, found
/// without reading every other child.
///
/// A group of *n* children is *n* Link messages in its object header, and
/// [`resolve_group_entries`] turns each into a [`GroupEntry`] with an owned name.
/// Resolving one path needs one of them, so a walk that opens each child of a
/// group in turn paid for *n* names *n* times over — 310 KiB and two thousand
/// allocations per lookup in a 1,024-child group, which is what made that walk
/// quadratic in the group's size (issue #228).
///
/// So the header is parsed asking only for the link that was named. The filter
/// drops nothing else, and only a *compact* v2 group stores its links as Link
/// messages — a dense group keeps them in a fractal heap and a v1 group in a
/// symbol table — so the fallback below reads a header that is whole as far as
/// it is concerned.
///
/// The object searched is classified before it is searched, so that one that is
/// no group at all answers [`ChildLookup::NotAGroup`] rather than an error the
/// caller cannot tell from an absent child; see that type.
///
/// The address answered is absolute; see [`ChildLookup::of`].
pub(crate) fn find_child_address(
    file_data: &[u8],
    group_address: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    name: &str,
) -> Result<ChildLookup, FormatError> {
    let mut saw_link = false;
    let header = {
        let mut wanted = wanted_link_only(name, &mut saw_link);
        ObjectHeader::parse_filtered(
            file_data,
            group_address.to_usize()?,
            offset_size,
            length_size,
            base_address,
            MessageFilter::Only(&mut wanted),
        )?
    };
    if !filtered_is_group(&header, saw_link) {
        return Ok(ChildLookup::NotAGroup);
    }
    if holds_compact_links(&header, offset_size)? {
        return ChildLookup::of(
            base_address,
            scan_compact_links(&header, offset_size, name)?,
        );
    }
    ChildLookup::of(
        base_address,
        resolve_group_entries(file_data, &header, offset_size, length_size, base_address)?
            .into_iter()
            .find(|e| e.name == name)
            .map(|e| e.object_header_address),
    )
}

/// Streaming counterpart of [`find_child_address`].
pub(crate) fn find_child_address_from_source<S: Source + ?Sized>(
    source: &S,
    group_address: u64,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
    name: &str,
) -> Result<ChildLookup, FormatError> {
    let mut saw_link = false;
    let header = {
        let mut wanted = wanted_link_only(name, &mut saw_link);
        ObjectHeader::parse_from_source_filtered(
            source,
            group_address,
            offset_size,
            length_size,
            base_address,
            MessageFilter::Only(&mut wanted),
        )?
    };
    if !filtered_is_group(&header, saw_link) {
        return Ok(ChildLookup::NotAGroup);
    }
    if holds_compact_links(&header, offset_size)? {
        return ChildLookup::of(
            base_address,
            scan_compact_links(&header, offset_size, name)?,
        );
    }
    ChildLookup::of(
        base_address,
        resolve_group_entries_from_source(source, &header, offset_size, length_size, base_address)?
            .into_iter()
            .find(|e| e.name == name)
            .map(|e| e.object_header_address),
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
    name: &'a str,
    saw_link: &'a mut bool,
) -> impl FnMut(MessageType, &[u8]) -> bool + 'a {
    move |ty, body| {
        if ty != MessageType::Link {
            return true;
        }
        *saw_link = true;
        link_is_named(body, name)
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
fn holds_compact_links(header: &ObjectHeader, offset_size: u8) -> Result<bool, FormatError> {
    Ok(!is_v1_group(header)
        && find_link_info(header, offset_size)?
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

/// The stored address of the hard link named `name` among this header's compact
/// Link messages.
///
/// Soft and external links are skipped, as [`resolve_compact_entries`] skips
/// them: they name no object header in this file.
///
/// # What a lookup still refuses
///
/// Every Link message here is read, rather than stopping at the match, and the
/// reason is that the ones this cannot read at all are refused: a lookup must
/// not depend on whether the damaged link sits before or after the one asked
/// for. That costs nothing, because the parse that produced this header was
/// asked for the wanted link and kept only it and the messages it could not
/// judge (see [`wanted_link_only`]).
///
/// What a lookup no longer refuses is a link whose *target* is malformed and
/// whose name is not the one asked for — resolving one path used to parse every
/// link's target, so any such link failed every lookup in the group. Listing the
/// group ([`resolve_group_entries`]) still reports it; a lookup reads the link it
/// was asked for.
fn scan_compact_links(
    object_header: &ObjectHeader,
    offset_size: u8,
    name: &str,
) -> Result<Option<u64>, FormatError> {
    let mut found = None;
    for msg in &object_header.messages {
        if msg.msg_type != MessageType::Link {
            continue;
        }
        let addr = LinkMessage::hard_link_address_if_named(&msg.data, offset_size, name)?;
        found = found.or(addr);
    }
    Ok(found)
}

/// Resolve entries from dense storage (fractal heap + B-tree v2).
fn resolve_dense_entries(
    file_data: &[u8],
    link_info: &LinkInfoMessage,
    fh_addr: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    // Parse fractal heap
    let fh = FractalHeapHeader::parse(file_data, fh_addr.to_usize()?, offset_size, length_size)?;

    // Parse B-tree v2 for name index
    let btree_addr = link_info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree_hdr =
        BTreeV2Header::parse(file_data, btree_addr.to_usize()?, offset_size, length_size)?;
    let records = collect_btree_v2_records(file_data, &btree_hdr, offset_size, length_size)?;

    let mut heap = fh.object_reader(offset_size, length_size);
    let mut entries = Vec::new();
    for record in &records {
        // For type 5 (name index): hash(4) + heap_id(heap_id_length)
        // For type 6 (creation order): creation_order(8) + heap_id(heap_id_length)
        let id_offset = if btree_hdr.tree_type == 5 {
            4 // skip hash
        } else {
            8 // skip creation_order
        };

        if record.data.len() < id_offset + fh.heap_id_length as usize {
            continue;
        }
        let id_bytes = &record.data[id_offset..id_offset + fh.heap_id_length as usize];

        // Read the link message from the fractal heap (managed or huge object).
        let link_data = heap.read(file_data, id_bytes)?;

        // Parse as Link message
        let link = LinkMessage::parse(&link_data, offset_size)?;
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
    offset_size: u8,
) -> Result<LinkInfoMessage, FormatError> {
    for msg in &object_header.messages {
        if msg.msg_type == MessageType::LinkInfo {
            return LinkInfoMessage::parse(&msg.data, offset_size);
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
        .any(|m| m.msg_type == MessageType::LinkInfo || m.msg_type == MessageType::Link)
}

fn is_v1_group(object_header: &ObjectHeader) -> bool {
    object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::SymbolTable)
}

/// Whether an object header describes a group at all, in either form.
///
/// The union of the two predicates above, which is to say: exactly the headers
/// [`resolve_group_entries`] will enumerate rather than refuse. A by-name group
/// lookup ([`crate::File::group`], [`crate::Group::group`]) asks the same
/// question to decide whether to open at all, and a path walk asks it through
/// [`filtered_is_group`] to decide whether it can descend at all. Each asks it
/// here, so a header this module would enumerate cannot be one they turn away.
pub(crate) fn is_group(object_header: &ObjectHeader) -> bool {
    is_v1_group(object_header) || is_v2_group(object_header)
}

/// The root-relative path of the object a path walk had descended into when it
/// tried to look `components[i]` up inside it: every component before `i`.
///
/// The whole prefix rather than the one component, because that is the path the
/// refusal is *about* — `a/b/c` stopped by a dataset at `a/b` names `a/b`, which
/// a caller can go and open, where a bare `b` would not say where to find it.
/// Empty at `i == 0`, the root group, which is how this crate names the root
/// throughout (a root [`crate::Group`] handle carries the empty path too).
///
/// Both walks resolve the same path the same way, so they name the object the
/// same way as well.
fn walked_prefix(components: &[&str], i: usize) -> String {
    components[..i].join("/")
}

/// Unified path resolution that works for both v1 and v2 groups.
///
/// Detects group version from object header messages and dispatches accordingly.
pub fn resolve_path_any(
    file_data: &[u8],
    superblock: &Superblock,
    path: &str,
) -> Result<u64, ResolveError> {
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if components.is_empty() {
        return Ok(superblock.root_group_address);
    }

    let os = superblock.offset_size;
    let ls = superblock.length_size;
    let base = superblock.base_address;

    let mut current_addr = superblock.root_group_address;

    for (i, component) in components.iter().enumerate() {
        match find_child_address(file_data, current_addr, os, ls, base, component)? {
            ChildLookup::Found(abs_addr) => {
                if i == components.len() - 1 {
                    return Ok(abs_addr);
                }
                current_addr = abs_addr;
            }
            ChildLookup::Absent => {
                return Err(FormatError::PathNotFound(String::from(*component)).into());
            }
            ChildLookup::NotAGroup => {
                return Err(ResolveError::NotAGroup(walked_prefix(&components, i)));
            }
        }
    }

    Ok(current_addr)
}

/// Resolve group entries from an object header, auto-detecting v1 vs v2.
///
/// `base_address` is the superblock base address, used to convert relative
/// addresses to absolute file offsets in v1 groups.
pub fn resolve_group_entries(
    file_data: &[u8],
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<GroupEntry>, FormatError> {
    if is_v1_group(object_header) {
        // v1: find SymbolTableMessage and use existing v1 code
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SymbolTable)
            .ok_or_else(|| FormatError::PathNotFound(String::from("no symbol table message")))?;
        let stm = SymbolTableMessage::parse(&sym_msg.data, offset_size)?;
        group_v1::resolve_v1_group_entries(file_data, &stm, offset_size, length_size, base_address)
    } else if is_v2_group(object_header) {
        resolve_v2_group_entries(
            file_data,
            object_header,
            offset_size,
            length_size,
            base_address,
        )
    } else {
        // Names no object, and cannot: an enumeration is handed a header and
        // nothing else. Where there is a path to give, the walks give it: they
        // classify through `find_child_address` before they descend and refuse a
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

/// Streaming counterpart of [`resolve_path_any`].
///
/// Resolves a path to an object-header address by reading the object headers
/// and (for dense groups) the fractal heap + B-tree v2 from a [`Source`].
/// Both group forms resolve: v2 (compact or dense) groups, and v1 symbol-table
/// groups via [`group_v1::resolve_v1_group_entries_from_source`].
pub fn resolve_path_any_from_source<S: Source + ?Sized>(
    source: &S,
    superblock: &Superblock,
    path: &str,
) -> Result<u64, ResolveError> {
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if components.is_empty() {
        return Ok(superblock.root_group_address);
    }

    let os = superblock.offset_size;
    let ls = superblock.length_size;
    let base = superblock.base_address;

    let mut current_addr = superblock.root_group_address;

    for (i, component) in components.iter().enumerate() {
        match find_child_address_from_source(source, current_addr, os, ls, base, component)? {
            ChildLookup::Found(abs_addr) => {
                if i == components.len() - 1 {
                    return Ok(abs_addr);
                }
                current_addr = abs_addr;
            }
            ChildLookup::Absent => {
                return Err(FormatError::PathNotFound(String::from(*component)).into());
            }
            ChildLookup::NotAGroup => {
                return Err(ResolveError::NotAGroup(walked_prefix(&components, i)));
            }
        }
    }

    Ok(current_addr)
}

/// Streaming counterpart of [`resolve_group_entries`], auto-detecting v1 vs v2.
///
/// `base_address` is the superblock base address, used to convert the relative
/// addresses stored in v1 (symbol-table) groups to absolute file offsets.
pub fn resolve_group_entries_from_source<S: Source + ?Sized>(
    source: &S,
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<GroupEntry>, FormatError> {
    if is_v1_group(object_header) {
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SymbolTable)
            .ok_or_else(|| FormatError::PathNotFound(String::from("no symbol table message")))?;
        let stm = SymbolTableMessage::parse(&sym_msg.data, offset_size)?;
        group_v1::resolve_v1_group_entries_from_source(
            source,
            &stm,
            offset_size,
            length_size,
            base_address,
        )
    } else if is_v2_group(object_header) {
        resolve_v2_group_entries_from_source(
            source,
            object_header,
            offset_size,
            length_size,
            base_address,
        )
    } else {
        Err(FormatError::PathNotFound(String::from(
            "object header is not a group",
        )))
    }
}

fn resolve_v2_group_entries_from_source<S: Source + ?Sized>(
    source: &S,
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<GroupEntry>, FormatError> {
    let link_info = find_link_info(object_header, offset_size)?;
    if let Some(fh_addr) = link_info.fractal_heap_address {
        let framed = BaseOffsetSource {
            inner: source,
            base: base_address,
        };
        resolve_dense_entries_from_source(&framed, &link_info, fh_addr, offset_size, length_size)
    } else {
        // Compact storage: links live in the (already-parsed) object header.
        resolve_compact_entries(object_header, offset_size)
    }
}

fn resolve_dense_entries_from_source<S: Source + ?Sized>(
    source: &S,
    link_info: &LinkInfoMessage,
    fh_addr: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    let fh = FractalHeapHeader::parse_from_source(source, fh_addr, offset_size, length_size)?;

    let btree_addr = link_info
        .btree_name_index_address
        .ok_or_else(|| FormatError::PathNotFound(String::from("no B-tree v2 name index")))?;
    let btree_hdr = BTreeV2Header::parse_from_source(source, btree_addr, offset_size, length_size)?;
    let records =
        collect_btree_v2_records_from_source(source, &btree_hdr, offset_size, length_size)?;

    let mut heap = fh.object_reader(offset_size, length_size);
    let mut entries = Vec::new();
    for record in &records {
        let id_offset = if btree_hdr.tree_type == 5 { 4 } else { 8 };
        if record.data.len() < id_offset + fh.heap_id_length as usize {
            continue;
        }
        let id_bytes = &record.data[id_offset..id_offset + fh.heap_id_length as usize];
        let link_data = heap.read_from_source(source, id_bytes)?;
        let link = LinkMessage::parse(&link_data, offset_size)?;
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
    use super::*;
    use crate::data_layout::DataLayout;
    use crate::data_read;
    use crate::dataspace::Dataspace;
    use crate::datatype::Datatype;
    use crate::signature;

    fn extract_dataset(
        _file_data: &[u8],
        hdr: &ObjectHeader,
        offset_size: u8,
        length_size: u8,
    ) -> (Datatype, Dataspace, DataLayout) {
        let dt_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .unwrap()
            .data;
        let ds_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .unwrap()
            .data;
        let dl_data = &hdr
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .unwrap()
            .data;
        let (dt, _) = Datatype::parse(dt_data).unwrap();
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
                    msg_type: MessageType::LinkInfo,
                    size: 18,
                    flags: 0,
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
                    msg_type: MessageType::Link,
                    size: link_data.len(),
                    flags: 0,
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

        let entries = resolve_v2_group_entries(&[], &oh, 8, 8, BaseAddress::ZERO).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test");
        assert_eq!(entries[0].object_header_address, 0x1000);
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
                    flags: 0,
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
        (MessageType::LinkInfo, d)
    }

    /// A Link Info message naming a fractal heap: dense storage.
    fn dense_link_info_message() -> (MessageType, Vec<u8>) {
        let mut d = vec![0, 0];
        d.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap address
        d.extend_from_slice(&0x2000u64.to_le_bytes()); // name index address
        (MessageType::LinkInfo, d)
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
            holds_compact_links(&no_link_info, 8).unwrap(),
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
            !holds_compact_links(&dense, 8).unwrap(),
            "a fractal heap decides, not the presence of a link message"
        );
    }

    /// Only hard links name an object header in this file, and only the link
    /// asked for answers.
    #[test]
    fn scanning_compact_links_answers_for_hard_links_alone() {
        let mut hard = vec![1u8, 0x00, 4];
        hard.extend_from_slice(b"data");
        hard.extend_from_slice(&0x4000u64.to_le_bytes());

        let mut soft = vec![1u8, 0x08, 1, 4];
        soft.extend_from_slice(b"soft");
        soft.extend_from_slice(&4u16.to_le_bytes());
        soft.extend_from_slice(b"/abc");

        let header = header_of(vec![
            compact_link_info(),
            (MessageType::Link, soft),
            (MessageType::Link, hard),
        ]);

        assert_eq!(
            scan_compact_links(&header, 8, "data").unwrap(),
            Some(0x4000)
        );
        assert_eq!(
            scan_compact_links(&header, 8, "soft").unwrap(),
            None,
            "a soft link names a path, not an object header"
        );
        assert_eq!(scan_compact_links(&header, 8, "absent").unwrap(), None);
    }

    #[test]
    fn integration_v2_groups_temperature() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();
        assert!(sb.version >= 2); // v2/v3 superblock

        let addr = resolve_path_any(file_data, &sb, "sensors/temperature").unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn integration_v2_groups_humidity() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "sensors/humidity").unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_i32(&raw, &dt).unwrap();
        assert_eq!(values, vec![45, 50, 55]);
    }

    #[test]
    fn integration_v2_many_links() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_many_links.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "dataset_015").unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![15.0]);
    }

    #[test]
    fn integration_resolve_path_any_v1() {
        // Test that resolve_path_any also works for v1 files
        let file_data: &[u8] = include_bytes!("../tests/fixtures/two_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "group1/values").unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_i32(&raw, &dt).unwrap();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn integration_resolve_path_any_v2() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let addr = resolve_path_any(file_data, &sb, "sensors/temperature").unwrap();
        let hdr =
            ObjectHeader::parse(file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![22.5, 23.1, 21.8]);
    }

    #[test]
    fn path_not_found_v2() {
        let file_data: &[u8] = include_bytes!("../tests/fixtures/v2_groups.h5");
        let sig_offset = signature::find_signature(file_data).unwrap();
        let sb = Superblock::parse(file_data, sig_offset).unwrap();

        let err = resolve_path_any(file_data, &sb, "nonexistent").unwrap_err();
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
            BaseAddress::ZERO,
        )
        .expect("the root header is readable")[0]
            .span;
        let end = (chunk_at + chunk_len).to_usize().unwrap() - 4;
        let repaired = crate::checksum::jenkins_lookup3(&bytes[chunk_at.to_usize().unwrap()..end]);
        bytes[end..end + 4].copy_from_slice(&repaired.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        // Each backend reaches `ChildLookup::of` through its own lookup
        // (`find_child_address` and `find_child_address_from_source`), so both
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
/// dense-attribute ones. Gated to 64-bit targets with the reference C library,
/// which is the only writer that produces a huge *link*: this crate's writer
/// stores even a 60,000-byte link name as a managed heap object, so the huge
/// path these tests cover is unreachable from a file it wrote.
#[cfg(all(
    test,
    feature = "__hdf5-1.10",
    not(target_pointer_width = "32"),
    target_endian = "little"
))]
mod huge_link_tests {
    use super::*;
    use crate::fractal_heap::{huge_index_decodes, reset_huge_index_decodes};
    use crate::signature;
    use crate::source::BytesSource;

    /// A file with one group of `count` links, each name long enough that its
    /// link message exceeds the heap's managed-object limit and is stored as a
    /// huge object.
    fn file_with_huge_links(count: usize) -> Vec<u8> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge_links.h5");
        {
            let file = hdf5::FileBuilder::new()
                .with_fapl(|fapl| fapl.libver_latest())
                .create(&path)
                .unwrap();
            let group = file.create_group("g").unwrap();
            for i in 0..count {
                let name = format!("d{i}_{}", "x".repeat(5000));
                group
                    .new_dataset::<i32>()
                    .shape((1,))
                    .create(name.as_str())
                    .unwrap()
                    .write(&[i as i32])
                    .unwrap();
            }
            file.close().unwrap();
        }
        std::fs::read(&path).unwrap()
    }

    /// Group `g`'s dense-link storage: its info message, its heap address, and
    /// the file's offset and length sizes.
    fn dense_link_info(bytes: &[u8]) -> (LinkInfoMessage, u64, u8, u8) {
        let sig = signature::find_signature(bytes).unwrap();
        let superblock = Superblock::parse(bytes, sig).unwrap();
        let (offset_size, length_size) = (superblock.offset_size, superblock.length_size);
        let group_addr = resolve_path_any(bytes, &superblock, "g").unwrap();
        let header = ObjectHeader::parse(
            bytes,
            group_addr.to_usize().unwrap(),
            offset_size,
            length_size,
        )
        .unwrap();
        let link_info = find_link_info(&header, offset_size).unwrap();
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

        reset_huge_index_decodes();
        let buffered =
            resolve_dense_entries(&bytes, &link_info, fh_addr, offset_size, length_size).unwrap();
        assert_eq!(buffered.len(), COUNT, "the walk must still read every link");
        assert_eq!(
            huge_index_decodes(),
            1,
            "the buffered walk parsed the huge-object index per link rather than per walk"
        );

        reset_huge_index_decodes();
        let source = BytesSource::new(bytes);
        let streamed = resolve_dense_entries_from_source(
            &source,
            &link_info,
            fh_addr,
            offset_size,
            length_size,
        )
        .unwrap();
        assert_eq!(streamed.len(), COUNT);
        assert_eq!(
            huge_index_decodes(),
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
