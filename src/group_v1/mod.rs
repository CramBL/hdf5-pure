//! Link lookup and hard-link listings for version 1 symbol-table groups.
//!
//! Readers resolve stored addresses using the context captured in the physical source or
//! resident image. Local heaps and symbol table nodes are parsed in physical coordinates.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
use core::num::NonZeroU16;

use hdf5_pure_format::__private::LinkStorageError;
use hdf5_pure_format::__private::LocalHeap;
use hdf5_pure_format::__private::LocalHeapDataSegment;
use hdf5_pure_format::__private::LocalHeapLocator;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use hdf5_pure_format::__private::SymbolTableEntry;
use hdf5_pure_format::__private::SymbolTableNode;
use hdf5_pure_format::__private::SymbolTableNodeLocator;

#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, StoredAddress};
#[cfg(test)]
use crate::address::{BaseAddress, BaseAddressExt};
use crate::btree_v1;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_message::LinkTarget;
use crate::object_path::LinkName;
use crate::source;
use crate::source::PhysicalBytes;
use crate::source::PhysicalSource;
use crate::source::RangeReader;
use crate::source::Source;
use crate::superblock::Superblock;
use crate::symbol_table::SymbolTableMessage;
use crate::width::FormatWidths;
use crate::width::OffsetWidth;

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    #[cfg(test)]
    mod native_crosschecks;
}

/// A resolved group entry (child name + object header address).
#[derive(Debug, Clone)]
pub struct GroupEntry {
    /// Name of the child object.
    pub name: String,
    /// Address of the child's object header.
    pub object_header_address: StoredAddress,
}

/// Returns one entry per child using validated file widths.
///
/// Reads the group through [`visit_v1_group_links`] using the context captured in `physical`.
/// A symbolic link ([`SYMBOLIC_LINK`]) identifies no object header in this file and is passed
/// over once its value parses, as a version 2 group's soft links are. `group_leaf_node_k` is the
/// value [`group_leaf_node_k`] returns for the file's superblock.
///
/// # Errors
///
/// Returns [`FormatError::InvalidSymbolTableCacheType`] if an entry's cache type is not 0, 1, or
/// 2. Returns errors from address resolution, group metadata parsing, and B-tree traversal.
pub fn resolve_v1_group_entries(
    physical: PhysicalBytes<'_>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<GroupEntry>, FormatError> {
    let mut entries = Vec::new();
    visit_v1_group_links(physical, sym_table_msg, widths, group_leaf_node_k, |link| {
        entries.extend(link.into_group_entry()?);
        Ok(())
    })
    .map(|()| entries)
}

/// Returns one entry per child from `source` using validated file widths.
///
/// Reads the group through [`visit_v1_group_links_from_source`] using the source's captured
/// context. Each entry holds the child's stored object-header address.
///
/// # Errors
///
/// Returns the errors of [`resolve_v1_group_entries`], and the error `source` reports
/// for a structure it cannot read.
pub fn resolve_v1_group_entries_from_source(
    source: PhysicalSource<'_, impl Source + ?Sized>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<GroupEntry>, FormatError> {
    let mut entries = Vec::new();
    visit_v1_group_links_from_source(source, sym_table_msg, widths, group_leaf_node_k, |link| {
        entries.extend(link.into_group_entry()?);
        Ok(())
    })
    .map(|()| entries)
}

/// Returns one link per entry of the version 1 group `sym_table_msg` describes, hard or soft.
///
/// Reads the group through [`visit_v1_group_links`] using the context captured in `physical`.
/// Each link has the ASCII character set and no creation index. `group_leaf_node_k` is the value
/// [`group_leaf_node_k`] returns for the file's superblock.
///
/// # Errors
///
/// Returns [`LinkStorageError::Format`] for address-resolution, group-parsing, or B-tree traversal
/// errors, [`LinkStorageError::UndefinedAddress`] for a hard link to the undefined address, and
/// the errors of [`OwnedLink::new`] for a name or a target it rejects.
pub(crate) fn resolve_v1_group_links(
    physical: PhysicalBytes<'_>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<OwnedLink>, LinkStorageError> {
    let mut links = Vec::new();
    visit_v1_group_links(physical, sym_table_msg, widths, group_leaf_node_k, |link| {
        links.push(link.into_owned_link(widths.offsets)?);
        Ok(())
    })
    .map(|()| links)
}

/// Returns one link per entry of the version 1 group `sym_table_msg` describes, read from
/// `source`.
///
/// Reads through [`visit_v1_group_links_from_source`] using the source's captured context.
/// Each link has the ASCII character set and no creation index.
///
/// # Errors
///
/// Returns the errors of [`resolve_v1_group_links`], and the error `source` reports for a
/// structure it cannot read.
pub(crate) fn resolve_v1_group_links_from_source(
    source: PhysicalSource<'_, impl Source + ?Sized>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<OwnedLink>, LinkStorageError> {
    let mut links = Vec::new();
    visit_v1_group_links_from_source(source, sym_table_msg, widths, group_leaf_node_k, |link| {
        links.push(link.into_owned_link(widths.offsets)?);
        Ok(())
    })
    .map(|()| links)
}

/// Returns the target of the version 1 group link named `name`, or [`None`] if absent.
///
/// Reads the group using the context captured in `physical`. Parses every entry's name and
/// cache type, and parses a symbolic target only for the first matching link.
///
/// # Errors
///
/// Returns the errors of [`visit_v1_group_links`] and, for a matching symbolic link, of
/// [`SoftValue::path`].
pub(crate) fn find_link(
    physical: PhysicalBytes<'_>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<Option<LinkTarget>, FormatError> {
    let mut found = None;
    visit_v1_group_links(physical, sym_table_msg, widths, group_leaf_node_k, |link| {
        if found.is_none() && link.name == name.as_str() {
            found = Some(link.target.to_link_target()?);
        }
        Ok(())
    })
    .map(|()| found)
}

/// Returns the target of the version 1 group link named `name` through `source`.
///
/// Returns [`None`] if the link is absent. Reads the group using the source's captured context.
/// Parses every entry's name and cache type, and parses a symbolic target only for the first
/// matching link.
///
/// # Errors
///
/// Returns the errors of [`visit_v1_group_links_from_source`] and, for a matching symbolic link,
/// of [`SoftValue::path`].
pub(crate) fn find_link_from_source(
    source: PhysicalSource<'_, impl Source + ?Sized>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
) -> Result<Option<LinkTarget>, FormatError> {
    let mut found = None;
    visit_v1_group_links_from_source(source, sym_table_msg, widths, group_leaf_node_k, |link| {
        if found.is_none() && link.name == name.as_str() {
            found = Some(link.target.to_link_target()?);
        }
        Ok(())
    })
    .map(|()| found)
}

/// Returns the "Group Leaf Node K" field of `superblock`, or `None` where the field is absent or 0.
///
/// A version 2 or 3 superblock does not have the field.
pub(crate) fn group_leaf_node_k(superblock: &Superblock) -> Option<NonZeroU16> {
    superblock.group_leaf_node_k.and_then(NonZeroU16::new)
}

/// Calls `visit` with each link in a resident version 1 group.
///
/// Resolves B-tree, local-heap, and symbol-table addresses with the context captured in
/// `physical`. Parses B-tree nodes in the complete original image, and the local heap and symbol
/// table nodes in physical ranges.
///
/// # Errors
///
/// Returns the errors of [`visit_links`] and of [`btree_v1::collect_symbol_table_nodes`].
fn visit_v1_group_links<E: From<FormatError>>(
    physical: PhysicalBytes<'_>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    visit: impl FnMut(LinkRef<'_>) -> Result<(), E>,
) -> Result<(), E> {
    visit_links(
        physical.bytes(),
        sym_table_msg,
        widths,
        physical.addresses(),
        group_leaf_node_k,
        || btree_v1::collect_symbol_table_nodes(physical, sym_table_msg.btree_address, widths),
        visit,
    )
}

/// Calls `visit` with each link of the version 1 group `sym_table_msg` describes, read from
/// `source` through the group's B-tree.
///
/// Reads B-tree nodes through the stored view from [`PhysicalSource::stored`] and parses the
/// local heap and symbol table nodes in physical coordinates. Uses the source's captured context
/// for both. Metadata reads preserve the original source's ownership, physical ranges, and cache
/// behavior.
///
/// # Errors
///
/// Returns the errors of [`visit_links`] and of
/// [`btree_v1::collect_symbol_table_nodes_from_source`].
fn visit_v1_group_links_from_source<E: From<FormatError>>(
    source: PhysicalSource<'_, impl Source + ?Sized>,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    group_leaf_node_k: Option<NonZeroU16>,
    visit: impl FnMut(LinkRef<'_>) -> Result<(), E>,
) -> Result<(), E> {
    visit_links(
        &source,
        sym_table_msg,
        widths,
        source.addresses(),
        group_leaf_node_k,
        || {
            btree_v1::collect_symbol_table_nodes_from_source(
                &source.stored(),
                sym_table_msg.btree_address,
                widths,
            )
        },
        visit,
    )
}

/// Calls `visit` with each link of the version 1 group `sym_table_msg` describes, read from
/// `file`.
///
/// `symbol_table_nodes` returns the addresses of the group's symbol table nodes. The name and the
/// cache type of each entry are parsed before `visit` is called with it. A symbolic link's value
/// is parsed by [`SoftValue::path`].
///
/// # Errors
///
/// Returns the [`FormatError`] of the first group structure that does not parse, the error
/// `symbol_table_nodes` or `visit` returns, and the errors of [`RangeReader::read_range`].
fn visit_links<E: From<FormatError>>(
    file: &(impl RangeReader + ?Sized),
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    symbol_table_nodes: impl FnOnce() -> Result<Vec<StoredAddress>, FormatError>,
    mut visit: impl FnMut(LinkRef<'_>) -> Result<(), E>,
) -> Result<(), E> {
    let heap = read_local_heap(file, sym_table_msg, widths, addresses)?;
    let range = heap.data_segment_range();
    let bytes = file.read_range(range)?;
    let names = heap.data_segment(source::legacy_framed_region(range, &bytes)?)?;

    for address in symbol_table_nodes()? {
        let node = read_symbol_table_node(file, widths, addresses, group_leaf_node_k, address)?;
        for entry in &node.entries {
            visit(LinkRef::parse(names, entry)?)?;
        }
    }
    Ok(())
}

/// A link of a version 1 group, with its name borrowed from the local heap's data segment.
#[derive(Clone, Copy, Debug)]
struct LinkRef<'a> {
    name: &'a str,
    target: TargetRef<'a>,
}

impl<'a> LinkRef<'a> {
    /// Parses `entry` into a link, with the name at its link name offset in `names`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidSymbolTableCacheType`] if the cache type of `entry` is not
    /// [`UNCACHED`], [`CACHED_GROUP`], or [`SYMBOLIC_LINK`], and the errors of
    /// [`LocalHeapDataSegment::name`] for the link name.
    fn parse(
        names: LocalHeapDataSegment<'a>,
        entry: &SymbolTableEntry,
    ) -> Result<Self, FormatError> {
        let name = names.name(entry.link_name_offset)?;
        let target = match entry.cache_type {
            UNCACHED | CACHED_GROUP => TargetRef::Hard(entry.object_header_address),
            SYMBOLIC_LINK => {
                // The first four scratch-pad bytes store the local-heap offset of the link value,
                // as defined in "Symbol Table Entry" of the format specification, version 4.0.
                let [a, b, c, d, ..] = entry.scratch_pad;
                TargetRef::Soft(SoftValue {
                    names,
                    offset: u64::from(u32::from_le_bytes([a, b, c, d])),
                })
            }
            other => return Err(FormatError::InvalidSymbolTableCacheType(other)),
        };
        Ok(Self { name, target })
    }

    /// Returns the child entry of a hard link, or `None` for a symbolic link whose value parses.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`SoftValue::path`] for a symbolic link.
    fn into_group_entry(self) -> Result<Option<GroupEntry>, FormatError> {
        let Self { name, target } = self;
        Ok(match target {
            TargetRef::Hard(object_header_address) => Some(GroupEntry {
                name: String::from(name),
                object_header_address,
            }),
            TargetRef::Soft(value) => {
                value.path()?;
                None
            }
        })
    }

    // `H5G__ent_to_link` in `H5Gent.c` (HDF5 1.14.6 and 2.2.0) gives a symbol table entry's link
    // the ASCII character set and no creation index.
    fn into_owned_link(self, offset_width: OffsetWidth) -> Result<OwnedLink, LinkStorageError> {
        let Self { name, target } = self;
        let target = match target {
            TargetRef::Hard(address) if address.is_undefined(offset_width) => {
                return Err(LinkStorageError::UndefinedAddress);
            }
            TargetRef::Hard(object_header_address) => OwnedLinkTarget::Hard {
                object_header_address,
            },
            TargetRef::Soft(value) => OwnedLinkTarget::soft(value.path()?.as_bytes().to_vec()),
        };
        OwnedLink::new(name.as_bytes().to_vec(), CharacterSet::Ascii, target, None)
    }
}

/// The target of a version 1 group link.
#[derive(Clone, Copy, Debug)]
enum TargetRef<'a> {
    /// The object header address the symbol table entry stores.
    Hard(StoredAddress),
    /// The value of a symbolic link.
    Soft(SoftValue<'a>),
}

impl TargetRef<'_> {
    /// Returns the owned [`LinkTarget`], with the value of a symbolic link parsed.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`SoftValue::path`] for a symbolic link.
    fn to_link_target(self) -> Result<LinkTarget, FormatError> {
        Ok(match self {
            Self::Hard(object_header_address) => LinkTarget::Hard {
                object_header_address,
            },
            Self::Soft(value) => LinkTarget::soft(String::from(value.path()?)),
        })
    }
}

/// The value of a symbolic link, at `offset` in the local heap's data segment `names`.
#[derive(Clone, Copy, Debug)]
struct SoftValue<'a> {
    names: LocalHeapDataSegment<'a>,
    offset: u64,
}

impl<'a> SoftValue<'a> {
    /// Returns the path the symbolic link points to.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`LocalHeapDataSegment::name`].
    fn path(self) -> Result<&'a str, FormatError> {
        self.names.name(self.offset)
    }
}

/// Reads the header of the local heap `sym_table_msg` locates from `file`, and parses it.
///
/// # Errors
///
/// Returns the errors of [`LocalHeapLocator::parse`] and of [`RangeReader::read_range`].
fn read_local_heap(
    file: &(impl RangeReader + ?Sized),
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
) -> Result<LocalHeap, FormatError> {
    let locator = LocalHeapLocator::new(
        widths,
        addresses.effective_base(),
        sym_table_msg.local_heap_address,
    );
    let range = locator.parse_range()?;
    locator.parse(source::legacy_framed_region(
        range,
        &file.read_range(range)?,
    )?)
}

/// Reads the symbol table node at `address` from `file`, its prefix and then its entries in use,
/// and parses it.
///
/// # Errors
///
/// Returns the errors of [`SymbolTableNodeLocator::parse`], of
/// [`SymbolTableEntriesRef::parse`](hdf5_pure_format::__private::SymbolTableEntriesRef::parse),
/// and of [`RangeReader::read_range`].
fn read_symbol_table_node(
    file: &(impl RangeReader + ?Sized),
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    address: StoredAddress,
) -> Result<SymbolTableNode, FormatError> {
    let locator = SymbolTableNodeLocator::new(
        widths,
        addresses.effective_base(),
        address,
        group_leaf_node_k,
    );
    let range = locator.parse_range()?;
    let entries = locator.parse(source::legacy_framed_region(
        range,
        &file.read_range(range)?,
    )?)?;
    let range = entries.parse_range();
    entries.parse(source::legacy_framed_region(
        range,
        &file.read_range(range)?,
    )?)
}

/// The symbol table entry cache type for an entry without cached metadata.
///
/// Defined in "Symbol Table Entry" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
const UNCACHED: u32 = 0;

/// The symbol table entry cache type for an entry with cached group metadata.
///
/// Defined in "Symbol Table Entry" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
const CACHED_GROUP: u32 = 1;

/// The symbol table entry cache type of a symbolic link. Its object header address is the
/// undefined address and its scratch pad holds the local heap offset of the link's value, so the
/// entry identifies no object header in this file. libhdf5 calls this cache type
/// `H5G_CACHED_SLINK`.
///
/// The cache types are defined in "Symbol Table Entry" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
pub(crate) const SYMBOLIC_LINK: u32 = 2;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::btree_v1;
    use test_util::image::Image;
    use test_util::local_heap;
    use test_util::symbol_table;
    use test_util::widths::Widths;

    use super::*;
    use crate::access_mode::AccessMode;
    use crate::address::AddressContext;
    use crate::convert::Narrow;
    use crate::message_type::MessageType;
    use crate::object_header::ObjectHeader;
    use crate::placed_source::PlacedSource;

    #[derive(Clone, Copy, Debug)]
    enum DataSegment {
        FollowsHeader,
        AtTheEnd,
    }

    #[derive(Clone, Copy, Debug)]
    enum Child {
        Uncached(&'static str, u64),
        CachedGroup(&'static str, u64),
        Soft(&'static str, &'static str),
    }

    impl Child {
        fn name(self) -> &'static str {
            match self {
                Self::Uncached(name, _) | Self::CachedGroup(name, _) | Self::Soft(name, _) => name,
            }
        }

        fn soft_target(self) -> Option<&'static str> {
            match self {
                Self::Uncached(..) | Self::CachedGroup(..) => None,
                Self::Soft(_, target) => Some(target),
            }
        }

        fn link_target(self) -> LinkTarget {
            match self {
                Self::Uncached(_, address) | Self::CachedGroup(_, address) => LinkTarget::Hard {
                    object_header_address: StoredAddress::new(address),
                },
                Self::Soft(_, target) => LinkTarget::soft(String::from(target)),
            }
        }

        fn link(self) -> (String, LinkTarget) {
            (String::from(self.name()), self.link_target())
        }

        fn owned_link(self) -> OwnedLink {
            let target = match self {
                Self::Uncached(_, address) | Self::CachedGroup(_, address) => {
                    OwnedLinkTarget::Hard {
                        object_header_address: StoredAddress::new(address),
                    }
                }
                Self::Soft(_, target) => OwnedLinkTarget::soft(target.as_bytes().to_vec()),
            };
            OwnedLink::new(
                self.name().as_bytes().to_vec(),
                CharacterSet::Ascii,
                target,
                None,
            )
            .unwrap()
        }

        fn entry(
            self,
            widths: Widths,
            name_offset: u64,
            target_offset: impl FnOnce() -> u64,
        ) -> symbol_table::Entry {
            match self {
                Self::Uncached(_, address) => symbol_table::Entry::new(name_offset, address),
                Self::CachedGroup(_, address) => symbol_table::Entry {
                    cache_type: CACHED_GROUP,
                    ..symbol_table::Entry::new(name_offset, address)
                },
                Self::Soft(..) => {
                    let mut scratch_pad = [0; symbol_table::SCRATCH_PAD];
                    scratch_pad[..4]
                        .copy_from_slice(&u32::try_from(target_offset()).unwrap().to_le_bytes());
                    symbol_table::Entry {
                        cache_type: SYMBOLIC_LINK,
                        scratch_pad,
                        ..symbol_table::Entry::new(
                            name_offset,
                            u64::MAX >> (64 - 8 * widths.offset),
                        )
                    }
                }
            }
        }
    }

    struct Group {
        bytes: Vec<u8>,
        message: SymbolTableMessage,
        heap_header_len: usize,
        data_segment: (u64, usize),
        btree: (u64, usize),
        nodes: Vec<(u64, usize)>,
    }

    impl Group {
        fn file(&self, base: u64) -> Vec<u8> {
            let mut file = vec![0; usize::try_from(base).unwrap()];
            file.extend_from_slice(&self.bytes);
            file
        }

        fn set_entry_field(&mut self, entry: usize, field_at: usize, value: u32) {
            let at = usize::try_from(self.nodes[0].0).unwrap()
                + symbol_table::NODE_PREFIX
                + entry * symbol_table::Entry::encoded_len(Widths::EIGHT)
                + field_at;
            test_util::bytes::set_uint_at(&mut self.bytes, at, 4, value.into());
        }

        fn reads(&self, base: u64) -> Vec<(u64, usize)> {
            let Self {
                bytes: _,
                message: _,
                heap_header_len,
                data_segment,
                btree: (btree_at, btree_len),
                nodes,
            } = self;
            [
                (0, *heap_header_len),
                *data_segment,
                (*btree_at, 8),
                (*btree_at, *btree_len),
            ]
            .into_iter()
            .chain(
                nodes
                    .iter()
                    .flat_map(|&(at, len)| [(at, 8), (at + 8, len - 8)]),
            )
            .map(|(at, len)| (base + at, len))
            .collect()
        }
    }

    fn group(nodes: &[&[Child]], widths: Widths, data_segment: DataSegment) -> Group {
        let children: Vec<Child> = nodes
            .iter()
            .flat_map(|children| children.iter().copied())
            .collect();
        let names: Vec<&str> = children
            .iter()
            .map(|child| child.name())
            .chain(children.iter().filter_map(|child| child.soft_target()))
            .collect();
        let segment = local_heap::Segment::of_names(0, &names);
        let header_len = local_heap::header_len(widths);
        let mut image = Image::new();
        image.place(0, &vec![0; header_len]);
        if matches!(data_segment, DataSegment::FollowsHeader) {
            image.append(&segment.bytes);
        }

        let btree = |node_addresses: &[u64]| {
            btree_v1::node_with_group_keys(
                btree_v1::NodeType::GROUP,
                0,
                &vec![0; node_addresses.len() + 1],
                node_addresses,
                widths,
            )
        };
        let btree_len = btree(&vec![0; nodes.len()]).len();
        let btree_at = image.append_aligned(&vec![0; btree_len], 8);

        let mut name_offsets = (0..children.len()).map(|index| segment.offset_of(index));
        let mut target_offsets =
            (children.len()..names.len()).map(|index| segment.offset_of(index));
        let nodes: Vec<(u64, usize)> = nodes
            .iter()
            .map(|children| {
                let entries: Vec<_> = children
                    .iter()
                    .map(|child| {
                        child.entry(widths, name_offsets.next().unwrap(), || {
                            target_offsets.next().unwrap()
                        })
                    })
                    .collect();
                let node = symbol_table::node(&entries, widths);
                (image.append_aligned(&node, 8) as u64, node.len())
            })
            .collect();
        image.place(
            btree_at,
            &btree(&nodes.iter().map(|&(at, _)| at).collect::<Vec<_>>()),
        );

        let segment_at = match data_segment {
            DataSegment::FollowsHeader => header_len,
            DataSegment::AtTheEnd => image.append_aligned(&segment.bytes, 8),
        } as u64;
        image.place(
            0,
            &local_heap::Segment::of_names(segment_at, &names).header(widths),
        );
        Group {
            bytes: image.build(),
            message: SymbolTableMessage {
                btree_address: StoredAddress::new(btree_at as u64),
                local_heap_address: StoredAddress::ZERO,
            },
            heap_header_len: header_len,
            data_segment: (segment_at, segment.bytes.len()),
            btree: (btree_at as u64, btree_len),
            nodes,
        }
    }

    fn format_widths(Widths { offset, length }: Widths) -> FormatWidths {
        FormatWidths::from_sizes(offset.try_into().unwrap(), length.try_into().unwrap()).unwrap()
    }

    fn addresses(base: u64) -> AddressContext {
        AddressContext::from_effective_base(BaseAddress::new(base))
    }

    fn k(k: u16) -> Option<NonZeroU16> {
        NonZeroU16::new(k)
    }

    fn listed(entries: Vec<GroupEntry>) -> Vec<(String, StoredAddress)> {
        entries
            .into_iter()
            .map(|entry| (entry.name, entry.object_header_address))
            .collect()
    }

    fn hard_children(nodes: &[&[Child]]) -> Vec<(String, StoredAddress)> {
        nodes
            .iter()
            .flat_map(|children| children.iter())
            .filter_map(|child| match *child {
                Child::Uncached(name, address) | Child::CachedGroup(name, address) => {
                    Some((String::from(name), StoredAddress::new(address)))
                }
                Child::Soft(..) => None,
            })
            .collect()
    }

    fn read_both(
        group: &Group,
        widths: Widths,
        base: u64,
        group_leaf_node_k: Option<NonZeroU16>,
    ) -> [Result<Vec<(String, StoredAddress)>, FormatError>; 2] {
        let widths = format_widths(widths);
        [
            resolve_v1_group_entries(
                PhysicalBytes::for_test(&group.file(base), addresses(base)),
                &group.message,
                widths,
                group_leaf_node_k,
            )
            .map(listed),
            resolve_v1_group_entries_from_source(
                PhysicalSource::for_test(
                    &PlacedSource::new(base, group.bytes.clone()),
                    addresses(base),
                ),
                &group.message,
                widths,
                group_leaf_node_k,
            )
            .map(listed),
        ]
    }

    fn read_links_both(
        group: &Group,
        widths: Widths,
        base: u64,
    ) -> [Result<Vec<(String, LinkTarget)>, FormatError>; 2] {
        let widths = format_widths(widths);
        let mut buffered = Vec::new();
        let mut streaming = Vec::new();
        [
            visit_v1_group_links(
                PhysicalBytes::for_test(&group.file(base), addresses(base)),
                &group.message,
                widths,
                k(2),
                |link| {
                    buffered.push(owned(link)?);
                    Ok(())
                },
            )
            .map(|()| buffered),
            visit_v1_group_links_from_source(
                PhysicalSource::for_test(
                    &PlacedSource::new(base, group.bytes.clone()),
                    addresses(base),
                ),
                &group.message,
                widths,
                k(2),
                |link| {
                    streaming.push(owned(link)?);
                    Ok(())
                },
            )
            .map(|()| streaming),
        ]
    }

    fn owned(link: LinkRef<'_>) -> Result<(String, LinkTarget), FormatError> {
        Ok((String::from(link.name), link.target.to_link_target()?))
    }

    fn look_up_both(
        group: &Group,
        widths: Widths,
        base: u64,
        name: &str,
    ) -> [Result<Option<LinkTarget>, FormatError>; 2] {
        let widths = format_widths(widths);
        let name = LinkName::new(name).unwrap();
        [
            find_link(
                PhysicalBytes::for_test(&group.file(base), addresses(base)),
                &group.message,
                widths,
                k(2),
                name,
            ),
            find_link_from_source(
                PhysicalSource::for_test(
                    &PlacedSource::new(base, group.bytes.clone()),
                    addresses(base),
                ),
                &group.message,
                widths,
                k(2),
                name,
            ),
        ]
    }

    fn owned_links_both(
        group: &Group,
        widths: Widths,
        base: u64,
    ) -> [Result<Vec<OwnedLink>, LinkStorageError>; 2] {
        let widths = format_widths(widths);
        [
            resolve_v1_group_links(
                PhysicalBytes::for_test(&group.file(base), addresses(base)),
                &group.message,
                widths,
                k(2),
            ),
            resolve_v1_group_links_from_source(
                PhysicalSource::for_test(
                    &PlacedSource::new(base, group.bytes.clone()),
                    addresses(base),
                ),
                &group.message,
                widths,
                k(2),
            ),
        ]
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::borrowed(false)]
    #[case::owned(true)]
    fn group_reads_preserve_metadata_ownership_and_cache_admission(
        #[case] owned: bool,
        #[values(0, 512)] base: u64,
    ) {
        use core::cell::Cell;

        use crate::source::BytesSource;
        use crate::source::MetadataCacheConfig;
        use crate::source::MetadataCachingSource;

        struct CountingSource {
            inner: BytesSource<Vec<u8>>,
            reads: Cell<u64>,
        }

        impl Source for CountingSource {
            fn len(&self) -> u64 {
                self.inner.len()
            }
            fn metadata_origin(&self) -> Result<u64, FormatError> {
                Ok(37)
            }
            fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
                self.reads.set(self.reads.get() + 1);
                self.inner.read_at(offset, buffer)
            }
        }

        let group = group(&[CHILDREN], Widths::EIGHT, DataSegment::AtTheEnd);
        let image = group.file(base);
        let borrowed = BytesSource::new(&image);
        let allocating = CountingSource {
            inner: BytesSource::new(image.clone()),
            reads: Cell::new(0),
        };
        let source: &dyn Source = if owned { &allocating } else { &borrowed };
        let cached = MetadataCachingSource::new(source, MetadataCacheConfig::new(8192));
        let physical = PhysicalSource::for_test(&cached, addresses(base));
        let widths = format_widths(Widths::EIGHT);
        let first =
            resolve_v1_group_links_from_source(physical, &group.message, widths, k(2)).unwrap();
        assert_eq!(first, owned_links(&[CHILDREN]));
        let before = cached.metadata_cache_stats().unwrap();
        let reads = allocating.reads.get();
        let second =
            resolve_v1_group_links_from_source(physical, &group.message, widths, k(2)).unwrap();
        assert_eq!(second, first);
        let after = cached.metadata_cache_stats().unwrap();
        if owned {
            assert!(before.entries() > 0);
            assert!(after.hits() > before.hits());
            assert_eq!(after.misses(), before.misses());
            assert_eq!(allocating.reads.get(), reads);
        } else {
            assert_eq!((after.hits(), after.entries(), after.bytes()), (0, 0, 0));
            assert_eq!(after.misses(), 2 * before.misses());
        }
        let stored = physical.stored();
        let address = group.message.btree_address;
        let metadata = stored
            .read_metadata(address, btree_v1::SIGNATURE.len())
            .unwrap();
        assert_eq!(metadata.as_ref(), btree_v1::SIGNATURE);
        assert_eq!(
            metadata.offset(),
            base + address.get() + if owned { 37 } else { 0 }
        );
        assert_eq!(metadata.shared().is_some(), owned);
        if !owned {
            assert_eq!(
                metadata.as_ptr(),
                image[(base + address.get()).to_usize().unwrap()..].as_ptr()
            );
        }
        let retained = metadata.shared();
        drop(metadata);
        drop(cached);
        if let Some(retained) = retained {
            assert_eq!(retained.as_ref(), btree_v1::SIGNATURE);
            assert_eq!(retained.offset(), base + address.get() + 37);
        }
    }

    fn owned_links(nodes: &[&[Child]]) -> Vec<OwnedLink> {
        nodes
            .iter()
            .flat_map(|children| children.iter())
            .map(|child| child.owned_link())
            .collect()
    }

    #[track_caller]
    fn assert_the_listings_and_the_lookup_reject(
        group: &Group,
        name: &str,
        expected: &FormatError,
    ) {
        assert_eq!(
            read_links_both(group, Widths::EIGHT, 0),
            [Err(expected.clone()), Err(expected.clone())]
        );
        assert_eq!(
            read_both(group, Widths::EIGHT, 0, k(2)),
            [Err(expected.clone()), Err(expected.clone())]
        );
        assert_eq!(
            look_up_both(group, Widths::EIGHT, 0, name),
            [Err(expected.clone()), Err(expected.clone())]
        );
        let expected = LinkStorageError::Format(expected.clone());
        assert_eq!(
            owned_links_both(group, Widths::EIGHT, 0),
            [Err(expected.clone()), Err(expected)]
        );
    }

    #[rstest]
    #[case::uncached(Child::Uncached("data", 0x1000))]
    #[case::cached_group(Child::CachedGroup("group", 0x2000))]
    #[case::absolute_soft_link(Child::Soft("absolute", "/group/data"))]
    #[case::relative_soft_link(Child::Soft("relative", "data"))]
    #[case::dangling_soft_link(Child::Soft("dangling", "missing"))]
    fn both_readers_read_each_link_as_its_typed_record(
        #[case] child: Child,
        #[values(Widths::new(2, 2), Widths::FOUR, Widths::new(8, 2), Widths::EIGHT)] widths: Widths,
        #[values(0, 512)] base: u64,
    ) {
        let nodes: &[&[Child]] = &[&[Child::Uncached("sibling", 0x300), child]];
        let group = group(nodes, widths, DataSegment::AtTheEnd);
        let expected = vec![Child::Uncached("sibling", 0x300).link(), child.link()];
        assert_eq!(
            read_links_both(&group, widths, base),
            [Ok(expected.clone()), Ok(expected)]
        );
        let expected = Some(child.link_target());
        assert_eq!(
            look_up_both(&group, widths, base, child.name()),
            [Ok(expected.clone()), Ok(expected)]
        );
        let expected = hard_children(nodes);
        assert_eq!(
            read_both(&group, widths, base, k(2)),
            [Ok(expected.clone()), Ok(expected)]
        );
        let expected = owned_links(nodes);
        assert_eq!(
            owned_links_both(&group, widths, base),
            [Ok(expected.clone()), Ok(expected)]
        );
    }

    #[rstest]
    fn resolving_the_links_rejects_a_hard_link_to_the_undefined_address(
        #[values(Widths::new(2, 2), Widths::FOUR, Widths::EIGHT)] widths: Widths,
    ) {
        let undefined = u64::MAX >> (64 - 8 * widths.offset);
        let group = group(
            &[&[
                Child::Uncached("child", 0x100),
                Child::Uncached("undefined", undefined),
            ]],
            widths,
            DataSegment::FollowsHeader,
        );
        assert_eq!(
            owned_links_both(&group, widths, 0),
            [
                Err(LinkStorageError::UndefinedAddress),
                Err(LinkStorageError::UndefinedAddress)
            ]
        );
    }

    #[rstest]
    #[case::first_unknown_type(3)]
    #[case::maximum_cache_type(u32::MAX)]
    fn every_read_rejects_an_unknown_symbol_table_cache_type(#[case] cache_type: u32) {
        let mut group = group(
            &[&[
                Child::Uncached("child", 0x100),
                Child::Uncached("other", 0x200),
            ]],
            Widths::EIGHT,
            DataSegment::FollowsHeader,
        );
        group.set_entry_field(
            1,
            symbol_table::Entry::cache_type_at(Widths::EIGHT),
            cache_type,
        );
        assert_the_listings_and_the_lookup_reject(
            &group,
            "child",
            &FormatError::InvalidSymbolTableCacheType(cache_type),
        );
    }

    #[rstest]
    #[case::at_the_end_of_the_data_segment(None)]
    #[case::maximum_offset(Some(u32::MAX))]
    fn a_soft_link_value_outside_the_local_heap_is_rejected_where_it_is_read(
        #[case] offset: Option<u32>,
    ) {
        let mut group = group(
            &[&[
                Child::Uncached("child", 0x100),
                Child::Soft("soft", "/child"),
            ]],
            Widths::EIGHT,
            DataSegment::FollowsHeader,
        );
        let data_segment_size = group.data_segment.1.to_u64();
        let offset = offset.unwrap_or(u32::try_from(data_segment_size).unwrap());
        group.set_entry_field(
            1,
            symbol_table::Entry::scratch_pad_at(Widths::EIGHT),
            offset,
        );
        assert_the_listings_and_the_lookup_reject(
            &group,
            "soft",
            &FormatError::InvalidLocalHeapNameOffset {
                offset: offset.into(),
                data_segment_size,
            },
        );
        assert_the_sibling_lookup_succeeds(&group);
    }

    #[test]
    fn a_soft_link_value_that_is_not_utf8_is_rejected_where_it_is_read() {
        let mut group = group(
            &[&[
                Child::Uncached("child", 0x100),
                Child::Soft("soft", "/child"),
            ]],
            Widths::EIGHT,
            DataSegment::FollowsHeader,
        );
        let segment_at = usize::try_from(group.data_segment.0).unwrap();
        let value_at =
            test_util::bytes::find_signature(&group.bytes[segment_at..], b"/child\0").unwrap();
        let value = segment_at + value_at..segment_at + value_at + "/child".len();
        group.bytes[value.start + 1] = 0xFF;
        let source = core::str::from_utf8(&group.bytes[value]).unwrap_err();
        assert_eq!(source.valid_up_to(), 1);
        assert_the_listings_and_the_lookup_reject(
            &group,
            "soft",
            &FormatError::InvalidLocalHeapName {
                offset: value_at.to_u64(),
                source,
            },
        );
        assert_the_sibling_lookup_succeeds(&group);
    }

    #[track_caller]
    fn assert_the_sibling_lookup_succeeds(group: &Group) {
        let hard = Some(Child::Uncached("child", 0x100).link_target());
        assert_eq!(
            look_up_both(group, Widths::EIGHT, 0, "child"),
            [Ok(hard.clone()), Ok(hard)]
        );
    }

    const CHILDREN: &[Child] = &[
        Child::Uncached("alpha", 0x1000),
        Child::CachedGroup("beta", 0x2000),
    ];

    const THREE_NODES: &[&[Child]] = &[
        &[
            Child::Uncached("a", 0x100),
            Child::Uncached("b", 0x200),
            Child::CachedGroup("c", 0x300),
            Child::Uncached("d", 0x400),
        ],
        &[Child::Uncached("e", 0x500), Child::Uncached("f", 0x600)],
        &[
            Child::CachedGroup("g", 0x700),
            Child::Soft("s", "/a"),
            Child::Uncached("h", 0x800),
            Child::Uncached("i", 0x900),
        ],
    ];

    #[rstest]
    fn both_readers_list_every_child(
        #[values(
            Widths::new(2, 2),
            Widths::new(2, 8),
            Widths::FOUR,
            Widths::new(8, 2),
            Widths::EIGHT
        )]
        widths: Widths,
        #[values(0, 512)] base: u64,
        #[values(DataSegment::FollowsHeader, DataSegment::AtTheEnd)] data_segment: DataSegment,
        #[values(&[CHILDREN][..], THREE_NODES)] nodes: &[&[Child]],
    ) {
        let group = group(nodes, widths, data_segment);
        let expected = hard_children(nodes);
        assert_eq!(
            read_both(&group, widths, base, k(2)),
            [Ok(expected.clone()), Ok(expected)]
        );
    }

    #[rstest]
    fn the_streaming_reader_reads_the_heap_header_its_data_segment_and_each_node(
        #[values(0, 512, 5 << 32)] base: u64,
        #[values(DataSegment::FollowsHeader, DataSegment::AtTheEnd)] data_segment: DataSegment,
    ) {
        let group = group(THREE_NODES, Widths::EIGHT, data_segment);
        let source = PlacedSource::new(base, group.bytes.clone());
        let entries = resolve_v1_group_entries_from_source(
            PhysicalSource::for_test(&source, addresses(base)),
            &group.message,
            format_widths(Widths::EIGHT),
            k(2),
        )
        .unwrap();
        assert_eq!(listed(entries), hard_children(THREE_NODES));
        assert_eq!(source.reads.into_inner(), group.reads(base));
    }

    #[rstest]
    #[case::data_segment(DataSegment::AtTheEnd)]
    #[case::node(DataSegment::FollowsHeader)]
    fn a_structure_past_the_end_of_the_file_is_rejected(#[case] data_segment: DataSegment) {
        let mut group = group(&[CHILDREN], Widths::EIGHT, data_segment);
        let (at, len) = match data_segment {
            DataSegment::AtTheEnd => group.data_segment,
            DataSegment::FollowsHeader => group.nodes[0],
        };
        assert_eq!(usize::try_from(at).unwrap() + len, group.bytes.len());
        let end = usize::try_from(at).unwrap() + len;
        group.bytes.truncate(end - 1);
        let expected = Err(FormatError::UnexpectedSourceEof {
            expected: end.to_u64(),
            available: (end - 1).to_u64(),
        });
        assert_eq!(
            read_both(&group, Widths::EIGHT, 0, k(2)),
            [expected.clone(), expected]
        );
    }

    #[rstest]
    #[case::heap_signature(0, b'X', FormatError::InvalidLocalHeapSignature)]
    #[case::heap_version(4, 1, FormatError::InvalidLocalHeapVersion(1))]
    fn a_malformed_local_heap_is_rejected(
        #[case] at: usize,
        #[case] byte: u8,
        #[case] expected: FormatError,
    ) {
        let mut group = group(&[CHILDREN], Widths::EIGHT, DataSegment::FollowsHeader);
        group.bytes[at] = byte;
        assert_eq!(
            read_both(&group, Widths::EIGHT, 0, k(2)),
            [Err(expected.clone()), Err(expected)]
        );
    }

    #[rstest]
    #[case::signature(0, b'X', FormatError::InvalidSymbolTableNodeSignature)]
    #[case::version(4, 2, FormatError::InvalidSymbolTableNodeVersion(2))]
    #[case::symbols(6, 5, FormatError::InvalidSymbolTableNodeSymbolCount { symbols: 5, capacity: 4 })]
    fn a_malformed_symbol_table_node_is_rejected(
        #[case] at: usize,
        #[case] byte: u8,
        #[case] expected: FormatError,
    ) {
        let mut group = group(THREE_NODES, Widths::EIGHT, DataSegment::FollowsHeader);
        let node_at = usize::try_from(group.nodes[1].0).unwrap();
        group.bytes[node_at + at] = byte;
        assert_eq!(
            read_both(&group, Widths::EIGHT, 0, k(2)),
            [Err(expected.clone()), Err(expected)]
        );
    }

    #[rstest]
    #[case::at_capacity(k(2), Ok(hard_children(THREE_NODES)))]
    #[case::above_capacity(
        k(1),
        Err(FormatError::InvalidSymbolTableNodeSymbolCount { symbols: 4, capacity: 2 })
    )]
    #[case::unknown_capacity(None, Ok(hard_children(THREE_NODES)))]
    fn a_node_holds_at_most_twice_the_group_leaf_node_k(
        #[case] group_leaf_node_k: Option<NonZeroU16>,
        #[case] expected: Result<Vec<(String, StoredAddress)>, FormatError>,
    ) {
        let group = group(THREE_NODES, Widths::EIGHT, DataSegment::FollowsHeader);
        assert_eq!(
            read_both(&group, Widths::EIGHT, 0, group_leaf_node_k),
            [expected.clone(), expected]
        );
    }

    #[rstest]
    #[case::past_the_data_segment(14, FormatError::InvalidLocalHeapNameOffset {
        offset: 14,
        data_segment_size: 14,
    })]
    #[case::without_a_terminator(11, FormatError::UnterminatedLocalHeapName { offset: 11 })]
    fn a_name_that_runs_past_the_data_segment_is_rejected(
        #[case] offset: u64,
        #[case] expected: FormatError,
    ) {
        let mut group = group(&[CHILDREN], Widths::EIGHT, DataSegment::AtTheEnd);
        let (segment_at, len) = group.data_segment;
        let segment_at = usize::try_from(segment_at).unwrap();
        group.bytes.splice(segment_at + len.., *b"xyz");
        test_util::bytes::set_uint_at(&mut group.bytes, 8, 8, 14);
        let node_at = usize::try_from(group.nodes[0].0).unwrap();
        test_util::bytes::set_uint_at(&mut group.bytes, node_at + 8, 8, offset);
        assert_eq!(
            read_both(&group, Widths::EIGHT, 0, k(2)),
            [Err(expected.clone()), Err(expected)]
        );
    }

    #[rstest]
    #[case::heap_header(0)]
    #[case::data_segment(1)]
    #[case::node_prefix(4)]
    #[case::node_entries(5)]
    fn a_failed_read_reports_the_source_error(#[case] failing_read: usize) {
        let group = group(&[CHILDREN], Widths::EIGHT, DataSegment::AtTheEnd);
        let source = FailingSource {
            inner: PlacedSource::new(512, group.bytes.clone()),
            failing_read: group.reads(512)[failing_read],
        };
        assert_eq!(
            resolve_v1_group_entries_from_source(
                PhysicalSource::for_test(&source, addresses(512)),
                &group.message,
                format_widths(Widths::EIGHT),
                k(2),
            )
            .unwrap_err(),
            FormatError::Source(String::from("injected read failure"))
        );
    }

    struct FailingSource {
        inner: PlacedSource,
        failing_read: (u64, usize),
    }

    impl Source for FailingSource {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.inner.read_at(offset, buf)
        }

        fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            if (offset, len) == self.failing_read {
                return Err(FormatError::Source(String::from("injected read failure")));
            }
            self.inner.read_metadata_at(offset, len)
        }
    }

    #[rstest]
    #[case::soft_link(&[
        Child::Uncached("data", 0x1000),
        Child::Soft("pointer", "data"),
        Child::CachedGroup("group", 0x2000),
    ])]
    #[case::hard_links_only(CHILDREN)]
    fn the_listing_passes_over_symbolic_links_and_resolving_the_links_includes_them(
        #[case] children: &[Child],
    ) {
        let group = group(&[children], Widths::EIGHT, DataSegment::FollowsHeader);
        let expected = hard_children(&[children]);
        assert_eq!(
            read_both(&group, Widths::EIGHT, 0, k(2)),
            [Ok(expected.clone()), Ok(expected)]
        );
        let expected = owned_links(&[children]);
        assert_eq!(
            owned_links_both(&group, Widths::EIGHT, 0),
            [Ok(expected.clone()), Ok(expected)]
        );
    }

    // Helper to extract dataset components from an object header
    fn extract_dataset(
        _file_data: &[u8],
        hdr: &crate::object_header::ObjectHeader,
        widths: FormatWidths,
    ) -> (
        crate::datatype::Datatype,
        crate::dataspace::Dataspace,
        crate::data_layout::DataLayout,
    ) {
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
        let ds =
            crate::dataspace::Dataspace::parse_with_length_width(ds_data, widths.lengths).unwrap();
        let dl = crate::data_layout::DataLayout::parse_with_widths(dl_data, widths).unwrap();
        (dt, ds, dl)
    }

    fn get_root_sym_table(
        file_data: &[u8],
        sb: &crate::superblock::Superblock,
    ) -> SymbolTableMessage {
        let root_header = ObjectHeader::parse_with_widths(
            file_data,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(sb.base_address)
                .resolve(sb.root_group_address)
                .unwrap()
                .get()
                .to_usize()
                .unwrap(),
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let sym_msg = root_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
            .unwrap();
        SymbolTableMessage::parse_with_offset_width(&sym_msg.data, sb.offset_size).unwrap()
    }

    // Integration tests with real HDF5 files

    #[test]
    fn integration_simple_dataset_full_traversal() {
        let bytes = test_util::legacy::simple_dataset();
        let file_data = bytes.as_slice();
        let sb = crate::superblock::read(file_data).unwrap().superblock;
        let root_sym = get_root_sym_table(file_data, &sb);

        let entries = resolve_v1_group_entries(
            PhysicalBytes::for_test(file_data, addresses(0)),
            &root_sym,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
            group_leaf_node_k(&sb),
        )
        .unwrap();
        let data_entry = entries
            .iter()
            .find(|e| e.name == "data")
            .expect("should have 'data'");

        let hdr = ObjectHeader::parse_with_widths(
            file_data,
            AccessMode::ReadOnly,
            data_entry.object_header_address.get() as usize,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(
            file_data,
            &hdr,
            FormatWidths {
                offsets: sb.offset_size,
                lengths: sb.length_size,
            },
        );
        let raw = crate::data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = crate::data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
