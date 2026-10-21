//! V1 group traversal: resolve group children and navigate paths.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
use core::num::NonZeroU16;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::LocalHeap;
use hdf5_pure_format::__private::LocalHeapLocator;
use hdf5_pure_format::__private::SymbolTableEntry;
use hdf5_pure_format::__private::SymbolTableNode;
use hdf5_pure_format::__private::SymbolTableNodeLocator;

#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, StoredAddress};
#[cfg(test)]
use crate::address::{BaseAddress, BaseAddressExt};
use crate::btree_v1;
use crate::error::FormatError;
use crate::link_message::LinkTarget;
use crate::object_path::LinkName;
use crate::source::RangeReader;
use crate::source::Source;
use crate::superblock::Superblock;
use crate::symbol_table::SymbolTableMessage;
use crate::width::FormatWidths;

/// A resolved group entry (child name + object header address).
#[derive(Debug, Clone)]
pub struct GroupEntry {
    /// Name of the child object.
    pub name: String,
    /// Address of the child's object header.
    pub object_header_address: StoredAddress,
    /// Cache type from the symbol table (SNOD) entry. The group walks never return a
    /// [`SYMBOLIC_LINK`] entry.
    #[allow(dead_code)]
    pub cache_type: u32,
}

/// Returns one entry per child using validated file widths.
///
/// A symbolic link ([`SYMBOLIC_LINK`]) identifies no object header in this file and is passed
/// over, as a version 2 group's soft links are. `group_leaf_node_k` is the value
/// [`group_leaf_node_k`] returns for the file's superblock.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first group structure that does not parse.
pub fn resolve_v1_group_entries(
    file_data: &[u8],
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<GroupEntry>, FormatError> {
    symbols(
        file_data,
        sym_table_msg,
        widths,
        addresses,
        group_leaf_node_k,
        || {
            btree_v1::collect_symbol_table_nodes(
                file_data,
                sym_table_msg.btree_address,
                widths,
                addresses,
            )
        },
        is_child,
    )
}

/// Returns one entry per child from `source` using validated file widths.
///
/// # Errors
///
/// Returns the errors of [`resolve_v1_group_entries`], and the error `source` reports
/// for a structure it cannot read.
pub fn resolve_v1_group_entries_from_source<S: Source + ?Sized>(
    source: &S,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<Vec<GroupEntry>, FormatError> {
    v1_symbols_from_source(
        source,
        sym_table_msg,
        widths,
        addresses,
        group_leaf_node_k,
        is_child,
    )
}

/// Returns the target of the version 1 group link named `name`.
///
/// # Errors
///
/// Returns [`FormatError::InvalidSymbolTableCacheType`] for an unsupported cache type,
/// and errors from the local heap and symbol table parsers and `file` reads.
pub(crate) fn find_link(
    file: &(impl RangeReader + ?Sized),
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    name: LinkName<'_>,
    symbol_table_nodes: impl FnOnce() -> Result<Vec<StoredAddress>, FormatError>,
) -> Result<Option<LinkTarget>, FormatError> {
    let heap = read_local_heap(file, sym_table_msg, widths, addresses)?;
    let range = heap.data_segment_range();
    let bytes = file.read_range(range)?;
    let names = heap.data_segment(FileRegion::new(range, &bytes)?)?;
    let mut found = None;
    for address in symbol_table_nodes()? {
        let node = read_symbol_table_node(file, widths, addresses, group_leaf_node_k, address)?;
        for entry in node.entries {
            let link_name = names.name(entry.link_name_offset)?;
            if link_name != name.as_str() {
                continue;
            }
            let target = match entry.cache_type {
                UNCACHED | CACHED_GROUP => LinkTarget::Hard {
                    object_header_address: entry.object_header_address,
                },
                SYMBOLIC_LINK => {
                    let [a, b, c, d, ..] = entry.scratch_pad;
                    // The first four scratch-pad bytes store the local-heap offset, as defined in
                    // "Symbol Table Entry" of the format specification, version 4.0.
                    let offset = u64::from(u32::from_le_bytes([a, b, c, d]));
                    LinkTarget::Soft {
                        target_path: String::from(names.name(offset)?),
                    }
                }
                other => return Err(FormatError::InvalidSymbolTableCacheType(other)),
            };
            found = found.or(Some(target));
        }
    }
    Ok(found)
}

/// Returns whether the version 1 group `sym_table_msg` describes holds a symbolic link
/// ([`SYMBOLIC_LINK`]), which [`resolve_v1_group_entries_from_source`] passes over.
///
/// # Errors
///
/// Returns the errors of [`resolve_v1_group_entries_from_source`].
pub(crate) fn v1_group_holds_symbolic_link_from_source<S: Source + ?Sized>(
    source: &S,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
) -> Result<bool, FormatError> {
    Ok(!v1_symbols_from_source(
        source,
        sym_table_msg,
        widths,
        addresses,
        group_leaf_node_k,
        |entry| entry.cache_type == SYMBOLIC_LINK,
    )?
    .is_empty())
}

/// Returns the "Group Leaf Node K" field of `superblock`, or `None` where the field is absent or 0.
///
/// A version 2 or 3 superblock does not have the field.
pub(crate) fn group_leaf_node_k(superblock: &Superblock) -> Option<NonZeroU16> {
    superblock.group_leaf_node_k.and_then(NonZeroU16::new)
}

/// Returns the entries of the version 1 group `sym_table_msg` describes for which `keep` returns
/// `true`.
fn v1_symbols_from_source<S: Source + ?Sized>(
    source: &S,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    keep: impl Fn(&SymbolTableEntry) -> bool,
) -> Result<Vec<GroupEntry>, FormatError> {
    symbols(
        source,
        sym_table_msg,
        widths,
        addresses,
        group_leaf_node_k,
        || {
            btree_v1::collect_symbol_table_nodes_from_source(
                source,
                sym_table_msg.btree_address,
                widths,
                addresses,
            )
        },
        keep,
    )
}

/// Returns whether `entry` points to an object header, which every entry but a symbolic link
/// ([`SYMBOLIC_LINK`]) does.
fn is_child(entry: &SymbolTableEntry) -> bool {
    entry.cache_type != SYMBOLIC_LINK
}

/// Returns the entries of the version 1 group `sym_table_msg` describes for which `keep` returns
/// `true`, read from `file`.
///
/// `symbol_table_nodes` returns the addresses of the group's symbol table nodes. The name of every
/// entry is parsed, including the names of the entries for which `keep` returns `false`.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first group structure that does not parse, and the errors
/// of [`RangeReader::read_range`].
fn symbols(
    file: &(impl RangeReader + ?Sized),
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
    group_leaf_node_k: Option<NonZeroU16>,
    symbol_table_nodes: impl FnOnce() -> Result<Vec<StoredAddress>, FormatError>,
    keep: impl Fn(&SymbolTableEntry) -> bool,
) -> Result<Vec<GroupEntry>, FormatError> {
    let heap = read_local_heap(file, sym_table_msg, widths, addresses)?;
    let range = heap.data_segment_range();
    let bytes = file.read_range(range)?;
    let names = heap.data_segment(FileRegion::new(range, &bytes)?)?;

    let mut entries = Vec::new();
    for address in symbol_table_nodes()? {
        let node = read_symbol_table_node(file, widths, addresses, group_leaf_node_k, address)?;
        for entry in node.entries {
            let link_name = names.name(entry.link_name_offset)?;
            if keep(&entry) {
                entries.push(GroupEntry {
                    name: String::from(link_name),
                    object_header_address: entry.object_header_address,
                    cache_type: entry.cache_type,
                });
            }
        }
    }
    Ok(entries)
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
    locator.parse(FileRegion::new(range, &file.read_range(range)?)?)
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
    let entries = locator.parse(FileRegion::new(range, &file.read_range(range)?)?)?;
    let range = entries.parse_range();
    entries.parse(FileRegion::new(range, &file.read_range(range)?)?)
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
/// entry identifies no object header in this file. The C library calls this cache type
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

    struct Group {
        bytes: Vec<u8>,
        message: SymbolTableMessage,
        heap_header_len: usize,
        data_segment: (u64, usize),
        btree: (u64, usize),
        nodes: Vec<(u64, usize)>,
    }

    impl Group {
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

    fn group(nodes: &[&[(&str, u64, u32)]], widths: Widths, data_segment: DataSegment) -> Group {
        let names: Vec<&str> = nodes
            .iter()
            .flat_map(|children| children.iter().map(|&(name, _, _)| name))
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

        let mut name_offsets = (0..names.len()).map(|index| segment.offset_of(index));
        let nodes: Vec<(u64, usize)> = nodes
            .iter()
            .map(|children| {
                let entries: Vec<_> = children
                    .iter()
                    .map(|&(_, header_address, cache_type)| symbol_table::Entry {
                        cache_type,
                        ..symbol_table::Entry::new(name_offsets.next().unwrap(), header_address)
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

    fn links(entries: Vec<GroupEntry>) -> Vec<(String, StoredAddress)> {
        entries
            .into_iter()
            .map(|entry| (entry.name, entry.object_header_address))
            .collect()
    }

    fn links_of(nodes: &[&[(&str, u64, u32)]]) -> Vec<(String, StoredAddress)> {
        nodes
            .iter()
            .flat_map(|children| children.iter())
            .map(|&(name, address, _)| (String::from(name), StoredAddress::new(address)))
            .collect()
    }

    fn read_both(
        group: &Group,
        widths: Widths,
        base: u64,
        group_leaf_node_k: Option<NonZeroU16>,
    ) -> [Result<Vec<(String, StoredAddress)>, FormatError>; 2] {
        let mut file = vec![0; usize::try_from(base).unwrap()];
        file.extend_from_slice(&group.bytes);
        let widths = format_widths(widths);
        [
            resolve_v1_group_entries(
                &file,
                &group.message,
                widths,
                addresses(base),
                group_leaf_node_k,
            )
            .map(links),
            resolve_v1_group_entries_from_source(
                &PlacedSource::new(base, group.bytes.clone()),
                &group.message,
                widths,
                addresses(base),
                group_leaf_node_k,
            )
            .map(links),
        ]
    }

    #[rstest]
    #[case(3)]
    #[case(u32::MAX)]
    fn a_lookup_rejects_an_unknown_symbol_table_cache_type(#[case] cache_type: u32) {
        let group = group(
            &[&[("child", 0x100, cache_type)]],
            Widths::EIGHT,
            DataSegment::FollowsHeader,
        );
        let widths = format_widths(Widths::EIGHT);
        let error = find_link(
            group.bytes.as_slice(),
            &group.message,
            widths,
            addresses(0),
            k(16),
            LinkName::new("child").unwrap(),
            || {
                Ok(group
                    .nodes
                    .iter()
                    .map(|&(address, _)| StoredAddress::new(address))
                    .collect())
            },
        )
        .unwrap_err();
        assert_eq!(error, FormatError::InvalidSymbolTableCacheType(cache_type));
    }

    const CHILDREN: &[(&str, u64, u32)] = &[("alpha", 0x1000, 0), ("beta", 0x2000, 1)];

    const THREE_NODES: &[&[(&str, u64, u32)]] = &[
        &[
            ("a", 0x100, 0),
            ("b", 0x200, 0),
            ("c", 0x300, 1),
            ("d", 0x400, 0),
        ],
        &[("e", 0x500, 0), ("f", 0x600, 0)],
        &[("g", 0x700, 1), ("h", 0x800, 0), ("i", 0x900, 0)],
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
        #[values(&[CHILDREN][..], THREE_NODES)] nodes: &[&[(&str, u64, u32)]],
    ) {
        let group = group(nodes, widths, data_segment);
        let expected = links_of(nodes);
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
            &source,
            &group.message,
            format_widths(Widths::EIGHT),
            addresses(base),
            k(2),
        )
        .unwrap();
        assert_eq!(links(entries), links_of(THREE_NODES));
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
    #[case::at_capacity(k(2), Ok(links_of(THREE_NODES)))]
    #[case::above_capacity(
        k(1),
        Err(FormatError::InvalidSymbolTableNodeSymbolCount { symbols: 4, capacity: 2 })
    )]
    #[case::unknown_capacity(None, Ok(links_of(THREE_NODES)))]
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
                &source,
                &group.message,
                format_widths(Widths::EIGHT),
                addresses(512),
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

    #[test]
    fn a_symbolic_link_is_not_a_child() {
        let group = group(
            &[&[
                ("data", 0x1000, 0),
                ("pointer", u64::MAX, SYMBOLIC_LINK),
                ("group", 0x2000, 1),
            ]],
            Widths::EIGHT,
            DataSegment::FollowsHeader,
        );
        let expected = links_of(&[&[("data", 0x1000, 0), ("group", 0x2000, 1)]]);
        assert_eq!(
            read_both(&group, Widths::EIGHT, 0, k(2)),
            [Ok(expected.clone()), Ok(expected)]
        );
        assert!(
            v1_group_holds_symbolic_link_from_source(
                &PlacedSource::new(0, group.bytes),
                &group.message,
                format_widths(Widths::EIGHT),
                addresses(0),
                k(2),
            )
            .unwrap()
        );
    }

    // Helper to extract dataset components from an object header
    fn extract_dataset(
        _file_data: &[u8],
        hdr: &crate::object_header::ObjectHeader,
        offset_size: u8,
        length_size: u8,
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
        let ds = crate::dataspace::Dataspace::parse(ds_data, length_size).unwrap();
        let dl = crate::data_layout::DataLayout::parse(dl_data, offset_size, length_size).unwrap();
        (dt, ds, dl)
    }

    fn get_root_sym_table(
        file_data: &[u8],
        sb: &crate::superblock::Superblock,
    ) -> SymbolTableMessage {
        let root_header = ObjectHeader::parse(
            file_data,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(sb.base_address)
                .absolute(sb.root_group_address)
                .unwrap()
                .to_usize()
                .unwrap(),
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let sym_msg = root_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SYMBOL_TABLE)
            .unwrap();
        SymbolTableMessage::parse(&sym_msg.data, sb.offset_size).unwrap()
    }

    // Integration tests with real HDF5 files

    #[test]
    fn integration_simple_dataset_full_traversal() {
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/simple_dataset.h5");
        let sb = crate::superblock::read(file_data).unwrap().superblock;
        let root_sym = get_root_sym_table(file_data, &sb);

        let entries = resolve_v1_group_entries(
            file_data,
            &root_sym,
            FormatWidths::from_sizes(sb.offset_size, sb.length_size).unwrap(),
            AddressContext::from_effective_base(BaseAddress::ZERO),
            group_leaf_node_k(&sb),
        )
        .unwrap();
        let data_entry = entries
            .iter()
            .find(|e| e.name == "data")
            .expect("should have 'data'");

        let hdr = ObjectHeader::parse(
            file_data,
            AccessMode::ReadOnly,
            data_entry.object_header_address.get() as usize,
            sb.offset_size,
            sb.length_size,
        )
        .unwrap();
        let (dt, ds, dl) = extract_dataset(file_data, &hdr, sb.offset_size, sb.length_size);
        let raw = crate::data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
        let values = crate::data_read::read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
