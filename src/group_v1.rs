//! V1 group traversal: resolve group children and navigate paths.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::LocalHeap;
use hdf5_pure_format::__private::LocalHeapLocator;

#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, StoredAddress};
#[cfg(test)]
use crate::address::{BaseAddress, BaseAddressExt};
use crate::btree_v1;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::RangeReader;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::symbol_table::{SymbolTableMessage, SymbolTableNode};
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
/// over, as a version 2 group's soft links are.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first group structure that does not parse.
pub fn resolve_v1_group_entries(
    file_data: &[u8],
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
) -> Result<Vec<GroupEntry>, FormatError> {
    let heap = read_local_heap(file_data, sym_table_msg, widths, addresses)?;
    let range = heap.data_segment_range();
    let bytes = file_data.read_range(range)?;
    let names = heap.data_segment(FileRegion::new(range, &bytes)?)?;

    let snod_addrs = btree_v1::collect_symbol_table_nodes(
        file_data,
        sym_table_msg.btree_address,
        widths,
        addresses,
    )?;

    let mut entries = Vec::new();
    for snod_addr in snod_addrs {
        let snod = SymbolTableNode::parse_with_widths(
            file_data,
            addresses.absolute(snod_addr)?.to_usize()?,
            widths,
        )?;
        for entry in snod
            .entries
            .iter()
            .filter(|entry| entry.cache_type != SYMBOLIC_LINK)
        {
            entries.push(GroupEntry {
                name: String::from(names.name(entry.link_name_offset)?),
                object_header_address: entry.object_header_address,
                cache_type: entry.cache_type,
            });
        }
    }

    Ok(entries)
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
) -> Result<Vec<GroupEntry>, FormatError> {
    let mut entries = v1_symbols_from_source(source, sym_table_msg, widths, addresses)?;
    entries.retain(|entry| entry.cache_type != SYMBOLIC_LINK);
    Ok(entries)
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
) -> Result<bool, FormatError> {
    Ok(
        v1_symbols_from_source(source, sym_table_msg, widths, addresses)?
            .iter()
            .any(|entry| entry.cache_type == SYMBOLIC_LINK),
    )
}

/// Returns every entry of the version 1 group `sym_table_msg` describes, symbolic links included.
fn v1_symbols_from_source<S: Source + ?Sized>(
    source: &S,
    sym_table_msg: &SymbolTableMessage,
    widths: FormatWidths,
    addresses: AddressContext,
) -> Result<Vec<GroupEntry>, FormatError> {
    let heap = read_local_heap(source, sym_table_msg, widths, addresses)?;
    let range = heap.data_segment_range();
    let bytes = source.read_range(range)?;
    let names = heap.data_segment(FileRegion::new(range, &bytes)?)?;

    let snod_addrs = btree_v1::collect_symbol_table_nodes_from_source(
        source,
        sym_table_msg.btree_address,
        widths,
        addresses,
    )?;

    let mut entries = Vec::new();
    for snod_addr in snod_addrs {
        let snod_offset = addresses.absolute(snod_addr)?;
        let snod = SymbolTableNode::parse_from_source_with_widths(
            &SourceMetadata(source),
            snod_offset,
            widths,
        )?;
        for entry in &snod.entries {
            entries.push(GroupEntry {
                name: String::from(names.name(entry.link_name_offset)?),
                object_header_address: entry.object_header_address,
                cache_type: entry.cache_type,
            });
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
        data_segment: (u64, usize),
        reads: Vec<(u64, usize)>,
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

        let mut name_offsets = (0..names.len()).map(|index| segment.offset_of(index));
        let mut node_reads = Vec::new();
        let node_addresses: Vec<u64> = nodes
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
                let at = image.append_aligned(&node, 8) as u64;
                node_reads.extend([(at, 8), (at, node.len())]);
                at
            })
            .collect();
        let btree = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            0,
            &vec![0; node_addresses.len() + 1],
            &node_addresses,
            widths,
        );
        let btree_at = image.append_aligned(&btree, 8) as u64;

        let segment_at = match data_segment {
            DataSegment::FollowsHeader => header_len,
            DataSegment::AtTheEnd => image.append_aligned(&segment.bytes, 8),
        } as u64;
        image.place(
            0,
            &local_heap::Segment::of_names(segment_at, &names).header(widths),
        );
        let mut reads = vec![
            (0, header_len),
            (segment_at, segment.bytes.len()),
            (btree_at, 8),
            (btree_at, btree.len()),
        ];
        reads.extend(node_reads);
        Group {
            bytes: image.build(),
            message: SymbolTableMessage {
                btree_address: StoredAddress::new(btree_at),
                local_heap_address: StoredAddress::ZERO,
            },
            data_segment: (segment_at, segment.bytes.len()),
            reads,
        }
    }

    fn in_file(base: u64, bytes: &[u8]) -> Vec<u8> {
        let mut file = vec![0; usize::try_from(base).unwrap()];
        file.extend_from_slice(bytes);
        file
    }

    fn format_widths(Widths { offset, length }: Widths) -> FormatWidths {
        FormatWidths::from_sizes(offset.try_into().unwrap(), length.try_into().unwrap()).unwrap()
    }

    fn addresses(base: u64) -> AddressContext {
        AddressContext::from_effective_base(BaseAddress::new(base))
    }

    fn links(entries: Vec<GroupEntry>) -> Vec<(String, StoredAddress)> {
        entries
            .into_iter()
            .map(|entry| (entry.name, entry.object_header_address))
            .collect()
    }

    const CHILDREN: &[(&str, u64, u32)] = &[("alpha", 0x1000, 0), ("beta", 0x2000, 1)];

    fn expected_links() -> Vec<(String, StoredAddress)> {
        CHILDREN
            .iter()
            .map(|&(name, address, _)| (String::from(name), StoredAddress::new(address)))
            .collect()
    }

    #[rstest]
    fn a_buffered_read_lists_every_child(
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
    ) {
        let group = group(&[CHILDREN], widths, data_segment);
        let entries = resolve_v1_group_entries(
            &in_file(base, &group.bytes),
            &group.message,
            format_widths(widths),
            addresses(base),
        )
        .unwrap();
        assert_eq!(links(entries), expected_links());
    }

    #[rstest]
    fn a_streamed_read_reads_the_heap_header_and_its_data_segment_once(
        #[values(
            Widths::new(2, 2),
            Widths::new(2, 8),
            Widths::FOUR,
            Widths::new(8, 2),
            Widths::EIGHT
        )]
        widths: Widths,
        #[values(0, 512, 5 << 32)] base: u64,
        #[values(DataSegment::FollowsHeader, DataSegment::AtTheEnd)] data_segment: DataSegment,
    ) {
        let group = group(&[CHILDREN], widths, data_segment);
        let source = PlacedSource::new(base, group.bytes);
        let entries = resolve_v1_group_entries_from_source(
            &source,
            &group.message,
            format_widths(widths),
            addresses(base),
        )
        .unwrap();
        assert_eq!(links(entries), expected_links());
        assert_eq!(
            source.reads.into_inner(),
            group
                .reads
                .iter()
                .map(|&(at, len)| (base + at, len))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_data_segment_past_the_end_of_the_file_is_rejected() {
        let mut group = group(&[CHILDREN], Widths::EIGHT, DataSegment::AtTheEnd);
        let (at, len) = group.data_segment;
        group.bytes.truncate(usize::try_from(at).unwrap() + len - 1);
        let expected = FormatError::UnexpectedEof {
            expected: usize::try_from(at).unwrap() + len,
            available: group.bytes.len(),
        };
        let widths = format_widths(Widths::EIGHT);
        assert_eq!(
            resolve_v1_group_entries(&group.bytes, &group.message, widths, addresses(0))
                .unwrap_err(),
            expected
        );
        assert_eq!(
            resolve_v1_group_entries_from_source(
                &PlacedSource::new(0, group.bytes),
                &group.message,
                widths,
                addresses(0)
            )
            .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::signature(0, b'X', FormatError::InvalidLocalHeapSignature)]
    #[case::version(4, 1, FormatError::InvalidLocalHeapVersion(1))]
    fn a_malformed_local_heap_is_rejected(
        #[case] at: usize,
        #[case] byte: u8,
        #[case] expected: FormatError,
    ) {
        let mut group = group(&[CHILDREN], Widths::EIGHT, DataSegment::FollowsHeader);
        group.bytes[at] = byte;
        let widths = format_widths(Widths::EIGHT);
        assert_eq!(
            resolve_v1_group_entries(&group.bytes, &group.message, widths, addresses(0))
                .unwrap_err(),
            expected
        );
        assert_eq!(
            resolve_v1_group_entries_from_source(
                &PlacedSource::new(0, group.bytes),
                &group.message,
                widths,
                addresses(0)
            )
            .unwrap_err(),
            expected
        );
    }

    #[test]
    fn a_name_offset_past_the_data_segment_is_rejected() {
        let mut group = group(&[CHILDREN], Widths::EIGHT, DataSegment::FollowsHeader);
        let (_, len) = group.data_segment;
        let node_at = usize::try_from(group.reads[4].0).unwrap();
        test_util::bytes::set_uint_at(&mut group.bytes, node_at + 8, 8, len as u64);
        let expected = FormatError::InvalidLocalHeapNameOffset {
            offset: len as u64,
            data_segment_size: len as u64,
        };
        let widths = format_widths(Widths::EIGHT);
        assert_eq!(
            resolve_v1_group_entries(&group.bytes, &group.message, widths, addresses(0))
                .unwrap_err(),
            expected
        );
        assert_eq!(
            resolve_v1_group_entries_from_source(
                &PlacedSource::new(0, group.bytes),
                &group.message,
                widths,
                addresses(0)
            )
            .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::header(0)]
    #[case::data_segment(1)]
    fn a_failed_read_reports_the_source_error(#[case] failing_read: usize) {
        let group = group(&[CHILDREN], Widths::EIGHT, DataSegment::AtTheEnd);
        let source = FailingSource {
            inner: PlacedSource::new(512, group.bytes),
            failing_offset: 512 + group.reads[failing_read].0,
        };
        assert_eq!(
            resolve_v1_group_entries_from_source(
                &source,
                &group.message,
                format_widths(Widths::EIGHT),
                addresses(512)
            )
            .unwrap_err(),
            FormatError::Source(String::from("injected read failure"))
        );
    }

    struct FailingSource {
        inner: PlacedSource,
        failing_offset: u64,
    }

    impl Source for FailingSource {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.inner.read_at(offset, buf)
        }

        fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            if offset == self.failing_offset {
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
        let widths = format_widths(Widths::EIGHT);
        let names = |entries: Vec<GroupEntry>| -> Vec<String> {
            entries.into_iter().map(|entry| entry.name).collect()
        };
        let in_memory =
            resolve_v1_group_entries(&group.bytes, &group.message, widths, addresses(0)).unwrap();
        assert_eq!(names(in_memory), ["data", "group"]);
        let source = PlacedSource::new(0, group.bytes);
        let streamed =
            resolve_v1_group_entries_from_source(&source, &group.message, widths, addresses(0))
                .unwrap();
        assert_eq!(names(streamed), ["data", "group"]);
        assert!(
            v1_group_holds_symbolic_link_from_source(&source, &group.message, widths, addresses(0))
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
