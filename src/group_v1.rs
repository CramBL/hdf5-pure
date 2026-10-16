//! V1 group traversal: resolve group children and navigate paths.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, StoredAddress};
#[cfg(test)]
use crate::address::{BaseAddress, BaseAddressExt};
use crate::btree_v1::{collect_symbol_table_nodes, collect_symbol_table_nodes_from_source};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::local_heap::LocalHeap;
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
    let heap = LocalHeap::parse_with_widths(
        file_data,
        addresses
            .absolute(sym_table_msg.local_heap_address)?
            .to_usize()?,
        widths,
    )?;

    let snod_addrs =
        collect_symbol_table_nodes(file_data, sym_table_msg.btree_address, widths, addresses)?;

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
            let name = heap.read_string(
                file_data,
                addresses.effective_base(),
                entry.link_name_offset,
            )?;
            entries.push(GroupEntry {
                name,
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
    let heap_addr = addresses.absolute(sym_table_msg.local_heap_address)?;
    let heap =
        LocalHeap::parse_from_source_with_widths(&SourceMetadata(source), heap_addr, widths)?;

    // Read the heap data segment once; every link name is sliced from it.
    let segment = source.read_metadata_at(
        addresses.absolute(heap.data_segment_address)?,
        heap.data_segment_size.to_usize()?,
    )?;

    let snod_addrs = collect_symbol_table_nodes_from_source(
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
            let name = heap.read_string_in_segment(&segment, entry.link_name_offset)?;
            entries.push(GroupEntry {
                name,
                object_header_address: entry.object_header_address,
                cache_type: entry.cache_type,
            });
        }
    }

    Ok(entries)
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
    use super::*;
    use crate::access_mode::AccessMode;
    use crate::address::AddressContext;
    use crate::message_type::MessageType;
    use crate::object_header::ObjectHeader;
    use crate::source::BytesSource;
    use test_util::image::Image;
    use test_util::widths::Widths;
    use test_util::{btree_v1, local_heap, symbol_table};

    /// A synthetic file holding one version 1 group: a local heap of the
    /// children's names, one symbol table node of their entries, and a B-tree
    /// leaf whose single child is that node.
    fn build_synthetic_group(
        children: &[(&str, u64, u32)],
        widths: Widths,
    ) -> (Vec<u8>, SymbolTableMessage) {
        let names: Vec<&str> = children.iter().map(|&(name, _, _)| name).collect();
        let mut image = Image::new();

        // The heap header sits at offset zero, so the group's message can
        // state a known address, and its data segment follows it.
        let segment_at = local_heap::header_len(widths);
        let segment = local_heap::Segment::of_names(segment_at as u64, &names);
        image.place(0, &segment.header(widths));
        image.place(segment_at, &segment.bytes);

        let entries: Vec<_> = children
            .iter()
            .enumerate()
            .map(
                |(index, &(_, header_address, cache_type))| symbol_table::Entry {
                    cache_type,
                    ..symbol_table::Entry::new(segment.offset_of(index), header_address)
                },
            )
            .collect();
        let node_at = image.append_aligned(&symbol_table::node(&entries, widths), 8);

        // One leaf entry, so two keys: the first name's heap offset and the
        // offset just past the last.
        let last_key = if children.is_empty() {
            0
        } else {
            segment.bytes.len() as u64
        };
        let keys = [
            btree_v1::group_key(0, widths),
            btree_v1::group_key(last_key, widths),
        ];
        let btree_at = image.append_aligned(
            &btree_v1::node(
                btree_v1::NodeType::GROUP,
                0,
                &keys,
                &[node_at as u64],
                widths,
            ),
            8,
        );

        // Room past the last structure for the tests that read off its end.
        image.append(&[0; 64]);

        let msg = SymbolTableMessage {
            btree_address: StoredAddress::new(btree_at as u64),
            local_heap_address: StoredAddress::new(0),
        };

        (image.build(), msg)
    }

    /// A symbolic link's entry carries the undefined address. Listing it as a
    /// child would send the walk to an object header that is not even there.
    #[test]
    fn a_symbolic_link_is_not_a_child() {
        let (file, msg) = build_synthetic_group(
            &[
                ("data", 0x1000, 0),
                ("pointer", u64::MAX, SYMBOLIC_LINK),
                ("group", 0x2000, 1),
            ],
            Widths::EIGHT,
        );
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        let names = |entries: Vec<GroupEntry>| -> Vec<String> {
            entries.into_iter().map(|entry| entry.name).collect()
        };
        let in_memory = resolve_v1_group_entries(&file, &msg, widths, addresses).unwrap();
        assert_eq!(names(in_memory), ["data", "group"]);
        let source = BytesSource::new(&file[..]);
        let streamed =
            resolve_v1_group_entries_from_source(&source, &msg, widths, addresses).unwrap();
        assert_eq!(names(streamed), ["data", "group"]);
        assert!(
            v1_group_holds_symbolic_link_from_source(&source, &msg, widths, addresses).unwrap()
        );
    }

    #[test]
    fn resolve_entries_two_children() {
        let (file, msg) =
            build_synthetic_group(&[("alpha", 0x1000, 0), ("beta", 0x2000, 0)], Widths::EIGHT);
        let entries = resolve_v1_group_entries(
            &file,
            &msg,
            FormatWidths::from_sizes(8, 8).unwrap(),
            AddressContext::from_effective_base(BaseAddress::ZERO),
        )
        .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "alpha");
        assert_eq!(entries[0].object_header_address, StoredAddress::new(0x1000));
        assert_eq!(entries[1].name, "beta");
        assert_eq!(entries[1].object_header_address, StoredAddress::new(0x2000));
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
        let sig_offset = crate::signature::find_signature(file_data).unwrap();
        let sb = crate::superblock::parse_superblock(file_data, sig_offset).unwrap();
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
