//! V1 group traversal: resolve group children and navigate paths.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::address::{BaseAddress, StoredAddress};
use crate::btree_v1::{collect_symbol_table_nodes, collect_symbol_table_nodes_from_source};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::local_heap::LocalHeap;
use crate::source::Source;
use crate::symbol_table::{SymbolTableMessage, SymbolTableNode};

/// A resolved group entry (child name + object header address).
#[derive(Debug, Clone)]
pub struct GroupEntry {
    /// Name of the child object.
    pub name: String,
    /// Address of the child's object header.
    pub object_header_address: StoredAddress,
    /// Cache type from the symbol table (SNOD) entry. Parsed for on-disk
    /// completeness; the reader does not currently act on it.
    #[allow(dead_code)]
    pub cache_type: u32,
}

/// Returns one entry per child of the version 1 group `sym_table_msg` describes.
///
/// The message names two structures: the local heap that holds the link names, and the B-tree
/// whose leaves lead to the symbol table nodes that hold the entries.
///
/// # Errors
///
/// Returns the [`FormatError`] of the first structure that does not parse: the local heap, the
/// B-tree, or one of the symbol table nodes.
pub fn resolve_v1_group_entries(
    file_data: &[u8],
    sym_table_msg: &SymbolTableMessage,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<GroupEntry>, FormatError> {
    let heap = LocalHeap::parse(
        file_data,
        base_address
            .absolute(sym_table_msg.local_heap_address)?
            .to_usize()?,
        offset_size,
        length_size,
    )?;

    let snod_addrs = collect_symbol_table_nodes(
        file_data,
        sym_table_msg.btree_address,
        offset_size,
        length_size,
        base_address,
    )?;

    let mut entries = Vec::new();
    for snod_addr in snod_addrs {
        let snod = SymbolTableNode::parse(
            file_data,
            base_address.absolute(snod_addr)?.to_usize()?,
            offset_size,
            length_size,
        )?;
        for entry in &snod.entries {
            let name = heap.read_string(file_data, base_address, entry.link_name_offset)?;
            entries.push(GroupEntry {
                name,
                object_header_address: entry.object_header_address,
                cache_type: entry.cache_type,
            });
        }
    }

    Ok(entries)
}

/// Streaming counterpart of [`resolve_v1_group_entries`].
///
/// Reads the local heap header, B-tree v1, and each symbol-table node from a
/// [`Source`] on demand. The heap's data segment, which holds the link names, is read once and
/// every name is sliced from that single buffer.
///
/// # Errors
///
/// Returns the errors of [`resolve_v1_group_entries`], and the error `source` reports for a
/// structure it cannot read.
pub fn resolve_v1_group_entries_from_source<S: Source + ?Sized>(
    source: &S,
    sym_table_msg: &SymbolTableMessage,
    offset_size: u8,
    length_size: u8,
    base_address: BaseAddress,
) -> Result<Vec<GroupEntry>, FormatError> {
    let heap_addr = base_address.absolute(sym_table_msg.local_heap_address)?;
    let heap = LocalHeap::parse_from_source(source, heap_addr, offset_size, length_size)?;

    // Read the heap data segment once; every link name is sliced from it.
    let segment = source.read_metadata_at(
        base_address.absolute(heap.data_segment_address)?,
        heap.data_segment_size.to_usize()?,
    )?;

    let snod_addrs = collect_symbol_table_nodes_from_source(
        source,
        sym_table_msg.btree_address,
        offset_size,
        length_size,
        base_address,
    )?;

    let mut entries = Vec::new();
    for snod_addr in snod_addrs {
        let snod_offset = base_address.absolute(snod_addr)?;
        let snod =
            SymbolTableNode::parse_from_source(source, snod_offset, offset_size, length_size)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_mode::AccessMode;
    use crate::message_type::MessageType;
    use crate::object_header::ObjectHeader;
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

    #[test]
    fn resolve_entries_two_children() {
        let (file, msg) =
            build_synthetic_group(&[("alpha", 0x1000, 0), ("beta", 0x2000, 0)], Widths::EIGHT);
        let entries = resolve_v1_group_entries(&file, &msg, 8, 8, BaseAddress::ZERO).unwrap();
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
        let (dt, _) = hdf5_pure_format::parse_datatype(dt_data).unwrap();
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
            sb.root_group_address as usize,
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
        let sb = crate::superblock::Superblock::parse(file_data, sig_offset).unwrap();
        let root_sym = get_root_sym_table(file_data, &sb);

        let entries = resolve_v1_group_entries(
            file_data,
            &root_sym,
            sb.offset_size,
            sb.length_size,
            BaseAddress::ZERO,
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
