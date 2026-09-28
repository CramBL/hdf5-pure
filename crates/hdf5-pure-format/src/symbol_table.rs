//! The symbol table message and the symbol table node, the structures of a group that stores its
//! links in a symbol table.
//!
//! The message in the group's object header holds the addresses of a version 1 B-tree and of a
//! local heap. The leaves of the B-tree point to symbol table nodes, and each entry of a node
//! holds the offset of a link name in the heap and the address of the object the link points to.

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes::read_length;
use crate::bytes::read_offset;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;

/// A symbol table message (type 0x0011), which locates the B-tree and the local heap of a group
/// that stores its links in a symbol table.
///
/// The message is defined in "The Symbol Table Message" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_stmgroup
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolTableMessage {
    /// The address of the group's version 1 B-tree, of node type 0.
    pub btree_address: StoredAddress,
    /// The address of the local heap that holds the group's link names.
    pub local_heap_address: StoredAddress,
}

impl SymbolTableMessage {
    /// Parses a symbol table message from its body, `data`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than two addresses, and
    /// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<SymbolTableMessage, FormatError> {
        let os = offset_size as usize;
        if data.len() < os * 2 {
            return Err(FormatError::UnexpectedEof {
                expected: os * 2,
                available: data.len(),
            });
        }
        let btree_address = StoredAddress::new(read_offset(data, 0, offset_size)?);
        let local_heap_address = StoredAddress::new(read_offset(data, os, offset_size)?);
        Ok(SymbolTableMessage {
            btree_address,
            local_heap_address,
        })
    }
}

/// One entry of a symbol table node: the offset of a link name, the address of the object the
/// link points to, and what the entry caches.
///
/// The entry is defined in "Symbol Table Entry" of the [format specification, version 4.0][spec].
/// The specification gives the link name offset the width of an address, and
/// [`SymbolTableNode::parse`] reads it at the width of a length. `H5G_ent_decode` reads it the same
/// way (`H5Gent.c`, HDF5 2.2.0).
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolTableEntry {
    /// The offset of the link name in the data segment of the group's local heap.
    pub link_name_offset: u64,
    /// The address of the object header of the object the link points to, and the undefined
    /// address in the entry of a symbolic link.
    pub object_header_address: StoredAddress,
    /// What the scratch pad caches: 0 for nothing, 1 for the B-tree and local heap addresses of a
    /// group, and 2 for the local heap offset of a symbolic link's value.
    pub cache_type: u32,
    /// The 16-byte scratch-pad space, laid out as [`cache_type`](Self::cache_type) selects.
    pub scratch_pad: [u8; 16],
}

/// A symbol table node, which holds entries of a group that stores its links in a symbol table.
///
/// The leaves of the group's version 1 B-tree point to its symbol table nodes. The node is defined
/// in "Group Symbol Table Nodes" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltable
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolTableNode {
    /// The entries in use, as many as the node's "Number of Symbols" field.
    pub entries: Vec<SymbolTableEntry>,
}

impl SymbolTableNode {
    /// Parses the symbol table node at `offset` in `file_data`.
    ///
    /// Reads the entries in use. The entries past them hold undefined values and are not read.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the node header or the entries in use run past
    /// the end of `file_data`, [`FormatError::InvalidSymbolTableNodeSignature`] if the node does
    /// not begin with `SNOD`, [`FormatError::InvalidSymbolTableNodeVersion`] if its version is not
    /// 1, and [`FormatError::InvalidLengthSize`] or [`FormatError::InvalidOffsetSize`] if the node
    /// has an entry in use and a width is not 2, 4, or 8.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<SymbolTableNode, FormatError> {
        // signature(4) + version(1) + reserved(1) + number_of_symbols(2) = 8
        if 8 > file_data.len() || offset > file_data.len() - 8 {
            return Err(FormatError::UnexpectedEof {
                expected: offset.saturating_add(8),
                available: file_data.len(),
            });
        }

        if &file_data[offset..offset + 4] != b"SNOD" {
            return Err(FormatError::InvalidSymbolTableNodeSignature);
        }

        let version = file_data[offset + 4];
        if version != 1 {
            return Err(FormatError::InvalidSymbolTableNodeVersion(version));
        }

        let num_symbols =
            u16::from_le_bytes([file_data[offset + 6], file_data[offset + 7]]) as usize;

        let os = offset_size as usize;
        let ls = length_size as usize;
        // Each entry: link_name_offset(ls) + obj_hdr_addr(os) + cache_type(4) + reserved(4) + scratch(16)
        let entry_size = ls + os + 4 + 4 + 16;
        let entries_start = offset + 8;
        let needed = entries_start + num_symbols * entry_size;
        if needed > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: needed,
                available: file_data.len(),
            });
        }

        let mut entries = Vec::with_capacity(num_symbols);
        let mut pos = entries_start;
        for _ in 0..num_symbols {
            let link_name_offset = read_length(file_data, pos, length_size)?;
            pos += ls;
            let object_header_address =
                StoredAddress::new(read_offset(file_data, pos, offset_size)?);
            pos += os;
            let cache_type = u32::from_le_bytes([
                file_data[pos],
                file_data[pos + 1],
                file_data[pos + 2],
                file_data[pos + 3],
            ]);
            pos += 4;
            // reserved 4 bytes
            pos += 4;
            let mut scratch_pad = [0u8; 16];
            scratch_pad.copy_from_slice(&file_data[pos..pos + 16]);
            pos += 16;

            entries.push(SymbolTableEntry {
                link_name_offset,
                object_header_address,
                cache_type,
                scratch_pad,
            });
        }

        Ok(SymbolTableNode { entries })
    }

    /// Parses the symbol table node at `address` in `source`.
    ///
    /// Reads the 8-byte node header for the number of entries in use, then the header and those
    /// entries, and parses them as [`parse`](Self::parse) does.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
    pub fn parse_from_source(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<SymbolTableNode, FormatError> {
        let header = source.read_metadata_at(address, 8)?;
        if &header[0..4] != b"SNOD" {
            return Err(FormatError::InvalidSymbolTableNodeSignature);
        }
        let version = header[4];
        if version != 1 {
            return Err(FormatError::InvalidSymbolTableNodeVersion(version));
        }
        let num_symbols = u16::from_le_bytes([header[6], header[7]]) as usize;

        let os = offset_size as usize;
        let ls = length_size as usize;
        // link_name_offset(ls) + obj_hdr_addr(os) + cache_type(4) + reserved(4) + scratch(16)
        let entry_size = ls + os + 4 + 4 + 16;
        let total = 8 + num_symbols * entry_size;
        let buf = source.read_metadata_at(address, total)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_util::symbol_table;
    use test_util::widths::Widths;

    #[test]
    fn parse_symbol_table_message_offset8() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x1000u64.to_le_bytes()); // btree
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // heap
        let msg = SymbolTableMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.btree_address, StoredAddress::new(0x1000));
        assert_eq!(msg.local_heap_address, StoredAddress::new(0x2000));
    }

    #[test]
    fn parse_symbol_table_message_offset4() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x800u32.to_le_bytes());
        data.extend_from_slice(&0x900u32.to_le_bytes());
        let msg = SymbolTableMessage::parse(&data, 4).unwrap();
        assert_eq!(msg.btree_address, StoredAddress::new(0x800));
        assert_eq!(msg.local_heap_address, StoredAddress::new(0x900));
    }

    /// A node of `(link name offset, object header address, cache type)`
    /// entries, none of which caches anything in its scratch pad.
    fn build_snod(entries: &[(u64, u64, u32)], widths: Widths) -> Vec<u8> {
        let entries: Vec<_> = entries
            .iter()
            .map(
                |&(link_name_offset, header_address, cache_type)| symbol_table::Entry {
                    cache_type,
                    ..symbol_table::Entry::new(link_name_offset, header_address)
                },
            )
            .collect();
        symbol_table::node(&entries, widths)
    }

    #[test]
    fn parse_snod_two_entries() {
        let data = build_snod(&[(0, 0x100, 0), (8, 0x200, 1)], Widths::EIGHT);
        let snod = SymbolTableNode::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(snod.entries.len(), 2);
        assert_eq!(snod.entries[0].link_name_offset, 0);
        assert_eq!(
            snod.entries[0].object_header_address,
            StoredAddress::new(0x100)
        );
        assert_eq!(snod.entries[0].cache_type, 0);
        assert_eq!(snod.entries[1].link_name_offset, 8);
        assert_eq!(
            snod.entries[1].object_header_address,
            StoredAddress::new(0x200)
        );
        assert_eq!(snod.entries[1].cache_type, 1);
    }

    #[test]
    fn parse_snod_differing_offset_and_length_sizes() {
        let data = build_snod(&[(0x12345678, 0x100, 0)], Widths::new(4, 8));
        let snod = SymbolTableNode::parse(&data, 0, 4, 8).unwrap();
        assert_eq!(snod.entries.len(), 1);
        assert_eq!(snod.entries[0].link_name_offset, 0x12345678);
        assert_eq!(
            snod.entries[0].object_header_address,
            StoredAddress::new(0x100)
        );
    }

    #[test]
    fn parse_snod_empty() {
        let data = build_snod(&[], Widths::EIGHT);
        let snod = SymbolTableNode::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(snod.entries.len(), 0);
    }

    #[test]
    fn parse_snod_invalid_signature() {
        let mut data = build_snod(&[], Widths::EIGHT);
        data[0] = b'X';
        let err = SymbolTableNode::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSymbolTableNodeSignature);
    }

    #[test]
    fn parse_snod_invalid_version() {
        let mut data = build_snod(&[], Widths::EIGHT);
        data[4] = 2; // bad version
        let err = SymbolTableNode::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSymbolTableNodeVersion(2));
    }
}
