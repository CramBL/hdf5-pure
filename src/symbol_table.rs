//! HDF5 Symbol Table Message and Symbol Table Node (SNOD) parsing.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bytes::read_length;
use crate::bytes::read_offset;
use crate::error::FormatError;
use crate::source::Source;

/// Symbol Table message (type 0x0011) found in v1 group object headers.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolTableMessage {
    /// Address of B-tree v1 (type 0) for this group.
    pub btree_address: u64,
    /// Address of the local heap for this group.
    pub local_heap_address: u64,
}

impl SymbolTableMessage {
    /// Parse a Symbol Table message from raw message data bytes.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<SymbolTableMessage, FormatError> {
        let os = offset_size as usize;
        if data.len() < os * 2 {
            return Err(FormatError::UnexpectedEof {
                expected: os * 2,
                available: data.len(),
            });
        }
        let btree_address = read_offset(data, 0, offset_size)?;
        let local_heap_address = read_offset(data, os, offset_size)?;
        Ok(SymbolTableMessage {
            btree_address,
            local_heap_address,
        })
    }
}

/// A single entry in a Symbol Table Node (SNOD).
#[derive(Debug, Clone)]
pub struct SymbolTableEntry {
    /// Byte offset of the link name in the local heap.
    pub link_name_offset: u64,
    /// Address of the child object's header.
    pub object_header_address: u64,
    /// Cache type: 0=none, 1=group, 2=symbolic link.
    pub cache_type: u32,
    /// 16-byte scratch pad (cached data). Parsed for on-disk-format
    /// completeness; not used by the reader.
    #[allow(dead_code)]
    pub scratch_pad: [u8; 16],
}

/// A parsed Symbol Table Node (SNOD).
#[derive(Debug, Clone)]
pub struct SymbolTableNode {
    /// The symbol table entries.
    pub entries: Vec<SymbolTableEntry>,
}

impl SymbolTableNode {
    /// Parse a Symbol Table Node at the given offset in the file data.
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
            let object_header_address = read_offset(file_data, pos, offset_size)?;
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

    /// Parse a Symbol Table Node from a [`Source`] on demand.
    ///
    /// Reads the 8-byte SNOD header to learn the symbol count, then the exact
    /// node body, so no more than one node is resident at a time.
    pub fn parse_from_source<S: Source + ?Sized>(
        source: &S,
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

    #[test]
    fn parse_symbol_table_message_offset8() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x1000u64.to_le_bytes()); // btree
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // heap
        let msg = SymbolTableMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.btree_address, 0x1000);
        assert_eq!(msg.local_heap_address, 0x2000);
    }

    #[test]
    fn parse_symbol_table_message_offset4() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x800u32.to_le_bytes());
        data.extend_from_slice(&0x900u32.to_le_bytes());
        let msg = SymbolTableMessage::parse(&data, 4).unwrap();
        assert_eq!(msg.btree_address, 0x800);
        assert_eq!(msg.local_heap_address, 0x900);
    }

    fn build_snod(entries: &[(u64, u64, u32)], offset_size: u8, length_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        // Pad so SNOD is at offset 0
        buf.extend_from_slice(b"SNOD");
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for &(name_off, ohdr_addr, cache_type) in entries {
            match length_size {
                4 => buf.extend_from_slice(&(name_off as u32).to_le_bytes()),
                8 => buf.extend_from_slice(&name_off.to_le_bytes()),
                _ => panic!("test length_size"),
            }
            match offset_size {
                4 => buf.extend_from_slice(&(ohdr_addr as u32).to_le_bytes()),
                8 => buf.extend_from_slice(&ohdr_addr.to_le_bytes()),
                _ => panic!("test offset_size"),
            }
            buf.extend_from_slice(&cache_type.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
            buf.extend_from_slice(&[0u8; 16]); // scratch pad
        }
        buf
    }

    #[test]
    fn parse_snod_two_entries() {
        let data = build_snod(&[(0, 0x100, 0), (8, 0x200, 1)], 8, 8);
        let snod = SymbolTableNode::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(snod.entries.len(), 2);
        assert_eq!(snod.entries[0].link_name_offset, 0);
        assert_eq!(snod.entries[0].object_header_address, 0x100);
        assert_eq!(snod.entries[0].cache_type, 0);
        assert_eq!(snod.entries[1].link_name_offset, 8);
        assert_eq!(snod.entries[1].object_header_address, 0x200);
        assert_eq!(snod.entries[1].cache_type, 1);
    }

    #[test]
    fn parse_snod_differing_offset_and_length_sizes() {
        let data = build_snod(&[(0x12345678, 0x100, 0)], 4, 8);
        let snod = SymbolTableNode::parse(&data, 0, 4, 8).unwrap();
        assert_eq!(snod.entries.len(), 1);
        assert_eq!(snod.entries[0].link_name_offset, 0x12345678);
        assert_eq!(snod.entries[0].object_header_address, 0x100);
    }

    #[test]
    fn parse_snod_empty() {
        let data = build_snod(&[], 8, 8);
        let snod = SymbolTableNode::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(snod.entries.len(), 0);
    }

    #[test]
    fn parse_snod_invalid_signature() {
        let mut data = build_snod(&[], 8, 8);
        data[0] = b'X';
        let err = SymbolTableNode::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSymbolTableNodeSignature);
    }

    #[test]
    fn parse_snod_invalid_version() {
        let mut data = build_snod(&[], 8, 8);
        data[4] = 2; // bad version
        let err = SymbolTableNode::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSymbolTableNodeVersion(2));
    }
}
