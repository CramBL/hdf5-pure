//! The symbol table message and the symbol table node, the structures of a group that stores its
//! links in a symbol table.
//!
//! The message in the group's object header holds the addresses of a version 1 B-tree and of a
//! local heap. The leaves of the B-tree point to symbol table nodes, and each entry of a node
//! holds the offset of a link name in the heap and the address of the object the link points to.
//!
//! [`SymbolTableNodeLocator`] parses a node's prefix from the region of the file the caller reads,
//! and the [`SymbolTableEntriesRef`] it returns parses the entries in use from a second region.

use alloc::vec::Vec;
use core::num::NonZeroU16;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::bytes::UnexpectedEof;
use crate::codec::Decode;
use crate::error::FormatError;
use crate::width::{FormatWidths, OffsetWidth};

/// A symbol table message (type 0x0011), which locates the B-tree and the local heap of a group
/// that stores its links in a symbol table.
///
/// The message is defined in "The Symbol Table Message" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_stmgroup
#[derive(Clone, Debug, PartialEq)]
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
        Self::parse_with_offset_width(data, OffsetWidth::try_from(offset_size)?)
    }

    /// Returns the symbol table message in `data` using a validated address width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than two addresses.
    #[inline]
    pub fn parse_with_offset_width(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<SymbolTableMessage, FormatError> {
        Self::decode(data, offset_width)
    }

    /// Writes both addresses as little-endian integers and returns the number of bytes written.
    ///
    /// Each address occupies `offset_width` bytes. Preserves stored address bits, including
    /// zero and the selected width's all-ones undefined address. Any output suffix remains
    /// unchanged, and an error leaves the entire output unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if either address exceeds the largest unsigned
    /// integer representable in `offset_width` bytes, checking the B-tree address first.
    /// Returns [`FormatError::UnexpectedEof`] if `output` is shorter than two addresses.
    /// Address errors take precedence over destination capacity errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::SymbolTableMessage;
    /// use hdf5_pure_format::{FormatError, OffsetWidth};
    ///
    /// let message = SymbolTableMessage {
    ///     btree_address: StoredAddress::new(0x1234),
    ///     local_heap_address: StoredAddress::undefined(OffsetWidth::Two),
    /// };
    /// let mut output = [0xaa; 6];
    /// assert_eq!(message.write_into(&mut output, OffsetWidth::Two)?, 4);
    /// assert_eq!(output, [0x34, 0x12, 0xff, 0xff, 0xaa, 0xaa]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn write_into(
        &self,
        output: &mut [u8],
        offset_width: OffsetWidth,
    ) -> Result<usize, FormatError> {
        Ok(PreparedSymbolTable::new(self, offset_width)?.write_into(output)?)
    }

    /// Reads the B-tree and local heap addresses in wire order.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if an address is incomplete, relative to `record`.
    #[inline]
    fn read_fields(record: &[u8], offsets: OffsetWidth) -> Result<Self, UnexpectedEof> {
        let mut cursor = bytes::ByteCursor::new(record);
        Ok(match offsets {
            OffsetWidth::Two => Self {
                btree_address: StoredAddress::new(u64::from(u16::from_le_bytes(
                    *cursor.take_array()?,
                ))),
                local_heap_address: StoredAddress::new(u64::from(u16::from_le_bytes(
                    *cursor.take_array()?,
                ))),
            },
            OffsetWidth::Four => Self {
                btree_address: StoredAddress::new(u64::from(u32::from_le_bytes(
                    *cursor.take_array()?,
                ))),
                local_heap_address: StoredAddress::new(u64::from(u32::from_le_bytes(
                    *cursor.take_array()?,
                ))),
            },
            OffsetWidth::Eight => Self {
                btree_address: StoredAddress::new(u64::from_le_bytes(*cursor.take_array()?)),
                local_heap_address: StoredAddress::new(u64::from_le_bytes(*cursor.take_array()?)),
            },
        })
    }
}

impl<'input> Decode<'input> for SymbolTableMessage {
    type Context = OffsetWidth;
    type Error = FormatError;

    /// Reads the B-tree and local heap addresses from one complete little-endian record.
    ///
    /// Reads two addresses at `offsets` bytes each. Leaves any input suffix unexamined.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the record is incomplete, relative to `input`.
    fn decode(input: &'input [u8], offsets: OffsetWidth) -> Result<Self, Self::Error> {
        let mut cursor = bytes::ByteCursor::new(input);
        Ok(match offsets {
            OffsetWidth::Two => Self::read_fields(cursor.take_array::<4>()?, offsets)?,
            OffsetWidth::Four => Self::read_fields(cursor.take_array::<8>()?, offsets)?,
            OffsetWidth::Eight => Self::read_fields(cursor.take_array::<16>()?, offsets)?,
        })
    }
}

/// A borrowed message whose stored addresses fit the selected width.
///
/// The encoder reads the checked addresses through an immutable borrow.
struct PreparedSymbolTable<'a> {
    message: &'a SymbolTableMessage,
    offsets: OffsetWidth,
}

impl<'a> PreparedSymbolTable<'a> {
    /// Borrows a message after checking both addresses against `offsets`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if either address exceeds the width, checking the
    /// B-tree address first.
    fn new(message: &'a SymbolTableMessage, offsets: OffsetWidth) -> Result<Self, FormatError> {
        let SymbolTableMessage {
            btree_address,
            local_heap_address,
        } = message;
        let undefined = StoredAddress::undefined(offsets);
        for address in [*btree_address, *local_heap_address] {
            if address > undefined {
                return Err(FormatError::AddressTooLarge {
                    address: address.get(),
                    offset_size: offsets.get(),
                });
            }
        }
        Ok(Self { message, offsets })
    }

    /// Acquires both address destinations before writing at the checked width.
    ///
    /// Preserves the output suffix and returns the bytes written.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if `output` is shorter than two addresses, leaving
    /// the entire output unchanged.
    #[inline]
    fn write_into(self, output: &mut [u8]) -> Result<usize, UnexpectedEof> {
        let available = output.len();
        let Self { message, offsets } = self;
        let SymbolTableMessage {
            btree_address,
            local_heap_address,
        } = *message;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "both addresses fit the width retained by the prepared message"
        )]
        match offsets {
            OffsetWidth::Two => {
                let record = bytes::ByteCursorMut::new(output).take_array::<4>()?;
                let Some((btree_bytes, rest)) = record.split_first_chunk_mut::<2>() else {
                    return Err(UnexpectedEof::new(
                        2 * usize::from(offsets.get()),
                        available,
                    ));
                };
                let Some(heap_bytes) = rest.first_chunk_mut::<2>() else {
                    return Err(UnexpectedEof::new(
                        2 * usize::from(offsets.get()),
                        available,
                    ));
                };
                *btree_bytes = (btree_address.get() as u16).to_le_bytes();
                *heap_bytes = (local_heap_address.get() as u16).to_le_bytes();
            }
            OffsetWidth::Four => {
                let record = bytes::ByteCursorMut::new(output).take_array::<8>()?;
                let Some((btree_bytes, rest)) = record.split_first_chunk_mut::<4>() else {
                    return Err(UnexpectedEof::new(
                        2 * usize::from(offsets.get()),
                        available,
                    ));
                };
                let Some(heap_bytes) = rest.first_chunk_mut::<4>() else {
                    return Err(UnexpectedEof::new(
                        2 * usize::from(offsets.get()),
                        available,
                    ));
                };
                *btree_bytes = (btree_address.get() as u32).to_le_bytes();
                *heap_bytes = (local_heap_address.get() as u32).to_le_bytes();
            }
            OffsetWidth::Eight => {
                let record = bytes::ByteCursorMut::new(output).take_array::<16>()?;
                let Some((btree_bytes, rest)) = record.split_first_chunk_mut::<8>() else {
                    return Err(UnexpectedEof::new(
                        2 * usize::from(offsets.get()),
                        available,
                    ));
                };
                let Some(heap_bytes) = rest.first_chunk_mut::<8>() else {
                    return Err(UnexpectedEof::new(
                        2 * usize::from(offsets.get()),
                        available,
                    ));
                };
                *btree_bytes = btree_address.get().to_le_bytes();
                *heap_bytes = local_heap_address.get().to_le_bytes();
            }
        }
        Ok(2 * usize::from(offsets.get()))
    }
}

/// The address of a symbol table node, from a leaf of a group's version 1 B-tree.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range), the node's prefix, and passes
/// them to [`parse`](Self::parse), which returns the [`SymbolTableEntriesRef`] of the entries in
/// use. The node is defined in "Group Symbol Table Nodes" of the [format specification, version
/// 4.0][spec].
///
/// # Examples
///
/// ```
/// # use core::num::NonZeroU16;
/// # use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion};
/// # use hdf5_pure_core::PhysicalAddress;
/// # use hdf5_pure_core::__private::StoredAddressExt;
/// # use hdf5_pure_core::{BaseAddress, StoredAddress};
/// # use hdf5_pure_format::__private::{FormatWidths, SymbolTableNodeLocator};
/// # use hdf5_pure_format::FormatError;
/// # use test_util::symbol_table::{self, Entry};
/// # use test_util::widths::Widths;
/// # let bytes = symbol_table::node(&[Entry::new(0, 0x800), Entry::new(6, 0x900)], Widths::EIGHT);
/// # let range = FileRange::new(PhysicalAddress::new(512 + 0x400), bytes.len() as u64)?;
/// # let file = FileRegion::new(range, &bytes)?;
/// let widths = FormatWidths::from_sizes(8, 8)?;
/// let k = NonZeroU16::new(4);
/// let locator =
///     SymbolTableNodeLocator::new(widths, BaseAddress::new(512), StoredAddress::new(0x400), k);
/// let entries = locator.parse(file.subregion(locator.parse_range()?)?)?;
/// let node = entries.parse(file.subregion(entries.parse_range())?)?;
///
/// let offsets: Vec<u64> = node.entries.iter().map(|entry| entry.link_name_offset).collect();
/// assert_eq!(offsets, [0, 6]);
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltable
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymbolTableNodeLocator {
    widths: FormatWidths,
    base: BaseAddress,
    address: StoredAddress,
    group_leaf_node_k: Option<NonZeroU16>,
}

impl SymbolTableNodeLocator {
    /// Creates the locator of the node at the base-relative `address`.
    ///
    /// `group_leaf_node_k` is the "Group Leaf Node K" a version 0 or 1 superblock stores.
    /// [`parse`](Self::parse) checks the node's symbol count against twice its value, and skips
    /// the check where it is `None`.
    pub const fn new(
        widths: FormatWidths,
        base: BaseAddress,
        address: StoredAddress,
        group_leaf_node_k: Option<NonZeroU16>,
    ) -> Self {
        Self {
            widths,
            base,
            address,
            group_leaf_node_k,
        }
    }

    /// Parses the node prefix in `region` and returns the reference to the entries in use.
    ///
    /// The parser ignores the reserved byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidSymbolTableNodeSignature`] if the
    /// bytes do not start with `SNOD`, [`FormatError::InvalidSymbolTableNodeVersion`] if the
    /// version is not 1, and [`FormatError::InvalidSymbolTableNodeSymbolCount`] if the node stores
    /// more than twice `group_leaf_node_k` symbols. Returns [`FormatError::OffsetOverflow`] if the
    /// base address plus the node's address, or the end of the prefix or of the entries in use, is
    /// past [`u64::MAX`].
    pub fn parse(&self, region: FileRegion<'_>) -> Result<SymbolTableEntriesRef, FormatError> {
        let Self {
            widths,
            base: _,
            address: _,
            group_leaf_node_k,
        } = *self;
        let range = self.parse_range()?;
        let mut cursor = bytes::ByteCursor::new(region.subregion(range)?.bytes());
        if cursor.array::<{ NODE_SIGNATURE.len() }>()? != NODE_SIGNATURE {
            return Err(FormatError::InvalidSymbolTableNodeSignature);
        }
        let version = cursor.u8()?;
        if version != NODE_VERSION {
            return Err(FormatError::InvalidSymbolTableNodeVersion(version));
        }
        cursor.u8()?;
        let symbols = cursor.u16()?;
        if let Some(capacity) = group_leaf_node_k.map(|k| 2 * u32::from(k.get()))
            && u32::from(symbols) > capacity
        {
            return Err(FormatError::InvalidSymbolTableNodeSymbolCount { symbols, capacity });
        }
        Ok(SymbolTableEntriesRef {
            range: FileRange::new(range.end(), u64::from(symbols) * entry_len(widths))?,
            widths,
            symbols,
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the node's prefix: the
    /// signature, the version, a reserved byte, and the number of symbols.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the base address plus the node's address, or the
    /// end of the prefix, is past [`u64::MAX`].
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        let Self {
            widths: _,
            base,
            address,
            group_leaf_node_k: _,
        } = *self;
        FileRange::new(base.resolve(address)?, NODE_PREFIX_LEN)
    }
}

/// The range of the entries a symbol table node holds in use, from
/// [`SymbolTableNodeLocator::parse`].
///
/// Every node has room for twice the superblock's "Group Leaf Node K" entries, and the range spans
/// the ones in use, as many as the node's "Number of Symbols" field. A caller reads the bytes of
/// [`parse_range`](Self::parse_range) and passes them to [`parse`](Self::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymbolTableEntriesRef {
    range: FileRange,
    widths: FormatWidths,
    symbols: u16,
}

impl SymbolTableEntriesRef {
    /// Parses the entries in use from `region` and returns the node.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range).
    pub fn parse(&self, region: FileRegion<'_>) -> Result<SymbolTableNode, FormatError> {
        let Self {
            range,
            widths,
            symbols,
        } = *self;
        let mut cursor = bytes::ByteCursor::new(region.subregion(range)?.bytes());
        let mut entries = Vec::with_capacity(usize::from(symbols));
        for _ in 0..symbols {
            let link_name_offset = cursor.length(widths.lengths)?;
            let object_header_address = cursor.address(widths.offsets)?;
            let cache_type = cursor.u32()?;
            cursor.array::<ENTRY_RESERVED_LEN>()?;
            let scratch_pad = cursor.array::<SCRATCH_PAD_LEN>()?;
            entries.push(SymbolTableEntry {
                link_name_offset,
                object_header_address,
                cache_type,
                scratch_pad,
            });
        }
        Ok(SymbolTableNode { entries })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the entries in use.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
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

/// One entry of a symbol table node: the offset of a link name, the address of the object the
/// link points to, and what the entry caches.
///
/// The entry is defined in "Symbol Table Entry" of the [format specification, version 4.0][spec].
/// The specification gives the link name offset the width of an address, and
/// [`SymbolTableEntriesRef::parse`] reads it at the width of a length. `H5G_ent_decode` reads it
/// the same way (`H5Gent.c`, HDF5 2.2.0).
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

fn entry_len(FormatWidths { offsets, lengths }: FormatWidths) -> u64 {
    u64::from(lengths.get()) + u64::from(offsets.get()) + ENTRY_FIXED_LEN
}

const NODE_SIGNATURE: [u8; 4] = *b"SNOD";

const NODE_VERSION: u8 = 1;

const NODE_PREFIX_LEN: u64 = 8;

const ENTRY_RESERVED_LEN: usize = 4;

const SCRATCH_PAD_LEN: usize = 16;

const ENTRY_FIXED_LEN: u64 = (4 + ENTRY_RESERVED_LEN + SCRATCH_PAD_LEN) as u64;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::symbol_table;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::StoredAddressExt;
    use crate::fixture;

    fn locator(widths: Widths, base: u64, address: u64, k: Option<u16>) -> SymbolTableNodeLocator {
        SymbolTableNodeLocator::new(
            fixture::format_widths(widths),
            BaseAddress::new(base),
            StoredAddress::new(address),
            k.map(|k| NonZeroU16::new(k).unwrap()),
        )
    }

    fn entries(n: u64) -> Vec<symbol_table::Entry> {
        (0..n)
            .map(|i| symbol_table::Entry {
                cache_type: u32::try_from(i % 3).unwrap(),
                scratch_pad: [u8::try_from(i).unwrap(); symbol_table::SCRATCH_PAD],
                ..symbol_table::Entry::new(8 * i, 0x100 + 0x40 * i)
            })
            .collect()
    }

    fn expected(n: u64) -> SymbolTableNode {
        SymbolTableNode {
            entries: (0..n)
                .map(|i| SymbolTableEntry {
                    link_name_offset: 8 * i,
                    object_header_address: StoredAddress::new(0x100 + 0x40 * i),
                    cache_type: u32::try_from(i % 3).unwrap(),
                    scratch_pad: [u8::try_from(i).unwrap(); 16],
                })
                .collect(),
        }
    }

    fn parse(locator: SymbolTableNodeLocator, node: &[u8]) -> Result<SymbolTableNode, FormatError> {
        let start = locator.parse_range()?.start().get();
        let file = fixture::region(start, node);
        let entries = locator.parse(file)?;
        entries.parse(file)
    }

    #[rstest]
    fn a_node_is_parsed_from_its_prefix_and_then_its_entries_in_use(
        #[values(
            Widths::new(2, 2),
            Widths::new(2, 8),
            Widths::FOUR,
            Widths::new(8, 4),
            Widths::EIGHT
        )]
        widths: Widths,
        #[values(0, 512, 5 << 32)] base: u64,
        #[values(0, 1, 8)] n: u64,
    ) {
        let node = symbol_table::node(&entries(n), widths);
        let locator = locator(widths, base, 0x200, Some(4));
        let prefix = locator.parse_range().unwrap();
        assert_eq!((prefix.start().get(), prefix.length()), (base + 0x200, 8));
        let entries = locator
            .parse(fixture::region(base + 0x200, &node[..8]))
            .unwrap();
        let range = entries.parse_range();
        let entry_len = (widths.offset + widths.length + 24) as u64;
        assert_eq!(
            (range.start().get(), range.length()),
            (base + 0x208, n * entry_len)
        );
        assert_eq!(
            entries
                .parse(fixture::region(base + 0x208, &node[8..]))
                .unwrap(),
            expected(n)
        );
    }

    #[rstest]
    #[case::signature(0, b'X', FormatError::InvalidSymbolTableNodeSignature)]
    #[case::version(4, 2, FormatError::InvalidSymbolTableNodeVersion(2))]
    fn a_malformed_node_is_rejected(
        #[case] at: usize,
        #[case] byte: u8,
        #[case] expected: FormatError,
    ) {
        let mut node = symbol_table::node(&entries(2), Widths::EIGHT);
        node[at] = byte;
        assert_eq!(
            parse(locator(Widths::EIGHT, 0, 0, Some(4)), &node).unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::at_capacity(Some(2), 4, Ok(expected(4)))]
    #[case::above_capacity(
        Some(2),
        5,
        Err(FormatError::InvalidSymbolTableNodeSymbolCount { symbols: 5, capacity: 4 })
    )]
    #[case::capacity_past_u16_max(Some(u16::MAX), 9, Ok(expected(9)))]
    #[case::unknown_capacity(None, 9, Ok(expected(9)))]
    fn the_symbol_count_is_at_most_twice_the_group_leaf_node_k(
        #[case] k: Option<u16>,
        #[case] n: u64,
        #[case] expected: Result<SymbolTableNode, FormatError>,
    ) {
        let node = symbol_table::node(&entries(n), Widths::EIGHT);
        assert_eq!(parse(locator(Widths::EIGHT, 0, 0, k), &node), expected);
    }

    #[test]
    fn a_region_short_of_the_prefix_is_rejected() {
        let node = symbol_table::node(&entries(2), Widths::EIGHT);
        assert_eq!(
            locator(Widths::EIGHT, 512, 0x40, Some(4))
                .parse(fixture::region(512 + 0x40, &node[..7]))
                .unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 512 + 0x40,
                required_length: 8,
                provided_start: 512 + 0x40,
                provided_length: 7,
            }
        );
    }

    #[test]
    fn a_region_short_of_the_entries_in_use_is_rejected() {
        let node = symbol_table::node(&entries(2), Widths::EIGHT);
        let entries = locator(Widths::EIGHT, 0, 0x40, Some(4))
            .parse(fixture::region(0x40, &node))
            .unwrap();
        assert_eq!(
            entries
                .parse(fixture::region(0x48, &node[8..node.len() - 1]))
                .unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 0x48,
                required_length: 80,
                provided_start: 0x48,
                provided_length: 79,
            }
        );
    }

    #[rstest]
    #[case::base(512, u64::MAX - 8, FormatError::OffsetOverflow { offset: u64::MAX - 8, length: 512 })]
    #[case::prefix_end(0, u64::MAX - 4, FormatError::OffsetOverflow { offset: u64::MAX - 4, length: 8 })]
    fn a_prefix_past_u64_max_is_rejected(
        #[case] base: u64,
        #[case] address: u64,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            locator(Widths::EIGHT, base, address, Some(4))
                .parse_range()
                .unwrap_err(),
            expected
        );
    }

    #[test]
    fn entries_past_u64_max_are_rejected() {
        let node = symbol_table::node(&entries(1), Widths::EIGHT);
        assert_eq!(
            locator(Widths::EIGHT, 0, u64::MAX - 8, Some(4))
                .parse(fixture::region(u64::MAX - 8, &node[..8]))
                .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 40,
            }
        );
    }
}
