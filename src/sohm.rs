//! The reads of a file's shared messages: the records of an index, and the body of a message from
//! the fractal heap of the index that covers its type.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub use hdf5_pure_format::SharedMessageTableMessage;
pub use hdf5_pure_format::SohmIndexHeader;
use hdf5_pure_format::SohmIndexKind;
pub use hdf5_pure_format::SohmLocation;
pub use hdf5_pure_format::SohmRecord;
pub use hdf5_pure_format::SohmTable;

use crate::address::StoredAddress;
use crate::btree_v2::{BTreeV2Header, collect_btree_v2_records_from_source};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::HeapObjectReader;
use crate::message_type::MessageType;
use crate::shared_message::FHEAP_ID_LEN;
use crate::source::Source;
use crate::source::SourceMetadata;

/// B-tree v2 type of a shared-message index (`H5B2_SOHM_INDEX_ID`).
const BTREE_SOHM_INDEX_TYPE: u8 = 7;

/// Every record of `index`, whichever kind of index it is.
///
/// An index with no address holds nothing: the library allocates neither a list
/// nor a B-tree until the first message is shared.
///
/// Only a [`Source`] form exists. Reading a message needs the master table and
/// the heap alone — the fractal-heap ID in a reference locates it directly — so
/// the records have one caller, the edit engine's screen, and it reads the file
/// through a `Source` whichever backing it has.
pub fn read_index_records_from_source<S: Source + ?Sized>(
    source: &S,
    index: &SohmIndexHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<SohmRecord>, FormatError> {
    let Some(address) = index.index_address else {
        return Ok(Vec::new());
    };
    match index.kind {
        SohmIndexKind::List => {
            let len = hdf5_pure_format::sohm_list_len(index.message_count, offset_size);
            let image = source.read_metadata_at(address.get(), len)?;
            hdf5_pure_format::parse_sohm_list(&image, index.message_count, offset_size)
        }
        SohmIndexKind::BTree => {
            let header = BTreeV2Header::parse_from_source(
                &SourceMetadata(source),
                address.get(),
                offset_size,
                length_size,
            )?;
            check_btree_type(&header)?;
            let records =
                collect_btree_v2_records_from_source(source, &header, offset_size, length_size)?;
            records
                .iter()
                .map(|record| SohmRecord::parse(&record.data, offset_size))
                .collect()
        }
    }
}

/// Refuse a B-tree that is not a shared-message index.
///
/// Every B-tree v2 in a file has the same header shape, so an index address that
/// named the wrong tree would decode its records as shared messages and report
/// heap IDs assembled from link names.
fn check_btree_type(header: &BTreeV2Header) -> Result<(), FormatError> {
    if header.tree_type != BTREE_SOHM_INDEX_TYPE {
        return Err(FormatError::InvalidSohmBTreeType(header.tree_type));
    }
    Ok(())
}

/// The index that must hold `message_type`, or an error naming why the lookup
/// cannot proceed.
fn index_for_read(
    table: &SohmTable,
    message_type: MessageType,
) -> Result<(&SohmIndexHeader, StoredAddress), FormatError> {
    let index = table
        .index_for(message_type)
        .ok_or(FormatError::SohmIndexMissing(message_type.to_u16()))?;
    let heap = index
        .heap_address
        .ok_or(FormatError::SohmIndexMissing(message_type.to_u16()))?;
    Ok((index, heap))
}

/// Read the body of the `message_type` message stored in the shared-message heap
/// under `heap_id`.
///
/// The bytes in the heap are the message exactly as an object header would carry
/// it, which is what lets the caller decode them with the ordinary parser for
/// that type.
pub fn read_heap_message(
    file_data: &[u8],
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let (_, heap_address) = index_for_read(table, message_type)?;
    let heap = FractalHeapHeader::parse(
        file_data,
        heap_address.get().to_usize()?,
        offset_size,
        length_size,
    )?;
    HeapObjectReader::new(&heap, offset_size, length_size)
        .read(file_data, &heap_id[..heap.heap_id_length as usize])
}

/// Streaming counterpart of [`read_heap_message`].
pub fn read_heap_message_from_source<S: Source + ?Sized>(
    source: &S,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let (_, heap_address) = index_for_read(table, message_type)?;
    let heap = FractalHeapHeader::parse_from_source(
        &SourceMetadata(source),
        heap_address.get(),
        offset_size,
        length_size,
    )?;
    HeapObjectReader::new(&heap, offset_size, length_size)
        .read_from_source(source, &heap_id[..heap.heap_id_length as usize])
}

#[cfg(test)]
mod tests {
    use test_util::sohm;
    use test_util::widths::Widths;

    use super::*;

    /// An index that was never used has no address, so reading it returns an empty list.
    #[test]
    fn an_index_with_no_address_holds_no_records() {
        let index = SohmIndexHeader {
            message_type_flags: 1 << 3,
            min_message_size: 250,
            list_max: 50,
            btree_min: 40,
            message_count: 2,
            kind: SohmIndexKind::List,
            index_address: None,
            heap_address: Some(StoredAddress::new(0x5678)),
        };
        let empty = crate::source::BytesSource::new(Vec::new());
        assert!(
            read_index_records_from_source(&empty, &index, 8, 8)
                .unwrap()
                .is_empty()
        );
    }

    /// A list index read through a [`Source`], at an address the header names.
    #[test]
    fn a_list_index_is_read_at_the_address_its_header_names() {
        let mut image = vec![0u8; 64];
        image.extend_from_slice(&sohm::list(&[sohm::heap_record(
            0xDEAD_BEEF,
            3,
            [7, 0, 0, 0, 0, 0, 0, 0],
            Widths::EIGHT,
        )]));
        let index = SohmIndexHeader {
            message_type_flags: 1 << 3,
            min_message_size: 250,
            list_max: 50,
            btree_min: 40,
            message_count: 1,
            kind: SohmIndexKind::List,
            index_address: Some(StoredAddress::new(64)),
            heap_address: Some(StoredAddress::new(0x5678)),
        };

        let source = crate::source::BytesSource::new(image);
        let records = read_index_records_from_source(&source, &index, 8, 8).unwrap();
        assert_eq!(
            records[0].location,
            SohmLocation::Heap {
                reference_count: 3,
                heap_id: [7, 0, 0, 0, 0, 0, 0, 0],
            }
        );
    }

    #[test]
    fn a_message_type_no_index_covers_is_named_in_the_error() {
        let table = SohmTable {
            indexes: vec![SohmIndexHeader {
                message_type_flags: 1 << 3,
                min_message_size: 250,
                list_max: 50,
                btree_min: 40,
                message_count: 2,
                kind: SohmIndexKind::List,
                index_address: Some(StoredAddress::new(0x1234)),
                heap_address: Some(StoredAddress::new(0x5678)),
            }],
        };
        assert_eq!(
            index_for_read(&table, MessageType::FILL_VALUE).unwrap_err(),
            FormatError::SohmIndexMissing(MessageType::FILL_VALUE.to_u16())
        );
    }

    /// An index that covers the type but has allocated no heap cannot hold the
    /// message either, and says so by the same name rather than parsing address
    /// zero as a fractal heap.
    #[test]
    fn an_index_without_a_heap_cannot_answer_a_lookup() {
        let table = SohmTable {
            indexes: vec![SohmIndexHeader {
                message_type_flags: 1 << 3,
                min_message_size: 250,
                list_max: 50,
                btree_min: 40,
                message_count: 2,
                kind: SohmIndexKind::List,
                index_address: Some(StoredAddress::new(0x1234)),
                heap_address: None,
            }],
        };
        assert_eq!(
            index_for_read(&table, MessageType::DATATYPE).unwrap_err(),
            FormatError::SohmIndexMissing(MessageType::DATATYPE.to_u16())
        );
    }
}
