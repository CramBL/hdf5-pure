//! The reads of a file's shared messages: the records of an index, and the body of a message from
//! the fractal heap of the index that covers its type.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::FractalHeapIdLayout;
#[cfg(any(feature = "std", test))]
pub use hdf5_pure_format::__private::SharedMessageTableMessage;
pub use hdf5_pure_format::__private::SohmIndexHeader;
use hdf5_pure_format::__private::SohmIndexKind;
#[cfg(any(feature = "std", test))]
pub use hdf5_pure_format::__private::SohmLocation;
pub use hdf5_pure_format::__private::SohmRecord;
pub use hdf5_pure_format::__private::SohmTable;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
#[cfg(test)]
use crate::address::BaseAddress;
#[cfg(test)]
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::btree_v2::BTreeV2Header;
use crate::error::FormatError;
#[cfg(any(test, doc))]
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::HeapObjectReader;
use crate::message_type::MessageType;
use crate::shared_message::FHEAP_ID_LEN;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::FormatWidths;

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
    widths: FormatWidths,
) -> Result<Vec<SohmRecord>, FormatError> {
    let Some(address) = index.index_address else {
        return Ok(Vec::new());
    };
    match index.kind {
        SohmIndexKind::List => {
            let len =
                hdf5_pure_format::__private::sohm_list_len(index.message_count, widths.offsets);
            let image = source.read_metadata_at(address.get(), len)?;
            hdf5_pure_format::__private::parse_sohm_list_with_offset_width(
                &image,
                index.message_count,
                widths.offsets,
            )
        }
        SohmIndexKind::BTree => {
            let header = BTreeV2Header::parse_from_source_with_widths(
                &SourceMetadata(source),
                address.get(),
                widths,
            )?;
            check_btree_type(&header)?;
            let records = crate::btree_v2::collect_btree_v2_records_from_source(
                source,
                &header,
                widths.offsets,
            )?;
            records
                .iter()
                .map(|record| SohmRecord::parse_with_offset_width(&record.data, widths.offsets))
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

/// Reads the body of a message identified by an eight-byte shared-message heap ID.
///
/// The caller can parse the returned bytes as an object header message of `message_type`.
/// `file_data` is framed at the effective base in `addresses`. Read-write access requires full
/// direct-block checksum validation before reading a managed heap message.
///
/// # Errors
///
/// Returns [`FormatError::SohmIndexMissing`] if `table` has no allocated heap for `message_type`,
/// [`FormatError::InvalidFractalHeapIdLength`] if the heap's declared ID length differs from
/// [`FHEAP_ID_LEN`], and the errors of
/// [`parse_header`](crate::fractal_heap::parse_header),
/// [`FractalHeapIdLayout::parse`], and [`HeapObjectReader::read`] if parsing or resolution fails.
pub fn read_heap_message(
    file_data: &[u8],
    access_mode: AccessMode,
    addresses: AddressContext,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (_, heap_address) = index_for_read(table, message_type)?;
    let heap = crate::fractal_heap::parse_header(file_data, addresses, heap_address, widths)?;
    let layout = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths);
    let id = layout.parse(heap_id)?;
    HeapObjectReader::new(&heap, addresses, heap_address, access_mode, widths).read(file_data, id)
}

/// Reads the body of a message identified by an eight-byte shared-message heap ID through a
/// [`Source`].
///
/// `source` is framed at the effective base in `addresses`. The reader parses a
/// [`FractalHeapHeader`] from source regions before resolving the object.
///
/// # Errors
///
/// Returns the errors of [`read_heap_message`], and the error `source` returns if a read fails.
pub fn read_heap_message_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    addresses: AddressContext,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (_, heap_address) = index_for_read(table, message_type)?;
    let heap =
        crate::fractal_heap::read_header_from_source(source, addresses, heap_address, widths)?;
    let layout = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths);
    let id = layout.parse(heap_id)?;
    HeapObjectReader::new(&heap, addresses, heap_address, access_mode, widths)
        .read_from_source(source, id)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::sohm;
    use test_util::widths::Widths;

    use crate::source::BytesSource;
    use crate::width::{LengthWidth, OffsetWidth};

    use super::*;

    const FORMAT_WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    #[rstest]
    #[case::buffered_long(false, 9)]
    #[case::source_long(true, 9)]
    #[case::buffered_short(false, 7)]
    #[case::source_short(true, 7)]
    fn a_shared_heap_declaring_a_different_id_length_is_rejected(
        #[case] streaming: bool,
        #[case] declared: u16,
    ) {
        assert_eq!(
            shared_message(streaming, declared),
            Err(FormatError::InvalidFractalHeapIdLength {
                expected: usize::from(declared),
                actual: FHEAP_ID_LEN,
            })
        );
    }

    #[rstest]
    #[case::buffered(false)]
    #[case::source(true)]
    fn a_shared_heap_with_eight_byte_ids_resolves_a_tiny_message(#[case] streaming: bool) {
        assert_eq!(shared_message(streaming, 8).unwrap(), b"a");
    }

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
            read_index_records_from_source(&empty, &index, FORMAT_WIDTHS)
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
        let records = read_index_records_from_source(&source, &index, FORMAT_WIDTHS).unwrap();
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

    fn shared_message(streaming: bool, declared: u16) -> Result<Vec<u8>, FormatError> {
        let mut header = FractalHeapHeader::parse(
            &fractal_heap::Header::new(256).build(Widths::EIGHT),
            0,
            8,
            8,
        )
        .unwrap();
        header.heap_id_length = declared;
        let image = header
            .serialize(OffsetWidth::Eight, LengthWidth::Eight)
            .unwrap();
        let table = SohmTable {
            indexes: vec![SohmIndexHeader {
                message_type_flags: 1 << MessageType::DATATYPE.to_u16(),
                min_message_size: 0,
                list_max: 0,
                btree_min: 0,
                message_count: 0,
                kind: SohmIndexKind::List,
                index_address: None,
                heap_address: Some(StoredAddress::new(0)),
            }],
        };
        let id = [TINY_ID_FIRST_BYTE, b'a', 0, 0, 0, 0, 0, 0];
        if streaming {
            read_heap_message_from_source(
                &BytesSource::new(image),
                AccessMode::ReadOnly,
                AddressContext::from_effective_base(BaseAddress::new(0)),
                &table,
                MessageType::DATATYPE,
                &id,
                FORMAT_WIDTHS,
            )
        } else {
            read_heap_message(
                &image,
                AccessMode::ReadOnly,
                AddressContext::from_effective_base(BaseAddress::ZERO),
                &table,
                MessageType::DATATYPE,
                &id,
                FORMAT_WIDTHS,
            )
        }
    }

    // A version 0 tiny ID has type 2 in bits 4 and 5 ("Fractal Heap", specification 4.0).
    const TINY_ID_FIRST_BYTE: u8 = 0x20;
}
