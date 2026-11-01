//! Reads shared-message tables, index records, and heap message bodies.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::FractalHeapIdLayout;
pub use hdf5_pure_format::__private::SharedMessageTableMessage;
pub use hdf5_pure_format::__private::SohmIndexHeader;
use hdf5_pure_format::__private::SohmIndexKind;
#[cfg(any(feature = "std", test))]
pub use hdf5_pure_format::__private::SohmLocation;
pub use hdf5_pure_format::__private::SohmRecord;
pub use hdf5_pure_format::__private::SohmTable;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::btree_v2;
use crate::btree_v2::BTreeV2Header;
use crate::btree_v2::BTreeV2Source;
use crate::error::FormatError;
#[cfg(any(test, doc))]
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::HeapObjectReader;
use crate::fractal_heap::HeapSource;
use crate::fractal_heap::LegacyHeapObjectReader;
use crate::message_type::MessageType;
use crate::shared_message::FHEAP_ID_LEN;
use crate::source::Source;
use crate::source::StoredBytes;
use crate::source::StoredMetadata;
use crate::source::StoredRead;
use crate::width::FormatWidths;
use crate::width::OffsetWidth;

/// B-tree v2 type of a shared-message index (`H5B2_SOHM_INDEX_ID`).
const BTREE_SOHM_INDEX_TYPE: u8 = 7;

/// Reads every record of a shared-message list or B-tree index.
///
/// Returns an empty vector if `index` has no allocated address. Metadata reads use stored
/// addresses through [`StoredRead::read_metadata`].
///
/// # Errors
///
/// Returns [`FormatError::InvalidSohmBTreeType`] if a B-tree has the wrong client type. Returns
/// the parser error if the index or a record is malformed, and source errors if a read fails.
pub fn read_index_records_from_source(
    source: &(impl StoredRead + ?Sized),
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
            let image = source.read_metadata(address, len)?;
            hdf5_pure_format::__private::parse_sohm_list_with_offset_width(
                &image,
                index.message_count,
                widths.offsets,
            )
        }
        SohmIndexKind::BTree => {
            let header = BTreeV2Header::parse_from_source_with_widths(
                &StoredMetadata(source),
                address.get(),
                widths,
            )?;
            check_btree_type(&header)?;
            let records = btree_v2::collect_btree_v2_records_from_source(
                &BTreeV2Source::new(source),
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

/// Reads the master table named by `message` from resident file bytes.
///
/// # Errors
///
/// Returns errors from [`SohmTable::read_with_offset_width`].
pub(crate) fn read_table(
    source: StoredBytes<'_>,
    message: &SharedMessageTableMessage,
    offset_width: OffsetWidth,
) -> Result<SohmTable, FormatError> {
    SohmTable::read_with_offset_width(source.metadata_frame().bytes(), message, offset_width)
}

/// Reads the master table named by `message` through stored-address metadata reads.
///
/// # Errors
///
/// Returns errors from [`SohmTable::read_from_source_with_offset_width`].
pub(crate) fn read_table_from_source(
    source: &(impl StoredRead + ?Sized),
    message: &SharedMessageTableMessage,
    offset_width: OffsetWidth,
) -> Result<SohmTable, FormatError> {
    SohmTable::read_from_source_with_offset_width(&StoredMetadata(source), message, offset_width)
}

/// Reads a message body from the shared-message heap for `message_type`.
///
/// The caller can parse the returned bytes as an object header message of `message_type`.
/// Borrows resident heap metadata through `source`. Read-write access requires complete
/// validation of checksummed direct blocks before reading a managed heap object.
///
/// # Errors
///
/// Returns [`FormatError::SohmIndexMissing`] if `table` has no allocated heap for `message_type`,
/// [`FormatError::InvalidFractalHeapIdLength`] if the heap's declared ID length differs from
/// [`FHEAP_ID_LEN`], and errors from
/// [`read_header_and_range_from_source`](crate::fractal_heap::read_header_and_range_from_source),
/// [`FractalHeapIdLayout::parse`], or [`HeapObjectReader::read`].
pub(crate) fn read_heap_message(
    source: StoredBytes<'_>,
    access_mode: AccessMode,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    read_heap_message_with(
        HeapSource::resident(source),
        access_mode,
        table,
        message_type,
        heap_id,
        widths,
    )
}

/// Reads a shared message body through stored-address metadata reads.
///
/// # Errors
///
/// Returns the errors of [`read_heap_message`], and source errors if a read fails.
pub(crate) fn read_heap_message_from_source(
    source: &dyn StoredRead,
    access_mode: AccessMode,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    read_heap_message_with(
        HeapSource::new(source),
        access_mode,
        table,
        message_type,
        heap_id,
        widths,
    )
}

fn read_heap_message_with(
    source: HeapSource<'_>,
    access_mode: AccessMode,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (_, heap_address) = index_for_read(table, message_type)?;
    let (heap, _) =
        crate::fractal_heap::read_header_and_range_from_source(&source, heap_address, widths)?;
    let layout = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths);
    let id = layout.parse(heap_id)?;
    HeapObjectReader::new(source, &heap, heap_address, access_mode, widths).read(id)
}

/// Rejects a B-tree whose client type differs from the shared-message index type.
///
/// # Errors
///
/// Returns [`FormatError::InvalidSohmBTreeType`] if the client type differs from
/// [`BTREE_SOHM_INDEX_TYPE`].
fn check_btree_type(header: &BTreeV2Header) -> Result<(), FormatError> {
    if header.tree_type != BTREE_SOHM_INDEX_TYPE {
        return Err(FormatError::InvalidSohmBTreeType(header.tree_type));
    }
    Ok(())
}

/// Returns the index for `message_type` and its allocated heap address.
///
/// # Errors
///
/// Returns [`FormatError::SohmIndexMissing`] if the index or its heap address is absent.
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
/// [`legacy_parse_header`](crate::fractal_heap::legacy_parse_header),
/// [`FractalHeapIdLayout::parse`], and [`LegacyHeapObjectReader::read`] if parsing or resolution fails.
pub fn legacy_read_heap_message(
    file_data: &[u8],
    access_mode: AccessMode,
    addresses: AddressContext,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (_, heap_address) = index_for_read(table, message_type)?;
    let heap =
        crate::fractal_heap::legacy_parse_header(file_data, addresses, heap_address, widths)?;
    let layout = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths);
    let id = layout.parse(heap_id)?;
    LegacyHeapObjectReader::new(&heap, addresses, heap_address, access_mode, widths)
        .read(file_data, id)
}

/// Reads the body of a message identified by an eight-byte shared-message heap ID through a
/// [`Source`].
///
/// `source` is framed at the effective base in `addresses`. The reader parses a
/// [`FractalHeapHeader`] from source regions before resolving the object.
///
/// # Errors
///
/// Returns the errors of [`legacy_read_heap_message`], and source errors if a read fails.
pub fn legacy_read_heap_message_from_source<S: Source + ?Sized>(
    source: &S,
    access_mode: AccessMode,
    addresses: AddressContext,
    table: &SohmTable,
    message_type: MessageType,
    heap_id: &[u8; FHEAP_ID_LEN],
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (_, heap_address) = index_for_read(table, message_type)?;
    let heap = crate::fractal_heap::legacy_read_header_from_source(
        source,
        addresses,
        heap_address,
        widths,
    )?;
    let layout = FractalHeapIdLayout::new(&heap, widths.offsets, widths.lengths);
    let id = layout.parse(heap_id)?;
    LegacyHeapObjectReader::new(&heap, addresses, heap_address, access_mode, widths)
        .read_from_source(source, id)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::btree_v2;
    use test_util::fractal_heap;
    use test_util::sohm;
    use test_util::widths::Widths;

    use crate::address::BaseAddress;
    use crate::address::BaseAddressExt;
    use crate::source::BytesSource;
    #[cfg(feature = "std")]
    use crate::source::MetadataCacheConfig;
    #[cfg(feature = "std")]
    use crate::source::MetadataCachingSource;
    use crate::source::PhysicalSource;
    use crate::width::{LengthWidth, OffsetWidth};

    use super::*;

    const FORMAT_WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    #[derive(Clone, Copy, Debug)]
    enum InvalidIndex {
        AddressOverflow,
        BTreeTruncated,
        BTreeType,
        ListSignature,
        ListTruncated,
    }

    #[rstest]
    #[case::address_overflow(InvalidIndex::AddressOverflow)]
    #[case::btree_truncated(InvalidIndex::BTreeTruncated)]
    #[case::btree_type(InvalidIndex::BTreeType)]
    #[case::list_signature(InvalidIndex::ListSignature)]
    #[case::list_truncated(InvalidIndex::ListTruncated)]
    fn stored_indexes_preserve_exact_errors(
        #[case] invalid: InvalidIndex,
        #[values(0, 512)] base: usize,
    ) {
        let btree = matches!(
            invalid,
            InvalidIndex::BTreeTruncated | InvalidIndex::BTreeType
        );
        let address = if matches!(invalid, InvalidIndex::AddressOverflow) {
            u64::MAX
        } else {
            64
        };
        let mut encoded = if btree {
            btree_v2::Header::new(6, 17, 256, 1).build(Widths::EIGHT)
        } else {
            sohm::list(&[sohm::heap_record(7, 3, [0; 8], Widths::EIGHT)])
        };
        let expected = match invalid {
            InvalidIndex::AddressOverflow => FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: if base == 0 {
                    encoded.len() as u64
                } else {
                    base as u64
                },
            },
            InvalidIndex::BTreeTruncated => {
                encoded.truncate(encoded.len() - 5);
                FormatError::UnexpectedEof {
                    expected: encoded.len() + 1,
                    available: encoded.len(),
                }
            }
            InvalidIndex::ListTruncated => {
                let expected = (base + 64 + encoded.len()) as u64;
                encoded.pop();
                FormatError::UnexpectedSourceEof {
                    expected,
                    available: expected - 1,
                }
            }
            InvalidIndex::BTreeType => FormatError::InvalidSohmBTreeType(6),
            InvalidIndex::ListSignature => {
                encoded[0] = 0;
                FormatError::InvalidSohmListSignature
            }
        };
        let mut image = vec![0; base + 64];
        image.extend_from_slice(&encoded);
        let source = BytesSource::new(image);
        let stored = PhysicalSource::for_test(
            &source,
            AddressContext::from_effective_base(BaseAddress::new(base as u64)),
        )
        .stored();
        let index = SohmIndexHeader {
            message_type_flags: 1 << MessageType::DATATYPE.to_u16(),
            min_message_size: 0,
            list_max: 50,
            btree_min: 40,
            message_count: 1,
            kind: if btree {
                SohmIndexKind::BTree
            } else {
                SohmIndexKind::List
            },
            index_address: Some(StoredAddress::new(address)),
            heap_address: None,
        };
        assert_eq!(
            read_index_records_from_source(&stored, &index, FORMAT_WIDTHS).unwrap_err(),
            expected,
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::borrowed(false)]
    #[case::owned(true)]
    fn shared_tables_preserve_metadata_ranges_owners_and_cache_admission(
        #[case] owned: bool,
        #[values(0, 512)] base: usize,
    ) {
        let index = sohm::Index {
            kind: sohm::Kind::LIST,
            message_type_flags: 1 << MessageType::DATATYPE.to_u16(),
            min_message_size: 1,
            list_max: 50,
            btree_min: 40,
            message_count: 2,
            index_address: Some(128),
            heap_address: Some(256),
        };
        let encoded = sohm::table(&[index], Widths::EIGHT);
        let mut image = vec![0; base + 64];
        image.extend_from_slice(&encoded);
        let borrowed = BytesSource::new(&image);
        let allocating = AllocatingSource {
            bytes: &image,
            origin: 37,
        };
        let source: &dyn Source = if owned { &allocating } else { &borrowed };
        let cached = MetadataCachingSource::new(source, MetadataCacheConfig::new(1024));
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base as u64));
        let stored = PhysicalSource::for_test(&cached, addresses).stored();
        let message = SharedMessageTableMessage {
            table_address: StoredAddress::new(64),
            index_count: 1,
        };
        let expected = SohmTable {
            indexes: vec![SohmIndexHeader {
                message_type_flags: index.message_type_flags,
                min_message_size: 1,
                list_max: 50,
                btree_min: 40,
                message_count: 2,
                kind: SohmIndexKind::List,
                index_address: Some(StoredAddress::new(128)),
                heap_address: Some(StoredAddress::new(256)),
            }],
        };
        assert_eq!(
            read_table_from_source(&stored, &message, OffsetWidth::Eight).unwrap(),
            expected
        );
        assert_eq!(
            read_table_from_source(&stored, &message, OffsetWidth::Eight).unwrap(),
            expected
        );
        let stats = cached.metadata_cache_stats().unwrap();
        assert_eq!(
            (stats.hits(), stats.misses(), stats.entries(), stats.bytes()),
            if owned {
                (1, 1, 1, encoded.len())
            } else {
                (0, 2, 0, 0)
            },
        );
        let metadata = stored
            .read_metadata(message.table_address, encoded.len())
            .unwrap();
        assert_eq!(metadata.as_ref(), encoded);
        assert_eq!(
            metadata.offset(),
            (base + 64) as u64 + if owned { 37 } else { 0 }
        );
        assert_eq!(metadata.shared().is_some(), owned);
        if !owned {
            assert_eq!(metadata.as_ptr(), image[base + 64..].as_ptr());
        }
        let retained = metadata.shared();
        drop(metadata);
        drop(cached);
        if let Some(retained) = retained {
            assert_eq!(retained.as_ref(), encoded);
            assert_eq!(retained.offset(), (base + 64 + 37) as u64);
        }
    }

    #[cfg(feature = "std")]
    struct AllocatingSource<'a> {
        bytes: &'a [u8],
        origin: u64,
    }

    #[cfg(feature = "std")]
    impl Source for AllocatingSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }
        fn metadata_origin(&self) -> Result<u64, FormatError> {
            Ok(self.origin)
        }
        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
            BytesSource::new(self.bytes).read_at(offset, buffer)
        }
    }

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
            read_index_records_from_source(
                &crate::source::zero_base_source(&empty),
                &index,
                FORMAT_WIDTHS
            )
            .unwrap()
            .is_empty()
        );
    }

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
        let records = read_index_records_from_source(
            &crate::source::zero_base_source(&source),
            &index,
            FORMAT_WIDTHS,
        )
        .unwrap();
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
                &crate::source::zero_base_source(&BytesSource::new(image)),
                AccessMode::ReadOnly,
                &table,
                MessageType::DATATYPE,
                &id,
                FORMAT_WIDTHS,
            )
        } else {
            read_heap_message(
                crate::source::zero_base_bytes(&image),
                AccessMode::ReadOnly,
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
