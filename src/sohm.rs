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
#[cfg(test)]
use crate::address::AddressContext;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::btree::v2 as btree_v2;
use crate::btree::v2::BTreeV2Header;
use crate::btree::v2::BTreeV2Source;
use crate::error::FormatError;
#[cfg(test)]
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::HeapObjectReader;
use crate::fractal_heap::HeapSource;
use crate::message_type::MessageType;
use crate::shared_message::FHEAP_ID_LEN;
#[cfg(test)]
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
/// addresses through [`StoredRead::read_metadata`]. For a B-tree, checks the client type and
/// record size after parsing the header and before reading the nodes.
///
/// # Errors
///
/// Returns [`FormatError::InvalidSohmBTreeType`] if a B-tree has the wrong client type, or
/// [`FormatError::DataSizeMismatch`] if its record size differs from [`SohmRecord::encoded_len`].
/// Propagates source and parser errors while reading the index and its records.
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
            SohmBTreeIndex::new(header, widths.offsets)?.read_records(source)
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

/// A shared-message B-tree header with a checked client type and record size.
struct SohmBTreeIndex {
    header: BTreeV2Header,
    offset_width: OffsetWidth,
}

impl SohmBTreeIndex {
    /// Checks the header at the address width used to read the records.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidSohmBTreeType`] if the client type is not 7, or
    /// [`FormatError::DataSizeMismatch`] if the record size differs from [`SohmRecord::encoded_len`].
    fn new(header: BTreeV2Header, offset_width: OffsetWidth) -> Result<Self, FormatError> {
        if header.tree_type != BTREE_SOHM_INDEX_TYPE {
            return Err(FormatError::InvalidSohmBTreeType(header.tree_type));
        }
        let expected = SohmRecord::encoded_len(offset_width);
        let actual = usize::from(header.record_size);
        if actual != expected {
            return Err(FormatError::DataSizeMismatch { expected, actual });
        }
        Ok(Self {
            header,
            offset_width,
        })
    }

    /// Collects the tree's records and decodes their shared-message locations.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`btree_v2::collect_btree_v2_records_from_source`] and
    /// [`SohmRecord::parse_with_offset_width`].
    fn read_records(
        &self,
        source: &(impl StoredRead + ?Sized),
    ) -> Result<Vec<SohmRecord>, FormatError> {
        let Self {
            header,
            offset_width,
        } = self;
        let records = btree_v2::collect_btree_v2_records_from_source(
            &BTreeV2Source::new(source),
            header,
            *offset_width,
        )?;
        records
            .iter()
            .map(|record| SohmRecord::parse_with_offset_width(&record.data, *offset_width))
            .collect()
    }
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

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::BTREE_V2_LINK_CREATION_ORDER;
    use rstest::rstest;
    use test_util::btree::v2 as btree_v2;
    use test_util::checksum;
    use test_util::fractal_heap;
    use test_util::image::Image;
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
            btree_v2::Header::new(BTREE_V2_LINK_CREATION_ORDER, sohm::RECORD_SIZE - 1, 256, 1)
                .build(Widths::EIGHT)
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
                let expected = address + u64::try_from(encoded.len()).unwrap();
                encoded.truncate(encoded.len() - 5);
                FormatError::UnexpectedSourceEof {
                    expected,
                    available: address + u64::try_from(encoded.len()).unwrap(),
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
            InvalidIndex::BTreeType => {
                FormatError::InvalidSohmBTreeType(BTREE_V2_LINK_CREATION_ORDER)
            }
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

    #[rstest]
    #[case::two_byte_leaf(
        btree_with_unpadded_record(TWO_BYTE_OFFSETS, TWO_BYTE_OBJECT_HEADER_SPAN),
        OffsetWidth::Two,
        TWO_BYTE_OBJECT_HEADER_SPAN
    )]
    #[case::four_byte_leaf(
        btree_with_unpadded_record(FOUR_BYTE_OFFSETS, FOUR_BYTE_OBJECT_HEADER_SPAN),
        OffsetWidth::Four,
        FOUR_BYTE_OBJECT_HEADER_SPAN
    )]
    #[case::two_byte_missing_leaf(
        btree_header(TWO_BYTE_OFFSETS, TWO_BYTE_OBJECT_HEADER_SPAN, 1),
        OffsetWidth::Two,
        TWO_BYTE_OBJECT_HEADER_SPAN
    )]
    #[case::four_byte_missing_leaf(
        btree_header(FOUR_BYTE_OFFSETS, FOUR_BYTE_OBJECT_HEADER_SPAN, 1),
        OffsetWidth::Four,
        FOUR_BYTE_OBJECT_HEADER_SPAN
    )]
    fn index_reader_rejects_an_unpadded_btree_stride_before_reading_nodes(
        #[case] image: Vec<u8>,
        #[case] offsets: OffsetWidth,
        #[case] stride: u16,
    ) {
        assert_index_records(
            &image,
            SohmIndexKind::BTree,
            1,
            offsets,
            Err(FormatError::DataSizeMismatch {
                expected: usize::from(sohm::RECORD_SIZE),
                actual: usize::from(stride),
            }),
        );
    }

    #[rstest]
    #[case::two(TWO_BYTE_OFFSETS, OffsetWidth::Two)]
    #[case::four(FOUR_BYTE_OFFSETS, OffsetWidth::Four)]
    #[case::eight(Widths::EIGHT, OffsetWidth::Eight)]
    fn index_reader_decodes_the_same_mixed_records_from_lists_and_btrees(
        #[case] widths: Widths,
        #[case] offsets: OffsetWidth,
    ) {
        let id = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut object_header =
            sohm::object_header_record(3, ATTRIBUTE_MESSAGE_TYPE, 2, 0x1234, widths);
        object_header[sohm::OBJECT_HEADER_ADDRESS_OFFSET + widths.offset..].fill(0xA5);
        let records = [object_header, sohm::heap_record(7, 4, id, widths)];
        let expected = vec![
            SohmRecord {
                hash: 3,
                location: SohmLocation::ObjectHeader {
                    message_type: ATTRIBUTE_MESSAGE_TYPE,
                    creation_index: 2,
                    address: StoredAddress::new(0x1234),
                },
            },
            SohmRecord {
                hash: 7,
                location: SohmLocation::Heap {
                    reference_count: 4,
                    heap_id: id,
                },
            },
        ];
        assert_index_records(
            &sohm::list(&records),
            SohmIndexKind::List,
            2,
            offsets,
            Ok(expected.clone()),
        );
        assert_index_records(
            &single_leaf_btree(widths, sohm::RECORD_SIZE, &records),
            SohmIndexKind::BTree,
            2,
            offsets,
            Ok(expected),
        );
    }

    #[test]
    fn index_reader_applies_btree_header_checksum_policy_before_checking_the_stride() {
        let stride = sohm::RECORD_SIZE + 1;
        let mut header = btree_header(Widths::EIGHT, stride, 1);
        let checksum_error = damage_checksum(&mut header);
        assert_index_records(
            &header,
            SohmIndexKind::BTree,
            1,
            OffsetWidth::Eight,
            Err(if cfg!(feature = "checksum") {
                checksum_error
            } else {
                FormatError::DataSizeMismatch {
                    expected: usize::from(sohm::RECORD_SIZE),
                    actual: usize::from(stride),
                }
            }),
        );
    }

    #[rstest]
    #[case::list(sohm::list(&[record_with_invalid_location()]), SohmIndexKind::List, 0)]
    #[case::btree(
        single_leaf_btree(Widths::EIGHT, sohm::RECORD_SIZE, &[record_with_invalid_location()]),
        SohmIndexKind::BTree,
        ROOT_ADDRESS,
    )]
    fn shared_message_index_reader_applies_checksum_policy_before_decoding_locations(
        #[case] mut image: Vec<u8>,
        #[case] kind: SohmIndexKind,
        #[case] container_at: usize,
    ) {
        let location_error = FormatError::InvalidSohmRecordLocation(INVALID_LOCATION);
        assert_index_records(
            &image,
            kind,
            1,
            OffsetWidth::Eight,
            Err(location_error.clone()),
        );
        let checksum_error = damage_checksum(&mut image[container_at..]);
        assert_index_records(
            &image,
            kind,
            1,
            OffsetWidth::Eight,
            Err(if cfg!(feature = "checksum") {
                checksum_error
            } else {
                location_error
            }),
        );
    }

    fn record_with_invalid_location() -> Vec<u8> {
        let mut record = sohm::heap_record(7, 3, [1; FHEAP_ID_LEN], Widths::EIGHT);
        record[0] = INVALID_LOCATION;
        record
    }

    fn btree_with_unpadded_record(widths: Widths, stride: u16) -> Vec<u8> {
        let mut record = sohm::object_header_record(7, ATTRIBUTE_MESSAGE_TYPE, 2, 64, widths);
        record.truncate(usize::from(stride));
        single_leaf_btree(widths, stride, &[record])
    }

    fn single_leaf_btree(widths: Widths, stride: u16, records: &[Vec<u8>]) -> Vec<u8> {
        let header = btree_header(widths, stride, u16::try_from(records.len()).unwrap());
        let mut image = Image::starting_with(&header);
        image.place(ROOT_ADDRESS, &btree_v2::leaf(sohm::BTREE_TYPE, records));
        image.build()
    }

    fn btree_header(widths: Widths, stride: u16, count: u16) -> Vec<u8> {
        btree_v2::Header::new(sohm::BTREE_TYPE, stride, ROOT_ADDRESS as u64, count).build(widths)
    }

    fn assert_index_records(
        image: &[u8],
        kind: SohmIndexKind,
        message_count: u16,
        offsets: OffsetWidth,
        expected: Result<Vec<SohmRecord>, FormatError>,
    ) {
        let index = SohmIndexHeader {
            message_type_flags: 1 << MessageType::ATTRIBUTE.to_u16(),
            min_message_size: 0,
            list_max: 50,
            btree_min: 40,
            message_count,
            kind,
            index_address: Some(StoredAddress::new(0)),
            heap_address: None,
        };
        let widths = FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        };
        assert_eq!(
            read_index_records_from_source(&crate::source::zero_base_bytes(image), &index, widths),
            expected,
        );
        let source = BytesSource::new(image);
        assert_eq!(
            read_index_records_from_source(
                &crate::source::zero_base_source(&source),
                &index,
                widths,
            ),
            expected,
        );
    }

    fn damage_checksum(bytes: &mut [u8]) -> FormatError {
        let (body, stored) = bytes
            .split_last_chunk_mut::<{ checksum::CHECKSUM }>()
            .unwrap();
        stored[0] ^= 1;
        FormatError::ChecksumMismatch {
            expected: u32::from_le_bytes(*stored),
            computed: checksum::lookup3(body),
        }
    }

    const ROOT_ADDRESS: usize = 256;
    const ATTRIBUTE_MESSAGE_TYPE: u8 = MessageType::ATTRIBUTE.to_u16().to_le_bytes()[0];
    const INVALID_LOCATION: u8 = 9;
    const TWO_BYTE_OFFSETS: Widths = Widths {
        offset: 2,
        length: 8,
    };
    const FOUR_BYTE_OFFSETS: Widths = Widths {
        offset: 4,
        length: 8,
    };
    const TWO_BYTE_OBJECT_HEADER_SPAN: u16 =
        (sohm::OBJECT_HEADER_ADDRESS_OFFSET + TWO_BYTE_OFFSETS.offset) as u16;
    const FOUR_BYTE_OBJECT_HEADER_SPAN: u16 =
        (sohm::OBJECT_HEADER_ADDRESS_OFFSET + FOUR_BYTE_OFFSETS.offset) as u16;

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
