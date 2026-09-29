//! The shared object header message structures: the Shared Message Table message, the master
//! table of indexes, and the records of an index.
//!
//! A file created with `H5Pset_shared_mesg_nindexes` and `H5Pset_shared_mesg_index` stores one
//! copy of each message it shares, in the fractal heap of the index that covers the message type.
//! An object that uses the message holds an 8-byte heap ID in place of its body, which
//! [`parse_shared_ref`](crate::shared_message::parse_shared_ref) reads into a
//! [`SharedLocation::SohmHeap`]. Three structures lead from the superblock to the heap:
//!
//! - the Shared Message Table message (type 0x000F) in the superblock extension, with the address
//!   of the master table and its number of indexes ([`SharedMessageTableMessage`])
//! - the master table, signature `SMTB`, with one [`SohmIndexHeader`] per index, which holds the
//!   message types of the index, the address of its fractal heap, and the address of its records
//!   ([`SohmTable`])
//! - the index, a list (signature `SMLI`) or a version 2 B-tree of type 7, with one [`SohmRecord`]
//!   per shared message
//!
//! A reader finds a message through the first two, since the heap ID in a reference locates the
//! message in the heap. A writer searches the records of the index for an equal message to share.
//!
//! [`SharedLocation::SohmHeap`]: crate::shared_message::SharedLocation::SohmHeap

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;
use crate::shared_message::FHEAP_ID_LEN;

/// The signature of the master table, `H5SM_TABLE_MAGIC`.
const TABLE_SIGNATURE: &[u8; 4] = b"SMTB";

/// The signature of a list index, `H5SM_LIST_MAGIC`.
const LIST_SIGNATURE: &[u8; 4] = b"SMLI";

/// The one version of the Shared Message Table message, of an index header, and of a list index,
/// `HDF5_SHAREDHEADER_VERSION` and `H5SM_LIST_VERSION`.
const SOHM_VERSION: u8 = 0;

/// The most indexes a file declares, `H5O_SHMESG_MAX_NINDEXES`.
///
/// A parser sizes the read of the master table from the number of indexes, and rejects a number
/// above this.
const MAX_INDEXES: u8 = 8;

/// The size in bytes of the parts of a master table outside its index headers: the signature and
/// the checksum.
const TABLE_FIXED_LEN: usize = 4 + 4;

/// The size in bytes of the fields of an index header outside its two addresses: the version (1),
/// the index type (1), the message type flags (2), the minimum message size (4), the list cutoff
/// (2), the B-tree cutoff (2), and the number of messages (2).
const INDEX_HEADER_FIXED_LEN: usize = 14;

/// The size in bytes of the parts of a list index outside its records: the signature and the
/// checksum after the records in use.
const LIST_FIXED_LEN: usize = 4 + 4;

/// The size in bytes of the fields every record begins with: the location (1) and the hash (4).
const RECORD_PREFIX_LEN: usize = 5;

/// The size in bytes of the fields of a record of a message in the heap after the location and
/// the hash: the reference count (4) and the heap ID (8).
const HEAP_LOCATION_LEN: usize = 4 + FHEAP_ID_LEN;

/// The location byte of a record of a message in the heap, `H5SM_IN_HEAP`.
const LOCATION_HEAP: u8 = 0;

/// The location byte of a record of a message in an object header, `H5SM_IN_OH`.
const LOCATION_OBJECT_HEADER: u8 = 1;

/// The Shared Message Table message (type 0x000F), which the superblock extension of a file that
/// shares messages holds.
///
/// The message is defined in "The Shared Message Table Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_shared
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessageTableMessage {
    /// The address of the master table.
    pub table_address: StoredAddress,
    /// The number of indexes in the master table. The master table does not store the count, so
    /// its parser takes it from the message.
    pub index_count: u8,
}

impl SharedMessageTableMessage {
    /// Parses the body of a Shared Message Table message.
    ///
    /// `offset_size` is the superblock's "Size of Offsets" byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidSohmTableVersion`] if the version is not 0,
    /// [`FormatError::UnexpectedEof`] if the body ends before the number of indexes,
    /// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and
    /// [`FormatError::InvalidSohmIndexCount`] if the number of indexes is 0 or above 8.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<Self, FormatError> {
        bytes::ensure_len(data, 0, 1)?;
        let version = data[0];
        if version != SOHM_VERSION {
            return Err(FormatError::InvalidSohmTableVersion(version));
        }
        let table_address = StoredAddress::new(bytes::read_offset(data, 1, offset_size)?);
        let pos = 1 + offset_size as usize;
        bytes::ensure_len(data, pos, 1)?;
        let index_count = data[pos];
        if index_count == 0 || index_count > MAX_INDEXES {
            return Err(FormatError::InvalidSohmIndexCount(index_count));
        }
        Ok(Self {
            table_address,
            index_count,
        })
    }
}

/// The way an index stores its records.
///
/// An index with more records than its [`list_max`](SohmIndexHeader::list_max) is a B-tree, and
/// an index with fewer than its [`btree_min`](SohmIndexHeader::btree_min) a list. `H5SM_init` in
/// `H5SM.c` (HDF5 2.2.0) starts an index as a list where its `list_max` is above 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SohmIndexKind {
    /// An unsorted list, signature `SMLI`.
    List,
    /// A version 2 B-tree of type 7.
    BTree,
}

/// The header of one index of the master table.
///
/// The index header is defined in "Shared Object Header Message Table" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_sohm
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmIndexHeader {
    /// The message types of the index, with the bit of each message type ID set, as in the
    /// `H5O_SHMESG_*_FLAG` macros of `H5Opublic.h` (HDF5 2.2.0): bit 1 for the dataspace, 3 for
    /// the datatype, 5 for the fill value, 11 for the filter pipeline, and 12 for the attribute.
    /// The specification assigns bits 0 to 4 to the five types, and a file the C library writes
    /// has the bits of the macros.
    pub message_type_flags: u16,
    /// The size in bytes below which a message is not shared.
    pub min_message_size: u32,
    /// The number of records above which the index is a B-tree.
    pub list_max: u16,
    /// The number of records below which the index is a list.
    pub btree_min: u16,
    /// The number of records in the index.
    pub message_count: u16,
    /// Whether [`index_address`](Self::index_address) is a list or a B-tree.
    pub kind: SohmIndexKind,
    /// The address of the list or the B-tree, or `None` while the index is empty.
    pub index_address: Option<StoredAddress>,
    /// The address of the fractal heap with the messages of the index, or `None` while the index
    /// is empty.
    pub heap_address: Option<StoredAddress>,
}

impl SohmIndexHeader {
    /// Returns `true` if the index covers `message_type`.
    ///
    /// The flags hold a bit for each message type ID below 16, so the index covers no type of 16
    /// or above.
    pub fn covers(&self, message_type: MessageType) -> bool {
        match u32::from(message_type.to_u16()) {
            bit if bit < 16 => self.message_type_flags & (1u16 << bit) != 0,
            _ => false,
        }
    }
}

/// The master table of the shared message indexes of a file, signature `SMTB`.
///
/// [`parse`](Self::parse) reads the fields in the order of `H5SM__cache_table_deserialize` in
/// `H5SMcache.c` (HDF5 2.2.0). The table is defined in "Shared Object Header Message Table" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_sohm
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmTable {
    /// The indexes, in the order the table stores them.
    pub indexes: Vec<SohmIndexHeader>,
}

/// Returns the size in bytes of one index header.
fn index_header_len(offset_size: u8) -> usize {
    INDEX_HEADER_FIXED_LEN + 2 * offset_size as usize
}

/// Returns the size in bytes of a master table of `index_count` indexes.
fn table_len(index_count: u8, offset_size: u8) -> usize {
    TABLE_FIXED_LEN + index_count as usize * index_header_len(offset_size)
}

/// Returns the size in bytes of one record of a shared message index.
///
/// A record has one of two shapes, and a list gives every record the size of the wider shape,
/// `H5SM_SOHM_ENTRY_SIZE`.
pub(crate) fn sohm_record_len(offset_size: u8) -> usize {
    let object_header_location = 1 + 1 + 2 + offset_size as usize;
    RECORD_PREFIX_LEN + HEAP_LOCATION_LEN.max(object_header_location)
}

impl SohmTable {
    /// Parses the master table at the start of `image`.
    ///
    /// `index_count` is the [`index_count`](SharedMessageTableMessage::index_count) of the Shared
    /// Message Table message. `offset_size` is the superblock's "Size of Offsets" byte. With the
    /// `checksum` feature, `parse` also verifies the table's checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidSohmIndexCount`] if `index_count` is 0 or above 8,
    /// [`FormatError::UnexpectedEof`] if `image` is shorter than the table,
    /// [`FormatError::InvalidSohmTableSignature`] if the table does not begin with `SMTB`,
    /// [`FormatError::InvalidSohmTableVersion`] if the version of an index is not 0,
    /// [`FormatError::InvalidSohmIndexKind`] if its index type is not 0 or 1,
    /// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and, with the
    /// `checksum` feature, [`FormatError::ChecksumMismatch`] if the stored checksum differs from
    /// the computed one.
    pub fn parse(image: &[u8], index_count: u8, offset_size: u8) -> Result<Self, FormatError> {
        if index_count == 0 || index_count > MAX_INDEXES {
            return Err(FormatError::InvalidSohmIndexCount(index_count));
        }
        let expected = table_len(index_count, offset_size);
        bytes::ensure_len(image, 0, expected)?;
        let image = &image[..expected];
        if &image[..4] != TABLE_SIGNATURE {
            return Err(FormatError::InvalidSohmTableSignature);
        }
        checksum::verify_trailing(image)?;

        let mut indexes = Vec::with_capacity(index_count as usize);
        let mut pos = 4;
        for _ in 0..index_count {
            let version = image[pos];
            if version != SOHM_VERSION {
                return Err(FormatError::InvalidSohmTableVersion(version));
            }
            let kind = match image[pos + 1] {
                0 => SohmIndexKind::List,
                1 => SohmIndexKind::BTree,
                other => return Err(FormatError::InvalidSohmIndexKind(other)),
            };
            let message_type_flags = u16::from_le_bytes([image[pos + 2], image[pos + 3]]);
            let min_message_size = u32::from_le_bytes([
                image[pos + 4],
                image[pos + 5],
                image[pos + 6],
                image[pos + 7],
            ]);
            let list_max = u16::from_le_bytes([image[pos + 8], image[pos + 9]]);
            let btree_min = u16::from_le_bytes([image[pos + 10], image[pos + 11]]);
            let message_count = u16::from_le_bytes([image[pos + 12], image[pos + 13]]);
            let mut at = pos + INDEX_HEADER_FIXED_LEN;
            let index_address =
                bytes::read_optional_offset(image, at, offset_size)?.map(StoredAddress::new);
            at += offset_size as usize;
            let heap_address =
                bytes::read_optional_offset(image, at, offset_size)?.map(StoredAddress::new);
            pos = at + offset_size as usize;

            indexes.push(SohmIndexHeader {
                message_type_flags,
                min_message_size,
                list_max,
                btree_min,
                message_count,
                kind,
                index_address,
                heap_address,
            });
        }
        Ok(Self { indexes })
    }

    /// Parses the master table at the address `message` holds, in `file_data`.
    ///
    /// `file_data` holds the file from its base address on.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and
    /// [`FormatError::ValueTooLargeForPlatform`] if the address exceeds `usize::MAX`.
    pub fn read(
        file_data: &[u8],
        message: &SharedMessageTableMessage,
        offset_size: u8,
    ) -> Result<Self, FormatError> {
        let at = message.table_address.get().to_usize()?;
        let len = table_len(message.index_count, offset_size);
        bytes::ensure_len(file_data, at, len)?;
        Self::parse(&file_data[at..at + len], message.index_count, offset_size)
    }

    /// Parses the master table at the address `message` holds, in `source`.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
    pub fn read_from_source(
        source: &(impl MetadataSource + ?Sized),
        message: &SharedMessageTableMessage,
        offset_size: u8,
    ) -> Result<Self, FormatError> {
        let len = table_len(message.index_count, offset_size);
        let image = source.read_metadata_at(message.table_address.get(), len)?;
        Self::parse(&image, message.index_count, offset_size)
    }

    /// Returns the index that covers `message_type`, or `None` if no index covers it.
    ///
    /// A message type is in one index at most, and the function returns the first index that
    /// covers it, as `H5SM__get_index` in `H5SM.c` (HDF5 2.2.0) does.
    pub fn index_for(&self, message_type: MessageType) -> Option<&SohmIndexHeader> {
        self.indexes.iter().find(|index| index.covers(message_type))
    }
}

/// The location of a shared message, as a record of its index gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SohmLocation {
    /// The message is in the fractal heap of the index.
    Heap {
        /// The number of times the file uses the message. The C library removes the message and its
        /// record when the count reaches 0.
        reference_count: u32,
        /// The heap ID of the message, as a shared message reference holds it.
        heap_id: [u8; FHEAP_ID_LEN],
    },
    /// The message is in the object header of the one object that uses it. The C library moves
    /// the message to the heap when a second object uses it.
    ObjectHeader {
        /// The message type ID.
        message_type: u8,
        /// The creation index of the message in the object header, which tells it apart from
        /// another message of its type in the header.
        creation_index: u16,
        /// The address of the object header.
        address: StoredAddress,
    },
}

/// One record of a shared message index, a list or a B-tree.
///
/// The record is defined in "Shared Object Header Message Table" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_sohm
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmRecord {
    /// The hash of the encoded message, which the index sorts and searches by.
    pub hash: u32,
    /// The location of the message.
    pub location: SohmLocation,
}

impl SohmRecord {
    /// Parses the record at the start of `data`.
    ///
    /// `offset_size` is the superblock's "Size of Offsets" byte. The function reads the fields in
    /// the order of `H5SM__message_decode` in `H5SMmessage.c` (HDF5 2.2.0).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` ends before the fields of the record,
    /// [`FormatError::InvalidSohmRecordLocation`] if the location byte is not 0 or 1, and
    /// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<Self, FormatError> {
        bytes::ensure_len(data, 0, RECORD_PREFIX_LEN)?;
        let hash = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
        let location = match data[0] {
            LOCATION_HEAP => {
                bytes::ensure_len(data, RECORD_PREFIX_LEN, HEAP_LOCATION_LEN)?;
                let at = RECORD_PREFIX_LEN;
                let reference_count =
                    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
                let mut heap_id = [0u8; FHEAP_ID_LEN];
                heap_id.copy_from_slice(&data[at + 4..at + 4 + FHEAP_ID_LEN]);
                SohmLocation::Heap {
                    reference_count,
                    heap_id,
                }
            }
            LOCATION_OBJECT_HEADER => {
                let at = RECORD_PREFIX_LEN;
                bytes::ensure_len(data, at, 4)?;
                let message_type = data[at + 1];
                let creation_index = u16::from_le_bytes([data[at + 2], data[at + 3]]);
                let address = StoredAddress::new(bytes::read_offset(data, at + 4, offset_size)?);
                SohmLocation::ObjectHeader {
                    message_type,
                    creation_index,
                    address,
                }
            }
            other => return Err(FormatError::InvalidSohmRecordLocation(other)),
        };
        Ok(Self { hash, location })
    }
}

/// Parses the list index at the start of `image`: the signature, `message_count` records, and the
/// checksum after them.
///
/// A file allocates a list for [`list_max`](SohmIndexHeader::list_max) records, and the checksum
/// follows the records in use, so the function finds the end of the list from `message_count`.
/// `offset_size` is the superblock's "Size of Offsets" byte. With the `checksum` feature, the
/// function also verifies the list's checksum. The list is defined in "Shared Object Header
/// Message Table" of the [format specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `image` is shorter than the list,
/// [`FormatError::InvalidSohmListSignature`] if the list does not begin with `SMLI`, the errors
/// [`SohmRecord::parse`] returns for a record, and, with the `checksum` feature,
/// [`FormatError::ChecksumMismatch`] if the stored checksum differs from the computed one.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_sohm
pub fn parse_sohm_list(
    image: &[u8],
    message_count: u16,
    offset_size: u8,
) -> Result<Vec<SohmRecord>, FormatError> {
    let stride = sohm_record_len(offset_size);
    let expected = LIST_FIXED_LEN + message_count as usize * stride;
    bytes::ensure_len(image, 0, expected)?;
    let image = &image[..expected];
    if &image[..4] != LIST_SIGNATURE {
        return Err(FormatError::InvalidSohmListSignature);
    }
    checksum::verify_trailing(image)?;

    let mut records = Vec::with_capacity(message_count as usize);
    for i in 0..message_count as usize {
        let at = 4 + i * stride;
        records.push(SohmRecord::parse(&image[at..at + stride], offset_size)?);
    }
    Ok(records)
}

/// Returns the size in bytes of a list index of `message_count` records, its signature and
/// checksum included.
pub fn sohm_list_len(message_count: u16, offset_size: u8) -> usize {
    LIST_FIXED_LEN + message_count as usize * sohm_record_len(offset_size)
}

#[cfg(test)]
mod tests {
    use test_util::image::Image;
    use test_util::sohm;
    use test_util::widths::Widths;

    use super::*;

    /// Returns a master table with one header per index of `indexes`, checksum included.
    fn table_image(indexes: &[SohmIndexHeader]) -> Vec<u8> {
        let indexes: Vec<_> = indexes
            .iter()
            .map(|index| sohm::Index {
                kind: match index.kind {
                    SohmIndexKind::List => sohm::Kind::LIST,
                    SohmIndexKind::BTree => sohm::Kind::BTREE,
                },
                message_type_flags: index.message_type_flags,
                min_message_size: index.min_message_size,
                list_max: index.list_max,
                btree_min: index.btree_min,
                message_count: index.message_count,
                index_address: index.index_address.map(StoredAddress::get),
                heap_address: index.heap_address.map(StoredAddress::get),
            })
            .collect();
        sohm::table(&indexes, Widths::EIGHT)
    }

    /// Recomputes the checksum of an image a test has changed, so the parse reaches the field the
    /// test changed.
    fn reseal(image: &mut [u8]) {
        test_util::checksum::restamp(image, 0, image.len());
    }

    fn sample_index() -> SohmIndexHeader {
        SohmIndexHeader {
            // Dataspace (bit 1), datatype (bit 3), attribute (bit 12).
            message_type_flags: (1 << 1) | (1 << 3) | (1 << 12),
            min_message_size: 250,
            list_max: 50,
            btree_min: 40,
            message_count: 2,
            kind: SohmIndexKind::List,
            index_address: Some(StoredAddress::new(0x1234)),
            heap_address: Some(StoredAddress::new(0x5678)),
        }
    }

    /// The count follows the address, and a parse of the count after the version returns the low
    /// byte of the address.
    #[test]
    fn the_table_message_reads_the_count_after_the_address() {
        let mut data = vec![0u8];
        data.extend_from_slice(&0x400u64.to_le_bytes());
        data.push(3);

        let message = SharedMessageTableMessage::parse(&data, 8).unwrap();
        assert_eq!(message.table_address, StoredAddress::new(0x400));
        assert_eq!(message.index_count, 3);
    }

    #[test]
    fn a_table_message_with_no_indexes_is_an_invalid_index_count() {
        let mut data = vec![0u8];
        data.extend_from_slice(&0x400u64.to_le_bytes());
        data.push(0);

        assert_eq!(
            SharedMessageTableMessage::parse(&data, 8).unwrap_err(),
            FormatError::InvalidSohmIndexCount(0)
        );
    }

    #[test]
    fn a_table_message_past_the_index_maximum_is_an_invalid_index_count() {
        let mut data = vec![0u8];
        data.extend_from_slice(&0x400u64.to_le_bytes());
        data.push(MAX_INDEXES + 1);

        assert_eq!(
            SharedMessageTableMessage::parse(&data, 8).unwrap_err(),
            FormatError::InvalidSohmIndexCount(MAX_INDEXES + 1)
        );
    }

    #[test]
    fn a_table_message_of_another_version_is_an_invalid_version() {
        let mut data = vec![1u8];
        data.extend_from_slice(&0x400u64.to_le_bytes());
        data.push(1);

        assert_eq!(
            SharedMessageTableMessage::parse(&data, 8).unwrap_err(),
            FormatError::InvalidSohmTableVersion(1)
        );
    }

    #[test]
    fn an_index_header_round_trips_through_its_image() {
        let index = sample_index();
        let table = SohmTable::parse(&table_image(std::slice::from_ref(&index)), 1, 8).unwrap();
        assert_eq!(table.indexes, vec![index]);
    }

    /// A reader that swaps the two bytes reads version 1 for a B-tree index and returns a version
    /// error.
    #[test]
    fn the_version_byte_precedes_the_index_type_byte() {
        let mut index = sample_index();
        index.kind = SohmIndexKind::BTree;
        let image = table_image(&[index]);

        assert_eq!(image[4], SOHM_VERSION);
        assert_eq!(image[5], 1);
        assert_eq!(
            SohmTable::parse(&image, 1, 8).unwrap().indexes[0].kind,
            SohmIndexKind::BTree
        );
    }

    #[test]
    fn an_undefined_index_or_heap_address_reads_as_none() {
        let mut index = sample_index();
        index.index_address = None;
        index.heap_address = None;
        let table = SohmTable::parse(&table_image(&[index]), 1, 8).unwrap();
        assert_eq!(table.indexes[0].index_address, None);
        assert_eq!(table.indexes[0].heap_address, None);
    }

    #[test]
    fn a_table_with_a_bad_signature_is_an_invalid_signature() {
        let mut image = table_image(&[sample_index()]);
        image[0] = b'X';
        assert_eq!(
            SohmTable::parse(&image, 1, 8).unwrap_err(),
            FormatError::InvalidSohmTableSignature
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn a_table_whose_checksum_disagrees_is_a_checksum_mismatch() {
        let mut image = table_image(&[sample_index()]);
        // Flip a bit in the list maximum, which no other field repeats.
        image[12] ^= 0x01;
        assert!(matches!(
            SohmTable::parse(&image, 1, 8).unwrap_err(),
            FormatError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn an_index_kind_the_format_does_not_define_is_an_invalid_index_kind() {
        let mut image = table_image(&[sample_index()]);
        image[5] = 2;
        reseal(&mut image);
        assert_eq!(
            SohmTable::parse(&image, 1, 8).unwrap_err(),
            FormatError::InvalidSohmIndexKind(2)
        );
    }

    #[test]
    fn a_table_shorter_than_its_index_count_is_an_unexpected_eof() {
        let image = table_image(&[sample_index()]);
        assert!(matches!(
            SohmTable::parse(&image, 2, 8).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn an_index_covers_exactly_the_types_whose_flag_bits_are_set() {
        let index = sample_index();
        assert!(index.covers(MessageType::DATASPACE));
        assert!(index.covers(MessageType::DATATYPE));
        assert!(index.covers(MessageType::ATTRIBUTE));
        assert!(!index.covers(MessageType::FILL_VALUE));
        assert!(!index.covers(MessageType::FILTER_PIPELINE));
        // Past the flag word entirely, so not representable and not covered.
        assert!(!index.covers(MessageType::ATTRIBUTE_INFO));
    }

    #[test]
    fn the_table_picks_the_index_covering_a_type() {
        let mut datatypes = sample_index();
        datatypes.message_type_flags = 1 << 3;
        let mut attributes = sample_index();
        attributes.message_type_flags = 1 << 12;
        attributes.heap_address = Some(StoredAddress::new(0x9999));
        let table = SohmTable {
            indexes: vec![datatypes, attributes],
        };

        assert_eq!(
            table
                .index_for(MessageType::ATTRIBUTE)
                .unwrap()
                .heap_address,
            Some(StoredAddress::new(0x9999))
        );
        assert_eq!(
            table.index_for(MessageType::DATATYPE).unwrap().heap_address,
            Some(StoredAddress::new(0x5678))
        );
        assert!(table.index_for(MessageType::FILL_VALUE).is_none());
    }

    /// A list walk that sized each record by its own shape would read every record after the first
    /// object-header record at the wrong offset, in a file with 2- or 4-byte addresses.
    #[test]
    fn both_record_shapes_share_one_stride() {
        assert_eq!(sohm_record_len(8), 17);
        assert_eq!(sohm_record_len(4), 17);
        // A 16-byte address makes the object-header shape the wider one.
        assert_eq!(sohm_record_len(16), 25);
    }

    #[test]
    fn a_heap_record_parses_to_its_reference_count_and_heap_id() {
        let id = [1, 2, 3, 4, 5, 6, 7, 8];
        let record =
            SohmRecord::parse(&sohm::heap_record(0xDEAD_BEEF, 3, id, Widths::EIGHT), 8).unwrap();
        assert_eq!(record.hash, 0xDEADBEEF);
        assert_eq!(
            record.location,
            SohmLocation::Heap {
                reference_count: 3,
                heap_id: id,
            }
        );
    }

    /// A reader that reads the reserved byte as the type ID reports type 0 (Nil) for every
    /// object-header record.
    #[test]
    fn an_object_header_record_skips_its_reserved_byte() {
        let mut data = vec![LOCATION_OBJECT_HEADER];
        data.extend_from_slice(&7u32.to_le_bytes());
        data.push(0); // reserved
        data.push(0x0C); // attribute
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&0x2000u64.to_le_bytes());

        let record = SohmRecord::parse(&data, 8).unwrap();
        assert_eq!(record.hash, 7);
        assert_eq!(
            record.location,
            SohmLocation::ObjectHeader {
                message_type: 0x0C,
                creation_index: 4,
                address: StoredAddress::new(0x2000),
            }
        );
    }

    #[test]
    fn a_record_location_the_format_does_not_define_is_an_invalid_record_location() {
        let mut data = vec![9u8];
        data.extend_from_slice(&[0u8; 16]);
        assert_eq!(
            SohmRecord::parse(&data, 8).unwrap_err(),
            FormatError::InvalidSohmRecordLocation(9)
        );
    }

    #[test]
    fn a_list_index_reads_every_record_it_declares() {
        let first = sohm::heap_record(0xDEAD_BEEF, 1, [1, 0, 0, 0, 0, 0, 0, 0], Widths::EIGHT);
        let second = sohm::heap_record(0xDEAD_BEEF, 2, [2, 0, 0, 0, 0, 0, 0, 0], Widths::EIGHT);
        let image = sohm::list(&[first, second]);

        let records = parse_sohm_list(&image, 2, 8).unwrap();
        assert_eq!(sohm_list_len(2, 8), image.len());
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[1].location,
            SohmLocation::Heap {
                reference_count: 2,
                heap_id: [2, 0, 0, 0, 0, 0, 0, 0],
            }
        );
    }

    /// The block of a list has room for `list_max` records, and the checksum follows the records
    /// in use, so a checksum over the whole block fails for every list that is not full.
    #[test]
    fn a_list_checksum_covers_only_the_records_in_use() {
        let record = sohm::heap_record(0xDEAD_BEEF, 1, [1, 0, 0, 0, 0, 0, 0, 0], Widths::EIGHT);
        let mut image = sohm::list(&[record]);
        // The unused end of the block, zeroed.
        image.extend_from_slice(&vec![0u8; 4 * sohm_record_len(8)]);

        assert_eq!(parse_sohm_list(&image, 1, 8).unwrap().len(), 1);
    }

    #[test]
    fn a_list_with_a_bad_signature_is_an_invalid_signature() {
        let mut image = sohm::list(&[sohm::heap_record(0xDEAD_BEEF, 1, [0; 8], Widths::EIGHT)]);
        image[0] = b'X';
        assert_eq!(
            parse_sohm_list(&image, 1, 8).unwrap_err(),
            FormatError::InvalidSohmListSignature
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn a_list_whose_checksum_disagrees_is_a_checksum_mismatch() {
        let mut image = sohm::list(&[sohm::heap_record(0xDEAD_BEEF, 1, [0; 8], Widths::EIGHT)]);
        image[6] ^= 0x01;
        assert!(matches!(
            parse_sohm_list(&image, 1, 8).unwrap_err(),
            FormatError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn a_list_shorter_than_its_record_count_is_an_unexpected_eof() {
        let image = sohm::list(&[sohm::heap_record(0xDEAD_BEEF, 1, [0; 8], Widths::EIGHT)]);
        assert!(matches!(
            parse_sohm_list(&image, 4, 8).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn a_table_is_read_from_the_address_its_message_stores() {
        let table = sohm::table(
            &[sohm::Index {
                kind: sohm::Kind::LIST,
                message_type_flags: 1 << 3,
                min_message_size: 250,
                list_max: 50,
                btree_min: 40,
                message_count: 1,
                index_address: Some(0x100),
                heap_address: None,
            }],
            Widths::EIGHT,
        );
        let mut image = Image::new();
        image.place(0x40, &table);
        let file = image.build();
        let table_message = SharedMessageTableMessage {
            table_address: StoredAddress::new(0x40),
            index_count: 1,
        };
        let expected_table = SohmTable {
            indexes: vec![SohmIndexHeader {
                message_type_flags: 1 << 3,
                min_message_size: 250,
                list_max: 50,
                btree_min: 40,
                message_count: 1,
                kind: SohmIndexKind::List,
                index_address: Some(StoredAddress::new(0x100)),
                heap_address: None,
            }],
        };

        assert_eq!(
            SohmTable::read(&file, &table_message, 8),
            Ok(expected_table.clone())
        );
        assert_eq!(
            SohmTable::read_from_source(file.as_slice(), &table_message, 8),
            Ok(expected_table)
        );
    }
}
