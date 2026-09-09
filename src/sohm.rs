//! HDF5 shared object header message (SOHM) storage.
//!
//! A file created with `H5Pset_shared_mesg_nindexes` /
//! `H5Pset_shared_mesg_index` stores one copy of each message it shares in a
//! *shared-message heap* instead of in the objects that use it. Every user then
//! carries an eight-byte fractal-heap ID in place of the message body, which
//! [`crate::shared_message`] parses as [`SharedLocation::SohmHeap`]. This module
//! is what turns that ID back into the message.
//!
//! Three structures stand between the ID and the bytes:
//!
//! - the **Shared Message Table** message (type 0x000F) in the superblock
//!   extension, which names the master table's address and how many indexes it
//!   holds ([`SharedMessageTableMessage`]);
//! - the **master table** itself, signature `SMTB`, one [`SohmIndexHeader`] per
//!   index. An index header says which message types the index covers, where its
//!   fractal heap is, and where its own index lives ([`SohmTable`]);
//! - the **index**, which is either an unsorted list (signature `SMLI`) or a
//!   version 2 B-tree of type 7, holding one [`SohmRecord`] per shared message.
//!
//! Reading a message needs only the first two: the fractal-heap ID in the
//! reference locates it directly, and the index exists so that a *writer* can
//! find an equal message to share. The index is read for a different question —
//! what a commit would strand. A record may name an object *header* rather than
//! a heap entry, and that address is one no rewrite repoints, so
//! [`crate::edit`] walks the records of every index before it moves or removes
//! anything.
//!
//! The layouts follow `H5SM__cache_table_deserialize`, `H5SM__cache_list_deserialize`
//! and `H5SM__message_decode` in the reference C library, which are the authority
//! on the field order — the published format description lists an index header's
//! version byte after its index-type byte, and the library writes it before.
//!
//! [`SharedLocation::SohmHeap`]: crate::shared_message::SharedLocation::SohmHeap

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::btree_v2::{BTreeV2Header, collect_btree_v2_records_from_source};
use crate::bytes::{ensure_len, read_offset};
use crate::convert::{TryToUsize, is_undefined_addr};
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::message_type::MessageType;
use crate::shared_message::FHEAP_ID_LEN;
use crate::source::Source;

/// Signature of the shared-message master table (`H5SM_TABLE_MAGIC`).
const TABLE_SIGNATURE: &[u8; 4] = b"SMTB";

/// Signature of a shared-message list index (`H5SM_LIST_MAGIC`).
const LIST_SIGNATURE: &[u8; 4] = b"SMLI";

/// The only version of the Shared Message Table message, of an index header, and
/// of a list index (`HDF5_SHAREDHEADER_VERSION` / `H5SM_LIST_VERSION`).
const SOHM_VERSION: u8 = 0;

/// B-tree v2 type of a shared-message index (`H5B2_SOHM_INDEX_ID`).
const BTREE_SOHM_INDEX_TYPE: u8 = 7;

/// Most indexes a file may declare (`H5O_SHMESG_MAX_NINDEXES`). A larger count
/// would size the master table's read off a byte the file chose.
const MAX_INDEXES: u8 = 8;

/// Bytes of a master table that are not index headers: the signature and the
/// trailing checksum.
const TABLE_FIXED_LEN: usize = 4 + 4;

/// Bytes of an index header that do not depend on the file's address width:
/// version(1) + index type(1) + message type flags(2) + minimum message size(4)
/// + list maximum(2) + B-tree minimum(2) + message count(2).
const INDEX_HEADER_FIXED_LEN: usize = 14;

/// Bytes a list index spends outside its records: the signature and the
/// checksum that follows the records actually in use.
const LIST_FIXED_LEN: usize = 4 + 4;

/// Bytes of a record that do not depend on the file's address width:
/// location(1) + hash(4).
const RECORD_PREFIX_LEN: usize = 5;

/// Bytes a heap-stored record's body occupies: reference count(4) + heap ID(8).
const HEAP_LOCATION_LEN: usize = 4 + FHEAP_ID_LEN;

/// Record location byte for a message in the shared-message heap
/// (`H5SM_IN_HEAP`).
const LOCATION_HEAP: u8 = 0;

/// Record location byte for a message left in an object header
/// (`H5SM_IN_OH`) — how the *first* user of a shared message is recorded, before
/// a second one moves it into the heap.
const LOCATION_OBJECT_HEADER: u8 = 1;

/// The Shared Message Table message (type 0x000F), carried by the superblock
/// extension of a file that shares messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessageTableMessage {
    /// Address of the master table.
    pub table_address: u64,
    /// How many indexes the master table holds. Fixed for the life of the file:
    /// the table's own size depends on it, so it is stored here rather than in
    /// the table.
    pub index_count: u8,
}

impl SharedMessageTableMessage {
    /// Parse the message body: version(1) + table address(offset size) +
    /// index count(1).
    pub fn parse(data: &[u8], offset_size: u8) -> Result<Self, FormatError> {
        ensure_len(data, 0, 1)?;
        let version = data[0];
        if version != SOHM_VERSION {
            return Err(FormatError::InvalidSohmTableVersion(version));
        }
        let table_address = read_offset(data, 1, offset_size)?;
        let pos = 1 + offset_size as usize;
        ensure_len(data, pos, 1)?;
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

/// How an index stores its records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SohmIndexKind {
    /// An unsorted list, signature `SMLI`. What a file starts with, and what it
    /// returns to once the index falls back to the B-tree minimum.
    List,
    /// A version 2 B-tree of type 7, which the library switches to once the list
    /// would exceed its maximum.
    BTree,
}

/// One index of the master table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmIndexHeader {
    /// Which message types this index covers, as a bit per raw message type ID
    /// (`H5O_SHMESG_*_FLAG`): bit 1 dataspace, 3 datatype, 5 fill value, 11
    /// filter pipeline, 12 attribute.
    pub message_type_flags: u16,
    /// Messages smaller than this are not shared.
    pub min_message_size: u32,
    /// Record count at which a list index becomes a B-tree.
    pub list_max: u16,
    /// Record count at which a B-tree index becomes a list again.
    pub btree_min: u16,
    /// How many records the index holds.
    pub message_count: u16,
    /// Whether [`index_address`](Self::index_address) names a list or a B-tree.
    pub kind: SohmIndexKind,
    /// Address of the list or B-tree, or `None` while the index holds nothing.
    pub index_address: Option<u64>,
    /// Address of the fractal heap holding this index's message bodies, or
    /// `None` while the index holds nothing.
    pub heap_address: Option<u64>,
}

impl SohmIndexHeader {
    /// Whether this index covers `message_type`.
    ///
    /// The flag word indexes by raw message type ID, so a type past bit 15 is
    /// not representable and cannot be covered.
    pub fn covers(&self, message_type: MessageType) -> bool {
        match u32::from(message_type.to_u16()) {
            bit if bit < 16 => self.message_type_flags & (1u16 << bit) != 0,
            _ => false,
        }
    }
}

/// The master table of shared-message indexes, signature `SMTB`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmTable {
    /// The indexes, in the order the table stores them.
    pub indexes: Vec<SohmIndexHeader>,
}

/// Encoded size of one index header.
fn index_header_len(offset_size: u8) -> usize {
    INDEX_HEADER_FIXED_LEN + 2 * offset_size as usize
}

/// Encoded size of a master table with `index_count` indexes.
fn table_len(index_count: u8, offset_size: u8) -> usize {
    TABLE_FIXED_LEN + index_count as usize * index_header_len(offset_size)
}

/// Encoded size of one shared-message record.
///
/// A record is one of two shapes and the wider one sizes both, so that a list's
/// entries stay a fixed stride (`H5SM_SOHM_ENTRY_SIZE`).
pub fn record_len(offset_size: u8) -> usize {
    let object_header_location = 1 + 1 + 2 + offset_size as usize;
    RECORD_PREFIX_LEN + HEAP_LOCATION_LEN.max(object_header_location)
}

/// Verify the trailing Jenkins checksum of a metadata block whose last four
/// bytes hold it.
#[cfg_attr(not(feature = "checksum"), allow(unused_variables))]
fn verify_checksum(image: &[u8]) -> Result<(), FormatError> {
    #[cfg(feature = "checksum")]
    {
        let split = image.len() - 4;
        let stored = u32::from_le_bytes([
            image[split],
            image[split + 1],
            image[split + 2],
            image[split + 3],
        ]);
        let computed = crate::checksum::jenkins_lookup3(&image[..split]);
        if computed != stored {
            return Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }
    }
    Ok(())
}

impl SohmTable {
    /// Parse a master table from its exact on-disk image.
    ///
    /// `index_count` comes from the Shared Message Table message rather than
    /// from the image: the table stores no count of its own, so a caller that
    /// guessed one would read a neighbouring structure as an index header.
    pub fn parse(image: &[u8], index_count: u8, offset_size: u8) -> Result<Self, FormatError> {
        if index_count == 0 || index_count > MAX_INDEXES {
            return Err(FormatError::InvalidSohmIndexCount(index_count));
        }
        let expected = table_len(index_count, offset_size);
        ensure_len(image, 0, expected)?;
        let image = &image[..expected];
        if &image[..4] != TABLE_SIGNATURE {
            return Err(FormatError::InvalidSohmTableSignature);
        }
        verify_checksum(image)?;

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
            let index_address = optional_address(image, at, offset_size)?;
            at += offset_size as usize;
            let heap_address = optional_address(image, at, offset_size)?;
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

    /// Read and parse the master table `message` names out of a whole-file image
    /// framed at the file's base address.
    pub fn read(
        file_data: &[u8],
        message: &SharedMessageTableMessage,
        offset_size: u8,
    ) -> Result<Self, FormatError> {
        let at = message.table_address.to_usize()?;
        let len = table_len(message.index_count, offset_size);
        ensure_len(file_data, at, len)?;
        Self::parse(&file_data[at..at + len], message.index_count, offset_size)
    }

    /// Streaming counterpart of [`Self::read`].
    pub fn read_from_source<S: Source + ?Sized>(
        source: &S,
        message: &SharedMessageTableMessage,
        offset_size: u8,
    ) -> Result<Self, FormatError> {
        let len = table_len(message.index_count, offset_size);
        let image = source.read_metadata_at(message.table_address, len)?;
        Self::parse(&image, message.index_count, offset_size)
    }

    /// The index covering `message_type`, or `None` when no index does.
    ///
    /// The reference C library takes the first match (`H5SM__get_index`), and a
    /// message type may appear in only one index, so first-match is exact rather
    /// than a preference.
    pub fn index_for(&self, message_type: MessageType) -> Option<&SohmIndexHeader> {
        self.indexes.iter().find(|index| index.covers(message_type))
    }
}

/// Read an address field that may carry the all-ones undefined marker.
fn optional_address(data: &[u8], pos: usize, offset_size: u8) -> Result<Option<u64>, FormatError> {
    let addr = read_offset(data, pos, offset_size)?;
    Ok((!is_undefined_addr(addr, offset_size)).then_some(addr))
}

/// Where an index says a shared message is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SohmLocation {
    /// In the index's fractal heap, under this ID, used by this many objects.
    Heap {
        /// How many messages in the file reference this one. The reference C
        /// library reclaims the heap entry and the record when a deletion takes
        /// it to zero; this crate does not write shared messages and does not
        /// change the count, so an edit that would have to is refused.
        reference_count: u32,
        /// The fractal-heap ID, as a shared reference carries it.
        heap_id: [u8; FHEAP_ID_LEN],
    },
    /// Still in the object header that wrote it. The reference C library records
    /// a shareable message this way when it is written into an object header
    /// that is already open, and moves it into the heap when a second user
    /// appears — so a record of this shape means exactly one user, and its
    /// address is a stored address like any other.
    ObjectHeader {
        /// Raw type ID of the message within that header.
        message_type: u8,
        /// The message's creation index within that header, which is what
        /// distinguishes it from another message of the same type there.
        creation_index: u16,
        /// Address of the object header.
        address: u64,
    },
}

/// One record of a shared-message index, in either index kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmRecord {
    /// Hash of the encoded message, which is what the index sorts and searches
    /// on.
    pub hash: u32,
    /// Where the message itself is.
    pub location: SohmLocation,
}

impl SohmRecord {
    /// Parse one record from the start of `data`.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<Self, FormatError> {
        ensure_len(data, 0, RECORD_PREFIX_LEN)?;
        let hash = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
        let location = match data[0] {
            LOCATION_HEAP => {
                ensure_len(data, RECORD_PREFIX_LEN, HEAP_LOCATION_LEN)?;
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
                ensure_len(data, at, 4)?;
                let message_type = data[at + 1];
                let creation_index = u16::from_le_bytes([data[at + 2], data[at + 3]]);
                let address = read_offset(data, at + 4, offset_size)?;
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

/// Parse a list index from its exact on-disk image — signature, `message_count`
/// records, and the checksum that follows them.
///
/// The block a file allocates for a list is wider than this: it is sized for
/// `list_max` records so the index can grow in place. The checksum covers only
/// the records in use, which is why the count rather than the allocation decides
/// where the image ends.
pub fn parse_list(
    image: &[u8],
    message_count: u16,
    offset_size: u8,
) -> Result<Vec<SohmRecord>, FormatError> {
    let stride = record_len(offset_size);
    let expected = LIST_FIXED_LEN + message_count as usize * stride;
    ensure_len(image, 0, expected)?;
    let image = &image[..expected];
    if &image[..4] != LIST_SIGNATURE {
        return Err(FormatError::InvalidSohmListSignature);
    }
    verify_checksum(image)?;

    let mut records = Vec::with_capacity(message_count as usize);
    for i in 0..message_count as usize {
        let at = 4 + i * stride;
        records.push(SohmRecord::parse(&image[at..at + stride], offset_size)?);
    }
    Ok(records)
}

/// Encoded size of a list index holding `message_count` records, checksum
/// included.
fn list_image_len(message_count: u16, offset_size: u8) -> usize {
    LIST_FIXED_LEN + message_count as usize * record_len(offset_size)
}

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
            let len = list_image_len(index.message_count, offset_size);
            let image = source.read_metadata_at(address, len)?;
            parse_list(&image, index.message_count, offset_size)
        }
        SohmIndexKind::BTree => {
            let header =
                BTreeV2Header::parse_from_source(source, address, offset_size, length_size)?;
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
) -> Result<(&SohmIndexHeader, u64), FormatError> {
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
        heap_address.to_usize()?,
        offset_size,
        length_size,
    )?;
    heap.object_reader(offset_size, length_size)
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
    let heap =
        FractalHeapHeader::parse_from_source(source, heap_address, offset_size, length_size)?;
    heap.object_reader(offset_size, length_size)
        .read_from_source(source, &heap_id[..heap.heap_id_length as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a master-table image with one index header, checksum included, so a
    /// parse test states the bytes rather than a fixture file.
    fn table_image(indexes: &[SohmIndexHeader]) -> Vec<u8> {
        let mut image = Vec::from(*TABLE_SIGNATURE);
        for index in indexes {
            image.push(SOHM_VERSION);
            image.push(match index.kind {
                SohmIndexKind::List => 0,
                SohmIndexKind::BTree => 1,
            });
            image.extend_from_slice(&index.message_type_flags.to_le_bytes());
            image.extend_from_slice(&index.min_message_size.to_le_bytes());
            image.extend_from_slice(&index.list_max.to_le_bytes());
            image.extend_from_slice(&index.btree_min.to_le_bytes());
            image.extend_from_slice(&index.message_count.to_le_bytes());
            image.extend_from_slice(&index.index_address.unwrap_or(u64::MAX).to_le_bytes());
            image.extend_from_slice(&index.heap_address.unwrap_or(u64::MAX).to_le_bytes());
        }
        let checksum = crate::checksum::jenkins_lookup3(&image);
        image.extend_from_slice(&checksum.to_le_bytes());
        image
    }

    /// Recompute the trailing checksum of an image a test has edited, so the
    /// test exercises the field it changed rather than the checksum.
    fn reseal(image: &mut [u8]) {
        let split = image.len() - 4;
        let checksum = crate::checksum::jenkins_lookup3(&image[..split]);
        image[split..].copy_from_slice(&checksum.to_le_bytes());
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
            index_address: Some(0x1234),
            heap_address: Some(0x5678),
        }
    }

    /// The message in the superblock extension is version, address, count — the
    /// count last, where a reader that expected it beside the version would take
    /// the low byte of the address for it.
    #[test]
    fn the_table_message_reads_the_count_after_the_address() {
        let mut data = vec![0u8];
        data.extend_from_slice(&0x400u64.to_le_bytes());
        data.push(3);

        let message = SharedMessageTableMessage::parse(&data, 8).unwrap();
        assert_eq!(message.table_address, 0x400);
        assert_eq!(message.index_count, 3);
    }

    #[test]
    fn a_table_message_with_no_indexes_is_refused() {
        let mut data = vec![0u8];
        data.extend_from_slice(&0x400u64.to_le_bytes());
        data.push(0);

        assert_eq!(
            SharedMessageTableMessage::parse(&data, 8).unwrap_err(),
            FormatError::InvalidSohmIndexCount(0)
        );
    }

    #[test]
    fn a_table_message_past_the_index_maximum_is_refused() {
        let mut data = vec![0u8];
        data.extend_from_slice(&0x400u64.to_le_bytes());
        data.push(MAX_INDEXES + 1);

        assert_eq!(
            SharedMessageTableMessage::parse(&data, 8).unwrap_err(),
            FormatError::InvalidSohmIndexCount(MAX_INDEXES + 1)
        );
    }

    #[test]
    fn a_table_message_of_another_version_is_refused() {
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

    /// The version byte comes before the index-type byte, which the published
    /// format description has the other way round. Reading them swapped turns a
    /// list into a version error and a B-tree into a list.
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
    fn a_table_with_a_bad_signature_is_refused() {
        let mut image = table_image(&[sample_index()]);
        image[0] = b'X';
        assert_eq!(
            SohmTable::parse(&image, 1, 8).unwrap_err(),
            FormatError::InvalidSohmTableSignature
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn a_table_whose_checksum_disagrees_is_refused() {
        let mut image = table_image(&[sample_index()]);
        // Flip a bit in the list maximum, which no other field repeats.
        image[12] ^= 0x01;
        assert!(matches!(
            SohmTable::parse(&image, 1, 8).unwrap_err(),
            FormatError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn an_index_kind_the_format_does_not_define_is_refused() {
        let mut image = table_image(&[sample_index()]);
        image[5] = 2;
        reseal(&mut image);
        assert_eq!(
            SohmTable::parse(&image, 1, 8).unwrap_err(),
            FormatError::InvalidSohmIndexKind(2)
        );
    }

    /// A table sized for more indexes than it holds runs off the end rather than
    /// reading whatever follows it as an index header.
    #[test]
    fn a_table_shorter_than_its_index_count_is_refused() {
        let image = table_image(&[sample_index()]);
        assert!(matches!(
            SohmTable::parse(&image, 2, 8).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn an_index_covers_exactly_the_types_its_flags_name() {
        let index = sample_index();
        assert!(index.covers(MessageType::Dataspace));
        assert!(index.covers(MessageType::Datatype));
        assert!(index.covers(MessageType::Attribute));
        assert!(!index.covers(MessageType::FillValue));
        assert!(!index.covers(MessageType::FilterPipeline));
        // Past the flag word entirely, so not representable and not covered.
        assert!(!index.covers(MessageType::AttributeInfo));
    }

    #[test]
    fn the_table_picks_the_index_covering_a_type() {
        let mut datatypes = sample_index();
        datatypes.message_type_flags = 1 << 3;
        let mut attributes = sample_index();
        attributes.message_type_flags = 1 << 12;
        attributes.heap_address = Some(0x9999);
        let table = SohmTable {
            indexes: vec![datatypes, attributes],
        };

        assert_eq!(
            table
                .index_for(MessageType::Attribute)
                .unwrap()
                .heap_address,
            Some(0x9999)
        );
        assert_eq!(
            table.index_for(MessageType::Datatype).unwrap().heap_address,
            Some(0x5678)
        );
        assert!(table.index_for(MessageType::FillValue).is_none());
    }

    /// Both record shapes are stored at the same stride, so a list walk that
    /// sized entries by their own shape would drift after the first heap record.
    #[test]
    fn both_record_shapes_share_one_stride() {
        assert_eq!(record_len(8), 17);
        assert_eq!(record_len(4), 17);
        // A 16-byte address makes the object-header shape the wider one.
        assert_eq!(record_len(16), 25);
    }

    fn heap_record_bytes(reference_count: u32, heap_id: [u8; 8], offset_size: u8) -> Vec<u8> {
        let mut data = vec![LOCATION_HEAP];
        data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        data.extend_from_slice(&reference_count.to_le_bytes());
        data.extend_from_slice(&heap_id);
        data.resize(record_len(offset_size), 0);
        data
    }

    #[test]
    fn a_heap_record_carries_its_reference_count_and_heap_id() {
        let id = [1, 2, 3, 4, 5, 6, 7, 8];
        let record = SohmRecord::parse(&heap_record_bytes(3, id, 8), 8).unwrap();
        assert_eq!(record.hash, 0xDEADBEEF);
        assert_eq!(
            record.location,
            SohmLocation::Heap {
                reference_count: 3,
                heap_id: id,
            }
        );
    }

    /// The object-header shape skips a reserved byte before the type ID; reading
    /// it without the skip reports type 0 (Nil) for every record.
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
                address: 0x2000,
            }
        );
    }

    #[test]
    fn a_record_location_the_format_does_not_define_is_refused() {
        let mut data = vec![9u8];
        data.extend_from_slice(&[0u8; 16]);
        assert_eq!(
            SohmRecord::parse(&data, 8).unwrap_err(),
            FormatError::InvalidSohmRecordLocation(9)
        );
    }

    fn list_image(records: &[Vec<u8>]) -> Vec<u8> {
        let mut image = Vec::from(*LIST_SIGNATURE);
        for record in records {
            image.extend_from_slice(record);
        }
        let checksum = crate::checksum::jenkins_lookup3(&image);
        image.extend_from_slice(&checksum.to_le_bytes());
        image
    }

    #[test]
    fn a_list_index_reads_every_record_it_declares() {
        let first = heap_record_bytes(1, [1, 0, 0, 0, 0, 0, 0, 0], 8);
        let second = heap_record_bytes(2, [2, 0, 0, 0, 0, 0, 0, 0], 8);
        let image = list_image(&[first, second]);

        let records = parse_list(&image, 2, 8).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[1].location,
            SohmLocation::Heap {
                reference_count: 2,
                heap_id: [2, 0, 0, 0, 0, 0, 0, 0],
            }
        );
    }

    /// The block a list lives in is sized for `list_max` records, so the
    /// checksum sits after the records in *use*. Verifying it over the whole
    /// allocation would fail every list that is not full.
    #[test]
    fn a_list_checksum_covers_only_the_records_in_use() {
        let record = heap_record_bytes(1, [1, 0, 0, 0, 0, 0, 0, 0], 8);
        let mut image = list_image(&[record]);
        // The unused tail of the allocation, which the file zeroes.
        image.extend_from_slice(&vec![0u8; 4 * record_len(8)]);

        assert_eq!(parse_list(&image, 1, 8).unwrap().len(), 1);
    }

    #[test]
    fn a_list_with_a_bad_signature_is_refused() {
        let mut image = list_image(&[heap_record_bytes(1, [0; 8], 8)]);
        image[0] = b'X';
        assert_eq!(
            parse_list(&image, 1, 8).unwrap_err(),
            FormatError::InvalidSohmListSignature
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn a_list_whose_checksum_disagrees_is_refused() {
        let mut image = list_image(&[heap_record_bytes(1, [0; 8], 8)]);
        image[6] ^= 0x01;
        assert!(matches!(
            parse_list(&image, 1, 8).unwrap_err(),
            FormatError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn a_list_shorter_than_its_record_count_is_refused() {
        let image = list_image(&[heap_record_bytes(1, [0; 8], 8)]);
        assert!(matches!(
            parse_list(&image, 4, 8).unwrap_err(),
            FormatError::UnexpectedEof { .. }
        ));
    }

    /// An index that has been declared but never used names neither a list nor a
    /// B-tree, and holds no records rather than failing.
    #[test]
    fn an_index_with_no_address_holds_no_records() {
        let mut index = sample_index();
        index.index_address = None;
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
        image.extend_from_slice(&list_image(&[heap_record_bytes(
            3,
            [7, 0, 0, 0, 0, 0, 0, 0],
            8,
        )]));
        let mut index = sample_index();
        index.index_address = Some(64);
        index.message_count = 1;

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
            indexes: vec![sample_index()],
        };
        assert_eq!(
            index_for_read(&table, MessageType::FillValue).unwrap_err(),
            FormatError::SohmIndexMissing(MessageType::FillValue.to_u16())
        );
    }

    /// An index that covers the type but has allocated no heap cannot hold the
    /// message either, and says so by the same name rather than parsing address
    /// zero as a fractal heap.
    #[test]
    fn an_index_without_a_heap_cannot_answer_a_lookup() {
        let mut index = sample_index();
        index.heap_address = None;
        let table = SohmTable {
            indexes: vec![index],
        };
        assert_eq!(
            index_for_read(&table, MessageType::Datatype).unwrap_err(),
            FormatError::SohmIndexMissing(MessageType::Datatype.to_u16())
        );
    }
}
