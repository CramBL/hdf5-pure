//! Dense link inspection with heap and index consistency checks.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::AddressRange;
use hdf5_pure_format::__private::BTREE_V2_LINK_CREATION_ORDER;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::BTreeV2LinkNameIndex;
use hdf5_pure_format::__private::BTreeV2Node;
use hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::FractalHeapLayout;
use hdf5_pure_format::__private::GROUP_HEAP_MAX_MANAGED_OBJECT;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::LinkStorageError;
use hdf5_pure_format::__private::OwnedLink;

use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::Error;
use crate::error::FormatError;
use crate::fractal_heap;
use crate::fractal_heap::HeapObjectReader;
use crate::fractal_heap::HeapSource;
use crate::source::StoredMetadata;
use crate::source::StoredRead;
use crate::width::FormatWidths;

/// Returns dense links after checking heap and index consistency.
///
/// Reads heap and index metadata through [`StoredRead::read_metadata`]. Checks name hashes,
/// name-index order, the managed-object count, and any creation-order index.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] for unsupported storage encodings, inconsistent keys or
/// counts, or links the editor cannot reproduce. Returns errors from address resolution,
/// metadata reads, heap and link parsing, or index traversal.
pub(crate) fn read_dense(
    source: &dyn StoredRead,
    widths: FormatWidths,
    info: &LinkInfoMessage,
    heap_address: StoredAddress,
) -> Result<Vec<OwnedLink>, Error> {
    let heap_source = HeapSource::new(source);
    let (header, _) =
        fractal_heap::read_header_and_range_from_source(&heap_source, heap_address, widths)?;
    if header.filtering.is_filtered()
        || header.huge_objects_count != 0
        || header.tiny_objects_count != 0
        || header.huge_objects_size != 0
        || header.tiny_objects_size != 0
        || !header
            .btree_huge_objects_address
            .is_undefined(widths.offsets)
    {
        return Err(Error::EditUnsupported(
            "a group's fractal heap uses filters, huge objects, or tiny objects",
        ));
    }
    if header.flags & !FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED != 0 {
        return Err(Error::EditUnsupported(
            "a group's fractal heap uses unsupported flags",
        ));
    }
    FractalHeapLayout::new(&header, widths.offsets, widths.lengths)
        .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
    let name_address = info
        .btree_name_index_address
        .ok_or(Error::EditUnsupported("a dense group has no name index"))?;
    let name_header = BTreeV2Header::parse_from_source_with_widths(
        &StoredMetadata(source),
        name_address.get(),
        widths,
    )?;
    let index = BTreeV2LinkNameIndex::new(&name_header, widths.offsets, header.heap_id_length)?;
    if name_header.record_size != NAME_RECORD_SIZE || header.heap_id_length != LINK_HEAP_ID_LENGTH {
        return Err(Error::EditUnsupported(
            "a group's name index uses an unsupported record encoding",
        ));
    }
    let ids = FractalHeapIdLayout::new(&header, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(
        heap_source,
        &header,
        heap_address,
        AccessMode::ReadWrite,
        widths,
    );
    let mut links = Vec::new();
    let mut creation = BTreeMap::new();
    let mut previous_hash = None;
    walk_index(source, index.nodes(), |record| {
        let (hash, id) = index.record(record)?;
        let view = ids.parse(id)?;
        if id.first() != Some(&MANAGED_HEAP_ID_FLAGS)
            || !matches!(view.kind(), FractalHeapIdKind::Managed { .. })
        {
            return Err(Error::EditUnsupported(
                "a group's link uses a non-managed heap ID",
            ));
        }
        let bytes = heap.read(view)?;
        let link = OwnedLink::parse(&bytes, widths.offsets).map_err(storage_error)?;
        if link.encoded_len(widths.offsets).map_err(storage_error)?
            > GROUP_HEAP_MAX_MANAGED_OBJECT.to_usize()?
        {
            return Err(Error::EditUnsupported(
                "a group's link exceeds the supported managed object size",
            ));
        }
        if checksum::jenkins_lookup3(link.name()) != hash
            || previous_hash.zip(links.last()).is_some_and(
                |(previous_hash, previous): (u32, &OwnedLink)| {
                    (previous_hash, previous.name()) >= (hash, link.name())
                },
            )
        {
            return Err(Error::EditUnsupported(
                "a group's name index has inconsistent keys",
            ));
        }
        previous_hash = Some(hash);
        if let Some(order) = link.creation_order() {
            if creation.insert(order, id.to_vec()).is_some() {
                return Err(Error::EditUnsupported(
                    "a group's links have repeated creation indexes",
                ));
            }
        }
        links.push(link);
        Ok(())
    })?;
    if header.managed_objects_count != links.len().to_u64() {
        return Err(Error::EditUnsupported(
            "a group's heap and name index have different link counts",
        ));
    }
    if let Some(Some(address)) = info.btree_creation_order_address {
        read_creation_index(source, widths, address, creation)?;
    }
    Ok(links)
}

/// Checks creation-order records against the name index's link IDs.
///
/// The caller passes each link's creation index and heap ID in `creation`. Checks strictly
/// increasing creation indexes and matches every record's heap ID against this map.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] for an unsupported record encoding, inconsistent indexes
/// or heap IDs, or omitted links. Returns errors from address resolution, metadata reads,
/// or B-tree parsing.
fn read_creation_index(
    source: &(impl StoredRead + ?Sized),
    widths: FormatWidths,
    address: StoredAddress,
    mut creation: BTreeMap<u64, Vec<u8>>,
) -> Result<(), Error> {
    let header = BTreeV2Header::parse_from_source_with_widths(
        &StoredMetadata(source),
        address.get(),
        widths,
    )?;
    if header.tree_type != BTREE_V2_LINK_CREATION_ORDER
        || header.record_size != CREATION_RECORD_SIZE
    {
        return Err(Error::EditUnsupported(
            "a group's creation-order index uses an unsupported record encoding",
        ));
    }
    let mut previous = None;
    let layout = BTreeV2Layout::new(&header, widths.offsets)?;
    walk_index(source, &layout, |record| {
        let (index, id) = record
            .split_first_chunk::<LINK_CREATION_ORDER_LENGTH>()
            .ok_or(FormatError::InvalidBTreeV2Signature)?;
        let index = u64::from_le_bytes(*index);
        if previous.is_some_and(|previous| previous >= index)
            || creation.remove(&index).as_deref() != Some(id)
        {
            return Err(Error::EditUnsupported(
                "a group's creation-order index disagrees with its links",
            ));
        }
        previous = Some(index);
        Ok(())
    })?;
    if !creation.is_empty() {
        return Err(Error::EditUnsupported(
            "a group's creation-order index omits links",
        ));
    }
    Ok(())
}

/// Calls `read_record` with each index record, traversing children between records.
///
/// Reads nodes in stored coordinates through [`StoredRead::read_metadata`].
///
/// # Errors
///
/// Returns the errors of [`IndexReader::walk`].
fn walk_index(
    source: &(impl StoredRead + ?Sized),
    layout: &BTreeV2Layout,
    mut read_record: impl FnMut(&[u8]) -> Result<(), Error>,
) -> Result<(), Error> {
    if let Some(root) = layout.root() {
        IndexReader {
            source,
            visited: BTreeSet::new(),
        }
        .walk(root, &mut read_record)?;
    }
    Ok(())
}

/// Reads one index while retaining the stored addresses of visited nodes.
///
/// Reads metadata through [`StoredRead::read_metadata`]. Retains visited addresses across
/// recursive calls.
struct IndexReader<'a, S: StoredRead + ?Sized> {
    source: &'a S,
    visited: BTreeSet<StoredAddress>,
}

impl<S: StoredRead + ?Sized> IndexReader<'_, S> {
    /// Returns the subtree's record count after visiting its records.
    ///
    /// Checks physical endpoint arithmetic before reading at the node's stored address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] for a repeated node address, overflowing
    /// record count, or inconsistent subtree total. Returns errors from address resolution,
    /// metadata reads, node parsing, or `read_record`.
    fn walk(
        &mut self,
        node: BTreeV2Node<'_>,
        read_record: &mut impl FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<u64, Error> {
        let Self { source, visited } = self;
        if !visited.insert(node.address()) {
            return Err(FormatError::InvalidBTreeV2Signature.into());
        }
        let length = node.encoded_len()?.to_u64();
        FileRange::new(source.source_address(node.address())?, length)?;
        let bytes = source.read_range(AddressRange::new(node.address(), length)?)?;
        let view = node.parse(&bytes)?;
        let mut count = 0u64;
        for record in 0..view.record_count() {
            if let Some(child) = view.child(record)? {
                count = count
                    .checked_add(self.walk(child, read_record)?)
                    .ok_or(FormatError::InvalidBTreeV2Signature)?;
            }
            read_record(view.record(record)?)?;
            count = count
                .checked_add(1)
                .ok_or(FormatError::InvalidBTreeV2Signature)?;
        }
        if let Some(child) = view.child(view.record_count())? {
            count = count
                .checked_add(self.walk(child, read_record)?)
                .ok_or(FormatError::InvalidBTreeV2Signature)?;
        }
        if count != node.total_records() {
            return Err(FormatError::InvalidBTreeV2Signature.into());
        }
        Ok(count)
    }
}

/// Converts link-storage errors to format or unsupported-edit errors.
pub(crate) fn storage_error(error: LinkStorageError) -> Error {
    match error {
        LinkStorageError::Format(error) => error.into(),
        LinkStorageError::CreationOrderCounterRange { .. }
        | LinkStorageError::IndexingWithoutTracking => {
            Error::EditUnsupported("a group's link creation-order policy cannot be reproduced")
        }
        _ => Error::EditUnsupported("a group's links or link-info metadata cannot be reproduced"),
    }
}

// "Version 2 B-trees" in the format specification, version 4.0, defines seven-byte heap IDs
// in type 5 and type 6 records.
const LINK_HEAP_ID_LENGTH: u16 = 7;
// "Version 2 B-trees" in the format specification, version 4.0, defines a four-byte name hash
// before the heap ID in type 5 records.
const NAME_RECORD_SIZE: u16 = 4 + LINK_HEAP_ID_LENGTH;
// "Version 2 B-trees" in the format specification, version 4.0, defines an eight-byte creation
// index before the heap ID in type 6 records.
const CREATION_RECORD_SIZE: u16 = 8 + LINK_HEAP_ID_LENGTH;
// "Version 2 B-trees" in the format specification, version 4.0, defines eight-byte creation
// indexes in type 6 records.
const LINK_CREATION_ORDER_LENGTH: usize = 8;
// "Fractal Heap" in the format specification, version 4.0, defines zero for the version and
// type bits of a version 0 managed heap ID.
const MANAGED_HEAP_ID_FLAGS: u8 = 0;
