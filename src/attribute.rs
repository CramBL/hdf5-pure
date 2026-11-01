//! The attributes of an object, read through its object header.
//!
//! In compact storage the object header stores one Attribute message per attribute, and in dense
//! storage the reader selects the object's attribute heap or the file's shared-message heap from
//! each index record's flags. The version 0 Attribute Info message stores the object's heap address
//! and is defined in "The Attribute Info Message" of the [format specification, version 4.0][spec].
//! A parsed message is an [`AttributeMessage`].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub use hdf5_pure_format::__private::AttributeMessage;
use hdf5_pure_format::__private::{BTREE_V2_ATTRIBUTE_CREATION_ORDER, BTREE_V2_ATTRIBUTE_NAME};
use hdf5_pure_format::__private::{
    DenseAttributeRecord, DenseAttributeTarget, FractalHeapIdLayout,
};
use hdf5_pure_space::__private::Extent;

use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::attribute_info::AttributeInfoMessage;
use crate::btree_v2;
use crate::btree_v2::BTreeV2Header;
use crate::btree_v2::BTreeV2Source;
use crate::convert::Narrow;
use crate::dense_storage::{self, DenseIndex};
use crate::error::FormatError;
use crate::fractal_heap;
#[cfg(test)]
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::HeapObjectReader;
use crate::fractal_heap::HeapSource;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::shared_message::SharedResolver;
use crate::shared_message::SourceResolver;
use crate::shared_message::StoredResolver;
use crate::sohm::{self, SohmTable};
use crate::source::OriginalSource;
use crate::source::PhysicalSource;
#[cfg(doc)]
use crate::source::Source;
use crate::source::StoredBytes;
use crate::source::StoredMetadata;
use crate::source::StoredRead;
#[cfg(test)]
use crate::width::LengthWidth;
use crate::width::{FormatWidths, OffsetWidth};

/// Returns the v2 B-tree allocations owned by a dense attribute set.
///
/// "The Attribute Info Message" of the [format specification, version 4.0][spec] identifies a
/// fractal heap, a version 2 B-tree name index, and optionally a creation-order index. The name
/// index uses client type 8 and the creation-order index uses client type 9. Only the two index
/// allocations participate in this ownership proof. The fractal heap remains outside it.
///
/// The complete set is proven before it is returned. If both indexes exist, their allocations must
/// be disjoint. A creation-order index is followed only when the Attribute Info message says that
/// creation order is indexed and supplies its address.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if either index has the wrong client type. Returns
/// the structural error from the version 2 B-tree storage walker if an index cannot be enumerated
/// completely. Returns [`FormatError::InvalidBTreeV2Signature`] if the two indexes overlap or an
/// index allocation contains the fractal-heap header address.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
pub(crate) fn collect_dense_attribute_index_storage_extents(
    source: &(impl StoredRead + ?Sized),
    attr_info: &AttributeInfoMessage,
    widths: FormatWidths,
) -> Result<Vec<Extent>, FormatError> {
    let Some(fractal_heap_address) = attr_info.fractal_heap_address else {
        return Ok(Vec::new());
    };
    let Some(name_address) = attr_info.btree_name_index_address else {
        return Ok(Vec::new());
    };

    let mut indexes = Vec::with_capacity(2);
    indexes.push(DenseIndex::new(name_address, BTREE_V2_ATTRIBUTE_NAME));
    if attr_info.indexes_creation_order
        && let Some(address) = attr_info.btree_creation_order_address
    {
        indexes.push(DenseIndex::new(address, BTREE_V2_ATTRIBUTE_CREATION_ORDER));
    }
    dense_storage::collect_dense_index_storage_extents_from_source(
        source,
        fractal_heap_address,
        indexes,
        widths,
    )
}

/// Extract all (compact) attribute messages from an object header.
///
/// Only used by tests; the reader uses [`extract_attributes_full`] (which also
/// handles dense storage). Gated so it is not shipped as dead code.
#[cfg(test)]
pub fn extract_attributes(
    header: &ObjectHeader,
    length_width: LengthWidth,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let mut attrs = Vec::new();
    for msg in &header.messages {
        if msg.msg_type == MessageType::ATTRIBUTE {
            let attr = AttributeMessage::parse_with_length_width(&msg.data, length_width)?;
            attrs.push(attr);
        }
    }
    Ok(attrs)
}

/// Reads compact and dense attributes, resolving their shared messages.
///
/// Borrows heap and index metadata from the resident image. Read-write access requires full
/// direct-block checksum validation when reading a checksummed managed heap object.
///
/// # Errors
///
/// Returns the attribute, index, heap, and shared-message parsing errors as [`FormatError`].
pub fn extract_attributes_full(
    source: StoredBytes<'_>,
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let resolver = StoredResolver::new(source, access_mode, widths, sohm);
    let mut attrs = Vec::new();

    for msg in &header.messages {
        if msg.msg_type == MessageType::ATTRIBUTE {
            let attr = if msg.flags.is_shared() {
                // The whole attribute message is shared: resolve the reference to
                // get the message, which may itself name a committed datatype.
                let resolved = resolver.resolve(&msg.data, MessageType::ATTRIBUTE)?;
                AttributeMessage::parse_resolving_with_length_width(
                    &resolved,
                    widths.lengths,
                    &resolver,
                )?
            } else {
                AttributeMessage::parse_resolving_with_length_width(
                    &msg.data,
                    widths.lengths,
                    &resolver,
                )?
            };
            attrs.push(attr);
        }
    }

    let attr_info = find_attribute_info(header, widths.offsets)?;
    if let Some(info) = attr_info
        && let Some(fh_addr) = info.fractal_heap_address
    {
        let dense_attrs =
            extract_dense_attributes(source, access_mode, &info, fh_addr, widths, sohm)?;
        attrs.extend(dense_attrs);
    }

    Ok(attrs)
}

/// Reads compact and dense attributes through the opened source.
///
/// Resolves heap and shared-message addresses using the source's effective address context.
///
/// # Errors
///
/// Returns the errors of [`extract_stored_attributes_from_source`].
pub fn extract_attributes_full_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    Ok(
        extract_stored_attributes_from_source(source, access_mode, header, widths, sohm)?
            .into_iter()
            .map(|a| a.message)
            .collect(),
    )
}

/// An attribute as its object stores it, with the creation index the storage
/// records for it.
#[derive(Debug, Clone)]
pub struct StoredAttribute {
    /// The attribute itself.
    pub message: AttributeMessage,
    /// Its creation index: the object-header message record's field for a
    /// compact attribute, the name index record's for a dense one. `None`
    /// where the object does not track attribute creation order — which is
    /// every object this crate's own whole-file writer produces.
    pub creation_index: Option<u16>,
}

/// Reads attributes and their creation indexes through the opened source.
///
/// Resolves heap and shared-message addresses using the source's effective address context.
/// Object-header records store creation indexes for compact attributes, and B-tree records
/// for dense attributes.
///
/// # Errors
///
/// Returns the parser error if an attribute message, a shared reference, or dense storage is
/// malformed, and the source error if a read fails.
pub fn extract_stored_attributes_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<StoredAttribute>, FormatError> {
    let resolver = SourceResolver::new(source, access_mode, widths, sohm);
    let mut attrs = Vec::new();

    for msg in &header.messages {
        if msg.msg_type == MessageType::ATTRIBUTE {
            let attr = if msg.flags.is_shared() {
                let resolved = resolver.resolve(&msg.data, MessageType::ATTRIBUTE)?;
                AttributeMessage::parse_resolving_with_length_width(
                    &resolved,
                    widths.lengths,
                    &resolver,
                )?
            } else {
                AttributeMessage::parse_resolving_with_length_width(
                    &msg.data,
                    widths.lengths,
                    &resolver,
                )?
            };
            attrs.push(StoredAttribute {
                message: attr,
                creation_index: msg.creation_order,
            });
        }
    }

    let attr_info = find_attribute_info(header, widths.offsets)?;
    if let Some(info) = attr_info
        && let Some(fh_addr) = info.fractal_heap_address
    {
        let dense_attrs = extract_dense_attributes_from_source(
            source,
            access_mode,
            &info,
            fh_addr,
            widths,
            sohm,
        )?;
        attrs.extend(dense_attrs);
    }

    Ok(attrs)
}

/// Find and parse the Attribute Info message from an object header.
fn find_attribute_info(
    header: &ObjectHeader,
    offset_width: OffsetWidth,
) -> Result<Option<AttributeInfoMessage>, FormatError> {
    for msg in &header.messages {
        if msg.msg_type == MessageType::ATTRIBUTE_INFO {
            let info = AttributeInfoMessage::parse_with_offset_width(&msg.data, offset_width)?;
            return Ok(Some(info));
        }
    }
    Ok(None)
}

/// Reads attributes through the name index, selecting each message's private or shared heap.
///
/// Private messages are read from `fh_addr`, and shared messages from the Attribute index in
/// `sohm`. Datatype and dataspace references within either message are resolved afterward.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `attr_info` contains no B-tree name index address,
/// [`FormatError::UnsupportedSohmReference`] if a shared record has no shared-message table,
/// and [`FormatError::SohmIndexMissing`] if its Attribute index or heap is absent. Returns the
/// parser error if a record, heap, or Attribute message is malformed.
fn extract_dense_attributes(
    source: StoredBytes<'_>,
    access_mode: AccessMode,
    attr_info: &AttributeInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let heap_source = HeapSource::resident(source);
    let (fh, _) = fractal_heap::read_header_and_range_from_source(&heap_source, fh_addr, widths)?;

    let btree_addr = attr_info
        .btree_name_index_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
    let btree_hdr = BTreeV2Header::parse_with_widths(
        source.metadata_frame().bytes(),
        btree_addr.get().to_usize()?,
        widths,
    )?;
    let records = btree_v2::collect_btree_v2_records(source, &btree_hdr, widths.offsets)?;

    let resolver = StoredResolver::new(source, access_mode, widths, sohm);
    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(heap_source, &fh, fh_addr, access_mode, widths);
    let mut attrs = Vec::new();
    for record in &records {
        let record =
            DenseAttributeRecord::parse(&record.data, btree_hdr.tree_type, fh.heap_id_length)?;
        let attr_data = match record.target(&layout)? {
            DenseAttributeTarget::Private(id) => heap.read(id.id())?,
            DenseAttributeTarget::Shared(id) => sohm::read_heap_message(
                source,
                access_mode,
                sohm.ok_or(FormatError::UnsupportedSohmReference)?,
                MessageType::ATTRIBUTE,
                id.bytes(),
                widths,
            )?,
        };

        // The data in the heap is a complete attribute message, and it names a
        // committed datatype the same way a compact one does.
        let attr = AttributeMessage::parse_resolving_with_length_width(
            &attr_data,
            widths.lengths,
            &resolver,
        )?;
        attrs.push(attr);
    }

    Ok(attrs)
}

/// Reads dense attributes and creation indexes through the opened source.
///
/// Resolves embedded datatype and dataspace references through the original [`Source`].
///
/// # Errors
///
/// Returns errors from [`visit_dense_attribute_messages_from_source`] or attribute parsing.
fn extract_dense_attributes_from_source(
    source: &(impl OriginalSource + ?Sized),
    access_mode: AccessMode,
    attr_info: &AttributeInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<StoredAttribute>, FormatError> {
    let resolver = SourceResolver::new(source, access_mode, widths, sohm);
    let mut attrs = Vec::new();
    let stored = PhysicalSource::new(source).stored();
    visit_dense_attribute_messages_from_source(
        &stored,
        access_mode,
        attr_info,
        fh_addr,
        widths,
        sohm,
        |_flags| true,
        |data, creation_index| {
            attrs.push(StoredAttribute {
                message: AttributeMessage::parse_resolving_with_length_width(
                    data,
                    widths.lengths,
                    &resolver,
                )?,
                creation_index,
            });
            Ok(())
        },
    )?;
    Ok(attrs)
}

/// Reads selected dense attribute messages through the name index.
///
/// A type 8 record of a version 2 B-tree stores message flags after the heap ID, as defined
/// in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// Calls `select` with each record's message flags before reading its heap object. Calls
/// `visit` with each selected message's bytes and optional creation index. Selected shared
/// messages are read through `sohm`. The reader decodes IDs and reads heap objects only for
/// selected records.
///
/// Reads heap and index metadata through [`StoredRead::read_metadata`].
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the name index is missing or a record is truncated.
/// Returns [`FormatError::UnsupportedSohmReference`] if a selected shared record has no
/// shared-message table, and [`FormatError::SohmIndexMissing`] if its Attribute index or heap
/// is absent. Returns the parser error if the index or heap cannot be read, or the error from
/// `visit`.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub(crate) fn visit_dense_attribute_messages_from_source(
    source: &dyn StoredRead,
    access_mode: AccessMode,
    attr_info: &AttributeInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
    mut select: impl FnMut(MessageFlags) -> bool,
    mut visit: impl FnMut(&[u8], Option<u16>) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let heap_source = HeapSource::new(source);
    let (fh, _) = fractal_heap::read_header_and_range_from_source(&heap_source, fh_addr, widths)?;

    let btree_addr = attr_info
        .btree_name_index_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
    let btree_hdr = BTreeV2Header::parse_from_source_with_widths(
        &StoredMetadata(source),
        btree_addr.get(),
        widths,
    )?;
    let records = btree_v2::collect_btree_v2_records_from_source(
        &BTreeV2Source::new(source),
        &btree_hdr,
        widths.offsets,
    )?;

    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(heap_source, &fh, fh_addr, access_mode, widths);
    for record in &records {
        let record =
            DenseAttributeRecord::parse(&record.data, btree_hdr.tree_type, fh.heap_id_length)?;
        if !select(record.flags()) {
            continue;
        }
        let attr_data = match record.target(&layout)? {
            DenseAttributeTarget::Private(id) => heap.read(id.id())?,
            DenseAttributeTarget::Shared(id) => sohm::read_heap_message_from_source(
                source,
                access_mode,
                sohm.ok_or(FormatError::UnsupportedSohmReference)?,
                MessageType::ATTRIBUTE,
                id.bytes(),
                widths,
            )?,
        };
        let creation_index = attr_info
            .max_creation_index
            .and_then(|_| u16::try_from(record.creation_order()).ok());
        visit(&attr_data, creation_index)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;

    use hdf5_pure_format::__private::AttributeRecord;
    use rstest::rstest;
    use test_util::btree_v2;
    use test_util::bytes as test_bytes;
    use test_util::checksum as test_checksum;
    use test_util::dataspace;
    use test_util::datatype;
    use test_util::fractal_heap as heap_fixture;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::AddressContext;
    use crate::address::BaseAddress;
    use crate::address::BaseAddressExt;
    use crate::data_read;
    use crate::message_flags::MessageFlags;
    use crate::reader::OpenedSource;
    use crate::source;
    use crate::source::BytesSource;
    #[cfg(feature = "std")]
    use crate::source::MetadataCacheConfig;
    #[cfg(feature = "std")]
    use crate::source::MetadataCachingSource;
    use crate::source::PhysicalBytes;
    use crate::source::Source;

    // "Fractal Heap" of the format specification, version 4.0, defines tiny heap ID type 2
    // in bits 4 and 5. Zero length bits encode a one-byte object.
    const TINY_ID_TAG: u8 = 0x20;

    const WIDTHS: Widths = Widths::EIGHT;
    const NODE_SIZE: u64 = 512;
    const NAME_RECORD_SIZE: u16 = AttributeRecord::name_record_size(8);

    fn shared_dense_record_fixture() -> (Vec<u8>, AttributeInfoMessage, FormatWidths) {
        let mut header = heap_fixture::Header::new(0xFFFF).build(WIDTHS);
        test_bytes::set_u16_at(&mut header, 5, 8);
        let len = header.len();
        test_checksum::restamp(&mut header, 0, len);
        let mut image = vec![0; 0x500];
        image[..header.len()].copy_from_slice(&header);
        let record = btree_v2::attribute_name_record(
            &[TINY_ID_TAG, b'a', 0, 0, 0, 0, 0, 0],
            MessageFlags::SHARED.get(),
            42,
            9,
        );
        let index = btree_v2::Header::new(BTREE_V2_ATTRIBUTE_NAME, NAME_RECORD_SIZE, 0x300, 1)
            .build(WIDTHS);
        let leaf = btree_v2::leaf(BTREE_V2_ATTRIBUTE_NAME, &[record]);
        image[0x200..0x200 + index.len()].copy_from_slice(&index);
        image[0x300..0x300 + leaf.len()].copy_from_slice(&leaf);
        let info = AttributeInfoMessage {
            max_creation_index: Some(42),
            indexes_creation_order: false,
            fractal_heap_address: Some(StoredAddress::new(0)),
            btree_name_index_address: Some(StoredAddress::new(0x200)),
            btree_creation_order_address: None,
        };
        (image, info, FormatWidths::from_sizes(8, 8).unwrap())
    }

    #[rstest]
    #[case::buffered_without_table(false, None)]
    #[case::source_without_table(true, None)]
    #[case::buffered_without_index(false, Some(SohmTable { indexes: vec![] }))]
    #[case::source_without_index(true, Some(SohmTable { indexes: vec![] }))]
    #[case::buffered_without_heap(false, Some(attribute_shared_table(None)))]
    #[case::source_without_heap(true, Some(attribute_shared_table(None)))]
    fn selected_shared_dense_records_report_shared_storage_errors(
        #[case] streaming: bool,
        #[case] table: Option<SohmTable>,
        #[values(0, 512)] base: usize,
    ) {
        let (image, info, widths) = shared_dense_record_fixture();
        let mut physical = vec![0; base];
        physical.extend_from_slice(&image);
        let expected = if table.is_some() {
            FormatError::SohmIndexMissing(MessageType::ATTRIBUTE.to_u16())
        } else {
            FormatError::UnsupportedSohmReference
        };
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base as u64));
        let error = if streaming {
            extract_dense_attributes_from_source(
                &OpenedSource::for_test(&BytesSource::new(&physical), addresses),
                AccessMode::ReadOnly,
                &info,
                StoredAddress::new(0),
                widths,
                table.as_ref(),
            )
            .unwrap_err()
        } else {
            extract_dense_attributes(
                PhysicalBytes::for_test(&physical, addresses)
                    .stored()
                    .unwrap(),
                AccessMode::ReadOnly,
                &info,
                StoredAddress::new(0),
                widths,
                table.as_ref(),
            )
            .unwrap_err()
        };
        assert_eq!(error, expected);
    }

    #[rstest]
    #[case::zero_base(0)]
    #[case::userblock(512)]
    fn dense_attribute_selectors_skip_shared_records_before_heap_access(#[case] base: usize) {
        let (image, info, widths) = shared_dense_record_fixture();
        let mut physical = vec![0; base];
        physical.extend_from_slice(&image);
        let source = CountingSource::new(physical);
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base as u64));
        let mut selected = Vec::new();
        visit_dense_attribute_messages_from_source(
            &PhysicalSource::for_test(&source, addresses).stored(),
            AccessMode::ReadWrite,
            &info,
            StoredAddress::new(0),
            widths,
            None,
            |flags| {
                selected.push(flags);
                !flags.is_shared()
            },
            |_, _| panic!("a skipped record must not be visited"),
        )
        .unwrap();
        assert_eq!(selected, [MessageFlags::SHARED]);
        assert_eq!(
            *source.reads.borrow(),
            [
                base as u64,
                base as u64,
                (base + 0x200) as u64,
                (base + 0x300) as u64
            ]
        );
    }

    #[cfg(feature = "std")]
    #[rstest]
    #[case::borrowed(false)]
    #[case::owned(true)]
    fn dense_visits_preserve_metadata_owners_ranges_and_cache_admission(
        #[case] owned: bool,
        #[values(0, 512)] base: usize,
    ) {
        let (image, info, widths) = shared_dense_record_fixture();
        let mut physical = vec![0; base];
        physical.extend_from_slice(&image);
        let borrowed = BytesSource::new(&physical);
        let mut allocating = CountingSource::new(physical.clone());
        allocating.origin = 37;
        let source: &dyn Source = if owned { &allocating } else { &borrowed };
        let cached = MetadataCachingSource::new(source, MetadataCacheConfig::new(4096));
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base as u64));
        let stored = PhysicalSource::for_test(&cached, addresses).stored();
        let table = attribute_shared_table(Some(StoredAddress::new(0)));
        let mut observed = Vec::new();
        let mut visit = || {
            visit_dense_attribute_messages_from_source(
                &stored,
                AccessMode::ReadOnly,
                &info,
                StoredAddress::new(0),
                widths,
                Some(&table),
                |_| true,
                |data, order| {
                    observed.push((data.to_vec(), order));
                    Ok(())
                },
            )
        };
        visit().unwrap();
        visit().unwrap();
        assert_eq!(
            observed,
            [(b"a".to_vec(), Some(42)), (b"a".to_vec(), Some(42))]
        );
        let stats = cached.metadata_cache_stats().unwrap();
        assert_eq!(
            (stats.hits(), stats.misses(), stats.entries()),
            if owned { (8, 4, 4) } else { (0, 12, 0) }
        );
        let len = btree_v2::NODE_PREFIX + usize::from(NAME_RECORD_SIZE) + test_checksum::CHECKSUM;
        let metadata = stored
            .read_metadata(StoredAddress::new(0x300), len)
            .unwrap();
        assert_eq!(metadata.as_ref(), &image[0x300..0x300 + len]);
        assert_eq!(
            metadata.offset(),
            (base + 0x300) as u64 + if owned { 37 } else { 0 }
        );
        assert_eq!(metadata.shared().is_some(), owned);
        if !owned {
            assert_eq!(metadata.as_ptr(), physical[base + 0x300..].as_ptr());
        }
        let retained = metadata.shared();
        drop(metadata);
        drop(cached);
        if let Some(retained) = retained {
            assert_eq!(retained.as_ref(), &image[0x300..0x300 + len]);
            assert_eq!(retained.offset(), (base + 0x300 + 37) as u64);
        }
    }

    #[rstest]
    #[case::invalid_heap(0, false, FormatError::InvalidFractalHeapSignature)]
    #[case::heap_before_missing_index(0, true, FormatError::InvalidFractalHeapSignature)]
    #[case::missing_index(0x200, true, FormatError::UnexpectedEof { expected: 1, available: 0 })]
    #[case::invalid_index(0x200, false, FormatError::InvalidBTreeV2Signature)]
    fn dense_visits_preserve_storage_error_precedence(
        #[case] corrupt_at: usize,
        #[case] missing_index: bool,
        #[case] expected: FormatError,
        #[values(0, 512)] base: usize,
    ) {
        let (mut image, mut info, widths) = shared_dense_record_fixture();
        image[corrupt_at] = 0;
        if missing_index {
            info.btree_name_index_address = None;
        }
        let mut physical = vec![0; base];
        physical.extend_from_slice(&image);
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base as u64));
        let source = BytesSource::new(&physical);
        assert_eq!(
            visit_dense_attribute_messages_from_source(
                &PhysicalSource::for_test(&source, addresses).stored(),
                AccessMode::ReadOnly,
                &info,
                StoredAddress::new(0),
                widths,
                None,
                |_| panic!("invalid storage is rejected before record selection"),
                |_, _| panic!("invalid storage is rejected before visiting messages"),
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(
            extract_dense_attributes(
                PhysicalBytes::for_test(&physical, addresses)
                    .stored()
                    .unwrap(),
                AccessMode::ReadOnly,
                &info,
                StoredAddress::new(0),
                widths,
                None,
            )
            .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::tracked(Some(42), Some(42))]
    #[case::untracked(None, None)]
    fn shared_dense_messages_preserve_tracked_creation_order(
        #[case] max_creation_index: Option<u16>,
        #[case] expected: Option<u16>,
    ) {
        let (image, mut info, widths) = shared_dense_record_fixture();
        info.max_creation_index = max_creation_index;
        let table = attribute_shared_table(Some(StoredAddress::new(0)));
        let mut observed = Vec::new();
        visit_dense_attribute_messages_from_source(
            &source::zero_base_source(&BytesSource::new(image)),
            AccessMode::ReadOnly,
            &info,
            StoredAddress::new(0),
            widths,
            Some(&table),
            |_| true,
            |data, order| {
                observed.push((data.to_vec(), order));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(observed, [(b"a".to_vec(), expected)]);
    }

    fn attribute_shared_table(heap_address: Option<StoredAddress>) -> SohmTable {
        SohmTable {
            indexes: vec![crate::sohm::SohmIndexHeader {
                message_type_flags: 1 << MessageType::ATTRIBUTE.to_u16(),
                min_message_size: 1,
                list_max: 50,
                btree_min: 40,
                message_count: 1,
                kind: hdf5_pure_format::__private::SohmIndexKind::List,
                index_address: None,
                heap_address,
            }],
        }
    }

    fn place_attribute_name_index(
        file: &mut [u8],
        header_at: usize,
        root_at: usize,
    ) -> Vec<Extent> {
        let record = vec![0; usize::from(NAME_RECORD_SIZE)];
        let leaf = btree_v2::leaf(BTREE_V2_ATTRIBUTE_NAME, &[record]);
        let header = btree_v2::Header::new(
            BTREE_V2_ATTRIBUTE_NAME,
            NAME_RECORD_SIZE,
            u64::try_from(root_at).unwrap(),
            1,
        )
        .build(WIDTHS);
        file[header_at..header_at + header.len()].copy_from_slice(&header);
        file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);
        vec![
            Extent::new(
                StoredAddress::new(u64::try_from(header_at).unwrap()),
                u64::try_from(header.len()).unwrap(),
            )
            .unwrap(),
            Extent::new(
                StoredAddress::new(u64::try_from(root_at).unwrap()),
                NODE_SIZE,
            )
            .unwrap(),
        ]
    }

    #[rstest]
    #[case::zero_base(0)]
    #[case::userblock(512)]
    fn dense_attribute_storage_walk_omits_creation_order_index_when_not_indexed(
        #[case] base: usize,
    ) {
        let mut file = vec![0; 0x800];
        let expected = place_attribute_name_index(&mut file, 0x100, 0x200);
        let info = AttributeInfoMessage {
            max_creation_index: Some(1),
            indexes_creation_order: false,
            fractal_heap_address: Some(StoredAddress::new(0x80)),
            btree_name_index_address: Some(StoredAddress::new(0x100)),
            // A defined address alone does not make the creation-order index live.
            btree_creation_order_address: Some(StoredAddress::new(0x500)),
        };

        let mut physical = vec![0; base];
        physical.extend_from_slice(&file);
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base as u64));
        let source = BytesSource::new(physical);
        let extents = collect_dense_attribute_index_storage_extents(
            &PhysicalSource::for_test(&source, addresses).stored(),
            &info,
            FormatWidths::from_sizes(8, 8).unwrap(),
        )
        .unwrap();

        assert_eq!(extents, expected);
    }

    /// A [`Source`] that records where each read started, so a walk can be asked
    /// how often it went back to a particular structure.
    struct CountingSource {
        inner: BytesSource<Vec<u8>>,
        reads: RefCell<Vec<u64>>,
        origin: u64,
    }

    impl CountingSource {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                inner: BytesSource::new(bytes),
                reads: RefCell::new(Vec::new()),
                origin: 0,
            }
        }

        fn reads_at(&self, offset: u64) -> usize {
            self.reads.borrow().iter().filter(|&&o| o == offset).count()
        }
    }

    impl Source for CountingSource {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn metadata_origin(&self) -> Result<u64, FormatError> {
            Ok(self.origin)
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push(offset);
            self.inner.read_at(offset, buf)
        }
    }

    /// A file whose root carries `count` attributes, each too large for a
    /// managed heap object, so every one of them resolves through the heap's
    /// huge-object B-tree.
    fn file_with_huge_attributes(count: usize) -> Vec<u8> {
        let mut builder = crate::FileBuilder::new();
        for i in 0..count {
            builder.set_attr(
                &format!("a{i}"),
                crate::AttrValue::StringArray(vec![format!("{i:0700}"); 100]),
            );
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder.finish().unwrap()
    }

    /// A file whose root carries `count` attributes small enough to be managed
    /// heap objects, so the heap holds no huge object at all.
    fn file_with_managed_attributes(count: usize) -> Vec<u8> {
        let mut builder = crate::FileBuilder::new();
        for i in 0..count {
            builder.set_attr(&format!("a{i}"), crate::AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder.finish().unwrap()
    }

    /// The root group's dense-attribute storage: its info message, its heap
    /// address, and the file's format widths.
    fn dense_attribute_info(bytes: &[u8]) -> (AttributeInfoMessage, StoredAddress, FormatWidths) {
        let crate::superblock::LocatedSuperblock {
            superblock, widths, ..
        } = crate::superblock::read(bytes).unwrap();
        let root = ObjectHeader::parse_with_widths(
            bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(superblock.base_address)
                .resolve(superblock.root_group_address)
                .unwrap()
                .get()
                .to_usize()
                .unwrap(),
            widths,
        )
        .unwrap();
        let info = find_attribute_info(&root, widths.offsets)
            .unwrap()
            .expect("this many attributes are stored densely");
        let fh_addr = info
            .fractal_heap_address
            .expect("dense storage names its heap");
        (info, fh_addr, widths)
    }

    #[test]
    fn a_dense_walk_parses_its_huge_object_index_once() {
        const COUNT: usize = 6;
        let bytes = file_with_huge_attributes(COUNT);
        let (info, fh_addr, widths) = dense_attribute_info(&bytes);
        let heap = FractalHeapHeader::parse_bytes(
            &bytes[fh_addr.get().to_usize().unwrap()..],
            widths.offsets,
            widths.lengths,
        )
        .unwrap();
        let btree_addr = heap.btree_huge_objects_address;

        let source = CountingSource::new(bytes);
        let attrs = extract_dense_attributes_from_source(
            &OpenedSource::for_test(
                &source,
                AddressContext::from_effective_base(BaseAddress::ZERO),
            ),
            AccessMode::ReadOnly,
            &info,
            fh_addr,
            widths,
            None,
        )
        .unwrap();

        assert_eq!(
            attrs.len(),
            COUNT,
            "the walk must still read every attribute"
        );
        assert_eq!(
            source.reads_at(btree_addr.get()),
            1,
            "the huge-object B-tree header was re-read per object"
        );
    }

    #[test]
    fn a_buffered_dense_walk_parses_its_huge_object_index_once() {
        const COUNT: usize = 6;
        let bytes = file_with_huge_attributes(COUNT);
        let (info, fh_addr, widths) = dense_attribute_info(&bytes);

        crate::fractal_heap::reset_huge_index_decodes();
        let attrs = extract_dense_attributes(
            source::zero_base_bytes(&bytes),
            AccessMode::ReadOnly,
            &info,
            fh_addr,
            widths,
            None,
        )
        .unwrap();

        assert_eq!(
            attrs.len(),
            COUNT,
            "the walk must still read every attribute"
        );
        assert_eq!(
            crate::fractal_heap::huge_index_decodes(),
            1,
            "the huge-object index was parsed per object rather than per walk"
        );
    }

    #[test]
    fn a_managed_dense_walk_never_parses_a_huge_object_index() {
        // Enough attributes to force dense storage, none of them large enough to
        // exceed the heap's managed-object limit.
        const COUNT: usize = 40;
        let bytes = file_with_managed_attributes(COUNT);
        let (info, fh_addr, widths) = dense_attribute_info(&bytes);

        crate::fractal_heap::reset_huge_index_decodes();
        let buffered = extract_dense_attributes(
            source::zero_base_bytes(&bytes),
            AccessMode::ReadOnly,
            &info,
            fh_addr,
            widths,
            None,
        )
        .unwrap();
        let source = BytesSource::new(bytes);
        let streamed = extract_dense_attributes_from_source(
            &OpenedSource::for_test(
                &source,
                AddressContext::from_effective_base(BaseAddress::ZERO),
            ),
            AccessMode::ReadOnly,
            &info,
            fh_addr,
            widths,
            None,
        )
        .unwrap();

        assert_eq!(buffered.len(), COUNT, "the walk must read every attribute");
        assert_eq!(streamed.len(), COUNT);
        assert_eq!(
            crate::fractal_heap::huge_index_decodes(),
            0,
            "a heap with no huge object parsed an index it has no use for"
        );
    }

    #[test]
    fn extract_attributes_from_header() {
        // Build a fake ObjectHeader with 3 attribute messages
        let mut msgs = Vec::new();
        for i in 0..3 {
            let name = format!("attr{}\0", i);
            let dt_bytes = datatype::f64_le();
            let ds_bytes = dataspace::scalar();

            let mut attr_data = Vec::new();
            attr_data.push(2); // version
            attr_data.push(0);
            attr_data.extend_from_slice(&(name.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(name.as_bytes());
            attr_data.extend_from_slice(&dt_bytes);
            attr_data.extend_from_slice(&ds_bytes);
            attr_data.extend_from_slice(&((i as f64) * 1.0).to_le_bytes());

            msgs.push(crate::object_header::HeaderMessage {
                msg_type: MessageType::ATTRIBUTE,
                size: attr_data.len(),
                flags: MessageFlags::NONE,
                creation_order: None,
                data: attr_data,
            });
        }

        let header = ObjectHeader {
            version: 2,
            messages: msgs,
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        };

        let attrs = extract_attributes(&header, LengthWidth::Eight).unwrap();
        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs[0].name, "attr0");
        assert_eq!(attrs[1].name, "attr1");
        assert_eq!(attrs[2].name, "attr2");
    }

    #[test]
    fn read_as_f64_scalar() {
        let name = b"v\0";
        let dt_bytes = datatype::f64_le();
        let ds_bytes = dataspace::scalar();

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(&3.14f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        let vals = data_read::read_as_f64(&attr.raw_data, &attr.datatype).unwrap();
        assert_eq!(vals, vec![3.14]);
    }

    #[test]
    fn read_as_string_fixed() {
        let name = b"s\0";
        let dt_bytes = datatype::fixed_string(5);
        let ds_bytes = dataspace::scalar();

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"world");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        let strs = data_read::read_as_strings(&attr.raw_data, &attr.datatype).unwrap();
        assert_eq!(strs, vec!["world"]);
    }

    #[test]
    fn read_as_strings_array() {
        let name = b"arr\0";
        let dt_bytes = datatype::fixed_string(4);
        let ds_bytes = dataspace::v1(1, dataspace::Flags::NONE, &[2], None);

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"abcdEFGH");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        let strs = data_read::read_as_strings(&attr.raw_data, &attr.datatype).unwrap();
        assert_eq!(strs, vec!["abcd", "EFGH"]);
    }
}
