//! The attributes of an object, read through its object header.
//!
//! In compact storage the object header stores one Attribute message per attribute, and in dense
//! storage a fractal heap stores them, which the header's Attribute Info message addresses. The
//! Attribute Info message is defined in "The Attribute Info Message" of the [format specification,
//! version 4.0][spec]. A parsed message is an [`AttributeMessage`].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub use hdf5_pure_format::__private::AttributeMessage;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::{BTREE_V2_ATTRIBUTE_CREATION_ORDER, BTREE_V2_ATTRIBUTE_NAME};
use hdf5_pure_space::__private::Extent;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::attribute_info::AttributeInfoMessage;
use crate::btree_v2::BTreeV2Header;
use crate::convert::Narrow;
use crate::dense_storage::{self, DenseIndex};
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::HeapObjectReader;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::shared_message::BufferedResolver;
use crate::shared_message::SharedResolver;
use crate::shared_message::SourceResolver;
use crate::sohm::SohmTable;
use crate::source::Source;
use crate::source::SourceMetadata;
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
pub(crate) fn collect_dense_attribute_index_storage_extents<S: Source + ?Sized>(
    source: &S,
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
    dense_storage::collect_dense_index_storage_extents(
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
/// `file_data` is framed at the effective base in `addresses`. Read-write access requires full
/// direct-block checksum validation when reading a checksummed managed heap object.
///
/// # Errors
///
/// Returns the attribute, index, heap, and shared-message parsing errors as [`FormatError`].
pub fn extract_attributes_full(
    file_data: &[u8],
    access_mode: AccessMode,
    addresses: AddressContext,
    header: &ObjectHeader,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let resolver = BufferedResolver::new(file_data, addresses, access_mode, widths, sohm);
    let mut attrs = Vec::new();

    // Collect compact attributes (inline in OH)
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

    // Check for dense attributes via AttributeInfo message
    let attr_info = find_attribute_info(header, widths.offsets)?;
    if let Some(info) = attr_info
        && let Some(fh_addr) = info.fractal_heap_address
    {
        let dense_attrs = extract_dense_attributes(
            file_data,
            access_mode,
            addresses,
            &info,
            fh_addr,
            widths,
            sohm,
        )?;
        attrs.extend(dense_attrs);
    }

    Ok(attrs)
}

/// Reads compact and dense attributes through a source framed at the effective base.
///
/// `addresses` holds the effective base used to frame `source`. Shared attribute references
/// are resolved through the same source.
///
/// # Errors
///
/// Returns the errors of [`extract_stored_attributes_from_source`].
pub fn extract_attributes_full_from_source<S: Source + ?Sized>(
    source: &S,
    addresses: AddressContext,
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    Ok(
        extract_stored_attributes_from_source(
            source,
            addresses,
            access_mode,
            header,
            widths,
            sohm,
        )?
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

/// Reads attributes and their stored creation indexes through a base-framed source.
///
/// `addresses` holds the effective base used to frame `source`. Compact attributes store
/// their creation indexes in object-header records, and dense attributes in B-tree records.
///
/// # Errors
///
/// Returns the parser error if an attribute message, a shared reference, or dense storage is
/// malformed, and the source error if a read fails.
pub fn extract_stored_attributes_from_source<S: Source + ?Sized>(
    source: &S,
    addresses: AddressContext,
    access_mode: AccessMode,
    header: &ObjectHeader,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<StoredAttribute>, FormatError> {
    let resolver = SourceResolver::new(source, addresses, access_mode, widths, sohm);
    let mut attrs = Vec::new();

    // Collect compact attributes (inline in OH)
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

    // Check for dense attributes via AttributeInfo message
    let attr_info = find_attribute_info(header, widths.offsets)?;
    if let Some(info) = attr_info
        && let Some(fh_addr) = info.fractal_heap_address
    {
        let dense_attrs = extract_dense_attributes_from_source(
            source,
            addresses,
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

/// Returns one attribute per record of the name index in `attr_info`, reading each message out
/// of the fractal heap at `fh_addr`.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `attr_info` contains no B-tree name index address,
/// and the [`FormatError`] of the first structure that does not parse.
fn extract_dense_attributes(
    file_data: &[u8],
    access_mode: AccessMode,
    addresses: AddressContext,
    attr_info: &AttributeInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    // Parse fractal heap
    let fh = crate::fractal_heap::parse_header(file_data, addresses, fh_addr, widths)?;

    // Parse B-tree v2 for name index (type 8)
    let btree_addr = attr_info
        .btree_name_index_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
    let btree_hdr =
        BTreeV2Header::parse_with_widths(file_data, btree_addr.get().to_usize()?, widths)?;
    let records = crate::btree_v2::collect_btree_v2_records(file_data, &btree_hdr, widths.offsets)?;

    let resolver = BufferedResolver::new(file_data, addresses, access_mode, widths, sohm);
    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(&fh, addresses, fh_addr, access_mode, widths);
    let mut attrs = Vec::new();
    for record in &records {
        let Some(id_bytes) = record.attribute_heap_id(fh.heap_id_length) else {
            continue;
        };

        // Read the attribute message from the fractal heap (managed or huge object).
        let attr_data = heap.read(file_data, layout.parse(id_bytes)?)?;

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

/// Streaming counterpart of [`extract_dense_attributes`]: walks the fractal heap
/// and B-tree v2 through a [`Source`] on demand.
fn extract_dense_attributes_from_source<S: Source + ?Sized>(
    source: &S,
    addresses: AddressContext,
    access_mode: AccessMode,
    attr_info: &AttributeInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
    sohm: Option<&SohmTable>,
) -> Result<Vec<StoredAttribute>, FormatError> {
    let resolver = SourceResolver::new(source, addresses, access_mode, widths, sohm);
    let mut attrs = Vec::new();
    visit_dense_attribute_messages_from_source(
        source,
        access_mode,
        addresses,
        attr_info,
        fh_addr,
        widths,
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
/// `visit` with each selected message's bytes and optional creation index. The caller resolves
/// shared messages when selecting them.
///
/// `source` is framed at the effective base in `addresses`.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the name index is missing or a record is truncated.
/// Returns the parser error if the index or heap cannot be read, or the error from `visit`.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub(crate) fn visit_dense_attribute_messages_from_source(
    source: &(impl Source + ?Sized),
    access_mode: AccessMode,
    addresses: AddressContext,
    attr_info: &AttributeInfoMessage,
    fh_addr: StoredAddress,
    widths: FormatWidths,
    mut select: impl FnMut(MessageFlags) -> bool,
    mut visit: impl FnMut(&[u8], Option<u16>) -> Result<(), FormatError>,
) -> Result<(), FormatError> {
    let fh = crate::fractal_heap::read_header_from_source(source, addresses, fh_addr, widths)?;

    let btree_addr = attr_info
        .btree_name_index_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
    let btree_hdr = BTreeV2Header::parse_from_source_with_widths(
        &SourceMetadata(source),
        btree_addr.get(),
        widths,
    )?;
    let records =
        crate::btree_v2::collect_btree_v2_records_from_source(source, &btree_hdr, widths.offsets)?;

    let layout = FractalHeapIdLayout::new(&fh, widths.offsets, widths.lengths);
    let mut heap = HeapObjectReader::new(&fh, addresses, fh_addr, access_mode, widths);
    for record in &records {
        let flags_at = usize::from(fh.heap_id_length);
        let flags = record
            .data
            .get(flags_at)
            .copied()
            .ok_or(FormatError::UnexpectedEof {
                expected: flags_at + 1,
                available: record.data.len(),
            })?;
        if !select(MessageFlags::new(flags)) {
            continue;
        }
        let id_bytes =
            record
                .attribute_heap_id(fh.heap_id_length)
                .ok_or(FormatError::UnexpectedEof {
                    expected: flags_at,
                    available: record.data.len(),
                })?;
        let attr_data = heap.read_from_source(source, layout.parse(id_bytes)?)?;
        visit(&attr_data, record_creation_index(record, &fh, attr_info))?;
    }
    Ok(())
}

/// The creation index a name-index (type 8) record carries, for an object that
/// tracks attribute creation order.
///
/// The field sits right after the heap ID and the message flags byte, and is 4
/// bytes wide where the Attribute Info message's maximum is 2 — the reference C
/// library writes the same value into both, so anything past `u16` is a record
/// this crate did not write and cannot reproduce. An object that does not track
/// the order stores something else there (this crate's whole-file writer stores
/// the attribute's position), so the tracked flag gates the read.
fn record_creation_index(
    record: &crate::btree_v2::BTreeV2Record,
    fh: &FractalHeapHeader,
    attr_info: &AttributeInfoMessage,
) -> Option<u16> {
    attr_info.max_creation_index?;
    u16::try_from(record.attribute_creation_order(fh.heap_id_length)?).ok()
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;

    use hdf5_pure_format::__private::AttributeRecord;
    use test_util::btree_v2;
    use test_util::dataspace;
    use test_util::datatype;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::AddressContext;
    use crate::address::{BaseAddress, BaseAddressExt};
    use crate::data_read;
    use crate::message_flags::MessageFlags;
    use crate::source::BytesSource;

    const WIDTHS: Widths = Widths::EIGHT;
    const NODE_SIZE: u64 = 512;
    const NAME_RECORD_SIZE: u16 = AttributeRecord::name_record_size(8);

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
                u64::try_from(header_at).unwrap(),
                u64::try_from(header.len()).unwrap(),
            )
            .unwrap(),
            Extent::new(u64::try_from(root_at).unwrap(), NODE_SIZE).unwrap(),
        ]
    }

    #[test]
    fn dense_attribute_storage_walk_omits_creation_order_index_when_not_indexed() {
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

        let extents = collect_dense_attribute_index_storage_extents(
            &BytesSource::new(file),
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
    }

    impl CountingSource {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                inner: BytesSource::new(bytes),
                reads: RefCell::new(Vec::new()),
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
        let sig = crate::signature::find_signature(bytes).unwrap();
        let (superblock, widths) =
            hdf5_pure_format::__private::parse_superblock_with_widths(bytes, sig).unwrap();
        let root = ObjectHeader::parse_with_widths(
            bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(superblock.base_address)
                .absolute(superblock.root_group_address)
                .unwrap()
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

    /// The streaming dense walk resolves every huge object against one parse of
    /// the heap's huge-object index, not one parse per object.
    ///
    /// Costs, not answers, are what regress here: reading the index per object
    /// returns exactly the same attributes while making the walk quadratic in
    /// their number, so the count of reads is the only thing that catches it.
    /// This one counts them end to end, at the B-tree's own address, rather than
    /// on the reader.
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
            &source,
            AddressContext::from_effective_base(BaseAddress::new(0)),
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

    /// The buffered dense walk holds to the same invariant, on the path
    /// `File::open` takes.
    ///
    /// The reader caching the index is only half of it; the other half is each
    /// walk building one reader for the whole heap rather than one per object,
    /// and that half is per call site.
    #[test]
    fn a_buffered_dense_walk_parses_its_huge_object_index_once() {
        const COUNT: usize = 6;
        let bytes = file_with_huge_attributes(COUNT);
        let (info, fh_addr, widths) = dense_attribute_info(&bytes);

        crate::fractal_heap::reset_huge_index_decodes();
        let attrs = extract_dense_attributes(
            &bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(BaseAddress::ZERO),
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

    /// A heap holding no huge object never parses a huge-object index, on either
    /// backend. The index is parsed on demand, and every dense walk that reads
    /// only managed objects is a walk that must not pay for one.
    #[test]
    fn a_managed_dense_walk_never_parses_a_huge_object_index() {
        // Enough attributes to force dense storage, none of them large enough to
        // exceed the heap's managed-object limit.
        const COUNT: usize = 40;
        let bytes = file_with_managed_attributes(COUNT);
        let (info, fh_addr, widths) = dense_attribute_info(&bytes);

        crate::fractal_heap::reset_huge_index_decodes();
        let buffered = extract_dense_attributes(
            &bytes,
            AccessMode::ReadOnly,
            AddressContext::from_effective_base(BaseAddress::ZERO),
            &info,
            fh_addr,
            widths,
            None,
        )
        .unwrap();
        let source = BytesSource::new(bytes);
        let streamed = extract_dense_attributes_from_source(
            &source,
            AddressContext::from_effective_base(BaseAddress::new(0)),
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
