//! Byte accounting for the storage fixtures.
//!
//! These fixtures use version 2 headers and ordinary contiguous, compact, or chunked datasets.
//! Variable-length values in the fixtures store top-level global heap references. Free space
//! comes from the live editing handle.

use std::collections::BTreeSet;

use test_util::bytes;
use test_util::global_heap;
use test_util_hdf5::reader::{ImageFile, ReadBackend};
use test_util_hdf5::storage;

use crate::access_mode::AccessMode;
use crate::address::StoredAddressExt;
use crate::attribute::{self, AttributeMessage};
use crate::attribute_info::AttributeInfoMessage;
use crate::data_layout::DataLayout;
use crate::fractal_heap_storage;
use crate::group;
use crate::link_info::LinkInfoMessage;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source::{PhysicalBytes, StoredBytes};
use crate::width::FormatWidths;
use crate::{AddressRange, Datatype, File, FreeSections, Layout, PhysicalAddress, StoredAddress};

use super::WriteEngine;

/// Stored allocations and live reusable sections in a controlled fixture.
///
/// Extents are physical (offset, length) pairs. Gaps include abandoned or untracked storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SpaceOwnership {
    pub(crate) physical_size: u64,
    pub(crate) logical_size: u64,
    pub(crate) userblock_space: Vec<(u64, u64)>,
    pub(crate) live_raw_space: Vec<(u64, u64)>,
    pub(crate) live_metadata_space: Vec<(u64, u64)>,
    pub(crate) free_space_manager_space: Vec<(u64, u64)>,
    pub(crate) reusable_free_space: Vec<(u64, u64)>,
    pub(crate) unclassified_space: Vec<(u64, u64)>,
    pub(crate) trailing_slack_space: Vec<(u64, u64)>,
}

impl WriteEngine {
    /// Accounts for the bytes visible through the fixture's handle, including buffered writes.
    pub(crate) fn storage_snapshot(&self) -> SpaceOwnership {
        let source = self.image();
        let snapshot = source
            .read_exact_at(0, usize::try_from(source.len()).unwrap())
            .unwrap();
        storage_snapshot(snapshot, self.reusable_free_sections().unwrap())
    }
}

/// Accounts for a fixture from its file image and its editing handle's reusable sections.
///
/// The same bytes supply the public path and dataset queries and the extent readers in
/// `test-util-hdf5`.
pub(crate) fn storage_snapshot(
    snapshot: Vec<u8>,
    free: FreeSections<PhysicalAddress>,
) -> SpaceOwnership {
    let image = ReadBackend::Bytes.open_image(snapshot.clone()).unwrap();
    let file = File::from_bytes(snapshot).unwrap();
    let bytes = file.as_bytes();
    let sb = file.superblock();
    let widths = FormatWidths {
        offsets: sb.offset_size,
        lengths: sb.length_size,
    };
    let base = file.resolve_address(StoredAddress::ZERO).unwrap();
    let physical_size = u64::try_from(bytes.len()).unwrap();
    let logical_size = sb.eof_address;
    let mut metadata = storage::file_metadata_extents(&image, bytes);
    let managers = storage::manager_extents(&image, bytes);
    metadata.retain(|range| !managers.contains(range));

    let mut fixture = FixtureStorage {
        source: PhysicalBytes::for_test(bytes, file.addresses())
            .stored()
            .unwrap(),
        image: &image,
        file: &file,
        widths,
        metadata,
        raw: Vec::new(),
        collections: BTreeSet::new(),
    };
    let mut groups = vec![file.root()];
    let root = file.root().header_address().unwrap();
    let mut seen = BTreeSet::new();
    while let Some(group) = groups.pop() {
        let address = group.header_address().unwrap();
        if !seen.insert(address) {
            continue;
        }
        fixture.header(address, address != root);
        for name in group.datasets().unwrap() {
            let dataset = group.dataset(&name).unwrap();
            let address = dataset.header_address().unwrap();
            if seen.insert(address) {
                fixture.header(address, true);
                match dataset.layout().unwrap() {
                    Layout::Contiguous {
                        address: Some(address),
                        size,
                    } if size != 0 => {
                        fixture
                            .raw
                            .push(AddressRange::new(PhysicalAddress::new(address), size).unwrap());
                    }
                    Layout::Chunked { .. } => {
                        fixture
                            .raw
                            .extend(dataset.chunks().unwrap().into_iter().map(|chunk| {
                                AddressRange::new(
                                    PhysicalAddress::new(chunk.address),
                                    chunk.storage_size,
                                )
                                .unwrap()
                            }));
                    }
                    _ => {}
                }
                if matches!(dataset.datatype().unwrap(), Datatype::VariableLength(_)) {
                    fixture.references(&dataset.read_raw().unwrap());
                }
            }
        }
        for name in group.groups().unwrap() {
            groups.push(group.group(&name).unwrap());
        }
    }
    fixture.raw.extend(fixture.collections.iter().map(|&start| {
        let at = usize::try_from(start.get()).unwrap();
        assert_eq!(&bytes[at..at + 4], global_heap::SIGNATURE);
        let len = bytes::uint_at(
            bytes,
            at + global_heap::SIZE_OFFSET,
            widths.lengths.get().into(),
        );
        AddressRange::new(start, len).unwrap()
    }));
    let mut result = SpaceOwnership {
        physical_size,
        logical_size,
        userblock_space: (base.get() > 0)
            .then_some((0, base.get()))
            .into_iter()
            .collect(),
        live_raw_space: extents(fixture.raw),
        live_metadata_space: extents(fixture.metadata),
        free_space_manager_space: extents(managers),
        reusable_free_space: free
            .sections()
            .iter()
            .map(|section| (section.address().get(), section.length().get()))
            .collect(),
        unclassified_space: Vec::new(),
        trailing_slack_space: (physical_size > logical_size)
            .then(|| (logical_size, physical_size - logical_size))
            .into_iter()
            .collect(),
    };
    let mut claims = result
        .userblock_space
        .iter()
        .chain(&result.live_raw_space)
        .chain(&result.live_metadata_space)
        .chain(&result.free_space_manager_space)
        .chain(&result.reusable_free_space)
        .copied()
        .collect::<Vec<_>>();
    claims.sort_unstable();
    let mut cursor = 0;
    for (start, length) in claims {
        assert!(
            cursor <= start,
            "allocations overlap at {start}, previous end {cursor}"
        );
        if cursor < start {
            result.unclassified_space.push((cursor, start - cursor));
        }
        cursor = start.checked_add(length).unwrap();
    }
    assert!(cursor <= logical_size && logical_size <= physical_size);
    if cursor < logical_size {
        result
            .unclassified_space
            .push((cursor, logical_size - cursor));
    }
    result
}

struct FixtureStorage<'a> {
    source: StoredBytes<'a>,
    file: &'a File,
    image: &'a ImageFile,
    widths: FormatWidths,
    raw: Vec<AddressRange<PhysicalAddress>>,
    metadata: Vec<AddressRange<PhysicalAddress>>,
    collections: BTreeSet<PhysicalAddress>,
}

impl FixtureStorage<'_> {
    fn references(&mut self, values: &[u8]) {
        let width = usize::from(self.widths.offsets.get());
        assert_eq!(values.len() % (width + 8), 0);
        for value in values.chunks_exact(width + 8) {
            let raw_address = bytes::uint_at(value, 4, width);
            let index = bytes::uint_at(value, 4 + width, 4);
            if index != 0 {
                self.collections.insert(
                    self.file
                        .resolve_address(StoredAddress::new(raw_address))
                        .unwrap(),
                );
            }
        }
    }

    fn header(&mut self, address: u64, include_header: bool) {
        let bytes = self.file.as_bytes();
        let physical = PhysicalAddress::new(address);
        if include_header {
            self.metadata
                .extend(storage::header_extents(self.image, bytes, physical));
        }
        let header = ObjectHeader::parse_with_base_and_widths(
            bytes,
            AccessMode::ReadOnly,
            usize::try_from(address).unwrap(),
            self.widths,
            self.file.base_address(),
        )
        .unwrap();
        for message in &header.messages {
            match message.msg_type {
                MessageType::DATA_LAYOUT => {
                    let layout = DataLayout::parse_with_widths(&message.data, self.widths).unwrap();
                    if matches!(layout, DataLayout::Chunked { .. }) {
                        let index = crate::chunked_read::chunk_index_spans_from_source(
                            &self.source,
                            &layout,
                            self.widths,
                        )
                        .unwrap();
                        self.metadata.extend(
                            index
                                .into_iter()
                                .map(|range| resolve(self.file, range.start(), range.length())),
                        );
                    }
                }
                MessageType::LINK_INFO => {
                    let info = LinkInfoMessage::parse_with_offset_width(
                        &message.data,
                        self.widths.offsets,
                    )
                    .unwrap();
                    if let Some(heap) = info.fractal_heap_address {
                        let index = group::v2::collect_dense_group_index_storage_extents(
                            &self.source,
                            &info,
                            self.widths,
                        )
                        .unwrap();
                        self.metadata.extend(
                            index
                                .iter()
                                .map(|range| resolve(self.file, range.start(), range.len().get())),
                        );
                        self.heap(heap);
                    }
                }
                MessageType::ATTRIBUTE_INFO => {
                    let info = AttributeInfoMessage::parse_with_offset_width(
                        &message.data,
                        self.widths.offsets,
                    )
                    .unwrap();
                    if let Some(heap) = info.fractal_heap_address() {
                        let index = attribute::collect_dense_attribute_index_storage_extents(
                            &self.source,
                            &info,
                            self.widths,
                        )
                        .unwrap();
                        self.metadata.extend(
                            index
                                .iter()
                                .map(|range| resolve(self.file, range.start(), range.len().get())),
                        );
                        self.heap(heap);
                        let mut messages = Vec::new();
                        attribute::visit_dense_attribute_messages_from_source(
                            &self.source,
                            AccessMode::ReadOnly,
                            &info,
                            heap,
                            self.widths,
                            None,
                            |_| true,
                            |data, _| {
                                messages
                                    .push(AttributeMessage::parse_with_widths(data, self.widths)?);
                                Ok(())
                            },
                        )
                        .unwrap();
                        for message in messages {
                            self.attribute(message);
                        }
                    }
                }
                MessageType::ATTRIBUTE => self.attribute(
                    AttributeMessage::parse_with_widths(&message.data, self.widths).unwrap(),
                ),
                _ => {}
            }
        }
    }

    fn heap(&mut self, address: StoredAddress) {
        let heap = fractal_heap_storage::collect_fractal_heap_storage_from_source(
            &self.source,
            address,
            self.widths,
        )
        .unwrap();
        self.metadata.extend(
            heap.allocations()
                .iter()
                .map(|(range, _)| resolve(self.file, range.start(), range.len().get())),
        );
    }

    fn attribute(&mut self, message: AttributeMessage) {
        if matches!(message.datatype, Datatype::VariableLength(_)) {
            self.references(&message.raw_data);
        }
    }
}

fn resolve(file: &File, start: StoredAddress, length: u64) -> AddressRange<PhysicalAddress> {
    AddressRange::new(file.resolve_address(start).unwrap(), length).unwrap()
}

fn extents(ranges: Vec<AddressRange<PhysicalAddress>>) -> Vec<(u64, u64)> {
    ranges
        .into_iter()
        .map(|range| (range.start().get(), range.length()))
        .collect()
}
