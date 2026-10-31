//! Derives allocation publication roles from fixture metadata and recorded byte operations.
//!
//! The tests use fixtures with a zero base address, eight-byte addresses and lengths, and completed
//! full chunks. The tests derive allocation ranges from the final chunk index and compare them with
//! the starting file. The tests identify preparation writes from allocation ranges and reference
//! writes from stored pointer fields, then compare their event positions with ownership publication
//! in each trace.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::disk_log::DiskOp;
use test_util::disk_log::Recording;
use test_util::publication_order::EofChange;

use crate::File;
use crate::access_mode::AccessMode;
use crate::address::DecodedStoredAddress;
use crate::address::{AddressContext, BaseAddress, BaseAddressExt};
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::object_header;
use crate::extensible_array;
use crate::free_sections::PersistedManagerSet;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source::BytesSource;

/// Parsed allocations and their writes for one persisted manager and EOF generation.
#[derive(Debug)]
pub(super) struct PublicationFacts {
    pub(super) operation: usize,
    pub(super) prepared: Vec<usize>,
    pub(super) allocation_content: Vec<usize>,
    pub(super) references: Vec<usize>,
    pub(super) allocations: Vec<FileRange>,
    pub(super) eof: EofChange,
}

impl PublicationFacts {
    /// Locates preparation, ownership publication, and reference writes for new allocations.
    ///
    /// # Panics
    ///
    /// Panics if fixture metadata does not parse or the base address is nonzero. Also panics if
    /// preparation, ownership publication, or reference writes cannot be located for each newly
    /// reachable allocation, or an ownership publication has no newly reachable allocation.
    pub(super) fn read(base: &[u8], final_bytes: &[u8], recording: &Recording) -> Vec<Self> {
        let allocations = DatasetGeometry::parse(final_bytes)
            .issued_after(&DatasetGeometry::parse(base), final_bytes);
        assert!(!allocations.is_empty(), "no newly reachable allocations");
        let mut snapshots = Vec::new();
        let mut image = base.to_vec();
        let mut previous_eof = Ownership::parse(base).eof;
        let mut start = 0;
        for event in &recording.events {
            let Some(operation) = event.operation() else {
                continue;
            };
            recording.operations[operation].apply(&mut image);
            if !matches!(
                &recording.operations[operation],
                DiskOp::Write { offset: 0, .. }
            ) {
                continue;
            }
            let ownership = Ownership::parse(&image);
            let prepared = ownership
                .initialized
                .iter()
                .flat_map(|range| {
                    prepared_operations(
                        recording,
                        *range,
                        start,
                        Preparation::Exact(region(&image, *range)),
                    )
                })
                .collect();
            let eof = ownership.eof;
            snapshots.push((
                Self {
                    operation,
                    prepared,
                    allocation_content: Vec::new(),
                    references: Vec::new(),
                    allocations: Vec::new(),
                    eof: if ownership.eof == previous_eof {
                        EofChange::Unchanged
                    } else {
                        EofChange::Extend
                    },
                },
                ownership,
            ));
            previous_eof = eof;
            start = operation + 1;
        }
        for allocation in allocations {
            let expected = region(final_bytes, allocation.range);
            let preparation = match allocation.kind {
                AllocationKind::Index => Preparation::Signature(
                    &expected[..test_util::extensible_array::SIGNATURE.len()],
                ),
                AllocationKind::Payload => Preparation::Exact(expected),
            };
            let prepared = prepared_operations(recording, allocation.range, 0, preparation);
            let completed = *prepared.iter().max().unwrap();
            let (publication, _) = snapshots.iter_mut().find(|(publication, ownership)| publication.operation >= completed && ownership.owns(allocation.range)).unwrap_or_else(|| panic!("no persisted manager/EOF generation owns {:?} after preparation operations {prepared:?}", allocation.range));
            publication.allocation_content.extend(&prepared);
            publication.prepared.extend(prepared);
            publication
                .references
                .push(allocation.first_reference(recording));
            publication.allocations.push(allocation.range);
        }
        snapshots
            .into_iter()
            .map(|(publication, _)| {
                assert!(
                    !publication.allocations.is_empty(),
                    "ownership publication has no parsed allocations: {publication:?}"
                );
                publication
            })
            .collect()
    }
    /// Replaces an operation index and adjusts subsequent indices for its removal.
    pub(super) fn fold_operation(&mut self, removed: usize, replacement: usize) {
        for operation in std::iter::once(&mut self.operation)
            .chain(&mut self.prepared)
            .chain(&mut self.allocation_content)
            .chain(&mut self.references)
        {
            if *operation == removed {
                *operation = replacement;
            } else if *operation > removed {
                *operation -= 1;
            }
        }
    }
}

struct DatasetGeometry {
    index: Vec<FileRange>,
    payloads: Vec<FileRange>,
    layout_pointer: FileRange,
}

impl DatasetGeometry {
    fn parse(bytes: &[u8]) -> Self {
        let located = crate::superblock::read(bytes).unwrap();
        assert_eq!(located.superblock.base_address.get(), 0);
        let widths = located.widths;
        let header =
            test_util::bytes::signature_offsets(bytes, test_util::object_header::v2::SIGNATURE)
                .into_iter()
                .find(|&at| {
                    ObjectHeader::parse(
                        bytes,
                        AccessMode::ReadOnly,
                        at,
                        widths.offsets.get(),
                        widths.lengths.get(),
                    )
                    .unwrap()
                    .messages
                    .iter()
                    .any(|message| message.msg_type == MessageType::DATA_LAYOUT)
                })
                .unwrap();
        let source = BytesSource::new(bytes);
        let addresses = AddressContext::from_effective_base(located.superblock.base_address);
        let mut parsed = None;
        for chunk in
            object_header::read_oh_chunks(&source, u64::try_from(header).unwrap(), addresses)
                .unwrap()
        {
            let (region, mut position) = chunk.message_region();
            while let Some((kind, body, end)) =
                object_header::next_record(chunk.layout(), region, position).unwrap()
            {
                if kind == MessageType::DATA_LAYOUT {
                    let (layout, pointer) =
                        DataLayout::parse_with_ea_address(&region[body..end], widths).unwrap();
                    let pointer = pointer.unwrap();
                    let start = chunk
                        .span
                        .start()
                        .checked_add(u64::try_from(body + pointer.start).unwrap())
                        .unwrap();
                    let pointer =
                        FileRange::new(start, u64::try_from(pointer.len()).unwrap()).unwrap();
                    let DataLayout::Chunked {
                        index: ChunkIndexLayout::ExtensibleArray { address, .. },
                        ..
                    } = layout
                    else {
                        panic!("the fixture requires an EA index")
                    };
                    parsed = Some((address, pointer));
                }
                position = end;
            }
        }
        let (address, layout_pointer) = parsed.unwrap();
        let index = address.map_or_else(Vec::new, |address| {
            extensible_array::index_spans(&source, widths, address)
                .unwrap()
                .into_iter()
                .map(|range| {
                    FileRange::new(
                        BaseAddress::ZERO.resolve(range.start()).unwrap(),
                        range.length(),
                    )
                    .unwrap()
                })
                .collect()
        });
        let file = File::from_bytes(bytes.to_vec()).unwrap();
        let payloads = file
            .dataset("d")
            .unwrap()
            .chunks()
            .unwrap()
            .into_iter()
            .map(|chunk| {
                FileRange::new(PhysicalAddress::new(chunk.address), chunk.storage_size).unwrap()
            })
            .collect();
        Self {
            index,
            payloads,
            layout_pointer,
        }
    }

    fn issued_after(&self, before: &Self, bytes: &[u8]) -> Vec<Allocation> {
        let mut allocations = Vec::new();
        let skip = if before.index.is_empty() {
            let header = self.index[0];
            let index = self.index[1];
            assert_eq!(header.end(), index.start());
            allocations.push(Allocation {
                range: FileRange::new(header.start(), index.end().get() - header.start().get())
                    .unwrap(),
                fields: vec![self.layout_pointer],
                kind: AllocationKind::Index,
            });
            2
        } else {
            0
        };
        for &range in self
            .index
            .iter()
            .skip(skip)
            .filter(|range| !before.index.contains(range))
            .chain(
                self.payloads
                    .iter()
                    .filter(|range| !before.payloads.contains(range)),
            )
        {
            assert!(range.end().get() <= u64::try_from(bytes.len()).unwrap());
            let encoded = range.start().get().to_le_bytes();
            let fields: Vec<_> = self
                .index
                .iter()
                .filter(|parent| **parent != range)
                .flat_map(|parent| {
                    let start = usize::try_from(parent.start().get()).unwrap();
                    let end = usize::try_from(parent.end().get()).unwrap();
                    test_util::bytes::signature_offsets(&bytes[start..end], &encoded)
                        .into_iter()
                        .map(move |local| {
                            FileRange::new(
                                PhysicalAddress::new(u64::try_from(start + local).unwrap()),
                                u64::try_from(encoded.len()).unwrap(),
                            )
                            .unwrap()
                        })
                })
                .collect();
            assert_eq!(
                fields.len(),
                1,
                "allocation {range:?} must have one independently located index pointer, got {fields:?}"
            );
            allocations.push(Allocation {
                range,
                fields,
                kind: if self.index.contains(&range) {
                    AllocationKind::Index
                } else {
                    AllocationKind::Payload
                },
            });
        }
        allocations
    }
}

struct Allocation {
    range: FileRange,
    fields: Vec<FileRange>,
    kind: AllocationKind,
}

impl Allocation {
    fn first_reference(&self, recording: &Recording) -> usize {
        let address = self.range.start().get().to_le_bytes();
        recording
            .events
            .iter()
            .find_map(|event| {
                let operation = event.operation()?;
                let DiskOp::Write { offset, bytes } = &recording.operations[operation] else {
                    return None;
                };
                self.fields
                    .iter()
                    .any(|field| {
                        let Some(local) = field
                            .start()
                            .get()
                            .checked_sub(*offset)
                            .and_then(|offset| usize::try_from(offset).ok())
                        else {
                            return false;
                        };
                        bytes.get(local..local + address.len()) == Some(address.as_slice())
                    })
                    .then_some(operation)
            })
            .unwrap_or_else(|| {
                panic!(
                    "no write publishes a reference to {:?} through {:?}",
                    self.range, self.fields
                )
            })
    }
}

enum AllocationKind {
    Index,
    Payload,
}

struct Ownership {
    eof: u64,
    free: Coverage,
    initialized: Vec<FileRange>,
}

impl Ownership {
    fn parse(bytes: &[u8]) -> Self {
        let located = crate::superblock::read(bytes).unwrap();
        let widths = located.widths;
        let superblock = located.superblock;
        let mut ownership = Self {
            eof: superblock.eof_address,
            free: Coverage::default(),
            initialized: Vec::new(),
        };
        let Some(extension) = superblock
            .superblock_extension_address
            .filter(|address| !address.is_undefined(widths.offsets))
        else {
            return ownership;
        };
        let addresses = AddressContext::from_effective_base(superblock.base_address);
        let source = BytesSource::new(bytes);
        let extension = addresses.resolve(extension).unwrap().get();
        let object = ObjectHeader::parse(
            bytes,
            AccessMode::ReadOnly,
            usize::try_from(extension).unwrap(),
            widths.offsets.get(),
            widths.lengths.get(),
        )
        .unwrap();
        let Some(message) = object
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::FILE_SPACE_INFO)
        else {
            return ownership;
        };
        let info = format::parse_file_space_info(widths, &message.data).unwrap();
        if !info.persist {
            return ownership;
        }
        let (report, metadata) = PersistedManagerSet::read(
            &source,
            widths,
            superblock.base_address,
            superblock.eof_address,
            &info,
        )
        .unwrap()
        .into_parts();
        ownership.free = Coverage::new(report.free_sections().sections().iter().map(|section| {
            Interval::new(
                section.address().get(),
                section.end_address().get() - section.address().get(),
            )
            .unwrap()
        }));
        ownership.initialized.extend(
            object_header::read_oh_chunks(&source, extension, addresses)
                .unwrap()
                .into_iter()
                .map(|chunk| chunk.span),
        );
        for address in &info.manager_addrs {
            let DecodedStoredAddress::Defined(address) = address else {
                continue;
            };
            let at = addresses.resolve(*address).unwrap().get();
            let header =
                FreeSpaceManagerHeader::parse(widths, &bytes[usize::try_from(at).unwrap()..])
                    .unwrap();
            ownership.initialized.push(
                *metadata
                    .iter()
                    .find(|range| range.start().get() == at)
                    .unwrap(),
            );
            if !header.section_list_addr().is_undefined(widths.offsets) {
                ownership.initialized.push(
                    FileRange::new(
                        addresses.resolve(header.section_list_addr()).unwrap(),
                        header.section_list_used(),
                    )
                    .unwrap(),
                );
            }
        }
        ownership
    }

    fn owns(&self, range: FileRange) -> bool {
        range.end().get() <= self.eof
            && self
                .free
                .intersection(&Coverage::new([Interval::new(
                    range.start().get(),
                    range.length(),
                )
                .unwrap()]))
                .is_empty()
    }
}

fn prepared_operations(
    recording: &Recording,
    range: FileRange,
    first_operation: usize,
    preparation: Preparation<'_>,
) -> Vec<usize> {
    let required = Coverage::new([Interval::new(range.start().get(), range.length()).unwrap()]);
    let mut written = Coverage::default();
    let mut operations = Vec::new();
    let mut signature_written = false;
    for event in &recording.events {
        let Some(operation) = event
            .operation()
            .filter(|operation| *operation >= first_operation)
        else {
            continue;
        };
        let DiskOp::Write { offset, bytes } = &recording.operations[operation] else {
            continue;
        };
        let coverage =
            Coverage::new([Interval::new(*offset, u64::try_from(bytes.len()).unwrap()).unwrap()])
                .intersection(&required);
        if coverage.is_empty() {
            continue;
        }
        let matches = match preparation {
            Preparation::Exact(expected) => coverage.intervals().iter().all(|interval| {
                let written_start = usize::try_from(interval.start() - offset).unwrap();
                let expected_start =
                    usize::try_from(interval.start() - range.start().get()).unwrap();
                let length = usize::try_from(interval.length()).unwrap();
                bytes.get(written_start..written_start + length)
                    == expected.get(expected_start..expected_start + length)
            }),
            Preparation::Signature(signature) => {
                if !signature_written {
                    signature_written = range
                        .start()
                        .get()
                        .checked_sub(*offset)
                        .and_then(|local| usize::try_from(local).ok())
                        .is_some_and(|local| {
                            bytes.get(local..local + signature.len()) == Some(signature)
                        });
                }
                signature_written
            }
        };
        if !matches {
            continue;
        }
        operations.push(operation);
        written = Coverage::new(
            written
                .intervals()
                .iter()
                .copied()
                .chain(coverage.intervals().iter().copied()),
        );
        if required.difference(&written).is_empty() {
            return operations;
        }
    }
    panic!(
        "allocation {range:?} is not initialized after operation {first_operation}; missing {:?}",
        required.difference(&written)
    );
}

#[derive(Clone, Copy)]
enum Preparation<'a> {
    Exact(&'a [u8]),
    Signature(&'a [u8]),
}

fn region(bytes: &[u8], range: FileRange) -> &[u8] {
    &bytes
        [usize::try_from(range.start().get()).unwrap()..usize::try_from(range.end().get()).unwrap()]
}
