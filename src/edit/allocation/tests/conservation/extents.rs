//! Checks complete EA and manager allocations, including unused capacity.

use std::fs;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayDataBlock;
use hdf5_pure_format::__private::ExtensibleArrayDataBlockRef;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use rstest::rstest;
use test_util::allocation_oracle;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::AllocationState;
use test_util::allocation_oracle::Context;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::allocation_oracle::Violation;
use test_util::checksum;
use test_util::disk_log;
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;

use crate::Error;
use crate::File;
use crate::FileBuilder;
use crate::FormatError;
use crate::MaxExtent;
use crate::access_mode::AccessMode;
use crate::address::DecodedStoredAddress;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::WriteEngine;
use crate::edit::context::FileContext;
use crate::extensible_array;
use crate::free_sections::PersistedFreeSections;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source;
use crate::source::BytesSource;
use crate::width::FormatWidths;

use super::CHUNK;
use super::Fact;
use super::Observation;
use super::Workload;
use super::WritePath;

mod policy;

#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::owned(WritePath::Owned)]
fn complete_secondary_and_paged_allocations_preserve_hole_coverage(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("extents.h5");
    let original = super::pure_fixture(&path, paged, true, filtered);
    compact_geometry(&path);
    if !deferred {
        WritePath::Immediate.append(
            &path,
            &(0..i32::try_from(CHUNK).unwrap()).collect::<Vec<_>>(),
        );
    }
    let hole = original.intersection(&Observation::read(&fs::read(&path).unwrap()).free);
    assert!(!hole.is_empty());
    let workload = Workload::record(&path, writer, ROUNDS);
    let replay = directory.path().join("prefix.h5");
    let mut saw_provisional = false;
    let mut saw_index_provisional = false;
    let mut saw_uninitialized = false;
    let mut manager_generations = Vec::new();
    workload.replay(&replay, &hole, |point, observation, allocations| {
        let bytes = fs::read(&replay).unwrap();
        let storage = IndexStorage::read(&bytes);
        saw_uninitialized |= storage.paged.iter().any(|block| !block.spare().is_empty());
        saw_provisional |= allocations.iter().any(|allocation| {
            allocation.state() == AllocationState::Provisional
                && !Coverage::new([allocation.complete()]).intersection(&hole).is_empty()
        });
        saw_index_provisional |= allocations.iter().any(|allocation| allocation.kind() == AllocationKind::Index && allocation.state() == AllocationState::Provisional);
        let managers = observation.live.iter().filter(|fact| fact.kind == AllocationKind::Manager).map(|fact| fact.range).collect::<Vec<_>>();
        if manager_generations.last() != Some(&managers) {
            manager_generations.push(managers);
        }
        for allocation in &storage.allocations {
            assert!(Coverage::new([allocation.range]).intersection(&observation.free).is_empty(), "{writer:?}/{paged}/{deferred}/{filtered}, prefix {point}: complete index allocation {allocation:?} advertised free");
        }
    });
    let storage = IndexStorage::read(&workload.final_bytes);
    storage.assert_paths();
    assert!(
        saw_provisional,
        "{writer:?}/{paged}/{deferred}/{filtered}: no provisional storage from H"
    );
    assert!(saw_index_provisional);
    assert!(
        saw_uninitialized,
        "{writer:?}/{paged}/{deferred}/{filtered}: no uninitialized page storage"
    );
    assert!(
        manager_generations.len() > 1,
        "no manager relocation: {manager_generations:?}"
    );
}

#[rstest]
#[case::plain(false)]
#[case::filtered(true)]
fn fresh_paged_storage_is_covered_by_eof_before_references(#[case] filtered: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fresh.h5");
    let mut builder = FileBuilder::new();
    let target = builder
        .create_dataset("d")
        .with_unallocated_storage(crate::make_i32_type(), &[0]);
    target
        .with_chunks(&[u64::try_from(CHUNK).unwrap()])
        .with_maxshape(&[MaxExtent::Unlimited]);
    if filtered {
        target.with_fletcher32();
    }
    builder.write(&path).unwrap();
    compact_geometry(&path);
    let base = fs::read(&path).unwrap();
    let base_eof = crate::superblock::read(base.as_slice())
        .unwrap()
        .superblock
        .eof_address;
    disk_log::start();
    WritePath::Immediate.append(
        &path,
        &(0..ROUNDS * CHUNK)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>(),
    );
    let recording = disk_log::finish();
    assert!(!recording.operations.is_empty());
    assert!(!recording.events.is_empty());
    let final_bytes = fs::read(&path).unwrap();
    let mut image = base;
    let mut saw_paged = false;
    for (point, operation) in recording.operations.iter().enumerate() {
        operation.apply(&mut image);
        let storage = IndexStorage::read(&image);
        for block in &storage.paged {
            assert!(
                block.fact.range.start() >= base_eof,
                "prefix {point}: fresh block {:?} below old EOF {base_eof}",
                block.fact
            );
            saw_paged = true;
        }
        let file = File::from_bytes(image.clone()).unwrap();
        let values = file.dataset("d").unwrap().read_i32().unwrap();
        assert_eq!(
            values,
            (0..values.len())
                .map(|value| i32::try_from(value).unwrap())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(image, final_bytes);
    assert!(saw_paged);
    let storage = IndexStorage::read(&final_bytes);
    storage.assert_paths();
    assert_eq!(
        File::from_bytes(final_bytes)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..ROUNDS * CHUNK)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::uninitialized_pages(StorageControl::Pages)]
#[case::manager_capacity(StorageControl::Manager)]
fn free_storage_controls_include_complete_owned_capacity(#[case] control: StorageControl) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("capacity.h5");
    let hole = super::pure_fixture(&path, false, true, false);
    compact_geometry(&path);
    let workload = Workload::record(&path, WritePath::Immediate, ROUNDS);
    let final_bytes = match control {
        StorageControl::Manager => padded_manager(&workload.final_bytes),
        StorageControl::Pages => workload
            .snapshots
            .iter()
            .enumerate()
            .find_map(|(point, _)| {
                let bytes = workload.prefix(point);
                IndexStorage::read(&bytes)
                    .paged
                    .iter()
                    .any(|block| !block.spare().is_empty())
                    .then_some(bytes)
            })
            .unwrap(),
    };
    let observation = Observation::read(&final_bytes);
    let storage = IndexStorage::read(&final_bytes);
    let spare = match control {
        StorageControl::Manager => {
            let fact = observation
                .live
                .iter()
                .find(|fact| {
                    fact.kind == AllocationKind::Manager
                        && fact.range.length() > fact.parsed.length()
                })
                .unwrap();
            Interval::new(fact.parsed.end(), fact.range.end() - fact.parsed.end()).unwrap()
        }
        StorageControl::Pages => *storage
            .paged
            .iter()
            .flat_map(|block| block.spare().intervals().to_vec())
            .next()
            .as_ref()
            .unwrap(),
    };
    assert!(spare.length() > 0);
    let free = Coverage::new(observation.free.intervals().iter().copied().chain([spare]));
    let damaged = match control {
        StorageControl::Manager => advertise_capacity(&final_bytes, &free),
        StorageControl::Pages => super::replace_free_sections(&final_bytes, &free),
    };
    let observed = Observation::read(&damaged);
    assert_eq!(observed.values, observation.values);
    let live = observed
        .live
        .iter()
        .map(|fact| fact.allocation(AllocationState::Linked))
        .collect::<Vec<_>>();
    let error = allocation_oracle::inspect(
        Context {
            case: "complete storage advertised free",
            interruption: workload.operations.len(),
            publication: observed.state(),
        },
        &hole,
        &observed.free,
        &observed.reachable(),
        &live,
    )
    .unwrap_err();
    let Violation::LiveSpaceAdvertisedFree { conflicting } = error.violation else {
        panic!("expected live capacity conflict, got {error:?}")
    };
    assert_eq!(conflicting, Coverage::new([spare]));
    fs::write(&path, &damaged).unwrap();
    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        observation.values
    );
    drop(reader);
    assert_eq!(fs::read(&path).unwrap(), damaged);
}

#[rstest]
#[case::published_eof(ExtentBoundary::Published)]
#[case::physical_source(ExtentBoundary::Physical)]
fn complete_page_capacity_requires_both_file_bounds(#[case] boundary: ExtentBoundary) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounds.h5");
    super::pure_fixture(&path, false, true, false);
    compact_geometry(&path);
    let workload = Workload::record(&path, WritePath::Immediate, ROUNDS);
    let bytes = workload
        .snapshots
        .iter()
        .enumerate()
        .find_map(|(point, _)| {
            let bytes = workload.prefix(point);
            IndexStorage::read(&bytes)
                .paged
                .iter()
                .any(|block| !block.spare().is_empty())
                .then_some(bytes)
        })
        .unwrap();
    let storage = IndexStorage::read(&bytes);
    let block = storage
        .paged
        .iter()
        .find(|block| !block.spare().is_empty())
        .unwrap();
    let complete = FileRange::new(
        PhysicalAddress::new(block.fact.range.start()),
        block.fact.range.length(),
    )
    .unwrap();
    let boundary_at = complete.end().get() - 1;
    assert!(block.fact.parsed.end() <= boundary_at);
    match boundary {
        ExtentBoundary::Published => {
            let mut damaged = bytes.clone();
            let mut superblock = crate::superblock::read(damaged.as_slice())
                .unwrap()
                .superblock;
            superblock.eof_address = boundary_at;
            let encoded = format::serialize_superblock(&superblock).unwrap();
            damaged[..encoded.len()].copy_from_slice(&encoded);
            let before = File::from_bytes(bytes.clone())
                .unwrap()
                .dataset("d")
                .unwrap()
                .read_i32()
                .unwrap();
            assert_eq!(
                File::from_bytes(damaged.clone())
                    .unwrap()
                    .dataset("d")
                    .unwrap()
                    .read_i32()
                    .unwrap(),
                before
            );
            let parsed_eof = crate::superblock::read(damaged.as_slice())
                .unwrap()
                .superblock
                .eof_address;
            assert_eq!(
                read_extent(&damaged, parsed_eof, complete).unwrap_err(),
                FormatError::UnexpectedSourceEof {
                    expected: complete.end().get(),
                    available: boundary_at
                }
            );
        }
        ExtentBoundary::Physical => {
            let prefix = &bytes[..usize::try_from(boundary_at).unwrap()];
            assert_eq!(
                read_extent(prefix, u64::try_from(bytes.len()).unwrap(), complete).unwrap_err(),
                FormatError::UnexpectedSourceEof {
                    expected: complete.end().get(),
                    available: boundary_at
                }
            );
        }
    }
    let parsed = FileRange::new(
        PhysicalAddress::new(block.fact.parsed.start()),
        block.fact.parsed.length(),
    )
    .unwrap();
    read_extent(&bytes, boundary_at, parsed).unwrap();
}

#[rstest]
#[case::existing(false)]
#[case::deferred(true)]
fn allocated_section_list_capacity_remains_owned_until_replacement(#[case] deferred: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manager-capacity.h5");
    let hole = super::pure_fixture(&path, false, true, false);
    compact_geometry(&path);
    if !deferred {
        WritePath::Immediate.append(
            &path,
            &(0..i32::try_from(CHUNK).unwrap()).collect::<Vec<_>>(),
        );
    }
    let baseline = padded_manager(&fs::read(&path).unwrap());
    let initial = Observation::read(&baseline);
    let capacity = initial
        .live
        .iter()
        .find(|fact| {
            fact.kind == AllocationKind::Manager && fact.range.length() > fact.parsed.length()
        })
        .copied()
        .unwrap();
    assert_eq!(
        capacity.range.length() - capacity.parsed.length(),
        MANAGER_SPARE
    );
    let hole = hole.intersection(&initial.free);
    fs::write(&path, &baseline).unwrap();
    let workload = Workload::record(&path, WritePath::Immediate, ROUNDS);
    workload.replay(
        &directory.path().join("prefix.h5"),
        &hole,
        |point, observation, _| {
            if observation.live.contains(&capacity) {
                assert!(
                    Coverage::new([capacity.range])
                        .intersection(&observation.free)
                        .is_empty(),
                    "prefix {point}: live capacity {capacity:?} advertised free"
                );
            }
        },
    );
    assert!(!workload.snapshots.last().unwrap().live.contains(&capacity));
}

#[rstest]
#[case::published_eof(ExtentBoundary::Published)]
#[case::physical_source(ExtentBoundary::Physical)]
fn complete_manager_capacity_requires_both_file_bounds(#[case] boundary: ExtentBoundary) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manager-bounds.h5");
    super::pure_fixture(&path, false, true, false);
    let bytes = padded_manager(&fs::read(&path).unwrap());
    let observation = Observation::read(&bytes);
    let capacity = observation
        .live
        .iter()
        .find(|fact| {
            fact.kind == AllocationKind::Manager && fact.range.length() > fact.parsed.length()
        })
        .unwrap();
    let boundary_at = capacity.range.end() - 1;
    assert!(capacity.parsed.end() < boundary_at);
    let mut damaged = bytes.clone();
    match boundary {
        ExtentBoundary::Published => {
            let mut superblock = crate::superblock::read(bytes.as_slice())
                .unwrap()
                .superblock;
            superblock.eof_address = boundary_at;
            let encoded = format::serialize_superblock(&superblock).unwrap();
            damaged[..encoded.len()].copy_from_slice(&encoded);
        }
        ExtentBoundary::Physical => damaged.truncate(usize::try_from(boundary_at).unwrap()),
    }
    let context = FileContext::new(crate::superblock::read(damaged.as_slice()).unwrap()).unwrap();
    let (_, info) = WriteEngine::extension_fsinfo(&BytesSource::new(&damaged), &context)
        .unwrap()
        .unwrap();
    read_extent(
        &damaged,
        context.superblock().eof_address,
        FileRange::new(
            PhysicalAddress::new(capacity.parsed.start()),
            capacity.parsed.length(),
        )
        .unwrap(),
    )
    .unwrap();
    let error = PersistedFreeSections::read(
        &BytesSource::new(&damaged),
        context.widths(),
        context.superblock().base_address,
        context.superblock().eof_address,
        &info,
    )
    .map_err(Error::from)
    .unwrap_err();
    match boundary {
        ExtentBoundary::Published => {
            let Error::Format(FormatError::InvalidFreeSpaceManager(reason)) = error else {
                panic!("expected manager allocation beyond EOF, got {error:?}")
            };
            assert_eq!(
                reason,
                format!(
                    "the manager allocation at {}, {} bytes long, exceeds the file allocation",
                    capacity.range.start(),
                    capacity.range.length()
                )
            );
        }
        ExtentBoundary::Physical => {
            assert!(
                matches!(error, Error::Format(FormatError::UnexpectedSourceEof { expected, available }) if expected == capacity.range.end() && available == boundary_at),
                "{error:?}"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum StorageControl {
    Manager,
    Pages,
}

#[derive(Clone, Copy, Debug)]
enum ExtentBoundary {
    Physical,
    Published,
}

/// Complete EA allocation ranges and initialized page ranges for the fixture dataset.
pub(super) struct IndexStorage {
    /// The EA header, index block, secondary blocks, and complete data blocks.
    pub(super) allocations: Vec<Fact>,
    direct: usize,
    secondary: usize,
    paged: Vec<PagedStorage>,
}

impl IndexStorage {
    /// Reads the fixture's EA allocations and checks their physical and published bounds.
    ///
    /// Returns empty storage when the dataset's EA header address is undefined.
    pub(super) fn read(bytes: &[u8]) -> Self {
        let context = FileContext::new(crate::superblock::read(bytes).unwrap()).unwrap();
        let file = File::from_bytes(bytes.to_vec()).unwrap();
        let DataLayout::Chunked {
            index: ChunkIndexLayout::ExtensibleArray { address, .. },
            ..
        } = file.dataset("d").unwrap().data_layout().unwrap()
        else {
            panic!("expected an EA dataset")
        };
        let mut storage = Self {
            allocations: Vec::new(),
            direct: 0,
            secondary: 0,
            paged: Vec::new(),
        };
        let Some(address) = address else {
            return storage;
        };
        let source = BytesSource::new(bytes);
        let header = extensible_array::read_header(&source, context.widths(), address).unwrap();
        storage.push(
            bytes,
            context.superblock().eof_address,
            header.range(),
            header.range(),
        );
        let reference = header.index_block().unwrap();
        let range = reference.parse_range();
        let index = reference
            .parse(read_extent(bytes, context.superblock().eof_address, range).unwrap())
            .unwrap();
        storage.push(bytes, context.superblock().eof_address, range, range);
        for block in index.data_blocks() {
            storage.direct += 1;
            storage.data(bytes, context.superblock().eof_address, block.unwrap());
        }
        for secondary in index.super_blocks() {
            storage.secondary += 1;
            let secondary = secondary.unwrap();
            let range = secondary.parse_range();
            let parsed = secondary
                .parse(read_extent(bytes, context.superblock().eof_address, range).unwrap())
                .unwrap();
            storage.push(bytes, context.superblock().eof_address, range, range);
            for block in parsed.data_blocks() {
                storage.data(bytes, context.superblock().eof_address, block.unwrap());
            }
        }
        storage
    }

    /// Checks that direct, secondary, and paged blocks are present.
    pub(super) fn assert_paths(&self) {
        let Self {
            allocations: _,
            direct,
            secondary,
            paged,
        } = self;
        assert!(*direct > 0);
        assert!(*secondary > 0);
        assert!(!paged.is_empty());
        assert!(
            paged
                .iter()
                .any(|block| block.fact.range.length() > block.fact.parsed.length())
        );
    }

    fn push(&mut self, bytes: &[u8], eof: u64, range: FileRange, parsed: FileRange) {
        read_extent(bytes, eof, range).unwrap();
        self.allocations.push(Fact {
            kind: AllocationKind::Index,
            range: interval(range),
            parsed: interval(parsed),
        });
    }

    fn data(&mut self, bytes: &[u8], eof: u64, reference: ExtensibleArrayDataBlockRef<'_>) {
        let range = reference.range();
        let parsed = reference.parse_range();
        self.push(bytes, eof, range, parsed);
        let block = reference
            .parse(read_extent(bytes, eof, parsed).unwrap())
            .unwrap();
        if let ExtensibleArrayDataBlock::Pages(pages) = block {
            let mut initialized = vec![interval(parsed)];
            for page in pages.pages() {
                let page = page.unwrap();
                let range = page.parse_range();
                page.parse(read_extent(bytes, eof, range).unwrap()).unwrap();
                initialized.push(interval(range));
            }
            self.paged.push(PagedStorage {
                fact: *self.allocations.last().unwrap(),
                initialized: Coverage::new(initialized),
            });
        }
    }
}

/// A complete paged data block and its initialized prefix and pages.
struct PagedStorage {
    fact: Fact,
    initialized: Coverage,
}

impl PagedStorage {
    /// Returns the allocation bytes outside the initialized prefix and pages.
    fn spare(&self) -> Coverage {
        Coverage::new([self.fact.range]).difference(&self.initialized)
    }
}

fn interval(range: FileRange) -> Interval {
    Interval::new(range.start().get(), range.length()).unwrap()
}

/// Reads a complete range within both the physical bytes and published EOF.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedSourceEof`] if `range` exceeds either bound.
fn read_extent(bytes: &[u8], eof: u64, range: FileRange) -> Result<FileRegion<'_>, FormatError> {
    let region = source::bytes_in(bytes, range)?;
    if range.end().get() > eof {
        return Err(source::unexpected_eof(range, eof));
    }
    FileRegion::new(range, region)
}

/// Rewrites an empty fixture's EA parameters to exercise secondary and paged blocks.
///
/// Sets the EA header address to undefined before writing the layout message.
pub(super) fn compact_geometry(path: &Path) {
    let mut bytes = fs::read(path).unwrap();
    let (header, mut layout) = test_util::bytes::signature_offsets(&bytes, v2::SIGNATURE)
        .into_iter()
        .find_map(|at| {
            ObjectHeader::parse(&bytes, AccessMode::ReadOnly, at, 8, 8)
                .unwrap()
                .messages
                .into_iter()
                .find(|message| {
                    message.msg_type == MessageType::DATA_LAYOUT
                        && matches!(
                            DataLayout::parse(&message.data, 8, 8).unwrap(),
                            DataLayout::Chunked { .. }
                        )
                })
                .map(|message| (at, message.data))
        })
        .unwrap();
    let (_, pointer) =
        DataLayout::parse_with_ea_address(&layout, FormatWidths::from_sizes(8, 8).unwrap())
            .unwrap();
    let pointer = pointer.unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(file.dataset("d").unwrap().shape().unwrap(), [0]);
    layout[pointer.clone()].fill(u8::MAX);
    let parameters = ExtensibleArrayCreationParameters::parse(&COMPACT_GEOMETRY).unwrap();
    layout[pointer.start - COMPACT_GEOMETRY.len()..pointer.start]
        .copy_from_slice(&parameters.encode());
    v2::replace_message(&mut bytes, header, FixtureMessage::DATA_LAYOUT, &layout);
    fs::write(path, bytes).unwrap();
}

/// Replaces the fixture's manager with a section list whose allocation exceeds its used length.
///
/// Preserves the free sections and extends both physical storage and published EOF for the capacity.
pub(super) fn padded_manager(original: &[u8]) -> Vec<u8> {
    let observation = Observation::read(original);
    let mut bytes = super::replace_free_sections(original, &observation.free);
    let context = FileContext::new(crate::superblock::read(bytes.as_slice()).unwrap()).unwrap();
    let (_, info) = WriteEngine::extension_fsinfo(&BytesSource::new(&bytes), &context)
        .unwrap()
        .unwrap();
    let address = info
        .manager_addrs
        .iter()
        .find_map(|address| match address {
            DecodedStoredAddress::Defined(address) => Some(*address),
            _ => None,
        })
        .unwrap();
    let at = usize::try_from(address.get()).unwrap();
    let length = usize::try_from(format::free_space_manager_header_len(context.widths())).unwrap();
    let header = FreeSpaceManagerHeader::parse(context.widths(), &bytes[at..at + length]).unwrap();
    assert_eq!(
        header.section_list_addr().get() + header.section_list_allocated(),
        u64::try_from(bytes.len()).unwrap()
    );
    let allocated = header.section_list_allocated() + MANAGER_SPARE;
    let field = at + length - checksum::CHECKSUM - usize::from(context.widths().lengths.get());
    bytes[field..field + 8].copy_from_slice(&allocated.to_le_bytes());
    checksum::restamp(&mut bytes, at, length);
    bytes.resize(bytes.len() + usize::try_from(MANAGER_SPARE).unwrap(), 0);
    let mut superblock = context.superblock().clone();
    superblock.eof_address = u64::try_from(bytes.len()).unwrap();
    let encoded = format::serialize_superblock(&superblock).unwrap();
    bytes[..encoded.len()].copy_from_slice(&encoded);
    bytes
}

/// Rewrites free sections within the fixture manager's existing section-list allocation.
fn advertise_capacity(original: &[u8], free: &Coverage) -> Vec<u8> {
    let context = FileContext::new(crate::superblock::read(original).unwrap()).unwrap();
    let (_, info) = WriteEngine::extension_fsinfo(&BytesSource::new(original), &context)
        .unwrap()
        .unwrap();
    let address = info
        .manager_addrs
        .iter()
        .find_map(|address| match address {
            DecodedStoredAddress::Defined(address) => Some(*address),
            _ => None,
        })
        .unwrap();
    let start = usize::try_from(address.get()).unwrap();
    let length = usize::try_from(format::free_space_manager_header_len(context.widths())).unwrap();
    let old =
        FreeSpaceManagerHeader::parse(context.widths(), &original[start..start + length]).unwrap();
    let sections = free
        .intervals()
        .iter()
        .map(|range| (range.start(), range.length()))
        .collect::<Vec<_>>();
    let (mut header, list) = test_util::free_space::raw_manager_eight(
        address.get(),
        old.section_list_addr().get(),
        &sections,
    );
    assert!(u64::try_from(list.len()).unwrap() < old.section_list_allocated());
    let field = header.len() - checksum::CHECKSUM - usize::from(context.widths().lengths.get());
    header[field..field + 8].copy_from_slice(&old.section_list_allocated().to_le_bytes());
    let size = header.len();
    checksum::restamp(&mut header, 0, size);
    let mut bytes = original.to_vec();
    bytes[start..start + length].copy_from_slice(&header);
    let list_at = usize::try_from(old.section_list_addr().get()).unwrap();
    bytes[list_at..list_at + list.len()].copy_from_slice(&list);
    bytes
}

/// Rewrites the extension's File Space Info message in its containing object-header chunk.
///
/// Searches continuation chunks as well as the initial chunk and recomputes the chunk checksum.
pub(super) fn replace_space_info(
    bytes: &mut [u8],
    context: &FileContext,
    extension: u64,
    encoded: &[u8],
) {
    let chunks = crate::edit::object_header::read_oh_chunks(
        &BytesSource::new(&*bytes),
        extension,
        context.addresses(),
    )
    .unwrap();
    for chunk in chunks {
        let (region, mut position) = chunk.message_region();
        while let Some((kind, body, end)) =
            crate::edit::object_header::next_record(chunk.layout(), region, position).unwrap()
        {
            if kind == MessageType::FILE_SPACE_INFO {
                assert_eq!(encoded.len(), end - body);
                let start = usize::try_from(chunk.span.start().get()).unwrap();
                bytes[start + body..start + end].copy_from_slice(encoded);
                checksum::restamp(bytes, start, usize::try_from(chunk.span.length()).unwrap());
                return;
            }
            position = end;
        }
    }
    panic!("no File Space Info record in extension chunks")
}

/// The fixture's EA parameters for reaching direct, secondary, and paged blocks.
const COMPACT_GEOMETRY: [u8; 5] = [32, 1, 2, 1, 1];
/// The number of complete chunks appended to exercise every fixture index path.
pub(super) const ROUNDS: usize = 19;
/// The unused bytes added to a fixture's section-list allocation.
const MANAGER_SPARE: u64 = 73;
