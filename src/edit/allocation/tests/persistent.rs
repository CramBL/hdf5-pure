use std::fs;
use std::path::Path;

use hdf5_pure_core::__private::{FileRange, PhysicalAddress};
use hdf5_pure_format::__private::{self as format, FreeSpaceManagerHeader};
use hdf5_pure_space::__private::AllocationClass;
use rstest::rstest;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::chunk_index_inplace::Store;
use crate::data_layout::DataLayout;
use crate::edit::context::FileContext;
use crate::edit::object_header;
use crate::edit::{AppendBuilder, AppendTarget, WriteEngine};
use crate::free_sections::{FreeSpaceManagerRole, PersistedManagerSet};
use crate::object_path::ObjectPathBuf;
use crate::source::{BytesSource, MetadataCacheConfig};
use crate::type_builders::DatasetBuilder;
use crate::{
    Error, File, FileBuilder, FileLocking, FileSpaceSettings, FileSpaceStrategy, FormatError,
    MaxExtent, MemoryStrategy, PersistedFreeSpacePolicy, SyncPolicy,
};

#[rstest]
#[case::flat_mirror(PersistentStrategy::Flat, MemoryStrategy::Mirrored, SyncPolicy::Always)]
#[case::flat_bounded(PersistentStrategy::Flat, MemoryStrategy::Bounded, SyncPolicy::OnClose)]
#[case::page_mirror(PersistentStrategy::Page, MemoryStrategy::Mirrored, SyncPolicy::Always)]
#[case::page_bounded(PersistentStrategy::Page, MemoryStrategy::Bounded, SyncPolicy::OnClose)]
fn provisional_ownership_preserves_every_unissued_hole_interval(
    #[case] strategy: PersistentStrategy,
    #[case] backing: MemoryStrategy,
    #[case] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("exact.h5");
    let hole = fixture(&path, strategy);
    let mut allocations = Vec::new();
    for cycle in 0..3 {
        let mut engine = open(&path, backing, policy);
        let root = engine.context.superblock().root_group_address;
        let mut store = engine.store();
        let payload = vec![u8::try_from(cycle + 1).unwrap(); 8192];
        let allocation = store.allocate(&payload, AllocationClass::Raw).unwrap();
        let range = allocation.range();
        assert!(
            range.start().get() >= hole.0 && range.end().get() <= hole.1,
            "cycle {cycle}: payload {range:?} does not reuse {hole:?}"
        );
        engine.image.flush().unwrap();
        let provisional = fs::read(&path).unwrap();
        let locations: Vec<_> = provisional
            .windows(payload.len())
            .enumerate()
            .filter_map(|(start, bytes)| (bytes == payload).then_some(start))
            .collect();
        assert_eq!(
            locations.len(),
            1,
            "cycle {cycle}: payload locations {locations:?}"
        );
        let payload_start = u64::try_from(locations[0]).unwrap();
        let actual = (
            payload_start,
            payload_start + u64::try_from(payload.len()).unwrap(),
        );
        assert_eq!(actual, (range.start().get(), range.end().get()));
        let OwnershipObservation { free, metadata } = observe(&provisional);
        assert_conserved(hole, &allocations, &metadata, &free, cycle, "provisional");
        let interrupted = directory.path().join(format!("provisional-{cycle}.h5"));
        fs::write(&interrupted, &provisional).unwrap();
        let file = File::open(&interrupted).unwrap();
        assert_eq!(
            file.dataset("values").unwrap().read_i32().unwrap(),
            [0, 1, 2, 3]
        );
        assert_eq!(
            file.dataset("ceiling").unwrap().read_u8().unwrap(),
            [9; 8192]
        );
        drop(file);
        let mut reopened = open(&interrupted, backing, policy);
        let mut allocation_store = reopened.store();
        let reused = allocation_store
            .allocate(&vec![41; payload.len()], AllocationClass::Raw)
            .unwrap();
        assert_eq!(
            reused.range(),
            range,
            "cycle {cycle}: unpublished payload storage was lost"
        );
        drop(reopened);
        let mut store = engine.store();
        let publication = store.publish_allocations().unwrap();
        let confirmed = publication.confirm(allocation).unwrap();
        assert_eq!(
            store.allocation_publication().require(&confirmed).unwrap(),
            range
        );
        allocations.push(actual);
        assert_eq!(engine.context.superblock().root_group_address, root);
        let published = fs::read(&path).unwrap();
        let OwnershipObservation { free, metadata } = observe(&published);
        assert_conserved(hole, &allocations, &metadata, &free, cycle, "confirmed");
        assert_eq!(
            &published[range.start().get() as usize..range.end().get() as usize],
            &payload
        );
        drop(engine);
        assert_eq!(fs::read(&path).unwrap(), published);
        let reopened = File::open(&path).unwrap();
        assert_eq!(
            reopened.dataset("values").unwrap().read_i32().unwrap(),
            [0, 1, 2, 3]
        );
        assert_eq!(
            reopened.dataset("ceiling").unwrap().read_u8().unwrap(),
            [9; 8192]
        );
        drop(reopened);
        assert_eq!(fs::read(&path).unwrap(), published);
    }
    let bytes = fs::read(&path).unwrap();
    let OwnershipObservation { free, .. } = observe(&bytes);
    let candidate = free
        .iter()
        .filter(|&&(start, end)| start >= hole.0 && end <= hole.1)
        .max_by_key(|&&(start, end)| end - start)
        .copied()
        .unwrap();
    let length = candidate.1 - candidate.0 - 128;
    assert!(length > 8192);
    assert!(
        free.iter()
            .all(|&range| range == candidate || range.1 - range.0 < length),
        "the fixture has another fitting interval: {free:?}"
    );
    let mut engine = open(&path, backing, policy);
    let mut store = engine.store();
    let allocation = store
        .allocate(
            &vec![17; usize::try_from(length).unwrap()],
            AllocationClass::Raw,
        )
        .unwrap();
    assert_eq!(allocation.range().start().get(), candidate.0);
    store
        .publish_allocations()
        .unwrap()
        .confirm(allocation)
        .unwrap();
}

#[rstest]
#[case::flat(PersistentStrategy::Flat)]
#[case::page(PersistentStrategy::Page)]
fn manager_publication_preserves_unrelated_staged_edits(#[case] strategy: PersistentStrategy) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged.h5");
    fixture(&path, strategy);
    let mut engine = open(&path, MemoryStrategy::Bounded, SyncPolicy::OnClose);
    engine
        .create_group(&ObjectPathBuf::parse("pending"))
        .unwrap();
    let mut values = AppendBuilder::new();
    values.append_i32(&[4, 5, 6, 7]);
    engine
        .append_inplace_gathered(
            AppendTarget::Path(&ObjectPathBuf::parse("values")),
            &values,
            4,
        )
        .unwrap();
    assert!(engine.has_staged_edits());
    let snapshot = File::from_bytes(fs::read(&path).unwrap()).unwrap();
    assert_eq!(snapshot.root().groups().unwrap(), Vec::<String>::new());
    assert_eq!(
        snapshot.dataset("values").unwrap().read_i32().unwrap(),
        (0..8).collect::<Vec<_>>()
    );
    drop(snapshot);
    engine.commit().unwrap();
    let snapshot = File::from_bytes(fs::read(&path).unwrap()).unwrap();
    assert_eq!(snapshot.root().groups().unwrap(), ["pending"]);
}

#[rstest]
#[case::flat(PersistentStrategy::Flat)]
#[case::page(PersistentStrategy::Page)]
fn a_prior_page_buffered_commit_is_issued_before_reuse(#[case] strategy: PersistentStrategy) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("prior-publication.h5");
    fixture(&path, strategy);
    let mut engine = open(&path, MemoryStrategy::Bounded, SyncPolicy::OnClose);
    engine.set_page_buffer_size(1 << 20).unwrap();
    let old = crate::superblock::read(fs::read(&path).unwrap().as_slice())
        .unwrap()
        .superblock
        .root_group_address;
    engine
        .create_group(&ObjectPathBuf::parse("published"))
        .unwrap();
    let mut anchor = DatasetBuilder::new("");
    anchor.with_u8_data(&vec![53; 1 << 17]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("anchor"), anchor)
        .unwrap();
    engine.commit().unwrap();
    let current = engine.context.superblock().root_group_address;
    assert_ne!(old, current);
    let before = fs::read(&path).unwrap();
    assert_ne!(
        crate::superblock::read(before.as_slice())
            .unwrap()
            .superblock
            .root_group_address,
        current
    );
    let mut store = engine.store();
    let allocation = store.allocate(&[37; 8192], AllocationClass::Raw).unwrap();
    let issued = fs::read(&path).unwrap();
    assert_eq!(
        crate::superblock::read(issued.as_slice())
            .unwrap()
            .superblock
            .root_group_address,
        current
    );
    let snapshot = File::from_bytes(issued).unwrap();
    assert_eq!(snapshot.root().groups().unwrap(), ["published"]);
    assert_eq!(
        snapshot.dataset("anchor").unwrap().read_u8().unwrap(),
        vec![53; 1 << 17]
    );
    assert_eq!(
        snapshot.dataset("values").unwrap().read_i32().unwrap(),
        [0, 1, 2, 3]
    );
    drop(snapshot);
    let mut store = engine.store();
    store
        .publish_allocations()
        .unwrap()
        .confirm(allocation)
        .unwrap();
    assert_eq!(engine.context.superblock().root_group_address, current);
}

#[rstest]
#[case::commit(Completion::Commit)]
#[case::finalize(Completion::Finalize)]
#[case::sync(Completion::Sync)]
fn lifecycle_operations_require_pending_allocation_confirmation(
    #[case] operation: Completion,
    #[values(PersistentStrategy::Flat, PersistentStrategy::Page)] strategy: PersistentStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pending-lifecycle.h5");
    fixture(&path, strategy);
    let mut engine = open(&path, MemoryStrategy::Bounded, SyncPolicy::OnClose);
    engine
        .create_group(&ObjectPathBuf::parse("pending"))
        .unwrap();
    let allocation = engine
        .store()
        .allocate(&[47; 8192], AllocationClass::Raw)
        .unwrap();
    let before = fs::read(&path).unwrap();
    let error = operation.apply(&mut engine).unwrap_err();
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected provisional allocation rejection, got {error:?}");
    };
    assert_eq!(
        reason,
        "reference publication precedes allocation ownership publication"
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(engine.has_staged_edits());
    let publication = engine.store().publish_allocations().unwrap();
    let allocation = publication.confirm(allocation).unwrap();
    operation.apply(&mut engine).unwrap();
    assert_eq!(
        engine.has_staged_edits(),
        !matches!(operation, Completion::Commit)
    );
    let OwnershipObservation { free, .. } = observe(&fs::read(&path).unwrap());
    let range = engine.context.allocations.require(&allocation).unwrap();
    assert!(
        free.iter()
            .all(|&(start, end)| end <= range.start().get() || range.end().get() <= start)
    );
}

#[derive(Clone, Copy)]
enum Completion {
    Commit,
    Finalize,
    Sync,
}

impl Completion {
    fn apply(self, engine: &mut WriteEngine) -> Result<(), Error> {
        match self {
            Self::Commit => engine.commit(),
            Self::Finalize => engine.finalize_persist(),
            Self::Sync => engine.force_sync(),
        }
    }
}

#[rstest]
#[case::header_checksum(ManagerDamage::HeaderChecksum)]
#[case::sections_checksum(ManagerDamage::SectionsChecksum)]
#[case::free_header(ManagerDamage::FreeHeader)]
#[case::free_section_capacity(ManagerDamage::FreeSectionCapacity)]
#[case::free_extension(ManagerDamage::FreeExtension)]
fn malformed_authoritative_managers_reject_allocation_before_mutation(
    #[case] damage: ManagerDamage,
    #[values(PersistentStrategy::Flat, PersistentStrategy::Page)] strategy: PersistentStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("malformed.h5");
    let hole = fixture(&path, strategy);
    let mut bytes = fs::read(&path).unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let context = FileContext::new(crate::superblock::read(bytes.as_slice()).unwrap()).unwrap();
    let report = file.persisted_free_sections().unwrap().unwrap();
    let (manager, list_address) = report
        .managers()
        .iter()
        .filter(|manager| {
            matches!(strategy, PersistentStrategy::Flat)
                || manager.role() == FreeSpaceManagerRole::PageLargeGeneric
        })
        .find_map(|manager| {
            manager
                .free_sections()
                .sections()
                .iter()
                .find_map(|section| {
                    let start = section.address().max(hole.0);
                    let end = section.end_address().min(hole.1);
                    (start + 128 <= end).then_some((manager, start))
                })
        })
        .unwrap();
    let address = manager.address();
    let at = usize::try_from(address.get()).unwrap();
    let header = FreeSpaceManagerHeader::parse(context.widths(), &bytes[at..]).unwrap();
    let header_length =
        usize::try_from(format::free_space_manager_header_len(context.widths())).unwrap();
    let expected = match damage {
        ManagerDamage::HeaderChecksum | ManagerDamage::SectionsChecksum => {
            let end = match damage {
                ManagerDamage::SectionsChecksum => {
                    usize::try_from(header.section_list_addr().get() + header.section_list_used())
                        .unwrap()
                }
                _ => at + header_length,
            };
            let checksum_length = core::mem::size_of::<u32>();
            let original =
                u32::from_le_bytes(bytes[end - checksum_length..end].try_into().unwrap());
            bytes[end - 1] ^= 1;
            let altered = u32::from_le_bytes(bytes[end - checksum_length..end].try_into().unwrap());
            FormatError::ChecksumMismatch {
                expected: altered,
                computed: original,
            }
        }
        ManagerDamage::FreeExtension
        | ManagerDamage::FreeHeader
        | ManagerDamage::FreeSectionCapacity => {
            let (extension, _) = WriteEngine::extension_fsinfo(&BytesSource::new(&bytes), &context)
                .unwrap()
                .unwrap();
            let extension = object_header::read_oh_chunks(
                &BytesSource::new(&bytes),
                extension,
                context.addresses(),
            )
            .unwrap()
            .remove(0)
            .span;
            let capacity = 128;
            let (free, allocation) = match damage {
                ManagerDamage::FreeHeader => (address.get(), (address.get(), header_length as u64)),
                ManagerDamage::FreeSectionCapacity => {
                    (list_address + capacity - 16, (list_address, capacity))
                }
                ManagerDamage::FreeExtension => (extension.0, extension),
                _ => panic!("checksum damage has no overlapping allocation"),
            };
            let (mut encoded_header, list) = format::serialize_free_space_manager(
                context.widths(),
                &[format::FreeSection {
                    addr: StoredAddress::new(free),
                    size: 16,
                }],
                address,
                StoredAddress::new(list_address),
                match strategy {
                    PersistentStrategy::Flat => format::SECTION_CLASS_SIMPLE,
                    PersistentStrategy::Page => format::SECTION_CLASS_LARGE,
                },
            )
            .unwrap();
            assert!(u64::try_from(list.len()).unwrap() < capacity - 16);
            assert!(list_address + capacity <= hole.1);
            let length_width = usize::from(context.widths().lengths.get());
            let allocated_at = encoded_header.len() - core::mem::size_of::<u32>() - length_width;
            encoded_header[allocated_at..allocated_at + length_width]
                .copy_from_slice(&capacity.to_le_bytes()[..length_width]);
            let encoded_length = encoded_header.len();
            crate::checksum::stamp_trailing(&mut encoded_header, 0, encoded_length);
            bytes[at..at + encoded_header.len()].copy_from_slice(&encoded_header);
            let list_at = usize::try_from(list_address).unwrap();
            bytes[list_at..list_at + list.len()].copy_from_slice(&list);
            let observed = observe(&bytes);
            assert!(observed.free.contains(&(free, free + 16)));
            assert!(
                observed
                    .metadata
                    .contains(&(allocation.0, allocation.0 + allocation.1))
            );
            let free = FileRange::new(PhysicalAddress::new(free), 16).unwrap();
            let allocation =
                FileRange::new(PhysicalAddress::new(allocation.0), allocation.1).unwrap();
            FormatError::InvalidFreeSpaceManager(format!(
                "the free section at {}, {} bytes long, overlaps published manager or extension storage at {}, {} bytes long",
                free.start().get(),
                free.length(),
                allocation.start().get(),
                allocation.length()
            ))
        }
    };
    drop(file);
    fs::write(&path, &bytes).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let mut store = engine.store();
    let error = match store.allocate(&[17; 16], AllocationClass::Raw) {
        Ok(_) => panic!("a malformed manager authorized an allocation"),
        Err(error) => error,
    };
    let Error::Format(actual) = error else {
        panic!("expected malformed manager rejection, got {error:?}");
    };
    assert_eq!(actual, expected);
    assert_eq!(fs::read(&path).unwrap(), bytes);
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[derive(Clone, Copy)]
enum ManagerDamage {
    FreeExtension,
    FreeHeader,
    FreeSectionCapacity,
    HeaderChecksum,
    SectionsChecksum,
}

#[derive(Clone, Copy)]
pub(super) enum PersistentStrategy {
    Flat,
    Page,
}

impl PersistentStrategy {
    fn settings(self) -> FileSpaceSettings {
        FileSpaceSettings::new(match self {
            Self::Flat => FileSpaceStrategy::FsmAggr,
            Self::Page => FileSpaceStrategy::Page,
        })
        .with_persistence(true)
        .with_threshold(1)
    }
}

pub(super) fn fixture(path: &Path, strategy: PersistentStrategy) -> (u64, u64) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(strategy.settings());
    builder
        .create_dataset("values")
        .with_i32_data(&[0, 1, 2, 3])
        .with_chunks(&[4])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder
        .create_dataset("scratch")
        .with_u8_data(&vec![7; 65_536]);
    builder.create_dataset("ceiling").with_u8_data(&[9; 8192]);
    builder.write(path).unwrap();
    let file = File::open(path).unwrap();
    let DataLayout::Contiguous {
        address: Some(address),
        size,
    } = file.dataset("scratch").unwrap().data_layout().unwrap()
    else {
        panic!("scratch storage must be contiguous");
    };
    let address = address.get();
    let DataLayout::Contiguous {
        address: Some(ceiling),
        ..
    } = file.dataset("ceiling").unwrap().data_layout().unwrap()
    else {
        panic!("the ceiling must have contiguous live storage");
    };
    let ceiling = ceiling.get();
    assert!(ceiling >= address + size);
    drop(file);
    let mut engine = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    drop(engine);
    let bytes = fs::read(path).unwrap();
    let OwnershipObservation { free, .. } = observe(&bytes);
    let hole = free
        .into_iter()
        .filter(|&(start, end)| start <= address && end >= address + size)
        .max_by_key(|&(start, end)| end - start)
        .unwrap();
    assert!(hole.1 <= ceiling && ceiling < bytes.len() as u64);
    assert!(size >= 65_536);
    hole
}

pub(super) fn open(path: &Path, backing: MemoryStrategy, policy: SyncPolicy) -> WriteEngine {
    let mut engine = WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine.set_sync_policy(policy);
    engine
}

struct OwnershipObservation {
    free: Vec<(u64, u64)>,
    metadata: Vec<(u64, u64)>,
}

fn observe(bytes: &[u8]) -> OwnershipObservation {
    let source = BytesSource::new(bytes);
    let context = FileContext::new(crate::superblock::read(bytes).unwrap()).unwrap();
    let (extension, info) = WriteEngine::extension_fsinfo(&source, &context)
        .unwrap()
        .unwrap();
    let eof = context
        .addresses()
        .relative(context.superblock().eof_address)
        .unwrap();
    let (report, metadata) = PersistedManagerSet::read(
        &source,
        context.widths(),
        context.addresses().effective_base(),
        eof.get(),
        &info,
    )
    .unwrap()
    .into_parts();
    let free = report
        .free_sections()
        .sections()
        .iter()
        .map(|section| (section.address(), section.end_address()))
        .collect();
    let mut metadata: Vec<_> = metadata
        .iter()
        .map(|range: &FileRange| (range.start().get(), range.end().get()))
        .collect();
    metadata.extend(
        object_header::read_oh_chunks(&source, extension, context.addresses())
            .unwrap()
            .into_iter()
            .map(|chunk| (chunk.span.0, chunk.span.0 + chunk.span.1)),
    );
    OwnershipObservation {
        free: normalized(free),
        metadata,
    }
}

fn normalized(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut result: Vec<(u64, u64)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = result.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            result.push((start, end));
        }
    }
    result
}

fn assert_conserved(
    hole: (u64, u64),
    payloads: &[(u64, u64)],
    metadata: &[(u64, u64)],
    free: &[(u64, u64)],
    cycle: usize,
    state: &str,
) {
    let owned: Vec<_> = payloads.iter().chain(metadata).copied().collect();
    for &(start, end) in &owned {
        assert!(
            free.iter().all(|&(a, b)| end <= a || b <= start),
            "cycle {cycle}, {state}: owned {start}..{end} intersects free {free:?}"
        );
    }
    let mut boundaries = vec![hole.0, hole.1];
    for &(start, end) in owned.iter().chain(free) {
        boundaries.extend([start.clamp(hole.0, hole.1), end.clamp(hole.0, hole.1)]);
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    for pair in boundaries.windows(2) {
        let start = pair[0];
        let end = pair[1];
        if !owned.iter().any(|&(a, b)| a <= start && end <= b) {
            assert!(
                free.iter().any(|&(a, b)| a <= start && end <= b),
                "cycle {cycle}, {state}: unused {start}..{end} is absent; H={hole:?}, A={owned:?}, F={free:?}"
            );
        }
    }
}
