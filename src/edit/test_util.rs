//! Editing fixtures and storage-ownership assertions for private tests.

use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::Extent as FreeExtent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use test_util::disk_log::DiskOp;

use crate::Datatype;
use crate::DatatypeByteOrder;
use crate::FileLocking;
use crate::FileSpaceStrategy;
use crate::FixedPointType;
use crate::FloatingPointType;
use crate::Layout;
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::MemoryStrategy;
use crate::edit::SpaceOwnership;
use crate::edit::WriteEngine;
use crate::edit::object_header;
use crate::edit::reclaim;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_writer::OFFSET_WIDTH;
use crate::fractal_heap_storage::FractalHeapAllocation;
use crate::free_space_manager;
use crate::image::WriteBuffering;
use crate::link_info::LinkInfoMessage;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPath;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;
use crate::source::MetadataCacheConfig;
use crate::source::PhysicalBytes;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::superblock;
use crate::type_builders::DatasetBuilder;

impl SpaceOwnership {
    /// Asserts complete classification of every byte in the file.
    ///
    /// # Panics
    ///
    /// Panics if the analysis is incomplete, any byte is unclassified, trailing slack is present,
    /// the physical and logical file lengths differ, or the extents do not partition the file
    /// into nonempty ranges.
    pub(crate) fn assert_exact(&self) {
        self.assert_classified(&[]);
    }

    /// Asserts that the ownership categories partition the file with the expected unclassified
    /// extents.
    ///
    /// # Panics
    ///
    /// Panics if the analysis is incomplete, the unclassified extents differ, trailing slack is
    /// present, the physical and logical file lengths differ, or the extents do not partition the
    /// file into nonempty ranges.
    pub(crate) fn assert_classified(&self, expected_unclassified: &[(u64, u64)]) {
        let Self {
            physical_size,
            logical_size,
            userblock_space,
            live_raw_space,
            live_metadata_space,
            free_space_manager_space,
            reusable_free_space,
            unclassified_space,
            analysis_complete,
            trailing_slack_space,
        } = self;
        assert!(*analysis_complete, "ownership analysis must be complete");
        assert_eq!(unclassified_space, expected_unclassified);
        assert_eq!(trailing_slack_space, &[]);
        assert_eq!(physical_size, logical_size);
        let mut extents = userblock_space
            .iter()
            .chain(live_raw_space)
            .chain(live_metadata_space)
            .chain(free_space_manager_space)
            .chain(reusable_free_space)
            .chain(unclassified_space)
            .chain(trailing_slack_space)
            .copied()
            .collect::<Vec<_>>();
        extents.sort_unstable();
        let mut end = 0;
        for (start, len) in extents {
            assert!(len > 0, "empty ownership extent at {start}");
            assert_eq!(start, end, "ownership partition has a gap or overlap");
            end = start.checked_add(len).unwrap();
            assert!(
                end <= *physical_size,
                "ownership extent exceeds physical EOF"
            );
        }
        assert_eq!(
            end, *physical_size,
            "ownership partition must cover the file"
        );
    }
}

/// Opens a writable fixture with metadata caching disabled.
///
/// # Panics
///
/// Panics if the fixture cannot be opened for editing.
pub(super) fn open_engine(path: &Path) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Auto,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap()
}

/// Deletes `g` after checking its reclaim plan against its subtree's allocations.
///
/// Requires twelve contiguous child datasets and twelve huge link objects.
/// Reopens the committed file to check that `g` is absent.
///
/// # Panics
///
/// Panics if metadata cannot be read, storage accounting or the reclaim plan differs,
/// deletion or commit fails, or `g` remains accessible.
pub(in crate::edit) fn assert_huge_subtree_reclaimed(path: &Path) {
    let reader = crate::reader::File::open(path).unwrap();
    let link_names = reader.group("g").unwrap().datasets().unwrap();
    assert_eq!(link_names.len(), 12);
    let data_blocks: Vec<(FileRange, FreeClass)> = link_names
        .iter()
        .map(|link_name| {
            let crate::Layout::Contiguous {
                address: Some(data_addr),
                size,
            } = reader
                .dataset(&format!("g/{link_name}"))
                .unwrap()
                .layout()
                .unwrap()
            else {
                panic!("the child {link_name:?} stores its data contiguously");
            };
            (
                FileRange::new(PhysicalAddress::new(data_addr), size).unwrap(),
                FreeClass::Page(PageType::Raw),
            )
        })
        .collect();
    drop(reader);

    let object = ObjectPathBuf::parse("/g");
    let mut session = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    let resolve = |object_path: &ObjectPathBuf| {
        crate::group_v2::resolve_path_any_from_source(
            &session.opened_source(),
            AccessMode::ReadWrite,
            session.superblock(),
            session.widths(),
            &object_path.as_path(),
        )
        .unwrap()
    };
    let addr = resolve(&object);
    let header_spans = session.oh_chunk_spans(addr).unwrap();
    let header = ObjectHeader::parse_from_source_with_widths(
        &SourceMetadata(&session.image()),
        AccessMode::ReadWrite,
        addr,
        session.widths(),
        BaseAddress::ZERO,
    )
    .unwrap();
    let link_info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .map(|message| {
            LinkInfoMessage::parse_with_offset_width(&message.data, OFFSET_WIDTH).unwrap()
        })
        .expect("a dense group has a Link Info message");
    let heap_address = link_info
        .fractal_heap_address
        .expect("a dense group's Link Info message stores its heap");
    let link_storage = session
        .dense_group_storage_spans(&header_spans, None, &link_info)
        .expect("the dense link storage is structurally complete");
    session.assert_ownership_covers_storage(&link_storage, false);
    let heap = crate::fractal_heap_storage::collect_fractal_heap_storage_from_source(
        &session.image(),
        AddressContext::from_effective_base(BaseAddress::ZERO),
        heap_address,
        crate::file_writer::WIDTHS,
    )
    .unwrap();
    assert_eq!(
        heap.allocations()
            .iter()
            .filter(|(_, kind)| *kind == FractalHeapAllocation::HugeObject)
            .count(),
        12
    );
    for &(extent, kind) in heap.allocations() {
        let span = (
            FileRange::new(
                session.addresses().resolve(extent.start()).unwrap(),
                extent.len(),
            )
            .unwrap(),
            FreeClass::Page(kind.allocation_class().into()),
        );
        assert!(
            link_storage.contains(&span),
            "{span:?} is not in {link_storage:?}"
        );
    }

    let mut expected: Vec<(FileRange, FreeClass)> = reclaim::meta_spans(header_spans)
        .chain(link_storage)
        .chain(data_blocks)
        .chain(link_names.iter().flat_map(|link_name| {
            let child = resolve(&ObjectPathBuf::parse(&format!("/g/{link_name}")));
            reclaim::meta_spans(session.oh_chunk_spans(child).unwrap())
        }))
        .collect();
    expected.sort_unstable_by_key(|&(range, _)| range.start());
    let incoming = session.count_incoming_hard_links().unwrap();
    let mut planned = Vec::new();
    session.collect_free_spans(addr, 0, &incoming, &mut planned);
    planned.sort_unstable_by_key(|&(range, _)| range.start());
    assert_eq!(planned, expected);

    session.delete(&object).unwrap();
    session.commit().unwrap();
    drop(session);
    let err = crate::reader::File::open(path)
        .unwrap()
        .group("g")
        .unwrap_err();
    let Error::Format(FormatError::PathNotFound(missing)) = &err else {
        panic!("expected PathNotFound for g, got {err:?}");
    };
    assert_eq!(missing, "g");
}

impl WriteEngine {
    /// Reads the complete file through the editing session's retained disk handle.
    ///
    /// Pending writes remain buffered, and the session lock remains held.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if reading the length or bytes fails, or [`Error::Format`] if the
    /// length exceeds [`usize::MAX`].
    pub(crate) fn disk_bytes(&self) -> Result<Vec<u8>, Error> {
        self.image.disk_bytes()
    }

    /// Asserts complete ownership analysis and coverage of every selected allocation.
    ///
    /// With `metadata`, coverage must come entirely from live metadata. Otherwise, both live
    /// metadata and raw storage count toward coverage.
    ///
    /// # Panics
    ///
    /// Panics if analysis fails, is incomplete, or the required storage classes do not cover
    /// every selected byte.
    pub(in crate::edit) fn assert_ownership_covers_storage(
        &self,
        storage: &[(FileRange, FreeClass)],
        metadata: bool,
    ) {
        let owned = self.space_ownership().unwrap();
        assert!(owned.analysis_complete);
        for &(range, _) in storage {
            let at = range.start().get();
            let len = range.length();
            let covered =
                test_util::range::covered_len(&owned.live_metadata_space, &(at..at + len));
            let raw = if metadata {
                0
            } else {
                test_util::range::covered_len(&owned.live_raw_space, &(at..at + len))
            };
            assert_eq!(
                covered + raw,
                len,
                "heap storage {at}..{} must be classified",
                at + len
            );
        }
    }
}

/// Checks rollback of writes through two aliases of one data block.
///
/// Uses an invalid extension address to provoke [`Error::EditUnsupported`].
/// Both aliases must retain their original `[1, 2, 3]` values after reopen.
/// With unbuffered writes, also checks disk-write order and newest-first undo.
///
/// # Panics
///
/// Panics if setup or reading fails, or rollback observations differ.
pub(in crate::edit) fn assert_hard_link_undo(path: &Path, buffering: WriteBuffering) {
    let opened = File::open(path).unwrap();
    let Layout::Contiguous {
        address: Some(data_address),
        size: 12,
    } = opened.dataset("aa").unwrap().layout().unwrap()
    else {
        panic!("expected allocated three-element storage")
    };
    drop(opened);
    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_write_buffering(buffering).unwrap();
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    for (name, data) in [("/aa", [9, 9, 9]), ("/bb", [8, 8, 8])] {
        s.stage_dataset_write(&ObjectPathBuf::parse(name), {
            let mut db = crate::type_builders::DatasetBuilder::new("");
            db.with_i32_data(&data);
            db
        })
        .unwrap();
    }
    s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[42]);
        db
    })
    .unwrap();
    test_util::disk_log::start();
    let error = s.commit().unwrap_err();
    let operations = test_util::disk_log::take();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected unsupported extension header, got {error:?}");
    };
    assert_eq!(reason, "an object does not use a version 2 object header");
    if buffering == WriteBuffering::Unbuffered {
        let writes = operations
            .into_iter()
            .filter_map(|operation| match operation {
                DiskOp::Write { offset, bytes } if offset == data_address => Some(bytes),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            writes,
            [
                [9i32; 3]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>(),
                [8i32; 3]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>(),
                [9i32; 3]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>(),
                [1i32, 2, 3]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>()
            ]
        );
    }
    drop(s);

    let f = crate::reader::File::open(path).unwrap();
    assert_eq!(
        f.dataset("aa").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the block both links share must hold the value it held before"
    );
    assert_eq!(f.dataset("bb").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}

/// Returns persisted sub-page fragments after checking that allocations cannot reuse them.
///
/// Requires 512-byte pages and a populated generic-large manager. Checks that
/// each fragment is unclassified and absent from reusable sections under `policy`.
///
/// # Panics
///
/// Panics if metadata cannot be read, the layout premise differs, or a fragment
/// is classified or reusable.
pub(in crate::edit) fn assert_generic_large_sections(
    path: &Path,
    policy: PersistedFreeSpacePolicy,
) -> Vec<(u64, u64)> {
    const PAGE: u64 = 512;
    let opened = crate::reader::File::open(path).unwrap();
    let info = opened
        .file_space_info()
        .unwrap()
        .expect("a persisting file");
    drop(opened);
    assert_eq!(info.page_size.get(), PAGE);
    let bytes = std::fs::read(path).unwrap();
    let src = crate::source::BytesSource::new(bytes.as_slice());
    let slot6 = info.manager_addrs[6];
    assert_ne!(
        slot6,
        DecodedStoredAddress::Undefined,
        "the large manager is populated"
    );
    let (sections, _) = free_space_manager::read_persisted_sections_source(
        &src,
        crate::file_writer::WIDTHS,
        BaseAddress::ZERO,
        &[slot6],
    )
    .unwrap();
    let fragments: Vec<(u64, u64)> = sections
        .iter()
        .map(|s| (s.addr.get(), s.size))
        .filter(|&(addr, size)| addr % PAGE != 0 || size % PAGE != 0)
        .collect();
    assert!(
        !fragments.is_empty(),
        "the generic-large manager contains sub-page fragments: {sections:?}"
    );
    let s = WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        policy,
    )
    .unwrap();
    let unclassified = s
        .space
        .paged_sections()
        .expect("a paged file installs paged state")
        .unclassified()
        .to_vec();
    let reusable: Vec<_> = s.space.reusable_sections().collect();
    for &(addr, size) in &fragments {
        let fragment =
            FreeExtent::new(StoredAddress::new(addr), size).expect("wire sections are non-empty");
        assert!(
            unclassified.contains(&fragment),
            "fragment ({addr}, {size}) must be recorded as unclassified, not lost"
        );
        assert!(
            !reusable.iter().any(|extent| {
                extent.start() < fragment.end() && fragment.start() < extent.end()
            }),
            "fragment ({addr}, {size}) must not be offered to any allocation: {reusable:?}"
        );
    }
    fragments
}

/// Deletes `victim` and checks that replacement raw data avoids its metadata pages.
///
/// Creates `added` with eight [`f64`] values equal to 2.5 and checks the completed
/// hierarchy, dataset metadata, and persisted page policy. `keep` must contain
/// `[1, 2, 3, 4]` as [`i32`] values in `order`.
///
/// # Panics
///
/// Panics if editing or reading fails, replacement raw data occupies a metadata
/// page, or any observation differs from the expected edited fixture.
pub(in crate::edit) fn assert_index_reclamation(path: &Path, order: DatatypeByteOrder) {
    const PAGE: u64 = 4096;
    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    let page_size = s.space.page_size().expect("a paged file has a page size");
    assert_eq!(page_size.get(), PAGE);
    let victim_addr = crate::group_v2::resolve_path_any(
        PhysicalBytes::for_test(
            s.image.as_slice().unwrap(),
            AddressContext::from_effective_base(s.context.superblock().base_address),
        ),
        AccessMode::ReadWrite,
        s.context.superblock(),
        s.context.widths(),
        &ObjectPath::parse("victim"),
    )
    .unwrap();
    let index_spans = s
        .chunked_index_spans(victim_addr)
        .expect("the extensible-array index is enumerable");
    assert!(!index_spans.is_empty());
    let mut meta_pages: Vec<u64> = vec![s.context.superblock().base_address.get() / PAGE];
    for range in &index_spans {
        for p in (range.start().get() / PAGE)..=((range.end().get() - 1) / PAGE) {
            meta_pages.push(p);
        }
    }
    meta_pages.sort_unstable();
    meta_pages.dedup();
    s.delete(&ObjectPathBuf::parse("/victim")).unwrap();
    s.commit().unwrap();
    let mut db = DatasetBuilder::new("added");
    db.with_f64_data(&[2.5f64; 8]).with_shape(&[8]);
    s.stage_created_dataset(&ObjectPathBuf::parse("/added"), db)
        .unwrap();
    s.commit().unwrap();
    drop(s);
    let f = crate::reader::File::open(path).unwrap();
    let added = f.dataset("added").unwrap();
    let crate::Layout::Contiguous {
        address: Some(addr),
        size,
    } = added.layout().unwrap()
    else {
        panic!("a small f64 dataset is stored contiguously");
    };
    for p in (addr / PAGE)..=((addr + size - 1) / PAGE) {
        assert!(
            !meta_pages.contains(&p),
            "the new dataset's raw data landed at ({addr}, {size}), in page {p}, \
             which held live metadata before the edit: {meta_pages:?}"
        );
    }

    f.assert_paged_index_edit(order);
}

/// Checks byte-for-byte preservation when rewriting a superblock extension's free-space record.
///
/// # Panics
///
/// Panics if the extension or record cannot be read or rewritten, or the rewritten
/// bytes or header properties differ.
pub(in crate::edit) fn assert_extension_records(source: &impl Source) {
    let located = superblock::read(source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let extension = addresses
        .resolve(located.superblock.superblock_extension_address.unwrap())
        .unwrap()
        .get();
    let region = WriteEngine::gather_oh_messages(source, extension, addresses).unwrap();
    let splice = region
        .splice_first(MessageType::FILE_SPACE_INFO)
        .unwrap()
        .unwrap();
    let info =
        hdf5_pure_format::__private::parse_file_space_info(located.widths, splice.record().body)
            .unwrap();

    let rewritten =
        object_header::rewrite_extension_region_bytes(&region, located.widths, &info).unwrap();

    assert_eq!(rewritten.props(), region.props());
    assert_eq!(&rewritten[..], &region[..]);
}

impl File {
    fn assert_paged_index_edit(&self, order: DatatypeByteOrder) {
        let root = self.root();
        let mut names = root.datasets().unwrap();
        names.sort();
        assert_eq!(names, ["added", "keep"]);
        assert!(root.groups().unwrap().is_empty());
        assert!(root.named_datatypes().unwrap().is_empty());
        assert!(root.attrs().unwrap().is_empty());
        let keep = self.dataset("keep").unwrap();
        assert_eq!(keep.shape().unwrap(), [4]);
        assert_eq!(
            keep.datatype().unwrap(),
            Datatype::FixedPoint(FixedPointType::new(4, order, true, 0, 32).unwrap())
        );
        assert_eq!(keep.read_i32().unwrap(), [1, 2, 3, 4]);
        let added = self.dataset("added").unwrap();
        assert_eq!(added.shape().unwrap(), [8]);
        assert_eq!(
            added.datatype().unwrap(),
            Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
                DatatypeByteOrder::LittleEndian
            ))
        );
        assert_eq!(added.read_f64().unwrap(), [2.5; 8]);
        for (dataset, expected_size) in [(&keep, 16), (&added, 64)] {
            assert!(dataset.attrs().unwrap().is_empty());
            assert!(dataset.filter_pipeline().unwrap().is_empty());
            assert_eq!(dataset.chunk_shape().unwrap(), None);
            let Layout::Contiguous {
                address: Some(_),
                size,
            } = dataset.layout().unwrap()
            else {
                panic!("expected contiguous storage")
            };
            assert_eq!(size, expected_size);
        }
        let info = self.file_space_info().unwrap().unwrap();
        assert_eq!(info.strategy, FileSpaceStrategy::Page);
        assert!(info.persist);
        assert_eq!(info.threshold, 1);
        assert_eq!(info.page_size.get(), 4096);
    }
}
