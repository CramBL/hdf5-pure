//! Editing fixtures and storage-ownership assertions for private tests.

use std::path::Path;

use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;

use crate::FileLocking;
use crate::PersistedFreeSpacePolicy;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::edit::MemoryStrategy;
use crate::edit::SpaceOwnership;
use crate::edit::WriteEngine;
use crate::edit::reclaim;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_writer::OFFSET_WIDTH;
use crate::fractal_heap_storage::FractalHeapAllocation;
use crate::link_info::LinkInfoMessage;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::source::SourceMetadata;

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
    let data_blocks: Vec<(u64, u64, FreeClass)> = link_names
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
            (data_addr, size, FreeClass::Page(PageType::Raw))
        })
        .collect();
    drop(reader);

    let object = ObjectPathBuf::parse("/g");
    let mut session = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    let resolve = |object_path: &ObjectPathBuf| {
        crate::group_v2::resolve_path_any_from_source(
            &session.image(),
            AccessMode::ReadWrite,
            session.superblock(),
            session.widths(),
            session.addresses(),
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
            extent.start(),
            extent.len(),
            FreeClass::Page(kind.allocation_class().into()),
        );
        assert!(
            link_storage.contains(&span),
            "{span:?} is not in {link_storage:?}"
        );
    }

    let mut expected: Vec<(u64, u64, FreeClass)> = reclaim::meta_spans(header_spans)
        .chain(link_storage)
        .chain(data_blocks)
        .chain(link_names.iter().flat_map(|link_name| {
            let child = resolve(&ObjectPathBuf::parse(&format!("/g/{link_name}")));
            reclaim::meta_spans(session.oh_chunk_spans(child).unwrap())
        }))
        .collect();
    expected.sort_unstable_by_key(|&(span_addr, _, _)| span_addr);
    let incoming = session.count_incoming_hard_links().unwrap();
    let mut planned = Vec::new();
    session.collect_free_spans(addr, 0, &incoming, &mut planned);
    planned.sort_unstable_by_key(|&(span_addr, _, _)| span_addr);
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
        storage: &[(u64, u64, FreeClass)],
        metadata: bool,
    ) {
        let owned = self.space_ownership().unwrap();
        assert!(owned.analysis_complete);
        for &(at, len, _) in storage {
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
