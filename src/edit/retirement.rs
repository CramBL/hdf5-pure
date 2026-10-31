//! Chunk allocation ownership for immediate and staged replacements.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{Extent, FreeClass, PageType};

use crate::address::AddressContext;
use crate::chunked_read::ChunkInfo;
use crate::error::FormatError;

/// A complete, nonempty chunk allocation captured from its index metadata.
///
/// The editor retains its physical range, stored-address extent, and raw-page class through the
/// replacement's publication. Reuse requires separate publication evidence.
#[derive(Debug)]
pub(crate) struct ChunkAllocation {
    range: FileRange,
    extent: Extent,
    class: FreeClass,
}

impl ChunkAllocation {
    /// Resolves the chunk record's address and retains its complete stored allocation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the physical address or allocation end exceeds
    /// [`u64::MAX`], or [`FormatError::InvalidChunkGeometry`] if the allocation is empty.
    pub(crate) fn capture(
        addresses: AddressContext,
        chunk: &ChunkInfo,
    ) -> Result<Self, FormatError> {
        let range = FileRange::new(addresses.resolve(chunk.address)?, chunk.chunk_size.get())?;
        let extent = Extent::new(chunk.address, range.length()).ok_or(
            FormatError::InvalidChunkGeometry("chunk allocation is empty"),
        )?;
        Ok(Self {
            range,
            extent,
            class: FreeClass::Page(PageType::Raw),
        })
    }

    /// Returns the complete physical range for ownership comparisons.
    pub(crate) fn range(&self) -> FileRange {
        self.range
    }

    /// Consumes allocation metadata into its stored-address extent and free-space class.
    pub(crate) fn into_extent(self) -> (Extent, FreeClass) {
        let Self {
            range: _,
            extent,
            class,
        } = self;
        (extent, class)
    }
}

/// Superseded chunk allocations associated with a replacement root.
///
/// The publisher consumes these allocations into pending free spans before root publication.
pub(super) struct RootRetirement {
    root: PhysicalAddress,
    chunks: Vec<ChunkAllocation>,
}

impl RootRetirement {
    /// Associates captured predecessor allocations with the replacement root.
    pub(super) fn new(root: PhysicalAddress, chunks: Vec<ChunkAllocation>) -> Self {
        Self { root, chunks }
    }

    /// Consumes predecessor ownership into the associated root's pending free spans.
    ///
    /// Combines the chunk allocations with `spans`, excluding empty or overlapping ranges and
    /// ranges beyond `eof`. The publisher must preserve the prior ownership until root publication
    /// succeeds.
    pub(super) fn into_pending_frees(
        self,
        mut spans: Vec<(FileRange, FreeClass)>,
        eof: u64,
    ) -> PendingRootFrees {
        let Self { root, chunks } = self;
        spans.extend(chunks.into_iter().map(
            |ChunkAllocation {
                 range,
                 extent: _,
                 class,
             }| { (range, class) },
        ));
        super::retain_disjoint_in_bounds(&mut spans, eof);
        PendingRootFrees { root, spans }
    }
}

/// A replacement root and the physical spans awaiting its publication.
pub(super) struct PendingRootFrees {
    pub(super) root: PhysicalAddress,
    pub(super) spans: Vec<(FileRange, FreeClass)>,
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;
    use std::path::Path;

    use hdf5_pure_core::PhysicalAddress;
    use hdf5_pure_space::__private::{FreeClass, PageType};
    use rstest::rstest;
    use test_util::disk_log;

    use super::{ChunkAllocation, RootRetirement};
    use crate::access_mode::AccessMode;
    use crate::address::{
        AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
    };
    use crate::chunked_read::{ChunkInfo, StoredChunkSize};
    use crate::edit::dataset::MovingWrite;
    use crate::edit::test_util as edit_test_util;
    use crate::edit::{AppendTarget, WriteEngine};
    use crate::error::{Error, FormatError};
    use crate::group_v2;
    use crate::image::WriteBuffering;
    use crate::object_path::{ObjectPath, ObjectPathBuf};
    use crate::superblock;
    use crate::{
        AppendBuilder, DeflateLevel, File, FileBuilder, FileSpaceSettings, FileSpaceStrategy,
        MaxExtent,
    };

    #[rstest]
    #[case::unfiltered(512, 4096, 0)]
    #[case::filtered(0, 31, 0)]
    #[case::exempt_filtered_record(512, 4096, u32::MAX)]
    #[case::large(0, u64::from(u32::MAX) + 1, 0)]
    fn root_retirement_preserves_the_complete_raw_allocation(
        #[case] base: u64,
        #[case] stored_size: u64,
        #[case] filter_mask: u32,
    ) {
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
        let chunk = ChunkInfo {
            address: StoredAddress::new(64),
            chunk_size: StoredChunkSize::v4(stored_size),
            filter_mask,
            offsets: vec![0],
        };
        let allocation = ChunkAllocation::capture(addresses, &chunk).unwrap();
        assert_eq!(allocation.range().start().get(), base + 64);
        assert_eq!(allocation.range().length(), stored_size);
        let root = PhysicalAddress::new(base + 8);
        let range = allocation.range();
        let retirement = RootRetirement::new(root, vec![allocation]);
        let pending = retirement.into_pending_frees(Vec::new(), base + 64 + stored_size);
        assert_eq!(pending.root, root);
        assert_eq!(pending.spans, vec![(range, FreeClass::Page(PageType::Raw))]);
    }

    #[rstest]
    #[case::empty(64, 0, FormatError::InvalidChunkGeometry("chunk allocation is empty"))]
    #[case::overflow(u64::MAX - 7, 8, FormatError::OffsetOverflow { offset: u64::MAX - 7, length: 8 })]
    fn chunk_allocation_rejects_invalid_geometry(
        #[case] address: u64,
        #[case] size: u64,
        #[case] expected: FormatError,
    ) {
        let chunk = ChunkInfo {
            address: StoredAddress::new(address),
            chunk_size: StoredChunkSize::v4(size),
            filter_mask: 0,
            offsets: vec![0],
        };
        let addresses = AddressContext::from_effective_base(BaseAddress::new(0));
        assert_eq!(
            ChunkAllocation::capture(addresses, &chunk).unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::unfiltered(false)]
    #[case::deflate(true)]
    fn staged_tail_retirement_preserves_storage_accounting(#[case] filtered: bool) {
        let path = test_util::temp::temp_path("staged-retirement.h5");
        create_partial_tail(&path, filtered, true);
        let mut engine = edit_test_util::open_engine(&path);
        let mut append = AppendBuilder::new();
        append.append_i32(&[3]);
        engine
            .stage_dataset_append_pending(&ObjectPathBuf::parse("d"), append)
            .unwrap();
        engine.commit().unwrap();
        engine.space_ownership().unwrap().assert_exact();
        engine.finalize_persist().unwrap();
        drop(engine);
        let engine = edit_test_util::open_engine(&path);
        engine.space_ownership().unwrap().assert_exact();
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            vec![0, 1, 2, 3]
        );
    }

    #[rstest]
    #[case::payload(false)]
    #[case::root(true)]
    fn failed_staged_tail_publication_keeps_the_predecessor_unavailable(
        #[case] root: bool,
        #[values(false, true)] persistent: bool,
    ) {
        let path = test_util::temp::temp_path("failed-retirement.h5");
        create_partial_tail(&path, false, persistent);
        let mut engine = if root {
            WriteEngine::open_torn_writes(&path, 0..48).unwrap()
        } else {
            edit_test_util::open_engine(&path)
        };
        engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
        let dataset_path = ObjectPathBuf::parse("d");
        let address = group_v2::resolve_path_any_from_source(
            &engine.image(),
            AccessMode::ReadWrite,
            engine.superblock(),
            engine.widths(),
            engine.addresses(),
            &ObjectPath::parse("d"),
        )
        .unwrap();
        let mut append = AppendBuilder::new();
        append.append_i32(&[3]);
        let write = WriteEngine::prepare_append(
            &engine.image(),
            address,
            &append,
            engine.addresses(),
            engine.widths(),
        )
        .unwrap();
        let MovingWrite::AppendedChunks {
            old_tail_allocation: Some(allocation),
            ..
        } = write
        else {
            panic!("expected a partial tail replacement");
        };
        let predecessor = allocation.range();
        engine
            .stage_dataset_append_pending(&dataset_path, append)
            .unwrap();
        if !root {
            disk_log::fail_write_after(0, 0);
        }
        let error = engine.commit().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected the injected write error, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(
            error.to_string(),
            if root {
                "simulated device write failure"
            } else {
                "injected disk write failure"
            }
        );
        assert_eq!(engine.publication.attempted(), root);
        let published = superblock::read(&engine.image()).unwrap();
        let published_dataset = group_v2::resolve_path_any_from_source(
            &engine.image(),
            AccessMode::ReadWrite,
            &published.superblock,
            published.widths,
            engine.addresses(),
            &ObjectPath::parse("d"),
        )
        .unwrap();
        let mut protected = vec![predecessor];
        protected.extend(
            engine
                .chunked_storage_spans(published_dataset)
                .unwrap()
                .into_iter()
                .map(|(range, _)| range),
        );
        protected.extend(engine.oh_chunk_spans(published_dataset).unwrap());
        protected.extend(
            engine
                .oh_chunk_spans(
                    engine
                        .addresses()
                        .resolve(published.superblock.root_group_address)
                        .unwrap()
                        .get(),
                )
                .unwrap(),
        );
        for range in engine.reusable_ranges() {
            let range = range.unwrap();
            for &protected in &protected {
                assert!(!range.overlaps(protected));
            }
        }
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            if root {
                vec![0, 1, 2, 3]
            } else {
                vec![0, 1, 2]
            }
        );
    }

    #[rstest]
    #[case::flat(FileSpaceStrategy::FsmAggr)]
    #[case::page(FileSpaceStrategy::Page)]
    fn reused_tail_replacements_persist_ownership_without_growing_eof(
        #[case] strategy: FileSpaceStrategy,
    ) {
        let path = test_util::temp::temp_path("reused.h5");
        create_reuse_source(&path, strategy);
        let mut engine = edit_test_util::open_engine(&path);
        engine.delete(&ObjectPathBuf::parse("spare")).unwrap();
        engine.commit().unwrap();
        append_one(&mut engine, 3);
        let eof = engine.image.len();
        append_one(&mut engine, 4);
        assert_eq!(engine.image.len(), eof);
        assert!(engine.fsm_dirty);
        engine.space_ownership().unwrap().assert_exact();
        engine.commit().unwrap();
        assert!(!engine.fsm_dirty);
        engine.finalize_persist().unwrap();
        drop(engine);
        let engine = edit_test_util::open_engine(&path);
        engine.space_ownership().unwrap().assert_exact();
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            [0, 1, 2, 3, 4]
        );
    }

    #[rstest]
    #[case::first_write(0, 0)]
    #[case::second_write(1, 0)]
    #[case::torn_second_write(1, 17)]
    fn failed_manager_cleanup_retains_confirmed_retirements(
        #[case] completed: usize,
        #[case] prefix: usize,
    ) {
        let path = test_util::temp::temp_path("cleanup.h5");
        create_reuse_source(&path, FileSpaceStrategy::FsmAggr);
        let file = File::open(&path).unwrap();
        let predecessor = file.dataset("d").unwrap().chunks().unwrap().remove(0);
        drop(file);
        let mut engine = edit_test_util::open_engine(&path);
        engine.delete(&ObjectPathBuf::parse("spare")).unwrap();
        engine.commit().unwrap();
        append_one(&mut engine, 3);
        let before = engine.reusable_free_sections().unwrap();
        assert!(engine.fsm_dirty);
        disk_log::fail_write_after(completed, prefix);
        let error = engine.finalize_persist().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected the injected write failure, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert!(engine.fsm_dirty);
        let retained = engine.reusable_free_sections().unwrap();
        assert!(
            retained
                .sections()
                .iter()
                .any(|section| section.address().get() <= predecessor.address
                    && predecessor.address + predecessor.storage_size
                        <= section.end_address().get())
        );
        if completed == 0 {
            assert_eq!(retained, before);
        }
        if completed == 0 {
            engine.finalize_persist().unwrap();
        } else {
            let issued = engine.disk_bytes().unwrap();
            assert!(matches!(
                engine.finalize_persist(),
                Err(Error::AllocationPublicationUncertain)
            ));
            assert_eq!(engine.disk_bytes().unwrap(), issued);
        }
        assert_eq!(engine.fsm_dirty, completed != 0);
        if completed == 0 {
            drop(engine);
            let engine = edit_test_util::open_engine(&path);
            engine.space_ownership().unwrap().assert_exact();
        } else {
            assert!(engine.publication.attempted());
            assert_eq!(engine.reusable_free_sections().unwrap(), retained);
            let ownership = engine.space_ownership().unwrap();
            assert!(ownership.analysis_complete);
            for &(start, len) in &ownership.reusable_free_space {
                for &(live, size) in ownership
                    .live_raw_space
                    .iter()
                    .chain(&ownership.live_metadata_space)
                {
                    assert!(start + len <= live || live + size <= start);
                }
            }
        }
    }

    #[test]
    fn nonpersistent_tail_retirement_is_reused_during_the_session() {
        let path = test_util::temp::temp_path("nonpersistent.h5");
        create_partial_tail(&path, false, false);
        let mut engine = edit_test_util::open_engine(&path);
        append_one(&mut engine, 3);
        let eof = engine.image.len();
        append_one(&mut engine, 4);
        assert_eq!(engine.image.len(), eof);
        assert!(!engine.fsm_dirty);
        assert!(engine.space.reusable_bytes().unwrap() >= 2048);
        engine.finalize_persist().unwrap();
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            [0, 1, 2, 3, 4]
        );
    }

    fn append_one(engine: &mut WriteEngine, value: i32) {
        let mut append = AppendBuilder::new();
        append.append_i32(&[value]);
        engine
            .append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &append, 4)
            .unwrap();
    }

    fn create_reuse_source(path: &Path, strategy: FileSpaceStrategy) {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
        builder
            .create_dataset("d")
            .with_i32_data(&[0, 1, 2])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[512]);
        builder
            .create_dataset("spare")
            .with_u8_data(&vec![0; 131_072]);
        builder.create_dataset("ceiling").with_i32_data(&[99]);
        builder.write(path).unwrap();
    }

    fn create_partial_tail(path: &Path, filtered: bool, persistent: bool) {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(persistent),
        );
        let dataset = builder.create_dataset("d");
        dataset
            .with_i32_data(&[0, 1, 2])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[512]);
        if filtered {
            dataset.with_deflate(DeflateLevel::Six);
        }
        builder.write(path).unwrap();
    }
}
