use std::io::Error as IoError;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::Path;

use rstest::rstest;

use crate::DeflateLevel;
use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::chunk_index_inplace::alloc_probe;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;
use crate::image::FileImage;
use crate::image::MirrorImage;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::source::Source;
use crate::type_builders::DatasetBuilder;

use super::AllocationClass;

#[rstest]
#[case::small(4, 4, 8)]
#[case::raw_refill(4, 4, 1200)]
#[case::index_blocks(4, 1, 1500)]
fn staged_and_immediate_edits_share_aggregate_windows(
    #[case] initial: i32,
    #[case] chunk: u64,
    #[case] end: i32,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("interleaved.h5");
    write_appendable(&path, initial, chunk, false, strategy);
    let mut engine = open_engine(&path, backing);
    let mut first = DatasetBuilder::new("");
    first.with_i32_data(&[41; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("first"), first)
        .unwrap();
    engine.commit().unwrap();
    let state = engine.aggregation.unwrap();
    let raw = state.policy.unused(AllocationClass::Raw).unwrap();
    let metadata = state.policy.unused(AllocationClass::Metadata).unwrap();
    let fsm_hit = engine.space.largest(AllocationClass::Raw) >= 16;
    alloc_probe::take();
    engine
        .append_inplace_i32_phased("values", &(initial..end).collect::<Vec<_>>(), 4)
        .unwrap();
    let (data_blocks, super_blocks) = alloc_probe::take();
    let after = engine.aggregation.unwrap();
    if end == 8 {
        assert_eq!(after.eoa, state.eoa);
        assert_eq!(
            after.policy.unused(AllocationClass::Metadata),
            Some(metadata)
        );
        if fsm_hit {
            assert_eq!(after.policy.unused(AllocationClass::Raw), Some(raw));
        } else {
            assert_eq!(
                after.policy.unused(AllocationClass::Raw).unwrap().start(),
                raw.start() + 16
            );
        }
    } else {
        assert!(after.eoa > state.eoa);
        assert!(data_blocks > 0);
        assert_ne!(
            after.policy.unused(AllocationClass::Metadata),
            Some(metadata)
        );
    }
    if chunk == 1 {
        assert!(super_blocks > 0);
    }
    let ownership = engine.space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    assert_eq!(ownership.logical_size, ownership.physical_size);
    assert_eq!(ownership.free_space_manager_space, []);
    for &(start, length) in &ownership.reusable_free_space {
        assert!(
            ownership
                .live_metadata_space
                .iter()
                .chain(&ownership.live_raw_space)
                .all(|&(address, size)| start + length <= address || address + size <= start)
        );
    }
    let mut last = DatasetBuilder::new("");
    last.with_i32_data(&[43; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("last"), last)
        .unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("first").unwrap().read_i32().unwrap(), [41; 16]);
    assert_eq!(file.dataset("last").unwrap().read_i32().unwrap(), [43; 16]);
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        (0..end).collect::<Vec<_>>()
    );
    assert_eq!(file.persisted_free_space().unwrap(), []);
}

#[rstest]
#[case::allocation(1)]
#[case::index(2)]
#[case::header(3)]
#[case::dimension(4)]
fn immediate_phase_stops_keep_reservations_consumed(
    #[case] phase: u8,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(false, true)] filtered: bool,
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("phase.h5");
    write_appendable(&path, 6, 4, filtered, strategy);
    let mut engine = open_engine(&path, backing);
    engine
        .append_inplace_i32_phased("values", &(6..29).collect::<Vec<_>>(), phase)
        .unwrap();
    let consumed = engine.aggregation.unwrap();
    assert!(consumed.policy.unused(AllocationClass::Raw).is_some());
    let end = if phase == 4 { 29 } else { 6 };
    engine
        .append_inplace_i32_phased("values", &[97, 98], 4)
        .unwrap();
    assert_ne!(engine.aggregation.unwrap(), consumed);
    assert_eq!(engine.superblock.eof_address, engine.image.len());
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    let mut expected = (0..end).collect::<Vec<_>>();
    expected.extend_from_slice(&[97, 98]);
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        expected
    );
}

#[rstest]
fn bounded_immediate_batches_share_aggregate_state(
    #[values(false, true)] filtered: bool,
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("batches.h5");
    write_appendable(&path, 6, 512, filtered, strategy);
    let mut engine = open_engine(&path, MemoryStrategy::Bounded);
    let end = 600_000;
    engine
        .append_inplace_i32_phased("values", &(6..end).collect::<Vec<_>>(), 4)
        .unwrap();
    engine
        .append_inplace_i32_phased("values", &[end, end + 1], 4)
        .unwrap();
    let ownership = engine.space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    assert_eq!(ownership.physical_size, ownership.logical_size);
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        (0..end + 2).collect::<Vec<_>>()
    );
}

#[rstest]
#[case::raw_allocation(FailureTarget::Raw)]
#[case::padding(FailureTarget::Padding)]
#[case::eof(FailureTarget::Eof)]
#[case::index_allocation(FailureTarget::IndexAllocation)]
#[case::index_eof(FailureTarget::IndexEof)]
#[case::index_pointer(FailureTarget::Index)]
#[case::header(FailureTarget::Header)]
#[case::dimension(FailureTarget::Dimension)]
fn failed_immediate_append_preserves_consumed_space(
    #[case] target: FailureTarget,
    #[values(
        FailureMode::WriteBefore,
        FailureMode::WriteAfter,
        FailureMode::BarrierBefore,
        FailureMode::BarrierAfter
    )]
    mode: FailureMode,
    #[values(Continuation::Drop, Continuation::Append, Continuation::Commit)] next: Continuation,
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    write_appendable(&path, 4, 1, false, strategy);
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let address = engine
        .append_prepare(AppendTarget::Path(&ObjectPathBuf::parse("values")))
        .unwrap();
    let state = &engine.located[&address].loc;
    let index = state.index_block_addr.get();
    let header = state.ea_addr.get() + HEADER_STATISTICS_OFFSET;
    let dimension = state.dim0_off;
    drop(engine);
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailureImage {
            inner: WriteEngine::read_mirror(handle)?,
            target,
            mode,
            armed: false,
            failed: false,
            eof: std::fs::metadata(&path).unwrap().len(),
            index: index..index + 200,
            header,
            dimension,
            eof_writes: 0,
        }))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let error = engine
        .append_inplace_i32_phased("values", &(4..12).collect::<Vec<_>>(), 4)
        .unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected immediate write failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert!(!engine.aggregation.unwrap().publication_confirmed);
    let published = File::from_bytes(engine.image.as_slice().unwrap().to_vec()).unwrap();
    let expected = published.dataset("values").unwrap().read_i32().unwrap();
    let new_dimension =
        matches!(target, FailureTarget::Dimension) && !matches!(mode, FailureMode::WriteBefore);
    assert_eq!(
        expected,
        (0..if new_dimension { 12 } else { 4 }).collect::<Vec<_>>()
    );
    let physical_length = engine.image.len();
    engine.finalize_persist().unwrap();
    assert_eq!(engine.image.len(), physical_length);
    if matches!(next, Continuation::Commit) {
        engine.create_group(&ObjectPathBuf::parse("after")).unwrap();
        engine.commit().unwrap();
    }
    if !matches!(next, Continuation::Drop) {
        engine
            .append_inplace_i32_phased("values", &[97, 98], 4)
            .unwrap();
        assert_eq!(engine.superblock.eof_address, engine.image.len());
        engine.finalize_persist().unwrap();
    }
    drop(engine);
    let file = File::open(&path).unwrap();
    let mut expected = expected;
    if !matches!(next, Continuation::Drop) {
        expected.extend_from_slice(&[97, 98]);
    }
    if matches!(next, Continuation::Commit) {
        assert_eq!(file.root().groups().unwrap(), ["after"]);
    }
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        expected
    );
    assert_eq!(file.persisted_free_space().unwrap(), []);
}

#[test]
fn failed_commit_restores_spare_returned_during_refill() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed_refill.h5");
    FileBuilder::new().write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailureImage {
            inner: WriteEngine::read_mirror(handle)?,
            target: FailureTarget::Raw,
            mode: FailureMode::WriteBefore,
            armed: false,
            failed: false,
            eof: start + 4096,
            index: 0..0,
            header: 0,
            dimension: 0,
            eof_writes: 0,
        }))
    })
    .unwrap();
    let reservation = engine.reserve(2000, AllocationClass::Metadata).unwrap();
    engine.place(reservation, &[19; 2000]).unwrap();
    let reservation = engine.reserve(64, AllocationClass::Raw).unwrap();
    engine.place(reservation, &[23; 64]).unwrap();
    let state = engine.aggregation;
    let free = engine.space_accounting().unwrap().reusable_free_space;
    engine.create_group(&ObjectPathBuf::parse("group")).unwrap();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected aggregate refill write failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert!(!engine.publish_attempted);
    assert_eq!(engine.aggregation, state);
    assert_eq!(engine.space_accounting().unwrap().reusable_free_space, free);
    assert_eq!(engine.space.reusable_sections().count(), 0);
    drop(engine);
    assert_eq!(
        File::open(&path).unwrap().root().groups().unwrap(),
        Vec::<String>::new()
    );
}

fn write_appendable(
    path: &Path,
    initial: i32,
    chunk: u64,
    filtered: bool,
    strategy: FileSpaceStrategy,
) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(strategy, false, 1);
    let dataset = builder.create_dataset("values");
    dataset
        .with_i32_data(&(0..initial).collect::<Vec<_>>())
        .with_shape(&[initial as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[chunk]);
    if filtered {
        dataset.with_deflate(DeflateLevel::Six);
    }
    builder.write(path).unwrap();
}

fn open_engine(path: &Path, backing: MemoryStrategy) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::default(),
        FileLocking::Enabled,
        backing,
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Continuation {
    Append,
    Commit,
    Drop,
}

#[derive(Clone, Copy, Debug)]
enum FailureTarget {
    Dimension,
    Eof,
    Header,
    Index,
    IndexAllocation,
    IndexEof,
    Padding,
    Raw,
}

#[derive(Clone, Copy, Debug)]
enum FailureMode {
    BarrierAfter,
    BarrierBefore,
    WriteAfter,
    WriteBefore,
}

struct FailureImage {
    inner: MirrorImage,
    target: FailureTarget,
    mode: FailureMode,
    armed: bool,
    failed: bool,
    eof: u64,
    index: Range<u64>,
    header: u64,
    dimension: u64,
    eof_writes: usize,
}

impl FailureImage {
    fn before_write(&mut self, offset: u64, bytes: &[u8]) -> Result<bool, Error> {
        if offset == 0 {
            self.eof_writes += 1;
        }
        let target = match self.target {
            FailureTarget::Raw => offset == self.eof,
            FailureTarget::Padding => offset > self.eof && bytes.iter().all(|&byte| byte == 0),
            FailureTarget::Eof => offset == 0,
            FailureTarget::IndexEof => offset == 0 && self.eof_writes == 2,
            FailureTarget::IndexAllocation => bytes.starts_with(DATA_BLOCK_SIGNATURE),
            FailureTarget::Index => self.index.contains(&offset),
            FailureTarget::Header => offset == self.header,
            FailureTarget::Dimension => offset == self.dimension,
        };
        if target && !self.failed {
            self.armed = true;
            if matches!(self.mode, FailureMode::WriteBefore) {
                self.failed = true;
                return Err(IoError::other("injected immediate write failure").into());
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn after_write(&mut self, target: bool) -> Result<(), Error> {
        if target && matches!(self.mode, FailureMode::WriteAfter) {
            self.failed = true;
            return Err(IoError::other("injected immediate write failure").into());
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<(), Error> {
        if self.armed && !self.failed && matches!(self.mode, FailureMode::BarrierBefore) {
            self.failed = true;
            return Err(IoError::other("injected immediate barrier failure").into());
        }
        self.inner.sync_data()?;
        if self.armed && !self.failed && matches!(self.mode, FailureMode::BarrierAfter) {
            self.failed = true;
            return Err(IoError::other("injected immediate barrier failure").into());
        }
        Ok(())
    }
}

impl FileImage for FailureImage {
    fn read_disk_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_disk_at(offset, bytes)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let target = self.before_write(self.inner.len(), bytes)?;
        let offset = self.inner.append(bytes)?;
        self.after_write(target)?;
        Ok(offset)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        let target = self.before_write(offset, bytes)?;
        self.inner.write_at(offset, bytes)?;
        self.after_write(target)
    }

    fn truncate(&mut self, length: u64) -> Result<(), Error> {
        self.inner.truncate(length)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.sync()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.sync()
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.sync()
    }

    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.inner.issued_write_order()
    }

    fn as_slice(&self) -> Option<&[u8]> {
        self.inner.as_slice()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(mode)
    }
}

impl Source for FailureImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, buffer)
    }
}

const HEADER_STATISTICS_OFFSET: u64 = 12;
const DATA_BLOCK_SIGNATURE: &[u8; 4] = b"EADB";
