use std::fs;
use std::io::Error as IoError;
use std::io::ErrorKind;

use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use rstest::rstest;

use crate::AttrValue;
use crate::DecodedStoredAddress;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::MemoryStrategy;
use crate::Object;
use crate::UserblockSize;
use crate::edit::WriteEngine;
use crate::image::FileImage;
use crate::image::MirrorImage;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::reader::test_util as reader_test_util;
use crate::source::Source;
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::mirrored(MemoryStrategy::Mirrored, 0, false)]
#[case::bounded(MemoryStrategy::Bounded, 0, false)]
#[case::userblock(MemoryStrategy::Mirrored, 512, false)]
#[case::tracked_space(MemoryStrategy::Mirrored, 0, true)]
fn fixed_width_root_updates_reach_stable_physical_length(
    #[case] backing: MemoryStrategy,
    #[case] userblock: u64,
    #[case] tracked: bool,
    #[values(1, 200, 4_096, u64::MAX)] threshold: u64,
    #[values(false, true)] reopen: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("root.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(threshold),
    );
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    if tracked {
        builder.create_dataset("hole").with_u8_data(&[9; 8_192]);
        builder.create_dataset("guard").with_u8_data(&[7; 8_192]);
    }
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let options = FileAccessProperties::new().with_memory_strategy(backing);
    let mut file = File::open_rw_with_options(&path, options).unwrap();
    if tracked {
        file.root().delete("hole").unwrap();
        file.commit().unwrap();
        file.close().unwrap();
        file = File::open_rw_with_options(&path, options).unwrap();
    }
    let root = file.superblock().root_group_address;
    let mut lengths = Vec::new();
    let mut closed_lengths = Vec::new();
    for generation in 0..12 {
        file.root()
            .set_attr("generation", AttrValue::I64(generation))
            .unwrap();
        file.commit().unwrap();
        assert_eq!(
            file.root().attrs().unwrap()["generation"],
            AttrValue::I64(generation)
        );
        lengths.push(fs::metadata(&path).unwrap().len());
        let ownership = reader_test_util::ownership(&file).unwrap();
        assert!(ownership.analysis_complete);
        let sections = file.reusable_free_sections().unwrap();
        let root_addr =
            file.superblock().base_address.get() + file.superblock().root_group_address.get();
        if threshold > 1 && !tracked {
            assert!(
                sections
                    .sections()
                    .iter()
                    .all(|section| root_addr < section.address()
                        || root_addr >= section.end_address())
            );
        }
        if reopen {
            file.close().unwrap();
            closed_lengths.push(fs::metadata(&path).unwrap().len());
            file = File::open_rw_with_options(&path, options).unwrap();
            if threshold > 1 && !tracked {
                assert_eq!(file.superblock().root_group_address, root);
            }
        }
    }
    assert!(
        lengths[4..].iter().all(|&len| len == lengths[4]),
        "{lengths:?}"
    );
    if reopen {
        assert!(
            closed_lengths[4..]
                .iter()
                .all(|&len| len == closed_lengths[4]),
            "{closed_lengths:?}"
        );
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    if threshold > 1 && !tracked {
        assert_eq!(file.superblock().root_group_address, root);
    }
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        file.root().attrs().unwrap()["generation"],
        AttrValue::I64(11)
    );
    assert_eq!(
        &fs::read(&path).unwrap()[..userblock as usize],
        &original[..userblock as usize]
    );
}

#[rstest]
#[case::eligible_for_ordinary_admission(1, AttrValue::I64(7))]
#[case::larger_root(u64::MAX, AttrValue::AsciiString("wider value".into()))]
fn ordinary_root_relocation_remains_available(#[case] threshold: u64, #[case] value: AttrValue) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("relocated.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(threshold),
    );
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let old_root = file.superblock().root_group_address;
    file.root().set_attr("generation", value.clone()).unwrap();
    file.commit().unwrap();
    assert_eq!(file.root().attrs().unwrap()["generation"], value);
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_ne!(file.superblock().root_group_address, old_root);
    file.persisted_free_sections().unwrap().unwrap();
}

#[test]
fn root_reclamation_skips_reference_bearing_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reference.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(u64::MAX),
    );
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("refs").with_path_references(&[""]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let old_root = file.superblock().root_group_address;
    file.root()
        .set_attr("generation", AttrValue::I64(7))
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_ne!(file.superblock().root_group_address, old_root);
    let refs = file.dataset("refs").unwrap().dereference().unwrap();
    let [Object::Group(root)] = refs.as_slice() else {
        panic!("expected a reference to the root, got {refs:?}");
    };
    assert_eq!(root.attrs().unwrap()["generation"], AttrValue::I64(7));
}

#[rstest]
#[case::first_publication(Failure::BeforeFirstPublish, false, false, false)]
#[case::first_publication_barrier(Failure::FirstBarrier, true, false, false)]
#[case::root_copy(Failure::RootCopy, true, false, false)]
#[case::second_publication(Failure::BeforeSecondPublish, true, false, false)]
#[case::second_publication_barrier(Failure::SecondBarrier, true, true, false)]
#[case::tail_reuse_copy(Failure::TailCopy, true, false, false)]
#[case::tail_reuse_publication(Failure::TailPublish, true, false, false)]
#[case::tail_reuse_barrier(Failure::TailBarrier, true, false, false)]
#[case::tail_reuse_truncation(Failure::TailTruncate, true, false, false)]
#[case::reference_proof_rollback(Failure::BeforeFirstPublish, false, false, true)]
fn root_reclamation_failure_preserves_a_readable_committed_root(
    #[case] failure: Failure,
    #[case] updated: bool,
    #[case] reclaimed: bool,
    #[case] references: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(u64::MAX),
    );
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    if references {
        builder
            .create_dataset("refs")
            .with_path_references(&["keep"]);
    }
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let root = file.superblock().root_group_address;
    let tail = file
        .superblock()
        .superblock_extension_address
        .unwrap()
        .get();
    drop(file);
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailingImage {
            inner: WriteEngine::read_mirror(handle)?,
            failure,
            root: root.get(),
            tail,
            publications: 0,
            failed: false,
        }))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "generation", AttrValue::I64(7))
        .unwrap();
    if references {
        engine.delete(&ObjectPathBuf::parse("refs")).unwrap();
        let mut replacement = DatasetBuilder::new("");
        replacement.with_i32_data(&[5]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("refs"), replacement)
            .unwrap();
    }
    let err = engine.commit().unwrap_err();
    let Error::Io(err) = err else {
        panic!("expected injected I/O failure, got {err:?}");
    };
    assert_eq!(err.kind(), ErrorKind::Other);
    if engine.publication.attempted() {
        assert!(!engine.aggregation.unwrap().publication_confirmed);
    }
    let after_failure = engine.issued_bytes();
    if engine.publication.attempted() {
        engine.finalize_persist().unwrap();
        assert_eq!(engine.issued_bytes(), after_failure);
    }
    if references {
        assert!(!engine.proved_free_of_references);
        engine
            .set_dataset_attr(&ObjectPathBuf::parse("keep"), "seen", AttrValue::I64(9))
            .unwrap();
        engine.commit().unwrap();
    }
    drop(engine);
    if !references {
        assert_eq!(fs::read(&path).unwrap(), after_failure);
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.root().attrs().unwrap()["generation"],
        AttrValue::I64(if updated { 7 } else { -1 })
    );
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    if references {
        let refs = file.dataset("refs").unwrap().dereference().unwrap();
        let [Object::Dataset(dataset)] = refs.as_slice() else {
            panic!("expected a reference to the dataset, got {refs:?}");
        };
        assert_eq!(dataset.attrs().unwrap()["seen"], AttrValue::I64(9));
    } else {
        assert_eq!(
            file.superblock().root_group_address == root,
            !updated || reclaimed
        );
    }
    file.persisted_free_sections().unwrap().unwrap();
}

#[rstest]
#[case::before_publication(Failure::BeforeFirstPublish, false)]
#[case::copy(Failure::TailCopy, true)]
#[case::publication(Failure::TailPublish, true)]
#[case::barrier(Failure::TailBarrier, true)]
fn tail_recycling_failure_preserves_committed_manager_sections(
    #[case] failure: Failure,
    #[case] updated: bool,
    #[values(false, true)] section_list: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manager-failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(4_096),
    );
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("hole").with_u8_data(&[9; 8_192]);
    builder.create_dataset("keep").with_u8_data(&[7; 8_192]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root().delete("hole").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let bytes = fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let root = file.superblock().root_group_address.get();
    let info = file.file_space_info().unwrap().unwrap();
    let manager = info
        .manager_addrs
        .iter()
        .find_map(|addr| match addr {
            DecodedStoredAddress::Defined(addr) => Some(addr.get()),
            DecodedStoredAddress::Undefined => None,
        })
        .unwrap();
    let tail = if section_list {
        FreeSpaceManagerHeader::parse(file.widths(), &bytes[usize::try_from(manager).unwrap()..])
            .unwrap()
            .section_list_addr()
            .get()
    } else {
        manager
    };
    drop(file);
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailingImage {
            inner: WriteEngine::read_mirror(handle)?,
            failure,
            root,
            tail,
            publications: 0,
            failed: false,
        }))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "generation", AttrValue::I64(7))
        .unwrap();
    let err = engine.commit().unwrap_err();
    let Error::Io(err) = err else {
        panic!("expected injected I/O failure, got {err:?}");
    };
    assert_eq!(err.kind(), ErrorKind::Other);
    if engine.publication.attempted() {
        assert!(!engine.aggregation.unwrap().publication_confirmed);
        let issued = engine.issued_bytes();
        engine.finalize_persist().unwrap();
        assert_eq!(engine.issued_bytes(), issued);
    }
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.root().attrs().unwrap()["generation"],
        AttrValue::I64(if updated { 7 } else { -1 })
    );
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 8_192]);
    assert!(
        file.persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .section_count()
            > 0
    );
    drop(file);
    let file = File::open_rw(&path).unwrap();
    assert!(
        reader_test_util::ownership(&file)
            .unwrap()
            .analysis_complete
    );
}

#[test]
fn empty_manager_recycling_recovers_exact_allocation_after_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty-manager.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(4_096),
    );
    builder.create_dataset("hole").with_u8_data(&[9; 8_192]);
    builder.create_dataset("keep").with_u8_data(&[7; 8_192]);
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    let mut replacement = DatasetBuilder::new("");
    replacement.with_u8_data(&[5; 8_192]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("replacement"), replacement)
        .unwrap();
    engine.commit().unwrap();
    let owned = engine.persist.as_ref().unwrap().old_blocks.clone();
    engine.force_sync().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .section_count(),
        0
    );
    assert!(
        file.file_space_info()
            .unwrap()
            .unwrap()
            .manager_addrs
            .iter()
            .all(|addr| *addr == DecodedStoredAddress::Undefined)
    );
    assert_eq!(
        file.dataset("replacement").unwrap().read_u8().unwrap(),
        [5; 8_192]
    );
    drop(file);
    let engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_eq!(engine.persist.as_ref().unwrap().old_blocks, owned);
}

#[derive(Clone, Copy)]
enum Failure {
    BeforeFirstPublish,
    BeforeSecondPublish,
    FirstBarrier,
    RootCopy,
    SecondBarrier,
    TailBarrier,
    TailCopy,
    TailPublish,
    TailTruncate,
}

struct FailingImage {
    inner: MirrorImage,
    failure: Failure,
    root: u64,
    tail: u64,
    publications: usize,
    failed: bool,
}

impl FailingImage {
    fn fail(failed: &mut bool) -> Result<(), Error> {
        *failed = true;
        Err(IoError::other("root reclamation failure").into())
    }
}

impl FileImage for FailingImage {
    fn read_disk_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_disk_at(offset, bytes)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        let Self {
            inner,
            failure,
            root,
            tail,
            publications,
            failed,
        } = self;
        if !*failed {
            if offset == 0
                && matches!(
                    (*failure, *publications),
                    (Failure::BeforeFirstPublish, 0)
                        | (Failure::BeforeSecondPublish, 2)
                        | (Failure::TailPublish, 1)
                )
            {
                return Self::fail(failed);
            }
            if offset == *tail && *publications == 1 && matches!(failure, Failure::TailCopy) {
                inner.write_at(offset, &bytes[..bytes.len() / 2])?;
                return Self::fail(failed);
            }
            if offset == *root && *publications == 2 && matches!(failure, Failure::RootCopy) {
                inner.write_at(offset, &bytes[..bytes.len() / 2])?;
                return Self::fail(failed);
            }
        }
        inner.write_at(offset, bytes)?;
        if offset == 0 {
            *publications += 1;
        }
        Ok(())
    }

    fn truncate(&mut self, length: u64) -> Result<(), Error> {
        let Self {
            inner,
            failure,
            root: _,
            tail: _,
            publications,
            failed,
        } = self;
        if !*failed && *publications == 2 && matches!(failure, Failure::TailTruncate) {
            return Self::fail(failed);
        }
        inner.truncate(length)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        let Self {
            inner,
            failure,
            root: _,
            tail: _,
            publications,
            failed,
        } = self;
        inner.sync_all()?;
        if !*failed
            && matches!(
                (*failure, *publications),
                (Failure::FirstBarrier, 1)
                    | (Failure::SecondBarrier, 3)
                    | (Failure::TailBarrier, 2)
            )
        {
            return Self::fail(failed);
        }
        Ok(())
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.inner.ordering_barrier()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(mode)
    }

    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.inner.issued_write_order()
    }
}

impl Source for FailingImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, bytes)
    }
}

impl WriteEngine {
    fn issued_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; usize::try_from(self.image.len()).unwrap()];
        self.image.read_disk_at(0, &mut bytes).unwrap();
        bytes
    }
}
