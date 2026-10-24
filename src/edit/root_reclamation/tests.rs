use std::fs;
use std::io::Error as IoError;
use std::io::ErrorKind;

use rstest::rstest;

use crate::AttrValue;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileLocking;
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
use crate::source::Source;
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::mirrored(MemoryStrategy::Mirrored, 0)]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::userblock(MemoryStrategy::Mirrored, 512)]
fn fixed_size_root_reclaims_its_original_allocation(
    #[case] backing: MemoryStrategy,
    #[case] userblock: u64,
    #[values(4_096, u64::MAX)] threshold: u64,
    #[values(false, true)] reopen: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("root.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, threshold);
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let options = FileAccessProperties::new().with_memory_strategy(backing);
    let mut file = File::open_rw_with_options(&path, options).unwrap();
    let root = file.superblock().root_group_address;
    let root_addr = file.superblock().base_address.get() + root.get();
    for generation in 0..4 {
        file.root()
            .set_attr("generation", AttrValue::I64(generation))
            .unwrap();
        file.commit().unwrap();
        assert_eq!(
            file.root().attrs().unwrap()["generation"],
            AttrValue::I64(generation)
        );
        let ownership = file.__space_ownership().unwrap();
        assert!(ownership.analysis_complete);
        let accounting = file.space_accounting().unwrap();
        assert!(
            accounting
                .reusable_free_space
                .iter()
                .all(|&(addr, len)| root_addr < addr || root_addr >= addr + len)
        );
        if reopen {
            file.close().unwrap();
            file = File::open_rw_with_options(&path, options).unwrap();
            assert_eq!(file.superblock().root_group_address, root);
        }
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.superblock().root_group_address, root);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [1, 2, 3]);
    assert_eq!(
        file.root().attrs().unwrap()["generation"],
        AttrValue::I64(3)
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
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, threshold);
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
    file.persisted_free_space().unwrap();
}

#[test]
fn root_reclamation_skips_reference_bearing_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reference.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, u64::MAX);
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
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, u64::MAX);
    builder.set_attr("generation", AttrValue::I64(-1));
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    if references {
        builder
            .create_dataset("refs")
            .with_path_references(&["keep"]);
    }
    builder.write(&path).unwrap();
    let root = File::open(&path).unwrap().superblock().root_group_address;
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailingImage {
            inner: WriteEngine::read_mirror(handle)?,
            failure,
            root: root.get(),
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
    if engine.publish_attempted {
        assert!(!engine.aggregation.unwrap().publication_confirmed);
    }
    let after_failure = engine.issued_bytes();
    if engine.publish_attempted {
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
    file.persisted_free_space().unwrap();
}

#[derive(Clone, Copy)]
enum Failure {
    BeforeFirstPublish,
    BeforeSecondPublish,
    FirstBarrier,
    RootCopy,
    SecondBarrier,
}

struct FailingImage {
    inner: MirrorImage,
    failure: Failure,
    root: u64,
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
            publications,
            failed,
        } = self;
        if !*failed {
            if offset == 0
                && matches!(
                    (*failure, *publications),
                    (Failure::BeforeFirstPublish, 0) | (Failure::BeforeSecondPublish, 1)
                )
            {
                return Self::fail(failed);
            }
            if offset == *root && *publications == 1 && matches!(failure, Failure::RootCopy) {
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
        self.inner.truncate(length)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        let Self {
            inner,
            failure,
            root: _,
            publications,
            failed,
        } = self;
        inner.sync_all()?;
        if !*failed
            && matches!(
                (*failure, *publications),
                (Failure::FirstBarrier, 1) | (Failure::SecondBarrier, 2)
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
