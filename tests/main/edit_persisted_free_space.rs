use std::path::Path;

use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::SpaceOwnership;
use rstest::rstest;
use test_util::temp;
use test_util::userblock::Userblock;
use test_util_hdf5::file_space;

#[rstest]
fn a_session_reuses_the_free_space_an_earlier_session_persisted(
    #[values(0, USERBLOCK)] userblock: u64,
) {
    let path = temp::temp_path(&format!("persisted_reuse_{userblock}.h5"));
    let stamped = write_fixture(&path, userblock);
    let hole = file_space::contiguous_extent(&File::open(&path).unwrap(), "b");

    let first = File::open_rw(&path).unwrap();
    first.root().delete("b").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let persisted = File::open(&path).unwrap().persisted_free_space().unwrap();
    assert!(
        persisted.contains(&(hole.start - userblock, HOLE_BYTES)),
        "{persisted:?}"
    );

    let second = File::open_rw(&path).unwrap();
    assert_eq!(
        second.space_accounting().unwrap().reusable_free_space,
        persisted
            .iter()
            .map(|&(at, len)| (at + userblock, len))
            .collect::<Vec<_>>()
    );
    second
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    second.commit().unwrap();
    second.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(
        file_space::contiguous_extent(&file, "d"),
        hole.start..hole.start + 1200
    );
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(file.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    stamped.assert_unchanged(&path);
}

#[rstest]
fn every_commit_keeps_the_userblock_and_writes_stored_manager_addresses(
    #[values(0, USERBLOCK)] userblock: u64,
) {
    let path = temp::temp_path(&format!("persisted_stored_addresses_{userblock}.h5"));
    let stamped = write_fixture(&path, userblock);
    let mut manager_addrs = Vec::new();
    for step in Step::ALL {
        let session = File::open_rw(&path).unwrap();
        step.apply(&session);
        session.commit().unwrap();
        let reusable = session.space_accounting().unwrap().reusable_free_space;
        let ownership = session.__space_ownership().unwrap();
        session.close().unwrap();

        stamped.assert_unchanged(&path);
        let SpaceOwnership {
            userblock_space,
            live_raw_space,
            live_metadata_space,
            free_space_manager_space,
            reusable_free_space,
            ..
        } = ownership;
        assert_eq!(
            userblock_space,
            Vec::from_iter((userblock > 0).then_some((0, userblock)))
        );
        assert!(
            live_raw_space
                .iter()
                .chain(&live_metadata_space)
                .chain(&free_space_manager_space)
                .chain(&reusable_free_space)
                .all(|&(at, _)| at >= userblock),
            "{step:?} allocated inside the userblock"
        );

        let file = File::open(&path).unwrap();
        let stored_free = file.persisted_free_space().unwrap();
        assert_eq!(
            stored_free,
            reusable
                .iter()
                .map(|&(at, len)| (at - userblock, len))
                .collect::<Vec<_>>(),
            "{step:?}"
        );
        let header = file.file_space_info().unwrap().manager_addrs[0];
        assert_eq!(
            free_space_manager_space
                .first()
                .map(|&(at, _)| at - userblock),
            (header != u64::MAX).then_some(header),
            "{step:?}"
        );
        assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
        manager_addrs.push(header);
    }
    assert!(
        manager_addrs.windows(2).any(|pair| pair[0] != pair[1]),
        "no commit relocated the manager: {manager_addrs:?}"
    );
}

#[rstest]
fn a_deletion_that_frees_the_end_of_the_file_truncates_it(
    #[values(0, USERBLOCK)] userblock: u64,
    #[values(false, true)] persist: bool,
) {
    let path = temp::temp_path(&format!("persisted_truncate_{userblock}_{persist}.h5"));
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, persist, 1);
    builder.write(&path).unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());

    let before = std::fs::metadata(&path).unwrap().len();
    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("tail", |builder| {
            builder.with_i32_data(&vec![5; TAIL_ELEMENTS]);
        })
        .unwrap();
    session.commit().unwrap();
    let grown = std::fs::metadata(&path).unwrap().len();
    session.root().delete("tail").unwrap();
    session.commit().unwrap();
    let truncated = std::fs::metadata(&path).unwrap().len();
    let free_end = session
        .space_accounting()
        .unwrap()
        .reusable_free_space
        .last()
        .map_or(0, |&(at, len)| at + len);
    session.close().unwrap();

    assert!(
        grown > before + TAIL_BYTES && truncated < before + TAIL_BYTES / 16,
        "{before} bytes grew to {grown} and shrank to {truncated}"
    );
    assert!(free_end <= truncated, "{free_end} is past {truncated}");
    let file = File::open(&path).unwrap();
    assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    stamped.assert_unchanged(&path);
}

#[derive(Clone, Copy, Debug)]
enum Step {
    AddD,
    AddE,
    DeleteB,
    DeleteC,
}

impl Step {
    const ALL: [Self; 4] = [Self::DeleteB, Self::AddD, Self::DeleteC, Self::AddE];

    fn apply(self, session: &File) {
        match self {
            Self::AddD => {
                session
                    .root()
                    .create_dataset("d", |builder| {
                        builder.with_i32_data(&[9; 300]);
                    })
                    .unwrap();
            }
            Self::AddE => {
                session
                    .root()
                    .create_dataset("e", |builder| {
                        builder.with_i32_data(&[8; 50]);
                    })
                    .unwrap();
            }
            Self::DeleteB => session.root().delete("b").unwrap(),
            Self::DeleteC => session.root().delete("c").unwrap(),
        }
    }
}

fn write_fixture(path: &Path, userblock: u64) -> Userblock {
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    builder.write(path).unwrap();
    Userblock::read(path, usize::try_from(userblock).unwrap())
}

fn userblock_builder(userblock: u64) -> FileBuilder {
    let mut builder = FileBuilder::new();
    if userblock > 0 {
        builder.with_userblock(userblock);
        builder.with_userblock_content(USERBLOCK_CONTENT);
    }
    builder
}

const HOLE_BYTES: u64 = 1600;
const TAIL_BYTES: u64 = 4 * TAIL_ELEMENTS as u64;
const TAIL_ELEMENTS: usize = 16_384;
const USERBLOCK: u64 = 512;
const USERBLOCK_CONTENT: &[u8] = b"persisted free space";
