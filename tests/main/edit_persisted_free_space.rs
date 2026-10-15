use std::collections::BTreeSet;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::SpaceOwnership;
use hdf5_pure_core::__private::StoredAddressExt;
use rstest::rstest;
use test_util::free_space;
use test_util::temp;
use test_util::userblock::Userblock;
use test_util_hdf5::file_space;

#[rstest]
fn a_session_reuses_the_free_space_an_earlier_session_persisted(
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(strategy);
    let path = temp::temp_path(&format!("persisted_reuse_{strategy:?}_{userblock}.h5"));
    let stamped = write_fixture(&path, strategy, userblock);
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
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(strategy);
    let path = temp::temp_path(&format!(
        "persisted_stored_addresses_{strategy:?}_{userblock}.h5"
    ));
    let stamped = write_fixture(&path, strategy, userblock);
    let mut manager_addrs = Vec::new();
    for step in Step::ALL {
        let session = File::open_rw(&path).unwrap();
        step.apply(&session);
        session.commit().unwrap();
        let reusable = session.space_accounting().unwrap().reusable_free_space;
        let ownership = session.__space_ownership().unwrap();
        if strategy == Strategy::Page {
            assert_owned_pages_homogeneous(&session, PAGE);
        }
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
        let headers = file
            .file_space_info()
            .unwrap()
            .manager_addrs
            .iter()
            .filter_map(|&addr| match addr {
                hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored.get()),
                hdf5_pure::DecodedStoredAddress::Undefined => None,
            })
            .collect::<BTreeSet<_>>();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            free_space_manager_space
                .iter()
                .map(|&(at, _)| at)
                .filter(
                    |&at| bytes[usize::try_from(at).unwrap()..].starts_with(free_space::SIGNATURE)
                )
                .map(|at| at - userblock)
                .collect::<BTreeSet<_>>(),
            headers,
            "{step:?}"
        );
        assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
        if strategy == Strategy::Page {
            assert_eq!((file.file_size() - userblock) % PAGE, 0, "{step:?}");
        }
        manager_addrs.push(headers);
    }
    assert!(
        manager_addrs.windows(2).any(|pair| pair[0] != pair[1]),
        "no commit relocated the manager: {manager_addrs:?}"
    );
}

#[rstest]
#[case::fsm_aggr_transient(Strategy::FsmAggr, false, 65_681, 0)]
#[case::fsm_aggr_persisting(Strategy::FsmAggr, true, 65_956, 1145)]
#[case::page_persisting(Strategy::Page, true, TAIL_BYTES, PAGE)]
fn a_deletion_that_frees_the_end_of_the_file_truncates_it(
    #[case] strategy: Strategy,
    #[case] persist: bool,
    #[case] growth: u64,
    #[case] kept: u64,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(strategy);
    let path = temp::temp_path(&format!(
        "persisted_truncate_{strategy:?}_{userblock}_{persist}.h5"
    ));
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    strategy.configure(&mut builder, persist);
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

    assert_eq!((grown - before, truncated - before), (growth, kept));
    assert!(free_end <= truncated, "{free_end} is past {truncated}");
    let file = File::open(&path).unwrap();
    assert_eq!(file.file_size(), truncated);
    if strategy == Strategy::Page {
        assert_eq!(
            file.file_space_info().unwrap().eoa_pre_fsm,
            Some(hdf5_pure::DecodedStoredAddress::Defined(
                hdf5_pure::StoredAddress::new(truncated - userblock)
            ))
        );
        assert_eq!((truncated - userblock) % PAGE, 0);
    }
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    stamped.assert_unchanged(&path);
}

#[rstest]
fn a_reopened_paged_session_reuses_each_page_type_in_its_own_pages(
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(Strategy::Page);
    let path = temp::temp_path(&format!("persisted_page_types_{userblock}.h5"));
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("pages").with_i32_data(&[2; 2048]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    Strategy::Page.configure(&mut builder, true);
    builder.write(&path).unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());
    let file = File::open(&path).unwrap();
    let pages = file_space::contiguous_extent(&file, "pages");
    let c = file_space::contiguous_extent(&file, "c");
    drop(file);
    assert_eq!(pages.start, userblock + 2 * PAGE);
    assert_eq!(c.end, userblock + PAGE + 800);

    let first = File::open_rw(&path).unwrap();
    first.root().delete("pages").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let persisted = File::open(&path).unwrap().persisted_free_space().unwrap();
    assert!(
        persisted.contains(&(pages.start - userblock, PAGE)),
        "{persisted:?}"
    );

    let second = File::open_rw(&path).unwrap();
    let before = second.__space_ownership().unwrap();
    second
        .root()
        .create_group_with("g", |group| {
            group.set_attr("payload", AttrValue::U8Array(vec![7; 3000]));
        })
        .unwrap();
    second
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    second.commit().unwrap();
    let after = second.__space_ownership().unwrap();
    assert_owned_pages_homogeneous(&second, PAGE);
    second.close().unwrap();

    assert_eq!(after.physical_size, before.physical_size);
    let new_metadata = after
        .live_metadata_space
        .iter()
        .filter(|extent| !before.live_metadata_space.contains(extent))
        .map(|&(at, _)| at)
        .collect::<BTreeSet<_>>();
    assert!(
        new_metadata.contains(&pages.start),
        "the group header {new_metadata:?} does not start the freed page"
    );
    assert!(
        new_metadata
            .iter()
            .any(|&at| (userblock..userblock + PAGE).contains(&at)),
        "no header {new_metadata:?} reused the first metadata page"
    );
    let file = File::open(&path).unwrap();
    assert_eq!(
        file_space::contiguous_extent(&file, "d"),
        c.end..c.end + 1200
    );
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(file.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
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

#[derive(Clone, Copy, Debug)]
enum Base {
    Userblock,
    Zero,
}

impl Base {
    /// Returns the userblock size, one page under `Strategy::Page`, since the writer requires a
    /// paged file's userblock to be a whole number of pages.
    fn userblock(self, strategy: Strategy) -> u64 {
        match (self, strategy) {
            (Self::Zero, _) => 0,
            (Self::Userblock, Strategy::FsmAggr) => USERBLOCK,
            (Self::Userblock, Strategy::Page) => PAGE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Strategy {
    FsmAggr,
    Page,
}

impl Strategy {
    fn configure(self, builder: &mut FileBuilder, persist: bool) {
        match self {
            Self::FsmAggr => {
                builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, persist, 1);
            }
            Self::Page => {
                builder
                    .with_file_space_strategy(FileSpaceStrategy::Page, persist, 1)
                    .with_file_space_page_size(PAGE);
            }
        }
    }
}

/// Asserts that no page, counted from the base address, holds both live raw data and metadata, the
/// free-space managers included, in the ownership report of `file`.
#[track_caller]
fn assert_owned_pages_homogeneous(file: &File, page_size: u64) {
    let SpaceOwnership {
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        ..
    } = file.__space_ownership().unwrap();
    let base = file.superblock().base_address.get();
    let pages = |extents: &[(u64, u64)]| {
        extents
            .iter()
            .flat_map(|&(at, len)| (at - base) / page_size..=(at + len - 1 - base) / page_size)
            .collect::<BTreeSet<_>>()
    };
    let metadata = pages(&live_metadata_space)
        .union(&pages(&free_space_manager_space))
        .copied()
        .collect::<BTreeSet<_>>();
    let mixed = pages(&live_raw_space)
        .intersection(&metadata)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(mixed, Vec::<u64>::new(), "pages hold raw data and metadata");
}

fn write_fixture(path: &Path, strategy: Strategy, userblock: u64) -> Userblock {
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    strategy.configure(&mut builder, true);
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
const PAGE: u64 = 4096;
const TAIL_BYTES: u64 = 4 * TAIL_ELEMENTS as u64;
const TAIL_ELEMENTS: usize = 16_384;
const USERBLOCK: u64 = 512;
const USERBLOCK_CONTENT: &[u8] = b"persisted free space";
