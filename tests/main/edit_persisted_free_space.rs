use std::collections::BTreeSet;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::UserblockSize;
use hdf5_pure_core::__private::StoredAddressExt;
use rstest::rstest;
use test_util::free_space;
use test_util::range;
use test_util::temp;
use test_util::userblock::Userblock;
use test_util_hdf5::file_space;
use test_util_hdf5::paged;
use test_util_hdf5::storage;

#[rstest]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::mirrored(MemoryStrategy::Mirrored, 0)]
#[case::userblock(MemoryStrategy::Mirrored, 4_096)]
fn paged_manager_tail_survives_reopen(#[case] memory: MemoryStrategy, #[case] userblock: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    for i in 0..60 {
        file.root()
            .create_dataset(&format!("b{i}"), |b| {
                b.with_i32_data(&[2; 100]);
            })
            .unwrap();
        file.commit().unwrap();
    }
    let before = file.space_accounting().unwrap().reusable_free_space;
    file.close().unwrap();
    let reopened = File::open_rw(&path).unwrap();
    assert_eq!(
        reopened.space_accounting().unwrap().reusable_free_space,
        before,
    );
    assert_eq!(reopened.dataset("a").unwrap().read_i32().unwrap(), [1; 100]);
    for i in 0..60 {
        assert_eq!(
            reopened
                .dataset(&format!("b{i}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            [2; 100],
        );
    }
}

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
    let bytes = std::fs::read(&path).unwrap();
    let second = File::open_rw(&path).unwrap();
    let mut available_or_reused = second.space_accounting().unwrap().reusable_free_space;
    available_or_reused.extend(storage::file_metadata_extents(&second, &bytes));
    assert_eq!(range::covered_len(&available_or_reused, &hole), HOLE_BYTES);
    let expected = persisted
        .iter()
        .filter(|&&(address, length)| {
            address + userblock >= hole.start
                && address + userblock + length <= hole.end
                && length >= 1200
        })
        .map(|&(address, _)| address + userblock)
        .min()
        .unwrap();
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
        expected..expected + 1200
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
        session.close().unwrap();
        if strategy == Strategy::Page {
            assert_pages_homogeneous(&path);
        }
        let settled = File::open_rw(&path).unwrap();
        let reusable = settled.space_accounting().unwrap().reusable_free_space;
        settled.close().unwrap();

        stamped.assert_unchanged(&path);
        let bytes = std::fs::read(&path).unwrap();
        let file = File::open(&path).unwrap();
        let free_space_manager_space = storage::manager_extents(&file, &bytes);
        let mut allocated = storage::file_metadata_extents(&file, &bytes);
        for dataset in file.root().datasets().unwrap() {
            let address = storage::root_object_address(&file, &bytes, &dataset);
            allocated.extend(storage::header_extents(&file, &bytes, address));
            allocated.extend(
                file_space::raw_extents(&file, &dataset)
                    .iter()
                    .map(|extent| (extent.start, extent.end - extent.start)),
            );
        }
        storage::assert_unavailable(&file, &allocated);
        assert!(reusable.iter().all(|&(at, _)| at >= userblock), "{step:?}");
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
#[case::fsm_aggr_transient(Strategy::FsmAggr, false, 67_584, 0)]
#[case::fsm_aggr_persisting(Strategy::FsmAggr, true, 65_536, 1_055)]
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
            file.file_space_info().unwrap().unwrap().eoa_pre_fsm,
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
    let before = second.file_size();
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
    assert_eq!(second.file_size(), before);
    second.close().unwrap();
    assert_pages_homogeneous(&path);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let mut new_metadata = storage::file_metadata_extents(&file, &bytes)
        .iter()
        .map(|&(at, _)| at)
        .collect::<BTreeSet<_>>();
    for name in ["g", "d"] {
        let at = storage::root_object_address(&file, &bytes, name);
        new_metadata.insert(at);
    }
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
                builder.with_file_space_settings(
                    FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(persist),
                );
            }
            Self::Page => {
                builder.with_file_space_settings(
                    FileSpaceSettings::new(FileSpaceStrategy::Page)
                        .with_persistence(persist)
                        .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
                );
            }
        }
    }
}

fn assert_pages_homogeneous(path: &Path) {
    let file = File::open(path).unwrap();
    let datasets = file.root().datasets().unwrap();
    file.close().unwrap();
    paged::assert_pages_homogeneous(
        path,
        PAGE,
        &datasets.iter().map(String::as_str).collect::<Vec<_>>(),
    );
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
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
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
