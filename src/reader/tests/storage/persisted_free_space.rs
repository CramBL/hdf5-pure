use std::collections::BTreeSet;
use std::path::Path;

use rstest::rstest;
use test_util::free_space;
use test_util::range;
use test_util::temp;
use test_util::userblock::Userblock;

use crate::AttrValue;
use crate::File;
use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FreeSectionCoordinates;
use crate::UserblockSize;
use crate::edit::SpaceOwnership;
use crate::reader::test_util as reader_test_util;

#[rstest]
fn a_session_reuses_the_free_space_an_earlier_session_persisted(
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(strategy);
    let path = temp::temp_path(&format!("persisted_reuse_{strategy:?}_{userblock}.h5"));
    let stamped = write_fixture(&path, strategy, userblock);
    let hole = reader_test_util::contiguous_extent(&File::open(&path).unwrap(), "b");

    let first = File::open_rw(&path).unwrap();
    first.root().delete("b").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let persisted = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    let second = File::open_rw(&path).unwrap();
    let ownership = reader_test_util::ownership(&second).unwrap();
    assert!(ownership.analysis_complete);
    let mut available_or_reused = ownership.reusable_free_space;
    available_or_reused.extend(ownership.live_metadata_space);
    available_or_reused.extend(ownership.free_space_manager_space);
    assert_eq!(range::covered_len(&available_or_reused, &hole), HOLE_BYTES);
    let expected = persisted
        .free_sections()
        .sections()
        .iter()
        .filter(|section| {
            section.address() + userblock >= hole.start
                && section.end_address() + userblock <= hole.end
                && section.length().get() >= 1_200
        })
        .map(|section| section.address() + userblock)
        .min()
        .unwrap();
    let reusable = second.reusable_free_sections().unwrap();
    assert_eq!(
        reusable.section_count(),
        persisted.free_sections().section_count()
    );
    for (physical, stored) in reusable
        .sections()
        .iter()
        .zip(persisted.free_sections().sections())
    {
        assert_eq!(physical.coordinates(), FreeSectionCoordinates::Absolute);
        assert_eq!(stored.coordinates(), FreeSectionCoordinates::Stored);
        assert_eq!(physical.address(), stored.address() + userblock);
        assert_eq!(physical.length(), stored.length());
    }
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
        reader_test_util::contiguous_extent(&file, "d"),
        expected..expected + 1_200
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
        if strategy == Strategy::Page {
            assert_owned_pages_homogeneous(&session, PAGE);
        }
        session.close().unwrap();
        let settled = File::open_rw(&path).unwrap();
        let reusable = settled.reusable_free_sections().unwrap();
        let ownership = reader_test_util::ownership(&settled).unwrap();
        settled.close().unwrap();

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
        let stored_free = file.persisted_free_sections().unwrap().unwrap();
        assert_eq!(
            stored_free.free_sections().section_count(),
            reusable.section_count(),
            "{step:?}"
        );
        for (section, physical) in stored_free
            .free_sections()
            .sections()
            .iter()
            .zip(reusable.sections())
        {
            assert_eq!(
                section.coordinates(),
                FreeSectionCoordinates::Stored,
                "{step:?}"
            );
            assert_eq!(
                physical.coordinates(),
                FreeSectionCoordinates::Absolute,
                "{step:?}"
            );
            assert_eq!(
                section.address(),
                physical.address() - userblock,
                "{step:?}"
            );
            assert_eq!(section.length(), physical.length(), "{step:?}");
        }
        let headers = file
            .file_space_info()
            .unwrap()
            .unwrap()
            .manager_addrs
            .iter()
            .filter_map(|&addr| match addr {
                crate::DecodedStoredAddress::Defined(stored) => Some(stored.get()),
                crate::DecodedStoredAddress::Undefined => None,
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
fn a_reopened_paged_session_reuses_each_page_type_in_its_own_pages(
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(Strategy::Page);
    let path = temp::temp_path(&format!("persisted_page_types_{userblock}.h5"));
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("pages").with_i32_data(&[2; 2_048]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    Strategy::Page.configure(&mut builder, true);
    builder.write(&path).unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());
    let file = File::open(&path).unwrap();
    let pages = reader_test_util::contiguous_extent(&file, "pages");
    let c = reader_test_util::contiguous_extent(&file, "c");
    drop(file);
    assert_eq!(pages.start, userblock + 2 * PAGE);
    assert_eq!(c.end, userblock + PAGE + 800);

    let first = File::open_rw(&path).unwrap();
    first.root().delete("pages").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let persisted = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    assert!(
        persisted.free_sections().sections().iter().any(|section| {
            section.coordinates() == FreeSectionCoordinates::Stored
                && section.address() == pages.start - userblock
                && section.length().get() == PAGE
        }),
        "{persisted:?}"
    );

    let second = File::open_rw(&path).unwrap();
    let before = reader_test_util::ownership(&second).unwrap();
    second
        .root()
        .create_group_with("g", |group| {
            group.set_attr("payload", AttrValue::U8Array(vec![7; 3_000]));
        })
        .unwrap();
    second
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    second.commit().unwrap();
    let after = reader_test_util::ownership(&second).unwrap();
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
        reader_test_util::contiguous_extent(&file, "d"),
        c.end..c.end + 1_200
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
    /// Returns zero for [`Self::Zero`], [`PAGE`] for a paged file, and [`USERBLOCK`] otherwise.
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

/// Asserts that raw and metadata extents occupy separate pages.
///
/// Free-space-manager allocations count as metadata. Pages are counted from the file's base
/// address.
///
/// # Panics
///
/// Panics if ownership cannot be read, an extent cannot be mapped to a page, or a page contains
/// both raw data and metadata.
#[track_caller]
fn assert_owned_pages_homogeneous(file: &File, page_size: u64) {
    let SpaceOwnership {
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        ..
    } = reader_test_util::ownership(file).unwrap();
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
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        builder.with_userblock_content(USERBLOCK_CONTENT);
    }
    builder
}

const HOLE_BYTES: u64 = 1_600;
const PAGE: u64 = 4_096;
const USERBLOCK: u64 = 512;
const USERBLOCK_CONTENT: &[u8] = b"persisted free space";
