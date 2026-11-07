#![cfg(feature = "__hdf5-1.10")]

use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5::dataset::Layout as NativeLayout;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy as NativeSpace;
use hdf5_pure::EditBacking;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::PersistedFreeSpacePolicy;
use rstest::rstest;
use test_util_hdf5::absence;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::numeric::ByteOrder;

#[rstest]
#[case::together(Removal::Together)]
#[case::containing_group(Removal::ContainingGroup)]
#[case::sequential(Removal::Sequential)]
#[case::surviving_alias(Removal::SurvivingAlias)]
fn final_link_commits_reuse_the_original_contiguous_allocation_after_reopen(
    #[case] removal: Removal,
    #[values(
        Session::BoundedFsm,
        Session::BoundedPage,
        Session::MirroredFsm,
        Session::MirroredPage
    )]
    session: Session,
) {
    let path = test_util::temp::temp_path("native-final-links.h5");
    let extent = fixture(&path, session, removal);
    let original_size = fs::metadata(&path).unwrap().len();
    let mut file = session.open(&path);
    if matches!(removal, Removal::Sequential) {
        file.group("links").unwrap().delete("a").unwrap();
        file.commit().unwrap();
        assert_extent(&file, &extent, false);
        file.close().unwrap();
        assert_native(&path, removal, Some(("links/b", 1)), false);
        file = session.open(&path);
        assert_extent(&file, &extent, false);
        file.group("links").unwrap().delete("b").unwrap();
    } else if matches!(removal, Removal::ContainingGroup) {
        file.root().delete("links").unwrap();
    } else {
        let group = file.group("links").unwrap();
        group.delete("a").unwrap();
        group.delete("b").unwrap();
    }
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    let retained = matches!(removal, Removal::SurvivingAlias);
    assert_extent(&file, &extent, !retained);
    assert_eq!(
        file.dataset("survivor").unwrap().read_i32().unwrap(),
        survivor()
    );
    file.close().unwrap();
    assert_native(&path, removal, retained.then_some(("outside", 1)), false);
    file_space_oracle::assert_persisted_sections(&path);
    file = session.open(&path);
    assert_extent(&file, &extent, !retained);
    if retained {
        assert_eq!(
            file.dataset("outside").unwrap().read_i32().unwrap(),
            source()
        );
        file.root().delete("outside").unwrap();
        file.commit().unwrap();
        assert_extent(&file, &extent, true);
        file.close().unwrap();
        assert_native(&path, removal, None, false);
        file_space_oracle::assert_persisted_sections(&path);
        file = session.open(&path);
        assert_extent(&file, &extent, true);
    }
    let before = file.file_size();
    let available = file.reusable_free_sections().unwrap();
    assert!(before <= original_size + GROWTH_ALLOWANCE);
    drop(
        file.root()
            .create_dataset("replacement", |dataset| {
                dataset.with_i32_data(&replacement());
            })
            .unwrap(),
    );
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    assert_eq!(
        file.dataset("replacement").unwrap().read_i32().unwrap(),
        replacement()
    );
    assert!(file.file_size() <= before + GROWTH_ALLOWANCE);
    assert!(file.file_size() <= original_size + GROWTH_ALLOWANCE);
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let dataset = native.dataset("replacement").unwrap();
    assert_eq!(dataset.layout(), NativeLayout::Contiguous);
    assert_eq!(dataset.storage_size(), extent.end - extent.start);
    let start = dataset.offset().unwrap();
    let replacement_extent = start..start + dataset.storage_size();
    assert!(
        available.sections().iter().any(|section| {
            section.address().get() <= extent.start
                && extent.end <= section.end_address().get()
                && section.address().get() <= replacement_extent.start
                && replacement_extent.end <= section.end_address().get()
        }),
        "original {extent:?}, replacement {replacement_extent:?}, available {available:?}"
    );
    assert!(replacement_extent.start < extent.end && extent.start < replacement_extent.end);
    drop(dataset);
    native.close().unwrap();
    assert_native(&path, removal, None, true);
    file_space_oracle::assert_persisted_sections(&path);
    let file = session.open(&path);
    assert_extent(&file, &replacement_extent, false);
    assert_eq!(
        file.dataset("replacement").unwrap().read_i32().unwrap(),
        replacement()
    );
    assert_eq!(
        file.dataset("survivor").unwrap().read_i32().unwrap(),
        survivor()
    );
    file.close().unwrap();
}

fn fixture(path: &Path, session: Session, removal: Removal) -> Range<u64> {
    let native = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties
                .file_space_strategy(NativeSpace::FreeSpaceManager {
                    paged: session.paged(),
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(PAGE_BYTES)
        })
        .create(path)
        .unwrap();
    let group = native.create_group("links").unwrap();
    let dataset = group
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .layout(NativeLayout::Contiguous)
        .shape([ELEMENTS])
        .create("a")
        .unwrap();
    dataset.write(&source()).unwrap();
    assert_eq!(dataset.layout(), NativeLayout::Contiguous);
    assert_eq!(dataset.storage_size(), PAYLOAD_BYTES);
    let start = dataset.offset().unwrap();
    let extent = start..start + dataset.storage_size();
    group.link_hard("a", "b").unwrap();
    let retained = matches!(removal, Removal::SurvivingAlias);
    if retained {
        native.link_hard("links/a", "outside").unwrap();
    }
    assert_eq!(
        group.loc_info_by_name("a").unwrap().num_links,
        if retained { 3 } else { 2 }
    );
    drop(dataset);
    drop(group);
    let dataset = native
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .layout(NativeLayout::Contiguous)
        .shape([SURVIVOR_ELEMENTS])
        .create("survivor")
        .unwrap();
    dataset.write(&survivor()).unwrap();
    assert_eq!(dataset.layout(), NativeLayout::Contiguous);
    assert_eq!(dataset.storage_size(), SURVIVOR_BYTES);
    assert!(extent.end <= dataset.offset().unwrap());
    drop(dataset);
    native.close().unwrap();
    assert!(extent.end < fs::metadata(path).unwrap().len());
    file_space_oracle::assert_persisted_sections(path);
    extent
}

fn assert_extent(file: &File, extent: &Range<u64>, reusable: bool) {
    let sections = file.reusable_free_sections().unwrap();
    let mut intersections = sections
        .sections()
        .iter()
        .filter_map(|section| {
            let start = extent.start.max(section.address().get());
            let end = extent.end.min(section.end_address().get());
            (start < end).then_some(start..end)
        })
        .collect::<Vec<_>>();
    if reusable {
        intersections.sort_unstable_by_key(|range| range.start);
        let mut next = extent.start;
        for range in intersections {
            assert_eq!(range.start, next, "{sections:?}");
            next = range.end;
        }
        assert_eq!(next, extent.end, "{sections:?}");
    } else {
        assert!(intersections.is_empty(), "{sections:?}");
    }
}

fn assert_native(path: &Path, removal: Removal, retained: Option<(&str, usize)>, replaced: bool) {
    let native = hdf5::File::open(path).unwrap();
    assert_eq!(
        native
            .dataset("survivor")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        survivor()
    );
    if let Some((path, count)) = retained {
        let dataset = native.dataset(path).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), source());
        assert_eq!(dataset.loc_info().unwrap().num_links, count);
        drop(dataset);
    }
    let mut expected = vec!["survivor"];
    if matches!(removal, Removal::ContainingGroup) {
        absence::assert_libhdf5_absent(&native.group("links").unwrap_err(), "links");
    } else {
        expected.push("links");
        let group = native.group("links").unwrap();
        absence::assert_libhdf5_absent(&group.dataset("a").unwrap_err(), "links/a");
        if retained.is_some_and(|(path, _)| path == "links/b") {
            assert_eq!(group.member_names().unwrap(), ["b"]);
        } else {
            absence::assert_libhdf5_absent(&group.dataset("b").unwrap_err(), "links/b");
            assert_eq!(group.member_names().unwrap(), Vec::<String>::new());
        }
        drop(group);
    }
    if retained.is_some_and(|(path, _)| path == "outside") {
        expected.push("outside");
    }
    if replaced {
        expected.push("replacement");
        assert_eq!(
            native
                .dataset("replacement")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            replacement()
        );
    }
    let mut names = native.member_names().unwrap();
    names.sort();
    expected.sort();
    assert_eq!(names, expected);
    native.close().unwrap();
}

fn source() -> Vec<i32> {
    (0..i32::try_from(ELEMENTS).unwrap()).collect()
}

fn replacement() -> Vec<i32> {
    (0..i32::try_from(ELEMENTS).unwrap())
        .map(|value| -value - 1)
        .collect()
}

fn survivor() -> Vec<i32> {
    vec![17; SURVIVOR_ELEMENTS]
}

#[derive(Clone, Copy, Debug)]
enum Removal {
    ContainingGroup,
    Sequential,
    SurvivingAlias,
    Together,
}

#[derive(Clone, Copy, Debug)]
enum Session {
    BoundedFsm,
    BoundedPage,
    MirroredFsm,
    MirroredPage,
}

impl Session {
    fn open(self, path: &Path) -> File {
        let memory = match self {
            Self::BoundedFsm | Self::BoundedPage => MemoryStrategy::Bounded,
            Self::MirroredFsm | Self::MirroredPage => MemoryStrategy::Mirrored,
        };
        let file = File::open_rw_with_options(
            path,
            FileAccessProperties::new()
                .with_memory_strategy(memory)
                .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
        )
        .unwrap();
        assert_eq!(
            file.edit_backing(),
            Some(match self {
                Self::BoundedFsm | Self::BoundedPage => EditBacking::Bounded,
                Self::MirroredFsm | Self::MirroredPage => EditBacking::Mirrored,
            })
        );
        let settings = file.file_space_settings().unwrap().unwrap();
        assert!(settings.persist());
        assert_eq!(settings.threshold(), 1);
        assert_eq!(
            settings.strategy(),
            if self.paged() {
                FileSpaceStrategy::Page
            } else {
                FileSpaceStrategy::FsmAggr
            }
        );
        file
    }

    fn paged(self) -> bool {
        matches!(self, Self::BoundedPage | Self::MirroredPage)
    }
}

const PAGE_BYTES: u64 = 4_096;
const PAYLOAD_BYTES: u64 = 1_048_576;
const ELEMENTS: usize = 262_144;
const SURVIVOR_BYTES: u64 = 65_536;
const SURVIVOR_ELEMENTS: usize = 16_384;
const GROWTH_ALLOWANCE: u64 = 65_536;
