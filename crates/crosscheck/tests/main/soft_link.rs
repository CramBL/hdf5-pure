#![cfg(feature = "hdf5")]

use std::path::Path;

use hdf5::{MajorErrorCode, MinorErrorCode};
use hdf5_pure::{Error, File, FormatError};
use rstest::rstest;
use test_util_hdf5::soft_link::GroupFormat;

#[rstest]
#[case::absolute("links/absolute", 42)]
#[case::relative("links/relative", 7)]
#[case::chained("links/chain", 7)]
#[case::group_suffix("links/group/value", 42)]
#[case::current_group("paths/current/value", 11)]
#[case::root_group("paths/root/value", 99)]
#[case::two_dots("paths/two_dots", 42)]
#[case::same_link_repeated("paths/current/current/value", 11)]
fn dataset_lookups_follow_soft_link_paths(
    #[values(GroupFormat::Compact, GroupFormat::Dense, GroupFormat::SymbolTable)]
    format: GroupFormat,
    #[values(Backend::Buffered, Backend::Streaming, Backend::Edit)] backend: Backend,
    #[case] spelling: &str,
    #[case] expected: i32,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft.h5");
    format.generate(&path);
    let reference = hdf5::File::open(&path).unwrap();
    assert_eq!(
        reference
            .dataset(spelling)
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [expected]
    );
    drop(reference);
    let file = backend.open(&path);
    assert_eq!(
        file.dataset(spelling).unwrap().read_i32().unwrap(),
        [expected]
    );
    assert_eq!(
        file.root().dataset(spelling).unwrap().read_i32().unwrap(),
        [expected]
    );
    let (parent, leaf) = spelling.rsplit_once('/').unwrap();
    assert_eq!(
        file.group(parent)
            .unwrap()
            .dataset(leaf)
            .unwrap()
            .read_i32()
            .unwrap(),
        [expected]
    );
    assert_eq!(
        file.group("g")
            .unwrap()
            .dataset(&format!("/{spelling}"))
            .unwrap()
            .read_i32()
            .unwrap(),
        [expected]
    );
}

#[rstest]
#[case::absolute("links/group", 42)]
#[case::current("paths/current", 11)]
#[case::root("paths/root", 99)]
#[case::chained("paths/current/current", 11)]
fn group_lookups_follow_soft_link_paths(
    #[values(GroupFormat::Compact, GroupFormat::Dense, GroupFormat::SymbolTable)]
    format: GroupFormat,
    #[values(Backend::Buffered, Backend::Streaming, Backend::Edit)] backend: Backend,
    #[case] spelling: &str,
    #[case] expected: i32,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft.h5");
    format.generate(&path);
    let reference = hdf5::File::open(&path).unwrap();
    assert_eq!(
        reference
            .group(spelling)
            .unwrap()
            .dataset("value")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [expected]
    );
    drop(reference);
    let file = backend.open(&path);
    assert_eq!(
        file.group(spelling)
            .unwrap()
            .dataset("value")
            .unwrap()
            .read_i32()
            .unwrap(),
        [expected]
    );
    assert_eq!(
        file.root()
            .group(spelling)
            .unwrap()
            .dataset("value")
            .unwrap()
            .read_i32()
            .unwrap(),
        [expected]
    );
    let (parent, leaf) = spelling.rsplit_once('/').unwrap();
    assert_eq!(
        file.group(parent)
            .unwrap()
            .group(leaf)
            .unwrap()
            .dataset("value")
            .unwrap()
            .read_i32()
            .unwrap(),
        [expected]
    );
    assert_eq!(
        file.group("g")
            .unwrap()
            .group(&format!("/{spelling}"))
            .unwrap()
            .dataset("value")
            .unwrap()
            .read_i32()
            .unwrap(),
        [expected]
    );
}

#[rstest]
fn hard_link_listings_are_preserved(
    #[values(GroupFormat::Compact, GroupFormat::Dense, GroupFormat::SymbolTable)]
    format: GroupFormat,
    #[values(Backend::Buffered, Backend::Streaming, Backend::Edit)] backend: Backend,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft.h5");
    format.generate(&path);
    let file = backend.open(&path);
    assert_eq!(file.root().datasets().unwrap(), ["value"]);
    let mut groups = file.root().groups().unwrap();
    groups.sort();
    assert_eq!(groups, ["cycles", "g", "links", "paths"]);
    assert_eq!(file.group("links").unwrap().datasets().unwrap(), ["value"]);
    assert_eq!(
        file.group("links").unwrap().groups().unwrap(),
        Vec::<String>::new()
    );
}

#[rstest]
#[case::dangling("links/dangling", FormatError::DanglingSoftLink { path: "links/dangling".into(), target: "missing".into() })]
#[case::absent("links/absent", FormatError::PathNotFound("links/absent".into()))]
#[case::missing_suffix("links/group/absent", FormatError::PathNotFound("links/group/absent".into()))]
#[case::self_cycle("links/self", FormatError::SoftLinkTraversalLimit { path: "links/self".into(), limit: 64 })]
#[case::two_link_cycle("cycles/a", FormatError::SoftLinkTraversalLimit { path: "cycles/a".into(), limit: 64 })]
#[case::expanding_cycle("cycles/growing", FormatError::SoftLinkTraversalLimit { path: "cycles/growing".into(), limit: 64 })]
fn lookup_errors_distinguish_missing_targets_and_cycles(
    #[values(GroupFormat::Compact, GroupFormat::Dense, GroupFormat::SymbolTable)]
    format: GroupFormat,
    #[values(Backend::Buffered, Backend::Streaming, Backend::Edit)] backend: Backend,
    #[case] spelling: &str,
    #[case] expected: FormatError,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft.h5");
    format.generate(&path);
    let file = backend.open(&path);
    let error = file.dataset(spelling).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(error, expected);
    let error = file.group(spelling).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(error, expected);
    let error = file.root().dataset(spelling).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(error, expected);
    let error = file.root().group(spelling).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(error, expected);
    let (parent, leaf) = spelling.rsplit_once('/').unwrap();
    let group = file.group(parent).unwrap();
    let error = group.dataset(leaf).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(error, expected);
    let error = group.group(leaf).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::dangling("links/dangling", MinorErrorCode::NotFound)]
#[case::self_cycle("links/self", MinorErrorCode::NLinks)]
#[case::two_link_cycle("cycles/a", MinorErrorCode::NLinks)]
#[case::expanding_cycle("cycles/growing", MinorErrorCode::NLinks)]
fn reference_lookup_reports_missing_targets_and_cycles(
    #[values(GroupFormat::Compact, GroupFormat::Dense, GroupFormat::SymbolTable)]
    format: GroupFormat,
    #[case] spelling: &str,
    #[case] minor: MinorErrorCode,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft.h5");
    format.generate(&path);
    let file = hdf5::File::open(&path).unwrap();
    let error = file.dataset(spelling).unwrap_err();
    assert!(
        error.contains_major(MajorErrorCode::SymbolTable),
        "{error:?}"
    );
    assert!(error.contains_minor(minor), "{error:?}");
}

#[rstest]
fn external_links_are_distinct_from_missing_names(
    #[values(GroupFormat::Compact, GroupFormat::Dense)] format: GroupFormat,
    #[values(Backend::Buffered, Backend::Streaming, Backend::Edit)] backend: Backend,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft.h5");
    format.generate(&path);
    let file = backend.open(&path);
    let error = file.dataset("links/external").unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::ExternalLinkUnsupported {
            path: "links/external".into(),
            filename: "absent.h5".into(),
            object_path: "/value".into(),
        }
    );
    let error = file.group("links").unwrap().group("external").unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::ExternalLinkUnsupported {
            path: "links/external".into(),
            filename: "absent.h5".into(),
            object_path: "/value".into(),
        }
    );
}

#[derive(Clone, Copy, Debug)]
enum Backend {
    Buffered,
    Edit,
    Streaming,
}

impl Backend {
    fn open(self, path: &Path) -> File {
        match self {
            Self::Buffered => File::open(path).unwrap(),
            Self::Edit => File::open_rw(path).unwrap(),
            Self::Streaming => File::open_streaming(path).unwrap(),
        }
    }
}
