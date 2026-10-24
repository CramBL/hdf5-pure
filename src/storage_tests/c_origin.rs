//! Ownership classifications for closed files from libhdf5 crosschecks.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::ops::Range;
use std::path::PathBuf;

use rstest::rstest;

use super::file_space;
use crate::{Error, File};

#[rstest]
fn ownership_of_libhdf5_and_mixed_writer_checkpoints(
    #[files("tests/data/c/ownership/*/")]
    #[dirs]
    directory: PathBuf,
) {
    let name = directory.file_name().unwrap().to_str().unwrap();
    let mut paths = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    paths.sort();
    assert!(!paths.is_empty(), "no checkpoints for {name}");
    let mut report = String::new();
    let temporary = tempfile::tempdir().unwrap();
    for source in paths {
        let path = temporary.path().join(source.file_name().unwrap());
        fs::copy(&source, &path).unwrap();
        let step = path.file_stem().unwrap().to_str().unwrap();
        let reader = File::open(&path).unwrap();
        let narrow = reader.superblock().offset_size.get() != 8;
        reader.close().unwrap();
        if narrow {
            let before = fs::read(&path).unwrap();
            let error = File::open_rw(&path).unwrap_err();
            let Error::EditUnsupported(reason) = error else {
                panic!("{name}/{step}: {error:?}");
            };
            assert_eq!(
                reason,
                "only 8-byte offsets and lengths are supported for in-place editing"
            );
            assert_eq!(fs::read(&path).unwrap(), before);
            writeln!(report, "{step}: unavailable={reason}").unwrap();
            continue;
        }
        super::with_reopened_file(&path, |file| {
            let owned =
                super::ownership(file).unwrap_or_else(|error| panic!("{name}/{step}: {error:?}"));
            owned.assert_classified(&owned.unclassified_space);
            assert_eq!(
                owned.reusable_free_space,
                file.space_accounting().unwrap().reusable_free_space,
                "{name}/{step}"
            );
            assert_eq!(
                owned.userblock_space,
                match file.superblock().base_address.get() {
                    0 => vec![],
                    base => vec![(0, base)],
                },
                "{name}/{step}"
            );
            let raw = raw_extents(file);
            assert_eq!(
                raw.iter().map(|range| range.end - range.start).sum::<u64>(),
                owned
                    .live_raw_space
                    .iter()
                    .map(|&(_, len)| len)
                    .sum::<u64>(),
                "{name}/{step}"
            );
            for range in raw {
                assert!(
                    owned
                        .live_raw_space
                        .iter()
                        .any(|&(at, len)| at <= range.start && range.end <= at + len),
                    "{name}/{step}: {range:?}"
                );
            }
            owned.write_checkpoint(&mut report, step);
        });
    }
    insta::assert_snapshot!(name, report);
}

#[test]
fn c_origin_page_preserves_nonzero_unclassified_storage() {
    let directory = tempfile::tempdir().unwrap();
    let path = test_util::copy_data(
        "c/ownership/page-persist-true-page-4096-threshold-1-base-0-c-origin-true/c-reuse.h5",
        directory.path(),
    );
    super::with_reopened_file(&path, |file| {
        let owned = super::ownership(file).unwrap();
        assert!(owned.analysis_complete);
        assert_eq!(
            owned
                .unclassified_space
                .iter()
                .map(|&(_, len)| len)
                .sum::<u64>(),
            288
        );
        assert!(!owned.free_space_manager_space.is_empty());
        owned.assert_classified(&owned.unclassified_space);
    });
}

#[test]
fn ownership_fixture_inventory_covers_each_committed_checkpoint() {
    let inventory = include_str!("../../tests/data/c/ownership.tsv");
    let expected = inventory
        .lines()
        .skip(1)
        .map(|row| {
            let columns = row.split('\t').collect::<Vec<_>>();
            assert_eq!(columns.len(), 4, "{row}");
            columns[3].to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(expected.len(), 524);
    let directory = test_util::data("c/ownership");
    let actual = fs::read_dir(&directory)
        .unwrap()
        .map(|case| case.unwrap().path())
        .filter(|case| case.is_dir())
        .flat_map(|case| {
            fs::read_dir(case)
                .unwrap()
                .map(|entry| {
                    entry
                        .unwrap()
                        .path()
                        .strip_prefix(&directory)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .replace('\\', "/")
                })
                .collect::<Vec<_>>()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
}

fn raw_extents(file: &File) -> Vec<Range<u64>> {
    let mut groups = vec![String::new()];
    let mut raw = Vec::new();
    while let Some(path) = groups.pop() {
        let group = if path.is_empty() {
            file.root()
        } else {
            file.group(&path).unwrap()
        };
        for name in group.datasets().unwrap() {
            raw.extend(file_space::raw_extents(file, &format!("{path}/{name}")));
        }
        groups.extend(
            group
                .groups()
                .unwrap()
                .iter()
                .map(|name| format!("{path}/{name}")),
        );
    }
    raw
}
