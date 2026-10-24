use std::fs;
use std::io::Write as _;

use rstest::rstest;

use crate::{File, FileBuilder, FileSpaceStrategy, UserblockSize};

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn ownership_distinguishes_stored_eof_from_trailing_bytes(
    #[case] strategy: FileSpaceStrategy,
    #[values(0, 4096)] base: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trailing.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(strategy, true, 1);
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    for (name, value, len) in [("a", 1, 100), ("b", 2, 1024), ("c", 3, 100)] {
        builder
            .create_dataset(name)
            .with_i32_data(&vec![value; len]);
    }
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let logical = fs::metadata(&path).unwrap().len();
    let slack = [0xA5; 257];
    {
        let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&slack).unwrap();
        file.sync_all().unwrap();
    }
    super::with_reopened_file(&path, |file| {
        let ownership = super::ownership(file).unwrap();
        assert!(ownership.analysis_complete);
        assert_eq!(ownership.logical_size, logical);
        assert_eq!(ownership.physical_size, logical + 257);
        assert_eq!(ownership.trailing_slack_space, vec![(logical, 257)]);
    });
}
