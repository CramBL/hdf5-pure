use std::path::Path;

use hdf5_pure::{EditBacking, Error, File, FileBuilder, MemoryStrategy, SyncPolicy, UserblockSize};
use rstest::rstest;
use test_util_hdf5::session;

#[rstest]
#[case::bounded(
    session::open_bounded,
    Some(MemoryStrategy::Bounded),
    None,
    EditBacking::Bounded
)]
#[case::automatic_bounded(session::open_rw, None, None, EditBacking::Bounded)]
#[case::automatic_mirrored(session::open_rw, None, Some(4096), EditBacking::Mirrored)]
fn editing_helpers_select_on_close_and_preserve_memory_strategy(
    #[case] open: fn(&Path) -> Result<File, Error>,
    #[case] strategy: Option<MemoryStrategy>,
    #[case] userblock: Option<u64>,
    #[case] backing: EditBacking,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.h5");
    let mut builder = FileBuilder::new();
    if let Some(size) = userblock {
        builder.with_userblock(UserblockSize::try_from(size).unwrap());
    }
    builder.write(&path).unwrap();

    let file = open(&path).unwrap();
    assert_eq!(file.access_properties().sync_policy(), SyncPolicy::OnClose);
    assert_eq!(file.access_properties().memory_strategy(), strategy);
    assert_eq!(file.edit_backing(), Some(backing));
    file.root()
        .create_dataset("values", |dataset| {
            dataset.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
}
