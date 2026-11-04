use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;

use crate::FileSpaceStrategy;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[rstest]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::default(OffsetWidth::Eight, LengthWidth::Eight)]
fn fragmented_repeated_commits_preserve_native_values_and_accounting(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(false, true)] mixed: bool,
    #[values(4, 12)] holes: usize,
) {
    let fixture = super::run(FormatWidths { offsets, lengths }, strategy, mixed, holes);
    file_space_oracle::assert_persisted_sections(&fixture.path);
    file_space_oracle::assert_manager_metadata_bytes(&fixture.path);
    let native = file::libhdf5_open(&fixture.path);
    let sizes = native.create_plist().unwrap().get_sizes().unwrap();
    assert_eq!(usize::from(sizes.sizeof_addr), usize::from(offsets.get()));
    assert_eq!(usize::from(sizes.sizeof_size), usize::from(lengths.get()));
    let end = i32::try_from(8 * holes * 4).unwrap();
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        [0; 16].into_iter().chain(0..end).collect::<Vec<_>>()
    );
    assert_eq!(
        native
            .dataset("batch_7")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        (end - i32::try_from(holes * 4).unwrap()..end).collect::<Vec<_>>()
    );
    let mut root_names = native.member_names().unwrap();
    root_names.sort();
    assert_eq!(root_names, ["batch_7", "d", "g"]);
    let group = native.group("g").unwrap();
    let mut names = group.member_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        (0..holes)
            .map(|index| format!("keep_{index:02}"))
            .collect::<Vec<_>>()
    );
    for index in 0..i32::try_from(holes).unwrap() {
        assert_eq!(
            group
                .dataset(&format!("keep_{index:02}"))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [index; 16]
        );
    }
    drop(group);
    native.close().unwrap();
    fixture.directory.close().unwrap();
}
