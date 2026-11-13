use hdf5_pure::{EditBacking, File};
use rstest::rstest;
use test_util_hdf5::global_heap_fixture::members::{Buffering, LayoutFixture, Terminal};

#[rstest]
#[case::explicit(Terminal::Explicit)]
#[case::full(Terminal::Full)]
#[case::implied(Terminal::Implied)]
fn grouped_member_removals_preserve_packed_records_and_parent_capacity(
    #[case] terminal: Terminal,
    #[values(1, 3)] parents: usize,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
) {
    let path = test_util::temp::temp_path("grouped-public-heap-members.h5");
    let fixture = LayoutFixture::write(&path, terminal, parents);
    let file = File::open_rw_with_options(&path, buffering.properties(backing)).unwrap();
    assert_eq!(file.edit_backing(), Some(backing));
    fixture.remove(&file);
    file.commit().unwrap();
    file.close().unwrap();
    fixture.assert_compacted(&path);
}
