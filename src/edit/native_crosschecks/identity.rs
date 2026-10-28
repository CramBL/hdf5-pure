//! Newest-first alias undo in files written by the reference library.

use rstest::rstest;
use test_util::identity::Fixture as IdentityFixture;
use test_util::temp;
use test_util_hdf5::identity;
use test_util_hdf5::native_fixtures;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::reader::ReadBackend;

use crate::DatatypeByteOrder;
use crate::edit;
use crate::edit::WRITE_GATHER_BYTES;
use crate::image::WriteBuffering;

#[rstest]
#[case::gathered(WriteBuffering::Operation { page_size: 4_096, max_bytes: WRITE_GATHER_BYTES })]
#[case::unbuffered(WriteBuffering::Unbuffered)]
fn native_alias_undo_replays_the_newest_write_first(
    #[case] buffering: WriteBuffering,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("native_hard_link_undo.h5");
    native_fixtures::hard_link_undo(&path, ByteOrder::LittleEndian);
    Fixture::HardLinkUndo.assert_native(&path, ByteOrder::LittleEndian);
    edit::test_util::assert_hard_link_undo(&path, buffering);
    let bytes = std::fs::read(&path).unwrap();
    let file = backend.open(&path).unwrap();
    identity::assert_contents(
        &file,
        IdentityFixture::HardLinkUndo,
        DatatypeByteOrder::LittleEndian,
    );
    identity::assert_storage(&file, &bytes, IdentityFixture::HardLinkUndo);
    drop(file);
    Fixture::HardLinkUndo.assert_native(&path, ByteOrder::LittleEndian);
}
