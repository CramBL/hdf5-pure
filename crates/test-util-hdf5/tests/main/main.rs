use hdf5_pure::SuperblockStatusFlags;
use test_util_hdf5::assertion::StatusOpen;

#[test]
fn a_source_read_message_reports_the_complete_snapshot_recovery() {
    assert_eq!(
        StatusOpen::SourceRead.message(SuperblockStatusFlags::write_marked()),
        "a byte source: the superblock marks the file as open for write (status flags 0x01), so a snapshot read is not safe. No SWMR writer holds it, so no reader can follow it as one. Pass FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot) to read it as it stands, which is a consistent snapshot once the writer has called File::sync or closed the file, or use File::from_bytes to read the bytes as they stand; clearing a flag a writer left behind with File::clear_status_flags needs a filesystem path."
    );
}
