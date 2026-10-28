//! SWMR status flags across the writer's lifetime and recovery.

use hdf5_pure::{File, SuperblockStatusFlags};

use test_util::superblock;
use test_util_hdf5::dataset::Unlimited;

#[test]
fn swmr_flag_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.h5");
    {
        let data: Vec<i32> = (0..5).collect();
        Unlimited::new("d", &data, 1).pure_create(&path);
    }
    assert_eq!(
        superblock::consistency_flags(&path),
        SuperblockStatusFlags::new().bits(),
        "freshly created file is not SWMR-flagged"
    );
    {
        let w = File::open_swmr_writer(&path).unwrap();
        assert_eq!(
            superblock::consistency_flags(&path),
            SuperblockStatusFlags::swmr_write_marked().bits(),
            "flag set while writer is open"
        );
        w.dataset("d").unwrap().append(&[5, 6, 7]).unwrap();
        assert_eq!(
            superblock::consistency_flags(&path),
            SuperblockStatusFlags::swmr_write_marked().bits(),
            "flag stays set across appends"
        );
        w.close().unwrap();
    }
    assert_eq!(
        superblock::consistency_flags(&path),
        SuperblockStatusFlags::new().bits(),
        "flag cleared on clean close"
    );

    // Simulate a crashed writer (flag left set), then recover.
    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("d").unwrap().append(&[8]).unwrap();
        // Skipping Drop leaves the flag set, as a crash would.
        #[expect(clippy::mem_forget, reason = "the test models a writer that crashed")]
        std::mem::forget(w);
    }
    assert_eq!(
        superblock::consistency_flags(&path),
        SuperblockStatusFlags::swmr_write_marked().bits(),
        "flag left set after simulated crash"
    );
    File::clear_status_flags(&path).unwrap();
    assert_eq!(
        superblock::consistency_flags(&path),
        SuperblockStatusFlags::new().bits(),
        "clear_status_flags recovers the file"
    );
}
