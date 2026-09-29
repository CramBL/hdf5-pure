//! The File Space Info messages the writers write, with and without persisted free space.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hdf5_pure_core::__private::FileSpaceInfoFields;
pub use hdf5_pure_core::FileSpaceInfo;
pub use hdf5_pure_core::FileSpaceStrategy;

/// A non-persisting message recording `strategy`, `threshold` and `page_size`.
/// This is the form the writer emits (no free-space manager blocks).
pub(crate) fn non_persistent(
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
) -> FileSpaceInfo {
    FileSpaceInfoFields {
        strategy,
        persist: false,
        threshold,
        page_size,
        page_end_meta_threshold: 0,
        eoa_pre_fsm: UNDEF,
        manager_addrs: Vec::new(),
    }
    .build()
}

/// A persisting message for a file with no free space yet (the form
/// [`FileBuilder`](crate::FileBuilder) emits for `persist = true`): the
/// persist flag is set and every manager slot is undefined because no FSM
/// space has been allocated. `eoa_pre_fsm` is left [`UNDEF`] here as a
/// placeholder. The writer overwrites it with the real end-of-allocation once
/// the layout is known, because libhdf5 requires a persisting file to record a
/// defined `eoa_fsm_fsalloc` (an assertion-enabled build aborts on the
/// undefined address, issue #178).
pub(crate) fn persistent_empty(
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
) -> FileSpaceInfo {
    FileSpaceInfoFields {
        strategy,
        persist: true,
        threshold,
        page_size,
        page_end_meta_threshold: 0,
        eoa_pre_fsm: UNDEF,
        manager_addrs: vec![UNDEF; NUM_FILE_FSM_MANAGERS],
    }
    .build()
}

/// A persisting message whose first free-space manager is at `manager0_addr`
/// (the others undefined), recording `eoa_pre_fsm`, the end-of-allocation
/// before the on-disk free-space-manager blocks were appended. This is the
/// form [`File::open_rw`](crate::File::open_rw) writes when it persists a non-empty
/// free list: every tracked region lives in that one manager.
pub(crate) fn persistent_single_manager(
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
    manager0_addr: u64,
    eoa_pre_fsm: u64,
) -> FileSpaceInfo {
    let mut manager_addrs = vec![UNDEF; NUM_FILE_FSM_MANAGERS];
    manager_addrs[0] = manager0_addr;
    FileSpaceInfoFields {
        strategy,
        persist: true,
        threshold,
        page_size,
        page_end_meta_threshold: 0,
        eoa_pre_fsm,
        manager_addrs,
    }
    .build()
}

/// A persisting message for a paged file whose free space is tracked by
/// per-page-type managers. `slots[k]` is the `FSHD` address of the manager for
/// page type `k + 1` (or [`UNDEF`] when that page type tracks no free space).
/// `eoa_pre_fsm` is the page-aligned end-of-allocation. This is the form the
/// paged writer emits: metadata free space lives in the SUPER manager
/// (`slots[0]`), small raw data in DRAW (`slots[2]`), and the trailing
/// fragments of large multi-page allocations in the generic-large manager
/// (`slots[6]`).
pub(crate) fn persistent_managers(
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
    slots: [u64; NUM_FILE_FSM_MANAGERS],
    eoa_pre_fsm: u64,
) -> FileSpaceInfo {
    FileSpaceInfoFields {
        strategy,
        persist: true,
        threshold,
        page_size,
        page_end_meta_threshold: 0,
        eoa_pre_fsm,
        manager_addrs: slots.to_vec(),
    }
    .build()
}

/// An undefined on-disk address (all bits set), HDF5's "no address" sentinel.
const UNDEF: u64 = u64::MAX;

/// The default free-space section threshold the C library uses.
pub(crate) const DEFAULT_THRESHOLD: u64 = 1;
/// The default file-space page size the C library uses.
pub(crate) const DEFAULT_PAGE_SIZE: u64 = 4096;
/// Number of free-space-manager address slots a persisting message carries (one
/// per file memory type); the reference C library writes twelve.
pub(crate) const NUM_FILE_FSM_MANAGERS: usize = 12;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_persistent_roundtrip_29_bytes() {
        for strategy in [
            FileSpaceStrategy::FsmAggr,
            FileSpaceStrategy::Page,
            FileSpaceStrategy::Aggr,
            FileSpaceStrategy::None,
        ] {
            let info = non_persistent(strategy, 1, 4096);
            let bytes = hdf5_pure_format::__private::serialize_file_space_info(&info);
            assert_eq!(bytes.len(), 29, "non-persistent message is 29 bytes");
            let parsed = hdf5_pure_format::__private::parse_file_space_info(&bytes, 8, 8).unwrap();
            assert_eq!(parsed, info);
            assert_eq!(parsed.eoa_pre_fsm, u64::MAX);
            assert!(parsed.manager_addrs.is_empty());
        }
    }

    #[test]
    fn matches_c_library_none_bytes() {
        // Exact bytes the reference C library (HDF5 1.14.6) wrote for strategy
        // NONE, captured via tmp/probe_fsinfo.py.
        let expected = [
            0x01u8, 0x03, 0x00, // version=1, strategy=NONE(3), persist=0
            0x01, 0, 0, 0, 0, 0, 0, 0, // threshold=1
            0x00, 0x10, 0, 0, 0, 0, 0, 0, // page_size=4096
            0x00, 0x00, // page end meta threshold
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // eoa = UNDEF
        ];
        let info = non_persistent(FileSpaceStrategy::None, 1, 4096);
        assert_eq!(
            hdf5_pure_format::__private::serialize_file_space_info(&info),
            expected
        );
    }

    #[test]
    fn persistent_managers_roundtrips_multiple_slots() {
        // The paged writer's form: SUPER (slot 0), DRAW (slot 2), and the
        // generic-large manager (slot 6) defined, the rest undefined.
        let mut slots = [UNDEF; NUM_FILE_FSM_MANAGERS];
        slots[0] = 841;
        slots[2] = 18384;
        slots[6] = 806;
        let info = persistent_managers(FileSpaceStrategy::Page, 0, 16384, slots, 65536);
        assert!(info.persist);
        assert_eq!(info.eoa_pre_fsm, 65536);
        let bytes = hdf5_pure_format::__private::serialize_file_space_info(&info);
        // 29-byte head + 12 * 8 manager slots.
        assert_eq!(bytes.len(), 29 + NUM_FILE_FSM_MANAGERS * 8);
        let parsed = hdf5_pure_format::__private::parse_file_space_info(&bytes, 8, 8).unwrap();
        assert_eq!(parsed, info);
        assert_eq!(parsed.manager_addrs[0], 841);
        assert_eq!(parsed.manager_addrs[2], 18384);
        assert_eq!(parsed.manager_addrs[6], 806);
        assert_eq!(parsed.manager_addrs[1], UNDEF);
    }
}
