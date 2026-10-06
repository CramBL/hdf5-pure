//! The File Space Info messages the writers write, with and without persisted free space.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hdf5_pure_core::__private::FileSpaceInfoFields;
pub use hdf5_pure_core::FileSpaceInfo;
pub use hdf5_pure_core::FileSpaceStrategy;
pub(crate) use hdf5_pure_format::__private::DEFAULT_PAGE_SIZE;
pub(crate) use hdf5_pure_format::__private::DEFAULT_THRESHOLD;
pub(crate) use hdf5_pure_format::__private::NUM_FILE_FSM_MANAGERS;

use crate::address::StoredAddress;
use crate::width::OffsetWidth;

/// The settings of the free-space managers of a strategy that has them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FreeSpaceSettings {
    /// Whether the managers persist free space across closes.
    pub(crate) persist: bool,
    /// The smallest free-space section the managers track, in bytes.
    pub(crate) threshold: u64,
}

impl FreeSpaceSettings {
    /// Returns `persist` and `threshold` as the settings of the free-space managers of `strategy`,
    /// or `None` if `strategy` has none.
    ///
    /// `H5F_FSPACE_STRATEGY_FSM_AGGR` and `H5F_FSPACE_STRATEGY_PAGE` have free-space managers,
    /// and `H5F_FSPACE_STRATEGY_AGGR` and `H5F_FSPACE_STRATEGY_NONE` have none, as defined for
    /// version 1 of "The File Space Info Message" in the [format specification, version 4.0][spec].
    /// `H5Pset_file_space_strategy` ignores `persist` and `threshold` for the two strategies
    /// without managers (`H5Pfcpl.c`, HDF5 1.14.6).
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo
    pub(crate) fn of(strategy: FileSpaceStrategy, persist: bool, threshold: u64) -> Option<Self> {
        match strategy {
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => {
                Some(Self { persist, threshold })
            }
            FileSpaceStrategy::Aggr | FileSpaceStrategy::None => None,
        }
    }
}

/// A non-persisting message recording `strategy`, `threshold` and `page_size`.
/// This is the form the writer emits (no free-space manager blocks). The
/// end-of-allocation address is undefined, all ones at `offsets`.
pub(crate) fn non_persistent(
    offsets: OffsetWidth,
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
        eoa_pre_fsm: StoredAddress::undefined(offsets.get()).get(),
        manager_addrs: Vec::new(),
    }
    .build()
}

/// Returns a persisting message with no free-space managers that records `eoa_pre_fsm` as the
/// end of allocation.
///
/// The persist flag is set and every manager address is undefined, all ones at `offsets`.
/// [`FileBuilder`](crate::FileBuilder) writes this form for a new
/// [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) file with `persist = true`, and a persisting
/// commit with no free space to record writes it with the end of allocation it publishes. libhdf5
/// reads `eoa_pre_fsm` into `eoa_fsm_fsalloc`, and an assertion-enabled build aborts in
/// `H5F__super_read` (`H5Fsuper.c`, HDF5 1.14.6) on a persisting file whose `eoa_pre_fsm` is
/// undefined (issue #178).
pub(crate) fn persistent_empty(
    offsets: OffsetWidth,
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
    eoa_pre_fsm: u64,
) -> FileSpaceInfo {
    let undefined = StoredAddress::undefined(offsets.get()).get();
    FileSpaceInfoFields {
        strategy,
        persist: true,
        threshold,
        page_size,
        page_end_meta_threshold: 0,
        eoa_pre_fsm,
        manager_addrs: vec![undefined; NUM_FILE_FSM_MANAGERS],
    }
    .build()
}

/// A persisting message whose first free-space manager is at `manager0_addr`
/// (the others undefined, all ones at `offsets`), recording `eoa_pre_fsm`, the
/// end-of-allocation before the on-disk free-space-manager blocks were
/// appended. This is the form [`File::open_rw`](crate::File::open_rw) writes
/// when it persists a non-empty free list: every tracked region lives in that
/// one manager.
pub(crate) fn persistent_single_manager(
    offsets: OffsetWidth,
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: u64,
    manager0_addr: u64,
    eoa_pre_fsm: u64,
) -> FileSpaceInfo {
    let mut manager_addrs =
        vec![StoredAddress::undefined(offsets.get()).get(); NUM_FILE_FSM_MANAGERS];
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
/// page type `k + 1` (or undefined when that page type tracks no free space).
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::address::BaseAddress;
    use crate::address::BaseAddressExt;
    use crate::file_writer::WIDTHS;

    #[test]
    fn non_persistent_roundtrip_29_bytes() {
        for strategy in [
            FileSpaceStrategy::FsmAggr,
            FileSpaceStrategy::Page,
            FileSpaceStrategy::Aggr,
            FileSpaceStrategy::None,
        ] {
            let info = non_persistent(OffsetWidth::Eight, strategy, 1, 4096);
            let bytes =
                hdf5_pure_format::__private::serialize_file_space_info(WIDTHS, &info).unwrap();
            assert_eq!(bytes.len(), 29, "non-persistent message is 29 bytes");
            let parsed = hdf5_pure_format::__private::parse_file_space_info(
                WIDTHS,
                BaseAddress::ZERO,
                0,
                &bytes,
            )
            .unwrap();
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
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // end of allocation, undefined
        ];
        let info = non_persistent(OffsetWidth::Eight, FileSpaceStrategy::None, 1, 4096);
        assert_eq!(
            hdf5_pure_format::__private::serialize_file_space_info(WIDTHS, &info).unwrap(),
            expected
        );
    }

    #[test]
    fn persistent_managers_roundtrips_multiple_slots() {
        // The paged writer's form: SUPER (slot 0), DRAW (slot 2), and the
        // generic-large manager (slot 6) defined, the rest undefined.
        let mut slots = [u64::MAX; NUM_FILE_FSM_MANAGERS];
        slots[0] = 841;
        slots[2] = 18384;
        slots[6] = 806;
        let info = persistent_managers(FileSpaceStrategy::Page, 0, 16384, slots, 65536);
        assert!(info.persist);
        assert_eq!(info.eoa_pre_fsm, 65536);
        let bytes = hdf5_pure_format::__private::serialize_file_space_info(WIDTHS, &info).unwrap();
        // 29-byte head + 12 * 8 manager slots.
        assert_eq!(bytes.len(), 29 + NUM_FILE_FSM_MANAGERS * 8);
        let parsed = hdf5_pure_format::__private::parse_file_space_info(
            WIDTHS,
            BaseAddress::ZERO,
            0,
            &bytes,
        )
        .unwrap();
        assert_eq!(parsed, info);
        assert_eq!(parsed.manager_addrs[0], 841);
        assert_eq!(parsed.manager_addrs[2], 18384);
        assert_eq!(parsed.manager_addrs[6], 806);
        assert_eq!(parsed.manager_addrs[1], u64::MAX);
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, 0xFFFF)]
    #[case::four(OffsetWidth::Four, 0xFFFF_FFFF)]
    #[case::eight(OffsetWidth::Eight, u64::MAX)]
    fn a_single_manager_message_leaves_its_other_slots_undefined_at_the_offset_width(
        #[case] offsets: OffsetWidth,
        #[case] undefined: u64,
    ) {
        let info = persistent_single_manager(
            offsets,
            FileSpaceStrategy::FsmAggr,
            DEFAULT_THRESHOLD,
            DEFAULT_PAGE_SIZE,
            0x0a48,
            0x0a48,
        );

        assert_eq!(
            info.manager_addrs,
            [vec![0x0a48], vec![undefined; NUM_FILE_FSM_MANAGERS - 1]].concat()
        );
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, 0xFFFF)]
    #[case::four(OffsetWidth::Four, 0xFFFF_FFFF)]
    #[case::eight(OffsetWidth::Eight, u64::MAX)]
    fn an_empty_message_records_its_managers_undefined_at_the_offset_width(
        #[case] offsets: OffsetWidth,
        #[case] undefined: u64,
    ) {
        let info = persistent_empty(
            offsets,
            FileSpaceStrategy::FsmAggr,
            DEFAULT_THRESHOLD,
            DEFAULT_PAGE_SIZE,
            0x0a48,
        );

        assert_eq!(
            (info.eoa_pre_fsm, info.manager_addrs),
            (0x0a48, vec![undefined; NUM_FILE_FSM_MANAGERS])
        );
    }
}
