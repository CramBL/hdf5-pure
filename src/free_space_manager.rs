//! Adapts persistent free-space managers between storage, semantic manager sets, and the wire
//! codec.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_format::__private::SECTION_CLASS_LARGE;
use hdf5_pure_format::__private::SECTION_CLASS_SIMPLE;
use hdf5_pure_format::__private::SECTION_CLASS_SMALL;
use hdf5_pure_space::__private::{ManagerKind, ManagerSections};

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::width::FormatWidths;

/// Reads the free sections of the managers at `manager_addrs` from the file `data`.
///
/// Adds `base`, the file's base address, to every address the file stores. Skips an undefined
/// manager address, and a manager whose section list address is undefined.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if `base` and a stored address sum past `u64::MAX`,
/// [`FormatError::ValueTooLargeForPlatform`] if a position in the file or the length of a section
/// list does not fit a `usize`, [`FormatError::InvalidFreeSpaceManager`] if a header or any part
/// of a section list's allocation is past the end of `data`, and the errors of
/// [`FreeSpaceManagerHeader::parse`] and
/// [`parse_section_info`](hdf5_pure_format::__private::parse_section_info) for each block it reads.
pub(crate) fn read_persisted_sections(
    data: &[u8],
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[DecodedStoredAddress],
) -> Result<Vec<FreeSection>, FormatError> {
    let mut sections = Vec::new();
    for &addr in manager_addrs {
        let DecodedStoredAddress::Defined(addr) = addr else {
            continue;
        };
        let a = base.absolute(addr)?.to_usize()?;
        let header = FreeSpaceManagerHeader::parse(
            widths,
            data.get(a..).ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(format!(
                    "the header at {a} is past the end of the file, {} bytes long",
                    data.len()
                ))
            })?,
        )?;
        if header.section_list_addr().is_undefined(widths.offsets) {
            continue;
        }
        let fa = base.absolute(header.section_list_addr())?.to_usize()?;
        let used = header.section_list_used().to_usize()?;
        let allocated = header.section_list_allocated().to_usize()?;
        fa.checked_add(allocated)
            .filter(|&end| end <= data.len())
            .ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(format!(
                    "the section list allocation at {fa}, {allocated} bytes long, runs past the \
                     end of the file, {} bytes long",
                    data.len()
                ))
            })?;
        let block = fa
            .checked_add(used)
            .and_then(|end| data.get(fa..end))
            .ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(format!(
                    "the section list at {fa}, {used} bytes long, runs past the end of the file, \
                     {} bytes long",
                    data.len()
                ))
            })?;
        sections.extend(hdf5_pure_format::__private::parse_section_info(
            widths, block, addr, &header,
        )?);
    }
    Ok(sections)
}

/// The recovered free sections and the complete physical `(addr, len)` allocations of their
/// `FSHD` and `FSSE` blocks.
pub(crate) type PersistedSections = (Vec<FreeSection>, Vec<(u64, u64)>);

/// Recovers free sections and complete manager allocations through a random-access
/// [`Source`](crate::Source).
///
/// Adds `base` to stored addresses and skips undefined manager or section list addresses.
/// Reads each section list's used bytes and records its allocated length for reclamation.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if an address conversion or allocation end overflows,
/// [`FormatError::InvalidFreeSpaceManager`] if a section list allocation exceeds the source
/// length, and [`FormatError::ValueTooLargeForPlatform`] if a required read length exceeds
/// [`usize::MAX`]. Returns source read errors and the errors of [`FreeSpaceManagerHeader::parse`]
/// and [`parse_section_info`](hdf5_pure_format::__private::parse_section_info) for each block.
pub(crate) fn read_persisted_sections_source<S: crate::source::Source>(
    src: &S,
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[DecodedStoredAddress],
) -> Result<PersistedSections, FormatError> {
    let mut sections = Vec::new();
    let mut blocks = Vec::new();
    let hdr_len = hdf5_pure_format::__private::free_space_manager_header_len(widths);
    for &addr in manager_addrs {
        let DecodedStoredAddress::Defined(addr) = addr else {
            continue;
        };
        let a = base.absolute(addr)?;
        let fshd = src.read_exact_at(a, hdr_len.to_usize()?)?;
        let header = FreeSpaceManagerHeader::parse(widths, &fshd)?;
        blocks.push((a, hdr_len));
        if header.section_list_addr().is_undefined(widths.offsets) {
            continue;
        }
        let fa = base.absolute(header.section_list_addr())?;
        let used = header.section_list_used();
        let allocated = header.section_list_allocated();
        let end = fa
            .checked_add(allocated)
            .ok_or(FormatError::OffsetOverflow {
                offset: fa,
                length: allocated,
            })?;
        if end > src.len() {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the section list allocation at {fa}, {allocated} bytes long, runs past the end of \
                 the file, {} bytes long",
                src.len()
            )));
        }
        let block = src.read_exact_at(fa, used.to_usize()?)?;
        sections.extend(hdf5_pure_format::__private::parse_section_info(
            widths, &block, addr, &header,
        )?);
        blocks.push((fa, allocated));
    }
    Ok((sections, blocks))
}

/// Returns the semantic kind represented by File Space Info manager `slot` for `strategy`.
pub(crate) fn manager_kind(strategy: FileSpaceStrategy, slot: usize) -> Option<ManagerKind> {
    match strategy {
        FileSpaceStrategy::FsmAggr => Some(ManagerKind::Flat),
        FileSpaceStrategy::Page => match slot {
            0 => Some(ManagerKind::PagedSmallMetadata),
            2 => Some(ManagerKind::PagedSmallRaw),
            6 => Some(ManagerKind::PagedLargeGeneric),
            _ => None,
        },
        FileSpaceStrategy::Aggr | FileSpaceStrategy::None => None,
    }
}

/// Rounds `value` up to the next multiple of `page`.
pub(crate) fn align_up(value: u64, page: FileSpacePageSize) -> u64 {
    value.div_ceil(page.get()) * page.get()
}

/// Places one semantic manager in a persistent tail and records its wire representation.
pub(crate) struct ManagerBlock {
    pub(crate) kind: ManagerKind,
    pub(crate) fshd_addr: StoredAddress,
    pub(crate) fsse_addr: StoredAddress,
    pub(crate) class: u8,
    pub(crate) sections: Vec<FreeSection>,
}

/// Holds the wire layout of a semantic manager set.
pub(crate) struct ManagerPlan {
    /// Manager address per File Space Info slot, the undefined address at the file's offset width
    /// where inactive.
    pub(crate) slots: [StoredAddress; NUM_FILE_FSM_MANAGERS],
    /// The active managers in ascending File Space Info slot order.
    pub(crate) blocks: Vec<ManagerBlock>,
    /// The first address past the last manager block, or `start` when none are active.
    pub(crate) end_of_managers: StoredAddress,
}

impl ManagerPlan {
    /// Returns `true` when there is no free space to record.
    pub(crate) fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Places `managers` contiguously from `start` using their HDF5 wire slots and section classes.
///
/// The semantic manager set decides which extents belong together. This adapter alone translates
/// those kinds to File Space Info slots, `FSSE` section classes, stored addresses, and encoded
/// lengths.
pub(crate) fn plan_managers(
    managers: &[ManagerSections],
    start: StoredAddress,
    widths: FormatWidths,
) -> ManagerPlan {
    let mut managers: Vec<_> = managers
        .iter()
        .map(|manager| {
            let (slot, class) = manager_wire(manager.kind());
            (slot, class, manager)
        })
        .collect();
    managers.sort_unstable_by_key(|&(slot, _, _)| slot);
    debug_assert!(
        managers.windows(2).all(|w| w[0].0 != w[1].0),
        "two semantic free-space managers map to the same File Space Info slot"
    );

    let mut slots = [StoredAddress::undefined(widths.offsets); NUM_FILE_FSM_MANAGERS];
    let mut blocks = Vec::with_capacity(managers.len());
    let mut cursor = start;
    for (slot, class, manager) in managers {
        let fshd_addr = cursor;
        let fsse_addr = fshd_addr.offset(
            hdf5_pure_format::__private::free_space_manager_header_len(widths),
        );
        let sections: Vec<_> = manager
            .sections()
            .iter()
            .map(|extent| FreeSection {
                addr: StoredAddress::new(extent.start()),
                size: extent.len(),
            })
            .collect();
        let section_sizes: Vec<_> = sections.iter().map(|section| section.size).collect();
        cursor = fsse_addr.offset(hdf5_pure_format::__private::section_info_len(
            widths,
            &section_sizes,
        ));
        slots[slot] = fshd_addr;
        blocks.push(ManagerBlock {
            kind: manager.kind(),
            fshd_addr,
            fsse_addr,
            class,
            sections,
        });
    }
    ManagerPlan {
        slots,
        blocks,
        end_of_managers: cursor,
    }
}

/// Returns the encoded length of `managers` without assigning them file positions.
pub(crate) fn manager_blocks_len(managers: &[ManagerSections], widths: FormatWidths) -> u64 {
    plan_managers(managers, StoredAddress::new(0), widths)
        .end_of_managers
        .get()
}

fn manager_wire(kind: ManagerKind) -> (usize, u8) {
    match kind {
        ManagerKind::Flat => (0, SECTION_CLASS_SIMPLE),
        ManagerKind::PagedLargeGeneric => (6, SECTION_CLASS_LARGE),
        ManagerKind::PagedSmallMetadata => (0, SECTION_CLASS_SMALL),
        ManagerKind::PagedSmallRaw => (2, SECTION_CLASS_SMALL),
    }
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;

    use hdf5_pure_space::__private::{Extent, SessionSpace};
    use rstest::rstest;
    use test_util::free_space;

    use crate::source::Source;
    use crate::width::OffsetWidth;

    use super::*;

    /// A free section at `addr` spanning `size` bytes.
    fn section(addr: u64, size: u64) -> FreeSection {
        FreeSection {
            addr: StoredAddress::new(addr),
            size,
        }
    }

    fn widths(offset_size: u8, length_size: u8) -> FormatWidths {
        FormatWidths::from_sizes(offset_size, length_size).unwrap()
    }

    #[rstest]
    #[case::header(
        736,
        1000,
        "the header at 1000 is past the end of the file, 736 bytes long"
    )]
    #[case::section_list(
        735,
        619,
        "the section list allocation at 701, 35 bytes long, runs past the end of the file, 735 bytes long"
    )]
    fn a_manager_past_the_end_of_the_file_fails_to_read(
        #[case] file_len: usize,
        #[case] manager_addr: u64,
        #[case] reason: &str,
    ) {
        // The header is at 619 and its 35-byte section list at 701, so the file ends at 736.
        let fshd = free_space::single_section_header();
        let fsse = free_space::single_section_info();
        let mut buf = vec![0u8; 736];
        buf[619..619 + fshd.len()].copy_from_slice(&fshd);
        buf[701..701 + fsse.len()].copy_from_slice(&fsse);
        buf.truncate(file_len);

        assert_eq!(
            read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(
                    manager_addr
                ))],
            ),
            Err(FormatError::InvalidFreeSpaceManager(reason.to_owned()))
        );
    }

    #[rstest]
    #[case::tight(0, 35)]
    #[case::spare_capacity(0, 51)]
    #[case::userblock(512, 51)]
    #[case::large_capacity(0, 1 << 33)]
    fn source_recovery_reads_used_bytes_and_records_allocated_bytes(
        #[case] base: u64,
        #[case] allocated: u64,
    ) {
        let source = manager_source(base, allocated, base + 701 + allocated);
        let (sections, blocks) = read_persisted_sections_source(
            &source,
            widths(8, 8),
            BaseAddress::new(base),
            &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
        )
        .unwrap();

        assert_eq!(sections, vec![section(2848, 1600)]);
        assert_eq!(blocks, vec![(base + 619, 82), (base + 701, allocated)]);
        assert_eq!(
            *source.reads.borrow(),
            vec![(base + 619, 82), (base + 701, 35)]
        );
    }

    #[rstest]
    #[case::past_source(
        0,
        51,
        751,
        FormatError::InvalidFreeSpaceManager(
            "the section list allocation at 701, 51 bytes long, runs past the end of the file, 751 bytes long".to_owned()
        )
    )]
    #[case::allocation_overflow(
        u64::MAX - 1000,
        300,
        u64::MAX,
        FormatError::OffsetOverflow { offset: u64::MAX - 299, length: 300 }
    )]
    fn source_recovery_rejects_invalid_allocation_geometry(
        #[case] base: u64,
        #[case] allocated: u64,
        #[case] file_len: u64,
        #[case] expected: FormatError,
    ) {
        let source = manager_source(base, allocated, file_len);
        assert_eq!(
            read_persisted_sections_source(
                &source,
                widths(8, 8),
                BaseAddress::new(base),
                &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(*source.reads.borrow(), vec![(base + 619, 82)]);
    }

    #[test]
    fn read_persisted_sections_follows_managers() {
        // Place the single-section FSHD@619 + FSSE@701 fixtures in a buffer at
        // their real offsets and read them through the manager-address indirection.
        let fshd = free_space::single_section_header();
        let fsse = free_space::single_section_info();
        let mut buf = vec![0u8; 701 + fsse.len()];
        buf[619..619 + fshd.len()].copy_from_slice(&fshd);
        buf[701..701 + fsse.len()].copy_from_slice(&fsse);

        let got = read_persisted_sections(
            &buf,
            widths(8, 8),
            BaseAddress::ZERO,
            &[
                DecodedStoredAddress::Defined(StoredAddress::new(619)),
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
            ],
        )
        .unwrap();
        assert_eq!(got, vec![section(2848, 1600)]);
        // No defined managers -> no sections.
        assert!(
            read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Undefined]
            )
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn an_undefined_manager_slot_is_skipped_in_a_four_byte_offset_file() {
        assert!(
            read_persisted_sections(
                &[],
                widths(4, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Undefined]
            )
            .unwrap()
            .is_empty(),
            "an all-0xFF manager address is the sentinel for an unused slot"
        );
    }

    #[test]
    fn a_manager_with_no_section_info_is_skipped_in_a_four_byte_offset_file() {
        let (fshd, _fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths(4, 8),
            &[],
            StoredAddress::new(0),
            StoredAddress::new(0xFFFF_FFFF),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();

        assert!(
            read_persisted_sections(
                &fshd,
                widths(4, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(0))]
            )
            .unwrap()
            .is_empty(),
            "a manager tracking nothing records no section info"
        );

        let (sections, blocks) = read_persisted_sections_source(
            &crate::source::BytesSource::new(&fshd),
            widths(4, 8),
            BaseAddress::ZERO,
            &[DecodedStoredAddress::Defined(StoredAddress::new(0))],
        )
        .unwrap();
        assert!(sections.is_empty());
        assert_eq!(
            blocks,
            vec![(
                0,
                hdf5_pure_format::__private::free_space_manager_header_len(widths(4, 8))
            )],
            "only the header block was read"
        );
    }

    #[test]
    fn serialize_roundtrips_through_parse() {
        let sections = [
            section(4096, 512),
            section(9000, 512),
            section(20000, 70000),
        ];
        let (fshd, _fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths(8, 8),
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_sections(), 3);
        assert_eq!(header.total_space(), 512 + 512 + 70000);
        assert_eq!(header.section_list_addr(), StoredAddress::new(1100));

        // Place both blocks in a buffer and read them back through the manager
        // indirection; the recovered sections match (order-independent).
        let (fshd, fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths(8, 8),
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let mut buf = vec![0u8; 1100 + fsse.len()];
        buf[1000..1000 + fshd.len()].copy_from_slice(&fshd);
        buf[1100..1100 + fsse.len()].copy_from_slice(&fsse);
        let mut got = read_persisted_sections(
            &buf,
            widths(8, 8),
            BaseAddress::ZERO,
            &[DecodedStoredAddress::Defined(StoredAddress::new(1000))],
        )
        .unwrap();
        got.sort_by_key(|s| s.addr);
        let mut want = sections.to_vec();
        want.sort_by_key(|s| s.addr);
        assert_eq!(got, want);
    }

    #[test]
    fn an_allocated_section_list_past_eof_is_rejected_by_both_read_paths() {
        let allocated = 44;
        let source = manager_source(0, allocated, 736);
        let mut buf = vec![0u8; 736];
        buf[619..701].copy_from_slice(&source.header);
        buf[701..].copy_from_slice(&source.sections);
        let expected = FormatError::InvalidFreeSpaceManager(format!(
            "the section list allocation at 701, {allocated} bytes long, runs past the \
             end of the file, {} bytes long",
            buf.len()
        ));

        assert_eq!(
            read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(
            read_persisted_sections_source(
                &source,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
            )
            .unwrap_err(),
            expected
        );
    }

    #[test]
    fn class_id_is_emitted_and_ignored_on_read() {
        // The class-id byte is written per section and round-trips (the reader
        // ignores its value, recovering the same offsets/sizes for any class).
        let sections = [section(2000, 12768)];
        for class in [
            SECTION_CLASS_SIMPLE,
            SECTION_CLASS_SMALL,
            SECTION_CLASS_LARGE,
        ] {
            let (fshd, fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
                widths(8, 8),
                &sections,
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                class,
            )
            .unwrap();
            // The last section record's class byte precedes the 4-byte checksum.
            assert_eq!(fsse[fsse.len() - 5], class, "class byte written");
            let mut buf = vec![0u8; 1100 + fsse.len()];
            buf[1000..1000 + fshd.len()].copy_from_slice(&fshd);
            buf[1100..1100 + fsse.len()].copy_from_slice(&fsse);
            let got = read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(1000))],
            )
            .unwrap();
            assert_eq!(got, sections, "class {class} round-trips");
        }
    }

    #[test]
    fn semantic_paged_managers_keep_the_existing_wire_slots_and_classes() {
        let page_size = FileSpacePageSize::DEFAULT;
        let mut space = SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 0, page_size);
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(100, 200).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallRaw),
            Extent::new(5000, 200).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedLargeGeneric),
            Extent::new(8192, page_size.get()).unwrap(),
        );
        let managers = ManagerSections::paged(
            &space.paged_sections().expect("the strategy is PAGE"),
            page_size,
        );
        let plan = plan_managers(&managers, StoredAddress::new(1000), widths(8, 8));

        assert_eq!(plan.blocks.len(), 3);
        assert_eq!(plan.slots[0], plan.blocks[0].fshd_addr);
        assert_eq!(plan.slots[2], plan.blocks[1].fshd_addr);
        assert_eq!(plan.slots[6], plan.blocks[2].fshd_addr);
        assert_eq!(plan.blocks[0].kind, ManagerKind::PagedSmallMetadata);
        assert_eq!(plan.blocks[1].kind, ManagerKind::PagedSmallRaw);
        assert_eq!(plan.blocks[2].kind, ManagerKind::PagedLargeGeneric);
        assert_eq!(plan.blocks[0].class, SECTION_CLASS_SMALL);
        assert_eq!(plan.blocks[1].class, SECTION_CLASS_SMALL);
        assert_eq!(plan.blocks[2].class, SECTION_CLASS_LARGE);
        assert_eq!(plan.blocks[0].sections, vec![section(100, 200)]);
        assert_eq!(plan.blocks[1].sections, vec![section(5000, 200)]);
        assert_eq!(
            plan.blocks[2].sections,
            vec![section(8192, page_size.get())]
        );
        assert!(
            plan.slots
                .iter()
                .enumerate()
                .all(|(slot, addr)| [0, 2, 6].contains(&slot)
                    || addr.is_undefined(OffsetWidth::Eight))
        );

        for block in &plan.blocks {
            let (fshd, fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
                widths(8, 8),
                &block.sections,
                block.fshd_addr,
                block.fsse_addr,
                block.class,
            )
            .unwrap();
            assert_eq!(
                block.fsse_addr,
                block.fshd_addr.offset(fshd.len() as u64),
                "FSSE follows its FSHD"
            );
            assert_eq!(
                block.fsse_addr.offset(fsse.len() as u64),
                if block.kind == ManagerKind::PagedLargeGeneric {
                    plan.end_of_managers
                } else {
                    plan.blocks
                        .iter()
                        .find(|candidate| candidate.fshd_addr > block.fshd_addr)
                        .unwrap()
                        .fshd_addr
                },
                "manager blocks remain contiguous"
            );
        }
    }

    #[test]
    fn manager_slots_translate_to_semantic_kinds_only_at_the_storage_boundary() {
        assert_eq!(
            manager_kind(FileSpaceStrategy::FsmAggr, 7),
            Some(ManagerKind::Flat)
        );
        assert_eq!(
            manager_kind(FileSpaceStrategy::Page, 0),
            Some(ManagerKind::PagedSmallMetadata)
        );
        assert_eq!(
            manager_kind(FileSpaceStrategy::Page, 2),
            Some(ManagerKind::PagedSmallRaw)
        );
        assert_eq!(
            manager_kind(FileSpaceStrategy::Page, 6),
            Some(ManagerKind::PagedLargeGeneric)
        );
        assert_eq!(manager_kind(FileSpaceStrategy::Page, 7), None);
    }

    #[rstest]
    #[case::zero(0, 0)]
    #[case::inside_the_first_page(1, 4096)]
    #[case::on_a_boundary(8192, 8192)]
    #[case::past_a_boundary(8193, 12_288)]
    fn align_up_rounds_to_the_next_page_boundary(#[case] value: u64, #[case] expected: u64) {
        assert_eq!(align_up(value, FileSpacePageSize::DEFAULT), expected);
    }

    struct ManagerSource {
        base: u64,
        file_len: u64,
        header: Vec<u8>,
        sections: Vec<u8>,
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl Source for ManagerSource {
        fn len(&self) -> u64 {
            self.file_len
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push((offset, buf.len()));
            let data = match offset.checked_sub(self.base) {
                Some(619) => &self.header,
                Some(701) => &self.sections,
                _ => {
                    return Err(FormatError::UnexpectedEof {
                        expected: buf.len(),
                        available: 0,
                    });
                }
            };
            buf.copy_from_slice(data.get(..buf.len()).ok_or(FormatError::UnexpectedEof {
                expected: buf.len(),
                available: data.len(),
            })?);
            Ok(())
        }
    }

    fn manager_source(base: u64, allocated: u64, file_len: u64) -> ManagerSource {
        let mut header = free_space::single_section_header();
        header[70..78].copy_from_slice(&allocated.to_le_bytes());
        let len = header.len();
        test_util::checksum::restamp(&mut header, 0, len);
        ManagerSource {
            base,
            file_len,
            header,
            sections: free_space::single_section_info(),
            reads: RefCell::new(Vec::new()),
        }
    }
}
