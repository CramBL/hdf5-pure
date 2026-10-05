//! The free sections the writer reads back from a file's persistent free-space managers, and the
//! layout of the managers of a paged file.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_format::__private::SECTION_CLASS_LARGE;
use hdf5_pure_format::__private::SECTION_CLASS_SMALL;

use crate::address::BaseAddressExt;
use crate::address::{BaseAddress, StoredAddress};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::width::FormatWidths;

/// Reads the free sections of the managers at `manager_addrs` from the file `data`.
///
/// Adds `base`, the file's base address, to every address the file stores. Skips a manager whose
/// address is undefined at the offset width of `widths`, and a manager whose section list address
/// is undefined.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if `base` and a stored address sum past `u64::MAX`,
/// [`FormatError::ValueTooLargeForPlatform`] if a position in the file or the length of a section
/// list does not fit a `usize`, [`FormatError::InvalidFreeSpaceManager`] if a header or a section
/// list is past the end of `data`, and the errors of [`FreeSpaceManagerHeader::parse`] and
/// [`parse_section_info`](hdf5_pure_format::__private::parse_section_info) for each block it reads.
pub(crate) fn read_persisted_sections(
    data: &[u8],
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[u64],
) -> Result<Vec<FreeSection>, FormatError> {
    let offset_size = widths.offsets.get();
    let mut sections = Vec::new();
    for &addr in manager_addrs {
        let addr = StoredAddress::new(addr);
        if addr.is_undefined(offset_size) {
            continue;
        }
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
        if header.section_list_addr().is_undefined(offset_size) {
            continue;
        }
        let fa = base.absolute(header.section_list_addr())?.to_usize()?;
        let used = header.section_list_used().to_usize()?;
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

/// The free sections recovered from the on-disk managers, paired with the
/// `(addr, len)` extents of every `FSHD`/`FSSE` block that was read (so a caller
/// rewriting the managers can reclaim those blocks).
pub(crate) type PersistedSections = (Vec<FreeSection>, Vec<(u64, u64)>);

/// The bounded-memory counterpart of [`read_persisted_sections`]: read every
/// persisted free section from the managers named in `manager_addrs` over a
/// random-access [`Source`](crate::Source), so the bounded backend can seed its
/// free list without a mirror. Only the small manager
/// blocks are read; nothing scales with file size.
pub(crate) fn read_persisted_sections_source<S: crate::source::Source>(
    src: &S,
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[u64],
) -> Result<PersistedSections, FormatError> {
    let offset_size = widths.offsets.get();
    let mut sections = Vec::new();
    let mut blocks = Vec::new();
    let hdr_len = hdf5_pure_format::__private::free_space_manager_header_len(widths);
    for &addr in manager_addrs {
        let addr = StoredAddress::new(addr);
        if addr.is_undefined(offset_size) {
            continue;
        }
        let a = base.absolute(addr)?;
        let fshd = src.read_exact_at(a, hdr_len.to_usize()?)?;
        let header = FreeSpaceManagerHeader::parse(widths, &fshd)?;
        blocks.push((a, hdr_len));
        if header.section_list_addr().is_undefined(offset_size) {
            continue;
        }
        let fa = base.absolute(header.section_list_addr())?;
        let used = header.section_list_used();
        let block = src.read_exact_at(fa, used.to_usize()?)?;
        sections.extend(hdf5_pure_format::__private::parse_section_info(
            widths, &block, addr, &header,
        )?);
        blocks.push((fa, used));
    }
    Ok((sections, blocks))
}

/// Page type of an allocation in a paged file (`H5F_FSPACE_STRATEGY_PAGE`). Such
/// a file never mixes metadata and raw data within one page, so the two kinds of
/// allocation are kept in separate pages and their free space is tracked by
/// separate managers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PageType {
    /// File metadata: object headers, extensible-array blocks, heaps, and the
    /// free-space blocks themselves.
    Meta,
    /// Raw dataset data: contiguous data blocks and chunk contents.
    Raw,
}

/// Round `value` up to the next multiple of `page` (`page` is a power of two >=
/// 512, validated at file creation).
pub(crate) fn align_up(value: u64, page: u64) -> u64 {
    value.div_ceil(page) * page
}

/// Split each free section at page boundaries so no section spans a page.
///
/// Coalescing a page-tail free section with freed blocks below it can produce a
/// run that crosses a page boundary or reaches `page`; splitting lets each
/// intra-page fragment stay in its SMALL-class manager while a whole free page is
/// routed to the generic-large manager, matching the reference library's
/// small-vs-large section classes. Total free bytes are preserved.
pub(crate) fn split_at_pages(sections: &[FreeSection], page: u64) -> Vec<FreeSection> {
    let mut out = Vec::new();
    for s in sections {
        let end = s.addr.get().saturating_add(s.size);
        let mut start = s.addr.get();
        while start < end {
            let boundary = (start / page + 1) * page;
            let piece_end = end.min(boundary);
            out.push(FreeSection {
                addr: StoredAddress::new(start),
                size: piece_end - start,
            });
            start = piece_end;
        }
    }
    out
}

/// One per-page-type manager's placement in a paged persist tail: where its
/// `FSHD`/`FSSE` blocks go, the section class its entries carry, and the sections
/// themselves. The slot that names it is recorded in
/// [`PagedManagerPlan::slots`].
pub(crate) struct PagedManagerBlock {
    pub(crate) fshd_addr: StoredAddress,
    pub(crate) fsse_addr: StoredAddress,
    pub(crate) class: u8,
    pub(crate) sections: Vec<FreeSection>,
}

/// The closed-form layout of a paged file's per-page-type free-space managers.
pub(crate) struct PagedManagerPlan {
    /// Manager address per File Space Info slot, the undefined address at the
    /// file's offset width where inactive.
    pub(crate) slots: [StoredAddress; NUM_FILE_FSM_MANAGERS],
    /// The active managers, in ascending address order. Empty when the file has
    /// no free space to record.
    pub(crate) blocks: Vec<PagedManagerBlock>,
    /// The first address past the last manager block (`start` when none).
    pub(crate) end_of_managers: StoredAddress,
}

impl PagedManagerPlan {
    /// True when there is no free space to record, so the caller should emit an
    /// empty-manager persist message rather than any manager blocks.
    pub(crate) fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Classes a paged file's free space into its per-page-type managers and places
/// their blocks contiguously from `start`.
///
/// Each section is first split at page boundaries, then classed by size: an
/// intra-page (`< page_size`) fragment stays in its SMALL-class per-type manager
/// — SUPER (slot 0) for metadata, DRAW (slot 2) for raw — while a whole free page
/// goes to the single generic-large manager (slot 6). This keeps a SMALL section
/// from ever spanning a page or reaching `page_size`, matching the reference
/// library.
///
/// Size is the only thing that decides between a SMALL manager and the large one,
/// which is why the caller tracks placeable free space by page *type* alone: the
/// split is recomputed here on every commit, so nothing upstream has to maintain
/// it (and maintaining it upstream would keep neighboring holes of different
/// classes from coalescing — issue #261).
///
/// `unclassified` is free space the caller could not assign a page type to,
/// because the generic-large manager it came from holds both kinds. It is written
/// back to that manager exactly as it was found: splitting or re-classing it would
/// state a page type the caller deliberately did not claim to know.
///
/// `FSSE` byte length depends only on section count and sizes (fixed field
/// widths), never on the addresses, so a single forward pass fixes every address
/// with no fixpoint iteration. Shared by the whole-file editor's and the bounded
/// backend's persist tails so the two produce identical layouts.
pub(crate) fn plan_paged_managers(
    meta: &[FreeSection],
    raw: &[FreeSection],
    unclassified: &[FreeSection],
    page_size: u64,
    start: StoredAddress,
    widths: FormatWidths,
) -> PagedManagerPlan {
    let mut slot0 = Vec::new();
    let mut slot2 = Vec::new();
    let mut slot6 = Vec::new();
    for s in split_at_pages(meta, page_size) {
        if s.size < page_size {
            slot0.push(s);
        } else {
            slot6.push(s);
        }
    }
    for s in split_at_pages(raw, page_size) {
        if s.size < page_size {
            slot2.push(s);
        } else {
            slot6.push(s);
        }
    }
    slot6.extend(unclassified.iter().copied());
    slot6.sort_unstable_by_key(|s| s.addr);
    debug_assert!(
        slot6.windows(2).all(|w| w[0].addr < w[1].addr),
        "one address is free in more than one of the metadata, raw, and \
         unclassified lists, so they have stopped being disjoint"
    );

    let mut slots = [StoredAddress::undefined(widths.offsets.get()); NUM_FILE_FSM_MANAGERS];
    let mut blocks = Vec::new();
    let mut cursor = start;
    for (slot, class, sections) in [
        (0usize, SECTION_CLASS_SMALL, slot0),
        (2usize, SECTION_CLASS_SMALL, slot2),
        (6usize, SECTION_CLASS_LARGE, slot6),
    ] {
        if sections.is_empty() {
            continue;
        }
        let fshd_addr = cursor;
        let fsse_addr = fshd_addr.offset(
            hdf5_pure_format::__private::free_space_manager_header_len(widths),
        );
        let section_sizes: Vec<u64> = sections.iter().map(|s| s.size).collect();
        cursor = fsse_addr.offset(hdf5_pure_format::__private::section_info_len(
            widths,
            &section_sizes,
        ));
        slots[slot] = fshd_addr;
        blocks.push(PagedManagerBlock {
            fshd_addr,
            fsse_addr,
            class,
            sections,
        });
    }
    PagedManagerPlan {
        slots,
        blocks,
        end_of_managers: cursor,
    }
}

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::SECTION_CLASS_SIMPLE;
    use rstest::rstest;
    use test_util::free_space;

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
        "the section list at 701, 35 bytes long, runs past the end of the file, 735 bytes long"
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
            read_persisted_sections(&buf, widths(8, 8), BaseAddress::ZERO, &[manager_addr]),
            Err(FormatError::InvalidFreeSpaceManager(reason.to_owned()))
        );
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
            &[619, u64::MAX, u64::MAX],
        )
        .unwrap();
        assert_eq!(got, vec![section(2848, 1600)]);
        // No defined managers -> no sections.
        assert!(
            read_persisted_sections(&buf, widths(8, 8), BaseAddress::ZERO, &[u64::MAX])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn an_undefined_manager_slot_is_skipped_in_a_four_byte_offset_file() {
        assert!(
            read_persisted_sections(&[], widths(4, 8), BaseAddress::ZERO, &[0xFFFF_FFFF])
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
            read_persisted_sections(&fshd, widths(4, 8), BaseAddress::ZERO, &[0])
                .unwrap()
                .is_empty(),
            "a manager tracking nothing records no section info"
        );

        let (sections, blocks) = read_persisted_sections_source(
            &crate::source::BytesSource::new(&fshd),
            widths(4, 8),
            BaseAddress::ZERO,
            &[0],
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
        let mut got =
            read_persisted_sections(&buf, widths(8, 8), BaseAddress::ZERO, &[1000]).unwrap();
        got.sort_by_key(|s| s.addr);
        let mut want = sections.to_vec();
        want.sort_by_key(|s| s.addr);
        assert_eq!(got, want);
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
            let got =
                read_persisted_sections(&buf, widths(8, 8), BaseAddress::ZERO, &[1000]).unwrap();
            assert_eq!(got, sections, "class {class} round-trips");
        }
    }

    #[test]
    fn split_at_pages_splits_on_boundaries_preserving_total() {
        let page = 4096;
        // A run that crosses one boundary splits into a sub-page head and a whole
        // free page (the accounting fix routes the >= page piece to slot 6).
        assert_eq!(
            split_at_pages(&[section(3740, 4452)], page),
            vec![
                section(3740, 356),  // [3740, 4096)
                section(4096, 4096), // [4096, 8192)
            ]
        );
        // A sub-page section is returned unchanged (the common case is a no-op).
        assert_eq!(
            split_at_pages(&[section(100, 200)], page),
            vec![section(100, 200)]
        );
        // A page-aligned multi-page run splits into whole pages.
        let whole = split_at_pages(&[section(0, 3 * 4096)], page);
        assert_eq!(whole.len(), 3);
        assert!(whole.iter().all(|s| s.size == 4096));
        // Splitting never loses or overlaps space: pieces are contiguous and sum
        // to the original size.
        let pieces = split_at_pages(&[section(5000, 10000)], page);
        assert_eq!(pieces.iter().map(|s| s.size).sum::<u64>(), 10000);
        let mut prev = pieces[0].addr.get();
        for s in &pieces {
            assert_eq!(s.addr.get(), prev);
            prev = s.addr.get() + s.size;
        }
        assert_eq!(prev, 15000);
    }
}
