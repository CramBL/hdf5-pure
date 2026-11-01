//! Pure placement and encoding of the managers for a persistent PAGE edit.

use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::{Extent, FreeClass, PagedTail, SessionSpace};

use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::free_space_manager::{EncodedManager, ManagerProjection};
use crate::width::FormatWidths;

/// Holds the encoded managers and the same final free-space state for adoption.
pub(super) struct PagedPersistencePlan {
    /// The complete reservation, final EOA, and free-space state to adopt.
    pub tail: PagedTail,
    /// The manager addresses to record in File Space Info, undefined for inactive managers.
    pub slots: [StoredAddress; NUM_FILE_FSM_MANAGERS],
    /// The encoded managers, with surplus capacity owned by the final section list.
    pub blocks: Vec<EncodedManager>,
}

impl PagedPersistencePlan {
    /// Plans placement using only durable free space or appended metadata pages.
    ///
    /// Bounds section-list capacity before placement, including the known metadata-page tail.
    /// Encodes surplus capacity as section-list padding, so allocation does not depend on exact
    /// serialized-size convergence. Returns `None` when `reuse_only` is set and no reservation fits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] or [`FormatError::ValueTooLargeForPlatform`] if
    /// capacity or address calculations overflow, [`FormatError::Internal`] if the strategy is
    /// not PAGE or the capacity does not contain the managers, or an encoding error if the
    /// managers cannot be serialized at `widths`.
    pub fn new(
        space: &SessionSpace,
        eoa: StoredAddress,
        freed: &[(Extent, FreeClass)],
        old_blocks: &[Extent],
        extension_len: u64,
        widths: FormatWidths,
        reuse_only: bool,
    ) -> Result<Option<Self>, FormatError> {
        space.page_size().ok_or_else(|| {
            FormatError::Internal("PAGE persistence requires PAGE session space".into())
        })?;
        if let Some(tail) = space.plan_paged_tail(eoa, freed, old_blocks, extension_len, false)? {
            if ManagerProjection::new(tail.post.admitted_snapshot(), widths)?.is_empty() {
                return Ok(Some(Self {
                    tail,
                    slots: [StoredAddress::undefined(widths.offsets); NUM_FILE_FSM_MANAGERS],
                    blocks: Vec::new(),
                }));
            }
        }
        let section_limit = space.paged_tail_section_limit(
            freed
                .iter()
                .map(|&(extent, _)| extent)
                .chain(old_blocks.iter().copied()),
        )?;
        let prefixes = format::section_info_capacity(widths, 0)? * (PAGED_MANAGERS - 1);
        let fixed = format::free_space_manager_header_len(widths) * PAGED_MANAGERS + prefixes;
        let fixed = fixed
            .checked_add(extension_len)
            .ok_or(FormatError::OffsetOverflow {
                offset: extension_len,
                length: fixed,
            })?;
        let mut capacity = format::section_info_capacity(widths, section_limit)?
            .checked_add(fixed)
            .ok_or(FormatError::OffsetOverflow {
                offset: fixed,
                length: section_limit as u64,
            })?;
        let reused = space
            .plan_paged_tail(eoa, freed, old_blocks, capacity, false)?
            .map(|tail| ProjectedTail::new(tail, widths))
            .transpose()?;
        let mut projected = match reused {
            Some(projected) if !projected.managers.is_empty() => projected,
            _ if reuse_only => return Ok(None),
            _ => ProjectedTail::new(
                space
                    .plan_paged_tail(eoa, freed, old_blocks, capacity, true)?
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "an appended PAGE tail requires no reusable extent".into(),
                        )
                    })?,
                widths,
            )?,
        };
        if projected.managers.is_empty() {
            capacity = capacity.checked_add(1).ok_or(FormatError::OffsetOverflow {
                offset: capacity,
                length: 1,
            })?;
            projected = ProjectedTail::new(
                space
                    .plan_paged_tail(eoa, freed, old_blocks, capacity, true)?
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "an appended PAGE tail requires no reusable extent".into(),
                        )
                    })?,
                widths,
            )?;
        }
        let ProjectedTail { tail, managers } = projected;
        let measured = managers.encoded_len();
        let used = extension_len
            .checked_add(measured)
            .ok_or(FormatError::OffsetOverflow {
                offset: extension_len,
                length: measured,
            })?;
        if used > capacity || managers.is_empty() {
            return Err(FormatError::Internal(
                "the PAGE manager capacity does not contain the final manager set".into(),
            ));
        }
        let start = tail.allocation.start().checked_add(extension_len)?;
        let plan = managers.into_plan(start)?;
        let slots = plan.slots;
        let blocks = plan
            .encode(tail.allocation.end())?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            tail,
            slots,
            blocks,
        }))
    }
}

/// A proposed PAGE tail and the manager projection measured from its free state.
struct ProjectedTail {
    tail: PagedTail,
    managers: ManagerProjection,
}

impl ProjectedTail {
    fn new(tail: PagedTail, widths: FormatWidths) -> Result<Self, FormatError> {
        let managers = ManagerProjection::new(tail.post.admitted_snapshot(), widths)?;
        Ok(Self { tail, managers })
    }
}

const PAGED_MANAGERS: u64 = 3;

#[cfg(test)]
mod tests {
    use hdf5_pure_space::__private::{ManagerKind, ManagerSections, PageType};
    use rstest::rstest;

    use crate::AddressRange;
    use crate::address::{StoredAddress, StoredAddressExt};
    use crate::file_space_info::{FileSpacePageSize, FileSpaceStrategy};
    use crate::free_space_manager;

    use super::*;

    #[test]
    fn exact_manager_sizes_can_oscillate_while_owned_capacity_terminates() {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            1,
            FileSpacePageSize::DEFAULT,
        );
        let freed = [(
            Extent::new(StoredAddress::new(100), 3_801).unwrap(),
            FreeClass::Page(PageType::Meta),
        )];
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let measure = |capacity| {
            let tail = space
                .plan_paged_tail(StoredAddress::new(8_192), &freed, &[], capacity, true)
                .unwrap()
                .unwrap();
            let managers =
                ManagerSections::paged(&tail.post.sections(), FileSpacePageSize::DEFAULT);
            160 + free_space_manager::manager_blocks_len(&managers, widths).unwrap()
        };
        assert_eq!((measure(286), measure(295)), (295, 286));
        let plan = PagedPersistencePlan::new(
            &space,
            StoredAddress::new(8_192),
            &freed,
            &[],
            160,
            widths,
            false,
        )
        .unwrap()
        .unwrap();
        assert!(plan.tail.appended);
        assert_eq!(plan.tail.post.sections().metadata().len(), 2);
        assert_complete_capacity(&plan, widths, 160);
        space.adopt_paged_post_free(plan.tail.post, true, None);
        assert_eq!(space.paged_sections().unwrap().metadata().len(), 2);
    }

    #[rstest]
    fn a_paged_tail_takes_exactly_the_space_it_fills(
        #[values(
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
            137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
            154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
            171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
            205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
            239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
            256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
            273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
            290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306,
            307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
            324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
            341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357,
            358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374,
            375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
            392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408,
            409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425,
            426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442,
            443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
            460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476,
            477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
            494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510,
            511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
            528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544,
            545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561,
            562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578,
            579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595,
            596, 597, 598, 599, 600
        )]
        hole: u64,
    ) {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            0,
            FileSpacePageSize::DEFAULT,
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(StoredAddress::new(4_096), hole).unwrap(),
        );
        let before = space.paged_sections().unwrap().metadata().to_vec();
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let plan = PagedPersistencePlan::new(
            &space,
            StoredAddress::new(12_288),
            &[],
            &[],
            100,
            widths,
            true,
        )
        .unwrap();
        assert_eq!(space.paged_sections().unwrap().metadata(), before);
        assert_eq!(plan.is_some(), hole >= 470);
        if let Some(plan) = plan {
            assert!(!plan.tail.appended);
            assert_eq!(plan.tail.allocation.start(), StoredAddress::new(4_096));
            assert_complete_capacity(&plan, widths, 100);
            let sections = plan.tail.post.sections();
            assert_eq!(
                plan.tail.allocation.len()
                    + sections
                        .metadata()
                        .iter()
                        .map(|extent| extent.len())
                        .sum::<u64>(),
                hole
            );
            assert!(sections.raw().is_empty());
            assert!(sections.unclassified().is_empty());
            assert!(
                sections
                    .metadata()
                    .iter()
                    .all(|extent| extent.start() >= plan.tail.allocation.end())
            );
        }
    }

    #[rstest]
    #[case::small_hole(64, false)]
    #[case::reusable(1_000, true)]
    #[case::whole_page(4_096, true)]
    fn pure_reuse_planning_keeps_all_capacity_owned(#[case] hole: u64, #[case] fits: bool) {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            1,
            FileSpacePageSize::DEFAULT,
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(StoredAddress::new(4_096), hole).unwrap(),
        );
        let before = space.paged_sections().unwrap().metadata().to_vec();
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let plan = PagedPersistencePlan::new(
            &space,
            StoredAddress::new(12_288),
            &[],
            &[],
            160,
            widths,
            true,
        )
        .unwrap();
        assert_eq!(plan.is_some(), fits);
        assert_eq!(space.paged_sections().unwrap().metadata(), before);
        if let Some(plan) = plan {
            assert!(!plan.tail.appended);
            assert_complete_capacity(&plan, widths, 160);
            assert_eq!(plan.tail.allocation.start(), StoredAddress::new(4_096));
        }
    }

    #[rstest]
    #[case::one_byte_tail(160, 512)]
    #[case::exact_capacity_fill(161, 512)]
    #[case::large_page(160, 8_192)]
    fn appended_capacity_preserves_even_the_smallest_known_tail(
        #[case] extension: u64,
        #[case] page: u64,
    ) {
        let page = FileSpacePageSize::try_from(page).unwrap();
        let space =
            SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), page.get() * 2, page);
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let plan = PagedPersistencePlan::new(
            &space,
            StoredAddress::new(page.get() * 2),
            &[],
            &[],
            extension,
            widths,
            false,
        )
        .unwrap()
        .unwrap();
        assert_complete_capacity(&plan, widths, extension);
        let tail = plan.tail.post.sections();
        assert_eq!(tail.metadata().len(), 1);
        assert_eq!(tail.metadata()[0].start(), plan.tail.allocation.end());
        assert_eq!(tail.metadata()[0].end(), plan.tail.final_eoa);
    }

    #[test]
    fn an_exact_extension_reservation_can_publish_without_managers() {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            1,
            FileSpacePageSize::DEFAULT,
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(StoredAddress::new(4_096), 160).unwrap(),
        );
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let plan = PagedPersistencePlan::new(
            &space,
            StoredAddress::new(8_192),
            &[],
            &[],
            160,
            widths,
            true,
        )
        .unwrap()
        .unwrap();
        assert!(plan.blocks.is_empty());
        assert_eq!(plan.tail.allocation.len(), 160);
        assert!(
            plan.slots
                .iter()
                .all(|address| address.is_undefined(widths.offsets))
        );
    }

    #[test]
    fn padded_managers_preserve_unclassified_generic_fragments() {
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            1,
            FileSpacePageSize::DEFAULT,
        );
        let fragment = Extent::new(StoredAddress::new(5_694), 450).unwrap();
        space.seed_persisted(Some(ManagerKind::PagedLargeGeneric), fragment);
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let plan = PagedPersistencePlan::new(
            &space,
            StoredAddress::new(8_192),
            &[],
            &[],
            160,
            widths,
            false,
        )
        .unwrap()
        .unwrap();
        assert!(plan.tail.appended);
        assert_eq!(plan.tail.post.sections().unclassified(), [fragment]);
        assert_complete_capacity(&plan, widths, 160);
    }

    fn assert_complete_capacity(plan: &PagedPersistencePlan, widths: FormatWidths, extension: u64) {
        let mut cursor = plan.tail.allocation.start().checked_add(extension).unwrap();
        let mut padded = false;
        for block in &plan.blocks {
            assert_eq!(block.header_address, cursor);
            cursor = cursor.checked_add(block.header.len() as u64).unwrap();
            assert_eq!(block.section_list_address, cursor);
            cursor = cursor.checked_add(block.section_list.len() as u64).unwrap();
            let header = format::FreeSpaceManagerHeader::parse(widths, &block.header).unwrap();
            assert_eq!(
                header.section_list_allocated(),
                block.section_list.len() as u64
            );
            assert_eq!(header.section_list_used(), header.section_list_allocated());
            let sections = format::parse_section_info(
                widths,
                &block.section_list,
                block.header_address,
                &header,
            )
            .unwrap();
            padded |= format::free_space_manager_len(widths, &sections)
                < (block.header.len() + block.section_list.len()) as u64;
            assert!(sections.iter().all(|section| {
                !AddressRange::new(section.addr, section.size)
                    .unwrap()
                    .overlaps(plan.tail.allocation.range())
            }));
        }
        assert_eq!(cursor, plan.tail.allocation.end());
        assert!(padded);
    }
}
