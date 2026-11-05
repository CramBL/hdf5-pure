use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_space::__private::AdmittedFreeSnapshot;
use hdf5_pure_space::__private::FreeList;
use hdf5_pure_space::__private::{ManagerKind, ManagerSections, PageType};
use rstest::rstest;
use test_util::range;
use test_util::snapshot::text::{Report, Value};

use crate::AddressRange;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::file_space_info::{FileSpacePageSize, FileSpaceStrategy};
use crate::free_space_manager;
use crate::message_type::MessageType;

use super::*;

#[rstest]
#[case::empty(&[], &[], &[(100, 100)])]
#[case::whole(&[(100, 100)], &[(100, 100)], &[])]
#[case::middle(&[(120, 40)], &[(120, 40)], &[(100, 20), (160, 40)])]
#[case::adjacent(&[(100, 20), (120, 80)], &[(100, 20), (120, 80)], &[])]
#[case::unordered(&[(170, 30), (100, 20), (120, 30)], &[(100, 20), (120, 30), (170, 30)], &[(150, 20)])]
#[case::several_gaps(&[(170, 20), (120, 30)], &[(120, 30), (170, 20)], &[(100, 20), (150, 20), (190, 10)])]
fn partitions_preserve_complete_allocations_and_exact_gaps(
    #[case] allocations: &[(u64, u64)],
    #[case] expected_occupied: &[(u64, u64)],
    #[case] expected_gaps: &[(u64, u64)],
) {
    let reservation = extent(100, 100);
    let partition = ReservationPartition::new(
        reservation,
        allocations
            .iter()
            .map(|&(start, length)| extent(start, length)),
    )
    .unwrap();
    assert_eq!(partition.reservation, reservation);
    assert_eq!(
        partition.occupied,
        expected_occupied
            .iter()
            .map(|&(start, length)| extent(start, length))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        partition.gaps,
        expected_gaps
            .iter()
            .map(|&(start, length)| extent(start, length))
            .collect::<Vec<_>>()
    );
    let pieces = partition
        .occupied
        .iter()
        .chain(&partition.gaps)
        .map(|allocation| (allocation.start().get(), allocation.len().get()))
        .collect::<Vec<_>>();
    assert_eq!(range::covered_len(&pieces, &(100..200)), 100);
    assert_eq!(pieces.iter().map(|&(_, length)| length).sum::<u64>(), 100);
    if expected_gaps.is_empty() {
        let covered = partition.into_occupied().unwrap();
        assert_eq!(covered.reservation(), reservation);
        assert_eq!(covered.into_retirement(), [reservation]);
    } else {
        assert_eq!(
            partition.into_occupied().unwrap_err(),
            FormatError::Internal("a PAGE reservation has unoccupied bytes".into())
        );
    }
}

#[rstest]
#[case::before(&[(99, 10)], "a PAGE block is outside its reservation")]
#[case::after(&[(195, 6)], "a PAGE block is outside its reservation")]
#[case::enclosing(&[(99, 102)], "a PAGE block is outside its reservation")]
#[case::duplicate(&[(100, 20), (100, 20)], "PAGE block allocations overlap")]
#[case::overlap(&[(100, 51), (150, 50)], "PAGE block allocations overlap")]
#[case::nested(&[(100, 100), (120, 20)], "PAGE block allocations overlap")]
fn partition_construction_rejects_invalid_occupied_ranges(
    #[case] allocations: &[(u64, u64)],
    #[case] reason: &str,
) {
    let error = ReservationPartition::new(
        extent(100, 100),
        allocations
            .iter()
            .map(|&(start, length)| extent(start, length)),
    )
    .unwrap_err();
    assert_eq!(error, FormatError::Internal(reason.into()));
}

#[rstest]
#[case::wide(1 << 35, 1 << 33)]
#[case::maximum_end(u64::MAX - 100, 100)]
fn reservation_partitions_preserve_u64_geometry(#[case] start: u64, #[case] length: u64) {
    let reservation = extent(start, length);
    let partition = ReservationPartition::new(
        reservation,
        [extent(start, 20), extent(start + 40, length - 40)],
    )
    .unwrap();
    assert_eq!(partition.reservation, reservation);
    assert_eq!(
        partition.occupied,
        [extent(start, 20), extent(start + 40, length - 40)]
    );
    assert_eq!(partition.gaps, [extent(start + 20, 20)]);
}

#[test]
fn partitions_include_complete_manager_capacity() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let free = FreeList::from_sorted([extent(4096, 4096)]).unwrap();
    let projection = ManagerProjection::new(AdmittedFreeSnapshot::flat(&free), widths).unwrap();
    assert_eq!(projection.block_lengths().collect::<Vec<_>>(), [[82, 35]]);
    let managers = projection
        .place(|[header, list]| Ok([extent(1100, header), extent(1182, list + 32)]))
        .unwrap();
    assert_eq!(
        managers.allocations().collect::<Vec<_>>(),
        [extent(1100, 82), extent(1182, 67)]
    );
    let partition = ReservationPartition::new(
        extent(1000, 249),
        core::iter::once(extent(1000, 100)).chain(managers.allocations()),
    )
    .unwrap();
    assert_eq!(
        partition.occupied,
        [extent(1000, 100), extent(1100, 82), extent(1182, 67)]
    );
    assert!(partition.gaps.is_empty());
    assert_eq!(
        partition.into_occupied().unwrap().into_retirement(),
        [extent(1000, 249)]
    );
}

#[rstest]
#[case::extension_only(160, 0)]
#[case::with_manager(4096, 1)]
fn paged_plans_retain_one_fully_occupied_reservation(#[case] hole: u64, #[case] managers: usize) {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 0, FileSpacePageSize::DEFAULT);
    space.seed_persisted(Some(ManagerKind::PagedSmallMetadata), extent(4096, hole));
    let plan = plan_persistence(
        &space,
        StoredAddress::new(12288),
        &[],
        &[],
        160,
        FormatWidths::from_sizes(8, 8).unwrap(),
        true,
    )
    .unwrap()
    .unwrap();
    let reservation = plan.tail.allocation;
    assert!(!plan.tail.appended);
    assert_eq!(plan.coverage.reservation(), reservation);
    assert_eq!(plan.blocks.len(), managers);
    assert_eq!(plan.extension.first().unwrap().start(), reservation.start());
    let expected_free = plan.tail.post.admitted_snapshot().reusable_bytes().unwrap();
    assert_eq!(expected_free.get() + reservation.len().get(), hole);
    let (tail, extension, slots, blocks, coverage) = plan.into_parts();
    assert_eq!(tail.allocation, reservation);
    assert_eq!(extension.first().unwrap().start(), reservation.start());
    assert_eq!(blocks.len(), managers);
    if managers == 0 {
        assert_eq!(extension, [reservation]);
        assert!(
            slots.iter().all(
                |address| address.is_undefined(FormatWidths::from_sizes(8, 8).unwrap().offsets)
            )
        );
    } else {
        assert_eq!(slots[0], blocks[0].header_address);
        assert!(
            slots[1..].iter().all(
                |address| address.is_undefined(FormatWidths::from_sizes(8, 8).unwrap().offsets)
            )
        );
    }
    assert_eq!(coverage.into_retirement(), [reservation]);
    space.adopt_paged_post_free(tail.post, false, None);
    assert_eq!(
        space.admitted_snapshot().reusable_bytes().unwrap(),
        expected_free
    );
}

#[test]
fn exact_manager_sizes_can_oscillate_while_owned_capacity_terminates() {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    let freed = [(
        Extent::new(StoredAddress::new(100), ByteLength::new(3_801)).unwrap(),
        FreeClass::Page(PageType::Meta),
    )];
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let measure = |capacity| {
        let tail = space
            .plan_paged_tail(StoredAddress::new(8_192), &freed, &[], capacity, true)
            .unwrap()
            .unwrap();
        let managers = ManagerSections::paged(&tail.post.sections(), FileSpacePageSize::DEFAULT);
        160 + free_space_manager::manager_blocks_len(&managers, widths).unwrap()
    };
    assert_eq!((measure(286), measure(295)), (295, 286));
    let plan = plan_persistence(
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
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
        138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
        156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
        174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
        228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
        246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
        264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
        282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,
        300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
        318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
        336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353,
        354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
        372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
        390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
        408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425,
        426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443,
        444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
        462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
        480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497,
        498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515,
        516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
        534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
        552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
        570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587,
        588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600
    )]
    hole: u64,
) {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 0, FileSpacePageSize::DEFAULT);
    space.seed_persisted(
        Some(ManagerKind::PagedSmallMetadata),
        Extent::new(StoredAddress::new(4_096), ByteLength::new(hole)).unwrap(),
    );
    let before = space.paged_sections().unwrap().metadata().to_vec();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let plan = plan_persistence(
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
            plan.tail.allocation.len().get()
                + sections
                    .metadata()
                    .iter()
                    .map(|extent| extent.len().get())
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
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    space.seed_persisted(
        Some(ManagerKind::PagedSmallMetadata),
        Extent::new(StoredAddress::new(4_096), ByteLength::new(hole)).unwrap(),
    );
    let before = space.paged_sections().unwrap().metadata().to_vec();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let plan = plan_persistence(
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
    let space = SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), page.get() * 2, page);
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let plan = plan_persistence(
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
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    space.seed_persisted(
        Some(ManagerKind::PagedSmallMetadata),
        Extent::new(StoredAddress::new(4_096), ByteLength::new(160)).unwrap(),
    );
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let plan = plan_persistence(
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
    assert_eq!(plan.tail.allocation.len(), ByteLength::new(160));
    assert!(
        plan.slots
            .iter()
            .all(|address| address.is_undefined(widths.offsets))
    );
}

#[test]
fn padded_managers_preserve_unclassified_generic_fragments() {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 1, FileSpacePageSize::DEFAULT);
    let fragment = Extent::new(StoredAddress::new(5_694), ByteLength::new(450)).unwrap();
    space.seed_persisted(Some(ManagerKind::PagedLargeGeneric), fragment);
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let plan = plan_persistence(
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

#[rstest]
#[case::two_pages(2)]
#[case::hole(256)]
#[case::many_pages(1 << 32)]
fn compact_runs_keep_projection_and_capacity_independent_of_page_count(#[case] pages: u64) {
    let page = FileSpacePageSize::DEFAULT.get();
    let durable = extent(100, 2_048);
    let run = extent(4 * page, pages * page);
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::DEFAULT,
        [
            (Some(ManagerKind::PagedSmallMetadata), durable),
            (Some(ManagerKind::PagedLargeGeneric), run),
        ],
    )
    .unwrap();
    let before: Vec<_> = space.admitted_snapshot().sections().collect();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let admitted = ManagerProjection::new(space.admitted_snapshot(), widths)
        .unwrap()
        .work();
    let semantic = space.paged_tail_section_limit([]).unwrap();
    assert_eq!(semantic, 2);
    assert_eq!((admitted.managers, admitted.sections), (2, 2));
    let plan = plan_persistence(
        &space,
        StoredAddress::new((pages + 5) * page),
        &[],
        &[],
        160,
        widths,
        true,
    )
    .unwrap()
    .unwrap();
    assert!(!plan.tail.appended);
    assert_eq!(plan.tail.post.sections().raw(), [run]);
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        before
    );
    assert_eq!(
        plan.tail
            .post
            .admitted_snapshot()
            .reusable_bytes()
            .unwrap()
            .get()
            + plan.tail.allocation.len().get(),
        durable.len().get() + run.len().get()
    );
    assert!(plan.tail.allocation.len() < ByteLength::new(page));
    assert_complete_capacity(&plan, widths, 160);
    let reserved = ManagerProjection::new(plan.tail.post.admitted_snapshot(), widths)
        .unwrap()
        .work();
    assert_eq!((reserved.managers, reserved.sections), (2, 2));
    let mut report = Report::new();
    report.table(
        "projection",
        [
            "phase",
            "managers",
            "records",
            "vector capacity",
            "encoded bytes",
        ],
        [
            [
                Value::text("admitted"),
                Value::Number(admitted.managers),
                Value::Number(admitted.sections),
                Value::Number(admitted.section_capacity),
                Value::Number(admitted.encoded_bytes),
            ],
            [
                Value::text("reserved"),
                Value::Number(reserved.managers),
                Value::Number(reserved.sections),
                Value::Number(reserved.section_capacity),
                Value::Number(reserved.encoded_bytes),
            ],
        ],
    );
    report.metrics(
        "capacity contributions",
        "records",
        [
            ("semantic extents", semantic as u64),
            ("reservation effects", RESERVATION_SECTION_LIMIT as u64),
            ("total bound", (semantic + RESERVATION_SECTION_LIMIT) as u64),
        ],
    );
    report.metrics(
        "owned storage",
        "bytes",
        [
            ("extension encoding", 160),
            (
                "extension allocation",
                plan.extension
                    .iter()
                    .map(|allocation| allocation.len().get())
                    .sum::<u64>(),
            ),
            ("manager encoding", reserved.encoded_bytes),
            (
                "manager allocations",
                plan.tail.allocation.len().get()
                    - plan
                        .extension
                        .iter()
                        .map(|allocation| allocation.len().get())
                        .sum::<u64>(),
            ),
            ("complete reservation", plan.tail.allocation.len().get()),
        ],
    );
    insta::assert_snapshot!("compact_page_capacity", report.finish());
}

#[rstest]
#[case::metadata(Managers::Metadata, 1)]
#[case::all_managers(Managers::All, 8)]
#[case::raw_count_one_byte(Managers::All, 255)]
#[case::raw_count_two_bytes(Managers::All, 256)]
fn compact_capacity_contains_size_groups_and_reservation_effects(
    #[case] managers: Managers,
    #[case] count: u64,
    #[values(4, 8)] width: u8,
) {
    let page = FileSpacePageSize::DEFAULT.get();
    let mut sections = Vec::new();
    for index in 0..count {
        sections.push((
            Some(ManagerKind::PagedSmallMetadata),
            extent((2 * index + 1) * page + 128, index + 1),
        ));
        if matches!(managers, Managers::All) {
            sections.push((
                Some(ManagerKind::PagedSmallRaw),
                extent((2 * index + 2) * page + 128, index + 1),
            ));
        }
    }
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::DEFAULT,
        sections,
    )
    .unwrap();
    let before: Vec<_> = space.admitted_snapshot().sections().collect();
    let mut freed = vec![(
        extent((2 * count + 21) * page + 4_000, 200),
        FreeClass::Page(PageType::Meta),
    )];
    if matches!(managers, Managers::All) {
        freed.push((extent((2 * count + 3) * page, 16 * page), FreeClass::Dead));
    }
    let old_blocks = [extent((2 * count + 20) * page + 128, 64)];
    let limit = space
        .paged_tail_section_limit(freed.iter().map(|&(extent, _)| extent).chain(old_blocks))
        .unwrap();
    let widths = FormatWidths::from_sizes(width, width).unwrap();
    let plan = plan_persistence(
        &space,
        StoredAddress::new((2 * count + 24) * page),
        &freed,
        &old_blocks,
        160,
        widths,
        false,
    )
    .unwrap()
    .unwrap();
    assert!(plan.tail.appended);
    assert_complete_capacity(&plan, widths, 160);
    let projected = ManagerProjection::new(plan.tail.post.admitted_snapshot(), widths)
        .unwrap()
        .work();
    assert!(projected.sections <= (limit + RESERVATION_SECTION_LIMIT) as u64);
    assert_eq!(
        projected.managers,
        match managers {
            Managers::Metadata => 1,
            Managers::All => 3,
        }
    );
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        before
    );
}

#[rstest]
#[case::two_pages(2)]
#[case::hole(256)]
fn a_reservation_preserves_a_compact_remaining_run(#[case] pages: u64) {
    let page = FileSpacePageSize::DEFAULT.get();
    let run = extent(page, pages * page);
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::DEFAULT,
        [(Some(ManagerKind::PagedLargeGeneric), run)],
    )
    .unwrap();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let semantic = space.paged_tail_section_limit([]).unwrap();
    assert_eq!(semantic, 1);
    let plan = plan_persistence(
        &space,
        StoredAddress::new((pages + 2) * page),
        &[],
        &[],
        160,
        widths,
        true,
    )
    .unwrap()
    .unwrap();
    assert!(!plan.tail.appended);
    assert_eq!(plan.tail.allocation.start(), run.start());
    assert_complete_capacity(&plan, widths, 160);
    let post = plan.tail.post.admitted_snapshot();
    assert_eq!(
        post.reusable_bytes().unwrap().get() + plan.tail.allocation.len().get(),
        run.len().get()
    );
    let managers = ManagerSections::from_snapshot(post);
    let generic = managers
        .iter()
        .find(|manager| manager.kind() == ManagerKind::PagedLargeGeneric)
        .unwrap();
    assert_eq!(generic.sections(), [extent(2 * page, (pages - 1) * page)]);
    assert!(
        managers
            .iter()
            .map(|manager| manager.sections().len())
            .sum::<usize>()
            <= semantic + RESERVATION_SECTION_LIMIT
    );
    assert_eq!(
        space
            .admitted_snapshot()
            .reusable_sections()
            .collect::<Vec<_>>(),
        [run]
    );
}

#[test]
fn legacy_per_page_input_is_persisted_as_a_complete_run() {
    let page = FileSpacePageSize::DEFAULT.get();
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::DEFAULT,
        [
            (Some(ManagerKind::PagedSmallMetadata), extent(100, 2_048)),
            (Some(ManagerKind::PagedLargeGeneric), extent(4 * page, page)),
            (Some(ManagerKind::PagedLargeGeneric), extent(5 * page, page)),
            (Some(ManagerKind::PagedLargeGeneric), extent(6 * page, page)),
        ],
    )
    .unwrap();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let plan = plan_persistence(
        &space,
        StoredAddress::new(8 * page),
        &[],
        &[],
        160,
        widths,
        true,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        plan.tail.post.sections().raw(),
        [extent(4 * page, 3 * page)]
    );
    let (slot, _) = free_space_manager::manager_wire(ManagerKind::PagedLargeGeneric);
    let generic = plan
        .blocks
        .iter()
        .find(|block| block.header_address == plan.slots[slot])
        .unwrap();
    let header = format::FreeSpaceManagerHeader::parse(widths, &generic.header).unwrap();
    let sections = format::parse_section_info(
        widths,
        &generic.section_list,
        generic.header_address,
        &header,
    )
    .unwrap();
    assert_eq!(
        sections,
        [format::FreeSection {
            addr: StoredAddress::new(4 * page),
            size: 3 * page
        }]
    );
    assert_complete_capacity(&plan, widths, 160);
}

#[test]
fn a_capacity_rejects_overflowing_combined_section_counts() {
    let error = ManagerCapacity::new(
        usize::MAX,
        RESERVATION_SECTION_LIMIT,
        160,
        FormatWidths::from_sizes(8, 8).unwrap(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        FormatError::ValueTooLargeForPlatform {
            value: (usize::MAX as u64).saturating_add(RESERVATION_SECTION_LIMIT as u64),
            target: "PAGE persistence section count",
        }
    );
}

enum Managers {
    All,
    Metadata,
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(length)).unwrap()
}

fn plan_persistence(
    space: &SessionSpace,
    eoa: StoredAddress,
    freed: &[(Extent, FreeClass)],
    old_blocks: &[Extent],
    extension: u64,
    widths: FormatWidths,
    reuse_only: bool,
) -> Result<Option<PagedPersistencePlan>, FormatError> {
    let prefix = format::ObjectHeaderPrefix::PLAIN;
    let records = prefix.layout.record(
        MessageType::DATATYPE,
        &vec![0; usize::try_from(extension - 15).unwrap()],
    );
    let geometry = ObjectHeaderAllocationGeometry::new(prefix, widths, &records)?;
    PagedPersistencePlan::new(space, eoa, freed, old_blocks, &geometry, reuse_only)
}

fn assert_complete_capacity(plan: &PagedPersistencePlan, widths: FormatWidths, extension: u64) {
    assert!(
        plan.extension
            .iter()
            .map(|allocation| allocation.len().get())
            .sum::<u64>()
            >= extension
    );
    let mut allocations = plan.extension.clone();
    for block in &plan.blocks {
        allocations.push(extent(
            block.header_address.get(),
            block.header.len() as u64,
        ));
        allocations.push(extent(
            block.section_list_address.get(),
            block.section_list.len() as u64,
        ));
    }
    let page = plan
        .tail
        .post
        .admitted_snapshot()
        .page_size()
        .unwrap()
        .get();
    for allocation in &allocations {
        if allocation.len() < ByteLength::new(page) {
            assert_eq!(
                allocation.start().get() / page,
                (allocation.end().get() - 1) / page,
                "{allocation:?}"
            );
        } else {
            assert_eq!(allocation.start().get() % page, 0, "{allocation:?}");
        }
    }
    for block in &plan.blocks {
        let header = format::FreeSpaceManagerHeader::parse(widths, &block.header).unwrap();
        assert_eq!(
            header.section_list_allocated(),
            block.section_list.len() as u64
        );
        assert_eq!(header.section_list_used(), header.section_list_allocated());
        let sections =
            format::parse_section_info(widths, &block.section_list, block.header_address, &header)
                .unwrap();
        assert_eq!(
            format::free_space_manager_len(widths, &sections),
            (block.header.len() + block.section_list.len()) as u64
        );
        assert!(
            sections.iter().all(
                |section| allocations.iter().all(|allocation| !AddressRange::new(
                    section.addr,
                    section.size
                )
                .unwrap()
                .overlaps(allocation.range()))
            )
        );
    }
    let partition = ReservationPartition::new(plan.tail.allocation, allocations).unwrap();
    let free = plan
        .tail
        .post
        .admitted_snapshot()
        .sections()
        .map(|(range, _)| range)
        .collect::<Vec<_>>();
    for gap in partition.gaps {
        assert_eq!(
            range::covered_len(
                &free
                    .iter()
                    .map(|allocation| (allocation.start().get(), allocation.len().get()))
                    .collect::<Vec<_>>(),
                &(gap.start().get()..gap.end().get())
            ),
            gap.len().get()
        );
    }
}

#[rstest]
#[case::ordinary(4_096)]
#[case::large(1 << 30)]
fn planning_reuses_an_aligned_block_after_rejecting_a_crossing_reservation(
    #[case] page: u64,
    #[values(false, true)] reuse_only: bool,
) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let page_size = FileSpacePageSize::try_from(page).unwrap();
    let crossing = extent(61 * page - 1_697, 2_489);
    let generic = extent(33 * page, 17 * page);
    let mut space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        page_size,
        [
            (Some(ManagerKind::PagedLargeGeneric), generic),
            (Some(ManagerKind::PagedSmallMetadata), crossing),
        ]
        .into_iter()
        .chain((0..108).map(|index| {
            (
                Some(ManagerKind::PagedSmallMetadata),
                extent((70 + index) * page, index + 1),
            )
        })),
    )
    .unwrap();
    let eoa = StoredAddress::new(200 * page);
    let semantic = space.paged_tail_section_limit([]).unwrap();
    let capacity = ManagerCapacity::new(semantic, RESERVATION_SECTION_LIMIT, 140, widths).unwrap();
    assert_eq!(semantic, 111);
    assert_eq!(capacity.get(), 2_489);
    let first = space
        .plan_paged_tail(eoa, &[], &[], capacity.get(), false)
        .unwrap()
        .unwrap();
    assert_eq!(first.allocation, crossing);
    let before = space.admitted_snapshot().reusable_bytes().unwrap();
    let plan = plan_persistence(&space, eoa, &[], &[], 140, widths, reuse_only)
        .unwrap()
        .unwrap();
    assert!(!plan.tail.appended);
    assert_eq!(plan.tail.allocation, extent(generic.start().get(), 2_489));
    assert_eq!(plan.tail.final_eoa, eoa);
    assert_eq!(
        plan.tail.post.admitted_snapshot().reusable_bytes().unwrap(),
        ByteLength::new(before.get() - 2_489)
    );
    assert_eq!(space.admitted_snapshot().reusable_bytes().unwrap(), before);
    assert_complete_capacity(&plan, widths, 140);
    assert!(plan.tail.reservation_request().is_block());
    assert_eq!(plan.tail.reservation_request().size().get(), 2_489);
    let reserved = space
        .reservation_batch(1)
        .reserve(plan.tail.reservation_request())
        .unwrap();
    assert_eq!(reserved.extent(), plan.tail.allocation);
    assert_eq!(
        space.admitted_snapshot().sections().collect::<Vec<_>>(),
        plan.tail
            .post
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>()
    );
}
