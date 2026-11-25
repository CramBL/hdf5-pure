use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::FileSpacePageSize;
use hdf5_pure_core::FileSpaceStrategy;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{FormatWidths, FreeSection, PreparedFreeSpaceManager};
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::PageClassEvidence;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::allocation;

#[rstest]
#[case::few_extents(16, 1)]
#[case::many_extents(4_096, 1)]
#[case::long_interior(16, 1_000_000_000)]
fn partial_page_queries_borrow_canonical_state(#[case] count: u64, #[case] stride: u64) {
    let space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::MIN,
        (0..count).map(|page| {
            (
                Some(ManagerKind::PagedSmallMetadata),
                extent(page * stride * 512 + 400, 16),
            )
        }),
    )
    .unwrap();
    let snapshot = space.admitted_snapshot();
    let query = extent(100, (count - 1) * stride * 512 + 200);
    let (copied, control) =
        allocation::measure(&format!("page_class_control_{count}_{stride}"), || {
            snapshot.sections().collect::<Vec<_>>()
        });
    assert_eq!(u64::try_from(copied.len()).unwrap(), count);
    assert!(
        copied
            .iter()
            .all(|(_, class)| *class == FreeClass::Page(PageType::Meta))
    );
    assert!(control.blocks > 0, "{control:?}");
    assert!(
        control.live_bytes >= count * u64::try_from(size_of::<(Extent, FreeClass)>()).unwrap(),
        "{control:?}"
    );
    let (class, measured) =
        allocation::measure(&format!("partial_page_class_{count}_{stride}"), || {
            snapshot.partial_page_class(query)
        });
    assert_eq!(class, PageClassEvidence::Known(PageType::Meta));
    assert_eq!(measured.blocks, 0, "{measured:?}");
    assert_eq!(measured.bytes, 0, "{measured:?}");
    assert_eq!(measured.peak_bytes, 0, "{measured:?}");
    assert_eq!(measured.live_bytes, 0, "{measured:?}");
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), ByteLength::new(length)).unwrap()
}

#[test]
fn prepared_manager_placement_and_writes_use_only_caller_buffers() {
    let widths = FormatWidths::from_sizes(8, 4).unwrap();
    let sections = (0..4096u64)
        .map(|index| FreeSection {
            addr: StoredAddress::new(0x10_0000 + 256 * index),
            size: 60 + index % 5,
        })
        .collect::<Vec<_>>();
    let first_header = StoredAddress::new(0x200);
    let first_list = StoredAddress::new(0x400);
    let second_header = StoredAddress::new(0x50_0000);
    let second_list = StoredAddress::new(0x800);
    let ((expected_header, expected_list), control) =
        allocation::measure("allocating_file_manager_writer", || {
            format::serialize_free_space_manager(
                widths,
                &sections,
                second_header,
                second_list,
                format::SECTION_CLASS_LARGE,
            )
            .unwrap()
        });
    assert!(control.blocks >= 2, "{control}");
    assert!(control.live_bytes >= (expected_header.len() + expected_list.len()) as u64);

    let plan =
        PreparedFreeSpaceManager::new(widths, sections, format::SECTION_CLASS_LARGE).unwrap();
    let mut header = vec![0xA5; plan.header_len()];
    let mut list = vec![0x5A; plan.section_list_len()];
    let ((header_len, list_len), measured) =
        allocation::measure("bounded_file_manager_writer", || {
            let first = plan.place(first_header, first_list, None).unwrap();
            first.header().write_into(&mut header).unwrap();
            first.section_list().write_into(&mut list).unwrap();

            let second = plan.place(second_header, second_list, None).unwrap();
            (
                second.header().write_into(&mut header).unwrap(),
                second.section_list().write_into(&mut list).unwrap(),
            )
        });
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(measured.peak_bytes, 0, "{measured}");
    assert_eq!(measured.live_bytes, 0, "{measured}");
    assert_eq!(
        (header_len, list_len),
        (expected_header.len(), expected_list.len())
    );
    assert_eq!(header, expected_header);
    assert_eq!(list, expected_list);
}

#[test]
fn allocating_manager_writer_bounds_peak_for_equal_sizes_at_two_byte_widths() {
    const SECTION_COUNT: usize = 4096;
    const GROUP_OVERHEAD: u64 = 1024;

    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let sections = (0..SECTION_COUNT)
        .map(|index| FreeSection {
            addr: StoredAddress::new(0x6000 + 8 * index as u64),
            size: 8,
        })
        .collect::<Vec<_>>();
    let owner = StoredAddress::new(0x100);
    let list = StoredAddress::new(0x200);
    let ((header_bytes, list_bytes), measured) =
        allocation::measure("narrow_allocating_file_manager_writer", || {
            format::serialize_free_space_manager(
                widths,
                &sections,
                owner,
                list,
                format::SECTION_CLASS_SIMPLE,
            )
            .unwrap()
        });

    let retained = u64::try_from(header_bytes.len() + list_bytes.len()).unwrap();
    let address_bytes = u64::try_from(SECTION_COUNT * size_of::<StoredAddress>()).unwrap();
    assert_eq!(measured.live_bytes, retained, "{measured}");
    assert!(
        measured.peak_bytes <= retained + address_bytes + GROUP_OVERHEAD,
        "{measured}"
    );
    let header = format::FreeSpaceManagerHeader::parse(widths, &header_bytes).unwrap();
    assert_eq!(
        format::parse_section_info(widths, &list_bytes, owner, &header).unwrap(),
        sections,
    );
}

#[test]
fn preparing_distinct_section_sizes_keeps_allocations_below_the_group_count() {
    const SECTION_COUNT: u64 = 4096;
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let sections = (1..=SECTION_COUNT)
        .map(|size| FreeSection {
            addr: StoredAddress::new(8192 * size),
            size,
        })
        .collect::<Vec<_>>();
    let (manager, measured) = allocation::measure("distinct_file_manager_groups", || {
        PreparedFreeSpaceManager::new(
            widths,
            sections.iter().copied(),
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap()
    });
    assert_eq!(manager.sections().collect::<Vec<_>>(), sections);
    assert!(measured.blocks > 0, "{measured}");
    assert!(measured.blocks < SECTION_COUNT / 2, "{measured}");
}
