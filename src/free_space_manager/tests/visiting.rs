use core::mem;

use test_util::allocation;

use super::*;

#[rstest]
#[case::tight(0, 35)]
#[case::padded(0, 51)]
#[case::userblock(512, 51)]
#[case::large_capacity(0, 1 << 33)]
fn allocation_only_recovery_preserves_reads_and_complete_ranges(
    #[case] base: u64,
    #[case] allocated: u64,
) {
    let source = manager_source(base, allocated, base + 701 + allocated);
    let ranges = read_persisted_allocations_source(
        &source,
        widths(8, 8),
        BaseAddress::new(base),
        &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
    )
    .unwrap();
    assert_eq!(
        ranges,
        vec![
            FileRange::new(PhysicalAddress::new(base + 619), 82).unwrap(),
            FileRange::new(PhysicalAddress::new(base + 701), allocated).unwrap(),
        ]
    );
    assert_eq!(
        *source.reads.borrow(),
        vec![(base + 619, 82), (base + 701, 35)]
    );
}

#[test]
fn allocation_only_recovery_retains_the_full_allocation_source_policy() {
    let source = manager_source(0, 51, 736);
    let error = read_persisted_allocations_source(
        &source,
        widths(8, 8),
        BaseAddress::ZERO,
        &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
    )
    .unwrap_err();
    assert_eq!(error, FormatError::InvalidFreeSpaceManager(
        "the section list allocation at 701, 51 bytes long, runs past the end of the file, 736 bytes long".to_owned()
    ));
    assert_eq!(*source.reads.borrow(), vec![(619, 82)]);
}

#[rstest]
#[case::one(1)]
#[case::wide_count(4_096)]
fn semantic_visiting_avoids_owned_records_and_preserves_sparse_addresses(#[case] count: usize) {
    let records = (0..count)
        .map(|index| (u64::try_from(index).unwrap() * (1 << 32), 7))
        .collect::<Vec<_>>();
    let (header, list) = free_space::raw_manager_eight(619, 701, &records);
    let header = FreeSpaceManagerHeader::parse(widths(8, 8), &header).unwrap();
    let (owned, owned_cost) = allocation::measure(&format!("manager_owned_{count}"), || {
        hdf5_pure_format::__private::parse_section_info(
            widths(8, 8),
            &list,
            StoredAddress::new(619),
            &header,
        )
        .unwrap()
    });
    assert_eq!(owned.len(), count);
    assert_eq!(owned.capacity(), count);
    assert!(owned_cost.live_bytes >= u64::try_from(count * mem::size_of::<FreeSection>()).unwrap());
    let mut visited = 0;
    let ((), visitor_cost) = allocation::measure(&format!("manager_visitor_{count}"), || {
        hdf5_pure_format::__private::visit_section_info(
            widths(8, 8),
            &list,
            StoredAddress::new(619),
            &header,
            |section| {
                assert_eq!(section, super::section(records[visited].0, 7));
                visited += 1;
            },
        )
        .unwrap();
    });
    assert_eq!(visited, count);
    assert_eq!(visitor_cost.blocks, 0, "{visitor_cost}");
    assert_eq!(visitor_cost.bytes, 0, "{visitor_cost}");
    assert_eq!(
        owned,
        records
            .iter()
            .map(|&(address, size)| section(address, size))
            .collect::<Vec<_>>()
    );
    let source = ManagerSource {
        base: 0,
        file_len: 701 + u64::try_from(list.len()).unwrap(),
        header: free_space::raw_manager_eight(619, 701, &records).0,
        sections: list,
        reads: RefCell::new(Vec::with_capacity(2)),
    };
    let managers = [DecodedStoredAddress::Defined(StoredAddress::new(619))];
    let (ranges, source_cost) = allocation::measure(&format!("manager_source_{count}"), || {
        read_persisted_allocations_source(&source, widths(8, 8), BaseAddress::ZERO, &managers)
            .unwrap()
    });
    assert_eq!(
        ranges,
        vec![
            FileRange::new(PhysicalAddress::new(619), 82).unwrap(),
            FileRange::new(
                PhysicalAddress::new(701),
                u64::try_from(source.sections.len()).unwrap()
            )
            .unwrap(),
        ]
    );
    assert_eq!(
        *source.reads.borrow(),
        vec![(619, 82), (701, source.sections.len())]
    );
    assert!(
        source_cost.bytes >= u64::try_from(source.sections.len()).unwrap(),
        "{source_cost}"
    );
    assert!(
        source_cost.bytes < u64::try_from(source.sections.len() + 1_024).unwrap(),
        "{source_cost}"
    );
}

#[test]
fn absent_manager_allocations_need_no_source_reads_or_record_storage() {
    let source = manager_source(0, 35, 736);
    let (ranges, cost) = allocation::measure("manager_source_empty", || {
        read_persisted_allocations_source(
            &source,
            widths(8, 8),
            BaseAddress::ZERO,
            &[DecodedStoredAddress::Undefined],
        )
        .unwrap()
    });
    assert!(ranges.is_empty());
    assert!(source.reads.borrow().is_empty());
    assert_eq!(cost.blocks, 0, "{cost}");
}

#[rstest]
#[case::eight(
    8,
    8,
    619,
    free_space::single_section_header(),
    free_space::single_section_info(),
    2848
)]
#[case::four(
    4,
    4,
    535,
    free_space::four_byte_widths_header(),
    free_space::four_byte_widths_section_info(),
    3648
)]
#[case::mixed(
    8,
    4,
    627,
    free_space::four_byte_lengths_header(),
    free_space::four_byte_lengths_section_info(),
    3648
)]
#[case::two(
    2,
    2,
    485,
    free_space::two_byte_widths_header(),
    free_space::two_byte_widths_section_info(),
    3648
)]
fn visiting_preserves_native_field_geometry(
    #[case] offsets: u8,
    #[case] lengths: u8,
    #[case] address: u64,
    #[case] header: Vec<u8>,
    #[case] list: Vec<u8>,
    #[case] section_address: u64,
) {
    let widths = widths(offsets, lengths);
    let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
    let mut visited = Vec::new();
    hdf5_pure_format::__private::visit_section_info(
        widths,
        &list,
        StoredAddress::new(address),
        &header,
        |section| visited.push(section),
    )
    .unwrap();
    assert_eq!(visited, vec![section(section_address, 1600)]);
    assert_eq!(
        hdf5_pure_format::__private::parse_section_info(
            widths,
            &list,
            StoredAddress::new(address),
            &header,
        )
        .unwrap(),
        visited
    );
}

#[derive(Clone, Copy, Debug)]
enum SectionFailure {
    Class,
    Prefix,
}

#[rstest]
#[case::prefix(SectionFailure::Prefix)]
#[case::late_class(SectionFailure::Class)]
fn owned_records_are_allocated_after_the_prefix_and_dropped_on_late_failure(
    #[case] failure: SectionFailure,
) {
    let count = 4_096;
    let records = (0..count)
        .map(|address| (address * 32, 7))
        .collect::<Vec<_>>();
    let (header, mut list) = free_space::raw_manager_eight(619, 701, &records);
    let header = FreeSpaceManagerHeader::parse(widths(8, 8), &header).unwrap();
    let (expected, prefix) = match failure {
        SectionFailure::Prefix => {
            list[4] = 1;
            (FormatError::UnsupportedVersion(1), 0)
        }
        SectionFailure::Class => {
            let class = list.len() - 5;
            list[class] = 3;
            let offset = class - 8;
            (
                FormatError::InvalidFreeSpaceManager(format!(
                    "the class of the section at byte {offset} of the section list is 3, not one of the 3 classes of the file client"
                )),
                usize::try_from(count - 1).unwrap(),
            )
        }
    };
    let length = list.len();
    test_util::checksum::restamp(&mut list, 0, length);
    let (owned_error, owned) =
        allocation::measure(&format!("manager_owned_failure_{failure:?}"), || {
            hdf5_pure_format::__private::parse_section_info(
                widths(8, 8),
                &list,
                StoredAddress::new(619),
                &header,
            )
            .unwrap_err()
        });
    assert_eq!(owned_error, expected);
    let mut visited = 0;
    let (visitor_error, visitor) =
        allocation::measure(&format!("manager_visit_failure_{failure:?}"), || {
            hdf5_pure_format::__private::visit_section_info(
                widths(8, 8),
                &list,
                StoredAddress::new(619),
                &header,
                |_| visited += 1,
            )
            .unwrap_err()
        });
    assert_eq!(visitor_error, expected);
    assert_eq!(visited, prefix);
    assert_eq!(owned.live_bytes, visitor.live_bytes);
    let record_bytes = match failure {
        SectionFailure::Class => count * u64::try_from(mem::size_of::<FreeSection>()).unwrap(),
        SectionFailure::Prefix => 0,
    };
    assert_eq!(owned.bytes, visitor.bytes + record_bytes);
    assert!(visitor.bytes < 1_024, "{visitor}");
}
