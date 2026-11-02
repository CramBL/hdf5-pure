use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::AdmittedFreeSnapshot;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeList;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::free_space;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::free_space_manager;
use crate::free_space_manager::ManagerProjection;
use crate::source::BytesSource;
use crate::width::FormatWidths;

#[rstest]
#[case::two_byte(
    2,
    2,
    485,
    519,
    3648,
    free_space::two_byte_widths_header(),
    free_space::two_byte_widths_section_info()
)]
#[case::four_byte(
    4,
    4,
    535,
    585,
    3648,
    free_space::four_byte_widths_header(),
    free_space::four_byte_widths_section_info()
)]
#[case::four_byte_lengths(
    8,
    4,
    627,
    681,
    3648,
    free_space::four_byte_lengths_header(),
    free_space::four_byte_lengths_section_info()
)]
#[case::eight_byte(
    8,
    8,
    619,
    701,
    2848,
    free_space::single_section_header(),
    free_space::single_section_info()
)]
fn measured_allocations_preserve_native_manager_sections(
    #[case] offsets: u8,
    #[case] lengths: u8,
    #[case] header_address: u64,
    #[case] list_address: u64,
    #[case] free_address: u64,
    #[case] expected_header: Vec<u8>,
    #[case] expected_list: Vec<u8>,
) {
    let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
    let free = FreeList::from_sorted([extent(free_address, 1600)]).unwrap();
    let projection = ManagerProjection::new(AdmittedFreeSnapshot::flat(&free), widths).unwrap();
    let measured = projection.block_lengths().next().unwrap();
    assert_eq!(measured[0], expected_header.len() as u64);
    let native_header = format::FreeSpaceManagerHeader::parse(widths, &expected_header).unwrap();
    assert_eq!(native_header.total_space(), 1600);
    assert_eq!(native_header.total_sections(), 1);
    let native_sections = format::parse_section_info(
        widths,
        &expected_list,
        StoredAddress::new(header_address),
        &native_header,
    )
    .unwrap();
    assert_eq!(
        native_sections,
        [format::FreeSection {
            addr: StoredAddress::new(free_address),
            size: 1600,
        }]
    );
    let plan = projection
        .place(|[header, list]| Ok([extent(header_address, header), extent(list_address, list)]))
        .unwrap();
    let encoded = plan.encode_allocations().next().unwrap().unwrap();
    assert_eq!(encoded.header_address.get(), header_address);
    assert_eq!(encoded.section_list_address.get(), list_address);
    assert_eq!(
        [
            encoded.header.len() as u64,
            encoded.section_list.len() as u64
        ],
        measured
    );
    let header = format::FreeSpaceManagerHeader::parse(widths, &encoded.header).unwrap();
    assert_eq!(header.total_space(), native_header.total_space());
    assert_eq!(header.total_sections(), native_header.total_sections());
    assert_eq!(header.section_list_addr().get(), list_address);
    assert_eq!(header.section_list_used(), measured[1]);
    assert_eq!(header.section_list_allocated(), measured[1]);
    assert_eq!(
        format::parse_section_info(
            widths,
            &encoded.section_list,
            encoded.header_address,
            &header
        )
        .unwrap(),
        native_sections
    );
}

#[test]
fn independent_placement_matches_known_bytes() {
    let plan = flat_projection(FormatWidths::from_sizes(8, 8).unwrap())
        .place(|[header, list]| Ok([extent(619, header), extent(1701, list)]))
        .unwrap();
    let block = plan.encode_allocations().next().unwrap().unwrap();
    let (header, list) = free_space::raw_manager_eight(619, 1701, &[(2848, 1600)]);
    assert_eq!(block.header_address.get(), 619);
    assert_eq!(block.section_list_address.get(), 1701);
    assert_eq!(block.header, header);
    assert_eq!(block.section_list, list);
}

#[rstest]
#[case::small(35)]
#[case::large(1 << 33)]
fn planning_complete_capacity_retains_u64_lengths(#[case] length: u64) {
    let plan = flat_projection(FormatWidths::from_sizes(8, 8).unwrap())
        .place(|[header, _]| Ok([extent(512, header), extent(1 << 35, length)]))
        .unwrap();
    assert_eq!(plan.end().get(), (1 << 35) + length);
    assert_eq!(plan.blocks[0].header, extent(512, 82));
    assert_eq!(plan.blocks[0].section_list, extent(1 << 35, length));
    assert_eq!(plan.blocks[0].section_list_len, 35);
}

#[rstest]
#[case::exact(0, 0)]
#[case::padded(17, 0)]
#[case::userblock(17, 512)]
fn independent_allocations_retain_complete_capacity_and_manager_identity(
    #[case] padding: u64,
    #[case] base: u64,
) {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let projection = paged_projection(widths);
    let mut allocations = [512, 1024, 1536].into_iter().zip([2560, 3072, 2048]);
    let plan = projection
        .place(|[header, list]| {
            let (header_address, list_address) = allocations.next().unwrap();
            Ok([
                extent(header_address, header),
                extent(list_address, list + padding),
            ])
        })
        .unwrap();
    assert_eq!(plan.end_of_managers.get(), 3107 + padding);
    assert_eq!(
        plan.slots.map(|address| address.get()),
        [
            512,
            u64::MAX,
            1024,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            1536,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX
        ]
    );
    assert_eq!(
        plan.blocks
            .iter()
            .map(|block| block.kind)
            .collect::<Vec<_>>(),
        [
            ManagerKind::PagedSmallMetadata,
            ManagerKind::PagedSmallRaw,
            ManagerKind::PagedLargeGeneric
        ]
    );
    let encoded = plan
        .encode_allocations()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut bytes = vec![0; (base + 4096) as usize];
    for (block, (address, size)) in encoded.iter().zip([(100, 200), (5000, 200), (8192, 4096)]) {
        assert_eq!(block.header.len(), 82);
        assert_eq!(block.section_list.len() as u64, 35 + padding);
        let header = format::FreeSpaceManagerHeader::parse(widths, &block.header).unwrap();
        assert_eq!(header.section_list_used(), 35 + padding);
        assert_eq!(header.section_list_allocated(), 35 + padding);
        assert_eq!(header.total_sections(), 1);
        assert_eq!(header.total_space(), size);
        assert_eq!(
            format::parse_section_info(widths, &block.section_list, block.header_address, &header)
                .unwrap(),
            [format::FreeSection {
                addr: StoredAddress::new(address),
                size
            }]
        );
        let header_start = (base + block.header_address.get()) as usize;
        let list_start = (base + block.section_list_address.get()) as usize;
        bytes[header_start..header_start + block.header.len()].copy_from_slice(&block.header);
        bytes[list_start..list_start + block.section_list.len()]
            .copy_from_slice(&block.section_list);
    }
    let managers =
        [512, 1024, 1536].map(|address| DecodedStoredAddress::Defined(StoredAddress::new(address)));
    let (sections, ranges) = free_space_manager::read_persisted_sections_source(
        &BytesSource::new(&bytes),
        widths,
        BaseAddress::new(base),
        &managers,
    )
    .unwrap();
    assert_eq!(
        sections,
        [
            format::FreeSection {
                addr: StoredAddress::new(100),
                size: 200
            },
            format::FreeSection {
                addr: StoredAddress::new(5000),
                size: 200
            },
            format::FreeSection {
                addr: StoredAddress::new(8192),
                size: 4096
            },
        ]
    );
    assert_eq!(
        ranges,
        [
            (512, 82),
            (2560, 35 + padding),
            (1024, 82),
            (3072, 35 + padding),
            (1536, 82),
            (2048, 35 + padding),
        ]
        .map(|(start, length)| FileRange::new(PhysicalAddress::new(base + start), length).unwrap())
    );
}

#[rstest]
#[case::empty(0)]
#[case::nonzero(4096)]
fn empty_manager_placement_preserves_the_contiguous_start(#[case] start: u64) {
    let widths = FormatWidths::from_sizes(4, 4).unwrap();
    let free = FreeList::new();
    let projection = ManagerProjection::new(AdmittedFreeSnapshot::flat(&free), widths).unwrap();
    assert_eq!(projection.block_lengths().len(), 0);
    let plan = projection.into_plan(StoredAddress::new(start)).unwrap();
    assert_eq!(plan.end_of_managers.get(), start);
    assert!(
        plan.slots
            .iter()
            .all(|address| address.is_undefined(widths.offsets))
    );
    assert_eq!(
        plan.encode(StoredAddress::new(start + 7)).unwrap().count(),
        0
    );
    let projection = ManagerProjection::new(AdmittedFreeSnapshot::flat(&free), widths).unwrap();
    assert_eq!(
        projection
            .place(|_| panic!("an empty projection requests no allocations"))
            .unwrap()
            .encode_allocations()
            .count(),
        0
    );
}

#[rstest]
#[case::header_short(
    500,
    81,
    1000,
    35,
    "manager allocations differ from their measured lengths"
)]
#[case::header_surplus(
    500,
    83,
    1000,
    35,
    "manager allocations differ from their measured lengths"
)]
#[case::list_short(
    500,
    82,
    1000,
    34,
    "manager allocations differ from their measured lengths"
)]
#[case::overlap(500, 82, 581, 35, "manager allocations overlap")]
#[case::header_in_free_space(
    2850,
    82,
    1000,
    35,
    "a manager allocation overlaps a projected free section"
)]
#[case::list_capacity_in_free_space(
    500,
    82,
    2800,
    100,
    "a manager allocation overlaps a projected free section"
)]
fn placement_rejects_invalid_complete_allocations(
    #[case] header: u64,
    #[case] header_len: u64,
    #[case] list: u64,
    #[case] list_len: u64,
    #[case] reason: &str,
) {
    let error = flat_projection(FormatWidths::from_sizes(8, 8).unwrap())
        .place(|_| Ok([extent(header, header_len), extent(list, list_len)]))
        .map(|_| ())
        .unwrap_err();
    assert_eq!(error, FormatError::Internal(reason.into()));
}

#[test]
fn placement_rejects_overlap_between_distinct_managers() {
    let error = paged_projection(FormatWidths::from_sizes(8, 8).unwrap())
        .place(|[header, list]| Ok([extent(512, header), extent(2048, list)]))
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal("manager allocations overlap".into())
    );
}

#[rstest]
#[case::address_width(65_520, 1000, 35, FormatError::AddressTooLarge { address: 65_554, offset_size: 2 })]
#[case::undefined_header(65_535, 1000, 35, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::undefined_list(500, 65_535, 35, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::list_end_width(500, 65_530, 35, FormatError::AddressTooLarge { address: 65_565, offset_size: 2 })]
#[case::length_width(500, 1000, 65_536, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
fn placement_rejects_unrepresentable_allocations(
    #[case] header: u64,
    #[case] list: u64,
    #[case] list_len: u64,
    #[case] expected: FormatError,
) {
    let error = flat_projection(FormatWidths::from_sizes(2, 2).unwrap())
        .place(|[header_len, _]| Ok([extent(header, header_len), extent(list, list_len)]))
        .map(|_| ())
        .unwrap_err();
    assert_eq!(error, expected);
}

#[test]
fn complete_allocations_can_end_at_the_widths_undefined_address() {
    let plan = flat_projection(FormatWidths::from_sizes(2, 2).unwrap())
        .place(|[header, list]| Ok([extent(500, header), extent(65_535 - list, list)]))
        .unwrap();
    assert_eq!(plan.end_of_managers.get(), 65_535);
    let encoded = plan.encode_allocations().next().unwrap().unwrap();
    assert_eq!(
        encoded.section_list_address.get() + encoded.section_list.len() as u64,
        65_535
    );
}

#[test]
fn surplus_capacity_cannot_cover_a_projected_free_section() {
    let plan = flat_projection(FormatWidths::from_sizes(8, 8).unwrap())
        .into_plan(StoredAddress::new(500))
        .unwrap();
    let error = plan
        .encode(StoredAddress::new(3000))
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal("a manager allocation overlaps a projected free section".into())
    );
}

#[test]
fn surplus_capacity_cannot_extend_a_nonfinal_section_list() {
    let mut locations = [512, 1024, 1536].into_iter().zip([2560, 3072, 2048]);
    let plan = paged_projection(FormatWidths::from_sizes(8, 8).unwrap())
        .place(|[header, list]| {
            let (header_start, list_start) = locations.next().unwrap();
            Ok([extent(header_start, header), extent(list_start, list)])
        })
        .unwrap();
    let error = plan
        .encode(StoredAddress::new(3500))
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::Internal("the final manager section list does not end its allocation".into())
    );
}

#[test]
fn placement_preserves_the_callers_error() {
    let mut calls = 0;
    let error = paged_projection(FormatWidths::from_sizes(8, 8).unwrap())
        .place(|[header, list]| {
            calls += 1;
            if calls == 2 {
                return Err(FormatError::LengthTooLarge {
                    length: 65_536,
                    length_size: 2,
                });
            }
            Ok([extent(512, header), extent(2048, list)])
        })
        .map(|_| ())
        .unwrap_err();
    assert_eq!(calls, 2);
    assert_eq!(
        error,
        FormatError::LengthTooLarge {
            length: 65_536,
            length_size: 2
        }
    );
}

fn flat_projection(widths: FormatWidths) -> ManagerProjection {
    let free = FreeList::from_sorted([extent(2848, 1600)]).unwrap();
    ManagerProjection::new(AdmittedFreeSnapshot::flat(&free), widths).unwrap()
}

fn paged_projection(widths: FormatWidths) -> ManagerProjection {
    let mut space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 0, FileSpacePageSize::DEFAULT);
    space.seed_persisted(Some(ManagerKind::PagedSmallMetadata), extent(100, 200));
    space.seed_persisted(Some(ManagerKind::PagedSmallRaw), extent(5000, 200));
    space.seed_persisted(Some(ManagerKind::PagedLargeGeneric), extent(8192, 4096));
    ManagerProjection::new(space.admitted_snapshot(), widths).unwrap()
}

fn extent(start: u64, length: u64) -> Extent {
    Extent::new(StoredAddress::new(start), length).unwrap()
}
