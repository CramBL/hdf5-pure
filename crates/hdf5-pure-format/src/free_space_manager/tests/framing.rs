use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::free_space;

use super::*;
use crate::address::StoredAddressExt;
use crate::free_space_manager::file_header::{
    FILE_FSM_CLIENT_ID, FILE_FSM_EXPAND_PCT, FILE_FSM_NUM_CLASSES, FILE_FSM_SHRINK_PCT,
};
use crate::free_space_manager::{
    self, FILE_FSM_MAX_SECTION_SIZE, FreeSpaceManagerHeader, SECTION_CLASS_SIMPLE, test_support,
};

#[rstest]
#[case::header(true)]
#[case::section_list(false)]
fn checked_regions_require_checksums(#[case] header_corrupt: bool) {
    let widths = test_support::widths(8, 8);
    let header_address = StoredAddress::new(619);
    let header_ref = FreeSpaceManagerHeaderRef::new(widths, PhysicalAddress::new(619)).unwrap();
    let mut header_bytes = free_space::single_section_header();
    let mut list_bytes = free_space::single_section_info();
    let frame = header_ref
        .parse_checked(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
        .unwrap();
    let list_ref = frame
        .section_list_ref(header_address, PhysicalAddress::new(701))
        .unwrap();
    assert_eq!(
        list_ref
            .parse_checked(FileRegion::new(list_ref.parse_range(), &list_bytes).unwrap())
            .unwrap()
            .records(),
        &list_bytes[13..list_bytes.len() - 4]
    );
    let bytes = if header_corrupt {
        &mut header_bytes
    } else {
        &mut list_bytes
    };
    let last = bytes.len() - 4;
    let computed = checksum::jenkins_lookup3(&bytes[..last]);
    bytes[last] ^= 1;
    let error = if header_corrupt {
        header_ref
            .parse_checked(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
            .unwrap_err()
    } else {
        list_ref
            .parse_checked(FileRegion::new(list_ref.parse_range(), &list_bytes).unwrap())
            .unwrap_err()
    };
    assert_eq!(
        error,
        FormatError::ChecksumMismatch {
            expected: computed ^ 1,
            computed
        }
    );
}

#[rstest]
#[case::two(test_support::widths(2, 2))]
#[case::four(test_support::widths(4, 4))]
#[case::eight(test_support::widths(8, 8))]
fn framing_classifies_undefined_section_list_at_its_width(#[case] widths: FormatWidths) {
    let (header_bytes, _) = free_space_manager::serialize_free_space_manager(
        widths,
        &[],
        StoredAddress::new(32),
        StoredAddress::undefined(widths.offsets),
        SECTION_CLASS_SIMPLE,
    )
    .unwrap();
    let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header_bytes).unwrap();
    assert_eq!(frame.section_list_addr(), DecodedStoredAddress::Undefined);
    assert_eq!(
        FreeSpaceManagerHeader::from_frame(&frame)
            .unwrap()
            .section_list_addr(),
        StoredAddress::undefined(widths.offsets)
    );
    assert_eq!(
        frame
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(64))
            .unwrap_err(),
        FormatError::InvalidFreeSpaceManager("the header stores no section list address".into())
    );
}

#[test]
fn common_framing_retains_client_specific_counts() {
    let mut bytes = free_space::single_section_header();
    bytes[5] = test_support::FRACTAL_HEAP_FSM_CLIENT_ID;
    test_util::bytes::set_slice_at(&mut bytes, 14, &4u64.to_le_bytes());
    test_util::bytes::set_slice_at(&mut bytes, 30, &3u64.to_le_bytes());
    test_util::bytes::set_slice_at(&mut bytes, 38, &FRACTAL_HEAP_FSM_NUM_CLASSES.to_le_bytes());
    let len = bytes.len();
    test_util::checksum::restamp(&mut bytes, 0, len);
    let frame = FreeSpaceManagerHeaderFrame::parse(test_support::widths(8, 8), &bytes).unwrap();
    assert_eq!(frame.total_sections(), 4);
    assert_eq!(frame.serialized_sections(), 1);
    assert_eq!(frame.unserialized_sections(), 3);
    assert_eq!(frame.section_classes(), FRACTAL_HEAP_FSM_NUM_CLASSES);
}

#[test]
fn common_framing_preserves_record_class_rejected_by_file_client() {
    let mut bytes = free_space::single_section_info();
    bytes[30] = 0xFF;
    let len = bytes.len();
    test_util::checksum::restamp(&mut bytes, 0, len);
    let reference = single_frame()
        .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
        .unwrap();
    assert_eq!(
        reference.parse(region(701, &bytes)).unwrap().records(),
        &bytes[13..31]
    );
    assert_eq!(
        free_space_manager::parse_section_info(
            test_support::widths(8, 8),
            &bytes,
            StoredAddress::new(619),
            &FreeSpaceManagerHeader::from_frame(&single_frame()).unwrap()
        ),
        Err(FormatError::InvalidFreeSpaceSectionClass {
            offset: 22,
            class_id: 255,
        })
    );
}

#[test]
fn section_list_region_ignores_bytes_after_used_range() {
    let mut bytes = free_space::single_section_info();
    bytes.extend([0xFF; 64]);
    let frame = FreeSpaceManagerHeaderFrame {
        section_list_allocated: bytes.len() as u64,
        ..single_frame()
    };
    let reference = frame
        .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
        .unwrap();
    assert_eq!(
        reference.parse(region(701, &bytes)).unwrap().records(),
        &bytes[13..31]
    );
}

#[rstest]
#[case::short(619, 81)]
#[case::shifted(620, 82)]
fn header_region_rejects_insufficient_bytes(#[case] start: u64, #[case] length: usize) {
    let reference =
        FreeSpaceManagerHeaderRef::new(test_support::widths(8, 8), PhysicalAddress::new(619))
            .unwrap();
    let bytes = vec![0; length];
    assert_eq!(
        reference.parse(region(start, &bytes)).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: 619,
            required_length: 82,
            provided_start: start,
            provided_length: length as u64,
        }
    );
}

#[test]
fn header_region_accepts_enclosing_bytes() {
    let mut bytes = vec![0; 3];
    bytes.extend(free_space::single_section_header());
    bytes.extend([0; 7]);
    let reference =
        FreeSpaceManagerHeaderRef::new(test_support::widths(8, 8), PhysicalAddress::new(619))
            .unwrap();
    assert_eq!(
        reference.parse(region(616, &bytes)).unwrap(),
        FreeSpaceManagerHeaderFrame {
            client_id: FILE_FSM_CLIENT_ID,
            total_space: 1600,
            total_sections: 1,
            serialized_sections: 1,
            unserialized_sections: 0,
            section_classes: FILE_FSM_NUM_CLASSES,
            shrink_percent: FILE_FSM_SHRINK_PCT,
            expand_percent: FILE_FSM_EXPAND_PCT,
            addr_space_bits: AddressSpaceBits::new(63).unwrap(),
            max_section_size: FILE_FSM_MAX_SECTION_SIZE,
            section_list_addr: DecodedStoredAddress::Defined(StoredAddress::new(701)),
            section_list_used: 35,
            section_list_allocated: 35,
            widths: test_support::widths(8, 8),
        }
    );
}

#[test]
fn header_reference_rejects_range_overflow() {
    assert_eq!(
        FreeSpaceManagerHeaderRef::new(
            test_support::widths(8, 8),
            PhysicalAddress::new(u64::MAX - 81)
        )
        .unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX - 81,
            length: 82
        }
    );
}

#[rstest]
#[case::spare(64)]
#[case::large(u64::MAX - 35)]
fn section_list_region_requires_only_used_bytes(#[case] spare: u64) {
    let bytes = free_space::single_section_info();
    let frame = FreeSpaceManagerHeaderFrame {
        section_list_allocated: bytes.len() as u64 + spare,
        ..single_frame()
    };
    let reference = frame
        .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(0))
        .unwrap();
    assert_eq!(reference.range().length(), bytes.len() as u64 + spare);
    assert_eq!(reference.parse_range().length(), bytes.len() as u64);
    assert_eq!(
        reference.parse(region(0, &bytes)).unwrap().records(),
        &bytes[13..31]
    );
}

#[test]
fn section_list_reference_checks_complete_allocation_overflow() {
    let frame = FreeSpaceManagerHeaderFrame {
        section_list_allocated: 64,
        ..single_frame()
    };
    assert_eq!(
        frame
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(u64::MAX - 35))
            .unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX - 35,
            length: 64
        }
    );
}

#[test]
fn section_list_reference_rejects_missing_prefix_geometry() {
    let frame = FreeSpaceManagerHeaderFrame {
        section_list_used: 16,
        ..single_frame()
    };
    assert_eq!(
        frame
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
            .unwrap_err(),
        FormatError::InvalidFreeSpaceManager(
            "the section list uses 16 bytes, fewer than its 17-byte prefix and checksum".into()
        )
    );
}

#[rstest]
#[case::two_undefined(
    test_support::widths(2, 2),
    0xFFFF,
    "the section list's expected header address is undefined"
)]
#[case::four_undefined(
    test_support::widths(4, 4),
    0xFFFF_FFFF,
    "the section list's expected header address is undefined"
)]
#[case::eight_undefined(
    test_support::widths(8, 8),
    u64::MAX,
    "the section list's expected header address is undefined"
)]
fn section_list_reference_rejects_undefined_header_address(
    #[case] widths: FormatWidths,
    #[case] address: u64,
    #[case] reason: &str,
) {
    let frame = FreeSpaceManagerHeaderFrame {
        widths,
        ..single_frame()
    };
    assert_eq!(
        frame
            .section_list_ref(StoredAddress::new(address), PhysicalAddress::new(701))
            .unwrap_err(),
        FormatError::InvalidFreeSpaceManager(reason.into())
    );
}

#[rstest]
#[case::two(test_support::widths(2, 2), 0x1_0000)]
#[case::four(test_support::widths(4, 4), 0x1_0000_0000)]
#[case::two_wide_sentinel(test_support::widths(2, 2), u64::MAX)]
#[case::four_wide_sentinel(test_support::widths(4, 4), u64::MAX)]
fn section_list_reference_rejects_header_address_outside_width(
    #[case] widths: FormatWidths,
    #[case] address: u64,
) {
    let frame = FreeSpaceManagerHeaderFrame {
        widths,
        ..single_frame()
    };
    assert_eq!(
        frame
            .section_list_ref(StoredAddress::new(address), PhysicalAddress::new(701))
            .unwrap_err(),
        FormatError::AddressTooLarge {
            address,
            offset_size: widths.offsets.get()
        }
    );
}

#[rstest]
#[case::short(701, 34)]
#[case::shifted(702, 35)]
fn section_list_region_rejects_insufficient_bytes(#[case] start: u64, #[case] length: usize) {
    let reference = single_frame()
        .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
        .unwrap();
    assert_eq!(
        reference
            .parse(region(start, &vec![0; length]))
            .unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: 701,
            required_length: 35,
            provided_start: start,
            provided_length: length as u64,
        }
    );
}

#[rstest]
#[case::signature(0, vec![b'X'], "the section list signature is b\"XSSE\", not b\"FSSE\"")]
#[case::backlink(5, 736u64.to_le_bytes().to_vec(), "the section list belongs to the header at 736, not to the header at 619")]
#[case::undefined_backlink(5, u64::MAX.to_le_bytes().to_vec(), "the section list belongs to the header at 18446744073709551615, not to the header at 619")]
fn section_list_region_rejects_invalid_framing(
    #[case] at: usize,
    #[case] value: Vec<u8>,
    #[case] reason: &str,
) {
    let mut bytes = free_space::single_section_info();
    test_util::bytes::set_slice_at(&mut bytes, at, &value);
    let len = bytes.len();
    test_util::checksum::restamp(&mut bytes, 0, len);
    let reference = single_frame()
        .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
        .unwrap();
    assert_eq!(
        reference.parse(region(701, &bytes)).unwrap_err(),
        FormatError::InvalidFreeSpaceManager(reason.into())
    );
}

#[cfg(feature = "checksum")]
#[rstest]
#[case::header(true)]
#[case::section_list(false)]
fn region_parsing_rejects_invalid_checksum(#[case] header: bool) {
    if header {
        let mut bytes = free_space::single_section_header();
        let computed = checksum::jenkins_lookup3(&bytes[..78]);
        bytes[78] ^= 0xFF;
        let reference =
            FreeSpaceManagerHeaderRef::new(test_support::widths(8, 8), PhysicalAddress::new(619))
                .unwrap();
        assert_eq!(
            reference.parse(region(619, &bytes)).unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: computed ^ 0xFF,
                computed
            }
        );
    } else {
        let mut bytes = free_space::single_section_info();
        let expected = checksum::jenkins_lookup3(&bytes[..31]);
        bytes[31] ^= 0xFF;
        let reference = single_frame()
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
            .unwrap();
        assert_eq!(
            reference.parse(region(701, &bytes)).unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: expected ^ 0xFF,
                computed: expected
            }
        );
    }
}

#[rstest]
#[case::header(true)]
#[case::section_list(false)]
fn unsupported_manager_versions_report_the_version(#[case] header: bool) {
    let mut bytes = if header {
        free_space::single_section_header()
    } else {
        free_space::single_section_info()
    };
    bytes[4] = 1;
    let length = bytes.len();
    test_util::checksum::restamp(&mut bytes, 0, length);
    if header {
        let reference =
            FreeSpaceManagerHeaderRef::new(test_support::widths(8, 8), PhysicalAddress::new(619))
                .unwrap();
        assert_eq!(
            reference.parse_checked(region(619, &bytes)).unwrap_err(),
            FormatError::UnsupportedVersion(1)
        );
        assert_eq!(
            FreeSpaceManagerHeader::parse(test_support::widths(8, 8), &bytes).unwrap_err(),
            FormatError::UnsupportedVersion(1)
        );
    } else {
        let frame = single_frame();
        let reference = frame
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
            .unwrap();
        assert_eq!(
            reference.parse_checked(region(701, &bytes)).unwrap_err(),
            FormatError::UnsupportedVersion(1)
        );
        assert_eq!(
            free_space_manager::parse_section_info(
                test_support::widths(8, 8),
                &bytes,
                StoredAddress::new(619),
                &FreeSpaceManagerHeader::from_frame(&frame).unwrap()
            )
            .unwrap_err(),
            FormatError::UnsupportedVersion(1)
        );
    }
}

fn single_frame() -> FreeSpaceManagerHeaderFrame {
    FreeSpaceManagerHeaderFrame::parse(
        test_support::widths(8, 8),
        &free_space::single_section_header(),
    )
    .unwrap()
}

fn region(start: u64, bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(start), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
}

const FRACTAL_HEAP_FSM_NUM_CLASSES: u16 = 4;
