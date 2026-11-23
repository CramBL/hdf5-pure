use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
use hdf5_pure_core::{BaseAddress, FormatError, PhysicalAddress, StoredAddress};
use hdf5_pure_format::__private::{
    ExtensibleArray, ExtensibleArrayHeader, ExtensibleArrayHeaderLocator, ExtensibleArrayStats,
    FormatWidths,
};
use rstest::rstest;
use test_util::checksum;
use test_util::extensible_array::{
    CHECKSUM_BYTES, Client, Creation, Header, HeaderStats, PREFIX_LEN, STATISTIC_COUNT, Width,
};

#[rstest]
#[case::two_byte_lengths(Width::Two, 0x8000)]
#[case::four_byte_lengths(Width::Four, 0x8012_3400)]
#[case::eight_byte_lengths(Width::Eight, 0x8123_4567_89ab_cd00)]
fn both_read_paths_retain_stored_statistics(
    #[case] lengths: Width,
    #[case] high: u64,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
) {
    let fields = HeaderStats {
        secondary_block_count: 3,
        secondary_block_size: high | 0x2b,
        data_block_count: 17,
        data_block_size: high | 0x4d,
        max_index_set: 0x0123,
        element_count: 0x0234,
    };
    let fixture = Header::new(offsets, lengths, Client::Unfiltered, Creation::Small)
        .with_index(INDEX_ADDRESS, fields)
        .unwrap();
    let body = fixture.build();
    let image = surrounded(&body, &[0x93; 9]);
    let widths = FormatWidths::from_sizes(offsets.get(), lengths.get()).unwrap();

    let contiguous =
        ExtensibleArrayHeader::parse(&image, HEADER_OFFSET, offsets.get(), lengths.get()).unwrap();
    let array = parse_region(&image, widths).unwrap();
    assert_eq!(array.header(), &contiguous);
    assert_eq!(
        array.range(),
        FileRange::new(
            PhysicalAddress::new(BASE + HEADER_OFFSET as u64),
            body.len() as u64,
        )
        .unwrap(),
    );
    let index = array.index_block().unwrap().parse_range();
    assert_eq!(index.start(), PhysicalAddress::new(BASE + INDEX_ADDRESS));
    assert!(index.start().get() > BASE + image.len() as u64);

    let expected = ExtensibleArrayStats {
        nsuper_blks: fields.secondary_block_count,
        super_blk_size: fields.secondary_block_size,
        ndata_blks: fields.data_block_count,
        data_blk_size: fields.data_block_size,
        max_idx_set: fields.max_index_set,
        nelmts: fields.element_count,
    };
    let statistics_len = STATISTIC_COUNT * usize::from(lengths.get());
    for header in [&contiguous, array.header()] {
        let statistics = header.stored_statistics();
        assert_eq!(*statistics, expected);
        let mut output = vec![0xc8; statistics_len + 3];
        assert_eq!(
            statistics
                .encode_fields(&mut output[1..], widths.lengths)
                .unwrap(),
            statistics_len,
        );
        let mut expected = vec![0xc8; output.len()];
        expected[1..1 + statistics_len]
            .copy_from_slice(&body[PREFIX_LEN..PREFIX_LEN + statistics_len]);
        assert_eq!(output, expected);
    }
}

#[rstest]
#[case::prefix(PREFIX_LEN - 1)]
#[case::statistics(STATISTICS_END - 1)]
#[case::address(ADDRESS_END - 1)]
#[case::checksum(ADDRESS_END + CHECKSUM_BYTES - 1)]
fn an_incomplete_header_reports_the_whole_required_extent(#[case] available: usize) {
    let (body, widths) = header_with_short_lengths();
    let image = surrounded(&body[..available], &[]);
    assert_eq!(
        ExtensibleArrayHeader::parse_with_widths(&image, HEADER_OFFSET, widths).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: HEADER_OFFSET + body.len(),
            available: HEADER_OFFSET + available,
        }
    );
    assert_eq!(
        parse_region(&image, widths).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: BASE + HEADER_OFFSET as u64,
            required_length: body.len() as u64,
            provided_start: BASE,
            provided_length: (HEADER_OFFSET + available) as u64,
        },
    );
}

#[test]
fn a_start_past_the_input_reports_the_required_end() {
    let (body, widths) = header_with_short_lengths();
    assert_eq!(
        ExtensibleArrayHeader::parse_with_widths(&[], HEADER_OFFSET, widths).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: HEADER_OFFSET + body.len(),
            available: 0,
        },
    );
}

#[rstest]
#[case::signature_before_version(&[(0, b'X'), (VERSION_AT, 1)], FormatError::InvalidExtensibleArraySignature)]
#[case::version_before_checksum(&[(VERSION_AT, 1)], FormatError::InvalidExtensibleArrayVersion(1))]
fn checks_signature_then_version_before_checksum(
    #[case] mutations: &[(usize, u8)],
    #[case] expected: FormatError,
) {
    let (mut body, widths) = header_with_short_lengths();
    for &(at, value) in mutations {
        body[at] = value;
    }
    assert_both_errors(&surrounded(&body, &[]), widths, expected);
}

#[rstest]
#[case::client_before_creation(&[(CLIENT_AT, 2), (MAX_INDEX_BITS_AT, 0)], FormatError::InvalidExtensibleArrayClient(2))]
#[case::creation(&[(MAX_INDEX_BITS_AT, 0)], FormatError::InvalidExtensibleArrayGeometry)]
fn checks_client_before_creation_geometry(
    #[case] mutations: &[(usize, u8)],
    #[case] expected: FormatError,
) {
    let (mut body, widths) = header_with_short_lengths();
    for &(at, value) in mutations {
        body[at] = value;
    }
    let length = body.len();
    checksum::restamp(&mut body, 0, length);
    assert_both_errors(&surrounded(&body, &[]), widths, expected);
}

#[cfg(feature = "checksum")]
#[rstest]
#[case::client(CLIENT_AT, 2)]
#[case::statistics(STATISTICS_END - 1, 0x81)]
#[case::index_address(ADDRESS_END - 1, 0x3a)]
fn verification_covers_the_complete_header_body(#[case] at: usize, #[case] byte: u8) {
    let (mut body, widths) = header_with_short_lengths();
    let checksum_at = body.len() - CHECKSUM_BYTES;
    let expected = test_util::bytes::u32_at(&body, checksum_at);
    body[at] = byte;
    let computed = checksum::lookup3(&body[..checksum_at]);
    assert_both_errors(
        &surrounded(&body, &[0xe9; 10]),
        widths,
        FormatError::ChecksumMismatch { expected, computed },
    );
}

#[test]
fn index_resolution_checks_the_base_plus_the_stored_address() {
    let widths = FormatWidths::from_sizes(4, 2).unwrap();
    let body = Header::new(
        Width::Four,
        Width::Two,
        Client::Unfiltered,
        Creation::Canonical,
    )
    .with_index(INDEX_ADDRESS, ZERO_STATISTICS)
    .unwrap()
    .build();
    let base = u64::MAX - 512;
    let locator = ExtensibleArrayHeaderLocator::new(
        widths,
        BaseAddress::new(base),
        StoredAddress::new(HEADER_OFFSET as u64),
    );
    let region = FileRegion::new(locator.parse_range().unwrap(), &body).unwrap();
    assert_eq!(
        locator.parse(region).unwrap_err(),
        FormatError::OffsetOverflow {
            offset: INDEX_ADDRESS,
            length: base
        },
    );
}

fn header_with_short_lengths() -> (Vec<u8>, FormatWidths) {
    let fixture = Header::new(
        Width::Four,
        Width::Two,
        Client::Unfiltered,
        Creation::Canonical,
    );
    (fixture.build(), FormatWidths::from_sizes(4, 2).unwrap())
}

fn surrounded(header: &[u8], suffix: &[u8]) -> Vec<u8> {
    let mut image = vec![0xa5; HEADER_OFFSET];
    image.extend_from_slice(header);
    image.extend_from_slice(suffix);
    image
}

fn parse_region(image: &[u8], widths: FormatWidths) -> Result<ExtensibleArray, FormatError> {
    let range = FileRange::new(PhysicalAddress::new(BASE), image.len() as u64).unwrap();
    let region = FileRegion::new(range, image).unwrap();
    let locator = ExtensibleArrayHeaderLocator::new(
        widths,
        BaseAddress::new(BASE),
        StoredAddress::new(HEADER_OFFSET as u64),
    );
    locator.parse(region)
}

fn assert_both_errors(image: &[u8], widths: FormatWidths, expected: FormatError) {
    assert_eq!(
        ExtensibleArrayHeader::parse_with_widths(image, HEADER_OFFSET, widths).unwrap_err(),
        expected,
    );
    assert_eq!(parse_region(image, widths).unwrap_err(), expected);
}

const BASE: u64 = 1 << 40;
const HEADER_OFFSET: usize = 19;
const INDEX_ADDRESS: u64 = 4096;

// "The Extensible Array Index" of the format specification, version 4.0, defines the version,
// client ID, element size, and maximum element bit count after the signature.
const VERSION_AT: usize = 4;
const CLIENT_AT: usize = VERSION_AT + 1;
const MAX_INDEX_BITS_AT: usize = CLIENT_AT + 2;
const STATISTICS_END: usize = PREFIX_LEN + STATISTIC_COUNT * 2;
const ADDRESS_END: usize = STATISTICS_END + 4;
const ZERO_STATISTICS: HeaderStats = HeaderStats {
    secondary_block_count: 0,
    secondary_block_size: 0,
    data_block_count: 0,
    data_block_size: 0,
    max_index_set: 0,
    element_count: 0,
};
