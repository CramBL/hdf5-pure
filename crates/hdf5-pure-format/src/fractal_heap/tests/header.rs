use core::mem;

use rstest::rstest;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::*;
use crate::address::StoredAddressExt;
use crate::filter_pipeline::FilterDescription;
use crate::filter_pipeline::FilterPipeline;
use crate::fractal_heap::test_support;

#[rstest]
#[case::zero(0)]
#[case::nonzero(512)]
#[case::large(u64::MAX - 1024)]
fn header_regions_preserve_physical_positions(#[case] address: u64) {
    let bytes = fractal_heap::Header::new(256).build(Widths::EIGHT);
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(address), bytes.len() as u64).unwrap(),
        &bytes,
    )
    .unwrap();
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(address),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    assert_eq!(
        locator.parse_range().unwrap(),
        FileRange::new(
            PhysicalAddress::new(address),
            FractalHeapHeaderFrame::PREFIX_LEN as u64
        )
        .unwrap()
    );
    let header_ref = locator.parse(region).unwrap();
    assert_eq!(header_ref.range(), region.range());
    assert_eq!(header_ref.parse_range(), region.range());
    assert_eq!(header_ref.parse(region).unwrap(), unfiltered_header());
}

#[rstest]
#[case::short(512, 8)]
#[case::before(511, 9)]
#[case::after(513, 9)]
fn a_header_locator_requires_its_physical_prefix(
    #[case] provided_start: u64,
    #[case] provided_length: u64,
) {
    let bytes = header_prefix(0);
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(provided_start), provided_length).unwrap(),
        &bytes[..usize::try_from(provided_length).unwrap()],
    )
    .unwrap();
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(512),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    assert_eq!(
        locator.parse(region).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: 512,
            required_length: FractalHeapHeaderFrame::PREFIX_LEN as u64,
            provided_start,
            provided_length,
        }
    );
}

#[test]
fn a_header_locator_rejects_an_overflowing_prefix_range() {
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(u64::MAX),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    assert_eq!(
        locator.parse_range().unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX,
            length: FractalHeapHeaderFrame::PREFIX_LEN as u64,
        }
    );
}

#[test]
fn a_header_locator_rejects_an_overflowing_complete_range() {
    let bytes = header_prefix(0);
    let address = u64::MAX - FractalHeapHeaderFrame::PREFIX_LEN as u64;
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(address), bytes.len() as u64).unwrap(),
        &bytes,
    )
    .unwrap();
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(address),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    assert_eq!(
        locator.parse(region).unwrap_err(),
        FormatError::OffsetOverflow {
            offset: address,
            length: 146
        }
    );
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two, 56, 62)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four, 80, 88)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight, 128, 140)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two, 62, 68)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four, 86, 94)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight, 134, 146)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, 74, 80)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four, 98, 106)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight, 146, 158)]
fn a_header_frame_sizes_all_fields_at_their_declared_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] plain_len: usize,
    #[case] filtered_len: usize,
    #[values(0, 1, 512, u16::MAX)] filter_encoded_len: u16,
) {
    let prefix = header_prefix(filter_encoded_len);
    assert_eq!(prefix.len(), FractalHeapHeaderFrame::PREFIX_LEN);
    let frame = FractalHeapHeaderFrame::parse(&prefix, offsets, lengths).unwrap();
    assert_eq!(
        frame.encoded_len(),
        if filter_encoded_len == 0 {
            plain_len
        } else {
            filtered_len + usize::from(filter_encoded_len)
        }
    );
}

#[rstest]
#[case::signature(*b"XXXX", FRACTAL_HEAP_VERSION)]
#[case::version(FRACTAL_HEAP_SIGNATURE, UNSUPPORTED_HEADER_VERSION)]
fn header_parsers_acquire_the_complete_prefix_before_interpreting_it(
    #[case] signature: [u8; 4],
    #[case] version: u8,
    #[values(0, 1, 2, 3, 4, 5, 6, 7, 8)] available: usize,
) {
    let bytes = header_with_prefix(signature, version);
    let prefix = &bytes[..available];
    let error = FormatError::UnexpectedEof {
        expected: FractalHeapHeaderFrame::PREFIX_LEN,
        available,
    };
    assert_eq!(
        FractalHeapHeaderFrame::parse(prefix, OffsetWidth::Eight, LengthWidth::Eight),
        Err(error.clone())
    );
    assert_eq!(FractalHeapHeader::parse(prefix, 0, 8, 8), Err(error));
    let error = FormatError::InsufficientRegion {
        required_start: 0,
        required_length: FractalHeapHeaderFrame::PREFIX_LEN as u64,
        provided_start: 0,
        provided_length: available as u64,
    };
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(0),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    let short_region = header_region(prefix);
    assert_eq!(locator.parse(short_region), Err(error.clone()));
    let original = header_prefix(0);
    let reference = locator.parse(header_region(&original)).unwrap();
    assert_eq!(reference.parse(short_region), Err(error));
}

#[rstest]
#[case::signature(*b"XXXX", FRACTAL_HEAP_VERSION, FormatError::InvalidFractalHeapSignature)]
#[case::version(
    FRACTAL_HEAP_SIGNATURE,
    UNSUPPORTED_HEADER_VERSION,
    FormatError::InvalidFractalHeapVersion(UNSUPPORTED_HEADER_VERSION)
)]
fn header_parsers_reject_invalid_framing_before_acquiring_the_body(
    #[case] signature: [u8; 4],
    #[case] version: u8,
    #[case] error: FormatError,
) {
    let offsets = OffsetWidth::Eight;
    let lengths = LengthWidth::Eight;
    let bytes = header_with_prefix(signature, version);
    let prefix = &bytes[..FractalHeapHeaderFrame::PREFIX_LEN];
    assert_eq!(
        FractalHeapHeaderFrame::parse(prefix, offsets, lengths),
        Err(error.clone())
    );
    assert_eq!(
        FractalHeapHeader::parse(prefix, 0, 8, 8),
        Err(error.clone())
    );
    assert_eq!(
        parse_header_region(prefix, offsets, lengths),
        Err(error.clone())
    );
    let locator = FractalHeapHeaderLocator::new(PhysicalAddress::new(0), offsets, lengths);
    let original = header_prefix(0);
    let reference = locator.parse(header_region(&original)).unwrap();
    assert_eq!(reference.parse(header_region(prefix)), Err(error));
}

#[rstest]
#[case::prefix(0)]
#[case::trailing_bytes(1024)]
fn a_header_frame_only_interprets_the_prefix(#[case] trailing_len: usize) {
    let mut bytes = header_prefix(512);
    bytes.resize(FractalHeapHeaderFrame::PREFIX_LEN + trailing_len, 0xFF);
    assert_eq!(
        FractalHeapHeaderFrame::parse(&bytes, OffsetWidth::Eight, LengthWidth::Eight)
            .unwrap()
            .encoded_len(),
        670
    );
}

#[rstest]
#[case::two_two(Widths { offset: 2, length: 2 }, OffsetWidth::Two, LengthWidth::Two)]
#[case::two_eight(Widths { offset: 2, length: 8 }, OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(Widths { offset: 8, length: 2 }, OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_eight(Widths::EIGHT, OffsetWidth::Eight, LengthWidth::Eight)]
fn complete_headers_retain_filter_metadata_with_two_and_eight_byte_fields(
    #[case] widths: Widths,
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(0, 1, 80)] client_count: usize,
    #[values(test_support::PIPELINE_VERSION_ONE, test_support::PIPELINE_VERSION_TWO)]
    pipeline_version: u8,
) {
    let pipeline = filter_pipeline(pipeline_version, client_count);
    let filtering = if client_count == 0 {
        FractalHeapFiltering::Unfiltered
    } else {
        FractalHeapFiltering::Filtered {
            root_direct_block_size: 91,
            root_filter_mask: 1,
            pipeline: pipeline.clone(),
        }
    };
    let mut builder = fractal_heap::Header::new(0x100).managed_object_count(3);
    if filtering.is_filtered() {
        builder = builder.filtering(91, 1, pipeline.to_bytes());
    }
    let bytes = builder.build(widths);
    let frame = FractalHeapHeaderFrame::parse(&bytes, offsets, lengths).unwrap();
    let extension_len = if filtering.is_filtered() {
        usize::from(lengths.get())
            + FRACTAL_HEAP_FILTER_MASK_LEN.to_usize().unwrap()
            + pipeline.to_bytes().len()
    } else {
        0
    };
    assert_eq!(
        frame.encoded_len(),
        FRACTAL_HEAP_HEADER_FIXED_LEN.to_usize().unwrap()
            + FRACTAL_HEAP_HEADER_LENGTH_FIELDS.to_usize().unwrap() * widths.length
            + FRACTAL_HEAP_HEADER_ADDRESS_FIELDS.to_usize().unwrap() * widths.offset
            + extension_len
    );
    assert_eq!(bytes.len(), frame.encoded_len());
    if client_count == 80 {
        assert!(bytes.len() > 256);
    }
    let parsed = FractalHeapHeader::parse_bytes(&bytes, offsets, lengths).unwrap();
    assert_eq!(parsed.filtering, filtering);
    assert_eq!(parsed.managed_objects_count, 3);
    let locator = FractalHeapHeaderLocator::new(PhysicalAddress::new(0), offsets, lengths);
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        &bytes,
    )
    .unwrap();
    let header_ref = locator.parse(region).unwrap();
    assert_eq!(header_ref.range().length(), bytes.len() as u64);
    assert_eq!(header_ref.parse(region), Ok(parsed.clone()));
    let mut file = vec![0xFF; 13];
    file.extend_from_slice(&bytes);
    file.extend_from_slice(&[0xFF; 7]);
    assert_eq!(
        FractalHeapHeader::parse(&file, 13, offsets.get(), lengths.get()),
        Ok(parsed.clone())
    );
    assert_eq!(
        FractalHeapHeaderLocator::new(PhysicalAddress::new(13), offsets, lengths)
            .parse(
                FileRegion::new(
                    FileRange::new(PhysicalAddress::new(0), file.len() as u64).unwrap(),
                    &file
                )
                .unwrap()
            )
            .unwrap()
            .parse(
                FileRegion::new(
                    FileRange::new(PhysicalAddress::new(0), file.len() as u64).unwrap(),
                    &file
                )
                .unwrap()
            ),
        Ok(parsed)
    );
}

#[rstest]
#[case::unfiltered(false, false)]
#[case::unfiltered_after_filtered_prefix(false, true)]
#[case::filtered_after_unfiltered_prefix(true, false)]
#[case::filtered(true, true)]
fn header_regions_parse_prefix_fields_from_the_complete_header(
    #[case] filtered_header: bool,
    #[case] filtered_prefix: bool,
) {
    let pipeline = filter_pipeline(test_support::PIPELINE_VERSION_TWO, 1);
    let encoded_pipeline = pipeline.to_bytes();
    let mut builder = fractal_heap::Header::new(256);
    let mut expected = unfiltered_header();
    if filtered_header {
        builder = builder.filtering(91, 1, encoded_pipeline.clone());
        expected.filtering = FractalHeapFiltering::Filtered {
            root_direct_block_size: 91,
            root_filter_mask: 1,
            pipeline,
        };
    }
    let mut bytes = builder.build(Widths::EIGHT);
    let complete_length = bytes.len() as u64;
    let mut prefix = header_prefix(if filtered_prefix {
        u16::try_from(encoded_pipeline.len()).unwrap()
    } else {
        0
    });
    prefix[HEADER_VERSION_END..HEADER_ID_LENGTH_END].copy_from_slice(&u16::MAX.to_le_bytes());
    let address = PhysicalAddress::new(512);
    let locator = FractalHeapHeaderLocator::new(address, OffsetWidth::Eight, LengthWidth::Eight);
    let prefix_region = FileRegion::new(locator.parse_range().unwrap(), &prefix).unwrap();
    let header_ref = locator.parse(prefix_region).unwrap();
    let framed_length = usize::try_from(header_ref.parse_range().length()).unwrap();
    bytes.resize(bytes.len().max(framed_length), 0xFF);
    let region = FileRegion::new(header_ref.parse_range(), &bytes[..framed_length]).unwrap();
    let result = header_ref.parse(region);
    if filtered_header && !filtered_prefix {
        assert_eq!(
            result.unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: address.get(),
                required_length: complete_length,
                provided_start: address.get(),
                provided_length: framed_length as u64,
            }
        );
    } else {
        assert_eq!(result.unwrap(), expected);
    }
}

#[rstest]
#[case::unfiltered_checksum(0, 145, 146)]
#[case::root_size_start(1, 142, 170)]
#[case::partial_root_size(1, 149, 170)]
#[case::mask_start(1, 150, 170)]
#[case::partial_mask(1, 153, 170)]
#[case::pipeline_start(1, 154, 170)]
#[case::partial_pipeline(1, 165, 170)]
#[case::checksum_start(1, 166, 170)]
#[case::partial_checksum(1, 169, 170)]
fn an_incomplete_header_requires_its_complete_framed_extent(
    #[case] client_count: usize,
    #[case] available: usize,
    #[case] expected: usize,
) {
    let mut builder = fractal_heap::Header::new(0x100);
    if client_count > 0 {
        builder = builder.filtering(
            91,
            1,
            filter_pipeline(test_support::PIPELINE_VERSION_TWO, client_count).to_bytes(),
        );
    }
    let mut bytes = builder.build(Widths::EIGHT);
    assert_eq!(bytes.len(), expected);
    bytes.truncate(available);
    let error = FormatError::UnexpectedEof {
        expected,
        available,
    };
    assert_eq!(
        FractalHeapHeader::parse(&bytes, 0, 8, 8),
        Err(error.clone())
    );
    assert_eq!(
        parse_header_region(&bytes, OffsetWidth::Eight, LengthWidth::Eight),
        Err(FormatError::InsufficientRegion {
            required_start: 0,
            required_length: expected as u64,
            provided_start: 0,
            provided_length: available as u64,
        })
    );
}

#[rstest]
#[case::truncated_client_data(vec![test_support::PIPELINE_VERSION_TWO, 1, test_support::FILTER_DEFLATE.to_le_bytes()[0], test_support::FILTER_DEFLATE.to_le_bytes()[1], 0, 0, 1, 0], FormatError::UnexpectedEof { expected: 12, available: 8 })]
#[case::invalid_version(vec![0xFF, 0], FormatError::InvalidFilterPipelineVersion(0xFF))]
#[case::too_many_filters(vec![test_support::PIPELINE_VERSION_TWO, FILTER_COUNT_MAXIMUM + 1], FormatError::InvalidFilterPipelineField { field: "filter count", value: usize::from(FILTER_COUNT_MAXIMUM) + 1, maximum: usize::from(FILTER_COUNT_MAXIMUM) })]
#[case::invalid_name(vec![test_support::PIPELINE_VERSION_TWO, 1, 0x2C, 1, 1, 0, 0, 0, 0, 0, b'a'], FormatError::InvalidFilterName { filter_id: 300, reason: "missing null terminator" })]
fn malformed_filter_information_returns_its_pipeline_error(
    #[case] pipeline: Vec<u8>,
    #[case] error: FormatError,
) {
    let bytes = fractal_heap::Header::new(0x100)
        .filtering(91, 1, pipeline)
        .build(Widths::EIGHT);
    assert_eq!(
        FractalHeapHeader::parse(&bytes, 0, 8, 8),
        Err(error.clone())
    );
    assert_eq!(
        parse_header_region(&bytes, OffsetWidth::Eight, LengthWidth::Eight),
        Err(error)
    );
}

#[rstest]
#[case::two_eight(Widths { offset: 2, length: 8 }, OffsetWidth::Two, LengthWidth::Eight, 142)]
#[case::eight_two(Widths { offset: 8, length: 2 }, OffsetWidth::Eight, LengthWidth::Two, 82)]
fn header_parsers_apply_checksum_policy_before_decoding_the_filter_pipeline(
    #[case] widths: Widths,
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] extent: usize,
) {
    let mut bytes = fractal_heap::Header::new(256)
        .filtering(91, 1, vec![0xFF, 0])
        .build(widths);
    assert_eq!(bytes.len(), extent);
    let checksum_pos = extent - test_util::checksum::CHECKSUM;
    let computed = test_util::checksum::lookup3(&bytes[..checksum_pos]);
    let expected = computed ^ 1;
    bytes[checksum_pos..].copy_from_slice(&expected.to_le_bytes());
    let invalid_pipeline = FormatError::InvalidFilterPipelineVersion(0xFF);
    let error = if cfg!(feature = "checksum") {
        FormatError::ChecksumMismatch { expected, computed }
    } else {
        invalid_pipeline.clone()
    };
    assert_eq!(
        FractalHeapHeader::parse_bytes(&bytes, offsets, lengths),
        Err(error.clone())
    );
    assert_eq!(parse_header_region(&bytes, offsets, lengths), Err(error));

    test_util::checksum::restamp(&mut bytes, 0, extent);
    assert_eq!(
        FractalHeapHeader::parse_bytes(&bytes, offsets, lengths),
        Err(invalid_pipeline.clone())
    );
    assert_eq!(
        parse_header_region(&bytes, offsets, lengths),
        Err(invalid_pipeline)
    );
}

#[cfg(feature = "checksum")]
#[rstest]
#[case::unfiltered(false)]
#[case::filtered(true)]
fn a_header_checksum_error_reports_the_final_checksum(#[case] filtered: bool) {
    let mut builder = fractal_heap::Header::new(0x100);
    if filtered {
        builder = builder.filtering(
            91,
            1,
            filter_pipeline(test_support::PIPELINE_VERSION_TWO, 1).to_bytes(),
        );
    }
    let mut bytes = builder.build(Widths::EIGHT);
    let checksum_pos = bytes.len() - FRACTAL_HEAP_CHECKSUM_LEN;
    let computed = crate::checksum::jenkins_lookup3(&bytes[..checksum_pos]);
    let expected = computed ^ 1;
    bytes[checksum_pos..].copy_from_slice(&expected.to_le_bytes());
    let error = FormatError::ChecksumMismatch { expected, computed };
    assert_eq!(
        FractalHeapHeader::parse(&bytes, 0, 8, 8),
        Err(error.clone())
    );
    assert_eq!(
        parse_header_region(&bytes, OffsetWidth::Eight, LengthWidth::Eight),
        Err(error)
    );
}

#[test]
fn a_header_parses_to_its_fields() {
    let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
    let expected = FractalHeapHeader {
        heap_id_length: 7,
        filtering: FractalHeapFiltering::Unfiltered,
        flags: 0,
        max_managed_object_size: 64,
        next_huge_object_id: 0,
        btree_huge_objects_address: StoredAddress::new(u64::MAX),
        free_space_in_managed_blocks: 0,
        managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
        managed_space: 0,
        allocated_managed_space: 0,
        direct_block_allocation_iterator_offset: 0,
        managed_objects_count: 1,
        huge_objects_size: 0,
        huge_objects_count: 0,
        tiny_objects_size: 0,
        tiny_objects_count: 0,
        table_width: 4,
        starting_block_size: 128,
        max_direct_block_size: 1024,
        max_heap_size: 16,
        start_root_rows: 2,
        root_block_address: StoredAddress::new(256),
        current_rows_in_root_indirect_block: 0,
    };

    assert_eq!(
        FractalHeapHeader::parse(&file_data, 0, 8, 8),
        Ok(expected.clone())
    );
    assert_eq!(
        parse_header_region(&file_data, OffsetWidth::Eight, LengthWidth::Eight),
        Ok(expected)
    );
}

#[rstest]
#[case::eight_byte_fields(Widths::EIGHT, OffsetWidth::Eight, LengthWidth::Eight)]
#[case::four_byte_fields(Widths::FOUR, OffsetWidth::Four, LengthWidth::Four)]
fn a_parsed_header_serializes_to_the_bytes_it_was_parsed_from(
    #[case] widths: Widths,
    #[case] offset_width: OffsetWidth,
    #[case] length_width: LengthWidth,
) {
    let bytes = fractal_heap::Header::new(0x100)
        .managed_object_count(3)
        .build(widths);
    let header = FractalHeapHeader::parse_bytes(&bytes, offset_width, length_width).unwrap();

    assert_eq!(
        header.serialize(offset_width, length_width),
        Ok(bytes.clone())
    );
    assert_eq!(
        FractalHeapHeader::serialized_size(offset_width, length_width),
        bytes.len()
    );
}

#[test]
fn serializing_a_filtered_header_is_an_error() {
    let header = FractalHeapHeader {
        filtering: test_support::filtered_heap(),
        ..test_support::dtable_header(512, 65536, 4)
    };
    assert_eq!(
        header.serialize(OffsetWidth::Eight, LengthWidth::Eight),
        Err(FormatError::UnsupportedFilteredHeapObject)
    );
}

fn unfiltered_header() -> FractalHeapHeader {
    FractalHeapHeader {
        heap_id_length: 7,
        filtering: FractalHeapFiltering::Unfiltered,
        flags: 0,
        max_managed_object_size: 64,
        next_huge_object_id: 0,
        btree_huge_objects_address: StoredAddress::new(u64::MAX),
        free_space_in_managed_blocks: 0,
        managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
        managed_space: 0,
        allocated_managed_space: 0,
        direct_block_allocation_iterator_offset: 0,
        managed_objects_count: 0,
        huge_objects_size: 0,
        huge_objects_count: 0,
        tiny_objects_size: 0,
        tiny_objects_count: 0,
        table_width: 4,
        starting_block_size: 128,
        max_direct_block_size: 1024,
        max_heap_size: 16,
        start_root_rows: 2,
        root_block_address: StoredAddress::new(256),
        current_rows_in_root_indirect_block: 0,
    }
}

fn parse_header_region(
    bytes: &[u8],
    offsets: OffsetWidth,
    lengths: LengthWidth,
) -> Result<FractalHeapHeader, FormatError> {
    let region = header_region(bytes);
    FractalHeapHeaderLocator::new(PhysicalAddress::new(0), offsets, lengths)
        .parse(region)?
        .parse(region)
}

fn header_region(bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
}

fn filter_pipeline(version: u8, client_count: usize) -> FilterPipeline {
    FilterPipeline::new(
        version,
        vec![FilterDescription {
            filter_id: test_support::FILTER_DEFLATE,
            name: if version == test_support::PIPELINE_VERSION_ONE {
                Some("deflate".into())
            } else {
                None
            },
            flags: 0,
            client_data: vec![6; client_count],
        }],
    )
    .unwrap()
}

fn header_prefix(filter_encoded_len: u16) -> Vec<u8> {
    let mut prefix = FRACTAL_HEAP_SIGNATURE.to_vec();
    prefix.push(FRACTAL_HEAP_VERSION);
    prefix.extend_from_slice(&8u16.to_le_bytes());
    prefix.extend_from_slice(&filter_encoded_len.to_le_bytes());
    prefix
}

fn header_with_prefix(signature: [u8; 4], version: u8) -> Vec<u8> {
    let mut bytes = fractal_heap::Header::new(256).build(Widths::EIGHT);
    bytes[..signature.len()].copy_from_slice(&signature);
    bytes[signature.len()] = version;
    bytes
}

const HEADER_VERSION_END: usize = FRACTAL_HEAP_SIGNATURE.len() + mem::size_of::<u8>();
const HEADER_ID_LENGTH_END: usize = HEADER_VERSION_END + mem::size_of::<u16>();
const UNSUPPORTED_HEADER_VERSION: u8 = FRACTAL_HEAP_VERSION + 1;
// "The Data Storage - Filter Pipeline Message", format specification version 4.0, limits
// pipelines to 32 filters.
const FILTER_COUNT_MAXIMUM: u8 = 32;
