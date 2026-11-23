use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::{BaseAddress, FormatError, StoredAddress};
use hdf5_pure_format::__private::{
    ChunkIndexLayout, DataLayout, ExtensibleArrayCreationParameters, ExtensibleArrayHeaderLocator,
    ExtensibleArrayHeaderWithoutIndexBlock, ExtensibleArrayInitialization, ExtensibleArrayPlan,
    FormatWidths, IndexSlots, LengthWidth, OffsetWidth, SlotOccupancy,
};
use rstest::rstest;
use test_util::data_layout;
use test_util::extensible_array::{self, Client, Creation, Header, HeaderStats, Width};

#[rstest]
#[case::canonical(Creation::Canonical)]
#[case::large_inline(Creation::LargeInline)]
#[case::no_inline(Creation::NoInline)]
fn standalone_header_encodes_absent_index_storage(
    #[case] creation: Creation,
    #[values(
        Client::Unfiltered,
        Client::Filtered(Width::Two),
        Client::Filtered(Width::Eight)
    )]
    client: Client,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
    #[values(Width::Two, Width::Four, Width::Eight)] lengths: Width,
    #[values(0, 3)] suffix: usize,
) {
    let fixture = Header::new(offsets, lengths, client, creation);
    let parameters = creation.layout_fields();
    let creation = ExtensibleArrayCreationParameters::parse(&parameters).unwrap();
    let widths = FormatWidths::from_sizes(offsets.get(), lengths.get()).unwrap();
    let header = ExtensibleArrayHeaderWithoutIndexBlock::new(
        creation,
        widths,
        client.id(),
        client.element_size(offsets),
    )
    .unwrap();
    let expected = fixture.build();
    let length = fixture.encoded_len();
    assert_eq!(header.encoded_size(), length);
    let mut output = vec![0xa5; 1 + length + suffix];
    assert_eq!(header.write_into(&mut output[1..]).unwrap(), length);
    let mut guarded = vec![0xa5; output.len()];
    guarded[1..1 + length].copy_from_slice(&expected);
    assert_eq!(output, guarded);

    let body = &output[1..1 + length];
    let statistics_start = extensible_array::PREFIX_LEN;
    let statistics_end =
        statistics_start + extensible_array::STATISTIC_COUNT * usize::from(lengths.get());
    assert_eq!(
        body[statistics_start..statistics_end],
        vec![0; statistics_end - statistics_start]
    );
    let address_end = statistics_end + usize::from(offsets.get());
    assert_eq!(
        body[statistics_end..address_end],
        vec![0xff; usize::from(offsets.get())]
    );

    let base = BaseAddress::new(512);
    let address = StoredAddress::new(1024);
    let locator = ExtensibleArrayHeaderLocator::new(widths, base, address);
    let range = FileRange::new(PhysicalAddress::new(1536), length as u64).unwrap();
    assert_eq!(locator.parse_range().unwrap(), range);
    let array = locator
        .parse(FileRegion::new(range, body).unwrap())
        .unwrap();
    assert_eq!(array.range(), range);
    assert_eq!(array.index_block(), None);
    let header = array.header();
    assert_eq!(header.client_id(), client.id());
    assert_eq!(header.element_size(), client.element_size(offsets));
    assert_eq!(header.creation_parameters(), creation);
    assert_eq!(header.stored_statistics().max_idx_set, 0);
    assert!(header.index_block_address().is_undefined(widths.offsets));
}

#[rstest]
#[case::short_widths(2, 2)]
#[case::unequal_widths(4, 8)]
#[case::wide_widths(8, 8)]
fn standalone_header_rejects_short_destinations_without_writing(
    #[case] offsets: u8,
    #[case] lengths: u8,
) {
    let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
    let header = ExtensibleArrayHeaderWithoutIndexBlock::new(
        ExtensibleArrayCreationParameters::canonical(),
        widths,
        Client::Unfiltered.id(),
        offsets,
    )
    .unwrap();
    let expected = header.encoded_size();
    for available in 0..expected {
        let mut output = vec![0xa5; expected + 2];
        assert_eq!(
            header
                .write_into(&mut output[1..1 + available])
                .unwrap_err(),
            FormatError::UnexpectedEof {
                expected,
                available
            },
        );
        assert_eq!(output, vec![0xa5; expected + 2]);
    }
}

#[rstest]
#[case::reserved_client(2, 4, FormatError::InvalidExtensibleArrayClient(2))]
#[case::unfiltered_size_differs_from_address_width(Client::Unfiltered.id(), 2, FormatError::InvalidExtensibleArrayGeometry)]
#[case::filtered_size_shorter_than_address_and_mask(Client::Filtered(Width::Two).id(), 7, FormatError::InvalidExtensibleArrayGeometry)]
#[case::zero_byte_filtered_size(Client::Filtered(Width::Two).id(), 8, FormatError::InvalidExtensibleArrayGeometry)]
#[case::one_byte_filtered_size(Client::Filtered(Width::Two).id(), 9, FormatError::InvalidExtensibleArrayGeometry)]
#[case::nine_byte_filtered_size(Client::Filtered(Width::Two).id(), 17, FormatError::InvalidExtensibleArrayGeometry)]
fn standalone_header_rejects_invalid_element_encodings(
    #[case] client: u8,
    #[case] element_size: u8,
    #[case] expected: FormatError,
) {
    let widths = FormatWidths::from_sizes(4, 8).unwrap();
    assert_eq!(
        ExtensibleArrayHeaderWithoutIndexBlock::new(
            ExtensibleArrayCreationParameters::canonical(),
            widths,
            client,
            element_size,
        )
        .unwrap_err(),
        expected,
    );
}

#[cfg(feature = "checksum")]
#[test]
fn standalone_header_checksum_corruption_is_rejected() {
    let fixture = Header::new(Width::Four, Width::Two, Client::Unfiltered, Creation::Small);
    let mut body = fixture.build();
    let checksum_at = body.len() - CHECKSUM_LEN;
    let computed = test_util::checksum::lookup3(&body[..checksum_at]);
    body[checksum_at] ^= 0x80;
    let expected = test_util::bytes::u32_at(&body, checksum_at);
    let locator = ExtensibleArrayHeaderLocator::new(
        FormatWidths::from_sizes(4, 2).unwrap(),
        BaseAddress::ZERO,
        StoredAddress::new(512),
    );
    let region = FileRegion::new(locator.parse_range().unwrap(), &body).unwrap();
    assert_eq!(
        locator.parse(region).unwrap_err(),
        FormatError::ChecksumMismatch { expected, computed },
    );
}

#[rstest]
#[case::canonical(Creation::Canonical)]
#[case::small(Creation::Small)]
#[case::large_inline(Creation::LargeInline)]
#[case::no_inline(Creation::NoInline)]
fn initialization_preserves_creation_geometry_and_complete_allocations(
    #[case] fixture_creation: Creation,
    #[values(false, true)] filtered: bool,
    #[values(Width::Two, Width::Four, Width::Eight)] offsets: Width,
    #[values(Width::Two, Width::Four, Width::Eight)] lengths: Width,
    #[values(0, 512, u64::from(u32::MAX) + 512)] base: u64,
) {
    let parameters = fixture_creation.layout_fields();
    let creation = ExtensibleArrayCreationParameters::parse(&parameters).unwrap();
    assert_eq!(creation.encode(), parameters);
    let widths = FormatWidths::from_sizes(offsets.get(), lengths.get()).unwrap();
    let initialization = ExtensibleArrayInitialization::new(creation, 4, widths, filtered).unwrap();
    let address = StoredAddress::new(1024);
    let encoded = initialization.encode_at(address).unwrap();
    let range = FileRange::new(
        PhysicalAddress::new(base + address.get()),
        encoded.len() as u64,
    )
    .unwrap();
    assert_eq!(initialization.allocation_len(), range.length());
    let region = FileRegion::new(range, &encoded).unwrap();
    let locator = ExtensibleArrayHeaderLocator::new(widths, BaseAddress::new(base), address);
    let array = locator.parse(region).unwrap();
    let header = array.header();
    assert_eq!(header.client_id(), u8::from(filtered));
    assert_eq!(header.stored_statistics().max_idx_set, 0);
    assert_eq!(header.creation_parameters(), creation);
    let header_range = array.range();
    assert_eq!(header_range, locator.parse_range().unwrap());
    let index = array.index_block().unwrap();
    assert_eq!(index.range(), index.parse_range());
    assert_eq!(index.range().start(), header_range.end());
    assert_eq!(index.range().end(), range.end());
    let index_view = index.parse(region).unwrap();
    assert_eq!(index_view.elements().records(4).count(), 0);
    assert_eq!(index_view.data_blocks().count(), 0);
    assert_eq!(index_view.super_blocks().count(), 0);
    let direct = 2 * (usize::from(parameters[2]) - 1);
    let secondary = usize::from(parameters[0]) + 1
        - parameters[3].trailing_zeros() as usize
        - 2 * parameters[2].trailing_zeros() as usize;
    let stride = usize::from(offsets.get()) + if filtered { 6 } else { 0 };
    let index_len = 6
        + usize::from(offsets.get())
        + usize::from(parameters[1]) * stride
        + (direct + secondary) * usize::from(offsets.get())
        + 4;
    assert_eq!(index.range().length(), index_len as u64);
    let client = if filtered {
        Client::Filtered(Width::Two)
    } else {
        Client::Unfiltered
    };
    let fixture = Header::new(offsets, lengths, client, fixture_creation);
    let header_length = fixture.encoded_len();
    assert_eq!(header_range.length(), header_length as u64);
    let expected_header = fixture
        .with_index(
            address.get() + header_length as u64,
            inline_stats(parameters[1]),
        )
        .unwrap()
        .build();
    assert_eq!(&encoded[..header_length], expected_header);
    assert_checksum(&encoded[header_length..]);
}

#[test]
fn placed_header_preserves_a_defined_index_address_of_zero() {
    let fixture = Header::new(
        Width::Two,
        Width::Four,
        Client::Unfiltered,
        Creation::Canonical,
    );
    let parameters = Creation::Canonical.layout_fields();
    let creation = ExtensibleArrayCreationParameters::parse(&parameters).unwrap();
    let widths = FormatWidths::from_sizes(2, 4).unwrap();
    let plan =
        ExtensibleArrayPlan::new(creation, SlotOccupancy::Slots(&[]), 0, 4, widths, false).unwrap();
    let header_address = StoredAddress::new(1024);
    let placement = plan
        .place(&[header_address, StoredAddress::new(0)])
        .unwrap();
    let slots = IndexSlots::dense(&[]);
    let encoding = placement.encoding(&slots).unwrap();
    let mut bytes = Vec::new();
    encoding
        .blocks()
        .next()
        .unwrap()
        .encode_into(&mut bytes)
        .unwrap();
    assert_eq!(
        bytes,
        fixture
            .with_index(0, inline_stats(parameters[1]))
            .unwrap()
            .build()
    );

    let locator = ExtensibleArrayHeaderLocator::new(widths, BaseAddress::new(512), header_address);
    let region = FileRegion::new(locator.parse_range().unwrap(), &bytes).unwrap();
    let array = locator.parse(region).unwrap();
    assert_eq!(
        array.index_block().unwrap().parse_range().start(),
        PhysicalAddress::new(512)
    );
}

#[rstest]
#[case::zero_maximum([0, 4, 4, 16, 10])]
#[case::large_maximum([65, 4, 4, 16, 10])]
#[case::zero_minimum([32, 4, 4, 0, 10])]
#[case::non_power_of_two_minimum([32, 4, 4, 3, 10])]
#[case::zero_pointers([32, 4, 0, 16, 10])]
#[case::single_pointer([32, 4, 1, 16, 10])]
#[case::non_power_of_two_pointers([32, 4, 3, 16, 10])]
#[case::page_larger_than_index([16, 4, 4, 16, 17])]
#[case::page_count_overflow([64, 4, 4, 16, 64])]
#[case::short_index_geometry([1, 4, 4, 16, 1])]
#[case::paged_direct_blocks([32, 4, 4, 16, 5])]
fn invalid_creation_geometry_is_rejected(#[case] parameters: [u8; 5]) {
    assert_eq!(
        ExtensibleArrayCreationParameters::parse(&parameters).unwrap_err(),
        FormatError::InvalidExtensibleArrayGeometry,
    );
}

#[rstest]
#[case::empty(0)]
#[case::maximum_only(1)]
#[case::missing_page_bits(4)]
fn short_creation_parameters_are_distinct_from_invalid_geometry(#[case] length: usize) {
    let bytes = [32, 4, 4, 16, 10];
    assert_eq!(
        ExtensibleArrayCreationParameters::parse(&bytes[..length]).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 5,
            available: length
        },
    );
}

#[rstest]
#[case::one_byte(1)]
#[case::three_bytes(3)]
#[case::eight_bytes(8)]
fn undefined_layout_index_retains_nondefault_creation_geometry(#[case] dimension_width: usize) {
    let parameters = [24, 8, 2, 8, 8];
    let mut bytes = vec![
        data_layout::VERSION_FOUR,
        data_layout::CHUNKED_CLASS,
        data_layout::NO_CHUNKED_FLAGS,
        2,
        u8::try_from(dimension_width).unwrap(),
    ];
    bytes.extend_from_slice(&4u64.to_le_bytes()[..dimension_width]);
    bytes.extend_from_slice(&4u64.to_le_bytes()[..dimension_width]);
    bytes.push(EXTENSIBLE_ARRAY_INDEX);
    bytes.extend_from_slice(&parameters);
    bytes.extend_from_slice(&[0xff; 4]);
    let DataLayout::Chunked {
        index,
        chunk_dimensions,
        ..
    } = DataLayout::parse(&bytes, 4, 8).unwrap()
    else {
        panic!("expected chunked layout");
    };
    assert_eq!(chunk_dimensions, [4, 4]);
    let ChunkIndexLayout::ExtensibleArray { creation, address } = index else {
        panic!("expected Extensible Array index");
    };
    assert_eq!(address, None);
    assert_eq!(creation.encode(), parameters);
}

#[rstest]
#[case::two_bytes(OffsetWidth::Two)]
#[case::four_bytes(OffsetWidth::Four)]
#[case::eight_bytes(OffsetWidth::Eight)]
fn initialization_rejects_complete_allocations_past_the_address_domain(
    #[case] offsets: OffsetWidth,
) {
    let initialization = ExtensibleArrayInitialization::new(
        ExtensibleArrayCreationParameters::canonical(),
        4,
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        false,
    )
    .unwrap();
    let address = StoredAddress::undefined(offsets).get() - initialization.allocation_len() + 1;
    let error = initialization
        .encode_at(StoredAddress::new(address))
        .unwrap_err();
    if offsets == OffsetWidth::Eight {
        assert_eq!(error, FormatError::InvalidExtensibleArrayGeometry);
    } else {
        assert_eq!(
            error,
            FormatError::AddressTooLarge {
                address: StoredAddress::undefined(offsets).get() + 1,
                offset_size: offsets.get(),
            }
        );
    }
}

fn assert_checksum(bytes: &[u8]) {
    let (content, checksum) = bytes.split_at(bytes.len() - CHECKSUM_LEN);
    assert_eq!(
        checksum,
        test_util::checksum::lookup3(content).to_le_bytes()
    );
}

fn inline_stats(inline_elements: u8) -> HeaderStats {
    HeaderStats {
        secondary_block_count: 0,
        secondary_block_size: 0,
        data_block_count: 0,
        data_block_size: 0,
        max_index_set: 0,
        element_count: u64::from(inline_elements),
    }
}

// "The Data Layout Message", format specification, version 4.0.
const EXTENSIBLE_ARRAY_INDEX: u8 = 4;

// "The Extensible Array Index", format specification, version 4.0.
const CHECKSUM_LEN: usize = 4;
