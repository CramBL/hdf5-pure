use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::{BaseAddress, FormatError, StoredAddress};
use hdf5_pure_format::__private::{
    ChunkIndexLayout, DataLayout, ExtensibleArrayCreationParameters, ExtensibleArrayHeaderLocator,
    ExtensibleArrayInitialization, FormatWidths, LengthWidth, OffsetWidth,
};
use rstest::rstest;
use test_util::data_layout;

#[rstest]
#[case::canonical([32, 4, 4, 16, 10])]
#[case::small([16, 8, 2, 8, 8])]
#[case::large_inline([24, 32, 8, 32, 12])]
#[case::no_inline([16, 0, 2, 8, 8])]
fn initialization_preserves_creation_geometry_and_complete_allocations(
    #[case] parameters: [u8; 5],
    #[values(false, true)] filtered: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[values(0, 512, u64::from(u32::MAX) + 512)] base: u64,
) {
    let creation = ExtensibleArrayCreationParameters::parse(&parameters).unwrap();
    assert_eq!(creation.encode(), parameters);
    let widths = FormatWidths { offsets, lengths };
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
    assert_eq!(header.client_id, u8::from(filtered));
    assert_eq!(header.max_idx_set, 0);
    assert_eq!(header.bit_geometry(), creation.bit_geometry());
    assert_eq!(header.idx_blk_elmts, parameters[1]);
    assert_eq!(header.super_blk_min_data_ptrs, parameters[2]);
    assert_eq!(header.min_dblk_nelmts, parameters[3]);
    assert_eq!(header.idx_blk_elmts, creation.inline_elements());
    assert_eq!(
        header.super_blk_min_data_ptrs,
        creation.min_data_blocks().get()
    );
    assert_eq!(header.min_dblk_nelmts, creation.min_elements().get());
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
    let statistics = &encoded[12..12 + 6 * usize::from(lengths.get())];
    assert!(
        statistics[..5 * usize::from(lengths.get())]
            .iter()
            .all(|&byte| byte == 0)
    );
    assert_eq!(statistics[5 * usize::from(lengths.get())], parameters[1]);
    assert!(
        statistics[5 * usize::from(lengths.get()) + 1..]
            .iter()
            .all(|&byte| byte == 0)
    );
    assert_checksum(&encoded[..header_range.length() as usize]);
    assert_checksum(&encoded[header_range.length() as usize..]);
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

// "The Data Layout Message", format specification, version 4.0.
const EXTENSIBLE_ARRAY_INDEX: u8 = 4;

// "The Extensible Array Index", format specification, version 4.0.
const CHECKSUM_LEN: usize = 4;
