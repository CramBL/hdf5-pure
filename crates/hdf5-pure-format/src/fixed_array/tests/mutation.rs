use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use rstest::rstest;
use test_util::fixed_array as fixture;

use super::support;
use super::support::CHUNK_BYTES;
use super::support::DATA_AT;
use super::support::HEADER_AT;
use super::support::PAGE_BITS;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record::ChunkRecord;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn initialized_pages_encode_checked_entry_windows(
    #[case] offsets: OffsetWidth,
    #[values(false, true)] filtered: bool,
) {
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Two,
    };
    let client = if filtered {
        fixture::Client::Filtered { size_bytes: 2 }
    } else {
        fixture::Client::Plain
    };
    let header = support::parse_header(
        widths,
        0,
        &fixture::header(
            support::fixture_widths(widths),
            client,
            PAGE_BITS,
            17,
            Some(DATA_AT),
        ),
    );
    let block = header.data_block().unwrap();
    let page = block.page_for_slot(16).unwrap().unwrap();
    assert_eq!(page.slots().unwrap(), 16..17);
    let mut bytes = vec![0xcc; usize::try_from(page.parse_range().length()).unwrap() + 1];
    let used = page.encode_empty(&mut bytes).unwrap();
    assert_eq!(used + 1, bytes.len());
    assert_eq!(bytes[used], 0xcc);
    assert!(
        page.parse(support::region(page.parse_range().start(), &bytes[..used]).unwrap())
            .unwrap()
            .records(CHUNK_BYTES)
            .next()
            .is_none()
    );
    let window = page
        .window(16, NonZeroUsize::new(1).unwrap())
        .unwrap()
        .unwrap();
    let mut output = vec![0xcc; usize::try_from(window.parse_range().length()).unwrap() + 1];
    let record = ChunkRecord {
        address: StoredAddress::new(0x3456),
        stored_size: 255,
        filter_mask: 0x1122_3344,
    };
    let written = window.encode_records(&mut output, &[Some(record)]).unwrap();
    assert_eq!(output[written], 0xcc);
    let mut expected = record;
    if !filtered {
        expected.stored_size = CHUNK_BYTES;
        expected.filter_mask = 0;
    }
    assert_eq!(
        window
            .parse(support::region(window.parse_range().start(), &output[..written]).unwrap())
            .unwrap()
            .records(CHUNK_BYTES)
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        [(16, expected)]
    );
    assert!(
        page.window(15, NonZeroUsize::new(1).unwrap())
            .unwrap()
            .is_none()
    );
    assert!(
        page.window(17, NonZeroUsize::new(1).unwrap())
            .unwrap()
            .is_none()
    );
    assert!(block.page_for_slot(17).unwrap().is_none());
    assert!(block.page_bitmap_byte(17).unwrap().is_none());
}

#[rstest]
#[case::size_overflow(&[Some(ChunkRecord { address: StoredAddress::new(32), stored_size: 65_536, filter_mask: 0 })], 8, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
#[case::address_sentinel(&[Some(ChunkRecord { address: StoredAddress::new(65_535), stored_size: 1, filter_mask: 0 })], 8, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::address_overflow(&[Some(ChunkRecord { address: StoredAddress::new(65_536), stored_size: 1, filter_mask: 0 })], 8, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::short_output(&[None], 7, FormatError::UnexpectedEof { expected: 8, available: 7 })]
#[case::count(&[], 8, FormatError::FixedArrayEntryCount { expected: 1, actual: 0 })]
#[case::later_error(&[Some(ChunkRecord { address: StoredAddress::new(32), stored_size: 1, filter_mask: 0 }), Some(ChunkRecord { address: StoredAddress::new(65_535), stored_size: 1, filter_mask: 0 })], 16, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
fn entry_window_errors_preserve_output(
    #[case] records: &[Option<ChunkRecord>],
    #[case] capacity: usize,
    #[case] expected: FormatError,
) {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let header = support::parse_header(
        widths,
        0,
        &fixture::header(
            support::fixture_widths(widths),
            fixture::Client::Filtered { size_bytes: 2 },
            PAGE_BITS,
            2,
            Some(DATA_AT),
        ),
    );
    let block = header.data_block().unwrap();
    let data = fixture::empty_data_block(
        support::fixture_widths(widths),
        fixture::Client::Filtered { size_bytes: 2 },
        PAGE_BITS,
        2,
        HEADER_AT,
    );
    let prefix = block
        .parse_unpaged_prefix(support::region(block.range().start(), &data).unwrap())
        .unwrap();
    let window = prefix
        .window(0, NonZeroUsize::new(records.len().max(1)).unwrap())
        .unwrap()
        .unwrap();
    let mut output = vec![0xcc; capacity];
    assert_eq!(
        window.encode_records(&mut output, records).unwrap_err(),
        expected
    );
    assert_eq!(output, vec![0xcc; capacity]);
}

#[rstest]
#[case::one(1, 255)]
#[case::three(3, 0x00ff_ffff)]
#[case::eight(8, u64::MAX)]
fn filtered_entry_size_is_independent_of_file_length_width(
    #[case] size_bytes: u8,
    #[case] value: u64,
) {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let client = fixture::Client::Filtered { size_bytes };
    let header = support::parse_header(
        widths,
        0,
        &fixture::header(
            support::fixture_widths(widths),
            client,
            PAGE_BITS,
            1,
            Some(DATA_AT),
        ),
    );
    let block = header.data_block().unwrap();
    let data = fixture::empty_data_block(
        support::fixture_widths(widths),
        client,
        PAGE_BITS,
        1,
        HEADER_AT,
    );
    let parsed = block
        .parse_unpaged_prefix(support::region(block.range().start(), &data).unwrap())
        .unwrap();
    let window = parsed
        .window(0, NonZeroUsize::new(1).unwrap())
        .unwrap()
        .unwrap();
    let mut output = vec![0xcc; usize::try_from(window.parse_range().length()).unwrap()];
    window
        .encode_records(
            &mut output,
            &[Some(ChunkRecord {
                address: StoredAddress::new(0x1234),
                stored_size: value,
                filter_mask: 0x1122_3344,
            })],
        )
        .unwrap();
    let mut expected = vec![0x34, 0x12];
    test_util::bytes::push_uint(&mut expected, value, usize::from(size_bytes));
    expected.extend_from_slice(&0x1122_3344_u32.to_le_bytes());
    assert_eq!(output, expected);
}
