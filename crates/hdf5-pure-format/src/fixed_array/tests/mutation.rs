use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::fixed_array as fixture;
use test_util::widths::Widths;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record::ChunkRecord;
use crate::error::FormatError;
use crate::fixed_array::FixedArrayDataBlock;
use crate::fixed_array::FixedArrayHeader;
use crate::fixed_array::FixedArrayHeaderLocator;
use crate::fixed_array::FixedArrayPlan;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn empty_emission_preserves_parsed_geometry_and_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(0, 512, u64::from(u32::MAX) + 1)] base: u64,
    #[values(15, 16, 17, 33)] entries: u64,
    #[values(false, true)] filtered: bool,
) {
    let widths = FormatWidths { offsets, lengths };
    let client = if filtered {
        fixture::Client::Filtered { size_bytes: 2 }
    } else {
        fixture::Client::Plain
    };
    let input = fixture::header(fixture_widths(widths), client, PAGE_BITS, entries, None);
    let header = parse_header(widths, base, &input);
    assert_eq!(header.entries(), entries);
    assert_eq!(header.page_bits(), PAGE_BITS);
    assert!(header.data_block().is_none());
    let placement = header
        .initialization_at(StoredAddress::new(DATA_AT))
        .unwrap();
    let encoded_header = placement.serialize_header().unwrap();
    assert_eq!(
        encoded_header,
        fixture::header(
            fixture_widths(widths),
            client,
            PAGE_BITS,
            entries,
            Some(DATA_AT)
        )
    );
    let mut encoded = Vec::new();
    placement
        .emit_empty_data_block::<FormatError>(|bytes| {
            encoded.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        encoded,
        fixture::empty_data_block(
            fixture_widths(widths),
            client,
            PAGE_BITS,
            usize::try_from(entries).unwrap(),
            HEADER_AT
        )
    );
    let header = parse_header(widths, base, &encoded_header);
    let block = header.data_block().unwrap();
    let data = region(block.range().start(), &encoded).unwrap();
    match block.parse(data).unwrap() {
        FixedArrayDataBlock::Elements(elements) => {
            assert!(elements.records(CHUNK_BYTES).next().is_none());
            assert!(block.page_for_slot(0).unwrap().is_none());
            assert!(block.page_bitmap_byte(0).unwrap().is_none());
        }
        FixedArrayDataBlock::PageBitmap(bitmap) => {
            assert!(bitmap.pages().next().is_none());
            let (page, initialized) = bitmap.page_for_slot(entries - 1).unwrap().unwrap();
            assert!(!initialized);
            assert!(page.slots().unwrap().contains(&(entries - 1)));
            assert!(bitmap.page_for_slot(entries).unwrap().is_none());
            let (byte, mask) = block.page_bitmap_byte(entries - 1).unwrap().unwrap();
            assert_eq!(byte.length(), 1);
            assert_eq!(
                byte.start().get(),
                base + DATA_AT + fixture::DATA_BLOCK_FIXED_PREFIX_LEN + u64::from(offsets.get())
            );
            let expected = if entries == 17 {
                fixture::SECOND_PAGE_MASK
            } else {
                fixture::THIRD_PAGE_MASK
            };
            assert_eq!(mask.get(), expected);
        }
    }
}

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
    let header = parse_header(
        widths,
        0,
        &fixture::header(fixture_widths(widths), client, PAGE_BITS, 17, Some(DATA_AT)),
    );
    let block = header.data_block().unwrap();
    let page = block.page_for_slot(16).unwrap().unwrap();
    assert_eq!(page.slots().unwrap(), 16..17);
    let mut bytes = vec![0xcc; usize::try_from(page.parse_range().length()).unwrap() + 1];
    let used = page.encode_empty(&mut bytes).unwrap();
    assert_eq!(used + 1, bytes.len());
    assert_eq!(bytes[used], 0xcc);
    assert!(
        page.parse(region(page.parse_range().start(), &bytes[..used]).unwrap())
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
            .parse(region(window.parse_range().start(), &output[..written]).unwrap())
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
    let header = parse_header(
        widths,
        0,
        &fixture::header(
            fixture_widths(widths),
            fixture::Client::Filtered { size_bytes: 2 },
            PAGE_BITS,
            2,
            Some(DATA_AT),
        ),
    );
    let block = header.data_block().unwrap();
    let data = fixture::empty_data_block(
        fixture_widths(widths),
        fixture::Client::Filtered { size_bytes: 2 },
        PAGE_BITS,
        2,
        HEADER_AT,
    );
    let prefix = block
        .parse_unpaged_prefix(region(block.range().start(), &data).unwrap())
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
#[case::unpaged(1_024, 8_210, 4)]
#[case::paged(1_024_000, 8_196_143, 2_004)]
fn empty_emission_bounds_output_windows(
    #[case] entries: u64,
    #[case] expected_bytes: u64,
    #[case] expected_calls: usize,
) {
    let plan = FixedArrayPlan::new(
        entries,
        CHUNK_BYTES,
        FormatWidths::from_sizes(8, 8).unwrap(),
        false,
    )
    .unwrap();
    let placed = plan
        .place(StoredAddress::new(HEADER_AT), StoredAddress::new(DATA_AT))
        .unwrap();
    let mut total = 0;
    let mut calls = 0;
    let mut largest = 0;
    placed
        .emit_empty_data_block::<FormatError>(|bytes| {
            calls += 1;
            total += bytes.len() as u64;
            largest = largest.max(bytes.len());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        (total, calls, largest),
        (expected_bytes, expected_calls, 4_096)
    );
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
    let header = parse_header(
        widths,
        0,
        &fixture::header(fixture_widths(widths), client, PAGE_BITS, 1, Some(DATA_AT)),
    );
    let block = header.data_block().unwrap();
    let data = fixture::empty_data_block(fixture_widths(widths), client, PAGE_BITS, 1, HEADER_AT);
    let parsed = block
        .parse_unpaged_prefix(region(block.range().start(), &data).unwrap())
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

#[test]
fn empty_page_rejects_a_short_output_before_modifying_it() {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let header = parse_header(
        widths,
        0,
        &fixture::header(
            fixture_widths(widths),
            fixture::Client::Plain,
            PAGE_BITS,
            17,
            Some(DATA_AT),
        ),
    );
    let page = header
        .data_block()
        .unwrap()
        .page_for_slot(16)
        .unwrap()
        .unwrap();
    let mut output = [0xcc; 5];
    assert_eq!(
        page.encode_empty(&mut output).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 6,
            available: 5
        }
    );
    assert_eq!(output, [0xcc; 5]);
}

#[rstest]
#[case::sentinel(65_535, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::start_overflow(65_536, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::end_overflow(65_534, FormatError::AddressTooLarge { address: 65_548, offset_size: 2 })]
fn parsed_initialization_rejects_unrepresentable_allocations(
    #[case] at: u64,
    #[case] expected: FormatError,
) {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let input = fixture::header(
        fixture_widths(widths),
        fixture::Client::Plain,
        PAGE_BITS,
        1,
        None,
    );
    let header = parse_header(widths, 512, &input);
    assert_eq!(
        header
            .initialization_at(StoredAddress::new(at))
            .unwrap_err(),
        expected
    );
    assert!(header.data_block().is_none());
    assert_eq!(
        input,
        fixture::header(
            fixture_widths(widths),
            fixture::Client::Plain,
            PAGE_BITS,
            1,
            None
        )
    );
}

#[test]
fn parsed_initialization_rejects_an_overflowing_collection() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let header = parse_header(
        widths,
        0,
        &fixture::header(
            fixture_widths(widths),
            fixture::Client::Plain,
            PAGE_BITS,
            u64::MAX,
            None,
        ),
    );
    let error = header
        .initialization_at(StoredAddress::new(DATA_AT))
        .unwrap_err();
    assert!(
        matches!(error, FormatError::InvalidFixedArrayGeometry),
        "{error:?}"
    );
}

#[rstest]
#[case::write(false)]
#[case::format(true)]
fn empty_emission_stops_at_the_original_sink_error(#[case] format: bool) {
    let plan = FixedArrayPlan::new(
        1_025,
        CHUNK_BYTES,
        FormatWidths::from_sizes(8, 8).unwrap(),
        false,
    )
    .unwrap();
    let placement = plan
        .place(StoredAddress::new(HEADER_AT), StoredAddress::new(DATA_AT))
        .unwrap();
    let mut calls = 0;
    let error = placement
        .emit_empty_data_block::<SinkError>(|_| {
            calls += 1;
            if calls == 2 {
                Err(if format {
                    FormatError::InvalidFixedArrayGeometry.into()
                } else {
                    SinkError::Write
                })
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    match error {
        SinkError::Write => assert!(!format),
        SinkError::Format(error) => {
            assert!(format);
            assert!(
                matches!(error, FormatError::InvalidFixedArrayGeometry),
                "{error:?}"
            );
        }
    }
    assert_eq!(calls, 2);
}

#[derive(Debug)]
enum SinkError {
    Format(FormatError),
    Write,
}

impl From<FormatError> for SinkError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

fn region(start: PhysicalAddress, bytes: &[u8]) -> Result<FileRegion<'_>, FormatError> {
    FileRegion::new(
        FileRange::new(start, u64::try_from(bytes.len()).unwrap())?,
        bytes,
    )
}

fn parse_header(widths: FormatWidths, base: u64, bytes: &[u8]) -> FixedArrayHeader {
    let base = BaseAddress::new(base);
    let locator = FixedArrayHeaderLocator::new(widths, base, StoredAddress::new(HEADER_AT));
    locator
        .parse(region(base.resolve(StoredAddress::new(HEADER_AT)).unwrap(), bytes).unwrap())
        .unwrap()
}

fn fixture_widths(widths: FormatWidths) -> Widths {
    Widths::new(
        usize::from(widths.offsets.get()),
        usize::from(widths.lengths.get()),
    )
}

const HEADER_AT: u64 = 0x40;
const DATA_AT: u64 = 0x100;
const PAGE_BITS: u8 = 4;
const CHUNK_BYTES: u64 = 16;
