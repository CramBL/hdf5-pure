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
use crate::fixed_array::FixedArrayDataBlock;
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
fn initialized_page_emission_matches_independent_bytes(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(false, true)] filtered: bool,
    #[values(0, 16)] first: u64,
) {
    let widths = FormatWidths { offsets, lengths };
    let client = if filtered {
        fixture::Client::Filtered { size_bytes: 2 }
    } else {
        fixture::Client::Plain
    };
    let plan = FixedArrayPlan::new(17, CHUNK_BYTES, widths, filtered)
        .unwrap()
        .with_page_bits(PAGE_BITS)
        .unwrap();
    let placed = plan
        .place(StoredAddress::new(HEADER_AT), StoredAddress::new(DATA_AT))
        .unwrap();
    let encoded_header = placed.serialize_header().unwrap();
    assert_eq!(
        encoded_header,
        fixture::header(
            support::fixture_widths(widths),
            client,
            PAGE_BITS,
            17,
            Some(DATA_AT)
        )
    );
    let header = support::parse_header(widths, 0, &encoded_header);
    let block = header.data_block().unwrap();
    let page = block.page_for_slot(first).unwrap().unwrap();
    let slot = if first == 0 { 7 } else { 16 };
    let record = ChunkRecord {
        address: StoredAddress::new(0x3456),
        stored_size: 255,
        filter_mask: 0x1122_3344,
    };
    let window = page
        .window(slot, NonZeroUsize::new(1).unwrap())
        .unwrap()
        .unwrap();
    let mut output = Vec::new();
    page.emit_records::<FormatError>(&window, &[Some(record)], |bytes| {
        output.extend_from_slice(bytes);
        Ok(())
    })
    .unwrap();
    let slots = page.slots().unwrap();
    let mut records = vec![None; usize::try_from(slots.end - slots.start).unwrap()];
    records[usize::try_from(slot - slots.start).unwrap()] = Some(fixture::Entry {
        address: record.address.get(),
        stored_size: record.stored_size,
        filter_mask: record.filter_mask,
    });
    assert_eq!(
        output,
        fixture::page(support::fixture_widths(widths), client, &records)
    );
    let mut field = [0xcc; 32];
    let length = header
        .initialization_plan()
        .unwrap()
        .encode_record(&mut field, Some(record))
        .unwrap();
    assert_eq!(
        &field[..length],
        &fixture::page(
            support::fixture_widths(widths),
            client,
            &[Some(fixture::Entry {
                address: record.address.get(),
                stored_size: record.stored_size,
                filter_mask: record.filter_mask,
            })]
        )[..length]
    );
    assert_eq!(field[length], 0xcc);
    assert_eq!(
        page.checksummed_range().unwrap().end(),
        page.checksum_range().unwrap().start()
    );
    assert_eq!(
        block.checksummed_range().unwrap().end(),
        block.checksum_range().unwrap().start()
    );
}

#[rstest]
#[case::size(Some(ChunkRecord { address: StoredAddress::new(32), stored_size: 65_536, filter_mask: 0 }), FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
#[case::sentinel(Some(ChunkRecord { address: StoredAddress::new(65_535), stored_size: 1, filter_mask: 0 }), FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
#[case::address(Some(ChunkRecord { address: StoredAddress::new(65_536), stored_size: 1, filter_mask: 0 }), FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
fn page_emission_rejects_a_later_record_before_writing(
    #[case] record: Option<ChunkRecord>,
    #[case] expected: FormatError,
) {
    let widths = FormatWidths::from_sizes(2, 8).unwrap();
    let header = support::parse_header(
        widths,
        0,
        &fixture::header(
            support::fixture_widths(widths),
            fixture::Client::Filtered { size_bytes: 2 },
            PAGE_BITS,
            17,
            Some(DATA_AT),
        ),
    );
    let page = header
        .data_block()
        .unwrap()
        .page_for_slot(0)
        .unwrap()
        .unwrap();
    let window = page
        .window(7, NonZeroUsize::new(2).unwrap())
        .unwrap()
        .unwrap();
    let mut calls = 0;
    let error = page
        .emit_records::<FormatError>(&window, &[None, record], |_| {
            calls += 1;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(error, expected);
    assert_eq!(calls, 0);
    page.emit_records::<FormatError>(
        &window,
        &[
            None,
            Some(ChunkRecord {
                address: StoredAddress::new(65_534),
                stored_size: 65_535,
                filter_mask: u32::MAX,
            }),
        ],
        |_| {
            calls += 1;
            Ok(())
        },
    )
    .unwrap();
    assert!(calls > 0);
}

#[test]
fn initialized_page_emission_bounds_its_windows() {
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let header = support::parse_header(
        widths,
        0,
        &fixture::header(
            support::fixture_widths(widths),
            fixture::Client::Plain,
            16,
            65_537,
            Some(DATA_AT),
        ),
    );
    let page = header
        .data_block()
        .unwrap()
        .page_for_slot(65_000)
        .unwrap()
        .unwrap();
    let window = page
        .window(65_000, NonZeroUsize::new(1).unwrap())
        .unwrap()
        .unwrap();
    let mut total = 0;
    let mut calls = 0;
    let mut maximum = 0;
    page.emit_records::<FormatError>(
        &window,
        &[Some(ChunkRecord {
            address: StoredAddress::new(32),
            stored_size: CHUNK_BYTES,
            filter_mask: 0,
        })],
        |bytes| {
            total += bytes.len() as u64;
            calls += 1;
            maximum = maximum.max(bytes.len());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!((total, calls, maximum), (524_292, 131, 4_096));
    let mut calls = 0;
    let error = page
        .emit_records::<FormatError>(&window, &[None], |_| {
            calls += 1;
            Err(FormatError::InvalidFixedArrayClient(99))
        })
        .unwrap_err();
    assert_eq!(error, FormatError::InvalidFixedArrayClient(99));
    assert_eq!(calls, 1);
}

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
    let input = fixture::header(
        support::fixture_widths(widths),
        client,
        PAGE_BITS,
        entries,
        None,
    );
    let header = support::parse_header(widths, base, &input);
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
            support::fixture_widths(widths),
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
            support::fixture_widths(widths),
            client,
            PAGE_BITS,
            usize::try_from(entries).unwrap(),
            HEADER_AT
        )
    );
    let header = support::parse_header(widths, base, &encoded_header);
    let block = header.data_block().unwrap();
    let data = support::region(block.range().start(), &encoded).unwrap();
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

#[test]
fn empty_page_rejects_a_short_output_before_modifying_it() {
    let widths = FormatWidths::from_sizes(2, 2).unwrap();
    let header = support::parse_header(
        widths,
        0,
        &fixture::header(
            support::fixture_widths(widths),
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
        support::fixture_widths(widths),
        fixture::Client::Plain,
        PAGE_BITS,
        1,
        None,
    );
    let header = support::parse_header(widths, 512, &input);
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
            support::fixture_widths(widths),
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
    let header = support::parse_header(
        widths,
        0,
        &fixture::header(
            support::fixture_widths(widths),
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
