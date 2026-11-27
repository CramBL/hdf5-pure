use hdf5_pure_core::__private::FileRange;
use rstest::rstest;

use super::support::ARRAY_AT;
use super::support::Fixture;
use super::support::WIDTHS;
use crate::chunk_record;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[rstest]
fn an_array_reads_back_at_every_width(
    #[values(
        FormatWidths { offsets: OffsetWidth::Two, lengths: LengthWidth::Two },
        FormatWidths { offsets: OffsetWidth::Four, lengths: LengthWidth::Four },
        FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight }
    )]
    widths: FormatWidths,
    #[values(0, 5, 1_024, 1_025, 3_000)] n: u64,
    #[values(false, true)] has_filters: bool,
    #[values(8, 300, 100_000, 1 << 32)] chunk_bytes: u64,
) {
    let fixture = Fixture::at(0, widths, chunk_bytes, n, has_filters);
    let read = fixture.records(chunk_bytes).unwrap();
    assert_eq!(
        read,
        chunk_record::slotted(chunk_record::chunks(chunk_bytes, n, has_filters))
    );
}

#[rstest]
#[case::plain(0)]
#[case::userblock(512)]
#[case::above_u32(1 << 40)]
fn an_array_reads_at_its_physical_position(#[case] base: u64, #[values(5, 3_000)] n: u64) {
    let fixture = Fixture::at(base, WIDTHS, 8, n, true);
    let header = fixture.header().unwrap();
    assert_eq!(header.range().start().get(), base + ARRAY_AT);
    assert_eq!(
        fixture.records(8).unwrap(),
        chunk_record::slotted(chunk_record::chunks(8, n, true))
    );
}

#[rstest]
#[case::header(Truncated::Header)]
#[case::prefix(Truncated::Prefix)]
#[case::page(Truncated::Page)]
fn a_region_short_of_its_range_is_rejected(#[case] truncated: Truncated) {
    let fixture = Fixture::new(3_000, false);
    let short = |range: FileRange| {
        fixture
            .image()
            .subregion(FileRange::new(range.start(), range.length() - 1).unwrap())
            .unwrap()
    };
    let (range, err) = match truncated {
        Truncated::Header => {
            let locator = fixture.locator();
            let range = locator.parse_range().unwrap();
            (range, locator.parse(short(range)).unwrap_err())
        }
        Truncated::Prefix => {
            let block = fixture.data_block();
            let range = block.parse_range().unwrap();
            (range, block.parse(short(range)).unwrap_err())
        }
        Truncated::Page => {
            let page = *fixture.pages().last().unwrap();
            let range = page.parse_range();
            (range, page.parse(short(range)).unwrap_err())
        }
    };
    assert_eq!(
        err,
        FormatError::InsufficientRegion {
            required_start: range.start().get(),
            required_length: range.length(),
            provided_start: range.start().get(),
            provided_length: range.length() - 1,
        }
    );
}

#[derive(Clone, Copy, Debug)]
enum Truncated {
    Header,
    Prefix,
    Page,
}

#[cfg(feature = "checksum")]
#[rstest]
#[case::header(Corrupted::Header)]
#[case::unpaged_data_block(Corrupted::DataBlock)]
#[case::prefix(Corrupted::Prefix)]
#[case::first_page(Corrupted::FirstPage)]
#[case::last_page(Corrupted::LastPage)]
fn a_corrupted_checksum_is_rejected(
    #[case] corrupted: Corrupted,
    #[values(false, true)] has_filters: bool,
) {
    let n = if matches!(corrupted, Corrupted::DataBlock) {
        5
    } else {
        3_000
    };
    let mut fixture = Fixture::new(n, has_filters);
    let range = match corrupted {
        Corrupted::Header => fixture.header().unwrap().range(),
        Corrupted::DataBlock | Corrupted::Prefix => fixture.data_block().parse_range().unwrap(),
        Corrupted::FirstPage => fixture.pages()[0].parse_range(),
        Corrupted::LastPage => fixture.pages().last().unwrap().parse_range(),
    };
    let start = fixture.local(range.start());
    let end = fixture.local(range.end());
    fixture.bytes[end - 1] ^= 0x01;
    let stored = u32::from_le_bytes(fixture.bytes[end - 4..end].try_into().unwrap());
    let computed = crate::checksum::jenkins_lookup3(&fixture.bytes[start..end - 4]);
    assert_eq!(
        fixture.records(8).unwrap_err(),
        FormatError::ChecksumMismatch {
            expected: stored,
            computed,
        }
    );
}

#[cfg(feature = "checksum")]
#[derive(Clone, Copy, Debug)]
enum Corrupted {
    Header,
    DataBlock,
    Prefix,
    FirstPage,
    LastPage,
}
