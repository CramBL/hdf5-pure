use hdf5_pure_core::__private::FileRange;
use rstest::rstest;

use super::support::{ARRAY_AT, FIRST_PAGED_SLOT, Fixture};
use crate::chunk_record;
use crate::error::FormatError;

#[derive(Clone, Copy, Debug)]
enum Block {
    Index,
    Super,
    Data,
    PagedData,
}

impl Block {
    fn fixture(self) -> Fixture {
        Fixture::new(
            match self {
                Self::Index | Self::Data => 20,
                Self::Super => 2_000,
                Self::PagedData => 140_000,
            },
            false,
        )
    }

    fn range(self, fixture: &Fixture) -> FileRange {
        match self {
            Self::Index => fixture
                .header()
                .unwrap()
                .index_block()
                .unwrap()
                .parse_range(),
            Self::Super => fixture.super_blocks()[0].parse_range(),
            Self::Data => fixture.data_blocks()[0].parse_range(),
            Self::PagedData => fixture.paged_data_blocks()[0].parse_range(),
        }
    }

    fn offset(self) -> u64 {
        match self {
            Self::Index | Self::Data => 0,
            Self::Super => 240,
            Self::PagedData => FIRST_PAGED_SLOT - 4,
        }
    }

    fn parse(self, fixture: &Fixture, range: FileRange) -> FormatError {
        let region = fixture.read(range).unwrap();
        match self {
            Self::Index => {
                let block = fixture.header().unwrap().index_block().unwrap();
                block.parse(region).unwrap_err()
            }
            Self::Super => fixture.super_blocks()[0].parse(region).unwrap_err(),
            Self::Data => fixture.data_blocks()[0].parse(region).unwrap_err(),
            Self::PagedData => fixture.paged_data_blocks()[0].parse(region).unwrap_err(),
        }
    }
}

#[rstest]
#[case::signature(0, b"X", FormatError::InvalidExtensibleArraySignature)]
#[case::version(4, &[1], FormatError::InvalidExtensibleArrayVersion(1))]
#[case::client(5, &[1], FormatError::InvalidExtensibleArrayBlockClient { expected: 0, actual: 1 })]
#[case::header_address(6, &[0x81], FormatError::InvalidExtensibleArrayBlockOwner { expected: ARRAY_AT, actual: 0x81 })]
fn a_malformed_block_prefix_is_rejected(
    #[case] field: usize,
    #[case] value: &[u8],
    #[case] expected: FormatError,
    #[values(Block::Index, Block::Super, Block::Data, Block::PagedData)] block: Block,
) {
    let mut fixture = block.fixture();
    let range = block.range(&fixture);
    let at = fixture.local(range.start()) + field;
    fixture.bytes[at..at + value.len()].copy_from_slice(value);
    fixture.restamp(range);
    assert_eq!(block.parse(&fixture, range), expected);
}

#[rstest]
fn a_block_offset_other_than_its_position_is_rejected(
    #[values(Block::Super, Block::Data, Block::PagedData)] block: Block,
) {
    let mut fixture = block.fixture();
    let range = block.range(&fixture);
    let at = fixture.local(range.start()) + 4 + 2 + 8;
    let offset = block.offset();
    fixture.bytes[at..at + 4].copy_from_slice(&u32::try_from(offset + 1).unwrap().to_le_bytes());
    fixture.restamp(range);
    assert_eq!(
        block.parse(&fixture, range),
        FormatError::InvalidExtensibleArrayBlockOffset {
            expected: offset,
            actual: offset + 1,
        }
    );
}

// Data block 1 is the one data block of super block 1, at offset 16, and data block 3 the
// second of super block 2, at offset 80.
#[rstest]
#[case::spec_value(1, 16, None)]
#[case::libhdf5_value(1, 48, None)]
#[case::other_value(1, 17, Some(16))]
#[case::spec_value_of_a_later_block(3, 80, None)]
#[case::libhdf5_value_of_a_later_block(3, 48 + 3 * 32, None)]
#[case::other_value_of_a_later_block(3, 48 + 2 * 32 + 1, Some(80))]
fn a_data_block_the_index_block_holds_takes_either_offset(
    #[case] block: usize,
    #[case] offset: u32,
    #[case] rejected: Option<u64>,
) {
    let mut fixture = Fixture::new(244, true);
    let range = fixture.data_blocks()[block].parse_range();
    let at = fixture.local(range.start()) + 4 + 2 + 8;
    fixture.bytes[at..at + 4].copy_from_slice(&offset.to_le_bytes());
    fixture.restamp(range);
    let records = fixture.records(8);
    match rejected {
        Some(expected) => assert_eq!(
            records.unwrap_err(),
            FormatError::InvalidExtensibleArrayBlockOffset {
                expected,
                actual: u64::from(offset),
            }
        ),
        None => assert_eq!(
            records.unwrap(),
            chunk_record::slotted(chunk_record::chunks(8, 244, true))
        ),
    }
}

// Data block 1 of super block 4 is the 8th data block in the array, and starts 64 elements
// past the super block's offset of 240.
#[test]
fn a_data_block_a_super_block_holds_takes_its_offset_alone() {
    let mut fixture = Fixture::new(500, false);
    let block = fixture.data_blocks()[6 + 1];
    assert_eq!(block.slots().unwrap(), 4 + 240 + 64..4 + 240 + 2 * 64);
    let range = block.parse_range();
    let at = fixture.local(range.start()) + 4 + 2 + 8;
    let ordinal_offset = 240u32 + 7 * 64;
    fixture.bytes[at..at + 4].copy_from_slice(&ordinal_offset.to_le_bytes());
    fixture.restamp(range);
    assert_eq!(
        fixture.records(8).unwrap_err(),
        FormatError::InvalidExtensibleArrayBlockOffset {
            expected: 240 + 64,
            actual: u64::from(ordinal_offset),
        }
    );
}

#[rstest]
fn a_header_address_at_the_physical_header_position_is_rejected(
    #[values(Block::Index, Block::Super, Block::Data, Block::PagedData)] block: Block,
) {
    let base = 1 << 40;
    let mut fixture = block.fixture();
    fixture.base = base;
    let range = block.range(&fixture);
    let at = fixture.local(range.start()) + 6;
    fixture.bytes[at..at + 8].copy_from_slice(&(base + ARRAY_AT).to_le_bytes());
    fixture.restamp(range);
    assert_eq!(
        block.parse(&fixture, range),
        FormatError::InvalidExtensibleArrayBlockOwner {
            expected: base + ARRAY_AT,
            actual: 2 * base + ARRAY_AT,
        }
    );
}

#[derive(Clone, Copy, Debug)]
enum Truncated {
    Header,
    IndexBlock,
    SuperBlock,
    DataBlock,
    PagedDataBlock,
    Page,
}

#[rstest]
fn a_region_short_of_its_range_is_rejected(
    #[values(
        Truncated::Header,
        Truncated::IndexBlock,
        Truncated::SuperBlock,
        Truncated::DataBlock,
        Truncated::PagedDataBlock,
        Truncated::Page
    )]
    truncated: Truncated,
) {
    let fixture = Fixture::new(140_000, false);
    let short = |range: FileRange| {
        fixture
            .read(FileRange::new(range.start(), range.length() - 1).unwrap())
            .unwrap()
    };
    let (range, err) = match truncated {
        Truncated::Header => {
            let locator = fixture.locator();
            let range = locator.parse_range().unwrap();
            (range, locator.parse(short(range)).unwrap_err())
        }
        Truncated::IndexBlock => {
            let block = fixture.header().unwrap().index_block().unwrap();
            let range = block.parse_range();
            (range, block.parse(short(range)).unwrap_err())
        }
        Truncated::SuperBlock => {
            let block = fixture.super_blocks()[0];
            let range = block.parse_range();
            (range, block.parse(short(range)).unwrap_err())
        }
        Truncated::DataBlock => {
            let block = fixture.data_blocks()[0];
            let range = block.parse_range();
            (range, block.parse(short(range)).unwrap_err())
        }
        Truncated::PagedDataBlock => {
            let block = fixture.paged_data_blocks()[0];
            let range = block.parse_range();
            (range, block.parse(short(range)).unwrap_err())
        }
        Truncated::Page => {
            let page = *fixture
                .pages(*fixture.paged_data_blocks().last().unwrap())
                .last()
                .unwrap();
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

#[cfg(feature = "checksum")]
#[derive(Clone, Copy, Debug)]
enum Corrupted {
    Header,
    IndexBlock,
    SuperBlock,
    DataBlock,
    PagedDataBlock,
    FirstPage,
    LastPage,
}

#[cfg(feature = "checksum")]
#[rstest]
fn a_corrupted_checksum_is_rejected(
    #[values(
        Corrupted::Header,
        Corrupted::IndexBlock,
        Corrupted::SuperBlock,
        Corrupted::DataBlock,
        Corrupted::PagedDataBlock,
        Corrupted::FirstPage,
        Corrupted::LastPage
    )]
    corrupted: Corrupted,
    #[values(false, true)] has_filters: bool,
) {
    let mut fixture = Fixture::new(140_000, has_filters);
    let range = match corrupted {
        Corrupted::Header => fixture.header().unwrap().range(),
        Corrupted::IndexBlock => fixture
            .header()
            .unwrap()
            .index_block()
            .unwrap()
            .parse_range(),
        Corrupted::SuperBlock => fixture.super_blocks()[0].parse_range(),
        Corrupted::DataBlock => fixture.data_blocks()[0].parse_range(),
        Corrupted::PagedDataBlock => fixture.paged_data_blocks()[0].parse_range(),
        Corrupted::FirstPage => fixture.pages(fixture.paged_data_blocks()[0])[0].parse_range(),
        Corrupted::LastPage => fixture
            .pages(*fixture.paged_data_blocks().last().unwrap())
            .last()
            .unwrap()
            .parse_range(),
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
