use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;

use super::support::Fixture;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::error::FormatError;
use crate::fixed_array::FixedArrayEntryWindow;
use crate::fixed_array::data_block::BlockStorage;
use crate::fixed_array::data_block::DATA_BLOCK_SIGNATURE;
use crate::fixed_array::framing::VERSION;
use crate::fixed_array::geometry::Geometry;
use crate::width::OffsetWidth;

#[rstest]
#[case::first(0, 14, 8_192, 1_024)]
#[case::last_full((1 << 29) - 1_024, 4_294_959_118, 8_192, 1 << 29)]
#[case::last_entry((1 << 29) - 1, 4_294_967_302, 8, 1 << 29)]
fn unpaged_windows_keep_full_width_geometry(
    #[case] first_slot: u64,
    #[case] offset: u64,
    #[case] length: u64,
    #[case] end_slot: u64,
) {
    let mut fixture = Fixture::new(5, false);
    let at = fixture.header_at();
    fixture.bytes[at + 7] = 29;
    fixture.bytes[at + 8..at + 16].copy_from_slice(&(1u64 << 29).to_le_bytes());
    fixture.restamp_header();
    let block = fixture.data_block();
    assert_eq!(block.range().length(), 4_294_967_314);
    let prefix = block.unpaged_prefix_range().unwrap().unwrap();
    let unpaged = block
        .parse_unpaged_prefix(fixture.image().subregion(prefix).unwrap())
        .unwrap();
    let window = unpaged
        .window(first_slot, NonZeroUsize::new(1_024).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(
        window.parse_range().start().get(),
        block.range().start().get() + offset
    );
    assert_eq!(window.parse_range().length(), length);
    assert_eq!(window.end_slot(), end_slot);
    assert_eq!(unpaged.checksummed_range().unwrap().length(), 4_294_967_310);
    assert_eq!(
        unpaged.checksum_range().unwrap().start().get(),
        block.range().start().get() + 4_294_967_310
    );
    assert_eq!(unpaged.window(1 << 29, NonZeroUsize::MIN).unwrap(), None);
}

#[test]
fn unpaged_window_parsing_preserves_large_slots() {
    let geometry = Geometry::new(
        ElementEncoding::new(
            OffsetWidth::Eight,
            chunk_record::ArrayClient::Chunks.id(),
            8,
        )
        .unwrap(),
        (1 << 32) + 4,
        33,
    )
    .unwrap();
    let block = BlockStorage::new(geometry, Some(PhysicalAddress::new(28)))
        .unwrap()
        .at(BaseAddress::ZERO, PhysicalAddress::new(0))
        .unwrap();
    let mut prefix = Vec::from(DATA_BLOCK_SIGNATURE);
    prefix.extend_from_slice(&[VERSION, chunk_record::ArrayClient::Chunks.id()]);
    prefix.extend_from_slice(&0u64.to_le_bytes());
    let range = block.unpaged_prefix_range().unwrap().unwrap();
    let unpaged = block
        .parse_unpaged_prefix(FileRegion::new(range, &prefix).unwrap())
        .unwrap();
    let window = unpaged
        .window(1 << 32, NonZeroUsize::new(2).unwrap())
        .unwrap()
        .unwrap();
    let mut bytes = Vec::from(123u64.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        window
            .parse(FileRegion::new(window.parse_range(), &bytes).unwrap())
            .unwrap()
            .records(8)
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        vec![(
            1 << 32,
            ChunkRecord {
                address: StoredAddress::new(123),
                stored_size: 8,
                filter_mask: 0
            }
        )]
    );
}

#[rstest]
#[case::maximum(u64::MAX - 1, Ok(u64::MAX))]
#[case::overflow(u64::MAX, Err(FormatError::InvalidFixedArrayGeometry))]
fn entry_window_checks_its_exclusive_end(
    #[case] first: u64,
    #[case] expected: Result<u64, FormatError>,
) {
    let encoding = ElementEncoding::new(
        OffsetWidth::Eight,
        chunk_record::ArrayClient::Chunks.id(),
        8,
    )
    .unwrap();
    assert_eq!(
        FixedArrayEntryWindow::new(PhysicalAddress::new(0), first, 1, encoding)
            .map(|window| window.end_slot()),
        expected,
    );
}
