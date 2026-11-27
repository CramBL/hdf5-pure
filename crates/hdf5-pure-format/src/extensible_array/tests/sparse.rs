use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;

use super::support::WIDTHS;
use crate::address::{BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt};
use crate::chunk_record::{ChunkRecord, IndexSlots};
use crate::error::FormatError;
use crate::extensible_array::header::HEADER_SIGNATURE;
use crate::extensible_array::{
    self, ExtensibleArrayDataBlock, ExtensibleArrayDataBlockRef, ExtensibleArrayHeaderLocator,
    ExtensibleArrayPageRef, ExtensibleArraySuperBlockRef,
};
use crate::width::{LengthWidth, OffsetWidth};

// An array of 64 index bits whose one chunk is in page 255 of data block 1 of super block 28,
// past slot `u32::MAX`, and that allocates no other block.
struct SparseArray {
    blocks: Vec<(u64, Vec<u8>)>,
}

impl SparseArray {
    const SLOT: u64 = 4_295_490_548;

    fn new(base: u64) -> Self {
        let header_len = 12 + 6 * 8 + 8 + 4;
        let index_block_at = base + header_len;
        // The prefix, 4 elements, 6 data block addresses, and 57 super block addresses.
        let index_block_len = 14 + 4 * 8 + (6 + 57) * 8 + 4;
        let super_block_at = index_block_at + index_block_len;
        // The prefix, the block offset, 256 pages for each of 16_384 data blocks, the data
        // block addresses, and the checksum.
        let bitmap_len = 16_384 * 32;
        let super_block_len = 14 + 8 + bitmap_len + 16_384 * 8 + 4;
        let data_block_at = super_block_at + super_block_len;

        let mut header = Vec::new();
        header.extend_from_slice(b"EAHD");
        header.extend_from_slice(&[0, 0, 8, 64, 4, 16, 4, 10]);
        for statistic in [
            1,
            super_block_len,
            1,
            26 + 256 * 8_196,
            Self::SLOT + 1,
            4 + 262_144,
        ] {
            header.extend_from_slice(&statistic.to_le_bytes());
        }
        header.extend_from_slice(&index_block_at.to_le_bytes());
        test_util::checksum::append(&mut header);

        let mut index_block = Vec::new();
        index_block.extend_from_slice(b"EAIB");
        index_block.extend_from_slice(&[0, 0]);
        index_block.extend_from_slice(&base.to_le_bytes());
        index_block.extend_from_slice(&[0xff; 4 * 8]);
        for block in 0..6 + 57 {
            let address = if block == 6 + 24 {
                super_block_at
            } else {
                u64::MAX
            };
            index_block.extend_from_slice(&address.to_le_bytes());
        }
        test_util::checksum::append(&mut index_block);

        let mut super_block = Vec::new();
        super_block.extend_from_slice(b"EASB");
        super_block.extend_from_slice(&[0, 0]);
        super_block.extend_from_slice(&base.to_le_bytes());
        super_block.extend_from_slice(&4_294_967_280u64.to_le_bytes());
        let mut bitmap = vec![0; usize::try_from(bitmap_len).unwrap()];
        bitmap[63] = 0x01;
        super_block.extend_from_slice(&bitmap);
        for data_block in 0..16_384 {
            let address = if data_block == 1 {
                data_block_at
            } else {
                u64::MAX
            };
            super_block.extend_from_slice(&address.to_le_bytes());
        }
        test_util::checksum::append(&mut super_block);

        let mut data_block = Vec::new();
        data_block.extend_from_slice(b"EADB");
        data_block.extend_from_slice(&[0, 0]);
        data_block.extend_from_slice(&base.to_le_bytes());
        data_block.extend_from_slice(&4_295_229_424u64.to_le_bytes());
        test_util::checksum::append(&mut data_block);
        let mut page = vec![0xff; 1_024 * 8];
        page[..8].copy_from_slice(&0x20u64.to_le_bytes());
        test_util::checksum::append(&mut page);
        let page_at = data_block_at + 26 + 255 * 8_196;

        Self {
            blocks: vec![
                (base, header),
                (index_block_at, index_block),
                (super_block_at, super_block),
                (data_block_at, data_block),
                (page_at, page),
            ],
        }
    }

    fn read(&self, range: FileRange) -> FileRegion<'_> {
        let (at, bytes) = self
            .blocks
            .iter()
            .find(|&&(at, _)| at == range.start().get())
            .unwrap();
        FileRegion::new(
            FileRange::new(PhysicalAddress::new(*at), bytes.len() as u64).unwrap(),
            bytes,
        )
        .unwrap()
        .subregion(range)
        .unwrap()
    }
}

#[rstest]
fn a_slot_past_u32_max_is_read(#[values(0, 1 << 32)] base: u64) {
    let array = SparseArray::new(base);
    let locator =
        ExtensibleArrayHeaderLocator::new(WIDTHS, BaseAddress::ZERO, StoredAddress::new(base));
    let header = locator
        .parse(array.read(locator.parse_range().unwrap()))
        .unwrap();
    let index_block = header.index_block().unwrap();
    let index_block = index_block
        .parse(array.read(index_block.parse_range()))
        .unwrap();
    assert_eq!(index_block.data_blocks().count(), 0);
    let super_blocks: Vec<ExtensibleArraySuperBlockRef> = index_block
        .super_blocks()
        .collect::<Result<_, _>>()
        .unwrap();
    let &[super_block] = &super_blocks[..] else {
        panic!("expected one super block, got {super_blocks:?}");
    };
    let super_block = super_block
        .parse(array.read(super_block.parse_range()))
        .unwrap();
    let data_blocks: Vec<ExtensibleArrayDataBlockRef<'_>> =
        super_block.data_blocks().collect::<Result<_, _>>().unwrap();
    let &[data_block] = &data_blocks[..] else {
        panic!("expected one data block, got {data_blocks:?}");
    };
    let ExtensibleArrayDataBlock::Pages(pages) = data_block
        .parse(array.read(data_block.parse_range()))
        .unwrap()
    else {
        panic!("the data block {data_block:?} is not paged");
    };
    let pages: Vec<ExtensibleArrayPageRef> = pages.pages().collect::<Result<_, _>>().unwrap();
    let &[page] = &pages[..] else {
        panic!("expected one page, got {pages:?}");
    };
    let records: Vec<(u64, ChunkRecord)> = page
        .parse(array.read(page.parse_range()))
        .unwrap()
        .records(8)
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        records,
        vec![(
            SparseArray::SLOT,
            ChunkRecord {
                address: StoredAddress::new(0x20),
                stored_size: 8,
                filter_mask: 0,
            }
        )]
    );
}
#[rstest]
#[case::maximum_end(false)]
#[case::overflow(true)]
fn builder_checks_the_complete_exclusive_end(#[case] overflow: bool) {
    let chunks = [ChunkRecord {
        address: StoredAddress::new(16),
        stored_size: 8,
        filter_mask: 0,
    }; 2];
    let slots = IndexSlots::dense(&chunks);
    let length = extensible_array::extensible_array_len(
        &slots,
        8,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        false,
    )
    .unwrap();
    let start = u64::MAX - length + u64::from(overflow);
    let result = extensible_array::build_extensible_array_at(
        &slots,
        8,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        false,
        StoredAddress::new(start),
    );
    if overflow {
        assert_eq!(
            result.unwrap_err(),
            FormatError::OffsetOverflow {
                offset: start,
                length
            }
        );
    } else {
        let bytes = result.unwrap();
        assert_eq!(bytes.len() as u64, length);
        assert_eq!(&bytes[..HEADER_SIGNATURE.len()], &HEADER_SIGNATURE);
        assert_eq!(
            StoredAddress::new(start).checked_add(length).unwrap(),
            StoredAddress::new(u64::MAX)
        );
    }
}
