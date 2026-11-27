use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::IndexSlots;
use crate::error::FormatError;
use crate::fixed_array;
use crate::fixed_array::header::HEADER_SIGNATURE;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

#[test]
fn build_fixed_array_valid_structure() {
    let chunks = vec![
        ChunkRecord {
            address: StoredAddress::new(0x1000),
            stored_size: 160,
            filter_mask: 0,
        },
        ChunkRecord {
            address: StoredAddress::new(0x10A0),
            stored_size: 160,
            filter_mask: 0,
        },
    ];
    let fa = fixed_array::build_fixed_array_at(
        &IndexSlots::dense(&chunks),
        160,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        false,
        StoredAddress::new(0x2000),
    )
    .unwrap();
    assert_eq!(&fa[0..4], b"FAHD");
    // FAHD size = 4+1+1+1+1+8+8+4 = 28
    // FADB starts at offset 28
    assert_eq!(&fa[28..32], b"FADB");
}

// Sweeps every count across the page size, where the data block becomes paged, and counts
// that leave a partial last page.
#[test]
fn fixed_array_len_matches_what_it_builds() {
    fn check(
        n: u64,
        chunk_bytes: u64,
        offset_size: OffsetWidth,
        length_size: LengthWidth,
        has_filters: bool,
    ) {
        let chunks: Vec<ChunkRecord> = (0..n)
            .map(|i| ChunkRecord {
                address: StoredAddress::new(0x1000 + i * 8),
                stored_size: 8,
                filter_mask: 0,
            })
            .collect();
        let planned = fixed_array::fixed_array_len(
            &IndexSlots::dense(&chunks),
            chunk_bytes,
            offset_size,
            length_size,
            has_filters,
        )
        .unwrap();
        let built = fixed_array::build_fixed_array_at(
            &IndexSlots::dense(&chunks),
            chunk_bytes,
            offset_size,
            length_size,
            has_filters,
            StoredAddress::new(0x10_0000),
        )
        .unwrap();
        assert_eq!(
            planned,
            built.len() as u64,
            "planned length must match the emitted array at n={n}, \
             chunk_bytes={chunk_bytes}, offset_size={offset_size:?}, \
             has_filters={has_filters}"
        );
    }

    for (offset_size, length_size) in [
        (OffsetWidth::Eight, LengthWidth::Eight),
        (OffsetWidth::Four, LengthWidth::Four),
    ] {
        for &has_filters in &[false, true] {
            // The array is paged past `1 << FIXED_ARRAY_PAGE_BITS` elements.
            for n in 0..=1_100u64 {
                check(n, 8, offset_size, length_size, has_filters);
            }
            // Several whole pages, and a count that leaves a partial one.
            for &n in &[4_096u64, 5_000, 100_000] {
                check(n, 8, offset_size, length_size, has_filters);
            }
        }
    }

    // Each chunk size selects a different chunk size field width.
    for &chunk_bytes in &chunk_record::CHUNK_BYTES {
        for &n in &[1u64, 1_024, 1_025, 5_000] {
            check(n, chunk_bytes, OffsetWidth::Eight, LengthWidth::Eight, true);
        }
    }
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
    let length =
        fixed_array::fixed_array_len(&slots, 8, OffsetWidth::Eight, LengthWidth::Eight, false)
            .unwrap();
    let start = u64::MAX - length + u64::from(overflow);
    let result = fixed_array::build_fixed_array_at(
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
