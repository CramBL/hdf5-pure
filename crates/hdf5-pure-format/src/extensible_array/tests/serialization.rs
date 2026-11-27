use alloc::vec;
use alloc::vec::Vec;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::chunk_record::{self, ChunkRecord, IndexSlots};
use crate::extensible_array::{self, ExtensibleArrayStats, SlotOccupancy};
use crate::width::{LengthWidth, OffsetWidth};

#[test]
fn header_and_index_block_signatures_appear_at_their_expected_offsets() {
    let chunks = vec![
        ChunkRecord {
            address: StoredAddress::new(0x1000),
            stored_size: 80,
            filter_mask: 0,
        },
        ChunkRecord {
            address: StoredAddress::new(0x1050),
            stored_size: 80,
            filter_mask: 0,
        },
    ];
    let ea = extensible_array::build_extensible_array_at(
        &IndexSlots::dense(&chunks),
        80,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        false,
        StoredAddress::new(0x2000),
    )
    .unwrap();
    assert_eq!(&ea[0..4], b"EAHD");
    // `EAIB` follows `EAHD`: 12 fixed + 6*8 stats + 8 address + 4 checksum = 72
    let header_len = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * 8 + 8 + 4;
    assert_eq!(&ea[header_len..header_len + 4], b"EAIB");
}

#[test]
fn extensible_array_stats_matches_what_it_builds() {
    for &n in &[1u64, 4, 20, 100, 244, 300, 2000, 50000, 131056, 140000] {
        let chunks: Vec<ChunkRecord> = (0..n)
            .map(|i| ChunkRecord {
                address: StoredAddress::new(0x1000 + i * 8),
                stored_size: 8,
                filter_mask: 0,
            })
            .collect();
        let ea = extensible_array::build_extensible_array_at(
            &IndexSlots::dense(&chunks),
            8,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
            StoredAddress::new(0x100000),
        )
        .unwrap();
        // Parse the 6 stats from the EAHD (12-byte fixed prefix, then 6 * ls).
        let stat =
            |k: usize| u64::from_le_bytes(ea[12 + k * 8..12 + k * 8 + 8].try_into().unwrap());
        let built = ExtensibleArrayStats {
            nsuper_blks: stat(0),
            super_blk_size: stat(1),
            ndata_blks: stat(2),
            data_blk_size: stat(3),
            max_idx_set: stat(4),
            nelmts: stat(5),
        };
        let computed = crate::extensible_array::extensible_array_layout(
            SlotOccupancy::Dense(n),
            n,
            8,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            false,
        )
        .unwrap()
        .stats;
        assert_eq!(computed, built, "stats mismatch at n={n}");
    }
}

// Sweeps every count through the first super block, then the deeper super blocks and the
// first paged data block.
#[test]
fn extensible_array_len_matches_what_it_builds() {
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
        let planned = extensible_array::extensible_array_len(
            &IndexSlots::dense(&chunks),
            chunk_bytes,
            offset_size,
            length_size,
            has_filters,
        )
        .unwrap();
        let built = extensible_array::build_extensible_array_at(
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
            // Contiguous across the inline, direct-block and first
            // super-block ranges.
            for n in 0..=250u64 {
                check(n, 8, offset_size, length_size, has_filters);
            }
            // The deeper super blocks, and the paged boundary: 131,060 is the
            // last element the unpaged super block 12 holds, 131,061 the first
            // that allocates a paged data block.
            for &n in &[300u64, 2_000, 50_000, 131_060, 131_061, 140_000] {
                check(n, 8, offset_size, length_size, has_filters);
            }
        }
    }

    // The stored chunk-size width varies across these filtered element encodings.
    for &chunk_bytes in &chunk_record::CHUNK_BYTES {
        for &n in &[1u64, 5, 244, 300, 2_000] {
            check(n, chunk_bytes, OffsetWidth::Eight, LengthWidth::Eight, true);
        }
    }
    // The four chunk sizes select four different widths, measured on an array of no chunks,
    // whose width comes from `chunk_bytes` alone.
    let widths: Vec<usize> = chunk_record::CHUNK_BYTES
        .iter()
        .map(|&chunk_bytes| {
            crate::extensible_array::extensible_array_layout(
                SlotOccupancy::Dense(0),
                0,
                chunk_bytes,
                OffsetWidth::Eight,
                LengthWidth::Eight,
                true,
            )
            .unwrap();
            chunk_record::chunk_element_encoding(chunk_bytes, OffsetWidth::Eight, true)
                .chunk_size_bytes
        })
        .collect();
    let mut distinct = widths.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        chunk_record::CHUNK_BYTES.len(),
        "each chunk size must select a different compressed-size field width, got {widths:?}"
    );
}
