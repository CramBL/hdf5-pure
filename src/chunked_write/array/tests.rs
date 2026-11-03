use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::convert::Infallible;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::ExtensibleArrayDataBlock;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use hdf5_pure_format::__private::FixedArrayDataBlock;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use hdf5_pure_format::__private::IndexSlots;
use rstest::rstest;

use super::ArrayMeasure;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_write::ChunkArrayGeometry;
use crate::chunked_write::ChunkArrayKind;
use crate::error::FormatError;
use crate::file_emission::EmissionError;
use crate::width::FormatWidths;

#[rstest]
#[case::empty_fixed(ChunkArrayKind::FixedArray, &[], 1, &[28, 26])]
#[case::empty_extensible(ChunkArrayKind::ExtensibleArray, &[], 1, &[72, 298])]
#[case::paged_fixed(ChunkArrayKind::FixedArray, &[0, 1024], 1025, &[28, 8227])]
#[case::extensible_data(ChunkArrayKind::ExtensibleArray, &[4], 5, &[72, 298, 150])]
fn visits_complete_blocks(
    #[case] kind: ChunkArrayKind,
    #[case] occupied: &[u64],
    #[case] slots: u64,
    #[case] expected: &[u64],
) {
    let measure = measure(kind, occupied, slots, false);
    let mut lengths = Vec::new();
    measure
        .visit_blocks(|length| {
            lengths.push(length);
            Ok(())
        })
        .unwrap();
    assert_eq!(lengths, expected);
    assert_eq!(measure.length().unwrap(), expected.iter().sum::<u64>());
}

#[rstest]
#[case::empty_fixed(ChunkArrayKind::FixedArray, &[], 5)]
#[case::empty_extensible(ChunkArrayKind::ExtensibleArray, &[], 5)]
#[case::paged_fixed(ChunkArrayKind::FixedArray, &[0, 1024], 1025)]
#[case::extensible_data(ChunkArrayKind::ExtensibleArray, &[4], 5)]
#[case::paged_extensible(ChunkArrayKind::ExtensibleArray, &[131_072], 131_073)]
fn contiguous_encoding_preserves_array_bytes(
    #[case] kind: ChunkArrayKind,
    #[case] occupied: &[u64],
    #[case] slots: u64,
    #[values(false, true)] filtered: bool,
) {
    let records = records(occupied, filtered);
    let slots_view = IndexSlots::new(&records, occupied, slots).unwrap();
    let address = StoredAddress::new(4096);
    let expected = kind
        .build_at(
            crate::file_writer::WIDTHS,
            &slots_view,
            8,
            filtered,
            address,
        )
        .unwrap();
    let measure = measure(kind, occupied, slots, filtered);
    assert_eq!(measure.build_at(&slots_view, address).unwrap(), expected);
}

#[rstest]
#[case::fixed(ChunkArrayKind::FixedArray, &[0, 1], 2)]
#[case::extensible(ChunkArrayKind::ExtensibleArray, &[0, 4], 5)]
fn writes_complete_blocks_at_disjoint_addresses(
    #[case] kind: ChunkArrayKind,
    #[case] occupied: &[u64],
    #[case] slots: u64,
    #[values(false, true)] filtered: bool,
) {
    let measure = measure(kind, occupied, slots, filtered);
    let records = records(occupied, filtered);
    let slots = IndexSlots::new(&records, occupied, slots).unwrap();
    let mut lengths = Vec::new();
    measure
        .visit_blocks(|length| {
            lengths.push(length);
            Ok(())
        })
        .unwrap();
    let addresses: Vec<_> = (0..lengths.len())
        .map(|index| StoredAddress::new(16_384 - u64::try_from(index).unwrap() * 4096))
        .collect();
    let end = addresses
        .iter()
        .zip(&lengths)
        .map(|(address, &length)| address.checked_add(length).unwrap().get())
        .max()
        .unwrap();
    let mut image = alloc::vec![0; usize::try_from(end).unwrap()];
    let mut observed = Vec::new();
    measure
        .write_at(&slots, &addresses, |address, bytes| {
            observed.push((address, u64::try_from(bytes.len()).unwrap()));
            let start = usize::try_from(address.get()).unwrap();
            image[start..start + bytes.len()].copy_from_slice(bytes);
            Ok::<(), Infallible>(())
        })
        .unwrap();
    assert_eq!(
        observed,
        addresses
            .iter()
            .copied()
            .zip(lengths.iter().copied())
            .collect::<Vec<_>>()
    );
    let base = BaseAddress::new(1_u64 << 40);
    let ranges: Vec<_> = addresses
        .iter()
        .zip(&lengths)
        .map(|(&address, &length)| FileRange::new(base.resolve(address).unwrap(), length).unwrap())
        .collect();
    let region = FileRegion::new(
        FileRange::new(
            base.resolve(StoredAddress::ZERO).unwrap(),
            u64::try_from(image.len()).unwrap(),
        )
        .unwrap(),
        &image,
    )
    .unwrap();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let actual = match kind {
        ChunkArrayKind::FixedArray => {
            let header = FixedArrayHeaderLocator::new(widths, base, addresses[0])
                .parse(region)
                .unwrap();
            assert_eq!(header.range(), ranges[0]);
            let data = header.data_block().unwrap();
            assert_eq!(base.relative(data.range().start()).unwrap(), addresses[1]);
            assert_eq!(data.range(), ranges[1]);
            let FixedArrayDataBlock::Elements(elements) = data.parse(region).unwrap() else {
                panic!("expected an unpaged fixed array data block");
            };
            elements.records(8).collect::<Result<Vec<_>, _>>().unwrap()
        }
        ChunkArrayKind::ExtensibleArray => {
            let header = ExtensibleArrayHeaderLocator::new(widths, base, addresses[0])
                .parse(region)
                .unwrap();
            assert_eq!(header.range(), ranges[0]);
            assert_eq!(header.header().index_block_address, addresses[1]);
            let index = header.index_block().unwrap();
            assert_eq!(index.range(), ranges[1]);
            let index = index.parse(region).unwrap();
            assert_eq!(
                index.super_blocks().collect::<Result<Vec<_>, _>>().unwrap(),
                []
            );
            let mut actual = index
                .elements()
                .records(8)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let blocks = index.data_blocks().collect::<Result<Vec<_>, _>>().unwrap();
            let [data] = blocks.as_slice() else {
                panic!("expected one extensible array data block, got {blocks:?}");
            };
            assert_eq!(base.relative(data.range().start()).unwrap(), addresses[2]);
            assert_eq!(data.range(), ranges[2]);
            let ExtensibleArrayDataBlock::Elements(elements) = data.parse(region).unwrap() else {
                panic!("expected an unpaged extensible array data block");
            };
            actual.extend(elements.records(8).collect::<Result<Vec<_>, _>>().unwrap());
            actual
        }
    };
    assert_eq!(
        actual,
        occupied.iter().copied().zip(records).collect::<Vec<_>>()
    );
}

#[rstest]
#[case::fixed(ChunkArrayKind::FixedArray, &[0, 1], 2)]
#[case::extensible(ChunkArrayKind::ExtensibleArray, &[4], 5)]
fn callback_errors_stop_block_emission(
    #[case] kind: ChunkArrayKind,
    #[case] occupied: &[u64],
    #[case] slots: u64,
    #[values(0, 1)] failure: usize,
) {
    let measure = measure(kind, occupied, slots, false);
    let records = records(occupied, false);
    let slots = IndexSlots::new(&records, occupied, slots).unwrap();
    let mut addresses = Vec::new();
    let mut cursor = StoredAddress::new(4096);
    measure
        .visit_blocks(|length| {
            addresses.push(cursor);
            cursor = cursor.checked_add(length)?;
            Ok(())
        })
        .unwrap();
    let mut calls = 0;
    let error = measure
        .write_at(&slots, &addresses, |_, _| {
            calls += 1;
            if calls == failure + 1 {
                Err(failure)
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    let EmissionError::Sink(actual) = error else {
        panic!("expected a callback error, got {error:?}");
    };
    assert_eq!(actual, failure);
    assert_eq!(calls, failure + 1);
}

#[rstest]
#[case::fixed(ChunkArrayKind::FixedArray, &[0, 1], 2, &[128, 140], FormatError::FixedArrayBlockOverlap { header: 128, data_block: 140 })]
#[case::extensible(ChunkArrayKind::ExtensibleArray, &[4], 5, &[128, 140, 1024], FormatError::ExtensibleArrayBlockOverlap { first: 0, second: 1 })]
#[case::fixed_count(ChunkArrayKind::FixedArray, &[0, 1], 2, &[128], FormatError::Internal("a fixed array requires two complete block placements".into()))]
#[case::extensible_count(ChunkArrayKind::ExtensibleArray, &[4], 5, &[128], FormatError::ExtensibleArrayBlockCount { expected: 3, actual: 1 })]
fn rejects_invalid_placements_before_emission(
    #[case] kind: ChunkArrayKind,
    #[case] occupied: &[u64],
    #[case] slots: u64,
    #[case] addresses: &[u64],
    #[case] expected: FormatError,
) {
    let measure = measure(kind, occupied, slots, false);
    let records = records(occupied, false);
    let slots = IndexSlots::new(&records, occupied, slots).unwrap();
    let addresses: Vec<_> = addresses.iter().copied().map(StoredAddress::new).collect();
    let mut calls = 0;
    let error = measure
        .write_at(&slots, &addresses, |_, _| {
            calls += 1;
            Ok::<(), Infallible>(())
        })
        .unwrap_err();
    assert_eq!(error.into_format(), expected);
    assert_eq!(calls, 0);
}

#[rstest]
#[case::fixed(ChunkArrayKind::FixedArray, 28)]
#[case::extensible(ChunkArrayKind::ExtensibleArray, 72)]
fn contiguous_encoding_rejects_address_overflow(#[case] kind: ChunkArrayKind, #[case] length: u64) {
    let measure = measure(kind, &[], 1, false);
    let error = measure
        .build_at(
            &IndexSlots::new(&[], &[], 1).unwrap(),
            StoredAddress::new(u64::MAX),
        )
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::OffsetOverflow {
            offset: u64::MAX,
            length
        }
    );
}

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn extensible_emission_retains_one_complete_block_buffer() {
    use test_util::allocation;

    let occupied: Vec<_> = (0..8192).collect();
    let records = records(&occupied, false);
    let slots = IndexSlots::new(&records, &occupied, 8192).unwrap();
    let measure = measure(ChunkArrayKind::ExtensibleArray, &occupied, 8192, false);
    let mut addresses = Vec::new();
    let mut cursor = StoredAddress::new(4096);
    let mut largest = 0;
    measure
        .visit_blocks(|length| {
            largest = largest.max(length);
            addresses.push(cursor);
            cursor = cursor.checked_add(length)?;
            Ok(())
        })
        .unwrap();
    let (control, positive) = allocation::measure("array_block_emission_control", || {
        std::hint::black_box(alloc::vec![0_u8; 1024])
    });
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.peak_bytes >= 1024, "{positive}");
    drop(control);
    let mut calls = 0;
    let mut bytes = 0;
    let ((), measured) = allocation::measure("array_block_emission", || {
        measure
            .write_at(&slots, &addresses, |_, block| {
                calls += 1;
                bytes += u64::try_from(block.len()).unwrap();
                Ok::<(), Infallible>(())
            })
            .unwrap();
    });
    assert_eq!(calls, addresses.len());
    assert_eq!(bytes, measure.length().unwrap());
    assert_eq!(measured.live_bytes, 0, "{measured}");
    assert!(measured.peak_bytes >= largest, "{measured}");
    assert!(
        measured.peak_bytes <= 2 * largest + u64::try_from(addresses.len()).unwrap() * 512 + 4096,
        "{measured}"
    );
    assert!(measured.peak_bytes < bytes, "{measured}");
}

fn measure(kind: ChunkArrayKind, occupied: &[u64], slots: u64, filtered: bool) -> ArrayMeasure {
    ArrayMeasure::new(ChunkArrayGeometry {
        widths: crate::file_writer::WIDTHS,
        kind,
        occupied: Cow::Borrowed(occupied),
        slots,
        chunk_bytes: 8,
        filtered,
    })
    .unwrap()
}

fn records(occupied: &[u64], filtered: bool) -> Vec<ChunkRecord> {
    occupied
        .iter()
        .map(|&slot| ChunkRecord {
            address: StoredAddress::new(0x10_0000 + slot * 8),
            stored_size: if filtered { 5 } else { 8 },
            filter_mask: u32::from(filtered),
        })
        .collect()
}
