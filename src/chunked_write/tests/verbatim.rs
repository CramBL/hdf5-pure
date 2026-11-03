#[cfg(feature = "std")]
use std::borrow::Cow;

use rstest::rstest;

use super::super::ChunkArrayKind;
use super::super::ChunkMeta;
use super::super::VerbatimChunks;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::ChunkedLayoutFlags;
use crate::data_layout::DataLayout;
use crate::dataspace::MaxExtent;
use crate::error::FormatError;

#[rstest]
#[case::single(&[1], None, &[0], 0, None)]
#[case::fixed(&[4], None, &[0, 1, 2, 3], 4, Some(ChunkArrayKind::FixedArray))]
#[case::fixed_sparse(&[2, 2], Some(&[MaxExtent::Fixed(2), MaxExtent::Fixed(8)][..]), &[0, 1, 8, 9], 16, Some(ChunkArrayKind::FixedArray))]
#[case::extensible(&[4], Some(&[MaxExtent::Unlimited][..]), &[0, 1, 2, 3], 4, Some(ChunkArrayKind::ExtensibleArray))]
#[case::rotated(&[2, 2], Some(&[MaxExtent::Fixed(8), MaxExtent::Unlimited][..]), &[0, 8, 1, 9], 10, Some(ChunkArrayKind::ExtensibleArray))]
fn prepared_verbatim_chunks_retain_records_and_shared_array_geometry(
    #[case] shape: &[u64],
    #[case] maxshape: Option<&[MaxExtent]>,
    #[case] occupied: &[u64],
    #[case] span: u64,
    #[case] array: Option<ChunkArrayKind>,
    #[values(false, true)] filtered: bool,
) {
    let dimensions = vec![1; shape.len()];
    let meta = vec![
        ChunkMeta {
            compressed_size: 8,
            filter_mask: u32::from(filtered)
        };
        occupied.len()
    ];
    let mut prepared = VerbatimChunks::new(
        crate::file_writer::WIDTHS,
        &meta,
        shape,
        &dimensions,
        convert::nz(8),
        ChunkedLayoutFlags::NONE,
        filtered.then_some(&[][..]),
        StoredAddress::ZERO,
        maxshape,
    )
    .unwrap();
    assert_eq!(
        prepared.payload_len(),
        u64::try_from(meta.len()).unwrap() * 8
    );
    assert_eq!(prepared.slot_of_chunk, occupied);
    assert_eq!(prepared.index_slots, span);
    let records = prepared.records.as_ptr();
    let capacity = prepared.records.capacity();
    prepared
        .assign_data_address(StoredAddress::new(1 << 40))
        .unwrap();
    assert_eq!(prepared.records.as_ptr(), records);
    assert_eq!(prepared.records.capacity(), capacity);
    for (position, record) in prepared.records.iter().enumerate() {
        assert_eq!(
            record.address.get(),
            (1 << 40) + u64::try_from(position).unwrap() * 8
        );
        assert_eq!(record.stored_size, 8);
        assert_eq!(record.filter_mask, u32::from(filtered));
    }
    #[cfg(feature = "std")]
    match (array, prepared.array_geometry()) {
        (None, None) => {}
        (Some(expected), Some(geometry)) => {
            match (expected, geometry.kind) {
                (ChunkArrayKind::FixedArray, ChunkArrayKind::FixedArray)
                | (ChunkArrayKind::ExtensibleArray, ChunkArrayKind::ExtensibleArray) => {}
                actual => panic!("unexpected array kinds: {actual:?}"),
            }
            let mut sorted = occupied.to_vec();
            sorted.sort_unstable();
            assert_eq!(geometry.occupied.as_ref(), sorted);
            assert_eq!(geometry.slots, span);
            assert_eq!(geometry.chunk_bytes, 8);
            assert_eq!(geometry.filtered, filtered);
            assert_eq!(
                matches!(geometry.occupied, Cow::Borrowed(_)),
                occupied.windows(2).all(|pair| pair[0] <= pair[1])
            );
        }
        _ => panic!("unexpected array geometry"),
    }
    let message = prepared.layout_message(StoredAddress::new(8_192)).unwrap();
    let DataLayout::Chunked {
        chunk_dimensions: dims,
        index,
        ..
    } = DataLayout::parse(&message, 8, 8).unwrap()
    else {
        panic!("expected a chunked layout");
    };
    let mut expected_dimensions = vec![1; shape.len()];
    expected_dimensions.push(8);
    assert_eq!(dims, expected_dimensions);
    match (array, index) {
        (None, ChunkIndexLayout::SingleChunk { address, .. }) => {
            assert_eq!(address.unwrap().get(), 1 << 40)
        }
        (Some(ChunkArrayKind::FixedArray), ChunkIndexLayout::FixedArray { address, .. })
        | (
            Some(ChunkArrayKind::ExtensibleArray),
            ChunkIndexLayout::ExtensibleArray { address, .. },
        ) => assert_eq!(address.unwrap().get(), 8_192),
        actual => panic!("unexpected chunk index: {actual:?}"),
    }
}

#[rstest]
#[case::last_valid(u64::MAX - 32, false)]
#[case::overflow(u64::MAX - 31, true)]
fn assigning_verbatim_addresses_preserves_records_on_overflow(
    #[case] address: u64,
    #[case] overflow: bool,
) {
    let meta = vec![
        ChunkMeta {
            compressed_size: 8,
            filter_mask: 0
        };
        4
    ];
    let mut prepared = VerbatimChunks::new(
        crate::file_writer::WIDTHS,
        &meta,
        &[4],
        &[1],
        convert::nz(8),
        ChunkedLayoutFlags::NONE,
        None,
        StoredAddress::new(512),
        None,
    )
    .unwrap();
    let before = prepared.records.clone();
    if overflow {
        let error = prepared
            .assign_data_address(StoredAddress::new(address))
            .unwrap_err();
        assert_eq!(
            error,
            FormatError::OffsetOverflow {
                offset: address,
                length: 32
            }
        );
        assert_eq!(prepared.records, before);
        assert_eq!(prepared.data_address.get(), 512);
    } else {
        prepared
            .assign_data_address(StoredAddress::new(address))
            .unwrap();
        assert_eq!(prepared.records.last().unwrap().address.get(), u64::MAX - 8);
        assert_eq!(prepared.data_address.get(), address);
    }
}

#[cfg(feature = "std")]
#[rstest]
#[case::above_four_gibibytes(1 << 40, false)]
#[case::last_valid(u64::MAX - 16, false)]
#[case::overflow(u64::MAX - 15, true)]
fn payload_descriptor_relocation_preserves_records_and_capacity(
    #[case] address: u64,
    #[case] overflow: bool,
) {
    let prepared = VerbatimChunks::new(
        crate::file_writer::WIDTHS,
        &[
            ChunkMeta {
                compressed_size: 7,
                filter_mask: 1,
            },
            ChunkMeta {
                compressed_size: 9,
                filter_mask: 2,
            },
        ],
        &[2, 1],
        &[1, 1],
        convert::nz(8),
        ChunkedLayoutFlags::NONE,
        Some(&[]),
        StoredAddress::new(512),
        Some(&[MaxExtent::Fixed(2), MaxExtent::Fixed(8)]),
    )
    .unwrap();
    let mut payload = prepared.into_payload();
    let before = payload.records.clone();
    let records = payload.records.as_ptr();
    let capacity = payload.records.capacity();
    let slots = payload.slot_of_chunk.clone();
    let result = payload.assign_data_address(StoredAddress::new(address));
    if overflow {
        assert_eq!(
            result.unwrap_err(),
            FormatError::OffsetOverflow {
                offset: address,
                length: 16
            }
        );
        assert_eq!(payload.records, before);
    } else {
        result.unwrap();
        let mut expected = before;
        expected[0].address = StoredAddress::new(address);
        expected[1].address = StoredAddress::new(address + 7);
        assert_eq!(payload.records, expected);
    }
    assert_eq!(payload.records.as_ptr(), records);
    assert_eq!(payload.records.capacity(), capacity);
    assert_eq!(payload.slot_of_chunk, slots);
    assert_eq!(payload.slot_of_chunk, [0, 8]);
    assert_eq!(payload.index_slots, 16);
    assert_eq!(payload.len(), 16);
}

#[rstest]
#[case::largest(u64::from(u32::MAX), false)]
#[case::too_wide(u64::from(u32::MAX) + 1, true)]
fn verbatim_preparation_checks_stored_dimension_widths(
    #[case] dimension: u64,
    #[case] too_wide: bool,
) {
    let meta = [ChunkMeta {
        compressed_size: dimension,
        filter_mask: 0,
    }];
    let result = VerbatimChunks::new(
        crate::file_writer::WIDTHS,
        &meta,
        &[1],
        &[dimension],
        convert::nz(1),
        ChunkedLayoutFlags::NONE,
        None,
        StoredAddress::ZERO,
        None,
    );
    if too_wide {
        let Err(error) = result else {
            panic!("expected a stored dimension width error");
        };
        assert_eq!(
            error,
            FormatError::ValueTooLargeForRepresentation {
                value: dimension,
                target: "u32"
            }
        );
    } else {
        let prepared = result.unwrap();
        assert_eq!(prepared.payload_len(), dimension);
        let message = prepared.layout_message(StoredAddress::new(512)).unwrap();
        let DataLayout::Chunked {
            chunk_dimensions, ..
        } = DataLayout::parse(&message, 8, 8).unwrap()
        else {
            panic!("expected a chunked layout");
        };
        assert_eq!(chunk_dimensions, [dimension, 1]);
    }
}

#[test]
fn verbatim_preparation_rejects_payload_length_overflow() {
    let meta = [
        ChunkMeta {
            compressed_size: u64::MAX,
            filter_mask: 0,
        },
        ChunkMeta {
            compressed_size: 1,
            filter_mask: 0,
        },
    ];
    let Err(error) = VerbatimChunks::new(
        crate::file_writer::WIDTHS,
        &meta,
        &[2],
        &[1],
        convert::nz(1),
        ChunkedLayoutFlags::NONE,
        None,
        StoredAddress::ZERO,
        None,
    ) else {
        panic!("expected payload length overflow");
    };
    assert_eq!(
        error,
        FormatError::OffsetOverflow {
            offset: u64::MAX,
            length: 1
        }
    );
}
