#![cfg(feature = "__hdf5-1.10")]
//! Cross-validation for Extensible-Array-indexed chunked datasets (one unlimited
//! dimension), in both directions against the reference C HDF5 library.
//!
//! These guard the EA block-size geometry, super blocks, and paged data blocks.
//! Sizes are chosen to span every structural range:
//!   - 20      inline (4) + first direct data block (16)
//!   - 300     all 6 direct data blocks + the first super block
//!   - 2000    several super blocks
//!   - 50000   deeper super-block nesting
//!   - 140000  past 131060 chunks: paged data blocks
//!
//! Before the geometry fix, anything past 20 chunks silently corrupted on read.

use std::path::Path;

use hdf5::Extent;
use hdf5::datatype::ByteOrder as CByteOrder;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::MessageType as FormatMessageType;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use hdf5_pure_format::__private::ExtensibleArrayInitialization;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::LengthWidth;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::OffsetWidth;
use rstest::rstest;
use test_util::bytes;
use test_util::checksum;
use test_util::object_header::MessageType;
use test_util::object_header::v2;

use test_util::extensible_array;
use test_util_hdf5::dataset::Unlimited;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::session;

#[rstest]
#[case::canonical([32, 4, 4, 16, 10])]
#[case::nondefault([24, 8, 2, 8, 8])]
fn pure_initialization_supports_native_first_writes(
    #[case] parameters: [u8; 5],
    #[values(false, true)] filtered: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("initialized.h5");
    {
        let file = file::libhdf5_create_v110(&path);
        let builder = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype())
            .chunk([1])
            .shape([Extent::resizable(0)]);
        let builder = if filtered {
            builder.shuffle().deflate(6)
        } else {
            builder
        };
        let dataset = builder.create("d").unwrap();
        assert_eq!(dataset.num_chunks().unwrap(), 0);
        drop(dataset);
        file.close().unwrap();
    }
    let mut image = std::fs::read(&path).unwrap();
    assert!(bytes::signature_offsets(&image, EA_HEADER_SIGNATURE).is_empty());
    let reader = File::from_bytes(image.clone()).unwrap();
    let mut superblock = reader.superblock().clone();
    drop(reader);
    let address = StoredAddress::new(u64::try_from(image.len()).unwrap());
    let creation = ExtensibleArrayCreationParameters::parse(&parameters).unwrap();
    let initialization = ExtensibleArrayInitialization::new(
        creation,
        4,
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        filtered,
    )
    .unwrap();
    let initialized = initialization.encode_at(address).unwrap();
    assert_eq!(initialized.len() as u64, initialization.allocation_len());
    image.extend_from_slice(&initialized);
    patch_layout(&mut image, |layout, tail| {
        let creation_at = tail + 1;
        let address_at = creation_at + parameters.len();
        assert_eq!(&layout[address_at..address_at + 8], &[0xff; 8]);
        layout[creation_at..address_at].copy_from_slice(&parameters);
        layout[address_at..address_at + 8].copy_from_slice(&address.get().to_le_bytes());
    });
    superblock.eof_address = u64::try_from(image.len()).unwrap();
    let encoded = hdf5_pure_format::__private::serialize_superblock(&superblock).unwrap();
    image[..encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, &image).unwrap();
    {
        let reader = File::open(&path).unwrap();
        let dataset = reader.dataset("d").unwrap();
        assert_eq!(dataset.shape().unwrap(), [0]);
        assert_eq!(dataset.read_i32().unwrap(), Vec::<i32>::new());
    }
    assert_eq!(std::fs::read(&path).unwrap(), image);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.shape(), [0]);
        assert_eq!(dataset.num_chunks().unwrap(), 0);
        dataset.resize([3]).unwrap();
        dataset.write_raw(&[3i32, 4, 5]).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), [3, 4, 5]);
        assert_eq!(dataset.num_chunks().unwrap(), 3);
        drop(dataset);
        file.close().unwrap();
    }
    let reader = File::open(&path).unwrap();
    let dataset = reader.dataset("d").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [3, 4, 5]);
    assert_eq!(dataset.chunks().unwrap().len(), 3);
    let image = std::fs::read(&path).unwrap();
    let header =
        ExtensibleArrayHeader::parse(&image, usize::try_from(address.get()).unwrap(), 8, 8)
            .unwrap();
    assert_eq!(header.max_index_bits().get(), parameters[0]);
    assert_eq!(header.idx_blk_elmts, parameters[1]);
    assert_eq!(header.super_blk_min_data_ptrs, parameters[2]);
    assert_eq!(header.min_dblk_nelmts, parameters[3]);
    assert_eq!(header.page_nelmts().get(), 1u64 << parameters[4]);
}

const SIZES: &[usize] = &[20, 300, 2000, 50000, 140000];

/// Create a 1-D unlimited, chunked i32 dataset with libhdf5,
/// using the latest format so the chunk index is an Extensible Array.
fn write_with_c(path: &std::path::Path, n: usize) {
    let file = file::libhdf5_create_v110(path);
    let ds = file
        .new_dataset::<i32>()
        .chunk((1,))
        .shape((Extent::resizable(n),))
        .create("d")
        .unwrap();
    let data: Vec<i32> = (0..n as i32).collect();
    ds.write(&data).unwrap();
    file.close().unwrap();
}

/// Create the same dataset with hdf5-pure.
fn write_with_pure(path: &std::path::Path, n: usize) {
    let data: Vec<i32> = (0..n as i32).collect();
    Unlimited::new("d", &data, 1).pure_create(path);
}

/// Direction 1: hdf5-pure writes, libhdf5 reads.
#[test]
fn pure_writes_c_reads() {
    for &n in SIZES {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ea.h5");
        write_with_pure(&path, n);

        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        let values = ds.read_raw::<i32>().unwrap();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(values.len(), n, "C read wrong length for n={n}");
        assert_eq!(values, expected, "C read wrong data for n={n}");
    }
}

/// Direction 2: libhdf5 writes, hdf5-pure reads.
#[test]
fn c_writes_pure_reads() {
    for &n in SIZES {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ea.h5");
        write_with_c(&path, n);

        let bytes = std::fs::read(&path).unwrap();
        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("d").unwrap();
        let values = ds.read_i32().unwrap();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(values.len(), n, "hdf5-pure read wrong length for n={n}");
        assert_eq!(values, expected, "hdf5-pure read wrong data for n={n}");
    }
}

#[test]
fn eahd_stats_match_c() {
    for &n in SIZES {
        let dir = tempfile::tempdir().unwrap();
        let c_path = dir.path().join("c.h5");
        let pure_path = dir.path().join("pure.h5");
        write_with_c(&c_path, n);
        write_with_pure(&pure_path, n);
        assert_eq!(
            extensible_array::header_stats(&pure_path),
            extensible_array::header_stats(&c_path),
            "EAHD stats differ from the C library at n={n}"
        );
    }
}

#[test]
fn eahd_stats_match_c_after_an_in_place_append() {
    for &(start, grown) in &[(4usize, 20usize), (16, 300), (300, 700)] {
        let dir = tempfile::tempdir().unwrap();
        let c_path = dir.path().join("c.h5");
        let pure_path = dir.path().join("pure.h5");
        write_with_c(&c_path, grown);
        write_with_pure(&pure_path, start);

        {
            let file = File::open_rw(&pure_path).unwrap();
            let mut ds = file.dataset("d").unwrap();
            let more: Vec<i32> = (start as i32..grown as i32).collect();
            ds.append(&more).unwrap();
            file.close().unwrap();
        }

        assert_eq!(
            extensible_array::header_stats(&pure_path),
            extensible_array::header_stats(&c_path),
            "EAHD stats after growing {start} -> {grown} in place differ from the C library's \
             for a dataset written whole at {grown}"
        );
        // The append itself must still be right, or the stats above would be
        // describing a file nobody can read.
        let file = hdf5::File::open(&pure_path).unwrap();
        let values = file.dataset("d").unwrap().read_raw::<i32>().unwrap();
        assert_eq!(values, (0..grown as i32).collect::<Vec<_>>());
    }
}

#[rstest]
#[case(Boundary::Inline)]
#[case(Boundary::Direct)]
#[case(Boundary::Secondary)]
#[case(Boundary::Paged)]
#[case(Boundary::Page)]
fn nondefault_geometry_round_trips_through_rust_append_and_c_mutation(
    #[case] boundary: Boundary,
    #[values(2, 4, 8, 16)] minimum_pointers: u8,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] bounded: bool,
    #[values(32, 64)] maximum_bits: u8,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append.h5");
    let oracle = dir.path().join("oracle.h5");
    let paged = matches!(boundary, Boundary::Page | Boundary::Paged);
    let min_elements = if paged { 1 } else { 16 };
    let page_bits = if paged { 5 } else { 10 };
    let start = match boundary {
        Boundary::Direct => 19,
        Boundary::Inline => 3,
        Boundary::Page => 2_082,
        Boundary::Paged => 2_050,
        Boundary::Secondary => 4 + 16 * (usize::from(minimum_pointers).pow(2) - 1) - 1,
    };
    let grown = start + 3;
    let final_len = grown + 1;
    let values: Vec<i32> = (0..final_len)
        .map(|index| i32::try_from(index).unwrap() % 19)
        .collect();
    for (target, len) in [(&path, start), (&oracle, grown)] {
        {
            let file = file::libhdf5_create_v110(target);
            let builder = file
                .new_dataset_builder()
                .empty_as(ByteOrder::LittleEndian.i32_datatype())
                .chunk([1])
                .shape([Extent::resizable(len)]);
            let builder = if filtered {
                builder.shuffle().deflate(6)
            } else {
                builder
            };
            let dataset = builder.create("d").unwrap();
            assert_eq!(
                dataset.dtype().unwrap().byte_order(),
                CByteOrder::LittleEndian
            );
        }
        patch_unallocated_layout(
            target,
            minimum_pointers,
            min_elements,
            page_bits,
            maximum_bits,
        );
        {
            let file = hdf5::File::open_rw(target).unwrap();
            let dataset = file.dataset("d").unwrap();
            dataset.write_raw(&values[..len]).unwrap();
            assert_eq!(dataset.read_raw::<i32>().unwrap(), values[..len]);
        }
    }
    {
        let file = if bounded {
            session::open_bounded(&path).unwrap()
        } else {
            File::open_rw(&path).unwrap()
        };
        let mut dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.read_i32().unwrap(), values[..start]);
        dataset.append(&values[start..grown]).unwrap();
        assert_eq!(dataset.read_i32().unwrap(), values[..grown]);
    }
    assert_eq!(
        extensible_array::header_stats(&path),
        extensible_array::header_stats(&oracle)
    );
    let chunks = {
        let file = hdf5::File::open_rw(&path).unwrap();
        let dataset = file.dataset("d").unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), values[..grown]);
        dataset.resize([final_len]).unwrap();
        dataset
            .write_slice(&values[grown..], grown..final_len)
            .unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), values);
        assert_eq!(dataset.num_chunks().unwrap(), final_len);
        [0, start - 1, start, grown - 1, grown].map(|slot| {
            let chunk = dataset.chunk_info(slot).unwrap();
            (slot, chunk.addr, chunk.size, chunk.filter_mask)
        })
    };
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), values);
    let indexed = dataset.chunks().unwrap();
    assert_eq!(indexed.len(), final_len);
    for (slot, address, size, mask) in chunks {
        assert_eq!(indexed[slot].offset, [u64::try_from(slot).unwrap()]);
        assert_eq!(
            (
                indexed[slot].address,
                indexed[slot].storage_size,
                indexed[slot].filter_mask().raw_bits()
            ),
            (address, size, mask)
        );
    }
}

#[rstest]
#[case(
    HEADER_MINIMUM_ELEMENTS,
    0,
    FormatError::InvalidExtensibleArrayGeometry
)]
#[case(
    HEADER_MINIMUM_POINTERS,
    1,
    FormatError::InvalidExtensibleArrayGeometry
)]
#[case(
    HEADER_MINIMUM_POINTERS,
    3,
    FormatError::InvalidExtensibleArrayGeometry
)]
#[case(
    HEADER_SIGNATURE_BYTE,
    b'X',
    FormatError::InvalidExtensibleArraySignature
)]
#[case(HEADER_VERSION, 1, FormatError::InvalidExtensibleArrayVersion(1))]
#[case(HEADER_CLIENT, 2, FormatError::InvalidExtensibleArrayClient(2))]
fn invalid_headers_are_rejected_before_append_writes(
    #[case] field: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.h5");
    {
        let file = file::libhdf5_create_v110(&path);
        let dataset = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype())
            .chunk([1])
            .shape([Extent::resizable(20)])
            .create("d")
            .unwrap();
        dataset.write_raw(&(0..20).collect::<Vec<i32>>()).unwrap();
    }
    let mut image = std::fs::read(&path).unwrap();
    let header = bytes::sole_signature(&image, EA_HEADER_SIGNATURE);
    image[header + field] = value;
    checksum::restamp(
        &mut image,
        header,
        ExtensibleArrayHeader::serialized_size(8, 8),
    );
    std::fs::write(&path, &image).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let error = dataset.append(&[20]).unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(error, expected);
    }
    assert_eq!(std::fs::read(&path).unwrap(), image);
}

fn patch_unallocated_layout(
    path: &Path,
    minimum_pointers: u8,
    min_elements: u8,
    page_bits: u8,
    maximum_bits: u8,
) {
    let mut image = std::fs::read(path).unwrap();
    patch_layout(&mut image, |layout, tail| {
        layout[tail + MAXIMUM_INDEX_BITS] = maximum_bits;
        layout[tail + SECONDARY_MINIMUM_POINTERS] = minimum_pointers;
        layout[tail + DATA_BLOCK_MINIMUM_ELEMENTS] = min_elements;
        layout[tail + PAGE_BITS] = page_bits;
    });
    std::fs::write(path, image).unwrap();
}

#[rstest]
#[case::inline(3, 4, None, [0, 0, 0, 0, 0])]
#[case::skipped_direct(51, 52, None, [0, 0, 1, 22, 32])]
#[case::secondary(244, 245, None, [1, 54, 1, 22, 64])]
#[case::skipped_secondary(1_012, 1_013, None, [1, 86, 1, 22, 128])]
#[case::paged(131_060, 131_061, None, [1, 598, 1, 30, 2_048])]
#[case::existing_block(245, 246, Some(244), [1, 54, 1, 22, 64])]
#[case::new_page(132_084, 132_085, Some(131_060), [1, 598, 1, 30, 2_048])]
#[case::multiple_blocks(244, 504, None, [2, 108, 5, 110, 384])]
fn sparse_append_statistics_match_allocated_blocks(
    #[case] start: usize,
    #[case] end: usize,
    #[case] existing: Option<usize>,
    #[case] counts: [u64; 5],
    #[values(false, true)] bounded: bool,
    #[values(false, true)] filtered: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse_append.h5");
    let oracle = dir.path().join("oracle.h5");
    {
        let file = file::libhdf5_create_v110(&path);
        let builder = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype())
            .chunk([1])
            .shape([Extent::resizable(start)]);
        let builder = if filtered {
            builder.shuffle().deflate(6)
        } else {
            builder
        };
        let dataset = builder.create("d").unwrap();
        dataset.write_slice(&[7i32], 0..1).unwrap();
        if let Some(slot) = existing {
            dataset.write_slice(&[8i32], slot..slot + 1).unwrap();
        }
        assert_eq!(
            dataset.num_chunks().unwrap(),
            1 + usize::from(existing.is_some())
        );
    }
    std::fs::copy(&path, &oracle).unwrap();
    let image = std::fs::read(&path).unwrap();
    let header = bytes::sole_signature(&image, EA_HEADER_SIGNATURE);
    let stride = u64::from(image[header + HEADER_ELEMENT_SIZE]);
    let appended = vec![9i32; end - start];
    {
        let file = hdf5::File::open_rw(&oracle).unwrap();
        let dataset = file.dataset("d").unwrap();
        dataset.resize([end]).unwrap();
        dataset.write_slice(&appended, start..end).unwrap();
    }
    {
        let file = if bounded {
            session::open_bounded(&path).unwrap()
        } else {
            File::open_rw(&path).unwrap()
        };
        file.dataset("d").unwrap().append(&appended).unwrap();
    }
    let [
        secondary_block_count,
        secondary_block_size,
        data_block_count,
        overhead,
        capacity,
    ] = counts;
    let expected = extensible_array::HeaderStats {
        secondary_block_count,
        secondary_block_size,
        data_block_count,
        data_block_size: capacity * stride + overhead,
        max_index_set: u64::try_from(end).unwrap(),
        element_count: 4 + capacity,
    };
    assert_eq!(extensible_array::header_stats(&oracle), expected);
    assert_eq!(extensible_array::header_stats(&path), expected);
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let mut expected_values = vec![0i32; end];
    expected_values[0] = 7;
    if let Some(slot) = existing {
        expected_values[slot] = 8;
    }
    expected_values[start..end].fill(9);
    assert_eq!(dataset.read_raw::<i32>().unwrap(), expected_values);
    assert_eq!(dataset.shape(), [end]);
    assert_eq!(dataset.read_slice_1d::<i32, _>(0..1).unwrap().to_vec(), [7]);
    assert_eq!(
        dataset
            .read_slice_1d::<i32, _>(start..end)
            .unwrap()
            .to_vec(),
        appended
    );
    if let Some(slot) = existing {
        assert_eq!(
            dataset
                .read_slice_1d::<i32, _>(slot..slot + 1)
                .unwrap()
                .to_vec(),
            [8]
        );
    }
    assert_eq!(
        dataset.num_chunks().unwrap(),
        1 + usize::from(existing.is_some()) + end - start
    );
}

#[rstest]
fn sparse_partial_tail_statistics_match_c_across_append_sessions(
    #[values(false, true)] bounded: bool,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] reopen: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial.h5");
    let oracle = dir.path().join("oracle.h5");
    let chunk = 4_096;
    let start = 244 * chunk + 1;
    let appended = vec![9i32; 300_000];
    {
        let file = file::libhdf5_create_v110(&path);
        let builder = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype())
            .chunk([chunk])
            .shape([Extent::resizable(start)]);
        let builder = if filtered {
            builder.shuffle().deflate(6)
        } else {
            builder
        };
        let dataset = builder.create("d").unwrap();
        dataset.write_slice(&[7i32], 0..1).unwrap();
        dataset.write_slice(&[8i32], start - 1..start).unwrap();
    }
    std::fs::copy(&path, &oracle).unwrap();
    let end = start + appended.len() + usize::from(!reopen);
    {
        let file = hdf5::File::open_rw(&oracle).unwrap();
        let dataset = file.dataset("d").unwrap();
        dataset.resize([end]).unwrap();
        dataset
            .write_slice(&appended, start..start + appended.len())
            .unwrap();
        if !reopen {
            dataset.write_slice(&[10i32], end - 1..end).unwrap();
        }
    }
    let open = || {
        if bounded {
            session::open_bounded(&path).unwrap()
        } else {
            File::open_rw(&path).unwrap()
        }
    };
    if reopen {
        for part in appended.chunks(100_000) {
            let file = open();
            file.dataset("d").unwrap().append(part).unwrap();
        }
    } else {
        let file = open();
        file.dataset("d").unwrap().append(&appended).unwrap();
        file.dataset("d").unwrap().append(&[10i32]).unwrap();
    }
    assert_eq!(
        extensible_array::header_stats(&path),
        extensible_array::header_stats(&oracle)
    );
    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let mut expected = vec![8];
    expected.extend_from_slice(&appended);
    if !reopen {
        expected.push(10);
    }
    assert_eq!(
        dataset
            .read_slice_1d::<i32, _>(start - 1..end)
            .unwrap()
            .to_vec(),
        expected
    );
    assert_eq!(dataset.num_chunks().unwrap(), 1 + end.div_ceil(chunk) - 244);
}

#[rstest]
#[cfg(target_pointer_width = "64")]
fn sparse_geometry_overflow_is_rejected_before_append_writes(
    #[values(false, true)] bounded: bool,
    #[values(false, true)] filtered: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse.h5");
    {
        let file = file::libhdf5_create_v110(&path);
        let dataset = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype())
            .chunk([1])
            .shape([Extent::resizable(1)]);
        let dataset = if filtered {
            dataset.shuffle().deflate(6)
        } else {
            dataset
        };
        dataset.create("d").unwrap();
    }
    let mut image = std::fs::read(&path).unwrap();
    patch_layout(&mut image, |layout, tail| {
        layout[tail + MAXIMUM_INDEX_BITS] = 64;
    });
    std::fs::write(&path, image).unwrap();
    let extent = usize::try_from(u64::MAX - 11).unwrap();
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let dataset = file.dataset("d").unwrap();
        dataset.write_raw(&[7i32]).unwrap();
        dataset.resize([extent]).unwrap();
        assert_eq!(dataset.shape(), [extent]);
        assert_eq!(dataset.read_slice_1d::<i32, _>(0..1).unwrap().to_vec(), [7]);
    }
    let image = std::fs::read(&path).unwrap();
    assert!(image.len() < 16_384);
    {
        let file = if bounded {
            session::open_bounded(&path).unwrap()
        } else {
            File::open_rw(&path).unwrap()
        };
        let mut dataset = file.dataset("d").unwrap();
        let error = dataset.append(&[9i32]).unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(error, FormatError::InvalidExtensibleArrayGeometry);
        drop(dataset);
        file.close().unwrap();
    }
    assert_eq!(std::fs::read(&path).unwrap(), image);
}

#[rstest]
#[case::checksum(false)]
#[case::truncated(true)]
fn corrupt_or_truncated_headers_are_rejected_before_append_writes(#[case] truncated: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("damaged.h5");
    {
        let file = file::libhdf5_create_v110(&path);
        let dataset = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype())
            .chunk([1])
            .shape([Extent::resizable(20)])
            .create("d")
            .unwrap();
        dataset.write_raw(&(0..20).collect::<Vec<i32>>()).unwrap();
    }
    let mut image = std::fs::read(&path).unwrap();
    let header = bytes::sole_signature(&image, EA_HEADER_SIGNATURE);
    let len = ExtensibleArrayHeader::serialized_size(8, 8);
    let expected = if truncated {
        let moved = u64::try_from(image.len()).unwrap();
        let prefix = image[header..header + len - 1].to_vec();
        image.extend_from_slice(&prefix);
        patch_layout(&mut image, |layout, tail| {
            let address = tail + DEFAULT_EA_LAYOUT_TAIL.len();
            let encoded = moved.to_le_bytes();
            layout[address..address + encoded.len()].copy_from_slice(&encoded);
        });
        FormatError::UnexpectedSourceEof {
            expected: moved + u64::try_from(len).unwrap(),
            available: u64::try_from(image.len()).unwrap(),
        }
    } else {
        let computed = checksum::lookup3(&image[header..header + len - CHECKSUM_BYTES]);
        let expected = computed ^ 1;
        image[header + len - CHECKSUM_BYTES..header + len].copy_from_slice(&expected.to_le_bytes());
        FormatError::ChecksumMismatch { expected, computed }
    };
    std::fs::write(&path, &image).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let error = dataset.append(&[20]).unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(error, expected);
    }
    assert_eq!(std::fs::read(&path).unwrap(), image);
}

fn patch_layout(image: &mut [u8], patch: impl FnOnce(&mut [u8], usize)) {
    let candidates: Vec<_> = image
        .windows(DEFAULT_EA_LAYOUT_TAIL.len())
        .enumerate()
        .filter_map(|(at, value)| (value == DEFAULT_EA_LAYOUT_TAIL).then_some(at))
        .collect();
    assert_eq!(candidates.len(), 1);
    let tail = candidates[0];
    let header = bytes::signature_offsets(image, OBJECT_HEADER_SIGNATURE)
        .into_iter()
        .filter(|&at| at < tail)
        .max()
        .unwrap();
    let parsed = ObjectHeader::parse(image, AccessMode::ReadOnly, header, 8, 8).unwrap();
    let mut layout = parsed
        .messages
        .into_iter()
        .find(|message| message.msg_type == FormatMessageType::DATA_LAYOUT)
        .unwrap()
        .data;
    let tail = layout
        .windows(DEFAULT_EA_LAYOUT_TAIL.len())
        .position(|value| value == DEFAULT_EA_LAYOUT_TAIL)
        .unwrap();
    patch(&mut layout, tail);
    v2::replace_message(image, header, MessageType::DATA_LAYOUT, &layout);
}

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Direct,
    Inline,
    Page,
    Paged,
    Secondary,
}

// `H5O__layout_encode` (`H5Olayout.c`, HDF5 2.2.0) writes the index type and the
// creation parameters in this order. Their defaults are in `H5Dpkg.h`.
const DEFAULT_EA_LAYOUT_TAIL: [u8; 6] = [4, 32, 4, 4, 16, 10];
const MAXIMUM_INDEX_BITS: usize = 1;
const SECONDARY_MINIMUM_POINTERS: usize = 3;
const DATA_BLOCK_MINIMUM_ELEMENTS: usize = 4;
const PAGE_BITS: usize = 5;
// "Version 2 Data Object Header Prefix" in format specification, version 4.0,
// defines the version 2 object header signature.
const OBJECT_HEADER_SIGNATURE: &[u8; 4] = b"OHDR";

// "The Extensible Array Index" in format specification, version 4.0, places the version 0
// header element size after the client ID.
const HEADER_ELEMENT_SIZE: usize = 6;

// "The Extensible Array Index" in format specification, version 4.0, defines the version 0
// header signature and the positions of the minimum-element and minimum-pointer fields.
const EA_HEADER_SIGNATURE: &[u8; 4] = b"EAHD";
const HEADER_MINIMUM_ELEMENTS: usize = 9;
const HEADER_MINIMUM_POINTERS: usize = 10;

// "The Extensible Array Index" in format specification, version 4.0, places the version 0
// header signature, version, and client fields before the creation parameters.
const HEADER_SIGNATURE_BYTE: usize = 0;
const HEADER_VERSION: usize = 4;
const HEADER_CLIENT: usize = 5;

// "The Extensible Array Index" in format specification, version 4.0, ends the version 0
// header with a four-byte checksum.
const CHECKSUM_BYTES: usize = 4;
