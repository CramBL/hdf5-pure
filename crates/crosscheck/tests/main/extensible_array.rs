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
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use hdf5_pure_format::__private::ObjectHeader;
use rstest::rstest;
use test_util::bytes;
use test_util::checksum;
use test_util::object_header::MessageType;
use test_util::object_header::v2;

use test_util::extensible_array;
use test_util_hdf5::dataset::Unlimited;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

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

/// The EA header's six statistics fields must match the C library byte-for-byte
/// (the C library recomputes and would reject inconsistent computed stats; the
/// stored max_idx_set / nelmts must also agree). Compares the headers of the two
/// files for identical datasets.
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

/// The same six statistics, published by the *in-place* append path rather than
/// by the whole-file writer.
///
/// `Located::update_ea_header` writes them on every append, and until this test
/// nothing read them back: zeroing `nelmts` or transposing two fields passed the
/// entire suite, C-library crosschecks included, because the reference library
/// recomputes the geometry from the block sizes rather than trusting these
/// fields on read. They are still what `h5stat` and `H5EA__hdr_debug` report and
/// what a future reader may believe, so they are checked against the one oracle
/// that cannot be wrong about them: the same dataset written whole, by the C
/// library, at the size the append grew it to.
///
/// Sizes are small on purpose — this grows the file one append at a time — but
/// span the inline elements, the first direct data blocks, and the first super
/// block, which is where the six values stop being constants.
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
        patch_unallocated_layout(target, minimum_pointers, min_elements, page_bits);
        {
            let file = hdf5::File::open_rw(target).unwrap();
            let dataset = file.dataset("d").unwrap();
            dataset.write_raw(&values[..len]).unwrap();
            assert_eq!(dataset.read_raw::<i32>().unwrap(), values[..len]);
        }
    }
    {
        let file = File::open_rw(&path).unwrap();
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
                indexed[slot].filter_mask
            ),
            (address, size, mask)
        );
    }
}

#[rstest]
#[case(HEADER_MINIMUM_ELEMENTS, 0)]
#[case(HEADER_MINIMUM_POINTERS, 1)]
#[case(HEADER_MINIMUM_POINTERS, 3)]
fn invalid_geometry_is_rejected_before_append_writes(#[case] field: usize, #[case] value: u8) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.h5");
    {
        let file = file::libhdf5_create_v110(&path);
        let dataset = file
            .new_dataset::<i32>()
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
        assert_eq!(error, FormatError::InvalidExtensibleArrayGeometry);
    }
    assert_eq!(std::fs::read(&path).unwrap(), image);
}

fn patch_unallocated_layout(path: &Path, minimum_pointers: u8, min_elements: u8, page_bits: u8) {
    let mut image = std::fs::read(path).unwrap();
    let candidates: Vec<_> = image
        .windows(DEFAULT_EA_LAYOUT_TAIL.len())
        .enumerate()
        .filter_map(|(at, value)| (value == DEFAULT_EA_LAYOUT_TAIL).then_some(at))
        .collect();
    assert_eq!(candidates.len(), 1);
    let tail = candidates[0];
    let header = bytes::signature_offsets(&image, OBJECT_HEADER_SIGNATURE)
        .into_iter()
        .filter(|&at| at < tail)
        .max()
        .unwrap();
    let parsed = ObjectHeader::parse(&image, AccessMode::ReadOnly, header, 8, 8).unwrap();
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
    layout[tail + SECONDARY_MINIMUM_POINTERS] = minimum_pointers;
    layout[tail + DATA_BLOCK_MINIMUM_ELEMENTS] = min_elements;
    layout[tail + PAGE_BITS] = page_bits;
    v2::replace_message(&mut image, header, MessageType::DATA_LAYOUT, &layout);
    std::fs::write(path, image).unwrap();
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
const SECONDARY_MINIMUM_POINTERS: usize = 3;
const DATA_BLOCK_MINIMUM_ELEMENTS: usize = 4;
const PAGE_BITS: usize = 5;
// "Version 2 Data Object Header Prefix" in format specification, version 4.0,
// defines the version 2 object header signature.
const OBJECT_HEADER_SIGNATURE: &[u8; 4] = b"OHDR";

// "The Extensible Array Index" in format specification, version 4.0, defines the version 0
// header signature and the positions of the minimum-element and minimum-pointer fields.
const EA_HEADER_SIGNATURE: &[u8; 4] = b"EAHD";
const HEADER_MINIMUM_ELEMENTS: usize = 9;
const HEADER_MINIMUM_POINTERS: usize = 10;
