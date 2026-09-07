// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for the layout / filter introspection API
//! (issue #149). The reference library *writes* datasets in every storage class
//! and chunk-index kind; this crate must classify them identically and — the
//! make-or-break check — the file addresses it reports for each chunk must be
//! absolute and correct, verified by seeking to them in the raw file bytes and
//! decoding.
//!
//! Covers the classes the pure writer cannot itself emit: `Compact` (the pure
//! writer always uses contiguous) and the legacy `BTreeV1` index (the pure
//! writer only emits the v4 indices), plus a v2-B-tree (`BTreeV2`) dataset, whose
//! chunks are classified but not yet enumerable.

use hdf5::Extent;
use hdf5::dataset::Layout as CLayout;
use hdf5::file::LibraryVersion;
use hdf5_pure::{Chunk, ChunkIndex, File, Layout};
use tempfile::tempdir;

/// A reference-library file whose datasets are created under the given libver
/// bounds. `V18/V18` forces the classic (v1 B-tree) chunked format; `V110/latest`
/// enables the v4 indices (single-chunk, fixed array, extensible array, v2 B-tree).
fn c_file(path: &std::path::Path, low: LibraryVersion, high: LibraryVersion) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, high))
        .create(path)
        .unwrap()
}

/// Decode the raw (unfiltered) i32 bytes a chunk occupies, read straight from the
/// file at the address the pure reader reported. Proves the address is an
/// absolute file offset and `storage_size` is exact.
fn chunk_i32_at(path: &std::path::Path, chunk: &Chunk) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap();
    let start = chunk.address as usize;
    let end = start + chunk.storage_size as usize;
    bytes[start..end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

fn read_c_i32(path: &std::path::Path, name: &str) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    let v = f.dataset(name).unwrap().read_raw::<i32>().unwrap();
    f.close().unwrap();
    v
}

// ---- storage classes --------------------------------------------------------

#[test]
fn c_contiguous_classifies() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("contig.h5");
    {
        let file = c_file(&path, LibraryVersion::V110, LibraryVersion::latest());
        let ds = file.new_dataset::<i32>().shape((6,)).create("d").unwrap();
        ds.write(&[0i32, 1, 2, 3, 4, 5]).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    match ds.layout().unwrap() {
        Layout::Contiguous { address, size } => {
            assert_eq!(size, 6 * 4);
            assert!(matches!(address, Some(a) if a != 0));
        }
        other => panic!("expected contiguous, got {other:?}"),
    }
    assert_eq!(ds.chunk_index().unwrap(), None);
    assert!(ds.chunks().is_err());
}

#[test]
fn c_compact_classifies() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("compact.h5");
    {
        let file = c_file(&path, LibraryVersion::V110, LibraryVersion::latest());
        let ds = file
            .new_dataset::<i32>()
            .layout(CLayout::Compact)
            .shape((4,))
            .create("d")
            .unwrap();
        ds.write(&[10i32, 20, 30, 40]).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    assert_eq!(ds.layout().unwrap(), Layout::Compact { size: 4 * 4 });
    assert_eq!(ds.chunk_index().unwrap(), None);
    assert!(ds.chunks().is_err());
    // The data still round-trips.
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30, 40]);
}

// ---- v4 chunk indices -------------------------------------------------------

#[test]
fn c_single_chunk_classifies() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("single.h5");
    {
        let file = c_file(&path, LibraryVersion::V110, LibraryVersion::latest());
        let ds = file
            .new_dataset::<i32>()
            .chunk((8,))
            .shape((8,)) // one chunk covers the fixed dataset
            .create("d")
            .unwrap();
        ds.write(&(0..8).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    assert_eq!(ds.chunk_index().unwrap(), Some(ChunkIndex::SingleChunk));
    let chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunk_i32_at(&path, &chunks[0]), (0..8).collect::<Vec<_>>());
}

#[test]
fn c_fixed_array_addresses_are_absolute() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fixed.h5");
    {
        let file = c_file(&path, LibraryVersion::V110, LibraryVersion::latest());
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((8,)) // fixed shape, 2 chunks -> fixed array
            .create("d")
            .unwrap();
        ds.write(&(0..8).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    assert_eq!(ds.chunk_index().unwrap(), Some(ChunkIndex::FixedArray));

    let mut chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    chunks.sort_by_key(|c| c.offset.clone());
    assert_eq!(chunks[0].offset, vec![0]);
    assert_eq!(chunks[1].offset, vec![4]);
    // Seek to each reported address in the raw file and decode: proves the
    // addresses are absolute file offsets and storage_size is exact.
    assert_eq!(chunk_i32_at(&path, &chunks[0]), vec![0, 1, 2, 3]);
    assert_eq!(chunk_i32_at(&path, &chunks[1]), vec![4, 5, 6, 7]);
}

#[test]
fn c_extensible_array_supports_append() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ea.h5");
    {
        let file = c_file(&path, LibraryVersion::V110, LibraryVersion::latest());
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(8),)) // one unlimited dim -> extensible array
            .create("d")
            .unwrap();
        ds.write(&(0..8).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    let index = ds.chunk_index().unwrap().unwrap();
    assert_eq!(index, ChunkIndex::ExtensibleArray);
    assert!(index.supports_inplace_append());

    let mut chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    chunks.sort_by_key(|c| c.offset.clone());
    assert_eq!(chunk_i32_at(&path, &chunks[0]), vec![0, 1, 2, 3]);
    assert_eq!(chunk_i32_at(&path, &chunks[1]), vec![4, 5, 6, 7]);
}

// ---- legacy v1 B-tree -------------------------------------------------------

#[test]
fn c_legacy_btree_v1_classifies_and_reads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("btree_v1.h5");
    {
        // The 1.8 format uses a v1 B-tree for every chunked dataset.
        let file = c_file(&path, LibraryVersion::V18, LibraryVersion::V18);
        let ds = file
            .new_dataset::<i32>()
            .chunk((4,))
            .shape((8,))
            .create("d")
            .unwrap();
        ds.write(&(0..8).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    assert_eq!(ds.chunk_index().unwrap(), Some(ChunkIndex::BTreeV1));
    assert!(!ChunkIndex::BTreeV1.supports_inplace_append());

    let mut chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    chunks.sort_by_key(|c| c.offset.clone());
    assert_eq!(chunk_i32_at(&path, &chunks[0]), vec![0, 1, 2, 3]);
    assert_eq!(chunk_i32_at(&path, &chunks[1]), vec![4, 5, 6, 7]);
    // Pure whole-dataset read agrees with the reference library.
    assert_eq!(ds.read_i32().unwrap(), read_c_i32(&path, "d"));
}

// ---- filters ----------------------------------------------------------------

#[test]
fn c_shuffle_then_deflate_pipeline() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("filters.h5");
    {
        let file = c_file(&path, LibraryVersion::V110, LibraryVersion::latest());
        let ds = file
            .new_dataset::<i32>()
            .shuffle()
            .deflate(6)
            .chunk((16,))
            .shape((64,))
            .create("d")
            .unwrap();
        ds.write(&(0..64).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();

    let pipeline = ds.filter_pipeline();
    let ids: Vec<u16> = pipeline.iter().map(|flt| flt.id).collect();
    assert_eq!(
        ids,
        vec![2, 1],
        "shuffle (2) then deflate (1), in write order"
    );
    assert_eq!(ds.filters(), vec![2, 1]);

    let shuffle = pipeline.iter().find(|flt| flt.id == 2).unwrap();
    let deflate = pipeline.iter().find(|flt| flt.id == 1).unwrap();
    assert_eq!(deflate.client_data, vec![6], "deflate level");
    assert_eq!(shuffle.client_data, vec![4], "shuffle element size (i32)");
    // The reference wrapper records filters as OPTIONAL (filter-flags bit 0),
    // "the same way h5py does it", so a reader may skip one it lacks. Our
    // `is_optional` reflects that faithfully — unlike the pure writer, which
    // writes them mandatory (see the pure `deflate_pipeline_details` test).
    assert!(
        shuffle.is_optional && deflate.is_optional,
        "h5py/metno mark filters optional"
    );
    for c in ds.chunks().unwrap() {
        assert_eq!(c.filter_mask, 0);
    }
}

// ---- v2 B-tree (classified, not yet enumerable) -----------------------------

#[test]
fn c_btree_v2_classifies_but_chunks_unsupported() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("btree_v2.h5");
    {
        // Two unlimited dimensions select a v2 B-tree chunk index. Write data so
        // the index is actually allocated (an unallocated index would report zero
        // chunks rather than exercise the missing walker).
        let file = c_file(&path, LibraryVersion::V110, LibraryVersion::latest());
        let ds = file
            .new_dataset::<i32>()
            .chunk((2, 2))
            .shape((Extent::resizable(4), Extent::resizable(4)))
            .create("d")
            .unwrap();
        ds.write_raw(&(0..16).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("d").unwrap();
    // Classified from the layout message alone.
    assert_eq!(ds.chunk_index().unwrap(), Some(ChunkIndex::BTreeV2));
    assert!(matches!(
        ds.layout().unwrap(),
        Layout::Chunked {
            index: ChunkIndex::BTreeV2,
            ..
        }
    ));
    // Enumerating a v2-B-tree index is not supported yet: a clear error, not a
    // silent empty list.
    assert!(ds.chunks().is_err());
}
