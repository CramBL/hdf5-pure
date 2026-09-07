//! Pure-Rust tests for the public chunk-layout and filter introspection API
//! (issue #149): `Dataset::layout`, `chunk_index`, `chunks`, and
//! `filter_pipeline`, plus the `ChunkIndex::supports_inplace_append` helper.
//!
//! The pure writer emits *contiguous* (never compact) storage for unchunked
//! data and picks the chunk index from the shape: unlimited maxshape ->
//! extensible array, a single chunk -> single-chunk, otherwise fixed array. The
//! `Compact` and legacy `BTreeV1` variants are exercised against
//! reference-C-library files in `crates/crosscheck/tests/layout_introspection.rs`.

use hdf5_pure::{ChunkIndex, Dataset, File, FileBuilder, Layout};
use tempfile::tempdir;

fn open(f: &File, name: &str) -> Dataset {
    f.dataset(name).unwrap()
}

/// Chunk offsets returned by `chunks()`, sorted for order-independent asserts.
fn sorted_offsets(ds: &Dataset) -> Vec<Vec<u64>> {
    let mut offs: Vec<Vec<u64>> = ds.chunks().unwrap().into_iter().map(|c| c.offset).collect();
    offs.sort();
    offs
}

// ---- contiguous -------------------------------------------------------------

#[test]
fn contiguous_fixed_dataset() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("contig.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..12).collect::<Vec<_>>())
            .with_shape(&[12]);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    match ds.layout().unwrap() {
        Layout::Contiguous { address, size } => {
            assert_eq!(size, 12 * 4, "12 i32 elements");
            assert!(matches!(address, Some(a) if a != 0), "allocated address");
        }
        other => panic!("expected contiguous, got {other:?}"),
    }
    assert!(!ds.is_chunked());
    assert_eq!(ds.chunk_index().unwrap(), None);
    assert!(
        ds.chunks().is_err(),
        "chunks() errors on a non-chunked dataset"
    );
    assert!(ds.filter_pipeline().is_empty());
    assert!(ds.filters().is_empty());
}

// ---- chunked index kinds ----------------------------------------------------

#[test]
fn chunked_single_chunk() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("single.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_chunks(&[8]); // one chunk covers the whole dataset
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    assert_eq!(
        ds.layout().unwrap(),
        Layout::Chunked {
            chunk_shape: vec![8],
            index: ChunkIndex::SingleChunk,
        }
    );
    assert_eq!(ds.chunk_index().unwrap(), Some(ChunkIndex::SingleChunk));
    assert!(!ChunkIndex::SingleChunk.supports_inplace_append());

    let chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].offset, vec![0]);
    assert_eq!(chunks[0].storage_size, 8 * 4, "unfiltered chunk byte size");
    assert_eq!(chunks[0].filter_mask, 0);
    assert!(chunks[0].address != 0);
}

#[test]
fn chunked_fixed_array() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("fixed.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_chunks(&[4]); // 2 chunks, fixed shape -> fixed array
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    assert_eq!(
        ds.layout().unwrap(),
        Layout::Chunked {
            chunk_shape: vec![4],
            index: ChunkIndex::FixedArray,
        }
    );
    assert!(!ChunkIndex::FixedArray.supports_inplace_append());

    let chunks = ds.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(sorted_offsets(&ds), vec![vec![0], vec![4]]);
    for c in &chunks {
        assert_eq!(c.storage_size, 4 * 4);
        assert_eq!(c.filter_mask, 0);
        assert!(c.address != 0);
    }
    // Distinct chunk addresses.
    assert_ne!(chunks[0].address, chunks[1].address);
}

#[test]
fn chunked_extensible_array_supports_append() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("ea.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_maxshape(&[u64::MAX]) // unlimited -> extensible array
            .with_chunks(&[4]);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    let index = ds.chunk_index().unwrap().unwrap();
    assert_eq!(index, ChunkIndex::ExtensibleArray);
    assert!(
        index.supports_inplace_append(),
        "extensible array is the appendable index"
    );
    assert!(matches!(
        ds.layout().unwrap(),
        Layout::Chunked {
            index: ChunkIndex::ExtensibleArray,
            ..
        }
    ));
    assert_eq!(ds.chunks().unwrap().len(), 2);
}

// ---- filters ----------------------------------------------------------------

#[test]
fn deflate_pipeline_details() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("deflate.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_chunks(&[4])
            .with_deflate(6);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    let pipeline = ds.filter_pipeline();
    assert_eq!(pipeline.len(), 1);
    let deflate = &pipeline[0];
    assert_eq!(deflate.id, 1, "deflate filter id");
    assert_eq!(deflate.client_data, vec![6], "compression level");
    assert!(!deflate.is_optional);
    assert_eq!(ds.filters(), vec![1], "lightweight id list agrees");

    for c in ds.chunks().unwrap() {
        assert!(c.storage_size > 0);
        assert_eq!(c.filter_mask, 0, "all filters applied to every chunk");
    }
}

#[test]
fn shuffle_then_deflate_pipeline_order() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("shuffle_deflate.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<_>>())
            .with_shape(&[64])
            .with_chunks(&[16])
            .with_shuffle()
            .with_deflate(6);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    let ids: Vec<u16> = ds.filter_pipeline().iter().map(|flt| flt.id).collect();
    assert_eq!(
        ids,
        vec![2, 1],
        "shuffle (2) then deflate (1), in write order"
    );
    assert_eq!(ds.filters(), vec![2, 1]);
    for flt in ds.filter_pipeline() {
        assert!(!flt.is_optional);
    }
}

// ---- unallocated + invariants ----------------------------------------------

#[test]
fn unallocated_extensible_has_no_chunks() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("empty_ea.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&[]) // empty
            .with_shape(&[0])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[8]);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    // Still classified as a chunked/extensible dataset...
    assert!(matches!(
        ds.layout().unwrap(),
        Layout::Chunked {
            index: ChunkIndex::ExtensibleArray,
            ..
        }
    ));
    // ...but no storage is allocated, so no chunks (Ok, not Err).
    assert_eq!(ds.chunks().unwrap(), vec![]);
}

#[test]
fn layout_chunk_shape_agrees_with_chunk_shape_accessor() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("agree.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..20).collect::<Vec<_>>())
            .with_shape(&[20])
            .with_chunks(&[6]);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    let Layout::Chunked { chunk_shape, .. } = ds.layout().unwrap() else {
        panic!("expected chunked");
    };
    assert_eq!(Some(chunk_shape), ds.chunk_shape().unwrap());
    assert!(ds.is_chunked());
}
