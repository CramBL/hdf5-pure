//! Pure-Rust tests for the public chunk-layout and filter introspection API
//! (issue #149): `Dataset::layout`, `chunk_index`, `chunks`, and
//! `filter_pipeline`, plus the `ChunkIndex::supports_inplace_append` helper.
//!
//! The pure writer emits *contiguous* (never compact) storage for unchunked
//! data and picks the chunk index from the shape: unlimited maxshape ->
//! extensible array, a single chunk -> single-chunk, otherwise fixed array. The
//! `Compact` and legacy `BTreeV1` variants are exercised against
//! reference-C-library files in `crates/crosscheck/tests/main/layout_introspection.rs`.

use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::FilterDisposition;
use hdf5_pure::FilterId;
use hdf5_pure::FormatError;
use hdf5_pure::UserblockSize;
use hdf5_pure::{ChunkIndex, Dataset, File, FileBuilder, Layout, MaxExtent};

#[rstest::rstest]
#[case::two_byte_max(65_535)]
#[case::three_byte_min(65_536)]
#[case::cell_ranger(80_000)]
#[case::larger(131_072)]
fn chunk_dimensions_at_and_above_the_two_byte_limit_preserve_values(#[case] chunk: u64) {
    let values: Vec<i32> = (0..200_000).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_i32_data(&values)
        .with_chunks(&[chunk])
        .with_maxshape(&[MaxExtent::Unlimited]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.shape().unwrap(), vec![200_000]);
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![chunk]));
    assert_eq!(dataset.read_i32().unwrap(), values);
}

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
    let dir = tempfile::tempdir().unwrap();
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
    let error = ds.chunks().unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::ChunkedReadError(reason)) if reason == "chunk enumeration requires a chunked dataset"),
        "{error:?}"
    );
    assert!(ds.filter_pipeline().unwrap().is_empty());
    assert!(ds.filters().unwrap().is_empty());
}

// ---- chunked index kinds ----------------------------------------------------

#[test]
fn chunked_single_chunk() {
    let dir = tempfile::tempdir().unwrap();
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
    assert_eq!(chunks[0].filter_mask().raw_bits(), 0);
    assert!(chunks[0].address != 0);
}

#[test]
fn chunked_fixed_array() {
    let dir = tempfile::tempdir().unwrap();
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
        assert_eq!(c.filter_mask().raw_bits(), 0);
        assert!(c.address != 0);
    }
    // Distinct chunk addresses.
    assert_ne!(chunks[0].address, chunks[1].address);
}

/// The oracle is the file's own bytes: an address left in the stored frame
/// points 512 bytes short of the chunk, at plausible data.
#[test]
fn chunk_addresses_of_a_userblock_file_are_absolute() {
    const USERBLOCK: u64 = 512;
    const CHUNK_ELEMS: usize = 4;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("userblock.h5");
    let values: Vec<i32> = (0..8).collect();
    {
        let mut b = FileBuilder::new();
        b.with_userblock(UserblockSize::try_from(USERBLOCK).unwrap());
        b.create_dataset("d")
            .with_i32_data(&values)
            .with_shape(&[values.len() as u64])
            .with_chunks(&[CHUNK_ELEMS as u64]);
        b.write(&p).unwrap();
    }
    let bytes = std::fs::read(&p).unwrap();

    let f = File::open(&p).unwrap();
    let chunks = open(&f, "d").chunks().unwrap();

    assert_eq!(chunks.len(), 2);
    for c in &chunks {
        assert!(
            c.address >= USERBLOCK,
            "a chunk cannot lie inside the userblock: {c:?}"
        );
        let start = usize::try_from(c.address).unwrap();
        let stored: Vec<i32> = bytes[start..start + usize::try_from(c.storage_size).unwrap()]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        let first = usize::try_from(c.offset[0]).unwrap();
        assert_eq!(
            stored,
            values[first..first + CHUNK_ELEMS],
            "the bytes at the reported address are the chunk's own: {c:?}"
        );
    }
}

#[test]
fn chunked_extensible_array_supports_append() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("ea.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_maxshape(&[MaxExtent::Unlimited]) // unlimited -> extensible array
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
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("deflate.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_chunks(&[4])
            .with_deflate(DeflateLevel::Six);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    let pipeline = ds.filter_pipeline().unwrap();
    assert_eq!(pipeline.len(), 1);
    let deflate = &pipeline[0];
    assert_eq!(deflate.id, FilterId::Deflate, "deflate filter id");
    assert_eq!(deflate.client_data(), vec![6], "compression level");
    assert!(!deflate.is_optional);
    assert_eq!(
        ds.filters().unwrap(),
        vec![FilterId::Deflate],
        "lightweight id list agrees"
    );

    for c in ds.chunks().unwrap() {
        assert!(c.storage_size > 0);
        assert_eq!(
            c.filter_mask().raw_bits(),
            0,
            "all filters applied to every chunk"
        );
    }
}

#[test]
fn shuffle_then_deflate_pipeline_order() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("shuffle_deflate.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<_>>())
            .with_shape(&[64])
            .with_chunks(&[16])
            .with_shuffle()
            .with_deflate(DeflateLevel::Six);
        b.write(&p).unwrap();
    }

    let f = File::open(&p).unwrap();
    let ds = open(&f, "d");

    let ids: Vec<FilterId> = ds
        .filter_pipeline()
        .unwrap()
        .iter()
        .map(|flt| flt.id)
        .collect();
    assert_eq!(
        ids,
        vec![FilterId::Shuffle, FilterId::Deflate],
        "shuffle (2) then deflate (1), in write order"
    );
    assert_eq!(
        ds.filters().unwrap(),
        vec![FilterId::Shuffle, FilterId::Deflate]
    );
    for flt in ds.filter_pipeline().unwrap() {
        assert!(!flt.is_optional);
    }
}

// ---- unallocated + invariants ----------------------------------------------

#[test]
fn unallocated_extensible_has_no_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("empty_ea.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&[]) // empty
            .with_shape(&[0])
            .with_maxshape(&[MaxExtent::Unlimited])
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
    let dir = tempfile::tempdir().unwrap();
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

#[test]
fn chunks_retain_their_own_pipeline_after_the_dataset_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pipelines.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("first")
        .with_i32_data(&[1, 2, 3, 4])
        .with_chunks(&[2])
        .with_shuffle()
        .with_fletcher32();
    builder
        .create_dataset("second")
        .with_i32_data(&[1, 2, 3, 4])
        .with_chunks(&[2])
        .with_deflate(DeflateLevel::Six)
        .with_fletcher32();
    builder.write(&path).unwrap();
    let (first, second) = {
        let file = File::open(&path).unwrap();
        (
            file.dataset("first").unwrap().chunks().unwrap(),
            file.dataset("second").unwrap().chunks().unwrap(),
        )
    };
    let first_view = first[0].filters();
    let second_view = second[0].filters();
    assert_eq!(
        first_view
            .dispositions()
            .map(|(filter, _)| filter.id)
            .collect::<Vec<_>>(),
        [FilterId::Shuffle, FilterId::Fletcher32]
    );
    assert_eq!(
        second_view
            .dispositions()
            .map(|(filter, _)| filter.id)
            .collect::<Vec<_>>(),
        [FilterId::Deflate, FilterId::Fletcher32]
    );
    assert!(
        first_view
            .dispositions()
            .all(|(_, state)| state == FilterDisposition::Applied)
    );
    assert_eq!(first_view.filter(2), None);
}
