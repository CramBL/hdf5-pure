// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! A chunk index numbers its slots over the dataset's *maximum* chunk grid, and
//! an Extensible Array rotates its unlimited dimension to the front first
//! (issue #299). Both facts are invisible to a round trip through this crate
//! alone — the writer and the reader agreed with each other while disagreeing
//! with every other HDF5 reader — so the oracle here is the reference C library
//! on the other side of each file.
//!
//! The combinations are swept rather than sampled. There are few enough of them
//! to enumerate, and the defect this file exists for was invisible in exactly
//! the sample the suite already had: every chunked crosscheck was either rank 1
//! or `maxshape == shape`, and growing the *first* dimension changes no
//! multiplier, so three of the four obvious fixtures pass over the bug.

use hdf5::file::LibraryVersion;
use hdf5_pure::{ChunkIndex, File, FileBuilder};
use tempfile::tempdir;

/// `H5S_UNLIMITED`.
const U: u64 = u64::MAX;

/// One swept combination: a label for the assertion messages, then the shape,
/// the chunk dimensions and the maximum shape.
type Case = (String, Vec<u64>, Vec<u64>, Vec<u64>);

/// One family of cases: a label, a shape, chunk dimensions, and the maximum
/// shapes to try against them.
type Family<'a> = (&'a str, &'a [u64], &'a [u64], &'a [&'a [u64]]);

/// Every shape/chunk/maximum-shape combination the sweep covers, with a label
/// for the assertion messages.
///
/// Ranks 1 to 3, and for each the maximum shape equal to the shape, larger in
/// each dimension separately and in all of them, and unlimited in each
/// dimension separately. Two chunk geometries per rank: one that yields several
/// chunks, and one whose single chunk covers the shape — that second family is
/// what distinguishes "one chunk stored" from "one chunk possible", and writing
/// a single-chunk layout for the first made the reference library abort.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let families: &[Family<'_>] = &[
        ("r1", &[7], &[3], &[&[7][..], &[16][..], &[U][..]]),
        ("r1 one chunk", &[3], &[4], &[&[3][..], &[8][..], &[U][..]]),
        (
            "r2",
            &[3, 3],
            &[2, 2],
            &[
                &[3, 3][..],
                &[8, 3][..],
                &[3, 8][..],
                &[8, 8][..],
                &[U, 3][..],
                &[3, U][..],
                &[U, 8][..],
                &[8, U][..],
            ],
        ),
        (
            "r2 one chunk",
            &[3, 3],
            &[4, 4],
            &[&[3, 3][..], &[8, 8][..], &[U, 3][..], &[3, U][..]],
        ),
        (
            "r3",
            &[2, 3, 4],
            &[1, 2, 2],
            &[
                &[2, 3, 4][..],
                &[4, 3, 4][..],
                &[2, 6, 4][..],
                &[2, 3, 8][..],
                &[4, 6, 8][..],
                &[U, 3, 4][..],
                &[2, U, 4][..],
                &[2, 3, U][..],
                &[U, 6, 8][..],
                &[4, 6, U][..],
            ],
        ),
    ];
    for (family, shape, chunks, maxshapes) in families {
        for ms in *maxshapes {
            out.push((
                format!("{family} shape={shape:?} chunks={chunks:?} max={ms:?}"),
                shape.to_vec(),
                chunks.to_vec(),
                ms.to_vec(),
            ));
        }
    }
    out
}

/// `1..=n` for a shape of `n` elements: every value distinct, so a chunk landing
/// at the wrong offset shows up as a wrong value rather than a coincidence.
fn values(shape: &[u64]) -> Vec<u32> {
    (1..=shape.iter().product::<u64>() as u32).collect()
}

fn pure_write(path: &std::path::Path, shape: &[u64], chunks: &[u64], maxshape: &[u64]) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(shape))
        .with_shape(shape)
        .with_maxshape(maxshape)
        .with_chunks(chunks);
    b.write(path).unwrap();
}

/// Write through the reference C library, pinned to the latest format so it
/// chooses a Fixed or Extensible Array.
///
/// Without the bound it writes a version-1 B-tree, whose keys carry each chunk's
/// coordinates explicitly — so it exercises none of the positional numbering
/// this file is about, and a read crosscheck against it passes no matter what.
fn c_write(path: &std::path::Path, shape: &[u64], chunks: &[u64], maxshape: &[u64]) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fp| fp.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<u32>()
        .chunk(chunks.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .shape(
            shape
                .iter()
                .zip(maxshape)
                .map(|(&s, &m)| hdf5::Extent::new(s as usize, (m != U).then_some(m as usize)))
                .collect::<Vec<_>>(),
        )
        .create("d")
        .unwrap();
    ds.write_raw(&values(shape)).unwrap();
    // The dataset handle keeps the file open; a close with it still alive
    // leaves the object header unflushed and the file unreadable.
    drop(ds);
    file.close().unwrap();
}

/// Whether the reference library will create this geometry at all.
///
/// It refuses a chunk larger than the dataset's current extent ("Chunk
/// dimensions exceed data shape"), where this crate accepts one and stores a
/// single partly-empty chunk. That drops both one-chunk families from the two
/// directions that need the C library to *write*; the case they exist for —
/// one chunk stored where the maximum shape allows more — is asserted against a
/// C *read* by
/// `one_stored_chunk_with_room_for_more_is_not_the_single_chunk_layout`, which
/// is the direction where getting it wrong aborts the process.
fn c_library_accepts(shape: &[u64], chunks: &[u64]) -> bool {
    chunks.iter().zip(shape).all(|(&c, &s)| c <= s)
}

fn c_read(path: &std::path::Path) -> Vec<u32> {
    c_read_named(path, "d")
}

fn c_read_named(path: &std::path::Path, name: &str) -> Vec<u32> {
    let file = hdf5::File::open(path).unwrap();
    let v = file.dataset(name).unwrap().read_raw::<u32>().unwrap();
    file.close().unwrap();
    v
}

fn pure_read(path: &std::path::Path) -> Vec<u32> {
    let file = File::open(path).unwrap();
    file.dataset("d").unwrap().read_u32().unwrap()
}

/// The pure writer's files must read back correctly *through the C library*.
///
/// The pure reader is asserted too, but it is the weaker half: it agreed with
/// the broken writer, which is exactly how the defect survived 2,000 tests.
#[test]
fn c_reads_every_maxshape_this_crate_writes() {
    let dir = tempdir().unwrap();
    for (label, shape, chunks, maxshape) in cases() {
        let path = dir.path().join("pure.h5");
        pure_write(&path, &shape, &chunks, &maxshape);
        assert_eq!(pure_read(&path), values(&shape), "pure read: {label}");
        assert_eq!(c_read(&path), values(&shape), "C read: {label}");
        std::fs::remove_file(&path).unwrap();
    }
}

/// The other direction: what the C library writes with a positional index, this
/// crate must read.
#[test]
fn pure_reads_every_maxshape_the_c_library_writes() {
    let dir = tempdir().unwrap();
    let mut skipped = 0;
    for (label, shape, chunks, maxshape) in cases() {
        if !c_library_accepts(&shape, &chunks) {
            skipped += 1;
            continue;
        }
        let path = dir.path().join("c.h5");
        c_write(&path, &shape, &chunks, &maxshape);
        assert_eq!(pure_read(&path), values(&shape), "{label}");
        std::fs::remove_file(&path).unwrap();
    }
    // A filter that quietly matched everything would leave this test passing
    // over an empty sweep, so the number it drops is asserted rather than
    // assumed.
    assert_eq!(skipped, 7, "only the one-chunk families are skipped");
}

/// The index *kind* has to match the reference library's choice too, and it is
/// chosen from the maximum grid rather than from the chunks stored.
///
/// Values alone do not pin this down: a single-chunk layout resolves chunk 0 to
/// the right address, so a dataset of one chunk that could hold four reads
/// correctly and still trips `H5D__single_idx_get_addr`'s assertion — the
/// reference library aborts the reading process rather than returning anything
/// at all. Comparing the kind against a C-written file of the same geometry is
/// what states the rule.
#[test]
fn the_index_kind_matches_what_the_reference_library_chooses() {
    let dir = tempdir().unwrap();
    let mut skipped = 0;
    for (label, shape, chunks, maxshape) in cases() {
        if !c_library_accepts(&shape, &chunks) {
            skipped += 1;
            continue;
        }
        let pure_path = dir.path().join("pure.h5");
        let c_path = dir.path().join("c.h5");
        pure_write(&pure_path, &shape, &chunks, &maxshape);
        c_write(&c_path, &shape, &chunks, &maxshape);

        let kind = |p: &std::path::Path| {
            File::open(p)
                .unwrap()
                .dataset("d")
                .unwrap()
                .chunk_index()
                .unwrap()
        };
        assert_eq!(kind(&pure_path), kind(&c_path), "{label}");

        std::fs::remove_file(&pure_path).unwrap();
        std::fs::remove_file(&c_path).unwrap();
    }
    assert_eq!(skipped, 7, "only the one-chunk families are skipped");
}

/// A single chunk that the maximum shape lets grow into a second is a Fixed
/// Array, not the single-chunk layout its one chunk suggests.
///
/// Spelled out on its own because it is the case with a *crash* on the other
/// side rather than a wrong value, and because it is the one an "is it one
/// chunk?" test would get wrong in the obvious way.
#[test]
fn one_stored_chunk_with_room_for_more_is_not_the_single_chunk_layout() {
    let dir = tempdir().unwrap();
    let fixed = dir.path().join("fixed.h5");
    let growable = dir.path().join("growable.h5");
    pure_write(&fixed, &[3, 3], &[4, 4], &[3, 3]);
    pure_write(&growable, &[3, 3], &[4, 4], &[8, 8]);

    let kind = |p: &std::path::Path| {
        File::open(p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .chunk_index()
            .unwrap()
    };
    assert_eq!(kind(&fixed), Some(ChunkIndex::SingleChunk));
    assert_eq!(kind(&growable), Some(ChunkIndex::FixedArray));
    assert_eq!(c_read(&growable), values(&[3, 3]));
}

/// Two unlimited dimensions are refused at write time.
///
/// The reference library indexes that dataspace with a version-2 B-tree, which
/// this crate does not write; producing an Extensible Array instead gave a file
/// `H5Dopen2` rejects outright ("already found unlimited dimension"), readable
/// by nothing but this crate.
#[test]
fn two_unlimited_dimensions_are_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("two.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(&[3, 3]))
        .with_shape(&[3, 3])
        .with_maxshape(&[U, U])
        .with_chunks(&[2, 2]);
    let err = b.write(&path).unwrap_err();
    assert!(format!("{err}").contains("at most one dimension"), "{err}");
}

/// The refusal covers the in-place session path too, not only the whole-file
/// writer.
///
/// A session validates a dataset's geometry as it stages it, so the refusal
/// arrives from `create_dataset` itself and nothing is ever staged to commit.
#[test]
fn two_unlimited_dimensions_are_refused_in_a_session_too() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("session.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_u32_data(&[1, 2, 3]);
    b.write(&path).unwrap();

    // Scoped, so the session's file lock is gone before either library reopens
    // the file below.
    {
        let session = hdf5_pure::File::open_rw(&path).unwrap();
        let err = session
            .root()
            .create_dataset("d", |b| {
                b.with_u32_data(&values(&[3, 3]))
                    .with_shape(&[3, 3])
                    .with_maxshape(&[U, U])
                    .with_chunks(&[2, 2]);
            })
            .unwrap_err();
        assert!(format!("{err}").contains("at most one dimension"), "{err}");
        assert!(
            !session.has_staged_edits(),
            "a refused addition must not be left staged"
        );
        session.commit().unwrap();
    }

    // Nothing was written: the file still holds what it did, and not a
    // half-written dataset the reference library would choke on.
    assert_eq!(
        c_read_named(&path, "seed"),
        vec![1, 2, 3],
        "a refused commit must leave the file as it was"
    );
    assert!(
        hdf5_pure::File::open(&path).unwrap().dataset("d").is_err(),
        "the refused dataset must not be present"
    );
}

/// A maximum shape whose chunk index would be mostly empty is refused rather
/// than written.
///
/// A Fixed Array declares a slot for every chunk the maximum allows, so this
/// dataset of four chunks would otherwise ask for an index of 10^12 elements.
/// The refusal names where the unused elements come from.
#[test]
fn a_maximum_shape_needing_a_mostly_empty_index_is_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("huge.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(&[3, 3]))
        .with_shape(&[3, 3])
        .with_maxshape(&[2_000_000, 2_000_000])
        .with_chunks(&[2, 2]);
    let err = b.write(&path).unwrap_err();
    let text = format!("{err}");
    assert!(text.contains("chunk index"), "{text}");
    assert!(text.contains("unused"), "{text}");

    // Room to grow along the dimension the dataset grows in costs no unused
    // elements at all, however far it reaches — that is the shape the refusal
    // points at, and it is accepted.
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(&[3, 3]))
        .with_shape(&[3, 3])
        .with_maxshape(&[U, 3])
        .with_chunks(&[2, 2]);
    b.write(&path).unwrap();
    assert_eq!(c_read(&path), values(&[3, 3]));

    // And an unlimited dimension paired with a wide fixed one is accepted, since
    // an Extensible Array allocates only the blocks its chunks land in: the
    // unused elements are the slack inside those, not the whole span. A bound on
    // the span refused this.
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&values(&[8, 8]))
        .with_shape(&[8, 8])
        .with_maxshape(&[U, 131_072])
        .with_chunks(&[1, 1]);
    b.write(&path).unwrap();
    assert_eq!(c_read(&path), values(&[8, 8]));
}

/// `repack` of a sparse chunked dataset the reference library wrote produces a
/// file of the same order, not one inflated by the gap between its chunks.
///
/// This is the user-visible face of the block allocation: reading such a file
/// always worked, so a crate that inflated it on rewrite could read files it
/// could not faithfully copy. At `maxshape [U, 65536]` the rewrite was 11x the
/// source, and one step wider it was refused outright.
#[test]
fn repack_of_a_c_written_sparse_dataset_stays_the_same_order_of_size() {
    let dir = tempdir().unwrap();
    for wide in [65_536u64, 131_072, 1_000_000] {
        let src = dir.path().join("c.h5");
        let dst = dir.path().join("repacked.h5");
        let shape = [8u64, 8];
        c_write(&src, &shape, &[1, 1], &[U, wide]);
        hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::new()).unwrap();

        let (src_len, dst_len) = (
            std::fs::metadata(&src).unwrap().len(),
            std::fs::metadata(&dst).unwrap().len(),
        );
        assert!(
            dst_len <= src_len * 2,
            "maxshape [U, {wide}]: repack wrote {dst_len} bytes for a {src_len}-byte source"
        );
        assert_eq!(pure_read(&dst), values(&shape));
        assert_eq!(c_read(&dst), values(&shape));

        std::fs::remove_file(&src).unwrap();
        std::fs::remove_file(&dst).unwrap();
    }
}

/// A resizable multi-dimensional dataset still reads correctly after it is
/// actually grown, which is the point of declaring a maximum shape at all.
///
/// The sweep above never resizes, so it would pass over a numbering that is
/// right for the original extent and wrong for the extended one.
#[test]
fn a_grown_multidimensional_dataset_reads_back_through_both_libraries() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("grown.h5");

    // Written by the C library, since this crate's own resize is rank-1 only.
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fp| fp.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(&path)
        .unwrap();
    let ds = file
        .new_dataset::<u32>()
        .chunk([2usize, 2])
        .shape([hdf5::Extent::new(3, None), hdf5::Extent::new(3, Some(8))])
        .create("d")
        .unwrap();
    ds.write_raw(&values(&[3, 3])).unwrap();
    ds.resize([6usize, 3]).unwrap();
    ds.write_raw(&(1u32..=18).collect::<Vec<_>>()).unwrap();
    drop(ds);
    file.close().unwrap();

    assert_eq!(pure_read(&path), (1u32..=18).collect::<Vec<_>>());
}

/// `repack` rewrites a dataset's chunks verbatim and rebuilds its index from
/// chunk sizes alone, through a second planner that has to number the slots the
/// same way the encoding path does.
///
/// The sweep above never reaches that planner: it plans from a shape it is
/// handed rather than from a dataset it encodes, and it was the one call site
/// that had no shape to number over at all.
#[test]
fn repack_preserves_a_resizable_multidimensional_dataset() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");

    // Maximum shape wider in the *trailing* dimension, so the slot numbering
    // differs from dense grid order: chunks land in slots 0, 1, 4 and 5.
    let data: Vec<u32> = (1..=64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&data)
        .with_shape(&[8, 8])
        .with_maxshape(&[8, 16])
        .with_chunks(&[4, 4])
        .with_deflate(6);
    b.write(&src).unwrap();

    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::new()).unwrap();

    assert_eq!(pure_read(&dst), data, "pure read of the repacked file");
    assert_eq!(c_read(&dst), data, "C read of the repacked file");
}

/// An in-place overwrite whose re-encoded chunks shrink rebuilds the index where
/// it sits, to record the new stored sizes. That rebuild has to put each chunk
/// back in the slot it came from.
///
/// The chunks are enumerated in dense grid order, so a rebuild that numbered
/// them densely would move every chunk past the first row — silently, since the
/// rebuilt index is the same length either way.
#[test]
fn a_shrinking_inplace_overwrite_keeps_each_chunk_in_its_slot() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shrink.h5");

    // Incompressible first, so the chunk slots are large.
    let noisy: Vec<u32> = (0..64u32)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ (i << 7))
        .collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&noisy)
        .with_shape(&[8, 8])
        .with_maxshape(&[8, 16])
        .with_chunks(&[4, 4])
        .with_deflate(6);
    b.write(&path).unwrap();
    let size_before = std::fs::metadata(&path).unwrap().len();

    // Distinct values that still deflate to less than the noise did, so every
    // chunk fits its slot with room to spare and the index is rebuilt in place.
    let tidy: Vec<u32> = (1..=64).collect();
    {
        let session = hdf5_pure::File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_u32_data(&tidy).with_shape(&[8, 8]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "the overwrite must fit its slots, or it relocates and tests a different path"
    );

    assert_eq!(pure_read(&path), tidy);
    assert_eq!(c_read(&path), tidy);
}

/// A gap in an Extensible Array can leave a *paged* data block whose first page
/// holds no chunk, and every page of a written block is an initialized page.
///
/// The page-init bitmap says which pages of a data block exist to be stepped
/// over. Marking the leading `n` of them for a block holding `n` pages' worth of
/// chunks says the same thing only while the array is gapless, which it stopped
/// being when slots started being numbered over the maximum grid. The chunks in
/// the pages after the gap then belong to no page any reader visits.
///
/// The geometry has to be this big: a data block is paged only past 1,024
/// elements, so the array must span enough slots to reach one. The chunks are
/// few; it is the span between them that costs, and about a megabyte of index is
/// the cheapest this defect can be reproduced for.
#[test]
fn a_paged_data_block_with_an_empty_first_page_keeps_its_chunks() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let shape = [40u64, 40];
    let data = values(&shape);
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&data)
        .with_shape(&shape)
        .with_chunks(&[1, 1])
        .with_maxshape(&[U, 3500]);
    b.write(&path).unwrap();

    assert_eq!(pure_read(&path), data, "pure read");
    assert_eq!(c_read(&path), data, "C read");
}

/// The reading half of the same fact: the reference library populates the pages
/// of a data block in whatever order chunks are written, so a cleared bit is a
/// page to step over rather than the end of the block.
///
/// Both of this crate's readers are asserted, because the buffered one walks the
/// pages from a byte slice and the streaming one sizes a read around them —
/// different code, same rule.
#[test]
fn pure_reads_a_c_written_block_whose_pages_are_not_leading() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_paged.h5");
    let shape = [40u64, 40];
    let data = values(&shape);
    c_write(&path, &shape, &[1, 1], &[U, 12_000]);

    assert_eq!(
        hdf5_pure::File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_u32()
            .unwrap(),
        data,
        "buffered reader"
    );
    assert_eq!(
        hdf5_pure::File::open_streaming(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_u32()
            .unwrap(),
        data,
        "streaming reader"
    );
}

/// The Extensible Array arm of the in-place index rebuild, alongside the Fixed
/// Array arm above. The two arms are separate match arms over separate builders,
/// so one passing says nothing about the other.
#[test]
fn a_shrinking_inplace_overwrite_keeps_ea_chunks_in_their_slots() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shrink_ea.h5");

    let noisy: Vec<u32> = (0..64u32)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ (i << 7))
        .collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_u32_data(&noisy)
        .with_shape(&[8, 8])
        .with_maxshape(&[U, 16])
        .with_chunks(&[4, 4])
        .with_deflate(6);
    b.write(&path).unwrap();
    assert_eq!(
        hdf5_pure::File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .chunk_index()
            .unwrap(),
        Some(ChunkIndex::ExtensibleArray),
        "this fixture exists to exercise the extensible arm"
    );
    let size_before = std::fs::metadata(&path).unwrap().len();

    let tidy: Vec<u32> = (1..=64).collect();
    {
        let session = hdf5_pure::File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_u32_data(&tidy).with_shape(&[8, 8]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "the overwrite must fit its slots, or it relocates and tests a different path"
    );
    assert_eq!(pure_read(&path), tidy);
    assert_eq!(c_read(&path), tidy);
}

/// An Extensible Array allocates a data block when a chunk lands in it, so a
/// sparse index costs what its chunks cost rather than what the gap between them
/// does.
///
/// The oracle is the array's own six header statistics, compared against the
/// reference library's for the same dataset. They are the strongest thing
/// available here: a size comparison would pass on a file whose blocks are the
/// right total size and the wrong shape, and the reference library recomputes
/// these on open and rejects a file whose stored figures disagree with what it
/// derives. Writing every block in the span instead put `ndata_blks` at 425
/// against libhdf5's 8, and the file at 38x its size, while reading back
/// perfectly through both libraries (issue #299).
#[test]
fn a_sparse_extensible_array_matches_the_c_library_block_statistics() {
    /// The six `EAHD` statistics, in stored order.
    fn stats_of(path: &std::path::Path) -> Vec<u64> {
        let b = std::fs::read(path).unwrap();
        let h = (0..b.len() - 4)
            .find(|&i| &b[i..i + 4] == b"EAHD")
            .expect("this dataset must be Extensible-Array indexed");
        (0..6)
            .map(|k| {
                let p = h + 12 + k * 8;
                u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
            })
            .collect()
    }

    let dir = tempdir().unwrap();
    for (shape, wide) in [
        ([16u64, 16], 1024u64),
        ([16, 16], 8192),
        ([16, 16], 65536),
        ([8, 8], 100_000),
        ([8, 8], 1_000_000),
    ] {
        let maxshape = [U, wide];
        let pure_path = dir.path().join("pure.h5");
        let c_path = dir.path().join("c.h5");
        pure_write(&pure_path, &shape, &[1, 1], &maxshape);
        c_write(&c_path, &shape, &[1, 1], &maxshape);

        assert_eq!(
            stats_of(&pure_path),
            stats_of(&c_path),
            "EAHD statistics differ for shape {shape:?} maxshape {maxshape:?}"
        );
        assert_eq!(pure_read(&pure_path), values(&shape));
        assert_eq!(c_read(&pure_path), values(&shape));

        std::fs::remove_file(&pure_path).unwrap();
        std::fs::remove_file(&c_path).unwrap();
    }
}
