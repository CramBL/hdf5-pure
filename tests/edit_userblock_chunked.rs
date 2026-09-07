//! Chunked-dataset editing on files that carry a userblock (non-zero base
//! address) — the chunked follow-up to the userblock slice of issue #104.
//!
//! Every stored chunk-index and chunk-data address in such a file is relative to
//! the userblock base. These tests exercise the three chunked write paths on a
//! userblock file and confirm the userblock bytes survive byte-for-byte:
//!
//!   * adding a chunked / filtered dataset (the relocatable blob is built with
//!     stored addresses, at end-of-file or in a freed region that fits it),
//!   * overwriting a chunked dataset in place (the index is walked on a
//!     base-relative view and the write offsets shifted back), and
//!   * overwriting a chunked dataset with a relocation (a fresh blob is built and
//!     the old chunk storage reclaimed), including reuse of that reclaimed space
//!     by a later commit in the same session.

use hdf5_pure::{File, FileBuilder};

use temp::temp_path;
use test_util::temp;

const UB: usize = 512;

const MARKER: &[u8] = b"USERBLOCK-CHUNK-0104";

/// Stamp a recognizable marker across the userblock region of `bytes` and return
/// the 512-byte userblock as written, for later byte-for-byte comparison.
fn stamp_userblock(bytes: &mut [u8]) -> Vec<u8> {
    bytes[..MARKER.len()].copy_from_slice(MARKER);
    bytes[UB - 1] = 0xAB;
    bytes[..UB].to_vec()
}

fn assert_userblock_unchanged(path: &std::path::Path, original: &[u8]) {
    let after = std::fs::read(path).unwrap();
    assert_eq!(
        &after[..UB],
        original,
        "userblock bytes changed across the edit"
    );
}

#[test]
fn userblock_chunked_add_roundtrip() {
    // Add a deflate-compressed multi-chunk dataset to a userblock file alongside
    // an untouched contiguous dataset, then read both back.
    let path = temp_path("hdf5_pure_ub_chunk_add.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("contig")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let added: Vec<f64> = (0..1000).map(|i| (i % 13) as f64 * 0.25).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("chunked", |b| {
                b.with_f64_data(&added)
                    .with_shape(&[1000])
                    .with_chunks(&[64])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("chunked").unwrap().read_f64().unwrap(), added);
    assert_eq!(
        file.dataset("contig").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_userblock_unchanged(&path, &userblock);
    assert_eq!(&std::fs::read(&path).unwrap()[..MARKER.len()], MARKER);
}

#[test]
fn userblock_chunked_unfiltered_inplace_overwrite() {
    // An unfiltered chunked dataset: each chunk's slot is exactly its raw size, so
    // a same-shape overwrite re-fills every slot in place — no relocation, so the
    // file does not grow.
    let path = temp_path("hdf5_pure_ub_chunk_inplace.h5");
    let original: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[200])
        .with_chunks(&[32]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();

    let updated: Vec<f64> = (0..200).map(|i| (i as f64) * -2.0 + 1.0).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("c")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let len_after = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        len_after, len_before,
        "an unfiltered same-shape chunked overwrite must stay in place (no growth)"
    );
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("c")
            .unwrap()
            .read_f64()
            .unwrap(),
        updated
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_chunked_shrinking_inplace_overwrite() {
    // A deflate dataset whose new contents re-encode *smaller* than their slots
    // still fits in place: the chunks are rewritten into their existing slots and
    // the index is rebuilt in place to record the new (smaller) sizes — no
    // relocation, so the file does not grow. This exercises the base-aware
    // in-place index rebuild on a userblock file.
    let path = temp_path("hdf5_pure_ub_chunk_shrink.h5");
    // Poorly compressible original (high-entropy) vs. highly compressible
    // replacement (all equal): each new chunk re-encodes far smaller than its slot.
    let original: Vec<f64> = (0..400).map(|i| (i as f64).sin() * 1e6).collect();
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[400])
        .with_chunks(&[40])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();

    let updated = vec![1.5f64; 400];
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("c")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let len_after = std::fs::metadata(&path).unwrap().len();
    assert!(
        len_after <= len_before,
        "a shrinking chunked overwrite must stay in place (no growth): {len_before} -> {len_after}"
    );
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("c")
            .unwrap()
            .read_f64()
            .unwrap(),
        updated
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn real_mat_add_chunked_dataset_preserves_userblock() {
    // Adding a chunked/deflate dataset to a real MATLAB v7.3 file: its chunk
    // index and chunk data addresses are written relative to the 512-byte MATLAB
    // userblock, and the added value plus an untouched original both read back.
    let src = std::path::Path::new("tests/data/matlab/test_string_v73.mat");
    let path = temp_path("hdf5_pure_ub_real_mat_chunk.mat");
    std::fs::copy(src, &path).unwrap();
    let original_userblock = std::fs::read(&path).unwrap()[..UB].to_vec();
    assert_eq!(&original_userblock[..10], b"MATLAB 7.3");

    let before = File::open(&path)
        .unwrap()
        .dataset("string_scalar")
        .unwrap()
        .read_u32()
        .unwrap();

    let added: Vec<f64> = (0..600).map(|i| (i % 17) as f64 * 0.5).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("hdf5_pure_chunk_probe", |b| {
                b.with_f64_data(&added)
                    .with_shape(&[600])
                    .with_chunks(&[64])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("hdf5_pure_chunk_probe")
            .unwrap()
            .read_f64()
            .unwrap(),
        added
    );
    assert_eq!(
        file.dataset("string_scalar").unwrap().read_u32().unwrap(),
        before
    );
    assert_eq!(
        &std::fs::read(&path).unwrap()[..UB],
        &original_userblock[..]
    );
}

#[test]
fn userblock_extensible_array_add_and_overwrite_roundtrip() {
    // An unlimited-maxshape dataset uses the extensible-array (v4 type 4) chunk
    // index, which embeds many internal addresses (EA header, index block,
    // secondary/data blocks) — all built off the planner base. This exercises that
    // index type's base arithmetic on a userblock file, for both add and a
    // relocating overwrite.
    let path = temp_path("hdf5_pure_ub_ea.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("keep").with_i32_data(&[7, 8, 9]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let added: Vec<f64> = (0..500).map(|i| (i as f64).sin() * 1e3).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("ea", |b| {
                b.with_f64_data(&added)
                    .with_shape(&[500])
                    .with_chunks(&[40])
                    .with_maxshape(&[u64::MAX])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("ea")
            .unwrap()
            .read_f64()
            .unwrap(),
        added
    );

    // Relocating overwrite of the extensible-array dataset (different compressed
    // sizes), which rebuilds the EA index + chunk blob and reclaims the old storage.
    let updated: Vec<f64> = (0..500).map(|i| (i as f64) * 0.001).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("ea")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("ea").unwrap().read_f64().unwrap(), updated);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_single_chunk_index_add_and_overwrite() {
    // A dataset whose shape equals its chunk shape stores a single chunk (v4 type 1
    // single-chunk index), whose address lives in the data-layout message rather
    // than a separate index structure. Exercise add + overwrite on a userblock file.
    let path = temp_path("hdf5_pure_ub_single_chunk.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let added: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("sc", |b| {
                b.with_f64_data(&added)
                    .with_shape(&[50])
                    .with_chunks(&[50])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("sc")
            .unwrap()
            .read_f64()
            .unwrap(),
        added
    );

    let updated: Vec<f64> = (0..50).map(|i| (i as f64).cos()).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("sc")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("sc").unwrap().read_f64().unwrap(), updated);
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_chunked_relocating_overwrite_roundtrip() {
    // A deflate dataset whose new contents compress to different per-chunk sizes
    // cannot re-fill its slots, so the overwrite rebuilds and relocates the chunk
    // storage and reclaims the old blob. The new value must read back.
    let path = temp_path("hdf5_pure_ub_chunk_reloc.h5");
    // Highly compressible original (all equal) vs. a varied replacement: their
    // compressed sizes differ, forcing the relocating path.
    let original = vec![7.0f64; 500];
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("keep").with_i32_data(&[11, 22, 33]);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[500])
        .with_chunks(&[50])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let updated: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("c")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().read_f64().unwrap(), updated);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![11, 22, 33]
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_chunked_add_reuses_a_freed_chunk_hole() {
    // A chunked dataset's data region is placed into a freed hole when one fits
    // (issue #261), and on a userblock file the address it is *built* for is the
    // stored (base-relative) one while the address it is *written* at is absolute.
    // Confusing the two is a 512-byte error that no size assertion would notice:
    // the file stays exactly as small, and only reading the data back through the
    // index catches it. So delete a chunked dataset, write another into the hole
    // it left, and read every chunk of it.
    let path = temp_path("hdf5_pure_ub_chunk_hole.h5");
    let victim: Vec<f64> = (0..2048).map(|i| (i % 11) as f64).collect();
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("victim")
        .with_f64_data(&victim)
        .with_shape(&[2048])
        .with_chunks(&[256]);
    // Above the victim, so the delete leaves an interior hole rather than a
    // trailing run the commit would truncate away.
    b.create_dataset("ceiling").with_i32_data(&[11, 22, 33]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();
    let len_start = std::fs::metadata(&path).unwrap().len();

    let replacement: Vec<f64> = (0..2048).map(|i| (i as f64) * 0.25 - 3.0).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("victim").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("fresh", |b| {
                b.with_f64_data(&replacement)
                    .with_shape(&[2048])
                    .with_chunks(&[256]);
            })
            .unwrap();
        s.commit().unwrap();
    }
    let len_end = std::fs::metadata(&path).unwrap().len();
    assert!(
        len_end < len_start + (replacement.len() * 8) as u64 / 10,
        "the fresh chunk blob should land in the freed hole \
         (start={len_start}, end={len_end})"
    );

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("fresh").unwrap().read_f64().unwrap(),
        replacement
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().read_i32().unwrap(),
        vec![11, 22, 33]
    );
    assert!(file.dataset("victim").is_err());
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_chunked_overwrite_reuses_reclaimed_space() {
    // Reclaim-correctness check (the over-reclaim tripwire). A relocating chunked
    // overwrite frees the old chunk storage into the session free list; a later
    // commit in the same session then allocates a small *contiguous* dataset into
    // that freed region. If reclaim had freed a live span (a base-address
    // mistake), the reuse would write over it; every dataset is read back here and
    // the C-library crosscheck confirms it independently.
    // `userblock_chunked_add_reuses_a_freed_chunk_hole` covers the same ground for
    // a chunk blob placed into the hole, where the base arithmetic runs on the
    // addresses the blob itself embeds.
    let path = temp_path("hdf5_pure_ub_chunk_reclaim.h5");
    // A moderately-compressible original lays down a sizeable contiguous chunk blob
    // (the hole that will be reclaimed). The high-entropy replacement re-encodes to
    // chunks that no longer fit those slots, forcing the relocating path.
    let original: Vec<f64> = (0..1200).map(|i| (i % 3) as f64).collect();
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("keep")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    b.create_dataset("c")
        .with_f64_data(&original)
        .with_shape(&[1200])
        .with_chunks(&[60])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let updated: Vec<f64> = (0..1200).map(|i| (i as f64).sin() * 1e9).collect();
    let reuse: Vec<f64> = (0..32).map(|i| (i as f64) - 7.0).collect();
    let reuse_data_bytes = (reuse.len() * 8) as u64;
    let len_after_commit1;
    {
        let s = File::open_rw(&path).unwrap();
        // Commit 1: relocate "c", reclaiming its old chunk storage.
        s.dataset("c")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
        len_after_commit1 = std::fs::metadata(&path).unwrap().len();
        // Commit 2: add a small contiguous dataset that fits the reclaimed hole.
        s.root()
            .create_dataset("reuse", |b| {
                b.with_f64_data(&reuse);
            })
            .unwrap();
        s.commit().unwrap();
    }

    // Commit 2 grew the file by less than the reuse dataset's own data bytes, which
    // is only possible if those bytes were written into reclaimed space rather than
    // appended — so the reuse genuinely overwrites the freed region (the tripwire).
    let len_after_commit2 = std::fs::metadata(&path).unwrap().len();
    assert!(
        len_after_commit2 < len_after_commit1 + reuse_data_bytes,
        "commit 2 should reuse reclaimed space, not append the data: {len_after_commit1} -> {len_after_commit2} (reuse data {reuse_data_bytes}B)"
    );

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().read_f64().unwrap(), updated);
    assert_eq!(file.dataset("reuse").unwrap().read_f64().unwrap(), reuse);
    assert_eq!(
        file.dataset("keep").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_userblock_unchanged(&path, &userblock);
}
