//! Page-homogeneity checking for genuine paged files (`FileSpaceStrategy::Page`).

use std::path::Path;

use hdf5_pure::{File, FileBuilder, FileSpaceStrategy, Layout};
use test_util::{
    btree_v2, bytes, fractal_heap, free_space, global_heap, local_heap, object_header, symbol_table,
};

use crate::dataset::Unlimited;

/// Signatures that must never appear in a page holding raw dataset bytes: object
/// headers and their continuations, the global heap, the free-space managers, the
/// fractal heap that backs dense attributes, and the v2 B-tree and v1
/// symbol-table/local-heap structures. Each is something this crate's editor
/// places through a *metadata* allocation, so finding one in a raw page means a
/// metadata allocation landed there.
///
/// Chunk-index signatures (`FAHD`/`FADB`, `EAHD`/`EAIB`/`EASB`/`EADB`) are
/// deliberately absent from this list. A chunk index is metadata by the format's
/// taxonomy but every writer in this crate emits it in the same run as the chunk
/// data it indexes, so it legitimately shares a raw page; the reclaim side tags it
/// raw to match. `TREE` is excluded for the same reason — a version 1 chunk index
/// uses it too, so its presence is ambiguous.
pub const METADATA_SIGNATURES: &[&[u8; 4]] = &[
    object_header::v2::SIGNATURE,
    object_header::v2::CONTINUATION_SIGNATURE,
    global_heap::SIGNATURE,
    free_space::SIGNATURE,
    free_space::SECTIONS_SIGNATURE,
    fractal_heap::SIGNATURE,
    fractal_heap::DIRECT_BLOCK_SIGNATURE,
    fractal_heap::INDIRECT_BLOCK_SIGNATURE,
    btree_v2::SIGNATURE,
    btree_v2::INTERNAL_SIGNATURE,
    btree_v2::LEAF_SIGNATURE,
    symbol_table::SIGNATURE,
    local_heap::SIGNATURE,
];

/// Writes a persisting paged fixture with small and large contiguous `i32` datasets.
///
/// The file contains `a = 0..100`, `b = 0..400`, and `big = 0..5000`. The `big` dataset
/// is large enough to exercise a dedicated raw run for the page sizes used by the tests.
pub fn write_small_large_i32(path: &Path, page_size: u64) {
    let small_a: Vec<i32> = (0..100).collect();
    let small_b: Vec<i32> = (0..400).collect();
    let big: Vec<i32> = (0..5000).collect();

    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_i32_data(&small_a);
    builder.create_dataset("b").with_i32_data(&small_b);
    builder.create_dataset("big").with_i32_data(&big);
    configure(&mut builder, page_size, true);
    builder.write(path).unwrap();
}

/// Writes a persisting paged fixture with small and large chunked `f64` datasets.
///
/// `s` uses 16-element chunks. `big` uses 1000-element chunks with shuffle and deflate,
/// so the fixture exercises compressed raw pages and their chunk index together.
pub fn write_chunked_f64(path: &Path, page_size: u64) {
    let small: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let big: Vec<f64> = (0..8000).map(|i| i as f64 * 0.5).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("s")
        .with_f64_data(&small)
        .with_shape(&[64])
        .with_chunks(&[16]);
    builder
        .create_dataset("big")
        .with_f64_data(&big)
        .with_shape(&[8000])
        .with_chunks(&[1000])
        .with_shuffle()
        .with_deflate(6);
    configure(&mut builder, page_size, true);
    builder.write(path).unwrap();
}

/// Writes a paged unlimited rank-1 `i32` dataset `d` seeded with `0..len`.
pub fn write_unlimited_i32(path: &Path, page_size: u64, persist: bool, len: i32, chunk: u64) {
    let data: Vec<i32> = (0..len).collect();
    let mut builder = FileBuilder::new();
    Unlimited::new("d", &data, chunk).add_to(&mut builder);
    configure(&mut builder, page_size, persist);
    builder.write(path).unwrap();
}

/// Writes a paged contiguous rank-1 `i32` dataset `d` seeded with `0..len`.
pub fn write_contiguous_i32(path: &Path, page_size: u64, persist: bool, len: i32) {
    let data: Vec<i32> = (0..len).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[len as u64]);
    configure(&mut builder, page_size, persist);
    builder.write(path).unwrap();
}

/// Asserts the on-disk invariants of a paged file with persisted free-space managers.
///
/// The file length and end-of-allocation are whole pages, the recorded page size matches
/// `page_size`, and persisted free sections are non-overlapping and within the file.
pub fn assert_consistent(path: &Path, page_size: u64) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        bytes.len() as u64 % page_size,
        0,
        "file is a whole number of pages"
    );

    let file = File::open(path).unwrap();
    assert_eq!(file.file_space_strategy(), Some(FileSpaceStrategy::Page));
    let info = file.file_space_info().expect("records a strategy");
    assert!(info.persist, "still persisting");
    assert_eq!(info.page_size.get(), page_size);
    assert_eq!(info.eoa_pre_fsm % page_size, 0, "EOA page-aligned");
    assert_eq!(info.eoa_pre_fsm, bytes.len() as u64, "EOA == file size");

    let mut free = file.persisted_free_space().unwrap();
    free.sort_by_key(|&(address, _)| address);
    let mut previous_end = 0u64;
    for (address, length) in free {
        assert!(address >= previous_end, "sections do not overlap");
        assert!(
            address + length <= bytes.len() as u64,
            "section within the file"
        );
        previous_end = address + length;
    }
}

/// Every page of `path` holding raw bytes of any of `datasets` must hold *only*
/// raw bytes.
///
/// A paged file never mixes metadata and raw data within one page — that is the
/// invariant page-typed allocation exists to maintain, and the one thing the C
/// library cannot report on, since it reads such a file happily either way. Raw
/// extents come from the public layout introspection; a page that overlaps one
/// must contain none of [`METADATA_SIGNATURES`].
pub fn assert_pages_homogeneous(path: &Path, page: u64, datasets: &[&str]) {
    let bytes = std::fs::read(path).unwrap();
    let f = File::open(path).unwrap();
    let mut raw: Vec<(u64, u64)> = Vec::new();
    for name in datasets {
        let ds = f.dataset(name).unwrap();
        match ds.layout().unwrap() {
            Layout::Contiguous {
                address: Some(addr),
                size,
            } => raw.push((addr, size)),
            Layout::Chunked { .. } => {
                for c in ds.chunks().unwrap() {
                    raw.push((c.address, c.storage_size));
                }
            }
            // Compact data lives inside the object header (metadata), and an
            // unallocated contiguous dataset owns no bytes at all.
            _ => {}
        }
    }
    assert!(!raw.is_empty(), "expected at least one raw extent to check");
    drop(f);

    let mut raw_pages: Vec<u64> = Vec::new();
    for (addr, size) in raw {
        if size == 0 {
            continue;
        }
        let first = addr / page;
        let last = (addr + size - 1) / page;
        for p in first..=last {
            raw_pages.push(p);
        }
    }
    raw_pages.sort_unstable();
    raw_pages.dedup();

    for p in raw_pages {
        let start = (p * page) as usize;
        let end = ((p + 1) * page).min(bytes.len() as u64) as usize;
        let window = &bytes[start..end];
        for sig in METADATA_SIGNATURES {
            assert!(
                bytes::find_signature(window, sig).is_none(),
                "page {p} holds raw data and the {} signature: a metadata \
                 allocation landed in a raw page",
                String::from_utf8_lossy(*sig)
            );
        }
    }
}

fn configure(builder: &mut FileBuilder, page_size: u64, persist: bool) {
    builder
        .with_file_space_strategy(FileSpaceStrategy::Page, persist, 0)
        .with_file_space_page_size(page_size);
}
