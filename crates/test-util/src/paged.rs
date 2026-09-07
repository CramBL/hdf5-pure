//! Page-homogeneity checking for genuine paged files (`FileSpaceStrategy::Page`).
//!
//! Included with `#[path = "common/paged.rs"] mod paged;` rather than through
//! `common/mod.rs`, so a binary that wants only this does not also pull in the
//! C-library helpers beside it.
#![allow(dead_code)]

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
    b"OHDR", b"OCHK", b"GCOL", b"FSHD", b"FSSE", b"FRHP", b"FHDB", b"FHIB", b"BTHD", b"BTIN",
    b"BTLF", b"SNOD", b"HEAP",
];

/// Every page of `path` holding raw bytes of any of `datasets` must hold *only*
/// raw bytes.
///
/// A paged file never mixes metadata and raw data within one page — that is the
/// invariant page-typed allocation exists to maintain, and the one thing the C
/// library cannot report on, since it reads such a file happily either way. Raw
/// extents come from the public layout introspection; a page that overlaps one
/// must contain none of [`METADATA_SIGNATURES`].
pub fn assert_pages_homogeneous(path: &std::path::Path, page: u64, datasets: &[&str]) {
    let bytes = std::fs::read(path).unwrap();
    let f = hdf5_pure::File::open(path).unwrap();
    let mut raw: Vec<(u64, u64)> = Vec::new();
    for name in datasets {
        let ds = f.dataset(name).unwrap();
        match ds.layout().unwrap() {
            hdf5_pure::Layout::Contiguous {
                address: Some(addr),
                size,
            } => raw.push((addr, size)),
            hdf5_pure::Layout::Chunked { .. } => {
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
                !window.windows(4).any(|w| w == *sig),
                "page {p} holds raw data and the {} signature: a metadata \
                 allocation landed in a raw page",
                std::str::from_utf8(*sig).unwrap()
            );
        }
    }
}
