//! Genuine paged file-space allocation (issue #173 Phase 2, B1): hdf5-pure
//! creates page-aligned files whose free space is tracked by per-page-type
//! free-space managers, and reads them back. C-library interop lives in
//! `crates/crosscheck/tests/file_space.rs`.

use hdf5_pure::{AttrValue, File, FileBuilder, FileSpaceStrategy, VlenStringReadOptions};

const PAGE: u64 = 16384;

use test_util::temp;

/// A fixture under the repository's gitignored `tmp/`, in a directory of its own
/// so two concurrent runs of this binary cannot collide on the name (issue #334).
fn tmp(name: &str) -> temp::TempPath {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tmp");
    std::fs::create_dir_all(&p).expect("create the repository's tmp directory");
    temp::temp_path_in(&p, name)
}

/// A persisting paged file with small and large datasets round-trips through
/// hdf5-pure: the EOA is page-aligned, the free space is tracked in managers,
/// and every dataset reads back.
#[test]
fn paged_persist_roundtrip() {
    let path = tmp("pure_paged_b1.h5");
    let small_a: Vec<i32> = (0..100).collect(); // 400 bytes
    let small_b: Vec<i32> = (0..400).collect(); // 1600 bytes
    let big: Vec<i32> = (0..5000).collect(); // 20000 bytes >= page -> large run

    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&small_a);
    b.create_dataset("b").with_i32_data(&small_b);
    b.create_dataset("big").with_i32_data(&big);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        bytes.len() as u64 % PAGE,
        0,
        "file is a whole number of pages"
    );

    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::Page));
    let info = f.file_space_info().expect("records a strategy");
    assert!(info.persist);
    assert_eq!(info.page_size, PAGE);
    assert_eq!(info.eoa_pre_fsm % PAGE, 0, "EOA recorded page-aligned");
    assert_eq!(info.eoa_pre_fsm, bytes.len() as u64, "EOA == file size");

    // Free space is tracked (page tails), non-overlapping and within the file.
    let free = f.persisted_free_space();
    assert!(
        !free.is_empty(),
        "paged persist tracks page-tail free space"
    );
    let mut sorted = free.clone();
    sorted.sort_by_key(|&(a, _)| a);
    let mut prev_end = 0u64;
    for (addr, len) in &sorted {
        assert!(*addr >= prev_end, "sections do not overlap");
        assert!(addr + len <= bytes.len() as u64, "section within the file");
        prev_end = addr + len;
    }

    // Every dataset reads back.
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), small_a);
    assert_eq!(f.dataset("b").unwrap().read_i32().unwrap(), small_b);
    assert_eq!(f.dataset("big").unwrap().read_i32().unwrap(), big);
}

/// A non-persisting paged file is still page-aligned, but records no managers.
#[test]
fn paged_non_persist_is_aligned_without_managers() {
    let path = tmp("pure_paged_b1_nopersist.h5");
    let data: Vec<i32> = (0..300).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&data);
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64 % PAGE, 0);

    let f = File::open(&path).unwrap();
    let info = f.file_space_info().unwrap();
    assert!(!info.persist);
    assert!(
        info.manager_addrs.is_empty(),
        "non-persist records no managers"
    );
    assert!(f.persisted_free_space().is_empty());
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), data);
}

/// A metadata-only paged file (no datasets) is one metadata page, page-aligned.
#[test]
fn paged_metadata_only() {
    let path = tmp("pure_paged_b1_meta.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64 % PAGE, 0);
    let f = File::open(&path).unwrap();
    let info = f.file_space_info().unwrap();
    assert!(info.persist);
    assert_eq!(info.eoa_pre_fsm % PAGE, 0);
}

/// A paged file with chunked datasets (small + a large compressed one) round-
/// trips through hdf5-pure. Chunked data carries its chunk index inline, so this
/// exercises a large page-aligned chunked run in the raw region.
#[test]
fn paged_chunked_roundtrip() {
    let path = tmp("pure_paged_b1_chunked.h5");
    let small: Vec<f64> = (0..64).map(|i| i as f64).collect(); // 512 bytes
    let big: Vec<f64> = (0..8000).map(|i| i as f64 * 0.5).collect(); // 64000 bytes >= page

    let mut b = FileBuilder::new();
    b.create_dataset("s")
        .with_f64_data(&small)
        .with_shape(&[64])
        .with_chunks(&[16]);
    {
        let ds = b
            .create_dataset("big")
            .with_f64_data(&big)
            .with_shape(&[8000])
            .with_chunks(&[1000]);
        ds.with_shuffle().with_deflate(6);
    }
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64 % PAGE, 0);

    let f = File::open(&path).unwrap();
    let info = f.file_space_info().unwrap();
    assert!(info.persist);
    assert_eq!(info.eoa_pre_fsm % PAGE, 0);
    assert_eq!(f.dataset("s").unwrap().read_f64().unwrap(), small);
    assert_eq!(f.dataset("big").unwrap().read_f64().unwrap(), big);
}

/// A large block that is an exact multiple of the page size leaves no fragment
/// (the generic-large manager stays empty).
#[test]
fn paged_exact_page_multiple_large_block() {
    let path = tmp("pure_paged_b1_exact.h5");
    // 4096 i32 = 16384 bytes = exactly one page.
    let big: Vec<i32> = (0..4096).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("exact").with_i32_data(&big);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64 % PAGE, 0);
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("exact").unwrap().read_i32().unwrap(), big);
    // No large fragment: the generic-large manager (slot 6) is undefined.
    assert_eq!(f.file_space_info().unwrap().manager_addrs[6], u64::MAX);
}

/// A paged file that exercises the metadata-heavy paths: a root attribute, a
/// group with a dataset, a variable-length string dataset (its element heap
/// references must be patched to the global-heap address in the raw region), and
/// a dataset attribute. All must round-trip through hdf5-pure.
#[test]
fn paged_groups_attrs_vlen() {
    let path = tmp("pure_paged_b1_meta_heavy.h5");
    let mut b = FileBuilder::new();
    b.set_attr("title", AttrValue::String("paged file".into()));
    {
        let ds = b.create_dataset("temps").with_f64_data(&[1.0, 2.0, 3.0]);
        ds.set_attr("unit", AttrValue::String("celsius".into()));
    }
    b.create_dataset("labels")
        .with_vlen_strings(&["alpha", "beta", "gamma", "delta"]);
    let mut grp = b.create_group("sensors");
    grp.create_dataset("readings").with_i32_data(&[10, 20, 30]);
    b.add_group(grp.finish());
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64 % PAGE, 0);

    let f = File::open(&path).unwrap();
    assert!(f.file_space_info().unwrap().persist);
    assert_eq!(
        f.dataset("temps").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("sensors/readings").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    let labels = f
        .dataset("labels")
        .unwrap()
        .read_vlen_strings(VlenStringReadOptions::default())
        .unwrap();
    assert_eq!(labels, vec!["alpha", "beta", "gamma", "delta"]);
}

/// A page size that is not a power of two >= 512 is rejected at build time.
#[test]
fn invalid_page_size_rejected() {
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(3000); // not a power of two
    assert!(b.finish().is_err());
}
