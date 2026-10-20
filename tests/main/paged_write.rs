//! Genuine paged file-space allocation (issue #173 Phase 2, B1): hdf5-pure
//! creates page-aligned files whose free space is tracked by per-page-type
//! free-space managers, and reads them back. C-library interop lives in
//! `crates/crosscheck/tests/main/file_space.rs`.

use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::VlenStringReadOptions;
use rstest::rstest;

const PAGE: u64 = 16384;

use test_util::temp;
use test_util::userblock::Userblock;
use test_util_hdf5::file_space;
use test_util_hdf5::paged;

#[rstest]
#[case::before_exact_fill(512, 0, 1, 17, 512)]
#[case::exact_fill(512, 0, 1, 18, 512)]
#[case::after_exact_fill(512, 0, 1, 19, 512)]
#[case::metadata_only(512, 0, 1, 18, 0)]
#[case::raw_manager_fills_metadata(512, 0, 1, 18, 511)]
#[case::large_manager_fills_metadata(512, 0, 1, 18, 513)]
#[case::raw_and_metadata_managers_fill_page(512, 0, 1, 412, 511)]
#[case::large_and_metadata_managers_fill_page(512, 0, 1, 412, 513)]
#[case::userblock(512, 4096, 1, 18, 512)]
#[case::threshold(512, 0, 4097, 18, 512)]
#[case::default_page(4096, 4096, 1, 3601, 4096)]
#[case::larger_page(8192, 8192, 1, 7697, 8192)]
fn created_page_metadata_tail_is_owned(
    #[case] page: u64,
    #[case] base: u64,
    #[case] threshold: u64,
    #[case] attribute_len: usize,
    #[case] raw_len: usize,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exact_fill.h5");
    paged::write_metadata_tail(&path, page, base, threshold, attribute_len, raw_len);
    let userblock = Userblock::read(&path, usize::try_from(base).unwrap());
    paged::assert_consistent(&path, page);
    if raw_len != 0 {
        paged::assert_pages_homogeneous(&path, page, &["d"]);
    }
    let exact_fill = [(512, 18), (4096, 3601), (8192, 7697)].contains(&(page, attribute_len))
        && u64::try_from(raw_len).unwrap() == page;
    if exact_fill || (page == 512 && attribute_len == 412 && (raw_len == 511 || raw_len == 513)) {
        assert_eq!(file_space::serialized_manager_padding(&path), 1);
    }
    if attribute_len == 18 && (raw_len == 511 || raw_len == 513) {
        assert_eq!(
            File::open(&path)
                .unwrap()
                .file_space_info()
                .unwrap()
                .manager_addrs
                .first(),
            Some(&hdf5_pure::DecodedStoredAddress::Undefined),
        );
    }
    let file = File::open_rw(&path).unwrap();
    let initial = file.space_accounting().unwrap().reusable_free_space;
    if exact_fill {
        assert_eq!(initial, vec![(base + page + 1, page - 1)]);
    }
    let unclassified = if raw_len == 513 {
        vec![(file_space::raw_extents(&file, "d")[0].end, 511)]
    } else {
        Vec::new()
    };
    assert_eq!(
        file.dataset("d").unwrap().read_u8().unwrap(),
        vec![7; raw_len]
    );
    assert_eq!(
        file.root().attrs().unwrap()["fill"],
        AttrValue::U8Array(vec![0; attribute_len])
    );
    file_space::assert_space_ownership(&file, &unclassified);
    file.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.space_accounting().unwrap().reusable_free_space,
        initial
    );
    file.root()
        .create_dataset("added", |builder| {
            builder.with_u8_data(&[9; 32]);
        })
        .unwrap();
    file.commit().unwrap();
    if exact_fill && threshold == 1 {
        assert!(
            file.__space_ownership()
                .unwrap()
                .live_metadata_space
                .iter()
                .any(|&(at, len)| {
                    initial
                        .iter()
                        .any(|&(free, size)| free <= at && at + len <= free + size)
                })
        );
    }
    let unclassified = if threshold > page {
        vec![(48, 136), (255, 258)]
    } else {
        unclassified
    };
    file_space::assert_space_ownership(&file, &unclassified);
    let reusable = file.space_accounting().unwrap().reusable_free_space;
    assert_eq!(
        file.dataset("d").unwrap().read_u8().unwrap(),
        vec![7; raw_len]
    );
    assert_eq!(
        file.dataset("added").unwrap().read_u8().unwrap(),
        vec![9; 32]
    );
    file.close().unwrap();
    let reopened = File::open_rw(&path).unwrap();
    assert_eq!(
        reopened.space_accounting().unwrap().reusable_free_space,
        reusable
    );
    assert_eq!(
        reopened.dataset("d").unwrap().read_u8().unwrap(),
        vec![7; raw_len]
    );
    assert_eq!(
        reopened.dataset("added").unwrap().read_u8().unwrap(),
        vec![9; 32]
    );
    file_space::assert_space_ownership(&reopened, &unclassified);
    reopened.close().unwrap();
    paged::assert_consistent(&path, page);
    if raw_len != 0 {
        paged::assert_pages_homogeneous(&path, page, &["d", "added"]);
    }
    userblock.assert_unchanged(&path);
}

/// A persisting paged file with small and large datasets round-trips through
/// hdf5-pure: the EOA is page-aligned, the free space is tracked in managers,
/// and every dataset reads back.
#[test]
fn paged_persist_roundtrip() {
    let path = temp::repo_temp_path("pure_paged_b1.h5");
    paged::write_small_large_i32(&path, PAGE);
    let small_a: Vec<i32> = (0..100).collect();
    let small_b: Vec<i32> = (0..400).collect();
    let big: Vec<i32> = (0..5000).collect();

    paged::assert_consistent(&path, PAGE);

    let f = File::open(&path).unwrap();
    assert!(
        !f.persisted_free_space().unwrap().is_empty(),
        "paged persist tracks page-tail free space"
    );

    // Every dataset reads back.
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), small_a);
    assert_eq!(f.dataset("b").unwrap().read_i32().unwrap(), small_b);
    assert_eq!(f.dataset("big").unwrap().read_i32().unwrap(), big);
}

/// A non-persisting paged file is still page-aligned, but records no managers.
#[test]
fn paged_non_persist_is_aligned_without_managers() {
    let path = temp::repo_temp_path("pure_paged_b1_nopersist.h5");
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
    assert!(f.persisted_free_space().unwrap().is_empty());
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), data);
}

/// A metadata-only paged file (no datasets) is one metadata page, page-aligned.
#[test]
fn paged_metadata_only() {
    let path = temp::repo_temp_path("pure_paged_b1_meta.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(PAGE);
    b.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64 % PAGE, 0);
    let f = File::open(&path).unwrap();
    let info = f.file_space_info().unwrap();
    assert!(info.persist);
    let Some(hdf5_pure::DecodedStoredAddress::Defined(eoa)) = info.eoa_pre_fsm else {
        panic!("expected Defined eoa_pre_fsm");
    };
    assert_eq!(eoa.get() % PAGE, 0);
}

/// A paged file with chunked datasets (small + a large compressed one) round-
/// trips through hdf5-pure. Chunked data carries its chunk index inline, so this
/// exercises a large page-aligned chunked run in the raw region.
#[test]
fn paged_chunked_roundtrip() {
    let path = temp::repo_temp_path("pure_paged_b1_chunked.h5");
    paged::write_chunked_f64(&path, PAGE);
    let small: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let big: Vec<f64> = (0..8000).map(|i| i as f64 * 0.5).collect();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64 % PAGE, 0);

    let f = File::open(&path).unwrap();
    let info = f.file_space_info().unwrap();
    assert!(info.persist);
    let Some(hdf5_pure::DecodedStoredAddress::Defined(eoa)) = info.eoa_pre_fsm else {
        panic!("expected Defined eoa_pre_fsm");
    };
    assert_eq!(eoa.get() % PAGE, 0);
    assert_eq!(f.dataset("s").unwrap().read_f64().unwrap(), small);
    assert_eq!(f.dataset("big").unwrap().read_f64().unwrap(), big);
}

/// A large block that is an exact multiple of the page size leaves no fragment
/// (the generic-large manager stays empty).
#[test]
fn paged_exact_page_multiple_large_block() {
    let path = temp::repo_temp_path("pure_paged_b1_exact.h5");
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
    assert_eq!(
        f.file_space_info().unwrap().manager_addrs[6],
        hdf5_pure::DecodedStoredAddress::Undefined
    );
}

/// A paged file that exercises the metadata-heavy paths: a root attribute, a
/// group with a dataset, a variable-length string dataset (its element heap
/// references must be patched to the global-heap address in the raw region), and
/// a dataset attribute. All must round-trip through hdf5-pure.
#[test]
fn paged_groups_attrs_vlen() {
    let path = temp::repo_temp_path("pure_paged_b1_meta_heavy.h5");
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

#[rstest]
#[case::below_the_minimum(64)]
#[case::above_the_maximum((1 << 30) + 1)]
#[case::a_power_of_two_above_the_maximum(2 << 30)]
fn the_writer_rejects_a_paged_page_size_outside_512_bytes_to_1_gib(#[case] page_size: u64) {
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(page_size);

    let err = b.finish().unwrap_err();
    let Error::Format(FormatError::InvalidFileSpacePageSize(rejected)) = err else {
        panic!("expected InvalidFileSpacePageSize, got {err:?}");
    };
    assert_eq!(rejected, page_size);
}
