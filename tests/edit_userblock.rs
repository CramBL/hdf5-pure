//! In-place editing of files that carry a userblock (non-zero base address) —
//! the userblock slice of issue #104.
//!
//! Every MATLAB v7.3 `.mat` file begins with a 512-byte userblock, so the HDF5
//! superblock sits at byte 512 and every stored address is relative to that
//! base. These tests exercise the editor on such files — a synthetic one built
//! by this crate and a real `.mat` fixture — and verify that the userblock bytes
//! survive an edit byte-for-byte.
//!
//! This slice supports contiguous value overwrites, additions of contiguous and
//! chunked/filtered datasets, overwrites of chunked datasets (in place when the
//! re-encoded chunks fit their slots, otherwise rebuilt and relocated with the
//! old storage reclaimed), group creation, and compact group attributes. The
//! delete, copy, and relocating contiguous/compact overwrite paths are covered in
//! `edit_userblock_followups.rs` and `crates/crosscheck/tests/edit_userblock.rs`. The one
//! userblock-specific operation still refused — cross-file copy from a userblock
//! *source* — is covered below; a refusal never corrupts the file.

use hdf5_pure::{AttrValue, File, FileBuilder, Object};

use temp::temp_path;
use test_util::temp;

const UB: usize = 512;

const MARKER: &[u8] = b"USERBLOCK-MARKER-0104";

/// Build a userblock file with two root datasets and a nested group+dataset,
/// stamping a recognizable marker across the userblock region. Returns the
/// 512-byte userblock as written, for later byte-for-byte comparison.
fn build_userblock_file(path: &std::path::Path) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("alpha")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.create_dataset("beta").with_i32_data(&[10, 20, 30]);
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f64_data(&[7.5, 8.5]);
    b.add_group(g.finish());
    let mut bytes = b.finish().unwrap();

    // The userblock region [0..512] is zero-filled by the writer; stamp a marker
    // at the start and a sentinel at the last byte to catch any stray write into
    // the userblock during an edit.
    bytes[..MARKER.len()].copy_from_slice(MARKER);
    bytes[UB - 1] = 0xAB;
    let userblock = bytes[..UB].to_vec();
    std::fs::write(path, &bytes).unwrap();
    userblock
}

#[test]
fn synthetic_userblock_file_roundtrip() {
    let path = temp_path("hdf5_pure_ub_roundtrip.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        // Same-length in-place overwrite of an existing contiguous dataset.
        s.dataset("alpha")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[9.0, 8.0, 7.0, 6.0]);
            })
            .unwrap();
        // Add contiguous datasets at the root and inside the nested group.
        s.root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[100, 200]);
            })
            .unwrap();
        s.root()
            .create_dataset("grp/added_inner", |b| {
                b.with_f64_data(&[3.25]);
            })
            .unwrap();
        // Create a new group and set a compact attribute on it, in one batch.
        s.root()
            .create_group_with("newgrp", |g| {
                g.set_attr("tag", AttrValue::AsciiString("v104".into()));
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Overwritten value.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![9.0, 8.0, 7.0, 6.0]
    );
    // Untouched originals.
    assert_eq!(
        file.dataset("beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    // Additions.
    assert_eq!(
        file.dataset("added").unwrap().read_i32().unwrap(),
        vec![100, 200]
    );
    assert_eq!(
        file.dataset("grp/added_inner").unwrap().read_f64().unwrap(),
        vec![3.25]
    );
    // New group and its attribute.
    let mut groups = file.root().groups().unwrap();
    groups.sort();
    assert!(groups.contains(&"newgrp".to_string()));
    let attrs = file.group("newgrp").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("tag"),
        Some(&AttrValue::AsciiString("v104".into()))
    );

    // The userblock bytes are preserved verbatim across the edit.
    let after = std::fs::read(&path).unwrap();
    assert_eq!(
        &after[..UB],
        &userblock[..],
        "userblock bytes changed across the edit"
    );
    assert_eq!(&after[..MARKER.len()], MARKER);
    assert_eq!(after[UB - 1], 0xAB);
}

#[test]
fn userblock_inplace_overwrite_only_takes_fast_path() {
    // A lone same-length overwrite takes the in-place fast path (no header
    // rewrite, no superblock flip); it must work on a userblock file and leave
    // the userblock untouched.
    let path = temp_path("hdf5_pure_ub_inplace_only.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("beta")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[-1, -2, -3]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("beta").unwrap().read_i32().unwrap(),
        vec![-1, -2, -3]
    );
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);
}

/// Cross-file copy *from* a userblock source is still refused (the source-file
/// base-address restriction in `copy_from`); the destination file must be left
/// byte-identical by the refusal. Delete, in-file copy, cross-file copy into a
/// userblock destination, and resizing overwrites are all supported now and are
/// exercised in `edit_userblock_followups.rs` / `crates/crosscheck/tests/edit_userblock.rs`.
#[test]
fn userblock_cross_file_copy_from_userblock_source_is_refused() {
    let src_path = temp_path("hdf5_pure_ub_xcopy_src_refuse.h5");
    let dst_path = temp_path("hdf5_pure_ub_xcopy_dst_refuse.h5");
    build_userblock_file(&src_path);
    build_userblock_file(&dst_path);
    let dst_before = std::fs::read(&dst_path).unwrap();

    let source = File::open(&src_path).unwrap();
    let s = File::open_rw(&dst_path).unwrap();
    // The eager read happens in `copy_from`, so the refusal surfaces there.
    let err = s.copy_from(&source, "alpha", "imported");
    assert!(
        err.is_err(),
        "cross-file copy from a userblock source should be refused"
    );
    drop(s);

    assert_eq!(
        std::fs::read(&dst_path).unwrap(),
        dst_before,
        "a refused cross-file copy must not modify the destination"
    );
}

/// An empty (zero-element) dataset added on a userblock file must round-trip
/// correctly. This is the one test in the suite that can actually
/// discriminate correct behavior from a broken `u64::MAX - base` regression:
/// at `base_address == 0` the empty-dataset sentinel (`u64::MAX`, written
/// with no subtraction) and a wrongly-adjusted sentinel are numerically
/// identical, so only a non-zero base can catch a mistake here.
#[test]
fn userblock_add_empty_dataset_roundtrip() {
    let path = temp_path("hdf5_pure_ub_add_empty.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("empty", |b| {
                b.with_f64_data(&[]).with_shape(&[0, 3]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let empty = file.dataset("empty").unwrap();
    assert_eq!(empty.shape().unwrap(), vec![0, 3]);
    assert_eq!(empty.read_f64().unwrap(), Vec::<f64>::new());
    // Untouched original still reads correctly.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);
}

/// The chunked counterpart of the test above: an empty chunked dataset added on
/// a userblock file, then grown.
///
/// A userblock file is the only backing that forces the *mirrored* engine, and
/// it is the only one where a base-relative address arithmetic slip is visible —
/// at `base_address == 0` a missing or spurious `- base` is numerically
/// invisible. An empty chunked dataset places a chunk index and nothing else, so
/// the index address is the whole of what the layout message has to get right,
/// and the append that follows proves the C library and this crate agree on
/// where it landed.
#[test]
fn userblock_add_empty_chunked_dataset_and_grow_it() {
    let path = temp_path("hdf5_pure_ub_add_empty_chunked.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        let root = s.root();
        root.create_dataset("col", |b| {
            b.with_i64_data(&[])
                .with_shape(&[0])
                .with_maxshape(&[u64::MAX])
                .with_chunks(&[4]);
        })
        .unwrap();
        // The fixed-shape flavor too: it is the one that carries no index at
        // all, so its layout message names the undefined address rather than a
        // base-relative one — the case a blanket `- base` would corrupt.
        root.create_dataset("fixed", |b| {
            b.with_i64_data(&[]).with_shape(&[0]).with_chunks(&[4]);
        })
        .unwrap();
        s.commit().unwrap();
    }

    {
        let file = File::open(&path).unwrap();
        assert_eq!(file.dataset("col").unwrap().shape().unwrap(), vec![0]);
        assert_eq!(
            file.dataset("col").unwrap().read_i64().unwrap(),
            Vec::<i64>::new()
        );
        assert_eq!(
            file.dataset("fixed").unwrap().read_i64().unwrap(),
            Vec::<i64>::new()
        );
    }

    // Grow it across several chunks, so the index the empty dataset placed is
    // read back at its base-relative address and extended.
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("col")
            .unwrap()
            .append_staged(|a| {
                a.append_i64(&(0..10).collect::<Vec<i64>>());
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("col").unwrap().read_i64().unwrap(),
        (0..10).collect::<Vec<i64>>()
    );
    // Untouched original still reads correctly, and the userblock is intact.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);
}

/// A provenance-tagged dataset added on a userblock file must round-trip
/// correctly, exercising the same base-relative address arithmetic as the
/// non-userblock case above but where a `- base` bug would actually be
/// observable.
#[cfg(feature = "provenance")]
#[test]
fn userblock_add_provenance_dataset_roundtrip() {
    use hdf5_pure::VerifyResult;

    let path = temp_path("hdf5_pure_ub_add_provenance.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("sensor", |b| {
                b.with_f64_data(&[1.0, 2.0, 3.0]).with_provenance(
                    "test-suite",
                    "2026-02-19T12:00:00Z",
                    Some("bench"),
                );
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("sensor").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::Ok);
    assert_eq!(
        ds.attrs().unwrap().get("_provenance_creator"),
        Some(&AttrValue::String("test-suite".into()))
    );
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);
}

/// A dataset with a variable-length attribute, added on a userblock file,
/// must round-trip correctly: the engine's `place_vl_collection`
/// `addr - base_address` arithmetic is only actually exercised (as opposed to
/// a no-op at `base == 0`) once `base` is non-zero.
#[test]
fn userblock_add_dataset_with_vlen_attribute_roundtrip() {
    let path = temp_path("hdf5_pure_ub_add_vlen_attr.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("labeled", |b| {
                b.with_i32_data(&[1, 2, 3]).set_attr(
                    "tags",
                    AttrValue::VarLenAsciiCharArray(vec!["one".into(), "two".into()]),
                );
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labeled").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![1, 2, 3]);
    assert_eq!(
        ds.attrs().unwrap().get("tags"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "one".into(),
            "two".into()
        ]))
    );
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);
}

/// A variable-length-string dataset added on a userblock file must round-trip
/// correctly, exercising the same `place_vl_collection` base-relative
/// arithmetic as the attribute case above but for dataset *data* rather than
/// an attribute.
#[test]
fn userblock_add_vlen_string_dataset_roundtrip() {
    let path = temp_path("hdf5_pure_ub_add_vlen_string_ds.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("labels", |b| {
                b.with_vlen_strings(&["alpha", "", "gamma"]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("labels").unwrap();
    assert_eq!(
        ds.read_string().unwrap(),
        vec!["alpha".to_string(), String::new(), "gamma".to_string()]
    );
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);
}

/// An object-reference dataset added on a userblock file must round-trip
/// correctly: `resolve_reference_target`'s `- base` adjustment for a resolved
/// address is otherwise only ever exercised as a no-op at `base == 0`.
#[test]
fn userblock_add_reference_dataset_roundtrip() {
    let path = temp_path("hdf5_pure_ub_add_ref.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["alpha"]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(targets.len(), 1);
    match &targets[0] {
        Object::Dataset(ds) => assert_eq!(
            ds.read_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0],
            "reference should resolve to `alpha`'s original contents"
        ),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    assert_eq!(&std::fs::read(&path).unwrap()[..UB], &userblock[..]);
}

#[test]
fn real_mat_add_dataset_preserves_userblock_and_data() {
    // Copy the fixture so the test never mutates the checked-in file.
    let src = std::path::Path::new("tests/data/matlab/test_string_v73.mat");
    let path = temp_path("hdf5_pure_ub_real_mat.mat");
    std::fs::copy(src, &path).unwrap();

    let original_userblock = std::fs::read(&path).unwrap()[..UB].to_vec();
    assert_eq!(
        &original_userblock[..10],
        b"MATLAB 7.3",
        "fixture is not a MATLAB v7.3 userblock file"
    );

    // Capture an existing variable's bytes before editing.
    let before = File::open(&path)
        .unwrap()
        .dataset("string_scalar")
        .unwrap()
        .read_u32()
        .unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("hdf5_pure_probe", |b| {
                b.with_f64_data(&[42.0, -1.0, 3.5]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // The added dataset reads back correctly.
    assert_eq!(
        file.dataset("hdf5_pure_probe").unwrap().read_f64().unwrap(),
        vec![42.0, -1.0, 3.5]
    );
    // The untouched original still reads identically.
    assert_eq!(
        file.dataset("string_scalar").unwrap().read_u32().unwrap(),
        before
    );
    // The original variables and MATLAB's MCOS groups are still present.
    let datasets = file.root().datasets().unwrap();
    for name in [
        "string_array",
        "string_empty",
        "string_scalar",
        "hdf5_pure_probe",
    ] {
        assert!(
            datasets.contains(&name.to_string()),
            "missing dataset {name}"
        );
    }
    let groups = file.root().groups().unwrap();
    assert!(groups.contains(&"#refs#".to_string()));
    assert!(groups.contains(&"#subsystem#".to_string()));

    // The MATLAB userblock (signature, version, endian indicator) is unchanged.
    let after_userblock = std::fs::read(&path).unwrap()[..UB].to_vec();
    assert_eq!(
        after_userblock, original_userblock,
        "MATLAB userblock changed across the edit"
    );
}
