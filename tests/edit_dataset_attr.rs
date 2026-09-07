//! Pure-Rust tests for `Dataset::set_attr` / `remove_attr`
//! (issue #146): compact dataset-attribute add / update / remove, staged and
//! applied on commit by relocating the dataset's object header while preserving its
//! data and chunk index. C-library interop — undefined-`AttributeInfo` acceptance
//! (dataset and group) and the single-hard-link refusal — lives in
//! `crates/crosscheck/tests/edit.rs`; edits that land in dense (fractal-heap) storage live in
//! `edit_dense_attr.rs`.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError};
use tempfile::tempdir;

use test_util::heap::has_fractal_heap;

fn build_contig(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
    b.write(path).unwrap();
}

fn build_chunked(path: &std::path::Path, n: i32, chunk: u64) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..n).collect::<Vec<_>>())
        .with_shape(&[n as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    b.write(path).unwrap();
}

#[test]
fn set_fixed_attrs_preserves_data() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("count", AttrValue::I64(42))
            .unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("unit", AttrValue::String("m/s".into()))
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), vec![1, 2, 3, 4]); // data untouched
    let attrs = d.attrs().unwrap();
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(42)));
    assert_eq!(attrs.get("unit"), Some(&AttrValue::String("m/s".into())));
}

#[test]
fn update_and_remove_attr() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("a", AttrValue::I64(1))
            .unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("b", AttrValue::I64(2))
            .unwrap();
        s.commit().unwrap();
    }
    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("a", AttrValue::I64(99))
            .unwrap(); // replace
        s.dataset("d").unwrap().remove_attr("b").unwrap(); // remove
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("a"), Some(&AttrValue::I64(99)));
    assert!(!attrs.contains_key("b"));
}

#[test]
fn set_vlstring_attr() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr(
                "labels",
                AttrValue::VarLenAsciiCharArray(vec!["alpha".into(), "beta".into()]),
            )
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("labels"),
        Some(&AttrValue::VarLenAsciiCharArray(vec![
            "alpha".into(),
            "beta".into()
        ]))
    );
}

#[test]
fn attr_edit_on_chunked_preserves_data_and_index() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_chunked(&p, 8, 4);

    {
        let s = File::open_rw(&p).unwrap();
        // Immediate append grows the Extensible Array in place.
        s.dataset("d").unwrap().append(&[8, 9, 10, 11]).unwrap(); // 0..12
        // A staged attribute edit relocates the header at commit (preserving the
        // grown index and data). The cache is invalidated at commit entry.
        s.dataset("d")
            .unwrap()
            .set_attr("tag", AttrValue::I64(5))
            .unwrap();
        s.commit().unwrap();
        // Append again: re-locates at the dataset's new header address.
        s.dataset("d").unwrap().append(&[12, 13]).unwrap(); // 0..14
    }

    let f = File::open(&p).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), (0..14).collect::<Vec<_>>());
    assert_eq!(d.attrs().unwrap().get("tag"), Some(&AttrValue::I64(5)));
}

#[test]
fn guard_refuses_append_after_staged_dataset_attr() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_chunked(&p, 4, 4);

    let s = File::open_rw(&p).unwrap();
    s.dataset("d").unwrap().append(&[4, 5, 6, 7]).unwrap(); // fine, nothing staged
    s.dataset("d")
        .unwrap()
        .set_attr("tag", AttrValue::I64(1))
        .unwrap(); // now stage an attr edit
    // A second in-place append is refused: commit would relocate the header this
    // append planned against.
    let err = s.dataset("d").unwrap().append(&[8]).unwrap_err();
    assert!(matches!(err, Error::AppendInPlaceUnsupported(_)));
}

#[test]
fn attr_edit_plus_append_dataset_same_commit_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_chunked(&p, 8, 4);

    let s = File::open_rw(&p).unwrap();
    s.dataset("d")
        .unwrap()
        .set_attr("tag", AttrValue::I64(1))
        .unwrap();
    s.dataset("d")
        .unwrap()
        .append_staged(|b| {
            b.append_i32(&[8, 9, 10, 11]);
        })
        .unwrap();
    let err = s.commit().unwrap_err();
    assert!(matches!(err, Error::EditUnsupported(_)));
}

#[test]
fn the_ninth_attribute_moves_the_set_out_of_the_header() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("d").with_i32_data(&[1, 2, 3, 4]);
        for i in 0..7i64 {
            ds.set_attr(&format!("a{i}"), AttrValue::I64(i));
        }
        b.write(&p).unwrap();
    }

    // Eight is what an object header holds compactly, so this edit stays in it.
    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("a7", AttrValue::I64(7))
            .unwrap();
        s.commit().unwrap();
    }
    assert!(
        !has_fractal_heap(&std::fs::read(&p).unwrap()),
        "eight attributes still fit the object header",
    );

    // The ninth does not, so the whole set moves to a fractal heap rather than
    // being refused (issue #102). The values are checked either side of the move.
    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("overflow", AttrValue::I64(9))
            .unwrap();
        s.commit().unwrap();
    }
    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "the ninth attribute must move the set into a fractal heap",
    );
    let f = File::open(&p).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), vec![1, 2, 3, 4]);
    let attrs = d.attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    for i in 0..8i64 {
        assert_eq!(attrs.get(&format!("a{i}")), Some(&AttrValue::I64(i)));
    }
    assert_eq!(attrs.get("overflow"), Some(&AttrValue::I64(9)));
}

#[test]
fn set_attr_on_missing_dataset_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    build_contig(&p);

    let s = File::open_rw(&p).unwrap();
    // Resolving the handle is where a missing dataset is now caught, rather than
    // at commit — the edit never gets staged at all.
    let err = s.dataset("nope").unwrap_err();
    assert!(
        matches!(err, Error::Format(FormatError::PathNotFound(_))),
        "expected PathNotFound, got {err:?}"
    );
    assert!(!s.has_staged_edits());
    s.commit().unwrap();
}
