// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! A dataset whose element bytes live in *external* files (`H5Pset_external`).
//!
//! Its data-layout message is contiguous with the data address **undefined** —
//! byte-for-byte the encoding a never-written dataset uses — and an External Data
//! Files message (header type 7) beside it names where the bytes actually are.
//! This crate does not follow those files, so "undefined address" must not be
//! read as "stores nothing": a reader that did would answer the fill value for
//! every element of a dataset that holds data, and a rewrite that did would emit
//! a schema-only dataset in place of it — both reporting success.

use hdf5_pure::{File, RepackOptions, repack};
use tempfile::tempdir;

use hdf5_pure_crosscheck::create_v18;

const N: usize = 16;

/// Write `path` holding `/d`, an `N`-element i32 dataset stored in `payload`.
fn write_external_fixture(path: &std::path::Path, payload: &str) {
    let file = create_v18(path);
    file.new_dataset::<i32>()
        .external(payload, 0, N * 4)
        .shape([N])
        .create("d")
        .unwrap()
        .write(&(1..=N as i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// `repack` refuses an externally stored dataset rather than reproducing it
/// without its data.
///
/// The C library reads 1..=16 from the source. Every byte of that lives outside
/// the file, so a rewrite has nothing to copy; the only honest answer is to
/// refuse. Before this was checked, `repack` returned `Ok` and the destination
/// answered the fill value for every element — the failure mode a
/// values-round-trip assertion could not see, because the source's *reader*
/// answered the same zeros. It refuses now, as of #331; repack's own refusal
/// still fires first, so this test does not rest on that.
#[test]
fn repack_refuses_external_data_storage() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let dst = dir.path().join("external_repacked.h5");
    let payload = dir.path().join("repack_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());

    // Ground truth: the C library follows the external file and finds the data.
    {
        let c = hdf5::File::open(&src).unwrap();
        let d = c.dataset("d").unwrap();
        assert_eq!(
            d.read_raw::<i32>().unwrap(),
            (1..=N as i32).collect::<Vec<i32>>(),
            "the source holds real data"
        );
        assert_eq!(d.storage_size(), (N * 4) as u64);
        c.close().unwrap();
    }

    // And the layout message is indistinguishable from a never-written one.
    {
        let f = File::open(&src).unwrap();
        assert!(matches!(
            f.dataset("d").unwrap().layout().unwrap(),
            hdf5_pure::Layout::Contiguous {
                address: None,
                size: 64
            }
        ));
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains('d') && msg.contains("external"),
            "the refusal must name the dataset and the reason: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// Every read path refuses an externally stored dataset rather than answering
/// the fill value for data it never looked at.
///
/// The layout message records a contiguous layout, an extent of 64 bytes, and an
/// **undefined** data address — and a never-written dataset of this shape and
/// type records those same eighteen bytes. Nothing in it says where the elements
/// are, so the External Data Files message beside it is the only thing telling
/// the two apart, and a read that consults the layout alone reports a dataset
/// holding 1..=16 as sixteen zeros, with `Ok` (issue #331).
///
/// The three calls below are the three places that build a read of that layout:
/// `read_raw`, the whole-dataset typed reads, and the windowed reads. Reverting
/// any one of them to the introspection accessor fails this test at that call.
/// The streaming repeat is not a fourth: it pins the refusal *above* the backend
/// dispatch, where one written into the buffered arm alone would pass every
/// assertion before it.
#[test]
fn every_read_path_refuses_external_data_storage() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let payload = dir.path().join("read_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());

    // Ground truth: the data exists, and the C library reads it.
    {
        let c = hdf5::File::open(&src).unwrap();
        assert_eq!(
            c.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            (1..=N as i32).collect::<Vec<i32>>(),
        );
        c.close().unwrap();
    }

    let refusal = |what: &str, err: hdf5_pure::Error| match err {
        hdf5_pure::Error::Format(hdf5_pure::FormatError::UnsupportedExternalStorage) => {}
        other => panic!("{what}: expected UnsupportedExternalStorage, got {other:?}"),
    };

    let f = File::open(&src).unwrap();
    let d = f.dataset("d").unwrap();
    refusal("read_raw", d.read_raw().unwrap_err());
    refusal("read_i32", d.read_i32().unwrap_err());
    refusal("read_raw_rows", d.read_raw_rows(0, 4).unwrap_err());
    drop(f);

    // A streaming file reads through a different backend, and the issue measured
    // it answering the same zeros.
    let s = File::open_streaming(&src).unwrap();
    refusal(
        "streaming read_i32",
        s.dataset("d").unwrap().read_i32().unwrap_err(),
    );
}

/// The refusal is scoped to the data: everything the header records about an
/// externally stored dataset still reads.
///
/// Refusing the metadata too would make the dataset invisible rather than
/// unreadable, and a caller cannot then see *why* it was refused — the layout
/// with no address is the evidence.
#[test]
fn external_data_storage_still_reports_its_metadata() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let payload = dir.path().join("meta_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());

    let f = File::open(&src).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(d.shape().unwrap(), vec![N as u64]);
    assert!(matches!(
        d.layout().unwrap(),
        hdf5_pure::Layout::Contiguous {
            address: None,
            size: 64
        }
    ));
    assert_eq!(d.datatype().unwrap().type_size(), 4);
}

/// No write reaches an externally stored dataset's elements either.
///
/// The elements are not in this file and the edit engine writes only this file,
/// but the layout message it consults records no address — so a staged write
/// took the dataset for never-allocated storage, appended the new bytes to the
/// HDF5 file and pointed the layout at them, reporting `Ok`. That left the file
/// holding two contradictory records of where its data lives: this crate read
/// the new bytes, while the reference library went on reading the external files
/// it had never touched, so the caller's write was visible to neither reader
/// once #331 made the read refuse.
///
/// Attribute edits are deliberately unaffected — they rewrite the object header,
/// not the elements — and replacing the dataset outright still works, which is
/// the way to stop a file depending on external files.
#[test]
fn no_write_reaches_an_external_dataset() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let payload = dir.path().join("write_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());
    let before = std::fs::metadata(&src).unwrap().len();

    let refusal = |what: &str, err: hdf5_pure::Error| match err {
        hdf5_pure::Error::EditUnsupported(msg) => assert!(
            msg.contains("external files"),
            "{what}: the refusal must name external storage: {msg}"
        ),
        other => panic!("{what}: expected EditUnsupported, got {other:?}"),
    };

    let f = File::open_rw(&src).unwrap();
    refusal(
        "write",
        f.dataset("d").unwrap().write(&[100i32; N]).unwrap_err(),
    );
    refusal(
        "write_staged",
        f.dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[100i32; N]);
            })
            .unwrap_err(),
    );
    refusal(
        "append_staged",
        f.dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[100]);
            })
            .unwrap_err(),
    );
    drop(f);

    // Nothing was staged, so nothing was written: a refusal that had already
    // appended the bytes would show here.
    assert_eq!(
        std::fs::metadata(&src).unwrap().len(),
        before,
        "a refused write must not have grown the file"
    );

    // An attribute is a header edit, not a data write, and still commits. It
    // rewrites the header elsewhere, so the file does grow — what must survive
    // is the address-less layout and the External Data Files message with it.
    let f = File::open_rw(&src).unwrap();
    f.dataset("d")
        .unwrap()
        .set_attr("tag", hdf5_pure::AttrValue::I64(1))
        .unwrap();
    f.commit().unwrap();
    drop(f);
    {
        let r = File::open(&src).unwrap();
        let d = r.dataset("d").unwrap();
        assert_eq!(
            d.attrs().unwrap().get("tag"),
            Some(&hdf5_pure::AttrValue::I64(1))
        );
        assert!(matches!(
            d.layout().unwrap(),
            hdf5_pure::Layout::Contiguous { address: None, .. }
        ));
    }
    let c = hdf5::File::open(&src).unwrap();
    assert_eq!(
        c.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        (1..=N as i32).collect::<Vec<i32>>(),
        "the external files still hold the data, untouched"
    );
    c.close().unwrap();

    // Replacing the dataset is the supported way out: the removal drops the
    // External Data Files message with the rest of the old header.
    let f = File::open_rw(&src).unwrap();
    f.root().delete("d").unwrap();
    f.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[7i32; N]);
        })
        .unwrap();
    f.commit().unwrap();
    drop(f);
    let r = File::open(&src).unwrap();
    assert_eq!(r.dataset("d").unwrap().read_i32().unwrap(), vec![7i32; N]);
}

/// `File::copy` must never reproduce an externally stored dataset, and must
/// refuse it **by name** (issue #336).
///
/// It used to refuse by accident: the copy planner read the undefined data
/// address as a real one and failed a bounds check — the same refusal a
/// *never-written* contiguous dataset got, a dataset that is perfectly valid and
/// ought to copy. Teaching `copy` to carry unallocated storage, as `repack`
/// learned in #293, removed that accident, so the refusal now has to come from
/// recognising the External Data Files message itself. Asserting the message
/// rather than merely `Err` is what distinguishes the two: a bounds check that
/// crept back would refuse this dataset just as firmly, and this test would go
/// on passing while the never-written datasets in `crates/crosscheck/tests/edit.rs` and
/// `edit_in_place` broke.
///
/// Both directions are covered, because they read the source at different
/// points: an in-file `copy` reads at commit and refuses there, while a
/// cross-file `copy_from` reads the other file as the copy is staged (so the
/// source need not outlive the call) and refuses on the spot.
#[test]
fn copy_refuses_an_external_dataset_by_name() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let payload = dir.path().join("copy_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());

    let expect = |err: hdf5_pure::Error, what: &str| {
        let text = err.to_string();
        assert!(
            text.contains("external files") && text.contains("H5Pset_external"),
            "{what}: expected external storage to be named, got {text}"
        );
    };

    let f = File::open_rw(&src).unwrap();
    f.copy("d", "d_copy").unwrap();
    let err = f.commit().unwrap_err();
    drop(f);
    expect(err, "in-file copy");
    assert!(
        File::open(&src).unwrap().dataset("d_copy").is_err(),
        "a refused copy must leave no dataset behind"
    );

    // Cross-file: the same dataset copied into a fresh file.
    let dst = dir.path().join("dst.h5");
    {
        let mut b = hdf5_pure::FileBuilder::new();
        b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
        b.write(&dst).unwrap();
    }
    let source = File::open(&src).unwrap();
    let f = File::open_rw(&dst).unwrap();
    let err = f.copy_from(&source, "d", "d_copy").unwrap_err();
    f.commit().unwrap();
    drop(f);
    expect(err, "cross-file copy_from");
    let after = File::open(&dst).unwrap();
    assert!(
        after.dataset("d_copy").is_err(),
        "a refused cross-file copy must leave no dataset behind"
    );
    assert_eq!(
        after.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the refused commit left the destination's own data alone"
    );
}
