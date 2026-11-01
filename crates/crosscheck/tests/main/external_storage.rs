#![cfg(feature = "__hdf5-1.10")]
//! External-storage metadata and rejection through the public API.

use hdf5::plist::DatasetCreate;
use hdf5::plist::dataset_create::ExternalFile;
use hdf5_pure::{File, RepackOptions};

use test_util_hdf5::{assertion, file};

const N: usize = 16;

// The sidecar stores 1..=16, with native metadata and readback checked after reopening.
fn write_external_fixture(path: &std::path::Path, payload: &str) {
    let file = file::libhdf5_create_v18(path);
    let properties = DatasetCreate::build()
        .external(payload, 0, N * 4)
        .finish()
        .unwrap();
    let expected = vec![ExternalFile {
        name: payload.to_owned(),
        offset: 0,
        size: N * 4,
    }];
    assert_eq!(properties.get_external().unwrap(), expected,);
    let dataset = file
        .new_dataset::<i32>()
        .set_dcpl(&properties)
        .shape([N])
        .create("d")
        .unwrap();
    dataset.write(&(1..=N as i32).collect::<Vec<_>>()).unwrap();
    drop(dataset);
    drop(properties);
    file.close().unwrap();
    let file = hdf5::File::open(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.dcpl().unwrap().get_external().unwrap(), expected,);
    assert_eq!(
        dataset.read_raw::<i32>().unwrap(),
        (1..=N as i32).collect::<Vec<_>>()
    );
    drop(dataset);
    file.close().unwrap();
    let bytes = std::fs::read(payload).unwrap();
    assert_eq!(bytes.len(), N * 4);
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(i32::from_ne_bytes)
        .collect::<Vec<_>>();
    assert_eq!(values, (1..=N as i32).collect::<Vec<_>>());
}

#[test]
fn repack_refuses_external_data_storage() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let dst = dir.path().join("external_repacked.h5");
    let payload = dir.path().join("repack_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());

    // The native reader follows the external file for the element values.
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

    // The contiguous layout has an undefined data address.
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

    let err = hdf5_pure::repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    assertion::repack_unsupported(
        err,
        "dataset d: external data storage (H5Pset_external) cannot be repacked -- its element bytes live in files this crate does not read",
    );
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

#[test]
fn every_read_path_refuses_external_data_storage() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let payload = dir.path().join("read_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());

    // The native reader follows the external file for the element values.
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

    // The streaming reader uses a different backend.
    let s = File::open_streaming(&src).unwrap();
    refusal(
        "streaming read_i32",
        s.dataset("d").unwrap().read_i32().unwrap_err(),
    );
}

#[test]
fn external_data_storage_still_reports_its_metadata() {
    let dir = tempfile::tempdir().unwrap();
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

#[test]
fn no_write_reaches_an_external_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let payload = dir.path().join("write_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());
    let before = std::fs::metadata(&src).unwrap().len();

    let refusal = |error| {
        assertion::edit_unsupported(
            error,
            "this dataset's elements live in external files (H5Pset_external), which this engine does not write; writing them into the HDF5 file would leave it disagreeing with the external files about where the data is. Delete the dataset and create it again in the same commit to replace it",
        );
    };

    let f = File::open_rw(&src).unwrap();
    refusal(f.dataset("d").unwrap().write(&[100i32; N]).unwrap_err());
    refusal(
        f.dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[100i32; N]);
            })
            .unwrap_err(),
    );
    refusal(
        f.dataset("d")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[100]);
            })
            .unwrap_err(),
    );
    drop(f);

    assert_eq!(
        std::fs::metadata(&src).unwrap().len(),
        before,
        "a refused write must not have grown the file"
    );

    // Attribute edits preserve the external-file message and undefined data address.
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

    // Dataset replacement removes the external-file message.
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

#[test]
fn copy_refuses_an_external_dataset_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("external.h5");
    let payload = dir.path().join("copy_payload.bin");
    write_external_fixture(&src, payload.to_str().unwrap());

    let expect = |err| {
        assertion::edit_unsupported(
            err,
            "a dataset stores its elements in external files (H5Pset_external), which a copy cannot reproduce -- its data lives in files this crate does not read",
        );
    };

    let f = File::open_rw(&src).unwrap();
    f.copy("d", "d_copy").unwrap();
    let err = f.commit().unwrap_err();
    drop(f);
    expect(err);
    assertion::path_not_found(
        File::open(&src).unwrap().dataset("d_copy").unwrap_err(),
        "d_copy",
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
    expect(err);
    let after = File::open(&dst).unwrap();
    assertion::path_not_found(after.dataset("d_copy").unwrap_err(), "d_copy");
    assert_eq!(
        after.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the refused commit left the destination's own data alone"
    );
}
