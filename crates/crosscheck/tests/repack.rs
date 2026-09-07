// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Cross-validation for whole-file repack (issue #21) against the reference
//! HDF5 C library: a file the C library *writes* is repacked by `hdf5_pure`,
//! and the result is read back by both readers. Also proves the fail-loud
//! contract on a real variable-length string dataset the C library produces.

use hdf5_pure::{File, RepackOptions, repack};
use tempfile::tempdir;

use hdf5_pure_crosscheck::assert_c_absent;

#[test]
fn c_file_repacked_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_src.h5");
    let dst = dir.path().join("repacked.h5");

    // The C library writes alpha (f64), doomed (i32, to be dropped), and a group
    // grp/beta (i32), using the 1.10+ format.
    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("alpha")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((4,))
            .create("doomed")
            .unwrap()
            .write(&[7i32, 8, 9, 10])
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<i32>()
            .shape((4,))
            .create("beta")
            .unwrap()
            .write(&[10i32, 20, 30, 40])
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new().drop_path("doomed")).unwrap();

    // hdf5-pure reads the repacked file: survivors intact, dropped gone.
    let f = File::open(&dst).unwrap();
    assert_eq!(
        f.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("grp/beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert!(f.dataset("doomed").is_err());

    // The reference C library agrees — the real interop proof.
    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        c.dataset("grp/beta").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_c_absent(&c.dataset("doomed").unwrap_err(), "doomed");
}

#[test]
fn c_reads_repacked_scale_offset() {
    // hdf5-pure writes an integer dataset compressed with lossless scale-offset,
    // hdf5-pure repacks it (a chunked dataset, so its compressed chunks are copied
    // verbatim with the source filter-pipeline message carried through), and the
    // reference C library decodes the result. That the C library reads the exact
    // values back proves the verbatim-copied chunk format and reused pipeline
    // message are valid and interoperable, not a layout that merely round-trips
    // in-crate.
    let dir = tempdir().unwrap();
    let src = dir.path().join("so_src.h5");
    let dst = dir.path().join("so_repacked.h5");

    let data: Vec<i32> = (0..1024).map(|i| 100 + i % 17).collect();
    {
        let mut b = hdf5_pure::FileBuilder::new();
        b.create_dataset("vals")
            .with_i32_data(&data)
            .with_chunks(&[128])
            .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
        b.write(&src).unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    assert_eq!(f.dataset("vals").unwrap().read_i32().unwrap(), data);

    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(c.dataset("vals").unwrap().read_raw::<i32>().unwrap(), data);
}

#[test]
fn repack_roundtrips_c_vlen_string_dataset() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen.h5");
    let dst = dir.path().join("vlen_repacked.h5");

    let words = ["alpha", "beta", "gamma", "", "δelta"];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((words.len(),))
            .create("labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    // A VL-string dataset the C library wrote must now round-trip faithfully.
    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // hdf5-pure reads the repacked values back, in order, including the empty
    // and non-ASCII elements.
    let f = File::open(&dst).unwrap();
    let labels = f.dataset("labels").unwrap();
    let got = labels.read_vlen_strings(Default::default()).unwrap();
    assert_eq!(got, words);

    // The datatype must remain variable-length, not be silently converted to a
    // fixed-length string.
    assert!(
        matches!(
            labels.datatype().unwrap(),
            hdf5_pure::Datatype::VariableLength { .. }
        ),
        "repacked datatype must stay variable-length"
    );

    // The reference C library agrees on both values and that the datatype is
    // variable-length — the real interop proof.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("labels").unwrap();
    let cvals = cds.read_raw::<VarLenUnicode>().unwrap();
    let cstrings: Vec<String> = cvals.iter().map(|v| v.as_str().to_string()).collect();
    assert_eq!(cstrings, words);
    assert!(
        cds.dtype().unwrap().is::<VarLenUnicode>(),
        "C library must see a variable-length Unicode string datatype"
    );
}

/// The C library fills a collection and allocates another, so a large enough
/// variable-length dataset it writes already spans several of them. Repack
/// re-stages that data through fresh collections of its own, which is only
/// faithful if both the read and the write sides carry each element's
/// collection along with its index.
#[test]
fn repack_roundtrips_c_vlen_dataset_spanning_many_collections() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_many.h5");
    let dst = dir.path().join("vlen_many_repacked.h5");

    // Past one collection's 65,535-object index, so the destination splits too.
    let count = u16::MAX as usize + 5_000;
    let words: Vec<String> = (0..count).map(|i| format!("s{i}")).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((count,))
            .create("labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let got = f
        .dataset("labels")
        .unwrap()
        .read_vlen_strings(Default::default())
        .unwrap();
    assert_eq!(got, words);

    // The reference library agrees, element for element.
    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c
        .dataset("labels")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap();
    assert_eq!(cvals.len(), count);
    for i in [0, 65_534, 65_535, count - 1] {
        assert_eq!(cvals[i].as_str(), words[i], "element {i} differs");
    }
}

#[test]
fn repack_roundtrips_vlen_string_2d() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_2d.h5");
    let dst = dir.path().join("vlen_2d_repacked.h5");

    // 2x3 grid, row-major.
    let words = ["a", "bb", "ccc", "", "ee", "ffffff"];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = words
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((2, 3))
            .create("grid")
            .unwrap()
            .write_raw(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let grid = f.dataset("grid").unwrap();
    assert_eq!(grid.shape().unwrap(), vec![2, 3]);
    assert_eq!(grid.read_vlen_strings(Default::default()).unwrap(), words);

    // C library agrees on shape and values.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("grid").unwrap();
    assert_eq!(cds.shape(), vec![2, 3]);
    let cvals = cds.read_raw::<VarLenUnicode>().unwrap();
    let cstrings: Vec<String> = cvals.iter().map(|v| v.as_str().to_string()).collect();
    assert_eq!(cstrings, words);
}

#[test]
fn repack_roundtrips_c_vlen_sequence_dataset() {
    use hdf5::types::VarLenArray;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_seq.h5");
    let dst = dir.path().join("vlen_seq_repacked.h5");

    // The C library writes a non-string VL dataset (`H5T_VLEN { i32 }`),
    // including an empty sequence.
    let seqs: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![], vec![-7, 42, 0, 99], vec![5]];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenArray<i32>> = seqs
            .iter()
            .map(|s| VarLenArray::from_slice(s.as_slice()))
            .collect();
        file.new_dataset::<VarLenArray<i32>>()
            .shape((seqs.len(),))
            .create("seqs")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // hdf5-pure must see the repacked datatype as a non-string variable-length
    // sequence (not silently converted).
    let f = File::open(&dst).unwrap();
    let ds = f.dataset("seqs").unwrap();
    assert!(
        matches!(
            ds.datatype().unwrap(),
            hdf5_pure::Datatype::VariableLength {
                is_string: false,
                ..
            }
        ),
        "repacked datatype must stay a non-string variable-length sequence"
    );

    // The reference C library reads the exact sequences back — the interop proof
    // that the re-staged global heap and rebuilt references are valid.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("seqs").unwrap();
    let cvals = cds.read_raw::<VarLenArray<i32>>().unwrap();
    let got: Vec<Vec<i32>> = cvals.iter().map(|v| v.as_slice().to_vec()).collect();
    assert_eq!(got, seqs);
}

#[test]
fn repack_roundtrips_c_object_reference_dataset() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_refs.h5");
    let dst = dir.path().join("refs_repacked.h5");

    // The C library writes two targets plus a dataset of object references to
    // them (one in the root, one in a subgroup).
    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("alpha")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0])
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<i32>()
            .shape((4,))
            .create("beta")
            .unwrap()
            .write(&[10i32, 20, 30, 40])
            .unwrap();
        let refs = vec![
            ObjectReference1::create(&file, "alpha").unwrap(),
            ObjectReference1::create(&file, "grp/beta").unwrap(),
            // A reference to the root group exercises the empty-path resolution.
            ObjectReference1::create(&file, "/").unwrap(),
        ];
        file.new_dataset::<ObjectReference1>()
            .shape((3,))
            .create("refs")
            .unwrap()
            .write(&refs)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // hdf5-pure sees the repacked datatype as an object reference (not converted).
    let f = File::open(&dst).unwrap();
    assert!(
        matches!(
            f.dataset("refs").unwrap().datatype().unwrap(),
            hdf5_pure::Datatype::Reference {
                ref_type: hdf5_pure::ReferenceType::Object,
                ..
            }
        ),
        "repacked datatype must stay an object reference"
    );

    // The reference C library dereferences each repacked reference to the right
    // object — the proof that the addresses were rewritten to the new locations
    // rather than copied stale.
    let c = hdf5::File::open(&dst).unwrap();
    let cds = c.dataset("refs").unwrap();
    let cvals = cds.read_raw::<ObjectReference1>().unwrap();
    assert_eq!(cvals.len(), 3);
    match cvals[0].dereference(&c).unwrap() {
        ReferencedObject::Dataset(d) => {
            assert_eq!(d.read_raw::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
        }
        other => panic!("ref 0 should resolve to a dataset, got {other:?}"),
    }
    match cvals[1].dereference(&c).unwrap() {
        ReferencedObject::Dataset(d) => {
            assert_eq!(d.read_raw::<i32>().unwrap(), vec![10, 20, 30, 40]);
        }
        other => panic!("ref 1 should resolve to a dataset, got {other:?}"),
    }
    match cvals[2].dereference(&c).unwrap() {
        ReferencedObject::Group(_) => {}
        other => panic!("ref 2 should resolve to the root group, got {other:?}"),
    }
}

#[test]
fn repack_refuses_reference_to_dropped_object() {
    use hdf5::{ObjectReference, ObjectReference1};

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_refs_drop.h5");
    let dst = dir.path().join("refs_drop_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("target")
            .unwrap()
            .write(&[1.0f64, 2.0])
            .unwrap();
        let refs = vec![ObjectReference1::create(&file, "target").unwrap()];
        file.new_dataset::<ObjectReference1>()
            .shape((1,))
            .create("refs")
            .unwrap()
            .write(&refs)
            .unwrap();
        file.close().unwrap();
    }

    // Dropping the referenced object must fail the repack by name rather than
    // silently leave a dangling reference.
    let err = repack(&src, &dst, &RepackOptions::new().drop_path("target")).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("refs") && msg.contains("target") && msg.contains("dropped"),
            "error should name the reference dataset and the dropped target: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// A chunked VL-string dataset the C library wrote repacks into an equally
/// chunked one, with its heap references rebuilt against the new file's
/// collections (issue #109). The C library reading the *output* is the check
/// that those rebuilt addresses are real.
#[test]
fn repack_roundtrips_chunked_vlen_string_dataset() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_chunked.h5");
    let dst = dir.path().join("vlen_chunked_repacked.h5");

    let expected: Vec<String> = (0..8).map(|i| format!("word{i}")).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<VarLenUnicode> = expected
            .iter()
            .map(|s| VarLenUnicode::from_str(s).unwrap())
            .collect();
        file.new_dataset::<VarLenUnicode>()
            .shape((8,))
            .chunk((4,))
            .create("chunked_labels")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    assert_eq!(
        File::open(&dst)
            .unwrap()
            .dataset("chunked_labels")
            .unwrap()
            .read_string()
            .unwrap(),
        expected
    );
    let f = hdf5::File::open(&dst).unwrap();
    let ds = f.dataset("chunked_labels").unwrap();
    assert!(
        ds.is_chunked(),
        "repack must reproduce the source's chunked layout, not flatten it"
    );
    let read: Vec<String> = ds
        .read_raw::<VarLenUnicode>()
        .unwrap()
        .into_iter()
        .map(|s| s.as_str().to_owned())
        .collect();
    assert_eq!(read, expected);
}

/// The compressed and resizable variants of the same path, written by this crate
/// and repacked: the pipeline and the unlimited dimension must survive, and the
/// C library must read the result.
#[test]
fn repack_roundtrips_filtered_and_resizable_vlen_string_datasets() {
    use hdf5::types::VarLenUnicode;

    let dir = tempdir().unwrap();
    let src = dir.path().join("filtered_vlen.h5");
    let dst = dir.path().join("filtered_vlen_repacked.h5");

    let words: Vec<String> = (0..40).map(|i| format!("label-{i}")).collect();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    {
        let mut b = hdf5_pure::FileBuilder::new();
        b.create_dataset("compressed")
            .with_vlen_strings(&refs)
            .with_chunks(&[8])
            .with_deflate(6);
        b.create_dataset("growable")
            .with_vlen_strings(&refs)
            .with_shape(&[words.len() as u64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[8]);
        b.write(&src).unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let out = File::open(&dst).unwrap();
    for name in ["compressed", "growable"] {
        assert_eq!(
            out.dataset(name).unwrap().read_string().unwrap(),
            words,
            "{name} did not survive the repack"
        );
    }

    let f = hdf5::File::open(&dst).unwrap();
    let compressed = f.dataset("compressed").unwrap();
    assert!(
        !compressed.filters().is_empty(),
        "repack must carry the deflate pipeline onto the rebuilt dataset"
    );
    assert!(
        f.dataset("growable").unwrap().is_resizable(),
        "repack must carry the unlimited dimension onto the rebuilt dataset"
    );
    for name in ["compressed", "growable"] {
        let read: Vec<String> = f
            .dataset(name)
            .unwrap()
            .read_raw::<VarLenUnicode>()
            .unwrap()
            .into_iter()
            .map(|s| s.as_str().to_owned())
            .collect();
        assert_eq!(read, words, "C library disagreed on {name}");
    }
}

/// A boolean attribute is an HDF5 enumeration, which [`AttrValue`] cannot carry
/// faithfully: it decodes through the integer base type, so the codes reach the
/// caller and `enum[FALSE, TRUE]` does not. Repack used to refuse the whole file
/// over one — the only honest answer while every attribute went through a
/// decode, since the alternative was writing the base type back in its place.
///
/// Copying the message verbatim answers it properly instead: an enumeration
/// carries no address, so its bytes mean the same thing in the destination and
/// the attribute survives with its members. This is the general form of the
/// refusal that went away, not a special case for booleans — a compound or
/// opaque attribute travels for the same reason (issue #241).
#[test]
fn repack_carries_an_attribute_attr_value_cannot_express_faithfully() {
    use hdf5::types::TypeDescriptor;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_boolattr.h5");
    let dst = dir.path().join("boolattr_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((2,))
            .create("data")
            .unwrap();
        ds.write(&[1.0f64, 2.0]).unwrap();
        ds.new_attr::<bool>()
            .shape(())
            .create("active")
            .unwrap()
            .write_scalar(&true)
            .unwrap();
        file.close().unwrap();
    }

    let source_type = {
        let c = hdf5::File::open(&src).unwrap();
        c.dataset("data")
            .unwrap()
            .attr("active")
            .unwrap()
            .dtype()
            .unwrap()
            .to_descriptor()
            .unwrap()
    };
    assert!(
        matches!(source_type, TypeDescriptor::Boolean),
        "the C library should have written an enumeration: {source_type:?}"
    );

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let attr = c.dataset("data").unwrap().attr("active").unwrap();
    assert_eq!(
        attr.dtype().unwrap().to_descriptor().unwrap(),
        source_type,
        "the enumeration must cross the repack as itself"
    );
    assert!(
        attr.read_scalar::<bool>().unwrap(),
        "and still hold its value"
    );
}

/// What the fail-loud contract still covers once verbatim copying has taken the
/// address-free datatypes off the refusal list.
///
/// An object-reference attribute stores an object-header address, so its bytes
/// cannot be copied, and [`AttrValue`] has no variant to re-encode it from
/// either. Repack must still name it and refuse rather than write a file the
/// attribute is missing from — dropping it silently is the failure this test
/// exists to catch, and the shrinking refusal list makes it the last one holding
/// that line for attributes.
#[test]
fn repack_refuses_an_attribute_whose_address_it_cannot_rewrite() {
    use hdf5::{ObjectReference, ObjectReference1};

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_refattr.h5");
    let dst = dir.path().join("refattr_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((2,))
            .create("data")
            .unwrap();
        ds.write(&[1.0f64, 2.0]).unwrap();
        ds.new_attr::<ObjectReference1>()
            .shape(())
            .create("points_at")
            .unwrap()
            .write_scalar(&ObjectReference1::create(&file, "data").unwrap())
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => {
            assert!(
                msg.contains("points_at") && msg.contains("data"),
                "error should name the attribute and its dataset: {msg}"
            );
        }
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

#[test]
fn c_deflate_dataset_repacked_verbatim_read_by_both() {
    // The reference C library writes a chunked + shuffle + deflate dataset.
    // hdf5-pure repacks it by copying the C library's *own* compressed chunk
    // streams verbatim (it never re-encodes them), then both readers decode the
    // result. That the C library reads its exact values back proves the copied
    // chunk bytes and the carried-through filter-pipeline message remain valid.
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_deflate.h5");
    let dst = dir.path().join("c_deflate_repacked.h5");

    let data: Vec<i32> = (0..4096).map(|i| i % 13).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((4096,))
            .chunk((512,))
            .shuffle()
            .deflate(6)
            .create("vals")
            .unwrap();
        ds.write_raw(data.as_slice()).unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    assert_eq!(f.dataset("vals").unwrap().read_i32().unwrap(), data);

    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(c.dataset("vals").unwrap().read_raw::<i32>().unwrap(), data);
}

#[test]
fn c_sparse_chunked_lossless_repacked_falls_back() {
    // The C library writes only the first chunks of a chunked dataset, leaving the
    // tail unallocated (a sparse chunk grid). The verbatim path needs a dense
    // grid, so repack falls back to the read-raw + re-encode path. With a lossless
    // pipeline (deflate) that fallback is faithful: the written values survive and
    // the unwritten tail reads back as the fill value (0).
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_sparse.h5");
    let dst = dir.path().join("c_sparse_repacked.h5");

    let n = 2000usize; // chunk 512 -> 4 chunks; only the first ~1000 written
    let written = 1000usize;
    let head: Vec<i32> = (1..=written as i32).collect();
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&src)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([n])
            .chunk([512])
            .deflate(4)
            .create("data")
            .unwrap();
        ds.write_slice(head.as_slice(), 0..written).unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let vals = f.dataset("data").unwrap().read_i32().unwrap();
    assert_eq!(vals.len(), n);
    assert_eq!(&vals[..written], head.as_slice());
    assert!(
        vals[written..].iter().all(|&v| v == 0),
        "unwritten tail should read back as the fill value"
    );
}

/// The same fixture in the lossless mode, which repack now reproduces rather
/// than refusing (issue #297).
///
/// It was refused for the fill value, not for the mode: the reference records a
/// defined fill value on **every** scale-offset dataset it writes, and the
/// writer here had nowhere to put one, so re-encoding would have dropped the
/// source's chunk-fill semantics. That covered essentially every C-written and
/// h5py-written scale-offset dataset whose grid has a hole in it.
///
/// The filter parameters are compared entry for entry, because the values alone
/// cannot see the difference: a chunk re-encoded with the fill value dropped
/// decodes to exactly the same numbers, just packed a code point wider and no
/// longer collapsing runs of the fill value.
#[test]
fn c_sparse_chunked_scale_offset_repacks_with_its_fill_value() {
    use hdf5::filters::ScaleOffset as CScaleOffset;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_sparse_so.h5");
    let dst = dir.path().join("c_sparse_so_repacked.h5");

    let n = 2000usize;
    let written = 1000usize;
    let fill = 7i32;
    // Copies of the fill value inside the written region, so the sentinel is
    // exercised where the data is rather than only where it is missing.
    let head: Vec<i32> = (0..written)
        .map(|i| {
            if i % 5 == 0 {
                fill
            } else {
                100 + (i % 17) as i32
            }
        })
        .collect();
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&src)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([n])
            .chunk([512])
            .scale_offset(CScaleOffset::Integer(0))
            .fill_value(fill)
            .create("data")
            .unwrap();
        ds.write_slice(head.as_slice(), 0..written).unwrap();
        file.close().unwrap();
    }

    let parms = |path: &std::path::Path| {
        File::open(path)
            .unwrap()
            .dataset("data")
            .unwrap()
            .filter_pipeline()
            .into_iter()
            .find(|f| f.id == 6)
            .expect("a scale-offset filter")
            .client_data
    };

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let mut want = head.clone();
    want.resize(n, fill);
    assert_eq!(
        File::open(&dst)
            .unwrap()
            .dataset("data")
            .unwrap()
            .read_i32()
            .unwrap(),
        want
    );
    assert_eq!(
        hdf5::File::open(&dst)
            .unwrap()
            .dataset("data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        want,
        "the C library must decode the re-encoded chunks"
    );
    assert_eq!(
        parms(&dst),
        parms(&src),
        "the rebuilt filter must record the source's fill value"
    );
}

#[test]
fn c_sparse_chunked_lossy_repack_refused() {
    // The C library writes only the first chunks of a chunked dataset compressed
    // with float D-scale scale-offset (a lossy filter), leaving the tail
    // unallocated (a sparse grid). The verbatim path needs a dense grid, so repack
    // would fall back to the read-raw + re-encode path — but re-encoding a lossy
    // filter is not guaranteed idempotent, so repack must refuse by name rather
    // than risk silently perturbing the data.
    use hdf5::filters::ScaleOffset as CScaleOffset;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_sparse_lossy.h5");
    let dst = dir.path().join("c_sparse_lossy_repacked.h5");

    let n = 2000usize;
    let written = 1000usize;
    let head: Vec<f64> = (0..written).map(|i| i as f64 * 0.01).collect();
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&src)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape([n])
            .chunk([512])
            .scale_offset(CScaleOffset::FloatDScale(3))
            .create("data")
            .unwrap();
        ds.write_slice(head.as_slice(), 0..written).unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("data") && msg.contains("scale-offset"),
            "error should name the dataset and reason: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// A compound whose members include a variable-length string stores a 16-byte
/// global-heap reference *inside* each element. Copying those element bytes
/// verbatim leaves the reference pointing at a collection in the source file's
/// address space, which the destination never allocates (issue #201). Repack has
/// to re-stage the payloads through the destination's own heap and rewrite the
/// embedded references, exactly as it does for a top-level VL dataset.
///
/// The C library is the load-bearing reader here: this crate's own reader
/// resolves the heap lazily and reads `read_raw` bytes without validating the
/// embedded addresses, so a self-round-trip alone would not catch a stale one.
#[test]
fn repack_roundtrips_c_compound_with_vlen_string_member() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug, PartialEq)]
    #[repr(C)]
    struct Labelled {
        id: i32,
        label: VarLenUnicode,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_vlen.h5");
    let dst = dir.path().join("compound_vlen_repacked.h5");

    let rows = [
        "alpha",
        "",
        "gamma",
        "δelta",
        "a longer label than the rest",
    ];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<Labelled> = rows
            .iter()
            .enumerate()
            .map(|(i, s)| Labelled {
                id: i as i32 * 10,
                label: VarLenUnicode::from_str(s).unwrap(),
            })
            .collect();
        file.new_dataset::<Labelled>()
            .shape((rows.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // The datatype must stay a compound carrying a variable-length member, not
    // be flattened or converted.
    let f = File::open(&dst).unwrap();
    let ds = f.dataset("rows").unwrap();
    match ds.datatype().unwrap() {
        hdf5_pure::Datatype::Compound { members, .. } => {
            assert_eq!(members.len(), 2);
            assert!(
                matches!(
                    members[1].datatype,
                    hdf5_pure::Datatype::VariableLength { .. }
                ),
                "the label member must stay variable-length"
            );
        }
        other => panic!("expected a compound datatype, got {other:?}"),
    }

    // The reference C library reads every row back — the interop proof that the
    // embedded heap references name collections in the *destination*.
    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Labelled>().unwrap();
    assert_eq!(cvals.len(), rows.len());
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(cvals[i].id, i as i32 * 10, "row {i} id differs");
        assert_eq!(cvals[i].label.as_str(), *row, "row {i} label differs");
    }
}

/// A compound can carry several variable-length members of different kinds, and
/// each one's reference stores a *count* in the units of its own base type — bytes
/// for a string, elements for a sequence. Re-staging has to keep those units
/// straight per member, not per dataset.
#[test]
fn repack_roundtrips_c_compound_with_two_vlen_members() {
    use hdf5::types::{VarLenArray, VarLenUnicode};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        id: i64,
        samples: VarLenArray<i32>,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_two_vlen.h5");
    let dst = dir.path().join("compound_two_vlen_repacked.h5");

    let labels = ["first", "", "third"];
    let samples: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![], vec![-9, 400_000]];
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<Row> = labels
            .iter()
            .zip(&samples)
            .enumerate()
            .map(|(i, (l, s))| Row {
                label: VarLenUnicode::from_str(l).unwrap(),
                id: i as i64 - 1,
                samples: VarLenArray::from_slice(s.as_slice()),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((labels.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), labels.len());
    for i in 0..labels.len() {
        assert_eq!(cvals[i].label.as_str(), labels[i], "row {i} label differs");
        assert_eq!(cvals[i].id, i as i64 - 1, "row {i} id differs");
        assert_eq!(
            cvals[i].samples.as_slice(),
            samples[i].as_slice(),
            "row {i} samples differ"
        );
    }
}

/// A chunked compound with a variable-length member has to reach the re-staging
/// path rather than the verbatim chunk copy: copying compressed chunks would
/// carry the source's heap addresses through untouched, and the destination's own
/// collections have to be placed before the chunks are encoded (issue #109).
#[test]
fn repack_roundtrips_c_chunked_compound_with_vlen_member() {
    use hdf5::types::VarLenUnicode;
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        id: i32,
        label: VarLenUnicode,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_chunked_compound_vlen.h5");
    let dst = dir.path().join("chunked_compound_vlen_repacked.h5");

    let n = 40usize;
    let labels: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let vals: Vec<Row> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| Row {
                id: i as i32,
                label: VarLenUnicode::from_str(l).unwrap(),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((n,))
            .chunk((8,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // Chunking must survive the rebuild, not be flattened to contiguous.
    let f = File::open(&dst).unwrap();
    assert!(
        matches!(
            f.dataset("rows").unwrap().layout().unwrap(),
            hdf5_pure::Layout::Chunked { .. }
        ),
        "repacked dataset must stay chunked"
    );

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), n);
    for i in 0..n {
        assert_eq!(cvals[i].id, i as i32, "row {i} id differs");
        assert_eq!(cvals[i].label.as_str(), labels[i], "row {i} label differs");
    }
}

/// An object-header address embedded in a compound member goes stale on rewrite
/// for the same reason a top-level one does: the destination puts the target
/// object somewhere else. Copying the element bytes verbatim leaves a file whose
/// fixed-size members read back correctly and whose references dereference into
/// whatever now occupies that address — which is why this test dereferences
/// rather than merely reading the rows.
#[test]
fn repack_roundtrips_c_compound_with_object_reference_member() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        id: i32,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_ref.h5");
    let dst = dir.path().join("compound_ref_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("alpha")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<i32>()
            .shape((2,))
            .create("beta")
            .unwrap()
            .write(&[40i32, 50])
            .unwrap();
        let vals = vec![
            Row {
                id: 7,
                target: ObjectReference1::create(&file, "alpha").unwrap(),
            },
            Row {
                id: 8,
                target: ObjectReference1::create(&file, "grp/beta").unwrap(),
            },
        ];
        file.new_dataset::<Row>()
            .shape((vals.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), 2);
    assert_eq!(cvals[0].id, 7);
    assert_eq!(cvals[1].id, 8);

    // Each embedded reference must resolve to its own target's *new* address, and
    // that target must still hold the right data.
    let expected: [&[i32]; 2] = [&[1, 2, 3], &[40, 50]];
    for (i, row) in cvals.iter().enumerate() {
        match row.target.dereference(&c).unwrap() {
            ReferencedObject::Dataset(ds) => assert_eq!(
                ds.read_raw::<i32>().unwrap(),
                expected[i],
                "row {i} reference resolves to the wrong dataset"
            ),
            other => panic!("row {i}: expected a dataset, got {other:?}"),
        }
    }
}

/// The refusals that guard the top-level object-reference path have to guard the
/// embedded one too: a reference into a dropped subtree cannot be rewritten to
/// anything meaningful, so the repack must fail by name rather than emit a
/// dangling address.
#[test]
fn repack_refuses_compound_reference_to_dropped_object() {
    use hdf5::{ObjectReference, ObjectReference1};

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        id: i32,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_ref_dropped.h5");
    let dst = dir.path().join("compound_ref_dropped_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((2,))
            .create("doomed")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        let vals = vec![Row {
            id: 1,
            target: ObjectReference1::create(&file, "doomed").unwrap(),
        }];
        file.new_dataset::<Row>()
            .shape((1,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new().drop_path("doomed")).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("rows") && msg.contains("doomed"),
            "error should name the dataset and its dropped target: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// A compound can carry a variable-length member *and* an object-reference
/// member. Both kinds of embedded address go stale on rewrite, so rewriting only
/// the first kind found leaves the other pointing into the source file — the
/// original defect surviving for the one shape that reaches both paths.
///
/// The reference is dereferenced rather than merely read back, because a stale
/// address reads as a perfectly ordinary 8 bytes.
#[test]
fn repack_roundtrips_c_compound_with_both_vlen_and_reference_members() {
    use hdf5::types::VarLenUnicode;
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        id: i32,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_compound_both.h5");
    let dst = dir.path().join("compound_both_repacked.h5");

    let labels = ["alpha", "", "gamma"];
    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("pointee")
            .unwrap()
            .write(&[11i32, 22, 33])
            .unwrap();
        let vals: Vec<Row> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| Row {
                label: VarLenUnicode::from_str(l).unwrap(),
                id: i as i32,
                target: ObjectReference1::create(&file, "pointee").unwrap(),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((labels.len(),))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let c = hdf5::File::open(&dst).unwrap();
    let cvals = c.dataset("rows").unwrap().read_raw::<Row>().unwrap();
    assert_eq!(cvals.len(), labels.len());
    for (i, row) in cvals.iter().enumerate() {
        assert_eq!(row.label.as_str(), labels[i], "row {i} label differs");
        assert_eq!(row.id, i as i32, "row {i} id differs");
        match row.target.dereference(&c).unwrap() {
            ReferencedObject::Dataset(ds) => assert_eq!(
                ds.read_raw::<i32>().unwrap(),
                vec![11, 22, 33],
                "row {i} reference resolves to the wrong data"
            ),
            other => panic!("row {i}: expected a dataset, got {other:?}"),
        }
    }
}

/// The object-reference refusals must not be reachable-around by adding a
/// variable-length member: a reference into a dropped subtree cannot be rewritten
/// to anything meaningful whatever else the compound carries.
#[test]
fn repack_refuses_dropped_target_in_a_compound_that_also_has_a_vlen_member() {
    use hdf5::types::VarLenUnicode;
    use hdf5::{ObjectReference, ObjectReference1};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_both_dropped.h5");
    let dst = dir.path().join("both_dropped_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((2,))
            .create("doomed")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        let vals = vec![Row {
            label: VarLenUnicode::from_str("x").unwrap(),
            target: ObjectReference1::create(&file, "doomed").unwrap(),
        }];
        file.new_dataset::<Row>()
            .shape((1,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new().drop_path("doomed")).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("rows") && msg.contains("doomed"),
            "error should name the dataset and its dropped target: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// Likewise the chunked refusal: an object address inside a compressed chunk
/// would need rewriting in place, and a variable-length member alongside it does
/// not make that possible.
#[test]
fn repack_refuses_chunked_compound_with_both_vlen_and_reference_members() {
    use hdf5::types::VarLenUnicode;
    use hdf5::{ObjectReference, ObjectReference1};
    use std::str::FromStr;

    #[derive(hdf5::H5Type, Clone, Debug)]
    #[repr(C)]
    struct Row {
        label: VarLenUnicode,
        target: ObjectReference1,
    }

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_both_chunked.h5");
    let dst = dir.path().join("both_chunked_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape((2,))
            .create("pointee")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        let vals: Vec<Row> = (0..8)
            .map(|i| Row {
                label: VarLenUnicode::from_str(&format!("r{i}")).unwrap(),
                target: ObjectReference1::create(&file, "pointee").unwrap(),
            })
            .collect();
        file.new_dataset::<Row>()
            .shape((8,))
            .chunk((4,))
            .create("rows")
            .unwrap()
            .write(&vals)
            .unwrap();
        file.close().unwrap();
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("rows") && msg.contains("object-reference"),
            "error should name the dataset and the reason: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
}

/// The attribute encodings the reference C library can write and this crate
/// cannot, carried across a repack and read back by that same library
/// (issue #241).
///
/// This is the half of the fidelity contract no pure-Rust test can reach.
/// `AttrValue` has no byte order, no sub-width precision and no rank above one,
/// so a source this crate *writes* cannot exhibit those losses at all — only a
/// C-written file can, which is exactly the file a user repacks.
/// Every assertion is made through the C library's own type system, so it states
/// what a consumer sees rather than what this crate's reader reports.
#[test]
fn c_written_attribute_encodings_survive_a_repack() {
    use hdf5::types::{FixedAscii, FloatSize, IntSize, TypeDescriptor, VarLenUnicode};
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_attr_src.h5");
    let dst = dir.path().join("c_attr_dst.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((4,))
            .create("data")
            .unwrap();
        ds.write(&[1.0f64, 2.0, 3.0, 4.0]).unwrap();

        // One writer for both owners: an attribute on the root group takes a
        // different path through the repack walk than one on a dataset, and the
        // widening hit both.
        let (ds_loc, root_loc): (&hdf5::Location, &hdf5::Location) = (&ds, &file);
        for owner in [ds_loc, root_loc] {
            owner
                .new_attr::<i8>()
                .create("i8")
                .unwrap()
                .write_scalar(&-3i8)
                .unwrap();
            owner
                .new_attr::<i32>()
                .create("i32")
                .unwrap()
                .write_scalar(&-7i32)
                .unwrap();
            owner
                .new_attr::<u16>()
                .create("u16")
                .unwrap()
                .write_scalar(&65535u16)
                .unwrap();
            owner
                .new_attr::<f32>()
                .create("f32")
                .unwrap()
                .write_scalar(&1.5f32)
                .unwrap();
            owner
                .new_attr::<i16>()
                .shape([3])
                .create("i16arr")
                .unwrap()
                .write(&[1i16, 2, 3])
                .unwrap();
            // Rank 2: `AttrValue`'s array variants are all one-dimensional, so a
            // decode flattens this to six elements.
            owner
                .new_attr::<i32>()
                .shape([2, 3])
                .create("rank2")
                .unwrap()
                .write_raw(&[1i32, 2, 3, 4, 5, 6])
                .unwrap();
            // A true variable-length string. Repack carries it across verbatim,
            // and a decode now reports it as `AttrValue::VarLenString` (#383).
            owner
                .new_attr::<VarLenUnicode>()
                .create("vlstr")
                .unwrap()
                .write_scalar(&VarLenUnicode::from_str("hello").unwrap())
                .unwrap();
            // The same, as an array: a scalar and a one-element-per-entry array
            // reach the writer by different paths.
            owner
                .new_attr::<VarLenUnicode>()
                .shape([2])
                .create("vlstrs")
                .unwrap()
                .write_raw(&[
                    VarLenUnicode::from_str("alpha").unwrap(),
                    VarLenUnicode::from_str("beta").unwrap(),
                ])
                .unwrap();
            // A fixed-width string declared far wider than its content: the
            // declared width is the part a decode drops, since it reports the
            // content and nothing else.
            owner
                .new_attr::<FixedAscii<16>>()
                .create("units")
                .unwrap()
                .write_scalar(&FixedAscii::<16>::from_ascii("m/s").unwrap())
                .unwrap();
            // Rank 2 over strings rather than numbers, which is the shape
            // issue #241 measured being flattened.
            owner
                .new_attr::<FixedAscii<4>>()
                .shape([2, 2])
                .create("grid")
                .unwrap()
                .write_raw(&[
                    FixedAscii::<4>::from_ascii("ab").unwrap(),
                    FixedAscii::<4>::from_ascii("cd").unwrap(),
                    FixedAscii::<4>::from_ascii("ef").unwrap(),
                    FixedAscii::<4>::from_ascii("gh").unwrap(),
                ])
                .unwrap();
        }
        file.close().unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    // One row per entry in issue #241's table of what a decode loses.
    let expected: [(&str, TypeDescriptor, Vec<usize>); 10] = [
        ("i8", TypeDescriptor::Integer(IntSize::U1), vec![]),
        ("i32", TypeDescriptor::Integer(IntSize::U4), vec![]),
        ("u16", TypeDescriptor::Unsigned(IntSize::U2), vec![]),
        ("f32", TypeDescriptor::Float(FloatSize::U4), vec![]),
        ("i16arr", TypeDescriptor::Integer(IntSize::U2), vec![3]),
        ("rank2", TypeDescriptor::Integer(IntSize::U4), vec![2, 3]),
        ("vlstr", TypeDescriptor::VarLenUnicode, vec![]),
        ("vlstrs", TypeDescriptor::VarLenUnicode, vec![2]),
        // 16, not 3: the declared width outlives the content.
        ("units", TypeDescriptor::FixedAscii(16), vec![]),
        ("grid", TypeDescriptor::FixedAscii(4), vec![2, 2]),
    ];

    let c = hdf5::File::open(&dst).unwrap();
    let ds = c.dataset("data").unwrap();
    let (ds_loc, root_loc): (&hdf5::Location, &hdf5::Location) = (&ds, &c);
    for (owner_name, owner) in [("dataset data", ds_loc), ("root group", root_loc)] {
        for (name, descriptor, shape) in &expected {
            let attr = owner
                .attr(name)
                .unwrap_or_else(|e| panic!("{owner_name} lost attribute {name:?}: {e}"));
            assert_eq!(
                &attr.dtype().unwrap().to_descriptor().unwrap(),
                descriptor,
                "{owner_name}: attribute {name:?} changed datatype across the repack"
            );
            assert_eq!(
                &attr.shape(),
                shape,
                "{owner_name}: attribute {name:?} changed shape across the repack"
            );
        }
        // The variable-length string is the one whose bytes cannot be copied, so
        // it proves the fallback re-encoded it against the destination's own heap
        // rather than leaving an address pointing into the source.
        assert_eq!(
            owner
                .attr("vlstr")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "hello",
            "{owner_name}: the variable-length string must still resolve"
        );
        assert_eq!(
            owner.attr("rank2").unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            owner
                .attr("units")
                .unwrap()
                .read_scalar::<FixedAscii<16>>()
                .unwrap()
                .as_str(),
            "m/s"
        );
        assert_eq!(
            owner
                .attr("vlstrs")
                .unwrap()
                .read_raw::<VarLenUnicode>()
                .unwrap()
                .iter()
                .map(|s| s.as_str().to_owned())
                .collect::<Vec<_>>(),
            ["alpha", "beta"]
        );
        assert_eq!(owner.attr("i8").unwrap().read_scalar::<i8>().unwrap(), -3);
        assert_eq!(
            owner.attr("f32").unwrap().read_scalar::<f32>().unwrap(),
            1.5
        );
    }
}

/// Storage the C library never allocated survives a repack as storage rather
/// than as the values reading it answers with (issue #293).
///
/// By default the reference library does not allocate a contiguous or chunked
/// dataset's storage until something is written to it, so one created and never
/// written holds nothing at all. (Compact data is inline in the layout message
/// and is always present, which is why it is not among the layouts below.)
/// Since #292 reading one answers its fill value for every element, and
/// repack used to write those values out — turning a schema-only file into a
/// fully materialized one of whatever size its shape declared.
///
/// Every layout the source can be in is covered here because each reaches a
/// different part of the writer: a contiguous dataset is preserved by leaving its
/// data address undefined, a chunked one by emitting no chunks and no index, and
/// a filtered one has to carry its pipeline across without a single chunk to
/// apply it to. The attribute on each is what proves the new path still ends at
/// `copy_dataset_attrs` — an early return that skipped it would lose every
/// attribute silently, which is the failure mode that makes each of repack's
/// other exits end there too.
#[test]
fn repack_preserves_c_written_unallocated_storage_in_every_layout() {
    const N: usize = 1000;
    const FILL: i32 = 7;
    // The bytes the elements would occupy if they were written out. Every
    // destination has to come in under this, whatever its metadata costs — a
    // coarse bound, and deliberately the weaker of the two checks: measured
    // before the fix, the *filtered* destination was 665 B, well inside it,
    // because deflate had squeezed a thousand copies of the fill value. What
    // catches that arm is the chunk count below.
    const MATERIALIZED: u64 = (N * core::mem::size_of::<i32>()) as u64;

    for layout in ["contiguous", "chunked", "filtered", "extensible"] {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src.h5");
        let dst = dir.path().join("dst.h5");
        {
            let file = hdf5::File::create(&src).unwrap();
            let b = file.new_dataset::<i32>().fill_value(FILL);
            // The C builder's shape call changes its type, so each layout is
            // built to completion in its own arm rather than accumulated.
            let ds = match layout {
                "contiguous" => b.shape((N,)).create("col"),
                "chunked" => b.shape((N,)).chunk((100,)).create("col"),
                "filtered" => b
                    .shape((N,))
                    .chunk((100,))
                    .shuffle()
                    .deflate(6)
                    .create("col"),
                _ => b
                    .shape((hdf5::Extent::resizable(N),))
                    .chunk((100,))
                    .create("col"),
            }
            .unwrap();
            ds.new_attr::<i32>()
                .shape(())
                .create("units")
                .unwrap()
                .write_scalar(&42i32)
                .unwrap();
            drop(ds);
            file.close().unwrap();
        }

        // Ground truth: the source stores nothing, and the C library reads it as
        // the fill value anyway.
        {
            let c = hdf5::File::open(&src).unwrap();
            assert_eq!(
                c.dataset("col").unwrap().read_raw::<i32>().unwrap(),
                vec![FILL; N],
                "[{layout}] C ground truth"
            );
        }

        repack(&src, &dst, &RepackOptions::new()).unwrap();

        let f = File::open(&dst).unwrap();
        let ds = f.dataset("col").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![N as u64], "[{layout}] shape");
        assert_eq!(
            ds.read_i32().unwrap(),
            vec![FILL; N],
            "[{layout}] values still read as the fill value"
        );
        assert_eq!(
            ds.fill_value::<i32>().unwrap(),
            Some(FILL),
            "[{layout}] fill value carried"
        );
        assert_eq!(
            ds.attrs().unwrap().get("units"),
            // The C library wrote it as a 4-byte signed integer, and it keeps
            // that width through the repack and back out of the reader (#350).
            Some(&hdf5_pure::AttrValue::I32(42)),
            "[{layout}] the attribute survived the new path"
        );

        // Nothing is stored. A contiguous dataset says so by leaving its address
        // undefined; a chunked one by holding no chunks. The chunked
        // destinations index differently from the source — the C library
        // defaults to the 1.8 format and its version-1 B-tree, and this crate
        // writes the 1.10 indices — so only the contiguous layout is comparable
        // field for field, and it has to be: the size recorded beside an
        // undefined address is the extent the dataset would occupy, and writing
        // zero there would contradict `Layout::Contiguous`'s own promise.
        match ds.layout().unwrap() {
            hdf5_pure::Layout::Contiguous { address, size } => {
                assert_eq!(
                    (address, size),
                    (None, MATERIALIZED),
                    "[{layout}] the destination must match the source's undefined \
                     address and declared extent"
                );
            }
            hdf5_pure::Layout::Chunked { .. } => assert_eq!(
                ds.chunks().unwrap().len(),
                0,
                "[{layout}] the destination wrote chunks"
            ),
            other => panic!("[{layout}] unexpected destination layout: {other:?}"),
        }

        if layout == "filtered" {
            assert_eq!(
                ds.filter_pipeline().len(),
                2,
                "[{layout}] shuffle + deflate must be carried onto a dataset with \
                 no chunk to apply them to"
            );
        }
        if layout == "extensible" {
            assert_eq!(
                ds.maxshape().unwrap(),
                Some(vec![u64::MAX]),
                "[{layout}] resizability carried"
            );
            // The one case where the destination is not smaller than the source:
            // an extensible dataset keeps its eagerly built Extensible Array over
            // zero chunks, the same index this crate gives every empty resizable
            // dataset, because an in-place append needs the index to exist
            // already. That costs a few hundred bytes of index and still stores
            // no chunk, which is the part this test is about.
            assert!(
                matches!(
                    ds.chunk_index().unwrap(),
                    Some(hdf5_pure::ChunkIndex::ExtensibleArray)
                ),
                "[{layout}] a resizable destination keeps its growable index"
            );
        }
        drop(f);

        let dst_len = std::fs::metadata(&dst).unwrap().len();
        assert!(
            dst_len < MATERIALIZED,
            "[{layout}] the destination ({dst_len} B) still carries the \
             {MATERIALIZED} B of elements the source never stored"
        );

        // And the reference library reads back what it wrote in the first place.
        let c = hdf5::File::open(&dst).unwrap();
        assert_eq!(
            c.dataset("col").unwrap().read_raw::<i32>().unwrap(),
            vec![FILL; N],
            "[{layout}] the C library reads the repacked dataset"
        );
    }
}

/// The other side of the same predicate: a dataset that stores *some* of its
/// chunks is not "unallocated", and repack must still carry every element it
/// holds.
///
/// A predicate that answered "stores nothing" from the chunk count being less
/// than the grid would throw the written chunks away, and the neighbouring
/// sparse tests already catch that through the values they lose. What this adds
/// beside the positive case is the geometry those do not have: one *interior*
/// chunk of ten written, so there are holes on both sides of it rather than only
/// a tail.
///
/// It also records what repack does with the holes, which is not what the
/// never-written case does with its whole grid. A sparse source cannot take the
/// verbatim path, so it falls back to read-and-re-encode and the destination
/// stores every slot — one chunk in, ten out, 2,959 B to 4,309 B on this
/// fixture. That is unchanged by #293 and is the case its predicate must *not*
/// catch; asserting the count rather than merely that it is non-zero is what
/// keeps the two apart, since "not empty" is equally true of one chunk and ten.
#[test]
fn repack_keeps_the_chunks_a_partly_written_dataset_holds() {
    const N: usize = 1000;
    const FILL: i32 = 7;
    let dir = tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");
    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((N,))
            .chunk((100,))
            .fill_value(FILL)
            .create("col")
            .unwrap();
        // One chunk of the ten, so the dataset is neither empty nor dense.
        ds.write_slice(&[5i32; 100], 300..400).unwrap();
        file.close().unwrap();
    }

    let mut expected = vec![FILL; N];
    expected[300..400].fill(5);
    {
        let c = hdf5::File::open(&src).unwrap();
        assert_eq!(
            c.dataset("col").unwrap().read_raw::<i32>().unwrap(),
            expected,
            "C ground truth"
        );
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let ds = f.dataset("col").unwrap();
    assert_eq!(
        ds.read_i32().unwrap(),
        expected,
        "the written chunk must survive a repack"
    );
    assert_eq!(
        ds.chunks().unwrap().len(),
        10,
        "the sparse fallback re-encodes the whole grid: the source stored one \
         chunk of the ten and the destination stores all ten, which is the \
         behaviour a dataset that stores *something* still gets"
    );
    drop(f);

    let c = hdf5::File::open(&dst).unwrap();
    assert_eq!(
        c.dataset("col").unwrap().read_raw::<i32>().unwrap(),
        expected,
        "the C library reads the repacked dataset"
    );
}

/// A never-written dataset carrying a filter this crate cannot re-apply is still
/// refused by name.
///
/// The unallocated path emits no chunk, so there is nothing for a lossy filter to
/// perturb and a case could be made for letting it through. It is not made here:
/// the filters are reproduced by the same `carry_shape_and_pipeline` the
/// re-encode path uses, which reaches an `unreachable!` for any filter
/// `check_pipeline` was supposed to have refused. Widening what repack accepts is
/// a separate decision from preserving storage, so the refusal set is unchanged —
/// and a path that skipped the check would panic rather than refuse, which is
/// what this pins.
#[test]
fn repack_still_refuses_a_never_written_dataset_with_a_lossy_filter() {
    use hdf5::filters::ScaleOffset as CScaleOffset;

    let dir = tempdir().unwrap();
    let src = dir.path().join("c_unwritten_lossy.h5");
    let dst = dir.path().join("c_unwritten_lossy_repacked.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .create(&src)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape([2000])
            .chunk([512])
            .scale_offset(CScaleOffset::FloatDScale(3))
            .create("data")
            .unwrap();
        drop(ds);
        file.close().unwrap();
    }

    // Nothing was written, so this is the unallocated path rather than the
    // sparse-fallback one the neighbouring test covers.
    {
        let f = File::open(&src).unwrap();
        assert!(f.dataset("data").unwrap().chunks().unwrap().is_empty());
    }

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();
    match err {
        hdf5_pure::Error::RepackUnsupported(msg) => assert!(
            msg.contains("data") && msg.contains("scale-offset"),
            "error should name the dataset and reason: {msg}"
        ),
        other => panic!("expected RepackUnsupported, got {other:?}"),
    }
    assert!(!dst.exists(), "dst must not be created when repack refuses");
}

/// Filter ids, so a pipeline assertion reads as filters rather than as numbers.
const FILTER_DEFLATE: u16 = 1;
const FILTER_SHUFFLE: u16 = 2;
const FILTER_FLETCHER32: u16 = 3;

/// A dataset's pipeline as `(filter id, is_optional)`, in stored order — the
/// two properties issue #333 is about, and the pair `h5dump` and `H5Pget_filter2`
/// report.
fn pipeline_of(path: &std::path::Path, dataset: &str) -> Vec<(u16, bool)> {
    File::open(path)
        .unwrap()
        .dataset(dataset)
        .unwrap()
        .filter_pipeline()
        .iter()
        .map(|f| (f.id, f.is_optional))
        .collect()
}

/// A source pipeline's filter *order* and per-filter *optional* flags survive a
/// re-encoding repack (issue #333).
///
/// The order is not cosmetic. `shuffle -> fletcher32 -> deflate` checksums the
/// shuffled bytes; moving fletcher32 last checksums the deflated bytes instead,
/// so the destination verifies a different thing on read than the source did.
/// And a filter that loses its optional flag becomes mandatory, so a reader that
/// cannot apply it must now fail where it was previously allowed to skip.
///
/// Driven through the *sparse* chunked source that forces the re-encode
/// fallback: the dense verbatim chunk-copy path copies the source's pipeline
/// message byte-exact and never had this defect.
#[test]
fn c_pipeline_order_and_optional_flags_survive_re_encoding_repack() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_order.h5");
    let dst = dir.path().join("c_order_repacked.h5");

    let n = 2000usize; // chunk 512 -> 4 chunks; only the first 1000 written
    let written = 1000usize;
    let head: Vec<i32> = (1..=written as i32).collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([n])
            .chunk([512])
            .shuffle()
            .fletcher32()
            .deflate(4)
            .create("data")
            .unwrap();
        ds.write_slice(head.as_slice(), 0..written).unwrap();
        file.close().unwrap();
    }

    let source_pipeline = pipeline_of(&src, "data");

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    assert_eq!(
        pipeline_of(&dst, "data"),
        source_pipeline,
        "repack must reproduce the source's filter order and optional flags"
    );

    // The data still round-trips, through both readers.
    let f = File::open(&dst).unwrap();
    let ds = f.dataset("data").unwrap();
    let vals = ds.read_i32().unwrap();
    assert_eq!(&vals[..written], head.as_slice());
    assert!(vals[written..].iter().all(|&v| v == 0));

    let c = hdf5::File::open(&dst).unwrap();
    let c_vals = c.dataset("data").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(&c_vals[..written], head.as_slice());
}

/// The same, through a re-encoding path that writes no chunks at all: a dataset
/// the C library created and never wrote to (issue #293), whose storage repack
/// reproduces as storage rather than materializing.
///
/// The pipeline is decoded and re-applied by the same `check_pipeline` +
/// `carry_shape_and_pipeline` pair as every other re-encoding path, so this is
/// not separate coverage of *that*. What it covers is the emit path afterwards:
/// the message has to survive `with_unallocated_storage`, a writer that encodes
/// nothing for it to describe. It is also the one test here that pins the C
/// library's pipeline as a literal rather than comparing destination to source.
#[test]
fn a_never_written_dataset_keeps_its_pipeline_order_and_optional_flags() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_unwritten.h5");
    let dst = dir.path().join("c_unwritten_repacked.h5");

    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<i32>()
            .shape([2000])
            .chunk([512])
            .shuffle()
            .fletcher32()
            .deflate(4)
            .create("data")
            .unwrap();
        file.close().unwrap();
    }

    let source = pipeline_of(&src, "data");
    assert_eq!(
        source,
        [
            (FILTER_SHUFFLE, true),
            (FILTER_FLETCHER32, false),
            (FILTER_DEFLATE, true)
        ],
        "the C library did not write the pipeline this test is about"
    );

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    assert_eq!(
        pipeline_of(&dst, "data"),
        source,
        "repack must reproduce the source's filter order and optional flags"
    );
}

/// And through `emit_vlen_string_dataset`, the third kind of re-encoding path.
///
/// A variable-length string dataset is never chunk-copied verbatim — its
/// elements are heap references that go stale on rewrite, so every one is
/// re-staged and re-encoded, whatever its layout. That routes it through the
/// same `check_pipeline` + `carry_shape_and_pipeline` pair, so it lost its
/// source's filter order and optional flags exactly as the fixed-size paths did
/// (issue #333). The existing variable-length repack tests could not see it:
/// they build their sources through this crate's own writer, which emits one
/// canonically ordered, all-mandatory pipeline.
// Before 2.0 the C library refuses a mandatory filter, fletcher32 here, on a
// variable-length dataset.
#[cfg(feature = "__hdf5-2")]
#[test]
fn a_vlen_string_dataset_keeps_its_pipeline_order_and_optional_flags() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("c_vlen_order.h5");
    let dst = dir.path().join("c_vlen_order_repacked.h5");

    let words: Vec<hdf5::types::VarLenUnicode> = (0..64)
        .map(|i| format!("value-{i}").parse().unwrap())
        .collect();
    {
        let file = hdf5::File::create(&src).unwrap();
        file.new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([words.len()])
            .chunk([8])
            .shuffle()
            .fletcher32()
            .deflate(4)
            .create("data")
            .unwrap()
            .write(&words)
            .unwrap();
        file.close().unwrap();
    }

    let source = pipeline_of(&src, "data");
    assert_eq!(
        source,
        [
            (FILTER_SHUFFLE, true),
            (FILTER_FLETCHER32, false),
            (FILTER_DEFLATE, true)
        ],
        "the C library did not write the pipeline this test is about"
    );

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    assert_eq!(
        pipeline_of(&dst, "data"),
        source,
        "repack must reproduce the source's filter order and optional flags"
    );

    // And the strings themselves survive the re-staging, through both readers.
    let f = File::open(&dst).unwrap();
    let read: Vec<String> = f
        .dataset("data")
        .unwrap()
        .read_vlen_strings(Default::default())
        .unwrap();
    let expected: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    assert_eq!(read, expected);

    let c = hdf5::File::open(&dst).unwrap();
    let c_read = c
        .dataset("data")
        .unwrap()
        .read_raw::<hdf5::types::VarLenUnicode>()
        .unwrap();
    assert_eq!(
        c_read.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
        expected
    );
}

/// The attribute encodings this crate writes only through a crate-private seam:
/// a fixed-width string with null termination and a Null dataspace, which the C
/// library's own API cannot express. `src/repack.rs` commits its source file as
/// `tests/data/pure/exotic_attributes.h5` and checks the fixture still
/// holds them. Repacked through the public API, the C library finds every
/// attribute it found in the source, and can open each.
#[test]
fn the_encodings_only_this_crate_writes_survive_a_repack_for_the_c_library() {
    let fixture = test_util::data("pure/exotic_attributes.h5");
    let dir = tempfile::tempdir().unwrap();
    let repacked = dir.path().join("repacked.h5");
    hdf5_pure::repack(&fixture, &repacked, &hdf5_pure::RepackOptions::new()).unwrap();

    let source = hdf5::File::open(&fixture).unwrap();
    let repacked = hdf5::File::open(&repacked).unwrap();
    let names = |file: &hdf5::File| {
        let dataset = file.dataset("data").unwrap();
        (file.attr_names().unwrap(), dataset.attr_names().unwrap())
    };
    let (root, dataset) = names(&source);
    assert!(
        !root.is_empty() && !dataset.is_empty(),
        "the fixture holds no attributes"
    );
    assert_eq!(names(&repacked), (root.clone(), dataset.clone()));
    // Opening reads the datatype and dataspace, which is where a malformed one
    // is caught.
    for name in &root {
        repacked
            .attr(name)
            .unwrap_or_else(|e| panic!("the C library could not open {name:?}: {e}"));
    }
    let data = repacked.dataset("data").unwrap();
    for name in &dataset {
        data.attr(name)
            .unwrap_or_else(|e| panic!("the C library could not open data/{name:?}: {e}"));
    }
}
