// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! The format boundary against the linked release of the C library, in both
//! directions.
//!
//! This crate writes one file per format bound and the C library reads each
//! back. A 1.8 library must reject the 1.10 format, whose version 3 superblock
//! it does not know, and open the 1.8 format. Every later release must open
//! both. The other way, the C library writes the same content under the oldest
//! and the newest format it has, and this crate reads each back.
//!
//! This is the one crosscheck that compiles against every release, 1.8
//! included, so the bounds on the C side go through the raw property list
//! call. The wrapper's builder for them is gated to 1.10.2.

use std::path::Path;

use hdf5::plist::FileAccess;
use hdf5::types::{FixedAscii, VarLenArray};
use hdf5::{H5Type, ObjectReference, ObjectReference1, ReferencedObject};
use hdf5_pure::mat::{self, Options};
use hdf5_pure::{AttrValue, FileBuilder, LibVer, make_i32_type};
use hdf5_sys::h5f::{
    H5F_ACC_TRUNC, H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST, H5F_libver_t, H5Fcreate,
};
use hdf5_sys::h5p::{H5P_DEFAULT, H5Pset_libver_bounds};
use serde::Serialize;
use tempfile::tempdir;

/// The superblock version byte. A `.mat` file carries a 512-byte user block
/// ahead of the signature, so it is searched for.
fn superblock_version(path: &Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    let signature = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .unwrap_or_else(|| panic!("{}: no signature", path.display()));
    bytes[signature + 8]
}

#[derive(Serialize)]
struct Inner {
    count: u32,
    flag: bool,
}

/// The `.mat` content. The ragged member interns objects under `#refs#`, the
/// struct sequence interns a group per element, the `None` interns an empty
/// marker, and the complex member is a compound datatype.
#[derive(Serialize)]
struct Demo {
    values: Vec<f64>,
    label: String,
    nested: Inner,
    empty: Vec<f64>,
    ragged: Vec<Vec<i32>>,
    records: Vec<Inner>,
    optional: Vec<Option<f64>>,
    #[serde(serialize_with = "mat::complex::f64_array")]
    signal: Vec<mat::Complex64>,
}

fn demo() -> Demo {
    Demo {
        values: vec![1.0, 2.0, 3.0],
        label: "demo".to_string(),
        nested: Inner {
            count: 7,
            flag: true,
        },
        empty: Vec::new(),
        ragged: vec![vec![1], vec![2, 3]],
        records: vec![
            Inner {
                count: 11,
                flag: false,
            },
            Inner {
                count: 13,
                flag: true,
            },
        ],
        optional: vec![Some(1.5), None],
        signal: vec![
            mat::Complex64::new(1.0, -2.0),
            mat::Complex64::new(0.5, 0.25),
        ],
    }
}

/// A plain `.h5`: attributes on all three kinds of object, a committed
/// datatype, a rank-3 dataspace, dense attributes and a group of many links.
/// The last three have 1.8 encodings that differ from the 1.10 ones.
fn write_plain(path: &Path, libver: LibVer) {
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, libver);
    b.set_attr("root_attr", AttrValue::AsciiString("r".into()));
    b.create_dataset("values")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .set_attr("units", AttrValue::AsciiString("m/s".into()));
    b.commit_datatype("reading_t", make_i32_type());
    b.create_dataset("typed")
        .with_i32_data(&[3, 1, 4])
        .with_committed_datatype("reading_t")
        .set_attr_committed("baseline", AttrValue::I32(9), "reading_t");
    b.create_dataset("cube")
        .with_i32_data(&(0..24).collect::<Vec<_>>())
        .with_shape(&[2, 3, 4]);
    let dense = b.create_dataset("dense_attrs");
    dense.with_i32_data(&[1, 2]);
    for i in 0..12 {
        dense.set_attr(&format!("a{i:02}"), AttrValue::I32(i));
    }
    let mut wide = b.create_group("wide");
    for i in 0..64 {
        wide.create_dataset(&format!("m{i:03}")).with_i32_data(&[i]);
    }
    b.add_group(wide.finish());
    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::I64(7));
    g.create_dataset("inner").with_i32_data(&[7, 8]);
    b.add_group(g.finish());
    b.write(path).expect("write the plain fixture");
}

/// A dataset's numbers as `f64`, read with the type the file declares.
fn numbers(dataset: &hdf5::Dataset) -> Vec<f64> {
    use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
    match dataset.dtype().unwrap().to_descriptor().unwrap() {
        TypeDescriptor::Float(FloatSize::U8) => dataset.read_raw::<f64>().unwrap(),
        TypeDescriptor::Float(FloatSize::U4) => dataset
            .read_raw::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
        TypeDescriptor::Integer(IntSize::U4) => dataset
            .read_raw::<i32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
        TypeDescriptor::Integer(IntSize::U8) => dataset
            .read_raw::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        TypeDescriptor::Unsigned(IntSize::U4) => dataset
            .read_raw::<u32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
        TypeDescriptor::Unsigned(IntSize::U8) => dataset
            .read_raw::<u64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        TypeDescriptor::Unsigned(IntSize::U1) => dataset
            .read_raw::<u8>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
        other => panic!("{}: unexpected element type {other:?}", dataset.name()),
    }
}

fn assert_numbers(file: &hdf5::File, path: &str, want: &[f64]) {
    let dataset = file
        .dataset(path)
        .unwrap_or_else(|e| panic!("the C library opens {path}: {e}"));
    assert_eq!(numbers(&dataset), want, "{path}");
}

fn fixed_string(location: &hdf5::Location, attr: &str) -> String {
    location
        .attr(attr)
        .unwrap_or_else(|e| panic!("the C library opens attribute {attr}: {e}"))
        .read_scalar::<FixedAscii<64>>()
        .unwrap_or_else(|e| panic!("the C library reads attribute {attr}: {e}"))
        .as_str()
        .to_string()
}

/// The paths the object references in `path` resolve to.
fn targets(file: &hdf5::File, path: &str) -> Vec<String> {
    file.dataset(path)
        .unwrap_or_else(|e| panic!("the C library opens {path}: {e}"))
        .read_raw::<ObjectReference1>()
        .unwrap_or_else(|e| panic!("the C library reads the references in {path}: {e}"))
        .iter()
        .map(|r| match r.dereference(file).unwrap() {
            ReferencedObject::Group(g) => g.name(),
            ReferencedObject::Dataset(d) => d.name(),
            ReferencedObject::Datatype(_) => panic!("{path} references a datatype"),
        })
        .collect()
}

/// What the C library must find in a plain fixture.
fn check_plain(c: &hdf5::File) {
    assert_numbers(c, "values", &[1.0, 2.0, 3.0]);
    assert_eq!(fixed_string(&c.dataset("values").unwrap(), "units"), "m/s");
    assert_eq!(fixed_string(c, "root_attr"), "r");
    let grp = c.group("grp").unwrap();
    assert_eq!(grp.attr("tag").unwrap().read_scalar::<i64>().unwrap(), 7);
    assert_numbers(c, "grp/inner", &[7.0, 8.0]);

    c.committed_datatype("reading_t")
        .expect("the C library opens the committed datatype");
    let typed = c.dataset("typed").unwrap();
    assert!(
        typed.dtype().unwrap().is_committed(),
        "typed names the committed type"
    );
    assert_numbers(c, "typed", &[3.0, 1.0, 4.0]);
    assert_eq!(
        typed
            .attr("baseline")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        9
    );

    let cube = c.dataset("cube").unwrap();
    assert_eq!(cube.shape(), vec![2, 3, 4]);
    assert_eq!(numbers(&cube), (0..24).map(f64::from).collect::<Vec<_>>());

    let dense = c.dataset("dense_attrs").unwrap();
    assert_eq!(dense.attr_names().unwrap().len(), 12);
    assert_eq!(dense.attr("a00").unwrap().read_scalar::<i32>().unwrap(), 0);
    assert_eq!(dense.attr("a11").unwrap().read_scalar::<i32>().unwrap(), 11);

    assert_eq!(c.group("wide").unwrap().member_names().unwrap().len(), 64);
    assert_numbers(c, "wide/m000", &[0.0]);
    assert_numbers(c, "wide/m063", &[63.0]);
}

/// A `#refs#` path as the `.mat` writer names them.
fn r(n: u64) -> String {
    format!("/#refs#/ref_{n:016x}")
}

/// What the C library must find in the `.mat` written from `demo()`.
fn check_mat(c: &hdf5::File) {
    assert_numbers(c, "values", &[1.0, 2.0, 3.0]);
    assert_numbers(c, "nested/count", &[7.0]);
    assert_numbers(c, "empty", &[0.0, 0.0]);

    assert_eq!(targets(c, "ragged"), [r(0), r(1)]);
    assert_eq!(targets(c, "records"), [r(2), r(3)]);
    assert_eq!(targets(c, "optional"), [r(4), r(5)]);
    assert_numbers(c, &r(0), &[1.0]);
    assert_numbers(c, &r(1), &[2.0, 3.0]);
    assert_numbers(c, &format!("{}/count", r(2)), &[11.0]);
    assert_numbers(c, &format!("{}/count", r(3)), &[13.0]);
    assert_numbers(c, &r(4), &[1.5]);
    assert_numbers(c, &r(5), &[0.0, 0.0]);
    for n in 0..6 {
        let object = c.group(&r(n)).map(|g| (*g).clone());
        let object = object.unwrap_or_else(|_| (**c.dataset(&r(n)).unwrap()).clone());
        assert_eq!(fixed_string(&object, "H5PATH"), r(n), "H5PATH of {}", r(n));
    }

    // `MATLAB_fields` is one variable-length array of single characters per
    // field name.
    let fields: Vec<String> = c
        .group(&r(2))
        .unwrap()
        .attr("MATLAB_fields")
        .unwrap()
        .read_raw::<VarLenArray<FixedAscii<1>>>()
        .expect("the C library reads MATLAB_fields")
        .iter()
        .map(|chars| chars.iter().map(|ch| ch.as_str().to_string()).collect())
        .collect();
    assert_eq!(fields, ["count", "flag"]);

    let signal = c.dataset("signal").unwrap();
    let descriptor = signal.dtype().unwrap().to_descriptor().unwrap();
    let hdf5::types::TypeDescriptor::Compound(compound) = descriptor else {
        panic!("signal is not a compound datatype: {descriptor:?}");
    };
    assert_eq!(compound.fields.len(), 2);
    let values: Vec<f64> = signal
        .read_raw::<Complex>()
        .unwrap()
        .into_iter()
        .flat_map(|z| [z.real, z.imag])
        .collect();
    assert_eq!(values, [1.0, -2.0, 0.5, 0.25]);
}

/// The compound the `.mat` writer uses for complex values.
#[derive(H5Type, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Complex {
    real: f64,
    imag: f64,
}

/// What the C library must find in the `.mat` written with the `string`
/// class: the MCOS subsystem, and one `#refs#` object without an `H5PATH`.
fn check_mat_string(c: &hdf5::File) {
    assert_numbers(c, "label", &[3707764736.0, 2.0, 1.0, 1.0, 1.0, 1.0]);
    assert_numbers(c, &r(12), &[0.0, 0.0]);
    assert_eq!(targets(c, &r(15)), [r(13), r(14)]);
    let canonical = c.dataset(&r(8)).unwrap();
    assert_eq!(fixed_string(&canonical, "MATLAB_class"), "canonical empty");
    assert!(
        !canonical
            .attr_names()
            .unwrap()
            .iter()
            .any(|a| a == "H5PATH"),
        "{} carries no H5PATH",
        r(8)
    );
}

#[test]
fn the_1_8_format_this_crate_writes_opens_in_every_release() {
    let dir = tempdir().unwrap();
    let plain = dir.path().join("plain_v18.h5");
    write_plain(&plain, LibVer::V18);
    let mat = dir.path().join("mat_v18.mat");
    mat::to_file(&demo(), &mat).unwrap();
    let string = dir.path().join("mat_string_v18.mat");
    let mut options = Options::default();
    options.string_class = mat::StringClass::String;
    mat::to_file_with_options(&demo(), &string, &options).unwrap();

    for path in [&plain, &mat, &string] {
        assert_eq!(superblock_version(path), 2, "{}", path.display());
    }
    check_plain(&hdf5::File::open(&plain).expect("the C library opens the 1.8 plain file"));
    check_mat(&hdf5::File::open(&mat).expect("the C library opens the 1.8 mat file"));
    check_mat_string(
        &hdf5::File::open(&string).expect("the C library opens the 1.8 string-class file"),
    );
}

/// The plain and `.mat` fixtures in the 1.10 format, both with a version 3
/// superblock.
fn write_v110(dir: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let plain = dir.join("plain_v110.h5");
    write_plain(&plain, LibVer::V110);
    let mat = dir.join("mat_v110.mat");
    let mut options = Options::default();
    options.libver = LibVer::V110;
    mat::to_file_with_options(&demo(), &mat, &options).unwrap();
    for path in [&plain, &mat] {
        assert_eq!(superblock_version(path), 3, "{}", path.display());
    }
    (plain, mat)
}

#[cfg(feature = "__hdf5-1.10")]
#[test]
fn the_1_10_format_this_crate_writes_opens_from_1_10_on() {
    let dir = tempdir().unwrap();
    let (plain, mat) = write_v110(dir.path());
    check_plain(&hdf5::File::open(&plain).expect("the C library opens the 1.10 plain file"));
    check_mat(&hdf5::File::open(&mat).expect("the C library opens the 1.10 mat file"));
}

#[cfg(not(feature = "__hdf5-1.10"))]
#[test]
fn the_1_10_format_this_crate_writes_is_rejected_before_1_10() {
    let dir = tempdir().unwrap();
    let (plain, mat) = write_v110(dir.path());
    for path in [&plain, &mat] {
        assert!(
            hdf5::File::open(path).is_err(),
            "HDF5 {:?} opened {}, which carries a version 3 superblock",
            hdf5::library_version(),
            path.display()
        );
    }
}

/// The C library's file with the given library bounds.
///
/// Created through the raw call: the wrapper's file builder copies a property
/// list into its own fields, and on 1.8 it has none for the bounds.
fn c_create(path: &Path, low: H5F_libver_t, high: H5F_libver_t) -> hdf5::File {
    let fapl = FileAccess::try_new().expect("a file access property list");
    let name = std::ffi::CString::new(path.to_str().expect("a temp path is UTF-8")).unwrap();
    // TODO: https://github.com/metno/hdf5-rust/issues/226
    // Safety: live ids, bounds the C library defines, and a file id handed to
    // the wrapper, which closes it.
    unsafe {
        assert_eq!(
            H5Pset_libver_bounds(fapl.id(), low, high),
            0,
            "H5Pset_libver_bounds"
        );
        let id = H5Fcreate(name.as_ptr(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl.id());
        assert!(id > 0, "the C library creates {}", path.display());
        hdf5::from_id::<hdf5::File>(id).expect("a file handle")
    }
}

/// The C library's version of the plain fixture, plus a reference dataset and
/// a compound dataset, which this crate's whole-file writer covers through
/// `.mat` output instead.
fn c_write(path: &Path, low: H5F_libver_t, high: H5F_libver_t) {
    let file = c_create(path, low, high);
    file.new_attr::<FixedAscii<1>>()
        .shape(())
        .create("root_attr")
        .unwrap()
        .write_scalar(&FixedAscii::<1>::from_ascii("r").unwrap())
        .unwrap();
    let values = file
        .new_dataset::<f64>()
        .shape([3])
        .create("values")
        .unwrap();
    values.write(&[1.0, 2.0, 3.0]).unwrap();
    values
        .new_attr::<FixedAscii<3>>()
        .shape(())
        .create("units")
        .unwrap()
        .write_scalar(&FixedAscii::<3>::from_ascii("m/s").unwrap())
        .unwrap();

    let reading_t = hdf5::Datatype::from_type::<i32>().unwrap();
    file.commit_datatype("reading_t", &reading_t).unwrap();
    let typed = file
        .new_dataset_builder()
        .empty_as(&reading_t)
        .shape([3])
        .create("typed")
        .unwrap();
    typed.write(&[3i32, 1, 4]).unwrap();
    typed
        .new_attr::<i32>()
        .shape(())
        .create("baseline")
        .unwrap()
        .write_scalar(&9i32)
        .unwrap();

    file.new_dataset::<i32>()
        .shape([2, 3, 4])
        .create("cube")
        .unwrap()
        .write_raw(&(0..24).collect::<Vec<i32>>())
        .unwrap();

    let dense = file
        .new_dataset::<i32>()
        .shape([2])
        .create("dense_attrs")
        .unwrap();
    dense.write(&[1i32, 2]).unwrap();
    for i in 0..12i32 {
        dense
            .new_attr::<i32>()
            .shape(())
            .create(format!("a{i:02}").as_str())
            .unwrap()
            .write_scalar(&i)
            .unwrap();
    }

    let wide = file.create_group("wide").unwrap();
    for i in 0..64i32 {
        wide.new_dataset::<i32>()
            .shape([1])
            .create(format!("m{i:03}").as_str())
            .unwrap()
            .write(&[i])
            .unwrap();
    }

    let grp = file.create_group("grp").unwrap();
    grp.new_attr::<i64>()
        .shape(())
        .create("tag")
        .unwrap()
        .write_scalar(&7i64)
        .unwrap();
    grp.new_dataset::<i32>()
        .shape([2])
        .create("inner")
        .unwrap()
        .write(&[7i32, 8])
        .unwrap();

    let refs = [
        ObjectReference1::create(&file, "grp").unwrap(),
        ObjectReference1::create(&file, "values").unwrap(),
    ];
    file.new_dataset::<ObjectReference1>()
        .shape([2])
        .create("refs")
        .unwrap()
        .write(&refs)
        .unwrap();

    file.new_dataset::<Complex>()
        .shape([2])
        .create("signal")
        .unwrap()
        .write(&[
            Complex {
                real: 1.0,
                imag: -2.0,
            },
            Complex {
                real: 0.5,
                imag: 0.25,
            },
        ])
        .unwrap();
    file.close().unwrap();
}

/// What this crate must find in the C library's file, the compound aside.
fn check_c_written(path: &Path) {
    let f =
        hdf5_pure::File::open(path).unwrap_or_else(|e| panic!("open {}: {e:?}", path.display()));
    assert_eq!(
        f.dataset("values").unwrap().read_f64().unwrap(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("values").unwrap().attrs().unwrap().get("units"),
        Some(&AttrValue::AsciiString("m/s".into()))
    );
    assert_eq!(
        f.root().attrs().unwrap().get("root_attr"),
        Some(&AttrValue::AsciiString("r".into()))
    );
    let grp = f.group("grp").unwrap();
    assert_eq!(grp.attrs().unwrap().get("tag"), Some(&AttrValue::I64(7)));
    assert_eq!(f.dataset("grp/inner").unwrap().read_i32().unwrap(), [7, 8]);

    assert_eq!(f.root().named_datatypes().unwrap(), ["reading_t"]);
    let typed = f.dataset("typed").unwrap();
    assert_eq!(typed.read_i32().unwrap(), [3, 1, 4]);
    assert_eq!(
        typed.attrs().unwrap().get("baseline"),
        Some(&AttrValue::I32(9))
    );

    let cube = f.dataset("cube").unwrap();
    assert_eq!(cube.shape().unwrap(), [2, 3, 4]);
    assert_eq!(cube.read_i32().unwrap(), (0..24).collect::<Vec<_>>());

    let dense = f.dataset("dense_attrs").unwrap().attrs().unwrap();
    assert_eq!(dense.len(), 12);
    assert_eq!(dense.get("a00"), Some(&AttrValue::I32(0)));
    assert_eq!(dense.get("a11"), Some(&AttrValue::I32(11)));

    let wide = f.group("wide").unwrap();
    assert_eq!(wide.datasets().unwrap().len(), 64);
    assert_eq!(f.dataset("wide/m063").unwrap().read_i32().unwrap(), [63]);

    let refs = f.dataset("refs").unwrap().dereference().unwrap();
    assert_eq!(refs.len(), 2);
    assert!(
        matches!(refs[0], hdf5_pure::Object::Group(_)),
        "refs[0] is the group"
    );
    match &refs[1] {
        hdf5_pure::Object::Dataset(d) => assert_eq!(d.read_f64().unwrap(), [1.0, 2.0, 3.0]),
        other => panic!("refs[1] is {other:?}, not the values dataset"),
    }
}

/// The compound in the C library's file, read through its declared members.
fn check_compound(path: &Path) {
    let f = hdf5_pure::File::open(path).unwrap();
    let signal = f.dataset("signal").unwrap();
    let hdf5_pure::Datatype::Compound { members, .. } = signal.datatype().unwrap() else {
        panic!("signal is not a compound datatype");
    };
    assert_eq!(members.len(), 2);
    let bytes = signal.read_raw().unwrap();
    let (pairs, rest) = bytes.as_chunks::<8>();
    assert!(rest.is_empty(), "signal holds whole f64 values");
    let values: Vec<f64> = pairs
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect();
    assert_eq!(values, [1.0, -2.0, 0.5, 0.25]);
}

#[test]
fn this_crate_reads_the_oldest_format_the_release_writes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oldest.h5");
    c_write(&path, H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST);
    assert_eq!(superblock_version(&path), 0);
    check_c_written(&path);
    check_compound(&path);
}

#[cfg(not(feature = "__hdf5-1.10"))]
#[test]
fn this_crate_reads_the_newest_format_the_release_writes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("newest.h5");
    c_write(&path, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    assert_eq!(superblock_version(&path), 2);
    check_c_written(&path);
    check_compound(&path);
}

#[cfg(all(feature = "__hdf5-1.10", not(feature = "__hdf5-2")))]
#[test]
fn this_crate_reads_the_newest_format_the_release_writes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("newest.h5");
    c_write(&path, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    assert_eq!(superblock_version(&path), 3);
    check_c_written(&path);
    check_compound(&path);
}

/// Under its latest bounds, HDF5 2.0 encodes the compound with datatype
/// message version 5, which this crate does not read.
// TODO: read datatype message version 5, and fold this into the test above.
#[cfg(feature = "__hdf5-2")]
#[test]
fn this_crate_reads_the_newest_format_the_release_writes_except_its_datatype_version_5() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("newest.h5");
    c_write(&path, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    assert_eq!(superblock_version(&path), 3);
    check_c_written(&path);
    let refused = hdf5_pure::File::open(&path)
        .unwrap()
        .dataset("signal")
        .unwrap()
        .datatype();
    assert!(
        matches!(
            refused,
            Err(hdf5_pure::Error::Format(
                hdf5_pure::FormatError::InvalidDatatypeVersion { .. }
            ))
        ),
        "signal: {refused:?}"
    );
}
