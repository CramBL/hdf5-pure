//! Write one file in each on-disk format this crate produces, for checking
//! against an HDF5 1.8 library.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example libver_fixtures --features serde -- <out-dir>
//! ```
//!
//! `scripts/check-hdf5-18.sh` drives this; it is a separate program rather than
//! a test because the thing it feeds is an external toolchain that cannot be a
//! dev-dependency (see that script for why).
//!
//! Both files hold the same content, so any difference an old library reports
//! between them is the format and nothing else.

use hdf5_pure::mat::{self, Options};
use hdf5_pure::{AttrValue, FileBuilder, LibVer, make_i32_type};
use serde::Serialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct Demo {
    values: Vec<f64>,
    label: String,
    nested: Inner,
    empty: Vec<f64>,
    /// Ragged, so it lowers to a cell array rather than a matrix. This is the
    /// only shape that interns objects under `#refs#`, where each one carries
    /// an `H5PATH` attribute and the parent dataset holds object references
    /// rather than data — two things no other fixture here puts in front of an
    /// old library.
    ragged: Vec<Vec<i32>>,
    /// A sequence of structs interns a *group* per element, so `H5PATH` lands
    /// on a group rather than a dataset, and `MATLAB_fields` — the only
    /// variable-length string array this crate writes, and so the only
    /// attribute of its kind an old library has to decode here — appears on an
    /// object under `#refs#`.
    records: Vec<Inner>,
    /// The `None` slot interns a `struct([])` empty marker of its own, which is
    /// an empty marker in a position no other fixture puts one.
    optional: Vec<Option<f64>>,
    /// Complex, so the file carries a compound datatype: two named members an
    /// old library has to walk rather than a single scalar encoding.
    #[serde(serialize_with = "mat::complex::f64_array")]
    signal: Vec<mat::Complex64>,
}

#[derive(Serialize)]
struct Inner {
    count: u32,
    flag: bool,
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

/// A plain `.h5` alongside the `.mat` pair: attributes on all three kinds of
/// object, which is what the attribute-count fix touches, and a committed
/// datatype, which is an object kind of its own.
fn write_h5(path: &Path, libver: LibVer) {
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, libver);
    b.set_attr("root_attr", AttrValue::AsciiString("r".into()));
    b.create_dataset("values")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .set_attr("units", AttrValue::AsciiString("m/s".into()));

    // A committed (`H5Tcommit`) datatype, named by a dataset and by an
    // attribute. Its users carry a reference to its object header in place of an
    // encoding, and the object itself carries a reference count — a shape no
    // other fixture here writes, and one an old library has to decode rather
    // than merely skip. A reference it cannot follow costs more than the type's
    // name: the C library abandons an object's whole attribute list when one
    // attribute fails to decode, so the repack count below sees it too.
    b.commit_datatype("reading_t", make_i32_type());
    b.create_dataset("typed")
        .with_i32_data(&[3, 1, 4])
        .with_committed_datatype("reading_t")
        .set_attr_committed("baseline", AttrValue::I32(9), "reading_t");

    // A rank-3 dataspace, dense attributes and a group of many links: three
    // shapes whose 1.8 encodings differ from the 1.10 ones this crate also
    // writes. Twelve attributes is past the point where a set moves out of the
    // object header into a fractal heap and a v2 B-tree.
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
    let wide = wide.finish();
    b.add_group(wide);

    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::I64(7));
    g.create_dataset("inner").with_i32_data(&[7, 8]);
    let g = g.finish();
    b.add_group(g);
    b.write(path).expect("write h5 fixture");
}

/// The expectations `scripts/verify_fixtures.py` checks, written here so a
/// fixture and the values it is checked against come from one place.
fn checks() -> Vec<Value> {
    let plain = "plain_v18.h5";
    let mat_file = "mat_v18.mat";
    let string_file = "mat_string_v18.mat";
    // The 1.10 pair holds the same content as the 1.8 pair, so a library that
    // opens both must read the same values out of each.
    let plain_v110 = "plain_v110.h5";
    let mat_v110_file = "mat_v110.mat";
    let r = |n: u64| format!("/#refs#/ref_{n:016x}");

    let mut checks = vec![
        json!({"kind": "data", "file": plain, "path": "/values", "data": [1.0, 2.0, 3.0]}),
        json!({"kind": "data", "file": plain, "path": "/grp/inner", "data": [7, 8]}),
        json!({"kind": "data", "file": plain, "path": "/typed", "data": [3, 1, 4]}),
        json!({"kind": "data", "file": plain, "path": "/cube", "data": (0..24).collect::<Vec<i32>>(), "dims": [2, 3, 4]}),
        json!({"kind": "data", "file": plain, "path": "/dense_attrs", "data": [1, 2]}),
        json!({"kind": "data", "file": plain, "path": "/wide/m000", "data": [0]}),
        json!({"kind": "data", "file": plain, "path": "/wide/m063", "data": [63]}),
        json!({"kind": "attr", "file": plain, "path": "/values", "name": "units", "data": "m/s"}),
        json!({"kind": "attr", "file": plain, "path": "/", "name": "root_attr", "data": "r"}),
        json!({"kind": "attr", "file": plain, "path": "/grp", "name": "tag", "data": 7}),
        json!({"kind": "attr", "file": plain, "path": "/typed", "name": "baseline", "data": 9}),
        json!({"kind": "attr", "file": plain, "path": "/dense_attrs", "name": "a00", "data": 0}),
        json!({"kind": "attr", "file": plain, "path": "/dense_attrs", "name": "a11", "data": 11}),
        json!({"kind": "attrs", "file": plain, "path": "/dense_attrs", "count": 12}),
        json!({"kind": "links", "file": plain, "path": "/wide", "count": 64}),
        json!({"kind": "named_type", "file": plain, "path": "/reading_t"}),
        json!({"kind": "data", "file": mat_file, "path": "/values", "data": [1.0, 2.0, 3.0]}),
        json!({"kind": "data", "file": mat_file, "path": "/nested/count", "data": [7]}),
        json!({"kind": "data", "file": mat_file, "path": "/empty", "data": [0, 0]}),
        json!({"kind": "refs", "file": mat_file, "path": "/ragged", "targets": [r(0), r(1)]}),
        json!({"kind": "refs", "file": mat_file, "path": "/records", "targets": [r(2), r(3)]}),
        json!({"kind": "refs", "file": mat_file, "path": "/optional", "targets": [r(4), r(5)]}),
        json!({"kind": "data", "file": mat_file, "path": r(0), "data": [1]}),
        json!({"kind": "data", "file": mat_file, "path": r(1), "data": [2, 3]}),
        json!({"kind": "data", "file": mat_file, "path": format!("{}/count", r(2)), "data": [11]}),
        json!({"kind": "data", "file": mat_file, "path": format!("{}/count", r(3)), "data": [13]}),
        json!({"kind": "data", "file": mat_file, "path": r(4), "data": [1.5]}),
        json!({"kind": "data", "file": mat_file, "path": r(5), "data": [0, 0]}),
        json!({"kind": "attr", "file": mat_file, "path": r(2), "name": "MATLAB_fields", "data": ["count", "flag"]}),
        json!({"kind": "data", "file": mat_file, "path": "/signal", "data": [1.0, -2.0, 0.5, 0.25]}),
        json!({"kind": "members", "file": mat_file, "path": "/signal", "count": 2}),
        json!({"kind": "data", "file": string_file, "path": "/label", "data": [3707764736u32, 2, 1, 1, 1, 1]}),
        json!({"kind": "data", "file": string_file, "path": r(12), "data": [0, 0]}),
        json!({"kind": "refs", "file": string_file, "path": r(15), "targets": [r(13), r(14)]}),
        json!({"kind": "attr", "file": string_file, "path": r(8), "name": "MATLAB_class", "data": "canonical empty"}),
        json!({"kind": "no_attr", "file": string_file, "path": r(8), "name": "H5PATH"}),
    ];

    checks.extend([
        json!({"kind": "data", "file": plain_v110, "path": "/values", "data": [1.0, 2.0, 3.0]}),
        json!({"kind": "data", "file": plain_v110, "path": "/grp/inner", "data": [7, 8]}),
        json!({"kind": "data", "file": mat_v110_file, "path": "/values", "data": [1.0, 2.0, 3.0]}),
    ]);

    // Every interned object carries its own path as `H5PATH`, which is what
    // MATLAB reads back to name it.
    for n in 0..6 {
        checks.push(
            json!({"kind": "attr", "file": mat_file, "path": r(n), "name": "H5PATH", "data": r(n)}),
        );
    }
    checks
}

fn superblock_version(path: &Path) -> u8 {
    let bytes = std::fs::read(path).expect("read fixture");
    let sig = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("fixture carries an HDF5 signature");
    bytes[sig + 8]
}

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "libver_fixtures".to_string()),
    );
    std::fs::create_dir_all(&out).expect("create output dir");

    // The MAT default since 0.34: the HDF5 1.8 format.
    let mat_v18 = out.join("mat_v18.mat");
    mat::to_file(&demo(), &mat_v18).expect("write 1.8 mat");

    // What every release through 0.33.0 wrote.
    let mat_v110 = out.join("mat_v110.mat");
    let mut opts = Options::default();
    opts.libver = LibVer::V110;
    mat::to_file_with_options(&demo(), &mat_v110, &opts).expect("write 1.10 mat");

    // The `string` class, in the 1.8 format. Two things reach an old library
    // only through this file: the MCOS subsystem — a `#subsystem#` group, a
    // `FileWrapper__` metadata blob, reference-array templates, and the one
    // `#refs#` object that carries no `H5PATH` — and the builder-backed
    // emitter, since `to_file` above goes through the walker instead and the
    // builder's only other output here is the 1.10 file, which 1.8 refuses by
    // design and so never reads.
    let mat_string_v18 = out.join("mat_string_v18.mat");
    let mut string_opts = Options::default();
    string_opts.string_class = mat::StringClass::String;
    mat::to_file_with_options(&demo(), &mat_string_v18, &string_opts)
        .expect("write 1.8 string-class mat");

    let h5_v18 = out.join("plain_v18.h5");
    write_h5(&h5_v18, LibVer::V18);
    let h5_v110 = out.join("plain_v110.h5");
    write_h5(&h5_v110, LibVer::V110);

    let manifest = out.join("fixtures.json");
    std::fs::write(
        &manifest,
        serde_json::to_string_pretty(&checks()).expect("serialize manifest"),
    )
    .expect("write manifest");

    for path in [&mat_v18, &mat_v110, &mat_string_v18, &h5_v18, &h5_v110] {
        println!("{} superblock {}", path.display(), superblock_version(path));
    }
}
