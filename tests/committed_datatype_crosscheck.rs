// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Committed (`H5Tcommit`) datatypes, written by the reference C library and read
//! back by hdf5-pure (issue #254).
//!
//! A committed datatype is stored in its own object header, and every object that
//! uses it stores a *reference* to that header in place of the type. Nothing in
//! the referring bytes says so — a version 2 reference to address `0x320` decodes
//! as a perfectly well-formed zero-width time datatype — so the only thing that
//! separates a reference from an encoding is a flag bit, and reading the bytes
//! without it returns the wrong type with no error anywhere.
//!
//! The read-side fixtures here are built through the C library's own
//! `H5Tcommit2`, so the type each object *should* report is the one the C library
//! reads back from the same file. The write side runs the other way: this crate
//! places the committed object and the C library is asked whether the type it
//! finds is committed, which is the question no pure-Rust round trip can answer
//! about itself.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use hdf5::{ObjectReference1, ReferencedObject};
use hdf5_pure::{AttrValue, Datatype, File, RepackOptions};
use hdf5_sys::h5a::{H5Aclose, H5Acreate2, H5Awrite};
use hdf5_sys::h5p::H5P_DEFAULT;
use tempfile::tempdir;

mod common;
use common::create_v18;

// The one raw call below, marked with the upstream issue that would remove
// it, bypasses the lock the wrapper serializes its own calls through. Every
// C-library use in this file takes this guard, so a raw call never races a
// wrapper call on another test thread. Poisoning is ignored: a panic in one
// test must not cascade into the others.
static C_LIB: Mutex<()> = Mutex::new(());

fn c_lib_guard() -> MutexGuard<'static, ()> {
    C_LIB.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a fixture puts in the file, so every assertion below names one constant
/// rather than repeating a literal the fixture could drift away from.
const ATTR_VALUE: i32 = 7;
const DATASET_VALUES: [i32; 3] = [10, 20, 30];

/// The i32 type every fixture commits, as this crate decodes it. Committing does
/// not change the encoding — only where it is stored — so the answer a resolved
/// reference must produce is exactly the answer an inline `H5T_STD_I32LE` gives.
fn committed_i32() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: hdf5_pure::DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    }
}

/// Commit `dtype` in `loc` under `name`. Afterwards `dtype` itself is the
/// committed type.
fn commit_type(loc: &hdf5::Group, name: &str, dtype: &hdf5::Datatype) {
    loc.commit_datatype(name, dtype)
        .unwrap_or_else(|e| panic!("commit {name:?}: {e}"));
}

/// Create `name` in `loc` with element type `dtype`, holding `values`. `chunk`
/// makes it chunked and resizable. `None` is contiguous storage.
fn write_i32_dataset(
    loc: &hdf5::Group,
    name: &str,
    dtype: &hdf5::Datatype,
    chunk: Option<usize>,
    values: &[i32],
) {
    let builder = loc.new_dataset_builder().empty_as(dtype);
    let dataset = match chunk {
        Some(size) => builder
            .chunk([size])
            .shape(hdf5::SimpleExtents::resizable([values.len()]))
            .create(name),
        None => builder.shape([values.len()]).create(name),
    }
    .unwrap_or_else(|e| panic!("create {name:?}: {e}"));
    dataset
        .write(values)
        .unwrap_or_else(|e| panic!("write {name:?}: {e}"));
}

/// What a fixture puts the committed `/mytype` to use for.
///
/// The two uses are separable on purpose. Repack refuses each of them in its own
/// place, and a fixture carrying both lets either refusal stand in for the other:
/// remove one and the file is still refused, by the other, with every test still
/// green.
#[derive(Clone, Copy, Default)]
struct Fixture {
    /// A `shared_attr` attribute on the root group and on `/data` whose datatype
    /// is the committed one.
    committed_attrs: bool,
    /// A `/typed` dataset whose *element* type is the committed one.
    committed_dataset: bool,
    /// Further committed attributes on the root, enough of them to push its
    /// attribute storage out of the object header and into a fractal heap — a
    /// separate decode path with the same defect.
    dense_attrs: usize,
}

/// Everything the committed type can be used for, which is what the read-side
/// tests want: one file exercising every path.
const EVERYTHING: Fixture = Fixture {
    committed_attrs: true,
    committed_dataset: true,
    dense_attrs: 0,
};

/// Write a file whose `/mytype` is a committed i32, put to the uses `fixture`
/// names. `/data` and the ordinary `plain` attribute beside each committed one
/// are always present.
fn write_committed_fixture(path: &Path, fixture: Fixture) {
    let file = create_v18(path);
    let dtype = hdf5::Datatype::from_type::<i32>().expect("transient i32 type");
    commit_type(&file, "mytype", &dtype);

    // An ordinary dataset to hang attributes on.
    let data = file
        .new_dataset::<f64>()
        .shape([1])
        .create("data")
        .expect("create /data");
    data.write(&[1.0f64]).expect("write /data");

    let owners: [&hdf5::Location; 2] = [&file, &data];
    for owner in owners {
        if fixture.committed_attrs {
            write_attr(owner, "shared_attr", Some(&dtype), ATTR_VALUE);
        }
        // An ordinary attribute beside it: the C library abandons an object's
        // whole attribute list when one attribute fails to decode, so a healthy
        // neighbour is what makes that collateral damage visible.
        write_attr(owner, "plain", None, -ATTR_VALUE);
    }

    for i in 0..fixture.dense_attrs {
        write_attr(&file, &format!("dense{i:02}"), Some(&dtype), i as i32);
    }

    if fixture.committed_dataset {
        // A dataset whose *element* type is the committed one.
        write_i32_dataset(&file, "typed", &dtype, None, &DATASET_VALUES);
    }
}

/// Create a one-element i32 attribute on `owner`. `committed` names the
/// committed type when the attribute is to reference one. `None` uses a fresh
/// transient copy, whose encoding the C library stores inline in the message.
fn write_attr(owner: &hdf5::Location, name: &str, committed: Option<&hdf5::Datatype>, value: i32) {
    let Some(dtype) = committed else {
        owner
            .new_attr::<i32>()
            .shape([1])
            .create(name)
            .unwrap_or_else(|e| panic!("create attribute {name}: {e}"))
            .write(&[value])
            .unwrap_or_else(|e| panic!("write attribute {name}: {e}"));
        return;
    };
    let space = hdf5::Dataspace::try_new([1]).expect("a one-element dataspace");
    let transient = hdf5::Datatype::from_type::<i32>().expect("transient i32 type");
    let cname = std::ffi::CString::new(name).unwrap();
    // TODO: https://github.com/metno/hdf5-rust/issues/232
    // Safety: every id is live, and the buffer is one element of the memory type.
    unsafe {
        let attr = H5Acreate2(
            owner.id(),
            cname.as_ptr(),
            dtype.id(),
            space.id(),
            H5P_DEFAULT,
            H5P_DEFAULT,
        );
        assert!(attr >= 0, "H5Acreate2 failed for {name}");
        let rc = H5Awrite(attr, transient.id(), std::ptr::from_ref(&value).cast());
        assert!(rc >= 0, "H5Awrite failed for {name}");
        H5Aclose(attr);
    }
}

/// An attribute whose datatype is a committed one reports the type it names, on
/// the root group and on a dataset alike — and reports its value, which the
/// zero-width type the reference used to decode as made unreadable.
#[test]
fn a_committed_attribute_datatype_resolves_to_the_type_it_names() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    let file = File::open(&path).unwrap();
    for (owner, datatypes, attrs) in [
        (
            "root group",
            file.root().attr_datatypes().unwrap(),
            file.root().attrs().unwrap(),
        ),
        (
            "/data",
            file.dataset("data").unwrap().attr_datatypes().unwrap(),
            file.dataset("data").unwrap().attrs().unwrap(),
        ),
    ] {
        assert_eq!(
            datatypes.get("shared_attr"),
            Some(&committed_i32()),
            "{owner}: a committed attribute datatype was not resolved"
        );
        assert_eq!(
            attrs.get("shared_attr"),
            Some(&AttrValue::I32Array(vec![ATTR_VALUE])),
            "{owner}: a committed attribute's value was dropped"
        );
        // The ordinary attribute beside it is unaffected. Both fixtures' attributes
        // share one rank-1 dataspace, so both arrive as arrays.
        assert_eq!(
            attrs.get("plain"),
            Some(&AttrValue::I32Array(vec![-ATTR_VALUE])),
            "{owner}: an inline attribute changed"
        );
    }
}

/// A dataset whose element type is committed reads its elements. The old decode
/// produced a zero-width type, which is the one thing that cannot hold any data
/// at all.
#[test]
fn a_committed_dataset_datatype_resolves_and_its_data_reads() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    let file = File::open(&path).unwrap();
    let typed = file.dataset("typed").unwrap();
    assert_eq!(typed.datatype().unwrap(), committed_i32());
    assert_eq!(typed.read_i32().unwrap(), DATASET_VALUES);
}

/// Dense attribute storage decodes committed datatypes the same way. The bytes
/// come out of a fractal heap rather than the object header, which is a second
/// decode path — and each reader backend walks that heap with its own code, so
/// both are checked here rather than one standing in for the other.
#[test]
fn committed_datatypes_resolve_in_dense_attribute_storage() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("dense.h5");
    // Comfortably past the eight-attribute threshold that moves storage to a heap.
    const DENSE: usize = 40;
    write_committed_fixture(
        &path,
        Fixture {
            dense_attrs: DENSE,
            ..EVERYTHING
        },
    );

    for (backend, file) in [
        ("buffered", File::open(&path).unwrap()),
        ("streaming", File::open_streaming(&path).unwrap()),
    ] {
        let datatypes = file.root().attr_datatypes().unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(
            datatypes.len(),
            DENSE + 2,
            "{backend}: every attribute must still be reported"
        );
        for i in 0..DENSE {
            let name = format!("dense{i:02}");
            assert_eq!(
                datatypes.get(&name),
                Some(&committed_i32()),
                "{backend}/{name}: a committed datatype was not resolved in dense storage"
            );
            assert_eq!(
                attrs.get(&name),
                Some(&AttrValue::I32Array(vec![i32::try_from(i).unwrap()])),
                "{backend}/{name}: a dense committed attribute's value was dropped"
            );
        }
    }
}

/// An in-place edit of an object that carries a committed attribute leaves that
/// attribute alone and still succeeds.
///
/// The editor walks an object-header region with no file context, so it cannot
/// decode the committed attribute — but it does not need to. The reference stays
/// valid inside the same file, and identifying an attribute by name never needs
/// its datatype, so refusing the whole object because one of its neighbours is
/// committed would be a limit with no cause.
#[test]
fn an_edit_passes_over_a_committed_attribute_it_does_not_touch() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    {
        let file = File::open_rw(&path).unwrap();
        file.root().set_attr("added", AttrValue::I64(5)).unwrap();
        file.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(5)));
    assert_eq!(
        attrs.get("shared_attr"),
        Some(&AttrValue::I32Array(vec![ATTR_VALUE])),
        "the committed attribute did not survive an edit beside it"
    );
    assert_eq!(
        file.root().attr_datatypes().unwrap().get("shared_attr"),
        Some(&committed_i32())
    );
    drop(file);

    // And the C library reads the result. This crate accepting its own output
    // proves nothing on its own: one bad attribute makes libhdf5 abandon an
    // object's whole attribute list, which is exactly the damage a rewrite of a
    // committed reference does and exactly what a pure-Rust round trip cannot see.
    let c_file = hdf5::File::open(&path).expect("C library must open the edited file");
    let mut names = c_file.attr_names().expect("C library must list attributes");
    names.sort();
    assert_eq!(
        names,
        vec![
            "added".to_string(),
            "plain".to_string(),
            "shared_attr".to_string()
        ]
    );
    assert_eq!(
        c_file
            .attr("shared_attr")
            .unwrap()
            .read_1d::<i32>()
            .unwrap()
            .to_vec(),
        vec![ATTR_VALUE]
    );
}

/// A cross-file copy of an object carrying a committed attribute is refused.
///
/// The copy moves the attribute's bytes verbatim, and those bytes are an address
/// in the *source* file — nothing in the destination is at it. The object-header
/// record's own shared flag does not report this: it describes the attribute
/// message, not the datatype field inside it, so the screen has to read the
/// attribute's flags byte to see it at all.
#[test]
fn a_cross_file_copy_refuses_a_committed_attribute() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("committed.h5");
    let dst_path = dir.path().join("dest.h5");
    write_committed_fixture(&src_path, EVERYTHING);
    hdf5_pure::FileBuilder::new().write(&dst_path).unwrap();

    let source = File::open(&src_path).unwrap();
    let dest = File::open_rw(&dst_path).unwrap();
    let err = dest
        .copy_from(&source, "data", "data")
        .expect_err("a committed attribute must not cross files");
    let message = err.to_string();
    assert!(
        message.contains("committed"),
        "the refusal must name the committed datatype, got: {message}"
    );
}

/// An in-place append to a dataset whose element type is committed is refused.
///
/// The append engine walks the object header itself and sizes each element from
/// the datatype message's bytes. Those bytes are a reference, and they decode as
/// a zero-width type — so without reading the record's shared flag the engine
/// takes its geometry from a type the dataset does not have. A typed append then
/// fails with a misleading complaint that the element type does not match, and
/// [`Dataset::append_raw`], which has no type to compare against, would stride
/// through the data one byte per element. The C library writes this shape for
/// `maxshape=(None,)` on a latest-format file.
#[test]
fn an_in_place_append_refuses_a_committed_element_type() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("appendable.h5");
    write_appendable_committed_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let mut typed = file.dataset("typed").unwrap();
    let err = typed
        .append(&[40i32])
        .expect_err("an append must not size its elements from a reference");
    let message = err.to_string();
    assert!(
        message.contains("committed (shared) datatype"),
        "the refusal must name the committed datatype, got: {message}"
    );
}

/// A latest-format file whose `/typed` is chunked, unlimited along its one
/// dimension, and typed by the committed `/mytype` — the extensible-array shape
/// this crate's in-place append engine maintains.
fn write_appendable_committed_fixture(path: &Path) {
    let file = create_v18(path);
    let dtype = hdf5::Datatype::from_type::<i32>().expect("transient i32 type");
    commit_type(&file, "mytype", &dtype);
    write_i32_dataset(&file, "typed", &dtype, Some(4), &DATASET_VALUES);
}

/// The streaming backend reads a committed datatype through the same reference,
/// resolved by reading the target object header on demand rather than indexing a
/// slice. Two backends, one answer.
#[test]
fn the_streaming_backend_resolves_a_committed_datatype_too() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);

    let file = File::open_streaming(&path).unwrap();
    assert_eq!(
        file.root().attr_datatypes().unwrap().get("shared_attr"),
        Some(&committed_i32())
    );
    assert_eq!(
        file.dataset("typed").unwrap().datatype().unwrap(),
        committed_i32()
    );
    assert_eq!(
        file.dataset("typed").unwrap().read_i32().unwrap(),
        DATASET_VALUES
    );
}

/// Whether the C library calls the datatype of `path`'s dataset a committed one,
/// and the address of the object it names.
///
/// `H5Tcommitted` is the crosscheck that matters: this crate reading back its own
/// reference proves only that it agrees with itself, while the C library saying
/// "committed" means the file really carries the shared-message encoding a named
/// type is made of.
fn dataset_type_is_committed(file: &hdf5::File, path: &str) -> bool {
    file.dataset(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .dtype()
        .expect("dataset datatype")
        .is_committed()
}

/// The same question for an attribute, which stores its reference in the
/// attribute message rather than in a header message record — a separate encoding
/// with a separate flag.
fn attr_type_is_committed(file: &hdf5::File, owner: &str, attr: &str) -> bool {
    let owner: hdf5::Location = if owner.is_empty() {
        (*file.group("/").expect("open root group")).clone()
    } else {
        (**file.dataset(owner).expect("open attribute owner")).clone()
    };
    owner
        .attr(attr)
        .unwrap_or_else(|e| panic!("open attribute {attr}: {e}"))
        .dtype()
        .expect("attribute datatype")
        .is_committed()
}

/// A repack carries every use of a committed type across as a use of the *same*
/// committed type, rather than inlining a copy of the encoding.
///
/// Inlining would read back correctly and still lose what makes the type named:
/// `h5dump` would stop printing `DATATYPE "/mytype"`, and objects that shared one
/// type would each declare their own. So the assertions are about identity — the
/// C library calling each type committed, and both users resolving to the one
/// object — not about the values alone. Before #254 this same call returned `Ok`
/// and produced a file libhdf5 could not read *any* attributes from.
#[test]
fn repack_reproduces_a_committed_datatype() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let src = dir.path().join("committed.h5");
    let dst = dir.path().join("repacked.h5");
    write_committed_fixture(&src, EVERYTHING);

    hdf5_pure::repack(&src, &dst, &RepackOptions::new()).expect("repack a committed datatype");

    // The C library's verdict on the output, taken before this crate's, because
    // it is the one that cannot be satisfied by agreeing with the writer.
    {
        let file = hdf5::File::open(&dst).expect("C library opens the repacked file");
        assert!(
            dataset_type_is_committed(&file, "typed"),
            "the repacked dataset must still name a committed type, not carry an inline copy"
        );
        assert!(
            attr_type_is_committed(&file, "", "shared_attr"),
            "the repacked root attribute must still name a committed type"
        );
        assert!(
            attr_type_is_committed(&file, "data", "shared_attr"),
            "the repacked dataset attribute must still name a committed type"
        );
        assert!(
            !attr_type_is_committed(&file, "data", "plain"),
            "an attribute that was inline in the source must not become committed"
        );
        assert_eq!(
            file.dataset("typed")
                .unwrap()
                .read_1d::<i32>()
                .unwrap()
                .to_vec(),
            DATASET_VALUES.to_vec()
        );
    }

    // And the named object itself is back, holding the type it held.
    let out = File::open(&dst).expect("hdf5-pure opens the repacked file");
    assert_eq!(
        out.dataset("typed").unwrap().datatype().unwrap(),
        committed_i32()
    );
    let root_attr = out.root().attr_datatypes().unwrap();
    assert_eq!(root_attr.get("shared_attr"), Some(&committed_i32()));
    assert_eq!(
        out.root().attrs().unwrap().get("shared_attr"),
        Some(&AttrValue::I32Array(vec![ATTR_VALUE]))
    );
}

/// Two users of one committed type stay two users of *one* type across a repack.
///
/// This is the assertion an inline copy passes right up until it is asked: with
/// the encoding written into each user, every value still reads and every type
/// still compares equal, and the file has silently become one where the objects
/// no longer share anything. Address identity is what distinguishes them.
#[test]
fn users_of_one_committed_type_still_share_one_object() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let src = dir.path().join("committed.h5");
    let dst = dir.path().join("repacked.h5");
    write_committed_fixture(&src, EVERYTHING);
    hdf5_pure::repack(&src, &dst, &RepackOptions::new()).expect("repack");

    let out = File::open(&dst).unwrap();
    assert_eq!(
        out.root().named_datatypes().unwrap(),
        vec!["mytype".to_string()],
        "the output must hold exactly one committed object"
    );
    // With one committed object in the file and the C library calling both users'
    // types committed, the two name that object: a committed type has nowhere
    // else to live.
    let file = hdf5::File::open(&dst).unwrap();
    assert!(dataset_type_is_committed(&file, "typed"));
    assert!(attr_type_is_committed(&file, "data", "shared_attr"));
    assert_eq!(
        out.root().named_datatype("mytype").unwrap(),
        committed_i32()
    );
}

/// A committed type a repack *drops* is refused by name rather than left dangling.
///
/// Dropping the object while something still names it is the one way a repack can
/// produce a reference to nothing, so it is refused where the naming object is
/// still known. The unreferenced case is the control: with nothing naming it, the
/// same drop is an ordinary one.
#[test]
fn repack_refuses_dropping_a_committed_type_still_in_use() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();

    for (what, fixture, expected) in [
        (
            "a dataset element type",
            Fixture {
                committed_dataset: true,
                ..Fixture::default()
            },
            Some("dataset typed: names the committed datatype"),
        ),
        (
            "an attribute datatype",
            Fixture {
                committed_attrs: true,
                ..Fixture::default()
            },
            Some("attribute \"shared_attr\": names the committed datatype"),
        ),
        // Nothing names it, so dropping it is just a drop.
        ("nothing", Fixture::default(), None),
    ] {
        let src = dir.path().join("committed.h5");
        let dst = dir.path().join("repacked.h5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
        write_committed_fixture(&src, fixture);

        let options = RepackOptions::new().drop_path("mytype");
        let result = hdf5_pure::repack(&src, &dst, &options);
        match expected {
            Some(expected) => {
                let message = result
                    .expect_err("dropping a type in use must fail")
                    .to_string();
                assert!(
                    message.contains(expected),
                    "dropping the type {what} names must say so; expected {expected:?}, \
                     got: {message}"
                );
            }
            None => {
                result.expect("dropping an unreferenced committed type is an ordinary drop");
                let out = File::open(&dst).unwrap();
                assert!(out.root().named_datatypes().unwrap().is_empty());
            }
        }
    }
}

/// A file this crate writes with a committed datatype is one the C library calls
/// committed.
///
/// This is the direction no self-consistency check can cover: reading back our own
/// reference proves the two halves of this crate agree, not that the file carries
/// what HDF5 calls a named type. `H5Tcommitted` is the C library's own answer, and
/// it is false for an inline encoding no matter how correct the bytes are.
#[test]
fn a_committed_datatype_this_crate_writes_reads_back_as_committed() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("written.h5");

    let mut builder = hdf5_pure::FileBuilder::new();
    builder.commit_datatype("mytype", committed_i32());
    builder.set_attr_committed("root_attr", AttrValue::I32(ATTR_VALUE), "mytype");
    builder
        .create_dataset("typed")
        .with_i32_data(&DATASET_VALUES)
        // Absolute, where every other test here names the type relatively. This
        // is the only coverage of the leading slash `normalize_object_path`
        // trims, so the two spellings must not be made uniform.
        .with_committed_datatype("/mytype")
        .set_attr_committed("shared_attr", AttrValue::I32(ATTR_VALUE), "mytype");
    builder.write(&path).expect("write a committed datatype");

    let file = hdf5::File::open(&path).expect("C library opens the file");
    assert!(
        dataset_type_is_committed(&file, "typed"),
        "the dataset's element type must be a named type, not an inline copy"
    );
    assert!(
        attr_type_is_committed(&file, "typed", "shared_attr"),
        "the dataset attribute's type must be a named type"
    );
    assert!(
        attr_type_is_committed(&file, "", "root_attr"),
        "the root attribute's type must be a named type"
    );
    assert_eq!(
        file.dataset("typed")
            .unwrap()
            .read_1d::<i32>()
            .unwrap()
            .to_vec(),
        DATASET_VALUES.to_vec(),
        "naming the type must not disturb the element bytes"
    );
    // The type object is a real child of the root group, not a stranded header.
    assert_eq!(
        File::open(&path).unwrap().root().named_datatypes().unwrap(),
        vec!["mytype".to_string()]
    );
}

/// The reference count this crate writes for a committed type is the one the C
/// library writes for the same file.
///
/// The count is hard links plus every message naming the type — `H5O_link`, once
/// per link and once per dataset or attribute created against it. Nothing
/// *reading* a file notices a wrong count; it decides what happens when the type
/// is later unlinked, and a count of 1 means the first unlink destroys a type
/// other objects are still using. So the C library's own number for an equivalent
/// file is the only available ground truth, and this test takes it from a file
/// `H5Tcommit2` wrote rather than from a constant.
#[test]
fn the_reference_count_matches_what_the_c_library_writes() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let src = dir.path().join("committed.h5");
    write_committed_fixture(&src, EVERYTHING);

    // One hard link, one dataset element type, two attribute datatypes.
    let expected = File::open(&src)
        .unwrap()
        .root()
        .named_datatype_references("mytype")
        .unwrap();
    assert_eq!(
        expected, 4,
        "the fixture links the type once and names it three times"
    );

    // A repack of that file.
    let dst = dir.path().join("repacked.h5");
    hdf5_pure::repack(&src, &dst, &RepackOptions::new()).expect("repack");
    assert_eq!(
        File::open(&dst)
            .unwrap()
            .root()
            .named_datatype_references("mytype")
            .unwrap(),
        expected,
        "a repack must carry the reference count, not reset it"
    );

    // And a file built from scratch with the same shape.
    let written = dir.path().join("written.h5");
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.commit_datatype("mytype", committed_i32());
    builder.set_attr_committed("shared_attr", AttrValue::I32(ATTR_VALUE), "mytype");
    builder
        .create_dataset("typed")
        .with_i32_data(&DATASET_VALUES)
        .with_committed_datatype("mytype")
        .set_attr_committed("shared_attr", AttrValue::I32(ATTR_VALUE), "mytype");
    builder.write(&written).expect("write");
    assert_eq!(
        File::open(&written)
            .unwrap()
            .root()
            .named_datatype_references("mytype")
            .unwrap(),
        expected,
        "the writer must count the same uses the C library counts"
    );
}

/// A dataset or attribute naming a type it does not match is refused, rather than
/// written as a file whose element bytes and declared type disagree.
///
/// The committed encoding is the only one in the file, so a reader takes it: an
/// i32 dataset that named an f64 type would have its bytes read as doubles, half
/// as many of them, with nothing anywhere reporting a problem.
#[test]
fn a_dataset_naming_a_type_it_does_not_match_is_refused() {
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.commit_datatype("mytype", hdf5_pure::make_f64_type());
    builder
        .create_dataset("typed")
        .with_i32_data(&DATASET_VALUES)
        .with_committed_datatype("mytype");
    let message = builder.finish().unwrap_err().to_string();
    assert!(
        message.contains("names the committed datatype") && message.contains("mytype"),
        "expected a mismatch refusal naming the type, got: {message}"
    );
}

/// A path that commits no datatype is refused too, for the same reason: the
/// reference would name nothing.
#[test]
fn a_dataset_naming_a_type_the_file_does_not_commit_is_refused() {
    let mut builder = hdf5_pure::FileBuilder::new();
    builder
        .create_dataset("typed")
        .with_i32_data(&DATASET_VALUES)
        .with_committed_datatype("nosuchtype");
    let message = builder.finish().unwrap_err().to_string();
    assert!(
        message.contains("no committed datatype is written at path \"nosuchtype\""),
        "expected an unknown-path refusal, got: {message}"
    );
}

/// Dropping the *group* a committed type lives in is refused where something
/// outside that group still names the type.
///
/// The drop set names the group, not the type, so a membership test on the type's
/// own path sees nothing dropped and lets the repack proceed to a dataset naming
/// an object the output does not contain. The whole subtree goes with a dropped
/// group, so the check has to look at ancestors too.
#[test]
fn repack_refuses_dropping_the_group_a_named_type_lives_in() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let src = dir.path().join("nested.h5");
    let dst = dir.path().join("repacked.h5");

    {
        let file = create_v18(&src);
        let dtype = hdf5::Datatype::from_type::<i32>().expect("transient i32 type");
        let group = file.create_group("types").expect("create /types");
        commit_type(&group, "mytype", &dtype);
        write_i32_dataset(&file, "typed", &dtype, None, &DATASET_VALUES);
    }

    let options = RepackOptions::new().drop_path("types");
    let message = hdf5_pure::repack(&src, &dst, &options)
        .expect_err("dropping the group the type lives in must fail")
        .to_string();
    assert!(
        message.contains("types/mytype") && message.contains("drops"),
        "the refusal must name the type inside the dropped group, got: {message}"
    );
}

/// An object reference *to* a committed datatype crosses a repack, pointing at
/// the type's new address.
///
/// A committed datatype is an object, so `H5Rcreate` can address one, and repack
/// resolves such a reference exactly as it resolves one to a dataset or group.
/// Before the named type object was reproduced there was nothing on the other end
/// to point at, and the reference was refused.
#[test]
fn an_object_reference_to_a_committed_type_survives_a_repack() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let src = dir.path().join("refs.h5");
    let dst = dir.path().join("repacked.h5");
    write_committed_fixture(&src, EVERYTHING);

    // Add a reference dataset pointing at `/mytype`, through the safe API.
    {
        let file = hdf5::File::open_rw(&src).expect("reopen fixture");
        let reference: ObjectReference1 = file
            .reference("mytype")
            .expect("reference the committed type");
        file.new_dataset::<ObjectReference1>()
            .shape([1])
            .create("refs")
            .expect("create /refs")
            .write(&[reference])
            .expect("write /refs");
    }

    hdf5_pure::repack(&src, &dst, &RepackOptions::new()).expect("repack a reference to a type");

    let file = hdf5::File::open(&dst).expect("C library opens the output");
    let refs = file
        .dataset("refs")
        .unwrap()
        .read_1d::<ObjectReference1>()
        .unwrap();
    let target = file
        .dereference(&refs[0])
        .expect("the reference must resolve in the output");
    assert!(
        matches!(target, ReferencedObject::Datatype(_)),
        "the reference must still point at the committed datatype, got {target:?}"
    );
}

/// An in-place edit refuses to add a dataset or attribute naming a committed
/// datatype, rather than quietly writing the type inline.
///
/// The edit engine appends into a file whose layout is already fixed, so there is
/// nowhere to place the named type object and nothing to resolve the path
/// against. Inlining would produce a dataset that reads back correctly while no
/// longer sharing the named type — the silent degradation this crate refuses.
#[test]
fn an_in_place_edit_refuses_a_committed_datatype() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");

    let mut builder = hdf5_pure::FileBuilder::new();
    builder.commit_datatype("mytype", committed_i32());
    builder
        .create_dataset("seed")
        .with_i32_data(&DATASET_VALUES);
    builder.write(&path).expect("write the base file");

    for (what, configure) in [
        (
            "a dataset element type",
            Box::new(|db: &mut hdf5_pure::DatasetBuilder| {
                db.with_i32_data(&DATASET_VALUES)
                    .with_committed_datatype("mytype");
            }) as Box<dyn Fn(&mut hdf5_pure::DatasetBuilder)>,
        ),
        (
            "an attribute datatype",
            Box::new(|db: &mut hdf5_pure::DatasetBuilder| {
                db.with_i32_data(&DATASET_VALUES).set_attr_committed(
                    "a",
                    AttrValue::I32(ATTR_VALUE),
                    "mytype",
                );
            }),
        ),
    ] {
        let file = File::open_rw(&path).expect("open for editing");
        let message = file
            .root()
            .create_dataset("added", |db| configure(db))
            .expect_err("naming a committed type in place must fail")
            .to_string();
        assert!(
            message.contains("committed"),
            "the refusal must name what it refused ({what}), got: {message}"
        );
    }

    // The file is unchanged: a refused edit adds nothing.
    let out = File::open(&path).unwrap();
    assert_eq!(out.root().datasets().unwrap(), vec!["seed".to_string()]);
}

/// What the C library calls a named datatype is what this crate's by-name
/// lookups accept, and nothing else (issue #364).
///
/// A dataset's object header carries a datatype message of its own — its element
/// type — so a lookup that went looking for a datatype message found one, and
/// `named_datatype` answered with it. `/typed` is the case that makes the wrong
/// answer indistinguishable from a right one: its element type *is* the
/// committed type, so both names produced the same value, and nothing in the
/// result said which object had been asked for.
///
/// The classification is taken from `H5O_get_info_by_name` rather than from a
/// list written here: it and the `H5O_obj_type` that `H5Topen` gates on bottom
/// out in the same `H5O__obj_class_real`. This crate and the reference library
/// must classify the same file the same way, and only one of them is ground
/// truth.
#[test]
fn only_what_the_c_library_calls_a_named_datatype_is_accepted() {
    let _c = c_lib_guard();
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed.h5");
    write_committed_fixture(&path, EVERYTHING);
    // A group as well, so all three of the C library's object classes are
    // present: a check written as "is not a dataset" would let this one through.
    {
        let file = hdf5::File::open_rw(&path).expect("reopen fixture");
        file.create_group("g").expect("create /g");
    }

    let c_file = hdf5::File::open(&path).expect("the C library opens the fixture");
    let names = c_file.member_names().expect("root members");
    let ours = File::open(&path).expect("hdf5-pure opens the fixture");
    let root = ours.root();

    let (mut named, mut kinds) = (Vec::new(), Vec::new());
    for name in &names {
        let c_says = c_file.loc_type_by_name(name).expect("C library classifies");
        kinds.push(c_says);
        let got = root.named_datatype(name);
        if c_says == hdf5::LocationType::NamedDatatype {
            named.push(name.clone());
            assert_eq!(
                got.expect("a committed type must read"),
                committed_i32(),
                "{name} is a named datatype to the C library"
            );
            root.named_datatype_references(name)
                .expect("a committed type must count");
        } else {
            assert!(
                matches!(&got, Err(hdf5_pure::Error::NotANamedDatatype(p)) if p == name),
                "{name} is {c_says:?} to the C library, and answered {got:?}"
            );
            assert!(matches!(
                root.named_datatype_references(name),
                Err(hdf5_pure::Error::NotANamedDatatype(_))
            ));
        }
    }

    // The fixture is only worth this much if it holds every class the C library
    // distinguishes: a check written as "is not a dataset" passes a file with no
    // group in it. And the listing has to agree with the per-name answers above.
    for kind in [
        hdf5::LocationType::Dataset,
        hdf5::LocationType::Group,
        hdf5::LocationType::NamedDatatype,
    ] {
        assert!(kinds.contains(&kind), "the fixture holds no {kind:?}");
    }
    assert_eq!(named, ["mytype"]);
    assert_eq!(root.named_datatypes().unwrap(), named);
}
