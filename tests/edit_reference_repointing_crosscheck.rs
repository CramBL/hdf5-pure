// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! The reference C library's view of a file whose stored object references this
//! crate repointed across a commit (issue #324).
//!
//! Two things only the C library can establish here. It writes the shapes this
//! crate cannot stage — a reference-typed *attribute*, a chunked reference
//! dataset — so the fixture is real rather than hand-built. And it validates an
//! object header's checksum on the way in, so a repointed attribute that opens
//! and reads is proof the header was resealed after its bytes changed, which no
//! pure-Rust read would notice.

use hdf5_pure::File;
use tempfile::tempdir;

mod common;
use common::create_v18;

/// Dirty `g` so its object header is rebuilt at a fresh address, then spend the
/// freed space, so a reference left behind resolves to reused bytes rather than
/// to a stale copy that still happens to read.
fn move_the_group_and_churn(path: &std::path::Path) {
    let session = File::open_rw(path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);
    for i in 0..10 {
        let session = File::open_rw(path).unwrap();
        session
            .root()
            .create_dataset(&format!("churn{i}"), |b| {
                b.with_i32_data(&[i]);
            })
            .unwrap();
        session.commit().unwrap();
    }
}

fn assert_resolves_to_the_moved_group(group: &hdf5::Group) {
    let mut names = group.member_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        vec!["extra".to_string(), "inner".to_string()],
        "the reference must resolve to the group as the commit left it"
    );
}

#[test]
fn a_c_written_reference_attribute_is_repointed_and_the_header_resealed() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let path = dir.path().join("attr_ref.h5");

    {
        let file = create_v18(&path);
        let g = file.create_group("g").unwrap();
        g.new_dataset::<i32>()
            .shape((3,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        let holder = file
            .new_dataset::<i32>()
            .shape((1,))
            .create("holder")
            .unwrap();
        holder.write(&[0i32]).unwrap();
        holder
            .new_attr::<ObjectReference1>()
            .shape((1,))
            .create("target")
            .unwrap()
            .write(&[ObjectReference1::create(&file, "g").unwrap()])
            .unwrap();
        file.close().unwrap();
    }

    move_the_group_and_churn(&path);

    // Opening the dataset and reading the attribute makes the C library verify
    // the object header's checksum: the repointed bytes live inside that header,
    // so a fixup that changed them without resealing fails here rather than in
    // the value.
    let c = hdf5::File::open(&path).unwrap();
    let values = c
        .dataset("holder")
        .unwrap()
        .attr("target")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    match values[0].dereference(&c).unwrap() {
        ReferencedObject::Group(g) => assert_resolves_to_the_moved_group(&g),
        other => panic!("expected the group, got {other:?}"),
    }
}

#[test]
fn a_c_written_reference_dataset_is_repointed() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let path = dir.path().join("ds_ref.h5");

    {
        let file = hdf5::File::create(&path).unwrap();
        let g = file.create_group("g").unwrap();
        g.new_dataset::<i32>()
            .shape((3,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.new_dataset::<ObjectReference1>()
            .shape((1,))
            .create("refs")
            .unwrap()
            .write(&[ObjectReference1::create(&file, "g").unwrap()])
            .unwrap();
        file.close().unwrap();
    }

    move_the_group_and_churn(&path);

    let c = hdf5::File::open(&path).unwrap();
    let values = c
        .dataset("refs")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    match values[0].dereference(&c).unwrap() {
        ReferencedObject::Group(g) => assert_resolves_to_the_moved_group(&g),
        other => panic!("expected the group, got {other:?}"),
    }
}

/// A chunked reference dataset is the one shape whose addresses this walk does
/// not reach: they live inside chunks, which a filtered dataset stores
/// compressed and which this does not decode either way.
///
/// The behaviour pinned here is the pre-#324 one, unchanged and not a
/// regression, and it is pinned so that a later change closing the gap inverts a
/// test rather than passing one silently. It is asserted as "not repointed"
/// rather than as a specific wrong answer, because what the stale address
/// resolves to depends on what reuses the span.
///
/// Read it as a **record of a limit**, not as a guard: reaching a chunked
/// dataset's elements would take new code rather than a changed line, so no
/// mutation flips this assertion alone. What is load-bearing is the `assert_ne!`
/// beside it — without it the test would pass on a file where the group never
/// moved, which is to say on nothing at all.
#[test]
fn a_chunked_reference_dataset_is_left_unrepointed() {
    use hdf5::{ObjectReference, ObjectReference1};

    let dir = tempdir().unwrap();
    let path = dir.path().join("chunked_ref.h5");

    {
        let file = hdf5::File::create(&path).unwrap();
        let g = file.create_group("g").unwrap();
        g.new_dataset::<i32>()
            .shape((3,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        let reference = ObjectReference1::create(&file, "g").unwrap();
        file.new_dataset::<ObjectReference1>()
            .shape((4,))
            .chunk((2,))
            .create("chunked")
            .unwrap()
            .write(&[reference; 4])
            .unwrap();
        // A contiguous reference dataset naming the same group, so this test is
        // about *where* the elements live rather than about whether the group
        // moved at all: the contiguous one is repointed, and the assertion below
        // is only meaningful because of it.
        file.new_dataset::<ObjectReference1>()
            .shape((1,))
            .create("contiguous")
            .unwrap()
            .write(&[reference])
            .unwrap();
        file.close().unwrap();
    }
    let read = |name: &str| {
        File::open(&path)
            .unwrap()
            .dataset(name)
            .unwrap()
            .read_raw()
            .unwrap()
    };
    let chunked_before = read("chunked");
    let contiguous_before = read("contiguous");

    move_the_group_and_churn(&path);

    assert_ne!(
        read("contiguous"),
        contiguous_before,
        "the group moved, so the reachable reference to it changed"
    );
    assert_eq!(
        read("chunked"),
        chunked_before,
        "a chunked reference dataset's elements are outside what this reaches, so \
         they must be left exactly as they were"
    );
    // And the file the commits produced is still one the C library will open.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("chunked").unwrap().shape(), vec![4]);
}

/// A reference held by an object the same commit **rebuilds**, which is the case
/// that decides *when* the repointing may run.
///
/// The root group is rebuilt by every commit, so an attribute on it is copied
/// into a fresh header carrying whatever address it held before. Repointing the
/// pre-commit tree would correct the superseded copy — bytes that are already
/// dead — and leave the header the commit actually published still naming the
/// vacated address. Only a walk over the *committed* tree fixes the right one.
#[test]
fn a_reference_on_a_rebuilt_header_is_repointed_in_the_header_the_commit_published() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let path = dir.path().join("root_attr_ref.h5");

    {
        let file = hdf5::File::create(&path).unwrap();
        let g = file.create_group("g").unwrap();
        g.new_dataset::<i32>()
            .shape((3,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.new_attr::<ObjectReference1>()
            .shape((1,))
            .create("target")
            .unwrap()
            .write(&[ObjectReference1::create(&file, "g").unwrap()])
            .unwrap();
        file.close().unwrap();
    }

    move_the_group_and_churn(&path);

    let c = hdf5::File::open(&path).unwrap();
    let values = c
        .attr("target")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    match values[0].dereference(&c).unwrap() {
        ReferencedObject::Group(g) => assert_resolves_to_the_moved_group(&g),
        other => panic!("expected the group, got {other:?}"),
    }
}

/// A dataset whose element type is a **committed** (`H5Tcommit`) object
/// reference is repointed like any other.
///
/// The walk has to follow the committed type to learn the elements are
/// references at all: the datatype message in such a dataset's header is a
/// pointer into the file's shared-message storage rather than an encoded type,
/// so reading its class byte in place answers nothing. Without the resolution
/// step the dataset is invisible here — elements, addresses and all — and
/// nothing else in the suite notices.
///
#[test]
fn a_reference_dataset_through_a_committed_datatype_is_repointed() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};
    let dir = tempdir().unwrap();
    let path = dir.path().join("committed_ref.h5");

    {
        let file = hdf5::File::create(&path).unwrap();
        let g = file.create_group("g").unwrap();
        g.new_dataset::<i32>()
            .shape((3,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        let reference = ObjectReference1::create(&file, "g").unwrap();

        // `H5T_STD_REF_OBJ` is one of the library's immutable predefined types,
        // which cannot be committed. The copy `from_type` hands back can.
        let reftype = hdf5::Datatype::from_type::<ObjectReference1>().unwrap();
        file.commit_datatype("reftype", &reftype).unwrap();
        file.new_dataset_builder()
            .empty_as(&reftype)
            .shape((1,))
            .create("refs")
            .unwrap()
            .write(&[reference])
            .unwrap();
        file.close().unwrap();
    }

    move_the_group_and_churn(&path);

    let c = hdf5::File::open(&path).unwrap();
    let values = c
        .dataset("refs")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    match values[0].dereference(&c).unwrap() {
        ReferencedObject::Group(g) => assert_resolves_to_the_moved_group(&g),
        other => panic!("expected the group, got {other:?}"),
    }
}

/// The reference C library's **default** output format — earliest library
/// bounds, so every object header is version 1 — is reached too.
///
/// This is not a legacy corner. `H5Fcreate` without `H5Pset_libver_bounds` uses
/// the earliest format that can express each object, and h5py's default is the
/// same, so a file that was never told otherwise carries version 1 headers
/// throughout. Those headers are read by a different parser here, and until they
/// were, #324 was fixed for files in the latest format and unfixed for the
/// format most files are actually in.
///
/// What that parser reaches is narrower — a contiguous dataset's elements, which
/// live outside the header — so an attribute's value in a version 1 header is
/// still left alone.
#[test]
fn an_earliest_format_reference_dataset_is_repointed() {
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let path = dir.path().join("earliest.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|p| p.libver_earliest())
            .create(&path)
            .unwrap();
        let g = file.create_group("g").unwrap();
        g.new_dataset::<i32>()
            .shape((3,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.new_dataset::<ObjectReference1>()
            .shape((1,))
            .create("refs")
            .unwrap()
            .write(&[ObjectReference1::create(&file, "g").unwrap()])
            .unwrap();
        file.close().unwrap();
    }
    // The premise, asserted rather than assumed: a version 2 header carries an
    // `OHDR` signature and a version 1 header carries none, so a file with no
    // `OHDR` anywhere is one this walk reaches only through the other parser.
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        bytes.windows(4).filter(|w| *w == b"OHDR").count(),
        0,
        "the fixture must actually be in the earliest format, or this test is \
         about nothing"
    );

    move_the_group_and_churn(&path);

    let c = hdf5::File::open(&path).unwrap();
    let values = c
        .dataset("refs")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    match values[0].dereference(&c).unwrap() {
        ReferencedObject::Group(g) => assert_resolves_to_the_moved_group(&g),
        other => panic!("expected the group, got {other:?}"),
    }
}
