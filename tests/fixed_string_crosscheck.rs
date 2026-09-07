// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! The reference C library reads the fixed-width string datasets this crate
//! writes (issue #355).
//!
//! A fixed-width string dataset is a datatype message declaring one width and a
//! run of element bytes padded to it, and a round trip through this crate alone
//! cannot tell a correct pair from a pair that is merely self-consistent: a
//! writer that padded to the wrong width and a reader that divided by the same
//! wrong width would agree with each other and with nothing else. The C library
//! divides by the width the *message* declares, so it is the side that can tell.
//!
//! It is also the authority on the datatype itself. `H5T_CSET_ASCII` and
//! `H5T_CSET_UTF8` are separate types on disk, and the C library will not hand a
//! dataset over as the charset it was not written as — measured here rather than
//! assumed — so a charset written wrong is not a cosmetic difference.
//!
//! Every call here goes through the safe `hdf5-metno` API, which serializes its
//! own C calls through an internal lock, so these tests need no extra guard.

use hdf5::types::{FixedAscii, FixedUnicode, TypeDescriptor};
use hdf5_pure::FileBuilder;
use tempfile::tempdir;

/// The width the C library reports comes from the datatype message, so a writer
/// that padded to a different stride than it declared fails here even though
/// this crate would read its own file back correctly.
#[test]
fn the_c_library_reports_the_width_this_crate_declared() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strings.h5");

    let values = ["north", "s", "", "east"];
    let mut b = FileBuilder::new();
    b.create_dataset("derived")
        .with_ascii_strings(&values)
        .unwrap();
    b.create_dataset("sized")
        .with_ascii_strings_sized(&values, 16)
        .unwrap();
    b.create_dataset("utf8").with_strings(&values).unwrap();
    b.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    for (name, expected) in [
        ("derived", TypeDescriptor::FixedAscii(5)),
        ("sized", TypeDescriptor::FixedAscii(16)),
        ("utf8", TypeDescriptor::FixedUnicode(5)),
    ] {
        assert_eq!(
            file.dataset(name)
                .unwrap()
                .dtype()
                .unwrap()
                .to_descriptor()
                .unwrap(),
            expected,
            "dataset {name} reached the file as the wrong string type"
        );
    }
}

/// The values, read by the C library at the width the C library found. The empty
/// element sits in the middle deliberately: it contributes no bytes of its own,
/// so a writer that emitted values rather than fixed-width slots would leave
/// every element after it shifted, and only an element that follows one can
/// catch that.
#[test]
fn the_c_library_reads_back_every_value() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("strings.h5");

    let values = ["north", "s", "", "east"];
    let mut b = FileBuilder::new();
    b.create_dataset("derived")
        .with_ascii_strings(&values)
        .unwrap();
    b.create_dataset("sized")
        .with_ascii_strings_sized(&values, 16)
        .unwrap();
    b.create_dataset("utf8").with_strings(&values).unwrap();
    b.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();

    let derived: Vec<FixedAscii<5>> = file
        .dataset("derived")
        .unwrap()
        .read_1d::<FixedAscii<5>>()
        .unwrap()
        .to_vec();
    assert_eq!(
        derived.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        values
    );

    let sized: Vec<FixedAscii<16>> = file
        .dataset("sized")
        .unwrap()
        .read_1d::<FixedAscii<16>>()
        .unwrap()
        .to_vec();
    assert_eq!(sized.iter().map(|s| s.as_str()).collect::<Vec<_>>(), values);

    let utf8: Vec<FixedUnicode<5>> = file
        .dataset("utf8")
        .unwrap()
        .read_1d::<FixedUnicode<5>>()
        .unwrap()
        .to_vec();
    assert_eq!(utf8.iter().map(|s| s.as_str()).collect::<Vec<_>>(), values);
}

/// The charset bit is load-bearing on the C side, not decoration: `FixedAscii`
/// and `FixedUnicode` are separate C-side types, and the library will not hand a
/// dataset over as the one it was not written as.
#[test]
fn the_c_library_holds_each_charset_to_its_own_reader() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("charsets.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("ascii")
        .with_ascii_strings(&["north"])
        .unwrap();
    b.create_dataset("utf8").with_strings(&["north"]).unwrap();
    b.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    assert!(
        file.dataset("utf8")
            .unwrap()
            .read_1d::<FixedAscii<5>>()
            .is_err(),
        "a UTF-8 dataset was handed over as ASCII"
    );
    assert!(
        file.dataset("ascii")
            .unwrap()
            .read_1d::<FixedUnicode<5>>()
            .is_err(),
        "an ASCII dataset was handed over as UTF-8"
    );
}

/// A multi-byte value is stored as bytes and read back as characters. The width
/// is 6 for a five-character value, which is the whole reason the UTF-8 entry
/// point measures what it does.
#[test]
fn the_c_library_reads_a_multi_byte_value_whole() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("units.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("units")
        .with_strings(&["mètre", "K", "°C"])
        .unwrap();
    b.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("units").unwrap();
    assert_eq!(
        ds.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::FixedUnicode(6)
    );
    let read: Vec<FixedUnicode<6>> = ds.read_1d::<FixedUnicode<6>>().unwrap().to_vec();
    assert_eq!(
        read.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ["mètre", "K", "°C"]
    );
}

/// A dataset whose values are all empty declares one byte rather than zero.
/// libhdf5 refuses a zero-width string datatype outright, so this is the case
/// where "readable at all" is the assertion.
#[test]
fn the_c_library_opens_an_all_empty_fixed_string_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("blank.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("blank")
        .with_ascii_strings(&["", "", ""])
        .unwrap();
    b.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("blank").unwrap();
    assert_eq!(
        ds.dtype().unwrap().to_descriptor().unwrap(),
        TypeDescriptor::FixedAscii(1)
    );
    let read: Vec<FixedAscii<1>> = ds.read_1d::<FixedAscii<1>>().unwrap().to_vec();
    assert_eq!(
        read.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ["", "", ""]
    );
}

/// The C library must be able to *extend* one too, since a declared width exists
/// so that later values fit: it appends through its own writer and this crate
/// reads every element back, so the two agree on the stride after a resize as
/// well as before one.
#[test]
fn the_c_library_extends_a_declared_width_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stations.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("station")
        .with_ascii_strings_sized(&["north", "s"], 16)
        .unwrap()
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4]);
    b.write(&path).unwrap();

    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let ds = file.dataset("station").unwrap();
        ds.resize([4]).unwrap();
        ds.write_slice(
            &[
                FixedAscii::<16>::from_ascii("north-northeast").unwrap(),
                FixedAscii::<16>::from_ascii("e").unwrap(),
            ],
            2..4,
        )
        .unwrap();
    }

    let file = hdf5_pure::File::open(&path).unwrap();
    let ds = file.dataset("station").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(
        ds.read_string().unwrap(),
        ["north", "s", "north-northeast", "e"]
    );
}
