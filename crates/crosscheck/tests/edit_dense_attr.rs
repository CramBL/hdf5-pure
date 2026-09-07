// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! In-place attribute edits that land in dense (fractal-heap) storage (issue #102).
//!
//! `File::open_rw` used to refuse any attribute edit that touched a heap: an
//! object already storing its attributes densely was refused by name, and a
//! compact object whose edit crossed the eight-attribute threshold was refused as
//! "would exceed compact storage" — even though the whole-file writer emits that
//! heap, and this crate's reader reads it. These tests drive the edits that used
//! to be refused and check the result in this crate's reader *and* the reference C
//! library, since a heap this crate alone can read is not the point.

use hdf5_pure::{
    AttrValue, Error, File, FileAccessProperties, FileBuilder, FormatError, MemoryStrategy,
    SyncPolicy,
};
use tempfile::tempdir;

use test_util::heap::{frhp_offsets, has_fractal_heap};

/// Past the writer's eight-attribute compact threshold.
const DENSE_COUNT: usize = 12;

fn dense_names() -> Vec<String> {
    (0..DENSE_COUNT).map(|i| format!("attr_{i:02}")).collect()
}

/// A file whose dataset `d` and group `g` each carry [`DENSE_COUNT`] attributes,
/// so both already use dense storage before any edit.
fn write_dense(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    {
        let ds = b.create_dataset("d");
        ds.with_i32_data(&[1, 2, 3, 4]);
        for (i, name) in dense_names().into_iter().enumerate() {
            ds.set_attr(&name, AttrValue::I64(i as i64));
        }
    }
    {
        let mut g = b.create_group("g");
        for (i, name) in dense_names().into_iter().enumerate() {
            g.set_attr(&name, AttrValue::I64(i as i64));
        }
        b.add_group(g.finish());
    }
    b.write(path).unwrap();
    assert!(
        has_fractal_heap(&std::fs::read(path).unwrap()),
        "the fixture must already store its attributes densely",
    );
}

/// A file whose dataset `d` carries three compact attributes.
fn write_compact(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    let ds = b.create_dataset("d");
    ds.with_i32_data(&[1, 2, 3, 4]);
    for i in 0..3 {
        ds.set_attr(&format!("attr_{i:02}"), AttrValue::I64(i));
    }
    b.write(path).unwrap();
    assert!(
        !has_fractal_heap(&std::fs::read(path).unwrap()),
        "the fixture must start out compact",
    );
}

/// Every attribute of `path`'s dataset `d`, read by the reference C library, so a
/// heap this crate builds is checked by the library that defines the format.
fn c_dataset_attrs(path: &std::path::Path) -> Vec<(String, i64)> {
    let f = hdf5::File::open(path).unwrap();
    let ds = f.dataset("d").unwrap();
    let mut out: Vec<(String, i64)> = ds
        .attr_names()
        .unwrap()
        .into_iter()
        .map(|name| {
            let v: i64 = ds.attr(&name).unwrap().read_scalar().unwrap();
            (name, v)
        })
        .collect();
    out.sort();
    out
}

/// One `i64` attribute read by the reference C library, from the dataset or the
/// group at `path`. The heap these tests build is only worth as much as the
/// library that defines the format can make of it, so every case checks at least
/// one value this way.
fn c_dataset_attr(file: &hdf5::File, dataset: &str, name: &str) -> i64 {
    file.dataset(dataset)
        .unwrap()
        .attr(name)
        .unwrap()
        .read_scalar()
        .unwrap()
}

fn c_group_attr(file: &hdf5::File, group: &str, name: &str) -> i64 {
    file.group(group)
        .unwrap()
        .attr(name)
        .unwrap()
        .read_scalar()
        .unwrap()
}

fn i64_attr(f: &File, path: &str, name: &str) -> i64 {
    match f.dataset(path).unwrap().attrs().unwrap().get(name) {
        Some(AttrValue::I64(v)) => *v,
        other => panic!("{name}: expected an i64 attribute, got {other:?}"),
    }
}

#[test]
fn a_compact_edit_crossing_the_threshold_moves_the_set_to_a_heap() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_compact(&p);

    {
        let s = File::open_rw(&p).unwrap();
        let mut d = s.dataset("d").unwrap();
        for i in 3..DENSE_COUNT {
            d.set_attr(&format!("attr_{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "crossing the compact threshold must move the set into a fractal heap",
    );
    let f = File::open(&p).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(d.read_i32().unwrap(), vec![1, 2, 3, 4], "data untouched");
    let attrs = d.attrs().unwrap();
    assert_eq!(attrs.len(), DENSE_COUNT);
    for i in 0..DENSE_COUNT {
        assert_eq!(
            attrs.get(&format!("attr_{i:02}")),
            Some(&AttrValue::I64(i as i64)),
            "attribute {i} did not survive the move to dense storage",
        );
    }
    // The three the file already carried came out of the header and back into a
    // heap object, so the C library reading them is what says the re-encoding was
    // faithful rather than merely self-consistent.
    assert_eq!(
        c_dataset_attrs(&p),
        (0..DENSE_COUNT)
            .map(|i| (format!("attr_{i:02}"), i as i64))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn a_dense_object_takes_a_new_attribute() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_dense(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("added", AttrValue::I64(99))
            .unwrap();
        s.group("g")
            .unwrap()
            .set_attr("added", AttrValue::I64(98))
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), DENSE_COUNT + 1);
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(99)));
    for (i, name) in dense_names().into_iter().enumerate() {
        assert_eq!(attrs.get(&name), Some(&AttrValue::I64(i as i64)));
    }
    let g = f.group("g").unwrap().attrs().unwrap();
    assert_eq!(g.len(), DENSE_COUNT + 1);
    assert_eq!(g.get("added"), Some(&AttrValue::I64(98)));

    let c = hdf5::File::open(&p).unwrap();
    assert_eq!(c_dataset_attr(&c, "d", "added"), 99);
    assert_eq!(c_dataset_attr(&c, "d", "attr_00"), 0);
    assert_eq!(c_group_attr(&c, "g", "added"), 98);
}

#[test]
fn a_dense_attribute_is_updated_in_place_and_keeps_its_position() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_dense(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("attr_05", AttrValue::I64(-5))
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), DENSE_COUNT, "an update must not add a member");
    assert_eq!(i64_attr(&f, "d", "attr_05"), -5);
    assert_eq!(i64_attr(&f, "d", "attr_11"), 11, "the rest are unchanged");
    assert_eq!(c_dataset_attrs(&p).len(), DENSE_COUNT);
}

#[test]
fn removing_one_dense_attribute_keeps_the_rest() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_dense(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d").unwrap().remove_attr("attr_03").unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), DENSE_COUNT - 1);
    assert!(!attrs.contains_key("attr_03"));
    assert_eq!(
        c_dataset_attrs(&p),
        (0..DENSE_COUNT)
            .filter(|i| *i != 3)
            .map(|i| (format!("attr_{i:02}"), i as i64))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn removing_every_dense_attribute_leaves_an_object_with_none() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    // Only the dataset is dense here, so the heaps in the file can be counted:
    // `write_dense`'s group would contribute one of its own.
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("d");
        ds.with_i32_data(&[1, 2, 3, 4]);
        for (i, name) in dense_names().into_iter().enumerate() {
            ds.set_attr(&name, AttrValue::I64(i as i64));
        }
        b.write(&p).unwrap();
    }
    let heaps_before = frhp_offsets(&std::fs::read(&p).unwrap()).len();
    assert_eq!(
        heaps_before, 1,
        "the fixture has exactly one attribute heap"
    );

    {
        let s = File::open_rw(&p).unwrap();
        let mut d = s.dataset("d").unwrap();
        for name in dense_names() {
            d.remove_attr(&name).unwrap();
        }
        s.commit().unwrap();
    }

    // An object with no attributes carries no Attribute Info message and no heap
    // — an *empty* heap would read back as zero attributes just the same, which
    // is why the count is what says which of the two was written. The one still
    // in the file is the superseded heap the rebuild left behind.
    assert!(
        frhp_offsets(&std::fs::read(&p).unwrap()).len() <= heaps_before,
        "removing the last attribute must not build a heap to hold none",
    );
    let f = File::open(&p).unwrap();
    let d = f.dataset("d").unwrap();
    assert!(d.attrs().unwrap().is_empty());
    assert_eq!(d.read_i32().unwrap(), vec![1, 2, 3, 4]);
    let c = hdf5::File::open(&p).unwrap();
    assert_eq!(c.dataset("d").unwrap().attr_names().unwrap().len(), 0);
}

#[test]
fn an_attribute_too_large_for_the_object_header_goes_to_a_heap() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_compact(&p);

    // 80,000 bytes of data: past the object header's 2-byte message-size field,
    // which is what used to make this "attribute is too large to encode in place"
    // however few attributes the object had.
    let big: Vec<i64> = (0..10_000).collect();
    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("big", AttrValue::I64Array(big.clone()))
            .unwrap();
        s.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "an attribute the header cannot describe must go to a heap",
    );
    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("big"), Some(&AttrValue::I64Array(big.clone())));
    assert_eq!(attrs.len(), 4, "the three compact ones came along");

    let c = hdf5::File::open(&p).unwrap();
    let got: Vec<i64> = c
        .dataset("d")
        .unwrap()
        .attr("big")
        .unwrap()
        .read_raw()
        .unwrap();
    assert_eq!(got, big);
}

#[test]
fn a_variable_length_attribute_survives_a_dense_rebuild() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_compact(&p);

    let strings = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
    {
        let s = File::open_rw(&p).unwrap();
        let mut d = s.dataset("d").unwrap();
        // The variable-length attribute is set *before* the edit crosses the
        // threshold, so it is rebuilt into the heap from a message whose element
        // references were still placeholders: the ordering this path has to keep.
        d.set_attr("names", AttrValue::VarLenAsciiCharArray(strings.clone()))
            .unwrap();
        for i in 3..DENSE_COUNT {
            d.set_attr(&format!("attr_{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "the edit crossed the threshold, so the set is in a heap",
    );
    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("names"),
        Some(&AttrValue::VarLenAsciiCharArray(strings.clone())),
    );

    // And again, editing the object that now stores it densely: the existing
    // variable-length attribute is read back out of the heap with resolved
    // references and re-emitted, which a placeholder-patching bug would break.
    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("added", AttrValue::I64(7))
            .unwrap();
        s.commit().unwrap();
    }
    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("names"),
        Some(&AttrValue::VarLenAsciiCharArray(strings.clone())),
        "the heap-resident variable-length attribute survived a second rebuild",
    );

    // The C library's view of the heap-resident attribute. This crate encodes a
    // `VarLenAsciiCharArray` as a variable-length sequence of one-character ASCII
    // strings, which is the type named here; what the read proves is that the
    // element references the rebuild patched resolve to the right heap bytes.
    let c = hdf5::File::open(&p).unwrap();
    let got: Vec<hdf5::types::VarLenArray<hdf5::types::FixedAscii<1>>> = c
        .dataset("d")
        .unwrap()
        .attr("names")
        .unwrap()
        .read_raw()
        .unwrap();
    assert_eq!(
        got.iter()
            .map(|word| word
                .as_slice()
                .iter()
                .map(|c| c.as_str())
                .collect::<String>())
            .collect::<Vec<_>>(),
        strings,
    );
}

#[test]
fn a_dataset_added_in_place_may_carry_a_dense_attribute_set() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_compact(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.root()
            .create_dataset("contig", |b| {
                b.with_i32_data(&[5, 6, 7]);
                for (i, name) in dense_names().into_iter().enumerate() {
                    b.set_attr(&name, AttrValue::I64(i as i64));
                }
            })
            .unwrap();
        s.root()
            .create_dataset("chunked", |b| {
                b.with_i32_data(&(0..64).collect::<Vec<_>>())
                    .with_shape(&[64])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[16]);
                for (i, name) in dense_names().into_iter().enumerate() {
                    b.set_attr(&name, AttrValue::I64(i as i64));
                }
            })
            .unwrap();
        s.commit().unwrap();
    }

    // Twelve *compact* attributes would read back just as well, so the storage
    // the writer chose is what this test is about: the fixture had no heap.
    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "a dataset added with more attributes than the header holds must get a heap",
    );
    let f = File::open(&p).unwrap();
    for name in ["contig", "chunked"] {
        let attrs = f.dataset(name).unwrap().attrs().unwrap();
        assert_eq!(attrs.len(), DENSE_COUNT, "{name}");
        for (i, attr) in dense_names().into_iter().enumerate() {
            assert_eq!(attrs.get(&attr), Some(&AttrValue::I64(i as i64)), "{name}");
        }
    }
    assert_eq!(
        f.dataset("contig").unwrap().read_i32().unwrap(),
        vec![5, 6, 7]
    );
    assert_eq!(
        f.dataset("chunked").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<_>>(),
    );

    let c = hdf5::File::open(&p).unwrap();
    for name in ["contig", "chunked"] {
        assert_eq!(
            c.dataset(name).unwrap().attr_names().unwrap().len(),
            DENSE_COUNT,
            "{name}"
        );
        assert_eq!(c_dataset_attr(&c, name, "attr_07"), 7, "{name}");
    }
}

#[test]
fn a_group_created_in_place_may_carry_a_dense_attribute_set() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_compact(&p);

    {
        let s = File::open_rw(&p).unwrap();
        s.root()
            .create_group_with("fresh", |g| {
                for (i, name) in dense_names().into_iter().enumerate() {
                    g.set_attr(&name, AttrValue::I64(i as i64));
                }
            })
            .unwrap();
        s.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "a group created with more attributes than the header holds must get a heap",
    );
    let f = File::open(&p).unwrap();
    let attrs = f.group("fresh").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), DENSE_COUNT);
    for (i, name) in dense_names().into_iter().enumerate() {
        assert_eq!(attrs.get(&name), Some(&AttrValue::I64(i as i64)));
    }

    let c = hdf5::File::open(&p).unwrap();
    assert_eq!(c_group_attr(&c, "fresh", "attr_09"), 9);
}

#[test]
fn the_root_group_takes_a_dense_attribute_set() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_compact(&p);

    {
        let s = File::open_rw(&p).unwrap();
        for (i, name) in dense_names().into_iter().enumerate() {
            s.root().set_attr(&name, AttrValue::I64(i as i64)).unwrap();
        }
        s.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "the root group's set outgrew its header and must be in a heap",
    );
    let f = File::open(&p).unwrap();
    let attrs = f.root().attrs().unwrap();
    assert_eq!(attrs.len(), DENSE_COUNT);
    assert_eq!(attrs.get("attr_04"), Some(&AttrValue::I64(4)));
    // The dataset's own compact attributes are a different object's storage and
    // must be untouched by the root's move to a heap.
    assert_eq!(f.dataset("d").unwrap().attrs().unwrap().len(), 3);

    let c = hdf5::File::open(&p).unwrap();
    assert_eq!(c_group_attr(&c, "/", "attr_04"), 4);
}

#[test]
fn a_dense_edit_keeps_a_chunked_dataset_readable() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("d");
        ds.with_i32_data(&(0..64).collect::<Vec<_>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[16]);
        for (i, name) in dense_names().into_iter().enumerate() {
            ds.set_attr(&name, AttrValue::I64(i as i64));
        }
        b.write(&p).unwrap();
    }

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("added", AttrValue::I64(1))
            .unwrap();
        s.commit().unwrap();
    }

    // The header moved; the chunk index and chunk data did not.
    let f = File::open(&p).unwrap();
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<_>>(),
    );
    let c = hdf5::File::open(&p).unwrap();
    assert_eq!(
        c.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        (0..64).collect::<Vec<_>>(),
    );
}

/// The bounded-memory backing commits through the same engine, but it mirrors a
/// window of the file rather than holding all of it, so a heap it places and a
/// heap it reads back are two different paths through that window.
#[test]
fn a_bounded_session_edits_a_dense_attribute_set() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_dense(&p);

    {
        let s = File::open_rw_with_options(
            &p,
            FileAccessProperties::new()
                .with_memory_strategy(MemoryStrategy::Bounded)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("added", AttrValue::I64(42))
            .unwrap();
        s.dataset("d").unwrap().remove_attr("attr_00").unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), DENSE_COUNT);
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(42)));
    assert!(!attrs.contains_key("attr_00"));
    assert_eq!(
        f.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );

    let c = hdf5::File::open(&p).unwrap();
    assert_eq!(c_dataset_attr(&c, "d", "added"), 42);
}

/// Heap storage lifts the limit on an attribute's *data*, not on the fields that
/// describe it: the attribute message's name, datatype and dataspace lengths are
/// 2-byte fields whatever storage holds the message. A name past that is the one
/// attribute edit a heap cannot rescue, and it is refused in the preflight — so
/// the file it was staged against is byte-for-byte unchanged.
#[test]
fn an_attribute_name_past_its_length_field_is_refused_without_writing() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("d.h5");
    write_compact(&p);
    let before = std::fs::read(&p).unwrap();

    let name = "n".repeat(u16::MAX as usize + 1);
    let s = File::open_rw(&p).unwrap();
    s.dataset("d")
        .unwrap()
        .set_attr(&name, AttrValue::I64(1))
        .unwrap();
    let err = s.commit().unwrap_err();
    assert!(
        matches!(
            err,
            Error::Format(FormatError::AttributeFieldTooLong { field: "name", .. })
        ),
        "expected the name-length refusal, got {err:?}",
    );
    drop(s);
    assert_eq!(std::fs::read(&p).unwrap(), before);
}

/// A commit's last act is to repoint the object references the file already
/// stores, and it reaches an attribute's value only while that attribute lives in
/// the object header — `reference_patch::scan_object` collects nothing from an
/// object whose attributes are in a heap. So an attribute holding a reference is
/// not moved to one, and the refusal comes before any byte is written.
///
/// The fixture has to come from the C library: `AttrValue` has no reference
/// variant, so no pure-Rust API stages such an attribute. The latest-format
/// bounds are what give the objects the version 2 headers this engine edits.
#[test]
fn an_attribute_holding_an_object_reference_is_not_moved_to_a_heap() {
    use hdf5::file::LibraryVersion;
    use hdf5::{ObjectReference, ObjectReference1};

    let dir = tempdir().unwrap();
    let p = dir.path().join("ref_attr.h5");
    {
        let c = hdf5::File::with_options()
            .with_fapl(|f| f.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&p)
            .unwrap();
        c.create_group("g").unwrap();
        let holder = c.new_dataset::<i32>().shape((1,)).create("d").unwrap();
        holder.write(&[0i32]).unwrap();
        holder
            .new_attr::<ObjectReference1>()
            .shape((1,))
            .create("target")
            .unwrap()
            .write(&[ObjectReference1::create(&c, "g").unwrap()])
            .unwrap();
        for i in 0..4 {
            holder
                .new_attr::<i64>()
                .create(format!("a{i}").as_str())
                .unwrap()
                .write_scalar(&(i as i64))
                .unwrap();
        }
        c.close().unwrap();
    }

    // Five attributes, one of them a reference: still compact, and editable.
    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("a4", AttrValue::I64(4))
            .unwrap();
        s.commit().unwrap();
    }
    let before = std::fs::read(&p).unwrap();
    assert!(
        !has_fractal_heap(&before),
        "six attributes still fit the object header",
    );

    // Three more would be nine, which is where the set moves to a heap — and
    // where the reference would stop being repointable.
    let s = File::open_rw(&p).unwrap();
    let mut d = s.dataset("d").unwrap();
    for i in 5..8 {
        d.set_attr(&format!("a{i}"), AttrValue::I64(i)).unwrap();
    }
    let err = s.commit().unwrap_err();
    assert!(
        err.to_string().contains("object reference"),
        "expected the reference refusal, got: {err}",
    );
    // The dataset handle keeps the session — and its exclusive lock on the file —
    // alive, so it has to go before the read below: mandatory on Windows,
    // advisory (and so invisible) everywhere else.
    drop(d);
    drop(s);
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "the refusal wrote nothing"
    );
}

/// The refusal above is on *moving* a reference attribute out of the header, not
/// on rebuilding a heap that already holds one. An object the C library already
/// stored densely keeps its references outside the repointing walk whatever this
/// edit does, so refusing there would cost the capability and buy nothing — and
/// the reference has to come back out of the rebuilt heap intact.
#[test]
fn an_already_dense_object_keeps_its_reference_attribute_through_a_rebuild() {
    use hdf5::file::LibraryVersion;
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let p = dir.path().join("dense_ref_attr.h5");
    {
        let c = hdf5::File::with_options()
            .with_fapl(|f| f.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&p)
            .unwrap();
        c.create_group("g").unwrap();
        let holder = c.new_dataset::<i32>().shape((1,)).create("d").unwrap();
        holder.write(&[0i32]).unwrap();
        holder
            .new_attr::<ObjectReference1>()
            .shape((1,))
            .create("target")
            .unwrap()
            .write(&[ObjectReference1::create(&c, "g").unwrap()])
            .unwrap();
        // Past max_compact, so the C library puts the whole set — the reference
        // attribute with it — in a fractal heap.
        for i in 0..DENSE_COUNT {
            holder
                .new_attr::<i64>()
                .create(format!("a{i:02}").as_str())
                .unwrap()
                .write_scalar(&(i as i64))
                .unwrap();
        }
        c.close().unwrap();
    }
    assert!(has_fractal_heap(&std::fs::read(&p).unwrap()));

    {
        let s = File::open_rw(&p).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("added", AttrValue::I64(7))
            .unwrap();
        s.commit().unwrap();
    }

    // Counted through the C library: `AttrValue` has no reference variant, so
    // `attrs()` omits the reference attribute whether or not it survived, and
    // would report the same 13 either way.
    let c = hdf5::File::open(&p).unwrap();
    let ds = c.dataset("d").unwrap();
    assert_eq!(ds.attr_names().unwrap().len(), DENSE_COUNT + 2);
    let v: i64 = ds.attr("added").unwrap().read_scalar().unwrap();
    assert_eq!(v, 7);
    let refs = ds
        .attr("target")
        .unwrap()
        .read_raw::<ObjectReference1>()
        .unwrap();
    match refs[0].dereference(&c).unwrap() {
        ReferencedObject::Group(_) => {}
        other => panic!("the rebuilt heap lost the reference: {other:?}"),
    }
}

/// The refusal above asks the repointing walk's own predicate, not "does this
/// datatype hold an address anywhere". The difference is a whole class of real
/// file: a dimension scale's `DIMENSION_LIST` is a *variable length of*
/// references, which the walk never repoints even with the attribute sitting in
/// the object header, so moving it to a heap costs nothing and refusing it would
/// cost the edit for no gain.
#[test]
fn a_variable_length_of_references_does_not_block_the_move_to_a_heap() {
    use hdf5::file::LibraryVersion;
    use hdf5::types::VarLenArray;
    use hdf5::{ObjectReference, ObjectReference1, ReferencedObject};

    let dir = tempdir().unwrap();
    let p = dir.path().join("dimension_list.h5");
    {
        let c = hdf5::File::with_options()
            .with_fapl(|f| f.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&p)
            .unwrap();
        c.create_group("g").unwrap();
        let d = c.new_dataset::<i32>().shape((1,)).create("d").unwrap();
        d.write(&[0i32]).unwrap();
        let r = ObjectReference1::create(&c, "g").unwrap();
        d.new_attr::<VarLenArray<ObjectReference1>>()
            .shape((1,))
            .create("DIMENSION_LIST")
            .unwrap()
            .write(&[VarLenArray::from_slice(&[r])])
            .unwrap();
        c.close().unwrap();
    }

    {
        let s = File::open_rw(&p).unwrap();
        let mut d = s.dataset("d").unwrap();
        for i in 0..9 {
            d.set_attr(&format!("a{i}"), AttrValue::I64(i)).unwrap();
        }
        s.commit().unwrap();
    }

    assert!(
        has_fractal_heap(&std::fs::read(&p).unwrap()),
        "ten attributes are past what the object header holds",
    );
    let c = hdf5::File::open(&p).unwrap();
    let ds = c.dataset("d").unwrap();
    assert_eq!(ds.attr_names().unwrap().len(), 10);
    let lists = ds
        .attr("DIMENSION_LIST")
        .unwrap()
        .read_raw::<VarLenArray<ObjectReference1>>()
        .unwrap();
    match lists[0][0].dereference(&c).unwrap() {
        ReferencedObject::Group(_) => {}
        other => panic!("the rebuilt heap lost the reference list: {other:?}"),
    }
}
