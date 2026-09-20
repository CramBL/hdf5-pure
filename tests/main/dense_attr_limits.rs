//! Dense (fractal-heap) attribute storage and the bounds that remain on it.
//!
//! An attribute too large to be a managed heap object is written as a fractal-heap
//! *huge* object: its bytes sit outside the managed blocks and a huge-objects v2
//! B-tree maps a generated ID to their address and length (issue #195). Before
//! that it was refused outright, which was itself a fix for writing a heap that
//! read back empty and aborted an assertion-enabled reference C library (#191).
//!
//! Dense storage is selected by either of two things: more than eight
//! attributes, or one attribute too large for an object-header message. Most
//! cases here use nine or more to reach it by count; the ones that deliberately
//! use fewer reach it by size, and say so.

use hdf5_pure::{AttrValue, Error, File, FileBuilder, FormatError};

use test_util::heap::{
    has_fractal_heap, huge_object_bytes, huge_object_count, indirect_block_count,
    managed_object_count, name_index_leaf_records, root_indirect_rows,
};

/// Nine attributes, the first sized to `payload` bytes of text and the rest
/// small, which is enough to select dense storage.
fn nine_attrs(payload: usize) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder
}

/// The text of a string attribute, whichever string variant the reader chose for
/// it. Which one it picks is a datatype question and not what these tests are
/// about; the content is.
#[track_caller]
fn text<'a>(attrs: &'a std::collections::HashMap<String, AttrValue>, name: &str) -> &'a str {
    match attrs.get(name) {
        Some(AttrValue::AsciiString(s) | AttrValue::String(s)) => s,
        other => panic!("expected string attribute {name:?}, got {other:?}"),
    }
}

/// The largest text payload that still fits a managed heap object, found by
/// probing down. The threshold is on the *serialized* attribute, whose overhead
/// is not visible from outside the crate, so it is measured rather than assumed.
fn largest_managed_payload() -> usize {
    for payload in (1..=70_000).rev() {
        let bytes = nine_attrs(payload)
            .finish()
            .expect("every size is writable");
        if huge_object_count(&bytes) == 0 {
            return payload;
        }
    }
    panic!("no payload was stored as a managed object");
}

#[test]
fn an_attribute_past_the_managed_object_limit_becomes_a_huge_object() {
    let payload = largest_managed_payload();

    let at_limit = nine_attrs(payload).finish().unwrap();
    assert_eq!(huge_object_count(&at_limit), 0);
    assert_eq!(
        File::from_bytes(at_limit)
            .unwrap()
            .root()
            .attrs()
            .unwrap()
            .len(),
        9
    );

    // One byte more changes the storage class, and nothing else.
    let past = nine_attrs(payload + 1).finish().unwrap();
    assert_eq!(huge_object_count(&past), 1);

    let file = File::from_bytes(past).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    let big = text(&attrs, "big");
    assert_eq!(big.len(), payload + 1);
    assert!(big.bytes().all(|b| b == b'y'), "huge object bytes differ");
}

/// Dense storage is not chosen by attribute count alone. An attribute too large
/// for an object-header message forces it on its own, which is the only way a
/// *single* large attribute can be written at all — the compact path's answer to
/// one is `AttributeMessageTooLarge`, no matter that dense storage could hold it.
///
/// This mirrors the reference C library's own disjunction, and it is what makes
/// the ordinary case work: one big attribute on an object with no others.
#[test]
fn one_large_attribute_forces_dense_storage_on_its_own() {
    for others in [0usize, 2, 7] {
        let mut builder = FileBuilder::new();
        builder.set_attr("big", AttrValue::AsciiString("y".repeat(100_000)));
        for i in 0..others {
            builder.set_attr(&format!("a{i}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);

        let bytes = builder
            .finish()
            .unwrap_or_else(|e| panic!("{} attributes refused: {e}", others + 1));
        assert_eq!(
            huge_object_count(&bytes),
            1,
            "with {} attributes the large one should be a huge object",
            others + 1
        );

        let file = File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(attrs.len(), others + 1);
        assert_eq!(text(&attrs, "big").len(), 100_000);
    }
}

/// The other half of that disjunction still holds: a small set of small
/// attributes stays in the object header, so the size rule has not quietly
/// promoted every file to dense storage.
#[test]
fn small_attributes_below_the_count_threshold_stay_compact() {
    let mut builder = FileBuilder::new();
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::AsciiString("y".repeat(1_000)));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert!(
        !has_fractal_heap(&bytes),
        "eight small attributes must stay compact"
    );

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.root().attrs().unwrap().len(), 8);
}

/// The shape reported in #191: nine attributes of 100,000 bytes each. It was
/// silently lost, then refused, and is now written — every one of them as a huge
/// object, since each is far past the managed limit.
#[test]
fn the_reported_shape_round_trips_through_huge_storage() {
    let mut builder = FileBuilder::new();
    for i in 0..9 {
        builder.set_attr(
            &format!("a{i}"),
            AttrValue::AsciiString("y".repeat(100_000)),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reported.h5");
    builder.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(huge_object_count(&bytes), 9);

    let file = File::open(&path).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    for i in 0..9 {
        assert_eq!(text(&attrs, &format!("a{i}")).len(), 100_000);
    }
}

/// Managed and huge objects share one heap, and an attribute set that mixes them
/// has to keep both indexes straight: the direct block is sized by the managed
/// objects alone, while the name index spans all of them.
#[test]
fn a_mixed_managed_and_huge_set_round_trips() {
    let mut builder = FileBuilder::new();
    for i in 0..6 {
        builder.set_attr(
            &format!("small{i}"),
            AttrValue::AsciiString("y".repeat(100)),
        );
    }
    for i in 0..3 {
        builder.set_attr(
            &format!("big{i}"),
            AttrValue::AsciiString("z".repeat(80_000)),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    // The heap counts the two classes separately and every attribute is in
    // exactly one of them, so a miscount here is a heap that describes itself
    // wrongly even if the data survives.
    assert_eq!(huge_object_count(&bytes), 3);
    assert_eq!(managed_object_count(&bytes), 6);
    let declared = huge_object_bytes(&bytes);
    assert!(
        (3 * 80_000..3 * 80_000 + 1_000).contains(&declared),
        "declared huge size {declared} does not account for three 80,000-byte payloads"
    );

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    for i in 0..6 {
        let s = text(&attrs, &format!("small{i}"));
        assert_eq!(s.len(), 100);
        assert!(s.bytes().all(|b| b == b'y'), "managed object bytes differ");
    }
    for i in 0..3 {
        let s = text(&attrs, &format!("big{i}"));
        assert_eq!(s.len(), 80_000);
        assert!(s.bytes().all(|b| b == b'z'), "huge object bytes differ");
    }
}

/// The emitter grows a doubling table of direct blocks, so a multi-megabyte set
/// of individually small attributes stays entirely managed.
#[test]
fn a_multi_megabyte_set_of_small_attributes_stays_managed() {
    let mut builder = FileBuilder::new();
    for i in 0..40 {
        builder.set_attr(&format!("a{i}"), AttrValue::AsciiString("y".repeat(60_000)));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert!(bytes.len() > 2_000_000, "expected a multi-megabyte heap");
    assert_eq!(huge_object_count(&bytes), 0);

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.root().attrs().unwrap().len(), 40);
}

/// The managed attributes used to live in one direct block rounded up to a power
/// of two, which capped the heap at the 2 GiB block the format allows and spent
/// close to half its space on padding. They now go in a doubling table of blocks
/// no larger than 64 KiB, reached through a root indirect block and, past the
/// root's own row of those, through nested ones.
///
/// Both depths are worth reading back through this crate as well as through the
/// C library: an attribute placed at the wrong heap offset still decodes, because
/// its neighbour's bytes are a valid message too.
#[test]
fn managed_attributes_span_as_many_blocks_as_they_need() {
    // Small attributes spilling out of one block, and attributes each nearly
    // filling the largest direct block, which is what forces the nesting.
    for (name, payload, count, indirect) in [("shallow", 200, 60, 1), ("nested", 64_000, 12, 2)] {
        let mut builder = FileBuilder::new();
        for i in 0..count {
            let text = char::from(b'a' + (i % 26) as u8)
                .to_string()
                .repeat(payload);
            builder.set_attr(&format!("a{i:04}"), AttrValue::AsciiString(text));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);

        let bytes = builder.finish().unwrap();
        assert_eq!(
            huge_object_count(&bytes),
            0,
            "{name}: these attributes must stay managed"
        );
        assert!(
            root_indirect_rows(&bytes) > 0,
            "{name}: the root should have grown into an indirect block"
        );
        assert!(
            indirect_block_count(&bytes) >= indirect,
            "{name}: expected at least {indirect} indirect blocks, found {}",
            indirect_block_count(&bytes)
        );

        let file = File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(attrs.len(), count);
        for i in 0..count {
            let expected = char::from(b'a' + (i % 26) as u8)
                .to_string()
                .repeat(payload);
            assert_eq!(text(&attrs, &format!("a{i:04}")), expected, "{name}");
        }
    }
}

/// The attribute count used to stop at 61,680, where a single B-tree leaf grown
/// to hold every record pushed the capacity width the reference C library derives
/// past the 2 bytes it allots. A name index of fixed 512-byte nodes grows a level
/// instead, so counts on both sides of that old bound are ordinary now.
#[test]
fn the_attribute_count_has_no_bound_at_the_old_single_leaf_limit() {
    let build = |n: usize| {
        let mut builder = FileBuilder::new();
        for i in 0..n {
            builder.set_attr(&format!("a{i:06}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder
    };

    // Just under, just over, and well past the count that used to be refused.
    for count in [61_680, 61_681, 70_000] {
        let bytes = build(count)
            .finish()
            .unwrap_or_else(|e| panic!("{count} attributes should be writable, got {e:?}"));
        let file = File::from_bytes(bytes).unwrap();
        let attrs = file.root().attrs().unwrap();
        assert_eq!(attrs.len(), count);
        // Reading every name back proves the tree's in-order traversal reached
        // every leaf, not just that the count in the header was right.
        for i in 0..count {
            let name = format!("a{i:06}");
            assert!(
                matches!(attrs.get(&name), Some(AttrValue::I64(v)) if *v == i as i64),
                "attribute {name} did not read back"
            );
        }
    }
}

/// Huge storage lifts the limit on an attribute's *data*, not on the fields that
/// describe it: the message encodes its name, datatype and dataspace lengths in
/// 2-byte fields, so a name past that must still be refused rather than truncated
/// into a message that decodes as something else.
#[test]
fn an_attribute_name_too_long_for_its_length_field_is_refused() {
    let mut builder = FileBuilder::new();
    builder.set_attr(&"n".repeat(70_000), AttrValue::I64(1));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    match builder.finish().unwrap_err() {
        Error::Format(FormatError::AttributeFieldTooLong {
            field, size, limit, ..
        }) => {
            assert_eq!(field, "name");
            assert_eq!(size, 70_001, "the null terminator counts toward the field");
            assert_eq!(limit, u16::MAX as usize);
        }
        other => panic!("expected AttributeFieldTooLong, got {other:?}"),
    }

    // A name one byte inside the field is written, and comes back intact.
    let name = "n".repeat(u16::MAX as usize - 1);
    let mut builder = FileBuilder::new();
    builder.set_attr(&name, AttrValue::I64(1));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 9);
    assert!(attrs.contains_key(&name));
}

/// Dense storage applies to group and dataset attributes too, and both reach the
/// same emitter.
#[test]
fn group_and_dataset_attributes_use_huge_storage_too() {
    let payload = largest_managed_payload() + 1;

    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    group.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        group.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    group.create_dataset("y").with_f64_data(&[1.0]);
    builder.add_group(group.finish());

    let bytes = builder.finish().unwrap();
    assert_eq!(huge_object_count(&bytes), 1);
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.group("g").unwrap().attrs().unwrap().len(), 9);

    let mut builder = FileBuilder::new();
    let ds = builder.create_dataset("x").with_f64_data(&[1.0]);
    ds.set_attr("big", AttrValue::AsciiString("y".repeat(payload)));
    for i in 0..8 {
        ds.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }

    let bytes = builder.finish().unwrap();
    assert_eq!(huge_object_count(&bytes), 1);
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.dataset("x").unwrap().attrs().unwrap().len(), 9);
}

/// Two attribute names whose hashes collide must be indexed in the order the
/// names compare, not the order they were set.
///
/// The name index is searched by name hash and, on a tie, by the name pulled back
/// out of the heap — `H5A__dense_fh_name_cmp` does a `strcmp` — so a colliding
/// pair stored in insertion order sends a binary search down the wrong branch and
/// one of the two becomes unfindable by name. Reading every attribute back walks
/// the node start to finish and cannot see it, which is why this asserts on the
/// record order itself. The C-library half is
/// `c_finds_attributes_whose_names_hash_alike`.
#[test]
fn colliding_name_hashes_are_indexed_in_name_order() {
    // Distinct names with the same jenkins_lookup3 hash, set in the order that
    // is the reverse of how they compare: "k155448" sorts before "k69209".
    const FIRST: &str = "k69209";
    const SECOND: &str = "k155448";
    assert!(SECOND < FIRST, "the fixture pair is not in reverse order");

    let mut builder = FileBuilder::new();
    builder.set_attr(FIRST, AttrValue::I64(11));
    builder.set_attr(SECOND, AttrValue::I64(22));
    for i in 0..8 {
        builder.set_attr(&format!("f{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert!(has_fractal_heap(&bytes), "ten attributes are dense");

    let records = name_index_leaf_records(&bytes, 10);
    let hash_of = |order: u32| {
        records
            .iter()
            .find(|(o, _)| *o == order)
            .expect("a record per attribute")
            .1
    };
    assert_eq!(
        hash_of(0),
        hash_of(1),
        "the fixture names no longer hash alike; pick a new colliding pair"
    );

    let colliding: Vec<u32> = records
        .iter()
        .map(|(order, _)| *order)
        .filter(|order| *order < 2)
        .collect();
    assert_eq!(
        colliding,
        vec![1, 0],
        "the colliding records are in insertion order, not name order"
    );

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 10);
    assert_eq!(attrs.get(FIRST), Some(&AttrValue::I64(11)));
    assert_eq!(attrs.get(SECOND), Some(&AttrValue::I64(22)));
}
