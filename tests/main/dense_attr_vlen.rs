//! Variable-length attributes in dense (fractal-heap) storage.
//!
//! A variable-length attribute's element bytes hold global-heap references that
//! only get real addresses late in the writer's layout pass. The dense heap
//! embeds a *copy* of those bytes, so a heap built before that patching stored
//! placeholders — and this crate's reader, resolving them through the global
//! heap, dropped the attribute rather than reporting anything. The reference C
//! library still listed it, since name, datatype and dataspace live in the
//! message header and only the data was unreachable.
//!
//! The writer now reserves each heap's span during address assignment and fills
//! it in after the references are patched. These tests cover both of the layouts
//! that do so, since the reserve and the build sit on opposite sides of a branch.

use hdf5_pure::{AttrValue, File, FileBuilder, FileSpaceStrategy};

use test_util::heap::{has_fractal_heap, huge_object_count};

/// A builder whose root carries one variable-length string attribute alongside
/// `others` small ones — enough of them to select dense storage by count, not by
/// the attribute's size.
fn root_with_vlen(others: usize, labels: &[&str]) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder.set_attr("labels", vlen(labels));
    for i in 0..others {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i as i64));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder
}

fn vlen(labels: &[&str]) -> AttrValue {
    AttrValue::VarLenAsciiCharArray(labels.iter().map(|s| (*s).to_string()).collect())
}

/// The strings of an array-of-strings attribute, whichever variant the reader
/// chose for it. Which one it picks is a datatype question; the content is what
/// these tests are about.
#[track_caller]
fn labels(attrs: &std::collections::HashMap<String, AttrValue>, name: &str) -> Vec<String> {
    match attrs.get(name) {
        Some(AttrValue::VarLenAsciiCharArray(v) | AttrValue::StringArray(v)) => v.clone(),
        other => panic!("expected an array-of-strings attribute {name:?}, got {other:?}"),
    }
}

#[test]
fn a_variable_length_attribute_survives_dense_storage() {
    let expected = ["alpha", "beta", "gamma"];
    let bytes = root_with_vlen(12, &expected).finish().unwrap();
    assert!(has_fractal_heap(&bytes));

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 13, "the whole set must read back");
    assert_eq!(labels(&attrs, "labels"), expected);
}

/// The paged layout places its global-heap collections in the metadata region
/// rather than after the dataset data, and reaches the heap build by a different
/// route. Both routes have to defer it, so both are pinned.
#[test]
fn a_paged_file_stores_variable_length_attributes_densely_too() {
    let expected = ["one", "two"];
    let mut builder = root_with_vlen(12, &expected);
    builder
        .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(4096);

    let bytes = builder.finish().unwrap();
    assert!(has_fractal_heap(&bytes));

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 13);
    assert_eq!(labels(&attrs, "labels"), expected);
}

/// Groups and datasets reach the writer's storage choice by their own routes, and
/// their heaps are reserved in their own loops.
#[test]
fn group_and_dataset_variable_length_attributes_survive_too() {
    let expected = ["p", "qq", "rrr"];
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("x").with_f64_data(&[1.0]);
    dataset.set_attr("labels", vlen(&expected));
    for i in 0..12 {
        dataset.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    let mut group = builder.create_group("g");
    group.set_attr("labels", vlen(&expected));
    for i in 0..12 {
        group.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    group.create_dataset("y").with_f64_data(&[2.0]);
    builder.add_group(group.finish());

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    for attrs in [
        file.dataset("x").unwrap().attrs().unwrap(),
        file.group("g").unwrap().attrs().unwrap(),
    ] {
        assert_eq!(attrs.len(), 13);
        assert_eq!(labels(&attrs, "labels"), expected);
    }
}

/// One variable-length attribute large enough to be stored as a *huge* object,
/// where its bytes sit outside the heap's managed blocks. That is also the case
/// that selects dense storage on size alone, so no filler attributes are needed
/// to reach it.
#[test]
fn a_huge_variable_length_attribute_keeps_its_values() {
    let expected: Vec<String> = (0..4_200).map(|i| format!("s{i}")).collect();
    let mut builder = FileBuilder::new();
    builder.set_attr("labels", AttrValue::VarLenAsciiCharArray(expected.clone()));
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let bytes = builder.finish().unwrap();
    assert_eq!(
        huge_object_count(&bytes),
        1,
        "16 bytes per element puts this past the managed-object limit"
    );

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(labels(&file.root().attrs().unwrap(), "labels"), expected);
}

/// Several variable-length attributes on one object, with distinct values, so a
/// heap that patched them all with the *same* collection address reads back wrong
/// rather than coincidentally right.
#[test]
fn several_variable_length_attributes_keep_their_own_values() {
    let mut builder = FileBuilder::new();
    let sets: Vec<Vec<String>> = (0..4)
        .map(|i| (0..3).map(|j| format!("v{i}_{j}")).collect())
        .collect();
    for (i, set) in sets.iter().enumerate() {
        builder.set_attr(
            &format!("l{i}"),
            AttrValue::VarLenAsciiCharArray(set.clone()),
        );
    }
    for i in 0..12 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), 16);
    for (i, set) in sets.iter().enumerate() {
        assert_eq!(labels(&attrs, &format!("l{i}")), *set);
    }
}

/// A *chunked* variable-length dataset places its global-heap collections early,
/// before the object headers, rather than after the dataset data — a different
/// placement path from the contiguous case below, and the one that most nearly
/// collides with the reserved heap spans. Run in both layouts, since the paged one
/// puts collections somewhere else again.
#[test]
fn chunked_variable_length_elements_coexist_with_dense_attributes() {
    let expected = ["attr-a", "attr-b", "attr-c"];
    let elements: Vec<String> = (0..7).map(|i| format!("elem{i}")).collect();

    for paged in [false, true] {
        let mut builder = root_with_vlen(12, &expected);
        builder
            .create_dataset("strings")
            .with_vlen_strings(&elements.iter().map(String::as_str).collect::<Vec<_>>())
            .with_chunks(&[3]);
        if paged {
            builder
                .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
                .with_file_space_page_size(4096);
        }

        let bytes = builder.finish().unwrap();
        assert!(has_fractal_heap(&bytes), "paged={paged}");
        let file = File::from_bytes(bytes).unwrap();
        assert_eq!(
            labels(&file.root().attrs().unwrap(), "labels"),
            expected,
            "paged={paged}"
        );
        assert_eq!(
            file.dataset("strings").unwrap().read_string().unwrap(),
            elements,
            "paged={paged}"
        );
    }
}

/// A dataset whose *elements* are variable-length, alongside dense attributes on
/// the same file. Element and attribute collections share one address cursor, so
/// a heap built at the wrong point would take addresses meant for the elements.
#[test]
fn dataset_elements_and_dense_attributes_share_the_heap_correctly() {
    let expected = ["attr-a", "attr-b"];
    let elements: Vec<String> = (0..5).map(|i| format!("elem{i}")).collect();

    let mut builder = root_with_vlen(12, &expected);
    builder
        .create_dataset("strings")
        .with_vlen_strings(&elements.iter().map(String::as_str).collect::<Vec<_>>());

    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(labels(&file.root().attrs().unwrap(), "labels"), expected);
    assert_eq!(
        file.dataset("strings").unwrap().read_string().unwrap(),
        elements
    );
}
