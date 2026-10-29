use super::super::*;
use crate::datatype::CharacterSet;
use crate::datatype::ReferenceType;
use crate::datatype::StringPadding;

use crate::{DatatypeByteOrder, File, FileBuilder, RepackOptions};

use super::super::test_util as repack_test_util;
use super::super::test_util::ExoticAttributes;

fn i32_type() -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    )
}

// Compare complete messages across the root, subgroup, and dataset ownership paths.
#[test]
fn a_position_independent_attribute_crosses_a_repack_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let (src, dst) = (dir.path().join("src.h5"), dir.path().join("dst.h5"));

    let attrs = || {
        [
            // Preserve integer widths in the complete messages.
            ("i32", AttrValue::I32(-7)),
            ("u32", AttrValue::U32(4_294_967_295)),
            // Charset and padding: an ASCII string and a UTF-8 one differ only
            // in the datatype's charset bit.
            ("ascii", AttrValue::AsciiString("m/s".into())),
            ("utf8", AttrValue::String("µm".into())),
            // A scalar and a one-element array are different dataspaces.
            ("one_elem", AttrValue::I64Array(vec![9])),
            ("scalar", AttrValue::I64(9)),
            ("f64s", AttrValue::F64Array(vec![1.5, -2.5])),
        ]
    };

    let mut b = FileBuilder::new();
    for (name, value) in attrs() {
        b.set_attr(name, value);
    }
    let ds = b.create_dataset("data").with_f64_data(&[1.0, 2.0]);
    for (name, value) in attrs() {
        ds.set_attr(name, value);
    }
    let mut g = b.create_group("grp");
    for (name, value) in attrs() {
        g.set_attr(name, value);
    }
    g.create_dataset("inner").with_i32_data(&[1]);
    b.add_group(g.finish());
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    for owner in ["", "data", "grp"] {
        let before = repack_test_util::messages_at(&src, owner);
        let after = repack_test_util::messages_at(&dst, owner);
        assert_eq!(
            before.keys().collect::<Vec<_>>(),
            after.keys().collect::<Vec<_>>(),
            "repack changed which attributes {owner:?} has"
        );
        for (name, source_message) in &before {
            assert_eq!(
                after.get(name),
                Some(source_message),
                "attribute {name:?} on {owner:?} was re-encoded rather than copied"
            );
        }
    }
}

// Verbatim messages cover string padding, null dataspaces, byte order, and rank.
#[test]
fn an_encoding_this_crate_has_no_attr_value_for_still_crosses_a_repack() {
    let dir = tempfile::tempdir().unwrap();
    let (src, dst) = (dir.path().join("src.h5"), dir.path().join("dst.h5"));

    let exotic = ExoticAttributes::new();
    exotic.write(&src);
    repack(&src, &dst, &RepackOptions::new()).unwrap();
    exotic.assert_repacked(&src, &dst);
}

#[test]
fn an_attribute_addressing_the_heap_is_re_encoded_not_copied() {
    let dir = tempfile::tempdir().unwrap();
    let (src, dst) = (dir.path().join("src.h5"), dir.path().join("dst.h5"));
    let fields: Vec<String> = vec!["x".into(), "y".into(), "velocity".into()];

    let mut b = FileBuilder::new();
    // Bulk that only the source carries, so the destination's heap cannot
    // land at the source's address by coincidence.
    b.create_dataset("bulk").with_f64_data(&vec![0.0; 4096]);
    b.set_attr("fields", AttrValue::VarLenAsciiCharArray(fields.clone()));
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new().drop_path("bulk")).unwrap();

    let before = &repack_test_util::messages_at(&src, "")["fields"];
    let after = &repack_test_util::messages_at(&dst, "")["fields"];
    assert_eq!(
        after.datatype, before.datatype,
        "the attribute must stay variable-length"
    );
    assert_eq!(after.dataspace, before.dataspace);
    assert_ne!(
        after.raw_data, before.raw_data,
        "a heap reference copied verbatim would point into the source file"
    );
    assert_eq!(
        File::open(&dst).unwrap().root().attrs().unwrap()["fields"],
        AttrValue::VarLenAsciiCharArray(fields),
        "and it must still resolve to its own strings"
    );
}

// Nested heap references and object addresses require position-dependent datatype classification.
#[test]
fn a_nested_address_makes_the_whole_datatype_position_dependent() {
    let vlen = Datatype::VariableLength(VariableLengthType::String(
        VariableLengthStringType::with_base_type(
            StringPadding::NullTerminate,
            CharacterSet::Ascii,
            i32_type(),
        )
        .unwrap(),
    ));
    let reference = Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let compound_of = |member: Datatype| Datatype::Compound {
        size: 24,
        members: vec![
            crate::datatype::__private::compound_member("plain".into(), 0, i32_type()),
            crate::datatype::__private::compound_member("nested".into(), 8, member),
        ],
    };
    let array_of = |base: Datatype| Datatype::Array {
        base_type: Box::new(base),
        dimensions: vec![2, 3],
    };

    for dependent in [
        vlen.clone(),
        reference.clone(),
        compound_of(vlen.clone()),
        compound_of(reference.clone()),
        array_of(vlen.clone()),
        array_of(reference.clone()),
        // Two levels down, which a non-recursive check would let through.
        array_of(compound_of(vlen)),
        compound_of(array_of(reference)),
    ] {
        assert!(
            !attr_bytes_are_position_independent(&dependent),
            "{dependent:?} holds an address and must not be copied verbatim"
        );
    }

    for independent in [
        i32_type(),
        Datatype::String {
            size: 8,
            padding: crate::datatype::StringPadding::NullPad,
            charset: CharacterSet::Utf8,
        },
        compound_of(i32_type()),
        array_of(i32_type()),
        Datatype::Enumeration {
            size: 4,
            base_type: Box::new(i32_type()),
            members: vec![],
        },
    ] {
        assert!(
            attr_bytes_are_position_independent(&independent),
            "{independent:?} holds no address and can be copied verbatim"
        );
    }
}
