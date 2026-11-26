use alloc::boxed::Box;
use alloc::vec;

use hdf5_pure_core::{
    CompoundMember, Datatype, DatatypeByteOrder, DatatypeMemberName, EnumMember, FixedPointType,
    FormatError, ReferenceType, VariableLengthType,
};
use test_util::datatype as fixture;
use test_util::datatype::containers::{self, EnumValue};

use super::super::header;
use super::super::message::version::{DATATYPE_VERSION_ONE, DATATYPE_VERSION_THREE};
use super::super::references;
use super::super::semantic;

#[test]
fn class_predicate_admits_reference_holding_datatypes() {
    let object_ref = || Datatype::Reference {
        size: 8,
        ref_type: ReferenceType::Object,
    };
    let i32_le = || {
        Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )
    };
    let holds_an_address = [
        ("a bare object reference", object_ref()),
        (
            "a dataset-region reference",
            Datatype::Reference {
                size: 12,
                ref_type: ReferenceType::DatasetRegion,
            },
        ),
        (
            "a compound holding one",
            Datatype::Compound {
                size: 12,
                members: vec![
                    CompoundMember::new(DatatypeMemberName::from(c"r"), 0, object_ref()),
                    CompoundMember::new(DatatypeMemberName::from(c"i"), 8, i32_le()),
                ],
            },
        ),
        (
            "an array of them",
            Datatype::Array {
                base_type: Box::new(object_ref()),
                dimensions: vec![2],
            },
        ),
        (
            "a variable length of them",
            Datatype::VariableLength(VariableLengthType::sequence(object_ref())),
        ),
        (
            "one nested two deep",
            Datatype::Array {
                base_type: Box::new(Datatype::Compound {
                    size: 8,
                    members: vec![CompoundMember::new(
                        DatatypeMemberName::from(c"r"),
                        0,
                        object_ref(),
                    )],
                }),
                dimensions: vec![3],
            },
        ),
    ];
    for (what, dt) in holds_an_address {
        assert!(
            references::datatype_holds_object_address(&dt),
            "{what} holds an object address"
        );
        let encoded = semantic::serialize_datatype(&dt).unwrap();
        assert!(
            references::class_may_hold_object_address(encoded[0]),
            "{what} encodes as class {}, which the gate rejects — a walk would \
             never parse it and would pass over the address inside it",
            encoded[0] & 0x0F
        );
    }
}

#[test]
fn object_address_checks_recognize_enumerations_with_reference_bases() {
    let parent = fixture::object_reference(8);
    let value = 0x1234u64.to_le_bytes();
    let bytes = containers::enumeration(
        DATATYPE_VERSION_THREE,
        [1, 0, 0],
        8,
        &parent,
        &[EnumValue {
            name: b"object",
            value: &value,
        }],
    );
    let (datatype, consumed) = semantic::parse_datatype(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    let Datatype::Enumeration(enumeration) = &datatype else {
        panic!("expected an enumeration, got {datatype:?}");
    };
    assert_eq!(
        enumeration.base_type(),
        &Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        }
    );
    assert_eq!(
        enumeration.members(),
        [EnumMember::new(
            DatatypeMemberName::new("object").unwrap(),
            value.to_vec()
        )]
    );
    assert!(references::datatype_holds_object_address(&datatype));
    assert!(references::class_may_hold_object_address(bytes[0]));
}

#[test]
fn class_predicate_admits_compounds_with_integer_members() {
    let ints = Datatype::Compound {
        size: 8,
        members: vec![CompoundMember::new(
            DatatypeMemberName::from(c"a"),
            0,
            Datatype::FixedPoint(
                FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap(),
            ),
        )],
    };
    assert!(!references::datatype_holds_object_address(&ints));
    assert!(
        references::class_may_hold_object_address(semantic::serialize_datatype(&ints).unwrap()[0]),
        "a compound of integers is admitted by the class gate and holds no address"
    );
}

#[test]
fn test_reference_object() {
    let buf = header::build_datatype_header(7, DATATYPE_VERSION_ONE, [0, 0, 0], 8);
    let (dt, _) = semantic::parse_datatype(&buf).unwrap();
    assert_eq!(
        dt,
        Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        }
    );
}

#[test]
fn test_reference_region() {
    let buf = header::build_datatype_header(7, DATATYPE_VERSION_ONE, [1, 0, 0], 12);
    let (dt, _) = semantic::parse_datatype(&buf).unwrap();
    assert_eq!(
        dt,
        Datatype::Reference {
            size: 12,
            ref_type: ReferenceType::DatasetRegion,
        }
    );
}

#[test]
fn test_error_invalid_reference_type() {
    let buf = header::build_datatype_header(7, DATATYPE_VERSION_ONE, [5, 0, 0], 8);
    let err = semantic::parse_datatype(&buf).unwrap_err();
    assert_eq!(err, FormatError::InvalidReferenceType(5));
}
