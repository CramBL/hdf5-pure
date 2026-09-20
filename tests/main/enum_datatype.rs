//! Round-trip coverage for HDF5 enumeration (H5T class 8) datasets through the
//! public write/read API.
//!
//! This is the hdf5-pure analog of the `hdf5-rs` enum tests: an enum is stored
//! as values of its integer base type, so a written enum dataset must read back
//! through the typed integer readers (via the base type) while the datatype
//! itself preserves the member name/value pairs. See issue #129.

use hdf5_pure::{
    DType, Datatype, EnumTypeBuilder, File, FileBuilder, FixedPointLayout, FormatError,
};

/// Build a file holding a single dataset `name`, serialize it, and reopen it.
fn write_then_open(build: impl FnOnce(&mut FileBuilder)) -> File {
    let mut builder = FileBuilder::new();
    build(&mut builder);
    let bytes = builder.finish().expect("finish");
    File::from_bytes(bytes).expect("from_bytes")
}

#[test]
fn i32_enum_dataset_preserves_members_and_values() {
    let dt = EnumTypeBuilder::i32_based()
        .value("RED", 0)
        .value("GREEN", 1)
        .value("BLUE", 2)
        .build()
        .unwrap();
    let file = write_then_open(|b| {
        b.create_dataset("color")
            .with_enum_i32_data(dt, &[0, 1, 2, 1]);
    });
    let ds = file.dataset("color").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);

    // The datatype round-trips as an Enumeration carrying its i32 base and the
    // full (name, value) member list.
    match ds.datatype().unwrap() {
        Datatype::Enumeration {
            base_type, members, ..
        } => {
            assert!(
                matches!(
                    *base_type,
                    Datatype::FixedPoint {
                        size: 4,
                        layout: FixedPointLayout { signed: true, .. },
                        ..
                    }
                ),
                "expected i32 base, got {base_type:?}"
            );
            let pairs: Vec<(&str, &[u8])> = members
                .iter()
                .map(|m| (m.name.as_str(), m.value.as_slice()))
                .collect();
            assert_eq!(
                pairs,
                vec![
                    ("RED", &0i32.to_le_bytes()[..]),
                    ("GREEN", &1i32.to_le_bytes()[..]),
                    ("BLUE", &2i32.to_le_bytes()[..]),
                ]
            );
        }
        other => panic!("expected Enumeration, got {other:?}"),
    }
}

#[test]
fn i32_enum_dataset_reads_back_through_base_type() {
    // Regression for the read asymmetry (issue #129): before the fix, read_i32
    // on an enum dataset failed `ensure_numeric` with TypeMismatch. An enum is
    // now decoded through its integer base, so signed, wider, and float reads
    // all resolve to the stored codes.
    let dt = EnumTypeBuilder::i32_based()
        .value("RED", 0)
        .value("GREEN", 1)
        .value("BLUE", 2)
        .build()
        .unwrap();
    let file = write_then_open(|b| {
        b.create_dataset("color")
            .with_enum_i32_data(dt, &[0, 1, 2, 1]);
    });
    let ds = file.dataset("color").unwrap();

    assert_eq!(ds.read_i32().unwrap(), vec![0, 1, 2, 1]);
    assert_eq!(ds.read_i64().unwrap(), vec![0, 1, 2, 1]);
    assert_eq!(ds.read_u32().unwrap(), vec![0, 1, 2, 1]);
    assert_eq!(ds.read_f64().unwrap(), vec![0.0, 1.0, 2.0, 1.0]);
}

#[test]
fn enum_dataset_classifies_as_dtype_enum_of_member_names() {
    let dt = EnumTypeBuilder::i32_based()
        .value("RED", 0)
        .value("GREEN", 1)
        .value("BLUE", 2)
        .build()
        .unwrap();
    let file = write_then_open(|b| {
        b.create_dataset("color").with_enum_i32_data(dt, &[0, 1, 2]);
    });
    let ds = file.dataset("color").unwrap();

    // The simplified DType surfaces the member names (values are dropped here;
    // recover them via `datatype()` when needed).
    assert_eq!(
        ds.dtype().unwrap(),
        DType::Enum(vec!["RED".into(), "GREEN".into(), "BLUE".into()])
    );
    assert_eq!(format!("{}", ds.dtype().unwrap()), "enum[RED, GREEN, BLUE]");
}

#[test]
fn u8_enum_dataset_roundtrips() {
    let dt = EnumTypeBuilder::u8_based()
        .u8_value("OFF", 0)
        .u8_value("ON", 1)
        .build()
        .unwrap();
    let file = write_then_open(|b| {
        b.create_dataset("switch")
            .with_enum_u8_data(dt, &[0, 1, 1, 0]);
    });
    let ds = file.dataset("switch").unwrap();

    assert_eq!(
        ds.dtype().unwrap(),
        DType::Enum(vec!["OFF".into(), "ON".into()])
    );
    // One byte per element on disk.
    assert_eq!(ds.read_raw().unwrap().len(), 4);
    // read_u8 already worked (it returns raw bytes); read_i32 works via the base.
    assert_eq!(ds.read_u8().unwrap(), vec![0, 1, 1, 0]);
    assert_eq!(ds.read_i32().unwrap(), vec![0, 1, 1, 0]);
}

#[test]
fn u8_enum_reads_identically_to_a_plain_u8_dataset() {
    // "An enum is its base type for value reads": a u8-based enum must decode
    // exactly as a plain u8 dataset. The unsigned readers recover 255; read_i32
    // reinterprets the 0xFF byte as signed (-1), identically in both cases — the
    // enum unwrap changes nothing about the per-reader integer semantics.
    let dt = EnumTypeBuilder::u8_based()
        .u8_value("LO", 0)
        .u8_value("HI", 255)
        .build()
        .unwrap();
    let file = write_then_open(|b| {
        b.create_dataset("level").with_enum_u8_data(dt, &[0, 255]);
        b.create_dataset("plain").with_u8_data(&[0, 255]);
    });
    let level = file.dataset("level").unwrap();
    let plain = file.dataset("plain").unwrap();

    assert_eq!(level.read_u8().unwrap(), vec![0, 255]);
    assert_eq!(level.read_u32().unwrap(), vec![0, 255]);
    assert_eq!(level.read_i32().unwrap(), plain.read_i32().unwrap());
    assert_eq!(level.read_u32().unwrap(), plain.read_u32().unwrap());
}

#[test]
fn enum_i32_data_into_u8_enum_type_is_rejected() {
    // `with_enum_i32_data` writes 4 bytes per value, but a u8-based enum declares
    // 1 byte per element. The writer's shape/data-length invariant catches the
    // mismatch rather than producing an unreadable file.
    let u8_enum = EnumTypeBuilder::u8_based()
        .u8_value("OFF", 0)
        .u8_value("ON", 1)
        .build()
        .unwrap();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("bad")
        .with_enum_i32_data(u8_enum, &[0, 1]);
    match builder.finish() {
        Err(hdf5_pure::Error::Format(FormatError::ShapeDataMismatch {
            expected,
            actual,
            element_size,
        })) => {
            assert_eq!(element_size.get(), 1);
            assert_eq!(expected, 2); // 2 elements * 1 byte
            assert_eq!(actual, 8); // 2 values * 4 bytes each
        }
        other => panic!("expected ShapeDataMismatch, got {other:?}"),
    }
}

#[test]
fn non_enum_reads_are_unaffected_by_the_enum_unwrap() {
    // The enum-unwrap on the read path must be a no-op for ordinary datatypes,
    // and a genuinely non-numeric datatype must still be rejected.
    let file = write_then_open(|b| {
        b.create_dataset("ints").with_i32_data(&[10, 20, 30]);
        b.create_dataset("text").with_vlen_strings(&["a", "bc"]);
    });
    assert_eq!(
        file.dataset("ints").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    let err = file.dataset("text").unwrap().read_i32().unwrap_err();
    assert!(
        matches!(
            err,
            hdf5_pure::Error::Format(FormatError::TypeMismatch { .. })
        ),
        "reading a vlen-string dataset as i32 should still be a TypeMismatch, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Enumerations over an arbitrary integer base type (issue #208)
// ---------------------------------------------------------------------------

/// `with_base` reaches base types the `i32`/`u8` constructors cannot, and
/// `value` encodes into the base type's width.
#[test]
fn enum_over_u16_base_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u16_enum.h5");

    let dt = EnumTypeBuilder::with_base(hdf5_pure::make_u16_type())
        .value("low", 1)
        .value("high", 40_000)
        .build()
        .unwrap();

    match &dt {
        Datatype::Enumeration {
            size,
            base_type,
            members,
        } => {
            assert_eq!(*size, 2, "element size comes from the base type");
            assert!(matches!(**base_type, Datatype::FixedPoint { size: 2, .. }));
            assert_eq!(members[0].value, 1u16.to_le_bytes().to_vec());
            assert_eq!(members[1].value, 40_000u16.to_le_bytes().to_vec());
        }
        other => panic!("expected an Enumeration, got {other:?}"),
    }

    let mut b = FileBuilder::new();
    b.create_dataset("e")
        .with_raw_data(dt, vec![1u8, 0, 0x40, 0x9C], 2)
        .with_shape(&[2]);
    b.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    match file.dataset("e").unwrap().datatype().unwrap() {
        Datatype::Enumeration { size, members, .. } => {
            assert_eq!(size, 2);
            assert_eq!(members.len(), 2);
            assert_eq!(members[1].name, "high");
            assert_eq!(members[1].value, 40_000u16.to_le_bytes().to_vec());
        }
        other => panic!("expected an Enumeration, got {other:?}"),
    }
}

/// `raw_value` carries stored bytes verbatim, for values an `i64` cannot express.
#[test]
fn raw_value_carries_bytes_verbatim() {
    let dt = EnumTypeBuilder::with_base(hdf5_pure::make_u64_type())
        .raw_value("max", &u64::MAX.to_le_bytes())
        .build()
        .unwrap();
    match &dt {
        Datatype::Enumeration { members, .. } => {
            assert_eq!(members[0].value, u64::MAX.to_le_bytes().to_vec());
        }
        other => panic!("expected an Enumeration, got {other:?}"),
    }
}

#[test]
fn build_refuses_a_value_too_wide_for_the_base() {
    let err = EnumTypeBuilder::with_base(hdf5_pure::make_u8_type())
        .value("too_big", 300)
        .build()
        .expect_err("300 does not fit one unsigned byte");
    assert!(
        matches!(err, FormatError::EnumMemberValueRange(ref n, 300, 1) if n == "too_big"),
        "got {err:?}"
    );
}

#[test]
fn build_refuses_a_negative_value_for_an_unsigned_base() {
    let err = EnumTypeBuilder::with_base(hdf5_pure::make_u16_type())
        .value("negative", -1)
        .build()
        .expect_err("an unsigned base cannot hold -1");
    assert!(
        matches!(err, FormatError::EnumMemberValueRange(..)),
        "got {err:?}"
    );
}

#[test]
fn build_refuses_raw_bytes_of_the_wrong_width() {
    let err = EnumTypeBuilder::with_base(hdf5_pure::make_u16_type())
        .raw_value("three_bytes", &[1, 2, 3])
        .build()
        .expect_err("three bytes cannot back a two-byte base");
    assert!(
        matches!(err, FormatError::EnumMemberValueSize(ref n, 2, 3) if n == "three_bytes"),
        "got {err:?}"
    );
}

#[test]
fn build_refuses_a_non_integer_base() {
    let err = EnumTypeBuilder::with_base(hdf5_pure::make_f64_type())
        .value("x", 1)
        .build()
        .expect_err("an enumeration base must be an integer");
    assert!(
        matches!(err, FormatError::EnumBaseNotInteger),
        "got {err:?}"
    );
}

/// A signed base keeps negative values, and the i32 path is unchanged.
#[test]
fn signed_bases_keep_negative_values() {
    let dt = EnumTypeBuilder::with_base(hdf5_pure::make_i16_type())
        .value("neg", -2)
        .build()
        .unwrap();
    match &dt {
        Datatype::Enumeration { members, .. } => {
            assert_eq!(members[0].value, (-2i16).to_le_bytes().to_vec());
        }
        other => panic!("expected an Enumeration, got {other:?}"),
    }
}
