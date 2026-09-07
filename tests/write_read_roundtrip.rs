use hdf5_pure::{
    AttrValue, CompoundTypeBuilder, DType, Datatype, Error, File, FileBuilder, FormatError,
    make_f64_type, make_i32_type,
};

#[test]
fn roundtrip_f64_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[4]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn roundtrip_i32_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_i32_data(&[10, 20, 30]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("values").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I32);
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

#[test]
fn roundtrip_u8_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("bytes")
        .with_u8_data(&[0xFF, 0x00, 0xAB]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("bytes").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U8);
}

#[test]
fn roundtrip_with_attributes() {
    let mut builder = FileBuilder::new();
    builder.set_attr("version", AttrValue::I64(2));
    builder
        .create_dataset("data")
        .with_f64_data(&[42.0])
        .set_attr("unit", AttrValue::String("m/s".into()));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();

    let root_attrs = file.root().attrs().unwrap();
    assert_eq!(root_attrs.get("version"), Some(&AttrValue::I64(2)));

    let ds = file.dataset("data").unwrap();
    let ds_attrs = ds.attrs().unwrap();
    assert_eq!(ds_attrs.get("unit"), Some(&AttrValue::String("m/s".into())));
}

#[test]
fn roundtrip_group_with_dataset() {
    let mut builder = FileBuilder::new();
    let mut grp = builder.create_group("sensors");
    grp.create_dataset("temperature")
        .with_f64_data(&[20.5, 21.0, 22.3]);
    grp.set_attr("location", AttrValue::String("lab".into()));
    let finished = grp.finish();
    builder.add_group(finished);

    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let grp = file.group("sensors").unwrap();
    let datasets = grp.datasets().unwrap();
    assert_eq!(datasets, vec!["temperature"]);

    let ds = file.dataset("sensors/temperature").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![20.5, 21.0, 22.3]);

    let grp_attrs = grp.attrs().unwrap();
    assert_eq!(
        grp_attrs.get("location"),
        Some(&AttrValue::String("lab".into()))
    );
}

#[test]
fn roundtrip_multiple_datasets() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_f64_data(&[1.0, 2.0, 3.0]);
    builder.create_dataset("y").with_f64_data(&[4.0, 5.0, 6.0]);
    builder.create_dataset("z").with_i32_data(&[7, 8, 9]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(
        file.dataset("x").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        file.dataset("y").unwrap().read_f64().unwrap(),
        vec![4.0, 5.0, 6.0]
    );
    assert_eq!(
        file.dataset("z").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
}

#[test]
fn roundtrip_write_to_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");

    let mut builder = FileBuilder::new();
    builder.create_dataset("data").with_f64_data(&[1.0, 2.0]);
    builder.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("data").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn roundtrip_i8_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i8_data(&[-128, -1, 0, 1, 127]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I8);
    assert_eq!(ds.read_i8().unwrap(), vec![-128, -1, 0, 1, 127]);
}

#[test]
fn roundtrip_i16_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i16_data(&[-32768, -1, 0, 1, 32767]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::I16);
    assert_eq!(ds.read_i16().unwrap(), vec![-32768, -1, 0, 1, 32767]);
}

#[test]
fn roundtrip_u16_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u16_data(&[0, 1, 1000, 65535]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U16);
    assert_eq!(ds.read_u16().unwrap(), vec![0, 1, 1000, 65535]);
}

#[test]
fn roundtrip_u32_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u32_data(&[0, 1, 1_000_000, u32::MAX]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U32);
    assert_eq!(ds.read_u32().unwrap(), vec![0, 1, 1_000_000, u32::MAX]);
}

#[test]
fn roundtrip_u64_dataset() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u64_data(&[0, 1, u64::MAX]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.dtype().unwrap(), DType::U64);
    assert_eq!(ds.read_u64().unwrap(), vec![0, 1, u64::MAX]);
}

#[test]
fn roundtrip_compound_complex64() {
    let complex_type = CompoundTypeBuilder::new()
        .f64_field("real")
        .f64_field("imag")
        .build()
        .unwrap();

    let values: Vec<(f64, f64)> = vec![(1.0, 2.0), (3.0, 4.0)];
    let mut raw = Vec::new();
    for (r, i) in &values {
        raw.extend_from_slice(&r.to_le_bytes());
        raw.extend_from_slice(&i.to_le_bytes());
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("complex")
        .with_compound_data(complex_type, raw, 2);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("complex").unwrap();
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "real");
            assert_eq!(fields[1].0, "imag");
        }
        other => panic!("expected compound, got {other}"),
    }
}

#[test]
fn roundtrip_compound_tuple_field_wise() {
    let values = [(-5i8, 10u64, 1.25f32), (7i8, u64::MAX, -3.5f32)];

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("tuples")
        .with_compound_values(&values)
        .unwrap();
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("tuples").unwrap();
    assert_eq!(ds.read_compound::<(i8, u64, f32)>().unwrap(), values);
    assert_eq!(ds.read_raw().unwrap().len(), values.len() * 13);

    match ds.datatype().unwrap() {
        Datatype::Compound { size, members } => {
            assert_eq!(size, 13);
            assert_eq!(
                members
                    .iter()
                    .map(|member| (member.name.as_str(), member.byte_offset))
                    .collect::<Vec<_>>(),
                vec![("0", 0), ("1", 1), ("2", 9)]
            );
        }
        other => panic!("expected compound, got {other:?}"),
    }
}

#[test]
fn padded_compound_decodes_without_using_tuple_layout() {
    let datatype = CompoundTypeBuilder::with_size(24)
        .i8_field("0", 0)
        .u64_field("1", 8)
        .f32_field("2", 16)
        .build()
        .unwrap();

    let values = [(-3i8, 42u64, 1.5f32), (9, u64::MAX, -2.25)];
    let mut raw = vec![0u8; values.len() * 24];
    for (index, (a, b, c)) in values.iter().copied().enumerate() {
        let record = &mut raw[index * 24..(index + 1) * 24];
        record[0] = a as u8;
        record[8..16].copy_from_slice(&b.to_le_bytes());
        record[16..20].copy_from_slice(&c.to_le_bytes());
    }

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("padded")
        .with_compound_data(datatype, raw.clone(), values.len() as u64);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("padded").unwrap();
    assert_eq!(ds.read_raw().unwrap(), raw);
    assert_eq!(ds.read_compound::<(i8, u64, f32)>().unwrap(), values);
}

#[test]
fn nested_compound_tuple_roundtrip() {
    let values = [((-2i8, 99u64), 3.25f32), ((7, 11), -0.5)];

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("nested")
        .with_compound_values(&values)
        .unwrap();
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("nested").unwrap();
    assert_eq!(ds.read_compound::<((i8, u64), f32)>().unwrap(), values);
}

#[test]
fn chunked_builder_rejects_invalid_geometry() {
    // Each malformed chunk-geometry request must be refused with
    // `InvalidChunkGeometry` rather than panicking in the chunk splitter or
    // producing an unreadable dataset.
    type Configure = fn(&mut hdf5_pure::DatasetBuilder);
    let bad: &[(&str, Configure, &str)] = &[
        (
            "chunk rank mismatch",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4, 5, 6])
                    .with_shape(&[2, 3])
                    .with_chunks(&[2]);
            },
            "chunk dimensions must have the same rank",
        ),
        (
            "zero chunk dim",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4])
                    .with_shape(&[4])
                    .with_chunks(&[0]);
            },
            "chunk dimensions must all be non-zero",
        ),
        (
            "maxshape rank mismatch",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4])
                    .with_shape(&[4])
                    .with_maxshape(&[u64::MAX, u64::MAX])
                    .with_chunks(&[2]);
            },
            "maxshape must have the same rank",
        ),
        (
            "maxshape below shape",
            |b| {
                b.with_i32_data(&[1, 2, 3, 4])
                    .with_shape(&[4])
                    .with_maxshape(&[2]);
            },
            "maxshape must be at least the current shape",
        ),
        (
            "scalar with chunks",
            |b| {
                b.with_f64_data(&[1.0]).with_shape(&[]).with_chunks(&[1]);
            },
            "a scalar dataset cannot be chunked",
        ),
        // Auto-chunking makes the shape one chunk, so a zero-element shape
        // resolves to a zero chunk dimension — the "zero chunk dim" case above,
        // reached without the caller naming one. It divided by zero in the
        // splitter until the guard learned to check the *resolved* dimensions.
        (
            "auto-chunked empty shape",
            |b| {
                b.with_i32_data(&[])
                    .with_shape(&[0])
                    .with_maxshape(&[u64::MAX]);
            },
            "explicit chunk dimensions",
        ),
        (
            "auto-chunked empty inner dim",
            |b| {
                b.with_i32_data(&[])
                    .with_shape(&[4, 0])
                    .with_maxshape(&[u64::MAX, u64::MAX]);
            },
            "explicit chunk dimensions",
        ),
    ];

    for (label, configure, expected) in bad {
        let mut builder = FileBuilder::new();
        configure(builder.create_dataset("bad"));
        let err = builder.finish().unwrap_err();
        let Error::Format(FormatError::InvalidChunkGeometry(reason)) = &err else {
            panic!("[{label}] expected InvalidChunkGeometry, got {err:?}");
        };
        // Each case names its own reason: a wildcard payload passes on a refusal
        // from any *other* geometry guard, which makes the case under test
        // unreachable without anything going red.
        assert!(
            reason.contains(expected),
            "[{label}] refusal must name {expected:?}, got {reason:?}"
        );
    }
}

#[test]
fn chunked_builder_accepts_empty_extensible_dataset() {
    // A zero-element extensible dataset (shape `[0]`, unlimited max) is a valid,
    // common pattern (create empty, append later); the geometry guard must not
    // reject it.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("stream")
        .with_i32_data(&[])
        .with_shape(&[0])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[16]);
    let bytes = builder
        .finish()
        .expect("empty extensible dataset should build");

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("stream").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0]);
    assert_eq!(ds.read_i32().unwrap(), Vec::<i32>::new());
}

/// A fixed-length string type of zero width — the shape of the datatype message
/// that panicked the chunked reader in issue #268, here built by a caller rather
/// than read from a file.
fn zero_width_type() -> Datatype {
    Datatype::String {
        size: 0,
        padding: hdf5_pure::StringPadding::NullPad,
        charset: hdf5_pure::CharacterSet::Ascii,
    }
}

/// The packed builder validates what its explicit sibling always validated: a
/// compound of no fields, and one whose fields pack to no bytes, are the two
/// cases `H5Tcreate(H5T_COMPOUND, ..)` refuses. It used to return them, and
/// `CompoundTypeBuilder::new().build()` was the shortest route to a datatype
/// that panicked the writer.
#[test]
fn packed_compound_builder_rejects_a_type_with_no_bytes() {
    assert!(matches!(
        CompoundTypeBuilder::new().build().unwrap_err(),
        FormatError::EmptyCompoundType
    ));

    // Not empty, but every field is zero-width, so the type still packs to
    // nothing. This is the second guard, and only this case reaches it.
    assert!(matches!(
        CompoundTypeBuilder::new()
            .field("a", zero_width_type())
            .build()
            .unwrap_err(),
        FormatError::InvalidCompoundSize
    ));
}

/// Nothing occupies zero bytes per element, and the writers divide by the
/// element size exactly as the readers do — the chunked splitter clamps a row
/// copy to an element boundary with `% element_size`, which panicked outright.
/// A caller-built `Datatype` never passes through the parse-side refusal, so
/// every door into the writer refuses one itself (issue #268).
#[test]
fn a_zero_width_element_type_is_refused_by_every_write_path() {
    // Chunked: the path that panicked. The element bytes are empty on purpose —
    // four elements of zero width *is* zero bytes, so this satisfies the
    // shape-versus-data check and reaches the chunk splitter, which is where the
    // division by the element size lives. Supplying non-empty data instead would
    // be refused earlier as a shape mismatch and never exercise it.
    let mut chunked = FileBuilder::new();
    chunked
        .create_dataset("d")
        .with_raw_data(zero_width_type(), Vec::new(), 4)
        .with_chunks(&[2]);

    // Contiguous: no panic, but it wrote a file this crate's own reader refuses.
    let mut contiguous = FileBuilder::new();
    contiguous
        .create_dataset("d")
        .with_raw_data(zero_width_type(), Vec::new(), 4);

    // Committed: written as an object of its own, for datasets to name.
    let mut committed = FileBuilder::new();
    committed.commit_datatype("t", zero_width_type());
    committed.create_dataset("d").with_i32_data(&[1]);

    for (path, result) in [
        ("chunked", chunked.finish()),
        ("contiguous", contiguous.finish()),
        ("committed", committed.finish()),
    ] {
        match result {
            Err(Error::Format(FormatError::ZeroSizedDatatype { class: 3 })) => {}
            other => panic!(
                "{path}: expected ZeroSizedDatatype for class 3, got {:?}",
                other.map(|b| b.len())
            ),
        }
    }
}

#[test]
fn explicit_compound_builder_rejects_invalid_layouts() {
    let err = CompoundTypeBuilder::with_size(8)
        .u64_field("a", 0)
        .u32_field("b", 6)
        .build()
        .unwrap_err();
    assert!(matches!(
        err,
        FormatError::CompoundFieldOutOfBounds { ref name, .. } if name == "b"
    ));

    let err = CompoundTypeBuilder::with_size(16)
        .u64_field("a", 0)
        .u64_field("b", 4)
        .build()
        .unwrap_err();
    assert!(matches!(
        err,
        FormatError::CompoundFieldOverlap {
            ref first,
            ref second
        } if first == "a" && second == "b"
    ));

    let err = CompoundTypeBuilder::with_size(16)
        .u64_field("a", 0)
        .u64_field("a", 8)
        .build()
        .unwrap_err();
    assert_eq!(err, FormatError::DuplicateCompoundField("a".into()));
}

#[test]
fn nested_compound_type_mismatch_shows_hierarchical_path() {
    let inner_mismatched = CompoundTypeBuilder::with_size(16)
        .i8_field("0", 0)
        .f64_field("1", 8) // mismatch: expected u64, got f64
        .build()
        .unwrap();

    let outer = CompoundTypeBuilder::with_size(24)
        .field("0", 0, inner_mismatched)
        .f32_field("1", 16)
        .build()
        .unwrap();

    let raw = vec![0u8; 24];
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("nested_mismatched")
        .with_compound_data(outer, raw, 1);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("nested_mismatched").unwrap();

    let err = ds.read_compound::<((i8, u64), f32)>().unwrap_err();
    match err {
        hdf5_pure::Error::Format(FormatError::CompoundFieldTypeMismatch(ref path)) => {
            assert_eq!(path, "0.1");
        }
        other => panic!(
            "expected CompoundFieldTypeMismatch(\"0.1\"), got {:?}",
            other
        ),
    }
}

#[test]
fn roundtrip_userblock_512() {
    let mut builder = FileBuilder::new();
    builder.with_userblock(512);
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0]);
    let mut bytes = builder.finish().unwrap();

    // Verify userblock is at the start (512 zero bytes)
    assert!(bytes.len() > 512);
    assert!(bytes[..512].iter().all(|&b| b == 0));

    // HDF5 signature should start at offset 512
    assert_eq!(&bytes[512..520], b"\x89HDF\r\n\x1a\n");

    // Write some userblock data (e.g., MATLAB header)
    bytes[0..6].copy_from_slice(b"MATLAB");

    // File should still be readable
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn roundtrip_nested_groups() {
    let mut builder = FileBuilder::new();

    let mut outer = builder.create_group("outer");
    outer.create_dataset("outer_data").with_f64_data(&[1.0]);

    let mut inner = outer.create_group("inner");
    inner
        .create_dataset("inner_data")
        .with_f64_data(&[2.0, 3.0]);
    inner.set_attr("depth", AttrValue::I64(2));
    outer.add_group(inner.finish());

    builder.add_group(outer.finish());

    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();

    // Read outer dataset
    let ds = file.dataset("outer/outer_data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0]);

    // Read inner dataset
    let ds = file.dataset("outer/inner/inner_data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![2.0, 3.0]);

    // Read inner group attrs
    let inner_grp = file.group("outer/inner").unwrap();
    let attrs = inner_grp.attrs().unwrap();
    assert_eq!(attrs.get("depth"), Some(&AttrValue::I64(2)));

    // Verify group hierarchy
    let outer_grp = file.group("outer").unwrap();
    let sub_groups = outer_grp.groups().unwrap();
    assert_eq!(sub_groups, vec!["inner"]);
    let datasets = outer_grp.datasets().unwrap();
    assert_eq!(datasets, vec!["outer_data"]);
}

#[test]
fn roundtrip_i32_u32_attributes() {
    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_int_decode", AttrValue::I32(1));
    builder.set_attr("MATLAB_empty", AttrValue::U32(1));
    builder.create_dataset("x").with_f64_data(&[0.0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();

    // Each reads back at the width it was written at (#350), which is what
    // MATLAB sees: `MATLAB_int_decode` is a 4-byte signed attribute in the file.
    assert_eq!(attrs.get("MATLAB_int_decode"), Some(&AttrValue::I32(1)));
    assert_eq!(attrs.get("MATLAB_empty"), Some(&AttrValue::U32(1)));
}

#[test]
fn roundtrip_ascii_string_attribute() {
    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    builder.create_dataset("x").with_f64_data(&[0.0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let attrs = file.root().attrs().unwrap();
    // The charset is what this test is about, so it asserts the variant: an
    // ASCII string reads back ASCII rather than as a UTF-8 `String`.
    assert_eq!(
        attrs.get("MATLAB_class"),
        Some(&AttrValue::AsciiString("double".into()))
    );
}

#[test]
fn roundtrip_complex32_convenience() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("z")
        .with_complex32_data(&[(1.0f32, 2.0f32), (3.0, 4.0)]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("z").unwrap();
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0], ("real".to_string(), DType::F32));
            assert_eq!(fields[1], ("imag".to_string(), DType::F32));
        }
        other => panic!("expected compound, got {other}"),
    }
    assert_eq!(ds.shape().unwrap(), vec![2]);
}

#[test]
fn roundtrip_complex64_convenience() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("z")
        .with_complex64_data(&[(1.0, -1.0), (0.0, 3.14)]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("z").unwrap();
    match ds.dtype().unwrap() {
        DType::Compound(fields) => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0], ("real".to_string(), DType::F64));
            assert_eq!(fields[1], ("imag".to_string(), DType::F64));
        }
        other => panic!("expected compound, got {other}"),
    }
}

#[test]
fn roundtrip_path_references() {
    // Simulate the MATLAB #refs# pattern:
    // Create a #refs# group with child datasets,
    // then create a reference dataset pointing to those children.
    let mut builder = FileBuilder::new();

    // Create #refs# group with two child datasets
    let mut refs_grp = builder.create_group("#refs#");
    refs_grp
        .create_dataset("child_a")
        .with_f64_data(&[1.0, 2.0]);
    refs_grp
        .create_dataset("child_b")
        .with_i32_data(&[10, 20, 30]);
    builder.add_group(refs_grp.finish());

    // Create a reference dataset pointing to those children by path
    builder
        .create_dataset("cell_refs")
        .with_path_references(&["#refs#/child_a", "#refs#/child_b"]);

    let bytes = builder.finish().unwrap();

    // Verify the file is valid
    let file = File::from_bytes(bytes).unwrap();

    // The referenced datasets should be readable
    let a = file.dataset("#refs#/child_a").unwrap();
    assert_eq!(a.read_f64().unwrap(), vec![1.0, 2.0]);

    let b = file.dataset("#refs#/child_b").unwrap();
    assert_eq!(b.read_i32().unwrap(), vec![10, 20, 30]);

    // The reference dataset should exist and have the right shape
    let refs = file.dataset("cell_refs").unwrap();
    assert_eq!(refs.shape().unwrap(), vec![2]);

    // Verify the reference dataset has the correct type
    assert_eq!(refs.dtype().unwrap(), DType::ObjectReference);
}

#[test]
fn roundtrip_path_references_2d_shape() {
    // MATLAB pattern: reference datasets with 2D shapes like [1, n] or [2, 1]
    let mut builder = FileBuilder::new();

    let mut refs_grp = builder.create_group("#refs#");
    refs_grp.create_dataset("a").with_f64_data(&[1.0]);
    refs_grp.create_dataset("b").with_f64_data(&[2.0]);
    refs_grp.create_dataset("c").with_f64_data(&[3.0]);
    refs_grp.create_dataset("d").with_f64_data(&[4.0]);
    builder.add_group(refs_grp.finish());

    // Shape [1, 4] — row vector of references
    builder
        .create_dataset("row_refs")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c", "#refs#/d"])
        .with_shape(&[1, 4]);

    // Shape [2, 1] — column vector of references
    builder
        .create_dataset("col_refs")
        .with_path_references(&["#refs#/a", "#refs#/b"])
        .with_shape(&[2, 1]);

    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();

    let row = file.dataset("row_refs").unwrap();
    assert_eq!(row.shape().unwrap(), vec![1, 4]);
    assert_eq!(row.dtype().unwrap(), DType::ObjectReference);

    let col = file.dataset("col_refs").unwrap();
    assert_eq!(col.shape().unwrap(), vec![2, 1]);
    assert_eq!(col.dtype().unwrap(), DType::ObjectReference);
}

#[test]
fn roundtrip_varlen_ascii_array_attr() {
    let mut builder = FileBuilder::new();
    builder.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["x".into(), "y".into(), "z".into()]),
    );
    builder.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = builder.finish().unwrap();

    // Verify we can read the file back
    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("x").unwrap();
    assert_eq!(ds.read_f64().unwrap(), vec![1.0]);

    // Check that the MATLAB_fields attribute exists
    let attrs = file.root().attrs().unwrap();
    assert!(attrs.contains_key("MATLAB_fields"));
}

#[test]
fn valid_hdf5_signature() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = builder.finish().unwrap();

    assert!(bytes.len() > 8);
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
}

#[test]
fn roundtrip_empty_dataset_zero_dims() {
    let mut builder = FileBuilder::new();
    builder.set_attr("MATLAB_empty", AttrValue::U32(1));
    builder
        .create_dataset("empty")
        .with_dtype(make_f64_type())
        .with_shape(&[0, 0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("empty").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![0, 0]);
    assert_eq!(ds.dtype().unwrap(), DType::F64);
    // Empty dataset should return empty data
    assert_eq!(ds.read_f64().unwrap(), vec![]);
}

#[test]
fn roundtrip_matlab_empty_shape_marker() {
    // MATLAB pattern: store original shape as u64 data with MATLAB_empty=1
    let matlab_dims: &[usize] = &[3, 4];
    let shape_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u64_data(&shape_data)
        .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()))
        .set_attr("MATLAB_empty", AttrValue::U32(1));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2]); // 2-element 1D array
    assert_eq!(ds.dtype().unwrap(), DType::U64);
    assert_eq!(ds.read_u64().unwrap(), vec![3, 4]);
}

#[test]
fn roundtrip_matlab_refs_subsystem_pattern() {
    // Exercises the MATLAB v7.3 string/cell subsystem pattern:
    //   #refs# group   — holds the actual data as child datasets
    //   #subsystem#     — holds an MCOS dataset with path references into #refs#
    //   Some #refs# children themselves contain references to siblings

    let mut builder = FileBuilder::new();

    // --- #refs# group with several child datasets ---
    let mut refs_grp = builder.create_group("#refs#");

    // String-value datasets (simulating MATLAB string objects)
    refs_grp
        .create_dataset("a")
        .with_u16_data(&[72, 101, 108, 108, 111]); // "Hello" UTF-16
    refs_grp
        .create_dataset("b")
        .with_u16_data(&[87, 111, 114, 108, 100]); // "World"
    refs_grp.create_dataset("c").with_u16_data(&[70, 111, 111]); // "Foo"

    // A child dataset that itself references siblings (cross-references within #refs#)
    refs_grp
        .create_dataset("cell_data")
        .with_path_references(&["#refs#/a", "#refs#/b", "#refs#/c"])
        .with_shape(&[1, 3]);

    // Metadata dataset with attributes
    refs_grp
        .create_dataset("type_info")
        .with_u8_data(&[0, 0, 0, 0, 0, 0, 1, 0])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));

    builder.add_group(refs_grp.finish());

    // --- #subsystem# group with MCOS dataset ---
    let mut subsys_grp = builder.create_group("#subsystem#");

    // MCOS dataset: array of references to objects in #refs#, shape [2, 1]
    subsys_grp
        .create_dataset("MCOS")
        .with_path_references(&["#refs#/cell_data", "#refs#/type_info"])
        .with_shape(&[2, 1]);

    builder.add_group(subsys_grp.finish());

    // --- Root-level dataset referencing into #refs# ---
    builder
        .create_dataset("data")
        .with_path_references(&["#refs#/a"])
        .with_shape(&[1, 1])
        .set_attr("MATLAB_class", AttrValue::AsciiString("string".into()));

    let bytes = builder.finish().unwrap();

    // --- Verify the file structure ---
    let file = File::from_bytes(bytes).unwrap();

    // Root children
    let root = file.root();
    let groups = root.groups().unwrap();
    assert!(groups.contains(&"#refs#".to_string()));
    assert!(groups.contains(&"#subsystem#".to_string()));

    // #refs# children are readable
    let a = file.dataset("#refs#/a").unwrap();
    assert_eq!(a.read_u16().unwrap(), vec![72, 101, 108, 108, 111]);
    let b = file.dataset("#refs#/b").unwrap();
    assert_eq!(b.read_u16().unwrap(), vec![87, 111, 114, 108, 100]);
    let c = file.dataset("#refs#/c").unwrap();
    assert_eq!(c.read_u16().unwrap(), vec![70, 111, 111]);

    // Cross-reference dataset in #refs# has correct 2D shape
    let cell = file.dataset("#refs#/cell_data").unwrap();
    assert_eq!(cell.shape().unwrap(), vec![1, 3]);
    assert_eq!(cell.dtype().unwrap(), DType::ObjectReference);

    // #subsystem#/MCOS has correct 2D shape
    let mcos = file.dataset("#subsystem#/MCOS").unwrap();
    assert_eq!(mcos.shape().unwrap(), vec![2, 1]);
    assert_eq!(mcos.dtype().unwrap(), DType::ObjectReference);

    // Root data dataset
    let data = file.dataset("data").unwrap();
    assert_eq!(data.shape().unwrap(), vec![1, 1]);
    assert_eq!(data.dtype().unwrap(), DType::ObjectReference);
}

#[test]
fn roundtrip_varlen_ascii_on_nested_group() {
    // MATLAB pattern: MATLAB_fields on a struct group (not root)
    let mut builder = FileBuilder::new();

    let mut grp = builder.create_group("my_struct");
    grp.create_dataset("x").with_f64_data(&[1.0, 2.0]);
    grp.create_dataset("y").with_f64_data(&[3.0, 4.0]);
    grp.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));
    grp.set_attr(
        "MATLAB_fields",
        AttrValue::VarLenAsciiCharArray(vec!["x".into(), "y".into()]),
    );
    builder.add_group(grp.finish());

    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    // Verify group children
    let g = file.group("my_struct").unwrap();
    let datasets = g.datasets().unwrap();
    assert!(datasets.contains(&"x".to_string()));
    assert!(datasets.contains(&"y".to_string()));

    // Verify data
    assert_eq!(
        file.dataset("my_struct/x").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.dataset("my_struct/y").unwrap().read_f64().unwrap(),
        vec![3.0, 4.0]
    );
}

#[test]
fn roundtrip_group_only_no_datasets() {
    // MATLAB pattern: nested structs where a group has only sub-groups and attrs
    let mut builder = FileBuilder::new();

    let mut outer = builder.create_group("outer");
    outer.set_attr("MATLAB_class", AttrValue::AsciiString("struct".into()));

    let mut child_a = outer.create_group("a");
    child_a.create_dataset("val").with_f64_data(&[1.0]);
    child_a.set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    outer.add_group(child_a.finish());

    let mut child_b = outer.create_group("b");
    child_b.create_dataset("val").with_i32_data(&[42]);
    child_b.set_attr("MATLAB_class", AttrValue::AsciiString("int32".into()));
    outer.add_group(child_b.finish());

    // outer has zero datasets, only sub-groups + attrs
    builder.add_group(outer.finish());

    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();

    let outer_grp = file.group("outer").unwrap();
    let groups = outer_grp.groups().unwrap();
    assert!(groups.contains(&"a".to_string()));
    assert!(groups.contains(&"b".to_string()));
    assert_eq!(outer_grp.datasets().unwrap().len(), 0);

    assert_eq!(
        file.dataset("outer/a/val").unwrap().read_f64().unwrap(),
        vec![1.0]
    );
    assert_eq!(
        file.dataset("outer/b/val").unwrap().read_i32().unwrap(),
        vec![42]
    );
}

#[test]
fn roundtrip_paged_fixed_array_many_chunks() {
    // More than 1024 chunks forces the Fixed Array index into its paged data
    // block layout (page size = 2^10 = 1024). 2500 chunks of size 1 spans three
    // pages, with a partial final page.
    let n = 2500usize;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[1]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![n as u64]);
    assert_eq!(ds.read_f64().unwrap(), data);
}

#[test]
fn roundtrip_paged_fixed_array_deflate() {
    // Paged data block on the filtered (client_id = 1) path: each element record
    // carries an address plus a variable-width compressed size and filter mask.
    let n = 1500usize;
    let data: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[1])
        .with_deflate(4);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_f64().unwrap(), data);
}

// ---------------------------------------------------------------------------
// Scale-offset filter (id 6) round-trips: write with hdf5-pure, read back.
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_scale_offset_integer_lossless() {
    // Multi-chunk i32 dataset, including a partial last chunk, lossless.
    let n = 250usize;
    let data: Vec<i32> = (0..n).map(|i| 1000 + (i as i32 % 37)).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[40]) // 250 / 40 -> 7 chunks, last partial
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), data);
}

#[test]
fn roundtrip_scale_offset_integer_negative_and_constant() {
    // Negative values plus an all-equal chunk (exercises minbits == 0).
    let mut data: Vec<i32> = (0..50).map(|i| -100 + i).collect();
    data.extend(std::iter::repeat_n(7, 50)); // constant chunk

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&data)
        .with_shape(&[data.len() as u64])
        .with_chunks(&[50])
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), data);
}

#[test]
fn roundtrip_scale_offset_u16_lossless() {
    let data: Vec<u16> = (0..100).map(|i| 40000 + (i as u16 % 13)).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u16_data(&data)
        .with_shape(&[data.len() as u64])
        .with_chunks(&[25])
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_u16().unwrap(), data);
}

#[test]
fn roundtrip_scale_offset_float_dscale_lossy() {
    let decimals = 3i32;
    let data: Vec<f64> = (0..200).map(|i| 1.0 + (i as f64) * 0.001).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[data.len() as u64])
        .with_chunks(&[64])
        .with_scale_offset(hdf5_pure::ScaleOffset::FloatDScale(decimals));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    let got = ds.read_f64().unwrap();
    let tol = 0.5 * 10f64.powi(-decimals);
    assert_eq!(got.len(), data.len());
    for (g, w) in got.iter().zip(data.iter()) {
        assert!((g - w).abs() <= tol, "got {g}, want {w}");
    }
}

#[test]
fn roundtrip_scale_offset_integer_then_deflate() {
    // Scale-offset followed by deflate: the pipeline applies [scaleoffset, deflate]
    // on write and reverses on read.
    let data: Vec<i32> = (0..300).map(|i| 5000 + (i % 19)).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&data)
        .with_shape(&[data.len() as u64])
        .with_chunks(&[64])
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0))
        .with_deflate(5);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), data);
}

#[test]
fn roundtrip_scale_offset_incompressible_then_deflate() {
    // Regression for #97 (PR review): a full i32 dynamic range forces
    // scale-offset's raw fallback, whose output is HEADER_LEN bytes LARGER than
    // the chunk. On read deflate is reversed first and must legitimately produce
    // that larger buffer, so the deflate output cap must account for the inner
    // scale-offset expansion instead of rejecting it as a decompression bomb.
    let n = 256usize;
    let data: Vec<i32> = (0..n)
        .map(|i| match i % 4 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => -(i as i32).wrapping_mul(7919),
            _ => (i as i32).wrapping_mul(104_729),
        })
        .collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[64])
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0))
        .with_deflate(5);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.read_i32().unwrap(), data);
}

#[test]
fn shape_data_mismatch_is_rejected() {
    use hdf5_pure::{Error, FormatError};

    // 3 elements supplied but a 2x2 (=4 element) shape requested.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("bad")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .with_shape(&[2, 2]);
    let err = builder.finish().unwrap_err();
    match err {
        Error::Format(FormatError::ShapeDataMismatch {
            expected,
            actual,
            element_size,
        }) => {
            assert_eq!(expected, 4 * 8); // shape needs 4 f64 = 32 bytes
            assert_eq!(actual, 3 * 8); // only 3 f64 = 24 bytes supplied
            assert_eq!(element_size.get(), 8); // f64
        }
        other => panic!("expected ShapeDataMismatch, got {other:?}"),
    }
}

/// A zero-element shape is held to the same shape/data agreement every other
/// shape is (issue #332), rather than exempted from it.
///
/// The exemption wrote the one combination that cannot be written correctly.
/// Contiguously it put the staged bytes at a defined address under a dataspace
/// declaring no elements, which the reference library calls corruption and
/// refuses to open, and which this crate's own reader answers `Ok([])` for
/// without ever looking at the region. Chunked it produced a grid of no chunks,
/// so the bytes went nowhere and the caller was told `Ok`.
///
/// Swept over every way to reach the shape rather than over one picked case:
/// the guard is a single condition, and each arm has to land on the same side
/// of it. The reference library's verdict on the file the exemption wrote is
/// recorded in the issue rather than kept as a fixture here — the writer can no
/// longer produce it, which is the point.
#[test]
fn zero_element_shape_with_data_is_rejected() {
    for shape in [&[0u64][..], &[0, 3][..], &[3, 0][..]] {
        for chunked in [false, true] {
            let mut builder = FileBuilder::new();
            {
                let db = builder
                    .create_dataset("bad")
                    .with_i32_data(&[1, 2, 3])
                    .with_shape(shape);
                if chunked {
                    db.with_chunks(&[2, 2][..shape.len()]);
                }
            }
            let case = format!("shape {shape:?}, chunked {chunked}");
            match builder.finish() {
                Err(Error::Format(FormatError::ShapeDataMismatch {
                    expected,
                    actual,
                    element_size,
                })) => {
                    assert_eq!(expected, 0, "{case}: the shape holds no elements");
                    assert_eq!(actual, 3 * 4, "{case}: three i32 were supplied");
                    assert_eq!(element_size.get(), 4, "{case}");
                }
                other => panic!("{case}: expected ShapeDataMismatch, got {other:?}"),
            }
        }
    }
}

/// Staging no data for a zero-element shape stays legal, which is the half of
/// the guard the refusal above must not take with it: both the extent and the
/// staged region are zero bytes there, so they agree.
///
/// Swept over how the emptiness is declared as well as over the shape. The two
/// are not the same case: staging an empty slice satisfies the writer's
/// "a dataset has data" requirement outright, while omitting the data reaches it
/// through the zero-element exemption, and only the second reads the shape to
/// decide. A sweep that fixed this axis at the empty slice passed unchanged
/// with the exemption narrowed to the *first* dimension, which would have made
/// a `[3, 0]` dataset with no data fail as one missing its data.
#[test]
fn zero_element_shape_without_data_is_accepted() {
    for shape in [&[0u64][..], &[0, 3][..], &[3, 0][..]] {
        for chunked in [false, true] {
            for staged_empty_slice in [false, true] {
                let mut builder = FileBuilder::new();
                {
                    let db = builder.create_dataset("empty");
                    if staged_empty_slice {
                        db.with_i32_data(&[]);
                    } else {
                        db.with_dtype(make_i32_type());
                    }
                    db.with_shape(shape);
                    if chunked {
                        db.with_chunks(&[2, 2][..shape.len()]);
                    }
                }
                let case =
                    format!("shape {shape:?}, chunked {chunked}, slice {staged_empty_slice}");
                let bytes = builder.finish().unwrap_or_else(|e| panic!("{case}: {e}"));
                let file = File::from_bytes(bytes).unwrap();
                let ds = file.dataset("empty").unwrap();
                assert_eq!(ds.shape().unwrap(), shape.to_vec(), "{case}");
                assert_eq!(ds.read_i32().unwrap(), Vec::<i32>::new(), "{case}");
                if !chunked {
                    // The contrast with the refused case, stated as a layout: an
                    // empty dataset names no data region at all. The exempted write
                    // named one of twelve bytes at a defined address, which is the
                    // thing the reference library refuses to open.
                    assert_eq!(
                        ds.layout().unwrap(),
                        hdf5_pure::Layout::Contiguous {
                            address: None,
                            size: 0
                        },
                        "{case}"
                    );
                }
            }
        }
    }
}

/// The refusal lands before any byte reaches the filesystem, which is what
/// decides whether it destroys the file it was overwriting.
///
/// `FileBuilder::write` creates its path on the first byte emitted rather than
/// up front, so a build refused before that point leaves the destination alone
/// while one refused during layout still leaves a partial file. That mechanism
/// is pinned elsewhere (`file_properties`, `attr_message_size_limit`); what this
/// adds is that *this* guard is among the refusals that reach it, which nothing
/// about where it sits in `flatten_dataset` says on its own.
#[test]
fn a_rejected_zero_element_write_leaves_the_destination_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("target.h5");

    let mut good = FileBuilder::new();
    good.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    good.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    let mut bad = FileBuilder::new();
    bad.create_dataset("bad")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[0]);
    bad.write(&path).unwrap_err();

    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused write must not disturb the file it was overwriting"
    );

    let fresh = dir.path().join("never.h5");
    let mut bad = FileBuilder::new();
    bad.create_dataset("bad")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[0]);
    bad.write(&fresh).unwrap_err();
    assert!(
        !fresh.exists(),
        "a refused write must not leave a stub behind"
    );
}

#[test]
fn matching_shape_and_data_is_accepted() {
    // Regression guard: a correct 2x3 shape must still write/read cleanly.
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("ok")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .with_shape(&[2, 3]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let ds = file.dataset("ok").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ---------------------------------------------------------------------------
// Userblock (non-zero base address) reads. A userblock shifts the HDF5 image
// forward, so every stored address is relative to the base; the reader must
// apply the base to contiguous data, the chunk index, and each chunk's data.
// (These exercise the in-memory read path; the C-library cross-checks live in
// `crates/crosscheck/tests/edit_userblock.rs`.)
// ---------------------------------------------------------------------------

/// Build a file with a 512-byte userblock and read a dataset back.
fn userblock_roundtrip_f64(build: impl FnOnce(&mut FileBuilder)) -> File {
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    build(&mut b);
    let mut bytes = b.finish().unwrap();
    // Stamp the userblock so a stray read into it would corrupt the marker.
    bytes[..6].copy_from_slice(b"UBMARK");
    File::from_bytes(bytes).unwrap()
}

#[test]
fn userblock_contiguous_read() {
    let file = userblock_roundtrip_f64(|b| {
        b.create_dataset("d").with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    });
    assert_eq!(
        file.dataset("d").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn userblock_chunked_deflate_read() {
    // A multi-chunk deflate dataset: exercises the chunk-index walk plus per-chunk
    // data reads, all base-relative.
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
    let expect = data.clone();
    let file = userblock_roundtrip_f64(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[1000])
            .with_deflate(6);
    });
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), expect);
}

#[test]
fn userblock_chunked_unfiltered_2d_read() {
    let data: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let expect = data.clone();
    let file = userblock_roundtrip_f64(|b| {
        b.create_dataset("d")
            .with_f64_data(&data)
            .with_shape(&[8, 8])
            .with_chunks(&[4, 4]);
    });
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![8, 8]);
    assert_eq!(ds.read_f64().unwrap(), expect);
}

#[test]
fn userblock_chunked_deflate_streaming_read() {
    // The streaming reader walks the chunk index through `BaseOffsetSource`. This
    // covers that path (and its metadata-cache forwarding) on a userblock file and
    // checks it matches a buffered read byte-for-byte.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_chunk_stream.h5");
    let data: Vec<f64> = (0..1000).map(|i| (i % 19) as f64 * 0.25).collect();

    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("d")
        .with_f64_data(&data)
        .with_shape(&[1000])
        .with_chunks(&[64])
        .with_deflate(6);
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    let buffered = File::open(&path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_f64()
        .unwrap();
    let streamed = File::open_streaming(&path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_f64()
        .unwrap();
    assert_eq!(buffered, data);
    assert_eq!(streamed, data);
}
