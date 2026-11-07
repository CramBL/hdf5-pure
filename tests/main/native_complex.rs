use hdf5_pure::{
    AttrValue, ComplexType, CompoundTypeBuilder, DType, Datatype, DatatypeByteOrder, Error, File,
    FileBuilder, FloatingPointType, FormatError, Layout, LibVer, MaxExtent, MessageType,
    RepackOptions, VariableLengthType,
};
use hdf5_pure_format::__private::{AccessMode, AttributeMessage, ObjectHeader};
use rstest::rstest;
use test_util::native_complex::{Component, Container};
use test_util::temp;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::storage;

#[rstest]
#[case::f32_le(Component::F32Le)]
#[case::f32_be(Component::F32Be)]
#[case::f64_le(Component::F64Le)]
#[case::f64_be(Component::F64Be)]
fn native_complex_public_inspection_preserves_type_and_extent(
    #[case] component: Component,
    #[values(
        Container::Native,
        Container::Array,
        Container::Compound,
        Container::MatlabCompound
    )]
    container: Container,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let file = backend.open_image(container.file(component)).unwrap();
    let dataset = file.dataset("values").unwrap();
    let expected = datatype(component, container);
    assert_eq!(dataset.datatype().unwrap(), expected);
    assert_eq!(dataset.dtype().unwrap(), dtype(component, container));
    assert_eq!(dataset.shape().unwrap(), [container.count()]);
    assert_eq!(
        dataset.element_size().unwrap(),
        u64::from(expected.type_size())
    );
    assert_eq!(dataset.read_raw().unwrap(), component.values());
    assert_eq!(
        dataset.read_raw_rows(0, 1).unwrap(),
        component.values()[..usize::try_from(expected.type_size()).unwrap()]
    );
    assert_eq!(
        dataset.read_raw_region(&[0], &[1]).unwrap(),
        dataset.read_raw_rows(0, 1).unwrap()
    );
    assert_eq!(
        dataset.attr_datatypes().unwrap(),
        [("values".into(), expected.clone())].into()
    );
    assert_eq!(
        file.root().attr_datatypes().unwrap(),
        [("values".into(), expected)].into()
    );
    assert!(dataset.attrs().unwrap().is_empty());
    assert!(file.root().attrs().unwrap().is_empty());
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
#[case::f32_le(Component::F32Le)]
#[case::f32_be(Component::F32Be)]
#[case::f64_le(Component::F64Le)]
#[case::f64_be(Component::F64Be)]
fn native_complex_numeric_readers_report_the_complex_class(#[case] component: Component) {
    let file = File::from_bytes(Container::Native.file(component)).unwrap();
    let dataset = file.dataset("values").unwrap();
    numeric_mismatch(dataset.read_f64().unwrap_err());
    numeric_mismatch(dataset.read_f32().unwrap_err());
    numeric_mismatch(dataset.read_i64().unwrap_err());
    numeric_mismatch(dataset.read_i32().unwrap_err());
    numeric_mismatch(dataset.read_i16().unwrap_err());
    numeric_mismatch(dataset.read_i8().unwrap_err());
    numeric_mismatch(dataset.read_u64().unwrap_err());
    numeric_mismatch(dataset.read_u32().unwrap_err());
    numeric_mismatch(dataset.read_u16().unwrap_err());
    numeric_mismatch(dataset.read_u8().unwrap_err());
    numeric_mismatch(dataset.read::<f64>().unwrap_err());
    numeric_mismatch(dataset.read_f64_rows(0, 1).unwrap_err());
    numeric_mismatch(dataset.read_f64_region(&[0], &[1]).unwrap_err());
}

#[rstest]
#[case::f32_le(Component::F32Le)]
#[case::f32_be(Component::F32Be)]
#[case::f64_le(Component::F64Le)]
#[case::f64_be(Component::F64Be)]
fn native_complex_repack_preserves_metadata_and_element_bytes(
    #[case] component: Component,
    #[values(
        Container::Native,
        Container::Array,
        Container::Compound,
        Container::MatlabCompound
    )]
    container: Container,
) {
    let source = temp::temp_path("native_complex_source.h5");
    let destination = temp::temp_path("native_complex_destination.h5");
    std::fs::write(&source, container.file(component)).unwrap();
    hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
    let image = std::fs::read(&destination).unwrap();
    let file = File::open(&destination).unwrap();
    let expected = datatype(component, container);
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.datatype().unwrap(), expected);
    assert_eq!(dataset.dtype().unwrap(), dtype(component, container));
    assert_eq!(dataset.shape().unwrap(), [container.count()]);
    assert_eq!(dataset.read_raw().unwrap(), component.values());
    assert_attribute_bytes(&file, &image, &expected, &component.values());
    drop(dataset);
    drop(file);
}

#[rstest]
#[case::f32_le(Component::F32Le)]
#[case::f32_be(Component::F32Be)]
#[case::f64_le(Component::F64Le)]
#[case::f64_be(Component::F64Be)]
fn native_complex_edits_preserve_datatypes_and_attributes(
    #[case] component: Component,
    #[values(
        Container::Native,
        Container::Array,
        Container::Compound,
        Container::MatlabCompound
    )]
    container: Container,
) {
    let path = temp::temp_path("native_complex_edit.h5");
    std::fs::write(&path, container.file(component)).unwrap();
    let expected = datatype(component, container);
    let replacement = vec![0; component.values().len()];
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("values").unwrap();
        dataset
            .write_staged(|builder| {
                builder.with_raw_data(expected.clone(), replacement.clone(), container.count());
            })
            .unwrap();
        file.commit().unwrap();
        dataset.set_attr("label", AttrValue::I32(7)).unwrap();
        file.root().set_attr("label", AttrValue::I32(8)).unwrap();
        file.commit().unwrap();
    }
    let image = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.datatype().unwrap(), expected);
    assert_eq!(dataset.read_raw().unwrap(), replacement);
    assert_eq!(
        dataset.attrs().unwrap(),
        [("label".into(), AttrValue::I32(7))].into()
    );
    assert_eq!(
        file.root().attrs().unwrap(),
        [("label".into(), AttrValue::I32(8))].into()
    );
    assert_attribute_bytes(&file, &image, &expected, &component.values());
    drop(dataset);
    drop(file);
}

#[rstest]
#[case::f32_le(Component::F32Le)]
#[case::f32_be(Component::F32Be)]
#[case::f64_le(Component::F64Le)]
#[case::f64_be(Component::F64Be)]
fn native_complex_raw_append_preserves_element_width_and_order(
    #[case] component: Component,
    #[values(
        Container::Native,
        Container::Array,
        Container::Compound,
        Container::MatlabCompound
    )]
    container: Container,
) {
    let path = temp::temp_path("native_complex_append.h5");
    let expected = datatype(component, container);
    let raw = component.values();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_raw_data(expected.clone(), raw.clone(), container.count())
        .with_chunks(&[1])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("values").unwrap();
    if component.big_endian() {
        let err = dataset.append_raw(&raw).unwrap_err();
        let Error::AppendInPlaceUnsupported(reason) = &err else {
            panic!("{err:?}");
        };
        assert_eq!(
            *reason,
            "append_raw onto this dataset's datatype (non-little-endian, variable-length, or reference) could misencode the bytes; use a typed append"
        );
        assert_eq!(dataset.shape().unwrap(), [container.count()]);
        assert_eq!(dataset.read_raw().unwrap(), raw);
    } else {
        dataset.append_raw(&raw).unwrap();
        assert_eq!(dataset.shape().unwrap(), [container.count() * 2]);
        assert_eq!(dataset.read_raw().unwrap(), raw.repeat(2));
    }
    assert_eq!(dataset.datatype().unwrap(), expected);
    let err = dataset.append(&[1.0f32; 8]).unwrap_err();
    let Error::AppendInPlaceUnsupported(reason) = &err else {
        panic!("{err:?}");
    };
    assert_eq!(
        *reason,
        "append datatype does not match the on-disk dataset (wrong element type or byte order)"
    );
    drop(dataset);
    drop(file);
}

#[test]
fn native_complex_overwrite_rejects_a_compound_substitution() {
    let path = temp::temp_path("native_complex_retype.h5");
    let original = Container::Native.file(Component::F32Le);
    std::fs::write(&path, &original).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("values")
            .unwrap()
            .write_staged(|builder| {
                builder.with_complex32_data(&[(0.0, 0.0), (0.0, 0.0)]);
            })
            .unwrap();
        let err = file.commit().unwrap_err();
        let Error::EditUnsupported(reason) = &err else {
            panic!("{err:?}");
        };
        assert_eq!(
            *reason,
            "write_dataset datatype does not match the on-disk dataset (overwrite, not retype)"
        );
    }
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[rstest]
#[case::native(datatype(Component::F32Le, Container::Native))]
#[case::array(datatype(Component::F32Le, Container::Array))]
#[case::compound(datatype(Component::F32Le, Container::Compound))]
#[case::sequence(Datatype::VariableLength(VariableLengthType::sequence(datatype(
    Component::F32Le,
    Container::Native
))))]
#[case::nested_sequence(Datatype::VariableLength(VariableLengthType::sequence(datatype(
    Component::F32Le,
    Container::Array
))))]
fn native_complex_creation_honors_explicit_content_ceilings(
    #[case] datatype: Datatype,
    #[values(
        None,
        Some(LibVer::V18),
        Some(LibVer::V110),
        Some(LibVer::V112),
        Some(LibVer::V114)
    )]
    ceiling: Option<LibVer>,
    #[values(
        CreationTarget::Dataset,
        CreationTarget::GroupDataset,
        CreationTarget::Committed,
        CreationTarget::GroupCommitted
    )]
    target: CreationTarget,
) {
    let mut builder = FileBuilder::new();
    if let Some(ceiling) = ceiling {
        builder.with_libver_bounds(LibVer::Earliest, ceiling);
    }
    let path = match target {
        CreationTarget::Dataset => {
            builder
                .create_dataset("value")
                .with_unallocated_storage(datatype.clone(), &[1]);
            "value"
        }
        CreationTarget::GroupDataset => {
            let mut group = builder.create_group("group");
            group
                .create_dataset("value")
                .with_unallocated_storage(datatype.clone(), &[1]);
            builder.add_group(group.finish());
            "group/value"
        }
        CreationTarget::Committed => {
            builder.commit_datatype("value", datatype.clone());
            "value"
        }
        CreationTarget::GroupCommitted => {
            let mut group = builder.create_group("group");
            group.commit_datatype("value", datatype.clone());
            builder.add_group(group.finish());
            "group/value"
        }
    };
    if let Some(ceiling) = ceiling {
        let error = builder.finish().unwrap_err();
        let Error::Format(actual) = error else {
            panic!("expected format error, got {error:?}");
        };
        assert_eq!(
            actual,
            FormatError::LibverTooOldForContent {
                content: "a native complex datatype",
                needs: "v2.0",
                writing: ceiling.name(),
            }
        );
        assert_eq!(
            actual.to_string(),
            format!(
                "a native complex datatype requires the v2.0 format, but the requested \
                 library-version bounds allow at most {}",
                ceiling.name(),
            ),
        );
    } else {
        let file = File::from_bytes(builder.finish().unwrap()).unwrap();
        assert_eq!(file.libver_bound(), LibVer::WRITER_DEFAULT);
        match target {
            CreationTarget::Dataset | CreationTarget::GroupDataset => {
                let dataset = file.dataset(path).unwrap();
                assert_eq!(dataset.datatype().unwrap(), datatype);
                assert_eq!(dataset.shape().unwrap(), [1]);
                assert_eq!(
                    dataset.layout().unwrap(),
                    Layout::Contiguous {
                        address: None,
                        size: u64::from(datatype.type_size()),
                    },
                );
            }
            CreationTarget::Committed | CreationTarget::GroupCommitted => {
                assert_eq!(file.root().named_datatype(path).unwrap(), datatype);
            }
        }
    }
}

#[rstest]
#[case::native(datatype(Component::F32Le, Container::Native))]
#[case::array(datatype(Component::F32Le, Container::Array))]
#[case::compound(datatype(Component::F32Le, Container::Compound))]
#[case::sequence(Datatype::VariableLength(VariableLengthType::sequence(datatype(
    Component::F32Le,
    Container::Native
))))]
#[case::nested_sequence(Datatype::VariableLength(VariableLengthType::sequence(datatype(
    Component::F32Le,
    Container::Array
))))]
fn native_complex_repack_honors_explicit_content_ceilings(
    #[case] datatype: Datatype,
    #[values(
        None,
        Some(LibVer::V18),
        Some(LibVer::V110),
        Some(LibVer::V112),
        Some(LibVer::V114)
    )]
    ceiling: Option<LibVer>,
    #[values(
        CreationTarget::Dataset,
        CreationTarget::GroupDataset,
        CreationTarget::Committed,
        CreationTarget::GroupCommitted
    )]
    target: CreationTarget,
) {
    let mut builder = FileBuilder::new();
    let path = match target {
        CreationTarget::Dataset => {
            builder
                .create_dataset("value")
                .with_unallocated_storage(datatype.clone(), &[1]);
            "value"
        }
        CreationTarget::GroupDataset => {
            let mut group = builder.create_group("group");
            group
                .create_dataset("value")
                .with_unallocated_storage(datatype.clone(), &[1]);
            builder.add_group(group.finish());
            "group/value"
        }
        CreationTarget::Committed => {
            builder.commit_datatype("value", datatype.clone());
            "value"
        }
        CreationTarget::GroupCommitted => {
            let mut group = builder.create_group("group");
            group.commit_datatype("value", datatype.clone());
            builder.add_group(group.finish());
            "group/value"
        }
    };
    let source = File::from_bytes(builder.finish().unwrap()).unwrap();
    let options = ceiling.map_or_else(RepackOptions::new, |ceiling| {
        RepackOptions::new().with_libver_bounds(LibVer::Earliest, ceiling)
    });
    let mut bytes = b"previous bytes".to_vec();
    let destination = temp::temp_path("native_complex_ceiling.h5");
    if let Some(ceiling) = ceiling {
        let error = hdf5_pure::repack_into(&source, &mut bytes, &options).unwrap_err();
        let Error::Format(actual) = error else {
            panic!("expected format error, got {error:?}");
        };
        assert_eq!(
            actual,
            FormatError::LibverTooOldForContent {
                content: "a native complex datatype",
                needs: "v2.0",
                writing: ceiling.name(),
            }
        );
        assert_eq!(bytes, b"previous bytes");
        let error = hdf5_pure::repack_to_path(&source, &destination, &options).unwrap_err();
        let Error::Format(path_error) = error else {
            panic!("expected format error, got {error:?}");
        };
        assert_eq!(path_error, actual);
        assert!(!destination.exists());
    } else {
        bytes.clear();
        hdf5_pure::repack_into(&source, &mut bytes, &options).unwrap();
        let output = File::from_bytes(bytes).unwrap();
        match target {
            CreationTarget::Dataset | CreationTarget::GroupDataset => {
                let dataset = output.dataset(path).unwrap();
                assert_eq!(dataset.datatype().unwrap(), datatype);
                assert_eq!(dataset.shape().unwrap(), [1]);
                assert_eq!(
                    dataset.layout().unwrap(),
                    Layout::Contiguous {
                        address: None,
                        size: u64::from(datatype.type_size()),
                    }
                );
            }
            CreationTarget::Committed | CreationTarget::GroupCommitted => {
                assert_eq!(output.root().named_datatype(path).unwrap(), datatype);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum CreationTarget {
    Committed,
    Dataset,
    GroupCommitted,
    GroupDataset,
}

fn datatype(component: Component, container: Container) -> Datatype {
    let order = if component.big_endian() {
        DatatypeByteOrder::BigEndian
    } else {
        DatatypeByteOrder::LittleEndian
    };
    let component = if component.size() == 4 {
        FloatingPointType::ieee754_binary32(order)
    } else {
        FloatingPointType::ieee754_binary64(order)
    };
    let native = Datatype::Complex(ComplexType::new(component).unwrap());
    match container {
        Container::Native => native,
        Container::Array => Datatype::Array {
            base_type: Box::new(native),
            dimensions: vec![2],
        },
        Container::Compound => CompoundTypeBuilder::with_size(native.type_size())
            .field("value", 0, native)
            .build()
            .unwrap(),
        Container::MatlabCompound => CompoundTypeBuilder::with_size(component.size().get() * 2)
            .field("real", 0, Datatype::FloatingPoint(component))
            .field(
                "imag",
                u64::from(component.size().get()),
                Datatype::FloatingPoint(component),
            )
            .build()
            .unwrap(),
    }
}

fn dtype(component: Component, container: Container) -> DType {
    let native = DType::Other(Box::new(datatype(component, Container::Native)));
    match container {
        Container::Native => native,
        Container::Array => DType::Array(Box::new(native), vec![2]),
        Container::Compound => DType::Compound(vec![("value".into(), native)]),
        Container::MatlabCompound => {
            let scalar = if component.size() == 4 {
                DType::F32
            } else {
                DType::F64
            };
            DType::Compound(vec![
                ("real".into(), scalar.clone()),
                ("imag".into(), scalar),
            ])
        }
    }
}

fn numeric_mismatch(err: Error) {
    let Error::Format(FormatError::TypeMismatch { expected, actual }) = &err else {
        panic!("{err:?}");
    };
    assert_eq!(*expected, "numeric");
    assert_eq!(*actual, "Complex");
}

fn assert_attribute_bytes(file: &File, image: &[u8], expected: &Datatype, raw: &[u8]) {
    let superblock = file.superblock();
    for address in [
        file.resolve_address(superblock.root_group_address).unwrap(),
        storage::root_object_address(file, image, "values"),
    ] {
        let header = ObjectHeader::parse(
            image,
            AccessMode::ReadOnly,
            usize::try_from(address.get()).unwrap(),
            superblock.offset_size.get(),
            superblock.length_size.get(),
        )
        .unwrap();
        let attributes: Vec<_> = header
            .messages
            .iter()
            .filter(|record| record.msg_type == MessageType::ATTRIBUTE)
            .map(|record| {
                AttributeMessage::parse(&record.data, superblock.length_size.get()).unwrap()
            })
            .filter(|attribute| attribute.name == "values")
            .collect();
        assert_eq!(attributes.len(), 1);
        assert_eq!(&attributes[0].datatype, expected);
        assert_eq!(attributes[0].raw_data, raw);
    }
}
