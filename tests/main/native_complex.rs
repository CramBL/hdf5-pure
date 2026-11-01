use hdf5_pure::{
    AttrValue, ComplexType, CompoundTypeBuilder, DType, Datatype, DatatypeByteOrder, Error, File,
    FileBuilder, FloatingPointType, FormatError, MaxExtent, MessageType, RepackOptions,
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
    let path = temp::temp_path("native_complex.h5");
    let image = container.file(component);
    std::fs::write(&path, &image).unwrap();
    let file = backend.open(&path).unwrap();
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
    drop(file);
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
