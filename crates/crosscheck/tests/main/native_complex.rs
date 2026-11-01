//! Native complex datatype and value interoperability with HDF5 2.x.

#![cfg(feature = "__hdf5-2")]

use std::path::Path;

use hdf5::Datatype as NativeDatatype;
use hdf5::file::LibraryVersion;
use hdf5_pure::{
    AttrValue, ComplexType, CompoundTypeBuilder, DType, Datatype, DatatypeByteOrder, Error, File,
    FileBuilder, FloatingPointType, FormatError, RepackOptions, VariableLengthType,
};
use hdf5_sys::h5a;
use hdf5_sys::h5d;
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s::{self, H5S_ALL};
use hdf5_sys::h5t::{self, H5T_class_t, hvl_t};
use rstest::rstest;

use test_util::native_complex::Component;
use test_util_hdf5::lock;

#[rstest]
#[case::f32_be(Component::F32Be)]
#[case::f32_le(Component::F32Le)]
#[case::f64_be(Component::F64Be)]
#[case::f64_le(Component::F64Le)]
fn native_complex_preserves_native_dataset_and_attribute_values(
    #[case] component: Component,
    #[values(
        FixedShape::Array,
        FixedShape::ArrayCompound,
        FixedShape::Compound,
        FixedShape::MatlabCompound,
        FixedShape::Native
    )]
    shape: FixedShape,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("destination.h5");
    let stored = shape.native_type(component);
    let memory = shape.native_type(native_component());
    {
        let file = native_file(&source);
        let dataset = file
            .new_dataset_builder()
            .empty_as(&stored)
            .shape(shape.count())
            .create("values")
            .unwrap();
        let attribute = dataset
            .new_attr_builder()
            .empty_as(&stored)
            .shape(shape.count())
            .create("values")
            .unwrap();
        hdf5::sync::sync(|| {
            // SAFETY: Live handles and the library lock cover both calls. The memory datatype and
            // each object's element count describe the four f64 components in `VALUES`, with no
            // pointer fields.
            unsafe {
                assert_eq!(
                    h5d::H5Dwrite(
                        dataset.id(),
                        memory.id(),
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        VALUES.as_ptr().cast(),
                    ),
                    0,
                );
                assert_eq!(
                    h5a::H5Awrite(attribute.id(), memory.id(), VALUES.as_ptr().cast()),
                    0
                );
            }
        });
        drop(attribute);
        drop(dataset);
        file.close().unwrap();
    }
    observe_fixed(&source, &stored, &memory, VALUES, VALUES);
    let expected = shape.pure_type(component);
    {
        let file = File::open(&source).unwrap();
        let dataset = file.dataset("values").unwrap();
        assert_eq!(dataset.datatype().unwrap(), expected);
        assert_eq!(
            dataset.shape().unwrap(),
            [u64::try_from(shape.count()).unwrap()]
        );
        assert_eq!(dataset.read_raw().unwrap(), component.values());
        assert_eq!(
            dataset.element_size().unwrap(),
            u64::from(expected.type_size())
        );
        assert_eq!(
            dataset.attr_datatypes().unwrap(),
            [("values".into(), expected.clone())].into()
        );
        if matches!(shape, FixedShape::Native) {
            assert_eq!(
                dataset.dtype().unwrap(),
                DType::Other(Box::new(expected.clone()))
            );
            let error = dataset.read_f64().unwrap_err();
            let Error::Format(FormatError::TypeMismatch { expected, actual }) = &error else {
                panic!("{error:?}");
            };
            assert_eq!(*expected, "numeric");
            assert_eq!(*actual, "Complex");
        }
    }
    hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
    observe_fixed(&destination, &stored, &memory, VALUES, VALUES);
    {
        let file = File::open_rw(&destination).unwrap();
        let mut dataset = file.dataset("values").unwrap();
        dataset
            .write_staged(|builder| {
                builder.with_raw_data(
                    expected,
                    vec![0; component.values().len()],
                    u64::try_from(shape.count()).unwrap(),
                );
            })
            .unwrap();
        file.commit().unwrap();
        dataset.set_attr("label", AttrValue::I32(7)).unwrap();
        file.commit().unwrap();
    }
    observe_fixed(&destination, &stored, &memory, [0.0; 4], VALUES);
}

#[rstest]
#[case::f32_be(Component::F32Be)]
#[case::f32_le(Component::F32Le)]
#[case::f64_be(Component::F64Be)]
#[case::f64_le(Component::F64Le)]
fn native_complex_preserves_native_sequence_values(
    #[case] component: Component,
    #[values(
        SequenceShape::ArrayCompound,
        SequenceShape::CompoundArray,
        SequenceShape::Native,
        SequenceShape::SequenceArrayCompound
    )]
    shape: SequenceShape,
    #[values(false, true)] attribute: bool,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("destination.h5");
    let type_file = directory.path().join("datatype.h5");
    let stored = shape.native_type(component);
    let memory = shape.native_type(native_component());
    {
        let file = native_file(&source);
        let dataset = file
            .new_dataset_builder()
            .empty_as(&stored)
            .shape(1)
            .create("values")
            .unwrap();
        let value = hvl_t {
            len: 2,
            p: VALUES.as_ptr().cast_mut().cast(),
        };
        hdf5::sync::sync(|| {
            assert_eq!(
                // SAFETY: The live memory type describes one VL descriptor with two complex f64
                // elements. Its pointer remains valid for the complete synchronous write.
                unsafe {
                    h5d::H5Dwrite(
                        dataset.id(),
                        memory.id(),
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        core::ptr::from_ref(&value).cast(),
                    )
                },
                0
            );
        });
        if attribute {
            let attr = dataset
                .new_attr_builder()
                .empty_as(&stored)
                .shape(1)
                .create("values")
                .unwrap();
            hdf5::sync::sync(|| {
                assert_eq!(
                    // SAFETY: The memory type describes one descriptor whose two complete values
                    // remain live, and the library lock covers the attribute write.
                    unsafe {
                        h5a::H5Awrite(attr.id(), memory.id(), core::ptr::from_ref(&value).cast())
                    },
                    0
                );
            });
        }
        drop(dataset);
        file.close().unwrap();
    }
    observe_sequence(&source, &stored, &memory, attribute);
    let expected = shape.pure_type(component);
    {
        let file = File::open(&source).unwrap();
        let dataset = file.dataset("values").unwrap();
        assert_eq!(dataset.datatype().unwrap(), expected);
        assert_eq!(dataset.shape().unwrap(), [1]);
        assert_eq!(dataset.element_size().unwrap(), 16);
        if attribute {
            assert_eq!(
                dataset.attr_datatypes().unwrap(),
                [("values".into(), expected.clone())].into()
            );
        }
    }
    let mut builder = FileBuilder::new();
    builder.commit_datatype("complex", expected);
    builder.write(&type_file).unwrap();
    {
        let file = hdf5::File::open(&type_file).unwrap();
        hdf5::sync::sync(|| {
            // SAFETY: The file is live and the datatype name is NUL-terminated.
            let datatype =
                owned_type(unsafe { h5t::H5Topen2(file.id(), c"complex".as_ptr(), H5P_DEFAULT) });
            assert_eq!(datatype, stored);
        });
        file.close().unwrap();
    }
    if attribute {
        let error = hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap_err();
        let Error::RepackUnsupported(reason) = &error else {
            panic!("{error:?}");
        };
        assert_eq!(
            reason,
            "dataset values: attribute \"values\" has a datatype that cannot be repacked faithfully yet"
        );
        assert!(!destination.exists());
    } else {
        hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
        observe_sequence(&destination, &stored, &memory, false);
    }
}

#[derive(Clone, Copy)]
enum SequenceShape {
    ArrayCompound,
    CompoundArray,
    Native,
    SequenceArrayCompound,
}

impl SequenceShape {
    fn native_type(self, component: Component) -> NativeDatatype {
        let base = native_complex(component);
        let base = if matches!(self, Self::SequenceArrayCompound) {
            native_array(&native_compound(&base), 1)
        } else {
            base
        };
        let sequence = hdf5::sync::sync(|| {
            // SAFETY: The base datatype remains live, and the library lock covers sequence creation.
            owned_type(unsafe { h5t::H5Tvlen_create(base.id()) })
        });
        match self {
            Self::ArrayCompound => native_array(&native_compound(&sequence), 1),
            Self::CompoundArray => native_compound(&native_array(&sequence, 1)),
            Self::Native | Self::SequenceArrayCompound => sequence,
        }
    }

    fn pure_type(self, component: Component) -> Datatype {
        let base = Datatype::Complex(ComplexType::new(pure_float(component)).unwrap());
        let base = if matches!(self, Self::SequenceArrayCompound) {
            pure_array(pure_compound(base), 1)
        } else {
            base
        };
        let sequence = Datatype::VariableLength(VariableLengthType::sequence(base));
        match self {
            Self::ArrayCompound => pure_array(pure_compound(sequence), 1),
            Self::CompoundArray => pure_compound(pure_array(sequence, 1)),
            Self::Native | Self::SequenceArrayCompound => sequence,
        }
    }
}

fn observe_sequence(
    path: &Path,
    stored: &NativeDatatype,
    memory: &NativeDatatype,
    attribute: bool,
) {
    let file = hdf5::File::open(path).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.dtype().unwrap(), *stored);
    let space = dataset.space().unwrap();
    assert_eq!(element_count(&space), 1);
    assert_eq!(memory.size(), core::mem::size_of::<hvl_t>());
    let mut value = hvl_t {
        len: 0,
        p: core::ptr::null_mut(),
    };
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The output holds one descriptor, matching the live memory type and dataset
            // extent. HDF5 allocates its two complete complex f64 elements.
            unsafe {
                h5d::H5Dread(
                    dataset.id(),
                    memory.id(),
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    core::ptr::from_mut(&mut value).cast(),
                )
            },
            0
        );
        assert_sequence_values(&value);
        assert_eq!(
            // SAFETY: The descriptor was allocated by `H5Dread` with this type and dataspace and is
            // reclaimed exactly once before the handles are released.
            unsafe {
                h5t::H5Treclaim(
                    memory.id(),
                    space.id(),
                    H5P_DEFAULT,
                    core::ptr::from_mut(&mut value).cast(),
                )
            },
            0
        );
        if attribute {
            let attr = dataset.attr("values").unwrap();
            assert_eq!(attr.dtype().unwrap(), *stored);
            assert_eq!(element_count(&attr.space().unwrap()), 1);
            value = hvl_t {
                len: 0,
                p: core::ptr::null_mut(),
            };
            assert_eq!(
                // SAFETY: The attribute has the same single-descriptor extent and memory type.
                unsafe {
                    h5a::H5Aread(
                        attr.id(),
                        memory.id(),
                        core::ptr::from_mut(&mut value).cast(),
                    )
                },
                0
            );
            assert_sequence_values(&value);
            assert_eq!(
                // SAFETY: The descriptor was allocated by `H5Aread` with this memory type, and the
                // matching one-element dataspace covers its single reclamation.
                unsafe {
                    h5t::H5Treclaim(
                        memory.id(),
                        space.id(),
                        H5P_DEFAULT,
                        core::ptr::from_mut(&mut value).cast(),
                    )
                },
                0
            );
        }
    });
    drop(space);
    drop(dataset);
    file.close().unwrap();
}

fn assert_sequence_values(value: &hvl_t) {
    assert_eq!(value.len, 2);
    assert!(!value.p.is_null());
    assert_eq!(
        // SAFETY: Callers supply an unreclaimed HDF5 VL allocation of two complex f64 elements. The
        // checked length and pointer cover four initialized, aligned f64 components.
        unsafe { core::slice::from_raw_parts(value.p.cast::<f64>(), 4) },
        VALUES
    );
}

#[derive(Clone, Copy)]
enum FixedShape {
    Array,
    ArrayCompound,
    Compound,
    MatlabCompound,
    Native,
}

impl FixedShape {
    fn count(self) -> usize {
        match self {
            Self::Array | Self::ArrayCompound => 1,
            Self::Compound | Self::MatlabCompound | Self::Native => 2,
        }
    }

    fn native_type(self, component: Component) -> NativeDatatype {
        let base = native_complex(component);
        match self {
            Self::Array => native_array(&base, 2),
            Self::ArrayCompound => native_array(&native_compound(&native_array(&base, 2)), 1),
            Self::Compound => native_compound(&base),
            Self::MatlabCompound => {
                let base = native_float(component);
                hdf5::sync::sync(|| {
                    // SAFETY: The lock serializes access, and each member fits the new compound.
                    let id = unsafe { h5t::H5Tcreate(H5T_class_t::H5T_COMPOUND, 2 * base.size()) };
                    let datatype = owned_type(id);
                    // SAFETY: The handles remain live, both names are NUL-terminated, and the
                    // members occupy two disjoint, complete floating-point slots.
                    unsafe {
                        assert_eq!(h5t::H5Tinsert(id, c"real".as_ptr(), 0, base.id()), 0);
                        assert_eq!(
                            h5t::H5Tinsert(id, c"imag".as_ptr(), base.size(), base.id()),
                            0
                        );
                    }
                    datatype
                })
            }
            Self::Native => base,
        }
    }

    fn pure_type(self, component: Component) -> Datatype {
        let float = pure_float(component);
        let base = Datatype::Complex(ComplexType::new(float).unwrap());
        match self {
            Self::Array => pure_array(base, 2),
            Self::ArrayCompound => pure_array(pure_compound(pure_array(base, 2)), 1),
            Self::Compound => pure_compound(base),
            Self::MatlabCompound => CompoundTypeBuilder::with_size(2 * float.size().get())
                .field("real", 0, Datatype::FloatingPoint(float))
                .field(
                    "imag",
                    u64::from(float.size().get()),
                    Datatype::FloatingPoint(float),
                )
                .build()
                .unwrap(),
            Self::Native => base,
        }
    }
}

fn observe_fixed(
    path: &Path,
    expected_type: &NativeDatatype,
    memory: &NativeDatatype,
    expected_data: [f64; 4],
    expected_attribute: [f64; 4],
) {
    let file = hdf5::File::open(path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let attribute = dataset.attr("values").unwrap();
    assert_eq!(dataset.dtype().unwrap(), *expected_type);
    assert_eq!(attribute.dtype().unwrap(), *expected_type);
    let mut data = [0.0f64; 4];
    let mut attr = [0.0f64; 4];
    assert_eq!(
        element_count(&dataset.space().unwrap())
            .checked_mul(memory.size())
            .unwrap(),
        core::mem::size_of_val(&data)
    );
    assert_eq!(
        element_count(&attribute.space().unwrap())
            .checked_mul(memory.size())
            .unwrap(),
        core::mem::size_of_val(&attr)
    );
    hdf5::sync::sync(|| {
        // SAFETY: The handles and lock cover both calls. Each output holds the four f64
        // components described by the memory datatype and the complete object extent.
        unsafe {
            assert_eq!(
                h5d::H5Dread(
                    dataset.id(),
                    memory.id(),
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    data.as_mut_ptr().cast()
                ),
                0
            );
            assert_eq!(
                h5a::H5Aread(attribute.id(), memory.id(), attr.as_mut_ptr().cast()),
                0
            );
        }
    });
    assert_eq!(data, expected_data);
    assert_eq!(attr, expected_attribute);
    drop(attribute);
    drop(dataset);
    file.close().unwrap();
}

fn element_count(space: &hdf5::Dataspace) -> usize {
    hdf5::sync::sync(|| {
        // SAFETY: The dataspace remains live and the library lock covers the extent query.
        let count = unsafe { h5s::H5Sget_simple_extent_npoints(space.id()) };
        usize::try_from(count).unwrap()
    })
}

fn native_file(path: &Path) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::latest(), LibraryVersion::latest())
        })
        .create(path)
        .unwrap()
}

fn native_complex(component: Component) -> NativeDatatype {
    let base = native_float(component);
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and `base` owns a live floating-point datatype.
        let id = unsafe { h5t::H5Tcomplex_create(base.id()) };
        let datatype = owned_type(id);
        assert_eq!(
            // SAFETY: The handle remains live and the lock serializes the query.
            unsafe { h5t::H5Tget_class(datatype.id()) },
            H5T_class_t::H5T_COMPLEX
        );
        datatype
    })
}

fn native_float(component: Component) -> NativeDatatype {
    let datatype = match component {
        Component::F32Be | Component::F32Le => NativeDatatype::from_type::<f32>().unwrap(),
        Component::F64Be | Component::F64Le => NativeDatatype::from_type::<f64>().unwrap(),
    };
    hdf5::sync::sync(|| {
        let order = if component.big_endian() {
            h5t::H5T_order_t::H5T_ORDER_BE
        } else {
            h5t::H5T_order_t::H5T_ORDER_LE
        };
        // SAFETY: The lock and live transient floating-point handle cover the setter.
        assert_eq!(unsafe { h5t::H5Tset_order(datatype.id(), order) }, 0);
    });
    datatype
}

fn native_array(base: &NativeDatatype, count: u64) -> NativeDatatype {
    hdf5::sync::sync(|| {
        // SAFETY: The base handle is live, and `count` describes one positive dimension.
        owned_type(unsafe { h5t::H5Tarray_create2(base.id(), 1, &count) })
    })
}

fn native_compound(base: &NativeDatatype) -> NativeDatatype {
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and the new compound has one complete base slot.
        let datatype =
            owned_type(unsafe { h5t::H5Tcreate(H5T_class_t::H5T_COMPOUND, base.size()) });
        assert_eq!(
            // SAFETY: The name is NUL-terminated and the member fits the live compound exactly.
            unsafe { h5t::H5Tinsert(datatype.id(), c"value".as_ptr(), 0, base.id()) },
            0
        );
        datatype
    })
}

fn owned_type(id: hid_t) -> NativeDatatype {
    assert!(id >= 0, "libhdf5 failed to create a datatype");
    // SAFETY: Every caller transfers an owned datatype ID under the library lock.
    unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap()
}

fn pure_float(component: Component) -> FloatingPointType {
    let order = if component.big_endian() {
        DatatypeByteOrder::BigEndian
    } else {
        DatatypeByteOrder::LittleEndian
    };
    match component {
        Component::F32Be | Component::F32Le => FloatingPointType::ieee754_binary32(order),
        Component::F64Be | Component::F64Le => FloatingPointType::ieee754_binary64(order),
    }
}

fn pure_array(base: Datatype, count: u32) -> Datatype {
    Datatype::Array {
        base_type: Box::new(base),
        dimensions: vec![count],
    }
}

fn pure_compound(base: Datatype) -> Datatype {
    CompoundTypeBuilder::with_size(base.type_size())
        .field("value", 0, base)
        .build()
        .unwrap()
}

fn native_component() -> Component {
    if cfg!(target_endian = "big") {
        Component::F64Be
    } else {
        Component::F64Le
    }
}

const VALUES: [f64; 4] = [1.25, -2.5, 3.5, 4.25];
