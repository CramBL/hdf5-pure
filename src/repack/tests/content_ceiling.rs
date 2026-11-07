use rstest::rstest;

use super::super::test_util as repack_test_util;
use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::type_builders;
use crate::{
    AttrValue, ComplexType, DatatypeByteOrder, File, FileBuilder, FloatingPointType, LibVer,
    RepackOptions,
};

#[rstest]
#[case::unbounded(None)]
#[case::v18(Some(LibVer::V18))]
#[case::v110(Some(LibVer::V110))]
#[case::v112(Some(LibVer::V112))]
#[case::v114(Some(LibVer::V114))]
fn native_complex_attribute_repack_honors_explicit_content_ceilings(
    #[case] ceiling: Option<LibVer>,
    #[values(AttributeOwner::Dataset, AttributeOwner::Group, AttributeOwner::Root)]
    owner: AttributeOwner,
    #[values(false, true)] dense: bool,
    #[values(false, true)] array: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("destination.h5");
    let component = FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian);
    let native = Datatype::Complex(ComplexType::new(component).unwrap());
    let datatype = if array {
        Datatype::Array {
            base_type: Box::new(native),
            dimensions: vec![2],
        }
    } else {
        native
    };
    let mut message = type_builders::build_attr_message("complex", &AttrValue::I32(0)).unwrap();
    message.datatype = datatype.clone();
    message.raw_data = vec![0; usize::try_from(datatype.type_size()).unwrap()];
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("group");
    let dataset = builder.create_dataset("data").with_i32_data(&[1]);
    if matches!(owner, AttributeOwner::Dataset) {
        dataset.set_attr_verbatim(message.clone(), None);
        if dense {
            for index in 0..10 {
                dataset.set_attr(&format!("filler{index}"), AttrValue::I32(index));
            }
        }
    }
    if dense {
        for index in 0..10 {
            match owner {
                AttributeOwner::Dataset => {}
                AttributeOwner::Group => {
                    group.set_attr(&format!("filler{index}"), AttrValue::I32(index));
                }
                AttributeOwner::Root => {
                    builder.set_attr(&format!("filler{index}"), AttrValue::I32(index));
                }
            }
        }
    }
    match owner {
        AttributeOwner::Dataset => {}
        AttributeOwner::Group => group.set_attr_verbatim(message, None),
        AttributeOwner::Root => builder.set_attr_verbatim(message, None),
    }
    builder.add_group(group.finish());
    builder.write(&source).unwrap();
    let options = ceiling.map_or_else(RepackOptions::new, |ceiling| {
        RepackOptions::new().with_libver_bounds(LibVer::Earliest, ceiling)
    });
    if let Some(ceiling) = ceiling {
        let error = crate::repack(&source, &destination, &options).unwrap_err();
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
        assert!(!destination.exists());
    } else {
        crate::repack(&source, &destination, &options).unwrap();
        let file = File::open(&destination).unwrap();
        let datatypes = match owner {
            AttributeOwner::Dataset => file.dataset("data").unwrap().attr_datatypes().unwrap(),
            AttributeOwner::Group => file.group("group").unwrap().attr_datatypes().unwrap(),
            AttributeOwner::Root => file.root().attr_datatypes().unwrap(),
        };
        assert_eq!(datatypes.get("complex"), Some(&datatype));
        assert_eq!(datatypes.len(), if dense { 11 } else { 1 });
        file.close().unwrap();
        let path = match owner {
            AttributeOwner::Dataset => "data",
            AttributeOwner::Group => "group",
            AttributeOwner::Root => "",
        };
        assert_eq!(
            repack_test_util::messages_at(&destination, path),
            repack_test_util::messages_at(&source, path)
        );
    }
}

#[rstest]
#[case::dataset(false)]
#[case::committed(true)]
fn repack_constrains_only_retained_datatypes(
    #[case] committed: bool,
    #[values(LibVer::V18, LibVer::V114)] ceiling: LibVer,
) {
    let component = FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian);
    let datatype = Datatype::Complex(ComplexType::new(component).unwrap());
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("drop");
    if committed {
        group.commit_datatype("complex", datatype);
    } else {
        group
            .create_dataset("complex")
            .with_unallocated_storage(datatype, &[1]);
    }
    builder.add_group(group.finish());
    builder.create_dataset("keep").with_i32_data(&[3, 7]);
    let source = File::from_bytes(builder.finish().unwrap()).unwrap();
    let mut bytes = Vec::new();
    crate::repack_into(
        &source,
        &mut bytes,
        &RepackOptions::new()
            .drop_path("drop")
            .with_libver_bounds(LibVer::Earliest, ceiling),
    )
    .unwrap();
    let output = File::from_bytes(bytes).unwrap();
    assert_eq!(output.root().datasets().unwrap(), ["keep"]);
    assert_eq!(output.root().groups().unwrap(), Vec::<String>::new());
    assert_eq!(output.dataset("keep").unwrap().read_i32().unwrap(), [3, 7]);
    assert_eq!(output.libver_bound(), ceiling.min(LibVer::WRITER_DEFAULT));
}

#[derive(Clone, Copy)]
enum AttributeOwner {
    Dataset,
    Group,
    Root,
}
