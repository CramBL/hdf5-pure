#![cfg(feature = "__hdf5-1.10")]

use hdf5::Extents;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5_pure::File;
use hdf5_pure_core::{LengthWidth, MaxExtent, MessageType, OffsetWidth};
use hdf5_pure_format::__private::{
    AccessMode, Dataspace, DataspaceType, FormatWidths, ObjectHeader,
};
use rstest::rstest;
use test_util::dataspace;
use test_util_hdf5::__private::c_object;

#[rstest]
#[case::two(LengthWidth::Two, Sizeof::Bytes2)]
#[case::four(LengthWidth::Four, Sizeof::Bytes4)]
#[case::eight(LengthWidth::Eight, Sizeof::Bytes8)]
fn dataspace_dimension_arrays_preserve_native_values_and_ranges(
    #[case] width: LengthWidth,
    #[case] size: Sizeof,
    #[values(Shape::Scalar, Shape::Fixed, Shape::Bounded)] shape: Shape,
    #[values(false, true)] latest: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dimensions.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            if latest {
                properties.libver_bounds(LibraryVersion::latest(), LibraryVersion::latest())
            } else {
                properties.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18)
            }
        })
        .with_fcpl(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr: Sizeof::Bytes8,
                sizeof_size: size,
            })
        })
        .create(&path)
        .unwrap();
    let builder = file.new_dataset::<u8>().shape(shape.extents());
    let builder = if matches!(shape, Shape::Bounded) {
        builder.chunk([3, 5])
    } else {
        builder
    };
    let dataset = builder.create("values").unwrap();
    drop(builder);
    let dimensions: Vec<_> = dataset
        .shape()
        .into_iter()
        .map(|dimension| u64::try_from(dimension).unwrap())
        .collect();
    let maxima: Vec<_> = dataset
        .space()
        .unwrap()
        .maxdims()
        .into_iter()
        .map(|maximum| MaxExtent::Fixed(u64::try_from(maximum.unwrap()).unwrap()))
        .collect();
    let address = c_object::header_address(&file, "values");
    drop(dataset);
    file.close().unwrap();

    let image = std::fs::read(&path).unwrap();
    let pure = File::from_bytes(image.clone()).unwrap();
    assert_eq!(pure.superblock().length_size, width);
    let header = ObjectHeader::parse_with_base_and_widths(
        &image,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: width,
        },
        pure.superblock().base_address,
    )
    .unwrap();
    let body = &header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATASPACE)
        .unwrap()
        .data;
    assert_eq!(body[0], if latest { 2 } else { 1 });
    let header_size = if latest {
        dataspace::V2_HEADER_LEN
    } else {
        dataspace::V1_HEADER_LEN
    };
    let expected_range = header_size..header_size + dimensions.len() * usize::from(width.get());
    assert_eq!(
        Dataspace::parse_with_dimension_fields(body, width).unwrap(),
        (
            Dataspace {
                space_type: match shape {
                    Shape::Scalar => DataspaceType::Scalar,
                    Shape::Fixed | Shape::Bounded => DataspaceType::Simple,
                },
                rank: u8::try_from(dimensions.len()).unwrap(),
                dimensions,
                max_dimensions: (body[2] & dataspace::Flags::MAX_DIMS.0 != 0).then_some(maxima),
            },
            expected_range,
        )
    );
    drop(pure);
    directory.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Bounded,
    Fixed,
    Scalar,
}

impl Shape {
    fn extents(self) -> Extents {
        match self {
            Self::Bounded => [(3, Some(7)), (5, Some(9))].into(),
            Self::Fixed => [3, 5].into(),
            Self::Scalar => ().into(),
        }
    }
}
