#![cfg(feature = "__libhdf5-1.10")]

use hdf5_pure::File;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::MaxExtent;
use hdf5_pure_core::MessageType;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::Dataspace;
use hdf5_pure_format::__private::DataspaceType;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::ObjectHeader;
use libhdf5::Extents;
use libhdf5::file::LibraryVersion;
use libhdf5::plist::file_create::Sizeof;
use libhdf5::plist::file_create::SizeofInfo;
use rstest::rstest;
use test_util::dataspace;
use test_util_hdf5::__private::c_object;

#[rstest]
#[case::two_version_one_scalar(LengthWidth::Two, Sizeof::Bytes2, Shape::Scalar, false)]
#[case::two_version_two_bounded(LengthWidth::Two, Sizeof::Bytes2, Shape::Bounded, true)]
#[case::four_version_one_bounded(LengthWidth::Four, Sizeof::Bytes4, Shape::Bounded, false)]
#[case::four_version_two_fixed(LengthWidth::Four, Sizeof::Bytes4, Shape::Fixed, true)]
#[case::eight_version_one_fixed(LengthWidth::Eight, Sizeof::Bytes8, Shape::Fixed, false)]
#[case::eight_version_two_scalar(LengthWidth::Eight, Sizeof::Bytes8, Shape::Scalar, true)]
fn dataspace_dimension_arrays_preserve_native_values_and_ranges(
    #[case] width: LengthWidth,
    #[case] size: Sizeof,
    #[case] shape: Shape,
    #[case] latest: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dimensions.h5");
    let file = libhdf5::File::with_options()
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
            Dataspace::new(
                match shape {
                    Shape::Scalar => DataspaceType::Scalar,
                    Shape::Fixed | Shape::Bounded => DataspaceType::Simple,
                },
                dimensions,
                (body[2] & dataspace::Flags::MAX_DIMS.0 != 0).then_some(maxima)
            )
            .unwrap(),
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
