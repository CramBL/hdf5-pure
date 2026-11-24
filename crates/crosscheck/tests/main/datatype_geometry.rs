use libhdf5::types::{IntSize, TypeDescriptor};
use libhdf5::{File, MajorErrorCode, MinorErrorCode};
use rstest::rstest;
use test_util::datatype::containers::{self, CompoundField, EnumValue, MemberOffset};
use test_util::datatype::leaf::{Leaf, Version};
use test_util::image::Image;
use test_util::link_info::{self, CreationIndex, Width};
use test_util::link_message::HardLink;
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageType};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util::{data_layout, dataspace};
use test_util_hdf5::lock;

#[rstest]
#[case::out_of_bounds(9, "member type extends outside its parent compound type")]
#[case::overlap(1, "member overlaps with previous member")]
fn libhdf5_rejects_invalid_compound_geometry(
    #[case] second_offset: u64,
    #[case] description: &str,
) {
    assert_geometry_error(
        &compound(second_offset),
        12,
        MinorErrorCode::CantDecode,
        description,
    );
}

#[rstest]
#[case::descending_offsets(0)]
#[case::ascending_offsets(8)]
fn libhdf5_reads_adjacent_compound_members_in_declaration_order(#[case] second_offset: u64) {
    let _guard = lock::libhdf5_guard();
    let datatype = native_descriptor(&compound(second_offset), 12).unwrap();
    let TypeDescriptor::Compound(compound) = datatype else {
        panic!("expected a compound datatype, got {datatype:?}");
    };
    assert_eq!(compound.size, 12);
    let fields = compound
        .fields
        .iter()
        .map(|field| (field.name.as_str(), field.offset))
        .collect::<Vec<_>>();
    assert_eq!(
        fields,
        [
            ("first", 4),
            ("second", usize::try_from(second_offset).unwrap())
        ]
    );
}

#[test]
fn libhdf5_rejects_enumeration_size_mismatch() {
    assert_geometry_error(
        &enumeration(8),
        8,
        MinorErrorCode::BadSize,
        "ENUM datatype size does not match parent",
    );
}

#[test]
fn libhdf5_reads_matching_enumeration_size() {
    let _guard = lock::libhdf5_guard();
    let datatype = native_descriptor(&enumeration(4), 4).unwrap();
    let TypeDescriptor::Enum(enumeration) = datatype else {
        panic!("expected an enumeration datatype, got {datatype:?}");
    };
    assert_eq!(enumeration.size, IntSize::U4);
    let [member] = enumeration.members.as_slice() else {
        panic!("expected one enumeration member, got {enumeration:?}");
    };
    assert_eq!((member.name.as_str(), member.value), ("zero", 0));
}

fn assert_geometry_error(datatype: &[u8], size: usize, minor: MinorErrorCode, description: &str) {
    let _guard = lock::libhdf5_guard();
    let error = native_descriptor(datatype, size).unwrap_err();
    let Some(stack) = error.stack() else {
        panic!("expected a native HDF5 error stack, got {error:?}");
    };
    let frame = stack.last().unwrap();
    assert_eq!(
        (
            frame.major_code(),
            frame.minor_code(),
            frame.func(),
            frame.desc()
        ),
        (
            MajorErrorCode::Datatype,
            minor,
            "H5O__dtype_decode_helper",
            description
        ),
        "{stack:?}"
    );
}

fn native_descriptor(datatype: &[u8], size: usize) -> Result<TypeDescriptor, libhdf5::Error> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("datatype.h5");
    std::fs::write(&path, dataset_image(datatype, size)).unwrap();
    let file = File::open(path).unwrap();
    let descriptor = file
        .dataset("value")
        .and_then(|dataset| dataset.dtype()?.to_descriptor());
    file.close().unwrap();
    descriptor
}

fn compound(second_offset: u64) -> Vec<u8> {
    let base = Leaf::LittleInteger.bytes(Version::One);
    containers::compound(
        Version::Three.get(),
        [2, 0, 0],
        12,
        MemberOffset::One,
        &[
            CompoundField {
                name: b"first",
                offset: 4,
                datatype: &base,
            },
            CompoundField {
                name: b"second",
                offset: second_offset,
                datatype: &base,
            },
        ],
    )
}

fn enumeration(size: u32) -> Vec<u8> {
    containers::enumeration(
        Version::Three.get(),
        [1, 0, 0],
        size,
        &Leaf::LittleInteger.bytes(Version::One),
        &[EnumValue {
            name: b"zero",
            value: &[0; 4],
        }],
    )
}

fn dataset_image(datatype: &[u8], size: usize) -> Vec<u8> {
    let superblock = Superblock::new(Widths::EIGHT);
    let mut image = Image::starting_with(&superblock.build());
    let dataset = Header::new().messages([
        Message::new(
            MessageType::DATASPACE,
            &dataspace::VersionTwo::scalar().build(),
        ),
        Message::new(MessageType::DATATYPE, datatype),
        Message::new(
            MessageType::DATA_LAYOUT,
            &data_layout::compact(&vec![0; size]),
        ),
    ]);
    let dataset_address = u64::try_from(image.append(&dataset.build())).unwrap();
    let link_info = link_info::Body::new(Width::Eight, None, None, None, CreationIndex::Absent)
        .unwrap()
        .build();
    let link = HardLink::new("value", dataset_address).build(Widths::EIGHT);
    let root = Header::new().messages([
        Message::new(MessageType::LINK_INFO, &link_info),
        Message::new(MessageType::LINK, &link),
    ]);
    let root_address = u64::try_from(image.append(&root.build())).unwrap();
    image.place(
        0,
        &superblock
            .root_header_address(root_address)
            .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
            .build(),
    );
    image.build()
}
