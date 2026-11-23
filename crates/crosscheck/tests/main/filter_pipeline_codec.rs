#![cfg(feature = "__libhdf5-1.10")]

use std::path::Path;

use hdf5_pure::{File, FilterId, MessageType};
use hdf5_pure_format::__private::{AccessMode, FilterDescription, FilterPipeline, ObjectHeader};
use libhdf5::filters::Filter as NativeFilter;
use libhdf5::plist::DatasetCreate;
use libhdf5_sys::h5p;
use libhdf5_sys::h5z::H5Z_FILTER_DEFLATE;
use rstest::rstest;
use test_util::filter_pipeline::{Pipeline, Version};
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;

#[rstest]
#[case::version_one(file::libhdf5_create_earliest, Version::One)]
#[case::version_two(file::libhdf5_create_v18, Version::Two)]
fn pipeline_decoding_preserves_native_filter_definition_flags(
    #[case] create: fn(&Path) -> libhdf5::File,
    #[case] version: Version,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("filter-flags.h5");
    let properties = DatasetCreate::build().chunk([1]).finish().unwrap();
    let level = [6];
    let status = libhdf5::sync::sync(|| {
        // SAFETY: The native lock serializes access, the property list is live, and
        // level holds its one advertised client value for the duration of the call.
        unsafe {
            h5p::H5Pset_filter(
                properties.id(),
                H5Z_FILTER_DEFLATE,
                u32::from(OPAQUE_DEFINITION_FLAGS),
                level.len(),
                level.as_ptr(),
            )
        }
    });
    assert_eq!(status, 0);
    write_dataset(create(&path), &properties);

    let mut fixture = Pipeline::new(version);
    let name = (version == Version::One).then_some("deflate");
    fixture
        .push(DEFLATE_ID, name, OPAQUE_DEFINITION_FLAGS, &level)
        .unwrap();
    let expected = fixture.build();
    let body = native_pipeline_body(&path);
    assert_eq!(&body[..expected.len()], expected);
    assert_eq!(
        FilterPipeline::parse(&body).unwrap(),
        FilterPipeline {
            version: version.byte(),
            filters: vec![FilterDescription {
                filter_id: DEFLATE_ID,
                name: name.map(str::to_owned),
                flags: u16::from(OPAQUE_DEFINITION_FLAGS),
                client_data: level.to_vec(),
            }],
        },
    );

    let native = libhdf5::File::open(&path).unwrap();
    let dataset = native.dataset("filtered").unwrap();
    assert_eq!(
        dataset.dcpl().unwrap().get_filters().unwrap(),
        [NativeFilter::Deflate(6)]
    );
    assert_eq!(dataset.read_raw::<u32>().unwrap(), [VALUE]);
    drop(dataset);
    native.close().unwrap();

    let pure = File::open(&path).unwrap();
    let dataset = pure.dataset("filtered").unwrap();
    let filters = dataset.filter_pipeline().unwrap();
    let [filter] = filters.as_slice() else {
        panic!("expected one filter, got {filters:?}");
    };
    assert_eq!(filter.id(), FilterId::Deflate);
    assert_eq!(filter.name(), name);
    assert_eq!(filter.flags(), u16::from(OPAQUE_DEFINITION_FLAGS));
    assert!(!filter.is_optional());
    assert_eq!(filter.client_data(), level);
    assert_eq!(dataset.read_u32().unwrap(), [VALUE]);
    drop(dataset);
    pure.close().unwrap();
    dir.close().unwrap();
}

#[test]
fn native_version_two_custom_filter_names_are_unpadded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("filter-name.h5");
    let properties = DatasetCreate::build().chunk([1]).lzf().finish().unwrap();
    write_dataset(file::libhdf5_create_v18(&path), &properties);

    let body = native_pipeline_body(&path);
    assert_eq!(&body[..2], [2, 1]);
    assert_eq!(u16::from_le_bytes(body[4..6].try_into().unwrap()), 4);
    assert_eq!(&body[10..14], b"lzf\0");

    let pure = File::open(&path).unwrap();
    let dataset = pure.dataset("filtered").unwrap();
    let filters = dataset.filter_pipeline().unwrap();
    let [filter] = filters.as_slice() else {
        panic!("expected one filter, got {filters:?}");
    };
    assert_eq!(filter.id(), FilterId::Lzf);
    assert_eq!(filter.name(), Some("lzf"));
    assert_eq!(dataset.read_u32().unwrap(), [VALUE]);
    drop(dataset);
    pure.close().unwrap();
    dir.close().unwrap();
}

fn write_dataset(file: libhdf5::File, properties: &DatasetCreate) {
    let dataset = file
        .new_dataset::<u32>()
        .set_dcpl(properties)
        .shape([1])
        .create("filtered")
        .unwrap();
    dataset.write(&[VALUE]).unwrap();
    drop(dataset);
    file.close().unwrap();
}

fn native_pipeline_body(path: &Path) -> Vec<u8> {
    let native = libhdf5::File::open(path).unwrap();
    let at = usize::try_from(c_object::header_address(&native, "filtered")).unwrap();
    native.close().unwrap();
    let pure = File::open(path).unwrap();
    let superblock = pure.superblock();
    let bytes = std::fs::read(path).unwrap();
    let header = ObjectHeader::parse(
        &bytes,
        AccessMode::ReadOnly,
        at,
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    pure.close().unwrap();
    let mut messages = header
        .messages
        .into_iter()
        .filter(|message| message.msg_type == MessageType::FILTER_PIPELINE);
    let message = messages.next().unwrap();
    assert!(messages.next().is_none());
    message.data
}

const VALUE: u32 = 0x1234_5678;
const OPAQUE_DEFINITION_FLAGS: u8 = 0xfe;
const DEFLATE_ID: u16 = 1;
