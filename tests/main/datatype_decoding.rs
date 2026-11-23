use hdf5_pure::{Error, File, FormatError};
use test_util::datatype::leaf::{Leaf, Version};
use test_util::datatype::{self, Class};
use test_util::image::Image;
use test_util::object_header::{Message, MessageType};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util::{data_layout, dataspace};

#[test]
fn dataset_datatype_rejects_message_version_fifteen() {
    let mut datatype = Leaf::Integer.bytes(Version::One);
    datatype[0] = 0xf0;
    let file = dataset_file(
        &datatype,
        &dataspace::VersionTwo::scalar().build(),
        &data_layout::compact(&[0; 4]),
    );
    let dataset = file.dataset("value").unwrap();
    let error = dataset.datatype().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected an unsupported datatype version, got {error:?}");
    };
    assert_eq!(error, FormatError::InvalidDatatypeMessageVersion(15));
}

#[test]
fn chunked_dataset_rejects_zero_width_string_elements() {
    let datatype = datatype::header(Class::STRING, Version::One.get(), [0; 3], 0);
    let file = dataset_file(
        &datatype,
        &dataspace::VersionTwo::simple(Widths::EIGHT, &[4], None).build(),
        &data_layout::chunked(u64::MAX, &[2], 1),
    );
    let dataset = file.dataset("value").unwrap();
    assert_eq!(dataset.shape().unwrap(), [4]);
    for (name, result) in [
        ("element_size", dataset.element_size().map(|_| ())),
        ("dtype", dataset.dtype().map(|_| ())),
        ("read_raw", dataset.read_raw().map(|_| ())),
        ("read_string", dataset.read_string().map(|_| ())),
    ] {
        let error = result.unwrap_err();
        let Error::Format(error) = error else {
            panic!("{name}: expected a datatype size error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::ZeroSizedDatatype {
                class: Class::STRING.0
            },
            "{name}"
        );
    }
}

fn dataset_file(datatype: &[u8], dataspace: &[u8], layout: &[u8]) -> File {
    let image = Image::starting_with(&vec![0; Superblock::new(Widths::EIGHT).build().len()]);
    let bytes = image.build_with_dataset(
        "value",
        [
            Message::new(MessageType::DATASPACE, dataspace),
            Message::new(MessageType::DATATYPE, datatype),
            Message::new(MessageType::DATA_LAYOUT, layout),
        ],
    );
    File::from_bytes(bytes).unwrap()
}
