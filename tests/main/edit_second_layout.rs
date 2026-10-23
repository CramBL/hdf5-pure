use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::Layout;
use rstest::rstest;
use test_util::bytes;
use test_util::data_layout;
use test_util::data_layout::UNDEFINED_ADDRESS;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::temp;

#[rstest]
#[case::contiguous_then_compact(None, data_layout::compact(&[0xEE; 4]))]
#[case::contiguous_then_chunked(None, data_layout::chunked(UNDEFINED_ADDRESS, &[ELEMENTS], 1))]
fn an_overwrite_writes_through_the_first_layout(
    #[case] first_layout: Option<Vec<u8>>,
    #[case] second_layout: Vec<u8>,
) {
    let path = temp::temp_path("second_layout.h5");
    write_two_layouts(&path, first_layout.as_deref(), &second_layout);

    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("d")
            .unwrap()
            .write_staged(|dataset| {
                dataset.with_u8_data(&NEW_VALUES);
            })
            .unwrap();
        file.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let layout = dataset.layout().unwrap();
    assert!(
        matches!(layout, Layout::Contiguous { size, .. } if size == u64::from(ELEMENTS)),
        "{layout:?}"
    );
    assert_eq!(dataset.read_u8().unwrap(), NEW_VALUES);
}

/// Writes a file to `path` whose dataset `d` holds `first_layout` as its first Data Layout
/// message, or the contiguous layout [`FileBuilder`] writes where `first_layout` is `None`, and
/// `second_layout` as a second one in place of its attribute.
fn write_two_layouts(path: &Path, first_layout: Option<&[u8]>, second_layout: &[u8]) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_u8_data(&[1; ELEMENTS as usize])
        .set_attr("a", AttrValue::I32Array(vec![0; 8]));
    let mut image = builder.finish().unwrap();
    let header_at = *bytes::signature_offsets(&image, v2::SIGNATURE)
        .last()
        .expect("the dataset's object header follows the root group's");
    if let Some(first_layout) = first_layout {
        v2::replace_message(
            &mut image,
            header_at,
            MessageType::DATA_LAYOUT,
            first_layout,
        );
    }
    v2::replace_record(
        &mut image,
        header_at,
        MessageType::ATTRIBUTE,
        Message::new(MessageType::DATA_LAYOUT, second_layout),
    );
    std::fs::write(path, image).unwrap();
}

const ELEMENTS: u32 = 14;
const NEW_VALUES: [u8; ELEMENTS as usize] = [2; ELEMENTS as usize];
