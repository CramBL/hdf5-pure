#![cfg(feature = "__hdf5-1.10")]
use hdf5::dataset::Layout as CLayout;
use hdf5_pure::File;
use rstest::rstest;
use test_util::bytes;
use test_util::data_layout;
use test_util::data_layout::UNDEFINED_ADDRESS;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util_hdf5::file;

#[rstest]
#[case::contiguous_then_compact(CLayout::Contiguous, data_layout::compact(&[0xEE; 4]))]
#[case::contiguous_then_chunked(
    CLayout::Contiguous,
    data_layout::chunked(UNDEFINED_ADDRESS, &[ELEMENTS], 1),
)]
fn libhdf5_reads_an_overwrite_through_the_first_layout(
    #[case] layout: CLayout,
    #[case] second_layout: Vec<u8>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("second_layout.h5");
    {
        let file = file::libhdf5_create_v18(&path);
        let dataset = file
            .new_dataset::<u8>()
            .shape([ELEMENTS as usize])
            .layout(layout)
            .create("d")
            .unwrap();
        dataset.write(&[1; ELEMENTS as usize]).unwrap();
        dataset
            .new_attr::<i32>()
            .shape([8])
            .create("a")
            .unwrap()
            .write(&[0; 8])
            .unwrap();
        file.close().unwrap();
    }
    let mut image = std::fs::read(&path).unwrap();
    let header_at = *bytes::signature_offsets(&image, v2::SIGNATURE)
        .last()
        .expect("the dataset's object header follows the root group's");
    v2::replace_record(
        &mut image,
        header_at,
        MessageType::ATTRIBUTE,
        Message::new(MessageType::DATA_LAYOUT, &second_layout),
    );
    std::fs::write(&path, image).unwrap();

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

    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.layout(), layout);
    assert_eq!(dataset.read_raw::<u8>().unwrap(), NEW_VALUES);
}

const ELEMENTS: u32 = 4;
const NEW_VALUES: [u8; ELEMENTS as usize] = [2; ELEMENTS as usize];
