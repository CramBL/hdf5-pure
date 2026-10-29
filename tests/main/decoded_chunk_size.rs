use std::io::Cursor;

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FillTime;
use hdf5_pure::FormatError;
use hdf5_pure::ReadSeekSource;
use hdf5_pure_filter::__private::FILTER_SHUFFLE;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::FillValueMessage;
use hdf5_pure_format::__private::FilterDescription;
use hdf5_pure_format::__private::FilterPipeline;
use rstest::rstest;
use test_util::btree_v1;
use test_util::data_layout;
use test_util::dataspace;
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::reader::ReadBackend;

#[test]
fn a_present_short_chunk_returns_a_size_mismatch() {
    let file = File::from_bytes(chunk_file(Some(&7_i32.to_le_bytes()), 2, false, 0)).unwrap();
    let read = file.dataset("d").unwrap().read_i32();
    let Err(Error::Format(FormatError::DataSizeMismatch { expected, actual })) = read else {
        panic!("expected DataSizeMismatch, got {read:?}");
    };
    assert_eq!((expected, actual), (8, 4));
}

#[rstest]
#[case::bytes(Backend::File(ReadBackend::Bytes))]
#[case::cursor(Backend::Cursor)]
#[case::streaming(Backend::File(ReadBackend::Streaming))]
fn malformed_chunks_are_rejected_before_cache_admission(
    #[values(vec![7; 4], vec![7; 12])] stored: Vec<u8>,
    #[case] backend: Backend,
    #[values(Read::Whole, Read::Rows, Read::Region)] read: Read,
    #[values(false, true)] skipped: bool,
) {
    let path = temp::temp_path("chunk.h5");
    let bytes = chunk_file(Some(&stored), 2, skipped, 17);
    std::fs::write(&path, &bytes).unwrap();
    let file = match backend {
        Backend::Cursor => {
            File::from_source(ReadSeekSource::new(Cursor::new(bytes)).unwrap()).unwrap()
        }
        Backend::File(backend) => backend.open(&path).unwrap(),
    };
    let dataset = file.dataset("d").unwrap();
    assert_eq!(
        dataset.chunks().unwrap()[0].storage_size,
        u64::try_from(stored.len()).unwrap()
    );
    let result = || match read {
        Read::Whole => dataset.read_i32(),
        Read::Rows => dataset.read_i32_rows(0, 1),
        Read::Region => dataset.read_i32_region(&[0], &[1]),
    };
    let expected = FormatError::DataSizeMismatch {
        expected: 8,
        actual: stored.len(),
    };
    let error = result().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
    assert_eq!(dataset.chunk_cache_stats().cached_chunks(), 0);
    assert_eq!(dataset.chunk_cache_stats().cached_bytes(), 0);
    let misses = dataset.chunk_cache_stats().misses();
    let error = result().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, expected);
    assert_eq!(dataset.chunk_cache_stats().cached_chunks(), 0);
    assert_eq!(dataset.chunk_cache_stats().cached_bytes(), 0);
    assert_eq!(dataset.chunk_cache_stats().hits(), 0);
    assert!(dataset.chunk_cache_stats().misses() > misses);
}

#[rstest]
#[case::complete_chunk(Some(vec![7_i32, 8]), 2, vec![7, 8])]
#[case::truncated_edge(Some(vec![7_i32, 8]), 1, vec![7])]
#[case::missing_final_element(Some(vec![7_i32, 8]), 3, vec![7, 8, 17])]
#[case::absent_chunk(None, 2, vec![17, 17])]
fn complete_edge_chunks_and_absent_chunks_retain_their_values(
    #[case] stored: Option<Vec<i32>>,
    #[case] extent: u64,
    #[case] expected: Vec<i32>,
    #[values(false, true)] skipped: bool,
) {
    let stored = stored.map(|values| {
        values
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>()
    });
    let file = File::from_bytes(chunk_file(stored.as_deref(), extent, skipped, 17)).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), expected);
    let before = dataset.chunk_cache_stats();
    assert_eq!(dataset.read_i32().unwrap(), expected);
    assert_eq!(dataset.read_i32_rows(0, extent).unwrap(), expected);
    assert_eq!(dataset.read_i32_region(&[0], &[extent]).unwrap(), expected);
    assert_eq!(
        dataset.chunk_cache_stats().cached_chunks(),
        usize::from(stored.is_some())
    );
    assert_eq!(
        dataset.chunk_cache_stats().cached_bytes(),
        if stored.is_some() { 8 } else { 0 }
    );
    if stored.is_some() {
        assert!(dataset.chunk_cache_stats().hits() > before.hits());
    }
}

#[derive(Clone, Copy)]
enum Backend {
    Cursor,
    File(ReadBackend),
}

#[derive(Clone, Copy)]
enum Read {
    Region,
    Rows,
    Whole,
}

fn chunk_file(stored: Option<&[u8]>, extent: u64, skipped: bool, fill: i32) -> Vec<u8> {
    let superblock = || {
        Superblock::new(Widths::EIGHT)
            .version(SUPERBLOCK_VERSION)
            .root_header_address(ROOT_HEADER_ADDRESS)
    };
    let mut image = Image::new();
    image.place(0, &superblock().build());
    image.place(usize::try_from(ROOT_HEADER_ADDRESS).unwrap(), &[0; 256]);
    let index = stored.map_or(UNDEFINED_ADDRESS, |stored| {
        let chunk = u64::try_from(image.append(stored)).unwrap();
        u64::try_from(image.append(&btree_v1::node(
            btree_v1::NodeType::CHUNK,
            0,
            &[
                btree_v1::chunk_key(
                    u32::try_from(stored.len()).unwrap(),
                    if skipped { SHUFFLE_FILTER_MASK } else { 0 },
                    &[0, 0],
                ),
                btree_v1::chunk_key(0, 0, &[2, 0]),
            ],
            &[chunk],
            Widths::EIGHT,
        )))
        .unwrap()
    });
    let mut header = Header::new()
        .message(Message::new(
            MessageType::DATATYPE,
            &format::serialize_datatype(&hdf5_pure::make_i32_type()),
        ))
        .message(Message::new(
            MessageType::DATASPACE,
            &dataspace::v2(
                1,
                dataspace::Flags::NONE,
                dataspace::Kind::SIMPLE,
                &[extent],
                None,
            ),
        ))
        .message(Message::new(
            MessageType::DATA_LAYOUT,
            &data_layout::chunked(index, &[2], 4),
        ));
    if skipped {
        let pipeline = FilterPipeline {
            version: FILTER_PIPELINE_VERSION,
            filters: vec![FilterDescription {
                filter_id: FILTER_SHUFFLE,
                name: None,
                flags: 0,
                client_data: vec![4],
            }],
        };
        header = header.message(Message::new(
            MessageType::FILTER_PIPELINE,
            &pipeline.serialize().unwrap(),
        ));
    }
    if fill != 0 {
        header = header.message(Message::new(
            MessageType::FILL_VALUE,
            &FillValueMessage::new(FillTime::IfSet, Some(fill.to_le_bytes().to_vec()))
                .serialize()
                .unwrap(),
        ));
    }
    let dataset = u64::try_from(image.append(&header.build())).unwrap();
    image.place(
        usize::try_from(ROOT_HEADER_ADDRESS).unwrap(),
        &Header::new()
            .message(Message::new(
                MessageType::LINK,
                &HardLink::new("d", dataset).build(Widths::EIGHT),
            ))
            .build(),
    );
    let eof = u64::try_from(image.as_bytes().len()).unwrap();
    image.place(0, &superblock().eof_address(eof).build());
    image.build()
}

const ROOT_HEADER_ADDRESS: u64 = 48;
// "Format Signature and Superblock" in format specification, version 4.0,
// defines superblock version 3.
const SUPERBLOCK_VERSION: u8 = 3;
// "Appendix A: Definitions" in format specification, version 4.0, defines an
// undefined address as all bits set.
const UNDEFINED_ADDRESS: u64 = u64::MAX;
// "The Data Storage - Filter Pipeline Message" in format specification, version 4.0,
// defines pipeline message version 2.
const FILTER_PIPELINE_VERSION: u8 = 2;
// "Version 1 B-trees" in format specification, version 4.0, defines the filter mask
// of a raw data chunk key, with one bit per pipeline entry, starting at bit zero.
const SHUFFLE_FILTER_MASK: u32 = 1;
