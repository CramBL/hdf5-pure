#![cfg(feature = "__hdf5-1.10")]

use std::fs;

use hdf5::File as CFile;
use hdf5::plist::GroupCreate;
use hdf5::plist::file_access::LibraryVersion;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure_format::__private::FilterDescription;
use hdf5_pure_format::__private::FilterPipeline;
use hdf5_pure_format::__private::FractalHeapFiltering;
use hdf5_pure_format::__private::FractalHeapHeader;
use rstest::rstest;
use tempfile::TempDir;
use test_util_hdf5::group_filter;

#[rstest]
#[case::earliest(LibraryVersion::V18)]
#[case::latest(LibraryVersion::latest())]
fn a_c_written_filtered_header_retains_the_root_fields_and_pipeline(
    #[case] bounds: LibraryVersion,
) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("filtered.h5");
    let gcpl = GroupCreate::build()
        .obj_track_times(false)
        .finish()
        .unwrap();
    assert_eq!(group_filter::set_deflate(&gcpl, 6), 0);
    let file = CFile::with_options()
        .with_fapl(|p| p.libver_bounds(bounds, bounds))
        .create(&path)
        .unwrap();
    let group = file
        .create_group_builder()
        .set_gcpl(&gcpl)
        .create("g")
        .unwrap();
    for i in 0..12 {
        group.create_group(&format!("child_{i:02}")).unwrap();
    }
    drop(group);
    file.close().unwrap();
    let expected_names: Vec<_> = (0..12).map(|i| format!("child_{i:02}")).collect();
    let bytes = fs::read(&path).unwrap();
    let header_offset = bytes
        .windows(FRACTAL_HEAP_SIGNATURE.len())
        .position(|window| window == FRACTAL_HEAP_SIGNATURE)
        .unwrap();
    let header = FractalHeapHeader::parse(&bytes, header_offset, 8, 8).unwrap();
    let FractalHeapFiltering::Filtered {
        root_direct_block_size,
        root_filter_mask,
        pipeline,
    } = &header.filtering
    else {
        panic!("expected filtered heap, got {:?}", header.filtering);
    };
    assert!(*root_direct_block_size > 0);
    assert!(*root_direct_block_size < header.starting_block_size);
    assert_eq!(*root_filter_mask, 0);
    assert_eq!(
        *pipeline,
        FilterPipeline {
            version: PIPELINE_VERSION_TWO,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: FILTER_OPTIONAL,
                client_data: vec![6],
            }],
        }
    );
    assert_eq!(
        FractalHeapHeader::parse_from_source(
            bytes.as_slice(),
            u64::try_from(header_offset).unwrap(),
            8,
            8
        ),
        Ok(header)
    );
    let c_file = CFile::open(&path).unwrap();
    let mut names = c_file.group("g").unwrap().member_names().unwrap();
    names.sort();
    assert_eq!(names, expected_names);
    c_file.close().unwrap();
    let buffered = File::open(&path)
        .unwrap()
        .group("g")
        .unwrap()
        .groups()
        .unwrap_err();
    assert!(
        matches!(
            buffered,
            Error::Format(FormatError::UnsupportedFilteredHeapObject)
        ),
        "{buffered:?}"
    );
    let streaming = File::open_streaming(&path)
        .unwrap()
        .group("g")
        .unwrap()
        .groups()
        .unwrap_err();
    assert!(
        matches!(
            streaming,
            Error::Format(FormatError::UnsupportedFilteredHeapObject)
        ),
        "{streaming:?}"
    );
}

// "Fractal Heap", format specification version 4.0, defines the header signature.
const FRACTAL_HEAP_SIGNATURE: &[u8; 4] = b"FRHP";
// "The Data Storage - Filter Pipeline Message", format specification version 4.0, defines the
// versions and filter fields.
const PIPELINE_VERSION_TWO: u8 = 2;
const FILTER_DEFLATE: u16 = 1;
const FILTER_OPTIONAL: u16 = 0x0001;
