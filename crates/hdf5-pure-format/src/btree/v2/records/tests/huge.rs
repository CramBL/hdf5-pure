use alloc::vec;
use alloc::vec::Vec;

use rstest::rstest;

use super::*;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2::BTreeV2Header;
use crate::btree::v2::BTreeV2Record;
use crate::bytes;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[rstest]
#[case::unfiltered_indirect(false, false, 0)]
#[case::filtered_indirect(true, false, 1)]
#[case::unfiltered_direct(false, true, 0)]
#[case::filtered_direct(true, true, 1)]
#[case::empty_pipeline_indirect(true, false, 0)]
#[case::empty_pipeline_direct(true, true, 0)]
fn huge_tracker_records_select_the_heap_encoding(
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[case] filtered: bool,
    #[case] direct: bool,
    #[case] filter_count: usize,
) {
    let mut header = FractalHeapHeader::parse_bytes(
        &test_util::fractal_heap::Header::new(256).build(test_util::widths::Widths::EIGHT),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    )
    .unwrap();
    header.btree_huge_objects_address = StoredAddress::new(0x200);
    header.heap_id_length = if direct { 29 } else { 3 };
    let decoded_size = if filter_count == 0 { 17 } else { 1024 };
    if filtered {
        header.filtering = crate::fractal_heap::FractalHeapFiltering::Filtered {
            root_direct_block_size: 0,
            root_filter_mask: 0,
            pipeline: crate::filter_pipeline::FilterPipeline::new(
                crate::filter_pipeline::VERSION_TWO,
                vec![
                    crate::filter_pipeline::FilterDescription {
                        filter_id: FILTER_DEFLATE,
                        name: None,
                        flags: 0,
                        client_data: vec![],
                    };
                    filter_count
                ],
            )
            .unwrap(),
        };
    }
    let widths = FormatWidths { offsets, lengths };
    let layout = HugeObjectRecordContext::new(&header, widths).unwrap();
    let expected_type = match (filtered, direct) {
        (false, false) => BTREE_V2_HUGE_OBJECT,
        (true, false) => BTREE_V2_FILTERED_HUGE_OBJECT,
        (false, true) => BTREE_V2_HUGE_OBJECT_DIRECT,
        (true, true) => BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT,
    };
    let mut data = Vec::new();
    bytes::write_offset(&mut data, 0x1234, offsets);
    bytes::write_length(&mut data, 17, lengths);
    if filtered {
        data.extend_from_slice(&0u32.to_le_bytes());
        bytes::write_length(&mut data, decoded_size, lengths);
    }
    if !direct {
        bytes::write_length(&mut data, 261, lengths);
    }
    let index = BTreeV2Header {
        tree_type: expected_type,
        node_size: 512,
        record_size: u16::try_from(data.len()).unwrap(),
        depth: 0,
        root_node_address: StoredAddress::new(0x400),
        num_records_in_root: 1,
        total_records: 1,
    };
    assert_eq!(layout.tree_type(), expected_type);
    assert_eq!(usize::from(layout.record_size()), data.len());
    let record = layout.resolve(&index).unwrap().parse(&data).unwrap();
    match record {
        HugeObjectTrackerRecord::Indirect(record) => assert_eq!(
            record,
            HugeObjectRecord {
                address: StoredAddress::new(0x1234),
                length: 17,
                id: 261
            }
        ),
        HugeObjectTrackerRecord::Direct(record) => assert_eq!(
            record,
            HugeObjectDirectRecord {
                address: StoredAddress::new(0x1234),
                length: 17
            }
        ),
        HugeObjectTrackerRecord::FilteredIndirect(record) => {
            assert_eq!(record.id(), 261);
            assert_eq!(record.location().address(), StoredAddress::new(0x1234));
            assert_eq!(record.location().encoded_size(), 17);
            assert_eq!(record.location().filter_mask(), 0);
            assert_eq!(record.location().decoded_size(), decoded_size);
        }
        HugeObjectTrackerRecord::FilteredDirect(record) => {
            assert_eq!(record.location().address(), StoredAddress::new(0x1234));
            assert_eq!(record.location().encoded_size(), 17);
            assert_eq!(record.location().filter_mask(), 0);
            assert_eq!(record.location().decoded_size(), decoded_size);
        }
    }
}

#[rstest]
#[case::indirect_wrong_type(false, 1, 36)]
#[case::indirect_short(false, 2, 35)]
#[case::indirect_long(false, 2, 37)]
#[case::direct_wrong_type(true, 3, 28)]
#[case::direct_short(true, 4, 27)]
#[case::direct_long(true, 4, 29)]
fn filtered_huge_record_contexts_reject_wrong_tracker_headers(
    #[case] direct: bool,
    #[case] tree_type: u8,
    #[case] record_size: u16,
) {
    let layout = HugeObjectRecordContext {
        widths: FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        encoding: if direct {
            HugeObjectRecordEncoding::FilteredDirect { filter_count: 1 }
        } else {
            HugeObjectRecordEncoding::FilteredIndirect { filter_count: 1 }
        },
    };
    let header = BTreeV2Header {
        tree_type,
        record_size,
        node_size: 512,
        depth: 0,
        root_node_address: StoredAddress::new(0x400),
        num_records_in_root: 1,
        total_records: 1,
    };
    assert_eq!(
        layout.resolve(&header).unwrap_err(),
        FormatError::UnexpectedHugeObjectBTree {
            tree_type,
            record_size: usize::from(record_size),
            required: if direct { 28 } else { 36 },
        }
    );
}

#[rstest]
#[case::indirect_short(false, 35)]
#[case::indirect_long(false, 37)]
#[case::direct_short(true, 27)]
#[case::direct_long(true, 29)]
fn filtered_huge_records_require_the_exact_record_bytes(
    #[case] direct: bool,
    #[case] length: usize,
) {
    let actual = if direct {
        FilteredHugeObjectDirectRecord::parse(
            &vec![0; length],
            OffsetWidth::Eight,
            LengthWidth::Eight,
            1,
        )
        .unwrap_err()
    } else {
        FilteredHugeObjectRecord::parse(&vec![0; length], OffsetWidth::Eight, LengthWidth::Eight, 1)
            .unwrap_err()
    };
    assert_eq!(
        actual,
        FormatError::UnexpectedHugeObjectBTree {
            tree_type: if direct {
                BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT
            } else {
                BTREE_V2_FILTERED_HUGE_OBJECT
            },
            record_size: length,
            required: if direct { 28 } else { 36 },
        }
    );
}

#[rstest]
fn an_encoded_huge_object_record_decodes_to_its_fields(
    #[values(OffsetWidth::Four, OffsetWidth::Eight)] offset_width: OffsetWidth,
    #[values(LengthWidth::Four, LengthWidth::Eight)] length_width: LengthWidth,
) {
    let record = HugeObjectRecord {
        address: StoredAddress::new(0x1234),
        length: 700,
        id: 3,
    };
    let mut data = Vec::new();
    record
        .encode(&mut data, offset_width, length_width)
        .unwrap();

    assert_eq!(
        data.len(),
        usize::from(HugeObjectRecord::size(offset_width, length_width))
    );
    let [os, ls] = [offset_width.get(), length_width.get()].map(usize::from);
    let mut expected = 0x1234u64.to_le_bytes()[..os].to_vec();
    expected.extend_from_slice(&700u64.to_le_bytes()[..ls]);
    expected.extend_from_slice(&3u64.to_le_bytes()[..ls]);
    assert_eq!(data, expected);
    assert_eq!(
        BTreeV2Record { data }.huge_object(offset_width, length_width),
        Ok(record)
    );
}

#[rstest]
fn a_direct_huge_object_record_decodes_to_its_fields(
    #[values(OffsetWidth::Four, OffsetWidth::Eight)] offset_width: OffsetWidth,
    #[values(LengthWidth::Four, LengthWidth::Eight)] length_width: LengthWidth,
) {
    let [os, ls] = [offset_width.get(), length_width.get()].map(usize::from);
    let mut data = 0x1234u64.to_le_bytes()[..os].to_vec();
    data.extend_from_slice(&700u64.to_le_bytes()[..ls]);

    assert_eq!(
        data.len(),
        usize::from(HugeObjectDirectRecord::size(offset_width, length_width))
    );
    assert_eq!(
        BTreeV2Record { data }.huge_object_direct(offset_width, length_width),
        Ok(HugeObjectDirectRecord {
            address: StoredAddress::new(0x1234),
            length: 700,
        })
    );
}

#[rstest]
#[case::exact(65_534, 65_535, Ok(vec![0xa5, 0xfe, 0xff, 0xff, 0xff]))]
#[case::undefined(65_535, 1, Err(FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 }))]
#[case::address(65_536, 1, Err(FormatError::AddressTooLarge { address: 65_536, offset_size: 2 }))]
#[case::length(1, 65_536, Err(FormatError::LengthTooLarge { length: 65_536, length_size: 2 }))]
fn direct_huge_record_fields_are_checked_before_appending(
    #[case] address: u64,
    #[case] length: u64,
    #[case] expected: Result<Vec<u8>, FormatError>,
) {
    let record = HugeObjectDirectRecord {
        address: StoredAddress::new(address),
        length,
    };
    let mut bytes = vec![0xa5];
    let result = record.encode(&mut bytes, OffsetWidth::Two, LengthWidth::Two);
    if result.is_err() {
        assert_eq!(bytes, [0xa5]);
    }
    assert_eq!(result.map(|()| bytes), expected);
}

#[test]
fn a_huge_object_record_shorter_than_its_fields_is_unexpected_eof() {
    let record = BTreeV2Record { data: vec![0; 20] };
    assert_eq!(
        record.huge_object(OffsetWidth::Eight, LengthWidth::Eight),
        Err(FormatError::UnexpectedEof {
            expected: 24,
            available: 20,
        })
    );
}
/// The deflate filter identifier from "The Data Storage - Filter Pipeline Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
const FILTER_DEFLATE: u16 = 1;
