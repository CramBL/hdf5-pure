use super::super::*;

#[test]
fn a_layout_reads_as_one_line() {
    assert_eq!(
        Layout::Compact { size: 40 }.to_string(),
        "compact (40 bytes)"
    );
    assert_eq!(
        Layout::Contiguous {
            address: Some(0x2a0),
            size: 128,
        }
        .to_string(),
        "contiguous (128 bytes at 0x2a0)"
    );
    assert_eq!(
        Layout::Chunked {
            chunk_shape: vec![4, 8],
            index: ChunkIndex::ExtensibleArray,
        }
        .to_string(),
        "chunked (4x8, extensible array index)"
    );
}

#[test]
fn an_unallocated_contiguous_dataset_says_so() {
    let layout = Layout::Contiguous {
        address: None,
        size: 64,
    };
    let shown = layout.to_string();
    assert_eq!(shown, "contiguous (64 bytes, unallocated)");
    assert!(!shown.contains("None"));
}

#[test]
fn a_filter_is_named_by_its_identifier_when_the_file_records_none() {
    let deflate = Filter::from_description(FilterDescription {
        filter_id: 1,
        name: None,
        flags: 0,
        client_data: vec![6],
    })
    .unwrap();
    assert_eq!(deflate.to_string(), "deflate(6)");

    let lzf = Filter::from_description(FilterDescription {
        filter_id: 32000,
        name: None,
        flags: 0,
        client_data: vec![],
    })
    .unwrap();
    assert_eq!(lzf.to_string(), "lzf");
}

#[test]
fn an_unregistered_filter_falls_back_to_its_recorded_name_then_its_id() {
    let named = Filter::from_description(FilterDescription {
        filter_id: 40000,
        name: Some("custom".into()),
        flags: crate::filter_pipeline::H5Z_FLAG_OPTIONAL,
        client_data: vec![],
    })
    .unwrap();
    assert_eq!(named.to_string(), "custom(id=40000) [optional]");

    let anonymous = Filter::from_description(FilterDescription {
        filter_id: 40001,
        name: None,
        flags: 0,
        client_data: vec![],
    })
    .unwrap();
    assert_eq!(anonymous.to_string(), "filter(id=40001)");
}

/// The file records this name, so it cannot reach a message unescaped.
#[test]
fn a_recorded_filter_name_cannot_carry_a_control_character() {
    let hostile = Filter::from_description(FilterDescription {
        filter_id: 40000,
        name: Some("evil\u{1b}[31m\nname".into()),
        flags: 0,
        client_data: vec![],
    })
    .unwrap();
    let shown = hostile.to_string();
    assert!(!shown.chars().any(char::is_control), "{shown}");
    assert_eq!(shown, "evil\\u{1b}[31m\\nname(id=40000)");
}

#[rstest::rstest]
#[case::empty(0, 0, false, vec![], 0)]
#[case::one_skipped(1, 1, false, vec![FilterDisposition::Skipped], 0)]
#[case::second_skipped(2, 2, false, vec![FilterDisposition::Applied, FilterDisposition::Skipped], 0)]
#[case::high_bit(2, 0x8000_0002, false, vec![FilterDisposition::Applied, FilterDisposition::Skipped], 0x8000_0000)]
#[case::raw_bypass(2, 0, true, vec![FilterDisposition::BypassedRaw; 2], 0)]
#[case::raw_with_mask(2, 3, true, vec![FilterDisposition::BypassedRaw; 2], 0)]
#[case::full(32, u32::MAX, false, vec![FilterDisposition::Skipped; 32], 0)]
fn a_chunk_retains_its_pipeline_and_interprets_only_its_positions(
    #[case] count: usize,
    #[case] bits: u32,
    #[case] bypassed: bool,
    #[case] expected: Vec<FilterDisposition>,
    #[case] uninterpreted: u32,
) {
    let filters: Arc<[Filter]> = (0..count)
        .map(|index| {
            Filter::from_description(FilterDescription {
                filter_id: 40000 + u16::try_from(index).unwrap(),
                name: Some(format!("custom{index}")),
                flags: crate::filter_pipeline::H5Z_FLAG_OPTIONAL,
                client_data: vec![index as u32, 17],
            })
            .unwrap()
        })
        .collect();
    let chunk = Chunk::new(Arc::clone(&filters), vec![0], 42, 16, bits, bypassed).unwrap();
    assert!(Arc::ptr_eq(&chunk.filters, &filters));
    let view = chunk.filters();
    assert_eq!(chunk.filter_mask().raw_bits(), bits);
    assert_eq!(view.uninterpreted_bits(), uninterpreted);
    assert_eq!(view.bypasses_pipeline(), bypassed);
    assert_eq!(
        view.dispositions()
            .map(|(_, disposition)| disposition)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(view.filter(count), None);
    assert_eq!(view.filter(32), None);
    assert_eq!(view.filter(usize::MAX), None);
    for (index, (filter, disposition)) in view.dispositions().enumerate() {
        assert_eq!(filter, &filters[index]);
        assert_eq!(view.filter(index), Some((filter, disposition)));
    }
}

#[cfg(feature = "zfp")]
#[test]
fn the_zfp_name_is_reached_through_its_own_identifier() {
    assert_eq!(FilterId::Zfp.name(), Some("zfp"));
}

/// The identifier is labeled, so it cannot read as one of the client-data
/// values it sits beside.
#[test]
fn an_unregistered_filter_keeps_its_id_apart_from_its_client_data() {
    let named = Filter::from_description(FilterDescription {
        filter_id: 40000,
        name: Some("custom".into()),
        flags: 0,
        client_data: vec![7, 8],
    })
    .unwrap();
    assert_eq!(named.to_string(), "custom(id=40000, 7, 8)");
}
