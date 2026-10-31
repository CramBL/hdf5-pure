use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::{BaseAddress, FormatError, StoredAddress};
use hdf5_pure_format::__private::{
    FILTER_PIPELINE_VERSION_TWO as VERSION_TWO, FilterDescription, FilterPipeline, FormatWidths,
    FractalHeapFiltering, FractalHeapHeader, FractalHeapHugeObjectFilters, FractalHeapHugeObjectId,
    FractalHeapHugeObjectLayout, FractalHeapHugeObjectRef, FractalHeapHugeObjectTracker,
    FractalHeapIdLayout, HugeIndirectKeyEncoding, LengthWidth, OffsetWidth,
};
use rstest::rstest;
use test_util::widths::Widths;

#[rstest]
fn huge_references_preserve_physical_allocations_and_filter_context(
    #[values(false, true)] filtered: bool,
    #[values(false, true)] direct: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[values(0, 512, u64::from(u32::MAX) + 512)] base: u64,
) {
    let fixture = Fixture::new(
        filtered,
        direct,
        base,
        1024,
        FormatWidths { offsets, lengths },
    );
    let reference = fixture.resolve();
    let tracker = fixture.tracker();
    let record = tracker.parse_record(&fixture.record).unwrap();
    assert_eq!(record.key(), if direct { None } else { Some(5) });
    assert_eq!(record.reference().range(), reference.range());
    assert_eq!(record.reference().decoded_size(), reference.decoded_size());
    assert_eq!(record.reference().filters(), reference.filters());
    assert_eq!(
        reference.range(),
        FileRange::new(PhysicalAddress::new(base + 1024), 3).unwrap()
    );
    assert_eq!(reference.parse_range(), reference.range());
    assert_eq!(reference.decoded_size(), if filtered { 9 } else { 3 });
    assert_eq!(fixture.layout.tracker_range().start().get(), base + 256);
    let encoded = reference
        .parse(FileRegion::new(reference.range(), b"abc").unwrap())
        .unwrap();
    assert_eq!(encoded, b"abc");
    assert_eq!(encoded.as_ptr(), b"abc".as_ptr());
    if filtered {
        let filters = reference.filters().unwrap();
        assert_eq!(filters.mask, 0);
        assert_eq!(filters.pipeline, &fixture.pipeline);
    } else {
        assert_eq!(reference.filters(), None);
    }
    let decoded = vec![7; usize::try_from(reference.decoded_size()).unwrap()];
    let view = fixture
        .layout
        .associate_decoded(&reference, &decoded)
        .unwrap();
    assert_eq!(view.bytes(), decoded);
    assert_eq!(view.bytes().as_ptr(), decoded.as_ptr());
    assert_eq!(view.reference().range(), reference.range());
}

#[rstest]
fn huge_references_preserve_selectively_skipped_filters(#[values(false, true)] direct: bool) {
    let mut fixture = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
    let FractalHeapFiltering::Filtered { pipeline, .. } = &mut fixture.header.filtering else {
        panic!("expected filtered fixture")
    };
    let mut second = pipeline.filters[0].clone();
    second.filter_id = 32_002;
    pipeline.filters.push(second);
    let expected = pipeline.clone();
    fixture.layout = FractalHeapHugeObjectLayout::new(
        &fixture.header,
        fixture.widths,
        BaseAddress::ZERO,
        HugeIndirectKeyEncoding::Libhdf5,
    )
    .unwrap();
    let mask = 1u32.to_le_bytes();
    let mask_at =
        usize::from(fixture.widths.offsets.get()) + usize::from(fixture.widths.lengths.get());
    if direct {
        fixture.id[mask_at + 1..mask_at + 1 + mask.len()].copy_from_slice(&mask);
    } else {
        fixture.record[mask_at..mask_at + mask.len()].copy_from_slice(&mask);
    }
    let reference = fixture.resolve();
    assert_eq!(
        reference.filters().unwrap(),
        FractalHeapHugeObjectFilters {
            pipeline: &expected,
            mask: 1
        }
    );
    assert_eq!(reference.decoded_size(), 9);
}

#[rstest]
#[case::short(512, 2)]
#[case::before(511, 3)]
#[case::after(513, 3)]
fn huge_payloads_require_their_encoded_region(#[case] start: u64, #[case] length: usize) {
    let fixture = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
    let reference = fixture.resolve();
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(start), length as u64).unwrap(),
        &b"abc"[..length],
    )
    .unwrap();
    assert_eq!(
        reference.parse(region).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: 512,
            required_length: 3,
            provided_start: start,
            provided_length: length as u64,
        }
    );
}

#[rstest]
fn huge_trackers_require_their_complete_header(#[values(0, 1)] missing: usize) {
    let fixture = Fixture::new(true, false, 512, 1024, EIGHT_WIDTHS);
    let range = fixture.layout.tracker_range();
    let length = fixture.tracker.len() - missing;
    let region = FileRegion::new(
        FileRange::new(range.start(), length as u64).unwrap(),
        &fixture.tracker[..length],
    )
    .unwrap();
    if missing == 0 {
        assert_eq!(
            fixture
                .layout
                .parse_tracker(region)
                .unwrap()
                .header()
                .total_records,
            1
        );
    } else {
        assert_eq!(
            fixture.layout.parse_tracker(region).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: range.start().get(),
                required_length: range.length(),
                provided_start: range.start().get(),
                provided_length: length as u64,
            }
        );
    }
}

#[rstest]
#[case::short(8)]
#[case::long(10)]
fn huge_decoded_payloads_require_the_declared_size(#[case] length: usize) {
    let fixture = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
    assert_eq!(
        fixture
            .layout
            .associate_decoded(&fixture.resolve(), &vec![0; length])
            .unwrap_err(),
        FormatError::DecodedSizeMismatch {
            expected: 9,
            actual: length as u64
        }
    );
}

#[rstest]
fn huge_references_reject_independent_layouts(#[values(false, true)] direct: bool) {
    let fixture = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
    let other = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
    let reference = fixture.resolve();
    assert_eq!(
        other
            .layout
            .associate_decoded(&reference, &[0; 9])
            .unwrap_err(),
        FormatError::WrongLayout
    );
    if !direct {
        let FractalHeapHugeObjectId::Indirect(lookup) =
            fixture.layout.parse_id(&fixture.id).unwrap()
        else {
            panic!("expected indirect ID")
        };
        let record = fixture.tracker().parse_record(&fixture.record).unwrap();
        let other_record = other.tracker().parse_record(&other.record).unwrap();
        assert_eq!(
            other.layout.resolve(&lookup, &other_record).unwrap_err(),
            FormatError::WrongLayout
        );
        assert_eq!(
            fixture.layout.resolve(&lookup, &other_record).unwrap_err(),
            FormatError::WrongLayout
        );
        assert_eq!(
            fixture.layout.resolve(&lookup, &record).unwrap().range(),
            reference.range()
        );
    }
}

#[test]
fn huge_layout_clones_preserve_identity_after_the_original_is_dropped() {
    let fixture = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
    let reference = fixture.resolve();
    let layout = fixture.layout.clone();
    let pipeline = fixture.pipeline.clone();
    drop(fixture);
    assert_eq!(reference.filters().unwrap().pipeline, &pipeline);
    let view = layout.associate_decoded(&reference, &[0; 9]).unwrap();
    drop(layout);
    drop(reference);
    assert_eq!(view.bytes(), &[0; 9]);
    assert_eq!(view.reference().filters().unwrap().pipeline, &pipeline);
}

#[test]
fn huge_lookup_compares_the_complete_numeric_key() {
    let mut fixture = Fixture::new(
        true,
        false,
        0,
        512,
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Two,
        },
    );
    fixture.header.heap_id_length = 8;
    fixture.layout = FractalHeapHugeObjectLayout::new(
        &fixture.header,
        fixture.widths,
        BaseAddress::ZERO,
        HugeIndirectKeyEncoding::Libhdf5,
    )
    .unwrap();
    fixture.id = vec![HUGE_ID_FIRST_BYTE, 5, 1, 0, 0, 0, 0, 0];
    let FractalHeapHugeObjectId::Indirect(lookup) = fixture.layout.parse_id(&fixture.id).unwrap()
    else {
        panic!("expected indirect ID")
    };
    let record = fixture.tracker().parse_record(&fixture.record).unwrap();
    assert_eq!(lookup.key().encoding(), HugeIndirectKeyEncoding::Libhdf5);
    assert_eq!(
        fixture.layout.resolve(&lookup, &record).unwrap_err(),
        FormatError::HugeObjectNotFound(261)
    );
    fixture.record.truncate(fixture.record.len() - 2);
    fixture.record.extend_from_slice(&261u16.to_le_bytes());
    let record = fixture.tracker().parse_record(&fixture.record).unwrap();
    assert_eq!(record.key(), Some(261));
    assert_eq!(
        fixture
            .layout
            .resolve(&lookup, &record)
            .unwrap()
            .range()
            .start()
            .get(),
        512
    );
}

#[rstest]
#[case::address(u64::MAX - 512, 1024, FormatError::OffsetOverflow { offset: 1024, length: u64::MAX - 512 })]
#[case::allocation(u64::MAX - 512, 510, FormatError::OffsetOverflow { offset: u64::MAX - 2, length: 3 })]
fn huge_references_reject_physical_range_overflow(
    #[case] base: u64,
    #[case] address: u64,
    #[case] expected: FormatError,
) {
    let mut fixture = Fixture::new(true, true, 0, address, EIGHT_WIDTHS);
    fixture.header.btree_huge_objects_address = StoredAddress::new(0);
    fixture.layout = FractalHeapHugeObjectLayout::new(
        &fixture.header,
        fixture.widths,
        BaseAddress::new(base),
        HugeIndirectKeyEncoding::Libhdf5,
    )
    .unwrap();
    assert_eq!(fixture.layout.parse_id(&fixture.id).unwrap_err(), expected);
}

#[rstest]
fn huge_layouts_require_a_defined_tracker(#[values(false, true)] direct: bool) {
    let mut fixture = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
    fixture.header.btree_huge_objects_address = StoredAddress::undefined(OffsetWidth::Eight);
    assert_eq!(
        FractalHeapHugeObjectLayout::new(
            &fixture.header,
            fixture.widths,
            BaseAddress::ZERO,
            HugeIndirectKeyEncoding::Libhdf5
        )
        .unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[rstest]
fn parsed_huge_ids_require_the_receiving_representation(
    #[values(false, true)] source_filtered: bool,
    #[values(false, true)] source_direct: bool,
    #[values(false, true)] target_filtered: bool,
    #[values(false, true)] target_direct: bool,
) {
    let source = Fixture::new(source_filtered, source_direct, 0, 512, EIGHT_WIDTHS);
    let target = Fixture::new(target_filtered, target_direct, 0, 512, EIGHT_WIDTHS);
    let id = source.ids().parse(&source.id).unwrap();
    let result = target.layout.resolve_id(id);
    if source_filtered == target_filtered && source_direct == target_direct {
        let resolved = result.unwrap();
        match resolved {
            FractalHeapHugeObjectId::Direct(reference) => {
                assert_eq!(reference.range().start().get(), 512)
            }
            FractalHeapHugeObjectId::Indirect(lookup) => assert_eq!(lookup.key().value(), 5),
        }
    } else {
        assert_eq!(result.unwrap_err(), FormatError::InvalidFractalHeapGeometry);
    }
}

#[rstest]
#[case::encoded(65_536, 9, FormatError::ValueTooLargeForRepresentation { value: 65_536, target: "huge-object encoded size" })]
#[case::decoded(3, 65_536, FormatError::ValueTooLargeForRepresentation { value: 65_536, target: "huge-object decoded size" })]
fn parsed_huge_ids_recheck_length_widths(
    #[case] encoded: u64,
    #[case] decoded: u64,
    #[case] expected: FormatError,
) {
    let mut source = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
    source.id[9..17].copy_from_slice(&encoded.to_le_bytes());
    source.id[21..29].copy_from_slice(&decoded.to_le_bytes());
    let target = Fixture::new(
        true,
        true,
        0,
        512,
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Two,
        },
    );
    let id = source.ids().parse(&source.id).unwrap();
    assert_eq!(target.layout.resolve_id(id).unwrap_err(), expected);
}

#[rstest]
#[case::mask(2)]
#[case::skipped_sizes(1)]
fn parsed_huge_ids_recheck_the_receiving_pipeline(#[case] mask: u32) {
    let mut source = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
    let FractalHeapFiltering::Filtered { pipeline, .. } = &mut source.header.filtering else {
        panic!("expected filtered fixture")
    };
    pipeline.filters.push(pipeline.filters[0].clone());
    source.layout = FractalHeapHugeObjectLayout::new(
        &source.header,
        EIGHT_WIDTHS,
        BaseAddress::ZERO,
        HugeIndirectKeyEncoding::Libhdf5,
    )
    .unwrap();
    source.id[17..21].copy_from_slice(&mask.to_le_bytes());
    let target = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
    let id = source.ids().parse(&source.id).unwrap();
    assert_eq!(
        target.layout.resolve_id(id).unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[test]
fn parsed_huge_ids_recheck_address_widths() {
    let source = Fixture::new(true, true, 0, u64::from(u16::MAX), EIGHT_WIDTHS);
    let target = Fixture::new(
        true,
        true,
        0,
        512,
        FormatWidths {
            offsets: OffsetWidth::Two,
            lengths: LengthWidth::Eight,
        },
    );
    let id = source.ids().parse(&source.id).unwrap();
    assert_eq!(
        target.layout.resolve_id(id).unwrap_err(),
        FormatError::InvalidFractalHeapGeometry
    );
}

#[rstest]
fn parsed_huge_ids_recheck_numeric_key_domains(
    #[values(
        HugeIndirectKeyEncoding::Libhdf5,
        HugeIndirectKeyEncoding::PublishedSpec
    )]
    encoding: HugeIndirectKeyEncoding,
) {
    let mut source = Fixture::new(true, false, 0, 512, EIGHT_WIDTHS);
    source.header.heap_id_length = 9;
    let mut id = vec![HUGE_ID_FIRST_BYTE];
    id.extend_from_slice(&65_536u64.to_le_bytes());
    let ids = FractalHeapIdLayout::with_huge_key_encoding(
        &source.header,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        encoding,
    );
    let target = Fixture::new(
        true,
        false,
        0,
        512,
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Two,
        },
    );
    assert_eq!(
        target
            .layout
            .resolve_id(ids.parse(&id).unwrap())
            .unwrap_err(),
        FormatError::ValueTooLargeForRepresentation {
            value: 65_536,
            target: "huge-object tracker key"
        }
    );
}

struct Fixture {
    header: FractalHeapHeader,
    widths: FormatWidths,
    pipeline: FilterPipeline,
    layout: FractalHeapHugeObjectLayout,
    id: Vec<u8>,
    tracker: Vec<u8>,
    record: Vec<u8>,
}

impl Fixture {
    fn new(filtered: bool, direct: bool, base: u64, address: u64, widths: FormatWidths) -> Self {
        let mut header = FractalHeapHeader::parse_bytes(
            &test_util::fractal_heap::Header::new(u64::MAX).build(Widths::EIGHT),
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .unwrap();
        header.root_block_address = StoredAddress::undefined(widths.offsets);
        header.btree_huge_objects_address = StoredAddress::new(256);
        header.heap_id_length = if direct { 64 } else { 2 };
        let pipeline = FilterPipeline {
            version: VERSION_TWO,
            filters: vec![FilterDescription {
                filter_id: 32_001,
                name: None,
                flags: 0,
                client_data: vec![],
            }],
        };
        header.filtering = if filtered {
            FractalHeapFiltering::Filtered {
                root_direct_block_size: 0,
                root_filter_mask: 0,
                pipeline: pipeline.clone(),
            }
        } else {
            FractalHeapFiltering::Unfiltered
        };
        let layout = FractalHeapHugeObjectLayout::new(
            &header,
            widths,
            BaseAddress::new(base),
            HugeIndirectKeyEncoding::Libhdf5,
        )
        .unwrap();
        let mut record = Vec::new();
        record.extend_from_slice(&address.to_le_bytes()[..usize::from(widths.offsets.get())]);
        record.extend_from_slice(&3u64.to_le_bytes()[..usize::from(widths.lengths.get())]);
        if filtered {
            record.extend_from_slice(&0u32.to_le_bytes());
            record.extend_from_slice(&9u64.to_le_bytes()[..usize::from(widths.lengths.get())]);
        }
        let tree_type = match (filtered, direct) {
            (false, false) => hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT,
            (true, false) => hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT,
            (false, true) => hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT_DIRECT,
            (true, true) => hdf5_pure_format::__private::BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT,
        };
        let mut id = vec![HUGE_ID_FIRST_BYTE];
        if direct {
            id.extend_from_slice(&record);
        } else {
            record.extend_from_slice(&5u64.to_le_bytes()[..usize::from(widths.lengths.get())]);
            id.push(5);
        }
        id.resize(usize::from(header.heap_id_length), 0);
        let tracker = test_util::btree_v2::Header::new(
            tree_type,
            u16::try_from(record.len()).unwrap(),
            512,
            1,
        )
        .build(Widths::new(
            usize::from(widths.offsets.get()),
            usize::from(widths.lengths.get()),
        ));
        Self {
            header,
            widths,
            pipeline,
            layout,
            id,
            tracker,
            record,
        }
    }

    fn ids(&self) -> FractalHeapIdLayout {
        FractalHeapIdLayout::new(&self.header, self.widths.offsets, self.widths.lengths)
    }

    fn tracker(&self) -> FractalHeapHugeObjectTracker {
        self.layout
            .parse_tracker(FileRegion::new(self.layout.tracker_range(), &self.tracker).unwrap())
            .unwrap()
    }

    fn resolve(&self) -> FractalHeapHugeObjectRef {
        match self.layout.parse_id(&self.id).unwrap() {
            FractalHeapHugeObjectId::Direct(reference) => reference,
            FractalHeapHugeObjectId::Indirect(lookup) => {
                let record = self.tracker().parse_record(&self.record).unwrap();
                self.layout.resolve(&lookup, &record).unwrap()
            }
        }
    }
}

const EIGHT_WIDTHS: FormatWidths = FormatWidths {
    offsets: OffsetWidth::Eight,
    lengths: LengthWidth::Eight,
};

/// The version-0 huge-ID type byte from "Fractal Heap" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const HUGE_ID_FIRST_BYTE: u8 = 0x10;
