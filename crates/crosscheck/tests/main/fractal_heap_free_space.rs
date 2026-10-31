#![cfg(feature = "hdf5")]

use std::fs;

use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_core::FormatError;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapFreeSection;
use hdf5_pure_format::__private::FractalHeapHeaderLocator;
use hdf5_pure_format::__private::FractalHeapSectionClass;
use hdf5_pure_format::__private::FreeSpaceManagerHeaderRef;
use rstest::rstest;
use test_util::fractal_heap;

#[rstest]
#[case::links(300, 100, false, true)]
#[case::attributes(300, 100, true, false)]
#[case::nested_links(800, 1024, false, true)]
#[case::nested_indirect_span(1300, 1024, false, true)]
#[case::small_links(12, 8, false, false)]
fn parses_c_written_managed_free_space(
    #[case] count: usize,
    #[case] name_len: usize,
    #[case] attributes: bool,
    #[case] delete_half: bool,
    #[values(0, 512)] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("managed-space.h5");
    let file = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_latest())
        .with_fcpl(|properties| properties.userblock(userblock))
        .create(&path)
        .unwrap();
    let group = file.create_group("g").unwrap();
    let names = (0..count)
        .map(|index| {
            let prefix = format!("{index:03}_");
            format!(
                "{prefix}{}",
                "a".repeat(name_len.saturating_sub(prefix.len()))
            )
        })
        .collect::<Vec<_>>();
    for name in &names {
        if attributes {
            group
                .new_attr::<i32>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&42)
                .unwrap();
        } else {
            group.link_soft("/unresolved", name).unwrap();
        }
    }
    if delete_half {
        for name in &names[..count / 2] {
            group.unlink(name).unwrap();
        }
    }
    drop(group);
    file.close().unwrap();
    let bytes = fs::read(&path).unwrap();
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(bytes.len()).unwrap()).unwrap(),
        &bytes,
    )
    .unwrap();
    let base = BaseAddress::new(userblock);
    let heap_offsets = fractal_heap::header_offsets(&bytes);
    assert_eq!(heap_offsets.len(), 1);
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(u64::try_from(heap_offsets[0]).unwrap()),
        widths.offsets,
        widths.lengths,
    );
    let heap_ref = locator.parse(region).unwrap();
    let heap = heap_ref.parse(region).unwrap();
    let header_addr = heap.managed_block_free_space_manager_address;
    let header_ref = FreeSpaceManagerHeaderRef::new(
        widths,
        PhysicalAddress::new(base.absolute(header_addr).unwrap()),
    )
    .unwrap();
    let frame = header_ref.parse(region).unwrap();
    assert_eq!(frame.client_id(), FRACTAL_HEAP_CLIENT_ID);
    let manager = frame.fractal_heap_header(&heap).unwrap();
    let DecodedStoredAddress::Defined(list_addr) = frame.section_list_addr() else {
        panic!("expected a managed section list")
    };
    let reference = manager
        .section_list_ref(
            header_addr,
            PhysicalAddress::new(base.absolute(list_addr).unwrap()),
        )
        .unwrap();
    let parsed = reference
        .parse(region.subregion(reference.parse_range()).unwrap())
        .unwrap();
    assert_eq!(
        (parsed.total_sections(), parsed.total_space()),
        (frame.total_sections(), frame.total_space())
    );
    assert_eq!(
        u64::try_from(parsed.sections().len()).unwrap(),
        frame.serialized_sections()
    );
    assert_eq!(
        frame.serialized_sections() + frame.unserialized_sections(),
        frame.total_sections()
    );
    assert!(parsed.total_sections() > 0);
    if delete_half {
        assert!(frame.unserialized_sections() > 0);
        let first_rows = parsed
            .sections()
            .iter()
            .filter_map(|section| match section {
                FractalHeapFreeSection::FirstRow { row, indirect } => Some((row, indirect)),
                FractalHeapFreeSection::Single { .. } => None,
            })
            .collect::<Vec<_>>();
        assert!(!first_rows.is_empty());
        for (row, indirect) in first_rows {
            assert_eq!(row.class(), FractalHeapSectionClass::FirstRow);
            assert_eq!(indirect.class(), FractalHeapSectionClass::Indirect);
        }
    }
    if count == 300 && !attributes {
        let FractalHeapFreeSection::FirstRow { row, indirect } = parsed.sections()[4] else {
            panic!("expected a first-row proxy at record 4")
        };
        assert_eq!(
            (
                row.offset(),
                row.size(),
                row.row(),
                row.column(),
                row.blocks()
            ),
            (0, 491, 0, 0, 4)
        );
        assert_eq!(
            (
                indirect.block_offset(),
                indirect.rows(),
                indirect.start_row(),
                indirect.start_column(),
                indirect.blocks()
            ),
            (0, 8, 0, 0, 16)
        );
        let normal = manager.row_section(indirect, 3).unwrap();
        assert_eq!(
            (
                normal.class(),
                normal.offset(),
                normal.size(),
                normal.row(),
                normal.column(),
                normal.blocks()
            ),
            (FractalHeapSectionClass::NormalRow, 8192, 2027, 3, 0, 4)
        );
    }
    if count == 800 {
        let indirect = parsed
            .sections()
            .iter()
            .find_map(|section| match section {
                FractalHeapFreeSection::FirstRow { row, indirect } if row.offset() == 0 => {
                    Some(*indirect)
                }
                _ => None,
            })
            .expect("a proxy covers the root direct rows");
        assert_eq!(
            (
                indirect.block_offset(),
                indirect.rows(),
                indirect.start_row(),
                indirect.start_column(),
                indirect.blocks()
            ),
            (0, 16, 0, 0, 34)
        );
        assert_eq!(
            manager.child_section(indirect, 0, 0).unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }
    #[cfg(feature = "__hdf5-1.10")]
    if count == 1300 {
        let indirect = parsed
            .sections()
            .iter()
            .find_map(|section| match section {
                FractalHeapFreeSection::FirstRow { row, indirect } if row.offset() == 524288 => {
                    Some(*indirect)
                }
                _ => None,
            })
            .expect("a proxy covers the first indirect child");
        assert_eq!(
            (
                indirect.block_offset(),
                indirect.rows(),
                indirect.start_row(),
                indirect.start_column(),
                indirect.blocks()
            ),
            (0, 16, 9, 0, 1)
        );
        let child = manager.child_section(indirect, 9, 0).unwrap();
        assert_eq!(
            (
                child.class(),
                child.block_offset(),
                child.rows(),
                child.start_row(),
                child.start_column(),
                child.blocks()
            ),
            (FractalHeapSectionClass::Indirect, 524288, 7, 0, 0, 28)
        );
        let first = manager.row_section(child, 0).unwrap();
        assert_eq!(
            (
                first.class(),
                first.offset(),
                first.size(),
                first.row(),
                first.column(),
                first.blocks()
            ),
            (FractalHeapSectionClass::FirstRow, 524288, 491, 0, 0, 4)
        );
        let normal = manager.row_section(child, 6).unwrap();
        assert_eq!(
            (
                normal.class(),
                normal.offset(),
                normal.size(),
                normal.row(),
                normal.column(),
                normal.blocks()
            ),
            (FractalHeapSectionClass::NormalRow, 589824, 16363, 6, 0, 4)
        );
    }
    let c = hdf5::File::open(&path).unwrap();
    let group = c.group("g").unwrap();
    if attributes {
        assert_eq!(group.attr_names().unwrap().len(), count);
    } else {
        let mut actual = group.member_names().unwrap();
        actual.sort_unstable();
        let mut expected = if delete_half {
            names[count / 2..].to_vec()
        } else {
            names
        };
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }
}

#[test]
fn c_written_dense_link_fixture_matches_the_preserved_metadata_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.h5");
    let file = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_latest())
        .create(&path)
        .unwrap();
    let group = file.create_group("g").unwrap();
    for index in 0..300 {
        group
            .link_soft("/unresolved", &format!("{index:03}_{}", "a".repeat(96)))
            .unwrap();
    }
    for index in 0..150 {
        group
            .unlink(&format!("{index:03}_{}", "a".repeat(96)))
            .unwrap();
    }
    drop(group);
    file.close().unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(
        &bytes[1219..1365],
        fractal_heap::dense_link_free_space_header()
    );
    assert_eq!(
        &bytes[1403..1485],
        fractal_heap::dense_link_free_space_manager()
    );
    assert_eq!(
        &bytes[5069..5147],
        fractal_heap::dense_link_free_space_sections()
    );
}

// The fractal-heap client ID is defined in "Free-space Index", format specification version 4.0.
const FRACTAL_HEAP_CLIENT_ID: u8 = 0;
