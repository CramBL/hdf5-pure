use std::cell::RefCell;
use std::fs;
use std::path::Path;

use rstest::rstest;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::object_copy::{self, ReferenceSite};
use test_util_hdf5::{attr, dataset, file};

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::attribute_info::AttributeInfoMessage;
use crate::error::{Error, FormatError};
use crate::group::v2 as group_v2;
use crate::message_type::MessageType;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::reader::OpenedSource;
use crate::reference_patch;
use crate::source::{BytesSource, Source};
use crate::superblock;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};
use crate::{
    AttrValue, ChunkIndex, DeflateLevel, File, FileAccessProperties, FileBuilder, Layout,
    MaxExtent, MemoryStrategy, PhysicalAddress,
};

use super::WriteEngine;

#[derive(Clone, Copy, Debug)]
enum Fixture {
    DenseVariableLength,
    Native,
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn copied_headers_and_payloads_preserve_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(Fixture::Native, Fixture::DenseVariableLength)] fixture: Fixture,
    #[values(false, true)] cross_file: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.h5");
    let output_path = directory.path().join("copied.h5");
    let widths = FormatWidths { offsets, lengths };
    create_source(&source_path, widths, fixture);
    let source_bytes = fs::read(&source_path).unwrap();
    assert_copy_geometry(&source_path, "tree", widths, fixture);

    if cross_file {
        let native = file::libhdf5_create_v110_with_widths(&output_path, offsets, lengths, 0);
        native
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.u32_datatype())
            .shape(1)
            .create("retained")
            .unwrap()
            .write_raw(&[7_u32])
            .unwrap();
        native.close().unwrap();
    } else {
        fs::write(&output_path, &source_bytes).unwrap();
    }
    let before = fs::read(&output_path).unwrap();
    let destination = File::open_rw(&output_path).unwrap();
    let result = if cross_file {
        let source = File::open(&source_path).unwrap();
        let copied = destination.copy_from(&source, "tree", "copied");
        source.close().unwrap();
        copied
    } else {
        destination.copy("tree", "copied")
    };
    let rejected = cross_file && matches!(fixture, Fixture::DenseVariableLength);
    if rejected {
        let error = result.unwrap_err();
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "variable-length or reference dense (fractal-heap) attributes cannot be copied to another file yet"
                )
            ),
            "{error:?}"
        );
    } else {
        result.unwrap();
    }
    assert_eq!(destination.has_staged_edits(), !rejected);
    destination.commit().unwrap();
    assert!(!destination.has_staged_edits());
    destination.close().unwrap();

    assert_eq!(fs::read(&source_path).unwrap(), source_bytes);
    assert_contents(&source_path, "tree", fixture, lengths);
    if rejected {
        assert_eq!(fs::read(&output_path).unwrap(), before);
    } else {
        assert_copy_geometry(&output_path, "copied", widths, fixture);
        assert_contents(&output_path, "copied", fixture, lengths);
    }
    if cross_file {
        let native = file::libhdf5_open(&output_path);
        let mut names = native.member_names().unwrap();
        names.sort_unstable();
        assert_eq!(
            names,
            if rejected {
                &["retained"][..]
            } else {
                &["copied", "retained"]
            }
        );
        assert_eq!(
            native
                .dataset("retained")
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            [7]
        );
        native.close().unwrap();
    } else {
        assert_copy_geometry(&output_path, "tree", widths, fixture);
        assert_contents(&output_path, "tree", fixture, lengths);
    }
    directory.close().unwrap();
}

fn assert_copy_geometry(path: &Path, group: &str, widths: FormatWidths, fixture: Fixture) {
    let file = File::open(path).unwrap();
    assert_eq!(file.superblock().offset_size, widths.offsets);
    assert_eq!(file.superblock().length_size, widths.lengths);
    for (name, index, chunk) in [
        ("single", ChunkIndex::SingleChunk, 3),
        ("chunked", ChunkIndex::FixedArray, 2),
        ("extensible", ChunkIndex::ExtensibleArray, 2),
    ] {
        if name == "extensible" && widths.lengths != LengthWidth::Eight {
            continue;
        }
        let dataset = file.dataset(&format!("{group}/{name}")).unwrap();
        assert_eq!(
            dataset.layout().unwrap(),
            Layout::Chunked {
                index,
                chunk_shape: vec![chunk]
            }
        );
    }
    let dataset = file.dataset(&format!("{group}/unallocated")).unwrap();
    assert_eq!(dataset.shape().unwrap(), [3]);
    assert_eq!(
        dataset.layout().unwrap(),
        Layout::Contiguous {
            address: None,
            size: 12
        }
    );
    drop(dataset);
    file.close().unwrap();

    let bytes = fs::read(path).unwrap();
    let source = BytesSource::new(&bytes);
    let located = superblock::read(&source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let address = group_v2::resolve_path_any_from_source(
        &OpenedSource::for_test(&source, addresses),
        AccessMode::ReadOnly,
        &located.superblock,
        widths,
        &ObjectPathBuf::parse(group).as_path(),
    )
    .unwrap();
    let header =
        WriteEngine::gather_oh_messages(&source, PhysicalAddress::new(address), addresses, widths)
            .unwrap();
    let mut dense = false;
    let mut position = 0;
    while let Some((kind, body, end)) = header.next_message(position).unwrap() {
        if kind == MessageType::ATTRIBUTE_INFO {
            dense |=
                AttributeInfoMessage::parse_with_offset_width(&header[body..end], widths.offsets)
                    .unwrap()
                    .fractal_heap_address
                    .is_some();
        }
        position = end;
    }
    assert_eq!(dense, matches!(fixture, Fixture::DenseVariableLength));
}

fn create_source(path: &Path, widths: FormatWidths, fixture: Fixture) {
    let values = [1_u32, 2, 3, 4, 5, 6];
    match fixture {
        Fixture::Native => {
            let native =
                file::libhdf5_create_v110_with_widths(path, widths.offsets, widths.lengths, 0);
            let group = native.create_group("tree").unwrap();
            group
                .new_attr::<u32>()
                .create("counter")
                .unwrap()
                .write_scalar(&17)
                .unwrap();
            group
                .new_dataset_builder()
                .empty_as(ByteOrder::LittleEndian.u32_datatype())
                .shape(6)
                .create("values")
                .unwrap()
                .write_raw(&values)
                .unwrap();
            group
                .new_dataset_builder()
                .empty_as(ByteOrder::LittleEndian.u32_datatype())
                .shape(6)
                .chunk(2)
                .deflate(1)
                .create("chunked")
                .unwrap()
                .write_raw(&values)
                .unwrap();
            group
                .new_dataset_builder()
                .empty_as(ByteOrder::LittleEndian.u32_datatype())
                .shape(3)
                .chunk(3)
                .create("single")
                .unwrap()
                .write_raw(&values[..3])
                .unwrap();
            if widths.lengths == LengthWidth::Eight {
                group
                    .new_dataset_builder()
                    .empty_as(ByteOrder::LittleEndian.u32_datatype())
                    .shape((6..,))
                    .chunk(2)
                    .deflate(1)
                    .create("extensible")
                    .unwrap()
                    .write_raw(&values)
                    .unwrap();
            }
            group
                .new_dataset_builder()
                .empty_as(ByteOrder::LittleEndian.u32_datatype())
                .shape(3)
                .create("unallocated")
                .unwrap();
        }
        Fixture::DenseVariableLength => {
            let mut builder = FileBuilder::new();
            builder
                .with_offset_width(widths.offsets)
                .with_length_width(widths.lengths);
            let mut group = builder.create_group("tree");
            for index in 0..9 {
                group.set_attr(&format!("number_{index}"), AttrValue::U32(index));
            }
            group.set_attr(
                "label",
                AttrValue::VarLenStringArray(vec!["first".into(), "second".into()]),
            );
            group.create_dataset("values").with_u32_data(&values);
            group
                .create_dataset("chunked")
                .with_u32_data(&values)
                .with_chunks(&[2])
                .with_deflate(DeflateLevel::One);
            group
                .create_dataset("single")
                .with_u32_data(&values[..3])
                .with_chunks(&[3]);
            group
                .create_dataset("unallocated")
                .with_unallocated_storage(crate::type_builders::make_u32_type(), &[3]);
            if widths.lengths == LengthWidth::Eight {
                group
                    .create_dataset("extensible")
                    .with_u32_data(&values)
                    .with_chunks(&[2])
                    .with_maxshape(&[MaxExtent::Unlimited])
                    .with_deflate(DeflateLevel::One);
            }
            group
                .create_dataset("strings")
                .with_vlen_strings(&["alpha", "", "gamma"]);
            builder.add_group(group.finish());
            builder.write(path).unwrap();
        }
    }
}

fn assert_contents(path: &Path, group_path: &str, fixture: Fixture, lengths: LengthWidth) {
    let native = file::libhdf5_open(path);
    let group = native.group(group_path).unwrap();
    let mut expected = vec!["chunked", "single", "unallocated", "values"];
    if matches!(fixture, Fixture::DenseVariableLength) {
        expected.push("strings");
    }
    if lengths == LengthWidth::Eight {
        expected.push("extensible");
    }
    expected.sort_unstable();
    let mut actual = group.member_names().unwrap();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    assert_eq!(
        group.dataset("values").unwrap().read_raw::<u32>().unwrap(),
        [1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        group.dataset("chunked").unwrap().read_raw::<u32>().unwrap(),
        [1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        group.dataset("single").unwrap().read_raw::<u32>().unwrap(),
        [1, 2, 3]
    );
    assert_eq!(
        group
            .dataset("unallocated")
            .unwrap()
            .read_raw::<u32>()
            .unwrap(),
        [0, 0, 0]
    );
    if lengths == LengthWidth::Eight {
        assert_eq!(
            group
                .dataset("extensible")
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            [1, 2, 3, 4, 5, 6]
        );
    }
    match fixture {
        Fixture::Native => assert_eq!(
            group.attr("counter").unwrap().read_scalar::<u32>().unwrap(),
            17
        ),
        Fixture::DenseVariableLength => {
            for index in 0..9 {
                assert_eq!(
                    group
                        .attr(&format!("number_{index}"))
                        .unwrap()
                        .read_scalar::<u32>()
                        .unwrap(),
                    index
                );
            }
            assert_eq!(attr::unicode_array(&group, "label"), ["first", "second"]);
        }
    }
    drop(group);
    drop(native);
    if matches!(fixture, Fixture::DenseVariableLength) {
        assert_eq!(
            dataset::read_libhdf5_strings(path, &format!("{group_path}/strings")),
            ["alpha", "", "gamma"]
        );
    }
    let reopened = File::open(path).unwrap();
    assert_eq!(
        reopened
            .dataset(&format!("{group_path}/values"))
            .unwrap()
            .read_u32()
            .unwrap(),
        [1, 2, 3, 4, 5, 6]
    );
    reopened.close().unwrap();
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn native_legacy_reference_screening_preserves_original_values(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(ReferenceSite::Attribute, ReferenceSite::Dataset)] site: ReferenceSite,
) {
    screen_native_legacy_reference(FormatWidths { offsets, lengths }, site);
}

#[rstest]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn native_dense_legacy_reference_screening_preserves_original_values(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    screen_native_legacy_reference(
        FormatWidths { offsets, lengths },
        ReferenceSite::DenseAttribute,
    );
}

fn screen_native_legacy_reference(widths: FormatWidths, site: ReferenceSite) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("references.h5");
    {
        let native =
            file::libhdf5_create_v110_with_widths(&path, widths.offsets, widths.lengths, 0);
        native
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.u32_datatype())
            .shape(1)
            .create("target")
            .unwrap()
            .write_raw(&[7_u32])
            .unwrap();
        object_copy::write_legacy_reference(&native, site);
        if site == ReferenceSite::Dataset {
            assert_eq!(
                native.dataset("references").unwrap().storage_size(),
                object_copy::legacy_reference_storage_size(widths.offsets)
            );
        } else {
            assert_eq!(
                native.attr("reference").unwrap().storage_size(),
                object_copy::legacy_reference_storage_size(widths.offsets)
            );
        }
    }
    object_copy::assert_legacy_reference(&path, site, 7);
    let bytes = std::fs::read(&path).unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let located = superblock::read(&source).unwrap();
    assert_eq!(located.superblock.offset_size.get(), widths.offsets.get());
    assert_eq!(located.superblock.length_size.get(), widths.lengths.get());
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let opened = OpenedSource::for_test(&source, addresses);
    let root = WriteEngine::gather_oh_messages(
        &source,
        addresses
            .resolve(located.superblock.root_group_address)
            .unwrap(),
        addresses,
        widths,
    )
    .unwrap();
    let mut dense = false;
    let mut position = 0;
    while let Some((kind, body, end)) = root.next_message(position).unwrap() {
        if kind == MessageType::ATTRIBUTE_INFO {
            dense |=
                AttributeInfoMessage::parse_with_offset_width(&root[body..end], widths.offsets)
                    .unwrap()
                    .fractal_heap_address
                    .is_some();
        }
        position = end;
    }
    assert_eq!(dense, site == ReferenceSite::DenseAttribute);
    let proof =
        reference_patch::prove_free_of_references(&opened, &located.superblock, widths, 1000);
    if widths.offsets == OffsetWidth::Eight {
        assert!(!proof.unwrap());
        let tree = WriteEngine::read_copy_subtree(
            &opened,
            AccessMode::ReadOnly,
            ObjectId::new(
                addresses
                    .resolve(located.superblock.root_group_address)
                    .unwrap(),
            ),
            false,
            widths,
        )
        .unwrap();
        assert_eq!(
            tree.has_dense_attributes(),
            site == ReferenceSite::DenseAttribute
        );
    } else {
        let error = proof.unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::ReferenceWidthUnsupported { offset_size }) if offset_size == widths.offsets.get()),
            "{error:?}"
        );
        let tree = WriteEngine::read_copy_subtree(
            &opened,
            AccessMode::ReadOnly,
            ObjectId::new(
                addresses
                    .resolve(located.superblock.root_group_address)
                    .unwrap(),
            ),
            false,
            widths,
        );
        let Err(error) = tree else {
            panic!("copying a narrow legacy reference must fail");
        };
        assert!(
            matches!(error, Error::Format(FormatError::ReferenceWidthUnsupported { offset_size }) if offset_size == widths.offsets.get()),
            "{error:?}"
        );
    }
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    object_copy::assert_legacy_reference(&path, site, 7);
    directory.close().unwrap();
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
fn cross_file_width_mismatch_preserves_both_files(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.h5");
    let destination_path = directory.path().join("destination.h5");
    create_source(
        &source_path,
        FormatWidths { offsets, lengths },
        Fixture::Native,
    );
    let mut builder = FileBuilder::new();
    builder.create_dataset("retained").with_u32_data(&[7]);
    builder.write(&destination_path).unwrap();
    let before_source = std::fs::read(&source_path).unwrap();
    let before_destination = std::fs::read(&destination_path).unwrap();
    let source = File::open(&source_path).unwrap();
    let destination = File::open_rw(&destination_path).unwrap();
    let error = destination
        .copy_from(&source, "tree", "copied")
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("cross-file copy requires matching address and length widths")
        ),
        "{error:?}"
    );
    source.close().unwrap();
    destination.close().unwrap();
    assert_eq!(std::fs::read(&source_path).unwrap(), before_source);
    assert_eq!(
        std::fs::read(&destination_path).unwrap(),
        before_destination
    );
    assert_contents(&source_path, "tree", Fixture::Native, lengths);
    {
        let native = file::libhdf5_open(&destination_path);
        assert_eq!(native.member_names().unwrap(), ["retained"]);
        assert_eq!(
            native
                .dataset("retained")
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            [7]
        );
    }
    directory.close().unwrap();
}

#[rstest]
#[case::same_file(false)]
#[case::other_file(true)]
fn supported_copy_commits_preserve_native_values(
    #[case] cross_file: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.h5");
    let destination_path = directory.path().join("destination.h5");
    create_source(
        &source_path,
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        Fixture::Native,
    );
    if cross_file {
        let mut builder = FileBuilder::new();
        builder.create_dataset("retained").with_u32_data(&[7]);
        builder.write(&destination_path).unwrap();
    }
    let output = if cross_file {
        &destination_path
    } else {
        &source_path
    };
    let destination = File::open_rw_with_options(
        output,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    if cross_file {
        let source = File::open(&source_path).unwrap();
        destination.copy_from(&source, "tree", "copied").unwrap();
        source.close().unwrap();
    } else {
        destination.copy("tree", "copied").unwrap();
    }
    destination.commit().unwrap();
    destination.close().unwrap();
    assert_contents(output, "copied", Fixture::Native, LengthWidth::Eight);
    directory.close().unwrap();
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn reference_inspection_skips_dataset_values(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(Fixture::Native, Fixture::DenseVariableLength)] fixture: Fixture,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("inspection.h5");
    let widths = FormatWidths { offsets, lengths };
    create_source(&path, widths, fixture);
    let bytes = std::fs::read(&path).unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let located = superblock::read(&source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let opened = OpenedSource::for_test(&source, addresses);
    let dataset = group_v2::resolve_path_any_from_source(
        &opened,
        AccessMode::ReadOnly,
        &located.superblock,
        widths,
        &ObjectPathBuf::parse("tree/values").as_path(),
    )
    .unwrap();
    let super::ObjModel::DatasetContiguous {
        data_addr,
        data_size,
        ..
    } = WriteEngine::read_object(
        &opened,
        AccessMode::ReadOnly,
        ObjectId::new(PhysicalAddress::new(dataset)),
        widths,
    )
    .unwrap()
    else {
        panic!("the source must have contiguous values");
    };
    assert_eq!(data_size, 24);
    let data_start = addresses.resolve(data_addr).unwrap().get();
    let audit = ReadAudit {
        source,
        reads: RefCell::new(Vec::new()),
        failure: None,
    };
    let opened = OpenedSource::for_test(&audit, addresses);
    let proof = if offsets == OffsetWidth::Eight {
        reference_patch::prove_free_of_references(&opened, &located.superblock, widths, 1000)
            .unwrap()
    } else {
        reference_patch::preflight_narrow_references(
            &opened,
            &located.superblock,
            widths,
            1000,
            false,
        )
        .unwrap()
    };
    assert_eq!(
        proof,
        offsets != OffsetWidth::Eight || matches!(fixture, Fixture::Native)
    );
    assert!(!audit.reads.borrow().is_empty());
    assert!(audit.reads.borrow().len() < 300);
    assert!(audit.reads.borrow().iter().all(|&(at, len)| {
        at + u64::try_from(len).unwrap() <= data_start || at >= data_start + data_size
    }));
    if offsets != OffsetWidth::Eight {
        let reads = audit.reads.borrow().len();
        assert!(
            reference_patch::preflight_narrow_references(
                &opened,
                &located.superblock,
                widths,
                1000,
                proof
            )
            .unwrap()
        );
        assert_eq!(audit.reads.borrow().len(), reads);
        assert!(
            reference_patch::preflight_narrow_references(
                &opened,
                &located.superblock,
                widths,
                1000,
                false
            )
            .unwrap()
        );
        assert!(audit.reads.borrow().len() > reads);
    }
    let raw = audit
        .read_exact_at(data_start, usize::try_from(data_size).unwrap())
        .unwrap();
    assert_eq!(
        raw,
        [1_u32, 2, 3, 4, 5, 6]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(audit.reads.borrow().last(), Some(&(data_start, 24)));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    directory.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum InspectionFailure {
    Budget,
    Source,
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
fn narrow_reference_inspection_preserves_source_failures(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(InspectionFailure::Budget, InspectionFailure::Source)] failure: InspectionFailure,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("inspection.h5");
    let widths = FormatWidths { offsets, lengths };
    create_source(&path, widths, Fixture::Native);
    let bytes = std::fs::read(&path).unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let located = superblock::read(&source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let audit = ReadAudit {
        source,
        reads: RefCell::new(Vec::new()),
        failure: matches!(failure, InspectionFailure::Source).then(|| {
            addresses
                .resolve(located.superblock.root_group_address)
                .unwrap()
                .get()
        }),
    };
    let error = reference_patch::preflight_narrow_references(
        &OpenedSource::for_test(&audit, addresses),
        &located.superblock,
        widths,
        1,
        false,
    )
    .unwrap_err();
    match failure {
        InspectionFailure::Budget => assert!(
            matches!(
                error,
                Error::Format(FormatError::ResourceLimit {
                    limit: 1,
                    required: 2
                })
            ),
            "{error:?}"
        ),
        InspectionFailure::Source => {
            let Error::Format(FormatError::Source(message)) = error else {
                panic!("expected Source, got {error:?}");
            };
            assert_eq!(message, "injected metadata read error");
        }
    }
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    directory.close().unwrap();
}

struct ReadAudit<'a> {
    source: BytesSource<&'a [u8]>,
    reads: RefCell<Vec<(u64, usize)>>,
    failure: Option<u64>,
}

impl Source for ReadAudit<'_> {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.reads.borrow_mut().push((offset, buffer.len()));
        if self.failure == Some(offset) {
            return Err(FormatError::Source("injected metadata read error".into()));
        }
        self.source.read_at(offset, buffer)
    }
}
