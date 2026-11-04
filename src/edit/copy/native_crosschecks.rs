use std::cell::RefCell;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::DenseAttrPlan;
use rstest::rstest;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::object_copy::{self, ReferenceSite};
use test_util_hdf5::{attr, dataset, file};

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::attribute_info::AttributeInfoMessage;
use crate::chunked_write::{self, ChunkMeta, ChunkProvider};
use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags, DataLayout};
use crate::edit::context;
use crate::edit::dataset::ChunkGeometry;
use crate::edit::object_header;
use crate::error::{Error, FormatError};
use crate::file_emission::EmissionError;
use crate::file_writer;
use crate::group::v2 as group_v2;
use crate::message_type::MessageType;
use crate::object_path::ObjectPathBuf;
use crate::reader::OpenedSource;
use crate::reference_patch;
use crate::source::{BytesSource, Source};
use crate::superblock;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};
use crate::{
    AttrValue, DeflateLevel, File, FileAccessProperties, FileBuilder, MaxExtent, MemoryStrategy,
    PhysicalAddress, StoredAddress,
};

use super::{CopyNode, CopyOutput, CopyTree, WriteEngine};

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
    let source_bytes = std::fs::read(&source_path).unwrap();
    let located = superblock::read(&BytesSource::new(source_bytes.as_slice())).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let source = BytesSource::new(source_bytes.as_slice());
    let opened = OpenedSource::for_test(&source, addresses);
    let address = group_v2::resolve_path_any_from_source(
        &opened,
        AccessMode::ReadOnly,
        &located.superblock,
        widths,
        &ObjectPathBuf::parse("tree").as_path(),
    )
    .unwrap();
    let tree =
        WriteEngine::read_copy_subtree(&opened, AccessMode::ReadOnly, address, cross_file, widths);
    if cross_file && matches!(fixture, Fixture::DenseVariableLength) {
        let Err(error) = tree else {
            panic!("a foreign heap reference must be rejected");
        };
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "variable-length or reference dense (fractal-heap) attributes cannot be copied to another file yet"
                )
            ),
            "{error:?}"
        );
        assert_eq!(std::fs::read(&source_path).unwrap(), source_bytes);
        assert!(!output_path.exists());
        directory.close().unwrap();
        return;
    }
    let tree = tree.unwrap();
    assert_copy_geometry(&tree, widths, fixture);
    let (mut bytes, located) = if cross_file {
        drop(file::libhdf5_create_v110_with_widths(
            &output_path,
            offsets,
            lengths,
            0,
        ));
        let bytes = std::fs::read(&output_path).unwrap();
        let located = superblock::read(&BytesSource::new(bytes.as_slice())).unwrap();
        (bytes, located)
    } else {
        (source_bytes.clone(), located)
    };
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let root = WriteEngine::gather_oh_messages(
        &BytesSource::new(bytes.as_slice()),
        addresses
            .resolve(located.superblock.root_group_address)
            .unwrap()
            .get(),
        addresses,
        widths,
    )
    .unwrap();
    let copied = write_tree(&mut bytes, addresses, widths, &tree);
    let mut root = root;
    root.push(
        MessageType::LINK,
        &file_writer::make_link("copied", copied)
            .serialize(offsets)
            .unwrap(),
    );
    let header = object_header::build_v2_object_header(&root).unwrap();
    let mut block = located.superblock;
    block.root_group_address = append_checked(&mut bytes, addresses, widths, &header);
    block.eof_address = u64::try_from(bytes.len()).unwrap();
    let encoded = hdf5_pure_format::__private::serialize_superblock(&block).unwrap();
    bytes[..encoded.len()].copy_from_slice(&encoded);
    assert_eq!(
        superblock::read(&BytesSource::new(bytes.as_slice()))
            .unwrap()
            .widths,
        widths
    );
    std::fs::write(&output_path, &bytes).unwrap();
    assert_contents(&output_path, "copied", fixture, lengths);
    assert_eq!(std::fs::read(&source_path).unwrap(), source_bytes);
    directory.close().unwrap();
}

fn assert_copy_geometry(tree: &CopyTree, widths: FormatWidths, fixture: Fixture) {
    assert_eq!(
        tree.has_dense_attributes(),
        matches!(fixture, Fixture::DenseVariableLength)
    );
    let mut fixed_arrays = 0;
    let mut single_chunks = 0;
    let mut unallocated = 0;
    let mut extensible_arrays = 0;
    for node in core::iter::once(&tree.root).chain(&tree.descendants) {
        assert_eq!(node.region().widths(), widths);
        match node {
            CopyNode::DatasetChunked { region, .. } => {
                let DataLayout::Chunked { index, .. } =
                    super::parse_chunked_header(region, widths).unwrap().layout
                else {
                    panic!("the copied source must be chunked");
                };
                match index {
                    ChunkIndexLayout::SingleChunk { .. } => single_chunks += 1,
                    ChunkIndexLayout::FixedArray { .. } => fixed_arrays += 1,
                    ChunkIndexLayout::ExtensibleArray { .. } => extensible_arrays += 1,
                    other => panic!("unexpected source chunk index {other:?}"),
                }
            }
            CopyNode::DatasetContiguous { data: None, .. } => unallocated += 1,
            _ => {}
        }
    }
    assert_eq!((fixed_arrays, single_chunks, unallocated), (1, 1, 1));
    assert_eq!(
        extensible_arrays,
        usize::from(widths.lengths == LengthWidth::Eight)
    );
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

fn write_tree(
    bytes: &mut Vec<u8>,
    addresses: AddressContext,
    widths: FormatWidths,
    tree: &CopyTree,
) -> StoredAddress {
    tree.emit(&mut CopyBuffer {
        bytes,
        addresses,
        widths,
    })
    .unwrap()
}

struct CopyBuffer<'a> {
    bytes: &'a mut Vec<u8>,
    addresses: AddressContext,
    widths: FormatWidths,
}

impl CopyOutput for CopyBuffer<'_> {
    fn output_widths(&self) -> FormatWidths {
        self.widths
    }

    fn place_data(&mut self, data: &[u8]) -> Result<StoredAddress, Error> {
        Ok(append_checked(
            self.bytes,
            self.addresses,
            self.widths,
            data,
        ))
    }

    fn place_chunks(
        &mut self,
        geometry: &ChunkGeometry,
        flags: ChunkedLayoutFlags,
        pipeline: Option<&[u8]>,
        meta: &[ChunkMeta],
        chunks: &dyn ChunkProvider,
    ) -> Result<Vec<u8>, Error> {
        let address = self.addresses.relative(PhysicalAddress::new(
            u64::try_from(self.bytes.len()).unwrap(),
        ))?;
        let layout = geometry.plan_verbatim(self.widths, flags, pipeline, meta, address)?;
        let mut blob = Vec::new();
        chunked_write::emit_chunked_data_verbatim(&mut blob, &layout.plan, chunks)
            .map_err(EmissionError::into_format)?;
        if u64::try_from(blob.len()).unwrap() != layout.plan.total_len {
            return Err(FormatError::Internal(
                "a copied chunk region differs from its plan".into(),
            )
            .into());
        }
        append_checked(self.bytes, self.addresses, self.widths, &blob);
        Ok(layout.layout_message)
    }

    fn place_attributes(&mut self, plan: DenseAttrPlan) -> Result<Vec<u8>, Error> {
        let address = self.addresses.relative(PhysicalAddress::new(
            u64::try_from(self.bytes.len()).unwrap(),
        ))?;
        let blob = plan.build(address)?;
        append_checked(self.bytes, self.addresses, self.widths, &blob.blob);
        Ok(blob.attr_info_message)
    }

    fn place_header(&mut self, header: &[u8]) -> Result<StoredAddress, Error> {
        Ok(append_checked(
            self.bytes,
            self.addresses,
            self.widths,
            header,
        ))
    }
}

fn append_checked(
    bytes: &mut Vec<u8>,
    addresses: AddressContext,
    widths: FormatWidths,
    body: &[u8],
) -> StoredAddress {
    let start = PhysicalAddress::new(u64::try_from(bytes.len()).unwrap());
    context::require_range(
        widths.offsets,
        addresses,
        FileRange::new(start, u64::try_from(body.len()).unwrap()).unwrap(),
    )
    .unwrap();
    bytes.extend_from_slice(body);
    addresses.relative(start).unwrap()
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
            .unwrap()
            .get(),
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
            addresses
                .resolve(located.superblock.root_group_address)
                .unwrap()
                .get(),
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
            addresses
                .resolve(located.superblock.root_group_address)
                .unwrap()
                .get(),
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
    } = WriteEngine::read_object(&opened, AccessMode::ReadOnly, dataset, widths).unwrap()
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
