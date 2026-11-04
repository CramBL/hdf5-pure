use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util_hdf5::attr;
use test_util_hdf5::file;

use crate::AttrValue;
use crate::address::AddressContext;
use crate::edit::WriteEngine;
use crate::edit::context;
use crate::edit::object_header;
use crate::message_type::MessageType;
use crate::reader::OpenedSource;
use crate::source::BytesSource;
use crate::superblock;
use crate::type_builders;
use crate::vl_data::VlElement;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

use super::{AttrOp, CompactAttrFixup, EditedAttrs};

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
fn attribute_plans_preserve_native_values_at_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(false, true)] dense: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("attributes.h5");
    {
        let native = file::libhdf5_create_with_widths(&path, offsets, lengths, 512);
        attr::write_unicode_array(&native, "original", &["alpha", "", "gamma"]);
        native
            .new_attr::<u32>()
            .create("counter")
            .unwrap()
            .write_scalar(&17)
            .unwrap();
    }
    let mut bytes = std::fs::read(&path).unwrap();
    let located = superblock::read(&BytesSource::new(bytes.as_slice())).unwrap();
    let widths = located.widths;
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let root = addresses
        .resolve(located.superblock.root_group_address)
        .unwrap()
        .get();
    let mut edited = {
        let source = BytesSource::new(bytes.as_slice());
        let region = WriteEngine::gather_oh_messages(&source, root, addresses, widths).unwrap();
        assert!(!super::region_uses_dense_attrs(&region).unwrap());
        let source = OpenedSource::for_test(&source, addresses);
        let slots = super::read_attr_slots(&source, &region).unwrap().0;
        assert_eq!(
            slots["original"].attribute.message().raw_data.len(),
            3 * VlElement::serialized_size(offsets)
        );
        let mut operations = vec![AttrOp::Set {
            name: "added".into(),
            value: AttrValue::VarLenStringArray(vec!["delta".into(), String::new(), "ζ".into()]),
        }];
        if dense {
            operations.extend((0..8).map(|index| AttrOp::Set {
                name: format!("number_{index}"),
                value: AttrValue::U32(index),
            }));
        }
        super::plan_attr_ops(&source, &region, &operations.iter().collect::<Vec<_>>()).unwrap()
    };
    match edited.attrs {
        EditedAttrs::RebuildCompact(mut compact) => {
            assert!(!dense);
            for CompactAttrFixup { range, collections } in compact.fixups {
                let addresses = append_collections(&mut bytes, widths, addresses, &collections);
                type_builders::patch_vl_refs_with_width(
                    offsets,
                    &mut compact.records[range],
                    &addresses,
                )
                .unwrap();
            }
            edited.region.push_bytes(&compact.records);
        }
        EditedAttrs::Dense(mut prepared) => {
            assert!(dense);
            for (index, collections) in prepared.vl {
                let addresses = append_collections(&mut bytes, widths, addresses, &collections);
                type_builders::patch_vl_refs_with_width(
                    offsets,
                    &mut prepared.set.attrs[index].raw_data,
                    &addresses,
                )
                .unwrap();
            }
            assert_eq!(prepared.set.prepared.as_ref().unwrap().widths(), widths);
            let plan = crate::file_writer::finalize_dense_attrs(
                prepared.set.prepared.take().unwrap(),
                &prepared.set.attrs,
            )
            .unwrap();
            let start = u64::try_from(bytes.len()).unwrap();
            let blob = plan
                .build(addresses.relative(PhysicalAddress::new(start)).unwrap())
                .unwrap();
            append_checked(&mut bytes, widths, addresses, &blob.blob);
            edited
                .region
                .push(MessageType::ATTRIBUTE_INFO, &blob.attr_info_message);
        }
        EditedAttrs::Compact => panic!("the operations must prepare new attributes"),
    }
    assert_eq!(
        super::region_uses_dense_attrs(&edited.region).unwrap(),
        dense
    );
    let header = object_header::build_v2_object_header(&edited.region).unwrap();
    let start = u64::try_from(bytes.len()).unwrap();
    append_checked(&mut bytes, widths, addresses, &header);
    let mut block = located.superblock;
    block.root_group_address = addresses.relative(PhysicalAddress::new(start)).unwrap();
    block.eof_address = u64::try_from(bytes.len()).unwrap();
    let encoded = hdf5_pure_format::__private::serialize_superblock(&block).unwrap();
    bytes[512..512 + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, &bytes).unwrap();
    {
        let native = file::libhdf5_open(&path);
        assert_eq!(
            attr::unicode_array(&native, "original"),
            ["alpha", "", "gamma"]
        );
        assert_eq!(attr::unicode_array(&native, "added"), ["delta", "", "ζ"]);
        assert_eq!(
            native
                .attr("counter")
                .unwrap()
                .read_scalar::<u32>()
                .unwrap(),
            17
        );
        if dense {
            for index in 0..8 {
                assert_eq!(
                    native
                        .attr(&format!("number_{index}"))
                        .unwrap()
                        .read_scalar::<u32>()
                        .unwrap(),
                    index
                );
            }
        }
    }
    let reopened = crate::File::from_bytes(bytes).unwrap();
    assert_eq!(
        reopened.root().attrs().unwrap()["added"],
        AttrValue::VarLenStringArray(vec!["delta".into(), String::new(), "ζ".into()])
    );
    reopened.close().unwrap();
    directory.close().unwrap();
}

fn append_collections(
    bytes: &mut Vec<u8>,
    widths: FormatWidths,
    addresses: AddressContext,
    collections: &[Vec<u8>],
) -> Vec<u64> {
    collections
        .iter()
        .map(|collection| {
            let start = u64::try_from(bytes.len()).unwrap();
            append_checked(bytes, widths, addresses, collection);
            addresses
                .relative(PhysicalAddress::new(start))
                .unwrap()
                .get()
        })
        .collect()
}

fn append_checked(
    bytes: &mut Vec<u8>,
    widths: FormatWidths,
    addresses: AddressContext,
    contents: &[u8],
) {
    context::require_range(
        widths.offsets,
        addresses,
        FileRange::new(
            PhysicalAddress::new(u64::try_from(bytes.len()).unwrap()),
            u64::try_from(contents.len()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    bytes.extend_from_slice(contents);
}
