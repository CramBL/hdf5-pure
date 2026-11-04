use alloc::collections::BTreeSet;

use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::global_heap;
use test_util::image::Image;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;

use crate::Datatype;
use crate::UserblockSize;
use crate::VariableLengthType;
use crate::attribute::AttributeMessage;
use crate::dataspace::Dataspace;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::link_info::LinkInfoMessage;
use crate::width::{LengthWidth, OffsetWidth};

#[rstest]
#[case::strings(global_heap_fixture::vlen_string(), &[0])]
#[case::sequences(global_heap_fixture::vlen_sequence(), &[0])]
#[case::compound(global_heap_fixture::with_integer(global_heap_fixture::vlen_string()), &[4])]
#[case::array_of_compounds(
    global_heap_fixture::array_of(global_heap_fixture::with_integer(global_heap_fixture::vlen_sequence())),
    &[4, 24]
)]
fn parsed_attributes_collect_each_embedded_reference(
    #[case] datatype: Datatype,
    #[case] slots: &[usize],
    #[values(UserblockSize::NONE, UserblockSize::MIN)] base: UserblockSize,
) {
    let stride = usize::try_from(datatype.type_size()).unwrap();
    let mut raw = vec![0; stride * 2];
    let mut expected = BTreeSet::from([7]);
    for element in 0..2 {
        for (index, &slot) in slots.iter().enumerate() {
            let collection = 4_096 + u64::try_from(element * 1_024 + index * 256).unwrap();
            let reference = global_heap::reference(5, collection, 1, Widths::EIGHT);
            let start = element * stride + slot;
            raw[start..start + reference.len()].copy_from_slice(&reference);
            expected.insert(base.get() + collection);
        }
    }
    let attribute = parsed_attribute(datatype, 2, raw);
    with_engine(base, |engine| {
        let mut referenced = BTreeSet::from([7]);
        assert!(
            engine
                .collect_parsed_attribute_collections(&attribute, &mut referenced)
                .unwrap()
        );
        assert_eq!(referenced, expected);
    });
}

#[rstest]
#[case::two_byte_undefined_value(65_535, 66_047)]
#[case::four_byte_undefined_value(4_294_967_295, 4_294_967_807)]
#[case::highest_physical_address(u64::MAX - 512, u64::MAX)]
fn parsed_attribute_addresses_remain_physical_after_resolution(
    #[case] stored: u64,
    #[case] physical: u64,
) {
    let raw = global_heap::reference(5, stored, 1, Widths::EIGHT);
    let attribute = parsed_attribute(global_heap_fixture::vlen_string(), 1, raw);
    with_engine(UserblockSize::MIN, |engine| {
        let mut referenced = BTreeSet::new();
        assert!(
            engine
                .collect_parsed_attribute_collections(&attribute, &mut referenced)
                .unwrap()
        );
        assert_eq!(referenced, BTreeSet::from([physical]));
    });
}

#[rstest]
#[case::fixed_width(Datatype::Opaque { size: 16, tag: Vec::new() }, true)]
#[case::object_reference(global_heap_fixture::object_reference(), false)]
#[case::nested_sequence(
    Datatype::VariableLength(VariableLengthType::sequence(global_heap_fixture::vlen_sequence())),
    false
)]
fn parsed_attributes_distinguish_absent_and_unsupported_references(
    #[case] datatype: Datatype,
    #[case] complete: bool,
) {
    let mut raw = global_heap::reference(5, 4_096, 1, Widths::EIGHT);
    raw.resize(usize::try_from(datatype.type_size()).unwrap(), 0);
    let attribute = parsed_attribute(datatype, 1, raw);
    with_engine(UserblockSize::NONE, |engine| {
        let mut referenced = BTreeSet::from([7]);
        assert_eq!(
            engine
                .collect_parsed_attribute_collections(&attribute, &mut referenced)
                .unwrap(),
            complete
        );
        assert_eq!(referenced, BTreeSet::from([7]));
    });
}

#[rstest]
#[case::truncated_element(
    vec![0; 15],
    FormatError::UnexpectedEof { expected: 16, available: 15 }
)]
#[case::undefined_nonempty_address(
    global_heap::reference(1, u64::MAX, 1, Widths::EIGHT),
    FormatError::VlDataError("non-empty VL element has an undefined heap address".into())
)]
#[case::missing_object_index(
    global_heap::reference(1, 4_096, 0, Widths::EIGHT),
    FormatError::GlobalHeapObjectNotFound { collection_address: 4_608, index: 0 }
)]
#[case::oversized_object_index(
    global_heap::reference(1, 4_096, 65_536, Widths::EIGHT),
    FormatError::VlDataError("global heap object index 65536 does not fit u16".into())
)]
#[case::address_overflow(
    global_heap::reference(1, u64::MAX - 1, 1, Widths::EIGHT),
    FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 512 }
)]
fn parsed_attributes_return_descriptor_errors(#[case] raw: Vec<u8>, #[case] expected: FormatError) {
    let attribute = parsed_attribute(global_heap_fixture::vlen_string(), 1, raw);
    with_engine(UserblockSize::MIN, |engine| {
        let mut referenced = BTreeSet::from([7]);
        assert_eq!(
            engine
                .collect_parsed_attribute_collections(&attribute, &mut referenced)
                .unwrap_err(),
            expected
        );
        assert_eq!(referenced, BTreeSet::from([7]));
    });
}

#[rstest]
#[case::undefined_address(u64::MAX)]
#[case::zero_address(0)]
fn parsed_attributes_skip_null_references(#[case] address: u64) {
    let raw = global_heap::reference(0, address, 0, Widths::EIGHT);
    let attribute = parsed_attribute(global_heap_fixture::vlen_string(), 1, raw);
    with_engine(UserblockSize::MIN, |engine| {
        let mut referenced = BTreeSet::new();
        assert!(
            engine
                .collect_parsed_attribute_collections(&attribute, &mut referenced)
                .unwrap()
        );
        assert_eq!(referenced, BTreeSet::new());
    });
}

#[rstest]
#[case::strings(global_heap_fixture::vlen_string(), 1, 0, None)]
#[case::sequences(global_heap_fixture::vlen_sequence(), 1, 0, None)]
#[case::array(
    global_heap_fixture::array_of(global_heap_fixture::vlen_string()),
    2,
    0,
    None
)]
#[case::compound(
    global_heap_fixture::with_integer(global_heap_fixture::vlen_string()),
    1,
    4,
    Some(20)
)]
fn parsed_attributes_use_stored_reference_strides(
    #[case] datatype: Datatype,
    #[case] slots: usize,
    #[case] first: usize,
    #[case] declared_stride: Option<usize>,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[values(UserblockSize::NONE, UserblockSize::MIN)] base: UserblockSize,
    #[values(false, true)] truncated: bool,
) {
    let widths = Widths::new(usize::from(offsets.get()), usize::from(lengths.get()));
    let reference_size = 4 + widths.offset + 4;
    let stride = declared_stride.unwrap_or(slots * reference_size);
    let mut raw = vec![0; stride * 2];
    let mut expected = BTreeSet::from([7]);
    for element in 0..2 {
        for slot in 0..slots {
            let collection = 4096 + u64::try_from(element * 1024 + slot * 256).unwrap();
            let descriptor = global_heap::reference(5, collection, 1, widths);
            let at = element * stride + first + slot * reference_size;
            raw[at..at + reference_size].copy_from_slice(&descriptor);
            expected.insert(base.get() + collection);
        }
    }
    if truncated {
        raw.pop().unwrap();
    }
    let attribute = parsed_attribute(datatype, 2, raw);
    with_width_engine(base, widths, |engine| {
        let mut referenced = BTreeSet::from([7]);
        let result = engine.collect_parsed_attribute_collections(&attribute, &mut referenced);
        if truncated {
            assert_eq!(
                result.unwrap_err(),
                FormatError::UnexpectedEof {
                    expected: stride * 2,
                    available: stride * 2 - 1,
                }
            );
            assert_eq!(referenced, BTreeSet::from([7]));
        } else {
            assert!(result.unwrap());
            assert_eq!(referenced, expected);
        }
    });
}

fn parsed_attribute(datatype: Datatype, count: u64, raw_data: Vec<u8>) -> AttributeMessage {
    AttributeMessage {
        name: "labels".into(),
        datatype,
        dataspace: Dataspace {
            space_type: format::DataspaceType::Simple,
            rank: 1,
            dimensions: vec![count],
            max_dimensions: None,
        },
        raw_data,
        datatype_location: format::DatatypeLocation::Inline,
    }
}

fn with_engine(base: UserblockSize, inspect: impl FnOnce(&WriteEngine)) {
    with_width_engine(base, Widths::EIGHT, inspect);
}

fn with_width_engine(base: UserblockSize, widths: Widths, inspect: impl FnOnce(&WriteEngine)) {
    let link_info = LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    let root = Header::new()
        .message(Message::new(
            MessageType::LINK_INFO,
            &link_info
                .serialize(OffsetWidth::try_from(u8::try_from(widths.offset).unwrap()).unwrap())
                .unwrap(),
        ))
        .build();
    let mut image = Image::new();
    let base_at = usize::try_from(base.get()).unwrap();
    image.place(base_at + ROOT_HEADER, &root);
    image.place(
        base_at,
        &Superblock::new(widths)
            .base_address(base.get())
            .eof_address(u64::try_from(image.as_bytes().len()).unwrap())
            .root_header_address(u64::try_from(ROOT_HEADER).unwrap())
            .build(),
    );
    let path = temp::temp_path("parsed_attribute_collections.h5");
    std::fs::write(&path, image.build()).unwrap();
    let engine = edit_test_util::open_engine(&path);
    inspect(&engine);
    drop(engine);
}

const ROOT_HEADER: usize = 64;
