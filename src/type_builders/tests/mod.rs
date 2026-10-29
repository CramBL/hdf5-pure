use core::num::NonZeroUsize;

use rstest::rstest;

use crate::CompoundTypeBuilder;
use crate::DeflateLevel;
use crate::VariableLengthType;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::datatype::{CharacterSet, Datatype};
use crate::file_writer::FileWriter;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::ObjectRefPatch;
use crate::type_builders::ObjectRefTarget;
use crate::type_builders::{self, AttrValue, DatasetBuilder, VlStringElement};
use crate::{Error, File, FileBuilder, FormatError};

mod attr_value_accessors;
#[cfg(feature = "std")]
mod attr_value_display;
mod fixed_strings;

#[rstest]
#[case::direct_reference(false)]
#[case::embedded_reference(true)]
fn chunked_reference_patches_preserve_embedded_values(#[case] embedded: bool) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("probe")
        .with_path_references(&["target"]);
    let targets = vec![
        ObjectRefTarget::Path(ObjectPathBuf::parse("target")),
        ObjectRefTarget::Raw(StoredAddress::new(0)),
        ObjectRefTarget::Raw(StoredAddress::new(u64::MAX)),
    ];
    let dataset = builder.create_dataset("refs");
    if embedded {
        let datatype = CompoundTypeBuilder::new()
            .field("marker", type_builders::make_i32_type())
            .field("reference", type_builders::make_object_reference_type())
            .build()
            .unwrap();
        let raw = [7i32, 11, 13]
            .into_iter()
            .flat_map(|value| value.to_le_bytes().into_iter().chain([0; 8]))
            .collect();
        let patches = targets
            .into_iter()
            .enumerate()
            .map(|(index, target)| ObjectRefPatch {
                byte_offset: index * 12 + 4,
                target,
            })
            .collect();
        dataset.with_embedded_object_references(datatype, raw, 3, patches);
    } else {
        dataset.with_object_references(targets);
    }
    dataset
        .with_chunks(&[2])
        .with_shuffle()
        .with_deflate(DeflateLevel::Six);
    builder.create_dataset("target").with_i32_data(&[42]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let address = file.dataset("probe").unwrap().read_raw().unwrap();
    let expected_addresses = [address.as_slice(), &[0; 8], &u64::MAX.to_le_bytes()].concat();
    let raw = file.dataset("refs").unwrap().read_raw().unwrap();
    let expected = if embedded {
        [7i32, 11, 13]
            .into_iter()
            .zip(expected_addresses.as_chunks::<8>().0)
            .flat_map(|(marker, reference)| {
                marker
                    .to_le_bytes()
                    .into_iter()
                    .chain(reference.iter().copied())
            })
            .collect::<Vec<_>>()
    } else {
        expected_addresses
    };
    assert_eq!(raw, expected);
}

#[rstest]
#[case::empty(0, Ok(0))]
#[case::u32_limit(u64::from(u32::MAX), Ok(u32::MAX))]
#[case::above_u32_limit(
    u64::from(u32::MAX) + 1,
    Err(FormatError::ValueTooLargeForRepresentation {
        value: 4_294_967_296,
        target: "u32",
    })
)]
fn a_descriptor_count_uses_the_full_u32_range(
    #[case] count: u64,
    #[case] expected: Result<u32, FormatError>,
) {
    assert_eq!(type_builders::vl_element_count(count), expected);
}

#[rstest]
#[case::byte_elements(1, 8)]
#[case::two_byte_elements(2, 4)]
#[case::four_byte_elements(4, 2)]
fn sequence_counts_are_measured_in_base_elements(#[case] width: usize, #[case] count: u32) {
    let elements = [
        VlStringElement::Bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]),
        VlStringElement::Null,
        VlStringElement::Bytes(Vec::new()),
    ];
    let (refs, staging) =
        type_builders::stage_vl_elements(&elements, NonZeroUsize::new(width).unwrap()).unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(&count.to_le_bytes());
    expected.extend_from_slice(&0u64.to_le_bytes());
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.extend_from_slice(&[0; 16]);
    expected.extend_from_slice(&0u32.to_le_bytes());
    expected.extend_from_slice(&0u64.to_le_bytes());
    expected.extend_from_slice(&2u32.to_le_bytes());
    assert_eq!(refs, expected);
    assert_eq!(staging.patch_offsets, vec![0, 32]);
    assert_eq!(staging.collections.len(), 1);
}

#[test]
fn a_sequence_setter_preserves_its_base_element_count() {
    let mut builder = DatasetBuilder::new("sequence");
    let datatype =
        Datatype::VariableLength(VariableLengthType::sequence(type_builders::make_u32_type()));
    builder
        .with_vlen_sequence_elements(
            datatype,
            &[VlStringElement::Bytes(vec![1, 0, 0, 0, 2, 0, 0, 0])],
        )
        .unwrap();
    assert_eq!(
        builder.data.unwrap().unwrap(),
        vec![2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    );
}

#[test]
fn a_staged_count_error_reaches_the_public_file_builder() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("text").stage_vlen_result(
        type_builders::make_vlen_string_type(CharacterSet::Utf8),
        1,
        type_builders::vl_element_count(4_294_967_296).map(|_| panic!("expected a count overflow")),
    );
    let error = builder.finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error")
    };
    assert_eq!(
        error,
        FormatError::ValueTooLargeForRepresentation {
            value: 4_294_967_296,
            target: "u32",
        }
    );
}

#[rstest]
#[case::utf8_strings(AttrValue::VarLenStringArray(vec!["é".into(), String::new()]))]
#[case::ascii_strings(AttrValue::VarLenAsciiStringArray(vec!["ab".into(), String::new()]))]
#[case::ascii_char_arrays(AttrValue::VarLenAsciiCharArray(vec!["ab".into(), String::new()]))]
fn attribute_descriptors_count_payload_bytes(#[case] value: AttrValue) {
    let message = type_builders::build_attr_message("text", &value).unwrap();
    assert_eq!(
        message.raw_data,
        vec![
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
            0, 0, 0
        ],
    );
}

#[rstest]
#[case::populated_shape(false)]
#[case::empty_shape(true)]
fn a_staged_count_error_reaches_finish_even_for_an_empty_shape(#[case] empty: bool) {
    let mut writer = FileWriter::new();
    let dataset = writer.create_dataset("text");
    dataset.with_vlen_strings(&["previous"]);
    dataset.stage_vlen_result(
        type_builders::make_vlen_string_type(CharacterSet::Utf8),
        1,
        type_builders::vl_element_count(4_294_967_296).map(|_| panic!("expected a count overflow")),
    );
    if empty {
        dataset.with_shape(&[0]);
    }
    assert!(dataset.vl_string_staging.is_none());
    assert!(dataset.reference_targets.is_none());
    assert_eq!(
        writer.finish().unwrap_err(),
        FormatError::ValueTooLargeForRepresentation {
            value: 4_294_967_296,
            target: "u32",
        },
    );
}

#[rstest]
#[case::numeric_data(false)]
#[case::string_data(true)]
fn replacing_element_data_clears_a_staged_count_error(#[case] strings: bool) {
    let mut writer = FileWriter::new();
    let dataset = writer.create_dataset("text");
    dataset.stage_vlen_result(
        type_builders::make_vlen_string_type(CharacterSet::Utf8),
        1,
        type_builders::vl_element_count(4_294_967_296).map(|_| panic!("expected a count overflow")),
    );
    if strings {
        dataset.with_vlen_strings(&["é"]);
    } else {
        dataset.with_u8_data(&[7]);
    }
    let file = File::from_bytes(writer.finish().unwrap()).unwrap();
    let dataset = file.dataset("text").unwrap();
    if strings {
        assert_eq!(dataset.read_string().unwrap(), vec!["é"]);
    } else {
        assert_eq!(dataset.read_raw().unwrap(), vec![7]);
    }
}

#[test]
fn a_string_datatype_mismatch_preserves_previous_staging() {
    let mut builder = DatasetBuilder::new("text");
    builder.with_vlen_strings(&["previous"]);
    let data = builder.data.clone();
    let offsets = builder
        .vl_string_staging
        .as_ref()
        .unwrap()
        .patch_offsets
        .clone();
    let error = builder.with_vlen_string_elements(type_builders::make_u8_type(), &[]);
    let Err(error) = error else {
        panic!("expected TypeMismatch")
    };
    assert_eq!(
        error,
        FormatError::TypeMismatch {
            expected: "VariableLength string",
            actual: "non-VariableLength string",
        }
    );
    assert_eq!(builder.data, data);
    assert_eq!(
        builder.vl_string_staging.as_ref().unwrap().patch_offsets,
        offsets
    );
}
