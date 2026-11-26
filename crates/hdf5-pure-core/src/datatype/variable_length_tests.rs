use rstest::rstest;

use super::VariableLengthStringType;
use crate::{CharacterSet, Datatype, FormatError, StringPadding};

#[rstest]
#[case::zero_width_string(Datatype::String {
    size: 0,
    padding: StringPadding::NullTerminate,
    charset: CharacterSet::Ascii,
}, STRING_CLASS)]
#[case::zero_width_opaque(Datatype::Opaque { size: 0, tag: Vec::new() }, OPAQUE_CLASS)]
#[case::zero_element_array(Datatype::Array {
    base_type: Box::new(Datatype::String {
        size: 1,
        padding: StringPadding::NullTerminate,
        charset: CharacterSet::Ascii,
    }),
    dimensions: vec![0],
}, ARRAY_CLASS)]
fn rejects_zero_sized_string_bases(#[case] base: Datatype, #[case] class: u8) {
    assert_eq!(
        VariableLengthStringType::with_base_type(
            StringPadding::NullTerminate,
            CharacterSet::Utf8,
            base,
        )
        .unwrap_err(),
        FormatError::ZeroSizedDatatype { class },
    );
}

// The format specification, version 4.0, defines these class codes in "The Datatype Message".
const STRING_CLASS: u8 = 0x03;
const OPAQUE_CLASS: u8 = 0x05;
const ARRAY_CLASS: u8 = 0x0A;
