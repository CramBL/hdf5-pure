use hdf5_pure::{
    CharacterSet, Datatype, Error, File, FileBuilder, FixedAscii, FormatError, H5Element,
    StringPadding,
};
use rstest::rstest;

#[rstest]
#[case::empty(b"", [0, 0, 0], "")]
#[case::short(b"ab", [b'a', b'b', 0], "ab")]
#[case::full(b"abc", *b"abc", "abc")]
#[case::embedded_null(b"a\0b", *b"a\0b", "a\0b")]
#[case::ascii_boundary(b"\x7f", [127, 0, 0], "\x7f")]
fn fixed_ascii_preserves_bytes_and_pads_short_input(
    #[case] input: &[u8],
    #[case] stored: [u8; 3],
    #[case] text: &str,
) {
    let value = FixedAscii::<3>::new(input).unwrap();
    assert_eq!(value.as_bytes(), &stored);
    assert_eq!(value.as_str(), text);
}

#[rstest]
#[case::non_ascii(b"a\xff", FormatError::InvalidAscii { index: 1, byte: 255 })]
#[case::utf8("é".as_bytes(), FormatError::InvalidAscii { index: 0, byte: 195 })]
#[case::too_long(b"abcd", FormatError::FixedStringTooLong { index: 0, len: 4, width: 3 })]
fn fixed_ascii_rejects_invalid_input(#[case] bytes: &[u8], #[case] expected: FormatError) {
    assert_eq!(FixedAscii::<3>::new(bytes).unwrap_err(), expected);
}

#[rstest]
#[case::contiguous_width_one(ascii::<1>(&["", "A", "\x7f"]), false)]
#[case::chunked_width_one(ascii::<1>(&["", "A", "\x7f"]), true)]
#[case::contiguous_width_four(ascii::<4>(&["ab", "", "wxyz", "a\0b"]), false)]
#[case::chunked_width_four(ascii::<4>(&["ab", "", "wxyz", "a\0b"]), true)]
#[case::contiguous_empty(Vec::<FixedAscii<7>>::new(), false)]
#[case::chunked_empty(Vec::<FixedAscii<7>>::new(), true)]
fn typed_ascii_preserves_elements_and_datatype<const N: usize>(
    #[case] values: Vec<FixedAscii<N>>,
    #[case] chunked: bool,
) {
    let mut builder = FileBuilder::new();
    let fill = FixedAscii::<N>::new("?").unwrap();
    let dataset = builder.create_dataset("values");
    dataset.with_data(&values).with_fill_value(fill);
    if chunked {
        dataset.with_chunks(&[3]);
    }
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape().unwrap(), [values.len() as u64]);
    assert_eq!(dataset.datatype().unwrap(), ascii_type(N as u32));
    assert_eq!(dataset.read::<FixedAscii<N>>().unwrap(), values);
    assert_eq!(dataset.fill_value::<FixedAscii<N>>().unwrap(), Some(fill));
    assert_eq!(
        dataset.read_raw().unwrap(),
        values
            .iter()
            .flat_map(|value| *value.as_bytes())
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::wrong_width(ascii_type(3), "String")]
#[case::utf8(Datatype::String { size: 2, padding: StringPadding::NullPad, charset: CharacterSet::Utf8 }, "String")]
#[case::null_terminate(Datatype::String { size: 2, padding: StringPadding::NullTerminate, charset: CharacterSet::Ascii }, "String")]
#[case::space_pad(Datatype::String { size: 2, padding: StringPadding::SpacePad, charset: CharacterSet::Ascii }, "String")]
#[case::byte_array(Datatype::Array { base_type: Box::new(hdf5_pure::make_u8_type()), dimensions: vec![2] }, "Array")]
#[case::numeric(hdf5_pure::make_u16_type(), "FixedPoint")]
fn typed_ascii_rejects_incompatible_datatypes(
    #[case] datatype: Datatype,
    #[case] actual: &'static str,
    #[values(0, 2)] count: usize,
) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("values").with_raw_data(
        datatype.clone(),
        vec![0; datatype.type_size() as usize * count],
        count as u64,
    );
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let expected = FormatError::TypeMismatch {
        expected: "null-padded ASCII string with matching width",
        actual,
    };
    assert_format_error(
        file.dataset("values")
            .unwrap()
            .read::<FixedAscii<2>>()
            .unwrap_err(),
        &expected,
    );
    assert_format_error(
        FixedAscii::<2>::convert_raw(&[], &datatype).unwrap_err(),
        &expected,
    );
}

#[test]
fn typed_ascii_rejects_non_ascii_stored_bytes_and_fill() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_raw_data(ascii_type(2), vec![b'a', 0, b'b', 255], 2)
        .with_fill_value([0u8, 128]);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_format_error(
        dataset.read::<FixedAscii<2>>().unwrap_err(),
        &FormatError::InvalidAscii {
            index: 3,
            byte: 255,
        },
    );
    assert_format_error(
        dataset.fill_value::<FixedAscii<2>>().unwrap_err(),
        &FormatError::InvalidAscii {
            index: 1,
            byte: 128,
        },
    );
}

#[test]
fn typed_ascii_rejects_incomplete_raw_elements() {
    assert_format_error(
        FixedAscii::<2>::convert_raw(b"abc", &ascii_type(2)).unwrap_err(),
        &FormatError::DataSizeMismatch {
            expected: 2,
            actual: 3,
        },
    );
}

#[test]
fn typed_ascii_rejects_zero_width_even_for_empty_input() {
    assert_eq!(
        FixedAscii::<0>::new("").unwrap_err(),
        FormatError::ZeroFixedStringWidth
    );
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_data::<FixedAscii<0>>(&[]);
    assert_format_error(
        builder.finish().unwrap_err(),
        &FormatError::ZeroFixedStringWidth,
    );
    assert_format_error(
        FixedAscii::<0>::convert_raw(&[], &ascii_type(1)).unwrap_err(),
        &FormatError::ZeroFixedStringWidth,
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn typed_ascii_rejects_unrepresentable_width_for_empty_input() {
    const WIDTH: usize = u32::MAX as usize + 1;
    let expected = FormatError::ValueTooLargeForRepresentation {
        value: WIDTH as u64,
        target: "u32",
    };
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_data::<FixedAscii<WIDTH>>(&[]);
    assert_format_error(builder.finish().unwrap_err(), &expected);
    assert_format_error(
        FixedAscii::<WIDTH>::convert_raw(&[], &ascii_type(1)).unwrap_err(),
        &expected,
    );
}

fn ascii<const N: usize>(values: &[&str]) -> Vec<FixedAscii<N>> {
    values
        .iter()
        .map(|value| FixedAscii::new(value).unwrap())
        .collect()
}

fn ascii_type(size: u32) -> Datatype {
    Datatype::String {
        size,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Ascii,
    }
}

fn assert_format_error(error: Error, expected: &FormatError) {
    let Error::Format(actual) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(&actual, expected);
}
