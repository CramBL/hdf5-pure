//! Object header continuation messages through the public read entry points, from a buffer
//! and from a source.

use std::io::Cursor;

use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure::ReadSeekSource;
use rstest::rstest;
use test_util::bytes;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2 as v2_bytes;
use test_util::widths::Widths;

#[rstest]
#[case::buffered(false)]
#[case::source(true)]
fn a_continuation_to_the_undefined_address_is_rejected(#[case] streaming: bool) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_f64_data(&[1.0])
        .set_attr(MARKER_ATTRIBUTE, AttrValue::I64(7));
    let mut image = builder.finish().unwrap();
    let marker_at = image
        .windows(MARKER_ATTRIBUTE.len())
        .position(|window| window == MARKER_ATTRIBUTE.as_bytes())
        .unwrap();
    let header_at = *bytes::signature_offsets(&image[..marker_at], v2_bytes::SIGNATURE)
        .last()
        .unwrap();
    v2_bytes::replace_record(
        &mut image,
        header_at,
        MessageType::ATTRIBUTE,
        Message::undefined_continuation(64, Widths::EIGHT),
    );
    let file = if streaming {
        File::from_source(ReadSeekSource::new(Cursor::new(image)).unwrap()).unwrap()
    } else {
        File::from_bytes(image).unwrap()
    };

    let err = file.dataset("d").unwrap_err();
    assert!(
        matches!(
            err,
            Error::Format(FormatError::UndefinedContinuationAddress)
        ),
        "{err:?}"
    );
}

/// The name of the attribute the test writes on the dataset, which it searches the image for to
/// find the dataset's object header.
const MARKER_ATTRIBUTE: &str = "continuation_marker";
