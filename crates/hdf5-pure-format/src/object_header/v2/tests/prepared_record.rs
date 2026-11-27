use alloc::vec;
use core::cell::Cell;

use super::*;

#[test]
fn a_short_destination_keeps_the_encoder_from_writing() {
    let seen_len = Cell::new(None);
    let record = failing_record(&seen_len);
    let mut output = [0x5a; 7];
    assert_eq!(
        record.write_into(&mut output),
        Err(FormatError::UnexpectedEof {
            expected: 8,
            available: 7
        })
    );
    assert_eq!(seen_len.get(), None);
    assert_eq!(output, [0x5a; 7]);
}

#[test]
fn a_body_error_leaves_the_complete_destination_unchanged() {
    let seen_len = Cell::new(None);
    let record = failing_record(&seen_len);
    let mut output = [0x5a; 13];
    assert_eq!(record.write_into(&mut output[1..]), Err(INJECTED_ERROR));
    assert_eq!(seen_len.get(), Some(2));
    assert_eq!(output, [0x5a; 13]);
}

#[test]
fn a_body_error_restores_the_original_append_length_and_bytes() {
    let seen_len = Cell::new(None);
    let record = failing_record(&seen_len);
    let mut output = vec![0x5a, 0x4b, 0x3c];
    assert_eq!(record.append_to(&mut output), Err(INJECTED_ERROR));
    assert_eq!(seen_len.get(), Some(2));
    assert_eq!(output, [0x5a, 0x4b, 0x3c]);
}

fn failing_record(seen_len: &Cell<Option<usize>>) -> PreparedMessageRecord<FailingBody<'_>> {
    let layout = MessageRecordLayout::from_header_flags(TRACKED);
    layout
        .prepare_record(MessageFlags::CONSTANT, 0x1234, FailingBody(seen_len))
        .unwrap()
}

struct FailingBody<'a>(&'a Cell<Option<usize>>);

impl MessageBody for FailingBody<'_> {
    const MESSAGE_TYPE: MessageType = MessageType::GROUP_INFO;
}

impl Encode for FailingBody<'_> {
    type Error = FormatError;

    fn encoded_len(&self) -> usize {
        2
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        self.0.set(Some(output.len()));
        Err(INJECTED_ERROR)
    }
}

const TRACKED: u8 = test_util::object_header::v2::HeaderFlags::TRACKS_CREATION_ORDER.0;
const INJECTED_ERROR: FormatError = FormatError::InvalidGroupInfoVersion(u8::MAX);
