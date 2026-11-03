use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use rstest::rstest;
use test_util::checksum;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::widths::Widths;

use crate::error::FormatError;

use super::super::ObjectReferenceCount;
use super::super::ObjectReferenceCountField;
use super::super::ObjectReferenceCountUpdate;
use super::super::ObjectReferenceCountUpdateError;

#[rstest]
#[case::version_1(Form::Version1)]
#[case::plain(Form::Plain)]
#[case::tracked(Form::Tracked)]
#[case::prefix_fields(Form::Prefix)]
#[case::continuation(Form::Continuation)]
#[case::tracked_continuation(Form::TrackedContinuation)]
fn update_emits_only_the_count_and_its_matching_chunk_checksum(
    #[case] form: Form,
    #[values(0, 1, u32::MAX)] count: u32,
    #[values(0, 8_589_934_592)] bias: u64,
) {
    let (field, mut current) = fixture(form, bias);
    let unrelated = current.len() - if field.flags().is_some() { 5 } else { 1 };
    current[unrelated] = 0xA7;
    stamp(field, &mut current);
    let before = current.clone();
    let pointer = current.as_ptr();
    let update = field
        .update(FileRegion::new(field.chunk(), &current).unwrap(), count)
        .unwrap();
    assert_eq!(current, before);
    assert_eq!(current.as_ptr(), pointer);
    let writes = update.writes().collect::<Vec<_>>();
    assert_eq!(writes[0], (field.range(), count.to_le_bytes().as_slice()));
    assert_eq!(writes.len(), if field.flags().is_some() { 2 } else { 1 });
    let mut expected = before;
    let at = usize::try_from(field.range().start().get() - field.chunk().start().get()).unwrap();
    expected[at..at + 4].copy_from_slice(&count.to_le_bytes());
    stamp(field, &mut expected);
    if field.flags().is_some() {
        let checksum_at = expected.len() - checksum::CHECKSUM;
        assert_eq!(
            writes[1],
            (
                FileRange::new(
                    field
                        .chunk()
                        .start()
                        .checked_add(u64::try_from(checksum_at).unwrap())
                        .unwrap(),
                    4
                )
                .unwrap(),
                &expected[checksum_at..]
            )
        );
    }
    assert_eq!(apply(field, &current, &update), expected);
}

#[rstest]
#[case::shifted(Region::Shifted)]
#[case::short(Region::Short)]
#[case::parent(Region::Parent)]
fn update_requires_the_complete_associated_chunk(#[case] shape: Region) {
    let (field, mut bytes) = fixture(Form::Plain, 0);
    let provided = match shape {
        Region::Shifted => FileRange::new(
            field.chunk().start().checked_add(1).unwrap(),
            field.chunk().length(),
        )
        .unwrap(),
        Region::Short => {
            bytes.pop();
            FileRange::new(field.chunk().start(), field.chunk().length() - 1).unwrap()
        }
        Region::Parent => {
            bytes.push(0);
            FileRange::new(field.chunk().start(), field.chunk().length() + 1).unwrap()
        }
    };
    let before = bytes.clone();
    let error = field
        .update(FileRegion::new(provided, &bytes).unwrap(), 1)
        .unwrap_err();
    assert_eq!(
        error,
        ObjectReferenceCountUpdateError::Chunk {
            expected: field.chunk(),
            provided
        }
    );
    assert_eq!(bytes, before);
}

#[rstest]
#[case::version_1(Form::Version1)]
#[case::version_2(Form::Plain)]
#[case::continuation(Form::Continuation)]
fn update_reports_a_changed_current_count_without_emitting_bytes(#[case] form: Form) {
    let (field, mut bytes) = fixture(form, 0);
    let at = usize::try_from(field.range().start().get() - field.chunk().start().get()).unwrap();
    bytes[at..at + 4].copy_from_slice(&3_u32.to_le_bytes());
    stamp(field, &mut bytes);
    let before = bytes.clone();
    assert_eq!(
        field
            .update(FileRegion::new(field.chunk(), &bytes).unwrap(), 1)
            .unwrap_err(),
        ObjectReferenceCountUpdateError::Value {
            field: field.range().start(),
            expected: 2,
            actual: 3
        }
    );
    assert_eq!(bytes, before);
}

#[test]
fn count_field_cannot_overlap_its_chunk_checksum() {
    let (field, bytes) = fixture(Form::Plain, 0);
    let field = ObjectReferenceCountField {
        chunk: FileRange::new(
            field.chunk().start(),
            field.range().end().get() - field.chunk().start().get(),
        )
        .unwrap(),
        ..field
    };
    let bytes = &bytes[..usize::try_from(field.chunk().length()).unwrap()];
    assert_eq!(
        field
            .update(FileRegion::new(field.chunk(), bytes).unwrap(), 1)
            .unwrap_err(),
        ObjectReferenceCountUpdateError::Format(FormatError::InsufficientRegion {
            required_start: field.range().start().get(),
            required_length: field.range().length(),
            provided_start: field.chunk().start().get(),
            provided_length: field.chunk().length() - 4,
        })
    );
}

#[test]
fn update_preserves_the_configured_input_checksum_policy() {
    let (field, mut bytes) = fixture(Form::Plain, 0);
    let checksum_at = bytes.len() - checksum::CHECKSUM;
    bytes[checksum_at] ^= 1;
    let result = field.update(FileRegion::new(field.chunk(), &bytes).unwrap(), 1);
    #[cfg(feature = "checksum")]
    {
        let expected = u32::from_le_bytes(bytes[checksum_at..].try_into().unwrap());
        assert_eq!(
            result.unwrap_err(),
            ObjectReferenceCountUpdateError::Format(FormatError::ChecksumMismatch {
                expected,
                computed: checksum::lookup3(&bytes[..checksum_at])
            })
        );
    }
    #[cfg(not(feature = "checksum"))]
    {
        let output = apply(field, &bytes, &result.unwrap());
        assert_eq!(
            &output[checksum_at..],
            checksum::lookup3(&output[..checksum_at]).to_le_bytes()
        );
    }
}

fn fixture(form: Form, bias: u64) -> (ObjectReferenceCountField, Vec<u8>) {
    let tracked = matches!(form, Form::Tracked | Form::TrackedContinuation);
    let flags = if tracked {
        v2::HeaderFlags::TRACKS_CREATION_ORDER
    } else if matches!(form, Form::Prefix) {
        v2::HeaderFlags::STORES_TIMES | v2::HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE
    } else {
        v2::HeaderFlags::default()
    };
    let message = Message::new(
        MessageType::OBJECT_REFERENCE_COUNT,
        &[0, 2, 0, 0, 0, 0xA5, 0x5A],
    )
    .with_flags(MessageFlags::CONSTANT)
    .with_creation_order(19);
    let records = vec![
        Message::new(MessageType::UNKNOWN, &[0xD5; 16]),
        message,
        Message::nil(7),
    ];
    let child = matches!(form, Form::Continuation | Form::TrackedContinuation)
        .then(|| v2::continuation_chunk(&records, flags));
    let root = if matches!(form, Form::Version1) {
        v1::Header::new()
            .reference_count(2)
            .message(Message::nil(16))
            .build()
    } else if let Some(child) = &child {
        v2::Header::new()
            .flags(flags)
            .continuation(512, child, Widths::EIGHT)
            .build()
    } else {
        v2::Header::new().flags(flags).messages(records).build()
    };
    let observed = super::observe(&root, child.as_deref(), bias, 8).unwrap();
    let field = match observed {
        ObjectReferenceCount::Version1(field) | ObjectReferenceCount::Version2(field) => field,
        other => panic!("expected an explicit count field, got {other:?}"),
    };
    (field, child.unwrap_or(root))
}

fn apply(
    field: ObjectReferenceCountField,
    current: &[u8],
    update: &ObjectReferenceCountUpdate,
) -> Vec<u8> {
    let mut output = current.to_vec();
    for (range, bytes) in update.writes() {
        let start = usize::try_from(range.start().get() - field.chunk().start().get()).unwrap();
        output[start..start + bytes.len()].copy_from_slice(bytes);
    }
    output
}

fn stamp(field: ObjectReferenceCountField, bytes: &mut [u8]) {
    if field.flags().is_some() {
        let at = bytes.len() - checksum::CHECKSUM;
        let checksum = checksum::lookup3(&bytes[..at]);
        bytes[at..].copy_from_slice(&checksum.to_le_bytes());
    }
}

#[derive(Clone, Copy)]
enum Form {
    Continuation,
    Plain,
    Prefix,
    Tracked,
    TrackedContinuation,
    Version1,
}
#[derive(Clone, Copy)]
enum Region {
    Parent,
    Shifted,
    Short,
}
