use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FloatingPointType;
use hdf5_pure::ObjectReferenceCountUpdateIssue;
use rstest::rstest;
use test_util::datatype;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

#[rstest]
#[case::zero(0)]
#[case::undercounted(1)]
fn inconsistent_original_counts_reject_a_commit_and_preserve_reading(#[case] observed: u32) {
    let path = test_util::temp::temp_path("undercounted-alias.h5");
    let mut fixture = Fixture::new();
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &number(), &17_f64.to_le_bytes());
    messages.push(Message::reference_count(observed));
    let target = fixture.append_header(messages);
    fixture.link("a", target);
    fixture.link("b", target);
    fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    file.root().delete("a").unwrap();
    file.root()
        .create_dataset("new", |builder| {
            builder.with_i32_data(&[29]);
        })
        .unwrap();
    let error = file.commit().unwrap_err();
    let Error::InvalidObjectReferenceCount {
        address,
        observed: actual,
        minimum,
    } = error
    else {
        panic!("expected original count contradiction, got {error:?}")
    };
    assert_eq!((address, actual, minimum), (target, observed, 2));
    assert_eq!(file.dataset("a").unwrap().read_f64().unwrap(), [17.0]);
    assert_eq!(file.dataset("b").unwrap().read_f64().unwrap(), [17.0]);
    assert_eq!(file.dataset("new").unwrap().shape().unwrap(), [1]);
    assert_eq!(file.reusable_free_sections().unwrap(), free);
    let Error::InvalidObjectReferenceCount {
        address,
        observed: actual,
        minimum,
    } = file.commit().unwrap_err()
    else {
        panic!("expected staged work to remain rejected")
    };
    assert_eq!((address, actual, minimum), (target, observed, 2));
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("a").unwrap().read_f64().unwrap(), [17.0]);
    assert_eq!(file.dataset("b").unwrap().read_f64().unwrap(), [17.0]);
    test_util_hdf5::assertion::path_not_found(file.dataset("new").unwrap_err(), "new");
}

#[rstest]
#[case::overflow(Some(u32::MAX))]
#[case::implicit(None)]
#[case::original_zero(Some(0))]
fn unsupported_copy_count_changes_preserve_the_original_file(#[case] original: Option<u32>) {
    let path = test_util::temp::temp_path("unsupported-copy-count.h5");
    let mut fixture = Fixture::new();
    let mut messages = vec![Message::new(MessageType::DATATYPE, &datatype::f64_le())];
    messages.extend(original.map(Message::reference_count));
    let target = fixture.append_header(messages);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &number(), &17_f64.to_le_bytes());
    *messages
        .iter_mut()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap() = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(target),
    )
    .with_flags(MessageFlags::SHARED);
    fixture.object("source", messages);
    fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.copy("source", "copy").unwrap();
    let error = file.commit().unwrap_err();
    match (original, error) {
        (Some(0), Error::ObjectReferenceCountUpdateUnsupported { address, issue }) => assert_eq!(
            (address, issue),
            (target, ObjectReferenceCountUpdateIssue::OriginalZero)
        ),
        (
            Some(original),
            Error::ObjectReferenceCountOverflow {
                address,
                original: actual,
                added,
                removed,
            },
        ) => assert_eq!((address, actual, added, removed), (target, original, 1, 0)),
        (None, Error::ObjectReferenceCountUpdateUnsupported { address, issue }) => assert_eq!(
            (address, issue),
            (target, ObjectReferenceCountUpdateIssue::ImplicitSourceField)
        ),
        (_, error) => panic!("unexpected copy count failure: {error:?}"),
    }
    assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
    test_util_hdf5::assertion::path_not_found(file.dataset("copy").unwrap_err(), "copy");
}

fn number() -> Datatype {
    Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
        DatatypeByteOrder::LittleEndian,
    ))
}

#[test]
fn an_unknown_count_subject_preserves_storage_on_rejection() {
    let path = test_util::temp::temp_path("unsupported-count-subject.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"alpha")]);
    let messages = Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    let target = fixture.append_header(messages);
    fixture.link("unknown", target);
    let keep = Form::CompactDataset.messages(&mut fixture, &number(), &17_f64.to_le_bytes());
    fixture.object("keep", keep);
    fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    file.root().delete("unknown").unwrap();
    let Error::ObjectReferenceCountUpdateUnsupported { address, issue } =
        file.commit().unwrap_err()
    else {
        panic!("expected an unsupported retained count")
    };
    assert_eq!(
        (address, issue),
        (target, ObjectReferenceCountUpdateIssue::RetainedZero)
    );
    assert_eq!(file.reusable_free_sections().unwrap(), free);
    assert_eq!(file.dataset("keep").unwrap().read_f64().unwrap(), [17.0]);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_f64().unwrap(), [17.0]);
}
