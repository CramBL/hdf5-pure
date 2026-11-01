use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::RepackOptions;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::GroupInfo;
use hdf5_pure_format::__private::ObjectHeader;
use rstest::rstest;
use test_util::group_info;
use test_util::group_info::HeaderVersion;
use test_util::group_info::Userblock;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::absent(None, &[0, 0])]
#[case::default(Some([0, 0].as_slice()), &[0, 0])]
#[case::thresholds(Some([0, 1, 12, 0, 8, 0].as_slice()), &[0, 1, 12, 0, 8, 0])]
#[case::one_way(Some([0, 1, 12, 0, 0, 0].as_slice()), &[0, 1, 12, 0, 0, 0])]
#[case::always_dense(Some([0, 1, 0, 0, 0, 0].as_slice()), &[0, 1, 0, 0, 0, 0])]
#[case::estimates(Some([0, 2, 7, 0, 20, 0].as_slice()), &[0, 2, 7, 0, 20, 0])]
#[case::default_estimates(Some([0, 2, 4, 0, 8, 0].as_slice()), &[0, 2, 4, 0, 8, 0])]
#[case::zero_estimates(Some([0, 2, 0, 0, 0, 0].as_slice()), &[0, 2, 0, 0, 0, 0])]
#[case::both(Some([0, 3, 12, 0, 8, 0, 7, 0, 20, 0].as_slice()), &[0, 3, 12, 0, 8, 0, 7, 0, 20, 0])]
#[case::maximum(Some([0, 3, 255, 255, 255, 255, 255, 255, 255, 255].as_slice()), &[0, 3, 255, 255, 255, 255, 255, 255, 255, 255])]
fn repack_preserves_independent_group_info_fields(
    #[case] body: Option<&[u8]>,
    #[case] expected: &[u8],
    #[values(HeaderVersion::V1, HeaderVersion::V2)] version: HeaderVersion,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.h5");
    let messages = body
        .map(|body| Message::new(MessageType::GROUP_INFO, body))
        .into_iter()
        .collect::<Vec<_>>();
    std::fs::write(
        &path,
        group_info::file(&messages, version, Userblock::Bytes512),
    )
    .unwrap();
    let source = backend.open(&path).unwrap();
    let mut bytes = Vec::new();
    hdf5_pure::repack_into(&source, &mut bytes, &RepackOptions::new()).unwrap();
    source.close().unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(file.root().groups().unwrap(), vec!["child"]);
    assert_eq!(
        file.group("child").unwrap().groups().unwrap(),
        Vec::<String>::new()
    );
    assert_group_info(&file, &bytes, "/", expected);
    let expected_policy = file.root().link_phase_change().unwrap();
    assert_eq!(
        file.group("child").unwrap().link_phase_change().unwrap(),
        expected_policy
    );
    if expected_policy.max_compact() > 0 {
        assert_group_info(&file, &bytes, "child", expected);
    }
    file.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::version(&[1, 0], FormatError::InvalidGroupInfoVersion(1))]
#[case::flags(&[0, 4], FormatError::InvalidGroupInfoFlags(4))]
#[case::short(&[0], FormatError::InvalidGroupInfoLength { expected: 2, actual: 1 })]
#[case::thresholds(&[0, 1, 8, 0, 9, 0], FormatError::InvalidGroupInfoThresholds { max_compact: 8, min_dense: 9 })]
#[case::estimates(&[0, 2, 7, 0], FormatError::InvalidGroupInfoLength { expected: 6, actual: 4 })]
fn repack_rejects_invalid_group_info_before_publication(
    #[case] body: &[u8],
    #[case] expected: FormatError,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    #[values(false, true)] root_only: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.h5");
    let destination = directory.path().join("destination.h5");
    std::fs::write(
        &path,
        group_info::file(
            &[Message::new(MessageType::GROUP_INFO, body)],
            HeaderVersion::V2,
            Userblock::Absent,
        ),
    )
    .unwrap();
    let source = backend.open(&path).unwrap();
    let options = if root_only {
        RepackOptions::new().drop_path("child")
    } else {
        RepackOptions::new()
    };
    let mut bytes = Vec::new();
    let error = hdf5_pure::repack_into(&source, &mut bytes, &options).unwrap_err();
    let Error::Format(actual) = error else {
        panic!("expected Format, got {error:?}")
    };
    assert_eq!(actual, expected);
    assert!(bytes.is_empty());
    let error = hdf5_pure::repack_to_path(&source, &destination, &options).unwrap_err();
    let Error::Format(actual) = error else {
        panic!("expected Format, got {error:?}")
    };
    assert_eq!(actual, expected);
    assert!(!destination.exists());
    source.close().unwrap();
    directory.close().unwrap();
}

#[test]
fn repack_rejects_duplicate_group_info_before_publication() {
    let message = Message::new(MessageType::GROUP_INFO, &[0, 0]);
    let source = File::from_bytes(group_info::file(
        &[message.clone(), message],
        HeaderVersion::V2,
        Userblock::Absent,
    ))
    .unwrap();
    let mut bytes = Vec::new();
    let error = hdf5_pure::repack_into(&source, &mut bytes, &RepackOptions::new()).unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::DuplicateGroupInfo)),
        "{error:?}"
    );
    assert!(bytes.is_empty());
}

fn assert_group_info(file: &File, bytes: &[u8], path: &str, expected: &[u8]) {
    let group = file.group(path).unwrap();
    let superblock = file.superblock();
    let header = ObjectHeader::parse(
        bytes,
        AccessMode::ReadOnly,
        usize::try_from(if path == "/" {
            superblock.root_group_address.get()
        } else {
            test_util_hdf5::storage::root_object_address(file, bytes, path).get()
        })
        .unwrap(),
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    let messages = header
        .messages
        .iter()
        .filter(|message| message.msg_type == hdf5_pure::MessageType::GROUP_INFO)
        .collect::<Vec<_>>();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].data, expected);
    let raw = GroupInfo::parse(expected).unwrap();
    let policy = group.link_phase_change().unwrap();
    assert_eq!(
        (policy.max_compact(), policy.min_dense()),
        (
            raw.phase_change().max_compact(),
            raw.phase_change().min_dense()
        )
    );
}
