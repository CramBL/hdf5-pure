use core::num::NonZeroU16;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileCreateProperties;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::UserblockSize;
use hdf5_pure::{
    CompactReversion, Error, File, FileBuilder, FormatError, LinkPhaseChange, ReadSeekSource,
    Source,
};
use rstest::rstest;
use test_util::group_info as group_info_bytes;
use test_util::group_info::{HeaderVersion, Userblock};
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::reader::ReadBackend;

const THRESHOLDS: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<12>()
    .min_dense::<8>()
    .build();
const ONE_WAY: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<12>()
    .disable_compact_reversion()
    .build();
const EQUAL: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<1>()
    .min_dense::<1>()
    .build();
const MAXIMUM: LinkPhaseChange = LinkPhaseChange::builder()
    .max_compact::<65_535>()
    .min_dense::<65_535>()
    .build();

#[rstest]
#[case::default(LinkPhaseChange::default(), 8, 6)]
#[case::thresholds(THRESHOLDS, 12, 8)]
#[case::one_way(ONE_WAY, 12, 0)]
#[case::always_dense(LinkPhaseChange::AlwaysDense, 0, 0)]
#[case::equal(EQUAL, 1, 1)]
#[case::maximum(MAXIMUM, 65_535, 65_535)]
fn completed_policies_have_effective_thresholds(
    #[case] policy: LinkPhaseChange,
    #[case] max_compact: u16,
    #[case] min_dense: u16,
) {
    assert_eq!(
        (policy.max_compact(), policy.min_dense()),
        (max_compact, min_dense)
    );
    match policy {
        LinkPhaseChange::AlwaysDense => assert_eq!((max_compact, min_dense), (0, 0)),
        LinkPhaseChange::Thresholds(thresholds) => {
            assert_eq!(
                thresholds.max_compact(),
                NonZeroU16::new(max_compact).unwrap()
            );
            assert_eq!(
                thresholds.compact_reversion(),
                match NonZeroU16::new(min_dense) {
                    Some(min_dense) => CompactReversion::Below(min_dense),
                    None => CompactReversion::Disabled,
                }
            );
        }
    }
}

#[rstest]
#[case::absent(None, LinkPhaseChange::default())]
#[case::default(Some([0, 0].as_slice()), LinkPhaseChange::default())]
#[case::thresholds(Some([0, 1, 12, 0, 8, 0].as_slice()), THRESHOLDS)]
#[case::one_way(Some([0, 1, 12, 0, 0, 0].as_slice()), ONE_WAY)]
#[case::always_dense(Some([0, 1, 0, 0, 0, 0].as_slice()), LinkPhaseChange::AlwaysDense)]
#[case::estimates(Some([0, 2, 7, 0, 20, 0].as_slice()), LinkPhaseChange::default())]
#[case::both(Some([0, 3, 12, 0, 8, 0, 7, 0, 20, 0].as_slice()), THRESHOLDS)]
#[case::maximum(Some([0, 1, 255, 255, 255, 255].as_slice()), MAXIMUM)]
fn group_inspection_reads_effective_policy(
    #[case] body: Option<&[u8]>,
    #[case] expected: LinkPhaseChange,
    #[values(HeaderVersion::V1, HeaderVersion::V2)] version: HeaderVersion,
    #[values(Userblock::Absent, Userblock::Bytes512)] userblock: Userblock,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let messages = body
        .map(|body| Message::new(MessageType::GROUP_INFO, body))
        .into_iter()
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("policy.h5");
    std::fs::write(&path, group_info_bytes::file(&messages, version, userblock)).unwrap();
    let file = backend.open(&path).unwrap();
    assert_eq!(file.root().link_phase_change().unwrap(), expected);
    assert_eq!(
        file.group("child").unwrap().link_phase_change().unwrap(),
        expected
    );
    file.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::missing_prefix(&[], FormatError::InvalidGroupInfoLength { expected: 2, actual: 0 })]
#[case::version(&[1, 0], FormatError::InvalidGroupInfoVersion(1))]
#[case::flags(&[0, 0x80], FormatError::InvalidGroupInfoFlags(0x80))]
#[case::short_thresholds(&[0, 1, 12, 0], FormatError::InvalidGroupInfoLength { expected: 6, actual: 4 })]
#[case::short_estimates(&[0, 2, 7, 0], FormatError::InvalidGroupInfoLength { expected: 6, actual: 4 })]
#[case::extra_byte(&[0, 0, 0], FormatError::InvalidGroupInfoLength { expected: 2, actual: 3 })]
#[case::inverted(&[0, 1, 3, 0, 4, 0], FormatError::InvalidGroupInfoThresholds { max_compact: 3, min_dense: 4 })]
#[case::zero_maximum(&[0, 1, 0, 0, 1, 0], FormatError::InvalidGroupInfoThresholds { max_compact: 0, min_dense: 1 })]
fn malformed_group_info_reports_its_format_error(
    #[case] body: &[u8],
    #[case] expected: FormatError,
    #[values(ReadBackend::Bytes, ReadBackend::Source)] backend: ReadBackend,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid-policy.h5");
    std::fs::write(
        &path,
        group_info_bytes::file(
            &[Message::new(MessageType::GROUP_INFO, body)],
            HeaderVersion::V2,
            Userblock::Absent,
        ),
    )
    .unwrap();
    let file = backend.open(&path).unwrap();
    let error = file.root().link_phase_change().unwrap_err();
    let Error::Format(actual) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(actual, expected);
    let error = file
        .group("child")
        .unwrap()
        .link_phase_change()
        .unwrap_err();
    let Error::Format(actual) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(actual, expected);
    file.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::legacy(HeaderVersion::V1)]
#[case::modern(HeaderVersion::V2)]
fn duplicate_group_info_is_rejected(#[case] version: HeaderVersion) {
    let message = Message::new(MessageType::GROUP_INFO, &[0, 0]);
    let file = File::from_bytes(group_info_bytes::file(
        &[message.clone(), message],
        version,
        Userblock::Absent,
    ))
    .unwrap();
    let error = file.root().link_phase_change().unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::DuplicateGroupInfo)),
        "{error:?}"
    );
}

#[rstest]
#[case::source(FormatError::Source("policy source failed".into()))]
#[case::short_source(FormatError::UnexpectedSourceEof { expected: 512, available: 128 })]
#[case::resource_limit(FormatError::ResourceLimit { limit: 4, required: 6 })]
fn policy_inspection_preserves_source_errors(#[case] expected: FormatError) {
    let fail = Arc::new(AtomicBool::new(false));
    let file = File::from_source(FailingSource {
        source: ReadSeekSource::new(Cursor::new(group_info_bytes::file(
            &[],
            HeaderVersion::V2,
            Userblock::Absent,
        )))
        .unwrap(),
        fail: Arc::clone(&fail),
        error: expected.clone(),
    })
    .unwrap();
    fail.store(true, Ordering::Relaxed);
    let error = file.root().link_phase_change().unwrap_err();
    let Error::Format(actual) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(actual, expected);
}

#[test]
fn staged_group_policy_is_unavailable_until_commit() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged-policy.h5");
    FileBuilder::new().write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let group = file.root().create_group("pending").unwrap();
    let error = group.link_phase_change().unwrap_err();
    let Error::NotCommitted(path) = error else {
        panic!("expected NotCommitted, got {error:?}");
    };
    assert_eq!(path, "pending");
    file.commit().unwrap();
    assert_eq!(
        group.link_phase_change().unwrap(),
        LinkPhaseChange::default()
    );
    drop(group);
    file.close().unwrap();
    directory.close().unwrap();
}

#[test]
fn policy_inspection_rejects_a_root_without_group_messages() {
    let mut bytes = group_info_bytes::file(&[], HeaderVersion::V2, Userblock::Absent);
    let root = usize::try_from(test_util::superblock::root_header_address(&bytes, 0)).unwrap();
    test_util::object_header::v2::replace_record(
        &mut bytes,
        root,
        MessageType::LINK_INFO,
        Message::nil(0),
    );
    test_util::object_header::v2::replace_record(
        &mut bytes,
        root,
        MessageType::LINK,
        Message::nil(0),
    );
    let file = File::from_bytes(bytes).unwrap();
    let error = file.root().link_phase_change().unwrap_err();
    let Error::NotAGroup(path) = error else {
        panic!("expected NotAGroup, got {error:?}");
    };
    assert_eq!(path, "");
}

#[test]
fn root_creation_policy_is_const_and_replacing_properties_restores_defaults() {
    const PROPERTIES: FileCreateProperties =
        FileCreateProperties::new().with_root_group_link_phase_change(THRESHOLDS);
    assert_eq!(PROPERTIES.root_group_link_phase_change(), THRESHOLDS);
    let mut builder = FileBuilder::new();
    builder.with_create_properties(PROPERTIES);
    builder.with_create_properties(FileCreateProperties::new());
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(
        file.root().link_phase_change().unwrap(),
        LinkPhaseChange::default()
    );
}

#[test]
fn detached_and_callback_groups_keep_independent_creation_policies() {
    let mut builder = FileBuilder::new();
    let mut parent = builder.create_group("parent");
    parent.with_link_phase_change(ONE_WAY);
    let mut child = parent.create_group("detached");
    child.with_link_phase_change(LinkPhaseChange::AlwaysDense);
    child.with_link_phase_change(THRESHOLDS);
    parent.add_group(child.finish());
    parent
        .create_group_with("callback", |group| {
            group.with_link_phase_change(LinkPhaseChange::AlwaysDense);
            Ok(())
        })
        .unwrap();
    builder.add_group(parent.finish());
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    assert_eq!(
        file.root().link_phase_change().unwrap(),
        LinkPhaseChange::default()
    );
    assert_eq!(
        file.group("parent").unwrap().link_phase_change().unwrap(),
        ONE_WAY
    );
    assert_eq!(
        file.group("parent/detached")
            .unwrap()
            .link_phase_change()
            .unwrap(),
        THRESHOLDS
    );
    assert_eq!(
        file.group("parent/callback")
            .unwrap()
            .link_phase_change()
            .unwrap(),
        LinkPhaseChange::AlwaysDense
    );
}

#[test]
fn owned_file_creation_keeps_the_root_policy_after_commit_and_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("root-policy.h5");
    let file = File::create_with_options(
        &path,
        FileCreateProperties::new().with_root_group_link_phase_change(THRESHOLDS),
        FileAccessProperties::new(),
    )
    .unwrap();
    assert_eq!(file.root().link_phase_change().unwrap(), THRESHOLDS);
    let child = file.root().create_group("child").unwrap();
    drop(child);
    file.commit().unwrap();
    assert_eq!(file.root().link_phase_change().unwrap(), THRESHOLDS);
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().link_phase_change().unwrap(), THRESHOLDS);
    file.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::default(LinkPhaseChange::default())]
#[case::thresholds(THRESHOLDS)]
#[case::one_way(ONE_WAY)]
#[case::always_dense(LinkPhaseChange::AlwaysDense)]
fn configured_group_headers_match_between_buffered_and_streaming_output(
    #[case] policy: LinkPhaseChange,
    #[values(false, true)] paged: bool,
) {
    let build = || {
        let mut properties = FileCreateProperties::new()
            .with_userblock(UserblockSize::try_from(4096).unwrap())
            .with_root_group_link_phase_change(policy);
        if paged {
            properties = properties.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
            );
        }
        let mut builder = FileBuilder::new();
        builder.with_create_properties(properties);
        for index in 0..13 {
            builder
                .create_dataset(&format!("root_{index}"))
                .with_i32_data(&[index]);
        }
        builder
            .create_group_with("nested", |group| {
                group.with_link_phase_change(policy);
                for index in 0..13 {
                    group
                        .create_dataset(&format!("child_{index}"))
                        .with_i32_data(&[index]);
                }
                Ok(())
            })
            .unwrap();
        builder
    };
    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();
    assert_eq!(streamed, buffered);
}

struct FailingSource {
    source: ReadSeekSource<Cursor<Vec<u8>>>,
    fail: Arc<AtomicBool>,
    error: FormatError,
}

impl Source for FailingSource {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        if self.fail.load(Ordering::Relaxed) {
            Err(self.error.clone())
        } else {
            self.source.read_at(offset, buffer)
        }
    }
}
