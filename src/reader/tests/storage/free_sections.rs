use std::iter;
use std::slice;

use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use rstest::rstest;
use test_util::file_space_info;
use test_util::object_header;
use test_util::widths::Widths;

use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::FreeSectionCoordinates;
use crate::FreeSections;
use crate::Layout;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::ReusableSpaceStats;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::address::BaseAddressExt;
use crate::checksum;
use crate::source::{BytesSource, Source};

#[rstest]
#[case::default(None)]
#[case::nonpersisting(Some((FileSpaceStrategy::FsmAggr, false)))]
#[case::aggregate(Some((FileSpaceStrategy::Aggr, false)))]
#[case::unmanaged(Some((FileSpaceStrategy::None, false)))]
#[case::empty_persistence(Some((FileSpaceStrategy::FsmAggr, true)))]
fn persistence_configuration_is_distinct_from_empty_sections(
    #[case] strategy: Option<(FileSpaceStrategy, bool)>,
    #[values(false, true)] streaming: bool,
) {
    let mut builder = FileBuilder::new();
    if let Some((strategy, persist)) = strategy {
        builder
            .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(persist));
    }
    let bytes = builder.finish().unwrap();
    let file = if streaming {
        File::from_source(BytesSource::new(bytes)).unwrap()
    } else {
        File::from_bytes(bytes).unwrap()
    };
    let settings = strategy
        .map(|(strategy, persist)| FileSpaceSettings::new(strategy).with_persistence(persist));
    assert_eq!(file.file_space_settings().unwrap(), settings);
    assert_eq!(
        file.file_space_info().unwrap().is_some(),
        settings.is_some()
    );
    assert_eq!(
        file.file_space_settings().unwrap().unwrap_or_default(),
        settings.unwrap_or_default()
    );
    let result = file.persisted_free_sections().unwrap();
    assert_eq!(
        result.is_some(),
        strategy.is_some_and(|(_, persist)| persist)
    );
    if let Some(report) = result {
        assert_eq!(report.free_sections().section_count(), 0);
        assert_eq!(report.free_sections().free_bytes(), 0);
        assert_eq!(report.managers(), []);
        assert_eq!(report.manager_metadata_bytes(), 0);
    }
    let error = file.reusable_free_sections().unwrap_err();
    assert!(matches!(error, Error::ReadOnly), "{error:?}");
    let error = file.reusable_space_stats().unwrap_err();
    assert!(matches!(error, Error::ReadOnly), "{error:?}");
}

#[rstest]
#[case::flat_persistent(FileSpaceStrategy::FsmAggr, true, 0)]
#[case::flat_transient(FileSpaceStrategy::FsmAggr, false, 0)]
#[case::page_persistent(FileSpaceStrategy::Page, true, 0)]
#[case::page_transient(FileSpaceStrategy::Page, false, 0)]
#[case::aggregates(FileSpaceStrategy::Aggr, false, 0)]
#[case::disabled(FileSpaceStrategy::None, false, 0)]
#[case::flat_userblock(FileSpaceStrategy::FsmAggr, true, 512)]
#[case::page_userblock(FileSpaceStrategy::Page, true, 4_096)]
fn reusable_space_stats_match_session_transitions(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[case] userblock: u64,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Auto)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("summary-state.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(persist));
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder
        .create_dataset("log")
        .with_i32_data(&vec![1; 4_096])
        .with_chunks(&[4_096])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder
        .create_dataset("scratch")
        .with_i32_data(&vec![7; 16_384]);
    builder.create_dataset("keep").with_i32_data(&[9; 256]);
    builder.write(&path).unwrap();
    let options = FileAccessProperties::new()
        .with_memory_strategy(memory)
        .with_sync_policy(SyncPolicy::OnClose);
    let file = File::open_rw_with_options(&path, options).unwrap();
    let initial = assert_summary_and_sections(&file);
    file.root().delete("scratch").unwrap();
    assert_eq!(assert_summary_and_sections(&file), initial);
    file.commit().unwrap();
    let committed = assert_summary_and_sections(&file);
    if userblock == 0 && (strategy != FileSpaceStrategy::Page || persist) {
        file.dataset("log")
            .unwrap()
            .append(&vec![2; 4_096])
            .unwrap();
    } else {
        file.dataset("log")
            .unwrap()
            .append_staged(|builder| {
                builder.append_i32(&vec![2; 4_096]);
            })
            .unwrap();
        assert_eq!(assert_summary_and_sections(&file), committed);
        file.commit().unwrap();
    }
    assert!(!file.has_staged_edits());
    let appended = assert_summary_and_sections(&file);
    file.root()
        .create_dataset("marker", |builder| {
            builder.with_i32_data(&[3; 4]);
        })
        .unwrap();
    assert_eq!(assert_summary_and_sections(&file), appended);
    file.commit().unwrap();
    assert_summary_and_sections(&file);
    let closed = file.clone();
    file.close().unwrap();
    let settled = assert_summary_and_sections(&closed);
    drop(closed);
    let reopened = File::open_rw_with_options(&path, options).unwrap();
    let recovered = assert_summary_and_sections(&reopened);
    assert_eq!(recovered.logical_size(), settled.logical_size());
    if persist {
        assert_eq!(
            recovered.reusable_free_bytes(),
            settled.reusable_free_bytes()
        );
    }
}

#[rstest]
fn stored_and_reusable_views_have_distinct_coordinates(
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(0, 4096)] userblock: u64,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Auto)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("free-sections.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.write(&path).unwrap();
    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(memory),
        )
        .unwrap();
        assert!(matches!(
            file.persisted_free_sections().unwrap_err(),
            Error::FreeSectionInspectionUnsupported
        ));
        file.root().delete("b").unwrap();
        assert_eq!(
            file.reusable_free_sections().unwrap().free_bytes(),
            file.reusable_space_stats().unwrap().reusable_free_bytes()
        );
        file.commit().unwrap();
        let typed = file.reusable_free_sections().unwrap();
        assert_eq!(
            typed.free_bytes(),
            file.reusable_space_stats().unwrap().reusable_free_bytes()
        );
        assert!(typed.sections().iter().all(|s| s.coordinates()
            == FreeSectionCoordinates::Absolute
            && s.address() >= userblock));
    }
    let before_inspection = std::fs::read(&path).unwrap();
    let buffered = File::open(&path).unwrap();
    let report = buffered.persisted_free_sections().unwrap().unwrap();
    assert_eq!(
        report,
        File::open_streaming(&path)
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
    );
    assert_eq!(std::fs::read(&path).unwrap(), before_inspection);
    assert!(report.manager_metadata_bytes() > 0);
    assert_eq!(
        report.manager_metadata_bytes(),
        report
            .managers()
            .iter()
            .map(|manager| manager.metadata_bytes())
            .sum::<u64>()
    );
    assert!(report.free_sections().section_count() > 0);
    assert!(
        report
            .free_sections()
            .sections()
            .iter()
            .all(|s| s.coordinates() == FreeSectionCoordinates::Stored)
    );
    assert_eq!(
        report
            .managers()
            .iter()
            .map(|m| m.free_sections().free_bytes())
            .sum::<u64>(),
        report.free_sections().free_bytes()
    );
}

#[rstest]
fn reusable_sections_follow_commits_and_immediate_appends(
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session-timing.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder
        .create_dataset("log")
        .with_i32_data(&[1; 4_096])
        .with_chunks(&[4_096])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder
        .create_dataset("scratch")
        .with_i32_data(&vec![7; 16_384]);
    builder.create_dataset("keep").with_i32_data(&[9; 4]);
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let opening_extension = assert_current_file_space_metadata(&file);
    let layout = file.dataset("scratch").unwrap().layout().unwrap();
    let Layout::Contiguous {
        address: Some(start),
        size,
    } = layout
    else {
        panic!("{layout:?}");
    };
    assert_eq!(size, 65_536);
    let scratch = start..start + size;
    let intersections = |report: &FreeSections| {
        report
            .sections()
            .iter()
            .filter_map(|section| {
                let start = section.address().max(scratch.start);
                let end = section.end_address().min(scratch.end);
                (start < end).then_some(start..end)
            })
            .collect::<Vec<_>>()
    };
    let before = file.reusable_free_sections().unwrap();
    assert_eq!(intersections(&before), []);
    file.root().delete("scratch").unwrap();
    assert!(file.has_staged_edits());
    assert_eq!(file.reusable_free_sections().unwrap(), before);
    assert_eq!(assert_current_file_space_metadata(&file), opening_extension);
    file.commit().unwrap();
    assert_ne!(assert_current_file_space_metadata(&file), opening_extension);
    assert_eq!(
        intersections(&file.reusable_free_sections().unwrap()),
        slice::from_ref(&scratch),
    );
    file.dataset("log").unwrap().append(&[2; 4_096]).unwrap();
    assert_current_file_space_metadata(&file);
    assert!(!file.has_staged_edits());
    let chunks = file.dataset("log").unwrap().chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        (chunks[1].address, chunks[1].storage_size),
        (scratch.start, 16_384)
    );
    assert_eq!(
        intersections(&file.reusable_free_sections().unwrap()),
        iter::once(scratch.start + 16_384..scratch.end).collect::<Vec<_>>(),
    );
    file.root()
        .create_dataset("replacement", |builder| {
            builder.with_i32_data(&[3; 32]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_current_file_space_metadata(&file);
    let layout = file.dataset("replacement").unwrap().layout().unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = layout
    else {
        panic!("{layout:?}");
    };
    assert!(address <= opening_extension && opening_extension < address + size);
    assert_eq!(
        file.dataset("replacement").unwrap().read_i32().unwrap(),
        [3; 32]
    );
}

fn assert_current_file_space_metadata(file: &File) -> u64 {
    let bytes = file
        .inner
        .with_source(|source| source.read_exact_at(0, usize::try_from(source.len()).unwrap()))
        .unwrap();
    let snapshot = File::from_bytes(bytes).unwrap();
    assert_eq!(
        file.file_space_info().unwrap(),
        snapshot.file_space_info().unwrap()
    );
    assert_eq!(
        file.file_space_settings().unwrap(),
        snapshot.file_space_settings().unwrap()
    );
    assert_eq!(
        file.file_space_settings().unwrap(),
        Some(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true))
    );
    snapshot
        .superblock()
        .superblock_extension_address
        .unwrap()
        .get()
}

#[rstest]
fn malformed_file_space_info_is_reported(#[values(false, true)] streaming: bool) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("malformed-fsinfo.h5");
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let extension = file
        .superblock()
        .superblock_extension_address
        .unwrap()
        .get();
    drop(file);
    let mut body =
        file_space_info::non_persistent_version_1(Widths::EIGHT, file_space_info::PAGE, 1, 4096);
    body[0] = 9;
    file_space_info::replace_message(&path, extension, &body);
    let file = if streaming {
        File::open_streaming(&path).unwrap()
    } else {
        File::open(&path).unwrap()
    };
    let error = file.file_space_info().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::UnsupportedFileSpaceInfoVersion(9))
        ),
        "{error:?}"
    );
    let error = file.file_space_settings().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::UnsupportedFileSpaceInfoVersion(9))
        ),
        "{error:?}"
    );
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
    let error = file.persisted_free_sections().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::UnsupportedFileSpaceInfoVersion(9))
        ),
        "{error:?}"
    );
}

#[rstest]
#[case::invalid_page_size(0, FormatError::InvalidFileSpacePageSize(0))]
#[case::oversized_page_size(1 << 31, FormatError::InvalidFileSpacePageSize(1 << 31))]
#[case::invalid_extension_version(4096, FormatError::InvalidObjectHeaderVersion(9))]
fn invalid_file_space_metadata_returns_the_format_error(
    #[case] page_size: u64,
    #[case] expected: FormatError,
    #[values(false, true)] streaming: bool,
) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let mut bytes = builder.finish().unwrap();
    let extension = usize::try_from(
        File::from_bytes(bytes.clone())
            .unwrap()
            .superblock()
            .superblock_extension_address
            .unwrap()
            .get(),
    )
    .unwrap();
    if matches!(expected, FormatError::InvalidObjectHeaderVersion(_)) {
        bytes[extension + 4] = 9;
    } else {
        let body = file_space_info::non_persistent_version_1(
            Widths::EIGHT,
            file_space_info::PAGE,
            1,
            page_size,
        );
        object_header::v2::replace_message(
            &mut bytes,
            extension,
            object_header::MessageType::FILE_SPACE_INFO,
            &body,
        );
    }
    let file = if streaming {
        File::from_source(BytesSource::new(bytes)).unwrap()
    } else {
        File::from_bytes(bytes).unwrap()
    };
    let Error::Format(error) = file.file_space_info().unwrap_err() else {
        panic!("expected a File Space Info format error");
    };
    assert_eq!(error, expected);
    let Error::Format(error) = file.file_space_settings().unwrap_err() else {
        panic!("expected a file-space settings format error");
    };
    assert_eq!(error, expected);
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
}

#[rstest]
fn an_extension_without_file_space_info_has_no_recorded_settings(
    #[values(false, true)] streaming: bool,
) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    let mut bytes = builder.finish().unwrap();
    let extension = usize::try_from(
        File::from_bytes(bytes.clone())
            .unwrap()
            .superblock()
            .superblock_extension_address
            .unwrap()
            .get(),
    )
    .unwrap();
    object_header::v2::replace_record(
        &mut bytes,
        extension,
        object_header::MessageType::FILE_SPACE_INFO,
        object_header::Message::nil(0),
    );
    let file = if streaming {
        File::from_source(BytesSource::new(bytes)).unwrap()
    } else {
        File::from_bytes(bytes).unwrap()
    };
    assert_eq!(file.file_space_info().unwrap(), None);
    assert_eq!(file.file_space_settings().unwrap(), None);
}

#[test]
fn unreadable_extension_returns_a_source_error_without_preventing_dataset_reads() {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let bytes = builder.finish().unwrap();
    let extension = File::from_bytes(bytes.clone())
        .unwrap()
        .superblock()
        .superblock_extension_address
        .unwrap()
        .get();
    let file = File::from_source(UnreadableExtension {
        source: BytesSource::new(bytes),
        extension,
    })
    .unwrap();
    let Error::Format(error) = file.file_space_info().unwrap_err() else {
        panic!("expected an extension source error");
    };
    assert_eq!(error, FormatError::Source("unreadable extension".into()));
    let Error::Format(error) = file.file_space_settings().unwrap_err() else {
        panic!("expected an extension source error");
    };
    assert_eq!(error, FormatError::Source("unreadable extension".into()));
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
}

struct UnreadableExtension {
    source: BytesSource<Vec<u8>>,
    extension: u64,
}

impl Source for UnreadableExtension {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        if offset == self.extension {
            return Err(FormatError::Source("unreadable extension".into()));
        }
        self.source.read_at(offset, buf)
    }
}

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, false, 0)]
#[case::persistent_fsm_aggr(FileSpaceStrategy::FsmAggr, true, 128)]
#[case::page(FileSpaceStrategy::Page, false, 0)]
#[case::persistent_page(FileSpaceStrategy::Page, true, 128)]
#[case::aggr(FileSpaceStrategy::Aggr, true, 128)]
#[case::none(FileSpaceStrategy::None, true, 128)]
fn recorded_settings_match_the_writer_configuration(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[case] threshold: u64,
    #[values(512, 8192)] page_size: u64,
    #[values(false, true)] userblock: bool,
    #[values(false, true)] streaming: bool,
) {
    let page_size = FileSpacePageSize::try_from(page_size).unwrap();
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(persist)
            .with_threshold(threshold)
            .with_page_size(page_size),
    );
    if userblock {
        builder.with_userblock(UserblockSize::try_from(8192).unwrap());
    }
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let bytes = builder.finish().unwrap();
    let file = if streaming {
        File::from_source(BytesSource::new(bytes)).unwrap()
    } else {
        File::from_bytes(bytes).unwrap()
    };
    let expected = FileSpaceSettings::new(strategy)
        .with_persistence(persist)
        .with_threshold(threshold)
        .with_page_size(page_size);
    assert_eq!(file.file_space_settings().unwrap(), Some(expected));
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(info.strategy, expected.strategy());
    assert_eq!(info.persist, expected.persist());
    assert_eq!(info.threshold, expected.threshold());
    assert_eq!(info.page_size, expected.page_size());
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
}

#[rstest]
#[case::header(false)]
#[case::section_list(true)]
fn corrupted_manager_checksums_return_the_decode_error(
    #[case] section_list: bool,
    #[values(false, true)] streaming: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corrupt-manager-checksum.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.with_userblock(UserblockSize::try_from(512).unwrap());
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("b").unwrap();
        file.commit().unwrap();
        file.close().unwrap();
    }
    let file = File::open(&path).unwrap();
    let report = file.persisted_free_sections().unwrap().unwrap();
    let base = file.superblock().base_address;
    let widths = FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    };
    let address = usize::try_from(base.absolute(report.managers()[0].address()).unwrap()).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let header = FreeSpaceManagerHeader::parse(widths, &bytes[address..]).unwrap();
    let (at, length) = if section_list {
        (
            usize::try_from(base.absolute(header.section_list_addr()).unwrap()).unwrap(),
            usize::try_from(header.section_list_used()).unwrap(),
        )
    } else {
        (
            address,
            usize::try_from(hdf5_pure_format::__private::free_space_manager_header_len(
                widths,
            ))
            .unwrap(),
        )
    };
    let checksum_len = size_of::<u32>();
    let computed = checksum::jenkins_lookup3(&bytes[at..at + length - checksum_len]);
    bytes[at + length - checksum_len] ^= 1;
    let expected = computed ^ 1;
    drop(file);
    let file = if streaming {
        File::from_source(BytesSource::new(bytes)).unwrap()
    } else {
        File::from_bytes(bytes).unwrap()
    };
    let error = file.persisted_free_sections().unwrap_err();
    assert!(
        matches!(error,
        Error::Format(FormatError::ChecksumMismatch { expected: stored, computed: actual })
            if stored == expected && actual == computed),
        "{error:?}"
    );
}

#[rstest]
fn relocated_sources_preserve_stored_section_coordinates(
    #[values(0, 1024)] signature: usize,
    #[values(false, true)] streaming: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("relocated-free-sections.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(512).unwrap());
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("b").unwrap();
        file.commit().unwrap();
    }
    let expected = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap();
    let expected_settings = File::open(&path).unwrap().file_space_settings().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let mut relocated = vec![0; signature];
    relocated.extend_from_slice(&bytes[512..]);
    let file = if streaming {
        File::from_source(BytesSource::new(relocated)).unwrap()
    } else {
        File::from_bytes(relocated).unwrap()
    };
    assert_eq!(file.persisted_free_sections().unwrap(), expected);
    assert_eq!(file.file_space_settings().unwrap(), expected_settings);
}

fn assert_summary_and_sections(file: &File) -> ReusableSpaceStats {
    let summary = file.reusable_space_stats().unwrap();
    assert_eq!(summary.logical_size(), file.file_size());
    let sections = file.reusable_free_sections().unwrap();
    test_util_hdf5::assert_free_sections_consistent!(
        &summary,
        &sections,
        FreeSectionCoordinates::Absolute
    );
    summary
}
