use std::ops::Range;
use std::path::Path;

use rstest::rstest;
use test_util::disk_log::{self, FailurePoint, IoEvent};
use test_util::file_space_info;
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageType};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{Fixture, Form};

use super::postpublication::{Control, Fault, RestorePrefix, assert_quarantined, root_range};
use super::prepublication::{Part, Prefix};
use super::{append_count_user, assert_values, stage, type_header};
use crate::{
    AttrValue, File, FileAccessProperties, FileLocking, FileSpaceStrategy, MemoryStrategy,
};

#[rstest]
#[case::restored_count(Fault::Count(Part::Count, Prefix::Half))]
#[case::restored_checksum(Fault::Count(Part::Checksum, Prefix::Half))]
#[case::restored_barrier_before(Fault::Barrier(FailurePoint::Before))]
#[case::restored_barrier_after(Fault::Barrier(FailurePoint::After))]
#[case::restore_write_zero(Fault::RestoreWrite(RestorePrefix::Zero))]
#[case::restore_write_complete(Fault::RestoreWrite(RestorePrefix::Complete))]
#[case::restore_barrier_before(Fault::RestoreBarrier(FailurePoint::Before))]
#[case::restore_barrier_after(Fault::RestoreBarrier(FailurePoint::After))]
fn page_count_failures_distinguish_confirmed_restoration_from_quarantine(#[case] fault: Fault) {
    let path = test_util::temp::temp_path("page-count-failure.h5");
    let (target, extension) = page_fixture(&path);
    let read_only = File::open(&path).unwrap();
    assert!(read_only.persisted_free_sections().unwrap().is_none());
    read_only.close().unwrap();
    let original = std::fs::read(&path).unwrap();
    assert!(original.len().is_multiple_of(PAGE));
    let control_path = test_util::temp::temp_path("page-count-control.h5");
    std::fs::write(&control_path, &original).unwrap();
    let properties = FileAccessProperties::new()
        .with_locking(FileLocking::Disabled)
        .with_memory_strategy(MemoryStrategy::Mirrored)
        .with_page_buffer_size(0);
    let reference = File::open_rw_with_options(&control_path, properties).unwrap();
    let info = reference.file_space_info().unwrap().unwrap();
    assert_eq!(
        (
            info.strategy,
            info.persist,
            info.page_size.get(),
            info.threshold
        ),
        (FileSpaceStrategy::Page, false, PAGE as u64, 1)
    );
    stage(&reference, 1, true, false);
    disk_log::start();
    reference.commit().unwrap();
    let recording = disk_log::finish();
    let published = std::fs::read(&control_path).unwrap();
    let start = usize::try_from(target).unwrap();
    assert_eq!(
        &published[start..start + type_header(2).len()],
        type_header(2)
    );
    assert_values(&reference, 1, false);
    let control = Control::new(&recording, target);
    reference.close().unwrap();
    let file = File::open_rw_with_options(&path, properties).unwrap();
    let before = std::fs::read(&path).unwrap();
    stage(&file, 1, true, false);
    disk_log::start();
    let event = control.inject(fault);
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    control.assert_failed(fault, event, &failed);
    fault.assert_error(&error);
    assert!(!file.has_staged_edits());
    let after = std::fs::read(&path).unwrap();
    let mut replay = before;
    for operation in &failed.operations {
        operation.apply(&mut replay);
    }
    assert_eq!(replay, after);
    let expected = type_header(
        if matches!(fault, Fault::RestoreWrite(RestorePrefix::Zero)) {
            2
        } else {
            3
        },
    );
    assert_eq!(&after[start..start + expected.len()], expected);
    assert_eq!(&after[extension.clone()], &original[extension]);
    let IoEvent::Write { requested, .. } = recording.events[control.publication] else {
        panic!("missing PAGE publication");
    };
    assert_eq!(&after[..requested], &published[..requested]);
    let root = root_range(&published);
    assert!(root.start >= original.len());
    assert_eq!(&after[root.clone()], &published[root.clone()]);
    let sections = file.reusable_free_sections().unwrap();
    assert!(sections.sections().iter().all(|section| {
        section.address().get() >= original.len() as u64
            && (section.end_address().get() <= root.start as u64
                || section.address().get() >= root.end as u64)
    }));
    assert_values(&file, 1, false);
    let partial = matches!(fault, Fault::RestoreWrite(_) | Fault::RestoreBarrier(_));
    if partial {
        assert_quarantined(&file, &path, &after);
        drop(file);
        File::clear_status_flags(&path).unwrap();
    } else {
        file.root()
            .set_attr("recovered", AttrValue::I64(41))
            .unwrap();
        assert!(file.has_staged_edits());
        file.commit().unwrap();
        assert!(!file.has_staged_edits());
        assert_eq!(
            file.root().attrs().unwrap()["recovered"],
            AttrValue::I64(41)
        );
        assert_values(&file, 1, false);
        file.close().unwrap();
    }
    let reopened = File::open(&path).unwrap();
    assert_values(&reopened, 1, false);
    if !partial {
        assert_eq!(
            reopened.root().attrs().unwrap()["recovered"],
            AttrValue::I64(41)
        );
    }
    reopened.close().unwrap();
}

fn page_fixture(path: &Path) -> (u64, Range<usize>) {
    let mut fixture = Fixture::new();
    let target = append_count_user(&mut fixture, &type_header(3), 0, true);
    let messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_f64_type(), &17_f64.to_le_bytes());
    fixture.object("value", messages);
    assert!(fixture.image.as_bytes().len() < PAGE);
    let raw = fixture.image.append_aligned(&[], PAGE);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &vec![7_i32.to_le_bytes(); PAGE / 4].concat(),
    );
    assert_eq!(fixture.image.as_bytes().len(), raw + PAGE);
    fixture.object("dead", messages);
    fixture.write(path);
    let mut image = std::fs::read(path).unwrap();
    let extension = image.len();
    image.extend(
        Header::new()
            .message(Message::new(
                MessageType::FILE_SPACE_INFO,
                &file_space_info::non_persistent_version_1(
                    Widths::EIGHT,
                    file_space_info::PAGE,
                    1,
                    PAGE as u64,
                ),
            ))
            .build(),
    );
    let end = image.len();
    assert_eq!(extension / PAGE, end / PAGE);
    image.resize(image.len().next_multiple_of(PAGE), 0);
    let superblock = Superblock::new(Widths::EIGHT)
        .version(3)
        .root_header_address(Fixture::ROOT)
        .extension_address(extension as u64)
        .eof_address(image.len() as u64)
        .build();
    image[..superblock.len()].copy_from_slice(&superblock);
    std::fs::write(path, image).unwrap();
    (target, extension..end)
}

const PAGE: usize = 4_096;
