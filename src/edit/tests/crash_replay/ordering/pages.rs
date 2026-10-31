//! Checks initialization bitmap publication after page preparation in allocated data blocks.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ExtensibleArrayCreationParameters;
use hdf5_pure_format::__private::ExtensibleArraySuperBlockRef;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Recording;
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;

use crate::Error;
use crate::File;
use crate::FormatError;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::access_mode::AccessMode;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::extensible_array;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source;
use crate::source::BytesSource;
use crate::width::FormatWidths;

use super::Buffering;
use super::Observation;
use super::WriteFailure;

#[rstest]
#[case::plain_durable(false, false, SyncPolicy::Always, Buffering::Session)]
#[case::filtered_durable(true, false, SyncPolicy::Always, Buffering::Unbuffered)]
#[case::page_durable(false, true, SyncPolicy::Always, Buffering::Unbuffered)]
#[case::filtered_page_durable(true, true, SyncPolicy::Always, Buffering::Session)]
#[case::plain_ordered(false, false, SyncPolicy::OnClose, Buffering::Unbuffered)]
#[case::filtered_ordered(true, false, SyncPolicy::OnClose, Buffering::Session)]
#[case::page_ordered(false, true, SyncPolicy::OnClose, Buffering::Session)]
#[case::filtered_page_ordered(true, true, SyncPolicy::OnClose, Buffering::Unbuffered)]
fn initialized_pages_follow_prepared_content_in_already_owned_blocks(
    #[case] filtered: bool,
    #[case] paged: bool,
    #[case] policy: SyncPolicy,
    #[case] buffering: Buffering,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("page.h5");
    let fixture = Fixture::prepare(&path, filtered, paged);
    let observed = fixture.record(&path, policy, buffering);
    let gate = fixture
        .page
        .discover(&observed.base, &observed.final_bytes, &observed.trace);
    assert_eq!(gate.require(&observed.trace, policy), Ok(()));
    for ownership in observed.gates() {
        ownership
            .parse(&observed.trace)
            .unwrap()
            .require(super::oracle_policy(policy))
            .unwrap();
    }
    fixture.assert_prefixes(&observed);
}

#[rstest]
#[case::missing_barrier(Control::Barrier, SyncPolicy::Always)]
#[case::failed_barrier(Control::Failed, SyncPolicy::Always)]
#[case::early_bitmap(Control::Reference, SyncPolicy::Always)]
#[case::ordered_early_bitmap(Control::Reference, SyncPolicy::OnClose)]
#[case::decoy_missing_barrier(Control::DecoyBarrier, SyncPolicy::Always)]
#[case::decoy_early_bitmap(Control::DecoyReference, SyncPolicy::Always)]
fn page_order_controls_reject_readable_completed_files(
    #[case] control: Control,
    #[case] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("control.h5");
    let fixture = Fixture::prepare(&path, true, true);
    let mut observed = fixture.record(&path, policy, Buffering::Unbuffered);
    let gate = fixture
        .page
        .discover(&observed.base, &observed.final_bytes, &observed.trace);
    gate.require(&observed.trace, policy).unwrap();
    if matches!(control, Control::DecoyBarrier | Control::DecoyReference) {
        let barrier = observed.trace.events[..gate.content]
            .iter()
            .rposition(|event| matches!(event, IoEvent::Synchronize { .. }))
            .unwrap();
        let bytes = source::bytes_in(&observed.base, fixture.page.range)
            .unwrap()
            .to_vec();
        let operation = observed.trace.operations.len();
        let offset = fixture.page.range.start().get();
        let requested = bytes.len();
        observed
            .trace
            .operations
            .push(DiskOp::Write { offset, bytes });
        observed.trace.events.insert(
            barrier,
            IoEvent::Write {
                offset,
                requested,
                written: requested,
                outcome: IoOutcome::Succeeded,
                operation: Some(operation),
            },
        );
    }
    let gate = fixture
        .page
        .discover(&observed.base, &observed.final_bytes, &observed.trace);
    match control {
        Control::DecoyReference | Control::Reference => {
            observed.trace.events.swap(gate.content, gate.reference)
        }
        Control::Barrier | Control::DecoyBarrier | Control::Failed => {
            let mut affected = 0;
            for index in (gate.content + 1..gate.reference).rev() {
                if let IoEvent::Synchronize { outcome, .. } = &mut observed.trace.events[index] {
                    affected += 1;
                    match control {
                        Control::Barrier | Control::DecoyBarrier => {
                            observed.trace.events.remove(index);
                        }
                        Control::Failed => *outcome = IoOutcome::Failed(ErrorKind::Other),
                        Control::DecoyReference | Control::Reference => {
                            panic!("the reference control changes write order")
                        }
                    }
                }
            }
            assert!(affected > 0);
        }
    }
    let altered = fixture
        .page
        .discover(&observed.base, &observed.final_bytes, &observed.trace);
    assert_eq!(
        altered.require(&observed.trace, policy),
        Err(match control {
            Control::Barrier | Control::DecoyBarrier => PageOrderError::Barrier,
            Control::Failed => PageOrderError::Failed,
            Control::DecoyReference | Control::Reference => PageOrderError::Reference,
        })
    );
    observed.assert_complete(&path);
}

#[rstest]
#[case::content_before(
    Target::Content,
    Fault::Write(WriteFailure::Before),
    SyncPolicy::Always
)]
#[case::content_prefix(
    Target::Content,
    Fault::Write(WriteFailure::Partial),
    SyncPolicy::Always
)]
#[case::content_after(Target::Content, Fault::Write(WriteFailure::After), SyncPolicy::Always)]
#[case::content_sync_before(Target::Barrier, Fault::Sync(FailurePoint::Before), SyncPolicy::Always)]
#[case::content_sync_after(Target::Barrier, Fault::Sync(FailurePoint::After), SyncPolicy::Always)]
#[case::bitmap_before(Target::Bitmap, Fault::Write(WriteFailure::Before), SyncPolicy::Always)]
#[case::bitmap_prefix(
    Target::Bitmap,
    Fault::Write(WriteFailure::Partial),
    SyncPolicy::Always
)]
#[case::bitmap_after(Target::Bitmap, Fault::Write(WriteFailure::After), SyncPolicy::Always)]
#[case::ordered_content(
    Target::Content,
    Fault::Write(WriteFailure::After),
    SyncPolicy::OnClose
)]
#[case::ordered_bitmap(Target::Bitmap, Fault::Write(WriteFailure::After), SyncPolicy::OnClose)]
fn page_initialization_errors_prevent_later_references_and_reuse(
    #[case] target: Target,
    #[case] fault: Fault,
    #[case] policy: SyncPolicy,
    #[values(Buffering::Unbuffered, Buffering::Session)] buffering: Buffering,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed.h5");
    let fixture = Fixture::prepare(&path, true, true);
    let observed = fixture.record(&path, policy, buffering);
    let gate = fixture
        .page
        .discover(&observed.base, &observed.final_bytes, &observed.trace);
    gate.require(&observed.trace, policy).unwrap();
    let target = match target {
        Target::Content => gate.content,
        Target::Bitmap => gate.reference,
        Target::Barrier => (gate.content + 1..gate.reference)
            .find(|&index| matches!(observed.trace.events[index], IoEvent::Synchronize { .. }))
            .unwrap(),
    };
    fs::write(&path, &observed.base).unwrap();
    let mut engine = super::open(&path, MemoryStrategy::Bounded, policy, buffering);
    disk_log::start();
    match fault {
        Fault::Write(prefix) => disk_log::fail_write_after(
            observed.trace.events[..target]
                .iter()
                .filter(|event| matches!(event, IoEvent::Write { .. }))
                .count(),
            match prefix {
                WriteFailure::Before => 0,
                WriteFailure::After => usize::MAX,
                WriteFailure::Partial => {
                    let IoEvent::Write { requested, .. } = observed.trace.events[target] else {
                        panic!("the target is a write")
                    };
                    requested / 2
                }
            },
        ),
        Fault::Sync(point) => disk_log::fail_sync_after(
            observed.trace.events[..target]
                .iter()
                .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                .count(),
            point,
        ),
    }
    let mut append = AppendBuilder::new();
    append.append_i32(&(fixture.start..fixture.start + 4).collect::<Vec<_>>());
    let error = engine
        .append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &append, 4)
        .unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected the injected storage error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        match fault {
            Fault::Write(_) => "injected disk write failure",
            Fault::Sync(_) => "injected disk synchronization failure",
        }
    );
    disk_log::clear_failures();
    let failed = disk_log::finish();
    assert_eq!(
        failed.events[target].outcome(),
        IoOutcome::Failed(ErrorKind::Other)
    );
    assert_eq!(&failed.events[..target], &observed.trace.events[..target]);
    assert_eq!(failed.events.len(), target + 1);
    if let Fault::Write(prefix) = fault {
        let IoEvent::Write {
            offset,
            requested,
            written,
            ..
        } = failed.events[target]
        else {
            panic!("the failed event is a write")
        };
        let IoEvent::Write {
            offset: expected_offset,
            requested: expected_length,
            ..
        } = observed.trace.events[target]
        else {
            panic!("the control event is a write")
        };
        assert_eq!((offset, requested), (expected_offset, expected_length));
        assert_eq!(
            written,
            match prefix {
                WriteFailure::Before => 0,
                WriteFailure::Partial => requested / 2,
                WriteFailure::After => requested,
            }
        );
    }
    let interrupted = engine.disk_bytes().unwrap();
    let mut replay = observed.base.clone();
    for operation in &failed.operations {
        operation.apply(&mut replay);
    }
    assert_eq!(replay, interrupted);
    let file = File::from_bytes(interrupted.clone()).unwrap();
    let values = file.dataset("d").unwrap().read_i32();
    if cfg!(feature = "checksum")
        && target == gate.reference
        && matches!(fault, Fault::Write(WriteFailure::Partial))
    {
        let block = source::bytes_in(&interrupted, fixture.page.parent.parse_range()).unwrap();
        let end = block.len() - test_util::checksum::CHECKSUM;
        let expected = u32::from_le_bytes(block[end..].try_into().unwrap());
        let computed = test_util::checksum::lookup3(&block[..end]);
        let error = values.unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a torn bitmap checksum, got {error:?}")
        };
        assert_eq!(error, FormatError::ChecksumMismatch { expected, computed });
        assert_ne!(expected, computed);
    } else {
        assert_eq!(values.unwrap(), (0..fixture.start).collect::<Vec<_>>());
    }
    file.persisted_free_sections().unwrap();
    disk_log::start();
    for result in [
        engine.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &append, 4),
        engine.commit(),
        engine.finalize_persist(),
        engine.force_sync(),
    ] {
        let error = result.unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    }
    drop(engine);
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(fs::read(&path).unwrap(), interrupted);
}

/// An append target with an allocated data block and an uninitialized next page.
struct Fixture {
    start: i32,
    page: Page,
}

impl Fixture {
    /// Creates the baseline and identifies the next page through a completed append.
    fn prepare(path: &Path, filtered: bool, paged: bool) -> Self {
        super::super::deferred_base(path, filtered, paged);
        compact_geometry(path);
        let initial = fs::read(path).unwrap();
        let mut engine = super::open(
            path,
            MemoryStrategy::Bounded,
            SyncPolicy::OnClose,
            Buffering::Unbuffered,
        );
        super::super::append(&mut engine, 0, 76);
        drop(engine);
        let completed = fs::read(path).unwrap();
        let pages = Page::read(&completed);
        let page = pages
            .iter()
            .find(|candidate| {
                pages.iter().any(|previous| {
                    previous.block == candidate.block && previous.slot < candidate.slot
                })
            })
            .unwrap()
            .clone();
        let start = i32::try_from(page.slot * 4).unwrap();
        fs::write(path, &initial).unwrap();
        let mut engine = super::open(
            path,
            MemoryStrategy::Bounded,
            SyncPolicy::OnClose,
            Buffering::Unbuffered,
        );
        super::super::append(&mut engine, 0, start);
        drop(engine);
        let base = fs::read(path).unwrap();
        let mut engine = super::open(
            path,
            MemoryStrategy::Bounded,
            SyncPolicy::OnClose,
            Buffering::Unbuffered,
        );
        super::super::append(&mut engine, start, 4);
        drop(engine);
        let page = Page::read(&fs::read(path).unwrap())
            .into_iter()
            .find(|candidate| candidate.slot == page.slot)
            .unwrap();
        fs::write(path, &base).unwrap();
        assert!(
            Page::read(&base)
                .iter()
                .any(|previous| previous.block == page.block)
        );
        assert!(
            !Page::in_parent(&base, page.parent)
                .iter()
                .any(|previous| previous.range == page.range)
        );
        assert!(page.block.end().get() <= u64::try_from(base.len()).unwrap());
        assert!(
            page.block.end().get()
                <= crate::superblock::read(base.as_slice())
                    .unwrap()
                    .superblock
                    .eof_address
        );
        let file = File::from_bytes(base).unwrap();
        let report = file.persisted_free_sections().unwrap().unwrap();
        assert!(report.free_sections().sections().iter().all(|section| {
            section.end_address().get() <= page.block.start().get()
                || section.address().get() >= page.block.end().get()
        }));
        Self { start, page }
    }

    fn record(&self, path: &Path, policy: SyncPolicy, buffering: Buffering) -> Observation {
        let base = fs::read(path).unwrap();
        let mut engine = super::open(path, MemoryStrategy::Bounded, policy, buffering);
        disk_log::start();
        super::super::append(&mut engine, self.start, 4);
        drop(engine);
        let trace = disk_log::finish();
        let observed = Observation::new(
            base,
            fs::read(path).unwrap(),
            trace,
            (0..self.start + 4).collect(),
        );
        observed.assert_complete(path);
        observed
    }

    fn assert_prefixes(&self, observed: &Observation) {
        let mut image = observed.base.clone();
        for (point, operation) in observed.trace.operations.iter().enumerate() {
            operation.apply(&mut image);
            let file = File::from_bytes(image.clone()).unwrap();
            let values = file.dataset("d").unwrap().read_i32().unwrap();
            assert!(
                values == (0..self.start).collect::<Vec<_>>()
                    || values == (0..self.start + 4).collect::<Vec<_>>(),
                "prefix {point}: unexpected values {values:?}"
            );
            file.persisted_free_sections().unwrap();
        }
        assert_eq!(image, observed.final_bytes);
    }
}

/// A page's stored extent and containing data and secondary blocks.
#[derive(Clone)]
struct Page {
    /// The extent of the page contents and checksum.
    range: FileRange,
    /// The full allocated data block, including uninitialized pages.
    block: FileRange,
    parent: ExtensibleArraySuperBlockRef,
    /// The first chunk-index slot in the page.
    slot: u64,
}

impl Page {
    fn read(bytes: &[u8]) -> Vec<Self> {
        let file = File::from_bytes(bytes.to_vec()).unwrap();
        let DataLayout::Chunked {
            index: ChunkIndexLayout::ExtensibleArray { address, .. },
            ..
        } = file.dataset("d").unwrap().data_layout().unwrap()
        else {
            panic!("expected an EA dataset")
        };
        let Some(address) = address else {
            return Vec::new();
        };
        let widths = crate::superblock::read(bytes).unwrap().widths;
        let header = extensible_array::read_header(
            &source::zero_base_source(&BytesSource::new(bytes)),
            widths,
            address,
        )
        .unwrap();
        let index = header.index_block().unwrap();
        let index = index.parse(region(bytes, index.parse_range())).unwrap();
        index
            .super_blocks()
            .flat_map(|parent| {
                let parent = parent.unwrap();
                Self::in_parent(bytes, parent)
            })
            .collect()
    }

    fn in_parent(bytes: &[u8], parent: ExtensibleArraySuperBlockRef) -> Vec<Self> {
        let parsed = parent.parse(region(bytes, parent.parse_range())).unwrap();
        let mut result = Vec::new();
        for block in parsed.data_blocks() {
            let block = block.unwrap();
            block.parse(region(bytes, block.parse_range())).unwrap();
            if let Some(pages) = block.pages().unwrap() {
                for page in pages.pages() {
                    let page = page.unwrap();
                    result.push(Self {
                        range: page.parse_range(),
                        block: block.range(),
                        parent,
                        slot: page.slots().unwrap().start,
                    });
                }
            }
        }
        result
    }

    /// Finds writes of the completed page contents and the page's initialization bit.
    fn discover(&self, base: &[u8], completed: &[u8], trace: &Recording) -> PageGate {
        let prepared = source::bytes_in(completed, self.range).unwrap();
        assert_ne!(prepared, source::bytes_in(base, self.range).unwrap());
        assert!(
            Self::in_parent(completed, self.parent)
                .iter()
                .any(|page| page.range == self.range)
        );
        let mut image = base.to_vec();
        let mut content = None;
        let mut reference = None;
        for (event, entry) in trace.events.iter().enumerate() {
            let Some(operation) = entry.operation() else {
                continue;
            };
            let operation = &trace.operations[operation];
            if let DiskOp::Write { offset, bytes } = operation {
                if *offset <= self.range.start().get()
                    && offset + u64::try_from(bytes.len()).unwrap() >= self.range.end().get()
                {
                    let local = usize::try_from(self.range.start().get() - offset).unwrap();
                    if bytes.get(local..local + prepared.len()) == Some(prepared) {
                        content.get_or_insert(event);
                    }
                }
                operation.apply(&mut image);
                if *offset < self.parent.parse_range().end().get()
                    && offset + u64::try_from(bytes.len()).unwrap()
                        > self.parent.parse_range().start().get()
                    && Self::in_parent(&image, self.parent)
                        .iter()
                        .any(|page| page.range == self.range)
                {
                    reference.get_or_insert(event);
                }
            } else {
                operation.apply(&mut image);
            }
        }
        PageGate {
            content: content.unwrap_or_else(|| {
                panic!(
                    "no page content write for {:?}: {:?}",
                    self.range, trace.events
                )
            }),
            reference: reference.unwrap_or_else(|| {
                panic!(
                    "no bitmap write for {:?}: {:?}",
                    self.parent.parse_range(),
                    trace.events
                )
            }),
        }
    }
}

/// Storage-event indexes for preparing a page and publishing its initialization bit.
struct PageGate {
    content: usize,
    reference: usize,
}

impl PageGate {
    fn require(&self, trace: &Recording, policy: SyncPolicy) -> Result<(), PageOrderError> {
        let Self { content, reference } = *self;
        if content >= reference {
            return Err(PageOrderError::Reference);
        }
        if trace.events[content..=reference]
            .iter()
            .any(|event| event.outcome() != IoOutcome::Succeeded)
        {
            return Err(PageOrderError::Failed);
        }
        if policy == SyncPolicy::Always
            && !trace.events[content + 1..reference].iter().any(|event| {
                matches!(
                    event,
                    IoEvent::Synchronize {
                        outcome: IoOutcome::Succeeded,
                        ..
                    }
                )
            })
        {
            return Err(PageOrderError::Barrier);
        }
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
enum PageOrderError {
    Barrier,
    Failed,
    Reference,
}

#[derive(Clone, Copy)]
enum Control {
    Barrier,
    DecoyBarrier,
    DecoyReference,
    Failed,
    Reference,
}

#[derive(Clone, Copy)]
enum Target {
    Barrier,
    Bitmap,
    Content,
}

#[derive(Clone, Copy)]
enum Fault {
    Sync(FailurePoint),
    Write(WriteFailure),
}

fn compact_geometry(path: &Path) {
    let mut bytes = fs::read(path).unwrap();
    let (header, mut layout) = test_util::bytes::signature_offsets(&bytes, v2::SIGNATURE)
        .into_iter()
        .find_map(|at| {
            ObjectHeader::parse(&bytes, AccessMode::ReadOnly, at, 8, 8)
                .unwrap()
                .messages
                .into_iter()
                .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
                .map(|message| (at, message.data))
        })
        .unwrap();
    let (_, pointer) =
        DataLayout::parse_with_ea_address(&layout, FormatWidths::from_sizes(8, 8).unwrap())
            .unwrap();
    let pointer = pointer.unwrap();
    let parameters = ExtensibleArrayCreationParameters::parse(&[32, 1, 2, 1, 1])
        .unwrap()
        .encode();
    layout[pointer.start - parameters.len()..pointer.start].copy_from_slice(&parameters);
    v2::replace_message(&mut bytes, header, FixtureMessage::DATA_LAYOUT, &layout);
    fs::write(path, bytes).unwrap();
}

fn region(bytes: &[u8], range: FileRange) -> FileRegion<'_> {
    FileRegion::new(range, source::bytes_in(bytes, range).unwrap()).unwrap()
}
