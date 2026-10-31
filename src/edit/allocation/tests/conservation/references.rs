//! Checks allocation and reference publication through recorded I/O attempts.
//!
//! Tests locate stored pointers and complete preparations in reconstructed file prefixes. They
//! inject write and synchronization errors at those operations and check retry or session rejection.
//! For altered traces, tests reconstruct the completed file and assert ordering or I/O outcome errors.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use rstest::rstest;
use test_util::allocation_oracle;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::Context;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::bytes as byte_values;
use test_util::checksum;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Recording;
use test_util::publication_order::EofChange;
use test_util::publication_order::OrderError as PublicationError;
use test_util::publication_order::Policy;
use test_util::publication_order::Publication;

use crate::Error;
use crate::File;
use crate::FormatError;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::edit::object_header;
use crate::extensible_array;
use crate::file_lock::FileLocking;
use crate::image::WriteBuffering;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::source::BytesSource;
use crate::source::MetadataCacheConfig;
use crate::source::RangeReader;

use super::CHUNK;
use super::Observation;
use super::Workload;
use super::extents;

#[rstest]
#[case::flat_plain(false, false, SyncPolicy::Always)]
#[case::flat_filtered(false, true, SyncPolicy::Always)]
#[case::page_plain(true, false, SyncPolicy::Always)]
#[case::page_filtered(true, true, SyncPolicy::Always)]
#[case::flat_ordered(false, false, SyncPolicy::OnClose)]
#[case::page_ordered(true, true, SyncPolicy::OnClose)]
fn stored_references_follow_complete_ownership_and_metadata(
    #[case] paged: bool,
    #[case] filtered: bool,
    #[case] policy: SyncPolicy,
    #[values(false, true)] buffered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("references.h5");
    let (workload, hole) = record(&path, paged, filtered, policy, buffered);
    let roles = Roles::read(&workload);
    roles.require(&workload, policy).unwrap();
    workload.replay(&directory.path().join("prefix.h5"), &hole, |_, _, _| {});
    inspect(&path, &workload.final_bytes, &values());
}

#[rstest]
#[case::payload(WriteTarget::Payload)]
#[case::later_payload(WriteTarget::LaterPayload)]
#[case::manager_header(WriteTarget::ManagerHeader)]
#[case::section_list(WriteTarget::SectionList)]
#[case::extension(WriteTarget::Extension)]
#[case::superblock(WriteTarget::Superblock)]
#[case::first_layout(WriteTarget::Layout)]
#[case::direct_child(WriteTarget::Direct)]
#[case::secondary_child(WriteTarget::Secondary)]
#[case::indirect_child(WriteTarget::Indirect)]
#[case::element_count(WriteTarget::Count)]
#[case::dataspace(WriteTarget::Dataspace)]
fn reported_write_errors_stop_dependent_references_and_reuse(
    #[case] target: WriteTarget,
    #[values(WriteFailure::Before, WriteFailure::Prefix, WriteFailure::After)] fault: WriteFailure,
    #[values(false, true)] buffered: bool,
) {
    run_failure(
        FaultSpec::Write(target, fault),
        SyncPolicy::Always,
        buffered,
        true,
    );
}

#[rstest]
#[case::payload(WriteTarget::Payload)]
#[case::later_payload(WriteTarget::LaterPayload)]
#[case::first_layout(WriteTarget::Layout)]
#[case::direct_child(WriteTarget::Direct)]
#[case::secondary_child(WriteTarget::Secondary)]
#[case::indirect_child(WriteTarget::Indirect)]
fn plain_reference_errors_stop_dependent_references_and_reuse(
    #[case] target: WriteTarget,
    #[values(WriteFailure::Before, WriteFailure::Prefix, WriteFailure::After)] fault: WriteFailure,
    #[values(false, true)] buffered: bool,
) {
    run_failure(
        FaultSpec::Write(target, fault),
        SyncPolicy::Always,
        buffered,
        false,
    );
}

#[rstest]
#[case::payload(WriteTarget::Payload)]
#[case::later_payload(WriteTarget::LaterPayload)]
#[case::first_layout(WriteTarget::Layout)]
#[case::secondary_child(WriteTarget::Secondary)]
#[case::dataspace(WriteTarget::Dataspace)]
fn ordered_write_errors_stop_dependent_references_and_reuse(
    #[case] target: WriteTarget,
    #[values(false, true)] buffered: bool,
) {
    run_failure(
        FaultSpec::Write(target, WriteFailure::After),
        SyncPolicy::OnClose,
        buffered,
        true,
    );
}

#[rstest]
#[case::prepared(SyncTarget::Content)]
#[case::owned(SyncTarget::Ownership)]
#[case::payload(SyncTarget::Payload)]
#[case::later_payload(SyncTarget::LaterPayload)]
#[case::direct_ownership(SyncTarget::DirectOwnership)]
#[case::secondary_ownership(SyncTarget::SecondaryOwnership)]
#[case::indirect_ownership(SyncTarget::IndirectOwnership)]
#[case::first_layout(SyncTarget::Layout)]
#[case::children(SyncTarget::Children)]
#[case::count(SyncTarget::Count)]
#[case::dataspace(SyncTarget::Dataspace)]
fn reported_sync_errors_stop_dependent_references_and_reuse(
    #[case] target: SyncTarget,
    #[values(FailurePoint::Before, FailurePoint::After)] fault: FailurePoint,
    #[values(false, true)] buffered: bool,
) {
    run_failure(
        FaultSpec::Synchronize(target, fault),
        SyncPolicy::Always,
        buffered,
        true,
    );
}

#[rstest]
#[case::layout_barrier(Control::LayoutBarrier, SyncPolicy::Always)]
#[case::children_barrier(Control::ChildrenBarrier, SyncPolicy::Always)]
#[case::count_barrier(Control::CountBarrier, SyncPolicy::Always)]
#[case::early_secondary(Control::Secondary, SyncPolicy::Always)]
#[case::early_indirect(Control::Indirect, SyncPolicy::Always)]
#[case::early_count(Control::Count, SyncPolicy::Always)]
#[case::early_dataspace(Control::Dataspace, SyncPolicy::Always)]
#[case::ordered_count(Control::Count, SyncPolicy::OnClose)]
#[case::ordered_dataspace(Control::Dataspace, SyncPolicy::OnClose)]
#[case::reported_write_failure(Control::WriteFailure, SyncPolicy::Always)]
#[case::reported_sync_failure(Control::SyncFailure, SyncPolicy::Always)]
#[case::incomplete_write(Control::IncompleteWrite, SyncPolicy::Always)]
#[case::ordered_write_failure(Control::WriteFailure, SyncPolicy::OnClose)]
#[case::late_payload(Control::LatePayload, SyncPolicy::Always)]
#[case::late_index(Control::LateIndex, SyncPolicy::Always)]
#[case::ordered_late_payload(Control::LatePayload, SyncPolicy::OnClose)]
#[case::ordered_late_index(Control::LateIndex, SyncPolicy::OnClose)]
fn reference_controls_reject_readable_completed_files(
    #[case] control: Control,
    #[case] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("control.h5");
    let (mut workload, _) = record(&path, true, true, policy, true);
    let roles = Roles::read(&workload);
    roles.require(&workload, policy).unwrap();
    let dependency = match control {
        Control::LatePayload | Control::LateIndex => {
            let (range, content, reference) = roles
                .pointers
                .iter()
                .find(|(_, content, reference)| {
                    if matches!(control, Control::LatePayload) {
                        content.contains(&roles.payload)
                    } else {
                        *reference == roles.layout
                    }
                })
                .unwrap();
            let preparation = if matches!(control, Control::LatePayload) {
                roles.payload
            } else {
                content[0]
            };
            Some((
                preparation,
                roles.publication(&workload, *range, content),
                *reference,
            ))
        }
        Control::Indirect | Control::Secondary => {
            let reference = if matches!(control, Control::Indirect) {
                roles.indirect
            } else {
                roles.secondary
            };
            Some((
                reference,
                roles.reference_publication(&workload, reference),
                reference,
            ))
        }
        _ => None,
    };
    let failed_event = match control {
        Control::WriteFailure | Control::IncompleteWrite => Some(roles.positions[roles.superblock]),
        Control::SyncFailure => Some(roles.sync_target(&workload, SyncTarget::Ownership)),
        _ => None,
    };
    match control {
        Control::LatePayload | Control::LateIndex => {
            let (preparation, publication, _) = dependency.unwrap();
            assert!(roles.positions[preparation] < roles.positions[publication]);
            let event = workload.events.remove(roles.positions[preparation]);
            workload.events.insert(roles.positions[publication], event);
        }
        Control::WriteFailure | Control::SyncFailure => {
            match &mut workload.events[failed_event.unwrap()] {
                IoEvent::Write { outcome, .. } | IoEvent::Synchronize { outcome, .. } => {
                    *outcome = IoOutcome::Failed(ErrorKind::Other)
                }
                IoEvent::Resize { .. } => panic!("the control selects a write or synchronization"),
            }
        }
        Control::IncompleteWrite => {
            let IoEvent::Write { written, .. } = &mut workload.events[failed_event.unwrap()] else {
                panic!("the control selects a write")
            };
            *written -= 1;
        }
        Control::LayoutBarrier | Control::ChildrenBarrier | Control::CountBarrier => {
            let (start, end) = match control {
                Control::LayoutBarrier => (roles.layout, roles.payload),
                Control::ChildrenBarrier => (roles.last_child, roles.count),
                Control::CountBarrier => (roles.count, roles.dataspace),
                Control::Count
                | Control::Dataspace
                | Control::Indirect
                | Control::Secondary
                | Control::WriteFailure
                | Control::SyncFailure
                | Control::IncompleteWrite
                | Control::LateIndex
                | Control::LatePayload => {
                    panic!("the case selects a different operation")
                }
            };
            let mut affected = 0;
            let mut position = 0;
            workload.events.retain(|event| {
                let remove = matches!(event, IoEvent::Synchronize { .. })
                    && roles.positions[start] < position
                    && position < roles.positions[end];
                affected += usize::from(remove);
                position += 1;
                !remove
            });
            assert!(affected > 0);
        }
        Control::Count => {
            let mut repeated = workload.events[roles.positions[roles.count]].clone();
            let operation = repeated.operation().unwrap();
            let replacement = workload.operations.len();
            workload
                .operations
                .push(workload.operations[operation].clone());
            let IoEvent::Write { operation, .. } = &mut repeated else {
                panic!("the count is a write")
            };
            *operation = Some(replacement);
            workload
                .events
                .insert(roles.positions[roles.last_child], repeated);
        }
        Control::Indirect | Control::Secondary => {
            let (_, publication, reference) = dependency.unwrap();
            workload
                .events
                .swap(roles.positions[publication], roles.positions[reference]);
        }
        Control::Dataspace => workload.events.swap(
            roles.positions[roles.dataspace],
            roles.positions[roles.count],
        ),
    }
    let altered = Roles::read(&workload);
    workload.assert_reconstruction();
    assert_eq!(
        altered.require(&workload, policy),
        Err(match control {
            Control::WriteFailure | Control::SyncFailure =>
                OrderError::Io(PublicationError::FailedIo {
                    event: failed_event.unwrap(),
                    outcome: IoOutcome::Failed(ErrorKind::Other)
                }),
            Control::IncompleteWrite => OrderError::Io(PublicationError::InvalidWrite {
                event: failed_event.unwrap()
            }),
            Control::LayoutBarrier | Control::ChildrenBarrier | Control::CountBarrier => {
                OrderError::Barrier
            }
            Control::Count | Control::Dataspace => OrderError::Reference,
            Control::Indirect | Control::Secondary | Control::LatePayload | Control::LateIndex => {
                let (content, publication, reference) = dependency.unwrap();
                OrderError::Io(PublicationError::ReferenceOrder {
                    content: altered.positions[content],
                    publication: altered.positions[publication],
                    reference: altered.positions[reference],
                })
            }
        })
    );
    inspect(&path, &workload.final_bytes, &values());
}

/// Injects an error at a discovered I/O event and checks the interrupted file and session.
fn run_failure(fault: FaultSpec, policy: SyncPolicy, buffered: bool, filtered: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed.h5");
    let (workload, hole) = record(&path, true, filtered, policy, buffered);
    let roles = Roles::read(&workload);
    roles.require(&workload, policy).unwrap();
    let (event, frame, prefix, partial) = match fault {
        FaultSpec::Synchronize(target, _) => {
            (roles.sync_target(&workload, target), None, None, None)
        }
        FaultSpec::Write(target, failure) => {
            let event = roles.write_target(target);
            let frame = roles
                .frames
                .iter()
                .find_map(|(role, range)| (*role == target).then_some(*range));
            let prefix = match failure {
                WriteFailure::Before => 0,
                WriteFailure::After => usize::MAX,
                WriteFailure::Prefix => workload.write_prefix(event, frame),
            };
            (
                event,
                frame,
                Some(prefix),
                matches!(failure, WriteFailure::Prefix).then_some(target),
            )
        }
    };
    fs::write(&path, &workload.base).unwrap();
    let mut engine = open(&path, policy, buffered);
    disk_log::start();
    match fault {
        FaultSpec::Synchronize(_, point) => disk_log::fail_sync_after(
            workload.events[..event]
                .iter()
                .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                .count(),
            point,
        ),
        FaultSpec::Write(_, _) => disk_log::fail_write_after(
            workload.events[..event]
                .iter()
                .filter(|event| matches!(event, IoEvent::Write { .. }))
                .count(),
            prefix.unwrap(),
        ),
    }
    let error = append(&mut engine).unwrap_err();
    let Error::Io(error) = error else {
        panic!("{fault:?}/{buffered}: expected injected I/O, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if matches!(fault, FaultSpec::Synchronize(_, _)) {
            "injected disk synchronization failure"
        } else {
            "injected disk write failure"
        }
    );
    disk_log::clear_failures();
    let failed = disk_log::finish();
    assert_eq!(&failed.events[..event], &workload.events[..event]);
    assert_eq!(
        failed.events[event].outcome(),
        IoOutcome::Failed(ErrorKind::Other)
    );
    assert!(
        failed.events[event + 1..]
            .iter()
            .all(|event| matches!(event, IoEvent::Resize { .. }))
    );
    match fault {
        FaultSpec::Write(_, _) => {
            let IoEvent::Write {
                offset,
                requested,
                written,
                ..
            } = failed.events[event]
            else {
                panic!("the failed event is a write")
            };
            let IoEvent::Write {
                offset: expected_offset,
                requested: expected_length,
                ..
            } = workload.events[event]
            else {
                panic!("the control event is a write")
            };
            assert_eq!((offset, requested), (expected_offset, expected_length));
            assert_eq!(written, prefix.unwrap().min(requested));
        }
        FaultSpec::Synchronize(_, _) => {
            let IoEvent::Synchronize { kind, .. } = failed.events[event] else {
                panic!("the failed event is a synchronization")
            };
            let IoEvent::Synchronize {
                kind: expected_kind,
                ..
            } = workload.events[event]
            else {
                panic!("the control event is a synchronization")
            };
            assert_eq!(kind, expected_kind);
        }
    }
    let interrupted = engine.disk_bytes().unwrap();
    let mut replay = workload.base.clone();
    for operation in &failed.operations {
        operation.apply(&mut replay);
    }
    assert_eq!(replay, interrupted, "{fault:?}: recorded bytes differ");
    if let Some(target) = partial.filter(|target| target.metadata_reference()) {
        if cfg!(feature = "checksum") {
            assert_torn(&interrupted, &workload, target, frame.unwrap());
        } else if matches!(
            target,
            WriteTarget::Superblock
                | WriteTarget::Layout
                | WriteTarget::Direct
                | WriteTarget::Secondary
                | WriteTarget::Indirect
        ) {
            let pointer = roles
                .reads
                .iter()
                .find(|pointer| pointer.role == target)
                .unwrap();
            assert_malformed_pointer(&interrupted, &workload, pointer, frame.unwrap());
        } else {
            workload.assert_readable_failure(&interrupted, &hole, event, fault);
        }
    } else {
        workload.assert_readable_failure(&interrupted, &hole, event, fault);
    }
    disk_log::start();
    if fault.known_unpublished() {
        let start = Observation::read(&interrupted).values.len();
        append_from(&mut engine, start).unwrap();
        drop(engine);
        let retry = disk_log::finish();
        assert!(!retry.events.is_empty());
        let completed = fs::read(&path).unwrap();
        let mut replay = interrupted;
        for operation in retry.operations {
            operation.apply(&mut replay);
        }
        assert_eq!(replay, completed);
        inspect(&path, &completed, &values());
    } else {
        let error = append(&mut engine).unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{fault:?}: {error:?}"
        );
        drop(engine);
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(fs::read(&path).unwrap(), interrupted);
    }
}

/// Byte-operation indices discovered from stored metadata and reconstructed prefixes.
///
/// The event positions are separate from the byte-operation indices: synchronization attempts
/// occupy event positions without changing the file bytes.
struct Roles {
    /// The event position of each byte operation.
    positions: Vec<usize>,
    payload: usize,
    later_payload: usize,
    manager_header: usize,
    section_list: usize,
    extension: usize,
    superblock: usize,
    layout: usize,
    direct: usize,
    secondary: usize,
    indirect: usize,
    last_child: usize,
    count: usize,
    dataspace: usize,
    /// Allocation ranges, preparation operations, and first reference operations.
    pointers: Vec<(Interval, Vec<usize>, usize)>,
    /// Ownership publication operations and their manager preparation operations.
    managers: Vec<(usize, Vec<usize>)>,
    /// Metadata blocks containing each selected reference or visibility field and its checksum.
    frames: Vec<(WriteTarget, Interval)>,
    reads: Vec<PointerRead>,
}

/// A pointer field and the target address and length discovered in the completed fixture.
struct PointerRead {
    role: WriteTarget,
    field: Interval,
    address: u64,
    length: u64,
}

impl Roles {
    /// Locates preparations, ownership publications, and references by parsing fixture bytes.
    fn read(workload: &Workload) -> Self {
        let mut positions = vec![usize::MAX; workload.operations.len()];
        for (position, event) in workload.events.iter().enumerate() {
            if let Some(operation) = event.operation() {
                assert_eq!(positions[operation], usize::MAX);
                positions[operation] = position;
            }
        }
        assert!(positions.iter().all(|position| *position != usize::MAX));
        let final_observation = workload.snapshots.last().unwrap();
        let index = final_observation
            .live
            .iter()
            .filter(|fact| fact.kind == AllocationKind::Index)
            .collect::<Vec<_>>();
        let header = index
            .iter()
            .find(|fact| {
                bytes(&workload.final_bytes, fact.parsed)
                    .starts_with(test_util::extensible_array::SIGNATURE)
            })
            .unwrap();
        let located = crate::superblock::read(workload.final_bytes.as_slice()).unwrap();
        let ea = extensible_array::read_header(
            &BytesSource::new(&workload.final_bytes),
            located.widths,
            StoredAddress::new(header.range.start()),
        )
        .unwrap();
        let ea_index = ea.index_block().unwrap();
        let source = BytesSource::new(&workload.final_bytes);
        let addresses = AddressContext::from_effective_base(located.superblock.base_address);
        let dataset = File::from_bytes(workload.final_bytes.clone())
            .unwrap()
            .dataset("d")
            .unwrap();
        let mut layout_field = None;
        let mut object_frame = None;
        let mut dataspace_body = None;
        for chunk in
            object_header::read_oh_chunks(&source, dataset.header_address().unwrap(), addresses)
                .unwrap()
        {
            let (region, mut position) = chunk.message_region();
            while let Some((kind, body, end)) =
                object_header::next_record(chunk.layout(), region, position).unwrap()
            {
                if kind == MessageType::DATASPACE {
                    dataspace_body = Some(
                        Interval::new(
                            chunk.span.start().get() + u64::try_from(body).unwrap(),
                            u64::try_from(end - body).unwrap(),
                        )
                        .unwrap(),
                    );
                }
                if kind == MessageType::DATA_LAYOUT {
                    let (layout, pointer) =
                        DataLayout::parse_with_ea_address(&region[body..end], located.widths)
                            .unwrap();
                    assert!(matches!(
                        layout,
                        DataLayout::Chunked {
                            index: ChunkIndexLayout::ExtensibleArray {
                                address: Some(_),
                                ..
                            },
                            ..
                        }
                    ));
                    object_frame =
                        Some(Interval::new(chunk.span.start().get(), chunk.span.length()).unwrap());
                    let pointer = pointer.unwrap();
                    layout_field = Some(
                        Interval::new(
                            chunk.span.start().get() + u64::try_from(body + pointer.start).unwrap(),
                            u64::try_from(pointer.len()).unwrap(),
                        )
                        .unwrap(),
                    );
                }
                position = end;
            }
        }
        let layout_field = layout_field.unwrap();
        assert!(
            bytes(&workload.base, layout_field)
                .iter()
                .all(|byte| *byte == u8::MAX)
        );
        let layout = workload.reference(layout_field, header.range.start());
        let mut frames = vec![
            (WriteTarget::Layout, object_frame.unwrap()),
            (WriteTarget::Dataspace, object_frame.unwrap()),
            (WriteTarget::Count, header.parsed),
        ];
        let mut reads = vec![PointerRead {
            role: WriteTarget::Layout,
            field: layout_field,
            address: header.range.start(),
            length: ea.range().length(),
        }];
        let mut direct = None;
        let mut secondary = None;
        let mut indirect = None;
        assert_eq!(header.range.end(), ea_index.range().start().get());
        let initial = Interval::new(
            header.range.start(),
            ea_index.range().end().get() - header.range.start(),
        )
        .unwrap();
        let mut pointers = vec![(
            initial,
            vec![
                workload.prepared_index(ea.range(), |region| {
                    ExtensibleArrayHeader::parse_with_widths(region.bytes(), 0, located.widths)
                        .map(|_| ())
                }),
                workload.prepared_index(ea_index.parse_range(), |region| {
                    ea_index.parse(region).map(|_| ())
                }),
            ],
            layout,
        )];
        let parsed_index = ea_index
            .parse(region(&workload.final_bytes, ea_index.parse_range()))
            .unwrap();
        let mut children = Vec::new();
        for child in parsed_index.data_blocks() {
            let child = child.unwrap();
            children.push((
                child.range(),
                child.parse_range().length(),
                workload.prepared_index(child.parse_range(), |region| {
                    child.parse(region).map(|_| ())
                }),
                ea_index.parse_range(),
                WriteTarget::Direct,
            ));
        }
        for parent in parsed_index.super_blocks() {
            let parent = parent.unwrap();
            children.push((
                parent.parse_range(),
                parent.parse_range().length(),
                workload.prepared_index(parent.parse_range(), |region| {
                    parent.parse(region).map(|_| ())
                }),
                ea_index.parse_range(),
                WriteTarget::Secondary,
            ));
            for child in parent
                .parse(region(&workload.final_bytes, parent.parse_range()))
                .unwrap()
                .data_blocks()
            {
                let child = child.unwrap();
                children.push((
                    child.range(),
                    child.parse_range().length(),
                    workload.prepared_index(child.parse_range(), |region| {
                        child.parse(region).map(|_| ())
                    }),
                    parent.parse_range(),
                    WriteTarget::Indirect,
                ));
            }
        }
        for (range, length, preparation, parent, kind) in children {
            let field = pointer_field(&workload.final_bytes, interval(parent), range.start().get());
            let operation = workload.reference(field, range.start().get());
            match kind {
                WriteTarget::Direct => {
                    if direct.is_none() {
                        reads.push(PointerRead {
                            role: kind,
                            field,
                            address: range.start().get(),
                            length,
                        });
                        frames.push((kind, interval(parent)));
                    }
                    direct.get_or_insert(operation);
                }
                WriteTarget::Secondary => {
                    if secondary.is_none() {
                        reads.push(PointerRead {
                            role: kind,
                            field,
                            address: range.start().get(),
                            length,
                        });
                        frames.push((kind, interval(parent)));
                    }
                    secondary.get_or_insert(operation);
                }
                WriteTarget::Indirect => {
                    if indirect.is_none() {
                        reads.push(PointerRead {
                            role: kind,
                            field,
                            address: range.start().get(),
                            length,
                        });
                        frames.push((kind, interval(parent)));
                    }
                    indirect.get_or_insert(operation);
                }
                _ => panic!("the case selects a different operation"),
            }
            pointers.push((interval(range), vec![preparation], operation));
        }
        for chunk in &final_observation.chunks {
            let fields = final_observation
                .element_blocks
                .iter()
                .filter_map(|parent| {
                    let offsets = test_util::bytes::signature_offsets(
                        bytes(&workload.final_bytes, *parent),
                        &chunk.1.to_le_bytes(),
                    );
                    if offsets.is_empty() {
                        None
                    } else {
                        Some(pointer_field(&workload.final_bytes, *parent, chunk.1))
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(fields.len(), 1);
            pointers.push((
                Interval::new(chunk.1, chunk.2).unwrap(),
                vec![workload.prepared(
                    Interval::new(chunk.1, chunk.2).unwrap(),
                    &workload.final_bytes,
                    0,
                )],
                workload.reference(fields[0], chunk.1),
            ));
        }
        let payload = final_observation
            .live
            .iter()
            .filter(|fact| {
                fact.kind == AllocationKind::Payload && !workload.snapshots[0].live.contains(fact)
            })
            .map(|fact| workload.prepared(fact.parsed, &workload.final_bytes, 0))
            .min()
            .unwrap();
        let later_payload = final_observation
            .chunks
            .iter()
            .filter(|chunk| chunk.0[0] >= u64::try_from(CHUNK).unwrap())
            .map(|chunk| {
                workload.prepared(
                    Interval::new(chunk.1, chunk.2).unwrap(),
                    &workload.final_bytes,
                    0,
                )
            })
            .min()
            .unwrap();
        assert_ne!(payload, later_payload);
        let mut image = workload.base.clone();
        let mut count = None;
        let mut dataspace = None;
        let dataspace_body = dataspace_body.unwrap();
        for event in &workload.events {
            let Some(operation) = event.operation() else {
                continue;
            };
            workload.operations[operation].apply(&mut image);
            if let Some(stored) = image.get(
                usize::try_from(header.parsed.start()).unwrap()
                    ..usize::try_from(header.parsed.end()).unwrap(),
            ) {
                if ExtensibleArrayHeader::parse_with_widths(stored, 0, located.widths).is_ok_and(
                    |parsed| parsed.max_idx_set == u64::try_from(extents::ROUNDS).unwrap(),
                ) {
                    count.get_or_insert(operation);
                }
            }
            if Dataspace::parse_with_length_width(
                bytes(&image, dataspace_body),
                located.widths.lengths,
            )
            .unwrap()
            .dimensions
                == [u64::try_from(extents::ROUNDS * CHUNK).unwrap()]
            {
                dataspace.get_or_insert(operation);
            }
        }
        let count = count.unwrap();
        let dataspace = dataspace.unwrap();
        let mut managers = Vec::new();
        let mut manager_header = None;
        let mut section_list = None;
        let mut extension = None;
        let mut superblock = None;
        let mut start = 0;
        for (operation, entry) in workload.operations.iter().enumerate() {
            if !matches!(entry, DiskOp::Write { offset: 0, .. }) {
                continue;
            }
            let image = workload.prefix(operation + 1);
            let observation = &workload.snapshots[operation + 1];
            let mut prepared_managers = Vec::new();
            for fact in observation.live.iter().filter(|fact| {
                fact.kind == AllocationKind::Manager && !workload.snapshots[0].live.contains(fact)
            }) {
                let preparation = workload.prepared(fact.parsed, &image, start);
                let stored = bytes(&image, fact.parsed);
                if stored.starts_with(test_util::free_space::SIGNATURE) {
                    manager_header.get_or_insert(preparation);
                }
                if stored.starts_with(test_util::free_space::SECTIONS_SIGNATURE) {
                    section_list.get_or_insert(preparation);
                }
                if stored.starts_with(test_util::object_header::v2::SIGNATURE) {
                    extension.get_or_insert(preparation);
                }
                prepared_managers.push(preparation);
            }
            if !prepared_managers.is_empty() {
                superblock.get_or_insert(operation);
                managers.push((operation, prepared_managers));
            }
            start = operation + 1;
        }
        let superblock_frame = final_observation
            .live
            .iter()
            .find(|fact| fact.kind == AllocationKind::Index && fact.range.start() == 0)
            .unwrap()
            .parsed;
        frames.push((WriteTarget::Superblock, superblock_frame));
        let first_publication = superblock.unwrap();
        let published = workload.prefix(first_publication + 1);
        let extension_address = crate::superblock::read(published.as_slice())
            .unwrap()
            .superblock
            .superblock_extension_address
            .unwrap()
            .get();
        let extension_length = workload.snapshots[first_publication + 1]
            .live
            .iter()
            .find(|fact| {
                fact.kind == AllocationKind::Manager && fact.parsed.start() == extension_address
            })
            .unwrap()
            .parsed
            .length();
        reads.push(PointerRead {
            role: WriteTarget::Superblock,
            field: pointer_field(&published, superblock_frame, extension_address),
            address: extension_address,
            length: extension_length,
        });
        let last_child = pointers
            .iter()
            .map(|(_, _, operation)| *operation)
            .max_by_key(|operation| positions[*operation])
            .unwrap();
        Self {
            positions,
            payload,
            later_payload,
            manager_header: manager_header.unwrap(),
            section_list: section_list.unwrap(),
            extension: extension.unwrap(),
            superblock: superblock.unwrap(),
            layout,
            direct: direct.unwrap(),
            secondary: secondary.unwrap(),
            indirect: indirect.unwrap(),
            last_child,
            count,
            dataspace,
            pointers,
            managers,
            frames,
            reads,
        }
    }

    /// Checks successful I/O, allocation dependencies, and final visibility ordering.
    ///
    /// Under [`SyncPolicy::Always`], also checks the intervening synchronization attempts.
    /// Under [`SyncPolicy::OnClose`], checks that the append completes without synchronization.
    ///
    /// # Errors
    ///
    /// Returns an [`OrderError`] if an I/O outcome, write length, dependency, or synchronization
    /// attempt violates the selected policy.
    fn require(&self, workload: &Workload, policy: SyncPolicy) -> Result<(), OrderError> {
        let recording = workload.successful_recording().map_err(OrderError::Io)?;
        let before = |first, second| {
            if self.positions[first] >= self.positions[second] {
                Err(OrderError::Reference)
            } else {
                Ok(())
            }
        };
        let barrier = |first, second| {
            before(first, second)?;
            if policy == SyncPolicy::Always
                && !workload.events[self.positions[first] + 1..self.positions[second]]
                    .iter()
                    .any(|event| {
                        matches!(
                            event,
                            IoEvent::Synchronize {
                                outcome: IoOutcome::Succeeded,
                                ..
                            }
                        )
                    })
            {
                return Err(OrderError::Barrier);
            }
            Ok(())
        };
        for (publication, managers) in &self.managers {
            for prepared in managers {
                barrier(*prepared, *publication)?;
            }
        }
        for (range, content, reference) in &self.pointers {
            let publication = self.publication(workload, *range, content);
            let managers = &self
                .managers
                .iter()
                .find(|(operation, _)| *operation == publication)
                .unwrap()
                .1;
            let mut prepared = content
                .iter()
                .chain(managers)
                .map(|operation| self.positions[*operation])
                .collect::<Vec<_>>();
            let first = *prepared.iter().min().unwrap();
            prepared.extend(
                (first..self.positions[publication])
                    .filter(|event| matches!(recording.events[*event], IoEvent::Write { .. })),
            );
            prepared.sort_unstable();
            prepared.dedup();
            let eof =
                if workload.snapshots[publication].eof == workload.snapshots[publication + 1].eof {
                    EofChange::Unchanged
                } else {
                    EofChange::Extend
                };
            Publication::parse(
                &recording,
                &prepared,
                self.positions[publication],
                self.positions[*reference],
                eof,
            )
            .and_then(|publication| {
                publication.require(match policy {
                    SyncPolicy::Always => Policy::Always,
                    SyncPolicy::OnClose => Policy::OnClose,
                })
            })
            .map_err(|error| match error {
                PublicationError::ContentBarrier { .. }
                | PublicationError::PublicationBarrier { .. } => OrderError::Barrier,
                error => OrderError::Io(error),
            })?;
        }
        barrier(self.layout, self.payload)?;
        barrier(self.last_child, self.count)?;
        barrier(self.count, self.dataspace)?;
        if policy == SyncPolicy::OnClose
            && workload
                .events
                .iter()
                .any(|event| matches!(event, IoEvent::Synchronize { .. }))
        {
            return Err(OrderError::Synchronization);
        }
        Ok(())
    }

    fn write_target(&self, target: WriteTarget) -> usize {
        let Self {
            positions,
            payload,
            later_payload,
            manager_header,
            section_list,
            extension,
            superblock,
            layout,
            direct,
            secondary,
            indirect,
            last_child: _,
            count,
            dataspace,
            pointers: _,
            managers: _,
            frames: _,
            reads: _,
        } = self;
        match target {
            WriteTarget::Payload => positions[*payload],
            WriteTarget::LaterPayload => positions[*later_payload],
            WriteTarget::ManagerHeader => positions[*manager_header],
            WriteTarget::SectionList => positions[*section_list],
            WriteTarget::Extension => positions[*extension],
            WriteTarget::Superblock => positions[*superblock],
            WriteTarget::Layout => positions[*layout],
            WriteTarget::Direct => positions[*direct],
            WriteTarget::Secondary => positions[*secondary],
            WriteTarget::Indirect => positions[*indirect],
            WriteTarget::Count => positions[*count],
            WriteTarget::Dataspace => positions[*dataspace],
        }
    }

    fn sync_target(&self, workload: &Workload, target: SyncTarget) -> usize {
        let Self {
            positions,
            payload,
            later_payload,
            extension,
            superblock,
            layout,
            direct,
            secondary,
            indirect,
            last_child,
            count,
            dataspace,
            manager_header: _,
            section_list: _,
            pointers: _,
            managers: _,
            frames: _,
            reads: _,
        } = self;
        match target {
            SyncTarget::Payload => self.content_barrier(workload, *payload),
            SyncTarget::LaterPayload => self.content_barrier(workload, *later_payload),
            SyncTarget::DirectOwnership => self.reference_barrier(workload, *direct),
            SyncTarget::SecondaryOwnership => self.reference_barrier(workload, *secondary),
            SyncTarget::IndirectOwnership => self.reference_barrier(workload, *indirect),
            SyncTarget::Content => self.sync_between(workload, *extension, *superblock),
            SyncTarget::Ownership => self.sync_between(workload, *superblock, *layout),
            SyncTarget::Layout => self.sync_between(workload, *layout, *payload),
            SyncTarget::Children => self.sync_between(workload, *last_child, *count),
            SyncTarget::Count => self.sync_between(workload, *count, *dataspace),
            SyncTarget::Dataspace => (positions[*dataspace] + 1..workload.events.len())
                .find(|position| matches!(workload.events[*position], IoEvent::Synchronize { .. }))
                .unwrap(),
        }
    }

    /// Finds the first manager publication after preparation with the range outside free space.
    ///
    /// Also requires the complete range to be within the published EOF.
    fn publication(&self, workload: &Workload, range: Interval, content: &[usize]) -> usize {
        self.managers
            .iter()
            .find_map(|(operation, _)| {
                let snapshot = &workload.snapshots[operation + 1];
                (*operation >= *content.iter().max().unwrap()
                    && range.end() <= snapshot.eof
                    && Coverage::new([range])
                        .intersection(&snapshot.free)
                        .is_empty())
                .then_some(*operation)
            })
            .unwrap()
    }

    fn content_barrier(&self, workload: &Workload, preparation: usize) -> usize {
        let (range, content, _) = self
            .pointers
            .iter()
            .find(|(_, content, _)| content.contains(&preparation))
            .unwrap();
        self.sync_between(
            workload,
            preparation,
            self.publication(workload, *range, content),
        )
    }

    fn reference_barrier(&self, workload: &Workload, reference: usize) -> usize {
        self.sync_between(
            workload,
            self.reference_publication(workload, reference),
            reference,
        )
    }

    fn reference_publication(&self, workload: &Workload, reference: usize) -> usize {
        self.pointers
            .iter()
            .filter(|(_, _, operation)| *operation == reference)
            .map(|(range, content, _)| self.publication(workload, *range, content))
            .max_by_key(|operation| self.positions[*operation])
            .unwrap()
    }

    fn sync_between(&self, workload: &Workload, first: usize, second: usize) -> usize {
        (self.positions[first] + 1..self.positions[second])
            .find(|position| matches!(workload.events[*position], IoEvent::Synchronize { .. }))
            .unwrap()
    }
}

impl Workload {
    /// Checks visible values and allocation conservation after a readable interrupted append.
    fn assert_readable_failure(
        &self,
        interrupted: &[u8],
        hole: &Coverage,
        event: usize,
        fault: FaultSpec,
    ) {
        let observation = Observation::read(interrupted);
        assert_eq!(
            observation.values,
            (0..i32::try_from(observation.values.len()).unwrap()).collect::<Vec<_>>()
        );
        assert!(
            self.snapshots
                .iter()
                .any(|snapshot| snapshot.values == observation.values)
        );
        let complete = self.events[..event]
            .iter()
            .filter_map(IoEvent::operation)
            .max()
            .map_or(0, |operation| operation + 1)
            + usize::from(matches!(fault, FaultSpec::Write(_, WriteFailure::After)));
        let facts = self.facts(complete, &observation);
        allocation_oracle::inspect(
            Context {
                case: "reported reference error",
                interruption: event,
                publication: observation.state(),
            },
            hole,
            &observation.free,
            &observation.reachable(),
            &facts,
        )
        .unwrap();
    }

    /// Copies a trace after checking successful outcomes and complete matching writes.
    ///
    /// # Errors
    ///
    /// Returns [`PublicationError::FailedIo`] if any attempt failed, or
    /// [`PublicationError::InvalidWrite`] if a write is empty, incomplete, or differs from its
    /// recorded byte operation.
    fn successful_recording(&self) -> Result<Recording, PublicationError> {
        for (event, entry) in self.events.iter().enumerate() {
            if entry.outcome() != IoOutcome::Succeeded {
                return Err(PublicationError::FailedIo {
                    event,
                    outcome: entry.outcome(),
                });
            }
            if let IoEvent::Write {
                offset,
                requested,
                written,
                operation,
                ..
            } = entry
            {
                if requested != written || *written == 0 || !operation.is_some_and(|operation| matches!(self.operations.get(operation), Some(DiskOp::Write { offset: at, bytes }) if at == offset && bytes.len() == *written)) {
                    return Err(PublicationError::InvalidWrite { event });
                }
            }
        }
        Ok(Recording {
            operations: self.operations.clone(),
            events: self.events.clone(),
        })
    }

    /// Finds the first write of the complete encoded address at `field` in event order.
    fn reference(&self, field: Interval, address: u64) -> usize {
        self.events
            .iter()
            .find_map(|event| {
                let operation = event.operation()?;
                let DiskOp::Write { offset, bytes } = &self.operations[operation] else {
                    return None;
                };
                let local = usize::try_from(field.start().checked_sub(*offset)?).ok()?;
                (bytes.get(local..local + usize::try_from(field.length()).unwrap())
                    == Some(address.to_le_bytes().as_slice()))
                .then_some(operation)
            })
            .unwrap()
    }

    /// Finds the first complete write matching `image` within `range`, starting at `start`.
    fn prepared(&self, range: Interval, image: &[u8], start: usize) -> usize {
        self.operations
            .iter()
            .enumerate()
            .skip(start)
            .find_map(|(operation, entry)| {
                let DiskOp::Write {
                    offset,
                    bytes: stored,
                } = entry
                else {
                    return None;
                };
                let local = usize::try_from(range.start().checked_sub(*offset)?).ok()?;
                (stored.get(local..local + usize::try_from(range.length()).unwrap())
                    == Some(bytes(image, range)))
                .then_some(operation)
            })
            .unwrap_or_else(|| panic!("no complete preparation for {range:?} after {start}"))
    }

    /// Finds the first complete block write with a valid checksum and a successful `parse` result.
    fn prepared_index(
        &self,
        range: FileRange,
        parse: impl Fn(FileRegion<'_>) -> Result<(), FormatError>,
    ) -> usize {
        self.operations
            .iter()
            .enumerate()
            .find_map(|(operation, entry)| {
                let DiskOp::Write {
                    offset,
                    bytes: written,
                } = entry
                else {
                    return None;
                };
                if *offset > range.start().get()
                    || offset + u64::try_from(written.len()).unwrap() < range.end().get()
                {
                    return None;
                }
                let image = self.prefix(operation + 1);
                let block = bytes(&image, interval(range));
                let end = block.len() - checksum::CHECKSUM;
                if block[end..] != checksum::lookup3(&block[..end]).to_le_bytes() {
                    return None;
                }
                parse(region(&image, range)).ok().map(|()| operation)
            })
            .unwrap_or_else(|| panic!("no independently parsed preparation for {range:?}"))
    }

    fn assert_reconstruction(&self) {
        let mut image = self.base.clone();
        for event in &self.events {
            if let Some(operation) = event.operation() {
                self.operations[operation].apply(&mut image);
            }
        }
        assert!(
            image == self.final_bytes,
            "altered replay changes the completed bytes"
        );
    }

    /// Returns a partial write length through the first changed byte before the block checksum.
    ///
    /// With `frame` set to [`None`], selects the first changed byte anywhere in the write.
    /// Returns zero for an event without a byte write.
    fn write_prefix(&self, event: usize, frame: Option<Interval>) -> usize {
        let Some(operation) = self.events[event].operation() else {
            return 0;
        };
        let DiskOp::Write {
            offset,
            bytes: written,
        } = &self.operations[operation]
        else {
            return 0;
        };
        let before = self.prefix(operation);
        let at = usize::try_from(*offset).unwrap();
        let first = written
            .iter()
            .enumerate()
            .position(|(local, byte)| {
                let address = u64::try_from(at + local).unwrap();
                frame.is_none_or(|range| {
                    range.start() <= address
                        && address < range.end() - u64::try_from(checksum::CHECKSUM).unwrap()
                }) && before.get(at + local) != Some(byte)
            })
            .unwrap();
        assert!(first + 1 < written.len());
        first + 1
    }
}

/// Checks the parser's checksum error against the interrupted block's stored and computed checksums.
fn assert_torn(image: &[u8], workload: &Workload, target: WriteTarget, range: Interval) {
    let block = bytes(image, range);
    let end = block.len() - checksum::CHECKSUM;
    let expected = u32::from_le_bytes(block[end..].try_into().unwrap());
    let computed = checksum::lookup3(&block[..end]);
    let located = crate::superblock::read(workload.final_bytes.as_slice()).unwrap();
    let file = File::from_bytes(workload.final_bytes.clone()).unwrap();
    let dataset = file.dataset("d").unwrap();
    let result = match target {
        WriteTarget::Superblock => crate::superblock::read(image).map(|_| ()),
        WriteTarget::Layout | WriteTarget::Dataspace => ObjectHeader::parse(
            image,
            AccessMode::ReadOnly,
            usize::try_from(dataset.header_address().unwrap()).unwrap(),
            located.widths.offsets.get(),
            located.widths.lengths.get(),
        )
        .map(|_| ()),
        WriteTarget::Count => ExtensibleArrayHeader::parse_with_widths(
            image,
            usize::try_from(range.start()).unwrap(),
            located.widths,
        )
        .map(|_| ()),
        WriteTarget::Direct | WriteTarget::Secondary | WriteTarget::Indirect => {
            let DataLayout::Chunked {
                index:
                    ChunkIndexLayout::ExtensibleArray {
                        address: Some(address),
                        ..
                    },
                ..
            } = dataset.data_layout().unwrap()
            else {
                panic!("the dataset has an EA index")
            };
            let ea = extensible_array::read_header(
                &BytesSource::new(&workload.final_bytes),
                located.widths,
                address,
            )
            .unwrap();
            let index = ea.index_block().unwrap();
            if matches!(target, WriteTarget::Indirect) {
                let parsed = index
                    .parse(region(&workload.final_bytes, index.parse_range()))
                    .unwrap();
                let parent = parsed
                    .super_blocks()
                    .map(Result::unwrap)
                    .find(|parent| interval(parent.parse_range()) == range)
                    .unwrap();
                parent
                    .parse(region(image, parent.parse_range()))
                    .map(|_| ())
            } else {
                index.parse(region(image, index.parse_range())).map(|_| ())
            }
        }
        _ => panic!("the case selects a different operation"),
    };
    assert_eq!(
        result.unwrap_err(),
        FormatError::ChecksumMismatch { expected, computed }
    );
    assert_ne!(expected, computed);
}

/// Checks parser addresses and errors against an independently decoded interrupted pointer field.
fn assert_malformed_pointer(
    image: &[u8],
    workload: &Workload,
    pointer: &PointerRead,
    frame: Interval,
) {
    let target = pointer.role;
    let stored = byte_values::u64_at(image, usize::try_from(pointer.field.start()).unwrap());
    assert_ne!(stored, pointer.address);
    let file = File::from_bytes(image.to_vec()).unwrap();
    if target == WriteTarget::Superblock {
        let address = crate::superblock::read(image)
            .unwrap()
            .superblock
            .superblock_extension_address
            .unwrap()
            .get();
        assert_eq!(address, stored);
        let version = byte_values::slice_at(image, usize::try_from(stored).unwrap(), 1)[0];
        let error = file.persisted_free_sections().unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a malformed extension, got {error:?}")
        };
        assert_eq!(error, FormatError::InvalidObjectHeaderVersion(version));
        return;
    }
    file.persisted_free_sections().unwrap();
    let located = crate::superblock::read(workload.final_bytes.as_slice()).unwrap();
    let complete = File::from_bytes(workload.final_bytes.clone()).unwrap();
    let DataLayout::Chunked {
        index:
            ChunkIndexLayout::ExtensibleArray {
                address: Some(address),
                ..
            },
        ..
    } = complete.dataset("d").unwrap().data_layout().unwrap()
    else {
        panic!("the dataset has an EA index")
    };
    let source = BytesSource::new(image);
    if target == WriteTarget::Layout {
        let DataLayout::Chunked {
            index:
                ChunkIndexLayout::ExtensibleArray {
                    address: Some(torn),
                    ..
                },
            ..
        } = file.dataset("d").unwrap().data_layout().unwrap()
        else {
            panic!("the layout stores a torn EA address")
        };
        assert_eq!(torn.get(), stored);
        assert_ne!(torn, address);
        let expected = stored.checked_add(pointer.length).unwrap();
        assert_eq!(
            extensible_array::read_header(&source, located.widths, torn).unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected,
                available: u64::try_from(image.len()).unwrap()
            }
        );
        return;
    }
    let header = extensible_array::read_header(
        &BytesSource::new(&workload.final_bytes),
        located.widths,
        address,
    )
    .unwrap();
    let index = header.index_block().unwrap();
    let parsed = index.parse(region(image, index.parse_range())).unwrap();
    let range = match target {
        WriteTarget::Direct => parsed
            .data_blocks()
            .map(Result::unwrap)
            .map(|block| block.parse_range())
            .find(|range| range.end().get() > u64::try_from(image.len()).unwrap())
            .unwrap(),
        WriteTarget::Secondary => parsed
            .super_blocks()
            .map(Result::unwrap)
            .map(|block| block.parse_range())
            .find(|range| range.end().get() > u64::try_from(image.len()).unwrap())
            .unwrap(),
        WriteTarget::Indirect => {
            let parent = parsed
                .super_blocks()
                .map(Result::unwrap)
                .find(|parent| interval(parent.parse_range()) == frame)
                .unwrap();
            parent
                .parse(region(image, parent.parse_range()))
                .unwrap()
                .data_blocks()
                .map(Result::unwrap)
                .map(|block| block.parse_range())
                .find(|range| range.end().get() > u64::try_from(image.len()).unwrap())
                .unwrap()
        }
        _ => panic!("the target stores a child pointer"),
    };
    assert_eq!(
        (range.start().get(), range.length()),
        (stored, pointer.length)
    );
    assert_eq!(
        source.read_range(range).unwrap_err(),
        FormatError::UnexpectedSourceEof {
            expected: stored.checked_add(pointer.length).unwrap(),
            available: u64::try_from(image.len()).unwrap()
        }
    );
}

/// Records first and subsequent appends from an undefined index with compact fixture geometry.
fn record(
    path: &Path,
    paged: bool,
    filtered: bool,
    policy: SyncPolicy,
    buffered: bool,
) -> (Workload, Coverage) {
    let original = super::pure_fixture(path, paged, true, filtered);
    extents::compact_geometry(path);
    let hole = original.intersection(&Observation::read(&fs::read(path).unwrap()).free);
    assert!(!hole.is_empty());
    let workload = Workload::record_with(path, extents::ROUNDS, |_| {
        let mut engine = open(path, policy, buffered);
        append(&mut engine).unwrap();
        let before = engine.disk_bytes().unwrap();
        drop(engine);
        assert_eq!(fs::read(path).unwrap(), before);
    });
    inspect(path, &workload.final_bytes, &values());
    (workload, hole)
}

fn open(path: &Path, policy: SyncPolicy, buffered: bool) -> WriteEngine {
    let mut engine = WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine.set_sync_policy(policy);
    engine
        .set_write_buffering(if buffered {
            WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 65_536,
            }
        } else {
            WriteBuffering::Unbuffered
        })
        .unwrap();
    engine
}

fn append(engine: &mut WriteEngine) -> Result<(), Error> {
    append_from(engine, 0)
}

fn append_from(engine: &mut WriteEngine, start: usize) -> Result<(), Error> {
    if start == 0 {
        let mut append = AppendBuilder::new();
        append.append_i32(&values()[..CHUNK]);
        engine.append_inplace_gathered(
            AppendTarget::Path(&ObjectPathBuf::parse("d")),
            &append,
            4,
        )?;
    }
    let mut append = AppendBuilder::new();
    append.append_i32(&values()[start.max(CHUNK)..]);
    engine.append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &append, 4)
}

fn values() -> Vec<i32> {
    (0..i32::try_from(extents::ROUNDS * CHUNK).unwrap()).collect()
}

fn inspect(path: &Path, image: &[u8], expected: &[i32]) {
    assert_eq!(fs::read(path).unwrap(), image);
    let file = File::open(path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
    file.persisted_free_sections().unwrap();
    drop(file);
    assert_eq!(fs::read(path).unwrap(), image);
}

fn bytes(image: &[u8], range: Interval) -> &[u8] {
    &image[usize::try_from(range.start()).unwrap()..usize::try_from(range.end()).unwrap()]
}

fn interval(range: FileRange) -> Interval {
    Interval::new(range.start().get(), range.length()).unwrap()
}

fn region(image: &[u8], range: FileRange) -> FileRegion<'_> {
    FileRegion::new(range, bytes(image, interval(range))).unwrap()
}

fn pointer_field(image: &[u8], parent: Interval, address: u64) -> Interval {
    let fields = test_util::bytes::signature_offsets(bytes(image, parent), &address.to_le_bytes());
    assert_eq!(fields.len(), 1);
    Interval::new(
        parent.start() + u64::try_from(fields[0]).unwrap(),
        u64::try_from(address.to_le_bytes().len()).unwrap(),
    )
    .unwrap()
}

/// A write selected by its role in allocation or reference publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WriteTarget {
    /// The final element count in the Extensible Array header.
    Count,
    /// The final dataset dimensions in the object header.
    Dataspace,
    /// A data-block address stored in the Extensible Array index block.
    Direct,
    Extension,
    /// A data-block address stored in a secondary block.
    Indirect,
    /// The first payload preparation after the initial chunk.
    LaterPayload,
    /// The first Extensible Array header address stored in the data layout message.
    Layout,
    ManagerHeader,
    Payload,
    /// A secondary-block address stored in the Extensible Array index block.
    Secondary,
    SectionList,
    Superblock,
}

impl WriteTarget {
    /// Returns whether the caller can retry after a failure before ownership publication.
    fn known_unpublished(self) -> bool {
        matches!(
            self,
            Self::Payload
                | Self::LaterPayload
                | Self::ManagerHeader
                | Self::SectionList
                | Self::Extension
        )
    }

    fn metadata_reference(self) -> bool {
        matches!(
            self,
            Self::Count
                | Self::Dataspace
                | Self::Direct
                | Self::Indirect
                | Self::Layout
                | Self::Secondary
                | Self::Superblock
        )
    }
}

/// A synchronization attempt selected at a discovered append boundary.
#[derive(Clone, Copy, Debug)]
enum SyncTarget {
    /// After the last child reference and before the final element count.
    Children,
    /// After manager preparation and before its superblock publication.
    Content,
    /// After the final element count and before the dataset dimensions.
    Count,
    /// After the final dataset dimensions.
    Dataspace,
    DirectOwnership,
    IndirectOwnership,
    LaterPayload,
    Layout,
    Ownership,
    Payload,
    SecondaryOwnership,
}

impl SyncTarget {
    /// Returns whether the caller can retry after a failure before ownership publication.
    fn known_unpublished(self) -> bool {
        matches!(self, Self::Content | Self::Payload | Self::LaterPayload)
    }
}

/// The number of bytes to write before reporting an injected error.
#[derive(Clone, Copy, Debug)]
enum WriteFailure {
    /// The complete requested write.
    After,
    /// Zero bytes.
    Before,
    /// Through the first changed byte, with the rest of the write omitted.
    Prefix,
}

/// An injected failure paired with the matching kind of I/O target.
#[derive(Clone, Copy, Debug)]
enum FaultSpec {
    Synchronize(SyncTarget, FailurePoint),
    Write(WriteTarget, WriteFailure),
}

impl FaultSpec {
    fn known_unpublished(self) -> bool {
        match self {
            Self::Synchronize(target, _) => target.known_unpublished(),
            Self::Write(target, _) => target.known_unpublished(),
        }
    }
}

/// An invalid trace alteration with identical completed file bytes.
#[derive(Clone, Copy)]
enum Control {
    ChildrenBarrier,
    Count,
    CountBarrier,
    Dataspace,
    IncompleteWrite,
    Indirect,
    LateIndex,
    LatePayload,
    LayoutBarrier,
    Secondary,
    SyncFailure,
    WriteFailure,
}

/// A failed I/O, dependency, or synchronization check on the recorded append.
#[derive(Debug, Eq, PartialEq)]
enum OrderError {
    Barrier,
    Io(PublicationError),
    Reference,
    Synchronization,
}
