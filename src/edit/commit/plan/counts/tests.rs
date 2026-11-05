#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use core::cell::Cell;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::Arc;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::AtomicU64;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ChunkIndexInfo;
use hdf5_pure_format::__private::ChunkedLayoutFlags;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::ObjectReferenceCountIssue;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::checksum;
use test_util::data_layout;
use test_util::dataspace;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use crate::Datatype;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FileLocking;
use crate::FixedPointType;
use crate::address::StoredAddress;
use crate::dataset_request::PreparedDataset;
use crate::edit::OhRegion;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::append::AppendBuilder;
use crate::edit::attributes::EditedAttrs;
use crate::edit::dataset::ChunkPayload;
use crate::edit::dataset::MovingWrite;
use crate::edit::dataset::WritePlan;
use crate::edit::object_header;
use crate::edit::object_header::PreparedHeader;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::file_writer::LENGTH_WIDTH;
use crate::file_writer::OFFSET_WIDTH;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::image::CountingImage;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::image::HandleImage;
use crate::object_graph::ObjectId;
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::MetadataCacheConfig;
use crate::type_builders::DatasetBuilder;

use super::super::OverwriteId;
use super::super::RelocatedDataset;
use super::super::ReplacedHeader;
use super::CountProjectionError;
use super::HeaderCountError;
use super::PreparedCounts;
use super::PreparedGroup;

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
#[case::compact(Replacement::Compact)]
#[case::contiguous(Replacement::Contiguous)]
#[case::chunked(Replacement::Chunked)]
#[case::append(Replacement::Append)]
#[case::attributes(Replacement::Attributes)]
#[case::group(Replacement::Group)]
#[case::dense_group(Replacement::DenseGroup)]
fn count_projection_reaches_each_prepared_header_form(#[case] replacement: Replacement) {
    let path = test_util::temp::temp_path("prepared-count-forms.h5");
    let mut fixture = Fixture::new();
    let records = messages(&mut fixture, replacement, 2);
    let target = fixture.append_header(records);
    fixture.link("first", target);
    fixture.link("second", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, id(target)).unwrap();
    let mut group = if matches!(replacement, Replacement::Group | Replacement::DenseGroup) {
        PreparedGroup {
            region: gather(&engine, target),
            replaces: Some(if matches!(replacement, Replacement::DenseGroup) {
                ReplacedHeader::DenseAttributes(source)
            } else {
                ReplacedHeader::Header(source)
            }),
            ..PreparedGroup::default()
        }
    } else {
        let write = match replacement {
            Replacement::Append => {
                let mut append = AppendBuilder::new();
                append.append_i32(&[19]);
                WriteEngine::prepare_append(
                    &engine.opened_source(),
                    id(target),
                    &append,
                    engine.context.widths(),
                )
                .unwrap()
            }
            Replacement::Attributes => MovingWrite::AttrEdit {
                region: gather(&engine, target),
                attrs: EditedAttrs::default(),
                replaces_dense_storage: false,
            },
            _ => {
                let mut builder = DatasetBuilder::new("");
                builder.with_i32_data(&[19]);
                let value = PreparedDataset::parse(builder)
                    .unwrap()
                    .into_overwrite()
                    .unwrap();
                let WritePlan::Moving(write) = engine
                    .prepare_write(id(target), &value, OverwriteId { index: 0 })
                    .unwrap()
                else {
                    panic!("expected an unallocated or compact replacement")
                };
                write
            }
        };
        PreparedGroup {
            writes: vec![RelocatedDataset {
                link: LinkNameBuf::new("first").unwrap(),
                source,
                write,
            }],
            ..PreparedGroup::default()
        }
    };
    let original = region_mut(&mut group).to_vec();
    let capacity = region_mut(&mut group).as_ptr();
    let payload = group
        .writes
        .first_mut()
        .and_then(|write| match &mut write.write {
            MovingWrite::Chunked {
                payload: ChunkPayload::Encoded { chunks, .. },
                ..
            } => Some(chunks.as_ptr()),
            _ => None,
        });
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("first"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    let prepared = PreparedCounts::new(Some(&plan), [&mut group]).unwrap();
    assert_eq!(prepared.updates.len(), 1);
    let disposition = prepared.apply().unwrap();
    assert_eq!(disposition.rebuilt.len(), 1);
    assert!(disposition.source.is_empty() && disposition.opaque.is_empty());
    assert_eq!(disposition.rebuilt[0].bound().original(), 2);
    assert_eq!(disposition.rebuilt[0].remaining(), 1);
    assert_eq!(plan.declared_remaining(id(target)), Some(1));
    let region = region_mut(&mut group);
    assert_eq!(region.reference_count_mut().unwrap().count(), 1);
    assert_eq!(region.as_ptr(), capacity);
    assert_eq!(region.len(), original.len());
    let record = v2::message_record(&Message::reference_count(2), v2::HeaderFlags::default());
    let expected = original
        .windows(record.len())
        .position(|body| body == record)
        .unwrap();
    let mut bytes = original;
    bytes[expected + record.len() - 4] = 1;
    assert_eq!(&region[..], bytes);
    let measured = PreparedHeader::new(region).unwrap();
    let geometry = measured.geometry().unwrap();
    let encoded = object_header::build_v2_object_header(region).unwrap();
    assert_eq!(
        *geometry.root_chunk_lengths(false).next().unwrap().start(),
        encoded.len() as u64
    );
    let checksum_at = encoded.len() - checksum::CHECKSUM;
    assert_eq!(
        &encoded[checksum_at..],
        checksum::lookup3(&encoded[..checksum_at]).to_le_bytes()
    );
    if let Some(pointer) = payload {
        let MovingWrite::Chunked {
            payload: ChunkPayload::Encoded { chunks, .. },
            ..
        } = &group.writes[0].write
        else {
            panic!("expected encoded chunks")
        };
        assert_eq!(chunks.as_ptr(), pointer);
    }
}

#[rstest]
#[case::implicit(Record::Implicit, 1)]
#[case::padded(Record::Padded, 7)]
#[case::shared(Record::Shared, 7)]
#[case::ambiguous(Record::Ambiguous, 7)]
fn record_projection_preserves_local_geometry_and_opaque_disposition(
    #[case] record: Record,
    #[case] count: u32,
) {
    let path = test_util::temp::temp_path("prepared-count-records.h5");
    let mut fixture = Fixture::new();
    let records = messages(&mut fixture, Replacement::Group, count);
    let target = fixture.append_header(records);
    fixture.link("first", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, id(target)).unwrap();
    let records = match record {
        Record::Implicit => vec![Message::new(MessageType::UNKNOWN, &[0xD5; 7])],
        Record::Padded => vec![
            Message::new(
                MessageType::OBJECT_REFERENCE_COUNT,
                &[0, 7, 0, 0, 0, 0xC4, 0xD7],
            )
            .with_flags(MessageFlags::CONSTANT)
            .with_creation_order(11),
        ],
        Record::Shared => vec![Message::reference_count(7).with_flags(MessageFlags::SHARED)],
        Record::Ambiguous => vec![Message::reference_count(7), Message::reference_count(7)],
    };
    let mut bytes = v2::message_records(&records, v2::HeaderFlags::TRACKS_CREATION_ORDER);
    bytes.extend_from_slice(&[0xA5; 2]);
    let original = bytes.clone();
    let mut group = PreparedGroup {
        region: OhRegion::new(
            crate::file_writer::WIDTHS,
            bytes,
            ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(
                v2::HeaderFlags::TRACKS_CREATION_ORDER.0,
            )),
        ),
        replaces: Some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("first"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    let result = PreparedCounts::new(Some(&plan), [&mut group])
        .unwrap()
        .apply()
        .unwrap();
    let mut expected = original;
    match record {
        Record::Implicit => {
            expected.splice(
                expected.len() - 2..expected.len() - 2,
                v2::message_record(
                    &Message::reference_count(0),
                    v2::HeaderFlags::TRACKS_CREATION_ORDER,
                ),
            );
        }
        Record::Padded => expected[7] = 6,
        Record::Shared | Record::Ambiguous => {}
    }
    assert_eq!(&group.region[..], expected);
    assert_eq!(result.source.len(), 0);
    if matches!(record, Record::Shared | Record::Ambiguous) {
        assert_eq!(result.rebuilt.len(), 0);
        assert_eq!(result.opaque.len(), 1);
        assert_eq!(result.opaque[0].0.bound().original(), 7);
        assert_eq!(result.opaque[0].0.remaining(), 6);
        assert_eq!(
            result.opaque[0].1,
            if matches!(record, Record::Shared) {
                HeaderCountError::Unsupported(ObjectReferenceCountIssue::Shared)
            } else {
                HeaderCountError::Ambiguous
            }
        );
    } else {
        assert_eq!(result.rebuilt.len(), 1);
        assert!(result.opaque.is_empty());
    }
}

#[rstest]
#[case::original(Failure::Original)]
#[case::phase(Failure::Phase)]
#[case::source(Failure::Source)]
#[case::repeated(Failure::Repeated)]
#[case::framing(Failure::Framing)]
fn late_projection_failures_leave_every_local_buffer_unchanged(#[case] failure: Failure) {
    let path = test_util::temp::temp_path("prepared-count-failures.h5");
    let foreign_path = test_util::temp::temp_path("foreign-count-failures.h5");
    let mut fixture = Fixture::new();
    let first = fixture.append_header(vec![Message::reference_count(2)]);
    let second = fixture.append_header(vec![Message::reference_count(2)]);
    fixture.link("first", first);
    fixture.link("second", second);
    fixture.write(&path);
    std::fs::copy(&path, &foreign_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let foreign = edit_test_util::open_engine(&foreign_path);
    let mut phase = AnalysisPhase::new(&engine);
    let mut other = AnalysisPhase::new(if matches!(failure, Failure::Source) {
        &foreign
    } else {
        &engine
    });
    let mut groups = [
        PreparedGroup {
            region: gather(&engine, first),
            replaces: Some(ReplacedHeader::Header(
                phase.reconstruction_source(&engine, id(first)).unwrap(),
            )),
            ..PreparedGroup::default()
        },
        PreparedGroup {
            region: gather(&engine, second),
            replaces: Some(ReplacedHeader::Header(match failure {
                Failure::Phase | Failure::Source => other
                    .reconstruction_source(
                        if matches!(failure, Failure::Source) {
                            &foreign
                        } else {
                            &engine
                        },
                        id(second),
                    )
                    .unwrap(),
                Failure::Repeated => phase.reconstruction_source(&engine, id(first)).unwrap(),
                _ => phase.reconstruction_source(&engine, id(second)).unwrap(),
            })),
            ..PreparedGroup::default()
        },
    ];
    match failure {
        Failure::Original => groups[1]
            .region
            .reference_count_mut()
            .unwrap()
            .set(3)
            .unwrap(),
        Failure::Framing => {
            let mut bytes = groups[1].region.to_vec();
            bytes.extend_from_slice(&[0x30, 20, 0, 0]);
            groups[1].region = groups[1].region.with_bytes(bytes);
        }
        _ => {}
    }
    let before = groups
        .iter()
        .map(|group| group.region.to_vec())
        .collect::<Vec<_>>();
    let deletions = ["first", "second"].map(|path| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse(path))
            .unwrap()
    });
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    let Err(error) = PreparedCounts::new(Some(&plan), &mut groups) else {
        panic!("expected a count projection failure")
    };
    match failure {
        Failure::Original => assert_eq!(
            error,
            CountProjectionError::Original {
                object: id(second),
                expected: 2,
                actual: 3
            }
        ),
        Failure::Phase | Failure::Source => {
            assert_eq!(error, CountProjectionError::Phase { object: id(second) })
        }
        Failure::Repeated => {
            assert_eq!(error, CountProjectionError::Repeated { object: id(first) })
        }
        Failure::Framing => {
            let CountProjectionError::Region {
                object,
                error: HeaderCountError::Framing(error),
            } = error
            else {
                panic!("expected a framing error, got {error:?}")
            };
            assert_eq!(object, id(second));
            assert_eq!(
                error,
                FormatError::UnexpectedEof {
                    expected: before[1].len() + 20,
                    available: before[1].len()
                }
            );
        }
    }
    assert_eq!(
        groups
            .iter()
            .map(|group| group.region.to_vec())
            .collect::<Vec<_>>(),
        before
    );
    assert_eq!(
        plan.changes()
            .map(|change| change.remaining())
            .collect::<Vec<_>>(),
        [1, 1]
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(false, true)]
#[case::unknown(true, false)]
#[case::indexed(true, true)]
fn count_projection_keeps_empty_work_allocation_free(#[case] changed: bool, #[case] known: bool) {
    let path = test_util::temp::temp_path("prepared-count-work.h5");
    let mut fixture = Fixture::new();
    let first = fixture.append_header(vec![if known {
        Message::reference_count(2)
    } else {
        Message::new(MessageType::OBJECT_REFERENCE_COUNT, &[1, 2, 0, 0, 0])
    }]);
    let second = fixture.append_header(vec![Message::reference_count(3)]);
    fixture.link("first", first);
    fixture.link("second", second);
    fixture.write(&path);
    let calls = Arc::new(AtomicU64::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    MetadataCacheConfig::disabled(),
                )),
                bytes.clone(),
                Arc::default(),
            )
            .with_read_calls(calls.clone()),
        ))
    })
    .unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, id(second)).unwrap();
    let mut group = PreparedGroup {
        region: gather(&engine, second),
        replaces: Some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("first"))
        .unwrap();
    let deletions = if changed { vec![deletion] } else { vec![] };
    let plan = phase.source_count_plan(&deletions).unwrap();
    group.region = group.region.with_bytes(vec![0x30, 20, 0, 0]);
    assert!(calls.load(Ordering::Relaxed) > 0 && bytes.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let visits = Cell::new(0);
    let original = group.region.to_vec();
    let (projection, measured) = allocation::measure("prepared_count_index", || {
        PreparedCounts::new(
            plan.as_ref(),
            core::iter::once(&mut group).inspect(|_| visits.set(visits.get() + 1)),
        )
        .unwrap()
    });
    let disposition = projection.apply().unwrap();
    assert_eq!(disposition.rebuilt.len(), 0);
    assert_eq!(disposition.source.len(), usize::from(changed && known));
    assert_eq!(
        disposition
            .source
            .first()
            .map(|change| change.bound().object()),
        (changed && known).then_some(id(first))
    );
    assert_eq!(
        disposition.source.first().map(|change| change.remaining()),
        (changed && known).then_some(1)
    );
    assert!(disposition.opaque.is_empty());
    assert_eq!(&group.region[..], original);
    assert_eq!(visits.get(), usize::from(changed && known));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    if changed && known {
        assert!(measured.blocks > 0 && measured.bytes > 0, "{measured}");
    } else {
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(measured.bytes, 0, "{measured}");
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(2)]
#[case::wide(128)]
fn one_index_projects_reversed_headers_without_copying_record_buffers(#[case] count: usize) {
    let path = test_util::temp::temp_path("indexed-prepared-counts.h5");
    let mut fixture = Fixture::new();
    let objects = (0..count)
        .map(|_| fixture.append_header(vec![Message::reference_count(2)]))
        .collect::<Vec<_>>();
    let owner = fixture.image.append(
        &v2::Header::new()
            .flags(v2::HeaderFlags::CHUNK_SIZE_WIDTH)
            .messages(objects.iter().enumerate().map(|(index, &target)| {
                global_heap_fixture::hard_link(&format!("object_{index}"), target)
            }))
            .build(),
    );
    fixture.link("owner", u64::try_from(owner).unwrap());
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let suffix = v2::message_record(
        &Message::new(MessageType::UNKNOWN, &[0xD5; 4_096]),
        v2::HeaderFlags::default(),
    );
    let mut groups = objects
        .iter()
        .rev()
        .map(|&target| {
            let mut region = gather(&engine, target);
            region.push_bytes(&suffix);
            PreparedGroup {
                region,
                replaces: Some(ReplacedHeader::Header(
                    phase.reconstruction_source(&engine, id(target)).unwrap(),
                )),
                ..PreparedGroup::default()
            }
        })
        .collect::<Vec<_>>();
    let pointers = groups
        .iter()
        .map(|group| group.region.as_ptr())
        .collect::<Vec<_>>();
    let deletions = (0..count)
        .map(|index| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(&format!("owner/object_{index}")))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    let visits = Cell::new(0);
    let (prepared, measured) = allocation::measure("wide_prepared_count_index", || {
        PreparedCounts::new(
            Some(&plan),
            groups.iter_mut().inspect(|_| visits.set(visits.get() + 1)),
        )
        .unwrap()
    });
    assert_eq!(visits.get(), count);
    assert_eq!(prepared.updates.len(), count);
    assert!(measured.blocks > 0 && measured.bytes > 0, "{measured}");
    assert!(measured.bytes < (count * 1_024) as u64, "{measured}");
    let disposition = prepared.apply().unwrap();
    assert_eq!(disposition.rebuilt.len(), count);
    assert!(disposition.source.is_empty() && disposition.opaque.is_empty());
    let prefix = v2::message_record(&Message::reference_count(1), v2::HeaderFlags::default());
    for (group, pointer) in groups.iter_mut().zip(pointers) {
        assert_eq!(group.region.as_ptr(), pointer);
        assert_eq!(&group.region[..prefix.len()], prefix);
        assert_eq!(&group.region[prefix.len()..], suffix);
        assert_eq!(group.region.reference_count_mut().unwrap().count(), 1);
    }
}

fn gather(engine: &WriteEngine, target: u64) -> OhRegion {
    WriteEngine::gather_oh_messages(
        &engine.image(),
        PhysicalAddress::new(target),
        engine.context.addresses(),
        engine.widths(),
    )
    .unwrap()
}

fn region_mut(group: &mut PreparedGroup) -> &mut OhRegion {
    if let Some(write) = group.writes.first_mut() {
        write.write.region_mut()
    } else {
        &mut group.region
    }
}

fn messages(fixture: &mut Fixture, replacement: Replacement, count: u32) -> Vec<Message> {
    let mut records = if matches!(replacement, Replacement::Group | Replacement::DenseGroup) {
        vec![]
    } else {
        Form::CompactDataset.messages(
            fixture,
            &Datatype::FixedPoint(FixedPointType::i32()),
            &17_i32.to_le_bytes(),
        )
    };
    for record in &mut records {
        if record.msg_type == MessageType::DATA_LAYOUT {
            if matches!(replacement, Replacement::Contiguous) {
                record.data = data_layout::contiguous(data_layout::UNDEFINED_ADDRESS, 4);
            }
            if matches!(replacement, Replacement::Chunked) {
                record.data = data_layout::chunked(data_layout::UNDEFINED_ADDRESS, &[1], 4);
            }
            if matches!(replacement, Replacement::Append) {
                record.data = DataLayout::encode_chunked(
                    ChunkedLayoutFlags::NONE,
                    &[1],
                    4,
                    ChunkIndexInfo::ExtensibleArray,
                    StoredAddress::undefined(OFFSET_WIDTH),
                    OFFSET_WIDTH,
                    LENGTH_WIDTH,
                )
                .unwrap();
            }
        }
        if record.msg_type == MessageType::DATASPACE && matches!(replacement, Replacement::Append) {
            record.data = dataspace::v2(
                1,
                dataspace::Flags::MAX_DIMS,
                dataspace::Kind::SIMPLE,
                &[0],
                Some(&[u64::MAX]),
            );
        }
    }
    if count != 1 {
        records.push(Message::reference_count(count));
    }
    records
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

#[derive(Clone, Copy)]
enum Replacement {
    Append,
    Attributes,
    Chunked,
    Compact,
    Contiguous,
    DenseGroup,
    Group,
}
#[derive(Clone, Copy)]
enum Record {
    Ambiguous,
    Implicit,
    Padded,
    Shared,
}
#[derive(Clone, Copy)]
enum Failure {
    Framing,
    Original,
    Phase,
    Repeated,
    Source,
}
