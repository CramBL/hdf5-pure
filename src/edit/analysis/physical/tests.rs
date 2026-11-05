use core::cell::Cell;
use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::SuperblockVersion;
use rstest::rstest;
use test_util::bytes;
use test_util::checksum;
use test_util::file_space_info;
use test_util::free_space;
use test_util::image::Image;
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util::superblock::v0;
use test_util::superblock::v2;
use test_util::widths::Widths;

use super::{FileStorageKind, FileStorageRange, StorageProblem};
use crate::address::{StoredAddress, StoredAddressExt};
use crate::edit::analysis::{AnalysisPhase, DISCOVERY_WORK};
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

#[rstest]
#[case::zero(SuperblockVersion::Zero)]
#[case::one(SuperblockVersion::One)]
#[case::two(SuperblockVersion::Two)]
#[case::three(SuperblockVersion::Three)]
fn file_framing_preserves_versions_widths_and_source_bounds(
    #[case] version: SuperblockVersion,
    #[values(Widths::new(2, 2), Widths::FOUR, Widths::EIGHT)] widths: Widths,
    #[values(0, 512)] base: u64,
) {
    let path = test_util::temp::temp_path("physical-file-framing.h5");
    let fixture = FileFixture::new(version, widths, base);
    let encoded = fixture.write(&path, None);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    calls.store(0, Ordering::Relaxed);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(base + ROOT)))
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let facts = phase.physical_storage(&source).unwrap();
    assert_eq!(
        (facts.physical_size(), facts.logical_eoa()),
        (fixture.physical, fixture.logical)
    );
    assert!(facts.problems().is_empty(), "{:?}", facts.problems());
    let ranges = facts
        .file()
        .iter()
        .map(|entry| (entry.kind, entry.range.start().get(), entry.range.length()))
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    if base > 0 {
        expected.push((FileStorageKind::Userblock, 0, base));
    }
    expected.push((
        FileStorageKind::Superblock,
        base,
        u64::try_from(encoded.len()).unwrap(),
    ));
    assert_eq!(ranges, expected);
    assert!(facts.metadata_capacity().is_empty());
    let before = calls.load(Ordering::Relaxed);
    assert!(before > 0);
    assert_eq!(
        phase.physical_storage(&source).unwrap().file(),
        facts_copy(&expected).as_slice()
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    let mut other = AnalysisPhase::new(&engine);
    assert_eq!(
        other.physical_storage(&source).err().unwrap(),
        FormatError::Internal("physical storage query belongs to another source phase".into())
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ExtensionSupport {
    Known,
    Opaque,
    RequiredUnknown,
}

#[rstest]
#[case::supported(ExtensionSupport::Known)]
#[case::opaque(ExtensionSupport::Opaque)]
#[case::incompatible(ExtensionSupport::RequiredUnknown)]
fn extension_framing_keeps_message_support_independent(#[case] support: ExtensionSupport) {
    let path = test_util::temp::temp_path("physical-extension.h5");
    let extension = Header::new()
        .message(Message::new(
            MessageType::FILE_SPACE_INFO,
            &file_space_info::non_persistent_version_1(
                Widths::EIGHT,
                file_space_info::PAGE,
                1,
                512,
            ),
        ))
        .build();
    FileFixture::new(SuperblockVersion::Three, Widths::EIGHT, 512).write(&path, Some(&extension));
    let (mut engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut header = Header::new().message(Message::new(
        MessageType::FILE_SPACE_INFO,
        &file_space_info::non_persistent_version_1(Widths::EIGHT, file_space_info::PAGE, 1, 512),
    ));
    if support != ExtensionSupport::Known {
        let message = Message::new(MessageType(0x7f), &[1, 2, 3]);
        header = header.message(if support == ExtensionSupport::RequiredUnknown {
            message.with_flags(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)
        } else {
            message
        });
    }
    let extension = header.build();
    engine.write_at(512 + EXTENSION, &extension).unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(512 + ROOT)))
        .unwrap();
    let facts = phase.physical_storage(&source).unwrap();
    assert_eq!(
        facts
            .file()
            .iter()
            .filter(|entry| entry.kind == FileStorageKind::Extension)
            .map(|entry| (entry.range.start().get(), entry.range.length()))
            .collect::<Vec<_>>(),
        [(512 + EXTENSION, u64::try_from(extension.len()).unwrap())]
    );
    match support {
        ExtensionSupport::Known => assert!(facts.problems().is_empty(), "{:?}", facts.problems()),
        ExtensionSupport::Opaque => assert!(
            matches!(facts.problems(), [StorageProblem::ExtensionMessage(kind)] if *kind == crate::MessageType::from_u16(0x7f)),
            "{:?}",
            facts.problems()
        ),
        ExtensionSupport::RequiredUnknown => assert!(
            matches!(
                facts.problems(),
                [StorageProblem::Framing {
                    kind: FileStorageKind::Extension,
                    error: FormatError::UnsupportedMessage(0x7f)
                }]
            ),
            "{:?}",
            facts.problems()
        ),
    }
    let before = calls.load(Ordering::Relaxed);
    let expected = usize::from(support != ExtensionSupport::Known);
    assert_eq!(
        phase.physical_storage(&source).unwrap().problems().len(),
        expected
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[rstest]
#[case::padded(false)]
#[case::outside_source(true)]
fn manager_framing_retains_allocated_capacity_and_failed_outcomes(#[case] outside: bool) {
    let path = test_util::temp::temp_path("physical-manager.h5");
    let (mut header, list) = free_space::raw_manager_eight(
        MANAGER,
        MANAGER + u64::try_from(free_space::single_section_header().len()).unwrap(),
        &[(6_144, 256)],
    );
    let list_at = MANAGER + u64::try_from(header.len()).unwrap();
    let allocated = if outside {
        8_193 - list_at
    } else {
        u64::try_from(list.len()).unwrap() + 128
    };
    let allocated_at = header.len() - checksum::CHECKSUM - Widths::EIGHT.length;
    bytes::set_uint_at(&mut header, allocated_at, Widths::EIGHT.length, allocated);
    let length = header.len();
    checksum::restamp(&mut header, 0, length);
    let extension = Header::new()
        .message(Message::new(
            MessageType::FILE_SPACE_INFO,
            &file_space_info::version_0(
                Widths::EIGHT,
                file_space_info::ALL_PERSIST,
                1,
                &[MANAGER, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX],
            ),
        ))
        .build();
    let mut fixture = FileFixture::new(SuperblockVersion::Three, Widths::EIGHT, 0);
    fixture.logical = fixture.physical;
    fixture.write(&path, Some(&extension));
    let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    use std::io::{Seek, SeekFrom, Write};
    file.seek(SeekFrom::Start(MANAGER)).unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&list).unwrap();
    drop(file);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(ROOT)))
        .unwrap();
    let facts = phase.physical_storage(&source).unwrap();
    if outside {
        assert!(
            matches!(facts.problems(), [StorageProblem::Framing { kind: FileStorageKind::Manager, error: FormatError::InvalidFreeSpaceManager(reason) }] if reason == &format!("the section list allocation at {list_at}, {allocated} bytes long, runs past the end of the file, 8192 bytes long")),
            "{:?}",
            facts.problems()
        );
    } else {
        assert!(facts.problems().is_empty(), "{:?}", facts.problems());
        assert_eq!(
            facts
                .file()
                .iter()
                .filter(|entry| entry.kind == FileStorageKind::Manager)
                .map(|entry| (entry.range.start().get(), entry.range.length()))
                .collect::<Vec<_>>(),
            [
                (MANAGER, u64::try_from(header.len()).unwrap()),
                (list_at, allocated)
            ]
        );
        assert!(
            facts
                .metadata_capacity()
                .iter()
                .any(|range| range.start().get() == list_at && range.length() == allocated)
        );
    }
    let before = calls.load(Ordering::Relaxed);
    assert_eq!(
        phase.physical_storage(&source).unwrap().problems().len(),
        usize::from(outside)
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[rstest]
#[case::address(LocalProblem::Address)]
#[case::context(LocalProblem::Context)]
#[case::legacy(LocalProblem::Legacy)]
#[case::publication(LocalProblem::Publication)]
#[case::superblock(LocalProblem::Superblock)]
fn local_storage_keeps_failed_and_inapplicable_facts_explicit(#[case] problem: LocalProblem) {
    let path = test_util::temp::temp_path("physical-source-problem.h5");
    let version = if matches!(problem, LocalProblem::Legacy) {
        SuperblockVersion::Zero
    } else {
        SuperblockVersion::Three
    };
    let fixture = FileFixture::new(version, Widths::EIGHT, 512);
    fixture.write(&path, None);
    if matches!(problem, LocalProblem::Legacy) {
        let mut bytes = fs::read(&path).unwrap();
        let fields = v0::Fields::read(&bytes, 512);
        bytes::set_uint_at(
            &mut bytes,
            fields.eof_address_at + Widths::EIGHT.offset,
            Widths::EIGHT.offset,
            2_048,
        );
        fs::write(&path, bytes).unwrap();
    }
    let (mut engine, calls) = edit_test_util::open_counted_engine(&path);
    match problem {
        LocalProblem::Address => {
            let mut superblock = engine.superblock().clone();
            superblock.superblock_extension_address = Some(StoredAddress::new(u64::MAX - 1));
            let encoded = hdf5_pure_format::__private::serialize_superblock(&superblock).unwrap();
            engine.write_at(512, &encoded).unwrap();
            let proposed = engine.context.propose(superblock).unwrap();
            engine.context.adopt(proposed).unwrap();
        }
        LocalProblem::Context => {
            let replacement = v2::Superblock::new(Widths::EIGHT)
                .version(SuperblockVersion::Three.get())
                .base_address(512)
                .eof_address(fixture.logical)
                .root_header_address(ROOT + 16)
                .build();
            engine.write_at(512, &replacement).unwrap();
        }
        LocalProblem::Legacy => {}
        LocalProblem::Publication => engine.context.allocations.make_uncertain(),
        LocalProblem::Superblock => engine.write_at(512, &[0]).unwrap(),
    }
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(512 + ROOT)))
        .unwrap();
    let facts = phase.physical_storage(&source).unwrap();
    match problem {
        LocalProblem::Address => assert!(
            matches!(facts.problems(), [StorageProblem::Address(FormatError::OffsetOverflow { offset, length: 512 })] if *offset == u64::MAX - 1),
            "{:?}",
            facts.problems()
        ),
        LocalProblem::Context => assert!(
            matches!(facts.problems(), [StorageProblem::Context]),
            "{:?}",
            facts.problems()
        ),
        LocalProblem::Legacy => assert!(
            matches!(facts.problems(), [StorageProblem::LegacyAllocation(address)] if address.get() == 2_048),
            "{:?}",
            facts.problems()
        ),
        LocalProblem::Publication => assert!(
            matches!(
                facts.problems(),
                [StorageProblem::Publication(
                    crate::Error::AllocationPublicationUncertain
                )]
            ),
            "{:?}",
            facts.problems()
        ),
        LocalProblem::Superblock => assert!(
            matches!(
                facts.problems(),
                [StorageProblem::Framing {
                    kind: FileStorageKind::Superblock,
                    error: FormatError::SignatureNotFound
                }]
            ),
            "{:?}",
            facts.problems()
        ),
    }
    let before = calls.load(Ordering::Relaxed);
    assert_eq!(phase.physical_storage(&source).unwrap().problems().len(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[derive(Clone, Copy)]
enum LocalProblem {
    Address,
    Context,
    Legacy,
    Publication,
    Superblock,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn file_storage_queries_read_no_payload_or_residual_bytes() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use test_util::allocation;

    let path = test_util::temp::temp_path("physical-file-resources.h5");
    let mut fixture = FileFixture::new(SuperblockVersion::Three, Widths::EIGHT, 0);
    fixture.physical = 1_048_576;
    let encoded = fixture.write(&path, None);
    let read_bytes = Arc::new(AtomicU64::new(0));
    let (engine, calls) =
        edit_test_util::open_counted_engine_with_bytes(&path, Arc::clone(&read_bytes));
    calls.store(0, Ordering::Relaxed);
    read_bytes.store(0, Ordering::Relaxed);
    let (mut phase, empty) =
        allocation::measure("physical_file_empty_phase", || AnalysisPhase::new(&engine));
    assert_eq!((empty.blocks, empty.bytes), (0, 0));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(ROOT)))
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let ((), measured) = allocation::measure("physical_file_query", || {
        let facts = phase.physical_storage(&source).unwrap();
        assert!(facts.problems().is_empty(), "{:?}", facts.problems());
        assert_eq!(facts.physical_size(), fixture.physical);
        assert_eq!(facts.logical_eoa(), fixture.logical);
        assert_eq!(facts.file().len(), 1);
        assert_eq!(facts.file()[0].kind, FileStorageKind::Superblock);
        assert_eq!(facts.file()[0].range.start(), PhysicalAddress::new(0));
        assert_eq!(
            facts.file()[0].range.length(),
            u64::try_from(encoded.len()).unwrap()
        );
    });
    assert!(
        measured.blocks > 0
            && measured.blocks <= 16
            && measured.bytes > 0
            && measured.bytes <= 4_096
            && measured.peak_bytes <= 4_096
            && measured.live_bytes <= 4_096,
        "{measured}"
    );
    let before = (
        calls.load(Ordering::Relaxed),
        read_bytes.load(Ordering::Relaxed),
    );
    assert!(
        before.0 > 0 && before.0 < 8 && before.1 > 0 && before.1 < 256,
        "{before:?}"
    );
    let ((), cached) = allocation::measure("physical_file_cached", || {
        assert_eq!(phase.physical_storage(&source).unwrap().file().len(), 1);
    });
    assert_eq!((cached.blocks, cached.bytes), (0, 0));
    assert_eq!(
        (
            calls.load(Ordering::Relaxed),
            read_bytes.load(Ordering::Relaxed)
        ),
        before
    );
    let (payload, positive) =
        allocation::measure("physical_file_payload_control", || vec![0_u8; 1_048_576]);
    assert_eq!(payload.len(), 1_048_576);
    assert!(positive.bytes >= 1_048_576);
}

fn facts_copy(ranges: &[(FileStorageKind, u64, u64)]) -> Vec<FileStorageRange> {
    ranges
        .iter()
        .map(|&(kind, address, length)| FileStorageRange {
            range: FileRange::new(PhysicalAddress::new(address), length).unwrap(),
            kind,
        })
        .collect()
}

struct FileFixture {
    version: SuperblockVersion,
    widths: Widths,
    base: u64,
    physical: u64,
    logical: u64,
}

impl FileFixture {
    fn new(version: SuperblockVersion, widths: Widths, base: u64) -> Self {
        Self {
            version,
            widths,
            base,
            physical: base + 8_192,
            logical: base + 8_064,
        }
    }
    fn write(&self, path: &Path, extension: Option<&[u8]>) -> Vec<u8> {
        let superblock = match self.version {
            SuperblockVersion::Zero | SuperblockVersion::One => {
                let builder = v0::Superblock::new(self.widths)
                    .base_address(self.base)
                    .eof_address(self.logical)
                    .root_group(0, ROOT);
                if self.version == SuperblockVersion::One {
                    builder.version_1().build()
                } else {
                    builder.build()
                }
            }
            SuperblockVersion::Two | SuperblockVersion::Three => {
                let builder = v2::Superblock::new(self.widths)
                    .version(self.version.get())
                    .base_address(self.base)
                    .eof_address(self.logical)
                    .root_header_address(ROOT);
                if extension.is_some() {
                    builder.extension_address(EXTENSION).build()
                } else {
                    builder.build()
                }
            }
        };
        let mut image = Image::new();
        image.place(usize::try_from(self.base).unwrap(), &superblock);
        image.place(
            usize::try_from(self.base + ROOT).unwrap(),
            &Header::new().build(),
        );
        if let Some(extension) = extension {
            image.place(usize::try_from(self.base + EXTENSION).unwrap(), extension);
        }
        fs::write(path, image.build()).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(self.physical)
            .unwrap();
        superblock
    }
}

const ROOT: u64 = 512;
const EXTENSION: u64 = 1_024;
const MANAGER: u64 = 4_096;
