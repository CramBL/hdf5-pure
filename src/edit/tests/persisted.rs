use std::path::Path;

use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::Extent;
use rstest::rstest;

use crate::DecodedStoredAddress;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpaceInfo;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::StoredAddress;
use crate::Superblock;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddressExt;
use crate::edit::WriteEngine;
use crate::file_lock::FileLocking;
use crate::source::BytesSource;
use crate::source::MetadataCacheConfig;

#[rstest]
#[case::zero(Damage::Zero)]
#[case::overflow(Damage::Overflow)]
#[case::past_eof(Damage::PastEof)]
#[case::slack(Damage::TrailingSlack)]
#[case::duplicate_manager(Damage::DuplicateManager)]
#[case::allocation_overlap(Damage::AllocationOverlap)]
#[case::free_overlap(Damage::FreeOverlap)]
#[case::free_duplicate(Damage::FreeDuplicate)]
#[case::cross_overlap(Damage::CrossOverlap)]
#[case::later_unknown(Damage::LaterUnknown)]
#[case::used_capacity(Damage::UsedCapacity)]
#[case::missing_capacity(Damage::MissingCapacity)]
#[case::unsupported_header(Damage::UnsupportedHeader)]
#[case::unsupported_list(Damage::UnsupportedList)]
#[case::invalid_class(Damage::InvalidClass)]
fn strict_persisted_corruption_has_context_and_preserves_the_file(
    #[case] damage: Damage,
    #[values(
        MemoryStrategy::Bounded,
        MemoryStrategy::Mirrored,
        MemoryStrategy::Auto
    )]
    memory: MemoryStrategy,
    #[values(false, true)] buffered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corrupt.h5");
    let mut fixture = Fixture::new(&path, FileSpaceStrategy::Page, 16);
    fixture.manager(0, &[(SECTION, 32)]);
    let mut slot = 0;
    let mut section = None;
    let mut list = Some(StoredAddress::new(HEADER + HEADER_LEN));
    let expected = match damage {
        Damage::Zero => {
            fixture.bytes[LIST + 14] = 0;
            FormatError::InvalidFreeSpaceManager(
                "the set at byte 13 of the section list holds sections of 0 bytes".into(),
            )
            .into()
        }
        Damage::Overflow => {
            fixture.manager(0, &[(u64::MAX - 15, 32)]);
            section = Some((StoredAddress::new(u64::MAX - 15), 32));
            FormatError::OffsetOverflow {
                offset: u64::MAX - 15,
                length: 32,
            }
            .into()
        }
        Damage::PastEof | Damage::TrailingSlack | Damage::LaterUnknown => {
            slot = if damage == Damage::LaterUnknown { 7 } else { 0 };
            fixture.manager(slot, &[(EOF - 16, 32)]);
            section = Some((StoredAddress::new(EOF - 16), 32));
            list = Some(StoredAddress::new(
                HEADER + u64::try_from(slot).unwrap() * STRIDE + HEADER_LEN,
            ));
            if damage == Damage::TrailingSlack {
                fixture.bytes.resize(usize::try_from(EOF + 64).unwrap(), 0);
            }
            FormatError::InvalidFreeSpaceManager(format!(
                "the free section at {}, 32 bytes long, exceeds the file allocation",
                EOF - 16
            ))
            .into()
        }
        Damage::DuplicateManager => {
            slot = 7;
            fixture.info.manager_addrs[slot] = fixture.info.manager_addrs[0];
            list = None;
            FormatError::InvalidFreeSpaceManager(format!(
                "the free-space manager at stored address {HEADER} is recorded in slots 0 and 7"
            ))
            .into()
        }
        Damage::AllocationOverlap => {
            fixture.manager(7, &[(SECTION + 64, 32)]);
            fixture.bytes[HEADER_AT + 70..HEADER_AT + 78]
                .copy_from_slice(&(7 * STRIDE).to_le_bytes());
            slot = 7;
            list = None;
            FormatError::InvalidFreeSpaceManager(format!(
                "the manager allocation at {}, {HEADER_LEN} bytes long, overlaps the manager allocation at {}, {} bytes long",
                HEADER + 7 * STRIDE, HEADER + HEADER_LEN, 7 * STRIDE
            )).into()
        }
        Damage::FreeOverlap | Damage::FreeDuplicate | Damage::CrossOverlap => {
            let address = SECTION
                + if damage == Damage::FreeDuplicate {
                    0
                } else {
                    16
                };
            if damage == Damage::CrossOverlap {
                slot = 7;
                fixture.manager(slot, &[(address, 32)]);
                list = Some(StoredAddress::new(HEADER + 7 * STRIDE + HEADER_LEN));
            } else {
                fixture.manager(0, &[(SECTION, 32), (address, 32)]);
            }
            section = Some((StoredAddress::new(address), 32));
            Error::OverlappingFreeSections {
                address,
                length: 32,
                previous_address: SECTION,
                previous_length: 32,
            }
        }
        Damage::UsedCapacity => {
            list = None;
            fixture.bytes[HEADER_AT + 62..HEADER_AT + 70].copy_from_slice(&36u64.to_le_bytes());
            FormatError::InvalidFreeSpaceManager(
                "the section list uses 36 bytes, more than the 35 bytes allocated to it".into(),
            )
            .into()
        }
        Damage::MissingCapacity => {
            fixture.bytes[HEADER_AT + 70..HEADER_AT + 78]
                .copy_from_slice(&(EOF - HEADER - HEADER_LEN).to_le_bytes());
            fixture.bytes.truncate(LIST + 35);
            FormatError::UnexpectedSourceEof {
                expected: EOF,
                available: u64::try_from(LIST + 35).unwrap(),
            }
            .into()
        }
        Damage::UnsupportedHeader => {
            fixture.bytes[HEADER_AT + 4] = 1;
            list = None;
            FormatError::UnsupportedVersion(1).into()
        }
        Damage::UnsupportedList => {
            fixture.bytes[LIST + 4] = 1;
            FormatError::UnsupportedVersion(1).into()
        }
        Damage::InvalidClass => {
            fixture.bytes[LIST + 30] = 3;
            FormatError::InvalidFreeSpaceManager(
                "the class of the section at byte 22 of the section list is 3, not one of the 3 classes of the file client".into(),
            ).into()
        }
    };
    test_util::checksum::restamp(
        &mut fixture.bytes,
        HEADER_AT,
        usize::try_from(HEADER_LEN).unwrap(),
    );
    if damage != Damage::UsedCapacity {
        let used = u64::from_le_bytes(
            fixture.bytes[HEADER_AT + 62..HEADER_AT + 70]
                .try_into()
                .unwrap(),
        );
        test_util::checksum::restamp(&mut fixture.bytes, LIST, usize::try_from(used).unwrap());
    }
    let bytes = fixture.write(&path);
    let error = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_page_buffer_size(if buffered { 1 << 20 } else { 0 })
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    )
    .unwrap_err();
    let Error::PersistedFreeSpace {
        manager_slot,
        header_address,
        section_list_address,
        section: actual_section,
        source,
    } = error
    else {
        panic!("{error:?}")
    };
    let expected_header = if damage == Damage::DuplicateManager {
        HEADER
    } else {
        HEADER + u64::try_from(slot).unwrap() * STRIDE
    };
    assert_eq!(
        (
            manager_slot,
            header_address,
            section_list_address,
            actual_section
        ),
        (slot, StoredAddress::new(expected_header), list, section)
    );
    match expected {
        Error::Format(expected) => {
            let Error::Format(actual) = *source else {
                panic!("{source:?}")
            };
            assert_eq!(actual, expected);
        }
        Error::OverlappingFreeSections {
            address,
            length,
            previous_address,
            previous_length,
        } => {
            let Error::OverlappingFreeSections {
                address: actual_address,
                length: actual_length,
                previous_address: actual_previous,
                previous_length: actual_previous_length,
            } = *source
            else {
                panic!("{source:?}")
            };
            assert_eq!(
                (
                    actual_address,
                    actual_length,
                    actual_previous,
                    actual_previous_length
                ),
                (address, length, previous_address, previous_length)
            );
        }
        error => panic!("{error:?}"),
    }
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        u64::try_from(bytes.len()).unwrap()
    );
}

#[test]
fn strict_persisted_later_failure_precedes_allocator_construction() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("later-manager.h5");
    let mut fixture = Fixture::new(&path, FileSpaceStrategy::Page, 16);
    fixture.manager(0, &[(SECTION, 32)]);
    fixture.manager(7, &[(EOF - 16, 32)]);
    let bytes = fixture.write(&path);
    let built = std::cell::Cell::new(false);
    let Err(error) = WriteEngine::open_imaged_with_policy(
        &path,
        Some(FileLocking::Enabled),
        PersistedFreeSpacePolicy::Strict,
        |handle, _| {
            built.set(true);
            Ok(Box::new(WriteEngine::read_mirror(handle)?))
        },
    ) else {
        panic!("expected invalid section geometry")
    };
    let Error::PersistedFreeSpace {
        manager_slot,
        header_address,
        section_list_address,
        section,
        source,
    } = error
    else {
        panic!("{error:?}")
    };
    assert_eq!(
        (manager_slot, header_address, section_list_address, section),
        (
            7,
            StoredAddress::new(HEADER + 7 * STRIDE),
            Some(StoredAddress::new(HEADER + 7 * STRIDE + HEADER_LEN)),
            Some((StoredAddress::new(EOF - 16), 32))
        )
    );
    let Error::Format(FormatError::InvalidFreeSpaceManager(reason)) = *source else {
        panic!("{source:?}")
    };
    assert_eq!(
        reason,
        format!(
            "the free section at {}, 32 bytes long, exceeds the file allocation",
            EOF - 16
        )
    );
    assert!(!built.get());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[rstest]
#[case::same_manager(false)]
#[case::different_managers(true)]
fn strict_persisted_adjacent_sections_are_admitted(#[case] separate: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("adjacent.h5");
    let mut fixture = Fixture::new(&path, FileSpaceStrategy::FsmAggr, 128);
    if separate {
        fixture.manager(0, &[(SECTION, 32)]);
        fixture.manager(7, &[(SECTION + 32, 32)]);
    } else {
        fixture.manager(0, &[(SECTION, 32), (SECTION + 32, 32)]);
    }
    fixture.manager(5, &[]);
    let bytes = fixture.write(&path);
    let engine = strict_engine(&path);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [Extent::new(SECTION, 64).unwrap()]
    );
    assert_eq!(
        engine.persist.as_ref().unwrap().old_blocks.len(),
        if separate { 7 } else { 5 }
    );
    drop(engine);
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}

#[rstest]
#[case::metadata(0, SECTION + 16, 8, true, false, false)]
#[case::raw(2, SECTION + 16, 8, false, true, false)]
#[case::generic_page(6, SECTION, 4096, false, true, false)]
#[case::generic_fragment(6, SECTION + 16, 8, false, false, true)]
#[case::unknown(7, SECTION + 16, 8, false, false, true)]
fn strict_persisted_page_roles_preserve_admission(
    #[case] slot: usize,
    #[case] address: u64,
    #[case] length: u64,
    #[case] metadata: bool,
    #[case] raw: bool,
    #[case] unclassified: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("page-role.h5");
    let mut fixture = Fixture::new(&path, FileSpaceStrategy::Page, 128);
    fixture.manager(slot, &[(address, length)]);
    fixture.manager(5, &[]);
    let bytes = fixture.write(&path);
    let engine = strict_engine(&path);
    let pages = engine.space.paged_sections().unwrap();
    let extent = [Extent::new(address, length).unwrap()];
    assert_eq!(pages.metadata(), if metadata { &extent[..] } else { &[] });
    assert_eq!(pages.raw(), if raw { &extent[..] } else { &[] });
    assert_eq!(
        pages.unclassified(),
        if unclassified { &extent[..] } else { &[] }
    );
    assert_eq!(engine.persist.as_ref().unwrap().old_blocks.len(), 5);
    drop(engine);
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}

fn strict_engine(path: &Path) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Damage {
    AllocationOverlap,
    CrossOverlap,
    DuplicateManager,
    FreeDuplicate,
    FreeOverlap,
    InvalidClass,
    LaterUnknown,
    MissingCapacity,
    Overflow,
    PastEof,
    TrailingSlack,
    UnsupportedHeader,
    UnsupportedList,
    UsedCapacity,
    Zero,
}

struct Fixture {
    bytes: Vec<u8>,
    info: FileSpaceInfo,
    original_info: Vec<u8>,
    superblock: Superblock,
    extension: (usize, usize),
}

impl Fixture {
    fn new(path: &Path, strategy: FileSpaceStrategy, threshold: u64) -> Self {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(strategy)
                .with_persistence(true)
                .with_threshold(threshold),
        );
        builder.write(path).unwrap();
        let file = File::open(path).unwrap();
        let mut info = file.file_space_info().unwrap().unwrap();
        let mut superblock = file.superblock().clone();
        drop(file);
        let mut bytes = std::fs::read(path).unwrap();
        let original_info =
            format::serialize_file_space_info_v1(crate::file_writer::WIDTHS, &info).unwrap();
        let chunks = crate::edit::object_header::read_oh_chunks(
            &BytesSource::new(&bytes),
            superblock.superblock_extension_address.unwrap().get(),
            AddressContext::from_effective_base(BaseAddress::ZERO),
        )
        .unwrap();
        assert_eq!(chunks.len(), 1);
        let (at, length) = chunks[0].span;
        info.manager_addrs.fill(DecodedStoredAddress::Undefined);
        superblock.eof_address = EOF;
        bytes.resize(usize::try_from(EOF).unwrap(), 0);
        Self {
            bytes,
            info,
            original_info,
            superblock,
            extension: (
                usize::try_from(at).unwrap(),
                usize::try_from(length).unwrap(),
            ),
        }
    }

    fn manager(&mut self, slot: usize, sections: &[(u64, u64)]) {
        let address = HEADER + u64::try_from(slot).unwrap() * STRIDE;
        let (header, list) = format::serialize_free_space_manager(
            crate::file_writer::WIDTHS,
            &sections
                .iter()
                .map(|&(addr, size)| format::FreeSection {
                    addr: StoredAddress::new(addr),
                    size,
                })
                .collect::<Vec<_>>(),
            StoredAddress::new(address),
            StoredAddress::new(address + HEADER_LEN),
            format::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let at = usize::try_from(address).unwrap();
        self.bytes[at..at + header.len()].copy_from_slice(&header);
        self.bytes[at + header.len()..at + header.len() + list.len()].copy_from_slice(&list);
        self.info.manager_addrs[slot] = DecodedStoredAddress::Defined(StoredAddress::new(address));
    }

    fn write(self, path: &Path) -> Vec<u8> {
        let Self {
            mut bytes,
            info,
            original_info,
            superblock,
            extension: (at, length),
        } = self;
        let payload =
            format::serialize_file_space_info_v1(crate::file_writer::WIDTHS, &info).unwrap();
        let offset = bytes[at..at + length]
            .windows(original_info.len())
            .position(|window| window == original_info)
            .unwrap()
            + at;
        assert_eq!(payload.len(), original_info.len());
        bytes[offset..offset + payload.len()].copy_from_slice(&payload);
        test_util::checksum::restamp(&mut bytes, at, length);
        let encoded = format::serialize_superblock(&superblock).unwrap();
        bytes[..encoded.len()].copy_from_slice(&encoded);
        std::fs::write(path, &bytes).unwrap();
        bytes
    }
}

const EOF: u64 = 65_536;
const HEADER: u64 = 8192;
const HEADER_AT: usize = 8192;
const HEADER_LEN: u64 = 82;
const LIST: usize = HEADER_AT + 82;
const SECTION: u64 = 16_384;
const STRIDE: u64 = 256;
