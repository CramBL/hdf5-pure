use std::collections::BTreeSet;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::EditBacking;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::PersistedFreeSpacePolicy;
use hdf5_pure::PhysicalAddress;
use hdf5_pure::StoredAddress;
use hdf5_pure::SyncPolicy;
use hdf5_pure::UserblockSize;
use hdf5_pure_core::__private::{BaseAddressExt, StoredAddressExt};
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::free_space;
use test_util::range;
use test_util::temp;
use test_util::userblock::Userblock;
use test_util_hdf5::file_space;
use test_util_hdf5::file_space::persisted::{
    EOF, HEADER, HEADER_AT, HEADER_LEN, LIST, ManagerSlot, PersistedImage, PersistedStrategy,
    SECTION, STRIDE,
};
use test_util_hdf5::paged;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::storage;

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
    let path = temp::temp_path("corrupt.h5");
    let mut fixture = PersistedImage::new(&path, PersistedStrategy::Page, 16);
    fixture.manager(ManagerSlot::Metadata, &[(SECTION, 32)]);
    let mut slot = ManagerSlot::Metadata;
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
            fixture.manager(ManagerSlot::Metadata, &[(u64::MAX - 15, 32)]);
            section = Some((StoredAddress::new(u64::MAX - 15), 32));
            FormatError::OffsetOverflow {
                offset: u64::MAX - 15,
                length: 32,
            }
            .into()
        }
        Damage::PastEof | Damage::TrailingSlack | Damage::LaterUnknown => {
            slot = if damage == Damage::LaterUnknown {
                ManagerSlot::Unknown
            } else {
                ManagerSlot::Metadata
            };
            fixture.manager(slot, &[(EOF - 16, 32)]);
            section = Some((StoredAddress::new(EOF - 16), 32));
            list = Some(StoredAddress::new(
                HEADER + u64::try_from(slot.index()).unwrap() * STRIDE + HEADER_LEN,
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
            slot = ManagerSlot::Unknown;
            fixture.info.manager_addrs[slot.index()] =
                fixture.info.manager_addrs[ManagerSlot::Metadata.index()];
            list = None;
            FormatError::InvalidFreeSpaceManager(format!(
                "the free-space manager at stored address {HEADER} is recorded in slots 0 and 7"
            ))
            .into()
        }
        Damage::AllocationOverlap => {
            fixture.manager(ManagerSlot::Unknown, &[(SECTION + 64, 32)]);
            fixture.bytes[HEADER_AT + 70..HEADER_AT + 78]
                .copy_from_slice(&(7 * STRIDE).to_le_bytes());
            slot = ManagerSlot::Unknown;
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
                slot = ManagerSlot::Unknown;
                fixture.manager(slot, &[(address, 32)]);
                list = Some(StoredAddress::new(HEADER + 7 * STRIDE + HEADER_LEN));
            } else {
                fixture.manager(ManagerSlot::Metadata, &[(SECTION, 32), (address, 32)]);
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
        HEADER + u64::try_from(slot.index()).unwrap() * STRIDE
    };
    assert_eq!(
        (
            manager_slot,
            header_address,
            section_list_address,
            actual_section
        ),
        (
            slot.index(),
            StoredAddress::new(expected_header),
            list,
            section
        )
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

#[cfg(not(feature = "checksum"))]
#[rstest]
fn strict_open_checks_persisted_checksums_without_the_checksum_feature(
    #[values(false, true)] header: bool,
    #[values(
        MemoryStrategy::Bounded,
        MemoryStrategy::Mirrored,
        MemoryStrategy::Auto
    )]
    memory: MemoryStrategy,
) {
    let path = temp::temp_path("checksums.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root().delete("b").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    let report = file.persisted_free_sections().unwrap().unwrap();
    let manager = &report.managers()[0];
    let slot = manager.slot();
    let address = manager.address();
    drop(file);
    let mut bytes = std::fs::read(&path).unwrap();
    let widths = format::FormatWidths::from_sizes(8, 8).unwrap();
    let at = usize::try_from(address.get()).unwrap();
    let parsed = format::FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap();
    let list = parsed.section_list_addr();
    let checksum_at = if header {
        at + usize::try_from(format::free_space_manager_header_len(widths)).unwrap() - 4
    } else {
        usize::try_from(list.get() + parsed.section_list_used()).unwrap() - 4
    };
    let checksum = u32::from_le_bytes(bytes[checksum_at..checksum_at + 4].try_into().unwrap());
    bytes[checksum_at] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let opened = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    );
    let error = opened.unwrap_err();
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
        (slot, address, (!header).then_some(list), None)
    );
    let Error::Format(hdf5_pure::FormatError::ChecksumMismatch { expected, computed }) = *source
    else {
        panic!("{source:?}")
    };
    assert_eq!((expected, computed), (checksum ^ 1, checksum));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let widths = format::FormatWidths::from_sizes(8, 8).unwrap();
    if header {
        let parsed = format::FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap();
        assert_eq!(parsed.section_list_addr(), list);
    } else {
        let sections = format::parse_section_info(
            widths,
            &bytes[usize::try_from(list.get()).unwrap()..],
            address,
            &parsed,
        )
        .unwrap();
        assert_eq!(sections.len(), manager.free_sections().section_count());
    }
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded, 0, EditBacking::Bounded)]
#[case::mirrored(MemoryStrategy::Mirrored, 0, EditBacking::Mirrored)]
#[case::auto(MemoryStrategy::Auto, 0, EditBacking::Bounded)]
#[case::mirrored_userblock(MemoryStrategy::Mirrored, PAGE, EditBacking::Mirrored)]
#[case::auto_userblock(MemoryStrategy::Auto, PAGE, EditBacking::Mirrored)]
fn strict_persisted_open_preserves_reuse_and_policy(
    #[case] memory: MemoryStrategy,
    #[case] userblock: u64,
    #[case] backing: EditBacking,
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(false, true)] buffered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("strict.h5");
    let tolerant_path = directory.path().join("tolerant.h5");
    let stamped = write_fixture(&path, strategy, userblock);
    let hole = file_space::contiguous_extent(&File::open(&path).unwrap(), "b");
    assert_eq!(hole.end - hole.start, HOLE_BYTES);
    let first = File::open_rw(&path).unwrap();
    first.root().delete("b").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let persisted = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    let expected = persisted
        .free_sections()
        .sections()
        .iter()
        .filter(|section| {
            section.address().get() + userblock >= hole.start
                && section.end_address().get() + userblock <= hole.end
                && section.length().get() >= 1200
        })
        .map(|section| section.address().get() + userblock)
        .min()
        .unwrap();
    std::fs::copy(&path, &tolerant_path).unwrap();
    let properties = FileAccessProperties::new()
        .with_memory_strategy(memory)
        .with_sync_policy(SyncPolicy::OnClose)
        .with_page_buffer_size(if buffered { 1 << 20 } else { 0 });
    let tolerant = File::open_rw_with_options(&tolerant_path, properties).unwrap();
    let strict = File::open_rw_with_options(
        &path,
        properties.with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    )
    .unwrap();
    assert_eq!(
        strict.access_properties().persisted_free_space_policy(),
        PersistedFreeSpacePolicy::Strict
    );
    assert_eq!(
        tolerant.access_properties().persisted_free_space_policy(),
        PersistedFreeSpacePolicy::Tolerant
    );
    assert_eq!(
        strict.reusable_free_sections().unwrap(),
        tolerant.reusable_free_sections().unwrap()
    );
    assert_eq!(strict.edit_backing(), Some(backing));
    assert_eq!(strict.edit_backing(), tolerant.edit_backing());
    strict
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    tolerant
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    strict.commit().unwrap();
    tolerant.commit().unwrap();
    assert_eq!(
        file_space::contiguous_extent(&strict, "d"),
        file_space::contiguous_extent(&tolerant, "d")
    );
    assert_eq!(
        file_space::contiguous_extent(&strict, "d"),
        expected..expected + 1200
    );
    assert_eq!(strict.file_size(), tolerant.file_size());
    strict.close().unwrap();
    tolerant.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [9; 300]);
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), [1; 100]);
    stamped.assert_unchanged(&path);
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::mirrored(MemoryStrategy::Mirrored, 0)]
#[case::auto(MemoryStrategy::Auto, 0)]
#[case::mirrored_userblock(MemoryStrategy::Mirrored, PAGE)]
#[case::auto_userblock(MemoryStrategy::Auto, PAGE)]
fn strict_persisted_open_rejects_invalid_checksums_without_writes(
    #[case] memory: MemoryStrategy,
    #[case] userblock: u64,
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(false, true)] header: bool,
    #[values(false, true)] buffered: bool,
) {
    let path = temp::temp_path("invalid-manager.h5");
    write_fixture(&path, strategy, userblock);
    let first = File::open_rw(&path).unwrap();
    first.root().delete("b").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let reader = File::open(&path).unwrap();
    let report = reader.persisted_free_sections().unwrap().unwrap();
    let manager = &report.managers()[0];
    let slot = manager.slot();
    let header_address = manager.address();
    let flags = reader.superblock().consistency_flags();
    let header_at = usize::try_from(
        reader
            .superblock()
            .base_address
            .absolute(header_address)
            .unwrap(),
    )
    .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let widths = format::FormatWidths::from_sizes(8, 8).unwrap();
    let decoded = format::FreeSpaceManagerHeader::parse(widths, &bytes[header_at..]).unwrap();
    let list_address = decoded.section_list_addr();
    let list_at = usize::try_from(
        reader
            .superblock()
            .base_address
            .absolute(list_address)
            .unwrap(),
    )
    .unwrap();
    drop(reader);
    let checksum_at = if header {
        header_at + usize::try_from(format::free_space_manager_header_len(widths)).unwrap() - 4
    } else {
        list_at + usize::try_from(decoded.section_list_used()).unwrap() - 4
    };
    let checksum = u32::from_le_bytes(bytes[checksum_at..checksum_at + 4].try_into().unwrap());
    bytes[checksum_at] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let properties = FileAccessProperties::new()
        .with_memory_strategy(memory)
        .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
        .with_sync_policy(SyncPolicy::OnClose)
        .with_page_buffer_size(if buffered { 1 << 20 } else { 0 });
    let error = File::open_rw_with_options(&path, properties).unwrap_err();
    let Error::PersistedFreeSpace {
        manager_slot,
        header_address: actual_header,
        section_list_address,
        section,
        source,
    } = error
    else {
        panic!("{error:?}");
    };
    assert_eq!(
        (manager_slot, actual_header, section_list_address, section),
        (
            slot,
            header_address,
            (!header).then_some(list_address),
            None
        )
    );
    let Error::Format(hdf5_pure::FormatError::ChecksumMismatch { expected, computed }) = *source
    else {
        panic!("{source:?}");
    };
    assert_eq!((expected, computed), (checksum ^ 1, checksum));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        u64::try_from(bytes.len()).unwrap()
    );
    assert_eq!(
        File::open(&path).unwrap().superblock().consistency_flags(),
        flags
    );
}

#[rstest]
fn strict_persisted_open_rejects_a_truncated_logical_file(
    #[values(
        MemoryStrategy::Bounded,
        MemoryStrategy::Mirrored,
        MemoryStrategy::Auto
    )]
    memory: MemoryStrategy,
) {
    let path = temp::temp_path("missing-tail.h5");
    file_space::write_persisted_interior_hole(&path);
    let reader = File::open(&path).unwrap();
    let mut superblock = reader.superblock().clone();
    drop(reader);
    let mut bytes = std::fs::read(&path).unwrap();
    let actual_len = u64::try_from(bytes.len()).unwrap();
    superblock.eof_address = actual_len + 64;
    let encoded = format::serialize_superblock(&superblock).unwrap();
    bytes[..encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, &bytes).unwrap();
    let error = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    )
    .unwrap_err();
    let Error::Format(hdf5_pure::FormatError::TruncatedFile {
        expected_eof,
        actual_len: observed,
    }) = error
    else {
        panic!("{error:?}");
    };
    assert_eq!((expected_eof, observed), (actual_len + 64, actual_len));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[rstest]
fn strict_persisted_open_accepts_absent_or_empty_persistence(#[values(false, true)] persist: bool) {
    assert_eq!(
        FileAccessProperties::new().persisted_free_space_policy(),
        PersistedFreeSpacePolicy::Tolerant
    );
    assert_eq!(
        FileAccessProperties::default().persisted_free_space_policy(),
        PersistedFreeSpacePolicy::Tolerant
    );
    let path = temp::temp_path("empty.h5");
    let mut builder = FileBuilder::new();
    if persist {
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
        );
    }
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    )
    .unwrap();
    assert_eq!(
        file.access_properties().persisted_free_space_policy(),
        PersistedFreeSpacePolicy::Strict
    );
    assert_eq!(
        file.reusable_space_stats().unwrap().reusable_free_bytes(),
        0
    );
    file.close().unwrap();
}

#[rstest]
#[case::overlap(16)]
#[case::duplicate(0)]
fn persisted_inspection_reports_overlapping_extents(
    #[case] displacement: u64,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let path = temp::temp_path("overlapping-sections.h5");
    file_space::write_persisted_interior_hole(&path);
    let file = File::open(&path).unwrap();
    let report = file.persisted_free_sections().unwrap().unwrap();
    let header_address = report.managers()[0].address();
    let hole = report
        .free_sections()
        .sections()
        .iter()
        .max_by_key(|section| section.length())
        .unwrap();
    assert!(hole.length().get() >= 1024);
    let previous_address = hole.address().get() + 256;
    let address = previous_address + displacement;
    let list_address = hole.address();
    let superblock = file.superblock();
    let widths = format::FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let header_at =
        usize::try_from(superblock.base_address.absolute(header_address).unwrap()).unwrap();
    let list_at = usize::try_from(superblock.base_address.absolute(list_address).unwrap()).unwrap();
    assert_eq!((widths.offsets.get(), widths.lengths.get()), (8, 8));
    let (header, list) = free_space::raw_manager_eight(
        header_address.get(),
        list_address.get(),
        &[(previous_address, 32), (address, 32)],
    );
    drop(file);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[header_at..header_at + header.len()].copy_from_slice(&header);
    bytes[list_at..list_at + list.len()].copy_from_slice(&list);
    std::fs::write(&path, bytes).unwrap();
    let file = backend.open(&path).unwrap();
    let error = file.persisted_free_sections().unwrap_err();
    let Error::OverlappingFreeSections {
        address: actual_address,
        length,
        previous_address: actual_previous,
        previous_length,
    } = error
    else {
        panic!("{error:?}");
    };
    assert_eq!(
        (actual_address, length, actual_previous, previous_length),
        (address, 32, previous_address, 32)
    );
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::mirrored(MemoryStrategy::Mirrored, 0)]
#[case::userblock(MemoryStrategy::Mirrored, 4_096)]
fn paged_manager_tail_survives_reopen(#[case] memory: MemoryStrategy, #[case] userblock: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    for i in 0..60 {
        file.root()
            .create_dataset(&format!("b{i}"), |b| {
                b.with_i32_data(&[2; 100]);
            })
            .unwrap();
        file.commit().unwrap();
    }
    let before = file.reusable_free_sections().unwrap();
    file.close().unwrap();
    let reopened = File::open_rw(&path).unwrap();
    assert_eq!(reopened.reusable_free_sections().unwrap(), before);
    assert_eq!(reopened.dataset("a").unwrap().read_i32().unwrap(), [1; 100]);
    for i in 0..60 {
        assert_eq!(
            reopened
                .dataset(&format!("b{i}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            [2; 100],
        );
    }
}

#[rstest]
fn a_session_reuses_the_free_space_an_earlier_session_persisted(
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(strategy);
    let path = temp::temp_path(&format!("persisted_reuse_{strategy:?}_{userblock}.h5"));
    let stamped = write_fixture(&path, strategy, userblock);
    let hole = file_space::contiguous_extent(&File::open(&path).unwrap(), "b");

    let first = File::open_rw(&path).unwrap();
    first.root().delete("b").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let persisted = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let second = File::open_rw(&path).unwrap();
    let available_or_reused = second.reusable_free_sections().unwrap();
    let metadata = storage::file_metadata_extents(&second, &bytes);
    assert_eq!(
        test_util_hdf5::covered_free_bytes!(&available_or_reused, &hole, PhysicalAddress)
            + range::covered_len(&metadata, &hole),
        HOLE_BYTES
    );
    let expected = persisted
        .free_sections()
        .sections()
        .iter()
        .filter(|section| {
            section.address().get() + userblock >= hole.start
                && section.end_address().get() + userblock <= hole.end
                && section.length().get() >= 1_200
        })
        .map(|section| section.address().get() + userblock)
        .min()
        .unwrap();
    let reusable = second.reusable_free_sections().unwrap();
    assert_eq!(
        reusable.section_count(),
        persisted.free_sections().section_count()
    );
    for (physical, stored) in reusable
        .sections()
        .iter()
        .zip(persisted.free_sections().sections())
    {
        let _: PhysicalAddress = physical.address();
        let _: StoredAddress = stored.address();
        assert_eq!(physical.address().get(), stored.address().get() + userblock);
        assert_eq!(physical.length(), stored.length());
    }
    second
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    second.commit().unwrap();
    second.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(
        file_space::contiguous_extent(&file, "d"),
        expected..expected + 1200
    );
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(file.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    stamped.assert_unchanged(&path);
}

#[rstest]
fn every_commit_keeps_the_userblock_and_writes_stored_manager_addresses(
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(strategy);
    let path = temp::temp_path(&format!(
        "persisted_stored_addresses_{strategy:?}_{userblock}.h5"
    ));
    let stamped = write_fixture(&path, strategy, userblock);
    let mut manager_addrs = Vec::new();
    for step in Step::ALL {
        let session = File::open_rw(&path).unwrap();
        step.apply(&session);
        session.commit().unwrap();
        session.close().unwrap();
        if strategy == Strategy::Page {
            assert_pages_homogeneous(&path);
        }
        let settled = File::open_rw(&path).unwrap();
        let reusable = settled.reusable_free_sections().unwrap();
        settled.close().unwrap();

        stamped.assert_unchanged(&path);
        let bytes = std::fs::read(&path).unwrap();
        let file = File::open(&path).unwrap();
        let free_space_manager_space = storage::manager_extents(&file, &bytes);
        let mut allocated = storage::file_metadata_extents(&file, &bytes);
        for dataset in file.root().datasets().unwrap() {
            let address = storage::root_object_address(&file, &bytes, &dataset);
            allocated.extend(storage::header_extents(file.superblock(), &bytes, address));
            allocated.extend(
                file_space::raw_extents(&file, &dataset)
                    .iter()
                    .map(|extent| (extent.start, extent.end - extent.start)),
            );
        }
        storage::assert_unavailable(&file, &allocated);
        assert!(
            reusable
                .sections()
                .iter()
                .all(|section| section.address().get() >= userblock),
            "{step:?}"
        );
        let stored_free = file.persisted_free_sections().unwrap().unwrap();
        assert_eq!(
            stored_free.free_sections().section_count(),
            reusable.section_count(),
            "{step:?}"
        );
        for (section, physical) in stored_free
            .free_sections()
            .sections()
            .iter()
            .zip(reusable.sections())
        {
            let _: StoredAddress = section.address();
            let _: PhysicalAddress = physical.address();
            assert_eq!(
                section.address().get(),
                physical.address().get() - userblock,
                "{step:?}"
            );
            assert_eq!(section.length(), physical.length(), "{step:?}");
        }
        let headers = file
            .file_space_info()
            .unwrap()
            .unwrap()
            .manager_addrs
            .iter()
            .filter_map(|&addr| match addr {
                hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored.get()),
                hdf5_pure::DecodedStoredAddress::Undefined => None,
            })
            .collect::<BTreeSet<_>>();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            free_space_manager_space
                .iter()
                .map(|&(at, _)| at)
                .filter(
                    |&at| bytes[usize::try_from(at).unwrap()..].starts_with(free_space::SIGNATURE)
                )
                .map(|at| at - userblock)
                .collect::<BTreeSet<_>>(),
            headers,
            "{step:?}"
        );
        assert_eq!(file.file_size(), std::fs::metadata(&path).unwrap().len());
        if strategy == Strategy::Page {
            assert_eq!((file.file_size() - userblock) % PAGE, 0, "{step:?}");
        }
        manager_addrs.push(headers);
    }
    assert!(
        manager_addrs.windows(2).any(|pair| pair[0] != pair[1]),
        "no commit relocated the manager: {manager_addrs:?}"
    );
}

#[rstest]
#[case::fsm_aggr_transient(Strategy::FsmAggr, false, 67_584, 0)]
#[case::fsm_aggr_persisting(Strategy::FsmAggr, true, 65_536, 1_055)]
#[case::page_persisting(Strategy::Page, true, TAIL_BYTES, PAGE)]
fn a_deletion_that_frees_the_end_of_the_file_truncates_it(
    #[case] strategy: Strategy,
    #[case] persist: bool,
    #[case] growth: u64,
    #[case] kept: u64,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(strategy);
    let path = temp::temp_path(&format!(
        "persisted_truncate_{strategy:?}_{userblock}_{persist}.h5"
    ));
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    strategy.configure(&mut builder, persist);
    builder.write(&path).unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());

    let before = std::fs::metadata(&path).unwrap().len();
    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("tail", |builder| {
            builder.with_i32_data(&vec![5; TAIL_ELEMENTS]);
        })
        .unwrap();
    session.commit().unwrap();
    let grown = std::fs::metadata(&path).unwrap().len();
    session.root().delete("tail").unwrap();
    session.commit().unwrap();
    let truncated = std::fs::metadata(&path).unwrap().len();
    let free_end = session
        .reusable_free_sections()
        .unwrap()
        .sections()
        .last()
        .map_or(0, |section| section.end_address().get());
    session.close().unwrap();

    assert_eq!((grown - before, truncated - before), (growth, kept));
    assert!(free_end <= truncated, "{free_end} is past {truncated}");
    let file = File::open(&path).unwrap();
    assert_eq!(file.file_size(), truncated);
    if strategy == Strategy::Page {
        assert_eq!(
            file.file_space_info().unwrap().unwrap().eoa_pre_fsm,
            Some(hdf5_pure::DecodedStoredAddress::Defined(
                hdf5_pure::StoredAddress::new(truncated - userblock)
            ))
        );
        assert_eq!((truncated - userblock) % PAGE, 0);
    }
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    stamped.assert_unchanged(&path);
}

#[rstest]
fn a_reopened_paged_session_reuses_each_page_type_in_its_own_pages(
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let userblock = base.userblock(Strategy::Page);
    let path = temp::temp_path(&format!("persisted_page_types_{userblock}.h5"));
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("pages").with_i32_data(&[2; 2048]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    Strategy::Page.configure(&mut builder, true);
    builder.write(&path).unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());
    let file = File::open(&path).unwrap();
    let pages = file_space::contiguous_extent(&file, "pages");
    let c = file_space::contiguous_extent(&file, "c");
    drop(file);
    assert_eq!(pages.start, userblock + 2 * PAGE);
    assert_eq!(c.end, userblock + PAGE + 800);

    let first = File::open_rw(&path).unwrap();
    first.root().delete("pages").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let persisted = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    assert!(
        persisted.free_sections().sections().iter().any(|section| {
            section.address().get() == pages.start - userblock && section.length().get() == PAGE
        }),
        "{persisted:?}"
    );

    let second = File::open_rw(&path).unwrap();
    let before = second.file_size();
    second
        .root()
        .create_group_with("g", |group| {
            group.set_attr("payload", AttrValue::U8Array(vec![7; 3000]));
        })
        .unwrap();
    second
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    second.commit().unwrap();
    assert_eq!(second.file_size(), before);
    second.close().unwrap();
    assert_pages_homogeneous(&path);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let mut new_metadata = storage::file_metadata_extents(&file, &bytes)
        .iter()
        .map(|&(at, _)| at)
        .collect::<BTreeSet<_>>();
    for name in ["g", "d"] {
        let at = storage::root_object_address(&file, &bytes, name);
        new_metadata.insert(at);
    }
    assert!(
        new_metadata.contains(&pages.start),
        "the group header {new_metadata:?} does not start the freed page"
    );
    assert!(
        new_metadata
            .iter()
            .any(|&at| (userblock..userblock + PAGE).contains(&at)),
        "no header {new_metadata:?} reused the first metadata page"
    );
    let file = File::open(&path).unwrap();
    assert_eq!(
        file_space::contiguous_extent(&file, "d"),
        c.end..c.end + 1200
    );
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), vec![9; 300]);
    assert_eq!(file.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    stamped.assert_unchanged(&path);
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

#[derive(Clone, Copy, Debug)]
enum Step {
    AddD,
    AddE,
    DeleteB,
    DeleteC,
}

impl Step {
    const ALL: [Self; 4] = [Self::DeleteB, Self::AddD, Self::DeleteC, Self::AddE];

    fn apply(self, session: &File) {
        match self {
            Self::AddD => {
                session
                    .root()
                    .create_dataset("d", |builder| {
                        builder.with_i32_data(&[9; 300]);
                    })
                    .unwrap();
            }
            Self::AddE => {
                session
                    .root()
                    .create_dataset("e", |builder| {
                        builder.with_i32_data(&[8; 50]);
                    })
                    .unwrap();
            }
            Self::DeleteB => session.root().delete("b").unwrap(),
            Self::DeleteC => session.root().delete("c").unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Base {
    Userblock,
    Zero,
}

impl Base {
    /// Returns the userblock size, one page under `Strategy::Page`, since the writer requires a
    /// paged file's userblock to be a whole number of pages.
    fn userblock(self, strategy: Strategy) -> u64 {
        match (self, strategy) {
            (Self::Zero, _) => 0,
            (Self::Userblock, Strategy::FsmAggr) => USERBLOCK,
            (Self::Userblock, Strategy::Page) => PAGE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Strategy {
    FsmAggr,
    Page,
}

impl Strategy {
    fn configure(self, builder: &mut FileBuilder, persist: bool) {
        match self {
            Self::FsmAggr => {
                builder.with_file_space_settings(
                    FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(persist),
                );
            }
            Self::Page => {
                builder.with_file_space_settings(
                    FileSpaceSettings::new(FileSpaceStrategy::Page)
                        .with_persistence(persist)
                        .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
                );
            }
        }
    }
}

fn assert_pages_homogeneous(path: &Path) {
    let file = File::open(path).unwrap();
    let datasets = file.root().datasets().unwrap();
    file.close().unwrap();
    paged::assert_pages_homogeneous(
        path,
        PAGE,
        &datasets.iter().map(String::as_str).collect::<Vec<_>>(),
    );
}

fn write_fixture(path: &Path, strategy: Strategy, userblock: u64) -> Userblock {
    let mut builder = userblock_builder(userblock);
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    strategy.configure(&mut builder, true);
    builder.write(path).unwrap();
    Userblock::read(path, usize::try_from(userblock).unwrap())
}

fn userblock_builder(userblock: u64) -> FileBuilder {
    let mut builder = FileBuilder::new();
    if userblock > 0 {
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        builder.with_userblock_content(USERBLOCK_CONTENT);
    }
    builder
}

const HOLE_BYTES: u64 = 1600;
const PAGE: u64 = 4096;
const TAIL_BYTES: u64 = 4 * TAIL_ELEMENTS as u64;
const TAIL_ELEMENTS: usize = 16_384;
const USERBLOCK: u64 = 512;
const USERBLOCK_CONTENT: &[u8] = b"persisted free space";
