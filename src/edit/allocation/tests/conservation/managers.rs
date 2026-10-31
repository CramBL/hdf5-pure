//! Checks manager rejection, retained values, and source error propagation.

use std::fs;
use std::io::Read;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::FreeSpaceManagerHeaderRef;
use rstest::rstest;
use test_util::allocation_oracle::AllocationKind;
use test_util::disk_log;
use test_util::free_space_damage::CHECKSUM_BYTES;
use test_util::free_space_damage::Damage;
use test_util::free_space_damage::HEADER_BYTES;
use test_util::free_space_damage::ManagerLayout;

use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FormatError;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::address::StoredAddressExt;
use crate::edit::WriteEngine;
use crate::edit::context::FileContext;
use crate::free_sections::FreeSectionsError;
use crate::free_sections::PersistedManagerContext;
use crate::free_sections::PersistedManagerSet;
use crate::free_sections::PersistedManagerSource;
use crate::source::BytesSource;
use crate::source::ReadSeekSource;
use crate::source::Source;

use super::Observation;
use super::WritePath;

#[rstest]
#[case::header_signature(Damage::HeaderSignature)]
#[case::header_version(Damage::HeaderVersion)]
#[case::header_client(Damage::HeaderClient)]
#[case::header_bits(Damage::HeaderBits)]
#[case::header_checksum(Damage::HeaderChecksum)]
#[case::used_length(Damage::UsedLength)]
#[case::undefined_list(Damage::UndefinedList)]
#[case::list_address(Damage::ListAddress)]
#[case::allocated_length(Damage::AllocatedLength)]
#[case::list_signature(Damage::ListSignature)]
#[case::list_version(Damage::ListVersion)]
#[case::list_owner(Damage::ListOwner)]
#[case::list_checksum(Damage::ListChecksum)]
fn authoritative_manager_damage_is_rejected_before_writes(
    #[case] damage: Damage,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(WritePath::Immediate, WritePath::Borrowed, WritePath::Owned)] writer: WritePath,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manager.h5");
    super::pure_fixture(&path, paged, deferred, false);
    reject(&path, damage, writer);
}

#[rstest]
#[case::header_signature(Damage::HeaderSignature)]
#[case::header_version(Damage::HeaderVersion)]
#[case::header_client(Damage::HeaderClient)]
#[case::header_bits(Damage::HeaderBits)]
#[case::header_checksum(Damage::HeaderChecksum)]
#[case::used_length(Damage::UsedLength)]
#[case::undefined_list(Damage::UndefinedList)]
#[case::list_address(Damage::ListAddress)]
#[case::allocated_length(Damage::AllocatedLength)]
#[case::list_signature(Damage::ListSignature)]
#[case::list_version(Damage::ListVersion)]
#[case::list_owner(Damage::ListOwner)]
#[case::list_checksum(Damage::ListChecksum)]
fn strict_open_preserves_bytes_after_manager_rejection(
    #[case] damage: Damage,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("strict.h5");
    super::pure_fixture(&path, paged, deferred, false);
    let original = fs::read(&path).unwrap();
    let manager = Manager::read(&original);
    let (damaged, expected) = self::damage(&original, damage);
    fs::write(&path, &damaged).unwrap();
    disk_log::start();
    let error = match File::open_rw_with_options(
        &path,
        options().with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    ) {
        Ok(_) => panic!("a malformed manager was accepted"),
        Err(error) => error,
    };
    let recording = disk_log::finish();
    let Error::PersistedFreeSpace {
        manager_slot,
        header_address,
        section_list_address,
        section,
        source,
    } = error
    else {
        panic!("expected persisted manager context, got {error:?}");
    };
    assert_eq!(manager_slot, manager.slot);
    assert_eq!(header_address.get(), u64::try_from(manager.header).unwrap());
    assert_eq!(section, None);
    let expected_list = match damage {
        Damage::HeaderSignature
        | Damage::HeaderVersion
        | Damage::HeaderBits
        | Damage::HeaderChecksum
        | Damage::UsedLength
        | Damage::UndefinedList => None,
        _ => Some(
            u64::try_from(if matches!(damage, Damage::ListAddress) {
                damaged.len()
            } else {
                manager.list
            })
            .unwrap(),
        ),
    };
    assert_eq!(
        section_list_address.map(|address| address.get()),
        expected_list
    );
    assert!(
        matches!(&*source, Error::Format(actual) if *actual == expected),
        "{source:?}"
    );
    assert!(recording.operations.is_empty(), "{recording:?}");
    assert_eq!(fs::read(&path).unwrap(), damaged);
}

#[rstest]
#[case::header(ManagerPart::Header)]
#[case::section_list(ManagerPart::SectionList)]
fn supplied_manager_bytes_and_physical_truncation_have_distinct_errors(
    #[case] target: ManagerPart,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("region.h5");
    super::pure_fixture(&path, false, false, false);
    let bytes = fs::read(&path).unwrap();
    let manager = Manager::read(&bytes);
    let context = FileContext::new(crate::superblock::read(bytes.as_slice()).unwrap()).unwrap();
    let reference = FreeSpaceManagerHeaderRef::new(
        context.widths(),
        PhysicalAddress::new(u64::try_from(manager.header).unwrap()),
    )
    .unwrap();
    let header_range = reference.parse_range();
    let frame = reference
        .parse_checked(region(&bytes, header_range))
        .unwrap();
    let list = frame
        .section_list_ref(
            crate::StoredAddress::new(u64::try_from(manager.header).unwrap()),
            PhysicalAddress::new(u64::try_from(manager.list).unwrap()),
        )
        .unwrap();
    let range = match target {
        ManagerPart::Header => header_range,
        ManagerPart::SectionList => list.parse_range(),
    };
    let supplied = FileRange::new(range.start(), range.length() - 1).unwrap();
    let error = match target {
        ManagerPart::Header => reference
            .parse_checked(region(&bytes, supplied))
            .unwrap_err(),
        ManagerPart::SectionList => list.parse_checked(region(&bytes, supplied)).unwrap_err(),
    };
    assert_eq!(
        error,
        FormatError::InsufficientRegion {
            required_start: range.start().get(),
            required_length: range.length(),
            provided_start: supplied.start().get(),
            provided_length: supplied.length(),
        }
    );
    let truncated = &bytes[..usize::try_from(supplied.end().get()).unwrap()];
    let error = BytesSource::new(truncated)
        .read_exact_at(
            range.start().get(),
            usize::try_from(range.length()).unwrap(),
        )
        .unwrap_err();
    assert_eq!(
        error,
        FormatError::UnexpectedSourceEof {
            expected: range.end().get(),
            available: supplied.end().get(),
        }
    );
    let storage = directory.path().join("storage.h5");
    fs::write(&storage, &bytes).unwrap();
    let source = ReadSeekSource::new(fs::File::open(&storage).unwrap()).unwrap();
    {
        let writer = fs::OpenOptions::new().write(true).open(&storage).unwrap();
        writer.set_len(supplied.end().get()).unwrap();
    }
    let mut destination = vec![0; usize::try_from(range.length()).unwrap()];
    let expected = (&truncated[usize::try_from(range.start().get()).unwrap()..])
        .read_exact(&mut destination)
        .unwrap_err();
    assert_eq!(
        source
            .read_exact_at(range.start().get(), destination.len())
            .unwrap_err(),
        FormatError::Source(expected.to_string()),
    );
    drop(source);
    assert_eq!(fs::read(&storage).unwrap(), truncated);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[rstest]
#[case::header(ManagerPart::Header)]
#[case::section_list(ManagerPart::SectionList)]
fn authoritative_manager_reads_preserve_resource_and_source_failures(
    #[case] target: ManagerPart,
    #[values(false, true)] resource_limit: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.h5");
    super::pure_fixture(&path, false, false, false);
    let bytes = fs::read(&path).unwrap();
    let manager = Manager::read(&bytes);
    let context = FileContext::new(crate::superblock::read(bytes.as_slice()).unwrap()).unwrap();
    let (_, info) = WriteEngine::extension_fsinfo(&BytesSource::new(&bytes), &context)
        .unwrap()
        .unwrap();
    let range = match target {
        ManagerPart::Header => manager.header_range(),
        ManagerPart::SectionList => manager.list_range(),
    };
    let expected = if resource_limit {
        FormatError::ResourceLimit {
            limit: range.length() - 1,
            required: range.length(),
        }
    } else {
        FormatError::Source("injected manager read failure".into())
    };
    let source = RestrictedSource {
        bytes: &bytes,
        range,
        failure: &expected,
    };
    let error = PersistedManagerSet::read(
        &source,
        context.widths(),
        context.addresses().effective_base(),
        context.superblock().eof_address,
        &info,
    )
    .unwrap_err();
    assert_eq!(error, FreeSectionsError::Format(expected));
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

/// Corrupts a nonempty manager and returns the expected parser error.
pub(super) fn damage(original: &[u8], damage: Damage) -> (Vec<u8>, FormatError) {
    let manager = Manager::read(original);
    let layout = ManagerLayout::read(original, manager.header);
    let bytes = layout.damage(original, damage);
    let malformed = FormatError::InvalidFreeSpaceManager;
    let expected = match damage {
        Damage::HeaderSignature => {
            malformed("the header signature is b\"\\x00SHD\", not b\"FSHD\"".into())
        }
        Damage::HeaderVersion | Damage::ListVersion => FormatError::UnsupportedVersion(1),
        Damage::HeaderClient => {
            malformed("the client ID is 0, not 1, the ID of the file client".into())
        }
        Damage::HeaderBits => {
            malformed("an address space of 0 bits is not 1 to 64 bits wide".into())
        }
        Damage::HeaderChecksum | Damage::ListChecksum => {
            let range = if matches!(damage, Damage::HeaderChecksum) {
                manager.header_range()
            } else {
                manager.list_range()
            };
            let end = usize::try_from(range.end().get()).unwrap();
            FormatError::ChecksumMismatch {
                expected: test_util::bytes::u32_at(&bytes, end - CHECKSUM_BYTES),
                computed: test_util::bytes::u32_at(original, end - CHECKSUM_BYTES),
            }
        }
        Damage::UsedLength => malformed(format!(
            "the section list uses {} bytes, more than the {} bytes allocated to it",
            layout.allocated + 1,
            layout.allocated
        )),
        Damage::UndefinedList => malformed(format!(
            "the header's section count is {}, and it stores no section list",
            layout.count
        )),
        Damage::ListAddress | Damage::AllocatedLength => {
            let address = if matches!(damage, Damage::ListAddress) {
                u64::try_from(original.len()).unwrap()
            } else {
                u64::try_from(layout.list).unwrap()
            };
            let length = if matches!(damage, Damage::AllocatedLength) {
                u64::try_from(original.len()).unwrap() - address + 1
            } else {
                layout.allocated
            };
            malformed(format!(
                "the manager allocation at {address}, {length} bytes long, exceeds the file allocation"
            ))
        }
        Damage::ListSignature => {
            malformed("the section list signature is b\"\\x00SSE\", not b\"FSSE\"".into())
        }
        Damage::ListOwner => malformed(format!(
            "the section list belongs to the header at 0, not to the header at {}",
            manager.header
        )),
    };
    (bytes, expected)
}

struct Manager {
    slot: usize,
    header: usize,
    header_length: usize,
    list: usize,
    list_length: usize,
}

impl Manager {
    fn read(bytes: &[u8]) -> Self {
        let file = File::from_bytes(bytes.to_vec()).unwrap();
        let report = file.persisted_free_sections().unwrap().unwrap();
        let manager = report
            .managers()
            .iter()
            .find(|manager| !manager.free_sections().sections().is_empty())
            .unwrap();
        let header = usize::try_from(manager.address().get()).unwrap();
        Self {
            slot: manager.slot(),
            header,
            header_length: HEADER_BYTES,
            list: ManagerLayout::read(bytes, header).list,
            list_length: ManagerLayout::read(bytes, header).used,
        }
    }

    fn header_range(&self) -> FileRange {
        FileRange::new(
            PhysicalAddress::new(u64::try_from(self.header).unwrap()),
            u64::try_from(self.header_length).unwrap(),
        )
        .unwrap()
    }

    fn list_range(&self) -> FileRange {
        FileRange::new(
            PhysicalAddress::new(u64::try_from(self.list).unwrap()),
            u64::try_from(self.list_length).unwrap(),
        )
        .unwrap()
    }
}

#[derive(Clone, Copy)]
enum ManagerPart {
    Header,
    SectionList,
}

/// Injects a caller's acquisition failure at one manager read range.
struct RestrictedSource<'a> {
    bytes: &'a [u8],
    range: FileRange,
    failure: &'a FormatError,
}

impl PersistedManagerSource for RestrictedSource<'_> {
    type Error = FreeSectionsError;

    fn len(&self) -> u64 {
        u64::try_from(self.bytes.len()).unwrap()
    }

    fn read(&self, range: FileRange) -> Result<Vec<u8>, Self::Error> {
        if range == self.range {
            return Err(self.failure.clone().into());
        }
        Ok(BytesSource::new(self.bytes).read_exact_at(
            range.start().get(),
            usize::try_from(range.length()).unwrap(),
        )?)
    }

    fn contextual_error(_: PersistedManagerContext, error: Self::Error) -> Self::Error {
        error
    }
}

/// Checks rejection before file writes or resizes and reads the original dataset values.
pub(super) fn reject(path: &Path, kind: Damage, writer: WritePath) {
    let original = fs::read(path).unwrap();
    let observation = Observation::read(&original);
    observation.inspect("manager rejection", 0, &observation.free, &mut Vec::new());
    let (damaged, expected) = damage(&original, kind);
    for fact in observation
        .live
        .iter()
        .filter(|fact| matches!(fact.kind, AllocationKind::Index | AllocationKind::Payload))
    {
        let start = usize::try_from(fact.range.start()).unwrap();
        let end = usize::try_from(fact.range.end()).unwrap();
        assert_eq!(&damaged[start..end], &original[start..end]);
    }
    assert_rejected(path, &damaged, &expected, writer, &observation);
}

fn assert_rejected(
    path: &Path,
    damaged: &[u8],
    expected: &FormatError,
    writer: WritePath,
    original: &Observation,
) {
    fs::write(path, damaged).unwrap();
    disk_log::start();
    let file = File::open_rw_with_options(path, options()).unwrap();
    let result = {
        let mut dataset = file.dataset("d").unwrap();
        match writer {
            WritePath::Immediate => dataset.append(&[7i32; super::CHUNK]),
            WritePath::Borrowed => match dataset.buffered_appender() {
                Ok(mut appender) => appender.append(&[7i32; super::CHUNK]),
                Err(error) => Err(error),
            },
            WritePath::Owned => match dataset.into_buffered_appender() {
                Ok(mut appender) => appender.append(&[7i32; super::CHUNK]),
                Err(error) => Err(error),
            },
        }
    };
    let error = result.unwrap_err();
    assert!(
        matches!(&error, Error::Format(actual) if actual == expected),
        "{error:?}; expected {expected:?}"
    );
    drop(file);
    let recording = disk_log::finish();
    assert!(recording.operations.is_empty(), "{recording:?}");
    assert_eq!(fs::read(path).unwrap(), damaged);
    let file = File::from_bytes(damaged.to_vec()).unwrap();
    let error = file.persisted_free_sections().unwrap_err();
    assert!(
        matches!(&error, Error::Format(actual) if actual == expected),
        "{error:?}"
    );
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        original.values
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().read_u8().unwrap(),
        [9; super::CEILING]
    );
}

fn region(bytes: &[u8], range: FileRange) -> FileRegion<'_> {
    FileRegion::new(
        range,
        &bytes[usize::try_from(range.start().get()).unwrap()
            ..usize::try_from(range.end().get()).unwrap()],
    )
    .unwrap()
}

fn options() -> FileAccessProperties {
    FileAccessProperties::new()
        .with_memory_strategy(MemoryStrategy::Bounded)
        .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Tolerant)
}
