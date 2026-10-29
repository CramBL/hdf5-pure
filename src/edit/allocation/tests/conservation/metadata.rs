//! Checks that malformed index metadata is rejected without filesystem writes or resizes.

use std::fs;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use rstest::rstest;
use test_util::checksum;
use test_util::disk_log;
use test_util::extensible_array::CHECKSUM_BYTES;

use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FormatError;
use crate::MemoryStrategy;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::source::BytesSource;

use super::CHUNK;
use super::Observation;
use super::WritePath;

#[rstest]
#[case::header_checksum(Damage::HeaderChecksum)]
#[case::index_checksum(Damage::IndexChecksum)]
#[case::index_signature(Damage::IndexSignature)]
#[case::index_owner(Damage::IndexOwner)]
#[case::direct_checksum(Damage::DirectChecksum)]
#[case::direct_signature(Damage::DirectSignature)]
fn malformed_indexes_reject_filesystem_appends_without_publication(
    #[case] damage: Damage,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
    #[values(WritePath::Immediate, WritePath::Borrowed, WritePath::Owned)] writer: WritePath,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.h5");
    super::pure_fixture(&path, paged, deferred, filtered);
    initialize(&path);
    reject(&path, damage, writer);
}

#[derive(Clone, Copy)]
pub(super) enum Damage {
    DirectChecksum,
    DirectSignature,
    HeaderChecksum,
    IndexChecksum,
    IndexOwner,
    IndexSignature,
}

/// Appends through the first direct data block and checks the complete values.
pub(super) fn initialize(path: &Path) {
    let original = Observation::read(&fs::read(path).unwrap());
    let values = (original.values.len()..5 * CHUNK + 1)
        .map(|value| i32::try_from(value).unwrap())
        .collect::<Vec<_>>();
    WritePath::Immediate.append(path, &values);
    let completed = Observation::read(&fs::read(path).unwrap());
    assert_eq!(
        completed.values,
        (0..5 * CHUNK + 1)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>()
    );
}

/// Damages one metadata field and checks the error, file bytes, and completed write trace.
pub(super) fn reject(path: &Path, damage: Damage, writer: WritePath) {
    let original = fs::read(path).unwrap();
    let observation = Observation::read(&original);
    let context =
        super::FileContext::new(crate::superblock::read(original.as_slice()).unwrap()).unwrap();
    let file = File::from_bytes(original.clone()).unwrap();
    let DataLayout::Chunked {
        index:
            ChunkIndexLayout::ExtensibleArray {
                address: Some(address),
                ..
            },
        ..
    } = file.dataset("d").unwrap().data_layout().unwrap()
    else {
        panic!("expected an allocated extensible array")
    };
    drop(file);
    let array = crate::extensible_array::read_header(
        &BytesSource::new(&original),
        context.widths(),
        address,
    )
    .unwrap();
    let reference = array.index_block().unwrap();
    let index = reference
        .parse(region(&original, reference.parse_range()))
        .unwrap();
    let range = match damage {
        Damage::HeaderChecksum => array.range(),
        Damage::DirectChecksum | Damage::DirectSignature => {
            index.data_blocks().next().unwrap().unwrap().parse_range()
        }
        _ => reference.parse_range(),
    };
    let at = usize::try_from(range.start().get()).unwrap();
    let length = usize::try_from(range.length()).unwrap();
    let mut bytes = original.clone();
    let expected = match damage {
        Damage::HeaderChecksum | Damage::IndexChecksum | Damage::DirectChecksum => {
            let computed = test_util::bytes::u32_at(&bytes, at + length - CHECKSUM_BYTES);
            bytes[at + length - 1] ^= 1;
            FormatError::ChecksumMismatch {
                expected: test_util::bytes::u32_at(&bytes, at + length - CHECKSUM_BYTES),
                computed,
            }
        }
        Damage::IndexOwner => {
            bytes[at + 6..at + 14].fill(0);
            checksum::restamp(&mut bytes, at, length);
            FormatError::InvalidExtensibleArrayBlockOwner {
                expected: array.range().start().get(),
                actual: 0,
            }
        }
        Damage::IndexSignature | Damage::DirectSignature => {
            bytes[at] = 0;
            checksum::restamp(&mut bytes, at, length);
            FormatError::InvalidExtensibleArraySignature
        }
    };
    fs::write(path, &bytes).unwrap();
    disk_log::start();
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap();
    let values = vec![7i32; CHUNK];
    let result = {
        let mut dataset = file.dataset("d").unwrap();
        match writer {
            WritePath::Immediate => dataset.append(&values),
            WritePath::Borrowed => match dataset.buffered_appender() {
                Ok(mut appender) => appender.append(&values),
                Err(error) => Err(error),
            },
            WritePath::Owned => match dataset.into_buffered_appender() {
                Ok(mut appender) => appender.append(&values),
                Err(error) => Err(error),
            },
        }
    };
    let error = result.unwrap_err();
    assert!(
        matches!(&error, Error::Format(actual) if *actual == expected),
        "{error:?}; expected {expected:?}"
    );
    drop(file);
    let recording = disk_log::finish();
    assert!(recording.operations.is_empty(), "{recording:?}");
    assert_eq!(fs::read(path).unwrap(), bytes);
    for (_, address, length, _) in &observation.chunks {
        let start = usize::try_from(*address).unwrap();
        let end = start + usize::try_from(*length).unwrap();
        assert_eq!(&bytes[start..end], &original[start..end]);
    }
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(
        file.dataset("ceiling").unwrap().read_u8().unwrap(),
        vec![9; super::CEILING]
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
