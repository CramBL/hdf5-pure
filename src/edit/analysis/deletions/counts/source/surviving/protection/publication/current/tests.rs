use core::cell::Cell;
use core::cell::RefCell;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::MetadataSource;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;

use crate::convert;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;

use super::CurrentMetadata;
use super::CurrentMetadataError;

#[rstest]
#[case::contained(5, 2, &[])]
#[case::left(2, 4, &[(2, 2)])]
#[case::right(6, 4, &[(8, 2)])]
#[case::spanning(2, 8, &[(2, 2), (8, 2)])]
#[case::before(0, 2, &[(0, 2)])]
#[case::after(9, 3, &[(9, 3)])]
#[case::empty_before(2, 0, &[])]
#[case::empty_end(12, 0, &[])]
fn reads_reuse_the_acquired_range_and_delegate_only_uncovered_bytes(
    #[case] start: u64,
    #[case] length: usize,
    #[case] expected: &[(u64, u64)],
    #[values(ReadMode::Metadata, ReadMode::Buffer, ReadMode::Owned)] mode: ReadMode,
    #[values(Backing::Borrowed, Backing::Owned, Backing::Shared)] backing: Backing,
    #[values(0, 512, 4_294_967_296, u64::MAX - 12)] bias: u64,
) {
    let source = ObservedSource::new(backing, bias);
    let acquired = range(bias + 4, 4);
    let view = CurrentMetadata::new(&source, acquired).unwrap();
    assert_eq!(view.region().range(), acquired);
    assert_eq!(view.region().bytes(), &DATA[4..8]);
    assert_eq!(view.len(), source.len());
    assert_eq!(*source.calls.borrow(), [(ReadKind::Metadata, acquired)]);
    source.calls.borrow_mut().clear();
    let actual = match mode {
        ReadMode::Metadata => {
            let bytes = view.read_metadata(bias + start, length).unwrap();
            assert_eq!(bytes.range(), range(bias + start, length.to_u64()));
            bytes.as_ref().to_vec()
        }
        ReadMode::Buffer => {
            let mut bytes = vec![0; length];
            view.read_at(bias + start, &mut bytes).unwrap();
            bytes
        }
        ReadMode::Owned => view.read_metadata_at(bias + start, length).unwrap(),
    };
    assert_eq!(
        actual,
        DATA[convert::slice_range(start, length.to_u64()).unwrap()]
    );
    assert_eq!(
        *source.calls.borrow(),
        expected
            .iter()
            .map(|&(start, length)| (
                match mode {
                    ReadMode::Buffer => ReadKind::Bytes,
                    ReadMode::Metadata | ReadMode::Owned => ReadKind::Metadata,
                },
                range(bias + start, length)
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        view.retained_bytes(),
        match backing {
            Backing::Borrowed => 0,
            Backing::Owned => 4,
            Backing::Shared => DATA.len(),
        }
    );
}

#[rstest]
#[case::eof(11, 2, FormatError::UnexpectedSourceEof { expected: 13, available: 12 })]
#[case::empty_past_eof(13, 0, FormatError::UnexpectedSourceEof { expected: 13, available: 12 })]
#[case::overflow(u64::MAX, 1, FormatError::OffsetOverflow { offset: u64::MAX, length: 1 })]
fn invalid_requests_fail_before_reading_source_bytes(
    #[case] start: u64,
    #[case] length: usize,
    #[case] expected: FormatError,
    #[values(ReadMode::Metadata, ReadMode::Buffer, ReadMode::Owned)] mode: ReadMode,
) {
    let source = ObservedSource::new(Backing::Borrowed, 0);
    let view = CurrentMetadata::new(&source, range(4, 4)).unwrap();
    source.calls.borrow_mut().clear();
    let error = match mode {
        ReadMode::Metadata => view.read_metadata(start, length).unwrap_err(),
        ReadMode::Buffer => view.read_at(start, &mut vec![0; length]).unwrap_err(),
        ReadMode::Owned => view.read_metadata_at(start, length).unwrap_err(),
    };
    assert_eq!(error, expected);
    assert!(source.calls.borrow().is_empty());
}

#[test]
fn acquiring_metadata_checks_the_complete_source_extent_before_narrowing() {
    let source = ObservedSource::new(Backing::Owned, 0);
    let Err(CurrentMetadataError::Read(error)) = CurrentMetadata::new(&source, range(0, u64::MAX))
    else {
        panic!("expected complete-allocation EOF")
    };
    assert_eq!(
        error,
        FormatError::UnexpectedSourceEof {
            expected: u64::MAX,
            available: DATA.len().to_u64()
        }
    );
    assert!(source.calls.borrow().is_empty());
}

#[test]
#[cfg(target_pointer_width = "32")]
fn acquiring_metadata_narrows_only_the_materialized_length() {
    let source = ObservedSource::new(Backing::Owned, u64::MAX - DATA.len().to_u64());
    let length = u64::from(u32::MAX) + 1;
    let Err(CurrentMetadataError::Read(error)) = CurrentMetadata::new(&source, range(0, length))
    else {
        panic!("expected local-length narrowing failure")
    };
    assert_eq!(
        error,
        FormatError::ValueTooLargeForPlatform {
            value: length,
            target: "usize"
        }
    );
    assert!(source.calls.borrow().is_empty());
}

#[rstest]
#[case::prefix(2, 2)]
#[case::suffix(8, 2)]
fn source_errors_from_uncovered_ranges_remain_exact(
    #[case] start: u64,
    #[case] length: u64,
    #[values(ReadMode::Metadata, ReadMode::Buffer, ReadMode::Owned)] mode: ReadMode,
) {
    let source = ObservedSource::new(Backing::Owned, 0);
    let view = CurrentMetadata::new(&source, range(4, 4)).unwrap();
    source.calls.borrow_mut().clear();
    source.failure.set(Some(range(start, length)));
    let error = match mode {
        ReadMode::Metadata => view.read_metadata(2, 8).unwrap_err(),
        ReadMode::Buffer => view.read_at(2, &mut [0; 8]).unwrap_err(),
        ReadMode::Owned => view.read_metadata_at(2, 8).unwrap_err(),
    };
    assert_eq!(error, FormatError::Source("injected read".into()));
    let kind = match mode {
        ReadMode::Buffer => ReadKind::Bytes,
        ReadMode::Metadata | ReadMode::Owned => ReadKind::Metadata,
    };
    assert_eq!(
        source.calls.borrow().last(),
        Some(&(kind, range(start, length)))
    );
}

#[test]
fn acquisition_preserves_source_errors_and_rejects_mislocated_metadata() {
    let source = ObservedSource::new(Backing::Owned, 0);
    source.failure.set(Some(range(4, 4)));
    let Err(CurrentMetadataError::Read(error)) = CurrentMetadata::new(&source, range(4, 4)) else {
        panic!("expected acquisition read failure")
    };
    assert_eq!(error, FormatError::Source("injected read".into()));
    let source = MislocatedSource;
    let Err(CurrentMetadataError::Range { expected, provided }) =
        CurrentMetadata::new(&source, range(4, 4))
    else {
        panic!("expected physical-range mismatch")
    };
    assert_eq!(expected, range(4, 4));
    assert_eq!(provided, range(5, 4));
}

#[rstest]
#[case::mislocated(Malformed::Mislocated)]
#[case::short(Malformed::Short)]
#[case::long(Malformed::Long)]
fn delegated_metadata_rejects_wrong_ranges_and_lengths(
    #[case] shape: Malformed,
    #[values(0, 2)] start: u64,
) {
    let source = MalformedSource(shape);
    let view = CurrentMetadata::new(&source, range(4, 4)).unwrap();
    let length = if start == 0 { 2 } else { 8 };
    let expected = match shape {
        Malformed::Mislocated => FormatError::InsufficientRegion {
            required_start: start,
            required_length: 2,
            provided_start: start + 1,
            provided_length: 2,
        },
        Malformed::Short => FormatError::FileRegionLengthMismatch {
            range_length: 2,
            bytes_length: 1,
        },
        Malformed::Long => FormatError::FileRegionLengthMismatch {
            range_length: 2,
            bytes_length: 3,
        },
    };
    assert_eq!(view.read_metadata(start, length).unwrap_err(), expected);
    assert_eq!(view.read_metadata_at(start, length).unwrap_err(), expected);
}

#[rstest]
#[case::borrowed(Backing::Borrowed)]
#[case::owned(Backing::Owned)]
#[case::shared(Backing::Shared)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn contained_views_and_buffered_reads_add_no_payload_allocations(#[case] backing: Backing) {
    let source = ObservedSource::new(backing, 0);
    let view = CurrentMetadata::new(&source, range(4, 4)).unwrap();
    source.calls.borrow_mut().clear();
    let control_name = format!("current-metadata-copy-control-{backing:?}");
    let (owned, control) =
        allocation::measure(&control_name, || view.read_metadata_at(5, 2).unwrap());
    assert_eq!(owned, DATA[5..7]);
    assert!(control.blocks > 0 && control.bytes >= 2, "{control}");
    let name = format!("current-metadata-contained-{backing:?}");
    let (bytes, measured) = allocation::measure(&name, || view.read_metadata(5, 2).unwrap());
    assert_eq!(bytes.as_ref(), &DATA[5..7]);
    assert_eq!(bytes.as_ptr(), view.region().bytes()[1..].as_ptr());
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(bytes.retained_bytes(), view.retained_bytes());
    assert!(source.calls.borrow().is_empty());
    let mut buffer = [0; 8];
    let name = format!("current-metadata-buffer-{backing:?}");
    let ((), measured) = allocation::measure(&name, || view.read_at(2, &mut buffer).unwrap());
    assert_eq!(buffer, DATA[2..10]);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(
        *source.calls.borrow(),
        [
            (ReadKind::Bytes, range(2, 2)),
            (ReadKind::Bytes, range(8, 2))
        ]
    );
    source.calls.borrow_mut().clear();
    let name = format!("current-metadata-joined-{backing:?}");
    let (bytes, measured) = allocation::measure(&name, || view.read_metadata(2, 8).unwrap());
    assert_eq!(bytes.as_ref(), &DATA[2..10]);
    assert_eq!(bytes.retained_bytes(), 8);
    assert!(measured.bytes >= 8 && measured.bytes <= 256, "{measured}");
    assert_eq!(
        *source.calls.borrow(),
        [
            (ReadKind::Metadata, range(2, 2)),
            (ReadKind::Metadata, range(8, 2))
        ]
    );
    source.calls.borrow_mut().clear();
    let name = format!("current-metadata-empty-and-eof-{backing:?}");
    let ((empty, error), measured) = allocation::measure(&name, || {
        (
            view.read_metadata(12, 0).unwrap(),
            view.read_metadata(11, 2).unwrap_err(),
        )
    });
    assert_eq!(empty.range(), range(12, 0));
    assert!(empty.as_ref().is_empty());
    assert_eq!(
        error,
        FormatError::UnexpectedSourceEof {
            expected: 13,
            available: 12
        }
    );
    assert_eq!(measured.blocks, 0, "{measured}");
    assert!(source.calls.borrow().is_empty());
}

#[derive(Clone, Copy, Debug)]
enum Backing {
    Borrowed,
    Owned,
    Shared,
}

#[derive(Clone, Copy)]
enum ReadMode {
    Buffer,
    Metadata,
    Owned,
}

struct ObservedSource {
    backing: MetadataBacking,
    bias: u64,
    calls: RefCell<Vec<(ReadKind, FileRange)>>,
    failure: Cell<Option<FileRange>>,
}

impl ObservedSource {
    fn new(backing: Backing, bias: u64) -> Self {
        Self {
            backing: match backing {
                Backing::Borrowed => MetadataBacking::Borrowed,
                Backing::Owned => MetadataBacking::Owned,
                Backing::Shared => MetadataBacking::Shared(
                    MetadataBytes::owned(range(bias, DATA.len().to_u64()), DATA.to_vec()).unwrap(),
                ),
            },
            bias,
            calls: RefCell::new(Vec::with_capacity(64)),
            failure: Cell::new(None),
        }
    }

    fn read(
        &self,
        offset: u64,
        length: usize,
        kind: ReadKind,
    ) -> Result<(FileRange, &[u8]), FormatError> {
        let Self {
            backing: _,
            bias,
            calls,
            failure,
        } = self;
        let request = range(offset, length.to_u64());
        calls.borrow_mut().push((kind, request));
        if failure
            .get()
            .is_some_and(|failure| failure.overlaps(request))
        {
            return Err(FormatError::Source("injected read".into()));
        }
        let start = offset
            .checked_sub(*bias)
            .ok_or_else(|| FormatError::Source("unmapped request".into()))?;
        let local = convert::slice_range(start, length.to_u64())?;
        let bytes = DATA
            .get(local)
            .ok_or_else(|| FormatError::Source("unmapped request".into()))?;
        Ok((request, bytes))
    }
}

impl Source for ObservedSource {
    fn len(&self) -> u64 {
        self.bias + DATA.len().to_u64()
    }
    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        buffer.copy_from_slice(self.read(offset, buffer.len(), ReadKind::Bytes)?.1);
        Ok(())
    }
    fn read_metadata(&self, offset: u64, length: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let (range, bytes) = self.read(offset, length, ReadKind::Metadata)?;
        match &self.backing {
            MetadataBacking::Borrowed => MetadataBytes::borrowed(range, bytes),
            MetadataBacking::Owned => MetadataBytes::owned(range, bytes.to_vec()),
            MetadataBacking::Shared(owner) => owner.subregion(range),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReadKind {
    Bytes,
    Metadata,
}

enum MetadataBacking {
    Borrowed,
    Owned,
    Shared(MetadataBytes<'static>),
}

#[derive(Clone, Copy)]
enum Malformed {
    Long,
    Mislocated,
    Short,
}

struct MalformedSource(Malformed);

impl Source for MalformedSource {
    fn len(&self) -> u64 {
        DATA.len().to_u64()
    }
    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        buffer.copy_from_slice(&DATA[convert::slice_range(offset, buffer.len().to_u64())?]);
        Ok(())
    }
    fn read_metadata(&self, offset: u64, length: usize) -> Result<MetadataBytes<'_>, FormatError> {
        if offset == 4 && length == 4 {
            return MetadataBytes::borrowed(range(4, 4), &DATA[4..8]);
        }
        let (start, length) = match self.0 {
            Malformed::Mislocated => (offset + 1, length),
            Malformed::Short => (offset, length - 1),
            Malformed::Long => (offset, length + 1),
        };
        MetadataBytes::borrowed(range(start, length.to_u64()), &DATA[..length])
    }
}

struct MislocatedSource;

impl Source for MislocatedSource {
    fn len(&self) -> u64 {
        DATA.len().to_u64()
    }
    fn read_at(&self, _offset: u64, _buffer: &mut [u8]) -> Result<(), FormatError> {
        Err(FormatError::Source("unexpected byte read".into()))
    }
    fn read_metadata(&self, offset: u64, length: usize) -> Result<MetadataBytes<'_>, FormatError> {
        MetadataBytes::borrowed(range(offset + 1, length.to_u64()), &DATA[..length])
    }
}

fn range(start: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(start), length).unwrap()
}

const DATA: &[u8; 12] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
