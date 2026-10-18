use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::Source;
use rstest::rstest;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::reader::ReadBackend;

fn overflowing_root_address_file() -> Vec<u8> {
    let mut image = vec![0; 512];
    image.extend_from_slice(
        &Superblock::new(Widths::EIGHT)
            .base_address(512)
            .eof_address(1024)
            .root_header_address(u64::MAX)
            .build(),
    );
    image.resize(1024, 0);
    image
}

fn assert_root_address_overflow(err: Error) {
    match err {
        Error::Format(FormatError::OffsetOverflow { offset, length }) => {
            assert_eq!(offset, u64::MAX);
            assert_eq!(length, 512);
        }
        other => panic!("expected root-group address overflow, got {other:?}"),
    }
}

#[rstest]
#[case(ReadBackend::Buffered)]
#[case(ReadBackend::Streaming)]
#[case(ReadBackend::Bytes)]
#[case(ReadBackend::Source)]
fn root_address_overflow_is_reported_when_navigation_resolves_it(#[case] backend: ReadBackend) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overflowing-base-address.h5");
    std::fs::write(&path, overflowing_root_address_file()).unwrap();
    let file = backend.open(&path).unwrap();
    assert_eq!(file.superblock().base_address.get(), 512);
    assert_eq!(file.superblock().root_group_address.get(), u64::MAX);
    assert_root_address_overflow(file.group("/").unwrap_err());
    assert_root_address_overflow(file.root().groups().unwrap_err());
    assert_root_address_overflow(file.dataset("child").unwrap_err());
}

#[rstest]
#[case::overflow(
    0, u64::MAX, 512,
    FormatError::OffsetOverflow { offset: u64::MAX, length: 512 }
)]
#[case::underflow(
    512, 511, 0,
    FormatError::EofRelocationUnderflow { stored_eof: 511, shift: 512 }
)]
fn opening_reports_eof_relocation_arithmetic_failure(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Streaming,
        ReadBackend::Bytes,
        ReadBackend::Source
    )]
    backend: ReadBackend,
    #[case] stored_base: u64,
    #[case] stored_eof: u64,
    #[case] signature_offset: usize,
    #[case] expected: FormatError,
) {
    let mut image = vec![0; signature_offset];
    image.extend_from_slice(
        &Superblock::new(Widths::EIGHT)
            .base_address(stored_base)
            .eof_address(stored_eof)
            .build(),
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid-eof-relocation.h5");
    std::fs::write(&path, image).unwrap();
    let error = backend.open(&path).unwrap_err();
    let Error::Format(format_error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(format_error, expected);
}

#[rstest]
#[case::short((1_u64 << 40) - 1)]
#[case::complete(1_u64 << 40)]
#[case::trailing((1_u64 << 40) + 1)]
fn source_completeness_preserves_lengths_above_u32(#[case] actual_len: u64) {
    let expected_eof = 1_u64 << 40;
    let source = SparseImage {
        prefix: Superblock::new(Widths::EIGHT)
            .eof_address(expected_eof)
            .build(),
        len: actual_len,
    };
    let result = File::from_source(source);
    if actual_len < expected_eof {
        let error = result.unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::TruncatedFile {
                expected_eof,
                actual_len
            }
        );
    } else {
        let file = result.unwrap();
        assert_eq!(file.file_size(), actual_len);
        assert_eq!(file.superblock().eof_address, expected_eof);
    }
}

struct SparseImage {
    prefix: Vec<u8>,
    len: u64,
}

impl Source for SparseImage {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        let Self { prefix, len } = self;
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: buf.len() as u64,
            })?;
        if end > *len {
            return Err(FormatError::UnexpectedEof {
                expected: buf.len(),
                available: usize::try_from(len.saturating_sub(offset)).unwrap(),
            });
        }
        for (index, byte) in buf.iter_mut().enumerate() {
            let position = offset
                .checked_add(index as u64)
                .ok_or(FormatError::OffsetOverflow {
                    offset,
                    length: index as u64,
                })?;
            *byte = usize::try_from(position)
                .ok()
                .and_then(|position| prefix.get(position))
                .copied()
                .unwrap_or(0);
        }
        Ok(())
    }
}
