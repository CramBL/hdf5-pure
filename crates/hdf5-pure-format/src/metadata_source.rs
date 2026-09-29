//! The reads the metadata parsers make of a file.

use alloc::vec::Vec;

use crate::convert::Narrow;
use crate::error::FormatError;

#[expect(
    clippy::len_without_is_empty,
    reason = "`len` is the byte length of the file the metadata is read from, as \
              `std::fs::Metadata::len` is, and no parse asks whether that file is empty"
)]
/// Supplies the bytes of a file to the parsers that read its metadata.
///
/// A parser that takes a `MetadataSource`, such as
/// [`ObjectHeader::parse_from_source`](crate::object_header::ObjectHeader::parse_from_source),
/// reads a structure one piece at a time and holds no more of the file than the piece it parses.
/// Every offset is an absolute position in the file. `[u8]` implements the trait for a file held
/// in memory.
pub trait MetadataSource {
    /// Returns the length of the file in bytes.
    fn len(&self) -> u64;

    /// Reads exactly `buf.len()` bytes starting at `offset` into `buf`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the file ends before `offset + buf.len()`. An
    /// implementation returns an error of its own if a read fails for another reason.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError>;

    /// Reads `len` bytes starting at `offset` into a new vector.
    ///
    /// An implementation may serve the read from a cache of earlier metadata reads.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the file ends before `offset + len`. An
    /// implementation returns an error of its own if a read fails for another reason.
    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError>;
}

impl MetadataSource for [u8] {
    fn len(&self) -> u64 {
        Narrow::to_u64(<[u8]>::len(self))
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        buf.copy_from_slice(region(self, offset, buf.len())?);
        Ok(())
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        Ok(region(self, offset, len)?.to_vec())
    }
}

fn region(bytes: &[u8], offset: u64, len: usize) -> Result<&[u8], FormatError> {
    let start = offset.to_usize()?;
    start
        .checked_add(len)
        .and_then(|end| bytes.get(start..end))
        .ok_or(FormatError::UnexpectedEof {
            expected: start.saturating_add(len),
            available: bytes.len(),
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn a_read_inside_the_bytes_returns_them() {
        let bytes = [1u8, 2, 3, 4, 5];
        let mut buf = [0u8; 2];
        bytes.read_at(1, &mut buf).unwrap();

        assert_eq!(MetadataSource::len(bytes.as_slice()), 5);
        assert_eq!(buf, [2, 3]);
        assert_eq!(bytes.read_metadata_at(3, 2).unwrap(), vec![4, 5]);
    }

    #[rstest]
    #[case::past_the_end(4, 2, 6)]
    #[case::overflowing(u64::from(u32::MAX), usize::MAX, usize::MAX)]
    fn a_read_past_the_bytes_returns_unexpected_eof(
        #[case] offset: u64,
        #[case] len: usize,
        #[case] expected: usize,
    ) {
        let bytes = [0u8; 5];

        assert_eq!(
            bytes.read_metadata_at(offset, len).unwrap_err(),
            FormatError::UnexpectedEof {
                expected,
                available: 5,
            }
        );
    }
}
