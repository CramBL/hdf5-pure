//! The reads the metadata parsers make of a file.

use alloc::vec::Vec;
use core::ops::Deref;

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
/// Offsets use the source's coordinate frame. `[u8]` implements the trait for a file held in
/// memory, beginning at offset zero.
pub trait MetadataSource {
    /// The byte view held by a parser for one metadata read.
    type Bytes<'a>: Deref<Target = [u8]>
    where
        Self: 'a;

    /// Reads exactly `len` bytes at `offset` for a borrowed parser view.
    ///
    /// An implementation may borrow resident bytes or share an immutable allocation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the file ends before `offset + len`,
    /// [`FormatError::OffsetOverflow`] if the range end exceeds [`u64::MAX`], and an error from
    /// the source if the read fails.
    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError>;

    /// Returns the length of the file in bytes.
    fn len(&self) -> u64;

    /// Reads exactly `buf.len()` bytes starting at `offset` into `buf`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the file ends before `offset + buf.len()`. An
    /// implementation returns [`FormatError::OffsetOverflow`] if `offset + buf.len()` exceeds
    /// [`u64::MAX`], and an error of its own if a read fails for another reason.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError>;

    /// Reads `len` bytes starting at `offset` into a new vector.
    ///
    /// An implementation may serve the read from a cache of earlier metadata reads.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the file ends before `offset + len`. An
    /// implementation returns [`FormatError::OffsetOverflow`] if `offset + len` exceeds
    /// [`u64::MAX`], and an error of its own if a read fails for another reason.
    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError>;
}

impl MetadataSource for [u8] {
    type Bytes<'a> = &'a [u8];

    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
        region(self, offset, len)
    }

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

pub(crate) fn region(bytes: &[u8], offset: u64, len: usize) -> Result<&[u8], FormatError> {
    let end = offset
        .checked_add(len.to_u64())
        .ok_or(FormatError::OffsetOverflow {
            offset,
            length: len.to_u64(),
        })?;
    if end > bytes.len().to_u64() {
        return Err(FormatError::UnexpectedSourceEof {
            expected: end,
            available: bytes.len().to_u64(),
        });
    }
    bytes
        .get(offset.to_usize()?..end.to_usize()?)
        .ok_or(FormatError::UnexpectedSourceEof {
            expected: end,
            available: bytes.len().to_u64(),
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
    #[case::large_offset(4_294_967_536, 64, 4_294_967_600)]
    fn a_read_past_the_bytes_returns_unexpected_eof(
        #[case] offset: u64,
        #[case] len: usize,
        #[case] expected: u64,
    ) {
        let bytes = [0u8; 5];

        assert_eq!(
            bytes.read_metadata_at(offset, len).unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected,
                available: 5,
            }
        );
    }
    #[test]
    fn a_wrapping_source_range_reports_arithmetic_overflow() {
        assert_eq!(
            [0u8; 5].read_metadata_at(u64::MAX, 1).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        );
    }
}
