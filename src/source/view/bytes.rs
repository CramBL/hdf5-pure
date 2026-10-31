//! Stored-coordinate reads from a borrowed file image.

use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::{FileRange, MetadataFrame};
use hdf5_pure_core::{AddressRange, MetadataBytes, StoredAddress};

use crate::address::{BaseAddress, BaseAddressExt};
use crate::convert::{self, Narrow};
use crate::error::FormatError;
use crate::source::{self, RangeReader};

use super::{PhysicalBytes, StoredRead};

/// A borrowed file image addressed relative to the opened image's effective base.
///
/// The view contains the suffix beginning at the effective base and its physical origin.
/// Payload and metadata reads can borrow directly from the original slice.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoredBytes<'a> {
    frame: MetadataFrame<'a>,
}

impl<'a> StoredBytes<'a> {
    /// Borrows the original image's bytes after its effective base.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the base exceeds the image's byte length,
    /// and errors from [`MetadataFrame::new`].
    pub(super) fn new(physical: PhysicalBytes<'a>) -> Result<Self, FormatError> {
        Ok(Self {
            frame: source::framed_metadata(physical.bytes, physical.addresses.effective_base())?,
        })
    }

    /// Borrows the bytes of a checked stored-coordinate interval.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedSourceEof`] if the interval ends beyond the view.
    pub(crate) fn borrow_range(
        self,
        range: AddressRange<StoredAddress>,
    ) -> Result<&'a [u8], FormatError> {
        bytes_in(self.frame.bytes(), range)
    }

    /// Returns the borrowed suffix with its physical origin for byte-oriented parsers.
    pub(crate) const fn metadata_frame(self) -> MetadataFrame<'a> {
        self.frame
    }

    /// Resolves a request to a physical interval within the borrowed image.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the start or end exceeds [`u64::MAX`], and
    /// [`FormatError::UnexpectedSourceEof`] if the end exceeds the image's physical end.
    fn physical_range(
        self,
        address: StoredAddress,
        length: usize,
    ) -> Result<FileRange, FormatError> {
        let start = BaseAddress::new(self.frame.origin().get()).resolve(address)?;
        let range = FileRange::new(start, length.to_u64())?;
        if range.end() > self.frame.range().end() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: range.end().get(),
                available: self.frame.range().end().get(),
            });
        }
        Ok(range)
    }
}

impl StoredRead for StoredBytes<'_> {
    fn len(&self) -> u64 {
        self.frame.bytes().len().to_u64()
    }
    fn read_at(&self, address: StoredAddress, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.physical_range(address, buffer.len())?;
        buffer.copy_from_slice(
            self.borrow_range(AddressRange::new(address, buffer.len().to_u64())?)?,
        );
        Ok(())
    }
    fn read_exact_at(&self, address: StoredAddress, length: usize) -> Result<Vec<u8>, FormatError> {
        let range = AddressRange::new(address, length.to_u64())?;
        if range.end().get() > self.len() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: range.end().get(),
                available: self.len(),
            });
        }
        let mut bytes = vec![0; length];
        self.read_at(address, &mut bytes)?;
        Ok(bytes)
    }
    fn read_metadata(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<MetadataBytes<'_>, FormatError> {
        let physical = self.physical_range(address, length)?;
        MetadataBytes::borrowed(
            physical,
            self.borrow_range(AddressRange::new(address, length.to_u64())?)?,
        )
    }
    fn read_metadata_at(
        &self,
        address: StoredAddress,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        Ok(self.read_metadata(address, length)?.into_vec())
    }
}

impl RangeReader<AddressRange<StoredAddress>> for StoredBytes<'_> {
    fn read_range(
        &self,
        range: AddressRange<StoredAddress>,
    ) -> Result<MetadataBytes<'_>, FormatError> {
        self.read_metadata(range.start(), range.length().to_usize()?)
    }
}

fn bytes_in(bytes: &[u8], range: AddressRange<StoredAddress>) -> Result<&[u8], FormatError> {
    if range.end().get() > bytes.len().to_u64() {
        return Err(FormatError::UnexpectedSourceEof {
            expected: range.end().get(),
            available: bytes.len().to_u64(),
        });
    }
    bytes
        .get(convert::slice_range(range.start().get(), range.length())?)
        .ok_or(FormatError::UnexpectedSourceEof {
            expected: range.end().get(),
            available: bytes.len().to_u64(),
        })
}
