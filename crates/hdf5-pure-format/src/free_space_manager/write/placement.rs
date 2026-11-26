use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::{FormatWidths, OffsetWidth};

use super::prepared::{self, PreparedFreeSpaceManager};

use super::encode::FreeSpaceManagerEncoding;

impl PreparedFreeSpaceManager {
    /// Prepares each block to store the other block's address.
    ///
    /// Pass the two addresses relative to the file's base address. A nonempty manager requires
    /// two defined addresses. For an empty manager, either address may be the undefined address
    /// at the file's offset width or [`u64::MAX`]. The writer encodes these as the undefined
    /// address at the file's offset width.
    ///
    /// Pass `None` for the shortest section list, or `Some` with the total list length in bytes,
    /// including its checksum. The writer pads the list with zero bytes before the checksum.
    /// The header stores the total length in both the used and allocated length fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{
    ///     FormatWidths, FreeSection, FreeSpaceManagerHeader, PreparedFreeSpaceManager,
    ///     SECTION_CLASS_SIMPLE, parse_section_info,
    /// };
    ///
    /// let widths = FormatWidths::from_sizes(8, 4)?;
    /// let sections = [FreeSection { addr: StoredAddress::new(256), size: 32 }];
    /// let plan = PreparedFreeSpaceManager::new(widths, sections, SECTION_CLASS_SIMPLE)?;
    /// let header_address = StoredAddress::new(512);
    /// let list_address = StoredAddress::new(1024);
    /// let allocation = plan.section_list_len() as u64 + 16;
    /// let manager = plan.place(header_address, list_address, Some(allocation))?;
    ///
    /// let header = manager.header();
    /// let mut header_bytes = vec![0; header.encoded_size()];
    /// assert_eq!(header.write_into(&mut header_bytes)?, header_bytes.len());
    ///
    /// let list = manager.section_list();
    /// let mut list_bytes = vec![0; list.encoded_size()];
    /// assert_eq!(list.write_into(&mut list_bytes)?, list_bytes.len());
    /// assert_eq!(list_bytes.len() as u64, allocation);
    ///
    /// let header = FreeSpaceManagerHeader::parse(widths, &header_bytes)?;
    /// assert_eq!(header.section_list_addr(), list_address);
    /// assert_eq!(header.section_list_used(), allocation);
    /// assert_eq!(parse_section_info(widths, &list_bytes, header_address, &header)?, sections);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if a block address exceeds the offset width, or
    /// [`FormatError::DefinedAddressIsUndefined`] if a block address equals the undefined address
    /// for a nonempty manager.
    ///
    /// Returns [`FormatError::FreeSpaceSectionListAllocationTooSmall`] if `allocation`
    /// is less than [`Self::section_list_len`], [`FormatError::LengthTooLarge`] if the length
    /// exceeds the file's length width, or [`FormatError::ValueTooLargeForPlatform`] if the
    /// section list is too long for a slice on this platform.
    pub fn place(
        &self,
        header_address: StoredAddress,
        section_list_address: StoredAddress,
        allocation: Option<u64>,
    ) -> Result<FreeSpaceManagerEncoding<'_>, FormatError> {
        let FormatWidths { offsets, lengths } = self.widths();
        let header_address = self.block_address(header_address, offsets)?;
        let section_list_address = self.block_address(section_list_address, offsets)?;
        let required = self.section_list_len().to_u64();
        let allocated = allocation.unwrap_or(required);
        if allocated < required {
            return Err(FormatError::FreeSpaceSectionListAllocationTooSmall {
                required,
                allocated,
            });
        }
        let allocation_len = prepared::slice_len(allocated, lengths)?;
        Ok(FreeSpaceManagerEncoding::new(Placement {
            plan: self,
            header_address,
            section_list_address,
            allocation_len,
        }))
    }

    fn block_address(
        &self,
        address: StoredAddress,
        offsets: OffsetWidth,
    ) -> Result<StoredAddress, FormatError> {
        if self.section_count() == 0 && (address.is_undefined(offsets) || address.get() == u64::MAX)
        {
            return Ok(StoredAddress::undefined(offsets));
        }
        bytes::require_defined_offset(address.get(), offsets)?;
        Ok(address)
    }
}

/// The block addresses and section-list allocation checked for one prepared manager.
pub(super) struct Placement<'a> {
    plan: &'a PreparedFreeSpaceManager,
    header_address: StoredAddress,
    section_list_address: StoredAddress,
    allocation_len: usize,
}

impl<'a> Placement<'a> {
    pub(super) fn plan(&self) -> &'a PreparedFreeSpaceManager {
        self.plan
    }

    pub(super) fn header_address(&self) -> StoredAddress {
        self.header_address
    }

    pub(super) fn section_list_address(&self) -> StoredAddress {
        self.section_list_address
    }

    pub(super) fn allocation_len(&self) -> usize {
        self.allocation_len
    }
}
