//! Address interpretation for an opened HDF5 image.

pub(crate) use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::PhysicalAddress;
pub(crate) use hdf5_pure_core::__private::StoredAddressExt;
pub use hdf5_pure_core::BaseAddress;
pub use hdf5_pure_core::StoredAddress;

use crate::error::FormatError;

/// The effective base and end-of-file position of an opened HDF5 image.
///
/// The reader derives both positions by applying the displacement from the encoded base to the
/// physical superblock signature position. The parsed superblock stores the encoded values.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OpenedImageContext {
    addresses: AddressContext,
    /// The physical position of the encoded end-of-file address after relocation.
    effective_eof: u64,
}

impl OpenedImageContext {
    /// Derives the effective base and end-of-file position from the physical signature position.
    ///
    /// The reader applies the same displacement to both encoded positions, as defined in
    /// "Format Signature and Superblock" of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if forward relocation exceeds [`u64::MAX`], or
    /// [`FormatError::EofRelocationUnderflow`] if backward relocation places EOF below zero.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
    pub(crate) fn reconcile(
        stored_base: BaseAddress,
        stored_eof: u64,
        signature_offset: u64,
    ) -> Result<Self, FormatError> {
        let shift = signature_offset.abs_diff(stored_base.get());
        let effective_eof = if signature_offset >= stored_base.get() {
            stored_eof
                .checked_add(shift)
                .ok_or(FormatError::OffsetOverflow {
                    offset: stored_eof,
                    length: shift,
                })?
        } else {
            stored_eof
                .checked_sub(shift)
                .ok_or(FormatError::EofRelocationUnderflow { stored_eof, shift })?
        };
        Ok(Self {
            addresses: AddressContext::from_effective_base(BaseAddress::new(signature_offset)),
            effective_eof,
        })
    }

    /// Requires the physical source to extend through the relocated end-of-file position.
    ///
    /// Bytes after that position are permitted. `H5F__super_read` applies the same EOF check at
    /// ordinary file open (`H5Fsuper.c`, HDF5 2.2.0).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TruncatedFile`] if `actual_len` is less than the effective EOF.
    pub(crate) fn require_complete(self, actual_len: u64) -> Result<(), FormatError> {
        if actual_len < self.effective_eof {
            return Err(FormatError::TruncatedFile {
                expected_eof: self.effective_eof,
                actual_len,
            });
        }
        Ok(())
    }

    /// Returns the stored-address mapping for the reconciled physical base.
    pub(crate) const fn addresses(self) -> AddressContext {
        self.addresses
    }

    /// Returns the recorded end of allocation relative to the effective base.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if the effective EOF is below the effective base.
    pub(crate) fn end_of_allocation(self) -> Result<StoredAddress, FormatError> {
        let Self {
            addresses,
            effective_eof,
        } = self;
        addresses.relative(effective_eof)
    }
}

/// The mapping between stored addresses and physical positions in an opened image.
///
/// The reader adds the effective base to a [`StoredAddress`] to locate metadata in the source.
/// The editor subtracts it when encoding a physical position in metadata.
/// For a relocated image, the effective base differs from the encoded superblock base.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AddressContext {
    /// The physical position corresponding to stored address zero.
    effective_base: BaseAddress,
}

impl AddressContext {
    /// Creates an address context with the supplied effective base.
    pub(crate) const fn from_effective_base(effective_base: BaseAddress) -> Self {
        Self { effective_base }
    }

    /// Returns the physical position of a stored address in this image.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum of the stored address and effective base
    /// exceeds [`u64::MAX`].
    pub(crate) fn absolute(self, stored: StoredAddress) -> Result<u64, FormatError> {
        Ok(self.resolve(stored)?.get())
    }

    /// Resolves a stored address using the opened image's effective base.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
    pub(crate) fn resolve(self, stored: StoredAddress) -> Result<PhysicalAddress, FormatError> {
        self.effective_base
            .absolute(stored)
            .map(PhysicalAddress::new)
    }

    /// Returns the stored address corresponding to a physical position in this image.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if the physical position is below the effective
    /// base.
    pub(crate) fn relative(self, absolute: u64) -> Result<StoredAddress, FormatError> {
        self.effective_base.relative(absolute)
    }

    /// Returns the base used to interpret metadata addresses in this image.
    pub(crate) const fn effective_base(self) -> BaseAddress {
        self.effective_base
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::AddressContext;
    use super::BaseAddress;
    use super::BaseAddressExt;
    use super::FormatError;
    use super::OpenedImageContext;
    use super::StoredAddress;
    use super::StoredAddressExt;

    #[rstest]
    #[case::plain(0, 0, 96, 96)]
    #[case::userblock(512, 512, 96, 608)]
    #[case::forward_relocation(512, 1024, 96, 1120)]
    #[case::backward_relocation(512, 0, 96, 96)]
    fn resolution_uses_the_effective_base_once(
        #[case] encoded_base: u64,
        #[case] signature: u64,
        #[case] stored: u64,
        #[case] expected: u64,
    ) {
        let context =
            OpenedImageContext::reconcile(BaseAddress::new(encoded_base), 4096, signature)
                .unwrap()
                .addresses();
        let physical = context.resolve(StoredAddress::new(stored)).unwrap();
        assert_eq!(physical.get(), expected);
        assert_eq!(context.relative(physical.get()).unwrap().get(), stored);
    }

    #[test]
    fn resolution_rejects_physical_address_overflow() {
        let context = AddressContext::from_effective_base(BaseAddress::new(512));
        assert_eq!(
            context.resolve(StoredAddress::new(u64::MAX)).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 512
            }
        );
    }

    #[rstest]
    #[case::short(4_096, 4_095)]
    #[case::equal(4_096, 4_096)]
    #[case::trailing(4_096, 4_608)]
    #[case::large_short(u64::MAX, u64::MAX - 1)]
    #[case::large_equal(u64::MAX, u64::MAX)]
    fn completeness_uses_physical_u64_length(#[case] expected_eof: u64, #[case] actual_len: u64) {
        let opened = OpenedImageContext::reconcile(BaseAddress::ZERO, expected_eof, 0).unwrap();
        if actual_len < expected_eof {
            assert_eq!(
                opened.require_complete(actual_len).unwrap_err(),
                FormatError::TruncatedFile {
                    expected_eof,
                    actual_len
                }
            );
        } else {
            opened.require_complete(actual_len).unwrap();
        }
    }

    #[rstest]
    #[case::plain(0, 0, 4096, 4096)]
    #[case::userblock(512, 512, 4096, 4096)]
    #[case::jammed(0, 512, 4096, 4608)]
    #[case::unjammed(512, 0, 4096, 3584)]
    #[case::large_shift(0, 1 << 63, 0, 1 << 63)]
    #[case::large_backward_shift(1 << 63, 0, u64::MAX, u64::MAX / 2)]
    fn reconciliation_moves_base_and_eof_together(
        #[case] stored_base: u64,
        #[case] signature_offset: u64,
        #[case] stored_eof: u64,
        #[case] effective_eof: u64,
    ) {
        let opened = OpenedImageContext::reconcile(
            BaseAddress::new(stored_base),
            stored_eof,
            signature_offset,
        )
        .unwrap();
        assert_eq!(opened.addresses().effective_base().get(), signature_offset);
        assert_eq!(opened.effective_eof, effective_eof);
        assert_eq!(
            opened.addresses().absolute(StoredAddress::new(16)).unwrap(),
            signature_offset + 16
        );
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
    fn reconciliation_rejects_eof_arithmetic_outside_u64(
        #[case] stored_base: u64,
        #[case] stored_eof: u64,
        #[case] signature_offset: u64,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            OpenedImageContext::reconcile(
                BaseAddress::new(stored_base),
                stored_eof,
                signature_offset,
            )
            .unwrap_err(),
            expected
        );
    }
}
