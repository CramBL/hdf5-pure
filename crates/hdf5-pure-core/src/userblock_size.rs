use crate::FormatError;

/// The size in bytes of the file prefix before the HDF5 signature.
///
/// The size is zero, or a power of two of at least 512 bytes. The signature offsets for
/// superblock versions 0 through 3 are defined in "Format Signature and Superblock" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UserblockSize(u64);

impl UserblockSize {
    /// Returns the reserved prefix length in bytes.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// A file without a userblock, with its HDF5 signature at byte zero.
    pub const NONE: Self = Self(0);
    /// The smallest nonempty userblock, occupying 512 bytes.
    pub const MIN: Self = Self(512);
}

impl TryFrom<u64> for UserblockSize {
    type Error = FormatError;

    /// Parses a userblock byte count.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidUserblockSize`] if `size` is nonzero and either smaller
    /// than 512 or not a power of two.
    fn try_from(size: u64) -> Result<Self, FormatError> {
        if size == Self::NONE.get() || (size >= Self::MIN.get() && size.is_power_of_two()) {
            Ok(Self(size))
        } else {
            Err(FormatError::InvalidUserblockSize(size))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0)]
    #[case(512)]
    #[case(1024)]
    #[case(2048)]
    #[case(1 << 63)]
    fn parses_userblock_size(#[case] size: u64) {
        assert_eq!(UserblockSize::try_from(size).unwrap().get(), size);
    }

    #[rstest]
    #[case(1)]
    #[case(511)]
    #[case(513)]
    #[case(700)]
    #[case(1023)]
    #[case(u64::MAX)]
    fn rejects_invalid_userblock_size(#[case] size: u64) {
        assert_eq!(
            UserblockSize::try_from(size),
            Err(FormatError::InvalidUserblockSize(size))
        );
    }
}
