use crate::address::BaseAddress;
use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The values encoded in a file's superblock.
///
/// The [base address](Self::base_address) and [end-of-file address](Self::eof_address) are
/// absolute file addresses. The other address fields hold base-relative [`StoredAddress`]
/// values. The reader interprets those values in the opened image's address context when
/// locating metadata in the physical source.
/// For a relocated image, the reader derives the effective base and end-of-file position by
/// applying the displacement from the encoded base to the physical signature position.
///
/// A field absent from this superblock version is `None`. An optional address field present
/// in the layout is `Some`, including when it contains the undefined address.
///
/// The superblock is defined in "Format Signature and Superblock" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Superblock {
    /// The encoding version of the superblock.
    pub version: SuperblockVersion,
    /// The width of file addresses.
    pub offset_size: OffsetWidth,
    /// The width of object sizes and other lengths.
    pub length_size: LengthWidth,
    /// Absolute file address of the first byte of HDF5 data, as encoded in the superblock.
    ///
    /// The reader uses the physical signature position as the effective base when opening a
    /// relocated image. This field retains the encoded value.
    pub base_address: BaseAddress,
    /// Absolute file address of the first byte past the HDF5 data, as encoded in the superblock.
    ///
    /// The reader applies the base relocation to derive its effective physical position.
    pub eof_address: u64,
    /// Root group object-header address relative to the HDF5 base address.
    ///
    /// Versions 0 and 1 store this address in the root symbol-table entry. Versions 2 and 3
    /// store it directly in the superblock. [`StoredAddress::get`] returns the encoded value.
    pub root_group_address: StoredAddress,
    /// Group leaf node K (v0/v1 only).
    pub group_leaf_node_k: Option<u16>,
    /// Group internal node K (v0/v1 only).
    pub group_internal_node_k: Option<u16>,
    /// Indexed storage internal node K (v1 only).
    pub indexed_storage_internal_node_k: Option<u16>,
    /// Free-space address relative to the HDF5 base address, in versions 0 and 1.
    pub free_space_address: Option<StoredAddress>,
    /// Driver information block address relative to the HDF5 base address, in versions 0 and 1.
    pub driver_info_address: Option<StoredAddress>,
    /// File consistency flags.
    pub consistency_flags: u32,
    /// Superblock extension address relative to the HDF5 base address, in versions 2 and 3.
    pub superblock_extension_address: Option<StoredAddress>,
    /// Jenkins lookup3 checksum (v2/v3 only).
    pub checksum: Option<u32>,
}

/// The encoding version of a [`Superblock`].
///
/// The superblock stores this value in its "Version Number of the Superblock" field, defined in
/// "Format Signature and Superblock" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuperblockVersion {
    /// Version 0, with a root group symbol-table entry.
    Zero,
    /// Version 1, with an indexed storage internal node K field.
    One,
    /// Version 2, with a checksum and a superblock extension address.
    Two,
    /// Version 3, with file locking flags.
    Three,
}

impl SuperblockVersion {
    /// Returns the version byte stored in the superblock.
    pub const fn get(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

impl TryFrom<u8> for SuperblockVersion {
    type Error = FormatError;

    /// Parses the superblock's version byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedVersion`] if `version` is not 0, 1, 2, or 3.
    fn try_from(version: u8) -> Result<Self, FormatError> {
        Ok(match version {
            0 => Self::Zero,
            1 => Self::One,
            2 => Self::Two,
            3 => Self::Three,
            version => return Err(FormatError::UnsupportedVersion(version)),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0, SuperblockVersion::Zero)]
    #[case(1, SuperblockVersion::One)]
    #[case(2, SuperblockVersion::Two)]
    #[case(3, SuperblockVersion::Three)]
    fn a_version_preserves_its_byte(#[case] byte: u8, #[case] version: SuperblockVersion) {
        assert_eq!(SuperblockVersion::try_from(byte), Ok(version));
        assert_eq!(version.get(), byte);
    }

    #[rstest]
    #[case(4)]
    #[case(16)]
    #[case(255)]
    fn an_unknown_version_is_rejected(#[case] byte: u8) {
        assert_eq!(
            SuperblockVersion::try_from(byte),
            Err(FormatError::UnsupportedVersion(byte))
        );
    }
}
