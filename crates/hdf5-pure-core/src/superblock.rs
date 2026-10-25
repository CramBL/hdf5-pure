use crate::address::BaseAddress;
use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The values encoded in a file's superblock.
///
/// The encoding version and file consistency flags are stored together in [`SuperblockStatus`].
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
pub struct Superblock {
    /// The encoding version and file consistency flags.
    pub(crate) status: SuperblockStatus,
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
    /// Superblock extension address relative to the HDF5 base address, in versions 2 and 3.
    pub superblock_extension_address: Option<StoredAddress>,
    /// Jenkins lookup3 checksum (v2/v3 only).
    pub checksum: Option<u32>,
}

impl Superblock {
    /// Changes the encoding version while preserving the file consistency flags.
    ///
    /// Other superblock fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SuperblockStatusFlagsTooWide`] if `version` is 2 or 3 and the
    /// current flags contain any bit above bit 7. The superblock is unchanged on error.
    pub fn set_version(&mut self, version: SuperblockVersion) -> Result<(), FormatError> {
        self.status = SuperblockStatus::new(version, self.consistency_flags())?;
        Ok(())
    }

    /// Replaces the file consistency flags while preserving the superblock version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SuperblockStatusFlagsTooWide`] if the version is 2 or 3 and `flags`
    /// contains any bit above bit 7. The superblock is unchanged on error.
    pub fn set_consistency_flags(
        &mut self,
        flags: SuperblockStatusFlags,
    ) -> Result<(), FormatError> {
        self.status = SuperblockStatus::new(self.version(), flags)?;
        Ok(())
    }

    /// Returns the version and its consistency flags at their encoded width.
    pub const fn status(&self) -> SuperblockStatus {
        self.status
    }

    /// Returns the encoding version stored with the consistency flags.
    pub const fn version(&self) -> SuperblockVersion {
        self.status.version()
    }

    /// Returns every stored flag bit, including reserved bits.
    pub const fn consistency_flags(&self) -> SuperblockStatusFlags {
        self.status.flags()
    }
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

/// The version and file consistency flags of a superblock.
///
/// The superblock stores this field in four bytes in versions 0 and 1 and in one byte in
/// versions 2 and 3. Every stored bit is preserved, including reserved bits and inconsistent
/// access flags. The field is defined in "Format Signature and Superblock" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuperblockStatus {
    /// Version 0, with a four-byte file consistency field.
    Zero(u32),
    /// Version 1, with a four-byte file consistency field.
    One(u32),
    /// Version 2, with a one-byte file consistency field.
    Two(u8),
    /// Version 3, with a one-byte file consistency field.
    Three(u8),
}

impl SuperblockStatus {
    /// Creates status flags at the width of `version`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SuperblockStatusFlagsTooWide`] if `version` is 2 or 3 and `flags`
    /// contains any bit above bit 7.
    pub fn new(
        version: SuperblockVersion,
        flags: SuperblockStatusFlags,
    ) -> Result<Self, FormatError> {
        Ok(match version {
            SuperblockVersion::Zero => Self::Zero(flags.bits()),
            SuperblockVersion::One => Self::One(flags.bits()),
            SuperblockVersion::Two => Self::Two(u8::try_from(flags)?),
            SuperblockVersion::Three => Self::Three(u8::try_from(flags)?),
        })
    }

    /// Returns the version whose field width is used.
    pub const fn version(self) -> SuperblockVersion {
        match self {
            Self::Zero(_) => SuperblockVersion::Zero,
            Self::One(_) => SuperblockVersion::One,
            Self::Two(_) => SuperblockVersion::Two,
            Self::Three(_) => SuperblockVersion::Three,
        }
    }

    /// Returns every flag bit, including reserved bits.
    pub const fn flags(self) -> SuperblockStatusFlags {
        let bits = match self {
            Self::Zero(bits) | Self::One(bits) => bits,
            Self::Two(bits) | Self::Three(bits) => bits as u32,
        };
        SuperblockStatusFlags::from_bits(bits)
    }
}

/// The file consistency flags of a [`Superblock`].
///
/// The reader preserves every stored bit, including reserved bits and an SWMR write mark
/// without a write mark. [`is_inconsistent`](Self::is_inconsistent) identifies that combination.
///
/// The superblock stores this field in four bytes in versions 0 and 1, and in one byte in versions
/// 2 and 3. Version 3 uses the access flags for file locking, as defined in "Format Signature and
/// Superblock" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SuperblockStatusFlags {
    bits: u32,
}

impl SuperblockStatusFlags {
    /// Creates empty status flags.
    pub const fn new() -> Self {
        Self { bits: 0 }
    }

    /// Creates status flags with write access marked.
    pub const fn write_marked() -> Self {
        Self {
            bits: SuperblockStatusFlag::WriteAccess.mask(),
        }
    }

    /// Creates status flags with both write access and SWMR write access marked.
    pub const fn swmr_write_marked() -> Self {
        Self {
            bits: SuperblockStatusFlag::WriteAccess.mask()
                | SuperblockStatusFlag::SwmrWriteAccess.mask(),
        }
    }

    /// Returns `true` if every status bit is clear.
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Returns `true` if `flag` is set.
    pub const fn contains(self, flag: SuperblockStatusFlag) -> bool {
        self.bits & flag.mask() != 0
    }

    /// Returns `true` if SWMR write access is marked without write access.
    pub const fn is_inconsistent(self) -> bool {
        self.contains(SuperblockStatusFlag::SwmrWriteAccess)
            && !self.contains(SuperblockStatusFlag::WriteAccess)
    }

    /// Returns every stored bit, including reserved bits.
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Returns all bits except [`WriteAccess`](SuperblockStatusFlag::WriteAccess) and
    /// [`SwmrWriteAccess`](SuperblockStatusFlag::SwmrWriteAccess).
    pub const fn unknown_bits(self) -> u32 {
        self.bits
            & !(SuperblockStatusFlag::WriteAccess.mask()
                | SuperblockStatusFlag::SwmrWriteAccess.mask())
    }

    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self { bits }
    }
}

impl TryFrom<SuperblockStatusFlags> for u8 {
    type Error = FormatError;

    /// Encodes the status flags for a version 2 or 3 superblock.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SuperblockStatusFlagsTooWide`] if any bit above bit 7 is set.
    fn try_from(flags: SuperblockStatusFlags) -> Result<Self, Self::Error> {
        Self::try_from(flags.bits)
            .map_err(|_error| FormatError::SuperblockStatusFlagsTooWide { bits: flags.bits })
    }
}

/// An access flag in a version 3 superblock's [`SuperblockStatusFlags`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuperblockStatusFlag {
    /// The file is open for write access.
    WriteAccess,
    /// The file is open for single-writer/multiple-reader write access.
    SwmrWriteAccess,
}

impl SuperblockStatusFlag {
    // "Format Signature and Superblock" of the format specification, version 4.0, defines
    // write access at bit 0 and SWMR write access at bit 2 of the file consistency flags.
    const fn mask(self) -> u32 {
        match self {
            Self::WriteAccess => 0x01,
            Self::SwmrWriteAccess => 0x04,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::__private::SuperblockFields;
    use crate::address::{BaseAddressExt, StoredAddressExt};

    #[rstest]
    #[case(
        SuperblockStatus::Zero(0x8000_0004),
        SuperblockVersion::Zero,
        0x8000_0004
    )]
    #[case(SuperblockStatus::One(u32::MAX), SuperblockVersion::One, u32::MAX)]
    #[case(SuperblockStatus::Two(0x84), SuperblockVersion::Two, 0x84)]
    #[case(SuperblockStatus::Three(0x04), SuperblockVersion::Three, 0x04)]
    fn construction_preserves_the_typed_status(
        #[case] status: SuperblockStatus,
        #[case] version: SuperblockVersion,
        #[case] bits: u32,
    ) {
        let superblock = superblock(status);
        assert_eq!(superblock.status(), status);
        assert_eq!(superblock.version(), version);
        assert_eq!(superblock.consistency_flags().bits(), bits);
    }

    #[rstest]
    #[case(SuperblockStatus::Zero(0), SuperblockStatus::Zero(u32::MAX))]
    #[case(SuperblockStatus::One(0), SuperblockStatus::One(0x8000_0004))]
    #[case(SuperblockStatus::Two(0), SuperblockStatus::Two(0x84))]
    #[case(SuperblockStatus::Three(0), SuperblockStatus::Three(0x04))]
    fn changing_flags_preserves_version_and_other_fields(
        #[case] initial: SuperblockStatus,
        #[case] expected: SuperblockStatus,
    ) {
        let mut superblock = superblock(initial);
        let mut before = superblock.clone();
        superblock.set_consistency_flags(expected.flags()).unwrap();
        before.status = expected;
        assert_eq!(superblock, before);
        assert_eq!(superblock.version(), initial.version());
    }

    #[rstest]
    #[case(
        SuperblockStatus::Zero(0x8000_0004),
        SuperblockStatus::One(0x8000_0004)
    )]
    #[case(SuperblockStatus::One(u32::MAX), SuperblockStatus::Zero(u32::MAX))]
    #[case(SuperblockStatus::Zero(0x84), SuperblockStatus::Two(0x84))]
    #[case(SuperblockStatus::One(0x04), SuperblockStatus::Three(0x04))]
    #[case(SuperblockStatus::Two(0xFF), SuperblockStatus::Zero(0xFF))]
    #[case(SuperblockStatus::Three(0x84), SuperblockStatus::One(0x84))]
    #[case(SuperblockStatus::Two(0x01), SuperblockStatus::Three(0x01))]
    #[case(SuperblockStatus::Three(0x04), SuperblockStatus::Two(0x04))]
    fn changing_version_preserves_flags_and_other_fields(
        #[case] initial: SuperblockStatus,
        #[case] expected: SuperblockStatus,
    ) {
        let mut superblock = superblock(initial);
        let mut before = superblock.clone();
        superblock.set_version(expected.version()).unwrap();
        before.status = expected;
        assert_eq!(superblock, before);
        assert_eq!(superblock.consistency_flags(), initial.flags());
    }

    #[rstest]
    #[case(0x100)]
    #[case(0x8000_0004)]
    #[case(u32::MAX)]
    fn rejected_version_narrowing_preserves_the_superblock(
        #[case] bits: u32,
        #[values(SuperblockVersion::Zero, SuperblockVersion::One)] legacy: SuperblockVersion,
        #[values(SuperblockVersion::Two, SuperblockVersion::Three)] modern: SuperblockVersion,
    ) {
        let mut superblock = superblock(
            SuperblockStatus::new(legacy, SuperblockStatusFlags::from_bits(bits)).unwrap(),
        );
        let before = superblock.clone();
        assert_eq!(
            superblock.set_version(modern),
            Err(FormatError::SuperblockStatusFlagsTooWide { bits })
        );
        assert_eq!(superblock, before);
    }

    #[rstest]
    #[case(0x100)]
    #[case(0x8000_0004)]
    #[case(u32::MAX)]
    fn rejected_wide_flags_preserve_the_superblock(
        #[case] bits: u32,
        #[values(SuperblockStatus::Two(0x84), SuperblockStatus::Three(0x04))]
        status: SuperblockStatus,
    ) {
        let mut superblock = superblock(status);
        let before = superblock.clone();
        assert_eq!(
            superblock.set_consistency_flags(SuperblockStatusFlags::from_bits(bits)),
            Err(FormatError::SuperblockStatusFlagsTooWide { bits })
        );
        assert_eq!(superblock, before);
    }

    #[rstest]
    #[case(
        SuperblockStatus::Zero(0x8000_0004),
        SuperblockVersion::Zero,
        0x8000_0004
    )]
    #[case(SuperblockStatus::One(u32::MAX), SuperblockVersion::One, u32::MAX)]
    #[case(SuperblockStatus::Two(0x84), SuperblockVersion::Two, 0x84)]
    #[case(SuperblockStatus::Three(0x04), SuperblockVersion::Three, 0x04)]
    fn status_preserves_its_version_and_encoded_bits(
        #[case] status: SuperblockStatus,
        #[case] version: SuperblockVersion,
        #[case] bits: u32,
    ) {
        assert_eq!(status.version(), version);
        assert_eq!(status.flags().bits(), bits);
        assert_eq!(
            SuperblockStatus::new(version, status.flags()).unwrap(),
            status
        );
    }

    #[rstest]
    #[case(0x100)]
    #[case(0x8000_0004)]
    #[case(u32::MAX)]
    fn modern_status_rejects_bits_wider_than_its_field(
        #[case] bits: u32,
        #[values(SuperblockVersion::Two, SuperblockVersion::Three)] version: SuperblockVersion,
    ) {
        assert_eq!(
            SuperblockStatus::new(version, SuperblockStatusFlags::from_bits(bits)),
            Err(FormatError::SuperblockStatusFlagsTooWide { bits })
        );
    }

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

    #[rstest]
    #[case(0x00, (true, false, false, false, 0))]
    #[case(0x01, (false, true, false, false, 0))]
    #[case(0x04, (false, false, true, true, 0))]
    #[case(0x05, (false, true, true, false, 0))]
    #[case(0x02, (false, false, false, false, 0x02))]
    #[case(0x8000_0004, (false, false, true, true, 0x8000_0000))]
    #[case(u32::MAX, (false, true, true, false, 0xFFFF_FFFA))]
    fn observed_status_flags_preserve_unknown_and_inconsistent_bits(
        #[case] bits: u32,
        #[case] expected: (bool, bool, bool, bool, u32),
    ) {
        let flags = SuperblockStatusFlags::from_bits(bits);
        assert_eq!(flags.bits(), bits);
        assert_eq!(
            (
                flags.is_empty(),
                flags.contains(SuperblockStatusFlag::WriteAccess),
                flags.contains(SuperblockStatusFlag::SwmrWriteAccess),
                flags.is_inconsistent(),
                flags.unknown_bits()
            ),
            expected,
        );
    }
    fn superblock(status: SuperblockStatus) -> Superblock {
        SuperblockFields {
            status,
            offset_size: OffsetWidth::Eight,
            length_size: LengthWidth::Eight,
            base_address: BaseAddress::ZERO,
            eof_address: 4096,
            root_group_address: StoredAddress::ZERO,
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            superblock_extension_address: None,
            checksum: None,
        }
        .build()
    }
}
