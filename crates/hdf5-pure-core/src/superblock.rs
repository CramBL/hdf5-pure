use crate::address::BaseAddress;
use crate::address::StoredAddress;

/// The values encoded in a file's superblock.
///
/// The [base address](Self::base_address) and [end-of-file address](Self::eof_address) are
/// absolute file addresses. The other address fields hold base-relative [`StoredAddress`]
/// values. The reader interprets those values in the opened image's address context when
/// locating metadata in the physical source.
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
    /// Superblock version (0–3).
    pub version: u8,
    /// Size of offsets in bytes (2, 4, or 8).
    pub offset_size: u8,
    /// Size of lengths in bytes (2, 4, or 8).
    pub length_size: u8,
    /// Absolute file address of the first byte of HDF5 data, as encoded in the superblock.
    pub base_address: BaseAddress,
    /// Absolute file address of the first byte past the HDF5 data.
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
