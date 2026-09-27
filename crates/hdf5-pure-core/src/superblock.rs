use crate::address::BaseAddress;

/// The superblock of a file, as `hdf5_pure::File::superblock` returns it.
///
/// The superblock is the first HDF5 structure in a file. It fixes the format every other
/// structure is read in: the version, the width of the offsets and lengths in every message,
/// and the [base address](Self::base_address) the rest of the file's addresses are relative
/// to. The fields hold the values as stored. A field that a superblock version does not have
/// is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Superblock {
    /// Superblock version (0–3).
    pub version: u8,
    /// Size of offsets in bytes (2, 4, or 8).
    pub offset_size: u8,
    /// Size of lengths in bytes (2, 4, or 8).
    pub length_size: u8,
    /// File base address.
    pub base_address: BaseAddress,
    /// End-of-file address.
    pub eof_address: u64,
    /// Root group object header address (v2/v3) or from symbol table entry (v0/v1).
    pub root_group_address: u64,
    /// Group leaf node K (v0/v1 only).
    pub group_leaf_node_k: Option<u16>,
    /// Group internal node K (v0/v1 only).
    pub group_internal_node_k: Option<u16>,
    /// Indexed storage internal node K (v1 only).
    pub indexed_storage_internal_node_k: Option<u16>,
    /// Free space address (v0/v1 only).
    pub free_space_address: Option<u64>,
    /// Driver info block address (v0/v1 only).
    pub driver_info_address: Option<u64>,
    /// File consistency flags.
    pub consistency_flags: u32,
    /// Superblock extension address (v2/v3 only).
    pub superblock_extension_address: Option<u64>,
    /// Jenkins lookup3 checksum (v2/v3 only).
    pub checksum: Option<u32>,
}
