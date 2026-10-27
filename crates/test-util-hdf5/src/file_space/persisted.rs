//! Mutable file images for persisted free-space parsing and writable opening.

use std::path::Path;

use hdf5_pure::DecodedStoredAddress;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceInfo;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::StoredAddress;
use hdf5_pure::Superblock;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_format::__private as format;

use crate::storage;

/// The manager-address slots used by persisted-space fixtures.
#[derive(Clone, Copy, Debug)]
pub enum ManagerSlot {
    /// The fixture's metadata manager.
    Metadata,
    /// The fixture's raw-data manager.
    Raw,
    /// The local-heap slot used for empty managers.
    LocalHeap,
    /// The fixture's generic large-section manager.
    Generic,
    /// A slot outside the recognized paged allocation roles.
    Unknown,
}

impl ManagerSlot {
    /// Returns the position in [`FileSpaceInfo::manager_addrs`].
    pub const fn index(self) -> usize {
        match self {
            Self::Metadata => 0,
            Self::Raw => 2,
            Self::LocalHeap => 5,
            Self::Generic => 6,
            Self::Unknown => 7,
        }
    }
}

/// The file-space strategies supported by [`PersistedImage`].
#[derive(Clone, Copy, Debug)]
pub enum PersistedStrategy {
    /// Classic free-space managers with aggregation.
    FsmAggr,
    /// Paged allocation.
    Page,
}

/// A mutable file image with fixed locations for persisted managers.
///
/// [`new`](Self::new) creates a file through the facade. Manager bytes are constructed
/// independently by [`test_util::free_space::raw_manager_eight`]. Callers may edit the bytes
/// and file-space settings before [`write`](Self::write).
///
/// # Examples
///
/// ```
/// use test_util_hdf5::file_space::persisted::{
///     ManagerSlot, PersistedImage, PersistedStrategy, SECTION,
/// };
///
/// # let path = test_util::temp::temp_path("persisted.h5");
/// let mut image = PersistedImage::new(&path, PersistedStrategy::Page, 1);
/// image.manager(ManagerSlot::Metadata, &[(SECTION, 32)]);
/// let bytes = image.write(&path);
/// assert_eq!(std::fs::read(&path)?, bytes);
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct PersistedImage {
    /// The complete physical file image, including edited manager bytes.
    pub bytes: Vec<u8>,
    /// The file-space message serialized by [`write`](Self::write).
    pub info: FileSpaceInfo,
    original_info: Vec<u8>,
    superblock: Superblock,
    extension: (usize, usize),
}

impl PersistedImage {
    /// Creates an empty file with persistent free-space tracking.
    ///
    /// Uses 8-byte addresses and lengths. The in-memory logical end of file is [`EOF`], and
    /// [`info`](Self::info) starts with undefined manager addresses.
    ///
    /// # Panics
    ///
    /// Panics if creating or inspecting the file fails, the widths are not 8 bytes, or the
    /// extension does not occupy one chunk.
    pub fn new(path: &Path, strategy: PersistedStrategy, threshold: u64) -> Self {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(match strategy {
                PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
                PersistedStrategy::Page => FileSpaceStrategy::Page,
            })
            .with_persistence(true)
            .with_threshold(threshold),
        );
        builder.write(path).unwrap();
        let file = File::open(path).unwrap();
        let mut info = file.file_space_info().unwrap().unwrap();
        let mut superblock = file.superblock().clone();
        let mut bytes = std::fs::read(path).unwrap();
        assert_eq!(
            (superblock.offset_size.get(), superblock.length_size.get()),
            (8, 8)
        );
        let widths = format::FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        };
        let original_info = format::serialize_file_space_info_v1(widths, &info).unwrap();
        let chunks = storage::header_extents(
            file.superblock(),
            &bytes,
            superblock.superblock_extension_address.unwrap().get(),
        );
        assert_eq!(chunks.len(), 1);
        let (at, length) = chunks[0];
        drop(file);
        info.manager_addrs.fill(DecodedStoredAddress::Undefined);
        superblock.eof_address = EOF;
        bytes.resize(usize::try_from(EOF).unwrap(), 0);
        Self {
            bytes,
            info,
            original_info,
            superblock,
            extension: (
                usize::try_from(at).unwrap(),
                usize::try_from(length).unwrap(),
            ),
        }
    }

    /// Encodes sections at the fixed location for `slot` and records its header address.
    ///
    /// Sections are `(stored_address, length)` pairs. An empty slice creates a defined empty
    /// manager. The header and section list have checksums.
    ///
    /// # Panics
    ///
    /// Panics if encoding fails, the manager does not fit in [`bytes`](Self::bytes), or
    /// [`info`](Self::info) has too few manager addresses.
    pub fn manager(&mut self, slot: ManagerSlot, sections: &[(u64, u64)]) {
        let address = HEADER + u64::try_from(slot.index()).unwrap() * STRIDE;
        let (header, list) =
            test_util::free_space::raw_manager_eight(address, address + HEADER_LEN, sections);
        let at = usize::try_from(address).unwrap();
        self.bytes[at..at + header.len()].copy_from_slice(&header);
        self.bytes[at + header.len()..at + header.len() + list.len()].copy_from_slice(&list);
        self.info.manager_addrs[slot.index()] =
            DecodedStoredAddress::Defined(StoredAddress::new(address));
    }

    /// Serializes the file-space message and superblock, writes the file, and returns its bytes.
    ///
    /// Recomputes the extension checksum. Manager bytes are written as supplied.
    ///
    /// # Panics
    ///
    /// Panics if serialization fails, the original file-space message cannot be found, its
    /// replacement changes size, the extension is outside the image, or writing fails.
    pub fn write(self, path: &Path) -> Vec<u8> {
        let Self {
            mut bytes,
            info,
            original_info,
            superblock,
            extension: (at, length),
        } = self;
        let widths = format::FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        };
        let payload = format::serialize_file_space_info_v1(widths, &info).unwrap();
        let offset = bytes[at..at + length]
            .windows(original_info.len())
            .position(|window| window == original_info)
            .unwrap()
            + at;
        assert_eq!(payload.len(), original_info.len());
        bytes[offset..offset + payload.len()].copy_from_slice(&payload);
        test_util::checksum::restamp(&mut bytes, at, length);
        let encoded = format::serialize_superblock(&superblock).unwrap();
        bytes[..encoded.len()].copy_from_slice(&encoded);
        std::fs::write(path, &bytes).unwrap();
        bytes
    }
}

/// The logical end of file in a newly constructed fixture.
pub const EOF: u64 = 65_536;
/// The stored address of the first manager header.
pub const HEADER: u64 = 8192;
/// The first manager header's offset in [`PersistedImage::bytes`].
pub const HEADER_AT: usize = 8192;
/// The length of [`test_util::free_space::single_section_header`], in bytes.
pub const HEADER_LEN: u64 = 82;
/// The first section list's offset in [`PersistedImage::bytes`].
pub const LIST: usize = HEADER_AT + 82;
/// The stored address used for free sections in the fixture.
pub const SECTION: u64 = 16_384;
/// The byte distance between neighboring manager slots.
pub const STRIDE: u64 = 256;
