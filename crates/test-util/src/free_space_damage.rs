//! Corruptions of free-space managers with eight-byte addresses and lengths.
//!
//! The version 0 header and section list are defined in "Free-space Index" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex

use crate::bytes;
use crate::checksum;

/// A field or checksum corruption applied by [`ManagerLayout::damage`].
#[derive(Clone, Copy, Debug)]
pub enum Damage {
    /// Sets the section-list allocation one byte beyond the input's end.
    AllocatedLength,
    /// Sets the address-space width to zero bits.
    HeaderBits,
    /// Flips one bit of the header checksum.
    HeaderChecksum,
    /// Sets the client ID to the fractal-heap client.
    HeaderClient,
    /// Replaces the first header signature byte with zero.
    HeaderSignature,
    /// Sets the header version to one.
    HeaderVersion,
    /// Sets the section-list address to the input's end.
    ListAddress,
    /// Flips one bit of the section-list checksum.
    ListChecksum,
    /// Sets the section list's owner address to zero.
    ListOwner,
    /// Replaces the first section-list signature byte with zero.
    ListSignature,
    /// Sets the section-list version to one.
    ListVersion,
    /// Sets the section-list address to undefined.
    UndefinedList,
    /// Sets the used section-list length one byte above its allocated length.
    UsedLength,
}

/// The locations, lengths, and section count of a manager with eight-byte widths.
///
/// Use inputs with a zero base address. Addresses are byte offsets in the input buffer passed
/// to [`Self::read`].
#[derive(Clone, Copy, Debug)]
pub struct ManagerLayout {
    /// The header's byte offset.
    pub header: usize,
    /// The section list's byte offset.
    pub list: usize,
    /// The used section-list length in bytes, including its checksum.
    pub used: usize,
    /// The allocated section-list length in bytes.
    pub allocated: u64,
    /// The total section count stored in the header.
    pub count: u64,
}

impl ManagerLayout {
    /// Reads manager fields at `header` with eight-byte addresses and lengths.
    ///
    /// Checks the header signature. The caller supplies a complete, valid manager.
    ///
    /// # Panics
    ///
    /// Panics if the header fields are outside `bytes`, the header signature is invalid, or a
    /// stored address or used length exceeds [`usize::MAX`].
    pub fn read(bytes: &[u8], header: usize) -> Self {
        assert_eq!(
            &bytes[header..header + SIGNATURE_BYTES],
            super::free_space::SIGNATURE
        );
        Self {
            header,
            list: usize::try_from(bytes::u64_at(bytes, header + LIST_ADDRESS)).unwrap(),
            used: usize::try_from(bytes::u64_at(bytes, header + USED_LENGTH)).unwrap(),
            allocated: bytes::u64_at(bytes, header + ALLOCATED_LENGTH),
            count: bytes::u64_at(bytes, header + SECTION_COUNT),
        }
    }

    /// Copies `original` and corrupts the selected field or checksum.
    ///
    /// Recomputes the affected structure's checksum for field corruptions. For checksum
    /// corruptions, flips one bit in the checksum.
    ///
    /// # Panics
    ///
    /// Panics if `self` does not describe a complete header and section list in `original`, or a
    /// corrupted length cannot be represented.
    ///
    /// # Examples
    ///
    /// ```
    /// use test_util::free_space;
    /// use test_util::free_space_damage::{Damage, HEADER_BYTES, ManagerLayout};
    ///
    /// let (header, list) = free_space::raw_manager_eight(0, HEADER_BYTES as u64, &[(256, 16)]);
    /// let original = [header, list].concat();
    /// let layout = ManagerLayout::read(&original, 0);
    /// assert_eq!(layout.count, 1);
    /// let damaged = layout.damage(&original, Damage::HeaderChecksum);
    /// assert_eq!(&damaged[..HEADER_BYTES - 1], &original[..HEADER_BYTES - 1]);
    /// assert_eq!(damaged[HEADER_BYTES - 1], original[HEADER_BYTES - 1] ^ 1);
    /// assert_eq!(&damaged[HEADER_BYTES..], &original[HEADER_BYTES..]);
    /// ```
    pub fn damage(self, original: &[u8], damage: Damage) -> Vec<u8> {
        let Self {
            header,
            list,
            used,
            allocated,
            count: _,
        } = self;
        let mut bytes = original.to_vec();
        let (at, length) = match damage {
            Damage::ListChecksum
            | Damage::ListOwner
            | Damage::ListSignature
            | Damage::ListVersion => (list, used),
            _ => (header, HEADER_BYTES),
        };
        match damage {
            Damage::HeaderSignature | Damage::ListSignature => bytes[at] = 0,
            Damage::HeaderVersion | Damage::ListVersion => bytes[at + VERSION] = 1,
            Damage::HeaderClient => bytes[header + CLIENT] = 0,
            Damage::HeaderBits => {
                bytes[header + ADDRESS_BITS..header + ADDRESS_BITS + ADDRESS_BITS_BYTES].fill(0)
            }
            Damage::HeaderChecksum | Damage::ListChecksum => {
                bytes[at + length - 1] ^= 1;
                return bytes;
            }
            Damage::UsedLength => bytes[header + USED_LENGTH..header + USED_LENGTH + WORD_BYTES]
                .copy_from_slice(&(allocated + 1).to_le_bytes()),
            Damage::UndefinedList => {
                bytes[header + LIST_ADDRESS..header + LIST_ADDRESS + WORD_BYTES].fill(0xff)
            }
            Damage::ListAddress => bytes[header + LIST_ADDRESS..header + LIST_ADDRESS + WORD_BYTES]
                .copy_from_slice(&u64::try_from(original.len()).unwrap().to_le_bytes()),
            Damage::AllocatedLength => bytes
                [header + ALLOCATED_LENGTH..header + ALLOCATED_LENGTH + WORD_BYTES]
                .copy_from_slice(
                    &u64::try_from(original.len() - list + 1)
                        .unwrap()
                        .to_le_bytes(),
                ),
            Damage::ListOwner => bytes[list + OWNER..list + OWNER + WORD_BYTES].fill(0),
        }
        checksum::restamp(&mut bytes, at, length);
        bytes
    }
}

/// The version 0 header length with eight-byte addresses and lengths.
///
/// Defined in "Free-space Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
pub const HEADER_BYTES: usize = 82;
/// The checksum length in a version 0 manager header or section list.
///
/// Defined in "Free-space Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
pub const CHECKSUM_BYTES: usize = core::mem::size_of::<u32>();
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const SIGNATURE_BYTES: usize = 4;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const VERSION: usize = 4;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const CLIENT: usize = 5;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const OWNER: usize = 5;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const SECTION_COUNT: usize = 14;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const ADDRESS_BITS: usize = 44;
// The field width is defined in "Free-space Index" of the format specification, version 4.0.
const ADDRESS_BITS_BYTES: usize = core::mem::size_of::<u16>();
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const LIST_ADDRESS: usize = 54;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const USED_LENGTH: usize = 62;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const ALLOCATED_LENGTH: usize = 70;
// The field layout is defined in "Free-space Index" of the format specification, version 4.0.
const WORD_BYTES: usize = core::mem::size_of::<u64>();
