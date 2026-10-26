//! Owned group links, creation-order policy, and compact or dense storage addresses.
//!
//! Link names and symbolic targets retain their encoded bytes. The format specification,
//! version 4.0, defines ["The Link Message"][link] and ["The Link Info Message"][info].
//!
//! [link]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
//! [info]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::bytes;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_info::LinkInfoMessage;
use crate::link_message::{EXTERNAL_LINK, HARD_LINK, LinkPrefix, SOFT_LINK};
use crate::width::{self, OffsetWidth, UintWidth};

/// A group link set with distinct names and consistent creation-order and storage metadata.
///
/// Tracked links have distinct indexes below the stored next-index counter. Compact storage
/// retains the indexing policy with an undefined creation-order index address. Dense storage
/// has defined heap and name-index addresses, plus a creation-order index when indexing is enabled.
/// Addresses remain stored file addresses at the width passed to the constructor.
///
/// The storage fields are defined in "The Link Info Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
#[derive(Clone, Debug, PartialEq)]
pub struct GroupLinkStorage {
    offset_width: OffsetWidth,
    links: Vec<OwnedLink>,
    creation_order: LinkCreationPolicy,
    addresses: Option<DenseLinkAddresses>,
}

impl GroupLinkStorage {
    /// Parses Link Info bytes and combines them with the group's owned links.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::Flags`] for reserved flags and
    /// [`LinkStorageError::TrailingBytes`] for bytes after the message. Returns the errors of
    /// [`Self::new`] for inconsistent links, policy, or addresses, and
    /// [`LinkStorageError::Format`] for malformed Link Info fields.
    pub fn parse(
        data: &[u8],
        links: Vec<OwnedLink>,
        offset_width: OffsetWidth,
    ) -> Result<Self, LinkStorageError> {
        let info = LinkInfoMessage::parse_with_offset_width(data, offset_width)?;
        let flags = data
            .get(1)
            .copied()
            .ok_or(LinkStorageError::StorageAddresses)?;
        if flags & LINK_INFO_RESERVED_FLAGS != 0 {
            return Err(LinkStorageError::Flags(flags));
        }
        if info.serialize(offset_width).len() != data.len() {
            return Err(LinkStorageError::TrailingBytes);
        }
        Self::new(&info, links, offset_width)
    }

    /// Combines Link Info metadata with links that can be encoded at `offset_width`.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::DuplicateName`] or
    /// [`LinkStorageError::DuplicateCreationOrder`] for repeated names or indexes,
    /// [`LinkStorageError::CreationOrderPresence`] if link indexes disagree with tracking, and
    /// [`LinkStorageError::CreationOrderCounter`] if an index is not below the next counter.
    /// Returns the errors of [`LinkCreationPolicy::from_info`] for an unsupported policy,
    /// [`LinkStorageError::StorageAddresses`] for inconsistent storage addresses, and the
    /// errors of [`OwnedLink::encoded_len`] for a link that cannot be encoded.
    /// Returns [`LinkStorageError::UndefinedAddress`] if a required storage address is undefined,
    /// or [`LinkStorageError::Format`] if it does not fit `offset_width`.
    pub fn new(
        info: &LinkInfoMessage,
        links: Vec<OwnedLink>,
        offset_width: OffsetWidth,
    ) -> Result<Self, LinkStorageError> {
        let creation_order = LinkCreationPolicy::from_info(info)?;
        let mut names = BTreeSet::new();
        let mut indexes = BTreeSet::new();
        for link in &links {
            link.encoded_len(offset_width)?;
            if !names.insert(link.name()) {
                return Err(LinkStorageError::DuplicateName(link.name().to_vec()));
            }
            match (creation_order.next_index(), link.creation_order()) {
                (None, None) => {}
                (Some(next), Some(index)) if index < next => {
                    if !indexes.insert(index) {
                        return Err(LinkStorageError::DuplicateCreationOrder(index));
                    }
                }
                (Some(next), Some(index)) => {
                    return Err(LinkStorageError::CreationOrderCounter { index, next });
                }
                _ => return Err(LinkStorageError::CreationOrderPresence),
            }
        }
        let addresses = DenseLinkAddresses::from_info(info, offset_width)?;
        Ok(Self {
            offset_width,
            links,
            creation_order,
            addresses,
        })
    }

    /// Inserts a link and assigns its creation index from the stored counter when tracked.
    ///
    /// Existing indexes and deletion gaps are preserved. An error leaves the set and counter
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::CreationOrderPresence`] if `link` already has an index,
    /// [`LinkStorageError::DuplicateName`] if its name is present, and
    /// [`LinkStorageError::CreationOrderExhausted`] if the counter cannot be incremented.
    /// Returns the errors of [`OwnedLink::encoded_len`] if the link cannot be encoded.
    pub fn insert(&mut self, mut link: OwnedLink) -> Result<(), LinkStorageError> {
        let Self {
            offset_width,
            links,
            creation_order,
            addresses: _,
        } = self;
        if link.creation_order.is_some() {
            return Err(LinkStorageError::CreationOrderPresence);
        }
        if links.iter().any(|existing| existing.name() == link.name()) {
            return Err(LinkStorageError::DuplicateName(link.name));
        }
        link.encoded_len(*offset_width)?;
        let mut next_policy = *creation_order;
        link.creation_order = next_policy.take()?;
        links.push(link);
        *creation_order = next_policy;
        Ok(())
    }

    /// Removes the link with these name bytes, preserving the counter and surviving indexes.
    ///
    /// Returns `None` if the name is absent.
    pub fn remove(&mut self, name: &[u8]) -> Option<OwnedLink> {
        self.links
            .iter()
            .position(|link| link.name() == name)
            .map(|index| self.links.remove(index))
    }

    /// Replaces a named hard link's stored target address at this group's encoding width.
    ///
    /// Returns `false` if the name is absent. An error leaves the target unchanged.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OwnedLink::set_hard_target`] for an unsupported replacement.
    pub fn set_hard_target(
        &mut self,
        name: &[u8],
        address: StoredAddress,
    ) -> Result<bool, LinkStorageError> {
        let Some(link) = self.links.iter_mut().find(|link| link.name() == name) else {
            return Ok(false);
        };
        link.set_hard_target(address, self.offset_width)?;
        Ok(true)
    }

    /// Clears dense-storage addresses while preserving the creation-order policy.
    pub fn use_compact_storage(&mut self) {
        self.addresses = None;
    }

    /// Sets the dense-storage addresses without changing links or creation-order policy.
    ///
    /// An error preserves the existing storage addresses.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::StorageAddresses`] if `creation_index` disagrees with the
    /// indexing policy, [`LinkStorageError::UndefinedAddress`] for an undefined address, and
    /// [`LinkStorageError::Format`] for an address wider than the stored encoding width.
    pub fn use_dense_storage(
        &mut self,
        heap: StoredAddress,
        name_index: StoredAddress,
        creation_index: Option<StoredAddress>,
    ) -> Result<(), LinkStorageError> {
        if creation_index.is_some() != self.creation_order.is_indexed() {
            return Err(LinkStorageError::StorageAddresses);
        }
        self.addresses = Some(DenseLinkAddresses::new(
            heap,
            name_index,
            creation_index,
            self.offset_width,
        )?);
        Ok(())
    }

    /// Returns the file-address width used to check link targets and storage addresses.
    pub fn offset_width(&self) -> OffsetWidth {
        self.offset_width
    }

    /// Returns the links in their retained order.
    pub fn links(&self) -> &[OwnedLink] {
        &self.links
    }

    /// Returns the tracking and indexing policy with its current next-index counter.
    pub fn creation_order(&self) -> LinkCreationPolicy {
        self.creation_order
    }

    /// Returns Link Info metadata with the current policy, counter, and storage addresses.
    pub fn info(&self) -> LinkInfoMessage {
        LinkInfoMessage {
            max_creation_order: self.creation_order.next_index(),
            fractal_heap_address: self.addresses.map(|addresses| addresses.heap),
            btree_name_index_address: self.addresses.map(|addresses| addresses.name_index),
            btree_creation_order_address: self.creation_order.is_indexed().then(|| {
                self.addresses
                    .and_then(|addresses| addresses.creation_index)
            }),
        }
    }
}

/// A group's link creation-order policy and checked next-index counter.
///
/// The policy is encoded by the tracking and indexing flags of "The Link Info Message" in
/// the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkCreationPolicy {
    /// Tracks link creation indexes and maintains a creation-order index.
    Indexed {
        /// The index assigned to the next inserted link.
        next: LinkCreationCounter,
    },
    /// Tracks link creation indexes without a creation-order index.
    Tracked {
        /// The index assigned to the next inserted link.
        next: LinkCreationCounter,
    },
    /// Omits link creation indexes and the counter.
    Untracked,
}

impl LinkCreationPolicy {
    /// Reads tracking, indexing, and the next-index counter from Link Info metadata.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::IndexingWithoutTracking`] if indexing is enabled without
    /// tracking, and [`LinkStorageError::CreationOrderCounterRange`] if the counter exceeds
    /// the range accepted by [`LinkCreationCounter::new`].
    pub fn from_info(info: &LinkInfoMessage) -> Result<Self, LinkStorageError> {
        let next = info
            .max_creation_order
            .map(LinkCreationCounter::new)
            .transpose()?;
        match (next, info.btree_creation_order_address.is_some()) {
            (None, false) => Ok(Self::Untracked),
            (Some(next), false) => Ok(Self::Tracked { next }),
            (Some(next), true) => Ok(Self::Indexed { next }),
            (None, true) => Err(LinkStorageError::IndexingWithoutTracking),
        }
    }

    /// Assigns the next tracked index and increments the counter, or returns `None` when untracked.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::CreationOrderExhausted`] if the counter is [`i64::MAX`].
    /// An error leaves the policy unchanged.
    pub fn take(&mut self) -> Result<Option<u64>, LinkStorageError> {
        let (Self::Tracked { next } | Self::Indexed { next }) = self else {
            return Ok(None);
        };
        let index = next.get();
        let following = index
            .checked_add(1)
            .filter(|&value| value <= MAX_LINK_CREATION_COUNTER)
            .ok_or(LinkStorageError::CreationOrderExhausted)?;
        *next = LinkCreationCounter(following);
        Ok(Some(index))
    }

    /// Returns the next tracked creation index, or `None` when untracked.
    pub fn next_index(self) -> Option<u64> {
        match self {
            Self::Untracked => None,
            Self::Tracked { next } | Self::Indexed { next } => Some(next.get()),
        }
    }

    /// Returns whether creation-order indexing is enabled.
    pub fn is_indexed(self) -> bool {
        matches!(self, Self::Indexed { .. })
    }
}

/// A next-link creation counter in `0..=i64::MAX`.
///
/// libhdf5 decodes the Link Info counter as a signed value and rejects negative values in
/// `H5O__linfo_decode` in `H5Olinfo.c` (HDF5 1.14.6 and 2.2.0). A counter equal to [`i64::MAX`]
/// is representable but cannot be incremented for another link.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkCreationCounter(u64);

impl LinkCreationCounter {
    /// Creates a counter that libhdf5 can decode as a nonnegative signed value.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::CreationOrderCounterRange`] if `next` exceeds [`i64::MAX`].
    pub fn new(next: u64) -> Result<Self, LinkStorageError> {
        if next > MAX_LINK_CREATION_COUNTER {
            return Err(LinkStorageError::CreationOrderCounterRange {
                next,
                maximum: MAX_LINK_CREATION_COUNTER,
            });
        }
        Ok(Self(next))
    }

    /// Returns the stored next-index value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// An owned Link message with byte-preserving name and symbolic-target fields.
///
/// Character-set markers are retained independently of the name bytes. The name is checked as a
/// link component, and targets are checked for syntax and encoded lengths without decoding their
/// character data.
/// Creation indexes are checked against a group's policy by [`GroupLinkStorage::new`].
/// The fields are defined in "The Link Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedLink {
    creation_order: Option<u64>,
    charset: CharacterSet,
    name: Vec<u8>,
    target: OwnedLinkTarget,
}

impl OwnedLink {
    /// Parses a version 1 Link message while preserving name and symbolic-target bytes.
    ///
    /// Supports hard, soft, and version 0 external links with zero external flags.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::Flags`] for reserved flags,
    /// [`LinkStorageError::TrailingBytes`] for bytes after the target,
    /// [`LinkStorageError::ExternalVersionFlags`] for unsupported external versions or flags,
    /// and [`LinkStorageError::ExternalValue`] for incomplete external framing.
    /// Returns [`LinkStorageError::UndefinedAddress`] for an undefined hard target and the
    /// errors of [`Self::new`] for invalid names or targets. Returns
    /// [`LinkStorageError::Format`] for malformed prefix fields, unsupported link types, or
    /// truncated fields.
    pub fn parse(data: &[u8], offset_width: OffsetWidth) -> Result<Self, LinkStorageError> {
        let prefix = LinkPrefix::parse(data)?;
        let flags = data
            .get(1)
            .copied()
            .ok_or(LinkStorageError::TrailingBytes)?;
        if flags & LINK_RESERVED_FLAGS != 0 {
            return Err(LinkStorageError::Flags(flags));
        }
        let mut fields = bytes::Fields::new(data, prefix.target_pos);
        let target = match prefix.link_type_code {
            HARD_LINK => match fields.decoded_address(offset_width)? {
                DecodedStoredAddress::Defined(object_header_address) => OwnedLinkTarget::Hard {
                    object_header_address,
                },
                DecodedStoredAddress::Undefined => return Err(LinkStorageError::UndefinedAddress),
            },
            SOFT_LINK | EXTERNAL_LINK => {
                let length = usize::from(fields.u16()?);
                let pos = fields.pos();
                bytes::ensure_len(data, pos, length)?;
                let value = data
                    .get(pos..)
                    .and_then(|bytes| bytes.get(..length))
                    .ok_or(LinkStorageError::TrailingBytes)?;
                if pos.checked_add(length) != Some(data.len()) {
                    return Err(LinkStorageError::TrailingBytes);
                }
                if prefix.link_type_code == SOFT_LINK {
                    OwnedLinkTarget::Soft {
                        target_path: value.to_vec(),
                    }
                } else {
                    let (&version_flags, strings) =
                        value.split_first().ok_or(LinkStorageError::ExternalValue)?;
                    if version_flags != EXTERNAL_VERSION_FLAGS {
                        return Err(LinkStorageError::ExternalVersionFlags(version_flags));
                    }
                    let separator = strings
                        .iter()
                        .position(|&byte| byte == 0)
                        .ok_or(LinkStorageError::ExternalValue)?;
                    let (filename, rest) = strings.split_at(separator);
                    let object_path = rest
                        .get(1..)
                        .and_then(|rest| rest.strip_suffix(&[0]))
                        .ok_or(LinkStorageError::ExternalValue)?;
                    OwnedLinkTarget::External {
                        filename: filename.to_vec(),
                        object_path: object_path.to_vec(),
                    }
                }
            }
            other => return Err(FormatError::InvalidLinkType(other).into()),
        };
        if prefix.link_type_code == HARD_LINK && fields.pos() != data.len() {
            return Err(LinkStorageError::TrailingBytes);
        }
        Self::new(
            prefix.name.to_vec(),
            prefix.charset,
            target,
            prefix.creation_order,
        )
    }

    /// Creates an owned link from encoded name and target bytes and an optional creation index.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::InvalidName`] for an empty name, `.` component, or a name
    /// containing a null or slash. Returns [`LinkStorageError::InvalidTarget`] for an empty or
    /// null-containing symbolic target, [`LinkStorageError::TargetLength`] if the target value
    /// exceeds its two-byte length field, and [`LinkStorageError::LengthOverflow`] if its
    /// framing length overflows [`usize`]. Returns [`LinkStorageError::UndefinedAddress`]
    /// for an eight-byte undefined hard target.
    pub fn new(
        name: Vec<u8>,
        charset: CharacterSet,
        target: OwnedLinkTarget,
        creation_order: Option<u64>,
    ) -> Result<Self, LinkStorageError> {
        if name.is_empty() || name == b"." || name.contains(&0) || name.contains(&b'/') {
            return Err(LinkStorageError::InvalidName(name));
        }
        target.encoded_len(OffsetWidth::Eight)?;
        Ok(Self {
            creation_order,
            charset,
            name,
            target,
        })
    }

    /// Replaces a hard link's target with a defined stored address of `offset_width`.
    ///
    /// An error preserves the existing target.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::UndefinedAddress`] for an undefined address,
    /// [`LinkStorageError::Format`] for an address that does not fit `offset_width`, and
    /// [`LinkStorageError::NotHardLink`] for a soft or external link.
    pub fn set_hard_target(
        &mut self,
        address: StoredAddress,
        offset_width: OffsetWidth,
    ) -> Result<(), LinkStorageError> {
        required_address(address, offset_width)?;
        let OwnedLinkTarget::Hard {
            object_header_address,
        } = &mut self.target
        else {
            return Err(LinkStorageError::NotHardLink);
        };
        *object_header_address = address;
        Ok(())
    }

    /// Returns the stored creation index, if present.
    pub fn creation_order(&self) -> Option<u64> {
        self.creation_order
    }

    /// Returns the name's character-set marker.
    pub fn charset(&self) -> CharacterSet {
        self.charset.clone()
    }

    /// Returns the name bytes without character decoding.
    pub fn name(&self) -> &[u8] {
        &self.name
    }

    /// Returns the hard address or symbolic-target bytes.
    pub fn target(&self) -> &OwnedLinkTarget {
        &self.target
    }

    /// Returns the complete serialized Link message length at `offset_width`.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::UndefinedAddress`] for an undefined hard target,
    /// [`LinkStorageError::Format`] for a hard target that does not fit `offset_width`, and
    /// [`LinkStorageError::LengthOverflow`] if the complete length exceeds [`usize::MAX`].
    pub fn encoded_len(&self, offset_width: OffsetWidth) -> Result<usize, LinkStorageError> {
        let Self {
            creation_order,
            charset,
            name,
            target,
        } = self;
        let name_width = UintWidth::smallest_for_len(name.len());
        let prefix = LINK_PREFIX_LEN
            + usize::from(name_width.get())
            + usize::from(creation_order.is_some()) * CREATION_ORDER_LEN
            + usize::from(charset != &CharacterSet::Ascii)
            + usize::from(!matches!(target, OwnedLinkTarget::Hard { .. }));
        let target_length = target.encoded_len(offset_width)?;
        prefix
            .checked_add(name.len())
            .and_then(|len| len.checked_add(target_length))
            .ok_or(LinkStorageError::LengthOverflow)
    }

    /// Encodes a Link message using the smallest field that can encode its name length.
    ///
    /// Preserves name bytes, target bytes, character-set marker, and creation index. The writer
    /// emits the type field for symbolic links and the character-set field for names marked UTF-8.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::encoded_len`] if the message cannot be encoded at
    /// `offset_width`.
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, LinkStorageError> {
        let length = self.encoded_len(offset_width)?;
        let Self {
            creation_order,
            charset,
            name,
            target,
        } = self;
        let mut out = Vec::with_capacity(length);
        let name_width = UintWidth::smallest_for_len(name.len());
        let mut flags = name_width.flag_bits();
        if creation_order.is_some() {
            flags |= CREATION_ORDER_PRESENT;
        }
        if charset != &CharacterSet::Ascii {
            flags |= CHARSET_PRESENT;
        }
        if !matches!(target, OwnedLinkTarget::Hard { .. }) {
            flags |= LINK_TYPE_PRESENT;
        }
        out.extend_from_slice(&[LINK_VERSION, flags]);
        match target {
            OwnedLinkTarget::Soft { .. } => out.push(SOFT_LINK),
            OwnedLinkTarget::External { .. } => out.push(EXTERNAL_LINK),
            OwnedLinkTarget::Hard { .. } => {}
        }
        if let Some(index) = creation_order {
            out.extend_from_slice(&index.to_le_bytes());
        }
        if charset != &CharacterSet::Ascii {
            out.push(match charset {
                CharacterSet::Ascii => ASCII_CHARSET,
                CharacterSet::Utf8 => UTF8_CHARSET,
            });
        }
        name_width.write(&mut out, name.len());
        out.extend_from_slice(name);
        match target {
            OwnedLinkTarget::Hard {
                object_header_address,
            } => bytes::try_write_decoded_address(
                &mut out,
                DecodedStoredAddress::Defined(*object_header_address),
                offset_width,
            )?,
            OwnedLinkTarget::Soft { target_path } => {
                let length = u16::try_from(target_path.len())
                    .map_err(|_error| LinkStorageError::TargetLength(target_path.len()))?;
                out.extend_from_slice(&length.to_le_bytes());
                out.extend_from_slice(target_path);
            }
            OwnedLinkTarget::External {
                filename,
                object_path,
            } => {
                let length = self
                    .target
                    .encoded_len(offset_width)?
                    .checked_sub(TARGET_LENGTH_LEN)
                    .ok_or(LinkStorageError::LengthOverflow)?;
                let length = u16::try_from(length)
                    .map_err(|_error| LinkStorageError::TargetLength(length))?;
                out.extend_from_slice(&length.to_le_bytes());
                out.push(EXTERNAL_VERSION_FLAGS);
                out.extend_from_slice(filename);
                out.push(0);
                out.extend_from_slice(object_path);
                out.push(0);
            }
        }
        debug_assert_eq!(out.len(), length);
        Ok(out)
    }
}

/// A hard link's stored address or a symbolic link's undecoded target bytes.
///
/// [`OwnedLink::new`] checks target syntax and encoded lengths. The target formats are defined
/// in "The Link Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OwnedLinkTarget {
    /// Refers to an object header in the same file.
    Hard {
        /// The stored object-header address.
        object_header_address: StoredAddress,
    },
    /// Refers to a path in the same file.
    Soft {
        /// The target path bytes, without a null terminator.
        target_path: Vec<u8>,
    },
    /// Refers to an object path in another HDF5 file.
    External {
        /// The filename bytes, without a null terminator.
        filename: Vec<u8>,
        /// The object path bytes, without a null terminator.
        object_path: Vec<u8>,
    },
}

impl OwnedLinkTarget {
    /// Returns the encoded target length, including a symbolic target's length field.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::UndefinedAddress`] or [`LinkStorageError::Format`] for an
    /// invalid hard address, [`LinkStorageError::InvalidTarget`] for invalid symbolic paths,
    /// [`LinkStorageError::TargetLength`] for a value longer than [`u16::MAX`], and
    /// [`LinkStorageError::LengthOverflow`] for a framing length that exceeds [`usize::MAX`].
    fn encoded_len(&self, offset_width: OffsetWidth) -> Result<usize, LinkStorageError> {
        let value_length = match self {
            Self::Hard {
                object_header_address,
            } => {
                required_address(*object_header_address, offset_width)?;
                return Ok(usize::from(offset_width.get()));
            }
            Self::Soft { target_path } => {
                Self::path(target_path)?;
                target_path.len()
            }
            Self::External {
                filename,
                object_path,
            } => {
                Self::path(filename)?;
                Self::path(object_path)?;
                filename
                    .len()
                    .checked_add(object_path.len())
                    .and_then(|len| len.checked_add(EXTERNAL_FRAMING_LEN))
                    .ok_or(LinkStorageError::LengthOverflow)?
            }
        };
        u16::try_from(value_length)
            .map_err(|_error| LinkStorageError::TargetLength(value_length))?;
        value_length
            .checked_add(TARGET_LENGTH_LEN)
            .ok_or(LinkStorageError::LengthOverflow)
    }

    /// Checks that a symbolic target is nonempty and has no null bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::InvalidTarget`] for an empty or null-containing target.
    fn path(bytes: &[u8]) -> Result<(), LinkStorageError> {
        if bytes.is_empty() || bytes.contains(&0) {
            return Err(LinkStorageError::InvalidTarget);
        }
        Ok(())
    }
}

/// Defined stored addresses of a dense link heap and its required indexes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DenseLinkAddresses {
    heap: StoredAddress,
    name_index: StoredAddress,
    creation_index: Option<StoredAddress>,
}

impl DenseLinkAddresses {
    /// Reads a complete dense address set, or `None` for compact storage.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::StorageAddresses`] for an incomplete set,
    /// [`LinkStorageError::UndefinedAddress`] for undefined addresses, and
    /// [`LinkStorageError::Format`] for addresses that do not fit `offset_width`.
    fn from_info(
        info: &LinkInfoMessage,
        offset_width: OffsetWidth,
    ) -> Result<Option<Self>, LinkStorageError> {
        match (
            info.fractal_heap_address,
            info.btree_name_index_address,
            info.btree_creation_order_address,
        ) {
            (None, None, None | Some(None)) => Ok(None),
            (Some(heap), Some(name), None) => Self::new(heap, name, None, offset_width).map(Some),
            (Some(heap), Some(name), Some(Some(index))) => {
                Self::new(heap, name, Some(index), offset_width).map(Some)
            }
            _ => Err(LinkStorageError::StorageAddresses),
        }
    }

    /// Checks the defined stored addresses of a dense link heap and its indexes.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::UndefinedAddress`] for undefined addresses and
    /// [`LinkStorageError::Format`] for addresses that do not fit `offset_width`.
    fn new(
        heap: StoredAddress,
        name_index: StoredAddress,
        creation_index: Option<StoredAddress>,
        offset_width: OffsetWidth,
    ) -> Result<Self, LinkStorageError> {
        required_address(heap, offset_width)?;
        required_address(name_index, offset_width)?;
        if let Some(index) = creation_index {
            required_address(index, offset_width)?;
        }
        Ok(Self {
            heap,
            name_index,
            creation_index,
        })
    }
}

/// Reports link metadata that cannot be represented or combined faithfully.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkStorageError {
    /// A link's creation index is not below the group's next-index counter.
    CreationOrderCounter {
        /// The link's creation index.
        index: u64,
        /// The stored next-index counter.
        next: u64,
    },
    /// The counter exceeds the nonnegative signed range used for writing.
    CreationOrderCounterRange {
        /// The rejected counter.
        next: u64,
        /// The largest supported counter.
        maximum: u64,
    },
    /// Assigning another creation index would exceed the supported counter range.
    CreationOrderExhausted,
    /// Creation-index presence disagrees with the group's tracking policy.
    CreationOrderPresence,
    /// Two links have the same creation index.
    DuplicateCreationOrder(u64),
    /// Two links have the same name bytes.
    DuplicateName(Vec<u8>),
    /// An external value lacks its required version byte, separator, or terminator.
    ExternalValue,
    /// An external value declares unsupported version or flag bits.
    ExternalVersionFlags(u8),
    /// A message sets reserved flag bits.
    Flags(u8),
    /// Parsing or address encoding failed.
    Format(FormatError),
    /// Creation-order indexing is enabled without tracking.
    IndexingWithoutTracking,
    /// A name is empty, is `.`, or contains a null or slash.
    InvalidName(Vec<u8>),
    /// A symbolic target is empty or contains a null byte.
    InvalidTarget,
    /// A serialized length exceeds the host's addressable buffer size.
    LengthOverflow,
    /// A hard-target replacement was requested for a symbolic link.
    NotHardLink,
    /// The storage addresses disagree with each other or with the indexing policy.
    StorageAddresses,
    /// A symbolic target value exceeds its two-byte length field.
    TargetLength(usize),
    /// Bytes remain after the declared message fields.
    TrailingBytes,
    /// A required stored address is undefined at its encoded width.
    UndefinedAddress,
}

impl From<FormatError> for LinkStorageError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Checks that a stored address is defined and fits its encoded width.
///
/// # Errors
///
/// Returns [`LinkStorageError::UndefinedAddress`] for the width's undefined value and
/// [`LinkStorageError::Format`] for an address wider than `offset_width`.
fn required_address(
    address: StoredAddress,
    offset_width: OffsetWidth,
) -> Result<(), LinkStorageError> {
    if address.get() > width::offset_width_max(offset_width) {
        return Err(FormatError::AddressTooLarge {
            address: address.get(),
            offset_size: offset_width.get(),
        }
        .into());
    }
    if address.is_undefined(offset_width) {
        return Err(LinkStorageError::UndefinedAddress);
    }
    Ok(())
}

/// The largest counter accepted by `H5O__linfo_decode` in `H5Olinfo.c` (HDF5 1.14.6 and 2.2.0).
const MAX_LINK_CREATION_COUNTER: u64 = 0x7fff_ffff_ffff_ffff;
/// The ASCII name-character-set code from the Link message table cited by [`LINK_VERSION`].
const ASCII_CHARSET: u8 = 0;
/// The UTF-8 name-character-set code from the same table as [`ASCII_CHARSET`].
const UTF8_CHARSET: u8 = 1;
/// The Link message version defined in "The Link Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
const LINK_VERSION: u8 = 1;
/// The combined version-and-flags length from the Link message table cited by [`LINK_VERSION`].
const LINK_PREFIX_LEN: usize = 2;
/// Bit 2 of the Link flags, declaring creation order in the table cited by [`LINK_VERSION`].
const CREATION_ORDER_PRESENT: u8 = 0x04;
/// Bit 3 of the Link flags, declaring the link type in the table cited by [`LINK_VERSION`].
const LINK_TYPE_PRESENT: u8 = 0x08;
/// Bit 4 of the Link flags, declaring the character set in the table cited by [`LINK_VERSION`].
const CHARSET_PRESENT: u8 = 0x10;
/// Bits 5 through 7, reserved in the Link flags table cited by [`LINK_VERSION`].
const LINK_RESERVED_FLAGS: u8 = 0xe0;
/// Bits 2 through 7, reserved in version 0 Link Info flags.
///
/// Defined in "The Link Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
const LINK_INFO_RESERVED_FLAGS: u8 = 0xfc;
/// The eight-byte creation-index field from the Link message table cited by [`LINK_VERSION`].
const CREATION_ORDER_LEN: usize = 8;
/// The two-byte symbolic-target length field from the table cited by [`LINK_VERSION`].
const TARGET_LENGTH_LEN: usize = 2;
/// The external framing byte and two null terminators from the table cited by [`LINK_VERSION`].
const EXTERNAL_FRAMING_LEN: usize = 3;
/// The zero external version-and-flags byte from the table cited by [`LINK_VERSION`].
const EXTERNAL_VERSION_FLAGS: u8 = 0;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use rstest::rstest;

    use crate::address::StoredAddressExt;
    use crate::link_message::{LinkMessage, LinkTarget};

    use super::*;

    #[rstest]
    #[case::hard(OwnedLinkTarget::Hard { object_header_address: StoredAddress::new(512) })]
    #[case::soft(OwnedLinkTarget::Soft { target_path: b"../target".to_vec() })]
    #[case::external(OwnedLinkTarget::External { filename: b"other.h5".to_vec(), object_path: b"/target".to_vec() })]
    #[case::raw_soft(OwnedLinkTarget::Soft { target_path: vec![0xfe] })]
    #[case::raw_external(OwnedLinkTarget::External { filename: vec![0xfe], object_path: vec![b'/', 0xff] })]
    fn owned_links_round_trip_raw_names_targets_and_creation_indexes(
        #[case] target: OwnedLinkTarget,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
        #[values(CharacterSet::Ascii, CharacterSet::Utf8)] charset: CharacterSet,
        #[values(None, Some(19))] creation_order: Option<u64>,
    ) {
        let link = OwnedLink::new(
            vec![0xff, b'x'],
            charset.clone(),
            target.clone(),
            creation_order,
        )
        .unwrap();
        let encoded = link.serialize(width).unwrap();
        assert_eq!(encoded.len(), link.encoded_len(width).unwrap());
        let parsed = OwnedLink::parse(&encoded, width).unwrap();
        assert_eq!(parsed, link);
        assert_eq!(parsed.name(), &[0xff, b'x']);
        assert_eq!(parsed.charset(), charset);
        assert_eq!(parsed.target(), &target);
        assert_eq!(parsed.creation_order(), creation_order);
    }

    #[rstest]
    #[case::hard(LinkTarget::Hard { object_header_address: StoredAddress::new(512) })]
    #[case::soft(LinkTarget::Soft { target_path: "../软".into() })]
    #[case::external(LinkTarget::External { filename: "文件.h5".into(), object_path: "/target".into() })]
    fn owned_links_preserve_messages_written_by_the_existing_serializer(
        #[case] target: LinkTarget,
    ) {
        let original = LinkMessage {
            name: "链".into(),
            link_target: target,
            creation_order: Some(7),
            charset: CharacterSet::Utf8,
        };
        let bytes = original.serialize(OffsetWidth::Eight);
        let link = OwnedLink::parse(&bytes, OffsetWidth::Eight).unwrap();
        assert_eq!(link.serialize(OffsetWidth::Eight).unwrap(), bytes);
        assert_eq!(
            LinkMessage::parse_with_offset_width(&bytes, OffsetWidth::Eight).unwrap(),
            original
        );
    }

    #[rstest]
    #[case::one_byte(255)]
    #[case::two_bytes(256)]
    #[case::two_bytes_max(65_535)]
    #[case::four_bytes(65_536)]
    fn name_length_field_boundaries_preserve_the_full_name(#[case] length: usize) {
        let link = hard_link(vec![b'x'; length], None);
        let bytes = link.serialize(OffsetWidth::Eight).unwrap();
        assert_eq!(bytes.len(), link.encoded_len(OffsetWidth::Eight).unwrap());
        assert_eq!(OwnedLink::parse(&bytes, OffsetWidth::Eight).unwrap(), link);
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::null(vec![0])]
    #[case::slash(b"a/b".to_vec())]
    #[case::current(b".".to_vec())]
    fn invalid_component_names_are_rejected(#[case] name: Vec<u8>) {
        assert_eq!(
            OwnedLink::new(
                name.clone(),
                CharacterSet::Ascii,
                OwnedLinkTarget::Soft {
                    target_path: b"target".to_vec()
                },
                None
            )
            .unwrap_err(),
            LinkStorageError::InvalidName(name)
        );
    }

    #[rstest]
    #[case::empty(vec![], LinkStorageError::InvalidTarget)]
    #[case::null(vec![b'a', 0], LinkStorageError::InvalidTarget)]
    #[case::long(vec![b'x'; 65_536], LinkStorageError::TargetLength(65_536))]
    fn invalid_target_paths_are_rejected(
        #[case] target_path: Vec<u8>,
        #[case] error: LinkStorageError,
    ) {
        assert_eq!(
            OwnedLink::new(
                b"a".to_vec(),
                CharacterSet::Ascii,
                OwnedLinkTarget::Soft { target_path },
                None
            )
            .unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::soft(OwnedLinkTarget::Soft { target_path: vec![b'x'; 65_535] })]
    #[case::external(OwnedLinkTarget::External { filename: vec![b'x'; 65_531], object_path: vec![b'/'] })]
    fn the_maximum_target_value_length_round_trips(#[case] target: OwnedLinkTarget) {
        let link = OwnedLink::new(b"a".to_vec(), CharacterSet::Ascii, target, None).unwrap();
        let bytes = link.serialize(OffsetWidth::Eight).unwrap();
        assert_eq!(OwnedLink::parse(&bytes, OffsetWidth::Eight).unwrap(), link);
    }

    #[rstest]
    #[case::empty_filename(vec![], vec![b'/'], LinkStorageError::InvalidTarget)]
    #[case::empty_path(vec![b'f'], vec![], LinkStorageError::InvalidTarget)]
    #[case::embedded_null(vec![b'f', 0], vec![b'/'], LinkStorageError::InvalidTarget)]
    #[case::too_long(vec![b'f'; 65_532], vec![b'/'], LinkStorageError::TargetLength(65_536))]
    fn external_target_framing_is_bounded(
        #[case] filename: Vec<u8>,
        #[case] object_path: Vec<u8>,
        #[case] error: LinkStorageError,
    ) {
        assert_eq!(
            OwnedLink::new(
                b"a".to_vec(),
                CharacterSet::Ascii,
                OwnedLinkTarget::External {
                    filename,
                    object_path
                },
                None
            )
            .unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::version(vec![2, 0, 1, b'a', 0, 0], LinkStorageError::Format(FormatError::InvalidLinkVersion(2)))]
    #[case::reserved_flags(vec![1, 0x20, 1, b'a', 0, 0], LinkStorageError::Flags(0x20))]
    #[case::charset(vec![1, CHARSET_PRESENT, 2, 1, b'a', 0, 0], LinkStorageError::Format(FormatError::InvalidCharacterSet(2)))]
    #[case::user_type(vec![1, LINK_TYPE_PRESENT, 65, 1, b'a'], LinkStorageError::Format(FormatError::InvalidLinkType(65)))]
    #[case::undefined(vec![1, 0, 1, b'a', 0xff, 0xff], LinkStorageError::UndefinedAddress)]
    #[case::trailing(vec![1, 0, 1, b'a', 0, 0, 1], LinkStorageError::TrailingBytes)]
    #[case::external_version(vec![1, LINK_TYPE_PRESENT, EXTERNAL_LINK, 1, b'a', 5, 0, 0x10, b'f', 0, b'/', 0], LinkStorageError::ExternalVersionFlags(0x10))]
    #[case::external_flags(vec![1, LINK_TYPE_PRESENT, EXTERNAL_LINK, 1, b'a', 5, 0, 1, b'f', 0, b'/', 0], LinkStorageError::ExternalVersionFlags(1))]
    #[case::external_separator(vec![1, LINK_TYPE_PRESENT, EXTERNAL_LINK, 1, b'a', 4, 0, 0, b'f', b'/', 0], LinkStorageError::ExternalValue)]
    #[case::external_terminator(vec![1, LINK_TYPE_PRESENT, EXTERNAL_LINK, 1, b'a', 4, 0, 0, b'f', 0, b'/'], LinkStorageError::ExternalValue)]
    fn unsupported_or_unreproducible_messages_return_explicit_errors(
        #[case] bytes: Vec<u8>,
        #[case] error: LinkStorageError,
    ) {
        assert_eq!(
            OwnedLink::parse(&bytes, OffsetWidth::Two).unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::version(0, 2)]
    #[case::flags(1, 2)]
    #[case::name_length(2, 3)]
    #[case::name(3, 4)]
    #[case::target(4, 12)]
    #[case::partial_target(11, 12)]
    fn truncated_messages_report_the_required_field_extent(
        #[case] length: usize,
        #[case] expected: usize,
    ) {
        let bytes = hard_link(b"a".to_vec(), None)
            .serialize(OffsetWidth::Eight)
            .unwrap();
        assert_eq!(
            OwnedLink::parse(&bytes[..length], OffsetWidth::Eight).unwrap_err(),
            LinkStorageError::Format(FormatError::UnexpectedEof {
                expected,
                available: length
            })
        );
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    #[case::eight(OffsetWidth::Eight)]
    fn undefined_hard_targets_are_classified_at_the_encoded_width(#[case] width: OffsetWidth) {
        let mut bytes = vec![LINK_VERSION, 0, 1, b'a'];
        bytes.extend_from_slice(&vec![0xff; usize::from(width.get())]);
        assert_eq!(
            OwnedLink::parse(&bytes, width).unwrap_err(),
            LinkStorageError::UndefinedAddress
        );
    }

    #[test]
    fn hard_target_replacement_preserves_other_link_metadata_and_rejects_narrow_sentinels() {
        let mut link = hard_link(vec![0xff], Some(13));
        link.set_hard_target(StoredAddress::new(0xffff), OffsetWidth::Eight)
            .unwrap();
        let saved = link.clone();
        assert_eq!(
            link.set_hard_target(StoredAddress::new(0xffff), OffsetWidth::Two)
                .unwrap_err(),
            LinkStorageError::UndefinedAddress
        );
        assert_eq!(link, saved);
        assert_eq!(
            link.serialize(OffsetWidth::Two).unwrap_err(),
            LinkStorageError::UndefinedAddress
        );
        assert_eq!(link.name(), &[0xff]);
        assert_eq!(link.creation_order(), Some(13));
        assert_eq!(link.charset(), CharacterSet::Ascii);
        assert_eq!(
            link.target(),
            &OwnedLinkTarget::Hard {
                object_header_address: StoredAddress::new(0xffff)
            }
        );
        link.set_hard_target(StoredAddress::new(4096), OffsetWidth::Two)
            .unwrap();
        assert_eq!(
            OwnedLink::parse(&link.serialize(OffsetWidth::Two).unwrap(), OffsetWidth::Two).unwrap(),
            link
        );
    }

    #[test]
    fn target_replacement_and_sizing_reject_oversized_addresses() {
        let mut link = hard_link(b"a".to_vec(), None);
        let saved = link.clone();
        let error = LinkStorageError::Format(FormatError::AddressTooLarge {
            address: 0x1_0000,
            offset_size: 2,
        });
        assert_eq!(
            link.set_hard_target(StoredAddress::new(0x1_0000), OffsetWidth::Two)
                .unwrap_err(),
            error
        );
        assert_eq!(link, saved);
        link.set_hard_target(StoredAddress::new(0x1_0000), OffsetWidth::Eight)
            .unwrap();
        assert_eq!(link.encoded_len(OffsetWidth::Two).unwrap_err(), error);
    }

    #[test]
    fn a_symbolic_link_cannot_be_retargeted_as_a_hard_link() {
        let mut link = OwnedLink::new(
            b"a".to_vec(),
            CharacterSet::Ascii,
            OwnedLinkTarget::Soft {
                target_path: b"target".to_vec(),
            },
            None,
        )
        .unwrap();
        let original = link.clone();
        assert_eq!(
            link.set_hard_target(StoredAddress::new(512), OffsetWidth::Eight)
                .unwrap_err(),
            LinkStorageError::NotHardLink
        );
        assert_eq!(link, original);
    }

    #[rstest]
    #[case::untracked(LinkCreationPolicy::Untracked)]
    #[case::tracked(LinkCreationPolicy::Tracked { next: counter(11) })]
    #[case::indexed(LinkCreationPolicy::Indexed { next: counter(11) })]
    fn link_info_round_trips_supported_policy_and_storage_combinations(
        #[case] policy: LinkCreationPolicy,
        #[values(false, true)] dense: bool,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    ) {
        let original = info(policy, dense);
        let links = vec![hard_link(b"a".to_vec(), policy.next_index().map(|_| 3))];
        let bytes = original.serialize(width);
        let storage = GroupLinkStorage::parse(&bytes, links.clone(), width).unwrap();
        assert_eq!(storage.info(), original);
        assert_eq!(storage.links(), links);
        assert_eq!(storage.creation_order(), policy);
    }

    #[rstest]
    #[case::untracked(LinkCreationPolicy::Untracked, None)]
    #[case::tracked(LinkCreationPolicy::Tracked { next: counter(11) }, Some(11))]
    #[case::indexed(LinkCreationPolicy::Indexed { next: counter(11) }, Some(11))]
    fn deletions_preserve_gaps_and_new_links_use_the_stored_counter(
        #[case] policy: LinkCreationPolicy,
        #[case] assigned: Option<u64>,
    ) {
        let indexes = policy.next_index().map(|_| (2, 7));
        let mut storage = GroupLinkStorage::new(
            &info(policy, true),
            vec![
                hard_link(b"a".to_vec(), indexes.map(|x| x.0)),
                hard_link(b"b".to_vec(), indexes.map(|x| x.1)),
            ],
            OffsetWidth::Eight,
        )
        .unwrap();
        assert_eq!(
            storage.remove(b"b").unwrap().creation_order(),
            indexes.map(|x| x.1)
        );
        assert!(storage.remove(b"missing").is_none());
        storage.insert(hard_link(b"c".to_vec(), None)).unwrap();
        assert_eq!(storage.links()[0].creation_order(), indexes.map(|x| x.0));
        assert_eq!(storage.links()[1].creation_order(), assigned);
        assert_eq!(
            storage.creation_order().next_index(),
            policy.next_index().map(|index| index + 1)
        );
        storage.use_compact_storage();
        assert_eq!(
            storage.info().btree_creation_order_address,
            policy.is_indexed().then_some(None)
        );
        let compact = storage.info().serialize(OffsetWidth::Eight);
        let reparsed =
            GroupLinkStorage::parse(&compact, storage.links().to_vec(), OffsetWidth::Eight)
                .unwrap();
        assert_eq!(storage, reparsed);
        storage
            .use_dense_storage(
                StoredAddress::new(512),
                StoredAddress::new(1024),
                policy.is_indexed().then(|| StoredAddress::new(2048)),
            )
            .unwrap();
        assert_eq!(
            storage.info().max_creation_order,
            policy.next_index().map(|index| index + 1)
        );
    }

    #[rstest]
    #[case::tracked(LinkCreationPolicy::Tracked { next: counter(MAX_LINK_CREATION_COUNTER) })]
    #[case::indexed(LinkCreationPolicy::Indexed { next: counter(MAX_LINK_CREATION_COUNTER) })]
    fn an_exhausted_counter_rejects_assignment_without_mutation(
        #[case] mut policy: LinkCreationPolicy,
    ) {
        let original = policy;
        assert_eq!(
            policy.take().unwrap_err(),
            LinkStorageError::CreationOrderExhausted
        );
        assert_eq!(policy, original);
        let mut storage =
            GroupLinkStorage::new(&info(policy, false), vec![], OffsetWidth::Eight).unwrap();
        let original = storage.clone();
        assert_eq!(
            storage.insert(hard_link(b"a".to_vec(), None)).unwrap_err(),
            LinkStorageError::CreationOrderExhausted
        );
        assert_eq!(storage, original);
    }

    #[rstest]
    #[case::tracked(LinkCreationPolicy::Tracked { next: counter(MAX_LINK_CREATION_COUNTER - 1) })]
    #[case::indexed(LinkCreationPolicy::Indexed { next: counter(MAX_LINK_CREATION_COUNTER - 1) })]
    fn the_last_incrementable_counter_is_assigned(#[case] mut policy: LinkCreationPolicy) {
        assert_eq!(policy.take(), Ok(Some(MAX_LINK_CREATION_COUNTER - 1)));
        assert_eq!(policy.next_index(), Some(MAX_LINK_CREATION_COUNTER));
        assert_eq!(policy.take(), Err(LinkStorageError::CreationOrderExhausted));
    }

    #[rstest]
    #[case::sign_bit(MAX_LINK_CREATION_COUNTER + 1)]
    #[case::unsigned_last(u64::MAX - 1)]
    #[case::unsigned_max(u64::MAX)]
    fn out_of_range_counters_cannot_construct_writer_policy_or_storage(
        #[case] next: u64,
        #[values(false, true)] indexed: bool,
    ) {
        let error = LinkStorageError::CreationOrderCounterRange {
            next,
            maximum: MAX_LINK_CREATION_COUNTER,
        };
        assert_eq!(LinkCreationCounter::new(next).unwrap_err(), error);
        let message = LinkInfoMessage {
            max_creation_order: Some(next),
            btree_creation_order_address: indexed.then_some(None),
            ..info(LinkCreationPolicy::Untracked, false)
        };
        assert_eq!(LinkCreationPolicy::from_info(&message).unwrap_err(), error);
        assert_eq!(
            GroupLinkStorage::new(
                &message,
                vec![hard_link(b"a".to_vec(), Some(0))],
                OffsetWidth::Eight
            )
            .unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::missing_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![hard_link(b"a".to_vec(), None)], LinkStorageError::CreationOrderPresence)]
    #[case::unexpected_index(LinkCreationPolicy::Untracked, vec![hard_link(b"a".to_vec(), Some(0))], LinkStorageError::CreationOrderPresence)]
    #[case::counter_at_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![hard_link(b"a".to_vec(), Some(5))], LinkStorageError::CreationOrderCounter { index: 5, next: 5 })]
    #[case::counter_below_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![hard_link(b"a".to_vec(), Some(7))], LinkStorageError::CreationOrderCounter { index: 7, next: 5 })]
    #[case::duplicate_index(LinkCreationPolicy::Tracked { next: counter(5) }, vec![hard_link(b"a".to_vec(), Some(0)), hard_link(b"b".to_vec(), Some(0))], LinkStorageError::DuplicateCreationOrder(0))]
    #[case::duplicate_name(LinkCreationPolicy::Untracked, vec![hard_link(b"a".to_vec(), None), hard_link(b"a".to_vec(), None)], LinkStorageError::DuplicateName(b"a".to_vec()))]
    fn inconsistent_link_sets_are_rejected(
        #[case] policy: LinkCreationPolicy,
        #[case] links: Vec<OwnedLink>,
        #[case] error: LinkStorageError,
    ) {
        assert_eq!(
            GroupLinkStorage::new(&info(policy, false), links, OffsetWidth::Eight).unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::heap_only(LinkInfoMessage { fractal_heap_address: Some(StoredAddress::new(512)), ..info(LinkCreationPolicy::Untracked, false) }, LinkStorageError::StorageAddresses)]
    #[case::name_only(LinkInfoMessage { btree_name_index_address: Some(StoredAddress::new(512)), ..info(LinkCreationPolicy::Untracked, false) }, LinkStorageError::StorageAddresses)]
    #[case::index_without_tracking(LinkInfoMessage { btree_creation_order_address: Some(None), ..info(LinkCreationPolicy::Untracked, false) }, LinkStorageError::IndexingWithoutTracking)]
    #[case::missing_dense_index(LinkInfoMessage { btree_creation_order_address: Some(None), ..info(LinkCreationPolicy::Indexed { next: counter(0) }, true) }, LinkStorageError::StorageAddresses)]
    #[case::compact_with_index(LinkInfoMessage { btree_creation_order_address: Some(Some(StoredAddress::new(512))), ..info(LinkCreationPolicy::Indexed { next: counter(0) }, false) }, LinkStorageError::StorageAddresses)]
    fn inconsistent_storage_addresses_are_rejected(
        #[case] message: LinkInfoMessage,
        #[case] error: LinkStorageError,
    ) {
        assert_eq!(
            GroupLinkStorage::new(&message, vec![], OffsetWidth::Eight).unwrap_err(),
            error
        );
    }

    #[test]
    fn invalid_dense_replacement_preserves_the_existing_representation() {
        let mut storage = GroupLinkStorage::new(
            &info(LinkCreationPolicy::Indexed { next: counter(0) }, false),
            vec![],
            OffsetWidth::Two,
        )
        .unwrap();
        let original = storage.clone();
        assert_eq!(
            storage
                .use_dense_storage(StoredAddress::new(512), StoredAddress::new(1024), None,)
                .unwrap_err(),
            LinkStorageError::StorageAddresses
        );
        assert_eq!(storage, original);
        assert_eq!(
            storage
                .use_dense_storage(
                    StoredAddress::new(0xffff),
                    StoredAddress::new(1024),
                    Some(StoredAddress::new(2048)),
                )
                .unwrap_err(),
            LinkStorageError::UndefinedAddress
        );
        assert_eq!(storage, original);
        assert_eq!(
            storage
                .use_dense_storage(
                    StoredAddress::new(u64::MAX),
                    StoredAddress::new(1024),
                    Some(StoredAddress::new(2048)),
                )
                .unwrap_err(),
            LinkStorageError::Format(FormatError::AddressTooLarge {
                address: u64::MAX,
                offset_size: 2
            })
        );
        assert_eq!(storage, original);
    }

    #[test]
    fn duplicate_or_preindexed_insertions_preserve_the_link_set_and_counter() {
        let mut storage = GroupLinkStorage::new(
            &info(LinkCreationPolicy::Indexed { next: counter(1) }, false),
            vec![hard_link(b"a".to_vec(), Some(0))],
            OffsetWidth::Eight,
        )
        .unwrap();
        let original = storage.clone();
        assert_eq!(
            storage.insert(hard_link(b"a".to_vec(), None)).unwrap_err(),
            LinkStorageError::DuplicateName(b"a".to_vec())
        );
        assert_eq!(storage, original);
        assert_eq!(
            storage
                .insert(hard_link(b"b".to_vec(), Some(8)))
                .unwrap_err(),
            LinkStorageError::CreationOrderPresence
        );
        assert_eq!(storage, original);
    }

    #[test]
    fn group_retargeting_preserves_names_creation_indexes_and_storage_policy() {
        let original_info = info(LinkCreationPolicy::Indexed { next: counter(12) }, true);
        let mut storage = GroupLinkStorage::new(
            &original_info,
            vec![hard_link(b"a".to_vec(), Some(3))],
            OffsetWidth::Eight,
        )
        .unwrap();
        assert!(
            !storage
                .set_hard_target(b"missing", StoredAddress::new(4096))
                .unwrap()
        );
        assert!(
            storage
                .set_hard_target(b"a", StoredAddress::new(4096))
                .unwrap()
        );
        assert_eq!(storage.info(), original_info);
        assert_eq!(storage.links()[0].name(), b"a");
        assert_eq!(storage.links()[0].creation_order(), Some(3));
        assert_eq!(
            storage.links()[0].target(),
            &OwnedLinkTarget::Hard {
                object_header_address: StoredAddress::new(4096)
            }
        );
    }

    #[test]
    fn group_encoding_width_rejects_oversized_targets_before_insertion() {
        let mut link = hard_link(b"a".to_vec(), None);
        link.set_hard_target(StoredAddress::new(0x1_0000), OffsetWidth::Eight)
            .unwrap();
        let message = info(LinkCreationPolicy::Untracked, false);
        let error = LinkStorageError::Format(FormatError::AddressTooLarge {
            address: 0x1_0000,
            offset_size: 2,
        });
        assert_eq!(
            GroupLinkStorage::new(&message, vec![link.clone()], OffsetWidth::Two).unwrap_err(),
            error
        );
        let mut storage = GroupLinkStorage::new(&message, vec![], OffsetWidth::Two).unwrap();
        assert_eq!(storage.offset_width(), OffsetWidth::Two);
        assert_eq!(storage.insert(link).unwrap_err(), error);
        assert!(storage.links().is_empty());
    }

    #[test]
    fn reserved_link_info_flags_and_trailing_bytes_are_rejected() {
        let mut bytes = info(LinkCreationPolicy::Untracked, false).serialize(OffsetWidth::Eight);
        bytes[1] = 4;
        assert_eq!(
            GroupLinkStorage::parse(&bytes, vec![], OffsetWidth::Eight).unwrap_err(),
            LinkStorageError::Flags(4)
        );
        bytes[1] = 0;
        bytes.push(0);
        assert_eq!(
            GroupLinkStorage::parse(&bytes, vec![], OffsetWidth::Eight).unwrap_err(),
            LinkStorageError::TrailingBytes
        );
    }

    fn counter(next: u64) -> LinkCreationCounter {
        LinkCreationCounter::new(next).unwrap()
    }

    fn hard_link(name: Vec<u8>, creation_order: Option<u64>) -> OwnedLink {
        OwnedLink::new(
            name,
            CharacterSet::Ascii,
            OwnedLinkTarget::Hard {
                object_header_address: StoredAddress::new(512),
            },
            creation_order,
        )
        .unwrap()
    }

    fn info(policy: LinkCreationPolicy, dense: bool) -> LinkInfoMessage {
        LinkInfoMessage {
            max_creation_order: policy.next_index(),
            fractal_heap_address: dense.then(|| StoredAddress::new(512)),
            btree_name_index_address: dense.then(|| StoredAddress::new(1024)),
            btree_creation_order_address: policy
                .is_indexed()
                .then(|| dense.then(|| StoredAddress::new(2048))),
        }
    }
}
