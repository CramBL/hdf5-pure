//! Owned group links, creation-order policy, and compact or dense storage addresses.
//!
//! Link names and symbolic targets retain their encoded bytes. The format specification,
//! version 4.0, defines ["The Link Message"][link] and ["The Link Info Message"][info].
//!
//! [link]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
//! [info]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_info::{LinkCreationOrderIndex, LinkInfoMessage};
use crate::link_message::LinkMessage;
use crate::link_message_read::{LinkMessageView, LinkTargetView};
use crate::link_message_v1::{
    self, LinkMessageV1, LinkName, LinkProjectionError, LinkTargetV1, LinkWireError, UserLinkKind,
};
use crate::link_message_write::{self, PreparedLink};
use crate::width::{self, OffsetWidth, UintWidth};

/// A group link set with distinct names and consistent creation-order and storage metadata.
///
/// Tracked links have distinct indexes below the stored next-index counter. Compact storage
/// retains the indexing policy with an undefined creation-order index address. Dense storage
/// has defined heap and name-index addresses, plus a creation-order index when indexing is enabled.
/// Link targets and storage addresses use the offset width retained by [`LinkInfoMessage`].
///
/// The storage fields are defined in "The Link Info Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
#[derive(Clone, Debug, PartialEq)]
pub struct GroupLinkStorage {
    links: Vec<OwnedLink>,
    creation_order: LinkCreationPolicy,
    addresses: LinkInfoMessage,
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
        if info.encoded_size() != data.len() {
            return Err(LinkStorageError::TrailingBytes);
        }
        Self::new(&info, links)
    }

    /// Combines Link Info metadata with links that can be encoded at its offset width.
    ///
    /// # Errors
    ///
    /// Propagates the metadata errors of [`Self::from_info`] and the link errors of
    /// [`Self::with_links`].
    pub fn new(info: &LinkInfoMessage, links: Vec<OwnedLink>) -> Result<Self, LinkStorageError> {
        Self::from_info(info)?.with_links(links)
    }

    /// Creates an empty group link set with the policy and addresses from `info`.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::StorageAddresses`] if the addresses disagree with each
    /// other or with the indexing policy, and the policy errors of [`LinkCreationPolicy::from_info`].
    pub fn from_info(info: &LinkInfoMessage) -> Result<Self, LinkStorageError> {
        let creation_order = LinkCreationPolicy::from_info(info)?;
        let mut addresses = match (
            info.fractal_heap_address(),
            info.btree_name_index_address(),
            info.creation_order_index(),
        ) {
            (None, None, LinkCreationOrderIndex::Absent | LinkCreationOrderIndex::Undefined)
            | (
                Some(_),
                Some(_),
                LinkCreationOrderIndex::Absent | LinkCreationOrderIndex::Defined(_),
            ) => *info,
            _ => return Err(LinkStorageError::StorageAddresses),
        };
        addresses.set_max_creation_order(None);
        Ok(Self {
            links: Vec::new(),
            creation_order,
            addresses,
        })
    }

    /// Replaces the links after checking their encoding, names, and creation indexes.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OwnedLink::encoded_len`] for a link that cannot be encoded,
    /// [`LinkStorageError::DuplicateName`] or [`LinkStorageError::DuplicateCreationOrder`]
    /// for repeated names or indexes, [`LinkStorageError::CreationOrderPresence`] if indexes
    /// disagree with tracking, or [`LinkStorageError::CreationOrderCounter`] if an index
    /// reaches or exceeds the next-index counter.
    pub fn with_links(mut self, links: Vec<OwnedLink>) -> Result<Self, LinkStorageError> {
        let creation_order = self.creation_order;
        let offset_width = self.offset_width();
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
        self.links = links;
        Ok(self)
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
            links,
            creation_order,
            addresses,
        } = self;
        if link.creation_order.is_some() {
            return Err(LinkStorageError::CreationOrderPresence);
        }
        if links.iter().any(|existing| existing.name() == link.name()) {
            return Err(LinkStorageError::DuplicateName(link.name.into_bytes()));
        }
        link.encoded_len(addresses.offset_width())?;
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
        let offset_width = self.offset_width();
        let Some(link) = self.links.iter_mut().find(|link| link.name() == name) else {
            return Ok(false);
        };
        link.set_hard_target(address, offset_width)?;
        Ok(true)
    }

    /// Clears dense-storage addresses while preserving the creation-order policy.
    pub fn use_compact_storage(&mut self) {
        self.addresses =
            LinkInfoMessage::compact(self.offset_width(), None, self.creation_order.is_indexed());
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
        self.addresses = LinkInfoMessage::new(
            self.offset_width(),
            None,
            Some(heap),
            Some(name_index),
            creation_index.map_or(
                LinkCreationOrderIndex::Absent,
                LinkCreationOrderIndex::Defined,
            ),
        )
        .map_err(|error| match error {
            FormatError::DefinedAddressIsUndefined { .. } => LinkStorageError::UndefinedAddress,
            error => LinkStorageError::Format(error),
        })?;
        Ok(())
    }

    /// Consumes the storage description and returns its owned link records.
    pub fn into_links(self) -> Vec<OwnedLink> {
        self.links
    }

    /// Returns the file-address width used to check link targets and storage addresses.
    pub fn offset_width(&self) -> OffsetWidth {
        self.addresses.offset_width()
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
        let mut info = self.addresses;
        info.set_max_creation_order(self.creation_order.next_index());
        info
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
            .max_creation_order()
            .map(LinkCreationCounter::new)
            .transpose()?;
        match (next, info.indexes_creation_order()) {
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

/// An owned version 1 Link message with raw bytes and retained encoding choices.
///
/// Parsing or converting from [`LinkMessageV1`] or [`LinkMessageView`] retains the name-length
/// width and explicit hard-kind and ASCII fields. Character-set markers are retained independently
/// of the name bytes. The name is checked as a link component, and targets are checked for syntax
/// and encoded lengths without decoding their character data.
/// Creation indexes are checked against a group's policy by [`GroupLinkStorage::new`].
/// The fields are defined in "The Link Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedLink {
    creation_order: Option<u64>,
    charset: CharacterSet,
    name: LinkName,
    target: OwnedLinkTarget,
    target_value_len: u16,
    explicit_type: bool,
    explicit_charset: bool,
}

impl OwnedLink {
    /// Parses a version 1 Link message while preserving name and symbolic-target bytes.
    ///
    /// Supports hard, soft, and version 0 external links with zero external flags.
    /// User-defined values are decoded completely before the kind is rejected. Convert a parsed
    /// [`LinkMessageView`] or [`LinkMessageV1`] with [`Self::try_from`] to distinguish an
    /// unsupported kind from a decoding error.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::Flags`] for reserved flags,
    /// [`LinkStorageError::TrailingBytes`] for bytes after the target,
    /// [`LinkStorageError::ExternalVersionFlags`] for unsupported external versions or flags,
    /// and [`LinkStorageError::ExternalValue`] for incomplete external framing.
    /// Returns [`LinkStorageError::UndefinedAddress`] for an undefined hard target and the
    /// errors of [`Self::new`] for invalid names or targets. Returns
    /// [`LinkStorageError::Format`] for malformed prefix fields or truncated fields, and
    /// [`FormatError::InvalidLinkType`] wrapped in [`LinkStorageError::Format`] for reserved
    /// or user-defined kinds.
    pub fn parse(data: &[u8], offset_width: OffsetWidth) -> Result<Self, LinkStorageError> {
        let link = LinkMessageView::parse(data, offset_width).map_err(LinkStorageError::from)?;
        Self::try_from(&link).map_err(LinkStorageError::from)
    }

    /// Parses the version 1 Link message at the start of `data` and returns it with its length.
    ///
    /// The bytes past the link are left unread, such as the padding after a message in a version 1
    /// object header.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::parse`] other than [`LinkStorageError::TrailingBytes`] for
    /// bytes after the target.
    pub fn parse_padded(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<(Self, usize), LinkStorageError> {
        let link = LinkMessageView::parse_prefix(data, offset_width)
            .map_err(link_message_v1::decoded_link_error)?;
        Ok((
            Self::try_from(&link).map_err(LinkStorageError::from)?,
            link.consumed(),
        ))
    }

    /// Creates an owned link from encoded name and target bytes and an optional creation index.
    ///
    /// Selects the smallest name-length width. The writer omits the hard-kind and ASCII fields.
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
        let name = LinkName::new(name).map_err(LinkStorageError::from)?;
        let explicit_type = !matches!(target, OwnedLinkTarget::Hard { .. });
        let target_value_len =
            link_message_write::initial_value_len(target.as_view(explicit_type))?;
        let explicit_charset = charset != CharacterSet::Ascii;
        Ok(Self {
            creation_order,
            charset,
            name,
            target,
            target_value_len,
            explicit_type,
            explicit_charset,
        })
    }

    /// Copies a text link's name and target using the canonical choices of [`LinkMessage::prepare`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`LinkMessage::prepare`].
    pub fn from_text(link: &LinkMessage, offset_width: OffsetWidth) -> Result<Self, FormatError> {
        let prepared = link.prepare(offset_width)?;
        let (target, explicit_type) = OwnedLinkTarget::copy_from_view(prepared.target())
            .map_err(|kind| FormatError::InvalidLinkType(kind.get()))?;
        Ok(Self::from_fields(
            prepared.creation_order(),
            prepared.charset(),
            LinkName::from_prepared(&prepared),
            target,
            prepared.target_value_len(),
            explicit_type,
        ))
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
        self.name.as_bytes()
    }

    /// Returns the hard address or symbolic-target bytes.
    pub fn target(&self) -> &OwnedLinkTarget {
        &self.target
    }

    /// Returns the complete serialized Link message length at `offset_width`.
    ///
    /// Includes the retained name-length width and optional fields.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::UndefinedAddress`] for an undefined hard target,
    /// [`LinkStorageError::Format`] for a hard target that does not fit `offset_width`, and
    /// [`LinkStorageError::LengthOverflow`] if the complete length exceeds [`usize::MAX`].
    pub fn encoded_len(&self, offset_width: OffsetWidth) -> Result<usize, LinkStorageError> {
        Ok(self.prepare(offset_width)?.encoded_size())
    }

    /// Encodes a Link message with its retained name-length width and optional fields.
    ///
    /// Preserves name bytes, target bytes, character-set declaration, and creation index.
    /// The returned bytes exclude object-header padding.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::encoded_len`] if the message cannot be encoded at
    /// `offset_width`.
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, LinkStorageError> {
        Ok(self.prepare(offset_width)?.to_bytes())
    }

    /// Prepares a body with borrowed name and target bytes and retained field choices.
    ///
    /// # Errors
    ///
    /// Returns the address and body length errors of [`Self::encoded_len`].
    pub fn prepare(&self, offset_width: OffsetWidth) -> Result<PreparedLink<'_>, LinkStorageError> {
        PreparedLink::for_owned(self, offset_width)
    }

    pub(crate) fn name_width(&self) -> UintWidth {
        self.name.width()
    }

    pub(crate) fn stored_charset(&self) -> Option<CharacterSet> {
        self.explicit_charset.then(|| self.charset.clone())
    }

    pub(crate) fn target_view(&self) -> LinkTargetView<'_> {
        self.target.as_view(self.explicit_type)
    }

    pub(crate) fn target_value_len(&self) -> u16 {
        self.target_value_len
    }

    fn from_fields(
        creation_order: Option<u64>,
        charset: Option<CharacterSet>,
        name: LinkName,
        target: OwnedLinkTarget,
        target_value_len: u16,
        explicit_type: bool,
    ) -> Self {
        Self {
            creation_order,
            explicit_type,
            explicit_charset: charset.is_some(),
            charset: charset.unwrap_or(CharacterSet::Ascii),
            name,
            target_value_len,
            target,
        }
    }
}

impl TryFrom<LinkMessageV1> for OwnedLink {
    type Error = LinkProjectionError;

    /// Converts a checked message while retaining its raw bytes and encoding choices.
    ///
    /// Preserves the name-length width, optional-field presence, declared character set, and
    /// creation index. Name and target bytes are not decoded as text.
    ///
    /// # Errors
    ///
    /// Returns [`LinkProjectionError::UnsupportedKind`] for a user-defined link.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::__private::{
    ///     LinkMessageV1, LinkProjectionError, OffsetWidth, OwnedLink, UserLinkKind,
    /// };
    ///
    /// let bytes = b"\x01\x01\x01\x00x\x34\x12";
    /// let wire = LinkMessageV1::parse(bytes, OffsetWidth::Two).unwrap();
    /// let owned = OwnedLink::try_from(wire).unwrap();
    /// assert_eq!(owned.serialize(OffsetWidth::Two).unwrap(), bytes);
    ///
    /// let wire = LinkMessageV1::parse(b"\x01\x08\xff\x01u\x00\x00", OffsetWidth::Two).unwrap();
    /// assert_eq!(
    ///     OwnedLink::try_from(wire).unwrap_err(),
    ///     LinkProjectionError::UnsupportedKind(UserLinkKind::new(255).unwrap()),
    /// );
    /// ```
    fn try_from(link: LinkMessageV1) -> Result<Self, Self::Error> {
        let (target, creation_order, charset, name, target_value_len) = link.into_parts();
        let explicit_type = match &target {
            LinkTargetV1::Hard { explicit_type, .. } => *explicit_type,
            _ => true,
        };
        let target = match target {
            LinkTargetV1::Hard {
                object_header_address,
                explicit_type: _,
            } => OwnedLinkTarget::Hard {
                object_header_address,
            },
            LinkTargetV1::Soft { target_path } => OwnedLinkTarget::soft(target_path),
            LinkTargetV1::External {
                filename,
                object_path,
            } => OwnedLinkTarget::external(filename, object_path),
            LinkTargetV1::UserDefined { kind, value: _ } => {
                return Err(LinkProjectionError::UnsupportedKind(kind));
            }
        };
        Ok(Self::from_fields(
            creation_order,
            charset,
            name,
            target,
            target_value_len,
            explicit_type,
        ))
    }
}

impl TryFrom<&LinkMessageView<'_>> for OwnedLink {
    type Error = LinkProjectionError;

    /// Copies the checked fields while retaining the raw bytes and encoding choices.
    ///
    /// # Errors
    ///
    /// Returns [`LinkProjectionError::UnsupportedKind`] for a user-defined link.
    fn try_from(link: &LinkMessageView<'_>) -> Result<Self, Self::Error> {
        let (target, explicit_type) = OwnedLinkTarget::copy_from_view(link.target())
            .map_err(LinkProjectionError::UnsupportedKind)?;
        Ok(Self::from_fields(
            link.creation_order(),
            link.charset(),
            LinkName::from_view(link),
            target,
            link.target_value_len(),
            explicit_type,
        ))
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
    /// Refers to a symbolic path.
    Indirect(Box<OwnedSymbolicLinkTarget>),
}

impl OwnedLinkTarget {
    /// Creates a soft link pointing to `target_path`.
    pub fn soft(target_path: Vec<u8>) -> Self {
        Self::Indirect(Box::new(OwnedSymbolicLinkTarget::Soft { target_path }))
    }

    /// Creates an external link pointing to `object_path` in `filename`.
    pub fn external(filename: Vec<u8>, object_path: Vec<u8>) -> Self {
        Self::Indirect(Box::new(OwnedSymbolicLinkTarget::External {
            filename,
            object_path,
        }))
    }

    fn copy_from_view(target: LinkTargetView<'_>) -> Result<(Self, bool), UserLinkKind> {
        Ok(match target {
            LinkTargetView::Hard {
                object_header_address,
                explicit_type,
            } => (
                Self::Hard {
                    object_header_address,
                },
                explicit_type,
            ),
            LinkTargetView::Soft { target_path } => (Self::soft(target_path.to_vec()), true),
            LinkTargetView::External {
                filename,
                object_path,
            } => (
                Self::external(filename.to_vec(), object_path.to_vec()),
                true,
            ),
            LinkTargetView::UserDefined { kind, .. } => return Err(kind),
        })
    }

    fn as_view(&self, explicit_type: bool) -> LinkTargetView<'_> {
        match self {
            Self::Hard {
                object_header_address,
            } => LinkTargetView::Hard {
                object_header_address: *object_header_address,
                explicit_type,
            },
            Self::Indirect(target) => match target.as_ref() {
                OwnedSymbolicLinkTarget::Soft { target_path } => {
                    LinkTargetView::Soft { target_path }
                }
                OwnedSymbolicLinkTarget::External {
                    filename,
                    object_path,
                } => LinkTargetView::External {
                    filename,
                    object_path,
                },
            },
        }
    }
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<OwnedLinkTarget>() == 16);

/// The undecoded target bytes of an [`OwnedLinkTarget::Indirect`] link.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OwnedSymbolicLinkTarget {
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
        match error {
            FormatError::InvalidLinkInfoFlags(flags) => Self::Flags(flags),
            error => Self::Format(error),
        }
    }
}

impl From<LinkWireError> for LinkStorageError {
    fn from(error: LinkWireError) -> Self {
        match error {
            LinkWireError::NameWidth { length, width } => {
                Self::Format(FormatError::LengthTooLarge {
                    length: length.to_u64(),
                    length_size: width.get(),
                })
            }
            LinkWireError::Storage(error) => error,
        }
    }
}

impl From<LinkProjectionError> for LinkStorageError {
    fn from(error: LinkProjectionError) -> Self {
        match error {
            LinkProjectionError::Format(error) => Self::Format(error),
            LinkProjectionError::UnsupportedKind(kind) => {
                Self::Format(FormatError::InvalidLinkType(kind.get()))
            }
        }
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
#[cfg(test)]
mod tests {
    use alloc::vec;

    use rstest::rstest;

    use crate::address::StoredAddressExt;
    use crate::link_message::{LinkMessage, LinkTarget};

    use super::*;

    #[rstest]
    #[case::hard(
        b"\x01\x00\x01l\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        OwnedLinkTarget::Hard { object_header_address: StoredAddress::new(512) },
        12,
    )]
    #[case::soft(
        b"\x01\x08\x01\x01l\x04\x00../x\x00\x00\x00\x00\x00\x00",
        OwnedLinkTarget::soft(b"../x".to_vec()),
        11,
    )]
    fn a_padded_parse_reads_the_link_before_the_padding(
        #[case] bytes: &[u8],
        #[case] target: OwnedLinkTarget,
        #[case] len: usize,
    ) {
        let expected = OwnedLink::new(b"l".to_vec(), CharacterSet::Ascii, target, None).unwrap();
        assert_eq!(
            OwnedLink::parse_padded(bytes, OffsetWidth::Eight),
            Ok((expected, len))
        );
        assert_eq!(
            OwnedLink::parse(bytes, OffsetWidth::Eight),
            Err(LinkStorageError::TrailingBytes)
        );
    }

    #[rstest]
    #[case::hard(
        b"\x01\x00\x01l\x00\x02\x00\x00\x00\x00\x00\x00",
        OwnedLinkTarget::Hard { object_header_address: StoredAddress::new(512) },
        LinkTarget::Hard { object_header_address: StoredAddress::new(512) },
    )]
    #[case::soft(
        b"\x01\x08\x01\x01l\x04\x00../x",
        OwnedLinkTarget::soft(b"../x".to_vec()),
        LinkTarget::soft("../x".into()),
    )]
    #[case::external(
        b"\x01\x08\x40\x01l\x09\x00\x00f.h5\x00/x\x00",
        OwnedLinkTarget::external(b"f.h5".to_vec(), b"/x".to_vec()),
        LinkTarget::external("f.h5".into(), "/x".into()),
    )]
    fn link_targets_preserve_independent_message_bytes(
        #[case] bytes: &[u8],
        #[case] owned_target: OwnedLinkTarget,
        #[case] target: LinkTarget,
    ) {
        let owned = OwnedLink::new(b"l".to_vec(), CharacterSet::Ascii, owned_target, None).unwrap();
        let message = LinkMessage {
            name: "l".into(),
            link_target: target,
            creation_order: None,
            charset: CharacterSet::Ascii,
        };
        assert_eq!(OwnedLink::parse(bytes, OffsetWidth::Eight).unwrap(), owned);
        assert_eq!(LinkMessage::parse(bytes, 8).unwrap(), message);
        assert_eq!(owned.encoded_len(OffsetWidth::Eight).unwrap(), bytes.len());
        assert_eq!(owned.clone().serialize(OffsetWidth::Eight).unwrap(), bytes);
        assert_eq!(
            message.clone().serialize(OffsetWidth::Eight).unwrap(),
            bytes
        );
    }

    #[rstest]
    #[case::hard(OwnedLinkTarget::Hard { object_header_address: StoredAddress::new(512) })]
    #[case::soft(OwnedLinkTarget::soft(b"../target".to_vec()))]
    #[case::external(OwnedLinkTarget::external(b"other.h5".to_vec(), b"/target".to_vec()))]
    #[case::raw_soft(OwnedLinkTarget::soft(vec![0xfe]))]
    #[case::raw_external(OwnedLinkTarget::external(vec![0xfe], vec![b'/', 0xff]))]
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
    #[case::soft(LinkTarget::soft("../软".into()))]
    #[case::external(LinkTarget::external("文件.h5".into(), "/target".into()))]
    fn owned_links_preserve_messages_written_by_the_existing_serializer(
        #[case] target: LinkTarget,
    ) {
        let original = LinkMessage {
            name: "链".into(),
            link_target: target,
            creation_order: Some(7),
            charset: CharacterSet::Utf8,
        };
        let bytes = original.serialize(OffsetWidth::Eight).unwrap();
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
                OwnedLinkTarget::soft(b"target".to_vec()),
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
                OwnedLinkTarget::soft(target_path),
                None
            )
            .unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::soft(OwnedLinkTarget::soft(vec![b'x'; 65_535]))]
    #[case::external(OwnedLinkTarget::external(vec![b'x'; 65_531], vec![b'/']))]
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
                OwnedLinkTarget::external(filename, object_path),
                None
            )
            .unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::version(vec![2, 0, 1, b'a', 0, 0], LinkStorageError::Format(FormatError::InvalidLinkVersion(2)))]
    #[case::reserved_flags(vec![1, 0x20, 1, b'a', 0, 0], LinkStorageError::Flags(0x20))]
    #[case::charset(vec![1, 0x10, 2, 1, b'a', 0, 0], LinkStorageError::Format(FormatError::InvalidCharacterSet(2)))]
    #[case::missing_user_value_length(vec![1, 0x08, 65, 1, b'a'], LinkStorageError::Format(FormatError::UnexpectedEof { expected: 7, available: 5 }))]
    #[case::undefined(vec![1, 0, 1, b'a', 0xff, 0xff], LinkStorageError::UndefinedAddress)]
    #[case::trailing(vec![1, 0, 1, b'a', 0, 0, 1], LinkStorageError::TrailingBytes)]
    #[case::external_version(vec![1, 0x08, 0x40, 1, b'a', 5, 0, 0x10, b'f', 0, b'/', 0], LinkStorageError::ExternalVersionFlags(0x10))]
    #[case::external_flags(vec![1, 0x08, 0x40, 1, b'a', 5, 0, 1, b'f', 0, b'/', 0], LinkStorageError::ExternalVersionFlags(1))]
    #[case::external_separator(vec![1, 0x08, 0x40, 1, b'a', 4, 0, 0, b'f', b'/', 0], LinkStorageError::ExternalValue)]
    #[case::external_terminator(vec![1, 0x08, 0x40, 1, b'a', 4, 0, 0, b'f', 0, b'/'], LinkStorageError::ExternalValue)]
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
        let mut bytes = vec![1, 0, 1, b'a'];
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
            OwnedLinkTarget::soft(b"target".to_vec()),
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
        let original = info_at(width, policy, dense);
        let links = vec![hard_link(b"a".to_vec(), policy.next_index().map(|_| 3))];
        let bytes = original.to_bytes();
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
            storage.info().creation_order_index(),
            if policy.is_indexed() {
                LinkCreationOrderIndex::Undefined
            } else {
                LinkCreationOrderIndex::Absent
            }
        );
        let compact = storage.info().to_bytes();
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
            storage.info().max_creation_order(),
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
        let mut storage = GroupLinkStorage::new(&info(policy, false), vec![]).unwrap();
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
        let message = LinkInfoMessage::compact(OffsetWidth::Eight, Some(next), indexed);
        assert_eq!(LinkCreationPolicy::from_info(&message).unwrap_err(), error);
        assert_eq!(
            GroupLinkStorage::new(&message, vec![hard_link(b"a".to_vec(), Some(0))]).unwrap_err(),
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
            GroupLinkStorage::new(&info(policy, false), links).unwrap_err(),
            error
        );
    }

    #[rstest]
    #[case::heap_only(LinkInfoMessage::new(OffsetWidth::Eight, None, Some(StoredAddress::new(512)), None, LinkCreationOrderIndex::Absent).unwrap(), LinkStorageError::StorageAddresses)]
    #[case::name_only(LinkInfoMessage::new(OffsetWidth::Eight, None, None, Some(StoredAddress::new(512)), LinkCreationOrderIndex::Absent).unwrap(), LinkStorageError::StorageAddresses)]
    #[case::index_without_tracking(
        LinkInfoMessage::compact(OffsetWidth::Eight, None, true),
        LinkStorageError::IndexingWithoutTracking
    )]
    #[case::missing_dense_index(LinkInfoMessage::new(OffsetWidth::Eight, Some(0), Some(StoredAddress::new(512)), Some(StoredAddress::new(1024)), LinkCreationOrderIndex::Undefined).unwrap(), LinkStorageError::StorageAddresses)]
    #[case::compact_with_index(LinkInfoMessage::new(OffsetWidth::Eight, Some(0), None, None, LinkCreationOrderIndex::Defined(StoredAddress::new(512))).unwrap(), LinkStorageError::StorageAddresses)]
    fn inconsistent_storage_addresses_are_rejected(
        #[case] message: LinkInfoMessage,
        #[case] error: LinkStorageError,
    ) {
        assert_eq!(GroupLinkStorage::new(&message, vec![]).unwrap_err(), error);
    }

    #[test]
    fn invalid_dense_replacement_preserves_the_existing_representation() {
        let mut storage = GroupLinkStorage::new(
            &info_at(
                OffsetWidth::Two,
                LinkCreationPolicy::Indexed { next: counter(0) },
                false,
            ),
            vec![],
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
        let mut storage =
            GroupLinkStorage::new(&original_info, vec![hard_link(b"a".to_vec(), Some(3))]).unwrap();
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
        let message = LinkInfoMessage::compact(OffsetWidth::Two, None, false);
        let error = LinkStorageError::Format(FormatError::AddressTooLarge {
            address: 0x1_0000,
            offset_size: 2,
        });
        assert_eq!(
            GroupLinkStorage::new(&message, vec![link.clone()]).unwrap_err(),
            error
        );
        let mut storage = GroupLinkStorage::new(&message, vec![]).unwrap();
        assert_eq!(storage.offset_width(), OffsetWidth::Two);
        assert_eq!(storage.insert(link).unwrap_err(), error);
        assert!(storage.links().is_empty());
    }

    #[test]
    fn reserved_link_info_flags_and_trailing_bytes_are_rejected() {
        let mut bytes = info(LinkCreationPolicy::Untracked, false).to_bytes();
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
        info_at(OffsetWidth::Eight, policy, dense)
    }

    fn info_at(width: OffsetWidth, policy: LinkCreationPolicy, dense: bool) -> LinkInfoMessage {
        if !dense {
            return LinkInfoMessage::compact(width, policy.next_index(), policy.is_indexed());
        }
        LinkInfoMessage::new(
            width,
            policy.next_index(),
            Some(StoredAddress::new(512)),
            Some(StoredAddress::new(1024)),
            if policy.is_indexed() {
                LinkCreationOrderIndex::Defined(StoredAddress::new(2048))
            } else {
                LinkCreationOrderIndex::Absent
            },
        )
        .unwrap()
    }
}
