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
use core::error::Error;
use core::fmt;

use super::creation_order::LinkCreationPolicy;
use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link::LinkMessage;
use crate::link::PreparedLink;
use crate::link::StoredLink;
use crate::link::encode;
use crate::link::info::{LinkCreationOrderIndex, LinkInfoMessage};
use crate::link::{
    LinkMessageV1, LinkName, LinkProjectionError, LinkTargetV1, LinkWireError, UserLinkKind,
};
use crate::link::{LinkMessageView, LinkTargetView};
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
        let link =
            LinkMessageView::parse_prefix(data, offset_width).map_err(LinkWireError::from)?;
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
        let target_value_len = encode::initial_value_len(target.as_view(explicit_type))?;
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
        Ok(PreparedLink::for_stored(self, offset_width)?)
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

impl StoredLink for OwnedLink {
    fn name_field(&self) -> (&[u8], UintWidth) {
        (self.name.as_bytes(), self.name.width())
    }
    fn target_field(&self) -> (LinkTargetView<'_>, u16) {
        (
            self.target.as_view(self.explicit_type),
            self.target_value_len,
        )
    }
    fn creation_index(&self) -> Option<u64> {
        self.creation_order
    }
    fn charset_field(&self) -> Option<CharacterSet> {
        self.explicit_charset.then(|| self.charset.clone())
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

/// An error reading or writing link metadata or setting a group's storage policy.
///
/// [`Error::source`] returns the wrapped [`FormatError`] for [`Self::Format`].
/// The other variants have no source. [`fmt::Display`] escapes the name bytes in
/// [`Self::DuplicateName`] and [`Self::InvalidName`].
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
    /// An underlying format error.
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
    /// A link target value exceeds its two-byte length field.
    TargetLength(usize),
    /// Bytes remain after the declared message fields.
    TrailingBytes,
    /// A required stored address is undefined at its encoded width.
    UndefinedAddress,
}

impl fmt::Display for LinkStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreationOrderCounter { index, next } => {
                write!(
                    f,
                    "link creation index {index} must be below counter {next}"
                )
            }
            Self::CreationOrderCounterRange { next, maximum } => {
                write!(
                    f,
                    "link creation order counter {next} exceeds maximum {maximum}"
                )
            }
            Self::CreationOrderExhausted => f.write_str("link creation order counter is exhausted"),
            Self::CreationOrderPresence => {
                f.write_str("link creation index presence disagrees with tracking policy")
            }
            Self::DuplicateCreationOrder(index) => {
                write!(f, "duplicate link creation index {index}")
            }
            Self::DuplicateName(name) => {
                write!(f, "duplicate link name b\"{}\"", name.escape_ascii())
            }
            Self::ExternalValue => {
                f.write_str("external link value lacks its version, separator, or terminator")
            }
            Self::ExternalVersionFlags(flags) => {
                write!(
                    f,
                    "unsupported external link version or flags: {flags:#04x}"
                )
            }
            Self::Flags(flags) => write!(f, "reserved link flag bits are set: {flags:#04x}"),
            Self::Format(error) => fmt::Display::fmt(error, f),
            Self::IndexingWithoutTracking => {
                f.write_str("link creation order indexing requires tracking")
            }
            Self::InvalidName(name) => write!(f, "invalid link name b\"{}\"", name.escape_ascii()),
            Self::InvalidTarget => {
                f.write_str("symbolic link target is empty or contains a null byte")
            }
            Self::LengthOverflow => {
                f.write_str("encoded link length exceeds the addressable buffer size")
            }
            Self::NotHardLink => f.write_str("replacing a hard link target requires a hard link"),
            Self::StorageAddresses => f.write_str(
                "link storage addresses disagree with each other or with the indexing policy",
            ),
            Self::TargetLength(length) => {
                write!(f, "link target length {length} exceeds its two-byte field")
            }
            Self::TrailingBytes => f.write_str("trailing bytes after link message fields"),
            Self::UndefinedAddress => f.write_str("required link address is undefined"),
        }
    }
}

impl Error for LinkStorageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Self::Format(error) = self {
            Some(error)
        } else {
            None
        }
    }
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
            LinkWireError::TrailingBytes => Self::TrailingBytes,
            LinkWireError::TargetLength(length) => Self::TargetLength(length),
            LinkWireError::Format(error) => match error {
                FormatError::InvalidLinkFlags(flags) => Self::Flags(flags),
                FormatError::InvalidLinkNameBytes(name) => Self::InvalidName(name),
                FormatError::DefinedAddressIsUndefined { .. } => Self::UndefinedAddress,
                FormatError::InvalidLinkTarget => Self::InvalidTarget,
                FormatError::InvalidExternalLinkValue => Self::ExternalValue,
                FormatError::InvalidExternalLinkVersionFlags(flags) => {
                    Self::ExternalVersionFlags(flags)
                }
                FormatError::LinkMessageLengthOverflow => Self::LengthOverflow,
                error => Self::Format(error),
            },
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

#[cfg(test)]
#[path = "tests/storage/mod.rs"]
mod tests;
