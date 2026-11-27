//! Version 1 Link messages with raw names, target bytes, and retained field widths.

use alloc::vec::Vec;

use super::encode::{self, PreparedLink};
use super::error::LinkWireError;
use super::name::{LinkName, LinkTargetV1};
use super::source::StoredLink;
use super::view::{LinkMessageView, LinkTargetView};
use crate::codec::{DecodeFrame, DecodePrefix};
use crate::datatype::CharacterSet;
use crate::width::{OffsetWidth, UintWidth};

/// A version 1 Link message with checked fields and retained encoding choices.
///
/// Names and symbolic targets are stored as bytes. The declared character set is retained
/// independently of those bytes, including an explicit ASCII declaration. The name-length width,
/// explicit hard-link type field, and creation-order field are also retained.
///
/// Group-level creation-order policies and application-defined traversal are the caller's
/// responsibility.
///
/// The fields are defined in "The Link Message" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::{LinkMessageV1, LinkWireError, OffsetWidth, UintWidth};
///
/// let bytes = b"\x01\x09\xff\x01\x00u\x02\x00\x00\xff";
/// let link = LinkMessageV1::parse(bytes, OffsetWidth::Two)?;
/// assert_eq!(link.name().width(), UintWidth::Two);
/// assert_eq!(link.serialize(OffsetWidth::Two)?, bytes);
/// # Ok::<(), LinkWireError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Clone, Debug, PartialEq)]
pub struct LinkMessageV1 {
    target: LinkTargetV1,
    target_value_len: u16,
    creation_order: Option<u64>,
    charset: Option<CharacterSet>,
    name: LinkName,
}

impl LinkMessageV1 {
    /// Parses one complete message, rejecting bytes after its target.
    ///
    /// Copies the name and target bytes after checking the complete body.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::parse_prefix`], or [`LinkWireError::TrailingBytes`] if bytes
    /// follow the target.
    pub fn parse(data: &[u8], offset_width: OffsetWidth) -> Result<Self, LinkWireError> {
        Ok(Self::from_view(&LinkMessageView::decode_frame(
            data,
            offset_width,
        )?))
    }

    /// Parses the message at the start of `data` and returns it with the number of bytes read.
    ///
    /// Copies the name and target after checking the message. The count includes all fields
    /// through the target and excludes any bytes that follow it.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Format`] for the decoding errors of
    /// [`LinkMessageView::parse_prefix`].
    pub fn parse_prefix(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<(Self, usize), LinkWireError> {
        Ok(LinkMessageView::decode_prefix(data, offset_width)?
            .map(|view| Self::from_view(&view))
            .into_parts())
    }

    /// Creates a message from checked name bytes and a target.
    ///
    /// With `None`, the writer omits the character-set field and declares ASCII. With `Some`,
    /// the writer stores the field, including for [`CharacterSet::Ascii`]. Name and target bytes
    /// are not decoded as text.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Format`] if a hard-link address is undefined at eight bytes,
    /// a symbolic path is empty or contains NUL, or the external value length overflows.
    /// Returns [`LinkWireError::TargetLength`] if the target value exceeds [`u16::MAX`] bytes.
    /// An external value includes its version byte and two terminating NUL bytes in this limit.
    /// A hard-link address is checked against the supplied width during encoding.
    pub fn new(
        name: LinkName,
        target: LinkTargetV1,
        creation_order: Option<u64>,
        charset: Option<CharacterSet>,
    ) -> Result<Self, LinkWireError> {
        let target_value_len = encode::initial_value_len(target.as_view())?;
        Ok(Self {
            target,
            target_value_len,
            creation_order,
            charset,
            name,
        })
    }

    /// Returns the stored target, including explicit hard-link type-field presence.
    pub fn target(&self) -> &LinkTargetV1 {
        &self.target
    }

    /// Returns the stored creation index, or `None` if the field is absent.
    pub fn creation_order(&self) -> Option<u64> {
        self.creation_order
    }

    /// Returns the explicit character-set declaration, or `None` for omitted ASCII.
    pub fn charset(&self) -> Option<CharacterSet> {
        self.charset.clone()
    }

    /// Returns the raw name and its retained length-field width.
    pub fn name(&self) -> &LinkName {
        &self.name
    }

    /// Returns the body length with the retained optional fields and name-length width.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Format`] if a hard-link address does not fit `offset_width`
    /// or is undefined at that width, or the body length exceeds [`usize::MAX`].
    pub fn encoded_len(&self, offset_width: OffsetWidth) -> Result<usize, LinkWireError> {
        Ok(self.prepare(offset_width)?.encoded_size())
    }

    /// Serializes the body with the retained optional fields and name-length width.
    ///
    /// The returned bytes exclude object-header padding.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::encoded_len`].
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, LinkWireError> {
        Ok(self.prepare(offset_width)?.to_bytes())
    }

    /// Prepares a body with borrowed bytes and its retained field choices.
    ///
    /// # Errors
    ///
    /// Returns the address and body length errors of [`Self::encoded_len`].
    pub fn prepare(&self, offset_width: OffsetWidth) -> Result<PreparedLink<'_>, LinkWireError> {
        PreparedLink::for_stored(self, offset_width)
    }

    fn from_view(link: &LinkMessageView<'_>) -> Self {
        Self {
            target: LinkTargetV1::from_view(link),
            target_value_len: link.target_value_len(),
            creation_order: link.creation_order(),
            charset: link.charset(),
            name: LinkName::from_view(link),
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        LinkTargetV1,
        Option<u64>,
        Option<CharacterSet>,
        LinkName,
        u16,
    ) {
        let Self {
            target,
            target_value_len,
            creation_order,
            charset,
            name,
        } = self;
        (target, creation_order, charset, name, target_value_len)
    }
}

impl StoredLink for LinkMessageV1 {
    fn name_field(&self) -> (&[u8], UintWidth) {
        (self.name.as_bytes(), self.name.width())
    }
    fn target_field(&self) -> (LinkTargetView<'_>, u16) {
        (self.target.as_view(), self.target_value_len)
    }
    fn creation_index(&self) -> Option<u64> {
        self.creation_order
    }
    fn charset_field(&self) -> Option<CharacterSet> {
        self.charset.clone()
    }
}

/// The link type of a hard link, from the Link type table of "The Link Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
pub(super) const HARD_LINK: u8 = 0;

/// The link type of a soft link, from the same table as [`HARD_LINK`].
pub(super) const SOFT_LINK: u8 = 1;

/// The link type of an external link, from the same table as [`HARD_LINK`].
pub(super) const EXTERNAL_LINK: u8 = 64;
