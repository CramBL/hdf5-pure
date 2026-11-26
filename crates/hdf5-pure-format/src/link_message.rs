//! HDF5 Link message parsing (message type 0x0006).

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use crate::address::StoredAddress;

use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_message_read::{LinkMessageView, LinkPrefix, LinkTargetView};
use crate::link_message_v1::LinkProjectionError;
use crate::link_message_write::PreparedLink;
use crate::width::OffsetWidth;

/// The type of a link in an HDF5 v2 group.
#[derive(Clone, Debug, PartialEq)]
pub enum LinkTarget {
    /// Hard link pointing to an object header address.
    Hard {
        object_header_address: StoredAddress,
    },
    /// Soft or external link pointing to a path.
    Indirect(Box<SymbolicLinkTarget>),
}

impl LinkTarget {
    /// Creates a soft link pointing to `target_path`.
    pub fn soft(target_path: String) -> Self {
        Self::Indirect(Box::new(SymbolicLinkTarget::Soft { target_path }))
    }

    /// Creates an external link pointing to `object_path` in `filename`.
    pub fn external(filename: String, object_path: String) -> Self {
        Self::Indirect(Box::new(SymbolicLinkTarget::External {
            filename,
            object_path,
        }))
    }
    pub(crate) fn as_view(&self) -> LinkTargetView<'_> {
        match self {
            Self::Hard {
                object_header_address,
            } => LinkTargetView::Hard {
                object_header_address: *object_header_address,
                explicit_type: false,
            },
            Self::Indirect(target) => match target.as_ref() {
                SymbolicLinkTarget::Soft { target_path } => LinkTargetView::Soft {
                    target_path: target_path.as_bytes(),
                },
                SymbolicLinkTarget::External {
                    filename,
                    object_path,
                } => LinkTargetView::External {
                    filename: filename.as_bytes(),
                    object_path: object_path.as_bytes(),
                },
            },
        }
    }
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<LinkTarget>() == 16);

/// The target of a soft or an external link.
#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicLinkTarget {
    /// Soft (symbolic) link with a target path string.
    Soft { target_path: String },
    /// External link pointing to a file and object path within it.
    External {
        filename: String,
        object_path: String,
    },
}

/// A parsed HDF5 Link message (type 0x0006).
#[derive(Debug, Clone, PartialEq)]
pub struct LinkMessage {
    /// Name of this link.
    pub name: String,
    /// What this link points to.
    pub link_target: LinkTarget,
    /// Creation order, if tracked.
    pub creation_order: Option<u64>,
    /// Character set of the link name.
    pub charset: CharacterSet,
}

/// Whether the stored name bytes `raw` name the link `wanted`.
///
/// [`LinkMessage::parse`] decodes a name with `from_utf8_lossy`, so a name that
/// is not valid UTF-8 is matched against the replacement characters a caller
/// would have received from it, the only spelling such a link can be requested by.
/// A valid name compares as bytes and allocates nothing.
fn name_matches(raw: &[u8], wanted: &str) -> bool {
    if raw == wanted.as_bytes() {
        return true;
    }
    core::str::from_utf8(raw).is_err() && String::from_utf8_lossy(raw) == wanted
}

/// Returns `true` if the stored name matches `name` or the name prefix is malformed.
///
/// Reads through the name. An object-header filter can retain malformed prefixes for
/// the later lookup to report.
pub fn link_is_named(data: &[u8], name: &str) -> bool {
    match LinkPrefix::parse(data) {
        Ok(prefix) => name_matches(prefix.name, name),
        Err(_) => true,
    }
}

impl LinkMessage {
    /// Returns the stored name bytes from the prefix of a version 1 Link message.
    ///
    /// Parses through the name, stopping before the target data. The returned slice borrows
    /// `data` and preserves invalid UTF-8 bytes.
    ///
    /// The prefix is defined in "The Link Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLinkVersion`] for a version other than 1,
    /// [`FormatError::InvalidLinkFlags`] for reserved flag bits,
    /// [`FormatError::InvalidCharacterSet`] for an unknown character set, or
    /// [`FormatError::UnexpectedEof`] for a truncated prefix. Returns
    /// [`FormatError::ValueTooLargeForPlatform`] if the name length exceeds [`usize::MAX`].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
    pub fn name_bytes(data: &[u8]) -> Result<&[u8], FormatError> {
        LinkPrefix::parse(data).map(|prefix| prefix.name)
    }

    /// Returns a borrowed hard-link name and its defined address for discovery.
    ///
    /// The caller can check the returned name bytes for component syntax. Returns [`None`]
    /// for a soft or external link after reading through its name.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. Returns
    /// [`FormatError::InvalidLinkType`] for any kind other than hard, soft, or external,
    /// [`FormatError::UnexpectedEof`] for a truncated hard address, or
    /// [`FormatError::DefinedAddressIsUndefined`] for the undefined address at `offset_width`.
    pub fn discover_hard_link(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<Option<(&[u8], StoredAddress)>, FormatError> {
        LinkPrefix::parse(data)?.discover_hard_link(offset_width)
    }

    /// Returns the object header address of the link in the Link message `data`, if it is a
    /// hard link named `name`.
    ///
    /// Compares the name before reading the target. Returns [`None`] for a different name or
    /// a soft or external link.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] for an unsupported `offset_size` or the
    /// errors of [`Self::hard_link_address_if_named_with_offset_width`].
    pub fn hard_link_address_if_named(
        data: &[u8],
        offset_size: u8,
        name: &str,
    ) -> Result<Option<StoredAddress>, FormatError> {
        Self::hard_link_address_if_named_with_offset_width(
            data,
            OffsetWidth::try_from(offset_size)?,
            name,
        )
    }

    /// Returns the object header address of the hard link named `name` using a validated address
    /// width.
    ///
    /// Returns [`None`] when the name differs. For a matching soft or external link, checks
    /// the name component and returns [`None`] before reading its target.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. For a matching name, returns
    /// the errors of [`Self::hard_link_address_range`] for an invalid kind, name component,
    /// or hard address.
    pub fn hard_link_address_if_named_with_offset_width(
        data: &[u8],
        offset_width: OffsetWidth,
        name: &str,
    ) -> Result<Option<StoredAddress>, FormatError> {
        let prefix = LinkPrefix::parse(data)?;
        if !name_matches(prefix.name, name) {
            return Ok(None);
        }
        Ok(prefix
            .read_hard_target(offset_width)?
            .map(|(address, _)| address))
    }

    /// Returns the target of the link named `name`, or `None` for another name.
    ///
    /// The name is compared in the message bytes before the target is parsed.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. For a matching name, returns
    /// the errors of [`Self::parse_with_offset_width`] for invalid name components,
    /// invalid targets, or failed text conversion.
    pub fn target_if_named(
        data: &[u8],
        offset_width: OffsetWidth,
        name: &str,
    ) -> Result<Option<LinkTarget>, FormatError> {
        let prefix = LinkPrefix::parse(data)?;
        if !name_matches(prefix.name, name) {
            return Ok(None);
        }
        Ok(Some(
            prefix
                .target_view(offset_width)?
                .to_text()
                .map_err(LinkProjectionError::into_format)?,
        ))
    }

    /// Returns the byte range of the object header address in the Link message `data`, or `None`
    /// for a soft or an external link.
    ///
    /// A caller that points a hard link at another object in place writes the new address over
    /// this range. For soft and external links, checks the name and returns [`None`] before
    /// reading the target.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. Returns [`FormatError::InvalidLinkType`]
    /// for any kind other than hard, soft, or external, [`FormatError::InvalidLinkNameBytes`] for
    /// an invalid name component, [`FormatError::UnexpectedEof`] for a truncated hard address,
    /// or [`FormatError::DefinedAddressIsUndefined`] for the undefined address at `offset_width`.
    pub fn hard_link_address_range(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<Option<Range<usize>>, FormatError> {
        Ok(LinkPrefix::parse(data)?
            .read_hard_target(offset_width)?
            .map(|(_, range)| range))
    }

    /// Serializes this message using the canonical field choices of [`Self::prepare`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::prepare`].
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, FormatError> {
        Ok(self.prepare(offset_width)?.to_bytes())
    }

    /// Prepares a Link body from the borrowed name and supported target.
    ///
    /// Uses the smallest name-length width and omits the hard-link type field and the ASCII
    /// character-set field. Preserves any creation-order field.
    ///
    /// # Errors
    ///
    /// Returns the name, address, and length-overflow errors of [`PreparedLink::hard`].
    /// Returns [`FormatError::InvalidLinkTarget`] for an empty path or one containing a NUL,
    /// or [`FormatError::LengthTooLarge`] if a symbolic target exceeds its two-byte length field.
    /// The external target size includes its version and flags byte and both NUL terminators.
    pub fn prepare(&self, offset_width: OffsetWidth) -> Result<PreparedLink<'_>, FormatError> {
        PreparedLink::from_text_fields(
            offset_width,
            &self.name,
            self.creation_order,
            &self.charset,
            self.link_target.as_view(),
        )
    }

    /// Parses the body of a Link message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and the
    /// errors of [`parse_with_offset_width`](Self::parse_with_offset_width) in the cases it lists.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<LinkMessage, FormatError> {
        Self::parse_with_offset_width(data, OffsetWidth::try_from(offset_size)?)
    }

    /// Parses and converts the complete Link message at the start of `data` to owned text.
    ///
    /// Uses the framing of [`LinkMessageView::parse_prefix`] and the text conversion of
    /// [`LinkMessageView::to_text`]. Ignores bytes after the target.
    ///
    /// # Errors
    ///
    /// Propagates the structural errors of [`LinkMessageView::parse_prefix`]. After decoding,
    /// returns [`FormatError::InvalidLinkType`] for a user-defined kind or
    /// [`FormatError::InvalidSoftLinkEncoding`] for a soft path with invalid UTF-8.
    pub fn parse_with_offset_width(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<LinkMessage, FormatError> {
        LinkMessageView::parse_prefix(data, offset_width)?
            .to_text()
            .map_err(LinkProjectionError::into_format)
    }
}

/// The link type of a hard link, from the Link type table of "The Link Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
pub(crate) const HARD_LINK: u8 = 0;

/// The link type of a soft link, from the same table as [`HARD_LINK`].
pub(crate) const SOFT_LINK: u8 = 1;

/// The link type of an external link, from the same table as [`HARD_LINK`].
pub(crate) const EXTERNAL_LINK: u8 = 64;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::link_message;
    use test_util::widths::Widths;

    use super::*;
    use crate::address::StoredAddressExt;

    #[rstest]
    #[case::hard(LinkTarget::Hard { object_header_address: StoredAddress::new(42) })]
    #[case::soft(LinkTarget::soft("target".into()))]
    #[case::external(LinkTarget::external("file.h5".into(), "/target".into()))]
    fn link_names_borrow_prefix_bytes_before_the_target(#[case] link_target: LinkTarget) {
        let data = LinkMessage {
            name: "link".into(),
            link_target,
            creation_order: Some(7),
            charset: CharacterSet::Utf8,
        }
        .serialize(OffsetWidth::Eight)
        .unwrap();
        let prefix = LinkPrefix::parse(&data).unwrap();
        let name = LinkMessage::name_bytes(&data[..prefix.target_start()]).unwrap();
        assert_eq!(name, b"link");
        assert_eq!(name.as_ptr(), prefix.name.as_ptr());
    }

    #[rstest]
    #[case::version(vec![2, 0], FormatError::InvalidLinkVersion(2))]
    #[case::truncated(vec![1, 0, 3, b'x'], FormatError::UnexpectedEof { expected: 6, available: 4 })]
    #[case::charset(vec![1, 0x10, 2, 1], FormatError::InvalidCharacterSet(2))]
    fn borrowed_link_names_reject_malformed_prefixes(
        #[case] data: Vec<u8>,
        #[case] expected: FormatError,
    ) {
        assert_eq!(LinkMessage::name_bytes(&data).unwrap_err(), expected);
    }

    #[test]
    fn borrowed_link_names_preserve_invalid_utf8() {
        let mut data = build_hard_link("a", 42, 8, None, None, 1);
        data[3] = 0xff;
        assert_eq!(LinkMessage::name_bytes(&data).unwrap(), &[0xff]);
    }

    fn build_hard_link(
        name: &str,
        addr: u64,
        offset_size: usize,
        creation_order: Option<u64>,
        charset: Option<u8>,
        name_size_width: usize,
    ) -> Vec<u8> {
        let mut link = link_message::HardLink::new(name, addr).name_size_width(name_size_width);
        if let Some(creation_order) = creation_order {
            link = link.creation_order(creation_order);
        }
        if let Some(charset) = charset {
            link = link.character_set(link_message::CharacterSet(charset));
        }
        link.build(Widths::new(offset_size, offset_size))
    }

    #[test]
    fn hard_link_ascii_no_creation_order() {
        let data = build_hard_link("mydata", 0x1000, 8, None, None, 1);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "mydata");
        assert_eq!(
            msg.link_target,
            LinkTarget::Hard {
                object_header_address: StoredAddress::new(0x1000)
            }
        );
        assert_eq!(msg.creation_order, None);
        assert_eq!(msg.charset, CharacterSet::Ascii);
    }

    #[test]
    fn hard_link_utf8_with_creation_order() {
        let data = build_hard_link("données", 0x2000, 8, Some(42), Some(1), 1);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "données");
        assert_eq!(
            msg.link_target,
            LinkTarget::Hard {
                object_header_address: StoredAddress::new(0x2000)
            }
        );
        assert_eq!(msg.creation_order, Some(42));
        assert_eq!(msg.charset, CharacterSet::Utf8);
    }

    #[test]
    fn soft_link() {
        let target = "/group1/dataset";
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x08); // flags: bit 3 = link type present, name size = 1 byte (bits 0-1 = 0)
        data.push(1); // link type = soft
        data.push(4); // name length = 4
        data.extend_from_slice(b"link");
        data.extend_from_slice(&(target.len() as u16).to_le_bytes());
        data.extend_from_slice(target.as_bytes());

        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "link");
        assert_eq!(msg.link_target, LinkTarget::soft(target.to_string()));
    }

    #[test]
    fn a_soft_link_target_with_invalid_utf8_is_rejected() {
        let mut data = LinkMessage {
            name: String::from("link"),
            link_target: LinkTarget::soft(String::from("x")),
            creation_order: None,
            charset: CharacterSet::Ascii,
        }
        .serialize(OffsetWidth::Eight)
        .unwrap();
        *data.last_mut().unwrap() = 0xff;
        let error = LinkMessage::target_if_named(&data, OffsetWidth::Eight, "link").unwrap_err();
        let FormatError::InvalidSoftLinkEncoding { source } = error else {
            panic!("expected InvalidSoftLinkEncoding, got {error:?}");
        };
        assert_eq!(source.valid_up_to(), 0);
        assert_eq!(source.error_len(), Some(1));
        assert_eq!(
            LinkMessage::target_if_named(&data, OffsetWidth::Eight, "other").unwrap(),
            None
        );
    }

    #[test]
    fn name_length_2bytes() {
        let data = build_hard_link("test", 0x500, 8, None, None, 2);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "test");
    }

    #[test]
    fn name_length_4bytes() {
        let data = build_hard_link("abcd", 0x600, 8, None, None, 4);
        let msg = LinkMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.name, "abcd");
    }

    #[test]
    fn a_named_hard_link_answers_with_its_address() {
        let data = build_hard_link("mydata", 0x1000, 8, None, None, 1);
        assert_eq!(
            LinkMessage::hard_link_address_if_named(&data, 8, "mydata").unwrap(),
            Some(StoredAddress::new(0x1000))
        );
        assert_eq!(
            LinkMessage::hard_link_address_if_named(&data, 8, "other").unwrap(),
            None
        );
    }

    /// A soft link refers to a path, not an object header, so a lookup passes over it
    /// however it is named.
    #[test]
    fn a_soft_link_of_the_wanted_name_is_not_an_address() {
        let target = "/group1/dataset";
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x08); // flags: link type present, 1-byte name length
        data.push(1); // link type = soft
        data.push(4); // name length
        data.extend_from_slice(b"link");
        data.extend_from_slice(&(target.len() as u16).to_le_bytes());
        data.extend_from_slice(target.as_bytes());

        assert_eq!(
            LinkMessage::hard_link_address_if_named(&data, 8, "link").unwrap(),
            None
        );
    }

    /// A name that is not valid UTF-8 reaches a caller as the replacement
    /// characters [`LinkMessage::parse`] decodes it to, and that spelling is the
    /// only one such a link can be requested by, so it is the one that matches.
    #[test]
    fn a_name_that_is_not_utf8_matches_the_spelling_a_reader_gets() {
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x00); // flags: hard link, 1-byte name length
        data.push(2); // name length
        data.extend_from_slice(&[0xFF, 0xFE]); // not UTF-8
        data.extend_from_slice(&0x2000u64.to_le_bytes());

        let decoded = LinkMessage::parse(&data, 8).unwrap().name;
        assert_eq!(
            LinkMessage::hard_link_address_if_named(&data, 8, &decoded).unwrap(),
            Some(StoredAddress::new(0x2000)),
            "a link found by listing must be findable by the name the listing gave"
        );
        assert!(link_is_named(&data, &decoded));
    }

    /// A message this cannot read is kept, not filtered away, so the parse
    /// that follows reports it. Returning "not this one" would hide it.
    /// A link type this crate does not know is a corrupt group, not a missing
    /// name, and [`LinkMessage::parse`] says so. A lookup must not soften that
    /// into "no such link".
    #[test]
    fn a_lookup_of_a_link_type_this_crate_does_not_know_returns_invalid_link_type() {
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x08); // flags: link type present, 1-byte name length
        data.push(99); // not a link type this crate defines
        data.push(1); // name length
        data.push(b'x');

        assert_eq!(
            LinkMessage::hard_link_address_if_named(&data, 8, "x").unwrap_err(),
            FormatError::InvalidLinkType(99)
        );
        // ...but only for the link that was requested: another name's lookup is
        // not this message's business, and the filter drops it before the scan.
        assert_eq!(
            LinkMessage::hard_link_address_if_named(&data, 8, "y").unwrap(),
            None
        );
        assert!(!link_is_named(&data, "y"));
    }

    #[test]
    fn an_unreadable_link_is_kept_by_the_filter() {
        assert!(link_is_named(&[2, 0, 0, 0], "anything"), "bad version");
        assert!(link_is_named(&[], "anything"), "empty body");
    }

    #[test]
    fn invalid_version() {
        let data = vec![2, 0, 0, 0]; // version 2
        let err = LinkMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLinkVersion(2));
    }

    #[test]
    fn a_soft_link_target_longer_than_its_length_field_is_rejected() {
        let error = LinkMessage {
            name: "link".into(),
            link_target: LinkTarget::soft("a".repeat(usize::from(u16::MAX) + 1)),
            creation_order: None,
            charset: CharacterSet::Ascii,
        }
        .serialize(OffsetWidth::Eight)
        .unwrap_err();
        assert_eq!(
            error,
            FormatError::LengthTooLarge {
                length: 65_536,
                length_size: 2
            }
        );
    }

    #[rstest]
    #[case::a_hard_link(build_hard_link("a", 0x1234, 8, None, None, 1), Ok(Some(4..12)))]
    #[case::a_soft_link(soft_link_body(), Ok(None))]
    #[case::a_hard_link_cut_short(
        build_hard_link("a", 0x1234, 8, None, None, 1)[..10].to_vec(),
        Err(FormatError::UnexpectedEof { expected: 12, available: 10 })
    )]
    fn a_hard_link_address_is_located_past_the_name(
        #[case] data: Vec<u8>,
        #[case] expected: Result<Option<Range<usize>>, FormatError>,
    ) {
        assert_eq!(
            LinkMessage::hard_link_address_range(&data, OffsetWidth::Eight),
            expected
        );
    }

    fn soft_link_body() -> Vec<u8> {
        LinkMessage {
            name: "link".into(),
            link_target: LinkTarget::soft("/group1".into()),
            creation_order: None,
            charset: CharacterSet::Ascii,
        }
        .serialize(OffsetWidth::Eight)
        .unwrap()
    }
}
