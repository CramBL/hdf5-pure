//! HDF5 Link message parsing (message type 0x0006).

use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use crate::address::StoredAddress;
use crate::bytes;
use crate::bytes::{ensure_len, read_offset, read_uint_width};
use crate::convert::Narrow;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::width::OffsetWidth;
use crate::width::UintWidth;

/// The type of a link in an HDF5 v2 group.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkTarget {
    /// Hard link pointing to an object header address.
    Hard {
        object_header_address: StoredAddress,
    },
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

/// Everything a Link message declares before its target data, with the name
/// left where it lies in the message bytes.
///
/// Splitting the parse here is what lets a lookup by name read a group's links
/// without allocating for the ones it rejects: [`LinkMessage::parse`] owns the
/// name because its caller keeps it, while
/// [`hard_link_address_if_named`](LinkMessage::hard_link_address_if_named)
/// compares the borrowed bytes and stops.
struct LinkPrefix<'a> {
    /// The link name as stored, undecoded.
    name: &'a [u8],
    /// 0 for a hard link, 1 soft, 64 external.
    link_type_code: u8,
    creation_order: Option<u64>,
    charset: CharacterSet,
    /// Offset of the target data that follows the name.
    target_pos: usize,
}

impl<'a> LinkPrefix<'a> {
    /// Parses a Link message up to its target data, which begins at
    /// [`target_pos`](Self::target_pos).
    fn parse(data: &'a [u8]) -> Result<Self, FormatError> {
        ensure_len(data, 0, 2)?;

        let version = data[0];
        if version != 1 {
            return Err(FormatError::InvalidLinkVersion(version));
        }

        let flags = data[1];
        let name_size_field_width = UintWidth::from_flags(flags);
        // Bit 2: creation order field present
        let has_creation_order = flags & 0x04 != 0;
        // Bit 3: link type field present
        let has_link_type = flags & 0x08 != 0;
        // Bit 4: link name character set field present
        let has_charset = flags & 0x10 != 0;

        let mut pos = 2;

        let link_type_code = if has_link_type {
            ensure_len(data, pos, 1)?;
            let v = data[pos];
            pos += 1;
            v
        } else {
            0 // hard link
        };

        let creation_order = if has_creation_order {
            ensure_len(data, pos, 8)?;
            let co = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            pos += 8;
            Some(co)
        } else {
            None
        };

        let charset = if has_charset {
            ensure_len(data, pos, 1)?;
            let cs = data[pos];
            pos += 1;
            match cs {
                0 => CharacterSet::Ascii,
                1 => CharacterSet::Utf8,
                _ => return Err(FormatError::InvalidCharacterSet(cs)),
            }
        } else {
            CharacterSet::Ascii
        };

        let name_len = read_uint_width(data, pos, name_size_field_width)?.to_usize()?;
        pos += usize::from(name_size_field_width.get());

        ensure_len(data, pos, name_len)?;
        let name = &data[pos..pos + name_len];
        pos += name_len;

        Ok(LinkPrefix {
            name,
            link_type_code,
            creation_order,
            charset,
            target_pos: pos,
        })
    }

    /// Returns the range of the hard link address in `data`, the message this prefix is parsed
    /// from, or `None` for a soft or an external link.
    fn hard_link_address_range(
        &self,
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<Option<Range<usize>>, FormatError> {
        match self.link_type_code {
            HARD_LINK => {
                let address = self.target_pos..self.target_pos + usize::from(offset_width.get());
                ensure_len(data, address.start, address.len())?;
                Ok(Some(address))
            }
            SOFT_LINK | EXTERNAL_LINK => Ok(None),
            // A link type this crate does not know is an error, as in
            // `LinkMessage::parse`: a lookup that returned no address would report a
            // corrupt group as a missing name.
            other => Err(FormatError::InvalidLinkType(other)),
        }
    }
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

/// Returns `true` if the Link message `data` holds the link `name`.
///
/// A message this cannot parse returns `true`: this exists to let an object-header
/// parse drop the links a lookup will not read (see
/// [`crate::object_header::MessageFilter`]), and a link whose name cannot even be
/// read is one the scan that follows must still see, so that it rejects the group
/// wherever the damage sits, and not only when it precedes the wanted link.
pub fn link_is_named(data: &[u8], name: &str) -> bool {
    match LinkPrefix::parse(data) {
        Ok(prefix) => name_matches(prefix.name, name),
        Err(_) => true,
    }
}

impl LinkMessage {
    /// Returns the object header address of the link in the Link message `data`, if it is a
    /// hard link named `name`.
    ///
    /// Returns `Ok(None)` for a link with another name, and for a soft or external link, which
    /// refers to a path in place of an object header. The name is compared where it lies in
    /// `data`, so a path lookup that reads every Link message of a group allocates nothing for
    /// the links it passes over.
    ///
    /// # Errors
    ///
    /// Returns the error [`parse`](Self::parse) returns if the message is malformed up to the end
    /// of the name, and, for a link named `name`, if its link type or its hard link address is
    /// malformed.
    pub fn hard_link_address_if_named(
        data: &[u8],
        offset_size: u8,
        name: &str,
    ) -> Result<Option<StoredAddress>, FormatError> {
        let prefix = LinkPrefix::parse(data)?;
        if !name_matches(prefix.name, name) {
            return Ok(None);
        }
        let offset_width = OffsetWidth::try_from(offset_size)?;
        prefix
            .hard_link_address_range(data, offset_width)?
            .map(|address| {
                Ok(StoredAddress::new(bytes::read_offset_width(
                    data,
                    address.start,
                    offset_width,
                )?))
            })
            .transpose()
    }

    /// Returns the byte range of the object header address in the Link message `data`, or `None`
    /// for a soft or an external link.
    ///
    /// A caller that points a hard link at another object in place writes the new address over
    /// this range.
    ///
    /// # Errors
    ///
    /// Returns the error [`parse`](Self::parse) returns if the message is malformed up to the end
    /// of the name, [`FormatError::InvalidLinkType`] if the link type is not hard, soft, or
    /// external, and [`FormatError::UnexpectedEof`] if `data` ends inside the address.
    pub fn hard_link_address_range(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<Option<Range<usize>>, FormatError> {
        LinkPrefix::parse(data)?.hard_link_address_range(data, offset_width)
    }

    /// Serializes this message's body, the inverse of [`parse`](Self::parse).
    ///
    /// # Panics
    ///
    /// Panics if the address of a hard link does not fit `offset_width`, or if the target of a
    /// soft or external link is longer than its 2-byte length field allows.
    pub fn serialize(&self, offset_width: OffsetWidth) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1); // version

        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len();
        let name_width = UintWidth::smallest_for_len(name_len);

        let is_hard = matches!(self.link_target, LinkTarget::Hard { .. });
        let has_link_type = !is_hard;
        let has_creation_order = self.creation_order.is_some();
        let has_charset = self.charset != CharacterSet::Ascii;

        let mut flags: u8 = 0;
        // Bits 0-1: size of name length field
        flags |= name_width.flag_bits();
        // Bit 2: creation order present
        if has_creation_order {
            flags |= 0x04;
        }
        // Bit 3: link type present
        if has_link_type {
            flags |= 0x08;
        }
        // Bit 4: charset present
        if has_charset {
            flags |= 0x10;
        }
        buf.push(flags);

        if has_link_type {
            match &self.link_target {
                LinkTarget::Soft { .. } => buf.push(1),
                LinkTarget::External { .. } => buf.push(64),
                _ => {}
            }
        }

        if let Some(co) = self.creation_order {
            buf.extend_from_slice(&co.to_le_bytes());
        }

        if has_charset {
            buf.push(match self.charset {
                CharacterSet::Ascii => 0,
                CharacterSet::Utf8 => 1,
            });
        }

        name_width.write(&mut buf, name_len);
        buf.extend_from_slice(name_bytes);

        match &self.link_target {
            LinkTarget::Hard {
                object_header_address,
            } => bytes::write_offset(&mut buf, object_header_address.get(), offset_width),
            LinkTarget::Soft { target_path } => {
                let path_bytes = target_path.as_bytes();
                push_link_information_length(&mut buf, path_bytes.len());
                buf.extend_from_slice(path_bytes);
            }
            LinkTarget::External {
                filename,
                object_path,
            } => {
                let mut ext_data = Vec::new();
                ext_data.push(0); // flags
                ext_data.extend_from_slice(filename.as_bytes());
                ext_data.push(0);
                ext_data.extend_from_slice(object_path.as_bytes());
                ext_data.push(0);
                push_link_information_length(&mut buf, ext_data.len());
                buf.extend_from_slice(&ext_data);
            }
        }

        buf
    }

    /// Parses the body of a Link message.
    ///
    /// `offset_size` is the width of a hard link's address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLinkVersion`] if the version is not 1,
    /// [`FormatError::InvalidCharacterSet`] if the character set is not ASCII or UTF-8,
    /// [`FormatError::InvalidLinkType`] if the link type is not hard, soft, or external,
    /// [`FormatError::InvalidOffsetSize`] if the link is a hard link and `offset_size` is not 2, 4,
    /// or 8, [`FormatError::ValueTooLargeForPlatform`] if the name length does not fit `usize`, and
    /// [`FormatError::UnexpectedEof`] if `data` ends inside a field.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<LinkMessage, FormatError> {
        let LinkPrefix {
            name,
            link_type_code,
            creation_order,
            charset,
            target_pos: mut pos,
        } = LinkPrefix::parse(data)?;
        let name = String::from_utf8_lossy(name).into_owned();

        // Link target data
        let link_target = match link_type_code {
            HARD_LINK => {
                // Hard link
                LinkTarget::Hard {
                    object_header_address: StoredAddress::new(read_offset(data, pos, offset_size)?),
                }
            }
            SOFT_LINK => {
                ensure_len(data, pos, 2)?;
                let soft_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                ensure_len(data, pos, soft_len)?;
                let target_path = String::from_utf8_lossy(&data[pos..pos + soft_len]).into_owned();
                LinkTarget::Soft { target_path }
            }
            EXTERNAL_LINK => {
                ensure_len(data, pos, 2)?;
                let ext_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                ensure_len(data, pos, ext_len)?;
                let ext_data = &data[pos..pos + ext_len];
                // External link value: flags(1) + null-terminated filename + null-terminated obj path
                // Skip the flags byte
                let start = if !ext_data.is_empty() { 1 } else { 0 };
                let rest = &ext_data[start..];
                let null1 = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                let filename = String::from_utf8_lossy(&rest[..null1]).into_owned();
                let after_null1 = if null1 + 1 < rest.len() {
                    null1 + 1
                } else {
                    rest.len()
                };
                let rest2 = &rest[after_null1..];
                let null2 = rest2.iter().position(|&b| b == 0).unwrap_or(rest2.len());
                let object_path = String::from_utf8_lossy(&rest2[..null2]).into_owned();
                LinkTarget::External {
                    filename,
                    object_path,
                }
            }
            other => return Err(FormatError::InvalidLinkType(other)),
        };

        Ok(LinkMessage {
            name,
            link_target,
            creation_order,
            charset,
        })
    }
}

fn push_link_information_length(buf: &mut Vec<u8>, len: usize) {
    let Ok(field) = u16::try_from(len) else {
        panic!("link information of {len} bytes does not fit the 2-byte length field");
    };
    buf.extend_from_slice(&field.to_le_bytes());
}

/// The link type of a hard link, from the Link type table of "The Link Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
const HARD_LINK: u8 = 0;

/// The link type of a soft link, from the same table as [`HARD_LINK`].
const SOFT_LINK: u8 = 1;

/// The link type of an external link, from the same table as [`HARD_LINK`].
const EXTERNAL_LINK: u8 = 64;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::link_message;
    use test_util::widths::Widths;

    use super::*;

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
        assert_eq!(
            msg.link_target,
            LinkTarget::Soft {
                target_path: target.to_string()
            }
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
    fn invalid_link_type() {
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x08); // flags: bit 3 = link type present
        data.push(99); // invalid link type
        data.push(1); // name length = 1
        data.push(b'x');
        let err = LinkMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLinkType(99));
    }

    #[test]
    #[should_panic(
        expected = "link information of 65536 bytes does not fit the 2-byte length field"
    )]
    fn a_soft_link_target_longer_than_its_length_field_panics() {
        LinkMessage {
            name: "link".into(),
            link_target: LinkTarget::Soft {
                target_path: "a".repeat(usize::from(u16::MAX) + 1),
            },
            creation_order: None,
            charset: CharacterSet::Ascii,
        }
        .serialize(OffsetWidth::Eight);
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
            link_target: LinkTarget::Soft {
                target_path: "/group1".into(),
            },
            creation_order: None,
            charset: CharacterSet::Ascii,
        }
        .serialize(OffsetWidth::Eight)
    }
}
