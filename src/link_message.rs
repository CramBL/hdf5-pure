//! HDF5 Link message parsing (message type 0x0006).

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::address::StoredAddress;
use crate::bytes::{ensure_len, read_offset, read_uint_width};
use crate::convert::Narrow;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
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

/// Parse a Link message up to (not including) its target data.
fn parse_prefix(data: &[u8]) -> Result<LinkPrefix<'_>, FormatError> {
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

    // Link type
    let link_type_code = if has_link_type {
        ensure_len(data, pos, 1)?;
        let v = data[pos];
        pos += 1;
        v
    } else {
        0 // hard link
    };

    // Creation order
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

    // Character set
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

    // Link name length
    let name_len = read_uint_width(data, pos, name_size_field_width)?.to_usize()?;
    pos += usize::from(name_size_field_width.get());

    // Link name
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

/// Whether the stored name bytes `raw` name the link `wanted`.
///
/// [`LinkMessage::parse`] decodes a name with `from_utf8_lossy`, so a name that
/// is not valid UTF-8 is matched against the replacement characters a caller
/// would have received from it — the only spelling such a link can be asked for.
/// A valid name compares as bytes and allocates nothing.
fn name_matches(raw: &[u8], wanted: &str) -> bool {
    if raw == wanted.as_bytes() {
        return true;
    }
    core::str::from_utf8(raw).is_err() && String::from_utf8_lossy(raw) == wanted
}

/// Whether the Link message `data` names the link `name`.
///
/// A message this cannot parse answers `true`: this exists to let an object-header
/// parse drop the links a lookup will not read (see
/// [`crate::object_header::MessageFilter`]), and a link whose name cannot even be
/// read is one the scan that follows must still see, so that it refuses the group
/// wherever the damage sits rather than only when it precedes the wanted link.
pub(crate) fn link_is_named(data: &[u8], name: &str) -> bool {
    match parse_prefix(data) {
        Ok(prefix) => name_matches(prefix.name, name),
        Err(_) => true,
    }
}

impl LinkMessage {
    /// The object-header address this Link message points at, if it is a *hard*
    /// link named `name`.
    ///
    /// `Ok(None)` means this link is not the one asked for — a different name, or
    /// a soft/external link, which name no object header in this file and are
    /// what [`crate::group_v2`] skips when resolving a path.
    ///
    /// This exists so resolving one path does not cost a `String` per link of
    /// the group: a group of *n* children is *n* Link messages in its object
    /// header, and building every name to find one made opening each child of a
    /// group quadratic in the group's size (issue #228).
    pub(crate) fn hard_link_address_if_named(
        data: &[u8],
        offset_size: u8,
        name: &str,
    ) -> Result<Option<StoredAddress>, FormatError> {
        let prefix = parse_prefix(data)?;
        if !name_matches(prefix.name, name) {
            return Ok(None);
        }
        match prefix.link_type_code {
            0 => Ok(Some(StoredAddress::new(read_offset(
                data,
                prefix.target_pos,
                offset_size,
            )?))),
            // A soft or external link names a path, not an object header here, so
            // it is passed over exactly as `crate::group_v2` passes over it when
            // resolving entries.
            1 | 64 => Ok(None),
            // A link type this crate does not know is refused rather than passed
            // over, because [`Self::parse`] refuses it: a lookup that answered
            // "no such link" would report a corrupt group as an ordinary missing
            // name.
            other => Err(FormatError::InvalidLinkType(other)),
        }
    }

    /// Serialize link message to HDF5 message bytes.
    pub fn serialize(&self, offset_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1); // version

        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len();
        let name_size_width: u8 = if name_len <= 0xFF {
            1
        } else if name_len <= 0xFFFF {
            2
        } else {
            4
        };

        let is_hard = matches!(self.link_target, LinkTarget::Hard { .. });
        let has_link_type = !is_hard;
        let has_creation_order = self.creation_order.is_some();
        let has_charset = self.charset != CharacterSet::Ascii;

        let mut flags: u8 = 0;
        // Bits 0-1: size of name length field
        let size_bits = match name_size_width {
            1 => 0u8,
            2 => 1,
            4 => 2,
            _ => 3,
        };
        flags |= size_bits;
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

        // `name_size_width` was chosen above (1/2/4 bytes) to be the smallest
        // field that holds `name_len`, so each arm's narrowing matches a width
        // already proven to fit the value.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "name_size_width selected above to fit name_len"
        )]
        match name_size_width {
            1 => buf.push(name_len as u8),
            2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&(name_len as u64).to_le_bytes()),
        }
        buf.extend_from_slice(name_bytes);

        match &self.link_target {
            LinkTarget::Hard {
                object_header_address,
            } =>
            {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "offset_size is the file's on-disk address width; each arm \
                              narrows to the width chosen to hold object header addresses"
                )]
                match offset_size {
                    2 => buf.extend_from_slice(&(object_header_address.get() as u16).to_le_bytes()),
                    4 => buf.extend_from_slice(&(object_header_address.get() as u32).to_le_bytes()),
                    8 => buf.extend_from_slice(&object_header_address.get().to_le_bytes()),
                    _ => {}
                }
            }
            LinkTarget::Soft { target_path } => {
                let path_bytes = target_path.as_bytes();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "soft-link length prefix is a 2-byte on-disk field; target \
                              paths are bounded by that HDF5 format limit"
                )]
                buf.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
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
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "external-link length prefix is a 2-byte on-disk field; the \
                              filename + object path are bounded by that HDF5 format limit"
                )]
                buf.extend_from_slice(&(ext_data.len() as u16).to_le_bytes());
                buf.extend_from_slice(&ext_data);
            }
        }

        buf
    }

    /// Parse a Link message from raw message data.
    ///
    /// `offset_size` is needed for hard link target addresses.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<LinkMessage, FormatError> {
        let LinkPrefix {
            name,
            link_type_code,
            creation_order,
            charset,
            target_pos: mut pos,
        } = parse_prefix(data)?;
        let name = String::from_utf8_lossy(name).into_owned();

        // Link target data
        let link_target = match link_type_code {
            0 => {
                // Hard link
                LinkTarget::Hard {
                    object_header_address: StoredAddress::new(read_offset(data, pos, offset_size)?),
                }
            }
            1 => {
                // Soft link
                ensure_len(data, pos, 2)?;
                let soft_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                ensure_len(data, pos, soft_len)?;
                let target_path = String::from_utf8_lossy(&data[pos..pos + soft_len]).into_owned();
                LinkTarget::Soft { target_path }
            }
            64 => {
                // External link
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a hard link message with given parameters.
    fn build_hard_link(
        name: &str,
        addr: u64,
        offset_size: u8,
        creation_order: Option<u64>,
        charset: Option<u8>,
        name_size_width: u8, // 1, 2, 4
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1); // version

        let mut flags: u8 = 0;
        // Bits 0-1: name length field size
        let size_bits = match name_size_width {
            1 => 0u8,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0,
        };
        flags |= size_bits;
        // Bit 2: creation order present
        if creation_order.is_some() {
            flags |= 0x04;
        }
        // hard link: don't set bit 3 (link type field not present)
        // Bit 4: charset present
        if charset.is_some() {
            flags |= 0x10;
        }
        buf.push(flags);

        // no link_type field for hard links (bit 1 not set)

        if let Some(co) = creation_order {
            buf.extend_from_slice(&co.to_le_bytes());
        }

        if let Some(cs) = charset {
            buf.push(cs);
        }

        // name length
        let name_len = name.len();
        match name_size_width {
            1 => buf.push(name_len as u8),
            2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&(name_len as u64).to_le_bytes()),
            _ => {}
        }

        buf.extend_from_slice(name.as_bytes());

        // hard link data: address
        match offset_size {
            4 => buf.extend_from_slice(&(addr as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&addr.to_le_bytes()),
            _ => {}
        }

        buf
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

    /// A soft link names a path, not an object header, so a lookup passes over it
    /// however it is named — the same way [`crate::group_v2`] leaves soft and
    /// external links out of the entries it resolves.
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
    /// only one such a link can be asked for — so it is the one that matches.
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

    /// A message this cannot read is kept rather than filtered away, so the parse
    /// that follows reports it. Answering "not this one" would hide it.
    /// A link type this crate does not know is a corrupt group, not a missing
    /// name, and [`LinkMessage::parse`] says so — a lookup must not soften that
    /// into "no such link".
    #[test]
    fn a_link_type_this_crate_does_not_know_is_refused_not_skipped() {
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0x08); // flags: link type present, 1-byte name length
        data.push(99); // not a link type this crate knows
        data.push(1); // name length
        data.push(b'x');

        assert_eq!(
            LinkMessage::hard_link_address_if_named(&data, 8, "x").unwrap_err(),
            FormatError::InvalidLinkType(99)
        );
        // ...but only for the link that was asked for: another name's lookup is
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
}
