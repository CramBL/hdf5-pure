//! The Attribute message, which stores the name, datatype, dataspace, and data of one attribute.
//!
//! The message is defined in "The Attribute Message" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attribute

use alloc::string::String;
use alloc::vec::Vec;

use crate::bytes;
use crate::convert::Narrow;
use crate::dataspace::Dataspace;
use crate::datatype;
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::shared_message::DatatypeLocation;
use crate::shared_message::SharedResolver;
use crate::shared_message::Unresolvable;
use crate::width::OffsetWidth;

/// An Attribute message: the name, datatype, dataspace, and data of one attribute.
///
/// The parsers read message versions 1, 2, and 3. [`serialize`](Self::serialize) writes version 2,
/// and [`serialize_v3`](Self::serialize_v3) version 3. The message is defined in "The Attribute
/// Message" of the [format specification, version 4.0][spec].
///
/// Two messages are equal when every field is equal, so a message that refers to a committed
/// datatype is not equal to one that encodes the same datatype.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attribute
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeMessage {
    /// The attribute's name, without the null terminator the message stores.
    pub name: String,
    /// The datatype of the attribute's elements, wherever the message stores it.
    pub datatype: Datatype,
    /// The attribute's dataspace.
    pub dataspace: Dataspace,
    /// The attribute's element bytes as the message stores them, the number of elements times the
    /// size of the datatype.
    pub raw_data: Vec<u8>,
    /// Whether the message refers to a committed datatype for [`datatype`](Self::datatype).
    pub datatype_location: DatatypeLocation,
}

impl AttributeMessage {
    /// Parses the name of the attribute message in `data`, without decoding its datatype or
    /// dataspace.
    ///
    /// The name field comes before the datatype field in every version, so a caller without
    /// access to the file reads the name of a message that refers to a committed datatype.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeVersion`] if the version is not 1, 2, or 3, and
    /// [`FormatError::UnexpectedEof`] if `data` ends before the end of the name.
    pub fn parse_name(data: &[u8]) -> Result<String, FormatError> {
        bytes::ensure_len(data, 0, V1_V2_PREFIX_SIZE)?;
        let version = data[0];
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let name_start = match version {
            VERSION_ONE | VERSION_TWO => V1_V2_PREFIX_SIZE,
            VERSION_THREE => V3_PREFIX_SIZE,
            _ => return Err(FormatError::InvalidAttributeVersion(version)),
        };
        bytes::ensure_len(data, name_start, name_size)?;
        Ok(extract_name(&data[name_start..name_start + name_size]))
    }

    /// Returns `true` if the flags byte of the attribute message in `data` marks its datatype or
    /// dataspace as shared.
    ///
    /// A shared field holds a reference to a message elsewhere in the file, so a caller checks
    /// this before it copies the message bytes to another file. Returns `false` for a version 1
    /// message, which has no flags byte. Returns `true` for a message too short to hold a flags
    /// byte and for a version other than 1, 2, or 3, so a caller rejects a message it cannot read.
    pub fn shares_a_field(data: &[u8]) -> bool {
        match data.first().copied() {
            // Version 1 has no flags byte. The field after the version is unused.
            Some(VERSION_ONE) => false,
            Some(VERSION_TWO | VERSION_THREE) => {
                data.get(1).is_none_or(|flags| flags & FLAG_ALL != 0)
            }
            _ => true,
        }
    }

    /// Parses an attribute message without access to the file it came from.
    ///
    /// A message whose datatype or dataspace is shared cannot be parsed this way. Use
    /// [`parse_resolving`](Self::parse_resolving) where the file is available. `length_size` is
    /// the width of a length the superblock stores.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnresolvedSharedMessage`] if the flags byte marks the datatype or
    /// dataspace as shared, and the errors of [`parse_resolving`](Self::parse_resolving) in the
    /// cases it lists.
    pub fn parse(data: &[u8], length_size: u8) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving(data, length_size, &Unresolvable)
    }

    /// Parses an attribute message, and resolves its datatype or dataspace through `resolver`
    /// where the flags byte marks the field as shared.
    ///
    /// `length_size` is the width of a length the superblock stores.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeVersion`] if the version is not 1, 2, or 3,
    /// [`FormatError::InvalidAttributeFlags`] if the flags byte sets a bit the format does not
    /// define, [`FormatError::UnexpectedEof`] if `data` ends inside a field,
    /// [`FormatError::OffsetOverflow`] if the element count times the element size overflows a
    /// `u64`, and [`FormatError::ValueTooLargeForPlatform`] if the product does not fit a `usize`.
    /// Returns the error of `resolver` if it fails, and the error of the datatype or dataspace
    /// parser if the field does not parse.
    pub fn parse_resolving(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving_at(data, length_size, resolver).map(|(attr, _)| attr)
    }

    /// Parses an attribute message as [`parse_resolving`](Self::parse_resolving) does, and returns
    /// the offset in `data` at which the element bytes start.
    ///
    /// A caller that rewrites an element in the file, such as a stored object reference, finds it
    /// through this offset. Version 1 pads the name, datatype, and dataspace fields to multiples of
    /// 8 bytes and versions 2 and 3 do not, so the offset comes from the same parse that reads
    /// [`raw_data`](Self::raw_data).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`parse_resolving`](Self::parse_resolving) in the cases it lists.
    pub fn parse_resolving_at(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        bytes::ensure_len(data, 0, 2)?;
        let version = data[0];

        match version {
            VERSION_ONE => Self::parse_v1(data, length_size),
            VERSION_TWO => Self::parse_v2(data, length_size, resolver),
            VERSION_THREE => Self::parse_v3(data, length_size, resolver),
            _ => Err(FormatError::InvalidAttributeVersion(version)),
        }
    }

    fn parse_v1(data: &[u8], length_size: u8) -> Result<(AttributeMessage, usize), FormatError> {
        // version(1) + reserved(1) + name_size(2) + datatype_size(2) + dataspace_size(2) = 8
        bytes::ensure_len(data, 0, V1_V2_PREFIX_SIZE)?;
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;

        let mut pos = V1_V2_PREFIX_SIZE;

        // Name (padded to 8-byte boundary)
        bytes::ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += pad8(name_size);

        // Datatype (padded to 8-byte boundary). Version 1 has no flags byte, so
        // neither field can be a reference.
        bytes::ensure_len(data, pos, datatype_size)?;
        let (datatype, _) = datatype::parse_datatype(&data[pos..pos + datatype_size])?;
        pos += pad8(datatype_size);

        // Dataspace (padded to 8-byte boundary)
        bytes::ensure_len(data, pos, dataspace_size)?;
        let dataspace = Dataspace::parse(&data[pos..pos + dataspace_size], length_size)?;
        pos += pad8(dataspace_size);

        // Raw data: `num_elements` × `type_size` bytes
        let raw_data = compute_raw_data(data, pos, &dataspace, &datatype)?;

        Ok((
            AttributeMessage {
                name,
                datatype,
                dataspace,
                raw_data,
                datatype_location: DatatypeLocation::Inline,
            },
            pos,
        ))
    }

    fn parse_v2(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        // version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2) = 8
        bytes::ensure_len(data, 0, V1_V2_PREFIX_SIZE)?;
        let flags = data[1];
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;

        let mut pos = V1_V2_PREFIX_SIZE;

        // Name (NO padding)
        bytes::ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += name_size;

        // Datatype (NO padding)
        bytes::ensure_len(data, pos, datatype_size)?;
        let dt_field = &data[pos..pos + datatype_size];
        pos += datatype_size;

        // Dataspace (NO padding)
        bytes::ensure_len(data, pos, dataspace_size)?;
        let ds_field = &data[pos..pos + dataspace_size];
        pos += dataspace_size;

        let (datatype, dataspace, datatype_location) =
            decode_type_and_space(dt_field, ds_field, flags, length_size, resolver)?;
        let raw_data = compute_raw_data(data, pos, &dataspace, &datatype)?;

        Ok((
            AttributeMessage {
                name,
                datatype,
                dataspace,
                raw_data,
                datatype_location,
            },
            pos,
        ))
    }

    fn parse_v3(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        // version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2) + encoding(1) = 9
        bytes::ensure_len(data, 0, V3_PREFIX_SIZE)?;
        let flags = data[1];
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;
        let _encoding = data[8]; // 0=ASCII, 1=UTF-8

        let mut pos = V3_PREFIX_SIZE;

        // Name (NO padding)
        bytes::ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += name_size;

        // Datatype (NO padding)
        bytes::ensure_len(data, pos, datatype_size)?;
        let dt_field = &data[pos..pos + datatype_size];
        pos += datatype_size;

        // Dataspace (NO padding)
        bytes::ensure_len(data, pos, dataspace_size)?;
        let ds_field = &data[pos..pos + dataspace_size];
        pos += dataspace_size;

        let (datatype, dataspace, datatype_location) =
            decode_type_and_space(dt_field, ds_field, flags, length_size, resolver)?;
        let raw_data = compute_raw_data(data, pos, &dataspace, &datatype)?;

        Ok((
            AttributeMessage {
                name,
                datatype,
                dataspace,
                raw_data,
                datatype_location,
            },
            pos,
        ))
    }

    /// Serializes the message as version 2.
    ///
    /// `offset_width` is the width of the address in a reference to a committed datatype, and
    /// `length_size` the width of a length, both as the superblock of the file the message goes
    /// into stores them. The message refers to a committed datatype where
    /// [`datatype_location`](Self::datatype_location) is committed, and encodes its dataspace in
    /// every case.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AttributeFieldTooLong`] if the name with its null terminator, the
    /// datatype, or the dataspace is longer than the 65,535 bytes its 2-byte size field can
    /// hold.
    pub fn serialize(
        &self,
        offset_width: OffsetWidth,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        self.serialize_version(VERSION_TWO, offset_width, length_size)
    }

    /// Serializes the message as version 3, which adds the character set of the name.
    ///
    /// The character set is written as ASCII. The widths and the fields are the same as for
    /// [`serialize`](Self::serialize).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AttributeFieldTooLong`] if the name with its null terminator, the
    /// datatype, or the dataspace is longer than the 65,535 bytes its 2-byte size field can
    /// hold.
    pub fn serialize_v3(
        &self,
        offset_width: OffsetWidth,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        self.serialize_version(VERSION_THREE, offset_width, length_size)
    }

    /// Returns the bytes of the message's datatype field: the encoding itself, or the
    /// reference standing in for it when the type is committed.
    ///
    /// A reference is written in `offset_width`, the width of the file the bytes
    /// are going into, and not in the width of whatever file the message was read
    /// from. Re-serializing a message parsed from a file with a different width is
    /// therefore a re-encoding, not a copy.
    fn datatype_field(&self, offset_width: OffsetWidth) -> Vec<u8> {
        match self.datatype_location.reference_bytes(offset_width) {
            Some(reference) => reference,
            None => datatype::serialize_datatype(&self.datatype),
        }
    }

    fn serialize_version(
        &self,
        version: u8,
        offset_width: OffsetWidth,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        let name_bytes = {
            let mut n = self.name.as_bytes().to_vec();
            n.push(0); // null terminator
            n
        };
        let dt_bytes = self.datatype_field(offset_width);
        let ds_bytes = self.dataspace.serialize(length_size);
        let name_size = self.field_size("name", &name_bytes)?;
        let datatype_size = self.field_size("datatype", &dt_bytes)?;
        let dataspace_size = self.field_size("dataspace", &ds_bytes)?;

        let mut buf = Vec::new();
        buf.push(version);
        // Version 1 has no flags byte, but nothing serializes one: both callers
        // request version 2 or 3, whose second byte says which fields are
        // references. Only the datatype is ever one here, since this crate does
        // not write a shared dataspace.
        buf.push(if self.datatype_location.is_committed() {
            FLAG_SHARED_DATATYPE
        } else {
            0
        });
        buf.extend_from_slice(&name_size.to_le_bytes());
        buf.extend_from_slice(&datatype_size.to_le_bytes());
        buf.extend_from_slice(&dataspace_size.to_le_bytes());
        if version >= VERSION_THREE {
            buf.push(name_character_set(CharacterSet::Ascii));
        }
        buf.extend_from_slice(&name_bytes);
        buf.extend_from_slice(&dt_bytes);
        buf.extend_from_slice(&ds_bytes);
        buf.extend_from_slice(&self.raw_data);
        Ok(buf)
    }

    fn field_size(&self, field: &'static str, encoded: &[u8]) -> Result<u16, FormatError> {
        encoded
            .len()
            .narrow_or_else(|| FormatError::AttributeFieldTooLong {
                name: self.name.clone(),
                field,
                size: encoded.len(),
                limit: usize::from(u16::MAX),
            })
    }
}

/// Decodes an attribute's datatype and dataspace fields, and resolves a field through `resolver`
/// where the flags byte marks it as shared.
///
/// Only the flags byte distinguishes a reference from an encoding: the datatype parser reads a
/// version 2 reference to address `0x320` as a time datatype, and rejects it only for its size of
/// zero.
///
/// Only the datatype's location is returned. A shared *dataspace* is resolved
/// and then indistinguishable from an inline one: a dataspace has no name, and no
/// HDF5 call reports one as shared, so writing it back inline loses nothing. A
/// committed datatype does have a name, which is why that one is tracked.
fn decode_type_and_space(
    dt_field: &[u8],
    ds_field: &[u8],
    flags: u8,
    length_size: u8,
    resolver: &dyn SharedResolver,
) -> Result<(Datatype, Dataspace, DatatypeLocation), FormatError> {
    // The C library rejects a flags byte that sets any other bit (`H5Oattr.c`, HDF5 2.1.0).
    if flags & !FLAG_ALL != 0 {
        return Err(FormatError::InvalidAttributeFlags(flags));
    }

    let (datatype, location) = if flags & FLAG_SHARED_DATATYPE != 0 {
        // A shared datatype is either a committed object every user names, or
        // one anonymous copy in the file's shared-message heap. Only the first
        // has an address, and only the first is a *named* type: a heap-stored
        // one has no path and no object, so it is reported as inline and written
        // back spelled out, which is what every reader of such a file already
        // shows.
        let address = resolver.committed_address(dt_field)?;
        let body = resolver.resolve(dt_field, MessageType::DATATYPE)?;
        (
            datatype::parse_datatype(&body)?.0,
            match address {
                Some(address) => DatatypeLocation::Committed(address),
                None => DatatypeLocation::Inline,
            },
        )
    } else {
        (
            datatype::parse_datatype(dt_field)?.0,
            DatatypeLocation::Inline,
        )
    };

    let dataspace = if flags & FLAG_SHARED_DATASPACE != 0 {
        let body = resolver.resolve(ds_field, MessageType::DATASPACE)?;
        Dataspace::parse(&body, length_size)?
    } else {
        Dataspace::parse(ds_field, length_size)?
    };

    Ok((datatype, dataspace, location))
}

/// Reads an attribute's value bytes from its message, sized by the datatype
/// and dataspace alone.
///
/// The Data field has no length of its own: the datatype and dataspace
/// descriptions define its size ("The Attribute Message" of the [format
/// specification, version 4.0][spec-attr]). A null dataspace has no elements
/// ("The Dataspace Message" of the [format specification, version 4.0][spec-space]),
/// so the field is empty. Any bytes past it in a version 1 object header
/// record are the record's alignment padding: header messages are aligned on
/// 8-byte boundaries there ("Version 1 Data Object Header Prefix" of the [format
/// specification, version 4.0][spec-hdr]).
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the element count times the
/// element size overflows a `u64`, [`FormatError::ValueTooLargeForPlatform`]
/// if that size does not fit a `usize`, and [`FormatError::UnexpectedEof`] if
/// the message holds fewer bytes than the datatype and dataspace describe.
///
/// [spec-attr]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attribute
/// [spec-space]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
/// [spec-hdr]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
fn compute_raw_data(
    data: &[u8],
    pos: usize,
    dataspace: &Dataspace,
    datatype: &Datatype,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = dataspace.num_elements();
    let elem_size = u64::from(datatype.type_size());
    let expected_size = num_elements
        .checked_mul(elem_size)
        .ok_or(FormatError::OffsetOverflow {
            offset: num_elements,
            length: elem_size,
        })?
        .to_usize()?;
    bytes::ensure_len(data, pos, expected_size)?;
    Ok(if expected_size > 0 {
        data.get(pos..pos + expected_size)
            .map(<[u8]>::to_vec)
            .unwrap_or_default()
    } else {
        Vec::new()
    })
}

/// Returns the name in `bytes` up to its first null byte, with invalid UTF-8 replaced.
fn extract_name(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Rounds the size of a version 1 field up to the next multiple of 8.
fn pad8(x: usize) -> usize {
    x.next_multiple_of(V1_FIELD_ALIGNMENT)
}

// The "Name Character Set Encoding" values of "The Attribute Message", format specification
// version 4.0.
fn name_character_set(charset: CharacterSet) -> u8 {
    match charset {
        CharacterSet::Ascii => 0,
        CharacterSet::Utf8 => 1,
    }
}

/// Bit 0 of an attribute message's flags byte: the datatype field holds a shared message
/// reference, to a committed datatype or into the shared-message heap, in place of the encoding
/// (`H5O_ATTR_FLAG_TYPE_SHARED`).
const FLAG_SHARED_DATATYPE: u8 = 0x01;

/// Bit 1: the same for the dataspace field (`H5O_ATTR_FLAG_SPACE_SHARED`).
const FLAG_SHARED_DATASPACE: u8 = 0x02;

/// Every flag bit the format defines (`H5O_ATTR_FLAG_ALL`).
const FLAG_ALL: u8 = FLAG_SHARED_DATATYPE | FLAG_SHARED_DATASPACE;

// The versions, the prefix sizes and the version 1 field alignment of "The Attribute Message",
// format specification version 4.0.
const VERSION_ONE: u8 = 1;
const VERSION_TWO: u8 = 2;
const VERSION_THREE: u8 = 3;
const V1_V2_PREFIX_SIZE: usize = 8;
const V3_PREFIX_SIZE: usize = 9;
const V1_FIELD_ALIGNMENT: usize = 8;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::attribute;
    use test_util::dataspace;
    use test_util::datatype;

    use super::*;
    use crate::address::StoredAddress;
    use crate::shared_message;

    #[rstest]
    #[case::version_1(
        attribute::Attribute::new(
            "temp",
            &datatype::f64_le(),
            &dataspace::scalar(),
            &98.6f64.to_le_bytes(),
        )
        .build(),
        "temp",
        98.6f64.to_le_bytes().to_vec()
    )]
    #[case::version_1_short_name(
        attribute::Attribute::new(
            "x",
            &datatype::f64_le(),
            &dataspace::scalar(),
            &42.0f64.to_le_bytes(),
        )
        .build(),
        "x",
        42.0f64.to_le_bytes().to_vec()
    )]
    #[case::version_2_fixed_string(
        attribute::Attribute::new("label", &datatype::fixed_string(5), &dataspace::scalar(), b"hello")
            .flags(attribute::Flags::NONE)
            .build(),
        "label",
        b"hello".to_vec()
    )]
    #[case::version_2_short_fields(
        attribute::Attribute::new("ab", &datatype::fixed_string(2), &dataspace::scalar(), b"hi")
            .flags(attribute::Flags::NONE)
            .build(),
        "ab",
        b"hi".to_vec()
    )]
    #[case::version_2_array(
        attribute::Attribute::new(
            "vals",
            &datatype::f64_le(),
            &dataspace::v1(1, dataspace::Flags::NONE, &[3], None),
            &[1.0f64, 2.0, 3.0].map(f64::to_le_bytes).concat(),
        )
        .flags(attribute::Flags::NONE)
        .build(),
        "vals",
        [1.0f64, 2.0, 3.0].map(f64::to_le_bytes).concat()
    )]
    #[case::version_3_utf8_name(
        attribute::Attribute::new("note", &datatype::fixed_string(3), &dataspace::scalar(), b"abc")
            .character_set(1)
            .build(),
        "note",
        b"abc".to_vec()
    )]
    fn a_message_parses_to_its_name_and_value(
        #[case] message: Vec<u8>,
        #[case] attr_name: &str,
        #[case] raw_data: Vec<u8>,
    ) {
        let attr = AttributeMessage::parse(&message, 8).unwrap();
        assert_eq!(
            (attr.name.as_str(), attr.raw_data, attr.datatype_location),
            (attr_name, raw_data, DatatypeLocation::Inline)
        );
    }

    #[test]
    fn a_one_byte_message_is_unexpected_eof() {
        let data = [1u8];
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: 2,
                available: 1
            }
        );
    }

    #[test]
    fn version_5_is_an_invalid_attribute_version() {
        let data = [5u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidAttributeVersion(5));
    }

    /// Returns a version 2 attribute message with the flags byte `flags`, whose datatype field is
    /// a 10-byte reference to the committed type at address `0x320`, laid out as libhdf5 1.14.6
    /// wrote one.
    fn attr_with_committed_reference(flags: u8) -> Vec<u8> {
        attribute::Attribute::new(
            "shared_attr",
            &datatype::committed_reference(0x320),
            &dataspace::v1(1, dataspace::Flags::NONE, &[1], None),
            &7.0f64.to_le_bytes(),
        )
        .flags(attribute::Flags(flags))
        .build()
    }

    /// A resolver that returns one fixed message body, standing in for the
    /// object header a committed datatype lives in.
    struct StubResolver(Vec<u8>);

    impl SharedResolver for StubResolver {
        fn resolve(&self, _reference: &[u8], _target: MessageType) -> Result<Vec<u8>, FormatError> {
            Ok(self.0.clone())
        }

        fn committed_address(
            &self,
            reference: &[u8],
        ) -> Result<Option<StoredAddress>, FormatError> {
            shared_message::committed_address_in(reference, 8, 8)
        }
    }

    #[test]
    fn a_shared_datatype_field_is_resolved_not_decoded() {
        let data = attr_with_committed_reference(FLAG_SHARED_DATATYPE);
        let attr =
            AttributeMessage::parse_resolving(&data, 8, &StubResolver(datatype::f64_le())).unwrap();

        assert_eq!(attr.name, "shared_attr");
        assert!(matches!(
            attr.datatype,
            Datatype::FloatingPoint { size: 8, .. }
        ));
        assert_eq!(
            attr.datatype_location,
            DatatypeLocation::Committed(StoredAddress::new(0x320)),
            "the attribute must record which committed object it named, not just that it named one"
        );
    }

    #[test]
    fn the_same_bytes_without_the_flag_are_rejected_as_a_zero_width_type() {
        let data = attr_with_committed_reference(0);
        let err = AttributeMessage::parse(&data, 8).unwrap_err();

        assert_eq!(
            err,
            FormatError::ZeroSizedDatatype { class: 2 },
            "the reference bytes decode as a class 2 (time) type of zero width"
        );
    }

    #[test]
    fn a_datatype_field_without_the_flag_is_recorded_as_inline() {
        let name = b"inline_attr\0";
        let dt_bytes = datatype::f64_le();
        let ds_bytes = dataspace::v1(1, dataspace::Flags::NONE, &[1], None);

        let mut data = vec![2u8, 0];
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(&1.5f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();

        assert_eq!(attr.name, "inline_attr");
        assert!(matches!(
            attr.datatype,
            Datatype::FloatingPoint { size: 8, .. }
        ));
        assert_eq!(attr.datatype_location, DatatypeLocation::Inline);
    }

    #[test]
    fn a_shared_datatype_field_is_rejected_without_a_resolver() {
        let data = attr_with_committed_reference(FLAG_SHARED_DATATYPE);
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnresolvedSharedMessage(MessageType::DATATYPE.to_u16())
        );
    }

    #[test]
    fn an_undefined_attribute_flag_bit_is_rejected() {
        let data = attr_with_committed_reference(0x04);
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidAttributeFlags(0x04));
    }

    #[test]
    fn a_name_reads_out_of_a_message_whose_datatype_is_a_reference() {
        let data = attr_with_committed_reference(FLAG_SHARED_DATATYPE);
        assert_eq!(AttributeMessage::parse_name(&data).unwrap(), "shared_attr");
        assert_eq!(
            AttributeMessage::parse(&data, 8),
            Err(FormatError::UnresolvedSharedMessage(
                MessageType::DATATYPE.to_u16()
            ))
        );
    }

    #[rstest]
    #[case::shared_datatype(attr_with_committed_reference(FLAG_SHARED_DATATYPE), true)]
    #[case::shared_dataspace(attr_with_committed_reference(FLAG_SHARED_DATASPACE), true)]
    #[case::no_shared_field(attr_with_committed_reference(0), false)]
    #[case::version_1_second_byte_unused(vec![1, 0xFF, 0, 0], false)]
    #[case::truncated(vec![2], true)]
    #[case::empty(Vec::new(), true)]
    #[case::unknown_version(vec![9, 0], true)]
    fn shares_a_field_reads_the_flags_byte(#[case] message: Vec<u8>, #[case] shares_a_field: bool) {
        assert_eq!(AttributeMessage::shares_a_field(&message), shares_a_field);
    }

    #[test]
    fn a_null_dataspace_attribute_ignores_record_padding() {
        // Five trailing zero bytes provide the record's 8-byte alignment padding.
        let name = b"empty\0";
        let dt_bytes = datatype::f64_le();
        let ds_bytes = vec![2u8, 0, 0, 2];

        let name_size = u16::try_from(name.len()).unwrap();
        let dt_size = u16::try_from(dt_bytes.len()).unwrap();
        let ds_size = u16::try_from(ds_bytes.len()).unwrap();

        let mut data = Vec::new();
        data.push(3);
        data.push(0);
        data.extend_from_slice(&name_size.to_le_bytes());
        data.extend_from_slice(&dt_size.to_le_bytes());
        data.extend_from_slice(&ds_size.to_le_bytes());
        data.push(1);
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(&[0u8; 5]);

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.raw_data, Vec::<u8>::new());
    }

    #[test]
    fn a_truncated_attribute_payload_is_rejected() {
        let name = b"truncated\0";
        let dt_bytes = datatype::f64_le();
        let ds_bytes = dataspace::scalar();

        let name_size = u16::try_from(name.len()).unwrap();
        let dt_size = u16::try_from(dt_bytes.len()).unwrap();
        let ds_size = u16::try_from(ds_bytes.len()).unwrap();

        let mut data = Vec::new();
        data.push(3);
        data.push(0);
        data.extend_from_slice(&name_size.to_le_bytes());
        data.extend_from_slice(&dt_size.to_le_bytes());
        data.extend_from_slice(&ds_size.to_le_bytes());
        data.push(1);
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(&[0u8; 4]);

        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        let FormatError::UnexpectedEof {
            expected,
            available,
        } = err
        else {
            panic!("expected UnexpectedEof, got {err:?}");
        };
        assert_eq!(available, data.len());
        assert_eq!(expected, data.len() + 4);
    }
}
