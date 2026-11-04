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
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

/// An Attribute message: the name, datatype, dataspace, and data of one attribute.
///
/// The parsers read message versions 1, 2, and 3. [`serialize`](Self::serialize) writes version 2
/// for ASCII names and version 3 for non-ASCII names. [`serialize_v3`](Self::serialize_v3) writes
/// version 3 for either character set. The message is defined in "The Attribute Message" of the
/// [format specification, version 4.0][spec].
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
    /// stored element size. Variable-length references and arrays of them occupy a size that
    /// depends on the file's address width.
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

    /// Borrows the datatype field when the attribute marks it as a shared reference.
    ///
    /// Checks field boundaries and flags for versions 2 and 3. Returns `None` for an inline
    /// datatype or a version 1 message with a complete fixed prefix. The version 1 result
    /// requires only that prefix.
    ///
    /// The returned bytes can refer to a committed datatype or the shared-message heap. Reference
    /// decoding, datatype and dataspace decoding, and value validation remain with the caller.
    /// Name and value bytes remain uninterpreted.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeVersion`] if the version is not 1, 2, or 3,
    /// [`FormatError::InvalidAttributeFlags`] if a version 2 or 3 flags byte sets an undefined bit,
    /// and [`FormatError::UnexpectedEof`] if `data` ends within the required prefix or a version
    /// 2 or 3 name, datatype, or dataspace field.
    pub fn shared_datatype_reference(data: &[u8]) -> Result<Option<&[u8]>, FormatError> {
        bytes::ensure_len(data, 0, 2)?;
        if data[0] == VERSION_ONE {
            bytes::ensure_len(data, 0, V1_V2_PREFIX_SIZE)?;
            return Ok(None);
        }
        let prefix = AttributePrefix::parse(data)?;
        prefix.dataspace(data)?;
        if prefix.flags & !FLAG_ALL != 0 {
            return Err(FormatError::InvalidAttributeFlags(prefix.flags));
        }
        Ok((prefix.flags & FLAG_SHARED_DATATYPE != 0).then_some(prefix.datatype))
    }

    /// Parses an attribute message without access to the file it came from.
    ///
    /// A message whose datatype or dataspace is shared cannot be parsed this way. Use
    /// [`parse_resolving`](Self::parse_resolving) where the file is available.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8. Returns
    /// [`FormatError::UnresolvedSharedMessage`] if the flags byte marks the datatype or dataspace
    /// as shared, and the errors of [`parse_with_length_width`](Self::parse_with_length_width) in
    /// the cases it lists.
    pub fn parse(data: &[u8], length_size: u8) -> Result<AttributeMessage, FormatError> {
        Self::parse_with_length_width(data, LengthWidth::try_from(length_size)?)
    }

    /// Parses an attribute message using the file's validated length width.
    ///
    /// A message whose datatype or dataspace is shared cannot be parsed this way. Use
    /// [`parse_resolving_with_length_width`](Self::parse_resolving_with_length_width) where the
    /// file is available. Assumes eight-byte file addresses for variable-length element storage.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnresolvedSharedMessage`] if the flags byte marks the datatype or
    /// dataspace as shared, and the errors of
    /// [`parse_resolving_with_length_width`](Self::parse_resolving_with_length_width) in the cases
    /// it lists.
    pub fn parse_with_length_width(
        data: &[u8],
        length_width: LengthWidth,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving_with_length_width(data, length_width, &Unresolvable)
    }

    /// Reads an attribute's datatype before parsing its dataspace or copying its values.
    ///
    /// Resolves a shared datatype through `resolver`. Checks that the prefix, name, and datatype
    /// fields fit in `data`, using version 1 field padding to locate the datatype. The dataspace
    /// and value bodies may be absent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeVersion`] for an unsupported version,
    /// [`FormatError::InvalidAttributeFlags`] for unsupported flags, or
    /// [`FormatError::UnexpectedEof`] if the prefix, name, or datatype is truncated. Propagates
    /// datatype-decoding and resolver errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private as format, Datatype, FormatError, LengthWidth,
    ///     MessageType, OffsetWidth, ReferenceType};
    /// use hdf5_pure_core::StoredAddress;
    ///
    /// struct InlineOnly;
    /// impl format::SharedResolver for InlineOnly {
    ///     fn resolve(&self, _: &[u8], target: MessageType)
    ///         -> Result<Vec<u8>, FormatError>
    ///     {
    ///         Err(FormatError::UnresolvedSharedMessage(target.to_u16()))
    ///     }
    ///     fn committed_address(&self, _: &[u8]) -> Result<Option<StoredAddress>, FormatError> {
    ///         Err(FormatError::UnresolvedSharedMessage(MessageType::DATATYPE.to_u16()))
    ///     }
    /// }
    ///
    /// let datatype = Datatype::Reference { size: 8, ref_type: ReferenceType::Object };
    /// let attribute = format::AttributeMessage {
    ///     name: "target".into(),
    ///     datatype: datatype.clone(),
    ///     dataspace: format::Dataspace {
    ///         space_type: format::DataspaceType::Scalar,
    ///         rank: 0,
    ///         dimensions: vec![],
    ///         max_dimensions: None,
    ///     },
    ///     raw_data: vec![],
    ///     datatype_location: format::DatatypeLocation::Inline,
    /// };
    /// let body = attribute.serialize(OffsetWidth::Two, 4)?;
    /// assert_eq!(
    ///     format::AttributeMessage::parse_datatype_resolving(&body, &InlineOnly)?,
    ///     datatype,
    /// );
    /// let error = format::AttributeMessage::parse_inspecting_datatype(
    ///     &body,
    ///     format::FormatWidths { offsets: OffsetWidth::Two, lengths: LengthWidth::Four },
    ///     &InlineOnly,
    ///     |datatype| match datatype {
    ///         Datatype::Reference { .. } => Err(FormatError::ReferenceWidthUnsupported {
    ///             offset_size: 2,
    ///         }),
    ///         _ => Ok(()),
    ///     },
    /// ).unwrap_err();
    /// assert_eq!(error, FormatError::ReferenceWidthUnsupported { offset_size: 2 });
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn parse_datatype_resolving(
        data: &[u8],
        resolver: &dyn SharedResolver,
    ) -> Result<Datatype, FormatError> {
        let prefix = AttributePrefix::parse(data)?;
        decode_datatype(prefix.datatype, prefix.flags, resolver).map(|(datatype, _)| datatype)
    }

    /// Parses an attribute body with independently selected address and length widths.
    ///
    /// Reads dataspace dimensions at `widths.lengths` and sizes variable-length element storage
    /// at `widths.offsets`. Shared datatype or dataspace fields require
    /// [`Self::parse_resolving_with_widths`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnresolvedSharedMessage`] if either field is shared, or the parsing
    /// errors of [`Self::parse_resolving_at_with_widths`].
    pub fn parse_with_widths(
        data: &[u8],
        widths: FormatWidths,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving_with_widths(data, widths, &Unresolvable)
    }

    /// Parses an attribute body and resolves its shared datatype or dataspace fields.
    ///
    /// The caller supplies a `resolver` for the same file as `widths`. Reads variable-length
    /// element storage at `widths.offsets` and dataspace dimensions at `widths.lengths`.
    ///
    /// # Errors
    ///
    /// Returns the parsing and resolution errors of [`Self::parse_resolving_at_with_widths`].
    pub fn parse_resolving_with_widths(
        data: &[u8],
        widths: FormatWidths,
        resolver: &dyn SharedResolver,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving_at_with_widths(data, widths, resolver).map(|(message, _)| message)
    }

    /// Parses an attribute message, and resolves its datatype or dataspace through `resolver`
    /// where the flags byte marks the field as shared.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8. Returns the
    /// errors of
    /// [`parse_resolving_with_length_width`](Self::parse_resolving_with_length_width) in the cases
    /// it lists.
    pub fn parse_resolving(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving_with_length_width(data, LengthWidth::try_from(length_size)?, resolver)
    }

    /// Parses an attribute message using the file's validated length width.
    ///
    /// The parser resolves a datatype or dataspace through `resolver` where the flags byte marks
    /// the field as shared. Assumes eight-byte file addresses for variable-length element storage.
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
    pub fn parse_resolving_with_length_width(
        data: &[u8],
        length_width: LengthWidth,
        resolver: &dyn SharedResolver,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving_at_with_length_width(data, length_width, resolver)
            .map(|(attr, _)| attr)
    }

    /// Parses an attribute message and returns the offset at which the element bytes start.
    ///
    /// A caller that rewrites an element in the file, such as a stored object reference, finds it
    /// through this offset. Version 1 pads the name, datatype, and dataspace fields to multiples of
    /// 8 bytes and versions 2 and 3 do not, so the offset comes from the same parse that reads
    /// [`raw_data`](Self::raw_data).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8. Returns the
    /// errors of
    /// [`parse_resolving_at_with_length_width`](Self::parse_resolving_at_with_length_width) in the
    /// cases it lists.
    pub fn parse_resolving_at(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        Self::parse_resolving_at_with_length_width(
            data,
            LengthWidth::try_from(length_size)?,
            resolver,
        )
    }

    /// Parses an attribute message and returns the offset at which the element bytes start.
    ///
    /// The parser uses the file's validated length width and resolves shared datatype or dataspace
    /// fields through `resolver`. Assumes eight-byte file addresses for variable-length elements.
    ///
    /// # Errors
    ///
    /// Returns the errors of
    /// [`parse_resolving_with_length_width`](Self::parse_resolving_with_length_width) in the cases
    /// it lists.
    pub fn parse_resolving_at_with_length_width(
        data: &[u8],
        length_width: LengthWidth,
        resolver: &dyn SharedResolver,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        Self::parse_resolving_at_with_widths(
            data,
            FormatWidths {
                offsets: OffsetWidth::Eight,
                lengths: length_width,
            },
            resolver,
        )
    }

    /// Parses an attribute body and returns the byte offset of its element storage.
    ///
    /// Resolves shared datatype and dataspace fields through `resolver`. Reads dimensions at
    /// `widths.lengths` and sizes variable-length element storage at `widths.offsets`. The returned
    /// offset is relative to the start of `data`, including any version 1 field padding.
    /// Checks complete datatype and dataspace field ranges before decoding or resolving them.
    /// [`Self::parse_inspecting_datatype`] supports earlier datatype screening.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeVersion`] if the version is unsupported,
    /// [`FormatError::InvalidAttributeFlags`] for unsupported flags, or
    /// [`FormatError::UnexpectedEof`] if a field is truncated. Returns
    /// [`FormatError::OffsetOverflow`] if the element count times its stored size exceeds
    /// [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if it exceeds [`usize::MAX`].
    /// Propagates datatype, dataspace, stored-element-size, and resolver errors.
    pub fn parse_resolving_at_with_widths(
        data: &[u8],
        widths: FormatWidths,
        resolver: &dyn SharedResolver,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        let prefix = AttributePrefix::parse(data)?;
        prefix.dataspace(data)?;
        Self::parse_after_prefix(data, widths, resolver, prefix, |_| Ok(()))
    }

    /// Parses an attribute after inspecting its resolved datatype.
    ///
    /// Calls `inspect` once after decoding the datatype, before reading the dataspace body or
    /// copying values. Resolves shared fields once each. If `inspect` returns an error, returns
    /// that error immediately. Reads accepted values at `widths` as in
    /// [`Self::parse_resolving_with_widths`].
    ///
    /// # Errors
    ///
    /// Returns the error from `inspect` or the parsing and resolution errors of
    /// [`Self::parse_resolving_at_with_widths`].
    ///
    /// # Examples
    ///
    /// See [`Self::parse_datatype_resolving`] for datatype inspection before reading values.
    pub fn parse_inspecting_datatype(
        data: &[u8],
        widths: FormatWidths,
        resolver: &dyn SharedResolver,
        inspect: impl FnOnce(&Datatype) -> Result<(), FormatError>,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_inspecting_datatype_at(data, widths, resolver, inspect)
            .map(|(attribute, _)| attribute)
    }

    /// Inspects the datatype before parsing the remaining fields and returns the value offset.
    ///
    /// # Errors
    ///
    /// Returns the inspection and parsing errors of [`Self::parse_inspecting_datatype`].
    fn parse_inspecting_datatype_at(
        data: &[u8],
        widths: FormatWidths,
        resolver: &dyn SharedResolver,
        inspect: impl FnOnce(&Datatype) -> Result<(), FormatError>,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        Self::parse_after_prefix(
            data,
            widths,
            resolver,
            AttributePrefix::parse(data)?,
            inspect,
        )
    }

    fn parse_after_prefix(
        data: &[u8],
        widths: FormatWidths,
        resolver: &dyn SharedResolver,
        prefix: AttributePrefix<'_>,
        inspect: impl FnOnce(&Datatype) -> Result<(), FormatError>,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        let (datatype, datatype_location) =
            decode_datatype(prefix.datatype, prefix.flags, resolver)?;
        inspect(&datatype)?;
        let (ds_field, raw_at) = prefix.dataspace(data)?;
        let dataspace = decode_dataspace(ds_field, prefix.flags, widths.lengths, resolver)?;
        let raw_data = compute_raw_data(data, raw_at, &dataspace, &datatype, widths.offsets)?;
        Ok((
            AttributeMessage {
                name: extract_name(prefix.name),
                datatype,
                dataspace,
                raw_data,
                datatype_location,
            },
            raw_at,
        ))
    }

    /// Serializes the message as version 2 for an ASCII name or version 3 otherwise.
    ///
    /// Writes the UTF-8 character-set declaration for a non-ASCII name.
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
    /// hold. Returns [`FormatError::OpaqueTagTooLong`] if an inline datatype contains an opaque
    /// tag longer than 247 bytes, or [`FormatError::InvalidOpaqueTagByte`] if such a tag contains
    /// a NUL byte. Returns the errors of [`Dataspace::serialize`] for dimensions that do not fit
    /// `length_size` bytes or an unsupported `length_size`, and the errors of
    /// [`DatatypeLocation::reference_bytes`] for a committed-datatype address that does not fit
    /// `offset_width` or equals its undefined marker.
    pub fn serialize(
        &self,
        offset_width: OffsetWidth,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        self.serialize_version(VERSION_TWO, offset_width, length_size)
    }

    /// Serializes the message as version 3 with its name's character set.
    ///
    /// Writes ASCII for an ASCII name and UTF-8 otherwise. The widths and the fields are the same
    /// as for [`serialize`](Self::serialize).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AttributeFieldTooLong`] if the name with its null terminator, the
    /// datatype, or the dataspace is longer than the 65,535 bytes its 2-byte size field can
    /// hold. Returns [`FormatError::OpaqueTagTooLong`] if an inline datatype contains an opaque
    /// tag longer than 247 bytes, or [`FormatError::InvalidOpaqueTagByte`] if such a tag contains
    /// a NUL byte. Returns the errors of [`Dataspace::serialize`] for dimensions that do not fit
    /// `length_size` bytes or an unsupported `length_size`, and the errors of
    /// [`DatatypeLocation::reference_bytes`] for a committed-datatype address that does not fit
    /// `offset_width` or equals its undefined marker.
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
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OpaqueTagTooLong`] if an inline datatype contains an opaque tag
    /// longer than 247 bytes, or [`FormatError::InvalidOpaqueTagByte`] if such a tag contains a NUL.
    /// Returns the errors of [`DatatypeLocation::reference_bytes`] for a committed-datatype
    /// address that exceeds `offset_width` or equals its undefined marker.
    fn datatype_field(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, FormatError> {
        match self.datatype_location.reference_bytes(offset_width)? {
            Some(reference) => Ok(reference),
            None => datatype::serialize_datatype_with_offset_width(&self.datatype, offset_width),
        }
    }

    fn serialize_version(
        &self,
        version: u8,
        offset_width: OffsetWidth,
        length_size: u8,
    ) -> Result<Vec<u8>, FormatError> {
        let charset = if self.name.is_ascii() {
            CharacterSet::Ascii
        } else {
            CharacterSet::Utf8
        };
        let version = if charset == CharacterSet::Utf8 {
            VERSION_THREE
        } else {
            version
        };
        let name_bytes = {
            let mut n = self.name.as_bytes().to_vec();
            n.push(0); // null terminator
            n
        };
        let dt_bytes = self.datatype_field(offset_width)?;
        let ds_bytes = self.dataspace.serialize(length_size)?;
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
            buf.push(name_character_set(charset));
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

/// An [`AttributeMessage`] paired with its encoded body and file widths.
///
/// The message and body are immutable until consumed. Compact writers use the body directly.
/// Dense writers use [`PreparedDenseAttrs::from_prepared`][dense] to prepare version 3 messages
/// and heap geometry at the retained file widths. [`Self::new`] selects eight-byte addresses and
/// lengths, and [`Self::with_widths`] accepts each width independently.
///
/// [dense]: crate::dense_attribute_write::PreparedDenseAttrs::from_prepared
pub struct PreparedAttributeMessage {
    widths: FormatWidths,
    message: AttributeMessage,
    body: Vec<u8>,
}

impl PreparedAttributeMessage {
    /// Serializes `message` with eight-byte addresses and lengths.
    ///
    /// # Errors
    ///
    /// Returns the encoding errors documented by [`AttributeMessage::serialize`].
    pub fn new(message: AttributeMessage) -> Result<Self, FormatError> {
        Self::with_widths(
            FormatWidths {
                offsets: OffsetWidth::Eight,
                lengths: LengthWidth::Eight,
            },
            message,
        )
    }

    /// Serializes `message` and retains the selected widths for dense-storage planning.
    ///
    /// The caller supplies raw element bytes already encoded at `widths.offsets`. This operation
    /// encodes metadata and copies those bytes into the message body.
    ///
    /// # Errors
    ///
    /// Returns the encoding errors of [`AttributeMessage::serialize`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private as format, Datatype, FormatError, LengthWidth,
    ///     OffsetWidth, VariableLengthType};
    ///
    /// let widths = format::FormatWidths {
    ///     offsets: OffsetWidth::Two,
    ///     lengths: LengthWidth::Four,
    /// };
    /// let message = format::AttributeMessage {
    ///     name: "empty".into(),
    ///     datatype: Datatype::VariableLength(VariableLengthType::sequence(Datatype::Opaque {
    ///         size: 1,
    ///         tag: vec![],
    ///     })),
    ///     dataspace: format::Dataspace {
    ///         space_type: format::DataspaceType::Scalar,
    ///         rank: 0,
    ///         dimensions: vec![],
    ///         max_dimensions: None,
    ///     },
    ///     raw_data: vec![0; 10],
    ///     datatype_location: format::DatatypeLocation::Inline,
    /// };
    /// let prepared = format::PreparedAttributeMessage::with_widths(widths, message.clone())?;
    /// let parsed = format::AttributeMessage::parse_with_widths(prepared.body(), widths)?;
    /// assert_eq!(parsed, message);
    /// assert_eq!(parsed.raw_data.len(), 10);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn with_widths(
        widths: FormatWidths,
        message: AttributeMessage,
    ) -> Result<Self, FormatError> {
        let body = message.serialize(widths.offsets, widths.lengths.get())?;
        Ok(Self {
            widths,
            message,
            body,
        })
    }

    /// Returns the file widths used to serialize the retained body.
    pub(crate) fn widths(&self) -> FormatWidths {
        self.widths
    }

    /// Returns the message used to encode the body.
    pub fn message(&self) -> &AttributeMessage {
        &self.message
    }

    /// Returns the body encoded by [`AttributeMessage::serialize`].
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    /// Returns the message and its encoded body, consuming the preparation.
    pub fn into_parts(self) -> (AttributeMessage, Vec<u8>) {
        (self.message, self.body)
    }

    /// Returns the message and its version 3 body, reusing the encoded fields.
    pub(crate) fn into_v3_parts(mut self) -> (AttributeMessage, Vec<u8>) {
        // Version 3 adds the name character set after the version 2 prefix, as defined in
        // "The Attribute Message" of the format specification, version 4.0.
        if self.body[0] == VERSION_TWO {
            self.body[0] = VERSION_THREE;
            self.body
                .insert(V1_V2_PREFIX_SIZE, name_character_set(CharacterSet::Ascii));
        }
        (self.message, self.body)
    }
}

/// Borrowed name and datatype fields with the position and size of the following dataspace.
///
/// Reading this prefix checks through the datatype only. The dataspace body is checked by
/// [`Self::dataspace`].
struct AttributePrefix<'a> {
    flags: u8,
    name: &'a [u8],
    datatype: &'a [u8],
    dataspace_size: usize,
    dataspace_at: usize,
    padded: bool,
}

impl<'a> AttributePrefix<'a> {
    /// Reads the fixed prefix, name, and datatype fields without reading the dataspace body.
    ///
    /// Version 1 fields are padded to multiples of eight bytes, as defined in "The Attribute
    /// Message" of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeVersion`] if the version is unsupported, or
    /// [`FormatError::UnexpectedEof`] if a required field is truncated.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attribute
    fn parse(data: &'a [u8]) -> Result<Self, FormatError> {
        bytes::ensure_len(data, 0, 2)?;
        let (prefix_size, padded, flags) = match data[0] {
            VERSION_ONE => (V1_V2_PREFIX_SIZE, true, 0),
            VERSION_TWO => (V1_V2_PREFIX_SIZE, false, data[1]),
            VERSION_THREE => (V3_PREFIX_SIZE, false, data[1]),
            version => return Err(FormatError::InvalidAttributeVersion(version)),
        };
        bytes::ensure_len(data, 0, prefix_size)?;
        let name_size = usize::from(u16::from_le_bytes([data[2], data[3]]));
        let datatype_size = usize::from(u16::from_le_bytes([data[4], data[5]]));
        let dataspace_size = usize::from(u16::from_le_bytes([data[6], data[7]]));
        bytes::ensure_len(data, prefix_size, name_size)?;
        let name = &data[prefix_size..prefix_size + name_size];
        let datatype_at = prefix_size + if padded { pad8(name_size) } else { name_size };
        bytes::ensure_len(data, datatype_at, datatype_size)?;
        let datatype = &data[datatype_at..datatype_at + datatype_size];
        let dataspace_at = datatype_at
            + if padded {
                pad8(datatype_size)
            } else {
                datatype_size
            };
        Ok(Self {
            flags,
            name,
            datatype,
            dataspace_size,
            dataspace_at,
            padded,
        })
    }

    /// Borrows the dataspace field and returns the byte offset of the values after its padding.
    ///
    /// The caller passes the same message used by [`Self::parse`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the dataspace field is truncated.
    fn dataspace(&self, data: &'a [u8]) -> Result<(&'a [u8], usize), FormatError> {
        bytes::ensure_len(data, self.dataspace_at, self.dataspace_size)?;
        let field = &data[self.dataspace_at..self.dataspace_at + self.dataspace_size];
        let raw_at = self.dataspace_at
            + if self.padded {
                pad8(self.dataspace_size)
            } else {
                self.dataspace_size
            };
        Ok((field, raw_at))
    }
}

/// Parses an inline dataspace or resolves its shared encoding at `length_width`.
///
/// # Errors
///
/// Returns dataspace-parsing or resolver errors.
fn decode_dataspace(
    field: &[u8],
    flags: u8,
    length_width: LengthWidth,
    resolver: &dyn SharedResolver,
) -> Result<Dataspace, FormatError> {
    if flags & FLAG_SHARED_DATASPACE != 0 {
        let body = resolver.resolve(field, MessageType::DATASPACE)?;
        Dataspace::parse_with_length_width(&body, length_width)
    } else {
        Dataspace::parse_with_length_width(field, length_width)
    }
}

/// Parses an inline or shared datatype and retains its committed-object location.
///
/// Resolves a shared datatype once. Shared-heap datatypes are reported as inline locations.
///
/// # Errors
///
/// Returns [`FormatError::InvalidAttributeFlags`] if an unsupported flag is set. Propagates
/// datatype-decoding and resolver errors.
fn decode_datatype(
    dt_field: &[u8],
    flags: u8,
    resolver: &dyn SharedResolver,
) -> Result<(Datatype, DatatypeLocation), FormatError> {
    // libhdf5 rejects a flags byte that sets any other bit (`H5Oattr.c`, HDF5 2.1.0).
    if flags & !FLAG_ALL != 0 {
        return Err(FormatError::InvalidAttributeFlags(flags));
    }

    Ok(if flags & FLAG_SHARED_DATATYPE != 0 {
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
    })
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
    offsets: OffsetWidth,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = dataspace.num_elements();
    let elem_size = u64::from(datatype::stored_type_size(datatype, offsets)?);
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
    use core::cell::Cell;

    use hdf5_pure_core::FixedPointType;
    use rstest::rstest;
    use test_util::attribute;
    use test_util::dataspace;
    use test_util::datatype;

    use super::*;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::dataspace::DataspaceType;
    use crate::datatype::StringPadding;
    use crate::datatype::byte_order::DatatypeByteOrder;
    use crate::shared_message;

    #[rstest]
    #[case::version_one(AttributeVersion::One, attribute::Flags::NONE, false)]
    #[case::inline_v2(AttributeVersion::Two, attribute::Flags::NONE, false)]
    #[case::inline_v3(AttributeVersion::Three, attribute::Flags::NONE, false)]
    #[case::datatype_v2(AttributeVersion::Two, attribute::Flags::SHARED_DATATYPE, true)]
    #[case::datatype_v3(AttributeVersion::Three, attribute::Flags::SHARED_DATATYPE, true)]
    #[case::dataspace_only(AttributeVersion::Two, attribute::Flags::SHARED_DATASPACE, false)]
    #[case::both_fields(
        AttributeVersion::Three,
        attribute::Flags::SHARED_DATATYPE | attribute::Flags::SHARED_DATASPACE,
        true
    )]
    fn shared_datatype_reference_borrows_the_flagged_field(
        #[case] version: AttributeVersion,
        #[case] flags: attribute::Flags,
        #[case] shared: bool,
    ) {
        let reference = datatype::committed_reference(0x320);
        let dataspace = dataspace::scalar();
        let fixture = attribute::Attribute::new("type", &reference, &dataspace, &[]);
        let fixture = match version {
            AttributeVersion::One => fixture,
            AttributeVersion::Two => fixture.flags(flags),
            AttributeVersion::Three => fixture.flags(flags).character_set(ASCII_NAME_CHARACTER_SET),
        };
        let bytes = fixture.build();
        let found = AttributeMessage::shared_datatype_reference(&bytes).unwrap();
        assert_eq!(found, shared.then_some(reference.as_slice()));
        if let Some(found) = found {
            let start = if version == AttributeVersion::Three {
                V3_PREFIX_SIZE
            } else {
                V1_V2_PREFIX_SIZE
            } + "type".len()
                + 1;
            assert_eq!(found.as_ptr(), bytes[start..].as_ptr());
        }
    }

    #[rstest]
    #[case::v2_prefix(VERSION_TWO, 7, 8)]
    #[case::v3_prefix(VERSION_THREE, 8, 9)]
    #[case::v2_name(VERSION_TWO, 9, 10)]
    #[case::v3_name(VERSION_THREE, 10, 11)]
    #[case::v2_datatype(VERSION_TWO, 12, 13)]
    #[case::v3_datatype(VERSION_THREE, 13, 14)]
    #[case::v2_dataspace(VERSION_TWO, 14, 15)]
    #[case::v3_dataspace(VERSION_THREE, 15, 16)]
    fn borrowed_and_owned_attribute_fields_report_the_same_truncation(
        #[case] version: u8,
        #[case] available: usize,
        #[case] expected: usize,
    ) {
        let fixture = attribute::Attribute::new("a", &[0xa5; 3], &[0xc3; 2], &[])
            .flags(attribute::Flags::SHARED_DATATYPE);
        let mut bytes = if version == VERSION_THREE {
            fixture.character_set(ASCII_NAME_CHARACTER_SET).build()
        } else {
            fixture.build()
        };
        bytes.truncate(available);
        let error = FormatError::UnexpectedEof {
            expected,
            available,
        };
        assert_eq!(
            AttributeMessage::shared_datatype_reference(&bytes).unwrap_err(),
            error
        );
        assert_eq!(AttributeMessage::parse(&bytes, 8).unwrap_err(), error);
    }

    #[rstest]
    #[case::empty(Vec::new(), FormatError::UnexpectedEof { expected: 2, available: 0 })]
    #[case::version_only(vec![VERSION_ONE], FormatError::UnexpectedEof { expected: 2, available: 1 })]
    #[case::short_v1(vec![VERSION_ONE, 0], FormatError::UnexpectedEof { expected: 8, available: 2 })]
    #[case::unknown_version(vec![5, 0], FormatError::InvalidAttributeVersion(5))]
    #[case::unknown_flags(
        attribute::Attribute::new("a", &[], &[], &[]).flags(attribute::Flags(0x80)).build(),
        FormatError::InvalidAttributeFlags(0x80)
    )]
    fn borrowed_and_owned_attributes_report_the_same_prefix_error(
        #[case] bytes: Vec<u8>,
        #[case] error: FormatError,
    ) {
        assert_eq!(
            AttributeMessage::shared_datatype_reference(&bytes).unwrap_err(),
            error
        );
        assert_eq!(AttributeMessage::parse(&bytes, 8).unwrap_err(), error);
    }

    #[test]
    fn shared_datatype_reference_ignores_v1_reserved_flags() {
        let mut bytes = attribute::Attribute::new("a", &[], &[], &[]).build();
        bytes[1] = u8::MAX;
        assert_eq!(
            AttributeMessage::shared_datatype_reference(&bytes).unwrap(),
            None
        );
    }

    #[test]
    fn shared_datatype_reference_preserves_heap_reference_bytes() {
        let reference = shared_message::encode_sohm_ref(&[0xa5; shared_message::FHEAP_ID_LEN]);
        let bytes = attribute::Attribute::new("a", &reference, &[], &[])
            .flags(attribute::Flags::SHARED_DATATYPE)
            .build();
        assert_eq!(
            AttributeMessage::shared_datatype_reference(&bytes).unwrap(),
            Some(reference.as_slice())
        );
    }

    #[test]
    fn shared_datatype_reference_borrows_fields_with_absent_and_large_values() {
        let mut bytes = attr_with_committed_reference(FLAG_SHARED_DATATYPE);
        bytes.truncate(bytes.len() - 8);
        let reference = AttributeMessage::shared_datatype_reference(&bytes)
            .unwrap()
            .unwrap();
        assert_eq!(reference, datatype::committed_reference(0x320));
        let resolver = StubResolver(crate::datatype::serialize_datatype(&i32_datatype()).unwrap());
        assert_eq!(
            AttributeMessage::parse_resolving_at_with_length_width(
                &bytes,
                LengthWidth::Eight,
                &resolver
            )
            .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: bytes.len() + 4,
                available: bytes.len()
            }
        );
        let prefix = bytes.len();
        bytes.resize(prefix + 4_000_000, 0xa5);
        let reference = AttributeMessage::shared_datatype_reference(&bytes)
            .unwrap()
            .unwrap();
        assert_eq!(reference, datatype::committed_reference(0x320));
        assert_eq!(
            reference.as_ptr(),
            bytes[V1_V2_PREFIX_SIZE + "shared_attr".len() + 1..].as_ptr()
        );
    }

    #[rstest]
    #[case::ascii_v2("scale", false, VERSION_TWO, ASCII_NAME_CHARACTER_SET)]
    #[case::ascii_forced_v3("scale", true, VERSION_THREE, ASCII_NAME_CHARACTER_SET)]
    #[case::utf8_auto_v3("échelle", false, VERSION_THREE, UTF8_NAME_CHARACTER_SET)]
    #[case::utf8_forced_v3("échelle", true, VERSION_THREE, UTF8_NAME_CHARACTER_SET)]
    fn serialized_name_declares_its_character_set(
        #[case] name: &str,
        #[case] version_three: bool,
        #[case] version: u8,
        #[case] character_set: u8,
    ) {
        let message = AttributeMessage {
            name: name.into(),
            datatype: i32_datatype(),
            dataspace: Dataspace {
                space_type: DataspaceType::Scalar,
                rank: 0,
                dimensions: Vec::new(),
                max_dimensions: None,
            },
            raw_data: 7i32.to_le_bytes().to_vec(),
            datatype_location: DatatypeLocation::Inline,
        };
        let bytes = if version_three {
            message.serialize_v3(OffsetWidth::Eight, 8).unwrap()
        } else {
            message.serialize(OffsetWidth::Eight, 8).unwrap()
        };
        let datatype = crate::datatype::serialize_datatype(&i32_datatype()).unwrap();
        let dataspace = dataspace::scalar();
        let value = 7i32.to_le_bytes();
        let expected = attribute::Attribute::new(name, &datatype, &dataspace, &value);
        let expected = if version == VERSION_THREE {
            expected.character_set(character_set).build()
        } else {
            expected.flags(attribute::Flags::NONE).build()
        };
        assert_eq!(bytes, expected);
        assert_eq!(AttributeMessage::parse(&bytes, 8).unwrap(), message);
    }

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

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum AttributeVersion {
        One,
        Two,
        Three,
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
            Datatype::FloatingPoint(numeric) if numeric.size().get() == 8
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
            Datatype::FloatingPoint(numeric) if numeric.size().get() == 8
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

    #[rstest]
    #[case::inline(
        DatatypeLocation::Inline,
        crate::datatype::serialize_datatype(&i32_datatype()).unwrap(),
        attribute::Flags::NONE
    )]
    #[case::committed(
        DatatypeLocation::Committed(StoredAddress::new(800)),
        datatype::committed_reference(800),
        attribute::Flags::SHARED_DATATYPE
    )]
    fn a_version_3_message_round_trips_with_an_inline_or_committed_datatype(
        #[case] datatype_location: DatatypeLocation,
        #[case] datatype_field: Vec<u8>,
        #[case] flags: attribute::Flags,
    ) {
        let attribute = AttributeMessage {
            name: "scale".into(),
            datatype: i32_datatype(),
            dataspace: Dataspace {
                space_type: DataspaceType::Scalar,
                rank: 0,
                dimensions: Vec::new(),
                max_dimensions: None,
            },
            raw_data: 7i32.to_le_bytes().to_vec(),
            datatype_location,
        };
        let bytes = attribute.serialize_v3(OffsetWidth::Eight, 8).unwrap();

        assert_eq!(
            bytes,
            attribute::Attribute::new(
                "scale",
                &datatype_field,
                &dataspace::scalar(),
                &7i32.to_le_bytes(),
            )
            .flags(flags)
            .character_set(0)
            .build()
        );
        assert_eq!(AttributeMessage::parse_name(&bytes), Ok("scale".into()));
        assert_eq!(
            AttributeMessage::shares_a_field(&bytes),
            flags != attribute::Flags::NONE
        );
        assert_eq!(
            AttributeMessage::parse_resolving_at(&bytes, 8, &CommittedI32),
            Ok((attribute, bytes.len() - 4))
        );
    }

    /// A resolver that returns the `i32` datatype for every reference, as if it were committed at
    /// the address the reference holds.
    struct CommittedI32;

    impl SharedResolver for CommittedI32 {
        fn resolve(&self, _reference: &[u8], _target: MessageType) -> Result<Vec<u8>, FormatError> {
            crate::datatype::serialize_datatype(&i32_datatype())
        }

        fn committed_address(
            &self,
            reference: &[u8],
        ) -> Result<Option<StoredAddress>, FormatError> {
            shared_message::committed_address_in(reference, 8, 8)
        }
    }

    #[rstest]
    #[case::version_one(VERSION_ONE)]
    #[case::version_two(VERSION_TWO)]
    #[case::version_three(VERSION_THREE)]
    fn datatype_metadata_reads_types_from_messages_with_truncated_values(#[case] version: u8) {
        let datatype = datatype::fixed_string(4);
        let dataspace = dataspace::scalar();
        let message = attribute::Attribute::new("label", &datatype, &dataspace, &[]);
        let message = match version {
            VERSION_TWO => message.flags(attribute::Flags::NONE),
            VERSION_THREE => message.character_set(0),
            _ => message,
        };
        let bytes = message.build();
        assert_eq!(
            AttributeMessage::parse_datatype_resolving(&bytes, &Unresolvable).unwrap(),
            Datatype::String {
                size: 4,
                padding: StringPadding::NullPad,
                charset: CharacterSet::Ascii,
            }
        );
        assert_eq!(
            AttributeMessage::parse(&bytes, 8).unwrap_err(),
            FormatError::UnexpectedEof {
                expected: bytes.len() + 4,
                available: bytes.len(),
            }
        );
    }

    #[test]
    fn datatype_metadata_reads_inline_types_with_a_shared_dataspace() {
        let bytes = attribute::Attribute::new("label", &datatype::fixed_string(4), &[], &[])
            .flags(attribute::Flags(FLAG_SHARED_DATASPACE))
            .build();
        let datatype = AttributeMessage::parse_datatype_resolving(&bytes, &Unresolvable).unwrap();
        assert_eq!(
            datatype,
            Datatype::String {
                size: 4,
                padding: StringPadding::NullPad,
                charset: CharacterSet::Ascii,
            }
        );
        assert_eq!(
            AttributeMessage::parse(&bytes, 8).unwrap_err(),
            FormatError::UnresolvedSharedMessage(MessageType::DATASPACE.to_u16())
        );
    }

    #[rstest]
    #[case::accepted(false)]
    #[case::rejected(true)]
    fn datatype_inspection_resolves_shared_fields_once(#[case] reject: bool) {
        let bytes = attribute::Attribute::new(
            "label",
            &datatype::committed_reference(0x320),
            &datatype::committed_reference(0x640),
            b"abcd",
        )
        .flags(attribute::Flags(
            FLAG_SHARED_DATATYPE | FLAG_SHARED_DATASPACE,
        ))
        .build();
        let resolver = InspectionResolver {
            datatypes: Cell::new(0),
            dataspaces: Cell::new(0),
        };
        let inspected = Cell::new(0);
        let result = AttributeMessage::parse_inspecting_datatype(
            &bytes,
            FormatWidths::from_sizes(8, 8).unwrap(),
            &resolver,
            |datatype| {
                inspected.set(inspected.get() + 1);
                assert_eq!(datatype.type_size(), 4);
                if reject {
                    Err(FormatError::ResourceLimit {
                        limit: 0,
                        required: 1,
                    })
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(inspected.get(), 1);
        assert_eq!(resolver.datatypes.get(), 1);
        assert_eq!(resolver.dataspaces.get(), usize::from(!reject));
        if reject {
            assert_eq!(
                result.unwrap_err(),
                FormatError::ResourceLimit {
                    limit: 0,
                    required: 1
                }
            );
        } else {
            let parsed = result.unwrap();
            assert_eq!(parsed.name, "label");
            assert_eq!(parsed.raw_data, b"abcd");
            assert_eq!(
                parsed.datatype_location,
                DatatypeLocation::Committed(StoredAddress::new(0x320))
            );
            assert_eq!(
                parsed.dataspace,
                Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: Vec::new(),
                    max_dimensions: None,
                }
            );
        }
    }

    struct InspectionResolver {
        datatypes: Cell<usize>,
        dataspaces: Cell<usize>,
    }

    impl SharedResolver for InspectionResolver {
        fn resolve(&self, _reference: &[u8], target: MessageType) -> Result<Vec<u8>, FormatError> {
            match target {
                MessageType::DATATYPE => {
                    self.datatypes.set(self.datatypes.get() + 1);
                    Ok(datatype::fixed_string(4))
                }
                MessageType::DATASPACE => {
                    self.dataspaces.set(self.dataspaces.get() + 1);
                    Ok(dataspace::scalar())
                }
                _ => Err(FormatError::UnresolvedSharedMessage(target.to_u16())),
            }
        }

        fn committed_address(
            &self,
            reference: &[u8],
        ) -> Result<Option<StoredAddress>, FormatError> {
            shared_message::committed_address_in(reference, 8, 8)
        }
    }

    #[test]
    fn datatype_metadata_resolves_shared_types_from_prefix_fields() {
        let bytes =
            attribute::Attribute::new("label", &datatype::committed_reference(0x320), &[], &[])
                .flags(attribute::Flags(FLAG_SHARED_DATATYPE))
                .build();
        assert_eq!(
            AttributeMessage::parse_datatype_resolving(
                &bytes,
                &StubResolver(datatype::fixed_string(4))
            )
            .unwrap(),
            Datatype::String {
                size: 4,
                padding: StringPadding::NullPad,
                charset: CharacterSet::Ascii,
            }
        );
    }

    #[rstest]
    #[case::version(vec![5, 0], FormatError::InvalidAttributeVersion(5))]
    #[case::short(vec![VERSION_ONE], FormatError::UnexpectedEof { expected: 2, available: 1 })]
    #[case::prefix(vec![VERSION_THREE, 0], FormatError::UnexpectedEof { expected: V3_PREFIX_SIZE, available: 2 })]
    fn datatype_metadata_rejects_malformed_prefixes(
        #[case] bytes: Vec<u8>,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            AttributeMessage::parse_datatype_resolving(&bytes, &Unresolvable).unwrap_err(),
            expected
        );
    }

    fn i32_datatype() -> Datatype {
        Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )
    }

    // "The Attribute Message" in format specification, version 4.0, defines the
    // version 3 Name Character Set Encoding values.
    const ASCII_NAME_CHARACTER_SET: u8 = 0;
    const UTF8_NAME_CHARACTER_SET: u8 = 1;
}
