//! HDF5 Attribute message parsing (message type 0x000C).

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::attribute_info::AttributeInfoMessage;
use crate::btree_v2::{
    BTreeV2Header, collect_btree_v2_records, collect_btree_v2_records_from_source,
};
use crate::bytes::ensure_len;
use crate::convert::TryToUsize;
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::shared_message::{
    self, BufferedResolver, DatatypeLocation, SharedResolver, SourceResolver, Unresolvable,
};
use crate::sohm::SohmTable;
use crate::source::Source;

/// Bit 0 of an attribute message's flags byte: the datatype field holds a
/// reference to a committed (shared) datatype rather than the datatype itself
/// (`H5O_ATTR_FLAG_TYPE_SHARED`).
const FLAG_SHARED_DATATYPE: u8 = 0x01;

/// Bit 1: the same for the dataspace field (`H5O_ATTR_FLAG_SPACE_SHARED`).
const FLAG_SHARED_DATASPACE: u8 = 0x02;

/// Every flag bit the format defines (`H5O_ATTR_FLAG_ALL`).
const FLAG_ALL: u8 = FLAG_SHARED_DATATYPE | FLAG_SHARED_DATASPACE;

/// A parsed HDF5 attribute message.
///
/// `PartialEq` compares every field, which is what makes "this attribute crossed
/// a rewrite unchanged" a single assertion — the shape repack's fidelity tests
/// take.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeMessage {
    /// Attribute name.
    pub name: String,
    /// Attribute datatype.
    pub datatype: Datatype,
    /// Attribute dataspace.
    pub dataspace: Dataspace,
    /// Raw attribute value data.
    pub raw_data: Vec<u8>,
    /// Whether [`Self::datatype`] is encoded in this message or named through a
    /// committed datatype object, which is a difference the field itself cannot
    /// show: both forms decode to the same type.
    pub datatype_location: DatatypeLocation,
}

/// Round up to the next multiple of 8.
fn pad8(x: usize) -> usize {
    (x + 7) & !7
}

impl AttributeMessage {
    /// Parse an attribute message from raw message bytes, without the file the
    /// message came from.
    ///
    /// An attribute whose datatype or dataspace field is a *reference* to a
    /// committed (shared) message cannot be decoded this way and is refused with
    /// [`FormatError::UnresolvedSharedMessage`]; use
    /// [`parse_resolving`](Self::parse_resolving) where the file is reachable.
    ///
    /// `length_size` is needed for dataspace dimension parsing.
    pub fn parse(data: &[u8], length_size: u8) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving(data, length_size, &Unresolvable)
    }

    /// Parse an attribute message, following a reference to a committed (shared)
    /// datatype or dataspace through `resolver` where the flags byte says the
    /// field holds one.
    pub fn parse_resolving(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<AttributeMessage, FormatError> {
        Self::parse_resolving_at(data, length_size, resolver).map(|(attr, _)| attr)
    }

    /// [`parse_resolving`](Self::parse_resolving), also reporting where the
    /// value bytes start within `data`.
    ///
    /// The offset is what lets a caller address an attribute's elements *in the
    /// file* rather than only in the copy `raw_data` holds — needed to repoint a
    /// stored object reference in place (issue #324). It is returned from the
    /// same field walk that produces `raw_data` rather than recomputed by a
    /// second one: the three versions pad their name, datatype and dataspace
    /// fields differently, and a separate derivation of the same offset would be
    /// free to drift from this one.
    pub(crate) fn parse_resolving_at(
        data: &[u8],
        length_size: u8,
        resolver: &dyn SharedResolver,
    ) -> Result<(AttributeMessage, usize), FormatError> {
        ensure_len(data, 0, 2)?;
        let version = data[0];

        match version {
            1 => Self::parse_v1(data, length_size),
            2 => Self::parse_v2(data, length_size, resolver),
            3 => Self::parse_v3(data, length_size, resolver),
            _ => Err(FormatError::InvalidAttributeVersion(version)),
        }
    }

    fn parse_v1(data: &[u8], length_size: u8) -> Result<(AttributeMessage, usize), FormatError> {
        // version(1) + reserved(1) + name_size(2) + datatype_size(2) + dataspace_size(2) = 8
        ensure_len(data, 0, 8)?;
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;

        let mut pos = 8;

        // Name (padded to 8-byte boundary)
        ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += pad8(name_size);

        // Datatype (padded to 8-byte boundary). Version 1 has no flags byte, so
        // neither field can be a reference.
        ensure_len(data, pos, datatype_size)?;
        let (datatype, _) = Datatype::parse(&data[pos..pos + datatype_size])?;
        pos += pad8(datatype_size);

        // Dataspace (padded to 8-byte boundary)
        ensure_len(data, pos, dataspace_size)?;
        let dataspace = Dataspace::parse(&data[pos..pos + dataspace_size], length_size)?;
        pos += pad8(dataspace_size);

        // Raw data: num_elements × type_size bytes
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
        ensure_len(data, 0, 8)?;
        let flags = data[1];
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;

        let mut pos = 8;

        // Name (NO padding)
        ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += name_size;

        // Datatype (NO padding)
        ensure_len(data, pos, datatype_size)?;
        let dt_field = &data[pos..pos + datatype_size];
        pos += datatype_size;

        // Dataspace (NO padding)
        ensure_len(data, pos, dataspace_size)?;
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
        ensure_len(data, 0, 9)?;
        let flags = data[1];
        let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
        let datatype_size = u16::from_le_bytes([data[4], data[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([data[6], data[7]]) as usize;
        let _encoding = data[8]; // 0=ASCII, 1=UTF-8

        let mut pos = 9;

        // Name (NO padding)
        ensure_len(data, pos, name_size)?;
        let name = extract_name(&data[pos..pos + name_size]);
        pos += name_size;

        // Datatype (NO padding)
        ensure_len(data, pos, datatype_size)?;
        let dt_field = &data[pos..pos + datatype_size];
        pos += datatype_size;

        // Dataspace (NO padding)
        ensure_len(data, pos, dataspace_size)?;
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

    /// Serialize attribute message (v2 format, no padding).
    pub fn serialize(&self, length_size: u8) -> Vec<u8> {
        self.serialize_version(2, length_size)
    }

    /// Serialize attribute message as v3 (adds character set encoding byte).
    pub fn serialize_v3(&self, length_size: u8) -> Vec<u8> {
        self.serialize_version(3, length_size)
    }

    /// The first of the message's three 2-byte header fields that this attribute
    /// would overflow, as `(field name, encoded length)`.
    ///
    /// [`Self::serialize_version`] writes the name, datatype and dataspace
    /// lengths into `u16` fields, so a value past 65,535 truncates and produces a
    /// message that decodes as something else entirely. Callers that can refuse
    /// must check this first; the attribute's *data* is not length-prefixed and
    /// so is not bounded here.
    pub(crate) fn v3_header_field_overflow(
        &self,
        length_size: u8,
    ) -> Option<(&'static str, usize)> {
        let limit = u16::MAX as usize;
        let fields = [
            // The null terminator the message carries counts toward the field.
            ("name", self.name.len() + 1),
            ("datatype", self.datatype_field().len()),
            ("dataspace", self.dataspace.serialize(length_size).len()),
        ];
        fields.into_iter().find(|&(_, len)| len > limit)
    }

    /// The bytes of the message's datatype field: the encoding itself, or the
    /// reference standing in for it when the type is committed.
    ///
    /// A reference is written in the *writer's* offset width rather than in the
    /// width of whatever file the message was read from, because that is the file
    /// the bytes are going into. Re-serializing a message parsed from a file with
    /// a different width is therefore a re-encoding, not a copy — which is what
    /// it already is for every other field.
    fn datatype_field(&self) -> Vec<u8> {
        match self
            .datatype_location
            .reference_bytes(crate::file_writer::OFFSET_SIZE)
        {
            Some(reference) => reference,
            None => self.datatype.serialize(),
        }
    }

    fn serialize_version(&self, version: u8, length_size: u8) -> Vec<u8> {
        let name_bytes = {
            let mut n = self.name.as_bytes().to_vec();
            n.push(0); // null terminator
            n
        };
        let dt_bytes = self.datatype_field();
        let ds_bytes = self.dataspace.serialize(length_size);

        let mut buf = Vec::new();
        buf.push(version);
        // Version 1 has no flags byte, but nothing serializes one: both callers
        // ask for version 2 or 3, whose second byte says which fields are
        // references. Only the datatype is ever one here — this crate does not
        // write a shared dataspace.
        buf.push(if self.datatype_location.is_committed() {
            FLAG_SHARED_DATATYPE
        } else {
            0
        });
        #[expect(
            clippy::cast_possible_truncation,
            reason = "attribute name length is written into the 2-byte name-size field of the attribute message"
        )]
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "serialized datatype length is written into the 2-byte datatype-size field of the attribute message"
        )]
        buf.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "serialized dataspace length is written into the 2-byte dataspace-size field of the attribute message"
        )]
        buf.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        if version >= 3 {
            buf.push(0x00); // character set encoding: ASCII
        }
        buf.extend_from_slice(&name_bytes);
        buf.extend_from_slice(&dt_bytes);
        buf.extend_from_slice(&ds_bytes);
        buf.extend_from_slice(&self.raw_data);
        buf
    }

    /// Read attribute value as f64 values.
    pub fn read_as_f64(&self) -> Result<Vec<f64>, FormatError> {
        data_read::read_as_f64(&self.raw_data, &self.datatype)
    }

    /// Read attribute value as i64 values.
    pub fn read_as_i64(&self) -> Result<Vec<i64>, FormatError> {
        data_read::read_as_i64(&self.raw_data, &self.datatype)
    }

    /// Read attribute value as u64 values.
    pub fn read_as_u64(&self) -> Result<Vec<u64>, FormatError> {
        data_read::read_as_u64(&self.raw_data, &self.datatype)
    }

    /// Read attribute value as a single string (first element).
    ///
    /// Only used by tests; gated so it is not shipped as dead code.
    #[cfg(test)]
    pub fn read_as_string(&self) -> Result<String, FormatError> {
        let strings = data_read::read_as_strings(&self.raw_data, &self.datatype)?;
        Ok(strings.into_iter().next().unwrap_or_default())
    }

    /// Read attribute value as a vector of fixed-length strings.
    pub fn read_as_strings(&self) -> Result<Vec<String>, FormatError> {
        data_read::read_as_strings(&self.raw_data, &self.datatype)
    }
}

/// Decode an attribute's datatype and dataspace fields.
///
/// A message's flags byte says, per field, whether the bytes are the encoding or
/// a *reference* to a committed (shared) message holding it. The two are not
/// distinguishable by inspection — a version 2 reference to address `0x320` reads
/// as a valid time datatype of size zero — so the flag is the only thing that
/// tells them apart, and reading the field without it is how a committed datatype
/// silently becomes the wrong type.
///
/// Only the datatype's origin is reported back. A shared *dataspace* is resolved
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
    // The C library refuses a flags byte with any other bit set, so a message
    // carrying one is not an attribute message this or any reader can trust.
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
        let body = resolver.resolve(dt_field, MessageType::Datatype)?;
        (
            Datatype::parse(&body)?.0,
            match address {
                Some(address) => DatatypeLocation::Committed(address),
                None => DatatypeLocation::Inline,
            },
        )
    } else {
        (Datatype::parse(dt_field)?.0, DatatypeLocation::Inline)
    };

    let dataspace = if flags & FLAG_SHARED_DATASPACE != 0 {
        let body = resolver.resolve(ds_field, MessageType::Dataspace)?;
        Dataspace::parse(&body, length_size)?
    } else {
        Dataspace::parse(ds_field, length_size)?
    };

    Ok((datatype, dataspace, location))
}

/// The name of an attribute message, without decoding its datatype or dataspace.
///
/// The in-place editor identifies attributes by name while walking an object
/// header region it has no file context for, and a committed datatype is exactly
/// what it cannot decode there. The name never depends on either field, so
/// reading it alone lets an edit pass over such an attribute rather than refuse
/// the whole object.
pub fn message_name(data: &[u8]) -> Result<String, FormatError> {
    ensure_len(data, 0, 8)?;
    let version = data[0];
    let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
    let name_start = match version {
        1 | 2 => 8,
        3 => 9,
        _ => return Err(FormatError::InvalidAttributeVersion(version)),
    };
    ensure_len(data, name_start, name_size)?;
    Ok(extract_name(&data[name_start..name_start + name_size]))
}

/// Whether an attribute message stores its datatype or dataspace as a reference
/// to a committed (shared) message, read from the flags byte alone.
///
/// A byte-level screen for callers that hold a message body and must decide
/// whether it may be copied, without decoding it. A malformed or truncated
/// message reports `true`, so a header this cannot read is refused rather than
/// waved through.
pub fn message_shares_a_field(data: &[u8]) -> bool {
    match data.first().copied() {
        // Version 1 has no flags byte; the field after the version is unused.
        Some(1) => false,
        Some(2 | 3) => data.get(1).is_none_or(|flags| flags & FLAG_ALL != 0),
        _ => true,
    }
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
    ensure_len(data, pos, expected_size)?;
    Ok(if expected_size > 0 {
        data.get(pos..pos + expected_size)
            .map(<[u8]>::to_vec)
            .unwrap_or_default()
    } else {
        Vec::new()
    })
}

/// Extract a name from raw bytes, stripping null terminator.
fn extract_name(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Extract all (compact) attribute messages from an object header.
///
/// Only used by tests; the reader uses [`extract_attributes_full`] (which also
/// handles dense storage). Gated so it is not shipped as dead code.
#[cfg(test)]
pub fn extract_attributes(
    header: &ObjectHeader,
    length_size: u8,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let mut attrs = Vec::new();
    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            let attr = AttributeMessage::parse(&msg.data, length_size)?;
            attrs.push(attr);
        }
    }
    Ok(attrs)
}

/// Extract all attributes from an object header, supporting both compact and dense storage.
///
/// This function handles:
/// - Compact attributes: inline Attribute messages (0x000C) in the object header
/// - Dense attributes: AttributeInfo message (0x0015) pointing to fractal heap + B-tree v2
/// - Shared messages: resolves shared datatype references for attribute messages
///
/// Use this instead of `extract_attributes` when reading files that may use dense storage
/// (e.g., objects with many attributes, typically >8).
pub fn extract_attributes_full(
    file_data: &[u8],
    header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    let resolver = BufferedResolver::new(file_data, offset_size, length_size, sohm);
    let mut attrs = Vec::new();

    // Collect compact attributes (inline in OH)
    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            let attr = if shared_message::is_shared(msg.flags) {
                // The whole attribute message is shared: resolve the reference to
                // get the message, which may itself name a committed datatype.
                let resolved = resolver.resolve(&msg.data, MessageType::Attribute)?;
                AttributeMessage::parse_resolving(&resolved, length_size, &resolver)?
            } else {
                AttributeMessage::parse_resolving(&msg.data, length_size, &resolver)?
            };
            attrs.push(attr);
        }
    }

    // Check for dense attributes via AttributeInfo message
    let attr_info = find_attribute_info(header, offset_size)?;
    if let Some(info) = attr_info
        && let Some(fh_addr) = info.fractal_heap_address
    {
        let dense_attrs =
            extract_dense_attributes(file_data, &info, fh_addr, offset_size, length_size, sohm)?;
        attrs.extend(dense_attrs);
    }

    Ok(attrs)
}

/// Streaming counterpart of [`extract_attributes_full`].
///
/// Reads compact attribute messages from the (already-parsed) object header,
/// resolves shared attribute references, and walks dense storage (fractal heap +
/// B-tree v2) through a [`Source`] on demand instead of indexing a whole-file
/// slice. Used by the streaming reader backend.
pub fn extract_attributes_full_from_source<S: Source + ?Sized>(
    source: &S,
    header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    Ok(
        extract_stored_attributes_from_source(source, header, offset_size, length_size, sohm)?
            .into_iter()
            .map(|a| a.message)
            .collect(),
    )
}

/// An attribute as its object stores it, with the creation index the storage
/// records for it.
#[derive(Debug, Clone)]
pub struct StoredAttribute {
    /// The attribute itself.
    pub message: AttributeMessage,
    /// Its creation index: the object-header message record's field for a
    /// compact attribute, the name index record's for a dense one. `None`
    /// where the object does not track attribute creation order — which is
    /// every object this crate's own whole-file writer produces.
    pub creation_index: Option<u16>,
}

/// [`extract_attributes_full_from_source`], keeping each attribute's stored
/// creation index.
///
/// The in-place editor needs it: an object that tracks attribute creation order
/// has to keep every attribute's index across an edit that rebuilds its storage,
/// and the index is the one thing an attribute message itself does not carry.
pub fn extract_stored_attributes_from_source<S: Source + ?Sized>(
    source: &S,
    header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&SohmTable>,
) -> Result<Vec<StoredAttribute>, FormatError> {
    let resolver = SourceResolver::new(source, offset_size, length_size, sohm);
    let mut attrs = Vec::new();

    // Collect compact attributes (inline in OH)
    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            let attr = if shared_message::is_shared(msg.flags) {
                let resolved = resolver.resolve(&msg.data, MessageType::Attribute)?;
                AttributeMessage::parse_resolving(&resolved, length_size, &resolver)?
            } else {
                AttributeMessage::parse_resolving(&msg.data, length_size, &resolver)?
            };
            attrs.push(StoredAttribute {
                message: attr,
                creation_index: msg.creation_order,
            });
        }
    }

    // Check for dense attributes via AttributeInfo message
    let attr_info = find_attribute_info(header, offset_size)?;
    if let Some(info) = attr_info
        && let Some(fh_addr) = info.fractal_heap_address
    {
        let dense_attrs = extract_dense_attributes_from_source(
            source,
            &info,
            fh_addr,
            offset_size,
            length_size,
            sohm,
        )?;
        attrs.extend(dense_attrs);
    }

    Ok(attrs)
}

/// Find and parse the Attribute Info message from an object header.
fn find_attribute_info(
    header: &ObjectHeader,
    offset_size: u8,
) -> Result<Option<AttributeInfoMessage>, FormatError> {
    for msg in &header.messages {
        if msg.msg_type == MessageType::AttributeInfo {
            let info = AttributeInfoMessage::parse(&msg.data, offset_size)?;
            return Ok(Some(info));
        }
    }
    Ok(None)
}

/// Extract attributes from dense storage (fractal heap + B-tree v2).
fn extract_dense_attributes(
    file_data: &[u8],
    attr_info: &AttributeInfoMessage,
    fh_addr: u64,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&SohmTable>,
) -> Result<Vec<AttributeMessage>, FormatError> {
    // Parse fractal heap
    let fh = FractalHeapHeader::parse(file_data, fh_addr.to_usize()?, offset_size, length_size)?;

    // Parse B-tree v2 for name index (type 8)
    let btree_addr = attr_info
        .btree_name_index_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
    let btree_hdr =
        BTreeV2Header::parse(file_data, btree_addr.to_usize()?, offset_size, length_size)?;
    let records = collect_btree_v2_records(file_data, &btree_hdr, offset_size, length_size)?;

    let resolver = BufferedResolver::new(file_data, offset_size, length_size, sohm);
    let mut heap = fh.object_reader(offset_size, length_size);
    let mut attrs = Vec::new();
    for record in &records {
        // Per HDF5 spec, both type 8 and type 9 records start with heap_id:
        //   Type 8: heap_id(8) + msg_flags(1) + creation_order(4) + hash(4)
        //   Type 9: heap_id(8) + msg_flags(1) + creation_order(4)
        let id_offset = 0;

        if record.data.len() < id_offset + fh.heap_id_length as usize {
            continue;
        }
        let id_bytes = &record.data[id_offset..id_offset + fh.heap_id_length as usize];

        // Read the attribute message from the fractal heap (managed or huge object).
        let attr_data = heap.read(file_data, id_bytes)?;

        // The data in the heap is a complete attribute message, and it names a
        // committed datatype the same way a compact one does.
        let attr = AttributeMessage::parse_resolving(&attr_data, length_size, &resolver)?;
        attrs.push(attr);
    }

    Ok(attrs)
}

/// Streaming counterpart of [`extract_dense_attributes`]: walks the fractal heap
/// and B-tree v2 through a [`Source`] on demand.
fn extract_dense_attributes_from_source<S: Source + ?Sized>(
    source: &S,
    attr_info: &AttributeInfoMessage,
    fh_addr: u64,
    offset_size: u8,
    length_size: u8,
    sohm: Option<&SohmTable>,
) -> Result<Vec<StoredAttribute>, FormatError> {
    let fh = FractalHeapHeader::parse_from_source(source, fh_addr, offset_size, length_size)?;

    let btree_addr = attr_info
        .btree_name_index_address
        .ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
    let btree_hdr = BTreeV2Header::parse_from_source(source, btree_addr, offset_size, length_size)?;
    let records =
        collect_btree_v2_records_from_source(source, &btree_hdr, offset_size, length_size)?;

    let resolver = SourceResolver::new(source, offset_size, length_size, sohm);
    let mut heap = fh.object_reader(offset_size, length_size);
    let mut attrs = Vec::new();
    for record in &records {
        // Both type 8 and type 9 records begin with the heap_id.
        let id_offset = 0;
        if record.data.len() < id_offset + fh.heap_id_length as usize {
            continue;
        }
        let id_bytes = &record.data[id_offset..id_offset + fh.heap_id_length as usize];
        let attr_data = heap.read_from_source(source, id_bytes)?;
        attrs.push(StoredAttribute {
            message: AttributeMessage::parse_resolving(&attr_data, length_size, &resolver)?,
            creation_index: record_creation_index(record, &fh, attr_info),
        });
    }

    Ok(attrs)
}

/// The creation index a name-index (type 8) record carries, for an object that
/// tracks attribute creation order.
///
/// The field sits right after the heap ID and the message flags byte, and is 4
/// bytes wide where the Attribute Info message's maximum is 2 — the reference C
/// library writes the same value into both, so anything past `u16` is a record
/// this crate did not write and cannot reproduce. An object that does not track
/// the order stores something else there (this crate's whole-file writer stores
/// the attribute's position), so the tracked flag gates the read.
fn record_creation_index(
    record: &crate::btree_v2::BTreeV2Record,
    fh: &FractalHeapHeader,
    attr_info: &AttributeInfoMessage,
) -> Option<u16> {
    attr_info.max_creation_index?;
    let at = fh.heap_id_length as usize + 1;
    let bytes = record.data.get(at..at + 4)?;
    u16::try_from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::BytesSource;
    use core::cell::RefCell;

    /// A [`Source`] that records where each read started, so a walk can be asked
    /// how often it went back to a particular structure.
    struct CountingSource {
        inner: BytesSource<Vec<u8>>,
        reads: RefCell<Vec<u64>>,
    }

    impl CountingSource {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                inner: BytesSource::new(bytes),
                reads: RefCell::new(Vec::new()),
            }
        }

        fn reads_at(&self, offset: u64) -> usize {
            self.reads.borrow().iter().filter(|&&o| o == offset).count()
        }
    }

    impl Source for CountingSource {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push(offset);
            self.inner.read_at(offset, buf)
        }
    }

    /// A file whose root carries `count` attributes, each too large for a
    /// managed heap object, so every one of them resolves through the heap's
    /// huge-object B-tree.
    fn file_with_huge_attributes(count: usize) -> Vec<u8> {
        let mut builder = crate::FileBuilder::new();
        for i in 0..count {
            builder.set_attr(
                &format!("a{i}"),
                crate::AttrValue::StringArray(vec![format!("{i:0700}"); 100]),
            );
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder.finish().unwrap()
    }

    /// A file whose root carries `count` attributes small enough to be managed
    /// heap objects, so the heap holds no huge object at all.
    fn file_with_managed_attributes(count: usize) -> Vec<u8> {
        let mut builder = crate::FileBuilder::new();
        for i in 0..count {
            builder.set_attr(&format!("a{i}"), crate::AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder.finish().unwrap()
    }

    /// The root group's dense-attribute storage: its info message, its heap
    /// address, and the file's offset and length sizes.
    fn dense_attribute_info(bytes: &[u8]) -> (AttributeInfoMessage, u64, u8, u8) {
        let sig = crate::signature::find_signature(bytes).unwrap();
        let superblock = crate::superblock::Superblock::parse(bytes, sig).unwrap();
        let (offset_size, length_size) = (superblock.offset_size, superblock.length_size);
        let root = ObjectHeader::parse(
            bytes,
            superblock.root_group_address.to_usize().unwrap(),
            offset_size,
            length_size,
        )
        .unwrap();
        let info = find_attribute_info(&root, offset_size)
            .unwrap()
            .expect("this many attributes are stored densely");
        let fh_addr = info
            .fractal_heap_address
            .expect("dense storage names its heap");
        (info, fh_addr, offset_size, length_size)
    }

    /// The streaming dense walk resolves every huge object against one parse of
    /// the heap's huge-object index, not one parse per object.
    ///
    /// Costs, not answers, are what regress here: reading the index per object
    /// returns exactly the same attributes while making the walk quadratic in
    /// their number, so the count of reads is the only thing that catches it.
    /// This one counts them end to end, at the B-tree's own address, rather than
    /// on the reader.
    #[test]
    fn a_dense_walk_parses_its_huge_object_index_once() {
        const COUNT: usize = 6;
        let bytes = file_with_huge_attributes(COUNT);
        let (info, fh_addr, offset_size, length_size) = dense_attribute_info(&bytes);
        let heap = FractalHeapHeader::parse(
            &bytes,
            fh_addr.to_usize().unwrap(),
            offset_size,
            length_size,
        )
        .unwrap();
        let btree_addr = heap.btree_huge_objects_address;

        let source = CountingSource::new(bytes);
        let attrs = extract_dense_attributes_from_source(
            &source,
            &info,
            fh_addr,
            offset_size,
            length_size,
            None,
        )
        .unwrap();

        assert_eq!(
            attrs.len(),
            COUNT,
            "the walk must still read every attribute"
        );
        assert_eq!(
            source.reads_at(btree_addr),
            1,
            "the huge-object B-tree header was re-read per object"
        );
    }

    /// The buffered dense walk holds to the same invariant, on the path
    /// `File::open` takes.
    ///
    /// The reader caching the index is only half of it; the other half is each
    /// walk building one reader for the whole heap rather than one per object,
    /// and that half is per call site.
    #[test]
    fn a_buffered_dense_walk_parses_its_huge_object_index_once() {
        const COUNT: usize = 6;
        let bytes = file_with_huge_attributes(COUNT);
        let (info, fh_addr, offset_size, length_size) = dense_attribute_info(&bytes);

        crate::fractal_heap::reset_huge_index_decodes();
        let attrs =
            extract_dense_attributes(&bytes, &info, fh_addr, offset_size, length_size, None)
                .unwrap();

        assert_eq!(
            attrs.len(),
            COUNT,
            "the walk must still read every attribute"
        );
        assert_eq!(
            crate::fractal_heap::huge_index_decodes(),
            1,
            "the huge-object index was parsed per object rather than per walk"
        );
    }

    /// A heap holding no huge object never parses a huge-object index, on either
    /// backend. The index is parsed on demand, and every dense walk that reads
    /// only managed objects is a walk that must not pay for one.
    #[test]
    fn a_managed_dense_walk_never_parses_a_huge_object_index() {
        // Enough attributes to force dense storage, none of them large enough to
        // exceed the heap's managed-object limit.
        const COUNT: usize = 40;
        let bytes = file_with_managed_attributes(COUNT);
        let (info, fh_addr, offset_size, length_size) = dense_attribute_info(&bytes);

        crate::fractal_heap::reset_huge_index_decodes();
        let buffered =
            extract_dense_attributes(&bytes, &info, fh_addr, offset_size, length_size, None)
                .unwrap();
        let source = BytesSource::new(bytes);
        let streamed = extract_dense_attributes_from_source(
            &source,
            &info,
            fh_addr,
            offset_size,
            length_size,
            None,
        )
        .unwrap();

        assert_eq!(buffered.len(), COUNT, "the walk must read every attribute");
        assert_eq!(streamed.len(), COUNT);
        assert_eq!(
            crate::fractal_heap::huge_index_decodes(),
            0,
            "a heap with no huge object parsed an index it has no use for"
        );
    }

    /// Build a datatype header for testing (8 bytes).
    fn build_dt_header(class: u8, version: u8, bf: [u8; 3], size: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        buf[0] = (class & 0x0F) | ((version & 0x0F) << 4);
        buf[1] = bf[0];
        buf[2] = bf[1];
        buf[3] = bf[2];
        buf[4..8].copy_from_slice(&size.to_le_bytes());
        buf
    }

    /// Build an f64 LE datatype message.
    fn build_f64_dt() -> Vec<u8> {
        let mut buf = build_dt_header(1, 1, [0x00, 0x00, 0x02], 8);
        let mut props = [0u8; 12];
        props[2..4].copy_from_slice(&64u16.to_le_bytes()); // bit_precision
        props[4] = 52; // exp_location
        props[5] = 11; // exp_size
        props[6] = 0; // mant_location
        props[7] = 52; // mant_size
        props[8..12].copy_from_slice(&1023u32.to_le_bytes()); // exp_bias
        buf.extend_from_slice(&props);
        buf
    }

    /// Build a scalar dataspace (v2).
    fn build_scalar_ds() -> Vec<u8> {
        vec![2, 0, 0, 0] // version=2, rank=0, flags=0, type=0(scalar)
    }

    /// Build a simple 1D dataspace (v1).
    fn build_simple_ds_v1(dim: u64) -> Vec<u8> {
        let mut buf = vec![1u8, 1, 0, 0, 0, 0, 0, 0]; // version=1, rank=1, flags=0, reserved(5)
        buf.extend_from_slice(&dim.to_le_bytes());
        buf
    }

    /// Build a fixed-length string datatype.
    fn build_string_dt(size: u32) -> Vec<u8> {
        // class=3, version=1, padding=NullPad(1), charset=ASCII(0) → bf0=0x01
        build_dt_header(3, 1, [0x01, 0, 0], size)
    }

    #[test]
    fn parse_v1_attribute_f64_scalar() {
        let name = b"temp\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_scalar_ds();

        let name_size = name.len();
        let dt_size = dt_bytes.len();
        let ds_size = ds_bytes.len();

        let mut data = Vec::new();
        data.push(1); // version
        data.push(0); // reserved
        data.extend_from_slice(&(name_size as u16).to_le_bytes());
        data.extend_from_slice(&(dt_size as u16).to_le_bytes());
        data.extend_from_slice(&(ds_size as u16).to_le_bytes());

        // Name padded to 8 bytes
        data.extend_from_slice(name);
        if data.len() % 8 != 0 || data.len() == 8 {
            // Pad name to 8-byte boundary from start of name
            let name_start = 8;
            let name_padded = pad8(name_size);
            while data.len() < name_start + name_padded {
                data.push(0);
            }
        }

        // Datatype padded to 8 bytes
        let dt_start = data.len();
        data.extend_from_slice(&dt_bytes);
        let dt_padded = pad8(dt_size);
        while data.len() < dt_start + dt_padded {
            data.push(0);
        }

        // Dataspace padded to 8 bytes
        let ds_start = data.len();
        data.extend_from_slice(&ds_bytes);
        let ds_padded = pad8(ds_size);
        while data.len() < ds_start + ds_padded {
            data.push(0);
        }

        // Raw data: f64 value 98.6
        data.extend_from_slice(&98.6f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "temp");
        assert_eq!(attr.dataspace.num_elements(), 1);
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 98.6).abs() < 1e-10);
    }

    #[test]
    fn parse_v2_attribute_fixed_string() {
        let name = b"label\0";
        let dt_bytes = build_string_dt(5);
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(2); // version
        data.push(0); // flags
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());

        // No padding in v2
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);

        // Raw data: "hello"
        data.extend_from_slice(b"hello");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "label");
        let s = attr.read_as_string().unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn parse_v3_attribute_utf8() {
        let name = b"note\0";
        let dt_bytes = build_string_dt(3);
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(3); // version
        data.push(0); // flags
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.push(1); // encoding = UTF-8

        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"abc");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "note");
        let s = attr.read_as_string().unwrap();
        assert_eq!(s, "abc");
    }

    #[test]
    fn parse_v2_attribute_1d_array() {
        let name = b"vals\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_simple_ds_v1(3);

        let mut data = Vec::new();
        data.push(2); // version
        data.push(0); // flags
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());

        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);

        // 3 f64 values
        data.extend_from_slice(&1.0f64.to_le_bytes());
        data.extend_from_slice(&2.0f64.to_le_bytes());
        data.extend_from_slice(&3.0f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "vals");
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn parse_v1_padding_alignment() {
        // Verify v1 pads name, dt, ds each to 8 bytes
        let name = b"x\0"; // 2 bytes → pad to 8
        let dt_bytes = build_f64_dt(); // 20 bytes → pad to 24
        let ds_bytes = build_scalar_ds(); // 4 bytes → pad to 8

        let mut data = Vec::new();
        data.push(1); // version
        data.push(0); // reserved
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());

        // Name padded to 8
        data.extend_from_slice(name);
        data.resize(8 + pad8(name.len()), 0);

        // DT padded to 8
        let dt_start = data.len();
        data.extend_from_slice(&dt_bytes);
        data.resize(dt_start + pad8(dt_bytes.len()), 0);

        // DS padded to 8
        let ds_start = data.len();
        data.extend_from_slice(&ds_bytes);
        data.resize(ds_start + pad8(ds_bytes.len()), 0);

        // raw data
        data.extend_from_slice(&42.0f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "x");
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals, vec![42.0]);
    }

    #[test]
    fn parse_v2_no_padding() {
        // Same as parse_v2_attribute_fixed_string but verifying no padding
        let name = b"ab\0"; // 3 bytes, no padding
        let dt_bytes = build_string_dt(2); // 8 bytes, no padding
        let ds_bytes = build_scalar_ds(); // 4 bytes, no padding

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"hi");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.name, "ab");
        assert_eq!(attr.read_as_string().unwrap(), "hi");
    }

    #[test]
    fn truncated_attribute_error() {
        let data = [1u8]; // too short
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_version_error() {
        let data = [5u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidAttributeVersion(5));
    }

    #[test]
    fn extract_attributes_from_header() {
        // Build a fake ObjectHeader with 3 attribute messages
        let mut msgs = Vec::new();
        for i in 0..3 {
            let name = format!("attr{}\0", i);
            let dt_bytes = build_f64_dt();
            let ds_bytes = build_scalar_ds();

            let mut attr_data = Vec::new();
            attr_data.push(2); // version
            attr_data.push(0);
            attr_data.extend_from_slice(&(name.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
            attr_data.extend_from_slice(name.as_bytes());
            attr_data.extend_from_slice(&dt_bytes);
            attr_data.extend_from_slice(&ds_bytes);
            attr_data.extend_from_slice(&((i as f64) * 1.0).to_le_bytes());

            msgs.push(crate::object_header::HeaderMessage {
                msg_type: MessageType::Attribute,
                size: attr_data.len(),
                flags: 0,
                creation_order: None,
                data: attr_data,
            });
        }

        let header = ObjectHeader {
            version: 2,
            messages: msgs,
            reference_count: None,
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        };

        let attrs = extract_attributes(&header, 8).unwrap();
        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs[0].name, "attr0");
        assert_eq!(attrs[1].name, "attr1");
        assert_eq!(attrs[2].name, "attr2");
    }

    /// A version 2 attribute message whose datatype field is a reference to a
    /// committed type, laid out exactly as libhdf5 1.14.6 wrote one: a 10-byte
    /// shared reference standing where an encoding usually is.
    fn attr_with_shared_datatype(flags: u8) -> Vec<u8> {
        let name = b"shared_attr\0";
        // version 2 shared reference: version, type = committed, address.
        let mut dt_field = vec![2u8, 2];
        dt_field.extend_from_slice(&0x320u64.to_le_bytes());
        let ds_bytes = build_simple_ds_v1(1);

        let mut data = vec![2u8, flags];
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_field.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_field);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(&7.0f64.to_le_bytes());
        data
    }

    /// A resolver that answers with one fixed message body, standing in for the
    /// object header a committed datatype lives in.
    struct StubResolver(Vec<u8>);

    impl SharedResolver for StubResolver {
        fn resolve(&self, _reference: &[u8], _target: MessageType) -> Result<Vec<u8>, FormatError> {
            Ok(self.0.clone())
        }

        fn committed_address(&self, reference: &[u8]) -> Result<Option<u64>, FormatError> {
            shared_message::committed_address_in(reference, 8, 8)
        }
    }

    /// The flags byte decides how the datatype field is read. Given a resolver,
    /// the attribute reports the referenced type — and says so.
    #[test]
    fn a_shared_datatype_field_is_resolved_not_decoded() {
        let data = attr_with_shared_datatype(FLAG_SHARED_DATATYPE);
        let attr =
            AttributeMessage::parse_resolving(&data, 8, &StubResolver(build_f64_dt())).unwrap();

        assert_eq!(attr.name, "shared_attr");
        assert!(matches!(
            attr.datatype,
            Datatype::FloatingPoint { size: 8, .. }
        ));
        assert_eq!(
            attr.datatype_location,
            DatatypeLocation::Committed(0x320),
            "the attribute must record which committed object it named, not just that it named one"
        );
    }

    /// With the flag clear the same bytes are decoded inline, which is how the
    /// reference used to be read: they form a syntactically well-formed time type
    /// of zero width. Nothing occupies zero bytes per element, so that decode is
    /// refused rather than returned, and the flag is left as the only thing that
    /// makes these bytes name a type at all.
    #[test]
    fn the_same_bytes_without_the_flag_are_refused_as_a_zero_width_type() {
        let data = attr_with_shared_datatype(0);
        let err = AttributeMessage::parse(&data, 8).unwrap_err();

        assert_eq!(
            err,
            FormatError::ZeroSizedDatatype { class: 2 },
            "the reference bytes decode as a class 2 (time) type of zero width"
        );
    }

    /// The other half of the flag's effect: with it clear over bytes that really
    /// are a datatype, the attribute carries the type it decoded and records
    /// that the type is its own, not a reference to a committed one.
    #[test]
    fn a_datatype_field_without_the_flag_is_recorded_as_inline() {
        let name = b"inline_attr\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_simple_ds_v1(1);

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

    /// Without the file the reference addresses, there is no honest answer, so
    /// the parse refuses rather than decoding the reference as a type.
    #[test]
    fn a_shared_datatype_field_is_refused_without_a_resolver() {
        let data = attr_with_shared_datatype(FLAG_SHARED_DATATYPE);
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnresolvedSharedMessage(MessageType::Datatype.to_u16())
        );
    }

    /// Only two flag bits exist. The C library refuses a message that sets any
    /// other, and a reader that shrugs at one is reading a message it cannot
    /// claim to understand.
    #[test]
    fn an_undefined_attribute_flag_bit_is_refused() {
        let data = attr_with_shared_datatype(0x04);
        let err = AttributeMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidAttributeFlags(0x04));
    }

    /// The name never depends on either field, which is what lets an in-place
    /// edit identify a committed attribute it cannot decode.
    #[test]
    fn a_name_reads_out_of_a_message_whose_datatype_is_a_reference() {
        let data = attr_with_shared_datatype(FLAG_SHARED_DATATYPE);
        assert_eq!(message_name(&data).unwrap(), "shared_attr");
        assert!(AttributeMessage::parse(&data, 8).is_err());
    }

    /// The byte-level screen agrees with the parse, on every version and on
    /// bytes too short to be a message at all.
    #[test]
    fn the_shared_field_screen_reads_the_flags_byte() {
        assert!(message_shares_a_field(&attr_with_shared_datatype(
            FLAG_SHARED_DATATYPE
        )));
        assert!(message_shares_a_field(&attr_with_shared_datatype(
            FLAG_SHARED_DATASPACE
        )));
        assert!(!message_shares_a_field(&attr_with_shared_datatype(0)));
        // Version 1 has no flags byte: the second byte is unused, whatever it says.
        assert!(!message_shares_a_field(&[1u8, 0xFF, 0, 0]));
        // A message this cannot read is refused, not waved through.
        assert!(message_shares_a_field(&[2u8]));
        assert!(message_shares_a_field(&[]));
        assert!(message_shares_a_field(&[9u8, 0]));
    }

    #[test]
    fn read_as_f64_scalar() {
        let name = b"v\0";
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(&3.14f64.to_le_bytes());

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        let vals = attr.read_as_f64().unwrap();
        assert_eq!(vals, vec![3.14]);
    }

    #[test]
    fn read_as_string_fixed() {
        let name = b"s\0";
        let dt_bytes = build_string_dt(5);
        let ds_bytes = build_scalar_ds();

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"world");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        assert_eq!(attr.read_as_string().unwrap(), "world");
    }

    #[test]
    fn read_as_strings_array() {
        let name = b"arr\0";
        let dt_bytes = build_string_dt(4);
        let ds_bytes = build_simple_ds_v1(2);

        let mut data = Vec::new();
        data.push(2);
        data.push(0);
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&dt_bytes);
        data.extend_from_slice(&ds_bytes);
        data.extend_from_slice(b"abcdEFGH");

        let attr = AttributeMessage::parse(&data, 8).unwrap();
        let strs = attr.read_as_strings().unwrap();
        assert_eq!(strs, vec!["abcd", "EFGH"]);
    }

    #[test]
    fn a_null_dataspace_attribute_ignores_record_padding() {
        // Five trailing zero bytes provide the record's 8-byte alignment padding.
        let name = b"empty\0";
        let dt_bytes = build_f64_dt();
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
        let dt_bytes = build_f64_dt();
        let ds_bytes = build_scalar_ds();

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
