//! HDF5 Filter Pipeline message parsing and serialization (message type 0x000B).
//!
//! Versions 1 and 2 are defined in "The Data Storage - Filter Pipeline Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use hdf5_pure_core::__private::FILTER_COUNT_MAX;

use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::codec::Decode;
use crate::error::FormatError;

// Section `subsubsec_fmt4_dataobject_hdr_msg_filter`, version 4.0, defines the pipeline versions and field widths.
const VERSION_ONE: u8 = 1;
/// Version 2 of the Filter Pipeline message, where only a filter with an identifier of 256 or
/// above stores the Name Length field, and the Name field where that length is not zero. The
/// message is defined in "The Data Storage - Filter Pipeline Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub const VERSION_TWO: u8 = 2;
const PREDEFINED_FILTER_LIMIT: u16 = 256;
const VERSION_TWO_HEADER_SIZE: usize = 2;
const VERSION_ONE_FILTER_HEADER_SIZE: usize = 8;
const FILTER_FIELD_SIZE: usize = 2;
const VERSION_TWO_FILTER_FIELDS_SIZE: usize = 4;
const VERSION_ONE_RESERVED_SIZE: usize = 6;
const NAME_ALIGNMENT: usize = 8;
const CLIENT_DATA_VALUE_SIZE: usize = 4;
const FIELD_LENGTH_MAX: usize = u16::MAX as usize;

/// The stored configuration of one filter in a pipeline message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilterDescription {
    /// Filter identification value.
    pub filter_id: u16,
    /// The recorded ASCII name, where present. Version 2 omits the field for IDs below 256.
    /// A zero stored length represents an absent name. A terminator with zero preceding
    /// characters represents an empty name.
    pub name: Option<String>,
    /// Bit 0 allows the writer to skip the filter if applying it fails. The parser preserves
    /// bits 1 through 7 as opaque flags and rejects set bits in positions 8 through 15.
    pub flags: u16,
    /// Client data values passed to the filter.
    pub client_data: Vec<u32>,
}

impl FilterDescription {
    /// Returns whether this filter is marked optional for output.
    pub fn is_optional(&self) -> bool {
        self.flags & FILTER_FLAG_OPTIONAL != 0
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_filter`, version 4.0, assigns bit zero to optional output.
const FILTER_FLAG_OPTIONAL: u16 = 0x0001;

/// A versioned filter pipeline message and its ordered filter descriptions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilterPipeline {
    /// Pipeline version (1 or 2).
    pub version: u8,
    /// Ordered list of filters.
    pub filters: Vec<FilterDescription>,
}

impl FilterPipeline {
    /// Parses a version 1 or 2 Filter Pipeline message body.
    ///
    /// Reads the version and count before checking the version. For version 1, reads the six
    /// reserved bytes before checking the count, then checks those bytes for zero.
    ///
    /// Reads each version 1 filter header as one span. For version 2, reads the identifier
    /// first, then the rest of the header selected by that identifier. After reading the header,
    /// checks that the version 1 name length is a multiple of eight, then checks the flags in
    /// either version. Reads the name field, client data, and required client data padding as
    /// one span before checking the name bytes and then the client data padding. Ignores bytes
    /// after the last declared filter.
    ///
    /// The parser accepts the eight definition flag bits in `H5Z_FLAG_DEFMASK` (`H5Zpublic.h`,
    /// HDF5 2.2.0). A stored name consists of ASCII bytes and a null terminator. In version 1,
    /// the parser requires zero padding to a multiple of eight bytes. For version 2,
    /// `H5O__pline_encode` writes the name through its terminator (`H5Opline.c`, HDF5 2.2.0).
    /// The parser requires the terminator to end the version 2 name field.
    ///
    /// # Errors
    ///
    /// Returns [`FilterPipelineError::Format`] for an unsupported version, truncated input,
    /// a flag bit set above bit 7, or nonzero version 1 reserved bytes or client data padding.
    /// For [`FormatError::UnexpectedEof`], `expected` is the end of the first incomplete span
    /// and `available` is the length of `data`, both in bytes.
    ///
    /// Returns [`FilterPipelineError::FieldTooLarge`] if the filter count exceeds 32, or
    /// [`FilterPipelineError::InvalidName`] if a stored name's length or contents are invalid.
    ///
    /// # Examples
    ///
    /// Reads a version 2 pipeline with an optional custom filter.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{FilterDescription, FilterPipeline};
    ///
    /// let body = b"\x02\x01\x2c\x01\x04\x00\x83\x00\x01\x00odd\0\x07\0\0\0";
    /// let pipeline = FilterPipeline::parse(body)?;
    /// assert_eq!(pipeline.version, 2);
    /// assert_eq!(pipeline.filters, [FilterDescription {
    ///     filter_id: 300,
    ///     name: Some("odd".into()),
    ///     flags: 0x83,
    ///     client_data: vec![7],
    /// }]);
    /// assert!(pipeline.filters[0].is_optional());
    /// # Ok::<(), hdf5_pure_format::__private::FilterPipelineError>(())
    /// ```
    pub fn parse(data: &[u8]) -> Result<FilterPipeline, FilterPipelineError> {
        Self::decode(data, ())
    }

    /// Serializes the filter pipeline into message-body bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FilterPipelineError::Format`] if the version is unsupported,
    /// [`FilterPipelineError::FieldTooLarge`] if the filter count exceeds 32 or a stored length
    /// exceeds 65,535, or
    /// [`FilterPipelineError::InvalidName`] if a name cannot be encoded.
    pub fn serialize(&self) -> Result<Vec<u8>, FilterPipelineError> {
        field_length("filter count", self.filters.len(), FILTER_COUNT_MAX)?;
        match self.version {
            VERSION_ONE => self.serialize_v1(),
            VERSION_TWO => self.serialize_v2(),
            _ => Err(FormatError::InvalidFilterPipelineVersion(self.version).into()),
        }
    }

    fn serialize_v1(&self) -> Result<Vec<u8>, FilterPipelineError> {
        let mut buf = Vec::new();
        buf.push(VERSION_ONE);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "serialize checks the filter count against the u8 field maximum before calling this method"
        )]
        buf.push(self.filters.len() as u8);
        buf.extend_from_slice(&[0u8; VERSION_ONE_RESERVED_SIZE]);

        for f in &self.filters {
            buf.extend_from_slice(&f.filter_id.to_le_bytes());

            let name_bytes = encoded_name(f.name.as_deref(), f.filter_id, VERSION_ONE)?;
            let name_length =
                field_length("filter name length", name_bytes.len(), FIELD_LENGTH_MAX)?;
            buf.extend_from_slice(&name_length.to_le_bytes());
            buf.extend_from_slice(&f.flags.to_le_bytes());
            let client_count =
                field_length("client data count", f.client_data.len(), FIELD_LENGTH_MAX)?;
            buf.extend_from_slice(&client_count.to_le_bytes());

            buf.extend_from_slice(&name_bytes);

            for &val in &f.client_data {
                buf.extend_from_slice(&val.to_le_bytes());
            }

            // Pad if odd number of client data values
            if f.client_data.len() % 2 != 0 {
                buf.extend_from_slice(&[0u8; CLIENT_DATA_VALUE_SIZE]);
            }
        }

        Ok(buf)
    }

    fn serialize_v2(&self) -> Result<Vec<u8>, FilterPipelineError> {
        let mut buf = Vec::new();
        buf.push(VERSION_TWO);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "serialize checks the filter count against the u8 field maximum before calling this method"
        )]
        buf.push(self.filters.len() as u8);

        for f in &self.filters {
            buf.extend_from_slice(&f.filter_id.to_le_bytes());

            if f.filter_id >= PREDEFINED_FILTER_LIMIT {
                let name_bytes = encoded_name(f.name.as_deref(), f.filter_id, VERSION_TWO)?;
                let name_length =
                    field_length("filter name length", name_bytes.len(), FIELD_LENGTH_MAX)?;
                buf.extend_from_slice(&name_length.to_le_bytes());
                buf.extend_from_slice(&f.flags.to_le_bytes());
                let client_count =
                    field_length("client data count", f.client_data.len(), FIELD_LENGTH_MAX)?;
                buf.extend_from_slice(&client_count.to_le_bytes());
                buf.extend_from_slice(&name_bytes);
            } else {
                if f.name.is_some() {
                    return Err(FilterPipelineError::InvalidName {
                        filter_id: f.filter_id,
                        reason: "predefined filters have no name field in version 2",
                    });
                }
                buf.extend_from_slice(&f.flags.to_le_bytes());
                let client_count =
                    field_length("client data count", f.client_data.len(), FIELD_LENGTH_MAX)?;
                buf.extend_from_slice(&client_count.to_le_bytes());
            }

            for &val in &f.client_data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        Ok(buf)
    }
}

impl Decode<'_> for FilterPipeline {
    type Context = ();
    type Error = FilterPipelineError;

    fn decode(data: &[u8], (): ()) -> Result<Self, Self::Error> {
        let mut cursor = ByteCursor::new(data);
        let [version, filter_count] = cursor.array::<VERSION_TWO_HEADER_SIZE>()?;
        let (layout, reserved) = match version {
            VERSION_ONE => (PipelineVersion::One, Some(cursor.array()?)),
            VERSION_TWO => (PipelineVersion::Two, None),
            version => return Err(FormatError::InvalidFilterPipelineVersion(version).into()),
        };
        let filter_count = usize::from(filter_count);
        field_length("filter count", filter_count, FILTER_COUNT_MAX)?;
        if let Some(reserved) = reserved.filter(|&bytes| bytes != [0; VERSION_ONE_RESERVED_SIZE]) {
            return Err(FormatError::InvalidFilterPipelineReservedBytes(reserved).into());
        }

        let mut filters = Vec::with_capacity(filter_count);
        for _ in 0..filter_count {
            let header = FilterRecordHeader::read(&mut cursor, layout)?;
            filters.push(header.read_body(&mut cursor, layout)?);
        }
        Ok(Self { version, filters })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PipelineVersion {
    One,
    Two,
}

struct FilterRecordHeader {
    id: u16,
    name_length: u16,
    flags: u16,
    client_count: u16,
}

impl FilterRecordHeader {
    fn read(cursor: &mut ByteCursor<'_>, version: PipelineVersion) -> Result<Self, UnexpectedEof> {
        let (id, name_length, [flags_lo, flags_hi, count_lo, count_hi]) = match version {
            PipelineVersion::One => {
                let [
                    id_lo,
                    id_hi,
                    name_length_lo,
                    name_length_hi,
                    flags_lo,
                    flags_hi,
                    count_lo,
                    count_hi,
                ] = cursor.array::<VERSION_ONE_FILTER_HEADER_SIZE>()?;
                (
                    u16::from_le_bytes([id_lo, id_hi]),
                    u16::from_le_bytes([name_length_lo, name_length_hi]),
                    [flags_lo, flags_hi, count_lo, count_hi],
                )
            }
            PipelineVersion::Two => {
                let id = u16::from_le_bytes(cursor.array::<FILTER_FIELD_SIZE>()?);
                if id < PREDEFINED_FILTER_LIMIT {
                    (id, 0, cursor.array::<VERSION_TWO_FILTER_FIELDS_SIZE>()?)
                } else {
                    let [
                        name_length_lo,
                        name_length_hi,
                        flags_lo,
                        flags_hi,
                        count_lo,
                        count_hi,
                    ] = cursor.array::<{ FILTER_FIELD_SIZE + VERSION_TWO_FILTER_FIELDS_SIZE }>()?;
                    (
                        id,
                        u16::from_le_bytes([name_length_lo, name_length_hi]),
                        [flags_lo, flags_hi, count_lo, count_hi],
                    )
                }
            }
        };
        Ok(Self {
            id,
            name_length,
            flags: u16::from_le_bytes([flags_lo, flags_hi]),
            client_count: u16::from_le_bytes([count_lo, count_hi]),
        })
    }

    fn read_body(
        self,
        cursor: &mut ByteCursor<'_>,
        version: PipelineVersion,
    ) -> Result<FilterDescription, FilterPipelineError> {
        if version == PipelineVersion::One
            && !usize::from(self.name_length).is_multiple_of(NAME_ALIGNMENT)
        {
            return Err(FilterPipelineError::InvalidName {
                filter_id: self.id,
                reason: "name length is not a multiple of eight",
            });
        }
        let Self {
            id,
            name_length,
            flags,
            client_count,
        } = self;
        if flags & !FILTER_DEFINITION_FLAG_MASK != 0 {
            return Err(FormatError::InvalidFilterFlags {
                filter_id: id,
                flags,
            }
            .into());
        }
        const { assert!(usize::BITS >= 32) };
        let name_length = usize::from(name_length);
        let client_count = usize::from(client_count);
        let has_padding = version == PipelineVersion::One && !client_count.is_multiple_of(2);
        let extent =
            name_length + (client_count + usize::from(has_padding)) * CLIENT_DATA_VALUE_SIZE;
        let (name, client_fields) = cursor.take(extent)?.split_at(name_length);
        let (words, _) = client_fields.as_chunks::<CLIENT_DATA_VALUE_SIZE>();
        let (client_values, padding) = words.split_at(client_count);

        let name = if name.is_empty() {
            None
        } else {
            Some(parse_name(name, id, version)?)
        };
        if let Some(&padding) = padding
            .first()
            .filter(|&&word| word != [0; CLIENT_DATA_VALUE_SIZE])
        {
            return Err(FormatError::InvalidFilterClientDataPadding {
                filter_id: id,
                padding,
            }
            .into());
        }
        Ok(FilterDescription {
            filter_id: id,
            name: name.map(ToString::to_string),
            flags,
            client_data: client_values
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect(),
        })
    }
}

// H5Z_FLAG_DEFMASK in H5Zpublic.h, HDF5 2.2.0, bounds the stored filter definition flags.
const FILTER_DEFINITION_FLAG_MASK: u16 = 0x00ff;

/// Errors from parsing or serializing a Filter Pipeline message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterPipelineError {
    /// A filter count or stored field length exceeds its format limit.
    FieldTooLarge {
        /// The field whose value exceeds the limit.
        field: &'static str,
        /// The count or length supplied.
        value: usize,
        /// The maximum value permitted for this field by the format.
        maximum: usize,
    },
    /// A format error in the version, framing, flags, reserved bytes, or client data padding.
    Format(FormatError),
    /// A filter name cannot be parsed or serialized in its message version.
    InvalidName {
        /// Identifier of the filter with the invalid name.
        filter_id: u16,
        /// The name constraint that failed.
        reason: &'static str,
    },
}

impl From<FormatError> for FilterPipelineError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<UnexpectedEof> for FilterPipelineError {
    fn from(error: UnexpectedEof) -> Self {
        Self::Format(error.into())
    }
}

fn field_length(
    field: &'static str,
    value: usize,
    maximum: usize,
) -> Result<u16, FilterPipelineError> {
    if value > maximum {
        return Err(FilterPipelineError::FieldTooLarge {
            field,
            value,
            maximum,
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value has passed the field maximum, which is at most u16::MAX"
    )]
    Ok(value as u16)
}

fn parse_name(
    bytes: &[u8],
    filter_id: u16,
    version: PipelineVersion,
) -> Result<&str, FilterPipelineError> {
    let Some(terminator) = bytes.iter().position(|&byte| byte == 0) else {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "missing null terminator",
        });
    };
    if version == PipelineVersion::One && bytes[terminator..].iter().any(|&byte| byte != 0) {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "nonzero byte after null terminator",
        });
    }
    if version == PipelineVersion::Two && terminator + 1 != bytes.len() {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "bytes after null terminator",
        });
    }
    let name = core::str::from_utf8(&bytes[..terminator]).map_err(|_invalid_utf8| {
        FilterPipelineError::InvalidName {
            filter_id,
            reason: "invalid UTF-8",
        }
    })?;
    if !name.is_ascii() {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "non-ASCII name",
        });
    }
    Ok(name)
}

fn encoded_name(
    name: Option<&str>,
    filter_id: u16,
    version: u8,
) -> Result<Vec<u8>, FilterPipelineError> {
    let Some(name) = name else {
        return Ok(Vec::new());
    };
    if !name.is_ascii() || name.contains('\0') {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "name must be ASCII without null bytes",
        });
    }
    let length = if version == VERSION_ONE {
        (name.len() + 1).next_multiple_of(NAME_ALIGNMENT)
    } else {
        name.len() + 1
    };
    field_length("filter name length", length, FIELD_LENGTH_MAX)?;
    let mut bytes = name.as_bytes().to_vec();
    bytes.resize(length, 0);
    Ok(bytes)
}
