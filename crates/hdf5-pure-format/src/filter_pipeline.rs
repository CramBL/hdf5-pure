//! HDF5 Filter Pipeline message parsing and serialization (message type 0x000B).
//!
//! Versions 1 and 2 are defined in "The Data Storage - Filter Pipeline Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::bytes;
use crate::error::FormatError;

// Section `subsubsec_fmt4_dataobject_hdr_msg_filter`, version 4.0, defines the pipeline versions and field widths.
const VERSION_ONE: u8 = 1;
const VERSION_TWO: u8 = 2;
const PREDEFINED_FILTER_LIMIT: u16 = 256;
const VERSION_ONE_HEADER_SIZE: usize = 8;
const VERSION_TWO_HEADER_SIZE: usize = 2;
const VERSION_ONE_FILTER_HEADER_SIZE: usize = 8;
const VERSION_TWO_FILTER_ID_SIZE: usize = 2;
const VERSION_TWO_FILTER_FIELDS_SIZE: usize = 4;
const VERSION_ONE_RESERVED_SIZE: usize = 6;
const NAME_ALIGNMENT: usize = 8;
const CLIENT_DATA_VALUE_SIZE: usize = 4;
const FILTER_COUNT_MAX: usize = 32;
const FIELD_LENGTH_MAX: usize = u16::MAX as usize;

/// Description of a single filter in a pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterDescription {
    /// Filter identification value.
    pub filter_id: u16,
    /// Optional filter name. Version 2 omits its field for filter IDs below 256.
    pub name: Option<String>,
    /// Bit 0 permits omission during output.
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
#[derive(Debug, Clone, PartialEq)]
pub struct FilterPipeline {
    /// Pipeline version (1 or 2).
    pub version: u8,
    /// Ordered list of filters.
    pub filters: Vec<FilterDescription>,
}

impl FilterPipeline {
    /// Parses a filter pipeline message body from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FilterPipelineError::Format`] for an unsupported version or a truncated field,
    /// [`FilterPipelineError::FieldTooLarge`] if the filter count exceeds 32, or
    /// [`FilterPipelineError::InvalidName`] if a stored name is malformed.
    pub fn parse(data: &[u8]) -> Result<FilterPipeline, FilterPipelineError> {
        bytes::ensure_len(data, 0, VERSION_TWO_HEADER_SIZE)?;
        let version = data[0];
        let number_of_filters = data[1] as usize;
        field_length("filter count", number_of_filters, FILTER_COUNT_MAX)?;

        match version {
            VERSION_ONE => Self::parse_v1(data, number_of_filters),
            VERSION_TWO => Self::parse_v2(data, number_of_filters),
            _ => Err(FormatError::InvalidFilterPipelineVersion(version).into()),
        }
    }

    fn parse_v1(
        data: &[u8],
        number_of_filters: usize,
    ) -> Result<FilterPipeline, FilterPipelineError> {
        // version(1) + nfilters(1) + reserved(6) = 8 bytes header
        bytes::ensure_len(data, 0, VERSION_ONE_HEADER_SIZE)?;
        let mut pos = VERSION_ONE_HEADER_SIZE;
        let mut filters = Vec::with_capacity(number_of_filters);

        for _ in 0..number_of_filters {
            bytes::ensure_len(data, pos, VERSION_ONE_FILTER_HEADER_SIZE)?;
            let filter_id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let name_length = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
            let flags = u16::from_le_bytes([data[pos + 4], data[pos + 5]]);
            let num_client_data = u16::from_le_bytes([data[pos + 6], data[pos + 7]]) as usize;
            pos += VERSION_ONE_FILTER_HEADER_SIZE;

            // Name (if present)
            let name = if name_length > 0 {
                bytes::ensure_len(data, pos, name_length)?;
                let name_bytes = &data[pos..pos + name_length];
                let name_str = parse_name(name_bytes, filter_id, VERSION_ONE)?;
                // Pad to 8-byte boundary
                pos += name_length;
                Some(name_str)
            } else {
                None
            };

            // Client data
            bytes::ensure_len(data, pos, num_client_data * CLIENT_DATA_VALUE_SIZE)?;
            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                let val =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                client_data.push(val);
                pos += 4;
            }

            // Padding to 8-byte boundary if `num_client_data` is odd
            if !num_client_data.is_multiple_of(2) {
                bytes::ensure_len(data, pos, CLIENT_DATA_VALUE_SIZE)?;
                pos += CLIENT_DATA_VALUE_SIZE;
            }

            filters.push(FilterDescription {
                filter_id,
                name,
                flags,
                client_data,
            });
        }

        Ok(FilterPipeline {
            version: VERSION_ONE,
            filters,
        })
    }

    fn parse_v2(
        data: &[u8],
        number_of_filters: usize,
    ) -> Result<FilterPipeline, FilterPipelineError> {
        // version(1) + nfilters(1) = 2 bytes header (no reserved in v2)
        let mut pos = VERSION_TWO_HEADER_SIZE;
        let mut filters = Vec::with_capacity(number_of_filters);

        for _ in 0..number_of_filters {
            bytes::ensure_len(data, pos, VERSION_TWO_FILTER_ID_SIZE)?;
            let filter_id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += VERSION_TWO_FILTER_ID_SIZE;

            let name_length = if filter_id >= PREDEFINED_FILTER_LIMIT {
                bytes::ensure_len(data, pos, VERSION_TWO_FILTER_ID_SIZE)?;
                let nl = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += VERSION_TWO_FILTER_ID_SIZE;
                nl
            } else {
                0
            };

            bytes::ensure_len(data, pos, VERSION_TWO_FILTER_FIELDS_SIZE)?;
            let flags = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let num_client_data = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
            pos += VERSION_TWO_FILTER_FIELDS_SIZE;

            let name = if name_length > 0 {
                bytes::ensure_len(data, pos, name_length)?;
                let name_bytes = &data[pos..pos + name_length];
                let name_str = parse_name(name_bytes, filter_id, VERSION_TWO)?;
                pos += name_length;
                Some(name_str)
            } else {
                None
            };

            bytes::ensure_len(data, pos, num_client_data * CLIENT_DATA_VALUE_SIZE)?;
            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                let val =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                client_data.push(val);
                pos += 4;
            }

            // No padding in v2

            filters.push(FilterDescription {
                filter_id,
                name,
                flags,
                client_data,
            });
        }

        Ok(FilterPipeline {
            version: VERSION_TWO,
            filters,
        })
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
    /// An unsupported pipeline version or truncated message body.
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

fn parse_name(bytes: &[u8], filter_id: u16, version: u8) -> Result<String, FilterPipelineError> {
    if version == VERSION_ONE && !bytes.len().is_multiple_of(NAME_ALIGNMENT) {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "name length is not a multiple of eight",
        });
    }
    let Some(terminator) = bytes.iter().position(|&byte| byte == 0) else {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "missing null terminator",
        });
    };
    if version == VERSION_ONE && bytes[terminator..].iter().any(|&byte| byte != 0) {
        return Err(FilterPipelineError::InvalidName {
            filter_id,
            reason: "nonzero byte after null terminator",
        });
    }
    if version == VERSION_TWO && terminator + 1 != bytes.len() {
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
    Ok(name.to_string())
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
