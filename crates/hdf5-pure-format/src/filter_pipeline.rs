//! HDF5 Filter Pipeline message parsing and serialization (message type 0x000B).
//!
//! Versions 1 and 2 are defined in "The Data Storage - Filter Pipeline Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt;

use hdf5_pure_core::__private::FILTER_COUNT_MAX;

use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::codec::{Decode, Encode};
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

// Two u16 lengths and the version 1 padding bound each record to fewer than 2^19 bytes.
const _: () = {
    assert!(usize::BITS >= 32);
    assert!(FILTER_COUNT_MAX <= u32::BITS as usize);
};

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

/// A checked Filter Pipeline message with owned filter descriptions.
///
/// A pipeline holds at most 32 filters. [`Self::parse`] checks stored records and [`Self::new`]
/// checks supplied descriptions. Both retain the message version and filter order for encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilterPipeline {
    version: PipelineVersion,
    filters: Vec<FilterDescription>,
    encoded_len: u32,
}

impl FilterPipeline {
    /// Checks the filter descriptions for the selected message version and takes ownership of them.
    ///
    /// # Errors
    ///
    /// Returns the version and field errors of [`PreparedFilterPipeline::new`].
    pub fn new(version: u8, filters: Vec<FilterDescription>) -> Result<Self, FilterPipelineError> {
        let PreparedFilterPipeline {
            version,
            encoded_len,
            filters: _,
        } = PreparedFilterPipeline::new(version, &filters)?;
        Ok(Self {
            version,
            filters,
            encoded_len,
        })
    }

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
    /// assert_eq!(pipeline.version(), 2);
    /// assert_eq!(pipeline.filters(), &[FilterDescription {
    ///     filter_id: 300,
    ///     name: Some("odd".into()),
    ///     flags: 0x83,
    ///     client_data: vec![7],
    /// }]);
    /// assert!(pipeline.filters()[0].is_optional());
    /// # Ok::<(), hdf5_pure_format::__private::FilterPipelineError>(())
    /// ```
    pub fn parse(data: &[u8]) -> Result<FilterPipeline, FilterPipelineError> {
        Self::decode(data, ())
    }

    /// Returns the message version, 1 or 2.
    pub fn version(&self) -> u8 {
        self.version.byte()
    }

    /// Returns the filters in encoding order.
    pub fn filters(&self) -> &[FilterDescription] {
        &self.filters
    }

    /// Returns the owned filter descriptions in encoding order.
    pub fn into_filters(self) -> Vec<FilterDescription> {
        self.filters
    }

    /// Borrows the checked descriptions for encoding using the retained size.
    pub fn prepare(&self) -> PreparedFilterPipeline<'_> {
        PreparedFilterPipeline {
            version: self.version,
            filters: &self.filters,
            encoded_len: self.encoded_len,
        }
    }

    /// Encodes the descriptions in their retained version into an owned message body.
    ///
    /// The writer uses the shortest name field for that version and writes zero padding
    /// where required.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.prepare().to_bytes()
    }
}

/// A borrowed pipeline with checked fields and an exact encoded size.
///
/// The writer can reuse it to encode the message body into a destination slice or an owned
/// buffer. Preparation borrows the filter descriptions and allocates no storage.
#[derive(Debug)]
pub struct PreparedFilterPipeline<'a> {
    version: PipelineVersion,
    filters: &'a [FilterDescription],
    encoded_len: u32,
}

impl<'a> PreparedFilterPipeline<'a> {
    /// Checks and borrows filter descriptions for repeated encoding in the selected version.
    ///
    /// Checks the version and filter count first. For each filter in order, checks its flags,
    /// name, stored name length, and client data count. The stored name length includes the
    /// terminator and any version 1 name padding.
    ///
    /// # Errors
    ///
    /// Returns [`FilterPipelineError::Format`] for an unsupported version or a flag bit set
    /// in positions 8 through 15. Returns [`FilterPipelineError::FieldTooLarge`] if the filter
    /// count exceeds 32, or if the stored name length or client data count exceeds [`u16::MAX`].
    /// Returns [`FilterPipelineError::InvalidName`] for a name containing non-ASCII characters
    /// or a null byte, or for a name supplied for a version 2 filter with an ID below 256.
    pub fn new(version: u8, filters: &'a [FilterDescription]) -> Result<Self, FilterPipelineError> {
        let version = PipelineVersion::try_from(version)?;
        check_field_length("filter count", filters.len(), FILTER_COUNT_MAX)?;
        let mut encoded_len = version.header_len();
        for filter in filters {
            check_filter_flags(filter.filter_id, filter.flags)?;
            if let Some(name) = &filter.name {
                if !version.stores_name_length(filter.filter_id) {
                    return Err(FilterPipelineError::InvalidName {
                        filter_id: filter.filter_id,
                        reason: "predefined filters have no name field in version 2",
                    });
                }
                if !name.is_ascii() || name.as_bytes().contains(&0) {
                    return Err(FilterPipelineError::InvalidName {
                        filter_id: filter.filter_id,
                        reason: "name must be ASCII without null bytes",
                    });
                }
            }
            let name_length = version.name_length(filter.name.as_deref());
            check_field_length("filter name length", name_length, FIELD_LENGTH_MAX)?;
            check_field_length(
                "client data count",
                filter.client_data.len(),
                FIELD_LENGTH_MAX,
            )?;
            encoded_len +=
                version.record_len(filter.filter_id, name_length, filter.client_data.len());
        }
        Ok(Self {
            version,
            filters,
            encoded_len: retain_extent(encoded_len),
        })
    }

    /// Returns the exact message-body size, including name alignment and client data padding.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the body at the start of `output` and returns the number of bytes written.
    ///
    /// Leaves bytes beyond the body unchanged and allocates no storage.
    ///
    /// # Errors
    ///
    /// Returns [`FilterPipelineOutputTooSmall`] if `output` is shorter than
    /// [`Self::encoded_size`]. Leaves the destination unchanged on error.
    ///
    /// # Examples
    ///
    /// Writes a named version 2 filter into a buffer with one extra byte.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{
    ///     FILTER_PIPELINE_VERSION_TWO, FilterDescription, FilterPipelineError,
    ///     PreparedFilterPipeline,
    /// };
    ///
    /// let filters = [FilterDescription {
    ///     filter_id: 300,
    ///     name: Some("odd".into()),
    ///     flags: 0x83,
    ///     client_data: vec![7],
    /// }];
    /// let pipeline = PreparedFilterPipeline::new(FILTER_PIPELINE_VERSION_TWO, &filters)?;
    /// let mut output = [0xa5; 19];
    /// assert_eq!(pipeline.write_into(&mut output), Ok(pipeline.encoded_size()));
    /// assert_eq!(&output, b"\x02\x01\x2c\x01\x04\x00\x83\x00\x01\x00odd\0\x07\0\0\0\xa5");
    /// # Ok::<(), FilterPipelineError>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FilterPipelineOutputTooSmall> {
        self.encode_into(output)
    }

    /// Encodes the body using a single allocation of [`Self::encoded_size`] bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; self.encoded_len()];
        self.write_body(&mut bytes);
        bytes
    }

    /// Writes the encoded message at the start of `body`, leaving unused bytes unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `body` is shorter than [`Self::encoded_size`].
    pub(crate) fn write_body(&self, body: &mut [u8]) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the parsed or prepared filter count is at most 32"
        )]
        let count = self.filters.len() as u8;
        let mut body = write_field(body, &[self.version.byte(), count]);
        if self.version == PipelineVersion::One {
            let (reserved, tail) = body.split_at_mut(VERSION_ONE_RESERVED_SIZE);
            reserved.fill(0);
            body = tail;
        }
        for FilterDescription {
            filter_id,
            name,
            flags,
            client_data,
        } in self.filters
        {
            body = write_field(body, &filter_id.to_le_bytes());
            let name_length = self.version.name_length(name.as_deref());
            if self.version.stores_name_length(*filter_id) {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the parsed or prepared name length for this version fits the u16 field"
                )]
                let stored = name_length as u16;
                body = write_field(body, &stored.to_le_bytes());
            }
            body = write_field(body, &flags.to_le_bytes());
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the parsed or prepared client count fits the u16 field"
            )]
            let count = client_data.len() as u16;
            body = write_field(body, &count.to_le_bytes());
            if let Some(name) = name {
                let (field, tail) = body.split_at_mut(name_length);
                write_field(field, name.as_bytes()).fill(0);
                body = tail;
            }
            let (fields, tail) = body.split_at_mut(client_data.len() * CLIENT_DATA_VALUE_SIZE);
            let (words, _) = fields.as_chunks_mut::<CLIENT_DATA_VALUE_SIZE>();
            for (word, value) in words.iter_mut().zip(client_data) {
                *word = value.to_le_bytes();
            }
            let (padding, tail) = tail.split_at_mut(self.version.client_padding(client_data.len()));
            padding.fill(0);
            body = tail;
        }
    }
}

impl Encode for PreparedFilterPipeline<'_> {
    type Error = FilterPipelineOutputTooSmall;

    fn encoded_len(&self) -> usize {
        self.encoded_len as usize
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let required = self.encoded_len();
        let available = output.len();
        let body = output
            .get_mut(..required)
            .ok_or(FilterPipelineOutputTooSmall {
                required,
                available,
            })?;
        self.write_body(body);
        Ok(required)
    }
}

/// The destination is shorter than the encoded Filter Pipeline body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilterPipelineOutputTooSmall {
    /// The exact number of bytes needed for the body.
    pub required: usize,
    /// The length of the supplied destination in bytes.
    pub available: usize,
}

impl fmt::Display for FilterPipelineOutputTooSmall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "filter pipeline requires {} output bytes, available {}",
            self.required, self.available
        )
    }
}

impl Error for FilterPipelineOutputTooSmall {}

impl From<FilterPipelineOutputTooSmall> for FormatError {
    /// Reports a short destination as [`FormatError::UnexpectedEof`], preserving both lengths.
    fn from(error: FilterPipelineOutputTooSmall) -> Self {
        Self::UnexpectedEof {
            expected: error.required,
            available: error.available,
        }
    }
}

fn retain_extent(encoded_len: usize) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a header and at most 32 records of fewer than 2^19 bytes fit in u32"
    )]
    let encoded_len = encoded_len as u32;
    encoded_len
}

fn write_field<'a>(output: &'a mut [u8], value: &[u8]) -> &'a mut [u8] {
    let (field, rest) = output.split_at_mut(value.len());
    field.copy_from_slice(value);
    rest
}

impl Decode<'_> for FilterPipeline {
    type Context = ();
    type Error = FilterPipelineError;

    fn decode(data: &[u8], (): ()) -> Result<Self, Self::Error> {
        let mut cursor = ByteCursor::new(data);
        let [version, filter_count] = cursor.array::<VERSION_TWO_HEADER_SIZE>()?;
        let version = PipelineVersion::try_from(version)?;
        let reserved = if version == PipelineVersion::One {
            Some(cursor.array()?)
        } else {
            None
        };
        let filter_count = usize::from(filter_count);
        check_field_length("filter count", filter_count, FILTER_COUNT_MAX)?;
        if let Some(reserved) = reserved.filter(|&bytes| bytes != [0; VERSION_ONE_RESERVED_SIZE]) {
            return Err(FormatError::InvalidFilterPipelineReservedBytes(reserved).into());
        }

        let mut filters = Vec::with_capacity(filter_count);
        let mut encoded_len = version.header_len();
        for _ in 0..filter_count {
            let header = FilterRecordHeader::read(&mut cursor, version)?;
            let filter = header.read_body(&mut cursor, version)?;
            encoded_len += version.record_len(
                filter.filter_id,
                version.name_length(filter.name.as_deref()),
                filter.client_data.len(),
            );
            filters.push(filter);
        }
        Ok(Self {
            version,
            filters,
            encoded_len: retain_extent(encoded_len),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PipelineVersion {
    One,
    Two,
}

impl PipelineVersion {
    fn byte(self) -> u8 {
        match self {
            Self::One => VERSION_ONE,
            Self::Two => VERSION_TWO,
        }
    }

    fn header_len(self) -> usize {
        VERSION_TWO_HEADER_SIZE + usize::from(self == Self::One) * VERSION_ONE_RESERVED_SIZE
    }

    fn stores_name_length(self, id: u16) -> bool {
        self == Self::One || id >= PREDEFINED_FILTER_LIMIT
    }

    fn name_length(self, name: Option<&str>) -> usize {
        let Some(name) = name else {
            return 0;
        };
        let length = name.len() + 1;
        match self {
            Self::One => length.next_multiple_of(NAME_ALIGNMENT),
            Self::Two => length,
        }
    }

    fn client_padding(self, client_count: usize) -> usize {
        usize::from(self == Self::One && !client_count.is_multiple_of(2)) * CLIENT_DATA_VALUE_SIZE
    }

    fn record_len(self, id: u16, name_length: usize, client_count: usize) -> usize {
        FILTER_FIELD_SIZE
            + VERSION_TWO_FILTER_FIELDS_SIZE
            + usize::from(self.stores_name_length(id)) * FILTER_FIELD_SIZE
            + name_length
            + client_count * CLIENT_DATA_VALUE_SIZE
            + self.client_padding(client_count)
    }
}

impl TryFrom<u8> for PipelineVersion {
    type Error = FormatError;

    fn try_from(version: u8) -> Result<Self, Self::Error> {
        match version {
            VERSION_ONE => Ok(Self::One),
            VERSION_TWO => Ok(Self::Two),
            _ => Err(FormatError::InvalidFilterPipelineVersion(version)),
        }
    }
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
        check_filter_flags(id, flags)?;
        let name_length = usize::from(name_length);
        let client_count = usize::from(client_count);
        let extent = name_length
            + client_count * CLIENT_DATA_VALUE_SIZE
            + version.client_padding(client_count);
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

/// An error reading a Filter Pipeline body or checking descriptions for encoding.
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

impl From<FilterPipelineError> for FormatError {
    /// Returns the error stored in [`FilterPipelineError::Format`]. Converts name and field
    /// errors to the corresponding [`FormatError`] variants.
    fn from(error: FilterPipelineError) -> Self {
        match error {
            FilterPipelineError::Format(error) => error,
            FilterPipelineError::FieldTooLarge {
                field,
                value,
                maximum,
            } => Self::InvalidFilterPipelineField {
                field,
                value,
                maximum,
            },
            FilterPipelineError::InvalidName { filter_id, reason } => {
                Self::InvalidFilterName { filter_id, reason }
            }
        }
    }
}

impl From<UnexpectedEof> for FilterPipelineError {
    fn from(error: UnexpectedEof) -> Self {
        Self::Format(error.into())
    }
}

fn check_field_length(
    field: &'static str,
    value: usize,
    maximum: usize,
) -> Result<(), FilterPipelineError> {
    if value > maximum {
        return Err(FilterPipelineError::FieldTooLarge {
            field,
            value,
            maximum,
        });
    }
    Ok(())
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

fn check_filter_flags(filter_id: u16, flags: u16) -> Result<(), FormatError> {
    if flags & !FILTER_DEFINITION_FLAG_MASK != 0 {
        return Err(FormatError::InvalidFilterFlags { filter_id, flags });
    }
    Ok(())
}
