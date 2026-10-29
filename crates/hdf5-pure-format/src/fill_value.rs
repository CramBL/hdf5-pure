//! Fill value message parsing and serialization.

use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::FillTime;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_type::MessageType;

/// An error indicating that serializing a fill value message failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FillValueError {
    /// A fill value length exceeds the 4-byte format limit.
    TooLarge {
        /// The byte length that exceeds the format limit.
        length: usize,
    },
}

/// A borrowed view of a parsed fill value message.
///
/// The message is defined in "The Data Storage - Fill Value Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fvmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FillValueView<'a> {
    fill_time: FillTime,
    value: Option<&'a [u8]>,
}

impl<'a> FillValueView<'a> {
    /// Creates a fill value view from a fill time policy and an optional borrowed value.
    pub const fn new(fill_time: FillTime, value: Option<&'a [u8]>) -> Self {
        Self { fill_time, value }
    }

    /// Parses a fill value message from its message type and payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFillValueMessageType`] if `msg_type` is not a fill value
    /// message type, [`FormatError::UnexpectedEof`] if `data` is truncated, or another
    /// [`FormatError`] if the message contents are malformed.
    pub fn parse(msg_type: MessageType, data: &'a [u8]) -> Result<Self, FormatError> {
        match msg_type {
            MessageType::FILL_VALUE_OLD => Self::parse_old(data),
            MessageType::FILL_VALUE => Self::parse_versioned(data),
            other => Err(FormatError::InvalidFillValueMessageType(other.to_u16())),
        }
    }

    /// Returns the fill time policy stored in the message.
    pub const fn fill_time(&self) -> FillTime {
        self.fill_time
    }

    /// Returns the borrowed fill value bytes, or `None` if no fill value is defined.
    pub const fn value(&self) -> Option<&'a [u8]> {
        self.value
    }

    /// Copies this view into an owned [`FillValueMessage`].
    pub fn to_owned(&self) -> FillValueMessage {
        FillValueMessage {
            fill_time: self.fill_time,
            value: self.value.map(Vec::from),
        }
    }

    /// Serializes this view as a version 3 fill value message.
    ///
    /// # Errors
    ///
    /// Returns [`FillValueError::TooLarge`] if the fill value length exceeds [`u32::MAX`].
    pub fn serialize(&self) -> Result<Vec<u8>, FillValueError> {
        serialize_fill_value_message_v3(self.fill_time, self.value)
    }

    fn parse_old(data: &'a [u8]) -> Result<Self, FormatError> {
        if data.len() < 4 {
            return Err(eof(4, data.len()));
        }
        let size = read_u32(data, 0)?.to_usize()?;
        let end = 4usize
            .checked_add(size)
            .ok_or(eof(usize::MAX, data.len()))?;
        if data.len() < end {
            return Err(eof(end, data.len()));
        }
        let value = if size == 0 { None } else { Some(&data[4..end]) };
        Ok(Self {
            fill_time: FillTime::IfSet,
            value,
        })
    }

    fn parse_versioned(data: &'a [u8]) -> Result<Self, FormatError> {
        let version = *data.first().ok_or(eof(1, data.len()))?;
        match version {
            VERSION_ONE => Self::parse_v1(data),
            VERSION_TWO => Self::parse_v2(data),
            VERSION_THREE => Self::parse_v3(data),
            other => Err(FormatError::UnsupportedFillValueVersion(other)),
        }
    }

    fn parse_v1(data: &'a [u8]) -> Result<Self, FormatError> {
        if data.len() < 4 {
            return Err(eof(4, data.len()));
        }
        let space_alloc_time = data[1];
        if space_alloc_time > SPACE_ALLOC_TIME_MAX {
            return Err(FormatError::InvalidSpaceAllocationTime(space_alloc_time));
        }
        let fill_time = FillTime::try_from(data[2])?;
        let defined = data[3];
        if defined > FILL_DEFINED_MAX {
            return Err(FormatError::InvalidFillValueDefined(defined));
        }
        if data.len() < V1_V2_VALUE_OFFSET {
            return Err(eof(V1_V2_VALUE_OFFSET, data.len()));
        }
        let size = read_u32(data, 4)?.to_usize()?;
        let end = V1_V2_VALUE_OFFSET
            .checked_add(size)
            .ok_or(eof(usize::MAX, data.len()))?;
        if data.len() < end {
            return Err(eof(end, data.len()));
        }
        let value = if defined == 0 || size == 0 {
            None
        } else {
            Some(&data[V1_V2_VALUE_OFFSET..end])
        };
        Ok(Self { fill_time, value })
    }

    fn parse_v2(data: &'a [u8]) -> Result<Self, FormatError> {
        if data.len() < 4 {
            return Err(eof(4, data.len()));
        }
        let space_alloc_time = data[1];
        if space_alloc_time > SPACE_ALLOC_TIME_MAX {
            return Err(FormatError::InvalidSpaceAllocationTime(space_alloc_time));
        }
        let fill_time = FillTime::try_from(data[2])?;
        let defined = data[3];
        if defined > FILL_DEFINED_MAX {
            return Err(FormatError::InvalidFillValueDefined(defined));
        }
        if defined == 0 {
            return Ok(Self {
                fill_time,
                value: None,
            });
        }
        if data.len() < V1_V2_VALUE_OFFSET {
            return Err(eof(V1_V2_VALUE_OFFSET, data.len()));
        }
        let size = read_u32(data, 4)?.to_usize()?;
        let end = V1_V2_VALUE_OFFSET
            .checked_add(size)
            .ok_or(eof(usize::MAX, data.len()))?;
        if data.len() < end {
            return Err(eof(end, data.len()));
        }
        let value = if size == 0 {
            None
        } else {
            Some(&data[V1_V2_VALUE_OFFSET..end])
        };
        Ok(Self { fill_time, value })
    }

    fn parse_v3(data: &'a [u8]) -> Result<Self, FormatError> {
        if data.len() < 2 {
            return Err(eof(2, data.len()));
        }
        let flags = data[1];
        if flags & V3_FLAGS_RESERVED != 0 {
            return Err(FormatError::InvalidFillValueFlags(flags));
        }
        if (flags & V3_FLAG_UNDEFINED != 0) && (flags & V3_FLAG_DEFINED != 0) {
            return Err(FormatError::InvalidFillValueFlags(flags));
        }
        let fill_time_code = (flags >> V3_FILL_TIME_SHIFT) & V3_FILL_TIME_MASK;
        let fill_time = FillTime::try_from(fill_time_code)?;

        if flags & V3_FLAG_DEFINED == 0 {
            return Ok(Self {
                fill_time,
                value: None,
            });
        }
        if data.len() < V3_VALUE_OFFSET {
            return Err(eof(V3_VALUE_OFFSET, data.len()));
        }
        let size = read_u32(data, 2)?.to_usize()?;
        let end = V3_VALUE_OFFSET
            .checked_add(size)
            .ok_or(eof(usize::MAX, data.len()))?;
        if data.len() < end {
            return Err(eof(end, data.len()));
        }
        let value = if size == 0 {
            None
        } else {
            Some(&data[V3_VALUE_OFFSET..end])
        };
        Ok(Self { fill_time, value })
    }
}

/// An owned fill value message.
///
/// The message is defined in "The Data Storage - Fill Value Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fvmessage
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FillValueMessage {
    fill_time: FillTime,
    value: Option<Vec<u8>>,
}

impl FillValueMessage {
    /// Creates an owned fill value message from a fill time policy and an optional value.
    pub const fn new(fill_time: FillTime, value: Option<Vec<u8>>) -> Self {
        Self { fill_time, value }
    }

    /// Returns the fill time policy stored in the message.
    pub const fn fill_time(&self) -> FillTime {
        self.fill_time
    }

    /// Returns the fill value bytes, or `None` if no fill value is defined.
    pub fn value(&self) -> Option<&[u8]> {
        self.value.as_deref()
    }

    /// Consumes the message and returns the fill value bytes, or `None` if no
    /// fill value is defined.
    pub fn into_value(self) -> Option<Vec<u8>> {
        self.value
    }

    /// Returns a borrowed view of this message.
    pub fn as_view(&self) -> FillValueView<'_> {
        FillValueView {
            fill_time: self.fill_time,
            value: self.value.as_deref(),
        }
    }

    /// Serializes this message as a version 3 fill value message.
    ///
    /// # Errors
    ///
    /// Returns [`FillValueError::TooLarge`] if the fill value length exceeds [`u32::MAX`].
    pub fn serialize(&self) -> Result<Vec<u8>, FillValueError> {
        serialize_fill_value_message_v3(self.fill_time, self.value.as_deref())
    }
}

/// Serializes a fill time policy and an optional fill value as a version 3 fill value message.
///
/// # Errors
///
/// Returns [`FillValueError::TooLarge`] if the fill value length exceeds [`u32::MAX`].
pub fn serialize_fill_value_message_v3(
    fill_time: FillTime,
    fill: Option<&[u8]>,
) -> Result<Vec<u8>, FillValueError> {
    let mut flags = (V3_FLAGS_DEFAULT & !(V3_FILL_TIME_MASK << V3_FILL_TIME_SHIFT))
        | (fill_time.to_u8() << V3_FILL_TIME_SHIFT);
    Ok(match fill {
        None => vec![VERSION_THREE, flags],
        Some(bytes) => {
            let len = fill_value_len(bytes.len())?;
            flags |= V3_FLAG_DEFINED;
            let mut msg = Vec::with_capacity(V3_VALUE_OFFSET + bytes.len());
            msg.push(VERSION_THREE);
            msg.push(flags);
            msg.extend_from_slice(&len.to_le_bytes());
            msg.extend_from_slice(bytes);
            msg
        }
    })
}

/// Serializes an optional fill value with the default fill time as a version 3 fill value message.
///
/// # Errors
///
/// Returns [`FillValueError::TooLarge`] if the fill value length exceeds [`u32::MAX`].
pub fn fill_value_message_v3(fill: Option<&[u8]>) -> Result<Vec<u8>, FillValueError> {
    serialize_fill_value_message_v3(FillTime::IfSet, fill)
}

/// Returns `true` if the fill value message specifies that the fill value is written to storage.
///
/// # Errors
///
/// Returns [`FormatError::InvalidFillValueMessageType`] if `msg_type` is not a fill value
/// message type, [`FormatError::UnexpectedEof`] if `data` is truncated, or another
/// [`FormatError`] if the message contents are malformed.
pub fn fill_value_is_written(msg_type: MessageType, data: &[u8]) -> Result<bool, FormatError> {
    let view = FillValueView::parse(msg_type, data)?;
    Ok(!view.fill_time().is_never())
}

/// Parses a defined fill value from message bytes, returning `None` if no fill value is defined.
///
/// # Errors
///
/// Returns [`FormatError::InvalidFillValueMessageType`] if `msg_type` is not a fill value
/// message type, [`FormatError::UnexpectedEof`] if `data` is truncated, or another
/// [`FormatError`] if the message contents are malformed.
pub fn parse_defined_fill_value(
    msg_type: MessageType,
    data: &[u8],
) -> Result<Option<Vec<u8>>, FormatError> {
    let view = FillValueView::parse(msg_type, data)?;
    Ok(view.value().map(Vec::from))
}

fn fill_value_len(len: usize) -> Result<u32, FillValueError> {
    match u32::try_from(len) {
        Ok(length) => Ok(length),
        Err(_out_of_range) => Err(FillValueError::TooLarge { length: len }),
    }
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, FormatError> {
    let end = offset.checked_add(4).ok_or(eof(usize::MAX, data.len()))?;
    let slice = data.get(offset..end).ok_or(eof(end, data.len()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn eof(expected: usize, available: usize) -> FormatError {
    FormatError::UnexpectedEof {
        expected,
        available,
    }
}

/// The default flags byte for a version 3 fill value message with late allocation and write time if set.
pub const V3_FLAGS_DEFAULT: u8 = 0x0a;

// "The Data Storage - Fill Value Message" of the format specification, version 4.0, defines the fill-value versions and value offsets.
const VERSION_ONE: u8 = 1;
const VERSION_TWO: u8 = 2;
const VERSION_THREE: u8 = 3;
const V1_V2_VALUE_OFFSET: usize = 8;
const V3_VALUE_OFFSET: usize = 6;

const V3_FLAG_UNDEFINED: u8 = 0x10;
const V3_FLAG_DEFINED: u8 = 0x20;
const V3_FLAGS_RESERVED: u8 = 0xc0;
const V3_FILL_TIME_MASK: u8 = 0x03;
const V3_FILL_TIME_SHIFT: u8 = 2;

const SPACE_ALLOC_TIME_MAX: u8 = 3;
const FILL_DEFINED_MAX: u8 = 1;

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn default_message_is_unchanged() {
        assert_eq!(fill_value_message_v3(None).unwrap(), vec![3, 0x0a]);
    }

    #[test]
    fn defined_message_matches_reference_library_bytes() {
        let msg = fill_value_message_v3(Some(&(-7i32).to_le_bytes())).unwrap();
        assert_eq!(msg, vec![3, 0x2a, 4, 0, 0, 0, 0xf9, 0xff, 0xff, 0xff]);
    }

    #[rstest]
    #[case::alloc_absent(FillTime::Alloc, None, vec![3, 0x02])]
    #[case::alloc_defined(FillTime::Alloc, Some(&[1u8, 2, 3, 4][..]), vec![3, 0x22, 4, 0, 0, 0, 1, 2, 3, 4])]
    #[case::never_absent(FillTime::Never, None, vec![3, 0x06])]
    #[case::never_defined(FillTime::Never, Some(&[1u8, 2, 3, 4][..]), vec![3, 0x26, 4, 0, 0, 0, 1, 2, 3, 4])]
    #[case::if_set_absent(FillTime::IfSet, None, vec![3, 0x0a])]
    #[case::if_set_defined(FillTime::IfSet, Some(&[1u8, 2, 3, 4][..]), vec![3, 0x2a, 4, 0, 0, 0, 1, 2, 3, 4])]
    fn canonical_v3_serialization(
        #[case] fill_time: FillTime,
        #[case] fill: Option<&[u8]>,
        #[case] expected: Vec<u8>,
    ) {
        let msg = serialize_fill_value_message_v3(fill_time, fill).unwrap();
        assert_eq!(msg, expected);

        let view = FillValueView::parse(MessageType::FILL_VALUE, &msg).unwrap();
        assert_eq!(view.fill_time(), fill_time);
        assert_eq!(view.value(), fill);

        let owned = view.to_owned();
        assert_eq!(owned.fill_time(), fill_time);
        assert_eq!(owned.value(), fill);
        assert_eq!(owned.serialize().unwrap(), expected);
        assert_eq!(view.serialize().unwrap(), expected);
    }

    #[test]
    fn default_fill_value_message_matches_unspecified_library_default() {
        let msg = FillValueMessage::default();
        assert_eq!(msg.fill_time(), FillTime::IfSet);
        assert_eq!(msg.value(), None);
        assert_eq!(msg.serialize().unwrap(), vec![3, 0x0a]);
    }

    #[rstest]
    #[case::version_one(
        MessageType::FILL_VALUE,
        &[1u8, 2, 1, 1, 4, 0, 0, 0, 42, 0, 0, 0],
        FillTime::Never,
        Some(&[42u8, 0, 0, 0][..])
    )]
    #[case::version_two(
        MessageType::FILL_VALUE,
        &[2u8, 2, 1, 1, 4, 0, 0, 0, 42, 0, 0, 0],
        FillTime::Never,
        Some(&[42u8, 0, 0, 0][..])
    )]
    #[case::version_three(
        MessageType::FILL_VALUE,
        &[3u8, 0x26, 4, 0, 0, 0, 42, 0, 0, 0],
        FillTime::Never,
        Some(&[42u8, 0, 0, 0][..])
    )]
    fn defined_fill_value_and_never_policy_coexist(
        #[case] msg_type: MessageType,
        #[case] bytes: &[u8],
        #[case] expected_time: FillTime,
        #[case] expected_value: Option<&[u8]>,
    ) {
        let view = FillValueView::parse(msg_type, bytes).unwrap();
        assert_eq!(view.fill_time(), expected_time);
        assert_eq!(view.value(), expected_value);
        assert!(!fill_value_is_written(msg_type, bytes).unwrap());
        assert_eq!(
            parse_defined_fill_value(msg_type, bytes).unwrap(),
            expected_value.map(Vec::from)
        );
    }

    #[rstest]
    #[case::v1_defined(
        MessageType::FILL_VALUE,
        &[1u8, 2, 2, 1, 2, 0, 0, 0, 0xed, 0xfe],
        FillTime::IfSet,
        Some(&[0xed, 0xfe][..])
    )]
    #[case::v1_undefined_trailing_bytes(
        MessageType::FILL_VALUE,
        &[1u8, 2, 2, 0, 4, 0, 0, 0, 1, 2, 3, 4],
        FillTime::IfSet,
        None
    )]
    #[case::v1_zero_length(
        MessageType::FILL_VALUE,
        &[1u8, 2, 2, 1, 0, 0, 0, 0],
        FillTime::IfSet,
        None
    )]
    #[case::v2_defined(
        MessageType::FILL_VALUE,
        &[2u8, 2, 2, 1, 4, 0, 0, 0, 0xf9, 0xff, 0xff, 0xff],
        FillTime::IfSet,
        Some(&[0xf9, 0xff, 0xff, 0xff][..])
    )]
    #[case::v2_undefined(
        MessageType::FILL_VALUE,
        &[2u8, 2, 2, 0],
        FillTime::IfSet,
        None
    )]
    #[case::v2_zero_length(
        MessageType::FILL_VALUE,
        &[2u8, 2, 2, 1, 0, 0, 0, 0],
        FillTime::IfSet,
        None
    )]
    #[case::v3_default(
        MessageType::FILL_VALUE,
        &[3u8, 0x0a],
        FillTime::IfSet,
        None
    )]
    #[case::v3_undefined(
        MessageType::FILL_VALUE,
        &[3u8, 0x1a],
        FillTime::IfSet,
        None
    )]
    #[case::v3_defined(
        MessageType::FILL_VALUE,
        &[3u8, 0x2a, 4, 0, 0, 0, 5, 6, 7, 8],
        FillTime::IfSet,
        Some(&[5, 6, 7, 8][..])
    )]
    #[case::v3_zero_length(
        MessageType::FILL_VALUE,
        &[3u8, 0x2a, 0, 0, 0, 0],
        FillTime::IfSet,
        None
    )]
    #[case::legacy_defined(
        MessageType::FILL_VALUE_OLD,
        &[4u8, 0, 0, 0, 10, 20, 30, 40],
        FillTime::IfSet,
        Some(&[10, 20, 30, 40][..])
    )]
    #[case::legacy_zero_length(
        MessageType::FILL_VALUE_OLD,
        &[0u8, 0, 0, 0],
        FillTime::IfSet,
        None
    )]
    fn parses_valid_producer_encodings(
        #[case] msg_type: MessageType,
        #[case] bytes: &[u8],
        #[case] expected_time: FillTime,
        #[case] expected_value: Option<&[u8]>,
    ) {
        let view = FillValueView::parse(msg_type, bytes).unwrap();
        assert_eq!(view.fill_time(), expected_time);
        assert_eq!(view.value(), expected_value);
        assert!(fill_value_is_written(msg_type, bytes).unwrap());
        assert_eq!(
            parse_defined_fill_value(msg_type, bytes).unwrap(),
            expected_value.map(Vec::from)
        );
    }

    #[rstest]
    #[case::reserved_bit_6(&[3u8, 0x4a], FormatError::InvalidFillValueFlags(0x4a))]
    #[case::reserved_bit_7(&[3u8, 0x8a], FormatError::InvalidFillValueFlags(0x8a))]
    #[case::both_reserved_bits(&[3u8, 0xca], FormatError::InvalidFillValueFlags(0xca))]
    #[case::conflicting_value_flags(&[3u8, 0x3a], FormatError::InvalidFillValueFlags(0x3a))]
    fn v3_rejects_invalid_flags(#[case] bytes: &[u8], #[case] expected: FormatError) {
        let err = FillValueView::parse(MessageType::FILL_VALUE, bytes).unwrap_err();
        assert_eq!(err, expected);
    }

    #[test]
    fn v3_rejects_invalid_fill_time() {
        let bytes = [3u8, 0x0e];
        let err = FillValueView::parse(MessageType::FILL_VALUE, &bytes).unwrap_err();
        assert_eq!(err, FormatError::InvalidFillTime(3));
    }

    #[rstest]
    #[case::v1_time_four(&[1u8, 4, 2, 0, 0, 0, 0, 0], FormatError::InvalidSpaceAllocationTime(4))]
    #[case::v2_time_five(&[2u8, 5, 2, 0], FormatError::InvalidSpaceAllocationTime(5))]
    fn rejects_invalid_space_allocation_time(#[case] bytes: &[u8], #[case] expected: FormatError) {
        let err = FillValueView::parse(MessageType::FILL_VALUE, bytes).unwrap_err();
        assert_eq!(err, expected);
    }

    #[rstest]
    #[case::v1_time_three(&[1u8, 2, 3, 0, 0, 0, 0, 0], FormatError::InvalidFillTime(3))]
    #[case::v2_time_nine(&[2u8, 2, 9, 0], FormatError::InvalidFillTime(9))]
    fn rejects_invalid_fill_time(#[case] bytes: &[u8], #[case] expected: FormatError) {
        let err = FillValueView::parse(MessageType::FILL_VALUE, bytes).unwrap_err();
        assert_eq!(err, expected);
    }

    #[rstest]
    #[case::v1_indicator_two(&[1u8, 2, 2, 2, 0, 0, 0, 0], FormatError::InvalidFillValueDefined(2))]
    #[case::v2_indicator_maximum(&[2u8, 2, 2, 255], FormatError::InvalidFillValueDefined(255))]
    fn rejects_invalid_defined_indicator(#[case] bytes: &[u8], #[case] expected: FormatError) {
        let err = FillValueView::parse(MessageType::FILL_VALUE, bytes).unwrap_err();
        assert_eq!(err, expected);
    }

    #[test]
    fn parse_rejects_invalid_message_type() {
        let err = FillValueView::parse(MessageType::from_u16(0x0001), &[0u8; 8]).unwrap_err();
        assert_eq!(err, FormatError::InvalidFillValueMessageType(0x0001));
    }

    #[rstest]
    #[case::version_zero(&[0u8, 0], FormatError::UnsupportedFillValueVersion(0))]
    #[case::version_four(&[4u8, 0], FormatError::UnsupportedFillValueVersion(4))]
    #[case::maximum_version(&[255u8, 0], FormatError::UnsupportedFillValueVersion(255))]
    fn rejects_unsupported_version(#[case] bytes: &[u8], #[case] expected: FormatError) {
        let err = FillValueView::parse(MessageType::FILL_VALUE, bytes).unwrap_err();
        assert_eq!(err, expected);
    }

    #[rstest]
    #[case::missing_version(MessageType::FILL_VALUE, &[], FormatError::UnexpectedEof { expected: 1, available: 0 })]
    #[case::v1_prefix(MessageType::FILL_VALUE, &[1u8, 2], FormatError::UnexpectedEof { expected: 4, available: 2 })]
    #[case::v1_length(MessageType::FILL_VALUE, &[1u8, 2, 2, 1, 4, 0], FormatError::UnexpectedEof { expected: 8, available: 6 })]
    #[case::v1_payload(MessageType::FILL_VALUE, &[1u8, 2, 2, 1, 4, 0, 0, 0, 1, 2], FormatError::UnexpectedEof { expected: 12, available: 10 })]
    #[case::v2_prefix(MessageType::FILL_VALUE, &[2u8, 2], FormatError::UnexpectedEof { expected: 4, available: 2 })]
    #[case::v2_length(MessageType::FILL_VALUE, &[2u8, 2, 2, 1, 4, 0], FormatError::UnexpectedEof { expected: 8, available: 6 })]
    #[case::v2_payload(MessageType::FILL_VALUE, &[2u8, 2, 2, 1, 4, 0, 0, 0, 1, 2], FormatError::UnexpectedEof { expected: 12, available: 10 })]
    #[case::v3_flags(MessageType::FILL_VALUE, &[3u8], FormatError::UnexpectedEof { expected: 2, available: 1 })]
    #[case::v3_length(MessageType::FILL_VALUE, &[3u8, 0x2a, 4, 0], FormatError::UnexpectedEof { expected: 6, available: 4 })]
    #[case::v3_payload(MessageType::FILL_VALUE, &[3u8, 0x2a, 4, 0, 0, 0, 1, 2], FormatError::UnexpectedEof { expected: 10, available: 8 })]
    #[case::legacy_missing_length(MessageType::FILL_VALUE_OLD, &[], FormatError::UnexpectedEof { expected: 4, available: 0 })]
    #[case::legacy_partial_length(MessageType::FILL_VALUE_OLD, &[4u8, 0], FormatError::UnexpectedEof { expected: 4, available: 2 })]
    #[case::legacy_payload(MessageType::FILL_VALUE_OLD, &[4u8, 0, 0, 0, 1, 2], FormatError::UnexpectedEof { expected: 8, available: 6 })]
    fn rejects_truncated_bytes(
        #[case] msg_type: MessageType,
        #[case] bytes: &[u8],
        #[case] expected: FormatError,
    ) {
        let err = FillValueView::parse(msg_type, bytes).unwrap_err();
        assert_eq!(err, expected);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn defined_message_length_rejects_values_past_u32() {
        assert_eq!(fill_value_len(u32::MAX as usize).unwrap(), u32::MAX);

        let length = u32::MAX as usize + 1;
        assert_eq!(
            fill_value_len(length).unwrap_err(),
            FillValueError::TooLarge { length }
        );
    }
}
