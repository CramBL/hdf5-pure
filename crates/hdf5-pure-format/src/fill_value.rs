use alloc::vec;
use alloc::vec::Vec;

use crate::error::FormatError;
use crate::message_type::MessageType;

/// Version-3 Fill Value message flags for the crate's default (no user-defined
/// value): Late space-allocation time (bits 0-1 = `0b10`) and IfSet fill-value
/// write time (bits 2-3 = `0b10`), with neither the *undefined* (bit 4) nor the
/// *defined* (bit 5) bit set. This is the "library default" fill the crate has
/// always written, matching what the reference C library records when the caller
/// sets no explicit fill value.
pub const V3_FLAGS_DEFAULT: u8 = 0x0a;

/// Errors from serializing a Fill Value message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FillValueError {
    /// A fill value exceeds the message's four-byte length field.
    TooLarge {
        /// Length of the supplied value in bytes.
        length: usize,
    },
}

/// The *Fill Value Defined* bit (bit 5) of the version-3 flags byte. When set,
/// the Size and Fill Value fields follow. Verified against the reference C
/// library, which writes flags `0x2a` (`V3_FLAGS_DEFAULT | V3_FLAG_DEFINED`) for
/// a contiguous dataset with a user-defined fill value.
const V3_FLAG_DEFINED: u8 = 0x20;

/// The *Fill Value Write Time* value meaning **Never** (`H5D_FILL_TIME_NEVER`):
/// the library never writes the fill value into storage, so unallocated storage
/// reads as zeros however the value itself is defined.
///
/// It lives in bits 2-3 of the version-3 flags byte and in byte 2 of the
/// version-1/2 layout. `H5D_FILL_TIME_ALLOC` is 0 and `H5D_FILL_TIME_IFSET`
/// (the default both the high-level crate and the C library write) is 2. Only *Never*
/// changes what a read of unallocated storage sees, so it is the only one named.
const FILL_TIME_NEVER: u8 = 1;

/// Whether the fill value would ever be written into storage, for the read path
/// that has to decide what unallocated storage looks like.
///
/// Returns `false` for a message whose *Fill Value Write Time* is `Never`. Such
/// a dataset's unallocated storage has **no defined contents at all**: the
/// reference C library never writes the fill value into it and has nothing to
/// read back, so `H5Dread` leaves the caller's buffer as it found it. That is
/// observable  -  the same file read through `hdf5-metno` yields zeros on macOS
/// and the fill bytes on Linux, from one uninitialized allocation to the next.
///
/// So this is not a case where the C library can be matched. It is a case where
/// it has no defined result to match. The high-level crate returns zeros because this is
/// deterministic and does not claim a specific value the file says was never
/// written  -  reading back the declared fill would assert data that nothing put
/// there.
///
/// A malformed or unrecognized message produces an error. A legacy
/// `0x0004` message carries no write time and is treated as written, which is
/// the only behavior that format can express.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the message ends before its write-time field, or
/// [`FormatError::UnsupportedFillValueVersion`] if its version is unrecognized.
pub fn fill_value_is_written(msg_type: MessageType, data: &[u8]) -> Result<bool, FormatError> {
    match msg_type {
        MessageType::FILL_VALUE_OLD => Ok(true),
        MessageType::FILL_VALUE => {
            let version = *data.first().ok_or(eof(1, data.len()))?;
            match version {
                // Versions 1 and 2 give the write time a byte of its own.
                VERSION_ONE | VERSION_TWO => {
                    Ok(*data.get(2).ok_or(eof(3, data.len()))? != FILL_TIME_NEVER)
                }
                // Version 3 packs it into bits 2-3 of the flags byte.
                VERSION_THREE => {
                    let flags = *data.get(1).ok_or(eof(2, data.len()))?;
                    Ok((flags >> 2) & 0b11 != FILL_TIME_NEVER)
                }
                other => Err(FormatError::UnsupportedFillValueVersion(other)),
            }
        }
        _ => Ok(true),
    }
}

/// Serialize the body of a version-3 Fill Value message ([`MessageType::FillValue`]).
///
/// With `fill = None` this is the library-default message the crate emits for a
/// dataset whose fill value was never set (`[version=3, flags=0x0a]`). With
/// `fill = Some(bytes)` the *Fill Value Defined* bit is set and the value  -
/// `bytes`, already encoded in the dataset's datatype  -  is appended after its
/// 4-byte length, exactly as the reference C library records a user-defined fill.
///
/// # Errors
///
/// Returns [`FillValueError::TooLarge`] if the fill value length exceeds the message's
/// four-byte length field.
pub fn fill_value_message_v3(fill: Option<&[u8]>) -> Result<Vec<u8>, FillValueError> {
    Ok(match fill {
        None => vec![VERSION_THREE, V3_FLAGS_DEFAULT],
        Some(bytes) => {
            let len = fill_value_len(bytes.len())?;
            let mut msg = Vec::with_capacity(V3_VALUE_OFFSET + bytes.len());
            msg.push(VERSION_THREE);
            msg.push(V3_FLAGS_DEFAULT | V3_FLAG_DEFINED);
            msg.extend_from_slice(&len.to_le_bytes());
            msg.extend_from_slice(bytes);
            msg
        }
    })
}

fn fill_value_len(len: usize) -> Result<u32, FillValueError> {
    match u32::try_from(len) {
        Ok(length) => Ok(length),
        Err(_out_of_range) => Err(FillValueError::TooLarge { length: len }),
    }
}

/// Read a *defined* fill value's raw bytes from a Fill Value message body.
///
/// `msg_type` selects the format: [`MessageType::FillValue`] (`0x0005`, versions
/// 1/2/3) or [`MessageType::FillValueOld`] (`0x0004`, the pre-versioning
/// format). `data` is the message body (everything after the object-header
/// message header).
///
/// Returns `Ok(Some(bytes))` when the message carries a user-defined fill value,
/// where `bytes` is the value encoded in the dataset's datatype. Returns
/// `Ok(None)` when no user-defined value is present  -  the library default, an
/// explicitly undefined fill, or a defined-but-empty (zero-length) value, all of
/// which mean "read unset regions as the type's implicit zero".
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if the body ends before a declared field, or
/// [`FormatError::UnsupportedFillValueVersion`] if the `0x0005` message has an unrecognized
/// version.
pub fn parse_defined_fill_value(
    msg_type: MessageType,
    data: &[u8],
) -> Result<Option<Vec<u8>>, FormatError> {
    match msg_type {
        // Legacy format: a 4-byte size followed by that many value bytes, with no
        // version or "defined" flag. A zero size means no fill value.
        MessageType::FILL_VALUE_OLD => {
            let size = read_u32(data, 0)? as usize;
            if size == 0 {
                return Ok(None);
            }
            Ok(Some(read_bytes(data, 4, size)?))
        }
        MessageType::FILL_VALUE => {
            let version = *data.first().ok_or(eof(1, data.len()))?;
            match version {
                // Versions 1 and 2 share a 4-byte prefix (version, space
                // allocation time, fill value write time, fill value defined).
                // In version 1 the Size and Fill Value fields are always present.
                // in version 2 they are present only when the "defined" byte is
                // nonzero.
                VERSION_ONE | VERSION_TWO => {
                    let defined = *data.get(3).ok_or(eof(4, data.len()))?;
                    if version == VERSION_TWO && defined == 0 {
                        return Ok(None);
                    }
                    let size = read_u32(data, 4)? as usize;
                    if size == 0 {
                        return Ok(None);
                    }
                    // A version-1 message can carry the fields while marking the
                    // value undefined. Honor the flag over the stored bytes.
                    if version == VERSION_ONE && defined == 0 {
                        return Ok(None);
                    }
                    Ok(Some(read_bytes(data, V1_V2_VALUE_OFFSET, size)?))
                }
                // Version 3 replaces the three separate time/defined bytes with a
                // single flags byte. The Size and Fill Value fields follow only
                // when the "defined" bit (bit 5) is set.
                VERSION_THREE => {
                    let flags = *data.get(1).ok_or(eof(2, data.len()))?;
                    if flags & V3_FLAG_DEFINED == 0 {
                        return Ok(None);
                    }
                    let size = read_u32(data, 2)? as usize;
                    if size == 0 {
                        return Ok(None);
                    }
                    Ok(Some(read_bytes(data, V3_VALUE_OFFSET, size)?))
                }
                other => Err(FormatError::UnsupportedFillValueVersion(other)),
            }
        }
        // Not a fill value message. Nothing to extract.
        _ => Ok(None),
    }
}

/// A little-endian `u32` at `offset`, or an EOF error if the four bytes are not
/// all present.
fn read_u32(data: &[u8], offset: usize) -> Result<u32, FormatError> {
    let end = offset + 4;
    let slice = data.get(offset..end).ok_or(eof(end, data.len()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// `len` bytes at `offset`, copied out, or an EOF error if the range runs past
/// the end of `data`.
fn read_bytes(data: &[u8], offset: usize, len: usize) -> Result<Vec<u8>, FormatError> {
    let end = offset.checked_add(len).ok_or(eof(usize::MAX, data.len()))?;
    let slice = data.get(offset..end).ok_or(eof(end, data.len()))?;
    Ok(slice.to_vec())
}

/// Build the `UnexpectedEof` error for a field that needed `expected` bytes but
/// only `available` were present.
fn eof(expected: usize, available: usize) -> FormatError {
    FormatError::UnexpectedEof {
        expected,
        available,
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_fvmessage`, version 4.0, defines the fill-value versions and value offsets.
const VERSION_ONE: u8 = 1;
const VERSION_TWO: u8 = 2;
const VERSION_THREE: u8 = 3;
const V1_V2_VALUE_OFFSET: usize = 8;
const V3_VALUE_OFFSET: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_message_is_unchanged() {
        // The historical default the crate has always written.
        assert_eq!(fill_value_message_v3(None).unwrap(), vec![3, 0x0a]);
    }

    #[test]
    fn defined_message_matches_reference_library_bytes() {
        // The reference C library writes exactly these bytes for a contiguous
        // i32 dataset with fill value -7 (confirmed by dumping a probe file):
        // version 3, flags 0x2a, size 4, value 0xFFFFFFF9.
        let msg = fill_value_message_v3(Some(&(-7i32).to_le_bytes())).unwrap();
        assert_eq!(msg, vec![3, 0x2a, 4, 0, 0, 0, 0xf9, 0xff, 0xff, 0xff]);
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

    #[test]
    fn v3_defined_round_trips_through_the_parser() {
        let value = 3.5f64.to_le_bytes();
        let msg = fill_value_message_v3(Some(&value)).unwrap();
        let got = parse_defined_fill_value(MessageType::FILL_VALUE, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&value[..]));
    }

    #[test]
    fn v3_default_parses_as_no_value() {
        let msg = fill_value_message_v3(None).unwrap();
        assert_eq!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn v3_explicitly_undefined_parses_as_no_value() {
        // Flags 0x1a: the "undefined" bit (bit 4) set, "defined" bit clear.
        let msg = [3u8, 0x1a];
        assert_eq!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn v2_defined_parses_the_value() {
        // version 2, alloc=2, write=2, defined=1, size=4, value=-7. These are the
        // exact bytes the reference C library writes by default (v2 message).
        let msg = [2u8, 2, 2, 1, 4, 0, 0, 0, 0xf9, 0xff, 0xff, 0xff];
        let got = parse_defined_fill_value(MessageType::FILL_VALUE, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&[0xf9, 0xff, 0xff, 0xff][..]));
    }

    #[test]
    fn v2_undefined_parses_as_no_value() {
        // defined byte = 0: the Size and Fill Value fields are absent.
        let msg = [2u8, 2, 2, 0];
        assert_eq!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn v1_defined_parses_the_value() {
        // version 1 always carries Size and Fill Value. Defined=1.
        let msg = [1u8, 2, 2, 1, 2, 0, 0, 0, 0xed, 0xfe];
        let got = parse_defined_fill_value(MessageType::FILL_VALUE, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&[0xed, 0xfe][..]));
    }

    #[test]
    fn v1_marked_undefined_ignores_stored_bytes() {
        // version 1 carries the fields but marks the value undefined (defined=0).
        let msg = [1u8, 2, 2, 0, 4, 0, 0, 0, 1, 2, 3, 4];
        assert_eq!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn old_message_parses_the_value() {
        // Legacy 0x0004 message: size 4, value bytes.
        let msg = [4u8, 0, 0, 0, 10, 20, 30, 40];
        let got = parse_defined_fill_value(MessageType::FILL_VALUE_OLD, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&[10, 20, 30, 40][..]));
    }

    #[test]
    fn old_message_zero_size_is_no_value() {
        let msg = [0u8, 0, 0, 0];
        assert_eq!(
            parse_defined_fill_value(MessageType::FILL_VALUE_OLD, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn truncated_size_field_errors() {
        // Defined v3 flags but the size field is cut short.
        let msg = [3u8, 0x2a, 4, 0];
        assert!(matches!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &msg),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn truncated_value_field_errors() {
        // Declares 8 value bytes but supplies only 2.
        let msg = [3u8, 0x2a, 8, 0, 0, 0, 0xaa, 0xbb];
        assert!(matches!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &msg),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn unknown_version_errors() {
        let msg = [9u8, 0];
        assert!(matches!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &msg),
            Err(FormatError::UnsupportedFillValueVersion(9))
        ));
    }

    #[test]
    fn empty_body_errors() {
        assert!(matches!(
            parse_defined_fill_value(MessageType::FILL_VALUE, &[]),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }
}
