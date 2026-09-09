//! Serialization and parsing for the dataset Fill Value message.
//!
//! HDF5 records a dataset's fill value in a *Fill Value* message. Two message
//! types exist: the current one (`0x0005`, [`MessageType::FillValue`]) with
//! on-disk versions 1, 2, and 3, and a legacy "old" one (`0x0004`,
//! [`MessageType::FillValueOld`]) that predates versioning. This module owns the
//! byte-level format for both directions:
//!
//! * [`fill_value_message_v3`] writes the version-3 message body the crate emits
//!   for every dataset — either the library-default fill (no user value) or, when
//!   the builder set one, a user-defined value with the *Fill Value Defined* bit.
//! * [`parse_defined_fill_value`] reads a user-defined value back out of any of
//!   the message variants, so a dataset's fill value round-trips and fill values
//!   in files written by the reference C library or h5py can be inspected.
//!
//! The value bytes are stored in the dataset's datatype (its size and byte
//! order); this module treats them opaquely and leaves interpretation to the
//! caller, which decodes them through the same typed-conversion path as a normal
//! read.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::num::NonZeroUsize;

use crate::error::FormatError;
use crate::message_type::MessageType;

/// What storage a dataset has never had written to it reads as.
///
/// HDF5 allocates a dataset's storage lazily, so a dataset can be *created* with
/// a shape and then read before anything is written — as a whole, when no chunk
/// index or contiguous block exists at all, or in parts, when some chunks of a
/// chunked dataset exist and others do not. The reference C library answers
/// those regions with the dataset's fill value, and this is the pattern that
/// answers them here.
///
/// `None` means the library default, which is the type's implicit zero — the
/// same thing [`parse_defined_fill_value`] returns `Ok(None)` for. Keeping that
/// case as `None` rather than as a buffer of zero bytes is what lets the common
/// path stay a plain zeroed allocation with nothing to tile.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FillPattern<'a> {
    /// Exactly one element's worth of fill bytes, or `None` for zeros.
    element: Option<&'a [u8]>,
    /// The Fill Value message could not be parsed, so what unallocated storage
    /// reads as is unknown. Distinct from `element: None`, which is a *known*
    /// zero: see [`FillPattern::UNKNOWN`].
    unknown: bool,
}

impl<'a> FillPattern<'a> {
    /// The implicit zero: what a dataset with no user-defined fill value reads
    /// as, and the conservative answer whenever a fill value cannot be used.
    pub(crate) const ZERO: Self = Self {
        element: None,
        unknown: false,
    };

    /// The dataset's Fill Value message could not be parsed, so what its
    /// unallocated storage reads as is unknown.
    ///
    /// This is deliberately *not* the zero pattern. Reading unallocated storage
    /// through it fails, because zeros would be a fabricated answer — but a
    /// dataset whose storage is fully allocated never consults the pattern, so
    /// it keeps reading exactly as it did before the fill path existed. That
    /// matters for forward compatibility: a future Fill Value message version
    /// this parser does not know would otherwise make every dataset in the file
    /// unreadable, including ones whose bytes are all present.
    ///
    /// [`crate::Dataset::fill_value`] still surfaces the underlying parse error
    /// to a caller who asks about the value itself.
    pub(crate) const UNKNOWN: Self = Self {
        element: None,
        unknown: true,
    };

    /// The pattern for a dataset whose Fill Value message carried `bytes`, over
    /// a datatype of `elem_size` bytes per element.
    ///
    /// A fill value is one element wide by definition — it is stored in the
    /// dataset's own datatype — so a message declaring some other length is
    /// malformed. Rather than tile it and shift every element after the first,
    /// such a value is dropped and the region reads as zeros: the fill pattern
    /// is consulted only where there is no data, so falling back there cannot
    /// corrupt a value that was actually written, and it leaves a dataset whose
    /// storage *is* allocated reading exactly as it did before.
    pub(crate) fn new(bytes: Option<&'a [u8]>, elem_size: NonZeroUsize) -> Self {
        match bytes {
            Some(b) if b.len() == elem_size.get() && b.iter().any(|&x| x != 0) => Self {
                element: Some(b),
                unknown: false,
            },
            // An all-zero fill value is the zero pattern; saying so here keeps
            // every buffer that would tile zeros on the plain allocation path.
            _ => Self::ZERO,
        }
    }

    /// One element's fill bytes, or `None` for the implicit zero — for a caller
    /// that must *record* the fill value rather than tile it, which is what the
    /// scale-offset filter's parameters do.
    ///
    /// # Errors
    ///
    /// [`FormatError::UnreadableFillValue`] for the unknown pattern, for the
    /// same reason [`apply`](Self::apply) refuses it: a value that could not be
    /// read cannot be recorded as zero, since a decoder would then treat real
    /// zeros as fill values.
    pub(crate) fn element(self) -> Result<Option<&'a [u8]>, FormatError> {
        if self.unknown {
            return Err(FormatError::UnreadableFillValue);
        }
        Ok(self.element)
    }

    /// A `len`-byte buffer holding the pattern, repeated from the start.
    ///
    /// `len` is a whole number of elements at every call site; a trailing
    /// partial element would simply be filled with the pattern's prefix rather
    /// than misaligned or panicking.
    pub(crate) fn buffer(self, len: usize) -> Result<Vec<u8>, FormatError> {
        if self.unknown {
            return Err(FormatError::UnreadableFillValue);
        }
        let mut buf = vec![0u8; len];
        self.apply(&mut buf)?;
        Ok(buf)
    }

    /// Tile the pattern across `buf`, which must begin on an element boundary.
    /// A no-op for the zero pattern, leaving an already-zeroed buffer untouched.
    ///
    /// # Errors
    ///
    /// [`FormatError::UnreadableFillValue`] for the unknown pattern. Zeros
    /// would be a fabricated answer, and unlike a read of unallocated storage —
    /// which at least fabricates it in memory — this one would write it to the
    /// file. Callers apply the pattern only where the fill value is actually
    /// needed, so a dataset whose Fill Value message this parser cannot read is
    /// still writable wherever the answer does not arise.
    pub(crate) fn apply(self, buf: &mut [u8]) -> Result<(), FormatError> {
        if self.unknown {
            return Err(FormatError::UnreadableFillValue);
        }
        let Some(element) = self.element else {
            return Ok(());
        };
        // `chunks_mut` bounds each slot to what remains, so a trailing partial
        // element takes the pattern's prefix rather than needing a guard.
        for slot in buf.chunks_mut(element.len()) {
            slot.copy_from_slice(&element[..slot.len()]);
        }
        Ok(())
    }
}

/// What the unwritten slots of an **allocated** chunk must hold, recovered from
/// a dataset's Fill Value message and owning the bytes so a caller can hand out
/// a borrowed [`FillPattern`].
///
/// The three cases are the ones the read path already distinguishes, and for
/// the same reason: "write zeros" and "we do not know what to write" are
/// different answers, and only one of them is safe to act on.
pub(crate) enum PaddingFill {
    /// Nothing is written to unwritten storage — no fill value, or a message
    /// recording `H5D_FILL_TIME_NEVER`, which promises nothing about it.
    Zero,
    /// One element's fill bytes, to be tiled across the uncovered slots.
    Value(Vec<u8>),
    /// The message could not be read, so what those slots must hold is
    /// undetermined. Writing zeros would put a guess on disk.
    Unknown,
}

impl PaddingFill {
    /// Decide from a Fill Value message body. `msg_type` distinguishes the
    /// versioned message (`FillValue`) from the legacy one (`FillValueOld`).
    ///
    /// The same two steps, in the same order, as [`crate::Dataset::fill_value`]
    /// takes for reads — with the write-time question asked first, since a
    /// value the file says is never written is not written here either.
    pub(crate) fn from_message(msg_type: MessageType, body: &[u8]) -> Self {
        match fill_value_is_written(msg_type, body) {
            Ok(false) => Self::Zero,
            Ok(true) => match parse_defined_fill_value(msg_type, body) {
                Ok(Some(bytes)) => Self::Value(bytes),
                Ok(None) => Self::Zero,
                Err(_) => Self::Unknown,
            },
            Err(_) => Self::Unknown,
        }
    }

    /// The pattern to tile, over a datatype of `elem_size` bytes per element.
    pub(crate) fn pattern(&self, elem_size: NonZeroUsize) -> FillPattern<'_> {
        match self {
            Self::Zero => FillPattern::ZERO,
            Self::Value(bytes) => FillPattern::new(Some(bytes), elem_size),
            Self::Unknown => FillPattern::UNKNOWN,
        }
    }
}

/// Version-3 Fill Value message flags for the crate's default (no user-defined
/// value): Late space-allocation time (bits 0-1 = `0b10`) and IfSet fill-value
/// write time (bits 2-3 = `0b10`), with neither the *undefined* (bit 4) nor the
/// *defined* (bit 5) bit set. This is the "library default" fill the crate has
/// always written, matching what the reference C library records when the caller
/// sets no explicit fill value.
const V3_FLAGS_DEFAULT: u8 = 0x0a;

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
/// (the default both this crate and the C library write) is 2; only *Never*
/// changes what a read of unallocated storage sees, so it is the only one named.
const FILL_TIME_NEVER: u8 = 1;

/// Whether the fill value would ever be written into storage, for the read path
/// that has to decide what unallocated storage looks like.
///
/// Returns `false` for a message whose *Fill Value Write Time* is `Never`. Such
/// a dataset's unallocated storage has **no defined contents at all**: the
/// reference C library never writes the fill value into it and has nothing to
/// read back, so `H5Dread` leaves the caller's buffer as it found it. That is
/// observable — the same file read through `hdf5-metno` yields zeros on macOS
/// and the fill bytes on Linux, from one uninitialized allocation to the next.
///
/// So this is not a case where the C library can be matched; it is a case where
/// it has no answer to match. Zeros is the answer chosen here because it is
/// deterministic and does not claim a specific value the file says was never
/// written — reading back the declared fill would assert data that nothing put
/// there.
///
/// A malformed or unrecognized message errors rather than guessing; a legacy
/// `0x0004` message carries no write time and is treated as written, which is
/// the only behavior that format can express.
pub(crate) fn fill_value_is_written(
    msg_type: MessageType,
    data: &[u8],
) -> Result<bool, FormatError> {
    match msg_type {
        MessageType::FillValueOld => Ok(true),
        MessageType::FillValue => {
            let version = *data.first().ok_or(eof(1, data.len()))?;
            match version {
                // Versions 1 and 2 give the write time a byte of its own.
                1 | 2 => Ok(*data.get(2).ok_or(eof(3, data.len()))? != FILL_TIME_NEVER),
                // Version 3 packs it into bits 2-3 of the flags byte.
                3 => {
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
/// `fill = Some(bytes)` the *Fill Value Defined* bit is set and the value —
/// `bytes`, already encoded in the dataset's datatype — is appended after its
/// 4-byte length, exactly as the reference C library records a user-defined fill.
///
/// A fill value is a single scalar element (at most a handful of bytes), so the
/// `u32` length field cannot overflow.
pub(crate) fn fill_value_message_v3(fill: Option<&[u8]>) -> Vec<u8> {
    match fill {
        None => vec![3, V3_FLAGS_DEFAULT],
        Some(bytes) => {
            let mut msg = Vec::with_capacity(6 + bytes.len());
            msg.push(3); // version
            msg.push(V3_FLAGS_DEFAULT | V3_FLAG_DEFINED);
            // A scalar fill value is only a few bytes wide, so its length always
            // fits `u32` (see the module contract).
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a scalar fill value is at most a few bytes; its length fits u32"
            )]
            let len = bytes.len() as u32;
            msg.extend_from_slice(&len.to_le_bytes());
            msg.extend_from_slice(bytes);
            msg
        }
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
/// `Ok(None)` when no user-defined value is present — the library default, an
/// explicitly undefined fill, or a defined-but-empty (zero-length) value, all of
/// which mean "read unset regions as the type's implicit zero".
///
/// # Errors
///
/// [`FormatError::UnexpectedEof`] if the body is truncated before a field it
/// declares (a short header, or a length that runs past the available bytes), and
/// [`FormatError::UnsupportedFillValueVersion`] for an unrecognized version of
/// the `0x0005` message.
pub(crate) fn parse_defined_fill_value(
    msg_type: MessageType,
    data: &[u8],
) -> Result<Option<Vec<u8>>, FormatError> {
    match msg_type {
        // Legacy format: a 4-byte size followed by that many value bytes, with no
        // version or "defined" flag. A zero size means no fill value.
        MessageType::FillValueOld => {
            let size = read_u32(data, 0)? as usize;
            if size == 0 {
                return Ok(None);
            }
            Ok(Some(read_bytes(data, 4, size)?))
        }
        MessageType::FillValue => {
            let version = *data.first().ok_or(eof(1, data.len()))?;
            match version {
                // Versions 1 and 2 share a 4-byte prefix (version, space
                // allocation time, fill value write time, fill value defined).
                // In version 1 the Size and Fill Value fields are always present;
                // in version 2 they are present only when the "defined" byte is
                // nonzero.
                1 | 2 => {
                    let defined = *data.get(3).ok_or(eof(4, data.len()))?;
                    if version == 2 && defined == 0 {
                        return Ok(None);
                    }
                    let size = read_u32(data, 4)? as usize;
                    if size == 0 {
                        return Ok(None);
                    }
                    // A version-1 message can carry the fields while marking the
                    // value undefined; honor the flag over the stored bytes.
                    if version == 1 && defined == 0 {
                        return Ok(None);
                    }
                    Ok(Some(read_bytes(data, 8, size)?))
                }
                // Version 3 replaces the three separate time/defined bytes with a
                // single flags byte; the Size and Fill Value fields follow only
                // when the "defined" bit (bit 5) is set.
                3 => {
                    let flags = *data.get(1).ok_or(eof(2, data.len()))?;
                    if flags & V3_FLAG_DEFINED == 0 {
                        return Ok(None);
                    }
                    let size = read_u32(data, 2)? as usize;
                    if size == 0 {
                        return Ok(None);
                    }
                    Ok(Some(read_bytes(data, 6, size)?))
                }
                other => Err(FormatError::UnsupportedFillValueVersion(other)),
            }
        }
        // Not a fill value message; nothing to extract.
        _ => Ok(None),
    }
}

#[cfg(test)]
mod fill_pattern_tests {

    /// `PaddingFill::from_message` keeps the three answers apart. Folding the
    /// unreadable case onto `Zero` would write a fabricated fill value into a
    /// file whose fill semantics this parser has already refused to report.
    #[test]
    fn an_unreadable_message_is_unknown_not_zero() {
        let elem = NonZeroUsize::new(4).unwrap();
        // A v3 message with the *defined* bit set and a 4-byte value.
        let mut defined = vec![3u8, V3_FLAGS_DEFAULT | 0x20, 4, 0, 0, 0];
        defined.extend_from_slice(&[7u8, 0, 0, 0]);
        assert!(matches!(
            PaddingFill::from_message(MessageType::FillValue, &defined),
            PaddingFill::Value(ref b) if b == &[7u8, 0, 0, 0]
        ));

        // The library default: nothing to write, and that is a decided answer.
        let default = vec![3u8, V3_FLAGS_DEFAULT];
        assert!(matches!(
            PaddingFill::from_message(MessageType::FillValue, &default),
            PaddingFill::Zero
        ));

        // A version this parser does not know: undetermined, not zero.
        let mut unknown_version = defined.clone();
        unknown_version[0] = 9;
        assert!(matches!(
            PaddingFill::from_message(MessageType::FillValue, &unknown_version),
            PaddingFill::Unknown
        ));
        assert!(
            PaddingFill::from_message(MessageType::FillValue, &unknown_version)
                .pattern(elem)
                .apply(&mut [0u8; 4])
                .is_err()
        );

        // Truncated before the fields it declares: likewise undetermined.
        assert!(matches!(
            PaddingFill::from_message(MessageType::FillValue, &defined[..3]),
            PaddingFill::Unknown
        ));
    }
    use super::*;
    use crate::convert::nz;

    /// The pattern tiles one element's bytes across a buffer of whole elements.
    #[test]
    fn a_defined_fill_tiles_across_the_buffer() {
        let seven = 7.0f64.to_le_bytes();
        let p = FillPattern::new(Some(&seven), nz(8));
        assert_eq!(p.buffer(24).unwrap(), seven.repeat(3));
        // A pattern with interior zero bytes must not be mistaken for the zero
        // pattern: only an *all*-zero value is.
        assert!(seven.iter().filter(|&&b| b == 0).count() >= 6);
    }

    /// A fill value whose length is not the element size is malformed. It is
    /// dropped rather than tiled, because tiling it would shift every element
    /// after the first — a wrong answer instead of a missing one.
    #[test]
    fn a_fill_value_of_the_wrong_width_is_dropped() {
        for bytes in [vec![1u8], vec![1, 2, 3], vec![1; 9], Vec::new()] {
            let p = FillPattern::new(Some(&bytes), nz(8));
            assert_eq!(
                p.buffer(16).unwrap(),
                vec![0u8; 16],
                "a {}-byte fill over an 8-byte element must not tile",
                bytes.len()
            );
        }
    }

    /// An all-zero fill value *is* the zero pattern, and says so, so every
    /// buffer that would tile zeros stays on the plain allocation path.
    #[test]
    fn an_all_zero_fill_is_the_zero_pattern() {
        assert_eq!(
            FillPattern::new(Some(&[0u8; 8]), nz(8)).buffer(16).unwrap(),
            vec![0u8; 16]
        );
        assert_eq!(
            FillPattern::new(None, nz(8)).buffer(16).unwrap(),
            vec![0u8; 16]
        );
        assert_eq!(FillPattern::ZERO.buffer(16).unwrap(), vec![0u8; 16]);
    }

    /// A single-byte element is the degenerate tiling case, and a zero-length
    /// buffer must not panic.
    #[test]
    fn single_byte_elements_and_empty_buffers() {
        let p = FillPattern::new(Some(&[0xAB]), nz(1));
        assert_eq!(p.buffer(5).unwrap(), vec![0xAB; 5]);
        assert_eq!(p.buffer(0).unwrap(), Vec::<u8>::new());
    }

    /// An unreadable fill value message is not the zero pattern: materializing
    /// it fails rather than fabricating zeros. A dataset that never needs the
    /// pattern — because all of its storage is allocated — never calls this, and
    /// so reads normally.
    #[test]
    fn an_unknown_fill_refuses_to_materialize() {
        assert!(matches!(
            FillPattern::UNKNOWN.buffer(16),
            Err(FormatError::UnreadableFillValue)
        ));
        // Length zero is still a refusal: the question is whether the value is
        // known, not how much of it was asked for.
        assert!(FillPattern::UNKNOWN.buffer(0).is_err());
        assert!(FillPattern::ZERO.buffer(0).is_ok());
    }

    /// The Fill Value Write Time decides whether unallocated storage sees the
    /// value at all, and it is read from a different field in each version.
    #[test]
    fn the_write_time_is_read_from_every_message_version() {
        // Version 3: bits 2-3 of the flags byte. 0x26 is Late alloc, Never, and Defined,
        // and 0x2a the same with IfSet. The C library writes Defined for
        // `H5D_FILL_TIME_NEVER` whenever a fill value is set (`H5Ofill.c`, HDF5 2.1.0).
        let v3 = |flags: u8| vec![3u8, flags, 4, 0, 0, 0, 7, 0, 0, 0];
        assert!(!fill_value_is_written(MessageType::FillValue, &v3(0x26)).unwrap());
        assert!(fill_value_is_written(MessageType::FillValue, &v3(0x2a)).unwrap());
        assert!(fill_value_is_written(MessageType::FillValue, &v3(0x22)).unwrap());

        // Versions 1 and 2: a byte of its own at index 2.
        for version in [1u8, 2] {
            let msg = |write_time: u8| vec![version, 2, write_time, 1, 4, 0, 0, 0, 7, 0, 0, 0];
            assert!(!fill_value_is_written(MessageType::FillValue, &msg(1)).unwrap());
            assert!(fill_value_is_written(MessageType::FillValue, &msg(0)).unwrap());
            assert!(fill_value_is_written(MessageType::FillValue, &msg(2)).unwrap());
        }

        // The legacy message has no write time to read.
        assert!(
            fill_value_is_written(MessageType::FillValueOld, &[4, 0, 0, 0, 7, 0, 0, 0]).unwrap()
        );

        // Truncated and unrecognized messages error rather than guessing.
        assert!(fill_value_is_written(MessageType::FillValue, &[]).is_err());
        assert!(fill_value_is_written(MessageType::FillValue, &[3]).is_err());
        assert!(fill_value_is_written(MessageType::FillValue, &[2, 0]).is_err());
        assert!(fill_value_is_written(MessageType::FillValue, &[9, 0]).is_err());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_message_is_unchanged() {
        // The historical default the crate has always written.
        assert_eq!(fill_value_message_v3(None), vec![3, 0x0a]);
    }

    #[test]
    fn defined_message_matches_reference_library_bytes() {
        // The reference C library writes exactly these bytes for a contiguous
        // i32 dataset with fill value -7 (confirmed by dumping a probe file):
        // version 3, flags 0x2a, size 4, value 0xFFFFFFF9.
        let msg = fill_value_message_v3(Some(&(-7i32).to_le_bytes()));
        assert_eq!(msg, vec![3, 0x2a, 4, 0, 0, 0, 0xf9, 0xff, 0xff, 0xff]);
    }

    #[test]
    fn v3_defined_round_trips_through_the_parser() {
        let value = 3.5f64.to_le_bytes();
        let msg = fill_value_message_v3(Some(&value));
        let got = parse_defined_fill_value(MessageType::FillValue, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&value[..]));
    }

    #[test]
    fn v3_default_parses_as_no_value() {
        let msg = fill_value_message_v3(None);
        assert_eq!(
            parse_defined_fill_value(MessageType::FillValue, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn v3_explicitly_undefined_parses_as_no_value() {
        // Flags 0x1a: the "undefined" bit (bit 4) set, "defined" bit clear.
        let msg = [3u8, 0x1a];
        assert_eq!(
            parse_defined_fill_value(MessageType::FillValue, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn v2_defined_parses_the_value() {
        // version 2, alloc=2, write=2, defined=1, size=4, value=-7. These are the
        // exact bytes the reference C library writes by default (v2 message).
        let msg = [2u8, 2, 2, 1, 4, 0, 0, 0, 0xf9, 0xff, 0xff, 0xff];
        let got = parse_defined_fill_value(MessageType::FillValue, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&[0xf9, 0xff, 0xff, 0xff][..]));
    }

    #[test]
    fn v2_undefined_parses_as_no_value() {
        // defined byte = 0: the Size and Fill Value fields are absent.
        let msg = [2u8, 2, 2, 0];
        assert_eq!(
            parse_defined_fill_value(MessageType::FillValue, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn v1_defined_parses_the_value() {
        // version 1 always carries Size and Fill Value; defined=1.
        let msg = [1u8, 2, 2, 1, 2, 0, 0, 0, 0xed, 0xfe];
        let got = parse_defined_fill_value(MessageType::FillValue, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&[0xed, 0xfe][..]));
    }

    #[test]
    fn v1_marked_undefined_ignores_stored_bytes() {
        // version 1 carries the fields but marks the value undefined (defined=0).
        let msg = [1u8, 2, 2, 0, 4, 0, 0, 0, 1, 2, 3, 4];
        assert_eq!(
            parse_defined_fill_value(MessageType::FillValue, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn old_message_parses_the_value() {
        // Legacy 0x0004 message: size 4, value bytes.
        let msg = [4u8, 0, 0, 0, 10, 20, 30, 40];
        let got = parse_defined_fill_value(MessageType::FillValueOld, &msg).unwrap();
        assert_eq!(got.as_deref(), Some(&[10, 20, 30, 40][..]));
    }

    #[test]
    fn old_message_zero_size_is_no_value() {
        let msg = [0u8, 0, 0, 0];
        assert_eq!(
            parse_defined_fill_value(MessageType::FillValueOld, &msg).unwrap(),
            None
        );
    }

    #[test]
    fn truncated_size_field_errors() {
        // Defined v3 flags but the size field is cut short.
        let msg = [3u8, 0x2a, 4, 0];
        assert!(matches!(
            parse_defined_fill_value(MessageType::FillValue, &msg),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn truncated_value_field_errors() {
        // Declares 8 value bytes but supplies only 2.
        let msg = [3u8, 0x2a, 8, 0, 0, 0, 0xaa, 0xbb];
        assert!(matches!(
            parse_defined_fill_value(MessageType::FillValue, &msg),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn unknown_version_errors() {
        let msg = [9u8, 0];
        assert!(matches!(
            parse_defined_fill_value(MessageType::FillValue, &msg),
            Err(FormatError::UnsupportedFillValueVersion(9))
        ));
    }

    #[test]
    fn empty_body_errors() {
        assert!(matches!(
            parse_defined_fill_value(MessageType::FillValue, &[]),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }
}
