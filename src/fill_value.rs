//! Fill patterns for unallocated dataset storage.
//!
//! The format crate parses and serializes Fill Value messages. This module
//! applies their bytes to unallocated storage and translates serialization
//! errors for file operations.

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::num::NonZeroUsize;

use hdf5_pure_format::FillValueError;

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

pub use hdf5_pure_format::fill_value_is_written;
pub use hdf5_pure_format::parse_defined_fill_value;

pub(crate) fn fill_value_message_v3(fill: Option<&[u8]>) -> Result<Vec<u8>, FormatError> {
    hdf5_pure_format::fill_value_message_v3(fill).map_err(|error| match error {
        FillValueError::TooLarge { length } => FormatError::SerializationError(format!(
            "fill value length {length} exceeds the u32 message field"
        )),
    })
}

#[cfg(test)]
mod fill_pattern_tests {
    use hdf5_pure_format::V3_FLAGS_DEFAULT;

    use super::*;
    use crate::convert;

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
            PaddingFill::from_message(MessageType::FILL_VALUE, &defined),
            PaddingFill::Value(ref b) if b == &[7u8, 0, 0, 0]
        ));

        // The library default: nothing to write, and that is a decided answer.
        let default = vec![3u8, V3_FLAGS_DEFAULT];
        assert!(matches!(
            PaddingFill::from_message(MessageType::FILL_VALUE, &default),
            PaddingFill::Zero
        ));

        // A version this parser does not know: undetermined, not zero.
        let mut unknown_version = defined.clone();
        unknown_version[0] = 9;
        assert!(matches!(
            PaddingFill::from_message(MessageType::FILL_VALUE, &unknown_version),
            PaddingFill::Unknown
        ));
        assert!(
            PaddingFill::from_message(MessageType::FILL_VALUE, &unknown_version)
                .pattern(elem)
                .apply(&mut [0u8; 4])
                .is_err()
        );

        // Truncated before the fields it declares: likewise undetermined.
        assert!(matches!(
            PaddingFill::from_message(MessageType::FILL_VALUE, &defined[..3]),
            PaddingFill::Unknown
        ));
    }

    /// The pattern tiles one element's bytes across a buffer of whole elements.
    #[test]
    fn a_defined_fill_tiles_across_the_buffer() {
        let seven = 7.0f64.to_le_bytes();
        let p = FillPattern::new(Some(&seven), convert::nz(8));
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
            let p = FillPattern::new(Some(&bytes), convert::nz(8));
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
            FillPattern::new(Some(&[0u8; 8]), convert::nz(8))
                .buffer(16)
                .unwrap(),
            vec![0u8; 16]
        );
        assert_eq!(
            FillPattern::new(None, convert::nz(8)).buffer(16).unwrap(),
            vec![0u8; 16]
        );
        assert_eq!(FillPattern::ZERO.buffer(16).unwrap(), vec![0u8; 16]);
    }

    /// A single-byte element is the degenerate tiling case, and a zero-length
    /// buffer must not panic.
    #[test]
    fn single_byte_elements_and_empty_buffers() {
        let p = FillPattern::new(Some(&[0xAB]), convert::nz(1));
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
        assert!(!fill_value_is_written(MessageType::FILL_VALUE, &v3(0x26)).unwrap());
        assert!(fill_value_is_written(MessageType::FILL_VALUE, &v3(0x2a)).unwrap());
        assert!(fill_value_is_written(MessageType::FILL_VALUE, &v3(0x22)).unwrap());

        // Versions 1 and 2: a byte of its own at index 2.
        for version in [1u8, 2] {
            let msg = |write_time: u8| vec![version, 2, write_time, 1, 4, 0, 0, 0, 7, 0, 0, 0];
            assert!(!fill_value_is_written(MessageType::FILL_VALUE, &msg(1)).unwrap());
            assert!(fill_value_is_written(MessageType::FILL_VALUE, &msg(0)).unwrap());
            assert!(fill_value_is_written(MessageType::FILL_VALUE, &msg(2)).unwrap());
        }

        // The legacy message has no write time to read.
        assert!(
            fill_value_is_written(MessageType::FILL_VALUE_OLD, &[4, 0, 0, 0, 7, 0, 0, 0]).unwrap()
        );

        // Truncated and unrecognized messages error rather than guessing.
        assert!(fill_value_is_written(MessageType::FILL_VALUE, &[]).is_err());
        assert!(fill_value_is_written(MessageType::FILL_VALUE, &[3]).is_err());
        assert!(fill_value_is_written(MessageType::FILL_VALUE, &[2, 0]).is_err());
        assert!(fill_value_is_written(MessageType::FILL_VALUE, &[9, 0]).is_err());
    }
}
