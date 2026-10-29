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

use hdf5_pure_core::FillTime;
use hdf5_pure_format::__private::FillValueError;
pub(crate) use hdf5_pure_format::__private::FillValueMessage;
pub(crate) use hdf5_pure_format::__private::FillValueView;

use crate::error::FormatError;
use crate::message_type::MessageType;

/// What storage a dataset has never had written to it reads as.
///
/// HDF5 allocates a dataset's storage lazily, so a dataset can be *created* with
/// a shape and then read before anything is written — as a whole, when no chunk
/// index or contiguous block exists at all, or in parts, when some chunks of a
/// chunked dataset exist and others do not. libhdf5 fills
/// those regions with the dataset's fill value. This implementation
/// uses the same fill behavior.
///
/// `None` means the library default, which is the type's implicit zero. Keeping
/// that case as `None` lets the common path stay a plain zeroed allocation with
/// nothing to tile.
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
    pub(crate) fn new(
        fill_time: FillTime,
        bytes: Option<&'a [u8]>,
        elem_size: NonZeroUsize,
    ) -> Self {
        if fill_time.is_never() {
            return Self::ZERO;
        }
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

    /// Creates a fill pattern from a Fill Value message.
    pub(crate) fn from_message(msg: &'a FillValueMessage, elem_size: NonZeroUsize) -> Self {
        Self::new(msg.fill_time(), msg.value(), elem_size)
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
        match FillValueView::parse(msg_type, body) {
            Ok(view) => Self::from_view(&view),
            Err(_) => Self::Unknown,
        }
    }

    /// Inspects a borrowed fill value view to choose the padding fill behavior.
    pub(crate) fn from_view(view: &FillValueView<'_>) -> Self {
        if view.fill_time().is_never() {
            Self::Zero
        } else {
            match view.value() {
                Some(bytes) => Self::Value(bytes.to_vec()),
                None => Self::Zero,
            }
        }
    }

    /// The pattern to tile, over a datatype of `elem_size` bytes per element.
    pub(crate) fn pattern(&self, elem_size: NonZeroUsize) -> FillPattern<'_> {
        match self {
            Self::Zero => FillPattern::ZERO,
            Self::Value(bytes) => FillPattern::new(FillTime::IfSet, Some(bytes), elem_size),
            Self::Unknown => FillPattern::UNKNOWN,
        }
    }
}

pub(crate) fn serialize_fill_value_message_v3(
    fill_time: FillTime,
    fill: Option<&[u8]>,
) -> Result<Vec<u8>, FormatError> {
    hdf5_pure_format::__private::serialize_fill_value_message_v3(fill_time, fill).map_err(|error| {
        match error {
            FillValueError::TooLarge { length } => FormatError::SerializationError(format!(
                "fill value length {length} exceeds the u32 message field"
            )),
        }
    })
}

#[cfg(test)]
mod tests;
