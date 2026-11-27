use crate::convert::Narrow;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_type::MessageType;
use crate::object_header::v2::continuation::OCHK_SIGNATURE;
use crate::object_header::v2::{CHECKSUM_LEN, MessageRecordLayout, ObjectHeaderPrefix};
use crate::width::{self, FormatWidths, UintWidth};

use super::geometry::ObjectHeaderAllocationGeometry;

/// The complete encoded length and padding of one chunk, with any space reserved for later chunks.
#[derive(Debug)]
pub(super) struct ChunkSpace {
    len: usize,
    padding: ChunkPadding,
    remaining: Option<RemainingSpace>,
}

impl ChunkSpace {
    /// Plans chunk 0 at the proposed size-field width within a contiguous allocation.
    ///
    /// Returns `Ok(None)` if the records do not fit or the proposed width cannot represent a
    /// valid chunk body for this allocation.
    ///
    /// # Errors
    ///
    /// Propagates errors from planning the padding and continuation space.
    pub(super) fn root(
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        width: UintWidth,
        allocation_len: usize,
    ) -> Result<Option<Self>, FormatError> {
        let fixed = geometry.root_fixed(width);
        let occupied = fixed + geometry.records_len();
        let Some(available) = allocation_len.checked_sub(occupied) else {
            return Ok(None);
        };
        let prefix = geometry.prefix();
        let space = Self::new(prefix, geometry.widths(), occupied, available, true)?;
        let body = space.len - fixed;
        Ok(((body == 0 || body >= prefix.layout.prefix_len())
            && UintWidth::smallest_for_len(body).get() <= width.get())
        .then_some(space))
    }

    /// Selects the smallest size-field width that permits a root chunk of exactly `length` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidObjectHeaderChunkLength`] if the root cannot occupy the
    /// supplied length, or [`FormatError::ValueTooLargeForPlatform`] if it exceeds [`usize::MAX`].
    pub(super) fn placed_root(
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        length: u64,
        continued: bool,
    ) -> Result<(UintWidth, Self), FormatError> {
        let width = geometry.root_width(length, continued)?;
        let fixed = geometry.root_fixed(width) + geometry.records_len();
        let space = Self::with_length(geometry, 0, fixed, length, continued)?;
        Ok((width, space))
    }

    pub(super) fn placed_continuation(
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        index: usize,
        length: u64,
        continued: bool,
    ) -> Result<Self, FormatError> {
        if !geometry
            .continuation_chunk_lengths(continued)
            .contains(&length)
        {
            return Err(FormatError::InvalidObjectHeaderChunkLength { index, length });
        }
        Self::with_length(
            geometry,
            index,
            OCHK_SIGNATURE.len() + CHECKSUM_LEN,
            length,
            continued,
        )
    }

    fn with_length(
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        index: usize,
        fixed: usize,
        length: u64,
        continued: bool,
    ) -> Result<Self, FormatError> {
        let occupied = fixed
            + if continued {
                geometry.continuation_len()
            } else {
                0
            };
        let len = length.to_usize()?;
        let padding = len
            .checked_sub(occupied)
            .ok_or(FormatError::InvalidObjectHeaderChunkLength { index, length })?;
        Ok(Self {
            len,
            padding: ChunkPadding::new(geometry.prefix().layout, padding)?,
            remaining: None,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn padding(&self) -> ChunkPadding {
        self.padding
    }

    pub(super) fn remaining(&self) -> Option<RemainingSpace> {
        self.remaining
    }

    fn new(
        prefix: ObjectHeaderPrefix,
        widths: FormatWidths,
        fixed: usize,
        available: usize,
        root: bool,
    ) -> Result<Self, FormatError> {
        let maximum = prefix.layout.prefix_len() + OBJECT_HEADER_MESSAGE_MAX;
        let limit = if root {
            usize::MAX
        } else {
            width::length_width_max(widths.lengths)
                .min(usize::MAX as u64)
                .to_usize()?
        };
        if available <= maximum && fixed + available <= limit {
            return Ok(Self {
                len: fixed + available,
                padding: ChunkPadding::new(prefix.layout, available)?,
                remaining: None,
            });
        }
        let continuation = prefix.layout.prefix_len()
            + usize::from(widths.offsets.get())
            + usize::from(widths.lengths.get());
        let minimum = OCHK_SIGNATURE.len() + prefix.layout.prefix_len() + CHECKSUM_LEN;
        let padding = maximum
            .min(limit.saturating_sub(fixed + continuation))
            .min(available - continuation - minimum);
        let len = fixed + padding + continuation;
        let remaining = available - padding - continuation;
        Ok(Self {
            len,
            padding: ChunkPadding::new(prefix.layout, padding)?,
            remaining: Some(RemainingSpace(remaining)),
        })
    }
}

/// The unused suffix of a contiguous allocation, large enough for a continuation chunk.
#[derive(Clone, Copy, Debug)]
pub(super) struct RemainingSpace(usize);

impl RemainingSpace {
    pub(super) fn continuation(
        self,
        geometry: &ObjectHeaderAllocationGeometry<'_>,
    ) -> Result<ChunkSpace, FormatError> {
        let fixed = OCHK_SIGNATURE.len() + CHECKSUM_LEN;
        ChunkSpace::new(
            geometry.prefix(),
            geometry.widths(),
            fixed,
            self.0 - fixed,
            false,
        )
    }
}

/// Padding in an object header chunk as a short gap or a NIL record.
#[derive(Clone, Copy, Debug)]
pub(super) enum ChunkPadding {
    /// The number of padding bytes in a gap shorter than a record prefix.
    Gap(u16),
    /// The length of the NIL body in bytes.
    Nil(u16),
}

impl ChunkPadding {
    /// Plans a NIL record when `len` can fit its prefix, or a gap otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ObjectHeaderMessageTooLarge`] if the padding cannot fit in one
    /// NIL record.
    fn new(layout: MessageRecordLayout, len: usize) -> Result<Self, FormatError> {
        let body = len.checked_sub(layout.prefix_len());
        let size = body.unwrap_or(len);
        let size =
            u16::try_from(size).map_err(|_error| FormatError::ObjectHeaderMessageTooLarge {
                message_type: MessageType::NIL.to_u16(),
                size,
            })?;
        Ok(if body.is_some() {
            Self::Nil(size)
        } else {
            Self::Gap(size)
        })
    }
}
