use crate::convert::Narrow;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_type::MessageType;
use crate::object_header::v2::continuation::OCHK_SIGNATURE;
use crate::object_header::v2::{CHECKSUM_LEN, MessageRecordLayout, ObjectHeaderPrefix};
use crate::width::{self, FormatWidths};

#[derive(Debug)]
pub(super) struct ChunkSpace {
    len: usize,
    padding: ChunkPadding,
    remaining: Option<usize>,
}

impl ChunkSpace {
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn padding(&self) -> ChunkPadding {
        self.padding
    }
    pub(super) fn remaining(&self) -> Option<usize> {
        self.remaining
    }

    pub(super) fn with_padding(
        layout: MessageRecordLayout,
        len: usize,
        padding: usize,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            len,
            padding: ChunkPadding::new(layout, padding)?,
            remaining: None,
        })
    }

    pub(super) fn new(
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
            remaining: Some(remaining),
        })
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
