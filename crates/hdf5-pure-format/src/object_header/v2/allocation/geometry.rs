use alloc::vec::Vec;
use core::ops::{Range, RangeInclusive};

use crate::convert::Narrow;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_type::MessageType;
use crate::object_header::header::OHDR_SIGNATURE;
use crate::object_header::v2::continuation::OCHK_SIGNATURE;
use crate::object_header::v2::worklist;
use crate::object_header::v2::{CHECKSUM_LEN, ObjectHeaderPrefix};
use crate::width::{self, FormatWidths, UintWidth};

/// Holds checked message records for planning version 2 object-header chunks.
///
/// Borrows the source records and retains the ranges of the non-NIL records. Planning keeps
/// those records in chunk 0, in source order, with their flags and creation indices.
/// Use [`Self::contiguous_allocations`] to inspect chunk ranges before consuming the geometry.
#[derive(Debug)]
pub struct ObjectHeaderAllocationGeometry<'a> {
    prefix: ObjectHeaderPrefix,
    widths: FormatWidths,
    messages: &'a [u8],
    records: Vec<Range<usize>>,
    records_len: usize,
}

impl<'a> ObjectHeaderAllocationGeometry<'a> {
    /// Parses the records once for subsequent allocation planning.
    ///
    /// The caller passes records in `prefix.layout`, with source continuations already flattened.
    /// Discards NIL records and trailing bytes shorter than a record prefix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a message body extends beyond `messages`, and
    /// [`FormatError::Internal`] if `messages` contains a continuation record.
    pub fn new(
        prefix: ObjectHeaderPrefix,
        widths: FormatWidths,
        messages: &'a [u8],
    ) -> Result<Self, FormatError> {
        let mut records = Vec::new();
        let mut records_len = 0;
        let mut start = 0;
        while let Some(record) = prefix.layout.next_message(messages, start)? {
            let end = record.body_range.end;
            if record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                return Err(FormatError::Internal(
                    "an object-header allocation requires flattened message records".into(),
                ));
            }
            if record.msg_type != MessageType::NIL {
                records.push(start..end);
                records_len += end - start;
            }
            start = end;
        }
        Ok(Self {
            prefix,
            widths,
            messages,
            records,
            records_len,
        })
    }

    pub(super) fn append_records(&self, out: &mut Vec<u8>) {
        for record in &self.records {
            out.extend_from_slice(&self.messages[record.clone()]);
        }
    }

    pub(super) fn prefix(&self) -> ObjectHeaderPrefix {
        self.prefix
    }

    pub(super) fn records_len(&self) -> usize {
        self.records_len
    }

    /// Returns the address and length widths used for chunk planning and encoding.
    pub fn widths(&self) -> FormatWidths {
        self.widths
    }

    /// Returns the complete chunk 0 length ranges that can represent the retained records.
    ///
    /// With `continued`, includes space for a continuation record. The ranges include the header
    /// prefix, its optional fields, and the checksum. They are ordered by increasing Size of
    /// Chunk #0 field width and may overlap. For empty input without a continuation, they include
    /// single-length ranges for empty chunk bodies.
    pub fn root_chunk_lengths(
        &self,
        continued: bool,
    ) -> impl Iterator<Item = RangeInclusive<u64>> + '_ {
        self.root_geometries(continued).map(|(_, range)| range)
    }

    /// Returns the complete continuation chunk lengths that fit the output length field.
    ///
    /// Includes the signature, checksum, and space for at most one NIL record. With `continued`,
    /// also includes a continuation record. A terminal chunk contains at least one NIL record.
    pub fn continuation_chunk_lengths(&self, continued: bool) -> RangeInclusive<u64> {
        let fixed = (OCHK_SIGNATURE.len() + CHECKSUM_LEN).to_u64();
        let occupied = if continued {
            self.continuation_len()
        } else {
            0
        }
        .to_u64();
        let prefix = self.prefix.layout.prefix_len().to_u64();
        let minimum = fixed + if continued { occupied } else { prefix };
        let maximum = (fixed + occupied + prefix + OBJECT_HEADER_MESSAGE_MAX.to_u64())
            .min(width::length_width_max(self.widths.lengths));
        minimum..=maximum
    }

    pub(super) fn root_width(
        &self,
        length: u64,
        continued: bool,
    ) -> Result<UintWidth, FormatError> {
        self.root_geometries(continued)
            .find_map(|(width, range)| range.contains(&length).then_some(width))
            .ok_or(FormatError::InvalidObjectHeaderChunkLength { index: 0, length })
    }

    fn root_geometries(
        &self,
        continued: bool,
    ) -> impl Iterator<Item = (UintWidth, RangeInclusive<u64>)> + '_ {
        [
            UintWidth::One,
            UintWidth::Two,
            UintWidth::Four,
            UintWidth::Eight,
        ]
        .into_iter()
        .flat_map(move |width| {
            let fixed = self.root_fixed(width).to_u64();
            let occupied = self.records_len.to_u64()
                + if continued {
                    self.continuation_len().to_u64()
                } else {
                    0
                };
            let prefix = self.prefix.layout.prefix_len().to_u64();
            let minimum = fixed + if occupied == 0 { prefix } else { occupied };
            let body_limit = u64::MAX >> (64 - 8 * width.get());
            let maximum =
                fixed + (occupied + prefix + OBJECT_HEADER_MESSAGE_MAX.to_u64()).min(body_limit);
            [
                (occupied == 0).then_some((width, fixed..=fixed)),
                (minimum <= maximum).then_some((width, minimum..=maximum)),
            ]
            .into_iter()
            .flatten()
        })
    }

    pub(super) fn root_fixed(&self, width: UintWidth) -> usize {
        OHDR_SIGNATURE.len()
            + 2
            + self.prefix.optional_len()
            + usize::from(width.get())
            + CHECKSUM_LEN
    }

    pub(super) fn continuation_len(&self) -> usize {
        self.prefix.layout.prefix_len()
            + usize::from(self.widths.offsets.get())
            + usize::from(self.widths.lengths.get())
    }

    /// The maximum number of complete chunks in one allocation, including chunk 0.
    pub const MAX_CHUNKS: usize = worklist::MAX_CONTINUATIONS;
}
