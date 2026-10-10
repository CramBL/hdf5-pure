//! The messages a parse keeps from the chunks of a version 2 object header.

use alloc::vec::Vec;

use crate::address::BaseAddressExt;
use crate::error::FormatError;
use crate::object_header::HeaderMessage;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::object_header::ParseContext;
use crate::object_header::v2::MessageRecord;
use crate::object_header::v2::MessageRecordLayout;
use crate::object_header::v2::ObjectHeaderContinuation;
use crate::object_header::v2::ParsedObjectHeaderPrefix;
use crate::object_header::v2::chunk::ChunkRecord;
use crate::object_header::v2::chunk::HeaderChunk;
use crate::object_header::v2::chunk::RootChunk;

/// The messages of a version 2 object header that a parse keeps, chunk by chunk.
///
/// The caller reads chunk 0 and passes it to [`new`](Self::new), then reads each block
/// [`next_continuation`](Self::next_continuation) returns and passes it to
/// [`ingest`](Self::ingest), until `next_continuation` returns `None`. [`finish`](Self::finish)
/// returns the header.
pub(super) struct HeaderCollector<'f> {
    context: ParseContext,
    filter: MessageFilter<'f>,
    prefix: ParsedObjectHeaderPrefix,
    messages: Vec<HeaderMessage>,
    continuations: Vec<ObjectHeaderContinuation>,
    continuations_left: u16,
}

impl<'f> HeaderCollector<'f> {
    /// Returns a collector of the messages `filter` keeps, with chunk 0 ingested.
    ///
    /// # Errors
    ///
    /// Returns the errors [`ingest`](Self::ingest) returns.
    pub(super) fn new(
        context: ParseContext,
        filter: MessageFilter<'f>,
        RootChunk { prefix, chunk }: RootChunk<'_>,
    ) -> Result<Self, FormatError> {
        let mut collector = Self {
            context,
            filter,
            prefix,
            messages: Vec::new(),
            continuations: Vec::new(),
            continuations_left: MAX_CONTINUATIONS,
        };
        collector.ingest(chunk)?;
        Ok(collector)
    }

    /// Adds the messages of `chunk` that the filter keeps to the header, and records the blocks its
    /// continuation messages refer to as blocks left to read.
    ///
    /// # Errors
    ///
    /// Returns the errors [`ChunkRecord::classify`] returns for a record of `chunk`.
    pub(super) fn ingest(&mut self, chunk: HeaderChunk<'_>) -> Result<(), FormatError> {
        if self.filter.keeps_all() {
            self.messages.reserve(chunk.message_count());
        }
        for record in chunk.records(self.context) {
            match record? {
                ChunkRecord::Nil => {}
                ChunkRecord::Message(message) => self.keep_if_filtered(message),
                ChunkRecord::Continuation(continuation) => self.continuations.push(continuation),
            }
        }
        Ok(())
    }

    /// Returns the next continuation block to read, or `None` once the collector has returned every
    /// block the ingested chunks refer to.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if the collector has returned
    /// [`MAX_CONTINUATIONS`] blocks already, and [`FormatError::OffsetOverflow`] if the absolute
    /// address of the block exceeds `u64`.
    pub(super) fn next_continuation(&mut self) -> Result<Option<ContinuationBlock>, FormatError> {
        let Some(continuation) = self.continuations.pop() else {
            return Ok(None);
        };
        self.continuations_left = self
            .continuations_left
            .checked_sub(1)
            .ok_or(FormatError::NestingDepthExceeded)?;
        Ok(Some(ContinuationBlock {
            address: self.context.base_address.absolute(continuation.address())?,
            length: continuation.length(),
            layout: self.prefix.prefix.layout,
        }))
    }

    /// Returns the object header, with the messages the filter kept.
    pub(super) fn finish(self) -> ObjectHeader {
        self.prefix.object_header(self.messages)
    }

    fn keep_if_filtered(
        &mut self,
        MessageRecord {
            msg_type,
            flags,
            creation_index,
            body,
            body_range: _,
        }: MessageRecord<'_>,
    ) {
        if self.filter.keeps(msg_type, body) {
            self.messages.push(HeaderMessage {
                msg_type,
                size: body.len(),
                flags,
                creation_order: creation_index,
                data: body.to_vec(),
            });
        }
    }
}

/// A continuation block for the caller to read: its absolute address and its length in bytes.
pub(super) struct ContinuationBlock {
    pub(super) address: u64,
    pub(super) length: u64,
    layout: MessageRecordLayout,
}

impl ContinuationBlock {
    /// Returns the chunk of `block`, the bytes the caller read at [`address`](Self::address).
    ///
    /// # Errors
    ///
    /// Returns the errors [`HeaderChunk::continuation`] returns.
    pub(super) fn parse<'a>(&self, block: &'a [u8]) -> Result<HeaderChunk<'a>, FormatError> {
        HeaderChunk::continuation(self.layout, block)
    }
}

/// Limits continuation traversal to protect the parser from cycles.
const MAX_CONTINUATIONS: u16 = 256;
