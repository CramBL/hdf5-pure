//! The chunks of a version 2 object header, chunk 0 and its continuation blocks, as their
//! message records.
//!
//! Chunk 0 is defined in ["Version 2 Data Object Header Prefix"][prefix] and a continuation block
//! in ["The Object Header Continuation Message"][continuation] of the format specification,
//! version 4.0.
//!
//! [prefix]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
//! [continuation]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_continuation

use crate::checksum;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::ParseContext;
use crate::object_header::v2;
use crate::object_header::v2::CHECKSUM_LEN;
use crate::object_header::v2::MessageRecord;
use crate::object_header::v2::MessageRecordLayout;
use crate::object_header::v2::ObjectHeaderContinuation;
use crate::object_header::v2::ParsedObjectHeaderPrefix;

/// Chunk 0 of a version 2 object header, with the prefix that opens it.
pub(super) struct RootChunk<'a> {
    pub(super) prefix: ParsedObjectHeaderPrefix,
    pub(super) chunk: HeaderChunk<'a>,
}

impl<'a> RootChunk<'a> {
    /// Returns chunk 0 of the header whose prefix is `prefix`, from `image`, the bytes of the chunk
    /// from the signature to the end of the checksum.
    ///
    /// # Errors
    ///
    /// Returns the errors [`checksummed`] returns.
    pub(super) fn parse(
        prefix: ParsedObjectHeaderPrefix,
        image: &'a [u8],
    ) -> Result<Self, FormatError> {
        Ok(Self {
            prefix,
            chunk: HeaderChunk {
                region: checksummed(image)?,
                records_start: prefix.len,
                layout: prefix.prefix.layout,
            },
        })
    }
}

/// The message records of one chunk of a version 2 object header, chunk 0 or a continuation
/// block.
///
/// [`RootChunk::parse`] and [`continuation`](Self::continuation) verify the checksum of the chunk
/// where the `checksum` feature is enabled.
#[derive(Clone, Copy)]
pub(super) struct HeaderChunk<'a> {
    /// The bytes of the chunk before its checksum.
    region: &'a [u8],
    /// The offset of the first message record in `region`.
    records_start: usize,
    layout: MessageRecordLayout,
}

impl<'a> HeaderChunk<'a> {
    /// Returns the chunk of the continuation block `block`, from its `OCHK` signature to the end of
    /// its checksum, whose records are in `layout`.
    ///
    /// # Errors
    ///
    /// Returns the errors [`v2::continuation_block_messages`] returns, and the errors
    /// [`checksummed`] returns.
    pub(super) fn continuation(
        layout: MessageRecordLayout,
        block: &'a [u8],
    ) -> Result<Self, FormatError> {
        let records_start = v2::continuation_block_messages(block)?.start;
        Ok(Self {
            region: checksummed(block)?,
            records_start,
            layout,
        })
    }

    /// Returns the message records of the chunk in the order the chunk stores them, each as
    /// [`ChunkRecord::classify`] returns it under `context`.
    pub(super) fn records(
        self,
        context: ParseContext,
    ) -> impl Iterator<Item = Result<ChunkRecord<'a>, FormatError>> {
        self.message_records()
            .map(move |record| ChunkRecord::classify(context, record))
    }

    /// Returns the number of message records in the chunk other than Nil and continuation
    /// messages, the number of messages a parse under
    /// [`MessageFilter::All`](crate::object_header::MessageFilter::All) keeps from it.
    pub(super) fn message_count(self) -> usize {
        self.message_records()
            .filter(|record| {
                !matches!(
                    record.msg_type,
                    MessageType::NIL | MessageType::OBJECT_HEADER_CONTINUATION
                )
            })
            .count()
    }

    /// Returns the message records of the chunk in the order the chunk stores them.
    ///
    /// The records end where fewer bytes remain than a record prefix, or at a record whose body
    /// runs past the chunk.
    fn message_records(self) -> impl Iterator<Item = MessageRecord<'a>> {
        let mut msg_start = self.records_start;
        core::iter::from_fn(move || {
            let record = self
                .layout
                .next_message(self.region, msg_start)
                .ok()
                .flatten()?;
            msg_start = record.body_range.end;
            Some(record)
        })
    }
}

/// A message record of a chunk, by what a parse does with it.
pub(super) enum ChunkRecord<'a> {
    /// An Object Header Continuation message, which refers to a continuation block to read.
    Continuation(ObjectHeaderContinuation),
    /// A message of any other type, which a filter may keep.
    Message(MessageRecord<'a>),
    /// A Nil message, which a parse skips.
    Nil,
}

impl<'a> ChunkRecord<'a> {
    /// Classifies `record` by its type, and parses the body of a continuation message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedMessage`] if `record` is of a type this crate does not
    /// recognize and its flags mark it as one to reject under `context.access_mode`, and the errors
    /// [`ObjectHeaderContinuation::parse`] returns for a continuation message.
    fn classify(context: ParseContext, record: MessageRecord<'a>) -> Result<Self, FormatError> {
        if let Some(id) = record.msg_type.unknown_id()
            && record.flags.must_be_understood(context.access_mode)
        {
            return Err(FormatError::UnsupportedMessage(id));
        }
        Ok(match record.msg_type {
            MessageType::NIL => Self::Nil,
            MessageType::OBJECT_HEADER_CONTINUATION => {
                // The address is not narrowed here, so that a parse through a
                // `MetadataSource` can read a continuation block past 4 GiB on a
                // 32-bit host.
                Self::Continuation(ObjectHeaderContinuation::parse(
                    record.body,
                    context.offset_size,
                    context.length_size,
                )?)
            }
            _ => Self::Message(record),
        })
    }
}

/// Verifies the checksum that ends `image` and returns the bytes before it.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `image` is shorter than the checksum, and
/// [`FormatError::ChecksumMismatch`] if the `checksum` feature is enabled and the checksum does
/// not match the bytes before it.
fn checksummed(image: &[u8]) -> Result<&[u8], FormatError> {
    checksum::verify_trailing(image)?;
    image
        .split_last_chunk::<CHECKSUM_LEN>()
        .map(|(covered, _)| covered)
        .ok_or(FormatError::UnexpectedEof {
            expected: CHECKSUM_LEN,
            available: image.len(),
        })
}
