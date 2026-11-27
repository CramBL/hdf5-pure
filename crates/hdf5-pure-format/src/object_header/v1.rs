use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use super::HeaderMessage;
use super::MessageFilter;
use super::ObjectHeader;
use super::ObjectHeaderChunk;
use super::ObjectHeaderContinuation;
use super::context::ParseContext;
use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;

/// Defines the fixed prefix width of a version 1 object header message.
///
/// The prefix contains a two-byte type, two-byte data size, one-byte flags
/// field, and three reserved bytes. The layout is defined in "Version 1 Data
/// Object Header Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
const MESSAGE_PREFIX_LEN: usize = 8;

// "Version 1 Data Object Header Prefix" in the format specification, version 4.0, includes
// padding to a multiple of eight bytes in the message body size.
const MESSAGE_ALIGNMENT: u16 = 8;

/// The parser resource limit on continuation hops after the root chunk.
const MAX_V1_CONTINUATION_DEPTH: u16 = 32;

impl ObjectHeader {
    /// Parses the root and its continuations into one depth-first message sequence.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if traversal exceeds 256 continuation
    /// visits or 32 continuation hops after the root.
    /// Returns [`FormatError::UnexpectedEof`] if the body of a framed continuation record is
    /// shorter than the combined offset and length widths, or
    /// [`FormatError::UndefinedContinuationAddress`] if its address is undefined.
    /// Propagates record, address-conversion, and buffer-bound errors.
    pub(super) fn parse_v1(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        filter: &mut MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        // version(1) + reserved(1) + num_messages(2) + ref_count(4) + header_size(4) = 12
        // then pad to 8-byte alignment from start of header
        bytes::ensure_len(data, offset, UNPADDED_HEADER_PREFIX_LEN)?;

        let version = data[offset];
        if version != 1 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }

        let num_messages = LittleEndian::read_u16(&data[offset + 2..offset + 4]);
        let reference_count = LittleEndian::read_u32(&data[offset + 4..offset + 8]);
        let header_data_size = LittleEndian::read_u32(&data[offset + 8..offset + 12]).to_usize()?;

        // Pad to 8-byte alignment: header prefix is 12 bytes, pad to 16
        let padding = 4; // pad 12-byte prefix to 16-byte alignment
        let msg_start = offset
            .checked_add(UNPADDED_HEADER_PREFIX_LEN + padding)
            .ok_or(FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: data.len(),
            })?;

        bytes::ensure_len(data, msg_start, header_data_size)?;

        let records = &data[msg_start..msg_start + header_data_size];
        let mut messages = root_messages(records, num_messages, filter);
        let mut collector = HeaderCollector {
            context,
            traversal: Traversal::new(),
            messages: &mut messages,
            filter,
        };
        collector.collect_buffered(data, records, Some(num_messages), 0)?;

        Ok(ObjectHeader {
            version: 1,
            prefix: None,
            messages,
            reference_count: Some(reference_count),
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        })
    }

    /// Parses a version 1 header and observes each chunk before decoding its records.
    ///
    /// Reads the complete root chunk before reserving space for decoded messages.
    /// Continuations are visited depth-first. The observer borrows the current record buffer.
    ///
    /// # Errors
    ///
    /// Returns the errors from [`Self::parse_v1`] and source-read errors. The parser rejects
    /// an over-budget continuation before reading or observing its target.
    pub(super) fn parse_v1_from_source<S: MetadataSource + ?Sized>(
        source: &S,
        context: ParseContext,
        address: u64,
        filter: &mut MessageFilter<'_>,
        observer: &mut Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    ) -> Result<ObjectHeader, FormatError> {
        // version(1) + reserved(1) + num_messages(2) + ref_count(4) +
        // header_size(4) = 12, padded to 16 before the first message.
        let prefix = source.read_metadata(address, UNPADDED_HEADER_PREFIX_LEN)?;
        let version = prefix[0];
        if version != 1 {
            return Err(FormatError::InvalidObjectHeaderVersion(version));
        }
        let num_messages = LittleEndian::read_u16(&prefix[2..4]);
        let reference_count = LittleEndian::read_u32(&prefix[4..8]);
        let header_data_size = u64::from(LittleEndian::read_u32(&prefix[8..12]));
        let region_addr =
            address
                .checked_add(ALIGNED_HEADER_PREFIX_LEN)
                .ok_or(FormatError::OffsetOverflow {
                    offset: address,
                    length: ALIGNED_HEADER_PREFIX_LEN,
                })?;

        let records = FileRange::new(PhysicalAddress::new(region_addr), header_data_size)?;
        let allocation = FileRange::new(
            PhysicalAddress::new(address),
            ALIGNED_HEADER_PREFIX_LEN
                .checked_add(header_data_size)
                .ok_or(FormatError::OffsetOverflow {
                    offset: address,
                    length: header_data_size,
                })?,
        )?;
        let region = source.read_metadata(region_addr, header_data_size.to_usize()?)?;
        observe_chunk(&region, records, allocation, observer)?;
        let mut messages = root_messages(&region, num_messages, filter);
        let mut collector = HeaderCollector {
            context,
            traversal: Traversal::new(),
            messages: &mut messages,
            filter,
        };
        collector.collect_from_source(source, &region, Some(num_messages), 0, observer)?;

        Ok(ObjectHeader {
            version: 1,
            prefix: None,
            messages,
            reference_count: Some(reference_count),
            flags: 0,
            access_time: None,
            modification_time: None,
            change_time: None,
            birth_time: None,
        })
    }
}

/// Reserves space for decoded messages from the acquired root chunk.
///
/// When the filter keeps all messages, caps the reservation by the declared count and
/// the number of record prefixes that fit in `records`. Returns an empty buffer with
/// zero capacity for a selective filter.
fn root_messages(
    records: &[u8],
    num_messages: u16,
    filter: &MessageFilter<'_>,
) -> Vec<HeaderMessage> {
    Vec::with_capacity(if filter.keeps_all() {
        usize::from(num_messages).min(records.len() / MESSAGE_PREFIX_LEN)
    } else {
        0
    })
}

/// Collects retained messages depth-first with one traversal allowance per header.
///
/// The parse context and filter apply to every chunk. Filtering affects retention only.
/// A root chunk may end with a gap shorter than a record prefix.
struct HeaderCollector<'a, 'f> {
    context: ParseContext,
    traversal: Traversal,
    messages: &'a mut Vec<HeaderMessage>,
    filter: &'a mut MessageFilter<'f>,
}

impl HeaderCollector<'_, '_> {
    /// Parses root records up to `max_messages`, or a continuation's complete record area.
    ///
    /// `depth` counts continuation hops after the root. Each continuation is admitted before
    /// accessing its target bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a continuation chunk ends inside a message
    /// prefix. Propagates record, continuation, and target-access errors.
    fn collect_buffered(
        &mut self,
        data: &[u8],
        mut records: &[u8],
        max_messages: Option<u16>,
        depth: u16,
    ) -> Result<(), FormatError> {
        let mut count = 0usize;
        while !records.is_empty() && max_messages.is_none_or(|max| count < usize::from(max)) {
            let Some((record, remaining)) =
                parse_message_record(records, self.context.access_mode)?
            else {
                if max_messages.is_none() {
                    return Err(FormatError::UnexpectedEof {
                        expected: MESSAGE_PREFIX_LEN,
                        available: records.len(),
                    });
                }
                break;
            };
            records = remaining;
            count += 1;
            if let Some(permit) = self.ingest(record, depth)? {
                self.follow_buffered(data, permit)?;
            }
        }
        Ok(())
    }

    fn follow_buffered(
        &mut self,
        data: &[u8],
        permit: ContinuationVisit,
    ) -> Result<(), FormatError> {
        let ContinuationVisit { range, depth } = permit;
        let offset = range.start().get().to_usize()?;
        let length = range.length().to_usize()?;
        bytes::ensure_len(data, offset, length)?;
        self.collect_buffered(data, &data[offset..offset + length], None, depth)
    }

    /// Parses root records up to `max_messages`, or a continuation's complete record area.
    ///
    /// Active ancestor buffers remain borrowed during recursive continuation reads.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a continuation chunk ends inside a message
    /// prefix. Propagates record, continuation, address-conversion, and source-read errors.
    fn collect_from_source<S: MetadataSource + ?Sized>(
        &mut self,
        source: &S,
        mut records: &[u8],
        max_messages: Option<u16>,
        depth: u16,
        observer: &mut Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    ) -> Result<(), FormatError> {
        let mut count = 0usize;
        while !records.is_empty() && max_messages.is_none_or(|max| count < usize::from(max)) {
            let Some((record, remaining)) =
                parse_message_record(records, self.context.access_mode)?
            else {
                if max_messages.is_none() {
                    return Err(FormatError::UnexpectedEof {
                        expected: MESSAGE_PREFIX_LEN,
                        available: records.len(),
                    });
                }
                break;
            };
            records = remaining;
            count += 1;
            if let Some(permit) = self.ingest(record, depth)? {
                self.follow_from_source(source, permit, observer)?;
            }
        }
        Ok(())
    }

    fn follow_from_source<S: MetadataSource + ?Sized>(
        &mut self,
        source: &S,
        permit: ContinuationVisit,
        observer: &mut Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    ) -> Result<(), FormatError> {
        let ContinuationVisit { range, depth } = permit;
        let region = source.read_metadata(range.start().get(), range.length().to_usize()?)?;
        observe_chunk(&region, range, range, observer)?;
        self.collect_from_source(source, &region, None, depth, observer)
    }

    /// Parses continuation fields before applying the message filter.
    ///
    /// Admits the continuation to traversal independently of whether the filter keeps the record.
    ///
    /// # Errors
    ///
    /// Returns errors from [`ObjectHeaderContinuation::parse_with_widths`] and
    /// [`Traversal::admit`].
    fn ingest(
        &mut self,
        record: MessageRecord<'_>,
        depth: u16,
    ) -> Result<Option<ContinuationVisit>, FormatError> {
        let continuation = if record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
            Some(ObjectHeaderContinuation::parse_with_widths(
                record.body,
                self.context.widths,
            )?)
        } else {
            None
        };
        self.keep(record);
        continuation
            .map(|continuation| {
                self.traversal
                    .admit(self.context.base_address, continuation, depth)
            })
            .transpose()
    }

    fn keep(&mut self, record: MessageRecord<'_>) {
        if record.msg_type != MessageType::NIL && self.filter.keeps(record.msg_type, record.body) {
            self.messages.push(HeaderMessage {
                msg_type: record.msg_type,
                size: record.body.len(),
                flags: record.flags,
                creation_order: None,
                data: record.body.to_vec(),
            });
        }
    }
}

fn observe_chunk(
    bytes: &[u8],
    records: FileRange,
    allocation: FileRange,
    observer: &mut Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
) -> Result<(), FormatError> {
    if let Some(observer) = observer.as_deref_mut() {
        observer(ObjectHeaderChunk::new(
            allocation,
            FileRegion::new(records, bytes)?,
            None,
        ));
    }
    Ok(())
}

/// Counts followed continuation references across all branches of one header.
///
/// Repeated targets and zero-length targets each cost one visit. The root chunk is excluded.
struct Traversal {
    visits: u16,
}

impl Traversal {
    fn new() -> Self {
        Self { visits: 0 }
    }

    /// Charges one visit and checks depth and target geometry before issuing a permit.
    ///
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if the visit or depth allowance is exhausted.
    /// Propagates address-resolution and range-overflow errors.
    fn admit(
        &mut self,
        base_address: BaseAddress,
        continuation: ObjectHeaderContinuation,
        parent_depth: u16,
    ) -> Result<ContinuationVisit, FormatError> {
        if parent_depth >= MAX_V1_CONTINUATION_DEPTH || self.visits >= MAX_V1_CONTINUATION_VISITS {
            return Err(FormatError::NestingDepthExceeded);
        }
        self.visits += 1;
        Ok(ContinuationVisit {
            range: FileRange::new(
                base_address.resolve(continuation.address())?,
                continuation.length(),
            )?,
            depth: parent_depth + 1,
        })
    }
}

/// A charged continuation visit with a checked physical range and depth.
///
/// Constructed by [`Traversal::admit`] and consumed when following the target. The parser
/// checks that target bytes are present before decoding records.
struct ContinuationVisit {
    range: FileRange,
    depth: u16,
}

/// The parser resource limit on followed continuation references per header.
const MAX_V1_CONTINUATION_VISITS: u16 = 256;

/// A version 1 object header message with a complete, aligned body.
///
/// The body borrows from the current header chunk. The type, size, flags,
/// reserved bytes, and data layout are defined in "Version 1 Data Object
/// Header Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
#[derive(Clone, Copy)]
struct MessageRecord<'a> {
    msg_type: MessageType,
    flags: MessageFlags,
    body: &'a [u8],
}

/// Parses the first record and returns it with the unconsumed bytes.
///
/// Returns `Ok(None)` when fewer than [`MESSAGE_PREFIX_LEN`] bytes remain.
/// Checks the declared size for alignment before acquiring the body.
///
/// # Errors
///
/// Returns [`FormatError::InvalidObjectHeaderMessageSize`] if the declared body size is not a
/// multiple of eight, or [`FormatError::UnexpectedEof`] if an aligned body extends past `data`.
/// EOF extents are relative to the start of `data`. After acquiring the body, returns
/// [`FormatError::UnsupportedMessage`] if an unknown message type must be understood under
/// `access_mode`.
///
/// The record layout is defined in "Version 1 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
fn parse_message_record(
    data: &[u8],
    access_mode: AccessMode,
) -> Result<Option<(MessageRecord<'_>, &[u8])>, FormatError> {
    let mut cursor = bytes::ByteCursor::new(data);
    let Ok(prefix) = cursor.take_array::<MESSAGE_PREFIX_LEN>() else {
        return Ok(None);
    };

    let body_len = LittleEndian::read_u16(&prefix[2..4]);
    if !body_len.is_multiple_of(MESSAGE_ALIGNMENT) {
        return Err(FormatError::InvalidObjectHeaderMessageSize(body_len));
    }
    let body = cursor.take(usize::from(body_len))?;

    let msg_type = MessageType::from_u16(LittleEndian::read_u16(&prefix[..2]));
    let flags = MessageFlags::new(prefix[4]);
    if let Some(id) = msg_type.unknown_id()
        && flags.must_be_understood(access_mode)
    {
        return Err(FormatError::UnsupportedMessage(id));
    }

    Ok(Some((
        MessageRecord {
            msg_type,
            flags,
            body,
        },
        cursor.remaining(),
    )))
}

/// The number of bytes before the alignment padding of a version 1 object header.
///
/// The fields are defined in "Version 1 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
const UNPADDED_HEADER_PREFIX_LEN: usize = 12;

/// The number of bytes before the first message of a version 1 object header.
///
/// This includes the fields and alignment padding defined in "Version 1 Data Object Header
/// Prefix" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
const ALIGNED_HEADER_PREFIX_LEN: u64 = 16;
