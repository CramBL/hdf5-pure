//! Object-header traversal and reconstruction for staged edits.

use hdf5_pure_format::__private::LayoutVersion;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::OBJECT_HEADER_PREFIX_MAX_LEN;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::RecordSplice;

use crate::address::AddressContext;
use crate::convert::Narrow;
use crate::data_layout::{COMPACT_DATA_OFFSET, DataLayout, LAYOUT_CLASS_COMPACT};
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_space_info::FileSpaceInfo;
use crate::file_writer::{LENGTH_SIZE, OFFSET_SIZE};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::Source;
use crate::width::FormatWidths;

use super::WriteEngine;

impl WriteEngine {
    /// Reads the extension header and replaces its File Space Info message bodies.
    ///
    /// `ext_addr` is an absolute file offset. The replacement is a persistent version 1 message
    /// encoded at the file's widths.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the header cannot be traversed or has no File
    /// Space Info message. Propagates parsing, source-read, and encoding errors.
    pub(super) fn rewrite_extension_region(
        &self,
        ext_addr: u64,
        info: &FileSpaceInfo,
    ) -> Result<OhRegion, Error> {
        let widths = self.widths;
        let region = Self::gather_oh_messages(&self.image(), ext_addr, self.addresses)?;
        rewrite_extension_region_bytes(&region, widths, info)
    }
    /// Collect every message of the object header at `addr` into one contiguous
    /// region, following continuation blocks across chunks and dropping the
    /// `Continuation` messages themselves. Re-emitting the result through
    /// [`build_v2_object_header`] collapses a multi-chunk header (as the
    /// reference C library often writes) into a single chunk, which is how this
    /// editor rebuilds headers. The chunk-0 prefix is validated by
    /// [`oh_region_at`]. Each continuation block must be a well-formed `OCHK`
    /// block within the file.
    ///
    /// Reads each header chunk out of `src` as one bounded buffer, so this serves a session whose file is not
    /// mirrored in memory (issue #198).
    pub(super) fn gather_oh_messages<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        base: AddressContext,
    ) -> Result<OhRegion, Error> {
        let chunks = read_oh_chunks(src, addr, base)?;
        // Every chunk of one header shares what chunk 0's prefix declared, which
        // is what makes concatenating their records into one region well defined,
        // and what carries the prefix's optional blocks through to the rebuild.
        let mut out = OhRegion::empty(chunks[0].props());
        for chunk in &chunks {
            let layout = chunk.layout();
            let (region, mut p) = chunk.message_region();
            while let Some((msg_type, _body, body_end)) = next_record(layout, region, p)? {
                if msg_type != MessageType::OBJECT_HEADER_CONTINUATION {
                    out.push_bytes(&region[p..body_end]);
                }
                p = body_end;
            }
        }
        Ok(out)
    }
}

/// Returns a copy of `region` whose first Data Layout message holds `raw` as its compact data.
///
/// The rewritten record keeps its flags, its creation index, and its layout version. Every
/// other record is copied unchanged, a later Data Layout message included. The compact fields
/// of layout versions 3 and 4 are defined in "The Data Layout Message" of the [format
/// specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if `region` lacks a Data Layout message, if the body of a
/// record it reads runs past the end of `region`, if the byte after the version of the first
/// Data Layout message, the layout class in versions 3 and 4, is not the compact class, or if
/// `raw` is longer than [`OBJECT_HEADER_MESSAGE_MAX`] less [`COMPACT_DATA_OFFSET`]. Returns
/// [`Error::Format`] if that byte is the compact class and the version is neither 3 nor 4.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub(super) fn rebuild_compact_layout_region(
    region: &OhRegion,
    raw: &[u8],
) -> Result<OhRegion, Error> {
    // This guard applies the bound of `encode_compact` to the whole message body
    // the layout becomes (version, class, the 2-byte inline size and the data),
    // and rejects a body that does not fit with `EditUnsupported`.
    if raw.len() > OBJECT_HEADER_MESSAGE_MAX - COMPACT_DATA_OFFSET {
        return Err(Error::EditUnsupported(
            "compact dataset data is too large to overwrite in place",
        ));
    }
    // libhdf5 reads the first Data Layout message of a header (`H5O_msg_read_oh`
    // in `H5Omessage.c`, HDF5 2.2.0).
    let splice = region
        .splice_first(MessageType::DATA_LAYOUT)?
        .ok_or(Error::EditUnsupported(
            "compact dataset header has no data-layout message",
        ))?;
    let &[version, LAYOUT_CLASS_COMPACT, ..] = splice.record().body else {
        return Err(Error::EditUnsupported(
            "compact-layout overwrite found a non-compact data layout",
        ));
    };
    let layout = DataLayout::encode_compact(LayoutVersion::try_from(version)?, raw)?;
    Ok(region.with_bytes(splice.replace_body(&layout)?))
}

/// Replaces every File Space Info body with a persistent version 1 encoding of `info`.
///
/// Preserves the original record flags, creation indexes, prefix properties, and unrelated
/// bytes. Both commit paths use this operation to prepare the extension region.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or no File Space Info
/// message is present. Returns [`Error::Format`] if `info` cannot be encoded.
pub(crate) fn rewrite_extension_region_bytes(
    region: &OhRegion,
    widths: FormatWidths,
    info: &FileSpaceInfo,
) -> Result<OhRegion, Error> {
    let new_body = hdf5_pure_format::__private::serialize_file_space_info_v1(widths, info)?;
    // The message body is the fixed-size File Space Info record (≤ 125 bytes),
    // so it always fits the u16 size field.
    let new_len: u16 = new_body
        .len()
        .narrow_or_else(|| Error::EditUnsupported("File Space Info message too large"))?;
    let mut out = Vec::with_capacity(region.len());
    let mut p = 0;
    let mut replaced = false;
    while let Some((msg_type, _body, body_end)) = region.next_message(p)? {
        if msg_type == MessageType::FILE_SPACE_INFO {
            out.push(region[p]); // message type byte
            out.extend_from_slice(&new_len.to_le_bytes());
            // Preserve the message flags (0x14) and, where the header carries
            // one, the record's creation index.
            out.extend_from_slice(&region[p + 3..p + region.layout().prefix_len()]);
            out.extend_from_slice(&new_body);
            replaced = true;
        } else {
            out.extend_from_slice(&region[p..body_end]);
        }
        p = body_end;
    }
    if !replaced {
        // Persistence is armed only when the extension already carries a File
        // Space Info message. An absent message is an error.
        return Err(Error::EditUnsupported(
            "a persisting file's superblock extension has no File Space Info message",
        ));
    }
    Ok(region.with_bytes(out))
}

/// Returns the absolute chunk 0 message range and parsed prefix properties.
///
/// `prefix` starts at `addr` and contains up to [`OBJECT_HEADER_PREFIX_MAX_LEN`] bytes.
/// `file_len` bounds the message region and its trailing checksum. The version 2 prefix and
/// chunk fields are defined in "Version 2 Data Object Header Prefix" of the [format
/// specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the header is not version 2, the prefix is truncated,
/// or the chunk or its checksum extends past `file_len`. Returns [`Error::Format`] for other
/// prefix errors.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub(super) fn oh_region_at(
    prefix: &[u8],
    addr: u64,
    file_len: u64,
) -> Result<(u64, u64, ObjectHeaderPrefix), Error> {
    let parsed = ObjectHeaderPrefix::parse(prefix).map_err(|err| match err {
        FormatError::InvalidObjectHeaderSignature | FormatError::InvalidObjectHeaderVersion(_) => {
            Error::EditUnsupported("an object does not use a version 2 object header")
        }
        FormatError::UnexpectedEof { .. } => Error::EditUnsupported("truncated object header"),
        err => Error::Format(err),
    })?;
    let region_start = addr
        .checked_add(parsed.len.to_u64())
        .ok_or(Error::EditUnsupported("truncated object header"))?;
    // The region is followed by a 4-byte checksum, which must also be present.
    let region_end = region_start
        .checked_add(parsed.chunk0_size)
        .filter(|e| e.checked_add(4).is_some_and(|end| end <= file_len))
        .ok_or(Error::EditUnsupported("truncated object header"))?;
    Ok((region_start, region_end, parsed.prefix))
}

/// A bounded version 2 header chunk and its full on-disk extent.
///
/// The buffer excludes the trailing checksum. The extent includes it for storage reclamation.
pub(crate) struct OhChunk {
    /// Absolute file address and full on-disk length of the chunk.
    pub(crate) span: (u64, u64),
    /// The chunk's bytes, from `span.0` through the end of its message region.
    buf: Vec<u8>,
    /// Offset of the first message within [`buf`](Self::buf).
    messages_start: usize,
    /// The prefix properties shared by every chunk of the header.
    props: ObjectHeaderPrefix,
}

impl OhChunk {
    /// Returns the chunk bytes and the offset at which its records start.
    ///
    /// The slice ends before the checksum.
    pub(crate) fn message_region(&self) -> (&[u8], usize) {
        (&self.buf, self.messages_start)
    }

    /// Returns the record layout parsed from the header's chunk 0 prefix.
    pub(crate) fn layout(&self) -> MessageRecordLayout {
        self.props.layout
    }

    /// Returns the prefix properties shared by every chunk of this header.
    pub(crate) fn props(&self) -> ObjectHeaderPrefix {
        self.props
    }
}

/// Read chunk 0 of the version 2 object header at `addr` out of `src`.
pub(super) fn read_oh_chunk0<S: Source + ?Sized>(src: &S, addr: u64) -> Result<OhChunk, Error> {
    let file_len = src.len();
    let window = file_len
        .saturating_sub(addr)
        .min(OBJECT_HEADER_PREFIX_MAX_LEN as u64)
        .to_usize()?;
    let prefix = src.read_metadata_at(addr, window)?;
    let (rs, re, props) = oh_region_at(&prefix, addr, file_len)?;
    // `re >= rs > addr`, so both differences are non-negative, and `oh_region_at`
    // has checked that the 4-byte checksum past `re` is present.
    let len = (re - addr).to_usize()?;
    Ok(OhChunk {
        span: (addr, len as u64 + 4),
        buf: src.read_metadata_at(addr, len)?,
        messages_start: (rs - addr).to_usize()?,
        props,
    })
}

/// Reads every chunk of a version 2 header, following continuation records.
///
/// Returns chunk 0 first. Both [`WriteEngine::gather_oh_messages`] and
/// [`WriteEngine::oh_chunk_spans`] use this traversal.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the header is not version 2, a chunk or record extends
/// past its source bounds, or the traversal exceeds [`MAX_OH_CHUNKS`]. Returns errors from
/// [`read_oh_chunk0`] and [`read_oh_continuation`].
pub(crate) fn read_oh_chunks<S: Source + ?Sized>(
    src: &S,
    addr: u64,
    base: AddressContext,
) -> Result<Vec<OhChunk>, Error> {
    let mut chunks = vec![read_oh_chunk0(src, addr)?];
    let mut i = 0;
    while i < chunks.len() {
        if chunks.len() > MAX_OH_CHUNKS {
            return Err(Error::EditUnsupported(
                "object header has too many continuation chunks",
            ));
        }
        // Collect this chunk's continuations before extending the list of chunks, so the
        // borrow of `chunks[i]` ends first.
        let mut found = Vec::new();
        let props = chunks[i].props();
        let layout = props.layout;
        let (region, mut p) = chunks[i].message_region();
        while let Some((msg_type, body, body_end)) = next_record(layout, region, p)? {
            if msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                found.push(read_oh_continuation(
                    src, region, body, body_end, base, props,
                )?);
            }
            p = body_end;
        }
        i += 1;
        chunks.extend(found);
    }
    Ok(chunks)
}

/// Reads the `OCHK` continuation block a continuation message points at.
///
/// `region[body..body_end]` is the continuation message's body: the block's
/// base-relative address followed by its length.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the message ends inside its fields, if the block runs
/// past the end of `src` or is shorter than its signature and checksum, or if the block does not
/// begin with `OCHK`. Returns [`Error::Format`] if the message stores the undefined address, if
/// adding the effective base to the address overflows, or if a read of `src` fails.
pub(super) fn read_oh_continuation<S: Source + ?Sized>(
    src: &S,
    region: &[u8],
    body: usize,
    body_end: usize,
    base: AddressContext,
    props: ObjectHeaderPrefix,
) -> Result<OhChunk, Error> {
    let continuation =
        match ObjectHeaderContinuation::parse(&region[body..body_end], OFFSET_SIZE, LENGTH_SIZE) {
            Ok(continuation) => continuation,
            Err(FormatError::UnexpectedEof { .. }) => {
                return Err(Error::EditUnsupported("malformed continuation message"));
            }
            Err(err) => return Err(Error::Format(err)),
        };
    let len = continuation.length();
    // The block address is stored relative to the base address. Shift it to an
    // absolute file offset before reading.
    let off = base.absolute(continuation.address())?;
    // An OCHK block is signature(4) + messages + checksum(4).
    let end = off
        .checked_add(len)
        .filter(|&e| e <= src.len() && len >= 8)
        .ok_or(Error::EditUnsupported("continuation block out of bounds"))?;
    let want = (end - off).to_usize()?;
    let mut buf = src.read_metadata_at(off, want)?;
    let messages = match hdf5_pure_format::__private::continuation_block_messages(&buf) {
        Ok(messages) => messages,
        Err(FormatError::InvalidObjectHeaderSignature) => {
            return Err(Error::EditUnsupported(
                "invalid continuation block signature",
            ));
        }
        Err(err) => return Err(Error::Format(err)),
    };
    // Trim the trailing checksum so the message walk stops at the last message.
    buf.truncate(messages.end);
    Ok(OhChunk {
        span: (off, len),
        buf,
        messages_start: messages.start,
        props,
    })
}

/// Returns the type and the body range of the message record at `msg_start` in `region`,
/// as [`MessageRecordLayout::next_message`] reads it.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the body of the record runs past the end
/// of `region`.
pub(crate) fn next_record(
    layout: MessageRecordLayout,
    region: &[u8],
    msg_start: usize,
) -> Result<Option<(MessageType, usize, usize)>, Error> {
    Ok(layout
        .next_message(region, msg_start)
        .map_err(record_error)?
        .map(|record| {
            (
                record.msg_type,
                record.body_range.start,
                record.body_range.end,
            )
        }))
}

/// Returns the editor's error for an error of [`MessageRecordLayout::next_message`].
///
/// A record whose body runs past the end of its region is reported as
/// [`Error::EditUnsupported`], and any other error as [`Error::Format`].
pub(super) fn record_error(err: FormatError) -> Error {
    match err {
        FormatError::UnexpectedEof { .. } => {
            Error::EditUnsupported("malformed object header message")
        }
        err => Error::Format(err),
    }
}

/// A version 2 message region and its header-prefix properties.
///
/// [`WriteEngine::gather_oh_messages`] collects records into this region, and
/// [`build_v2_object_header`] serializes it. The record layout and optional prefix fields are
/// retained when the bytes are replaced.
#[derive(Clone, Debug, Default)]
pub(crate) struct OhRegion {
    bytes: Vec<u8>,
    props: ObjectHeaderPrefix,
}

impl core::ops::Deref for OhRegion {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}

impl OhRegion {
    /// Creates a region with the supplied record bytes and prefix properties.
    pub(crate) fn new(bytes: Vec<u8>, props: ObjectHeaderPrefix) -> Self {
        Self { bytes, props }
    }

    /// Creates an empty region with the supplied prefix properties.
    pub(super) fn empty(props: ObjectHeaderPrefix) -> Self {
        Self::new(Vec::new(), props)
    }

    /// Creates a region with replacement bytes and the same prefix properties.
    pub(super) fn with_bytes(&self, bytes: Vec<u8>) -> Self {
        Self::new(bytes, self.props)
    }

    /// Returns the layout used to parse and serialize this region.
    pub(crate) fn layout(&self) -> MessageRecordLayout {
        self.props.layout
    }

    /// Returns the prefix properties used to rebuild the header.
    pub(crate) fn props(&self) -> ObjectHeaderPrefix {
        self.props
    }

    /// Returns the type and body range of the record at `msg_start`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the record extends past the region.
    pub(crate) fn next_message(
        &self,
        msg_start: usize,
    ) -> Result<Option<(MessageType, usize, usize)>, Error> {
        next_record(self.props.layout, &self.bytes, msg_start)
    }

    /// Selects the first record of `msg_type` for replacement, or returns `None` if absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the body of a record it reads runs past the end of
    /// the region.
    pub(super) fn splice_first(
        &self,
        msg_type: MessageType,
    ) -> Result<Option<RecordSplice<'_>>, Error> {
        RecordSplice::first(self.layout(), &self.bytes, msg_type).map_err(record_error)
    }

    /// Returns a record's attribute creation index, or `None` if it is not stored.
    pub(super) fn creation_index(&self, msg_start: usize) -> Option<u16> {
        self.props.layout.creation_index(&self.bytes, msg_start)
    }

    /// Appends records already encoded in this region's layout.
    pub(super) fn push_bytes(&mut self, record: &[u8]) {
        self.bytes.extend_from_slice(record);
    }

    /// Returns the record bytes for a patch that preserves record lengths.
    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Appends a message record with flags and creation index set to zero.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte type field or `body` is longer than
    /// [`OBJECT_HEADER_MESSAGE_MAX`], as [`MessageRecordLayout::record`] does.
    pub(super) fn push(&mut self, msg_type: MessageType, body: &[u8]) {
        let record = self.props.layout.record(msg_type, body);
        self.push_bytes(&record);
    }

    /// Appends a shared-message reference with the shared flag set.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte type field or `reference` is longer than
    /// [`OBJECT_HEADER_MESSAGE_MAX`], as [`MessageRecordLayout::record_with_creation_index`] does.
    pub(super) fn push_shared(&mut self, msg_type: MessageType, reference: &[u8]) {
        let record = self.props.layout.record_with_creation_index(
            msg_type,
            MessageFlags::SHARED,
            0,
            reference,
        );
        self.push_bytes(&record);
    }
}

/// Re-encode a shared-message reference in the modern form, for a message being
/// rewrapped from a version 1 object header into a version 2 one.
///
/// A version 1 header carries the oldest reference encoding: a symbol-table
/// entry, with the object-header address buried behind a local-heap address.
/// Copying those bytes into a version 2 header would put an encoding there that
/// no writer of such a header has produced since the format gained one. Parsing
/// and re-encoding gives the shape every current reader and writer uses.
///
/// Which shape that is depends on where the message lives. An object header is
/// named by the version 2 form this crate already writes for a committed
/// datatype. The shared-message heap has no encoding before version 3, so a heap
/// reference gets that one. Both name exactly what the version 1 reference named,
/// so the message keeps its single copy and its reference count is unchanged.
pub(super) fn modernize_shared_reference(
    body: &[u8],
    widths: FormatWidths,
) -> Result<Vec<u8>, Error> {
    let reference = crate::shared_message::parse_shared_ref_with_widths(body, widths)?;
    Ok(match reference.location {
        crate::shared_message::SharedLocation::ObjectHeader(addr) => {
            crate::shared_message::encode_committed_ref(addr, widths.offsets)
        }
        crate::shared_message::SharedLocation::SohmHeap(id) => {
            crate::shared_message::encode_sohm_ref(&id)
        }
    })
}

/// Builds a single-chunk version 2 header with normalized attribute information.
///
/// Calls [`super::ensure_attribute_info`] before serializing the prefix, records, and checksum.
/// Preserves the prefix's optional fields and updates existing modification and change times
/// when the system clock is available.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region. Propagates other
/// errors from [`super::ensure_attribute_info`].
pub(crate) fn build_v2_object_header(region: &OhRegion) -> Result<Vec<u8>, Error> {
    let mut owned = region.clone();
    super::ensure_attribute_info(&mut owned)?;
    Ok(build_v2_object_header_verbatim(&owned))
}

/// [`build_v2_object_header`] without the attribute-storage normalization, for
/// the region that function has already normalized.
///
/// **The header's optional prefix blocks are re-emitted, and its timestamps are
/// stamped.** A region carries whatever chunk 0's prefix declared
/// ([`ObjectHeaderPrefix`]), so a rewrite of a header from libhdf5,
/// h5py or netCDF-4 puts its four timestamps and any attribute phase-change
/// thresholds back where it found them, with the flag bits that announce them.
///
/// Of those, the **modification and change times are moved to now**: this
/// function runs once per rebuilt header, and every rebuild is a modification.
/// Access and birth times are the object's own history and are copied verbatim
/// ([`ObjectTimes::touched`](hdf5_pure_format::__private::ObjectTimes::touched) says
/// how that compares with `H5O_touch_oh`).
/// Under `no_std` there is no clock to read, so all four are preserved as they
/// were.
/// A header this crate creates from nothing omits timestamps.
pub(super) fn build_v2_object_header_verbatim(region: &OhRegion) -> Vec<u8> {
    let prefix = region.props();
    let stamped = ObjectHeaderPrefix {
        times: prefix.times.map(|times| match unix_time_now() {
            Some(now) => times.touched(now),
            None => times,
        }),
        ..prefix
    };
    stamped.encode_header(region)
}

/// Returns the current Unix time, saturated to the timestamp field's maximum.
///
/// Returns `None` if the system clock is before the Unix epoch.
#[cfg(feature = "std")]
pub(super) fn unix_time_now() -> Option<u32> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| u32::try_from(d.as_secs()).unwrap_or(u32::MAX))
}

/// Returns `None` when the system clock is unavailable.
#[cfg(not(feature = "std"))]
pub(super) fn unix_time_now() -> Option<u32> {
    None
}

/// The traversal limit for header chunks, including cyclic continuation chains.
const MAX_OH_CHUNKS: usize = 256;
