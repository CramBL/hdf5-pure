use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::cell::RefCell;
use core::ops::Deref;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::checksum;
use test_util::image::Image;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType as WireType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::object_header::v2::Timestamps;
use test_util::widths::Widths;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::metadata_source;
use crate::metadata_source::MetadataSource;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::width::FormatWidths;

use super::*;

#[rstest]
#[case::version_1(Version::One)]
#[case::version_2_plain(Version::Plain)]
#[case::version_2_tracked(Version::Tracked)]
fn chunk_observation_preserves_reads_records_and_complete_geometry(
    #[case] version: Version,
    #[values(2, 4, 8)] width: u8,
    #[values(0, 8_589_934_592)] bias: u64,
) {
    let fixture = Fixture::new(version, width);
    let source = CountedSource::new(&fixture.bytes, bias);
    let address = PhysicalAddress::new(bias + ROOT);
    let base = BaseAddress::new(bias + BASE);
    let widths = FormatWidths::from_sizes(width, width).unwrap();
    let keep = |kind, _: &[u8]| kind == MessageType::DATATYPE;
    let ordinary = ObjectHeader::parse_from_source_filtered_with_widths(
        &source,
        AccessMode::ReadWrite,
        address.get(),
        widths,
        base,
        MessageFilter::Only(&mut { keep }),
    )
    .unwrap();
    let reads = source.reads.take();
    assert_eq!(reads.len(), fixture.parts.len() + 2);
    let allocated_bytes = fixture
        .parts
        .iter()
        .map(|part| part.bytes.len())
        .sum::<usize>();
    let probe_bytes = match version {
        Version::One => 0,
        Version::Plain | Version::Tracked => {
            crate::object_header::OBJECT_HEADER_PREFIX_MAX_LEN + v2::SIGNATURE.len()
        }
    };
    assert_eq!(
        reads.iter().map(|(_, len)| *len).sum::<usize>(),
        allocated_bytes + probe_bytes
    );
    let peak = source.peak.get();
    assert_eq!(source.live.get(), 0);
    assert!(
        peak >= fixture
            .parts
            .iter()
            .map(|part| part.bytes.len())
            .max()
            .unwrap()
    );
    assert!(
        peak < fixture
            .parts
            .iter()
            .map(|part| part.bytes.len())
            .sum::<usize>()
    );
    source.peak.set(0);
    let mut seen = Vec::new();
    let observed = ObjectHeader::parse_from_source_with_chunk_observer(
        &source,
        AccessMode::ReadWrite,
        address,
        widths,
        base,
        MessageFilter::Only(&mut { keep }),
        &mut |chunk| {
            let part = &fixture.parts[version.order()[seen.len()]];
            assert_eq!(chunk.allocation().start().get(), bias + part.address);
            assert_eq!(chunk.allocation().length(), part.bytes.len().to_u64());
            assert_eq!(
                chunk.records().range().start().get(),
                bias + part.address + part.prefix.to_u64()
            );
            assert_eq!(chunk.records().bytes(), &part.records);
            assert_eq!(chunk.v2_layout(), version.layout());
            assert!(chunk.allocation().contains(chunk.records().range()));
            let read = source.reads.borrow().last().copied().unwrap();
            let offset = usize::try_from(chunk.records().range().start().get() - read.0).unwrap();
            assert_eq!(
                chunk.records().bytes().as_ptr(),
                source.pointer.get().wrapping_add(offset)
            );
            seen.push(chunk.allocation());
        },
    )
    .unwrap();
    assert_eq!(source.reads.take(), reads);
    assert_eq!(source.peak.get(), peak);
    assert_eq!(source.live.get(), 0);
    assert_eq!(seen.len(), 4);
    assert_eq!(observed.flags, fixture.flags);
    assert_eq!(
        (
            observed.access_time,
            observed.modification_time,
            observed.change_time,
            observed.birth_time
        ),
        (
            fixture.times.map(|times| times.access),
            fixture.times.map(|times| times.modification),
            fixture.times.map(|times| times.change),
            fixture.times.map(|times| times.birth)
        )
    );
    let actual = observed
        .messages
        .iter()
        .map(|message| message.data.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        version
            .order()
            .map(|index| fixture.parts[index].value.as_slice())
    );
    assert_eq!(
        actual,
        ordinary
            .messages
            .iter()
            .map(|message| message.data.as_slice())
            .collect::<Vec<_>>()
    );
    assert_eq!(observed.messages.capacity(), ordinary.messages.capacity());
}

#[rstest]
#[case::read_only(AccessMode::ReadOnly)]
#[case::read_write(AccessMode::ReadWrite)]
fn chunk_observation_preserves_rejection_after_an_earlier_chunk(#[case] mode: AccessMode) {
    let child = v2::continuation_chunk(
        &[Message::new(WireType::UNKNOWN, &[1]).with_flags(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)],
        v2::HeaderFlags::default(),
    );
    let root = v2::Header::new()
        .continuation(FIRST as usize, &child, Widths::EIGHT)
        .build();
    let mut image = Image::starting_with(&root);
    image.place(FIRST as usize, &child);
    let bytes = image.build();
    let mut observed = Vec::new();
    let err = ObjectHeader::parse_from_source_with_chunk_observer(
        bytes.as_slice(),
        mode,
        PhysicalAddress::new(0),
        FormatWidths::from_sizes(8, 8).unwrap(),
        BaseAddress::ZERO,
        MessageFilter::Only(&mut |_, _| false),
        &mut |chunk| observed.push(chunk.allocation()),
    )
    .unwrap_err();
    assert_eq!(err, FormatError::UnsupportedMessage(WireType::UNKNOWN.0));
    assert_eq!(
        observed
            .iter()
            .map(|range| (range.start().get(), range.length()))
            .collect::<Vec<_>>(),
        [(0, root.len().to_u64()), (FIRST, child.len().to_u64())]
    );
}

#[rstest]
#[case::version_1(Version::One)]
#[case::version_2_plain(Version::Plain)]
#[case::version_2_tracked(Version::Tracked)]
fn a_source_failure_preserves_only_preceding_chunk_observations(#[case] version: Version) {
    let mut fixture = Fixture::new(version, 8);
    let expected_end = fixture.bytes.len().to_u64();
    fixture.bytes.pop();
    let mut observed = Vec::new();
    let error = ObjectHeader::parse_from_source_with_chunk_observer(
        fixture.bytes.as_slice(),
        AccessMode::ReadOnly,
        PhysicalAddress::new(ROOT),
        FormatWidths::from_sizes(8, 8).unwrap(),
        BaseAddress::new(BASE),
        MessageFilter::All,
        &mut |chunk| observed.push(chunk.allocation().start().get()),
    )
    .unwrap_err();
    assert_eq!(
        error,
        FormatError::UnexpectedSourceEof {
            expected: expected_end,
            available: expected_end - 1,
        }
    );
    let expected = match version {
        Version::One => vec![ROOT, BASE + FIRST],
        Version::Plain | Version::Tracked => vec![ROOT, BASE + FIRST, BASE + SECOND],
    };
    assert_eq!(observed, expected);
}

#[test]
fn chunk_observation_preserves_checksum_feature_policy() {
    let mut child = v2::continuation_chunk(&[Message::nil(32)], v2::HeaderFlags::default());
    let last = child.len() - 1;
    child[last] ^= 1;
    let root = v2::Header::new()
        .continuation(FIRST as usize, &child, Widths::EIGHT)
        .build();
    let mut image = Image::starting_with(&root);
    image.place(FIRST as usize, &child);
    let bytes = image.build();
    let mut count = 0;
    let result = ObjectHeader::parse_from_source_with_chunk_observer(
        bytes.as_slice(),
        AccessMode::ReadOnly,
        PhysicalAddress::new(0),
        FormatWidths::from_sizes(8, 8).unwrap(),
        BaseAddress::ZERO,
        MessageFilter::All,
        &mut |_| count += 1,
    );
    #[cfg(feature = "checksum")]
    {
        let error = result.unwrap_err();
        let covered = child.len() - checksum::CHECKSUM;
        assert_eq!(
            error,
            FormatError::ChecksumMismatch {
                expected: u32::from_le_bytes(child[covered..].try_into().unwrap()),
                computed: checksum::lookup3(&child[..covered]),
            }
        );
        assert_eq!(count, 1);
    }
    #[cfg(not(feature = "checksum"))]
    {
        assert!(result.unwrap().messages.is_empty());
        assert_eq!(count, 2);
    }
}

#[derive(Clone, Copy)]
enum Version {
    One,
    Plain,
    Tracked,
}

impl Version {
    fn layout(self) -> Option<MessageRecordLayout> {
        match self {
            Self::One => None,
            Self::Plain => Some(MessageRecordLayout::PLAIN),
            Self::Tracked => Some(MessageRecordLayout::from_header_flags(
                v2::HeaderFlags::TRACKS_CREATION_ORDER.0
                    | v2::HeaderFlags::INDEXES_CREATION_ORDER.0,
            )),
        }
    }
    fn flags(self) -> v2::HeaderFlags {
        match self {
            Self::One | Self::Plain => v2::HeaderFlags::CHUNK_SIZE_WIDTH,
            Self::Tracked => {
                v2::HeaderFlags::CHUNK_SIZE_WIDTH
                    | v2::HeaderFlags::TRACKS_CREATION_ORDER
                    | v2::HeaderFlags::INDEXES_CREATION_ORDER
                    | v2::HeaderFlags::STORES_TIMES
                    | v2::HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE
            }
        }
    }
    fn order(self) -> [usize; 4] {
        match self {
            Self::One => [0, 1, 3, 2],
            Self::Plain | Self::Tracked => [0, 1, 2, 3],
        }
    }
    fn records(self, messages: &[Message]) -> Vec<u8> {
        match self {
            Self::One => v1::message_records(messages),
            Self::Plain | Self::Tracked => v2::message_records(messages, self.flags()),
        }
    }
}

struct Part {
    address: u64,
    bytes: Vec<u8>,
    records: Vec<u8>,
    prefix: usize,
    value: Vec<u8>,
}
struct Fixture {
    bytes: Vec<u8>,
    parts: Vec<Part>,
    flags: u8,
    times: Option<Timestamps>,
}

impl Fixture {
    fn new(version: Version, width: u8) -> Self {
        let times = match version {
            Version::Tracked => Some(Timestamps {
                access: 11,
                modification: 22,
                change: 33,
                birth: 44,
            }),
            Version::One | Version::Plain => None,
        };
        let flags = match version {
            Version::One => 0,
            Version::Plain | Version::Tracked => version.flags().0,
        };
        let widths = Widths {
            offset: usize::from(width),
            length: usize::from(width),
        };
        let make = |address, value: &[u8], mut continuations: Vec<Message>| {
            let mut value = value.to_vec();
            if matches!(version, Version::One) {
                value.resize(
                    value.len().div_ceil(V1_MESSAGE_ALIGNMENT) * V1_MESSAGE_ALIGNMENT,
                    0,
                );
            }
            let mut messages = vec![
                Message::new(WireType::DATATYPE, &value).with_creation_order(19),
                Message::nil(512),
                Message::new(WireType::UNKNOWN, &[0xA5; 512]),
            ];
            messages.append(&mut continuations);
            if matches!(version, Version::One) {
                for message in &mut messages {
                    message.data.resize(
                        message.data.len().div_ceil(V1_MESSAGE_ALIGNMENT) * V1_MESSAGE_ALIGNMENT,
                        0,
                    );
                }
            }
            let records = version.records(&messages);
            let bytes = match version {
                Version::One => {
                    if address == ROOT {
                        v1::Header::new().messages(messages).build()
                    } else {
                        records.clone()
                    }
                }
                Version::Plain | Version::Tracked => {
                    if address == ROOT {
                        let header = v2::Header::new().flags(version.flags()).messages(messages);
                        match times {
                            Some(times) => header.timestamps(times).build(),
                            None => header.build(),
                        }
                    } else {
                        v2::continuation_chunk(&messages, version.flags())
                    }
                }
            };
            let prefix = match version {
                Version::One => {
                    if address == ROOT {
                        v1::PREFIX
                    } else {
                        0
                    }
                }
                Version::Plain | Version::Tracked => {
                    bytes.len() - records.len() - checksum::CHECKSUM
                }
            };
            Part {
                address,
                bytes,
                records,
                prefix,
                value,
            }
        };
        let nested = make(BASE + NESTED, b"nested", vec![]);
        let first = make(
            BASE + FIRST,
            b"first",
            vec![Message::continuation(
                NESTED,
                nested.bytes.len().to_u64(),
                widths,
            )],
        );
        let second = make(BASE + SECOND, b"second", vec![]);
        let root = make(
            ROOT,
            b"root",
            vec![
                Message::continuation(FIRST, first.bytes.len().to_u64(), widths),
                Message::continuation(SECOND, second.bytes.len().to_u64(), widths),
            ],
        );
        let mut image = Image::starting_with(&[]);
        for part in [&root, &first, &second, &nested] {
            image.place(usize::try_from(part.address).unwrap(), &part.bytes);
        }
        Self {
            bytes: image.build(),
            parts: vec![root, first, second, nested],
            flags,
            times,
        }
    }
}

struct CountedSource<'a> {
    bytes: &'a [u8],
    bias: u64,
    reads: RefCell<Vec<(u64, usize)>>,
    live: Cell<usize>,
    peak: Cell<usize>,
    pointer: Cell<*const u8>,
}
impl<'a> CountedSource<'a> {
    fn new(bytes: &'a [u8], bias: u64) -> Self {
        Self {
            bytes,
            bias,
            reads: RefCell::default(),
            live: Cell::new(0),
            peak: Cell::new(0),
            pointer: Cell::new(core::ptr::null()),
        }
    }
    fn region(&self, offset: u64, len: usize) -> Result<&[u8], FormatError> {
        self.reads.borrow_mut().push((offset, len));
        let offset = offset
            .checked_sub(self.bias)
            .ok_or(FormatError::UnexpectedSourceEof {
                expected: offset,
                available: self.len(),
            })?;
        metadata_source::region(self.bytes, offset, len)
    }
}
impl MetadataSource for CountedSource<'_> {
    type Bytes<'a>
        = ReadBuffer<'a>
    where
        Self: 'a;
    fn len(&self) -> u64 {
        self.bias + self.bytes.len().to_u64()
    }
    fn read_metadata(&self, offset: u64, len: usize) -> Result<ReadBuffer<'_>, FormatError> {
        let bytes = self.region(offset, len)?.to_vec();
        self.live.set(self.live.get() + bytes.len());
        self.peak.set(self.peak.get().max(self.live.get()));
        self.pointer.set(bytes.as_ptr());
        Ok(ReadBuffer {
            bytes,
            live: &self.live,
        })
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        buf.copy_from_slice(self.region(offset, buf.len())?);
        Ok(())
    }
    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.region(offset, len)?.to_vec())
    }
}
struct ReadBuffer<'a> {
    bytes: Vec<u8>,
    live: &'a Cell<usize>,
}
impl Deref for ReadBuffer<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}
impl Drop for ReadBuffer<'_> {
    fn drop(&mut self) {
        self.live.set(self.live.get() - self.bytes.len());
    }
}

const BASE: u64 = 512;
const ROOT: u64 = 1_536;
const FIRST: u64 = 8_192;
const SECOND: u64 = 12_288;
const NESTED: u64 = 16_384;

// "Version 1 Data Object Header Prefix" of the format specification, version 4.0,
// aligns message data to eight bytes.
const V1_MESSAGE_ALIGNMENT: usize = 8;
