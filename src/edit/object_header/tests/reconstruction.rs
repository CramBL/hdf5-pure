use core::num::NonZeroUsize;
use std::cell::Cell;

use test_util::checksum;
use test_util::object_header::v2::{self, Header, HeaderFlags, Timestamps};
use test_util::object_header::{Message, MessageFlags as WireFlags, MessageType as WireType};
use test_util::superblock::v2 as superblock_bytes;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::temp;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::access_mode::AccessMode;
use crate::address::DecodedStoredAddress;
use crate::edit::OpeningExtension;
use crate::edit::context::FileContext;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::object_header::ObjectHeader;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::ReadSeekSource;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::SourceMetadata;
use crate::superblock;

use super::*;

#[rstest]
#[case::plain(HeaderFlags::default())]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER)]
fn compact_reconstruction_preserves_records_and_optional_prefixes(
    #[case] flags: HeaderFlags,
    #[values(false, true)] times: bool,
    #[values(false, true)] phase_change: bool,
    #[values(1, 4)] chunks: usize,
) {
    let fixture = Predecessor::new(flags, times, phase_change, chunks);
    let source = CountedSource::new(&fixture.bytes);
    let prepared = ExtensionReconstruction::read(&source, fixture.root, fixture.addresses).unwrap();
    assert_eq!(&prepared.region[..], &fixture.records);
    assert_eq!(prepared.region.props(), fixture.props);
    assert!(prepared.region.bytes.capacity() < fixture.records.len() * 2);
    assert_eq!(source.calls.get(), u64::try_from(chunks + 1).unwrap());
    assert_eq!(source.bytes.get(), fixture.read_bytes);
    let reads = (source.calls.get(), source.bytes.get());
    let expected_records = |body: &[u8]| {
        v2::message_records(
            &[
                Message::new(WireType::FILE_SPACE_INFO, body)
                    .with_flags(WireFlags::FORBID_SHARING)
                    .with_creation_order(7),
                Message::new(WireType::UNKNOWN, &[0xA5; 13])
                    .with_flags(WireFlags::MARK_IF_UNKNOWN)
                    .with_creation_order(11),
            ],
            flags,
        )
        .repeat(chunks)
    };
    let placeholder = file_space_info::persistent_placeholder(
        FileSpaceStrategy::Page,
        7,
        FileSpacePageSize::DEFAULT,
    );
    let body =
        hdf5_pure_format::__private::serialize_file_space_info_v1(WIDTHS, &placeholder).unwrap();
    let replacement = prepared.rewrite(WIDTHS, &placeholder).unwrap();
    assert_eq!(&replacement[..], &expected_records(&body));
    assert_eq!(replacement.props(), fixture.props);
    let final_info = file_space_info::persistent_managers(
        FileSpaceStrategy::Page,
        9,
        FileSpacePageSize::try_from(8_192).unwrap(),
        core::array::from_fn(|index| {
            DecodedStoredAddress::Defined(StoredAddress::new(
                32_768 + u64::try_from(index).unwrap() * 4_096,
            ))
        }),
        StoredAddress::new(1_048_576),
    );
    let final_body =
        hdf5_pure_format::__private::serialize_file_space_info_v1(WIDTHS, &final_info).unwrap();
    assert_ne!(body, final_body);
    let replacement = prepared.rewrite(WIDTHS, &final_info).unwrap();
    assert_eq!(&replacement[..], &expected_records(&final_body));
    assert_eq!(replacement.props(), fixture.props);
    let allocation = Extent::new(StoredAddress::new(1_024), 262_144).unwrap();
    let encoded = build_v2_object_header_allocation(&replacement, WIDTHS, allocation).unwrap();
    assert_eq!(u64::try_from(encoded.len()).unwrap(), allocation.len());
    assert_eq!((source.calls.get(), source.bytes.get()), reads);
    let raw = WriteEngine::gather_oh_messages(&source, fixture.root, fixture.addresses).unwrap();
    assert!(raw.len() >= chunks * 60_000);
    assert_eq!(&raw.without_nil().unwrap()[..], &fixture.records);
    let collected = read_oh_chunks(&source, fixture.root, fixture.addresses).unwrap();
    assert_eq!(collected.len(), chunks);
    if chunks == 1 {
        assert_eq!(collected.capacity(), 1);
    }
}

#[rstest]
#[case::plain(HeaderFlags::default())]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER)]
fn compact_reconstruction_checks_records_after_nil_and_file_space_info(#[case] flags: HeaderFlags) {
    let mut fixture = Predecessor::new(flags, false, false, 4);
    let record = v2::message_record(&Message::new(WireType::UNKNOWN, &[0xA5; 13]), flags);
    let start = fixture.bytes.len() - checksum::CHECKSUM - record.len();
    fixture.bytes[start + 1..start + 3].copy_from_slice(&14u16.to_le_bytes());
    let source = BytesSource::new(&fixture.bytes);
    let err = ExtensionReconstruction::read(&source, fixture.root, fixture.addresses)
        .err()
        .unwrap();
    let Error::EditUnsupported(reason) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(reason, "malformed object header message");
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::four_chunks(4)]
#[case::sixty_four_chunks(64)]
fn compact_reconstruction_bounds_retained_and_peak_bytes(#[case] chunks: usize) {
    let fixture = Predecessor::new(HeaderFlags::default(), false, false, chunks);
    let source = CountedSource::new(&fixture.bytes);
    let name = format!("compact_extension_{chunks}");
    let (prepared, compact) = allocation::measure(&name, || {
        ExtensionReconstruction::read(&source, fixture.root, fixture.addresses).unwrap()
    });
    assert_eq!(&prepared.region[..], &fixture.records);
    let name = format!("raw_extension_control_{chunks}");
    let (raw, control) = allocation::measure(&name, || {
        WriteEngine::gather_oh_messages(&source, fixture.root, fixture.addresses).unwrap()
    });
    assert!(control.bytes >= fixture.read_bytes, "{control}");
    assert!(
        control.live_bytes >= u64::try_from(raw.len()).unwrap(),
        "{control}"
    );
    assert!(compact.bytes >= fixture.read_bytes, "{compact}");
    assert!(
        compact.live_bytes >= u64::try_from(fixture.records.len()).unwrap(),
        "{compact}"
    );
    assert!(
        compact.live_bytes < u64::try_from(fixture.records.len() * 2).unwrap(),
        "{compact}"
    );
    assert!(compact.peak_bytes < 80_000, "{compact}");
    assert!(
        compact.bytes < fixture.read_bytes + u64::try_from(chunks * 1_024).unwrap(),
        "{compact}"
    );
    assert!(
        compact.live_bytes * 100 < control.live_bytes,
        "{compact} versus {control}"
    );
    eprintln!(
        "{chunks} chunks: source calls={}, bytes={}, retained record bytes={}, capacity={}",
        source.calls.get() / 2,
        source.bytes.get() / 2,
        prepared.region.len(),
        prepared.region.bytes.capacity()
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::one_chunk(1)]
#[case::four_chunks(4)]
#[case::sixty_four_chunks(64)]
fn opening_extension_reuses_one_source_pass_and_retains_only_facts(#[case] chunks: usize) {
    let mut fixture = Predecessor::new(HeaderFlags::default(), true, true, chunks);
    let context = fixture.opening_context();
    let path = temp::temp_path("opening-extension-cost.h5");
    std::fs::write(&path, &fixture.bytes).unwrap();
    let source = CountedDisk {
        inner: ReadSeekSource::new(std::fs::File::open(&path).unwrap()).unwrap(),
        calls: Cell::new(0),
        bytes: Cell::new(0),
    };
    let (facts, measured) = allocation::measure(&format!("opening_extension_{chunks}"), || {
        OpeningExtension::read(&source, &context).unwrap()
    });
    let calls = source.calls.get();
    let bytes = source.bytes.get();
    assert_eq!(calls, u64::try_from(chunks + 2).unwrap());
    assert!(bytes >= u64::try_from(chunks * 60_000).unwrap());
    let (control, previous) = allocation::measure(&format!("opening_control_{chunks}"), || {
        previous_opening_extension(&source, &context)
    });
    assert_eq!(facts.info, control.info);
    let (address, info) = facts.info.unwrap();
    assert_eq!(address, fixture.root);
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist);
    assert_eq!(info.threshold, 1);
    assert_eq!(
        info.manager_addrs,
        vec![DecodedStoredAddress::Undefined; info_bytes::VERSION_0_MANAGERS]
    );
    let allocations = facts.allocations.unwrap();
    assert_eq!(allocations.len(), chunks);
    assert_eq!(allocations[0].start().get(), fixture.root);
    assert_eq!(allocations, control.allocations.unwrap());
    assert_eq!(source.calls.get() - calls, 2 * calls - 1);
    assert_eq!(
        source.bytes.get() - bytes,
        2 * bytes - u64::try_from(v2::SIGNATURE.len() + checksum::CHECKSUM).unwrap()
    );
    assert!(previous.peak_bytes >= u64::try_from(chunks * 60_000).unwrap());
    assert!(
        measured.live_bytes >= u64::try_from(chunks * 16).unwrap(),
        "{measured}"
    );
    assert!(
        measured.live_bytes <= u64::try_from(chunks * 64 + 128).unwrap(),
        "{measured}"
    );
    assert!(measured.peak_bytes < 140_000, "{measured}");
    assert!(
        measured.bytes < bytes + u64::try_from(chunks * 1_024).unwrap(),
        "{measured}"
    );
    eprintln!(
        "{chunks} chunks: opening {measured}; previous {previous}; physical calls={calls}, bytes={bytes}; previous calls={}, bytes={}",
        source.calls.get() - calls,
        source.bytes.get() - bytes
    );
}

#[rstest]
#[case::last_supported(256, false)]
#[case::first_unsupported(257, true)]
fn opening_extension_retains_the_editor_chunk_limit(#[case] chunks: usize, #[case] rejected: bool) {
    let mut fixture = Predecessor::new(HeaderFlags::default(), false, false, chunks);
    let context = fixture.opening_context();
    let source = BytesSource::new(&fixture.bytes);
    let facts = OpeningExtension::read(&source, &context).unwrap();
    assert_eq!(facts.info.unwrap().1.threshold, 1);
    if rejected {
        let error = facts.allocations.unwrap_err();
        let Error::EditUnsupported(reason) = error else {
            panic!("expected EditUnsupported, got {error:?}");
        };
        assert_eq!(reason, "object header has too many continuation chunks");
    } else {
        let allocations = facts.allocations.unwrap();
        assert_eq!(allocations.len(), chunks);
        assert_eq!(allocations[0].start().get(), fixture.root);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Debug)]
enum SkippedFacts {
    Absent,
    NonPersistent,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::absent(SkippedFacts::Absent)]
#[case::non_persistent(SkippedFacts::NonPersistent)]
fn opening_without_retirement_facts_skips_owned_range_storage(
    #[case] case: SkippedFacts,
    #[values(1, 2, 8)] chunks: usize,
) {
    let message = match case {
        SkippedFacts::Absent => None,
        SkippedFacts::NonPersistent => Some(Message::new(
            WireType::FILE_SPACE_INFO,
            &info_bytes::non_persistent_version_1(Widths::EIGHT, info_bytes::PAGE, 9, 4_096),
        )),
    };
    let fixture = OpeningFixture::new(NonZeroUsize::new(chunks).unwrap(), message);
    let source = CountedSource::new(&fixture.bytes);
    let context = &fixture.context;
    let (facts, measured) = allocation::measure(&format!("opening_skip_{case:?}_{chunks}"), || {
        OpeningExtension::read(&source, context).unwrap()
    });
    let (previous, control) =
        allocation::measure(&format!("opening_skip_control_{case:?}_{chunks}"), || {
            previous_opening_extension(&source, context)
        });
    assert_eq!(source.calls.get(), 2 * u64::try_from(chunks + 2).unwrap());
    assert_eq!(facts.info, previous.info);
    match case {
        SkippedFacts::Absent => assert_eq!(facts.info, None),
        SkippedFacts::NonPersistent => {
            let (at, info) = facts.info.unwrap();
            assert_eq!(at, fixture.ranges[0].start().get());
            assert!(!info.persist);
            assert_eq!(info.strategy, FileSpaceStrategy::Page);
            assert_eq!(info.threshold, 9);
            assert_eq!(info.page_size.get(), 4_096);
        }
    }
    let ranges = facts.allocations.unwrap();
    assert!(ranges.is_empty());
    assert_eq!(ranges.capacity(), 0);
    assert!(previous.allocations.unwrap().is_empty());
    assert!(control.blocks >= 3, "{control}");
    assert!(
        control.bytes >= u64::try_from(chunks * 128).unwrap(),
        "{control}"
    );
    assert_eq!(measured.live_bytes, 0, "{measured}");
    assert!(
        measured.blocks <= control.blocks,
        "{measured} versus {control}"
    );
    assert!(
        measured.bytes <= control.bytes,
        "{measured} versus {control}"
    );
    if matches!(case, SkippedFacts::Absent) {
        assert_eq!(measured.blocks, control.blocks);
        assert_eq!(measured.bytes, control.bytes);
    }
    eprintln!(
        "{case:?}, {chunks} chunks: inline scratch size={}, candidate size={}",
        core::mem::size_of::<ExtensionRanges>(),
        core::mem::size_of::<ExtensionAllocations>()
    );
}

#[rstest]
#[case::two_chunks(2)]
#[case::eight_chunks(8)]
fn a_late_persisting_message_keeps_all_preceding_allocation_ranges(#[case] chunks: usize) {
    let message = Message::new(
        WireType::FILE_SPACE_INFO,
        &info_bytes::version_0(
            Widths::EIGHT,
            info_bytes::ALL_PERSIST,
            7,
            &[u64::MAX; info_bytes::VERSION_0_MANAGERS],
        ),
    );
    let fixture = OpeningFixture::new(NonZeroUsize::new(chunks).unwrap(), Some(message));
    let source = CountedSource::new(&fixture.bytes);
    let facts = OpeningExtension::read(&source, &fixture.context).unwrap();
    let (address, info) = facts.info.unwrap();
    assert_eq!(address, fixture.ranges[0].start().get());
    assert!(info.persist);
    assert_eq!(info.threshold, 7);
    assert_eq!(facts.allocations.unwrap(), fixture.ranges);
    assert_eq!(source.calls.get(), u64::try_from(chunks + 2).unwrap());
}

struct OpeningFixture {
    bytes: Vec<u8>,
    context: FileContext,
    ranges: Vec<FileRange>,
}

impl OpeningFixture {
    fn new(chunks: NonZeroUsize, message: Option<Message>) -> Self {
        let chunks = chunks.get();
        let flags = HeaderFlags::CHUNK_SIZE_WIDTH;
        let mut header = Header::new().flags(flags).message(Message::nil(128));
        if chunks == 1
            && let Some(message) = &message
        {
            header = header.message(message.clone());
        }
        let mut pieces = Vec::new();
        for index in 1..chunks {
            let mut messages = vec![Message::nil(128)];
            if index == chunks - 1
                && let Some(message) = &message
            {
                messages.push(message.clone());
            }
            let piece = v2::continuation_chunk(&messages, flags);
            let at = 8_192 + (index - 1) * 1_024;
            header = header.continuation(at, &piece, Widths::EIGHT);
            pieces.push((at, piece));
        }
        let header = header.build();
        let address = 128;
        let end = pieces
            .last()
            .map_or(address + header.len(), |(at, piece)| at + piece.len());
        let mut bytes = superblock_bytes::Superblock::new(Widths::EIGHT)
            .extension_address(u64::try_from(address).unwrap())
            .eof_address(u64::try_from(end).unwrap())
            .build();
        bytes.resize(end, 0);
        bytes[address..address + header.len()].copy_from_slice(&header);
        let mut ranges = vec![
            FileRange::new(
                PhysicalAddress::new(u64::try_from(address).unwrap()),
                u64::try_from(header.len()).unwrap(),
            )
            .unwrap(),
        ];
        for (at, piece) in pieces {
            bytes[at..at + piece.len()].copy_from_slice(&piece);
            ranges.push(
                FileRange::new(
                    PhysicalAddress::new(u64::try_from(at).unwrap()),
                    u64::try_from(piece.len()).unwrap(),
                )
                .unwrap(),
            );
        }
        let context = FileContext::new(superblock::read(&bytes[..]).unwrap()).unwrap();
        Self {
            bytes,
            context,
            ranges,
        }
    }
}

/// Prepares equivalent facts through separate message and allocation traversals.
///
/// Returns decoded settings and ranges after releasing the temporary raw chunks.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn previous_opening_extension(source: &impl Source, context: &FileContext) -> OpeningExtension {
    let address = context
        .addresses()
        .resolve(context.superblock().superblock_extension_address.unwrap())
        .unwrap()
        .get();
    let header = ObjectHeader::parse_from_source_with_widths(
        &SourceMetadata(source),
        AccessMode::ReadWrite,
        address,
        context.widths(),
        context.addresses().effective_base(),
    )
    .unwrap();
    let info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::FILE_SPACE_INFO)
        .map(|message| {
            hdf5_pure_format::__private::parse_file_space_info(context.widths(), &message.data)
                .unwrap()
        });
    drop(header);
    let allocations = if info.as_ref().is_some_and(|info| info.persist) {
        read_oh_chunks(source, address, context.addresses())
            .map(|chunks| chunks.into_iter().map(|chunk| chunk.span).collect())
    } else {
        Ok(Vec::new())
    };
    OpeningExtension {
        info: info.map(|info| (address, info)),
        allocations,
        shared_table: Ok(None),
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct CountedDisk {
    inner: ReadSeekSource<std::fs::File>,
    calls: Cell<u64>,
    bytes: Cell<u64>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Source for CountedDisk {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.calls.set(self.calls.get() + 1);
        self.bytes
            .set(self.bytes.get() + u64::try_from(bytes.len()).unwrap());
        self.inner.read_at(offset, bytes)
    }
}

struct Predecessor {
    bytes: Vec<u8>,
    root: u64,
    addresses: AddressContext,
    records: Vec<u8>,
    props: ObjectHeaderPrefix,
    read_bytes: u64,
}

impl Predecessor {
    fn new(flags: HeaderFlags, times: bool, phase_change: bool, chunks: usize) -> Self {
        let base = 4_096;
        let root = 8_192;
        let flags = flags | HeaderFlags::CHUNK_SIZE_WIDTH;
        let flags = if phase_change {
            flags | HeaderFlags::STORES_ATTRIBUTE_PHASE_CHANGE
        } else {
            flags
        };
        let flags = if times {
            flags | HeaderFlags::STORES_TIMES
        } else {
            flags
        };
        let info = Message::new(
            WireType::FILE_SPACE_INFO,
            &info_bytes::version_0(
                Widths::EIGHT,
                info_bytes::ALL_PERSIST,
                1,
                &[u64::MAX; info_bytes::VERSION_0_MANAGERS],
            ),
        )
        .with_flags(WireFlags::FORBID_SHARING)
        .with_creation_order(7);
        let unrelated = Message::new(WireType::UNKNOWN, &[0xA5; 13])
            .with_flags(WireFlags::MARK_IF_UNKNOWN)
            .with_creation_order(11);
        let mut records = v2::message_records(&[info.clone(), unrelated.clone()], flags);
        let mut pieces = Vec::new();
        let mut at = 131_072;
        let nil = Message::nil(60_000);
        let mut header = Header::new()
            .flags(flags)
            .message(info.clone())
            .message(nil.clone())
            .message(unrelated.clone());
        if times {
            header = header.timestamps(Timestamps {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4,
            });
        }
        for _ in 1..chunks {
            let piece =
                v2::continuation_chunk(&[nil.clone(), info.clone(), unrelated.clone()], flags);
            header = header.continuation(at - base, &piece, Widths::EIGHT);
            records.extend_from_slice(&v2::message_records(
                &[info.clone(), unrelated.clone()],
                flags,
            ));
            let end = at + piece.len();
            pieces.push((at, piece));
            at = end;
        }
        let header = header.build();
        let parsed = ObjectHeaderPrefix::parse(&header).unwrap();
        let props = parsed.prefix;
        let read_bytes = u64::try_from(
            OBJECT_HEADER_PREFIX_MAX_LEN + header.len() - checksum::CHECKSUM
                + pieces.iter().map(|(_, bytes)| bytes.len()).sum::<usize>(),
        )
        .unwrap();
        let mut bytes = vec![0; at.max(root + header.len())];
        bytes[root..root + header.len()].copy_from_slice(&header);
        for (at, piece) in pieces {
            bytes[at..at + piece.len()].copy_from_slice(&piece);
        }
        Self {
            bytes,
            root: u64::try_from(root).unwrap(),
            addresses: AddressContext::from_effective_base(BaseAddress::new(
                u64::try_from(base).unwrap(),
            )),
            records,
            props,
            read_bytes,
        }
    }
    fn opening_context(&mut self) -> FileContext {
        let block = superblock_bytes::Superblock::new(Widths::EIGHT)
            .base_address(4_096)
            .extension_address(self.root - 4_096)
            .eof_address(u64::try_from(self.bytes.len()).unwrap())
            .build();
        self.bytes[4_096..4_096 + block.len()].copy_from_slice(&block);
        FileContext::new(superblock::read(&self.bytes[..]).unwrap()).unwrap()
    }
}

struct CountedSource<'a> {
    inner: BytesSource<&'a [u8]>,
    calls: Cell<u64>,
    bytes: Cell<u64>,
}

impl<'a> CountedSource<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            inner: BytesSource::new(bytes),
            calls: Cell::new(0),
            bytes: Cell::new(0),
        }
    }
}

impl Source for CountedSource<'_> {
    fn len(&self) -> u64 {
        self.inner.len()
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.calls.set(self.calls.get() + 1);
        self.bytes
            .set(self.bytes.get() + u64::try_from(bytes.len()).unwrap());
        self.inner.read_at(offset, bytes)
    }
}
