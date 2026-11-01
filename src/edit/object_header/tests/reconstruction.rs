use std::cell::Cell;

use test_util::checksum;
use test_util::object_header::v2::{self, Header, HeaderFlags, Timestamps};
use test_util::object_header::{Message, MessageFlags as WireFlags, MessageType as WireType};

use crate::address::DecodedStoredAddress;

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
