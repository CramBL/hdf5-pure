#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::io;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::io::Read;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::io::Seek;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_format::__private::SohmIndexKind;

use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags as WireFlags;
use test_util::object_header::MessageType as WireType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::sohm;
use test_util::temp;
use test_util::widths::Widths;

use crate::address::DecodedStoredAddress;
use crate::address::StoredAddressExt;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::sohm::SharedMessageTableMessage;
use crate::sohm::SohmIndexHeader;
use crate::sohm::SohmTable;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::ValidatedSource;

use super::*;

#[derive(Clone, Copy, Debug)]
enum HeaderVersion {
    NestedOne(usize),
    One,
    Two,
}

#[rstest]
#[case::v1(HeaderVersion::One, true)]
#[case::v2(HeaderVersion::Two, true)]
#[case::v1_nested_limit(HeaderVersion::NestedOne(32), true)]
#[case::v1_nested_over_limit(HeaderVersion::NestedOne(33), false)]
fn resident_opening_preserves_complete_extension_values(
    #[case] version: HeaderVersion,
    #[case] readable: bool,
) {
    let fixture = Fixture::new(version, 2);
    let file = File::from_bytes(fixture.bytes.clone()).unwrap();
    if readable {
        assert_file(&file, &fixture.info);
    } else {
        assert!(file.inner.sohm_table.is_none());
        let error = file.file_space_info().unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::NestingDepthExceeded)),
            "{error:?}"
        );
        assert_eq!(
            file.dataset("keep").unwrap().read_i32().unwrap(),
            [11, 22, 33, 44]
        );
    }
    assert_eq!(file.as_bytes(), fixture.bytes);
    file.close().unwrap();
}

#[rstest]
#[case::at_limit(31, true)]
#[case::over_limit(32, false)]
fn streaming_opening_preserves_v1_continuation_depth(#[case] depth: usize, #[case] readable: bool) {
    let fixture = Fixture::new(HeaderVersion::NestedOne(depth), 1);
    let file = File::from_source_with_options(
        BytesSource::new(fixture.bytes),
        properties(MemoryStrategy::Bounded),
    )
    .unwrap();
    if readable {
        assert_file(&file, &fixture.info);
    } else {
        assert!(file.inner.sohm_table.is_none());
        let error = file.file_space_info().unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::NestingDepthExceeded)),
            "{error:?}"
        );
        assert_eq!(
            file.dataset("keep").unwrap().read_i32().unwrap(),
            [11, 22, 33, 44]
        );
    }
    file.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::v1(HeaderVersion::One)]
#[case::v2(HeaderVersion::Two)]
fn a_complete_resident_open_retains_only_selected_extension_facts(#[case] version: HeaderVersion) {
    let fixture = Fixture::new(version, 8);
    let properties = properties(MemoryStrategy::Mirrored);
    let (current, measured) =
        allocation::measure(&format!("complete_resident_{version:?}"), || {
            File::from_bytes_with_options(fixture.bytes.clone(), properties).unwrap()
        });
    let (prior, control) = allocation::measure(&format!("prior_resident_{version:?}"), || {
        PreviousOpening::from_bytes(fixture.bytes.clone(), properties)
    });
    assert!(control.blocks > 0, "{control}");
    assert!(
        measured.blocks < control.blocks,
        "{measured} versus {control}"
    );
    assert!(
        measured.bytes < control.bytes,
        "{measured} versus {control}"
    );
    assert!(
        measured.live_bytes < control.live_bytes,
        "{measured} versus {control}"
    );
    eprintln!("{version:?}: complete resident opening {measured}; prior {control}");
    assert_file(&current, &fixture.info);
    assert_eq!(current.as_bytes(), fixture.bytes);
    assert_eq!(prior.cached_info.as_ref(), Some(&fixture.info));
    let prior = prior.into_file();
    assert_file(&prior, &fixture.info);
    assert_eq!(prior.as_bytes(), fixture.bytes);
    current.close().unwrap();
    prior.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::v1(HeaderVersion::One)]
#[case::v2(HeaderVersion::Two)]
fn a_complete_source_open_reads_the_extension_once(
    #[case] version: HeaderVersion,
    #[values(1, 8)] chunks: usize,
) {
    let fixture = Fixture::new(version, chunks);
    let current_path = temp::temp_path("shared-open.h5");
    let prior_path = temp::temp_path("prior-open.h5");
    std::fs::write(&current_path, &fixture.bytes).unwrap();
    std::fs::write(&prior_path, &fixture.bytes).unwrap();
    let current_counts = Arc::new(Reads::default());
    let prior_counts = Arc::new(Reads::default());
    let properties = properties(MemoryStrategy::Bounded);
    let (current, measured) =
        allocation::measure(&format!("complete_source_{version:?}_{chunks}"), || {
            File::from_source_with_options(
                DiskSource::new(&current_path, Arc::clone(&current_counts)),
                properties,
            )
            .unwrap()
        });
    let (prior, control) =
        allocation::measure(&format!("prior_source_{version:?}_{chunks}"), || {
            let source: Box<dyn Source + Send + Sync> = Box::new(ValidatedSource::new(
                DiskSource::new(&prior_path, Arc::clone(&prior_counts)),
            ));
            let (superblock, widths, opened) =
                FileInner::parse_superblock(source.as_ref()).unwrap();
            opened.require_complete(source.len()).unwrap();
            file_lock::check_status_flags(
                &superblock,
                OpenIntent::Read(properties.write_mark_policy),
                OpenTarget::Source,
            )
            .unwrap();
            let empty = OpeningMetadata::from_message(
                &OpenedSource::for_test(source.as_ref(), opened.addresses()),
                Ok(None),
                widths,
            );
            PreviousOpening::new(FileInner::from_parts(
                Backend::Streaming(source),
                AccessMode::ReadOnly,
                superblock,
                widths,
                opened,
                ReaderOpening {
                    handle: None,
                    extension: Some(empty),
                },
                properties,
            ))
        });
    assert_eq!(
        prior_counts.calls.load(Ordering::Relaxed) - current_counts.calls.load(Ordering::Relaxed),
        u64::try_from(chunks + 2).unwrap()
    );
    assert!(
        prior_counts.bytes.load(Ordering::Relaxed)
            >= current_counts.bytes.load(Ordering::Relaxed)
                + u64::try_from(chunks * 60_000).unwrap()
    );
    assert!(control.blocks > 0, "{control}");
    assert!(
        measured.bytes < control.bytes,
        "{measured} versus {control}"
    );
    assert!(
        measured.live_bytes < control.live_bytes,
        "{measured} versus {control}"
    );
    assert_eq!(
        current_counts.backend_bytes.load(Ordering::Relaxed),
        current_counts.bytes.load(Ordering::Relaxed)
    );
    assert_eq!(
        prior_counts.backend_bytes.load(Ordering::Relaxed),
        prior_counts.bytes.load(Ordering::Relaxed)
    );
    assert!(
        current_counts.backend_calls.load(Ordering::Relaxed)
            >= current_counts.calls.load(Ordering::Relaxed)
    );
    assert!(
        prior_counts.backend_calls.load(Ordering::Relaxed)
            >= prior_counts.calls.load(Ordering::Relaxed)
    );
    report(
        "source",
        version,
        chunks,
        &measured,
        &control,
        &current_counts,
        &prior_counts,
    );
    assert_file(&current, &fixture.info);
    assert_eq!(prior.cached_info.as_ref(), Some(&fixture.info));
    let prior = prior.into_file();
    assert_file(&prior, &fixture.info);
    current.close().unwrap();
    prior.close().unwrap();
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::bounded(MemoryStrategy::Bounded)]
#[case::mirrored(MemoryStrategy::Mirrored)]
fn a_complete_writable_open_reuses_prepared_extension_facts(
    #[case] memory: MemoryStrategy,
    #[values(1, 8)] chunks: usize,
) {
    let fixture = Fixture::new(HeaderVersion::Two, chunks);
    let current_path = temp::temp_path("shared-writer-open.h5");
    let prior_path = temp::temp_path("prior-writer-open.h5");
    std::fs::write(&current_path, &fixture.bytes).unwrap();
    std::fs::write(&prior_path, &fixture.bytes).unwrap();
    let properties = properties(memory);
    let (current, measured) =
        allocation::measure(&format!("complete_writer_{memory:?}_{chunks}"), || {
            File::open_rw_with_options(&current_path, properties).unwrap()
        });
    let (prior, control) =
        allocation::measure(&format!("prior_writer_{memory:?}_{chunks}"), || {
            let mut session = WriteEngine::open_rw_with_strategy(
                &prior_path,
                properties.metadata_cache,
                properties.locking,
                memory,
                properties.persisted_free_space_policy,
            )
            .unwrap();
            session.set_libver_bounds(properties.libver_bounds).unwrap();
            session.set_sync_policy(properties.sync_policy);
            session
                .set_page_buffer_size(properties.page_buffer_size)
                .unwrap();
            let superblock = session.superblock().clone();
            let widths = session.widths();
            let opened = OpenedImageContext::reconcile(
                superblock.base_address,
                superblock.eof_address,
                session.addresses().effective_base().get(),
            )
            .unwrap();
            let empty = OpeningMetadata::from_message(&session.opened_source(), Ok(None), widths);
            PreviousOpening::new(FileInner::from_parts(
                Backend::Edit(Box::new(Mutex::new(session))),
                AccessMode::ReadWrite,
                superblock,
                widths,
                opened,
                ReaderOpening {
                    handle: None,
                    extension: Some(empty),
                },
                properties,
            ))
        });
    assert!(control.blocks > 0, "{control}");
    assert!(
        measured.blocks < control.blocks,
        "{measured} versus {control}"
    );
    assert!(
        measured.bytes < control.bytes,
        "{measured} versus {control}"
    );
    assert!(
        measured.live_bytes < control.live_bytes,
        "{measured} versus {control}"
    );
    eprintln!("{memory:?}, {chunks} chunks: complete opening {measured}; prior {control}");
    assert_eq!(current.disk_bytes().unwrap(), fixture.bytes);
    assert_file(&current, &fixture.info);
    assert_eq!(prior.cached_info.as_ref(), Some(&fixture.info));
    let prior = prior.into_file();
    assert_eq!(prior.disk_bytes().unwrap(), fixture.bytes);
    assert_file(&prior, &fixture.info);
    current.close().unwrap();
    prior.close().unwrap();
}

#[rstest]
#[case::unchanged(0)]
#[case::source_written(4_096)]
fn reader_construction_reloads_after_an_opening_write(#[case] page_buffer: usize) {
    let fixture = Fixture::new(HeaderVersion::Two, 8);
    let path = temp::temp_path("opening-write-phase.h5");
    std::fs::write(&path, &fixture.bytes).unwrap();
    let bytes = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicU64::new(0));
    let opening =
        WriteEngine::open_prepared_counting(&path, Arc::clone(&bytes), Arc::clone(&calls)).unwrap();
    bytes.store(0, Ordering::Relaxed);
    calls.store(0, Ordering::Relaxed);
    let file = File {
        inner: Arc::new(
            FileInner::from_opening_session(
                opening,
                properties(MemoryStrategy::Bounded).with_page_buffer_size(page_buffer),
            )
            .unwrap(),
        ),
    };
    let observed = (calls.load(Ordering::Relaxed), bytes.load(Ordering::Relaxed));
    if page_buffer == 0 {
        assert_eq!(observed, (1, u64::try_from(fixture.table_len).unwrap()));
    } else {
        assert!(observed.0 >= 11, "{observed:?}");
        assert!(observed.1 >= 8 * 60_000, "{observed:?}");
    }
    assert_file(&file, &fixture.info);
    file.close().unwrap();
}

#[test]
fn public_file_space_queries_observe_publication_after_opening() {
    let path = temp::temp_path("fresh-file-space-query.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder
        .create_dataset("keep")
        .with_i32_data(&[11, 22, 33, 44]);
    builder.create_dataset("remove").with_u8_data(&[7; 8_192]);
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(&path, properties(MemoryStrategy::Bounded)).unwrap();
    let before = file.file_space_info().unwrap().unwrap();
    file.root().delete("remove").unwrap();
    file.commit().unwrap();
    let after = file.file_space_info().unwrap().unwrap();
    assert_ne!(before.manager_addrs, after.manager_addrs);
    let bytes = file.disk_bytes().unwrap();
    assert_eq!(
        File::from_bytes(bytes).unwrap().file_space_info().unwrap(),
        Some(after)
    );
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        [11, 22, 33, 44]
    );
    file.close().unwrap();
}

fn properties(memory: MemoryStrategy) -> FileAccessProperties {
    FileAccessProperties::new()
        .with_memory_strategy(memory)
        .with_sync_policy(SyncPolicy::OnClose)
        .with_metadata_cache(MetadataCacheConfig::disabled())
        .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct PreviousOpening {
    file: FileInner,
    cached_info: Option<FileSpaceInfo>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl PreviousOpening {
    fn from_bytes(data: Vec<u8>, properties: FileAccessProperties) -> Arc<Self> {
        let (superblock, widths, opened) = FileInner::parse_superblock(data.as_slice()).unwrap();
        opened
            .require_complete(u64::try_from(data.len()).unwrap())
            .unwrap();
        let empty = OpeningMetadata::from_message(
            &OpenedSource::for_test(&BytesSource::new(&data), opened.addresses()),
            Ok(None),
            widths,
        );
        Self::new(FileInner::from_parts(
            Backend::InMemory(data),
            AccessMode::ReadOnly,
            superblock,
            widths,
            opened,
            ReaderOpening {
                handle: None,
                extension: Some(empty),
            },
            properties,
        ))
    }

    fn new(mut file: FileInner) -> Arc<Self> {
        let cached_info = file.parse_file_space_info().ok().flatten();
        file.sohm_table = (|| {
            let rel = file.superblock.superblock_extension_address?;
            let address = file.addresses().resolve(rel).ok()?.get();
            let header = file.parse_header(address).ok()?;
            let body = header
                .messages
                .iter()
                .find(|message| message.msg_type == MessageType::SHARED_MESSAGE_TABLE)?;
            let message =
                SharedMessageTableMessage::parse_with_offset_width(&body.data, file.widths.offsets)
                    .ok()?;
            file.with_opened_source(|source| {
                OpeningMetadata::from_message(source, Ok(Some(message)), file.widths).into_table()
            })
        })();
        Arc::new(Self { file, cached_info })
    }

    fn into_file(self: Arc<Self>) -> File {
        let previous = Arc::try_unwrap(self).ok().unwrap();
        File {
            inner: Arc::new(previous.file),
        }
    }
}

fn assert_file(file: &File, info: &FileSpaceInfo) {
    assert_eq!(file.file_space_info().unwrap().as_ref(), Some(info));
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        [11, 22, 33, 44]
    );
    assert_eq!(
        file.inner.sohm_table.as_deref(),
        Some(&SohmTable {
            indexes: vec![SohmIndexHeader {
                message_type_flags: 1 << WireType::DATATYPE.0,
                min_message_size: 8,
                list_max: 4,
                btree_min: 2,
                message_count: 0,
                kind: SohmIndexKind::List,
                index_address: None,
                heap_address: None,
            }],
        })
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Default)]
struct Reads {
    calls: AtomicU64,
    bytes: AtomicU64,
    backend_calls: AtomicU64,
    backend_bytes: AtomicU64,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct DiskSource {
    source: ReadSeekSource<CountedFile>,
    counts: Arc<Reads>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl DiskSource {
    fn new(path: &Path, counts: Arc<Reads>) -> Self {
        Self {
            source: ReadSeekSource::new(CountedFile {
                file: std::fs::File::open(path).unwrap(),
                counts: Arc::clone(&counts),
            })
            .unwrap(),
            counts,
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Source for DiskSource {
    fn len(&self) -> u64 {
        self.source.len()
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.counts.calls.fetch_add(1, Ordering::Relaxed);
        self.counts
            .bytes
            .fetch_add(u64::try_from(bytes.len()).unwrap(), Ordering::Relaxed);
        self.source.read_at(offset, bytes)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct CountedFile {
    file: std::fs::File,
    counts: Arc<Reads>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Read for CountedFile {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.counts.backend_calls.fetch_add(1, Ordering::Relaxed);
        let n = self.file.read(buffer)?;
        self.counts
            .backend_bytes
            .fetch_add(u64::try_from(n).unwrap(), Ordering::Relaxed);
        Ok(n)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Seek for CountedFile {
    fn seek(&mut self, position: io::SeekFrom) -> io::Result<u64> {
        self.file.seek(position)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn report(
    label: &str,
    version: HeaderVersion,
    chunks: usize,
    measured: &allocation::Measured,
    control: &allocation::Measured,
    current: &Reads,
    prior: &Reads,
) {
    eprintln!(
        "backend reads={} bytes={}, prior reads={} bytes={}",
        current.backend_calls.load(Ordering::Relaxed),
        current.backend_bytes.load(Ordering::Relaxed),
        prior.backend_calls.load(Ordering::Relaxed),
        prior.backend_bytes.load(Ordering::Relaxed)
    );
    eprintln!(
        "{label} {version:?}, {chunks} chunks: {measured}; prior {control}; source reads={} bytes={}, prior reads={} bytes={}",
        current.calls.load(Ordering::Relaxed),
        current.bytes.load(Ordering::Relaxed),
        prior.calls.load(Ordering::Relaxed),
        prior.bytes.load(Ordering::Relaxed)
    );
}

struct Fixture {
    bytes: Vec<u8>,
    info: FileSpaceInfo,
    table_len: usize,
}

impl Fixture {
    fn new(version: HeaderVersion, chunks: usize) -> Self {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
        );
        builder
            .create_dataset("keep")
            .with_i32_data(&[11, 22, 33, 44]);
        let mut bytes = builder.finish().unwrap();
        let original = File::from_bytes(bytes.clone()).unwrap();
        let mut block = original.superblock().clone();
        let widths = original.inner.widths;
        let mut info = original.file_space_info().unwrap().unwrap();
        let table = sohm::table(
            &[sohm::Index {
                kind: sohm::Kind::LIST,
                message_type_flags: 1 << WireType::DATATYPE.0,
                min_message_size: 8,
                list_max: 4,
                btree_min: 2,
                message_count: 0,
                index_address: None,
                heap_address: None,
            }],
            Widths::EIGHT,
        );
        let pieces = Self::pieces(
            version,
            chunks,
            &hdf5_pure_format::__private::serialize_file_space_info_v1(widths, &info).unwrap(),
        );
        let end = pieces.last().map(|(at, bytes)| at + bytes.len()).unwrap();
        info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(
            u64::try_from(end).unwrap(),
        )));
        let pieces = Self::pieces(
            version,
            chunks,
            &hdf5_pure_format::__private::serialize_file_space_info_v1(widths, &info).unwrap(),
        );
        bytes.resize(end, 0);
        bytes[4_096..4_096 + table.len()].copy_from_slice(&table);
        for (at, piece) in pieces {
            bytes[at..at + piece.len()].copy_from_slice(&piece);
        }
        block.superblock_extension_address = Some(StoredAddress::new(8_192));
        block.eof_address = u64::try_from(end).unwrap();
        let header = hdf5_pure_format::__private::serialize_superblock(&block).unwrap();
        bytes[..header.len()].copy_from_slice(&header);
        Self {
            bytes,
            info,
            table_len: table.len(),
        }
    }

    fn pieces(version: HeaderVersion, chunks: usize, info: &[u8]) -> Vec<(usize, Vec<u8>)> {
        let message = |kind, data: &[u8]| {
            let mut data = data.to_vec();
            if matches!(version, HeaderVersion::One | HeaderVersion::NestedOne(_)) {
                data.resize(
                    data.len().div_ceil(v1::MESSAGE_ALIGNMENT) * v1::MESSAGE_ALIGNMENT,
                    0,
                );
            }
            Message::new(kind, &data)
        };
        let records = vec![
            Message::nil(60_000),
            message(WireType::FILE_SPACE_INFO, info),
            message(
                WireType::SHARED_MESSAGE_TABLE,
                &sohm::table_message(4_096, 1, Widths::EIGHT),
            ),
            message(WireType::UNKNOWN, &[0xA5; 13]).with_flags(WireFlags::MARK_IF_UNKNOWN),
        ];
        if let HeaderVersion::NestedOne(depth) = version {
            let mut pieces = Vec::with_capacity(depth + 1);
            let mut next = None;
            for index in (0..depth).rev() {
                let mut records = vec![Message::nil(8)];
                if let Some((at, len)) = next {
                    records.push(Message::continuation(at, len, Widths::EIGHT));
                }
                let piece = v1::message_records(&records);
                let at = 131_072 + index * 1_024;
                next = Some((
                    u64::try_from(at).unwrap(),
                    u64::try_from(piece.len()).unwrap(),
                ));
                pieces.push((at, piece));
            }
            let mut root = v1::Header::new().messages(records);
            if let Some((at, len)) = next {
                root = root.message(Message::continuation(at, len, Widths::EIGHT));
            }
            pieces.push((8_192, root.build()));
            pieces.sort_unstable_by_key(|(at, _)| *at);
            return pieces;
        }
        let mut pieces = Vec::new();
        for index in 1..chunks {
            let piece = match version {
                HeaderVersion::One | HeaderVersion::NestedOne(_) => v1::message_records(&records),
                HeaderVersion::Two => {
                    v2::continuation_chunk(&records, v2::HeaderFlags::CHUNK_SIZE_WIDTH)
                }
            };
            pieces.push((131_072 + (index - 1) * 65_536, piece));
        }
        let root = match version {
            HeaderVersion::One | HeaderVersion::NestedOne(_) => {
                let mut header = v1::Header::new().messages(records);
                for (at, piece) in &pieces {
                    header = header.continuation(*at, piece, Widths::EIGHT);
                }
                header.build()
            }
            HeaderVersion::Two => {
                let mut header = v2::Header::new()
                    .flags(v2::HeaderFlags::CHUNK_SIZE_WIDTH)
                    .messages(records);
                for (at, piece) in &pieces {
                    header = header.continuation(*at, piece, Widths::EIGHT);
                }
                header.build()
            }
        };
        pieces.insert(0, (8_192, root));
        pieces
    }
}
