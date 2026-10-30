use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex};

use test_util::superblock;

use super::super::*;
use crate::FreeSectionCoordinates;
use crate::chunked_write::{ChunkMeta, ChunkProvider};
use crate::convert;
use crate::data_layout::ChunkedLayoutFlags;
use crate::dataspace::MaxExtent;
use crate::layout_info::Layout;

type Calls = Arc<Mutex<Vec<usize>>>;

#[rstest::rstest]
#[case::default(None)]
#[case::fsm_session(Some((FileSpaceStrategy::FsmAggr, false)))]
#[case::fsm_persistent(Some((FileSpaceStrategy::FsmAggr, true)))]
#[case::aggr(Some((FileSpaceStrategy::Aggr, false)))]
#[case::none(Some((FileSpaceStrategy::None, false)))]
#[case::page(Some((FileSpaceStrategy::Page, true)))]
fn mixed_reservations_stream_identically(
    #[case] strategy: Option<(FileSpaceStrategy, bool)>,
    #[values(0, 4_096)] userblock: u64,
) {
    let build = |produced_calls: &Calls, chunk_calls: &Calls| {
        let mut builder = FileBuilder::new();
        if let Some((strategy, persist)) = strategy {
            builder.with_file_space_settings(
                FileSpaceSettings::new(strategy).with_persistence(persist),
            );
        }
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        if userblock != 0 {
            builder.with_userblock_content(b"writer prefix");
        }
        builder.commit_datatype("integer_type", crate::type_builders::make_i32_type());
        builder.set_attr("root_label", AttrValue::VarLenString("root value".into()));
        for index in 0..9 {
            builder.set_attr(&format!("root_{index}"), AttrValue::I64(index));
        }
        let mut group = builder.create_group("group");
        group.set_attr("group_label", AttrValue::VarLenString("group value".into()));
        for index in 0..9 {
            group.set_attr(&format!("group_{index}"), AttrValue::I64(index));
        }
        let dataset = group.create_dataset("values");
        dataset
            .with_i32_data(&[1, 2, 3])
            .with_committed_datatype("integer_type");
        dataset.set_attr(
            "dataset_label",
            AttrValue::VarLenString("dataset value".into()),
        );
        for index in 0..9 {
            dataset.set_attr(&format!("dataset_{index}"), AttrValue::I64(index));
        }
        builder.add_group(group.finish());
        builder.create_dataset("empty").with_u8_data(&[]);
        builder
            .create_dataset("references")
            .with_path_references(&["group/values"]);
        builder
            .create_dataset("strings")
            .with_vlen_strings(&["first", "second"]);
        builder
            .create_dataset("chunked_strings")
            .with_vlen_strings(&["early", "heap", "references"])
            .with_chunks(&[1])
            .with_deflate(DeflateLevel::Six);
        builder
            .create_dataset("filtered")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0])
            .with_chunks(&[2])
            .with_deflate(DeflateLevel::Six);
        builder.create_dataset("produced").with_produced_data(
            crate::type_builders::make_u8_type(),
            &[67],
            67,
            16,
            Box::new(BlockProvider {
                total: 67,
                block: 16,
                calls: Arc::clone(produced_calls),
                short_block: None,
            }),
        );
        stage_lazy(
            &mut builder,
            "verbatim",
            vec![f64_chunk(&[1.0, 2.0]), f64_chunk(&[3.0, 4.0])],
            &[4],
            &[2],
            Some(&[MaxExtent::Unlimited]),
            Arc::clone(chunk_calls),
            None,
        );
        if strategy.is_none_or(|(strategy, _)| {
            strategy == FileSpaceStrategy::Aggr || strategy == FileSpaceStrategy::FsmAggr
        }) {
            builder.set_attr("huge", AttrValue::AsciiString("h".repeat(70_000)));
        }
        builder
    };
    let produced_calls = Calls::default();
    let chunk_calls = Calls::default();
    let buffered = build(&produced_calls, &chunk_calls).finish().unwrap();
    assert_eq!(*produced_calls.lock().unwrap(), [0, 1, 2, 3, 4]);
    assert_eq!(*chunk_calls.lock().unwrap(), [0, 1]);
    produced_calls.lock().unwrap().clear();
    chunk_calls.lock().unwrap().clear();
    let mut sink = NonseekableSink::default();
    build(&produced_calls, &chunk_calls)
        .finish_to(&mut sink)
        .unwrap();
    assert_eq!(sink.bytes, buffered);
    assert_eq!(*produced_calls.lock().unwrap(), [0, 1, 2, 3, 4]);
    assert_eq!(*chunk_calls.lock().unwrap(), [0, 1]);
    let file = crate::reader::File::from_bytes(buffered).unwrap();
    assert_eq!(
        file.dataset("group/values").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
    assert_eq!(
        file.dataset("chunked_strings")
            .unwrap()
            .read_vlen_strings(crate::vl_data::VlenStringReadOptions::default())
            .unwrap(),
        ["early", "heap", "references"]
    );
    assert_eq!(
        file.dataset("strings")
            .unwrap()
            .read_vlen_strings(crate::vl_data::VlenStringReadOptions::default())
            .unwrap(),
        ["first", "second"]
    );
    assert_eq!(
        file.dataset("produced").unwrap().read_raw().unwrap(),
        BlockProvider::expected(67)
    );
    assert_eq!(read_back_f64(&sink.bytes, "verbatim"), [1.0, 2.0, 3.0, 4.0]);
}

#[rstest::rstest]
#[case::metadata(0)]
#[case::raw(1)]
#[case::alternating(2)]
#[case::block_edges(3)]
#[case::empty(4)]
fn aggregated_creation_streams_final_addresses(
    #[case] workload: usize,
    #[values(false, true)] persist: bool,
    #[values(0, 512)] userblock: u64,
    #[values(None, Some(FileSpaceStrategy::Aggr), Some(FileSpaceStrategy::FsmAggr))]
    strategy: Option<FileSpaceStrategy>,
) {
    let build = || {
        let mut builder = FileBuilder::new();
        if let Some(strategy) = strategy {
            builder.with_file_space_settings(
                FileSpaceSettings::new(strategy)
                    .with_persistence(persist)
                    .with_threshold(9),
            );
        }
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        match workload {
            0 => {
                for i in 0..40 {
                    let group = builder.create_group(&format!("group_{i}")).finish();
                    builder.add_group(group);
                }
            }
            1 => {
                for i in 0..40 {
                    builder
                        .create_dataset(&format!("values_{i}"))
                        .with_u8_data(&[i; 17]);
                }
            }
            2 => {
                for i in 0..20 {
                    let mut group = builder.create_group(&format!("group_{i}"));
                    group.create_dataset("values").with_i32_data(&[i; 13]);
                    builder.add_group(group.finish());
                }
            }
            3 => {
                builder.create_dataset("first").with_u8_data(&[11; 17]);
                builder.create_dataset("large").with_u8_data(&[13; 4_097]);
                builder.create_dataset("last").with_u8_data(&[17; 19]);
            }
            _ => {}
        }
        builder
    };
    let persistent = persist && strategy == Some(FileSpaceStrategy::FsmAggr);
    let bytes = build().finish().unwrap();
    let mut sink = NonseekableSink::default();
    build().finish_to(&mut sink).unwrap();
    assert_eq!(sink.bytes, bytes);
    let length = bytes.len() as u64;
    let file = crate::reader::File::from_bytes(bytes).unwrap();
    assert_eq!(file.superblock().eof_address, length);
    assert_eq!(
        file.file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
        strategy
    );
    if let Some(strategy) = strategy {
        let info = file.file_space_info().unwrap().unwrap();
        assert_eq!((info.strategy, info.persist), (strategy, persistent));
        if !persistent {
            assert_eq!(info.manager_addrs, []);
        }
    }
    let sections = file.persisted_free_sections().unwrap();
    if persistent {
        assert!(
            sections
                .unwrap()
                .free_sections()
                .sections()
                .iter()
                .all(|section| {
                    section.coordinates() == FreeSectionCoordinates::Stored
                        && section.length().get() >= 9
                        && section.end_address() <= length - userblock
                })
        );
    } else {
        assert_eq!(sections, None);
    }
    match workload {
        0 => assert_eq!(file.root().groups().unwrap().len(), 40),
        1 => {
            for i in 0..40 {
                assert_eq!(
                    file.dataset(&format!("values_{i}"))
                        .unwrap()
                        .read_raw()
                        .unwrap(),
                    [i; 17]
                );
            }
        }
        2 => {
            for i in 0..20 {
                assert_eq!(
                    file.dataset(&format!("group_{i}/values"))
                        .unwrap()
                        .read_i32()
                        .unwrap(),
                    [i; 13]
                );
            }
        }
        3 => {
            let first = userblock
                + hdf5_pure_format::__private::serialize_superblock(file.superblock())
                    .unwrap()
                    .len() as u64
                + hdf5_pure_space::__private::AGGREGATE_BLOCK_SIZE;
            assert_eq!(
                file.dataset("first").unwrap().layout().unwrap(),
                Layout::Contiguous {
                    address: Some(first),
                    size: 17
                }
            );
            assert_eq!(
                file.dataset("large").unwrap().layout().unwrap(),
                Layout::Contiguous {
                    address: Some(first + 17),
                    size: 4_097
                }
            );
            assert_eq!(
                file.dataset("last").unwrap().layout().unwrap(),
                Layout::Contiguous {
                    address: Some(first + 4_114),
                    size: 19
                }
            );
            if persistent {
                assert!(length > first + 4_133);
            } else {
                assert_eq!(length, first + 4_133);
            }
            assert!(file.dataset("last").unwrap().header_address().unwrap() < first);
            assert_eq!(file.dataset("first").unwrap().read_raw().unwrap(), [11; 17]);
            assert_eq!(
                file.dataset("large").unwrap().read_raw().unwrap(),
                [13; 4_097]
            );
            assert_eq!(file.dataset("last").unwrap().read_raw().unwrap(), [17; 19]);
        }
        _ => {
            assert_eq!(file.root().groups().unwrap(), Vec::<String>::new());
            assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
        }
    }
}

#[rstest::rstest]
#[case::first(0)]
#[case::later(2)]
fn aggregated_creation_reports_provider_errors(
    #[case] short: usize,
    #[values(false, true)] persist: bool,
    #[values(false, true)] streamed: bool,
    #[values(None, Some(FileSpaceStrategy::Aggr), Some(FileSpaceStrategy::FsmAggr))]
    strategy: Option<FileSpaceStrategy>,
) {
    let mut builder = FileBuilder::new();
    if let Some(strategy) = strategy {
        builder
            .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(persist));
    }
    let calls = Calls::default();
    builder.create_dataset("data").with_produced_data(
        crate::type_builders::make_u8_type(),
        &[67],
        67,
        16,
        Box::new(BlockProvider {
            total: 67,
            block: 16,
            calls: Arc::clone(&calls),
            short_block: Some(short),
        }),
    );
    let error = if streamed {
        builder.finish_to(NonseekableSink::default()).unwrap_err()
    } else {
        builder.finish().unwrap_err()
    };
    let Error::Format(FormatError::SerializationError(reason)) = error else {
        panic!("expected SerializationError, got {error:?}");
    };
    assert_eq!(
        reason,
        "a produced dataset's block does not match its planned size"
    );
    assert_eq!(*calls.lock().unwrap(), (0..=short).collect::<Vec<_>>());
}

#[rstest::rstest]
fn aggregated_creation_reports_sink_errors(
    #[values(false, true)] persist: bool,
    #[values(None, Some(FileSpaceStrategy::Aggr), Some(FileSpaceStrategy::FsmAggr))]
    strategy: Option<FileSpaceStrategy>,
) {
    struct BrokenOutput;
    impl Write for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut builder = FileBuilder::new();
    if let Some(strategy) = strategy {
        builder
            .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(persist));
    }
    builder.create_dataset("data").with_u8_data(&[17; 19]);
    let error = builder.finish_to(BrokenOutput).unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected Io, got {error:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
}

#[rstest::rstest]
#[case::internal_emission(0, false, false)]
#[case::internal_partial_write(4_096, false, false)]
#[case::public_emission(0, false, true)]
#[case::public_partial_write(4_096, false, true)]
#[case::public_flush(usize::MAX, true, true)]
fn streaming_results_preserve_io_causes(
    #[case] remaining: usize,
    #[case] fail_flush: bool,
    #[case] public: bool,
    #[values(false, true)] raw_os_error: bool,
) {
    let cause = if raw_os_error {
        io::Error::from_raw_os_error(123)
    } else {
        io::Error::new(io::ErrorKind::StorageFull, OutputCause(73))
    };
    let output = FailingOutput {
        remaining,
        fail_flush,
        error: Some(cause),
    };
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_u8_data(&vec![17; 32_768]);
    let error = if public {
        let error = builder.finish_to(output).unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected Io, got {error:?}");
        };
        error
    } else {
        let error = builder
            .writer
            .finish_to_sink(&mut WriteSink::new(output))
            .unwrap_err();
        let EmissionError::Sink(error) = error else {
            panic!("expected Sink, got {error:?}");
        };
        error
    };
    if raw_os_error {
        assert_eq!(error.raw_os_error(), Some(123));
    } else {
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        assert_eq!(
            error
                .get_ref()
                .unwrap()
                .downcast_ref::<OutputCause>()
                .unwrap()
                .0,
            73
        );
    }
}

#[derive(Debug)]
struct OutputCause(u32);

impl fmt::Display for OutputCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "output failure {}", self.0)
    }
}

impl StdError for OutputCause {}

struct FailingOutput {
    remaining: usize,
    fail_flush: bool,
    error: Option<io::Error>,
}

impl Write for FailingOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(self
                .error
                .take()
                .unwrap_or_else(|| io::ErrorKind::StorageFull.into()));
        }
        let written = bytes.len().min(self.remaining);
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.fail_flush {
            return Err(self
                .error
                .take()
                .unwrap_or_else(|| io::ErrorKind::StorageFull.into()));
        }
        Ok(())
    }
}

#[derive(Default)]
struct NonseekableSink {
    bytes: Vec<u8>,
}

impl Write for NonseekableSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let count = bytes.len().min(17);
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Stores encoded chunks and records provider calls.
///
/// `short_slot` selects a chunk to shorten by one byte.
struct MemProvider {
    chunks: Vec<Vec<u8>>,
    calls: Calls,
    short_slot: Option<usize>,
}

impl ChunkProvider for MemProvider {
    #[expect(
        clippy::panic_in_result_fn,
        reason = "a test provider asserts the emitter's contract"
    )]
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        self.calls.lock().unwrap().push(index);
        assert!(
            out.is_empty(),
            "the emitter hands the provider an empty buffer"
        );
        out.extend_from_slice(&self.chunks[index]);
        if self.short_slot == Some(index) {
            out.pop();
        }
        Ok(())
    }
}

fn f64_chunk(vals: &[f64]) -> Vec<u8> {
    let mut v = Vec::new();
    for &x in vals {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

fn meta_of(chunk_bytes: &[Vec<u8>]) -> Vec<ChunkMeta> {
    chunk_bytes
        .iter()
        .map(|c| ChunkMeta {
            compressed_size: c.len() as u64,
            filter_mask: 0,
        })
        .collect()
}

/// Stages an unfiltered f64 dataset using lazily supplied raw chunk bytes.
fn stage_lazy(
    b: &mut FileBuilder,
    name: &str,
    chunk_bytes: Vec<Vec<u8>>,
    dims: &[u64],
    chunk_dims: &[u64],
    maxshape: Option<&[MaxExtent]>,
    calls: Calls,
    short_slot: Option<usize>,
) {
    let meta = meta_of(&chunk_bytes);
    let provider = MemProvider {
        chunks: chunk_bytes,
        calls,
        short_slot,
    };
    b.create_dataset(name).with_raw_chunks_lazy(
        crate::type_builders::make_f64_type(),
        dims,
        maxshape,
        chunk_dims,
        convert::nz(8),
        ChunkedLayoutFlags::NONE,
        None,
        meta,
        Box::new(provider),
    );
}

/// Returns a builder with one unfiltered f64 dataset named `d` and a lazy chunk provider.
fn build_lazy(
    chunk_bytes: Vec<Vec<u8>>,
    dims: &[u64],
    chunk_dims: &[u64],
    maxshape: Option<&[MaxExtent]>,
    calls: Calls,
    short_slot: Option<usize>,
) -> FileBuilder {
    let mut b = FileBuilder::new();
    stage_lazy(
        &mut b,
        "d",
        chunk_bytes,
        dims,
        chunk_dims,
        maxshape,
        calls,
        short_slot,
    );
    b
}

fn read_back_f64(bytes: &[u8], path: &str) -> Vec<f64> {
    let file = crate::reader::File::from_bytes(bytes.to_vec()).unwrap();
    let raw = file.dataset(path).unwrap().read_raw().unwrap();
    raw.as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect()
}

#[test]
fn streamed_output_matches_buffered_and_streams_one_chunk_at_a_time() {
    let chunks = vec![
        f64_chunk(&[1.0, 2.0]),
        f64_chunk(&[3.0, 4.0]),
        f64_chunk(&[5.0, 6.0]),
    ];

    let calls_buf = Arc::new(Mutex::new(Vec::new()));
    let buffered = build_lazy(chunks.clone(), &[6], &[2], None, calls_buf.clone(), None)
        .finish()
        .unwrap();

    let calls_str = Arc::new(Mutex::new(Vec::new()));
    let mut streamed = Vec::new();
    build_lazy(chunks.clone(), &[6], &[2], None, calls_str.clone(), None)
        .finish_to(&mut streamed)
        .unwrap();

    assert_eq!(
        buffered, streamed,
        "streamed output must be byte-identical to buffered output"
    );
    // Check that each provider is read once per slot, in ascending order.
    assert_eq!(*calls_buf.lock().unwrap(), vec![0, 1, 2]);
    assert_eq!(*calls_str.lock().unwrap(), vec![0, 1, 2]);
    assert_eq!(
        read_back_f64(&buffered, "d"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn file_builder_keeps_its_auto_traits() {
    // Check the public builder's auto traits with boxed lazy providers.
    fn assert_auto_traits<T: Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
    assert_auto_traits::<FileBuilder>();
}

#[rstest::rstest]
#[case::first_chunk(0)]
#[case::later_chunk(2)]
fn streaming_writer_rejects_provider_size_mismatch(#[case] short_slot: usize) {
    let chunks = vec![
        f64_chunk(&[1.0, 2.0]),
        f64_chunk(&[3.0, 4.0]),
        f64_chunk(&[5.0, 6.0]),
    ];
    let calls = Arc::new(Mutex::new(Vec::new()));
    let error = build_lazy(
        chunks,
        &[6],
        &[2],
        None,
        Arc::clone(&calls),
        Some(short_slot),
    )
    .finish()
    .unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::ChunkedReadError(
            "verbatim chunk provider returned a chunk whose size differs from the planned size"
                .into()
        )
    );
    assert_eq!(*calls.lock().unwrap(), (0..=short_slot).collect::<Vec<_>>());
}

/// Asserts equal buffered and streamed bytes and the expected decoded values.
fn assert_variant_streams_identically(
    chunks: Vec<Vec<u8>>,
    dims: &[u64],
    chunk_dims: &[u64],
    maxshape: Option<&[MaxExtent]>,
    expected: &[f64],
) {
    let buffered = build_lazy(
        chunks.clone(),
        dims,
        chunk_dims,
        maxshape,
        Arc::new(Mutex::new(Vec::new())),
        None,
    )
    .finish()
    .unwrap();
    let mut streamed = Vec::new();
    build_lazy(
        chunks,
        dims,
        chunk_dims,
        maxshape,
        Arc::new(Mutex::new(Vec::new())),
        None,
    )
    .finish_to(&mut streamed)
    .unwrap();
    assert_eq!(
        buffered, streamed,
        "index variant dims={dims:?} chunk={chunk_dims:?} must stream identically"
    );
    assert_eq!(
        read_back_f64(&buffered, "d"),
        expected,
        "index variant dims={dims:?} chunk={chunk_dims:?} must read back correctly"
    );
}

#[test]
fn streamed_equals_buffered_across_index_variants() {
    // single-chunk, fixed-array (>1 chunk), and extensible-array (unlimited
    // max shape) all lay out from sizes alone, so each must stream identically
    // and read back to the right values.
    assert_variant_streams_identically(vec![f64_chunk(&[1.0, 2.0])], &[2], &[2], None, &[1.0, 2.0]);
    assert_variant_streams_identically(
        (0..5)
            .map(|i| f64_chunk(&[i as f64, i as f64 + 0.5]))
            .collect(),
        &[10],
        &[2],
        None,
        &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    );
    assert_variant_streams_identically(
        vec![f64_chunk(&[1.0, 2.0]), f64_chunk(&[3.0, 4.0])],
        &[4],
        &[2],
        Some(&[MaxExtent::Unlimited]),
        &[1.0, 2.0, 3.0, 4.0],
    );
}

/// Accepts a fixed byte budget, then reports an I/O error on subsequent writes.
struct FailAfter {
    remaining: usize,
}
impl Write for FailAfter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Err(std::io::Error::other("write limit reached"));
        }
        let n = buf.len().min(self.remaining);
        self.remaining -= n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn streaming_io_error_surfaces_as_error_io() {
    // Exceed the internal buffer to exercise an I/O failure during emission.
    let chunks: Vec<Vec<u8>> = (0..12).map(|_| f64_chunk(&[1.0; 256])).collect(); // 12 * 2 KiB
    let builder = build_lazy(
        chunks,
        &[3072],
        &[256],
        None,
        Arc::new(Mutex::new(Vec::new())),
        None,
    );
    let err = builder
        .finish_to(FailAfter { remaining: 4096 })
        .unwrap_err();
    let Error::Io(error) = err else {
        panic!("expected Io, got {err:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.raw_os_error(), None);
    assert_eq!(error.to_string(), "write limit reached");
}

#[test]
fn streamed_dataset_with_attribute_and_contiguous_sibling() {
    // Combine lazy chunks, attributes, populated contiguous storage, and empty storage.
    let build = || {
        let chunks = vec![f64_chunk(&[1.0, 2.0]), f64_chunk(&[3.0, 4.0])];
        let meta = meta_of(&chunks);
        let provider = MemProvider {
            chunks,
            calls: Arc::new(Mutex::new(Vec::new())),
            short_slot: None,
        };
        let mut b = FileBuilder::new();
        // Configure the one streamed dataset and its attribute on the same
        // builder (a second `create_dataset` would add a *different* dataset).
        b.create_dataset("chunked")
            .with_raw_chunks_lazy(
                crate::type_builders::make_f64_type(),
                &[4],
                None,
                &[2],
                convert::nz(8),
                ChunkedLayoutFlags::NONE,
                None,
                meta,
                Box::new(provider),
            )
            .set_attr("units", AttrValue::I64(7));
        b.create_dataset("contig")
            .with_f64_data(&[10.0, 11.0, 12.0]);
        b.create_dataset("empty").with_f64_data(&[]);
        b
    };
    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();
    assert_eq!(buffered, streamed, "mixed file must stream identically");
    assert_eq!(
        read_back_f64(&buffered, "chunked"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(read_back_f64(&buffered, "contig"), vec![10.0, 11.0, 12.0]);
}

#[test]
fn userblock_content_leads_the_file_on_both_output_paths() {
    const HEADER: &[u8] = b"a wrapper format's header";
    let build = || {
        let mut b = FileBuilder::new();
        b.with_userblock(UserblockSize::MIN)
            .with_userblock_content(HEADER);
        b.create_dataset("x").with_f64_data(&[1.0, 2.0]);
        b
    };
    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();

    assert_eq!(buffered, streamed, "content must stream identically");
    assert_eq!(&buffered[..HEADER.len()], HEADER);
    assert!(
        buffered[HEADER.len()..512].iter().all(|&b| b == 0),
        "the rest of the region stays zero-filled"
    );
    // Check the signature immediately after the userblock.
    assert_eq!(&buffered[512..520], b"\x89HDF\r\n\x1a\n");
    assert_eq!(read_back_f64(&buffered, "x"), vec![1.0, 2.0]);
}

#[test]
fn an_unset_userblock_stays_zero_filled() {
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::MIN);
    b.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = b.finish().unwrap();
    assert!(bytes[..512].iter().all(|&b| b == 0));
}

/// Produces bounded blocks, including a short final block when the total is not divisible.
struct BlockProvider {
    total: usize,
    block: usize,
    calls: Calls,
    /// The block to shorten by one byte for emitter length-error cases.
    short_block: Option<usize>,
}

impl BlockProvider {
    /// Returns a byte from an offset-dependent ramp.
    fn byte(i: usize) -> u8 {
        (i % 251) as u8
    }

    fn expected(total: usize) -> Vec<u8> {
        (0..total).map(Self::byte).collect()
    }
}

impl ChunkProvider for BlockProvider {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        self.calls.lock().unwrap().push(index);
        let start = index * self.block;
        let mut end = (start + self.block).min(self.total);
        if self.short_block == Some(index) {
            end -= 1;
        }
        out.extend((start..end).map(Self::byte));
        Ok(())
    }
}

// Use a region length that requires a short final block.
#[test]
fn a_produced_contiguous_dataset_matches_a_materialized_one() {
    const TOTAL: usize = 8 * 1000 + 8 * 3; // 1003 f64
    const BLOCK: usize = 8 * 256;
    let expected = BlockProvider::expected(TOTAL);

    let materialize = || {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_raw_data(
                crate::type_builders::make_f64_type(),
                expected.clone(),
                1003,
            )
            .with_shape(&[1003]);
        b
    };
    let produce = |calls: &Calls| {
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_produced_data(
            crate::type_builders::make_f64_type(),
            &[1003],
            TOTAL as u64,
            BLOCK as u64,
            Box::new(BlockProvider {
                total: TOTAL,
                block: BLOCK,
                calls: Arc::clone(calls),
                short_block: None,
            }),
        );
        b
    };

    let calls = Calls::default();
    let produced = produce(&calls).finish().unwrap();
    assert_eq!(
        materialize().finish().unwrap(),
        produced,
        "a produced region must be byte-for-byte a materialized one"
    );
    // Every block once, ascending, with a short tail at the end.
    let n = TOTAL.div_ceil(BLOCK);
    assert_eq!(*calls.lock().unwrap(), (0..n).collect::<Vec<_>>());
    const { assert!(TOTAL % BLOCK != 0, "the fixture must leave a short tail") };

    let calls = Calls::default();
    let mut streamed = Vec::new();
    produce(&calls).finish_to(&mut streamed).unwrap();
    assert_eq!(produced, streamed, "and identical on the streaming path");
    assert_eq!(read_back_f64(&produced, "d").len(), 1003);
}

// Compare planned placement using a produced region's declared length with materialized bytes.
#[test]
fn a_produced_dataset_is_placed_correctly_in_a_paged_file() {
    // Larger than the 4 KiB page, so it lands in the large run whose
    // page-aligned fragments the free-space managers describe.
    const TOTAL: usize = 8 * 4096;
    const BLOCK: usize = 8 * 512;
    let expected = BlockProvider::expected(TOTAL);

    let build = |produced: bool| {
        let mut b = FileBuilder::new();
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
        let ds = b.create_dataset("d");
        if produced {
            ds.with_produced_data(
                crate::type_builders::make_f64_type(),
                &[4096],
                TOTAL as u64,
                BLOCK as u64,
                Box::new(BlockProvider {
                    total: TOTAL,
                    block: BLOCK,
                    calls: Calls::default(),
                    short_block: None,
                }),
            );
        } else {
            ds.with_raw_data(
                crate::type_builders::make_f64_type(),
                expected.clone(),
                4096,
            )
            .with_shape(&[4096]);
        }
        b
    };

    let materialized = build(false).finish().unwrap();
    let produced = build(true).finish().unwrap();
    assert_eq!(
        materialized, produced,
        "a paged file must place a produced region exactly where it places a materialized one"
    );
    let values = read_back_f64(&produced, "d");
    assert_eq!(values.len(), 4096);
    assert!(
        values.into_iter().map(f64::to_bits).eq(expected
            .as_chunks::<{ size_of::<f64>() }>()
            .0
            .iter()
            .map(|bytes| u64::from_le_bytes(*bytes)))
    );
}

// Stage the producer directly to exercise the emitter's own block-length check.
#[rstest::rstest]
#[case::first_block(0)]
#[case::later_block(2)]
fn a_produced_block_of_the_wrong_size_is_rejected_by_the_emitter(#[case] short: usize) {
    const TOTAL: usize = 8 * 1000;
    const BLOCK: usize = 8 * 256;
    let calls = Calls::default();
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_produced_data(
        crate::type_builders::make_f64_type(),
        &[1000],
        TOTAL as u64,
        BLOCK as u64,
        Box::new(BlockProvider {
            total: TOTAL,
            block: BLOCK,
            calls: Arc::clone(&calls),
            short_block: Some(short),
        }),
    );
    let error = b.finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected Format, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::SerializationError(
            "a produced dataset's block does not match its planned size".into()
        )
    );
    assert_eq!(*calls.lock().unwrap(), (0..=short).collect::<Vec<_>>());
}

#[rstest::rstest]
#[case::no_userblock(0)]
#[case::minimum_userblock(512)]
#[case::one_kibibyte(1024)]
#[case::two_kibibytes(2048)]
#[case::four_kibibytes(4096)]
fn writes_valid_userblock_sizes(#[case] size: u64) {
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(size).unwrap());
    builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);
    let bytes = builder.finish().unwrap();
    let offset = usize::try_from(size).unwrap();
    assert_eq!(
        &bytes[offset..offset + superblock::SIGNATURE.len()],
        superblock::SIGNATURE
    );
    assert_eq!(read_back_f64(&bytes, "x"), vec![1.0, 2.0]);
}

#[test]
fn userblock_content_longer_than_its_region_is_refused() {
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::MIN)
        .with_userblock_content(&[7u8; 513]);
    b.create_dataset("x").with_f64_data(&[1.0]);
    match b.finish() {
        Err(Error::Format(FormatError::UserblockContentTooLarge { content, userblock })) => {
            assert_eq!((content, userblock), (513, 512));
        }
        other => panic!("expected UserblockContentTooLarge, got {other:?}"),
    }
}
