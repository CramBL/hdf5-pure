//! Streaming a MAT v7.3 file: `MatBuilder::finish_to` and the producer-backed
//! `write_blocks`.
//!
//! Two properties carry this feature, and every test here is one of them:
//!
//! - **Byte-identity.** A streamed file equals the buffered one, and a
//!   producer-backed dataset equals the materialized one. Without that, choosing
//!   the low-memory API silently changes the file and no known-good fixture
//!   applies to it.
//! - **Nothing is fully resident.** The producer is called during emission, once
//!   per block, into a buffer it shares with every other call. A refactor that
//!   reintroduced buffering would leave every correctness test green.

use hdf5_pure::mat::{Block, Blocking, Compression, DataProducer, MatBuilder, MatError, Options};
use hdf5_pure::{AttrValue, File};
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

/// What a [`DataProducer`] did, observed from the outside.
#[derive(Default, Debug)]
struct Log {
    /// Block indices, in the order they were asked for.
    calls: Vec<usize>,
    /// Address of the buffer handed over on each call. The writer promises one
    /// buffer for the whole dataset, and a reallocation would change this.
    buffers: Vec<usize>,
    /// Largest buffer length seen, i.e. the most data resident at once.
    high_water: usize,
}

/// Generates `f64` elements as a function of their linear index, so a dataset of
/// any size costs nothing to produce and the expected bytes are computable
/// without holding them.
struct Ramp {
    log: Arc<Mutex<Log>>,
    /// Deliberately break the contract on this block, to prove the emitter
    /// checks rather than trusts.
    corrupt: Option<(usize, i64)>,
    /// Fail outright on this block.
    fail_at: Option<usize>,
}

impl Ramp {
    /// Takes no blocking: every request carries its own contract, so a producer
    /// cannot be built against a split that disagrees with the dataset.
    fn new() -> (Self, Arc<Mutex<Log>>) {
        let log = Arc::new(Mutex::new(Log::default()));
        (
            Self {
                log: Arc::clone(&log),
                corrupt: None,
                fail_at: None,
            },
            log,
        )
    }

    /// The value this producer yields for linear element `i`.
    fn value(i: u64) -> f64 {
        i as f64 * 0.5
    }
}

impl DataProducer for Ramp {
    fn block_bytes(&self, block: Block, out: &mut Vec<u8>) -> Result<(), MatError> {
        if self.fail_at == Some(block.index) {
            return Err(MatError::Custom("the source went away".into()));
        }
        for i in 0..block.elements {
            out.extend_from_slice(&Self::value(block.first_element + i).to_le_bytes());
        }
        match self.corrupt {
            Some((at, delta)) if at == block.index => {
                if delta < 0 {
                    out.truncate(out.len() - (-delta) as usize);
                } else {
                    out.extend_from_slice(&vec![0u8; delta as usize]);
                }
            }
            _ => {}
        }
        let mut log = self.log.lock().unwrap();
        log.calls.push(block.index);
        log.buffers.push(out.as_ptr() as usize);
        log.high_water = log.high_water.max(out.len());
        Ok(())
    }
}

/// Every element the ramp stands for, materialized. Only used to build the
/// buffered file a streamed one is compared against.
fn ramp_elements(n: u64) -> Vec<f64> {
    (0..n).map(Ramp::value).collect()
}

/// A `Write` that keeps the bytes' length but not the bytes, so a test can write
/// a dataset far larger than it would want resident.
#[derive(Default)]
struct Discard {
    written: u64,
}

impl std::io::Write for Discard {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.written += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Open a file this crate wrote, checking first that it is a `.mat` and not
/// merely a readable HDF5 file.
///
/// `File::open` skips the userblock without inspecting it, so a `.mat` that lost
/// its MATLAB signature entirely still reads back here — and MATLAB would refuse
/// it. Every test that reaches into a written file goes through this, so the
/// signature is checked wherever the contents are.
fn open_mat(path: &std::path::Path) -> File {
    let bytes = std::fs::read(path).unwrap();
    hdf5_pure::mat::userblock::verify_header(&bytes)
        .expect("a file written as a .mat must carry the MATLAB v7.3 userblock");
    File::open(path).unwrap()
}

fn read_f64(path: &std::path::Path, name: &str) -> Vec<f64> {
    open_mat(path).dataset(name).unwrap().read_f64().unwrap()
}

fn read_class(path: &std::path::Path, name: &str) -> String {
    let file = open_mat(path);
    let attrs = file.dataset(name).unwrap().attrs().unwrap();
    match &attrs["MATLAB_class"] {
        AttrValue::AsciiString(s) | AttrValue::String(s) => s.clone(),
        other => panic!("unexpected class attribute: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// finish_to
// ---------------------------------------------------------------------------

/// The whole point of the sink-based finish: it must not be a second way to
/// build a file, only a second way to deliver the same one. Content here covers
/// every part of the builder that used to run after `FileBuilder::finish` —
/// the MCOS subsystem, the `#refs#` group, and the userblock.
#[test]
fn a_streamed_file_is_byte_identical_to_a_buffered_one() {
    let build = || {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_f64("plain", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        mb.write_char("text", "hello").unwrap();
        mb.write_string_object("modern", &["a".into(), "b".into()], &[2, 1])
            .unwrap();
        mb.struct_("nested", |s| {
            s.write_scalar_i32("n", 7)?;
            s.cell("items", &[2, 1], |c| {
                c.push_scalar_f64(1.5)?;
                c.push_string("two")?;
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
        mb
    };

    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();

    assert_eq!(
        buffered, streamed,
        "the streamed file must be byte-for-byte the buffered one"
    );
    assert_eq!(&buffered[..6], b"MATLAB", "userblock leads the file");
    assert_eq!(&buffered[124..128], &[0x00, 0x02, b'I', b'M']);
    assert_eq!(&buffered[512..516], b"\x89HDF", "superblock follows it");
}

/// The serde entry points are a second assembly path — `to_bytes` builds a
/// `FileBuilder` directly while `to_bytes_with_options` goes through
/// `MatBuilder` — so each needs its own streaming check.
///
/// Gated because those four functions are: `MatBuilder` itself needs only
/// `std`, so every other test in this file runs on the default feature set.
#[cfg(feature = "serde")]
#[test]
fn both_serde_paths_stream_the_bytes_they_return() {
    #[derive(serde::Serialize)]
    struct Payload {
        values: Vec<f64>,
        label: String,
    }
    let value = Payload {
        values: vec![1.0, 2.0, 3.0],
        label: "measurement".into(),
    };

    let buffered = hdf5_pure::mat::to_bytes(&value).unwrap();
    let mut streamed = Vec::new();
    hdf5_pure::mat::to_writer(&value, &mut streamed).unwrap();
    assert_eq!(buffered, streamed, "to_bytes / to_writer must agree");

    let options = Options::default();
    let buffered = hdf5_pure::mat::to_bytes_with_options(&value, &options).unwrap();
    let mut streamed = Vec::new();
    hdf5_pure::mat::to_writer_with_options(&value, &options, &mut streamed).unwrap();
    assert_eq!(
        buffered, streamed,
        "to_bytes_with_options / to_writer_with_options must agree"
    );
}

/// Streaming to a path must not truncate the destination before the value is
/// known to be writable.
///
/// `to_file` used to serialize into a `Vec<u8>` and hand it to `fs::write`, so a
/// refused value returned before touching the filesystem. Creating the file first
/// and streaming into it would silently destroy whatever was already at that
/// path, for an input the crate never even attempted to write.
#[cfg(feature = "serde")]
#[test]
fn a_refused_value_leaves_the_destination_alone() {
    use std::collections::HashMap;

    let dir = tempdir().unwrap();
    let path = dir.path().join("existing.mat");
    std::fs::write(&path, b"an earlier file").unwrap();

    // Map keys must be strings; this is refused by the serializer, before any
    // part of the file could be assembled.
    let mut refused: HashMap<i32, f64> = HashMap::new();
    refused.insert(1, 2.0);

    assert!(hdf5_pure::mat::to_file(&refused, &path).is_err());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"an earlier file",
        "a refused value must leave the destination untouched"
    );

    let options = Options::default();
    assert!(hdf5_pure::mat::to_file_with_options(&refused, &path, &options).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"an earlier file");
}

/// The three finalizers share one path, so a file delivered by any of them is
/// the same file — including one holding a produced dataset, which is the
/// combination `write` is otherwise never checked on. (`producing_and_streaming_compose`
/// covers `finish` against `finish_to`; the read-back tests use `write` but only
/// read values.) `write` is the one that hits the filesystem, which is where a
/// stray difference shows up first.
#[test]
fn every_finalizer_delivers_the_same_file() {
    const DIMS: [usize; 2] = [4, 100_001];
    let dir = tempdir().unwrap();
    let path = dir.path().join("same.mat");
    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    assert!(
        blocking.block_count > 1 && blocking.last_block_elements < blocking.block_elements,
        "the produced dataset must span blocks and leave a short tail"
    );

    let build = || {
        let (producer, _) = Ramp::new();
        let mut mb = MatBuilder::new(Options::default());
        mb.write_f64("v", &[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        mb.write_blocks::<f64>("produced", &DIMS, Box::new(producer))
            .unwrap();
        mb.write_char("t", "abc").unwrap();
        mb
    };
    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();
    build().write(&path).unwrap();

    assert_eq!(buffered, streamed, "finish and finish_to must agree");
    assert_eq!(
        buffered,
        std::fs::read(&path).unwrap(),
        "and write must deliver that same file"
    );
    assert_eq!(read_f64(&path, "produced").len(), CHANNELS_TIMES_SAMPLES);
}

const CHANNELS_TIMES_SAMPLES: usize = 4 * 100_001;

// ---------------------------------------------------------------------------
// write_blocks
// ---------------------------------------------------------------------------

/// If the two paths disagreed, the streamed one could not be reviewed against
/// any fixture built the ordinary way. Sized to span several blocks so the
/// blocking itself is exercised rather than a single-chunk special case.
#[test]
fn a_produced_dataset_is_byte_identical_to_a_materialized_one() {
    // MATLAB [4, 200_000]: a slice is 4 f64, so this spans several 1 MiB blocks.
    const DIMS: [usize; 2] = [4, 200_000];
    let n = (DIMS[0] * DIMS[1]) as u64;

    let mut mb = MatBuilder::new(Options::default());
    mb.write_f64("samples", &DIMS, &ramp_elements(n)).unwrap();
    let materialized = mb.finish().unwrap();

    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    let (producer, log) = Ramp::new();
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("samples", &DIMS, Box::new(producer))
        .unwrap();
    let produced = mb.finish().unwrap();

    assert!(
        blocking.block_count > 1,
        "the fixture must span more than one block, or it proves nothing about blocking"
    );
    assert_eq!(
        materialized, produced,
        "a produced dataset must be byte-for-byte a materialized one"
    );
    assert_eq!(log.lock().unwrap().calls.len(), blocking.block_count);
}

/// The memory property, asserted rather than inspected: the producer is called
/// only during emission, once per block in order, and always into the same
/// buffer — so at no point is more than one block resident.
#[test]
fn blocks_are_pulled_once_each_in_order_during_the_write() {
    const DIMS: [usize; 2] = [8, 500_000]; // 32 MB of f64
    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    let (producer, log) = Ramp::new();

    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("samples", &DIMS, Box::new(producer))
        .unwrap();
    assert!(
        log.lock().unwrap().calls.is_empty(),
        "staging must not touch the producer: layout works from the shape alone"
    );

    let mut sink = Discard::default();
    mb.finish_to(&mut sink).unwrap();

    let log = log.lock().unwrap();
    assert_eq!(
        log.calls,
        (0..blocking.block_count).collect::<Vec<_>>(),
        "every block exactly once, ascending"
    );
    assert!(
        sink.written > 32_000_000,
        "the whole dataset really was written ({} bytes)",
        sink.written
    );
    // One buffer for the whole dataset. Distinct addresses would mean a fresh
    // allocation per block, which is what the `&mut Vec<u8>` contract exists to
    // avoid; growing high water would mean blocks accumulating.
    let first = log.buffers[0];
    assert!(
        log.buffers.iter().all(|&b| b == first),
        "the producer must be handed the same buffer every time"
    );
    assert_eq!(
        log.high_water,
        blocking.block_len(0),
        "never more than one block resident"
    );
}

/// The blocking is chosen so that a block is a contiguous run of MATLAB's linear
/// element order. Reading back proves it: the elements must come out in the same
/// order a materialized write would put them.
#[test]
fn produced_elements_read_back_in_order() {
    const DIMS: [usize; 2] = [3, 7];
    let dir = tempdir().unwrap();
    let path = dir.path().join("ramp.mat");

    let (producer, _) = Ramp::new();
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("ramp", &DIMS, Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    assert_eq!(read_f64(&path, "ramp"), ramp_elements(21));
    assert_eq!(read_class(&path, "ramp"), "double");
}

/// An element count that does not divide by the block size leaves a short last
/// block. It is written short, not padded — the data region is exactly the
/// dataset — so the dataset must read back at its declared length with nothing
/// appended.
#[test]
fn a_short_last_block_is_written_short() {
    // 3 x 100_001 = 300_003 f64 against a 131_072-element block: three blocks,
    // the last a partial remainder.
    const DIMS: [usize; 2] = [3, 100_001];
    let n = (DIMS[0] * DIMS[1]) as u64;
    let dir = tempdir().unwrap();
    let path = dir.path().join("uneven.mat");

    let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
    assert!(
        blocking.last_block_elements < blocking.block_elements,
        "the fixture must actually leave a short tail, or it proves nothing"
    );
    let (producer, log) = Ramp::new();
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("uneven", &DIMS, Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    let read = read_f64(&path, "uneven");
    assert_eq!(read.len() as u64, n, "no padding leaked into the dataset");
    assert_eq!(read, ramp_elements(n));
    assert_eq!(
        log.lock().unwrap().calls,
        (0..blocking.block_count).collect::<Vec<_>>()
    );
}

/// A produced dataset is the same in a struct as at the root: `write_blocks`
/// resolves its target the way every other writer does.
#[test]
fn a_produced_dataset_can_live_inside_a_struct() {
    const DIMS: [usize; 2] = [2, 3];
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.mat");

    let (producer, _) = Ramp::new();
    let mut mb = MatBuilder::new(Options::default());
    mb.struct_("outer", |s| {
        s.write_scalar_i32("tag", 1)?;
        Ok(())
    })
    .unwrap();
    mb.write_blocks::<f64>("top", &DIMS, Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    assert_eq!(read_f64(&path, "top"), ramp_elements(6));
    let file = File::open(&path).unwrap();
    assert!(file.dataset("outer/tag").is_ok());
}

/// A block of the wrong size shifts every address after it, so it is refused
/// instead of written. Both directions, because a long block and a short one
/// fail differently inside the emitter.
#[test]
fn a_block_of_the_wrong_size_is_refused() {
    const DIMS: [usize; 2] = [4, 100_000];
    for delta in [-8i64, 8] {
        let blocking = Blocking::plan::<f64>(&DIMS).unwrap();
        let (mut producer, _) = Ramp::new();
        producer.corrupt = Some((1, delta));
        let mut mb = MatBuilder::new(Options::default());
        mb.write_blocks::<f64>("samples", &DIMS, Box::new(producer))
            .unwrap();

        match mb.finish_to(Discard::default()) {
            Err(MatError::BlockSizeMismatch {
                block,
                expected,
                actual,
            }) => {
                assert_eq!(block, 1);
                assert_eq!(expected, blocking.block_len(1));
                assert_eq!(actual as i64, expected as i64 + delta);
            }
            other => panic!("expected BlockSizeMismatch for delta {delta}, got {other:?}"),
        }
    }
}

/// A producer's own failure has to survive the trip through the writer's
/// provider seam, which speaks a different error type. Flattening it to a
/// message would make a mid-write failure untraceable to its cause.
#[test]
fn a_producers_own_error_is_surfaced_verbatim() {
    const DIMS: [usize; 2] = [4, 100_000];
    let (mut producer, _) = Ramp::new();
    producer.fail_at = Some(1);

    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("samples", &DIMS, Box::new(producer))
        .unwrap();

    match mb.finish_to(Discard::default()) {
        Err(MatError::Custom(msg)) => assert_eq!(msg, "the source went away"),
        other => panic!("expected the producer's own error, got {other:?}"),
    }
}

/// The same trip, for a producer that failed in its *own* error type rather
/// than a message. This is the one path that puts a caller's error down and
/// picks it up again — the provider seam stashes it and hands the writer a
/// placeholder to carry, which the finalizer swaps back — so it is the one that
/// could substitute a different error and still look right from the outside.
#[test]
fn a_producers_error_keeps_its_own_type() {
    use std::error::Error;

    #[derive(Debug, PartialEq)]
    struct AcquisitionStopped {
        channel: u32,
    }

    impl std::fmt::Display for AcquisitionStopped {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "channel {} stopped", self.channel)
        }
    }

    impl Error for AcquisitionStopped {}

    struct Stopping;

    impl DataProducer for Stopping {
        fn block_bytes(&self, _block: Block, _out: &mut Vec<u8>) -> Result<(), MatError> {
            Err(MatError::from_source(AcquisitionStopped { channel: 3 }))
        }
    }

    const DIMS: [usize; 2] = [4, 100_000];
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("samples", &DIMS, Box::new(Stopping))
        .unwrap();

    let err = mb
        .finish_to(Discard::default())
        .expect_err("the producer failed");

    assert_eq!(err.to_string(), "channel 3 stopped");
    assert_eq!(
        err.source()
            .and_then(|e| e.downcast_ref::<AcquisitionStopped>()),
        Some(&AcquisitionStopped { channel: 3 }),
        "the producer's own type has to survive the stash and the swap"
    );
}

/// Compression is refused up front rather than silently dropped: a caller who
/// asked for deflate and got an unfiltered dataset would have no way to notice.
#[test]
fn compression_is_refused_rather_than_ignored() {
    let mut options = Options::default();
    options.compression = Compression::Deflate {
        level: 6,
        shuffle: false,
    };
    let (producer, _) = Ramp::new();
    let mut mb = MatBuilder::new(options);
    match mb
        .write_blocks::<f64>("x", &[2, 4], Box::new(producer))
        .map(|_| ())
    {
        Err(MatError::CompressionUnsupportedForBlocks) => {}
        other => panic!("expected CompressionUnsupportedForBlocks, got {other:?}"),
    }
}

/// An empty shape writes the same marker every other writer does, and never
/// asks the producer for anything.
#[test]
fn an_empty_shape_writes_a_marker_and_asks_for_nothing() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.mat");

    let materialized = {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_f64("nothing", &[0, 0], &[]).unwrap();
        mb.finish().unwrap()
    };

    let (producer, log) = Ramp::new();
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>("nothing", &[0, 0], Box::new(producer))
        .unwrap();
    mb.write(&path).unwrap();

    assert_eq!(Blocking::plan::<f64>(&[0, 0]).unwrap().block_count, 0);
    assert!(log.lock().unwrap().calls.is_empty());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        materialized,
        "an empty produced dataset must match an empty materialized one"
    );
}

/// Yields a fixed byte pattern, whatever the element type. Used to write the
/// same bytes through the produced and materialized paths.
struct Bytes(Vec<u8>);

impl DataProducer for Bytes {
    fn block_bytes(&self, _block: Block, out: &mut Vec<u8>) -> Result<(), MatError> {
        out.extend_from_slice(&self.0);
        Ok(())
    }
}

/// Every element type must produce the same file its materialized sibling
/// produces — not merely the right `MATLAB_class` string.
///
/// The class attribute alone is far too weak. `BlockElement::datatype` builds
/// each type independently of the `with_*_data` writer it has to match, so the
/// two agree only by parallel construction in two files. A comparison of the
/// *bytes* is what holds them together: it catches a complex pair whose `real`
/// and `imag` members are swapped, and a `logical` built over `i8` instead of
/// `u8`, neither of which changes a single class string.
#[test]
fn every_element_type_matches_its_materialized_sibling() {
    let complex: Vec<(i16, i16)> = vec![(1, -2), (3, -4), (5, -6), (7, -8)];
    let logical: Vec<u8> = vec![1, 0, 1, 1];
    let counts: Vec<u32> = vec![7, 8, 9, 10];

    let raw = |bytes: &[u8]| -> Box<dyn DataProducer> { Box::new(Bytes(bytes.to_vec())) };
    let complex_bytes: Vec<u8> = complex
        .iter()
        .flat_map(|(re, im)| [re.to_le_bytes(), im.to_le_bytes()])
        .flatten()
        .collect();
    let count_bytes: Vec<u8> = counts.iter().flat_map(|c| c.to_le_bytes()).collect();

    let produced = {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_blocks::<(i16, i16)>("iq", &[2, 2], raw(&complex_bytes))
            .unwrap();
        mb.write_blocks::<bool>("flags", &[2, 2], raw(&logical))
            .unwrap();
        mb.write_blocks::<u32>("counts", &[2, 2], raw(&count_bytes))
            .unwrap();
        mb.finish().unwrap()
    };

    let materialized = {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_complex_i16("iq", &[2, 2], &complex).unwrap();
        mb.write_logical("flags", &[2, 2], &logical).unwrap();
        mb.write_u32("counts", &[2, 2], &counts).unwrap();
        mb.finish().unwrap()
    };

    assert_eq!(
        produced, materialized,
        "a produced dataset must be byte-for-byte its materialized sibling, for every element type"
    );

    // And the classes are what MATLAB reads them as, which the byte comparison
    // would also catch but says nothing about on its own.
    let dir = tempdir().unwrap();
    let path = dir.path().join("classes.mat");
    std::fs::write(&path, &produced).unwrap();
    assert_eq!(read_class(&path, "iq"), "int16");
    assert_eq!(read_class(&path, "flags"), "logical");
    assert_eq!(read_class(&path, "counts"), "uint32");
}

/// A produced dataset must be the same file whichever way it is delivered, so
/// the two features compose rather than interacting.
#[test]
fn producing_and_streaming_compose() {
    const DIMS: [usize; 2] = [4, 50_000];

    let buffered = {
        let (producer, _) = Ramp::new();
        let mut mb = MatBuilder::new(Options::default());
        mb.write_blocks::<f64>("s", &DIMS, Box::new(producer))
            .unwrap();
        mb.finish().unwrap()
    };
    let mut streamed = Vec::new();
    {
        let (producer, _) = Ramp::new();
        let mut mb = MatBuilder::new(Options::default());
        mb.write_blocks::<f64>("s", &DIMS, Box::new(producer))
            .unwrap();
        mb.finish_to(&mut streamed).unwrap();
    }
    assert_eq!(buffered, streamed);
}
