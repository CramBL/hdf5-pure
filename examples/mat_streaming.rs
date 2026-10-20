//! Writing a MATLAB v7.3 `.mat` file without holding it, or its data, in memory.
//!
//! Two independent pieces, shown together because they compose:
//!
//! - [`MatBuilder::write_blocks`] stages a dataset whose bytes a [`DataProducer`]
//!   supplies one block at a time *during the write*. Layout works from the shape
//!   alone, so the producer is never called before emission begins.
//! - [`MatBuilder::finish_to`] assembles the file straight onto an `io::Write`
//!   instead of returning it, front-to-back and never seeking.
//!
//! Together they write a `.mat` of any size in about one block of memory. The
//! file is byte-for-byte what the ordinary buffered calls produce, which this
//! example checks rather than asserts.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example mat_streaming
//! ```

use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::mat::{Block, Blocking, DataProducer, MatBuilder, MatError, Options};
use std::sync::{Arc, Mutex};

/// Stands in for an acquisition: yields interleaved channel samples computed
/// from their position, so a dataset of any size costs nothing to generate.
///
/// A real one would drain a capture buffer, read a socket, or `mmap` a raw file.
/// The only contract is that block `i` continues where block `i - 1` stopped, and
/// that it writes exactly [`Blocking::block_len`] bytes.
struct Acquisition {
    channels: u64,
    /// Records what the writer asked for, so `main` can show the call pattern.
    calls: Arc<Mutex<Vec<usize>>>,
}

impl Acquisition {
    /// Sample value for channel `c` at timestep `t`.
    fn sample(channel: u64, step: u64) -> f64 {
        step as f64 + channel as f64 / 10.0
    }
}

impl DataProducer for Acquisition {
    fn block_bytes(&self, block: Block, out: &mut Vec<u8>) -> Result<(), MatError> {
        self.calls.lock().expect("not poisoned").push(block.index);

        // The block says where it starts and how much to write, so this producer
        // holds no position of its own. Because the shape is
        // `[channels, samples]`, that linear order is channel-interleaved and
        // runs forward through time.
        for i in 0..block.elements {
            let element = block.first_element + i;
            let value = Self::sample(element % self.channels, element / self.channels);
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("capture.mat");

    const CHANNELS: usize = 4;
    const SAMPLES: usize = 500_000;
    // `[channels, samples]`, not the transpose. MATLAB is column-major, so this
    // shape puts one timestep's channels next to each other and the blocks run
    // forward through time — the order an acquisition produces them in. The
    // transpose would store channel 0's entire history first, which no producer
    // can emit as a capture proceeds.
    let dims = [CHANNELS, SAMPLES];

    // ---- Plan the blocking before building the producer -----------------
    // `Blocking::plan` is deterministic and derived from the shape alone, so the
    // producer can be built against exactly the split the writer will use.
    let blocking = Blocking::plan::<f64>(&dims).expect("a plannable shape");
    println!(
        "{} elements in {} blocks of {} bytes ({} in the last)",
        CHANNELS * SAMPLES,
        blocking.block_count,
        blocking.block_len(0),
        blocking.block_len(blocking.block_count - 1),
    );

    // ---- Stage the dataset, then stream the file to disk -----------------
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mb = MatBuilder::new(Options::default());
    mb.write_blocks::<f64>(
        "samples",
        &dims,
        Box::new(Acquisition {
            channels: CHANNELS as u64,
            calls: Arc::clone(&calls),
        }),
    )
    .expect("staging a produced dataset")
    // Ordinary writers still work alongside it, and chain with it; only this one
    // dataset is lazy.
    .write_char("units", "volts")
    .expect("a char array");

    assert!(
        calls.lock().expect("not poisoned").is_empty(),
        "staging must not touch the producer: layout works from the shape alone"
    );

    // `write` is `finish_to` onto a file: the whole `.mat` is assembled straight
    // to disk, so the output never becomes an in-memory `Vec<u8>` either.
    mb.write(&path).expect("streaming the file to disk");

    let calls = calls.lock().expect("not poisoned").clone();
    println!(
        "producer called {} times, first {:?}, ascending: {}",
        calls.len(),
        &calls[..3.min(calls.len())],
        calls.windows(2).all(|w| w[0] < w[1]),
    );

    // ---- The file is an ordinary .mat -----------------------------------
    let file = File::open(&path).expect("opening what we wrote");
    let ds = file.dataset("samples").expect("the dataset");
    // HDF5 stores MATLAB dimensions reversed, so `[4, 500_000]` reads back as
    // `[500_000, 4]` on disk and as `[4, 500_000]` in MATLAB.
    assert_eq!(
        ds.shape().expect("a shape"),
        vec![SAMPLES as u64, CHANNELS as u64]
    );

    let values = ds.read_f64().expect("reading it back");
    assert_eq!(values.len(), CHANNELS * SAMPLES);
    assert_eq!(values[0], Acquisition::sample(0, 0));
    assert_eq!(values[1], Acquisition::sample(1, 0));
    assert_eq!(values[CHANNELS], Acquisition::sample(0, 1));
    // Straddle a block boundary, where an off-by-one would show as a jump.
    let edge = blocking.block_elements as usize;
    assert_eq!(
        values[edge],
        Acquisition::sample(edge as u64 % CHANNELS as u64, edge as u64 / CHANNELS as u64)
    );

    println!("read back {} samples across {} channels", SAMPLES, CHANNELS);

    // ---- Same bytes as the buffered path --------------------------------
    // The point of the produced path is that it is not a *different* way to
    // write the file, only a cheaper one. Building the same content the ordinary
    // way must give the identical file.
    let materialized = {
        let mut all = Vec::with_capacity(CHANNELS * SAMPLES);
        for step in 0..SAMPLES as u64 {
            for channel in 0..CHANNELS as u64 {
                all.push(Acquisition::sample(channel, step));
            }
        }
        let mut mb = MatBuilder::new(Options::default());
        mb.write_f64("samples", &dims, &all).expect("a plain write");
        mb.write_char("units", "volts").expect("a char array");
        mb.finish().expect("the buffered file")
    };
    assert_eq!(
        materialized,
        std::fs::read(&path).expect("re-reading the file"),
        "a produced dataset must be byte-for-byte a materialized one"
    );
    println!("byte-identical to the same content written the ordinary way");

    // ---- Compression is refused, not silently dropped -------------------
    // A block's on-disk size has to be known before any byte is written, and a
    // compressed one is not knowable without compressing it.
    let mut options = Options::default();
    options.compression = hdf5_pure::mat::Compression::Deflate {
        level: DeflateLevel::Six,
        shuffle: false,
    };
    let mut mb = MatBuilder::new(options);
    let refused = mb
        .write_blocks::<f64>(
            "x",
            &[1, 4],
            Box::new(Acquisition {
                channels: 1,
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
        )
        .map(|_| ());
    match refused {
        Err(MatError::CompressionUnsupportedForBlocks) => {
            println!("compression refused up front, as documented");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}
