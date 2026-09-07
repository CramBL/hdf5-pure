//! End-to-end tests for `File::open_streaming` (issue #27).
//!
//! Writes a real HDF5 file, then reads every dataset both via the buffered
//! `File::open` and the lazy `File::open_streaming`, asserting identical
//! results. This exercises the whole streaming stack end to end: superblock
//! detection, v2 group path resolution, object-header parsing, and contiguous /
//! Fixed-Array / Extensible-Array chunked data reads — all from a `Read + Seek`
//! source that never buffers the whole file.

use hdf5_pure::{ChunkCacheConfig, File, FileAccessProperties, FileBuilder, MetadataCacheConfig};

#[test]
fn open_streaming_matches_buffered() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("streaming.h5");

    let contig: Vec<f64> = (0..100).map(|i| i as f64 * 1.5).collect();
    let fixed_chunked: Vec<i32> = (0..1000).collect();
    let unlimited_chunked: Vec<i32> = (0..500).map(|i| i * 3).collect();
    let inner: Vec<f64> = vec![10.0, 20.0, 30.0];

    {
        let mut b = FileBuilder::new();
        b.create_dataset("contig")
            .with_f64_data(&contig)
            .with_shape(&[100]);
        // Fixed shape + chunks -> Fixed Array (or implicit) chunk index.
        b.create_dataset("fixed_chunked")
            .with_i32_data(&fixed_chunked)
            .with_shape(&[1000])
            .with_chunks(&[64]);
        // Unlimited dimension -> Extensible Array chunk index.
        b.create_dataset("unlimited_chunked")
            .with_i32_data(&unlimited_chunked)
            .with_shape(&[500])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        // Nested group + dataset exercises v2 group path resolution.
        let mut g = b.create_group("grp");
        g.create_dataset("inner")
            .with_f64_data(&inner)
            .with_shape(&[3]);
        b.add_group(g.finish());
        b.write(&path).unwrap();
    }

    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();

    for name in ["contig", "fixed_chunked", "unlimited_chunked", "grp/inner"] {
        let b_shape = buffered.dataset(name).unwrap().shape().unwrap();
        let s_shape = streaming.dataset(name).unwrap().shape().unwrap();
        assert_eq!(b_shape, s_shape, "shape mismatch for {name}");
    }

    // Contiguous f64.
    assert_eq!(
        streaming.dataset("contig").unwrap().read_f64().unwrap(),
        contig
    );
    // Fixed-Array-indexed chunked i32.
    assert_eq!(
        streaming
            .dataset("fixed_chunked")
            .unwrap()
            .read_i32()
            .unwrap(),
        fixed_chunked
    );
    // Extensible-Array-indexed chunked i32.
    assert_eq!(
        streaming
            .dataset("unlimited_chunked")
            .unwrap()
            .read_i32()
            .unwrap(),
        unlimited_chunked
    );
    // Dataset reached through a nested group.
    assert_eq!(
        streaming.dataset("grp/inner").unwrap().read_f64().unwrap(),
        inner
    );

    // And the streaming reads match the buffered reads byte-for-byte.
    assert_eq!(
        buffered.dataset("contig").unwrap().read_f64().unwrap(),
        streaming.dataset("contig").unwrap().read_f64().unwrap()
    );
    assert_eq!(
        buffered
            .dataset("unlimited_chunked")
            .unwrap()
            .read_i32()
            .unwrap(),
        streaming
            .dataset("unlimited_chunked")
            .unwrap()
            .read_i32()
            .unwrap()
    );
}

#[test]
fn open_streaming_with_access_properties_reads_chunked_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("streaming_options.h5");
    let data: Vec<i32> = (0..256).map(|i| i * 2).collect();

    {
        let mut b = FileBuilder::new();
        b.create_dataset("chunked")
            .with_i32_data(&data)
            .with_shape(&[256])
            .with_chunks(&[32]);
        b.write(&path).unwrap();
    }

    let properties = FileAccessProperties::new()
        .with_metadata_cache(MetadataCacheConfig::new(4096).with_max_entry_bytes(512))
        .with_chunk_cache(ChunkCacheConfig::disabled());
    let file = File::open_streaming_with_options(&path, properties).unwrap();
    assert_eq!(file.access_properties(), properties);

    let dataset = file.dataset("chunked").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), data);
    assert_eq!(dataset.read_i32().unwrap(), data);
}

/// Build a file of `count` small datasets, enough distinct object headers that
/// a metadata cache has something to hold and a tight budget something to evict.
fn write_many_datasets(path: &std::path::Path, count: usize) {
    let mut b = FileBuilder::new();
    let values: Vec<i32> = (0..32).collect();
    for i in 0..count {
        b.create_dataset(&format!("d{i:03}"))
            .with_i32_data(&values)
            .with_shape(&[32]);
    }
    b.write(path).unwrap();
}

/// The metadata-cache budget is a number chosen before a single read has
/// happened; these are the figures that say whether it was the right one
/// (issue #353).
#[test]
fn metadata_cache_stats_report_what_the_budget_bought() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mdc_stats.h5");
    const DATASETS: usize = 32;
    write_many_datasets(&path, DATASETS);

    let read_every = |file: &File| {
        for i in 0..DATASETS {
            file.dataset(&format!("d{i:03}"))
                .unwrap()
                .read_i32()
                .unwrap();
        }
    };

    // No budget: nothing to report. `None` rather than an all-zero snapshot,
    // which would read as a cache that is on and doing nothing.
    let uncached = File::open_streaming(&path).unwrap();
    read_every(&uncached);
    assert_eq!(uncached.metadata_cache_stats(), None);
    // A buffered open holds the whole file already, so a metadata cache would
    // be a second copy of bytes it has.
    assert_eq!(File::open(&path).unwrap().metadata_cache_stats(), None);

    const BUDGET: usize = 1 << 20;
    let file = File::open_streaming_with_options(
        &path,
        FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(BUDGET)),
    )
    .unwrap();

    read_every(&file);
    let warm = file
        .metadata_cache_stats()
        .expect("a budget was set, so there is a cache to report");
    assert!(warm.misses() > 0, "the first pass is what populates it");
    assert!(warm.entries() > 0 && warm.bytes() > 0);
    assert!(warm.bytes() <= BUDGET, "the budget is a bound: {warm:?}");
    assert_eq!(
        warm.evictions(),
        0,
        "1 MiB holds this file's metadata whole: {warm:?}"
    );
    assert_eq!(
        warm.invalidations(),
        0,
        "a read-only open writes nothing to invalidate against: {warm:?}"
    );

    // The reads that fill a cache miss by definition, so a rate taken over the
    // whole run charges the steady state for the warm-up. Reset, then measure
    // the part that repeats.
    file.reset_metadata_cache_stats();
    let cleared = file.metadata_cache_stats().unwrap();
    assert_eq!(cleared.reads(), 0);
    assert_eq!(cleared.hit_rate(), None);
    assert_eq!(
        (cleared.entries(), cleared.bytes()),
        (warm.entries(), warm.bytes()),
        "a reset clears counters and evicts nothing"
    );

    read_every(&file);
    let steady = file.metadata_cache_stats().unwrap();
    assert!(steady.reads() > 0);
    assert_eq!(
        steady.hit_rate(),
        Some(1.0),
        "a second pass over the same objects re-reads the same windows, and the \
         budget held every one of them: {steady:?}"
    );
}

/// The other half of the same question: a budget too small to hold the working
/// set says so, in the one figure that means "raising this would help".
#[test]
fn a_budget_too_small_for_the_file_reports_evictions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mdc_tight.h5");
    const DATASETS: usize = 32;
    write_many_datasets(&path, DATASETS);

    const BUDGET: usize = 1024;
    let file = File::open_streaming_with_options(
        &path,
        FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(BUDGET)),
    )
    .unwrap();
    for i in 0..DATASETS {
        file.dataset(&format!("d{i:03}"))
            .unwrap()
            .read_i32()
            .unwrap();
    }

    let stats = file.metadata_cache_stats().unwrap();
    assert!(
        stats.evictions() > 0,
        "32 datasets do not fit in 1 KiB of metadata: {stats:?}"
    );
    assert!(
        stats.bytes() <= BUDGET,
        "the budget still bounds it: {stats:?}"
    );
}

/// Recursively assert the streaming backend reports the identical groups,
/// datasets, shapes, and attributes as the buffered backend for the group at
/// `path`. Returns the number of attributes compared, so a caller can confirm
/// the walk actually exercised attribute reads rather than passing trivially on
/// empty maps.
fn assert_group_parity(buffered: &File, streaming: &File, path: &str) -> usize {
    let display = if path.is_empty() { "/" } else { path };
    let bg = if path.is_empty() {
        buffered.root()
    } else {
        buffered.group(path).unwrap()
    };
    let sg = if path.is_empty() {
        streaming.root()
    } else {
        streaming.group(path).unwrap()
    };

    let b_attrs = bg.attrs().unwrap();
    let s_attrs = sg.attrs().unwrap();
    assert_eq!(b_attrs, s_attrs, "group attrs mismatch at '{display}'");
    let mut count = b_attrs.len();

    let mut b_ds = bg.datasets().unwrap();
    b_ds.sort();
    let mut s_ds = sg.datasets().unwrap();
    s_ds.sort();
    assert_eq!(b_ds, s_ds, "datasets mismatch at '{display}'");
    for name in &b_ds {
        let full = child_path(path, name);
        let bd = buffered.dataset(&full).unwrap();
        let sd = streaming.dataset(&full).unwrap();
        assert_eq!(
            bd.shape().unwrap(),
            sd.shape().unwrap(),
            "shape mismatch for '{full}'"
        );
        let bda = bd.attrs().unwrap();
        let sda = sd.attrs().unwrap();
        assert_eq!(bda, sda, "dataset attrs mismatch for '{full}'");
        count += bda.len();
    }

    let mut b_g = bg.groups().unwrap();
    b_g.sort();
    let mut s_g = sg.groups().unwrap();
    s_g.sort();
    assert_eq!(b_g, s_g, "subgroups mismatch at '{display}'");
    for name in &b_g {
        count += assert_group_parity(buffered, streaming, &child_path(path, name));
    }

    count
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

/// The streaming backend must resolve v1 (symbol-table) groups and read
/// compact, dense (fractal-heap), and variable-length attributes identically to
/// the buffered backend. Each fixture's expected attribute count guards against
/// a trivially-empty parity pass.
#[test]
fn streaming_matches_buffered_groups_and_attributes_across_fixtures() {
    // (fixture, expected total attributes across the whole walk)
    let cases = [
        ("two_groups.h5", 0),        // v1 groups, no attributes
        ("nested_groups.h5", 0),     // nested v1 groups
        ("simple_dataset.h5", 0),    // v1 root group + dataset
        ("attrs.h5", 4),             // v1 groups + compact attrs (root + dataset)
        ("mixed_attrs.h5", 3),       // v1 subgroup + scalar/array compact attrs
        ("vl_strings.h5", 1),        // v1 root + VL-string attr (global heap)
        ("dense_attrs.h5", 50),      // v2 dataset + dense (fractal-heap) attrs
        ("dense_attrs_root.h5", 20), // v2 root group + dense attrs
        ("v2_groups.h5", 0),         // v2 groups, no attributes
    ];

    for (fixture, expected_attrs) in cases {
        let path = format!("tests/data/unattributed/{fixture}");
        let buffered = File::open(&path).unwrap();
        let streaming = File::open_streaming(&path).unwrap();
        let counted = assert_group_parity(&buffered, &streaming, "");
        assert_eq!(
            counted, expected_attrs,
            "attribute count mismatch for {fixture}"
        );
    }
}

/// `File::from_source` reads what `open_streaming` reads, and reads no more of
/// the file than the work needs.
///
/// The point of the entry is a caller whose bytes are not a path — an object
/// store, or a guest that receives byte ranges from its host — so the source
/// here is neither a file nor a slice the reader could have mapped: it serves
/// the bytes from behind a counter.
///
/// That counter is what makes this test load-bearing rather than decorative. A
/// read that quietly materialized the file would still return the right
/// numbers; only the byte count separates that from reading on demand.
#[test]
fn from_source_matches_streaming_and_reads_on_demand() {
    use hdf5_pure::{FormatError, Source};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Counting {
        bytes: Vec<u8>,
        served: Arc<AtomicU64>,
    }

    impl Source for Counting {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            let at = usize::try_from(offset).expect("the fixture fits this platform");
            let end = at + buf.len();
            if end > self.bytes.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: end,
                    available: self.bytes.len(),
                });
            }
            buf.copy_from_slice(&self.bytes[at..end]);
            self.served.fetch_add(buf.len() as u64, Ordering::Relaxed);
            Ok(())
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("from_source.h5");

    // Rows enough that one window is a small share of the file: the read below
    // asks for 64 of 8,000, each row 32 columns of f64.
    let rows = 8_000usize;
    let cols = 32usize;
    let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("frames")
            .with_f64_data(&data)
            .with_shape(&[rows as u64, cols as u64])
            .with_chunks(&[64, cols as u64]);
        b.write(&path).unwrap();
    }

    let whole = std::fs::read(&path).unwrap();
    let served = Arc::new(AtomicU64::new(0));
    let file = File::from_source(Box::new(Counting {
        bytes: whole.clone(),
        served: Arc::clone(&served),
    }))
    .unwrap();

    let window = file
        .dataset("frames")
        .unwrap()
        .read_f64_rows(100, 64)
        .unwrap();
    assert_eq!(
        window,
        data[100 * cols..164 * cols],
        "a source read differs from what was written"
    );

    let by_path = File::open_streaming(&path)
        .unwrap()
        .dataset("frames")
        .unwrap()
        .read_f64_rows(100, 64)
        .unwrap();
    assert_eq!(window, by_path, "a source read differs from a path read");

    let served = served.load(Ordering::Relaxed);
    assert!(
        served < whole.len() as u64 / 4,
        "served {served} of {} bytes: the source was drained, not read on demand",
        whole.len()
    );
    // And the floor: the window's own bytes came through this source, so a
    // count that collapses toward zero means the counter stopped counting, not
    // that the read got cheaper.
    let window_bytes = (64 * cols * size_of::<f64>()) as u64;
    assert!(
        served >= window_bytes,
        "served {served} bytes, fewer than the {window_bytes} the window itself is: \
         the count is not measuring the read"
    );
}

/// A source that returns fewer bytes than it was asked for is refused, not
/// followed into a parser.
///
/// `Source::read_exact_at` and `read_metadata_at` have default bodies that
/// cannot come back short, but they are overridable, and a source that batches
/// or coalesces remote reads is exactly the caller with a reason to override
/// them. The parsers index what comes back at offsets computed from the length
/// they *asked* for, so an unchecked short buffer is a slice panic inside the
/// object-header parser, blaming the file format for what the source did.
#[test]
fn a_source_that_returns_a_short_buffer_is_refused() {
    use hdf5_pure::{FormatError, Source};

    struct Short(Vec<u8>);

    impl Source for Short {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            let at = usize::try_from(offset).expect("the fixture fits this platform");
            let end = at + buf.len();
            if end > self.0.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: end,
                    available: self.0.len(),
                });
            }
            buf.copy_from_slice(&self.0[at..end]);
            Ok(())
        }

        /// One byte short of the request, which is the whole defect.
        fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            let at = usize::try_from(offset).expect("the fixture fits this platform");
            let end = (at + len).min(self.0.len()).saturating_sub(1);
            Ok(self.0[at.min(end)..end].to_vec())
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short_reads.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("values")
            .with_f64_data(&(0..256).map(f64::from).collect::<Vec<_>>())
            .with_shape(&[256]);
        b.write(&path).unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();

    // Whichever read reaches the short buffer first, the answer is an error
    // that names the source rather than a panic inside a parser.
    let err = match File::from_source(Short(bytes)) {
        Err(err) => err,
        Ok(file) => file
            .dataset("values")
            .and_then(|d| d.read_f64())
            .expect_err("a source returning short buffers read a dataset successfully"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("bytes for a") && msg.contains("read at offset"),
        "the refusal does not say the source came back short: {msg}"
    );
}
