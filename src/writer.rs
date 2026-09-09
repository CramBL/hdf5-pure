//! Writing API: FileBuilder and GroupBuilder for creating HDF5 files.

use std::io::Write;

use crate::chunked_write::ByteSink;
use crate::file_writer::FileWriter as FormatWriter;
use crate::type_builders::{
    AttrValue, DatasetBuilder as FormatDatasetBuilder, FinishedGroup,
    GroupBuilder as FormatGroupBuilder,
};

use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::file_create_properties::FileCreateProperties;
use crate::file_space_info::FileSpaceStrategy;
use crate::libver::LibVer;

/// Builder for creating a new HDF5 file.
///
/// # Example
///
/// ```no_run
/// use hdf5_pure::FileBuilder;
/// use hdf5_pure::AttrValue;
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
/// builder.set_attr("version", AttrValue::I64(1));
/// builder.write("output.h5").unwrap();
/// ```
pub struct FileBuilder {
    writer: FormatWriter,
}

impl FileBuilder {
    /// Create a new file builder.
    pub fn new() -> Self {
        Self {
            writer: FormatWriter::new(),
        }
    }

    /// Create a dataset at the root level. Returns a mutable reference to
    /// a `DatasetBuilder` for configuring data, shape, and attributes.
    pub fn create_dataset(&mut self, name: &str) -> &mut FormatDatasetBuilder {
        self.writer.create_dataset(name)
    }

    /// Create a group builder. Call `.finish()` on the returned builder
    /// to complete it, then pass to `add_group()`.
    pub fn create_group(&mut self, name: &str) -> FormatGroupBuilder {
        self.writer.create_group(name)
    }

    /// Add a finished group to the file.
    pub fn add_group(&mut self, group: FinishedGroup) {
        self.writer.add_group(group);
    }

    /// Commit `datatype` in the root group under `name`, the way `H5Tcommit`
    /// does: the type is written as an object of its own, and datasets and
    /// attributes reference it by path instead of encoding it again.
    ///
    /// A committed datatype is what a C-library reader reports by name — `h5dump`
    /// prints `DATATYPE "/mytype"` for a dataset using one — and what netCDF-4
    /// writes for every user-defined type. It is also the only way several
    /// objects in a file can be said to share *one* type rather than to each
    /// declare an identical one.
    ///
    /// Name it from a dataset with
    /// [`DatasetBuilder::with_committed_datatype`](crate::DatasetBuilder::with_committed_datatype)
    /// or from an attribute with
    /// [`DatasetBuilder::set_attr_committed`](crate::DatasetBuilder::set_attr_committed)
    /// and its group and root counterparts. A name that no committed datatype
    /// matches, or one whose type disagrees with the naming object's, fails the
    /// write rather than producing a file whose element bytes and declared type
    /// do not match.
    ///
    /// ```
    /// use hdf5_pure::{FileBuilder, make_i32_type};
    ///
    /// let mut b = FileBuilder::new();
    /// b.commit_datatype("mytype", make_i32_type());
    /// b.create_dataset("d")
    ///     .with_i32_data(&[1, 2, 3])
    ///     .with_committed_datatype("mytype");
    /// let bytes = b.finish().unwrap();
    /// # assert!(hdf5_pure::is_hdf5_bytes(&bytes));
    /// ```
    pub fn commit_datatype(&mut self, name: &str, datatype: Datatype) {
        self.writer.commit_datatype(name, datatype);
    }

    /// Attach a root-group attribute whose datatype is the committed one at
    /// `path`. See [`commit_datatype`](Self::commit_datatype) and
    /// [`DatasetBuilder::set_attr_committed`](crate::DatasetBuilder::set_attr_committed).
    pub fn set_attr_committed(&mut self, name: &str, value: AttrValue, path: &str) {
        self.writer.set_root_attr_committed(name, value, path);
    }

    /// Apply every creation property in `properties` at once — the `fcpl`
    /// analogue of handing a property list to `H5Fcreate`.
    ///
    /// Each property is applied exactly as the individual setter would, so this
    /// **overwrites** any value set individually before the call — including the
    /// properties `properties` leaves unset, which are reset to their defaults
    /// rather than left behind. The two spellings interoperate in the order that
    /// says so: apply a shared [`FileCreateProperties`] first, then override one
    /// property for this file.
    ///
    /// The reset matters most for the library-version bounds, which select the
    /// on-disk format rather than merely validating it: a stale 1.8 bound
    /// surviving a property list that names no version would decide the bytes
    /// this file is written in.
    ///
    /// ```
    /// use hdf5_pure::{FileBuilder, FileCreateProperties, LibVer};
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    /// // The list names no version, so the bound above is dropped with it.
    /// builder.with_create_properties(FileCreateProperties::new().with_userblock(512));
    /// builder.create_dataset("values").with_f64_data(&[1.0]);
    ///
    /// let bytes = builder.finish().unwrap();
    /// assert_eq!(bytes[512 + 8], 3); // the default format, not the 1.8 one
    /// ```
    #[doc(alias = "fcpl")]
    pub fn with_create_properties(&mut self, properties: FileCreateProperties) -> &mut Self {
        self.writer.apply_create_properties(&properties);
        self
    }

    /// Set the userblock size in bytes: zero (no userblock), or a power of two of
    /// at least 512. The region is filled with zeros.
    ///
    /// Any other size is refused by [`finish`](Self::finish) /
    /// [`finish_to`](Self::finish_to) / [`write`](Self::write) with
    /// [`FormatError::InvalidUserblockSize`].
    /// The size *is* the superblock's base address, and a reader scans for the
    /// signature at 0, 512, 1024, and so on doubling — so an unaligned size would
    /// hide the superblock where nothing looks for it.
    ///
    /// To put something in it, prefer
    /// [`with_userblock_content`](Self::with_userblock_content), which works on
    /// every output path. Patching the bytes afterwards only works with the
    /// buffered [`finish`](Self::finish); the streaming
    /// [`finish_to`](Self::finish_to) / [`write`](Self::write) have already emitted
    /// the region by the time they return.
    pub fn with_userblock(&mut self, size: u64) -> &mut Self {
        self.writer.with_userblock(size);
        self
    }

    /// Set the bytes that occupy the head of the userblock region, so the writer
    /// emits them as part of the file. The remainder of the region stays
    /// zero-filled, and content longer than the userblock set by
    /// [`with_userblock`](Self::with_userblock) is refused by every output path —
    /// [`finish`](Self::finish), [`finish_to`](Self::finish_to), and
    /// [`write`](Self::write) — with
    /// [`FormatError::UserblockContentTooLarge`].
    ///
    /// Because the userblock leads the file in address order, this is what lets a
    /// wrapper format's header — MATLAB v7.3's, for instance — be produced by the
    /// non-seekable [`finish_to`](Self::finish_to) with no second pass.
    ///
    /// # Example
    ///
    /// ```
    /// use hdf5_pure::FileBuilder;
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_userblock(512);
    /// builder.with_userblock_content(b"my wrapper format's header");
    /// builder.create_dataset("x").with_f64_data(&[1.0, 2.0]);
    ///
    /// let bytes = builder.finish().unwrap();
    /// assert_eq!(&bytes[..26], b"my wrapper format's header");
    /// // The rest of the region is zero-filled, and the HDF5 signature follows it.
    /// assert!(bytes[26..512].iter().all(|&b| b == 0));
    /// assert_eq!(&bytes[512..516], b"\x89HDF");
    /// ```
    pub fn with_userblock_content(&mut self, content: &[u8]) -> &mut Self {
        self.writer.with_userblock_content(content);
        self
    }

    /// Constrain the on-disk format version of the file, mirroring HDF5's
    /// `H5Pset_libver_bounds`. The file is written in the newest format the
    /// bounds allow, between [`LibVer::WRITER_OLDEST`] and
    /// [`LibVer::WRITER_DEFAULT`]; bounds that leave no such format fail with
    /// [`Error::Format`] wrapping [`FormatError::LibverBoundsUnsatisfiable`].
    ///
    /// `high` selects the format. `Earliest..=V18` writes the HDF5 1.8 format —
    /// a version 2 superblock and version 3 data-layout messages — and anything
    /// reaching 1.10 writes the 1.10 one. That is what a file destined for an
    /// older reader wants: MATLAB's MAT v7.3 loader, for instance, is HDF5
    /// 1.8.12 before R2021b, which does not understand a version 3 superblock.
    ///
    /// `low` only rules formats out: as in the C library it licenses newer
    /// encodings without requiring them, so a lower bound of `V112`, `V114` or
    /// `LATEST` is satisfied by the 1.10 format rather than refused. It does not
    /// license `high` away — an inverted range such as `V114..=V110` is refused
    /// with
    /// [`FormatError::LibverBoundsUnsatisfiable`],
    /// as `H5Pset_libver_bounds` refuses one.
    ///
    /// Content the 1.8 format cannot express is refused rather than silently
    /// upgraded, with
    /// [`FormatError::LibverTooOldForContent`]:
    /// a chunked, filtered, or resizable dataset needs the 1.10 chunk indices,
    /// and a file-space setting — a strategy or a page size — needs the 1.10
    /// File Space Info message.
    /// [`File::open_swmr_writer`](crate::File::open_swmr_writer) likewise needs
    /// a version 3 superblock, so a file written to the 1.8 bound cannot host a
    /// SWMR writer.
    ///
    /// ```
    /// use hdf5_pure::{FileBuilder, LibVer};
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    /// builder.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
    /// let bytes = builder.finish().unwrap();
    /// assert_eq!(bytes[8], 2); // version 2 superblock, readable by HDF5 1.8
    /// ```
    ///
    /// This differs from the C library, which picks the *oldest* format the
    /// content needs and reads `low` as a floor; on `Earliest..=Latest`
    /// `H5Fcreate` writes a version 0 superblock where this writes a version 3
    /// one. Leaving the bounds unset is the same as leaving `high` at `Latest`.
    pub fn with_libver_bounds(&mut self, low: LibVer, high: LibVer) -> &mut Self {
        self.writer.with_libver_bounds(low, high);
        self
    }

    /// Set the file-space management strategy, mirroring HDF5's
    /// `H5Pset_file_space_strategy`. The strategy, persist flag, and free-space
    /// section `threshold` are recorded in the file's superblock extension, so
    /// the reference C library and a later reopen observe the choice.
    ///
    /// `persist = true` records that freed space should be tracked on disk across
    /// closes. A brand-new file has nothing to track, so this only records the
    /// intent; freeing space in a later [`File::open_rw`](crate::File::open_rw) then
    /// writes the on-disk free-space-manager blocks that survive a reopen.
    pub fn with_file_space_strategy(
        &mut self,
        strategy: FileSpaceStrategy,
        persist: bool,
        threshold: u64,
    ) -> &mut Self {
        self.writer
            .with_file_space_strategy(strategy, persist, threshold);
        self
    }

    /// Set the file-space page size, mirroring HDF5's
    /// `H5Pset_file_space_page_size`. Recorded in the superblock extension.
    pub fn with_file_space_page_size(&mut self, page_size: u64) -> &mut Self {
        self.writer.with_file_space_page_size(page_size);
        self
    }

    /// Set an attribute on the root group.
    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.writer.set_root_attr(name, value);
    }

    /// Attach an already-encoded attribute message to the root group, written
    /// exactly as given.
    ///
    /// See [`AttrSpec::Verbatim`](crate::type_builders::AttrSpec::Verbatim) for
    /// what this preserves that [`set_attr`](Self::set_attr) cannot, and for the
    /// datatypes it must not be used with.
    pub(crate) fn set_attr_verbatim(&mut self, message: crate::attribute::AttributeMessage) {
        self.writer.set_root_attr_verbatim(message);
    }

    /// Attach a variable-length string attribute to the root group with the given
    /// datatype and dataspace, staging `strings` into a heap of this file's own.
    /// See [`AttrSpec::VerbatimVarLen`](crate::type_builders::AttrSpec::VerbatimVarLen).
    pub(crate) fn set_attr_var_len_verbatim(
        &mut self,
        message: crate::attribute::AttributeMessage,
        strings: Vec<String>,
    ) {
        self.writer.set_root_attr_var_len_verbatim(message, strings);
    }

    /// Whether the staged content needs the 1.10 format — see
    /// [`FileWriter::needs_latest_format`](crate::file_writer::FileWriter::needs_latest_format).
    pub(crate) fn needs_latest_format(&self) -> bool {
        self.writer.needs_latest_format()
    }

    /// Serialize the file to bytes in memory.
    pub fn finish(self) -> Result<Vec<u8>, Error> {
        Ok(self.writer.finish()?)
    }

    /// Serialize the file directly to a [`Write`] sink, without first buffering
    /// the whole file in memory.
    ///
    /// Produces byte-for-byte the same file as [`finish`](Self::finish), but a
    /// dataset staged for verbatim chunk *streaming* (repack's out-of-core path)
    /// has its chunks pulled from the source and written one at a time, so peak
    /// memory stays bounded by a single chunk plus the file metadata rather than
    /// the whole dataset.
    ///
    /// The sink is written front-to-back with no seeks, so it can be a socket
    /// or a pipe as readily as a file. That is possible because the writer
    /// computes every object's address before it emits a byte, rather than
    /// seeking back to patch addresses the way a backpatching writer would.
    ///
    /// A failure partway leaves whatever was already written on the sink. With a
    /// non-seekable sink there is nothing to roll back, so a caller needing
    /// all-or-nothing should write to a temporary path and rename on success.
    ///
    /// # Example
    ///
    /// ```
    /// use hdf5_pure::FileBuilder;
    ///
    /// let build = || {
    ///     let mut b = FileBuilder::new();
    ///     b.create_dataset("x").with_f64_data(&[1.0, 2.0, 3.0]);
    ///     b
    /// };
    ///
    /// let mut streamed: Vec<u8> = Vec::new();
    /// build().finish_to(&mut streamed).unwrap();
    /// assert_eq!(build().finish().unwrap(), streamed);
    /// ```
    pub fn finish_to<W: Write>(self, w: W) -> Result<(), Error> {
        let mut sink = WriteSink::new(std::io::BufWriter::new(w));
        if let Err(fe) = self.writer.finish_to_sink(&mut sink) {
            // If the failure came from the sink's I/O, surface the real
            // `io::Error`; otherwise it is a genuine format error.
            return match sink.err.take() {
                Some(io_err) => Err(Error::Io(io_err)),
                None => Err(Error::Format(fe)),
            };
        }
        sink.into_inner().flush().map_err(Error::Io)
    }

    /// Serialize and write the file to the given path.
    ///
    /// Streams the file to disk (see [`finish_to`](Self::finish_to)), so a repack
    /// staging streamed chunks does not hold the whole output in memory.
    ///
    /// The path is created when the first byte is ready, not when the call
    /// starts, so a build refused before any byte is emitted — unsatisfiable or
    /// too-old library-version bounds, an invalid userblock — leaves whatever was
    /// at `path` untouched. A failure *after* that (an I/O error, or a refusal
    /// the layout reaches) still leaves a partial file, as
    /// [`finish_to`](Self::finish_to) describes.
    pub fn write<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), Error> {
        self.finish_to(LazyFile {
            path: path.as_ref().to_path_buf(),
            file: None,
        })
    }
}

/// A [`Write`] that creates its file on the first byte written to it.
///
/// [`FileBuilder::write`] used `std::fs::File::create` up front, which
/// truncates: a build the writer refuses before emitting anything — the
/// library-version and userblock checks all run there, deliberately — returned
/// its error having already emptied the file at the destination path. The whole
/// point of refusing early is that nothing is destroyed, and that has to include
/// the file the caller is overwriting.
struct LazyFile {
    path: std::path::PathBuf,
    file: Option<std::fs::File>,
}

impl Write for LazyFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let file = match &mut self.file {
            Some(f) => f,
            slot => slot.insert(std::fs::File::create(&self.path)?),
        };
        file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Nothing was written, so there is no file to flush and none to create:
        // a refused build must not leave an empty one behind either.
        match &mut self.file {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

/// Adapts a [`std::io::Write`] to the writer's [`ByteSink`] so a file can be
/// assembled straight onto the sink. Because `ByteSink` is `no_std` and cannot
/// carry a `std::io::Error`, an I/O failure is stashed here and the surrounding
/// [`FileBuilder::finish_to`] turns it back into [`Error::Io`].
struct WriteSink<W: Write> {
    inner: W,
    written: u64,
    err: Option<std::io::Error>,
}

impl<W: Write> WriteSink<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            written: 0,
            err: None,
        }
    }

    fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> ByteSink for WriteSink<W> {
    fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        match self.inner.write_all(bytes) {
            Ok(()) => {
                self.written += bytes.len() as u64;
                Ok(())
            }
            Err(e) => {
                self.err = Some(e);
                // A placeholder format error; `finish_to` replaces it with the
                // stashed `io::Error` above, so its message is never surfaced.
                Err(FormatError::SerializationError(
                    "streaming output write failed".into(),
                ))
            }
        }
    }

    fn put_zeros(&mut self, n: usize) -> Result<(), FormatError> {
        // Emit padding in bounded blocks so a large userblock never allocates a
        // matching buffer.
        const ZEROS: [u8; 4096] = [0u8; 4096];
        let mut remaining = n;
        while remaining > 0 {
            let take = remaining.min(ZEROS.len());
            self.put(&ZEROS[..take])?;
            remaining -= take;
        }
        Ok(())
    }

    fn position(&self) -> u64 {
        self.written
    }
}

impl Default for FileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::chunked_write::{ChunkMeta, ChunkProvider};
    use crate::convert::nz;
    use std::sync::{Arc, Mutex};

    type Calls = Arc<Mutex<Vec<usize>>>;

    /// A test [`ChunkProvider`] serving fixed in-memory chunk bytes, recording
    /// the order of `chunk_bytes` calls so a test can assert the streaming
    /// writer pulls each chunk exactly once, in ascending slot order. With
    /// `short_slot` set, that one slot returns one byte fewer than planned
    /// (size-mismatch). `Arc<Mutex<_>>` (not `Rc<RefCell<_>>`) keeps it
    /// `Send + Sync`, as the `ChunkProvider` supertrait requires.
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

    /// Stage one lazily-streamed, unfiltered chunked f64 dataset named `name` on
    /// `b`. Unfiltered means the "compressed" bytes are the raw element bytes, so
    /// the produced file is a plain chunked f64 dataset that reads back.
    fn stage_lazy(
        b: &mut FileBuilder,
        name: &str,
        chunk_bytes: Vec<Vec<u8>>,
        dims: &[u64],
        chunk_dims: &[u64],
        maxshape: Option<&[u64]>,
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
            nz(8),
            None,
            meta,
            Box::new(provider),
        );
    }

    /// Build a file with one lazily-streamed chunked f64 dataset named `d`.
    fn build_lazy(
        chunk_bytes: Vec<Vec<u8>>,
        dims: &[u64],
        chunk_dims: &[u64],
        maxshape: Option<&[u64]>,
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

        // The streaming (io::Write) path and the buffered (Vec) path must produce
        // byte-for-byte the same file.
        assert_eq!(
            buffered, streamed,
            "streamed output must be byte-identical to buffered output"
        );
        // Each chunk is pulled exactly once, in ascending slot order — i.e. the
        // writer streams chunk-by-chunk rather than collecting them all.
        assert_eq!(*calls_buf.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(*calls_str.lock().unwrap(), vec![0, 1, 2]);
        // And the file reads back to the original values.
        assert_eq!(
            read_back_f64(&buffered, "d"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn file_builder_keeps_its_auto_traits() {
        // The lazy chunk provider is boxed into `FileBuilder`; a bare boxed trait
        // object would strip `Send`/`Sync` (fixed by the `ChunkProvider`
        // supertrait) and `UnwindSafe`/`RefUnwindSafe` (fixed by wrapping it in
        // `AssertUnwindSafe`). Removing any of these auto-trait impls is a semver
        // break that `cargo-semver-checks` enforces in CI, so pin all four here.
        fn assert_auto_traits<
            T: Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe,
        >() {
        }
        assert_auto_traits::<FileBuilder>();
    }

    #[test]
    fn streaming_writer_rejects_provider_size_mismatch() {
        // A provider returning fewer bytes than planned — on slot 0 *or* any later
        // slot — must be rejected rather than written as a corrupt file.
        for short_slot in [0usize, 2] {
            let chunks = vec![
                f64_chunk(&[1.0, 2.0]),
                f64_chunk(&[3.0, 4.0]),
                f64_chunk(&[5.0, 6.0]),
            ];
            let calls = Arc::new(Mutex::new(Vec::new()));
            let err = build_lazy(chunks, &[6], &[2], None, calls, Some(short_slot))
                .finish()
                .unwrap_err();
            match err {
                Error::Format(FormatError::ChunkedReadError(_)) => {}
                other => panic!("slot {short_slot}: expected ChunkedReadError, got {other:?}"),
            }
        }
    }

    /// Assert the buffered and streamed outputs are byte-identical for one chunked
    /// layout, and that the produced file reads back to `expected`.
    fn assert_variant_streams_identically(
        chunks: Vec<Vec<u8>>,
        dims: &[u64],
        chunk_dims: &[u64],
        maxshape: Option<&[u64]>,
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
        // The streamed file decodes to the expected values (not merely parses).
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
        assert_variant_streams_identically(
            vec![f64_chunk(&[1.0, 2.0])],
            &[2],
            &[2],
            None,
            &[1.0, 2.0],
        );
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
            Some(&[u64::MAX]),
            &[1.0, 2.0, 3.0, 4.0],
        );
    }

    /// A `Write` that accepts `limit` bytes total, then fails every later write —
    /// to exercise the streaming I/O-error path.
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
        // A dataset large enough to exceed the internal BufWriter so writes occur
        // mid-stream; the sink fails partway and `finish_to` must surface it as
        // `Error::Io`, not a format error or a panic.
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
        assert!(
            matches!(err, Error::Io(_)),
            "a failing sink must surface as Error::Io, got {err:?}"
        );
    }

    #[test]
    fn streamed_dataset_with_attribute_and_contiguous_sibling() {
        // One file mixing a streamed (lazy chunked) dataset that also carries an
        // attribute, a plain contiguous dataset, and a zero-element contiguous
        // dataset — exercising the assembly loop's InMemory + Streamed dispatch and
        // attribute handling together. Buffered and streamed must agree and read
        // back.
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
                    nz(8),
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

    /// Userblock content is part of the file the writer emits, so both output
    /// paths carry it — which is the whole point of the setter, since the
    /// streaming path has no bytes left to patch by the time it returns.
    #[test]
    fn userblock_content_leads_the_file_on_both_output_paths() {
        const HEADER: &[u8] = b"a wrapper format's header";
        let build = || {
            let mut b = FileBuilder::new();
            b.with_userblock(512).with_userblock_content(HEADER);
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
        // The superblock still begins exactly where the userblock ends, so the
        // content displaced nothing.
        assert_eq!(&buffered[512..520], b"\x89HDF\r\n\x1a\n");
        assert_eq!(read_back_f64(&buffered, "x"), vec![1.0, 2.0]);
    }

    /// Without the setter the region is all zeros, as it was before — so the
    /// setter cannot have changed any existing file's bytes.
    #[test]
    fn an_unset_userblock_stays_zero_filled() {
        let mut b = FileBuilder::new();
        b.with_userblock(512);
        b.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = b.finish().unwrap();
        assert!(bytes[..512].iter().all(|&b| b == 0));
    }

    /// Serves a contiguous dataset's bytes block by block, so a test can stage a
    /// produced region without materializing it. Blocks are `block` bytes except
    /// the last, matching what the emitter asks for.
    struct BlockProvider {
        total: usize,
        block: usize,
        calls: Calls,
        /// Return one byte too few for this block, so a test can reach the
        /// emitter's own size check. The MAT layer checks lengths before the
        /// writer ever sees them, so without this knob that check is unreachable.
        short_block: Option<usize>,
    }

    impl BlockProvider {
        /// The byte this provider yields at offset `i`. A ramp rather than a
        /// constant, so a block emitted at the wrong offset is visible.
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

    /// A produced contiguous region must be the dataset a materialized one would
    /// be — same bytes, same addresses — on both output paths. Sized so the
    /// blocks do not divide the region evenly, since a short last block is where
    /// an off-by-one in the emitter's loop would show up.
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

    /// A paged file classifies each dataset as small or large and reserves
    /// free-space sections from its data length *before* the region is built. A
    /// produced region has no bytes to measure at that point, only a declared
    /// size, so this is the path where trusting the wrong one misplaces the file.
    #[test]
    fn a_produced_dataset_is_placed_correctly_in_a_paged_file() {
        // Larger than the 4 KiB page, so it lands in the large run whose
        // page-aligned fragments the free-space managers describe.
        const TOTAL: usize = 8 * 4096;
        const BLOCK: usize = 8 * 512;
        let expected = BlockProvider::expected(TOTAL);

        let build = |produced: bool| {
            let mut b = FileBuilder::new();
            b.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
            b.with_file_space_page_size(4096);
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
        assert_eq!(read_back_f64(&produced, "d").len(), 4096);
    }

    /// The emitter checks each produced block's length itself, rather than
    /// trusting whatever staged the region.
    ///
    /// The MAT layer's adapter checks first and reports a typed error, so that
    /// path never reaches this guard — which is exactly why it needs its own
    /// test: it is the only thing protecting a producer staged any other way,
    /// and a short block would slide every address after this dataset.
    #[test]
    fn a_produced_block_of_the_wrong_size_is_refused_by_the_emitter() {
        const TOTAL: usize = 8 * 1000;
        const BLOCK: usize = 8 * 256;
        for short in [0usize, 2] {
            let mut b = FileBuilder::new();
            b.create_dataset("d").with_produced_data(
                crate::type_builders::make_f64_type(),
                &[1000],
                TOTAL as u64,
                BLOCK as u64,
                Box::new(BlockProvider {
                    total: TOTAL,
                    block: BLOCK,
                    calls: Calls::default(),
                    short_block: Some(short),
                }),
            );
            match b.finish() {
                Err(Error::Format(FormatError::SerializationError(msg))) => {
                    assert!(
                        msg.contains("planned size"),
                        "expected the block-size refusal, got {msg:?}"
                    );
                }
                other => panic!("expected a refusal for a short block {short}, got {other:?}"),
            }
        }
    }

    /// A userblock size the format does not define is refused rather than
    /// written.
    ///
    /// The size is the superblock's base address, and readers scan the doubling
    /// sequence 0, 512, 1024, … for the signature. An unaligned size therefore
    /// produced a file that this crate itself could not reopen — silently, since
    /// nothing checked. Sizing the region to a wrapper format's header, which is
    /// exactly what `with_userblock_content` invites, is how a caller reaches it.
    #[test]
    fn a_userblock_size_the_format_does_not_define_is_refused() {
        for size in [1u64, 511, 600, 1000, 1536] {
            let mut b = FileBuilder::new();
            b.with_userblock(size);
            b.create_dataset("x").with_f64_data(&[1.0]);
            match b.finish() {
                Err(Error::Format(FormatError::InvalidUserblockSize(reported))) => {
                    assert_eq!(reported, size);
                }
                other => panic!("expected {size} to be refused, got {other:?}"),
            }
        }

        // The sizes the format does define still work, and the file reopens —
        // which is the property the refusal above exists to protect.
        for size in [0u64, 512, 1024, 2048, 4096] {
            let mut b = FileBuilder::new();
            b.with_userblock(size);
            b.create_dataset("x").with_f64_data(&[1.0, 2.0]);
            let bytes = b.finish().expect("a valid userblock size");
            assert_eq!(
                &bytes[size as usize..size as usize + 4],
                b"\x89HDF",
                "the superblock begins exactly where the userblock ends"
            );
            assert_eq!(
                read_back_f64(&bytes, "x"),
                vec![1.0, 2.0],
                "a file with a {size}-byte userblock must reopen"
            );
        }
    }

    /// Content past the end of the region would push the superblock down and
    /// produce a file nothing can open, so it is refused instead.
    #[test]
    fn userblock_content_longer_than_its_region_is_refused() {
        let mut b = FileBuilder::new();
        b.with_userblock(512).with_userblock_content(&[7u8; 513]);
        b.create_dataset("x").with_f64_data(&[1.0]);
        match b.finish() {
            Err(Error::Format(FormatError::UserblockContentTooLarge { content, userblock })) => {
                assert_eq!((content, userblock), (513, 512));
            }
            other => panic!("expected UserblockContentTooLarge, got {other:?}"),
        }
    }
}
