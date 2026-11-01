//! Builders for creating HDF5 files and groups.

use std::io::Write;

#[cfg(test)]
use crate::DeflateLevel;
#[cfg(test)]
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::UserblockSize;
use crate::chunked_write::ByteSink;
use crate::file_emission::EmissionError;
use crate::file_writer::FileWriter as FormatWriter;
use crate::type_builders::{
    AttrValue, DatasetBuilder as FormatDatasetBuilder, FinishedGroup,
    GroupBuilder as FormatGroupBuilder,
};

use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::file_create_properties::FileCreateProperties;
#[cfg(test)]
use crate::file_space_info::FileSpaceStrategy;
use crate::libver::LibVer;
use crate::object_path::ObjectPathBuf;

/// Builder for creating a new HDF5 file.
///
/// Staged attribute names are parsed before output by [`Self::finish`], [`Self::finish_to`], and
/// [`Self::write`]. These methods return [`Error::Format`] with
/// [`FormatError::InvalidAttributeName`] for empty names or embedded null bytes, and
/// [`FormatError::AttributeFieldTooLong`] if a name and its null terminator exceed [`u16::MAX`]
/// bytes. Slashes, dots, and whitespace are permitted.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{AttrValue, File, FileBuilder};
/// # let path = test_util::temp::temp_path("output.h5");
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
/// builder.set_attr("version", AttrValue::I64(1));
/// builder.write(&path)?;
/// let file = File::open(&path)?;
/// assert_eq!(file.dataset("data")?.read_f64()?, [1.0, 2.0, 3.0]);
/// assert_eq!(file.root().attrs()?["version"], AttrValue::I64(1));
/// file.close()?;
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
pub struct FileBuilder {
    writer: FormatWriter,
}

impl FileBuilder {
    /// Creates an empty file builder.
    pub fn new() -> Self {
        Self {
            writer: FormatWriter::new(),
        }
    }

    /// Creates a dataset in the root group under `name`, and returns the builder that
    /// configures its data, shape, and attributes.
    ///
    /// `name` is one link name: not empty, not `.`, and holding no `/`.
    /// The write reports [`FormatError::InvalidLinkName`] for a name that is not a component
    /// of an object path.
    pub fn create_dataset(&mut self, name: &str) -> &mut FormatDatasetBuilder {
        self.writer.create_dataset(name)
    }

    /// Returns a builder for a group in the root group under `name`. Call
    /// [`finish`](crate::GroupBuilder::finish) on it, then pass the result to
    /// [`add_group`](Self::add_group).
    ///
    /// `name` is one link name: not empty, not `.`, and holding no `/`.
    /// The write reports [`FormatError::InvalidLinkName`] for a name that is not a component
    /// of an object path.
    pub fn create_group(&mut self, name: &str) -> FormatGroupBuilder {
        self.writer.create_group(name)
    }

    /// Configures and attaches a group in the root group under `name`.
    ///
    /// Calls `configure` once with a temporary builder, then attaches the finished group when
    /// `configure` succeeds. The writer attaches the group under `name`, including when the
    /// callback replaces the builder. If `configure` returns an error or panics, the writer leaves
    /// this file's staged groups unchanged.
    /// The writer checks staged datasets and attributes when writing the file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::InvalidLinkName`] if `name` is empty, is
    /// `.`, or contains `/`, before calling `configure`. Returns [`Error::Format`] with the
    /// callback's error if `configure` fails.
    #[doc = concat!(
        "\n# Examples\n\n```\n",
        include_str!("../../examples/attaching_groups.rs"),
        "\n```"
    )]
    pub fn create_group_with(
        &mut self,
        name: &str,
        configure: impl FnOnce(&mut FormatGroupBuilder) -> Result<(), FormatError>,
    ) -> Result<(), Error> {
        self.writer.create_group_with(name, configure)?;
        Ok(())
    }

    /// Attaches a finished group to the root group.
    pub fn add_group(&mut self, group: FinishedGroup) {
        self.writer.add_group(group);
    }

    /// Stages a named datatype in the root group for datasets and attributes to share.
    ///
    /// `name` must be a nonempty object-path component other than `.`. The writer reports
    /// [`FormatError::InvalidLinkName`] for an invalid component when producing output.
    ///
    /// Reference the datatype with
    /// [`DatasetBuilder::with_committed_datatype`](crate::DatasetBuilder::with_committed_datatype),
    /// [`DatasetBuilder::set_attr_committed`](crate::DatasetBuilder::set_attr_committed),
    /// [`GroupBuilder::set_attr_committed`](crate::GroupBuilder::set_attr_committed), or
    /// [`Self::set_attr_committed`]. The writer rejects unresolved references and datatypes that
    /// disagree with the referring dataset or attribute.
    ///
    /// # Examples
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
    /// assert!(hdf5_pure::is_hdf5_bytes(&bytes));
    /// ```
    pub fn commit_datatype(&mut self, name: &str, datatype: Datatype) {
        self.writer.commit_datatype(name, datatype);
    }

    /// Stages a root-group attribute using the committed datatype at `path`.
    ///
    /// The datatype implied by `value` must match the datatype committed at `path`.
    /// See [`Self::commit_datatype`] for committed types and [`Self::set_attr`] for replacement
    /// and ordering.
    pub fn set_attr_committed(&mut self, name: &str, value: AttrValue, path: &str) {
        self.writer.set_root_attr_committed(name, value, path);
    }

    /// Replaces every file-creation property, including the root group's link policy.
    ///
    /// The writer resets properties left unset in `properties` to their defaults, including
    /// library-version bounds. Apply a shared [`FileCreateProperties`] first, then override
    /// individual properties for this file with setters.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::UserblockSize;
    /// use hdf5_pure::{FileBuilder, FileCreateProperties, LibVer};
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    /// // Reset the bounds to the default format.
    /// builder.with_create_properties(
    ///     FileCreateProperties::new().with_userblock(UserblockSize::MIN),
    /// );
    /// builder.create_dataset("values").with_f64_data(&[1.0]);
    ///
    /// let bytes = builder.finish().unwrap();
    /// assert_eq!(bytes[512 + 8], 3); // version 3 superblock after the userblock
    /// ```
    #[doc(alias = "fcpl")]
    pub fn with_create_properties(&mut self, properties: FileCreateProperties) -> &mut Self {
        self.writer.apply_create_properties(&properties);
        self
    }

    /// Sets the size of the reserved prefix before the HDF5 signature.
    ///
    /// Use [`UserblockSize::MIN`] for 512 bytes and [`UserblockSize::try_from`] to check a raw
    /// byte count. Pass [`UserblockSize::NONE`] to reserve zero bytes.
    ///
    /// The writer fills the prefix with zeros unless content is supplied through
    /// [`with_userblock_content`](Self::with_userblock_content). Paged output requires a
    /// userblock aligned to the file-space page size.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{File, FileBuilder, UserblockSize};
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_userblock(UserblockSize::MIN);
    /// let raw_size = 1024;
    /// builder.with_userblock(UserblockSize::try_from(raw_size)?);
    /// builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    /// let bytes = builder.finish()?;
    /// let file = File::from_bytes(bytes)?;
    /// assert_eq!(file.superblock().base_address.get(), raw_size);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn with_userblock(&mut self, size: UserblockSize) -> &mut Self {
        self.writer.with_userblock(size);
        self
    }

    /// Sets the bytes at the start of the reserved userblock.
    ///
    /// The writer writes `content`, then fills the remaining prefix with zeros.
    /// [`finish`](Self::finish), [`finish_to`](Self::finish_to), and [`write`](Self::write)
    /// return [`Error::Format`] with [`FormatError::UserblockContentTooLarge`] if the content
    /// exceeds the size set by [`with_userblock`](Self::with_userblock).
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::UserblockSize;
    /// use hdf5_pure::FileBuilder;
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_userblock(UserblockSize::MIN);
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

    /// Sets the library-version bounds used to select the output format.
    ///
    /// The writer supports [`LibVer::V18`] and [`LibVer::V110`] output and selects the newest
    /// permitted format. With an upper bound of [`LibVer::V18`], the writer emits a version 2
    /// superblock and version 3 data-layout messages. With [`LibVer::V110`] or a newer upper bound,
    /// it emits a version 3 superblock. Without explicit bounds, it uses [`LibVer::WRITER_DEFAULT`].
    ///
    /// The writer accepts a lower bound newer than [`LibVer::WRITER_DEFAULT`] when the range is
    /// ordered. With both bounds set to [`LibVer::LATEST`], it selects the 1.10 format. When
    /// producing output, it rejects inverted bounds and an upper bound of [`LibVer::Earliest`]
    /// with [`Error::Format`] containing [`FormatError::LibverBoundsUnsatisfiable`].
    ///
    /// This writer requires its 1.10 output format for chunked, filtered, or resizable datasets
    /// and explicit file-space settings. Under a 1.8 upper bound, it rejects that content with
    /// [`Error::Format`] containing [`FormatError::LibverTooOldForContent`].
    /// [`File::open_swmr_writer`](crate::File::open_swmr_writer) requires a version 3 superblock.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{FileBuilder, LibVer};
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    /// builder.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
    /// let bytes = builder.finish().unwrap();
    /// assert_eq!(bytes[8], 2); // version 2 superblock
    /// ```
    pub fn with_libver_bounds(&mut self, low: LibVer, high: LibVer) -> &mut Self {
        self.writer.with_libver_bounds(low, high);
        self
    }

    /// Sets the file-space configuration recorded in the superblock extension.
    ///
    /// Applies the same setting as [`FileCreateProperties::with_file_space_settings`], replacing
    /// the strategy, persistence flag, threshold, and page size together. Explicit default
    /// settings request a recorded File Space Info message and require the HDF5 1.10 format.
    ///
    /// [`Self::with_create_properties`] replaces all creation properties, including clearing this
    /// setting when the supplied properties leave it unset.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{
    ///     File, FileBuilder, FileCreateProperties, FileSpacePageSize, FileSpaceSettings,
    /// };
    ///
    /// let user_input = 3000;
    /// let size = FileSpacePageSize::try_from(user_input)?;
    /// let settings = FileSpaceSettings::default().with_page_size(size);
    /// let properties = FileCreateProperties::new().with_file_space_settings(settings);
    /// assert_eq!(properties.file_space_settings(), Some(settings));
    ///
    /// let mut builder = FileBuilder::new();
    /// builder.with_create_properties(properties);
    /// builder.with_file_space_settings(settings.with_page_size(FileSpacePageSize::DEFAULT));
    /// let file = File::from_bytes(builder.finish()?)?;
    /// assert_eq!(
    ///     file.file_space_settings()?.map(|settings| settings.page_size()),
    ///     Some(FileSpacePageSize::DEFAULT)
    /// );
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn with_file_space_settings(&mut self, settings: FileSpaceSettings) -> &mut Self {
        self.writer.with_file_space_settings(settings);
        self
    }

    /// Stages an attribute for the root group.
    ///
    /// The writer replaces a staged attribute with the same name and preserves its first insertion
    /// position. The replacement includes its value and datatype information.
    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.writer.set_root_attr(name, value);
    }

    /// Attaches an already-encoded attribute message to the root group, written as given except
    /// for its datatype location.
    ///
    /// See [`Self::set_attr`] for replacement and ordering.
    ///
    /// Where `committed_datatype_path` is set, the written message refers to the committed
    /// datatype at that path. See
    /// [`AttrSpec::Verbatim`](crate::type_builders::AttrSpec::Verbatim) for
    /// what this preserves that [`set_attr`](Self::set_attr) cannot, and for the
    /// datatypes it must not be used with.
    pub(crate) fn set_attr_verbatim(
        &mut self,
        message: crate::attribute::AttributeMessage,
        committed_datatype_path: Option<ObjectPathBuf>,
    ) {
        self.writer
            .set_root_attr_verbatim(message, committed_datatype_path);
    }

    /// Attaches a variable-length string attribute to the root group with the datatype and
    /// dataspace of `message`, staging `strings` into a heap of this file's own.
    ///
    /// See [`Self::set_attr`] for replacement and ordering.
    ///
    /// Where `committed_datatype_path` is set, the written message refers to the committed
    /// datatype at that path. See
    /// [`AttrSpec::VerbatimVarLen`](crate::type_builders::AttrSpec::VerbatimVarLen).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if a string exceeds
    /// [`u32::MAX`] bytes.
    pub(crate) fn set_attr_var_len_verbatim(
        &mut self,
        message: crate::attribute::AttributeMessage,
        strings: Vec<String>,
        committed_datatype_path: Option<ObjectPathBuf>,
    ) -> Result<(), FormatError> {
        self.writer
            .set_root_attr_var_len_verbatim(message, strings, committed_datatype_path)
    }

    /// Serializes the staged file into memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the staged content cannot be serialized, including a
    /// variable-length element count greater than [`u32::MAX`].
    pub fn finish(self) -> Result<Vec<u8>, Error> {
        Ok(self.writer.finish()?)
    }

    /// Serializes the file directly to a [`Write`] sink, without first buffering
    /// the whole file in memory.
    ///
    /// Produces byte-for-byte the same file as [`finish`](Self::finish). For datasets staged
    /// with verbatim chunk providers, reads and writes one encoded chunk at a time. The writer
    /// retains file metadata and supplied dataset buffers during layout planning.
    ///
    /// The writer plans addresses before output and writes forward without seeking. On failure,
    /// bytes already written remain in the sink. For atomic replacement, write to a temporary
    /// path and rename it after successful output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] with the original write or flush error if output fails, or
    /// [`Error::Format`] if the settings or data cannot be serialized.
    ///
    /// # Examples
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
        self.writer
            .finish_to_sink(&mut sink)
            .map_err(|error| match error {
                EmissionError::Format(error) => Error::Format(error),
                EmissionError::Sink(error) => Error::Io(error),
            })?;
        sink.into_inner().flush().map_err(Error::Io)
    }

    /// Serializes the file to `path`.
    ///
    /// The writer creates `path` when the first byte is ready. A rejection before output,
    /// including incompatible format bounds or userblock settings, leaves an existing file
    /// at `path` unchanged. Later failures can leave partial output, as
    /// [`finish_to`](Self::finish_to) describes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] for storage errors, or [`Error::Format`] if the settings or data
    /// are invalid.
    pub fn write<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), Error> {
        self.finish_to(LazyFile {
            path: path.as_ref().to_path_buf(),
            file: None,
        })
    }
}

/// A [`Write`] adapter that creates its destination on the first write call.
pub(crate) struct LazyFile {
    pub(crate) path: std::path::PathBuf,
    pub(crate) file: Option<std::fs::File>,
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
        // Flushing an unopened destination must not create or truncate it.
        match &mut self.file {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

/// Tracks the output position of a [`Write`] sink during file assembly.
///
/// Returns the original [`std::io::Error`] from failed writes. [`FileBuilder::finish_to`] converts
/// the sink failure to [`Error::Io`].
pub(crate) struct WriteSink<W: Write> {
    inner: W,
    written: u64,
}

impl<W: Write> WriteSink<W> {
    /// Starts tracking output at position zero.
    pub(crate) fn new(inner: W) -> Self {
        Self { inner, written: 0 }
    }

    /// Returns the writer after the recorded writes.
    pub(crate) fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> ByteSink for WriteSink<W> {
    type Error = std::io::Error;

    fn put(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.inner.write_all(bytes)?;
        self.written += bytes.len() as u64;
        Ok(())
    }

    fn put_zeros(&mut self, n: usize) -> Result<(), Self::Error> {
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
mod tests;
