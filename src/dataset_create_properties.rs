//! Dataset creation properties, the analogue of an HDF5 dataset creation property list.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_core::FillTime;

use crate::DeflateLevel;
use crate::ScaleOffset;
#[cfg(feature = "zfp")]
use crate::ZfpRate;
use crate::chunked_write::ChunkOptions;
use crate::chunked_write::FilterKind;
#[cfg(feature = "std")]
use crate::element::H5Element;
use crate::type_builders::DatasetBuilder;

/// The creation properties of a dataset: its chunk dimensions, filters, and fill settings.
///
/// This is the analogue of an HDF5 dataset creation property list (`dcpl`), the list a caller
/// passes to `H5Dcreate` to set how the dataset is stored, apart from its data and its shape.
/// [`DatasetBuilder::with_create_properties`] applies the properties to a dataset. Each method has
/// the name and the effect of the [`DatasetBuilder`] method that sets the same property, and a
/// dataset written with these properties has the bytes of one written with those methods.
///
/// The writer checks the properties against the dataset when the file is written, and returns
/// [`FormatError::FilterError`] for two filters that cannot be combined.
///
/// A filter method adds its filter at the position the [`DatasetBuilder`] methods give it, and a
/// second call for the same filter replaces it in place. The writer derives the chunk dimensions
/// of a filtered dataset from its shape unless [`with_chunks`](Self::with_chunks) sets them.
///
/// # Supported properties
///
/// | HDF5 property (C API) | `hdf5-pure` | Status | Behavior |
/// |---|---|---|---|
/// | `H5Pset_chunk` | [`with_chunks`](Self::with_chunks) | **Supported** | Sets the chunk dimensions, one per dataset dimension. |
/// | `H5Pset_deflate` | [`with_deflate`](Self::with_deflate) | **Supported** | Adds the Deflate filter at a [`DeflateLevel`]. |
/// | `H5Pset_shuffle` | [`with_shuffle`](Self::with_shuffle) | **Supported** | Adds the byte shuffle filter. |
/// | `H5Pset_fletcher32` | [`with_fletcher32`](Self::with_fletcher32) | **Supported** | Adds the Fletcher32 checksum filter. |
/// | `H5Pset_scaleoffset` | [`with_scale_offset`](Self::with_scale_offset) | **Supported** | Adds the scale-offset filter in a [`ScaleOffset`] mode. |
/// | `H5Pset_filter`, LZF (filter 32000) | [`with_lzf`](Self::with_lzf) | **Supported** | Adds the LZF filter that h5py registers, recorded as optional. The tests compare the filter pipeline with the one h5py records. |
/// | `H5Pset_filter`, ZFP (filter 32013) | [`with_zfp`](Self::with_zfp) | **Supported** (`zfp` feature) | Adds the fixed-rate ZFP filter of the H5Z-ZFP plugin. The tests compare the encoded bytes with those of the plugin. |
/// | `H5Pset_fill_value` | [`with_fill_value`](Self::with_fill_value) | **Supported** (`std` feature) | Sets a fill value of the dataset's element type. |
/// | `H5Pset_fill_time` | [`with_fill_time`](Self::with_fill_time) | **Supported** | Sets when the writer writes the fill value to storage. |
/// | `H5Pset_layout` | none | **Unsupported** | The writer stores a dataset chunked where chunk dimensions, a filter, or a maximum shape is set, and contiguous otherwise. |
/// | `H5Pset_alloc_time` | none | **Unsupported** | The writer records late allocation for every dataset. |
/// | `H5Pset_external` | none | **Unsupported** | The writer stores the raw data in the HDF5 file. |
/// | `H5Pset_szip`, `H5Pset_nbit`, `H5Pset_filter` with another filter | none | **Unsupported** | The writer applies the filters above and no other. |
/// | `H5Pset_chunk_opts` | none | **Unsupported** | The writer filters every chunk, the partial edge chunks included. |
/// | `H5Pset_virtual` | none | **Unsupported** | The writer stores every dataset contiguous or chunked, as the `H5Pset_layout` row describes. |
/// | `H5Pset_dset_no_attrs_hint` | none | **Unsupported** | The writer does not expose the dataset object-header minimization hint. |
///
/// # Examples
///
/// ```
/// use hdf5_pure::{DatasetCreateProperties, DeflateLevel, File};
/// # use hdf5_pure::FileBuilder;
/// # let values = [3, 1, 4, 1, 5, 9, 2, 6];
/// # let mut builder = FileBuilder::new();
///
/// let properties = DatasetCreateProperties::new()
///     .with_chunks(&[4])
///     .with_shuffle()
///     .with_deflate(DeflateLevel::Six);
///
/// builder
///     .create_dataset("values")
///     .with_i32_data(&values)
///     .with_create_properties(properties);
///
/// let file = File::from_bytes(builder.finish()?)?;
/// assert_eq!(file.dataset("values")?.chunk_shape()?, Some(vec![4]));
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
///
/// [`FormatError::FilterError`]: crate::FormatError::FilterError
#[doc(alias = "dcpl")]
#[derive(Clone, Debug, Default)]
pub struct DatasetCreateProperties {
    chunk_options: ChunkOptions,
    fill: Option<Vec<u8>>,
    fill_time: Option<FillTime>,
}

impl DatasetCreateProperties {
    /// Creates properties with no chunk dimensions, no filters, and no fill settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the chunk dimensions, one per dataset dimension.
    #[doc(alias = "H5Pset_chunk")]
    pub fn with_chunks(mut self, chunk_dims: &[u64]) -> Self {
        self.chunk_options.chunk_dims = Some(chunk_dims.to_vec());
        self
    }

    /// Adds the Deflate filter at `level`.
    ///
    /// [`DatasetBuilder::with_deflate`] lists the filters it cannot be combined with.
    #[doc(alias = "H5Pset_deflate")]
    pub fn with_deflate(self, level: DeflateLevel) -> Self {
        self.with_filter(FilterKind::Deflate(level))
    }

    /// Adds the byte shuffle filter.
    ///
    /// [`DatasetBuilder::with_shuffle`] lists the filters it cannot be combined with.
    #[doc(alias = "H5Pset_shuffle")]
    pub fn with_shuffle(self) -> Self {
        self.with_filter(FilterKind::Shuffle)
    }

    /// Adds the Fletcher32 checksum filter.
    #[doc(alias = "H5Pset_fletcher32")]
    pub fn with_fletcher32(self) -> Self {
        self.with_filter(FilterKind::Fletcher32)
    }

    /// Adds the scale-offset filter in `mode`.
    ///
    /// The mode has to match the dataset's datatype. [`DatasetBuilder::with_scale_offset`]
    /// describes the modes and lists the filters it cannot be combined with.
    #[doc(alias = "H5Pset_scaleoffset")]
    pub fn with_scale_offset(self, mode: ScaleOffset) -> Self {
        self.with_filter(FilterKind::scale_offset(mode))
    }

    /// Adds the LZF filter, which h5py registers as filter 32000.
    ///
    /// The writer records the filter as optional. [`DatasetBuilder::with_lzf`] lists the filters
    /// it cannot be combined with.
    pub fn with_lzf(self) -> Self {
        self.with_filter(FilterKind::Lzf)
    }

    /// Adds the fixed-rate ZFP filter at `rate` bits per scalar.
    ///
    /// The H5Z-ZFP plugin registers the filter as 32013. [`DatasetBuilder::with_zfp`] lists the
    /// scalar types and chunk shapes it supports and the filters it cannot be combined with.
    #[cfg(feature = "zfp")]
    pub fn with_zfp(self, rate: ZfpRate) -> Self {
        self.with_filter(FilterKind::Zfp(rate))
    }

    /// Sets the fill value, which a reader returns for an element that has never been written
    /// unless the fill time is [`FillTime::Never`].
    ///
    /// `T` has to have the width of the dataset's element type. The writer rejects a dataset
    /// where the widths differ with [`FormatError::FillValueSizeMismatch`].
    ///
    /// [`FormatError::FillValueSizeMismatch`]: crate::FormatError::FillValueSizeMismatch
    #[cfg(feature = "std")]
    #[doc(alias = "H5Pset_fill_value")]
    pub fn with_fill_value<T: H5Element>(mut self, value: T) -> Self {
        self.fill = Some(T::fill_bytes(value));
        self
    }

    /// Sets when the writer writes the fill value to storage.
    ///
    /// The writer writes it on allocation ([`FillTime::Alloc`]), never ([`FillTime::Never`]), or
    /// when a fill value is set ([`FillTime::IfSet`]). A dataset without this property has
    /// [`FillTime::IfSet`].
    #[doc(alias = "H5Pset_fill_time")]
    pub fn with_fill_time(mut self, fill_time: FillTime) -> Self {
        self.fill_time = Some(fill_time);
        self
    }

    fn with_filter(mut self, kind: FilterKind) -> Self {
        self.chunk_options.set_filter(kind);
        self
    }
}

impl DatasetBuilder {
    /// Replaces the chunk dimensions, filters, fill value, and fill time of this dataset with
    /// `properties`.
    ///
    /// A property that `properties` leaves unset takes its default, even where a method called
    /// before this call set it. A method called after this call changes the one property it sets
    /// and keeps the others. This is the analogue of passing a dataset creation property list to
    /// `H5Dcreate`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{DatasetCreateProperties, DeflateLevel, File, FileBuilder};
    ///
    /// let compressed = DatasetCreateProperties::new()
    ///     .with_chunks(&[4])
    ///     .with_deflate(DeflateLevel::Six);
    ///
    /// let mut builder = FileBuilder::new();
    /// builder
    ///     .create_dataset("data")
    ///     .with_i32_data(&[1, 2, 3, 4, 5, 6, 7, 8])
    ///     .with_create_properties(compressed)
    ///     .with_chunks(&[2]);
    ///
    /// let file = File::from_bytes(builder.finish()?)?;
    /// assert_eq!(file.dataset("data")?.chunk_shape()?, Some(vec![2]));
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    #[doc(alias = "dcpl")]
    pub fn with_create_properties(&mut self, properties: DatasetCreateProperties) -> &mut Self {
        let DatasetCreateProperties {
            chunk_options,
            fill,
            fill_time,
        } = properties;
        self.chunk_options = chunk_options;
        self.fill = fill;
        self.fill_time = fill_time;
        self
    }
}
