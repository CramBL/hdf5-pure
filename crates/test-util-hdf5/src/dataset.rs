//! The rank-1 unlimited dataset the append tests grow, built and read by `hdf5-pure` and, behind
//! the `hdf5` feature, by the reference C library.

use std::path::Path;

#[cfg(feature = "__hdf5-1.10")]
use hdf5::Extent;
use hdf5_pure::{DatasetBuilder, File, FileBuilder, H5Element, MaxExtent, ScaleOffset};

/// One stage of a filter pipeline. A dataset applies its stages in the order it lists them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Filter {
    Deflate(u8),
    Shuffle,
    Fletcher32,
    ScaleOffset(ScaleOffset),
    Lzf,
}

/// Shuffle followed by deflate at level 6, the common append interoperability pipeline.
pub const SHUFFLE_DEFLATE_6: &[Filter] = &[Filter::Shuffle, Filter::Deflate(6)];

/// A malformed dataset geometry request shared by writer and editor validation tests.
#[derive(Clone, Copy)]
pub struct InvalidGeometryCase {
    pub label: &'static str,
    pub configure: fn(&mut DatasetBuilder),
    pub expected: &'static str,
}

/// Geometry errors both initial writes and in-place dataset creation must reject.
pub const INVALID_GEOMETRY_CASES: &[InvalidGeometryCase] = &[
    InvalidGeometryCase {
        label: "auto-chunked empty shape",
        configure: |builder| {
            builder
                .with_f64_data(&[])
                .with_shape(&[0])
                .with_maxshape(&[MaxExtent::Unlimited]);
        },
        expected: "explicit chunk dimensions",
    },
    InvalidGeometryCase {
        label: "chunk rank mismatch",
        configure: |builder| {
            builder
                .with_i32_data(&[1, 2, 3, 4, 5, 6])
                .with_shape(&[2, 3])
                .with_chunks(&[2]);
        },
        expected: "chunk dimensions must have the same rank",
    },
    InvalidGeometryCase {
        label: "zero chunk dim",
        configure: |builder| {
            builder
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_chunks(&[0]);
        },
        expected: "chunk dimensions must all be non-zero",
    },
    InvalidGeometryCase {
        label: "maxshape rank mismatch",
        configure: |builder| {
            builder
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Unlimited])
                .with_chunks(&[2]);
        },
        expected: "maxshape must have the same rank",
    },
    InvalidGeometryCase {
        label: "scalar with chunks",
        configure: |builder| {
            builder
                .with_f64_data(&[1.0])
                .with_shape(&[])
                .with_chunks(&[1]);
        },
        expected: "a scalar dataset cannot be chunked",
    },
    InvalidGeometryCase {
        label: "maxshape below shape",
        configure: |builder| {
            builder
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_maxshape(&[MaxExtent::Fixed(2)]);
        },
        expected: "maxshape must be at least the current shape",
    },
];

/// Additional geometry errors covered by complete-file writer validation tests.
pub const WRITER_ONLY_INVALID_GEOMETRY_CASES: &[InvalidGeometryCase] = &[
    InvalidGeometryCase {
        label: "auto-chunked empty inner dim",
        configure: |builder| {
            builder
                .with_i32_data(&[])
                .with_shape(&[4, 0])
                .with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Unlimited]);
        },
        expected: "explicit chunk dimensions",
    },
    InvalidGeometryCase {
        label: "a fixed maximum at the unlimited marker",
        configure: |builder| {
            builder
                .with_i32_data(&[1, 2, 3, 4])
                .with_shape(&[4])
                .with_maxshape(&[MaxExtent::Fixed(u64::MAX)])
                .with_chunks(&[2]);
        },
        expected: "the format's unlimited marker",
    },
];

/// Dataset shapes used by both staged and immediate append refusal tests.
#[derive(Clone, Copy, Debug)]
pub enum AppendRefusalFixture {
    /// A contiguous dataset, which has no chunk index to extend.
    Contiguous,
    /// A chunked dataset with a finite maximum extent.
    FixedChunked,
    /// A rank-2 chunked dataset with one unlimited dimension.
    Rank2,
    /// A rank-1 unlimited `i32` dataset with a deflate filter.
    UnlimitedI32,
}

impl AppendRefusalFixture {
    /// Writes this refusal fixture to `path`.
    #[track_caller]
    pub fn write(self, path: &Path) {
        let mut builder = FileBuilder::new();
        match self {
            Self::Contiguous => {
                builder
                    .create_dataset("d")
                    .with_i32_data(&[1, 2, 3])
                    .with_shape(&[3]);
            }
            Self::FixedChunked => {
                builder
                    .create_dataset("d")
                    .with_i32_data(&(0..6).collect::<Vec<_>>())
                    .with_shape(&[6])
                    .with_maxshape(&[MaxExtent::Fixed(100)])
                    .with_chunks(&[3]);
            }
            Self::Rank2 => {
                builder
                    .create_dataset("d")
                    .with_i32_data(&(0..12).collect::<Vec<_>>())
                    .with_shape(&[3, 4])
                    .with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Fixed(4)])
                    .with_chunks(&[1, 4]);
            }
            Self::UnlimitedI32 => {
                Unlimited::new("d", &(0..4).collect::<Vec<i32>>(), 4)
                    .filters(&[Filter::Deflate(6)])
                    .add_to(&mut builder);
            }
        }
        builder
            .write(path)
            .unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    }

    /// Returns the initial `i32` values stored in `d`.
    pub fn expected_i32(self) -> Vec<i32> {
        match self {
            Self::Contiguous => vec![1, 2, 3],
            Self::FixedChunked => (0..6).collect(),
            Self::Rank2 => (0..12).collect(),
            Self::UnlimitedI32 => (0..4).collect(),
        }
    }
}

/// A reusable rank-1 `i32` append interoperability fixture.
#[derive(Clone, Copy, Debug)]
pub struct AppendInteropFixture {
    source: AppendInteropSource,
    len: usize,
    chunk: u64,
    filters: &'static [Filter],
    incompressible_seed: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
enum AppendInteropSource {
    Pure,
    #[cfg(feature = "__hdf5-1.10")]
    Libhdf5,
}

impl AppendInteropFixture {
    /// Returns a pure-written filtered fixture seeded with `0..len`.
    pub const fn filtered_pure(len: usize, chunk: u64) -> Self {
        Self {
            source: AppendInteropSource::Pure,
            len,
            chunk,
            filters: SHUFFLE_DEFLATE_6,
            incompressible_seed: None,
        }
    }

    /// Returns a pure-written unfiltered fixture seeded with `0..len`.
    pub const fn unfiltered_pure(len: usize, chunk: u64) -> Self {
        Self {
            source: AppendInteropSource::Pure,
            len,
            chunk,
            filters: &[],
            incompressible_seed: None,
        }
    }

    /// Returns a C-library-written filtered fixture seeded with `0..len`.
    #[cfg(feature = "__hdf5-1.10")]
    pub const fn filtered_libhdf5(len: usize, chunk: u64) -> Self {
        Self {
            source: AppendInteropSource::Libhdf5,
            len,
            chunk,
            filters: SHUFFLE_DEFLATE_6,
            incompressible_seed: None,
        }
    }

    /// Returns a C-library-written filtered fixture with deterministic incompressible values.
    #[cfg(feature = "__hdf5-1.10")]
    pub const fn incompressible_libhdf5(seed: u32, len: usize, chunk: u64) -> Self {
        Self {
            source: AppendInteropSource::Libhdf5,
            len,
            chunk,
            filters: SHUFFLE_DEFLATE_6,
            incompressible_seed: Some(seed),
        }
    }

    /// Writes the fixture and returns its initial values.
    #[track_caller]
    pub fn write(self, path: &Path) -> Vec<i32> {
        let values = match self.incompressible_seed {
            Some(seed) => incompressible(seed, self.len),
            None => {
                let end = i32::try_from(self.len).expect("an append fixture length that fits i32");
                (0..end).collect()
            }
        };
        let fixture = Unlimited::new("d", &values, self.chunk).filters(self.filters);
        match self.source {
            AppendInteropSource::Pure => fixture.pure_create(path),
            #[cfg(feature = "__hdf5-1.10")]
            AppendInteropSource::Libhdf5 => fixture.libhdf5_create(path),
        }
        values
    }
}

/// A chunked dataset of one dimension whose maximum is unlimited, seeded with `data`.
///
/// Under the 1.10 format or newer both libraries index its chunks with an Extensible Array, the
/// index an in-place append grows.
#[derive(Clone, Debug)]
pub struct Unlimited<'a, T> {
    name: &'a str,
    data: &'a [T],
    chunk: u64,
    filters: Vec<Filter>,
    fill: Option<T>,
}

impl<'a, T: Copy> Unlimited<'a, T> {
    pub fn new(name: &'a str, data: &'a [T], chunk: u64) -> Self {
        Self {
            name,
            data,
            chunk,
            filters: Vec::new(),
            fill: None,
        }
    }

    pub fn filters(mut self, filters: &[Filter]) -> Self {
        self.filters = filters.to_vec();
        self
    }

    pub fn fill_value(mut self, fill: T) -> Self {
        self.fill = Some(fill);
        self
    }
}

impl<T: H5Element> Unlimited<'_, T> {
    /// Writes a file holding only this dataset.
    #[track_caller]
    pub fn pure_create(&self, path: &Path) {
        let mut builder = FileBuilder::new();
        self.add_to(&mut builder);
        builder
            .write(path)
            .unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    }

    /// Adds this dataset to a file whose other properties the caller sets.
    pub fn add_to(&self, builder: &mut FileBuilder) {
        let Self {
            name,
            data,
            chunk,
            ref filters,
            fill,
        } = *self;
        let dataset = builder
            .create_dataset(name)
            .with_data(data)
            .with_shape(&[data.len() as u64])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[chunk]);
        for filter in filters {
            match *filter {
                Filter::Deflate(level) => dataset.with_deflate(u32::from(level)),
                Filter::Shuffle => dataset.with_shuffle(),
                Filter::Fletcher32 => dataset.with_fletcher32(),
                Filter::ScaleOffset(mode) => dataset.with_scale_offset(mode),
                Filter::Lzf => dataset.with_lzf(),
            };
        }
        if let Some(fill) = fill {
            dataset.with_fill_value(fill);
        }
    }
}

#[cfg(feature = "__hdf5-1.10")]
impl<T: hdf5::H5Type + Copy> Unlimited<'_, T> {
    /// Has the C library write a file in the 1.10 format holding only this dataset.
    #[track_caller]
    pub fn libhdf5_create(&self, path: &Path) {
        let Self {
            name,
            data,
            chunk,
            ref filters,
            fill,
        } = *self;
        let file = crate::file::libhdf5_create_v110(path);
        let chunk = usize::try_from(chunk).expect("a chunk length this host can address");
        let mut builder = file
            .new_dataset::<T>()
            .chunk((chunk,))
            .shape((Extent::resizable(data.len()),));
        for filter in filters {
            builder = match *filter {
                Filter::Deflate(level) => builder.deflate(level),
                Filter::Shuffle => builder.shuffle(),
                Filter::Fletcher32 => builder.fletcher32(),
                Filter::ScaleOffset(mode) => builder.scale_offset(libhdf5_scale_offset(mode)),
                Filter::Lzf => panic!("the crosschecks' libhdf5 is built without the LZF filter"),
            };
        }
        if let Some(fill) = fill {
            builder = builder.fill_value(fill);
        }
        let dataset = builder
            .create(name)
            .unwrap_or_else(|e| panic!("create {name:?} in {path:?}: {e}"));
        if !data.is_empty() {
            dataset
                .write_raw(data)
                .unwrap_or_else(|e| panic!("write {name:?} in {path:?}: {e}"));
        }
        file.close()
            .unwrap_or_else(|e| panic!("close {path:?}: {e}"));
    }
}

/// Every element of the dataset at `name`, read by `hdf5-pure`.
#[track_caller]
pub fn read_pure<T: H5Element>(path: &Path, name: &str) -> Vec<T> {
    File::open(path)
        .unwrap_or_else(|e| panic!("open {path:?}: {e}"))
        .dataset(name)
        .unwrap_or_else(|e| panic!("open {name:?} in {path:?}: {e}"))
        .read()
        .unwrap_or_else(|e| panic!("read {name:?} in {path:?}: {e}"))
}

/// Every element of the dataset at `name`, read by the C library.
#[cfg(feature = "hdf5")]
#[track_caller]
pub fn read_libhdf5<T: hdf5::H5Type>(path: &Path, name: &str) -> Vec<T> {
    let file = hdf5::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let values = file
        .dataset(name)
        .unwrap_or_else(|e| panic!("open {name:?} in {path:?}: {e}"))
        .read_raw()
        .unwrap_or_else(|e| panic!("read {name:?} in {path:?}: {e}"));
    file.close()
        .unwrap_or_else(|e| panic!("close {path:?}: {e}"));
    values
}

/// Appends `values` to the dataset at `name` in place, in one session that closes before this
/// returns.
#[track_caller]
pub fn pure_append<T: H5Element>(path: &Path, name: &str, values: &[T]) {
    let file = File::open_rw(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    file.dataset(name)
        .unwrap_or_else(|e| panic!("open {name:?} in {path:?}: {e}"))
        .append(values)
        .unwrap_or_else(|e| panic!("append to {name:?} in {path:?}: {e}"));
    file.close()
        .unwrap_or_else(|e| panic!("close {path:?}: {e}"));
}

/// Appends `values` to the dataset at `name` through a staged append, which rebuilds its chunk
/// index at the commit.
#[track_caller]
pub fn pure_append_staged<T: H5Element>(path: &Path, name: &str, values: &[T]) {
    let file = File::open_rw(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    file.dataset(name)
        .unwrap_or_else(|e| panic!("open {name:?} in {path:?}: {e}"))
        .append_staged(|builder| {
            builder.append(values);
        })
        .unwrap_or_else(|e| panic!("stage an append to {name:?} in {path:?}: {e}"));
    file.commit()
        .unwrap_or_else(|e| panic!("commit {path:?}: {e}"));
}

/// `len` pseudo-random values from `seed`, which deflate cannot shrink: the C library stores each
/// chunk of them unfiltered and records a nonzero filter mask for it.
pub fn incompressible(seed: u32, len: usize) -> Vec<i32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x as i32
        })
        .collect()
}

#[cfg(feature = "__hdf5-1.10")]
fn libhdf5_scale_offset(mode: ScaleOffset) -> hdf5::filters::ScaleOffset {
    match mode {
        ScaleOffset::Integer(minbits) => hdf5::filters::ScaleOffset::Integer(
            u16::try_from(minbits).expect("a minimum bit count libhdf5 takes"),
        ),
        ScaleOffset::FloatDScale(scale) => hdf5::filters::ScaleOffset::FloatDScale(
            u8::try_from(scale).expect("a decimal scale factor libhdf5 takes"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use hdf5_pure::{ChunkIndex, File, Layout, MaxExtent};

    use crate::dataset::{self, Filter, Unlimited};

    #[test]
    fn a_created_dataset_is_unlimited_and_grows_by_either_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.h5");
        Unlimited::new("d", &[1i32, 2, 3], 2)
            .filters(&[Filter::Shuffle, Filter::Deflate(6)])
            .fill_value(9)
            .pure_create(&path);
        dataset::pure_append(&path, "d", &[4i32]);
        dataset::pure_append_staged(&path, "d", &[5i32, 6]);

        assert_eq!(
            dataset::read_pure::<i32>(&path, "d"),
            vec![1, 2, 3, 4, 5, 6]
        );
        let file = File::open(&path).unwrap();
        let d = file.dataset("d").unwrap();
        assert_eq!(d.maxshape().unwrap(), Some(vec![MaxExtent::Unlimited]));
        assert_eq!(
            d.layout().unwrap(),
            Layout::Chunked {
                chunk_shape: vec![2],
                index: ChunkIndex::ExtensibleArray,
            }
        );
        assert_eq!(d.filters(), vec![SHUFFLE, DEFLATE]);
        assert_eq!(d.fill_value::<i32>().unwrap(), Some(9));
    }

    // The filter identifiers of section `subsubsec_fmt4_dataobject_hdr_msg_filter`, version 4.0.
    const DEFLATE: u16 = 1;
    const SHUFFLE: u16 = 2;
}
