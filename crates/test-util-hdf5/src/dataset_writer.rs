//! Dataset creation and append paths for shared writer fixtures.

use std::path::Path;

use hdf5_pure::DatasetBuilder;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FillTime;
use hdf5_pure::H5Element;
use hdf5_pure::MaxExtent;
use hdf5_pure::ScaleOffset;

/// A path through which `hdf5-pure` creates a dataset in a new file.
#[derive(Clone, Copy, Debug)]
pub enum DatasetWriter {
    /// [`FileBuilder::finish`], with its image written to the path afterwards.
    Buffered,
    /// A dataset staged and committed in a read-write session of an empty file.
    Staged,
    /// [`FileBuilder::write`], which streams the file to disk.
    Streaming,
}

impl DatasetWriter {
    /// Writes a file at `path` holding one dataset, `dataset_name`, that `configure` sets up.
    ///
    /// # Panics
    ///
    /// Panics if any step of the write fails.
    pub fn create(
        self,
        path: &Path,
        dataset_name: &str,
        configure: impl FnOnce(&mut DatasetBuilder),
    ) {
        match self {
            Self::Buffered => {
                let mut builder = FileBuilder::new();
                configure(builder.create_dataset(dataset_name));
                let bytes = builder
                    .finish()
                    .unwrap_or_else(|e| panic!("finish {path:?}: {e}"));
                std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
            }
            Self::Staged => {
                FileBuilder::new()
                    .write(path)
                    .unwrap_or_else(|e| panic!("write {path:?}: {e}"));
                let file = File::open_rw(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
                file.root()
                    .create_dataset(dataset_name, configure)
                    .unwrap_or_else(|e| panic!("stage {dataset_name:?} in {path:?}: {e}"));
                file.commit()
                    .unwrap_or_else(|e| panic!("commit {path:?}: {e}"));
            }
            Self::Streaming => {
                let mut builder = FileBuilder::new();
                configure(builder.create_dataset(dataset_name));
                builder
                    .write(path)
                    .unwrap_or_else(|e| panic!("write {path:?}: {e}"));
            }
        }
    }
}

/// A write path for one full floating-point chunk encoded with decimal scale-offset.
#[derive(Clone, Copy, Debug)]
pub enum ScaleOffsetWriter {
    /// [`FileBuilder::finish`], with its image written to disk.
    Buffered,
    /// A committed empty creation followed by [`hdf5_pure::Dataset::append`].
    CreateAppend,
    /// A pending empty creation followed by [`hdf5_pure::Dataset::append_staged`].
    CreateAppendStaged,
    /// A streamed empty creation followed by [`hdf5_pure::Dataset::append_staged`].
    StagedAppend,
    /// [`FileBuilder::write`], with the complete chunk supplied at creation.
    Streaming,
}

impl ScaleOffsetWriter {
    /// Writes `values` to dataset `col` with four-element chunks and one decimal digit of scaling.
    ///
    /// Creation-only paths use a fixed extent. Append paths start with an empty dataset and an
    /// unlimited maximum extent. The caller supplies [`f32`] or [`f64`] elements.
    ///
    /// # Panics
    ///
    /// Panics if any creation, append, or file operation fails.
    pub fn write<T: H5Element + From<f32>>(
        self,
        path: &Path,
        values: &[T; 4],
        fill: T,
        fill_time: FillTime,
    ) {
        let configure = |dataset: &mut DatasetBuilder, data: &[T]| {
            dataset
                .with_data(data)
                .with_chunks(&[4])
                .with_fill_value(fill)
                .with_fill_time(fill_time)
                .with_scale_offset(ScaleOffset::FloatDScale(1));
        };
        match self {
            Self::Buffered | Self::Streaming => {
                let writer = match self {
                    Self::Buffered => DatasetWriter::Buffered,
                    _ => DatasetWriter::Streaming,
                };
                writer.create(path, "col", |dataset| configure(dataset, values));
            }
            Self::CreateAppend | Self::CreateAppendStaged => {
                FileBuilder::new().write(path).unwrap();
                let file = File::open_rw(path).unwrap();
                file.root()
                    .create_dataset("col", |dataset| {
                        configure(dataset, &[]);
                        dataset.with_maxshape(&[MaxExtent::Unlimited]);
                    })
                    .unwrap();
                match self {
                    Self::CreateAppend => {
                        file.commit().unwrap();
                        file.dataset("col").unwrap().append(values).unwrap();
                    }
                    _ => file
                        .dataset("col")
                        .unwrap()
                        .append_staged(|builder| {
                            builder.append(values);
                        })
                        .unwrap(),
                }
                file.commit().unwrap();
                file.close().unwrap();
            }
            Self::StagedAppend => {
                DatasetWriter::Streaming.create(path, "col", |dataset| {
                    configure(dataset, &[]);
                    dataset.with_maxshape(&[MaxExtent::Unlimited]);
                });
                crate::dataset::pure_append_staged(path, "col", values);
            }
        }
    }
}
