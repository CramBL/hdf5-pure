//! The three `hdf5-pure` paths that write a file holding one created dataset.

use std::path::Path;

use hdf5_pure::DatasetBuilder;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;

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
