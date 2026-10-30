//! Append modes and filter pipelines shared by dataset tests.

use hdf5_pure::{Dataset, Error, File, H5Element};

use super::{Filter, SHUFFLE_DEFLATE_6};

/// A public append API used to grow a test dataset.
#[derive(Clone, Copy, Debug)]
pub enum AppendMode {
    /// An appender that borrows the dataset and finishes after two-element batches.
    Buffered,
    /// An immediate append of the complete input.
    Immediate,
    /// An appender that owns the dataset and finishes after two-element batches.
    OwnedBuffered,
    /// A staged append followed by a commit of the file's pending edits.
    Staged,
}

impl AppendMode {
    /// Appends `data`, finishing buffered appenders or committing staged edits before returning.
    ///
    /// Buffered modes submit two-element batches. Empty input is submitted once so that invalid
    /// element widths are reported.
    ///
    /// # Errors
    ///
    /// Returns errors from creating the appender, appending elements, finishing the appender,
    /// or committing staged edits.
    ///
    /// # Panics
    ///
    /// Panics if the append buffer's capacity exceeds [`isize::MAX`] bytes.
    pub fn append<T: H5Element>(
        self,
        file: &File,
        dataset: &mut Dataset,
        data: &[T],
    ) -> Result<(), Error> {
        match self {
            Self::Buffered => {
                let mut appender = dataset.buffered_appender()?;
                if data.is_empty() {
                    appender.append(data)?;
                }
                for batch in data.chunks(2) {
                    appender.append(batch)?;
                }
                appender.finish()
            }
            Self::Immediate => dataset.append(data),
            Self::OwnedBuffered => {
                let mut appender = dataset.clone().into_buffered_appender()?;
                if data.is_empty() {
                    appender.append(data)?;
                }
                for batch in data.chunks(2) {
                    appender.append(batch)?;
                }
                appender.finish()
            }
            Self::Staged => {
                dataset.append_staged(|append| {
                    append.append(data);
                })?;
                file.commit()
            }
        }
    }
}

/// A filter pipeline used by append tests.
#[derive(Clone, Copy, Debug)]
pub enum AppendFilters {
    /// Storage without filters.
    Plain,
    /// Shuffle followed by deflate at level 6.
    ShuffleDeflate,
}

impl AppendFilters {
    /// Returns the ordered pipeline used to create the dataset.
    pub fn filters(self) -> &'static [Filter] {
        match self {
            Self::Plain => &[],
            Self::ShuffleDeflate => SHUFFLE_DEFLATE_6,
        }
    }
}
