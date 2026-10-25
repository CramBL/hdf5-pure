//! Internal hooks for PAGE publication crosschecks.

use std::path::Path;

use crate::Error;
use crate::File;
use crate::SyncPolicy;

pub use crate::page_publication_failure::PagePublicationFailure;

/// Opens a mirrored non-persistent PAGE file with one injected publication error.
///
/// The editor uses `sync` for publication barriers and gathers writes when `gathered` is true.
/// The editor reports [`Error::Io`] at the phase specified by `failure`.
///
/// # Examples
///
/// ```
/// use std::io::ErrorKind;
///
/// use hdf5_pure::{Error, File, FileBuilder, FileSpaceSettings, FileSpaceStrategy, SyncPolicy};
/// use hdf5_pure::__private::{self, PagePublicationFailure};
///
/// # let directory = tempfile::tempdir()?;
/// # let path = directory.path().join("publication.h5");
/// let mut builder = FileBuilder::new();
/// builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
/// builder.create_dataset("keep").with_u8_data(&[7; 64]);
/// builder.write(&path)?;
/// let file = __private::open_page_publication_failure(
///     &path, PagePublicationFailure::AfterPublish, SyncPolicy::OnClose, true,
/// )?;
/// file.root().create_dataset("added", |dataset| {
///     dataset.with_u8_data(&[9; 64]);
/// })?;
/// let error = file.commit().unwrap_err();
/// let Error::Io(error) = error else {
///     panic!("expected injected publication failure, got {error:?}");
/// };
/// assert_eq!(error.kind(), ErrorKind::Other);
/// assert_eq!(error.to_string(), "injected PAGE publication failure");
/// assert_eq!(file.dataset("added")?.read_u8()?, [9; 64]);
/// file.root().create_dataset("second", |dataset| {
///     dataset.with_u8_data(&[13; 64]);
/// })?;
/// file.commit()?;
/// drop(file);
/// let reopened = File::open(&path)?;
/// assert_eq!(reopened.root().datasets()?, ["keep", "added", "second"]);
/// assert_eq!(reopened.dataset("keep")?.read_u8()?, [7; 64]);
/// assert_eq!(reopened.dataset("added")?.read_u8()?, [9; 64]);
/// assert_eq!(reopened.dataset("second")?.read_u8()?, [13; 64]);
/// # Ok::<(), Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the file uses another strategy or enables PAGE
/// persistence. Returns the opening or configuration error if the editor cannot open the file.
pub fn open_page_publication_failure(
    path: &Path,
    failure: PagePublicationFailure,
    sync: SyncPolicy,
    gathered: bool,
) -> Result<File, Error> {
    File::open_page_publication_failure(path, failure, sync, gathered)
}

/// Reads issued file bytes while leaving pending writes buffered.
///
/// # Examples
///
/// ```
/// use std::io::ErrorKind;
///
/// use hdf5_pure::{Error, File, FileBuilder, FileSpaceSettings, FileSpaceStrategy, SyncPolicy};
/// use hdf5_pure::__private::{self, PagePublicationFailure};
///
/// # let directory = tempfile::tempdir()?;
/// # let path = directory.path().join("snapshot.h5");
/// # let mut builder = FileBuilder::new();
/// # builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
/// # builder.create_dataset("keep").with_u8_data(&[7; 64]);
/// # builder.write(&path)?;
/// let original = std::fs::read(&path)?;
/// let file = __private::open_page_publication_failure(
///     &path, PagePublicationFailure::AfterDrain, SyncPolicy::OnClose, true,
/// )?;
/// file.root().create_dataset("added", |dataset| {
///     dataset.with_u8_data(&[9; 64]);
/// })?;
/// assert_eq!(__private::page_publication_snapshot(&file)?, original);
/// let error = file.commit().unwrap_err();
/// let Error::Io(error) = error else {
///     panic!("expected injected publication failure, got {error:?}");
/// };
/// assert_eq!(error.kind(), ErrorKind::Other);
/// assert_eq!(error.to_string(), "injected PAGE publication failure");
/// let published = __private::page_publication_snapshot(&file)?;
/// let snapshot = File::from_bytes(published.clone())?;
/// assert_eq!(snapshot.root().datasets()?, ["keep", "added"]);
/// assert_eq!(snapshot.dataset("keep")?.read_u8()?, [7; 64]);
/// assert_eq!(snapshot.dataset("added")?.read_u8()?, [9; 64]);
/// assert_eq!(__private::page_publication_snapshot(&file)?, published);
/// drop(file);
/// assert_eq!(std::fs::read(&path)?, published);
/// # Ok::<(), Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::ReadOnly`] if `file` has no editing session. Returns the read or conversion
/// error if the editor cannot read the complete image length through the disk handle.
pub fn page_publication_snapshot(file: &File) -> Result<Vec<u8>, Error> {
    file.page_publication_snapshot()
}
