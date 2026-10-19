//! Publication-failure injection for PAGE interoperability tests.

use std::io::Error as IoError;

use crate::Error;
use crate::FormatError;
use crate::image::FileImage;
use crate::image::MirrorImage;
use crate::image::WriteBuffering;
use crate::source::Source;

/// A publication phase at which a PAGE crosscheck reports one I/O error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagePublicationFailure {
    /// Reports an error after the publication barrier and any requested synchronization.
    AfterDrain,
    /// Reports an error after draining the superblock write, before the editor calls its
    /// publication barrier.
    AfterPublish,
    /// Reports an error after issuing the superblock write, before any requested synchronization.
    PublishedSync,
}

/// A mirror that reports one error at the selected publication phase.
pub(crate) struct PublicationFailureImage {
    inner: MirrorImage,
    /// The physical offset of the superblock signature.
    publication_offset: u64,
    failure: PagePublicationFailure,
    published: bool,
    failed: bool,
}

impl PublicationFailureImage {
    /// Wraps a mirror to inject an error when publishing its superblock.
    pub(crate) fn new(
        inner: MirrorImage,
        publication_offset: u64,
        failure: PagePublicationFailure,
    ) -> Self {
        Self {
            inner,
            publication_offset,
            failure,
            published: false,
            failed: false,
        }
    }

    /// Issues pending writes, optionally synchronizes them, and injects the selected error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] at the selected phase, or an error from the mirror's barrier.
    fn barrier(&mut self, durable: bool) -> Result<(), Error> {
        self.inner.ordering_barrier()?;
        if !self.failed && self.published && self.failure == PagePublicationFailure::PublishedSync {
            return self.fail();
        }
        if durable {
            self.inner.sync_all()?;
        }
        if !self.failed && self.published && self.failure == PagePublicationFailure::AfterDrain {
            return self.fail();
        }
        Ok(())
    }

    fn fail(&mut self) -> Result<(), Error> {
        self.failed = true;
        Err(IoError::other("injected PAGE publication failure").into())
    }
}

impl FileImage for PublicationFailureImage {
    fn read_disk_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_disk_at(offset, bytes)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.inner.write_at(offset, bytes)?;
        if offset == self.publication_offset {
            self.published = true;
            if !self.failed && self.failure == PagePublicationFailure::AfterPublish {
                self.inner.ordering_barrier()?;
                return self.fail();
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        self.inner.truncate(len)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.barrier(true)
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.barrier(false)
    }

    #[cfg(test)]
    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.inner.issued_write_order()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(mode)
    }

    fn as_slice(&self) -> Option<&[u8]> {
        self.inner.as_slice()
    }
}

impl Source for PublicationFailureImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, bytes)
    }
}
