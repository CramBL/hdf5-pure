//! Failure injection and barrier recording for PAGE publication tests.

use std::io::Error as IoError;
#[cfg(test)]
use std::sync::{Arc, Mutex};

#[cfg(test)]
use test_util::disk_log;

use crate::Error;
use crate::FormatError;
use crate::image::FileImage;
use crate::image::MirrorImage;
use crate::image::WriteBuffering;
use crate::source::Source;

/// A failure phase for PAGE publication tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagePublicationFailure {
    /// Reports an error after the publication barrier and any requested synchronization.
    AfterDrain,
    /// Reports an error after draining the superblock write, before the editor calls its
    /// publication barrier.
    AfterPublish,
    /// Appends half the buffer to the image before reporting an error.
    #[cfg(test)]
    Append,
    /// Reports an error before draining the content writes.
    #[cfg(test)]
    BeforeDrain,
    /// Reports an error before accepting the superblock write.
    #[cfg(test)]
    BeforePublish,
    /// Reports an error during the content barrier after draining its writes.
    #[cfg(test)]
    ContentSync,
    /// The number of disk writes to skip and the prefix length to write before reporting an error.
    #[cfg(test)]
    DiskWrite(usize, usize),
    /// The superblock prefix length to write before reporting an error.
    #[cfg(test)]
    PublicationWrite(usize),
    /// Reports errors at the publication barrier and each subsequent direct file read.
    #[cfg(test)]
    PublishedRead,
    /// Reports an error after issuing the superblock write, before any requested synchronization.
    PublishedSync,
    /// Reports an error at every publication barrier, including recovery attempts.
    #[cfg(test)]
    RepeatedPublishedSync,
}

/// A mirror that injects errors during PAGE publication tests.
pub(crate) struct PublicationFailureImage {
    inner: MirrorImage,
    /// The physical offset of the superblock signature.
    publication_offset: u64,
    failure: PagePublicationFailure,
    published: bool,
    failed: bool,
    #[cfg(test)]
    barriers: Option<Arc<Mutex<Vec<BarrierRecord>>>>,
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
            #[cfg(test)]
            barriers: None,
        }
    }

    /// Wraps a mirror and records the caller's barrier attempts.
    ///
    /// Reads use the source interface. Truncation and data-only synchronization return errors.
    #[cfg(test)]
    pub(crate) fn with_barriers(
        inner: MirrorImage,
        publication_offset: u64,
        failure: PagePublicationFailure,
        barriers: Arc<Mutex<Vec<BarrierRecord>>>,
    ) -> Self {
        let mut image = Self::new(inner, publication_offset, failure);
        image.barriers = Some(barriers);
        image
    }

    /// Issues pending writes, optionally synchronizes them, and injects the selected error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] at the selected phase, or an error from the mirror's barrier.
    fn barrier(&mut self, durable: bool) -> Result<(), Error> {
        #[cfg(test)]
        let index = self.barriers.as_ref().map(|barriers| {
            let mut barriers = barriers.lock().unwrap();
            let index = barriers.len();
            barriers.push(BarrierRecord {
                durable,
                published: self.published,
                completed: false,
            });
            index
        });
        #[cfg(test)]
        if !self.failed
            && !self.published
            && matches!(self.failure, PagePublicationFailure::BeforeDrain)
        {
            self.fail()?;
        }
        self.inner.ordering_barrier()?;
        if !self.failed && self.published && self.failure == PagePublicationFailure::PublishedSync {
            self.fail()?;
        }
        #[cfg(test)]
        if (self.published && matches!(self.failure, PagePublicationFailure::RepeatedPublishedSync))
            || (!self.failed
                && ((!self.published
                    && matches!(self.failure, PagePublicationFailure::ContentSync))
                    || (self.published
                        && matches!(self.failure, PagePublicationFailure::PublishedRead))))
        {
            self.fail()?;
        }
        if durable {
            self.inner.sync_all()?;
        }
        #[cfg(test)]
        if let (Some(barriers), Some(index)) = (&self.barriers, index) {
            barriers.lock().unwrap()[index].completed = true;
        }
        if !self.failed
            && self.published
            && matches!(self.failure, PagePublicationFailure::AfterDrain)
        {
            self.fail()?;
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
        #[cfg(test)]
        if self.published && matches!(self.failure, PagePublicationFailure::PublishedRead) {
            return Err(FormatError::Source(
                "injected PAGE publication read failure".into(),
            ));
        }
        self.inner.read_disk_at(offset, bytes)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        #[cfg(test)]
        if !self.failed && matches!(self.failure, PagePublicationFailure::Append) {
            self.inner.append(&bytes[..bytes.len() / 2])?;
            self.fail()?;
        }
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        #[cfg(test)]
        if !self.failed
            && offset == self.publication_offset
            && let PagePublicationFailure::PublicationWrite(prefix) = self.failure
        {
            self.failed = true;
            disk_log::fail_write_after(0, prefix);
        }
        #[cfg(test)]
        if !self.failed
            && offset == self.publication_offset
            && matches!(self.failure, PagePublicationFailure::BeforePublish)
        {
            self.fail()?;
        }
        self.inner.write_at(offset, bytes)?;
        if offset == self.publication_offset {
            self.published = true;
            if !self.failed && matches!(self.failure, PagePublicationFailure::AfterPublish) {
                self.inner.ordering_barrier()?;
                self.fail()?;
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        #[cfg(test)]
        if self.barriers.is_some() {
            return Err(IoError::other("unexpected PAGE truncation").into());
        }
        self.inner.truncate(len)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        #[cfg(test)]
        if self.barriers.is_some() {
            return Err(IoError::other("unexpected data-only PAGE synchronization").into());
        }
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
        #[cfg(test)]
        if self.barriers.is_some() {
            return None;
        }
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

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BarrierRecord {
    /// Whether the caller requested durable synchronization.
    pub(crate) durable: bool,
    /// Whether the image accepted the superblock write before this barrier.
    pub(crate) published: bool,
    /// Whether the inner barrier completed, including synchronization when requested.
    pub(crate) completed: bool,
}
