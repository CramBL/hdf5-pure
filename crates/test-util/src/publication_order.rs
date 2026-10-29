//! Checks allocation publication order in recorded file I/O.
//!
//! Callers identify content writes, the ownership publication, and the first reference from
//! independently parsed file metadata. [`Publication::parse`] checks their order and reported
//! outcomes. [`Publication::require`] checks the synchronization required by the selected policy.

use crate::disk_log::DiskOp;
use crate::disk_log::IoEvent;
use crate::disk_log::IoOutcome;
use crate::disk_log::Recording;
use crate::disk_log::Synchronization;

/// The synchronization required between allocation publication phases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    /// Requires successful synchronization before ownership publication and before references.
    Always,
    /// Requires call ordering and successful I/O, without synchronization between phases.
    OnClose,
}

/// Whether ownership publication extends the recorded end of file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EofChange {
    /// An extension requiring metadata synchronization under [`Policy::Always`].
    Extend,
    /// An unchanged end of file requiring data synchronization under [`Policy::Always`].
    Unchanged,
}

/// An invalid publication trace, with indices into [`Recording::events`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrderError {
    /// Successful content synchronization is missing before ownership publication.
    ContentBarrier {
        /// The last content event.
        content: usize,
        /// The ownership publication event.
        publication: usize,
    },
    /// The caller supplied no content writes.
    EmptyContent,
    /// An I/O call failed between the first content write and the reference.
    FailedIo {
        /// The event that failed the check.
        event: usize,
        outcome: IoOutcome,
    },
    /// A supplied index does not identify a successful, complete, nonempty replayed write.
    InvalidWrite {
        /// The event that failed the check.
        event: usize,
    },
    /// The required synchronization is missing between ownership publication and the reference.
    PublicationBarrier {
        /// The ownership publication event.
        publication: usize,
        /// The first reference event.
        reference: usize,
        /// The synchronization required after publication.
        required: Synchronization,
    },
    /// Content, ownership publication, and the reference are out of order or share a write.
    ReferenceOrder {
        /// The last content event.
        content: usize,
        /// The ownership publication event.
        publication: usize,
        /// The first reference event.
        reference: usize,
    },
    /// A write before ownership publication is missing from the supplied content indices.
    UnlistedContent {
        /// The event that failed the check.
        event: usize,
    },
}

/// A recorded publication with complete content writes ordered before ownership and references.
///
/// The supplied event indices are checked for I/O ordering. The caller establishes allocation
/// ownership and the meaning of each write.
#[derive(Debug)]
pub struct Publication<'a> {
    recording: &'a Recording,
    content_end: usize,
    publication: usize,
    reference: usize,
    eof: EofChange,
}

impl<'a> Publication<'a> {
    /// Checks successful writes in content, ownership publication, and reference order.
    ///
    /// All indices are into [`Recording::events`]. Each supplied write must match its completed
    /// [`DiskOp`]. All writes from the earliest content event up to `publication` must be listed in
    /// `content`. Every I/O call from the earliest content event through `reference` must have
    /// succeeded.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::EmptyContent`] if `content` is empty, or [`OrderError::InvalidWrite`]
    /// if a supplied index does not identify a complete, nonempty, successful replayed write.
    /// Returns [`OrderError::ReferenceOrder`] if the phases overlap or are out of order,
    /// [`OrderError::FailedIo`] if an intervening call failed, or [`OrderError::UnlistedContent`]
    /// if an intervening write is missing from `content`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Seek, SeekFrom};
    ///
    /// use test_util::disk_log::{self, Synchronization};
    /// use test_util::publication_order::{EofChange, Policy, Publication};
    ///
    /// let directory = tempfile::tempdir()?;
    /// let mut file = tempfile::tempfile_in(directory.path())?;
    /// disk_log::start();
    /// file.seek(SeekFrom::Start(8))?;
    /// disk_log::write_all(&mut file, 8, b"content")?;
    /// disk_log::synchronize(&file, Synchronization::Data)?;
    /// file.seek(SeekFrom::Start(0))?;
    /// disk_log::write_all(&mut file, 0, b"owned")?;
    /// disk_log::synchronize(&file, Synchronization::All)?;
    /// file.seek(SeekFrom::Start(5))?;
    /// disk_log::write_all(&mut file, 5, b"ref")?;
    /// let recording = disk_log::finish();
    /// let publication = Publication::parse(&recording, &[0], 2, 4, EofChange::Extend).unwrap();
    /// assert_eq!(publication.require(Policy::Always), Ok(()));
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn parse(
        recording: &'a Recording,
        content: &[usize],
        publication: usize,
        reference: usize,
        eof: EofChange,
    ) -> Result<Self, OrderError> {
        let content_start = *content.iter().min().ok_or(OrderError::EmptyContent)?;
        let content_end = *content.iter().max().ok_or(OrderError::EmptyContent)?;
        for &event in content.iter().chain([publication, reference].iter()) {
            let Some(IoEvent::Write {
                offset,
                requested,
                written,
                outcome: IoOutcome::Succeeded,
                operation: Some(operation),
            }) = recording.events.get(event)
            else {
                return Err(OrderError::InvalidWrite { event });
            };
            if requested != written
                || *written == 0
                || !matches!(recording.operations.get(*operation), Some(DiskOp::Write { offset: at, bytes }) if offset == at && bytes.len() == *written)
            {
                return Err(OrderError::InvalidWrite { event });
            }
        }
        if content_end >= publication || publication >= reference {
            return Err(OrderError::ReferenceOrder {
                content: content_end,
                publication,
                reference,
            });
        }
        for (event, entry) in recording.events[content_start..=reference]
            .iter()
            .enumerate()
        {
            if entry.outcome() != IoOutcome::Succeeded {
                return Err(OrderError::FailedIo {
                    event: content_start + event,
                    outcome: entry.outcome(),
                });
            }
            if content_start + event < publication
                && matches!(entry, IoEvent::Write { .. })
                && !content.contains(&(content_start + event))
            {
                return Err(OrderError::UnlistedContent {
                    event: content_start + event,
                });
            }
        }
        Ok(Self {
            recording,
            content_end,
            publication,
            reference,
            eof,
        })
    }

    /// Checks synchronization between the parsed publication phases.
    ///
    /// [`Policy::Always`] requires successful synchronization after the last content write and
    /// after ownership publication. The latter must synchronize metadata when the recorded end of
    /// file extends. [`Policy::OnClose`] accepts the ordering established by [`Self::parse`].
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::ContentBarrier`] if content synchronization is missing, or
    /// [`OrderError::PublicationBarrier`] if ownership synchronization is missing or too weak.
    pub fn require(&self, policy: Policy) -> Result<(), OrderError> {
        if policy == Policy::OnClose {
            return Ok(());
        }
        if !self.recording.events[self.content_end + 1..self.publication]
            .iter()
            .any(|event| {
                matches!(
                    event,
                    IoEvent::Synchronize {
                        outcome: IoOutcome::Succeeded,
                        ..
                    }
                )
            })
        {
            return Err(OrderError::ContentBarrier {
                content: self.content_end,
                publication: self.publication,
            });
        }
        let required = match self.eof {
            EofChange::Extend => Synchronization::All,
            EofChange::Unchanged => Synchronization::Data,
        };
        if !self.recording.events[self.publication + 1..self.reference]
            .iter()
            .any(|event| {
                matches!(event, IoEvent::Synchronize { kind, outcome: IoOutcome::Succeeded }
                    if *kind == Synchronization::All || *kind == required)
            })
        {
            return Err(OrderError::PublicationBarrier {
                publication: self.publication,
                reference: self.reference,
                required,
            });
        }
        Ok(())
    }
}
