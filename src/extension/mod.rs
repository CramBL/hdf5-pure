//! Decoded superblock-extension facts within an unchanged source phase.

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectHeaderChunk;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::error::FormatError;
use crate::file_space_info::FileSpaceInfo;
use crate::message_type::MessageType;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::sohm::SharedMessageTableMessage;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::FormatWidths;

/// Independent first-message outcomes under an immutable source borrow.
///
/// Header parsing completes before a summary is returned. File Space Info and Shared Message
/// Table decoding errors are stored independently. The caller keeps the source unchanged while
/// using the summary.
pub(crate) struct ExtensionSummary<'a, S: Source + ?Sized> {
    _source: &'a S,
    address: PhysicalAddress,
    file_space: Result<Option<FileSpaceInfo>, FormatError>,
    shared_table: Result<Option<SharedMessageTableMessage>, FormatError>,
}

impl<'a, S: Source + ?Sized> ExtensionSummary<'a, S> {
    /// Reads the first File Space Info and Shared Message Table bodies during a complete parse.
    ///
    /// Decodes the selected bodies without retaining raw messages. An optional observer receives
    /// borrowed chunk views from the same traversal.
    ///
    /// # Errors
    ///
    /// Returns header and source-read errors from [`read_header`]. Message-body errors remain
    /// separate outcomes in the returned summary.
    pub(crate) fn read(
        source: &'a S,
        mode: AccessMode,
        address: PhysicalAddress,
        widths: FormatWidths,
        addresses: AddressContext,
        observer: Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    ) -> Result<Self, FormatError> {
        Self::read_selected(
            source,
            mode,
            address,
            widths,
            addresses,
            observer,
            FileSpaceRead::Decode,
        )
    }

    /// Selects File Space Info decoding while reading the complete extension header.
    ///
    /// # Errors
    ///
    /// Returns header and source-read errors from [`read_header`].
    fn read_selected(
        source: &'a S,
        mode: AccessMode,
        address: PhysicalAddress,
        widths: FormatWidths,
        addresses: AddressContext,
        observer: Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
        file_space_read: FileSpaceRead,
    ) -> Result<Self, FormatError> {
        let mut file_space = None;
        let mut shared_table = SharedTableSummary { first: None };
        let mut capture = |kind, body: &[u8]| {
            if kind == MessageType::FILE_SPACE_INFO
                && file_space_read == FileSpaceRead::Decode
                && file_space.is_none()
            {
                file_space = Some(hdf5_pure_format::__private::parse_file_space_info(
                    widths, body,
                ));
            }
            shared_table.observe(widths, kind, body);
            false
        };
        read_header(
            source,
            mode,
            address,
            widths,
            addresses,
            observer,
            &mut capture,
        )?;
        Ok(Self {
            _source: source,
            address,
            file_space: file_space.transpose(),
            shared_table: shared_table.into_message(),
        })
    }

    /// Returns the first File Space Info message with the extension's physical address.
    ///
    /// Returns `None` if that message is absent.
    ///
    /// # Errors
    ///
    /// Returns the retained decoding error from the first File Space Info body.
    pub(crate) fn into_file_space_info(
        self,
    ) -> Result<Option<(PhysicalAddress, FileSpaceInfo)>, FormatError> {
        Ok(self.file_space?.map(|info| (self.address, info)))
    }

    /// Returns the first Shared Message Table message, or `None` if absent.
    ///
    /// # Errors
    ///
    /// Returns the retained decoding error from the first Shared Message Table body.
    fn into_shared_table(self) -> Result<Option<SharedMessageTableMessage>, FormatError> {
        self.shared_table
    }
}

/// The first Shared Message Table outcome after a complete header parse.
///
/// A readable message-body error is retained for [`into_message`](Self::into_message), separately
/// from a failure to parse the header. The caller uses the result within its unchanged source phase.
pub(crate) struct SharedTableSummary {
    first: Option<Result<SharedMessageTableMessage, FormatError>>,
}

impl SharedTableSummary {
    /// Reads the shared-table outcome while skipping File Space Info decoding.
    ///
    /// # Errors
    ///
    /// Returns header and source-read errors from [`read_header`]. A message-body error remains
    /// in the returned summary for [`into_message`](Self::into_message).
    pub(crate) fn read(
        source: &(impl Source + ?Sized),
        mode: AccessMode,
        address: PhysicalAddress,
        widths: FormatWidths,
        addresses: AddressContext,
    ) -> Result<Self, FormatError> {
        let summary = ExtensionSummary::read_selected(
            source,
            mode,
            address,
            widths,
            addresses,
            None,
            FileSpaceRead::Skip,
        )?;
        Ok(Self {
            first: summary.into_shared_table().transpose(),
        })
    }

    /// Returns the first Shared Message Table message, or `None` if absent.
    ///
    /// # Errors
    ///
    /// Returns the retained decoding error from the first Shared Message Table body.
    pub(crate) fn into_message(self) -> Result<Option<SharedMessageTableMessage>, FormatError> {
        self.first.transpose()
    }

    /// Records the first Shared Message Table decoding outcome.
    fn observe(&mut self, widths: FormatWidths, kind: MessageType, body: &[u8]) {
        if kind == MessageType::SHARED_MESSAGE_TABLE && self.first.is_none() {
            self.first = Some(SharedMessageTableMessage::parse_with_offset_width(
                body,
                widths.offsets,
            ));
        }
    }
}

/// Whether the selected extension query decodes File Space Info.
#[derive(Eq, PartialEq)]
enum FileSpaceRead {
    Decode,
    Skip,
}

/// Reads a complete header with message filtering and optional chunk observations.
///
/// # Errors
///
/// Returns errors from [`ObjectHeader::parse_from_source_with_chunk_observer`] or
/// [`ObjectHeader::parse_from_source_filtered_with_widths`].
fn read_header(
    source: &(impl Source + ?Sized),
    mode: AccessMode,
    address: PhysicalAddress,
    widths: FormatWidths,
    addresses: AddressContext,
    observer: Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    capture: &mut dyn FnMut(MessageType, &[u8]) -> bool,
) -> Result<(), FormatError> {
    match observer {
        Some(observer) => ObjectHeader::parse_from_source_with_chunk_observer(
            &SourceMetadata(source),
            mode,
            address,
            widths,
            addresses.effective_base(),
            MessageFilter::Only(capture),
            observer,
        ),
        None => ObjectHeader::parse_from_source_filtered_with_widths(
            &SourceMetadata(source),
            mode,
            address.get(),
            widths,
            addresses.effective_base(),
            MessageFilter::Only(capture),
        ),
    }
    .map(|_| ())
}

#[cfg(test)]
mod tests;
