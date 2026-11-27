use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;

use super::{OBJECT_HEADER_PREFIX_MAX_LEN, ObjectHeaderPrefix};
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source;
use crate::metadata_source::MetadataSource;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::object_header::ObjectHeaderChunk;
use crate::object_header::context::ParseContext;
use crate::object_header::v2::chunk::RootChunk;
use crate::object_header::v2::collector::HeaderCollector;

impl ObjectHeader {
    /// Reads the initial chunk at the absolute byte offset `offset` and follows its continuations.
    ///
    /// # Errors
    ///
    /// Propagates errors from locating chunks, parsing their framing, and reading their records.
    pub(in crate::object_header) fn parse_v2(
        data: &[u8],
        context: ParseContext,
        offset: usize,
        filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        bytes::ensure_len(data, offset, 6)?;
        let prefix = ObjectHeaderPrefix::parse(&data[offset..])?;
        let image = metadata_source::region(data, offset.to_u64(), prefix.chunk0_len()?)?;
        let mut collector =
            HeaderCollector::new(context, filter, RootChunk::parse(prefix, image)?)?;
        while let Some(block) = collector.next_continuation()? {
            let bytes = metadata_source::region(data, block.address, block.length.to_usize()?)?;
            collector.ingest(block.parse(bytes)?)?;
        }
        Ok(collector.finish())
    }

    /// Parses a version 2 header and observes each chunk after checking its framing.
    ///
    /// Checksum verification follows the `checksum` feature. Calls the observer before checking
    /// message compatibility and filtering records. Continuations are visited breadth-first.
    ///
    /// # Errors
    ///
    /// Returns framing, message-compatibility, address-conversion, and source-read errors.
    pub(in crate::object_header) fn parse_v2_from_source<S: MetadataSource + ?Sized>(
        source: &S,
        context: ParseContext,
        address: u64,
        filter: MessageFilter<'_>,
        observer: &mut Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    ) -> Result<ObjectHeader, FormatError> {
        let head_len = OBJECT_HEADER_PREFIX_MAX_LEN
            .to_u64()
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let prefix = ObjectHeaderPrefix::parse(&source.read_metadata(address, head_len)?)?;
        let image = source.read_metadata(address, prefix.chunk0_len()?)?;
        let root = RootChunk::parse(prefix, &image)?;
        if let Some(observer) = observer.as_deref_mut() {
            observer(root.chunk.observation(FileRange::new(
                PhysicalAddress::new(address),
                image.len().to_u64(),
            )?)?);
        }
        let mut collector = HeaderCollector::new(context, filter, root)?;
        while let Some(block) = collector.next_continuation()? {
            let bytes = source.read_metadata(block.address, block.length.to_usize()?)?;
            let chunk = block.parse(&bytes)?;
            if let Some(observer) = observer.as_deref_mut() {
                observer(chunk.observation(FileRange::new(
                    PhysicalAddress::new(block.address),
                    block.length,
                )?)?);
            }
            collector.ingest(chunk)?;
        }
        Ok(collector.finish())
    }
}
