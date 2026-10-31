//! Owned physical references and decoded bytes for fractal-heap huge objects.
//!
//! The caller reads tracker metadata and encoded payloads using the returned physical ranges.
//! Filter execution and B-tree traversal belong to the caller. Decoded payload bytes are borrowed
//! through [`FractalHeapHugeObjectView`], with their encoded allocation retained separately.

use alloc::sync::Arc;

use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion};
use hdf5_pure_core::BaseAddress;
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::btree_v2::{
    self, BTreeV2Header, HugeObjectRecordContext, HugeObjectRecordDecoder, HugeObjectTrackerRecord,
};
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::fractal_heap::{
    FilteredHugeObjectLocation, FractalHeapFiltering, FractalHeapHeader, FractalHeapIdKind,
    FractalHeapIdLayout, FractalHeapIdView, HugeIndirectKey, HugeIndirectKeyEncoding,
    HugeObjectReference,
};
use crate::width::FormatWidths;

/// The address base, field widths, and filter context of a heap's huge objects.
///
/// Each independently constructed layout has its own identity. Clones and derived references
/// share that identity. The caller reads metadata and payloads from one consistent source revision.
#[derive(Clone, Debug)]
pub struct FractalHeapHugeObjectLayout {
    context: Arc<HugeObjectContext>,
}

impl FractalHeapHugeObjectLayout {
    /// Derives huge-object decoding parameters from the heap header and file context.
    ///
    /// The reader converts stored addresses to physical positions using `base`. The caller
    /// selects the indirect key width with `encoding`. A defined huge-object tracker is required
    /// for both ID forms.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the tracker address is undefined
    /// or exceeds the offset width, or the pipeline contains more than 32 filters. Returns
    /// [`FormatError::OffsetOverflow`] if the physical tracker header range exceeds [`u64::MAX`].
    pub fn new(
        header: &FractalHeapHeader,
        widths: FormatWidths,
        base: BaseAddress,
        encoding: HugeIndirectKeyEncoding,
    ) -> Result<Self, FormatError> {
        let records = HugeObjectRecordContext::new(header, widths)?;
        let tracker_range = FileRange::new(
            PhysicalAddress::new(base.absolute(header.btree_huge_objects_address)?),
            btree_v2::btree_v2_header_size(widths.offsets, widths.lengths) as u64,
        )?;
        Ok(Self {
            context: Arc::new(HugeObjectContext {
                base,
                widths,
                ids: FractalHeapIdLayout::with_huge_key_encoding(
                    header,
                    widths.offsets,
                    widths.lengths,
                    encoding,
                ),
                records,
                tracker_range,
                pipeline: match &header.filtering {
                    FractalHeapFiltering::Filtered { pipeline, .. } => Some(pipeline.clone()),
                    FractalHeapFiltering::Unfiltered => None,
                },
            }),
        })
    }

    /// Returns an encoded payload reference or a numeric tracker lookup from a huge heap ID.
    ///
    /// Direct IDs are resolved without reading the tracker. The returned payload reference
    /// holds a physical range, with its stored address converted using this layout's base.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`FractalHeapIdLayout::parse`] for an invalid ID, and
    /// [`FormatError::InvalidFractalHeapGeometry`] for a managed or tiny ID. Returns
    /// [`FormatError::OffsetOverflow`] if the physical payload range exceeds [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_core::BaseAddress;
    /// # use hdf5_pure_core::__private::{
    /// #     BaseAddressExt, FileRange, FileRegion, StoredAddressExt,
    /// # };
    /// # use hdf5_pure_core::PhysicalAddress;
    /// # use hdf5_pure_format::FormatError;
    /// # use hdf5_pure_format::__private::{
    /// #     FormatWidths, FractalHeapHeader, FractalHeapHugeObjectId, FractalHeapHugeObjectLayout,
    /// #     HugeIndirectKeyEncoding, LengthWidth, OffsetWidth,
    /// # };
    /// # use test_util::{fractal_heap, widths::Widths};
    /// # let encoded = fractal_heap::Header::new(u64::MAX).build(Widths::EIGHT);
    /// # let mut header = FractalHeapHeader::parse_bytes(
    /// #     &encoded, OffsetWidth::Eight, LengthWidth::Eight,
    /// # )?;
    /// # header.heap_id_length = 17;
    /// # header.btree_huge_objects_address = hdf5_pure_core::StoredAddress::new(256);
    /// # let widths = FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight };
    /// let objects = FractalHeapHugeObjectLayout::new(
    ///     &header, widths, BaseAddress::new(512), HugeIndirectKeyEncoding::Libhdf5,
    /// )?;
    /// let id = [0x10, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0];
    /// let FractalHeapHugeObjectId::Direct(reference) = objects.parse_id(&id)? else {
    ///     unreachable!();
    /// };
    /// assert_eq!(reference.range(), FileRange::new(PhysicalAddress::new(1536), 3)?);
    /// let region = FileRegion::new(reference.range(), b"abc")?;
    /// assert_eq!(reference.parse(region)?, b"abc");
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn parse_id(&self, bytes: &[u8]) -> Result<FractalHeapHugeObjectId, FormatError> {
        self.resolve_id(self.context.ids.parse(bytes)?)
    }

    /// Resolves a parsed huge ID using this layout's widths, address base, and filter context.
    ///
    /// The parser checks the ID's direct or indirect representation and its filtered or
    /// unfiltered form against the heap's parameters. Fields parsed under another context are
    /// checked against this layout's numeric widths and filter pipeline before address resolution.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the ID has another representation,
    /// a managed or tiny type, an undefined or oversized address, a zero length, or a filter mask
    /// incompatible with this layout's pipeline. Also returns this error if all filters are
    /// skipped and the encoded and decoded sizes differ.
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if a size or numeric key exceeds
    /// this layout's length width, or [`FormatError::OffsetOverflow`] if the physical payload
    /// range exceeds [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_core::BaseAddress;
    /// # use hdf5_pure_core::__private::{BaseAddressExt, StoredAddressExt};
    /// # use hdf5_pure_format::FormatError;
    /// # use hdf5_pure_format::__private::{
    /// #     FormatWidths, FractalHeapHeader, FractalHeapHugeObjectId, FractalHeapHugeObjectLayout,
    /// #     FractalHeapIdLayout, HugeIndirectKeyEncoding, LengthWidth, OffsetWidth,
    /// # };
    /// # use test_util::{fractal_heap, widths::Widths};
    /// # let encoded = fractal_heap::Header::new(u64::MAX).build(Widths::EIGHT);
    /// # let mut header = FractalHeapHeader::parse_bytes(
    /// #     &encoded, OffsetWidth::Eight, LengthWidth::Eight,
    /// # )?;
    /// # header.heap_id_length = 17;
    /// # header.btree_huge_objects_address = hdf5_pure_core::StoredAddress::new(256);
    /// # let widths = FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight };
    /// # let objects = FractalHeapHugeObjectLayout::new(
    /// #     &header, widths, BaseAddress::new(512), HugeIndirectKeyEncoding::Libhdf5,
    /// # )?;
    /// let ids = FractalHeapIdLayout::new(&header, widths.offsets, widths.lengths);
    /// let bytes = [0x10, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0];
    /// let id = ids.parse(&bytes)?;
    /// let FractalHeapHugeObjectId::Direct(reference) = objects.resolve_id(id)? else {
    ///     unreachable!();
    /// };
    /// assert_eq!(reference.range().start().get(), 1536);
    /// assert_eq!(reference.range().length(), 3);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn resolve_id(
        &self,
        id: FractalHeapIdView<'_>,
    ) -> Result<FractalHeapHugeObjectId, FormatError> {
        let kind = id.kind();
        let tree_type = self.context.records.tree_type();
        let matches = matches!(
            (tree_type, kind),
            (
                crate::btree_v2::BTREE_V2_HUGE_OBJECT,
                FractalHeapIdKind::Huge(HugeObjectReference::Indexed(_))
            ) | (
                crate::btree_v2::BTREE_V2_FILTERED_HUGE_OBJECT,
                FractalHeapIdKind::Huge(HugeObjectReference::FilteredIndexed(_))
            ) | (
                crate::btree_v2::BTREE_V2_HUGE_OBJECT_DIRECT,
                FractalHeapIdKind::Huge(HugeObjectReference::Inline { .. })
            ) | (
                crate::btree_v2::BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT,
                FractalHeapIdKind::Huge(HugeObjectReference::FilteredInline(_))
            )
        );
        if !matches {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(match kind {
            FractalHeapIdKind::Huge(HugeObjectReference::Inline { addr, len }) => {
                FractalHeapHugeObjectId::Direct(self.reference(
                    addr,
                    len,
                    HugeObjectPayload::Unfiltered,
                )?)
            }
            FractalHeapIdKind::Huge(HugeObjectReference::FilteredInline(location)) => {
                let count = self
                    .context
                    .pipeline
                    .as_ref()
                    .ok_or(FormatError::InvalidFractalHeapGeometry)?
                    .filters
                    .len();
                if count < u32::BITS as usize && location.filter_mask() >> count != 0 {
                    return Err(FormatError::InvalidFractalHeapGeometry);
                }
                if location.filter_mask().count_ones() as usize == count
                    && location.encoded_size() != location.decoded_size()
                {
                    return Err(FormatError::InvalidFractalHeapGeometry);
                }
                FractalHeapHugeObjectId::Direct(self.filtered_reference(location)?)
            }
            FractalHeapIdKind::Huge(
                HugeObjectReference::Indexed(key) | HugeObjectReference::FilteredIndexed(key),
            ) => {
                let bits = u32::from(self.context.widths.lengths.get()) * 8;
                if key.value().checked_shr(bits).is_some_and(|high| high != 0) {
                    return Err(FormatError::ValueTooLargeForRepresentation {
                        value: key.value(),
                        target: "huge-object tracker key",
                    });
                }
                FractalHeapHugeObjectId::Indirect(FractalHeapHugeObjectLookup {
                    context: Arc::clone(&self.context),
                    key,
                })
            }
            _ => return Err(FormatError::InvalidFractalHeapGeometry),
        })
    }

    /// Parses the tracker header from its physical region and checks its record encoding.
    ///
    /// The caller traverses the B-tree using the returned header and supplies individual record
    /// bytes to [`FractalHeapHugeObjectTracker::parse_record`]. With the `checksum` feature, the
    /// parser verifies the header checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`tracker_range`](Self::tracker_range), the header errors of
    /// [`BTreeV2Header::parse_with_widths`] for malformed header bytes, or
    /// [`FormatError::UnexpectedHugeObjectBTree`] if the record type or size differs from the
    /// heap's required encoding.
    pub fn parse_tracker(
        &self,
        region: FileRegion<'_>,
    ) -> Result<FractalHeapHugeObjectTracker, FormatError> {
        let header = BTreeV2Header::parse_with_widths(
            region.subregion(self.tracker_range())?.bytes(),
            0,
            self.context.widths,
        )?;
        let decoder = self.context.records.resolve(&header)?;
        Ok(FractalHeapHugeObjectTracker {
            layout: self.clone(),
            header,
            decoder,
        })
    }

    /// Returns the payload reference for a lookup whose numeric key matches `record`.
    ///
    /// The caller finds the candidate record by traversing the tracker. Comparison uses the
    /// complete numeric key, including when the heap ID and tracker have different key widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::WrongLayout`] if either argument belongs to an independently
    /// constructed layout, or [`FormatError::HugeObjectNotFound`] if the record has no numeric
    /// key or its key differs from the lookup's.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_core::BaseAddress;
    /// # use hdf5_pure_core::__private::{BaseAddressExt, FileRegion, StoredAddressExt};
    /// # use hdf5_pure_format::FormatError;
    /// # use hdf5_pure_format::__private::{
    /// #     BTREE_V2_HUGE_OBJECT, FormatWidths, FractalHeapHeader, FractalHeapHugeObjectId,
    /// #     FractalHeapHugeObjectLayout, HugeIndirectKeyEncoding, LengthWidth, OffsetWidth,
    /// # };
    /// # use test_util::{btree_v2, fractal_heap, widths::Widths};
    /// # let encoded = fractal_heap::Header::new(u64::MAX).build(Widths::EIGHT);
    /// # let mut header = FractalHeapHeader::parse_bytes(
    /// #     &encoded, OffsetWidth::Eight, LengthWidth::Eight,
    /// # )?;
    /// # header.heap_id_length = 2;
    /// # header.btree_huge_objects_address = hdf5_pure_core::StoredAddress::new(256);
    /// # let widths = FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight };
    /// # let objects = FractalHeapHugeObjectLayout::new(
    /// #     &header, widths, BaseAddress::ZERO, HugeIndirectKeyEncoding::Libhdf5,
    /// # )?;
    /// # let encoded_tracker = btree_v2::Header::new(BTREE_V2_HUGE_OBJECT, 24, 512, 1)
    /// #     .build(Widths::EIGHT);
    /// # let encoded_record = [0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    /// #     5, 0, 0, 0, 0, 0, 0, 0];
    /// let FractalHeapHugeObjectId::Indirect(lookup) = objects.parse_id(&[0x10, 5])? else {
    ///     unreachable!();
    /// };
    /// let tracker_region = FileRegion::new(lookup.tracker_range(), &encoded_tracker)?;
    /// let tracker = objects.parse_tracker(tracker_region)?;
    /// let record = tracker.parse_record(&encoded_record)?;
    /// let reference = objects.resolve(&lookup, &record)?;
    /// assert_eq!(reference.range().start().get(), 1024);
    /// assert_eq!(reference.range().length(), 3);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn resolve(
        &self,
        lookup: &FractalHeapHugeObjectLookup,
        record: &FractalHeapHugeObjectRecord,
    ) -> Result<FractalHeapHugeObjectRef, FormatError> {
        self.require_context(&lookup.context)?;
        self.require_context(&record.reference.context)?;
        if record.key != Some(lookup.key.value()) {
            return Err(FormatError::HugeObjectNotFound(lookup.key.value()));
        }
        Ok(record.reference.clone())
    }

    /// Borrows decoded bytes after checking their length against the payload reference.
    ///
    /// The caller reverses any active filters before passing `bytes`. The returned view retains
    /// the encoded allocation through an owned reference and borrows the supplied decoded bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::WrongLayout`] if `reference` belongs to an independently
    /// constructed layout, or [`FormatError::DecodedSizeMismatch`] if the byte count differs
    /// from [`FractalHeapHugeObjectRef::decoded_size`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_core::BaseAddress;
    /// # use hdf5_pure_core::__private::{BaseAddressExt, StoredAddressExt};
    /// # use hdf5_pure_format::FormatError;
    /// # use hdf5_pure_format::__private::{
    /// #     FILTER_PIPELINE_VERSION_TWO, FilterPipeline, FormatWidths, FractalHeapFiltering,
    /// #     FractalHeapHeader, FractalHeapHugeObjectId, FractalHeapHugeObjectLayout,
    /// #     HugeIndirectKeyEncoding, LengthWidth, OffsetWidth,
    /// # };
    /// # use test_util::{fractal_heap, widths::Widths};
    /// # let encoded = fractal_heap::Header::new(u64::MAX).build(Widths::EIGHT);
    /// # let mut header = FractalHeapHeader::parse_bytes(
    /// #     &encoded, OffsetWidth::Eight, LengthWidth::Eight,
    /// # )?;
    /// # header.heap_id_length = 29;
    /// # header.btree_huge_objects_address = hdf5_pure_core::StoredAddress::new(256);
    /// # header.filtering = FractalHeapFiltering::Filtered {
    /// #     root_direct_block_size: 0, root_filter_mask: 0,
    /// #     pipeline: FilterPipeline { version: FILTER_PIPELINE_VERSION_TWO, filters: vec![] },
    /// # };
    /// # let widths = FormatWidths { offsets: OffsetWidth::Eight, lengths: LengthWidth::Eight };
    /// # let objects = FractalHeapHugeObjectLayout::new(
    /// #     &header, widths, BaseAddress::ZERO, HugeIndirectKeyEncoding::Libhdf5,
    /// # )?;
    /// # let id = [0x10, 0, 4, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    /// #     0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0];
    /// let FractalHeapHugeObjectId::Direct(reference) = objects.parse_id(&id)? else {
    ///     unreachable!();
    /// };
    /// assert_eq!(reference.decoded_size(), 3);
    /// let view = objects.associate_decoded(&reference, b"abc")?;
    /// assert_eq!(view.bytes(), b"abc");
    /// assert_eq!(view.reference().range(), reference.range());
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn associate_decoded<'bytes>(
        &self,
        reference: &FractalHeapHugeObjectRef,
        bytes: &'bytes [u8],
    ) -> Result<FractalHeapHugeObjectView<'bytes>, FormatError> {
        self.require_context(&reference.context)?;
        if bytes.len() as u64 != reference.decoded_size() {
            return Err(FormatError::DecodedSizeMismatch {
                expected: reference.decoded_size(),
                actual: bytes.len() as u64,
            });
        }
        Ok(FractalHeapHugeObjectView {
            reference: reference.clone(),
            bytes,
        })
    }

    /// Returns the complete physical range required to parse the tracker header.
    pub fn tracker_range(&self) -> FileRange {
        self.context.tracker_range
    }

    fn filtered_reference(
        &self,
        location: FilteredHugeObjectLocation,
    ) -> Result<FractalHeapHugeObjectRef, FormatError> {
        let bits = u32::from(self.context.widths.lengths.get()) * 8;
        if location
            .decoded_size()
            .checked_shr(bits)
            .is_some_and(|high| high != 0)
        {
            return Err(FormatError::ValueTooLargeForRepresentation {
                value: location.decoded_size(),
                target: "huge-object decoded size",
            });
        }
        self.reference(
            location.address(),
            location.encoded_size(),
            HugeObjectPayload::Filtered(location),
        )
    }

    fn reference(
        &self,
        address: StoredAddress,
        length: u64,
        payload: HugeObjectPayload,
    ) -> Result<FractalHeapHugeObjectRef, FormatError> {
        if address.get() >= StoredAddress::undefined(self.context.widths.offsets).get()
            || length == 0
        {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let bits = u32::from(self.context.widths.lengths.get()) * 8;
        if length.checked_shr(bits).is_some_and(|high| high != 0) {
            return Err(FormatError::ValueTooLargeForRepresentation {
                value: length,
                target: "huge-object encoded size",
            });
        }
        Ok(FractalHeapHugeObjectRef {
            context: Arc::clone(&self.context),
            range: FileRange::new(
                PhysicalAddress::new(self.context.base.absolute(address)?),
                length,
            )?,
            payload,
        })
    }

    fn require_context(&self, context: &Arc<HugeObjectContext>) -> Result<(), FormatError> {
        if !Arc::ptr_eq(&self.context, context) {
            return Err(FormatError::WrongLayout);
        }
        Ok(())
    }
}

/// A huge heap ID resolved to a physical payload or an indirect tracker lookup.
#[derive(Clone, Debug)]
pub enum FractalHeapHugeObjectId {
    /// The object's location is stored in the ID.
    Direct(FractalHeapHugeObjectRef),
    /// The caller resolves a numeric key through a tracker record.
    Indirect(FractalHeapHugeObjectLookup),
}

/// A numeric tracker key associated with its owning huge-object layout.
///
/// Obtained from [`FractalHeapHugeObjectLayout::parse_id`] for an indirect huge ID.
#[derive(Clone, Debug)]
pub struct FractalHeapHugeObjectLookup {
    context: Arc<HugeObjectContext>,
    key: HugeIndirectKey,
}

impl FractalHeapHugeObjectLookup {
    /// Returns the numeric key and the dialect used to decode its width.
    pub const fn key(&self) -> HugeIndirectKey {
        self.key
    }

    /// Returns the complete physical range required to parse the tracker header.
    pub fn tracker_range(&self) -> FileRange {
        self.context.tracker_range
    }
}

/// A parsed tracker header with its heap's checked record encoding.
///
/// Obtained from [`FractalHeapHugeObjectLayout::parse_tracker`]. The caller traverses the
/// B-tree and supplies its record bytes to [`parse_record`](Self::parse_record).
#[derive(Clone, Debug)]
pub struct FractalHeapHugeObjectTracker {
    layout: FractalHeapHugeObjectLayout,
    header: BTreeV2Header,
    decoder: HugeObjectRecordDecoder,
}

impl FractalHeapHugeObjectTracker {
    /// Decodes one tracker record and resolves its payload's stored address to a physical range.
    ///
    /// # Errors
    ///
    /// Returns the record errors of [`HugeObjectRecordDecoder::parse`] for invalid encoded
    /// fields or record geometry. Returns [`FormatError::OffsetOverflow`] if the payload's
    /// physical range exceeds [`u64::MAX`].
    pub fn parse_record(&self, bytes: &[u8]) -> Result<FractalHeapHugeObjectRecord, FormatError> {
        let (key, reference) = match self.decoder.parse(bytes)? {
            HugeObjectTrackerRecord::Indirect(record) => (
                Some(record.id),
                self.layout.reference(
                    record.address,
                    record.length,
                    HugeObjectPayload::Unfiltered,
                )?,
            ),
            HugeObjectTrackerRecord::FilteredIndirect(record) => (
                Some(record.id()),
                self.layout.filtered_reference(record.location())?,
            ),
            HugeObjectTrackerRecord::Direct(record) => (
                None,
                self.layout.reference(
                    record.address,
                    record.length,
                    HugeObjectPayload::Unfiltered,
                )?,
            ),
            HugeObjectTrackerRecord::FilteredDirect(record) => {
                (None, self.layout.filtered_reference(record.location())?)
            }
        };
        Ok(FractalHeapHugeObjectRecord { key, reference })
    }

    /// Returns the parsed header for caller-controlled B-tree traversal.
    pub const fn header(&self) -> &BTreeV2Header {
        &self.header
    }
}

/// A parsed tracker record with an owned physical payload reference.
///
/// Obtained from [`FractalHeapHugeObjectTracker::parse_record`]. Indirect records have a
/// numeric lookup key, and direct records have no key.
#[derive(Clone, Debug)]
pub struct FractalHeapHugeObjectRecord {
    key: Option<u64>,
    reference: FractalHeapHugeObjectRef,
}

impl FractalHeapHugeObjectRecord {
    /// Returns the numeric lookup key for an indirect record, or [`None`] for a direct record.
    pub const fn key(&self) -> Option<u64> {
        self.key
    }

    /// Returns the owned reference to the encoded physical allocation.
    pub const fn reference(&self) -> &FractalHeapHugeObjectRef {
        &self.reference
    }
}

/// A huge object's complete encoded allocation and inherited decoding context.
///
/// The physical range uses the encoded length. [`decoded_size`](Self::decoded_size) gives the
/// expected payload length after reversing any active filters. The owned context retains the
/// originating layout's identity and filter pipeline.
#[derive(Clone, Debug)]
pub struct FractalHeapHugeObjectRef {
    context: Arc<HugeObjectContext>,
    range: FileRange,
    payload: HugeObjectPayload,
}

impl FractalHeapHugeObjectRef {
    /// Borrows the encoded payload bytes from a physical region containing the allocation.
    ///
    /// The caller executes any active filters before associating decoded bytes through
    /// [`FractalHeapHugeObjectLayout::associate_decoded`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range).
    pub fn parse<'bytes>(&self, region: FileRegion<'bytes>) -> Result<&'bytes [u8], FormatError> {
        Ok(region.subregion(self.parse_range())?.bytes())
    }

    /// Returns the complete encoded physical allocation.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the required encoded payload range, equal to [`range`](Self::range).
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }

    /// Returns the expected decoded payload length in bytes.
    pub const fn decoded_size(&self) -> u64 {
        match self.payload {
            HugeObjectPayload::Unfiltered => self.range.length(),
            HugeObjectPayload::Filtered(location) => location.decoded_size(),
        }
    }

    /// Returns the inherited pipeline and per-object mask for a filtered heap.
    pub fn filters(&self) -> Option<FractalHeapHugeObjectFilters<'_>> {
        match self.payload {
            HugeObjectPayload::Unfiltered => None,
            HugeObjectPayload::Filtered(location) => {
                self.context
                    .pipeline
                    .as_ref()
                    .map(|pipeline| FractalHeapHugeObjectFilters {
                        pipeline,
                        mask: location.filter_mask(),
                    })
            }
        }
    }
}

/// The inherited filter pipeline and per-object skipped-filter mask for a huge object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapHugeObjectFilters<'a> {
    /// The heap's pipeline in encoding order.
    pub pipeline: &'a FilterPipeline,
    /// Set bits identify filters to skip by their index in the pipeline.
    pub mask: u32,
}

/// Borrowed decoded bytes associated with an owned encoded-object reference.
///
/// Obtained from [`FractalHeapHugeObjectLayout::associate_decoded`] after the caller supplies
/// bytes with the expected decoded length.
#[derive(Clone, Debug)]
pub struct FractalHeapHugeObjectView<'bytes> {
    reference: FractalHeapHugeObjectRef,
    bytes: &'bytes [u8],
}

impl<'bytes> FractalHeapHugeObjectView<'bytes> {
    /// Returns the owned reference to the encoded physical allocation.
    pub const fn reference(&self) -> &FractalHeapHugeObjectRef {
        &self.reference
    }

    /// Returns the caller-supplied decoded payload bytes.
    pub const fn bytes(&self) -> &'bytes [u8] {
        self.bytes
    }
}

/// Shared parameters whose allocation identifies one huge-object layout.
#[derive(Debug)]
struct HugeObjectContext {
    base: BaseAddress,
    widths: FormatWidths,
    ids: FractalHeapIdLayout,
    records: HugeObjectRecordContext,
    tracker_range: FileRange,
    pipeline: Option<FilterPipeline>,
}

#[derive(Clone, Copy, Debug)]
enum HugeObjectPayload {
    Filtered(FilteredHugeObjectLocation),
    Unfiltered,
}
