//! Owned physical references and decoded bytes for fractal-heap huge objects.
//!
//! The caller reads tracker metadata and encoded payloads using the returned physical ranges.
//! Filter execution and B-tree traversal belong to the caller. Decoded payload bytes are borrowed
//! through [`FractalHeapHugeObjectView`], with their encoded allocation retained separately.

use alloc::sync::Arc;

use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, PhysicalAddress};
use hdf5_pure_core::BaseAddress;

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
    /// #     BaseAddressExt, FileRange, FileRegion, PhysicalAddress, StoredAddressExt,
    /// # };
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

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hdf5_pure_core::__private::StoredAddressExt;
    use rstest::rstest;
    use test_util::widths::Widths;

    use crate::address::StoredAddress;
    use crate::filter_pipeline::{FilterDescription, VERSION_TWO};
    use crate::width::{LengthWidth, OffsetWidth};

    use super::*;

    #[rstest]
    fn huge_references_preserve_physical_allocations_and_filter_context(
        #[values(false, true)] filtered: bool,
        #[values(false, true)] direct: bool,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
        #[values(0, 512, u64::from(u32::MAX) + 512)] base: u64,
    ) {
        let fixture = Fixture::new(
            filtered,
            direct,
            base,
            1024,
            FormatWidths { offsets, lengths },
        );
        let reference = fixture.resolve();
        let tracker = fixture.tracker();
        let record = tracker.parse_record(&fixture.record).unwrap();
        assert_eq!(record.key(), if direct { None } else { Some(5) });
        assert_eq!(record.reference().range(), reference.range());
        assert_eq!(record.reference().decoded_size(), reference.decoded_size());
        assert_eq!(record.reference().filters(), reference.filters());
        assert_eq!(
            reference.range(),
            FileRange::new(PhysicalAddress::new(base + 1024), 3).unwrap()
        );
        assert_eq!(reference.parse_range(), reference.range());
        assert_eq!(reference.decoded_size(), if filtered { 9 } else { 3 });
        assert_eq!(fixture.layout.tracker_range().start().get(), base + 256);
        let encoded = reference
            .parse(FileRegion::new(reference.range(), b"abc").unwrap())
            .unwrap();
        assert_eq!(encoded, b"abc");
        assert_eq!(encoded.as_ptr(), b"abc".as_ptr());
        if filtered {
            let filters = reference.filters().unwrap();
            assert_eq!(filters.mask, 0);
            assert_eq!(filters.pipeline, &fixture.pipeline);
        } else {
            assert_eq!(reference.filters(), None);
        }
        let decoded = vec![7; usize::try_from(reference.decoded_size()).unwrap()];
        let view = fixture
            .layout
            .associate_decoded(&reference, &decoded)
            .unwrap();
        assert_eq!(view.bytes(), decoded);
        assert_eq!(view.bytes().as_ptr(), decoded.as_ptr());
        assert_eq!(view.reference().range(), reference.range());
    }

    #[rstest]
    fn huge_references_preserve_selectively_skipped_filters(#[values(false, true)] direct: bool) {
        let mut fixture = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
        let FractalHeapFiltering::Filtered { pipeline, .. } = &mut fixture.header.filtering else {
            panic!("expected filtered fixture")
        };
        let mut second = pipeline.filters[0].clone();
        second.filter_id = 32_002;
        pipeline.filters.push(second);
        let expected = pipeline.clone();
        fixture.layout = FractalHeapHugeObjectLayout::new(
            &fixture.header,
            fixture.widths,
            BaseAddress::ZERO,
            HugeIndirectKeyEncoding::Libhdf5,
        )
        .unwrap();
        let mask = 1u32.to_le_bytes();
        let mask_at =
            usize::from(fixture.widths.offsets.get()) + usize::from(fixture.widths.lengths.get());
        if direct {
            fixture.id[mask_at + 1..mask_at + 1 + mask.len()].copy_from_slice(&mask);
        } else {
            fixture.record[mask_at..mask_at + mask.len()].copy_from_slice(&mask);
        }
        let reference = fixture.resolve();
        assert_eq!(
            reference.filters().unwrap(),
            FractalHeapHugeObjectFilters {
                pipeline: &expected,
                mask: 1
            }
        );
        assert_eq!(reference.decoded_size(), 9);
    }

    #[rstest]
    #[case::short(512, 2)]
    #[case::before(511, 3)]
    #[case::after(513, 3)]
    fn huge_payloads_require_their_encoded_region(#[case] start: u64, #[case] length: usize) {
        let fixture = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
        let reference = fixture.resolve();
        let region = FileRegion::new(
            FileRange::new(PhysicalAddress::new(start), length as u64).unwrap(),
            &b"abc"[..length],
        )
        .unwrap();
        assert_eq!(
            reference.parse(region).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 512,
                required_length: 3,
                provided_start: start,
                provided_length: length as u64,
            }
        );
    }

    #[rstest]
    fn huge_trackers_require_their_complete_header(#[values(0, 1)] missing: usize) {
        let fixture = Fixture::new(true, false, 512, 1024, EIGHT_WIDTHS);
        let range = fixture.layout.tracker_range();
        let length = fixture.tracker.len() - missing;
        let region = FileRegion::new(
            FileRange::new(range.start(), length as u64).unwrap(),
            &fixture.tracker[..length],
        )
        .unwrap();
        if missing == 0 {
            assert_eq!(
                fixture
                    .layout
                    .parse_tracker(region)
                    .unwrap()
                    .header()
                    .total_records,
                1
            );
        } else {
            assert_eq!(
                fixture.layout.parse_tracker(region).unwrap_err(),
                FormatError::InsufficientRegion {
                    required_start: range.start().get(),
                    required_length: range.length(),
                    provided_start: range.start().get(),
                    provided_length: length as u64,
                }
            );
        }
    }

    #[rstest]
    #[case::short(8)]
    #[case::long(10)]
    fn huge_decoded_payloads_require_the_declared_size(#[case] length: usize) {
        let fixture = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
        assert_eq!(
            fixture
                .layout
                .associate_decoded(&fixture.resolve(), &vec![0; length])
                .unwrap_err(),
            FormatError::DecodedSizeMismatch {
                expected: 9,
                actual: length as u64
            }
        );
    }

    #[rstest]
    fn huge_references_reject_independent_layouts(#[values(false, true)] direct: bool) {
        let fixture = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
        let other = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
        let reference = fixture.resolve();
        assert_eq!(
            other
                .layout
                .associate_decoded(&reference, &[0; 9])
                .unwrap_err(),
            FormatError::WrongLayout
        );
        if !direct {
            let FractalHeapHugeObjectId::Indirect(lookup) =
                fixture.layout.parse_id(&fixture.id).unwrap()
            else {
                panic!("expected indirect ID")
            };
            let record = fixture.tracker().parse_record(&fixture.record).unwrap();
            let other_record = other.tracker().parse_record(&other.record).unwrap();
            assert_eq!(
                other.layout.resolve(&lookup, &other_record).unwrap_err(),
                FormatError::WrongLayout
            );
            assert_eq!(
                fixture.layout.resolve(&lookup, &other_record).unwrap_err(),
                FormatError::WrongLayout
            );
            assert_eq!(
                fixture.layout.resolve(&lookup, &record).unwrap().range(),
                reference.range()
            );
        }
    }

    #[test]
    fn huge_layout_clones_preserve_identity_after_the_original_is_dropped() {
        let fixture = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
        let reference = fixture.resolve();
        let layout = fixture.layout.clone();
        let pipeline = fixture.pipeline.clone();
        drop(fixture);
        assert_eq!(reference.filters().unwrap().pipeline, &pipeline);
        let view = layout.associate_decoded(&reference, &[0; 9]).unwrap();
        drop(layout);
        drop(reference);
        assert_eq!(view.bytes(), &[0; 9]);
        assert_eq!(view.reference().filters().unwrap().pipeline, &pipeline);
    }

    #[test]
    fn huge_lookup_compares_the_complete_numeric_key() {
        let mut fixture = Fixture::new(
            true,
            false,
            0,
            512,
            FormatWidths {
                offsets: OffsetWidth::Eight,
                lengths: LengthWidth::Two,
            },
        );
        fixture.header.heap_id_length = 8;
        fixture.layout = FractalHeapHugeObjectLayout::new(
            &fixture.header,
            fixture.widths,
            BaseAddress::ZERO,
            HugeIndirectKeyEncoding::Libhdf5,
        )
        .unwrap();
        fixture.id = vec![HUGE_ID_FIRST_BYTE, 5, 1, 0, 0, 0, 0, 0];
        let FractalHeapHugeObjectId::Indirect(lookup) =
            fixture.layout.parse_id(&fixture.id).unwrap()
        else {
            panic!("expected indirect ID")
        };
        let record = fixture.tracker().parse_record(&fixture.record).unwrap();
        assert_eq!(lookup.key().encoding(), HugeIndirectKeyEncoding::Libhdf5);
        assert_eq!(
            fixture.layout.resolve(&lookup, &record).unwrap_err(),
            FormatError::HugeObjectNotFound(261)
        );
        fixture.record.truncate(fixture.record.len() - 2);
        fixture.record.extend_from_slice(&261u16.to_le_bytes());
        let record = fixture.tracker().parse_record(&fixture.record).unwrap();
        assert_eq!(record.key(), Some(261));
        assert_eq!(
            fixture
                .layout
                .resolve(&lookup, &record)
                .unwrap()
                .range()
                .start()
                .get(),
            512
        );
    }

    #[rstest]
    #[case::address(u64::MAX - 512, 1024, FormatError::OffsetOverflow { offset: 1024, length: u64::MAX - 512 })]
    #[case::allocation(u64::MAX - 512, 510, FormatError::OffsetOverflow { offset: u64::MAX - 2, length: 3 })]
    fn huge_references_reject_physical_range_overflow(
        #[case] base: u64,
        #[case] address: u64,
        #[case] expected: FormatError,
    ) {
        let mut fixture = Fixture::new(true, true, 0, address, EIGHT_WIDTHS);
        fixture.header.btree_huge_objects_address = StoredAddress::new(0);
        fixture.layout = FractalHeapHugeObjectLayout::new(
            &fixture.header,
            fixture.widths,
            BaseAddress::new(base),
            HugeIndirectKeyEncoding::Libhdf5,
        )
        .unwrap();
        assert_eq!(fixture.layout.parse_id(&fixture.id).unwrap_err(), expected);
    }

    #[rstest]
    fn huge_layouts_require_a_defined_tracker(#[values(false, true)] direct: bool) {
        let mut fixture = Fixture::new(true, direct, 0, 512, EIGHT_WIDTHS);
        fixture.header.btree_huge_objects_address = StoredAddress::undefined(OffsetWidth::Eight);
        assert_eq!(
            FractalHeapHugeObjectLayout::new(
                &fixture.header,
                fixture.widths,
                BaseAddress::ZERO,
                HugeIndirectKeyEncoding::Libhdf5
            )
            .unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }

    #[rstest]
    fn parsed_huge_ids_require_the_receiving_representation(
        #[values(false, true)] source_filtered: bool,
        #[values(false, true)] source_direct: bool,
        #[values(false, true)] target_filtered: bool,
        #[values(false, true)] target_direct: bool,
    ) {
        let source = Fixture::new(source_filtered, source_direct, 0, 512, EIGHT_WIDTHS);
        let target = Fixture::new(target_filtered, target_direct, 0, 512, EIGHT_WIDTHS);
        let id = source.ids().parse(&source.id).unwrap();
        let result = target.layout.resolve_id(id);
        if source_filtered == target_filtered && source_direct == target_direct {
            let resolved = result.unwrap();
            match resolved {
                FractalHeapHugeObjectId::Direct(reference) => {
                    assert_eq!(reference.range().start().get(), 512)
                }
                FractalHeapHugeObjectId::Indirect(lookup) => assert_eq!(lookup.key().value(), 5),
            }
        } else {
            assert_eq!(result.unwrap_err(), FormatError::InvalidFractalHeapGeometry);
        }
    }

    #[rstest]
    #[case::encoded(65_536, 9, FormatError::ValueTooLargeForRepresentation { value: 65_536, target: "huge-object encoded size" })]
    #[case::decoded(3, 65_536, FormatError::ValueTooLargeForRepresentation { value: 65_536, target: "huge-object decoded size" })]
    fn parsed_huge_ids_recheck_length_widths(
        #[case] encoded: u64,
        #[case] decoded: u64,
        #[case] expected: FormatError,
    ) {
        let mut source = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
        source.id[9..17].copy_from_slice(&encoded.to_le_bytes());
        source.id[21..29].copy_from_slice(&decoded.to_le_bytes());
        let target = Fixture::new(
            true,
            true,
            0,
            512,
            FormatWidths {
                offsets: OffsetWidth::Eight,
                lengths: LengthWidth::Two,
            },
        );
        let id = source.ids().parse(&source.id).unwrap();
        assert_eq!(target.layout.resolve_id(id).unwrap_err(), expected);
    }

    #[rstest]
    #[case::mask(2)]
    #[case::skipped_sizes(1)]
    fn parsed_huge_ids_recheck_the_receiving_pipeline(#[case] mask: u32) {
        let mut source = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
        let FractalHeapFiltering::Filtered { pipeline, .. } = &mut source.header.filtering else {
            panic!("expected filtered fixture")
        };
        pipeline.filters.push(pipeline.filters[0].clone());
        source.layout = FractalHeapHugeObjectLayout::new(
            &source.header,
            EIGHT_WIDTHS,
            BaseAddress::ZERO,
            HugeIndirectKeyEncoding::Libhdf5,
        )
        .unwrap();
        source.id[17..21].copy_from_slice(&mask.to_le_bytes());
        let target = Fixture::new(true, true, 0, 512, EIGHT_WIDTHS);
        let id = source.ids().parse(&source.id).unwrap();
        assert_eq!(
            target.layout.resolve_id(id).unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }

    #[test]
    fn parsed_huge_ids_recheck_address_widths() {
        let source = Fixture::new(true, true, 0, u64::from(u16::MAX), EIGHT_WIDTHS);
        let target = Fixture::new(
            true,
            true,
            0,
            512,
            FormatWidths {
                offsets: OffsetWidth::Two,
                lengths: LengthWidth::Eight,
            },
        );
        let id = source.ids().parse(&source.id).unwrap();
        assert_eq!(
            target.layout.resolve_id(id).unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }

    #[rstest]
    fn parsed_huge_ids_recheck_numeric_key_domains(
        #[values(
            HugeIndirectKeyEncoding::Libhdf5,
            HugeIndirectKeyEncoding::PublishedSpec
        )]
        encoding: HugeIndirectKeyEncoding,
    ) {
        let mut source = Fixture::new(true, false, 0, 512, EIGHT_WIDTHS);
        source.header.heap_id_length = 9;
        let mut id = vec![HUGE_ID_FIRST_BYTE];
        id.extend_from_slice(&65_536u64.to_le_bytes());
        let ids = FractalHeapIdLayout::with_huge_key_encoding(
            &source.header,
            OffsetWidth::Eight,
            LengthWidth::Eight,
            encoding,
        );
        let target = Fixture::new(
            true,
            false,
            0,
            512,
            FormatWidths {
                offsets: OffsetWidth::Eight,
                lengths: LengthWidth::Two,
            },
        );
        assert_eq!(
            target
                .layout
                .resolve_id(ids.parse(&id).unwrap())
                .unwrap_err(),
            FormatError::ValueTooLargeForRepresentation {
                value: 65_536,
                target: "huge-object tracker key"
            }
        );
    }

    struct Fixture {
        header: FractalHeapHeader,
        widths: FormatWidths,
        pipeline: FilterPipeline,
        layout: FractalHeapHugeObjectLayout,
        id: Vec<u8>,
        tracker: Vec<u8>,
        record: Vec<u8>,
    }

    impl Fixture {
        fn new(
            filtered: bool,
            direct: bool,
            base: u64,
            address: u64,
            widths: FormatWidths,
        ) -> Self {
            let mut header = FractalHeapHeader::parse_bytes(
                &test_util::fractal_heap::Header::new(u64::MAX).build(Widths::EIGHT),
                OffsetWidth::Eight,
                LengthWidth::Eight,
            )
            .unwrap();
            header.root_block_address = StoredAddress::undefined(widths.offsets);
            header.btree_huge_objects_address = StoredAddress::new(256);
            header.heap_id_length = if direct { 64 } else { 2 };
            let pipeline = FilterPipeline {
                version: VERSION_TWO,
                filters: vec![FilterDescription {
                    filter_id: 32_001,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                }],
            };
            header.filtering = if filtered {
                FractalHeapFiltering::Filtered {
                    root_direct_block_size: 0,
                    root_filter_mask: 0,
                    pipeline: pipeline.clone(),
                }
            } else {
                FractalHeapFiltering::Unfiltered
            };
            let layout = FractalHeapHugeObjectLayout::new(
                &header,
                widths,
                BaseAddress::new(base),
                HugeIndirectKeyEncoding::Libhdf5,
            )
            .unwrap();
            let mut record = Vec::new();
            record.extend_from_slice(&address.to_le_bytes()[..usize::from(widths.offsets.get())]);
            record.extend_from_slice(&3u64.to_le_bytes()[..usize::from(widths.lengths.get())]);
            if filtered {
                record.extend_from_slice(&0u32.to_le_bytes());
                record.extend_from_slice(&9u64.to_le_bytes()[..usize::from(widths.lengths.get())]);
            }
            let tree_type = match (filtered, direct) {
                (false, false) => crate::btree_v2::BTREE_V2_HUGE_OBJECT,
                (true, false) => crate::btree_v2::BTREE_V2_FILTERED_HUGE_OBJECT,
                (false, true) => crate::btree_v2::BTREE_V2_HUGE_OBJECT_DIRECT,
                (true, true) => crate::btree_v2::BTREE_V2_FILTERED_HUGE_OBJECT_DIRECT,
            };
            let mut id = vec![HUGE_ID_FIRST_BYTE];
            if direct {
                id.extend_from_slice(&record);
            } else {
                record.extend_from_slice(&5u64.to_le_bytes()[..usize::from(widths.lengths.get())]);
                id.push(5);
            }
            id.resize(usize::from(header.heap_id_length), 0);
            let tracker = test_util::btree_v2::Header::new(
                tree_type,
                u16::try_from(record.len()).unwrap(),
                512,
                1,
            )
            .build(Widths::new(
                usize::from(widths.offsets.get()),
                usize::from(widths.lengths.get()),
            ));
            Self {
                header,
                widths,
                pipeline,
                layout,
                id,
                tracker,
                record,
            }
        }

        fn ids(&self) -> FractalHeapIdLayout {
            FractalHeapIdLayout::new(&self.header, self.widths.offsets, self.widths.lengths)
        }

        fn tracker(&self) -> FractalHeapHugeObjectTracker {
            self.layout
                .parse_tracker(FileRegion::new(self.layout.tracker_range(), &self.tracker).unwrap())
                .unwrap()
        }

        fn resolve(&self) -> FractalHeapHugeObjectRef {
            match self.layout.parse_id(&self.id).unwrap() {
                FractalHeapHugeObjectId::Direct(reference) => reference,
                FractalHeapHugeObjectId::Indirect(lookup) => {
                    let record = self.tracker().parse_record(&self.record).unwrap();
                    self.layout.resolve(&lookup, &record).unwrap()
                }
            }
        }
    }

    const EIGHT_WIDTHS: FormatWidths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };

    /// The version-0 huge-ID type byte from "Fractal Heap" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    const HUGE_ID_FIRST_BYTE: u8 = 0x10;
}
