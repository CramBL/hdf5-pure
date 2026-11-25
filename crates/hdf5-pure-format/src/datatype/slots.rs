//! The positions of object references and variable-length descriptors in stored elements.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::Range;

use crate::convert::Narrow;
use crate::datatype::{self, Datatype, ReferenceType, VariableLengthType};
use crate::error::FormatError;
use crate::variable_length_descriptor::VariableLengthDescriptor;
use crate::width::OffsetWidth;

/// Prepares to read the eight-byte object references in elements of `datatype`.
///
/// The preparation uses each array's base type, stride, and repetition count to represent its
/// slots. The caller supplies element bytes to [`EmbeddedReferenceSlots::stored_references`].
///
/// Returns `None` for unsupported references, such as dataset region references or object
/// references inside a variable-length sequence, or if the element size or slot geometry exceeds
/// the bounds of the datatype or the platform. A successful plan with zero slots is empty.
pub fn embedded_reference_slots(datatype: &Datatype) -> Option<EmbeddedReferenceSlots> {
    SlotPlan::prepare(datatype, OffsetWidth::Eight, SlotKind::Object)
        .ok()
        .flatten()
        .map(|plan| EmbeddedReferenceSlots { plan })
}

/// The size of a stored element and the positions of its eight-byte object references.
#[derive(Debug)]
pub struct EmbeddedReferenceSlots {
    plan: SlotPlan,
}

impl EmbeddedReferenceSlots {
    /// Returns the size in bytes of a complete element.
    pub const fn element_size(&self) -> usize {
        self.plan.element_size
    }

    /// Returns the number of reference slots per element.
    pub fn len(&self) -> usize {
        self.plan.len()
    }

    /// Returns `true` if the datatype has zero reference slots.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Yields each reference's byte offset from the start of `raw` and its stored address.
    ///
    /// Pass bytes encoded with the datatype used to prepare the slots. For each complete
    /// element, the iterator visits compound members in declaration order and array entries in
    /// element order. It skips a trailing partial element.
    ///
    /// # Errors
    ///
    /// Yields [`FormatError::Internal`] if an address field extends past its element.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{Datatype, FormatError, ReferenceType};
    /// use hdf5_pure_format::__private::embedded_reference_slots;
    ///
    /// let datatype = Datatype::Reference { size: 8, ref_type: ReferenceType::Object };
    /// let slots = embedded_reference_slots(&datatype).unwrap();
    /// let raw = 0x1000u64.to_le_bytes();
    /// let references = slots.stored_references(&raw).collect::<Result<Vec<_>, _>>()?;
    /// assert_eq!(references, [(0, 0x1000)]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn stored_references<'a>(
        &'a self,
        raw: &'a [u8],
    ) -> impl Iterator<Item = Result<(usize, u64), FormatError>> + 'a {
        raw.chunks_exact(self.plan.element_size.max(1))
            .enumerate()
            .flat_map(move |(index, element)| {
                self.plan.slots_in(element).map(move |(offset, _)| {
                    let field = element
                        .get(offset..)
                        .and_then(<[u8]>::first_chunk::<8>)
                        .ok_or_else(reference_slot_outside_element)?;
                    Ok((
                        index * self.plan.element_size + offset,
                        u64::from_le_bytes(*field),
                    ))
                })
            })
    }
}

#[cold]
fn reference_slot_outside_element() -> FormatError {
    FormatError::Internal("an object reference slot extends past its element".into())
}

/// Prepares to read variable-length descriptors sized for `offset_width`.
///
/// Returns `Ok(None)` if the descriptor positions or count exceed the bounds of the datatype or
/// the platform. A variable-length sequence has a slot for the sequence's own descriptor. Read its
/// heap payload separately to inspect its base elements.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if an array size exceeds the four-byte datatype size
/// field, or [`FormatError::ValueTooLargeForPlatform`] if a stored size exceeds [`usize::MAX`].
pub fn embedded_vlen_slots(
    datatype: &Datatype,
    offset_width: OffsetWidth,
) -> Result<Option<EmbeddedVlenSlots>, FormatError> {
    SlotPlan::prepare(datatype, offset_width, SlotKind::VariableLength)
        .map(|plan| plan.map(|plan| EmbeddedVlenSlots { plan, offset_width }))
}

/// The positions of variable-length descriptors in an element stored at a given address width.
#[derive(Debug)]
pub struct EmbeddedVlenSlots {
    plan: SlotPlan,
    offset_width: OffsetWidth,
}

impl EmbeddedVlenSlots {
    /// Returns the file address width used to size the stored descriptors.
    pub const fn offset_width(&self) -> OffsetWidth {
        self.offset_width
    }

    /// Returns the size in bytes of a complete element at the selected address width.
    pub const fn element_size(&self) -> usize {
        self.plan.element_size
    }

    /// Returns the number of descriptors per element.
    pub fn len(&self) -> usize {
        self.plan.len()
    }

    /// Returns `true` if the datatype has zero descriptors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Yields each descriptor's byte range and the byte width of a value in its heap payload.
    ///
    /// For strings the width is one. For sequences it is the stored size of the base type,
    /// with a minimum of one. The iterator is empty if `element` is shorter than
    /// [`Self::element_size`]. It visits compound members in declaration order and array entries
    /// in element order.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::__private::embedded_vlen_slots;
    /// use hdf5_pure_format::{
    ///     CharacterSet, Datatype, FormatError, OffsetWidth, StringPadding, VariableLengthType,
    /// };
    ///
    /// let string = Datatype::VariableLength(VariableLengthType::string(
    ///     StringPadding::NullTerminate,
    ///     CharacterSet::Utf8,
    /// ));
    /// let slots = embedded_vlen_slots(&string, OffsetWidth::Four)?.unwrap();
    /// let mut ranges = slots.slots_in(&[0; 12]);
    /// assert_eq!(ranges.next(), Some((0..12, 1)));
    /// assert_eq!(ranges.next(), None);
    /// assert_eq!(slots.slots_in(&[0; 11]).next(), None);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn slots_in(&self, element: &[u8]) -> impl Iterator<Item = (Range<usize>, usize)> + '_ {
        let size = VariableLengthDescriptor::serialized_size(self.offset_width);
        self.plan
            .slots_in(element)
            .map(move |(offset, multiplier)| (offset..offset + size, multiplier))
    }
}

#[derive(Debug)]
struct SlotPlan {
    element_size: usize,
    slots: Option<SlotGroup>,
}

impl SlotPlan {
    fn prepare(
        datatype: &Datatype,
        offset_width: OffsetWidth,
        kind: SlotKind,
    ) -> Result<Option<Self>, FormatError> {
        let Some(geometry) = DatatypeSlots::prepare(datatype, offset_width, kind)? else {
            return Ok(None);
        };
        let element_size: usize = geometry.size.narrow()?;
        if geometry
            .slots
            .as_ref()
            .is_some_and(|group| group.count > element_size / kind.field_width(offset_width))
        {
            return Ok(None);
        }
        Ok(Some(Self {
            element_size,
            slots: geometry.slots,
        }))
    }

    fn len(&self) -> usize {
        self.slots.as_ref().map_or(0, |group| group.count)
    }

    fn slots_in(&self, element: &[u8]) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.slots
            .as_ref()
            .filter(|_| element.len() >= self.element_size)
            .into_iter()
            .flat_map(|group| (0..group.count).map(|index| group.slot_at(index)))
    }
}

#[derive(Clone, Copy)]
enum SlotKind {
    Object,
    VariableLength,
}

impl SlotKind {
    fn field_width(self, offset_width: OffsetWidth) -> usize {
        match self {
            Self::Object => 8,
            Self::VariableLength => VariableLengthDescriptor::serialized_size(offset_width),
        }
    }
}

struct DatatypeSlots {
    size: u32,
    slots: Option<SlotGroup>,
}

impl DatatypeSlots {
    fn prepare(
        datatype: &Datatype,
        offset_width: OffsetWidth,
        kind: SlotKind,
    ) -> Result<Option<Self>, FormatError> {
        let (size, slots) = match datatype {
            Datatype::Reference {
                ref_type: ReferenceType::Object,
                size: 8,
            } if matches!(kind, SlotKind::Object) => (8, Some(SlotGroup::field(8, 1))),
            Datatype::VariableLength(variable) if matches!(kind, SlotKind::VariableLength) => {
                let size = datatype::stored_type_size(datatype, offset_width)?;
                let multiplier = match variable {
                    VariableLengthType::String(_) => 1,
                    VariableLengthType::Sequence { element_type } => {
                        let size: usize =
                            datatype::stored_type_size(element_type, offset_width)?.narrow()?;
                        size.max(1)
                    }
                };
                (size, Some(SlotGroup::field(size.narrow()?, multiplier)))
            }
            Datatype::Compound { size, members } => {
                let mut children = Vec::new();
                let mut count = 0usize;
                let mut end = 0;
                let extent: usize = (*size).narrow()?;
                for member in members {
                    let Ok(offset) = usize::try_from(member.byte_offset) else {
                        return Ok(None);
                    };
                    let Some(geometry) = Self::prepare(&member.datatype, offset_width, kind)?
                    else {
                        return Ok(None);
                    };
                    let Some(group) = geometry.slots else {
                        continue;
                    };
                    let Some(member_end) = offset.checked_add(group.end) else {
                        return Ok(None);
                    };
                    if member_end > extent {
                        return Ok(None);
                    }
                    let Some(next_count) = count.checked_add(group.count) else {
                        return Ok(None);
                    };
                    count = next_count;
                    end = end.max(member_end);
                    children.push(SlotMember {
                        offset,
                        end_index: count,
                        group,
                    });
                }
                (
                    *size,
                    (!children.is_empty()).then_some(SlotGroup {
                        count,
                        end,
                        placement: Placement::Members(children),
                    }),
                )
            }
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                let Some(base) = Self::prepare(base_type, offset_width, kind)? else {
                    return Ok(None);
                };
                if base.slots.is_none() && matches!(kind, SlotKind::Object) {
                    let size = dimensions
                        .iter()
                        .fold(base.size, |size, &extent| size.saturating_mul(extent));
                    return Ok(Some(Self { size, slots: None }));
                }
                let size = datatype::array_type_size(base.size, dimensions)?;
                let slots = match base.slots {
                    Some(group) if size != 0 => {
                        let stride: usize = base.size.narrow()?;
                        let size: usize = size.narrow()?;
                        debug_assert!(stride > 0 && group.end <= stride);
                        let repetitions = size / stride;
                        let Some(count) = group.count.checked_mul(repetitions) else {
                            return Ok(None);
                        };
                        Some(SlotGroup {
                            count,
                            end: size - stride + group.end,
                            placement: Placement::Repeat {
                                stride,
                                group: Box::new(group),
                            },
                        })
                    }
                    _ => None,
                };
                (size, slots)
            }
            _ => {
                if matches!(kind, SlotKind::Object)
                    && datatype::datatype_holds_object_address(datatype)
                {
                    return Ok(None);
                }
                (datatype.type_size(), None)
            }
        };
        Ok(Some(Self { size, slots }))
    }
}

#[derive(Debug)]
struct SlotGroup {
    count: usize,
    end: usize,
    placement: Placement,
}

impl SlotGroup {
    fn field(width: usize, multiplier: usize) -> Self {
        Self {
            count: 1,
            end: width,
            placement: Placement::Field { multiplier },
        }
    }

    /// Returns the byte offset from the start of the group and the heap payload's element width.
    ///
    /// The second value is one for an object reference. Pass an `index` less than `self.count`.
    fn slot_at(&self, index: usize) -> (usize, usize) {
        debug_assert!(index < self.count);
        match &self.placement {
            Placement::Field { multiplier } => (0, *multiplier),
            Placement::Members(members) => {
                let member = &members[members.partition_point(|member| member.end_index <= index)];
                let (offset, multiplier) = member
                    .group
                    .slot_at(index - (member.end_index - member.group.count));
                (member.offset + offset, multiplier)
            }
            Placement::Repeat { stride, group } => {
                let (offset, multiplier) = group.slot_at(index % group.count);
                (index / group.count * stride + offset, multiplier)
            }
        }
    }
}

#[derive(Debug)]
enum Placement {
    Field {
        multiplier: usize,
    },
    Members(Vec<SlotMember>),
    Repeat {
        stride: usize,
        group: Box<SlotGroup>,
    },
}

#[derive(Debug)]
struct SlotMember {
    offset: usize,
    end_index: usize,
    group: SlotGroup,
}
