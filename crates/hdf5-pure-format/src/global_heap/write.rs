//! Preparation and encoding of version 1 global heap collections.

use alloc::vec::Vec;
use core::num::NonZeroU16;

use super::ALIGNMENT;
use super::GCOL_RESERVED_LEN;
use super::GCOL_SIGNATURE;
use super::GCOL_VERSION;
use super::GLOBAL_HEAP_MAX_OBJECTS;
use super::GlobalHeapCollectionError;
use super::IndexSet;
use super::MIN_COLLECTION_SIZE;
use super::OBJECT_RESERVED_LEN;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
use crate::width::LengthWidth;

/// A nonzero member index, reference count, and borrowed payload for a version 1 collection.
///
/// The index identifies the member within a collection. [`PreparedGlobalHeapCollection::new`]
/// checks that the supplied indices are unique. The encoder preserves the supplied count and
/// payload, including a zero count or an empty payload.
///
/// # Reference counts
///
/// "Global Heap" of the [format specification, version 4.0][spec] describes positive counts for
/// referenced objects and deletion when a count reaches zero. `H5HG_insert` initializes counts
/// to zero, and `H5HG_link` requires an explicit `H5HG_remove` call to delete an object
/// (`H5HG.c`, HDF5 2.0.0). The encoder retains counts as stored metadata. The caller establishes
/// whether a member is referenced or eligible for removal independently of its count.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalHeapMember<'a> {
    index: NonZeroU16,
    reference_count: u16,
    payload: &'a [u8],
}

impl<'a> GlobalHeapMember<'a> {
    /// Creates a member with the supplied stored fields and borrowed payload.
    pub const fn new(index: NonZeroU16, reference_count: u16, payload: &'a [u8]) -> Self {
        Self {
            index,
            reference_count,
            payload,
        }
    }

    /// Returns the member's nonzero collection index.
    pub const fn index(&self) -> NonZeroU16 {
        self.index
    }

    /// Returns the stored reference count without inferring whether the member is referenced.
    pub const fn reference_count(&self) -> u16 {
        self.reference_count
    }

    /// Returns the borrowed payload without its alignment padding.
    pub const fn payload(&self) -> &'a [u8] {
        self.payload
    }
}

/// A version 1 collection prepared for encoding at a checked capacity and length width.
///
/// Preparation checks unique indices, complete member containment, and the capacity's minimum,
/// alignment, length-field bound, and host-size bound. Members are borrowed in their supplied
/// physical order. The capacity includes the collection header, member headers, padded payloads,
/// and terminal free space.
///
/// The encoder writes an index-zero free-space header when the remaining extent can hold it,
/// including that header in the stored free-space size. A shorter terminal span has an implied
/// header, and a full collection has no free-space record. Padding and terminal free bytes are
/// zeroed. The collection is defined in "Global Heap" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// The encoder preserves sparse indices, supplied order, and counts:
///
/// ```
/// use core::num::NonZeroU16;
/// use hdf5_pure_format::__private::{GlobalHeapMember, PreparedGlobalHeapCollection};
/// use hdf5_pure_format::{FormatError, LengthWidth};
///
/// let members = [
///     GlobalHeapMember::new(NonZeroU16::new(400).unwrap(), 0, b"alpha"),
///     GlobalHeapMember::new(NonZeroU16::new(7).unwrap(), 9, b""),
/// ];
/// let prepared = PreparedGlobalHeapCollection::new(LengthWidth::Two, 4096, &members)?;
/// assert_eq!(prepared.member_count(), 2);
/// assert_eq!(prepared.length_width(), LengthWidth::Two);
/// assert_eq!(prepared.encoded_size(), 4096);
/// let bytes = prepared.serialize()?;
/// assert_eq!(bytes.len(), 4096);
/// assert_eq!(&bytes[16..20], &[0x90, 1, 0, 0]);
/// assert_eq!(&bytes[32..37], b"alpha");
/// assert_eq!(&bytes[40..44], &[7, 0, 9, 0]);
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Debug)]
pub struct PreparedGlobalHeapCollection<'a> {
    lengths: LengthWidth,
    capacity: usize,
    header_size: usize,
    members: Members<'a>,
    free: FreeTail,
}

impl<'a> PreparedGlobalHeapCollection<'a> {
    /// Prepares members in supplied order within an exact `capacity` in bytes.
    ///
    /// `capacity` must be at least 4096, a multiple of eight, and representable both as a length
    /// field of `lengths` bytes and as [`usize`]. All member headers and padded payloads must fit
    /// after the collection header. Preparation checks the members without allocating output.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DuplicateGlobalHeapMemberIndex`] for a repeated index,
    /// [`FormatError::ValueTooLargeForRepresentation`] for more than [`u16::MAX`] members,
    /// [`FormatError::GlobalHeapCapacityTooSmall`] for a capacity below the minimum,
    /// [`FormatError::UnalignedGlobalHeapCapacity`] for a capacity not divisible by eight,
    /// [`FormatError::LengthTooLarge`] for a capacity outside `lengths`, or
    /// [`FormatError::GlobalHeapCapacityExceeded`] if the members require more space.
    /// Returns [`FormatError::OffsetOverflow`] for size arithmetic overflow, or
    /// [`FormatError::ValueTooLargeForPlatform`] if the capacity does not fit [`usize`].
    ///
    /// # Examples
    ///
    /// Duplicate indices are rejected even when their counts are zero:
    ///
    /// ```
    /// use core::num::NonZeroU16;
    /// use hdf5_pure_format::__private::{GlobalHeapMember, PreparedGlobalHeapCollection};
    /// use hdf5_pure_format::{FormatError, LengthWidth};
    ///
    /// let member = GlobalHeapMember::new(NonZeroU16::MIN, 0, b"alpha");
    /// assert_eq!(
    ///     PreparedGlobalHeapCollection::new(LengthWidth::Eight, 4096, &[member, member])
    ///         .unwrap_err(),
    ///     FormatError::DuplicateGlobalHeapMemberIndex { index: 1 },
    /// );
    /// ```
    pub fn new(
        lengths: LengthWidth,
        capacity: u64,
        members: &'a [GlobalHeapMember<'a>],
    ) -> Result<Self, FormatError> {
        Self::prepare(
            lengths,
            Members::supplied(members)?,
            Capacity::Exact(capacity),
        )
    }

    /// Prepares consecutive indices, count one, and capacity for an explicit free-space header.
    ///
    /// Borrows the payload slice directly. The caller's convenience API does not require a
    /// temporary array of [`GlobalHeapMember`] values or a uniqueness check.
    ///
    /// # Errors
    ///
    /// Returns [`GlobalHeapCollectionError::TooManyObjects`] for an excessive member count and
    /// [`GlobalHeapCollectionError::TooLarge`] if the derived capacity exceeds `lengths`.
    /// Returns [`GlobalHeapCollectionError::Format`] for arithmetic or host-size overflow.
    pub(super) fn from_objects(
        lengths: LengthWidth,
        objects: &'a [&'a [u8]],
    ) -> Result<Self, GlobalHeapCollectionError> {
        if objects.len() > GLOBAL_HEAP_MAX_OBJECTS {
            return Err(GlobalHeapCollectionError::TooManyObjects {
                count: objects.len(),
            });
        }
        Self::prepare(lengths, Members::Consecutive(objects), Capacity::Derived).map_err(|error| {
            match error {
                FormatError::LengthTooLarge { length, .. } => GlobalHeapCollectionError::TooLarge {
                    size: length,
                    length_width: lengths,
                },
                error => GlobalHeapCollectionError::Format(error),
            }
        })
    }

    /// Encodes the complete collection into one owned buffer.
    ///
    /// Writes the supplied indices, counts, and payloads in physical order at the prepared length
    /// width. The returned buffer has exactly [`Self::encoded_size`] bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AllocationFailed`] if reserving the output buffer fails.
    /// Returns [`FormatError::Internal`] if the encoded members differ from the prepared extent.
    ///
    /// # Examples
    ///
    /// A 4096-byte allocation can end immediately after a member, in an implied free span, or
    /// after an explicit free-space header:
    ///
    /// ```
    /// use core::num::NonZeroU16;
    /// use hdf5_pure_format::__private::{GlobalHeapMember, PreparedGlobalHeapCollection};
    /// use hdf5_pure_format::{FormatError, LengthWidth};
    ///
    /// for payload_size in [4064, 4056, 4048] {
    ///     let payload = vec![0xAB; payload_size];
    ///     let members = [GlobalHeapMember::new(NonZeroU16::MIN, 0, &payload)];
    ///     let bytes = PreparedGlobalHeapCollection::new(LengthWidth::Eight, 4096, &members)?
    ///         .serialize()?;
    ///     let end = 32 + payload_size;
    ///     assert_eq!(bytes.len(), 4096);
    ///     assert_eq!(&bytes[32..end], payload);
    ///     if payload_size == 4048 {
    ///         assert_eq!(&bytes[end..end + 8], &[0; 8]);
    ///         assert_eq!(&bytes[end + 8..], &16u64.to_le_bytes());
    ///     } else {
    ///         assert!(bytes[end..].iter().all(|byte| *byte == 0));
    ///     }
    /// }
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn serialize(&self) -> Result<Vec<u8>, FormatError> {
        let Self {
            lengths,
            capacity,
            header_size,
            members,
            free,
        } = self;
        let mut output = Vec::new();
        output
            .try_reserve_exact(*capacity)
            .map_err(|source| FormatError::AllocationFailed {
                values: *capacity,
                source,
            })?;
        output.extend_from_slice(&GCOL_SIGNATURE);
        output.push(GCOL_VERSION);
        output.extend_from_slice(&[0; GCOL_RESERVED_LEN]);
        bytes::write_length(&mut output, capacity.to_u64(), *lengths);
        output.resize(*header_size, 0);
        for GlobalHeapMember {
            index,
            reference_count,
            payload,
        } in members.iter()
        {
            let start = output.len();
            output.extend_from_slice(&index.get().to_le_bytes());
            output.extend_from_slice(&reference_count.to_le_bytes());
            output.extend_from_slice(&[0; OBJECT_RESERVED_LEN]);
            bytes::write_length(&mut output, payload.len().to_u64(), *lengths);
            output.resize(start + *header_size, 0);
            output.extend_from_slice(payload);
            output.resize(output.len().next_multiple_of(ALIGNMENT), 0);
        }
        if output.len() != *capacity - free.size() {
            return Err(FormatError::Internal(
                "global heap records do not match their prepared extent".into(),
            ));
        }
        if let FreeTail::Explicit(size) = free {
            output.extend_from_slice(&FREE_SPACE_INDEX.to_le_bytes());
            output.extend_from_slice(&FREE_SPACE_REFERENCE_COUNT.to_le_bytes());
            output.extend_from_slice(&[0; OBJECT_RESERVED_LEN]);
            bytes::write_length(&mut output, size.to_u64(), *lengths);
        }
        output.resize(*capacity, 0);
        Ok(output)
    }

    /// Returns the width used for collection and object sizes.
    pub const fn length_width(&self) -> LengthWidth {
        self.lengths
    }

    /// Returns the checked allocation size, including headers, padding, and free space.
    pub const fn encoded_size(&self) -> usize {
        self.capacity
    }

    /// Returns the number of supplied members, excluding the terminal free-space record.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    fn prepare(
        lengths: LengthWidth,
        members: Members<'a>,
        capacity: Capacity,
    ) -> Result<Self, FormatError> {
        let header = super::header_len(lengths);
        let used = members.iter().try_fold(header, |used, member| {
            super::add_collection_size(
                super::add_collection_size(used, header)?,
                super::aligned_collection_size(member.payload.len().to_u64())?,
            )
        })?;
        let capacity = match capacity {
            Capacity::Derived => super::aligned_collection_size(
                super::add_collection_size(used, header)?.max(MIN_COLLECTION_SIZE.to_u64()),
            )?,
            Capacity::Exact(capacity) => capacity,
        };
        if capacity < MIN_COLLECTION_SIZE.to_u64() {
            return Err(FormatError::GlobalHeapCapacityTooSmall {
                capacity,
                minimum: MIN_COLLECTION_SIZE.to_u64(),
            });
        }
        if !capacity.is_multiple_of(ALIGNMENT.to_u64()) {
            return Err(FormatError::UnalignedGlobalHeapCapacity {
                capacity,
                alignment: ALIGNMENT.to_u64(),
            });
        }
        if !width::length_width_holds(lengths, capacity) {
            return Err(FormatError::LengthTooLarge {
                length: capacity,
                length_size: lengths.get(),
            });
        }
        if used > capacity {
            return Err(FormatError::GlobalHeapCapacityExceeded {
                capacity,
                required: used,
            });
        }
        let capacity_size = capacity.to_usize()?;
        let remaining = capacity - used;
        let free = if remaining == 0 {
            FreeTail::Full
        } else if remaining < header {
            FreeTail::Implied(remaining.to_usize()?)
        } else {
            FreeTail::Explicit(remaining.to_usize()?)
        };
        Ok(Self {
            lengths,
            capacity: capacity_size,
            header_size: header.to_usize()?,
            members,
            free,
        })
    }
}

/// Borrowed input with either checked supplied indices or bounded consecutive indices.
#[derive(Debug)]
enum Members<'a> {
    Consecutive(&'a [&'a [u8]]),
    Supplied(&'a [GlobalHeapMember<'a>]),
}

impl<'a> Members<'a> {
    fn supplied(members: &'a [GlobalHeapMember<'a>]) -> Result<Self, FormatError> {
        if members.len() > GLOBAL_HEAP_MAX_OBJECTS {
            return Err(FormatError::ValueTooLargeForRepresentation {
                value: members.len().to_u64(),
                target: "u16",
            });
        }
        if members.len() > 1 {
            let mut indices = IndexSet::new();
            for member in members {
                if !indices.insert(member.index) {
                    return Err(FormatError::DuplicateGlobalHeapMemberIndex {
                        index: member.index.get(),
                    });
                }
            }
        }
        Ok(Self::Supplied(members))
    }

    fn len(&self) -> usize {
        match self {
            Self::Consecutive(objects) => objects.len(),
            Self::Supplied(members) => members.len(),
        }
    }

    fn iter(&self) -> impl Iterator<Item = GlobalHeapMember<'a>> + '_ {
        (0..self.len()).map(move |position| match self {
            Self::Consecutive(objects) => {
                // The consecutive source is created only after its length is bounded by u16::MAX.
                // Its 1-based position fits u16 and cannot be zero.
                let index =
                    NonZeroU16::new(u16::try_from(position + 1).expect("bounded member count"))
                        .expect("one-based member index");
                GlobalHeapMember::new(index, 1, objects[position])
            }
            Self::Supplied(members) => members[position],
        })
    }
}

/// An exact allocation size or a derived size with room for an explicit free-space header.
enum Capacity {
    Derived,
    Exact(u64),
}

/// The terminal extent established from the checked capacity and complete member spans.
#[derive(Debug)]
enum FreeTail {
    /// Free bytes including a complete index-zero header.
    Explicit(usize),
    /// No bytes after the final member.
    Full,
    /// Positive free bytes too short for an object header.
    Implied(usize),
}

impl FreeTail {
    fn size(&self) -> usize {
        match self {
            Self::Explicit(size) | Self::Implied(size) => *size,
            Self::Full => 0,
        }
    }
}

// Version 1 free-space objects use zero ("Global Heap", format specification, version 4.0).
const FREE_SPACE_INDEX: u16 = 0;

// Version 1 free-space objects use zero ("Global Heap", format specification, version 4.0).
const FREE_SPACE_REFERENCE_COUNT: u16 = 0;
