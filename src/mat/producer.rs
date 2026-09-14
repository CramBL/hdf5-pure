//! Staging a dataset whose bytes are produced at write time.
//!
//! A [`DataProducer`] hands the writer one block of raw elements at a time,
//! during the emission pass, so a dataset written through
//! [`MatBuilder::write_blocks`](crate::mat::MatBuilder::write_blocks) is never
//! fully resident. Paired with
//! [`MatBuilder::finish_to`](crate::mat::MatBuilder::finish_to) — which does not
//! hold the assembled file either — a `.mat` is written in one block plus the
//! file's metadata, whatever its size.
//!
//! The dataset that comes out is byte-for-byte the one the ordinary writers
//! produce for the same content: contiguous storage, laid out from the shape
//! alone, with the blocks written back to back into the region the layout
//! reserved. The choice of API changes the peak memory and nothing else.
//!
//! # Uncompressed only
//!
//! The writer computes every object's address before it emits a byte, which
//! means it needs the data region's exact size up front. Uncompressed, that is
//! pure geometry. Compressed, it is not knowable without compressing — which
//! would buffer the very data this path exists to avoid. So a producer-backed
//! dataset is always stored unfiltered, and asking for one on a builder
//! configured for deflate is refused rather than silently downgraded.
//!
//! # Element order
//!
//! MATLAB is column-major and HDF5 is row-major, so a MATLAB array is stored
//! with its dimensions reversed and its elements in the very order MATLAB reads
//! them. A producer therefore emits elements in **MATLAB's linear order**: the
//! first index varies fastest. Block `i` is a contiguous run of that order,
//! continuing where block `i - 1` stopped.
//!
//! For an acquisition that matters, because it fixes which shape to ask for.
//! `[channels, samples]` puts all of a timestep's channels next to each other,
//! so the blocks run forward through time — the order the samples arrive in. The
//! transpose, `[samples, channels]`, stores channel 0's entire history before
//! channel 1's, so no producer can emit it as an acquisition proceeds.

use crate::chunked_write::ChunkProvider;
use crate::convert::Narrow;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::mat::class::MatClass;
use crate::mat::error::MatError;
use crate::type_builders::{
    CompoundTypeBuilder, make_f32_type, make_f64_type, make_i8_type, make_i16_type, make_i32_type,
    make_i64_type, make_u8_type, make_u16_type, make_u32_type, make_u64_type,
};
use std::sync::{Arc, Mutex};

/// Byte size a block aims for. Large enough that a producer call's overhead is
/// negligible, small enough to be an unremarkable allocation. Blocks are whole
/// numbers of elements, so an element wider than this gets one element per block
/// rather than a split that would cut an element in half.
const TARGET_BLOCK_BYTES: u64 = 1 << 20;

/// One block the writer is asking for: everything a producer needs to fill it,
/// handed over rather than looked up.
///
/// The contract travels with the request on purpose. A producer that had to
/// fetch its own copy of the blocking beforehand could fetch one that disagrees
/// with the dataset it is later staged against — a mismatch nothing would catch
/// until the write was already partly on the sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Block {
    /// Which block this is. Blocks are requested in ascending order, once each,
    /// starting at zero.
    pub index: usize,
    /// Index of this block's first element in MATLAB's linear element order, so
    /// a producer computing values from position does not have to track how far
    /// it has come.
    pub first_element: u64,
    /// Elements this block carries. The final block may carry fewer than the
    /// others; every other block is the same size.
    pub elements: u64,
    /// Bytes one element occupies. A complex element counts both components.
    pub element_size: usize,
}

impl Block {
    /// Bytes the producer must write for this block: `elements * element_size`.
    pub fn len(&self) -> usize {
        // Every planned block fits `usize` — checked when the blocking is built.
        // `Block` is `#[non_exhaustive]` with public fields, so a caller can still
        // copy one and rewrite them; saturating means no producer can satisfy the
        // result and the write fails as a size mismatch rather than on a number
        // that quietly wrapped.
        usize::try_from(self.elements)
            .ok()
            .and_then(|n| n.checked_mul(self.element_size))
            .unwrap_or(usize::MAX)
    }

    /// Whether this block carries no elements at all. Only an empty dataset
    /// produces one, and an empty dataset is never asked for a block.
    pub fn is_empty(&self) -> bool {
        self.elements == 0
    }
}

/// Yields a dataset's raw, uncompressed element bytes on demand, one block at a
/// time.
///
/// The writer calls [`block_bytes`](Self::block_bytes) once per block, in
/// ascending index order, during the emission pass — never during layout, which
/// works from the dataset's shape alone. Implementations therefore do not need
/// to be re-runnable, and generating each block on the fly is the intended use.
///
/// `Send + Sync` is required because a staged producer is owned by the builder,
/// and the builder would otherwise lose those auto-traits.
pub trait DataProducer: Send + Sync {
    /// Append `block`'s raw little-endian element bytes to `out`, in MATLAB's
    /// linear element order (see the [module docs](self)).
    ///
    /// `out` is handed over empty and is the same buffer on every call, so it
    /// costs one buffer for the whole dataset rather than one per block. Write
    /// exactly [`Block::len`] bytes; any other count is refused with
    /// [`MatError::BlockSizeMismatch`] rather than written, because a block of
    /// the wrong size shifts every address after it.
    fn block_bytes(&self, block: Block, out: &mut Vec<u8>) -> Result<(), MatError>;
}

/// How a producer-backed dataset was split for the write pass.
///
/// [`MatBuilder::write_blocks`](crate::mat::MatBuilder::write_blocks) chooses this
/// and hands each [`Block`] to the producer, so a caller needs it only to size a
/// buffer or report progress. [`Blocking::plan`] computes it from a shape alone,
/// ahead of staging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Blocking {
    /// Number of blocks the producer will be asked for: indices `0..block_count`.
    /// Zero for an empty dataset, which is written as a MATLAB empty marker and
    /// never calls the producer.
    pub block_count: usize,
    /// Elements in every block but the last.
    pub block_elements: u64,
    /// Elements in the last block. Equal to `block_elements` when the dataset
    /// divides evenly.
    pub last_block_elements: u64,
    /// Bytes one element occupies. A complex element counts both components.
    pub element_size: usize,
}

impl Blocking {
    /// The block at `index`, as the producer will be handed it.
    ///
    /// Every block but the last is the same size; the last is whatever remains.
    /// An out-of-range index reports an empty block, since the writer never asks
    /// for one.
    pub fn block(&self, index: usize) -> Block {
        // Ordered so an out-of-range index leaves before any arithmetic: `index`
        // is caller-supplied, and `index + 1` overflows at `usize::MAX`.
        let elements = if index >= self.block_count {
            0
        } else if index == self.block_count - 1 {
            self.last_block_elements
        } else {
            self.block_elements
        };
        Block {
            index,
            first_element: self.block_elements.saturating_mul(index as u64),
            elements,
            element_size: self.element_size,
        }
    }

    /// Bytes the producer must write for block `index`.
    pub fn block_len(&self, index: usize) -> usize {
        self.block(index).len()
    }

    /// Total bytes the dataset's elements occupy.
    ///
    /// Saturating, for the same reason [`block_len`](Self::block_len) is careful:
    /// the fields are public, so a caller can copy this and rewrite them into a
    /// combination the planner would never produce.
    pub fn total_len(&self) -> u64 {
        if self.block_count == 0 {
            return 0;
        }
        self.block_elements
            .saturating_mul(self.block_count as u64 - 1)
            .saturating_add(self.last_block_elements)
            .saturating_mul(self.element_size as u64)
    }

    /// The blocking [`write_blocks`](crate::mat::MatBuilder::write_blocks) will
    /// choose for a dataset of this MATLAB shape and element type.
    ///
    /// Deterministic, and computed from the shape alone, so a producer can be
    /// built against it before the dataset is staged.
    pub fn plan<T: BlockElement>(matlab_dims: &[usize]) -> Result<Blocking, MatError> {
        plan_blocking(total_elements(matlab_dims), T::ELEMENT_SIZE)
    }
}

/// Elements a MATLAB shape holds. An empty shape (`[]`) is one element, matching
/// [`matrix_dims`](crate::mat::dims::matrix_dims)'s scalar collapse.
///
/// Saturates rather than overflows: dimensions come from the caller, so an
/// absurd shape must reach the planner's refusal rather than panic a debug build
/// on the way there. A saturated count is larger than any host can address, so it
/// is refused for the right reason.
pub(crate) fn total_elements(matlab_dims: &[usize]) -> u64 {
    matlab_dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
        .unwrap_or(u64::MAX)
}

/// Split `total` elements into blocks of about [`TARGET_BLOCK_BYTES`].
///
/// Blocks are whole numbers of elements and successive runs of the dataset's
/// linear order, which is what lets a producer treat block `i` as "the next
/// chunk of my stream" rather than having to index into a grid.
pub(crate) fn plan_blocking(total: u64, element_size: usize) -> Result<Blocking, MatError> {
    if total == 0 || element_size == 0 {
        return Ok(Blocking {
            block_count: 0,
            block_elements: 0,
            last_block_elements: 0,
            element_size,
        });
    }
    // The data region's byte count is the first thing that has to be real. The
    // writer accumulates it into a `usize` address cursor and must be able to
    // emit it, so a shape whose bytes exceed this host's address space is refused
    // here — before the count reaches arithmetic that would wrap, and before a
    // producer is asked for the first of eighteen exabytes of blocks.
    let total_bytes = total
        .checked_mul(element_size as u64)
        .filter(|&bytes| usize::try_from(bytes).is_ok())
        .ok_or_else(|| too_large(total.saturating_mul(element_size as u64)))?;

    let per_block = (TARGET_BLOCK_BYTES / element_size as u64).clamp(1, total);
    let block_count = total.div_ceil(per_block);
    let last = total - (block_count - 1) * per_block;

    // The emitter builds one block in memory at a time, so the block — not the
    // dataset — has to fit this host. Only reachable on a 32-bit target, and only
    // for an element wider than the target block size.
    let block_bytes = per_block * element_size as u64;
    if usize::try_from(block_bytes).is_err() {
        return Err(too_large(block_bytes));
    }
    debug_assert!(total_bytes >= block_bytes);

    Ok(Blocking {
        block_count: block_count.narrow_or_else(|| too_large(block_count))?,
        block_elements: per_block,
        last_block_elements: last,
        element_size,
    })
}

fn too_large(value: u64) -> MatError {
    MatError::Hdf5(crate::error::Error::Format(
        FormatError::ValueTooLargeForPlatform {
            value,
            target: "usize",
        },
    ))
}

/// Adapts a [`DataProducer`] to the writer's block provider: checks each block's
/// length and carries a producer's own error back out.
///
/// The error detour exists because the writer's provider seam speaks
/// [`FormatError`], which cannot carry a [`MatError`]. Stashing the real error
/// here and swapping it back in `MatBuilder`'s finalizers keeps a producer's
/// failure intact instead of flattening it to a message.
pub(crate) struct ProducerChunks {
    pub(crate) producer: Box<dyn DataProducer>,
    pub(crate) blocking: Blocking,
    pub(crate) error: Arc<Mutex<Option<MatError>>>,
}

impl ProducerChunks {
    /// Record `error` (the first one wins) and return the placeholder the writer
    /// will carry until the finalizer swaps the real one back in.
    fn fail(&self, error: MatError) -> FormatError {
        // A poisoned lock still holds the slot, and dropping the producer's real
        // error to report a placeholder instead is the one outcome worth avoiding
        // here.
        let mut slot = self
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slot.get_or_insert(error);
        FormatError::SerializationError("a dataset's block producer failed".into())
    }
}

impl ChunkProvider for ProducerChunks {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let block = self.blocking.block(index);
        let expected = block.len();
        if let Err(e) = self.producer.block_bytes(block, out) {
            return Err(self.fail(e));
        }
        if out.len() != expected {
            return Err(self.fail(MatError::BlockSizeMismatch {
                block: index,
                expected,
                actual: out.len(),
            }));
        }
        Ok(())
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An element type a producer-backed dataset can hold.
///
/// Implemented for the ten numeric widths MATLAB has classes for, for `bool`
/// (MATLAB `logical`), and for the `(T, T)` pair of any of those widths, which
/// is a complex array of that class — the same `(real, imag)` framing
/// [`write_complex_f64`](crate::mat::MatBuilder::write_complex_f64) and its
/// siblings take. Sealed: the set of MATLAB classes is fixed by the format.
pub trait BlockElement: sealed::Sealed {
    /// The MATLAB class the dataset reports. For a complex pair this is the
    /// *component* class, which is how MATLAB tells `complex(int16(..))` from a
    /// complex `double`.
    const CLASS: MatClass;
    /// Bytes one element occupies on disk. A complex pair counts both components.
    const ELEMENT_SIZE: usize;
    /// The `MATLAB_int_decode` attribute value, for the classes that carry one.
    const INT_DECODE: Option<i32> = None;
    /// The HDF5 datatype of one element.
    fn datatype() -> Datatype;
}

macro_rules! block_elements {
    ($($ty:ty => $class:ident, $make:ident),* $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            impl BlockElement for $ty {
                const CLASS: MatClass = MatClass::$class;
                const ELEMENT_SIZE: usize = size_of::<$ty>();
                fn datatype() -> Datatype {
                    $make()
                }
            }

            impl sealed::Sealed for ($ty, $ty) {}
            impl BlockElement for ($ty, $ty) {
                const CLASS: MatClass = MatClass::$class;
                const ELEMENT_SIZE: usize = 2 * size_of::<$ty>();
                fn datatype() -> Datatype {
                    CompoundTypeBuilder::new()
                        .field("real", $make())
                        .field("imag", $make())
                        .build()
                        .expect("two fields of a nonzero-width component")
                }
            }
        )*
    };
}

block_elements! {
    f64 => Double, make_f64_type,
    f32 => Single, make_f32_type,
    i8  => Int8,   make_i8_type,
    i16 => Int16,  make_i16_type,
    i32 => Int32,  make_i32_type,
    i64 => Int64,  make_i64_type,
    u8  => UInt8,  make_u8_type,
    u16 => UInt16, make_u16_type,
    u32 => UInt32, make_u32_type,
    u64 => UInt64, make_u64_type,
}

impl sealed::Sealed for bool {}
impl BlockElement for bool {
    const CLASS: MatClass = MatClass::Logical;
    const ELEMENT_SIZE: usize = 1;
    const INT_DECODE: Option<i32> = Some(1);
    fn datatype() -> Datatype {
        make_u8_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocking is derived from the element count alone, so these are the
    /// numbers a caller can plan against before staging anything. Derived here
    /// by a different route than the planner uses — summing the block lengths
    /// rather than re-running its arithmetic.
    #[test]
    fn the_blocks_partition_the_dataset() {
        // 3_000_000 f64 = 24 MB; a 1 MiB block is 131_072 elements.
        let b = plan_blocking(3_000_000, 8).unwrap();
        assert_eq!(b.block_elements, 131_072);
        assert!(b.block_count > 1, "the fixture must span several blocks");

        let total: usize = (0..b.block_count).map(|i| b.block_len(i)).sum();
        assert_eq!(total as u64, 3_000_000 * 8);
        assert_eq!(total as u64, b.total_len());
        // Every block but the last is full, and the last is a partial remainder
        // rather than a padded one.
        for i in 0..b.block_count - 1 {
            assert_eq!(b.block_len(i), 131_072 * 8);
        }
        assert!(b.last_block_elements < b.block_elements);
        assert_eq!(b.block_len(b.block_count), 0, "no block past the last");
    }

    /// An element wider than the target block is not split: a block is a whole
    /// number of elements, so one element is the floor.
    #[test]
    fn an_element_wider_than_the_target_becomes_the_block() {
        let huge = (TARGET_BLOCK_BYTES + 8) as usize;
        let b = plan_blocking(3, huge).unwrap();
        assert_eq!(b.block_elements, 1);
        assert_eq!(b.block_count, 3);
        assert_eq!(b.block_len(0), huge);
    }

    /// A dataset that fits one block has no short tail, and `last` still reports
    /// the real count rather than zero.
    #[test]
    fn a_dataset_smaller_than_a_block_is_one_full_block() {
        let b = plan_blocking(8, 8).unwrap();
        assert_eq!(b.block_count, 1);
        assert_eq!(b.block_elements, 8);
        assert_eq!(b.last_block_elements, 8);
        assert_eq!(b.block_len(0), 64);
        assert_eq!(b.total_len(), 64);
    }

    /// An empty dataset asks the producer for nothing at all.
    #[test]
    fn an_empty_shape_plans_no_blocks() {
        let b = plan_blocking(total_elements(&[0, 0]), 8).unwrap();
        assert_eq!(b.block_count, 0);
        assert_eq!(b.block_len(0), 0);
        assert_eq!(b.total_len(), 0);
    }

    /// A shape nothing could hold is refused rather than wrapped into a small
    /// plausible one. `total_len` and the block arithmetic are all derived from
    /// the byte count, so it has to be the thing that is checked.
    ///
    /// The boundary is this host's address space, not a fixed number: the
    /// emitter builds the region in memory, so what plans on a 64-bit target is
    /// legitimately refused on a 32-bit one. Written in terms of `usize::MAX` so
    /// it states that rule rather than one target's answer to it.
    #[test]
    fn a_shape_whose_bytes_overflow_is_refused() {
        // Refused on any target: the byte count overflows `u64` itself.
        assert!(plan_blocking(u64::MAX, 8).is_err());

        // The largest shape that plans is the one whose bytes still fit `usize`,
        // and one element more is refused.
        let max_elements = (usize::MAX as u64) / 8;
        let b = plan_blocking(max_elements, 8).unwrap();
        assert_eq!(b.total_len(), max_elements * 8);
        assert!(
            usize::try_from(b.total_len()).is_ok(),
            "the largest plannable region must be one this host can address"
        );
        assert!(plan_blocking(max_elements + 1, 8).is_err());

        // Dimensions are multiplied the same way. Three of `usize::MAX` exceed
        // `u64` on every target, so this pins that they saturate rather than
        // panicking a debug build on the way to the refusal.
        assert_eq!(
            total_elements(&[usize::MAX, usize::MAX, usize::MAX]),
            u64::MAX
        );
        assert!(Blocking::plan::<f64>(&[usize::MAX, usize::MAX, usize::MAX]).is_err());
    }

    /// A complex pair is one element of two components, so it blocks exactly as
    /// a real array of twice the width would, and reports the component class.
    #[test]
    fn a_complex_pair_counts_both_components() {
        assert_eq!(<(i16, i16) as BlockElement>::ELEMENT_SIZE, 4);
        assert_eq!(<(i16, i16) as BlockElement>::CLASS, MatClass::Int16);
        let b = Blocking::plan::<(i16, i16)>(&[2, 10]).unwrap();
        assert_eq!(b.element_size, 4);
        assert_eq!(b.total_len(), 2 * 10 * 4);
    }

    /// `MatBuilder` now transitively holds a boxed producer and an
    /// `Arc<Mutex<..>>`, either of which could strip an auto-trait. Losing one is
    /// a semver break, so pin all four the way `FileBuilder` does.
    #[test]
    fn mat_builder_keeps_its_auto_traits() {
        fn assert_auto_traits<
            T: Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe,
        >() {
        }
        assert_auto_traits::<crate::mat::MatBuilder>();
    }

    /// Each block hands the producer a start and a length that together tile the
    /// dataset exactly: block `i` begins where block `i - 1` ended, and the last
    /// one ends at the final element. A producer relies on this to generate
    /// values from position without tracking how far it has come.
    #[test]
    fn the_blocks_tile_the_dataset_without_gap_or_overlap() {
        let b = plan_blocking(3_000_000, 8).unwrap();
        let mut expected_start = 0u64;
        for i in 0..b.block_count {
            let block = b.block(i);
            assert_eq!(block.index, i);
            assert_eq!(
                block.first_element,
                expected_start,
                "block {i} must start where block {} ended",
                i.wrapping_sub(1)
            );
            assert!(!block.is_empty());
            assert_eq!(block.len(), block.elements as usize * 8);
            expected_start += block.elements;
        }
        assert_eq!(
            expected_start, 3_000_000,
            "the blocks together must be the whole dataset"
        );
        // Past the end there is nothing to write, and nowhere to write it.
        assert!(b.block(b.block_count).is_empty());
        assert_eq!(b.block(b.block_count).len(), 0);
    }

    /// The fields are public, so a caller can copy a `Blocking` and rewrite them.
    /// Nothing they can write may panic these two: an out-of-range index is
    /// documented to report zero, and a forged size reports a saturated total
    /// rather than a wrapped one.
    #[test]
    fn a_forged_blocking_reports_rather_than_panics() {
        let planned = plan_blocking(10, 8).unwrap();
        assert_eq!(planned.block_len(usize::MAX), 0);
        assert_eq!(planned.block_len(planned.block_count), 0);
        // Including the `first_element` a wild index would imply.
        assert!(planned.block(usize::MAX).is_empty());

        let mut forged = planned;
        forged.block_count = 3;
        forged.block_elements = u64::MAX / 2;
        forged.last_block_elements = u64::MAX;
        assert_eq!(forged.total_len(), u64::MAX);
        // And the per-block length saturates rather than wrapping into a small
        // plausible number, so no producer can accidentally satisfy it.
        assert_eq!(forged.block_len(0), usize::MAX);
    }

    /// `logical` is the one class that carries a decode flag, and it is stored
    /// one byte per element.
    #[test]
    fn logical_is_a_byte_with_a_decode_flag() {
        assert_eq!(<bool as BlockElement>::ELEMENT_SIZE, 1);
        assert_eq!(<bool as BlockElement>::INT_DECODE, Some(1));
        assert_eq!(<f64 as BlockElement>::INT_DECODE, None);
    }
}
