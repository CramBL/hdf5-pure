use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::__private as format_boundary;
use crate::btree::v2::AttributeRecord;
use crate::btree::v2::BTREE_V2_ATTRIBUTE_CREATION_ORDER;
use crate::btree::v2::BTREE_V2_ATTRIBUTE_NAME;
use crate::btree::v2::BTREE_V2_HUGE_OBJECT;
use crate::btree::v2::BTREE_V2_HUGE_OBJECT_DIRECT;
use crate::btree::v2::HugeObjectDirectRecord;
use crate::btree::v2::HugeObjectRecord;
use crate::btree::v2::write::BTREE_V2_NODE_SIZE;
use crate::btree::v2::write::BTreeV2Plan;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::write::ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE;
use crate::fractal_heap::write::ATTRIBUTE_HEAP_MAX_HEAP_SPACE;
use crate::fractal_heap::write::ManagedHeapPlan;
use crate::fractal_heap::write::ManagedHeapPlanError;
use crate::width::FormatWidths;

use super::attributes::{AttributeIndexes, EncodedDenseAttribute, EncodedDenseAttrs};
/// The length in bytes of each attribute heap ID.
pub(super) const DENSE_ATTR_HEAP_ID_LENGTH: u16 = 8;

/// The length in bytes of a name index record.
pub(super) const DENSE_ATTR_BTREE_RECORD: u16 =
    AttributeRecord::name_record_size(DENSE_ATTR_HEAP_ID_LENGTH);

/// The length in bytes of a creation-order index record.
pub(super) const DENSE_ATTR_CORDER_BTREE_RECORD: u16 =
    AttributeRecord::creation_order_record_size(DENSE_ATTR_HEAP_ID_LENGTH);

/// Heap classification, index ordering, and relative offsets retained during finalization.
pub(super) struct DenseAttrGeometry {
    widths: FormatWidths,
    max_managed: u32,
    max_direct: u64,
    block_lengths: Vec<u64>,
    /// Huge-object ID for each attribute whose bytes go outside the managed
    /// blocks. `None` for a managed one.
    huge_id_of: Vec<Option<u64>>,
    huge_count: usize,
    huge_total: u64,
    managed_plan: ManagedHeapPlan,
    name_plan: BTreeV2Plan,
    corder_plan: Option<BTreeV2Plan>,
    huge_plan: Option<BTreeV2Plan>,
    indexes: AttributeIndexes,
    /// Where each part of the blob starts, relative to the heap's own address.
    offsets: DenseAttrOffsets,
}

/// Byte offsets from the heap header for a contiguous allocation, with its total length.
pub(super) struct DenseAttrOffsets {
    pub(super) managed_off: u64,
    pub(super) btree_off: u64,
    pub(super) name_nodes_off: u64,
    pub(super) corder_bthd_off: u64,
    pub(super) corder_nodes_off: u64,
    pub(super) huge_bthd_off: u64,
    pub(super) huge_nodes_off: u64,
    pub(super) huge_data_off: u64,
    pub(super) total_len: u64,
}

impl DenseAttrGeometry {
    /// Plans heap blocks and indexes for the encoded attributes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if the huge-object count or the total encoded
    /// length of the huge objects exceeds the file's length width. Returns the other geometry
    /// errors documented by [`dense_attrs_plan`](super::dense_attrs_plan).
    pub(super) fn prepare(
        encoded: EncodedDenseAttrs,
    ) -> Result<(Self, Vec<EncodedDenseAttribute>), FormatError> {
        let (widths, attributes, indexes) = encoded.into_parts();
        let count = attributes.len();
        let max_direct = crate::fractal_heap::write::maximum_direct_block_size(
            ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE,
            widths.lengths,
        );
        let max_managed: u32 = (max_direct
            - crate::fractal_heap::write::attribute_heap_direct_block_header(widths.offsets)
                .to_u64())
        .narrow()?;

        // Huge object IDs start at one in attribute order. The header stores the last assigned ID.
        // `H5HF__huge_new_id` pre-increments the counter (`H5HFhuge.c`, HDF5 2.2.0).
        let mut huge_id_of: Vec<Option<u64>> = vec![None; count];
        let mut huge_count: usize = 0;
        for (slot, attribute) in huge_id_of.iter_mut().zip(&attributes) {
            if attribute.bytes().len().to_u64() > u64::from(max_managed) {
                huge_count += 1;
                *slot = Some(huge_count as u64);
            }
        }
        let huge_total: u64 = attributes
            .iter()
            .zip(&huge_id_of)
            .filter(|(_, id)| id.is_some())
            .map(|(s, _)| s.bytes().len() as u64)
            .sum();
        bytes::require_length(huge_total, widths.lengths)?;
        bytes::require_length(huge_count.to_u64(), widths.lengths)?;

        // Only managed objects occupy the doubling table, so only they are planned
        // into it. `managed_sizes` holds their sizes in attribute order, and the
        // plan's object indices are indices into that order.
        let managed_sizes: Vec<u64> = attributes
            .iter()
            .zip(&huge_id_of)
            .filter(|(_, id)| id.is_none())
            .map(|(s, _)| s.bytes().len() as u64)
            .collect();
        let managed_plan = ManagedHeapPlan::attributes_with_widths(&managed_sizes, widths)
            .map_err(dense_attr_heap_error)?;

        // The writer uses the same widths for every version 2 B-tree header.
        let bthd_size = format_boundary::btree_v2_header_size(widths.offsets, widths.lengths);
        debug_assert_eq!(
            bthd_size,
            4 + 1
                + 1
                + 4
                + 2
                + 2
                + 1
                + 1
                + widths.offsets.get() as usize
                + 2
                + widths.lengths.get() as usize
                + 4
        );

        let plan_index = |kind, count, record_size| {
            BTreeV2Plan::new(kind, count, record_size, BTREE_V2_NODE_SIZE, widths.offsets)
                .ok_or_else(|| {
                    FormatError::Internal(format!(
                        "cannot plan a type-{kind} B-tree with {count} dense attribute records"
                    ))
                })
        };
        let name_plan = plan_index(BTREE_V2_ATTRIBUTE_NAME, count, DENSE_ATTR_BTREE_RECORD)?;
        let corder_plan = indexes
            .indexed()
            .then(|| {
                plan_index(
                    BTREE_V2_ATTRIBUTE_CREATION_ORDER,
                    count,
                    DENSE_ATTR_CORDER_BTREE_RECORD,
                )
            })
            .transpose()?;
        let direct =
            FractalHeapHeader::unfiltered_huge_ids_direct(DENSE_ATTR_HEAP_ID_LENGTH, widths);
        let huge_plan = (huge_count > 0)
            .then(|| {
                let (kind, size) = if direct {
                    (
                        BTREE_V2_HUGE_OBJECT_DIRECT,
                        HugeObjectDirectRecord::size(widths.offsets, widths.lengths),
                    )
                } else {
                    (
                        BTREE_V2_HUGE_OBJECT,
                        HugeObjectRecord::size(widths.offsets, widths.lengths),
                    )
                };
                plan_index(kind, huge_count, size)
            })
            .transpose()?;

        // The blob contains the heap header, managed blocks, name index, optional creation-order
        // index, and optional huge-object index and payloads, in that order. The offsets are
        // relative to the heap address, and the total length is independent of that address.
        let managed_off = FractalHeapHeader::serialized_size(widths.offsets, widths.lengths) as u64;
        let btree_off = managed_off + managed_plan.region_size();
        let name_nodes_off = btree_off + bthd_size as u64;
        let corder_bthd_off = name_nodes_off + name_plan.nodes_size();
        let corder_nodes_off = corder_bthd_off
            + if corder_plan.is_some() {
                bthd_size as u64
            } else {
                0
            };
        let huge_bthd_off =
            corder_nodes_off + corder_plan.as_ref().map_or(0, BTreeV2Plan::nodes_size);
        let huge_nodes_off = huge_bthd_off + bthd_size as u64;
        let huge_data_off = huge_nodes_off + huge_plan.as_ref().map_or(0, BTreeV2Plan::nodes_size);
        // With no huge objects the blob ends where the huge index would have begun.
        // With them it ends past their bytes.
        let total_len = if huge_count > 0 {
            huge_data_off
                .checked_add(huge_total)
                .ok_or(FormatError::OffsetOverflow {
                    offset: huge_data_off,
                    length: huge_total,
                })?
        } else {
            huge_bthd_off
        };

        total_len.to_usize()?;

        let block_lengths = core::iter::once(managed_off)
            .chain(managed_plan.block_lengths())
            .chain(
                core::iter::once(&name_plan)
                    .chain(corder_plan.iter())
                    .chain(huge_plan.iter())
                    .flat_map(|tree| {
                        core::iter::once(bthd_size.to_u64()).chain(tree.block_lengths())
                    }),
            )
            .chain(
                attributes
                    .iter()
                    .zip(&huge_id_of)
                    .filter(|(_, id)| id.is_some())
                    .map(|(attribute, _)| attribute.bytes().len().to_u64()),
            )
            .collect();

        Ok((
            Self {
                widths,
                max_managed,
                max_direct,
                block_lengths,
                huge_id_of,
                huge_count,
                huge_total,
                managed_plan,
                name_plan,
                corder_plan,
                huge_plan,
                indexes,
                offsets: DenseAttrOffsets {
                    managed_off,
                    btree_off,
                    name_nodes_off,
                    corder_bthd_off,
                    corder_nodes_off,
                    huge_bthd_off,
                    huge_nodes_off,
                    huge_data_off,
                    total_len,
                },
            },
            attributes,
        ))
    }

    pub(super) fn widths(&self) -> FormatWidths {
        self.widths
    }

    pub(super) fn max_managed(&self) -> u32 {
        self.max_managed
    }

    pub(super) fn max_direct(&self) -> u64 {
        self.max_direct
    }

    pub(super) fn block_lengths(&self) -> &[u64] {
        &self.block_lengths
    }

    pub(super) fn huge_id_of(&self) -> &[Option<u64>] {
        &self.huge_id_of
    }

    pub(super) fn huge_count(&self) -> usize {
        self.huge_count
    }

    pub(super) fn huge_total(&self) -> u64 {
        self.huge_total
    }

    pub(super) fn managed_plan(&self) -> &ManagedHeapPlan {
        &self.managed_plan
    }

    pub(super) fn name_plan(&self) -> &BTreeV2Plan {
        &self.name_plan
    }

    pub(super) fn corder_plan(&self) -> Option<&BTreeV2Plan> {
        self.corder_plan.as_ref()
    }

    pub(super) fn huge_plan(&self) -> Option<&BTreeV2Plan> {
        self.huge_plan.as_ref()
    }

    pub(super) fn offsets(&self) -> &DenseAttrOffsets {
        &self.offsets
    }

    pub(super) fn indexes(&self) -> &AttributeIndexes {
        &self.indexes
    }
}

fn dense_attr_heap_error(error: ManagedHeapPlanError) -> FormatError {
    match error {
        ManagedHeapPlanError::HeapSpace => FormatError::DenseAttributeHeapTooLarge {
            limit: ATTRIBUTE_HEAP_MAX_HEAP_SPACE,
        },
        ManagedHeapPlanError::Host { bytes } => FormatError::ValueTooLargeForPlatform {
            value: bytes,
            target: "usize",
        },
        ManagedHeapPlanError::Geometry | ManagedHeapPlanError::ObjectSize { .. } => {
            FormatError::Internal(format!(
                "dense attribute managed heap planning failed: {error:?}"
            ))
        }
    }
}

#[cfg(test)]
#[path = "tests/geometry.rs"]
mod tests;
