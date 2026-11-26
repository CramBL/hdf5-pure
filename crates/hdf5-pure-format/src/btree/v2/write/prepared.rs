use alloc::vec;
use alloc::vec::Vec;

use super::plan::BTreeV2Plan;
use crate::convert::Narrow;
use crate::error::FormatError;

/// The root record count and subtree totals for a plan whose nodes fit their allocations.
///
/// The writer computes subtree totals with checked arithmetic in child-before-parent order.
/// Each child count fits its field in the parent. The root total equals the tree's record count.
pub(super) struct CheckedNodeEncoding<'a> {
    plan: &'a BTreeV2Plan,
    subtree_total: Vec<u64>,
    root_records: u16,
}

impl<'a> CheckedNodeEncoding<'a> {
    pub(super) fn root_records(&self) -> u16 {
        self.root_records
    }

    pub(super) fn subtree_total(&self, index: usize) -> u64 {
        self.subtree_total[index]
    }

    pub(super) fn plan(&self) -> &'a BTreeV2Plan {
        self.plan
    }

    pub(super) fn new(plan: &'a BTreeV2Plan) -> Result<Self, FormatError> {
        let node_size = plan.node_size().to_usize()?;
        let mut subtree_total = vec![0u64; plan.nodes().len()];
        for (index, node) in plan.nodes().iter().enumerate() {
            let record_count = node.records().len().to_u64();
            let pointer_width = usize::from(plan.offset_width().get())
                + plan.info().max_nrec_size()
                + plan.info().total_nrec_size(node.depth());
            if node
                .encoded_size(plan.record_size(), pointer_width)
                .is_none_or(|length| length > node_size)
            {
                return Err(FormatError::Internal(
                    "planned B-tree node exceeds its allocation".into(),
                ));
            }
            let mut total = record_count;
            for &child in node.children() {
                let child_total = *subtree_total
                    .get(..index)
                    .and_then(|totals| totals.get(child))
                    .ok_or_else(|| {
                        FormatError::Internal("B-tree child does not precede its parent".into())
                    })?;
                let child_count = plan.nodes()[child].records().len().to_u64();
                let total_width = plan.info().total_nrec_size(node.depth());
                if !count_fits(child_count, plan.info().max_nrec_size())
                    || (total_width != 0 && !count_fits(child_total, total_width))
                {
                    return Err(FormatError::Internal(
                        "planned B-tree count exceeds its field width".into(),
                    ));
                }
                total = total.checked_add(child_total).ok_or_else(|| {
                    FormatError::Internal("planned B-tree subtree count exceeds u64::MAX".into())
                })?;
            }
            subtree_total[index] = total;
        }
        if subtree_total.last().copied().unwrap_or(0) != plan.total_records() {
            return Err(FormatError::Internal(
                "planned B-tree subtree count differs from the declared total".into(),
            ));
        }
        let root_records = u16::try_from(
            plan.nodes().last().map_or(0, |root| root.records().len()),
        )
        .map_err(|_error| FormatError::Internal("B-tree root count exceeds u16::MAX".into()))?;
        Ok(Self {
            plan,
            subtree_total,
            root_records,
        })
    }
}

pub(super) fn count_fits(value: u64, width: usize) -> bool {
    width <= core::mem::size_of::<u64>()
        && (width == core::mem::size_of::<u64>() || value >> (width * 8) == 0)
}
