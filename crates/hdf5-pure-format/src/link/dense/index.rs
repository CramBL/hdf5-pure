use alloc::vec::Vec;

use super::error::DenseLinkPlanError;
use crate::btree::v2 as btree_v2;
use crate::btree::v2::write::BTreeV2Plan;
use crate::btree::v2::write::BTreeV2WriteDialect;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapIdLayout;
use crate::fractal_heap::PreparedHeapId;
use crate::link::storage::LinkStorageError;
use crate::link::storage::OwnedLink;
use crate::width::FormatWidths;

/// Sorted index records, their B-tree layout, and the offsets of the header and nodes.
pub(super) struct PlannedIndex {
    plan: BTreeV2Plan,
    records: Vec<u8>,
    /// The header offset from the start of the dense-link allocation.
    offset: u64,
    /// The first node offset from the start of the dense-link allocation.
    nodes_offset: u64,
}

impl PlannedIndex {
    pub(super) fn tree(&self) -> &BTreeV2Plan {
        &self.plan
    }

    pub(super) fn records(&self) -> &[u8] {
        &self.records
    }

    pub(super) fn offset(&self) -> u64 {
        self.offset
    }

    pub(super) fn nodes_offset(&self) -> u64 {
        self.nodes_offset
    }

    /// Plans an index with its header at the byte offset in `cursor`.
    ///
    /// On success, advances `cursor` past the header and all allocated nodes.
    ///
    /// # Errors
    ///
    /// Returns [`DenseLinkPlanError::Format`] if the expected record byte count exceeds
    /// [`usize::MAX`] or differs from the length of `records`. Returns
    /// [`DenseLinkPlanError::IndexGeometry`] if the records cannot be laid out in the index,
    /// or [`DenseLinkPlanError::LengthOverflow`] if a length or offset exceeds [`u64::MAX`].
    pub(super) fn new(
        widths: FormatWidths,
        tree_type: u8,
        count: usize,
        record_len: u16,
        records: Vec<u8>,
        cursor: &mut u64,
    ) -> Result<Self, DenseLinkPlanError> {
        let FormatWidths {
            offsets: offset_width,
            lengths: length_width,
        } = widths;
        let expected = count
            .to_u64()
            .checked_mul(u64::from(record_len))
            .ok_or(DenseLinkPlanError::LengthOverflow)?;
        expected.to_usize()?;
        if records.len().to_u64() != expected {
            return Err(FormatError::Internal(
                "dense link index record length differs from its plan".into(),
            )
            .into());
        }
        let plan = BTreeV2Plan::new_with_undefined_empty_root(
            BTreeV2WriteDialect::Libhdf5Global,
            tree_type,
            count,
            record_len,
            GROUP_INDEX_NODE_SIZE,
            offset_width,
        )
        .ok_or(DenseLinkPlanError::IndexGeometry)?;
        let offset = *cursor;
        let nodes_offset = offset
            .checked_add(btree_v2::btree_v2_header_size(offset_width, length_width).to_u64())
            .ok_or(DenseLinkPlanError::LengthOverflow)?;
        *cursor = nodes_offset
            .checked_add(
                plan.checked_nodes_size()
                    .ok_or(DenseLinkPlanError::LengthOverflow)?,
            )
            .ok_or(DenseLinkPlanError::LengthOverflow)?;
        Ok(Self {
            plan,
            records,
            offset,
            nodes_offset,
        })
    }
}

/// The group's links paired by position with their encoded heap IDs.
pub(super) struct LinkIndexEntries<'a> {
    links: &'a [OwnedLink],
    layout: &'a FractalHeapIdLayout,
    ids: &'a [Vec<u8>],
}

impl<'a> LinkIndexEntries<'a> {
    pub(super) fn new(
        links: &'a [OwnedLink],
        layout: &'a FractalHeapIdLayout,
        ids: &'a [Vec<u8>],
    ) -> Result<Self, FormatError> {
        if links.len() != ids.len() {
            return Err(FormatError::Internal(
                "dense link count differs from heap ID count".into(),
            ));
        }
        Ok(Self { links, layout, ids })
    }

    /// Encodes records in ascending creation order.
    ///
    /// # Errors
    ///
    /// Returns [`DenseLinkPlanError::Link`] if a link lacks a creation index, or
    /// [`DenseLinkPlanError::Format`] if a heap ID is invalid for the supplied layout.
    pub(super) fn creation_records(&self) -> Result<Vec<u8>, DenseLinkPlanError> {
        let mut order = self
            .links
            .iter()
            .enumerate()
            .map(|(index, link)| {
                link.creation_order()
                    .map(|creation| (creation, index))
                    .ok_or(LinkStorageError::CreationOrderPresence)
            })
            .collect::<Result<Vec<_>, _>>()?;
        order.sort_unstable_by_key(|&(creation, _)| creation);
        let mut records = Vec::new();
        for (creation, index) in order {
            let id = PreparedHeapId::new(self.layout, &self.ids[index])?;
            btree_v2::LinkIndexKey::CreationOrder(creation).append_record(&mut records, id);
        }
        Ok(records)
    }

    /// Encodes records in ascending order of name hash, then of name bytes.
    ///
    /// # Errors
    ///
    /// Returns [`DenseLinkPlanError::Format`] if a heap ID is invalid for the supplied layout.
    pub(super) fn name_records(&self) -> Result<Vec<u8>, DenseLinkPlanError> {
        let mut order = self
            .links
            .iter()
            .enumerate()
            .map(|(index, link)| (checksum::jenkins_lookup3(link.name()), index))
            .collect::<Vec<_>>();
        order.sort_unstable_by(|&(hash_a, index_a), &(hash_b, index_b)| {
            hash_a
                .cmp(&hash_b)
                .then_with(|| self.links[index_a].name().cmp(self.links[index_b].name()))
        });
        let mut name_records = Vec::new();
        for (hash, index) in order {
            let id = PreparedHeapId::new(self.layout, &self.ids[index])?;
            btree_v2::LinkIndexKey::NameHash(hash).append_record(&mut name_records, id);
        }
        Ok(name_records)
    }
}

// H5G__dense_create in H5Gdense.c, HDF5 2.2.0, uses seven-byte IDs and 512-byte index nodes.
pub(super) const HEAP_ID_LEN: u16 = 7;
const GROUP_INDEX_NODE_SIZE: u32 = 512;
/// The four-byte name hash and seven-byte heap ID of a type-5 record.
///
/// Defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub(crate) const NAME_RECORD_LEN: u16 = 4 + HEAP_ID_LEN;
// "Version 2 B-trees", format specification version 4.0, defines an eight-byte creation index
// and seven-byte heap ID for type-6 records.
pub(super) const CREATION_RECORD_LEN: u16 = 8 + HEAP_ID_LEN;
