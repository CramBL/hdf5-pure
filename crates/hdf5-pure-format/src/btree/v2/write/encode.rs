use alloc::vec::Vec;

use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

use super::placement::BTreeV2Placement;
use super::plan::{BTreeV2Plan, CheckedRecords};
use super::prepared::{self, CheckedNodeEncoding};

/// The split percentage of the header, 100 in `H5A_NAME_BT2_SPLIT_PERC` and
/// `H5HF_HUGE_BT2_SPLIT_PERC` (HDF5 2.2.0).
const SPLIT_PERCENT: u8 = 100;

/// The merge percentage of the header, 40 in `H5A_NAME_BT2_MERGE_PERC` and
/// `H5HF_HUGE_BT2_MERGE_PERC` (HDF5 2.2.0).
const MERGE_PERCENT: u8 = 40;

/// A serialized version 2 B-tree: the header, and the nodes it points to.
pub struct BTreeV2Image {
    /// The header, signature `BTHD`.
    pub header: Vec<u8>,
    /// The nodes back to back, each `node_size` bytes, with the root last.
    pub nodes: Vec<u8>,
}

impl BTreeV2Plan {
    /// Serializes the tree, with its nodes from `nodes_address` on.
    ///
    /// `records` holds the records back to back in their sort order, `record_count * record_size`
    /// bytes. For a tree with nodes, the header points to the root, the last node. For an empty
    /// tree planned with [`Self::new_with_undefined_empty_root`], the root address is undefined and
    /// `nodes_address` is unused.
    ///
    /// Checks the record buffer, node geometry, counts, and address ranges before encoding the
    /// header or nodes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `offset_width` differs from the plan's address width,
    /// the record buffer length is incorrect, or the plan's geometry or counts are inconsistent.
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the total node size or exclusive end exceeds
    /// [`u64::MAX`]. For a tree with nodes, this check precedes the address-width checks.
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if the nodes start at the undefined
    /// address, or [`FormatError::AddressTooLarge`] if the starting address or exclusive end
    /// exceeds `offset_width`. The exclusive end may equal the undefined address value.
    ///
    /// Returns [`FormatError::LengthTooLarge`] if the record count exceeds `length_width`, and
    /// [`FormatError::ValueTooLargeForPlatform`] if the total node size exceeds [`usize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FormatError, LengthWidth, OffsetWidth, StoredAddress};
    /// use hdf5_pure_format::__private::{
    ///     BTreeV2Plan, BTREE_V2_ATTRIBUTE_NAME, BTREE_V2_NODE_SIZE,
    /// };
    ///
    /// let plan = BTreeV2Plan::new(
    ///     BTREE_V2_ATTRIBUTE_NAME, 2, 17, BTREE_V2_NODE_SIZE, OffsetWidth::Eight,
    /// ).unwrap();
    /// let records = [0; 34];
    /// let image = plan.serialize(
    ///     &records, StoredAddress::new(96), OffsetWidth::Eight, LengthWidth::Eight,
    /// )?;
    /// assert_eq!(image.nodes.len(), usize::try_from(BTREE_V2_NODE_SIZE).unwrap());
    /// let error = plan.serialize(
    ///     &records, StoredAddress::new(u64::MAX), OffsetWidth::Eight, LengthWidth::Eight,
    /// ).err().unwrap();
    /// assert_eq!(error, FormatError::OffsetOverflow {
    ///     offset: u64::MAX,
    ///     length: u64::from(BTREE_V2_NODE_SIZE),
    /// });
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn serialize(
        &self,
        records: &[u8],
        nodes_address: StoredAddress,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<BTreeV2Image, FormatError> {
        if offset_width != self.offset_width() {
            return Err(FormatError::Internal(
                "B-tree serialization address width differs from the plan".into(),
            ));
        }
        let length = self
            .checked_nodes_size()
            .ok_or(FormatError::OffsetOverflow {
                offset: nodes_address.get(),
                length: u64::MAX,
            })?;
        if !self.nodes().is_empty() {
            nodes_address.checked_add(length)?;
            self.block_end(nodes_address, length)?;
        }
        let records = self.prepare_records(records)?;
        let encoding = CheckedNodeEncoding::new(self)?;
        let emission = CheckedNodeEmission {
            encoding: &encoding,
            records,
            addresses: CheckedNodeAddresses::Contiguous {
                start: nodes_address,
                node_size: self.node_size(),
            },
        };
        let root_address = self
            .nodes()
            .len()
            .checked_sub(1)
            .map(|index| emission.addresses.address(index))
            .transpose()?;
        let header = CheckedHeaderEncoding::new(&encoding, root_address, length_width)?;
        let mut nodes = Vec::with_capacity(length.to_usize()?);
        let header = header.encode()?;
        for index in 0..self.nodes().len() {
            emission.encode_node(&mut nodes, index)?;
        }
        Ok(BTreeV2Image { header, nodes })
    }
}

impl BTreeV2Placement<'_> {
    /// Serializes the header with the placed root address.
    ///
    /// For an empty tree planned with [`BTreeV2Plan::new_with_undefined_empty_root`], the root
    /// address is undefined.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if the planned record count exceeds `length_width`.
    pub fn serialize_header(&self, length_width: LengthWidth) -> Result<Vec<u8>, FormatError> {
        CheckedHeaderEncoding::new(
            self.encoding(),
            self.addresses().last().copied(),
            length_width,
        )?
        .encode()
    }

    /// Serializes one complete node per iterator step, in placement order.
    ///
    /// Each item contains the node's stored address and encoded bytes, including its checksum
    /// and trailing zero padding. The order is the same as [`Self::blocks`].
    ///
    /// Pass the records back to back in their sort order, with the count and size used to build
    /// the plan. The record buffer length is checked before the iterator is returned. See
    /// [`BTreeV2Plan::place`] for an example.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the supplied byte length differs from the planned
    /// length, or the planned byte length exceeds [`u64::MAX`].
    pub fn serialize_nodes<'a>(
        &'a self,
        records: &'a [u8],
    ) -> Result<
        impl ExactSizeIterator<Item = Result<(StoredAddress, Vec<u8>), FormatError>> + 'a,
        FormatError,
    > {
        let records = self.encoding().plan().prepare_records(records)?;
        let emission = CheckedNodeEmission {
            encoding: self.encoding(),
            records,
            addresses: CheckedNodeAddresses::Placed(self.addresses()),
        };
        Ok(self
            .addresses()
            .iter()
            .copied()
            .enumerate()
            .map(move |(index, address)| {
                let mut data = Vec::with_capacity(self.encoding().plan().node_size().to_usize()?);
                emission.encode_node(&mut data, index)?;
                Ok((address, data))
            }))
    }
}

/// Node-address lookup for contiguous or independently placed allocations.
enum CheckedNodeAddresses<'a> {
    Contiguous {
        start: StoredAddress,
        node_size: u32,
    },
    Placed(&'a [StoredAddress]),
}

impl CheckedNodeAddresses<'_> {
    fn address(&self, index: usize) -> Result<StoredAddress, FormatError> {
        match self {
            Self::Contiguous { start, node_size } => {
                let offset = index.to_u64().checked_mul(u64::from(*node_size)).ok_or(
                    FormatError::OffsetOverflow {
                        offset: start.get(),
                        length: u64::MAX,
                    },
                )?;
                start.checked_add(offset)
            }
            Self::Placed(addresses) => Ok(addresses[index]),
        }
    }
}

/// Checked node geometry and record bytes paired with the selected address layout.
///
/// The internal and leaf node layouts are defined in "Version 2 B-trees" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
struct CheckedNodeEmission<'a, 'b> {
    encoding: &'a CheckedNodeEncoding<'a>,
    records: CheckedRecords<'b>,
    addresses: CheckedNodeAddresses<'a>,
}

impl CheckedNodeEmission<'_, '_> {
    fn encode_node(&self, data: &mut Vec<u8>, index: usize) -> Result<(), FormatError> {
        let plan = self.encoding.plan();
        let node = &plan.nodes()[index];
        let start = data.len();
        let record_size = usize::from(plan.record_size().get());
        let node_size = plan.node_size().to_usize()?;
        data.extend_from_slice(if node.depth() == 0 { b"BTLF" } else { b"BTIN" });
        data.push(0);
        data.push(plan.tree_type());
        for &record in node.records() {
            data.extend_from_slice(
                &self.records.bytes()[record * record_size..(record + 1) * record_size],
            );
        }
        for &child in node.children() {
            bytes::try_write_decoded_address(
                data,
                DecodedStoredAddress::Defined(self.addresses.address(child)?),
                plan.offset_width(),
            )?;
            write_uint(
                data,
                plan.nodes()[child].records().len() as u64,
                plan.info().max_nrec_size(),
            );
            write_uint(
                data,
                self.encoding.subtree_total(child),
                plan.info().total_nrec_size(node.depth()),
            );
        }
        let checksum = crate::checksum::jenkins_lookup3(&data[start..]);
        data.extend_from_slice(&checksum.to_le_bytes());
        debug_assert!(data.len() - start <= node_size);
        data.resize(start + node_size, 0);
        Ok(())
    }
}

/// Header fields checked for their selected offset and length widths.
///
/// A defined root address fits the offset width and differs from its undefined-address value.
/// The root record count and total record count fit their header fields.
///
/// The header layout is defined in "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
struct CheckedHeaderEncoding<'a> {
    plan: &'a BTreeV2Plan,
    root_address: DecodedStoredAddress,
    root_records: u16,
    length_width: LengthWidth,
}

impl<'a> CheckedHeaderEncoding<'a> {
    fn new(
        nodes: &CheckedNodeEncoding<'a>,
        root_address: Option<StoredAddress>,
        length_width: LengthWidth,
    ) -> Result<Self, FormatError> {
        let plan = nodes.plan();
        if !width::length_width_holds(length_width, plan.total_records()) {
            return Err(FormatError::LengthTooLarge {
                length: plan.total_records(),
                length_size: length_width.get(),
            });
        }
        let root_address = if let Some(address) = root_address {
            plan.block_end(address, 0)?;
            DecodedStoredAddress::Defined(address)
        } else {
            DecodedStoredAddress::Undefined
        };
        Ok(Self {
            plan,
            root_address,
            root_records: nodes.root_records(),
            length_width,
        })
    }

    fn encode(&self) -> Result<Vec<u8>, FormatError> {
        let plan = self.plan;
        let mut header = Vec::with_capacity(crate::btree::v2::btree_v2_header_size(
            plan.offset_width(),
            self.length_width,
        ));
        header.extend_from_slice(b"BTHD");
        header.push(0);
        header.push(plan.tree_type());
        header.extend_from_slice(&plan.node_size().to_le_bytes());
        header.extend_from_slice(&plan.record_size().get().to_le_bytes());
        header.extend_from_slice(&plan.depth().to_le_bytes());
        header.push(SPLIT_PERCENT);
        header.push(MERGE_PERCENT);
        bytes::try_write_decoded_address(&mut header, self.root_address, plan.offset_width())?;
        header.extend_from_slice(&self.root_records.to_le_bytes());
        bytes::try_write_length(&mut header, plan.total_records(), self.length_width)?;
        let checksum = crate::checksum::jenkins_lookup3(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        Ok(header)
    }
}

/// Appends `value` to `buf` as a little-endian integer of `width` bytes.
///
/// A width of 0 appends no bytes, for a field the child pointers of a node do not have.
fn write_uint(buf: &mut Vec<u8>, value: u64, width: usize) {
    debug_assert!(width == 0 || prepared::count_fits(value, width));
    for i in 0..width {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "masked to one byte by the shift-and-truncate"
        )]
        buf.push((value >> (i * 8)) as u8);
    }
}

#[cfg(test)]
#[path = "tests/encode.rs"]
mod tests;
