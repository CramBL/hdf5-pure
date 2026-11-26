use alloc::vec::Vec;

use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::block::{CHECKSUM_LEN, DIRECT_SIGNATURE, INDIRECT_SIGNATURE, VERSION};
use crate::width::OffsetWidth;

/// The heap header address, managed offset, and field widths for one block.
pub(super) struct BlockPrefix {
    pub(super) owner: StoredAddress,
    pub(super) offset_width: OffsetWidth,
    pub(super) heap_offset: u64,
    pub(super) block_offset_bytes: usize,
}

impl BlockPrefix {
    pub(super) fn encode_indirect(
        &self,
        size: u64,
        children: impl Iterator<Item = Result<Option<StoredAddress>, FormatError>>,
    ) -> Result<Vec<u8>, FormatError> {
        let mut data = self.encode(size, &INDIRECT_SIGNATURE)?;
        for child in children {
            bytes::try_write_decoded_address(
                &mut data,
                child?.map_or(
                    DecodedStoredAddress::Undefined,
                    DecodedStoredAddress::Defined,
                ),
                self.offset_width,
            )?;
        }
        let checksum = crate::checksum::jenkins_lookup3(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(data.len() as u64, size);
        Ok(data)
    }

    pub(super) fn encode_direct(
        &self,
        size: u64,
        header_size: u64,
        append_objects: impl FnOnce(&mut Vec<u8>),
    ) -> Result<Vec<u8>, FormatError> {
        let mut data = self.encode(size, &DIRECT_SIGNATURE)?;
        let checksum_at = data.len();
        data.extend_from_slice(&[0; CHECKSUM_LEN]);
        debug_assert_eq!(data.len() as u64, header_size);
        append_objects(&mut data);
        data.resize(size.to_usize()?, 0);
        let checksum = crate::checksum::jenkins_lookup3(&data);
        data[checksum_at..checksum_at + CHECKSUM_LEN].copy_from_slice(&checksum.to_le_bytes());
        Ok(data)
    }

    fn encode(&self, size: u64, signature: &[u8; 4]) -> Result<Vec<u8>, FormatError> {
        let mut data = Vec::with_capacity(size.to_usize()?);
        data.extend_from_slice(signature);
        data.push(VERSION);
        bytes::try_write_decoded_address(
            &mut data,
            DecodedStoredAddress::Defined(self.owner),
            self.offset_width,
        )?;
        let encoded = self.heap_offset.to_le_bytes();
        data.extend_from_slice(
            encoded
                .get(..self.block_offset_bytes)
                .ok_or(FormatError::InvalidFractalHeapGeometry)?,
        );
        Ok(data)
    }
}
