use hdf5_pure_core::__private::FileRange;
use test_util::disk_log::DiskOp;

use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::source::{BytesSource, zero_base_source};
use crate::{File, PhysicalAddress};

/// Reachable data and super block allocations in an Extensible Array fixture.
#[derive(Debug, Default)]
pub(crate) struct BlockAllocations {
    pub(crate) data: Vec<FileRange>,
    pub(crate) super_blocks: Vec<FileRange>,
}

impl BlockAllocations {
    pub(crate) fn read(bytes: &[u8], dataset: &str) -> Self {
        let located = crate::superblock::read(bytes).unwrap();
        assert_eq!(located.superblock.base_address.get(), 0);
        let file = File::from_bytes(bytes.to_vec()).unwrap();
        let DataLayout::Chunked {
            index: ChunkIndexLayout::ExtensibleArray { address, .. },
            ..
        } = file.dataset(dataset).unwrap().data_layout().unwrap()
        else {
            panic!("expected an Extensible Array fixture");
        };
        let mut allocations = Self::default();
        let Some(address) = address else {
            return allocations;
        };
        let source = BytesSource::new(bytes);
        let spans =
            super::index_spans(&zero_base_source(&source), located.widths, address).unwrap();
        for span in spans {
            let start = span.start().get();
            let range = FileRange::new(PhysicalAddress::new(start), span.length()).unwrap();
            let at = usize::try_from(start).unwrap();
            match &bytes[at..at + 4] {
                b"EADB" => allocations.data.push(range),
                b"EASB" => allocations.super_blocks.push(range),
                b"EAHD" | b"EAIB" => {}
                signature => {
                    panic!("unexpected Extensible Array signature {signature:?} at {start}")
                }
            }
        }
        allocations
    }

    pub(crate) fn new_since(mut self, before: &Self) -> Self {
        assert!(before.data.iter().all(|range| self.data.contains(range)));
        assert!(
            before
                .super_blocks
                .iter()
                .all(|range| self.super_blocks.contains(range))
        );
        self.data.retain(|range| !before.data.contains(range));
        self.super_blocks
            .retain(|range| !before.super_blocks.contains(range));
        self
    }

    pub(crate) fn assert_issued(&self, operations: &[DiskOp]) {
        for (ranges, signature) in [(&self.data, b"EADB"), (&self.super_blocks, b"EASB")] {
            for range in ranges {
                assert!(
                    operations.iter().any(|operation| {
                        let DiskOp::Write { offset, bytes } = operation else {
                            return false;
                        };
                        let Some(local) = range
                            .start()
                            .get()
                            .checked_sub(*offset)
                            .and_then(|local| usize::try_from(local).ok())
                        else {
                            return false;
                        };
                        bytes
                            .get(local..)
                            .is_some_and(|contents| contents.starts_with(signature))
                    }),
                    "no completed write initialized {range:?} with {signature:?}"
                );
            }
        }
    }
}
