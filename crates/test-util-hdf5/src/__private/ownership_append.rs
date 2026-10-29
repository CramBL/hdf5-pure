//! Native fixtures and observations for append allocation from persisted free space.

use std::path::Path;

use hdf5::Extent;
use hdf5::datatype::ByteOrder as NativeByteOrder;
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5_pure::{File, FreeSpaceManagerRole};

use crate::dataset;
use crate::file_space_oracle;
use crate::numeric::ByteOrder;

use super::array_index;

/// Stores the initial values and reusable extent of a native append fixture.
pub struct Fixture {
    initial: Vec<u8>,
    hole: (u64, u64),
    incompressible: bool,
}

impl Fixture {
    /// Writes dataset `d` with a reusable extent below a retained dataset.
    ///
    /// If `paged` is true, libhdf5 uses paged free-space management. If `filtered` is true,
    /// libhdf5 applies deflate. Initial values are zeros unless `incompressible` is true.
    /// Pass `paged = true` when both `filtered` and `incompressible` are true.
    ///
    /// # Panics
    ///
    /// Panics if native fixture creation or its allocation and datatype checks fail.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::File;
    /// use test_util_hdf5::__private::ownership_append::{self, Fixture};
    ///
    /// # let _guard = test_util_hdf5::lock::libhdf5_guard();
    /// # let directory = tempfile::tempdir()?;
    /// let path = directory.path().join("ownership.h5");
    /// let fixture = Fixture::write(&path, true, false, false);
    /// let file = File::open_rw(&path)?;
    /// let mut dataset = file.dataset("d")?;
    /// dataset.append(&ownership_append::payload(0x3cb6_7710))?;
    /// drop(dataset);
    /// file.close()?;
    /// fixture.assert_contents(&path, 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write(path: &Path, paged: bool, filtered: bool, incompressible: bool) -> Self {
        let native = hdf5::File::with_options()
            .with_fapl(|properties| properties.libver_v110())
            .with_fcpl(|properties| {
                properties.file_space_strategy(NativeStrategy::FreeSpaceManager {
                    paged,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(path)
            .unwrap();
        let initial = if incompressible {
            payload(0xc603_8f14)
        } else {
            vec![0; 8192]
        };
        let builder = native
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.u8_datatype())
            .shape([Extent::resizable(initial.len())])
            .chunk([initial.len()]);
        let builder = if filtered {
            builder.deflate(6)
        } else {
            builder
        };
        let target = builder.create("d").unwrap();
        drop(builder);
        assert_eq!(
            target.dtype().unwrap().byte_order(),
            NativeByteOrder::LittleEndian
        );
        target.write_raw(&initial).unwrap();
        drop(target);
        let scratch = native
            .new_dataset_builder()
            .with_data(&vec![7_u8; 65_536])
            .create("scratch")
            .unwrap();
        let hole = (scratch.offset().unwrap(), scratch.storage_size());
        drop(scratch);
        let ceiling = native
            .new_dataset_builder()
            .with_data(&vec![9_u8; 8192])
            .create("ceiling")
            .unwrap();
        assert!(ceiling.offset().unwrap() >= hole.0 + hole.1);
        drop(ceiling);
        native.unlink("scratch").unwrap();
        native.close().unwrap();
        file_space_oracle::assert_persisted_sections(path);
        if incompressible && filtered {
            let original = std::fs::read(path).unwrap();
            let pure = File::open(path).unwrap();
            let page = pure.file_space_info().unwrap().unwrap().page_size.get();
            let report = pure.persisted_free_sections().unwrap().unwrap();
            assert!(report.managers().iter().any(|manager| {
                manager.role() == FreeSpaceManagerRole::PageLargeGeneric
                    && manager.free_sections().sections().iter().any(|section| {
                        !section.address().is_multiple_of(page)
                            && section.address().next_multiple_of(page) <= hole.0
                            && hole.0 + hole.1 <= section.end_address() / page * page
                    })
            }));
            assert!(array_index::libhdf5_chunks(path, 0)[0].size() > initial.len() as u64);
            drop(pure);
            assert_eq!(std::fs::read(path).unwrap(), original);
        }
        Self {
            initial,
            hole,
            incompressible,
        }
    }

    /// Asserts appended values, chunk records, and reuse of the fixture's free extent.
    ///
    /// Pass a completed file or an issued-byte snapshot after appending [`payload`] with seeds
    /// `0x3cb6_7710` through `0x3cb6_7710 + cycle`, inclusive. The checks include the retained
    /// dataset, persisted free sections, manager allocation bytes, and unchanged file bytes.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be read through either library or an observation differs from
    /// the expected values, allocation, or metadata.
    pub fn assert_contents(&self, path: &Path, cycle: u32) {
        let Self {
            ref initial,
            hole,
            incompressible,
        } = *self;
        let before = std::fs::read(path).unwrap();
        file_space_oracle::assert_persisted_sections(path);
        file_space_oracle::assert_manager_metadata_bytes(path);
        let native = hdf5::File::open(path).unwrap();
        let mut expected = initial.clone();
        for completed in 0..=cycle {
            expected.extend(payload(0x3cb6_7710 + completed));
        }
        assert_eq!(
            native.dataset("d").unwrap().read_raw::<u8>().unwrap(),
            expected
        );
        assert_eq!(
            native.dataset("ceiling").unwrap().read_raw::<u8>().unwrap(),
            [9; 8192]
        );
        native.close().unwrap();
        let pure = File::open(path).unwrap();
        assert_eq!(pure.dataset("d").unwrap().read_u8().unwrap(), expected);
        let chunks = array_index::libhdf5_chunks(path, 0);
        assert_eq!(array_index::pure_chunks(&pure), chunks);
        let latest = chunks.last().unwrap();
        assert!(
            latest.address() < hole.0 + hole.1 && hole.0 < latest.address() + latest.size(),
            "cycle {cycle}: chunk {latest:?} did not reuse the native hole {hole:?}"
        );
        if incompressible {
            assert!(
                hole.0 <= latest.address() && latest.address() + latest.size() <= hole.0 + hole.1,
                "cycle {cycle}: chunk {latest:?}, hole {hole:?}"
            );
            if cycle == 0 {
                assert_eq!(latest.address(), hole.0, "chunk {latest:?}, hole {hole:?}");
            }
        }
        let free = pure.persisted_free_sections().unwrap().unwrap();
        for chunk in chunks {
            assert!(
                free.free_sections()
                    .sections()
                    .iter()
                    .all(
                        |section| chunk.address() + chunk.size() <= section.address()
                            || section.end_address() <= chunk.address()
                    )
            );
        }
        drop(pure);
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
}

/// Returns 8192 deterministic bytes for one append, generated from `seed`.
///
/// Each generated integer is serialized in little-endian order.
pub fn payload(seed: u32) -> Vec<u8> {
    dataset::incompressible(seed, 2048)
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect()
}
