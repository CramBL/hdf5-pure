//! Fixtures for admitting contiguous retired chunks beside an eligible allocation.
//!
//! The helpers check publication and writable recovery against persisted sections and native reads.
//! The benchmark and crosschecks share the same fixture and commit verification.

use std::fs;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FreeSections;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::PhysicalAddress;
use hdf5_pure::SyncPolicy;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use hdf5_pure_format::__private::FormatWidths;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util::bytes;

use crate::file_space;
use crate::file_space_oracle;
use crate::storage;

/// The writer used to create the source file.
#[derive(Clone, Copy, Debug)]
pub enum Origin {
    /// The libhdf5 writer.
    Native,
    /// The [`FileBuilder`] writer.
    Pure,
}

/// The number of contiguous chunks in a flat fixture.
#[derive(Clone, Copy, Debug)]
pub enum Scale {
    EightThousand,
    FourThousand,
    TwoThousand,
}

/// The tracking threshold in bytes for a flat fixture.
#[derive(Clone, Copy, Debug)]
pub enum Threshold {
    /// A 4,096-byte minimum allocation length for independent admission.
    Allocation,
    /// A one-byte minimum allocation length for independent admission.
    One,
}

/// The allocation geometry and tracking policy of a fixture.
#[derive(Clone, Copy, Debug)]
pub enum Chain {
    /// 512-byte chunks with [`FileSpaceStrategy::FsmAggr`].
    Flat { scale: Scale, threshold: Threshold },
    /// 256 whole-page chunks of 4,096 bytes with a 12,288-byte threshold.
    Page,
}

impl Chain {
    /// Returns the number of chunks retired in the edit.
    pub fn chunks(self) -> usize {
        match self {
            Self::Flat { scale, .. } => match scale {
                Scale::EightThousand => 8_000,
                Scale::FourThousand => 4_000,
                Scale::TwoThousand => 2_000,
            },
            Self::Page => 256,
        }
    }

    /// Returns the minimum independently eligible allocation length in bytes.
    pub fn threshold(self) -> u64 {
        match self {
            Self::Flat {
                threshold: Threshold::One,
                ..
            } => 1,
            Self::Flat {
                threshold: Threshold::Allocation,
                ..
            } => 4_096,
            Self::Page => 12_288,
        }
    }

    fn chunk_bytes(self) -> u64 {
        match self {
            Self::Flat { .. } => 512,
            Self::Page => 4_096,
        }
    }

    fn strategy(self) -> FileSpaceStrategy {
        match self {
            Self::Flat { .. } => FileSpaceStrategy::FsmAggr,
            Self::Page => FileSpaceStrategy::Page,
        }
    }

    fn write_native(self, path: &Path) {
        let file = hdf5::File::with_options()
            .with_fapl(|properties| properties.libver_latest().meta_block_size(1 << 20))
            .with_fcpl(|properties| {
                properties.file_space_strategy(NativeStrategy::FreeSpaceManager {
                    paged: matches!(self, Self::Page),
                    persist: true,
                    threshold: self.threshold(),
                })
            })
            .create(path)
            .unwrap();
        let elements = usize::try_from(self.chunk_bytes() / 8).unwrap();
        let dataset = file
            .new_dataset::<u64>()
            .shape(self.chunks() * elements)
            .chunk(elements)
            .create("chunks")
            .unwrap();
        let seed = file
            .new_dataset::<u64>()
            .shape(usize::try_from(self.threshold().max(4_096) / 8).unwrap())
            .create("seed")
            .unwrap();
        let keep = file
            .new_dataset::<u32>()
            .shape(KEEP)
            .create("keep")
            .unwrap();
        keep.write_raw(&vec![77; KEEP]).unwrap();
        dataset
            .write_raw(&vec![42; self.chunks() * elements])
            .unwrap();
        drop(dataset);
        file.flush().unwrap();
        seed.write_raw(&vec![
            0_u64;
            usize::try_from(self.threshold().max(4_096) / 8)
                .unwrap()
        ])
        .unwrap();
        drop(seed);
        drop(keep);
        file.close().unwrap();
    }

    fn write_pure(self, path: &Path) {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(self.strategy())
                .with_persistence(true)
                .with_threshold(self.threshold()),
        );
        if matches!(self, Self::Page) {
            builder.create_dataset("anchor").with_u64_data(&vec![
                0;
                usize::try_from(
                    self.threshold() / 8
                )
                .unwrap()
            ]);
        }
        builder
            .create_dataset("chunks")
            .with_u64_data(&vec![
                42;
                self.chunks()
                    * usize::try_from(self.chunk_bytes() / 8).unwrap()
            ])
            .with_chunks(&[self.chunk_bytes() / 8]);
        builder.create_dataset("keep").with_u32_data(&[77; KEEP]);
        builder.write(path).unwrap();
    }
}

/// A source file with contiguous chunks, an eligible seed, and a surviving dataset.
///
/// Native files have a separate seed allocation after the chunks. Pure flat files have an
/// adjacent chunk-index allocation.
///
/// Pure PAGE files have a separate seed before the chunks. Their complete array blocks precede
/// the seed, fit within one page when smaller than a page, and otherwise start at page boundaries.
pub struct Fixture {
    seed: Option<&'static str>,
    chain: Chain,
    expected_reclaimed: Coverage,
    persisted: Coverage,
    survivor: Coverage,
    initial_bytes: u64,
}

impl Fixture {
    /// Writes a source file and checks its allocation geometry and dataset values.
    ///
    /// The file is closed before returning. Pass the same `path` to [`Self::commit`].
    ///
    /// # Panics
    ///
    /// Panics if a file operation fails, allocation geometry differs from the selected fixture,
    /// or native reads report different dataset values.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::MemoryStrategy;
    /// use test_util_hdf5::file_space::admission::{Chain, Fixture, Origin, Scale, Threshold};
    ///
    /// # let directory = tempfile::tempdir()?;
    /// let path = directory.path().join("admission.h5");
    /// let fixture = Fixture::create(
    ///     &path,
    ///     Origin::Native,
    ///     Chain::Flat { scale: Scale::TwoThousand, threshold: Threshold::Allocation },
    /// );
    /// let result = fixture.commit(&path, MemoryStrategy::Bounded);
    /// assert!(!result.free.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn create(path: &Path, origin: Origin, chain: Chain) -> Self {
        match origin {
            Origin::Native => chain.write_native(path),
            Origin::Pure => chain.write_pure(path),
        }
        let file = File::open_streaming(path).unwrap();
        assert_eq!(file.superblock().base_address.get(), 0);
        let chunks = file.dataset("chunks").unwrap().chunks().unwrap();
        assert_eq!(chunks.len(), chain.chunks());
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.storage_size == chain.chunk_bytes())
        );
        assert!(
            chunks
                .windows(2)
                .all(|pair| pair[0].address + pair[0].storage_size == pair[1].address)
        );
        let first = chunks[0].address;
        let last = chunks.last().unwrap();
        let end = last.address + last.storage_size;
        let mut expected_reclaimed = chunks
            .into_iter()
            .map(|chunk| Interval::new(chunk.address, chunk.storage_size).unwrap())
            .collect::<Vec<_>>();
        let keep = file_space::contiguous_extent(&file, "keep");
        let seed = match origin {
            Origin::Native => {
                let seed = file_space::contiguous_extent(&file, "seed");
                assert_eq!(seed.start, end);
                assert_eq!(seed.end - seed.start, chain.threshold().max(4_096));
                expected_reclaimed.push(Interval::new(seed.start, seed.end - seed.start).unwrap());
                Some("seed")
            }
            Origin::Pure => {
                let image = fs::read(path).unwrap();
                let address = PhysicalAddress::new(
                    u64::try_from(bytes::sole_signature(&image, &FIXED_ARRAY_HEADER_SIGNATURE))
                        .unwrap(),
                );
                let header = FixedArrayHeaderLocator::new(
                    FormatWidths {
                        offsets: file.superblock().offset_size,
                        lengths: file.superblock().length_size,
                    },
                    file.superblock().base_address,
                    file.relative_address(address).unwrap(),
                )
                .parse(
                    FileRegion::new(
                        FileRange::new(
                            PhysicalAddress::new(0),
                            u64::try_from(image.len()).unwrap(),
                        )
                        .unwrap(),
                        &image,
                    )
                    .unwrap(),
                )
                .unwrap();
                let block = header.data_block().unwrap().range();
                assert_eq!(block.start(), header.range().end());
                match chain {
                    Chain::Flat { .. } => {
                        assert_eq!(address.get(), end);
                        assert!(block.length() >= chain.threshold());
                        expected_reclaimed.extend([header.range(), block].map(|range| {
                            Interval::new(range.start().get(), range.length()).unwrap()
                        }));
                        None
                    }
                    Chain::Page => {
                        let seed = file_space::contiguous_extent(&file, "anchor");
                        assert_eq!(seed.end, first);
                        assert_eq!(seed.end - seed.start, chain.threshold());
                        let page = file
                            .file_space_settings()
                            .unwrap()
                            .unwrap()
                            .page_size()
                            .get();
                        for range in [header.range(), block] {
                            assert!(range.end().get() <= seed.start);
                            assert!(
                                range.end().get() <= keep.start || keep.end <= range.start().get()
                            );
                            if range.length() < page {
                                assert_eq!(
                                    range.start().get() / page,
                                    (range.end().get() - 1) / page
                                );
                            } else {
                                assert_eq!(range.start().get() % page, 0);
                            }
                        }
                        expected_reclaimed
                            .push(Interval::new(seed.start, seed.end - seed.start).unwrap());
                        Some("anchor")
                    }
                }
            }
        };
        let persisted = Coverage::new(
            file.persisted_free_sections()
                .unwrap()
                .unwrap()
                .free_sections()
                .sections()
                .iter()
                .map(|section| {
                    Interval::new(
                        file.resolve_address(section.address()).unwrap().get(),
                        section.length().get(),
                    )
                    .unwrap()
                }),
        );
        let survivor = Coverage::new([Interval::new(keep.start, keep.end - keep.start).unwrap()]);
        file.close().unwrap();
        let native = hdf5::File::open(path).unwrap();
        assert_eq!(
            native.dataset("chunks").unwrap().read_raw::<u64>().unwrap(),
            vec![42; chain.chunks() * usize::try_from(chain.chunk_bytes() / 8).unwrap()]
        );
        assert_eq!(
            native.dataset("keep").unwrap().read_raw::<u32>().unwrap(),
            vec![77; KEEP]
        );
        native.close().unwrap();
        Self {
            seed,
            chain,
            expected_reclaimed: Coverage::new(expected_reclaimed),
            persisted,
            survivor,
            initial_bytes: fs::metadata(path).unwrap().len(),
        }
    }

    /// Deletes the chunks and seed, times publication, and checks persisted and recovered space.
    ///
    /// Only [`File::commit`] is timed. Opening, staging deletions, closing, and verification are
    /// excluded. The session uses [`SyncPolicy::OnClose`]. The `path` must name the unchanged
    /// file written by [`Self::create`]. All file handles are released before returning.
    ///
    /// Checks that retired allocations become reusable, are reused for metadata, or are
    /// truncated. Also checks surviving values, persisted sections, cached totals, manager
    /// storage exclusion, and writable recovery, including native section and checksum checks.
    ///
    /// # Panics
    ///
    /// Panics if a file operation fails or a publication, accounting, or recovery check fails.
    pub fn commit(self, path: &Path, memory: MemoryStrategy) -> Observation {
        let file = File::open_rw_with_options(
            path,
            FileAccessProperties::new()
                .with_memory_strategy(memory)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        let before = file.reusable_free_sections().unwrap();
        let before_coverage = Coverage::new(before.sections().iter().map(|section| {
            Interval::new(section.address().get(), section.length().get()).unwrap()
        }));
        assert!(
            self.expected_reclaimed
                .intersection(&before_coverage)
                .is_empty()
        );
        let unclassified = self.persisted.difference(&before_coverage);
        file.root().delete("chunks").unwrap();
        if let Some(seed) = self.seed {
            file.root().delete(seed).unwrap();
        }
        assert_eq!(file.reusable_free_sections().unwrap(), before);
        let start = Instant::now();
        file.commit().unwrap();
        let elapsed = start.elapsed();
        let admitted = file.reusable_free_sections().unwrap();
        assert_eq!(
            file.reusable_space_stats().unwrap().reusable_free_bytes(),
            admitted.free_bytes()
        );
        file.close().unwrap();
        self.inspect(path, memory, elapsed, admitted, unclassified)
    }

    fn inspect(
        self,
        path: &Path,
        memory: MemoryStrategy,
        elapsed: Duration,
        admitted: FreeSections<PhysicalAddress>,
        unclassified: Coverage,
    ) -> Observation {
        let Self {
            seed: _,
            chain,
            expected_reclaimed,
            persisted: _,
            survivor,
            initial_bytes,
        } = self;
        let image = fs::read(path).unwrap();
        let final_bytes = u64::try_from(image.len()).unwrap();
        let reopened = File::open_streaming(path).unwrap();
        assert_eq!(reopened.superblock().eof_address, final_bytes);
        let report = reopened.persisted_free_sections().unwrap().unwrap();
        let sections: Vec<_> = report
            .free_sections()
            .sections()
            .iter()
            .map(|section| {
                (
                    reopened.resolve_address(section.address()).unwrap(),
                    section.length().get(),
                )
            })
            .collect();
        let free = Coverage::new(admitted.sections().iter().map(|section| {
            Interval::new(section.address().get(), section.length().get()).unwrap()
        }));
        let unclassified =
            unclassified.intersection(&Coverage::new([Interval::new(0, final_bytes).unwrap()]));
        assert!(free.intersection(&unclassified).is_empty());
        assert_eq!(
            Coverage::new(
                sections
                    .iter()
                    .map(|&(start, len)| Interval::new(start.get(), len).unwrap())
            ),
            Coverage::new(
                free.intervals()
                    .iter()
                    .copied()
                    .chain(unclassified.intervals().iter().copied())
            )
        );
        assert_eq!(
            report.free_sections().free_bytes(),
            admitted.free_bytes()
                + unclassified
                    .intervals()
                    .iter()
                    .map(|interval| interval.length())
                    .sum::<u64>()
        );
        assert_eq!(reopened.root().datasets().unwrap(), ["keep"]);
        assert_eq!(
            reopened.dataset("keep").unwrap().read_u32().unwrap(),
            vec![77; KEEP]
        );
        assert!(free.intersection(&survivor).is_empty());
        let metadata = storage::file_metadata_extents(&reopened, &image);
        storage::assert_unavailable(&reopened, &metadata);
        let accounted = Coverage::new(
            free.intervals()
                .iter()
                .copied()
                .chain(
                    metadata
                        .iter()
                        .map(|range| Interval::new(range.start().get(), range.length()).unwrap()),
                )
                .chain(
                    (final_bytes < initial_bytes)
                        .then(|| Interval::new(final_bytes, initial_bytes - final_bytes).unwrap()),
                ),
        );
        assert!(
            expected_reclaimed.difference(&accounted).is_empty(),
            "unaccounted retired allocations: {}",
            expected_reclaimed.difference(&accounted)
        );
        let info = reopened.file_space_info().unwrap().unwrap();
        assert_eq!(
            (info.strategy, info.persist, info.threshold),
            (chain.strategy(), true, chain.threshold())
        );
        reopened.close().unwrap();
        let recovered = File::open_rw_with_options(
            path,
            FileAccessProperties::new()
                .with_memory_strategy(memory)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        assert_eq!(recovered.reusable_free_sections().unwrap(), admitted);
        assert_eq!(
            recovered
                .reusable_space_stats()
                .unwrap()
                .reusable_free_bytes(),
            admitted.free_bytes()
        );
        recovered.close().unwrap();
        assert_eq!(fs::read(path).unwrap(), image);
        file_space_oracle::assert_persisted_sections(path);
        assert_eq!(
            file_space_oracle::assert_manager_metadata_bytes(path),
            report.manager_metadata_bytes()
        );
        assert_native(path);
        Observation {
            elapsed,
            initial_bytes,
            final_bytes,
            free,
            sections: report.free_sections().section_count(),
            manager_bytes: report.manager_metadata_bytes(),
        }
    }
}

/// The timing and verified file-space measurements from a committed fixture.
pub struct Observation {
    /// Elapsed time for [`File::commit`], excluding the subsequent close.
    pub elapsed: Duration,
    /// Physical file length before staging deletions.
    pub initial_bytes: u64,
    /// Physical file length after committing and closing.
    pub final_bytes: u64,
    /// Reusable physical byte ranges after publication, also checked on writable reopen.
    pub free: Coverage,
    /// Number of persisted free sections, including sections unavailable for reuse.
    pub sections: usize,
    /// Total storage allocated to persisted free-space managers.
    pub manager_bytes: u64,
}

/// Checks that libhdf5 reads only the surviving dataset and its expected values.
///
/// # Panics
///
/// Panics if opening, reading, or closing fails, or names or values differ.
pub fn assert_native(path: &Path) {
    let file = hdf5::File::open(path).unwrap();
    assert_eq!(file.member_names().unwrap(), ["keep"]);
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<u32>().unwrap(),
        vec![77; KEEP]
    );
    file.close().unwrap();
}

/// The signature of a version 0 fixed array header, defined in "The Fixed Array Index" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr
const FIXED_ARRAY_HEADER_SIGNATURE: [u8; 4] = *b"FAHD";
const KEEP: usize = 4_096;
