//! Workloads and observations for comparisons of file address and length widths.

use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::CharacterSet;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Dataset;
use hdf5_pure::Datatype;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FilterId;
use hdf5_pure::FixedPointType;
use hdf5_pure::FormatError;
use hdf5_pure::Layout;
use hdf5_pure::LengthWidth;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::MetadataCacheConfig;
use hdf5_pure::MetadataCacheStats;
use hdf5_pure::OffsetWidth;
use hdf5_pure::PersistedFreeSpacePolicy;
use hdf5_pure::StringPadding;
use hdf5_pure::SyncPolicy;
use hdf5_pure::VariableLengthType;

use crate::file_space::persisted::PersistedStrategy;

/// Configuration of a file-width comparison workload.
///
/// The fixture contains dense group links and attributes, contiguous and chunked numeric
/// datasets, variable-length strings, and scratch datasets for reclamation. Callers select
/// the widths, series index, persistent space strategy, and series filtering independently.
/// An Extensible Array series requires eight-byte lengths.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{LengthWidth, OffsetWidth};
/// use test_util_hdf5::__private::width_workload::{Case, Index, Phase};
/// use test_util_hdf5::file_space::persisted::PersistedStrategy;
///
/// let directory = tempfile::tempdir()?;
/// let path = directory.path().join("width-workload.h5");
/// let case = Case::new(
///     OffsetWidth::Two,
///     LengthWidth::Eight,
///     Index::Finite,
///     PersistedStrategy::FsmAggr,
///     false,
/// )
/// .unwrap();
/// case.create(&path)?;
/// let file = case.open(&path, 65_536)?;
/// case.append(&file)?;
/// file.commit()?;
/// case.stage_reclamation(&file)?;
/// file.commit()?;
/// case.stage_reuse(&file)?;
/// file.commit()?;
/// file.close()?;
///
/// let file = case.open_read(&path, 65_536)?;
/// case.verify(&file, Phase::Reuse);
/// let observed = case.observe(&file)?;
/// assert!(observed.dataset_bytes > 0);
/// assert!(observed.manager_bytes > 0);
/// file.close()?;
/// directory.close()?;
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Case {
    offsets: OffsetWidth,
    lengths: LengthWidth,
    index: Index,
    strategy: PersistedStrategy,
    filtered: bool,
}

impl Case {
    /// Creates a workload configuration for the selected widths and storage settings.
    ///
    /// # Errors
    ///
    /// Returns [`Unsupported`] if `index` is [`Index::Extensible`] and `lengths` is narrower
    /// than eight bytes.
    pub fn new(
        offsets: OffsetWidth,
        lengths: LengthWidth,
        index: Index,
        strategy: PersistedStrategy,
        filtered: bool,
    ) -> Result<Self, Unsupported> {
        if index == Index::Extensible && lengths != LengthWidth::Eight {
            return Err(Unsupported { lengths });
        }
        Ok(Self {
            offsets,
            lengths,
            index,
            strategy,
            filtered,
        })
    }

    /// Writes the initial fixture to `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if fixture creation or writing fails.
    pub fn create(&self, path: &Path) -> Result<(), Error> {
        let Self {
            offsets,
            lengths,
            index: _,
            strategy: _,
            filtered,
        } = *self;
        let mut builder = FileBuilder::new();
        builder
            .with_offset_width(offsets)
            .with_length_width(lengths)
            .with_file_space_settings(self.settings());
        builder.create_group_with("metadata", |group| {
            for value in 0_u32..9 {
                group.set_attr(&format!("attribute_{value}"), AttrValue::U32(value));
                group
                    .create_dataset(&format!("member_{value}"))
                    .with_u32_data(&[value; 3]);
            }
            Ok(())
        })?;
        builder.create_group_with("data", |group| {
            let series = group.create_dataset("series");
            series
                .with_i32_data(&(0..8).collect::<Vec<_>>())
                .with_chunks(&[4])
                .with_maxshape(&[self.maximum()]);
            if filtered {
                series.with_deflate(DeflateLevel::Six);
            }
            group
                .create_dataset("single")
                .with_i32_data(&[1])
                .with_chunks(&[4])
                .with_maxshape(&[MaxExtent::Fixed(4)]);
            group.create_dataset("contiguous").with_i32_data(&[13; 32]);
            group
                .create_dataset("strings")
                .with_vlen_strings(&["alpha", "", "ζ"]);
            Ok(())
        })?;
        builder.create_group_with("holes", |group| {
            for index in 0..4 {
                group
                    .create_dataset(&format!("scratch_{index}"))
                    .with_u8_data(&[17; 128]);
                group
                    .create_dataset(&format!("spacer_{index}"))
                    .with_u8_data(&[31; 64]);
            }
            Ok(())
        })?;
        builder.write(path)
    }

    /// Appends to the series and single-chunk datasets in place.
    ///
    /// The caller commits and closes the file before inspecting persisted managers through
    /// [`Self::observe`].
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if a dataset lookup or either append fails.
    pub fn append(&self, file: &File) -> Result<(), Error> {
        file.dataset("data/series")?.append(&[8, 9, 10, 11])?;
        file.dataset("data/single")?.append(&[2])?;
        Ok(())
    }

    /// Stages scratch-dataset deletion and a metadata attribute replacement.
    ///
    /// The caller publishes the edits with [`File::commit`].
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if a group lookup or staging an edit fails.
    pub fn stage_reclamation(&self, file: &File) -> Result<(), Error> {
        let group = file.group("holes")?;
        for index in 0..4 {
            group.delete(&format!("scratch_{index}"))?;
        }
        file.group("metadata")?
            .set_attr("attribute_0", AttrValue::U32(37))?;
        Ok(())
    }

    /// Stages a replacement dataset in the fixture's holes group.
    ///
    /// The caller publishes the edit with [`File::commit`].
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the group lookup or staging the dataset fails.
    pub fn stage_reuse(&self, file: &File) -> Result<(), Error> {
        file.group("holes")?
            .create_dataset("replacement", |builder| {
                builder.with_u8_data(&[43; 128]);
            })?;
        Ok(())
    }

    /// Returns the selected file address width.
    pub const fn offsets(&self) -> OffsetWidth {
        self.offsets
    }

    /// Returns the selected file length width.
    pub const fn lengths(&self) -> LengthWidth {
        self.lengths
    }

    /// Returns the series dataset's index configuration.
    pub const fn index(&self) -> Index {
        self.index
    }

    /// Returns whether the series dataset uses deflate filtering.
    pub const fn filtered(&self) -> bool {
        self.filtered
    }

    /// Returns persistent file-space settings with a threshold of one byte.
    ///
    /// The PAGE size is [`FileSpacePageSize::MIN`].
    pub fn settings(&self) -> FileSpaceSettings {
        FileSpaceSettings::new(match self.strategy {
            PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
            PersistedStrategy::Page => FileSpaceStrategy::Page,
        })
        .with_persistence(true)
        .with_threshold(1)
        .with_page_size(FileSpacePageSize::MIN)
    }

    /// Returns access settings for bounded editing and metadata reads.
    ///
    /// Persisted-space recovery is strict, synchronization uses [`SyncPolicy::OnClose`], and
    /// the metadata-read-cache budget is `cache_bytes`.
    pub fn access(&self, cache_bytes: usize) -> FileAccessProperties {
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose)
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
            .with_metadata_cache(MetadataCacheConfig::new(cache_bytes))
    }

    /// Opens the fixture for bounded editing with a metadata-cache budget of `cache_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if opening the file fails.
    pub fn open(&self, path: &Path, cache_bytes: usize) -> Result<File, Error> {
        File::open_rw_with_options(path, self.access(cache_bytes))
    }

    /// Opens the fixture for streaming reads with a metadata-cache budget of `cache_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if opening the file fails.
    pub fn open_read(&self, path: &Path, cache_bytes: usize) -> Result<File, Error> {
        File::open_streaming_with_options(path, self.access(cache_bytes))
    }

    /// Checks the fixture's contents, schema, widths, and creation settings for `phase`.
    ///
    /// # Panics
    ///
    /// Panics if reading the fixture fails or an observation differs from the expected state.
    pub fn verify(&self, file: &File, phase: Phase) {
        let Self {
            offsets,
            lengths,
            index,
            strategy: _,
            filtered,
        } = *self;
        assert_eq!(file.superblock().offset_size, offsets);
        assert_eq!(file.superblock().length_size, lengths);
        assert_eq!(file.file_space_settings().unwrap(), Some(self.settings()));
        let mut groups = file.root().groups().unwrap();
        groups.sort();
        assert_eq!(groups, ["data", "holes", "metadata"]);
        assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
        assert!(file.root().attrs().unwrap().is_empty());
        let metadata = file.group("metadata").unwrap();
        assert!(metadata.groups().unwrap().is_empty());
        let mut members = metadata.datasets().unwrap();
        members.sort();
        assert_eq!(
            members,
            (0..9)
                .map(|index| format!("member_{index}"))
                .collect::<Vec<_>>()
        );
        let mut attributes = metadata.attrs().unwrap();
        for value in 0_u32..9 {
            assert_schema(
                &metadata.dataset(&format!("member_{value}")).unwrap(),
                Datatype::FixedPoint(FixedPointType::u32()),
                3,
            );
            assert_eq!(
                attributes.remove(&format!("attribute_{value}")),
                Some(AttrValue::U32(if value == 0 && phase.reclaimed() {
                    37
                } else {
                    value
                }))
            );
            assert_eq!(
                file.dataset(&format!("metadata/member_{value}"))
                    .unwrap()
                    .read_u32()
                    .unwrap(),
                [value; 3]
            );
            assert!(
                metadata
                    .dataset(&format!("member_{value}"))
                    .unwrap()
                    .attrs()
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(attributes.is_empty(), "{attributes:?}");
        let data = file.group("data").unwrap();
        assert!(data.groups().unwrap().is_empty());
        assert!(data.attrs().unwrap().is_empty());
        let mut members = data.datasets().unwrap();
        members.sort();
        assert_eq!(members, ["contiguous", "series", "single", "strings"]);
        for link_name in &members {
            assert!(data.dataset(link_name).unwrap().attrs().unwrap().is_empty());
        }
        let series = file.dataset("data/series").unwrap();
        assert_schema(
            &series,
            Datatype::FixedPoint(FixedPointType::i32()),
            if phase == Phase::Create { 8 } else { 12 },
        );
        assert_eq!(
            series.read_i32().unwrap(),
            (0..if phase == Phase::Create { 8 } else { 12 }).collect::<Vec<_>>()
        );
        assert_eq!(series.maxshape().unwrap(), Some(vec![self.maximum()]));
        assert_eq!(
            series.filters().unwrap(),
            if filtered {
                vec![FilterId::Deflate]
            } else {
                Vec::new()
            }
        );
        assert_eq!(
            series.layout().unwrap(),
            Layout::Chunked {
                chunk_shape: vec![4],
                index: match index {
                    Index::Extensible => ChunkIndex::ExtensibleArray,
                    Index::Finite => ChunkIndex::FixedArray,
                },
            }
        );
        let single = file.dataset("data/single").unwrap();
        assert_schema(
            &single,
            Datatype::FixedPoint(FixedPointType::i32()),
            if phase == Phase::Create { 1 } else { 2 },
        );
        assert_eq!(single.maxshape().unwrap(), Some(vec![MaxExtent::Fixed(4)]));
        assert!(single.filters().unwrap().is_empty());
        assert_eq!(
            single.read_i32().unwrap(),
            if phase == Phase::Create {
                vec![1]
            } else {
                vec![1, 2]
            }
        );
        assert_eq!(
            single.layout().unwrap(),
            Layout::Chunked {
                chunk_shape: vec![4],
                index: ChunkIndex::SingleChunk,
            }
        );
        let contiguous = file.dataset("data/contiguous").unwrap();
        assert_schema(&contiguous, Datatype::FixedPoint(FixedPointType::i32()), 32);
        assert_eq!(
            contiguous.read_i32().unwrap(),
            [13; 32],
            "width workload contiguous values differ"
        );
        let strings = file.dataset("data/strings").unwrap();
        assert_schema(
            &strings,
            Datatype::VariableLength(VariableLengthType::string(
                StringPadding::NullTerminate,
                CharacterSet::Utf8,
            )),
            3,
        );
        assert_eq!(strings.read_string().unwrap(), ["alpha", "", "ζ"]);
        let holes = file.group("holes").unwrap();
        assert!(holes.groups().unwrap().is_empty());
        assert!(holes.attrs().unwrap().is_empty());
        let mut members = holes.datasets().unwrap();
        members.sort();
        let mut expected = (0..4)
            .map(|index| format!("spacer_{index}"))
            .collect::<Vec<_>>();
        if !phase.reclaimed() {
            expected.extend((0..4).map(|index| format!("scratch_{index}")));
        }
        if phase == Phase::Reuse {
            expected.push("replacement".into());
            assert_eq!(
                file.dataset("holes/replacement")
                    .unwrap()
                    .read_u8()
                    .unwrap(),
                [43; 128]
            );
        }
        expected.sort();
        assert_eq!(members, expected);
        for link_name in &members {
            assert_schema(
                &holes.dataset(link_name).unwrap(),
                Datatype::FixedPoint(FixedPointType::u8()),
                if link_name.starts_with("spacer_") {
                    64
                } else {
                    128
                },
            );
            assert!(
                holes
                    .dataset(link_name)
                    .unwrap()
                    .attrs()
                    .unwrap()
                    .is_empty()
            );
        }
        for index in 0..4 {
            assert_eq!(
                file.dataset(&format!("holes/spacer_{index}"))
                    .unwrap()
                    .read_u8()
                    .unwrap(),
                [31; 64]
            );
            if !phase.reclaimed() {
                assert_eq!(
                    file.dataset(&format!("holes/scratch_{index}"))
                        .unwrap()
                        .read_u8()
                        .unwrap(),
                    [17; 128]
                );
            }
        }
    }

    /// Collects storage and metadata-cache observations from a read-only fixture.
    ///
    /// The caller closes the editing handle before opening the fixture for observation.
    /// Dataset storage includes encoded variable-length descriptors. The dataset total excludes
    /// global heaps, indexes, object headers, and allocation padding.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FreeSectionInspectionUnsupported`] if the file is open for editing.
    /// Returns [`Error::EditUnsupported`] if persisted managers are absent or a dataset has an
    /// unexpected storage layout. Returns [`Error::Format`] if free bytes exceed the file size or
    /// dataset-storage totals overflow. Propagates errors from persisted-space inspection and
    /// dataset metadata reads.
    pub fn observe(&self, file: &File) -> Result<Observation, Error> {
        let persisted = file
            .persisted_free_sections()?
            .ok_or(Error::EditUnsupported(
                "width workload requires persisted free-space managers",
            ))?;
        let sections = persisted.free_sections();
        let non_reusable_bytes = file
            .file_size()
            .checked_sub(sections.free_bytes())
            .ok_or_else(|| FormatError::Internal("persisted free bytes exceed file size".into()))?;
        let dataset_bytes = self.dataset_storage_bytes(file)?;
        Ok(Observation {
            file_bytes: file.file_size(),
            reusable_bytes: sections.free_bytes(),
            non_reusable_bytes,
            reusable_sections: sections.section_count(),
            manager_bytes: persisted.manager_metadata_bytes(),
            dataset_bytes,
            metadata_cache: file.metadata_cache_stats(),
        })
    }

    fn maximum(&self) -> MaxExtent {
        match self.index {
            Index::Extensible => MaxExtent::Unlimited,
            Index::Finite => MaxExtent::Fixed(24),
        }
    }

    fn dataset_storage_bytes(&self, file: &File) -> Result<u64, Error> {
        let add = |offset: u64, length: u64| {
            offset
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow { offset, length })
        };
        let mut total = 0_u64;
        for group_name in ["metadata", "data", "holes"] {
            let group = file.group(group_name)?;
            for link_name in group.datasets()? {
                let dataset = group.dataset(&link_name)?;
                let length = match dataset.layout()? {
                    Layout::Contiguous {
                        address: Some(_),
                        size,
                    } => size,
                    Layout::Chunked { .. } => dataset
                        .chunks()?
                        .into_iter()
                        .map(|chunk| chunk.storage_size)
                        .try_fold(0, add)?,
                    _ => {
                        return Err(Error::EditUnsupported(
                            "width workload has unexpected dataset storage",
                        ));
                    }
                };
                total = add(total, length)?;
            }
        }
        Ok(total)
    }
}

/// The series dataset's maximum extent and chunk index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Index {
    /// An unlimited series with an Extensible Array index.
    Extensible,
    /// A finite series with a Fixed Array index.
    Finite,
}

/// Storage and cache measurements collected from one read-only fixture.
///
/// File-space totals are byte counts over the file image, including allocation padding. The
/// non-reusable remainder includes metadata, data, and padding outside persisted free sections.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    /// Physical byte length of the file.
    pub file_bytes: u64,
    /// Bytes in persisted reusable free sections.
    pub reusable_bytes: u64,
    /// File bytes outside persisted reusable free sections.
    pub non_reusable_bytes: u64,
    /// Persisted reusable section count.
    pub reusable_sections: usize,
    /// Allocated bytes of persisted manager metadata, including unused section-list capacity.
    pub manager_bytes: u64,
    /// Encoded dataset storage, excluding global heaps, chunk indexes, and allocation padding.
    pub dataset_bytes: u64,
    /// Metadata-read-cache statistics after collecting the storage observations.
    ///
    /// The byte count is retained buffer capacity. This field is [`None`] when the cache is
    /// disabled.
    pub metadata_cache: Option<MetadataCacheStats>,
}

/// The expected fixture state after a completed operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Appended series and single-chunk datasets.
    Append,
    /// Initial fixture contents.
    Create,
    /// Committed scratch deletion and attribute replacement.
    Reclaim,
    /// Committed replacement dataset creation.
    Reuse,
}

impl Phase {
    fn reclaimed(self) -> bool {
        matches!(self, Self::Reclaim | Self::Reuse)
    }
}

/// A length width at which the writer indexes the unlimited series with a Fixed Array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Unsupported {
    /// Requested length width.
    pub lengths: LengthWidth,
}

fn assert_schema(dataset: &Dataset, datatype: Datatype, length: u64) {
    assert_eq!(
        dataset.datatype().unwrap(),
        datatype,
        "width workload datatype differs"
    );
    assert_eq!(
        dataset.shape().unwrap(),
        [length],
        "width workload shape differs"
    );
}

#[cfg(test)]
mod tests;
