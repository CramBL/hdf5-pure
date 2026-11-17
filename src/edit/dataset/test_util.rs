use core::num::NonZeroUsize;

use crate::DeflateLevel;
use crate::MemoryStrategy;
use crate::chunked_write;
use crate::chunked_write::CompressedChunkSet;
use crate::dataset_request::PreparedFill;
use crate::dataspace::MaxExtent;
use crate::file_writer;
use crate::fill_value::FillPattern;
use crate::filters::ChunkContext;
use crate::object_path::LinkNameBuf;
use crate::type_builders;
use crate::type_builders::DatasetBuilder;

use super::FlatDataset;

pub(in crate::edit) mod lifecycle;

/// Chunked dataset descriptions for creation and ownership checks.
#[derive(Clone, Copy)]
pub(in crate::edit) enum Creation {
    /// Four-element chunks with an unlimited dimension.
    Extensible(usize),
    /// Four-element chunks with a fixed extent.
    Fixed(usize),
    /// Forty chunks within a maximum extent of 320 elements.
    FixedSparse,
    /// One initial chunk with values for an additional chunk after staging.
    Pending,
    /// A two-dimensional array with a fixed first maximum and an unlimited second dimension.
    Rotated(u64),
    /// Eight fill elements without allocated payload.
    Unallocated,
}

impl Creation {
    pub(in crate::edit) fn configure(self, builder: &mut DatasetBuilder, filtered: bool) {
        if matches!(self, Self::Unallocated) {
            builder
                .with_unallocated_storage(type_builders::make_i32_type(), &[8])
                .with_fill_value(-7_i32);
        } else {
            builder.with_i32_data(&self.initial());
        }
        match self {
            Self::Extensible(_) | Self::Pending | Self::Unallocated => {
                builder
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            }
            Self::Fixed(_) => {
                builder.with_chunks(&[4]);
            }
            Self::FixedSparse => {
                builder
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Fixed(320)]);
            }
            Self::Rotated(maximum) => {
                builder
                    .with_shape(&[2, 2])
                    .with_chunks(&[1, 1])
                    .with_maxshape(&[MaxExtent::Fixed(maximum), MaxExtent::Unlimited]);
            }
        }
        if filtered {
            builder.with_deflate(DeflateLevel::Six);
        }
    }

    pub(in crate::edit) fn expected(self) -> Vec<i32> {
        if matches!(self, Self::Pending) {
            return (1..=8).collect();
        }
        self.initial()
    }

    /// Returns element coordinates for the expected allocated chunks.
    pub(in crate::edit) fn coordinates(self) -> Vec<Vec<u64>> {
        match self {
            Self::Extensible(count) | Self::Fixed(count) => (0..count)
                .map(|index| vec![u64::try_from(index * 4).unwrap()])
                .collect(),
            Self::FixedSparse => (0..40).map(|index| vec![index * 4]).collect(),
            Self::Pending => vec![vec![0], vec![4]],
            Self::Unallocated => Vec::new(),
            Self::Rotated(_) => vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
        }
    }

    fn initial(self) -> Vec<i32> {
        match self {
            Self::Extensible(count) | Self::Fixed(count) => (0..count * 4)
                .map(|index| i32::try_from(index / 16 % 7).unwrap())
                .collect(),
            Self::FixedSparse => Self::Fixed(40).initial(),
            Self::Pending => (1..=4).collect(),
            Self::Unallocated => vec![-7; 8],
            Self::Rotated(_) => vec![11, 12, 13, 14],
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::edit) enum CreationAccess {
    Bounded,
    Mirrored,
    Userblock,
}

impl CreationAccess {
    pub(in crate::edit) fn base(self) -> u64 {
        match self {
            Self::Bounded | Self::Mirrored => 0,
            Self::Userblock => 8_192,
        }
    }

    pub(in crate::edit) fn strategy(self) -> MemoryStrategy {
        match self {
            Self::Bounded => MemoryStrategy::Bounded,
            Self::Mirrored | Self::Userblock => MemoryStrategy::Mirrored,
        }
    }
}

pub(in crate::edit) fn prepared(creation: Creation, filtered: bool) -> CompressedChunkSet {
    let mut builder = DatasetBuilder::new("d");
    creation.configure(&mut builder, filtered);
    let FlatDataset {
        name: _,
        dt,
        ds,
        storage,
        extent: _,
        attrs: _,
        chunk_options,
        maxshape,
        vl_attrs: _,
        dense_attrs: _,
        fill,
        fill_time,
        #[cfg(feature = "provenance")]
            provenance: _,
    } = super::flatten_dataset(file_writer::WIDTHS, builder, LinkNameBuf::new("d").unwrap())
        .unwrap();
    let dimensions = chunk_options.resolve_chunk_dims(&ds.dimensions);
    let context = ChunkContext::from_datatype(&dimensions, &dt).unwrap();
    let element = NonZeroUsize::new(usize::try_from(context.element_size.get()).unwrap()).unwrap();
    chunked_write::compress_chunks(
        file_writer::WIDTHS,
        storage.raw(),
        &ds.dimensions,
        context,
        &chunk_options,
        maxshape.as_deref(),
        FillPattern::from_value(fill.as_ref().map(PreparedFill::as_bytes), element),
        fill_time,
        storage.allocation(),
    )
    .unwrap()
}
