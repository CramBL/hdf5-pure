use std::fmt::Debug;
use std::path::Path;

use hdf5::Extent;
use hdf5::H5Type;
use hdf5::datatype::ByteOrder;
use hdf5::filters::Filter;
use hdf5::plist::dataset_create::{FillTime as NativeFillTime, Layout as NativeLayout};
use hdf5::types::FixedAscii as NativeAscii;
use hdf5::types::TypeDescriptor;
use hdf5_pure::{
    AttrValue, DatasetBuilder, File, FileAccessProperties, FileBuilder, FileSpaceSettings,
    FileSpaceStrategy, FillTime, FixedAscii, H5Element, Layout, MaxExtent, MemoryStrategy,
    SyncPolicy,
};
use rstest::rstest;
use test_util_hdf5::dataset::AppendMode;
use test_util_hdf5::file;

use super::Creation;

#[rstest]
#[case::numeric(-7_i32, 0_i32, -7_i32, 0_i32)]
#[case::byte_array([0_u8, 128, 255], [0_u8; 3], [0_u8, 128, 255], [0_u8; 3])]
#[case::ascii(
    FixedAscii::<4>::new("abc").unwrap(), FixedAscii::<4>::new("").unwrap(),
    NativeAscii::<4>::from_ascii("abc").unwrap(), NativeAscii::<4>::from_ascii("").unwrap()
)]
fn typed_schemas_match_explicit_and_native_storage<T, N>(
    #[case] fill: T,
    #[case] zero: T,
    #[case] native_fill: N,
    #[case] native_zero: N,
    #[values(
        Schema::Scalar,
        Schema::Contiguous,
        Schema::EmptyChunked,
        Schema::FixedChunked,
        Schema::FilteredChunked,
        Schema::EmptyUnlimited,
        Schema::FilteredEmptyUnlimited,
        Schema::Unlimited
    )]
    schema: Schema,
    #[values(false, true)] explicit_fill: bool,
    #[values(Creation::Buffered, Creation::Streaming, Creation::Edit)] creation: Creation,
) where
    T: H5Element + Debug + PartialEq,
    N: H5Type + Copy + Debug + PartialEq,
{
    let directory = tempfile::tempdir().unwrap();
    let native_path = directory.path().join("native.h5");
    let explicit_path = directory.path().join("explicit.h5");
    let typed_path = directory.path().join("typed.h5");
    let native = file::libhdf5_create_v110(&native_path);
    let builder = native.new_dataset::<N>();
    let mut builder = if matches!(schema, Schema::Scalar) {
        builder.shape(())
    } else {
        builder.shape(schema.native_extents())
    };
    if schema.chunked() {
        builder = builder.chunk([2]);
    }
    if schema.filtered() {
        builder = builder.fletcher32();
    }
    if explicit_fill {
        builder = builder.fill_value(native_fill);
    }
    builder = builder.fill_time(NativeFillTime::IfSet);
    let dataset = builder.create("schema").unwrap();
    drop(builder);
    dataset
        .new_attr::<i32>()
        .shape(())
        .create("code")
        .unwrap()
        .write_scalar(&42)
        .unwrap();
    drop(dataset);
    native.close().unwrap();
    creation.write(&explicit_path, |dataset| {
        dataset.with_unallocated_storage(T::datatype().unwrap(), schema.dims());
        schema.configure(dataset);
        if explicit_fill {
            dataset.with_fill_value(fill);
        }
    });
    creation.write(&typed_path, |dataset| {
        dataset.with_unallocated_storage_for::<T>(schema.dims());
        schema.configure(dataset);
        if explicit_fill {
            dataset.with_fill_value(fill);
        }
    });
    assert_eq!(
        std::fs::read(&typed_path).unwrap(),
        std::fs::read(&explicit_path).unwrap()
    );
    let fill = if explicit_fill { fill } else { zero };
    let native_fill = if explicit_fill {
        native_fill
    } else {
        native_zero
    };
    assert_pure_schema(&typed_path, schema, fill, explicit_fill.then_some(fill));
    assert_pure_schema(&explicit_path, schema, fill, explicit_fill.then_some(fill));
    let canonical_order = matches!(N::type_descriptor(), TypeDescriptor::Integer(_))
        .then_some(ByteOrder::LittleEndian);
    assert_native_schema(&typed_path, schema, native_fill, canonical_order);
    assert_native_schema(&explicit_path, schema, native_fill, canonical_order);
    assert_native_schema(&native_path, schema, native_fill, None);
    let source = File::open(&native_path).unwrap();
    assert_eq!(
        source.dataset("schema").unwrap().read::<T>().unwrap(),
        vec![fill; schema.elements()]
    );
    source.close().unwrap();
    if matches!(
        schema,
        Schema::EmptyUnlimited | Schema::FilteredEmptyUnlimited
    ) {
        super::assert_deferred_layout(&typed_path, "schema");
        super::assert_deferred_layout(&explicit_path, "schema");
        super::assert_deferred_layout(&native_path, "schema");
    }
}

#[rstest]
#[case::byte_array(
    vec![[0_u8, 128, 255], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
    core::convert::identity::<[u8; 3]>, FileSpaceStrategy::FsmAggr, MemoryStrategy::Mirrored, SyncPolicy::Always
)]
#[case::ascii(
    ["ab", "", "a\0b", "wxyz", "A", "\x7f", "last"].map(|text| FixedAscii::<4>::new(text).unwrap()).to_vec(),
    native_ascii, FileSpaceStrategy::Page, MemoryStrategy::Bounded, SyncPolicy::OnClose
)]
fn typed_fixed_elements_initialize_and_append_batches<T, N>(
    #[case] values: Vec<T>,
    #[case] native_value: fn(T) -> N,
    #[case] strategy: FileSpaceStrategy,
    #[case] backing: MemoryStrategy,
    #[case] sync: SyncPolicy,
    #[values(
        AppendMode::Immediate,
        AppendMode::Staged,
        AppendMode::Buffered,
        AppendMode::OwnedBuffered
    )]
    mode: AppendMode,
    #[values(false, true)] filtered: bool,
    #[values(Creation::Buffered, Creation::Streaming, Creation::Edit)] creation: Creation,
) where
    T: H5Element + Debug + PartialEq,
    N: H5Type + Copy + Debug + PartialEq,
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("append.h5");
    let configure = |schema: &mut DatasetBuilder| {
        schema
            .with_unallocated_storage_for::<T>(&[0])
            .with_chunks(&[4])
            .with_maxshape(&[MaxExtent::Unlimited]);
        if filtered {
            schema.with_fletcher32();
        }
    };
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    match creation {
        Creation::Buffered | Creation::Streaming => {
            configure(builder.create_dataset("schema"));
            if matches!(creation, Creation::Buffered) {
                std::fs::write(&path, builder.finish().unwrap()).unwrap();
            } else {
                builder.write(&path).unwrap();
            }
        }
        Creation::Edit => {
            builder.write(&path).unwrap();
            let file = File::open_rw(&path).unwrap();
            file.root().create_dataset("schema", configure).unwrap();
            file.commit().unwrap();
            file.close().unwrap();
        }
    }
    let original = std::fs::read(&path).unwrap();
    super::assert_deferred_layout(&path, "schema");
    let access = || {
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(sync)
    };
    let file = File::open_rw_with_options(&path, access()).unwrap();
    let mut dataset = file.dataset("schema").unwrap();
    let reusable = file.reusable_space_stats().unwrap();
    mode.append::<T>(&file, &mut dataset, &[]).unwrap();
    mode.append::<T>(&file, &mut dataset, &[]).unwrap();
    assert_eq!(file.reusable_space_stats().unwrap(), reusable);
    assert_eq!(dataset.datatype().unwrap(), T::datatype().unwrap());
    assert!(dataset.chunks().unwrap().is_empty());
    drop(dataset);
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original);
    super::assert_deferred_layout(&path, "schema");
    let native = hdf5::File::open(&path).unwrap();
    let dataset = native.dataset("schema").unwrap();
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        N::type_descriptor()
    );
    assert_eq!(dataset.storage_size(), 0);
    assert_eq!(dataset.num_chunks(), Some(0));
    drop(dataset);
    native.close().unwrap();
    let file = File::open_rw_with_options(&path, access()).unwrap();
    let mut dataset = file.dataset("schema").unwrap();
    mode.append(&file, &mut dataset, &values[..3]).unwrap();
    assert_eq!(dataset.read::<T>().unwrap(), values[..3]);
    assert_eq!(dataset.chunks().unwrap().len(), 1);
    drop(dataset);
    file.close().unwrap();
    let file = File::open_rw_with_options(&path, access()).unwrap();
    let mut dataset = file.dataset("schema").unwrap();
    mode.append(&file, &mut dataset, &values[3..]).unwrap();
    mode.append::<T>(&file, &mut dataset, &[]).unwrap();
    assert_eq!(dataset.shape().unwrap(), [7]);
    assert_eq!(dataset.chunks().unwrap().len(), 2);
    assert_eq!(dataset.datatype().unwrap(), T::datatype().unwrap());
    assert_eq!(dataset.read::<T>().unwrap(), values);
    drop(dataset);
    file.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let dataset = native.dataset("schema").unwrap();
    assert_eq!(dataset.shape(), [7]);
    assert_eq!(dataset.num_chunks(), Some(2));
    assert!(dataset.storage_size() > 0);
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        N::type_descriptor()
    );
    assert_eq!(
        dataset.read_raw::<N>().unwrap(),
        values.into_iter().map(native_value).collect::<Vec<_>>()
    );
    drop(dataset);
    native.close().unwrap();
}

fn assert_pure_schema<T: H5Element + Debug + PartialEq>(
    path: &Path,
    schema: Schema,
    fill: T,
    declared_fill: Option<T>,
) {
    let file = File::open(path).unwrap();
    let dataset = file.dataset("schema").unwrap();
    assert_eq!(dataset.datatype().unwrap(), T::datatype().unwrap());
    assert_eq!(dataset.shape().unwrap(), schema.dims());
    assert_eq!(
        dataset.maxshape().unwrap(),
        schema.maxshape().map(|maximum| vec![maximum])
    );
    assert_eq!(dataset.fill_value::<T>().unwrap(), declared_fill);
    assert_eq!(dataset.fill_time().unwrap(), FillTime::IfSet);
    assert_eq!(dataset.attrs().unwrap()["code"], AttrValue::I32(42));
    assert_eq!(dataset.read::<T>().unwrap(), vec![fill; schema.elements()]);
    if schema.chunked() {
        assert!(dataset.chunks().unwrap().is_empty());
    } else {
        assert_eq!(
            dataset.layout().unwrap(),
            Layout::Contiguous {
                address: None,
                size: u64::try_from(schema.elements()).unwrap()
                    * u64::from(T::datatype().unwrap().type_size()),
            }
        );
    }
    drop(dataset);
    file.close().unwrap();
}

fn assert_native_schema<N: H5Type + Copy + Debug + PartialEq>(
    path: &Path,
    schema: Schema,
    fill: N,
    expected_order: Option<ByteOrder>,
) {
    let file = hdf5::File::open(path).unwrap();
    let dataset = file.dataset("schema").unwrap();
    let datatype = dataset.dtype().unwrap();
    assert_eq!(datatype.to_descriptor().unwrap(), N::type_descriptor());
    if let Some(order) = expected_order {
        assert_eq!(datatype.byte_order(), order);
    }
    assert_eq!(
        dataset.shape(),
        schema
            .dims()
            .iter()
            .map(|&dim| usize::try_from(dim).unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        dataset.space().unwrap().maxdims(),
        if matches!(schema, Schema::Scalar) {
            vec![]
        } else {
            vec![schema.maximum()]
        }
    );
    assert_eq!(dataset.offset(), None);
    assert_eq!(dataset.chunk(), schema.chunked().then(|| vec![2]));
    assert_eq!(
        dataset.layout(),
        if schema.chunked() {
            NativeLayout::Chunked
        } else {
            NativeLayout::Contiguous
        }
    );
    assert_eq!(
        dataset.filters(),
        if schema.filtered() {
            vec![Filter::Fletcher32]
        } else {
            vec![]
        }
    );
    assert_eq!(
        dataset.dcpl().unwrap().get_fill_time().unwrap(),
        NativeFillTime::IfSet
    );
    assert_eq!(
        dataset.dcpl().unwrap().get_fill_value_as::<N>().unwrap(),
        Some(fill)
    );
    assert_eq!(
        dataset.attr("code").unwrap().read_scalar::<i32>().unwrap(),
        42
    );
    assert_eq!(dataset.storage_size(), 0);
    assert_eq!(
        dataset.read_raw::<N>().unwrap(),
        vec![fill; schema.elements()]
    );
    assert_eq!(dataset.storage_size(), 0);
    if schema.chunked() {
        assert_eq!(dataset.num_chunks(), Some(0));
    }
    drop(datatype);
    drop(dataset);
    file.close().unwrap();
}

#[derive(Clone, Copy)]
enum Schema {
    Contiguous,
    EmptyChunked,
    EmptyUnlimited,
    FilteredChunked,
    FilteredEmptyUnlimited,
    FixedChunked,
    Scalar,
    Unlimited,
}

impl Schema {
    fn configure(self, dataset: &mut DatasetBuilder) {
        dataset.set_attr("code", AttrValue::I32(42));
        if self.chunked() {
            dataset
                .with_chunks(&[2])
                .with_maxshape(&[self.maxshape().unwrap()]);
        }
        if self.filtered() {
            dataset.with_fletcher32();
        }
    }

    fn dims(self) -> &'static [u64] {
        match self {
            Self::Scalar => &[],
            Self::EmptyChunked | Self::EmptyUnlimited | Self::FilteredEmptyUnlimited => &[0],
            _ => &[3],
        }
    }

    fn native_extents(self) -> Vec<Extent> {
        self.dims()
            .iter()
            .map(|&size| Extent::new(usize::try_from(size).unwrap(), self.maximum()))
            .collect()
    }

    fn elements(self) -> usize {
        self.dims()
            .iter()
            .map(|&size| usize::try_from(size).unwrap())
            .product()
    }

    fn maximum(self) -> Option<usize> {
        match self.maxshape() {
            Some(MaxExtent::Unlimited) => None,
            Some(MaxExtent::Fixed(value)) => Some(usize::try_from(value).unwrap()),
            None => self
                .dims()
                .first()
                .map(|&size| usize::try_from(size).unwrap()),
        }
    }

    fn maxshape(self) -> Option<MaxExtent> {
        match self {
            Self::Scalar | Self::Contiguous => None,
            Self::EmptyChunked | Self::FixedChunked | Self::FilteredChunked => {
                Some(MaxExtent::Fixed(4))
            }
            _ => Some(MaxExtent::Unlimited),
        }
    }

    fn chunked(self) -> bool {
        !matches!(self, Self::Scalar | Self::Contiguous)
    }

    fn filtered(self) -> bool {
        matches!(self, Self::FilteredChunked | Self::FilteredEmptyUnlimited)
    }
}

fn native_ascii(value: FixedAscii<4>) -> NativeAscii<4> {
    NativeAscii::from_ascii(value.as_bytes()).unwrap()
}
