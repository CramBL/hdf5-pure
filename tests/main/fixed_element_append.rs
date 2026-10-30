use core::fmt::Debug;

use hdf5_pure::{
    CharacterSet, DatasetBuilder, Datatype, DatatypeByteOrder, Error, File, FileAccessProperties,
    FileBuilder, FixedAscii, FixedPointType, FormatError, H5Element, MaxExtent, MemoryStrategy,
    StringPadding, SyncPolicy,
};
use rstest::rstest;

use test_util_hdf5::dataset::{AppendFilters, AppendMode, Unlimited};
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::bytes_one(vec![[0u8], [127], [128], [255], [42], [1], [2]])]
#[case::bytes_three(vec![[0u8, 128, 255]; 7])]
#[case::ascii_one(vec![FixedAscii::<1>::new("A").unwrap(); 7])]
#[case::ascii_seven(vec![FixedAscii::<7>::new("a\0b").unwrap(); 7])]
fn fixed_elements_append_and_reopen<T: H5Element + Debug + PartialEq>(
    #[case] values: Vec<T>,
    #[values(
        AppendMode::Immediate,
        AppendMode::Staged,
        AppendMode::Buffered,
        AppendMode::OwnedBuffered
    )]
    mode: AppendMode,
    #[values(0, 2)] initial: usize,
    #[values(AppendFilters::Plain, AppendFilters::ShuffleDeflate)] filters: AppendFilters,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("values.h5");
    Unlimited::new("values", &values[..initial], 4)
        .filters(filters.filters())
        .pure_create(&path);
    let datatype = {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(MemoryStrategy::Bounded)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        let mut target = file.dataset("values").unwrap();
        let datatype = target.datatype().unwrap();
        mode.append(&file, &mut target, &values[initial..]).unwrap();
        assert_eq!(target.shape().unwrap(), [values.len() as u64]);
        assert_eq!(target.datatype().unwrap(), datatype);
        assert_eq!(target.read::<T>().unwrap(), values);
        datatype
    };
    let file = ReadBackend::Streaming.open(&path).unwrap();
    let target = file.dataset("values").unwrap();
    assert_eq!(target.shape().unwrap(), [values.len() as u64]);
    assert_eq!(target.datatype().unwrap(), datatype);
    assert_eq!(target.read::<T>().unwrap(), values);
}

#[rstest]
#[case::byte_arrays(Vec::<[u8; 3]>::new())]
#[case::ascii(Vec::<FixedAscii<7>>::new())]
fn empty_fixed_appends_preserve_storage<T: H5Element>(
    #[case] values: Vec<T>,
    #[values(
        AppendMode::Immediate,
        AppendMode::Staged,
        AppendMode::Buffered,
        AppendMode::OwnedBuffered
    )]
    mode: AppendMode,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.h5");
    Unlimited::new("values", &values, 4).pure_create(&path);
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut target = file.dataset("values").unwrap();
        let datatype = target.datatype().unwrap();
        mode.append(&file, &mut target, &values).unwrap();
        assert_eq!(target.shape().unwrap(), [0]);
        assert_eq!(target.datatype().unwrap(), datatype);
        assert!(target.chunks().unwrap().is_empty());
        assert!(target.read::<T>().unwrap().is_empty());
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[rstest]
#[case::numeric_one_byte_order(Datatype::FixedPoint(FixedPointType::new(1, DatatypeByteOrder::BigEndian, false, 0, 8).unwrap()), vec![0u8; 6])]
#[case::byte_width(array_type(3), vec![[0u8; 2]; 6])]
#[case::byte_base(Datatype::Array { base_type: Box::new(hdf5_pure::make_i8_type()), dimensions: vec![2] }, vec![[0u8; 2]; 6])]
#[case::byte_string(ascii_type(2), vec![[0u8; 2]; 6])]
#[case::ascii_width(ascii_type(3), vec![FixedAscii::<2>::new("a").unwrap(); 6])]
#[case::ascii_charset(Datatype::String { size: 2, padding: StringPadding::NullPad, charset: CharacterSet::Utf8 }, vec![FixedAscii::<2>::new("a").unwrap(); 6])]
#[case::ascii_padding(Datatype::String { size: 2, padding: StringPadding::SpacePad, charset: CharacterSet::Ascii }, vec![FixedAscii::<2>::new("a").unwrap(); 6])]
fn incompatible_fixed_appends_preserve_file<T: H5Element>(
    #[case] datatype: Datatype,
    #[case] values: Vec<T>,
    #[values(
        AppendMode::Immediate,
        AppendMode::Staged,
        AppendMode::Buffered,
        AppendMode::OwnedBuffered
    )]
    mode: AppendMode,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mismatch.h5");
    let mut builder = FileBuilder::new();
    let raw = vec![0; datatype.type_size() as usize * 2];
    builder
        .create_dataset("values")
        .with_raw_data(datatype.clone(), raw.clone(), 2)
        .with_chunks(&[4])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut target = file.dataset("values").unwrap();
        let error = mode.append(&file, &mut target, &values).unwrap_err();
        match mode {
            AppendMode::Staged => {
                let Error::AppendUnsupported(reason) = error else {
                    panic!("expected AppendUnsupported, got {error:?}");
                };
                assert_eq!(
                    reason,
                    "append datatype does not match the on-disk dataset (wrong element type or byte order)"
                );
            }
            _ => {
                let Error::AppendInPlaceUnsupported(reason) = error else {
                    panic!("expected AppendInPlaceUnsupported, got {error:?}");
                };
                assert_eq!(
                    reason,
                    "append datatype does not match the on-disk dataset (wrong element type or byte order)"
                );
            }
        }
        assert_eq!(target.shape().unwrap(), [2]);
        assert_eq!(target.datatype().unwrap(), datatype);
        assert_eq!(target.read_raw().unwrap(), raw);
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[rstest]
#[case::zero_width(Vec::<FixedAscii<0>>::new(), FormatError::ZeroFixedStringWidth)]
#[cfg_attr(target_pointer_width = "64", case::width_overflow(
    Vec::<FixedAscii<OVERFLOW_WIDTH>>::new(),
    FormatError::ValueTooLargeForRepresentation { value: OVERFLOW_WIDTH as u64, target: "u32" }
))]
fn invalid_ascii_append_widths_preserve_target<T: H5Element>(
    #[case] values: Vec<T>,
    #[case] expected: FormatError,
    #[values(
        AppendMode::Immediate,
        AppendMode::Staged,
        AppendMode::Buffered,
        AppendMode::OwnedBuffered
    )]
    mode: AppendMode,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid.h5");
    Unlimited::new("values", &[1u8, 2], 4).pure_create(&path);
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut target = file.dataset("values").unwrap();
        let error = mode.append(&file, &mut target, &values).unwrap_err();
        let Error::Format(actual) = error else {
            panic!("expected Format, got {error:?}");
        };
        assert_eq!(actual, expected);
        assert_eq!(target.read::<u8>().unwrap(), [1, 2]);
        assert!(!file.has_staged_edits());
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

fn array_type(width: u32) -> Datatype {
    Datatype::Array {
        base_type: Box::new(hdf5_pure::make_u8_type()),
        dimensions: vec![width],
    }
}

fn ascii_type(size: u32) -> Datatype {
    Datatype::String {
        size,
        padding: StringPadding::NullPad,
        charset: CharacterSet::Ascii,
    }
}

#[cfg(target_pointer_width = "64")]
const OVERFLOW_WIDTH: usize = u32::MAX as usize + 1;

#[rstest]
#[case::width_one(vec![[0u8], [127], [128], [255], [42], [1], [2]])]
#[case::width_three(vec![[0u8, 128, 255]; 7])]
fn byte_array_appends_preserve_big_endian_base_metadata<const N: usize>(
    #[case] values: Vec<[u8; N]>,
    #[values(
        TargetState::Committed(AppendMode::Immediate),
        TargetState::Committed(AppendMode::Staged),
        TargetState::Committed(AppendMode::Buffered),
        TargetState::Committed(AppendMode::OwnedBuffered),
        TargetState::Pending
    )]
    state: TargetState,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("big-endian.h5");
    let datatype = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::BigEndian, false, 0, 8).unwrap(),
        )),
        dimensions: vec![N as u32],
    };
    let configure = |builder: &mut DatasetBuilder| {
        builder
            .with_raw_data(datatype.clone(), values[..2].as_flattened().to_vec(), 2)
            .with_chunks(&[4])
            .with_maxshape(&[MaxExtent::Unlimited]);
    };
    let mut builder = FileBuilder::new();
    if let TargetState::Committed(_) = state {
        configure(builder.create_dataset("values"));
    }
    builder.write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let mut target = match state {
            TargetState::Committed(_) => file.dataset("values").unwrap(),
            TargetState::Pending => file.root().create_dataset("values", configure).unwrap(),
        };
        match state {
            TargetState::Committed(mode) => mode.append(&file, &mut target, &values[2..]).unwrap(),
            TargetState::Pending => {
                target
                    .append_staged(|append| {
                        append.append(&values[2..]);
                    })
                    .unwrap();
                file.commit().unwrap();
            }
        }
        assert_eq!(target.shape().unwrap(), [values.len() as u64]);
        assert_eq!(target.datatype().unwrap(), datatype);
        assert_eq!(target.read::<[u8; N]>().unwrap(), values);
    }
    let file = File::open_streaming(&path).unwrap();
    let target = file.dataset("values").unwrap();
    assert_eq!(target.datatype().unwrap(), datatype);
    assert_eq!(target.read::<[u8; N]>().unwrap(), values);
}

#[derive(Clone, Copy, Debug)]
enum TargetState {
    Committed(AppendMode),
    Pending,
}
