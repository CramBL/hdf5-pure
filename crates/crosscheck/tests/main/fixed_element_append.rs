#![cfg(feature = "__hdf5-1.10")]

use core::fmt::Debug;

use hdf5::H5Type;
use hdf5::types::FixedAscii as NativeAscii;
use hdf5_pure::{File, FileAccessProperties, FixedAscii, H5Element, MemoryStrategy, SyncPolicy};
use rstest::rstest;

use test_util_hdf5::dataset::{AppendFilters, AppendMode, Unlimited};

#[rstest]
#[case::bytes_one(vec![[0u8], [127], [128], [255], [42], [1], [2]], vec![[0u8], [127], [128], [255], [42], [1], [2]])]
#[case::bytes_three(vec![[0u8, 128, 255]; 7], vec![[0u8, 128, 255]; 7])]
#[case::ascii_one(ascii::<1>(&["", "A", "\x7f", "B", "C", "D", "E"]), native_ascii::<1>(&["", "A", "\x7f", "B", "C", "D", "E"]))]
#[case::ascii_seven(ascii::<7>(&["ab", "", "a\0b", "1234567", "\x7f", "xyz", "end"]), native_ascii::<7>(&["ab", "", "a\0b", "1234567", "\x7f", "xyz", "end"]))]
fn fixed_appends_match_libhdf5<T, N>(
    #[case] values: Vec<T>,
    #[case] native: Vec<N>,
    #[values(
        AppendMode::Immediate,
        AppendMode::Staged,
        AppendMode::Buffered,
        AppendMode::OwnedBuffered
    )]
    mode: AppendMode,
    #[values(Producer::Pure, Producer::Native)] producer: Producer,
    #[values(0, 2)] initial: usize,
    #[values(AppendFilters::Plain, AppendFilters::ShuffleDeflate)] filters: AppendFilters,
) where
    T: H5Element + Debug + PartialEq,
    N: H5Type + Copy + Debug + PartialEq,
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("interop.h5");
    match producer {
        Producer::Pure => Unlimited::new("values", &values[..initial], 4)
            .filters(filters.filters())
            .pure_create(&path),
        Producer::Native => Unlimited::new("values", &native[..initial], 4)
            .filters(filters.filters())
            .libhdf5_create(&path),
    }
    {
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
        assert_eq!(target.shape().unwrap(), [7]);
        assert_eq!(target.datatype().unwrap(), datatype);
        assert_eq!(target.chunk_shape().unwrap(), Some(vec![4]));
        assert_eq!(target.read::<T>().unwrap(), values);
    }
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        let target = file.dataset("values").unwrap();
        assert_eq!(target.shape(), [7]);
        assert_eq!(target.chunk(), Some(vec![4]));
        assert_eq!(
            target.dtype().unwrap().to_descriptor().unwrap(),
            N::type_descriptor()
        );
        assert_eq!(target.read_raw::<N>().unwrap(), native);
        assert_eq!(
            target.filters(),
            match filters {
                AppendFilters::Plain => vec![],
                AppendFilters::ShuffleDeflate => vec![
                    hdf5::filters::Filter::Shuffle,
                    hdf5::filters::Filter::Deflate(hdf5_pure::DeflateLevel::Six.get()),
                ],
            }
        );
        let mut extended = native.clone();
        extended.push(native[0]);
        target.resize([8]).unwrap();
        target.write_raw(&extended).unwrap();
        target.flush().unwrap();
        drop(target);
        file.close().unwrap();
    }
    let file = File::open_streaming(&path).unwrap();
    let target = file.dataset("values").unwrap();
    let mut expected = values.clone();
    expected.push(values[0]);
    assert_eq!(target.shape().unwrap(), [8]);
    assert_eq!(target.read::<T>().unwrap(), expected);
}

#[derive(Clone, Copy, Debug)]
enum Producer {
    Native,
    Pure,
}

fn ascii<const N: usize>(values: &[&str]) -> Vec<FixedAscii<N>> {
    values
        .iter()
        .map(|value| FixedAscii::new(value).unwrap())
        .collect()
}

fn native_ascii<const N: usize>(values: &[&str]) -> Vec<NativeAscii<N>> {
    values
        .iter()
        .map(|value| NativeAscii::from_ascii(value).unwrap())
        .collect()
}
