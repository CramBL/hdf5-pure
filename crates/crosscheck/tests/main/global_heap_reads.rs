#![cfg(feature = "hdf5")]
use hdf5::dataset::Layout as CLayout;
use hdf5::plist::file_create::Sizeof;
use hdf5::types::VarLenUnicode;
use hdf5_pure::AttrValue;
use rstest::rstest;
use test_util_hdf5::reader::ReadBackend;

use super::global_heap_framing;
use super::global_heap_framing::STRING_COUNT;

#[rstest]
fn hdf5_pure_reads_the_strings_libhdf5_writes(
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_size: Sizeof,
    #[values(0, USERBLOCK)] userblock: u64,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    {
        let file = global_heap_framing::create_libhdf5_file(&path, sizeof_size, userblock);
        let values: Vec<VarLenUnicode> = global_heap_framing::strings()
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        for (name, layout) in [
            ("contiguous", CLayout::Contiguous),
            ("chunked", CLayout::Chunked),
            ("compact", CLayout::Compact),
        ] {
            let dataset = file.new_dataset::<VarLenUnicode>().shape([STRING_COUNT]);
            let dataset = match layout {
                CLayout::Chunked => dataset.chunk([STRING_COUNT / 3]),
                layout => dataset.layout(layout),
            };
            dataset.create(name).unwrap().write(&values).unwrap();
        }
        file.new_attr::<VarLenUnicode>()
            .shape([STRING_COUNT])
            .create("labels")
            .unwrap()
            .write(&values)
            .unwrap();
        file.close().unwrap();
    }

    let file = backend.open(&path).unwrap();
    let read: Vec<(&str, Vec<String>)> = ["contiguous", "chunked", "compact"]
        .into_iter()
        .map(|name| (name, file.dataset(name).unwrap().read_string().unwrap()))
        .collect();
    assert_eq!(
        read,
        ["contiguous", "chunked", "compact"].map(|name| (name, global_heap_framing::strings()))
    );
    assert_eq!(
        file.root().attrs().unwrap()["labels"],
        AttrValue::VarLenStringArray(global_heap_framing::strings())
    );
}

#[cfg(feature = "__hdf5-1.10")]
#[rstest]
fn a_repack_preserves_the_sequences_libhdf5_writes(
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_size: Sizeof,
    #[values(0, USERBLOCK)] userblock: u64,
) {
    use hdf5::types::VarLenArray;

    #[derive(hdf5::H5Type, Clone, Debug, PartialEq)]
    #[repr(C)]
    struct Labelled {
        id: i32,
        label: VarLenUnicode,
        samples: [VarLenArray<i32>; 2],
    }

    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("vlen.h5");
    let repacked = dir.path().join("repacked.h5");
    let sequences: Vec<Vec<i32>> = (0..STRING_COUNT as i32)
        .map(|i| (0..i % 5).map(|j| i * 10 + j).collect())
        .collect();
    let labelled: Vec<Labelled> = global_heap_framing::strings()
        .iter()
        .zip(&sequences)
        .enumerate()
        .map(|(i, (label, sequence))| Labelled {
            id: i as i32,
            label: label.parse().unwrap(),
            samples: [
                VarLenArray::from_slice(sequence),
                VarLenArray::from_slice(&sequence[..sequence.len() / 2]),
            ],
        })
        .collect();
    {
        let file = global_heap_framing::create_libhdf5_file(&source, sizeof_size, userblock);
        let values: Vec<VarLenArray<i32>> = sequences
            .iter()
            .map(|sequence| VarLenArray::from_slice(sequence))
            .collect();
        file.new_dataset::<VarLenArray<i32>>()
            .shape([STRING_COUNT])
            .create("sequences")
            .unwrap()
            .write(&values)
            .unwrap();
        file.new_dataset::<Labelled>()
            .shape([STRING_COUNT])
            .create("labelled")
            .unwrap()
            .write(&labelled)
            .unwrap();
        file.close().unwrap();
    }

    hdf5_pure::repack(&source, &repacked, &hdf5_pure::RepackOptions::new()).unwrap();

    let file = hdf5::File::open(&repacked).unwrap();
    let read: Vec<Vec<i32>> = file
        .dataset("sequences")
        .unwrap()
        .read_raw::<VarLenArray<i32>>()
        .unwrap()
        .iter()
        .map(|sequence| sequence.to_vec())
        .collect();
    assert_eq!(read, sequences);
    assert_eq!(
        file.dataset("labelled")
            .unwrap()
            .read_raw::<Labelled>()
            .unwrap(),
        labelled
    );
}

const USERBLOCK: u64 = 512;
