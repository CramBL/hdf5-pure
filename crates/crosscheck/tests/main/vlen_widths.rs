#![cfg(feature = "hdf5")]

use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5::types::VarLenArray;
use hdf5::types::VarLenAscii;
use hdf5_pure::LengthWidth;
use hdf5_pure::OffsetWidth;
use hdf5_pure::RepackOptions;
use rstest::rstest;
use test_util_hdf5::reader::ReadBackend;

#[derive(Clone, Copy, Debug)]
enum Storage {
    Chunked,
    Contiguous,
    Filtered,
    Unallocated,
}

#[rstest]
#[case::two_two(Sizeof::Bytes2, Sizeof::Bytes2, 10)]
#[case::two_four(Sizeof::Bytes2, Sizeof::Bytes4, 10)]
#[case::two_eight(Sizeof::Bytes2, Sizeof::Bytes8, 10)]
#[case::four_two(Sizeof::Bytes4, Sizeof::Bytes2, 12)]
#[case::four_four(Sizeof::Bytes4, Sizeof::Bytes4, 12)]
#[case::four_eight(Sizeof::Bytes4, Sizeof::Bytes8, 12)]
#[case::eight_two(Sizeof::Bytes8, Sizeof::Bytes2, 16)]
#[case::eight_four(Sizeof::Bytes8, Sizeof::Bytes4, 16)]
#[case::eight_eight(Sizeof::Bytes8, Sizeof::Bytes8, 16)]
fn reads_native_variable_length_references_with_independent_widths(
    #[case] sizeof_addr: Sizeof,
    #[case] sizeof_size: Sizeof,
    #[case] element_size: u64,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
    #[values(
        Storage::Contiguous,
        Storage::Chunked,
        Storage::Filtered,
        Storage::Unallocated
    )]
    storage: Storage,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");
    let expected = ["alpha", "", "gamma", "delta"];
    {
        let file = hdf5::File::with_options()
            .with_create_plist(|properties| {
                properties.sizes(SizeofInfo {
                    sizeof_addr,
                    sizeof_size,
                })
            })
            .create(&path)
            .unwrap();
        let array = file
            .new_dataset::<[VarLenAscii; 2]>()
            .shape(2)
            .create("arrays")
            .unwrap();
        let arrays = [
            [
                VarLenAscii::from_ascii("first").unwrap(),
                VarLenAscii::from_ascii("second").unwrap(),
            ],
            [
                VarLenAscii::from_ascii("third").unwrap(),
                VarLenAscii::from_ascii("fourth").unwrap(),
            ],
        ];
        array.write_raw(&arrays).unwrap();
        assert_eq!(array.storage_size(), 4 * element_size);
        assert_eq!(array.read_raw::<[VarLenAscii; 2]>().unwrap(), arrays);
        let sequences = [
            VarLenArray::from_slice(&[3u16, 5]),
            VarLenArray::from_slice(&[7u16]),
        ];
        let sequence = file
            .new_dataset::<VarLenArray<u16>>()
            .shape(2)
            .create("sequences")
            .unwrap();
        sequence.write_raw(&sequences).unwrap();
        assert_eq!(sequence.storage_size(), 2 * element_size);
        assert_eq!(sequence.read_raw::<VarLenArray<u16>>().unwrap(), sequences);
        let builder = file.new_dataset::<VarLenAscii>().shape(4);
        let builder = match storage {
            Storage::Contiguous | Storage::Unallocated => builder,
            Storage::Chunked => builder.chunk(2),
            Storage::Filtered => builder.chunk(2).deflate(6),
        };
        let dataset = builder.create("strings").unwrap();
        if !matches!(storage, Storage::Unallocated) {
            let values = expected.map(|value| VarLenAscii::from_ascii(value).unwrap());
            dataset.write_raw(&values).unwrap();
            if matches!(storage, Storage::Contiguous) {
                assert_eq!(dataset.storage_size(), 4 * element_size);
            }
            assert_eq!(dataset.read_raw::<VarLenAscii>().unwrap(), values);
        }
    }
    let file = backend.open(&path).unwrap();
    let array = file.dataset("arrays").unwrap();
    assert_eq!(array.element_size().unwrap(), 2 * element_size);
    assert_eq!(array.read_raw().unwrap().len() as u64, 4 * element_size);
    assert_eq!(
        array.read_raw_region(&[1], &[1]).unwrap().len() as u64,
        2 * element_size
    );
    let sequence = file.dataset("sequences").unwrap();
    assert_eq!(sequence.element_size().unwrap(), element_size);
    assert_eq!(sequence.read_raw().unwrap().len() as u64, 2 * element_size);
    drop(array);
    drop(sequence);
    let dataset = file.dataset("strings").unwrap();
    let expected = if matches!(storage, Storage::Unallocated) {
        [""; 4]
    } else {
        expected
    };
    assert_eq!(dataset.element_size().unwrap(), element_size);
    assert_eq!(dataset.read_raw().unwrap().len() as u64, 4 * element_size);
    assert_eq!(dataset.read_string().unwrap(), expected);
    assert_eq!(
        dataset.read_string_region(&[1], &[2]).unwrap(),
        expected[1..3]
    );
    assert_eq!(
        dataset.read_raw_region(&[1], &[2]).unwrap().len() as u64,
        2 * element_size
    );
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
#[case::two_two(Sizeof::Bytes2, Sizeof::Bytes2)]
#[case::two_four(Sizeof::Bytes2, Sizeof::Bytes4)]
#[case::two_eight(Sizeof::Bytes2, Sizeof::Bytes8)]
#[case::four_two(Sizeof::Bytes4, Sizeof::Bytes2)]
#[case::four_four(Sizeof::Bytes4, Sizeof::Bytes4)]
#[case::four_eight(Sizeof::Bytes4, Sizeof::Bytes8)]
#[case::eight_two(Sizeof::Bytes8, Sizeof::Bytes2)]
#[case::eight_four(Sizeof::Bytes8, Sizeof::Bytes4)]
#[case::eight_eight(Sizeof::Bytes8, Sizeof::Bytes8)]
fn repack_preserves_eight_byte_output_widths_for_variable_length_sources(
    #[case] sizeof_addr: Sizeof,
    #[case] sizeof_size: Sizeof,
) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.h5");
    let destination = dir.path().join("destination.h5");
    let strings = [
        VarLenAscii::from_ascii("first").unwrap(),
        VarLenAscii::from_ascii("").unwrap(),
    ];
    let sequences = [
        VarLenArray::from_slice(&[3u16, 5]),
        VarLenArray::from_slice(&[7u16]),
    ];
    {
        let file = hdf5::File::with_options()
            .with_create_plist(|properties| {
                properties.sizes(SizeofInfo {
                    sizeof_addr,
                    sizeof_size,
                })
            })
            .create(&source)
            .unwrap();
        file.new_dataset::<VarLenAscii>()
            .shape(2)
            .create("strings")
            .unwrap()
            .write_raw(&strings)
            .unwrap();
        file.new_dataset::<VarLenArray<u16>>()
            .shape(2)
            .create("sequences")
            .unwrap()
            .write_raw(&sequences)
            .unwrap();
    }
    hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
    let pure = hdf5_pure::File::open(&destination).unwrap();
    assert_eq!(pure.superblock().offset_size, OffsetWidth::Eight);
    assert_eq!(pure.superblock().length_size, LengthWidth::Eight);
    pure.close().unwrap();
    let file = hdf5::File::open(&destination).unwrap();
    assert_eq!(
        file.dataset("strings")
            .unwrap()
            .read_raw::<VarLenAscii>()
            .unwrap(),
        strings
    );
    assert_eq!(
        file.dataset("sequences")
            .unwrap()
            .read_raw::<VarLenArray<u16>>()
            .unwrap(),
        sequences
    );
    file.close().unwrap();
}

#[rstest]
#[case::two(Sizeof::Bytes2)]
#[case::four(Sizeof::Bytes4)]
fn repack_rejects_embedded_variable_length_references_with_narrow_offsets(
    #[case] sizeof_addr: Sizeof,
) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.h5");
    let destination = dir.path().join("destination.h5");
    {
        let file = hdf5::File::with_options()
            .with_create_plist(|properties| {
                properties.sizes(SizeofInfo {
                    sizeof_addr,
                    sizeof_size: Sizeof::Bytes8,
                })
            })
            .create(&source)
            .unwrap();
        file.new_dataset::<[VarLenAscii; 2]>()
            .shape(1)
            .create("arrays")
            .unwrap()
            .write_raw(&[[
                VarLenAscii::from_ascii("first").unwrap(),
                VarLenAscii::from_ascii("second").unwrap(),
            ]])
            .unwrap();
    }
    let error = hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap_err();
    let hdf5_pure::Error::RepackUnsupported(reason) = error else {
        panic!("expected RepackUnsupported, got {error:?}");
    };
    assert_eq!(
        reason,
        "dataset \"arrays\": embedded variable-length references with narrow offsets cannot be restaged into eight-byte offsets"
    );
    assert!(!destination.exists());
}
