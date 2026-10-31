#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::Extent;
use hdf5::H5Type;
use hdf5::Location;
use hdf5::dataset::AllocTime;
use hdf5::file::LibraryVersion;
use hdf5::plist::GroupCreate;
use hdf5::plist::dataset_create::Layout;
use hdf5::types::VarLenArray;
use hdf5::types::VarLenUnicode;
use hdf5_pure::AddressRange;
use hdf5_pure::ChunkIndex;
use hdf5_pure::File;
use hdf5_pure::PhysicalAddress;
use ndarray::ArrayView2;
use ndarray::s;
use rstest::rstest;
use tempfile::TempDir;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::storage;

#[rstest]
fn native_collections_survive_deletion_of_another_reference(
    #[values(LibraryVersion::Earliest, LibraryVersion::V18)] low: LibraryVersion,
    #[values(
        Form::CompactAttribute,
        Form::DenseAttribute,
        Form::ContiguousDataset,
        Form::CompactDataset,
        Form::FillValue
    )]
    form: Form,
    #[values(
        Payload::Strings,
        Payload::Sequences,
        Payload::Compounds,
        Payload::ArraysOfStrings,
        Payload::ArraysOfCompounds
    )]
    payload: Payload,
) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("liveness.h5");
    let file = libhdf5_file(&path, low);
    let group = file.create_group("g").unwrap();
    match payload {
        Payload::Strings => form.write(&group, &strings()),
        Payload::Sequences => form.write(&group, &sequences()),
        Payload::Compounds => form.write(&group, &records()),
        Payload::ArraysOfStrings => form.write(&group, &[strings(), strings()]),
        Payload::ArraysOfCompounds => {
            let [first, _, last] = records();
            form.write(&group, &[[first, last]]);
        }
    }
    drop(group);
    file.close().unwrap();

    let before = {
        let file = hdf5::File::open(&path).unwrap();
        payload.read(form, &file.group("g").unwrap())
    };
    assert_eq!(before, payload.expected(form));
    assert_collections_survive_owner_deletion(&path);
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(payload.read(form, &file.group("g").unwrap()), before);
}

#[rstest]
#[case::version_1_btree(ChunkIndex::BTreeV1, false)]
#[case::filtered_version_1_btree(ChunkIndex::BTreeV1, true)]
#[case::single_chunk(ChunkIndex::SingleChunk, false)]
#[case::filtered_single_chunk(ChunkIndex::SingleChunk, true)]
#[case::implicit(ChunkIndex::Implicit, false)]
#[case::fixed_array(ChunkIndex::FixedArray, false)]
#[case::filtered_fixed_array(ChunkIndex::FixedArray, true)]
#[case::extensible_array(ChunkIndex::ExtensibleArray, false)]
#[case::filtered_extensible_array(ChunkIndex::ExtensibleArray, true)]
#[case::version_2_btree(ChunkIndex::BTreeV2, false)]
#[case::filtered_version_2_btree(ChunkIndex::BTreeV2, true)]
fn native_chunked_collections_survive_deletion_of_another_reference(
    #[case] index: ChunkIndex,
    #[case] filtered: bool,
) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("liveness_chunked.h5");
    let low = match index {
        ChunkIndex::BTreeV1 => LibraryVersion::Earliest,
        _ => LibraryVersion::V110,
    };
    let file = libhdf5_file(&path, low);
    write_chunked(&file, "strings", index, filtered, &strings());
    write_chunked(&file, "records", index, filtered, &records());
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    for name in ["strings", "records"] {
        let dataset = file.dataset(name).unwrap();
        assert_eq!(dataset.chunk_index().unwrap(), Some(index));
        let allocated = match index {
            ChunkIndex::SingleChunk => 1,
            ChunkIndex::Implicit => 4,
            _ => 2,
        };
        assert_eq!(dataset.chunks().unwrap().len(), allocated);
    }
    drop(file);
    assert_collections_survive_owner_deletion(&path);
    let file = hdf5::File::open(&path).unwrap();
    let values = file
        .dataset("strings")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap();
    assert_eq!(
        [values[0].clone(), values[1].clone(), values[6].clone()],
        strings()
    );
    let values = file
        .dataset("records")
        .unwrap()
        .read_raw::<Record>()
        .unwrap();
    assert_eq!(
        [values[0].clone(), values[1].clone(), values[6].clone()],
        records()
    );
}

#[rstest]
#[case::only_through_a_second_hard_link(Linking::SecondHardLinkOnly)]
#[case::through_two_paths(Linking::TwoHardLinks)]
#[case::beside_a_soft_link(Linking::SoftLink)]
fn linked_native_datasets_retain_their_collections_during_deletion(#[case] linking: Linking) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("liveness_links.h5");
    let file = libhdf5_file(&path, LibraryVersion::Earliest);
    let first = file.create_group("a").unwrap();
    Form::ContiguousDataset.write(&first, &["alpha".parse::<VarLenUnicode>().unwrap()]);
    file.create_group("b").unwrap();
    match linking {
        Linking::SecondHardLinkOnly => {
            file.link_hard("/a/d", "/b/d").unwrap();
            first.unlink("d").unwrap();
        }
        Linking::SoftLink => file.link_soft("/a/d", "/b/d").unwrap(),
        Linking::TwoHardLinks => file.link_hard("/a/d", "/b/d").unwrap(),
    }
    drop(first);
    file.close().unwrap();

    assert_collections_survive_owner_deletion(&path);
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("b/d")
            .unwrap()
            .read_raw::<VarLenUnicode>()
            .unwrap(),
        vec!["alpha".parse::<VarLenUnicode>().unwrap()]
    );
}

/// Writes the dataset `name` of 1 by 7 elements, compressed with deflate if `filtered` is `true`,
/// with the first two of `values` in its first two elements and the last in its last element.
///
/// From a lower library version bound of 1.10 on, libhdf5 chooses the chunk index from the shape,
/// the filters, and the allocation time (`H5D__chunk_construct` in `H5Dchunk.c`, HDF5 2.2.0), and
/// the dataset has the shape and allocation time for which it chooses `index`. In chunks of 1 by
/// 2, chunks 1 and 2 stay unallocated, except under [`ChunkIndex::Implicit`]: libhdf5 chooses it
/// only for early allocation, and allocates all four chunks when it creates the dataset. Under
/// [`ChunkIndex::SingleChunk`], the dataset is one chunk. Under [`ChunkIndex::BTreeV1`], the
/// caller creates the file with a lower bound earlier than 1.10.
fn write_chunked<T: H5Type>(
    group: &hdf5::Group,
    name: &str,
    index: ChunkIndex,
    filtered: bool,
    values: &[T; 3],
) {
    let builder = match index {
        ChunkIndex::BTreeV2 => group
            .new_dataset::<T>()
            .chunk((1, 2))
            .shape((Extent::resizable(1), Extent::resizable(7))),
        ChunkIndex::ExtensibleArray => group
            .new_dataset::<T>()
            .chunk((1, 2))
            .shape((1, Extent::resizable(7))),
        ChunkIndex::SingleChunk => group.new_dataset::<T>().chunk((1, 7)).shape((1, 7)),
        ChunkIndex::Implicit => group
            .new_dataset::<T>()
            .chunk((1, 2))
            .shape((1, 7))
            .alloc_time(Some(AllocTime::Early)),
        _ => group.new_dataset::<T>().chunk((1, 2)).shape((1, 7)),
    };
    let builder = if filtered {
        builder.deflate(6)
    } else {
        builder
    };
    let dataset = builder.create(name).unwrap();
    dataset
        .write_slice(
            ArrayView2::from_shape((1, 2), &values[..2]).unwrap(),
            s![0..1, 0..2],
        )
        .unwrap();
    dataset
        .write_slice(
            ArrayView2::from_shape((1, 1), &values[2..]).unwrap(),
            s![0..1, 6..7],
        )
        .unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Form {
    CompactAttribute,
    CompactDataset,
    ContiguousDataset,
    DenseAttribute,
    FillValue,
}

impl Form {
    fn expected<T: Clone>(self, values: &[T]) -> Vec<T> {
        if matches!(self, Self::FillValue) {
            vec![values[0].clone(); values.len()]
        } else {
            values.to_vec()
        }
    }

    fn read<T: H5Type>(self, group: &hdf5::Group) -> Vec<T> {
        match self {
            Self::CompactAttribute => group.attr("a").unwrap().read_raw::<T>().unwrap(),
            Self::DenseAttribute => group
                .group("dense")
                .unwrap()
                .attr("a")
                .unwrap()
                .read_raw::<T>()
                .unwrap(),
            Self::CompactDataset | Self::ContiguousDataset | Self::FillValue => {
                group.dataset("d").unwrap().read_raw::<T>().unwrap()
            }
        }
    }

    fn write<T: H5Type + Clone>(self, group: &hdf5::Group, values: &[T]) {
        match self {
            Self::CompactAttribute => write_attribute(group, values),
            Self::DenseAttribute => {
                let properties = GroupCreate::build()
                    .attr_phase_change(0, 0)
                    .finish()
                    .unwrap();
                let dense = group
                    .create_group_builder()
                    .set_gcpl(&properties)
                    .create("dense")
                    .unwrap();
                write_attribute(&dense, values);
            }
            Self::CompactDataset | Self::ContiguousDataset => {
                let layout = match self {
                    Self::CompactDataset => Layout::Compact,
                    _ => Layout::Contiguous,
                };
                group
                    .new_dataset::<T>()
                    .shape([values.len()])
                    .layout(layout)
                    .create("d")
                    .unwrap()
                    .write(values)
                    .unwrap();
            }
            Self::FillValue => {
                group
                    .new_dataset::<T>()
                    .shape([values.len()])
                    .fill_value(values[0].clone())
                    .create("d")
                    .unwrap();
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Linking {
    SecondHardLinkOnly,
    SoftLink,
    TwoHardLinks,
}

#[derive(Clone, Copy, Debug)]
enum Payload {
    ArraysOfCompounds,
    ArraysOfStrings,
    Compounds,
    Sequences,
    Strings,
}

impl Payload {
    fn expected(self, form: Form) -> Values {
        match self {
            Self::Strings => Values::Strings(form.expected(&strings())),
            Self::Sequences => Values::Sequences(form.expected(&sequences())),
            Self::Compounds => Values::Compounds(form.expected(&records())),
            Self::ArraysOfStrings => {
                Values::ArraysOfStrings(form.expected(&[strings(), strings()]))
            }
            Self::ArraysOfCompounds => {
                let [first, _, last] = records();
                Values::ArraysOfCompounds(form.expected(&[[first, last]]))
            }
        }
    }

    fn read(self, form: Form, group: &hdf5::Group) -> Values {
        match self {
            Self::Strings => Values::Strings(form.read(group)),
            Self::Sequences => Values::Sequences(form.read(group)),
            Self::Compounds => Values::Compounds(form.read(group)),
            Self::ArraysOfStrings => Values::ArraysOfStrings(form.read(group)),
            Self::ArraysOfCompounds => Values::ArraysOfCompounds(form.read(group)),
        }
    }
}

#[derive(Debug, PartialEq)]
enum Values {
    ArraysOfCompounds(Vec<[Record; 2]>),
    ArraysOfStrings(Vec<[VarLenUnicode; 3]>),
    Compounds(Vec<Record>),
    Sequences(Vec<VarLenArray<i32>>),
    Strings(Vec<VarLenUnicode>),
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, hdf5::H5Type)]
struct Record {
    n: i32,
    s: VarLenUnicode,
    v: VarLenArray<i32>,
}

fn strings() -> [VarLenUnicode; 3] {
    ["alpha", "", "beta"].map(|s| s.parse().unwrap())
}

fn sequences() -> [VarLenArray<i32>; 3] {
    [&[1, 2, 3][..], &[], &[4]].map(VarLenArray::from_slice)
}

fn records() -> [Record; 3] {
    let [s0, s1, s2] = strings();
    let [v0, v1, v2] = sequences();
    [
        Record { n: 1, s: s0, v: v0 },
        Record { n: 2, s: s1, v: v1 },
        Record { n: 3, s: s2, v: v2 },
    ]
}

fn write_attribute<T: H5Type>(location: &Location, values: &[T]) {
    location
        .new_attr::<T>()
        .shape([values.len()])
        .create("a")
        .unwrap()
        .write(values)
        .unwrap();
}

fn libhdf5_file(path: &Path, low: LibraryVersion) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(low, LibraryVersion::latest()))
        .create(path)
        .unwrap()
}

fn assert_collections_survive_owner_deletion(path: &Path) {
    let before = std::fs::read(path).unwrap();
    let collections = global_heap::collection_ranges(&before, Widths::EIGHT);
    assert!(!collections.is_empty());
    for (index, collection) in collections.iter().enumerate() {
        let file = File::open_rw(path).unwrap();
        file.root()
            .create_dataset("probe", |builder| {
                builder.with_vlen_strings(&["x"]);
            })
            .unwrap();
        file.commit().unwrap();
        let reference = test_util_hdf5::file_space::contiguous_extent(&file, "probe");
        file.close().unwrap();
        let mut image = std::fs::read(path).unwrap();
        let first_object = usize::try_from(collection.start).unwrap() + COLLECTION_PREFIX_LEN;
        let object = u32::from(u16::from_le_bytes(
            *image[first_object..].first_chunk().unwrap(),
        ));
        assert!(object > 0);
        let at = usize::try_from(reference.start).unwrap();
        image[at..at + global_heap_fixture::REFERENCE_SIZE].copy_from_slice(
            &global_heap::reference(1, collection.start, object, Widths::EIGHT),
        );
        std::fs::write(path, image).unwrap();
        let file = File::open_rw(path).unwrap();
        file.root().delete("probe").unwrap();
        file.commit().unwrap();
        let pressure = usize::try_from(collection.end - collection.start).unwrap();
        file.root()
            .create_dataset(&format!("pressure{index}"), |builder| {
                builder.with_u8_data(&vec![0x5a; pressure]);
            })
            .unwrap();
        file.commit().unwrap();
        storage::assert_unavailable(
            &file,
            &collections
                .iter()
                .map(|extent| {
                    AddressRange::new(
                        PhysicalAddress::new(extent.start),
                        extent.end - extent.start,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>(),
        );
        file.close().unwrap();
        let after = std::fs::read(path).unwrap();
        for extent in &collections {
            let range =
                usize::try_from(extent.start).unwrap()..usize::try_from(extent.end).unwrap();
            assert_eq!(
                &after[range.clone()],
                &before[range],
                "a surviving owner must retain its collection"
            );
        }
        let file = hdf5::File::open(path).unwrap();
        assert!(!file.link_exists("probe"));
        assert_eq!(
            file.dataset(&format!("pressure{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![0x5a; pressure]
        );
    }
}

/// The version 1 collection header size with eight-byte length fields.
///
/// Defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
const COLLECTION_PREFIX_LEN: usize = 16;
