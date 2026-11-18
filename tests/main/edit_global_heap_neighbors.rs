use std::fs;
use std::ops::Range;

use hdf5_pure::{AttrValue, Error, File, FileAccessProperties, FormatError, MemoryStrategy};
use rstest::rstest;
use test_util::global_heap::{self, Object};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

#[rstest]
#[case::bounded_bad_members(Damage::Members)]
#[case::unsupported_version(Damage::Version)]
#[case::unknown_size(Damage::Size)]
fn healthy_collection_reclamation_requires_a_bounded_damaged_neighbor(
    #[case] damage: Damage,
    #[values(Reclamation::Members, Reclamation::Whole)] reclamation: Reclamation,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let mut source = Fixture::new();
    let damaged = source.collection(&[Object::new(7, b"broken")]);
    damage.apply(&mut source, damaged);
    let objects = [
        Object::new(7, b"removed"),
        Object {
            index: 91,
            reference_count: 3,
            data: b"keep!",
        },
    ];
    let healthy = source.collection(if matches!(reclamation, Reclamation::Members) {
        &objects
    } else {
        &objects[..1]
    });
    let references = [("damaged", damaged, 7, 6), ("victim", healthy, 7, 7)];
    for (name, parent, index, length) in references {
        let messages = Form::CompactDataset.messages(
            &mut source,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(length, parent, index),
        );
        source.object(name, messages);
    }
    let messages = match reclamation {
        Reclamation::Members => Form::CompactDataset.messages(
            &mut source,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(5, healthy, 91),
        ),
        Reclamation::Whole => Form::CompactDataset.messages(
            &mut source,
            &hdf5_pure::make_i32_type(),
            &17_i32.to_le_bytes(),
        ),
    };
    source.object("keeper", messages);
    let path = test_util::temp::temp_path("bounded-heap-neighbor.h5");
    source.write(&path);
    let before = fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(memory),
    )
    .unwrap();
    damage.assert_error(
        file.dataset("damaged").unwrap().read_string().unwrap_err(),
        damaged,
    );
    assert_eq!(
        file.dataset("victim").unwrap().read_string().unwrap(),
        ["removed"]
    );
    reclamation.assert_value(&file);
    file.root().delete("victim").unwrap();
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    assert_excluded(&file, parent_range(damaged));
    match (damage, reclamation) {
        (Damage::Members, Reclamation::Whole) => {
            let range = parent_range(healthy);
            let free = file.reusable_free_sections().unwrap();
            assert!(
                free.sections()
                    .iter()
                    .any(|section| section.address().get() <= range.start
                        && range.end <= section.end_address().get()),
                "healthy allocation {range:?} is absent from {free:?}"
            );
        }
        _ => assert_excluded(&file, parent_range(healthy)),
    }
    file.root().set_attr("later", AttrValue::I32(23)).unwrap();
    file.commit().unwrap();
    reclamation.assert_value(&file);
    assert_excluded(&file, parent_range(damaged));
    file.close().unwrap();
    let after = fs::read(&path).unwrap();
    let damaged_bytes = bytes_range(parent_range(damaged));
    assert_eq!(&after[damaged_bytes.clone()], &before[damaged_bytes]);
    let healthy = bytes_range(parent_range(healthy));
    if matches!(damage, Damage::Members) {
        if matches!(reclamation, Reclamation::Members) {
            let original = &before[healthy.clone()];
            let mut expected = original.to_vec();
            expected[16..40].copy_from_slice(&original[40..64]);
            expected[40..56].fill(0);
            expected[48..56].copy_from_slice(&4_056_u64.to_le_bytes());
            assert_ne!(&after[healthy.clone()], original);
            assert_eq!(&after[healthy], expected);
        }
    } else {
        assert_eq!(&after[healthy.clone()], &before[healthy]);
    }
    let file = File::open(&path).unwrap();
    reclamation.assert_value(&file);
    damage.assert_error(
        file.dataset("damaged").unwrap().read_string().unwrap_err(),
        damaged,
    );
    assert!(
        file.root()
            .datasets()
            .unwrap()
            .iter()
            .all(|name| name != "victim")
    );
    assert_eq!(
        file.root().attrs().unwrap().get("later"),
        Some(&AttrValue::I32(23))
    );
}

#[derive(Clone, Copy, Debug)]
enum Damage {
    Members,
    Version,
    Size,
}

impl Damage {
    fn apply(self, source: &mut Fixture, address: u64) {
        let (offset, bytes) = match self {
            Self::Members => (24, 4_096_u64.to_le_bytes().to_vec()),
            Self::Version => (4, vec![2]),
            Self::Size => (8, 4_100_u64.to_le_bytes().to_vec()),
        };
        source
            .image
            .place(usize::try_from(address).unwrap() + offset, &bytes);
    }

    fn assert_error(self, error: Error, address: u64) {
        let Error::Format(error) = error else {
            panic!("expected a collection error, got {error:?}");
        };
        assert_eq!(
            error,
            match self {
                Self::Members => FormatError::GlobalHeapObjectOutsideCollection {
                    address: address + 16,
                    index: 7,
                    collection_end: address + 4_096
                },
                Self::Version => FormatError::InvalidGlobalHeapVersion(2),
                Self::Size => FormatError::UnalignedGlobalHeapCollection {
                    collection_address: address,
                    size: 4_100
                },
            }
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Reclamation {
    Members,
    Whole,
}

impl Reclamation {
    fn assert_value(self, file: &File) {
        match self {
            Self::Members => assert_eq!(
                file.dataset("keeper").unwrap().read_string().unwrap(),
                ["keep!"]
            ),
            Self::Whole => assert_eq!(file.dataset("keeper").unwrap().read_i32().unwrap(), [17]),
        }
    }
}

fn parent_range(address: u64) -> Range<u64> {
    address..address + u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap()
}
fn bytes_range(range: Range<u64>) -> Range<usize> {
    usize::try_from(range.start).unwrap()..usize::try_from(range.end).unwrap()
}

fn assert_excluded(file: &File, range: Range<u64>) {
    let free = file.reusable_free_sections().unwrap();
    assert!(
        free.sections()
            .iter()
            .all(|section| section.end_address().get() <= range.start
                || range.end <= section.address().get()),
        "retained allocation {range:?} overlaps {free:?}"
    );
}
