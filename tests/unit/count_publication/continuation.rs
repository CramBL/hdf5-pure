use std::fs;

use rstest::rstest;
use test_util::bytes;
use test_util::data_layout;
use test_util::dataspace;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp, IoEvent};
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::{Message, MessageType, v2};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::{Error, File, FileAccessProperties, FileLocking, MemoryStrategy};

#[rstest]
#[case::two_bytes(Widths::new(2, 2))]
#[case::four_bytes(Widths::FOUR)]
fn continued_count_decrease_uses_physical_addresses_and_preserves_the_allocation(
    #[case] widths: Widths,
) {
    const BASE: usize = 512;
    let path = test_util::temp::temp_path("continued-count.h5");
    let userblock = [0xa5; BASE];
    let mut image = Image::starting_with(&userblock);
    image.append(&[0; 64]);
    let original = count_chunk(2);
    let continuation = image.append(&original);
    let header = dataset_header(17.0, continuation - BASE, &original, widths);
    let target = image.append(&header);
    let mut info = vec![0, 0];
    bytes::push_undefined_address(&mut info, widths.offset);
    bytes::push_undefined_address(&mut info, widths.offset);
    let root = image.append(
        &v2::Header::new()
            .messages([
                Message::new(MessageType::LINK_INFO, &info),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("alias", (target - BASE) as u64).build(widths),
                ),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("kept", (target - BASE) as u64).build(widths),
                ),
            ])
            .build(),
    );
    let superblock = Superblock::new(widths)
        .version(3)
        .base_address(BASE as u64)
        .root_header_address((root - BASE) as u64)
        .eof_address(image.as_bytes().len() as u64)
        .build();
    image.place(BASE, &superblock);
    fs::write(&path, image.as_bytes()).unwrap();

    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
    )
    .unwrap();
    assert_eq!(file.superblock().base_address.get(), BASE as u64);
    assert_eq!(
        file.dataset("kept").unwrap().header_address().unwrap(),
        target as u64
    );
    assert_eq!(file.dataset("alias").unwrap().read_f64().unwrap(), [17.0]);

    file.root().delete("alias").unwrap();
    disk_log::start();
    file.commit().unwrap();
    file.sync().unwrap();
    let recording = disk_log::finish();
    let published = recording
        .operations
        .iter()
        .position(|op| write_covers(op, BASE, superblock.len()))
        .unwrap();
    for address in [continuation + 9, continuation + original.len() - 4] {
        assert!(
            recording.operations[..=published]
                .iter()
                .all(|op| !write_covers(op, address, 4))
        );
        assert!(
            recording.operations[published + 1..]
                .iter()
                .any(|op| write_covers(op, address, 4))
        );
    }
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| {
                section.end_address().get() <= continuation as u64
                    || section.address().get() >= (continuation + original.len()) as u64
            })
    );
    file.close().unwrap();

    let after = fs::read(&path).unwrap();
    assert_eq!(after[..BASE], userblock);
    assert_eq!(after[target..target + header.len()], header);
    assert_eq!(
        after[continuation..continuation + original.len()],
        count_chunk(1)
    );
    let reopened = File::open(&path).unwrap();
    assert_eq!(
        usize::from(reopened.superblock().offset_size.get()),
        widths.offset
    );
    assert_eq!(
        usize::from(reopened.superblock().length_size.get()),
        widths.length
    );
    assert_eq!(reopened.root().datasets().unwrap(), ["kept"]);
    assert_eq!(
        reopened.dataset("kept").unwrap().read_f64().unwrap(),
        [17.0]
    );
}

#[rstest]
#[case::overlapping_continuation(true)]
#[case::separate_continuations(false)]
fn count_decreases_require_separate_continuation_allocations(
    #[case] shared: bool,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("count-continuation-overlap.h5");
    let original = count_chunk(2);
    let mut fixture = Fixture::new();
    let first = fixture.image.append(&original);
    let second = if shared {
        first
    } else {
        fixture.image.append(&original)
    };
    let continuations = [first, second];
    let headers = [
        dataset_header(17.0, first, &original, Widths::EIGHT),
        dataset_header(29.0, second, &original, Widths::EIGHT),
    ];
    let targets = headers
        .each_ref()
        .map(|header| fixture.image.append(header));
    for (names, address) in [(["a", "b"], targets[0]), (["c", "d"], targets[1])] {
        for name in names {
            fixture.link(name, u64::try_from(address).unwrap());
        }
    }
    fixture.write(&path);

    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_memory_strategy(memory),
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    assert_values(&file, targets, true);
    for at in continuations {
        assert_eq!(&before[at..at + original.len()], original);
    }

    file.root().delete("a").unwrap();
    file.root().delete("c").unwrap();
    assert!(file.has_staged_edits());
    let expected = if shared {
        for _ in 0..2 {
            disk_log::start();
            let error = file.commit().unwrap_err();
            let recording = disk_log::finish();
            let Error::EditUnsupported(reason) = error else {
                panic!("unexpected count conflict: {error:?}");
            };
            assert_eq!(
                reason,
                "the original reference-count updates cannot be protected"
            );
            assert!(
                recording
                    .events
                    .iter()
                    .all(|event| !matches!(event, IoEvent::Write { .. }))
            );
            assert!(recording.operations.is_empty());
            assert!(file.has_staged_edits());
            assert_eq!(file.reusable_free_sections().unwrap(), free);
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_values(&file, targets, true);
        }
        drop(file);
        original
    } else {
        disk_log::start();
        file.commit().unwrap();
        file.sync().unwrap();
        let recording = disk_log::finish();
        assert!(!file.has_staged_edits());
        for at in continuations {
            assert!(
                recording
                    .operations
                    .iter()
                    .any(|op| write_covers(op, at + 9, 4))
            );
            assert!(
                file.reusable_free_sections()
                    .unwrap()
                    .sections()
                    .iter()
                    .all(|section| {
                        section.end_address().get() <= u64::try_from(at).unwrap()
                            || section.address().get()
                                >= u64::try_from(at + original.len()).unwrap()
                    })
            );
        }
        file.close().unwrap();
        count_chunk(1)
    };

    let after = fs::read(&path).unwrap();
    if shared {
        assert_eq!(after, before);
    }
    for (target, header) in targets.into_iter().zip(headers) {
        assert_eq!(&after[target..target + header.len()], header);
    }
    for at in continuations {
        assert_eq!(&after[at..at + expected.len()], expected);
    }
    let reopened = File::open(&path).unwrap();
    assert_values(&reopened, targets, shared);
}

fn assert_values(file: &File, targets: [usize; 2], aliases: bool) {
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(
        names,
        if aliases {
            &["a", "b", "c", "d"][..]
        } else {
            &["b", "d"]
        }
    );
    let bindings = [
        (targets[0], "b", 17.0),
        (targets[1], "d", 29.0),
        (targets[0], "a", 17.0),
        (targets[1], "c", 29.0),
    ];
    let count = if aliases { bindings.len() } else { 2 };
    for &(target, name, value) in &bindings[..count] {
        let dataset = file.dataset(name).unwrap();
        assert_eq!(
            dataset.header_address().unwrap(),
            u64::try_from(target).unwrap()
        );
        assert_eq!(dataset.read_f64().unwrap(), [value]);
    }
}

fn dataset_header(value: f64, continuation: usize, chunk: &[u8], widths: Widths) -> Vec<u8> {
    v2::Header::new()
        .messages([
            Message::new(MessageType::DATASPACE, &dataspace::scalar()),
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
            Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::compact(&value.to_le_bytes()),
            ),
        ])
        .continuation(continuation, chunk, widths)
        .build()
}

fn write_covers(operation: &DiskOp, address: usize, length: usize) -> bool {
    matches!(operation, DiskOp::Write { offset, bytes }
        if *offset <= address as u64
            && (address + length) as u64 <= *offset + bytes.len() as u64)
}

fn count_chunk(count: u32) -> Vec<u8> {
    let mut padding = Message::nil(15);
    padding.data.fill(0xa7);
    v2::continuation_chunk(
        &[Message::reference_count(count), padding],
        v2::HeaderFlags::default(),
    )
}
