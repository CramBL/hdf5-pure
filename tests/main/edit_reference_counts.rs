use std::path::Path;

use hdf5_pure::{Datatype, Error, File, FixedPointType};
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::v2;
use test_util_hdf5::global_heap_fixture::{Fixture, Form};

#[rstest]
#[case::zero(0, 1)]
#[case::undercount(2, 3)]
fn reference_count_contradictions_report_the_physical_object_and_counts(
    #[case] observed: u32,
    #[case] minimum: u64,
) {
    let error = Error::InconsistentReferenceCount {
        address: 2048,
        observed,
        minimum,
    };
    assert_eq!(
        error.to_string(),
        format!("object at 2048 has reference count {observed}, below the known minimum {minimum}")
    );
}

#[rstest]
#[case::final_aliases(2, 2, true, false)]
#[case::surviving_alias(2, 1, false, false)]
#[case::excess_count(3, 2, false, false)]
#[case::ambiguous_count(2, 2, false, true)]
fn public_commits_retire_only_zero_count_allocations(
    #[case] original: u32,
    #[case] removed: usize,
    #[case] retired: bool,
    #[case] ambiguous: bool,
) {
    let path = test_util::temp::temp_path("public-final-aliases.h5");
    let (raw, header, length) = aliases(&path, original, ambiguous);
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    for name in &["a", "b"][..removed] {
        file.root().delete(name).unwrap();
    }
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    let sections = file.reusable_free_sections().unwrap();
    assert_eq!(
        sections.sections().iter().any(|section| {
            section.address().get() <= raw
                && raw + 4_096 <= section.address().get() + section.length().get()
        }),
        retired
    );
    if !retired {
        assert!(sections.sections().iter().all(|section| {
            section.address().get() + section.length().get() <= raw
                || section.address().get() >= raw + 4_096
        }));
    }
    if removed == 1 {
        assert_eq!(file.dataset("b").unwrap().read_i32().unwrap(), [7; 1_024]);
    }
    assert_eq!(file.dataset("survivor").unwrap().read_i32().unwrap(), [11]);
    file.close().unwrap();
    if !retired {
        let after = std::fs::read(&path).unwrap();
        let mut expected = before.clone();
        if !ambiguous {
            v2::replace_message(
                &mut expected,
                usize::try_from(header).unwrap(),
                test_util::object_header::MessageType::OBJECT_REFERENCE_COUNT,
                &Message::reference_count(original - removed as u32).data,
            );
        }
        let start = usize::try_from(header).unwrap();
        assert_eq!(
            &after[start..start + length],
            &expected[start..start + length]
        );
    }
}

#[rstest]
#[case::zero(0)]
#[case::undercount(1)]
fn public_count_contradictions_preserve_bytes_and_staged_retry(#[case] original: u32) {
    let path = test_util::temp::temp_path("public-count-contradiction.h5");
    let (_, header, _) = aliases(&path, original, false);
    let before = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root().delete("a").unwrap();
    file.root().delete("b").unwrap();
    let free = file.reusable_free_sections().unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(error, Error::InconsistentReferenceCount { address, observed, minimum }
            if (address, observed, minimum) == (header, original, 2)),
        "{error:?}"
    );
    assert!(file.has_staged_edits());
    assert_eq!(file.reusable_free_sections().unwrap(), free);
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), [7; 1_024]);
    assert_eq!(file.dataset("b").unwrap().read_i32().unwrap(), [7; 1_024]);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

fn aliases(path: &Path, count: u32, ambiguous: bool) -> (u64, u64, usize) {
    let mut fixture = Fixture::new();
    let raw = fixture.image.as_bytes().len() as u64;
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &vec![7_i32.to_le_bytes(); 1_024].concat(),
    );
    messages.insert(0, Message::reference_count(count));
    if ambiguous {
        messages.push(Message::reference_count(count));
    }
    let header = fixture.append_header(messages);
    let length = fixture.image.as_bytes().len() - usize::try_from(header).unwrap();
    fixture.link("a", header);
    fixture.link("b", header);
    let survivor = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &11_i32.to_le_bytes(),
    );
    fixture.object("survivor", survivor);
    fixture.write(path);
    (raw, header, length)
}
