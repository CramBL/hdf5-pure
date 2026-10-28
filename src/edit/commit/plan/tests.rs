//! Commit preparation, replacement allocation, and staged-input ownership.
//!
//! Allocation measurements cover [`PreparedCommit::prepare`] after staging and before application.

use rstest::rstest;

use super::{CommitPlan, CommitPreparation, InPlaceWrite, PreparedCommit};
#[cfg(feature = "deflate")]
use crate::DeflateLevel;
use crate::File;
use crate::edit::WriteEngine;
use crate::edit::dataset::MovingWrite;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::allocated_one_mib(
    false,
    262_144,
    "overwrite_allocated_one_mib",
    "overwrite_allocated_one_mib_control"
)]
#[case::allocated_four_mib(
    false,
    1_048_576,
    "overwrite_allocated_four_mib",
    "overwrite_allocated_four_mib_control"
)]
#[case::unallocated_one_mib(
    true,
    262_144,
    "overwrite_unallocated_one_mib",
    "overwrite_unallocated_one_mib_control"
)]
#[case::unallocated_four_mib(
    true,
    1_048_576,
    "overwrite_unallocated_four_mib",
    "overwrite_unallocated_four_mib_control"
)]
fn overwrite_preparation_allocations_do_not_scale_with_replacement_bytes(
    #[case] unallocated: bool,
    #[case] elements: usize,
    #[case] region: &str,
    #[case] control: &str,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("allocation.h5");
    let original = vec![0; elements];
    let replacement: Vec<i32> = (0..elements)
        .map(|index| i32::try_from(index).unwrap())
        .collect();
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("d");
    if unallocated {
        dataset.with_unallocated_storage(
            crate::type_builders::make_i32_type(),
            &[u64::try_from(elements).unwrap()],
        );
    } else {
        dataset.with_i32_data(&original);
    }
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), i32_dataset(&replacement))
        .unwrap();
    let bytes = engine.staged.writes()[0].1.raw();
    let length = u64::try_from(bytes.len()).unwrap();
    let pointer = bytes.as_ptr();
    let (copy, positive) =
        test_util::allocation::measure(control, || std::hint::black_box(bytes.to_vec()));
    assert_eq!(copy, bytes);
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.bytes >= length, "{positive}");
    drop(copy);
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let (prepared, measured) =
        test_util::allocation::measure(region, || PreparedCommit::prepare(&mut engine, staged));
    let CommitPreparation::Prepared(prepared) = prepared else {
        panic!("expected the replacement to prepare");
    };
    assert_eq!(prepared.overwrites.writes[0].1.raw().as_ptr(), pointer);
    assert!(measured.blocks > 0, "{measured}");
    assert!(measured.bytes < 65_536, "{measured}");
    assert!(measured.bytes < length / 8, "{measured}");
    assert_eq!(matches!(prepared.plan, CommitPlan::Tree(_)), unallocated);
    engine.apply_commit(prepared).unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), replacement);
}

#[rstest]
#[case::in_place(false)]
#[case::tree(true)]
fn prepared_applications_retain_their_own_staged_inputs(#[case] create: bool) {
    let directory = tempfile::tempdir().unwrap();
    let first_path = directory.path().join("first.h5");
    let second_path = directory.path().join("second.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("d").with_i32_data(&[1, 2]);
    builder.write(&first_path).unwrap();
    std::fs::copy(&first_path, &second_path).unwrap();
    let mut first = edit_test_util::open_engine(&first_path);
    let mut second = edit_test_util::open_engine(&second_path);
    first
        .stage_dataset_write(&ObjectPathBuf::parse("d"), i32_dataset(&[3, 4]))
        .unwrap();
    second
        .stage_dataset_write(&ObjectPathBuf::parse("d"), i32_dataset(&[5, 6]))
        .unwrap();
    if create {
        first
            .stage_created_dataset(&ObjectPathBuf::parse("first"), i32_dataset(&[7]))
            .unwrap();
        second
            .stage_created_dataset(&ObjectPathBuf::parse("second"), i32_dataset(&[8, 9]))
            .unwrap();
    }
    let first_staged = std::mem::take(&mut first.staged);
    let second_staged = std::mem::take(&mut second.staged);
    first.begin_commit(&first_staged);
    second.begin_commit(&second_staged);
    let CommitPreparation::Prepared(first_prepared) =
        PreparedCommit::prepare(&mut first, first_staged)
    else {
        panic!("expected the first staged set to prepare");
    };
    let CommitPreparation::Prepared(second_prepared) =
        PreparedCommit::prepare(&mut second, second_staged)
    else {
        panic!("expected the second staged set to prepare");
    };
    second.apply_commit(second_prepared).unwrap();
    first.apply_commit(first_prepared).unwrap();
    drop(second);
    drop(first);
    let first = File::open(&first_path).unwrap();
    let second = File::open(&second_path).unwrap();
    assert_eq!(first.dataset("d").unwrap().read_i32().unwrap(), [3, 4]);
    assert_eq!(second.dataset("d").unwrap().read_i32().unwrap(), [5, 6]);
    if create {
        assert_eq!(first.dataset("first").unwrap().read_i32().unwrap(), [7]);
        assert_eq!(
            second.dataset("second").unwrap().read_i32().unwrap(),
            [8, 9]
        );
        assert_eq!(first.root().datasets().unwrap(), ["d", "first"]);
        assert_eq!(second.root().datasets().unwrap(), ["d", "second"]);
    }
}

#[rstest]
#[case::fixed(Replacement::Fixed)]
#[case::variable_length(Replacement::VariableLength)]
fn rejected_preparation_retains_replacement_buffers_for_retry(#[case] replacement: Replacement) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rejected.h5");
    let mut builder = FileBuilder::new();
    match replacement {
        Replacement::Fixed => builder.create_dataset("d").with_i32_data(&[1, 2]),
        Replacement::VariableLength => builder
            .create_dataset("d")
            .with_vlen_strings(&["before", "values"]),
    };
    builder.create_dataset("bad").with_i32_data(&[1]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let mut values = DatasetBuilder::new("");
    match replacement {
        Replacement::Fixed => values.with_i32_data(&[3, 4]),
        Replacement::VariableLength => values.with_vlen_strings(&["replacement", "strings"]),
    };
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), values)
        .unwrap();
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("bad"), i32_dataset(&[5, 6]))
        .unwrap();
    let snapshot = |engine: &WriteEngine| {
        engine
            .staged
            .writes()
            .iter()
            .map(|(path, values)| {
                (
                    path.clone(),
                    values.raw().as_ptr(),
                    values.raw().to_vec(),
                    values.vl_string_staging().map(|staging| {
                        (
                            staging.patch_offsets.clone(),
                            staging
                                .collections
                                .iter()
                                .map(|collection| (collection.as_ptr(), collection.clone()))
                                .collect::<Vec<_>>(),
                        )
                    }),
                )
            })
            .collect::<Vec<_>>()
    };
    let expected = snapshot(&engine);
    let error = engine.commit().unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected an overwrite shape error, got {error:?}");
    };
    assert_eq!(
        reason,
        "write_dataset shape does not match the on-disk dataset (overwrite, not reshape)"
    );
    assert_eq!(snapshot(&engine), expected);
    assert_eq!(engine.page_publication_snapshot().unwrap(), before);
    let error = engine.commit().unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected the same overwrite shape error on retry, got {error:?}");
    };
    assert_eq!(
        reason,
        "write_dataset shape does not match the on-disk dataset (overwrite, not reshape)"
    );
    assert_eq!(snapshot(&engine), expected);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn empty_commit_preserves_bytes_after_mixed_overwrites() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("fixed").with_i32_data(&[1, 2]);
    builder
        .create_dataset("strings")
        .with_vlen_strings(&["before", "values"]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("fixed"), i32_dataset(&[3, 4]))
        .unwrap();
    let mut strings = DatasetBuilder::new("");
    strings.with_vlen_strings(&["replacement", "strings"]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("strings"), strings)
        .unwrap();
    engine.commit().unwrap();
    assert!(!engine.has_staged_edits());
    let committed = engine.page_publication_snapshot().unwrap();
    engine.commit().unwrap();
    assert_eq!(engine.page_publication_snapshot().unwrap(), committed);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), committed);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("fixed").unwrap().read_i32().unwrap(), [3, 4]);
    assert_eq!(
        file.dataset("strings").unwrap().read_string().unwrap(),
        ["replacement", "strings"]
    );
}

#[rstest]
#[case::fixed_in_place(Replacement::Fixed, false)]
#[case::fixed_tree(Replacement::Fixed, true)]
#[case::variable_length(Replacement::VariableLength, false)]
fn prepared_overwrites_transfer_the_original_buffers(
    #[case] replacement: Replacement,
    #[case] create: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("owned.h5");
    let mut builder = FileBuilder::new();
    match replacement {
        Replacement::Fixed => builder.create_dataset("d").with_i32_data(&[1, 2]),
        Replacement::VariableLength => builder
            .create_dataset("d")
            .with_vlen_strings(&["before", "values"]),
    };
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let mut values = DatasetBuilder::new("");
    match replacement {
        Replacement::Fixed => values.with_i32_data(&[3, 4]),
        Replacement::VariableLength => values.with_vlen_strings(&["replacement", "strings"]),
    };
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), values)
        .unwrap();
    if create {
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("new"), i32_dataset(&[5]))
            .unwrap();
    }
    let staged_values = &engine.staged.writes()[0].1;
    let original = staged_values.raw().as_ptr();
    let expected = staged_values.raw().to_vec();
    let collections = staged_values.vl_string_staging().map(|staging| {
        staging
            .collections
            .iter()
            .map(|collection| collection.as_ptr())
            .collect::<Vec<_>>()
    });
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected the overwrite to prepare");
    };
    assert_eq!(prepared.overwrites.writes[0].1.raw().as_ptr(), original);
    assert_eq!(prepared.overwrites.writes[0].1.raw(), expected);
    assert_eq!(
        prepared.overwrites.writes[0]
            .1
            .vl_string_staging()
            .map(|staging| {
                staging
                    .collections
                    .iter()
                    .map(|collection| collection.as_ptr())
                    .collect::<Vec<_>>()
            }),
        collections
    );
    let (plan, staged, mut overwrites) = prepared.into_parts();
    let source = match plan {
        CommitPlan::InPlace(writes) => {
            assert_eq!(writes.len(), 1);
            let InPlaceWrite::Replacement { source, .. } = writes.into_iter().next().unwrap()
            else {
                panic!("expected replacement bytes");
            };
            source
        }
        CommitPlan::Tree(mut tree) => match replacement {
            Replacement::Fixed => {
                assert_eq!(tree.inplace_writes.len(), 1);
                let InPlaceWrite::Replacement { source, .. } =
                    tree.inplace_writes.into_iter().next().unwrap()
                else {
                    panic!("expected replacement bytes");
                };
                source
            }
            Replacement::VariableLength => {
                let writes = &mut tree
                    .groups
                    .get_mut(&ObjectPathBuf::root())
                    .unwrap()
                    .prepared
                    .writes;
                let MovingWrite::Contiguous { bytes, .. } = writes.pop().unwrap().write else {
                    panic!("expected a relocating variable-length overwrite");
                };
                bytes
            }
        },
    };
    let bytes = engine
        .resolve_overwrite_bytes(overwrites.take(source))
        .unwrap();
    assert_eq!(bytes.as_ptr(), original);
    match replacement {
        Replacement::Fixed => assert_eq!(bytes, expected),
        Replacement::VariableLength => {
            assert_ne!(bytes, expected);
            assert_eq!(
                engine.vl_overwrite_heaps[&ObjectPathBuf::parse("d")].len(),
                collections.unwrap().len()
            );
        }
    }
    assert!(overwrites.writes[0].1.raw().is_empty());
    assert!(overwrites.writes[0].1.vl_string_staging().is_none());
    drop(staged);
    drop(engine);
}

#[cfg(feature = "deflate")]
#[rstest]
#[case::in_place(false)]
#[case::relocating(true)]
fn chunk_encoding_releases_original_replacement_buffers(#[case] filtered: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("chunks.h5");
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("d");
    dataset.with_i32_data(&vec![0; 1_024]).with_chunks(&[1_024]);
    if filtered {
        dataset.with_deflate(DeflateLevel::Six);
    }
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let expected: Vec<i32> = (0..1_024).collect();
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), i32_dataset(&expected))
        .unwrap();
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected the chunked overwrite to prepare");
    };
    assert_eq!(matches!(&prepared.plan, CommitPlan::Tree(_)), filtered);
    assert!(prepared.overwrites.writes[0].1.raw().is_empty());
    engine.apply_commit(prepared).unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
}

#[cfg(feature = "deflate")]
#[test]
fn overwrite_identifiers_select_matching_payloads_after_grouping() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ordered.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_group_with("z", |group| {
            group.create_dataset("fixed").with_i32_data(&[1, 2]);
            Ok(())
        })
        .unwrap();
    builder
        .create_group_with("a", |group| {
            group
                .create_dataset("strings")
                .with_vlen_strings(&["before", "values"]);
            Ok(())
        })
        .unwrap();
    builder
        .create_group_with("m", |group| {
            group
                .create_dataset("encoded")
                .with_i32_data(&vec![0; 1_024])
                .with_chunks(&[1_024])
                .with_deflate(DeflateLevel::Six);
            Ok(())
        })
        .unwrap();
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("z/fixed"), i32_dataset(&[3, 4]))
        .unwrap();
    let mut strings = DatasetBuilder::new("");
    strings.with_vlen_strings(&["replacement", "strings"]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("a/strings"), strings)
        .unwrap();
    let encoded: Vec<i32> = (0..1_024).collect();
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("m/encoded"), i32_dataset(&encoded))
        .unwrap();
    engine.commit().unwrap();
    assert!(!engine.has_staged_edits());
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("z/fixed").unwrap().read_i32().unwrap(), [3, 4]);
    assert_eq!(
        file.dataset("a/strings").unwrap().read_string().unwrap(),
        ["replacement", "strings"]
    );
    assert_eq!(
        file.dataset("m/encoded").unwrap().read_i32().unwrap(),
        encoded
    );
}

fn i32_dataset(values: &[i32]) -> DatasetBuilder {
    let mut builder = DatasetBuilder::new("");
    builder.with_i32_data(values);
    builder
}

#[derive(Clone, Copy)]
enum Replacement {
    Fixed,
    VariableLength,
}
