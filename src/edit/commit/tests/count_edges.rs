use std::fs;

use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::members::ParentImage;

use crate::{AttrValue, Error, File, FileAccessProperties, FileLocking};

#[rstest]
#[case::overflow(CountEdge::Overflow)]
#[case::cancelled_addition(CountEdge::Cancelled)]
#[case::original_contradiction(CountEdge::Contradiction)]
fn committed_type_copies_check_original_evidence_and_final_net_counts(#[case] edge: CountEdge) {
    let path = test_util::temp::temp_path("copy-count-edges.h5");
    let original = if matches!(edge, CountEdge::Contradiction) {
        1
    } else {
        u32::MAX
    };
    let target = super::fixture_counts(
        &path,
        true,
        super::published_counts::Version::Two,
        1,
        original,
    )[0];
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_page_buffer_size(0),
    )
    .unwrap();
    let before = fs::read(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    let parent = global_heap::collection_ranges(&before, Widths::EIGHT)[0].start;
    let expected = super::type_header(original);
    let extent = target as usize..target as usize + expected.len();
    assert_eq!(&before[extent.clone()], expected);
    let mut types = file.root().named_datatypes().unwrap();
    types.sort();
    assert_eq!(types, ["alias", "type"]);
    assert_eq!(
        file.root().named_datatype_references("type").unwrap(),
        original
    );

    file.copy("source", "copy").unwrap();
    if !matches!(edge, CountEdge::Overflow) {
        file.root().delete("alias").unwrap();
    }
    file.dataset("heap")
        .unwrap()
        .set_attr("a", AttrValue::I64(9))
        .unwrap();
    let committed = matches!(edge, CountEdge::Cancelled);
    if committed {
        file.commit().unwrap();
        assert!(!file.has_staged_edits());
        let after = fs::read(&path).unwrap();
        assert_eq!(&after[extent], expected);
        ParentImage::read(&before, parent).assert_unchanged(&after);
        assert_eq!(
            file.root().named_datatype_references("type").unwrap(),
            u32::MAX
        );
        file.close().unwrap();
    } else {
        for _ in 0..2 {
            let error = file.commit().unwrap_err();
            match (edge, &error) {
                (CountEdge::Overflow, Error::EditUnsupported(reason)) => {
                    assert_eq!(
                        *reason,
                        "the planned reference count does not fit its stored field"
                    );
                }
                (
                    CountEdge::Contradiction,
                    Error::InconsistentReferenceCount {
                        address,
                        observed,
                        minimum,
                    },
                ) => assert_eq!((*address, *observed, *minimum), (target, 1, 2)),
                _ => panic!("unexpected count error: {error:?}"),
            }
            assert!(file.has_staged_edits());
            assert_eq!(fs::read(&path).unwrap(), before);
            assert_eq!(file.reusable_free_sections().unwrap(), free);
        }
        drop(file);
    }

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.root().named_datatype_references("type").unwrap(),
        original
    );
    let mut types = file.root().named_datatypes().unwrap();
    types.sort();
    assert_eq!(
        types,
        if committed {
            vec!["type"]
        } else {
            vec!["alias", "type"]
        }
    );
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(
        names,
        if committed {
            vec!["copy", "dead", "heap", "keep", "source", "value"]
        } else {
            vec!["dead", "heap", "keep", "source", "value"]
        }
    );
    for name in if committed {
        vec!["source", "value", "copy"]
    } else {
        vec!["source", "value"]
    } {
        assert_eq!(file.dataset(name).unwrap().read_f64().unwrap(), [17.0]);
    }
    assert_eq!(
        file.dataset("dead").unwrap().read_i32().unwrap(),
        [7; 1_024]
    );
    assert_eq!(file.dataset("heap").unwrap().read_i32().unwrap(), [17]);
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["retained"]
    );
    assert_eq!(
        file.dataset("heap").unwrap().attrs().unwrap()["a"],
        if committed {
            AttrValue::I64(9)
        } else {
            AttrValue::VarLenAsciiStringArray(vec!["removed".into()])
        }
    );
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum CountEdge {
    Cancelled,
    Contradiction,
    Overflow,
}
