use std::collections::HashMap;
use std::fs;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;
use test_util_hdf5::global_heap_fixture::members::Buffering;
use test_util_hdf5::global_heap_fixture::members::ParentImage;

use super::Fault;
use super::HeapScenario;
use super::Opening;
use super::Retry;
use crate::AttrValue;
use crate::EditBacking;
use crate::File;
use crate::FileSpaceSettings;
use crate::SyncPolicy;

impl Fault {
    pub(in crate::edit) fn run_native(
        self,
        settings: FileSpaceSettings,
        backing: EditBacking,
        buffering: Buffering,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let control = directory.path().join("control.h5");
        let path = directory.path().join("failure.h5");
        __private::write_heap_failure_source(&control, settings);
        let original = fs::read(&control).unwrap();
        fs::write(&path, &original).unwrap();
        let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
        let [range] = ranges.as_slice() else {
            panic!("expected one shared parent: {ranges:?}");
        };
        let parent = ParentImage::read(&original, range.start);
        assert_eq!(parent.indices().len(), 3);
        let retained = parent.index_for(b"retained");
        let fixture = NativeScenario {
            retained,
            parent,
            original,
            parents: [range.start],
            settings,
        };
        self.execute(
            &fixture,
            &fixture,
            &control,
            &path,
            Opening::Selected(backing, buffering, SyncPolicy::OnClose),
            Retry::FreshEdit,
        );
        directory.close().unwrap();
    }
}

struct NativeScenario {
    parent: ParentImage,
    retained: u16,
    original: Vec<u8>,
    parents: [u64; 1],
    settings: FileSpaceSettings,
}

impl HeapScenario for NativeScenario {
    fn parent_range(&self, index: usize) -> FileRange {
        assert_eq!(index, 0);
        self.parent.range()
    }
    fn assert_parents(&self, image: &[u8], completed: usize) {
        assert!(completed <= 1);
        if completed == 1 {
            self.parent.assert_compacted(image, &[self.retained]);
        } else {
            self.parent.assert_unchanged(image);
        }
    }
    fn parents(&self) -> &[u64] {
        &self.parents
    }
    fn original(&self) -> &[u8] {
        &self.original
    }
    fn stage(&self, file: &File) {
        stage(file);
    }
    fn assert_open(&self, file: &File) {
        assert_strategy(file, self.settings);
    }
    fn assert_values(&self, file: &File, published: bool, number: i32, after: Option<i64>) {
        assert_values(file, published, number, after);
    }
    fn advance(&self, _file: &File) {}
    fn fresh_number(&self, number: i32) -> i32 {
        number
    }
    fn assert_closed(&self, path: &Path, published: bool, number: i32, after: Option<i64>) {
        let native = file::libhdf5_open(path);
        assert_eq!(
            native
                .dataset("numbers")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [number]
        );
        assert_eq!(
            attributes::ascii_values(&native.attr("third").unwrap()),
            ["retained"]
        );
        if let Some(expected) = after {
            assert_eq!(
                native.attr("after").unwrap().read_scalar::<i64>().unwrap(),
                expected
            );
        }
        let mut names = native.attr_names().unwrap();
        names.sort();
        assert_eq!(
            names,
            if published {
                if after.is_some() {
                    vec!["after", "third"]
                } else {
                    vec!["third"]
                }
            } else if after.is_some() {
                vec!["after", "first", "second", "third"]
            } else {
                vec!["first", "second", "third"]
            }
        );
        native.close().unwrap();
    }
}

/// Stages removal of the two obsolete attributes and replacement of numeric data with 23.
///
/// # Panics
///
/// Panics if a removal or numeric write cannot be staged.
pub(in crate::edit) fn stage(file: &File) {
    file.root().remove_attr("first").unwrap();
    file.root().remove_attr("second").unwrap();
    file.dataset("numbers").unwrap().write(&[23_i32]).unwrap();
}

fn assert_strategy(file: &File, expected: FileSpaceSettings) {
    let settings = file.file_space_settings().unwrap().unwrap();
    assert_eq!(settings, expected);
}

/// Checks numeric data and the complete root attribute map for the publication state.
///
/// `published` selects removal of the original attributes, and `after` is the optional fresh
/// scalar attribute value.
pub(in crate::edit) fn assert_values(
    file: &File,
    published: bool,
    number: i32,
    after: Option<i64>,
) {
    assert_eq!(
        file.dataset("numbers").unwrap().read_i32().unwrap(),
        [number]
    );
    let mut expected = HashMap::from([(
        "third".into(),
        AttrValue::VarLenAsciiString("retained".into()),
    )]);
    if !published {
        expected.insert(
            "first".into(),
            AttrValue::VarLenAsciiString("removed".into()),
        );
        expected.insert(
            "second".into(),
            AttrValue::VarLenAsciiString("discard".into()),
        );
    }
    if let Some(value) = after {
        expected.insert("after".into(), AttrValue::I64(value));
    }
    assert_eq!(file.root().attrs().unwrap(), expected);
}
