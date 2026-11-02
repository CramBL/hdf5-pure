#![cfg(feature = "__hdf5-1.10")]
//! Native checks of version 2 timestamps and attribute thresholds across edits and repacking.
//!
//! Fixtures are created and inspected through libhdf5.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use hdf5::plist::group_create::{AttrPhaseChange, GroupCreate, GroupCreateBuilder};
use hdf5_pure::{AttrValue, File};

use test_util_hdf5::file;

/// The custom attribute thresholds stored on the fixture objects.
const PHASE_CHANGE: AttrPhaseChange = AttrPhaseChange {
    max_compact: 32,
    min_dense: 24,
};

const OBJECTS: [&str; 2] = ["/g", "/d"];

/// A group creation property list carrying the phase-change pair.
fn gcpl_with_phase_change() -> GroupCreate {
    GroupCreateBuilder::new()
        .attr_phase_change(PHASE_CHANGE.max_compact, PHASE_CHANGE.min_dense)
        .finish()
        .expect("a group creation property list")
}

/// Creates a group and dataset with timestamps, one integer attribute, and custom thresholds.
fn write_fixture(path: &Path) {
    let file = file::libhdf5_create_v18(path);
    let group = file
        .create_group_builder()
        .set_gcpl(&gcpl_with_phase_change())
        .create("g")
        .expect("create group");
    // The wrapper's dataset builder turns time tracking off. libhdf5's
    // default is on, and that default is what this fixture is about.
    let dataset = file
        .new_dataset::<i32>()
        .obj_track_times(true)
        .with_dcpl(|p| p.attr_phase_change(PHASE_CHANGE.max_compact, PHASE_CHANGE.min_dense))
        .shape([4])
        .create("d")
        .expect("create dataset");
    let owners: [&hdf5::Location; 2] = [&group, &dataset];
    for owner in owners {
        owner
            .new_attr::<i32>()
            .shape(())
            .create("kept")
            .expect("create attribute")
            .write_scalar(&1i32)
            .expect("write attribute");
    }
    drop(dataset);
    drop(group);
    file.close().unwrap();
}

/// The object info the C library reports for `object` in `path`.
fn object_info(path: &Path, object: &str) -> hdf5::LocationInfo {
    hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()))
        .loc_info_by_name(object)
        .unwrap_or_else(|e| panic!("the C library reads info for {object}: {e}"))
}

/// The attribute phase-change thresholds the C library reports for `object`'s
/// creation property list.
fn phase_change(path: &Path, object: &str) -> AttrPhaseChange {
    let file = hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
    if object == "/g" {
        file.group(object)
            .unwrap()
            .gcpl()
            .unwrap()
            .attr_phase_change()
    } else {
        file.dataset(object)
            .unwrap()
            .dcpl()
            .unwrap()
            .attr_phase_change()
    }
}

fn now_secs() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the test host's clock is after 1970")
            .as_secs(),
    )
    .expect("seconds since the epoch fit an i64")
}

/// Editing an attribute in place rebuilds both object headers. Their four
/// timestamps and their attribute phase-change thresholds have to come out the
/// other side, with the modification and change times moved to the edit.
#[test]
fn an_in_place_edit_keeps_a_headers_times_and_phase_change_thresholds() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_fixture(&p);

    let before: Vec<hdf5::LocationInfo> = OBJECTS.iter().map(|o| object_info(&p, o)).collect();
    for (object, info) in OBJECTS.iter().zip(&before) {
        assert!(
            info.btime > 0,
            "{object} was written without timestamps, so this fixture proves nothing: {info:?}",
        );
        assert_eq!(
            phase_change(&p, object),
            PHASE_CHANGE,
            "{object} was written without the phase-change block",
        );
    }

    // The stored timestamps are whole seconds, so the fixture's own times have to
    // land in a strictly earlier second than the edit for "the edit moved this
    // one" and "the edit left that one alone" to be different claims. One second
    // of wall clock buys both; there is no other way to age a time the C library
    // stamps itself.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let edit_started = now_secs();

    {
        let s = File::open_rw(&p).expect("the editor opens the fixture");
        s.group("g")
            .expect("the group is reachable")
            .set_attr("added", AttrValue::I64(7))
            .expect("the group takes an attribute");
        s.dataset("d")
            .expect("the dataset is reachable")
            .set_attr("added", AttrValue::I64(7))
            .expect("the dataset takes an attribute");
        s.commit().expect("the commit lands");
    }
    let edit_ended = now_secs();

    for (object, was) in OBJECTS.iter().zip(&before) {
        let now = object_info(&p, object);
        assert_eq!(
            now.btime, was.btime,
            "{object} lost or moved its birth time; the header prefix was dropped",
        );
        assert_eq!(
            now.atime, was.atime,
            "{object} moved its access time, which a rewrite is not",
        );
        assert!(
            (edit_started..=edit_ended).contains(&now.mtime),
            "{object} reports modification time {} outside the edit's [{edit_started}, {edit_ended}]",
            now.mtime,
        );
        assert!(
            (edit_started..=edit_ended).contains(&now.ctime),
            "{object} reports change time {} outside the edit's [{edit_started}, {edit_ended}]",
            now.ctime,
        );
        assert_eq!(
            phase_change(&p, object),
            PHASE_CHANGE,
            "{object} lost the attribute phase-change thresholds the header stored",
        );
    }

    // The edit itself did what it said, so none of the above is a verdict on a
    // file that quietly lost its attributes.
    let f = File::open(&p).unwrap();
    for attrs in [
        f.group("g").unwrap().attrs().unwrap(),
        f.dataset("d").unwrap().attrs().unwrap(),
    ] {
        assert_eq!(attrs.get("added"), Some(&AttrValue::I64(7)));
        assert_eq!(attrs.get("kept"), Some(&AttrValue::I32(1)));
    }
}

#[test]
fn attribute_lifecycles_preserve_timestamps_through_repack() {
    let directory = tempfile::tempdir().unwrap();
    let mut source = directory.path().join("source.h5");
    write_fixture(&source);
    let births: Vec<_> = OBJECTS
        .iter()
        .map(|object| object_info(&source, object).btime)
        .collect();
    assert!(births.iter().all(|time| *time > 0));
    let mut previous = 1;
    for (stage, count) in [9, 32, 33, 24, 23, 33].into_iter().enumerate() {
        {
            let file = File::open_rw(&source).unwrap();
            for index in previous.min(count)..previous.max(count) {
                let name = format!("value{index:03}");
                if count > previous {
                    let value = AttrValue::I64(i64::try_from(index).unwrap());
                    file.group("g")
                        .unwrap()
                        .set_attr(&name, value.clone())
                        .unwrap();
                    file.dataset("d").unwrap().set_attr(&name, value).unwrap();
                } else {
                    file.group("g").unwrap().remove_attr(&name).unwrap();
                    file.dataset("d").unwrap().remove_attr(&name).unwrap();
                }
            }
            file.commit().unwrap();
            file.close().unwrap();
        }
        let before: Vec<_> = OBJECTS
            .iter()
            .map(|object| object_info(&source, object))
            .collect();
        let destination = directory.path().join(format!("stage{stage}.h5"));
        hdf5_pure::repack(&source, &destination, &hdf5_pure::RepackOptions::new()).unwrap();
        for ((object, was), birth) in OBJECTS.iter().zip(before).zip(&births) {
            let now = object_info(&destination, object);
            assert_eq!(now.num_attrs, count);
            assert_eq!(
                (now.atime, now.mtime, now.ctime, now.btime),
                (was.atime, was.mtime, was.ctime, *birth)
            );
            assert_eq!(phase_change(&destination, object), PHASE_CHANGE);
        }
        {
            let pure = File::open(&destination).unwrap();
            let native = hdf5::File::open(&destination).unwrap();
            for object in OBJECTS {
                let attributes = if object == "/g" {
                    pure.group(object).unwrap().attrs().unwrap()
                } else {
                    pure.dataset(object).unwrap().attrs().unwrap()
                };
                let expected: HashMap<_, _> = std::iter::once(("kept".into(), AttrValue::I32(1)))
                    .chain((1..count).map(|index| {
                        (
                            format!("value{index:03}"),
                            AttrValue::I64(i64::try_from(index).unwrap()),
                        )
                    }))
                    .collect();
                assert_eq!(attributes, expected);
                let location: hdf5::Location = if object == "/g" {
                    native.group(object).unwrap().as_location().unwrap()
                } else {
                    native.dataset(object).unwrap().as_location().unwrap()
                };
                assert_eq!(
                    location.attr("kept").unwrap().read_scalar::<i32>().unwrap(),
                    1
                );
                for index in 1..count {
                    assert_eq!(
                        location
                            .attr(&format!("value{index:03}"))
                            .unwrap()
                            .read_scalar::<i64>()
                            .unwrap(),
                        i64::try_from(index).unwrap()
                    );
                }
            }
            assert_eq!(pure.dataset("d").unwrap().read_i32().unwrap(), [0; 4]);
            assert_eq!(
                native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
                [0; 4]
            );
            native.close().unwrap();
            pure.close().unwrap();
        }
        source = destination;
        previous = count;
    }
}
