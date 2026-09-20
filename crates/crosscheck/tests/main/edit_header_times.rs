#![cfg(feature = "__hdf5-1.10")]
//! The optional prefix blocks of a version 2 object header survive an in-place
//! edit (PR #422).
//!
//! A version 2 object header may carry two blocks between its flags byte and its
//! chunk-0 size field: four timestamps (access, modification, change, birth) and
//! the attribute phase-change thresholds. The reference C library stores the
//! timestamps on **every** v2 header it writes — `H5O_CRT_OHDR_FLAGS_DEF` is
//! `H5O_HDR_STORE_TIMES` — so this reaches every file libhdf5, h5py or netCDF-4
//! produces in the latest format, and `EditSession` used to drop both blocks on
//! any rewrite, leaving `H5Oget_info` reporting four zeroed times.
//!
//! The fixture has to come from the C library: nothing in this crate's whole-file
//! writer emits either block, so a fixture written here would only test the
//! editor against itself. And the assertion has to come from the C library too —
//! this crate exposes no timestamp reader at all, so the C library's object info
//! is the only thing that can say the file still means what it meant.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use hdf5::plist::group_create::{AttrPhaseChange, GroupCreate, GroupCreateBuilder};
use hdf5_pure::{AttrValue, File};
use tempfile::tempdir;

use hdf5_pure_crosscheck::create_v18;

/// A phase-change pair the C library would never write by default (its defaults
/// are 8 and 6), and one it therefore stores in the header prefix rather than
/// leaving implied.
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

/// A file with a group `/g` and a dataset `/d`, each carrying one integer
/// attribute and a non-default attribute phase-change pair.
///
/// Nothing here asks for timestamps: the C library stores them on every version 2
/// header it writes, which is the whole point.
fn write_fixture(path: &Path) {
    let file = create_v18(path);
    let group = file
        .create_group_builder()
        .set_gcpl(&gcpl_with_phase_change())
        .create("g")
        .expect("create group");
    // The wrapper's dataset builder turns time tracking off. The C library's
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
    let dir = tempdir().unwrap();
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
