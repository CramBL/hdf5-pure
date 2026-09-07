// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! In-place edits of objects that track attribute creation order (issue #416).
//!
//! `H5Pset_attr_creation_order` — h5py's `track_order=True`, and what netCDF-4
//! sets on every object it writes — makes every object-header message record six
//! bytes wide instead of four, carrying a creation index. `EditSession` used to
//! refuse such an object outright.
//!
//! The fixtures have to come from the reference C library: nothing in this
//! crate's whole-file writer emits creation order, so a fixture written here
//! would only test the editor against itself. And the interesting assertion is
//! not "the attributes are still there" — this crate's own reader ignores
//! creation order entirely, and would say yes to a file whose indexes are
//! nonsense. It is what the C library makes of the order afterwards:
//! `H5Aiterate2` over `H5_INDEX_CRT_ORDER` (h5py's `track_order` iteration) and
//! `H5Aget_info_by_name`'s `corder`, which name the exact index each attribute
//! carries.
//!
//! The tail of the file covers the same ground for **link** creation order, the
//! separate mechanism `H5Pset_link_creation_order` turns on: there the judges
//! are `H5Literate2` over `H5_INDEX_CRT_ORDER`, `H5Lget_info2`'s `corder`, and
//! `H5Gget_info`'s `max_corder`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::AttrCreationOrder;
use hdf5::plist::file_create::FileCreateBuilder;
use hdf5::plist::group_create::GroupCreate;
use hdf5_pure::{AttrValue, Error, File};
use hdf5_sys::h5::H5_index_t::{self, H5_INDEX_CRT_ORDER, H5_INDEX_NAME};
use hdf5_sys::h5::H5_iter_order_t::H5_ITER_INC;
use hdf5_sys::h5::{herr_t, hsize_t};
use hdf5_sys::h5a::{
    H5A_info_t, H5Aclose, H5Aget_info_by_name, H5Aget_name, H5Aiterate2, H5Aopen_by_idx,
};
use hdf5_sys::h5g::{H5G_info_t, H5Gget_info};
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5l::{H5L_info_t, H5Literate};
use hdf5_sys::h5p::{H5P_DEFAULT, H5Pset_attr_creation_order, H5Pset_link_creation_order};
use tempfile::tempdir;

// The raw calls below, each marked with the upstream issue that would remove
// it, bypass the lock the wrapper serializes its own calls through. Every
// C-library use in this file takes this guard, so a raw call never races a
// wrapper call on another test thread. Poisoning is ignored: a panic in one
// test must not cascade into the others.
static C_LIB: Mutex<()> = Mutex::new(());

fn c_lib_guard() -> MutexGuard<'static, ()> {
    C_LIB.lock().unwrap_or_else(|e| e.into_inner())
}

fn cstr(s: &str) -> CString {
    CString::new(s).expect("a test name holds no NUL")
}

/// Whether the object creation property list should also index creation order,
/// which is what adds the creation-order B-tree once attributes go dense.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Indexed {
    No,
    Yes,
}

impl Indexed {
    fn flags(self) -> AttrCreationOrder {
        match self {
            Self::No => AttrCreationOrder::TRACKED,
            Self::Yes => AttrCreationOrder::TRACKED | AttrCreationOrder::INDEXED,
        }
    }
}

/// A group creation property list tracking attribute creation order, and link
/// creation order too when `links`.
fn tracking_gcpl(indexed: Indexed, links: bool) -> GroupCreate {
    let plist = GroupCreate::try_new().expect("a group creation property list");
    let flags = indexed.flags().bits();
    // TODO: https://github.com/metno/hdf5-rust/issues/229 and https://github.com/metno/hdf5-rust/issues/230
    // Safety: a live property list id and the flags the C library defines.
    unsafe {
        assert_eq!(H5Pset_attr_creation_order(plist.id(), flags), 0);
        if links {
            assert_eq!(H5Pset_link_creation_order(plist.id(), flags), 0);
        }
    }
    plist
}

/// Write a file whose group `/g` and dataset `/d` both track attribute creation
/// order, each carrying one integer attribute per name in `names`, created in
/// that order and valued by its position.
///
/// The file creation property list carries the same setting so the root group
/// tracks it too, which is what h5py's `File(..., track_order=True)` does.
fn write_tracked(path: &Path, names: &[String], indexed: Indexed) {
    let _c = c_lib_guard();
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()))
        .with_fcpl(|p| p.attr_creation_order(indexed.flags()))
        .create(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let group = file
        .create_group_builder()
        .set_gcpl(&tracking_gcpl(indexed, false))
        .create("g")
        .expect("create group");
    let dataset = file
        .new_dataset::<i32>()
        .with_dcpl(|p| p.attr_creation_order(indexed.flags()))
        .shape([4])
        .create("d")
        .expect("create dataset");
    let owners: [&hdf5::Location; 2] = [&group, &dataset];
    for owner in owners {
        for (i, name) in names.iter().enumerate() {
            owner
                .new_attr::<i32>()
                .shape(())
                .create(name.as_str())
                .unwrap_or_else(|e| panic!("create attribute {name}: {e}"))
                .write_scalar(&(i as i32))
                .unwrap_or_else(|e| panic!("write attribute {name}: {e}"));
        }
    }
    file.close().unwrap();
}

/// A file whose group `/g` tracks *link* creation order, as netCDF-4 writes,
/// holding one dataset per name in `links`, created in that order.
fn write_link_tracked(path: &Path, links: &[&str]) {
    let _c = c_lib_guard();
    let flags = Indexed::Yes.flags();
    let fcpl = FileCreateBuilder::new()
        .attr_creation_order(flags)
        .finish()
        .expect("a file creation property list");
    // TODO: https://github.com/metno/hdf5-rust/issues/230
    // Safety: a live property list id and the flags the C library defines.
    assert_eq!(
        unsafe { H5Pset_link_creation_order(fcpl.id(), flags.bits()) },
        0
    );
    let mut builder = hdf5::File::with_options();
    builder.with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()));
    builder
        .set_fcpl(&fcpl)
        .expect("set the file creation property list");
    let file = builder
        .create(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let group = file
        .create_group_builder()
        .set_gcpl(&tracking_gcpl(Indexed::Yes, true))
        .create("g")
        .expect("create group");
    for link in links {
        group
            .new_dataset::<i32>()
            .shape([4])
            .create(*link)
            .unwrap_or_else(|e| panic!("create dataset {link}: {e}"));
    }
    file.close().unwrap();
}

/// Collect attribute names into the `Vec<String>` `op_data` points at.
unsafe extern "C" fn collect(
    _loc: hid_t,
    name: *const c_char,
    _info: *const H5A_info_t,
    op_data: *mut c_void,
) -> herr_t {
    // Safety: every caller passes a `&mut Vec<String>` as `op_data`, and the C
    // library hands back a NUL-terminated attribute name.
    unsafe {
        let names = &mut *op_data.cast::<Vec<String>>();
        names.push(CStr::from_ptr(name).to_string_lossy().into_owned());
    }
    0
}

/// A link as the C library lists it: its name, and its creation index when the
/// group records one.
struct Link {
    name: String,
    creation_index: Option<i64>,
}

/// Collect links into the `Vec<Link>` `op_data` points at.
unsafe extern "C" fn collect_link(
    _group: hid_t,
    name: *const c_char,
    info: *const H5L_info_t,
    op_data: *mut c_void,
) -> herr_t {
    // Safety: every caller passes a `&mut Vec<Link>` as `op_data`, and the C
    // library hands back a NUL-terminated link name and a filled-in info.
    unsafe {
        let links = &mut *op_data.cast::<Vec<Link>>();
        let info = &*info;
        links.push(Link {
            name: CStr::from_ptr(name).to_string_lossy().into_owned(),
            creation_index: (info.corder_valid != 0).then_some(info.corder),
        });
    }
    0
}

/// The links of `group` in `path`, in the order `idx_type` orders them.
fn links_of(path: &Path, group: &str, idx_type: H5_index_t) -> Vec<Link> {
    let _c = c_lib_guard();
    let file = hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
    let g = file
        .group(group)
        .unwrap_or_else(|e| panic!("the C library opens /{group}: {e}"));
    let mut links: Vec<Link> = Vec::new();
    let mut idx: hsize_t = 0;
    // TODO: https://github.com/metno/hdf5-rust/issues/228
    // Safety: `collect_link` interprets `op_data` as the `Vec<Link>` passed here.
    let rc = unsafe {
        H5Literate(
            g.id(),
            idx_type,
            H5_ITER_INC,
            &raw mut idx,
            Some(collect_link),
            (&raw mut links).cast(),
        )
    };
    assert_eq!(rc, 0, "the C library iterates the links of /{group}");
    links
}

/// An open handle to `/g` or `/d`, whichever `path` names.
struct Owner {
    _file: hdf5::File,
    location: hdf5::Location,
}

impl Owner {
    fn open(path: &Path, object: &str) -> Self {
        let _c = c_lib_guard();
        let file = hdf5::File::open(path)
            .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
        let location = if object == "g" {
            (*file
                .group(object)
                .unwrap_or_else(|e| panic!("the C library opens /{object}: {e}")))
            .clone()
        } else {
            (**file
                .dataset(object)
                .unwrap_or_else(|e| panic!("the C library opens /{object}: {e}")))
            .clone()
        };
        Self {
            _file: file,
            location,
        }
    }

    /// Attribute names in the order `idx_type` orders them.
    fn names(&self, idx_type: H5_index_t) -> Vec<String> {
        let _c = c_lib_guard();
        let mut names: Vec<String> = Vec::new();
        let mut idx: hsize_t = 0;
        // TODO: https://github.com/metno/hdf5-rust/issues/231
        // Safety: `collect` interprets `op_data` as the `Vec<String>` passed here.
        let rc = unsafe {
            H5Aiterate2(
                self.location.id(),
                idx_type,
                H5_ITER_INC,
                &raw mut idx,
                Some(collect),
                (&raw mut names).cast(),
            )
        };
        assert_eq!(rc, 0, "the C library iterates attributes");
        names
    }

    /// The name of the `n`th attribute in creation order, read through
    /// `H5Aopen_by_idx`.
    ///
    /// Distinct from [`names`](Self::names) on purpose: `H5Aiterate2` with an
    /// increasing order builds its table from the *name* index whatever the
    /// requested index type, so it reads the creation index out of the name
    /// index's records and never touches the creation-order B-tree. This walks
    /// `H5A__dense_open_by_idx`, which indexes straight into that B-tree
    /// whenever the Attribute Info message names one — so a dense object whose
    /// message declares the index has to actually carry it.
    fn name_by_creation_index(&self, n: u64) -> String {
        let _c = c_lib_guard();
        let here = cstr(".");
        // TODO: https://github.com/metno/hdf5-rust/issues/231
        // Safety: the attribute id is closed below. The buffer is sized by the
        // length the library reports for the name.
        unsafe {
            let attr = H5Aopen_by_idx(
                self.location.id(),
                here.as_ptr(),
                H5_INDEX_CRT_ORDER,
                H5_ITER_INC,
                n,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            assert!(
                attr > 0,
                "the C library opens attribute {n} by creation order"
            );
            let len = H5Aget_name(attr, 0, std::ptr::null_mut());
            assert!(len > 0, "the C library reports a name length");
            let mut buf = vec![0u8; len as usize + 1];
            let got = H5Aget_name(attr, buf.len(), buf.as_mut_ptr().cast());
            assert_eq!(got, len, "the C library reads the name");
            H5Aclose(attr);
            buf.truncate(len as usize);
            String::from_utf8(buf).expect("a fixture name is UTF-8")
        }
    }

    /// The value of the integer attribute `name`.
    fn value(&self, name: &str) -> i32 {
        let _c = c_lib_guard();
        self.location
            .attr(name)
            .unwrap_or_else(|e| panic!("the C library opens attribute {name}: {e}"))
            .read_scalar::<i32>()
            .unwrap_or_else(|e| panic!("the C library reads attribute {name}: {e}"))
    }
}

/// [`links_in_creation_order_of`] for `/g`, the group every fixture here tracks.
fn links_in_creation_order(path: &Path) -> (Vec<String>, i64) {
    links_in_creation_order_of(path, "g")
}

/// The names of `group`'s links in link creation order, and the highest creation
/// index the group has ever assigned — the counter a deletion must leave alone.
fn links_in_creation_order_of(path: &Path, group: &str) -> (Vec<String>, i64) {
    let names = links_of(path, group, H5_INDEX_CRT_ORDER)
        .into_iter()
        .map(|link| link.name)
        .collect();
    let _c = c_lib_guard();
    let file = hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
    let g = file
        .group(group)
        .unwrap_or_else(|e| panic!("the C library opens /{group}: {e}"));
    let mut info = MaybeUninit::<H5G_info_t>::uninit();
    // TODO: https://github.com/metno/hdf5-rust/issues/233
    // Safety: a live group id, and `info` is written by the call before it is read.
    let info = unsafe {
        assert_eq!(
            H5Gget_info(g.id(), info.as_mut_ptr()),
            0,
            "the C library reads group info"
        );
        info.assume_init()
    };
    (names, info.max_corder)
}

/// [`link_creation_index_of`] for `/g`, the group every fixture here tracks.
fn link_creation_index(path: &Path, link: &str) -> i64 {
    link_creation_index_of(path, "g", link)
}

/// The creation index the C library reports for the link `<group>/<link>`, which
/// must be one the group actually records: a link written without one reads back
/// with `corder_valid` clear, and the assertion below is what catches that.
fn link_creation_index_of(path: &Path, group: &str, link: &str) -> i64 {
    let found = links_of(path, group, H5_INDEX_NAME)
        .into_iter()
        .find(|l| l.name == link)
        .unwrap_or_else(|| panic!("/{group}/{link} is not a link the C library lists"));
    found.creation_index.unwrap_or_else(|| {
        panic!(
            "/{group}/{link} carries no creation index, so it is unnumbered in the group's \
             link order"
        )
    })
}

/// The creation index the C library reports for attribute `attr` of `/object`.
fn creation_index(path: &Path, object: &str, attr: &str) -> u32 {
    let _c = c_lib_guard();
    let file = hdf5::File::open(path)
        .unwrap_or_else(|e| panic!("the C library opens {}: {e}", path.display()));
    let oname = cstr(object);
    let aname = cstr(attr);
    let mut info = MaybeUninit::<H5A_info_t>::uninit();
    // TODO: https://github.com/metno/hdf5-rust/issues/231
    // Safety: a live file id, and `info` is written by the call before it is read.
    let info = unsafe {
        let rc = H5Aget_info_by_name(
            file.id(),
            oname.as_ptr(),
            aname.as_ptr(),
            info.as_mut_ptr(),
            H5P_DEFAULT,
        );
        assert_eq!(rc, 0, "the C library reads info for {object}/{attr}");
        info.assume_init()
    };
    assert!(
        info.corder_valid != 0,
        "{object}/{attr} carries no creation index, so the object stopped tracking the order",
    );
    info.corder
}

fn names(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("a{i:02}")).collect()
}

/// Both objects of a fixture, so every test covers a group and a dataset — the
/// two headers the editor rebuilds by different routes.
const OBJECTS: [&str; 2] = ["g", "d"];

/// Set `name` on both `/g` and `/d` through one edit session.
fn set_on_both(path: &Path, name: &str, value: i64) {
    let s = File::open_rw(path).expect("the editor opens a tracked file");
    s.group("g")
        .expect("the group is reachable")
        .set_attr(name, AttrValue::I64(value))
        .expect("the group takes an attribute");
    s.dataset("d")
        .expect("the dataset is reachable")
        .set_attr(name, AttrValue::I64(value))
        .expect("the dataset takes an attribute");
    s.commit().expect("the commit lands");
}

#[test]
fn an_object_tracking_creation_order_can_be_edited_at_all() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);

    set_on_both(&p, "added", 99);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_NAME),
            ["a00", "a01", "a02", "added"],
            "/{object} lost an attribute by name",
        );
        // The new attribute takes the next creation index, so it iterates last —
        // which is exactly what h5py's `track_order` iteration shows.
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            ["a00", "a01", "a02", "added"],
            "/{object} did not put the new attribute last in creation order",
        );
        assert_eq!(owner.value("added"), 99);
        assert_eq!(owner.value("a01"), 1, "/{object} lost an existing value");
    }
    assert_eq!(creation_index(&p, "g", "added"), 3);
    assert_eq!(creation_index(&p, "d", "added"), 3);
}

#[test]
fn the_editor_reads_a_tracked_object_back_itself() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);
    set_on_both(&p, "added", 7);

    let f = File::open(&p).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 4);
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(7)));
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap().len(), 4);
}

#[test]
fn overwriting_an_attribute_keeps_the_creation_index_it_had() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(3), Indexed::Yes);

    set_on_both(&p, "a00", -1);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            ["a00", "a01", "a02"],
            "/{object} moved an overwritten attribute in the creation order",
        );
        assert_eq!(owner.value("a00"), -1);
    }
    assert_eq!(creation_index(&p, "g", "a00"), 0);
    assert_eq!(creation_index(&p, "d", "a00"), 0);
}

#[test]
fn deleting_an_attribute_leaves_a_gap_rather_than_renumbering() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_tracked(&p, &names(4), Indexed::Yes);

    // One from the middle and one from the end: the middle deletion is what
    // leaves a gap in the surviving indexes, and the end one is what separates
    // the counter the Attribute Info message records from the highest index
    // still in use.
    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        let mut d = s.dataset("d").unwrap();
        for name in ["a01", "a03"] {
            g.remove_attr(name).unwrap();
            d.remove_attr(name).unwrap();
        }
        s.commit().unwrap();
    }
    // A fresh attribute takes index 4, neither a freed index nor one past the
    // highest survivor: the reference C library hands them out from a counter
    // that only rises.
    set_on_both(&p, "added", 5);

    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(owner.names(H5_INDEX_CRT_ORDER), ["a00", "a02", "added"]);
        assert_eq!(creation_index(&p, object, "a00"), 0);
        assert_eq!(
            creation_index(&p, object, "a02"),
            2,
            "/{object} renumbered the attributes that survived",
        );
        assert_eq!(
            creation_index(&p, object, "added"),
            4,
            "/{object} lowered the creation-index counter a deletion must leave alone",
        );
    }
}

#[test]
fn a_compact_set_crossing_the_threshold_carries_its_creation_order_into_the_heap() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Eight compact attributes; one is deleted and six more added in one
    // session. That is past the writer's compact threshold, so the whole set is
    // rebuilt into a fractal heap — and the deletion is what makes each
    // attribute's creation index differ from its position in the rebuilt set,
    // so an index taken from the position would show up here.
    write_tracked(&p, &names(8), Indexed::Yes);

    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        let mut d = s.dataset("d").unwrap();
        g.remove_attr("a01").unwrap();
        d.remove_attr("a01").unwrap();
        for i in 8..14 {
            g.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
            d.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    let expected: Vec<String> = names(14).into_iter().filter(|n| n != "a01").collect();
    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            expected,
            "/{object} lost the creation order on the way into the heap",
        );
        assert_eq!(owner.names(H5_INDEX_NAME), expected);
        // Straight through the creation-order B-tree, which this object's
        // Attribute Info message declares.
        for (n, name) in expected.iter().enumerate() {
            assert_eq!(
                &owner.name_by_creation_index(n as u64),
                name,
                "/{object} does not carry the creation-order index it declares",
            );
        }
        for name in &expected {
            let created: u32 = name[1..].parse().expect("a fixture name is a index");
            assert_eq!(owner.value(name), created as i32, "/{object} {name}");
            assert_eq!(
                creation_index(&p, object, name),
                created,
                "/{object} gave {name} an index that is not the order it was created in",
            );
        }
    }
}

#[test]
fn a_tracked_set_goes_dense_without_a_creation_order_index_when_the_object_has_none() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Tracked but not *indexed*: the heap gets a name index only, and the
    // creation index each attribute carries lives in that index's records.
    write_tracked(&p, &names(8), Indexed::No);

    {
        let s = File::open_rw(&p).unwrap();
        let mut d = s.dataset("d").unwrap();
        for i in 8..13 {
            d.set_attr(&format!("a{i:02}"), AttrValue::I64(i as i64))
                .unwrap();
        }
        s.commit().unwrap();
    }

    let owner = Owner::open(&p, "d");
    assert_eq!(owner.names(H5_INDEX_NAME), names(13));
    for (i, name) in names(13).iter().enumerate() {
        assert_eq!(creation_index(&p, "d", name), i as u32);
    }
}

#[test]
fn an_object_already_dense_keeps_its_creation_order_across_an_edit() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Twelve attributes is past the C library's own compact threshold, so both
    // objects already store their attributes in a fractal heap.
    write_tracked(&p, &names(12), Indexed::Yes);

    set_on_both(&p, "added", 42);

    let mut expected: Vec<String> = names(12);
    expected.push("added".to_string());
    for object in OBJECTS {
        let owner = Owner::open(&p, object);
        assert_eq!(
            owner.names(H5_INDEX_CRT_ORDER),
            expected,
            "/{object} reordered a dense set that was already tracked",
        );
        assert_eq!(owner.value("added"), 42);
        for (n, name) in expected.iter().enumerate() {
            assert_eq!(
                &owner.name_by_creation_index(n as u64),
                name,
                "/{object} does not carry the creation-order index it declares",
            );
        }
        for (i, name) in names(12).iter().enumerate() {
            assert_eq!(creation_index(&p, object, name), i as u32);
        }
        assert_eq!(creation_index(&p, object, "added"), 12);
    }
}

/// A group that tracks link creation order takes additions: each new link is
/// numbered from the running maximum its Link Info message records, and that
/// maximum is bumped by the number added.
///
/// The reference C library is the judge, because this crate's own reader ignores
/// link creation order entirely: `H5Literate2` over `H5_INDEX_CRT_ORDER` walks
/// the per-link creation indexes, `H5Lget_info2`'s `corder` names the index one
/// link carries, and `H5Gget_info`'s `max_corder` is the running counter.
#[test]
fn links_created_in_a_tracked_group_take_the_next_creation_indexes() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_link_tracked(&p, &["first", "second"]);

    // Its attributes are editable, as they were before this change.
    {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .set_attr("note", AttrValue::I64(1))
            .unwrap();
        s.commit().unwrap();
    }
    assert_eq!(Owner::open(&p, "g").value("note"), 1);

    // A dataset and a child group, both added in one commit.
    {
        let s = File::open_rw(&p).unwrap();
        let g = s.group("g").unwrap();
        g.create_dataset("added_d", |d| {
            d.with_i32_data(&[1, 2]);
        })
        .unwrap();
        g.create_group("added_g").unwrap();
        s.commit().expect("a tracked group takes an addition");
    }

    let (names, max_corder) = links_in_creation_order(&p);
    assert_eq!(
        names,
        ["first", "second", "added_d", "added_g"],
        "the additions must follow the originals in link creation order",
    );
    assert_eq!(
        max_corder, 4,
        "the counter must advance by the number of links added",
    );
    // Two links added in one commit are numbered consecutively, in the order
    // they were placed.
    assert_eq!(link_creation_index(&p, "added_d"), 2);
    assert_eq!(link_creation_index(&p, "added_g"), 3);
    // The originals keep the indexes they were created with.
    assert_eq!(link_creation_index(&p, "first"), 0);
    assert_eq!(link_creation_index(&p, "second"), 1);
}

/// Deleting a link from a group that tracks link creation order leaves a gap in
/// the order and touches no counter, the same shape this crate already gives an
/// attribute deletion — and the next addition takes the index past the old
/// maximum rather than reusing the gap.
#[test]
fn a_tracked_group_leaves_a_deletion_gap_and_adds_past_it() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_link_tracked(&p, &["first", "middle", "last"]);
    assert_eq!(
        links_in_creation_order(&p),
        (
            vec![
                "first".to_string(),
                "middle".to_string(),
                "last".to_string()
            ],
            3
        ),
        "the fixture does not track link creation order",
    );

    {
        let s = File::open_rw(&p).unwrap();
        s.group("g").unwrap().delete("middle").unwrap();
        s.commit().expect("a deletion needs no creation index");
    }

    let (names, max_corder) = links_in_creation_order(&p);
    assert_eq!(
        names,
        ["first", "last"],
        "the survivors lost their creation order",
    );
    assert_eq!(
        max_corder, 3,
        "a deletion must not lower the group's link creation-index counter",
    );

    {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .create_dataset("added", |d| {
                d.with_i32_data(&[1, 2]);
            })
            .unwrap();
        s.commit().expect("a tracked group takes an addition");
    }

    let (names, max_corder) = links_in_creation_order(&p);
    assert_eq!(names, ["first", "last", "added"]);
    assert_eq!(max_corder, 4);
    assert_eq!(
        link_creation_index(&p, "added"),
        3,
        "the addition must take the index past the old maximum, not the gap the \
         deletion left at 1",
    );
}

/// A group at the compact-storage threshold its Group Info message declares
/// cannot take another link: past it the reference C library moves the links
/// into a fractal heap indexed by a creation-order B-tree, which this crate does
/// not write (issue #102). The refusal lands before any byte changes.
#[test]
fn a_tracked_group_at_its_compact_threshold_is_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // Eight links is the C library's default maximum compact value, so the
    // fixture is the largest tracked group that still stores its links in the
    // object header.
    let links: Vec<String> = (0..8).map(|i| format!("d{i}")).collect();
    let refs: Vec<&str> = links.iter().map(String::as_str).collect();
    write_link_tracked(&p, &refs);
    assert_eq!(links_in_creation_order(&p).1, 8);
    let before = std::fs::read(&p).unwrap();

    // Scoped so the session — and the file lock it holds, which Windows
    // enforces — is gone before the bytes are read back.
    let err = {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .create_dataset("extra", |d| {
                d.with_i32_data(&[1, 2]);
            })
            .unwrap();
        s.commit().unwrap_err()
    };
    assert!(
        matches!(&err, Error::EditUnsupported(m)
            if m.contains("link creation order") && m.contains("dense")),
        "got: {err}",
    );
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "the refusal wrote bytes"
    );
}

/// A tracked group whose links are *already* dense is refused for the same
/// reason, and refused as it is read rather than as the addition is planned.
#[test]
fn a_dense_tracked_group_is_refused() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    // One past the default maximum compact value, so the C library wrote the
    // links into a fractal heap.
    let links: Vec<String> = (0..9).map(|i| format!("d{i}")).collect();
    let refs: Vec<&str> = links.iter().map(String::as_str).collect();
    write_link_tracked(&p, &refs);
    assert_eq!(links_in_creation_order(&p).1, 9);
    let before = std::fs::read(&p).unwrap();

    // Scoped so the session — and the file lock it holds, which Windows
    // enforces — is gone before the bytes are read back.
    let err = {
        let s = File::open_rw(&p).unwrap();
        s.group("g")
            .unwrap()
            .create_dataset("extra", |d| {
                d.with_i32_data(&[1, 2]);
            })
            .unwrap();
        s.commit().unwrap_err()
    };
    assert!(
        matches!(&err, Error::EditUnsupported(m) if m.contains("dense")),
        "got: {err}",
    );
    assert_eq!(
        std::fs::read(&p).unwrap(),
        before,
        "the refusal wrote bytes"
    );
}

/// Copying a group that tracks link creation order reproduces that order: the
/// copy's Link Info message comes over verbatim, so each copied link has to keep
/// the creation index the source recorded for it or the copy would claim an
/// order none of its links carries.
#[test]
fn a_copied_group_keeps_its_link_creation_order() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("t.h5");
    write_link_tracked(&p, &["first", "second", "third"]);

    {
        let s = File::open_rw(&p).unwrap();
        s.copy("g", "g2").unwrap();
        s.commit().expect("a tracked group is copyable");
    }

    assert_eq!(
        links_in_creation_order_of(&p, "g2"),
        (
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ],
            3
        ),
        "the copy lost the source's link creation order",
    );
    for (i, name) in ["first", "second", "third"].iter().enumerate() {
        assert_eq!(link_creation_index_of(&p, "g2", name), i as i64);
    }
}
