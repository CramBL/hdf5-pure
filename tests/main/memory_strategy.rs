//! `MemoryStrategy`: what a read-write open is allowed to spend on holding the
//! file, who decides, and how a caller finds out what it got (issue #198,
//! steps 3 and 4).
//!
//! `File::open_rw` picks its backing from the file rather than making the caller
//! pick a function. The bounded engine cannot edit every file — a userblock still
//! needs the whole-file mirror — and these tests pin what happens then: `open_rw`
//! falls back, an explicit `Bounded` refuses, and either way the resulting `File`
//! says which one it is, because "bounded" is a memory guarantee and a caller who
//! was quietly given the mirror instead has no other way to notice.
//!
//! The other half is that the fallback set stays *small*, since every file added
//! to it is a caller who silently pays `O(file size)` for a file the bounded
//! engine could have edited (issue #389). The matrix at the end of this file
//! pins it as a biconditional over the creation properties rather than as a list
//! of examples: `Auto` mirrors a file if and only if `Bounded` refuses it.

use hdf5_pure::{
    EditBacking, File, FileAccessProperties, FileBuilder, FileCreateProperties, FileSpaceStrategy,
    LibVer, MaxExtent, MemoryStrategy, SyncPolicy,
};
use test_util_hdf5::dataset::Unlimited;

fn with_strategy(strategy: MemoryStrategy) -> FileAccessProperties {
    FileAccessProperties::new().with_memory_strategy(strategy)
}

/// A latest-format file the bounded engine handles: no userblock, v2+ superblock.
fn plain_file(path: &std::path::Path) {
    Unlimited::new("d", &[1i32, 2, 3], 3).pure_create(path);
}

/// The same file behind a 512-byte userblock, which the bounded engine refuses.
fn userblock_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    Unlimited::new("d", &[1i32, 2, 3], 3).add_to(&mut b);
    b.write(path).unwrap();
}

/// Paged, but *not* persisting its free-space managers: neither engine can edit
/// it, so no strategy makes it editable.
fn paged_nonpersist_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0);
    Unlimited::new("d", &[1i32, 2, 3], 3).add_to(&mut b);
    b.write(path).unwrap();
}

/// The headline of step 4: the plain `open_rw` a caller reaches for first now
/// edits an ordinary file with bounded memory, without being asked and without a
/// second function to know about.
#[test]
fn open_rw_edits_an_ordinary_file_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.h5");
    plain_file(&path);

    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
    file.dataset("d").unwrap().append(&[4i32]).unwrap();
    file.close().unwrap();

    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![1, 2, 3, 4]
    );
}

/// And it keeps working on a file the bounded engine cannot edit, which is what
/// makes the dispatch safe to put behind the unqualified entry point.
#[test]
fn open_rw_falls_back_to_the_mirror_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    userblock_file(&path);

    let file = File::open_rw(&path).unwrap();

    // The whole point of the observable: a caller learns the fallback was taken
    // and memory now scales with the file.
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));

    // And it is a real editing session, not just an open that succeeded.
    //
    // Through the *staged* path, because a userblock defeats the immediate
    // in-place append on either engine — the mirror buys the staged edit surface
    // here, not in-place append. `Dataset::append` still refuses, naming
    // `append_staged`.
    let mut ds = file.dataset("d").unwrap();
    let err = ds
        .append(&[4i32])
        .expect_err("in-place append needs no userblock");
    assert!(
        matches!(&err, hdf5_pure::Error::AppendInPlaceUnsupported(m) if m.contains("append_staged")),
        "the refusal should name append_staged as the fallback, got: {err:?}"
    );
    ds.append_staged(|b| {
        b.append_i32(&[4, 5]);
    })
    .unwrap();
    // Drop the dataset handle before closing: it holds a clone of the same inner
    // file, so the exclusive lock outlives `close` while it is alive, and the
    // read below fails on Windows (os error 33) where OS locks are mandatory.
    drop(ds);
    file.close().unwrap();

    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4, 5]
    );
}

#[test]
fn bounded_refuses_the_file_open_rw_would_mirror() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_strict.h5");
    userblock_file(&path);

    let err = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Bounded))
        .expect_err("Bounded must not silently spend O(file size) memory");
    let msg = err.to_string();
    assert!(
        msg.contains("userblock"),
        "the refusal should name the userblock, got: {msg}"
    );
}

/// `Mirrored` is the escape hatch back to what `open_rw` did before it learned to
/// dispatch: whole-file mirror, whatever the file looks like.
#[test]
fn mirrored_takes_the_mirror_even_on_a_bounded_capable_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain_mirrored.h5");
    plain_file(&path);

    let file = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Mirrored)).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    file.dataset("d").unwrap().append(&[4i32]).unwrap();
    file.close().unwrap();
}

/// An explicit strategy overrides the entry point's own default in *both*
/// directions, which is what makes the deprecated function a pure default and not
/// a separate capability.
#[test]
fn an_explicit_strategy_overrides_either_entry_point() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_override.h5");
    userblock_file(&path);

    // open_rw defaults to falling back; asking for Bounded refuses instead.
    let err = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Bounded))
        .expect_err("an explicit Bounded must override open_rw's fallback");
    assert!(
        err.to_string().contains("userblock"),
        "the refusal should name what the bounded engine cannot edit, got: {err}"
    );

    // ...and an explicit Auto falls back where Bounded refuses.
    let file = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Auto)).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
}

/// A file neither engine can edit is refused whatever is preferred — falling back
/// would only trade a clear error at open for the same error at commit, with work
/// already staged. `Mirrored` still opens it, because that is a request for a
/// backing rather than a preference, and reading such a file is legitimate.
#[test]
fn a_paged_file_without_persisted_free_space_is_refused_unless_the_mirror_is_demanded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    paged_nonpersist_file(&path);

    for properties in [
        FileAccessProperties::new(),
        with_strategy(MemoryStrategy::Bounded),
        with_strategy(MemoryStrategy::Auto),
    ] {
        let err = File::open_rw_with_options(&path, properties).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("persisted free space"),
            "{:?} should refuse naming the persisted free space, got: {msg}",
            properties.memory_strategy()
        );
    }

    let file = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Mirrored))
        .expect("Mirrored is a request for a backing, and the mirror can read this file");
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}

/// The shared refusals must be checked *before* the bounded-only ones, or falling
/// back for a userblock skips them and hands back a session that cannot commit.
///
/// A userblock is the case that makes the ordering observable: persisted free space
/// is declined for a non-zero base address, so this file reaches the engine looking
/// exactly like a paged non-persisting one, while also being bounded-ineligible. If
/// the fallback ran first it would mirror the file, accept staged work, and only
/// refuse at `commit` — the late refusal this dispatch exists to eliminate.
#[test]
fn a_shared_refusal_beats_a_fallback_the_same_file_qualifies_for() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_paged.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(4096);
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 0);
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3]);
    b.write(&path).unwrap();

    let err = File::open_rw(&path).expect_err("neither backing can edit this file");
    let msg = err.to_string();
    assert!(
        msg.contains("persisted free space"),
        "the shared paged refusal must win over the userblock fallback, got: {msg}"
    );
}

#[test]
fn a_file_with_no_editing_session_has_no_strategy_to_report() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro.h5");
    plain_file(&path);

    assert_eq!(File::open(&path).unwrap().edit_backing(), None);
    assert_eq!(
        File::open_streaming(&path).unwrap().edit_backing(),
        None,
        "a streaming open builds no editing session either"
    );
}

#[test]
fn the_requested_strategy_survives_on_the_properties() {
    // The request and the outcome are different questions; the properties keep
    // answering the first one even after an open resolved it to the second.
    let props = with_strategy(MemoryStrategy::Auto);
    assert_eq!(props.memory_strategy(), Some(MemoryStrategy::Auto));
    assert_eq!(
        FileAccessProperties::new().memory_strategy(),
        None,
        "unset must stay unset, so each entry point can supply its own default"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("props.h5");
    userblock_file(&path);
    let file = File::open_rw_with_options(&path, props).unwrap();
    assert_eq!(
        file.access_properties().memory_strategy(),
        Some(MemoryStrategy::Auto)
    );
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
}

/// The request type and the outcome type are deliberately distinct, so that
/// `Auto` — a preference between the two backends, not a third backend — cannot
/// be written into a comparison against what a file actually got. This is what
/// the type buys over a rename: `edit_backing() == Some(Auto)` does not compile.
#[test]
fn an_outcome_converts_back_into_the_request_that_pins_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pin.h5");
    plain_file(&path);

    let got = File::open_rw(&path).unwrap().edit_backing().unwrap();
    assert_eq!(got, EditBacking::Bounded);

    // A caller that wants a later reopen to land on the same backing, whatever
    // dispatch would have chosen, converts the outcome into the request.
    let file = File::open_rw_with_options(&path, with_strategy(got.into())).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
    assert_eq!(
        file.access_properties().memory_strategy(),
        Some(MemoryStrategy::Bounded),
        "the conversion must produce the strict request, not the permissive one"
    );
}

/// The SWMR writer always mirrors. `Auto` and unset are *satisfied* by that, so
/// they must keep opening; only `Bounded` is a guarantee it cannot meet, and
/// meeting a guarantee by ignoring it is how a caller silently spends the memory
/// it asked not to.
#[test]
fn the_swmr_writer_refuses_a_bounded_guarantee_it_cannot_meet() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swmr.h5");
    plain_file(&path);

    let err = File::open_swmr_writer_with_options(&path, with_strategy(MemoryStrategy::Bounded))
        .expect_err("the SWMR writer cannot honor a bounded guarantee");
    let msg = err.to_string();
    assert!(
        msg.contains("whole-file mirror") && msg.contains("MemoryStrategy::Auto"),
        "the refusal must name the backing it uses and a strategy that works, got: {msg}"
    );

    for satisfied in [
        None,
        Some(MemoryStrategy::Auto),
        Some(MemoryStrategy::Mirrored),
    ] {
        let props = satisfied.map_or_else(FileAccessProperties::new, with_strategy);
        let file = File::open_swmr_writer_with_options(&path, props)
            .unwrap_or_else(|e| panic!("{satisfied:?} is satisfied by the mirror, got: {e}"));
        assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
        file.close().unwrap();
    }
}

/// `create_with_options` promises a file *and* an open handle. A pair that the
/// reopen would refuse has to fail before the write, or the call leaves a file on
/// disk and returns `Err` — which reads like a failed create but is not one.
#[test]
fn create_refuses_a_pair_it_could_not_reopen_before_writing_anything() {
    let dir = tempfile::tempdir().unwrap();

    // Refused by every backing: a paged file with no persisted free space cannot
    // be grown in place at all.
    let paged = dir.path().join("paged.h5");
    let err = File::create_with_options(
        &paged,
        FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::Page, false, 1),
        FileAccessProperties::new(),
    )
    .expect_err("a paged non-persisting file cannot be reopened read-write");
    assert!(
        err.to_string().contains("persist = true"),
        "the refusal must name the property to change, not tell the caller to \
         recreate the file it is creating: {err}"
    );
    assert!(
        !paged.exists(),
        "the file must not be written when the reopen is known to fail"
    );

    // Refused by the bounded backing only, which is a property *pair* rather than
    // a property: the same fcpl is fine under any other strategy.
    let ub = dir.path().join("ub.h5");
    let err = File::create_with_options(
        &ub,
        FileCreateProperties::new().with_userblock(512),
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .expect_err("a userblock and a bounded guarantee cannot both hold");
    assert!(
        err.to_string().contains("userblock") && err.to_string().contains("Bounded"),
        "the refusal must name both halves of the contradiction: {err}"
    );
    assert!(!ub.exists(), "nothing may be written for a refused pair");

    // A userblock on a *paged* file reaches the first refusal by a different
    // route, and it was the one pair this function did not screen: free space is
    // never persisted for a non-zero base address, so the creation properties can
    // ask for `persist = true` and still produce a file no read-write open
    // accepts. The pair is the defect, not either half.
    let ub_paged = dir.path().join("ub_paged.h5");
    let err = File::create_with_options(
        &ub_paged,
        FileCreateProperties::new()
            .with_userblock(4096)
            .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
            .with_file_space_page_size(4096),
        FileAccessProperties::new(),
    )
    .expect_err("a userblock and a paged file cannot both hold");
    assert!(
        err.to_string().contains("userblock") && err.to_string().contains("Page"),
        "the refusal must name both halves of the contradiction: {err}"
    );
    assert!(
        !ub_paged.exists(),
        "nothing may be written for a refused pair"
    );

    // And the same userblock creates fine when nothing demands the bounded
    // engine, so the check is on the pair and not on the userblock alone.
    let ok = dir.path().join("ok.h5");
    let file = File::create_with_options(
        &ok,
        FileCreateProperties::new().with_userblock(512),
        FileAccessProperties::new(),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
}

/// A file that persists its free-space managers under the default `FsmAggr`
/// strategy — an ordinary growing file, and the one issue #389 reported as
/// mirrored. Nothing about it is bounded-ineligible: version 3 superblock, no
/// userblock.
fn fsm_persisting_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 0);
    Unlimited::new("samples", &[0u64, 1, 2, 3], 4).add_to(&mut b);
    b.write(path).unwrap();
}

/// Persisting free space is not a bounded-only limitation, so `open_rw` must
/// edit such a file through the bounded engine rather than the whole-file mirror
/// (issue #389). The mirror is for a pre-v2 superblock or a userblock, and this
/// file is neither.
#[test]
fn open_rw_edits_a_persisting_fsm_file_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fsm.h5");
    fsm_persisting_file(&path);

    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.edit_backing(),
        Some(EditBacking::Bounded),
        "a persisting FsmAggr file is bounded-eligible; mirroring it would spend \
         O(file size) memory on a file the bounded engine edits"
    );

    let mut ds = file.dataset("samples").unwrap();
    ds.append(&[4u64, 5, 6, 7]).unwrap();
    drop(ds);
    // A staged edit as well as the immediate append, so the commit this session
    // performs is a real one rather than an empty transaction.
    file.root()
        .create_dataset("second", |b| {
            b.with_u64_data(&[10, 11]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();

    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened.dataset("samples").unwrap().read_u64().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        reopened.dataset("second").unwrap().read_u64().unwrap(),
        vec![10, 11]
    );
    assert_eq!(
        reopened.file_space_strategy(),
        Some(FileSpaceStrategy::FsmAggr),
        "the bounded session must leave the file's recorded strategy alone"
    );
}

/// The reporter's exact call in issue #389: create a persisting `FsmAggr` file
/// with a default access list, then append to an unlimited chunked dataset. The
/// backing is decided once at open and does not drift as the file grows, so the
/// second assertion is the one that pins "RSS tracks file length" as impossible
/// rather than merely unmeasured.
#[test]
fn create_with_options_of_a_persisting_fsm_file_opens_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("created.h5");

    let file = File::create_with_options(
        &path,
        FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 0),
        FileAccessProperties::new(),
    )
    .unwrap();
    assert_eq!(
        file.edit_backing(),
        Some(EditBacking::Bounded),
        "a default access list means MemoryStrategy::Auto, which must prefer the \
         bounded engine on a file it can edit"
    );

    file.root()
        .create_dataset("samples", |b| {
            b.with_u64_data(&[0u64; 256])
                .with_shape(&[256])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_chunks(&[256]);
        })
        .unwrap();
    file.commit().unwrap();

    let batch: Vec<u64> = (0..256).collect();
    let mut ds = file.dataset("samples").unwrap();
    for _ in 0..64 {
        ds.append(&batch).unwrap();
    }
    drop(ds);

    assert_eq!(
        file.edit_backing(),
        Some(EditBacking::Bounded),
        "the backing is fixed at open, so a file grown by appends stays bounded"
    );
    file.close().unwrap();

    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened
            .dataset("samples")
            .unwrap()
            .read_u64()
            .unwrap()
            .len(),
        256 + 64 * 256
    );
}

/// The invariant behind `Auto`, stated over the property space rather than over
/// one file: `Auto` mirrors only where `Bounded` refuses.
///
/// One direction alone would not pin it. "Bounded succeeds implies Auto is
/// bounded" is satisfied by an engine that never mirrors anything, and "Auto
/// mirrored implies Bounded refuses" by one that always mirrors; together they
/// say the fallback set is exactly the bounded-ineligible set, which is what
/// makes the fallback list in `MemoryStrategy::Auto`'s documentation exhaustive
/// rather than illustrative.
#[test]
fn auto_matches_bounded_on_every_pair_bounded_accepts() {
    let fcpls = [
        ("default", FileCreateProperties::new()),
        (
            "fsm_aggr-persisting",
            FileCreateProperties::new().with_file_space_strategy(
                FileSpaceStrategy::FsmAggr,
                true,
                0,
            ),
        ),
        (
            "fsm_aggr-transient",
            FileCreateProperties::new().with_file_space_strategy(
                FileSpaceStrategy::FsmAggr,
                false,
                0,
            ),
        ),
        (
            "aggr-persisting",
            FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::Aggr, true, 0),
        ),
        (
            "aggr-transient",
            FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::Aggr, false, 0),
        ),
        (
            "none-persisting",
            FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::None, true, 0),
        ),
        (
            "none-transient",
            FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::None, false, 0),
        ),
        (
            "page-persisting",
            FileCreateProperties::new()
                .with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
                .with_file_space_page_size(4096),
        ),
        (
            "page-transient",
            FileCreateProperties::new()
                .with_file_space_strategy(FileSpaceStrategy::Page, false, 0)
                .with_file_space_page_size(4096),
        ),
        ("userblock", FileCreateProperties::new().with_userblock(512)),
        (
            "libver-1.8",
            FileCreateProperties::new().with_libver_bounds(LibVer::Earliest, LibVer::V18),
        ),
    ];

    // One barrier per close rather than one per operation: this matrix creates
    // and reopens two files per row, and measures none of it.
    let access = |strategy| {
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .with_memory_strategy(strategy)
    };
    let backing_of = |file: File| -> Option<EditBacking> {
        let backing = file.edit_backing();
        file.close().unwrap();
        backing
    };

    // A biconditional both of whose branches are unreachable holds vacuously, so
    // the rows that took each one are counted and the counts are asserted below.
    let (mut checked, mut mirrored) = (0, 0);

    let dir = tempfile::tempdir().unwrap();
    for (label, create) in fcpls {
        let path = dir.path().join(format!("{label}.h5"));
        let bounded_path = dir.path().join(format!("{label}-bounded.h5"));

        // A pair no strategy can create says nothing about the fallback, and is
        // pinned by `create_refuses_a_pair_it_could_not_reopen_before_writing_anything`.
        let Ok(auto) =
            File::create_with_options(&path, create, access(MemoryStrategy::Auto)).map(backing_of)
        else {
            continue;
        };
        let bounded =
            File::create_with_options(&bounded_path, create, access(MemoryStrategy::Bounded))
                .map(backing_of);
        assert_matches_bounded(label, "create_with_options", auto, &bounded);

        // Create and open share one dispatch, and this is what says so: the same
        // file, reopened, resolves the same way.
        let auto_reopened = File::open_rw_with_options(&path, access(MemoryStrategy::Auto))
            .map(backing_of)
            .unwrap_or_else(|e| panic!("{label}: Auto created this file and must reopen it: {e}"));
        assert_eq!(
            auto_reopened, auto,
            "{label}: reopening must resolve to the backing the create resolved to"
        );
        let bounded_reopened =
            File::open_rw_with_options(&path, access(MemoryStrategy::Bounded)).map(backing_of);
        assert_matches_bounded(
            label,
            "open_rw_with_options",
            auto_reopened,
            &bounded_reopened,
        );

        checked += 1;
        mirrored += usize::from(auto == Some(EditBacking::Mirrored));
    }

    assert_eq!(
        mirrored, 1,
        "the userblock is the only creatable property here the bounded engine \
         refuses; another row reaching the mirror is a fallback set that grew"
    );
    assert!(
        checked >= 9,
        "only {checked} of the property rows were creatable at all, so this \
         matrix is no longer covering the space it names"
    );
}

/// The biconditional shared by both halves of the matrix above: `Auto` is
/// bounded wherever `Bounded` is accepted, and mirrors only where it is refused.
#[track_caller]
fn assert_matches_bounded(
    label: &str,
    entry_point: &str,
    auto: Option<EditBacking>,
    bounded: &Result<Option<EditBacking>, hdf5_pure::Error>,
) {
    match bounded {
        Ok(backing) => {
            assert_eq!(
                *backing,
                Some(EditBacking::Bounded),
                "{label}: an explicit Bounded that opens must report the bounded backing"
            );
            assert_eq!(
                auto,
                Some(EditBacking::Bounded),
                "{label}: {entry_point} under Auto mirrored a file Bounded edits, which \
                 is the O(file size) memory tax of issue #389"
            );
        }
        Err(e) => {
            assert_eq!(
                auto,
                Some(EditBacking::Mirrored),
                "{label}: {entry_point} under Bounded refused ({e}), so Auto's only way \
                 to open this file is the mirror"
            );
            assert!(
                matches!(e, hdf5_pure::Error::EditUnsupported(_)),
                "{label}: a bounded-only limitation must be reported as EditUnsupported, \
                 got: {e:?}"
            );
        }
    }
}
