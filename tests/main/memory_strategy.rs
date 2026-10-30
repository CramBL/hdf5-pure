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

use hdf5_pure::EditBacking;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileCreateProperties;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::LibVer;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::SyncPolicy;
use hdf5_pure::UserblockSize;
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
    b.with_userblock(UserblockSize::MIN);
    Unlimited::new("d", &[1i32, 2, 3], 3).add_to(&mut b);
    b.write(path).unwrap();
}

/// Writes a non-persistent PAGE file that requires mirrored staged editing.
fn paged_nonpersist_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page).with_threshold(0));
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
    test_util_hdf5::assertion::append_in_place_unsupported(
        err,
        "in-place append does not support a file with a userblock (non-zero base address); use Dataset::append_staged",
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
    test_util_hdf5::assertion::edit_unsupported(
        err,
        "bounded read-write access does not support a file with a userblock (non-zero base address); leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to fall back to the whole-file mirror here",
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
    test_util_hdf5::assertion::edit_unsupported(
        err,
        "bounded read-write access does not support a file with a userblock (non-zero base address); leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to fall back to the whole-file mirror here",
    );

    // ...and an explicit Auto falls back where Bounded refuses.
    let file = File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Auto)).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
}

#[test]
fn nonpersistent_page_uses_the_mirror_unless_bounded_is_required() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    paged_nonpersist_file(&path);
    let error =
        File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Bounded)).unwrap_err();
    test_util_hdf5::assertion::edit_unsupported(
        error,
        "bounded editing of PAGE files without persisted free space is unsupported; use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits",
    );
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [1, 2, 3]);
}

#[test]
fn nonpersistent_page_with_userblock_uses_the_mirror() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_paged.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(4096).unwrap());
    b.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page).with_threshold(0));
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3]);
    b.write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    file.root()
        .create_dataset("added", |builder| {
            builder.with_i32_data(&[4]);
        })
        .unwrap();
    file.commit().unwrap();
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
#[rstest::rstest]
#[case::default(None)]
#[case::auto(Some(MemoryStrategy::Auto))]
#[case::mirrored(Some(MemoryStrategy::Mirrored))]
fn the_swmr_writer_refuses_a_bounded_guarantee_it_cannot_meet(
    #[case] satisfied: Option<MemoryStrategy>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swmr.h5");
    plain_file(&path);

    let err = File::open_swmr_writer_with_options(&path, with_strategy(MemoryStrategy::Bounded))
        .expect_err("the SWMR writer cannot honor a bounded guarantee");
    test_util_hdf5::assertion::edit_unsupported(
        err,
        "the SWMR writer always holds the file in a whole-file mirror; leave MemoryStrategy unset, or pass MemoryStrategy::Auto or MemoryStrategy::Mirrored, to open it",
    );

    let props = satisfied.map_or_else(FileAccessProperties::new, with_strategy);
    let file = File::open_swmr_writer_with_options(&path, props)
        .unwrap_or_else(|e| panic!("{satisfied:?} is satisfied by the mirror, got: {e}"));
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    file.close().unwrap();
}

/// `create_with_options` promises a file *and* an open handle. A pair that the
/// reopen would refuse has to fail before the write, or the call leaves a file on
/// disk and returns `Err` — which reads like a failed create but is not one.
#[test]
fn create_refuses_a_pair_it_could_not_reopen_before_writing_anything() {
    let dir = tempfile::tempdir().unwrap();

    // Non-persistent PAGE editing requires the mirror.
    let paged = dir.path().join("paged.h5");
    let err = File::create_with_options(
        &paged,
        FileCreateProperties::new()
            .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page)),
        with_strategy(MemoryStrategy::Bounded),
    )
    .expect_err("bounded editing requires persisted PAGE managers");
    let Error::EditUnsupported(reason) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        reason,
        "bounded editing of PAGE files without persisted free space is unsupported; \
        use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits"
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
        FileCreateProperties::new().with_userblock(UserblockSize::MIN),
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .expect_err("a userblock and a bounded guarantee cannot both hold");
    test_util_hdf5::assertion::edit_unsupported(
        err,
        "a userblock cannot be combined with MemoryStrategy::Bounded: the bounded engine cannot edit a file with a non-zero base address, so creating one this way would write the file and then refuse to open it; drop the userblock, or leave MemoryStrategy unset to mirror this file",
    );
    assert!(!ub.exists(), "nothing may be written for a refused pair");

    // And the same userblock creates fine when nothing demands the bounded
    // engine, so the check is on the pair and not on the userblock alone.
    let ok = dir.path().join("ok.h5");
    let file = File::create_with_options(
        &ok,
        FileCreateProperties::new().with_userblock(UserblockSize::MIN),
        FileAccessProperties::new(),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
}

#[test]
fn create_with_options_writes_a_paged_file_with_a_userblock_that_reopens_read_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged_userblock.h5");
    let file = File::create_with_options(
        &path,
        paged_userblock_properties(4096),
        FileAccessProperties::new(),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    file.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();

    let err =
        File::open_rw_with_options(&path, with_strategy(MemoryStrategy::Bounded)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::EditUnsupported(
                "bounded read-write access does not support a file with a userblock (non-zero \
                 base address); leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to \
                 fall back to the whole-file mirror here"
            )
        ),
        "{err:?}"
    );
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    file.root()
        .create_dataset("e", |b| {
            b.with_i32_data(&[4, 5]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(file.dataset("e").unwrap().read_i32().unwrap(), vec![4, 5]);
}

#[test]
fn create_with_options_rejects_a_paged_userblock_that_is_not_a_whole_number_of_pages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged_partial_userblock.h5");
    let err = File::create_with_options(
        &path,
        paged_userblock_properties(512),
        FileAccessProperties::new(),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Format(FormatError::UserblockNotPageAligned(512, 4096))
        ),
        "{err:?}"
    );
    assert!(!path.exists());
}

fn paged_userblock_properties(userblock: u64) -> FileCreateProperties {
    FileCreateProperties::new()
        .with_userblock(UserblockSize::try_from(userblock).unwrap())
        .with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::DEFAULT),
        )
}

/// A file that persists its free-space managers under the default `FsmAggr`
/// strategy — an ordinary growing file, and the one issue #389 reported as
/// mirrored. Nothing about it is bounded-ineligible: version 3 superblock, no
/// userblock.
fn fsm_persisting_file(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
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
        reopened
            .file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
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
        FileCreateProperties::new().with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                .with_persistence(true)
                .with_threshold(0),
        ),
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
        reopened.dataset("samples").unwrap().read_u64().unwrap(),
        core::iter::repeat_n(0u64, 256)
            .chain(batch.iter().copied().cycle().take(64 * 256))
            .collect::<Vec<_>>()
    );
}

/// Creation and reopening agree on the backing selected by `Auto`.
///
/// Each case specifies whether the bounded engine accepts its creation properties.
#[rstest::rstest]
#[case::default(FileCreateProperties::new(), BoundedOutcome::Eligible)]
#[case::fsm_aggr_persisting(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                    .with_persistence(true)
                    .with_threshold(0),
            ), BoundedOutcome::Eligible)]
#[case::fsm_aggr_transient(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_threshold(0),
            ), BoundedOutcome::Eligible)]
#[case::aggr_persisting(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Aggr)
                    .with_persistence(true)
                    .with_threshold(0),
            ), BoundedOutcome::Eligible)]
#[case::aggr_transient(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Aggr).with_threshold(0),
            ), BoundedOutcome::Eligible)]
#[case::none_persisting(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::None)
                    .with_persistence(true)
                    .with_threshold(0),
            ), BoundedOutcome::Eligible)]
#[case::none_transient(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::None).with_threshold(0),
            ), BoundedOutcome::Eligible)]
#[case::page_persisting(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_persistence(true)
                    .with_threshold(0)
                    .with_page_size(FileSpacePageSize::DEFAULT),
            ), BoundedOutcome::Eligible)]
#[case::page_transient(FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_threshold(0)
                    .with_page_size(FileSpacePageSize::DEFAULT),
            ), BoundedOutcome::NonpersistentPage)]
#[case::userblock(FileCreateProperties::new().with_userblock(UserblockSize::MIN), BoundedOutcome::Userblock)]
#[case::page_persisting_userblock(FileCreateProperties::new()
                .with_userblock(UserblockSize::try_from(4096).unwrap())
                .with_file_space_settings(
                    FileSpaceSettings::new(FileSpaceStrategy::Page)
                        .with_persistence(true)
                        .with_threshold(0)
                        .with_page_size(FileSpacePageSize::DEFAULT),
                ), BoundedOutcome::Userblock)]
#[case::libver_1_8(FileCreateProperties::new().with_libver_bounds(LibVer::Earliest, LibVer::V18), BoundedOutcome::Eligible)]
fn auto_matches_bounded_on_every_pair_bounded_accepts(
    #[case] create: FileCreateProperties,
    #[case] outcome: BoundedOutcome,
) {
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auto.h5");
    let bounded_path = dir.path().join("bounded.h5");
    let expected = match outcome {
        BoundedOutcome::Eligible => EditBacking::Bounded,
        BoundedOutcome::NonpersistentPage | BoundedOutcome::Userblock => EditBacking::Mirrored,
    };
    let auto = File::create_with_options(&path, create, access(MemoryStrategy::Auto))
        .map(backing_of)
        .unwrap();
    assert_eq!(auto, Some(expected));
    let bounded = File::create_with_options(&bounded_path, create, access(MemoryStrategy::Bounded))
        .map(backing_of);
    outcome.assert_created(bounded);
    assert_eq!(
        bounded_path.exists(),
        matches!(outcome, BoundedOutcome::Eligible)
    );
    let reopened = File::open_rw_with_options(&path, access(MemoryStrategy::Auto))
        .map(backing_of)
        .unwrap();
    assert_eq!(reopened, Some(expected));
    let bounded =
        File::open_rw_with_options(&path, access(MemoryStrategy::Bounded)).map(backing_of);
    outcome.assert_opened(bounded);
}

/// The expected bounded-engine result for a creation-property case.
#[derive(Clone, Copy)]
enum BoundedOutcome {
    /// The bounded engine can create and reopen the file.
    Eligible,
    /// PAGE allocation without persisted free-space managers.
    NonpersistentPage,
    /// A nonzero userblock before the HDF5 superblock.
    Userblock,
}

impl BoundedOutcome {
    /// Asserts a bounded backing or the complete creation rejection for this case.
    ///
    /// # Panics
    ///
    /// Panics if an eligible case does not return a bounded backing, or if an ineligible case
    /// does not return [`Error::EditUnsupported`] with the expected reason.
    fn assert_created(self, result: Result<Option<EditBacking>, Error>) {
        let reason = match self {
            Self::Eligible => {
                assert_eq!(result.unwrap(), Some(EditBacking::Bounded));
                return;
            }
            Self::NonpersistentPage => {
                "bounded editing of PAGE files without persisted free space is unsupported; use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits"
            }
            Self::Userblock => {
                "a userblock cannot be combined with MemoryStrategy::Bounded: the bounded engine cannot edit a file with a non-zero base address, so creating one this way would write the file and then refuse to open it; drop the userblock, or leave MemoryStrategy unset to mirror this file"
            }
        };
        test_util_hdf5::assertion::edit_unsupported(result.unwrap_err(), reason);
    }

    /// Asserts a bounded backing or the complete reopen rejection for this case.
    ///
    /// # Panics
    ///
    /// Panics if an eligible case does not return a bounded backing, or if an ineligible case
    /// does not return [`Error::EditUnsupported`] with the expected reason.
    fn assert_opened(self, result: Result<Option<EditBacking>, Error>) {
        let reason = match self {
            Self::Eligible => {
                assert_eq!(result.unwrap(), Some(EditBacking::Bounded));
                return;
            }
            Self::NonpersistentPage => {
                "bounded editing of PAGE files without persisted free space is unsupported; use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits"
            }
            Self::Userblock => {
                "bounded read-write access does not support a file with a userblock (non-zero base address); leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to fall back to the whole-file mirror here"
            }
        };
        test_util_hdf5::assertion::edit_unsupported(result.unwrap_err(), reason);
    }
}
