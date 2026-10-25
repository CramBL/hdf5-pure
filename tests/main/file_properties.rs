//! One `FileAccessProperties` (`fapl`) for every open, and `FileCreateProperties`
//! (`fcpl`) as a reusable creation-property value (issues #204, #205).

use hdf5_pure::FileSpacePageSize;
use hdf5_pure::UserblockSize;
use hdf5_pure::{
    ChunkCacheConfig, Error, File, FileAccessProperties, FileBuilder, FileCreateProperties,
    FileLocking, FileSpaceStrategy, FormatError, LibVer, MemoryStrategy,
};
use test_util::superblock;

fn write_chunked(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&(0..4096).map(|i| i as f64).collect::<Vec<_>>())
        .with_shape(&[4096])
        .with_chunks(&[256]);
    b.write(path).unwrap();
}

/// The whole point of #204: one properties value configured once, then handed to
/// both a read path and a read-write path.
#[test]
fn one_properties_value_serves_read_and_read_write_opens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.h5");
    write_chunked(&path);

    let properties = FileAccessProperties::new()
        .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(1 << 20))
        .with_locking(FileLocking::Disabled);

    // Read path.
    {
        let file = File::open_with_options(&path, properties).unwrap();
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap().len(),
            4096
        );
    }
    // Read-write path — the same value, which had no spelling before #204.
    {
        let file = File::open_rw_with_options(&path, properties).unwrap();
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap().len(),
            4096
        );
    }
    // Read-write path again, this time with the bounded engine demanded: the
    // same value carries the memory strategy alongside everything else.
    {
        let bounded = properties.with_memory_strategy(MemoryStrategy::Bounded);
        let file = File::open_rw_with_options(&path, bounded).unwrap();
        assert_eq!(
            file.dataset("data").unwrap().read_f64().unwrap().len(),
            4096
        );
    }
}

/// The mirror backend previously discarded access properties entirely
/// (`from_rw_session` hardcoded the defaults), so a chunk cache configured for
/// `open_rw` did nothing. It now applies.
#[test]
fn open_rw_honors_the_configured_chunk_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.h5");
    write_chunked(&path);

    // Each handle holds an exclusive lock for its life, so scope them.
    {
        let enabled = FileAccessProperties::new()
            .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(1 << 20));
        let file = File::open_rw_with_options(&path, enabled).unwrap();
        let ds = file.dataset("data").unwrap();
        ds.read_f64().unwrap();
        let stats = ds.chunk_cache_stats();
        assert!(
            stats.cached_chunks() > 0,
            "an enabled chunk cache should retain chunks on the mirror backend, got {stats:?}"
        );
    }
    {
        let disabled = FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled());
        let file = File::open_rw_with_options(&path, disabled).unwrap();
        let ds = file.dataset("data").unwrap();
        ds.read_f64().unwrap();
        assert_eq!(
            ds.chunk_cache_stats().cached_chunks(),
            0,
            "a disabled chunk cache must retain nothing"
        );
    }
}

/// The bounded backend hardcoded `FileLocking::Enabled`, so it could not be used
/// where OS locking is unavailable. The policy now comes from the properties, and
/// applies whichever engine the memory strategy selects.
#[test]
fn the_bounded_engine_honors_the_locking_policy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bounded_lock.h5");
    write_chunked(&path);

    let unlocked = FileAccessProperties::new()
        .with_locking(FileLocking::Disabled)
        .with_memory_strategy(MemoryStrategy::Bounded);
    let _first = File::open_rw_with_options(&path, unlocked).unwrap();
    // Nothing is locked, so a second bounded open and a plain read both succeed.
    File::open_rw_with_options(&path, unlocked)
        .expect("a second Disabled bounded open should succeed: neither took a lock");
    File::open(&path).expect("a reader should succeed against a Disabled bounded writer");
}

#[test]
fn the_bounded_engine_still_locks_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bounded_default.h5");
    write_chunked(&path);

    let bounded = FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded);
    let _held = File::open_rw_with_options(&path, bounded).unwrap();
    assert!(
        File::open_rw_with_options(&path, bounded).is_err(),
        "the default policy must still take an exclusive lock"
    );
}

/// #205: the same creation properties, applied as a value or one at a time, must
/// produce the identical file.
#[test]
fn create_properties_match_the_individual_setters() {
    let properties = FileCreateProperties::new()
        .with_userblock(UserblockSize::MIN)
        .with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 8)
        .with_file_space_page_size(FileSpacePageSize::DEFAULT);

    let from_value = {
        let mut b = FileBuilder::new();
        b.with_create_properties(properties);
        b.create_dataset("d").with_f64_data(&[1.0, 2.0]);
        b.finish().unwrap()
    };
    let from_setters = {
        let mut b = FileBuilder::new();
        b.with_userblock(UserblockSize::MIN);
        b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 8);
        b.with_file_space_page_size(FileSpacePageSize::DEFAULT);
        b.create_dataset("d").with_f64_data(&[1.0, 2.0]);
        b.finish().unwrap()
    };
    assert_eq!(
        from_value, from_setters,
        "applying an fcpl value must equal setting each property individually"
    );
}

/// A layout defined once in a helper and reused, which is the composition #205
/// exists to enable.
#[test]
fn create_properties_are_reusable_across_files() {
    fn shared_layout() -> FileCreateProperties {
        FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 4)
    }

    let dir = tempfile::tempdir().unwrap();
    for name in ["a.h5", "b.h5"] {
        let path = dir.path().join(name);
        let mut b = FileBuilder::new();
        b.with_create_properties(shared_layout());
        b.create_dataset("d").with_f64_data(&[1.0]);
        b.write(&path).unwrap();

        let file = File::open(&path).unwrap();
        let info = file
            .file_space_info()
            .unwrap()
            .expect("the strategy should be recorded in the superblock extension");
        assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
        assert!(info.persist);
        assert_eq!(info.threshold, 4);
    }
}

/// `File::create` gained creation properties: they were previously unreachable
/// through the owned-handle path, which always got the defaults.
#[test]
fn create_with_options_records_creation_properties() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("created.h5");

    let file = File::create_with_options(
        &path,
        FileCreateProperties::new().with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 16),
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    file.root().create_group("g").unwrap();
    file.commit().unwrap();
    drop(file);

    let file = File::open(&path).unwrap();
    assert!(file.group("g").is_ok());
    let info = file
        .file_space_info()
        .unwrap()
        .expect("strategy should be recorded");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert_eq!(info.threshold, 16);
}

#[test]
fn create_with_default_options_matches_create() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.h5");
    let b = dir.path().join("b.h5");

    File::create(&a).unwrap().close().unwrap();
    File::create_with_options(&b, FileCreateProperties::new(), FileAccessProperties::new())
        .unwrap()
        .close()
        .unwrap();

    assert_eq!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "default properties must reproduce the plain `create` file"
    );
}

#[test]
fn const_creation_properties_distinguish_unset_and_explicit_default_page_sizes() {
    const DEFAULT: FileCreateProperties = FileCreateProperties::new();
    const WITH_PAGE: FileCreateProperties =
        DEFAULT.with_file_space_page_size(FileSpacePageSize::DEFAULT);
    assert_eq!(DEFAULT.file_space_page_size(), None);
    assert_eq!(
        WITH_PAGE.file_space_page_size(),
        Some(FileSpacePageSize::DEFAULT)
    );
}

#[test]
fn creation_properties_preserve_typed_userblock_size() {
    const DEFAULT: FileCreateProperties = FileCreateProperties::new();
    const WITH_USERBLOCK: FileCreateProperties = DEFAULT.with_userblock(UserblockSize::MIN);
    assert_eq!(DEFAULT.userblock(), UserblockSize::NONE);
    assert_eq!(WITH_USERBLOCK.userblock(), UserblockSize::MIN);
    let copied = WITH_USERBLOCK;
    let mut builder = FileBuilder::new();
    builder.with_create_properties(copied);
    builder.create_dataset("d").with_f64_data(&[1.0]);
    let bytes = builder.finish().unwrap();
    assert_eq!(
        &bytes[512..512 + superblock::SIGNATURE.len()],
        superblock::SIGNATURE
    );
}

/// `with_create_properties` documents that it overwrites any value set
/// individually before the call. That has to include the properties the list
/// leaves *unset*, or handing over a property list defines the file only
/// partly — and for the library-version bounds, the leftover decides the on-disk
/// format of a file whose property list names no version at all.
#[test]
fn create_properties_reset_the_properties_they_do_not_carry() {
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 8);
    b.with_file_space_page_size(FileSpacePageSize::DEFAULT);
    b.with_userblock(UserblockSize::try_from(1024).unwrap());
    // Names none of the four above.
    b.with_create_properties(FileCreateProperties::new());
    b.create_dataset("d").with_f64_data(&[1.0]);
    let overwritten = b.finish().unwrap();

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_f64_data(&[1.0]);
    let pristine = b.finish().unwrap();

    assert_eq!(
        overwritten, pristine,
        "an empty property list must leave the builder at its defaults"
    );
    assert_eq!(
        overwritten[8], 3,
        "the stale 1.8 bound decided the format of a list that named no version"
    );
}

/// A refusal must not be destructive. Every library-version and userblock check
/// runs before the writer emits a byte, so `FileBuilder::write` has nothing to
/// write and must leave the destination path exactly as it found it — the
/// `File::create` it used to open up front truncates, so the caller lost the
/// file they were overwriting *and* got an error.
#[test]
fn a_refused_build_does_not_touch_the_destination_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("existing.h5");
    let previous = b"the caller's existing file";
    std::fs::write(&path, previous).unwrap();

    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("d")
        .with_i32_data(&(0..100).collect::<Vec<_>>())
        .with_chunks(&[10]);
    assert!(matches!(
        b.write(&path),
        Err(Error::Format(FormatError::LibverTooOldForContent { .. }))
    ));
    assert_eq!(
        std::fs::read(&path).unwrap(),
        previous,
        "the refused build overwrote the destination"
    );

    // And a build refused at a path that did not exist creates nothing, rather
    // than leaving an empty file where a reader would then find a truncated one.
    let fresh = dir.path().join("fresh.h5");
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("d")
        .with_i32_data(&(0..100).collect::<Vec<_>>())
        .with_chunks(&[10]);
    assert!(b.write(&fresh).is_err());
    assert!(!fresh.exists(), "the refused build created an empty file");

    // The same path still writes normally when the build is accepted.
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("d").with_f64_data(&[1.0, 2.0]);
    b.write(&path).unwrap();
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_f64()
            .unwrap(),
        vec![1.0, 2.0]
    );
}

/// An editing session adds whatever the content needs by default, which is what
/// lets a file the C library wrote under its own bounds be edited at all — but
/// adding a chunked dataset to an HDF5 1.8 file needs the version 4 layout and a
/// 1.10 chunk index, so it makes that file need 1.10 while its superblock still
/// says 1.8. Setting the fapl's library-version bound is how a caller says the
/// file must stay loadable by the old reader it was written for.
#[test]
fn an_editing_session_honors_the_library_version_bound() {
    fn stage_chunked(session: &File) {
        session
            .root()
            .create_dataset("added", |b| {
                b.with_i32_data(&(0..100).collect::<Vec<_>>())
                    .with_chunks(&[10]);
            })
            .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v18.h5");
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
    b.write(&path).unwrap();

    let bounded = FileAccessProperties::new().with_libver_bounds(LibVer::Earliest, LibVer::V18);
    {
        let session = File::open_rw_with_options(&path, bounded).unwrap();
        stage_chunked(&session);
        let err = session.commit().unwrap_err();
        assert!(
            matches!(
                err,
                Error::Format(FormatError::LibverTooOldForContent { .. })
            ),
            "a chunked addition under a 1.8 bound gave {err:?}"
        );
    }

    // The commit is all-or-nothing: the refusal left the file as it was.
    {
        let file = File::open(&path).unwrap();
        assert!(file.dataset("added").is_err(), "the refusal wrote anyway");
        assert_eq!(
            file.dataset("values").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }

    // A contiguous addition is expressible in the 1.8 format and still lands, so
    // the bound refuses the content rather than the session.
    {
        let session = File::open_rw_with_options(&path, bounded).unwrap();
        session
            .root()
            .create_dataset("plain", |b| {
                b.with_f64_data(&[4.0, 5.0]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("plain")
            .unwrap()
            .read_f64()
            .unwrap(),
        vec![4.0, 5.0]
    );

    // Unset, the same chunked edit is accepted — the refusal is the caller's
    // choice, not a new blanket rule, so issue #101's C-library-file edits keep
    // working.
    {
        let session = File::open_rw_with_options(&path, FileAccessProperties::new()).unwrap();
        stage_chunked(&session);
        session.commit().unwrap();
    }
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("added")
            .unwrap()
            .read_i32()
            .unwrap()
            .len(),
        100
    );
}

#[rstest::rstest]
#[case(0)]
#[case(512)]
#[case(1024)]
#[case(2048)]
fn userblock_size_sets_signature_offset_in_buffered_streamed_and_owned_output(
    #[case] raw_size: u64,
) {
    let size = UserblockSize::try_from(raw_size).unwrap();
    let properties = FileCreateProperties::new().with_userblock(size);
    let build = || {
        let mut builder = FileBuilder::new();
        builder.with_create_properties(properties);
        if size != UserblockSize::NONE {
            builder.with_userblock_content(b"wrapper");
        }
        builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
        builder
    };
    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();
    assert_eq!(buffered, streamed);
    let offset = usize::try_from(raw_size).unwrap();
    assert_eq!(
        &buffered[offset..offset + superblock::SIGNATURE.len()],
        superblock::SIGNATURE
    );
    if size != UserblockSize::NONE {
        assert_eq!(&buffered[..7], b"wrapper");
        assert!(buffered[7..offset].iter().all(|&byte| byte == 0));
    }
    let file = File::from_bytes(buffered).unwrap();
    assert_eq!(file.superblock().base_address.get(), size.get());
    assert_eq!(
        file.dataset("data").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("owned.h5");
    let file = File::create_with_options(&path, properties, FileAccessProperties::new()).unwrap();
    file.root()
        .create_dataset("data", |builder| {
            builder.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let file = File::open_streaming(&path).unwrap();
    assert_eq!(file.superblock().base_address.get(), size.get());
    assert_eq!(
        file.dataset("data").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        &bytes[offset..offset + superblock::SIGNATURE.len()],
        superblock::SIGNATURE
    );
    assert!(bytes[..offset].iter().all(|&byte| byte == 0));
}

#[rstest::rstest]
#[case::unset(None, None)]
#[case::explicit_default(None, Some(FileSpacePageSize::DEFAULT))]
#[case::strategy_only(Some(FileSpaceStrategy::FsmAggr), None)]
#[case::minimum(None, Some(FileSpacePageSize::MIN))]
#[case::irregular(None, Some(FileSpacePageSize::try_from(3000).unwrap()))]
#[case::maximum(None, Some(FileSpacePageSize::MAX))]
fn creation_paths_preserve_page_size_configuration(
    #[case] strategy: Option<FileSpaceStrategy>,
    #[case] page_size: Option<FileSpacePageSize>,
) {
    let mut properties = FileCreateProperties::new();
    if let Some(strategy) = strategy {
        properties = properties.with_file_space_strategy(strategy, false, 1);
    }
    if let Some(page_size) = page_size {
        properties = properties.with_file_space_page_size(page_size);
    }
    assert_eq!(
        properties
            .file_space_settings()
            .map(|settings| settings.page_size()),
        page_size.or(strategy.map(|_| FileSpacePageSize::DEFAULT))
    );
    let build = || {
        let mut builder = FileBuilder::new();
        builder.with_file_space_page_size(FileSpacePageSize::MAX);
        builder.with_create_properties(properties);
        builder
    };
    let expected = (strategy.is_some() || page_size.is_some()).then(|| {
        (
            strategy.unwrap_or(FileSpaceStrategy::FsmAggr),
            page_size.unwrap_or(FileSpacePageSize::DEFAULT),
        )
    });
    let bytes = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();
    assert_eq!(streamed, bytes);
    let file = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(
        file.file_space_info()
            .unwrap()
            .map(|info| (info.strategy, info.page_size)),
        expected
    );

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("written.h5");
    build().write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let file = File::open_streaming(&path).unwrap();
    assert_eq!(
        file.file_space_info()
            .unwrap()
            .map(|info| (info.strategy, info.page_size)),
        expected
    );

    let owned_path = directory.path().join("owned.h5");
    let file =
        File::create_with_options(&owned_path, properties, FileAccessProperties::new()).unwrap();
    assert_eq!(
        file.file_space_info()
            .unwrap()
            .map(|info| (info.strategy, info.page_size)),
        expected
    );
    file.close().unwrap();
    assert_eq!(std::fs::read(&owned_path).unwrap(), bytes);
}
