// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Cross-validation of the file-space strategy (`H5Pset_file_space_strategy`)
//! against the reference HDF5 C library: a strategy hdf5-pure writes is read back
//! by the C library, and a strategy the C library writes is read back by
//! hdf5-pure — including a persisted (persist=true) file whose File Space Info
//! message carries free-space-manager addresses.

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::{
    AttrValue, File, FileAccessProperties, FileBuilder, FileSpaceStrategy, Layout, MemoryStrategy,
};

/// Open with the bounded engine demanded rather than merely preferred: these
/// tests are about that engine, so a file it stops accepting must fail here
/// rather than quietly retarget the whole file at the mirror.
fn open_bounded(path: &std::path::Path) -> Result<File, hdf5_pure::Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
}

use tempfile::tempdir;

mod common;
use common::assert_c_absent;

#[test]
fn c_library_reads_our_strategy() {
    let dir = tempdir().unwrap();
    for (i, (ours, expected)) in [
        (
            FileSpaceStrategy::FsmAggr,
            CStrategy::FreeSpaceManager {
                paged: false,
                persist: false,
                threshold: 1,
            },
        ),
        (
            FileSpaceStrategy::Page,
            CStrategy::FreeSpaceManager {
                paged: true,
                persist: false,
                threshold: 1,
            },
        ),
        // The C library maps the AGGR strategy onto its `PageAggregation` name.
        (FileSpaceStrategy::Aggr, CStrategy::PageAggregation),
        (FileSpaceStrategy::None, CStrategy::None),
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("ours_{i}.h5"));
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[10, 20, 30]);
        b.with_file_space_strategy(ours, false, 1)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();

        // The C library opens our file, reads the data, and recovers the strategy
        // from the superblock-extension File Space Info message.
        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            vec![10, 20, 30]
        );
        let strat = f.create_plist().unwrap().get_file_space_strategy().unwrap();
        assert_eq!(
            strat, expected,
            "strategy {ours:?} read back by the C library"
        );
    }
}

#[test]
fn we_read_c_library_persisted_free_space() {
    // The C library writes a persisted FSM file with real free space (a deleted
    // dataset). hdf5-pure must follow the File Space Info manager addresses to the
    // on-disk FSHD/FSSE blocks and recover the freed sections.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_persisted_fsm.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((100,))
            .create("a")
            .unwrap()
            .write(&vec![1i32; 100])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((400,))
            .create("b")
            .unwrap()
            .write(&vec![2i32; 400])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((100,))
            .create("c")
            .unwrap()
            .write(&vec![3i32; 100])
            .unwrap();
        file.unlink("b").unwrap(); // frees b's storage into the persisted FSM
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::FsmAggr));
    assert!(f.file_space_info().unwrap().persist);

    let free = f.persisted_free_space();
    assert!(
        !free.is_empty(),
        "expected persisted free sections from the deleted dataset"
    );
    // The freed regions are non-overlapping and account for real space (b held
    // 400 * 4 = 1600 bytes of raw data, freed as at least one large section).
    let total: u64 = free.iter().map(|(_, len)| *len).sum();
    assert!(
        total >= 1600,
        "freed space {total} should cover the dataset"
    );
    assert!(free.iter().any(|&(_, len)| len >= 1600));

    // Surviving datasets still read correctly alongside the persisted managers.
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
}

#[test]
fn c_library_reads_our_persisted_free_space() {
    // The mirror of `we_read_c_library_persisted_free_space`: hdf5-pure writes a
    // persisted file with real free space (a deleted dataset), and the reference
    // C library opens it, recovers the strategy, reads the survivors, and loads
    // the on-disk free-space managers we wrote.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_persisted.h5");

    // Create a persisted file, then free a dataset's storage in place.
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("big").with_i32_data(&[7; 400]); // 1600 bytes of raw data
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
    }

    // hdf5-pure's own reader recovers the persisted sections (covering "big").
    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    assert!(
        total_ours >= 1600,
        "we persist the freed storage: {total_ours}"
    );
    drop(ours);

    // The C library opens the same file: strategy and persist flag round-trip,
    // the survivors read byte-exact, and `H5Fget_freespace` parses our managers.
    let f = hdf5::File::open(&path).unwrap();
    let strat = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert_eq!(
        strat,
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        },
        "C library recovers our persisted FSM strategy"
    );
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 100]
    );
    assert_c_absent(&f.dataset("big").unwrap_err(), "big");

    // Loading the managers requires parsing our FSHD/FSSE blocks; the C library
    // reports at least the freed dataset's storage as free space.
    let free_c = f.free_space() as i64;
    assert!(
        free_c >= 1600,
        "C library loads our free-space managers and reports the freed space (got {free_c})"
    );
}

#[test]
fn c_library_reads_managers_we_placed_mid_file() {
    // The test above makes free space and stops, so its managers are written at
    // end-of-file — the only place they could go on a file that had no free space
    // yet. Once a file *has* free space, a commit places them in it (issue #358),
    // and the C library has to find the extension and the managers at addresses
    // below live data, and to accept an `eoa_fsm_fsalloc` that is the whole file
    // rather than the start of the manager blocks. Several commits, so the last
    // set of managers is one that moved into space a previous set vacated.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_persisted_mid_file.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    // Freed below, and large enough that every tail these commits write fits in
    // what it leaves however the free list fragments.
    b.create_dataset("big").with_i32_data(&[7; 4000]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();
    let mut sizes = Vec::new();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        for i in 0..4 {
            s.root().set_attr("n", AttrValue::I64(i)).unwrap();
            s.commit().unwrap();
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
        // A dataset larger than every hole, so it is written past everything and
        // the managers this commit rewrites — drawn from holes below it — end up
        // with live data above them whichever hole they land in.
        s.root()
            .create_dataset("above", |b| {
                b.with_i32_data(&[5; 8000]);
            })
            .unwrap();
        s.commit().unwrap();
    }
    assert_eq!(
        sizes[1],
        *sizes.last().unwrap(),
        "each rewrite of the managers should land in space an earlier one left, \
         leaving the file the size it was (sizes {sizes:?})"
    );

    let ours = File::open(&path).unwrap();
    let managers: Vec<u64> = ours
        .file_space_info()
        .expect("a persisting file records its managers")
        .manager_addrs
        .iter()
        .copied()
        .filter(|&a| a != u64::MAX)
        .collect();
    assert!(!managers.is_empty(), "the file has free space to record");
    let Layout::Contiguous {
        address: Some(above),
        ..
    } = ours.dataset("above").unwrap().layout().unwrap()
    else {
        panic!("`above` is a contiguous dataset with allocated storage");
    };
    for m in &managers {
        assert!(
            *m < above,
            "a manager at {m} should sit in a hole below the data appended after \
             it, at {above}"
        );
    }
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    // The C library reads the same file: survivors byte-exact, and its own
    // free-space total — which it can only get by parsing the managers we placed
    // mid-file — agrees with ours to the byte.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 100]
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "the C library's free-space total must match the managers we wrote"
    );
    drop(f);

    // And *writes* to it. This is the path that consumes `eoa_fsm_fsalloc`: a
    // read-write open loads the free-space managers, frees the blocks holding
    // them, and takes the file's end-of-allocation back to that recorded value —
    // which a reused tail records as the whole file rather than as its own start.
    // Reading proves the C library parses what we wrote; only this proves it acts
    // on it without losing the file.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((30,))
            .create("added")
            .unwrap()
            .write(&(0..30).collect::<Vec<i32>>())
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 100]
    );
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..30).collect::<Vec<i32>>()
    );
    drop(f);
    // Our own reader survives the C library's rewrite of the same file.
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    assert_eq!(
        ours.dataset("above").unwrap().read_i32().unwrap(),
        vec![5; 8000]
    );
}

#[test]
fn we_read_c_library_strategy() {
    let dir = tempdir().unwrap();

    // Each case: a C-written strategy and what hdf5-pure should report. The C
    // library only writes a File Space Info message for non-default settings, so
    // these all differ from the default (FSM_AGGR, non-persistent).
    let write_c = |path: &std::path::Path, strategy: CStrategy| {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(move |fcpl| fcpl.file_space_strategy(strategy))
            .create(path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("d")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.close().unwrap();
    };

    // NONE and AGGR: non-persistent, no manager addresses.
    let p_none = dir.path().join("c_none.h5");
    write_c(&p_none, CStrategy::None);
    let f = File::open(&p_none).unwrap();
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::None));
    assert!(!f.file_space_info().unwrap().persist);
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);

    let p_aggr = dir.path().join("c_aggr.h5");
    write_c(&p_aggr, CStrategy::PageAggregation);
    assert_eq!(
        File::open(&p_aggr).unwrap().file_space_strategy(),
        Some(FileSpaceStrategy::Aggr)
    );

    // FSM_AGGR with persist=true: the message carries free-space-manager
    // addresses, which we parse (without following them to their on-disk blocks).
    let p_persist = dir.path().join("c_persist.h5");
    write_c(
        &p_persist,
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 5,
        },
    );
    let f = File::open(&p_persist).unwrap();
    let info = f
        .file_space_info()
        .expect("persisted file records a strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist, "persist flag is read");
    assert_eq!(info.threshold, 5);
    assert!(
        !info.manager_addrs.is_empty(),
        "a persisted file carries free-space-manager addresses"
    );
    // The data still reads correctly.
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}

#[test]
fn c_library_reads_our_fresh_persisting_file() {
    // Regression for issue #178: a fresh `persist = true` non-paged file must record
    // a defined `eoa_fsm_fsalloc` in its File Space Info message. hdf5-pure once
    // wrote the UNDEF sentinel, which an assertion-enabled libhdf5 rejects on open
    // (H5Fsuper.c: `fs_persist => eoa_fsm_fsalloc != UNDEF`). No dataset is deleted,
    // so the file genuinely tracks no free space — the corner the strategy
    // crosschecks (persist = false, or persist = true only after a delete) missed.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_fresh_persist.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1);
    b.write(&path).unwrap();

    // Nothing is freed, so our own tracked free space is zero.
    let total_ours: u64 = File::open(&path)
        .unwrap()
        .persisted_free_space()
        .iter()
        .map(|(_, l)| l)
        .sum();
    assert_eq!(total_ours, 0);

    // The C library opens the persisting file (which would abort before the fix),
    // recovers the persist flag, reads the data, and reports a matching (zero)
    // free-space total.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        }
    );
    assert_eq!(
        f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    let free_c = f.free_space() as i64;
    assert_eq!(free_c as u64, total_ours);
}

#[test]
fn c_library_reads_our_paged_file() {
    // hdf5-pure writes a genuine paged (H5F_FSPACE_STRATEGY_PAGE) persisting file
    // with small and large datasets. The reference C library must recover the
    // paged strategy, read every dataset, load the per-page-type free-space
    // managers (H5Fget_freespace parses our FSHD/FSSE blocks), and reopen the
    // file read-write to mutate it (proving the paged layout is consistent).
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged.h5");

    let small_a: Vec<i32> = (0..100).collect();
    let small_b: Vec<i32> = (0..400).collect();
    let big: Vec<i32> = (0..5000).collect(); // 20000 bytes >= page: its own run

    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&small_a);
    b.create_dataset("b").with_i32_data(&small_b);
    b.create_dataset("big").with_i32_data(&big);
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(16384);
    b.write(&path).unwrap();

    // hdf5-pure's own view of the tracked free space (SUPER + DRAW + LARGE tails).
    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    assert!(total_ours > 0, "paged persist tracks page-tail free space");
    drop(ours);

    // The C library recovers the paged strategy and reads every dataset.
    let f = hdf5::File::open(&path).unwrap();
    let strat = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert_eq!(
        strat,
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy"
    );
    assert_eq!(f.dataset("a").unwrap().read_raw::<i32>().unwrap(), small_a);
    assert_eq!(f.dataset("b").unwrap().read_raw::<i32>().unwrap(), small_b);
    assert_eq!(f.dataset("big").unwrap().read_raw::<i32>().unwrap(), big);

    // Loading the managers requires parsing our FSHD/FSSE blocks; the C library's
    // free-space total equals the sum of the sections we wrote.
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C library free-space total matches our paged managers"
    );
    drop(f);

    // The C library reopens the paged file read-write and adds a dataset, then
    // reads everything back: the paged layout survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((30,))
            .create("added")
            .unwrap()
            .write(&(0..30).collect::<Vec<i32>>())
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("a").unwrap().read_raw::<i32>().unwrap(), small_a);
    assert_eq!(f.dataset("big").unwrap().read_raw::<i32>().unwrap(), big);
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..30).collect::<Vec<i32>>()
    );
}

#[test]
fn c_library_reads_our_paged_chunked_file() {
    // Paged interop for chunked datasets: hdf5-pure writes a paged persisting
    // file with a small chunked dataset and a large compressed (shuffle+deflate)
    // chunked dataset whose data begins a page-aligned run. The C library must
    // read both, parse the managers, and reopen read-write.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_chunked.h5");

    let small: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let big: Vec<f64> = (0..8000).map(|i| i as f64 * 0.5).collect();

    let mut b = FileBuilder::new();
    b.create_dataset("s")
        .with_f64_data(&small)
        .with_shape(&[64])
        .with_chunks(&[16]);
    {
        let ds = b
            .create_dataset("big")
            .with_f64_data(&big)
            .with_shape(&[8000])
            .with_chunks(&[1000]);
        ds.with_shuffle().with_deflate(6);
    }
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(16384);
    b.write(&path).unwrap();

    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
    );
    assert_eq!(f.dataset("s").unwrap().read_raw::<f64>().unwrap(), small);
    assert_eq!(f.dataset("big").unwrap().read_raw::<f64>().unwrap(), big);
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space matches our managers"
    );
    drop(f);

    // C reopens read-write and appends a dataset without corrupting the file.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<f64>()
            .shape((4,))
            .create("extra")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("big").unwrap().read_raw::<f64>().unwrap(), big);
    assert_eq!(
        f.dataset("extra").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn c_library_reads_our_bounded_mutated_paged_file() {
    // hdf5-pure creates a genuine paged persisting file, then grows it through the
    // bounded backend, appending enough rows to force
    // extensible-array index growth so the append allocates metadata as well as
    // raw chunks (exercising page segregation). The reference C library must then
    // recover the paged strategy, read every row, load the per-page-type managers
    // (H5Fget_freespace parses our FSHD/FSSE blocks), and reopen it read-write.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_mutated.h5");

    // Create with one chunk, then bounded-append to 5000 rows.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();
    }
    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..5000).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    // hdf5-pure's own view of the tracked free space after the mutation.
    let want: Vec<i32> = (0..5000).collect();
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    // The C library recovers the paged strategy and reads every row.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after a bounded mutation"
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    // Loading the managers parses our rewritten FSHD/FSSE blocks; the C library's
    // free-space total equals the sum of the sections we wrote.
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches our rewritten paged managers"
    );
    drop(f);

    // The C library reopens the mutated paged file read-write and appends more
    // rows, then reads everything back: the paged layout survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        let ds = f.dataset("d").unwrap();
        ds.resize((5100,)).unwrap();
        ds.write_slice(&(5000..5100).collect::<Vec<i32>>(), 5000..5100)
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    let after = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(after.len(), 5100);
    assert_eq!(after[..5000], want[..]);
    assert_eq!(after[5099], 5099);
}

#[test]
fn c_library_reads_our_staged_mutated_paged_file() {
    // The whole-file editor's counterpart of the bounded crosscheck above (issue
    // #198): hdf5-pure creates a genuine paged persisting file, then mutates it
    // through `File::open_rw` + staged edits + `commit`, which allocates both raw
    // data (a new dataset's values, appended chunks) and metadata (object headers,
    // the rebuilt chunk index, the rewritten managers). The reference C library
    // must then recover the paged strategy, read every dataset, load the
    // per-page-type managers, and reopen the file read-write.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_staged.h5");

    {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..64).collect::<Vec<i32>>())
            .with_shape(&[64])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[64]);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();
    }
    {
        let file = File::open_rw(&path).unwrap();
        // A staged append (raw chunks + a rebuilt extensible-array index) and a
        // fresh contiguous dataset (raw data + a new object header), so the commit
        // allocates both page types and has to switch between them.
        let mut ds = file.dataset("d").unwrap();
        ds.append_staged(|b| {
            b.append_i32(&(64..2000).collect::<Vec<i32>>());
        })
        .unwrap();
        file.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&vec![2.5f64; 512]);
            })
            .unwrap();
        file.commit().unwrap();
    }

    let want: Vec<i32> = (0..2000).collect();
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    assert_eq!(
        ours.dataset("added").unwrap().read_f64().unwrap(),
        vec![2.5f64; 512]
    );
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    // The C library recovers the paged strategy and reads both datasets.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after a staged commit"
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<f64>().unwrap(),
        vec![2.5f64; 512]
    );
    // Loading the managers parses the per-page-type FSHD/FSSE blocks the commit
    // wrote; the C library's free-space total equals the sum of our sections.
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches our rewritten paged managers"
    );
    drop(f);

    // The C library reopens the file read-write and appends: the paged layout the
    // staged commit produced survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        let ds = f.dataset("d").unwrap();
        ds.resize((2100,)).unwrap();
        ds.write_slice(&(2000..2100).collect::<Vec<i32>>(), 2000..2100)
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    let after = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(after.len(), 2100);
    assert_eq!(after[..2000], want[..]);
    assert_eq!(after[2099], 2099);
}

#[test]
fn c_library_reads_our_paged_file_after_free_space_reuse() {
    // Issue #261: a paged file now draws on its own free space, so a delete
    // followed by a write of the same shape reuses the pages the delete released
    // instead of appending past them. Reuse is confined to the page type being
    // placed, which keeps every page holding one kind of byte — an invariant the
    // C library depends on for page buffering and cannot report on directly. What
    // it *can* do is prove the result is a well-formed paged file: recover the
    // strategy, read every dataset back, and re-parse the rewritten per-page-type
    // managers into the same free-space total we record.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_reused.h5");

    let kept: Vec<f64> = (0..4096).map(|i| (i % 17) as f64).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("victim")
            .with_f64_data(&vec![1.0; 8192])
            .with_chunks(&[1024]);
        b.create_dataset("ceiling").with_f64_data(&kept);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();

    let replacement: Vec<f64> = (0..8192).map(|i| (i % 29) as f64).collect();
    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("victim").unwrap();
        file.commit().unwrap();
        file.root()
            .create_dataset("replacement", |b| {
                b.with_f64_data(&replacement).with_chunks(&[1024]);
            })
            .unwrap();
        file.commit().unwrap();
    }

    let size_after = std::fs::metadata(&path).unwrap().len();
    assert!(
        size_after < size_before + 8192 * 8 / 4,
        "the replacement should reuse the freed raw pages (was {size_before}, \
         now {size_after})"
    );

    let ours = File::open(&path).unwrap();
    assert_eq!(
        ours.dataset("replacement").unwrap().read_f64().unwrap(),
        replacement
    );
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after a commit that reused free space"
    );
    assert_eq!(
        f.dataset("replacement").unwrap().read_raw::<f64>().unwrap(),
        replacement,
        "the C library resolves a chunk index placed in reused pages"
    );
    assert_eq!(
        f.dataset("ceiling").unwrap().read_raw::<f64>().unwrap(),
        kept
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches the managers the reusing commit rewrote"
    );
    drop(f);

    // And the file is still writable by the C library afterwards.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape([4])
            .create("c_added")
            .unwrap()
            .write_raw(&[1i32, 2, 3, 4])
            .unwrap();
        f.close().unwrap();
    }
    let ours = File::open(&path).unwrap();
    assert_eq!(
        ours.dataset("c_added").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        ours.dataset("replacement").unwrap().read_f64().unwrap(),
        replacement
    );
}

#[test]
fn c_library_reads_our_paged_file_after_a_cross_page_type_claim() {
    // Issue #286: a page holding nothing at all belongs to no page type, so either
    // kind may open one — which is what lets metadata reuse the pages a deleted
    // dataset's chunk data released, and how free space survives a close at all
    // (whole free pages are recorded in the manager that names no type, so every
    // one of them comes back as raw). It is also the mechanism most able to break
    // page homogeneity, the invariant the C library depends on for page buffering
    // and cannot report on directly. What it can do is prove the result is a
    // well-formed paged file it reads, re-parses, and writes to.
    //
    // The shape below is what forces the claim: delete the chunked dataset to free
    // whole raw pages, then commit metadata too large for any metadata hole that
    // remains, so the allocator has to open one of those raw pages for it.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_crosstype.h5");

    let kept: Vec<f64> = (0..512).map(|i| (i % 17) as f64).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("victim")
            .with_f64_data(&vec![1.0; 8192])
            .with_chunks(&[1024]);
        b.create_dataset("ceiling").with_f64_data(&kept);
        b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
            .with_file_space_page_size(4096);
        b.write(&path).unwrap();
    }

    let label = "x".repeat(3000);
    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("victim").unwrap();
        file.commit().unwrap();
        for i in 0..8 {
            file.root()
                .create_group_with(&format!("g{i}"), |g| {
                    g.set_attr("label", AttrValue::String(label.clone()));
                })
                .unwrap();
        }
        file.commit().unwrap();
    }

    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("ceiling").unwrap().read_f64().unwrap(), kept);
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after metadata opened a freed raw page"
    );
    for i in 0..8 {
        assert!(
            f.group(&format!("g{i}")).is_ok(),
            "the C library reads a group header placed in a page a delete released"
        );
    }
    assert_eq!(
        f.dataset("ceiling").unwrap().read_raw::<f64>().unwrap(),
        kept
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches the managers the claiming commit rewrote"
    );
    drop(f);

    // And the file is still writable by the C library afterwards.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape([4])
            .create("c_added")
            .unwrap()
            .write_raw(&[5i32, 6, 7, 8])
            .unwrap();
        f.close().unwrap();
    }
    let ours = File::open(&path).unwrap();
    assert_eq!(
        ours.dataset("c_added").unwrap().read_i32().unwrap(),
        vec![5, 6, 7, 8]
    );
    assert_eq!(ours.dataset("ceiling").unwrap().read_f64().unwrap(), kept);
}

#[test]
fn pure_bounded_mutates_c_created_paged_file() {
    // The reverse direction: the reference C library creates a genuine paged
    // persisting file with an unlimited chunked dataset; hdf5-pure grows it through
    // the bounded backend; then both the C library and hdf5-pure read every row
    // back and the C library re-parses the rewritten managers.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_paged_pure_mutated.h5");

    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(4096)
            })
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((64,))
            .shape((hdf5::Extent::resizable(64),))
            .create("d")
            .unwrap();
        ds.write(&(0..64).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    // hdf5-pure grows the C-created paged file.
    {
        let file = open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..4000).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let want: Vec<i32> = (0..4000).collect();
    // hdf5-pure reads its own mutation back.
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    drop(ours);

    // The C library reads it back and re-parses the managers.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 1,
        },
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space matches our managers"
    );
}

#[test]
fn c_library_reads_our_paged_file_after_group_churn() {
    // The reclaim a paged file performs under delete-and-recreate churn is a claim
    // about which bytes are free, and the reference library is the reader that has
    // to agree with it (issue #388). hdf5-pure creates a paged persisting file,
    // then repeatedly creates and deletes a scratch group of empty resizable
    // chunked datasets — the shape whose chunk index the reclaim used to drop,
    // leaving the page it sat in unaccountable. The C library must recover the
    // strategy, read the surviving objects, and report exactly the free space our
    // managers record, which it can only do by parsing the FSHD/FSSE blocks the
    // churn rewrote.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ours_paged_churn.h5");

    let mut b = FileBuilder::new();
    b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0)
        .with_file_space_page_size(16384);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3, 4]);
    b.write(&path).unwrap();

    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        for cycle in 0..20 {
            let name = format!("scratch_{cycle}");
            file.root()
                .create_group_with(&name, |g| {
                    for i in 0..3 {
                        g.create_dataset(&format!("log{i}"), |b| {
                            b.with_i32_data(&[])
                                .with_shape(&[0])
                                .with_chunks(&[64])
                                .with_maxshape(&[u64::MAX]);
                        });
                    }
                })
                .unwrap();
            file.commit().unwrap();
            file.root().delete(&name).unwrap();
            file.commit().unwrap();
        }
        // One scratch group survives, so the file is not merely back to its
        // original shape when the C library reads it.
        file.root()
            .create_group_with("live", |g| {
                g.create_dataset("log", |b| {
                    b.with_i32_data(&[7i32, 8, 9])
                        .with_shape(&[3])
                        .with_chunks(&[2])
                        .with_maxshape(&[u64::MAX]);
                });
            })
            .unwrap();
        file.commit().unwrap();
        file.close().unwrap();
    }

    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
    assert!(total_ours > 0, "the churn left free space to account for");
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after the churn"
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        f.dataset("live/log").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches the managers the churn rewrote"
    );
    drop(f);

    // And the C library reopens the churned file read-write and writes into it.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<f64>()
            .shape((4,))
            .create("extra")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("extra").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        f.dataset("live/log").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
}

/// A file the commit shortened by releasing the free run at its end is read back
/// by the reference library — data, strategy, and free-space total alike
/// (issue #418).
///
/// The release publishes a *smaller* end-of-file and then truncates the file to
/// it, so the two figures the C library cross-checks on open — the superblock's
/// end-of-file and the length of the file it is opening — move together, and the
/// free-space managers it parses must name no section above either. `h5stat`
/// asks the same question through `H5Fget_freespace`, which can only answer it
/// by loading the FSHD/FSSE blocks the release rewrote; a section pointing past
/// the shortened file would be one it hands to the next writer.
///
/// Both strategies, because the run is released in whole pages on a paged file
/// and in bytes on a flat one, and then the C library writes into the shortened
/// file and hdf5-pure reads the result.
#[test]
fn c_library_reads_our_file_after_a_release_shortened_it() {
    let dir = tempdir().unwrap();
    const ROWS: usize = 65536;
    let payload: Vec<i32> = (0..ROWS as i32).collect();

    for (label, strategy, paged) in [
        ("flat", FileSpaceStrategy::FsmAggr, false),
        ("paged", FileSpaceStrategy::Page, true),
    ] {
        let path = dir.path().join(format!("ours_released_{label}.h5"));
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(strategy, true, 0);
        if paged {
            b.with_file_space_page_size(16384);
        }
        b.create_dataset("keep").with_i32_data(&[1, 2, 3, 4]);
        b.write(&path).unwrap();

        let peak = {
            let file = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
            )
            .unwrap();
            file.root()
                .create_group_with("scratch", |g| {
                    for c in 0..2 {
                        g.create_dataset(&format!("c{c}"), |b| {
                            b.with_i32_data(&payload)
                                .with_shape(&[ROWS as u64])
                                .with_chunks(&[512])
                                .with_maxshape(&[u64::MAX]);
                        });
                    }
                })
                .unwrap();
            file.commit().unwrap();
            let peak = file.file_size();
            file.root().delete("scratch").unwrap();
            file.commit().unwrap();
            file.close().unwrap();
            peak
        };

        let len = std::fs::metadata(&path).unwrap().len();
        assert!(
            len < peak / 4,
            "{label}: the delete should have released the end of the file (peak \
             {peak}, now {len})"
        );

        let ours = File::open(&path).unwrap();
        assert_eq!(ours.file_size(), len, "{label}: superblock EOF matches");
        let free_ours: Vec<(u64, u64)> = ours.persisted_free_space();
        for &(addr, size) in &free_ours {
            assert!(
                addr + size <= len,
                "{label}: a persisted section [{addr}, {}) runs past the {len}-byte \
                 file",
                addr + size
            );
        }
        let total_ours: u64 = free_ours.iter().map(|(_, l)| l).sum();
        drop(ours);

        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3, 4],
            "{label}: the C library reads the survivor out of the shortened file"
        );
        let free_c = f.free_space() as i64;
        assert_eq!(
            free_c as u64, total_ours,
            "{label}: the C free-space total matches the managers the release rewrote"
        );
        drop(f);

        // And the C library writes into the shortened file, which hdf5-pure reads.
        {
            let f = hdf5::File::open_rw(&path).unwrap();
            f.new_dataset::<f64>()
                .shape((4,))
                .create("extra")
                .unwrap()
                .write(&[1.0f64, 2.0, 3.0, 4.0])
                .unwrap();
            f.close().unwrap();
        }
        let ours = File::open(&path).unwrap();
        assert_eq!(
            ours.dataset("extra").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0],
            "{label}: hdf5-pure reads what the C library added"
        );
        assert_eq!(
            ours.dataset("keep").unwrap().read_i32().unwrap(),
            vec![1, 2, 3, 4]
        );
    }
}

/// An in-place `Dataset::append` on a persisting file spends holes an earlier
/// commit left, and it does so by taking them out of the on-disk managers first
/// (issue #387). The reference library is the reader that has to agree with the
/// result: it re-parses those rewritten managers, reads every appended value
/// back, and reports exactly the free space we record.
///
/// Both strategies, because the reserve is drawn from one list on a flat file and
/// from the raw-typed one on a paged file, and only the C library's own accounting
/// says the paged managers still segregate what they should.
#[test]
fn c_library_reads_our_file_after_an_append_reused_free_space() {
    let dir = tempdir().unwrap();

    // Comfortably more than one reserve batch, so the append draws from the hole
    // rather than declining it.
    let payload: Vec<i32> = (0..768 * 1024).collect();

    for (label, strategy, paged) in [
        ("flat", FileSpaceStrategy::FsmAggr, false),
        ("paged", FileSpaceStrategy::Page, true),
    ] {
        let path = dir.path().join(format!("ours_append_reuse_{label}.h5"));
        let mut b = FileBuilder::new();
        b.with_file_space_strategy(strategy, true, 1);
        if paged {
            b.with_file_space_page_size(16384);
        }
        b.create_dataset("keep").with_i32_data(&[1, 2, 3, 4]);
        b.write(&path).unwrap();

        let make_log = |file: &File, name: &str| {
            file.root()
                .create_dataset(name, |b| {
                    b.with_i32_data(&[])
                        .with_shape(&[0])
                        .with_chunks(&[16384])
                        .with_maxshape(&[u64::MAX]);
                })
                .unwrap();
            file.commit().unwrap();
        };

        {
            let file = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
            )
            .unwrap();
            make_log(&file, "log");
            file.dataset("log").unwrap().append(&payload).unwrap();
            // A live object above the churn, so the delete leaves an interior hole
            // rather than a run the commit simply truncates away.
            file.root()
                .create_dataset("ceiling", |b| {
                    b.with_i32_data(&[9, 9, 9]);
                })
                .unwrap();
            file.commit().unwrap();
            let after_first = std::fs::metadata(&path).unwrap().len();

            file.root().delete("log").unwrap();
            file.commit().unwrap();

            make_log(&file, "log2");
            file.dataset("log2").unwrap().append(&payload).unwrap();
            file.close().unwrap();

            let after_second = std::fs::metadata(&path).unwrap().len();
            // The delete left a hole one payload wide. An append that reuses it
            // grows the file barely at all; one that goes past the end grows it by
            // the whole payload. Half of that, in bytes, is the line — the same
            // figure the elements-times-two form above it worked out to, said in
            // the unit the file size is in.
            let payload_bytes = std::mem::size_of_val(&payload[..]) as u64;
            assert!(
                after_second < after_first + payload_bytes / 2,
                "{label}: the second append should have reused the hole (was \
                 {after_first}, now {after_second}, payload {payload_bytes} bytes)"
            );
        }

        let ours = File::open(&path).unwrap();
        assert_eq!(ours.dataset("log2").unwrap().read_i32().unwrap(), payload);
        let total_ours: u64 = ours.persisted_free_space().iter().map(|(_, l)| l).sum();
        drop(ours);

        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.create_plist().unwrap().get_file_space_strategy().unwrap(),
            CStrategy::FreeSpaceManager {
                paged,
                persist: true,
                threshold: 1,
            },
            "{label}: C library recovers our strategy after an append reused free space"
        );
        assert_eq!(
            f.dataset("log2").unwrap().read_raw::<i32>().unwrap(),
            payload,
            "{label}: the C library reads every value the reusing append wrote"
        );
        assert_eq!(
            f.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            f.dataset("ceiling").unwrap().read_raw::<i32>().unwrap(),
            vec![9, 9, 9]
        );
        let free_c = f.free_space() as i64;
        assert_eq!(
            free_c as u64, total_ours,
            "{label}: C free-space total matches the managers the reservation rewrote"
        );
        drop(f);

        // And the C library reopens the file read-write and allocates out of those
        // managers itself — the session that would have written over live data had
        // the append spent a hole the managers still advertised.
        {
            let f = hdf5::File::open_rw(&path).unwrap();
            f.new_dataset::<f64>()
                .shape((4096,))
                .create("c_added")
                .unwrap()
                .write(&vec![2.5f64; 4096])
                .unwrap();
            f.close().unwrap();
        }
        let ours = File::open(&path).unwrap();
        assert_eq!(
            ours.dataset("log2").unwrap().read_i32().unwrap(),
            payload,
            "{label}: the C library's own allocation must not have landed on live chunks"
        );
        assert_eq!(
            ours.dataset("c_added").unwrap().read_f64().unwrap(),
            vec![2.5f64; 4096]
        );
    }
}
