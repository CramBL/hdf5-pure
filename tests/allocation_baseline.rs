//! Exact allocation figures for chunked datasets and large link groups.
//!
//! This executable measures exact allocation figures under a process-wide allocator.
//! The measured workloads include creating and growing an HDF5 dataset, and parsing
//! and assembling ten thousand hard-link messages. Allocation scaling rules run
//! separately in `tests/allocation_bounds.rs`.
//!
//! Exact figures depend on the target, toolchain, and feature set. CI runs this test on Ubuntu
//! x86_64 with the repository's pinned Rust toolchain. The test requires the exact default feature
//! set. The feature list in its `cfg` attribute is maintained by hand and must include each added
//! feature.
//!
//! Run `just soundness::heap-baseline` to compare recorded figures. Record new figures with:
//!
//! ```sh
//! just soundness::heap-baseline-record
//! ```
//!
//! Record new figures after allocation reductions too, so later increases are compared with the
//! reduced counts.
#![cfg(all(
    // Require the feature set used to record the allocation baseline.
    feature = "std",
    feature = "checksum",
    feature = "deflate",
    not(any(
        feature = "fast-deflate",
        feature = "provenance",
        feature = "zfp",
        feature = "ndarray",
        feature = "serde",
        feature = "num-complex",
    ))
))]

use hdf5_pure::{File, FileBuilder, MaxExtent};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// Writes and reads an 8 MiB dataset, appends sixteen chunks to an unlimited dataset,
/// and measures ten thousand hard-link messages.
#[test]
#[ignore = "Measures exact allocation counts in the pinned default-feature environment"]
fn writing_and_reading_a_chunked_dataset_matches_its_recorded_figures() {
    // The fixture's location is built *before* the profiler starts, so nothing
    // it costs is in the baseline. A temporary directory's path length follows
    // `TMPDIR`, and the figures here are exact: measured inside, a longer
    // `TMPDIR` moved `currBytes` from 177 to 672 and reported it as "some path
    // started allocating more", which is the wrong diagnosis for a machine.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("baseline.h5");

    let _profiler = heapscope::Profiler::builder()
        .no_output()
        .build()
        .unwrap_or_else(|e| {
            panic!(
                "heapscope could not start: {e}\n\
                 On x86_64 this is usually the missing frame pointers that \
                 `.cargo/config.toml` sets for that target; a build that \
                 overrides RUSTFLAGS drops them."
            )
        });

    const N0: usize = 1024 * 1024;
    const CHUNK_ELEMS: u64 = 512;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS]);
    // The appends below need an unlimited dimension, and it must be a *second*
    // dataset: `with_maxshape(&[MaxExtent::Unlimited])` is what selects an Extensible Array
    // chunk index over a Fixed Array one (`chunked_write::use_extensible`), so
    // adding it to "t" would have moved the 2,048-chunk write off the index this
    // file exists to pin and left the Fixed Array builder with no exact figures
    // at all.
    builder
        .create_dataset("growing")
        .with_f64_data(&data[..CHUNK_ELEMS as usize])
        .with_shape(&[CHUNK_ELEMS])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK_ELEMS]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let all = file.dataset("t").unwrap().read_raw().unwrap();
    let read_len = all.len();
    let first_mismatch = all.chunks_exact(8).enumerate().find_map(|(index, bytes)| {
        (bytes != (index as f64).to_le_bytes())
            .then(|| (index, <[u8; 8]>::try_from(bytes).unwrap()))
    });
    drop(all);
    drop(file);

    // Sixteen appends of a chunk each, which is the write path that runs many
    // times over one file. It is here rather than in `allocation_bounds.rs`
    // because what it costs is a *constant* per call: the rules there bound how
    // an append scales with the dataset, and no bound loose enough to hold on
    // every platform would notice a buffer that doubles its way to the batch
    // size instead of reserving it. This would, as a line in a diff.
    {
        let file = hdf5_pure::File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        let mut ds = file.dataset("growing").unwrap();
        let batch: Vec<f64> = (0..CHUNK_ELEMS).map(|i| i as f64).collect();
        for _ in 0..16 {
            ds.append(&batch).unwrap();
        }
    }

    heapscope::assert_baseline!("tests/baselines/chunked_write_read.txt");
    assert_eq!(read_len, N0 * 8);
    assert_eq!(first_mismatch, None);
    {
        let file = File::open(&path).unwrap();
        let dataset = file.dataset("growing").unwrap();
        let batch: Vec<_> = (0..CHUNK_ELEMS).map(|index| index as f64).collect();
        assert_eq!(dataset.shape().unwrap(), &[CHUNK_ELEMS * 17]);
        assert_eq!(dataset.read_f64().unwrap(), batch.repeat(17));
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    hard_link_messages_match_recorded_figures(&_profiler);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn hard_link_messages_match_recorded_figures(profiler: &heapscope::Profiler) {
    use hdf5_pure::{OffsetWidth, StoredAddress};
    use hdf5_pure_core::__private::StoredAddressExt;
    use hdf5_pure_format::__private::{
        GroupLinkStorage, LinkInfoMessage, OwnedLink, OwnedLinkTarget,
    };
    use test_util::allocation;
    use test_util::link_message::HardLink;
    use test_util::widths::Widths;

    const LINK_COUNT: usize = 10_000;
    let (probe, positive) =
        allocation::measure_with_profiler(profiler, "hard_link_messages_control", || {
            std::hint::black_box(vec![0u8; 1_024])
        });
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.bytes >= 1_024, "{positive}");
    assert!(positive.live_bytes >= 1_024, "{positive}");
    assert!(positive.peak_bytes >= 1_024, "{positive}");
    drop(probe);

    let mut encoded_links = Vec::with_capacity(LINK_COUNT);
    let mut expected = Vec::with_capacity(LINK_COUNT);
    for index in 0..LINK_COUNT {
        let name = format!("link_{index:05}");
        let address = (u64::try_from(index).unwrap() + 1) * 64;
        encoded_links.push(HardLink::new(&name, address).build(Widths::EIGHT));
        expected.push((name.into_bytes(), StoredAddress::new(address)));
    }

    let info_bytes = LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    }
    .serialize(OffsetWidth::Eight);

    let (storage, measured) = allocation::measure_with_profiler(
        profiler,
        "hard_link_messages_parse_and_assemble",
        || {
            let mut links = Vec::with_capacity(LINK_COUNT);
            for data in &encoded_links {
                links.push(OwnedLink::parse(data, OffsetWidth::Eight).unwrap());
            }
            GroupLinkStorage::parse(&info_bytes, links, OffsetWidth::Eight).unwrap()
        },
    );

    assert_eq!(storage.links().len(), LINK_COUNT);
    for (link, (name, address)) in storage.links().iter().zip(&expected) {
        assert_eq!(link.name(), name);
        assert_eq!(
            link.target(),
            &OwnedLinkTarget::Hard {
                object_header_address: *address
            }
        );
    }

    let retained_floor = u64::try_from(
        LINK_COUNT * std::mem::size_of::<OwnedLink>()
            + expected.iter().map(|(name, _)| name.len()).sum::<usize>(),
    )
    .unwrap();
    assert!(measured.blocks > LINK_COUNT as u64, "{measured}");
    assert!(measured.live_bytes >= retained_floor, "{measured}");
    assert!(measured.peak_bytes >= measured.live_bytes, "{measured}");
    assert!(measured.bytes >= measured.peak_bytes, "{measured}");

    let report = format!("Hard-link messages: 10,000\n{}", measured.snapshot_report(),);
    insta::with_settings!({snapshot_path => "baselines"}, {
        insta::assert_snapshot!("ten_thousand_hard_link_messages", report);
    });
}
