use std::path::Path;

use hdf5_pure_format::__private::ExtensibleArrayHeader;
use rstest::rstest;

use crate::{
    ChunkIndex, CompoundTypeBuilder, DeflateLevel, Error, File, FileAccessProperties, FileBuilder,
    FileSpaceSettings, FileSpaceStrategy, MaxExtent, MemoryStrategy, SyncPolicy,
};

use super::{
    Recording, Verdict, persisted_free_space_holds_nothing_live, recorded_eof_covers_the_file,
};

#[rstest]
#[case::single(Index::Single, 2, 4, 2, Payload::Plain, false)]
#[case::fixed(Index::Fixed, 6, 4, 5, Payload::Plain, false)]
#[case::fixed_filtered(Index::Fixed, 6, 4, 5, Payload::Filtered, false)]
#[case::ea_aligned(Index::Extensible, 8, 2, 6, Payload::Plain, false)]
#[case::ea_partial(Index::Extensible, 6, 4, 5, Payload::Plain, false)]
#[case::ea_deflated_partial(Index::Extensible, 6, 4, 5, Payload::Deflate, false)]
#[case::ea_filtered_partial(Index::Extensible, 9, 4, 3, Payload::Filtered, false)]
#[case::ea_filtered_new_tail(Index::Extensible, 8, 4, 9, Payload::Filtered, false)]
#[case::ea_filtered_empty(Index::Extensible, 0, 4, 3, Payload::Filtered, false)]
#[case::ea_indirect_blocks(Index::Extensible, 50, 1, 200, Payload::Plain, false)]
#[case::ea_data_block_pages(Index::Extensible, 131_000, 1, 1_000, Payload::Plain, false)]
#[case::persisted_page(Index::Extensible, 9, 2, 6, Payload::Plain, true)]
#[case::compound(Index::Extensible, 6, 4, 5, Payload::Compound, true)]
fn append_keeps_the_published_extent_at_each_completed_write(
    #[case] index: Index,
    #[case] initial: i32,
    #[case] chunk: u64,
    #[case] extra: i32,
    #[case] payload: Payload,
    #[case] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("append.h5");
    let (index, maximum) = match index {
        Index::Single => (ChunkIndex::SingleChunk, MaxExtent::Fixed(chunk)),
        Index::Fixed => (ChunkIndex::FixedArray, MaxExtent::Fixed(chunk * 8)),
        Index::Extensible => (ChunkIndex::ExtensibleArray, MaxExtent::Unlimited),
    };
    let mut builder = FileBuilder::new();
    if paged {
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
        );
    }
    let dataset = builder
        .create_dataset("d")
        .with_maxshape(&[maximum])
        .with_chunks(&[chunk]);
    if matches!(payload, Payload::Compound) {
        let datatype = CompoundTypeBuilder::new()
            .i32_field("count")
            .i32_field("difference")
            .build()
            .unwrap();
        dataset.with_compound_data(datatype, payload.encode(0..initial), initial as u64);
    } else {
        dataset.with_i32_data(&(0..initial).collect::<Vec<_>>());
    }
    match payload {
        Payload::Plain => {}
        Payload::Deflate => {
            dataset.with_deflate(DeflateLevel::Six);
        }
        Payload::Filtered | Payload::Compound => {
            dataset.with_shuffle().with_deflate(DeflateLevel::Four);
        }
    }
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().chunk_index().unwrap(),
        Some(index)
    );
    if paged {
        assert_eq!(
            file.file_space_settings().unwrap().unwrap().strategy(),
            FileSpaceStrategy::Page
        );
    }
    file.close().unwrap();

    let before = payload.encode(0..initial);
    let end = initial + extra;
    let after = payload.encode(0..end);
    let recording = Recording::of("append", &path, |path| {
        let file = open_for_append(path, backing).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        if matches!(payload, Payload::Compound) {
            dataset.append_raw(&payload.encode(initial..end)).unwrap();
        } else {
            dataset.append(&(initial..end).collect::<Vec<_>>()).unwrap();
        }
        drop(dataset);
        file.close().unwrap();
    });
    let saw_extent_change = std::cell::Cell::new(false);
    recording.replay_every_prefix(
        directory.path(),
        |path| {
            if let Err(reason) = recorded_eof_covers_the_file(path) {
                return Verdict::Silent(reason);
            }
            if paged {
                match persisted_free_space_holds_nothing_live(path) {
                    Verdict::Clean => {}
                    other => return other,
                }
            }
            Verdict::of(read(path), |(extent, values)| {
                if !saw_extent_change.get() && extent == [initial as u64] && values == before {
                    return Ok(());
                }
                if extent == [end as u64] && values == after {
                    saw_extent_change.set(true);
                    return Ok(());
                }
                Err(format!("unexpected extent or values at {extent:?}"))
            })
        },
        |path| {
            let (extent, values) = read(path).map_err(|error| format!("{error:?}"))?;
            if extent == [end as u64] && values == after {
                Ok(())
            } else {
                Err(format!("append did not publish {end} values"))
            }
        },
    );
    assert!(saw_extent_change.get());
}

#[rstest]
fn interrupted_append_can_be_reopened_with_different_values(
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("abandoned.h5");
    super::super::common::build_appendable(&path, 50, 1);
    let recording = Recording::of("abandoned", &path, |path| {
        let file = open_for_append(path, backing).unwrap();
        file.dataset("d")
            .unwrap()
            .append(&(1000..1200).collect::<Vec<i32>>())
            .unwrap();
        file.close().unwrap();
    });
    let initial = Payload::Plain.encode(0..50);
    let abandoned = [&initial, &Payload::Plain.encode(1000..1200)[..]].concat();
    let retry = Payload::Plain.encode(0..150);
    let [header] = test_util::bytes::signature_offsets(&recording.base, b"EAHD")[..] else {
        panic!("expected one Extensible Array header");
    };
    let max_index = |bytes: &[u8]| {
        ExtensibleArrayHeader::parse(bytes, header, 8, 8)
            .unwrap()
            .stored_statistics()
            .max_idx_set
    };
    assert_eq!(max_index(&recording.base), 50);
    assert_eq!(max_index(&std::fs::read(&path).unwrap()), 250);

    let retried_after_index_growth = std::cell::Cell::new(false);
    recording.replay_every_prefix(
        directory.path(),
        |path| {
            let observed = match read(path) {
                Ok(observed) => observed,
                Err(error) => return Verdict::Loud(format!("{error:?}")),
            };
            if observed.0 == [250] && observed.1 == abandoned {
                return Verdict::Clean;
            }
            if observed.0 != [50] || observed.1 != initial {
                return Verdict::Silent(format!("unexpected extent or values at {:?}", observed.0));
            }
            if max_index(&std::fs::read(path).unwrap()) == 250 {
                retried_after_index_growth.set(true);
            }
            let result = (|| {
                let file = open_for_append(path, backing)?;
                file.dataset("d")?
                    .append(&(50..150).collect::<Vec<i32>>())?;
                file.close()?;
                read(path)
            })();
            Verdict::of(result, |(extent, values)| {
                if extent == [150] && values == retry {
                    Ok(())
                } else {
                    Err(format!(
                        "retry published unexpected extent or values at {extent:?}"
                    ))
                }
            })
        },
        |path| {
            let (extent, values) = read(path).map_err(|error| format!("{error:?}"))?;
            if extent == [250] && values == abandoned {
                Ok(())
            } else {
                Err("the uninterrupted append did not finish".into())
            }
        },
    );
    assert!(
        retried_after_index_growth.get(),
        "no prefix with the grown index and original extent admitted a retry"
    );
}

enum Index {
    Single,
    Fixed,
    Extensible,
}

enum Payload {
    Plain,
    Deflate,
    Filtered,
    Compound,
}

impl Payload {
    fn encode(&self, values: std::ops::Range<i32>) -> Vec<u8> {
        if matches!(self, Self::Compound) {
            values
                .flat_map(|value| {
                    value
                        .to_le_bytes()
                        .into_iter()
                        .chain((!value).to_le_bytes())
                })
                .collect()
        } else {
            values.flat_map(i32::to_le_bytes).collect()
        }
    }
}

fn open_for_append(path: &Path, backing: MemoryStrategy) -> Result<File, Error> {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::Always),
    )
}

fn read(path: &Path) -> Result<(Vec<u64>, Vec<u8>), Error> {
    let file = File::open(path)?;
    let dataset = file.dataset("d")?;
    Ok((dataset.shape()?, dataset.read_raw()?))
}
