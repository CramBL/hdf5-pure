use ::test_util::range;
use rstest::rstest;
use test_util_hdf5::__private::array_index;
use test_util_hdf5::__private::creation_holes;
use test_util_hdf5::__private::creation_holes::HoleLayout;
use test_util_hdf5::file;

use crate::DeflateLevel;
use crate::File;
use crate::FileAccessProperties;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::dataspace::MaxExtent;
use crate::edit::test_util;

#[rstest]
#[case::flat_fixed(false, false)]
#[case::flat_extensible(false, true)]
#[case::page_fixed(true, false)]
#[case::page_extensible(true, true)]
fn creation_reuses_native_holes_and_preserves_survivors_and_chunk_records(
    #[case] paged: bool,
    #[case] extensible: bool,
    #[values(false, true)] filtered: bool,
    #[values(HoleLayout::Contiguous, HoleLayout::Fragmented, HoleLayout::WholePages)]
    layout: HoleLayout,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-holes.h5");
    let holes = creation_holes::create(&path, paged, layout);
    let contiguous = matches!(layout, HoleLayout::Contiguous);
    let count = if matches!(layout, HoleLayout::WholePages) {
        4_096
    } else {
        2_048
    };
    let expected: Vec<i32> = (0..count)
        .scan(0x34a9_12b7_u32, |state, _| {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            Some(i32::from_le_bytes(state.to_le_bytes()))
        })
        .collect();
    let pure = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let available = pure
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| section.address().get()..section.end_address().get())
        .collect::<Vec<_>>();
    if !contiguous {
        assert!(
            pure.reusable_free_sections()
                .unwrap()
                .sections()
                .iter()
                .all(|section| section.length().get() < u64::try_from(expected.len() * 4).unwrap())
        );
    }
    let dataset = pure
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&expected).with_chunks(&[128]);
            if extensible {
                builder.with_maxshape(&[MaxExtent::Unlimited]);
            }
            if filtered {
                builder.with_deflate(DeflateLevel::Six);
            }
        })
        .unwrap();
    pure.commit().unwrap();
    let header = dataset.header_address().unwrap();
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), expected.len() / 128);
    let published = holes
        .iter()
        .map(|section| (section.start, section.end - section.start))
        .collect::<Vec<_>>();
    for chunk in &chunks {
        let extent = chunk.address..chunk.address + chunk.storage_size;
        assert!(
            available
                .iter()
                .any(|section| section.start <= extent.start && extent.end <= section.end),
            "{chunk:?}, {available:?}"
        );
        assert_eq!(
            range::covered_len(&published, &extent),
            chunk.storage_size,
            "{chunk:?}, {holes:?}"
        );
    }
    if contiguous {
        assert!(
            chunks
                .windows(2)
                .all(|pair| pair[0].address + pair[0].storage_size == pair[1].address)
        );
    }
    assert_eq!(dataset.read_i32().unwrap(), expected);
    drop(dataset);
    pure.close().unwrap();
    let engine = test_util::open_engine(&path);
    if paged {
        engine.assert_array_metadata_pages(header);
    }
    assert!(engine.space_ownership().unwrap().analysis_complete);
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    for index in 0..if contiguous { 1 } else { 16 } {
        assert_eq!(
            native
                .dataset(&format!("spacer{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![31; layout.spacer_size()]
        );
    }
    native.close().unwrap();
    let native_records = array_index::libhdf5_chunks(&path, 0);
    let pure = File::open(&path).unwrap();
    assert_eq!(pure.dataset("d").unwrap().read_i32().unwrap(), expected);
    let pure_records = pure
        .dataset("d")
        .unwrap()
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| {
            let mask = chunk.filter_mask().raw_bits();
            (chunk.offset, chunk.address, chunk.storage_size, mask)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        pure_records,
        native_records
            .iter()
            .map(|chunk| (
                chunk.offset().to_vec(),
                chunk.address(),
                chunk.size(),
                chunk.filter_mask()
            ))
            .collect::<Vec<_>>()
    );
    pure.close().unwrap();
}
