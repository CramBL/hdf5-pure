use std::fs;
use std::path::Path;

use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

use super::*;

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn ea_publication_preserves_native_values_and_recorded_widths(
    #[case] offsets: OffsetWidth,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
    #[values(0, 3)] initial: i32,
    #[values(1, 4)] chunk: u64,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.h5");
    let widths = FormatWidths {
        offsets,
        lengths: LengthWidth::Eight,
    };
    producer.write(&input, widths, initial, chunk, filtered);
    let mut store = WindowProbeStore::open(fs::read(&input).unwrap());
    let mut expected = initial;
    for end in [5, 21, 331] {
        let address = dataset_address(&store);
        let mut state = edit::locate_dataset_state(&store, address).unwrap();
        assert_eq!(state.index.current_dim(), u64::try_from(expected).unwrap());
        assert_eq!(state.index.chunk_elems(), chunk);
        assert_eq!(state.pipeline.is_some(), filtered);
        if expected == 0 && matches!(producer, Producer::Native) {
            assert!(matches!(state.index, AppendIndex::Unallocated(_)));
        }
        let values: Vec<i32> = (expected..end).collect();
        let raw = i32_bytes(&values);
        let plan = match &state.index {
            AppendIndex::Allocated(index) => plan_ea_append(
                &store,
                index,
                &state.datatype,
                &state.spatial,
                state.element_size,
                state.pipeline.as_ref(),
                true,
                LossyTailReencoding::Reject,
                &raw,
                values.len().to_u64(),
                state.fill.pattern(state.element_size),
            )
            .unwrap(),
            AppendIndex::Unallocated(index) => {
                let (plan, initialization) = plan_initial_ea_append(
                    index,
                    widths,
                    &state.datatype,
                    &state.spatial,
                    state.element_size,
                    state.pipeline.as_ref(),
                    &raw,
                    values.len().to_u64(),
                    state.fill.pattern(state.element_size),
                )
                .unwrap();
                state.index = AppendIndex::Allocated(
                    initialize_ea_index(&mut store, address, index, &initialization).unwrap(),
                );
                plan
            }
        };
        apply_ea_append(&mut store, state.index.allocated_mut().unwrap(), plan, 4).unwrap();
        let index = state.index.allocated().unwrap();
        assert_eq!(index.current_dim, u64::try_from(end).unwrap());
        assert_eq!(
            index.num_chunks,
            u64::try_from(end).unwrap().div_ceil(chunk)
        );
        assert_eq!(index.offset_width, offsets);
        assert_eq!(index.length_width, LengthWidth::Eight);
        assert_eq!(
            index.allocation_stats.unwrap().max_idx_set,
            index.num_chunks
        );
        if end == 331 && chunk == 1 {
            let array = metadata_io::read_header(&store, index.ea_addr).unwrap();
            let bytes = IndexBlockBytes::read(&store, &array).unwrap();
            let block = bytes.parse().unwrap();
            let secondary = block
                .super_blocks()
                .next()
                .expect("the emitted index must address a secondary block")
                .unwrap();
            let range = secondary.parse_range();
            metadata_io::require_range(&store, range).unwrap();
            let bytes = store
                .read_exact_at(range.start().get(), range.length().to_usize().unwrap())
                .unwrap();
            secondary
                .parse(FileRegion::new(range, &bytes).unwrap())
                .unwrap();
        }
        assert_eq!(store.superblock.eof_address, store.data.len().to_u64());
        let issued = directory.path().join("issued.h5");
        fs::write(&issued, &store.data).unwrap();
        let native = file::libhdf5_open(&issued);
        let properties = native.create_plist().unwrap().get_sizes().unwrap();
        assert_eq!(
            usize::from(properties.sizeof_addr),
            usize::from(offsets.get())
        );
        assert_eq!(
            usize::from(properties.sizeof_size),
            usize::from(LengthWidth::Eight.get())
        );
        let dataset = native.dataset("d").unwrap();
        assert_eq!(dataset.shape(), [usize::try_from(end).unwrap()]);
        assert_eq!(dataset.chunk(), Some(vec![usize::try_from(chunk).unwrap()]));
        assert_eq!(
            dataset.read_raw::<i32>().unwrap(),
            (0..end).collect::<Vec<_>>()
        );
        drop(dataset);
        native.close().unwrap();
        let reopened = crate::File::from_bytes(store.data.clone()).unwrap();
        assert_eq!(
            reopened.dataset("d").unwrap().read_i32().unwrap(),
            (0..end).collect::<Vec<_>>()
        );
        reopened.close().unwrap();
        store = WindowProbeStore::open(store.data);
        expected = end;
    }
    directory.close().unwrap();
}

#[derive(Clone, Copy)]
enum Producer {
    Native,
    Rust,
}

impl Producer {
    fn write(self, path: &Path, widths: FormatWidths, initial: i32, chunk: u64, filtered: bool) {
        let values: Vec<i32> = (0..initial).collect();
        match self {
            Self::Native => {
                let native =
                    file::libhdf5_create_v110_with_widths(path, widths.offsets, widths.lengths, 0);
                let mut builder = native
                    .new_dataset_builder()
                    .empty_as(ByteOrder::LittleEndian.i32_datatype())
                    .shape((usize::try_from(initial).unwrap()..,))
                    .chunk((usize::try_from(chunk).unwrap(),));
                if filtered {
                    builder = builder.deflate(1);
                }
                let dataset = builder.create("d").unwrap();
                if initial != 0 {
                    dataset.write_raw(&values).unwrap();
                }
                drop(dataset);
                native.close().unwrap();
            }
            Self::Rust => {
                let mut builder = FileBuilder::new();
                builder
                    .with_offset_width(widths.offsets)
                    .with_length_width(widths.lengths);
                let dataset = builder
                    .create_dataset("d")
                    .with_i32_data(&values)
                    .with_shape(&[u64::try_from(initial).unwrap()])
                    .with_maxshape(&[MaxExtent::Unlimited])
                    .with_chunks(&[chunk]);
                if filtered {
                    dataset.with_deflate(DeflateLevel::One);
                }
                fs::write(path, builder.finish().unwrap()).unwrap();
            }
        }
    }
}
