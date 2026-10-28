//! Allocation bounds and payload-read checks for PAGE file creation.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use test_util::allocation;

use super::FileWriter;
use crate::File;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::Layout;
use crate::chunked_write::ChunkProvider;
use crate::error::FormatError;

#[rstest]
#[case::allocated_small(256, false, "page_small_creation", "page_small_control")]
#[case::many_allocated_small(1024, false, "page_many_small_creation", "page_many_small_control")]
#[case::persistent_small(
    256,
    true,
    "page_persistent_small_creation",
    "page_persistent_small_control"
)]
#[case::many_persistent_small(
    1024,
    true,
    "page_many_persistent_small_creation",
    "page_many_persistent_small_control"
)]
fn paged_creation_bounds_metadata_and_reads_each_payload_once(
    #[case] count: usize,
    #[case] persist: bool,
    #[case] region: &'static str,
    #[case] control_region: &'static str,
) {
    let settings = FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_persistence(persist)
        .with_page_size(FileSpacePageSize::try_from(512).unwrap());
    let reads = Arc::new(AtomicUsize::new(0));
    let names: Vec<_> = (0..count).map(|index| format!("d{index:06}")).collect();
    let mut writer = FileWriter::new();
    writer.with_file_space_settings(settings);
    for (index, name) in names.iter().enumerate() {
        writer.create_dataset(name).with_produced_data(
            crate::make_i32_type(),
            &[4],
            16,
            16,
            Box::new(CountingProvider {
                value: i32::try_from(index).unwrap(),
                reads: Arc::clone(&reads),
            }),
        );
    }
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    let (control, positive) =
        allocation::measure(control_region, || std::hint::black_box(vec![0_u8; 1024]));
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.peak_bytes >= 1024, "{positive}");
    drop(control);
    let (image, measured) = allocation::measure(region, || writer.finish().unwrap());
    assert_eq!(reads.load(Ordering::Relaxed), count);
    assert_eq!(measured.live_bytes, image.capacity() as u64, "{measured}");
    assert!(measured.peak_bytes >= image.capacity() as u64, "{measured}");
    assert!(
        measured.peak_bytes <= image.capacity() as u64 + count as u64 * 8192,
        "{measured}"
    );
    let file = File::from_bytes(image).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), Some(settings));
    let mut observed = file.root().datasets().unwrap();
    observed.sort_unstable();
    assert_eq!(observed, names);
    assert_eq!(file.root().groups().unwrap(), Vec::<String>::new());
    for (index, name) in names.iter().enumerate() {
        let dataset = file.dataset(name).unwrap();
        assert_eq!(dataset.shape().unwrap(), [4]);
        let layout = dataset.layout().unwrap();
        let Layout::Contiguous {
            address: Some(_),
            size,
        } = layout
        else {
            panic!("expected allocated contiguous storage, got {layout:?}");
        };
        assert_eq!(size, 16);
        assert_eq!(
            dataset.read_i32().unwrap(),
            [i32::try_from(index).unwrap(); 4]
        );
    }
}

struct CountingProvider {
    value: i32,
    reads: Arc<AtomicUsize>,
}

impl ChunkProvider for CountingProvider {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        if index != 0 {
            return Err(FormatError::Internal(
                "a contiguous test payload has one block".into(),
            ));
        }
        let Self { value, reads } = self;
        reads.fetch_add(1, Ordering::Relaxed);
        for _ in 0..4 {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}
