use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::Mutex;

use rstest::rstest;

use super::super::REPACK_BUDGET;
use super::super::RepackSource;
use super::super::SourceObjects;
use crate::File;
use crate::FileBuilder;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::LibVer;
use crate::MaxExtent;
use crate::Source;
use crate::error::FormatError;
use crate::file_writer::source::Plan;
use crate::source::BytesSource;

#[rstest]
#[case::fixed(false, None)]
#[case::extensible(true, None)]
#[case::fixed_page(false, Some(FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true)))]
#[case::extensible_page(true, Some(FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true)))]
fn array_descriptors_read_metadata_before_fetching_each_payload_once(
    #[case] unlimited: bool,
    #[case] settings: Option<FileSpaceSettings>,
    #[values(false, true)] filtered: bool,
) {
    let values: Vec<_> = (0_i32..128).collect();
    let mut builder = FileBuilder::new();
    for index in 0..3 {
        let dataset = builder.create_dataset(&format!("d{index}"));
        dataset.with_i32_data(&values).with_chunks(&[32]);
        if unlimited {
            dataset.with_maxshape(&[MaxExtent::Unlimited]);
        }
        if filtered {
            dataset.with_fletcher32();
        }
    }
    let image = builder.finish().unwrap();
    let observation = File::from_bytes(image.clone()).unwrap();
    let payload: Vec<_> = (0..3)
        .flat_map(|index| {
            observation
                .dataset(&format!("d{index}"))
                .unwrap()
                .chunks()
                .unwrap()
                .into_iter()
                .map(|chunk| (chunk.address, chunk.storage_size as usize))
        })
        .collect();
    assert_eq!(payload.len(), 12);
    assert!(
        payload
            .iter()
            .all(|&(_, length)| length == if filtered { 132 } else { 128 })
    );
    drop(observation);
    let reads = Arc::new(Mutex::new(Vec::new()));
    let file = Arc::new(
        File::from_source(CountedRanges {
            source: BytesSource::new(image),
            reads: Arc::clone(&reads),
        })
        .unwrap(),
    );
    let objects = SourceObjects::discover(&file, &BTreeSet::new(), REPACK_BUDGET).unwrap();
    assert!(!reads.lock().unwrap().is_empty());
    reads.lock().unwrap().clear();
    let plan = Plan::new(RepackSource { file, objects }, settings, None, LibVer::V110).unwrap();
    let planning = reads.lock().unwrap().clone();
    assert!(!planning.is_empty());
    assert!(planning.iter().all(|&(start, length)| {
        payload
            .iter()
            .all(|&(chunk, size)| start + length as u64 <= chunk || start >= chunk + size as u64)
    }));
    reads.lock().unwrap().clear();
    let output = plan.finish().unwrap();
    let emission = reads.lock().unwrap().clone();
    let payload_reads: Vec<_> = emission
        .iter()
        .copied()
        .filter(|&(start, length)| {
            payload
                .iter()
                .any(|&(chunk, size)| start < chunk + size as u64 && start + length as u64 > chunk)
        })
        .collect();
    assert_eq!(payload_reads, payload);
    eprintln!(
        "planning: {} calls/{} bytes; emission: {} calls/{} bytes; payload: {} calls/{} bytes",
        planning.len(),
        planning.iter().map(|(_, size)| size).sum::<usize>(),
        emission.len(),
        emission.iter().map(|(_, size)| size).sum::<usize>(),
        payload_reads.len(),
        payload_reads.iter().map(|(_, size)| size).sum::<usize>()
    );
    let file = File::from_bytes(output).unwrap();
    assert_eq!(file.file_space_settings().unwrap(), settings);
    for index in 0..3 {
        let dataset = file.dataset(&format!("d{index}")).unwrap();
        assert_eq!(dataset.shape().unwrap(), [128]);
        assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![32]));
        assert_eq!(dataset.read_i32().unwrap(), values);
    }
}

struct CountedRanges {
    source: BytesSource<Vec<u8>>,
    reads: Arc<Mutex<Vec<(u64, usize)>>>,
}

impl Source for CountedRanges {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        let Self { source, reads } = self;
        reads.lock().unwrap().push((offset, bytes.len()));
        source.read_at(offset, bytes)
    }
}
