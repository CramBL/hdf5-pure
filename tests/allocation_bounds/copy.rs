use std::fs;
use std::path::Path;

use hdf5_pure::{
    AttrValue, ChunkIndex, EditBacking, File, FileAccessProperties, FileBuilder, FileSpaceSettings,
    FileSpaceStrategy, Layout, LengthWidth, MemoryStrategy, OffsetWidth, SyncPolicy,
};
use hdf5_pure_format::__private::{AccessMode, FormatWidths, ObjectHeader, ObjectHeaderPrefix};
use rstest::rstest;
use test_util::allocation::{self, Measured};
use test_util::object_header::v2;

#[rstest]
#[case::narrow_offsets(OffsetWidth::Two, LengthWidth::Eight, 256, "narrow_offsets")]
#[case::narrow_lengths(OffsetWidth::Eight, LengthWidth::Two, 256, "narrow_lengths")]
#[case::small(OffsetWidth::Eight, LengthWidth::Eight, 32, "small")]
#[case::large(OffsetWidth::Eight, LengthWidth::Eight, 8192, "large")]
fn chunked_public_copy_bounds_header_allocations(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] padding: usize,
    #[case] case: &str,
    #[values(CopySource::SameFile, CopySource::OtherFile)] operation: CopySource,
) {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.h5");
    let destination_path = directory.path().join("target.h5");
    let widths = FormatWidths { offsets, lengths };
    let attribute = "x".repeat(padding);
    let mut source_builder = builder(widths);
    source_builder
        .create_dataset("values")
        .with_u32_data(&[1, 2])
        .with_chunks(&[1])
        .set_attr("padding", AttrValue::String(attribute.clone()));
    source_builder.write(&source_path).unwrap();
    let source_bytes = fs::read(&source_path).unwrap();
    let source_header = dataset_header(&source_bytes, widths);
    assert!(source_header.len() >= padding);
    let source = File::open(&source_path).unwrap();
    assert_values(&source, "values", &attribute);
    let empty = builder(widths).finish().unwrap();
    let baseline = match operation {
        CopySource::SameFile => &source_bytes,
        CopySource::OtherFile => &empty,
    };

    let region = format!("public_chunked_copy_{case}_{operation:?}");
    let (file, measured) = measure_copy(
        &destination_path,
        baseline,
        &source,
        operation,
        None,
        &region,
    );
    assert_result(file, &destination_path, widths, operation, &attribute);
    let (file, control) = measure_copy(
        &destination_path,
        baseline,
        &source,
        operation,
        Some(source_header),
        &format!("{region}_clone"),
    );
    assert_result(file, &destination_path, widths, operation, &attribute);
    source.close().unwrap();
    assert_eq!(fs::read(&source_path).unwrap(), source_bytes);

    // Same-file copy reads the source header during commit. Cross-file copy reads it while
    // staging the copy, so the measurements allow for that work in different regions.
    let header = source_header.len() as u64;
    let (stage_bytes, stage_live, commit_bytes, commit_peak) = match operation {
        CopySource::SameFile => (
            4 * 1024,
            1024,
            13 * header + 20 * 1024,
            4 * header + 12 * 1024,
        ),
        CopySource::OtherFile => (
            4 * header + 8 * 1024,
            header + 4 * 1024,
            8 * header + 16 * 1024,
            3 * header + 12 * 1024,
        ),
    };
    assert!(
        (1..stage_bytes).contains(&measured.stage.bytes),
        "{measured:?}"
    );
    assert!(measured.stage.live_bytes < stage_live, "{measured:?}");
    assert!(
        (1..commit_bytes).contains(&measured.commit.bytes),
        "{measured:?}"
    );
    assert!(measured.commit.peak_bytes < commit_peak, "{measured:?}");
    assert!(measured.commit.live_bytes < 4 * 1024, "{measured:?}");
    assert!(measured.close.bytes < 4 * 1024, "{measured:?}");
    assert!(
        (1..4 * 1024).contains(&measured.reopen.bytes),
        "{measured:?}"
    );

    assert!(control.commit.live_bytes >= header, "{control:?}");
    assert!(
        control.commit.peak_bytes >= measured.commit.peak_bytes + header / 2,
        "{measured:?}; {control:?}"
    );
    if padding == 8192 {
        assert!(control.commit.bytes >= commit_bytes, "{control:?}");
        assert!(control.commit.peak_bytes >= commit_peak, "{control:?}");
    }
}

fn builder(widths: FormatWidths) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(widths.offsets)
        .with_length_width(widths.lengths)
        .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
    builder
}

#[derive(Clone, Copy, Debug)]
enum CopySource {
    SameFile,
    OtherFile,
}

#[derive(Debug)]
struct MeasuredCopy {
    stage: Measured,
    commit: Measured,
    close: Measured,
    reopen: Measured,
}

fn measure_copy(
    path: &Path,
    baseline: &[u8],
    source: &File,
    operation: CopySource,
    extra_header: Option<&[u8]>,
    region: &str,
) -> (File, MeasuredCopy) {
    fs::write(path, baseline).unwrap();
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
    let ((), stage) = allocation::measure(&format!("{region}_stage"), || {
        match operation {
            CopySource::SameFile => file.copy("values", "copied"),
            CopySource::OtherFile => file.copy_from(source, "values", "copied"),
        }
        .unwrap();
    });
    assert!(file.has_staged_edits());
    let (retained, commit) = allocation::measure(&format!("{region}_commit"), || {
        let retained = extra_header.map(|bytes| std::hint::black_box(bytes.to_vec()));
        file.commit().unwrap();
        retained
    });
    assert_eq!(retained.as_deref(), extra_header);
    drop(retained);
    assert!(!file.has_staged_edits());
    let ((), close) = allocation::measure(&format!("{region}_close"), || file.close().unwrap());
    let (reopened, reopen) = allocation::measure(&format!("{region}_reopen"), || {
        File::open_streaming(path).unwrap()
    });
    (
        reopened,
        MeasuredCopy {
            stage,
            commit,
            close,
            reopen,
        },
    )
}

fn dataset_header(bytes: &[u8], widths: FormatWidths) -> &[u8] {
    let at = test_util::bytes::signature_offsets(bytes, v2::SIGNATURE)
        .into_iter()
        .find(|&at| {
            ObjectHeader::parse_with_widths(bytes, AccessMode::ReadOnly, at, widths)
                .unwrap()
                .messages
                .iter()
                .any(|message| message.msg_type == hdf5_pure::MessageType::DATA_LAYOUT)
        })
        .unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[at..]).unwrap();
    let end = at
        + prefix.len
        + usize::try_from(prefix.chunk0_size).unwrap()
        + test_util::checksum::CHECKSUM;
    &bytes[at..end]
}

fn assert_result(
    file: File,
    path: &Path,
    widths: FormatWidths,
    operation: CopySource,
    attribute: &str,
) {
    assert_eq!(file.superblock().offset_size, widths.offsets);
    assert_eq!(file.superblock().length_size, widths.lengths);
    if matches!(operation, CopySource::SameFile) {
        assert_values(&file, "values", attribute);
    }
    assert_values(&file, "copied", attribute);
    let chunks = file.dataset("copied").unwrap().chunks().unwrap();
    file.close().unwrap();
    let bytes = fs::read(path).unwrap();
    assert_eq!(chunks.len(), 2);
    for (index, chunk) in chunks.into_iter().enumerate() {
        assert_eq!(chunk.offset, [index as u64]);
        assert_eq!(chunk.storage_size, 4);
        assert_eq!(chunk.filter_mask().raw_bits(), 0);
        let at = usize::try_from(chunk.address).unwrap();
        assert_eq!(&bytes[at..at + 4], &(index as u32 + 1).to_le_bytes());
    }
}

fn assert_values(file: &File, name: &str, attribute: &str) {
    let dataset = file.dataset(name).unwrap();
    assert_eq!(dataset.shape().unwrap(), [2]);
    assert_eq!(
        dataset.layout().unwrap(),
        Layout::Chunked {
            chunk_shape: vec![1],
            index: ChunkIndex::FixedArray
        }
    );
    assert_eq!(dataset.read_u32().unwrap(), [1, 2]);
    let attributes = dataset.attrs().unwrap();
    assert_eq!(attributes.len(), 1);
    assert_eq!(attributes["padding"], AttrValue::String(attribute.into()));
}
