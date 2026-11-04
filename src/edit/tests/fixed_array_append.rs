use std::fs::OpenOptions;
use std::io::Seek;
use std::io::SeekFrom;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use rstest::rstest;
use test_util::allocation;
use test_util::allocation::Measured;
use test_util::checksum;
use test_util::disk_log;
use test_util::disk_log::DiskOp;

use crate::File;
use crate::FileBuilder;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::SyncPolicy;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::AppendOptions;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::edit::observations;
use crate::image::WriteBuffering;
use crate::message_type::MessageType;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::one_chunk(1)]
#[case::batch(32)]
fn fixed_array_appends_bound_scratch_and_batch_metadata_io(#[case] chunks: u32) {
    let small = measure(1_024, chunks);
    let large = measure(65_536, chunks);
    assert!(
        large.peak_bytes <= small.peak_bytes + 4_096,
        "{small:?}; {large:?}"
    );
}

fn measure(entries: u64, chunks: u32) -> Measured {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounded.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
    builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(entries * 4)])
        .with_chunks(&[4]);
    builder.write(&path).unwrap();
    let bytes = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicU64::new(0));
    let mut engine =
        WriteEngine::open_prepared_counting(&path, Arc::clone(&bytes), Arc::clone(&calls))
            .unwrap()
            .into_engine();
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let target = ObjectPathBuf::parse("d");
    observations::reset_chunk_ownership_analyses();
    engine.append_geometry(AppendTarget::Path(&target)).unwrap();
    assert_eq!(observations::chunk_ownership_analyses(), 1);
    let block = data_block(&engine);
    assert!(engine.image.reserve_write_observations(2_048));
    let prior_writes = engine.image.issued_write_order().len();
    let end = i32::try_from(chunks * 4).unwrap();
    let raw = (1..end).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
    let region = format!("fixed_array_{entries}_{chunks}");
    let (result, measured) = allocation::measure(&region, || {
        engine.append_inplace_bytes(
            AppendTarget::Path(&target),
            AppendOptions::new(),
            None,
            &raw,
            4,
        )
    });
    result.unwrap();
    assert!(engine.image.issued_write_order().len() > prior_writes);
    assert_eq!(measured.live_bytes, 0, "{measured:?}");
    assert!(measured.peak_bytes < 32_768, "{measured:?}");
    assert!(measured.blocks > 0, "{measured:?}");
    let raw = (end..end * 2)
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    bytes.store(0, Ordering::Relaxed);
    calls.store(0, Ordering::Relaxed);
    disk_log::start();
    engine
        .append_inplace_bytes(
            AppendTarget::Path(&target),
            AppendOptions::new(),
            None,
            &raw,
            4,
        )
        .unwrap();
    let recording = disk_log::finish();
    let metadata = recording
        .operations
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { offset, bytes }
                if *offset < block.end().get()
                    && *offset + u64::try_from(bytes.len()).unwrap() > block.start().get() =>
            {
                Some(u64::try_from(bytes.len()).unwrap())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        metadata,
        [
            u64::from(chunks) * u64::from(engine.context.widths().offsets.get()),
            u64::try_from(checksum::CHECKSUM).unwrap()
        ]
    );
    assert!(
        bytes.load(Ordering::Relaxed) < 65_536,
        "{entries}: {} bytes",
        bytes.load(Ordering::Relaxed)
    );
    assert!(
        calls.load(Ordering::Relaxed) < 80 + u64::from(chunks) * 2,
        "{entries}: {} calls",
        calls.load(Ordering::Relaxed)
    );
    assert_eq!(observations::chunk_ownership_analyses(), 1);
    bytes.store(0, Ordering::Relaxed);
    let control_region = format!("fixed_array_full_index_{entries}_{chunks}");
    let (control, control_measured) = allocation::measure(&control_region, || {
        engine
            .image()
            .read_exact_at(
                block.start().get(),
                usize::try_from(block.length()).unwrap(),
            )
            .unwrap()
    });
    assert_eq!(u64::try_from(control.len()).unwrap(), block.length());
    assert_eq!(bytes.load(Ordering::Relaxed), block.length());
    assert!(
        control_measured.peak_bytes >= block.length(),
        "{control_measured:?}"
    );
    assert!(control_measured.blocks > 0, "{control_measured:?}");
    if entries > 1_024 {
        assert!(control_measured.peak_bytes > 32_768, "{control_measured:?}");
    }
    drop(engine);
    let mut output = OpenOptions::new().write(true).open(&path).unwrap();
    output.seek(SeekFrom::Start(block.start().get())).unwrap();
    disk_log::start();
    disk_log::write_all(&mut output, block.start().get(), &control).unwrap();
    let control_recording = disk_log::finish();
    assert_eq!(
        control_recording.operations,
        [DiskOp::Write {
            offset: block.start().get(),
            bytes: control
        }]
    );
    assert!(block.length() > metadata.iter().sum::<u64>());
    drop(output);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..end * 2).collect::<Vec<_>>()
    );
    file.close().unwrap();
    measured
}

fn data_block(engine: &WriteEngine) -> FileRange {
    let header = *engine.located.keys().next().unwrap();
    let widths = engine.context.widths();
    let addresses = engine.context.addresses();
    let region =
        WriteEngine::gather_oh_messages(&engine.image(), header, addresses, widths).unwrap();
    let mut position = 0;
    let layout = loop {
        let record = region
            .layout()
            .next_message(&region, position)
            .unwrap()
            .unwrap();
        if record.msg_type == MessageType::DATA_LAYOUT {
            break DataLayout::parse_with_widths(record.body, widths).unwrap();
        }
        position = record.body_range.end;
    };
    let DataLayout::Chunked {
        index: ChunkIndexLayout::FixedArray {
            address: Some(address),
        },
        ..
    } = layout
    else {
        panic!("expected an allocated Fixed Array")
    };
    let locator = FixedArrayHeaderLocator::new(widths, addresses.effective_base(), address);
    let range = locator.parse_range().unwrap();
    let bytes = engine
        .image()
        .read_exact_at(
            range.start().get(),
            usize::try_from(range.length()).unwrap(),
        )
        .unwrap();
    locator
        .parse(FileRegion::new(range, &bytes).unwrap())
        .unwrap()
        .data_block()
        .unwrap()
        .range()
}
