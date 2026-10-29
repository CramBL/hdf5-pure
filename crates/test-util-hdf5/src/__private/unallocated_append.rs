//! Native empty-dataset fixtures with an unallocated Extensible Array index.

use std::fs;
use std::path::Path;

use hdf5::Extent;
use hdf5::plist::file_create::FileSpaceStrategy;
use hdf5_pure::{File, MessageType};
use hdf5_pure_core::__private::{BaseAddressExt, FileRegion};
use hdf5_pure_core::BaseAddress;
use hdf5_pure_format::__private::{AccessMode, ChunkIndexLayout, DataLayout, ObjectHeader};
use hdf5_pure_format::__private::{ExtensibleArrayHeaderLocator, FormatWidths};
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;

use crate::numeric::ByteOrder;

use super::c_object;

/// Asserts the allocated index geometry, contiguous header and index block, and free-space exclusion.
///
/// Pass the object header address returned by [`fixture`] and the same creation parameters.
///
/// # Panics
///
/// Panics if the file cannot be read or parsed, the index geometry differs from `parameters`,
/// the index block does not immediately follow its header, or either extends beyond the file
/// boundary or overlaps persisted free space.
pub fn inspect_initialized_index(path: &Path, header: usize, parameters: [u8; 5]) {
    let bytes = fs::read(path).unwrap();
    let messages = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, header, 8, 8).unwrap();
    let layout = messages
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let DataLayout::Chunked {
        index:
            ChunkIndexLayout::ExtensibleArray {
                address: Some(address),
                creation,
            },
        ..
    } = DataLayout::parse(&layout.data, 8, 8).unwrap()
    else {
        panic!("expected allocated EA layout");
    };
    assert_eq!(creation.encode(), parameters);
    let locator = ExtensibleArrayHeaderLocator::new(
        FormatWidths::from_sizes(8, 8).unwrap(),
        BaseAddress::new(0),
        address,
    );
    let range = locator.parse_range().unwrap();
    let array = locator
        .parse(
            FileRegion::new(
                range,
                &bytes[usize::try_from(range.start().get()).unwrap()
                    ..usize::try_from(range.end().get()).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let fields = array.header();
    assert_eq!(fields.bit_geometry(), creation.bit_geometry());
    assert_eq!(fields.idx_blk_elmts, creation.inline_elements());
    assert_eq!(fields.min_dblk_nelmts, creation.min_elements().get());
    assert_eq!(
        fields.super_blk_min_data_ptrs,
        creation.min_data_blocks().get()
    );
    let index = array.index_block().unwrap();
    assert_eq!(index.range().start(), array.range().end());
    let range = index.range();
    index
        .parse(
            FileRegion::new(
                range,
                &bytes[usize::try_from(range.start().get()).unwrap()
                    ..usize::try_from(range.end().get()).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    let pure = File::open(path).unwrap();
    assert!(range.end().get() <= pure.superblock().eof_address);
    let free = pure.persisted_free_sections().unwrap().unwrap();
    for section in free.free_sections().sections() {
        assert!(
            section.end_address() <= address.get() || section.address() >= range.end().get(),
            "allocated pair {}..{} overlaps free {}..{}",
            address.get(),
            range.end().get(),
            section.address(),
            section.end_address()
        );
    }
    pure.close().unwrap();
    assert_eq!(fs::read(path).unwrap(), bytes);
}

/// Writes a native empty dataset with reusable space and returns its object header address.
///
/// Dataset `d` has four-element chunks and a fill value of -7. If `filtered` is true, libhdf5
/// applies shuffle and deflate. If `paged` is true, libhdf5 uses paged free-space management.
/// The Extensible Array creation parameters are replaced with `parameters` after file closure.
///
/// # Panics
///
/// Panics if the fixture cannot be written or its layout differs from the expected unallocated index.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use test_util_hdf5::__private::unallocated_append;
///
/// # let _guard = test_util_hdf5::lock::libhdf5_guard();
/// # let directory = tempfile::tempdir()?;
/// let path = directory.path().join("empty.h5");
/// let parameters = [24, 8, 2, 8, 8];
/// let header = unallocated_append::fixture(&path, false, false, parameters);
/// let file = File::open_rw(&path)?;
/// let mut dataset = file.dataset("d")?;
/// dataset.append(&[1_i32, 2, 3, 4])?;
/// assert_eq!(dataset.read_i32()?, [1, 2, 3, 4]);
/// drop(dataset);
/// file.close()?;
/// unallocated_append::inspect_initialized_index(&path, header, parameters);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn fixture(path: &Path, filtered: bool, paged: bool, parameters: [u8; 5]) -> usize {
    let native = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties.file_space_strategy(FileSpaceStrategy::FreeSpaceManager {
                paged,
                persist: true,
                threshold: 1,
            })
        })
        .create(path)
        .unwrap();
    let builder = native
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .shape([Extent::resizable(0)])
        .chunk([4])
        .fill_value(-7_i32);
    let builder = if filtered {
        builder.shuffle().deflate(6)
    } else {
        builder
    };
    let target = builder.create("d").unwrap();
    drop(builder);
    assert_eq!(target.num_chunks(), Some(0));
    assert_eq!(target.storage_size(), 0);
    drop(target);
    let header = usize::try_from(c_object::header_address(&native, "d")).unwrap();
    let scratch = native
        .new_dataset_builder()
        .with_data(&vec![7_u8; 16_384])
        .create("scratch")
        .unwrap();
    let hole_end = scratch.offset().unwrap() + scratch.storage_size();
    drop(scratch);
    let ceiling = native
        .new_dataset_builder()
        .with_data(&vec![9_u8; 4096])
        .create("ceiling")
        .unwrap();
    assert!(ceiling.offset().unwrap() >= hole_end);
    drop(ceiling);
    native.unlink("scratch").unwrap();
    native.close().unwrap();
    let mut image = fs::read(path).unwrap();
    let mut layout = ObjectHeader::parse(&image, AccessMode::ReadOnly, header, 8, 8)
        .unwrap()
        .messages
        .into_iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap()
        .data;
    let candidates: Vec<_> = layout
        .windows(parameters.len())
        .enumerate()
        .filter_map(|(at, field)| (field == [32, 4, 4, 16, 10]).then_some(at))
        .collect();
    assert_eq!(candidates.len(), 1);
    let at = candidates[0];
    layout[at..at + parameters.len()].copy_from_slice(&parameters);
    v2::replace_message(&mut image, header, FixtureMessage::DATA_LAYOUT, &layout);
    assert_unallocated_layout(&image, header, parameters);
    fs::write(path, image).unwrap();
    header
}

/// Asserts an unallocated Extensible Array index with creation parameters `parameters`.
///
/// # Panics
///
/// Panics if `image` cannot be parsed with eight-byte addresses and lengths, the object header at
/// `header` has no Extensible Array layout, or the index is allocated or has different parameters.
pub fn assert_unallocated_layout(image: &[u8], header: usize, parameters: [u8; 5]) {
    let file = File::from_bytes(image.to_vec()).unwrap();
    assert_eq!(file.superblock().offset_size.get(), 8);
    assert_eq!(file.superblock().length_size.get(), 8);
    drop(file);
    let header = ObjectHeader::parse(image, AccessMode::ReadOnly, header, 8, 8).unwrap();
    let layout = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let DataLayout::Chunked {
        index: ChunkIndexLayout::ExtensibleArray { address, creation },
        ..
    } = DataLayout::parse(&layout.data, 8, 8).unwrap()
    else {
        panic!("expected an extensible-array layout");
    };
    assert_eq!(address, None);
    assert_eq!(creation.encode(), parameters);
}
