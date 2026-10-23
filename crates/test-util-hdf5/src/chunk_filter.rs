//! C-library fixtures with applied and skipped filters in the same chunk.

use std::mem;
use std::path::Path;

use hdf5::Extent;
use hdf5::filters::Filter as CFilter;
use hdf5::sync;
use hdf5_sys::h5d;
use hdf5_sys::h5p::H5P_DEFAULT;

use crate::file;
use crate::numeric::ByteOrder;

/// Writes shuffled little-endian integer chunks through `H5Dwrite_chunk` with deflate skipped.
///
/// Bits above the two-filter pipeline are passed through unchanged.
///
/// # Panics
///
/// Panics if the low two bits of `mask` differ from `0b10`, or if file creation or chunk writing
/// fails.
pub fn write_masked_chunks(path: &Path, mask: u32, chunks: &[[i32; 4]]) {
    assert_eq!(mask & PIPELINE_BITS, SKIPPED_DEFLATE);
    let file = file::libhdf5_create_v110(path);
    let dataset = file
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .shape([Extent::resizable(chunks.len() * 4)])
        .chunk([4])
        .set_filters(&[CFilter::Shuffle, CFilter::Deflate(6)])
        .create("d")
        .unwrap();
    for (index, chunk) in chunks.iter().enumerate() {
        let shuffled: Vec<u8> = (0..mem::size_of::<i32>())
            .flat_map(|byte| chunk.iter().map(move |value| value.to_le_bytes()[byte]))
            .collect();
        let offset = [u64::try_from(index * 4).unwrap()];
        let status = sync::sync(|| {
            // SAFETY: The live dataset is chunked with rank one. The offset and byte buffer remain
            // valid for the call, and their sizes describe one complete shuffled chunk.
            unsafe {
                h5d::H5Dwrite_chunk(
                    dataset.id(),
                    H5P_DEFAULT,
                    mask,
                    offset.as_ptr(),
                    shuffled.len(),
                    shuffled.as_ptr().cast(),
                )
            }
        });
        assert_eq!(status, 0);
    }
}

const PIPELINE_BITS: u32 = 0b11;
const SKIPPED_DEFLATE: u32 = 0b10;
