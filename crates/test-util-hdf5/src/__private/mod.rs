//! Fixtures and observations shared by public API tests and owner-local inspection tests.

use hdf5_pure::File;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::ObjectHeader;
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;

#[cfg(feature = "__hdf5-1.10")]
pub mod array_index;
#[cfg(feature = "__hdf5-1.10")]
pub mod c_object;
#[cfg(feature = "__hdf5-1.10")]
pub mod ownership_append;
#[cfg(feature = "__hdf5-1.10")]
pub mod unallocated_append;

/// Sets the extensible-array index address to undefined in a single-dataset fixture.
///
/// Pass a file with one data layout message in chunk zero of a version 2 object header. The
/// dataset must use an extensible-array index. The object header checksum is updated after the
/// mutation.
///
/// The index address in a version 4 data layout message is defined in "The Data Layout Message"
/// of the [format specification, version 4.0][spec].
///
/// # Panics
///
/// Panics if the file or object headers cannot be parsed, there is not exactly one data layout
/// message, the data layout message is outside chunk zero, or the modified layout does not
/// describe an unallocated extensible-array index.
///
/// # Examples
///
/// ```
/// use hdf5_pure::{File, FileBuilder, MaxExtent};
/// use test_util_hdf5::__private;
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("d")
///     .with_i32_data(&[])
///     .with_shape(&[0])
///     .with_chunks(&[4])
///     .with_maxshape(&[MaxExtent::Unlimited]);
/// let original = builder.finish()?;
/// let mut image = original.clone();
/// __private::clear_index_header(&mut image);
/// assert_ne!(image, original);
/// let file = File::from_bytes(image)?;
/// assert_eq!(file.dataset("d")?.read_i32()?, Vec::<i32>::new());
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub fn clear_index_header(image: &mut [u8]) {
    let file = File::from_bytes(image.to_vec()).unwrap();
    let widths = (
        file.superblock().offset_size.get(),
        file.superblock().length_size.get(),
    );
    drop(file);
    let candidates: Vec<_> = test_util::bytes::signature_offsets(image, v2::SIGNATURE)
        .into_iter()
        .filter_map(|at| {
            ObjectHeader::parse(image, AccessMode::ReadOnly, at, widths.0, widths.1)
                .unwrap()
                .messages
                .into_iter()
                .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
                .map(|message| (at, message.data))
        })
        .collect();
    assert_eq!(candidates.len(), 1);
    let (header, mut layout) = candidates.into_iter().next().unwrap();
    let start = layout.len() - usize::from(widths.0);
    layout[start..].fill(u8::MAX);
    assert!(matches!(
        DataLayout::parse(&layout, widths.0, widths.1).unwrap(),
        DataLayout::Chunked {
            index: ChunkIndexLayout::ExtensibleArray { address: None, .. },
            ..
        }
    ));
    v2::replace_message(image, header, FixtureMessage::DATA_LAYOUT, &layout);
}
