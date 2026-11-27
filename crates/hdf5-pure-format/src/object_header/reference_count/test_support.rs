use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use test_util::image::Image;

use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::error::FormatError;
use crate::object_header::MessageFilter;
use crate::object_header::{
    ObjectHeader, ObjectHeaderChunk, ObjectReferenceCount, ObjectReferenceCountObserver,
};
use crate::width::FormatWidths;

pub(super) fn observe(
    root: &[u8],
    child: Option<&[u8]>,
    bias: u64,
    width: u8,
) -> Result<ObjectReferenceCount, FormatError> {
    let mut image = Image::starting_with(&[]);
    image.place(512, root);
    if let Some(child) = child {
        image.place(1_024, child);
    }
    let bytes = image.build();
    let observer = ObjectReferenceCountObserver::new(PhysicalAddress::new(bias + 512));
    let mut error = None;
    let header = ObjectHeader::parse_from_source_with_chunk_observer(
        bytes.as_slice(),
        AccessMode::ReadWrite,
        PhysicalAddress::new(512),
        FormatWidths::from_sizes(width, width).unwrap(),
        BaseAddress::new(512),
        MessageFilter::Only(&mut |kind, _| {
            observer.observe_message_type(kind);
            false
        }),
        &mut |chunk| {
            let shift = |range: FileRange| {
                FileRange::new(range.start().checked_add(bias).unwrap(), range.length()).unwrap()
            };
            let chunk = ObjectHeaderChunk::new(
                shift(chunk.allocation()),
                FileRegion::new(shift(chunk.records().range()), chunk.records().bytes()).unwrap(),
                chunk.v2_layout(),
            );
            if error.is_none() {
                error = observer.observe(chunk).err();
            }
        },
    )?;
    if let Some(error) = error {
        return Err(error);
    }
    observer.finish(&header)
}
