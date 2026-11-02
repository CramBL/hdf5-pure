//! Native fixtures and page-buffered readback for edited object headers.

use std::path::Path;

use hdf5::dataset::Layout;
use hdf5::datatype::ByteOrder as NativeByteOrder;
use hdf5::plist::file_create::FileSpaceStrategy as NativeStrategy;

use crate::numeric::ByteOrder;

/// The dataset layout in a native source fixture.
#[derive(Clone, Copy)]
pub enum SourceLayout {
    Chunked,
    Compact,
    Contiguous,
}

/// The additional edit checked alongside source values and group attributes.
#[derive(Clone, Copy)]
pub enum Observation {
    /// A newly created dataset and its padding attribute.
    Added,
    /// A padding attribute added to the source dataset.
    Attribute,
    /// A copied subtree and its retained dataset attribute.
    Copied,
    /// The source dataset's values and retained attribute.
    Values,
}

/// Checks edited values, attributes, and byte order through a native page-buffered reader.
///
/// Opens with a page buffer four times `page` bytes long. Checks `expected` values in
/// `parent/nested/data`, padding attributes on its ancestor groups, and the additional edit
/// selected by `observation`. Checks little-endian byte order for each dataset it reads.
///
/// # Panics
///
/// Panics if a value, attribute, or dataset byte order differs, or if libhdf5 cannot open,
/// read, or close the file.
pub fn assert_edited(
    path: &Path,
    page: u64,
    expected: &[i32],
    padding: usize,
    observation: Observation,
) {
    let native = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.page_buffer_size(usize::try_from(4 * page).unwrap(), 0, 0)
        })
        .open(path)
        .unwrap();
    assert_eq!(
        native
            .dataset("parent/nested/data")
            .unwrap()
            .dtype()
            .unwrap()
            .byte_order(),
        NativeByteOrder::LittleEndian
    );
    assert_eq!(
        native
            .dataset("parent/nested/data")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        expected
    );
    assert_eq!(
        native.attr("padding").unwrap().read_raw::<u8>().unwrap(),
        vec![31; padding]
    );
    assert_eq!(
        native
            .group("parent")
            .unwrap()
            .attr("padding")
            .unwrap()
            .read_raw::<u8>()
            .unwrap(),
        vec![31; padding]
    );
    assert_eq!(
        native
            .group("parent/nested")
            .unwrap()
            .attr("padding")
            .unwrap()
            .read_raw::<u8>()
            .unwrap(),
        vec![31; padding]
    );
    if matches!(observation, Observation::Added) {
        assert_eq!(
            native
                .dataset("parent/nested/added")
                .unwrap()
                .dtype()
                .unwrap()
                .byte_order(),
            NativeByteOrder::LittleEndian
        );
        assert_eq!(
            native
                .dataset("parent/nested/added")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [11, 12, 13, 14]
        );
        assert_eq!(
            native
                .dataset("parent/nested/added")
                .unwrap()
                .attr("padding")
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![31; padding]
        );
    }
    if matches!(observation, Observation::Attribute) {
        assert_eq!(
            native
                .dataset("parent/nested/data")
                .unwrap()
                .attr("padding")
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![31; padding]
        );
    }
    if matches!(observation, Observation::Copied) {
        assert_eq!(
            native
                .dataset("copied/nested/data")
                .unwrap()
                .dtype()
                .unwrap()
                .byte_order(),
            NativeByteOrder::LittleEndian
        );
        assert_eq!(
            native
                .dataset("copied/nested/data")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [1, 2, 3, 4]
        );
    }
    assert_eq!(
        native
            .dataset("parent/nested/data")
            .unwrap()
            .attr("source_padding")
            .unwrap()
            .read_raw::<u8>()
            .unwrap(),
        vec![29; padding]
    );
    if matches!(observation, Observation::Copied) {
        assert_eq!(
            native
                .dataset("copied/nested/data")
                .unwrap()
                .attr("source_padding")
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![29; padding]
        );
    }
    native.close().unwrap();
}

/// Writes a native PAGE fixture with a nested dataset and a retained padding attribute.
///
/// Creates `parent/nested/data` with a little-endian [`i32`] datatype, values `[1, 2, 3, 4]`,
/// and a `source_padding` attribute containing `padding` copies of 29. Sets the page size to
/// `page` bytes, the userblock size to `base` bytes, and free-space persistence to `persist`.
///
/// # Panics
///
/// Panics if libhdf5 cannot create, write, or close the fixture.
pub fn write_native(
    path: &Path,
    page: u64,
    base: u64,
    layout: SourceLayout,
    persist: bool,
    padding: usize,
) {
    let file = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties
                .userblock(base)
                .file_space_page_size(page)
                .file_space_strategy(NativeStrategy::FreeSpaceManager {
                    paged: true,
                    persist,
                    threshold: 1,
                })
        })
        .create(path)
        .unwrap();
    let parent = file.create_group("parent").unwrap();
    let nested = parent.create_group("nested").unwrap();
    let builder = nested
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype());
    let dataset = if matches!(layout, SourceLayout::Chunked) {
        builder.shape((4..,)).chunk([2]).create("data").unwrap()
    } else if matches!(layout, SourceLayout::Compact) {
        builder
            .shape([4])
            .layout(Layout::Compact)
            .create("data")
            .unwrap()
    } else {
        builder.shape([4]).create("data").unwrap()
    };
    dataset.write(&[1i32, 2, 3, 4]).unwrap();
    dataset
        .new_attr::<u8>()
        .shape([padding])
        .create("source_padding")
        .unwrap()
        .write_raw(&vec![29u8; padding])
        .unwrap();
    drop(dataset);
    drop(nested);
    drop(parent);
    file.close().unwrap();
}
