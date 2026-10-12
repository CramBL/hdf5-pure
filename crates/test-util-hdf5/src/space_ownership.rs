//! Native fixtures with storage outside the complete ownership analysis.

use std::path::Path;

use hdf5::File as Hdf5File;
use hdf5::file::LibraryVersion;
use hdf5::plist::GroupCreate;
use hdf5::plist::file_create::SharedMessageIndex;
use hdf5::plist::file_create::SharedMessageType;

/// Selects the messages shared between objects in a native fixture.
#[derive(Clone, Copy, Debug)]
pub enum SharedStorage {
    /// Shares entire attribute messages.
    Attribute,
    /// Shares the datatype and dataspace within each attribute.
    AttributeFields,
    /// Shares the dataspace of chunked datasets.
    Dataspace,
}

/// Writes shared-message storage with `H5Pset_shared_mesg_index`.
///
/// Attribute fixtures use compact storage unless `dense` is true. Dataset fixtures use
/// chunked storage and ignore `dense`.
///
/// # Panics
///
/// Panics if the C library cannot create, write, read, or close the fixture.
pub fn libhdf5_write_shared_storage(path: &Path, storage: SharedStorage, dense: bool) {
    let message_types = match storage {
        SharedStorage::Attribute => SharedMessageType::ATTRIBUTE,
        SharedStorage::AttributeFields => {
            SharedMessageType::DATATYPE | SharedMessageType::SIMPLE_DATASPACE
        }
        SharedStorage::Dataspace => SharedMessageType::SIMPLE_DATASPACE,
    };
    let file = Hdf5File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties.shared_mesg_indexes(&[SharedMessageIndex {
                message_types,
                min_message_size: 1,
            }])
        })
        .create(path)
        .unwrap();
    match storage {
        SharedStorage::Dataspace => {
            for index in 0..2 {
                let dataset = file
                    .new_dataset::<i32>()
                    .shape([8])
                    .chunk([4])
                    .create(format!("d{index}").as_str())
                    .unwrap();
                dataset.write(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
                assert_eq!(dataset.read_raw::<i32>().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
            }
        }
        SharedStorage::Attribute | SharedStorage::AttributeFields => {
            let properties = if dense {
                GroupCreate::build()
                    .attr_phase_change(0, 0)
                    .finish()
                    .unwrap()
            } else {
                GroupCreate::build().finish().unwrap()
            };
            for index in 0..2 {
                let group = file
                    .create_group_builder()
                    .set_gcpl(&properties)
                    .create(format!("g{index}").as_str())
                    .unwrap();
                if dense {
                    let values = vec![17_i64; 8_192];
                    let attribute = group
                        .new_attr::<i64>()
                        .shape([values.len()])
                        .create("shared")
                        .unwrap();
                    attribute.write(&values).unwrap();
                    assert_eq!(attribute.read_raw::<i64>().unwrap(), values);
                } else {
                    let attribute = group.new_attr::<i64>().shape(()).create("shared").unwrap();
                    attribute.write_scalar(&17).unwrap();
                    assert_eq!(attribute.read_scalar::<i64>().unwrap(), 17);
                }
            }
        }
    }
    file.close().unwrap();
}

/// Writes a dense group with a filtered fractal heap through `H5Pset_deflate`.
///
/// # Panics
///
/// Panics if the C library cannot set the filter, create or read the group, or close the file.
pub fn libhdf5_write_filtered_heap(path: &Path) {
    let properties = GroupCreate::build()
        .obj_track_times(false)
        .finish()
        .unwrap();
    assert_eq!(crate::group_filter::set_deflate(&properties, 6), 0);
    let file = Hdf5File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .create(path)
        .unwrap();
    let group = file
        .create_group_builder()
        .set_gcpl(&properties)
        .create("filtered")
        .unwrap();
    for index in 0..12 {
        group.create_group(&format!("g{index:02}")).unwrap();
    }
    let mut names = group.member_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        (0..12)
            .map(|index| format!("g{index:02}"))
            .collect::<Vec<_>>()
    );
    drop(group);
    file.close().unwrap();
}
