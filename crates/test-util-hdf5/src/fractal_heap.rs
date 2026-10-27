//! Filtered huge-object fixtures written through the bundled libhdf5.

use core::ffi::{c_int, c_uint, c_void};
use core::ptr;
use std::path::Path;

use hdf5::File;

/// Writes a Deflate-filtered huge object and its lookup datasets through libhdf5.
///
/// Pass `true` in `direct` for directly accessed huge-object IDs. The object contains 12,000 bytes,
/// with byte `i` equal to `i % 251`. The `heap_id` dataset contains its ID, and the scalar
/// `heap_address` dataset contains its heap address. The helper checks the complete payload
/// through libhdf5 before closing the file.
///
/// # Panics
///
/// Panics if the library version differs from HDF5 2.2.0, the file uses a non-native VOL
/// connector, an HDF5 operation fails, or the payload readback differs from the written bytes.
///
/// # Examples
///
/// ```
/// use test_util_hdf5::fractal_heap;
///
/// # let directory = tempfile::tempdir()?;
/// let path = directory.path().join("heap.h5");
/// fractal_heap::write_filtered_huge(&path, true);
/// let file = hdf5::File::open(&path)?;
/// assert_eq!(file.dataset("heap_id")?.shape(), [29]);
/// assert_eq!(file.dataset("heap_address")?.shape(), []);
/// file.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_filtered_huge(path: &Path, direct: bool) {
    assert_eq!(hdf5::library_version(), (2, 2, 0));
    let file = File::with_options()
        .with_fapl(|properties| properties.libver_latest())
        .create(path)
        .unwrap();
    drop(file.create_group("context").unwrap());
    let payload: Vec<_> = (0..12_000)
        .map(|index| u8::try_from(index % 251).unwrap())
        .collect();
    let mut result = None;
    file.iter_visit_default(|_, _| {
        let mut parameters = HeapCreate {
            managed: DoublingTable {
                width: 4,
                start_block_size: 512,
                max_direct_size: 65_536,
                max_index: 32,
                start_root_rows: 1,
            },
            checksum_dblocks: true,
            max_man_size: 4096,
            // H5HF__hdr_create in H5HFhdr.c, HDF5 2.2.0, uses one for direct huge IDs.
            id_len: u16::from(direct),
            pipeline: Pipeline::default(),
        };
        let mut id = [0; 64];
        let mut id_len = 0;
        let mut address = 0;
        let mut decoded = vec![0; payload.len()];
        let level = 6;
        // SAFETY: The file remains live under H5Literate's library lock and API context, and
        // the native VOL check precedes use of its underlying file pointer.
        // Parameters match the bundled 2.2.0 ABI, ID capacity is checked before insertion, and
        // the readback buffer has the inserted payload's length.
        unsafe {
            let object = H5VL_vol_object(file.id());
            assert!(!object.is_null());
            let mut native = false;
            assert!(H5VL_object_is_native(object, &mut native) >= 0);
            assert!(native);
            let internal = H5VL_object(file.id());
            assert!(!internal.is_null());
            assert!(H5AC_ignore_tags(internal) >= 0);
            assert!(H5Z_append(&mut parameters.pipeline, FILTER_DEFLATE, 0, 1, &level) >= 0);
            let heap = H5HF_create(internal, &parameters);
            assert!(!heap.is_null());
            assert!(H5HF_get_id_len(heap, &mut id_len) >= 0);
            assert!(id_len <= id.len());
            assert!(H5HF_get_heap_addr(heap, &mut address) >= 0);
            assert!(
                H5HF_insert(
                    heap,
                    payload.len(),
                    payload.as_ptr().cast(),
                    id.as_mut_ptr()
                ) >= 0
            );
            assert!(H5HF_read(heap, id.as_ptr(), decoded.as_mut_ptr().cast()) >= 0);
            assert_eq!(decoded, payload);
            assert!(H5HF_close(heap) >= 0);
            assert!(H5O_msg_reset(FILTER_PIPELINE_MESSAGE_ID, &mut parameters.pipeline) >= 0);
        }
        result = Some((address, id[..id_len].to_vec()));
        Ok(())
    })
    .unwrap();
    let (address, id) = result.unwrap();
    file.new_dataset::<u8>()
        .shape([id.len()])
        .create("heap_id")
        .unwrap()
        .write_raw(&id)
        .unwrap();
    file.new_dataset::<u64>()
        .shape(())
        .create("heap_address")
        .unwrap()
        .write_scalar(&address)
        .unwrap();
    file.close().unwrap();
}

// H5HF_create_t and H5HF_dtable_cparam_t in H5HFprivate.h, HDF5 2.2.0, define these creation fields.
#[repr(C)]
struct HeapCreate {
    managed: DoublingTable,
    checksum_dblocks: bool,
    max_man_size: u32,
    id_len: u16,
    pipeline: Pipeline,
}

#[repr(C)]
struct DoublingTable {
    width: c_uint,
    start_block_size: usize,
    max_direct_size: usize,
    max_index: c_uint,
    start_root_rows: c_uint,
}

// H5O_pline_t and its H5O_shared_t prefix in H5Oprivate.h, HDF5 2.2.0, have this native layout.
#[derive(Default)]
#[repr(C)]
struct Pipeline {
    shared: SharedMessage,
    version: c_uint,
    allocated: usize,
    used: usize,
    filters: *mut c_void,
}

#[repr(C)]
struct SharedMessage {
    kind: c_uint,
    file: *mut c_void,
    message_type: c_uint,
    location: SharedLocation,
}

impl Default for SharedMessage {
    fn default() -> Self {
        Self {
            kind: 0,
            file: ptr::null_mut(),
            message_type: 0,
            location: SharedLocation {
                message: MessageLocation {
                    index: 0,
                    address: 0,
                },
            },
        }
    }
}

#[repr(C)]
union SharedLocation {
    message: MessageLocation,
    heap_id: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct MessageLocation {
    index: u32,
    address: u64,
}

unsafe extern "C" {
    fn H5VL_vol_object(id: i64) -> *mut c_void;
    fn H5VL_object_is_native(object: *const c_void, native: *mut bool) -> c_int;
    fn H5VL_object(id: i64) -> *mut c_void;
    fn H5AC_ignore_tags(file: *mut c_void) -> c_int;
    fn H5Z_append(
        pipeline: *mut Pipeline,
        filter: c_int,
        flags: c_uint,
        count: usize,
        values: *const c_uint,
    ) -> c_int;
    fn H5HF_create(file: *mut c_void, parameters: *const HeapCreate) -> *mut c_void;
    fn H5HF_get_id_len(heap: *mut c_void, length: *mut usize) -> c_int;
    fn H5HF_get_heap_addr(heap: *const c_void, address: *mut u64) -> c_int;
    fn H5HF_insert(heap: *mut c_void, length: usize, bytes: *const c_void, id: *mut u8) -> c_int;
    fn H5HF_read(heap: *mut c_void, id: *const u8, bytes: *mut c_void) -> c_int;
    fn H5HF_close(heap: *mut c_void) -> c_int;
    fn H5O_msg_reset(message_type: c_uint, message: *mut Pipeline) -> c_int;
}

/// The filter-pipeline message identifier, `H5O_PLINE_ID` in H5Oprivate.h, HDF5 2.2.0.
const FILTER_PIPELINE_MESSAGE_ID: c_uint = 0x000b;
/// The Deflate identifier from "The Data Storage - Filter Pipeline Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
const FILTER_DEFLATE: c_int = 1;
