#![cfg(feature = "__hdf5-1.10")]
//! Native LZF decoding, pipeline metadata, and incompressible rewrites.

use std::fmt::Debug;
use std::path::Path;
use std::ptr;

use hdf5::H5Type;
use hdf5::filters::Filter as NativeFilter;
use hdf5::filters::FilterInfo;
use hdf5_pure::File;
use hdf5_pure::FilterDisposition;
use hdf5_pure::FilterId;
use hdf5_sys::h5p;
use hdf5_sys::h5z::H5Z_FLAG_OPTIONAL;
use test_util::temp;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::Filter;
use test_util_hdf5::dataset::Unlimited;
use test_util_hdf5::lzf;

#[test]
fn native_reader_observes_and_rewrites_fresh_pure_lzf_output() {
    assert_lzf_available();
    let path = temp::temp_path("pure-lzf.h5");
    let fixture = lzf::Fixture::new();
    std::fs::write(&path, fixture.pure_bytes()).unwrap();
    assert_native_values(&path, &fixture, fixture.noise());
    let rewritten = fixture.rewritten_noise();
    let native = hdf5::File::open_rw(&path).unwrap();
    let noise = native.dataset("incompressible_u8").unwrap();
    let chunks = noise.num_chunks().unwrap();
    assert_eq!(chunks, fixture.noise().len() / 1024);
    for index in 0..chunks {
        let info = noise.chunk_info(index).unwrap();
        assert_eq!(info.filter_mask, 0);
        assert!(info.size > 1024);
    }
    noise.write_raw(&rewritten).unwrap();
    noise.flush().unwrap();
    drop(noise);
    native.flush().unwrap();
    native.close().unwrap();

    assert_native_values(&path, &fixture, &rewritten);
    let bytes = std::fs::read(&path).unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let noise = native.dataset("incompressible_u8").unwrap();
    assert_eq!(noise.num_chunks(), Some(chunks));
    for index in 0..chunks {
        let info = noise.chunk_info(index).unwrap();
        assert_eq!(info.offset, [u64::try_from(index * 1024).unwrap()]);
        assert_eq!(info.filter_mask, 1);
        assert_eq!(info.size, 1024);
        let start = usize::try_from(info.addr).unwrap();
        assert_eq!(
            &bytes[start..start + 1024],
            &rewritten[index * 1024..(index + 1) * 1024]
        );
    }
    drop(noise);
    native.close().unwrap();

    let pure = File::open(&path).unwrap();
    let noise = pure.dataset("incompressible_u8").unwrap();
    assert_eq!(noise.read_u8().unwrap(), rewritten);
    let decoded_chunks = noise.chunks().unwrap();
    assert_eq!(decoded_chunks.len(), chunks);
    for chunk in decoded_chunks {
        let (filter, disposition) = chunk.filters().filter(0).unwrap();
        assert_eq!(filter.id(), FilterId::Lzf);
        assert!(filter.is_optional());
        assert_eq!(disposition, FilterDisposition::Skipped);
        assert!(!chunk.filters().bypasses_pipeline());
    }
    drop(noise);
    pure.close().unwrap();
}

#[test]
fn native_lzf_constructor_encodes_a_shuffled_partial_edge() {
    assert_lzf_available();
    let path = temp::temp_path("native-lzf.h5");
    let values: Vec<i32> = (0..257).collect();
    Unlimited::new("values", &values, 128)
        .filters(&[Filter::Shuffle, Filter::Lzf])
        .libhdf5_create(&path);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "values"), values);
    let pure = File::open(&path).unwrap();
    let dataset = pure.dataset("values").unwrap();
    assert_eq!(
        dataset.filters().unwrap(),
        [FilterId::Shuffle, FilterId::Lzf]
    );
    assert_eq!(dataset.read_i32().unwrap(), values);
    drop(dataset);
    pure.close().unwrap();
}

fn assert_lzf_available() {
    assert_eq!(
        NativeFilter::get_info(NativeFilter::LZF.id()),
        FilterInfo {
            is_available: true,
            encode_enabled: true,
            decode_enabled: true,
        }
    );
}

fn assert_native_values(path: &Path, fixture: &lzf::Fixture, noise: &[u8]) {
    let native = hdf5::File::open(path).unwrap();
    let mut names = native.member_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        [
            "incompressible_u8",
            "multichunk_i64",
            "plain_i32",
            "shuffle_f64"
        ]
    );
    assert_native_dataset(
        &native,
        "plain_i32",
        fixture.integers(),
        256,
        Pipeline::Plain,
    );
    assert_native_dataset(
        &native,
        "shuffle_f64",
        fixture.shuffled(),
        128,
        Pipeline::Shuffled,
    );
    assert_native_dataset(
        &native,
        "multichunk_i64",
        fixture.partial(),
        128,
        Pipeline::Plain,
    );
    assert_native_dataset(&native, "incompressible_u8", noise, 1024, Pipeline::Plain);
    native.close().unwrap();
}

fn assert_native_dataset<T: H5Type + PartialEq + Debug>(
    file: &hdf5::File,
    name: &str,
    expected: &[T],
    chunk: usize,
    pipeline: Pipeline,
) {
    let dataset = file.dataset(name).unwrap();
    assert_eq!(dataset.shape(), [expected.len()]);
    assert_eq!(dataset.chunk(), Some(vec![chunk]));
    assert_eq!(dataset.read_raw::<T>().unwrap(), expected);
    let property = dataset.dcpl().unwrap();
    assert!(property.all_filters_avail());
    let element_bytes = u32::try_from(size_of::<T>()).unwrap();
    let chunk_bytes = u32::try_from(chunk * size_of::<T>()).unwrap();
    let mut expected_filters = Vec::new();
    if matches!(pipeline, Pipeline::Shuffled) {
        expected_filters.push((NativeFilter::Shuffle, 0, vec![element_bytes]));
    }
    expected_filters.push((
        NativeFilter::LZF,
        H5Z_FLAG_OPTIONAL,
        vec![LZF_FILTER_VERSION, STORED_LIBLZF_VERSION, chunk_bytes],
    ));
    assert_eq!(
        property.get_filters().unwrap(),
        expected_filters
            .iter()
            .map(|(filter, _, _)| filter.clone())
            .collect::<Vec<_>>()
    );
    for (index, (filter, expected_flags, expected_parameters)) in
        expected_filters.into_iter().enumerate()
    {
        let mut flags = 0;
        let mut parameters = vec![0; expected_parameters.len()];
        let mut parameter_count = parameters.len();
        let index = u32::try_from(index).unwrap();
        let id = hdf5::sync::sync(|| {
            // SAFETY: The binding lock and live property list cover the call. The parameter
            // buffer holds `parameter_count` writable words. No name or configuration is requested.
            unsafe {
                h5p::H5Pget_filter2(
                    property.id(),
                    index,
                    &mut flags,
                    &mut parameter_count,
                    parameters.as_mut_ptr(),
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            }
        });
        assert_eq!(id, filter.id());
        assert_eq!(flags, expected_flags);
        assert_eq!(parameter_count, expected_parameters.len());
        assert_eq!(parameters, expected_parameters);
    }
}

#[derive(Clone, Copy)]
enum Pipeline {
    Plain,
    Shuffled,
}

// `set_local_lzf` in `hl/filters/lzf.rs`, binding revision 605a2e2, records filter version 4.
const LZF_FILTER_VERSION: u32 = 4;
// h5py 3.16.0 records stored liblzf version 1.5 in `tests/data/h5py/lzf/manifest.json`.
// `set_local_lzf` in `hl/filters/lzf.rs`, binding revision 605a2e2, preserves a nonzero stored version.
const STORED_LIBLZF_VERSION: u32 = 0x0105;
