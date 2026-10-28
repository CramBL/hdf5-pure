//! Native observations of private chunked-writer output.

use test_util::temp;
use test_util_hdf5::file;
use test_util_hdf5::fixed_string;

use crate::file_writer::AttrValue;
use crate::file_writer::FileWriter;

#[test]
fn native_reader_observes_both_chunked_datasets() {
    let path = temp::temp_path("chunked.h5");
    let first: Vec<f64> = (0..50).map(f64::from).collect();
    let second: Vec<f64> = (0..30).map(|value| f64::from(value * 10)).collect();
    let mut writer = FileWriter::new();
    writer
        .create_dataset("a")
        .with_f64_data(&first)
        .with_shape(&[50])
        .with_chunks(&[25]);
    writer
        .create_dataset("b")
        .with_f64_data(&second)
        .with_shape(&[30])
        .with_chunks(&[10]);
    std::fs::write(&path, writer.finish().unwrap()).unwrap();

    let native = file::libhdf5_open(&path);
    let mut names = native.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["a", "b"]);
    let dataset = native.dataset("a").unwrap();
    assert_eq!(dataset.shape(), [50]);
    assert_eq!(dataset.chunk(), Some(vec![25]));
    assert_eq!(dataset.read_raw::<f64>().unwrap(), first);
    drop(dataset);
    let dataset = native.dataset("b").unwrap();
    assert_eq!(dataset.shape(), [30]);
    assert_eq!(dataset.chunk(), Some(vec![10]));
    assert_eq!(dataset.read_raw::<f64>().unwrap(), second);
    drop(dataset);
    native.close().unwrap();
}

#[test]
fn native_reader_observes_chunked_values_and_string_attribute() {
    let path = temp::temp_path("chunked-attribute.h5");
    let expected: Vec<f64> = (0..50).map(f64::from).collect();
    let mut writer = FileWriter::new();
    writer
        .create_dataset("data")
        .with_f64_data(&expected)
        .with_shape(&[50])
        .with_chunks(&[25])
        .set_attr("units", AttrValue::String("meters".to_owned()));
    std::fs::write(&path, writer.finish().unwrap()).unwrap();

    let native = file::libhdf5_open(&path);
    assert_eq!(native.member_names().unwrap(), ["data"]);
    let dataset = native.dataset("data").unwrap();
    assert_eq!(dataset.shape(), [50]);
    assert_eq!(dataset.chunk(), Some(vec![25]));
    assert_eq!(dataset.read_raw::<f64>().unwrap(), expected);
    assert_eq!(dataset.attr_names().unwrap(), ["units"]);
    assert_eq!(
        fixed_string::read_libhdf5_attribute::<6>(&dataset, "units"),
        "meters"
    );
    drop(dataset);
    native.close().unwrap();
}
