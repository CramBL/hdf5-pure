//! Checks dense heap fixture values through hdf5-pure and the C library.

use std::path::Path;

use hdf5_pure::File;

/// Checks dataset values through buffered, streaming, and C-library reads.
///
/// Each dataset `g/{name}` must contain the index of `name` in `names` as its only value.
///
/// # Panics
///
/// Panics if a file operation fails or a dataset differs from its expected value.
pub fn assert_links_resolve(path: &Path, names: &[String]) {
    let buffered = File::open(path).unwrap();
    let streaming = File::open_streaming(path).unwrap();
    let reference = hdf5::File::open(path).unwrap();
    for (i, name) in names.iter().enumerate() {
        let p = format!("g/{name}");
        assert_eq!(
            reference.dataset(&p).unwrap().read_raw::<i32>().unwrap(),
            vec![i32::try_from(i).unwrap()]
        );
        assert_eq!(
            buffered.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "buffered link {i}"
        );
        assert_eq!(
            streaming.dataset(&p).unwrap().read_i32().unwrap(),
            vec![i as i32],
            "streaming link {i}"
        );
    }
}

/// Checks an edited fixture through the C library, then checks a C-written dataset.
///
/// The fixture must contain `probe` with 32 values equal to 19. Creates `c_probe` with the value 23.
///
/// # Panics
///
/// Panics if a file operation fails or either dataset differs from its expected values.
pub fn assert_edit_round_trip(path: &Path) {
    let file = hdf5::File::open_rw(path).unwrap();
    assert_eq!(
        file.dataset("probe").unwrap().read_raw::<i32>().unwrap(),
        [19; 32]
    );
    file.new_dataset::<i32>()
        .shape([1])
        .create("c_probe")
        .unwrap()
        .write_raw(&[23])
        .unwrap();
    file.close().unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(file.dataset("c_probe").unwrap().read_i32().unwrap(), [23]);
}
