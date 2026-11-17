use core::num::NonZeroUsize;

use rstest::rstest;

use super::ResidualGroupSource;

#[test]
fn residual_padding_rejects_a_truncated_header_before_writing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("short-padding.h5");
    let error = std::panic::catch_unwind(|| {
        ResidualGroupSource::Pad(NonZeroUsize::new(1).unwrap()).write(&path)
    })
    .unwrap_err();
    let Some(message) = error.downcast_ref::<&str>() else {
        panic!("expected a static padding error");
    };
    assert_eq!(*message, "residual padding cannot shorten the header");
    assert!(!path.exists());
}

#[rstest]
#[case::complete(NORMAL_HEADER_SIZE)]
#[case::zero_tail(NORMAL_HEADER_SIZE + 1)]
fn residual_padding_preserves_the_complete_source(#[case] length: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("complete-padding.h5");
    let (parent, residual, keeper) =
        ResidualGroupSource::Pad(NonZeroUsize::new(length).unwrap()).write(&path);
    assert_eq!(
        (parent, residual, keeper.start().get(), keeper.length()),
        (560, 4_800, 4_745, 55)
    );
    let image = std::fs::read(&path).unwrap();
    assert_eq!(image.len(), RESIDUAL_START + length);
    assert!(
        image[RESIDUAL_START + NORMAL_HEADER_SIZE..]
            .iter()
            .all(|&byte| byte == 0)
    );
    let file = hdf5_pure::File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["gone", "keep"]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
    file.close().unwrap();
}

const RESIDUAL_START: usize = 4_800;
const NORMAL_HEADER_SIZE: usize = 53;
