pub use hdf5_pure_format::jenkins_lookup3;
pub use hdf5_pure_format::verify_trailing;

#[cfg(test)]
pub(crate) fn stamp_trailing(file: &mut [u8], at: usize, len: usize) {
    let checksum = jenkins_lookup3(&file[at..at + len - 4]);
    file[at + len - 4..at + len].copy_from_slice(&checksum.to_le_bytes());
}
