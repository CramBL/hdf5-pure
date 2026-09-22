use std::path::Path;

use crate::bytes;

pub struct Userblock(Vec<u8>);

impl Userblock {
    #[track_caller]
    pub fn read(path: impl AsRef<Path>, size: usize) -> Self {
        let file = crate::read_file(path.as_ref());
        Self(bytes::slice_at(&file, 0, size).to_vec())
    }

    #[track_caller]
    pub fn stamp(file: &mut [u8], size: usize, marker: &[u8]) -> Self {
        assert!(
            marker.len() < size,
            "a {}-byte marker fills a {size}-byte userblock, leaving no far end to compare",
            marker.len()
        );
        bytes::set_slice_at(file, 0, marker);
        bytes::set_u8_at(file, size - 1, FAR_END);
        Self(bytes::slice_at(file, 0, size).to_vec())
    }

    #[track_caller]
    pub fn assert_unchanged(&self, path: impl AsRef<Path>) {
        let file = crate::read_file(path.as_ref());
        assert_eq!(
            bytes::slice_at(&file, 0, self.0.len()),
            self.0,
            "userblock bytes changed across the edit"
        );
    }

    pub fn bytes(&self) -> &[u8] {
        &self.0
    }
}

// Stamped over the userblock's last byte, so a comparison covers its whole
// span and not only the marker at its start.
const FAR_END: u8 = 0xAB;

#[cfg(test)]
mod tests {
    use crate::temp;
    use crate::userblock::Userblock;

    #[test]
    fn stamps_the_marker_and_the_far_end() {
        let mut file = vec![0u8; 16];
        let userblock = Userblock::stamp(&mut file, 8, b"MARK");
        assert_eq!(file, *b"MARK\0\0\0\xAB\0\0\0\0\0\0\0\0");
        assert_eq!(userblock.bytes(), b"MARK\0\0\0\xAB");

        let path = temp::temp_path("userblock.h5");
        std::fs::write(&path, &file).expect("write the fixture");
        userblock.assert_unchanged(&path);
        assert_eq!(Userblock::read(&path, 8).bytes(), b"MARK\0\0\0\xAB");
    }

    #[test]
    #[should_panic(expected = "userblock bytes changed across the edit")]
    fn reports_a_userblock_the_file_no_longer_holds() {
        let mut file = vec![0u8; 16];
        let userblock = Userblock::stamp(&mut file, 8, b"MARK");

        let path = temp::temp_path("userblock.h5");
        std::fs::write(&path, vec![0u8; 16]).expect("write the fixture");
        userblock.assert_unchanged(&path);
    }
}
