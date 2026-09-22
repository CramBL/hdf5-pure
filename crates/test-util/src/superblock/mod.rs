//! The superblock: section `subsec_fmt4_boot_super`, version 4.0.

use std::path::Path;

use crate::bytes;

pub mod v0;
pub mod v2;

#[track_caller]
pub fn version(path: impl AsRef<Path>) -> u8 {
    let (file, at) = located(path.as_ref());
    version_at(&file, at)
}

#[track_caller]
pub fn consistency_flags(path: impl AsRef<Path>) -> u32 {
    let path = path.as_ref();
    let (file, at) = located(path);
    match version_at(&file, at) {
        // Signature(8) + version(1) + free-space version(1) + root symbol-table
        // entry version(1) + reserved(1) + shared-header-message version(1) +
        // size of offsets(1) + size of lengths(1) + reserved(1) + group leaf
        // node K(2) + group internal node K(2): section
        // `subsec_fmt4_boot_super`, version 4.0.
        0 | 1 => bytes::u32_at(
            &file,
            at + SIGNATURE.len() + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 2,
        ),
        // Version 2 narrowed the field to one byte and moved it up behind the
        // two size fields: signature(8) + version(1) + size of offsets(1) +
        // size of lengths(1), same section.
        2 | 3 => u32::from(bytes::u8_at(&file, at + SIGNATURE.len() + 1 + 1 + 1)),
        other => panic!("{path:?}: superblock version {other} has no consistency-flags field"),
    }
}

/// The version of the superblock beginning at `at` in `file`, for a caller
/// that has already located it.
#[track_caller]
pub fn version_at(file: &[u8], at: usize) -> u8 {
    // The version byte follows the signature: section `subsec_fmt4_boot_super`,
    // version 4.0.
    bytes::u8_at(file, at + SIGNATURE.len())
}

// The file's bytes and the offset its superblock begins at.
#[track_caller]
fn located(path: &Path) -> (Vec<u8>, usize) {
    let file = crate::read_file(path);
    // 0, 512, and every doubling of that: a file may carry a userblock of any
    // of those sizes ahead of its superblock, and nothing else may hold the
    // signature. Section `subsec_fmt4_boot_super`, version 4.0.
    let at = std::iter::once(0)
        .chain((0..).map(|doubling| 512usize << doubling))
        .take_while(|at| *at < file.len())
        .find(|at| {
            file.get(*at..)
                .is_some_and(|rest| rest.starts_with(SIGNATURE))
        })
        .unwrap_or_else(|| {
            panic!(
                "{path:?}: no HDF5 signature at offset 0, 512 or a doubling of it in {} bytes",
                file.len()
            )
        });
    (file, at)
}

/// The byte a superblock spends on the width of an offset or a length field.
#[track_caller]
pub(crate) fn width_byte(width: usize) -> u8 {
    u8::try_from(width).expect("a superblock field width is one to eight bytes")
}

pub const SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::superblock;
    use crate::temp::{self, TempPath};

    fn superblock_behind(userblock: usize, version: u8, flags: u8) -> TempPath {
        let mut file = vec![0xEE; userblock];
        file.extend_from_slice(superblock::SIGNATURE);
        file.extend_from_slice(&[version, 8, 8, flags]);
        written(&file)
    }

    fn written(file: &[u8]) -> TempPath {
        let path = temp::temp_path("superblock.h5");
        std::fs::write(&path, file).expect("write the fixture");
        path
    }

    #[rstest]
    #[case::no_userblock(0)]
    #[case::a_doubled_userblock(1024)]
    fn locates_the_superblock_behind_a_userblock(#[case] userblock: usize) {
        let path = superblock_behind(userblock, 3, 0x05);
        assert_eq!(superblock::version(&path), 3);
        assert_eq!(superblock::consistency_flags(&path), 0x05);
    }

    #[test]
    fn reads_the_four_byte_flags_of_a_version_1_superblock() {
        let mut file = superblock::SIGNATURE.to_vec();
        // version(1) + free space(1) + symbol table(1) + reserved(1) + shared
        // message(1) + offsets(1) + lengths(1) + reserved(1), then the two K
        // values, then the flags.
        file.extend_from_slice(&[1, 0, 0, 0, 0, 8, 8, 0]);
        file.extend_from_slice(&4u16.to_le_bytes());
        file.extend_from_slice(&16u16.to_le_bytes());
        file.extend_from_slice(&0x0000_0005u32.to_le_bytes());
        let path = written(&file);

        assert_eq!(superblock::version(&path), 1);
        assert_eq!(superblock::consistency_flags(&path), 0x05);
    }
}
