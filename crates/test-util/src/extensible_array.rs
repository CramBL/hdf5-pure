use std::path::Path;

use crate::bytes;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeaderStats {
    pub secondary_block_count: u64,
    pub secondary_block_size: u64,
    pub data_block_count: u64,
    pub data_block_size: u64,
    pub max_index_set: u64,
    pub element_count: u64,
}

#[track_caller]
pub fn header_stats(path: impl AsRef<Path>) -> HeaderStats {
    let file = crate::read_file(path.as_ref());
    let header = bytes::sole_signature(&file, SIGNATURE);
    // Signature(4) + version(1) + client ID(1) + element size(1) + maximum
    // elements bits(1) + index block elements(1) + data block minimum
    // elements(1) + secondary block minimum data pointers(1) + maximum data
    // block page elements bits(1), then the six statistics in the order the
    // header stores them: section `subsec_fmt4_appendixc_extarr`, version 4.0.
    let stat = |field: usize| {
        bytes::u64_at(
            &file,
            header + SIGNATURE.len() + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + field * LENGTH,
        )
    };
    HeaderStats {
        secondary_block_count: stat(0),
        secondary_block_size: stat(1),
        data_block_count: stat(2),
        data_block_size: stat(3),
        max_index_set: stat(4),
        element_count: stat(5),
    }
}

// Each statistic is a length field, eight bytes wide in every file this crate
// writes and in every file it is compared against.
const LENGTH: usize = 8;

// Section `subsec_fmt4_appendixc_extarr`, version 4.0.
const SIGNATURE: &[u8; 4] = b"EAHD";

#[cfg(test)]
mod tests {
    use crate::extensible_array::{self, HeaderStats};
    use crate::temp;

    #[test]
    fn reads_the_six_statistics_in_stored_order() {
        let mut file = vec![0u8; 8];
        file.extend_from_slice(extensible_array::SIGNATURE);
        file.extend_from_slice(&[0, 0, 4, 32, 4, 16, 4, 16]);
        for stat in 1..=6u64 {
            file.extend_from_slice(&stat.to_le_bytes());
        }
        let path = temp::temp_path("extensible_array.h5");
        std::fs::write(&path, &file).expect("write the fixture");

        assert_eq!(
            extensible_array::header_stats(&path),
            HeaderStats {
                secondary_block_count: 1,
                secondary_block_size: 2,
                data_block_count: 3,
                data_block_size: 4,
                max_index_set: 5,
                element_count: 6,
            }
        );
    }

    #[test]
    #[should_panic(expected = "expected one EAHD in the file, found 0")]
    fn refuses_a_file_with_no_extensible_array() {
        let path = temp::temp_path("extensible_array.h5");
        std::fs::write(&path, vec![0u8; 64]).expect("write the fixture");
        extensible_array::header_stats(&path);
    }
}
