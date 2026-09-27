pub use hdf5_pure_core::Superblock;

#[cfg(all(test, feature = "std"))]
mod tests {
    use test_util::superblock::v2;
    use test_util::widths::Widths;

    use super::*;
    use crate::error::FormatError;
    use crate::source::{BytesSource, ReadSeekSource, Source, SourceMetadata};

    #[test]
    fn a_superblock_read_through_a_source_matches_the_buffered_parse() {
        let mut data = vec![0u8; 4096];
        let superblock = superblock_bytes();
        data[512..512 + superblock.len()].copy_from_slice(&superblock);

        let buffered = hdf5_pure_format::parse_superblock(&data, 512).unwrap();
        let memory = parse_through(&BytesSource::new(&data), 512).unwrap();
        let seek = parse_through(
            &ReadSeekSource::new(std::io::Cursor::new(data)).unwrap(),
            512,
        )
        .unwrap();

        assert_eq!(memory, buffered);
        assert_eq!(seek, buffered);
        assert_eq!(seek.root_group_address, 48);
    }

    #[test]
    fn a_superblock_read_through_a_source_has_its_checksum_validated() {
        let mut data = superblock_bytes();
        *data.last_mut().unwrap() ^= 0xFF;
        let source = ReadSeekSource::new(std::io::Cursor::new(data)).unwrap();

        let err = parse_through(&source, 0).unwrap_err();
        assert!(
            matches!(err, FormatError::ChecksumMismatch { .. }),
            "{err:?}"
        );
    }

    fn parse_through(
        source: &impl Source,
        signature_offset: u64,
    ) -> Result<Superblock, FormatError> {
        hdf5_pure_format::parse_superblock_from_source(&SourceMetadata(source), signature_offset)
    }

    fn superblock_bytes() -> Vec<u8> {
        v2::Superblock::new(Widths::EIGHT)
            .version(2)
            .eof_address(2048)
            .root_header_address(48)
            .build()
    }
}
