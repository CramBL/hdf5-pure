#[cfg(any(feature = "std", test))]
pub use hdf5_pure_format::__private::HeaderMessage;
pub use hdf5_pure_format::__private::MessageFilter;
pub use hdf5_pure_format::__private::ObjectHeader;

#[cfg(all(test, feature = "std"))]
mod tests {
    use rstest::rstest;
    use test_util::image::Image;
    use test_util::object_header::Message;
    use test_util::object_header::MessageType as RecordType;
    use test_util::object_header::v1 as v1_bytes;
    use test_util::object_header::v2 as v2_bytes;
    use test_util::widths::Widths;

    use super::*;
    use crate::access_mode::AccessMode;
    use crate::address::BaseAddress;
    use crate::address::BaseAddressExt;
    use crate::error::FormatError;
    use crate::message_type::MessageType;
    use crate::source::BytesSource;
    use crate::source::ReadSeekSource;
    use crate::source::Source;
    use crate::source::SourceMetadata;

    #[derive(Clone, Copy, Debug)]
    enum Version {
        One,
        Two,
    }

    #[rstest]
    #[case::version_1(Version::One)]
    #[case::version_2(Version::Two)]
    fn a_header_read_through_a_source_matches_the_buffered_parse(#[case] version: Version) {
        let data = header_with_continuation(version);
        let buffered = ObjectHeader::parse(&data, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        let memory = parse_through(&BytesSource::new(&data)).unwrap();
        let seek = parse_through(&ReadSeekSource::new(std::io::Cursor::new(data.clone())).unwrap())
            .unwrap();

        let expected = vec![
            (MessageType::DATASPACE, DATASPACE_BODY.to_vec()),
            (MessageType::DATATYPE, DATATYPE_BODY.to_vec()),
        ];
        for (path, header) in [("buffered", buffered), ("memory", memory), ("seek", seek)] {
            let messages = header
                .messages
                .into_iter()
                .filter(|message| message.msg_type != MessageType::OBJECT_HEADER_CONTINUATION)
                .map(|message| (message.msg_type, message.data))
                .collect::<Vec<_>>();
            assert_eq!(messages, expected, "{version:?} {path}");
        }
    }

    #[rstest]
    #[case::message_overrunning_chunk_0(Truncation::MessageOverrunningChunk0, 24, 16)]
    #[case::partial_prefix_in_a_continuation(Truncation::PartialPrefixInContinuation, 8, 1)]
    fn a_truncated_header_read_through_a_source_returns_unexpected_eof(
        #[values(Backend::Memory, Backend::Seek)] backend: Backend,
        #[case] truncation: Truncation,
        #[case] expected: usize,
        #[case] available: usize,
    ) {
        let data = truncated_header(truncation);
        let result = match backend {
            Backend::Memory => parse_through(&BytesSource::new(&data)),
            Backend::Seek => {
                parse_through(&ReadSeekSource::new(std::io::Cursor::new(data)).unwrap())
            }
        };

        assert_eq!(
            result.unwrap_err(),
            FormatError::UnexpectedEof {
                expected,
                available
            }
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Backend {
        Memory,
        Seek,
    }

    #[derive(Clone, Copy, Debug)]
    enum Truncation {
        MessageOverrunningChunk0,
        PartialPrefixInContinuation,
    }

    fn parse_through(source: &impl Source) -> Result<ObjectHeader, FormatError> {
        ObjectHeader::parse_from_source(
            &SourceMetadata(source),
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::ZERO,
        )
    }

    fn truncated_header(truncation: Truncation) -> Vec<u8> {
        let mut continuation =
            v1_bytes::message_record(&Message::new(RecordType::DATATYPE, &DATATYPE_BODY));
        if matches!(truncation, Truncation::PartialPrefixInContinuation) {
            continuation.push(0);
        }
        let header =
            v1_bytes::Header::new().continuation(CONTINUATION_OFFSET, &continuation, Widths::EIGHT);
        let header = match truncation {
            Truncation::MessageOverrunningChunk0 => header.declared_data_size(16),
            Truncation::PartialPrefixInContinuation => header,
        };
        let mut image = Image::starting_with(&header.build());
        image.place(CONTINUATION_OFFSET, &continuation);
        image.build()
    }

    fn header_with_continuation(version: Version) -> Vec<u8> {
        let dataspace = Message::new(RecordType::DATASPACE, &DATASPACE_BODY);
        let datatype = Message::new(RecordType::DATATYPE, &DATATYPE_BODY);
        let (header, continuation) = match version {
            Version::One => {
                let continuation = v1_bytes::message_record(&datatype);
                let header = v1_bytes::Header::new()
                    .message(dataspace)
                    .continuation(CONTINUATION_OFFSET, &continuation, Widths::EIGHT)
                    .build();
                (header, continuation)
            }
            Version::Two => {
                let continuation =
                    v2_bytes::continuation_chunk(&[datatype], v2_bytes::HeaderFlags::default());
                let header = v2_bytes::Header::new()
                    .message(dataspace)
                    .continuation(CONTINUATION_OFFSET, &continuation, Widths::EIGHT)
                    .build();
                (header, continuation)
            }
        };
        let mut image = Image::starting_with(&header);
        image.place(CONTINUATION_OFFSET, &continuation);
        image.build()
    }

    const CONTINUATION_OFFSET: usize = 256;
    const DATASPACE_BODY: [u8; 8] = [42, 0, 0, 0, 0, 0, 0, 0];
    const DATATYPE_BODY: [u8; 8] = [0xBE, 0xEF, 0, 0, 0, 0, 0, 0];
}
