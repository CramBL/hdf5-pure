use std::cell::RefCell;

use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FormatError, StoredAddress};
use hdf5_pure_format::__private::{BTreeV2Header, FormatWidths, MetadataSource};
use rstest::rstest;
use test_util::checksum;

#[rstest]
#[case::valid(None)]
#[case::invalid_signature(Some((0, b'X')))]
#[case::invalid_version(Some((4, 1)))]
fn rejects_every_truncation_before_interpreting_the_header(
    #[case] changed_byte: Option<(usize, u8)>,
    #[values(0, PREFIX_LEN)] offset: usize,
    #[values(Parser::Buffered, Parser::Source)] parser: Parser,
) {
    let mut file = vec![0xA5; offset];
    let mut header = header_bytes();
    if let Some((index, value)) = changed_byte {
        header[index] = value;
    }
    file.extend_from_slice(&header);

    for available in 0..file.len() {
        let expected = offset + HEADER_LEN;
        let error = match parser {
            Parser::Buffered => FormatError::UnexpectedEof {
                expected,
                available,
            },
            Parser::Source => FormatError::UnexpectedSourceEof {
                expected: u64::try_from(expected).unwrap(),
                available: u64::try_from(available).unwrap(),
            },
        };
        let input = &file[..available];
        assert_eq!(parser.parse(input, offset).unwrap_err(), error);
        assert_eq!(parser.parse_with_widths(input, offset).unwrap_err(), error);
    }
}

#[rstest]
#[case::invalid_signature(0, b'X', FormatError::InvalidBTreeV2Signature)]
#[case::invalid_version(4, 1, FormatError::InvalidBTreeV2Version(1))]
fn rejects_invalid_signature_or_version_in_a_complete_header(
    #[case] index: usize,
    #[case] value: u8,
    #[case] error: FormatError,
    #[values(Parser::Buffered, Parser::Source)] parser: Parser,
) {
    let mut file = file_with_header();
    file[PREFIX_LEN + index] = value;
    assert_eq!(parser.parse(&file, PREFIX_LEN).unwrap_err(), error);
    assert_eq!(
        parser.parse_with_widths(&file, PREFIX_LEN).unwrap_err(),
        error
    );
}

#[rstest]
fn parses_retained_header_fields_with_unequal_widths(
    #[values(Parser::Buffered, Parser::Source)] parser: Parser,
) {
    let mut file = file_with_header();
    file.extend_from_slice(&[0xA5; 64]);
    assert_eq!(parser.parse(&file, PREFIX_LEN).unwrap(), expected_header());
    assert_eq!(
        parser.parse_with_widths(&file, PREFIX_LEN).unwrap(),
        expected_header()
    );
}

#[rstest]
fn compares_the_complete_checksum_only_when_enabled(
    #[values(Parser::Buffered, Parser::Source)] parser: Parser,
) {
    let mut file = file_with_header();
    let computed = checksum::lookup3(HEADER_BODY);
    let stored = !computed;
    file[PREFIX_LEN + HEADER_BODY.len()..].copy_from_slice(&stored.to_le_bytes());
    let result = if cfg!(feature = "checksum") {
        Err(FormatError::ChecksumMismatch {
            expected: stored,
            computed,
        })
    } else {
        Ok(expected_header())
    };
    assert_eq!(parser.parse(&file, PREFIX_LEN), result);
    assert_eq!(parser.parse_with_widths(&file, PREFIX_LEN), result);
}

#[test]
fn reads_only_the_header_from_the_source() {
    let mut file = file_with_header();
    file.extend_from_slice(&[0xA5; 64]);
    let source = RecordingSource::new(&file);
    let address = u64::try_from(PREFIX_LEN).unwrap();
    let header = BTreeV2Header::parse_from_source_with_widths(&source, address, widths()).unwrap();
    assert_eq!(header, expected_header());
    assert_eq!(*source.requests.borrow(), [(address, HEADER_LEN)]);
}

#[test]
fn rejects_an_overflowing_source_extent_before_reading() {
    let source = RecordingSource::new(&[]);
    let length = u64::try_from(HEADER_LEN).unwrap();
    let address = u64::MAX - length + 1;
    assert_eq!(
        BTreeV2Header::parse_from_source_with_widths(&source, address, widths()).unwrap_err(),
        FormatError::OffsetOverflow {
            offset: address,
            length,
        }
    );
    assert!(source.requests.borrow().is_empty());
}

#[rstest]
#[case::offset(3, LENGTH_SIZE, FormatError::InvalidOffsetSize(3))]
#[case::length(OFFSET_SIZE, 3, FormatError::InvalidLengthSize(3))]
fn rejects_invalid_widths_before_acquiring_the_header(
    #[case] offset_size: u8,
    #[case] length_size: u8,
    #[case] error: FormatError,
) {
    let source = RecordingSource::new(&[]);
    assert_eq!(
        BTreeV2Header::parse(&[], 0, offset_size, length_size).unwrap_err(),
        error
    );
    assert_eq!(
        BTreeV2Header::parse_from_source(&source, 0, offset_size, length_size).unwrap_err(),
        error
    );
    assert!(source.requests.borrow().is_empty());
}

#[derive(Clone, Copy)]
enum Parser {
    Buffered,
    Source,
}

impl Parser {
    fn parse(self, file: &[u8], offset: usize) -> Result<BTreeV2Header, FormatError> {
        match self {
            Self::Buffered => BTreeV2Header::parse(file, offset, OFFSET_SIZE, LENGTH_SIZE),
            Self::Source => BTreeV2Header::parse_from_source(
                file,
                u64::try_from(offset).unwrap(),
                OFFSET_SIZE,
                LENGTH_SIZE,
            ),
        }
    }

    fn parse_with_widths(self, file: &[u8], offset: usize) -> Result<BTreeV2Header, FormatError> {
        match self {
            Self::Buffered => BTreeV2Header::parse_with_widths(file, offset, widths()),
            Self::Source => BTreeV2Header::parse_from_source_with_widths(
                file,
                u64::try_from(offset).unwrap(),
                widths(),
            ),
        }
    }
}

struct RecordingSource<'a> {
    bytes: &'a [u8],
    requests: RefCell<Vec<(u64, usize)>>,
}

impl<'a> RecordingSource<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            requests: RefCell::new(Vec::new()),
        }
    }
}

impl MetadataSource for RecordingSource<'_> {
    type Bytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
        self.requests.borrow_mut().push((offset, len));
        self.bytes.read_metadata(offset, len)
    }

    fn len(&self) -> u64 {
        MetadataSource::len(self.bytes)
    }

    fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FormatError> {
        Err(FormatError::Source(
            "unexpected read into a caller buffer".into(),
        ))
    }

    fn read_metadata_at(&self, _offset: u64, _len: usize) -> Result<Vec<u8>, FormatError> {
        Err(FormatError::Source("unexpected vector read".into()))
    }
}

fn widths() -> FormatWidths {
    FormatWidths::from_sizes(OFFSET_SIZE, LENGTH_SIZE).unwrap()
}

fn file_with_header() -> Vec<u8> {
    let mut file = vec![0xA5; PREFIX_LEN];
    file.extend_from_slice(&header_bytes());
    file
}

fn header_bytes() -> Vec<u8> {
    let mut header = HEADER_BODY.to_vec();
    checksum::append(&mut header);
    assert_eq!(header.len(), HEADER_LEN);
    header
}

fn expected_header() -> BTreeV2Header {
    BTreeV2Header {
        tree_type: 5,
        node_size: 0x1123,
        record_size: 0x0115,
        depth: 10,
        root_node_address: StoredAddress::new(0x5423_6712),
        num_records_in_root: 7,
        total_records: 0x0000_0001_9876_5432,
    }
}

const OFFSET_SIZE: u8 = 4;
const LENGTH_SIZE: u8 = 8;
const PREFIX_LEN: usize = 19;
const HEADER_LEN: usize = 34;
const HEADER_BODY: &[u8] = b"BTHD\x00\x05\x23\x11\x00\x00\x15\x01\x0A\x00\x64\x28\
    \x12\x67\x23\x54\x07\x00\x32\x54\x76\x98\x01\x00\x00\x00";
