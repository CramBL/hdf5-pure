use core::cell::RefCell;

use hdf5_pure_core::MetadataBytes;
use rstest::rstest;
use test_util::btree::v1 as btree_v1;
use test_util::widths::Widths;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::btree::v1::{BTreeV1ChunkNode, BTreeV1ChunkNodeBytes};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;

#[path = "fixture.rs"]
mod fixture;

/// Returns the size, the filter mask, and the offsets of each key of `node`, with the child
/// after it.
fn entries(node: BTreeV1ChunkNode<'_>) -> Vec<(u32, u32, Vec<u64>, StoredAddress)> {
    node.entries()
        .map(|entry| {
            let (key, child) = entry.unwrap();
            (
                key.chunk_size,
                key.filter_mask,
                key.offsets().collect(),
                child,
            )
        })
        .collect()
}

/// A source over `bytes` that records the offset and the length of each read.
struct RecordingSource<'a> {
    bytes: &'a [u8],
    reads: RefCell<Vec<(u64, usize)>>,
}

impl MetadataSource for RecordingSource<'_> {
    type Bytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
        self.reads.borrow_mut().push((offset, len));
        self.bytes.read_metadata(offset, len)
    }

    fn len(&self) -> u64 {
        MetadataSource::len(self.bytes)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.reads.borrow_mut().push((offset, buf.len()));
        self.bytes.read_at(offset, buf)
    }

    fn read_metadata_at(&self, _offset: u64, _len: usize) -> Result<Vec<u8>, FormatError> {
        Err(FormatError::Source("unexpected vector read".into()))
    }
}

#[rstest]
#[case::leaf(0)]
#[case::internal(1)]
fn chunk_readers_decode_leaf_and_internal_nodes(
    #[case] level: u8,
    #[values(Widths::new(2, 8), Widths::new(4, 2), Widths::new(8, 4))] widths: Widths,
) {
    let keys = [
        btree_v1::chunk_key(160, 0, &[0, 0]),
        btree_v1::chunk_key(96, 2, &[20, 0]),
        btree_v1::chunk_key(0, 0, &[40, 0]),
    ];
    let node = btree_v1::node(
        btree_v1::NodeType::CHUNK,
        level,
        &keys,
        &[0x1000, 0x2000],
        widths,
    );
    let offset_size = u8::try_from(widths.offset).unwrap();
    let expected = vec![
        (160, 0, vec![0, 0], StoredAddress::new(0x1000)),
        (96, 2, vec![20, 0], StoredAddress::new(0x2000)),
    ];

    let parsed = BTreeV1ChunkNode::parse(&node, 0, offset_size, 2).unwrap();
    assert_eq!(
        (
            parsed.node_level(),
            parsed.entries_used(),
            parsed.stored_len()
        ),
        (level, 2, node.len())
    );
    assert_eq!(entries(parsed), expected);
    assert_eq!(
        parsed.children().map(Result::unwrap).collect::<Vec<_>>(),
        vec![StoredAddress::new(0x1000), StoredAddress::new(0x2000)]
    );

    let source = RecordingSource {
        bytes: &node,
        reads: RefCell::new(Vec::new()),
    };
    let header_len = fixture::header_len(widths);
    let read = BTreeV1ChunkNodeBytes::read_from_source(&source, 0, offset_size, 2).unwrap();
    assert_eq!(read.body.as_ptr(), node[header_len..].as_ptr());
    let read = read.node();
    assert_eq!(read.entries.as_ptr(), node[header_len..].as_ptr());
    assert_eq!(
        (read.node_level(), read.entries_used(), read.stored_len()),
        (level, 2, node.len())
    );
    assert_eq!(entries(read), expected);
    assert_eq!(
        source.reads.into_inner(),
        vec![
            (0, header_len),
            (header_len.to_u64(), node.len() - header_len)
        ]
    );
}

#[rstest]
#[case::a_group_node(
    btree_v1::SIGNATURE.len(),
    btree_v1::NodeType::GROUP.0,
    FormatError::InvalidBTreeNodeType(btree_v1::NodeType::GROUP.0)
)]
#[case::a_corrupted_signature(0, b'X', FormatError::InvalidBTreeSignature)]
fn chunk_readers_reject_a_complete_invalid_header_without_a_body(
    #[case] field: usize,
    #[case] byte: u8,
    #[case] expected: FormatError,
    #[values(Widths::new(2, 8), Widths::new(4, 2), Widths::new(8, 4))] widths: Widths,
    #[values(0, 7)] offset: usize,
) {
    let header_len = fixture::header_len(widths);
    let offset_size = u8::try_from(widths.offset).unwrap();
    let mut data = vec![0; offset];
    let mut node = fixture::node(widths, 0);
    node[field] = byte;
    data.extend_from_slice(&node[..header_len]);
    let source = RecordingSource {
        bytes: &data,
        reads: RefCell::new(Vec::new()),
    };
    assert_eq!(
        BTreeV1ChunkNode::parse(&data, offset, offset_size, 2).unwrap_err(),
        expected
    );
    assert_eq!(
        BTreeV1ChunkNodeBytes::read_from_source(&source, offset.to_u64(), offset_size, 2)
            .unwrap_err(),
        expected
    );
    assert_eq!(source.reads.into_inner(), [(offset.to_u64(), header_len)]);
}

#[rstest]
#[case::bad_signature(0, b'X')]
#[case::wrong_node_type(btree_v1::SIGNATURE.len(), btree_v1::NodeType::GROUP.0)]
fn chunk_readers_acquire_the_header_before_reading_invalid_fields(
    #[case] field: usize,
    #[case] byte: u8,
    #[values(Widths::new(2, 8), Widths::new(4, 2), Widths::new(8, 4))] widths: Widths,
    #[values(0, 7)] offset: usize,
) {
    let header_len = fixture::header_len(widths);
    let mut node = fixture::node(widths, 0);
    node[field] = byte;
    for len in 0..header_len {
        let mut data = vec![0; offset];
        data.extend_from_slice(&node[..len]);
        assert_chunk_eof(&data, offset, widths, offset + header_len);
    }
}

#[rstest]
#[case::selected_entry(fixture::KEY_LEN + 1)]
#[case::trailing_key(1)]
fn chunk_readers_require_the_selected_entries_and_trailing_key(
    #[case] missing: usize,
    #[values(Widths::new(2, 8), Widths::new(4, 2), Widths::new(8, 4))] widths: Widths,
    #[values(0, 1)] level: u8,
) {
    let offset = 7;
    let mut bytes = vec![0; offset];
    let node = fixture::node(widths, level);
    bytes.extend_from_slice(&node[..node.len() - missing]);
    assert_chunk_eof(&bytes, offset, widths, offset + node.len());
}

fn assert_chunk_eof(data: &[u8], offset: usize, widths: Widths, expected: usize) {
    let offset_size = u8::try_from(widths.offset).unwrap();
    assert_eq!(
        BTreeV1ChunkNode::parse(data, offset, offset_size, 2).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available: data.len(),
        }
    );
    assert_eq!(
        BTreeV1ChunkNodeBytes::read_from_source(data, offset.to_u64(), offset_size, 2).unwrap_err(),
        FormatError::UnexpectedSourceEof {
            expected: expected.to_u64(),
            available: data.len().to_u64(),
        }
    );
}

struct PartialSource<'a> {
    bytes: &'a [u8],
    address: u64,
}

impl MetadataSource for PartialSource<'_> {
    type Bytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
        let start = usize::try_from(offset.checked_sub(self.address).unwrap()).unwrap();
        let rest = self.bytes.get(start..).unwrap_or_default();
        Ok(&rest[..rest.len().min(len)])
    }

    fn len(&self) -> u64 {
        self.bytes.len().to_u64()
    }

    fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FormatError> {
        Err(FormatError::Source("unexpected payload read".into()))
    }

    fn read_metadata_at(&self, _offset: u64, _len: usize) -> Result<Vec<u8>, FormatError> {
        Err(FormatError::Source("unexpected vector read".into()))
    }
}

#[rstest]
#[case::empty_header(0, 24)]
#[case::short_header(8, 24)]
#[case::missing_body(24, 80)]
#[case::short_entry(55, 80)]
#[case::missing_trailing_key(56, 80)]
#[case::short_trailing_key(79, 80)]
fn chunk_reader_reports_short_metadata_as_eof(#[case] available: usize, #[case] expected: usize) {
    let node = fixture::node(Widths::EIGHT, 0);
    assert_eq!(node.len(), 80);
    let source = PartialSource {
        bytes: &node[..available],
        address: 0,
    };
    assert_eq!(
        BTreeV1ChunkNodeBytes::read_from_source(&source, 0, 8, 2).unwrap_err(),
        FormatError::UnexpectedEof {
            expected,
            available
        }
    );
}

#[test]
fn chunk_reader_shares_the_source_body_allocation() {
    struct SharedSource {
        header: MetadataBytes<'static>,
        body: MetadataBytes<'static>,
    }

    impl MetadataSource for SharedSource {
        type Bytes<'a> = MetadataBytes<'a>;

        fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
            let bytes = match offset {
                0 => &self.header,
                offset if offset == self.body.offset() => &self.body,
                _ => return Err(FormatError::Source("unexpected metadata offset".into())),
            };
            if bytes.len() != len {
                return Err(FormatError::DataSizeMismatch {
                    expected: len,
                    actual: bytes.len(),
                });
            }
            Ok(bytes.clone())
        }

        fn len(&self) -> u64 {
            self.body.offset() + self.body.len().to_u64()
        }

        fn read_at(&self, _offset: u64, _buf: &mut [u8]) -> Result<(), FormatError> {
            Err(FormatError::Source("unexpected payload read".into()))
        }

        fn read_metadata_at(&self, _offset: u64, _len: usize) -> Result<Vec<u8>, FormatError> {
            Err(FormatError::Source("unexpected vector read".into()))
        }
    }

    let mut node = btree_v1::node(
        btree_v1::NodeType::CHUNK,
        0,
        &[
            btree_v1::chunk_key(160, 2, &[20, 0]),
            btree_v1::chunk_key(0, 0, &[40, 0]),
        ],
        &[0x1000],
        Widths::EIGHT,
    );
    let header_len = fixture::header_len(Widths::EIGHT);
    let body = node.split_off(header_len);
    let pointer = body.as_ptr();
    let source = SharedSource {
        header: MetadataBytes::from_vec(0, node).unwrap(),
        body: MetadataBytes::from_vec(header_len.to_u64(), body).unwrap(),
    };
    let first = BTreeV1ChunkNodeBytes::read_from_source(&source, 0, 8, 2).unwrap();
    let second = BTreeV1ChunkNodeBytes::read_from_source(&source, 0, 8, 2).unwrap();
    assert_eq!(first.body.as_ptr(), pointer);
    assert_eq!(second.body.as_ptr(), pointer);
    assert_eq!(first.node().stored_len(), 80);
    assert_eq!(
        entries(first.node()),
        vec![(160, 2, vec![20, 0], StoredAddress::new(0x1000))]
    );
    assert_eq!(entries(second.node()), entries(first.node()));
}

#[rstest]
#[case::body_address(u64::MAX, 2, u64::MAX, 24)]
#[case::key_size(0, usize::MAX, usize::MAX.to_u64(), 8)]
#[case::entry_size(0, usize::MAX / 8 - 1, (usize::MAX - 7).to_u64(), 8)]
fn chunk_reader_reports_address_and_size_overflow(
    #[case] address: u64,
    #[case] ndims: usize,
    #[case] offset: u64,
    #[case] length: u64,
) {
    let node = btree_v1::node(
        btree_v1::NodeType::CHUNK,
        0,
        &[btree_v1::chunk_key(0, 0, &[0, 0])],
        &[],
        Widths::EIGHT,
    );
    let source = PartialSource {
        bytes: &node,
        address,
    };
    assert_eq!(
        BTreeV1ChunkNodeBytes::read_from_source(&source, address, 8, ndims).unwrap_err(),
        FormatError::OffsetOverflow { offset, length }
    );
}
