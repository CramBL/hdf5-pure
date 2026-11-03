use alloc::vec::Vec;
use core::cell::RefCell;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::image::Image;
use test_util::object_header::v1;
use test_util::object_header::{Message, MessageFlags, MessageType as WireType};
use test_util::widths::Widths;

use super::Driver;
use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;
use crate::object_header::{MessageFilter, ObjectHeader};
use crate::width::FormatWidths;

#[rstest]
fn exponential_version_1_fanout_exhausts_the_shared_allowance(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
    #[values(false, true)] filtered: bool,
) {
    let data = fanout();
    assert_eq!(data.len(), 656);
    let mut drop_all = |_: MessageType, _: &[u8]| false;
    let filter = if filtered {
        MessageFilter::Only(&mut drop_all)
    } else {
        MessageFilter::All
    };
    assert_eq!(
        driver.parse(&data, filter).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
}

#[rstest]
#[case::distinct_siblings(Shape::Siblings)]
#[case::repeated_siblings(Shape::Repeated)]
#[case::nested_siblings(Shape::Nested)]
#[case::empty_targets(Shape::Empty)]
fn version_1_accepts_256_visits_and_preserves_every_record(
    #[case] shape: Shape,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
    #[values(false, true)] filtered: bool,
) {
    let (data, expected) = shape.image(256);
    let mut drop_all = |_: MessageType, _: &[u8]| false;
    let filter = if filtered {
        MessageFilter::Only(&mut drop_all)
    } else {
        MessageFilter::All
    };
    let header = driver.parse(&data, filter).unwrap();
    if filtered {
        assert!(header.messages.is_empty());
        assert_eq!(header.messages.capacity(), 0);
    } else {
        assert_records(&header, &expected);
    }
}

#[rstest]
#[case::distinct_siblings(Shape::Siblings)]
#[case::repeated_siblings(Shape::Repeated)]
#[case::nested_siblings(Shape::Nested)]
#[case::empty_targets(Shape::Empty)]
fn version_1_rejects_visit_257_across_the_whole_header(
    #[case] shape: Shape,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
    #[values(false, true)] filtered: bool,
) {
    let (data, _) = shape.image(257);
    let mut drop_all = |_: MessageType, _: &[u8]| false;
    let filter = if filtered {
        MessageFilter::Only(&mut drop_all)
    } else {
        MessageFilter::All
    };
    assert_eq!(
        driver.parse(&data, filter).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
}

#[rstest]
#[case::at_limit(32)]
#[case::past_limit(33)]
fn version_1_depth_counts_only_continuation_hops(
    #[case] hops: usize,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let mut image = Image::new();
    let leaf = Message::new(WireType::DATATYPE, &[0xA5; 8]);
    image.place(128 + hops * 32, &v1::message_record(&leaf));
    let mut expected = Vec::new();
    for hop in 0..hops {
        let length = if hop + 1 == hops { 16 } else { 24 };
        let continuation =
            Message::continuation((128 + (hop + 1) * 32).to_u64(), length, Widths::EIGHT);
        if hop == 0 {
            image.place(0, &v1::Header::new().message(continuation.clone()).build());
        } else {
            image.place(128 + hop * 32, &v1::message_record(&continuation));
        }
        expected.push(continuation);
    }
    expected.push(leaf);
    let parsed = driver.parse(image.as_bytes(), MessageFilter::All);
    if hops == 32 {
        assert_records(&parsed.unwrap(), &expected);
    } else {
        assert_eq!(parsed.unwrap_err(), FormatError::NestingDepthExceeded);
    }
}

#[rstest]
#[case::fanout(true)]
#[case::repeats(false)]
fn a_version_1_source_never_fetches_or_observes_visit_257(#[case] exponential: bool) {
    let data = if exponential {
        fanout()
    } else {
        Shape::Repeated.image(257).0
    };
    let source = CountedSource::new(&data);
    let mut observed = Vec::new();
    let mut drop_all = |_: MessageType, _: &[u8]| false;
    let error = ObjectHeader::parse_from_source_with_chunk_observer(
        &source,
        AccessMode::ReadOnly,
        PhysicalAddress::new(0),
        FormatWidths::from_sizes(8, 8).unwrap(),
        BaseAddress::ZERO,
        MessageFilter::Only(&mut drop_all),
        &mut |chunk| {
            assert_eq!(chunk.v2_layout(), None);
            observed.push((chunk.allocation(), chunk.records().range()));
        },
    )
    .unwrap_err();
    assert_eq!(error, FormatError::NestingDepthExceeded);
    let reads = source.reads.borrow();
    assert_eq!(reads.len(), 3 + 256);
    assert_eq!(reads[0], (0, 4));
    assert_eq!(reads[1], (0, 12));
    assert_eq!(reads[2].0, 16);
    assert_eq!(observed.len(), 1 + 256);
    assert_eq!(observed[0].0.start().get(), 0);
    assert_eq!(observed[0].0.length(), 16 + reads[2].1.to_u64());
    for ((allocation, records), &(address, length)) in observed.iter().skip(1).zip(&reads[3..]) {
        assert_eq!(allocation, records);
        assert_eq!(allocation.start().get(), address);
        assert_eq!(allocation.length(), length.to_u64());
    }
    if !exponential {
        assert!(reads[3..].iter().all(|read| *read == (TARGET.to_u64(), 16)));
    }
}

#[rstest]
#[case::repeated(Shape::Repeated)]
#[case::empty(Shape::Empty)]
fn successful_observed_version_1_parses_charge_each_follow_and_reset_per_parse(
    #[case] shape: Shape,
) {
    let (data, _) = shape.image(256);
    let source = CountedSource::new(&data);
    let mut visits = 0;
    let mut drop_all = |_: MessageType, _: &[u8]| false;
    let parse = |visits: &mut usize| {
        ObjectHeader::parse_from_source_with_chunk_observer(
            &source,
            AccessMode::ReadOnly,
            PhysicalAddress::new(0),
            FormatWidths::from_sizes(8, 8).unwrap(),
            BaseAddress::ZERO,
            MessageFilter::Only(&mut drop_all),
            &mut |_| *visits += 1,
        )
    };
    let mut parse = parse;
    let first = parse(&mut visits).unwrap();
    let second = parse(&mut visits).unwrap();
    assert!(first.messages.is_empty());
    assert!(second.messages.is_empty());
    assert_eq!(first.messages.capacity(), 0);
    assert_eq!(second.messages.capacity(), 0);
    assert_eq!(visits, 2 * 257);
    assert_eq!(source.reads.borrow().len(), 2 * 259);
}

#[rstest]
fn the_version_1_budget_precedes_target_resolution_and_access(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let (mut data, _) = Shape::Empty.image(256);
    let mut header = v1::Header::new()
        .messages((0..256).map(|_| Message::continuation(TARGET.to_u64(), 0, Widths::EIGHT)));
    header = header.message(Message::continuation(u64::MAX - 1, 8, Widths::EIGHT));
    let root = header.build();
    data[..root.len()].copy_from_slice(&root);
    assert_eq!(
        driver.parse(&data, MessageFilter::All).unwrap_err(),
        FormatError::NestingDepthExceeded
    );
}

#[rstest]
#[case::two(Widths::new(2, 2))]
#[case::four(Widths::FOUR)]
#[case::eight(Widths::EIGHT)]
#[case::mixed(Widths::new(2, 4))]
fn version_1_continuations_preserve_base_widths_and_depth_first_order(
    #[case] widths: Widths,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let base = 512usize;
    let a = Message::new(WireType::DATATYPE, &[1; 8]).with_flags(MessageFlags::CONSTANT);
    let b = Message::new(WireType::DATATYPE, &[2; 8]);
    let c = Message::new(WireType::DATATYPE, &[3; 8]);
    let leaf = v1::message_record(&b);
    let nested = aligned_continuation(256, leaf.len().to_u64(), widths);
    let chunk = v1::message_records(&[a.clone(), nested.clone(), c.clone()]);
    let first = aligned_continuation(128, chunk.len().to_u64(), widths);
    let sibling = aligned_continuation(384, leaf.len().to_u64(), widths);
    let root = v1::Header::new()
        .messages([first.clone(), sibling.clone()])
        .build();
    let mut image = Image::new();
    image.place(base, &root);
    image.place(base + 128, &chunk);
    image.place(base + 256, &leaf);
    image.place(base + 384, &leaf);
    let sizes = FormatWidths::from_sizes(
        u8::try_from(widths.offset).unwrap(),
        u8::try_from(widths.length).unwrap(),
    )
    .unwrap();
    let header = match driver {
        Driver::Buffered => ObjectHeader::parse_filtered_with_widths(
            image.as_bytes(),
            AccessMode::ReadOnly,
            base,
            sizes,
            BaseAddress::new(base.to_u64()),
            MessageFilter::All,
        ),
        Driver::Source => ObjectHeader::parse_from_source_filtered_with_widths(
            image.as_bytes(),
            AccessMode::ReadOnly,
            base.to_u64(),
            sizes,
            BaseAddress::new(base.to_u64()),
            MessageFilter::All,
        ),
    }
    .unwrap();
    assert_records(&header, &[first, a, nested, b.clone(), c, sibling, b]);
}

#[rstest]
#[case::two(Widths::new(2, 2))]
#[case::four(Widths::FOUR)]
#[case::eight(Widths::EIGHT)]
fn version_1_rejects_width_specific_undefined_continuations(
    #[case] widths: Widths,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let mut message = Message::undefined_continuation(0, widths);
    message
        .data
        .resize(message.data.len().next_multiple_of(8), 0);
    let data = v1::Header::new().message(message).build();
    let sizes = FormatWidths::from_sizes(
        u8::try_from(widths.offset).unwrap(),
        u8::try_from(widths.length).unwrap(),
    )
    .unwrap();
    let error = match driver {
        Driver::Buffered => ObjectHeader::parse_with_widths(&data, AccessMode::ReadOnly, 0, sizes),
        Driver::Source => ObjectHeader::parse_from_source_filtered_with_widths(
            data.as_slice(),
            AccessMode::ReadOnly,
            0,
            sizes,
            BaseAddress::ZERO,
            MessageFilter::All,
        ),
    }
    .unwrap_err();
    assert_eq!(error, FormatError::UndefinedContinuationAddress);
}

#[rstest]
#[case::empty(v1::Header::new().build(), 0)]
#[case::single(v1::Header::new().message(Message::new(WireType::DATATYPE, &[7; 8])).build(), 1)]
fn version_1_without_continuations_retains_its_messages(
    #[case] data: Vec<u8>,
    #[case] count: usize,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let header = driver.parse(&data, MessageFilter::All).unwrap();
    assert_eq!(header.version, 1);
    assert_eq!(header.reference_count, Some(1));
    assert_eq!(header.messages.len(), count);
    if count == 1 {
        assert_records(&header, &[Message::new(WireType::DATATYPE, &[7; 8])]);
    }
}

#[rstest]
#[case::address_overflow(u64::MAX - 1, 0, 4, FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 4 })]
#[case::range_overflow(u64::MAX - 1, 8, 0, FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 8 })]
fn version_1_checks_target_geometry_before_reading(
    #[case] address: u64,
    #[case] length: u64,
    #[case] base: u64,
    #[case] expected: FormatError,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let data = v1::Header::new()
        .message(Message::continuation(address, length, Widths::EIGHT))
        .build();
    let error = match driver {
        Driver::Buffered => ObjectHeader::parse_with_base(
            &data,
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::new(base),
        ),
        Driver::Source => ObjectHeader::parse_from_source(
            data.as_slice(),
            AccessMode::ReadOnly,
            0,
            8,
            8,
            BaseAddress::new(base),
        ),
    }
    .unwrap_err();
    assert_eq!(error, expected);
}

#[rstest]
fn version_1_continuation_truncation_preserves_the_backend_error(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let data = v1::Header::new()
        .message(Message::continuation(128, 16, Widths::EIGHT))
        .build();
    let expected = match driver {
        Driver::Buffered => FormatError::UnexpectedEof {
            expected: 144,
            available: data.len(),
        },
        Driver::Source => FormatError::UnexpectedSourceEof {
            expected: 144,
            available: data.len().to_u64(),
        },
    };
    assert_eq!(
        driver.parse(&data, MessageFilter::All).unwrap_err(),
        expected
    );
}

#[rstest]
#[case::empty(&[])]
#[case::short(&[0; 8])]
fn short_version_1_continuation_bodies_remain_unfollowed(
    #[case] body: &[u8],
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let message = Message::new(WireType::OBJECT_HEADER_CONTINUATION, body);
    let data = v1::Header::new().message(message.clone()).build();
    assert_records(
        &driver.parse(&data, MessageFilter::All).unwrap(),
        &[message],
    );
}

#[rstest]
#[case::optional(MessageFlags::NONE, AccessMode::ReadWrite, false)]
#[case::read_only(
    MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
    AccessMode::ReadOnly,
    false
)]
#[case::read_write(
    MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
    AccessMode::ReadWrite,
    true
)]
#[case::mandatory_read(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS, AccessMode::ReadOnly, true)]
#[case::mandatory_write(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS, AccessMode::ReadWrite, true)]
fn filtered_version_1_nested_records_preserve_access_mode(
    #[case] flags: MessageFlags,
    #[case] mode: AccessMode,
    #[case] rejected: bool,
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let child = v1::message_record(&Message::new(WireType::UNKNOWN, &[0; 8]).with_flags(flags));
    let mut image = Image::starting_with(
        &v1::Header::new()
            .continuation(128, &child, Widths::EIGHT)
            .build(),
    );
    image.place(128, &child);
    let mut filtered_unknown = false;
    let mut filter = |kind: MessageType, _: &[u8]| {
        filtered_unknown |= kind.unknown_id().is_some();
        false
    };
    let result = match driver {
        Driver::Buffered => ObjectHeader::parse_filtered(
            image.as_bytes(),
            mode,
            0,
            8,
            8,
            BaseAddress::ZERO,
            MessageFilter::Only(&mut filter),
        ),
        Driver::Source => ObjectHeader::parse_from_source_filtered(
            image.as_bytes(),
            mode,
            0,
            8,
            8,
            BaseAddress::ZERO,
            MessageFilter::Only(&mut filter),
        ),
    };
    if rejected {
        assert_eq!(
            result.unwrap_err(),
            FormatError::UnsupportedMessage(WireType::UNKNOWN.0)
        );
        assert!(!filtered_unknown);
    } else {
        let header = result.unwrap();
        assert!(header.messages.is_empty());
        assert_eq!(header.messages.capacity(), 0);
        assert!(filtered_unknown);
    }
}

#[rstest]
fn a_version_1_continuation_length_is_narrowed_only_for_materialization(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let length = 1u64 << 32;
    let data = v1::Header::new()
        .message(Message::continuation(128, length, Widths::EIGHT))
        .build();
    let expected = if usize::BITS == 32 {
        FormatError::ValueTooLargeForPlatform {
            value: length,
            target: "usize",
        }
    } else {
        match driver {
            Driver::Buffered => FormatError::UnexpectedEof {
                expected: usize::try_from(128 + length).unwrap(),
                available: data.len(),
            },
            Driver::Source => FormatError::UnexpectedSourceEof {
                expected: 128 + length,
                available: data.len().to_u64(),
            },
        }
    };
    assert_eq!(
        driver.parse(&data, MessageFilter::All).unwrap_err(),
        expected
    );
}

#[rstest]
fn version_1_continuations_traverse_their_complete_record_area(
    #[values(Driver::Buffered, Driver::Source)] driver: Driver,
) {
    let payload = Message::new(WireType::DATATYPE, &[0xAD; 8]);
    let mut records = vec![Message::nil(0); usize::from(u16::MAX) + 1];
    records.push(payload.clone());
    let chunk = v1::message_records(&records);
    let continuation = Message::continuation(128, chunk.len().to_u64(), Widths::EIGHT);
    let mut image = Image::starting_with(&v1::Header::new().message(continuation.clone()).build());
    image.place(128, &chunk);
    assert_records(
        &driver.parse(image.as_bytes(), MessageFilter::All).unwrap(),
        &[continuation, payload],
    );
}

fn assert_records(header: &ObjectHeader, expected: &[Message]) {
    assert_eq!(header.messages.len(), expected.len());
    for (actual, expected) in header.messages.iter().zip(expected) {
        assert_eq!(actual.msg_type, MessageType::from_u16(expected.msg_type.0));
        assert_eq!(actual.data, expected.data);
        assert_eq!(actual.size, expected.data.len());
        assert_eq!(actual.flags.get(), expected.flags.0);
        assert_eq!(actual.creation_order, None);
    }
}

fn aligned_continuation(address: u64, length: u64, widths: Widths) -> Message {
    let mut message = Message::continuation(address, length, widths);
    message
        .data
        .resize(message.data.len().next_multiple_of(8), 0);
    message
}

fn fanout() -> Vec<u8> {
    let mut image = Image::starting_with(&vec![0; 656]);
    for level in 0..10 {
        let next = 128 + (level + 1) * 48;
        let length = if level == 9 { 0 } else { 48 };
        let reference = Message::continuation(next.to_u64(), length, Widths::EIGHT);
        image.place(
            128 + level * 48,
            &v1::message_records(&[reference.clone(), reference]),
        );
    }
    image.place(
        0,
        &v1::Header::new()
            .message(Message::continuation(128, 48, Widths::EIGHT))
            .build(),
    );
    image.build()
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Empty,
    Nested,
    Repeated,
    Siblings,
}

impl Shape {
    fn image(self, visits: usize) -> (Vec<u8>, Vec<Message>) {
        let mut image = Image::new();
        let mut root = v1::Header::new();
        let mut expected = Vec::new();
        let mut remaining = visits;
        let mut index = 0usize;
        while remaining > 0 {
            let leaf = Message::new(WireType::DATATYPE, &[u8::try_from(index % 256).unwrap(); 8]);
            let target = match self {
                Shape::Siblings => TARGET + index * 32,
                Shape::Repeated | Shape::Nested | Shape::Empty => TARGET,
            };
            let (chunk, retained) = match self {
                Shape::Empty => (Vec::new(), Vec::new()),
                Shape::Nested if remaining >= 2 => {
                    let child = Message::new(WireType::DATATYPE, &[0xCC; 8]);
                    image.place(TARGET + 128, &v1::message_record(&child));
                    let nested = Message::continuation((TARGET + 128).to_u64(), 16, Widths::EIGHT);
                    let leaf = Message::new(WireType::DATATYPE, &[0xAA; 8]);
                    remaining -= 1;
                    (
                        v1::message_records(&[leaf.clone(), nested.clone()]),
                        vec![leaf, nested, child],
                    )
                }
                Shape::Repeated | Shape::Nested => {
                    let leaf = Message::new(WireType::DATATYPE, &[0xAA; 8]);
                    (v1::message_record(&leaf), vec![leaf])
                }
                Shape::Siblings => (v1::message_record(&leaf), vec![leaf]),
            };
            let target = if matches!(self, Shape::Nested) && remaining == 1 {
                TARGET + 256
            } else {
                target
            };
            image.place(target, &chunk);
            let continuation =
                Message::continuation(target.to_u64(), chunk.len().to_u64(), Widths::EIGHT);
            root = root.message(continuation.clone());
            expected.push(continuation);
            expected.extend(retained);
            remaining -= 1;
            index += 1;
        }
        image.place(0, &root.build());
        (image.build(), expected)
    }
}

struct CountedSource<'a> {
    data: &'a [u8],
    reads: RefCell<Vec<(u64, usize)>>,
}

impl<'a> CountedSource<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            reads: RefCell::new(Vec::new()),
        }
    }
}

impl MetadataSource for CountedSource<'_> {
    type Bytes<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn read_metadata(&self, offset: u64, len: usize) -> Result<Self::Bytes<'_>, FormatError> {
        self.reads.borrow_mut().push((offset, len));
        self.data.read_metadata(offset, len)
    }

    fn len(&self) -> u64 {
        self.data.len().to_u64()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        buf.copy_from_slice(self.read_metadata(offset, buf.len())?);
        Ok(())
    }

    fn read_metadata_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.read_metadata(offset, len)?.to_vec())
    }
}

const TARGET: usize = 16_384;
