use core::num::NonZeroU32;
use std::ffi::CStr;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::super::tests as protection_tests;
use crate::UserblockSize;
use crate::object_graph::ObjectId;

pub(super) fn fixture(
    path: &Path,
    form: Form,
    base: u64,
    count: NonZeroU32,
    comment: &CStr,
) -> CountFixture {
    let userblock = UserblockSize::try_from(base).unwrap();
    let mut fixture = Fixture::new();
    let retired = protection_tests::append_retirements(&mut fixture, userblock);
    let chunk = form.chunk(count.get(), comment);
    let at = fixture.image.append(&chunk);
    let object = if matches!(form, Form::Continuation) {
        let header = v2::Header::new()
            .continuation(at, &chunk, Widths::EIGHT)
            .build();
        u64::try_from(fixture.image.append(&header)).unwrap()
    } else {
        u64::try_from(at).unwrap()
    };
    fixture.link("a", object);
    fixture.link("b", object);
    protection_tests::write_fixture(fixture, path, userblock);
    CountFixture {
        object: ObjectId::new(PhysicalAddress::new(object + base)),
        allocation: FileRange::new(
            PhysicalAddress::new(u64::try_from(at).unwrap() + base),
            u64::try_from(chunk.len()).unwrap(),
        )
        .unwrap(),
        retired,
    }
}

pub(super) struct CountFixture {
    pub(super) object: ObjectId,
    pub(super) allocation: FileRange,
    pub(super) retired: [(FileRange, FreeClass); 2],
}

#[derive(Clone, Copy)]
pub(super) enum Form {
    Continuation,
    Version1,
    Version2,
}

impl Form {
    pub(super) fn chunk(self, count: u32, comment: &CStr) -> Vec<u8> {
        let mut comment = Message::new(MessageType::OBJECT_COMMENT, comment.to_bytes_with_nul());
        match self {
            Self::Version1 => {
                comment.data.resize(
                    comment.data.len().next_multiple_of(v1::MESSAGE_ALIGNMENT),
                    0,
                );
                v1::Header::new()
                    .reference_count(count)
                    .message(comment)
                    .build()
            }
            Self::Version2 | Self::Continuation => {
                let messages = [Message::reference_count(count), comment];
                let flags = if u8::try_from(
                    v2::message_records(&messages, v2::HeaderFlags::default()).len(),
                )
                .is_ok()
                {
                    v2::HeaderFlags::default()
                } else {
                    v2::HeaderFlags::CHUNK_SIZE_WIDTH
                };
                if matches!(self, Self::Version2) {
                    v2::Header::new().flags(flags).messages(messages).build()
                } else {
                    v2::continuation_chunk(&messages, flags)
                }
            }
        }
    }
}
